/*
 * Copyright (c) 2026 CasparCG Contributors
 *
 * This file is part of CasparCG (www.casparcg.com).
 *
 * CasparCG is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * CasparCG is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with CasparCG. If not, see <http://www.gnu.org/licenses/>.
 *
 * Pure-Vulkan GPU readback strategy for DeckLink output.
 *
 * Instead of using CUDA to read the VK mixer's render attachment and pack
 * v210, this strategy stays entirely in the Vulkan ecosystem:
 *   1. Imports the VK texture via VK_KHR_external_memory_win32
 *   2. GPU-waits on the mixer's timeline semaphore (VK_KHR_external_semaphore_win32)
 *   3. Runs a GLSL compute shader to pack v210 (or extract BGRA)
 *   4. Copies packed data to host-visible staging buffer
 *   5. Triple-buffered with VkFence for async pipeline
 *
 * Benefits over cuda_vk_strategy:
 *   - No CUDA runtime → no GPU context switching overhead
 *   - VK compute runs on the same scheduler as VK render → better SM utilization
 *   - Eliminates CUDA↔VK semaphore/memory interop overhead
 */

#include "../StdAfx.h"

#include "vk_readback_strategy.h"

#include <common/log.h>
#include <common/timer.h>

#ifdef ENABLE_VULKAN
#ifdef _WIN32
#include <windows.h>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_win32.h>
#else
#include <vulkan/vulkan.h>
#include <unistd.h>
#endif

#include <accelerator/vulkan/util/texture_wrapper.h>
#include <accelerator/vulkan/util/platform_config.h>
#include <common/array.h>
#include <common/bit_depth.h>
#include <core/frame/frame.h>
#include <core/frame/pixel_format.h>

#include "vk_readback_v210_spv.h"
#include "vk_readback_bgra_spv.h"
#endif

#include <atomic>
#include <deque>
#include <mutex>
#include <vector>
#include <cstring>
#include <stdexcept>

namespace caspar { namespace decklink {

#ifdef ENABLE_VULKAN

namespace {

#define VK_CHECK(call, msg)                                                                                            \
    do {                                                                                                               \
        VkResult res_ = (call);                                                                                        \
        if (res_ != VK_SUCCESS) {                                                                                      \
            CASPAR_LOG(error) << L"[vk_readback] " << msg << L" failed: " << res_;                                     \
            throw std::runtime_error(std::string(msg) + " failed: " + std::to_string(res_));                           \
        }                                                                                                              \
    } while (0)

uint32_t find_memory_type(VkPhysicalDevice phys, uint32_t filter, VkMemoryPropertyFlags props)
{
    VkPhysicalDeviceMemoryProperties mem;
    vkGetPhysicalDeviceMemoryProperties(phys, &mem);
    for (uint32_t i = 0; i < mem.memoryTypeCount; ++i)
        if ((filter & (1u << i)) && (mem.memoryTypes[i].propertyFlags & props) == props)
            return i;
    // Fallback: if HOST_CACHED was requested but not available, try without it
    if (props & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) {
        VkMemoryPropertyFlags fallback = (props & ~VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
        for (uint32_t i = 0; i < mem.memoryTypeCount; ++i)
            if ((filter & (1u << i)) && (mem.memoryTypes[i].propertyFlags & fallback) == fallback)
                return i;
    }
    throw std::runtime_error("Failed to find suitable Vulkan memory type");
}

} // anonymous namespace

// ===========================================================================
// vk_readback_strategy::impl
// ===========================================================================

struct vk_readback_strategy::impl
{
    const bool is_hdr_;
    const bool use_bt2020_;
    const bool dma_only_;
    const bool needs_v210_;
    // How many frames the DeckLink driver can hold scheduled at once. A staging
    // slot handed to it must not be rewritten for at least that long.
    const int  buffer_depth_;
    spl::shared_ptr<format_strategy> fallback_;

    // Vulkan device (consumer-side, same physical GPU as mixer)
    VkInstance       instance_      = VK_NULL_HANDLE;
    VkPhysicalDevice phys_device_   = VK_NULL_HANDLE;
    VkDevice         device_        = VK_NULL_HANDLE;
    uint32_t         compute_qf_    = UINT32_MAX;   // compute queue family index
    VkQueue          compute_queue_ = VK_NULL_HANDLE;

    // Compute pipeline (v210 or BGRA)
    VkShaderModule        shader_module_   = VK_NULL_HANDLE;
    VkDescriptorSetLayout desc_layout_     = VK_NULL_HANDLE;
    VkPipelineLayout      pipe_layout_     = VK_NULL_HANDLE;
    VkPipeline            pipeline_        = VK_NULL_HANDLE;

    // Sampler for texelFetch (nearest, clamp-to-edge)
    VkSampler sampler_ = VK_NULL_HANDLE;

    // Command pool & buffers
    VkCommandPool cmd_pool_ = VK_NULL_HANDLE;

    // Pipeline depth: a slot's fence is not waited on until PIPELINE_DEPTH frames
    // later, so the readback has that long to land. Distinct from how long a slot
    // must stay valid, which is buffer_depth_ (see the slot pool below).
    static constexpr int PIPELINE_DEPTH = 2;

    struct frame_slot
    {
        VkCommandBuffer cmd          = VK_NULL_HANDLE;
        VkFence         fence        = VK_NULL_HANDLE;

        // Device-local SSBO for compute output
        VkBuffer        dev_buf      = VK_NULL_HANDLE;
        VkDeviceMemory  dev_mem      = VK_NULL_HANDLE;

        // Host-visible staging buffer (mapped)
        VkBuffer        stage_buf    = VK_NULL_HANDLE;
        VkDeviceMemory  stage_mem    = VK_NULL_HANDLE;
        void*           mapped       = nullptr;

        // Descriptor set for this slot
        VkDescriptorSet desc_set     = VK_NULL_HANDLE;
    };
    // Slots are pooled, NOT rotated.
    //
    // This used to be a fixed array of 3 rotated by write_idx_, with the mapped
    // staging pointer handed to DeckLink through a no-op deleter. DeckLink keeps
    // buffer_depth() frames scheduled and DMAs each at display time, so a 3-slot
    // rotation overwrote staging memory the card still owned. Sized
    // buffer_depth_ + PIPELINE_DEPTH + 1 and reclaimed by the shared_ptr deleter,
    // a slot cannot be reused until the driver has released the frame using it.
    std::vector<frame_slot> slots_;
    VkDescriptorPool        desc_pool_ = VK_NULL_HANDLE;

    // free_ is touched from the DeckLink completion thread (via the deleter) as
    // well as from whichever thread calls convert_frame_for_port, so it needs its
    // own lock. Nothing in the deleter touches Vulkan -- it only returns an index.
    std::mutex       free_mtx_;
    std::vector<int> free_;

    // Slots whose readback has been submitted, oldest first.
    std::deque<int> inflight_;

    size_t buf_size_ = 0;   // current allocation size

    // Imported texture cache (like CUDA strategy's cached_slots_)
    struct imported_texture
    {
        void*          handle   = nullptr;
        VkDeviceMemory mem      = VK_NULL_HANDLE;
        VkImage        image    = VK_NULL_HANDLE;
        VkImageView    view     = VK_NULL_HANDLE;
        int            w        = 0;
        int            h        = 0;
    };
    static constexpr int MAX_TEX_CACHE = 8;
    imported_texture tex_cache_[MAX_TEX_CACHE] = {};
    int              num_cached_tex_ = 0;

    // Imported semaphore cache
    struct imported_sem
    {
        void*       handle = nullptr;
        VkSemaphore sem    = VK_NULL_HANDLE;
    };
    static constexpr int MAX_SEM_CACHE = 8;
    imported_sem sem_cache_[MAX_SEM_CACHE] = {};
    int          num_cached_sem_ = 0;

    // Format tracking
    VkFormat current_format_ = VK_FORMAT_UNDEFINED;

    // Shared free-list state, so a slot handed to DeckLink can be returned even if
    // the strategy is destroyed first (the deleter only pushes an index into this
    // block; the Vulkan objects are torn down by ~impl once the device is idle).
    struct free_list
    {
        std::mutex       mtx;
        std::vector<int> idx;
    };
    std::shared_ptr<free_list> free_list_ = std::make_shared<free_list>();

    // Take a free slot, or -1 when every slot is still held by DeckLink.
    int acquire_slot()
    {
        std::lock_guard<std::mutex> lk(free_list_->mtx);
        if (free_list_->idx.empty())
            return -1;
        const int i = free_list_->idx.back();
        free_list_->idx.pop_back();
        return i;
    }

    // Hand the slot's mapped staging pointer out; the deleter reclaims the slot
    // when DeckLink releases the frame that used it.
    std::shared_ptr<void> make_staging_ref(int slot_index)
    {
        auto fl = free_list_;
        return std::shared_ptr<void>(slots_[slot_index].mapped, [fl, slot_index](void*) {
            std::lock_guard<std::mutex> lk(fl->mtx);
            fl->idx.push_back(slot_index);
        });
    }

    // A blank output frame, for the frames where the pipeline has nothing to hand
    // back yet, or where every slot is still held by DeckLink.
    //
    // It must NOT be produced by falling through to the CPU strategy. This consumer
    // reports needs_cpu_frame_data()==false whenever a GPU readback mode is active
    // (decklink_consumer.cpp), so the mixer deliberately skips the host readback and
    // const_frame::image_data() is `unavailable` -- the CPU v210 packer then reads an
    // empty array and takes the process down. Zero-filled matches what the
    // consumer's own preroll schedules before playback starts.
    int row_bytes_for(int width) const
    {
        return (is_hdr_ || use_bt2020_ || needs_v210_) ? ((width + 47) / 48) * 128 : width * 4;
    }

    std::shared_ptr<void> blank_output(const core::video_format_desc& decklink_format_desc)
    {
        const size_t bytes =
            static_cast<size_t>(row_bytes_for(decklink_format_desc.width)) * decklink_format_desc.height;
        auto b = acquire_pinned_output(blank_pool_, bytes, 128); // same helper the CPU strategies use
        if (b)
            std::memset(b.get(), 0, bytes);
        return b;
    }
    std::shared_ptr<gpu_output_buffer_pool> blank_pool_;

    // Queue the slot whose readback was just submitted and hand back the oldest
    // completed one, preserving the previous 2-frame pipeline depth.
    //
    // Returns -1 while the queue is still filling: for the first PIPELINE_DEPTH
    // frames no submitted readback is old enough to have landed. The caller must
    // emit a blank frame for those -- see blank_output(), and do NOT route them to
    // the CPU strategy. A slot leaves this queue exactly once -- handing one out
    // while it is still queued is the defect being fixed, so the filling case must
    // not hand out the slot it just pushed.
    int push_and_take(int submitted)
    {
        inflight_.push_back(submitted);
        if ((int)inflight_.size() <= PIPELINE_DEPTH)
            return -1;

        const int oldest = inflight_.front();
        inflight_.pop_front();

        auto& s = slots_[oldest];
        VK_CHECK(vkWaitForFences(device_, 1, &s.fence, VK_TRUE, UINT64_MAX), "vkWaitForFences");

        // HOST_CACHED staging needs an explicit invalidate before the CPU reads it.
        VkMappedMemoryRange range{};
        range.sType  = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        range.memory = s.stage_mem;
        range.offset = 0;
        range.size   = VK_WHOLE_SIZE;
        vkInvalidateMappedMemoryRanges(device_, 1, &range);
        return oldest;
    }

    // Wait for all in-flight slot fences (targeted alternative to vkDeviceWaitIdle).
    // Fences start signaled, so this is safe to call even before first submit.
    void wait_all_slot_fences()
    {
        std::vector<VkFence> fences;
        fences.reserve(slots_.size());
        for (auto& s : slots_)
            if (s.fence) fences.push_back(s.fence);
        if (!fences.empty())
            vkWaitForFences(device_, static_cast<uint32_t>(fences.size()), fences.data(), VK_TRUE,
                            500'000'000); // 500ms timeout
    }

    // Timing diagnostics
    int    frame_count_      = 0;
    double accum_import_ms_  = 0.0;
    double accum_wait_ms_    = 0.0;
    double accum_submit_ms_  = 0.0;
    double accum_total_ms_   = 0.0;

    bool initialized_ = false;

    // ─── Construction / Destruction ────────────────────────────────────────

    impl(bool is_hdr, bool use_bt2020, spl::shared_ptr<format_strategy> fallback, bool dma_only, bool needs_v210,
         int buffer_depth)
        : is_hdr_(is_hdr)
        , use_bt2020_(use_bt2020)
        , dma_only_(dma_only)
        , needs_v210_(needs_v210)
        , buffer_depth_(buffer_depth > 0 ? buffer_depth : 4)
        , fallback_(std::move(fallback))
    {
    }

    ~impl()
    {
        if (device_ != VK_NULL_HANDLE) {
            vkDeviceWaitIdle(device_);

            // Destroy imported semaphores
            for (int i = 0; i < num_cached_sem_; ++i)
                if (sem_cache_[i].sem) vkDestroySemaphore(device_, sem_cache_[i].sem, nullptr);

            // Destroy imported textures
            for (int i = 0; i < num_cached_tex_; ++i) {
                auto& t = tex_cache_[i];
                if (t.view)  vkDestroyImageView(device_, t.view, nullptr);
                if (t.image) vkDestroyImage(device_, t.image, nullptr);
                if (t.mem)   vkFreeMemory(device_, t.mem, nullptr);
            }

            // Destroy frame slots
            for (auto& s : slots_) {
                if (s.fence)     vkDestroyFence(device_, s.fence, nullptr);
                if (s.dev_buf)   vkDestroyBuffer(device_, s.dev_buf, nullptr);
                if (s.dev_mem)   vkFreeMemory(device_, s.dev_mem, nullptr);
                if (s.stage_buf) vkDestroyBuffer(device_, s.stage_buf, nullptr);
                if (s.stage_mem) {
                    if (s.mapped) vkUnmapMemory(device_, s.stage_mem);
                    vkFreeMemory(device_, s.stage_mem, nullptr);
                }
            }

            if (desc_pool_)     vkDestroyDescriptorPool(device_, desc_pool_, nullptr);
            if (pipeline_)      vkDestroyPipeline(device_, pipeline_, nullptr);
            if (pipe_layout_)   vkDestroyPipelineLayout(device_, pipe_layout_, nullptr);
            if (desc_layout_)   vkDestroyDescriptorSetLayout(device_, desc_layout_, nullptr);
            if (shader_module_) vkDestroyShaderModule(device_, shader_module_, nullptr);
            if (sampler_)       vkDestroySampler(device_, sampler_, nullptr);
            if (cmd_pool_)      vkDestroyCommandPool(device_, cmd_pool_, nullptr);

            vkDestroyDevice(device_, nullptr);
        }
        if (instance_) vkDestroyInstance(instance_, nullptr);
    }

    // ─── Lazy initialization (on first frame, when we know the GPU LUID) ──

    void ensure_initialized(const uint8_t* target_luid)
    {
        if (initialized_) return;

        create_instance();
        select_physical_device(target_luid);
        create_device();

        if (!dma_only_) {
            create_sampler();
            create_pipeline();
            create_descriptor_pool();
        }
        create_command_pool();
        create_frame_slots();

        initialized_ = true;
        CASPAR_LOG(info) << L"[vk_readback] Initialized pure-Vulkan readback strategy ("
                         << (dma_only_ ? L"DMA-only " : L"")
                         << (is_hdr_ ? L"HDR " : L"SDR ")
                         << (use_bt2020_ ? L"BT.2020" : L"BT.709") << L")";
    }

    void create_instance()
    {
        VkApplicationInfo app{};
        app.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app.pApplicationName   = "CasparCG-DeckLink-Readback";
        app.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        app.apiVersion         = VK_API_VERSION_1_2;

        const char* exts[] = {
            VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
            VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME,
            VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME,
        };

        VkInstanceCreateInfo ci{};
        ci.sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        ci.pApplicationInfo        = &app;
        ci.enabledExtensionCount   = 3;
        ci.ppEnabledExtensionNames = exts;

        VK_CHECK(vkCreateInstance(&ci, nullptr, &instance_), "vkCreateInstance");
    }

    void select_physical_device(const uint8_t* target_luid)
    {
        uint32_t count = 0;
        vkEnumeratePhysicalDevices(instance_, &count, nullptr);
        std::vector<VkPhysicalDevice> devs(count);
        vkEnumeratePhysicalDevices(instance_, &count, devs.data());

        for (auto& pd : devs) {
            VkPhysicalDeviceIDProperties id_props{};
            id_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;

            VkPhysicalDeviceProperties2 props2{};
            props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
            props2.pNext = &id_props;

            vkGetPhysicalDeviceProperties2(pd, &props2);

            if (id_props.deviceLUIDValid &&
                std::memcmp(id_props.deviceLUID, target_luid, VK_LUID_SIZE) == 0) {
                phys_device_ = pd;
                CASPAR_LOG(debug) << L"[vk_readback] Matched physical device: "
                                  << props2.properties.deviceName;
                return;
            }
        }

        // Fallback: use first discrete GPU
        for (auto& pd : devs) {
            VkPhysicalDeviceProperties props;
            vkGetPhysicalDeviceProperties(pd, &props);
            if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
                phys_device_ = pd;
                CASPAR_LOG(warning) << L"[vk_readback] LUID match failed, using first discrete GPU: "
                                    << props.deviceName;
                return;
            }
        }

        if (!devs.empty()) {
            phys_device_ = devs[0];
            CASPAR_LOG(warning) << L"[vk_readback] No discrete GPU found, using first device";
        } else {
            throw std::runtime_error("[vk_readback] No Vulkan physical devices found");
        }
    }

    void create_device()
    {
        // Find the best queue family for our workload
        uint32_t qf_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(phys_device_, &qf_count, nullptr);
        std::vector<VkQueueFamilyProperties> qf_props(qf_count);
        vkGetPhysicalDeviceQueueFamilyProperties(phys_device_, &qf_count, qf_props.data());

        if (dma_only_) {
            // DMA mode: prefer a transfer-only queue (maps to the Copy/DMA engine,
            // no SM contention with CUDA or graphics workloads).
            for (uint32_t i = 0; i < qf_count; ++i) {
                if ((qf_props[i].queueFlags & VK_QUEUE_TRANSFER_BIT) &&
                    !(qf_props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
                    !(qf_props[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
                    compute_qf_ = i;
                    CASPAR_LOG(info) << L"[vk_readback] DMA mode: using transfer-only queue family=" << i;
                    break;
                }
            }
            // Fallback: any queue with transfer capability
            if (compute_qf_ == UINT32_MAX) {
                for (uint32_t i = 0; i < qf_count; ++i) {
                    if (qf_props[i].queueFlags & VK_QUEUE_TRANSFER_BIT) {
                        compute_qf_ = i;
                        CASPAR_LOG(warning) << L"[vk_readback] DMA mode: no transfer-only queue, using family=" << i;
                        break;
                    }
                }
            }
        } else {
            // Compute mode: prefer a compute-only queue family (avoids contention with graphics)
            for (uint32_t i = 0; i < qf_count; ++i) {
                if ((qf_props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
                    !(qf_props[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
                    compute_qf_ = i;
                    break;
                }
            }
            // Fallback to any compute-capable queue
            if (compute_qf_ == UINT32_MAX) {
                for (uint32_t i = 0; i < qf_count; ++i) {
                    if (qf_props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                        compute_qf_ = i;
                        break;
                    }
                }
            }
        }
        if (compute_qf_ == UINT32_MAX)
            throw std::runtime_error("[vk_readback] No suitable queue family found");

        float priority = 1.0f;
        VkDeviceQueueCreateInfo queue_ci{};
        queue_ci.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queue_ci.queueFamilyIndex = compute_qf_;
        queue_ci.queueCount       = 1;
        queue_ci.pQueuePriorities = &priority;

        using namespace caspar::accelerator::vulkan;
        const char* dev_exts[] = {
            VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
            platform::kExtMemExtName,
            VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
            platform::kExtSemExtName,
            VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME,
        };

        VkPhysicalDeviceTimelineSemaphoreFeatures timeline_feat{};
        timeline_feat.sType              = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES;
        timeline_feat.timelineSemaphore  = VK_TRUE;

        VkPhysicalDeviceFeatures2 features2{};
        features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        features2.pNext = &timeline_feat;

        VkDeviceCreateInfo dev_ci{};
        dev_ci.sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        dev_ci.pNext                   = &features2;
        dev_ci.queueCreateInfoCount    = 1;
        dev_ci.pQueueCreateInfos       = &queue_ci;
        dev_ci.enabledExtensionCount   = 5;
        dev_ci.ppEnabledExtensionNames = dev_exts;

        VK_CHECK(vkCreateDevice(phys_device_, &dev_ci, nullptr, &device_), "vkCreateDevice");
        vkGetDeviceQueue(device_, compute_qf_, 0, &compute_queue_);

        CASPAR_LOG(debug) << L"[vk_readback] Created VkDevice, compute queue family=" << compute_qf_;
    }

    void create_sampler()
    {
        VkSamplerCreateInfo si{};
        si.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        si.magFilter    = VK_FILTER_NEAREST;
        si.minFilter    = VK_FILTER_NEAREST;
        si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        VK_CHECK(vkCreateSampler(device_, &si, nullptr, &sampler_), "vkCreateSampler");
    }

    void create_pipeline()
    {
        // Select shader based on output format.
        // V210 is needed for HDR, BT.2020, or when pixel-format=yuv (needs_v210).
        const uint32_t* spv_data;
        size_t          spv_size;
        if (is_hdr_ || use_bt2020_ || needs_v210_) {
            // v210 path
            spv_data = vk_readback_v210_comp_spv;
            spv_size = vk_readback_v210_comp_spv_size;
        } else {
            // SDR BGRA path — only when pixel-format=rgba (no V210 needed)
            spv_data = vk_readback_bgra_comp_spv;
            spv_size = vk_readback_bgra_comp_spv_size;
        }

        VkShaderModuleCreateInfo sm_ci{};
        sm_ci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        sm_ci.codeSize = spv_size;
        sm_ci.pCode    = spv_data;
        VK_CHECK(vkCreateShaderModule(device_, &sm_ci, nullptr, &shader_module_), "vkCreateShaderModule");

        // Descriptor set layout: binding 0 = combined image sampler, binding 1 = SSBO
        VkDescriptorSetLayoutBinding bindings[2]{};
        bindings[0].binding         = 0;
        bindings[0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

        bindings[1].binding         = 1;
        bindings[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[1].descriptorCount = 1;
        bindings[1].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutCreateInfo dsl_ci{};
        dsl_ci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dsl_ci.bindingCount = 2;
        dsl_ci.pBindings    = bindings;
        VK_CHECK(vkCreateDescriptorSetLayout(device_, &dsl_ci, nullptr, &desc_layout_),
                 "vkCreateDescriptorSetLayout");

        // Push constant range
        VkPushConstantRange push{};
        push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push.offset     = 0;
        // v210: 11 ints (44B), BGRA: 8 ints (32B). Both grew by the four destination-placement
        // fields. THE RANGE HAS TO GROW WITH THEM -- pushing 44 bytes into a 28-byte declared
        // range is a VUID violation, and the kind that works on one driver and corrupts on the
        // next rather than failing cleanly.
        push.size       = (is_hdr_ || use_bt2020_ || needs_v210_) ? 44 : 32;

        VkPipelineLayoutCreateInfo pl_ci{};
        pl_ci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pl_ci.setLayoutCount         = 1;
        pl_ci.pSetLayouts            = &desc_layout_;
        pl_ci.pushConstantRangeCount = 1;
        pl_ci.pPushConstantRanges    = &push;
        VK_CHECK(vkCreatePipelineLayout(device_, &pl_ci, nullptr, &pipe_layout_),
                 "vkCreatePipelineLayout");

        VkComputePipelineCreateInfo cp_ci{};
        cp_ci.sType              = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cp_ci.stage.sType        = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cp_ci.stage.stage        = VK_SHADER_STAGE_COMPUTE_BIT;
        cp_ci.stage.module       = shader_module_;
        cp_ci.stage.pName        = "main";
        cp_ci.layout             = pipe_layout_;
        VK_CHECK(vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &cp_ci, nullptr, &pipeline_),
                 "vkCreateComputePipelines");
    }

    void create_descriptor_pool()
    {
        VkDescriptorPoolSize sizes[2]{};
        sizes[0].type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        sizes[0].descriptorCount = static_cast<uint32_t>(slot_count());
        sizes[1].type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        sizes[1].descriptorCount = static_cast<uint32_t>(slot_count());

        VkDescriptorPoolCreateInfo dp_ci{};
        dp_ci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dp_ci.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        dp_ci.maxSets       = static_cast<uint32_t>(slot_count());
        dp_ci.poolSizeCount = 2;
        dp_ci.pPoolSizes    = sizes;
        VK_CHECK(vkCreateDescriptorPool(device_, &dp_ci, nullptr, &desc_pool_),
                 "vkCreateDescriptorPool");
    }

    void create_command_pool()
    {
        VkCommandPoolCreateInfo cp_ci{};
        cp_ci.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cp_ci.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        cp_ci.queueFamilyIndex = compute_qf_;
        VK_CHECK(vkCreateCommandPool(device_, &cp_ci, nullptr, &cmd_pool_), "vkCreateCommandPool");
    }

    // Enough slots for everything that can hold one at once: buffer_depth_ frames
    // queued in the driver, PIPELINE_DEPTH readbacks in flight here, and one being
    // filled.
    int slot_count() const { return buffer_depth_ + PIPELINE_DEPTH + 1; }

    void create_frame_slots()
    {
        slots_.assign(slot_count(), frame_slot{});
        {
            std::lock_guard<std::mutex> lk(free_list_->mtx);
            free_list_->idx.clear();
            for (int i = static_cast<int>(slots_.size()) - 1; i >= 0; --i)
                free_list_->idx.push_back(i);
        }

        VkCommandBufferAllocateInfo cb_ai{};
        cb_ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cb_ai.commandPool        = cmd_pool_;
        cb_ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cb_ai.commandBufferCount = 1;

        for (auto& s : slots_) {
            VK_CHECK(vkAllocateCommandBuffers(device_, &cb_ai, &s.cmd), "vkAllocateCommandBuffers");

            VkFenceCreateInfo fence_ci{};
            fence_ci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            fence_ci.flags = VK_FENCE_CREATE_SIGNALED_BIT; // start signaled
            VK_CHECK(vkCreateFence(device_, &fence_ci, nullptr, &s.fence), "vkCreateFence");
        }
    }

    // ─── Buffer allocation ─────────────────────────────────────────────────

    void ensure_buffers(int dst_w, int dst_h)
    {
        size_t needed;
        if (dma_only_) {
            // Raw RGBA pixels: 4 bytes/pixel (8-bit) or 8 bytes/pixel (16-bit)
            size_t bpp = (is_hdr_ || current_format_ == VK_FORMAT_R16G16B16A16_UNORM) ? 8 : 4;
            needed = (size_t)dst_w * dst_h * bpp;
        } else if (is_hdr_ || use_bt2020_ || needs_v210_) {
            size_t v210_row = (size_t)((dst_w + 47) / 48) * 128;
            needed = v210_row * dst_h;
        } else {
            needed = (size_t)dst_w * dst_h * 4;
        }

        if (buf_size_ >= needed && !slots_.empty() && slots_[0].stage_buf != VK_NULL_HANDLE)
            return;

        // Wait for all in-flight work before reallocating
        wait_all_slot_fences();

        // The staging memory about to be unmapped and freed may still be referenced
        // by frames sitting in the DeckLink queue -- there is no way to recall those,
        // so this remains a hazard on a live format change (documented, not fixed:
        // it needs the driver's frames drained, which the consumer does not expose).
        // At least stop handing the slots out: clear the pipeline and rebuild the
        // free list so no stale index survives the reallocation.
        inflight_.clear();
        {
            std::lock_guard<std::mutex> lk(free_list_->mtx);
            free_list_->idx.clear();
            for (int i = static_cast<int>(slots_.size()) - 1; i >= 0; --i)
                free_list_->idx.push_back(i);
        }

        for (auto& s : slots_) {
            if (s.dev_buf)   { vkDestroyBuffer(device_, s.dev_buf, nullptr); s.dev_buf = VK_NULL_HANDLE; }
            if (s.dev_mem)   { vkFreeMemory(device_, s.dev_mem, nullptr); s.dev_mem = VK_NULL_HANDLE; }
            if (s.stage_buf) { vkDestroyBuffer(device_, s.stage_buf, nullptr); s.stage_buf = VK_NULL_HANDLE; }
            if (s.stage_mem) {
                if (s.mapped) { vkUnmapMemory(device_, s.stage_mem); s.mapped = nullptr; }
                vkFreeMemory(device_, s.stage_mem, nullptr);
                s.stage_mem = VK_NULL_HANDLE;
            }
            if (s.desc_set != VK_NULL_HANDLE) {
                vkFreeDescriptorSets(device_, desc_pool_, 1, &s.desc_set);
                s.desc_set = VK_NULL_HANDLE;
            }
        }

        for (auto& s : slots_) {
            if (!dma_only_) {
                // Compute mode: device-local buffer (compute output)
                s.dev_buf = create_buffer(needed,
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, s.dev_mem);
            }

            // Host-visible staging buffer
            s.stage_buf = create_buffer(needed,
                VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
                s.stage_mem);

            VK_CHECK(vkMapMemory(device_, s.stage_mem, 0, needed, 0, &s.mapped), "vkMapMemory");

            if (!dma_only_) {
                // Allocate descriptor set
                VkDescriptorSetAllocateInfo ds_ai{};
                ds_ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
                ds_ai.descriptorPool     = desc_pool_;
                ds_ai.descriptorSetCount = 1;
                ds_ai.pSetLayouts        = &desc_layout_;
                VK_CHECK(vkAllocateDescriptorSets(device_, &ds_ai, &s.desc_set), "vkAllocateDescriptorSets");

                // Write SSBO descriptor (binding 1) — texture descriptor (binding 0) is written per-frame
                VkDescriptorBufferInfo buf_info{};
                buf_info.buffer = s.dev_buf;
                buf_info.offset = 0;
                buf_info.range  = needed;

                VkWriteDescriptorSet write{};
                write.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                write.dstSet          = s.desc_set;
                write.dstBinding      = 1;
                write.descriptorCount = 1;
                write.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                write.pBufferInfo     = &buf_info;
                vkUpdateDescriptorSets(device_, 1, &write, 0, nullptr);
            }
        }

        buf_size_ = needed;

        CASPAR_LOG(debug) << L"[vk_readback] Allocated " << slots_.size() << L" buffer slots, "
                          << (needed / 1024) << L" KB each"
                          << (dma_only_ ? L" (DMA-only, raw pixels)" : L"");
    }

    VkBuffer create_buffer(size_t size, VkBufferUsageFlags usage,
                           VkMemoryPropertyFlags mem_props, VkDeviceMemory& out_mem)
    {
        VkBufferCreateInfo buf_ci{};
        buf_ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        buf_ci.size  = size;
        buf_ci.usage = usage;

        VkBuffer buf;
        VK_CHECK(vkCreateBuffer(device_, &buf_ci, nullptr, &buf), "vkCreateBuffer");

        VkMemoryRequirements reqs;
        vkGetBufferMemoryRequirements(device_, buf, &reqs);

        VkMemoryAllocateInfo alloc{};
        alloc.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        alloc.allocationSize  = reqs.size;
        alloc.memoryTypeIndex = find_memory_type(phys_device_, reqs.memoryTypeBits, mem_props);

        VK_CHECK(vkAllocateMemory(device_, &alloc, nullptr, &out_mem), "vkAllocateMemory");
        VK_CHECK(vkBindBufferMemory(device_, buf, out_mem, 0), "vkBindBufferMemory");

        return buf;
    }

    // ─── Texture import ────────────────────────────────────────────────────

    imported_texture* ensure_texture_import(void* win32_handle, unsigned long long alloc_size,
                                            int width, int height, bool is_16bit)
    {
        // Check cache
        for (int i = 0; i < num_cached_tex_; ++i) {
            auto& t = tex_cache_[i];
            if (t.handle == win32_handle && t.w == width && t.h == height)
                return &t;
        }

        // Evict oldest if full
        if (num_cached_tex_ >= MAX_TEX_CACHE) {
            wait_all_slot_fences();
            auto& old = tex_cache_[0];
            invalidate_sem_for_handle(old.handle);
            if (old.view)  vkDestroyImageView(device_, old.view, nullptr);
            if (old.image) vkDestroyImage(device_, old.image, nullptr);
            if (old.mem)   vkFreeMemory(device_, old.mem, nullptr);
            for (int i = 1; i < num_cached_tex_; ++i)
                tex_cache_[i-1] = tex_cache_[i];
            num_cached_tex_--;
        }

        auto& t = tex_cache_[num_cached_tex_];
        t = {};

        VkFormat format = is_16bit ? VK_FORMAT_R16G16B16A16_UNORM : VK_FORMAT_R8G8B8A8_UNORM;
        current_format_ = format;

        using namespace caspar::accelerator::vulkan;

        // Import external memory
#ifdef _WIN32
        VkImportMemoryWin32HandleInfoKHR import_info{};
        import_info.sType      = VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_KHR;
        import_info.handleType = platform::kExternalMemoryHandleType;
        import_info.handle     = win32_handle;
#else
        int dup_fd = dup(vulkan_common::platform::from_opaque(win32_handle));
        if (dup_fd < 0)
            throw std::runtime_error("[vk_readback] Failed to dup fd for VK import");

        VkImportMemoryFdInfoKHR import_info{};
        import_info.sType      = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR;
        import_info.handleType = platform::kExternalMemoryHandleType;
        import_info.fd         = dup_fd;
#endif

        VkMemoryAllocateInfo alloc{};
        alloc.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        alloc.pNext           = &import_info;
        alloc.allocationSize  = alloc_size;

        // Query valid memory types for this handle, then pick a device-local one
        uint32_t valid_type_bits = ~0u;
#ifdef _WIN32
        auto vkGetMemoryWin32HandlePropertiesKHR_ =
            (PFN_vkGetMemoryWin32HandlePropertiesKHR)vkGetDeviceProcAddr(device_, "vkGetMemoryWin32HandlePropertiesKHR");
        if (vkGetMemoryWin32HandlePropertiesKHR_) {
            VkMemoryWin32HandlePropertiesKHR handle_props{};
            handle_props.sType = VK_STRUCTURE_TYPE_MEMORY_WIN32_HANDLE_PROPERTIES_KHR;
            if (vkGetMemoryWin32HandlePropertiesKHR_(device_,
                    platform::kExternalMemoryHandleType,
                    win32_handle, &handle_props) == VK_SUCCESS) {
                valid_type_bits = handle_props.memoryTypeBits;
            }
        }
#else
        auto vkGetMemoryFdPropertiesKHR_ =
            (PFN_vkGetMemoryFdPropertiesKHR)vkGetDeviceProcAddr(device_, "vkGetMemoryFdPropertiesKHR");
        if (vkGetMemoryFdPropertiesKHR_) {
            VkMemoryFdPropertiesKHR handle_props{};
            handle_props.sType = VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR;
            if (vkGetMemoryFdPropertiesKHR_(device_,
                    platform::kExternalMemoryHandleType,
                    dup_fd, &handle_props) == VK_SUCCESS) {
                valid_type_bits = handle_props.memoryTypeBits;
            }
        }
#endif

        VkPhysicalDeviceMemoryProperties mem_props;
        vkGetPhysicalDeviceMemoryProperties(phys_device_, &mem_props);
        bool found = false;
        for (uint32_t i = 0; i < mem_props.memoryTypeCount; ++i) {
            if ((valid_type_bits & (1u << i)) &&
                (mem_props.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
                alloc.memoryTypeIndex = i;
                found = true;
                break;
            }
        }
        if (!found) {
            // Fallback: any valid memory type
            for (uint32_t i = 0; i < mem_props.memoryTypeCount; ++i) {
                if (valid_type_bits & (1u << i)) {
                    alloc.memoryTypeIndex = i;
                    found = true;
                    break;
                }
            }
            if (!found)
                throw std::runtime_error("No valid memory type for imported VK texture");
        }

        {
            VkResult alloc_res = vkAllocateMemory(device_, &alloc, nullptr, &t.mem);
            if (alloc_res != VK_SUCCESS) {
#ifndef _WIN32
                // On Linux, vkAllocateMemory failure does NOT consume the fd
                if (dup_fd >= 0) ::close(dup_fd);
#endif
                throw std::runtime_error("[vk_readback] vkAllocateMemory (import) failed: " +
                                         std::to_string(alloc_res));
            }
        }

#ifndef _WIN32
        // On Linux, successful vkAllocateMemory with VkImportMemoryFdInfoKHR
        // consumes the fd — mark as invalid to prevent double-close.
        dup_fd = -1;
#endif

        // Create VkImage for the imported memory
        VkExternalMemoryImageCreateInfo ext_img{};
        ext_img.sType       = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
        ext_img.handleTypes = platform::kExternalMemoryHandleType;

        VkImageCreateInfo img_ci{};
        img_ci.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        img_ci.pNext         = &ext_img;
        img_ci.imageType     = VK_IMAGE_TYPE_2D;
        img_ci.format        = format;
        img_ci.extent        = {(uint32_t)width, (uint32_t)height, 1};
        img_ci.mipLevels     = 1;
        img_ci.arrayLayers   = 1;
        img_ci.samples       = VK_SAMPLE_COUNT_1_BIT;
        img_ci.tiling        = VK_IMAGE_TILING_OPTIMAL;
        img_ci.usage         = dma_only_
                                   ? (VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT)
                                   : VK_IMAGE_USAGE_SAMPLED_BIT;
        img_ci.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
        img_ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        VK_CHECK(vkCreateImage(device_, &img_ci, nullptr, &t.image), "vkCreateImage (import)");
        VK_CHECK(vkBindImageMemory(device_, t.image, t.mem, 0), "vkBindImageMemory (import)");

        // Create image view
        VkImageViewCreateInfo view_ci{};
        view_ci.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        view_ci.image                           = t.image;
        view_ci.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
        view_ci.format                          = format;
        view_ci.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        view_ci.subresourceRange.baseMipLevel   = 0;
        view_ci.subresourceRange.levelCount     = 1;
        view_ci.subresourceRange.baseArrayLayer = 0;
        view_ci.subresourceRange.layerCount     = 1;

        VK_CHECK(vkCreateImageView(device_, &view_ci, nullptr, &t.view), "vkCreateImageView (import)");

        t.handle = win32_handle;
        t.w      = width;
        t.h      = height;
        num_cached_tex_++;

        CASPAR_LOG(debug) << L"[vk_readback] Imported VK texture " << num_cached_tex_
                          << L"/" << MAX_TEX_CACHE << L" (" << width << L"x" << height << L")";
        return &t;
    }

    // ─── Semaphore import ──────────────────────────────────────────────────

    VkSemaphore ensure_semaphore_import(void* win32_handle)
    {
        for (int i = 0; i < num_cached_sem_; ++i)
            if (sem_cache_[i].handle == win32_handle)
                return sem_cache_[i].sem;

        if (num_cached_sem_ >= MAX_SEM_CACHE) {
            wait_all_slot_fences();
            vkDestroySemaphore(device_, sem_cache_[0].sem, nullptr);
            for (int i = 1; i < num_cached_sem_; ++i)
                sem_cache_[i-1] = sem_cache_[i];
            num_cached_sem_--;
        }

        // Create a timeline semaphore and import the handle
        VkSemaphoreTypeCreateInfo type_ci{};
        type_ci.sType         = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
        type_ci.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;

        VkSemaphoreCreateInfo sem_ci{};
        sem_ci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        sem_ci.pNext = &type_ci;

        VkSemaphore sem;
        VK_CHECK(vkCreateSemaphore(device_, &sem_ci, nullptr, &sem), "vkCreateSemaphore");

#ifdef _WIN32
        VkImportSemaphoreWin32HandleInfoKHR import_info{};
        import_info.sType      = VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_WIN32_HANDLE_INFO_KHR;
        import_info.semaphore  = sem;
        import_info.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
        import_info.handle     = win32_handle;

        auto vkImportSemaphoreWin32HandleKHR_ =
            (PFN_vkImportSemaphoreWin32HandleKHR)vkGetDeviceProcAddr(device_, "vkImportSemaphoreWin32HandleKHR");
        if (!vkImportSemaphoreWin32HandleKHR_) {
            vkDestroySemaphore(device_, sem, nullptr);
            throw std::runtime_error("[vk_readback] vkImportSemaphoreWin32HandleKHR not available");
        }

        auto res = vkImportSemaphoreWin32HandleKHR_(device_, &import_info);
        if (res != VK_SUCCESS) {
            vkDestroySemaphore(device_, sem, nullptr);
            throw std::runtime_error("[vk_readback] vkImportSemaphoreWin32HandleKHR failed: " +
                                     std::to_string(res));
        }
#else
        VkImportSemaphoreFdInfoKHR import_info{};
        import_info.sType      = VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR;
        import_info.semaphore  = sem;
        import_info.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
        // vkImportSemaphoreFdKHR consumes the fd on success — dup first
        import_info.fd         = dup(static_cast<int>(reinterpret_cast<intptr_t>(win32_handle)));
        if (import_info.fd < 0) {
            vkDestroySemaphore(device_, sem, nullptr);
            throw std::runtime_error("[vk_readback] Failed to dup semaphore fd");
        }

        auto vkImportSemaphoreFdKHR_ =
            (PFN_vkImportSemaphoreFdKHR)vkGetDeviceProcAddr(device_, "vkImportSemaphoreFdKHR");
        if (!vkImportSemaphoreFdKHR_) {
            ::close(import_info.fd);
            vkDestroySemaphore(device_, sem, nullptr);
            throw std::runtime_error("[vk_readback] vkImportSemaphoreFdKHR not available");
        }

        auto res = vkImportSemaphoreFdKHR_(device_, &import_info);
        if (res != VK_SUCCESS) {
            // On failure, fd is NOT consumed — close it
            ::close(import_info.fd);
            vkDestroySemaphore(device_, sem, nullptr);
            throw std::runtime_error("[vk_readback] vkImportSemaphoreFdKHR failed: " +
                                     std::to_string(res));
        }
        // On success, fd is consumed by the driver — do not close
#endif

        sem_cache_[num_cached_sem_] = {win32_handle, sem};
        num_cached_sem_++;

        CASPAR_LOG(info) << L"[vk_readback] Imported VK timeline semaphore (slot "
                         << num_cached_sem_ << L"/" << MAX_SEM_CACHE << L")";
        return sem;
    }

    void invalidate_sem_for_handle(void* handle)
    {
        if (!handle) return;
        for (int i = 0; i < num_cached_sem_; ++i) {
            if (sem_cache_[i].handle == handle) {
                wait_all_slot_fences();
                vkDestroySemaphore(device_, sem_cache_[i].sem, nullptr);
                for (int j = i + 1; j < num_cached_sem_; ++j)
                    sem_cache_[j-1] = sem_cache_[j];
                num_cached_sem_--;
                return;
            }
        }
    }

    // ─── DMA-only frame readback ──────────────────────────────────────────

    // In DMA mode, we copy the raw image subregion to host staging using
    // vkCmdCopyImageToBuffer (DMA engine only — no compute shader, no SM contention).
    // Returns a shared_ptr<void> pointing at the staging buffer's raw RGBA pixels,
    // or nullptr on failure.  The caller wraps these pixels in a const_frame and
    // passes them through the CPU v210 strategy.
    struct dma_result
    {
        std::shared_ptr<void> pixels;
        int                   width  = 0;
        int                   height = 0;
        // Set when `pixels` is an already-packed blank output frame rather than raw
        // staging pixels: the caller must pass it straight through instead of
        // running the CPU RGBA->v210 conversion over it.
        bool                  is_blank = false;
    };

    dma_result convert_frame_dma(
        const core::video_format_desc& decklink_format_desc,
        const port_configuration&      config,
        const core::const_frame&       frame)
    {
        auto tex = frame.texture();
        if (!tex) return {};

        auto* wrapper = dynamic_cast<accelerator::vulkan::texture_wrapper*>(tex.get());
        if (!wrapper) return {};

        caspar::timer total_timer;
        caspar::timer step_timer;

        auto vk_tex = wrapper->vk_texture();
        if (!vk_tex) return {};

        // export_native_handle() returns native_handle_t: void* on Windows, an int fd on
        // Linux. Declaring it `void*` did not compile there. The validity test stays on the
        // NATIVE value because the invalid handle is nullptr on Windows and -1 on Linux, so
        // `!mem_handle` was wrong in both directions on Linux.
        const auto native_mem = vk_tex->export_native_handle();
        if (native_mem == vulkan_common::platform::kInvalidHandle) return {};
        void* mem_handle = vulkan_common::platform::to_opaque(native_mem);

        const uint8_t* luid = vk_tex->device_luid();
        if (!luid) return {};

        bool is_16bit = vk_tex->depth() != common::bit_depth::bit8;
        int src_w = vk_tex->width();
        int src_h = vk_tex->height();

        ensure_initialized(luid);

        // Import texture (cached)
        step_timer = caspar::timer();
        auto* imported = ensure_texture_import(mem_handle, vk_tex->alloc_size(), src_w, src_h, is_16bit);
        double import_ms = step_timer.elapsed() * 1000.0;

        int src_x = config.src_x;
        int src_y = config.src_y;
        int dst_w = decklink_format_desc.width;
        int dst_h = decklink_format_desc.height;

        ensure_buffers(dst_w, dst_h);

        // Every slot still held by DeckLink means we cannot start another readback;
        // let the caller use the CPU strategy for this frame.
        const int cur_write = acquire_slot();
        if (cur_write < 0)
            return {blank_output(decklink_format_desc), dst_w, dst_h, /*is_blank=*/true};
        auto& slot = slots_[cur_write];

        // Import timeline semaphore for GPU-side wait
        void*    sem_handle = wrapper->render_semaphore_handle();
        uint64_t sem_value  = wrapper->render_semaphore_value();
        VkSemaphore wait_sem = VK_NULL_HANDLE;
        if (sem_handle && sem_value > 0) {
            try {
                wait_sem = ensure_semaphore_import(sem_handle);
            } catch (const std::exception& e) {
                CASPAR_LOG(warning) << L"[vk_readback] DMA: semaphore import failed: " << e.what();
                wrapper->ensure_render_complete();
            }
        } else {
            if (frame_count_ < 3)
                CASPAR_LOG(warning) << L"[vk_readback] DMA: no timeline semaphore (sem_handle="
                                    << sem_handle << L" sem_value=" << sem_value << L")";
            wrapper->ensure_render_complete();
        }

        // Record command buffer: barrier + image-to-buffer copy
        step_timer = caspar::timer();
        vkResetCommandBuffer(slot.cmd, 0);

        VkCommandBufferBeginInfo begin{};
        begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(slot.cmd, &begin), "vkBeginCommandBuffer (dma)");

        // Image layout transition: external → transfer src
        VkImageMemoryBarrier img_barrier{};
        img_barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        img_barrier.srcAccessMask       = 0;
        img_barrier.dstAccessMask       = VK_ACCESS_TRANSFER_READ_BIT;
        img_barrier.oldLayout           = VK_IMAGE_LAYOUT_GENERAL;
        img_barrier.newLayout           = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        img_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL;
        img_barrier.dstQueueFamilyIndex = compute_qf_;
        img_barrier.image               = imported->image;
        img_barrier.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

        vkCmdPipelineBarrier(slot.cmd,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &img_barrier);

        // Copy subregion of image to host-visible staging buffer
        VkBufferImageCopy region{};
        region.bufferOffset      = 0;
        region.bufferRowLength   = 0; // tightly packed
        region.bufferImageHeight = 0; // tightly packed
        region.imageSubresource  = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageOffset       = {src_x, src_y, 0};
        region.imageExtent       = {(uint32_t)dst_w, (uint32_t)dst_h, 1};

        vkCmdCopyImageToBuffer(slot.cmd, imported->image,
                               VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               slot.stage_buf, 1, &region);

        // Transition image back to GENERAL and release to external queue
        VkImageMemoryBarrier release_barrier{};
        release_barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        release_barrier.srcAccessMask       = VK_ACCESS_TRANSFER_READ_BIT;
        release_barrier.dstAccessMask       = 0;
        release_barrier.oldLayout           = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        release_barrier.newLayout           = VK_IMAGE_LAYOUT_GENERAL;
        release_barrier.srcQueueFamilyIndex = compute_qf_;
        release_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL;
        release_barrier.image               = imported->image;
        release_barrier.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

        vkCmdPipelineBarrier(slot.cmd,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            0, 0, nullptr, 0, nullptr, 1, &release_barrier);

        VK_CHECK(vkEndCommandBuffer(slot.cmd), "vkEndCommandBuffer (dma)");

        // Submit with optional timeline semaphore wait
        VkSubmitInfo submit{};
        submit.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers    = &slot.cmd;

        VkTimelineSemaphoreSubmitInfo timeline_submit{};
        VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;

        if (wait_sem != VK_NULL_HANDLE) {
            timeline_submit.sType                     = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
            timeline_submit.waitSemaphoreValueCount   = 1;
            timeline_submit.pWaitSemaphoreValues      = &sem_value;

            submit.pNext              = &timeline_submit;
            submit.waitSemaphoreCount = 1;
            submit.pWaitSemaphores    = &wait_sem;
            submit.pWaitDstStageMask  = &wait_stage;
        }
        // Reset fence just before submit — if any prior VK call failed and threw,
        // the fence remains signaled, preventing deadlock on the next wait.
        vkResetFences(device_, 1, &slot.fence);
        VK_CHECK(vkQueueSubmit(compute_queue_, 1, &submit, slot.fence), "vkQueueSubmit (dma)");
        double submit_ms = step_timer.elapsed() * 1000.0;

        // Hand back the readback submitted PIPELINE_DEPTH frames ago, and TIME IT.
        // `push_and_take` contains the only call here that blocks the DeckLink thread --
        // vkWaitForFences with UINT64_MAX. Reading `total_ms` before it, as this code did
        // until 2026-08-20, excluded the one wait the number exists to account for.
        caspar::timer wait_timer;
        const int     ready   = push_and_take(cur_write);
        double        wait_ms = wait_timer.elapsed() * 1000.0;

        double total_ms = total_timer.elapsed() * 1000.0;

        // Periodic timing report
        accum_import_ms_  += import_ms;
        accum_wait_ms_    += wait_ms;
        accum_submit_ms_  += submit_ms;
        accum_total_ms_   += total_ms;
        frame_count_++;
        if (frame_count_ % 50 == 0) {
            double n = 50.0;
            CASPAR_LOG(info) << L"[vk_readback] DMA DIAG avg/50: "
                             << L"import=" << (accum_import_ms_ / n) << L"ms "
                             << L"wait=" << (accum_wait_ms_ / n) << L"ms "
                             << L"submit=" << (accum_submit_ms_ / n) << L"ms "
                             << L"total=" << (accum_total_ms_ / n) << L"ms";
            accum_import_ms_ = accum_wait_ms_ = accum_submit_ms_ = accum_total_ms_ = 0.0;
        }

        if (ready < 0)
            return {blank_output(decklink_format_desc), dst_w, dst_h, /*is_blank=*/true};

        return {make_staging_ref(ready), dst_w, dst_h};
    }

    // ─── Frame conversion ──────────────────────────────────────────────────

    std::shared_ptr<void> convert_frame(
        const core::video_format_desc& decklink_format_desc,
        const port_configuration&      config,
        const core::const_frame&       frame)
    {
        auto tex = frame.texture();
        if (!tex) return nullptr;

        auto* wrapper = dynamic_cast<accelerator::vulkan::texture_wrapper*>(tex.get());
        if (!wrapper) return nullptr;

        caspar::timer total_timer;
        caspar::timer step_timer;

        // Get VK texture info
        auto vk_tex = wrapper->vk_texture();
        if (!vk_tex) return nullptr;

        // See the note at the other call site: native_handle_t is an fd on Linux, and the
        // invalid value is -1 rather than 0.
        const auto native_mem = vk_tex->export_native_handle();
        if (native_mem == vulkan_common::platform::kInvalidHandle) return nullptr;
        void* mem_handle = vulkan_common::platform::to_opaque(native_mem);

        const uint8_t* luid = vk_tex->device_luid();
        if (!luid) return nullptr;

        bool is_16bit = vk_tex->depth() != common::bit_depth::bit8;
        int src_w = vk_tex->width();
        int src_h = vk_tex->height();

        // Lazy init — first frame tells us the GPU LUID
        ensure_initialized(luid);

        // Import texture (cached)
        step_timer = caspar::timer();
        auto* imported = ensure_texture_import(mem_handle, vk_tex->alloc_size(), src_w, src_h, is_16bit);
        double import_ms = step_timer.elapsed() * 1000.0;

        int src_x = config.src_x;
        int src_y = config.src_y;
        int dst_w = decklink_format_desc.width;
        int dst_h = decklink_format_desc.height;

        // DESTINATION PLACEMENT. `region_w`/`region_h` of 0 mean the whole frame, which is the
        // no-subregion case; passing dest 0,0 and region dst_w x dst_h then reduces the
        // shaders' arithmetic to exactly what it was before they understood placement, so that
        // path stays bit-identical.
        //
        // These used to be dropped, and the consumer coerced itself to the CPU readback to
        // honour them (see config.cpp). They are cheap on the GPU once the shader walks OUTPUT
        // pixels rather than source ones: a V210 group is 6 pixels, so a dest_x that is not a
        // multiple of 6 straddles a group -- which needs read-modify-write if you walk the
        // source, and nothing at all if every output group is computed from scratch.
        int dest_x   = config.dest_x;
        int dest_y   = config.dest_y;
        int region_w = config.region_w > 0 ? config.region_w : dst_w;
        int region_h = config.region_h > 0 ? config.region_h : dst_h;

        // Ensure output buffers
        ensure_buffers(dst_w, dst_h);

        // Every slot still held by DeckLink means we cannot start another readback;
        // let the caller use the CPU strategy for this frame.
        const int cur_write = acquire_slot();
        if (cur_write < 0)
            return blank_output(decklink_format_desc);
        auto& slot = slots_[cur_write];

        // Import timeline semaphore for GPU-side wait
        void*    sem_handle = wrapper->render_semaphore_handle();
        uint64_t sem_value  = wrapper->render_semaphore_value();
        VkSemaphore wait_sem = VK_NULL_HANDLE;
        if (sem_handle && sem_value > 0) {
            try {
                wait_sem = ensure_semaphore_import(sem_handle);
            } catch (const std::exception& e) {
                CASPAR_LOG(warning) << L"[vk_readback] Semaphore import failed: " << e.what();
                wrapper->ensure_render_complete(); // CPU fallback
            }
        } else {
            if (frame_count_ < 3)
                CASPAR_LOG(warning) << L"[vk_readback] No timeline semaphore on frame (sem_handle="
                                    << sem_handle << L" sem_value=" << sem_value << L") - using CPU fence wait";
            wrapper->ensure_render_complete(); // No semaphore available
        }

        // Update descriptor set binding 0 (texture) for this frame
        VkDescriptorImageInfo img_info{};
        img_info.sampler     = sampler_;
        img_info.imageView   = imported->view;
        img_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkWriteDescriptorSet write{};
        write.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet          = slot.desc_set;
        write.dstBinding      = 0;
        write.descriptorCount = 1;
        write.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write.pImageInfo      = &img_info;
        vkUpdateDescriptorSets(device_, 1, &write, 0, nullptr);

        // Record command buffer
        step_timer = caspar::timer();
        vkResetCommandBuffer(slot.cmd, 0);

        VkCommandBufferBeginInfo begin{};
        begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(slot.cmd, &begin), "vkBeginCommandBuffer");

        // Image layout transition: acquire from external queue family
        VkImageMemoryBarrier img_barrier{};
        img_barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        img_barrier.srcAccessMask       = 0;
        img_barrier.dstAccessMask       = VK_ACCESS_SHADER_READ_BIT;
        img_barrier.oldLayout           = VK_IMAGE_LAYOUT_GENERAL;
        img_barrier.newLayout           = VK_IMAGE_LAYOUT_GENERAL;
        img_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL;
        img_barrier.dstQueueFamilyIndex = compute_qf_;
        img_barrier.image               = imported->image;
        img_barrier.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

        vkCmdPipelineBarrier(slot.cmd,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &img_barrier);

        // Bind pipeline and descriptors
        vkCmdBindPipeline(slot.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
        vkCmdBindDescriptorSets(slot.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_layout_,
                                0, 1, &slot.desc_set, 0, nullptr);

        // Push constants and dispatch
        if (is_hdr_ || use_bt2020_ || needs_v210_) {
            int groups_per_row = (dst_w + 5) / 6;
            int push_data[11] = {src_x, src_y, dst_w, dst_h, groups_per_row,
                                 use_bt2020_ ? 1 : 0, is_16bit ? 1 : 0,
                                 dest_x, dest_y, region_w, region_h};
            vkCmdPushConstants(slot.cmd, pipe_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, 44, push_data);

            uint32_t gx = (uint32_t)((groups_per_row + 63) / 64);
            uint32_t gy = (uint32_t)dst_h;
            vkCmdDispatch(slot.cmd, gx, gy, 1);
        } else {
            int push_data[8] = {src_x, src_y, dst_w, dst_h, dest_x, dest_y, region_w, region_h};
            vkCmdPushConstants(slot.cmd, pipe_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, 32, push_data);

            uint32_t gx = (uint32_t)((dst_w + 15) / 16);
            uint32_t gy = (uint32_t)((dst_h + 15) / 16);
            vkCmdDispatch(slot.cmd, gx, gy, 1);
        }

        // Barrier: compute write → transfer read
        VkBufferMemoryBarrier buf_barrier{};
        buf_barrier.sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        buf_barrier.srcAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;
        buf_barrier.dstAccessMask       = VK_ACCESS_TRANSFER_READ_BIT;
        buf_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        buf_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        buf_barrier.buffer              = slot.dev_buf;
        buf_barrier.offset              = 0;
        buf_barrier.size                = buf_size_;

        vkCmdPipelineBarrier(slot.cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, nullptr, 1, &buf_barrier, 0, nullptr);

        // Copy device buffer → staging buffer
        VkBufferCopy copy{};
        copy.size = buf_size_;
        vkCmdCopyBuffer(slot.cmd, slot.dev_buf, slot.stage_buf, 1, &copy);

        VK_CHECK(vkEndCommandBuffer(slot.cmd), "vkEndCommandBuffer");

        // Submit with optional timeline semaphore wait
        VkSubmitInfo submit{};
        submit.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers    = &slot.cmd;

        VkTimelineSemaphoreSubmitInfo timeline_submit{};
        VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

        if (wait_sem != VK_NULL_HANDLE) {
            timeline_submit.sType                     = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
            timeline_submit.waitSemaphoreValueCount   = 1;
            timeline_submit.pWaitSemaphoreValues      = &sem_value;

            submit.pNext              = &timeline_submit;
            submit.waitSemaphoreCount = 1;
            submit.pWaitSemaphores    = &wait_sem;
            submit.pWaitDstStageMask  = &wait_stage;
        }

        // Reset fence just before submit to prevent deadlock if prior VK calls threw.
        vkResetFences(device_, 1, &slot.fence);
        VK_CHECK(vkQueueSubmit(compute_queue_, 1, &submit, slot.fence), "vkQueueSubmit");
        double submit_ms = step_timer.elapsed() * 1000.0;

        // Hand back the readback submitted PIPELINE_DEPTH frames ago, and TIME IT.
        // `push_and_take` contains the only call here that blocks the DeckLink thread --
        // vkWaitForFences with UINT64_MAX. Reading `total_ms` before it, as this code did
        // until 2026-08-20, excluded the one wait the number exists to account for.
        caspar::timer wait_timer;
        const int     ready   = push_and_take(cur_write);
        double        wait_ms = wait_timer.elapsed() * 1000.0;

        double total_ms = total_timer.elapsed() * 1000.0;

        // Periodic timing report
        accum_import_ms_  += import_ms;
        accum_wait_ms_    += wait_ms;
        accum_submit_ms_  += submit_ms;
        accum_total_ms_   += total_ms;
        frame_count_++;
        if (frame_count_ % 50 == 0) {
            double n = 50.0;
            CASPAR_LOG(info) << L"[vk_readback] DIAG avg/50: "
                             << L"import=" << (accum_import_ms_ / n) << L"ms "
                             << L"wait=" << (accum_wait_ms_ / n) << L"ms "
                             << L"submit=" << (accum_submit_ms_ / n) << L"ms "
                             << L"total=" << (accum_total_ms_ / n) << L"ms";
            accum_import_ms_ = accum_wait_ms_ = accum_submit_ms_ = accum_total_ms_ = 0.0;
        }

        if (ready < 0)
            return blank_output(decklink_format_desc);

        return make_staging_ref(ready);
    }
};

#endif // ENABLE_VULKAN

// ===========================================================================
// Public interface
// ===========================================================================

vk_readback_strategy::vk_readback_strategy(bool is_hdr, bool use_bt2020,
                                           spl::shared_ptr<format_strategy> fallback,
                                           bool dma_only,
                                           bool needs_v210,
                                           int  buffer_depth)
#ifdef ENABLE_VULKAN
    : impl_(std::make_unique<impl>(is_hdr, use_bt2020, std::move(fallback), dma_only, needs_v210, buffer_depth))
#else
    : impl_(nullptr)
#endif
{
    CASPAR_LOG(info) << L"[vk_readback] Pure-Vulkan GPU readback: "
                     << (dma_only ? L"DMA-only " : L"")
                     << (is_hdr ? L"HDR " : L"SDR ")
                     << (use_bt2020 ? L"BT.2020" : L"BT.709")
                     << (needs_v210 ? L" V210" : L"");
}

vk_readback_strategy::~vk_readback_strategy() = default;

BMDPixelFormat vk_readback_strategy::get_pixel_format()
{
#ifdef ENABLE_VULKAN
    if (impl_)
        return (impl_->is_hdr_ || impl_->use_bt2020_ || impl_->needs_v210_) ? bmdFormat10BitYUV : bmdFormat8BitBGRA;
#endif
    return bmdFormat8BitBGRA;
}

int vk_readback_strategy::get_row_bytes(int width)
{
#ifdef ENABLE_VULKAN
    if (impl_) {
        if (impl_->is_hdr_ || impl_->use_bt2020_ || impl_->needs_v210_)
            return ((width + 47) / 48) * 128;
        else
            return width * 4;
    }
#endif
    return width * 4;
}

std::shared_ptr<void> vk_readback_strategy::allocate_frame_data(const core::video_format_desc& format_desc)
{
#ifdef ENABLE_VULKAN
    if (impl_)
        return impl_->fallback_->allocate_frame_data(format_desc);
#endif
    return nullptr;
}

std::shared_ptr<void> vk_readback_strategy::convert_frame_for_port(
    const core::video_format_desc& channel_format_desc,
    const core::video_format_desc& decklink_format_desc,
    const port_configuration&      config,
    const core::const_frame&       frame1,
    const core::const_frame&       frame2,
    BMDFieldDominance              field_dominance)
{
#ifdef ENABLE_VULKAN
    if (impl_) {
        try {
            if (impl_->dma_only_) {
                // DMA mode: VK DMA copy raw pixels → CPU v210 conversion via fallback
                auto dma = impl_->convert_frame_dma(decklink_format_desc, config, frame1);
                if (dma.pixels && dma.is_blank) {
                    // Already a packed blank output frame (pipeline filling, or every
                    // slot still held by DeckLink). Must not be run through the CPU
                    // RGBA->v210 conversion below, and must not fall through to the
                    // CPU strategy: needs_cpu_frame_data() is false for GPU readback
                    // modes, so frame1 carries no host pixels.
                    return dma.pixels;
                }
                if (dma.pixels) {
                    // Wrap the staging buffer as a const_frame for the CPU v210 strategy.
                    // The staging has raw RGBA pixels (UNORM 8 or 16-bit) for the subregion
                    // only, already cropped to (dst_w × dst_h).
                    bool is_16bit = (impl_->current_format_ == VK_FORMAT_R16G16B16A16_UNORM);
                    int bpp = is_16bit ? 8 : 4;
                    size_t data_size = (size_t)dma.width * dma.height * bpp;

                    // Build a pixel_format_desc matching the raw pixels.
                    //
                    // THE ORDER DEPENDS ON THE DEPTH, and this said `bgra` unconditionally.
                    // The mixer writes BGRA into an 8-bit R8G8B8A8_UNORM attachment but
                    // writes RGBA directly at 16-bit — the same rule `vk_readback_v210.comp`
                    // branches on with its `is_16bit` push constant, and the reason
                    // `screen_consumer.cpp` swizzles only when `!hbd`. Declaring bgra for
                    // 16-bit handed the CPU v210 converter red where it expected blue.
                    //
                    // Measured over the 1->4 loopback, yuv@16bit on gpu-readback-mode=
                    // vulkan-dma: 7.68 dB against the reference and 52.4 dB against it with
                    // red and blue exchanged, where `auto` and `cpu` score 52.53. The 8-bit
                    // wire on the same mode was always correct, which is why this survived.
                    auto pfd = core::pixel_format_desc(is_16bit ? core::pixel_format::rgba
                                                                : core::pixel_format::bgra,
                                                       impl_->use_bt2020_ ? core::color_space::bt2020
                                                                          : core::color_space::bt709,
                                                       core::color_transfer::sdr);
                    auto depth = is_16bit ? common::bit_depth::bit16 : common::bit_depth::bit8;
                    pfd.planes.push_back(core::pixel_format_desc::plane(dma.width, dma.height, 4, depth));

                    // Create an array wrapping the staging pointer (kept alive by dma.pixels shared_ptr)
                    auto pixel_storage = dma.pixels; // extend lifetime
                    caspar::array<const std::uint8_t> image_arr(
                        reinterpret_cast<const std::uint8_t*>(dma.pixels.get()),
                        data_size,
                        std::move(pixel_storage));

                    std::vector<caspar::array<const std::uint8_t>> planes;
                    planes.push_back(std::move(image_arr));

                    // Empty audio — we don't use it in format_strategy
                    caspar::array<const std::int32_t> no_audio;

                    core::const_frame staging_frame(nullptr, std::move(planes), std::move(no_audio), pfd);

                    // The CPU fallback strategy converts raw RGBA→v210.
                    // Use a zero-offset config since the subregion was already extracted by DMA.
                    port_configuration flat_config = config;
                    flat_config.src_x    = 0;
                    flat_config.src_y    = 0;
                    flat_config.dest_x   = 0;
                    flat_config.dest_y   = 0;
                    flat_config.region_w = 0;
                    flat_config.region_h = 0;

                    // Build a "virtual" channel format matching the DMA output dimensions
                    auto staging_format = decklink_format_desc;

                    return impl_->fallback_->convert_frame_for_port(
                        staging_format, decklink_format_desc, flat_config,
                        staging_frame, staging_frame, field_dominance);
                }
            } else {
                // Compute shader mode
                auto result = impl_->convert_frame(decklink_format_desc, config, frame1);
                if (result)
                    return result;
            }
        } catch (const std::exception& e) {
            CASPAR_LOG(warning) << L"[vk_readback] GPU conversion failed: " << e.what()
                                << L" - falling back to CPU";
        }
        // Fallback. Only usable when the frame actually carries host pixels: this
        // consumer reports needs_cpu_frame_data()==false for GPU readback modes, so
        // for a GPU-only frame the CPU packer would read an `unavailable`
        // image_data() and crash. Emit a blank frame in that case instead.
        // (This also covers interlaced output, which has always taken this path.)
        if (frame1 && frame1.host_image_state() == core::host_image_availability::unavailable) {
            CASPAR_LOG(debug) << L"[vk_readback] No host pixels for the CPU fallback; emitting a blank frame.";
            return impl_->blank_output(decklink_format_desc);
        }
        return impl_->fallback_->convert_frame_for_port(
            channel_format_desc, decklink_format_desc, config, frame1, frame2, field_dominance);
    }
#endif
    return nullptr;
}

spl::shared_ptr<format_strategy> try_create_vk_readback_strategy(
    bool is_hdr, bool use_bt2020,
    spl::shared_ptr<format_strategy> fallback,
    bool dma_only,
    bool needs_v210,
    int  buffer_depth)
{
#ifdef ENABLE_VULKAN
    try {
        // Quick check: Vulkan instance creation must work
        VkInstanceCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        VkApplicationInfo app{};
        app.sType      = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app.apiVersion = VK_API_VERSION_1_2;
        ci.pApplicationInfo = &app;
        VkInstance test_inst;
        if (vkCreateInstance(&ci, nullptr, &test_inst) == VK_SUCCESS) {
            vkDestroyInstance(test_inst, nullptr);
            return spl::make_shared<vk_readback_strategy>(
                is_hdr, use_bt2020, std::move(fallback), dma_only, needs_v210, buffer_depth);
        }
        CASPAR_LOG(warning) << L"[vk_readback] Vulkan not available, using CPU fallback";
    } catch (const std::exception& e) {
        CASPAR_LOG(warning) << L"[vk_readback] Failed to create VK readback strategy: " << e.what();
    }
#endif
    return fallback;
}

}} // namespace caspar::decklink
