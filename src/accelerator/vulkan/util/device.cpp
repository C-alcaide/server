/*
 * Copyright 2025
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
 * Author: Niklas Andersson, niklas@niklaspandersson.se
 */

#include "device.h"

#include "../image/image_kernel.h"
#include "buffer.h"
#include "pipeline.h"
#include <cstdlib>
#include <string>
#include <mutex>
#include <map>
#include "glsl_compiler.h"

#pragma warning(push)
#pragma warning(disable : 4838 4309)
#include "vk_image_fragment_src.h"
#pragma warning(pop)
#include "platform_config.h"
#include "texture.h"

#include <common/array.h>
#include <common/assert.h>
#include <common/env.h>
#include <common/except.h>
#include <common/os/thread.h>
#include <common/vulkan/gpu_luid.h>
#include <common/vulkan/icd_filter.h>

#include <VkBootstrap.h>
#include <vulkan/vulkan.hpp>

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#define VMA_IMPLEMENTATION
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4189)
#endif
#include <vk_mem_alloc.h>
#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include <boost/asio/deadline_timer.hpp>
#include <boost/asio/dispatch.hpp>
#include <boost/asio/spawn.hpp>
#include <boost/property_tree/ptree.hpp>

#include <tbb/concurrent_queue.h>
#include <tbb/concurrent_unordered_map.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <deque>
#include <future>
#include <thread>

namespace caspar { namespace accelerator { namespace vulkan {

using namespace boost::asio;

inline VKAPI_ATTR VkBool32 VKAPI_CALL default_debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                                                             VkDebugUtilsMessageTypeFlagsEXT        messageType,
                                                             const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
                                                             void*)
{
    auto ms = vkb::to_string_message_severity(messageSeverity);
    auto mt = vkb::to_string_message_type(messageType);
    if (messageType & VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT) {
        CASPAR_LOG(info) << "[" << ms << ": " << mt << "] - " << pCallbackData->pMessageIdName << ", "
                         << pCallbackData->pMessage;
        // printf("[%s: %s] - %s\n%s\n", ms, mt, pCallbackData->pMessageIdName, pCallbackData->pMessage);
    } else {
        if (pCallbackData->pMessage != nullptr) {
            CASPAR_LOG(info) << "[" << ms << ": " << mt << "] " << pCallbackData->pMessage;
            // printf("[%s: %s]\n%s\n", ms, mt, pCallbackData->pMessage);
        }
    }

    return VK_FALSE; // Applications must return false here (Except Validation, if return true, will skip calling to
                     // driver)
}

void transitionImageLayout(const vk::Image&        image,
                           vk::ImageLayout         oldLayout,
                           vk::AccessFlags2        srcAccessMask,
                           vk::PipelineStageFlags2 srcStage,
                           vk::ImageLayout         newLayout,
                           vk::AccessFlags2        dstAccessMask,
                           vk::PipelineStageFlags2 dstStage,
                           vk::CommandBuffer       cmdBuffer)
{
    auto range = vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);

    vk::ImageMemoryBarrier2 barrier{};
    barrier.oldLayout = oldLayout, barrier.newLayout = newLayout, barrier.srcQueueFamilyIndex = vk::QueueFamilyIgnored,
    barrier.dstQueueFamilyIndex = vk::QueueFamilyIgnored, barrier.image = image, barrier.subresourceRange = range;

    barrier.srcAccessMask = srcAccessMask;
    barrier.srcStageMask  = srcStage;

    barrier.dstAccessMask = dstAccessMask;
    barrier.dstStageMask  = dstStage;

    vk::DependencyInfo dep_info;
    dep_info.setImageMemoryBarriers(barrier);

    cmdBuffer.pipelineBarrier2(dep_info);
}

struct device::impl : public std::enable_shared_from_this<impl>
{
    using texture_queue_t = tbb::concurrent_bounded_queue<std::shared_ptr<texture>>;
    using buffer_queue_t  = tbb::concurrent_bounded_queue<std::shared_ptr<buffer>>;

    // 0 = unorm8, 1 = unorm16, 2 = fp16. The render format has to be part of the key:
    // a VkImage's format is fixed at creation, so pooling fp16 with unorm16 would hand
    // an fp16 attachment back as a unorm16 one. Same hazard as the OGL texture pool.
    std::array<tbb::concurrent_unordered_map<size_t, texture_queue_t>, 3>                attachment_pools_;
    std::array<std::array<tbb::concurrent_unordered_map<size_t, texture_queue_t>, 4>, 2> device_pools_;
    std::array<tbb::concurrent_unordered_map<size_t, buffer_queue_t>, 2>                 host_pools_;

    /// [8-bit|16-bit][component count] -> can this GPU sample that packed format.
    /// Index 0 is unused (there is no 0-component format). Filled by
    /// probe_sampled_formats() at construction; see can_sample_packed().
    std::array<std::array<bool, 5>, 2> sampled_ok_{};

    std::wstring version_;

    vkb::Instance                      _vkb_instance;
    vkb::PhysicalDevice                _vkb_physical_device;
    vk::PhysicalDeviceMemoryProperties _memoryProperties;
    vk::PhysicalDevice                 _physical_device;
    vk::Device                         _device;
    vk::Queue                          _queue;
    uint32_t                           _queue_family = 0;
    // A queue reserved for another API to submit on -- see `getDecodeQueue`. Null, and
    // _decode_queue_family == _queue_family, when this GPU has no separate compute family.
    vk::Queue                          _decode_queue;
    uint32_t                           _decode_queue_family = 0;
    bool                               _decode_queue_dedicated = false;
    vk::CommandPool                    _command_pool;
    VmaAllocator                       _allocator;

    uint8_t _device_luid[8] = {};
    bool    _device_luid_valid = false;

    // One pipeline per attachment format, indexed as attachment_pools_ is. These are the
    // BASE pipelines, built from the SPIR-V glslc produced at configure time, and they are
    // kept as a fixed array rather than folded into the cache below so that the path taken
    // by every existing draw is unchanged and cannot be perturbed by cache behaviour.
    std::array<std::shared_ptr<pipeline>, 3> _pipelines;

    // Pipelines for generated colour transforms, keyed on (variant id, attachment format).
    //
    // A Vulkan pipeline is bound to its shader module, so a variant is a whole pipeline and
    // not merely a different program -- which is why this exists on Vulkan and the OGL side
    // needed only a shader cache. The format is part of the key because a pipeline is also
    // bound to its attachment format: the same transform on an 8-bit and an fp16 channel are
    // two pipelines.
    //
    // Unbounded on purpose, unlike the OGL shader cache's retention window. The number of
    // live variants is bounded by the number of distinct colour spaces in use on this device,
    // which is small and operator-driven; evicting a pipeline that a later draw needs would
    // trade a bounded amount of memory for an unbounded recompile stall on the frame path.
    std::map<std::pair<std::string, int>, std::shared_ptr<pipeline>> _variant_pipelines;
    std::mutex                                                      _variant_mutex;

    struct inflight_command_buffer
    {
        vk::CommandBuffer cmd;
        uint64_t          semaphore_value;
    };
    std::deque<inflight_command_buffer> _transfer_cmd_buffers;
    vk::Semaphore                       _semaphore;
    uint64_t                            _semaphore_value{0};

    io_context                             io_context_;
    decltype(make_work_guard(io_context_)) work_;
    std::thread                            thread_;
    std::thread::id                        thread_id_;

    // ── Dispatch-thread instrumentation ──────────────────────────────────
    // Every upload, composition pass, LUT pass and readback for every channel
    // on this GPU is funnelled onto one io_context on one thread.
    // VULKAN_MIXER_IMPLEMENTATION.md defends that as avoiding external
    // synchronisation, and the plan to add a transfer queue and per-thread
    // command recording assumes the thread is a scaling wall. Both positions
    // were argued from the code, never measured, so measure them:
    //
    //   wait   -- how long work sits enqueued before the thread picks it up.
    //             Near zero means there is no contention to relieve.
    //   exec   -- how long it then takes.
    //   busy   -- share of wall-clock the thread spends executing. Approaching
    //             100% is the only thing that makes parallel recording urgent.
    //   depth  -- peak simultaneous outstanding work.
    //
    // Published under vk.dispatch.* in info(), which "GL INFO" returns for both
    // backends (its message says OpenGL only; the path is generic).
    // Split by kind, because the aggregate cannot tell a saturated thread from an
    // idle one held up by one long item -- and those want opposite fixes.
    enum class dispatch_kind
    {
        other = 0, // composition, LUT passes, allocation, gc
        upload,    // host -> device staging copies
        readback,  // device -> host, including the wait for it to complete
        count_
    };
    static const wchar_t* kind_name(dispatch_kind k)
    {
        switch (k) {
            case dispatch_kind::upload:
                return L"upload";
            case dispatch_kind::readback:
                return L"readback";
            default:
                return L"other";
        }
    }

    struct dispatch_stats
    {
        std::atomic<int64_t>  depth{0};
        std::atomic<int64_t>  depth_peak{0};
        std::atomic<uint64_t> count{0};
        std::atomic<uint64_t> wait_us{0};
        std::atomic<uint64_t> wait_us_peak{0};
        std::atomic<uint64_t> exec_us{0};
        std::atomic<uint64_t> exec_us_peak{0};
    };
    mutable dispatch_stats stats_;
    mutable std::array<dispatch_stats, static_cast<size_t>(dispatch_kind::count_)> kind_stats_;
    std::chrono::steady_clock::time_point stats_since_{std::chrono::steady_clock::now()};

    // Command-buffer recycling health. submitSingleTimeCommands reclaims at most
    // one finished buffer per call, and only if the *oldest* has completed, so the
    // in-flight deque can never shrink faster than it grows. Track both the depth
    // and how often we had to allocate a fresh buffer instead of reusing one --
    // a rising allocation rate is what a failing recycler looks like from outside.
    // Written only from the dispatch thread; read from info() on another, hence
    // atomic.
    std::atomic<int64_t>  cmd_buffers_inflight_{0};
    std::atomic<uint64_t> cmd_buffers_allocated_{0};
    std::atomic<uint64_t> cmd_buffers_reused_{0};

    void* thread_native_handle_ = nullptr;

    /// CPU time actually consumed by the dispatch thread, in microseconds, or 0
    /// if unavailable. Distinguishes "this thread is the bottleneck" from "this
    /// machine is oversubscribed and the thread cannot get scheduled".
    uint64_t dispatch_thread_cpu_us() const
    {
#ifdef _WIN32
        FILETIME creation{}, exit{}, kernel{}, user{};
        if (thread_native_handle_ &&
            GetThreadTimes(reinterpret_cast<HANDLE>(thread_native_handle_), &creation, &exit, &kernel, &user)) {
            const auto to_us = [](const FILETIME& ft) {
                return ((static_cast<uint64_t>(ft.dwHighDateTime) << 32) | ft.dwLowDateTime) / 10ULL;
            };
            return to_us(kernel) + to_us(user);
        }
#endif
        return 0;
    }

    template <typename T>
    static void atomic_max(std::atomic<T>& target, T value)
    {
        auto prev = target.load(std::memory_order_relaxed);
        while (prev < value && !target.compare_exchange_weak(prev, value, std::memory_order_relaxed))
            ;
    }

    /// Wraps a dispatch handler so its queue wait and execution time are
    /// recorded. Returns a move-only handler, which is what asio wants anyway.
    template <typename Handler>
    auto instrument(Handler&& handler, dispatch_kind kind = dispatch_kind::other)
    {
        auto&      kstats   = kind_stats_[static_cast<size_t>(kind)];
        const auto enqueued = std::chrono::steady_clock::now();
        atomic_max<int64_t>(stats_.depth_peak, stats_.depth.fetch_add(1, std::memory_order_relaxed) + 1);
        atomic_max<int64_t>(kstats.depth_peak, kstats.depth.fetch_add(1, std::memory_order_relaxed) + 1);

        return [this, &kstats, enqueued, handler = std::forward<Handler>(handler)]() mutable {
            const auto started = std::chrono::steady_clock::now();
            // Decrement and record even if the handler throws: a leaked depth
            // count would make the queue look permanently backed up.
            struct finish_guard
            {
                dispatch_stats*                       all;
                dispatch_stats*                       kind;
                std::chrono::steady_clock::time_point started;
                ~finish_guard()
                {
                    const auto exec_us = static_cast<uint64_t>(
                        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() -
                                                                             started)
                            .count());
                    for (auto* s : {all, kind}) {
                        s->exec_us.fetch_add(exec_us, std::memory_order_relaxed);
                        atomic_max<uint64_t>(s->exec_us_peak, exec_us);
                        s->depth.fetch_sub(1, std::memory_order_relaxed);
                    }
                }
            } guard{&stats_, &kstats, started};

            const auto wait_us = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::microseconds>(started - enqueued).count());
            for (auto* s : {&stats_, &kstats}) {
                s->wait_us.fetch_add(wait_us, std::memory_order_relaxed);
                atomic_max<uint64_t>(s->wait_us_peak, wait_us);
                s->count.fetch_add(1, std::memory_order_relaxed);
            }

            handler();
        };
    }

    void publish_dispatch_stats(boost::property_tree::wptree& info) const
    {
        const auto window_us = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - stats_since_)
                .count());

        const auto publish = [&](const std::wstring& prefix, const dispatch_stats& s) {
            const auto count   = s.count.load(std::memory_order_relaxed);
            const auto exec_us = s.exec_us.load(std::memory_order_relaxed);
            const auto wait_us = s.wait_us.load(std::memory_order_relaxed);

            info.add(prefix + L".count", count);
            info.add(prefix + L".depth", s.depth.load(std::memory_order_relaxed));
            info.add(prefix + L".depth_peak", s.depth_peak.load(std::memory_order_relaxed));
            info.add(prefix + L".wait_avg_us", count ? wait_us / count : 0);
            info.add(prefix + L".wait_peak_us", s.wait_us_peak.load(std::memory_order_relaxed));
            info.add(prefix + L".exec_avg_us", count ? exec_us / count : 0);
            info.add(prefix + L".exec_peak_us", s.exec_us_peak.load(std::memory_order_relaxed));
            // A double, not integer percent: an idle thread is the expected answer
            // on a lightly loaded server, and "0" cannot tell 0.4% from 0.004%.
            info.add(prefix + L".busy_percent",
                     window_us ? (static_cast<double>(exec_us) * 100.0) / static_cast<double>(window_us) : 0.0);
            info.add(prefix + L".exec_total_ms", exec_us / 1000);
        };

        publish(L"vk.dispatch", stats_);
        info.add(L"vk.dispatch.window_ms", window_us / 1000);

        // busy_percent above is wall-clock: it counts time an item was in
        // progress, including time the thread was preempted. cpu_percent is the
        // thread's own CPU consumption. When the two diverge, the thread is
        // waiting or descheduled rather than working, and giving it more threads
        // to record on would not help.
        const auto cpu_us = dispatch_thread_cpu_us();
        info.add(L"vk.dispatch.cpu_total_ms", cpu_us / 1000);
        info.add(L"vk.dispatch.cpu_percent",
                 window_us ? (static_cast<double>(cpu_us) * 100.0) / static_cast<double>(window_us) : 0.0);

        const auto allocated = cmd_buffers_allocated_.load(std::memory_order_relaxed);
        const auto reused    = cmd_buffers_reused_.load(std::memory_order_relaxed);
        info.add(L"vk.cmd_buffers.inflight", cmd_buffers_inflight_.load(std::memory_order_relaxed));
        info.add(L"vk.cmd_buffers.allocated", allocated);
        info.add(L"vk.cmd_buffers.reused", reused);
        info.add(L"vk.cmd_buffers.reuse_percent",
                 (allocated + reused) ? (static_cast<double>(reused) * 100.0) / static_cast<double>(allocated + reused)
                                      : 0.0);

        for (size_t i = 0; i < kind_stats_.size(); ++i)
            publish(std::wstring(L"vk.dispatch_by_kind.") + kind_name(static_cast<dispatch_kind>(i)), kind_stats_[i]);
    }

    explicit impl(int gpu_index)
        : work_(make_work_guard(io_context_))
    {
        CASPAR_LOG(info) << L"Initializing Vulkan Device (gpu_index=" << gpu_index << L").";

        vulkan_common::filter_stale_nvidia_icds();

        auto instance_builder = vkb::InstanceBuilder()
#ifdef _DEBUG
                                    .enable_validation_layers(true)
                                    .set_debug_messenger_severity(VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                                                  VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
                                    .set_debug_messenger_type(VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                                                              VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                                                              VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT)
                                    .set_debug_callback(default_debug_callback)
#endif
                                    .set_app_name("CasparCG")
                                    .set_headless(true)
                                    .set_engine_name("CasparCG")
                                    .require_api_version(VK_API_VERSION_1_3);
        auto instance_ret = instance_builder.build();
        if (!instance_ret) {
            CASPAR_THROW_EXCEPTION(caspar_exception()
                                   << msg_info("Failed to create Vulkan instance: " + instance_ret.error().message()));
        }
        _vkb_instance = instance_ret.value();

        VULKAN_HPP_DEFAULT_DISPATCHER.init(_vkb_instance.fp_vkGetInstanceProcAddr);
        VULKAN_HPP_DEFAULT_DISPATCHER.init(vk::Instance(_vkb_instance.instance));

        // Find suitable physical device
        auto gpu_selector = vkb::PhysicalDeviceSelector(_vkb_instance);

        vk::PhysicalDeviceFeatures features10;
        features10.textureCompressionBC = true;

        vk::PhysicalDeviceVulkan12Features features12;
        features12.descriptorIndexing                        = true;
        features12.descriptorBindingPartiallyBound           = true;
        features12.runtimeDescriptorArray                    = true;
        features12.shaderSampledImageArrayNonUniformIndexing = true;
        features12.timelineSemaphore                         = true;
        features12.scalarBlockLayout                         = true;

        vk::PhysicalDeviceVulkan13Features features13;
        features13.dynamicRendering = true;
        features13.synchronization2 = true;

        vk::PhysicalDeviceDynamicRenderingLocalReadFeaturesKHR localReadFeatures;
        localReadFeatures.dynamicRenderingLocalRead = true;

        auto gpu_devices_res = gpu_selector.set_minimum_version(1, 3)
                                   .set_required_features(features10)
                                   .set_required_features_12(features12)
                                   .set_required_features_13(features13)
                                   .add_required_extension(VK_KHR_DYNAMIC_RENDERING_LOCAL_READ_EXTENSION_NAME)
                                   .add_required_extension_features(localReadFeatures)
                                   .add_required_extension(platform::kExtMemExtName)
                                   .add_required_extension(platform::kExtSemExtName)
                                   .prefer_gpu_device_type(vkb::PreferredDeviceType::discrete)
                                   .select_devices();
        if (!gpu_devices_res || gpu_devices_res.value().empty()) {
            CASPAR_THROW_EXCEPTION(caspar_exception()
                                   << msg_info("Failed to select physical device: " +
                                               (gpu_devices_res ? std::string("no suitable GPU")
                                                                : gpu_devices_res.error().message())));
        }

        // Resolve gpu_index against the SAME index space the vulkan_output consumer
        // uses: a raw vkEnumeratePhysicalDevices list deduplicated by LUID (first
        // occurrence wins). vk-bootstrap's select_devices() feature-filters and may
        // omit GPUs (e.g. an older Pascal lacking a required extension), which would
        // otherwise shift indices and silently break mixer/output GPU affinity.
        // We therefore map gpu_index -> target LUID here, then pick the matching
        // device out of the feature-suitable set.
        std::array<uint8_t, 8> target_luid{};
        bool                   have_target_luid = false;
        {
            uint32_t raw_count = 0;
            vkEnumeratePhysicalDevices(_vkb_instance.instance, &raw_count, nullptr);
            std::vector<VkPhysicalDevice> raw_devices(raw_count);
            if (raw_count > 0)
                vkEnumeratePhysicalDevices(_vkb_instance.instance, &raw_count, raw_devices.data());

            auto unique_devices = vulkan_common::deduplicate_by_luid(raw_devices);

            // Extract unique LUIDs in order to map gpu_index -> target LUID
            std::vector<std::array<uint8_t, 8>> unique_luids;
            for (auto pd : unique_devices) {
                std::array<uint8_t, 8> luid{};
                if (vulkan_common::query_device_luid(pd, luid))
                    unique_luids.push_back(luid);
            }

            if (gpu_index >= 0 && gpu_index < static_cast<int>(unique_luids.size())) {
                target_luid      = unique_luids[gpu_index];
                have_target_luid = true;
            } else {
                CASPAR_LOG(warning) << L"[accelerator] Requested mixer gpu_index " << gpu_index
                                    << L" is out of range (" << unique_luids.size()
                                    << L" unique GPU(s) enumerated). Falling back to first suitable GPU.";
            }
        }

        // Pick, among the feature-suitable devices, the one whose LUID matches the
        // target. Fall back to the first suitable device if there is no match.
        auto all_gpus = gpu_devices_res.value();
        int  selected_index = 0;
        bool matched         = false;
        if (have_target_luid) {
            for (size_t i = 0; i < all_gpus.size(); ++i) {
                std::array<uint8_t, 8> luid{};
                if (vulkan_common::query_device_luid(all_gpus[i].physical_device, luid) && luid == target_luid) {
                    selected_index = static_cast<int>(i);
                    matched        = true;
                    break;
                }
            }
            if (!matched) {
                CASPAR_LOG(warning) << L"[accelerator] Requested mixer gpu_index " << gpu_index
                                    << L" does not support the required mixer features; "
                                       L"falling back to first suitable GPU (output may incur a cross-GPU copy).";
            }
        }
        _vkb_physical_device = all_gpus[selected_index];

        CASPAR_LOG(info) << "Selected Vulkan device [" << selected_index
                         << "]: " << _vkb_physical_device.properties.deviceName;

        vk::PhysicalDeviceRobustness2FeaturesEXT robustness2Features;
        robustness2Features.nullDescriptor = true;
        _vkb_physical_device.enable_extension_features_if_present(robustness2Features);

        // Create the logical device
        auto device_builder = vkb::DeviceBuilder(_vkb_physical_device);
        _physical_device    = vk::PhysicalDevice(_vkb_physical_device.physical_device);

        auto device_res = device_builder.build();
        if (!device_res) {
            CASPAR_THROW_EXCEPTION(caspar_exception()
                                   << msg_info("Failed to create device: " + device_res.error().message()));
        }
        auto vkb_device = device_res.value();
        _device         = vk::Device(vkb_device.device);
        VULKAN_HPP_DEFAULT_DISPATCHER.init(_device);
        _queue            = vk::Queue(vkb_device.get_queue(vkb::QueueType::graphics).value());
        auto queue_family = vkb_device.get_queue_index(vkb::QueueType::graphics).value();
        _queue_family     = queue_family;

        // A SEPARATE queue for another API to submit on, so sharing this device never
        // means sharing a queue.
        //
        // Everything this class submits goes through one graphics queue with no mutex --
        // `_queue.submit` at two sites, protected only by the invariant that all device
        // work runs on the dispatch thread. A second API submitting to that same queue
        // from its own thread is undefined behaviour, and FFmpeg's own queue mutexes
        // would not help: they guard FFmpeg's submissions, not ours. Handing out a queue
        // from a different family removes the question instead of synchronising it.
        //
        // Falls back to the graphics family when the GPU offers no distinct compute
        // family. That is NOT silently equivalent, so it is recorded and reported by
        // `hasDedicatedDecodeQueue()`; a caller that needs isolation must check rather
        // than assume.
        _decode_queue_family    = queue_family;
        _decode_queue_dedicated = false;
        // `get_queue(compute)` and not `get_dedicated_queue(compute)`. vk-bootstrap's
        // "dedicated" means COMPUTE with neither GRAPHICS nor TRANSFER
        // (VkBootstrap.cpp:1047-1055, called with undesired_flags = TRANSFER), and
        // NVIDIA's compute-only family carries TRANSFER -- so it rejected the family
        // FFmpeg itself reports as "queue family 2 (queues: 8) for compute" on this GPU,
        // and the first version of this logged "no dedicated compute queue family" on a
        // card that plainly has one. The plain accessor uses `get_separate_queue_index`,
        // which is the question actually being asked: a compute family that is not the
        // graphics one.
        if (auto q = vkb_device.get_queue(vkb::QueueType::compute)) {
            if (auto qf = vkb_device.get_queue_index(vkb::QueueType::compute)) {
                _decode_queue           = vk::Queue(q.value());
                _decode_queue_family    = qf.value();
                _decode_queue_dedicated = _decode_queue_family != queue_family;
            }
        }
        if (!_decode_queue_dedicated) {
            CASPAR_LOG(info) << L"[vk::device] no dedicated compute queue family on this GPU; "
                                L"an external decoder would have to share the graphics queue";
        } else {
            CASPAR_LOG(info) << L"[vk::device] reserved compute queue family "
                             << _decode_queue_family << L" for external decoders (graphics is "
                             << queue_family << L")";
        }

        vk::CommandPoolCreateInfo pool_info;
        pool_info.flags            = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
        pool_info.queueFamilyIndex = queue_family;

        _command_pool = _device.createCommandPool(pool_info);

        vk::SemaphoreTypeCreateInfo timeline_info{};
        timeline_info.semaphoreType = vk::SemaphoreType::eTimeline;
        timeline_info.initialValue  = 0;
        vk::SemaphoreCreateInfo semaphore_info{};
        semaphore_info.pNext = &timeline_info;
        _semaphore           = _device.createSemaphore(semaphore_info);

        VmaVulkanFunctions vulkanFunctions    = {};
        vulkanFunctions.vkGetInstanceProcAddr = _vkb_instance.fp_vkGetInstanceProcAddr;
        vulkanFunctions.vkGetDeviceProcAddr   = vkb_device.fp_vkGetDeviceProcAddr;

        VmaAllocatorCreateInfo allocatorCreateInfo = {};
        allocatorCreateInfo.flags                  = VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT;
        allocatorCreateInfo.vulkanApiVersion       = VK_API_VERSION_1_3;
        allocatorCreateInfo.physicalDevice         = _physical_device;
        allocatorCreateInfo.device                 = _device;
        allocatorCreateInfo.instance               = _vkb_instance.instance;
        allocatorCreateInfo.pVulkanFunctions       = &vulkanFunctions;

        vmaCreateAllocator(&allocatorCreateInfo, &_allocator);

        _memoryProperties = _physical_device.getMemoryProperties();

        probe_sampled_formats();

        // Query device LUID for cross-GPU identification
        {
            std::array<uint8_t, 8> luid{};
            _device_luid_valid = vulkan_common::query_device_luid(_physical_device, luid);
            if (_device_luid_valid)
                std::memcpy(_device_luid, luid.data(), 8);
        }

        // Proving the runtime GLSL->SPIR-V path with the BASE shader, before any generated
        // colour transform exists. With CASPAR_VK_RUNTIME_SHADER=1 every pipeline is built
        // from source compiled by shaderc instead of the SPIR-V glslc produced at configure
        // time, so the conformance and grading batteries answer one question on its own:
        // can we compile this shader at runtime and get the same pixels?
        //
        // Keeping that separable matters because the alternative is discovering a shaderc
        // problem while also debugging an OCIO splice, with no way to tell which is at
        // fault. Off by default, and the default path is byte-identical to before.
        std::vector<uint32_t> runtime_frag;
        if (const char* env = std::getenv("CASPAR_VK_RUNTIME_SHADER"); env && *env == '1') {
            runtime_frag = compile_glsl_fragment_to_spirv(
                std::string(reinterpret_cast<const char*>(fragment_shader_src)), "base(runtime)");
            if (runtime_frag.empty()) {
                CASPAR_LOG(error) << L"[vulkan::device] CASPAR_VK_RUNTIME_SHADER=1 but the base "
                                     L"shader did not compile; falling back to the built-in SPIR-V";
            } else {
                CASPAR_LOG(info) << L"[vulkan::device] base fragment shader compiled at runtime: "
                                 << runtime_frag.size() << L" SPIR-V words";
            }
        }

        _pipelines[0] = std::make_shared<pipeline>(_device, vk::Format::eR8G8B8A8Unorm, _memoryProperties, runtime_frag);
        _pipelines[1] = std::make_shared<pipeline>(_device, vk::Format::eR16G16B16A16Unorm, _memoryProperties, runtime_frag);
        _pipelines[2] = std::make_shared<pipeline>(_device, vk::Format::eR16G16B16A16Sfloat, _memoryProperties, runtime_frag);

        thread_ = std::thread([&] {
            set_thread_name(L"Vulkan Device");
            io_context_.run();
        });
        // Taken from the thread object on this thread rather than assigned from inside
        // the lambda. That was an unsynchronised write read by dispatch_sync() and
        // allocateCommandBuffers(), and it left a window after the thread started but
        // before the lambda ran in which thread_id_ still held a default-constructed
        // id -- so an early dispatch_sync() would post-and-wait instead of running
        // inline, and the assertion in allocateCommandBuffers() could fire even when
        // the caller really was the device thread.
        thread_id_ = thread_.get_id();
        // Kept so busy_percent can be reported as real CPU time, not wall time.
        // On a server whose consumers are compressing video, the dispatch thread
        // spends part of every item descheduled, and a wall-clock measurement
        // charges that to Vulkan.
        thread_native_handle_ = reinterpret_cast<void*>(thread_.native_handle());
    }

    ~impl()
    {
        work_.reset();
        thread_.join();

        _device.waitIdle();

        for (auto& pool : host_pools_)
            pool.clear();

        for (auto& pool : attachment_pools_)
            pool.clear();

        for (auto& pools : device_pools_)
            for (auto& pool : pools)
                pool.clear();

        _transfer_cmd_buffers.clear();
        _device.destroySemaphore(_semaphore);

        _device.destroyCommandPool(_command_pool);
        vmaDestroyAllocator(_allocator);
        for (auto& pipeline : _pipelines) {
            pipeline.reset();
        }

        _device.destroy();
        vkb::destroy_instance(_vkb_instance);
    }

    template <typename Func>
    auto spawn_async(Func&& func)
    {
        using result_type = decltype(func(std::declval<yield_context>()));
        using task_type   = std::packaged_task<result_type(yield_context)>;

        auto task   = task_type(std::forward<Func>(func));
        auto future = task.get_future();
        boost::asio::spawn(io_context_,
                           std::move(task)
#if BOOST_VERSION >= 108000
                               ,
                           [](std::exception_ptr e) {
                               if (e)
                                   std::rethrow_exception(e);
                           }
#endif
        );
        return future;
    }

    template <typename Func>
    auto dispatch_async(Func&& func, dispatch_kind kind = dispatch_kind::other)
    {
        using result_type = decltype(func());
        using task_type   = std::packaged_task<result_type()>;

        auto task   = task_type(std::forward<Func>(func));
        auto future = task.get_future();
        boost::asio::dispatch(io_context_, instrument(std::move(task), kind));
        return future;
    }

    template <typename Func>
    auto dispatch_sync(Func&& func) -> decltype(func())
    {
        if (std::this_thread::get_id() == thread_id_)
            return func();
        return dispatch_async(std::forward<Func>(func)).get();
    }

    std::wstring version() { return version_; }

    uint32_t findDedicatedMemoryType(uint32_t typeMask, vk::MemoryPropertyFlags properties)
    {
        for (uint32_t i = 0; i < _memoryProperties.memoryTypeCount; ++i) {
            if ((typeMask & (1 << i)) &&
                ((_memoryProperties.memoryTypes[i].propertyFlags & properties) == properties)) {
                return i;
            }
        }
        throw std::runtime_error("Failed to find suitable memory type");
    }

    uint64_t submitSingleTimeCommands(std::function<void(const vk::CommandBuffer&)> func)
    {
        vk::CommandBuffer cmd_buffer = nullptr;
        if (_transfer_cmd_buffers.size() > 1) {
            auto completed = _device.getSemaphoreCounterValue(_semaphore);

            // try to reuse the oldest existing command buffer
            if (_transfer_cmd_buffers.front().semaphore_value <= completed) {
                cmd_buffer = _transfer_cmd_buffers.front().cmd;
                cmd_buffer.reset();
                _transfer_cmd_buffers.pop_front();
                cmd_buffers_reused_.fetch_add(1, std::memory_order_relaxed);
            }
        }

        if (!cmd_buffer) {
            // create a new command buffer
            vk::CommandBufferAllocateInfo allocInfo{};
            allocInfo.commandPool        = _command_pool;
            allocInfo.level              = vk::CommandBufferLevel::ePrimary;
            allocInfo.commandBufferCount = 1;

            cmd_buffer = _device.allocateCommandBuffers(allocInfo)[0];
            cmd_buffers_allocated_.fetch_add(1, std::memory_order_relaxed);
        }

        cmd_buffer.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        func(cmd_buffer);
        cmd_buffer.end();

        auto                            signal_value = ++_semaphore_value;
        vk::TimelineSemaphoreSubmitInfo timelineInfo{};
        timelineInfo.setSignalSemaphoreValues(signal_value);

        vk::SubmitInfo submitInfo{};
        submitInfo.setCommandBuffers(cmd_buffer);
        submitInfo.setSignalSemaphores(_semaphore);
        submitInfo.pNext = &timelineInfo;
        _queue.submit(submitInfo);

        _transfer_cmd_buffers.push_back({cmd_buffer, signal_value});
        cmd_buffers_inflight_.store(static_cast<int64_t>(_transfer_cmd_buffers.size()), std::memory_order_relaxed);

        return signal_value;
    }

    std::vector<vk::CommandBuffer> allocateCommandBuffers(uint32_t count)
    {
        // _command_pool has to be externally synchronised (Vulkan requires it for
        // vkAllocateCommandBuffers and every vkCmd* on buffers from it), and
        // submitSingleTimeCommands() allocates from it on the device thread.
        //
        // This used to assert that the caller was already on that thread, and the
        // assertion fired routinely: image_kernel's constructor calls this, and the
        // kernel is a member of the mixer, so it is constructed on whichever thread
        // builds the channel -- never the device thread. The assertion was reporting a
        // genuine race with the transfer path, not a bad assumption.
        //
        // Dispatch to the device thread instead. dispatch_sync runs func() inline when
        // already there, so the transfer path keeps its direct call.
        return dispatch_sync([this, count] {
            return _device.allocateCommandBuffers(
                vk::CommandBufferAllocateInfo(_command_pool, vk::CommandBufferLevel::ePrimary, count));
        });
    }
    void submit(const vk::SubmitInfo& submitInfo, vk::Fence fence) { _queue.submit(submitInfo, fence); }

    std::shared_ptr<texture>
    create_attachment(int                   width,
                      int                   height,
                      common::bit_depth     depth,
                      uint32_t              components_count,
                      common::render_format render_format)
    {
        CASPAR_VERIFY(width > 0 && height > 0);

        const auto depth_pool_index = attachment_pool_index(depth, render_format);
        const auto format           = attachment_format(depth, render_format);

        // TODO (perf) Shared pool.
        auto pool   = &attachment_pools_[depth_pool_index][static_cast<size_t>(width) << 16 | static_cast<size_t>(height)];
        auto extent = vk::Extent3D{static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1};

        std::shared_ptr<texture> tex;
        if (!pool->try_pop(tex)) {
            // Chain external memory info so the attachment can be exported to
            // other VkDevices on the same physical GPU (VK→VK zero-copy output).
            vk::ExternalMemoryImageCreateInfo extMemImageInfo{};
            extMemImageInfo.handleTypes = static_cast<vk::ExternalMemoryHandleTypeFlagBits>(platform::kExternalMemoryHandleType);

            vk::ImageCreateInfo imageInfo{};
            imageInfo.pNext         = &extMemImageInfo;
            imageInfo.imageType     = vk::ImageType::e2D;
            imageInfo.format        = format;
            imageInfo.extent        = extent;
            imageInfo.mipLevels     = 1;
            imageInfo.arrayLayers   = 1;
            imageInfo.initialLayout = vk::ImageLayout::eUndefined;
            imageInfo.samples       = vk::SampleCountFlagBits::e1;
            imageInfo.tiling        = vk::ImageTiling::eOptimal;
            imageInfo.usage         = vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eInputAttachment |
                              vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferDst |
                              vk::ImageUsageFlagBits::eSampled;
            imageInfo.sharingMode = vk::SharingMode::eExclusive;
            auto image            = _device.createImage(imageInfo);

            auto memReq = _device.getImageMemoryRequirements(image);

            vk::ExportMemoryAllocateInfo exportMemInfo{};
            exportMemInfo.handleTypes = static_cast<vk::ExternalMemoryHandleTypeFlagBits>(platform::kExternalMemoryHandleType);

            vk::MemoryAllocateInfo allocInfo{};
            allocInfo.pNext           = &exportMemInfo;
            allocInfo.allocationSize = memReq.size;
            allocInfo.memoryTypeIndex =
                findDedicatedMemoryType(memReq.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);

            auto imageMemory = _device.allocateMemory(allocInfo);
            _device.bindImageMemory(image, imageMemory, 0);
            auto range = vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);

            vk::ImageViewCreateInfo createInfo(
                {}, image, vk::ImageViewType::e2D, format, vk::ComponentMapping(), range);

            auto imageView = _device.createImageView(createInfo);

            tex = std::make_shared<texture>(width,
                                            height,
                                            components_count,
                                            depth,
                                            image,
                                            imageMemory,
                                            imageView,
                                            _device,
                                            memReq.size,
                                            render_format);
            if (_device_luid_valid)
                tex->set_device_luid(_device_luid);
        }

        submitSingleTimeCommands([&](vk::CommandBuffer cmd) {
            transitionImageLayout(
                tex->id(),
                vk::ImageLayout::eUndefined,
                vk::AccessFlagBits2::eNone,
                vk::PipelineStageFlagBits2::eTopOfPipe,
                vk::ImageLayout::eRenderingLocalRead,
                vk::AccessFlagBits2::eColorAttachmentWrite | vk::AccessFlagBits2::eInputAttachmentRead,
                vk::PipelineStageFlagBits2::eColorAttachmentOutput | vk::PipelineStageFlagBits2::eFragmentShader,
                cmd);
        });

        tex->set_depth(depth);

        auto ptr = tex.get();
        return std::shared_ptr<texture>(
            ptr, [tex = std::move(tex), pool, self = shared_from_this()](texture*) mutable { pool->push(tex); });
    }

    /// Which pipeline / attachment pool a (depth, render_format) pair selects. fp16 is a
    /// third row rather than a variant of the 16-bit row, because a VkImage's format is
    /// immutable and a pipeline is built against one attachment format.
    static int attachment_pool_index(common::bit_depth depth, common::render_format render_format)
    {
        if (render_format == common::render_format::fp16)
            return 2;
        return depth == common::bit_depth::bit8 ? 0 : 1;
    }

    static vk::Format attachment_format(common::bit_depth depth, common::render_format render_format)
    {
        if (render_format == common::render_format::fp16)
            return vk::Format::eR16G16B16A16Sfloat;
        return depth == common::bit_depth::bit8 ? vk::Format::eR8G8B8A8Unorm : vk::Format::eR16G16B16A16Unorm;
    }

    /// The sampled-image format an upload of `stride` packed components at `depth` uses.
    /// Uploads are always unorm: their source is an integer AVFrame.
    static vk::Format internal_format(int stride, common::bit_depth depth)
    {
        static const vk::Format INTERNAL_FORMAT[][5] = {{vk::Format::eUndefined,
                                                         vk::Format::eR8Unorm,
                                                         vk::Format::eR8G8Unorm,
                                                         vk::Format::eR8G8B8Unorm,
                                                         vk::Format::eR8G8B8A8Unorm},
                                                        {vk::Format::eUndefined,
                                                         vk::Format::eR16Unorm,
                                                         vk::Format::eR16G16Unorm,
                                                         vk::Format::eR16G16B16Unorm,
                                                         vk::Format::eR16G16B16A16Unorm}};

        if (stride < 1 || stride > 4)
            return vk::Format::eUndefined;
        return INTERNAL_FORMAT[depth == common::bit_depth::bit8 ? 0 : 1][stride];
    }

    /// Can this GPU sample a packed `stride`-component image of `depth`?
    ///
    /// Only stride 3 can answer false in practice -- a 3-component format is the one
    /// entry in the table above that Vulkan does not oblige an implementation to support
    /// as a sampled image, and this GPU does not. Callers that ask FIRST can drop or
    /// convert the item; callers that do not get the throw in create_texture, from inside
    /// the channel's tick, once per frame.
    ///
    /// Answered from a table filled once, because the mixer asks per plane per item per
    /// frame and this must not become a driver round-trip on the channel thread.
    bool can_sample_packed(int stride, common::bit_depth depth) const
    {
        if (stride < 1 || stride > 4)
            return false;
        return sampled_ok_[depth == common::bit_depth::bit8 ? 0 : 1][stride];
    }

    void probe_sampled_formats()
    {
        for (int d = 0; d < 2; ++d) {
            const auto depth = d == 0 ? common::bit_depth::bit8 : common::bit_depth::bit16;
            for (int stride = 1; stride <= 4; ++stride) {
                const auto format = internal_format(stride, depth);
                const auto props  = _physical_device.getFormatProperties(format);
                sampled_ok_[d][stride] =
                    static_cast<bool>(props.optimalTilingFeatures & vk::FormatFeatureFlagBits::eSampledImage);
            }
            if (!sampled_ok_[d][3]) {
                CASPAR_LOG(info) << L"[vk::device] this GPU cannot sample packed 3-component "
                                 << (d == 0 ? L"8" : L"16") << L"-bit images; producers must convert packed "
                                 << (d == 0 ? L"24" : L"48") << L"-bit RGB before the mixer.";
            }
        }
    }

    std::shared_ptr<texture> create_texture(int width, int height, int stride, common::bit_depth depth, bool clear)
    {
        CASPAR_VERIFY(stride > 0 && stride < 5);
        CASPAR_VERIFY(width > 0 && height > 0);

        auto depth_pool_index = depth == common::bit_depth::bit8 ? 0 : 1;
        auto format           = internal_format(stride, depth);

        // Packed 3-byte RGB, on a GPU that cannot sample it. This is a hard stop and not
        // a fallback, because there is nothing to fall back to here: widening to 4
        // components would need a second staging buffer and a per-pixel expansion, which
        // belongs at the call site that still has the source planes.
        //
        // It reached here twice, both times from a producer that had not asked
        // can_sample_packed() first: an FFV1 RGB clip decoding to rgb24, and -- for far
        // longer, on every opaque PNG or JPEG -- the image producer. Both are fixed, and
        // the Vulkan image_mixer now asks before uploading, so this should be unreachable.
        // Keep it, and keep it specific: a driver error a caller cannot interpret is what
        // made the first one take a session to find.
        //
        // Restricted to stride 3 deliberately, rather than gating every stride on
        // can_sample_packed(). 16-bit UNORM formats are not on Vulkan's mandatory
        // sampled-image list either, and the 16-bit mixer works on this hardware today --
        // so asking the driver about them here could only turn a working path into a
        // throw on some future GPU, for no diagnostic gain.
        if (stride == 3 && !can_sample_packed(stride, depth)) {
            CASPAR_THROW_EXCEPTION(
                not_supported() << msg_info("This GPU cannot sample 3-component images at this depth, so the Vulkan "
                                            "mixer cannot take a packed 3-byte-per-pixel layout (rgb24/bgr24 and "
                                            "friends). Convert to a 4-component or planar format before the mixer, "
                                            "or use the OpenGL accelerator."));
        }

        auto pool   = &device_pools_[depth_pool_index][stride - 1][(width << 16 & 0xFFFF0000) | (height & 0x0000FFFF)];
        auto extent = vk::Extent3D{static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1};
        std::shared_ptr<texture> tex;
        if (!pool->try_pop(tex)) {
            vk::ImageCreateInfo imageInfo{};
            imageInfo.imageType     = vk::ImageType::e2D;
            imageInfo.format        = format;
            imageInfo.extent        = extent;
            imageInfo.mipLevels     = 1;
            imageInfo.arrayLayers   = 1;
            imageInfo.initialLayout = vk::ImageLayout::eUndefined;
            imageInfo.samples       = vk::SampleCountFlagBits::e1;
            imageInfo.tiling        = vk::ImageTiling::eOptimal;
            imageInfo.usage         = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled;
            imageInfo.sharingMode   = vk::SharingMode::eExclusive;
            auto image              = _device.createImage(imageInfo);

            auto memReq = _device.getImageMemoryRequirements(image);

            vk::MemoryAllocateInfo allocInfo{};
            allocInfo.allocationSize = memReq.size;
            allocInfo.memoryTypeIndex =
                findDedicatedMemoryType(memReq.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);

            auto imageMemory = _device.allocateMemory(allocInfo);
            _device.bindImageMemory(image, imageMemory, 0);
            auto clearValue = vk::ClearColorValue(std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f});
            auto range      = vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);

            vk::ImageViewCreateInfo createInfo(
                {}, image, vk::ImageViewType::e2D, format, vk::ComponentMapping(), range);

            auto imageView = _device.createImageView(createInfo);

            tex = std::make_shared<texture>(width, height, stride, depth, image, imageMemory, imageView, _device);
        }
        tex->set_depth(depth);

        auto ptr = tex.get();
        return std::shared_ptr<texture>(
            ptr, [tex = std::move(tex), pool, self = shared_from_this()](texture*) mutable { pool->push(tex); });
    }

    std::shared_ptr<buffer> create_buffer(int size, bool write)
    {
        CASPAR_VERIFY(size > 0);

        // TODO (perf) Shared pool.
        auto pool = &host_pools_[static_cast<int>(write ? 1 : 0)][size];

        std::shared_ptr<buffer> buf;
        if (!pool->try_pop(buf)) {
            buf = std::make_shared<buffer>(size, write, _allocator);
        }

        auto ptr = buf.get();
        return std::shared_ptr<buffer>(ptr, [buf = std::move(buf), self = shared_from_this()](buffer*) mutable {
            auto pool = &self->host_pools_[static_cast<int>(buf->write() ? 1 : 0)][buf->size()];
            pool->push(std::move(buf));
        });
    }

    array<uint8_t> create_array(int size)
    {
        auto buf = create_buffer(size, true);
        auto ptr = reinterpret_cast<uint8_t*>(buf->data());
        return array<uint8_t>(ptr, buf->size(), std::move(buf));
    }

    std::future<std::shared_ptr<texture>>
    copy_async(const array<const uint8_t>& source, int width, int height, int stride, common::bit_depth depth)
    {
        return dispatch_async(
            [this, source, width, height, stride, depth]() {
            std::shared_ptr<buffer> buf;

            // The array may already carry the staging buffer it was written into
            // (frame_factory hands out mapped VK memory), which lets us skip a
            // memcpy. Only when that buffer belongs to *this* device, though: a
            // VkBuffer referenced in a submit to another device is undefined
            // behaviour, and in practice takes the GPU down with ErrorDeviceLost.
            // A routed frame reaches a mixer on a different GPU exactly like this.
            auto tmp = source.storage<std::shared_ptr<buffer>>();
            if (tmp && *tmp && (*tmp)->allocator() == _allocator) {
                buf = *tmp;
            } else {
                buf = create_buffer(static_cast<int>(source.size()), true);
                std::memcpy(buf->data(), source.data(), source.size());
            }

            auto tex = create_texture(width, height, stride, depth, false);

            vk::BufferImageCopy region(0,
                                       0,
                                       0,
                                       vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, 0, 1),
                                       vk::Offset3D(0, 0, 0),
                                       vk::Extent3D(width, height, 1));

            submitSingleTimeCommands([&](vk::CommandBuffer cmd) {
                transitionImageLayout(tex->id(),
                                      vk::ImageLayout::eUndefined,
                                      vk::AccessFlagBits2::eNone,
                                      vk::PipelineStageFlagBits2::eTopOfPipe,

                                      vk::ImageLayout::eTransferDstOptimal,
                                      vk::AccessFlagBits2::eTransferWrite,
                                      vk::PipelineStageFlagBits2::eTransfer,
                                      cmd);

                cmd.copyBufferToImage(buf->id(), tex->id(), vk::ImageLayout::eTransferDstOptimal, region);

                transitionImageLayout(tex->id(),
                                      vk::ImageLayout::eTransferDstOptimal,
                                      vk::AccessFlagBits2::eTransferWrite,
                                      vk::PipelineStageFlagBits2::eTransfer,

                                      vk::ImageLayout::eShaderReadOnlyOptimal,
                                      vk::AccessFlagBits2::eShaderRead,
                                      vk::PipelineStageFlagBits2::eFragmentShader,
                                      cmd);
            });

            // No need to wait here, GPU-GPU deps (the usage of this texture on the device) are enforced by the memory
            // barriers
            return tex;
            },
            dispatch_kind::upload);
    }

    std::future<std::shared_ptr<texture>>
    copy_compressed_async(const array<const uint8_t>& source, int width, int height, vk::Format format)
    {
        return dispatch_async(
            [this, source, width, height, format]() {
            std::shared_ptr<buffer> buf;

            auto tmp = source.storage<std::shared_ptr<buffer>>();
            if (tmp) {
                buf = *tmp;
            } else {
                buf = create_buffer(static_cast<int>(source.size()), true);
                std::memcpy(buf->data(), source.data(), source.size());
            }

            // Create a VkImage with the compressed BC format
            auto extent = vk::Extent3D{static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1};

            vk::ImageCreateInfo imageInfo{};
            imageInfo.imageType     = vk::ImageType::e2D;
            imageInfo.format        = format;
            imageInfo.extent        = extent;
            imageInfo.mipLevels     = 1;
            imageInfo.arrayLayers   = 1;
            imageInfo.initialLayout = vk::ImageLayout::eUndefined;
            imageInfo.samples       = vk::SampleCountFlagBits::e1;
            imageInfo.tiling        = vk::ImageTiling::eOptimal;
            imageInfo.usage         = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled;
            imageInfo.sharingMode   = vk::SharingMode::eExclusive;

            auto image = _device.createImage(imageInfo);

            auto memReq = _device.getImageMemoryRequirements(image);

            vk::MemoryAllocateInfo allocInfo{};
            allocInfo.allocationSize  = memReq.size;
            allocInfo.memoryTypeIndex =
                findDedicatedMemoryType(memReq.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);

            auto imageMemory = _device.allocateMemory(allocInfo);
            _device.bindImageMemory(image, imageMemory, 0);

            auto range = vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);
            vk::ImageViewCreateInfo viewInfo({}, image, vk::ImageViewType::e2D, format, vk::ComponentMapping(), range);
            auto imageView = _device.createImageView(viewInfo);

            auto tex = std::make_shared<texture>(width, height, 4, common::bit_depth::bit8,
                                                 image, imageMemory, imageView, _device, memReq.size);
            // size() now reports width*height*4, which is what the image *represents*,
            // not what it holds. Flag it so copy_async refuses to read it back by that
            // count -- see the comment there.
            tex->set_compressed(true);

            // Copy staging buffer → compressed image (extent is in texels, not blocks)
            vk::BufferImageCopy region(0,
                                       0,
                                       0,
                                       vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, 0, 1),
                                       vk::Offset3D(0, 0, 0),
                                       vk::Extent3D(width, height, 1));

            submitSingleTimeCommands([&](vk::CommandBuffer cmd) {
                transitionImageLayout(tex->id(),
                                      vk::ImageLayout::eUndefined,
                                      vk::AccessFlagBits2::eNone,
                                      vk::PipelineStageFlagBits2::eTopOfPipe,

                                      vk::ImageLayout::eTransferDstOptimal,
                                      vk::AccessFlagBits2::eTransferWrite,
                                      vk::PipelineStageFlagBits2::eTransfer,
                                      cmd);

                cmd.copyBufferToImage(buf->id(), tex->id(), vk::ImageLayout::eTransferDstOptimal, region);

                transitionImageLayout(tex->id(),
                                      vk::ImageLayout::eTransferDstOptimal,
                                      vk::AccessFlagBits2::eTransferWrite,
                                      vk::PipelineStageFlagBits2::eTransfer,

                                      vk::ImageLayout::eShaderReadOnlyOptimal,
                                      vk::AccessFlagBits2::eShaderRead,
                                      vk::PipelineStageFlagBits2::eFragmentShader,
                                      cmd);
            });

            return tex;
            },
            dispatch_kind::upload);
    }

    std::shared_ptr<texture> reduce_texture(const std::shared_ptr<texture>& source, int levels)
    {
        if (!source || source->compressed())
            return nullptr;

        const int n = std::clamp(levels, 0, 8);

        std::vector<std::pair<int, int>> chain;
        int                              w = source->width();
        int                              h = source->height();
        for (int i = 0; i < n; ++i) {
            const int nw = std::max(1, w / 2);
            const int nh = std::max(1, h / 2);
            if (nw == w && nh == h)
                break; // already 1x1
            chain.emplace_back(nw, nh);
            w = nw;
            h = nh;
        }
        // Always at least one pass, so the result is an 8-bit 4-component image
        // whatever the source depth -- the documented contract of the caller.
        if (chain.empty())
            chain.emplace_back(w, h);

        return dispatch_sync([&]() -> std::shared_ptr<texture> {
            try {
                auto cur   = source;
                int  cur_w = source->width();
                int  cur_h = source->height();
                // The mixer's attachment arrives in eColorAttachmentOptimal; every
                // intermediate after it is left in eTransferDstOptimal by its blit.
                auto cur_layout = vk::ImageLayout::eColorAttachmentOptimal;
                auto cur_access = vk::AccessFlagBits2::eColorAttachmentWrite;
                auto cur_stage  = vk::PipelineStageFlagBits2::eColorAttachmentOutput;

                std::shared_ptr<texture> dst;
                for (auto [nw, nh] : chain) {
                    // create_attachment, not create_texture: a pooled texture is
                    // only eTransferDst|eSampled, and this has to be blitted from
                    // and then read back, both of which need eTransferSrc.
                    dst = create_attachment(nw, nh, common::bit_depth::bit8, 4, common::render_format::unorm);
                    if (!dst)
                        return nullptr;

                    const auto src_img = cur->id();
                    const auto dst_img = dst->id();
                    const int  sw = cur_w, sh = cur_h;

                    submitSingleTimeCommands([&](vk::CommandBuffer cmd) {
                        transitionImageLayout(src_img, cur_layout, cur_access, cur_stage,
                                              vk::ImageLayout::eTransferSrcOptimal,
                                              vk::AccessFlagBits2::eTransferRead,
                                              vk::PipelineStageFlagBits2::eTransfer, cmd);
                        // create_attachment hands it back in eRenderingLocalRead;
                        // it is overwritten whole, so eUndefined is honest and skips
                        // preserving contents.
                        transitionImageLayout(dst_img, vk::ImageLayout::eUndefined,
                                              vk::AccessFlagBits2::eNone,
                                              vk::PipelineStageFlagBits2::eTopOfPipe,
                                              vk::ImageLayout::eTransferDstOptimal,
                                              vk::AccessFlagBits2::eTransferWrite,
                                              vk::PipelineStageFlagBits2::eTransfer, cmd);

                        const auto layers = vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, 0, 1);
                        vk::ImageBlit2 region{};
                        region.srcSubresource = layers;
                        region.dstSubresource = layers;
                        region.srcOffsets[0]  = vk::Offset3D{0, 0, 0};
                        region.srcOffsets[1]  = vk::Offset3D{sw, sh, 1};
                        region.dstOffsets[0]  = vk::Offset3D{0, 0, 0};
                        region.dstOffsets[1]  = vk::Offset3D{nw, nh, 1};

                        vk::BlitImageInfo2 blit{};
                        blit.srcImage       = src_img;
                        blit.srcImageLayout = vk::ImageLayout::eTransferSrcOptimal;
                        blit.dstImage       = dst_img;
                        blit.dstImageLayout = vk::ImageLayout::eTransferDstOptimal;
                        blit.filter         = vk::Filter::eLinear;
                        blit.setRegions(region);
                        cmd.blitImage2(blit);
                    });

                    cur        = dst;
                    cur_w      = nw;
                    cur_h      = nh;
                    cur_layout = vk::ImageLayout::eTransferDstOptimal;
                    cur_access = vk::AccessFlagBits2::eTransferWrite;
                    cur_stage  = vk::PipelineStageFlagBits2::eTransfer;
                }

                // Hand it back in the layout copy_async() assumes, so the caller can
                // chain the two without knowing anything about layouts.
                submitSingleTimeCommands([&](vk::CommandBuffer cmd) {
                    transitionImageLayout(cur->id(), cur_layout, cur_access, cur_stage,
                                          vk::ImageLayout::eColorAttachmentOptimal,
                                          vk::AccessFlagBits2::eColorAttachmentWrite,
                                          vk::PipelineStageFlagBits2::eColorAttachmentOutput, cmd);
                });

                return cur;
            } catch (const vk::SystemError& e) {
                CASPAR_LOG(warning) << L"[vulkan::device] reduce_texture failed: " << u16(e.what());
                return nullptr;
            }
        });
    }

    std::future<array<const uint8_t>> copy_async(const std::shared_ptr<texture>& source)
    {
        // A block-compressed image cannot be read back this way. create_buffer below
        // sizes the staging buffer from source->size() (width*height*stride), but
        // copyImageToBuffer writes only the BC blocks -- an eighth of that for BC1 --
        // and leaves the remainder untouched. The caller then gets a buffer of exactly
        // the length it expected whose tail is zeroes and whose head is raw block data
        // read as pixels, which is indistinguishable from a successful readback. That
        // is how PRINT RAW on a native-HAP frame produced a frame 88% black instead of
        // failing. Owners of compressed textures must supply their own host decode
        // (see hap_producer's wrapper); there is nothing sensible to return here.
        if (source && source->compressed()) {
            CASPAR_LOG(warning) << L"vulkan::device::copy_async: refusing to read back a "
                                   L"block-compressed texture -- the caller must decode on the host";
            return std::async(std::launch::deferred, [] { return array<const uint8_t>(); });
        }

        auto f = dispatch_async(
            [this, source]() -> std::pair<std::shared_ptr<buffer>, uint64_t> {
            auto buf = create_buffer(source->size(), false);

            vk::CopyImageToBufferInfo2 copyInfo{};
            copyInfo.dstBuffer      = buf->id();
            copyInfo.srcImage       = source->id();
            copyInfo.srcImageLayout = vk::ImageLayout::eTransferSrcOptimal;

            vk::BufferImageCopy2 region{};
            region.bufferOffset     = 0;
            region.imageSubresource = vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, 0, 1);
            region.imageOffset      = vk::Offset3D{0, 0, 0};
            region.imageExtent =
                vk::Extent3D{static_cast<uint32_t>(source->width()), static_cast<uint32_t>(source->height()), 1};
            copyInfo.setRegions(region);

            auto signal_value = submitSingleTimeCommands([&](vk::CommandBuffer cmd) {
                transitionImageLayout(source->id(),
                                      vk::ImageLayout::eColorAttachmentOptimal,
                                      vk::AccessFlagBits2::eColorAttachmentWrite,
                                      vk::PipelineStageFlagBits2::eColorAttachmentOutput,

                                      vk::ImageLayout::eTransferSrcOptimal,
                                      vk::AccessFlagBits2::eTransferRead,
                                      vk::PipelineStageFlagBits2::eTransfer,
                                      cmd);
                cmd.copyImageToBuffer2(copyInfo);

                // Make the copy's writes visible to the host domain. Required
                // (not just the invalidate below) whenever the readback buffer's
                // memory type is HOST_VISIBLE but not HOST_COHERENT.
                vk::BufferMemoryBarrier2 hostBarrier{};
                hostBarrier.srcStageMask       = vk::PipelineStageFlagBits2::eTransfer;
                hostBarrier.srcAccessMask      = vk::AccessFlagBits2::eTransferWrite;
                hostBarrier.dstStageMask       = vk::PipelineStageFlagBits2::eHost;
                hostBarrier.dstAccessMask      = vk::AccessFlagBits2::eHostRead;
                hostBarrier.srcQueueFamilyIndex = vk::QueueFamilyIgnored;
                hostBarrier.dstQueueFamilyIndex = vk::QueueFamilyIgnored;
                hostBarrier.buffer              = buf->id();
                hostBarrier.offset              = 0;
                hostBarrier.size                = VK_WHOLE_SIZE;

                vk::DependencyInfo hostDepInfo{};
                hostDepInfo.setBufferMemoryBarriers(hostBarrier);
                cmd.pipelineBarrier2(hostDepInfo);
            });

            return {buf, signal_value};
            },
            dispatch_kind::readback);

        return std::async(std::launch::deferred, [this, f = std::move(f)]() mutable {
            auto [buf, signal_value] = f.get();
            vk::SemaphoreWaitInfo waitInfo{};
            waitInfo.setSemaphores(_semaphore);
            waitInfo.setValues(signal_value);
            auto res = _device.waitSemaphores(waitInfo, 1000000000);
            if (res != vk::Result::eSuccess) {
                CASPAR_LOG(warning) << L"[Vulkan] Timeout waiting for readback semaphore";
            }

            // Invalidate CPU caches in case the allocator picked a HOST_VISIBLE
            // but non-coherent memory type for this readback buffer — otherwise
            // the CPU can read stale cache lines instead of the GPU's writes.
            buf->invalidate();

            auto ptr  = reinterpret_cast<uint8_t*>(buf->data());
            auto size = buf->size();
            return array<const uint8_t>(ptr, size, std::move(buf));
        });
    }

    boost::property_tree::wptree info() const
    {
        boost::property_tree::wptree info;

        boost::property_tree::wptree pooled_device_buffers;
        size_t                       total_pooled_device_buffer_size  = 0;
        size_t                       total_pooled_device_buffer_count = 0;

        for (size_t i = 0; i < device_pools_.size(); ++i) {
            auto& depth_pools = device_pools_.at(i);
            for (size_t j = 0; j < depth_pools.size(); ++j) {
                auto& pools      = depth_pools.at(j);
                bool  mipmapping = j > 3;
                auto  stride     = mipmapping ? j - 3 : j + 1;

                for (auto& pool : pools) {
                    auto width  = pool.first >> 16;
                    auto height = pool.first & 0x0000FFFF;
                    auto size   = width * height * stride;
                    auto count  = pool.second.size();

                    if (count == 0)
                        continue;

                    boost::property_tree::wptree pool_info;

                    pool_info.add(L"stride", stride);
                    pool_info.add(L"mipmapping", mipmapping);
                    pool_info.add(L"width", width);
                    pool_info.add(L"height", height);
                    pool_info.add(L"size", size);
                    pool_info.add(L"count", count);

                    total_pooled_device_buffer_size += size * count;
                    total_pooled_device_buffer_count += count;

                    pooled_device_buffers.add_child(L"device_buffer_pool", pool_info);
                }
            }
        }

        info.add_child(L"gl.details.pooled_device_buffers", pooled_device_buffers);

        boost::property_tree::wptree pooled_host_buffers;
        size_t                       total_read_size   = 0;
        size_t                       total_write_size  = 0;
        size_t                       total_read_count  = 0;
        size_t                       total_write_count = 0;

        for (size_t i = 0; i < host_pools_.size(); ++i) {
            auto& pools    = host_pools_.at(i);
            auto  is_write = i == 1;

            for (auto& pool : pools) {
                auto size  = pool.first;
                auto count = pool.second.size();

                if (count == 0)
                    continue;

                boost::property_tree::wptree pool_info;

                pool_info.add(L"usage", is_write ? L"write_only" : L"read_only");
                pool_info.add(L"size", size);
                pool_info.add(L"count", count);

                pooled_host_buffers.add_child(L"host_buffer_pool", pool_info);

                (is_write ? total_write_count : total_read_count) += count;
                (is_write ? total_write_size : total_read_size) += size * count;
            }
        }

        info.add_child(L"gl.details.pooled_host_buffers", pooled_host_buffers);
        info.add(L"gl.summary.pooled_device_buffers.total_count", total_pooled_device_buffer_count);
        info.add(L"gl.summary.pooled_device_buffers.total_size", total_pooled_device_buffer_size);
        // info.add_child(L"gl.summary.all_device_buffers", texture::info());
        info.add(L"gl.summary.pooled_host_buffers.total_read_count", total_read_count);
        info.add(L"gl.summary.pooled_host_buffers.total_write_count", total_write_count);
        info.add(L"gl.summary.pooled_host_buffers.total_read_size", total_read_size);
        info.add(L"gl.summary.pooled_host_buffers.total_write_size", total_write_size);
        info.add_child(L"gl.summary.all_host_buffers", buffer::info());

        publish_dispatch_stats(info);

        return info;
    }

    std::future<void> gc()
    {
        return spawn_async([this](yield_context yield) {
            CASPAR_LOG(info) << " vulkan: Running GC.";

            try {
                for (auto& depth_pools : device_pools_) {
                    for (auto& pools : depth_pools) {
                        for (auto& pool : pools)
                            pool.second.clear();
                    }
                }
                for (auto& pools : host_pools_) {
                    for (auto& pool : pools)
                        pool.second.clear();
                }
                for (auto& pools : attachment_pools_) {
                    for (auto& pool : pools)
                        pool.second.clear();
                }
            } catch (...) {
                CASPAR_LOG_CURRENT_EXCEPTION();
            }
        });
    }
};

device::device(int gpu_index)
    : impl_(new impl(gpu_index))
{
}
device::~device() {}

vk::PhysicalDeviceMemoryProperties device::getMemoryProperties() { return impl_->_memoryProperties; }

// ---- for handing this device to another API -------------------------------------------
//
// FFmpeg 8 can decode ProRes, FFV1 and DPX with Vulkan COMPUTE shaders, and the way to
// get those frames without a copy is to let it use the device the mixer already owns --
// `AVVulkanDeviceContext` is designed to be filled in by the application rather than
// created by FFmpeg. It needs more than the device handle: the instance, the queue
// family, and the loader entry point.
//
// Measured 2026-08-20 against `ffmpeg -init_hw_device vulkan`, which builds its OWN device
// on this GPU with 5 queue families (graphics/transfer/compute/decode/encode) and 23
// extensions. This device has ONE graphics family. That is not necessarily a problem for
// the compute codecs -- `prores_vulkan` declares `VK_QUEUE_COMPUTE_BIT` and NVIDIA's
// graphics family carries COMPUTE -- but it is why the Vulkan VIDEO codecs (h264/hevc/
// av1/vp9, which need VK_KHR_video_decode_queue and a decode family) are a separate
// question from the compute ones.
/// A queue another API may submit on without racing the mixer. See the note at its
/// creation for why this exists rather than a lock. Null when `hasDedicatedDecodeQueue()`
/// is false, in which case the family index equals the graphics one.
vk::Queue device::getDecodeQueue() const { return impl_->_decode_queue; }

uint32_t device::getDecodeQueueFamily() const { return impl_->_decode_queue_family; }

bool device::hasDedicatedDecodeQueue() const { return impl_->_decode_queue_dedicated; }

vk::Instance device::getVkInstance() const { return vk::Instance(impl_->_vkb_instance.instance); }

uint32_t device::getGraphicsQueueFamily() const { return impl_->_queue_family; }

PFN_vkGetInstanceProcAddr device::getInstanceProcAddr() const
{
    return impl_->_vkb_instance.fp_vkGetInstanceProcAddr;
}
std::vector<vk::CommandBuffer>     device::allocateCommandBuffers(uint32_t count)
{
    return impl_->allocateCommandBuffers(count);
}
void       device::submit(const vk::SubmitInfo& submitInfo, vk::Fence fence) { impl_->submit(submitInfo, fence); }
vk::Device         device::getVkDevice() const { return impl_->_device; }
vk::PhysicalDevice device::getVkPhysicalDevice() const { return impl_->_physical_device; }
vk::CommandPool    device::getCommandPool() const { return impl_->_command_pool; }
std::shared_ptr<pipeline> device::get_pipeline(common::bit_depth depth, common::render_format render_format)
{
    return impl_->_pipelines[impl::attachment_pool_index(depth, render_format)];
}

std::shared_ptr<pipeline> device::get_variant_pipeline(common::bit_depth            depth,
                                                      common::render_format        render_format,
                                                      const std::string&           variant_id,
                                                      const std::vector<uint32_t>& frag_spirv)
{
    // An empty id means the base program; do not build a second copy of it.
    if (variant_id.empty() || frag_spirv.empty())
        return get_pipeline(depth, render_format);

    const auto format_index = impl::attachment_pool_index(depth, render_format);
    const auto key          = std::make_pair(variant_id, format_index);

    std::lock_guard<std::mutex> lock(impl_->_variant_mutex);

    auto it = impl_->_variant_pipelines.find(key);
    if (it != impl_->_variant_pipelines.end())
        return it->second;

    // Built on the calling thread. Callers must not reach this from the frame path -- see
    // the note on the OGL shader cache; the same applies here and more so, because a Vulkan
    // pipeline build is the driver's SPIR-V-to-ISA step on top of the shaderc compile.
    auto built = std::make_shared<pipeline>(impl_->_device,
                                            impl::attachment_format(depth, render_format),
                                            impl_->_memoryProperties,
                                            frag_spirv);
    impl_->_variant_pipelines.emplace(key, built);

    CASPAR_LOG(info) << L"[vulkan::device] built a variant pipeline for '" << u16(variant_id)
                     << L"' at format index " << format_index << L" (" << impl_->_variant_pipelines.size()
                     << L" cached)";
    return built;
}

std::shared_ptr<texture> device::create_attachment(int                   width,
                                                   int                   height,
                                                   common::bit_depth     depth,
                                                   uint32_t              components_count,
                                                   common::render_format render_format)
{
    return impl_->create_attachment(width, height, depth, components_count, render_format);
}

void device::reset_attachment_layout(const std::shared_ptr<class texture>& tex)
{
    impl_->submitSingleTimeCommands([&](vk::CommandBuffer cmd) {
        transitionImageLayout(
            tex->id(),
            vk::ImageLayout::eUndefined,
            vk::AccessFlagBits2::eNone,
            vk::PipelineStageFlagBits2::eTopOfPipe,
            vk::ImageLayout::eRenderingLocalRead,
            vk::AccessFlagBits2::eColorAttachmentWrite | vk::AccessFlagBits2::eInputAttachmentRead,
            vk::PipelineStageFlagBits2::eColorAttachmentOutput | vk::PipelineStageFlagBits2::eFragmentShader,
            cmd);
    });
}

std::shared_ptr<texture> device::create_texture(int width, int height, int stride, common::bit_depth depth)
{
    return impl_->create_texture(width, height, stride, depth, true);
}

bool device::can_sample_packed(int stride, common::bit_depth depth) const
{
    return impl_->can_sample_packed(stride, depth);
}

std::shared_ptr<texture>
device::create_exportable_texture(int width, int height, int stride, common::bit_depth depth)
{
    return dispatch_sync([&]() -> std::shared_ptr<texture> {
        CASPAR_VERIFY(stride > 0 && stride < 5);
        CASPAR_VERIFY(width > 0 && height > 0);

        static vk::Format INTERNAL_FORMAT[][5] = {{vk::Format::eUndefined,
                                                    vk::Format::eR8Unorm,
                                                    vk::Format::eR8G8Unorm,
                                                    vk::Format::eR8G8B8Unorm,
                                                    vk::Format::eR8G8B8A8Unorm},
                                                   {vk::Format::eUndefined,
                                                    vk::Format::eR16Unorm,
                                                    vk::Format::eR16G16Unorm,
                                                    vk::Format::eR16G16B16Unorm,
                                                    vk::Format::eR16G16B16A16Unorm}};

        auto depth_pool_index = depth == common::bit_depth::bit8 ? 0 : 1;
        auto format           = INTERNAL_FORMAT[depth_pool_index][stride];
        auto extent = vk::Extent3D{static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1};
        auto dev    = impl_->_device;

        // External memory export info chained into image create
        vk::ExternalMemoryImageCreateInfo extMemImageInfo{};
        extMemImageInfo.handleTypes = static_cast<vk::ExternalMemoryHandleTypeFlagBits>(platform::kExternalMemoryHandleType);

        vk::ImageCreateInfo imageInfo{};
        imageInfo.pNext         = &extMemImageInfo;
        imageInfo.imageType     = vk::ImageType::e2D;
        imageInfo.format        = format;
        imageInfo.extent        = extent;
        imageInfo.mipLevels     = 1;
        imageInfo.arrayLayers   = 1;
        imageInfo.initialLayout = vk::ImageLayout::eUndefined;
        imageInfo.samples       = vk::SampleCountFlagBits::e1;
        imageInfo.tiling        = vk::ImageTiling::eOptimal;
        // eColorAttachment is what lets another API render *into* this image
        // rather than only sample it -- the GL -> Vulkan route imports the
        // memory and attaches the GL texture to a framebuffer. Proved on the
        // reference GPU: RGBA8 + eColorAttachment exports, imports into GL,
        // renders, and reads back byte-identically at 1080p (eOptimal tiling;
        // eLinear is rejected by GL as "memory object too small").
        imageInfo.usage = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled |
                          vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eColorAttachment;
        imageInfo.sharingMode = vk::SharingMode::eExclusive;
        auto image            = dev.createImage(imageInfo);

        auto memReq = dev.getImageMemoryRequirements(image);

        // Export memory allocate info
        vk::ExportMemoryAllocateInfo exportMemInfo{};
        exportMemInfo.handleTypes = static_cast<vk::ExternalMemoryHandleTypeFlagBits>(platform::kExternalMemoryHandleType);

        vk::MemoryAllocateInfo allocInfo{};
        allocInfo.pNext           = &exportMemInfo;
        allocInfo.allocationSize  = memReq.size;
        allocInfo.memoryTypeIndex = impl_->findDedicatedMemoryType(
            memReq.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);

        auto imageMemory = dev.allocateMemory(allocInfo);
        dev.bindImageMemory(image, imageMemory, 0);

        auto range = vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);
        vk::ImageViewCreateInfo createInfo(
            {}, image, vk::ImageViewType::e2D, format, vk::ComponentMapping(), range);
        auto imageView = dev.createImageView(createInfo);

        return std::make_shared<texture>(width, height, stride, depth, image, imageMemory, imageView, dev, memReq.size);
    });
}

array<uint8_t> device::create_array(int size) { return impl_->create_array(size); }
std::future<std::shared_ptr<texture>>
device::copy_async(const array<const uint8_t>& source, int width, int height, int stride, common::bit_depth depth)
{
    return impl_->copy_async(source, width, height, stride, depth);
}
std::future<std::shared_ptr<texture>>
device::copy_compressed_async(const array<const uint8_t>& source, int width, int height, vk::Format format)
{
    return impl_->copy_compressed_async(source, width, height, format);
}
std::future<array<const uint8_t>> device::copy_async(const std::shared_ptr<texture>& source)
{
    return impl_->copy_async(source);
}
std::shared_ptr<texture> device::reduce_texture(const std::shared_ptr<texture>& source, int levels)
{
    return impl_->reduce_texture(source, levels);
}
void device::dispatch(std::function<void()> func)
{
    boost::asio::dispatch(impl_->io_context_, impl_->instrument(std::move(func)));
}
std::wstring                 device::version() const { return impl_->version(); }
boost::property_tree::wptree device::info() const { return impl_->info(); }
std::future<void>            device::gc() { return impl_->gc(); }
}}} // namespace caspar::accelerator::vulkan
