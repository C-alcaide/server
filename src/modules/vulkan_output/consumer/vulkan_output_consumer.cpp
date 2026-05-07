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
 */

#include "vulkan_output_consumer.h"
#include "config.h"
#include "../util/vulkan_device.h"
#include "../util/vulkan_interop.h"
#include "../util/shared_texture_pool.h"
#include "../util/gpu_affinity_context.h"
#ifdef CASPAR_CUDA_PEER_ENABLED
#include "../util/cuda_peer_transfer.h"
#endif
#include "../util/nvapi_helpers.h"
#include "../util/color_convert_pipeline.h"

#include <accelerator/ogl/image/image_mixer.h>
#include <accelerator/ogl/util/device.h>
#include <accelerator/ogl/util/texture.h>

#include <common/diagnostics/graph.h>
#include <common/except.h>
#include <common/executor.h>
#include <common/future.h>
#include <common/log.h>
#include <common/memory.h>
#include <common/timer.h>

#include <core/consumer/channel_info.h>
#include <core/consumer/frame_consumer.h>
#include <core/frame/frame.h>
#include <core/video_channel.h>
#include <core/video_format.h>

#include <boost/algorithm/string.hpp>

#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>

namespace caspar { namespace vulkan_output {

namespace {

// ─── Swapchain management ───────────────────────────────────────────────

struct swapchain_resources
{
    VkSwapchainKHR           swapchain   = VK_NULL_HANDLE;
    VkSurfaceKHR             surface     = VK_NULL_HANDLE;
    std::vector<VkImage>     images;
    std::vector<VkImageView> image_views;
    VkSemaphore              image_available = VK_NULL_HANDLE;
    VkSemaphore              render_finished = VK_NULL_HANDLE;
    VkFence                  in_flight       = VK_NULL_HANDLE;
    VkCommandPool            cmd_pool        = VK_NULL_HANDLE;
    VkCommandBuffer          cmd_buffer      = VK_NULL_HANDLE;
    uint32_t                 width           = 0;
    uint32_t                 height          = 0;
};

// ─── Consumer implementation ────────────────────────────────────────────

class vulkan_output_consumer : public core::frame_consumer
{
  public:
    vulkan_output_consumer(configuration config, std::shared_ptr<accelerator::ogl::device> ogl_device)
        : config_(std::move(config))
        , ogl_device_(std::move(ogl_device))
        , executor_(L"vulkan_output")
    {
    }

    ~vulkan_output_consumer() override
    {
        executor_.invoke([this] { destroy_resources(); });
    }

    void initialize(const core::video_format_desc& format_desc,
                    const core::channel_info&      channel_info,
                    int                            port_index) override
    {
        format_desc_ = format_desc;
        port_index_  = port_index;

        graph_->set_text(print());
        graph_->set_color("frame-time", diagnostics::color(0.1f, 1.0f, 0.1f));
        graph_->set_color("dropped-frame", diagnostics::color(0.9f, 0.2f, 0.2f));
        graph_->set_color("late-frame", diagnostics::color(0.6f, 0.3f, 0.9f));
        graph_->set_color("buffered-video", diagnostics::color(0.2f, 0.9f, 0.9f));
        graph_->set_color("tick-time", diagnostics::color(0.0f, 0.6f, 0.9f));
        diagnostics::register_graph(graph_);

        executor_.invoke([this] { init_vulkan(); });
    }

    std::future<bool> send(const core::video_field field, core::const_frame frame) override
    {
        // Skip field B for interlaced - same pattern as screen consumer
        if (field == core::video_field::b)
            return caspar::make_ready_future(true);

        // Buffer the frame
        {
            std::unique_lock<std::mutex> lock(buffer_mutex_);
            if (buffer_.size() >= static_cast<size_t>(config_.buffer_depth)) {
                graph_->set_tag(diagnostics::tag_severity::WARNING, "late-frame");
                buffer_.pop(); // Drop oldest
            }
            buffer_.push(std::move(frame));
        }
        buffer_cv_.notify_one();

        return caspar::make_ready_future(true);
    }

    std::wstring print() const override
    {
        return L"vulkan_output[gpu:" + std::to_wstring(config_.gpu_index) + L"|output:" +
               std::to_wstring(config_.output_index) + L"]";
    }

    std::wstring name() const override { return L"vulkan-output"; }

    int index() const override
    {
        return 500 + config_.gpu_index * 10 + config_.output_index;
    }

    bool has_synchronization_clock() const override { return true; }

    std::future<bool> call(const std::vector<std::wstring>& params) override
    {
        if (params.empty())
            return caspar::make_ready_future(false);

        if (boost::iequals(params[0], L"IDENTIFY")) {
            identify_frames_remaining_ = static_cast<int>(format_desc_.fps * 3);
            CASPAR_LOG(info) << print() << L" IDENTIFY triggered.";
            return caspar::make_ready_future(true);
        }

        CASPAR_LOG(warning) << print() << L" Unknown CALL command: " << params[0];
        return caspar::make_ready_future(false);
    }

    core::monitor::state state() const override
    {
        core::monitor::state s;
        std::lock_guard<std::mutex> cfg_lock(config_mutex_);
        s["vulkan-output/gpu"]    = std::to_wstring(config_.gpu_index);
        s["vulkan-output/output"] = std::to_wstring(config_.output_index);
        s["vulkan-output/tier"]   = device_ ? (device_->tier() == gpu_tier::pro ? std::wstring(L"pro")
                                                                                 : std::wstring(L"consumer"))
                                            : std::wstring(L"none");
        s["vulkan-output/frames"] = std::to_wstring(frames_presented_.load());
        s["vulkan-output/display-lost"] = std::wstring(display_lost_.load() ? L"true" : L"false");
        s["vulkan-output/sync-group"]   = std::to_wstring(config_.sync_group);
        s["vulkan-output/present-barrier"] = std::wstring(present_barrier_enabled_ ? L"true" : L"false");
        s["vulkan-output/delay"]           = std::to_wstring(config_.delay_frames);

        // NvAPI Quadro Sync status
        if (nvapi_ && nvapi_->is_available() && config_.gsync_enabled) {
            auto sync_st = nvapi_->get_sync_status(config_.gpu_index);
            s["vulkan-output/gsync/available"]  = std::wstring(sync_st.available ? L"true" : L"false");
            s["vulkan-output/gsync/synced"]     = std::wstring(sync_st.synced ? L"true" : L"false");
            s["vulkan-output/gsync/house-sync"] = std::wstring(sync_st.house_sync ? L"true" : L"false");
            s["vulkan-output/gsync/role"]       = sync_st.role == sync_role::master ? std::wstring(L"master")
                                                  : sync_st.role == sync_role::slave ? std::wstring(L"slave")
                                                                                     : std::wstring(L"none");
        }

        return s;
    }

  private:
    void init_vulkan()
    {
        device_ = std::make_unique<vulkan_device>(config_.gpu_index, config_.output_index);

        // Find our target display using the main device's instance (not a temporary one)
        // to ensure the VkDisplayKHR handles are valid for our device's lifetime
        display_info target{};
        bool         found = false;

        if (device_->tier() == gpu_tier::pro) {
            auto displays = device_->enumerate_displays_on_device();
            for (const auto& d : displays) {
                if (d.output_index == config_.output_index) {
                    target = d;
                    target.gpu_index = config_.gpu_index;
                    found  = true;
                    break;
                }
            }
        }

        if (!found && device_->tier() == gpu_tier::consumer) {
            // Consumer fallback: create a borderless fullscreen window
            create_fse_window();
        } else if (!found) {
            CASPAR_THROW_EXCEPTION(caspar_exception()
                                   << msg_info("Vulkan output not found: gpu=" + std::to_string(config_.gpu_index) +
                                               " output=" + std::to_string(config_.output_index)));
        }

        // Create surface
        if (device_->tier() == gpu_tier::pro && found) {
            // Convert fps to millihertz for display mode matching
            uint32_t refresh_mhz = static_cast<uint32_t>(format_desc_.fps * 1000.0 + 0.5);
            swapchain_.surface = device_->create_display_surface(target, refresh_mhz);
            display_handle_ = target.display_handle;
        } else if (fse_hwnd_) {
            swapchain_.surface = device_->create_win32_surface(fse_hwnd_);
        }

        setup_present_barrier(); // Must be before create_swapchain so the barrier struct is chained
        create_swapchain();
        set_hdr_metadata();
        create_sync_objects();
        setup_vblank_timing();
        setup_nvapi();
        create_command_pool();

        // Create shared texture pool for zero-copy if OGL device is available
        if (ogl_device_) {
            // Verify GPU match: OGL and VK must be on the same physical device for interop
            bool gpu_match = true;
            if (device_->device_luid_valid()) {
                uint8_t ogl_luid[8] = {};
                bool    ogl_luid_valid = false;
                ogl_device_->dispatch_sync([&] {
                    // GL_EXT_memory_object exposes GL_DEVICE_LUID_EXT
                    auto glGetUnsignedBytevEXT = reinterpret_cast<void(APIENTRY*)(GLenum, GLubyte*)>(
                        wglGetProcAddress("glGetUnsignedBytevEXT"));
                    if (glGetUnsignedBytevEXT) {
                        glGetUnsignedBytevEXT(0x9462 /*GL_DEVICE_LUID_EXT*/, ogl_luid);
                        ogl_luid_valid = true;
                    }
                });
                if (ogl_luid_valid && memcmp(ogl_luid, device_->device_luid(), 8) != 0) {
                    gpu_match = false;
                    CASPAR_LOG(info) << print()
                        << L" OGL context is on a different GPU than Vulkan output (gpu_index="
                        << device_->gpu_index() << L"). Creating affinity bridge...";
                }
            }

            if (gpu_match) {
                // Same GPU: direct zero-copy from mixer OGL texture → shared VK texture
                try {
                    bool use_16bit = (config_.transfer != hdr_transfer::sdr);
                    shared_pool_ = std::make_unique<shared_texture_pool>(
                        ogl_device_, *device_, format_desc_.width, format_desc_.height, use_16bit);
                    CASPAR_LOG(info) << print() << L" Zero-copy OGL→VK interop enabled (same GPU)"
                                     << (use_16bit ? L" (16-bit for HDR)." : L".");
                } catch (const std::exception& e) {
                    CASPAR_LOG(warning) << print() << L" Zero-copy unavailable, falling back to CPU: " << e.what();
                    shared_pool_.reset();
                }
            } else {
                // Different GPU: create affinity OGL context on target GPU, then interop from there
                try {
                    affinity_ctx_ = std::make_unique<gpu_affinity_context>(
                        device_->gpu_index(), format_desc_.width, format_desc_.height);

                    // Verify the affinity context's LUID matches the Vulkan device
                    if (affinity_ctx_->device_luid_valid() && device_->device_luid_valid()) {
                        if (memcmp(affinity_ctx_->device_luid(), device_->device_luid(), 8) != 0) {
                            CASPAR_LOG(error) << print()
                                << L" Affinity context LUID doesn't match Vulkan device — driver issue?";
                            affinity_ctx_.reset();
                        }
                    }

                    if (affinity_ctx_) {
                        // Create shared_texture_pool using the affinity context as the OGL side
                        bool use_16bit = (config_.transfer != hdr_transfer::sdr);
                        // shared_texture_pool needs an ogl::device-like interface for dispatch_sync.
                        // We'll create it on the affinity context thread directly.
                        affinity_ctx_->dispatch_sync([&] {
                            shared_pool_ = std::make_unique<shared_texture_pool>(
                                *device_, format_desc_.width, format_desc_.height, use_16bit);
                        });

#ifdef CASPAR_CUDA_PEER_ENABLED
                        // Try CUDA peer DMA for direct GPU→GPU transfer (no CPU memcpy)
                        try {
                            int src_cuda_dev = -1;
                            ogl_device_->dispatch_sync([&] {
                                src_cuda_dev = cuda_peer_transfer::cuda_device_for_current_gl_context();
                            });
                            int dst_cuda_dev = -1;
                            affinity_ctx_->dispatch_sync([&] {
                                dst_cuda_dev = cuda_peer_transfer::cuda_device_for_current_gl_context();
                            });

                            if (src_cuda_dev >= 0 && dst_cuda_dev >= 0 && src_cuda_dev != dst_cuda_dev) {
                                cuda_peer_ = std::make_unique<cuda_peer_transfer>(
                                    src_cuda_dev, dst_cuda_dev,
                                    format_desc_.width, format_desc_.height, use_16bit);
                                CASPAR_LOG(info) << print()
                                    << L" CUDA peer DMA enabled (device " << src_cuda_dev
                                    << L" → device " << dst_cuda_dev << L")";
                            }
                        } catch (const std::exception& e) {
                            CASPAR_LOG(warning) << print()
                                << L" CUDA peer transfer unavailable: " << e.what()
                                << L" — using PBO upload fallback.";
                            cuda_peer_.reset();
                        }
#endif // CASPAR_CUDA_PEER_ENABLED

                        CASPAR_LOG(info) << print()
                            << L" Cross-GPU interop enabled via affinity bridge (GPU "
                            << device_->gpu_index() << L")"
                            << (use_16bit ? L" (16-bit for HDR)." : L".");
                    }
                } catch (const std::exception& e) {
                    CASPAR_LOG(warning) << print()
                        << L" GPU affinity context failed: " << e.what()
                        << L" — falling back to CPU upload (SDR only).";
                    affinity_ctx_.reset();
                    shared_pool_.reset();
                }
            }
        }

        // Create color space conversion pipeline if needed (before present thread starts
        // to avoid a data race — present_loop reads color_pipeline_ every frame)
        if (config_.gamut != output_gamut::bt709 || config_.eotf != output_eotf::srgb) {
            try {
                color_pipeline_ = std::make_unique<color_convert_pipeline>(
                    *device_, format_desc_.width, format_desc_.height);
                color_pipeline_->update_config(config_.gamut, config_.eotf,
                                               static_cast<float>(config_.max_cll));
                CASPAR_LOG(info) << print() << L" Color space conversion enabled.";
            } catch (const std::exception& e) {
                CASPAR_LOG(error) << print()
                    << L" Failed to create color conversion pipeline: " << e.what();
                color_pipeline_.reset();
            }
        }

        // Start present thread
        running_ = true;
        present_thread_ = std::thread([this] { present_loop(); });

        // Show identify overlay if configured
        if (config_.identify_on_start) {
            identify_frames_remaining_ = static_cast<int>(format_desc_.fps * 3); // 3 seconds
        }

        CASPAR_LOG(info) << print() << L" initialized. Tier: "
                         << (device_->tier() == gpu_tier::pro ? L"Pro (direct display)" : L"Consumer (FSE)")
                         << (config_.delay_frames > 0 ? L" Delay: " + std::to_wstring(config_.delay_frames) + L" frames" : L"");
    }

    void create_swapchain()
    {
        VkSurfaceCapabilitiesKHR caps;
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device_->physical_device(), swapchain_.surface, &caps);

        swapchain_.width  = (caps.currentExtent.width != UINT32_MAX)
                                ? caps.currentExtent.width
                                : format_desc_.width;
        swapchain_.height = (caps.currentExtent.height != UINT32_MAX)
                                ? caps.currentExtent.height
                                : format_desc_.height;

        uint32_t image_count = (std::max)(caps.minImageCount + 1,
                                           static_cast<uint32_t>(config_.buffer_depth));
        if (caps.maxImageCount > 0)
            image_count = (std::min)(image_count, caps.maxImageCount);

        // Pick surface format (BGRA8 for SDR, A2B10G10R10 or RGBA16F for HDR)
        VkSurfaceFormatKHR surface_format = pick_surface_format();

        VkSwapchainCreateInfoKHR create_info{};
        create_info.sType            = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        create_info.surface          = swapchain_.surface;
        create_info.minImageCount    = image_count;
        create_info.imageFormat      = surface_format.format;
        create_info.imageColorSpace  = surface_format.colorSpace;
        create_info.imageExtent      = {swapchain_.width, swapchain_.height};
        create_info.imageArrayLayers = 1;
        create_info.imageUsage       = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        create_info.preTransform     = caps.currentTransform;
        create_info.compositeAlpha   = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        create_info.presentMode      = VK_PRESENT_MODE_FIFO_KHR; // VSync
        create_info.clipped          = VK_TRUE;
        create_info.oldSwapchain     = VK_NULL_HANDLE;

        // Chain VK_NV_present_barrier if enabled for sync group
        VkSwapchainPresentBarrierCreateInfoNV barrier_info{};
        if (present_barrier_enabled_) {
            barrier_info.sType                 = VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_BARRIER_CREATE_INFO_NV;
            barrier_info.pNext                 = nullptr;
            barrier_info.presentBarrierEnable  = VK_TRUE;
            create_info.pNext                  = &barrier_info;
        }

        auto result = vkCreateSwapchainKHR(device_->device(), &create_info, nullptr, &swapchain_.swapchain);
        if (result != VK_SUCCESS)
            CASPAR_THROW_EXCEPTION(caspar_exception() << msg_info("Failed to create Vulkan swapchain"));

        // Get swapchain images
        uint32_t sc_image_count = 0;
        vkGetSwapchainImagesKHR(device_->device(), swapchain_.swapchain, &sc_image_count, nullptr);
        swapchain_.images.resize(sc_image_count);
        vkGetSwapchainImagesKHR(device_->device(), swapchain_.swapchain, &sc_image_count, swapchain_.images.data());

        // Create image views
        swapchain_.image_views.resize(sc_image_count);
        for (uint32_t i = 0; i < sc_image_count; ++i) {
            VkImageViewCreateInfo view_info{};
            view_info.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            view_info.image                           = swapchain_.images[i];
            view_info.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
            view_info.format                          = surface_format.format;
            view_info.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
            view_info.subresourceRange.baseMipLevel   = 0;
            view_info.subresourceRange.levelCount     = 1;
            view_info.subresourceRange.baseArrayLayer = 0;
            view_info.subresourceRange.layerCount     = 1;

            vkCreateImageView(device_->device(), &view_info, nullptr, &swapchain_.image_views[i]);
        }

        CASPAR_LOG(info) << print() << L" Swapchain created: " << swapchain_.width << L"x" << swapchain_.height
                         << L" (" << sc_image_count << L" images)";
    }

    VkSurfaceFormatKHR pick_surface_format()
    {
        uint32_t count = 0;
        vkGetPhysicalDeviceSurfaceFormatsKHR(device_->physical_device(), swapchain_.surface, &count, nullptr);
        std::vector<VkSurfaceFormatKHR> formats(count);
        vkGetPhysicalDeviceSurfaceFormatsKHR(device_->physical_device(), swapchain_.surface, &count, formats.data());

        if (config_.transfer == hdr_transfer::pq || config_.transfer == hdr_transfer::hlg) {
            // Prefer 10-bit or 16-bit for HDR
            for (const auto& f : formats) {
                if (f.format == VK_FORMAT_A2B10G10R10_UNORM_PACK32 &&
                    f.colorSpace == VK_COLOR_SPACE_HDR10_ST2084_EXT)
                    return f;
            }
            for (const auto& f : formats) {
                if (f.format == VK_FORMAT_R16G16B16A16_SFLOAT)
                    return f;
            }
        }

        // Prefer BGRA8 SRGB for SDR
        for (const auto& f : formats) {
            if (f.format == VK_FORMAT_B8G8R8A8_UNORM &&
                f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
                return f;
        }

        return formats[0];
    }

    void create_sync_objects()
    {
        VkSemaphoreCreateInfo sem_info{};
        sem_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        vkCreateSemaphore(device_->device(), &sem_info, nullptr, &swapchain_.image_available);
        vkCreateSemaphore(device_->device(), &sem_info, nullptr, &swapchain_.render_finished);

        VkFenceCreateInfo fence_info{};
        fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        vkCreateFence(device_->device(), &fence_info, nullptr, &swapchain_.in_flight);
    }

    void create_command_pool()
    {
        VkCommandPoolCreateInfo pool_info{};
        pool_info.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool_info.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pool_info.queueFamilyIndex = device_->present_queue_family();
        vkCreateCommandPool(device_->device(), &pool_info, nullptr, &swapchain_.cmd_pool);

        VkCommandBufferAllocateInfo alloc_info{};
        alloc_info.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        alloc_info.commandPool        = swapchain_.cmd_pool;
        alloc_info.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc_info.commandBufferCount = 1;
        vkAllocateCommandBuffers(device_->device(), &alloc_info, &swapchain_.cmd_buffer);
    }

    void set_hdr_metadata()
    {
        if (config_.transfer == hdr_transfer::sdr)
            return;

        auto vkSetHdrMetadataEXT = reinterpret_cast<PFN_vkSetHdrMetadataEXT>(
            vkGetDeviceProcAddr(device_->device(), "vkSetHdrMetadataEXT"));
        if (!vkSetHdrMetadataEXT) {
            CASPAR_LOG(warning) << print() << L" VK_EXT_hdr_metadata not available, skipping HDR metadata.";
            return;
        }

        // SMPTE ST 2086 mastering display color primaries (BT.2020 primaries, D65 white)
        VkHdrMetadataEXT hdr{};
        hdr.sType = VK_STRUCTURE_TYPE_HDR_METADATA_EXT;
        // Display primaries (BT.2020)
        hdr.displayPrimaryRed   = {0.708f, 0.292f};
        hdr.displayPrimaryGreen = {0.170f, 0.797f};
        hdr.displayPrimaryBlue  = {0.131f, 0.046f};
        hdr.whitePoint          = {0.3127f, 0.3290f};
        // Luminance range
        hdr.maxLuminance        = static_cast<float>(config_.max_cll);
        hdr.minLuminance        = 0.001f;
        hdr.maxContentLightLevel      = static_cast<float>(config_.max_cll);
        hdr.maxFrameAverageLightLevel = static_cast<float>(config_.max_fall);

        vkSetHdrMetadataEXT(device_->device(), 1, &swapchain_.swapchain, &hdr);

        CASPAR_LOG(info) << print() << L" HDR metadata set: "
                         << (config_.transfer == hdr_transfer::pq ? L"PQ" : L"HLG")
                         << L" MaxCLL=" << config_.max_cll << L" MaxFALL=" << config_.max_fall;
    }

    void setup_present_barrier()
    {
        if (config_.sync_group == 0)
            return;

        // Enable present barrier on swapchain if extension is present
        // VK_NV_present_barrier provides automatic frame-lock across all swapchains
        // in the same barrier group — the driver synchronizes presents.
        present_barrier_enabled_ = device_->has_extension("VK_NV_present_barrier");

        if (present_barrier_enabled_) {
            CASPAR_LOG(info) << print() << L" Present barrier enabled for sync group " << config_.sync_group
                             << L". Presents will be frame-locked by driver.";
        } else {
            CASPAR_LOG(warning) << print() << L" VK_NV_present_barrier not available. "
                                << L"Sync group " << config_.sync_group << L" will use best-effort timing.";
        }
    }

    void setup_vblank_timing()
    {
        if (display_handle_ == VK_NULL_HANDLE)
            return;

        // VK_EXT_display_control: register for VBlank events to measure drift
        vblank_supported_ = device_->has_extension(VK_EXT_DISPLAY_CONTROL_EXTENSION_NAME);
        if (vblank_supported_) {
            CASPAR_LOG(info) << print() << L" VBlank timing via VK_EXT_display_control enabled.";
            graph_->set_color("vblank-drift", diagnostics::color(0.9f, 0.5f, 0.1f));
        }
    }

    void setup_nvapi()
    {
        nvapi_ = std::make_unique<nvapi_helpers>();
        if (!nvapi_->is_available())
            return;

        // EDID auto-detection: read HDR capabilities from the connected display
        if (config_.edid_auto_hdr) {
            auto edid = nvapi_->read_edid(config_.gpu_index, config_.output_index);
            if (edid.supports_hdr && config_.transfer == hdr_transfer::sdr) {
                {
                    std::lock_guard<std::mutex> cfg_lock(config_mutex_);
                    config_.transfer = hdr_transfer::pq;
                    if (edid.max_luminance > 0)
                        config_.max_cll = static_cast<int>(edid.max_luminance);
                }
                CASPAR_LOG(info) << print() << L" EDID auto-detected HDR (PQ) display: "
                                 << edid.manufacturer << L" " << edid.model
                                 << L" MaxCLL=" << config_.max_cll << L" cd/m²";
                // Re-apply HDR metadata now that we know the display capabilities
                set_hdr_metadata();
            }
        }

        // Quadro Sync configuration
        if (config_.gsync_enabled && nvapi_->gsync_device_count() > 0) {
            auto source = (config_.gsync_source == gsync_reference::external)
                              ? sync_source::house_sync
                              : sync_source::vsync;

            if (config_.gsync_master) {
                nvapi_->configure_sync(config_.gpu_index, config_.output_index, source);
            }

            auto status = nvapi_->get_sync_status(config_.gpu_index);
            CASPAR_LOG(info) << print() << L" Quadro Sync: "
                             << (status.synced ? L"LOCKED" : L"UNLOCKED")
                             << L" role=" << (status.role == sync_role::master ? L"master" : L"slave")
                             << (status.house_sync ? L" house_sync=" + std::to_wstring(status.house_sync_freq) + L"Hz" : L"");
        }
    }

    void present_loop()
    {
        // Wait until buffer has enough frames to satisfy the configured delay
        // before starting to present. This introduces a fixed N-frame latency
        // to video output, allowing operators to compensate for downstream
        // pipeline delay (scalers, audio de-embedders, LED processors).
        const auto min_fill = static_cast<size_t>(config_.delay_frames + 1);

        while (running_) {
            core::const_frame frame;
            {
                std::unique_lock<std::mutex> lock(buffer_mutex_);
                buffer_cv_.wait_for(lock, std::chrono::milliseconds(50),
                                    [this, min_fill] { return buffer_.size() >= min_fill || !running_; });
                if (!running_)
                    break;
                if (buffer_.size() < min_fill)
                    continue;
                frame = std::move(buffer_.front());
                buffer_.pop();
                graph_->set_value("buffered-video",
                                  static_cast<double>(buffer_.size()) / config_.buffer_depth);
            }

            caspar::timer frame_timer;
            present_frame(frame);
            graph_->set_value("frame-time", frame_timer.elapsed() * format_desc_.fps * 0.5);

            // Measure VBlank drift: time between present and actual VBlank signal
            if (vblank_supported_ && display_handle_ != VK_NULL_HANDLE) {
                if (vblank_fence_ == VK_NULL_HANDLE) {
                    vblank_fence_ = device_->create_vblank_fence(display_handle_);
                }
                if (vblank_fence_ != VK_NULL_HANDLE) {
                    caspar::timer vblank_timer;
                    auto wait_result = vkWaitForFences(device_->device(), 1, &vblank_fence_, VK_TRUE, 50000000);
                    if (wait_result == VK_SUCCESS) {
                        auto drift = vblank_timer.elapsed() * format_desc_.fps;
                        graph_->set_value("vblank-drift", drift * 0.5);
                    }
                    // Destroy and re-create — VK_EXT_display_control fences are single-shot
                    vkDestroyFence(device_->device(), vblank_fence_, nullptr);
                    vblank_fence_ = VK_NULL_HANDLE;
                }
            }

            auto tick = tick_timer_.elapsed() * format_desc_.fps * 0.5;
            graph_->set_value("tick-time", tick);
            tick_timer_.restart();

            // Pump window messages for FSE window (needed for WM_DISPLAYCHANGE etc.)
            if (fse_hwnd_) {
                MSG msg;
                while (PeekMessageW(&msg, fse_hwnd_, 0, 0, PM_REMOVE)) {
                    TranslateMessage(&msg);
                    DispatchMessageW(&msg);
                }
            }
        }
    }

    bool has_subregion() const
    {
        return config_.src_x != 0 || config_.src_y != 0 || config_.region_w != 0 || config_.region_h != 0 ||
               config_.dest_x != 0 || config_.dest_y != 0;
    }

    VkImageBlit compute_blit_region(uint32_t src_width, uint32_t src_height) const
    {
        VkImageBlit blit{};
        blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};

        if (has_subregion()) {
            const int32_t sw_max = static_cast<int32_t>(src_width);
            const int32_t sh_max = static_cast<int32_t>(src_height);
            const int32_t dw_max = static_cast<int32_t>(swapchain_.width);
            const int32_t dh_max = static_cast<int32_t>(swapchain_.height);

            // Clamp source offsets to source dimensions
            int32_t sx = (std::clamp)(config_.src_x, 0, sw_max);
            int32_t sy = (std::clamp)(config_.src_y, 0, sh_max);
            int sw = config_.region_w > 0 ? config_.region_w : sw_max - sx;
            int sh = config_.region_h > 0 ? config_.region_h : sh_max - sy;
            // Clamp source region end to source bounds
            sw = (std::min)(sw, sw_max - sx);
            sh = (std::min)(sh, sh_max - sy);

            // Clamp destination offsets to swapchain dimensions
            int32_t dx = (std::clamp)(config_.dest_x, 0, dw_max);
            int32_t dy = (std::clamp)(config_.dest_y, 0, dh_max);
            // Clamp destination end to swapchain bounds
            int dw = (std::min)(sw, dw_max - dx);
            int dh = (std::min)(sh, dh_max - dy);

            blit.srcOffsets[0] = {sx, sy, 0};
            blit.srcOffsets[1] = {sx + sw, sy + sh, 1};
            blit.dstOffsets[0] = {dx, dy, 0};
            blit.dstOffsets[1] = {dx + dw, dy + dh, 1};
        } else {
            blit.srcOffsets[0] = {0, 0, 0};
            blit.srcOffsets[1] = {static_cast<int32_t>(src_width), static_cast<int32_t>(src_height), 1};
            blit.dstOffsets[0] = {0, 0, 0};
            blit.dstOffsets[1] = {static_cast<int32_t>(swapchain_.width), static_cast<int32_t>(swapchain_.height), 1};
        }

        return blit;
    }

    void recreate_swapchain()
    {
        vkDeviceWaitIdle(device_->device());

        for (auto& iv : swapchain_.image_views)
            vkDestroyImageView(device_->device(), iv, nullptr);
        swapchain_.image_views.clear();
        swapchain_.images.clear();

        auto old = swapchain_.swapchain;
        swapchain_.swapchain = VK_NULL_HANDLE;

        // Verify surface is still valid
        VkSurfaceCapabilitiesKHR caps;
        auto cap_result =
            vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device_->physical_device(), swapchain_.surface, &caps);
        if (cap_result != VK_SUCCESS) {
            CASPAR_LOG(warning) << print() << L" Surface lost during hot-plug. Waiting for reconnect...";
            if (old != VK_NULL_HANDLE)
                vkDestroySwapchainKHR(device_->device(), old, nullptr);
            display_lost_ = true;
            return;
        }

        if (old != VK_NULL_HANDLE)
            vkDestroySwapchainKHR(device_->device(), old, nullptr);

        create_swapchain();
        set_hdr_metadata();

        // Recreate color pipeline if swapchain dimensions changed (hot-plug to different display)
        if (color_pipeline_ && (color_pipeline_->width() != swapchain_.width ||
                                color_pipeline_->height() != swapchain_.height)) {
            try {
                color_pipeline_ = std::make_unique<color_convert_pipeline>(
                    *device_, swapchain_.width, swapchain_.height);
                color_pipeline_->update_config(config_.gamut, config_.eotf,
                                               static_cast<float>(config_.max_cll));
                CASPAR_LOG(info) << print() << L" Color pipeline recreated for new dimensions.";
            } catch (const std::exception& e) {
                CASPAR_LOG(error) << print() << L" Failed to recreate color pipeline: " << e.what();
                color_pipeline_.reset();
            }
        }

        display_lost_ = false;
        CASPAR_LOG(info) << print() << L" Swapchain recreated successfully.";
    }

    void present_frame(const core::const_frame& frame)
    {
        if (display_lost_) {
            switch (config_.on_disconnect) {
                case disconnect_behavior::hold:
                    // Hold last frame — don't retry, just return silently
                    return;
                case disconnect_behavior::black:
                    // Attempt recovery every N frames (same as retry)
                    // but don't log dropped-frame since black is intentional
                    if (++hotplug_retry_counter_ % 50 == 0)
                        recreate_swapchain();
                    if (display_lost_)
                        return;
                    break;
                case disconnect_behavior::retry:
                default:
                    // Attempt recovery every N frames
                    if (++hotplug_retry_counter_ % 50 == 0) {
                        recreate_swapchain();
                    }
                    if (display_lost_)
                        return;
                    break;
            }
        }

        auto dev = device_->device();

        // Wait for previous frame to finish
        vkWaitForFences(dev, 1, &swapchain_.in_flight, VK_TRUE, UINT64_MAX);
        vkResetFences(dev, 1, &swapchain_.in_flight);

        // Acquire next swapchain image
        uint32_t image_index = 0;
        auto     result      = vkAcquireNextImageKHR(
            dev, swapchain_.swapchain, UINT64_MAX, swapchain_.image_available, VK_NULL_HANDLE, &image_index);

        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
            // VK_SUBOPTIMAL means the acquire succeeded and signaled image_available.
            // We must reset the semaphore before retrying acquire after recreation,
            // otherwise the retry would double-signal an already-signaled semaphore (UB).
            vkDestroySemaphore(dev, swapchain_.image_available, nullptr);
            VkSemaphoreCreateInfo sem_info{};
            sem_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
            vkCreateSemaphore(dev, &sem_info, nullptr, &swapchain_.image_available);

            recreate_swapchain();
            if (display_lost_)
                return;
            // Retry acquire after recreation
            result = vkAcquireNextImageKHR(
                dev, swapchain_.swapchain, UINT64_MAX, swapchain_.image_available, VK_NULL_HANDLE, &image_index);
            if (result != VK_SUCCESS) {
                // NOTE: Dropping a frame here after recreation is expected (by design).
                // The swapchain was just recreated; one dropped frame is acceptable.
                graph_->set_tag(diagnostics::tag_severity::WARNING, "dropped-frame");
                return;
            }
        } else if (result == VK_ERROR_SURFACE_LOST_KHR) {
            CASPAR_LOG(warning) << print() << L" Display disconnected (surface lost).";
            display_lost_ = true;
            graph_->set_tag(diagnostics::tag_severity::WARNING, "dropped-frame");
            return;
        }

        // Record command buffer: copy frame data → swapchain image
        vkResetCommandBuffer(swapchain_.cmd_buffer, 0);

        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(swapchain_.cmd_buffer, &begin_info);

        // Transition swapchain image to transfer dst
        VkImageMemoryBarrier barrier{};
        barrier.sType                           = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout                       = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout                       = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
        barrier.image                           = swapchain_.images[image_index];
        barrier.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.baseMipLevel   = 0;
        barrier.subresourceRange.levelCount     = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount     = 1;
        barrier.srcAccessMask                   = 0;
        barrier.dstAccessMask                   = VK_ACCESS_TRANSFER_WRITE_BIT;

        vkCmdPipelineBarrier(swapchain_.cmd_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);

        // When color conversion is active, blit into intermediate image instead of swapchain.
        // Skip during identify overlay — it writes directly to the swapchain and the
        // intermediate would contain uninitialized data if compute ran.
        const bool identify_active = identify_frames_remaining_ > 0;
        const bool color_convert_active = color_pipeline_ && color_pipeline_->is_active() && !identify_active;
        VkImage blit_dest_image = color_convert_active ? color_pipeline_->image()
                                                       : swapchain_.images[image_index];

        if (color_convert_active) {
            // Transition intermediate to TRANSFER_DST
            VkImageMemoryBarrier int_barrier{};
            int_barrier.sType                           = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            int_barrier.oldLayout                       = VK_IMAGE_LAYOUT_UNDEFINED;
            int_barrier.newLayout                       = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            int_barrier.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
            int_barrier.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
            int_barrier.image                           = color_pipeline_->image();
            int_barrier.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
            int_barrier.subresourceRange.baseMipLevel   = 0;
            int_barrier.subresourceRange.levelCount     = 1;
            int_barrier.subresourceRange.baseArrayLayer = 0;
            int_barrier.subresourceRange.layerCount     = 1;
            int_barrier.srcAccessMask                   = 0;
            int_barrier.dstAccessMask                   = VK_ACCESS_TRANSFER_WRITE_BIT;

            vkCmdPipelineBarrier(swapchain_.cmd_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &int_barrier);
        }

        // Output identification overlay: clear to a unique color for 3 seconds
        bool used_shared_pool = false;
        bool wrote_to_intermediate = false; // Tracks whether intermediate image received valid data
        if (identify_frames_remaining_ > 0) {
            --identify_frames_remaining_;
            // Each output gets a distinct color: cycle through R/G/B/C/M/Y based on output_index
            static const float colors[][4] = {
                {0.0f, 0.3f, 0.8f, 1.0f}, // Blue (output 1)
                {0.0f, 0.8f, 0.3f, 1.0f}, // Green (output 2)
                {0.8f, 0.2f, 0.0f, 1.0f}, // Red (output 3)
                {0.0f, 0.8f, 0.8f, 1.0f}, // Cyan (output 4)
                {0.8f, 0.0f, 0.8f, 1.0f}, // Magenta (output 5)
                {0.8f, 0.8f, 0.0f, 1.0f}, // Yellow (output 6)
            };
            int idx = (std::max)(0, config_.output_index - 1) % 6;
            VkClearColorValue clear_color = {{colors[idx][0], colors[idx][1], colors[idx][2], colors[idx][3]}};
            VkImageSubresourceRange range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCmdClearColorImage(swapchain_.cmd_buffer, swapchain_.images[image_index],
                                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clear_color, 1, &range);
        } else if (shared_pool_ && !affinity_ctx_ && frame.texture()) {
            // Zero-copy path (same GPU): blit OGL texture → shared texture → VK blit to swapchain
            auto ogl_tex = std::dynamic_pointer_cast<accelerator::ogl::texture>(frame.texture());
            if (ogl_tex) {
                // Blit on OGL thread: copy from mixer output into shared exportable texture
                ogl_device_->dispatch_sync([&] {
                    shared_pool_->blit_from_texture(ogl_tex->id(), ogl_tex->width(), ogl_tex->height());
                    shared_pool_->signal_gl();
                });

                // Advance indices so current_vk_image() returns the slot just written
                shared_pool_->swap();

                // Transition shared image for transfer src
                VkImageMemoryBarrier src_barrier{};
                src_barrier.sType                           = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                src_barrier.oldLayout                       = VK_IMAGE_LAYOUT_UNDEFINED;
                src_barrier.newLayout                       = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
                src_barrier.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
                src_barrier.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
                src_barrier.image                           = shared_pool_->current_vk_image();
                src_barrier.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
                src_barrier.subresourceRange.baseMipLevel   = 0;
                src_barrier.subresourceRange.levelCount     = 1;
                src_barrier.subresourceRange.baseArrayLayer = 0;
                src_barrier.subresourceRange.layerCount     = 1;
                src_barrier.srcAccessMask                   = 0;
                src_barrier.dstAccessMask                   = VK_ACCESS_TRANSFER_READ_BIT;

                vkCmdPipelineBarrier(swapchain_.cmd_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                     VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &src_barrier);

                // Blit shared VK image → swapchain image (subregion-aware)
                auto blit_region = compute_blit_region(shared_pool_->width(), shared_pool_->height());

                vkCmdBlitImage(swapchain_.cmd_buffer, shared_pool_->current_vk_image(),
                               VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               blit_dest_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                               1, &blit_region, VK_FILTER_LINEAR);

                used_shared_pool = true;
                wrote_to_intermediate = color_convert_active;
            } else {
                // Texture isn't OGL — CPU fallback (target swapchain directly;
                // upload_frame_cpu writes BGRA8 which is incompatible with RGBA16F intermediate)
                upload_frame_cpu(frame, swapchain_.images[image_index]);
            }
        } else if (shared_pool_ && affinity_ctx_) {
            // Cross-GPU path: transfer frame from GPU A → GPU B
            bool transferred = false;

#ifdef CASPAR_CUDA_PEER_ENABLED
            if (cuda_peer_ && frame.texture()) {
                // CUDA peer DMA: direct GPU→GPU via PCIe DMA engine (no CPU copy)
                auto ogl_tex = std::dynamic_pointer_cast<accelerator::ogl::texture>(frame.texture());
                if (ogl_tex) {
                    try {
                        // Phase 1: Read mixer texture into GPU A staging (on OGL thread)
                        ogl_device_->dispatch_sync([&] {
                            cuda_peer_->read_source(ogl_tex->id(), ogl_tex->width(), ogl_tex->height());
                        });

                        // Phase 2: DMA from GPU A → GPU B (no GL context needed)
                        cuda_peer_->peer_copy();

                        // Phase 3: Write to landing texture + blit into shared pool (on affinity thread)
                        affinity_ctx_->dispatch_sync([&] {
                            cuda_peer_->write_dest();
                            shared_pool_->blit_from_texture(
                                cuda_peer_->dest_texture(), format_desc_.width, format_desc_.height);
                            shared_pool_->signal_gl();
                        });

                        transferred = true;
                    } catch (const std::exception& e) {
                        CASPAR_LOG(warning) << print()
                            << L" CUDA peer transfer failed, falling back to PBO: " << e.what();
                    }
                }
            }
#endif // CASPAR_CUDA_PEER_ENABLED

            if (!transferred) {
                // PBO fallback: CPU pixels → PBO upload → shared texture
                const auto& img = frame.image_data(0);
                const auto* pixels = img.data();
                if (pixels && img.size() > 0) {
                    affinity_ctx_->dispatch_sync([&] {
                        GLuint tex_id = affinity_ctx_->upload_frame(
                            pixels, format_desc_.width, format_desc_.height, format_desc_.width * 4);
                        shared_pool_->blit_from_texture(tex_id, format_desc_.width, format_desc_.height);
                        shared_pool_->signal_gl();
                    });
                    transferred = true;
                } else {
                    upload_frame_cpu(frame, swapchain_.images[image_index]);
                }
            }

            if (transferred) {
                shared_pool_->swap();

                // Transition shared image for transfer src
                VkImageMemoryBarrier src_barrier{};
                src_barrier.sType                           = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                src_barrier.oldLayout                       = VK_IMAGE_LAYOUT_UNDEFINED;
                src_barrier.newLayout                       = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
                src_barrier.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
                src_barrier.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
                src_barrier.image                           = shared_pool_->current_vk_image();
                src_barrier.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
                src_barrier.subresourceRange.baseMipLevel   = 0;
                src_barrier.subresourceRange.levelCount     = 1;
                src_barrier.subresourceRange.baseArrayLayer = 0;
                src_barrier.subresourceRange.layerCount     = 1;
                src_barrier.srcAccessMask                   = 0;
                src_barrier.dstAccessMask                   = VK_ACCESS_TRANSFER_READ_BIT;

                vkCmdPipelineBarrier(swapchain_.cmd_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                     VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &src_barrier);

                auto blit_region = compute_blit_region(shared_pool_->width(), shared_pool_->height());
                vkCmdBlitImage(swapchain_.cmd_buffer, shared_pool_->current_vk_image(),
                               VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               blit_dest_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                               1, &blit_region, VK_FILTER_LINEAR);

                used_shared_pool = true;
                wrote_to_intermediate = color_convert_active;
            } else {
                upload_frame_cpu(frame, swapchain_.images[image_index]);
            }
        } else if (interop_ && interop_->vk_image() != VK_NULL_HANDLE) {
            // Legacy interop path (manual handle import)
            auto blit_region = compute_blit_region(format_desc_.width, format_desc_.height);

            vkCmdBlitImage(swapchain_.cmd_buffer, interop_->vk_image(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           blit_dest_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit_region,
                           VK_FILTER_LINEAR);
            wrote_to_intermediate = color_convert_active;
        } else {
            // CPU fallback: upload pixel data via staging buffer
            // Always targets swapchain — BGRA8 pixels are incompatible with RGBA16F intermediate
            upload_frame_cpu(frame, swapchain_.images[image_index]);
        }

        // ─── Color space conversion (compute shader) ───────────────────────────
        // Only run if the intermediate was actually written to by a GPU blit path.
        if (color_convert_active && wrote_to_intermediate) {
            // Transition intermediate: TRANSFER_DST → GENERAL (for compute read/write)
            VkImageMemoryBarrier cs_barrier{};
            cs_barrier.sType                           = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            cs_barrier.oldLayout                       = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            cs_barrier.newLayout                       = VK_IMAGE_LAYOUT_GENERAL;
            cs_barrier.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
            cs_barrier.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
            cs_barrier.image                           = color_pipeline_->image();
            cs_barrier.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
            cs_barrier.subresourceRange.baseMipLevel   = 0;
            cs_barrier.subresourceRange.levelCount     = 1;
            cs_barrier.subresourceRange.baseArrayLayer = 0;
            cs_barrier.subresourceRange.layerCount     = 1;
            cs_barrier.srcAccessMask                   = VK_ACCESS_TRANSFER_WRITE_BIT;
            cs_barrier.dstAccessMask                   = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

            vkCmdPipelineBarrier(swapchain_.cmd_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &cs_barrier);

            // Dispatch compute shader
            color_pipeline_->dispatch(swapchain_.cmd_buffer, color_pipeline_->width(), color_pipeline_->height());

            // Transition intermediate: GENERAL → TRANSFER_SRC (for final blit to swapchain)
            cs_barrier.oldLayout     = VK_IMAGE_LAYOUT_GENERAL;
            cs_barrier.newLayout     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            cs_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            cs_barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

            vkCmdPipelineBarrier(swapchain_.cmd_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &cs_barrier);

            // Blit intermediate → swapchain (swapchain is already in TRANSFER_DST from initial barrier)
            // Use full-extent source — the subregion was already applied when blitting into
            // the intermediate. Applying compute_blit_region() again would crop twice.
            VkImageBlit final_region{};
            final_region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            final_region.srcOffsets[0]  = {0, 0, 0};
            final_region.srcOffsets[1]  = {static_cast<int32_t>(color_pipeline_->width()),
                                           static_cast<int32_t>(color_pipeline_->height()), 1};
            final_region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            final_region.dstOffsets[0]  = {0, 0, 0};
            final_region.dstOffsets[1]  = {static_cast<int32_t>(swapchain_.width),
                                           static_cast<int32_t>(swapchain_.height), 1};
            vkCmdBlitImage(swapchain_.cmd_buffer, color_pipeline_->image(),
                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           swapchain_.images[image_index], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                           1, &final_region, VK_FILTER_LINEAR);
        }

        // Transition swapchain image to present layout
        VkImageMemoryBarrier present_barrier{};
        present_barrier.sType                           = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        present_barrier.oldLayout                       = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        present_barrier.newLayout                       = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        present_barrier.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
        present_barrier.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
        present_barrier.image                           = swapchain_.images[image_index];
        present_barrier.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        present_barrier.subresourceRange.baseMipLevel   = 0;
        present_barrier.subresourceRange.levelCount     = 1;
        present_barrier.subresourceRange.baseArrayLayer = 0;
        present_barrier.subresourceRange.layerCount     = 1;
        present_barrier.srcAccessMask                   = VK_ACCESS_TRANSFER_WRITE_BIT;
        present_barrier.dstAccessMask                   = 0;

        vkCmdPipelineBarrier(swapchain_.cmd_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr, 1, &present_barrier);

        vkEndCommandBuffer(swapchain_.cmd_buffer);

        // Submit — wait on shared pool semaphore too if zero-copy was used
        VkSemaphore          wait_semaphores[2] = {swapchain_.image_available, VK_NULL_HANDLE};
        VkPipelineStageFlags wait_stages[2]     = {VK_PIPELINE_STAGE_TRANSFER_BIT,
                                                   VK_PIPELINE_STAGE_TRANSFER_BIT};
        uint32_t             wait_count         = 1;

        if (used_shared_pool) {
            wait_semaphores[1] = shared_pool_->wait_semaphore_vk();
            wait_count         = 2;
        }

        VkSubmitInfo submit_info{};
        submit_info.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.waitSemaphoreCount   = wait_count;
        submit_info.pWaitSemaphores      = wait_semaphores;
        submit_info.pWaitDstStageMask    = wait_stages;
        submit_info.commandBufferCount   = 1;
        submit_info.pCommandBuffers      = &swapchain_.cmd_buffer;
        submit_info.signalSemaphoreCount = 1;
        submit_info.pSignalSemaphores    = &swapchain_.render_finished;

        auto submit_result = vkQueueSubmit(device_->present_queue(), 1, &submit_info, swapchain_.in_flight);
        if (submit_result == VK_ERROR_DEVICE_LOST) {
            CASPAR_LOG(error) << print() << L" GPU device lost (TDR) during submit. Output halted.";
            display_lost_ = true;
            return;
        }

        // Present
        VkPresentInfoKHR present_info{};
        present_info.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        present_info.waitSemaphoreCount = 1;
        present_info.pWaitSemaphores    = &swapchain_.render_finished;
        present_info.swapchainCount     = 1;
        present_info.pSwapchains        = &swapchain_.swapchain;
        present_info.pImageIndices      = &image_index;

        auto present_result = vkQueuePresentKHR(device_->present_queue(), &present_info);
        if (present_result == VK_ERROR_OUT_OF_DATE_KHR || present_result == VK_SUBOPTIMAL_KHR) {
            recreate_swapchain();
        } else if (present_result == VK_ERROR_SURFACE_LOST_KHR) {
            display_lost_ = true;
            CASPAR_LOG(warning) << print() << L" Display disconnected during present.";
        } else if (present_result == VK_ERROR_DEVICE_LOST) {
            CASPAR_LOG(error) << print() << L" GPU device lost (TDR) during present. Output halted.";
            display_lost_ = true;
            return;
        }

        ++frames_presented_;
    }

    void upload_frame_cpu(const core::const_frame& frame, VkImage dst_image)
    {
        // CPU path: create/reuse staging buffer and copy pixel data
        // WARNING: This path assumes swapchain format matches BGRA8.
        // If HDR is active (10-bit/16-bit swapchain), the raw byte copy will produce
        // incorrect colors for non-black content. Log once and use a black clear instead.
        // Note: all-zero bytes ARE valid black in any format, but non-zero BGRA8 values
        // would be misinterpreted as 10-bit or float data, producing garbled output.
        if (config_.transfer == hdr_transfer::pq || config_.transfer == hdr_transfer::hlg) {
            static bool warned = false;
            if (!warned) {
                CASPAR_LOG(warning) << print()
                    << L" CPU upload path used with HDR swapchain format — non-black frames will be incorrect."
                    << L" Use GPU interop (shared_pool) for correct HDR output.";
                warned = true;
            }
            // Black clear — invisible on LED walls, correct for CLEAR/empty frames
            VkClearColorValue clear_color = {{0.0f, 0.0f, 0.0f, 0.0f}};
            VkImageSubresourceRange range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCmdClearColorImage(swapchain_.cmd_buffer, dst_image,
                                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clear_color, 1, &range);
            return;
        }

        const auto& img = frame.image_data(0);
        const auto* pixels = img.data();
        if (!pixels)
            return;

        auto        size      = img.size();
        if (size == 0)
            return;

        // Validate buffer size matches expected frame dimensions to avoid GPU overread
        const size_t expected_size = static_cast<size_t>(format_desc_.width) * format_desc_.height * 4;
        if (size < expected_size) {
            CASPAR_LOG(warning) << print() << L" CPU upload: pixel buffer (" << size
                                << L" bytes) smaller than expected (" << expected_size << L"). Skipping frame.";
            return;
        }

        auto        dev       = device_->device();
        auto        phys_dev  = device_->physical_device();

        // Create or reuse staging buffer
        if (staging_buffer_ == VK_NULL_HANDLE || staging_buffer_size_ < size) {
            if (staging_buffer_ != VK_NULL_HANDLE) {
                vkDestroyBuffer(dev, staging_buffer_, nullptr);
                vkFreeMemory(dev, staging_memory_, nullptr);
            }

            VkBufferCreateInfo buf_info{};
            buf_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            buf_info.size  = size;
            buf_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
            if (vkCreateBuffer(dev, &buf_info, nullptr, &staging_buffer_) != VK_SUCCESS) {
                CASPAR_LOG(error) << print() << L" Failed to create staging buffer";
                staging_buffer_ = VK_NULL_HANDLE;
                return;
            }

            VkMemoryRequirements mem_reqs;
            vkGetBufferMemoryRequirements(dev, staging_buffer_, &mem_reqs);

            VkPhysicalDeviceMemoryProperties mem_props;
            vkGetPhysicalDeviceMemoryProperties(phys_dev, &mem_props);

            uint32_t type_index = UINT32_MAX;
            for (uint32_t i = 0; i < mem_props.memoryTypeCount; ++i) {
                if ((mem_reqs.memoryTypeBits & (1u << i)) &&
                    (mem_props.memoryTypes[i].propertyFlags &
                     (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
                        (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
                    type_index = i;
                    break;
                }
            }

            if (type_index == UINT32_MAX) {
                CASPAR_LOG(error) << print() << L" No suitable memory type for staging buffer";
                vkDestroyBuffer(dev, staging_buffer_, nullptr);
                staging_buffer_ = VK_NULL_HANDLE;
                return;
            }

            VkMemoryAllocateInfo alloc_info{};
            alloc_info.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            alloc_info.allocationSize  = mem_reqs.size;
            alloc_info.memoryTypeIndex = type_index;
            if (vkAllocateMemory(dev, &alloc_info, nullptr, &staging_memory_) != VK_SUCCESS) {
                CASPAR_LOG(error) << print() << L" Failed to allocate staging memory";
                vkDestroyBuffer(dev, staging_buffer_, nullptr);
                staging_buffer_ = VK_NULL_HANDLE;
                return;
            }
            vkBindBufferMemory(dev, staging_buffer_, staging_memory_, 0);

            staging_buffer_size_ = size;
        }

        // Map and copy
        void* mapped = nullptr;
        if (vkMapMemory(dev, staging_memory_, 0, size, 0, &mapped) != VK_SUCCESS) {
            CASPAR_LOG(error) << print() << L" Failed to map staging memory";
            return;
        }
        memcpy(mapped, pixels, size);
        vkUnmapMemory(dev, staging_memory_);

        // Copy buffer → image
        VkBufferImageCopy region{};
        region.bufferOffset      = 0;
        region.bufferRowLength   = 0;
        region.bufferImageHeight = 0;
        region.imageSubresource  = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageOffset       = {0, 0, 0};
        // Use actual frame dimensions — the pixel buffer is format_desc_ sized, not swapchain sized.
        // Using swapchain dimensions would overread the buffer if swapchain > frame.
        region.imageExtent       = {static_cast<uint32_t>(format_desc_.width),
                                    static_cast<uint32_t>(format_desc_.height), 1};

        vkCmdCopyBufferToImage(swapchain_.cmd_buffer, staging_buffer_, dst_image,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    }

    // Custom window procedure that ignores WM_CLOSE/WM_SYSCOMMAND SC_CLOSE
    // to prevent DefWindowProcW from destroying our FSE window unexpectedly (use-after-free)
    static LRESULT CALLBACK fse_wnd_proc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
    {
        switch (msg) {
            case WM_CLOSE:
                return 0; // Ignore — we manage window lifetime explicitly
            case WM_SYSCOMMAND:
                if ((wParam & 0xFFF0) == SC_CLOSE)
                    return 0;
                break;
        }
        return DefWindowProcW(hwnd, msg, wParam, lParam);
    }

    void create_fse_window()
    {
        // Create a borderless window for fullscreen exclusive on consumer GPUs
        WNDCLASSEXW wc{};
        wc.cbSize        = sizeof(WNDCLASSEXW);
        wc.style         = CS_HREDRAW | CS_VREDRAW;
        wc.lpfnWndProc   = fse_wnd_proc;
        wc.hInstance     = GetModuleHandle(nullptr);
        wc.lpszClassName = L"CasparVulkanOutput";
        RegisterClassExW(&wc);

        // Enumerate monitors and select by output_index
        struct monitor_enum_data
        {
            int target_index;
            int current_index;
            RECT rect;
            bool found;
        };
        monitor_enum_data data{config_.output_index, 0, {}, false};

        EnumDisplayMonitors(
            nullptr, nullptr,
            [](HMONITOR, HDC, LPRECT rect, LPARAM lparam) -> BOOL {
                auto* d = reinterpret_cast<monitor_enum_data*>(lparam);
                d->current_index++;
                if (d->current_index == d->target_index) {
                    d->rect  = *rect;
                    d->found = true;
                    return FALSE; // Stop enumeration
                }
                return TRUE;
            },
            reinterpret_cast<LPARAM>(&data));

        int x = 0, y = 0;
        int w = format_desc_.width;
        int h = format_desc_.height;

        if (data.found) {
            x = data.rect.left;
            y = data.rect.top;
            w = data.rect.right - data.rect.left;
            h = data.rect.bottom - data.rect.top;
            CASPAR_LOG(info) << print() << L" FSE window on monitor " << config_.output_index << L" at ("
                             << x << L"," << y << L") " << w << L"x" << h;
        } else {
            CASPAR_LOG(warning) << print() << L" Monitor " << config_.output_index
                                << L" not found, using primary display.";
        }

        fse_hwnd_ = CreateWindowExW(WS_EX_TOPMOST, L"CasparVulkanOutput", L"CasparCG Vulkan Output",
                                    WS_POPUP | WS_VISIBLE, x, y, w, h, nullptr, nullptr,
                                    GetModuleHandle(nullptr), nullptr);

        ShowWindow(fse_hwnd_, SW_SHOW);
    }

    void destroy_resources()
    {
        running_ = false;
        buffer_cv_.notify_all();
        if (present_thread_.joinable())
            present_thread_.join();

        if (device_) {
            auto dev = device_->device();
            vkDeviceWaitIdle(dev);

            if (staging_buffer_ != VK_NULL_HANDLE) {
                vkDestroyBuffer(dev, staging_buffer_, nullptr);
                vkFreeMemory(dev, staging_memory_, nullptr);
            }

            if (vblank_fence_ != VK_NULL_HANDLE) {
                vkDestroyFence(dev, vblank_fence_, nullptr);
                vblank_fence_ = VK_NULL_HANDLE;
            }

            for (auto& iv : swapchain_.image_views)
                vkDestroyImageView(dev, iv, nullptr);

            if (swapchain_.swapchain != VK_NULL_HANDLE)
                vkDestroySwapchainKHR(dev, swapchain_.swapchain, nullptr);
            if (swapchain_.image_available != VK_NULL_HANDLE)
                vkDestroySemaphore(dev, swapchain_.image_available, nullptr);
            if (swapchain_.render_finished != VK_NULL_HANDLE)
                vkDestroySemaphore(dev, swapchain_.render_finished, nullptr);
            if (swapchain_.in_flight != VK_NULL_HANDLE)
                vkDestroyFence(dev, swapchain_.in_flight, nullptr);
            if (swapchain_.cmd_pool != VK_NULL_HANDLE)
                vkDestroyCommandPool(dev, swapchain_.cmd_pool, nullptr);
            if (swapchain_.surface != VK_NULL_HANDLE)
                vkDestroySurfaceKHR(device_->instance(), swapchain_.surface, nullptr);
        }

        // Destroy resources that reference VkDevice handles BEFORE destroying the device.
        // Order matters: color_pipeline_ and cuda_peer_ hold raw VkDevice/VkQueue handles.
        color_pipeline_.reset();
#ifdef CASPAR_CUDA_PEER_ENABLED
        cuda_peer_.reset();
#endif
        affinity_ctx_.reset();
        interop_.reset();
        shared_pool_.reset();
        device_.reset();
        nvapi_.reset();

        if (fse_hwnd_) {
            DestroyWindow(fse_hwnd_);
            fse_hwnd_ = nullptr;
            UnregisterClassW(L"CasparVulkanOutput", GetModuleHandle(nullptr));
        }
    }

    // NOTE: config_ is read without mutex on the present thread for performance.
    // config_mutex_ only protects modifications from state() reads on other threads.
    // config_ MUST NOT be modified after the present thread starts (after initialize()).
    configuration                    config_;
    mutable std::mutex               config_mutex_; // Protects state() reads from config_ writes
    core::video_format_desc          format_desc_;
    int                              port_index_ = 0;
    spl::shared_ptr<diagnostics::graph> graph_;
    executor                         executor_;

    std::shared_ptr<accelerator::ogl::device> ogl_device_;
    std::unique_ptr<vulkan_device>   device_;
    std::unique_ptr<vulkan_interop>  interop_;
    std::unique_ptr<shared_texture_pool> shared_pool_;
    std::unique_ptr<gpu_affinity_context> affinity_ctx_; // For cross-GPU interop
#ifdef CASPAR_CUDA_PEER_ENABLED
    std::unique_ptr<cuda_peer_transfer> cuda_peer_;    // CUDA peer DMA (cross-GPU, no CPU copy)
#endif
    std::unique_ptr<nvapi_helpers>   nvapi_;
    std::unique_ptr<color_convert_pipeline> color_pipeline_; // Color space conversion (compute)
    swapchain_resources              swapchain_{};
    HWND                             fse_hwnd_ = nullptr;
    VkDisplayKHR                     display_handle_ = VK_NULL_HANDLE;

    // Frame buffer
    std::queue<core::const_frame>    buffer_;
    std::mutex                       buffer_mutex_;
    std::condition_variable          buffer_cv_;

    // Present thread
    std::thread                      present_thread_;
    std::atomic<bool>                running_{false};

    // Display hot-plug
    std::atomic<bool>                display_lost_{false};
    uint64_t                         hotplug_retry_counter_ = 0;
    std::atomic<uint64_t>            frames_presented_{0};

    // Present barrier
    bool                             present_barrier_enabled_ = false;

    // Output identification
    std::atomic<int>                 identify_frames_remaining_{0};

    // Timing
    caspar::timer                    tick_timer_;
    bool                             vblank_supported_ = false;
    VkFence                          vblank_fence_     = VK_NULL_HANDLE;

    // CPU staging
    VkBuffer                         staging_buffer_      = VK_NULL_HANDLE;
    VkDeviceMemory                   staging_memory_      = VK_NULL_HANDLE;
    size_t                           staging_buffer_size_ = 0;
};

} // anonymous namespace

// ─── Factory functions ──────────────────────────────────────────────────

spl::shared_ptr<core::frame_consumer> create_consumer(const std::vector<std::wstring>&                         params,
                                                      const core::video_format_repository&                     format_repository,
                                                      const std::vector<spl::shared_ptr<core::video_channel>>& channels,
                                                      const core::channel_info& channel_info)
{
    if (params.size() < 1 || !boost::iequals(params.at(0), L"VULKAN_OUTPUT"))
        return core::frame_consumer::empty();

    configuration config;

    if (params.size() > 1)
        config.output_index = std::stoi(params.at(1));
    if (params.size() > 2)
        config.gpu_index = std::stoi(params.at(2));

    // Optional video-mode parameter
    for (size_t i = 1; i < params.size(); ++i) {
        if (boost::iequals(params[i], L"MODE") && i + 1 < params.size())
            config.video_mode = params[++i];
        else if (boost::iequals(params[i], L"DELAY") && i + 1 < params.size())
            config.delay_frames = std::stoi(params[++i]);
    }

    // Extract OGL device from the channel for zero-copy interop
    std::shared_ptr<accelerator::ogl::device> ogl_device;
    for (const auto& ch : channels) {
        if (ch->index() == channel_info.index) {
            auto* ogl_mixer = dynamic_cast<accelerator::ogl::image_mixer*>(ch->frame_factory().get());
            if (ogl_mixer)
                ogl_device = ogl_mixer->get_ogl_device();
            break;
        }
    }

    return spl::make_shared<vulkan_output_consumer>(std::move(config), std::move(ogl_device));
}

spl::shared_ptr<core::frame_consumer>
create_preconfigured_consumer(const boost::property_tree::wptree&                      ptree,
                              const core::video_format_repository&                     format_repository,
                              const std::vector<spl::shared_ptr<core::video_channel>>& channels,
                              const core::channel_info&                                channel_info)
{
    auto config = parse_config(ptree);

    // Extract OGL device from the channel for zero-copy interop
    std::shared_ptr<accelerator::ogl::device> ogl_device;
    for (const auto& ch : channels) {
        if (ch->index() == channel_info.index) {
            auto* ogl_mixer = dynamic_cast<accelerator::ogl::image_mixer*>(ch->frame_factory().get());
            if (ogl_mixer)
                ogl_device = ogl_mixer->get_ogl_device();
            break;
        }
    }

    return spl::make_shared<vulkan_output_consumer>(std::move(config), std::move(ogl_device));
}

std::wstring enumerate_outputs()
{
    std::wstring result;
    auto         displays = vulkan_device::enumerate_displays();

    if (displays.empty()) {
        result = L"No Vulkan display outputs found.\n";
        return result;
    }

    result += L"Vulkan Display Outputs:\n";
    result += L"───────────────────────────────────────────────────────────────\n";

    // Initialize NvAPI for EDID info
    nvapi_helpers nvapi;

    for (const auto& d : displays) {
        result += L"  GPU " + std::to_wstring(d.gpu_index) + L" Output " + std::to_wstring(d.output_index) + L": ";
        result += d.gpu_name + L" - " + d.display_name;
        if (d.width > 0)
            result += L" (" + std::to_wstring(d.width) + L"x" + std::to_wstring(d.height) + L")";
        result += L" [" + std::wstring(d.tier == gpu_tier::pro ? L"Pro" : L"Consumer") + L"]\n";

        // EDID info via NvAPI
        if (nvapi.is_available()) {
            auto edid = nvapi.read_edid(d.gpu_index, d.output_index);
            if (!edid.raw_edid.empty()) {
                result += L"         EDID: " + edid.manufacturer + L" " + edid.model;
                if (edid.supports_hdr)
                    result += L" [HDR MaxCLL=" + std::to_wstring(edid.max_luminance) + L"cd/m²]";
                if (edid.supports_10bit)
                    result += L" [10-bit]";
                result += L"\n";
            }
        }
    }

    // GSync status
    if (nvapi.is_available() && nvapi.gsync_device_count() > 0) {
        result += L"\nQuadro Sync:\n";
        result += L"  " + std::to_wstring(nvapi.gsync_device_count()) + L" GSync device(s) detected\n";
        if (!displays.empty()) {
            auto status = nvapi.get_sync_status(displays[0].gpu_index);
            result += L"  Status: " + std::wstring(status.synced ? L"LOCKED" : L"UNLOCKED");
            if (status.house_sync)
                result += L" | House sync: " + std::to_wstring(status.house_sync_freq) + L"Hz";
            result += L"\n";
        }
    }

    return result;
}

}} // namespace caspar::vulkan_output
