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
#include "../util/vk_device_manager.h"
#include "../util/gpu_frame_cache.h"
#include "../util/shared_texture_pool.h"
#include "../util/interop_context.h"
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

#include <chrono>
#include <condition_variable>
#include <future>
#include <iomanip>
#include <mutex>
#include <queue>
#include <sstream>
#include <thread>

namespace caspar { namespace vulkan_output {

namespace {

// ─── Swapchain management ───────────────────────────────────────────────

static constexpr int MAX_FRAMES_IN_FLIGHT = 3;

struct frame_sync_objects
{
    VkSemaphore     image_available = VK_NULL_HANDLE;
    VkSemaphore     render_finished = VK_NULL_HANDLE;
    VkFence         in_flight       = VK_NULL_HANDLE;
    VkCommandBuffer cmd_buffer      = VK_NULL_HANDLE;
};

struct swapchain_resources
{
    VkSwapchainKHR           swapchain   = VK_NULL_HANDLE;
    VkSurfaceKHR             surface     = VK_NULL_HANDLE;
    std::vector<VkImage>     images;
    std::vector<VkImageView> image_views;
    VkCommandPool            cmd_pool        = VK_NULL_HANDLE;
    uint32_t                 width           = 0;
    uint32_t                 height          = 0;

    // Per-frame-in-flight sync objects (double/triple buffered)
    frame_sync_objects       frames[MAX_FRAMES_IN_FLIGHT];
    int                      current_frame = 0;

    // Convenience accessors for current frame's resources
    VkSemaphore&     image_available_ref() { return frames[current_frame].image_available; }
    VkSemaphore&     render_finished_ref() { return frames[current_frame].render_finished; }
    VkFence&         in_flight_ref()       { return frames[current_frame].in_flight; }
    VkCommandBuffer& cmd_buffer_ref()      { return frames[current_frame].cmd_buffer; }

    void advance_frame() { current_frame = (current_frame + 1) % MAX_FRAMES_IN_FLIGHT; }
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

        // Zero-copy path: blit OGL texture into shared pool via frame cache.
        // The frame cache ensures only one transfer per GPU per frame — the first
        // consumer to call submit_frame() does the actual blit, others wait.
        // For cross-GPU, the cache coordinates CUDA peer DMA or PBO upload.
        if (frame_cache_ && !frame_cache_->is_cross_gpu() && frame.texture()) {
            auto ogl_tex = std::dynamic_pointer_cast<accelerator::ogl::texture>(frame.texture());
            if (ogl_tex) {
                auto* cache = frame_cache_.get();
                ++frame_generation_;
                cache->notify_frame(frame_generation_, [cache, ogl_tex, ogl_dev = ogl_device_] {
                    auto* pool = cache->pool();
                    auto* ictx = cache->interop_ctx();
                    if (ictx) {
                        ictx->dispatch_sync([pool, ogl_tex] {
                            pool->blit_from_texture(ogl_tex->id(), ogl_tex->width(), ogl_tex->height());
                            pool->signal_gl();
                            pool->swap();
                        });
                    } else if (ogl_dev) {
                        ogl_dev->dispatch_sync([pool, ogl_tex] {
                            pool->blit_from_texture(ogl_tex->id(), ogl_tex->width(), ogl_tex->height());
                            pool->signal_gl();
                            pool->swap();
                        });
                    }
                });
            }
        }

        // Block until the present thread has consumed a frame (backpressure).
        // This makes the display vsync the sole reference clock for the channel,
        // eliminating drift between a software timer and the hardware refresh rate.
        // Timeout allows buffer_depth frames to complete even under GL contention.
        {
            auto timeout_ms = static_cast<int64_t>(config_.buffer_depth * 1000.0 / format_desc_.fps) + 50;
            std::unique_lock<std::mutex> lock(buffer_mutex_);
            if (!buffer_cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [this] {
                    return buffer_.size() < static_cast<size_t>(config_.buffer_depth) || !running_;
                })) {
                // Timed out — present thread stalled (display lost, TDR, etc.)
                graph_->set_tag(diagnostics::tag_severity::WARNING, "late-frame");
                if (!buffer_.empty())
                    buffer_.pop();
            }
            if (!running_)
                return caspar::make_ready_future(true);
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
        device_ = vk_device_manager::get(config_.gpu_index);

        // Acquire a dedicated queue for this consumer's submit/present operations.
        // Each consumer gets its own queue so submissions are fully parallel.
        my_queue_idx_ = device_->acquire_queue();
        my_queue_     = device_->queue(my_queue_idx_);

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

        if (!found) {
            // Pro GPU without VK_KHR_display (e.g. RTX A-series on Windows desktop mode),
            // or consumer GPU: use FSE window path.
            // EDID emulation: inject synthetic EDID if the target output has no monitor
            if (config_.edid_emulation) {
                if (!nvapi_)
                    nvapi_ = std::make_unique<nvapi_helpers>();
                if (nvapi_->is_available()) {
                    uint32_t edid_w = config_.region_w > 0 ? config_.region_w : format_desc_.width;
                    uint32_t edid_h = config_.region_h > 0 ? config_.region_h : format_desc_.height;
                    injected_edid_display_id_ = nvapi_->inject_edid(
                        config_.gpu_index, config_.output_index,
                        edid_w, edid_h, format_desc_.fps);
                    if (injected_edid_display_id_ != 0) {
                        // Give Windows time to enumerate the new display
                        std::this_thread::sleep_for(std::chrono::milliseconds(2000));
                    }
                }
            }
            // Consumer fallback: create a borderless fullscreen window
            // Launch blanker BEFORE our window so it's behind us in z-order
            if (config_.display_blanker)
                launch_display_blanker();
            create_fse_window();
        }

        // Create surface
        if (adapter_mismatch_) {
            CASPAR_LOG(warning) << print()
                                << L" Display is on a non-NVIDIA adapter (e.g. IddCx/VDD). "
                                   L"Vulkan presentation disabled — using GDI fallback (half-rate blit).";
            device_dead_ = true; // Prevents Vulkan present path; GDI path handles rendering
        } else if (device_->tier() == gpu_tier::pro && found) {
            // Convert fps to millihertz for display mode matching
            uint32_t refresh_mhz = static_cast<uint32_t>(format_desc_.fps * 1000.0 + 0.5);
            swapchain_.surface = device_->create_display_surface(target, refresh_mhz);
            display_handle_ = target.display_handle;
        } else if (fse_hwnd_) {
            swapchain_.surface = device_->create_win32_surface(fse_hwnd_);
        }

        if (!adapter_mismatch_) {
            setup_present_barrier(); // Must be before create_swapchain so the barrier struct is chained
            create_swapchain();
            set_hdr_metadata();
            create_sync_objects();
            setup_nvapi();
            create_command_pool();
        }

        // Create per-GPU frame cache for zero-copy transfer (shared across consumers on same GPU)
        if (ogl_device_ && !adapter_mismatch_) {
            try {
                bool use_16bit = (config_.transfer != hdr_transfer::sdr);
                frame_cache_ = gpu_frame_cache::get(
                    config_.gpu_index, device_, ogl_device_,
                    format_desc_.width, format_desc_.height, use_16bit);
                frame_cache_->add_consumer();
                CASPAR_LOG(info) << print() << L" Frame cache acquired (consumers="
                                 << frame_cache_->consumer_count() << L")";
            } catch (const std::exception& e) {
                CASPAR_LOG(warning) << print()
                    << L" Frame cache unavailable, falling back to CPU: " << e.what();
                frame_cache_.reset();
            }
        }

        // Create color space conversion pipeline if needed (before present thread starts
        // to avoid a data race — present_loop reads color_pipeline_ every frame)
        if (!adapter_mismatch_ && (config_.gamut != output_gamut::bt709 || config_.eotf != output_eotf::srgb)) {
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
                         << (adapter_mismatch_ ? L"GDI fallback (cross-adapter)" :
                             (device_->tier() == gpu_tier::pro && found) ? L"Pro (direct display)" :
                             device_->tier() == gpu_tier::pro ? L"Pro (fullscreen)" : L"Consumer (fullscreen)")
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
        create_info.presentMode      = pick_present_mode();
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

        // ─── Full-screen exclusive: DEFAULT mode ───────────────────────────
        // VK_FULL_SCREEN_EXCLUSIVE_DEFAULT_EXT lets the driver decide when to
        // grant exclusive display access. No explicit acquire/release needed.
        // The driver grants exclusivity when the borderless window fully covers
        // the display, and revokes it on focus loss — transparent to the app.
        VkSurfaceFullScreenExclusiveInfoEXT fse_info{};
        bool fse_chained = false;
        if (fse_hwnd_ && device_->has_extension(VK_EXT_FULL_SCREEN_EXCLUSIVE_EXTENSION_NAME)) {
            fse_info.sType               = VK_STRUCTURE_TYPE_SURFACE_FULL_SCREEN_EXCLUSIVE_INFO_EXT;
            fse_info.pNext               = const_cast<void*>(create_info.pNext);
            fse_info.fullScreenExclusive = VK_FULL_SCREEN_EXCLUSIVE_DEFAULT_EXT;
            create_info.pNext            = &fse_info;
            fse_chained = true;
        }

        auto result = vkCreateSwapchainKHR(device_->device(), &create_info, nullptr, &swapchain_.swapchain);
        if (result != VK_SUCCESS) {
            if (fse_chained) {
                // FSE chain caused failure — retry without it
                CASPAR_LOG(warning) << print() << L" Swapchain creation failed with FSE chain (result="
                                    << result << L"). Retrying without FSE.";
                create_info.pNext = fse_info.pNext; // restore original pNext
                result = vkCreateSwapchainKHR(device_->device(), &create_info, nullptr, &swapchain_.swapchain);
                if (result != VK_SUCCESS)
                    CASPAR_THROW_EXCEPTION(caspar_exception() << msg_info("Failed to create Vulkan swapchain"));
                fse_chained = false;
            } else {
                CASPAR_THROW_EXCEPTION(caspar_exception() << msg_info("Failed to create Vulkan swapchain"));
            }
        }

        fse_acquired_ = false; // No explicit acquire for DEFAULT mode

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
        if (count == 0) {
            CASPAR_THROW_EXCEPTION(caspar_exception()
                                   << msg_info("Surface reports zero available formats"));
        }
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

    VkPresentModeKHR pick_present_mode()
    {
        // MAILBOX is ideal for externally-paced playout:
        //  - The channel clock controls frame submission rate (25fps from send())
        //  - MAILBOX lets the GPU process frames immediately (no vsync queue stall)
        //  - Display still refreshes at vsync, picking the latest submitted frame
        //  - Since we submit at exactly display rate, every frame IS displayed
        //  - Eliminates fence wait spikes caused by FIFO backpressure + GPU contention
        // Fall back to FIFO_RELAXED (late frames present immediately instead of
        // waiting for next vsync) or FIFO (guaranteed available per spec).
        uint32_t count = 0;
        vkGetPhysicalDeviceSurfacePresentModesKHR(
            device_->physical_device(), swapchain_.surface, &count, nullptr);
        std::vector<VkPresentModeKHR> modes(count);
        vkGetPhysicalDeviceSurfacePresentModesKHR(
            device_->physical_device(), swapchain_.surface, &count, modes.data());

        // If present barrier is enabled, FIFO is required for sync group correctness
        if (present_barrier_enabled_)
            return VK_PRESENT_MODE_FIFO_KHR;

        for (auto m : modes) {
            if (m == VK_PRESENT_MODE_MAILBOX_KHR) {
                CASPAR_LOG(info) << print() << L" Using MAILBOX present mode (low-latency).";
                return VK_PRESENT_MODE_MAILBOX_KHR;
            }
        }
        for (auto m : modes) {
            if (m == VK_PRESENT_MODE_FIFO_RELAXED_KHR) {
                CASPAR_LOG(info) << print() << L" Using FIFO_RELAXED present mode.";
                return VK_PRESENT_MODE_FIFO_RELAXED_KHR;
            }
        }
        return VK_PRESENT_MODE_FIFO_KHR;
    }

    void create_sync_objects()
    {
        VkSemaphoreCreateInfo sem_info{};
        sem_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        VkFenceCreateInfo fence_info{};
        fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
            vkCreateSemaphore(device_->device(), &sem_info, nullptr, &swapchain_.frames[i].image_available);
            vkCreateSemaphore(device_->device(), &sem_info, nullptr, &swapchain_.frames[i].render_finished);
            vkCreateFence(device_->device(), &fence_info, nullptr, &swapchain_.frames[i].in_flight);
        }
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
        alloc_info.commandBufferCount = MAX_FRAMES_IN_FLIGHT;
        VkCommandBuffer cmd_buffers[MAX_FRAMES_IN_FLIGHT];
        vkAllocateCommandBuffers(device_->device(), &alloc_info, cmd_buffers);
        for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
            swapchain_.frames[i].cmd_buffer = cmd_buffers[i];
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

    void setup_nvapi()
    {
        if (!nvapi_)
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

        // EDID persistence: lock the monitor's EDID so the display survives cable disconnect
        if (config_.persist_edid) {
            nvapi_->persist_edid(config_.gpu_index, config_.output_index);
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
        SetThreadDescription(GetCurrentThread(), L"Vulkan Present");

        // Wait until buffer has enough frames to satisfy the configured delay
        // before starting to present. This introduces a fixed N-frame latency
        // to video output, allowing operators to compensate for downstream
        // pipeline delay (scalers, audio de-embedders, LED processors).
        const auto min_fill = static_cast<size_t>(config_.delay_frames + 1);

        // Frame pacer: With MAILBOX present mode, vkQueuePresentKHR returns
        // immediately (no vsync blocking), so the present loop must self-pace
        // to the target frame rate.
        //
        // Grid-aligned pacing: all outputs at the same frame rate converge to
        // the same time grid regardless of when they were started. This ensures
        // frame-accurate sync across multiple outputs on the same GPU — they
        // all compute the same next-deadline independently and submit within
        // the same vsync window.
        //
        // With FIFO mode (Quadro Sync / present_barrier), vkWaitForFences
        // provides hardware-locked pacing and the sleep is a no-op.
        const auto frame_interval_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::microseconds(static_cast<int64_t>(1'000'000.0 / format_desc_.fps)));

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
            buffer_cv_.notify_one(); // Wake send() if it's blocking on full buffer

            // Compute next grid-aligned deadline from steady_clock epoch.
            // Every output at this fps computes the same deadline value,
            // so they all wake and submit within microseconds of each other.
            {
                auto now_ns = std::chrono::steady_clock::now().time_since_epoch().count();
                auto interval = frame_interval_ns.count();
                auto next_tick_ns = ((now_ns / interval) + 1) * interval;
                std::this_thread::sleep_until(
                    std::chrono::steady_clock::time_point(
                        std::chrono::steady_clock::duration(next_tick_ns)));
            }

            caspar::timer frame_timer;
            present_frame(frame);
            auto frame_elapsed = frame_timer.elapsed();
            graph_->set_value("frame-time", frame_elapsed * format_desc_.fps * 0.5);

            auto tick = tick_timer_.elapsed() * format_desc_.fps * 0.5;
            graph_->set_value("tick-time", tick);
            tick_timer_.restart();

            // FPS counter
            ++fps_frame_count_;
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(now - fps_update_time_).count();
            if (elapsed >= 1.0) {
                double fps = fps_frame_count_ / elapsed;
                fps_frame_count_ = 0;
                fps_update_time_ = now;
                std::wstringstream stats;
                stats.precision(2);
                stats << std::fixed << print()
                      << L" " << format_desc_.width << L"x" << format_desc_.height
                      << L" fps: " << fps;
                graph_->set_text(stats.str());
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

            // Degenerate region — skip blit entirely (returns zero-extent region)
            if (sw <= 0 || sh <= 0 || dw <= 0 || dh <= 0)
                return blit;

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

    void try_acquire_fse()
    {
        if (fse_acquired_ || !fse_hwnd_ || !swapchain_.swapchain)
            return;
        if (!device_->has_extension(VK_EXT_FULL_SCREEN_EXCLUSIVE_EXTENSION_NAME))
            return;

        auto vkAcquireFSE = reinterpret_cast<PFN_vkAcquireFullScreenExclusiveModeEXT>(
            vkGetDeviceProcAddr(device_->device(), "vkAcquireFullScreenExclusiveModeEXT"));
        if (!vkAcquireFSE)
            return;

        auto result = vkAcquireFSE(device_->device(), swapchain_.swapchain);
        if (result == VK_SUCCESS) {
            fse_acquired_ = true;
            CASPAR_LOG(info) << print() << L" Full-screen exclusive re-acquired.";
        }
        // Silently ignore failure — will retry next frame
    }

    void release_fse(VkSwapchainKHR sc)
    {
        if (!fse_acquired_ || sc == VK_NULL_HANDLE)
            return;

        auto vkReleaseFSE = reinterpret_cast<PFN_vkReleaseFullScreenExclusiveModeEXT>(
            vkGetDeviceProcAddr(device_->device(), "vkReleaseFullScreenExclusiveModeEXT"));
        if (vkReleaseFSE) {
            vkReleaseFSE(device_->device(), sc);
        }
        fse_acquired_ = false;
    }

    void recreate_swapchain()
    {
        if (device_dead_)
            return;

        // Wait only for this consumer's queue to drain (not the entire device).
        // Other consumers on the same VkDevice continue submitting unblocked.
        VkResult wait_result = vkQueueWaitIdle(my_queue_);
        if (wait_result == VK_ERROR_DEVICE_LOST) {
            CASPAR_LOG(error) << print() << L" Device lost during swapchain recreation. Output permanently halted.";
            display_lost_ = true;
            device_dead_  = true;
            return;
        }

        for (auto& iv : swapchain_.image_views)
            vkDestroyImageView(device_->device(), iv, nullptr);
        swapchain_.image_views.clear();
        swapchain_.images.clear();

        auto old = swapchain_.swapchain;
        swapchain_.swapchain = VK_NULL_HANDLE;

        // Release FSE before destroying the old swapchain
        release_fse(old);

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
        if (device_dead_) {
            if (adapter_mismatch_ && fse_hwnd_) {
                present_frame_gdi(frame);
            }
            return;
        }

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

        // Get current frame's sync objects
        auto& frame_sync = swapchain_.frames[swapchain_.current_frame];

        // Wait for this frame slot's previous use to finish.
        // Use a finite timeout (2× frame period) to detect GPU stalls without
        // blocking indefinitely. With MAILBOX mode the fence is typically ready
        // immediately; with FIFO it may take up to one vsync period.
        const uint64_t fence_timeout_ns = static_cast<uint64_t>(2'000'000'000.0 / format_desc_.fps); // 2 frames in ns
        auto fence_result = vkWaitForFences(dev, 1, &frame_sync.in_flight, VK_TRUE, fence_timeout_ns);
        if (fence_result == VK_TIMEOUT) {
            // GPU severely behind — skip this frame rather than stall further
            graph_->set_tag(diagnostics::tag_severity::WARNING, "dropped-frame");
            return;
        }
        vkResetFences(dev, 1, &frame_sync.in_flight);

        // Acquire next swapchain image
        uint32_t image_index = 0;
        auto     result      = vkAcquireNextImageKHR(
            dev, swapchain_.swapchain, 1'000'000'000ULL, frame_sync.image_available, VK_NULL_HANDLE, &image_index);

        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
            // VK_SUBOPTIMAL means the acquire succeeded and signaled image_available.
            // We must reset the semaphore before retrying acquire after recreation,
            // otherwise the retry would double-signal an already-signaled semaphore (UB).
            vkDestroySemaphore(dev, frame_sync.image_available, nullptr);
            VkSemaphoreCreateInfo sem_info{};
            sem_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
            vkCreateSemaphore(dev, &sem_info, nullptr, &frame_sync.image_available);

            recreate_swapchain();
            if (display_lost_)
                return;
            // Retry acquire after recreation
            result = vkAcquireNextImageKHR(
                dev, swapchain_.swapchain, 1'000'000'000ULL, frame_sync.image_available, VK_NULL_HANDLE, &image_index);
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
        } else if (result == VK_TIMEOUT || result == VK_NOT_READY) {
            graph_->set_tag(diagnostics::tag_severity::WARNING, "dropped-frame");
            return;
        } else if (result == VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT) {
            // FSE was lost (alt-tab, focus change, etc.) — try to reacquire
            fse_acquired_ = false;
            try_acquire_fse();
        }

        // Record command buffer: copy frame data → swapchain image
        auto cmd = frame_sync.cmd_buffer;
        vkResetCommandBuffer(cmd, 0);

        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &begin_info);

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

        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
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

            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
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
            vkCmdClearColorImage(cmd, swapchain_.images[image_index],
                                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clear_color, 1, &range);
        } else if (frame_cache_ && frame_cache_->pool() && frame_generation_ > 0) {
            // Frame cache path: shared pool was populated in send() (same-GPU)
            // or cross-GPU transfer was done in send() via submit_frame().
            auto* pool = frame_cache_->pool();

            // Cross-GPU: perform transfer now if not yet done for this frame.
            // frame_generation_ is only incremented here for cross-GPU (not in send()),
            // so the present thread is the sole writer — no data race.
            if (frame_cache_->is_cross_gpu()) {
                ++frame_generation_;
                auto* cache = frame_cache_.get();
                cache->submit_frame(frame_generation_, [&, cache] {
                    bool transferred = false;
#ifdef CASPAR_CUDA_PEER_ENABLED
                    auto* cuda = cache->cuda_peer();
                    if (cuda && frame.texture()) {
                        auto ogl_tex = std::dynamic_pointer_cast<accelerator::ogl::texture>(frame.texture());
                        if (ogl_tex) {
                            try {
                                ogl_device_->dispatch_sync([&] {
                                    cuda->read_source(ogl_tex->id(), ogl_tex->width(), ogl_tex->height());
                                });
                                cuda->peer_copy();
                                cache->affinity_ctx()->dispatch_sync([&] {
                                    cuda->write_dest();
                                    pool->blit_from_texture(
                                        cuda->dest_texture(), format_desc_.width, format_desc_.height);
                                    pool->signal_gl();
                                });
                                transferred = true;
                            } catch (const std::exception& e) {
                                CASPAR_LOG(warning) << print()
                                    << L" CUDA peer failed, PBO fallback: " << e.what();
                            }
                        }
                    }
                    if (!transferred)
#endif
                    {
                        const auto& img = frame.image_data(0);
                        const auto* pixels = img.data();
                        if (pixels && img.size() > 0 && cache->affinity_ctx()) {
                            cache->affinity_ctx()->dispatch_sync([&] {
                                GLuint tex_id = cache->affinity_ctx()->upload_frame(
                                    pixels, format_desc_.width, format_desc_.height, format_desc_.width * 4);
                                pool->blit_from_texture(tex_id, format_desc_.width, format_desc_.height);
                                pool->signal_gl();
                            });
                            transferred = true;
                        }
                    }
                    if (transferred) {
                        pool->swap();
                    }
                });
            }

            // Transition shared image for transfer src.
            // IMPORTANT: oldLayout must be GENERAL (not UNDEFINED) to preserve GL-written data.
            // GL→VK interop images are always in GENERAL after glSignalSemaphoreEXT.
            // Using UNDEFINED would allow the driver to discard the image contents.
            VkImageMemoryBarrier src_barrier{};
            src_barrier.sType                           = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            src_barrier.oldLayout                       = VK_IMAGE_LAYOUT_GENERAL;
            src_barrier.newLayout                       = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            src_barrier.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
            src_barrier.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
            src_barrier.image                           = pool->current_vk_image();
            src_barrier.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
            src_barrier.subresourceRange.baseMipLevel   = 0;
            src_barrier.subresourceRange.levelCount     = 1;
            src_barrier.subresourceRange.baseArrayLayer = 0;
            src_barrier.subresourceRange.layerCount     = 1;
            src_barrier.srcAccessMask                   = 0;
            src_barrier.dstAccessMask                   = VK_ACCESS_TRANSFER_READ_BIT;

            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &src_barrier);

            auto blit_region = compute_blit_region(pool->width(), pool->height());
            vkCmdBlitImage(cmd, pool->current_vk_image(),
                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           blit_dest_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                           1, &blit_region, VK_FILTER_LINEAR);

            used_shared_pool = true;
            wrote_to_intermediate = color_convert_active;
        } else {
            // CPU fallback: upload pixel data via staging buffer
            // Always targets swapchain — BGRA8 pixels are incompatible with RGBA16F intermediate
            upload_frame_cpu(frame, swapchain_.images[image_index], cmd);
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

            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &cs_barrier);

            // Dispatch compute shader
            color_pipeline_->dispatch(cmd, color_pipeline_->width(), color_pipeline_->height());

            // Transition intermediate: GENERAL → TRANSFER_SRC (for final blit to swapchain)
            cs_barrier.oldLayout     = VK_IMAGE_LAYOUT_GENERAL;
            cs_barrier.newLayout     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            cs_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            cs_barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
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
            vkCmdBlitImage(cmd, color_pipeline_->image(),
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

        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr, 1, &present_barrier);

        if (vkEndCommandBuffer(cmd) != VK_SUCCESS) {
            CASPAR_LOG(error) << print() << L" vkEndCommandBuffer failed — skipping frame.";
            graph_->set_tag(diagnostics::tag_severity::WARNING, "dropped-frame");
            return;
        }

        // Submit — wait on timeline semaphore if zero-copy was used (replaces binary semaphore)
        VkSemaphore          wait_semaphores[2] = {frame_sync.image_available, VK_NULL_HANDLE};
        VkPipelineStageFlags wait_stages[2]     = {VK_PIPELINE_STAGE_TRANSFER_BIT,
                                                   VK_PIPELINE_STAGE_TRANSFER_BIT};
        uint32_t             wait_count         = 1;

        // Timeline semaphore values array (must match wait_semaphores count).
        // Binary semaphores use value 0 (ignored). Timeline uses the frame generation.
        uint64_t             wait_values[2]     = {0, 0};

        VkTimelineSemaphoreSubmitInfo timeline_submit{};
        timeline_submit.sType                     = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
        timeline_submit.waitSemaphoreValueCount   = wait_count; // Must match submit_info.waitSemaphoreCount
        timeline_submit.pWaitSemaphoreValues      = wait_values;
        timeline_submit.signalSemaphoreValueCount = 1;
        uint64_t signal_value                     = 0; // render_finished is binary
        timeline_submit.pSignalSemaphoreValues    = &signal_value;

        VkSubmitInfo submit_info{};
        submit_info.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.pNext                = &timeline_submit;
        submit_info.waitSemaphoreCount   = wait_count;
        submit_info.pWaitSemaphores      = wait_semaphores;
        submit_info.pWaitDstStageMask    = wait_stages;
        submit_info.commandBufferCount   = 1;
        submit_info.pCommandBuffers      = &frame_sync.cmd_buffer;
        submit_info.signalSemaphoreCount = 1;
        submit_info.pSignalSemaphores    = &frame_sync.render_finished;

        // Add timeline semaphore wait if shared pool was used.
        // The coordinator submit (in frame_cache) bridges the binary GL→VK semaphore
        // to the timeline, so ALL consumers can independently wait on the same value.
        if (used_shared_pool && frame_cache_ && frame_cache_->timeline_semaphore() != VK_NULL_HANDLE) {
            wait_semaphores[1] = frame_cache_->timeline_semaphore();
            wait_values[1]     = frame_generation_;  // Wait for our frame's transfer to complete
            wait_count         = 2;
            submit_info.waitSemaphoreCount        = wait_count;
            timeline_submit.waitSemaphoreValueCount = wait_count;
        }

        VkResult submit_result;
        VkResult present_result;

        // Lock per-queue mutex — if multiple consumers share a physical queue
        // (wrapping), this serializes their vkQueueSubmit/vkQueuePresentKHR calls.
        std::lock_guard<std::mutex> queue_lock(device_->queue_mutex_for(my_queue_idx_));

        submit_result = vkQueueSubmit(my_queue_, 1, &submit_info, frame_sync.in_flight);
        if (submit_result == VK_ERROR_DEVICE_LOST) {
            CASPAR_LOG(error) << print() << L" GPU device lost (TDR) during submit. Output halted.";
            display_lost_ = true;
            device_dead_  = true;
            return;
        }

        // Present
        VkPresentInfoKHR present_info{};
        present_info.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        present_info.waitSemaphoreCount = 1;
        present_info.pWaitSemaphores    = &frame_sync.render_finished;
        present_info.swapchainCount     = 1;
        present_info.pSwapchains        = &swapchain_.swapchain;
        present_info.pImageIndices      = &image_index;

        present_result = vkQueuePresentKHR(my_queue_, &present_info);

        if (present_result == VK_ERROR_OUT_OF_DATE_KHR || present_result == VK_SUBOPTIMAL_KHR) {
            recreate_swapchain();
        } else if (present_result == VK_ERROR_SURFACE_LOST_KHR) {
            display_lost_ = true;
            CASPAR_LOG(warning) << print() << L" Display disconnected during present.";
        } else if (present_result == VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT) {
            fse_acquired_ = false;
            try_acquire_fse();
        } else if (present_result == VK_ERROR_DEVICE_LOST) {
            CASPAR_LOG(error) << print() << L" GPU device lost (TDR) during present. Output halted.";
            display_lost_ = true;
            device_dead_  = true;
            return;
        }

        ++frames_presented_;

        // Advance to next frame-in-flight slot for the next iteration
        swapchain_.advance_frame();
    }

    void present_frame_gdi(const core::const_frame& frame)
    {
        // GDI fallback for indirect displays (IddCx/VDD) that don't expose DXGI outputs.
        // StretchDIBits acquires a global win32k.sys kernel lock, so with multiple outputs
        // the system becomes sluggish. Throttle to half frame rate to reduce lock contention
        // while still providing visual feedback for rehearsal/preview.
        ++gdi_frame_counter_;
        if (gdi_frame_counter_ % 2 != 0) {
            return;
        }

        const auto& img = frame.image_data(0);
        const auto* pixels = img.data();
        if (!pixels || img.size() == 0)
            return;

        auto src_w = static_cast<int>(frame.width());
        auto src_h = static_cast<int>(frame.height());
        if (src_w == 0 || src_h == 0)
            return;

        RECT client_rect;
        GetClientRect(fse_hwnd_, &client_rect);
        int dst_w = client_rect.right - client_rect.left;
        int dst_h = client_rect.bottom - client_rect.top;
        if (dst_w == 0 || dst_h == 0)
            return;

        BITMAPINFO bmi{};
        bmi.bmiHeader.biSize        = sizeof(BITMAPINFOHEADER);
        bmi.bmiHeader.biWidth       = src_w;
        bmi.bmiHeader.biHeight      = -src_h; // top-down
        bmi.bmiHeader.biPlanes      = 1;
        bmi.bmiHeader.biBitCount    = 32;
        bmi.bmiHeader.biCompression = BI_RGB;

        HDC hdc = GetDC(fse_hwnd_);
        if (hdc) {
            SetStretchBltMode(hdc, COLORONCOLOR);
            StretchDIBits(hdc,
                          0, 0, dst_w, dst_h,       // destination
                          0, 0, src_w, src_h,       // source
                          pixels, &bmi,
                          DIB_RGB_COLORS, SRCCOPY);
            ReleaseDC(fse_hwnd_, hdc);
        }
    }

    void upload_frame_cpu(const core::const_frame& frame, VkImage dst_image, VkCommandBuffer cmd)
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
            vkCmdClearColorImage(cmd, dst_image,
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

        vkCmdCopyBufferToImage(cmd, staging_buffer_, dst_image,
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
            case WM_MOUSEACTIVATE:
                return MA_NOACTIVATEANDEAT; // Don't steal focus on click
            case WM_SETCURSOR:
                SetCursor(nullptr); // Hide cursor on output windows
                return TRUE;
            case WM_ERASEBKGND: {
                // Paint solid black to ensure window is visually opaque
                // even before the first Vulkan present.
                RECT rc;
                GetClientRect(hwnd, &rc);
                FillRect(reinterpret_cast<HDC>(wParam), &rc,
                         static_cast<HBRUSH>(GetStockObject(BLACK_BRUSH)));
                return 1;
            }
            case WM_PAINT: {
                PAINTSTRUCT ps;
                HDC hdc = BeginPaint(hwnd, &ps);
                FillRect(hdc, &ps.rcPaint,
                         static_cast<HBRUSH>(GetStockObject(BLACK_BRUSH)));
                EndPaint(hwnd, &ps);
                return 0;
            }
        }
        return DefWindowProcW(hwnd, msg, wParam, lParam);
    }

    void launch_display_blanker()
    {
        // Launch display_blanker.exe once per process. Multiple vulkan_output consumers
        // may have display_blanker=true, but we only need one blanker process.
        // The blanker is persistent — it survives CasparCG crashes/restarts and must
        // be closed manually via its system tray icon. If already running (from a
        // previous CasparCG session), the single-instance mutex causes it to exit
        // immediately, which is fine.
        static std::once_flag blanker_once;

        std::call_once(blanker_once, [this] {
            // Build command line: match the same display-name pattern this consumer uses,
            // or fall back to --all (blanks all non-primary monitors).
            std::wstring exe_path;
            {
                // Look for display_blanker.exe next to casparcg.exe
                wchar_t module_path[MAX_PATH] = {};
                GetModuleFileNameW(nullptr, module_path, MAX_PATH);
                exe_path = module_path;
                auto last_sep = exe_path.find_last_of(L"\\/");
                if (last_sep != std::wstring::npos)
                    exe_path = exe_path.substr(0, last_sep + 1);
                exe_path += L"display_blanker.exe";
            }

            if (GetFileAttributesW(exe_path.c_str()) == INVALID_FILE_ATTRIBUTES) {
                CASPAR_LOG(warning) << print()
                    << L" display_blanker.exe not found next to casparcg.exe. Desktop blanking disabled.";
                return;
            }

            std::wstring cmd = L"\"" + exe_path + L"\"";

            if (!config_.display_name.empty()) {
                cmd += L" --match \"" + config_.display_name + L"\"";
            } else {
                cmd += L" --all";
            }

            STARTUPINFOW si{};
            si.cb = sizeof(si);

            PROCESS_INFORMATION pi{};

            // CreateProcessW needs a mutable buffer for the command line
            std::vector<wchar_t> cmd_buf(cmd.begin(), cmd.end());
            cmd_buf.push_back(L'\0');

            if (CreateProcessW(nullptr, cmd_buf.data(), nullptr, nullptr, FALSE,
                               0, nullptr, nullptr, &si, &pi)) {
                CloseHandle(pi.hProcess);
                CloseHandle(pi.hThread);
                CASPAR_LOG(info) << print() << L" Display blanker launched (PID " << pi.dwProcessId << L")";
                // Small delay to let blanker windows appear before our FSE window
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
            } else {
                auto err = GetLastError();
                if (err != ERROR_ALREADY_EXISTS) {
                    CASPAR_LOG(warning) << print()
                        << L" Failed to launch display_blanker.exe (error " << err << L")";
                }
            }
        });
    }

    void create_fse_window()
    {
        // Create the FSE window on a dedicated message thread so it always pumps messages.
        // Windows routes messages to the thread that created the window, so if that thread
        // doesn't pump, clicking the window causes a "Not Responding" hang.
        std::promise<HWND> hwnd_promise;
        auto hwnd_future = hwnd_promise.get_future();

        fse_msg_thread_ = std::thread([this, &hwnd_promise] {
            SetThreadDescription(GetCurrentThread(), L"Vulkan FSE MsgPump");

            WNDCLASSEXW wc{};
            wc.cbSize        = sizeof(WNDCLASSEXW);
            wc.style         = CS_HREDRAW | CS_VREDRAW;
            wc.lpfnWndProc   = fse_wnd_proc;
            wc.hInstance     = GetModuleHandle(nullptr);
            wc.hbrBackground = static_cast<HBRUSH>(GetStockObject(BLACK_BRUSH));
            wc.lpszClassName = L"CasparVulkanOutput";
            if (!RegisterClassExW(&wc)) {
                if (GetLastError() != ERROR_CLASS_ALREADY_EXISTS) {
                    CASPAR_LOG(warning) << print() << L" RegisterClassExW failed: " << GetLastError();
                }
            }

            // Enumerate monitors and select by display-name or output_index
            struct monitor_enum_data
            {
                int target_index;
                int current_index;
                RECT rect;
                bool found;
                bool adapter_mismatch; // true if display is on a different GPU (e.g. VDD/IddCx)
            };
            monitor_enum_data data{config_.output_index, 0, {}, false, false};

            // If display-name is configured, try matching by device name first
            // Log all attached displays for diagnostics
            {
                DISPLAY_DEVICEW dd_log{};
                dd_log.cb = sizeof(dd_log);
                for (DWORD i = 0; EnumDisplayDevicesW(nullptr, i, &dd_log, 0); ++i) {
                    if (dd_log.StateFlags & DISPLAY_DEVICE_ATTACHED_TO_DESKTOP) {
                        DISPLAY_DEVICEW mon_log{};
                        mon_log.cb = sizeof(mon_log);
                        std::wstring mon_str = L"(none)";
                        if (EnumDisplayDevicesW(dd_log.DeviceName, 0, &mon_log, 0)) {
                            mon_str = mon_log.DeviceString;
                        }
                        CASPAR_LOG(debug) << print() << L" Display[" << i << L"]: "
                                          << dd_log.DeviceName << L" | " << dd_log.DeviceString
                                          << L" | Monitor=" << mon_str
                                          << L" | ID=" << mon_log.DeviceID;
                    }
                    dd_log.cb = sizeof(dd_log);
                }
            }

            if (!config_.display_name.empty()) {
                DISPLAY_DEVICEW dd{};
                dd.cb = sizeof(dd);
                int match_count = 0;
                for (DWORD i = 0; EnumDisplayDevicesW(nullptr, i, &dd, 0); ++i) {
                    if (!(dd.StateFlags & DISPLAY_DEVICE_ATTACHED_TO_DESKTOP)) {
                        dd.cb = sizeof(dd);
                        continue;
                    }
                    // Get the monitor name for this adapter
                    DISPLAY_DEVICEW mon{};
                    mon.cb = sizeof(mon);
                    if (EnumDisplayDevicesW(dd.DeviceName, 0, &mon, 0)) {
                        std::wstring dev_str(mon.DeviceString);
                        std::wstring dev_id(mon.DeviceID);
                        // Case-insensitive substring match on DeviceString or DeviceID
                        if (boost::icontains(dev_str, config_.display_name) ||
                            boost::icontains(dev_id, config_.display_name)) {
                            match_count++;
                            if (match_count == config_.output_index || config_.output_index <= 1) {
                                // Get this monitor's position
                                DEVMODEW dm{};
                                dm.dmSize = sizeof(dm);
                                if (EnumDisplaySettingsW(dd.DeviceName, ENUM_CURRENT_SETTINGS, &dm)) {
                                    data.rect.left   = dm.dmPosition.x;
                                    data.rect.top    = dm.dmPosition.y;
                                    data.rect.right  = dm.dmPosition.x + static_cast<LONG>(dm.dmPelsWidth);
                                    data.rect.bottom = dm.dmPosition.y + static_cast<LONG>(dm.dmPelsHeight);
                                    data.found = true;
                                    // Check if this display's adapter matches our Vulkan GPU
                                    std::wstring adapter_str(dd.DeviceString);
                                    if (!boost::icontains(adapter_str, L"NVIDIA")) {
                                        data.adapter_mismatch = true;
                                    }
                                    CASPAR_LOG(info) << print() << L" Matched display-name \""
                                                     << config_.display_name << L"\" → "
                                                     << mon.DeviceString << L" (" << dd.DeviceName << L")"
                                                     << (data.adapter_mismatch ? L" [GDI fallback]" : L"");
                                }
                                break;
                            }
                        }
                    }
                    dd.cb = sizeof(dd);
                }
            }

            // Fallback: select by monitor index
            if (!data.found) {
                EnumDisplayMonitors(
                    nullptr, nullptr,
                    [](HMONITOR, HDC, LPRECT rect, LPARAM lparam) -> BOOL {
                        auto* d = reinterpret_cast<monitor_enum_data*>(lparam);
                        d->current_index++;
                        if (d->current_index == d->target_index) {
                            d->rect  = *rect;
                            d->found = true;
                            return FALSE;
                        }
                        return TRUE;
                    },
                    reinterpret_cast<LPARAM>(&data));
            }

            int x = 0, y = 0;
            int w = format_desc_.width;
            int h = format_desc_.height;

            if (data.found) {
                x = data.rect.left;
                y = data.rect.top;
                w = data.rect.right - data.rect.left;
                h = data.rect.bottom - data.rect.top;
                CASPAR_LOG(info) << print() << L" Fullscreen window on monitor " << config_.output_index << L" at ("
                                 << x << L"," << y << L") " << w << L"x" << h;
            } else {
                CASPAR_LOG(warning) << print() << L" Monitor " << config_.output_index
                                    << L" not found. Window will be created off-screen (no output until monitor is available).";
                // Place window far off-screen so it doesn't cover the operator's display.
                // Vulkan swapchain will still be created on this window; frames are rendered
                // but not visible until the monitor appears in the desktop topology.
                x = -32000;
                y = -32000;
            }

            HWND hwnd = CreateWindowExW(WS_EX_TOPMOST | WS_EX_TOOLWINDOW | WS_EX_APPWINDOW,
                                        L"CasparVulkanOutput", L"CasparCG Vulkan Output",
                                        WS_POPUP | WS_VISIBLE, x, y, w, h,
                                        nullptr, nullptr, GetModuleHandle(nullptr), nullptr);

            // Attach to the foreground thread's input queue so that
            // SetForegroundWindow works reliably from a background thread.
            DWORD fg_thread = 0;
            {
                HWND fg_wnd = GetForegroundWindow();
                if (fg_wnd)
                    fg_thread = GetWindowThreadProcessId(fg_wnd, nullptr);
            }
            DWORD our_thread = GetCurrentThreadId();
            bool attached = false;
            if (fg_thread && fg_thread != our_thread) {
                attached = !!AttachThreadInput(our_thread, fg_thread, TRUE);
            }

            SetForegroundWindow(hwnd);
            BringWindowToTop(hwnd);
            SetWindowPos(hwnd, HWND_TOPMOST, x, y, w, h,
                         SWP_SHOWWINDOW);
            ShowWindow(hwnd, SW_SHOW);

            if (attached) {
                AttachThreadInput(our_thread, fg_thread, FALSE);
            }

            // Pump all pending messages so activation/focus messages are processed
            // before the main thread tries to acquire FSE.  FSE requires the window
            // to be the actual foreground window, not just queued for activation.
            {
                MSG init_msg;
                while (PeekMessageW(&init_msg, nullptr, 0, 0, PM_REMOVE)) {
                    TranslateMessage(&init_msg);
                    DispatchMessageW(&init_msg);
                }
            }

            adapter_mismatch_ = data.adapter_mismatch;
            hwnd_promise.set_value(hwnd);

            // Message pump — runs until WM_QUIT or fse_msg_running_ is false
            MSG msg;
            while (fse_msg_running_) {
                if (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
                    if (msg.message == WM_QUIT)
                        break;
                    TranslateMessage(&msg);
                    DispatchMessageW(&msg);
                } else {
                    MsgWaitForMultipleObjects(0, nullptr, FALSE, 50, QS_ALLINPUT);
                }
            }

            // Destroy window from the thread that created it (Win32 requirement)
            if (hwnd) {
                DestroyWindow(hwnd);
            }
        });

        fse_hwnd_ = hwnd_future.get();
    }

    void destroy_resources()
    {
        running_ = false;
        buffer_cv_.notify_all();
        if (present_thread_.joinable())
            present_thread_.join();

        if (device_) {
            auto dev = device_->device();

            // Wait for GPU to finish with a hard timeout to prevent unkillable process.
            // NVIDIA driver can hang indefinitely in vkQueueWaitIdle when display timing
            // is disrupted (disconnected, TDR, or GL contention deadlock).
            // If timeout fires, skip ALL Vulkan resource destruction.
            bool gpu_stuck = device_dead_.load();
            if (!gpu_stuck) {
                auto idle_future = std::async(std::launch::async, [this] {
                    return vkQueueWaitIdle(my_queue_);
                });
                if (idle_future.wait_for(std::chrono::seconds(2)) == std::future_status::timeout) {
                    CASPAR_LOG(warning) << print() << L" vkQueueWaitIdle timed out (2s) — skipping Vulkan cleanup.";
                    gpu_stuck = true;
                } else {
                    idle_future.get(); // Collect result (ignore errors during shutdown)
                }
            }

            if (!gpu_stuck) {
                if (staging_buffer_ != VK_NULL_HANDLE) {
                    vkDestroyBuffer(dev, staging_buffer_, nullptr);
                    vkFreeMemory(dev, staging_memory_, nullptr);
                }

                for (auto& iv : swapchain_.image_views)
                    vkDestroyImageView(dev, iv, nullptr);

                // Release FSE before swapchain destruction
                release_fse(swapchain_.swapchain);

                if (swapchain_.swapchain != VK_NULL_HANDLE)
                    vkDestroySwapchainKHR(dev, swapchain_.swapchain, nullptr);
                for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
                    if (swapchain_.frames[i].image_available != VK_NULL_HANDLE)
                        vkDestroySemaphore(dev, swapchain_.frames[i].image_available, nullptr);
                    if (swapchain_.frames[i].render_finished != VK_NULL_HANDLE)
                        vkDestroySemaphore(dev, swapchain_.frames[i].render_finished, nullptr);
                    if (swapchain_.frames[i].in_flight != VK_NULL_HANDLE)
                        vkDestroyFence(dev, swapchain_.frames[i].in_flight, nullptr);
                }
                if (swapchain_.cmd_pool != VK_NULL_HANDLE)
                    vkDestroyCommandPool(dev, swapchain_.cmd_pool, nullptr);
                if (swapchain_.surface != VK_NULL_HANDLE)
                    vkDestroySurfaceKHR(device_->instance(), swapchain_.surface, nullptr);
            } else {
                CASPAR_LOG(warning) << print() << L" GPU stuck — Vulkan resources leaked (OS will reclaim on exit).";
            }

            // Destroy resources that reference VkDevice handles BEFORE destroying the device.
            if (!gpu_stuck) {
                color_pipeline_.reset();
                if (frame_cache_) {
                    frame_cache_->remove_consumer();
                    frame_cache_.reset();
                }
                device_.reset();
            } else {
                // Release without calling destructors that touch VkDevice.
                // The leaked async thread holds a VkDevice reference that prevents kernel cleanup,
                // but the OS will reclaim all GPU resources when the process terminates.
                color_pipeline_.release();
                frame_cache_.reset(); // shared_ptr — safe to reset even if stuck
                device_.reset();
            }
        }

        // Remove injected EDID before releasing NvAPI
        if (injected_edid_display_id_ != 0 && nvapi_) {
            nvapi_->remove_edid(config_.gpu_index, injected_edid_display_id_);
            injected_edid_display_id_ = 0;
        }
        nvapi_.reset();

        if (fse_hwnd_) {
            // Signal message thread to stop, then post WM_NULL to wake MsgWait.
            // The message thread owns the window and calls DestroyWindow before exiting
            // (Win32 requires DestroyWindow from the creating thread).
            fse_msg_running_ = false;
            PostMessageW(fse_hwnd_, WM_NULL, 0, 0);
            if (fse_msg_thread_.joinable())
                fse_msg_thread_.join();
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
    std::shared_ptr<vulkan_device>   device_;
    std::shared_ptr<gpu_frame_cache> frame_cache_;
    std::unique_ptr<nvapi_helpers>   nvapi_;
    std::unique_ptr<color_convert_pipeline> color_pipeline_; // Color space conversion (compute)
    swapchain_resources              swapchain_{};
    HWND                             fse_hwnd_ = nullptr;
    VkDisplayKHR                     display_handle_ = VK_NULL_HANDLE;
    uint32_t                         my_queue_idx_ = 0; // Exclusive queue for this consumer
    VkQueue                          my_queue_ = VK_NULL_HANDLE;

    // Frame buffer
    std::queue<core::const_frame>    buffer_;
    std::mutex                       buffer_mutex_;
    std::condition_variable          buffer_cv_;

    // Present thread
    std::thread                      present_thread_;
    std::atomic<bool>                running_{false};

    // FSE window message thread
    std::thread                      fse_msg_thread_;
    std::atomic<bool>                fse_msg_running_{true};

    // Display hot-plug
    std::atomic<bool>                display_lost_{false};
    std::atomic<bool>                device_dead_{false}; // TDR — device permanently invalid
    bool                             adapter_mismatch_{false}; // Display on different GPU — no Vulkan presentation
    uint64_t                         hotplug_retry_counter_ = 0;
    std::atomic<uint64_t>            frames_presented_{0};
    uint64_t                         frame_generation_{0};  // Monotonic counter for frame cache coordination

    // Present barrier
    bool                             present_barrier_enabled_ = false;
    bool                             fse_acquired_ = false; // VK_EXT_full_screen_exclusive acquired

    // Output identification
    std::atomic<int>                 identify_frames_remaining_{0};

    // Timing
    caspar::timer                    tick_timer_;

    // FPS counter
    std::chrono::steady_clock::time_point fps_update_time_ = std::chrono::steady_clock::now();
    int                              fps_frame_count_ = 0;

    // CPU staging
    VkBuffer                         staging_buffer_      = VK_NULL_HANDLE;
    VkDeviceMemory                   staging_memory_      = VK_NULL_HANDLE;
    size_t                           staging_buffer_size_ = 0;

    // EDID emulation state
    uint32_t                         injected_edid_display_id_ = 0;

    // GDI fallback frame throttle (reduces win32k.sys lock contention)
    uint64_t                         gdi_frame_counter_ = 0;
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
