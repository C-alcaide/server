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
#include <cuda_runtime.h>
#endif
#include "../util/nvapi_helpers.h"
#include "../util/color_convert_pipeline.h"

#include <accelerator/ogl/image/image_mixer.h>
#include <accelerator/ogl/util/device.h>
#include <accelerator/ogl/util/texture.h>

#include <accelerator/vulkan/util/texture_wrapper.h>
#include <accelerator/vulkan/util/texture.h>

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

#include <dwmapi.h>

#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
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

// ─── TDR self-termination watchdog ──────────────────────────────────────
// After a TDR (VK_ERROR_DEVICE_LOST), Vulkan calls may block indefinitely
// inside the kernel-mode NVIDIA driver, preventing the process from exiting.
// This creates a zombie process that holds file locks and can only be freed
// by a reboot or driver reset (Win+Ctrl+Shift+B).
//
// Solution: When any consumer detects DEVICE_LOST, start a one-shot watchdog
// thread that calls TerminateProcess() after a grace period.  The grace period
// allows clean shutdown to complete if possible.  The watchdog thread never
// touches GPU APIs so it cannot be blocked by the driver.
namespace {
std::once_flag tdr_watchdog_flag;

void start_tdr_watchdog(int grace_seconds = 10)
{
    std::call_once(tdr_watchdog_flag, [grace_seconds] {
        std::thread([grace_seconds] {
            SetThreadDescription(GetCurrentThread(), L"TDR Watchdog");
            CASPAR_LOG(error) << L"[vulkan_output] TDR detected — process will be forcefully terminated in "
                              << grace_seconds << L" seconds if shutdown does not complete.";
            std::this_thread::sleep_for(std::chrono::seconds(grace_seconds));
            CASPAR_LOG(fatal) << L"[vulkan_output] TDR watchdog: clean shutdown did not complete in "
                              << grace_seconds << L"s — calling TerminateProcess to prevent zombie.";
            boost::log::core::get()->flush();
            TerminateProcess(GetCurrentProcess(), 1);
        }).detach();
    });
}
} // namespace

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

    // Sets Per-Monitor DPI awareness on the calling thread.
    // Returns the previous DPI awareness context (to restore later) or nullptr.
    // NOTE: We use thread-level DPI awareness only (not process-level) to avoid
    // interfering with OpenGL contexts — the NVIDIA OGL driver (nvoglv64.dll)
    // crashes with 0xc0000409 when process-level DPI awareness changes under it.
    static DPI_AWARENESS_CONTEXT set_thread_dpi_awareness()
    {
        using SetThreadDpiAwarenessContextFn = DPI_AWARENESS_CONTEXT(WINAPI*)(DPI_AWARENESS_CONTEXT);
        static auto fn = reinterpret_cast<SetThreadDpiAwarenessContextFn>(
            GetProcAddress(GetModuleHandleW(L"user32.dll"), "SetThreadDpiAwarenessContext"));
        if (fn)
            return fn(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
        return nullptr;
    }

    static void restore_thread_dpi_awareness(DPI_AWARENESS_CONTEXT prev)
    {
        if (!prev) return;
        using SetThreadDpiAwarenessContextFn = DPI_AWARENESS_CONTEXT(WINAPI*)(DPI_AWARENESS_CONTEXT);
        static auto fn = reinterpret_cast<SetThreadDpiAwarenessContextFn>(
            GetProcAddress(GetModuleHandleW(L"user32.dll"), "SetThreadDpiAwarenessContext"));
        if (fn)
            fn(prev);
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
        mixer_auto_color_convert_ = channel_info.auto_color_convert;

        // Clamp delay_frames to buffer_depth - 1 to prevent the present loop
        // from stalling permanently (min_fill = delay_frames + 1 must be ≤ buffer_depth).
        if (config_.delay_frames >= config_.buffer_depth) {
            CASPAR_LOG(warning) << print() << L" delay (" << config_.delay_frames
                                << L") >= buffer-depth (" << config_.buffer_depth
                                << L"). Clamping delay to " << (config_.buffer_depth - 1);
            config_.delay_frames = config_.buffer_depth - 1;
        }

        // Clamp delay_ms to one frame period — larger values would cause the
        // present loop to miss frames and should use delay_frames instead.
        if (config_.delay_ms < 0.0)
            config_.delay_ms = 0.0;
        double max_delay_ms = 1000.0 / format_desc.fps;
        if (config_.delay_ms > max_delay_ms) {
            CASPAR_LOG(warning) << print() << L" delay-ms (" << config_.delay_ms
                                << L") exceeds one frame period (" << max_delay_ms
                                << L" ms). Clamping.";
            config_.delay_ms = max_delay_ms;
        }

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

        diag_sends_.fetch_add(1, std::memory_order_relaxed);

        // ─── VK-native path (Vulkan mixer → Vulkan output, same GPU) ─────────
        // When the mixer is Vulkan, frame.texture() is a texture_wrapper holding
        // an exportable VkImage.  Skip the entire GL↔VK interop chain — no
        // shared_texture_pool, no interop_context, no GL semaphores.
        // The VkImage memory is imported on the output's VkDevice via Win32 HANDLE
        // and blitted directly to the swapchain in present_frame().
        bool vk_texture_path = false;
        if (gate_open_ && frame.texture() && !adapter_mismatch_) {
            auto vk_wrapper = std::dynamic_pointer_cast<accelerator::vulkan::texture_wrapper>(frame.texture());
            if (vk_wrapper) {
                vk_texture_path = true;
                // No GPU transfer needed — the mixer's VkImage is on the same
                // physical GPU and will be imported in present_frame().
            } else if (!send_diag_logged_) {
                CASPAR_LOG(warning) << print() << L" send(): frame.texture() is not a texture_wrapper"
                                    << L" (typeid=" << typeid(*frame.texture()).name() << L")";
                send_diag_logged_ = true;
            }
        } else if (send_diag_counter_++ % 300 == 0) {
            CASPAR_LOG(trace) << print() << L" send(): VK path skipped"
                             << L" gate=" << gate_open_
                             << L" tex=" << (frame.texture() ? L"yes" : L"no")
                             << L" mismatch=" << adapter_mismatch_;
        }

        // Zero-copy path: blit OGL texture into shared pool via frame cache.
        // The frame cache ensures only one transfer per GPU per frame — the first
        // consumer to call submit_frame() does the actual blit, others wait.
        // For cross-GPU, the cache coordinates CUDA peer DMA or PBO upload.
        // Skip frame cache operations until the startup gate opens — doing a
        // coordinator vkQueueSubmit while a peer consumer is still creating
        // swapchains on the same VkDevice can trigger TDR.
        // Skip when VK texture path is active — no GL interop needed.
        if (!vk_texture_path && gate_open_ && frame_cache_ && !frame_cache_->is_cross_gpu() && frame.texture()) {
            auto ogl_tex = std::dynamic_pointer_cast<accelerator::ogl::texture>(frame.texture());
            if (ogl_tex) {
                auto* cache = frame_cache_.get();
                ++frame_generation_;
                // Synchronous transfer: blit + GL signal + coordinator submit all
                // happen HERE, so the timeline semaphore is guaranteed signaled
                // before the frame enters the buffer.  Eliminates the pump thread
                // race where present_frame could wait on a timeline value the pump
                // hadn't signaled yet.
                cache->submit_frame(frame_generation_, [cache, ogl_tex, ogl_dev = ogl_device_.get()] {
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

        // Backpressure: When the buffer is full, drop the oldest frame rather
        // than blocking the channel tick thread.  This ensures a slow consumer
        // (e.g. cross-GPU with CUDA peer transfer) doesn't throttle the whole
        // channel — the fastest consumer's vsync becomes the reference clock.
        // The present thread will skip the dropped frame, producing a brief
        // stutter on the slow output, which is preferable to halving the frame
        // rate on ALL outputs.
        {
            std::unique_lock<std::mutex> lock(buffer_mutex_);
            if (!running_)
                return caspar::make_ready_future(true);
            if (buffer_.size() >= static_cast<size_t>(config_.buffer_depth)) {
                // Buffer full — drop oldest to make room
                buffer_.pop();
                graph_->set_tag(diagnostics::tag_severity::WARNING, "dropped-frame");
                diag_drops_.fetch_add(1, std::memory_order_relaxed);
            }
            buffer_.push({std::move(frame), frame_generation_.load(std::memory_order_relaxed)});
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

    bool needs_cpu_frame_data() const override { return false; }

    int index() const override
    {
        return 500 + config_.gpu_index * 10 + config_.output_index;
    }

    bool has_synchronization_clock() const override { return false; }

    core::av_pipeline_info av_pipeline() const override
    {
        core::av_pipeline_info info;
        std::lock_guard<std::mutex> lock(config_mutex_);
        info.has_video               = true;
        info.video_depth_frames      = config_.delay_frames + 1;
        info.video_delay_ms          = config_.delay_ms;
        info.video_delay_adjustable  = true;
        return info;
    }

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
        s["vulkan-output/delay-ms"]        = std::to_wstring(config_.delay_ms);

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
            setup_nvapi();           // EDID auto-detect + hardware HDR probe (before swapchain for format selection)
            setup_present_barrier(); // Must be before create_swapchain so the barrier struct is chained
            create_swapchain();
            if (!hw_hdr_active_)
                set_hdr_metadata();  // Only set Vulkan HDR metadata when NOT using hardware HDR
            create_sync_objects();
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

        // Color space conversion pipeline: DISABLED.
        // The mixer (via auto_color_convert or manual MIXER COLOR commands) already
        // converts to the channel's target color space (gamut + transfer + tone mapping).
        // The vulkan output consumer receives already-converted frames. Running
        // color_convert.comp on top would double-convert.
        // Hardware HDR (NvAPI UHDA) is also handled upstream when active.
        //
        // Currently config_.gamut and config_.eotf have NO effect — the channel's
        // color-space and color-transfer determine all pixel values in the framebuffer.
        //
        // FUTURE RE-ENABLEMENT PLAN:
        // To properly support per-consumer gamut adaptation (e.g. channel=bt2020 but
        // vulkan-output display is P3), this pipeline should be re-enabled with the
        // following architecture:
        //   1. Channel color-space remains the mixer's working space (determines what
        //      sources get converted TO during compositing).
        //   2. This compute shader converts FROM the channel gamut TO config_.gamut/eotf
        //      (the display's native gamut + transfer).
        //   3. The shader input gamut comes from channel_info (passed at initialize()),
        //      not from config — so it always matches what the mixer actually output.
        //   4. Enable only when config_.gamut != channel gamut OR config_.eotf != channel
        //      transfer (skip the no-op identity conversion).
        //   5. tone_map_op applies here as a display-specific artistic transform AFTER
        //      the gamut conversion (e.g. soft-knee HLG OOTF for this display's peak nits).
        //   6. When hw_hdr_active_ (NvAPI UHDA), the display engine already handles
        //      gamut mapping at scanout — skip this pipeline entirely.
        if (false && !adapter_mismatch_ && !hw_hdr_active_ && config_.tone_map_op != 0) {
            try {
                color_pipeline_ = std::make_unique<color_convert_pipeline>(
                    *device_, format_desc_.width, format_desc_.height);
                // Use display_peak_luminance for tone-mapping, max_cll for PQ encoding
                float lum = config_.tone_map_op != 0
                    ? config_.display_peak_luminance
                    : static_cast<float>(config_.max_cll);
                color_pipeline_->update_config(config_.gamut, config_.eotf,
                                               lum, config_.tone_map_op);
                CASPAR_LOG(info) << print() << L" Color space conversion enabled.";
            } catch (const std::exception& e) {
                CASPAR_LOG(error) << print()
                    << L" Failed to create color conversion pipeline: " << e.what();
                color_pipeline_.reset();
            }
        }

        // Signal that this consumer's heavy init (window + surface + swapchain)
        // is complete.  The present thread will wait for all peers on this GPU
        // before starting to submit frames — preventing TDR caused by one
        // consumer presenting while another is still creating its swapchain.
        vk_device_manager::consumer_ready(config_.gpu_index);

        // Start present thread (it will gate on wait_all_ready before presenting)
        running_ = true;
        present_thread_ = std::thread([this] { present_loop(); });

        // Diagnostic watchdog: every second, if no frame completed since the
        // last check, log the last reached stage of present_frame() so we can
        // see where the present thread is wedged without spamming the log
        // when frames are flowing.
        watchdog_thread_ = std::thread([this] {
            SetThreadDescription(GetCurrentThread(), L"VK Present Watchdog");
            uint64_t last_submits = 0;
            while (running_) {
                std::this_thread::sleep_for(std::chrono::seconds(1));
                if (!running_) break;
                uint64_t submits = watchdog_submits_.load(std::memory_order_relaxed);
                if (submits == last_submits) {
                    CASPAR_LOG(warning) << print() << L" WATCHDOG: no frame submitted in 1s. "
                                        << L"stage=" << diag_stage_.load(std::memory_order_relaxed)
                                        << L" gate=" << (gate_open_ ? 1 : 0)
                                        << L" running=" << running_.load()
                                        << L" buf=" << diag_buf_size_last_
                                        << L" pres=" << diag_presents_.load()
                                        << L" submits=" << submits
                                        << L" gen=" << frame_generation_;
                }
                last_submits = submits;
            }
        });

        // Show identify overlay if configured
        if (config_.identify_on_start) {
            identify_frames_remaining_ = static_cast<int>(format_desc_.fps * 3); // 3 seconds
        }

        CASPAR_LOG(info) << print() << L" initialized. Tier: "
                         << (adapter_mismatch_ ? L"GDI fallback (cross-adapter)" :
                             (device_->tier() == gpu_tier::pro && found) ? L"Pro (direct display)" :
                             device_->tier() == gpu_tier::pro ? L"Pro (fullscreen)" : L"Consumer (fullscreen)")
                         << (config_.delay_frames > 0 || config_.delay_ms > 0.0
                                 ? L" Delay: " + std::to_wstring(config_.delay_frames) + L" frames"
                                   + (config_.delay_ms > 0.0 ? L" + " + std::to_wstring(config_.delay_ms) + L" ms" : L"")
                                 : L"");
    }

    void create_swapchain()
    {
        // Ensure Per-Monitor DPI awareness on this thread so that
        // vkGetPhysicalDeviceSurfaceCapabilitiesKHR returns physical pixel
        // dimensions (e.g. 3840×2160) instead of DPI-scaled logical coordinates.
        auto prev_dpi_ctx = set_thread_dpi_awareness();

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
        if (caps.maxImageCount > 0) {
            if (image_count > caps.maxImageCount) {
                CASPAR_LOG(warning) << print() << L" Requested swapchain image count "
                                    << image_count << L" exceeds maxImageCount "
                                    << caps.maxImageCount << L". Clamping.";
            }
            image_count = (std::min)(image_count, caps.maxImageCount);
        }

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

        restore_thread_dpi_awareness(prev_dpi_ctx);
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

        if (hw_hdr_active_) {
            // Hardware HDR (NvAPI UHDA): source is linear scRGB FP16.
            // Prefer RGBA16F with extended sRGB linear color space (VK_EXT_swapchain_colorspace).
            // The display engine does PQ encoding + gamut mapping at scanout.
            for (const auto& f : formats) {
                if (f.format == VK_FORMAT_R16G16B16A16_SFLOAT &&
                    f.colorSpace == VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT)
                    return f;
            }
            // Fallback: any FP16 format (color space may be wrong but display engine overrides)
            for (const auto& f : formats) {
                if (f.format == VK_FORMAT_R16G16B16A16_SFLOAT)
                    return f;
            }
            // Last resort: 10-bit (loses values >1.0, HDR highlights clamped to 80 nits)
            CASPAR_LOG(warning) << print()
                << L" No RGBA16F surface format available for hardware HDR. "
                << L"HDR highlights above 80 nits will be clipped.";
        }

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

        // Join the software sync barrier.  All consumers in the same sync group
        // will synchronize their present calls via a cv-based barrier in
        // vk_device_manager.  This provides best-effort frame-lock without
        // requiring Quadro Sync hardware.
        //
        // NOTE: VK_NV_present_barrier is NOT used here because it only works
        // across swapchains on the SAME VkDevice.  Cross-GPU setups (different
        // physical GPUs = different VkDevices) cause the driver to deadlock
        // waiting for a peer that never arrives → TDR.
        sync_group_token_ = vk_device_manager::sync_group_join(config_.sync_group);
        CASPAR_LOG(info) << print() << L" Joined software sync group " << config_.sync_group;
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

        // Hardware HDR: let the display engine perform PQ encoding + gamut mapping
        // Source stays linear scRGB FP16 — no compute shader EOTF needed.
        if (config_.transfer == hdr_transfer::pq || config_.transfer == hdr_transfer::hlg) {
            nvapi_display_id_ = nvapi_->resolve_display_id(config_.gpu_index, config_.output_index);
            if (nvapi_display_id_ != 0 && nvapi_->supports_hdr_output(nvapi_display_id_)) {
                int cll  = config_.max_cll  > 0 ? config_.max_cll  : 1000;
                int fall = config_.max_fall > 0 ? config_.max_fall : 400;
                hw_hdr_active_ = nvapi_->enable_hdr_output(nvapi_display_id_, cll, fall);
                if (hw_hdr_active_) {
                    CASPAR_LOG(info) << print()
                        << L" Hardware HDR active — display engine handles PQ + BT.2020.";
                }
            }
        }
    }

    void present_loop()
    {
        SetThreadDescription(GetCurrentThread(), L"Vulkan Present");
        set_thread_dpi_awareness(); // Ensure physical pixel coordinates for VK surface queries

        // ── Startup gate: wait until ALL consumers on this GPU have finished
        // their heavy init (window + surface + swapchain creation) before
        // submitting any frames.  This prevents TDR caused by GPU contention
        // between a presenting consumer and a peer still creating resources.
        vk_device_manager::wait_all_ready(config_.gpu_index);
        gate_open_ = true;  // Allow send() to use frame cache now

        // Wait until buffer has enough frames to satisfy the configured delay
        // before starting to present. This introduces a fixed N-frame latency
        // to video output, allowing operators to compensate for downstream
        // pipeline delay (scalers, audio de-embedders, LED processors).
        const auto min_fill = static_cast<size_t>(config_.delay_frames + 1);

        // Sub-frame delay: additional sleep before each present to fine-tune
        // A/V sync beyond frame-granularity (e.g. matching PortAudio audio path).
        const auto sub_frame_delay = std::chrono::microseconds(
            static_cast<int64_t>(config_.delay_ms * 1000.0));

        // Frame pacer interval: used by the phase-aligned pacer (MAILBOX mode)
        // to sleep until the next frame tick. Not used in FIFO mode (present
        // barrier) where hardware provides the timing.
        const auto frame_interval_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::microseconds(static_cast<int64_t>(1'000'000.0 / format_desc_.fps)));

        while (running_) {
            core::const_frame frame;
            uint64_t frame_gen = 0; // timeline generation for THIS frame
            {
                std::unique_lock<std::mutex> lock(buffer_mutex_);
                buffer_cv_.wait_for(lock, std::chrono::milliseconds(50),
                                    [this, min_fill] { return buffer_.size() >= min_fill || !running_; });
                if (!running_) {
                    break;
                }
                if (buffer_.size() < min_fill) {
                    // Heartbeat once per second so we can see the present loop is
                    // alive even when no frames are flowing.
                    auto hb_now = std::chrono::steady_clock::now();
                    auto hb_elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(
                        hb_now - fps_update_time_).count();
                    if (hb_elapsed >= 1.0) {
                        fps_update_time_ = hb_now;
                        CASPAR_LOG(info) << print() << L" idle gate=" << (gate_open_ ? 1 : 0)
                                         << L" buf=" << buffer_.size()
                                         << L"/" << min_fill
                                         << L" sends=" << diag_sends_.exchange(0)
                                         << L" pres=" << diag_presents_.exchange(0)
                                         << L" submits=" << diag_submits_.exchange(0)
                                         << L" gen=" << frame_generation_;
                    }
                    continue;
                }
                diag_buf_size_last_ = static_cast<int>(buffer_.size());
                auto& entry = buffer_.front();
                frame     = std::move(entry.first);
                frame_gen = entry.second;
                buffer_.pop();
                graph_->set_value("buffered-video",
                                  static_cast<double>(buffer_.size()) / config_.buffer_depth);
            }
            buffer_cv_.notify_one(); // Wake send() if it's blocking on full buffer

            // Frame pacer: With MAILBOX present mode, vkQueuePresentKHR returns
            // immediately (no vsync blocking), so the present loop must self-pace
            // to the target frame rate.
            //
            // Phase-aligned pacing: instead of aligning to the steady_clock epoch
            // (which has a random phase vs vsync, causing periodic judder), we
            // align to the first frame's present time.  All outputs on the same
            // GPU start their present loops after wait_all_ready() at the same
            // moment, so they share the same phase and submit within microseconds
            // of each other — preserving multi-output sync.
            //
            // With FIFO mode (Quadro Sync / present_barrier), vkWaitForFences
            // provides hardware-locked pacing and the sleep is skipped entirely.
            if (!present_barrier_enabled_) {
                if (pacer_epoch_ns_ == 0) {
                    // First frame: record epoch as now. All outputs that passed
                    // wait_all_ready() at the same time get the same epoch.
                    pacer_epoch_ns_ = std::chrono::steady_clock::now().time_since_epoch().count();
                }
                auto now_ns = std::chrono::steady_clock::now().time_since_epoch().count();
                auto interval = frame_interval_ns.count();
                // Compute next tick relative to our epoch (not system boot)
                auto elapsed_since_epoch = now_ns - pacer_epoch_ns_;
                auto ticks_elapsed = elapsed_since_epoch / interval;
                auto next_tick_ns = pacer_epoch_ns_ + (ticks_elapsed + 1) * interval;
                std::this_thread::sleep_until(
                    std::chrono::steady_clock::time_point(
                        std::chrono::steady_clock::duration(next_tick_ns)));
                if (!running_)
                    break; // Exit promptly during shutdown — don't present another frame
            }

            // Sub-frame delay: hold the frame an additional N milliseconds
            // beyond the frame-granularity delay to fine-tune A/V sync.
            if (sub_frame_delay.count() > 0)
                std::this_thread::sleep_for(sub_frame_delay);

            caspar::timer frame_timer;
            present_frame(frame, frame_gen);
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
                      << L" " << swapchain_.width << L"x" << swapchain_.height
                      << L" fps: " << fps;
                graph_->set_text(stats.str());
                // Diagnostic: also log to file once per second so we can tell
                // from the log whether the present loop is actually running.
                CASPAR_LOG(trace) << print() << L" fps=" << fps
                                 << L" buf=" << diag_buf_size_last_
                                 << L" sends=" << diag_sends_.load()
                                 << L" pres=" << diag_presents_.load()
                                 << L" submits=" << diag_submits_.load()
                                 << L" gen=" << frame_generation_;
            }

            // Periodic 5s TIMING log (trace level for test analysis)
            ++diag_timing_frames_;
            {
                auto timing_elapsed = std::chrono::duration<double>(now - diag_timing_start_).count();
                if (timing_elapsed >= 5.0 && diag_timing_frames_ > 0) {
                    int drops = diag_drops_.exchange(0, std::memory_order_relaxed);
                    CASPAR_LOG(trace) << print() << L" TIMING: frames=" << diag_timing_frames_
                                      << L" drops=" << drops;
                    diag_timing_frames_ = 0;
                    diag_timing_start_ = now;
                    // Also reset 1s counters
                    diag_sends_.store(0, std::memory_order_relaxed);
                    diag_presents_.store(0, std::memory_order_relaxed);
                    diag_submits_.store(0, std::memory_order_relaxed);
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

            // Degenerate region — skip blit entirely (returns zero-extent region)
            if (sw <= 0 || sh <= 0 || dw <= 0 || dh <= 0)
                return blit;

            blit.srcOffsets[0] = {sx, sy, 0};
            blit.srcOffsets[1] = {sx + sw, sy + sh, 1};
            blit.dstOffsets[0] = {dx, dy, 0};
            blit.dstOffsets[1] = {dx + dw, dy + dh, 1};
        } else {
            // No explicit subregion: 1:1 pixel mapping, cropped to the smaller
            // of source and destination.  This matches DeckLink behavior where
            // the output naturally takes only the first output_width pixels of
            // each line — no scaling ever happens.
            int32_t copy_w = (std::min)(static_cast<int32_t>(src_width),  static_cast<int32_t>(swapchain_.width));
            int32_t copy_h = (std::min)(static_cast<int32_t>(src_height), static_cast<int32_t>(swapchain_.height));
            blit.srcOffsets[0] = {0, 0, 0};
            blit.srcOffsets[1] = {copy_w, copy_h, 1};
            blit.dstOffsets[0] = {0, 0, 0};
            blit.dstOffsets[1] = {copy_w, copy_h, 1};
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
            start_tdr_watchdog();
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
        if (!hw_hdr_active_)
            set_hdr_metadata();

        // Recreate color pipeline if swapchain dimensions changed (hot-plug to different display)
        if (color_pipeline_ && (color_pipeline_->width() != swapchain_.width ||
                                color_pipeline_->height() != swapchain_.height)) {
            try {
                color_pipeline_ = std::make_unique<color_convert_pipeline>(
                    *device_, swapchain_.width, swapchain_.height);
                float lum = config_.tone_map_op != 0
                    ? config_.display_peak_luminance
                    : static_cast<float>(config_.max_cll);
                color_pipeline_->update_config(config_.gamut, config_.eotf,
                                               lum, config_.tone_map_op);
                CASPAR_LOG(info) << print() << L" Color pipeline recreated for new dimensions.";
            } catch (const std::exception& e) {
                CASPAR_LOG(error) << print() << L" Failed to recreate color pipeline: " << e.what();
                color_pipeline_.reset();
            }
        }

        display_lost_ = false;
        CASPAR_LOG(info) << print() << L" Swapchain recreated successfully.";
    }

    void present_frame(const core::const_frame& frame, uint64_t frame_gen)
    {
        diag_presents_.fetch_add(1, std::memory_order_relaxed);
        diag_stage_.store(1, std::memory_order_relaxed); // entered

        // Fast-path bail during shutdown.
        // that may block far beyond their nominal timeouts when the GPU/driver
        // is in a bad state (NVIDIA vkAcquireNextImageKHR / vkWaitForFences can
        // ignore their timeout argument and hang forever). Skip the whole thing
        // once running_ goes false so the present loop exits within one cv tick.
        if (!running_)
            return;

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

#ifdef CASPAR_CUDA_PEER_ENABLED
        // ── Early D2H kick: start async GPU→host transfer BEFORE fence wait ──
        // Overlaps the ~4ms D2H with the ~2-3ms fence wait + swapchain acquire,
        // reducing effective present_frame time by 2-3ms per frame.
        // Only active after the first frame established cross-GPU CUDA mode.
        bool early_d2h_kicked = false;
        if (cuda_vk_peer_init_ && !cuda_vk_peer_failed_ && !cuda_d2h_active_) {
            auto vk_wrapper = frame.texture()
                ? std::dynamic_pointer_cast<accelerator::vulkan::texture_wrapper>(frame.texture())
                : nullptr;
            if (vk_wrapper) {
                vk_wrapper->ensure_render_complete();
                auto vk_tex = vk_wrapper->vk_texture();
                HANDLE handle = vk_tex->export_win32_handle();
                if (handle) {
                    bool use_16bit = (vk_tex->depth() != common::bit_depth::bit8);
                    auto alloc_sz  = static_cast<unsigned long long>(vk_tex->alloc_size());
                    early_d2h_kicked = begin_cuda_d2h(handle,
                        static_cast<uint32_t>(vk_tex->width()),
                        static_cast<uint32_t>(vk_tex->height()),
                        use_16bit, alloc_sz, vk_tex->device_luid());
                }
            }
        }
#endif

        // Wait for this frame slot's previous use to finish.
        // Use a finite timeout (2× frame period) to detect GPU stalls without
        // blocking indefinitely. With MAILBOX mode the fence is typically ready
        // immediately; with FIFO it may take up to one vsync period.
        const uint64_t fence_timeout_ns = static_cast<uint64_t>(2'000'000'000.0 / format_desc_.fps); // 2 frames in ns
        diag_stage_.store(2, std::memory_order_relaxed); // waitFences
        auto fence_result = vkWaitForFences(dev, 1, &frame_sync.in_flight, VK_TRUE, fence_timeout_ns);
        if (fence_result == VK_TIMEOUT) {
            // GPU severely behind — skip this frame rather than stall further
            graph_->set_tag(diagnostics::tag_severity::WARNING, "dropped-frame");
            return;
        }
        if (!running_) {
            return;
        }

        // ─── Debug capture: extract readback data from previous frame ───────
        // The readback buffer was filled by vkCmdCopyImageToBuffer in the previous
        // command buffer submission. Now that the fence is signaled, the data is ready.
        // We store the raw RGBA16F data — the test runner converts to 16-bit PNG.
        if (debug_readback_pending_ && debug_readback_mapped_) {
            debug_readback_pending_ = false;
            uint32_t w = debug_readback_width_;
            uint32_t h = debug_readback_height_;
            size_t byte_count = static_cast<size_t>(w) * h * 8; // 4 × fp16 = 8 bytes/pixel
            std::lock_guard<std::mutex> lock(debug_frame_mutex_);
            debug_frame_data_.resize(byte_count);
            std::memcpy(debug_frame_data_.data(), debug_readback_mapped_, byte_count);
            debug_frame_w_      = static_cast<int>(w);
            debug_frame_h_      = static_cast<int>(h);
            debug_frame_format_ = 1; // RGBA16F
        }
        // NOTE: Don't reset fence here. Reset just before vkQueueSubmit so that
        // early returns (surface lost, acquire timeout, FSE lost) leave the fence
        // signaled. Otherwise the next use of this frame slot would wait for a
        // fence that never gets signaled, causing a spurious timeout + dropped frame.

        // Acquire next swapchain image
        uint32_t image_index = 0;
        diag_stage_.store(3, std::memory_order_relaxed); // acquire
        auto     result      = vkAcquireNextImageKHR(
            dev, swapchain_.swapchain, 1'000'000'000ULL, frame_sync.image_available, VK_NULL_HANDLE, &image_index);
        if (!running_) {
            return;
        }

        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
            // VK_SUBOPTIMAL means the acquire succeeded and signaled image_available.
            // We must reset the semaphore before retrying acquire after recreation,
            // otherwise the retry would double-signal an already-signaled semaphore (UB).
            // Wait for our queue to drain first — the signaled semaphore may still
            // be referenced by a pending GPU operation. Destroying it while in-flight
            // is a Vulkan spec violation that can crash the driver.
            {
                std::lock_guard<std::mutex> queue_lock(device_->queue_mutex_for(my_queue_idx_));
                vkQueueWaitIdle(my_queue_);
            }
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
            // FSE was lost (alt-tab, focus change, etc.) — image was NOT acquired.
            // Try to reacquire FSE for next frame. Must drop this frame.
            fse_acquired_ = false;
            try_acquire_fse();
            graph_->set_tag(diagnostics::tag_severity::WARNING, "dropped-frame");
            return;
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

            // Clear intermediate to black when subregion is active, so the compute
            // shader doesn't process uninitialized texels outside the blit region
            // (NaN/denorm floats → garbage after PQ/HLG EOTF).
            if (has_subregion()) {
                VkClearColorValue clear_color = {{0.0f, 0.0f, 0.0f, 0.0f}};
                VkImageSubresourceRange range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                vkCmdClearColorImage(cmd, color_pipeline_->image(),
                                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clear_color, 1, &range);
            }
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
        } else if (auto vk_wrapper = frame.texture()
                       ? std::dynamic_pointer_cast<accelerator::vulkan::texture_wrapper>(frame.texture())
                       : nullptr) {
            // ─── VK-native zero-copy path ───────────────────────────────────────
            // The mixer is Vulkan — import the render attachment's external memory
            // on the output's VkDevice and blit directly to the swapchain.
            if (present_path_counter_++ % 300 == 0) {
                CASPAR_LOG(trace) << print() << L" present path: VK-native zero-copy";
            }
            // Wait for the mixer's GPU rendering to finish before importing.
            // This fence wait runs on the present thread, NOT the channel tick,
            // so the mixer can start the next frame in parallel.
            vk_wrapper->ensure_render_complete();

            auto vk_tex = vk_wrapper->vk_texture();
            HANDLE handle = vk_tex->export_win32_handle();
            if (handle) {
                auto w = static_cast<uint32_t>(vk_tex->width());
                auto h = static_cast<uint32_t>(vk_tex->height());
                VkFormat fmt = (vk_tex->depth() == common::bit_depth::bit8)
                                   ? VK_FORMAT_B8G8R8A8_UNORM
                                   : VK_FORMAT_R16G16B16A16_UNORM;
                auto* imported = import_vk_texture(handle, w, h, fmt);
                if (imported) {
                    // Memory barrier to make the mixer's writes visible on this
                    // device.  We use GENERAL→TRANSFER_SRC_OPTIMAL instead of
                    // UNDEFINED→TRANSFER_SRC_OPTIMAL because transitioning FROM
                    // UNDEFINED is defined to discard image contents (Vulkan spec
                    // §12.4).  The imported external memory already contains valid
                    // rendered pixels (the mixer fence-waits before returning the
                    // frame), so we must NOT discard them.
                    //
                    // GENERAL is safe as oldLayout here: the image was imported
                    // from external memory and has never been used on this device,
                    // so no actual layout metadata exists — the driver treats
                    // GENERAL as "contents preserved, make memory available".
                    VkImageMemoryBarrier src_barrier{};
                    src_barrier.sType                           = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                    src_barrier.oldLayout                       = VK_IMAGE_LAYOUT_GENERAL;
                    src_barrier.newLayout                       = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
                    src_barrier.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
                    src_barrier.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
                    src_barrier.image                           = imported->image;
                    src_barrier.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
                    src_barrier.subresourceRange.baseMipLevel   = 0;
                    src_barrier.subresourceRange.levelCount     = 1;
                    src_barrier.subresourceRange.baseArrayLayer = 0;
                    src_barrier.subresourceRange.layerCount     = 1;
                    src_barrier.srcAccessMask                   = 0;
                    src_barrier.dstAccessMask                   = VK_ACCESS_TRANSFER_READ_BIT;

                    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &src_barrier);

                    auto blit_region = compute_blit_region(w, h);
                    vkCmdBlitImage(cmd, imported->image,
                                   VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                   blit_dest_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                   1, &blit_region, VK_FILTER_LINEAR);

                    wrote_to_intermediate = color_convert_active;
                } else {
                    // VK import failed — likely cross-GPU. Try CUDA peer DMA.
#ifdef CASPAR_CUDA_PEER_ENABLED
                    bool cuda_ok = false;
                    if (!cuda_vk_peer_failed_) {
                        if (early_d2h_kicked) {
                            // D2H was kicked before fence wait — just complete.
                            cuda_ok = complete_cuda_d2h(blit_dest_image, cmd);
                        } else {
                            // First frame or early kick wasn't possible — synchronous fallback.
                            bool use_16bit = (vk_tex->depth() != common::bit_depth::bit8);
                            auto alloc_sz  = static_cast<unsigned long long>(vk_tex->alloc_size());
                            cuda_ok = cross_gpu_cuda_transfer(handle, w, h, use_16bit, alloc_sz,
                                                              vk_tex->device_luid(), blit_dest_image, cmd);
                        }
                    }
                    if (cuda_ok) {
                        if (color_convert_active)
                            wrote_to_intermediate = true;
                    } else
#endif
                    {
                        // CUDA unavailable or failed — CPU fallback
                        upload_frame_cpu(frame, blit_dest_image, cmd);
                        if (color_convert_active)
                            wrote_to_intermediate = true;
                    }
                }
            } else {
                // Handle export failed — fall back to CPU staging
                upload_frame_cpu(frame, blit_dest_image, cmd);
                if (color_convert_active)
                    wrote_to_intermediate = true;
            }
        } else if (frame_cache_ && frame_cache_->pool() &&
                   (frame_generation_ > 0 || frame_cache_->is_cross_gpu())) {
            if (present_path_counter_++ % 300 == 0) {
                CASPAR_LOG(trace) << print() << L" present path: shared pool (gen="
                                 << frame_generation_ << L" cross_gpu=" << frame_cache_->is_cross_gpu() << L")";
            }
            // Frame cache path: shared pool was populated in send() (same-GPU)
            // or cross-GPU uses CPU staging (Pascal GL→VK interop is broken).
            auto* pool = frame_cache_->pool();

            // Check if we have valid frame data to transfer.
            bool has_frame_data = false;
            if (frame_cache_->is_cross_gpu()) {
                has_frame_data = frame.texture() || (frame.image_data(0).data() && frame.image_data(0).size() > 0);
            } else {
                has_frame_data = !!frame.texture();
            }

            if (!has_frame_data) {
                VkClearColorValue clear_color = {{0.0f, 0.0f, 0.0f, 1.0f}};
                VkImageSubresourceRange range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                vkCmdClearColorImage(cmd, swapchain_.images[image_index],
                                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clear_color, 1, &range);
            } else if (frame_cache_->is_cross_gpu()) {
                // Cross-GPU: use CPU staging buffer → vkCmdCopyBufferToImage.
                // Pascal GPUs cannot write to VK-external-memory textures via GL
                // (glCopyImageSubData → GL_OUT_OF_MEMORY, glTexSubImage2D → black).
                // Pure Vulkan staging avoids GL interop entirely.
                upload_frame_cpu(frame, blit_dest_image, cmd);
                if (color_convert_active) {
                    wrote_to_intermediate = true;
                }
            } else {
                // Same-GPU: shared pool was populated in send() via submit_frame().
                // Transition shared image for transfer src.
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
            }
        } else if (frame_cache_ && frame_cache_->pool()) {
            if (present_path_counter_++ % 300 == 0) {
                CASPAR_LOG(trace) << print() << L" present path: black (pool exists, gen=0)";
            }
            // Pool exists but not yet populated (gen=0, no OGL texture received yet).
            // Clear to black — the CPU upload path uses format_desc_ (channel resolution)
            // which may exceed the swapchain image dimensions, causing an out-of-bounds
            // GPU write and TDR.
            VkClearColorValue clear_color = {{0.0f, 0.0f, 0.0f, 1.0f}};
            VkImageSubresourceRange range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCmdClearColorImage(cmd, swapchain_.images[image_index],
                                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clear_color, 1, &range);
        } else if (!frame.texture()) {
            if (present_path_counter_++ % 300 == 0) {
                CASPAR_LOG(trace) << print() << L" present path: black (null texture, no frame_cache)";
            }
            // No texture available (VK mixer empty frame, or no content loaded).
            // Clear to black instead of upload_frame_cpu — the channel resolution
            // (format_desc_) may exceed the swapchain image dimensions, and
            // vkCmdCopyBufferToImage with an oversized region causes TDR.
            VkClearColorValue clear_color = {{0.0f, 0.0f, 0.0f, 1.0f}};
            VkImageSubresourceRange range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCmdClearColorImage(cmd, swapchain_.images[image_index],
                                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clear_color, 1, &range);
        } else {
            if (present_path_counter_++ % 300 == 0) {
                CASPAR_LOG(trace) << print() << L" present path: CPU upload fallback"
                                 << L" tex=" << (frame.texture() ? L"yes" : L"no")
                                 << L" cache=" << (frame_cache_ ? L"yes" : L"no");
            }
            // No frame cache or pool — CPU fallback: upload pixel data via staging buffer
            upload_frame_cpu(frame, swapchain_.images[image_index], cmd);
        }

#ifdef CASPAR_CUDA_PEER_ENABLED
        // Drain any early D2H that wasn't consumed (e.g., VK import succeeded
        // or frame path didn't reach the cross-GPU branch).
        if (cuda_d2h_active_) {
            cudaSetDevice(cuda_vk_src_device_);
            cudaStreamSynchronize(cuda_vk_src_stream_);
            cuda_d2h_active_ = false;
        }
#endif

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

            // ─── Debug readback: copy post-conversion RGBA16F to host-visible buffer ───
            if (debug_capture_enabled_.load(std::memory_order_relaxed)) {
                ensure_debug_readback_buffer(color_pipeline_->width(), color_pipeline_->height());
                if (debug_readback_buffer_ != VK_NULL_HANDLE) {
                    VkBufferImageCopy copy_region{};
                    copy_region.bufferOffset      = 0;
                    copy_region.bufferRowLength   = 0; // tightly packed
                    copy_region.bufferImageHeight = 0;
                    copy_region.imageSubresource  = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                    copy_region.imageOffset       = {0, 0, 0};
                    copy_region.imageExtent       = {color_pipeline_->width(), color_pipeline_->height(), 1};
                    vkCmdCopyImageToBuffer(cmd, color_pipeline_->image(),
                                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                           debug_readback_buffer_, 1, &copy_region);
                    debug_readback_width_   = color_pipeline_->width();
                    debug_readback_height_  = color_pipeline_->height();
                    debug_readback_pending_ = true;
                }
            }
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

        VkSubmitInfo submit_info{};
        submit_info.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.waitSemaphoreCount   = wait_count;
        submit_info.pWaitSemaphores      = wait_semaphores;
        submit_info.pWaitDstStageMask    = wait_stages;
        submit_info.commandBufferCount   = 1;
        submit_info.pCommandBuffers      = &frame_sync.cmd_buffer;
        submit_info.signalSemaphoreCount = 1;
        submit_info.pSignalSemaphores    = &frame_sync.render_finished;

        // Only chain VkTimelineSemaphoreSubmitInfo when we actually have a
        // timeline semaphore to wait on.  Some drivers mis-handle the pNext
        // chain when all semaphores are binary.
        VkTimelineSemaphoreSubmitInfo timeline_submit{};
        uint64_t wait_values[2]  = {0, 0};
        uint64_t signal_value    = 0;

        if (used_shared_pool && frame_cache_ && frame_cache_->timeline_semaphore() != VK_NULL_HANDLE) {
            wait_semaphores[1] = frame_cache_->timeline_semaphore();
            wait_values[1]     = frame_gen;  // Wait for THIS frame's transfer to complete
            wait_count         = 2;

            timeline_submit.sType                     = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
            timeline_submit.waitSemaphoreValueCount   = wait_count;
            timeline_submit.pWaitSemaphoreValues      = wait_values;
            timeline_submit.signalSemaphoreValueCount = 1;
            timeline_submit.pSignalSemaphoreValues    = &signal_value;

            submit_info.pNext              = &timeline_submit;
            submit_info.waitSemaphoreCount = wait_count;
        }

        VkResult submit_result;
        VkResult present_result;

        // Software sync barrier: wait for all consumers in the sync group
        // to reach this point before submitting.  Aligns present calls to
        // within ~1ms across GPUs without Quadro Sync hardware.
        if (config_.sync_group > 0 && sync_group_token_) {
            vk_device_manager::sync_group_wait(config_.sync_group);
        }

        // Lock per-queue mutex — if multiple consumers share a physical queue
        // (wrapping), this serializes their vkQueueSubmit/vkQueuePresentKHR calls.
        diag_stage_.store(5, std::memory_order_relaxed); // pre queue_lock
        std::lock_guard<std::mutex> queue_lock(device_->queue_mutex_for(my_queue_idx_));
        // Reset fence just before submit. This is deliberately late — earlier in
        // present_frame() there are multiple early-return paths (surface lost,
        // acquire timeout, FSE lost). If we reset the fence before those checks,
        // an early return would leave the fence unsignaled and the next use of
        // this frame slot would timeout waiting for it.
        vkResetFences(dev, 1, &frame_sync.in_flight);
        diag_stage_.store(6, std::memory_order_relaxed); // submit
        submit_result = vkQueueSubmit(my_queue_, 1, &submit_info, frame_sync.in_flight);
        if (submit_result == VK_SUCCESS) {
            diag_submits_.fetch_add(1, std::memory_order_relaxed);
            watchdog_submits_.fetch_add(1, std::memory_order_relaxed);
        }

        if (submit_result == VK_ERROR_DEVICE_LOST) {
            CASPAR_LOG(error) << print() << L" GPU device lost (TDR) during submit."
                              << L" queue=" << my_queue_idx_
                              << L" waits=" << submit_info.waitSemaphoreCount
                              << L" timeline=" << (submit_info.pNext ? L"yes" : L"no")
                              << L" gen=" << frame_generation_
                              << L" Output halted.";
            display_lost_ = true;
            device_dead_  = true;
            start_tdr_watchdog();
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
        diag_stage_.store(7, std::memory_order_relaxed); // present
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
            start_tdr_watchdog();
            return;
        }

        ++frames_presented_;

        // Debug capture: for non-color-pipeline path, store frame pixels directly.
        // (When color pipeline is active, readback is recorded in the command buffer above.)
        if (debug_capture_enabled_.load(std::memory_order_relaxed) && !debug_readback_pending_) {
            if (!color_pipeline_ || !color_pipeline_->is_active()) {
                const auto& img = frame.image_data(0);
                if (img.data() && img.size() > 0) {
                    std::lock_guard<std::mutex> lock(debug_frame_mutex_);
                    debug_frame_data_.assign(img.data(), img.data() + img.size());
                    debug_frame_w_ = static_cast<int>(frame.width());
                    debug_frame_h_ = static_cast<int>(frame.height());
                    debug_frame_format_ = 0; // BGRA8
                }
            }
        }

        // Advance to next frame-in-flight slot for the next iteration
        swapchain_.advance_frame();
    }

    // ─── Debug readback buffer management ────────────────────────────────────
    void ensure_debug_readback_buffer(uint32_t w, uint32_t h)
    {
        size_t required = static_cast<size_t>(w) * h * 8; // RGBA16F = 8 bytes/pixel
        if (debug_readback_buffer_ != VK_NULL_HANDLE && debug_readback_size_ >= required) {
            return; // Already allocated and large enough
        }

        VkDevice dev = device_->device();

        // Destroy old buffer if resizing
        if (debug_readback_buffer_ != VK_NULL_HANDLE) {
            vkDestroyBuffer(dev, debug_readback_buffer_, nullptr);
            vkFreeMemory(dev, debug_readback_memory_, nullptr);
            debug_readback_buffer_ = VK_NULL_HANDLE;
            debug_readback_memory_ = VK_NULL_HANDLE;
            debug_readback_mapped_ = nullptr;
        }

        VkBufferCreateInfo buf_info{};
        buf_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        buf_info.size  = required;
        buf_info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        buf_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateBuffer(dev, &buf_info, nullptr, &debug_readback_buffer_) != VK_SUCCESS) {
            debug_readback_buffer_ = VK_NULL_HANDLE;
            return;
        }

        VkMemoryRequirements mem_reqs;
        vkGetBufferMemoryRequirements(dev, debug_readback_buffer_, &mem_reqs);

        // Find host-visible, host-coherent memory type
        VkPhysicalDeviceMemoryProperties mem_props;
        vkGetPhysicalDeviceMemoryProperties(device_->physical_device(), &mem_props);
        uint32_t mem_type_idx = UINT32_MAX;
        VkMemoryPropertyFlags desired = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        for (uint32_t i = 0; i < mem_props.memoryTypeCount; ++i) {
            if ((mem_reqs.memoryTypeBits & (1u << i)) &&
                (mem_props.memoryTypes[i].propertyFlags & desired) == desired) {
                mem_type_idx = i;
                break;
            }
        }
        if (mem_type_idx == UINT32_MAX) {
            vkDestroyBuffer(dev, debug_readback_buffer_, nullptr);
            debug_readback_buffer_ = VK_NULL_HANDLE;
            return;
        }

        VkMemoryAllocateInfo alloc_info{};
        alloc_info.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        alloc_info.allocationSize  = mem_reqs.size;
        alloc_info.memoryTypeIndex = mem_type_idx;

        if (vkAllocateMemory(dev, &alloc_info, nullptr, &debug_readback_memory_) != VK_SUCCESS) {
            vkDestroyBuffer(dev, debug_readback_buffer_, nullptr);
            debug_readback_buffer_ = VK_NULL_HANDLE;
            return;
        }

        vkBindBufferMemory(dev, debug_readback_buffer_, debug_readback_memory_, 0);
        vkMapMemory(dev, debug_readback_memory_, 0, required, 0, &debug_readback_mapped_);
        debug_readback_size_ = required;
    }

    void destroy_debug_readback_buffer()
    {
        if (debug_readback_buffer_ != VK_NULL_HANDLE) {
            VkDevice dev = device_->device();
            vkUnmapMemory(dev, debug_readback_memory_);
            vkDestroyBuffer(dev, debug_readback_buffer_, nullptr);
            vkFreeMemory(dev, debug_readback_memory_, nullptr);
            debug_readback_buffer_ = VK_NULL_HANDLE;
            debug_readback_memory_ = VK_NULL_HANDLE;
            debug_readback_mapped_ = nullptr;
            debug_readback_size_   = 0;
        }
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

        // Debug capture: GDI path has no color pipeline, store frame pixels directly.
        if (debug_capture_enabled_.load(std::memory_order_relaxed)) {
            std::lock_guard<std::mutex> lock(debug_frame_mutex_);
            debug_frame_data_.assign(img.data(), img.data() + img.size());
            debug_frame_w_ = src_w;
            debug_frame_h_ = src_h;
            debug_frame_format_ = 0; // BGRA8
        }
    }

    // ─── VK-native import cache ─────────────────────────────────────────────
    // When the mixer is Vulkan, we import the mixer's render attachment memory
    // on the output's VkDevice via Win32 HANDLE.  The cache avoids re-importing
    // the same memory every frame (render attachments are pooled).
    struct vk_imported_image
    {
        VkImage        image        = VK_NULL_HANDLE;
        VkDeviceMemory memory       = VK_NULL_HANDLE;
        VkImageView    view         = VK_NULL_HANDLE;
        uint32_t       width        = 0;
        uint32_t       height       = 0;
        HANDLE         src_handle   = nullptr; // Cache key — the mixer texture's exported handle (NOT owned)
        HANDLE         owned_handle = nullptr; // Duplicated handle — owned by this cache entry, must be closed
    };

    // ─── VK-native import: import mixer's exportable VkImage memory ─────────
    // Returns a pointer to a cached vk_imported_image, or nullptr on failure.
    // The returned image is in VK_IMAGE_LAYOUT_UNDEFINED and needs a transition
    // to TRANSFER_SRC before blitting.
    vk_imported_image* import_vk_texture(HANDLE src_handle, uint32_t width, uint32_t height, VkFormat format)
    {
        // Check cache first — the mixer pools render attachments, so we'll see
        // the same handle repeatedly.
        for (auto& imp : vk_import_cache_) {
            if (imp.src_handle == src_handle && imp.width == width && imp.height == height)
                return &imp;
        }

        // Evict oldest entries if cache grows too large (prevents unbounded
        // growth when attachment pool creates new textures on resolution change).
        constexpr size_t MAX_IMPORT_CACHE = 8;
        while (vk_import_cache_.size() >= MAX_IMPORT_CACHE) {
            auto& old = vk_import_cache_.front();
            auto  dev = device_->device();
            if (old.view != VK_NULL_HANDLE)   vkDestroyImageView(dev, old.view, nullptr);
            if (old.image != VK_NULL_HANDLE)  vkDestroyImage(dev, old.image, nullptr);
            if (old.memory != VK_NULL_HANDLE) vkFreeMemory(dev, old.memory, nullptr);
            if (old.owned_handle)             CloseHandle(old.owned_handle);
            vk_import_cache_.erase(vk_import_cache_.begin());
        }

        auto dev = device_->device();

        // Import the Win32 handle as VkDeviceMemory on the output's VkDevice.
        // Duplicate the handle so the import cache owns its own copy.
        HANDLE dup_handle = nullptr;
        if (!DuplicateHandle(GetCurrentProcess(), src_handle, GetCurrentProcess(),
                             &dup_handle, 0, FALSE, DUPLICATE_SAME_ACCESS)) {
            if (!vk_import_fail_logged_) {
                CASPAR_LOG(warning) << print() << L" Failed to duplicate Win32 handle for VK import (cross-GPU; will use CUDA peer or CPU fallback)";
                vk_import_fail_logged_ = true;
            }
            return nullptr;
        }

        VkImportMemoryWin32HandleInfoKHR importInfo{};
        importInfo.sType      = VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_KHR;
        importInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
        importInfo.handle     = dup_handle;

        // Create a VkImage compatible with the imported memory
        VkExternalMemoryImageCreateInfo extMemImageInfo{};
        extMemImageInfo.sType       = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
        extMemImageInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;

        VkImageCreateInfo imageInfo{};
        imageInfo.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.pNext         = &extMemImageInfo;
        imageInfo.imageType     = VK_IMAGE_TYPE_2D;
        imageInfo.format        = format;
        imageInfo.extent        = {width, height, 1};
        imageInfo.mipLevels     = 1;
        imageInfo.arrayLayers   = 1;
        imageInfo.samples       = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.tiling        = VK_IMAGE_TILING_OPTIMAL;
        imageInfo.usage         = VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        imageInfo.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        VkImage image = VK_NULL_HANDLE;
        if (vkCreateImage(dev, &imageInfo, nullptr, &image) != VK_SUCCESS) {
            if (!vk_import_fail_logged_) {
                CASPAR_LOG(warning) << print() << L" Failed to create VkImage for import (cross-GPU; will use CUDA peer or CPU fallback)";
                vk_import_fail_logged_ = true;
            }
            CloseHandle(dup_handle);
            return nullptr;
        }

        VkMemoryRequirements memReqs;
        vkGetImageMemoryRequirements(dev, image, &memReqs);

        VkPhysicalDeviceMemoryProperties memProps;
        vkGetPhysicalDeviceMemoryProperties(device_->physical_device(), &memProps);

        uint32_t memTypeIdx = UINT32_MAX;
        for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
            if ((memReqs.memoryTypeBits & (1u << i)) &&
                (memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
                memTypeIdx = i;
                break;
            }
        }
        if (memTypeIdx == UINT32_MAX) {
            if (!vk_import_fail_logged_) {
                CASPAR_LOG(warning) << print() << L" No suitable memory type for VK import (cross-GPU; will use CUDA peer or CPU fallback)";
                vk_import_fail_logged_ = true;
            }
            vkDestroyImage(dev, image, nullptr);
            CloseHandle(dup_handle);
            return nullptr;
        }

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.pNext           = &importInfo;
        allocInfo.allocationSize  = memReqs.size;
        allocInfo.memoryTypeIndex = memTypeIdx;

        VkDeviceMemory importedMem = VK_NULL_HANDLE;
        if (vkAllocateMemory(dev, &allocInfo, nullptr, &importedMem) != VK_SUCCESS) {
            if (!vk_import_fail_logged_) {
                CASPAR_LOG(warning) << print() << L" VK memory import failed (cross-GPU; will use CUDA peer or CPU fallback)";
                vk_import_fail_logged_ = true;
            }
            vkDestroyImage(dev, image, nullptr);
            CloseHandle(dup_handle);
            return nullptr;
        }

        if (vkBindImageMemory(dev, image, importedMem, 0) != VK_SUCCESS) {
            if (!vk_import_fail_logged_) {
                CASPAR_LOG(warning) << print() << L" Failed to bind imported VK memory (cross-GPU; will use CUDA peer or CPU fallback)";
                vk_import_fail_logged_ = true;
            }
            vkFreeMemory(dev, importedMem, nullptr);
            vkDestroyImage(dev, image, nullptr);
            CloseHandle(dup_handle);
            return nullptr;
        }

        vk_import_cache_.push_back({image, importedMem, VK_NULL_HANDLE, width, height, src_handle, dup_handle});

        if (!vk_mixer_logged_) {
            CASPAR_LOG(info) << print() << L" VK mixer detected \u2014 using VK-native zero-copy path"
                             << L" (bypassing GL\u2194VK interop)";
            vk_mixer_logged_ = true;
        }

        return &vk_import_cache_.back();
    }

#ifdef CASPAR_CUDA_PEER_ENABLED

    // Ensure a VK image exists on the consumer's GPU at the full mixer resolution.
    // This is the blit source for the subregion crop into the swapchain.
    void ensure_cuda_staging_image(uint32_t width, uint32_t height, bool use_16bit)
    {
        if (cuda_staging_image_ != VK_NULL_HANDLE &&
            cuda_staging_w_ == width && cuda_staging_h_ == height)
            return;

        auto dev = device_->device();

        // Tear down previous
        if (cuda_staging_image_ != VK_NULL_HANDLE) {
            vkDestroyImage(dev, cuda_staging_image_, nullptr);
            vkFreeMemory(dev, cuda_staging_memory_, nullptr);
            cuda_staging_image_  = VK_NULL_HANDLE;
            cuda_staging_memory_ = VK_NULL_HANDLE;
        }

        VkFormat fmt = use_16bit ? VK_FORMAT_R16G16B16A16_UNORM : VK_FORMAT_B8G8R8A8_UNORM;

        VkImageCreateInfo img_ci{};
        img_ci.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        img_ci.imageType     = VK_IMAGE_TYPE_2D;
        img_ci.format        = fmt;
        img_ci.extent        = {width, height, 1};
        img_ci.mipLevels     = 1;
        img_ci.arrayLayers   = 1;
        img_ci.samples       = VK_SAMPLE_COUNT_1_BIT;
        img_ci.tiling        = VK_IMAGE_TILING_OPTIMAL;
        img_ci.usage         = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        img_ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        if (vkCreateImage(dev, &img_ci, nullptr, &cuda_staging_image_) != VK_SUCCESS) {
            cuda_staging_image_ = VK_NULL_HANDLE;
            return;
        }

        VkMemoryRequirements mem_reqs;
        vkGetImageMemoryRequirements(dev, cuda_staging_image_, &mem_reqs);

        VkPhysicalDeviceMemoryProperties mem_props;
        vkGetPhysicalDeviceMemoryProperties(device_->physical_device(), &mem_props);

        uint32_t type_index = UINT32_MAX;
        for (uint32_t i = 0; i < mem_props.memoryTypeCount; ++i) {
            if ((mem_reqs.memoryTypeBits & (1u << i)) &&
                (mem_props.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
                type_index = i;
                break;
            }
        }

        if (type_index == UINT32_MAX) {
            vkDestroyImage(dev, cuda_staging_image_, nullptr);
            cuda_staging_image_ = VK_NULL_HANDLE;
            return;
        }

        VkMemoryAllocateInfo alloc_info{};
        alloc_info.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        alloc_info.allocationSize  = mem_reqs.size;
        alloc_info.memoryTypeIndex = type_index;

        if (vkAllocateMemory(dev, &alloc_info, nullptr, &cuda_staging_memory_) != VK_SUCCESS) {
            vkDestroyImage(dev, cuda_staging_image_, nullptr);
            cuda_staging_image_ = VK_NULL_HANDLE;
            return;
        }

        vkBindImageMemory(dev, cuda_staging_image_, cuda_staging_memory_, 0);
        cuda_staging_w_ = width;
        cuda_staging_h_ = height;
    }

    // ─── Cross-GPU VK→VK transfer via CUDA direct D2H ────────────────────────
    // When the mixer's VK texture is on a different physical GPU from the
    // consumer, import the VK texture into CUDA on the source GPU and DMA
    // directly to the consumer's host-visible VK staging buffer.
    //
    // Key optimizations vs. the naive peer-copy path:
    //  1. CUDA import cache — the mixer uses N render-attachment slots (triple
    //     buffering), each with a different Win32 HANDLE. Without caching, every
    //     frame triggers cudaImportExternalMemory + cudaDestroyExternalMemory
    //     which costs ~150 ms round-trip. We cache up to kMaxCudaVkImports
    //     entries so reimport only happens during the first N frames.
    //  2. Direct D2H — instead of peer-copy (GPU 0 → host → GPU 1) then another
    //     D2H (GPU 1 → host → VK buffer), we copy straight from the CUDA array
    //     on GPU 0 to the host-visible VK staging buffer via cudaHostRegister.
    //     This halves the PCIe traffic and eliminates the GPU 1 staging buffer.
    //
    // Returns true if the transfer succeeded, false to fall back to CPU.
    // Phase 1: begin_cuda_d2h — starts async D2H, no sync. Can be called
    // before fence wait to overlap D2H (~4ms) with VK sync (~2-3ms).
    bool begin_cuda_d2h(HANDLE src_handle, uint32_t width, uint32_t height,
                        bool use_16bit, unsigned long long vk_alloc_size,
                        const uint8_t* src_luid)
    {
        try {
            size_t pixel_size = use_16bit ? 8 : 4;

            // ── Subregion optimization ──────────────────────────────────────
            uint32_t sub_x = 0, sub_y = 0, sub_w = width, sub_h = height;
            if (has_subregion()) {
                sub_x = static_cast<uint32_t>((std::clamp)(config_.src_x, 0, static_cast<int>(width)));
                sub_y = static_cast<uint32_t>((std::clamp)(config_.src_y, 0, static_cast<int>(height)));
                sub_w = config_.region_w > 0
                    ? static_cast<uint32_t>((std::min)(config_.region_w, static_cast<int>(width - sub_x)))
                    : width - sub_x;
                sub_h = config_.region_h > 0
                    ? static_cast<uint32_t>((std::min)(config_.region_h, static_cast<int>(height - sub_y)))
                    : height - sub_y;
            }
            sub_w = (std::min)(sub_w, swapchain_.width);
            sub_h = (std::min)(sub_h, swapchain_.height);
            if (sub_w == 0 || sub_h == 0) return false;

            size_t sub_row_bytes   = sub_w * pixel_size;
            size_t sub_total_bytes = static_cast<size_t>(sub_w) * sub_h * pixel_size;

            // ── Lazy init: find the source CUDA device ──────────────────────
            if (!cuda_vk_peer_init_) {
                int cuda_dev_count = 0;
                cudaGetDeviceCount(&cuda_dev_count);
                if (cuda_dev_count < 1) {
                    cuda_vk_peer_failed_ = true;
                    return false;
                }

                cuda_vk_src_device_ = -1;
                if (src_luid) {
                    for (int i = 0; i < cuda_dev_count; ++i) {
                        cudaDeviceProp prop;
                        if (cudaGetDeviceProperties(&prop, i) != cudaSuccess) continue;
                        if (memcmp(prop.luid, src_luid, 8) == 0) {
                            cuda_vk_src_device_ = i;
                            break;
                        }
                    }
                }
                if (cuda_vk_src_device_ < 0) {
                    if (src_luid) {
                        // LUID was provided but no CUDA device matched — wrong GPU would be selected.
                        // Fall back to non-CUDA path instead of silently using device 0.
                        CASPAR_LOG(warning) << print() << L" CUDA LUID matching failed — no CUDA device matches "
                                            << L"the Vulkan device LUID. Falling back to CPU staging path.";
                        cuda_vk_peer_failed_ = true;
                        cuda_vk_peer_init_   = true;
                        return false;
                    }
                    cuda_vk_src_device_ = 0; // No LUID available (single-GPU likely) — use device 0
                }

                cudaSetDevice(cuda_vk_src_device_);
                cudaStreamCreate(&cuda_vk_src_stream_);

                cuda_vk_peer_init_   = true;
                cuda_vk_total_bytes_ = sub_total_bytes;
                cuda_vk_row_bytes_   = sub_row_bytes;
                cuda_vk_height_      = sub_h;

                CASPAR_LOG(info) << print() << L" CUDA VK cross-GPU transfer initialized (direct D2H)"
                                 << L" src_device=" << cuda_vk_src_device_
                                 << L", subregion " << sub_w << L"x" << sub_h
                                 << L" from " << sub_x << L"," << sub_y
                                 << L", " << (sub_total_bytes / 1024 / 1024) << L" MB/frame";
            }

            if (cuda_vk_peer_failed_)
                return false;

            // ── Look up or import VK texture into CUDA (cached) ─────────────
            cudaArray_t src_array = nullptr;
            for (int i = 0; i < cuda_vk_import_count_; ++i) {
                if (cuda_vk_imports_[i].handle == src_handle) {
                    src_array = cuda_vk_imports_[i].array;
                    break;
                }
            }

            if (!src_array) {
                // Not in cache — import this VK texture handle into CUDA
                cudaSetDevice(cuda_vk_src_device_);

                if (cuda_vk_import_count_ >= kMaxCudaVkImports) {
                    // Cache full — evict oldest entry
                    auto& oldest = cuda_vk_imports_[0];
                    if (oldest.mipmap) cudaFreeMipmappedArray(oldest.mipmap);
                    if (oldest.ext_mem) cudaDestroyExternalMemory(oldest.ext_mem);
                    memmove(&cuda_vk_imports_[0], &cuda_vk_imports_[1],
                            sizeof(cuda_vk_import_entry) * (kMaxCudaVkImports - 1));
                    cuda_vk_imports_[kMaxCudaVkImports - 1] = {};
                    --cuda_vk_import_count_;
                }

                auto& entry = cuda_vk_imports_[cuda_vk_import_count_];
                entry.handle = src_handle;

                cudaExternalMemoryHandleDesc extMemDesc{};
                extMemDesc.type                = cudaExternalMemoryHandleTypeOpaqueWin32;
                extMemDesc.handle.win32.handle = src_handle;
                extMemDesc.size                = vk_alloc_size;
                extMemDesc.flags               = 0;

                if (cudaImportExternalMemory(&entry.ext_mem, &extMemDesc) != cudaSuccess) {
                    CASPAR_LOG(warning) << print() << L" CUDA VK import failed for handle " << src_handle;
                    entry = {};
                    return false;
                }

                cudaExternalMemoryMipmappedArrayDesc mipmapDesc{};
                mipmapDesc.offset     = 0;
                int bits = use_16bit ? 16 : 8;
                mipmapDesc.formatDesc = cudaCreateChannelDesc(bits, bits, bits, bits,
                                                               cudaChannelFormatKindUnsigned);
                mipmapDesc.extent.width  = width;
                mipmapDesc.extent.height = height;
                mipmapDesc.extent.depth  = 0;
                mipmapDesc.numLevels     = 1;
                mipmapDesc.flags         = cudaArrayDefault;

                if (cudaExternalMemoryGetMappedMipmappedArray(&entry.mipmap, entry.ext_mem,
                                                              &mipmapDesc) != cudaSuccess) {
                    cudaDestroyExternalMemory(entry.ext_mem);
                    entry = {};
                    return false;
                }

                if (cudaGetMipmappedArrayLevel(&entry.array, entry.mipmap, 0) != cudaSuccess) {
                    cudaFreeMipmappedArray(entry.mipmap);
                    cudaDestroyExternalMemory(entry.ext_mem);
                    entry = {};
                    return false;
                }

                src_array = entry.array;
                ++cuda_vk_import_count_;
                CASPAR_LOG(info) << print() << L" CUDA VK import cached (slot "
                                 << cuda_vk_import_count_ << L"/" << kMaxCudaVkImports
                                 << L", handle=" << src_handle << L")";
            }

            // ── Ensure VK staging buffer exists ─────────────────────────────
            auto dev      = device_->device();
            auto phys_dev = device_->physical_device();

            if (staging_buffer_ == VK_NULL_HANDLE || staging_buffer_size_ < sub_total_bytes) {
                // Unregister old CUDA mapping before freeing
                if (cuda_vk_host_registered_ && staging_mapped_) {
                    cudaSetDevice(cuda_vk_src_device_);
                    cudaHostUnregister(staging_mapped_);
                    cuda_vk_host_registered_ = false;
                }
                if (staging_mapped_) {
                    vkUnmapMemory(dev, staging_memory_);
                    staging_mapped_ = nullptr;
                }
                if (staging_buffer_ != VK_NULL_HANDLE) {
                    vkDestroyBuffer(dev, staging_buffer_, nullptr);
                    vkFreeMemory(dev, staging_memory_, nullptr);
                }

                VkBufferCreateInfo buf_info{};
                buf_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
                buf_info.size  = sub_total_bytes;
                buf_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
                if (vkCreateBuffer(dev, &buf_info, nullptr, &staging_buffer_) != VK_SUCCESS) {
                    staging_buffer_ = VK_NULL_HANDLE;
                    return false;
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
                    vkDestroyBuffer(dev, staging_buffer_, nullptr);
                    staging_buffer_ = VK_NULL_HANDLE;
                    return false;
                }

                VkMemoryAllocateInfo alloc_info{};
                alloc_info.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
                alloc_info.allocationSize  = mem_reqs.size;
                alloc_info.memoryTypeIndex = type_index;
                if (vkAllocateMemory(dev, &alloc_info, nullptr, &staging_memory_) != VK_SUCCESS) {
                    vkDestroyBuffer(dev, staging_buffer_, nullptr);
                    staging_buffer_ = VK_NULL_HANDLE;
                    return false;
                }
                vkBindBufferMemory(dev, staging_buffer_, staging_memory_, 0);
                staging_buffer_size_ = sub_total_bytes;

                // Persistent map — HOST_COHERENT memory stays mapped for lifetime
                if (vkMapMemory(dev, staging_memory_, 0, sub_total_bytes, 0, &staging_mapped_) != VK_SUCCESS) {
                    staging_mapped_ = nullptr;
                    return false;
                }
            }

            // ── Register VK staging buffer with CUDA for direct GPU→host DMA ─
            if (!cuda_vk_host_registered_ && staging_mapped_) {
                cudaSetDevice(cuda_vk_src_device_);
                if (cudaHostRegister(staging_mapped_, sub_total_bytes,
                                     cudaHostRegisterDefault) == cudaSuccess) {
                    cuda_vk_host_registered_ = true;
                    CASPAR_LOG(info) << print() << L" CUDA host-registered VK staging buffer ("
                                     << (sub_total_bytes / 1024 / 1024) << L" MB)";
                } else {
                    CASPAR_LOG(warning) << print()
                        << L" cudaHostRegister failed — using pinned fallback";
                }
            }

            // ── Direct D2H: CUDA array on GPU 0 → VK staging buffer (host) ──
            // This is a single PCIe transfer — no peer copy, no GPU 1 staging.
            void* dst_ptr = staging_mapped_;
            bool using_pinned_fallback = false;

            if (!cuda_vk_host_registered_) {
                // Fallback: use a CUDA pinned host buffer + memcpy
                if (!cuda_vk_pinned_host_) {
                    cudaSetDevice(cuda_vk_src_device_);
                    if (cudaMallocHost(&cuda_vk_pinned_host_, sub_total_bytes) != cudaSuccess) {
                        return false;
                    }
                }
                dst_ptr = cuda_vk_pinned_host_;
                using_pinned_fallback = true;
            }

            cudaSetDevice(cuda_vk_src_device_);
            if (cudaMemcpy2DFromArrayAsync(dst_ptr, sub_row_bytes,
                                            src_array,
                                            sub_x * pixel_size, sub_y,
                                            sub_row_bytes, sub_h,
                                            cudaMemcpyDeviceToHost,
                                            cuda_vk_src_stream_) != cudaSuccess) {
                return false;
            }

            // Store state for phase 2 (complete_cuda_d2h).
            cuda_d2h_active_       = true;
            cuda_d2h_using_pinned_ = using_pinned_fallback;
            cuda_d2h_sub_w_        = sub_w;
            cuda_d2h_sub_h_        = sub_h;
            cuda_d2h_16bit_        = use_16bit;
            return true;

        } catch (const std::exception& e) {
            CASPAR_LOG(warning) << print() << L" CUDA VK begin_cuda_d2h failed: " << e.what();
            cuda_vk_peer_failed_ = true;
            return false;
        }
    }

    // Phase 2: complete_cuda_d2h — sync the async D2H and record VK commands
    // to copy staging buffer → staging image → blit to destination.
    bool complete_cuda_d2h(VkImage dst_image, VkCommandBuffer cmd)
    {
        if (!cuda_d2h_active_) return false;
        cuda_d2h_active_ = false;

        try {
            cudaSetDevice(cuda_vk_src_device_);
            cudaStreamSynchronize(cuda_vk_src_stream_);

            if (cuda_d2h_using_pinned_) {
                size_t pixel_size = cuda_d2h_16bit_ ? 8 : 4;
                size_t total_bytes = static_cast<size_t>(cuda_d2h_sub_w_) * cuda_d2h_sub_h_ * pixel_size;
                memcpy(staging_mapped_, cuda_vk_pinned_host_, total_bytes);
            }

            uint32_t sub_w = cuda_d2h_sub_w_;
            uint32_t sub_h = cuda_d2h_sub_h_;
            bool use_16bit = cuda_d2h_16bit_;

            // ── VK staging buffer → staging image → blit to swapchain ───────
            ensure_cuda_staging_image(sub_w, sub_h, use_16bit);
            if (cuda_staging_image_ == VK_NULL_HANDLE)
                return false;

            // Transition staging image UNDEFINED → TRANSFER_DST
            {
                VkImageMemoryBarrier b{};
                b.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                b.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
                b.newLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                b.image               = cuda_staging_image_;
                b.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                b.srcAccessMask       = 0;
                b.dstAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
                vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                     VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
            }

            // Copy buffer → staging image
            VkBufferImageCopy region{};
            region.bufferOffset      = 0;
            region.bufferRowLength   = 0;
            region.bufferImageHeight = 0;
            region.imageSubresource  = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            region.imageOffset       = {0, 0, 0};
            region.imageExtent       = {sub_w, sub_h, 1};

            vkCmdCopyBufferToImage(cmd, staging_buffer_, cuda_staging_image_,
                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

            // Transition staging image TRANSFER_DST → TRANSFER_SRC
            {
                VkImageMemoryBarrier b{};
                b.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                b.oldLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                b.newLayout           = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
                b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                b.image               = cuda_staging_image_;
                b.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                b.srcAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
                b.dstAccessMask       = VK_ACCESS_TRANSFER_READ_BIT;
                vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                     VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
            }

            // Blit staging image → dst_image
            VkImageBlit blit{};
            blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            int32_t copy_w = static_cast<int32_t>((std::min)(sub_w, swapchain_.width));
            int32_t copy_h = static_cast<int32_t>((std::min)(sub_h, swapchain_.height));
            blit.srcOffsets[0] = {0, 0, 0};
            blit.srcOffsets[1] = {copy_w, copy_h, 1};
            blit.dstOffsets[0] = {0, 0, 0};
            blit.dstOffsets[1] = {copy_w, copy_h, 1};
            vkCmdBlitImage(cmd, cuda_staging_image_,
                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           dst_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                           1, &blit, VK_FILTER_LINEAR);

            return true;
        } catch (const std::exception& e) {
            CASPAR_LOG(warning) << print() << L" CUDA VK complete_cuda_d2h failed: " << e.what();
            return false;
        }
    }

    // Combined: begin + complete in one call (for first-frame fallback).
    bool cross_gpu_cuda_transfer(HANDLE src_handle, uint32_t width, uint32_t height,
                                 bool use_16bit, unsigned long long vk_alloc_size,
                                 const uint8_t* src_luid, VkImage dst_image, VkCommandBuffer cmd)
    {
        if (!begin_cuda_d2h(src_handle, width, height, use_16bit, vk_alloc_size, src_luid))
            return false;
        return complete_cuda_d2h(dst_image, cmd);
    }

    void cleanup_cuda_vk_peer()
    {
        // Sync the stream BEFORE freeing any CUDA objects that may still be
        // referenced by in-flight async work (e.g., cudaMemcpy2DFromArrayAsync
        // kicked by begin_cuda_d2h).  If present_frame returned early due to
        // shutdown, the async D2H was never synced by complete_cuda_d2h.
        // Destroying mipmap arrays while the stream references them is a
        // use-after-free on the GPU.
        if (cuda_vk_src_stream_) {
            cudaSetDevice(cuda_vk_src_device_);
            cudaStreamSynchronize(cuda_vk_src_stream_);
        }
        cuda_d2h_active_ = false;

        // Free cached CUDA imports
        for (int i = 0; i < cuda_vk_import_count_; ++i) {
            auto& e = cuda_vk_imports_[i];
            if (e.mipmap) cudaFreeMipmappedArray(e.mipmap);
            if (e.ext_mem) cudaDestroyExternalMemory(e.ext_mem);
            e = {};
        }
        cuda_vk_import_count_ = 0;

        // Unregister VK staging buffer from CUDA
        if (cuda_vk_host_registered_ && staging_mapped_) {
            cudaSetDevice(cuda_vk_src_device_);
            cudaHostUnregister(staging_mapped_);
            cuda_vk_host_registered_ = false;
        }
        // Free pinned fallback buffer
        if (cuda_vk_pinned_host_) {
            cudaSetDevice(cuda_vk_src_device_);
            cudaFreeHost(cuda_vk_pinned_host_);
            cuda_vk_pinned_host_ = nullptr;
        }
        if (cuda_vk_src_stream_) {
            cudaSetDevice(cuda_vk_src_device_);
            cudaStreamDestroy(cuda_vk_src_stream_);
            cuda_vk_src_stream_ = nullptr;
        }
        if (cuda_staging_image_ != VK_NULL_HANDLE) {
            auto dev = device_->device();
            vkDestroyImage(dev, cuda_staging_image_, nullptr);
            vkFreeMemory(dev, cuda_staging_memory_, nullptr);
            cuda_staging_image_  = VK_NULL_HANDLE;
            cuda_staging_memory_ = VK_NULL_HANDLE;
        }
    }
#endif // CASPAR_CUDA_PEER_ENABLED

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
            if (staging_mapped_) {
                vkUnmapMemory(dev, staging_memory_);
                staging_mapped_ = nullptr;
            }
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

        // Map and copy — reuse persistent mapping if CUDA path already mapped it
        void* mapped = staging_mapped_;
        if (!mapped) {
            if (vkMapMemory(dev, staging_memory_, 0, size, 0, &mapped) != VK_SUCCESS) {
                CASPAR_LOG(error) << print() << L" Failed to map staging memory";
                return;
            }
        }
        memcpy(mapped, pixels, size);
        if (!staging_mapped_)
            vkUnmapMemory(dev, staging_memory_);

        // Clear the destination image first if the copy won't cover the entire
        // surface, so borders don't contain undefined content.
        {
            VkClearColorValue clear_color = {{0.0f, 0.0f, 0.0f, 1.0f}};
            VkImageSubresourceRange range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCmdClearColorImage(cmd, dst_image,
                                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clear_color, 1, &range);
        }

        // Copy buffer → image
        // For cross-GPU with subregion, use bufferRowLength and bufferOffset
        // to copy only the cropped region from the staging buffer.
        VkBufferImageCopy region{};
        region.bufferImageHeight = 0;
        region.imageSubresource  = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};

        if (has_subregion()) {
            const int32_t fw = format_desc_.width;
            const int32_t fh = format_desc_.height;
            const int32_t dw = static_cast<int32_t>(swapchain_.width);
            const int32_t dh = static_cast<int32_t>(swapchain_.height);

            int32_t sx = (std::clamp)(config_.src_x, 0, fw);
            int32_t sy = (std::clamp)(config_.src_y, 0, fh);
            int32_t sw = config_.region_w > 0 ? config_.region_w : fw - sx;
            int32_t sh = config_.region_h > 0 ? config_.region_h : fh - sy;
            sw = (std::min)(sw, fw - sx);
            sh = (std::min)(sh, fh - sy);
            // Clamp to destination
            int32_t dx = (std::clamp)(config_.dest_x, 0, dw);
            int32_t dy = (std::clamp)(config_.dest_y, 0, dh);
            sw = (std::min)(sw, dw - dx);
            sh = (std::min)(sh, dh - dy);

            if (sw <= 0 || sh <= 0) return;

            region.bufferOffset    = (static_cast<VkDeviceSize>(sy) * fw + sx) * 4;
            region.bufferRowLength = static_cast<uint32_t>(fw);
            region.imageOffset     = {dx, dy, 0};
            region.imageExtent     = {static_cast<uint32_t>(sw), static_cast<uint32_t>(sh), 1};
        } else {
            // No subregion: copy full frame (original behavior).
            // Use actual frame dimensions — the pixel buffer is format_desc_ sized.
            // Clamp copy region to destination image bounds. The format_desc_
            // describes the channel resolution (e.g. 7680x2160) which may far exceed
            // the swapchain/destination dimensions (e.g. 2560x1440). An oversized
            // vkCmdCopyBufferToImage region is a Vulkan spec violation that causes TDR.
            uint32_t copy_w = (std::min)(static_cast<uint32_t>(format_desc_.width),  swapchain_.width);
            uint32_t copy_h = (std::min)(static_cast<uint32_t>(format_desc_.height), swapchain_.height);
            region.bufferOffset    = 0;
            region.bufferRowLength = static_cast<uint32_t>(format_desc_.width);
            region.imageOffset     = {0, 0, 0};
            region.imageExtent     = {copy_w, copy_h, 1};
        }

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
                if ((wParam & 0xFFF0) == SC_CLOSE ||
                    (wParam & 0xFFF0) == SC_MINIMIZE ||
                    (wParam & 0xFFF0) == SC_RESTORE)
                    return 0;
                break;
            case WM_MOUSEACTIVATE:
                return MA_NOACTIVATEANDEAT; // Don't steal focus on click
            case WM_SETCURSOR:
                SetCursor(nullptr); // Hide cursor on output windows
                return TRUE;
            case WM_WINDOWPOSCHANGING: {
                // Prevent DWM or other windows from minimizing/resizing/moving
                // or changing the z-order of the output window.  When other windows
                // open (e.g. screen consumer), Windows may try to insert them above
                // our TOPMOST window.  Blocking SWP_NOZORDER changes and forcing
                // our window to stay TOPMOST prevents the DWM compositor from
                // showing desktop fragments through the Vulkan swapchain surface.
                auto* pos = reinterpret_cast<WINDOWPOS*>(lParam);
                pos->flags |= SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER;
                pos->hwndInsertAfter = HWND_TOPMOST;
                // Block hide (Win+D, Aero Peek, Show Desktop)
                if (pos->flags & SWP_HIDEWINDOW)
                    pos->flags &= ~SWP_HIDEWINDOW;
                return 0;
            }
            case WM_ACTIVATE:
                if (LOWORD(wParam) == WA_INACTIVE) {
                    // Another window stole focus — re-assert TOPMOST so the output
                    // stays on top without taking focus back.
                    SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0,
                                 SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE);
                }
                return 0;
            case WM_SIZE:
                if (wParam == SIZE_MINIMIZED)
                    return 0; // Block minimize
                break;
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
            case WM_COPYDATA: {
                // ─── Debug capture: write raw frame data to file ───
                // Used by automated test runners. Sends WM_COPYDATA with the output
                // file path. Server writes raw pixel data (RGBA16F or BGRA8) with a
                // 16-byte header. Returns 1 on success, 0 if no data available yet.
                auto* self = reinterpret_cast<vulkan_output_consumer*>(
                    GetWindowLongPtr(hwnd, GWLP_USERDATA));
                if (!self)
                    return 0;

                auto* cds = reinterpret_cast<const COPYDATASTRUCT*>(lParam);
                if (!cds || cds->dwData != 0x43565031) // "CVP1" magic
                    return DefWindowProcW(hwnd, msg, wParam, lParam);

                // Enable capture on first request
                self->debug_capture_enabled_.store(true, std::memory_order_release);

                // Extract file path from COPYDATASTRUCT (UTF-16)
                if (!cds->lpData || cds->cbData < 4)
                    return 0;
                std::wstring path(static_cast<const wchar_t*>(cds->lpData),
                                  cds->cbData / sizeof(wchar_t));
                // Remove null terminator if present
                while (!path.empty() && path.back() == L'\0')
                    path.pop_back();
                if (path.empty())
                    return 0;

                // Write the frame data to file under lock
                std::lock_guard<std::mutex> lock(self->debug_frame_mutex_);
                if (self->debug_frame_data_.empty() || self->debug_frame_w_ == 0)
                    return 0;

                HANDLE hFile = CreateFileW(path.c_str(), GENERIC_WRITE, 0, nullptr,
                                           CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
                if (hFile == INVALID_HANDLE_VALUE)
                    return 0;

                // Header: 4-byte magic + uint32 width + uint32 height + uint32 format
                // Format: 0 = BGRA8 (4 bytes/px), 1 = RGBA16F (8 bytes/px)
                struct { char magic[4]; uint32_t w, h, fmt; } header = {
                    {'C','V','P','1'},
                    static_cast<uint32_t>(self->debug_frame_w_),
                    static_cast<uint32_t>(self->debug_frame_h_),
                    self->debug_frame_format_
                };
                DWORD written = 0;
                WriteFile(hFile, &header, sizeof(header), &written, nullptr);
                WriteFile(hFile, self->debug_frame_data_.data(),
                          static_cast<DWORD>(self->debug_frame_data_.size()), &written, nullptr);
                CloseHandle(hFile);
                return 1;
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

            // Set Per-Monitor DPI awareness so EnumDisplayMonitors returns
            // physical pixel coordinates (e.g. 3840x2160) instead of logical
            // pixels scaled by the Windows DPI setting (e.g. 2560x1440 at 150%).
            set_thread_dpi_awareness();

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

            HWND hwnd = CreateWindowExW(WS_EX_TOPMOST | WS_EX_TOOLWINDOW,
                                        L"CasparVulkanOutput", L"CasparCG Vulkan Output",
                                        WS_POPUP | WS_VISIBLE, x, y, w, h,
                                        nullptr, nullptr, GetModuleHandle(nullptr), nullptr);

            // Store instance pointer for WM_COPYDATA debug capture
            SetWindowLongPtr(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(this));

            // ── DWM protection: prevent Aero Peek / Show Desktop from hiding this window ──
            {
                BOOL exclude_peek = TRUE;
                DwmSetWindowAttribute(hwnd, DWMWA_EXCLUDED_FROM_PEEK,
                                      &exclude_peek, sizeof(exclude_peek));
                BOOL disallow_peek = TRUE;
                DwmSetWindowAttribute(hwnd, DWMWA_DISALLOW_PEEK,
                                      &disallow_peek, sizeof(disallow_peek));
            }

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
        // Wake any present thread blocked in frame_cache_->submit_frame() or
        // the pump cv. Without this, a consumer parked waiting for a peer
        // consumer's stalled transfer would never see running_ flip and the
        // present_thread join would time out.
        if (frame_cache_)
            frame_cache_->request_shutdown();

        // Join watchdog first — it only sleeps 1s so this is fast.
        if (watchdog_thread_.joinable()) {
            try { watchdog_thread_.join(); } catch (...) {}
        }

        // Join present thread with a hard timeout. The present thread can get stuck
        // in cross-GPU dispatch_sync (ogl_device or affinity_ctx) if those executors
        // are blocked during shutdown. Without a timeout, this makes the process
        // unkillable (zombie).
        //
        // CRITICAL: We MUST NOT detach() a present thread that holds Vulkan/CUDA/GL
        // resources (imported external memory, timeline semaphores, swapchain images,
        // CUDA peer contexts). A detached zombie thread continues poking the NVIDIA
        // kernel-mode driver with stale handles; after a few minutes the WDDM scheduler
        // deadlocks and HANGS THE ENTIRE MACHINE (observed: Kernel-Power 41 critical,
        // no minidump, hard reset required).
        //
        // The only safe action when the present thread refuses to exit is
        // TerminateProcess(). std::_Exit / ExitProcess are NOT sufficient because
        // they run DllMain(DLL_PROCESS_DETACH) on every loaded DLL — the NVIDIA
        // driver DLL will wait for our zombie thread inside its DllMain and
        // deadlock inside ExitProcess. TerminateProcess bypasses DllMain entirely.
        if (present_thread_.joinable()) {
            bool joined = false;
            try {
                auto join_future = std::async(std::launch::async, [this] {
                    present_thread_.join();
                });
                // Poll the join every second so we can log when it actually exits
                // and see if it just took longer than the previous 5s budget.
                for (int sec = 1; sec <= 5; ++sec) {
                    if (join_future.wait_for(std::chrono::seconds(1)) == std::future_status::ready) {
                        joined = true;
                        break;
                    }
                }
                if (!joined) {
                    CASPAR_LOG(fatal) << print() << L" Present thread join timed out (5s). "
                                                    L"Calling TerminateProcess to prevent WDDM kernel deadlock "
                                                    L"(detaching or ExitProcess can hang the OS via DllMain).";
                    boost::log::core::get()->flush();
                    ::TerminateProcess(::GetCurrentProcess(), 0);
                }
            } catch (...) {
                CASPAR_LOG(fatal) << print() << L" Present thread join threw — calling TerminateProcess.";
                boost::log::core::get()->flush();
                ::TerminateProcess(::GetCurrentProcess(), 0);
            }
            (void)joined;
        }

        if (device_) {
            auto dev = device_->device();

            // Wait for GPU to finish with a hard timeout to prevent unkillable process.
            // NVIDIA driver can hang indefinitely in vkQueueWaitIdle when display timing
            // is disrupted (disconnected, TDR, or GL contention deadlock).
            // If timeout fires, the GPU subsystem is wedged — any further VK/GL/CUDA
            // calls from other consumers' destructors will hang the WDDM scheduler and
            // can take down the entire OS. TerminateProcess to bypass DllMain.
            bool gpu_stuck = device_dead_.load();
            if (!gpu_stuck) {
                auto idle_future = std::async(std::launch::async, [this] {
                    return vkQueueWaitIdle(my_queue_);
                });
                if (idle_future.wait_for(std::chrono::seconds(2)) == std::future_status::timeout) {
                    CASPAR_LOG(fatal) << print() << L" vkQueueWaitIdle timed out (2s). "
                                                    L"GPU subsystem wedged — calling TerminateProcess "
                                                    L"to prevent WDDM kernel deadlock.";
                    boost::log::core::get()->flush();
                    ::TerminateProcess(::GetCurrentProcess(), 0);
                } else {
                    idle_future.get(); // Collect result (ignore errors during shutdown)
                }
            }

            if (!gpu_stuck) {
                if (staging_mapped_) {
                    vkUnmapMemory(dev, staging_memory_);
                    staging_mapped_ = nullptr;
                }
                if (staging_buffer_ != VK_NULL_HANDLE) {
                    vkDestroyBuffer(dev, staging_buffer_, nullptr);
                    vkFreeMemory(dev, staging_memory_, nullptr);
                }

                destroy_debug_readback_buffer();

                // Clean up VK import cache
                for (auto& imp : vk_import_cache_) {
                    if (imp.view != VK_NULL_HANDLE)
                        vkDestroyImageView(dev, imp.view, nullptr);
                    if (imp.image != VK_NULL_HANDLE)
                        vkDestroyImage(dev, imp.image, nullptr);
                    if (imp.memory != VK_NULL_HANDLE)
                        vkFreeMemory(dev, imp.memory, nullptr);
                    if (imp.owned_handle)
                        CloseHandle(imp.owned_handle);
                }
                vk_import_cache_.clear();

#ifdef CASPAR_CUDA_PEER_ENABLED
                cleanup_cuda_vk_peer();
#endif

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
                // GPU is stuck but vkDestroy* calls are safe on a lost device — they
                // are no-ops or return immediately. Using reset() instead of release()
                // ensures Vulkan objects are properly freed rather than leaked.
                color_pipeline_.reset();
                frame_cache_.reset(); // shared_ptr — safe to reset even if stuck
                device_.reset();
            }
        }

        // Remove injected EDID before releasing NvAPI
        if (injected_edid_display_id_ != 0 && nvapi_) {
            nvapi_->remove_edid(config_.gpu_index, injected_edid_display_id_);
            injected_edid_display_id_ = 0;
        }

        // Disable hardware HDR before releasing NvAPI
        if (hw_hdr_active_ && nvapi_display_id_ != 0 && nvapi_) {
            nvapi_->disable_hdr_output(nvapi_display_id_);
            hw_hdr_active_ = false;
            nvapi_display_id_ = 0;
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
    bool                             mixer_auto_color_convert_ = true;
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

    // Frame buffer (frame + its associated timeline generation)
    std::queue<std::pair<core::const_frame, uint64_t>> buffer_;
    std::mutex                       buffer_mutex_;
    std::condition_variable          buffer_cv_;

    // Present thread
    std::thread                      present_thread_;
    std::atomic<bool>                running_{false};
    std::atomic<bool>                gate_open_{false}; // True once all peers on this GPU are ready

    // FSE window message thread
    std::thread                      fse_msg_thread_;
    std::atomic<bool>                fse_msg_running_{true};

    // Display hot-plug
    std::atomic<bool>                display_lost_{false};
    std::atomic<bool>                device_dead_{false}; // TDR — device permanently invalid
    bool                             adapter_mismatch_{false}; // Display on different GPU — no Vulkan presentation
    uint64_t                         hotplug_retry_counter_ = 0;
    std::atomic<uint64_t>                frames_presented_{0};
    std::atomic<uint64_t>                frame_generation_{0};  // Monotonic counter for frame cache coordination

    // Present barrier
    bool                             present_barrier_enabled_ = false;
    std::shared_ptr<void>            sync_group_token_;  // Software present barrier membership
    bool                             fse_acquired_ = false; // VK_EXT_full_screen_exclusive acquired

    // Hardware HDR (NvAPI display engine)
    uint32_t                         nvapi_display_id_ = 0;  // NvAPI display ID for this output
    bool                             hw_hdr_active_ = false;  // Hardware EOTF active (skip compute shader EOTF)

    // Output identification
    std::atomic<int>                 identify_frames_remaining_{0};

    // Timing
    caspar::timer                    tick_timer_;
    int64_t                          pacer_epoch_ns_ = 0; // First-frame epoch for phase-aligned pacing

    // FPS counter
    std::chrono::steady_clock::time_point fps_update_time_ = std::chrono::steady_clock::now();
    int                              fps_frame_count_ = 0;

    // Diagnostic counters (atomic so send/present threads can both increment)
    std::atomic<int> diag_sends_{0};      // calls to send()
    std::atomic<int> diag_presents_{0};   // entries to present_frame
    std::atomic<int> diag_submits_{0};    // successful vkQueueSubmits (reset by fps counter)
    std::atomic<int> diag_drops_{0};      // dropped frames (buffer full in send())
    std::atomic<uint64_t> watchdog_submits_{0}; // monotonic submit counter for watchdog (never reset)
    int              diag_buf_size_last_ = 0; // last buffer size before pop
    // Stage marker for watchdog: incremented at each milestone in present_frame.
    // Values: 0=idle, 1=entered, 2=waitFences, 3=acquired, 4=cache_submit,
    // 5=blit_recorded, 6=submitted, 7=presented
    std::atomic<int> diag_stage_{0};
    std::thread      watchdog_thread_;

    // Periodic 5s TIMING log
    std::chrono::steady_clock::time_point diag_timing_start_ = std::chrono::steady_clock::now();
    int diag_timing_frames_ = 0;



    // ─── Debug capture (WM_COPYDATA) — GPU readback for post-conversion frames ───
    // DEBUGGING TOOL: Zero-cost when disabled (one relaxed atomic load per frame).
    // Enabled on first WM_COPYDATA received. When active, records an additional
    // vkCmdCopyImageToBuffer after color conversion to read back the final output.
    // The readback has 1-frame latency (frame N's data is available at frame N+1's fence).
    // Data is written to file on WM_COPYDATA request (binary: header + raw pixels).
    std::atomic<bool>                debug_capture_enabled_{false};
    std::mutex                       debug_frame_mutex_;
    std::vector<uint8_t>             debug_frame_data_;       // Raw pixels (RGBA16F or BGRA8)
    int                              debug_frame_w_{0};
    int                              debug_frame_h_{0};
    uint32_t                         debug_frame_format_{0};  // 0=BGRA8, 1=RGBA16F
    VkBuffer                         debug_readback_buffer_   = VK_NULL_HANDLE;
    VkDeviceMemory                   debug_readback_memory_   = VK_NULL_HANDLE;
    void*                            debug_readback_mapped_   = nullptr;
    size_t                           debug_readback_size_     = 0;
    uint32_t                         debug_readback_width_    = 0;
    uint32_t                         debug_readback_height_   = 0;
    bool                             debug_readback_pending_  = false; // True if a readback was recorded last frame

    // CPU staging
    VkBuffer                         staging_buffer_      = VK_NULL_HANDLE;
    VkDeviceMemory                   staging_memory_      = VK_NULL_HANDLE;
    size_t                           staging_buffer_size_ = 0;
    void*                            staging_mapped_      = nullptr; // persistent map

    // ─── VK-native import cache ─────────────────────────────────────────────
    std::vector<vk_imported_image>   vk_import_cache_;
    bool                             vk_mixer_logged_ = false; // Log once when VK mixer detected
    bool                             vk_import_fail_logged_ = false; // Log VK import failure once (cross-GPU)
    uint64_t                         present_path_counter_ = 0;    // Log present path every N frames
    bool                             send_diag_logged_ = false;    // Log once the send() path decision
    uint64_t                         send_diag_counter_ = 0;       // Periodic send() path logging

#ifdef CASPAR_CUDA_PEER_ENABLED
    // ─── Cross-GPU CUDA VK transfer state ───────────────────────────────────
    bool                   cuda_vk_peer_init_   = false;
    bool                   cuda_vk_peer_failed_ = false;
    int                    cuda_vk_src_device_  = -1;
    cudaStream_t           cuda_vk_src_stream_  = nullptr;
    size_t                 cuda_vk_total_bytes_  = 0;
    size_t                 cuda_vk_row_bytes_    = 0;
    uint32_t               cuda_vk_height_       = 0;
    bool                   cuda_vk_host_registered_ = false;
    void*                  cuda_vk_pinned_host_     = nullptr;

    // Early D2H kick state: async D2H is started before fence wait, completed after.
    bool                   cuda_d2h_active_       = false;  // async D2H in flight
    bool                   cuda_d2h_using_pinned_ = false;  // used pinned fallback
    uint32_t               cuda_d2h_sub_w_        = 0;      // subregion width for VK blit
    uint32_t               cuda_d2h_sub_h_        = 0;      // subregion height for VK blit
    bool                   cuda_d2h_16bit_        = false;   // format for staging image

    // Cache of CUDA-imported VK textures (avoids costly reimport per frame).
    // The mixer uses triple buffering with pooled attachments, so after warmup
    // we see a stable set of handles. Keep enough slots for the initial burst
    // before the pool stabilizes (3 slots × ~2 attachments each = 6).
    static constexpr int kMaxCudaVkImports = 8;
    struct cuda_vk_import_entry {
        HANDLE               handle  = nullptr;
        cudaExternalMemory_t ext_mem = nullptr;
        cudaMipmappedArray_t mipmap  = nullptr;
        cudaArray_t          array   = nullptr;
    };
    cuda_vk_import_entry   cuda_vk_imports_[kMaxCudaVkImports] = {};
    int                    cuda_vk_import_count_ = 0;

    VkImage                cuda_staging_image_    = VK_NULL_HANDLE;
    VkDeviceMemory         cuda_staging_memory_   = VK_NULL_HANDLE;
    uint32_t               cuda_staging_w_        = 0;
    uint32_t               cuda_staging_h_        = 0;
#endif

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
        else if (boost::iequals(params[i], L"DELAY_MS") && i + 1 < params.size())
            config.delay_ms = std::stod(params[++i]);
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
