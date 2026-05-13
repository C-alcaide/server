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

#include "gpu_frame_cache.h"
#include "gpu_affinity_context.h"
#include "interop_context.h"
#include "shared_texture_pool.h"
#include "vulkan_device.h"

#ifdef CASPAR_CUDA_PEER_ENABLED
#include "cuda_peer_transfer.h"
#endif

#include <accelerator/ogl/util/device.h>

#include <common/except.h>
#include <common/log.h>

#include <GL/glew.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>

namespace caspar { namespace vulkan_output {

std::mutex                                                        gpu_frame_cache::registry_mutex_;
std::map<gpu_frame_cache::cache_key, std::weak_ptr<gpu_frame_cache>> gpu_frame_cache::registry_;

std::shared_ptr<gpu_frame_cache> gpu_frame_cache::get(
    int                                       gpu_index,
    std::shared_ptr<vulkan_device>             device,
    std::shared_ptr<accelerator::ogl::device>  ogl_device,
    uint32_t                                   width,
    uint32_t                                   height,
    bool                                       use_16bit)
{
    std::lock_guard<std::mutex> lock(registry_mutex_);

    cache_key key{gpu_index, width, height, use_16bit};
    auto it = registry_.find(key);
    if (it != registry_.end()) {
        auto ptr = it->second.lock();
        if (ptr) {
            CASPAR_LOG(info) << L"[vulkan] Sharing frame cache for GPU " << gpu_index
                             << L" " << width << L"x" << height
                             << L" (consumers=" << ptr->consumer_count() << L")";
            return ptr;
        }
        registry_.erase(it);
    }

    // Use shared_ptr with custom deleter to clean up registry
    auto ptr = std::shared_ptr<gpu_frame_cache>(
        new gpu_frame_cache(gpu_index, device, ogl_device, width, height, use_16bit),
        [key, gpu_index](gpu_frame_cache* cache) {
            {
                std::lock_guard<std::mutex> lock(registry_mutex_);
                registry_.erase(key);
            }
            CASPAR_LOG(info) << L"[vulkan] Releasing frame cache for GPU " << gpu_index;
            delete cache;
        });

    registry_[key] = ptr;
    return ptr;
}

gpu_frame_cache::gpu_frame_cache(
    int                                       gpu_index,
    std::shared_ptr<vulkan_device>             device,
    std::shared_ptr<accelerator::ogl::device>  ogl_device,
    uint32_t                                   width,
    uint32_t                                   height,
    bool                                       use_16bit)
    : gpu_index_(gpu_index)
    , device_(std::move(device))
{
    // Acquire a dedicated queue for coordinator submits (binary → timeline bridge)
    coord_queue_idx_ = device_->acquire_queue();

    // Create timeline semaphore (Vulkan 1.2+ core feature)
    VkSemaphoreTypeCreateInfo type_info{};
    type_info.sType         = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    type_info.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    type_info.initialValue  = 0;

    VkSemaphoreCreateInfo sem_info{};
    sem_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    sem_info.pNext = &type_info;

    if (vkCreateSemaphore(device_->device(), &sem_info, nullptr, &timeline_sem_) != VK_SUCCESS) {
        CASPAR_THROW_EXCEPTION(caspar_exception() << msg_info("Failed to create timeline semaphore"));
    }

    // Check GPU match: OGL mixer vs Vulkan output
    bool gpu_match = true;
    if (device_->device_luid_valid() && ogl_device) {
        uint8_t ogl_luid[8] = {};
        bool    ogl_luid_valid = false;
        std::string ogl_renderer;
        ogl_device->dispatch_sync([&] {
            auto glGetUnsignedBytevEXT = reinterpret_cast<void(APIENTRY*)(GLenum, GLubyte*)>(
                wglGetProcAddress("glGetUnsignedBytevEXT"));
            if (glGetUnsignedBytevEXT) {
                glGetUnsignedBytevEXT(0x9462 /*GL_DEVICE_LUID_EXT*/, ogl_luid);
                ogl_luid_valid = true;
            }
            // Get GL_RENDERER as fallback for GPU matching
            const char* r = reinterpret_cast<const char*>(glGetString(GL_RENDERER));
            if (r) ogl_renderer = r;
            while (glGetError() != GL_NO_ERROR) {}
        });

        const uint8_t zero_luid[8] = {};

        auto format_luid = [](const uint8_t* luid) -> std::wstring {
            wchar_t buf[32];
            swprintf(buf, 32, L"%02X%02X%02X%02X-%02X%02X%02X%02X",
                     luid[0], luid[1], luid[2], luid[3],
                     luid[4], luid[5], luid[6], luid[7]);
            return buf;
        };

        bool luid_usable = ogl_luid_valid &&
                           memcmp(ogl_luid, zero_luid, 8) != 0 &&
                           memcmp(device_->device_luid(), zero_luid, 8) != 0;

        if (luid_usable) {
            // Primary method: compare LUIDs
            if (memcmp(ogl_luid, device_->device_luid(), 8) != 0)
                gpu_match = false;
            CASPAR_LOG(info) << L"[vulkan] Frame cache GPU " << gpu_index
                             << L": LUID match=" << (gpu_match ? L"yes" : L"no")
                             << L"  OGL=" << format_luid(ogl_luid)
                             << L"  VK=" << format_luid(device_->device_luid());
        } else if (!ogl_renderer.empty()) {
            // Fallback: compare GL_RENDERER against Vulkan device name
            VkPhysicalDeviceProperties vk_props;
            vkGetPhysicalDeviceProperties(device_->physical_device(), &vk_props);
            std::string vk_name(vk_props.deviceName);

            // GL_RENDERER is typically "NVIDIA RTX A4000/PCIe/SSE2" — extract the
            // GPU model by trimming everything from the first '/'.
            auto slash = ogl_renderer.find('/');
            std::string ogl_model = (slash != std::string::npos)
                                        ? ogl_renderer.substr(0, slash)
                                        : ogl_renderer;
            // Trim trailing spaces
            while (!ogl_model.empty() && ogl_model.back() == ' ')
                ogl_model.pop_back();

            // Check if the VK device name contains the OGL model (or vice versa)
            gpu_match = (vk_name.find(ogl_model) != std::string::npos) ||
                        (ogl_model.find(vk_name) != std::string::npos);

            CASPAR_LOG(info) << L"[vulkan] Frame cache GPU " << gpu_index
                             << L": LUID unavailable, renderer match=" << (gpu_match ? L"yes" : L"no")
                             << L"  OGL=\"" << ogl_model.c_str()
                             << L"\"  VK=\"" << vk_name.c_str() << L"\"";
        } else {
            // No identification available — assume cross-GPU to be safe
            gpu_match = false;
            CASPAR_LOG(warning) << L"[vulkan] Frame cache GPU " << gpu_index
                                << L": Cannot identify OGL GPU — assuming cross-GPU.";
        }
    }

    cross_gpu_ = !gpu_match;

    if (gpu_match) {
        init_same_gpu(ogl_device, width, height, use_16bit);
    } else {
        init_cross_gpu(ogl_device, width, height, use_16bit);
    }

    // Start pump thread (handles same-GPU transfers in the background)
    pump_running_ = true;
    pump_thread_ = std::thread([this] { pump_loop(); });
}

void gpu_frame_cache::init_same_gpu(
    std::shared_ptr<accelerator::ogl::device> ogl_device,
    uint32_t width, uint32_t height, bool use_16bit)
{
    pool_ = std::make_unique<shared_texture_pool>(ogl_device, *device_, width, height, use_16bit);
    CASPAR_LOG(info) << L"[vulkan] Frame cache: zero-copy OGL→VK (same GPU " << gpu_index_ << L")"
                     << (use_16bit ? L" 16-bit" : L" 8-bit");

    // Dedicated GL context for blit (avoids blocking mixer thread)
    ogl_device->dispatch_sync([this] {
        interop_ctx_ = std::make_unique<interop_context>();
        if (!interop_ctx_->valid()) {
            CASPAR_LOG(warning) << L"[vulkan] Frame cache: shared GL context unavailable, "
                                   L"blit will use OGL device thread.";
            interop_ctx_.reset();
        }
    });
}

void gpu_frame_cache::init_cross_gpu(
    std::shared_ptr<accelerator::ogl::device> ogl_device,
    uint32_t width, uint32_t height, bool use_16bit)
{
    affinity_ctx_ = std::make_unique<gpu_affinity_context>(gpu_index_, width, height);

    // Verify LUID match (only if both sides report valid non-zero LUIDs)
    if (affinity_ctx_->device_luid_valid() && device_->device_luid_valid()) {
        const uint8_t zero_luid[8] = {};
        bool affinity_zero = memcmp(affinity_ctx_->device_luid(), zero_luid, 8) == 0;
        bool vk_zero       = memcmp(device_->device_luid(), zero_luid, 8) == 0;
        if (!affinity_zero && !vk_zero &&
            memcmp(affinity_ctx_->device_luid(), device_->device_luid(), 8) != 0) {
            CASPAR_LOG(warning) << L"[vulkan] Frame cache: affinity LUID mismatch "
                                   L"(may be harmless with duplicate VkPhysicalDevices). Continuing.";
        }
    }

    // Create shared pool on affinity thread
    affinity_ctx_->dispatch_sync([&] {
        pool_ = std::make_unique<shared_texture_pool>(*device_, width, height, use_16bit);
    });

#ifdef CASPAR_CUDA_PEER_ENABLED
    try {
        int src_cuda_dev = -1;
        ogl_device->dispatch_sync([&] {
            src_cuda_dev = cuda_peer_transfer::cuda_device_for_current_gl_context();
        });
        int dst_cuda_dev = -1;
        affinity_ctx_->dispatch_sync([&] {
            dst_cuda_dev = cuda_peer_transfer::cuda_device_for_current_gl_context();
        });

        if (src_cuda_dev >= 0 && dst_cuda_dev >= 0 && src_cuda_dev != dst_cuda_dev) {
            cuda_peer_ = std::make_unique<cuda_peer_transfer>(
                src_cuda_dev, dst_cuda_dev, width, height, use_16bit);
            CASPAR_LOG(info) << L"[vulkan] Frame cache: CUDA peer DMA (device "
                             << src_cuda_dev << L" → " << dst_cuda_dev << L")";
        }
    } catch (const std::exception& e) {
        CASPAR_LOG(warning) << L"[vulkan] Frame cache: CUDA peer unavailable: " << e.what();
        cuda_peer_.reset();
    }
#endif

    CASPAR_LOG(info) << L"[vulkan] Frame cache: cross-GPU bridge (GPU " << gpu_index_ << L")"
                     << (use_16bit ? L" 16-bit" : L" 8-bit");
}

gpu_frame_cache::~gpu_frame_cache()
{
    // Stop pump thread
    pump_running_ = false;
    pump_cv_.notify_one();
    if (pump_thread_.joinable())
        pump_thread_.join();

    // Wait for coordinator queue to drain with a hard timeout.
    // The coordinator queue can hang indefinitely if a pending VK submit is
    // waiting on a binary GL→VK semaphore that was never signaled (race during
    // shutdown).
    //
    // CRITICAL: We MUST NOT leak the async future / detach a thread that is
    // blocked inside vkQueueWaitIdle. A detached zombie thread continues
    // holding kernel-mode references on the NVIDIA driver; after a few minutes
    // the WDDM scheduler deadlocks and HANGS THE ENTIRE MACHINE (observed:
    // Kernel-Power 41 critical, no minidump, hard reset required).
    //
    // We also cannot use std::_Exit / ExitProcess: those invoke
    // DllMain(DLL_PROCESS_DETACH) on every loaded DLL and the NVIDIA driver
    // DLL will block inside its DllMain waiting for our zombie thread, hanging
    // ExitProcess forever. Only TerminateProcess bypasses DllMain.
    if (device_ && timeline_sem_ != VK_NULL_HANDLE) {
        try {
            auto idle_future = std::async(std::launch::async, [this] {
                return vkQueueWaitIdle(device_->queue(coord_queue_idx_));
            });
            if (idle_future.wait_for(std::chrono::seconds(2)) == std::future_status::timeout) {
                CASPAR_LOG(fatal) << L"[vulkan] Frame cache coordinator vkQueueWaitIdle timed out (2s). "
                                  << L"GPU subsystem wedged — calling TerminateProcess to prevent WDDM kernel deadlock.";
                boost::log::core::get()->flush();
                ::TerminateProcess(::GetCurrentProcess(), 0);
            } else {
                idle_future.get(); // Collect result (may throw on device lost — caught below)
                vkDestroySemaphore(device_->device(), timeline_sem_, nullptr);
            }
        } catch (...) {
            CASPAR_LOG(warning) << L"[vulkan] Frame cache coordinator cleanup threw — ignoring.";
            // Don't destroy timeline_sem_ — device may be lost.
        }
    }
}

uint64_t gpu_frame_cache::submit_frame(uint64_t generation, const std::function<void()>& transfer_fn)
{
    std::unique_lock<std::mutex> lock(frame_mutex_);

    // If this generation is already transferred, return immediately
    if (generation <= current_generation_.load(std::memory_order_acquire))
        return current_generation_.load(std::memory_order_relaxed);

    // If another consumer is currently transferring, wait for it.
    // CRITICAL: bounded wait + shutdown check. If the consumer that won the
    // transfer race is wedged (e.g. waiting on a dead OGL executor during
    // shutdown), an unbounded cv wait here parks this consumer's present
    // thread forever and forces TerminateProcess at shutdown.
    if (transfer_in_progress_) {
        while (transfer_in_progress_ &&
               generation > current_generation_.load(std::memory_order_relaxed) &&
               !shutting_down_.load(std::memory_order_acquire)) {
            frame_cv_.wait_for(lock, std::chrono::milliseconds(100));
        }
        if (shutting_down_.load(std::memory_order_acquire))
            return current_generation_.load(std::memory_order_relaxed);
        if (generation <= current_generation_.load(std::memory_order_relaxed))
            return current_generation_.load(std::memory_order_relaxed);
        // transfer_in_progress_ became false but generation still ours — fall
        // through to perform the transfer.
    }

    // This consumer wins — perform the transfer
    transfer_in_progress_ = true;
    lock.unlock();

    transfer_fn();

    // Coordinator submit: bridge binary GL→VK semaphore to timeline
    do_coordinator_submit(generation);

    lock.lock();
    current_generation_.store(generation, std::memory_order_release);
    transfer_in_progress_ = false;
    timeline_value_.store(generation, std::memory_order_release);
    frame_cv_.notify_all();

    return generation;
}

void gpu_frame_cache::request_shutdown()
{
    shutting_down_.store(true, std::memory_order_release);
    {
        std::lock_guard<std::mutex> lock(frame_mutex_);
    }
    frame_cv_.notify_all();
    {
        std::lock_guard<std::mutex> lock(pump_mutex_);
    }
    pump_cv_.notify_all();
}

uint64_t gpu_frame_cache::notify_frame(uint64_t generation, std::function<void()> transfer_fn)
{
    std::lock_guard<std::mutex> lock(pump_mutex_);
    pump_queue_.push({generation, std::move(transfer_fn)});
    pump_cv_.notify_one();
    return generation;
}

void gpu_frame_cache::pump_loop()
{
    SetThreadDescription(GetCurrentThread(), L"VK Frame Pump");

    while (pump_running_) {
        pump_work work{};
        {
            std::unique_lock<std::mutex> lock(pump_mutex_);
            pump_cv_.wait(lock, [this] { return !pump_queue_.empty() || !pump_running_; });
            if (!pump_running_)
                break;

            // Drain queue and keep only the latest work (skip stale frames)
            while (!pump_queue_.empty()) {
                work = std::move(pump_queue_.front());
                pump_queue_.pop();
            }
        }

        // Skip if already processed (duplicate notification from multiple consumers)
        if (work.generation <= current_generation_.load(std::memory_order_acquire))
            continue;

        // Execute the transfer (blit + signal_gl + swap on the interop/affinity thread)
        work.transfer_fn();

        // Coordinator submit: bridge binary semaphore → timeline
        do_coordinator_submit(work.generation);

        // Update generation (thread-safe via atomic)
        current_generation_.store(work.generation, std::memory_order_release);
        timeline_value_.store(work.generation, std::memory_order_release);
    }
}

void gpu_frame_cache::do_coordinator_submit(uint64_t generation)
{
    if (!pool_ || timeline_sem_ == VK_NULL_HANDLE)
        return;

    VkSemaphore wait_sem = pool_->wait_semaphore_vk();
    if (wait_sem == VK_NULL_HANDLE)
        return;

    VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;

    // Timeline semaphore submit info: values array must match semaphore arrays.
    // Binary semaphore wait value is ignored (set to 0).
    // Timeline semaphore signal value = frame generation.
    uint64_t wait_value   = 0;          // Binary semaphore — value ignored
    uint64_t signal_value = generation;  // Timeline signal = frame generation

    VkTimelineSemaphoreSubmitInfo timeline_info{};
    timeline_info.sType                     = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    timeline_info.waitSemaphoreValueCount   = 1;
    timeline_info.pWaitSemaphoreValues      = &wait_value;
    timeline_info.signalSemaphoreValueCount = 1;
    timeline_info.pSignalSemaphoreValues    = &signal_value;

    VkSubmitInfo submit_info{};
    submit_info.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.pNext                = &timeline_info;
    submit_info.waitSemaphoreCount   = 1;
    submit_info.pWaitSemaphores      = &wait_sem;
    submit_info.pWaitDstStageMask    = &wait_stage;
    submit_info.commandBufferCount   = 0;
    submit_info.signalSemaphoreCount = 1;
    submit_info.pSignalSemaphores    = &timeline_sem_;

    std::lock_guard<std::mutex> lock(device_->queue_mutex_for(coord_queue_idx_));
    VkResult result = vkQueueSubmit(device_->queue(coord_queue_idx_), 1, &submit_info, VK_NULL_HANDLE);
    if (result == VK_ERROR_DEVICE_LOST) {
        CASPAR_LOG(error) << L"[vulkan] Coordinator submit: DEVICE_LOST (TDR) on GPU "
                          << gpu_index_ << L" queue=" << coord_queue_idx_
                          << L" gen=" << generation;
    } else if (result != VK_SUCCESS) {
        CASPAR_LOG(error) << L"[vulkan] Coordinator submit failed (result=" << result
                          << L") on GPU " << gpu_index_ << L" — timeline semaphore may stall consumers.";
    } else {
        CASPAR_LOG(debug) << L"[vulkan] Coordinator submit OK: GPU " << gpu_index_
                           << L" gen=" << generation << L" queue=" << coord_queue_idx_;
    }
}

void gpu_frame_cache::frame_done()
{
    // Placeholder for future read-tracking if needed
}

void gpu_frame_cache::add_consumer()
{
    consumer_count_.fetch_add(1, std::memory_order_relaxed);
}

void gpu_frame_cache::remove_consumer()
{
    consumer_count_.fetch_sub(1, std::memory_order_relaxed);
}

}} // namespace caspar::vulkan_output
