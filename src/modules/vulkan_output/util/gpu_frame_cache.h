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

#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>

#define VK_USE_PLATFORM_WIN32_KHR
#include <vulkan/vulkan.h>

namespace caspar { namespace accelerator { namespace ogl {
class device;
}}} // namespace caspar::accelerator::ogl

namespace caspar { namespace vulkan_output {

class vulkan_device;
class shared_texture_pool;
class interop_context;
class gpu_affinity_context;
#ifdef CASPAR_CUDA_PEER_ENABLED
class cuda_peer_transfer;
#endif

// Per-GPU frame cache: ensures only one OGL→VK transfer (or one cross-GPU
// CUDA peer / PBO copy) happens per GPU per frame, regardless of how many
// vulkan_output consumers share that GPU.
//
// For an 8-output setup across 2 GPUs this reduces PCIe/NVLink traffic from
// 4× to 1× per remote GPU.
//
// Usage (from vulkan_output_consumer):
//   1. send(): cache->submit_frame(frame)   — first caller does the transfer,
//                                              others wait on the barrier
//   2. present_frame(): use cache->pool()   — all consumers read from same pool
//   3. present_frame(): cache->frame_done() — last consumer signals completion
//
// Thread safety: submit_frame() may be called concurrently from multiple
// consumer executor threads.  Internal synchronisation ensures only one
// transfer runs per frame generation.
class gpu_frame_cache
{
  public:
    // Get or create a cache for the given GPU + dimensions + format.
    // Returns the same cache for all consumers targeting the same GPU
    // with the same frame dimensions.
    static std::shared_ptr<gpu_frame_cache> get(
        int                                      gpu_index,
        std::shared_ptr<vulkan_device>            device,
        std::shared_ptr<accelerator::ogl::device> ogl_device,
        uint32_t                                  width,
        uint32_t                                  height,
        bool                                      use_16bit);

    ~gpu_frame_cache();

    gpu_frame_cache(const gpu_frame_cache&)            = delete;
    gpu_frame_cache& operator=(const gpu_frame_cache&) = delete;

    // Returns the shared texture pool (valid after construction)
    shared_texture_pool* pool() const { return pool_.get(); }

    // Returns the interop context (may be null if unavailable)
    interop_context* interop_ctx() const { return interop_ctx_.get(); }

    // Returns the affinity context (non-null only for cross-GPU)
    gpu_affinity_context* affinity_ctx() const { return affinity_ctx_.get(); }

#ifdef CASPAR_CUDA_PEER_ENABLED
    cuda_peer_transfer* cuda_peer() const { return cuda_peer_.get(); }
#endif

    // Is this a cross-GPU cache?
    bool is_cross_gpu() const { return cross_gpu_; }

    // Submit a frame for transfer.  The first consumer to call this for a given
    // frame generation performs the actual OGL→VK blit (or cross-GPU transfer).
    // Subsequent callers for the same generation return immediately.
    // After transfer, issues a coordinator submit that bridges the GL→VK binary
    // semaphore to the timeline semaphore.
    // Returns the timeline value that will be signaled when the frame is ready.
    uint64_t submit_frame(uint64_t generation, const std::function<void()>& transfer_fn);

    // Non-blocking: posts transfer work to the pump thread.
    // The pump executes the transfer in the background and signals the timeline
    // semaphore.  Use for same-GPU transfers from send() where the present
    // thread should not block on the blit.
    // Returns the timeline value that will be signaled when complete.
    uint64_t notify_frame(uint64_t generation, std::function<void()> transfer_fn);

    // Timeline semaphore for multi-queue synchronization.
    // Consumers add this to their vkQueueSubmit wait list (with the value returned
    // by submit_frame/notify_frame) to ensure the shared pool data is visible.
    VkSemaphore timeline_semaphore() const { return timeline_sem_; }

    // Current signaled timeline value.  If a consumer's pending value <= this,
    // the GPU-side wait will complete immediately (no stall).
    uint64_t current_timeline_value() const { return timeline_value_.load(std::memory_order_acquire); }

    // Signal that one consumer has finished reading the current frame.
    void frame_done();

    // Number of consumers sharing this cache
    int consumer_count() const { return consumer_count_.load(std::memory_order_relaxed); }

    // Register/unregister consumers
    void add_consumer();
    void remove_consumer();

  private:
    gpu_frame_cache(int                                       gpu_index,
                    std::shared_ptr<vulkan_device>             device,
                    std::shared_ptr<accelerator::ogl::device>  ogl_device,
                    uint32_t                                   width,
                    uint32_t                                   height,
                    bool                                       use_16bit);

    void init_same_gpu(std::shared_ptr<accelerator::ogl::device> ogl_device,
                       uint32_t width, uint32_t height, bool use_16bit);
    void init_cross_gpu(std::shared_ptr<accelerator::ogl::device> ogl_device,
                        uint32_t width, uint32_t height, bool use_16bit);

    int                                      gpu_index_;
    std::shared_ptr<vulkan_device>           device_;
    std::unique_ptr<shared_texture_pool>     pool_;
    std::unique_ptr<interop_context>         interop_ctx_;
    std::unique_ptr<gpu_affinity_context>    affinity_ctx_;
#ifdef CASPAR_CUDA_PEER_ENABLED
    std::unique_ptr<cuda_peer_transfer>      cuda_peer_;
#endif
    bool                                     cross_gpu_ = false;

    // Frame generation tracking (first-caller-wins for blocking submit_frame)
    std::mutex              frame_mutex_;
    std::condition_variable frame_cv_;
    std::atomic<uint64_t>   current_generation_{0};
    bool                    transfer_in_progress_ = false;

    // Timeline semaphore: bridges binary GL→VK semaphore to multi-queue timeline.
    // After each transfer, a coordinator submit waits on the binary semaphore
    // and signals the timeline with the frame generation value.
    VkSemaphore             timeline_sem_ = VK_NULL_HANDLE;
    std::atomic<uint64_t>   timeline_value_{0};
    uint32_t                coord_queue_idx_ = 0;

    // Producer pump thread (same-GPU path: transfer runs in background)
    struct pump_work
    {
        uint64_t                generation;
        std::function<void()>   transfer_fn;
    };
    std::thread             pump_thread_;
    std::mutex              pump_mutex_;
    std::condition_variable pump_cv_;
    std::queue<pump_work>   pump_queue_;
    std::atomic<bool>       pump_running_{false};

    void pump_loop();
    void do_coordinator_submit(uint64_t generation);

    // Consumer tracking
    std::atomic<int>        consumer_count_{0};

    // Registry
    static std::mutex                                         registry_mutex_;
    static std::map<int, std::weak_ptr<gpu_frame_cache>>      registry_;
};

}} // namespace caspar::vulkan_output
