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
    // Returns the generation number of the transferred frame.
    uint64_t submit_frame(uint64_t generation, const std::function<void()>& transfer_fn);

    // Consume the GL→VK semaphore for the current frame.  Returns true for
    // the first caller per generation (who must add the semaphore to their
    // vkQueueSubmit wait list).  Subsequent callers return false — in-order
    // queue execution on the shared VkQueue guarantees they see the completed
    // blit without an explicit wait.
    // MUST be called under device_->queue_mutex() to ensure the consumer that
    // consumes the semaphore also submits first.
    bool try_consume_semaphore();

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

    // Frame generation tracking
    std::mutex              frame_mutex_;
    std::condition_variable frame_cv_;
    uint64_t                current_generation_ = 0;
    bool                    transfer_in_progress_ = false;
    std::atomic<bool>       semaphore_consumed_{false};

    // Consumer tracking
    std::atomic<int>        consumer_count_{0};

    // Registry
    static std::mutex                                         registry_mutex_;
    static std::map<int, std::weak_ptr<gpu_frame_cache>>      registry_;
};

}} // namespace caspar::vulkan_output
