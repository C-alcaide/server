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

// Page-locked output-buffer pool for the DeckLink consumer.
//
// The DeckLink driver DMAs the scheduled frame's memory to the card; when that
// memory is page-locked the driver avoids a per-frame pin/unpin (and, for the
// GPU strategies, cudaMemcpyAsync can proceed without an implicit pin). This is
// the shared Tier-1 foundation used by every strategy (CPU, cuda_vk, vk_readback,
// OGL) and the buffer that Tier-2 (NVIDIA DVP) registers once per allocation.
//
// acquire() returns a std::shared_ptr<void> whose deleter returns the buffer to
// the free list; the page-locked allocation itself is kept and reused. The
// allocations outlive the pool object (they are owned by an internal state block
// held by both the pool and every outstanding buffer), so a frame still in the
// DeckLink queue at shutdown stays valid until ScheduledFrameCompleted releases
// its shared_ptr.
#pragma once

#include <cstddef>
#include <memory>
#include <mutex>
#include <vector>

namespace caspar { namespace decklink {

class gpu_output_buffer_pool
{
  public:
    enum class pin_kind
    {
        host_locked, // VirtualAlloc + VirtualLock (no CUDA; any build)
        cuda_pinned, // cudaMallocHost (fast DtoH; CUDA builds only)
    };

    // buffer_bytes  : size of every pooled buffer (sized to the largest frame).
    // initial_count : buffers pre-allocated up front (>= 2; typically buffer depth + 1).
    // kind          : page-locking backend.
    gpu_output_buffer_pool(std::size_t buffer_bytes, int initial_count, pin_kind kind);
    ~gpu_output_buffer_pool();

    gpu_output_buffer_pool(const gpu_output_buffer_pool&)            = delete;
    gpu_output_buffer_pool& operator=(const gpu_output_buffer_pool&) = delete;

    // Acquire a pooled buffer that can hold at least `bytes`. If `bytes` exceeds
    // the pool's buffer size a one-off (still page-locked, not pooled) buffer is
    // returned. Returns an empty shared_ptr on allocation failure.
    std::shared_ptr<void> acquire(std::size_t bytes);

    std::size_t buffer_bytes() const { return buffer_bytes_; }
    pin_kind    kind() const { return kind_; }
    // True when every pooled allocation was successfully page-locked.
    bool pinned() const;

  private:
    struct state;
    std::shared_ptr<state> state_;
    std::size_t            buffer_bytes_;
    pin_kind               kind_;
};

}} // namespace caspar::decklink
