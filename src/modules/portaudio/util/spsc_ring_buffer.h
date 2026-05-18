/*
 * Copyright (c) 2025 CasparCG Contributors
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
#include <cstddef>
#include <cstring>
#include <vector>

namespace caspar { namespace portaudio {

/// Lock-free Single-Producer Single-Consumer ring buffer for audio samples.
/// Producer: CasparCG channel thread (push via send())
/// Consumer: PortAudio/ASIO callback thread (pull in stream callback)
///
/// Stores interleaved int32_t samples. Capacity must be a power of two.
class spsc_ring_buffer
{
  public:
    explicit spsc_ring_buffer(size_t capacity_samples)
        : capacity_(next_power_of_two(capacity_samples))
        , mask_(capacity_ - 1)
        , buffer_(capacity_, 0)
    {
    }

    /// Returns the number of samples available for reading.
    size_t read_available() const
    {
        auto w = write_pos_.load(std::memory_order_acquire);
        auto r = read_pos_.load(std::memory_order_relaxed);
        return w - r;
    }

    /// Returns the number of sample slots available for writing.
    size_t write_available() const
    {
        auto r = read_pos_.load(std::memory_order_acquire);
        auto w = write_pos_.load(std::memory_order_relaxed);
        return capacity_ - (w - r);
    }

    /// Write samples into the ring buffer. Returns number of samples actually written.
    /// Called from the producer thread only.
    size_t write(const int32_t* data, size_t count)
    {
        size_t avail = write_available();
        size_t to_write = (count < avail) ? count : avail;
        if (to_write == 0)
            return 0;

        auto w = write_pos_.load(std::memory_order_relaxed);
        size_t idx = w & mask_;

        // Handle wrap-around
        size_t first_chunk = capacity_ - idx;
        if (first_chunk >= to_write) {
            std::memcpy(&buffer_[idx], data, to_write * sizeof(int32_t));
        } else {
            std::memcpy(&buffer_[idx], data, first_chunk * sizeof(int32_t));
            std::memcpy(&buffer_[0], data + first_chunk, (to_write - first_chunk) * sizeof(int32_t));
        }

        write_pos_.store(w + to_write, std::memory_order_release);
        return to_write;
    }

    /// Read samples from the ring buffer. Returns number of samples actually read.
    /// Called from the consumer (audio callback) thread only.
    size_t read(int32_t* data, size_t count)
    {
        size_t avail = read_available();
        size_t to_read = (count < avail) ? count : avail;
        if (to_read == 0)
            return 0;

        auto r = read_pos_.load(std::memory_order_relaxed);
        size_t idx = r & mask_;

        // Handle wrap-around
        size_t first_chunk = capacity_ - idx;
        if (first_chunk >= to_read) {
            std::memcpy(data, &buffer_[idx], to_read * sizeof(int32_t));
        } else {
            std::memcpy(data, &buffer_[idx], first_chunk * sizeof(int32_t));
            std::memcpy(data + first_chunk, &buffer_[0], (to_read - first_chunk) * sizeof(int32_t));
        }

        read_pos_.store(r + to_read, std::memory_order_release);
        return to_read;
    }

    /// Reset the ring buffer to empty state. NOT thread-safe — call only when stream is stopped.
    void reset()
    {
        write_pos_.store(0, std::memory_order_relaxed);
        read_pos_.store(0, std::memory_order_relaxed);
    }

    size_t capacity() const { return capacity_; }

  private:
    static size_t next_power_of_two(size_t v)
    {
        v--;
        v |= v >> 1;
        v |= v >> 2;
        v |= v >> 4;
        v |= v >> 8;
        v |= v >> 16;
        v |= v >> 32;
        v++;
        return v < 64 ? 64 : v;
    }

    const size_t           capacity_;
    const size_t           mask_;
    std::vector<int32_t>   buffer_;
    alignas(64) std::atomic<size_t> write_pos_{0};
    alignas(64) std::atomic<size_t> read_pos_{0};
};

}} // namespace caspar::portaudio
