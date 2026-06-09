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
 *
 * This module requires the NVIDIA CUDA Toolkit (https://developer.nvidia.com/cuda-toolkit).
 */

// async_file_writer_linux.cpp
// Linux io_uring implementation of AsyncFileWriter.
// Mirrors the Windows IOCP behaviour: O_DIRECT + sector-aligned ring buffers
// with async write submission and completion polling via io_uring.

#ifndef WIN32

#include "async_file_writer.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <liburing.h>

#include <common/utf.h>

// ---------------------------------------------------------------------------
// Destructor
// ---------------------------------------------------------------------------
AsyncFileWriter::~AsyncFileWriter()
{
    if (fd_ >= 0) {
        drain_all();
        close(nullptr, 0);
    }
    for (int i = 0; i < kRingSize; i++) {
        if (slots_[i].buf) {
            free(slots_[i].buf);
            slots_[i].buf = nullptr;
        }
    }
    if (ring_initialized_) {
        io_uring_queue_exit(&ring_);
        ring_initialized_ = false;
    }
}

// ---------------------------------------------------------------------------
// open()
// ---------------------------------------------------------------------------
bool AsyncFileWriter::open(const wchar_t *path,
                           size_t         slot_bytes,
                           size_t         sector_size)
{
    slot_bytes_  = slot_bytes  ? slot_bytes  : kDefaultSlotBytes;
    sector_size_ = sector_size ? sector_size : kDefaultSectorSize;

    // Round slot_bytes up to sector boundary
    if (slot_bytes_ % sector_size_)
        slot_bytes_ = ((slot_bytes_ / sector_size_) + 1) * sector_size_;

    // Convert wide path to UTF-8
    std::string utf8_path = caspar::u8(path);

    fd_ = ::open(utf8_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_DIRECT, 0644);
    if (fd_ < 0) {
        fprintf(stderr, "[AsyncFileWriter] open() failed: %s (errno %d)\n",
                utf8_path.c_str(), errno);
        return false;
    }

    // Initialize io_uring with kRingSize entries
    int ret = io_uring_queue_init(kRingSize * 2, &ring_, 0);
    if (ret < 0) {
        fprintf(stderr, "[AsyncFileWriter] io_uring_queue_init failed: %s\n", strerror(-ret));
        ::close(fd_);
        fd_ = -1;
        return false;
    }
    ring_initialized_ = true;

    // Allocate sector-aligned write buffers
    for (int i = 0; i < kRingSize; i++) {
        void *buf = nullptr;
        int err = posix_memalign(&buf, sector_size_, slot_bytes_);
        if (err != 0 || !buf) {
            fprintf(stderr, "[AsyncFileWriter] posix_memalign(%zu, %zu) failed for slot %d\n",
                    sector_size_, slot_bytes_, i);
            for (int j = 0; j < i; j++) {
                free(slots_[j].buf);
                slots_[j].buf = nullptr;
            }
            io_uring_queue_exit(&ring_);
            ring_initialized_ = false;
            ::close(fd_);
            fd_ = -1;
            return false;
        }
        slots_[i].buf = static_cast<uint8_t*>(buf);
        slots_[i].in_flight = false;
    }

    path_ = path;
    write_offset_    = 0;
    next_slot_       = 0;
    in_flight_count_ = 0;
    return true;
}

// ---------------------------------------------------------------------------
// wait_for_slot() — wait for a specific in-flight slot to complete
// ---------------------------------------------------------------------------
bool AsyncFileWriter::wait_for_slot(int idx)
{
    if (!slots_[idx].in_flight) return true;

    // Drain completions until target slot is done
    while (slots_[idx].in_flight) {
        struct io_uring_cqe *cqe = nullptr;
        int ret = io_uring_wait_cqe(&ring_, &cqe);
        if (ret < 0) {
            fprintf(stderr, "[AsyncFileWriter] io_uring_wait_cqe failed: %s\n", strerror(-ret));
            return false;
        }

        // user_data stores the slot index
        int completed_slot = static_cast<int>(reinterpret_cast<uintptr_t>(io_uring_cqe_get_data(cqe)));
        if (cqe->res < 0) {
            fprintf(stderr, "[AsyncFileWriter] io_uring write error on slot %d: %s\n",
                    completed_slot, strerror(-cqe->res));
            io_uring_cqe_seen(&ring_, cqe);
            return false;
        }

        if (completed_slot >= 0 && completed_slot < kRingSize) {
            slots_[completed_slot].in_flight = false;
            --in_flight_count_;
        }

        io_uring_cqe_seen(&ring_, cqe);
    }
    return true;
}

// ---------------------------------------------------------------------------
// drain_all() — wait for all in-flight writes
// ---------------------------------------------------------------------------
bool AsyncFileWriter::drain_all()
{
    bool ok = true;
    while (in_flight_count_ > 0) {
        struct io_uring_cqe *cqe = nullptr;
        int ret = io_uring_wait_cqe(&ring_, &cqe);
        if (ret < 0) {
            fprintf(stderr, "[AsyncFileWriter] drain_all io_uring error: %s\n", strerror(-ret));
            ok = false;
            break;
        }

        int completed_slot = static_cast<int>(reinterpret_cast<uintptr_t>(io_uring_cqe_get_data(cqe)));
        if (cqe->res < 0) {
            fprintf(stderr, "[AsyncFileWriter] drain_all write error on slot %d: %s\n",
                    completed_slot, strerror(-cqe->res));
            ok = false;
        }

        if (completed_slot >= 0 && completed_slot < kRingSize && slots_[completed_slot].in_flight) {
            slots_[completed_slot].in_flight = false;
            --in_flight_count_;
        }

        io_uring_cqe_seen(&ring_, cqe);
    }
    return ok;
}

// ---------------------------------------------------------------------------
// acquire_free_slot() — returns index of a free slot, blocking if all busy
// ---------------------------------------------------------------------------
int AsyncFileWriter::acquire_free_slot()
{
    // Fast path: round-robin check for free slot
    for (int n = 0; n < kRingSize; n++) {
        int i = (next_slot_ + n) % kRingSize;
        if (!slots_[i].in_flight) {
            next_slot_ = (i + 1) % kRingSize;
            return i;
        }
    }

    // All slots busy: wait for the oldest in-flight write
    int wait_idx = next_slot_ % kRingSize;
    if (!wait_for_slot(wait_idx)) return -1;
    next_slot_ = (wait_idx + 1) % kRingSize;
    return wait_idx;
}

// ---------------------------------------------------------------------------
// write()
// ---------------------------------------------------------------------------
bool AsyncFileWriter::write(const void *data, size_t size)
{
    if (!size) return true;
    if (size > slot_bytes_) {
        fprintf(stderr, "[AsyncFileWriter] write(%zu) exceeds slot size (%zu)\n",
                size, slot_bytes_);
        return false;
    }

    int idx = acquire_free_slot();
    if (idx < 0) return false;

    Slot &s = slots_[idx];

    // Copy data into sector-aligned buffer, zero-pad the last sector
    size_t aligned = (size + sector_size_ - 1) & ~(sector_size_ - 1);
    memcpy(s.buf, data, size);
    if (aligned > size)
        memset(s.buf + size, 0, aligned - size);

    // Submit async write via io_uring
    struct io_uring_sqe *sqe = io_uring_get_sqe(&ring_);
    if (!sqe) {
        // SQ full — drain one completion and retry
        drain_all();
        sqe = io_uring_get_sqe(&ring_);
        if (!sqe) {
            fprintf(stderr, "[AsyncFileWriter] io_uring_get_sqe failed after drain\n");
            return false;
        }
    }

    io_uring_prep_write(sqe, fd_, s.buf, static_cast<unsigned>(aligned),
                        static_cast<__u64>(write_offset_));
    io_uring_sqe_set_data(sqe, reinterpret_cast<void*>(static_cast<uintptr_t>(idx)));

    s.in_flight = true;
    ++in_flight_count_;
    write_offset_ += aligned;

    int ret = io_uring_submit(&ring_);
    if (ret < 0) {
        s.in_flight = false;
        --in_flight_count_;
        fprintf(stderr, "[AsyncFileWriter] io_uring_submit failed: %s\n", strerror(-ret));
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// flush()
// ---------------------------------------------------------------------------
bool AsyncFileWriter::flush()
{
    return drain_all();
}

// ---------------------------------------------------------------------------
// close()
// ---------------------------------------------------------------------------
bool AsyncFileWriter::close(const void *tail, size_t tail_size)
{
    bool ok = drain_all();

    // Append tail (e.g. moov atom) via a regular buffered write so it doesn't
    // need to be sector-aligned.
    if (tail && tail_size > 0 && fd_ >= 0) {
        // Close the O_DIRECT fd first, then reopen without O_DIRECT for the tail
        ::close(fd_);
        fd_ = -1;

        std::string utf8_path = caspar::u8(path_.c_str());
        int bf = ::open(utf8_path.c_str(), O_WRONLY, 0644);
        if (bf >= 0) {
            ssize_t written = pwrite(bf, tail, tail_size, static_cast<off_t>(write_offset_));
            if (written < 0 || static_cast<size_t>(written) != tail_size)
                ok = false;
            ::close(bf);
        } else {
            fprintf(stderr, "[AsyncFileWriter] close: could not reopen for tail write (errno %d)\n", errno);
            ok = false;
        }
    } else if (fd_ >= 0) {
        // Truncate file to actual written size (O_DIRECT may have written
        // extra zeroes in the last sector-aligned block)
        if (ftruncate(fd_, static_cast<off_t>(write_offset_)) < 0) {
            // Non-fatal: file may have trailing zeroes
        }
        ::close(fd_);
        fd_ = -1;
    }

    if (ring_initialized_) {
        io_uring_queue_exit(&ring_);
        ring_initialized_ = false;
    }

    // Free slot buffers
    for (int i = 0; i < kRingSize; i++) {
        if (slots_[i].buf) {
            free(slots_[i].buf);
            slots_[i].buf = nullptr;
        }
    }

    path_.clear();
    return ok;
}

#endif // !WIN32
