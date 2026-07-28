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

#include "../StdAfx.h"

#include "gpu_output_buffer_pool.h"

#include <common/log.h>

#ifdef _WIN32
#include <windows.h>
#endif

#ifdef DECKLINK_CUDA_VK_ENABLED
#include <cuda_runtime.h>
#endif

namespace caspar { namespace decklink {

namespace {

#ifdef _WIN32
// VirtualLock can only pin pages that fit in the process working set, whose
// default maximum is tiny (~1.5 MB). Grow the min/max working set to cover the
// buffers we are about to lock (accumulated across pools), else VirtualLock fails.
void raise_working_set(std::size_t extra_bytes)
{
    static std::mutex m;
    std::lock_guard<std::mutex> lk(m);
    HANDLE h     = ::GetCurrentProcess();
    SIZE_T mn = 0, mx = 0;
    if (!::GetProcessWorkingSetSize(h, &mn, &mx))
        return;
    const SIZE_T margin  = static_cast<SIZE_T>(32) << 20; // 32 MB headroom
    SIZE_T       want_mn = mn + extra_bytes + margin;
    SIZE_T       want_mx = mx > want_mn ? mx : want_mn + (static_cast<SIZE_T>(64) << 20);
    ::SetProcessWorkingSetSize(h, want_mn, want_mx);
}
#endif

// Allocate a page-locked buffer of `bytes`. Sets `locked` to whether page-locking
// actually succeeded (allocation still returns a usable buffer if locking fails,
// e.g. the process working-set quota is exhausted — the frame just isn't pinned).
void* alloc_pinned(std::size_t bytes, gpu_output_buffer_pool::pin_kind kind, bool& locked)
{
    locked = false;
#ifdef DECKLINK_CUDA_VK_ENABLED
    if (kind == gpu_output_buffer_pool::pin_kind::cuda_pinned) {
        void* p = nullptr;
        if (cudaMallocHost(&p, bytes) == cudaSuccess && p) {
            locked = true;
            return p;
        }
        // Fall through to host_locked on CUDA pinned-alloc failure.
        cudaGetLastError();
    }
#endif
#ifdef _WIN32
    void* p = ::VirtualAlloc(nullptr, bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (!p)
        return nullptr;
    raise_working_set(bytes);
    locked = ::VirtualLock(p, bytes) != 0;
    return p;
#else
    (void)kind;
    return std::malloc(bytes);
#endif
}

void free_pinned(void* p, std::size_t bytes, gpu_output_buffer_pool::pin_kind kind)
{
    if (!p)
        return;
#ifdef DECKLINK_CUDA_VK_ENABLED
    if (kind == gpu_output_buffer_pool::pin_kind::cuda_pinned) {
        cudaFreeHost(p);
        return;
    }
#endif
#ifdef _WIN32
    ::VirtualUnlock(p, bytes);
    ::VirtualFree(p, 0, MEM_RELEASE);
#else
    (void)bytes;
    (void)kind;
    std::free(p);
#endif
}

} // namespace

// Owned jointly by the pool and every outstanding buffer so allocations stay
// valid until the last frame leaves the DeckLink queue.
struct gpu_output_buffer_pool::state
{
    std::mutex         mtx;
    std::size_t        buffer_bytes = 0;
    pin_kind           kind         = pin_kind::host_locked;
    bool               all_pinned   = true;
    std::vector<void*> free_list; // available pooled buffers
    std::vector<void*> pooled;    // every pooled allocation (== buffer_bytes)

    ~state()
    {
        for (void* p : pooled)
            free_pinned(p, buffer_bytes, kind);
    }
};

gpu_output_buffer_pool::gpu_output_buffer_pool(std::size_t buffer_bytes, int initial_count, pin_kind kind)
    : state_(std::make_shared<state>())
    , buffer_bytes_(buffer_bytes)
    , kind_(kind)
{
    state_->buffer_bytes = buffer_bytes;
    state_->kind         = kind;
    if (initial_count < 2)
        initial_count = 2;
    for (int i = 0; i < initial_count; ++i) {
        bool  locked = false;
        void* p      = alloc_pinned(buffer_bytes, kind, locked);
        if (!p)
            break;
        state_->all_pinned = state_->all_pinned && locked;
        state_->pooled.push_back(p);
        state_->free_list.push_back(p);
    }
    if (!state_->all_pinned)
        CASPAR_LOG(warning) << L"[decklink] gpu_output_buffer_pool: some output buffers could not be page-locked "
                               L"(working-set quota?); output still functional, just not pinned.";
    else
        CASPAR_LOG(info) << L"[decklink] gpu_output_buffer_pool: " << static_cast<int>(state_->pooled.size())
                         << L" x " << static_cast<double>(buffer_bytes) / (1024.0 * 1024.0)
                         << L" MB page-locked output buffers ("
                         << (kind == pin_kind::cuda_pinned ? L"cuda-pinned" : L"host-locked") << L").";
}

gpu_output_buffer_pool::~gpu_output_buffer_pool() = default;

bool gpu_output_buffer_pool::pinned() const
{
    std::lock_guard<std::mutex> lk(state_->mtx);
    return state_->all_pinned;
}

std::shared_ptr<void> gpu_output_buffer_pool::acquire(std::size_t bytes)
{
    auto st = state_;

    // Oversized request: one-off page-locked buffer, not returned to the pool.
    if (bytes > st->buffer_bytes) {
        bool  locked = false;
        void* p      = alloc_pinned(bytes, st->kind, locked);
        if (!p)
            return {};
        auto kind = st->kind;
        return std::shared_ptr<void>(p, [kind, bytes](void* q) { free_pinned(q, bytes, kind); });
    }

    void* p = nullptr;
    {
        std::lock_guard<std::mutex> lk(st->mtx);
        if (!st->free_list.empty()) {
            p = st->free_list.back();
            st->free_list.pop_back();
        }
    }

    if (!p) {
        // Pool exhausted: grow by one page-locked slot (kept for reuse thereafter).
        bool  locked = false;
        p            = alloc_pinned(st->buffer_bytes, st->kind, locked);
        if (!p)
            return {};
        std::lock_guard<std::mutex> lk(st->mtx);
        st->all_pinned = st->all_pinned && locked;
        st->pooled.push_back(p);
    }

    // Deleter holds `st` alive and returns the buffer to the free list.
    return std::shared_ptr<void>(p, [st](void* q) {
        std::lock_guard<std::mutex> lk(st->mtx);
        st->free_list.push_back(q);
    });
}

}} // namespace caspar::decklink
