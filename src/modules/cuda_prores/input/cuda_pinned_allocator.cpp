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
 * The Blackmagic DeckLink SDK is proprietary software; headers are included
 * under the terms of the Blackmagic Design SDK License Agreement.
 */

// cuda_pinned_allocator.cpp
#include "cuda_pinned_allocator.h"
#include <cassert>
#include <cstdio>

CudaPinnedAllocator::CudaPinnedAllocator(size_t max_frame_bytes, int pool_size)
    : max_frame_bytes_(max_frame_bytes)
{
    assert(pool_size >= 2);
    for (int i = 0; i < pool_size; i++) {
        void *ptr = nullptr;
        cudaError_t e = cudaMallocHost(&ptr, max_frame_bytes);
        if (e != cudaSuccess || !ptr) {
            fprintf(stderr, "[CudaPinnedAllocator] cudaMallocHost(%zu) failed: %s\n",
                    max_frame_bytes, cudaGetErrorString(e));
            break;
        }
        pool_.push_back(ptr);
        all_.push_back(ptr);
    }
}

CudaPinnedAllocator::~CudaPinnedAllocator()
{
    for (void *p : all_)
        cudaFreeHost(p);
}

HRESULT CudaPinnedAllocator::AllocateBuffer(uint32_t bufferSize, void **allocatedBuffer)
{
    if (!allocatedBuffer) return E_POINTER;
    if ((size_t)bufferSize > max_frame_bytes_) {
        // Frame larger than expected — fall back to on-demand alloc
        void *ptr = nullptr;
        cudaError_t e = cudaMallocHost(&ptr, bufferSize);
        if (e != cudaSuccess) return E_OUTOFMEMORY;
        // Note: this buffer won't be returned to the pool on ReleaseBuffer
        // (identified by its absence from all_).  It will be freed immediately
        // in ReleaseBuffer.
        *allocatedBuffer = ptr;
        return S_OK;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    if (!pool_.empty()) {
        *allocatedBuffer = pool_.back();
        pool_.pop_back();
        return S_OK;
    }

    // Pool empty — DeckLink requested more buffers than the initial pool size
    // (common during signal lock when it pre-allocates its internal ring).
    // Grow the pool by one slot; track it in all_ so it is freed at destruction.
    void *ptr = nullptr;
    cudaError_t e = cudaMallocHost(&ptr, max_frame_bytes_);
    if (e != cudaSuccess) return E_OUTOFMEMORY;
    all_.push_back(ptr);
    *allocatedBuffer = ptr;
    return S_OK;
}

HRESULT CudaPinnedAllocator::ReleaseBuffer(void *buffer)
{
    if (!buffer) return S_OK;

    // Check if it belongs to the pool
    std::lock_guard<std::mutex> lock(mutex_);
    for (void *p : all_) {
        if (p == buffer) {
            pool_.push_back(buffer);
            return S_OK;
        }
    }

    // Not from pool (was a temporary oversized alloc) — free it
    cudaFreeHost(buffer);
    return S_OK;
}
