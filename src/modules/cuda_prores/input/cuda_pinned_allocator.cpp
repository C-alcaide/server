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

    // Pool exhausted — allocate a temporary buffer.  Should not happen in
    // normal operation (signals encode is falling behind capture).
    fprintf(stderr, "[CudaPinnedAllocator] Pool exhausted — allocating temporary buffer\n");
    void *ptr = nullptr;
    cudaError_t e = cudaMallocHost(&ptr, bufferSize);
    if (e != cudaSuccess) return E_OUTOFMEMORY;
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
