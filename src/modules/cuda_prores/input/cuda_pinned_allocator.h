// cuda_pinned_allocator.h
// IDeckLinkMemoryAllocator implementation that provides CUDA pinned (page-locked)
// host buffers for DeckLink video input frames.
//
// When registered via IDeckLinkInput::SetVideoInputFrameMemoryAllocator(), the
// DeckLink driver DMAs incoming SDI frames directly into these pinned buffers.
// Because they are page-locked, cudaMemcpyAsync(HostToDevice) can proceed
// without the driver needing to pin/unpin pages — eliminating one full-frame
// CPU memcpy from the capture hot path.
//
// Architecture
// ─────────────────────────────────────────────────────────────────────────────
//  Auto-growing pool of pinned buffers sized for the largest expected frame
//  (4K V210: 3840 × 2160 × 8/3 ≈ 20 971 520 bytes, rounded to 4MB boundary).
//  AllocateBuffer() pops a free buffer; if the pool is empty it grows by
//  allocating one more slot (cudaMallocHost once, reused forever).
//  ReleaseBuffer() returns the buffer to the pool.
//  All allocated buffers are freed in the destructor.
#pragma once

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <comutil.h>    // pulls in COM/RPC headers that define 'interface' keyword

#include <cuda_runtime.h>
#include <mutex>
#include <vector>
#include <cstdint>
#include <cassert>

// DeckLink SDK COM interface (path resolved via ../decklink include added by CMakeLists.txt)
#if defined(_MSC_VER)
#include "interop/DeckLinkAPI.h"
#else
#include "linux_interop/DeckLinkAPI.h"
#endif

class CudaPinnedAllocator final : public IDeckLinkMemoryAllocator {
public:
    // max_frame_bytes: maximum single frame size in bytes.
    //   For 4K V210: ceil(3840*2160 * 10/8 * (4/3)) = 20971520 (rounded to 32-byte align)
    // pool_size: number of pre-allocated buffers in the ring (must be >= 2).
    explicit CudaPinnedAllocator(size_t max_frame_bytes = 21 * 1024 * 1024,
                                 int    pool_size        = 4);
    ~CudaPinnedAllocator();

    // IDeckLinkMemoryAllocator
    HRESULT STDMETHODCALLTYPE AllocateBuffer(uint32_t bufferSize, void **allocatedBuffer) override;
    HRESULT STDMETHODCALLTYPE ReleaseBuffer(void *buffer) override;
    HRESULT STDMETHODCALLTYPE Commit() override  { return S_OK; }
    HRESULT STDMETHODCALLTYPE Decommit() override { return S_OK; }

    // IUnknown (minimal COM — not reference counted; lifetime managed by owner)
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID, void **) override { return E_NOINTERFACE; }
    ULONG   STDMETHODCALLTYPE AddRef()  override { return 1; }
    ULONG   STDMETHODCALLTYPE Release() override { return 1; }

private:
    size_t            max_frame_bytes_;
    std::mutex        mutex_;
    std::vector<void*> pool_;   // free pinned buffers
    std::vector<void*> all_;    // all allocated (for cleanup)
};
