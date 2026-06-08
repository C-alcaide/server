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
 * CUDA-VK direct GPU decklink strategy.
 *
 * Instead of the VK mixer doing GPU→CPU readback (132 MB/frame at 7680×2160×16-bit)
 * and the CPU doing v210 packing with AVX2, this strategy:
 *   1. Imports the VK mixer's render attachment into CUDA (external memory interop)
 *   2. Reads only the configured subregion via a CUDA surface object
 *   3. Packs v210 on GPU (1 kernel launch)
 *   4. Copies the packed v210 (~22 MB for 3840×2160) to pinned host memory
 *
 * Benefits:
 *   - Eliminates VK readback from the VK queue → less GPU contention with CUDA decode
 *   - v210 packing runs on GPU (faster than 6-thread AVX2)
 *   - 6× less PCIe bandwidth (v210 subregion vs full BGRA16 readback)
 *   - The CUDA decode stream is never blocked by decklink work
 *   - Returns needs_cpu_frame_data()=false so the VK mixer skips copy_async() entirely
 */

#include "../StdAfx.h"

#include "cuda_vk_strategy.h"

#include <common/log.h>
#include <common/diagnostics/graph.h>
#include <common/timer.h>

#ifdef ENABLE_VULKAN
#include <accelerator/vulkan/util/texture_wrapper.h>
#endif

#include <cuda_runtime.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

#include <atomic>
#include <stdexcept>

// C-linkage CUDA kernel launchers (defined in cuda_vk_kernels.cu)
extern "C" {
cudaError_t cuda_vk_launch_surface_to_v210(
    cudaSurfaceObject_t surf, uint32_t* d_v210,
    int src_x, int src_y, int dst_w, int dst_h, int src_w, int src_h,
    int is_16bit, int use_bt2020, cudaStream_t stream);

cudaError_t cuda_vk_launch_surface_to_bgra8(
    cudaSurfaceObject_t surf, uint8_t* d_bgra,
    int src_x, int src_y, int dst_w, int dst_h, int src_w, int src_h,
    cudaStream_t stream);
}

namespace caspar { namespace decklink {

namespace {

void cuda_check(cudaError_t err, const char* what)
{
    if (err != cudaSuccess) {
        std::string msg = std::string(what) + ": " + cudaGetErrorString(err);
        CASPAR_LOG(error) << L"[cuda_vk_strategy] " << msg.c_str();
        throw std::runtime_error(msg);
    }
}

} // anonymous namespace

// ===========================================================================
// cuda_vk_strategy — implementation
// ===========================================================================

struct cuda_vk_strategy::impl
{
    const bool use_bt2020_;
    const bool is_hdr_;
    const bool needs_v210_;

    // CUDA resources
    int              cuda_device_ = 0;
    cudaStream_t     stream_      = nullptr;
    uint32_t*        d_v210_[3]   = {};  // triple-buffered device V210 output
    uint8_t*         d_bgra_[3]   = {};  // triple-buffered device BGRA8 output (SDR only)

    // Async triple-buffer: three pinned host buffers (rotating).
    // Each frame we: (1) wait for the buffer written 2 frames ago (the "read"
    // buffer), (2) launch kernel + D2H into buf[write_idx_], (3) return the
    // read buffer.  With 3 buffers and a rotating index, the buffer we sync
    // on has had 2 full frame intervals to complete its D2H — matching the
    // OGL PBO triple-buffer pipeline depth that achieves perfect 60fps.
    static constexpr int NUM_ASYNC_BUFS = 3;
    void*            h_pinned_[NUM_ASYNC_BUFS] = {};  // pinned host output buffers
    size_t           h_pinned_sz_  = 0;
    int              write_idx_    = 0;   // buffer being written to this frame
    int              warmup_count_ = 0;   // counts up to NUM_ASYNC_BUFS-1 during startup

    // Per-buffer CUDA events: recorded after D2H for each buffer.
    // Before reusing buffer[i], we wait on d2h_event_[i] (which was recorded
    // NUM_ASYNC_BUFS-1 frames ago, giving 2 frame intervals for GPU work to
    // complete — significantly more slack than the previous 1-frame double-buffer).
    cudaEvent_t      d2h_event_[NUM_ASYNC_BUFS] = {};

    // Multi-slot import cache: the VK attachment pool rotates through N textures
    // (typically 3-4). We cache the CUDA import for each one so we only pay the
    // ~25ms cudaImportExternalMemory cost once per slot, not every frame.
    struct imported_slot
    {
        void*                  handle  = nullptr;
        cudaExternalMemory_t   ext_mem = nullptr;
        cudaMipmappedArray_t   mipmap  = nullptr;
        cudaArray_t            array   = nullptr;
        cudaSurfaceObject_t    surf    = 0;
        int                    w       = 0;
        int                    h       = 0;

        void cleanup()
        {
            if (surf)    { cudaDestroySurfaceObject(surf); surf = 0; }
            array = nullptr;  // Owned by mipmap
            if (mipmap)  { cudaFreeMipmappedArray(mipmap); mipmap = nullptr; }
            if (ext_mem) { cudaDestroyExternalMemory(ext_mem); ext_mem = nullptr; }
            handle = nullptr;
            w = h = 0;
        }
    };
    static constexpr int MAX_CACHED_SLOTS = 8;
    imported_slot cached_slots_[MAX_CACHED_SLOTS];
    int           num_cached_ = 0;

    // Fallback strategy for non-VK frames
    spl::shared_ptr<format_strategy> fallback_;

    // Imported VK timeline semaphores for GPU-side render-complete wait.
    // Cached per frame_data slot handle (typically 3-4 rotating handles).
    struct cached_semaphore {
        void*                   handle = nullptr;
        cudaExternalSemaphore_t sem    = nullptr;
    };
    static constexpr int MAX_CACHED_SEMS = 8;
    cached_semaphore cached_sems_[MAX_CACHED_SEMS];
    int              num_cached_sems_ = 0;
    bool             gpu_wait_available_ = true;  // assume available until proven otherwise
    int              gpu_wait_fail_count_ = 0;    // consecutive failures; retry after threshold
    static constexpr int GPU_WAIT_RETRY_INTERVAL = 500; // frames between retry attempts

    // Shared sentinel to prevent dangling pointers.  The returned shared_ptr<void>
    // from convert_v210/convert_bgra captures this; if the consumer outlives the
    // strategy, the custom deleter safely no-ops instead of accessing freed memory.
    std::shared_ptr<std::atomic<bool>> alive_ = std::make_shared<std::atomic<bool>>(true);

    // Wrap a pinned host pointer in a shared_ptr that holds a weak reference to
    // the alive_ sentinel.  The pinned memory is NOT freed by the deleter — it's
    // owned by h_pinned_[] and freed in ~impl().  The sentinel just prevents
    // access after the strategy is destroyed.
    std::shared_ptr<void> make_pinned_ref(void* pinned)
    {
        auto guard = alive_;
        return std::shared_ptr<void>(pinned, [guard](void*) {
            // No-op deleter: pinned memory is owned by impl, not by this shared_ptr.
            // The captured 'guard' keeps the sentinel alive so that even if the
            // consumer's shared_ptr outlives the strategy, no crash occurs.
        });
    }

    // Timing diagnostics (periodic log)
    int      frame_count_ = 0;
    double   accum_fence_ms_   = 0.0;
    double   accum_import_ms_  = 0.0;
    double   accum_sync_ms_    = 0.0;
    double   accum_launch_ms_  = 0.0;
    double   accum_total_ms_   = 0.0;

    impl(bool is_hdr, bool use_bt2020, spl::shared_ptr<format_strategy> fallback, bool needs_v210)
        : use_bt2020_(use_bt2020)
        , is_hdr_(is_hdr)
        , needs_v210_(needs_v210)
        , fallback_(std::move(fallback))
    {
        // Use the same CUDA device as the primary GPU (device 0).
        // This matches the VK mixer which also runs on the primary discrete GPU.
        cuda_device_ = 0;
        cuda_check(cudaSetDevice(cuda_device_), "cudaSetDevice");
        cuda_check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "cudaStreamCreate");
        for (int i = 0; i < NUM_ASYNC_BUFS; ++i)
            cuda_check(cudaEventCreateWithFlags(&d2h_event_[i], cudaEventDisableTiming), "cudaEventCreate");
    }

    ~impl()
    {
        // Mark sentinel so any outstanding shared_ptr<void> from convert_v210/bgra
        // won't attempt to use this impl's pinned buffers after destruction.
        alive_->store(false, std::memory_order_release);

        // Ensure all async GPU work has completed before destroying resources.
        // Without this, in-flight D2H copies or semaphore waits could access
        // freed buffers or destroyed events.
        if (stream_) {
            cudaSetDevice(cuda_device_);
            cudaStreamSynchronize(stream_);
        }

        for (int i = 0; i < num_cached_sems_; ++i) {
            if (cached_sems_[i].sem) cudaDestroyExternalSemaphore(cached_sems_[i].sem);
        }
        num_cached_sems_ = 0;
        for (int i = 0; i < num_cached_; ++i)
            cached_slots_[i].cleanup();
        num_cached_ = 0;
        for (int i = 0; i < NUM_ASYNC_BUFS; ++i) {
            if (d2h_event_[i]) { cudaEventDestroy(d2h_event_[i]); d2h_event_[i] = nullptr; }
        }
        for (int i = 0; i < NUM_ASYNC_BUFS; ++i) {
            if (d_v210_[i]) { cudaFree(d_v210_[i]); d_v210_[i] = nullptr; }
            if (d_bgra_[i]) { cudaFree(d_bgra_[i]); d_bgra_[i] = nullptr; }
        }
        for (int i = 0; i < NUM_ASYNC_BUFS; ++i) {
            if (h_pinned_[i]) { cudaFreeHost(h_pinned_[i]); h_pinned_[i] = nullptr; }
        }
        if (stream_) { cudaStreamDestroy(stream_); stream_ = nullptr; }
    }

    // Returns the CUDA surface object for the given VK texture handle.
    // Caches imports so repeated calls with the same handle are free.
    cudaSurfaceObject_t ensure_import(void* win32_handle, unsigned long long alloc_size,
                                      int width, int height, bool is_16bit)
    {
        // Check if already cached
        for (int i = 0; i < num_cached_; ++i) {
            auto& s = cached_slots_[i];
            if (s.handle == win32_handle && s.w == width && s.h == height)
                return s.surf;
        }

        // Need a new slot — evict oldest if full
        if (num_cached_ >= MAX_CACHED_SLOTS) {
            // The evicted texture handle may also have a cached semaphore.
            // Invalidate it so stale CUDA semaphore objects aren't reused
            // if the OS recycles the Win32 HANDLE value (#10).
            invalidate_sem_for_handle(cached_slots_[0].handle);
            cached_slots_[0].cleanup();
            for (int i = 1; i < num_cached_; ++i)
                cached_slots_[i - 1] = cached_slots_[i];
            num_cached_--;
        }

        auto& slot = cached_slots_[num_cached_];
        slot = {};

        // Import the VK texture's device memory into CUDA
        cudaExternalMemoryHandleDesc extMemDesc{};
#ifdef _WIN32
        extMemDesc.type                = cudaExternalMemoryHandleTypeOpaqueWin32;
        extMemDesc.handle.win32.handle = win32_handle;
#else
        extMemDesc.type       = cudaExternalMemoryHandleTypeOpaqueFd;
        extMemDesc.handle.fd  = dup(static_cast<int>(reinterpret_cast<intptr_t>(win32_handle)));
#endif
        extMemDesc.size                = alloc_size;
        extMemDesc.flags               = 0;
        auto mem_err = cudaImportExternalMemory(&slot.ext_mem, &extMemDesc);
#ifndef _WIN32
        // On Linux, cudaImportExternalMemory does NOT consume the fd on failure
        if (mem_err != cudaSuccess && extMemDesc.handle.fd >= 0)
            ::close(extMemDesc.handle.fd);
#endif
        cuda_check(mem_err, "cudaImportExternalMemory");

        // Map as mipmapped array
        cudaExternalMemoryMipmappedArrayDesc mipmapDesc{};
        mipmapDesc.offset = 0;
        if (is_16bit) {
            mipmapDesc.formatDesc = cudaCreateChannelDesc(16, 16, 16, 16, cudaChannelFormatKindUnsigned);
        } else {
            mipmapDesc.formatDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
        }
        mipmapDesc.extent.width  = static_cast<unsigned>(width);
        mipmapDesc.extent.height = static_cast<unsigned>(height);
        mipmapDesc.extent.depth  = 0;
        mipmapDesc.numLevels     = 1;
        mipmapDesc.flags         = cudaArrayDefault;
        cuda_check(cudaExternalMemoryGetMappedMipmappedArray(&slot.mipmap, slot.ext_mem, &mipmapDesc),
                   "cudaExternalMemoryGetMappedMipmappedArray");

        // Get level 0
        cuda_check(cudaGetMipmappedArrayLevel(&slot.array, slot.mipmap, 0), "cudaGetMipmappedArrayLevel");

        // Create surface object for reading
        cudaResourceDesc resDesc{};
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = slot.array;
        cuda_check(cudaCreateSurfaceObject(&slot.surf, &resDesc), "cudaCreateSurfaceObject");

        slot.handle = win32_handle;
        slot.w      = width;
        slot.h      = height;
        num_cached_++;

        CASPAR_LOG(debug) << L"[cuda_vk_strategy] Imported VK texture slot "
                          << num_cached_ << L"/" << MAX_CACHED_SLOTS
                          << L" (" << width << L"x" << height << L")";

        return slot.surf;
    }

    void ensure_output_buffers(int dst_w, int dst_h, bool need_v210)
    {
        if (need_v210) {
            size_t v210_row = (size_t)((dst_w + 47) / 48) * 128;
            size_t v210_sz  = v210_row * dst_h;
            if (!d_v210_[0] || h_pinned_sz_ < v210_sz) {
                for (int i = 0; i < NUM_ASYNC_BUFS; ++i) {
                    if (d_v210_[i]) cudaFree(d_v210_[i]);
                    cuda_check(cudaMalloc(&d_v210_[i], v210_sz), "cudaMalloc v210");
                }
                for (int i = 0; i < NUM_ASYNC_BUFS; ++i) {
                    if (h_pinned_[i]) cudaFreeHost(h_pinned_[i]);
                    cuda_check(cudaMallocHost(&h_pinned_[i], v210_sz), "cudaMallocHost v210");
                }
                h_pinned_sz_ = v210_sz;
                warmup_count_ = 0;  // buffers reallocated, must re-fill pipeline
            }
        } else {
            size_t bgra_sz = (size_t)dst_w * dst_h * 4;
            if (!d_bgra_[0] || h_pinned_sz_ < bgra_sz) {
                for (int i = 0; i < NUM_ASYNC_BUFS; ++i) {
                    if (d_bgra_[i]) cudaFree(d_bgra_[i]);
                    cuda_check(cudaMalloc(&d_bgra_[i], bgra_sz), "cudaMalloc bgra");
                }
                for (int i = 0; i < NUM_ASYNC_BUFS; ++i) {
                    if (h_pinned_[i]) cudaFreeHost(h_pinned_[i]);
                    cuda_check(cudaMallocHost(&h_pinned_[i], bgra_sz), "cudaMallocHost bgra");
                }
                h_pinned_sz_ = bgra_sz;
                warmup_count_ = 0;  // buffers reallocated, must re-fill pipeline
            }
        }
    }

    // Invalidate any cached semaphore associated with a given Win32 HANDLE.
    // Called when a texture slot is evicted so stale CUDA semaphore objects
    // aren't reused if the OS recycles the handle value.
    void invalidate_sem_for_handle(void* handle)
    {
        if (!handle) return;
        for (int i = 0; i < num_cached_sems_; ++i) {
            if (cached_sems_[i].handle == handle) {
                // Sync stream to ensure no in-flight GPU wait references this semaphore.
                if (stream_) cudaStreamSynchronize(stream_);
                cudaDestroyExternalSemaphore(cached_sems_[i].sem);
                for (int j = i + 1; j < num_cached_sems_; ++j)
                    cached_sems_[j - 1] = cached_sems_[j];
                num_cached_sems_--;
                CASPAR_LOG(debug) << L"[cuda_vk_strategy] Invalidated cached semaphore for evicted handle";
                return;
            }
        }
    }

    // Import VK timeline semaphore into CUDA (once) and enqueue a GPU-side wait.
    // Returns true if the GPU wait was enqueued, false if fallback to CPU wait is needed.
    bool try_gpu_wait(void* sem_handle, uint64_t sem_value)
    {
        if (!sem_handle || sem_value == 0)
            return false;

        // If GPU wait was disabled due to a previous failure, periodically retry
        // in case the failure was transient (e.g. driver hiccup).
        if (!gpu_wait_available_) {
            gpu_wait_fail_count_++;
            if (gpu_wait_fail_count_ < GPU_WAIT_RETRY_INTERVAL)
                return false;
            CASPAR_LOG(info) << L"[cuda_vk_strategy] Retrying GPU-side semaphore wait after "
                             << gpu_wait_fail_count_ << L" frames";
            gpu_wait_available_ = true;  // try again
            gpu_wait_fail_count_ = 0;
        }

        // Find or import the semaphore for this handle
        cudaExternalSemaphore_t cuda_sem = nullptr;
        for (int i = 0; i < num_cached_sems_; ++i) {
            if (cached_sems_[i].handle == sem_handle) {
                cuda_sem = cached_sems_[i].sem;
                break;
            }
        }

        if (!cuda_sem) {
            // Import new semaphore
            cudaExternalSemaphoreHandleDesc desc{};
#ifdef _WIN32
            desc.type                = cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32;
            desc.handle.win32.handle = sem_handle;
#else
            desc.type       = cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd;
            desc.handle.fd  = dup(static_cast<int>(reinterpret_cast<intptr_t>(sem_handle)));
#endif
            desc.flags               = 0;
            cudaExternalSemaphore_t new_sem = nullptr;
            auto err = cudaImportExternalSemaphore(&new_sem, &desc);
            if (err != cudaSuccess) {
#ifndef _WIN32
                // On Linux, cudaImportExternalSemaphore does NOT consume fd on failure
                if (desc.handle.fd >= 0) ::close(desc.handle.fd);
#endif
                CASPAR_LOG(warning) << L"[cuda_vk_strategy] Failed to import VK semaphore: "
                                    << cudaGetErrorString(err) << L" — falling back to CPU wait";
                gpu_wait_available_ = false;
                gpu_wait_fail_count_ = 0;
                return false;
            }

            // Store in cache — evict oldest if full
            if (num_cached_sems_ >= MAX_CACHED_SEMS) {
                // Sync stream before destroying — an in-flight GPU wait may
                // reference this semaphore from a previous frame.
                if (stream_) cudaStreamSynchronize(stream_);
                cudaDestroyExternalSemaphore(cached_sems_[0].sem);
                for (int i = 1; i < num_cached_sems_; ++i)
                    cached_sems_[i - 1] = cached_sems_[i];
                num_cached_sems_--;
            }
            cached_sems_[num_cached_sems_] = {sem_handle, new_sem};
            num_cached_sems_++;
            cuda_sem = new_sem;
            CASPAR_LOG(info) << L"[cuda_vk_strategy] Imported VK timeline semaphore for GPU-side wait"
                             << L" (slot " << num_cached_sems_ << L"/" << MAX_CACHED_SEMS << L")";
        }

        // Enqueue wait on the CUDA stream — the stream will block GPU-side
        // until the VK render signals this timeline value.
        cudaExternalSemaphoreWaitParams waitParams{};
        waitParams.params.fence.value = sem_value;
        auto err = cudaWaitExternalSemaphoresAsync(&cuda_sem, &waitParams, 1, stream_);
        if (err != cudaSuccess) {
            CASPAR_LOG(warning) << L"[cuda_vk_strategy] cudaWaitExternalSemaphoresAsync failed: "
                                << cudaGetErrorString(err);
            gpu_wait_available_ = false;
            gpu_wait_fail_count_ = 0;
            return false;
        }
        return true;
    }

    std::shared_ptr<void> convert_v210(
        const core::video_format_desc& channel_format_desc,
        const core::video_format_desc& decklink_format_desc,
        const port_configuration&      config,
        const core::const_frame&       frame)
    {
#ifdef ENABLE_VULKAN
        // Get the VK texture from the frame
        auto tex = frame.texture();
        if (!tex) return nullptr;

        auto* wrapper = dynamic_cast<accelerator::vulkan::texture_wrapper*>(tex.get());
        if (!wrapper) return nullptr;

        caspar::timer total_timer;
        caspar::timer step_timer;

        // Get semaphore info (cheap accessors, no blocking)
        void*    sem_handle = wrapper->render_semaphore_handle();
        uint64_t sem_value  = wrapper->render_semaphore_value();

        auto vk_tex = wrapper->vk_texture();
        if (!vk_tex) return nullptr;

        // Get Win32 handle for CUDA import
        void* handle = vk_tex->export_native_handle();
        if (!handle) return nullptr;

        bool is_16bit = vk_tex->depth() != common::bit_depth::bit8;
        int src_w = vk_tex->width();
        int src_h = vk_tex->height();

        // Calculate subregion
        int src_x = config.src_x;
        int src_y = config.src_y;
        int dst_w = decklink_format_desc.width;
        int dst_h = decklink_format_desc.height;

        // Import the VK texture into CUDA (cached — effectively free after first few frames)
        step_timer = caspar::timer();
        cudaSetDevice(cuda_device_);
        auto surf = ensure_import(handle, vk_tex->alloc_size(), src_w, src_h, is_16bit);

        // Ensure output buffers (both ping-pong)
        size_t v210_row = (size_t)((dst_w + 47) / 48) * 128;
        size_t v210_sz  = v210_row * dst_h;
        ensure_output_buffers(dst_w, dst_h, true);
        double import_ms = step_timer.elapsed() * 1000.0;

        int cur_write = write_idx_;
        // Read buffer is the one written 2 frames ago (oldest in the ring).
        // With triple-buffering, this buffer has had 2 full frame intervals
        // for its D2H to complete, compared to only 1 with double-buffering.
        int cur_read  = (cur_write + 1) % NUM_ASYNC_BUFS;

        // Wait for buffer[cur_read]'s D2H to complete — this is the data we'll return.
        // With 3 buffers, this event was recorded 2 frames ago so the wait is
        // typically a no-op (already complete), matching OGL PBO pipeline depth.
        step_timer = caspar::timer();
        if (warmup_count_ >= NUM_ASYNC_BUFS - 1)
            cuda_check(cudaEventSynchronize(d2h_event_[cur_read]), "cudaEventSynchronize");
        double sync_ms = step_timer.elapsed() * 1000.0;

        // Enqueue GPU-side wait for VK render completion on the CUDA stream.
        // This replaces the CPU fence wait — the GPU blocks on the semaphore
        // instead of the CPU, freeing the DeckLink thread to return immediately.
        double fence_ms = 0.0;
        step_timer = caspar::timer();
        if (sem_handle && gpu_wait_available_) {
            if (!try_gpu_wait(sem_handle, sem_value)) {
                wrapper->ensure_render_complete();  // fallback
            }
        } else {
            wrapper->ensure_render_complete();  // no semaphore available
        }
        fence_ms = step_timer.elapsed() * 1000.0;

        // Launch kernel: read VK surface → pack v210 into device buffer[cur_write]
        step_timer = caspar::timer();
        auto err = cuda_vk_launch_surface_to_v210(
            surf,
            d_v210_[cur_write],
            src_x, src_y,
            dst_w, dst_h,
            src_w, src_h,
            is_16bit ? 1 : 0,
            use_bt2020_ ? 1 : 0,
            stream_);

        if (err != cudaSuccess) {
            CASPAR_LOG(warning) << L"[cuda_vk_strategy] kernel launch failed: " << cudaGetErrorString(err);
            return nullptr;
        }

        // Enqueue D2H copy into the write buffer — does NOT block
        cuda_check(cudaMemcpyAsync(h_pinned_[cur_write], d_v210_[cur_write], v210_sz,
                                   cudaMemcpyDeviceToHost, stream_), "cudaMemcpyAsync D2H");
        // Record event so we know when this buffer's D2H is done
        cuda_check(cudaEventRecord(d2h_event_[cur_write], stream_), "cudaEventRecord");
        double launch_ms = step_timer.elapsed() * 1000.0;

        double total_ms = total_timer.elapsed() * 1000.0;

        // Advance write index through the ring (0 → 1 → 2 → 0 ...)
        write_idx_ = (cur_write + 1) % NUM_ASYNC_BUFS;

        // Periodic timing report (every 50 frames)
        accum_fence_ms_   += fence_ms;
        accum_import_ms_  += import_ms;
        accum_sync_ms_    += sync_ms;
        accum_launch_ms_  += launch_ms;
        accum_total_ms_   += total_ms;
        frame_count_++;
        if (frame_count_ % 50 == 0) {
            double n = 50.0;
            CASPAR_LOG(debug) << L"[cuda_vk_strategy] avg over 50 frames: "
                              << L"fence=" << (accum_fence_ms_ / n) << L"ms "
                              << L"import=" << (accum_import_ms_ / n) << L"ms "
                              << L"sync=" << (accum_sync_ms_ / n) << L"ms "
                              << L"launch=" << (accum_launch_ms_ / n) << L"ms "
                              << L"total=" << (accum_total_ms_ / n) << L"ms";
            accum_fence_ms_ = accum_import_ms_ = accum_sync_ms_ = accum_launch_ms_ = accum_total_ms_ = 0.0;
        }

        if (warmup_count_ < NUM_ASYNC_BUFS - 1) {
            // Warmup: not enough frames queued yet to return a completed buffer.
            // Block-wait for the current frame so we have valid data to return.
            cuda_check(cudaStreamSynchronize(stream_), "cudaStreamSynchronize (warmup)");
            warmup_count_++;
            auto pinned = h_pinned_[cur_write];
            return make_pinned_ref(pinned);
        }

        // Steady state: return the buffer written 2 frames ago (fully complete)
        auto pinned = h_pinned_[cur_read];
        return make_pinned_ref(pinned);
#else
        return nullptr;
#endif
    }

    std::shared_ptr<void> convert_bgra(
        const core::video_format_desc& channel_format_desc,
        const core::video_format_desc& decklink_format_desc,
        const port_configuration&      config,
        const core::const_frame&       frame)
    {
#ifdef ENABLE_VULKAN
        auto tex = frame.texture();
        if (!tex) return nullptr;

        auto* wrapper = dynamic_cast<accelerator::vulkan::texture_wrapper*>(tex.get());
        if (!wrapper) return nullptr;

        caspar::timer total_timer;
        caspar::timer step_timer;

        void*    sem_handle = wrapper->render_semaphore_handle();
        uint64_t sem_value  = wrapper->render_semaphore_value();

        auto vk_tex = wrapper->vk_texture();
        if (!vk_tex) return nullptr;

        void* handle = vk_tex->export_native_handle();
        if (!handle) return nullptr;

        // The BGRA8 kernel only supports 8-bit textures. For 16-bit textures,
        // fall back to CPU path (the V210 path handles 16-bit natively).
        if (vk_tex->depth() != common::bit_depth::bit8) return nullptr;

        int src_w = vk_tex->width();
        int src_h = vk_tex->height();
        int src_x = config.src_x;
        int src_y = config.src_y;
        int dst_w = decklink_format_desc.width;
        int dst_h = decklink_format_desc.height;

        step_timer = caspar::timer();
        cudaSetDevice(cuda_device_);
        auto surf = ensure_import(handle, vk_tex->alloc_size(), src_w, src_h, false);

        size_t bgra_sz = (size_t)dst_w * dst_h * 4;
        ensure_output_buffers(dst_w, dst_h, false);
        double import_ms = step_timer.elapsed() * 1000.0;

        int cur_write = write_idx_;
        int cur_read  = (cur_write + 1) % NUM_ASYNC_BUFS;

        step_timer = caspar::timer();
        if (warmup_count_ >= NUM_ASYNC_BUFS - 1)
            cuda_check(cudaEventSynchronize(d2h_event_[cur_read]), "cudaEventSynchronize");
        double sync_ms = step_timer.elapsed() * 1000.0;

        // GPU-side wait for VK render, fallback to CPU fence
        step_timer = caspar::timer();
        if (sem_handle && gpu_wait_available_) {
            if (!try_gpu_wait(sem_handle, sem_value)) {
                wrapper->ensure_render_complete();
            }
        } else {
            wrapper->ensure_render_complete();
        }
        double fence_ms = step_timer.elapsed() * 1000.0;

        step_timer = caspar::timer();
        auto err = cuda_vk_launch_surface_to_bgra8(
            surf, d_bgra_[cur_write],
            src_x, src_y,
            dst_w, dst_h,
            src_w, src_h,
            stream_);

        if (err != cudaSuccess) {
            CASPAR_LOG(warning) << L"[cuda_vk_strategy] BGRA kernel launch failed: " << cudaGetErrorString(err);
            return nullptr;
        }

        cuda_check(cudaMemcpyAsync(h_pinned_[cur_write], d_bgra_[cur_write], bgra_sz,
                                   cudaMemcpyDeviceToHost, stream_), "cudaMemcpyAsync D2H");
        cuda_check(cudaEventRecord(d2h_event_[cur_write], stream_), "cudaEventRecord");
        double launch_ms = step_timer.elapsed() * 1000.0;

        double total_ms = total_timer.elapsed() * 1000.0;

        write_idx_ = (cur_write + 1) % NUM_ASYNC_BUFS;

        // Periodic timing report (shared with v210 path)
        accum_fence_ms_   += fence_ms;
        accum_import_ms_  += import_ms;
        accum_sync_ms_    += sync_ms;
        accum_launch_ms_  += launch_ms;
        accum_total_ms_   += total_ms;
        frame_count_++;
        if (frame_count_ % 50 == 0) {
            double n = 50.0;
            CASPAR_LOG(debug) << L"[cuda_vk_strategy] avg over 50 frames (BGRA): "
                              << L"fence=" << (accum_fence_ms_ / n) << L"ms "
                              << L"import=" << (accum_import_ms_ / n) << L"ms "
                              << L"sync=" << (accum_sync_ms_ / n) << L"ms "
                              << L"launch=" << (accum_launch_ms_ / n) << L"ms "
                              << L"total=" << (accum_total_ms_ / n) << L"ms";
            accum_fence_ms_ = accum_import_ms_ = accum_sync_ms_ = accum_launch_ms_ = accum_total_ms_ = 0.0;
        }

        if (warmup_count_ < NUM_ASYNC_BUFS - 1) {
            cuda_check(cudaStreamSynchronize(stream_), "cudaStreamSynchronize (warmup)");
            warmup_count_++;
            auto pinned = h_pinned_[cur_write];
            return make_pinned_ref(pinned);
        }

        auto pinned = h_pinned_[cur_read];
        return make_pinned_ref(pinned);
#else
        return nullptr;
#endif
    }
};

// ===========================================================================
// Public interface
// ===========================================================================

cuda_vk_strategy::cuda_vk_strategy(bool is_hdr, bool use_bt2020,
                                   spl::shared_ptr<format_strategy> fallback,
                                   bool needs_v210)
    : impl_(std::make_unique<impl>(is_hdr, use_bt2020, std::move(fallback), needs_v210))
{
    CASPAR_LOG(info) << L"[cuda_vk_strategy] GPU-direct decklink: "
                     << (is_hdr ? L"HDR " : L"SDR ")
                     << (use_bt2020 ? L"BT.2020" : L"BT.709")
                     << (needs_v210 ? L" V210" : L"")
                     << L" v210 packing on CUDA";
}

cuda_vk_strategy::~cuda_vk_strategy() = default;

BMDPixelFormat cuda_vk_strategy::get_pixel_format()
{
    return (impl_->is_hdr_ || impl_->needs_v210_) ? bmdFormat10BitYUV : impl_->fallback_->get_pixel_format();
}

int cuda_vk_strategy::get_row_bytes(int width)
{
    if (impl_->is_hdr_ || impl_->needs_v210_) {
        return ((width + 47) / 48) * 128;
    }
    return impl_->fallback_->get_row_bytes(width);
}

std::shared_ptr<void> cuda_vk_strategy::allocate_frame_data(const core::video_format_desc& format_desc)
{
    // Not used for GPU path — the pinned buffer is managed internally
    return impl_->fallback_->allocate_frame_data(format_desc);
}

std::shared_ptr<void> cuda_vk_strategy::convert_frame_for_port(
    const core::video_format_desc& channel_format_desc,
    const core::video_format_desc& decklink_format_desc,
    const port_configuration&      config,
    const core::const_frame&       frame1,
    const core::const_frame&       frame2,
    BMDFieldDominance              field_dominance)
{
    // For progressive frames with a VK texture, use the GPU path
    if (frame1 && field_dominance == bmdProgressiveFrame) {
        try {
            std::shared_ptr<void> result;
            if (impl_->is_hdr_ || impl_->needs_v210_) {
                result = impl_->convert_v210(channel_format_desc, decklink_format_desc, config, frame1);
            } else {
                result = impl_->convert_bgra(channel_format_desc, decklink_format_desc, config, frame1);
            }
            if (result) return result;
            static bool logged_null = false;
            if (!logged_null) {
                CASPAR_LOG(debug) << L"[cuda_vk_strategy] GPU path returned null (no VK texture) — using CPU fallback";
                logged_null = true;
            }
        } catch (const std::exception& ex) {
            CASPAR_LOG(warning) << L"[cuda_vk_strategy] GPU path failed, falling back to CPU: " << ex.what();
        }
    } else {
        static bool logged_skip = false;
        if (!logged_skip) {
            CASPAR_LOG(debug) << L"[cuda_vk_strategy] skipping GPU path: frame1="
                              << (frame1 ? L"yes" : L"no")
                              << L" field_dominance=" << static_cast<int>(field_dominance);
            logged_skip = true;
        }
    }

    // Fallback to CPU strategy for interlaced, empty frames, or on error
    return impl_->fallback_->convert_frame_for_port(
        channel_format_desc, decklink_format_desc, config, frame1, frame2, field_dominance);
}

// ===========================================================================
// Factory
// ===========================================================================

spl::shared_ptr<format_strategy> try_create_cuda_vk_strategy(
    bool is_hdr, bool use_bt2020,
    spl::shared_ptr<format_strategy> fallback,
    bool needs_v210)
{
    try {
        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
            CASPAR_LOG(info) << L"[cuda_vk_strategy] No CUDA devices — using CPU strategy";
            return fallback;
        }
        return spl::make_shared_ptr(
            std::shared_ptr<format_strategy>(
                std::make_shared<cuda_vk_strategy>(is_hdr, use_bt2020, std::move(fallback), needs_v210)));
    } catch (const std::exception& ex) {
        CASPAR_LOG(warning) << L"[cuda_vk_strategy] Init failed: " << ex.what() << L" — using CPU strategy";
        return fallback;
    }
}

}} // namespace caspar::decklink
