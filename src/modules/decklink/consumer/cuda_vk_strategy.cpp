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
#include <common/utf.h>

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
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

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
    // How many frames the DeckLink driver can hold scheduled at once. Output
    // buffers handed to it must stay untouched for at least that long.
    const int  buffer_depth_;

    // CUDA resources
    int              cuda_device_ = 0;
    cudaStream_t     stream_      = nullptr;
    static constexpr int NUM_DEV_BUFS = 3;
    uint32_t*        d_v210_[NUM_DEV_BUFS] = {};  // device V210 pack targets
    uint8_t*         d_bgra_[NUM_DEV_BUFS] = {};  // device BGRA8 pack targets (SDR only)
    size_t           dev_buf_sz_  = 0;
    int              dev_idx_     = 0;

    // Pipeline depth: the D2H for frame N is not waited on until frame
    // N + PIPELINE_DEPTH, so the copy has that many frame intervals to land.
    // This is a latency/throughput knob only -- it says nothing about how long a
    // buffer must stay valid, which is what buffer_depth_ governs.
    static constexpr int PIPELINE_DEPTH = 2;

    // Output buffers come from a refcounted pool, NOT a fixed ring.
    //
    // Previously this was three pinned buffers rotated by write_idx_, handed to
    // DeckLink through a shared_ptr with a no-op deleter. DeckLink keeps
    // buffer_depth() frames (4-5) scheduled and DMAs each at display time, so a
    // 3-slot ring rewrote the buffer it had just handed over -- the buffer
    // returned on frame N was the D2H target again on frame N+1, while still
    // queued. That produced stale and torn SDI frames with nothing logged.
    //
    // gpu_output_buffer_pool already solves this for the CPU strategies: its
    // shared_ptr deleter returns the buffer to the free list, so a buffer cannot
    // be reused until DeckLink has released the frame holding it.
    std::shared_ptr<gpu_output_buffer_pool> pool_;
    size_t                                  pool_buf_sz_ = 0;

    // In-flight D2H copies, oldest first. Each entry owns its pooled buffer for
    // as long as it is queued here; handing it out transfers that ownership to
    // the caller (and ultimately to DeckLink).
    struct inflight
    {
        std::shared_ptr<void> buf;
        cudaEvent_t           done = nullptr;
    };
    std::deque<inflight>     inflight_;
    std::vector<cudaEvent_t> event_pool_;   // recycled events

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

    // Bind to the CUDA device that is the same physical GPU as the VK texture, and
    // create the stream on it.
    //
    // This used to be hardcoded to device 0 on the assumption that "the VK mixer
    // also runs on the primary discrete GPU". This fork lets the mixer be placed on
    // any GPU (accelerator::vulkan::device takes a gpu_index, deduplicated by LUID),
    // so on a multi-GPU rig importing a GPU-1 texture into a device-0 context either
    // fails outright or silently degrades to a peer copy.
    //
    // Vulkan hands us the owning device's LUID with the texture, and
    // cudaDeviceProp::luid is the same Win32 adapter LUID, so the two match directly.
    // Resolved once on the first frame; on failure it stays on device 0, which is
    // the old behaviour.
    bool cuda_device_ready_ = false;

    void ensure_cuda_device(const uint8_t* vk_luid)
    {
        if (cuda_device_ready_)
            return;
        cuda_device_ready_ = true; // one attempt either way

        int count = 0;
        if (vk_luid && cudaGetDeviceCount(&count) == cudaSuccess) {
            bool matched = false;
            for (int d = 0; d < count; ++d) {
                cudaDeviceProp p{};
                if (cudaGetDeviceProperties(&p, d) != cudaSuccess)
                    continue;
                if (std::memcmp(p.luid, vk_luid, sizeof(p.luid)) == 0) {
                    cuda_device_ = d;
                    matched      = true;
                    CASPAR_LOG(info) << L"[cuda_vk_strategy] Bound to CUDA device " << d << L" (" << p.name
                                     << L") matching the mixer's Vulkan GPU by LUID.";
                    break;
                }
            }
            if (!matched)
                CASPAR_LOG(warning) << L"[cuda_vk_strategy] No CUDA device matches the mixer's Vulkan GPU LUID; "
                                       L"falling back to device 0 (texture import may fail on a multi-GPU rig).";
        } else if (!vk_luid) {
            CASPAR_LOG(info) << L"[cuda_vk_strategy] Vulkan texture reports no LUID; using CUDA device 0.";
        }

        cuda_check(cudaSetDevice(cuda_device_), "cudaSetDevice");
        cuda_check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "cudaStreamCreate");
    }

    /// Decline the GPU path, saying which precondition failed.
    ///
    /// Every caller of this used to `return nullptr` into one shared caller-side message,
    /// `GPU path returned null (no VK texture)`, which named one of four possible causes
    /// and gave no way to tell them apart. That matters more here than usual: declining
    /// ends at `blank_output()`, a memset-zero buffer, which is illegal super-black on the
    /// wire and reads downstream exactly like a dead card or an unplugged cable.
    ///
    /// At warning, because this is a silent full-black output rather than a tuning detail,
    /// and it should not need `debug` to be visible.
    ///
    /// COUNTED, not one-shot. A one-shot version of this was actively misleading: the
    /// channel legitimately has no layers during preroll, so `texture()` is null before
    /// anything is playing (image_mixer.cpp's `layers.empty()` bypass returns a null
    /// texture by design). The first occurrence therefore always fires at startup and says
    /// nothing about playback -- "logged once at startup, harmless" and "happening on every
    /// frame" produced identical logs. The running total is what separates them.
    std::shared_ptr<void> decline(const char* why) const
    {
        static std::mutex                            m;
        static std::map<std::string, std::uint64_t>  counts;
        std::uint64_t n = 0;
        {
            std::lock_guard<std::mutex> lock(m);
            n = ++counts[why];
        }
        // First one, then sparsely: enough to show it is persistent without flooding.
        if (n == 1 || n == 10 || n == 100 || (n % 500) == 0) {
            CASPAR_LOG(warning) << L"[cuda_vk_strategy] declining the GPU path (occurrence "
                                << n << L"): " << u16(why)
                                << L" -- this frame's output is blank."
                                << (n == 1 ? L" A single occurrence at startup is expected:"
                                             L" the channel has no layers before playout."
                                           : L"");
        }
        return nullptr;
    }

    // A blank output frame, for the frames where the pipeline has nothing to hand
    // back yet.
    //
    // It must NOT be produced by falling through to the CPU strategy. This consumer
    // reports needs_cpu_frame_data()==false whenever a GPU readback mode is active
    // (decklink_consumer.cpp), so the mixer deliberately skips the host readback and
    // const_frame::image_data() is `unavailable` -- the CPU v210 packer then reads an
    // empty array and takes the process down. Zero-filled matches what the
    // consumer's own preroll schedules before playback starts.
    std::shared_ptr<void> blank_output(size_t bytes)
    {
        ensure_pool(bytes);
        auto b = pool_->acquire(bytes);
        if (b)
            std::memset(b.get(), 0, bytes);
        return b; // null only if the pool is out of memory; caller then declines
    }

    // Acquire an event, reusing a retired one when available.
    cudaEvent_t take_event()
    {
        if (!event_pool_.empty()) {
            auto e = event_pool_.back();
            event_pool_.pop_back();
            return e;
        }
        cudaEvent_t e = nullptr;
        cuda_check(cudaEventCreateWithFlags(&e, cudaEventDisableTiming), "cudaEventCreate");
        return e;
    }

    // Ensure the pool exists and its buffers are big enough for `bytes`.
    //
    // Sized buffer_depth_ + PIPELINE_DEPTH + 1: DeckLink can hold buffer_depth_
    // frames, this strategy holds up to PIPELINE_DEPTH more in inflight_, and one
    // is being filled. The pool grows on demand anyway, so this is the steady-state
    // count rather than a hard cap.
    void ensure_pool(size_t bytes)
    {
        if (pool_ && pool_buf_sz_ >= bytes)
            return;
        // A larger frame needs a new pool; the old one stays alive as long as any
        // buffer it handed out is still referenced (its state block is shared).
        pool_        = std::make_shared<gpu_output_buffer_pool>(
            bytes, buffer_depth_ + PIPELINE_DEPTH + 1, gpu_output_buffer_pool::pin_kind::cuda_pinned);
        pool_buf_sz_ = bytes;
    }

    // Timing diagnostics (periodic log)
    int      frame_count_ = 0;
    double   accum_fence_ms_   = 0.0;
    double   accum_import_ms_  = 0.0;
    double   accum_wait_ms_    = 0.0;
    double   accum_launch_ms_  = 0.0;
    double   accum_total_ms_   = 0.0;

    impl(bool is_hdr, bool use_bt2020, spl::shared_ptr<format_strategy> fallback, bool needs_v210, int buffer_depth)
        : use_bt2020_(use_bt2020)
        , is_hdr_(is_hdr)
        , needs_v210_(needs_v210)
        , buffer_depth_(buffer_depth > 0 ? buffer_depth : 4)
        , fallback_(std::move(fallback))
    {
        // The CUDA device is not chosen here: it has to match whichever GPU the VK
        // mixer is on, which is only discoverable from a frame's texture. See
        // ensure_cuda_device(), called on the first frame -- the stream is created
        // there too, since a stream belongs to the device current when it is made.
    }

    ~impl()
    {
        // Ensure all async GPU work has completed before destroying resources.
        // Without this, in-flight D2H copies or semaphore waits could access
        // freed buffers or destroyed events.
        //
        // The output buffers themselves need no such care: they belong to
        // gpu_output_buffer_pool, whose allocations outlive the pool object for
        // exactly as long as any frame still references them -- so a frame left in
        // the DeckLink queue at shutdown stays valid until the driver releases it.
        // (The old code freed them here with cudaFreeHost while the card could
        // still be reading, and relied on an `alive_` sentinel that nothing ever
        // checked.)
        if (stream_) {
            cudaSetDevice(cuda_device_);
            cudaStreamSynchronize(stream_);
        }

        drain_inflight();
        for (auto e : event_pool_)
            cudaEventDestroy(e);
        event_pool_.clear();

        for (int i = 0; i < num_cached_sems_; ++i) {
            if (cached_sems_[i].sem) cudaDestroyExternalSemaphore(cached_sems_[i].sem);
        }
        num_cached_sems_ = 0;
        for (int i = 0; i < num_cached_; ++i)
            cached_slots_[i].cleanup();
        num_cached_ = 0;
        for (int i = 0; i < NUM_DEV_BUFS; ++i) {
            if (d_v210_[i]) { cudaFree(d_v210_[i]); d_v210_[i] = nullptr; }
            if (d_bgra_[i]) { cudaFree(d_bgra_[i]); d_bgra_[i] = nullptr; }
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

    // Device-side pack targets. One per pipeline stage (not per DeckLink frame --
    // the card never sees these), rotated by dev_idx_.
    void ensure_output_buffers(int dst_w, int dst_h, bool need_v210)
    {
        const size_t need = need_v210 ? (size_t)((dst_w + 47) / 48) * 128 * dst_h
                                      : (size_t)dst_w * dst_h * 4;
        auto** bufs = need_v210 ? reinterpret_cast<void**>(d_v210_) : reinterpret_cast<void**>(d_bgra_);
        if (bufs[0] && dev_buf_sz_ >= need)
            return;

        // Drain first: these buffers are the source of D2H copies that may still be
        // in flight, and cudaFree on a live copy's source is a use-after-free. The
        // old code reallocated without any synchronisation at all.
        drain_inflight();
        cuda_check(cudaStreamSynchronize(stream_), "cudaStreamSynchronize (realloc)");

        for (int i = 0; i < NUM_DEV_BUFS; ++i) {
            if (d_v210_[i]) { cudaFree(d_v210_[i]); d_v210_[i] = nullptr; }
            if (d_bgra_[i]) { cudaFree(d_bgra_[i]); d_bgra_[i] = nullptr; }
        }
        for (int i = 0; i < NUM_DEV_BUFS; ++i) {
            if (need_v210)
                cuda_check(cudaMalloc(&d_v210_[i], need), "cudaMalloc v210");
            else
                cuda_check(cudaMalloc(&d_bgra_[i], need), "cudaMalloc bgra");
        }
        dev_buf_sz_ = need;
        dev_idx_    = 0;
    }

    // Wait out and release every queued copy. Their pooled buffers go back to the
    // pool as the entries are destroyed (unless DeckLink still holds a reference).
    void drain_inflight()
    {
        for (auto& f : inflight_) {
            if (f.done) {
                cudaEventSynchronize(f.done);
                event_pool_.push_back(f.done);
            }
        }
        inflight_.clear();
    }

    // Queue this frame's copy and hand back the oldest completed one, preserving
    // the previous 2-frame pipeline depth.
    //
    // Returns null while the queue is still filling: for the first PIPELINE_DEPTH
    // frames there is genuinely no copy old enough to have completed. The caller
    // must emit a blank frame for those -- see blank_output(), and do NOT route them
    // to the CPU strategy. Crucially, a buffer leaves this queue exactly once --
    // handing one out while it is still queued is the whole defect being fixed, so
    // the fill case must not hand out the entry it just pushed.
    std::shared_ptr<void> push_and_take(std::shared_ptr<void> buf)
    {
        auto ev = take_event();
        cuda_check(cudaEventRecord(ev, stream_), "cudaEventRecord");
        inflight_.push_back({std::move(buf), ev});

        if ((int)inflight_.size() <= PIPELINE_DEPTH)
            return nullptr;

        auto oldest = std::move(inflight_.front());
        inflight_.pop_front();
        cuda_check(cudaEventSynchronize(oldest.done), "cudaEventSynchronize");
        event_pool_.push_back(oldest.done);
        return std::move(oldest.buf);
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
                                    << cudaGetErrorString(err) << L" - falling back to CPU wait";
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
        // Four separate preconditions, each of which used to `return nullptr` into a
        // single caller-side message reading "GPU path returned null (no VK texture)".
        // Failing any of them ends in `blank_output()` -- a memset-zero buffer, i.e.
        // illegal super-black on the wire -- so the four causes are indistinguishable from
        // each other AND from a dead card. Naming them is the difference between a
        // diagnosis and a shrug.
        auto tex = frame.texture();
        if (!tex)
            return decline("the frame carries no texture at all (const_frame::texture() is "
                           "null) -- the mixer did not attach one, or something retagged "
                           "the frame and dropped it");

        auto* wrapper = dynamic_cast<accelerator::vulkan::texture_wrapper*>(tex.get());
        if (!wrapper)
            return decline("the frame's texture is not a vulkan texture_wrapper -- an "
                           "OpenGL mixer output, a producer's own texture, or RTTI not "
                           "matching across the module boundary");

        caspar::timer total_timer;
        caspar::timer step_timer;

        // Get semaphore info (cheap accessors, no blocking)
        void*    sem_handle = wrapper->render_semaphore_handle();
        uint64_t sem_value  = wrapper->render_semaphore_value();

        auto vk_tex = wrapper->vk_texture();
        if (!vk_tex)
            return decline("the texture_wrapper holds no vk_texture");

        // Get Win32 handle for CUDA import
        void* handle = vk_tex->export_native_handle();
        if (!handle)
            return decline("the vk_texture exports no native handle -- the image was not "
                           "created with an external-memory handle type, so CUDA cannot "
                           "import it");

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
        ensure_cuda_device(vk_tex->device_luid());
        cudaSetDevice(cuda_device_);
        auto surf = ensure_import(handle, vk_tex->alloc_size(), src_w, src_h, is_16bit);

        // Ensure output buffers (both ping-pong)
        size_t v210_row = (size_t)((dst_w + 47) / 48) * 128;
        size_t v210_sz  = v210_row * dst_h;
        ensure_output_buffers(dst_w, dst_h, true);
        double import_ms = step_timer.elapsed() * 1000.0;

        // A fresh pooled buffer for this frame's D2H. Its shared_ptr is what keeps
        // it out of circulation until DeckLink is finished with the frame.
        ensure_pool(v210_sz);
        auto out_buf = pool_->acquire(v210_sz);
        if (!out_buf)
            return nullptr;   // pool exhausted and out of memory: fall back to CPU

        const int cur_dev = dev_idx_;
        dev_idx_          = (cur_dev + 1) % NUM_DEV_BUFS;

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
            d_v210_[cur_dev],
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

        // Enqueue D2H copy into this frame's own buffer — does NOT block
        cuda_check(cudaMemcpyAsync(out_buf.get(), d_v210_[cur_dev], v210_sz,
                                   cudaMemcpyDeviceToHost, stream_), "cudaMemcpyAsync D2H");
        double launch_ms = step_timer.elapsed() * 1000.0;

        // Hand the buffer over, and TIME IT. `push_and_take` contains the only call in
        // this path that blocks the DeckLink thread -- cudaEventSynchronize on the frame
        // PIPELINE_DEPTH back. Reading `total_ms` before this, as this code did until
        // 2026-08-20, excluded the one wait the number was being used to rule out.
        caspar::timer wait_timer;
        auto          ready   = push_and_take(std::move(out_buf));
        double        wait_ms = wait_timer.elapsed() * 1000.0;

        double total_ms = total_timer.elapsed() * 1000.0;

        // Periodic timing report (every 50 frames)
        accum_fence_ms_   += fence_ms;
        accum_import_ms_  += import_ms;
        accum_wait_ms_    += wait_ms;
        accum_launch_ms_  += launch_ms;
        accum_total_ms_   += total_ms;
        frame_count_++;
        if (frame_count_ % 50 == 0) {
            double n = 50.0;
            CASPAR_LOG(debug) << L"[cuda_vk_strategy] avg over 50 frames: "
                              << L"fence=" << (accum_fence_ms_ / n) << L"ms "
                              << L"import=" << (accum_import_ms_ / n) << L"ms "
                              << L"wait=" << (accum_wait_ms_ / n) << L"ms "
                              << L"launch=" << (accum_launch_ms_ / n) << L"ms "
                              << L"total=" << (accum_total_ms_ / n) << L"ms";
            accum_fence_ms_ = accum_import_ms_ = accum_wait_ms_ = accum_launch_ms_ = accum_total_ms_ = 0.0;
        }

        if (ready)
            return ready;
        return blank_output(v210_sz); // pipeline still filling
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
        // Same four preconditions as convert_v210, named for the same reason.
        auto tex = frame.texture();
        if (!tex)
            return decline("the frame carries no texture at all (const_frame::texture() is "
                           "null)");

        auto* wrapper = dynamic_cast<accelerator::vulkan::texture_wrapper*>(tex.get());
        if (!wrapper)
            return decline("the frame's texture is not a vulkan texture_wrapper");

        caspar::timer total_timer;
        caspar::timer step_timer;

        void*    sem_handle = wrapper->render_semaphore_handle();
        uint64_t sem_value  = wrapper->render_semaphore_value();

        auto vk_tex = wrapper->vk_texture();
        if (!vk_tex)
            return decline("the texture_wrapper holds no vk_texture");

        void* handle = vk_tex->export_native_handle();
        if (!handle)
            return decline("the vk_texture exports no native handle");

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
        ensure_cuda_device(vk_tex->device_luid());
        cudaSetDevice(cuda_device_);
        auto surf = ensure_import(handle, vk_tex->alloc_size(), src_w, src_h, false);

        size_t bgra_sz = (size_t)dst_w * dst_h * 4;
        ensure_output_buffers(dst_w, dst_h, false);
        double import_ms = step_timer.elapsed() * 1000.0;

        ensure_pool(bgra_sz);
        auto out_buf = pool_->acquire(bgra_sz);
        if (!out_buf)
            return nullptr;

        const int cur_dev = dev_idx_;
        dev_idx_          = (cur_dev + 1) % NUM_DEV_BUFS;

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
            surf, d_bgra_[cur_dev],
            src_x, src_y,
            dst_w, dst_h,
            src_w, src_h,
            stream_);

        if (err != cudaSuccess) {
            CASPAR_LOG(warning) << L"[cuda_vk_strategy] BGRA kernel launch failed: " << cudaGetErrorString(err);
            return nullptr;
        }

        cuda_check(cudaMemcpyAsync(out_buf.get(), d_bgra_[cur_dev], bgra_sz,
                                   cudaMemcpyDeviceToHost, stream_), "cudaMemcpyAsync D2H");
        double launch_ms = step_timer.elapsed() * 1000.0;

        // Hand the buffer over, and TIME IT. `push_and_take` contains the only call in
        // this path that blocks the DeckLink thread -- cudaEventSynchronize on the frame
        // PIPELINE_DEPTH back. Reading `total_ms` before this, as this code did until
        // 2026-08-20, excluded the one wait the number was being used to rule out.
        caspar::timer wait_timer;
        auto          ready   = push_and_take(std::move(out_buf));
        double        wait_ms = wait_timer.elapsed() * 1000.0;

        double total_ms = total_timer.elapsed() * 1000.0;

        // Periodic timing report (shared with v210 path)
        accum_fence_ms_   += fence_ms;
        accum_import_ms_  += import_ms;
        accum_wait_ms_    += wait_ms;
        accum_launch_ms_  += launch_ms;
        accum_total_ms_   += total_ms;
        frame_count_++;
        if (frame_count_ % 50 == 0) {
            double n = 50.0;
            CASPAR_LOG(debug) << L"[cuda_vk_strategy] avg over 50 frames (BGRA): "
                              << L"fence=" << (accum_fence_ms_ / n) << L"ms "
                              << L"import=" << (accum_import_ms_ / n) << L"ms "
                              << L"wait=" << (accum_wait_ms_ / n) << L"ms "
                              << L"launch=" << (accum_launch_ms_ / n) << L"ms "
                              << L"total=" << (accum_total_ms_ / n) << L"ms";
            accum_fence_ms_ = accum_import_ms_ = accum_wait_ms_ = accum_launch_ms_ = accum_total_ms_ = 0.0;
        }

        if (ready)
            return ready;
        return blank_output(bgra_sz); // pipeline still filling
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
                                   bool needs_v210,
                                   int  buffer_depth)
    : impl_(std::make_unique<impl>(is_hdr, use_bt2020, std::move(fallback), needs_v210, buffer_depth))
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
            // No message here on purpose: `decline()` has already named which precondition
            // failed. This used to assert "no VK texture", which was one of four possible
            // causes stated as though it were the only one.
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

    // Fallback to CPU strategy for interlaced, empty frames, or on error.
    //
    // Only usable when the frame actually carries host pixels. This consumer reports
    // needs_cpu_frame_data()==false whenever a GPU readback mode is active, so for a
    // GPU-only frame the CPU v210 packer reads an `unavailable` image_data() and
    // takes the process down. Emit a blank frame in that case instead. (This also
    // covers interlaced output, which has always taken this path and has the same
    // problem -- a pre-existing hazard, not one introduced here.)
    if (frame1 && frame1.host_image_state() == core::host_image_availability::unavailable) {
        const size_t bytes = static_cast<size_t>(get_row_bytes(decklink_format_desc.width)) *
                             decklink_format_desc.height;
        CASPAR_LOG(debug) << L"[cuda_vk_strategy] No host pixels for the CPU fallback; emitting a blank frame.";
        return impl_->blank_output(bytes);
    }

    return impl_->fallback_->convert_frame_for_port(
        channel_format_desc, decklink_format_desc, config, frame1, frame2, field_dominance);
}

// ===========================================================================
// Factory
// ===========================================================================

spl::shared_ptr<format_strategy> try_create_cuda_vk_strategy(
    bool is_hdr, bool use_bt2020,
    spl::shared_ptr<format_strategy> fallback,
    bool needs_v210,
    int  buffer_depth)
{
    try {
        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
            CASPAR_LOG(info) << L"[cuda_vk_strategy] No CUDA devices - using CPU strategy";
            return fallback;
        }
        return spl::make_shared_ptr(
            std::shared_ptr<format_strategy>(
                std::make_shared<cuda_vk_strategy>(is_hdr, use_bt2020, std::move(fallback), needs_v210, buffer_depth)));
    } catch (const std::exception& ex) {
        CASPAR_LOG(warning) << L"[cuda_vk_strategy] Init failed: " << ex.what() << L" - using CPU strategy";
        return fallback;
    }
}

}} // namespace caspar::decklink
