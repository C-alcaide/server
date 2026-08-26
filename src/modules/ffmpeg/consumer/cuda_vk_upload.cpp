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

#include "cuda_vk_upload.h"

#include <core/frame/frame.h>

#include <common/log.h>

#ifdef CASPAR_FFMPEG_HAS_CUDA

#include <cuda.h>
#include <cuda_runtime.h>

#include <mutex>

#include "../../cuda_gl_interop_lock.h"

#ifndef _WIN32
#include <unistd.h>
#endif

#include <array>
#include <string>

namespace caspar { namespace ffmpeg {

namespace {

/// How many imports to keep. The mixer's attachment pool is capped at four, so four
/// covers a steady-state channel without ever re-importing.
constexpr int kSlots = 4;

} // namespace

struct cuda_vk_uploader::impl
{
    struct slot
    {
        void*                       handle = nullptr;
        cudaExternalMemory_t        ext    = nullptr;
        cudaMipmappedArray_t        mipmap = nullptr;
        cudaArray_t                 array  = nullptr;
        int                         width  = 0;
        int                         height = 0;
    };

    std::array<slot, kSlots> slots_{};
    int                      next_    = 0;
    CUcontext                context_ = nullptr;
    /// Our own stream, so the wait below is a wait for OUR copy rather than for the device.
    /// See `copy_to_device`: on the default stream this synchronised against every other CUDA
    /// user in the process, and there are several.
    cudaStream_t             stream_  = nullptr;
    /// `cudaEventBlockingSync`, and that flag is the entire reason this exists rather than a
    /// plain `cudaStreamSynchronize`. CUDA's default wait SPINS: the thread burns a core
    /// polling until the copy lands. At 50 fps that is a per-frame busy-wait on the consumer
    /// thread, paid to save a readback -- which is how a zero-copy path ends up costing more
    /// CPU than the copy it removes. A blocking event deschedules instead.
    cudaEvent_t              done_    = nullptr;
    /// The mixer's timeline semaphore, imported once. Lets the COPY wait for the render on the
    /// GPU instead of this thread waiting for it on the CPU -- see `copy_to_device`.
    cudaExternalSemaphore_t  sem_        = nullptr;
    void*                    sem_handle_ = nullptr;
    std::string              last_error_;

    ~impl() { release(); }

    void release()
    {
        if (sem_) {
            cudaDestroyExternalSemaphore(sem_);
            sem_        = nullptr;
            sem_handle_ = nullptr;
        }
        if (done_) {
            cudaEventDestroy(done_);
            done_ = nullptr;
        }
        if (stream_) {
            cudaStreamDestroy(stream_);
            stream_ = nullptr;
        }
        {
            // The release half, on teardown. A consumer being destroyed while another is
            // importing is exactly the interleaving this lock exists for.
            std::lock_guard<std::mutex> interop_lk(caspar::cuda_gl_interop_mutex());
            for (auto& s : slots_) {
                if (s.mipmap)
                    cudaFreeMipmappedArray(s.mipmap);
                if (s.ext)
                    cudaDestroyExternalMemory(s.ext);
                s = slot{};
            }
        }
        next_ = 0;
    }

    /// Pushes FFmpeg's CUDA context for the duration of a call.
    ///
    /// Device pointers are not valid across contexts, and writing the encoder's frames
    /// through a pointer belonging to another context kills the process with an access
    /// violation and no exception to catch -- the same trap cuda_gl_upload documents.
    struct scoped_context
    {
        bool pushed = false;
        explicit scoped_context(CUcontext ctx)
        {
            if (ctx && cuCtxPushCurrent(ctx) == CUDA_SUCCESS)
                pushed = true;
        }
        ~scoped_context()
        {
            if (pushed) {
                CUcontext popped = nullptr;
                cuCtxPopCurrent(&popped);
            }
        }
    };

    slot* find_or_import(void* handle, unsigned long long size, int width, int height)
    {
        // SERIALISED for the whole routine, not per call. This evicts (destroy) and then
        // imports, and `cuda_gl_interop_lock.h` covers both halves of that pair -- an
        // asynchronous swap elsewhere in the process puts its import beside this destroy, and
        // the driver interop layer is not thread-safe across the two even for distinct
        // resources. Held across the mapped-array calls too, because the failure paths below
        // destroy the memory they just imported.
        //
        // NOT one-time setup: this is a CACHE MISS path. The mixer's attachment pool rotates
        // and is rebuilt on a raster change, so imports and evictions recur for the life of
        // the consumer -- which is what makes the lock load-bearing rather than belt-and-braces.
        std::lock_guard<std::mutex> interop_lk(caspar::cuda_gl_interop_mutex());
        for (auto& s : slots_) {
            if (s.array && s.handle == handle && s.width == width && s.height == height)
                return &s;
        }

        // Same handle at a different size means the attachment pool was rebuilt under
        // it; drop the stale import rather than adding a second entry.
        for (auto& s : slots_) {
            if (s.handle == handle) {
                if (s.mipmap)
                    cudaFreeMipmappedArray(s.mipmap);
                if (s.ext)
                    cudaDestroyExternalMemory(s.ext);
                s = slot{};
            }
        }

        slot* target = nullptr;
        for (auto& s : slots_) {
            if (!s.array) {
                target = &s;
                break;
            }
        }
        if (!target) {
            // Full: evict round-robin.
            target = &slots_[next_];
            next_  = (next_ + 1) % kSlots;
            if (target->mipmap)
                cudaFreeMipmappedArray(target->mipmap);
            if (target->ext)
                cudaDestroyExternalMemory(target->ext);
            *target = slot{};
        }

        cudaExternalMemoryHandleDesc desc{};
#ifdef _WIN32
        desc.type                = cudaExternalMemoryHandleTypeOpaqueWin32;
        desc.handle.win32.handle = handle;
#else
        desc.type      = cudaExternalMemoryHandleTypeOpaqueFd;
        desc.handle.fd = dup(static_cast<int>(reinterpret_cast<intptr_t>(handle)));
#endif
        desc.size  = size;
        desc.flags = 0;

        auto err = cudaImportExternalMemory(&target->ext, &desc);
#ifndef _WIN32
        // cudaImportExternalMemory does not consume the fd on failure.
        if (err != cudaSuccess && desc.handle.fd >= 0)
            ::close(desc.handle.fd);
#endif
        if (err != cudaSuccess) {
            last_error_ = std::string("cudaImportExternalMemory: ") + cudaGetErrorString(err);
            *target     = slot{};
            return nullptr;
        }

        cudaExternalMemoryMipmappedArrayDesc mm{};
        mm.offset        = 0;
        mm.formatDesc    = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
        mm.extent.width  = static_cast<unsigned>(width);
        mm.extent.height = static_cast<unsigned>(height);
        mm.extent.depth  = 0;
        mm.numLevels     = 1;
        mm.flags         = cudaArrayDefault;

        err = cudaExternalMemoryGetMappedMipmappedArray(&target->mipmap, target->ext, &mm);
        if (err != cudaSuccess) {
            last_error_ = std::string("cudaExternalMemoryGetMappedMipmappedArray: ") + cudaGetErrorString(err);
            cudaDestroyExternalMemory(target->ext);
            *target = slot{};
            return nullptr;
        }

        err = cudaGetMipmappedArrayLevel(&target->array, target->mipmap, 0);
        if (err != cudaSuccess) {
            last_error_ = std::string("cudaGetMipmappedArrayLevel: ") + cudaGetErrorString(err);
            cudaFreeMipmappedArray(target->mipmap);
            cudaDestroyExternalMemory(target->ext);
            *target = slot{};
            return nullptr;
        }

        target->handle = handle;
        target->width  = width;
        target->height = height;
        return target;
    }
};

cuda_vk_uploader::cuda_vk_uploader()
    : impl_(new impl())
{
}

cuda_vk_uploader::~cuda_vk_uploader() = default;

bool cuda_vk_uploader::available()
{
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

void cuda_vk_uploader::set_context(void* cu_context) { impl_->context_ = static_cast<CUcontext>(cu_context); }

void cuda_vk_uploader::release()
{
    // Under the context the imports were made in. The runtime API acts on whichever
    // context is current, so destroying them under a different one -- or after
    // FFmpeg's has gone -- is how this takes the process down at teardown.
    impl::scoped_context ctx(impl_->context_);
    impl_->release();
}

const char* cuda_vk_uploader::last_error() const
{
    return impl_->last_error_.empty() ? "" : impl_->last_error_.c_str();
}

bool cuda_vk_uploader::copy_to_device(const std::shared_ptr<core::texture>& tex, void* dst, std::size_t dst_pitch)
{
    if (!tex || !dst)
        return false;

    auto* handle = tex->export_native_handle();
    const auto size = tex->export_alloc_size();
    const int  w    = tex->tex_width();
    const int  h    = tex->tex_height();
    if (!handle || size == 0 || w <= 0 || h <= 0) {
        impl_->last_error_ = "the composited texture is not exportable";
        return false;
    }

    impl::scoped_context ctx(impl_->context_);

    auto* s = impl_->find_or_import(handle, size, w, h);
    if (!s)
        return false;

    // The attachment is RGBA8, so a texel is four bytes in R,G,B,A order and this is a
    // straight byte move -- which is why the frames context declares AV_PIX_FMT_RGB0
    // and no colour conversion is needed, exactly as on the OpenGL path.
    // Created here rather than in the constructor because it must belong to the context
    // pushed above -- a stream outlives the call but not the context it was made in.
    if (!impl_->stream_) {
        auto serr = cudaStreamCreateWithFlags(&impl_->stream_, cudaStreamNonBlocking);
        if (serr != cudaSuccess) {
            impl_->last_error_ = std::string("cudaStreamCreateWithFlags: ") + cudaGetErrorString(serr);
            return false;
        }
    }
    if (!impl_->done_) {
        auto eerr = cudaEventCreateWithFlags(&impl_->done_,
                                             cudaEventBlockingSync | cudaEventDisableTiming);
        if (eerr != cudaSuccess) {
            impl_->last_error_ = std::string("cudaEventCreateWithFlags: ") + cudaGetErrorString(eerr);
            return false;
        }
    }

    // ── Ordering against the mixer, on the GPU rather than on this thread ────────────────
    // The mixer may still be rendering into this attachment and the consumer runs on its own
    // thread, so the two must be ordered. `ensure_render_complete()` does it with a CPU fence
    // wait, which blocks this thread for as long as the mixer needs -- every frame, at frame
    // rate. Importing the mixer's timeline semaphore lets the copy wait on the GPU instead, so
    // the CPU only ever waits for the copy itself.
    //
    // Falls back to the CPU wait when the texture exposes no semaphore, which is the case for
    // any wrapper built by the bare constructor. Correct either way; the semaphore is faster.
    bool ordered = false;
    if (auto* sem_handle = tex->render_semaphore_handle()) {
        if (impl_->sem_ && impl_->sem_handle_ != sem_handle) {
            cudaDestroyExternalSemaphore(impl_->sem_);
            impl_->sem_        = nullptr;
            impl_->sem_handle_ = nullptr;
        }
        if (!impl_->sem_) {
            cudaExternalSemaphoreHandleDesc sd{};
            sd.type                = cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32;
            sd.handle.win32.handle = sem_handle;
            // NOT `cudaExternalSemaphoreHandleTypeOpaqueWin32`: that is a BINARY semaphore, and
            // importing a timeline one under it makes every wait return immediately -- an
            // ordering bug with no error and no symptom until the picture tears.
            if (cudaImportExternalSemaphore(&impl_->sem_, &sd) == cudaSuccess)
                impl_->sem_handle_ = sem_handle;
            else
                impl_->sem_ = nullptr;
        }
        if (impl_->sem_) {
            cudaExternalSemaphoreWaitParams wp{};
            wp.params.fence.value = tex->render_semaphore_value();
            if (cudaWaitExternalSemaphoresAsync(&impl_->sem_, &wp, 1, impl_->stream_) == cudaSuccess)
                ordered = true;
        }
    }
    if (!ordered)
        tex->ensure_render_complete();

    auto err = cudaMemcpy2DFromArrayAsync(
        dst, dst_pitch, s->array, 0, 0, static_cast<std::size_t>(w) * 4, static_cast<std::size_t>(h),
        cudaMemcpyDeviceToDevice, impl_->stream_);
    if (err != cudaSuccess) {
        impl_->last_error_ = std::string("cudaMemcpy2DFromArrayAsync: ") + cudaGetErrorString(err);
        return false;
    }

    // **Our stream, not the device.** This was `cudaDeviceSynchronize()`, which blocks until
    // EVERY context on the device is idle -- and this process runs several CUDA users at once:
    // the FFmpeg consumer, `cuda_prores`, and, on the GStreamer egress path, the NVENC encoder
    // consuming the very buffer being filled. So each frame waited on unrelated work, and the
    // more the GPU was doing the longer it waited.
    //
    // Measured on the GStreamer CUDA egress at 1080p50: 2.09 cores device-wide against the host
    // readback's 1.96, i.e. the "zero-copy" route cost MORE CPU than the readback it replaces.
    // Correctness never needed the wider wait -- the copy has one producer and one consumer.
    err = cudaEventRecord(impl_->done_, impl_->stream_);
    if (err == cudaSuccess)
        err = cudaEventSynchronize(impl_->done_);
    if (err != cudaSuccess) {
        impl_->last_error_ = std::string("waiting for the copy: ") + cudaGetErrorString(err);
        return false;
    }

    return true;
}

}} // namespace caspar::ffmpeg

#else // CASPAR_FFMPEG_HAS_CUDA not defined

namespace caspar { namespace ffmpeg {

struct cuda_vk_uploader::impl
{
};

cuda_vk_uploader::cuda_vk_uploader()  = default;
cuda_vk_uploader::~cuda_vk_uploader() = default;

bool        cuda_vk_uploader::available() { return false; }
void        cuda_vk_uploader::set_context(void*) {}
void        cuda_vk_uploader::release() {}
const char* cuda_vk_uploader::last_error() const { return "CUDA support was not built"; }
bool cuda_vk_uploader::copy_to_device(const std::shared_ptr<core::texture>&, void*, std::size_t) { return false; }

}} // namespace caspar::ffmpeg

#endif
