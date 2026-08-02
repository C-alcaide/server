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
    std::string              last_error_;

    ~impl() { release(); }

    void release()
    {
        for (auto& s : slots_) {
            if (s.mipmap)
                cudaFreeMipmappedArray(s.mipmap);
            if (s.ext)
                cudaDestroyExternalMemory(s.ext);
            s = slot{};
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

    // The mixer may still be rendering into this attachment; the consumer runs on its
    // own thread. This is what orders the two.
    tex->ensure_render_complete();

    impl::scoped_context ctx(impl_->context_);

    auto* s = impl_->find_or_import(handle, size, w, h);
    if (!s)
        return false;

    // The attachment is RGBA8, so a texel is four bytes in R,G,B,A order and this is a
    // straight byte move -- which is why the frames context declares AV_PIX_FMT_RGB0
    // and no colour conversion is needed, exactly as on the OpenGL path.
    auto err = cudaMemcpy2DFromArray(
        dst, dst_pitch, s->array, 0, 0, static_cast<std::size_t>(w) * 4, static_cast<std::size_t>(h),
        cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        impl_->last_error_ = std::string("cudaMemcpy2DFromArray: ") + cudaGetErrorString(err);
        return false;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        impl_->last_error_ = std::string("cudaDeviceSynchronize: ") + cudaGetErrorString(err);
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
