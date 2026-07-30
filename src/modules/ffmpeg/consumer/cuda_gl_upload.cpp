/*
 * Copyright (c) 2026 CasparCG Contributors
 *
 * This file is part of CasparCG (www.casparcg.com)
 * and is licensed under the GNU General Public License v3.
 */

#include "../StdAfx.h"

#include "cuda_gl_upload.h"

#include <common/log.h>

#ifdef CASPAR_FFMPEG_HAS_CUDA

#include "../../cuda_gl_interop_lock.h"

#include <accelerator/ogl/util/device.h>
#include <accelerator/ogl/util/texture.h>

#include <GL/glew.h>

#include <cuda.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include <array>

namespace caspar { namespace ffmpeg {

struct cuda_gl_uploader::impl
{
    // Registering a GL texture with CUDA is expensive and the mixer recycles its
    // textures through a pool, so the same handful of names come back round.
    // Cache the registrations and evict the oldest when full.
    struct slot
    {
        int                     gl_id    = 0;
        cudaGraphicsResource_t  resource = nullptr;
    };
    static constexpr int      max_slots = 8;
    std::array<slot, max_slots> slots_{};
    int                         count_ = 0;

    const char* last_error_ = "";
    CUcontext   context_    = nullptr;
    // Remembered so the registrations can be torn down on the same thread that
    // made them, while the context is still alive.
    std::weak_ptr<accelerator::ogl::device> gl_device_;

    /// Makes the encoder's CUDA context current for as long as it is alive.
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
                CUcontext prev = nullptr;
                cuCtxPopCurrent(&prev);
            }
        }
    };

    void unregister_all()
    {
        if (count_ == 0)
            return;
        scoped_context              ctx(context_);
        std::lock_guard<std::mutex> lk(caspar::cuda_gl_interop_mutex());
        for (int i = 0; i < count_; ++i) {
            if (slots_[i].resource)
                cudaGraphicsUnregisterResource(slots_[i].resource);
        }
        count_ = 0;
    }

    ~impl()
    {
        // By now release() should already have run on the GL thread. If it did
        // not -- an unexpected teardown order -- dropping the registrations is
        // still better than unregistering from the wrong thread with a dead
        // context, which is an access violation rather than a leak.
        count_ = 0;
    }

    cudaGraphicsResource_t find_or_register(int gl_id)
    {
        // Registration binds to the current context, so it has to happen inside
        // the same one the copies and the encoder use.
        for (int i = 0; i < count_; ++i) {
            if (slots_[i].gl_id == gl_id)
                return slots_[i].resource;
        }

        // Serialised process-wide: the driver's GL interop layer is not
        // thread-safe across register/unregister even for distinct textures, and
        // getting that wrong crashes inside nvoglv64.dll. See
        // cuda_gl_interop_lock.h -- this is why that mutex exists.
        std::lock_guard<std::mutex> lk(caspar::cuda_gl_interop_mutex());

        if (count_ >= max_slots) {
            cudaGraphicsUnregisterResource(slots_[0].resource);
            for (int i = 1; i < count_; ++i)
                slots_[i - 1] = slots_[i];
            count_--;
        }

        auto& s  = slots_[count_];
        s        = {};
        s.gl_id  = gl_id;
        auto err = cudaGraphicsGLRegisterImage(
            &s.resource, static_cast<GLuint>(gl_id), GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly);
        if (err != cudaSuccess) {
            last_error_ = cudaGetErrorString(err);
            s.resource  = nullptr;
            return nullptr;
        }
        count_++;
        return s.resource;
    }
};

cuda_gl_uploader::cuda_gl_uploader()
    : impl_(std::make_unique<impl>())
{
}

cuda_gl_uploader::~cuda_gl_uploader() = default;

bool cuda_gl_uploader::available()
{
    int devices = 0;
    return cudaGetDeviceCount(&devices) == cudaSuccess && devices > 0;
}

const char* cuda_gl_uploader::last_error() const { return impl_->last_error_; }

void cuda_gl_uploader::set_context(void* cu_context) { impl_->context_ = static_cast<CUcontext>(cu_context); }

void cuda_gl_uploader::release()
{
    if (auto dev = impl_->gl_device_.lock()) {
        dev->dispatch_sync([&] { impl_->unregister_all(); });
    }
}

bool cuda_gl_uploader::copy_to_device(accelerator::ogl::texture& tex, void* dst, size_t dst_pitch)
{
    if (impl_->gl_device_.expired()) {
        if (auto dev = tex.get_device())
            impl_->gl_device_ = dev;
    }

    impl::scoped_context ctx(impl_->context_);

    auto resource = impl_->find_or_register(tex.id());
    if (!resource)
        return false;

    auto err = cudaGraphicsMapResources(1, &resource, nullptr);
    if (err != cudaSuccess) {
        impl_->last_error_ = cudaGetErrorString(err);
        return false;
    }

    cudaArray_t array = nullptr;
    err               = cudaGraphicsSubResourceGetMappedArray(&array, resource, 0, 0);
    if (err == cudaSuccess) {
        // The mixer's target texture is GL_RGBA8, so a texel is four bytes in
        // R,G,B,A order and this is a straight byte copy -- which is why the
        // frames context declares AV_PIX_FMT_RGB0 and no conversion is needed.
        err = cudaMemcpy2DFromArray(dst,
                                    dst_pitch,
                                    array,
                                    0,
                                    0,
                                    static_cast<size_t>(tex.width()) * 4,
                                    static_cast<size_t>(tex.height()),
                                    cudaMemcpyDeviceToDevice);
    }
    if (err != cudaSuccess)
        impl_->last_error_ = cudaGetErrorString(err);

    cudaGraphicsUnmapResources(1, &resource, nullptr);

    if (err != cudaSuccess)
        return false;

    // The copy is issued on the default stream; NVENC reads the buffer from its
    // own stream, so it has to be complete before the frame is handed over.
    auto sync = cudaStreamSynchronize(nullptr);
    if (sync != cudaSuccess) {
        impl_->last_error_ = cudaGetErrorString(sync);
        return false;
    }
    return true;
}

}} // namespace caspar::ffmpeg

#else // !CASPAR_FFMPEG_HAS_CUDA

namespace caspar { namespace ffmpeg {

struct cuda_gl_uploader::impl
{
};

cuda_gl_uploader::cuda_gl_uploader()  = default;
cuda_gl_uploader::~cuda_gl_uploader() = default;

bool        cuda_gl_uploader::available() { return false; }
void        cuda_gl_uploader::set_context(void*) {}
void        cuda_gl_uploader::release() {}
const char* cuda_gl_uploader::last_error() const { return "built without CUDA"; }
bool cuda_gl_uploader::copy_to_device(accelerator::ogl::texture&, void*, size_t) { return false; }

}} // namespace caspar::ffmpeg

#endif
