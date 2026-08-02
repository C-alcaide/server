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

#pragma once

/**
 * CUDA <-> OpenGL zero-copy: a mixer texture a CUDA kernel can write into directly.
 *
 * The OpenGL counterpart of cuda_vk_texture.h. Both let a CUDA producer hand the mixer
 * a frame without a host round trip; which applies is decided by
 * frame_factory::gpu_device_backend().
 *
 * ── Where this came from ─────────────────────────────────────────────────────
 * cuda_prores and cuda_notchlc each carry a private copy of this class and the two have
 * drifted: one has a move constructor, throws on a double map and checks the unmap; the
 * other warns instead of throwing and discards the unmap's error. This is the canonical
 * version -- the stricter behaviour -- for new callers. The existing pair are left
 * alone deliberately: merging them changes one decoder's behaviour under GPU error
 * conditions, which wants ProRes and NotchLC content to validate rather than a diff.
 *
 * ── Two things that are not obvious ──────────────────────────────────────────
 * cudaGraphicsGLRegisterImage and cudaGraphicsUnregisterResource are not thread-safe
 * against each other, even for distinct textures; two producers swapping on one layer
 * call them concurrently and the driver dereferences freed memory. Hold
 * caspar::cuda_gl_interop_mutex() across both -- this class does. See
 * BUILDING_WORKFLOW.md #4.
 *
 * And every call here -- construction, map, unmap, destruction -- must run on the
 * thread whose GL context owns the texture. Registering on the GL thread and mapping
 * from elsewhere fails with "invalid OpenGL or DirectX context".
 */

#include "cuda_gl_interop_lock.h"

#include <accelerator/ogl/util/texture.h>

// glew.h must come before any other GL header, including cuda_gl_interop.h, which
// pulls in <GL/gl.h> on Windows.
#ifdef WIN32
#include <GL/glew.h>
#else
#include <GL/gl.h>
#endif
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include <common/log.h>

#include <memory>
#include <stdexcept>
#include <string>

namespace caspar {

inline void cuda_gl_check(cudaError_t e, const char* what)
{
    if (e != cudaSuccess) {
        std::string msg = std::string(what) + ": " + cudaGetErrorString(e);
        CASPAR_LOG(error) << L"[cuda_gl_texture] " << msg.c_str();
        throw std::runtime_error(msg);
    }
}

/// The CUDA device interoperable with the *current* GL context, or 0 if that cannot be
/// determined. Needs a current GL context.
///
/// This matters on a multi-GPU machine: CUDA's default device is not necessarily the
/// one the mixer's context lives on, and mapping a GL texture from the wrong device
/// fails rather than degrading.
inline int select_cuda_gl_device()
{
    unsigned int count = 0;
    int          devices[8];
    cudaError_t  err = cudaGLGetDevices(&count, devices, 8, cudaGLDeviceListAll);
    if (err != cudaSuccess || count == 0) {
        CASPAR_LOG(warning) << L"[cuda_gl_texture] cudaGLGetDevices failed (" << cudaGetErrorString(err)
                            << L") - using device 0";
        return 0;
    }
    return devices[0];
}

/// RAII registration of a mixer GL texture with CUDA. `gl_tex` must outlive this
/// object: the registration aliases it.
class CudaGLTexture
{
  public:
    explicit CudaGLTexture(std::shared_ptr<accelerator::ogl::texture> gl_tex)
        : gl_tex_(std::move(gl_tex))
    {
        std::lock_guard<std::mutex> lock(cuda_gl_interop_mutex());
        cuda_gl_check(cudaGraphicsGLRegisterImage(&resource_,
                                                  static_cast<GLuint>(gl_tex_->id()),
                                                  GL_TEXTURE_2D,
                                                  cudaGraphicsRegisterFlagsWriteDiscard),
                      "cudaGraphicsGLRegisterImage");
    }

    ~CudaGLTexture()
    {
        if (mapped_) {
            cudaGraphicsUnmapResources(1, &resource_, nullptr);
            mapped_ = false;
        }
        if (resource_) {
            std::lock_guard<std::mutex> lock(cuda_gl_interop_mutex());
            cudaGraphicsUnregisterResource(resource_);
        }
    }

    CudaGLTexture(const CudaGLTexture&)            = delete;
    CudaGLTexture& operator=(const CudaGLTexture&) = delete;

    CudaGLTexture(CudaGLTexture&& o) noexcept
        : gl_tex_(std::move(o.gl_tex_))
        , resource_(o.resource_)
        , mapped_(o.mapped_)
    {
        o.resource_ = nullptr;
        o.mapped_   = false;
    }

    /// Maps into CUDA and returns the array to write. Valid until unmap().
    cudaArray_t map(cudaStream_t stream = nullptr)
    {
        if (mapped_)
            throw std::logic_error("CudaGLTexture already mapped");
        cuda_gl_check(cudaGraphicsMapResources(1, &resource_, stream), "cudaGraphicsMapResources");
        mapped_ = true;

        cudaArray_t array = nullptr;
        cuda_gl_check(cudaGraphicsSubResourceGetMappedArray(&array, resource_, 0, 0),
                      "cudaGraphicsSubResourceGetMappedArray");
        return array;
    }

    /// Unmaps, after the writes are submitted, on the same stream as map().
    void unmap(cudaStream_t stream = nullptr)
    {
        if (!mapped_)
            return;
        cuda_gl_check(cudaGraphicsUnmapResources(1, &resource_, stream), "cudaGraphicsUnmapResources");
        mapped_ = false;
    }

    const std::shared_ptr<accelerator::ogl::texture>& gl_texture() const { return gl_tex_; }

    /// True when nothing downstream still holds the texture, so this slot can be
    /// reused. Mirrors CudaVkTexture::is_free(): the mixer keeps a frame's texture
    /// after the producer has moved on, so a pool must not hand one back too early.
    bool is_free() const { return gl_tex_.use_count() == 1 && !mapped_; }

    std::shared_ptr<core::texture> core_texture() const
    {
        return std::static_pointer_cast<core::texture>(gl_tex_);
    }

  private:
    std::shared_ptr<accelerator::ogl::texture> gl_tex_;
    cudaGraphicsResource_t                     resource_ = nullptr;
    bool                                       mapped_   = false;
};

} // namespace caspar
