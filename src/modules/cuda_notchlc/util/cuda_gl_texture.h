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
 * NotchLC is a codec specification by Derivative Inc., available under the
 * Creative Commons Attribution 4.0 International License.
 */

// util/cuda_gl_texture.h
// Thin RAII wrapper that registers an OGL texture with CUDA for zero-copy
// device→GL writes.  Identical to cuda_prores/util/cuda_gl_texture.h except
// the namespace is caspar::cuda_notchlc.
#pragma once

#include <accelerator/ogl/util/texture.h>

#ifdef WIN32
#include <GL/glew.h>
#endif
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include <common/log.h>

#include <stdexcept>
#include <string>

namespace caspar { namespace cuda_notchlc {

inline void cuda_check(cudaError_t e, const char* what)
{
    if (e != cudaSuccess) {
        std::string msg = std::string(what) + ": " + cudaGetErrorString(e);
        CASPAR_LOG(error) << L"[cuda_gl_texture] " << msg.c_str();
        throw std::runtime_error(msg);
    }
}

inline int select_cuda_gl_device()
{
    unsigned int cuda_dev_count = 0;
    int          cuda_devices[8];
    cudaError_t  err = cudaGLGetDevices(&cuda_dev_count, cuda_devices, 8, cudaGLDeviceListAll);
    if (err != cudaSuccess || cuda_dev_count == 0) {
        CASPAR_LOG(warning) << L"[cuda_notchlc] cudaGLGetDevices failed ("
                            << cudaGetErrorString(err) << L") -- using device 0";
        return 0;
    }
    CASPAR_LOG(info) << L"[cuda_notchlc] GL-interoperable CUDA device: " << cuda_devices[0];
    return cuda_devices[0];
}

class CudaGLTexture
{
  public:
    explicit CudaGLTexture(std::shared_ptr<accelerator::ogl::texture> gl_tex)
        : gl_tex_(std::move(gl_tex)), resource_(nullptr), mapped_(false)
    {
        cuda_check(
            cudaGraphicsGLRegisterImage(
                &resource_,
                static_cast<GLuint>(gl_tex_->id()),
                GL_TEXTURE_2D,
                cudaGraphicsRegisterFlagsWriteDiscard),
            "cudaGraphicsGLRegisterImage");
    }

    ~CudaGLTexture()
    {
        if (mapped_) {
            CASPAR_LOG(warning) << L"[cuda_notchlc] CudaGLTexture destroyed while still mapped";
            cudaGraphicsUnmapResources(1, &resource_, nullptr);
        }
        if (resource_)
            cudaGraphicsUnregisterResource(resource_);
    }

    CudaGLTexture(const CudaGLTexture&)            = delete;
    CudaGLTexture& operator=(const CudaGLTexture&) = delete;

    cudaArray_t map(cudaStream_t stream)
    {
        cuda_check(cudaGraphicsMapResources(1, &resource_, stream), "cudaGraphicsMapResources");
        mapped_ = true;
        cudaArray_t arr = nullptr;
        cuda_check(cudaGraphicsSubResourceGetMappedArray(&arr, resource_, 0, 0),
                   "cudaGraphicsSubResourceGetMappedArray");
        return arr;
    }

    void unmap(cudaStream_t stream)
    {
        if (!mapped_) return;
        cudaGraphicsUnmapResources(1, &resource_, stream);
        mapped_ = false;
    }

    std::shared_ptr<accelerator::ogl::texture> gl_texture() const { return gl_tex_; }

  private:
    std::shared_ptr<accelerator::ogl::texture> gl_tex_;
    cudaGraphicsResource_t                     resource_;
    bool                                       mapped_;
};

}} // namespace caspar::cuda_notchlc
