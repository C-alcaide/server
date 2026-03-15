// cuda_gl_texture.h
// Thin RAII wrapper that registers an OGL texture with CUDA for zero-copy
// device→GL writes.
//
// Usage:
//   // On the GL dispatch thread (via device->dispatch_sync):
//   auto tex = ogl_device->create_texture(w, h, 4, common::bit_depth::bit16);
//
//   // Outside GL thread (CUDA thread / producer thread):
//   CudaGLTexture cgt(tex);           // registers with CUDA once
//   cudaArray_t arr = cgt.map(stream);
//   cudaMemcpy2DToArrayAsync(arr, 0, 0, d_bgra16, pitch, w*8, h,
//                            cudaMemcpyDeviceToDevice, stream);
//   cgt.unmap(stream);
//   cudaStreamSynchronize(stream);    // wait for copy before GL consumer
//
// Thread-safety: map/unmap must be called from the same CUDA stream.
// A single CudaGLTexture instance must NOT be mapped concurrently.
//
// Lifetime: CudaGLTexture MUST be destroyed before the underlying ogl::texture
// is released (i.e. before the shared_ptr ref-count drops to zero).
// ---------------------------------------------------------------------------
#pragma once

#include <accelerator/ogl/util/texture.h>

// glew.h must be included before any other GL headers (including cuda_gl_interop.h
// which pulls in <GL/gl.h> on Windows). Include it explicitly here.
#ifdef WIN32
#include <GL/glew.h>
#endif
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include <common/log.h>

#include <stdexcept>
#include <string>

namespace caspar { namespace cuda_prores {

// Check a CUDA runtime call and throw on failure.
inline void cuda_check(cudaError_t e, const char* what)
{
    if (e != cudaSuccess) {
        std::string msg = std::string(what) + ": " + cudaGetErrorString(e);
        CASPAR_LOG(error) << L"[cuda_gl_texture] " << msg.c_str();
        throw std::runtime_error(msg);
    }
}

// ---------------------------------------------------------------------------
// Selects the CUDA device that is interoperable with the current GL context.
// Returns the device index, or 0 if none found (fallback).
// Must be called from a thread that has an active GL context.
// ---------------------------------------------------------------------------
inline int select_cuda_gl_device()
{
    unsigned int  cuda_dev_count = 0;
    int           cuda_devices[8];
    cudaError_t   err = cudaGLGetDevices(&cuda_dev_count, cuda_devices, 8,
                                         cudaGLDeviceListAll);
    if (err != cudaSuccess || cuda_dev_count == 0) {
        CASPAR_LOG(warning) << L"[cuda_gl_texture] cudaGLGetDevices failed ("
                            << cudaGetErrorString(err) << L") — using device 0";
        return 0;
    }
    CASPAR_LOG(info) << L"[cuda_gl_texture] GL-interoperable CUDA device: "
                     << cuda_devices[0];
    return cuda_devices[0];
}

// ---------------------------------------------------------------------------
// RAII wrapper for a CUDA-registered GL texture.
// ---------------------------------------------------------------------------
class CudaGLTexture
{
  public:
    // Register texture with CUDA.  gl_tex must remain valid for the lifetime
    // of this object.  Must be called from a thread on which the matching
    // CUDA device is current (cudaSetDevice call not required here — the
    // interop device is the one used for all subsequent map/unmap calls).
    explicit CudaGLTexture(std::shared_ptr<accelerator::ogl::texture> gl_tex)
        : gl_tex_(std::move(gl_tex))
        , resource_(nullptr)
        , mapped_(false)
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
            // Best-effort unmap on destruction — should not normally happen.
            cudaGraphicsUnmapResources(1, &resource_, nullptr);
            mapped_ = false;
        }
        if (resource_)
            cudaGraphicsUnregisterResource(resource_);
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

    // Map the GL texture into CUDA address space and return a cudaArray_t.
    // The array is valid until unmap() is called.
    // stream: the CUDA stream that will write into the array.
    cudaArray_t map(cudaStream_t stream)
    {
        if (mapped_)
            throw std::logic_error("CudaGLTexture already mapped");
        cuda_check(
            cudaGraphicsMapResources(1, &resource_, stream),
            "cudaGraphicsMapResources");
        mapped_ = true;

        cudaArray_t array = nullptr;
        cuda_check(
            cudaGraphicsSubResourceGetMappedArray(&array, resource_, 0, 0),
            "cudaGraphicsSubResourceGetMappedArray");
        return array;
    }

    // Unmap the GL texture.  Must be called after all CUDA writes to the
    // array are submitted (on the same stream as map).
    void unmap(cudaStream_t stream)
    {
        if (!mapped_)
            return;
        cuda_check(
            cudaGraphicsUnmapResources(1, &resource_, stream),
            "cudaGraphicsUnmapResources");
        mapped_ = false;
    }

    const std::shared_ptr<accelerator::ogl::texture>& gl_texture() const
    {
        return gl_tex_;
    }

  private:
    std::shared_ptr<accelerator::ogl::texture> gl_tex_;
    cudaGraphicsResource_t                     resource_;
    bool                                       mapped_;
};

}} // namespace caspar::cuda_prores
