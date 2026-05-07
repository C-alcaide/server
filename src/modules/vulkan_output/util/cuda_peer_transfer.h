/*
 * Copyright (c) 2026 CasparCG Contributors
 *
 * CasparCG is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#pragma once

#ifdef CASPAR_CUDA_PEER_ENABLED

#include <GL/glew.h>

#include <cstdint>

// Forward-declare CUDA types to avoid pulling cuda headers into every TU
struct cudaGraphicsResource;
typedef struct cudaGraphicsResource* cudaGraphicsResource_t;
typedef struct CUstream_st*          cudaStream_t;

namespace caspar { namespace vulkan_output {

/**
 * GPU-to-GPU transfer using CUDA peer copy (PCIe DMA, no CPU memcpy).
 *
 * Transfers an OGL texture from GPU A (mixer) to a GL texture on GPU B (output)
 * using cudaMemcpyPeerAsync which leverages the GPU copy engines for direct
 * device-to-device DMA over PCIe/NVLink.
 *
 * Usage from the consumer present path:
 *   // Phase 1: on OGL device thread (GPU A context current)
 *   ogl_device_->dispatch_sync([&] {
 *       cuda_peer_->read_source(ogl_tex_id, w, h);
 *   });
 *
 *   // Phase 2: peer DMA (no GL context needed)
 *   cuda_peer_->peer_copy();
 *
 *   // Phase 3: on affinity thread (GPU B context current)
 *   affinity_ctx_->dispatch_sync([&] {
 *       cuda_peer_->write_dest();
 *       shared_pool_->blit_from_texture(cuda_peer_->dest_texture(), w, h);
 *       shared_pool_->signal_gl();
 *   });
 */
class cuda_peer_transfer
{
  public:
    /// src_cuda_device / dst_cuda_device: CUDA device indices (from cudaGLGetDevices)
    /// width, height: frame dimensions
    /// use_16bit: true for BGRA16 (HDR), false for BGRA8
    cuda_peer_transfer(int src_cuda_device, int dst_cuda_device,
                       uint32_t width, uint32_t height, bool use_16bit);
    ~cuda_peer_transfer();

    cuda_peer_transfer(const cuda_peer_transfer&)            = delete;
    cuda_peer_transfer& operator=(const cuda_peer_transfer&) = delete;

    /// Phase 1: Read source OGL texture into GPU A staging buffer.
    /// MUST be called on a thread with GPU A's GL context current.
    /// If texture_id differs from previously registered, re-registers.
    void read_source(GLuint texture_id, int width, int height);

    /// Phase 2: DMA from GPU A staging → GPU B staging via peer copy.
    /// Can be called from any thread (no GL context required).
    void peer_copy();

    /// Phase 3: Write GPU B staging buffer into the landing texture.
    /// MUST be called on a thread with GPU B's GL context current.
    void write_dest();

    /// Returns the GL texture ID on GPU B that received the frame.
    /// Valid after write_dest() completes. Use with blit_from_texture().
    GLuint dest_texture() const { return dest_gl_texture_; }

    /// Query whether CUDA peer access is available between the two devices.
    static bool is_peer_access_available(int src_device, int dst_device);

    /// Select the CUDA device associated with the current GL context.
    /// Must be called from a thread with an active GL context.
    static int cuda_device_for_current_gl_context();

  private:
    void ensure_source_registered(GLuint texture_id);
    void unregister_source();
    void ensure_dest_registered();
    void unregister_dest();

    int      src_device_;
    int      dst_device_;
    uint32_t width_;
    uint32_t height_;
    bool     use_16bit_;
    size_t   pixel_size_;    // 4 or 8 bytes
    size_t   row_bytes_;     // width * pixel_size
    size_t   total_bytes_;   // width * height * pixel_size

    // Staging buffers (linear device memory)
    void* src_staging_ = nullptr; // on src_device_
    void* dst_staging_ = nullptr; // on dst_device_

    // CUDA streams (one per device)
    cudaStream_t src_stream_ = nullptr;
    cudaStream_t dst_stream_ = nullptr;

    // Source side (GPU A)
    cudaGraphicsResource_t src_resource_   = nullptr;
    GLuint                 src_texture_id_ = 0; // currently registered texture

    // Destination side (GPU B) — persistent landing texture
    GLuint                 dest_gl_texture_ = 0;
    cudaGraphicsResource_t dst_resource_    = nullptr;

    bool peer_access_enabled_ = false;
};

}} // namespace caspar::vulkan_output

#endif // CASPAR_CUDA_PEER_ENABLED
