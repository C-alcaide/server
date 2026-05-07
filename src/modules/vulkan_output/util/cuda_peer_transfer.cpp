/*
 * Copyright (c) 2026 CasparCG Contributors
 *
 * CasparCG is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#include "cuda_peer_transfer.h"

#ifdef CASPAR_CUDA_PEER_ENABLED

#include <common/log.h>
#include <common/except.h>

#include <GL/glew.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <stdexcept>
#include <string>

namespace caspar { namespace vulkan_output {

namespace {

void cuda_check(cudaError_t err, const char* context)
{
    if (err != cudaSuccess) {
        auto msg = std::string(context) + ": " + cudaGetErrorString(err);
        CASPAR_LOG(error) << L"[cuda_peer_transfer] " << msg.c_str();
        CASPAR_THROW_EXCEPTION(caspar_exception() << msg_info(msg));
    }
}

} // anonymous namespace

// ─── Static helpers ─────────────────────────────────────────────────────────

bool cuda_peer_transfer::is_peer_access_available(int src_device, int dst_device)
{
    int can_access = 0;
    cudaError_t err = cudaDeviceCanAccessPeer(&can_access, src_device, dst_device);
    if (err != cudaSuccess) {
        CASPAR_LOG(debug) << L"[cuda_peer_transfer] cudaDeviceCanAccessPeer failed: "
                          << cudaGetErrorString(err);
        return false;
    }
    return can_access != 0;
}

int cuda_peer_transfer::cuda_device_for_current_gl_context()
{
    unsigned int count = 0;
    int          devices[8];
    cudaError_t  err = cudaGLGetDevices(&count, devices, 8, cudaGLDeviceListCurrentFrame);
    if (err != cudaSuccess || count == 0) {
        // Fallback: try all devices
        err = cudaGLGetDevices(&count, devices, 8, cudaGLDeviceListAll);
        if (err != cudaSuccess || count == 0) {
            CASPAR_LOG(warning) << L"[cuda_peer_transfer] cudaGLGetDevices failed: "
                                << cudaGetErrorString(err);
            return -1;
        }
    }
    return devices[0];
}

// ─── Constructor / Destructor ───────────────────────────────────────────────

cuda_peer_transfer::cuda_peer_transfer(int src_cuda_device, int dst_cuda_device,
                                       uint32_t width, uint32_t height, bool use_16bit)
    : src_device_(src_cuda_device)
    , dst_device_(dst_cuda_device)
    , width_(width)
    , height_(height)
    , use_16bit_(use_16bit)
    , pixel_size_(use_16bit ? 8 : 4)
    , row_bytes_(width * (use_16bit ? 8 : 4))
    , total_bytes_(static_cast<size_t>(width) * height * (use_16bit ? 8 : 4))
{
    CASPAR_LOG(info) << L"[cuda_peer_transfer] Initializing peer transfer: device "
                     << src_device_ << L" → device " << dst_device_
                     << L" (" << width_ << L"x" << height_
                     << (use_16bit_ ? L" 16-bit" : L" 8-bit") << L")";

    // Enable peer access if hardware supports it (NVLink / same IOH)
    // Even without peer access, cudaMemcpyPeer still works — it just goes
    // through system memory internally. But enabling gives direct DMA.
    if (is_peer_access_available(src_device_, dst_device_)) {
        cudaSetDevice(src_device_);
        cudaError_t err = cudaDeviceEnablePeerAccess(dst_device_, 0);
        if (err == cudaSuccess || err == cudaErrorPeerAccessAlreadyEnabled) {
            peer_access_enabled_ = true;
            CASPAR_LOG(info) << L"[cuda_peer_transfer] Direct peer access enabled (NVLink/PCIe P2P)";
        }

        cudaSetDevice(dst_device_);
        err = cudaDeviceEnablePeerAccess(src_device_, 0);
        if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled) {
            CASPAR_LOG(debug) << L"[cuda_peer_transfer] Reverse peer access not available";
        }
    } else {
        CASPAR_LOG(info) << L"[cuda_peer_transfer] No direct P2P — peer copy will stage through system RAM "
                         << L"(still faster than CPU memcpy)";
    }

    // Allocate staging buffer on source device
    cudaSetDevice(src_device_);
    cuda_check(cudaMalloc(&src_staging_, total_bytes_), "cudaMalloc src_staging");
    cuda_check(cudaStreamCreate(&src_stream_), "cudaStreamCreate src");

    // Allocate staging buffer on destination device
    cudaSetDevice(dst_device_);
    cuda_check(cudaMalloc(&dst_staging_, total_bytes_), "cudaMalloc dst_staging");
    cuda_check(cudaStreamCreate(&dst_stream_), "cudaStreamCreate dst");

    CASPAR_LOG(info) << L"[cuda_peer_transfer] Staging buffers allocated ("
                     << (total_bytes_ / 1024 / 1024) << L" MB each)";
}

cuda_peer_transfer::~cuda_peer_transfer()
{
    // Unregister GL resources
    if (src_resource_) {
        cudaSetDevice(src_device_);
        cudaGraphicsUnregisterResource(src_resource_);
    }
    if (dst_resource_) {
        cudaSetDevice(dst_device_);
        cudaGraphicsUnregisterResource(dst_resource_);
    }

    // Delete destination GL texture (requires GPU B's GL context to be current)
    if (dest_gl_texture_) {
        if (wglGetCurrentContext() != nullptr) {
            glDeleteTextures(1, &dest_gl_texture_);
        }
        // If no GL context is current, the texture will be freed when the
        // owning GL context is destroyed during shutdown.
    }

    // Free staging buffers
    if (src_staging_) {
        cudaSetDevice(src_device_);
        cudaFree(src_staging_);
    }
    if (dst_staging_) {
        cudaSetDevice(dst_device_);
        cudaFree(dst_staging_);
    }

    // Destroy streams
    if (src_stream_) {
        cudaSetDevice(src_device_);
        cudaStreamDestroy(src_stream_);
    }
    if (dst_stream_) {
        cudaSetDevice(dst_device_);
        cudaStreamDestroy(dst_stream_);
    }

    // Disable peer access
    if (peer_access_enabled_) {
        cudaSetDevice(src_device_);
        cudaDeviceDisablePeerAccess(dst_device_);
        cudaSetDevice(dst_device_);
        cudaDeviceDisablePeerAccess(src_device_);
    }
}

// ─── Source registration ────────────────────────────────────────────────────

void cuda_peer_transfer::ensure_source_registered(GLuint texture_id)
{
    if (src_resource_ && src_texture_id_ == texture_id) {
        return; // Already registered
    }

    unregister_source();

    cudaSetDevice(src_device_);
    cuda_check(
        cudaGraphicsGLRegisterImage(&src_resource_, texture_id, GL_TEXTURE_2D,
                                    cudaGraphicsRegisterFlagsReadOnly),
        "cudaGraphicsGLRegisterImage (source)");
    src_texture_id_ = texture_id;
}

void cuda_peer_transfer::unregister_source()
{
    if (src_resource_) {
        cudaSetDevice(src_device_);
        cudaGraphicsUnregisterResource(src_resource_);
        src_resource_   = nullptr;
        src_texture_id_ = 0;
    }
}

// ─── Destination registration ───────────────────────────────────────────────

void cuda_peer_transfer::ensure_dest_registered()
{
    if (dst_resource_) {
        return; // Already registered
    }

    // Create the landing texture on GPU B if not yet created
    if (!dest_gl_texture_) {
        glGenTextures(1, &dest_gl_texture_);
        glBindTexture(GL_TEXTURE_2D, dest_gl_texture_);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

        GLenum internal_format = use_16bit_ ? GL_RGBA16 : GL_RGBA8;
        GLenum pixel_format    = GL_BGRA;
        GLenum pixel_type      = use_16bit_ ? GL_UNSIGNED_SHORT : GL_UNSIGNED_BYTE;
        glTexImage2D(GL_TEXTURE_2D, 0, internal_format, width_, height_, 0,
                     pixel_format, pixel_type, nullptr);
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    cudaSetDevice(dst_device_);
    cuda_check(
        cudaGraphicsGLRegisterImage(&dst_resource_, dest_gl_texture_, GL_TEXTURE_2D,
                                    cudaGraphicsRegisterFlagsWriteDiscard),
        "cudaGraphicsGLRegisterImage (dest)");
}

void cuda_peer_transfer::unregister_dest()
{
    if (dst_resource_) {
        cudaSetDevice(dst_device_);
        cudaGraphicsUnregisterResource(dst_resource_);
        dst_resource_ = nullptr;
    }
}

// ─── Phase 1: Read source texture → GPU A staging ───────────────────────────

void cuda_peer_transfer::read_source(GLuint texture_id, int width, int height)
{
    // Must be called on OGL device thread (GPU A GL context current)
    ensure_source_registered(texture_id);

    cudaSetDevice(src_device_);

    // Map the GL texture for CUDA read
    cuda_check(
        cudaGraphicsMapResources(1, &src_resource_, src_stream_),
        "cudaGraphicsMapResources (source)");

    cudaArray_t src_array = nullptr;
    cuda_check(
        cudaGraphicsSubResourceGetMappedArray(&src_array, src_resource_, 0, 0),
        "cudaGraphicsSubResourceGetMappedArray");

    // Copy from 2D array → linear staging buffer
    cuda_check(
        cudaMemcpy2DFromArrayAsync(
            src_staging_,          // dst (linear)
            row_bytes_,            // dst pitch
            src_array,             // src (cudaArray)
            0, 0,                  // src offset (x, y)
            row_bytes_,            // width in bytes
            height_,               // height
            cudaMemcpyDeviceToDevice,
            src_stream_),
        "cudaMemcpy2DFromArrayAsync");

    // Unmap so GL can use the texture again
    cuda_check(
        cudaGraphicsUnmapResources(1, &src_resource_, src_stream_),
        "cudaGraphicsUnmapResources (source)");

    // Wait for the copy to complete before peer_copy() reads src_staging
    cuda_check(cudaStreamSynchronize(src_stream_), "cudaStreamSynchronize (source)");
}

// ─── Phase 2: Peer DMA from GPU A staging → GPU B staging ──────────────────

void cuda_peer_transfer::peer_copy()
{
    // No GL context required — just device-to-device DMA
    // TODO(perf): This is synchronous and blocks the present thread for the duration
    // of the DMA transfer. Consider using cudaMemcpyPeerAsync + stream + event fence
    // to overlap the transfer with other present work (double-buffer staging).
    cuda_check(
        cudaMemcpyPeer(dst_staging_, dst_device_, src_staging_, src_device_, total_bytes_),
        "cudaMemcpyPeer");
}

// ─── Phase 3: Write GPU B staging → destination texture ─────────────────────

void cuda_peer_transfer::write_dest()
{
    // Must be called on affinity thread (GPU B GL context current)
    ensure_dest_registered();

    cudaSetDevice(dst_device_);

    // Map the destination GL texture for CUDA write
    cuda_check(
        cudaGraphicsMapResources(1, &dst_resource_, dst_stream_),
        "cudaGraphicsMapResources (dest)");

    cudaArray_t dst_array = nullptr;
    cuda_check(
        cudaGraphicsSubResourceGetMappedArray(&dst_array, dst_resource_, 0, 0),
        "cudaGraphicsSubResourceGetMappedArray (dest)");

    // Copy from linear staging → 2D array
    cuda_check(
        cudaMemcpy2DToArrayAsync(
            dst_array,             // dst (cudaArray)
            0, 0,                  // dst offset (x, y)
            dst_staging_,          // src (linear)
            row_bytes_,            // src pitch
            row_bytes_,            // width in bytes
            height_,               // height
            cudaMemcpyDeviceToDevice,
            dst_stream_),
        "cudaMemcpy2DToArrayAsync");

    // Unmap — makes the texture available to GL again
    cuda_check(
        cudaGraphicsUnmapResources(1, &dst_resource_, dst_stream_),
        "cudaGraphicsUnmapResources (dest)");

    // Synchronize so GL sees the completed write
    cuda_check(cudaStreamSynchronize(dst_stream_), "cudaStreamSynchronize (dest)");
}

}} // namespace caspar::vulkan_output

#endif // CASPAR_CUDA_PEER_ENABLED
