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
#ifdef _WIN32
#include <GL/wglew.h>
#else
#include <EGL/egl.h>
#endif
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
                     << src_device_ << L" -> device " << dst_device_
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
        }

        cudaSetDevice(dst_device_);
        err = cudaDeviceEnablePeerAccess(src_device_, 0);
        if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled) {
            CASPAR_LOG(debug) << L"[cuda_peer_transfer] Reverse peer access not available";
        }

        // Query P2P link type and performance attributes
        int nvlink_supported = 0;
        cudaDeviceGetP2PAttribute(&nvlink_supported,
                                  cudaDevP2PAttrNativeAtomicSupported,
                                  src_device_, dst_device_);

        int perf_rank = 0;
        cudaDeviceGetP2PAttribute(&perf_rank,
                                  cudaDevP2PAttrPerformanceRank,
                                  src_device_, dst_device_);

        // NVLink: native atomic support + high perf rank
        // PCIe P2P: peer access enabled but no native atomics
        if (nvlink_supported) {
            CASPAR_LOG(info) << L"[cuda_peer_transfer] NVLink detected between device "
                             << src_device_ << L" and device " << dst_device_
                             << L" (perf_rank=" << perf_rank << L")";
            CASPAR_LOG(info) << L"[cuda_peer_transfer] NVLink provides ~600 GB/s bidirectional bandwidth";
        } else {
            CASPAR_LOG(info) << L"[cuda_peer_transfer] PCIe P2P direct access between device "
                             << src_device_ << L" and device " << dst_device_
                             << L" (perf_rank=" << perf_rank
                             << L", no NVLink - limited to PCIe bandwidth)";
        }
    } else {
        CASPAR_LOG(info) << L"[cuda_peer_transfer] No direct P2P - peer copy will stage through system RAM "
                         << L"(still faster than CPU memcpy)";
    }

    // Allocate staging buffer on source device
    cudaSetDevice(src_device_);
    cuda_check(cudaMalloc(&src_staging_, total_bytes_), "cudaMalloc src_staging");
    cuda_check(cudaStreamCreate(&src_stream_), "cudaStreamCreate src");
    cuda_check(cudaStreamCreate(&peer_stream_), "cudaStreamCreate peer");
    cuda_check(cudaEventCreateWithFlags(&src_ready_event_, cudaEventDisableTiming),
               "cudaEventCreate src_ready");
    cuda_check(cudaEventCreateWithFlags(&peer_event_, cudaEventDisableTiming),
               "cudaEventCreate peer");

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
    if (dst_pbo_resource_) {
        cudaSetDevice(dst_device_);
        cudaGraphicsUnregisterResource(dst_pbo_resource_);
    }

    // Delete destination GL resources (requires GPU B's GL context to be current)
#ifdef _WIN32
    bool has_gl_context = (wglGetCurrentContext() != nullptr);
#else
    bool has_gl_context = (eglGetCurrentContext() != EGL_NO_CONTEXT);
#endif
    if (has_gl_context) {
        if (dest_pbo_)
            glDeleteBuffers(1, &dest_pbo_);
        if (dest_gl_texture_)
            glDeleteTextures(1, &dest_gl_texture_);
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

    // Destroy streams and events
    if (src_ready_event_) {
        cudaSetDevice(src_device_);
        cudaEventDestroy(src_ready_event_);
    }
    if (peer_event_) {
        cudaSetDevice(src_device_);
        cudaEventDestroy(peer_event_);
    }
    if (peer_stream_) {
        cudaSetDevice(src_device_);
        cudaStreamDestroy(peer_stream_);
    }
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

    // Drain any stale GL errors left by CUDA-GL interop driver internals
    while (glGetError() != GL_NO_ERROR) {}
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

// ─── Destination helpers ────────────────────────────────────────────────────

void cuda_peer_transfer::ensure_dest_texture()
{
    if (dest_gl_texture_)
        return;

    glGenTextures(1, &dest_gl_texture_);
    glBindTexture(GL_TEXTURE_2D, dest_gl_texture_);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    GLenum internal_format = use_16bit_ ? GL_RGBA16 : GL_RGBA8;
    GLenum pixel_format    = GL_RGBA;  // matches CUDA byte order (not GL_BGRA)
    GLenum pixel_type      = use_16bit_ ? GL_UNSIGNED_SHORT : GL_UNSIGNED_BYTE;
    glTexImage2D(GL_TEXTURE_2D, 0, internal_format, width_, height_, 0,
                 pixel_format, pixel_type, nullptr);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void cuda_peer_transfer::ensure_dest_pbo()
{
    if (dest_pbo_)
        return;

    glGenBuffers(1, &dest_pbo_);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, dest_pbo_);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, static_cast<GLsizeiptr>(total_bytes_),
                 nullptr, GL_STREAM_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    cudaSetDevice(dst_device_);
    cuda_check(
        cudaGraphicsGLRegisterBuffer(&dst_pbo_resource_, dest_pbo_,
                                     cudaGraphicsRegisterFlagsWriteDiscard),
        "cudaGraphicsGLRegisterBuffer (dest PBO)");

    CASPAR_LOG(info) << L"[cuda_peer_transfer] Dest PBO created and registered ("
                     << (total_bytes_ / 1024 / 1024) << L" MB)";
}

// ─── Phase 1: Read source texture → GPU A staging ───────────────────────────

void cuda_peer_transfer::read_source(GLuint texture_id, int width, int height)
{
    // Must be called on OGL device thread (GPU A GL context current)
    ensure_source_registered(texture_id);

    cudaSetDevice(src_device_);

    // GPU-side wait: ensure any in-flight peer DMA has finished reading
    // src_staging_ before we overwrite it with the new frame.
    cuda_check(
        cudaStreamWaitEvent(src_stream_, peer_event_, 0),
        "cudaStreamWaitEvent (read_source)");

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

    // Signal that src_staging_ is ready for peer_copy() to read.
    cuda_check(
        cudaEventRecord(src_ready_event_, src_stream_),
        "cudaEventRecord (src_ready)");

    // CUDA-GL interop can leave stale GL error flags (GL_INVALID_OPERATION)
    // from internal driver state changes during cudaGraphicsMapResources /
    // cudaGraphicsUnmapResources.  Drain them so the next GL call (e.g.
    // blit_from_texture on the same OGL executor thread) doesn't trip the
    // GL error check and throw.
    while (glGetError() != GL_NO_ERROR) {}
}

// ─── Phase 2: Peer DMA from GPU A staging → GPU B staging ──────────────────

void cuda_peer_transfer::peer_copy()
{
    // No GL context required — async device-to-device DMA.
    // Runs on a dedicated stream so the calling thread returns immediately.
    // read_source() GPU-waits on peer_event_ before overwriting src_staging_.
    // write_dest()  GPU-waits on peer_event_ before reading dst_staging_.
    cudaSetDevice(src_device_);

    // GPU-wait for src_stream_ to finish writing src_staging_.
    cuda_check(
        cudaStreamWaitEvent(peer_stream_, src_ready_event_, 0),
        "cudaStreamWaitEvent (src_ready)");

    cuda_check(
        cudaMemcpyPeerAsync(dst_staging_, dst_device_,
                            src_staging_, src_device_,
                            total_bytes_, peer_stream_),
        "cudaMemcpyPeerAsync");
    cuda_check(
        cudaEventRecord(peer_event_, peer_stream_),
        "cudaEventRecord (peer_copy)");
}

// ─── Phase 3: Write GPU B staging → destination texture ─────────────────────

void cuda_peer_transfer::write_dest()
{
    // Must be called on affinity thread (GPU B GL context current).
    //
    // Pipeline: dst_staging_ (CUDA linear) → GL PBO (via CUDA) → GL texture.
    // This avoids cudaMemcpy2DToArray into a CUDA-registered GL texture which
    // has tiling/coherency issues on Pascal GPUs (P4000), producing visible
    // horizontal lines.  The PBO path keeps everything on GPU B with no CPU
    // round-trip.
    ensure_dest_texture();
    ensure_dest_pbo();

    cudaSetDevice(dst_device_);

    // GPU-side wait: ensure async peer DMA has finished writing
    // dst_staging_ before we read it into the PBO.
    // NOTE: peer_event_ was created on src_device_, but we wait on it from
    // dst_stream_ (dst_device_).  Cross-device cudaStreamWaitEvent is supported
    // on NVIDIA GPUs with modern drivers (CUDA 11+), but is not formally
    // guaranteed by the CUDA programming guide for all configurations.
    // Tested working on Ampere+Pascal with driver 550+.
    cuda_check(
        cudaStreamWaitEvent(dst_stream_, peer_event_, 0),
        "cudaStreamWaitEvent (write_dest)");

    // Step 1: Map the GL PBO for CUDA write
    cuda_check(
        cudaGraphicsMapResources(1, &dst_pbo_resource_, dst_stream_),
        "cudaGraphicsMapResources (dest PBO)");

    void*  pbo_ptr  = nullptr;
    size_t pbo_size = 0;
    cuda_check(
        cudaGraphicsResourceGetMappedPointer(&pbo_ptr, &pbo_size, dst_pbo_resource_),
        "cudaGraphicsResourceGetMappedPointer (dest PBO)");

    // Step 2: Copy from staging → PBO (device-to-device on GPU B, very fast)
    cuda_check(
        cudaMemcpyAsync(pbo_ptr, dst_staging_, total_bytes_,
                        cudaMemcpyDeviceToDevice, dst_stream_),
        "cudaMemcpy dst_staging -> PBO");

    // Step 3: Unmap PBO so GL can use it
    cuda_check(
        cudaGraphicsUnmapResources(1, &dst_pbo_resource_, dst_stream_),
        "cudaGraphicsUnmapResources (dest PBO)");

    // Wait for CUDA to finish writing the PBO
    cuda_check(cudaStreamSynchronize(dst_stream_), "cudaStreamSynchronize (dest PBO)");

    // Step 4: GL upload PBO → texture (on-GPU DMA, no CPU involvement)
    GLenum pixel_format = GL_RGBA;  // CUDA staging has RGBA byte order
    GLenum pixel_type   = use_16bit_ ? GL_UNSIGNED_SHORT : GL_UNSIGNED_BYTE;
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, dest_pbo_);
    glBindTexture(GL_TEXTURE_2D, dest_gl_texture_);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width_, height_,
                    pixel_format, pixel_type, nullptr);
    glBindTexture(GL_TEXTURE_2D, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

}} // namespace caspar::vulkan_output

#endif // CASPAR_CUDA_PEER_ENABLED
