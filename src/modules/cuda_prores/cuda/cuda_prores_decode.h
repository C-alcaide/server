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
 * ProRes format reference: Apple Inc. "ProRes RAW White Paper" (public documentation).
 */

// cuda_prores_decode.h
// GPU-side decode context and entry points for the ProRes CUDA decoder.
// ---------------------------------------------------------------------------
#pragma once

#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>

// ---------------------------------------------------------------------------
// ProResDecodeCtx — all device-side resources for one decode slot
// (triple-buffered: 3 instances are maintained by the producer).
// ---------------------------------------------------------------------------
struct ProResDecodeCtx {
    // ── Geometry ────────────────────────────────────────────────────────────
    int width, height;
    int profile;            // ProResProfile (0=Proxy .. 4=4444)
    int mbs_per_slice;      // power of 2
    int slices_per_row;     // (width/16) / mbs_per_slice
    int num_slices;         // slices_per_row * (height/16)
    int coeff_stride;       // (y_n + cb_n + cr_n) * 64 per slice

    // ── Device buffers ──────────────────────────────────────────────────────
    uint8_t*  d_bitstream;       // raw icpf frame bytes (worst-case: ~width*height*3)
    uint32_t* d_slice_starts;    // [num_slices] byte offset from d_bitstream start
    uint16_t* d_slice_sizes;     // [num_slices] slice byte sizes
    int16_t*  d_dec_coeffs;      // [num_slices × coeff_stride] entropy decode output
    uint8_t*  d_q_scales;        // [num_slices] q_scale per slice
    int16_t*  d_y;               // [height × width]  planar luma
    int16_t*  d_cb;              // [height × width/2] planar Cb
    int16_t*  d_cr;              // [height × width/2] planar Cr
    uint16_t* d_bgra16;          // [height × width × 4] output (BGRA16)

    // ── Slice start table (built by CPU each frame) ─────────────────────────
    // Pinned host staging for the slice index (faster H→D transfer).
    uint32_t* h_slice_starts;    // pinned
    uint16_t* h_slice_sizes;     // pinned

    // ── Stream ──────────────────────────────────────────────────────────────
    cudaStream_t stream;

    // ── CUDA-GL texture (filled after first frame) ───────────────────────
    // Owned by the producer (CudaGLTexture wrapper lives there).
    // The cudaArray_t is obtained via CudaGLTexture::map() per frame.
};

// ---------------------------------------------------------------------------
// Allocate / free a ProResDecodeCtx.
// Must be called from the CUDA device thread (cudaSetDevice already called).
// ---------------------------------------------------------------------------
cudaError_t prores_decode_ctx_create(ProResDecodeCtx* ctx,
                                     int width, int height,
                                     int profile,
                                     int mbs_per_slice,
                                     int slices_per_row,
                                     int num_slices,
                                     size_t max_frame_bytes);

void prores_decode_ctx_destroy(ProResDecodeCtx* ctx);

// ---------------------------------------------------------------------------
// Decode one ProRes 422 progressive frame.
//
// Steps:
//   1. Upload icpf frame bytes → d_bitstream (H→D async)
//   2. Build slice start/size table on CPU, upload → d_slice_starts/sizes
//   3. Launch k_prores_entropy_decode
//   4. Launch k_prores_idct_dequant  (Y, Cb, Cr planes)
//   5. Launch k_ycbcr422p10_to_bgra16
//   6. cudaMemcpy2DToArrayAsync → d_gl_array (zero-copy to GL texture)
//   7. cudaStreamSynchronize (waits for all GPU work on ctx->stream)
//
// The caller must:
//   - map the GL texture BEFORE calling this function
//   - unmap the GL texture AFTER cudaStreamSynchronize
//
// color_matrix: 9 = BT.2020, else BT.709
// Returns the CUDA error of the first failure (cudaSuccess on success).
// ---------------------------------------------------------------------------
cudaError_t prores_decode_frame(
    ProResDecodeCtx*  ctx,
    const uint8_t*    h_icpf_data,       // host: full icpf frame (w/ 8-byte box hdr)
    size_t            icpf_size,
    int               color_matrix,
    bool              is_interlaced,
    cudaArray_t       d_gl_array);       // mapped cudaArray from CudaGLTexture::map()

// Headless variant — outputs to a plain host buffer instead of a GL texture.
// Useful for unit tests and offline processing (no OpenGL context required).
// h_bgra16_out must point to at least ctx->width * ctx->height * 4 * sizeof(uint16_t) bytes.
cudaError_t prores_decode_frame_to_host(
    ProResDecodeCtx*  ctx,
    const uint8_t*    h_icpf_data,
    size_t            icpf_size,
    int               color_matrix,
    bool              is_interlaced,
    uint16_t*         h_bgra16_out);     // host output: width*height*4 uint16_t
