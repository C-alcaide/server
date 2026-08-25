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
    size_t    max_frame_bytes;   // allocation size of d_bitstream; frames larger than
                                  // this must be rejected, not uploaded (the first packet
                                  // seen sizes this buffer and later frames can be larger)
    uint8_t*  d_bitstream;       // raw icpf frame bytes (worst-case: ~width*height*3)
    uint32_t* d_slice_starts;    // [num_slices] byte offset from d_bitstream start
    uint16_t* d_slice_sizes;     // [num_slices] slice byte sizes
    int16_t*  d_dec_coeffs;      // [num_slices × coeff_stride] entropy decode output
    uint16_t* d_q_scales;        // [num_slices] q_scale per slice (FFmpeg-remapped, may exceed 255)
    int16_t*  d_y;               // [height × width]  planar luma
    int16_t*  d_cb;              // [height × width/2] planar Cb  (422), [height × width] (4444)
    int16_t*  d_cr;              // as d_cb
    int16_t*  d_alpha;           // [height × width] planar alpha — ProRes 4444 only (nullptr for 422)
    uint16_t* d_bgra16;          // [height × width × 4] output (BGRA16)

    // ── Misc ────────────────────────────────────────────────────────────────
    bool      is_444;            // true when profile == 4 (ProRes 4444 / 4444 XQ)

    // ── Slice start table (built by CPU each frame) ─────────────────────────
    // Pinned host staging for the slice index (faster H→D transfer).
    uint32_t* h_slice_starts;    // pinned
    uint16_t* h_slice_sizes;     // pinned
    // Pinned host buffer for CPU-decoded alpha (ProRes 4444 only).
    // Alpha uses a different encoding (unpack_alpha RLE), NOT DCT.
    // decoded as full-range 10-bit [0,1023]; uploaded via cudaMemcpyAsync.
    int16_t*  h_alpha;           // pinned, n_pix elements; nullptr for 422
    int       alpha_bits;        // 0=no alpha, 8=8-bit, 16=16-bit (from frame header)
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

// Async variant: submits all GPU work to ctx->stream WITHOUT the final
// cudaStreamSynchronize.  The caller must sync ctx->stream before accessing
// the GL texture and before calling CudaGLTexture::unmap().
// Use this to pipeline GPU decode of frame N with CPU work for frame N-1.
cudaError_t prores_decode_frame_async(
    ProResDecodeCtx*  ctx,
    const uint8_t*    h_icpf_data,
    size_t            icpf_size,
    int               color_matrix,
    bool              is_interlaced,
    cudaArray_t       d_gl_array);

// Planar variant -- writes the decoder's OWN planes out instead of a converted picture, so
// the MIXER does the YCbCr->RGB conversion from the frame's declared colour space.
//
// Half the bytes of the BGRA16 route at 4:2:2 (8.3 MB against 16.6 MB a 1080p frame), in the
// VRAM a slot holds and again in what the mixer samples every frame. Used by the Vulkan path
// only: the OpenGL one targets a single opaque `cudaArray_t` per slot, and three of them would
// mean three `cudaGraphicsGLRegisterImage` registrations per slot -- the call this module has
// already had to serialise behind a process-wide lock.
//
// The planes carry 10-bit LIMITED-RANGE codes clipped to [4, 1019], neutral chroma 512 -- the
// same convention as FFmpeg's yuv422p10le. The caller must back them with R16 textures DECLARED
// `bit10`, because the mixer takes its precision factor from the TEXTURE's depth; `bit16` gives
// a factor of 1 and a picture 4x too dark.
//
// Sizes: Y and alpha are width x height. Cb/Cr are width/2 x height for 4:2:2 and width x height
// for 4444. Pass nullptr for d_alpha_array on anything but 4444.
//
// THE PRODUCER CALLS THIS FOR 4:2:2 ONLY, and the alpha argument is carried rather than used.
// Planar 4444 is four full-size 16-bit planes -- 8 bytes a pixel, exactly what BGRA16 costs --
// so it saves nothing, and it loses picture: FFmpeg decodes ProRes 4444 to yuva444p12 while
// this decoder produces 10-bit planes, which normalises alpha to 0.99904 against the
// reference's 0.99985 and lands the premultiply a fraction low. Measured at `any diff` 0.50%
// -> 13.87% on prores_4444a_bt709_sdr, all 1 LSB, all in one alpha band. The alpha path here
// is correct and tested only as far as that measurement went; treat it as unexercised.
//
// No colour_matrix argument -- that answer belongs to the shader now.
cudaError_t prores_decode_frame_planar_async(
    ProResDecodeCtx*  ctx,
    const uint8_t*    h_icpf_data,
    size_t            icpf_size,
    bool              is_interlaced,
    cudaArray_t       d_y_array,
    cudaArray_t       d_cb_array,
    cudaArray_t       d_cr_array,
    cudaArray_t       d_alpha_array);   // 4444 only; nullptr otherwise

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
