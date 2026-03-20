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

// cuda_prores_decode.cu
// ProRes 422 GPU decode orchestration — calls entropy, IDCT, and colour kernels.
//
// #define PRORES_TABLES_DEFINE_CONSTANTS is NOT set here because
// cuda_prores_frame.cu already defines the __constant__ symbols.
// We use the `extern __constant__` declarations from cuda_prores_tables.cuh.
#include "cuda_prores_decode.h"

// Kernel headers — included here (once) so the device code is compiled
// into this single translation unit.
#include "cuda_prores_entropy_decode.cuh"
#include "cuda_prores_idct.cuh"
#include "cuda_prores_to_bgra16.cuh"

// Logging: use CasparCG stream logger when available, plain stderr otherwise.
#if __has_include(<common/log.h>)
#include <common/log.h>
#define PRORES_DEC_LOG_ERROR(narrow_msg) CASPAR_LOG(error) << L"" narrow_msg
#else
#include <cstdio>
#define PRORES_DEC_LOG_ERROR(narrow_msg) fprintf(stderr, "%s\n", narrow_msg)
#endif

#include <cuda_runtime.h>
#include <stdint.h>
#include <cstring>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call)                                              \
    do {                                                              \
        cudaError_t _e = (call);                                      \
        if (_e != cudaSuccess) return _e;                             \
    } while (0)

static inline void safe_free(void*& p)
{
    if (p) { cudaFree(p); p = nullptr; }
}
static inline void safe_free_host(void*& p)
{
    if (p) { cudaFreeHost(p); p = nullptr; }
}

// ---------------------------------------------------------------------------
// prores_decode_ctx_create
// ---------------------------------------------------------------------------

cudaError_t prores_decode_ctx_create(ProResDecodeCtx* ctx,
                                     int width, int height,
                                     int profile,
                                     int mbs_per_slice,
                                     int slices_per_row,
                                     int num_slices,
                                     size_t max_frame_bytes)
{
    memset(ctx, 0, sizeof(*ctx));
    ctx->width          = width;
    ctx->height         = height;
    ctx->profile        = profile;
    ctx->mbs_per_slice  = mbs_per_slice;
    ctx->slices_per_row = slices_per_row;
    ctx->num_slices     = num_slices;

    const bool is_444 = (profile == 4);
    ctx->is_444 = is_444;

    // For ProRes 422 variants: Cb/Cr are 4:2:2 (half-width).
    // For ProRes 4444:         Cb/Cr/Alpha are 4:4:4 (full-width).
    const int chroma_n_per_mb = is_444 ? 4 : 2;  // blocks per MB in chroma plane
    const int y_n  = 4 * mbs_per_slice;
    const int cb_n = chroma_n_per_mb * mbs_per_slice;
    const int cr_n = cb_n;
    const int a_n  = is_444 ? y_n : 0;           // alpha blocks (4444 only)
    ctx->coeff_stride = (y_n + cb_n + cr_n + a_n) * 64;

    const size_t n_pix      = (size_t)width * height;
    const size_t n_chroma   = is_444 ? n_pix : (size_t)(width / 2) * height;
    const size_t coeff_bytes = (size_t)num_slices * ctx->coeff_stride * sizeof(int16_t);

    CUDA_CHECK(cudaMalloc(&ctx->d_bitstream,   max_frame_bytes));
    CUDA_CHECK(cudaMalloc((void**)&ctx->d_slice_starts, (size_t)num_slices * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc((void**)&ctx->d_slice_sizes,  (size_t)num_slices * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc((void**)&ctx->d_dec_coeffs,   coeff_bytes));
    CUDA_CHECK(cudaMalloc((void**)&ctx->d_q_scales,     (size_t)num_slices * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc((void**)&ctx->d_y,            n_pix    * sizeof(int16_t)));
    CUDA_CHECK(cudaMalloc((void**)&ctx->d_cb,           n_chroma * sizeof(int16_t)));
    CUDA_CHECK(cudaMalloc((void**)&ctx->d_cr,           n_chroma * sizeof(int16_t)));
    ctx->d_alpha = nullptr;
    if (is_444)
        CUDA_CHECK(cudaMalloc((void**)&ctx->d_alpha,    n_pix    * sizeof(int16_t)));
    CUDA_CHECK(cudaMalloc((void**)&ctx->d_bgra16,       n_pix * 4 * sizeof(uint16_t)));

    // Pinned host staging for slice index.
    CUDA_CHECK(cudaMallocHost((void**)&ctx->h_slice_starts, (size_t)num_slices * sizeof(uint32_t)));
    CUDA_CHECK(cudaMallocHost((void**)&ctx->h_slice_sizes,  (size_t)num_slices * sizeof(uint16_t)));

    CUDA_CHECK(cudaStreamCreate(&ctx->stream));

    // Upload quant tables and scan order to __constant__ memory (first call
    // is a no-op if already done by the encoder init; subsequent calls just
    // overwrite with the same values).
    CUDA_CHECK(prores_tables_upload());

    return cudaSuccess;
}

void prores_decode_ctx_destroy(ProResDecodeCtx* ctx)
{
    if (!ctx) return;
    if (ctx->stream) { cudaStreamDestroy(ctx->stream); ctx->stream = nullptr; }

    safe_free((void*&)ctx->d_bitstream);
    safe_free((void*&)ctx->d_slice_starts);
    safe_free((void*&)ctx->d_slice_sizes);
    safe_free((void*&)ctx->d_dec_coeffs);
    safe_free((void*&)ctx->d_q_scales);
    safe_free((void*&)ctx->d_y);
    safe_free((void*&)ctx->d_cb);
    safe_free((void*&)ctx->d_cr);
    safe_free((void*&)ctx->d_alpha);
    safe_free((void*&)ctx->d_bgra16);
    safe_free_host((void*&)ctx->h_slice_starts);
    safe_free_host((void*&)ctx->h_slice_sizes);
}

// ---------------------------------------------------------------------------
// Parse slice index from icpf frame (CPU-side).
//
// The picture header sits at: d_bitstream + 8 (frame box hdr) + frame_hdr_size
// Slice size index:  h_slice_index[i] = BE16 size of slice i
// Slice data starts: picture_data_start = picture_header_base + pic_hdr_size + num_slices*2
//
// Returns false on parse failure.
// ---------------------------------------------------------------------------
static bool build_slice_table(
    const uint8_t* data, size_t size,
    int num_slices,
    uint32_t* h_slice_starts,   // output: byte offsets from data[0]
    uint16_t* h_slice_sizes)    // output: byte sizes
{
    if (size < 28) return false;

    // Skip 8-byte box header.
    const uint8_t* fhdr = data + 8;
    const size_t   fhdr_avail = size - 8;

    // Frame header size (BE16).
    if (fhdr_avail < 2) return false;
    const int frame_hdr_size = ((int)fhdr[0] << 8) | fhdr[1];
    if (frame_hdr_size < 18 || (size_t)frame_hdr_size > fhdr_avail)
        return false;

    // Picture header starts immediately after frame header.
    const uint8_t* phdr = fhdr + frame_hdr_size;
    const size_t   phdr_avail = fhdr_avail - (size_t)frame_hdr_size;

    if (phdr_avail < 8) return false;
    const int pic_hdr_size = phdr[0] >> 3;   // in bytes
    if (pic_hdr_size < 8 || (size_t)pic_hdr_size > phdr_avail) return false;

    // uint32_t pic_data_size = AV_RB32(phdr + 1);  // total picture data

    // Slice index sits right after the picture header.
    const uint8_t* index_ptr = phdr + pic_hdr_size;
    const size_t   index_bytes = (size_t)num_slices * 2;

    if ((ptrdiff_t)(index_ptr + index_bytes - data) > (ptrdiff_t)size)
        return false;

    // Slice data starts right after the index.
    const uint8_t* slice_data_base = index_ptr + index_bytes;
    const uint32_t base_offset = (uint32_t)(slice_data_base - data);

    uint32_t cursor = 0;
    for (int s = 0; s < num_slices; s++) {
        uint16_t slice_sz = (uint16_t)(((uint16_t)index_ptr[s*2] << 8) | index_ptr[s*2 + 1]);
        h_slice_starts[s] = base_offset + cursor;
        h_slice_sizes [s] = slice_sz;
        cursor += slice_sz;
    }
    return true;
}

// ---------------------------------------------------------------------------
// prores_decode_frame
// ---------------------------------------------------------------------------

cudaError_t prores_decode_frame(
    ProResDecodeCtx* ctx,
    const uint8_t*   h_icpf_data,
    size_t           icpf_size,
    int              color_matrix,
    bool             is_interlaced,
    cudaArray_t      d_gl_array)
{
    cudaStream_t s = ctx->stream;

    // ── 1. Build slice table (CPU) ─────────────────────────────────────────
    if (!build_slice_table(h_icpf_data, icpf_size,
                           ctx->num_slices,
                           ctx->h_slice_starts,
                           ctx->h_slice_sizes)) {
        PRORES_DEC_LOG_ERROR("[cuda_prores_decode] build_slice_table failed");
        return cudaErrorInvalidValue;
    }

    // ── 2. Upload frame data + slice index (async) ─────────────────────────
    CUDA_CHECK(cudaMemcpyAsync(ctx->d_bitstream, h_icpf_data, icpf_size,
                               cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(ctx->d_slice_starts, ctx->h_slice_starts,
                               (size_t)ctx->num_slices * sizeof(uint32_t),
                               cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(ctx->d_slice_sizes, ctx->h_slice_sizes,
                               (size_t)ctx->num_slices * sizeof(uint16_t),
                               cudaMemcpyHostToDevice, s));

    // ── 3. Entropy decode (1 thread/slice) ────────────────────────────────
    CUDA_CHECK(launch_entropy_decode(
        ctx->d_bitstream,
        ctx->d_slice_starts,
        ctx->d_slice_sizes,
        ctx->d_dec_coeffs,
        ctx->d_q_scales,
        ctx->mbs_per_slice,
        ctx->num_slices,
        ctx->width / 16,           // mb_width
        ctx->slices_per_row,       // for partial-slice computation
        is_interlaced,
        ctx->is_444,
        s));

    // ── 4a. IDCT+dequant — Luma ────────────────────────────────────────────
    const int y_n      = 4 * ctx->mbs_per_slice;
    const int cb_n     = (ctx->is_444 ? 4 : 2) * ctx->mbs_per_slice;
    const int chroma_w = ctx->is_444 ? ctx->width : ctx->width / 2;

    CUDA_CHECK(launch_idct_dequant(
        ctx->d_dec_coeffs,
        ctx->d_q_scales,
        ctx->d_y,
        ctx->width, ctx->height,
        ctx->slices_per_row,
        ctx->mbs_per_slice,
        ctx->coeff_stride,
        0,            // comp_coeff_offset for Y
        ctx->profile,
        /*is_chroma=*/false,
        is_interlaced,
        4,            // comp_blocks_per_mb: Y always has 4 blocks/MB
        s));

    // ── 4b. IDCT+dequant — Cb ─────────────────────────────────────────────
    CUDA_CHECK(launch_idct_dequant(
        ctx->d_dec_coeffs,
        ctx->d_q_scales,
        ctx->d_cb,
        chroma_w, ctx->height,
        ctx->slices_per_row,
        ctx->mbs_per_slice,
        ctx->coeff_stride,
        y_n * 64,     // comp_coeff_offset for Cb
        ctx->profile,
        /*is_chroma=*/true,
        is_interlaced,
        ctx->is_444 ? 4 : 2,   // 4444 chroma uses luma-style geometry
        s));

    // ── 4c. IDCT+dequant — Cr ─────────────────────────────────────────────
    CUDA_CHECK(launch_idct_dequant(
        ctx->d_dec_coeffs,
        ctx->d_q_scales,
        ctx->d_cr,
        chroma_w, ctx->height,
        ctx->slices_per_row,
        ctx->mbs_per_slice,
        ctx->coeff_stride,
        (y_n + cb_n) * 64,   // comp_coeff_offset for Cr
        ctx->profile,
        /*is_chroma=*/true,
        is_interlaced,
        ctx->is_444 ? 4 : 2,   // 4444 chroma uses luma-style geometry
        s));

    // ── 4d. IDCT+dequant — Alpha (ProRes 4444 only) ───────────────────────
    if (ctx->is_444) {
        CUDA_CHECK(launch_idct_dequant(
            ctx->d_dec_coeffs,
            ctx->d_q_scales,
            ctx->d_alpha,
            ctx->width, ctx->height,
            ctx->slices_per_row,
            ctx->mbs_per_slice,
            ctx->coeff_stride,
            (y_n + cb_n + cb_n) * 64,  // comp_coeff_offset for Alpha
            ctx->profile,
            /*is_chroma=*/false,        // alpha uses luma quant table
            is_interlaced,
            4,  // Alpha always uses luma-style geometry (full-width, 4 blocks/MB)
            s));
    }

    // ── 5. YCbCr → BGRA16 ──────────────────────────────────────────────────
    if (ctx->is_444) {
        CUDA_CHECK(launch_ycbcr444_to_bgra16(
            ctx->d_y, ctx->d_cb, ctx->d_cr, ctx->d_alpha,
            ctx->d_bgra16,
            ctx->width, ctx->height,
            color_matrix,
            s));
    } else {
        CUDA_CHECK(launch_ycbcr_to_bgra16(
            ctx->d_y, ctx->d_cb, ctx->d_cr,
            ctx->d_bgra16,
            ctx->width, ctx->height,
            color_matrix,
            s));
    }

    // ── 6. D→D copy to GL texture (zero host transfer) ────────────────────
    // Each BGRA16 row is width * 8 bytes (4 channels × 2 bytes).
    // The GL texture format is GL_RGBA16 (GLSL reads as R=blue, G=green, B=red
    // wait — we should output in RGBA order for GL_RGBA16.
    // The CasparCG OGL image_mixer reads textures as RGBA.
    // Our d_bgra16 is in BGRA order (B first), which mismatches GL_RGBA16.
    // We write RGBA here: dst[0]=R, dst[1]=G, dst[2]=B, dst[3]=A.
    // But k_ycbcr422p10_to_bgra16 writes: dst[0]=B, dst[1]=G, dst[2]=R, dst[3]=A.
    // The memory layout used in CasparCG for BGRA textures (GL_BGRA / GL_RGBA):
    // Looking at FORMAT[] in texture.cpp:
    //   FORMAT[4] = GL_BGRA  (stride=4)
    // So the OGL texture expects BGRA stored in memory (matches our output).
    // cudaMemcpy2DToArrayAsync copies raw bytes; the array is backed by the
    // GL texture with format GL_RGBA16 (internal), GL_BGRA (pixel format).
    // So BGRA in memory → rendered as BGRA by GL → correct.

    const size_t row_bytes = (size_t)ctx->width * 4 * sizeof(uint16_t); // BGRA16 row
    CUDA_CHECK(cudaMemcpy2DToArrayAsync(
        d_gl_array,
        /*wOffset=*/0, /*hOffset=*/0,
        ctx->d_bgra16,
        /*spitch=*/row_bytes,
        /*width=*/row_bytes,
        /*height=*/(size_t)ctx->height,
        cudaMemcpyDeviceToDevice,
        s));

    // ── 7. Synchronise ────────────────────────────────────────────────────
    CUDA_CHECK(cudaStreamSynchronize(s));

    return cudaSuccess;
}

// ---------------------------------------------------------------------------
// prores_decode_frame_async
// ---------------------------------------------------------------------------
// Identical to prores_decode_frame but WITHOUT the final cudaStreamSynchronize.
// All GPU work is queued to ctx->stream and returns immediately.
// The caller MUST:
//   1. Call cudaStreamSynchronize(ctx->stream) before accessing the GL texture.
//   2. Call CudaGLTexture::unmap() only AFTER that sync.
// This allows the caller to overlap CPU work (demuxer read, audio, frame alloc)
// with the GPU decode of the current frame, forming a 2-stage pipeline that
// eliminates the idle-GPU gap in the synchronous version.
cudaError_t prores_decode_frame_async(
    ProResDecodeCtx* ctx,
    const uint8_t*   h_icpf_data,
    size_t           icpf_size,
    int              color_matrix,
    bool             is_interlaced,
    cudaArray_t      d_gl_array)
{
    cudaStream_t s = ctx->stream;

    if (!build_slice_table(h_icpf_data, icpf_size,
                           ctx->num_slices,
                           ctx->h_slice_starts,
                           ctx->h_slice_sizes)) {
        PRORES_DEC_LOG_ERROR("[cuda_prores_decode] build_slice_table failed (async)");
        return cudaErrorInvalidValue;
    }

    CUDA_CHECK(cudaMemcpyAsync(ctx->d_bitstream, h_icpf_data, icpf_size,
                               cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(ctx->d_slice_starts, ctx->h_slice_starts,
                               (size_t)ctx->num_slices * sizeof(uint32_t),
                               cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(ctx->d_slice_sizes, ctx->h_slice_sizes,
                               (size_t)ctx->num_slices * sizeof(uint16_t),
                               cudaMemcpyHostToDevice, s));

    CUDA_CHECK(launch_entropy_decode(
        ctx->d_bitstream,
        ctx->d_slice_starts,
        ctx->d_slice_sizes,
        ctx->d_dec_coeffs,
        ctx->d_q_scales,
        ctx->mbs_per_slice,
        ctx->num_slices,
        ctx->width / 16,
        ctx->slices_per_row,
        is_interlaced,
        ctx->is_444,
        s));

    const int y_n      = 4 * ctx->mbs_per_slice;
    const int cb_n     = (ctx->is_444 ? 4 : 2) * ctx->mbs_per_slice;
    const int chroma_w = ctx->is_444 ? ctx->width : ctx->width / 2;

    CUDA_CHECK(launch_idct_dequant(
        ctx->d_dec_coeffs, ctx->d_q_scales, ctx->d_y,
        ctx->width, ctx->height, ctx->slices_per_row,
        ctx->mbs_per_slice, ctx->coeff_stride, 0,
        ctx->profile, false, is_interlaced, 4, s));

    CUDA_CHECK(launch_idct_dequant(
        ctx->d_dec_coeffs, ctx->d_q_scales, ctx->d_cb,
        chroma_w, ctx->height, ctx->slices_per_row,
        ctx->mbs_per_slice, ctx->coeff_stride, y_n * 64,
        ctx->profile, true, is_interlaced, ctx->is_444 ? 4 : 2, s));

    CUDA_CHECK(launch_idct_dequant(
        ctx->d_dec_coeffs, ctx->d_q_scales, ctx->d_cr,
        chroma_w, ctx->height, ctx->slices_per_row,
        ctx->mbs_per_slice, ctx->coeff_stride, (y_n + cb_n) * 64,
        ctx->profile, true, is_interlaced, ctx->is_444 ? 4 : 2, s));

    if (ctx->is_444) {
        CUDA_CHECK(launch_idct_dequant(
            ctx->d_dec_coeffs, ctx->d_q_scales, ctx->d_alpha,
            ctx->width, ctx->height, ctx->slices_per_row,
            ctx->mbs_per_slice, ctx->coeff_stride, (y_n + cb_n + cb_n) * 64,
            ctx->profile, false, is_interlaced, 4, s));
        CUDA_CHECK(launch_ycbcr444_to_bgra16(
            ctx->d_y, ctx->d_cb, ctx->d_cr, ctx->d_alpha,
            ctx->d_bgra16, ctx->width, ctx->height, color_matrix, s));
    } else {
        CUDA_CHECK(launch_ycbcr_to_bgra16(
            ctx->d_y, ctx->d_cb, ctx->d_cr,
            ctx->d_bgra16, ctx->width, ctx->height, color_matrix, s));
    }

    const size_t row_bytes = (size_t)ctx->width * 4 * sizeof(uint16_t);
    CUDA_CHECK(cudaMemcpy2DToArrayAsync(
        d_gl_array,
        0, 0,
        ctx->d_bgra16,
        row_bytes, row_bytes, (size_t)ctx->height,
        cudaMemcpyDeviceToDevice,
        s));

    // No cudaStreamSynchronize -- caller is responsible.
    return cudaSuccess;
}
cudaError_t prores_decode_frame_to_host(
    ProResDecodeCtx* ctx,
    const uint8_t*   h_icpf_data,
    size_t           icpf_size,
    int              color_matrix,
    bool             is_interlaced,
    uint16_t*        h_bgra16_out)
{
    cudaStream_t s = ctx->stream;

    if (!build_slice_table(h_icpf_data, icpf_size,
                           ctx->num_slices,
                           ctx->h_slice_starts,
                           ctx->h_slice_sizes)) {
        PRORES_DEC_LOG_ERROR("[cuda_prores_decode] build_slice_table failed (to_host)");
        return cudaErrorInvalidValue;
    }

    CUDA_CHECK(cudaMemcpyAsync(ctx->d_bitstream, h_icpf_data, icpf_size,
                               cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(ctx->d_slice_starts, ctx->h_slice_starts,
                               (size_t)ctx->num_slices * sizeof(uint32_t),
                               cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(ctx->d_slice_sizes, ctx->h_slice_sizes,
                               (size_t)ctx->num_slices * sizeof(uint16_t),
                               cudaMemcpyHostToDevice, s));

    const int y_n      = 4 * ctx->mbs_per_slice;
    const int cb_n     = (ctx->is_444 ? 4 : 2) * ctx->mbs_per_slice;
    const int chroma_w = ctx->is_444 ? ctx->width : ctx->width / 2;

    CUDA_CHECK(launch_entropy_decode(
        ctx->d_bitstream, ctx->d_slice_starts, ctx->d_slice_sizes,
        ctx->d_dec_coeffs, ctx->d_q_scales,
        ctx->mbs_per_slice, ctx->num_slices,
        ctx->width / 16, ctx->slices_per_row,
        is_interlaced, ctx->is_444, s));

    CUDA_CHECK(launch_idct_dequant(
        ctx->d_dec_coeffs, ctx->d_q_scales, ctx->d_y,
        ctx->width, ctx->height, ctx->slices_per_row,
        ctx->mbs_per_slice, ctx->coeff_stride, 0,
        ctx->profile, false, is_interlaced, 4, s));

    CUDA_CHECK(launch_idct_dequant(
        ctx->d_dec_coeffs, ctx->d_q_scales, ctx->d_cb,
        chroma_w, ctx->height, ctx->slices_per_row,
        ctx->mbs_per_slice, ctx->coeff_stride, y_n * 64,
        ctx->profile, true, is_interlaced, ctx->is_444 ? 4 : 2, s));

    CUDA_CHECK(launch_idct_dequant(
        ctx->d_dec_coeffs, ctx->d_q_scales, ctx->d_cr,
        chroma_w, ctx->height, ctx->slices_per_row,
        ctx->mbs_per_slice, ctx->coeff_stride, (y_n + cb_n) * 64,
        ctx->profile, true, is_interlaced, ctx->is_444 ? 4 : 2, s));

    if (ctx->is_444) {
        CUDA_CHECK(launch_idct_dequant(
            ctx->d_dec_coeffs, ctx->d_q_scales, ctx->d_alpha,
            ctx->width, ctx->height, ctx->slices_per_row,
            ctx->mbs_per_slice, ctx->coeff_stride, (y_n + cb_n + cb_n) * 64,
            ctx->profile, false, is_interlaced, 4, s));
        CUDA_CHECK(launch_ycbcr444_to_bgra16(
            ctx->d_y, ctx->d_cb, ctx->d_cr, ctx->d_alpha,
            ctx->d_bgra16, ctx->width, ctx->height, color_matrix, s));
    } else {
        CUDA_CHECK(launch_ycbcr_to_bgra16(
            ctx->d_y, ctx->d_cb, ctx->d_cr,
            ctx->d_bgra16, ctx->width, ctx->height, color_matrix, s));
    }

    // Copy linear BGRA16 from device to host (row-by-row, contiguous).
    const size_t row_bytes = (size_t)ctx->width * 4 * sizeof(uint16_t);
    CUDA_CHECK(cudaMemcpy2DAsync(
        h_bgra16_out,   /*dpitch=*/row_bytes,
        ctx->d_bgra16,  /*spitch=*/row_bytes,
        /*width=*/row_bytes, /*height=*/(size_t)ctx->height,
        cudaMemcpyDeviceToHost, s));

    CUDA_CHECK(cudaStreamSynchronize(s));
    return cudaSuccess;
}