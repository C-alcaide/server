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

// cuda_prores_idct.cuh
// CUDA inverse 8×8 DCT + dequantise for ProRes 422 10-bit decoder.
//
// Implements FFmpeg's prores_idct_10 (proresdsp.c) using simple_idct_template.c
// with EXTRA_SHIFT defined and BIT_DEPTH=10:
//
//   Constants: W1=22725 W2=21407 W3=19265 W4=16384 W5=12873 W6=8867 W7=4520
//   ROW_SHIFT = 13  (total effective shift = 13 + extra_shift = 15)
//   COL_SHIFT = 18
//   DC_SHIFT  = 1   (for DC-only row: output = (coeff + 1) >> 1)
//
// Pipeline (matches FFmpeg prores_idct_10 exactly):
//   1. Dequantise + unscan: s_block[nat] = dec_coeff[scan_pos] * qmat_val
//   2. Eight row IDCTs (idctRowCondDC_extrashift_10)
//   3. DC bias add: s_block[0..7] += 8192   (restores encoder's -0x4000 + row-scale compensation)
//   4. Eight column IDCTs (idctSparseCol_extrashift_10)
//   5. Clip to [4, 1019] and write to planar int16_t output
//
// Kernel:
//   One CUDA thread block per 8×8 DCT block.
//   64 threads: thread `j` handles coefficient/pixel at position j (0..63).
//   Shared memory: 64 int32_t values.
//
// Notes:
//   - The input d_dec_coeffs is indexed by [slice][block][scan_pos].
//     This kernel uses a flat linear block index derived from the spatial
//     position in the output plane.
//   - q_scale is stored per-slice in d_q_scales[].
//   - Quant matrices from cuda_prores_tables.cuh __constant__ memory.
// ---------------------------------------------------------------------------
#pragma once

#include "cuda_prores_tables.cuh"
#include <cuda_runtime.h>
#include <stdint.h>

// ─── IDCT constants (FFmpeg simple_idct 10-bit, EXTRA_SHIFT) ─────────────────
#define IDCT_W1 22725
#define IDCT_W2 21407
#define IDCT_W3 19265
#define IDCT_W4 16384
#define IDCT_W5 12873
#define IDCT_W6  8867
#define IDCT_W7  4520

// ─── Row IDCT ────────────────────────────────────────────────────────────────
// In-place 8-point inverse DCT on v[0..7] (int32_t).
// Shift = ROW_SHIFT + extra_shift = 13 + 2 = 15.
// DC-only fast path: if v[1..7]==0, output = (v[0]+1)>>1 for all positions.
__device__ __forceinline__ void idct_row(int32_t* v)
{
    // DC-only case
    if (!(v[1]|v[2]|v[3]|v[4]|v[5]|v[6]|v[7])) {
        // DC_SHIFT=1, extra_shift=2: output = (v[0] + 1) >> 1
        int32_t dc = (v[0] + 1) >> 1;
        v[0]=v[1]=v[2]=v[3]=v[4]=v[5]=v[6]=v[7] = dc;
        return;
    }

    const int shift = 15;  // ROW_SHIFT(13) + extra_shift(2)
    const int round = 1 << (shift - 1);  // 0x4000

    int64_t a0 = (int64_t)IDCT_W4 * v[0] + round;
    int64_t a1 = a0;
    int64_t a2 = a0;
    int64_t a3 = a0;

    a0 += (int64_t)IDCT_W2 * v[2];
    a1 += (int64_t)IDCT_W6 * v[2];
    a2 -= (int64_t)IDCT_W6 * v[2];
    a3 -= (int64_t)IDCT_W2 * v[2];

    int64_t b0 = (int64_t)IDCT_W1 * v[1] + (int64_t)IDCT_W3 * v[3];
    int64_t b1 = (int64_t)IDCT_W3 * v[1] - (int64_t)IDCT_W7 * v[3];
    int64_t b2 = (int64_t)IDCT_W5 * v[1] - (int64_t)IDCT_W1 * v[3];
    int64_t b3 = (int64_t)IDCT_W7 * v[1] - (int64_t)IDCT_W5 * v[3];

    if (v[4]|v[5]|v[6]|v[7]) {
        a0 += (int64_t) IDCT_W4 * v[4] + (int64_t)IDCT_W6 * v[6];
        a1 -= (int64_t) IDCT_W4 * v[4] + (int64_t)IDCT_W2 * v[6];
        a2 -= (int64_t) IDCT_W4 * v[4] - (int64_t)IDCT_W2 * v[6];
        a3 += (int64_t) IDCT_W4 * v[4] - (int64_t)IDCT_W6 * v[6];

        b0 += (int64_t)IDCT_W5 * v[5] + (int64_t)IDCT_W7 * v[7];
        b1 -= (int64_t)IDCT_W1 * v[5] + (int64_t)IDCT_W5 * v[7];
        b2 += (int64_t)IDCT_W7 * v[5] + (int64_t)IDCT_W3 * v[7];
        b3 += (int64_t)IDCT_W3 * v[5] - (int64_t)IDCT_W1 * v[7];
    }

    v[0] = (int32_t)((a0 + b0) >> shift);
    v[7] = (int32_t)((a0 - b0) >> shift);
    v[1] = (int32_t)((a1 + b1) >> shift);
    v[6] = (int32_t)((a1 - b1) >> shift);
    v[2] = (int32_t)((a2 + b2) >> shift);
    v[5] = (int32_t)((a2 - b2) >> shift);
    v[3] = (int32_t)((a3 + b3) >> shift);
    v[4] = (int32_t)((a3 - b3) >> shift);
}

// ─── Column IDCT ─────────────────────────────────────────────────────────────
// In-place 8-point inverse DCT on a column of s_block[].
// col points to s_block[col_idx]; elements are at col[0], col[8], ..., col[56].
// COL_SHIFT = 18; includes the rounding term (1<<(COL_SHIFT-1))/W4 = 8.
__device__ __forceinline__ void idct_col(int32_t* col)
{
    // a0 = W4 * (col[0] + 8) with the rounding bias (1<<17)/W4 = 8
    int64_t a0 = (int64_t)IDCT_W4 * ((int64_t)col[0*8] + 8LL);
    int64_t a1 = a0;
    int64_t a2 = a0;
    int64_t a3 = a0;

    a0 += (int64_t)IDCT_W2 * col[2*8];
    a1 += (int64_t)IDCT_W6 * col[2*8];
    a2 -= (int64_t)IDCT_W6 * col[2*8];
    a3 -= (int64_t)IDCT_W2 * col[2*8];

    int64_t b0 = (int64_t)IDCT_W1 * col[1*8] + (int64_t)IDCT_W3 * col[3*8];
    int64_t b1 = (int64_t)IDCT_W3 * col[1*8] - (int64_t)IDCT_W7 * col[3*8];
    int64_t b2 = (int64_t)IDCT_W5 * col[1*8] - (int64_t)IDCT_W1 * col[3*8];
    int64_t b3 = (int64_t)IDCT_W7 * col[1*8] - (int64_t)IDCT_W5 * col[3*8];

    if (col[4*8]) {
        a0 += (int64_t) IDCT_W4 * col[4*8];
        a1 -= (int64_t) IDCT_W4 * col[4*8];
        a2 -= (int64_t) IDCT_W4 * col[4*8];
        a3 += (int64_t) IDCT_W4 * col[4*8];
    }
    if (col[5*8]) {
        b0 += (int64_t)IDCT_W5 * col[5*8];
        b1 -= (int64_t)IDCT_W1 * col[5*8];
        b2 += (int64_t)IDCT_W7 * col[5*8];
        b3 += (int64_t)IDCT_W3 * col[5*8];
    }
    if (col[6*8]) {
        a0 += (int64_t)IDCT_W6 * col[6*8];
        a1 -= (int64_t)IDCT_W2 * col[6*8];
        a2 += (int64_t)IDCT_W2 * col[6*8];
        a3 -= (int64_t)IDCT_W6 * col[6*8];
    }
    if (col[7*8]) {
        b0 += (int64_t)IDCT_W7 * col[7*8];
        b1 -= (int64_t)IDCT_W5 * col[7*8];
        b2 += (int64_t)IDCT_W3 * col[7*8];
        b3 -= (int64_t)IDCT_W1 * col[7*8];
    }

    const int shift = 18;  // COL_SHIFT
    col[0*8] = (int32_t)((a0 + b0) >> shift);
    col[7*8] = (int32_t)((a0 - b0) >> shift);
    col[1*8] = (int32_t)((a1 + b1) >> shift);
    col[6*8] = (int32_t)((a1 - b1) >> shift);
    col[2*8] = (int32_t)((a2 + b2) >> shift);
    col[5*8] = (int32_t)((a2 - b2) >> shift);
    col[3*8] = (int32_t)((a3 + b3) >> shift);
    col[4*8] = (int32_t)((a3 - b3) >> shift);
}

// ─── IDCT + dequantise kernel ─────────────────────────────────────────────────
// blockIdx.x = block column in plane
// blockIdx.y = block row in plane
// threadIdx.x = coefficient index, 0..63
//
// The entropy decode output (d_dec_coeffs) is indexed as:
//   d_dec_coeffs[slice * blocks_per_slice_component * 64 + block_in_slice * 64 + scan_pos]
// This kernel converts (blk_x, blk_y) to its (slice index, block-in-slice)
// coordinates to find the right coefficient data.
//
// For luma (4 Y-blocks per MB per slice, 1 MB column = 16px):
//   mb_col = blk_x / 2  (each MB is 2x2 = 4 Y blocks: so blk_x/2 and blk_y/2)
// Wait — the ProRes MB is 16×16 pixels, containing:
//   4 Y-blocks (2 columns × 2 rows of 8×8): top-left, top-right, bot-left, bot-right
// For chroma (4:2:2): each MB is halved horizontally:
//   2 Cb-blocks (1 column × 2 rows): top, bottom
//   2 Cr-blocks (1 column × 2 rows): top, bottom
//
// Block layout within a slice for Y (mbs_per_slice=M):
//   Blocks 0..M-1          = top-left of each MB (row 0, col 0..M-1 of MB-columns in slice)
//   Blocks M..2M-1         = top-right
//   Blocks 2M..3M-1        = bottom-left
//   Blocks 3M..4M-1        = bottom-right
//
// Actually the encoding order within a slice is:
//   For the Y plane, the 4*M 8×8 blocks are in the order the encoder processed them.
//   From k_dct_quantise which uses blk_x/blk_y indexing directly:
//     blk_idx = blk_y * blocks_per_row + blk_x
//   So blocks within a slice span columns [slice_mb_col*2 .. (slice_mb_col+M)*2-1]
//   and rows [mb_row*2 .. mb_row*2+1].
//   The slice index into the coefficient array is: s = mb_row * slices_per_row + col_slice
//
// The per-block slice and block-in-slice calculation (for kernel parameter indexing):
//   slice = blk_y/2 * slices_per_row + blk_x/(2*mbs_per_slice)   [for Y blocks]
//   The encoder laid out coefficient data in the contiguous slice coefficient buffer.
//
// To correctly reconstruct the slice block layout, we pass:
//   slices_per_row  = (width/16) / mbs_per_slice
//   mbs_per_slice   = 1 << log2_smb_w
//
// Y component block layout in the coefficient buffer per slice:
//   y_n = 4 * mbs_per_slice blocks, arranged in row-major scan
//   ordering: for MB column m (0..M-1), rows r∈{0,1}:
//     block = r * (2*M) + m*2 + col_in_mb
// This matches how k_dct_quantise enumerates blocks.
// ---------------------------------------------------------------------------
__global__ void k_prores_idct_dequant(
    const int16_t* __restrict__ d_dec_coeffs,    // entropy decode output
    const uint8_t* __restrict__ d_q_scales,      // per-slice q_scale
    int16_t*       __restrict__ d_out_plane,     // output planar (Y, Cb, or Cr)
    int plane_width,                              // luma: width,  chroma: width/2
    int plane_height,                             // height
    int slices_per_row,                           // luma: (width/16)/mbs
    int mbs_per_slice,                            // mbs per slice
    int coeff_stride,                             // (y_n+cb_n+cr_n)*64 per slice
    int comp_coeff_offset,                        // offset in coeff_stride for this component
    int comp_blocks_per_mb,                       // Y=4, Cb=2, Cr=2  (not used: we use blocks layout)
    int profile,
    bool is_chroma,
    bool is_interlaced)
{
    const int tid   = threadIdx.x; // 0..63
    const int blk_x = blockIdx.x;  // block column in plane (0 .. plane_width/8 - 1)
    const int blk_y = blockIdx.y;  // block row    in plane (0 .. plane_height/8 - 1)

    // ── Map (blk_x, blk_y) → (slice_idx, block_in_slice) ─────────────────
    // For Y blocks: each MB in X is 2 Y-blocks wide. A slice spans mbs_per_slice MBs.
    // mb_col_in_plane = blk_x / 2
    // slice_col       = mb_col_in_plane / mbs_per_slice
    // mb_col_in_slice = mb_col_in_plane % mbs_per_slice
    // mb_row          = blk_y / 2
    // block_row_in_mb = blk_y % 2       (0=top  8px, 1=bottom 8px of MB)
    // block_col_in_mb = blk_x % 2       (0=left 8px, 1=right  8px of MB)

    // For Cb/Cr (half width):
    // mb_col_in_plane = blk_x        (1 chroma block per MB column per row)
    // slice_col       = blk_x / mbs_per_slice
    // mb_col_in_slice = blk_x % mbs_per_slice
    // mb_row          = blk_y / 2
    // block_row_in_mb = blk_y % 2
    // (no block_col_in_mb for chroma; 1 col per MB)

    int mb_col_in_plane, mb_col_in_slice, slice_col, mb_row;
    int block_in_slice;

    if (!is_chroma) {
        // Luma
        mb_col_in_plane = blk_x >> 1;
        slice_col        = mb_col_in_plane / mbs_per_slice;
        mb_col_in_slice  = mb_col_in_plane % mbs_per_slice;
        mb_row           = blk_y >> 1;
        int brow_mb      = blk_y & 1;  // row within MB (0=top, 1=bottom)
        int bcol_mb      = blk_x & 1;  // col within MB (0=left, 1=right)
        // Apple ProRes spec (and FFmpeg): blocks within a slice are in
        // within-MB order: for each MB m (0..M-1), the 4 luma blocks are
        //   block = 4*m + brow_mb*2 + bcol_mb
        // i.e., MB0[TL,TR,BL,BR], MB1[TL,TR,BL,BR], ...
        block_in_slice = mb_col_in_slice * 4 + brow_mb * 2 + bcol_mb;
    } else {
        // Chroma
        mb_col_in_plane = blk_x;
        slice_col       = blk_x / mbs_per_slice;
        mb_col_in_slice = blk_x % mbs_per_slice;
        mb_row          = blk_y >> 1;
        int brow_mb     = blk_y & 1;
        // Apple ProRes spec: chroma blocks within a slice are in
        // within-MB order: for each MB m (0..M-1), the 2 chroma blocks are
        //   block = 2*m + brow_mb
        // i.e., MB0[top, bottom], MB1[top, bottom], ...
        block_in_slice  = mb_col_in_slice * 2 + brow_mb;
    }

    int slice_idx = mb_row * slices_per_row + slice_col;

    // ── Load dequantised coefficients from entropy decode output ──────────
    // d_dec_coeffs layout per slice: [y_n*64 | cb_n*64 | cr_n*64]
    // comp_coeff_offset: 0 for Y, y_n*64 for Cb, (y_n+cb_n)*64 for Cr
    const int16_t* slice_comp = d_dec_coeffs
        + (ptrdiff_t)slice_idx * coeff_stride
        + comp_coeff_offset
        + (ptrdiff_t)block_in_slice * 64;

    // q_scale for this slice
    const int q_scale = (int)d_q_scales[slice_idx];

    // Quantisation table value at scan position tid.
    const uint8_t q_val = is_chroma ? c_quant_chroma[profile][tid]
                                    : c_quant_luma  [profile][tid];

    // ── Dequantise into shared memory ──────────────────────────────────────
    __shared__ int32_t s_block[64];

    // The entropy decoder (decode_ac_plane) writes coefficients at buffer index
    // = natural raster position (block[b*64 + scan[i]] where scan[i] = nat_pos).
    // So slice_comp[tid] = quantised coefficient for natural position `tid`.
    // c_quant_luma/chroma is also stored in natural raster order, so
    // q_val = quant[tid] is the correct quantisation value for this position.
    // No scan-order remapping is needed here; place directly at s_block[tid].
    {
        int32_t dequant = (int32_t)slice_comp[tid] * ((int32_t)q_val * q_scale);
        s_block[tid] = dequant;
    }
    __syncthreads();

    // ── Row IDCTs (8 rows, thread 0..7 each handle one row) ───────────────
    if (tid < 8) {
        int32_t row[8];
        for (int c = 0; c < 8; c++) row[c] = s_block[tid * 8 + c];
        idct_row(row);
        for (int c = 0; c < 8; c++) s_block[tid * 8 + c] = row[c];
    }
    __syncthreads();

    // ── DC bias: add 8192 to s_block[0..7] (the Y=0 row of each column) ───
    // This is equivalent to FFmpeg's "block[i] += 8192" for i=0..7 after row IDCTs.
    if (tid < 8) {
        s_block[tid] += 8192;
    }
    __syncthreads();

    // ── Column IDCTs (8 columns, thread 0..7 each handle one column) ──────
    if (tid < 8) {
        idct_col(&s_block[tid]);
    }
    __syncthreads();

    // ── Clip and write to output plane ────────────────────────────────────
    // ProRes decoder clips to [4, 1019] (same as FFmpeg CLIP_MIN=4, CLIP_MAX_10=1019).
    constexpr int CLIP_MIN = 4;
    constexpr int CLIP_MAX = 1019;

    const int px = blk_x * 8 + (tid % 8);
    const int py = blk_y * 8 + (tid / 8);

    if (px < plane_width && py < plane_height) {
        int32_t v = s_block[tid];
        v = (v < CLIP_MIN) ? CLIP_MIN : (v > CLIP_MAX ? CLIP_MAX : v);
        d_out_plane[py * plane_width + px] = (int16_t)v;
    }
}

// Launcher helper — call once for each component plane.
inline cudaError_t launch_idct_dequant(
    const int16_t*  d_dec_coeffs,
    const uint8_t*  d_q_scales,
    int16_t*        d_out_plane,
    int             plane_width,
    int             plane_height,
    int             slices_per_row,
    int             mbs_per_slice,
    int             coeff_stride,
    int             comp_coeff_offset,
    int             profile,
    bool            is_chroma,
    bool            is_interlaced,
    cudaStream_t    stream)
{
    dim3 threads(64);
    dim3 blocks(plane_width / 8, (plane_height + 7) / 8);
    k_prores_idct_dequant<<<blocks, threads, 0, stream>>>(
        d_dec_coeffs, d_q_scales, d_out_plane,
        plane_width, plane_height,
        slices_per_row, mbs_per_slice,
        coeff_stride, comp_coeff_offset,
        0 /*comp_blocks_per_mb - unused*/,
        profile, is_chroma, is_interlaced);
    return cudaGetLastError();
}
