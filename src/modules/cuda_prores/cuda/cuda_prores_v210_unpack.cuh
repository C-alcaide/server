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

// cuda_prores_v210_unpack.cuh
// CUDA kernel to unpack V210 (10-bit packed 4:2:2 YCbCr) from DeckLink
// into planar YUV422P10 (int16_t planes) in device memory.
//
// V210 bit layout (per 32-bit word, groups of 3 words = 6 pixels):
//   Word 0:  Cb0[9:0]  Y0[9:0]  Cr0[9:0]   [bits 29:20 | 19:10 | 9:0]
//   Word 1:   Y1[9:0]  Cb1[9:0]  Y2[9:0]
//   Word 2:  Cr1[9:0]   Y3[9:0]  Cb2[9:0]   (Cb2/Cr2 belong to Y4/Y5)
//   ...
// Each group of 4 32-bit words encodes 6 luma + 2 Cb + 2 Cr samples.
//
// DeckLink delivers data in bmdFormat10BitYUV (= V210).
// Output is int16_t with values in [0, 1023] (10-bit unsigned, no DC leveling).
// DC leveling (subtracting 512) is done in the DCT kernel.
//
// Reference: Apple V210 documentation; SMPTE 422M packed format.
#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

// One CUDA thread handles one group of 6 pixels (4 input words).
// Launch as: <<<(width*height/6 + 127)/128, 128>>>
// Precondition: width must be a multiple of 48 (V210 row alignment).
__global__ void k_v210_unpack(
    const uint32_t * __restrict__ d_v210,   // input:  V210 packed, row-major
    int16_t        * __restrict__ d_y,      // output: luma   [height * width]
    int16_t        * __restrict__ d_cb,     // output: Cb     [height * width/2]
    int16_t        * __restrict__ d_cr,     // output: Cr     [height * width/2]
    int width,
    int height)
{
    // Each group processes 6 luma + 2 chroma-pair pixels.
    const int words_per_row = ((width + 5) / 6) * 4; // V210 row stride in 32-bit words
    const int groups_per_row = width / 6;
    const int total_groups   = groups_per_row * height;

    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total_groups) return;

    int row  = gid / groups_per_row;
    int col6 = gid % groups_per_row; // 0-based column in 6-pixel units

    // Byte offset of this group in the V210 stream
    const uint32_t *src = d_v210 + row * words_per_row + col6 * 4;

    uint32_t w0 = src[0];
    uint32_t w1 = src[1];
    uint32_t w2 = src[2];
    uint32_t w3 = src[3];

    // Unpack 10-bit fields (bits 9:0, 19:10, 29:20 of each 32-bit word)
    int16_t Cb0 = (int16_t)( w0        & 0x3FFu);
    int16_t Y0  = (int16_t)((w0 >> 10) & 0x3FFu);
    int16_t Cr0 = (int16_t)((w0 >> 20) & 0x3FFu);

    int16_t Y1  = (int16_t)( w1        & 0x3FFu);
    int16_t Cb1 = (int16_t)((w1 >> 10) & 0x3FFu);
    int16_t Y2  = (int16_t)((w1 >> 20) & 0x3FFu);

    int16_t Cr1 = (int16_t)( w2        & 0x3FFu);
    int16_t Y3  = (int16_t)((w2 >> 10) & 0x3FFu);
    int16_t Cb2 = (int16_t)((w2 >> 20) & 0x3FFu);

    int16_t Y4  = (int16_t)( w3        & 0x3FFu);
    int16_t Cr2 = (int16_t)((w3 >> 10) & 0x3FFu);
    int16_t Y5  = (int16_t)((w3 >> 20) & 0x3FFu);

    // Write luma: 6 consecutive samples at pixel column col6*6
    int y_base  = row * width + col6 * 6;
    d_y[y_base + 0] = Y0;
    d_y[y_base + 1] = Y1;
    d_y[y_base + 2] = Y2;
    d_y[y_base + 3] = Y3;
    d_y[y_base + 4] = Y4;
    d_y[y_base + 5] = Y5;

    // Write chroma: 3 Cb + 3 Cr samples at chroma column col6*3
    int c_base  = row * (width / 2) + col6 * 3;
    d_cb[c_base + 0] = Cb0;
    d_cb[c_base + 1] = Cb1;
    d_cb[c_base + 2] = Cb2;
    d_cr[c_base + 0] = Cr0;
    d_cr[c_base + 1] = Cr1;
    d_cr[c_base + 2] = Cr2;
}

// ---------------------------------------------------------------------------
// Field-aware V210 unpack for interlaced frames.
//
// Extracts one field from an interlaced V210 frame into a half-height planar
// buffer.  V210 delivers both fields interleaved line-by-line:
//   field 0 (top / even) : full_row = 0, 2, 4, ...  → field_row 0, 1, 2, ...
//   field 1 (bottom / odd): full_row = 1, 3, 5, ...  → field_row 0, 1, 2, ...
//
// Output planes hold field_height = full_height / 2 rows.
// ---------------------------------------------------------------------------
__global__ void k_v210_unpack_field(
    const uint32_t * __restrict__ d_v210,
    int16_t        * __restrict__ d_y,
    int16_t        * __restrict__ d_cb,
    int16_t        * __restrict__ d_cr,
    int width,
    int full_height,
    int field)   // 0 = top (even rows), 1 = bottom (odd rows)
{
    // CEIL, matching the words_per_row above and the packer. This used to be `width / 6`
    // -- FLOOR against a CEIL stride -- so for any width not divisible by six the last
    // `width % 6` luma samples of each field were never written. The launcher below documented
    // that as "pre-zeroed by caller"; the interlaced caller in `cuda_prores_frame.cu` does no
    // such memset, so those columns carried whatever was in VRAM from a previous frame or the
    // other field. 1920 and 720 divide by six, which is why standard interlaced rasters never
    // showed it; a DCI width does not. Found by audit 2026-08-26.
    //
    // The partial last group is handled by bounds-checking each write below, so there is no
    // pre-zeroing contract left to violate.
    const int words_per_row  = ((width + 5) / 6) * 4;
    const int groups_per_row = (width + 5) / 6;
    const int field_height   = full_height / 2;
    const int total_groups   = groups_per_row * field_height;

    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total_groups) return;

    int field_row = gid / groups_per_row;   // 0 .. field_height-1
    int col6      = gid % groups_per_row;

    // Map field row to the full-frame row
    int full_row = field + field_row * 2;

    const uint32_t *src = d_v210 + full_row * words_per_row + col6 * 4;

    uint32_t w0 = src[0], w1 = src[1], w2 = src[2], w3 = src[3];

    int16_t Cb0 = (int16_t)( w0        & 0x3FFu);
    int16_t Y0  = (int16_t)((w0 >> 10) & 0x3FFu);
    int16_t Cr0 = (int16_t)((w0 >> 20) & 0x3FFu);

    int16_t Y1  = (int16_t)( w1        & 0x3FFu);
    int16_t Cb1 = (int16_t)((w1 >> 10) & 0x3FFu);
    int16_t Y2  = (int16_t)((w1 >> 20) & 0x3FFu);

    int16_t Cr1 = (int16_t)( w2        & 0x3FFu);
    int16_t Y3  = (int16_t)((w2 >> 10) & 0x3FFu);
    int16_t Cb2 = (int16_t)((w2 >> 20) & 0x3FFu);

    int16_t Y4  = (int16_t)( w3        & 0x3FFu);
    int16_t Cr2 = (int16_t)((w3 >> 10) & 0x3FFu);
    int16_t Y5  = (int16_t)((w3 >> 20) & 0x3FFu);

    const int px    = col6 * 6;             // first luma column of this group
    const int cx     = col6 * 3;            // first chroma column of this group
    const int chroma_w = width / 2;
    int y_base = field_row * width    + px;
    int c_base = field_row * chroma_w + cx;

    // Bounded per sample, because the last group of a width like 2048 is partial. Writing the
    // full six would run into the next row -- which is why the old kernel skipped the group
    // entirely rather than clamping, and why the tail was left to a memset that never came.
    const int16_t yv[6] = {Y0, Y1, Y2, Y3, Y4, Y5};
    for (int i = 0; i < 6; ++i)
        if (px + i < width)
            d_y[y_base + i] = yv[i];

    const int16_t cbv[3] = {Cb0, Cb1, Cb2};
    const int16_t crv[3] = {Cr0, Cr1, Cr2};
    for (int i = 0; i < 3; ++i) {
        if (cx + i < chroma_w) {
            d_cb[c_base + i] = cbv[i];
            d_cr[c_base + i] = crv[i];
        }
    }
}

inline cudaError_t launch_v210_unpack_field(
    const uint32_t *d_v210,
    int16_t *d_y, int16_t *d_cb, int16_t *d_cr,
    int width, int full_height,
    int field, cudaStream_t stream)
{
    // width need not be a multiple of 6, and no longer needs the caller to pre-zero anything:
    // the kernel covers every column, bounds-checking the partial last group.
    // CEIL, and it MUST match `groups_per_row` in the kernel. The grid size is computed here
    // and the group index is recomputed there, so a floor here and a ceil there means the tail
    // group simply never gets a thread -- the kernel's bounds checks would be dead code and the
    // edge columns would still be missed. Two places, one number.
    const int field_height = full_height / 2;
    const int total_groups = ((width + 5) / 6) * field_height;
    int threads = 128;
    int blocks  = (total_groups + threads - 1) / threads;
    k_v210_unpack_field<<<blocks, threads, 0, stream>>>(
        d_v210, d_y, d_cb, d_cr, width, full_height, field);
    return cudaGetLastError();
}

// Convenience launcher: handles grid/block sizing.
//
// FLOOR here, unlike the field launcher above, and deliberately: `prores_encode_frame` still
// zeroes the planes before calling this one, so its tail columns come out black rather than
// stale. The progressive ENCODE path no longer uses this route at all -- it converts straight
// to planes -- so this remains only for the V210 bypass consumer, which supplies real V210 and
// relies on that memset. Left alone rather than "fixed" for consistency: changing it would mean
// re-verifying the bypass consumer, and the memset already makes it correct.
// width need not be a multiple of 6 — edge pixels stay pre-zeroed by caller.
inline cudaError_t launch_v210_unpack(
    const uint32_t *d_v210,
    int16_t        *d_y,
    int16_t        *d_cb,
    int16_t        *d_cr,
    int width, int height,
    cudaStream_t stream)
{
    int total_groups = (width / 6) * height;
    int threads = 128;
    int blocks  = (total_groups + threads - 1) / threads;
    k_v210_unpack<<<blocks, threads, 0, stream>>>(
        d_v210, d_y, d_cb, d_cr, width, height);
    return cudaGetLastError();
}
