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

// cuda_prores_to_bgra16.cuh
// CUDA kernel: YCbCr 4:2:2 10-bit limited-range → BGRA 16-bit full-range.
//
// Input:
//   d_y   [height × width]          int16_t  luma    [4, 1019]
//   d_cb  [height × (width/2)]      int16_t  Cb      [4, 1019]  neutral=512
//   d_cr  [height × (width/2)]      int16_t  Cr      [4, 1019]  neutral=512
//
// Output:
//   d_bgra16 [height × width × 4]   uint16_t  BGRA, 16 bits per channel
//
// Color matrices:
//   BT.709  (color_matrix == 1, or any non-BT.2020 value)
//   BT.2020 (color_matrix == 9)
//
// Signal convention:
//   Limited range 10-bit:  Y ∈ [64,940], Cb/Cr swing ±448 from neutral 512
//   Output: full-range 16-bit mapped from [4,1019] to [0,65535]
//
// The output is in PQ signal space for HDR content (the decoder does NOT
// apply the PQ EOTF — that is the downstream GLSL shader's responsibility).
//
// Fixed-point arithmetic (Q16):
//   All multiplications use 32-bit operands producing 64-bit intermediates.
//   Final clip to [0, 65535].
//
// Subsampling: 4:2:2 horizontal.  Each pair of luma pixels shares one
// Cb/Cr pair.  Thread tx handles luma pixel x; Cb/Cr at index x/2.
// ---------------------------------------------------------------------------
#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

// ─── Fixed-point YCbCr→RGB coefficients ─────────────────────────────────────
// These convert limited-range 10-bit YCbCr → full-range 16-bit RGB.
//
// Step 1: Normalise
//   y_n  = Y  - 64        [0 .. 876]  → maps to [0, 65535] via scale 74.81
//   cb_n = Cb - 512       [−448 .. +448]
//   cr_n = Cr - 512       [−448 .. +448]
//
// The exact scale from limited 10-bit Y to 16-bit:
//   65535 / 876 = 74.81...  →  Q16: round(65535*65536/876) = 4,902,781
//
// For chroma, the full swing range is 896 (= 224*4, from code 64 to 960).
// Chroma normalisation: E_Cr = cr_n / 896 ∈ [−0.5, +0.5].
// The chroma scale to 16-bit is 65535 / 896 = 73.14...
// We combine this with the BT matrix coefficients in a single Q16 multiply.
//
// BT.2020 non-constant luminance (ITU-R BT.2020):
//   R_full = 65535/876 * y_n + 65535/896 * 1.4746 * cr_n
//          = 74.81 * y_n + 107.86 * cr_n
//   G_full = 74.81 * y_n - 65535/896 * 0.1645 * cb_n - 65535/896 * 0.5713 * cr_n
//          = 74.81 * y_n - 12.03 * cb_n - 41.78 * cr_n
//   B_full = 74.81 * y_n + 65535/896 * 1.8814 * cb_n
//          = 74.81 * y_n + 137.61 * cb_n
//
// As Q16 integers (multiply × 65536, round):
//   BT.2020:
#define BT2020_Y_SCALE   4902781  // round(65535*65536/876)
#define BT2020_CR_TO_R   7074432  // 107.93 * 65536
#define BT2020_CB_TO_G   -789280  // -12.04 * 65536
#define BT2020_CR_TO_G  -2739776  // -41.78 * 65536
#define BT2020_CB_TO_B   9027584  // 137.73 * 65536
//
// BT.709 (ITU-R BT.709):
//   Kr=0.2126, Kg=0.7152, Kb=0.0722
//   R_full = 74.81*y_n + 65535/896*1.5748*cr_n
//          = 74.81*y_n + 115.18*cr_n
//   G_full = 74.81*y_n - 65535/896*0.1873*cb_n - 65535/896*0.4681*cr_n
//          = 74.81*y_n - 13.70*cb_n - 34.24*cr_n
//   B_full = 74.81*y_n + 65535/896*1.8556*cb_n
//          = 74.81*y_n + 135.72*cb_n
//   As Q16:
#define BT709_Y_SCALE   4902781  // same Y scale
#define BT709_CR_TO_R   7548896  // 115.18 * 65536
#define BT709_CB_TO_G   -897792  // -13.70 * 65536
#define BT709_CR_TO_G  -2245632  // -34.27 * 65536
#define BT709_CB_TO_B   8901216  // 135.82 * 65536
//
// BT.601 (ITU-R BT.601 / SMPTE-C, CICP matrix 5 or 6):
//   Kr=0.299, Kg=0.587, Kb=0.114
//   R_full = 74.81*y_n + 65535/896*1.402*cr_n
//          = 74.81*y_n + 102.55*cr_n
//   G_full = 74.81*y_n - 65535/896*0.344136*cb_n - 65535/896*0.714136*cr_n
//          = 74.81*y_n - 25.17*cb_n - 52.24*cr_n
//   B_full = 74.81*y_n + 65535/896*1.772*cb_n
//          = 74.81*y_n + 129.60*cb_n
//   As Q16:
#define BT601_Y_SCALE   4902781  // same Y scale
#define BT601_CR_TO_R   6720256  // 102.55 * 65536
#define BT601_CB_TO_G  -1649152  // -25.17 * 65536
#define BT601_CR_TO_G  -3424896  // -52.26 * 65536
#define BT601_CB_TO_B   8495616  // 129.60 * 65536

// ─── Kernel ──────────────────────────────────────────────────────────────────
//
// Grid: (ceil(width/32) × ceil(height/16))  blocks of (32 × 16) threads
//   One thread per luma pixel.
// ---------------------------------------------------------------------------
__global__ void k_ycbcr422p10_to_bgra16(
    const int16_t* __restrict__ d_y,
    const int16_t* __restrict__ d_cb,
    const int16_t* __restrict__ d_cr,
    uint16_t*      __restrict__ d_bgra16,
    int width,
    int height,
    int color_matrix)   // 9=BT.2020, 5/6=BT.601, else BT.709
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const int luma_idx  = y * width + x;
    const int chroma_x  = x >> 1;   // 4:2:2 horizontal sub-sampling
    const int chroma_idx = y * (width >> 1) + chroma_x;

    // Load and level-shift.
    int iy  = (int)d_y [luma_idx]   - 64;
    int icb = (int)d_cb[chroma_idx] - 512;
    int icr = (int)d_cr[chroma_idx] - 512;

    int64_t r, g, b;

    if (color_matrix == 9) {
        // BT.2020
        int64_t y64  = (int64_t)iy  * BT2020_Y_SCALE;
        r = (y64 + (int64_t)icr * BT2020_CR_TO_R) >> 16;
        g = (y64 + (int64_t)icb * BT2020_CB_TO_G + (int64_t)icr * BT2020_CR_TO_G) >> 16;
        b = (y64 + (int64_t)icb * BT2020_CB_TO_B) >> 16;
    } else if (color_matrix == 5 || color_matrix == 6) {
        // BT.601 (5=EBU/PAL, 6=SMPTE-C/NTSC)
        int64_t y64  = (int64_t)iy  * BT601_Y_SCALE;
        r = (y64 + (int64_t)icr * BT601_CR_TO_R) >> 16;
        g = (y64 + (int64_t)icb * BT601_CB_TO_G + (int64_t)icr * BT601_CR_TO_G) >> 16;
        b = (y64 + (int64_t)icb * BT601_CB_TO_B) >> 16;
    } else {
        // BT.709 (default — covers code 1, unspecified 0, and any unknown value)
        int64_t y64  = (int64_t)iy  * BT709_Y_SCALE;
        r = (y64 + (int64_t)icr * BT709_CR_TO_R) >> 16;
        g = (y64 + (int64_t)icb * BT709_CB_TO_G + (int64_t)icr * BT709_CR_TO_G) >> 16;
        b = (y64 + (int64_t)icb * BT709_CB_TO_B) >> 16;
    }

    // Clip to [0, 65535].
    auto clip16 = [](int64_t v) -> uint16_t {
        return (uint16_t)(v < 0 ? 0 : (v > 65535 ? 65535 : v));
    };

    uint16_t* dst = d_bgra16 + luma_idx * 4;
    dst[0] = clip16(b);
    dst[1] = clip16(g);
    dst[2] = clip16(r);
    dst[3] = 0xFFFFu;  // full alpha
}

// Launcher.
inline cudaError_t launch_ycbcr_to_bgra16(
    const int16_t* d_y,
    const int16_t* d_cb,
    const int16_t* d_cr,
    uint16_t*      d_bgra16,
    int            width,
    int            height,
    int            color_matrix,
    cudaStream_t   stream)
{
    dim3 threads(32, 16);
    dim3 blocks((width  + 31) / 32,
                (height + 15) / 16);
    k_ycbcr422p10_to_bgra16<<<blocks, threads, 0, stream>>>(
        d_y, d_cb, d_cr, d_bgra16, width, height, color_matrix);
    return cudaGetLastError();
}

// ─── ProRes 4444 kernel: YCbCr 4:4:4 10-bit + alpha → BGRA 16-bit ──────────
//
// Differences from the 422 kernel:
//   • No chroma subsampling: each pixel reads its own Cb/Cr (chroma_idx = luma_idx)
//   • Real alpha plane: d_alpha[luma_idx] in full-range 10-bit [0, 1023]
//     (decoded by CPU unpack_alpha; NOT limited-range, NO level shift).
//     Alpha is expanded to 16-bit by bit-replication: (ia<<6)|(ia>>4).
// ---------------------------------------------------------------------------
__global__ void k_ycbcr444p10_to_bgra16(
    const int16_t* __restrict__ d_y,
    const int16_t* __restrict__ d_cb,
    const int16_t* __restrict__ d_cr,
    const int16_t* __restrict__ d_alpha,
    uint16_t*      __restrict__ d_bgra16,
    int width,
    int height,
    int color_matrix)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const int luma_idx = y * width + x;

    // Load and level-shift.  4:4:4 means chroma_idx == luma_idx.
    int iy  = (int)d_y  [luma_idx] - 64;
    int icb = (int)d_cb [luma_idx] - 512;
    int icr = (int)d_cr [luma_idx] - 512;
    int ia  = (int)d_alpha[luma_idx];   // full-range 10-bit [0, 1023]

    int64_t r, g, b;

    if (color_matrix == 9) {
        int64_t y64 = (int64_t)iy * BT2020_Y_SCALE;
        r = (y64 + (int64_t)icr * BT2020_CR_TO_R) >> 16;
        g = (y64 + (int64_t)icb * BT2020_CB_TO_G + (int64_t)icr * BT2020_CR_TO_G) >> 16;
        b = (y64 + (int64_t)icb * BT2020_CB_TO_B) >> 16;
    } else if (color_matrix == 5 || color_matrix == 6) {
        int64_t y64 = (int64_t)iy * BT601_Y_SCALE;
        r = (y64 + (int64_t)icr * BT601_CR_TO_R) >> 16;
        g = (y64 + (int64_t)icb * BT601_CB_TO_G + (int64_t)icr * BT601_CR_TO_G) >> 16;
        b = (y64 + (int64_t)icb * BT601_CB_TO_B) >> 16;
    } else {
        int64_t y64 = (int64_t)iy * BT709_Y_SCALE;
        r = (y64 + (int64_t)icr * BT709_CR_TO_R) >> 16;
        g = (y64 + (int64_t)icb * BT709_CB_TO_G + (int64_t)icr * BT709_CR_TO_G) >> 16;
        b = (y64 + (int64_t)icb * BT709_CB_TO_B) >> 16;
    }

    // Expand 10-bit full-range alpha → 16-bit by bit-replication.
    uint16_t a16 = (uint16_t)((ia << 6) | (ia >> 4));

    auto clip16 = [](int64_t v) -> uint16_t {
        return (uint16_t)(v < 0 ? 0 : (v > 65535 ? 65535 : v));
    };

    uint16_t* dst = d_bgra16 + luma_idx * 4;
    dst[0] = clip16(b);
    dst[1] = clip16(g);
    dst[2] = clip16(r);
    dst[3] = a16;
}

// Launcher for ProRes 4444 (4:4:4 + real alpha).
inline cudaError_t launch_ycbcr444_to_bgra16(
    const int16_t* d_y,
    const int16_t* d_cb,
    const int16_t* d_cr,
    const int16_t* d_alpha,
    uint16_t*      d_bgra16,
    int            width,
    int            height,
    int            color_matrix,
    cudaStream_t   stream)
{
    dim3 threads(32, 16);
    dim3 blocks((width  + 31) / 32,
                (height + 15) / 16);
    k_ycbcr444p10_to_bgra16<<<blocks, threads, 0, stream>>>(
        d_y, d_cb, d_cr, d_alpha, d_bgra16, width, height, color_matrix);
    return cudaGetLastError();
}
