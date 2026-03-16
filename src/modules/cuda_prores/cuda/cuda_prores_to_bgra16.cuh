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
//   65535 / 876 = 74.81...  ≈  Q16: 65535*65536/876 = 4,898,641/65536 ≈ 74.81 → Q16 = 4,898,560
//
// For luma-to-R,G,B we combine the Y scale with the matrix coefficients
// in a single Q16 multiplication.
//
// BT.2020 non-constant luminance (ITU-R BT.2020):
//   R_full = 65535/876 * (Y - 64) + 65535/448 * 1.4746 * (Cr - 512) / 2
//     Note: the chroma scale is 65535/(448) because Cb/Cr range is ±448 for limited range.
//   R_full = 74.81 * y_n + 65535/448 * 1.4746 * cr_n
//           = 74.81 * y_n + 215.85 * cr_n
//   G_full = 74.81 * y_n - 65535/448 * 0.1645 * cb_n - 65535/448 * 0.5713 * cr_n
//           = 74.81 * y_n - 24.09 * cb_n - 83.65 * cr_n
//   B_full = 74.81 * y_n + 65535/448 * 1.8814 * cb_n
//           = 74.81 * y_n + 275.46 * cb_n
//
// As Q16 integers (multiply × 65536, round):
//   BT.2020:
#define BT2020_Y_SCALE   4899072  // 74.81 * 65536
#define BT2020_CR_TO_R  14148864  // 215.85 * 65536
#define BT2020_CB_TO_G  -1578560  // -24.09 * 65536  (note: positive numerator in formula: -0.1645*(65535/448))
#define BT2020_CR_TO_G  -5479552  // -83.65 * 65536
#define BT2020_CB_TO_B  18055168  // 275.46 * 65536
//
// BT.709 (ITU-R BT.709):
//   Kr=0.2126, Kg=0.7152, Kb=0.0722
//   R_full = 74.81*(Y-64) + 65535/448*1.5748*(Cr-512)
//          = 74.81*y + 230.36*cr
//   G_full = 74.81*y - 65535/448*0.1873*cb - 65535/448*0.4681*cr
//          = 74.81*y - 27.40*cb - 68.53*cr
//   B_full = 74.81*y + 65535/448*1.8556*cb
//          = 74.81*y + 271.64*cb
//   As Q16:
#define BT709_Y_SCALE   4899072  // same Y scale
#define BT709_CR_TO_R  15097792  // 230.36 * 65536
#define BT709_CB_TO_G  -1795584  // -27.40 * 65536
#define BT709_CR_TO_G  -4491264  // -68.53 * 65536
#define BT709_CB_TO_B  17802432  // 271.64 * 65536
//
// BT.601 (ITU-R BT.601 / SMPTE-C, CICP matrix 5 or 6):
//   Kr=0.299, Kg=0.587, Kb=0.114
//   R_full = 74.81*y + 65535/448*1.402*cr
//          = 74.81*y + 205.09*cr
//   G_full = 74.81*y - 65535/448*0.344136*cb - 65535/448*0.714136*cr
//          = 74.81*y - 50.33*cb - 104.47*cr
//   B_full = 74.81*y + 65535/448*1.772*cb
//          = 74.81*y + 259.21*cb
//   As Q16:
#define BT601_Y_SCALE   4899072  // same Y scale
#define BT601_CR_TO_R  13440512  // 205.09 * 65536
#define BT601_CB_TO_G  -3298304  // -50.33 * 65536
#define BT601_CR_TO_G  -6849792  // -104.47 * 65536
#define BT601_CB_TO_B  16991232  // 259.21 * 65536

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
