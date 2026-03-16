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

// cuda_bgra_to_field422p10.cuh
// Extract one real field from a yadif-processed BGRA frame into half-height
// planar YUV422P10 (int16_t planes), for the CasparCG full-stack ProRes consumer.
//
// yadif=mode=send_field_nospatial delivers a full 1920×1080 BGRA frame per field.
// For field::a (TFF top field): real source rows are at even indices (0,2,4,...).
// For field::b (TFF bottom field): real source rows are at odd  indices (1,3,5,...).
// For BFF: field::a has real data at odd rows, field::b at even rows.
//
// The kernel reads every other row (stride 2) according to the field parity
// and converts BGRA to 10-bit limited-range YCbCr 4:2:2, reusing the same
// BT.709 coefficients as cuda_bgra_to_v210.cuh.
//
// Output: field_height = full_height/2 rows, planar (not V210):
//   d_y  [width        × field_height] int16_t  Y  luma   [64..940]
//   d_cb [(width/2)    × field_height] int16_t  Cb chroma [64..960]
//   d_cr [(width/2)    × field_height] int16_t  Cr chroma [64..960]
//
// Also provides a BGRA64LE (16-bit channels) variant for channels configured
// with bit_depth == bit16.  Uses int64 fixed-point to preserve all 16 bits of
// each channel when computing the 10-bit limited-range output.
// ─────────────────────────────────────────────────────────────────────────────
#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include "cuda_bgra_to_v210.cuh"  // reuse bgra8_to_ycbcr10()

// ---------------------------------------------------------------------------
// BGRA8 → half-height YUV422P10
// One thread per horizontal pixel pair → one Cb, one Cr, two Y samples.
// ---------------------------------------------------------------------------
__global__ void k_bgra8_to_field422p10(
    const uint8_t * __restrict__ d_bgra,  // full-height BGRA8 (width × full_height × 4)
    int16_t       * __restrict__ d_y,     // output Y  (width       × field_height)
    int16_t       * __restrict__ d_cb,    // output Cb ((width/2)   × field_height)
    int16_t       * __restrict__ d_cr,    // output Cr ((width/2)   × field_height)
    int width,
    int full_height,
    int field_parity)  // 0=even src rows (top field), 1=odd src rows (bottom field)
{
    const int x2  = blockIdx.x * blockDim.x + threadIdx.x;  // 0 .. width/2-1
    const int fy  = blockIdx.y * blockDim.y + threadIdx.y;  // 0 .. field_height-1
    const int field_height = full_height / 2;
    if (x2 >= width / 2 || fy >= field_height) return;

    // Map field row → full-frame row (same stride pattern as k_v210_unpack_field)
    const int full_row = fy * 2 + field_parity;
    const uint8_t *row = d_bgra + (size_t)full_row * width * 4;
    const uint8_t *p0  = row + x2 * 8;    // pixel at (x2*2)
    const uint8_t *p1  = p0 + 4;          // pixel at (x2*2 + 1)

    // BGRA order: byte 0=B, 1=G, 2=R, 3=A
    int Y0, Cb0, Cr0, Y1, Cb1, Cr1;
    bgra8_to_ycbcr10(p0[2], p0[1], p0[0], Y0, Cb0, Cr0);
    bgra8_to_ycbcr10(p1[2], p1[1], p1[0], Y1, Cb1, Cr1);

    d_y[fy * width        + x2 * 2    ] = (int16_t)Y0;
    d_y[fy * width        + x2 * 2 + 1] = (int16_t)Y1;
    // 4:2:2 horizontal subsampling: average the pair's Cb/Cr
    d_cb[fy * (width / 2) + x2] = (int16_t)((Cb0 + Cb1 + 1) >> 1);
    d_cr[fy * (width / 2) + x2] = (int16_t)((Cr0 + Cr1 + 1) >> 1);
}

// ---------------------------------------------------------------------------
// BGRA64LE → half-height YUV422P10
// Each BGRA channel is a uint16_t in [0, 65535]; layout per pixel: B,G,R,A.
// Uses int64 fixed-point (coefficients scaled by 2^25 instead of 2^17) to
// preserve all significant bits of the 16-bit input.
// ---------------------------------------------------------------------------
__global__ void k_bgra64le_to_field422p10(
    const uint16_t * __restrict__ d_bgra64,  // full-height BGRA64LE (width × full_height × 4 × uint16)
    int16_t        * __restrict__ d_y,
    int16_t        * __restrict__ d_cb,
    int16_t        * __restrict__ d_cr,
    int width,
    int full_height,
    int field_parity)
{
    const int x2 = blockIdx.x * blockDim.x + threadIdx.x;
    const int fy = blockIdx.y * blockDim.y + threadIdx.y;
    const int field_height = full_height / 2;
    if (x2 >= width / 2 || fy >= field_height) return;

    const int full_row = fy * 2 + field_parity;
    // Each pixel is 4 × uint16_t (BGRA order: index 0=B, 1=G, 2=R, 3=A)
    const uint16_t *row = d_bgra64 + (size_t)full_row * width * 4;
    const uint16_t *p0  = row + x2 * 8;    // pixel at (x2*2)
    const uint16_t *p1  = p0 + 4;          // pixel at (x2*2 + 1)

    // BT.709 full-range [0..65535] → 10-bit limited [64..940/960]
    // Same matrix as bgra8_to_ycbcr10 but coefficients scaled for 16-bit input:
    //   shift = 17 + 8 = 25  (because 65535 / 255 ≈ 256, so 8 extra shift bits)
    //   Y  = 64  + (( 95787*R + 322497*G +  32557*B) >> 25)
    //   Cb = 512 + ((-52785*R - 177620*G + 230389*B) >> 25)
    //   Cr = 512 + (( 230389*R - 209206*G -  21123*B) >> 25)
    auto conv16 = [](uint16_t pB, uint16_t pG, uint16_t pR,
                     int16_t &Y, int16_t &Cb, int16_t &Cr) {
        long long R = pR, G = pG, B = pB;
        int y  = 64  + (int)(( 95787LL * R + 322497LL * G +  32557LL * B) >> 25);
        int cb = 512 + (int)((-52785LL * R - 177620LL * G + 230389LL * B) >> 25);
        int cr = 512 + (int)(( 230389LL * R - 209206LL * G -  21123LL * B) >> 25);
        Y  = (int16_t)(y  < 64 ? 64 : (y  > 940 ? 940 : y));
        Cb = (int16_t)(cb < 64 ? 64 : (cb > 960 ? 960 : cb));
        Cr = (int16_t)(cr < 64 ? 64 : (cr > 960 ? 960 : cr));
    };

    int16_t Y0, Cb0, Cr0, Y1, Cb1, Cr1;
    conv16(p0[0], p0[1], p0[2], Y0, Cb0, Cr0);  // p0[0]=B, p0[1]=G, p0[2]=R
    conv16(p1[0], p1[1], p1[2], Y1, Cb1, Cr1);

    d_y[fy * width        + x2 * 2    ] = Y0;
    d_y[fy * width        + x2 * 2 + 1] = Y1;
    d_cb[fy * (width / 2) + x2] = (int16_t)((Cb0 + Cb1 + 1) >> 1);
    d_cr[fy * (width / 2) + x2] = (int16_t)((Cr0 + Cr1 + 1) >> 1);
}

// ---------------------------------------------------------------------------
// Launchers
// ---------------------------------------------------------------------------
inline cudaError_t launch_bgra8_to_field422p10(
    const uint8_t *d_bgra,
    int16_t *d_y, int16_t *d_cb, int16_t *d_cr,
    int width, int full_height, int field_parity,
    cudaStream_t stream)
{
    dim3 block(32, 8);
    dim3 grid((width / 2 + block.x - 1) / block.x,
              (full_height / 2 + block.y - 1) / block.y);
    k_bgra8_to_field422p10<<<grid, block, 0, stream>>>(
        d_bgra, d_y, d_cb, d_cr, width, full_height, field_parity);
    return cudaGetLastError();
}

inline cudaError_t launch_bgra64le_to_field422p10(
    const uint16_t *d_bgra64,
    int16_t *d_y, int16_t *d_cb, int16_t *d_cr,
    int width, int full_height, int field_parity,
    cudaStream_t stream)
{
    dim3 block(32, 8);
    dim3 grid((width / 2 + block.x - 1) / block.x,
              (full_height / 2 + block.y - 1) / block.y);
    k_bgra64le_to_field422p10<<<grid, block, 0, stream>>>(
        d_bgra64, d_y, d_cb, d_cr, width, full_height, field_parity);
    return cudaGetLastError();
}
