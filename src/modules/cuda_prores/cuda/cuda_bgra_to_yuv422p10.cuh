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

// cuda_bgra_to_yuv422p10.cuh
// BGRA8 -> planar YUV422P10, in ONE pass, for the progressive ProRes 422 encode path.
//
// WHAT THIS REPLACES, and why it is a deletion rather than an optimisation. That path used to
// go BGRA -> V210 -> planes: `k_bgra_to_v210` packed the frame into V210 and `k_v210_unpack`
// immediately took it apart again into exactly the planes the DCT wanted. Nothing read the V210
// in between. Per 1080p frame that cost a 5.3 MB write, a 5.3 MB read, a 5.3 MB allocation, and
// three full-plane `cudaMemsetAsync` -- about 19 MB of traffic to arrive where a single pass
// arrives directly.
//
// The memsets are the part worth explaining, because they were a workaround rather than waste.
// `k_bgra_to_v210` sizes its rows with CEIL division, `(width + 5) / 6` groups, while
// `k_v210_unpack` reads back with FLOOR, `width / 6` -- so for any width not divisible by six
// the last `width % 6` luma samples were never written, and the planes had to be zeroed first
// to keep stale VRAM out of the picture. 1280 and 2048 and 4096 are all such widths; 1920 and
// 3840 are not, which is why it never showed. Indexing per pixel PAIR, as this kernel does,
// covers every column by construction and needs no pre-zeroing.
//
// This is deliberately NOT built on a kernel-fusion framework. `docs/plans/FKL_INTEGRATION_ANALYSIS.md`
// recommended exactly this change in 2025 as the thing to do WITHOUT one -- a two-kernel
// element-wise chain collapsing into one is a kernel you write, not a dependency you take.
//
// Identical arithmetic to the two kernels it replaces: `bgra8_to_ycbcr10` per pixel, then
// `(a + b + 1) >> 1` on each chroma pair. V210 carries 10-bit samples exactly, so the old
// round trip was lossless and this output is bit-identical -- which is what makes
// `encode-parity` a real check on it rather than an approximation.
//
// Output, matching what the DCT stage already expects:
//   d_y  [width       x height] int16_t  Y  [64..940]
//   d_cb [(width / 2) x height] int16_t  Cb [64..960]
//   d_cr [(width / 2) x height] int16_t  Cr [64..960]
// ---------------------------------------------------------------------------
#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include "cuda_bgra_to_v210.cuh"  // reuse bgra8_to_ycbcr10()

// ---------------------------------------------------------------------------
// One thread per horizontal pixel PAIR: two Y samples, one Cb, one Cr.
// ---------------------------------------------------------------------------
__global__ void k_bgra8_to_yuv422p10(
    const uint8_t * __restrict__ d_bgra,  // BGRA8, row-major, stride = width * 4
    int16_t       * __restrict__ d_y,     // Y  (width       x height)
    int16_t       * __restrict__ d_cb,    // Cb ((width / 2) x height)
    int16_t       * __restrict__ d_cr,    // Cr ((width / 2) x height)
    int width,
    int height)
{
    const int x2 = blockIdx.x * blockDim.x + threadIdx.x;  // 0 .. width/2 - 1
    const int y  = blockIdx.y * blockDim.y + threadIdx.y;  // 0 .. height - 1
    if (x2 >= width / 2 || y >= height)
        return;

    const uint8_t *row = d_bgra + (size_t)y * width * 4;
    const uint8_t *p0  = row + (size_t)x2 * 8;   // pixel 2*x2
    const uint8_t *p1  = p0 + 4;                 // pixel 2*x2 + 1

    // BGRA in memory: byte 0 = B, 1 = G, 2 = R, 3 = A. Alpha is ignored: 422 carries none.
    int Y0, Cb0, Cr0, Y1, Cb1, Cr1;
    bgra8_to_ycbcr10(p0[2], p0[1], p0[0], Y0, Cb0, Cr0);
    bgra8_to_ycbcr10(p1[2], p1[1], p1[0], Y1, Cb1, Cr1);

    d_y[(size_t)y * width + x2 * 2    ] = (int16_t)Y0;
    d_y[(size_t)y * width + x2 * 2 + 1] = (int16_t)Y1;
    // 4:2:2 horizontal subsampling, averaged the same way the V210 packer averaged it.
    d_cb[(size_t)y * (width / 2) + x2] = (int16_t)((Cb0 + Cb1 + 1) >> 1);
    d_cr[(size_t)y * (width / 2) + x2] = (int16_t)((Cr0 + Cr1 + 1) >> 1);
}

// ---------------------------------------------------------------------------
inline cudaError_t launch_bgra8_to_yuv422p10(
    const uint8_t *d_bgra,
    int16_t       *d_y,
    int16_t       *d_cb,
    int16_t       *d_cr,
    int            width,
    int            height,
    cudaStream_t   stream)
{
    if (!d_bgra || !d_y || !d_cb || !d_cr || width <= 0 || height <= 0)
        return cudaErrorInvalidValue;

    const dim3 block(32, 8);
    const dim3 grid((unsigned)(((width / 2) + block.x - 1) / block.x),
                    (unsigned)((height + block.y - 1) / block.y));
    k_bgra8_to_yuv422p10<<<grid, block, 0, stream>>>(d_bgra, d_y, d_cb, d_cr, width, height);
    return cudaGetLastError();
}
