/*
 * Copyright (c) 2026 CasparCG Contributors
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
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

// ---------------------------------------------------------------------------
// In-place red/blue exchange on a packed 8888 buffer.
//
// WHY THIS EXISTS RATHER THAN A FLAG ON THE CONVERSION KERNELS. Three different
// kernels read `d_bgra_` -- `launch_bgra_to_v210` for progressive 422,
// `prores_encode_frame_444` for 4444, and `launch_bgra8_to_field422p10` for
// interlaced -- so a byte-order flag would have to be threaded through all
// three and every future consumer would have to know about it. Doing the
// exchange once, at the point where the buffer is filled, keeps `d_bgra_`
// meaning exactly one thing: BGRA, whatever route filled it.
//
// The route that needs it is the GPU-direct one. The mixer's OpenGL attachment
// is GL_RGBA8 with an external format of GL_BGRA, so GL swizzles on every host
// transfer: the host buffer a readback produces and the texture's own bytes are
// NOT in the same order. `cudaMemcpy2DFromArray` copies the bytes untouched, so
// the GPU path and the host path deliver opposite orders to the same kernel.
//
// Measured 2026-08-22, the first time the GPU-direct path had ever actually
// run: mean 166.31 LSB against the host path, with only 5% of the disagreement
// at a chroma edge -- flat areas wrong everywhere, which is the signature of an
// exchange rather than of a resampling difference.
// ---------------------------------------------------------------------------
__global__ void k_swap_rb_8888(uint8_t* __restrict__ d_px, int width, int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    // uchar4 rather than three byte accesses: one 32-bit load and one store per
    // pixel, which is what keeps this off the critical path at 4K.
    uint8_t* p = d_px + ((size_t)y * width + x) * 4;
    const uint8_t t = p[0];
    p[0]            = p[2];
    p[2]            = t;
}

static inline cudaError_t launch_swap_rb_8888(uint8_t*     d_px,
                                              int          width,
                                              int          height,
                                              cudaStream_t stream)
{
    if (width <= 0 || height <= 0)
        return cudaErrorInvalidValue;

    dim3 block(32, 8);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    k_swap_rb_8888<<<grid, block, 0, stream>>>(d_px, width, height);
    return cudaGetLastError();
}
