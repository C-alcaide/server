// cuda_bgra_to_yuva444p10.cuh
// Direct BGRA8 → planar YCbCrA 4:4:4:4 10-bit conversion for ProRes 4444.
//
// Unlike the 422 pipeline (BGRA → V210 → unpack → YCbCr422P10), this kernel
// produces full-resolution Cb/Cr planes (no horizontal subsampling) plus an
// optional 10-bit alpha plane for the ProRes 4444 / 4444 XQ encoder.
//
// Color space: BT.709 full-range RGB → limited-range 10-bit YCbCr using the
// same fixed-point coefficients as cuda_bgra_to_v210.cuh.
//
// Alpha encoding: 8-bit [0..255] → 10-bit [0..1023] via bit replication:
//   A10 = (A8 << 2) | (A8 >> 6)   →  0 → 0,  255 → 1023  exactly.
//
// Output planes (all int16_t, row-major, stride = width):
//   d_y     [height * width]   Y,  range [64..940]
//   d_cb    [height * width]   Cb, range [64..960]  (full res — no subsampling)
//   d_cr    [height * width]   Cr, range [64..960]
//   d_alpha [height * width]   α,  range [0..1023]  (pass nullptr to skip)
#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include "cuda_bgra_to_v210.cuh"    // reuse bgra8_to_ycbcr10()

// ---------------------------------------------------------------------------
// Main kernel: one thread per pixel
// ---------------------------------------------------------------------------
__global__ void k_bgra_to_yuva444p10(
    const uint8_t * __restrict__ d_bgra,
    int16_t       * __restrict__ d_y,
    int16_t       * __restrict__ d_cb,
    int16_t       * __restrict__ d_cr,
    int16_t       * __restrict__ d_alpha,   // may be nullptr
    int width, int height)
{
    const int px  = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y;
    if (row >= height || px >= width) return;

    const uint8_t *p = d_bgra + ((size_t)row * width + px) * 4;
    const uint8_t B = p[0], G = p[1], R = p[2], A = p[3];

    int Y, Cb, Cr;
    bgra8_to_ycbcr10(R, G, B, Y, Cb, Cr);

    const size_t idx = (size_t)row * width + px;
    d_y [idx] = (int16_t)Y;
    d_cb[idx] = (int16_t)Cb;
    d_cr[idx] = (int16_t)Cr;
    if (d_alpha)
        d_alpha[idx] = (int16_t)((A << 2) | (A >> 6));  // 8-bit → 10-bit expansion
}

// ---------------------------------------------------------------------------
// Host-side launcher
// ---------------------------------------------------------------------------
static inline cudaError_t launch_bgra_to_yuva444p10(
    const uint8_t *d_bgra,
    int16_t       *d_y,
    int16_t       *d_cb,
    int16_t       *d_cr,
    int16_t       *d_alpha,   // may be nullptr — pass nullptr to skip alpha plane
    int            width,
    int            height,
    cudaStream_t   stream)
{
    dim3 block(32, 1);
    dim3 grid((width + 31) / 32, height);
    k_bgra_to_yuva444p10<<<grid, block, 0, stream>>>(
        d_bgra, d_y, d_cb, d_cr, d_alpha, width, height);
    return cudaGetLastError();
}
