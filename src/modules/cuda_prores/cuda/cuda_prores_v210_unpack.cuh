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

// Convenience launcher: handles grid/block sizing and precondition check.
// Returns cudaErrorInvalidValue if width is not a multiple of 6.
inline cudaError_t launch_v210_unpack(
    const uint32_t *d_v210,
    int16_t        *d_y,
    int16_t        *d_cb,
    int16_t        *d_cr,
    int width, int height,
    cudaStream_t stream)
{
    if (width % 6 != 0) return cudaErrorInvalidValue;
    int total_groups = (width / 6) * height;
    int threads = 128;
    int blocks  = (total_groups + threads - 1) / threads;
    k_v210_unpack<<<blocks, threads, 0, stream>>>(
        d_v210, d_y, d_cb, d_cr, width, height);
    return cudaGetLastError();
}
