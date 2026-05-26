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
 * CUDA kernel: read VK texture (via surface object) and pack to V210.
 * Supports both 8-bit (BGRA8) and 16-bit (BGRA16) textures.
 * Performs sub-region extraction on GPU — only the configured DeckLink
 * viewport is read and packed, minimizing both GPU work and PCIe bandwidth.
 *
 * Color matrices: BT.709 (SDR) and BT.2020 (HDR).
 * V210 output: 10-bit limited-range studio-swing YCbCr 4:2:2.
 */
#pragma once
#include <cuda_runtime.h>
#include <cstdint>

// ---------------------------------------------------------------------------
// BT.709 full-range → 10-bit limited-range YCbCr (same as cuda_bgra_to_v210.cuh)
// ---------------------------------------------------------------------------
__device__ __forceinline__
void bgra16_to_ycbcr10_bt709(int R10, int G10, int B10,
                              int& Y, int& Cb, int& Cr)
{
    // Fixed-point (×2^20) BT.709 coefficients for 10-bit input
    // Y  = 64  + (0.2126*R + 0.7152*G + 0.0722*B) * 876/1023
    // Cb = 512 + (-0.1146*R - 0.3854*G + 0.5*B) * 896/1023
    // Cr = 512 + (0.5*R - 0.4542*G - 0.0458*B) * 896/1023
    Y  = 64  + ((222951 * R10 + 750098 * G10 + 75663 * B10) >> 20);
    Cb = 512 + ((-100459 * R10 - 337802 * G10 + 438223 * B10) >> 20);
    Cr = 512 + ((438223 * R10 - 398337 * G10 - 39908 * B10) >> 20);

    Y  = max(64, min(Y, 940));
    Cb = max(64, min(Cb, 960));
    Cr = max(64, min(Cr, 960));
}

// ---------------------------------------------------------------------------
// BT.2020 full-range → 10-bit limited-range YCbCr
// ---------------------------------------------------------------------------
__device__ __forceinline__
void bgra16_to_ycbcr10_bt2020(int R10, int G10, int B10,
                               int& Y, int& Cb, int& Cr)
{
    // BT.2020 coefficients: KR=0.2627, KG=0.6780, KB=0.0593
    Y  = 64  + ((275375 * R10 + 710743 * G10 + 62594 * B10) >> 20);
    Cb = 512 + ((-146420 * R10 - 377856 * G10 + 524288 * B10) >> 20);
    Cr = 512 + ((524288 * R10 - 482393 * G10 - 41857 * B10) >> 20);

    Y  = max(64, min(Y, 940));
    Cb = max(64, min(Cb, 960));
    Cr = max(64, min(Cr, 960));
}

// ---------------------------------------------------------------------------
// Main kernel: read VK texture surface, extract subregion, pack V210
// ---------------------------------------------------------------------------
// Each thread handles one V210 group (6 source pixels → 4 uint32 words).
// The surface is the VK mixer's render attachment (RGBA16 or RGBA8).
//
// Parameters:
//   surf       — CUDA surface object wrapping the imported VK texture
//   d_v210     — output V210 buffer (device or pinned host)
//   src_x/y    — subregion origin in the source texture
//   dst_w/h    — output dimensions (DeckLink frame)
//   src_stride — source texture width (for bounds clamping)
//   is_16bit   — true = RGBA16Unorm texture, false = RGBA8Unorm
//   use_bt2020 — true = BT.2020 matrix, false = BT.709
// ---------------------------------------------------------------------------
__global__ void k_vk_surface_to_v210(
    cudaSurfaceObject_t surf,
    uint32_t* __restrict__ d_v210,
    int src_x, int src_y,
    int dst_w, int dst_h,
    int src_w, int src_h,
    bool is_16bit,
    bool use_bt2020)
{
    const int group_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int row     = blockIdx.y;

    if (row >= dst_h) return;

    const int groups_per_row = (dst_w + 5) / 6;
    if (group_x >= groups_per_row) return;

    const int px_base = group_x * 6;
    const int sy      = src_y + row;

    // Load 6 pixels from the surface, converting to 10-bit
    int R[6], G[6], B[6];

    #pragma unroll
    for (int i = 0; i < 6; ++i) {
        const int sx = src_x + px_base + i;
        if (sx < src_w && sy < src_h && (px_base + i) < dst_w) {
            if (is_16bit) {
                ushort4 pixel;
                surf2Dread(&pixel, surf, sx * (int)sizeof(ushort4), sy);
                // 16-bit: shader writes RGBA directly (no .bgra swizzle).
                // pixel.x = Red, pixel.y = Green, pixel.z = Blue.
                R[i] = pixel.x >> 6;
                G[i] = pixel.y >> 6;
                B[i] = pixel.z >> 6;
            } else {
                uchar4 pixel;
                surf2Dread(&pixel, surf, sx * (int)sizeof(uchar4), sy);
                // Same BGRA convention for 8-bit: pixel.x = Blue, pixel.z = Red.
                R[i] = pixel.z << 2;
                G[i] = pixel.y << 2;
                B[i] = pixel.x << 2;
            }
        } else {
            R[i] = 0; G[i] = 0; B[i] = 0;
        }
    }

    // Convert to YCbCr
    int Y[6], Cb_raw[6], Cr_raw[6];
    if (use_bt2020) {
        #pragma unroll
        for (int i = 0; i < 6; ++i)
            bgra16_to_ycbcr10_bt2020(R[i], G[i], B[i], Y[i], Cb_raw[i], Cr_raw[i]);
    } else {
        #pragma unroll
        for (int i = 0; i < 6; ++i)
            bgra16_to_ycbcr10_bt709(R[i], G[i], B[i], Y[i], Cb_raw[i], Cr_raw[i]);
    }

    // 4:2:2 chroma subsampling: average pairs
    int Cb0 = (Cb_raw[0] + Cb_raw[1] + 1) >> 1;
    int Cr0 = (Cr_raw[0] + Cr_raw[1] + 1) >> 1;
    int Cb1 = (Cb_raw[2] + Cb_raw[3] + 1) >> 1;
    int Cr1 = (Cr_raw[2] + Cr_raw[3] + 1) >> 1;
    int Cb2 = (Cb_raw[4] + Cb_raw[5] + 1) >> 1;
    int Cr2 = (Cr_raw[4] + Cr_raw[5] + 1) >> 1;

    // Pack V210 words
    uint32_t w0 = (uint32_t)Cb0        | ((uint32_t)Y[0] << 10) | ((uint32_t)Cr0 << 20);
    uint32_t w1 = (uint32_t)Y[1]       | ((uint32_t)Cb1  << 10) | ((uint32_t)Y[2] << 20);
    uint32_t w2 = (uint32_t)Cr1        | ((uint32_t)Y[3] << 10) | ((uint32_t)Cb2 << 20);
    uint32_t w3 = (uint32_t)Y[4]       | ((uint32_t)Cr2  << 10) | ((uint32_t)Y[5] << 20);

    uint32_t* out = d_v210 + (size_t)row * groups_per_row * 4 + group_x * 4;
    out[0] = w0;
    out[1] = w1;
    out[2] = w2;
    out[3] = w3;
}

// ---------------------------------------------------------------------------
// Simpler kernel for BGRA8 subregion extraction (SDR BGRA decklink path)
// Reads subregion from VK surface, writes linear BGRA8 to output buffer.
// ---------------------------------------------------------------------------
__global__ void k_vk_surface_to_bgra8(
    cudaSurfaceObject_t surf,
    uint8_t* __restrict__ d_bgra,
    int src_x, int src_y,
    int dst_w, int dst_h,
    int src_w, int src_h)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dst_w || y >= dst_h) return;

    const int sx = src_x + x;
    const int sy = src_y + y;

    uchar4 pixel;
    if (sx < src_w && sy < src_h) {
        surf2Dread(&pixel, surf, sx * (int)sizeof(uchar4), sy);
    } else {
        pixel = make_uchar4(0, 0, 0, 255);
    }

    // Texture stores BGRA (mixer writes fragColor=col.bgra), DeckLink expects BGRA.
    // pixel.x = B, pixel.y = G, pixel.z = R — already in correct order for bmdFormat8BitBGRA.
    uint8_t* out = d_bgra + ((size_t)y * dst_w + x) * 4;
    out[0] = pixel.x; // B
    out[1] = pixel.y; // G
    out[2] = pixel.z; // R
    out[3] = pixel.w; // A
}

// ---------------------------------------------------------------------------
// Host-side launchers
// ---------------------------------------------------------------------------
static inline cudaError_t launch_vk_surface_to_v210(
    cudaSurfaceObject_t surf,
    uint32_t*           d_v210,
    int src_x, int src_y,
    int dst_w, int dst_h,
    int src_w, int src_h,
    bool is_16bit,
    bool use_bt2020,
    cudaStream_t stream)
{
    const int groups_per_row = (dst_w + 5) / 6;
    dim3 block(64, 1);
    dim3 grid((groups_per_row + 63) / 64, dst_h);
    k_vk_surface_to_v210<<<grid, block, 0, stream>>>(
        surf, d_v210, src_x, src_y, dst_w, dst_h, src_w, src_h, is_16bit, use_bt2020);
    return cudaGetLastError();
}

static inline cudaError_t launch_vk_surface_to_bgra8(
    cudaSurfaceObject_t surf,
    uint8_t*            d_bgra,
    int src_x, int src_y,
    int dst_w, int dst_h,
    int src_w, int src_h,
    cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((dst_w + 15) / 16, (dst_h + 15) / 16);
    k_vk_surface_to_bgra8<<<grid, block, 0, stream>>>(
        surf, d_bgra, src_x, src_y, dst_w, dst_h, src_w, src_h);
    return cudaGetLastError();
}
