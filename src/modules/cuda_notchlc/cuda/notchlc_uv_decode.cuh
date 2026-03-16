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
 * NotchLC is a codec specification by Derivative Inc., available under the
 * Creative Commons Attribution 4.0 International License.
 */

// notchlc_uv_decode.cuh
// CUDA kernel: decode NotchLC UV (chroma) plane.
//
// One thread per 16×16 UV block.
// Reference: FFmpeg libavcodec/notchlc.c :: decode_blocks() UV section,
// lines 349-470 — translated exactly.
// ---------------------------------------------------------------------------
#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

namespace caspar { namespace cuda_notchlc {

// Read LE uint16 from a device pointer at a given byte offset.
__device__ __forceinline__ uint16_t read_le16(const uint8_t* base, uint32_t off)
{
    return (uint16_t)base[off] | ((uint16_t)base[off+1] << 8);
}

// Read LE uint32 from a device pointer at a given byte offset.
__device__ __forceinline__ uint32_t read_le32(const uint8_t* base, uint32_t off)
{
    return (uint32_t)base[off]
         | ((uint32_t)base[off+1] << 8)
         | ((uint32_t)base[off+2] << 16)
         | ((uint32_t)base[off+3] << 24);
}

// Expand 8-bit endpoint to 12-bit: (x << 4) | (x & 0xF)  [FFmpeg notation]
__device__ __forceinline__ int expand8to12(uint8_t x)
{
    return ((int)x << 4) | ((int)x & 0xF);
}

// 3-step linear interpolation (FFmpeg: u0 + ((udif * loc_2bit + 2) / 3))
__device__ __forceinline__ int lerp3(int b, int d, int l)
{
    return b + ((d * l + 2) / 3);
}

// ---------------------------------------------------------------------------
// k_notch_uv_decode
//
// d_uncompressed  — full uncompressed frame (device pointer)
// uv_offset_ofs   — byte offset to uv_offset_data (one LE u32 per 16×16 block)
// uv_data_ofs     — byte offset to uv_data blob
// width, height   — frame dimensions (multiples of 16)
// d_out_u/v       — output: uint16_t[height × width], 12-bit values
//
// IMPLEMENTATION NOTE: u,v values are written DIRECTLY to global memory as they
// are computed. This avoids a 2×256 int local array (2 KB per thread) that the
// prior version used, which forced the CUDA compiler to spill all 512 values to
// DRAM for every one of ~295 K threads — a massive throughput bottleneck.
// ---------------------------------------------------------------------------
__global__ void k_notch_uv_decode(
    const uint8_t* __restrict__ d_uncompressed,
    uint32_t uv_offset_ofs,
    uint32_t uv_data_ofs,
    int      width,
    int      height,
    uint16_t* __restrict__ d_out_u,
    uint16_t* __restrict__ d_out_v)
{
    const int blocks_x = (width  + 15) / 16;
    const int blocks_y = (height + 15) / 16;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= blocks_x * blocks_y) return;

    const int bx = idx % blocks_x;
    const int by = idx / blocks_x;

    // Read the uv block byte offset from the offset table.
    const uint8_t* dgb = d_uncompressed + uv_data_ofs
                       + read_le32(d_uncompressed, uv_offset_ofs + idx * 4) * 4u;
    uint32_t dp = 0;  // byte offset within dgb that advances as we read

    const uint16_t is8x8  = read_le16(dgb, dp + 0);
    const uint16_t escape = read_le16(dgb, dp + 2);
    dp += 4;

    // Base pixel coordinates for this block in the output planes.
    // NotchLC requires width/height to be multiples of 16, so every block is
    // fully interior — no bounds check needed on pixel writes.
    const int px_x = bx * 16;
    const int px_y = by * 16;

    if (escape == 0 && is8x8 == 0) {
        // ── 16×16 uniform mode ─────────────────────────────────────────
        int u0 = expand8to12(dgb[dp+0]), v0 = expand8to12(dgb[dp+1]);
        int u1 = expand8to12(dgb[dp+2]), v1 = expand8to12(dgb[dp+3]);
        uint32_t loc = read_le32(dgb, dp + 4);
        const int udif = u1 - u0, vdif = v1 - v0;
        for (int i = 0; i < 16; i += 4) {
            for (int j = 0; j < 16; j += 4) {
                const int l = (int)(loc & 3u); loc >>= 2;
                const uint16_t uv = (uint16_t)lerp3(u0, udif, l);
                const uint16_t vv = (uint16_t)lerp3(v0, vdif, l);
                for (int ii = 0; ii < 4; ii++) {
                    for (int jj = 0; jj < 4; jj++) {
                        const int oi = (px_y + i + ii) * width + (px_x + j + jj);
                        d_out_u[oi] = uv;
                        d_out_v[oi] = vv;
                    }
                }
            }
        }
    } else {
        // ── Per-quadrant mode ─────────────────────────────────────────
        uint16_t is8x8_remain = is8x8;
        for (int qi = 0; qi < 2; qi++) {
            for (int qj = 0; qj < 2; qj++) {
                const int oi_base = qi * 8, oj_base = qj * 8;
                const int bit = is8x8_remain & 1;
                is8x8_remain >>= 1;

                if (bit) {
                    // ── 8×8 per-quadrant uniform mode ─────────────────
                    int u0 = expand8to12(dgb[dp+0]), v0 = expand8to12(dgb[dp+1]);
                    int u1 = expand8to12(dgb[dp+2]), v1 = expand8to12(dgb[dp+3]);
                    uint32_t loc = read_le32(dgb, dp + 4);
                    dp += 8;
                    const int udif = u1 - u0, vdif = v1 - v0;
                    for (int ii = 0; ii < 8; ii += 2) {
                        for (int jj = 0; jj < 8; jj += 2) {
                            const int l = (int)(loc & 3u); loc >>= 2;
                            const uint16_t uv = (uint16_t)lerp3(u0, udif, l);
                            const uint16_t vv = (uint16_t)lerp3(v0, vdif, l);
                            for (int iii = 0; iii < 2; iii++) {
                                for (int jjj = 0; jjj < 2; jjj++) {
                                    const int o = (px_y + oi_base + ii + iii) * width
                                                + (px_x + oj_base + jj + jjj);
                                    d_out_u[o] = uv;
                                    d_out_v[o] = vv;
                                }
                            }
                        }
                    }
                } else if (escape) {
                    // ── 4×4 per-quadrant escape mode ──────────────────
                    for (int ii = 0; ii < 8; ii += 4) {
                        for (int jj = 0; jj < 8; jj += 4) {
                            int u0 = expand8to12(dgb[dp+0]), v0 = expand8to12(dgb[dp+1]);
                            int u1 = expand8to12(dgb[dp+2]), v1 = expand8to12(dgb[dp+3]);
                            uint32_t loc = read_le32(dgb, dp + 4);
                            dp += 8;
                            const int udif = u1 - u0, vdif = v1 - v0;
                            for (int iii = 0; iii < 4; iii++) {
                                for (int jjj = 0; jjj < 4; jjj++) {
                                    const int l = (int)(loc & 3u); loc >>= 2;
                                    const int o = (px_y + oi_base + ii + iii) * width
                                                + (px_x + oj_base + jj + jjj);
                                    d_out_u[o] = (uint16_t)lerp3(u0, udif, l);
                                    d_out_v[o] = (uint16_t)lerp3(v0, vdif, l);
                                }
                            }
                        }
                    }
                } else {
                    // is8x8 bit = 0 AND escape = 0: this 8×8 quadrant is all zeros.
                    // (Original code relied on int u[16][16] = {} zero-initialisation.)
                    for (int ii = 0; ii < 8; ii++) {
                        for (int jj = 0; jj < 8; jj++) {
                            const int o = (px_y + oi_base + ii) * width
                                        + (px_x + oj_base + jj);
                            d_out_u[o] = 0;
                            d_out_v[o] = 0;
                        }
                    }
                }
            }
        }
    }
}

}} // namespace caspar::cuda_notchlc
