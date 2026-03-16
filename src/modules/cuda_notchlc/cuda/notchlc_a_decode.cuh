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

// notchlc_a_decode.cuh
// CUDA kernel: decode NotchLC alpha plane.
//
// One thread per 16×16 alpha block.
// Reference: FFmpeg libavcodec/notchlc.c :: decode_blocks() alpha section.
// ---------------------------------------------------------------------------
#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

namespace caspar { namespace cuda_notchlc {

// ---------------------------------------------------------------------------
// k_notch_a_fill_opaque
// Called instead of k_notch_a_decode when uv_count_offset == a_control_word_offset
// (i.e. the format has no alpha channel — fill all pixels with 4095 = fully opaque).
// ---------------------------------------------------------------------------
__global__ void k_notch_a_fill_opaque(uint16_t* __restrict__ d_out_a, int n_pixels)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_pixels) d_out_a[idx] = 4095u;
}

// ---------------------------------------------------------------------------
// k_notch_a_decode
//
// d_uncompressed   — full uncompressed frame (device pointer)
// a_ctrl_ofs       — byte offset to alpha control word table
// uv_data_ofs      — byte offset to uv_data (alpha data is offset from here)
// a_data_ofs_field — the a_data_offset field value (already multiplied by 4)
// uv_count_ofs     — uv_count_offset (= y_data_offset - a_data_offset)
// width, height    — frame dimensions (multiples of 16)
// d_out_a          — output: uint16_t[height × width], 12-bit values
// ---------------------------------------------------------------------------
__global__ void k_notch_a_decode(
    const uint8_t* __restrict__ d_uncompressed,
    uint32_t a_ctrl_ofs,
    uint32_t uv_data_ofs,
    uint32_t a_data_ofs_field,
    int      width,
    int      height,
    uint16_t* __restrict__ d_out_a)
{
    const int blocks_x = (width  + 15) / 16;
    const int blocks_y = (height + 15) / 16;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= blocks_x * blocks_y) return;

    const int bx = idx % blocks_x;
    const int by = idx / blocks_x;

    // Read alpha control (8 bytes: m LE u32, offset LE u32) for this block.
    const uint8_t* ctrl_p = d_uncompressed + a_ctrl_ofs + idx * 8;
    uint32_t m = (uint32_t)ctrl_p[0] | ((uint32_t)ctrl_p[1] << 8)
               | ((uint32_t)ctrl_p[2] << 16) | ((uint32_t)ctrl_p[3] << 24);
    uint32_t raw_offset = (uint32_t)ctrl_p[4] | ((uint32_t)ctrl_p[5] << 8)
                        | ((uint32_t)ctrl_p[6] << 16) | ((uint32_t)ctrl_p[7] << 24);
    if (raw_offset >= 0x40000000u) {
        // Invalid offset — fill opaque.
        const int px_x = bx * 16, px_y = by * 16;
        for (int y = 0; y < 16 && (px_y+y) < height; y++)
            for (int x = 0; x < 16 && (px_x+x) < width; x++)
                d_out_a[(px_y+y)*width + (px_x+x)] = 4095u;
        return;
    }
    // Byte offset to the alpha data blob for this block:
    //   offset = raw_offset * 4 + uv_data_ofs + a_data_ofs_field
    uint32_t abs_off = raw_offset * 4u + uv_data_ofs + a_data_ofs_field;
    const uint8_t* dgb = d_uncompressed + abs_off;

    // 8-byte record per 16×16 block:
    //   [0]       alpha0 (uint8)
    //   [1]       alpha1 (uint8)
    //   [2..7]    48 bits of 3-bit per 4×4 sub-block indices (LSB-first)
    // There are 16 sub-blocks × 3 bits = 48 bits = 6 bytes.
    // (FFmpeg reads this as a single LE u64 and masks.)
    uint64_t ctrl64 = 0;
    for (int byte_i = 0; byte_i < 8; byte_i++)
        ctrl64 |= (uint64_t)dgb[byte_i] << (byte_i * 8);

    const unsigned alpha0 = (unsigned)(ctrl64 & 0xFFu);
    const unsigned alpha1 = (unsigned)((ctrl64 >> 8) & 0xFFu);
    // Remaining 48 bits: 3 bits per 4×4 sub-block (one index for the sub-block).
    uint64_t control = ctrl64 >> 16;

    const int px_x = bx * 16, px_y = by * 16;

    // m: 2-bit mode per 4×4 sub-block (16 sub-blocks = 32 bits total, LSB first).
    // control: 3-bit alpha index per 4×4 sub-block (all 16 pixels get the SAME value).
    for (int by4 = 0; by4 < 4; by4++) {
        for (int bx4 = 0; bx4 < 4; bx4++) {
            const uint32_t mode   = m & 3u;       m       >>= 2;
            const uint64_t cidx   = control & 7u; control >>= 3;

            // Compute the uniform alpha value for all 16 pixels in this sub-block.
            uint16_t val;
            switch (mode) {
            case 0:  val = 0u;    break;
            case 1:  val = 4095u; break;
            case 2:  {
                // alpha = (alpha0 + (alpha1 - alpha0) * cidx)
                // Result is 8-bit → shift left 4 to reach 12-bit.
                int a = (int)alpha0 + (int)((int)alpha1 - (int)alpha0) * (int)cidx;
                val = (uint16_t)min(4095, max(0, a << 4));
                break;
            }
            default: val = 4095u; break;   // invalid mode → opaque fallback
            }

            // Fill all 16 pixels in this 4×4 sub-block with the same value.
            for (int yi = 0; yi < 4; yi++) {
                for (int xi = 0; xi < 4; xi++) {
                    const int out_y = px_y + by4*4 + yi;
                    const int out_x = px_x + bx4*4 + xi;
                    if (out_y < height && out_x < width)
                        d_out_a[out_y * width + out_x] = val;
                }
            }
        }
    }
}

}} // namespace caspar::cuda_notchlc
