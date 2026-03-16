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

// notchlc_y_decode.cuh
// CUDA kernel: decode NotchLC Y (luma) plane.
//
// One thread per 4×4 Y block.
// Reference: FFmpeg libavcodec/notchlc.c :: decode_blocks() Y section.
//
// ---------------------------------------------------------------------------
#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

namespace caspar { namespace cuda_notchlc {

// ---------------------------------------------------------------------------
// Device-side LSB-first bit reader
// Reads up to 16 bits at a time from an arbitrary byte+bit offset.
// Each thread maintains its own state — no shared memory.
// ---------------------------------------------------------------------------
struct LsbBitReader {
    const uint8_t* __restrict__ data;
    int byte_off;
    int bit_off;   // bits already consumed within data[byte_off]

    __device__ LsbBitReader(const uint8_t* base, int byte_offset)
        : data(base), byte_off(byte_offset), bit_off(0) {}

    // Read `n` bits (1..16) LSB-first.
    __device__ uint32_t get_bits(int n)
    {
        // Load a 64-bit window and shift out already-consumed bits.
        // Accessing up to byte_off+7 is safe given the uncompressed buffer
        // is over-allocated with padding.
        uint64_t w = 0;
        // Load 8 bytes little-endian from current position
        const uint8_t* p = data + byte_off;
        w  = (uint64_t)p[0];
        w |= (uint64_t)p[1] << 8;
        w |= (uint64_t)p[2] << 16;
        w |= (uint64_t)p[3] << 24;
        w |= (uint64_t)p[4] << 32;
        w |= (uint64_t)p[5] << 40;
        w |= (uint64_t)p[6] << 48;
        w |= (uint64_t)p[7] << 56;

        w >>= bit_off;
        uint32_t val = (uint32_t)(w & ((1u << n) - 1u));

        // Advance position
        bit_off += n;
        byte_off += bit_off >> 3;
        bit_off  &= 7;
        return val;
    }
};

// ---------------------------------------------------------------------------
// k_y_compute_widths
//
// Pass 1 of the two-pass Y decode.  One thread per 4×4 Y block.
// Reads each block's 4 control values and writes the total bit-count consumed
// by that block (sum over 4 rows of 4 pixels × (ctrl+1) bits) into d_bit_widths.
// A subsequent CUB segmented exclusive scan converts these widths into per-block
// bit start offsets, eliminating the O(blocks_x) serial scan from each decode thread.
// ---------------------------------------------------------------------------
__global__ void k_y_compute_widths(
    const uint8_t* __restrict__ d_uncompressed,
    uint32_t y_control_ofs,
    int      total_blocks,
    uint32_t* __restrict__ d_bit_widths)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_blocks) return;

    const uint8_t* p = d_uncompressed + y_control_ofs + idx * 4;
    uint32_t item = (uint32_t)p[0]
                  | ((uint32_t)p[1] <<  8)
                  | ((uint32_t)p[2] << 16)
                  | ((uint32_t)p[3] << 24);

    uint32_t total = 0;
    for (int r = 0; r < 4; r++)
        total += 4u * (((item >> (24 + r * 2)) & 3u) + 1u);
    d_bit_widths[idx] = total;
}

// ---------------------------------------------------------------------------
// k_y_prefix_rows
//
// Pass 2 of the two-pass Y decode.  One thread per row-group (by).
// Each thread walks sequentially over all blocks in its row, accumulating a
// running bit offset and writing the exclusive prefix sum into d_bit_offsets.
// Runs in O(blocks_x) serial work per thread, but entirely parallel across the
// blocks_y row-groups — total work O(blocks_x * blocks_y) vs the original
// O(blocks_x^2 * blocks_y / 2).  No library dependencies required.
// ---------------------------------------------------------------------------
__global__ void k_y_prefix_rows(
    const uint32_t* __restrict__ d_bit_widths,
    uint32_t*       __restrict__ d_bit_offsets,
    int blocks_x,
    int blocks_y)
{
    const int by = blockIdx.x * blockDim.x + threadIdx.x;
    if (by >= blocks_y) return;

    const uint32_t* row_w = d_bit_widths  + by * blocks_x;
    uint32_t*       row_o = d_bit_offsets + by * blocks_x;

    uint32_t acc = 0;
    for (int bx = 0; bx < blocks_x; bx++) {
        row_o[bx] = acc;
        acc += row_w[bx];
    }
}

// ---------------------------------------------------------------------------
// k_notch_y_decode
//
// Pass 2 of the two-pass Y decode.  One thread per 4×4 Y block.
// d_uncompressed  — full uncompressed NotchLC frame (device pointer)
// y_control_ofs   — byte offset within d_uncompressed to start of y control table
// y_row_ofs_ofs   — byte offset to the y_data_row_offsets array (one LE u32 per group-of-4 rows)
// y_data_ofs      — byte offset to the luma bitstream data
// width, height   — frame dimensions (must be multiples of 4)
// d_bit_offsets   — exclusive prefix sum produced by k_y_compute_widths + CUB scan;
//                   d_bit_offsets[by * blocks_x + bx] = bit start offset within the
//                   row-group bitstream for block (bx, by)
// d_out_y         — output: uint16_t[height × width], 12-bit values in low bits
// ---------------------------------------------------------------------------
__global__ void k_notch_y_decode(
    const uint8_t* __restrict__ d_uncompressed,
    uint32_t y_control_ofs,
    uint32_t y_row_ofs_ofs,
    uint32_t y_data_ofs,
    int      width,
    int      height,
    const uint32_t* __restrict__ d_bit_offsets,
    uint16_t* __restrict__ d_out_y)
{
    const int blocks_x = width  / 4;
    const int blocks_y = height / 4;
    const int total_blocks = blocks_x * blocks_y;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_blocks) return;

    const int bx = idx % blocks_x;
    const int by = idx / blocks_x;

    // ── Read y_min, y_max, control[0..3] from control word (LE u32) ──────
    const uint8_t* ctrl_base = d_uncompressed + y_control_ofs;
    uint32_t item;
    {
        const uint8_t* p = ctrl_base + (by * blocks_x + bx) * 4;
        item = (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
    }
    const uint32_t y_min  = item & 0xFFFu;
    const uint32_t y_max  = (item >> 12) & 0xFFFu;
    const uint32_t y_diff = y_max - y_min;

    // control[i] is 2 bits at bits [25:24], [27:26], [29:28], [31:30]
    uint32_t control[4];
    control[0] = (item >> 24) & 3u;
    control[1] = (item >> 26) & 3u;
    control[2] = (item >> 28) & 3u;
    control[3] = (item >> 30) & 3u;

    // ── Read row bitstream offset for this 4-row group ────────────────────
    // y_data_row_offsets: one LE u32 per group of 4 rows (i.e. one per by)
    const uint8_t* row_ofs_base = d_uncompressed + y_row_ofs_ofs;
    uint32_t row_off;
    {
        const uint8_t* p = row_ofs_base + by * 4;
        row_off = (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
    }

    // The bitstream for all blocks in this row-group is sequential LSB-first.
    // Each block reads from a different bit offset within the row's bitstream.
    // The starting bit offset is precomputed by k_y_compute_widths + CUB scan,
    // eliminating the O(blocks_x) serial scan that was the main perf bottleneck.
    uint32_t bit_offset = d_bit_offsets[by * blocks_x + bx];

    // ── Decode 4 rows of 4 pixels each ───────────────────────────────────
    uint16_t* out_row = d_out_y + by * 4 * width + bx * 4;

    for (int r = 0; r < 4; r++) {
        const int nb_bits = (int)(control[r] + 1);
        const uint32_t div = (1u << nb_bits) - 1u;
        const uint32_t add = div - 1u;

        LsbBitReader br(d_uncompressed + y_data_ofs + row_off, (int)(bit_offset >> 3));
        br.bit_off = (int)(bit_offset & 7u);

        uint16_t* dst = out_row + r * width;
        for (int px = 0; px < 4; px++) {
            uint32_t bits = br.get_bits(nb_bits);
            uint32_t val  = y_min + ((y_diff * bits + add) / div);
            if (val > 4095u) val = 4095u;
            dst[px] = (uint16_t)val;
        }

        bit_offset += 4u * (uint32_t)nb_bits;
    }
}

}} // namespace caspar::cuda_notchlc
