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

// cuda_prores_bitreader.cuh
// MSB-first sequential bit reader for ProRes entropy decode on the GPU.
//
// ProRes uses MSB-first bit packing (the encoder's BitPacker writes bits from
// the most-significant position downward).  This reader is the exact dual:
// bits are consumed from the MSB of each byte toward the LSB.
//
// Designed for single-thread sequential use (one thread per slice).
// All functions are __device__ __forceinline__.
#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

// ---------------------------------------------------------------------------
// BitReader: state for an MSB-first bit stream reader.
//
// bit_pos   current read position (0 = first bit of buf[0])
// buf_bits  total available bits  (= byte_count * 8)
// ---------------------------------------------------------------------------
struct BitReader {
    const uint8_t* buf;
    unsigned       bit_pos;
    unsigned       buf_bits;
};

__device__ __forceinline__ void br_init(BitReader* br,
                                        const uint8_t* buf,
                                        unsigned byte_count)
{
    br->buf      = buf;
    br->bit_pos  = 0;
    br->buf_bits = byte_count * 8u;
}

// Number of bits remaining in the stream.
__device__ __forceinline__ int br_bits_left(const BitReader* br)
{
    return (int)br->buf_bits - (int)br->bit_pos;
}

// Peek the next 32 bits (MSB-first) without advancing the position.
// Bits beyond the end of the buffer are treated as 0.
__device__ __forceinline__ uint32_t br_peek32(const BitReader* br)
{
    const unsigned byte_pos = br->bit_pos >> 3;
    const unsigned bit_off  = br->bit_pos & 7u;

    const unsigned buf_bytes = br->buf_bits >> 3;

    // Load 5 bytes to cover any bit_off [0,7] over a 32-bit window.
    uint32_t b0 = (byte_pos     < buf_bytes) ? (uint32_t)br->buf[byte_pos]     : 0u;
    uint32_t b1 = (byte_pos + 1 < buf_bytes) ? (uint32_t)br->buf[byte_pos + 1] : 0u;
    uint32_t b2 = (byte_pos + 2 < buf_bytes) ? (uint32_t)br->buf[byte_pos + 2] : 0u;
    uint32_t b3 = (byte_pos + 3 < buf_bytes) ? (uint32_t)br->buf[byte_pos + 3] : 0u;
    uint32_t b4 = (byte_pos + 4 < buf_bytes) ? (uint32_t)br->buf[byte_pos + 4] : 0u;

    // Pack into 64-bit value, then shift to align bit_pos to the MSB.
    uint64_t acc = ((uint64_t)b0 << 32) | ((uint64_t)b1 << 24)
                 | ((uint64_t)b2 << 16) | ((uint64_t)b3 <<  8)
                 |  (uint64_t)b4;

    // Shift: we want bits [bit_pos .. bit_pos+31] at positions [31..0]
    // acc has bit_pos-th bit at position (40 - bit_off - 1) from bit 39.
    // After the 5-byte load into the 40-bit range [39..0]:
    //   bit 39 = buf[byte_pos] bit 7
    //   bit 38 = buf[byte_pos] bit 6
    //   ...
    //   bit (39 - bit_off) = first bit we want in position 31
    // So we shift right by (40 - 32 - bit_off) = (8 - bit_off).
    return (uint32_t)(acc >> (8u - bit_off));
}

// Advance the read position by n bits (1 ≤ n ≤ 32).
__device__ __forceinline__ void br_skip(BitReader* br, unsigned n)
{
    br->bit_pos += n;
}

// Read n bits and advance.
__device__ __forceinline__ uint32_t br_read(BitReader* br, unsigned n)
{
    uint32_t bits = br_peek32(br) >> (32u - n);
    br_skip(br, n);
    return bits;
}

// Count leading zeros in the MSB-first stream (= av_log2 complement).
// Returns the number of leading 0 bits before the first 1 bit.
// Uses __clz on the 32-bit peek (bit 31 of peek32 is the next bit).
__device__ __forceinline__ unsigned br_count_leading_zeros(const BitReader* br)
{
    uint32_t v = br_peek32(br);
    // If all zeros (stream exhausted or truly all-zero), return 31.
    return (v == 0u) ? 31u : (unsigned)__clz(v);
}
