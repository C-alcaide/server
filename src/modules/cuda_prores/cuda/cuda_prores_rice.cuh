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

// cuda_prores_rice.cuh
// ProRes 422 adaptive Rice entropy coding — CUDA device functions.
//
// Reference: Apple ProRes White Paper (2014);
//            FFmpeg libavcodec/proresenc_kostya.c (LGPL 2.1+, Kostya Shishkov)
//
// These device functions are invoked one-thread-per-slice from the two-pass
// entropy kernels in cuda_prores_entropy.cu.  They must NOT be called from
// divergent warp paths that share __shared__ state.
#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

// ---------------------------------------------------------------------------
// MSB-first bit packer  (sequential, single-thread use per slice)
// ---------------------------------------------------------------------------

struct BitPacker {
    uint8_t *buf;       // current write pointer
    uint64_t accum;     // bit accumulator (MSB-justified)
    int      accum_bits;// number of valid bits in accum
};

__device__ __forceinline__ void bp_init(BitPacker *bp, uint8_t *buf)
{
    bp->buf        = buf;
    bp->accum      = 0;
    bp->accum_bits = 0;
}

// Flush fully-accumulated bytes to output buffer.
__device__ __forceinline__ void bp_flush(BitPacker *bp)
{
    while (bp->accum_bits >= 8) {
        bp->accum_bits -= 8;
        *(bp->buf++)    = (uint8_t)(bp->accum >> bp->accum_bits);
        bp->accum      &= (1ULL << bp->accum_bits) - 1;
    }
}

// Append n bits (n <= 30) from val (LSB-justified) to the stream.
__device__ __forceinline__ void bp_put(BitPacker *bp, uint32_t val, int n)
{
    bp->accum       = (bp->accum << n) | (uint64_t)(val & ((1u << n) - 1));
    bp->accum_bits += n;
    bp_flush(bp);
}

// Zero-pad to the next full byte boundary.
__device__ __forceinline__ void bp_pad_byte(BitPacker *bp)
{
    if (bp->accum_bits > 0)
        bp_put(bp, 0u, 8 - bp->accum_bits);
}

// Bytes written (including partial byte) since bp_init.
__device__ __forceinline__ int bp_bytes(const BitPacker *bp, const uint8_t *buf_start)
{
    return (int)(bp->buf - buf_start) + (bp->accum_bits > 0 ? 1 : 0);
}

// ---------------------------------------------------------------------------
// ProRes hybrid Rice / exp-Golomb VLC coding
//
// Reference: FFmpeg libavcodec/proresenc_kostya.c encode_vlc_codeword()
//            Apple ProRes White Paper §"Entropy Coding"
//
// Each codebook byte encodes three parameters:
//   switch_bits = (codebook & 3) + 1   encoder-side (decoder uses cb & 3)
//   rice_order  =  codebook >> 5
//   exp_order   = (codebook >> 2) & 7
//
// Bitstream format — Rice path (val < switch_val):
//   q = val >> rice_order   (quotient)
//   prefix : q zero-bits then one 1-bit
//   suffix : low rice_order bits of val (LSB → MSB in stream order)
//
// Bitstream format — exp-Golomb path (val >= switch_val):
//   leading_zeros = exponent - exp_order + switch_bits
//   then (exponent+1)-bit representation of adjusted value
// ---------------------------------------------------------------------------

// Count bits needed for VLC codeword — no side effects.
__device__ __forceinline__ unsigned vlc_count(unsigned codebook, unsigned val)
{
    unsigned sw = (codebook & 3u) + 1u;
    unsigned ro = codebook >> 5;
    unsigned eo = (codebook >> 2) & 7u;
    unsigned sv = sw << ro;

    if (val >= sv) {
        val -= sv - (1u << eo);
        unsigned exp = 31u - __clz(val);        // floor(log2(val))
        return exp * 2u - eo + sw + 1u;
    } else {
        return (val >> ro) + ro + 1u;
    }
}

// Write ProRes VLC codeword to BitPacker.
__device__ void vlc_encode(BitPacker *bp, unsigned codebook, unsigned val)
{
    unsigned sw = (codebook & 3u) + 1u;
    unsigned ro = codebook >> 5;
    unsigned eo = (codebook >> 2) & 7u;
    unsigned sv = sw << ro;

    if (val >= sv) {
        // Exp-Golomb path
        val -= sv - (1u << eo);
        unsigned exp = 31u - __clz(val);
        // Write leading zeros
        unsigned nz = exp - eo + sw;
        for (unsigned n = nz; n > 0u; ) {
            int chunk = (n >= 30u) ? 30 : (int)n;
            bp_put(bp, 0u, chunk);
            n -= (unsigned)chunk;
        }
        // Write (exp+1)-bit value MSB-first
        unsigned nb = exp + 1u;
        if (nb > 30u) {
            bp_put(bp, (val >> (nb - 30u)) & 0x3FFFFFFFu, 30);
            nb -= 30u;
        }
        bp_put(bp, val & ((1u << nb) - 1u), (int)nb);
    } else {
        // Rice path: write q zero-bits, one 1-bit, then rice_order remainder
        unsigned q = val >> ro;
        for (unsigned n = q; n > 0u; ) {
            int chunk = (n >= 30u) ? 30 : (int)n;
            bp_put(bp, 0u, chunk);
            n -= (unsigned)chunk;
        }
        bp_put(bp, 1u, 1);
        if (ro)
            bp_put(bp, val & ((1u << ro) - 1u), (int)ro);
    }
}
