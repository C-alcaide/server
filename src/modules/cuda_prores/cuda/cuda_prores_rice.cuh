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
// Adaptive Rice coding
// ---------------------------------------------------------------------------

// Count bits required to Rice-encode unsigned value val with parameter k.
// Does NOT write anything — used in the counting (Pass 1) kernel.
__device__ __forceinline__ int rice_count(unsigned val, int k)
{
    // unary quotient length + terminating zero + k remainder bits
    return (int)(val >> k) + 1 + k;
}

// Write Rice(k) codeword for unsigned value val.
// Unary prefix: q one-bits then a zero; remainder: low k bits of val.
__device__ void rice_encode(BitPacker *bp, unsigned val, int k)
{
    unsigned q = val >> k;

    // Write unary part in chunks of ≤30 bits to stay within bp_put limits.
    while (q >= 30) {
        bp_put(bp, 0x3FFFFFFFu, 30);
        q -= 30;
    }
    if (q > 0)
        bp_put(bp, (1u << q) - 1u, (int)q);
    bp_put(bp, 0u, 1); // terminating zero

    // Write k-bit remainder.
    if (k > 0)
        bp_put(bp, val & ((1u << k) - 1u), k);
}

// Update Rice parameter k based on the quotient of the just-encoded value.
// ProRes adaptation: increment k when quotient is non-zero, decrement otherwise.
// k is clamped to [0, 11].
__device__ __forceinline__ void rice_adapt_k(int *k, unsigned val)
{
    unsigned q = val >> *k;
    if (q == 0) { if (*k > 0)   --(*k); }
    else         { if (*k < 11)  ++(*k); }
}
