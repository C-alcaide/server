// cuda_prores_entropy_decode.cuh
// Inverse ProRes 422 VLC entropy decode — CUDA device functions.
//
// Exact inverse of cuda_prores_entropy.cu / cuda_prores_rice.cuh.
// Bit-stream conventions:
//   • MSB-first (same as encoder's BitPacker)
//   • Codebooks: same constants as the encoder (c_dc_codebook, c_run_to_cb, c_level_to_cb)
//   • Sign mapping: TOSIGNED(code) = (code >> 1) ^ (-(code & 1))  (zigzag)
//   • DC sign prediction: cumulative sign state per block
//   • AC traversal: scan-position-major × block-minor (interleaved)
//
// Produce output in the same format the encoder consumed:
//   d_dec_coeffs[slice][block][64] — indices are scan positions 0..63
//   d_dec_coeffs[s * (y_n+cb_n+cr_n)*64 + b*64 + scan_pos] = quantised coeff
// ---------------------------------------------------------------------------
#pragma once

#include "cuda_prores_bitreader.cuh"
#include "cuda_prores_tables.cuh"

#include <cuda_runtime.h>
#include <stdint.h>

// ─── Codebooks (redeclared as static __constant__ for the decode TU) ─────────
// These are identical to the encoder; static to avoid collisions with the
// encoder's __constant__ variables (which are in a separate .cu TU).

static __device__ __constant__ uint8_t c_dc_cb_dec[7]  =
    { 0x04, 0x28, 0x28, 0x4D, 0x4D, 0x70, 0x70 };

static __device__ __constant__ uint8_t c_run_to_cb_dec[16] =
    { 0x06, 0x06, 0x05, 0x05, 0x04, 0x29, 0x29, 0x29,
      0x29, 0x28, 0x28, 0x28, 0x28, 0x28, 0x28, 0x4C };

static __device__ __constant__ uint8_t c_level_to_cb_dec[10] =
    { 0x04, 0x0A, 0x05, 0x06, 0x04, 0x28, 0x28, 0x28, 0x28, 0x4C };

#define FIRST_DC_CB_DEC 0xB8u

// ─── TOSIGNED ────────────────────────────────────────────────────────────────
// Converts a non-negative zigzag code to signed value.
// Inverse of make_code(x) = (2*|x|) ^ (x>>31).
__device__ __forceinline__ int tosigned(unsigned code)
{
    return (int)(code >> 1) ^ -(int)(code & 1u);
}

// ─── VLC decode ──────────────────────────────────────────────────────────────
// Decodes one ProRes Rice/exp-Golomb codeword from the bit reader.
// Direct CUDA port of DECODE_CODEWORD in FFmpeg proresdec.c.
//
// codebook layout (same as encoder):
//   switch_bits = codebook & 3
//   rice_order  = codebook >> 5
//   exp_order   = (codebook >> 2) & 7
//
// Bit format — Rice (q ≤ switch_bits):
//   q zero-bits · 1-bit · rice_order-bit suffix
//
// Bit format — Exp-Golomb (q > switch_bits):
//   Reads (exp_order - switch_bits + q*2) bits total from current position.
//
__device__ __forceinline__ unsigned vlc_decode(BitReader* br, unsigned codebook)
{
    unsigned sw_bits = codebook & 3u;       // switch_bits
    unsigned ro      = codebook >> 5;        // rice_order
    unsigned eo      = (codebook >> 2) & 7u; // exp_order

    // Leading-zero count from MSB of 32-bit peek window.
    uint32_t buf = br_peek32(br);
    unsigned q   = (buf == 0u) ? 31u : (unsigned)__clz(buf);

    if (q > sw_bits) {
        // Exp-Golomb path.
        // Total bits consumed: eo - sw_bits + q*2
        // (derived from: nz = exp - eo + sw_bits + 1, total = nz + exp + 1 = 2*q + eo - sw_bits)
        unsigned total = eo - sw_bits + (q << 1u);
        // Show `total` bits starting at current position, then skip them.
        uint32_t raw = buf >> (32u - total);
        br_skip(br, total);
        // Reconstruct value: raw - (1<<eo) + (sw_bits+1)<<ro
        return raw - (1u << eo) + ((sw_bits + 1u) << ro);
    } else if (ro > 0u) {
        // Rice path with suffix.
        br_skip(br, q + 1u);   // skip q leading zeros + terminating 1
        uint32_t suffix = br_read(br, ro);
        return (q << ro) | suffix;
    } else {
        // Pure unary (rice_order = 0).
        br_skip(br, q + 1u);
        return q;
    }
}

// ─── DC plane decode ─────────────────────────────────────────────────────────
// Decodes DC coefficients for n_blocks in a single component.
// Output: blocks[b * 64 + 0] = DC (scan position 0) for block b.
// Exact inverse of encode_dc_plane().
__device__ void decode_dc_plane(BitReader* br, int16_t* blocks, int n_blocks)
{
    // First block: FIRST_DC_CB
    unsigned code   = vlc_decode(br, FIRST_DC_CB_DEC);
    int16_t  prev   = (int16_t)tosigned(code);
    blocks[0]       = prev;

    int cb_idx = 5;
    int sign   = 0;  // sign predictor (-1 or 0)

    for (int i = 1; i < n_blocks; i++) {
        unsigned dc_cb = c_dc_cb_dec[cb_idx > 6 ? 6 : cb_idx];
        code = vlc_decode(br, dc_cb);

        // Update sign predictor (matches FFmpeg decode_dc_coeffs exactly).
        if (code != 0u)
            sign ^= -(int)(code & 1u);  // flip sign on odd code
        else
            sign = 0;

        // Delta reconstruction: undo sign prediction, then apply TOSIGNED.
        // Equivalent to: delta = ((code+1)>>1) with sign applied.
        int delta = (int)(((code + 1u) >> 1) ^ (unsigned)sign) - (int)(unsigned)sign;
        prev     = (int16_t)(prev + (int16_t)delta);
        blocks[i * 64] = prev;

        // Adaptive codebook: next codebook index = min(code, 6)
        cb_idx = ((int)code < 6) ? (int)code : 6;
    }
}

// ─── AC plane decode ─────────────────────────────────────────────────────────
// Decodes AC coefficients for n_blocks in a single component.
// Output: blocks[b * 64 + scan_pos] = signed AC coeff.
// DC slot [b*64+0] is already written by decode_dc_plane; ACs go to [1..63].
// Exact inverse of encode_ac_plane().
//
// Traversal is scan-position-major × block-minor:
//   pos = i * n_blocks + b   (i=scan_pos, b=block_idx)
// Starting value: pos = n_blocks - 1  (so first pos after "run" with run=0
// is n_blocks, i.e. i=1, b=0).
__device__ void decode_ac_plane(BitReader* br, int16_t* blocks, int n_blocks,
                                 const uint8_t* scan)
{
    const int log2_n = __ffs((unsigned)n_blocks) - 1; // log2(n_blocks), n_blocks is a power of 2
    const int mask   = n_blocks - 1;
    const int max    = 64 * n_blocks;

    unsigned run   = 4u;  // initial codebook selectors
    unsigned level = 2u;

    int pos = mask;  // start at n_blocks-1

    for (;;) {
        int bits_left = br_bits_left(br);
        if (bits_left <= 0) break;

        // Stop if remaining bits are all zero (trailing padding).
        if (bits_left < 32) {
            uint32_t tail = br_peek32(br) >> (32 - bits_left);
            if (tail == 0u) break;
        }

        // Decode run (zero-coefficient count before this non-zero coeff).
        unsigned run_cb = c_run_to_cb_dec[run  < 16u ? run  : 15u];
        unsigned run_v  = vlc_decode(br, run_cb);
        pos += (int)run_v + 1;
        if (pos >= max) break;

        // Decode level (|coeff| - 1).
        unsigned lv_cb = c_level_to_cb_dec[level < 10u ? level : 9u];
        unsigned lv_v  = vlc_decode(br, lv_cb);
        unsigned abs_c = lv_v + 1u;

        // Sign bit (1 = negative).
        int sign_bit = (int)(br_read(br, 1u));

        // Write to output at scan position i, block b.
        int  i = pos >> log2_n;          // scan position  [1..63]
        int  b = pos & mask;             // block index
        int16_t coef = sign_bit ? -(int16_t)abs_c : (int16_t)abs_c;
        blocks[(ptrdiff_t)b * 64 + scan[i]] = coef;

        // Update codebook selectors.
        run   = (run_v < 15u) ? run_v : 15u;
        level = (abs_c <  9u) ? abs_c :  9u;
    }
}

// ─── ProRes 422 slice decode kernel ─────────────────────────────────────────
// One thread per slice.
//
// Input:
//   d_frame_data  : full icpf frame data (device memory)
//   d_slice_starts: byte offset from d_frame_data start to each slice's data
//   d_slice_sizes : byte size of each slice
//
// Output:
//   d_dec_coeffs  : [num_slices][(y_n + cb_n + cr_n) * 64] int16_t
//                   scan-ordered quantised coefficients (DC in [*][0])
//   d_q_scales    : [num_slices] uint8_t  q_scale read from slice header
//
// Slice header (6 bytes):
//   [0]     header_size_bits (= 48, i.e. 6*8 bytes of header)
//   [1]     q_scale
//   [2..3]  Y_size  (BE16, bytes of luma plane entropy data)
//   [4..5]  Cb_size (BE16)
//   Cr_size = slice_size - 6 - Y_size - Cb_size
//
// mbs_per_slice: maximum macroblock columns per slice (power of 2; from picture header)
// num_slices   : total number of slices in this picture
// mb_width     : frame width in macroblocks (= ceil(frame_width / 16))
// slices_per_row : number of slices per MB row (= ceil(mb_width / mbs_per_slice))
// ---------------------------------------------------------------------------  
__global__ void k_prores_entropy_decode(
    const uint8_t* __restrict__ d_frame_data,
    const uint32_t* __restrict__ d_slice_starts,   // byte offset per slice     
    const uint16_t* __restrict__ d_slice_sizes,    // byte size per slice       
    int16_t*        __restrict__ d_dec_coeffs,      // output
    uint8_t*        __restrict__ d_q_scales,        // output: q_scale per slice
    int mbs_per_slice,
    int num_slices,
    int mb_width,
    int slices_per_row,
    bool is_interlaced)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= num_slices) return;

    // Compute the actual number of MBs in this slice (last slice in each row
    // may have fewer MBs than mbs_per_slice if mb_width is not a multiple).
    const int s_col = s % slices_per_row;
    const int mbs_actual = min(mbs_per_slice, mb_width - s_col * mbs_per_slice);

    // Fixed stride uses mbs_per_slice (the maximum) so all slices have the
    // same coeff_stride in d_dec_coeffs.  Only the DECODE CALLS use mbs_actual.
    const int y_n_max  = 4 * mbs_per_slice;
    const int cb_n_max = 2 * mbs_per_slice;
    const int y_n_act  = 4 * mbs_actual;  // actual blocks to decode
    const int cb_n_act = 2 * mbs_actual;

    // Slice data pointer.
    const uint8_t* slice = d_frame_data + d_slice_starts[s];
    const int      total = (int)d_slice_sizes[s];

    if (total < 6) return;

    // ── Slice header ─────────────────────────────────────────────────────
    // byte 0: header_size_bits (typically 48 = 6*8)
    // byte 1: q_scale
    // bytes 2..3: Y_size (BE16)
    // bytes 4..5: Cb_size (BE16)
    int  hdr_bytes = slice[0] / 8;          // header size in bytes
    if (hdr_bytes < 6 || hdr_bytes > total)
        return;

    uint8_t q_scale = slice[1];
    int     y_size  = ((int)slice[2] << 8) | slice[3];
    int     cb_size = ((int)slice[4] << 8) | slice[5];
    int     cr_size = total - hdr_bytes - y_size - cb_size;

    if (y_size < 0 || cb_size < 0 || cr_size < 0)
        return;
    if (hdr_bytes + y_size + cb_size + cr_size > total)
        return;

    d_q_scales[s] = q_scale;

    // ── Coefficient buffer for this slice (fixed stride = mbs_per_slice) ──
    const ptrdiff_t stride = (ptrdiff_t)(y_n_max + cb_n_max + cb_n_max) * 64;
    int16_t* y_blks  = d_dec_coeffs + (ptrdiff_t)s * stride;
    int16_t* cb_blks = y_blks  + (ptrdiff_t)y_n_max * 64;
    int16_t* cr_blks = cb_blks + (ptrdiff_t)cb_n_max * 64;

    // Zero all coefficients (use the full fixed-stride allocation so all
    // positions, including unused slots of partial slices, are zeroed).
    for (int i = 0; i < (y_n_max + cb_n_max + cb_n_max) * 64; i++) {
        y_blks[i] = 0;
    }

    // Select scan table.
    const uint8_t* scan = is_interlaced ? c_scan_order_interlaced : c_scan_order;

    // ── Luma ─────────────────────────────────────────────────────────────
    {
        BitReader br;
        br_init(&br, slice + hdr_bytes, (unsigned)y_size);
        decode_dc_plane(&br, y_blks, y_n_act);   // decode only actual blocks
        decode_ac_plane(&br, y_blks, y_n_act, scan);
    }

    // ── Cb ───────────────────────────────────────────────────────────────
    if (cb_size > 0) {
        BitReader br;
        br_init(&br, slice + hdr_bytes + y_size, (unsigned)cb_size);
        decode_dc_plane(&br, cb_blks, cb_n_act);
        decode_ac_plane(&br, cb_blks, cb_n_act, scan);
    }

    // ── Cr ───────────────────────────────────────────────────────────────
    if (cr_size > 0) {
        BitReader br;
        br_init(&br, slice + hdr_bytes + y_size + cb_size, (unsigned)cr_size);
        decode_dc_plane(&br, cr_blks, cb_n_act);
        decode_ac_plane(&br, cr_blks, cb_n_act, scan);
    }
}

// Convenience launcher.
inline cudaError_t launch_entropy_decode(
    const uint8_t*  d_frame_data,
    const uint32_t* d_slice_starts,
    const uint16_t* d_slice_sizes,
    int16_t*        d_dec_coeffs,
    uint8_t*        d_q_scales,
    int             mbs_per_slice,
    int             num_slices,
    int             mb_width,
    int             slices_per_row,
    bool            is_interlaced,
    cudaStream_t    stream)
{
    constexpr int BLOCK = 256;
    int grid = (num_slices + BLOCK - 1) / BLOCK;
    k_prores_entropy_decode<<<grid, BLOCK, 0, stream>>>(
        d_frame_data, d_slice_starts, d_slice_sizes,
        d_dec_coeffs, d_q_scales,
        mbs_per_slice, num_slices, mb_width, slices_per_row, is_interlaced);
    return cudaGetLastError();
}
