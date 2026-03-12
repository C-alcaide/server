// cuda_prores_entropy.cu
// ProRes 422 two-pass slice entropy encoding on the GPU.
//
// Architecture
// ─────────────────────────────────────────────────────────────────────────────
//  Pass 1  k_count_bits     1 thread/slice  →  d_bit_counts[s*3+0/1/2] = Y/Cb/Cr bits
//  Proc    k_bits_to_bytes  convert bits → bytes (ceil division)
//  Proc    k_compute_sizes  d_sizes[s] = 6 + Y_bytes + Cb_bytes + Cr_bytes
//  Scan    CUB ExclusiveSum d_sizes → d_offsets[0..num_slices-1]
//  Proc    k_set_total      d_offsets[num_slices] = total bytes
//  Pass 2  k_encode_slices  1 thread/slice  →  write slice header + Y + Cb + Cr
//
// Slice format (ProRes 422 progressive, 3 planes):
//   [0x30]           1 byte  slice header size in bits = 6*8 = 48
//   [q_scale]        1 byte  quantiser used for this slice
//   [Y_size  BE16]   2 bytes luma plane byte count
//   [Cb_size BE16]   2 bytes Cb plane byte count
//   [Y  data]        Y_size  bytes, byte-aligned
//   [Cb data]        Cb_size bytes, byte-aligned
//   [Cr data]        total - 6 - Y_size - Cb_size bytes, byte-aligned
//
// Reference: Apple ProRes White Paper §"Entropy Coding";
//            FFmpeg libavcodec/proresenc_kostya.c (LGPL 2.1+, Kostya Shishkov)
//            FFmpeg libavcodec/proresdec.c (§decode_slice_thread)
// ---------------------------------------------------------------------------

#include "cuda_prores_rice.cuh"     // BitPacker, vlc_count, vlc_encode
#include "cuda_prores_tables.cuh"   // c_scan_order (not needed here; blocks already reordered)

#include <cub/device/device_scan.cuh>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstdint>

// ─── ProRes VLC codebooks (from FFmpeg proresdata.c) ─────────────────────────

static __device__ __constant__ uint8_t c_dc_codebook[7]  =
    { 0x04, 0x28, 0x28, 0x4D, 0x4D, 0x70, 0x70 };

static __device__ __constant__ uint8_t c_run_to_cb[16]   =
    { 0x06, 0x06, 0x05, 0x05, 0x04, 0x29, 0x29, 0x29,
      0x29, 0x28, 0x28, 0x28, 0x28, 0x28, 0x28, 0x4C };

static __device__ __constant__ uint8_t c_level_to_cb[10] =
    { 0x04, 0x0A, 0x05, 0x06, 0x04, 0x28, 0x28, 0x28, 0x28, 0x4C };

#define FIRST_DC_CB 0xB8u   // rice_order=5, exp_order=6, switch_bits=0

// ─── Sign mapping: MAKE_CODE / TOSIGNED ──────────────────────────────────────
// ProRes uses sign-magnitude with a "zigzag" sign embedding identical to FFmpeg:
//  MAKE_CODE(x)  = (x*2) ^ GET_SIGN(x),  GET_SIGN(x)=x>>31
//  TOSIGNED(c)   = (c>>1) ^ (-(c&1))

__device__ __forceinline__ unsigned make_code(int val)
{
    int sign = val >> 31;   // -1 or 0
    return (unsigned)(val * 2) ^ (unsigned)sign;
}

// ─── DC plane encoder / counter ──────────────────────────────────────────────
//
// blocks:   quantised, scan-reordered coefficients for this component
//           layout: [n_blocks][64]  —  blk[i*64 + 0] is the bias-corrected DC
// n_blocks: number of 8×8 blocks in this component for the slice
// bp:       nullptr → count mode (returns bit sum, no writes)
//           non-null → write mode (encodes and returns bit sum)
//
// Encoding matches FFmpeg encode_dcs() exactly:
//   First block: FIRST_DC_CB codebook
//   Subsequent: adaptive delta + sign-prediction, codebook = ff_prores_dc_codebook
// ─────────────────────────────────────────────────────────────────────────────
__device__ int encode_dc_plane(const int16_t *blocks, int n_blocks, BitPacker *bp)
{
    int bits = 0;

    // First block DC
    int prev_dc = (int)blocks[0];
    unsigned code = make_code(prev_dc);
    if (bp) vlc_encode(bp, FIRST_DC_CB, code);
    bits += (int)vlc_count(FIRST_DC_CB, code);

    int sign     = 0;     // sign of previous delta (0 or -1)
    int codebook = 5;     // index into c_dc_codebook[] for next block

    for (int i = 1; i < n_blocks; i++) {
        int dc       = (int)blocks[i * 64];
        int delta    = dc - prev_dc;
        int new_sign = delta >> 31;           // -1 if negative, 0 if non-negative
        delta        = (delta ^ sign) - sign; // apply sign prediction
        code         = make_code(delta);

        unsigned cb = c_dc_codebook[codebook];
        if (bp) vlc_encode(bp, cb, code);
        bits += (int)vlc_count(cb, code);

        // Adaptive codebook selection: next codebook tracks magnitude of current code
        codebook = (int)code < 6 ? (int)code : 6;
        sign     = new_sign;
        prev_dc  = dc;
    }

    return bits;
}

// ─── AC plane encoder / counter ──────────────────────────────────────────────
//
// The traversal is scan-position-major × block-minor (matching FFmpeg encode_acs):
//   outer: i = 1..63    scan positions  (blocks[][i] already contains scan-reordered coeff)
//   inner: b = 0..n-1   blocks in slice
//   When level != 0: emit (run, abs_level-1, sign), then reset run counter.
//   Trailing zeros are NOT coded; the decoder stops when it runs out of bits.
// ─────────────────────────────────────────────────────────────────────────────
__device__ int encode_ac_plane(const int16_t *blocks, int n_blocks, BitPacker *bp)
{
    int bits     = 0;
    int prev_run = 4;   // initial run codebook selector
    int prev_lv  = 2;   // initial level codebook selector
    int run      = 0;

    for (int i = 1; i < 64; i++) {           // scan position 1..63
        for (int b = 0; b < n_blocks; b++) { // block in slice
            int c = (int)blocks[b * 64 + i]; // quantised coeff at scan pos i, block b

            if (c == 0) {
                run++;
            } else {
                int abs_c   = c < 0 ? -c : c;
                unsigned ru = (unsigned)run;
                unsigned lv = (unsigned)(abs_c - 1);

                unsigned run_cb = c_run_to_cb[prev_run < 16 ? prev_run : 15];
                unsigned lv_cb  = c_level_to_cb[prev_lv < 10 ? prev_lv : 9];

                if (bp) {
                    vlc_encode(bp, run_cb, ru);
                    vlc_encode(bp, lv_cb,  lv);
                    bp_put(bp, c < 0 ? 1u : 0u, 1); // sign bit
                }
                bits += (int)vlc_count(run_cb, ru)
                      + (int)vlc_count(lv_cb,  lv)
                      + 1;

                prev_run = (int)ru < 15 ? (int)ru : 15;
                prev_lv  = abs_c     <  9 ? abs_c  :  9;
                run      = 0;
            }
        }
    }
    // Trailing zeros: not coded — decoder stops when bitstream is exhausted.
    return bits;
}

// ─── Pass 1: count bits per component per slice ───────────────────────────────
//
// d_coeffs_slice layout per slice s:
//   [4*mbs * 64] int16  →  Y blocks
//   [2*mbs * 64] int16  →  Cb blocks
//   [2*mbs * 64] int16  →  Cr blocks
//
// d_bit_counts output: [num_slices * 3]
//   index s*3+0 = Y bits,  s*3+1 = Cb bits,  s*3+2 = Cr bits
// ─────────────────────────────────────────────────────────────────────────────
__global__ void k_count_bits(
    const int16_t *d_coeffs_slice,
    uint32_t      *d_bit_counts,
    int            mbs_per_slice,
    int            num_slices)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= num_slices) return;

    const int y_n  = 4 * mbs_per_slice;
    const int c_n  = 2 * mbs_per_slice;
    const ptrdiff_t stride = (ptrdiff_t)(y_n + c_n + c_n) * 64;

    const int16_t *y_blks  = d_coeffs_slice + (ptrdiff_t)s * stride;
    const int16_t *cb_blks = y_blks  + (ptrdiff_t)y_n * 64;
    const int16_t *cr_blks = cb_blks + (ptrdiff_t)c_n * 64;

    d_bit_counts[s * 3 + 0] = (uint32_t)(encode_dc_plane(y_blks,  y_n, nullptr)
                                        + encode_ac_plane(y_blks,  y_n, nullptr));
    d_bit_counts[s * 3 + 1] = (uint32_t)(encode_dc_plane(cb_blks, c_n, nullptr)
                                        + encode_ac_plane(cb_blks, c_n, nullptr));
    d_bit_counts[s * 3 + 2] = (uint32_t)(encode_dc_plane(cr_blks, c_n, nullptr)
                                        + encode_ac_plane(cr_blks, c_n, nullptr));
}

// ─── Convert bit counts → byte counts (ceil) ─────────────────────────────────

__global__ void k_bits_to_bytes(uint32_t *v, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // Round bits up to the nearest EVEN number of bytes (2-byte / 16-bit alignment).
    // ProRes requires each colour component to be 2-byte aligned so that
    // the seek table can store size/2 as a uint16 (decoder reads entry << 1).
    if (i < n) v[i] = ((v[i] + 15u) >> 4) << 1;
}

// ─── Compute per-slice total byte sizes ───────────────────────────────────────
// d_sizes[s] = 6-byte slice header + Y bytes + Cb bytes + Cr bytes

__global__ void k_compute_slice_sizes(
    const uint32_t *d_byte_counts,  // [num_slices * 3], already byte-rounded
    uint32_t       *d_sizes,        // [num_slices]
    int             num_slices)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= num_slices) return;
    d_sizes[s] = 6u + d_byte_counts[s * 3 + 0]
                    + d_byte_counts[s * 3 + 1]
                    + d_byte_counts[s * 3 + 2];
}

// ─── Fill total bytes at d_offsets[num_slices] ────────────────────────────────
// CUB ExclusiveSum writes n elements; the total is at offsets[n-1] + sizes[n-1].

__global__ void k_set_total(uint32_t *offsets, const uint32_t *sizes, int n)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
        offsets[n] = offsets[n - 1] + sizes[n - 1];
}

// ─── Pass 2: write slices ─────────────────────────────────────────────────────

__global__ void k_encode_slices(
    const int16_t  *d_coeffs_slice,
    const uint32_t *d_slice_offsets,  // [num_slices + 1]
    const uint32_t *d_byte_counts,    // [num_slices * 3] — bytes after ceil div
    uint8_t        *d_output,
    int             q_scale,
    int             mbs_per_slice,
    int             num_slices)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= num_slices) return;

    const int y_n  = 4 * mbs_per_slice;
    const int c_n  = 2 * mbs_per_slice;
    const ptrdiff_t stride = (ptrdiff_t)(y_n + c_n + c_n) * 64;

    const int16_t *y_blks  = d_coeffs_slice + (ptrdiff_t)s * stride;
    const int16_t *cb_blks = y_blks  + (ptrdiff_t)y_n * 64;
    const int16_t *cr_blks = cb_blks + (ptrdiff_t)c_n * 64;

    uint32_t y_bytes  = d_byte_counts[s * 3 + 0];
    uint32_t cb_bytes = d_byte_counts[s * 3 + 1];

    uint8_t *out = d_output + d_slice_offsets[s];

    // 6-byte slice header (hdr_size 6 bytes → 48 bits → 0x30)
    out[0] = 0x30;
    out[1] = (uint8_t)q_scale;
    out[2] = (uint8_t)(y_bytes  >> 8);
    out[3] = (uint8_t)(y_bytes  & 0xFF);
    out[4] = (uint8_t)(cb_bytes >> 8);
    out[5] = (uint8_t)(cb_bytes & 0xFF);

    BitPacker bp;

    // Y component
    bp_init(&bp, out + 6);
    encode_dc_plane(y_blks,  y_n, &bp);
    encode_ac_plane(y_blks,  y_n, &bp);
    bp_pad_byte(&bp);

    // Cb component
    bp_init(&bp, out + 6 + y_bytes);
    encode_dc_plane(cb_blks, c_n, &bp);
    encode_ac_plane(cb_blks, c_n, &bp);
    bp_pad_byte(&bp);

    // Cr component
    bp_init(&bp, out + 6 + y_bytes + cb_bytes);
    encode_dc_plane(cr_blks, c_n, &bp);
    encode_ac_plane(cr_blks, c_n, &bp);
    bp_pad_byte(&bp);
}

// ─── Host: run two-pass entropy coding ────────────────────────────────────────
//
// d_coeffs_slice   device: [num_slices][blocks_per_slice][64] int16  (interleaved layout)
// num_slices        total slices in the frame
// mbs_per_slice     macroblock columns per slice (power of 2)
// q_scale           quantiser for slice header byte
// d_output          device: output bitstream (must be large enough)
// d_slice_offsets   device: [num_slices+1], filled by this call
// d_bit_counts      device: [num_slices * 3] temp (Y/Cb/Cr bits→bytes per slice)
// d_slice_sizes     device: [num_slices] temp (per-slice total bytes)
// d_cub_temp        device: CUB temp buffer
// cub_temp_bytes    size of CUB temp buffer

void cuda_prores_enc_frame_raw(
    const int16_t *d_coeffs_slice,
    int            num_slices,
    int            mbs_per_slice,
    int            q_scale,
    uint8_t       *d_output,
    uint32_t      *d_slice_offsets,
    uint32_t      *d_bit_counts,
    uint32_t      *d_slice_sizes,
    void          *d_cub_temp,
    size_t         cub_temp_bytes,
    cudaStream_t   stream)
{
    const int T = 128;
    const int G = (num_slices + T - 1) / T;

    // Pass 1: bit counts per component (3 per slice)
    k_count_bits<<<G, T, 0, stream>>>(
        d_coeffs_slice, d_bit_counts, mbs_per_slice, num_slices);

    // Bit → byte (ceil) for all 3*num_slices entries
    const int G3 = (num_slices * 3 + T - 1) / T;
    k_bits_to_bytes<<<G3, T, 0, stream>>>(d_bit_counts, num_slices * 3);

    // Per-slice total byte count = 6 + Y + Cb + Cr
    k_compute_slice_sizes<<<G, T, 0, stream>>>(d_bit_counts, d_slice_sizes, num_slices);

    // Exclusive prefix sum → byte offset per slice
    cub::DeviceScan::ExclusiveSum(
        d_cub_temp, cub_temp_bytes,
        d_slice_sizes, d_slice_offsets,
        num_slices, stream);

    // Set total bytes at offsets[num_slices]
    k_set_total<<<1, 1, 0, stream>>>(d_slice_offsets, d_slice_sizes, num_slices);

    // Pass 2: write slices
    k_encode_slices<<<G, T, 0, stream>>>(
        d_coeffs_slice, d_slice_offsets, d_bit_counts, d_output,
        q_scale, mbs_per_slice, num_slices);
}

// ═══════════════════════════════════════════════════════════════════════════
// ProRes 4444 / 4444 XQ entropy encoder
//
// Layout differences vs. 422:
//   c_n = 4 * mbs_per_slice  (full-res chroma, not 2*mbs)
//   a_n = 4 * mbs_per_slice  (alpha plane; zero if !has_alpha)
//   stride = (4 + 4 + 4 + a_n/4) * mbs * 64
//   d_bit_counts: 4 entries per slice  [Y, Cb, Cr, A]
//
// Slice header (8 bytes = 0x40 bits):
//   [0x40]         header size in bits
//   [q_scale]      quantiser
//   [Y_size  BE16] luma byte count
//   [Cb_size BE16] Cb byte count
//   [A_size  BE16] alpha byte count (0 when !has_alpha)
//   Data: Y bytes, Cb bytes, Cr bytes (implicit), A bytes
// ═══════════════════════════════════════════════════════════════════════════

__global__ void k_count_bits_444(
    const int16_t *d_coeffs_slice,
    uint32_t      *d_bit_counts,    // [num_slices * 4]
    int            mbs_per_slice,
    int            num_slices,
    int            has_alpha)       // 0 or 1
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= num_slices) return;

    const int y_n = 4 * mbs_per_slice;
    const int c_n = 4 * mbs_per_slice;  // full-res chroma
    const int a_n = has_alpha ? 4 * mbs_per_slice : 0;
    const ptrdiff_t stride = (ptrdiff_t)(y_n + c_n + c_n + a_n) * 64;

    const int16_t *y_blks     = d_coeffs_slice + (ptrdiff_t)s * stride;
    const int16_t *cb_blks    = y_blks  + (ptrdiff_t)y_n * 64;
    const int16_t *cr_blks    = cb_blks + (ptrdiff_t)c_n * 64;
    const int16_t *alpha_blks = cr_blks + (ptrdiff_t)c_n * 64;

    d_bit_counts[s * 4 + 0] = (uint32_t)(encode_dc_plane(y_blks,  y_n, nullptr)
                                        + encode_ac_plane(y_blks,  y_n, nullptr));
    d_bit_counts[s * 4 + 1] = (uint32_t)(encode_dc_plane(cb_blks, c_n, nullptr)
                                        + encode_ac_plane(cb_blks, c_n, nullptr));
    d_bit_counts[s * 4 + 2] = (uint32_t)(encode_dc_plane(cr_blks, c_n, nullptr)
                                        + encode_ac_plane(cr_blks, c_n, nullptr));
    d_bit_counts[s * 4 + 3] = has_alpha
        ? (uint32_t)(encode_dc_plane(alpha_blks, a_n, nullptr)
                   + encode_ac_plane(alpha_blks, a_n, nullptr))
        : 0u;
}

// Per-slice total bytes for 4444: 8-byte header + four planes.
__global__ void k_compute_slice_sizes_444(
    const uint32_t *d_byte_counts,  // [num_slices * 4], already byte-rounded
    uint32_t       *d_sizes,        // [num_slices]
    int             num_slices)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= num_slices) return;
    d_sizes[s] = 8u + d_byte_counts[s * 4 + 0]   // 8-byte header (not 6)
                    + d_byte_counts[s * 4 + 1]
                    + d_byte_counts[s * 4 + 2]
                    + d_byte_counts[s * 4 + 3];
}

__global__ void k_encode_slices_444(
    const int16_t  *d_coeffs_slice,
    const uint32_t *d_slice_offsets,
    const uint32_t *d_byte_counts,    // [num_slices * 4]
    uint8_t        *d_output,
    int             q_scale,
    int             mbs_per_slice,
    int             num_slices,
    int             has_alpha)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= num_slices) return;

    const int y_n = 4 * mbs_per_slice;
    const int c_n = 4 * mbs_per_slice;
    const int a_n = has_alpha ? 4 * mbs_per_slice : 0;
    const ptrdiff_t stride = (ptrdiff_t)(y_n + c_n + c_n + a_n) * 64;

    const int16_t *y_blks     = d_coeffs_slice + (ptrdiff_t)s * stride;
    const int16_t *cb_blks    = y_blks  + (ptrdiff_t)y_n * 64;
    const int16_t *cr_blks    = cb_blks + (ptrdiff_t)c_n * 64;
    const int16_t *alpha_blks = cr_blks + (ptrdiff_t)c_n * 64;

    const uint32_t y_bytes  = d_byte_counts[s * 4 + 0];
    const uint32_t cb_bytes = d_byte_counts[s * 4 + 1];
    const uint32_t cr_bytes = d_byte_counts[s * 4 + 2];
    const uint32_t a_bytes  = d_byte_counts[s * 4 + 3];

    uint8_t *out = d_output + d_slice_offsets[s];

    // 8-byte ProRes 4444 slice header
    out[0] = 0x40;   // header_bits = 64 = 8 bytes
    out[1] = (uint8_t)q_scale;
    out[2] = (uint8_t)(y_bytes  >> 8);
    out[3] = (uint8_t)(y_bytes  & 0xFF);
    out[4] = (uint8_t)(cb_bytes >> 8);
    out[5] = (uint8_t)(cb_bytes & 0xFF);
    out[6] = (uint8_t)(a_bytes  >> 8);   // alpha_size (0 when !has_alpha)
    out[7] = (uint8_t)(a_bytes  & 0xFF);

    BitPacker bp;

    // Y component
    bp_init(&bp, out + 8);
    encode_dc_plane(y_blks,  y_n, &bp);
    encode_ac_plane(y_blks,  y_n, &bp);
    bp_pad_byte(&bp);

    // Cb component
    bp_init(&bp, out + 8 + y_bytes);
    encode_dc_plane(cb_blks, c_n, &bp);
    encode_ac_plane(cb_blks, c_n, &bp);
    bp_pad_byte(&bp);

    // Cr component (size is implicit: total - 8 - y - cb - a)
    bp_init(&bp, out + 8 + y_bytes + cb_bytes);
    encode_dc_plane(cr_blks, c_n, &bp);
    encode_ac_plane(cr_blks, c_n, &bp);
    bp_pad_byte(&bp);

    // Alpha component (placed after Cr, per Apple ProRes spec)
    if (has_alpha) {
        bp_init(&bp, out + 8 + y_bytes + cb_bytes + cr_bytes);
        encode_dc_plane(alpha_blks, a_n, &bp);
        encode_ac_plane(alpha_blks, a_n, &bp);
        bp_pad_byte(&bp);
    }
}

// ---------------------------------------------------------------------------
// Host: run the 4444 two-pass entropy pipeline
// d_bit_counts must be [num_slices * 4]
// ---------------------------------------------------------------------------
void cuda_prores_enc_frame_raw_444(
    const int16_t *d_coeffs_slice,
    int            num_slices,
    int            mbs_per_slice,
    int            q_scale,
    uint8_t       *d_output,
    uint32_t      *d_slice_offsets,
    uint32_t      *d_bit_counts,    // [num_slices * 4]
    uint32_t      *d_slice_sizes,
    void          *d_cub_temp,
    size_t         cub_temp_bytes,
    cudaStream_t   stream,
    bool           has_alpha)
{
    const int T = 128;
    const int G = (num_slices + T - 1) / T;

    // Pass 1: bit counts (4 entries per slice: Y, Cb, Cr, A)
    k_count_bits_444<<<G, T, 0, stream>>>(
        d_coeffs_slice, d_bit_counts, mbs_per_slice, num_slices, (int)has_alpha);

    // Bit -> byte (ceil) for all 4 * num_slices entries
    const int G4 = (num_slices * 4 + T - 1) / T;
    k_bits_to_bytes<<<G4, T, 0, stream>>>(d_bit_counts, num_slices * 4);

    // Per-slice total byte count = 8 (header) + Y + Cb + Cr + A
    k_compute_slice_sizes_444<<<G, T, 0, stream>>>(d_bit_counts, d_slice_sizes, num_slices);

    // Exclusive prefix sum -> per-slice byte offsets
    cub::DeviceScan::ExclusiveSum(
        d_cub_temp, cub_temp_bytes,
        d_slice_sizes, d_slice_offsets,
        num_slices, stream);

    // Set total bytes at d_slice_offsets[num_slices]
    k_set_total<<<1, 1, 0, stream>>>(d_slice_offsets, d_slice_sizes, num_slices);

    // Pass 2: write slices
    k_encode_slices_444<<<G, T, 0, stream>>>(
        d_coeffs_slice, d_slice_offsets, d_bit_counts, d_output,
        q_scale, mbs_per_slice, num_slices, (int)has_alpha);
}