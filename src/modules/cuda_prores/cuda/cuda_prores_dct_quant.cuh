// cuda_prores_dct_quant.cuh
// CUDA kernel for forward 8×8 DCT + ProRes quantisation.
//
// Pipeline:
//   Planar int16_t YUV422P10 (values 0..1023 from V210 unpack)
//   → Forward 8×8 DCT — raw 10-bit values, NO level shift
//   → Quantise (divide by q_table[scan[i]] * q_scale, rounding)
//   → Scan-reorder using PRORES_SCAN_ORDER (ProRes progressive scan)
//   → Output: int16_t coefficients [num_slices][blocks_per_slice][64]
//
// DCT convention (matches FFmpeg ff_jpeg_fdct_islow_10):
//   Input:  raw uint10 values [0, 1023], stored as int32_t.
//   Output: scaled by 8 relative to the true DCT (absorbed by quantisation).
//   For a flat block of all-512 the DC coefficient = 512 × 32 = 0x4000.
//   encode_dcs subtracts this bias: (dc - 0x4000) / scale.
//
//   Row pass: PASS1_BITS=1, CONST_BITS=13, output scaled ×2.
//   Column pass: OUT_SHIFT=2; result fits in int16_t.
//
// Strategy
//   One CUDA thread block per 8×8 input block (64 threads).
//   For 4K (3840×2160) with 4 slices/row:
//     luma blocks  = 480 × 270 = 129 600
//     chroma blocks= 64 800 × 2 (Cb + Cr)
#pragma once

#include "cuda_prores_tables.cuh"
#include <cuda_runtime.h>
#include <stdint.h>

// ---------------------------------------------------------------------------
// IJG Loeffler 8-point forward DCT — integer, 2-pass (row + column).
// Matches FFmpeg ff_jpeg_fdct_islow_10 (jfdctint_template.c, BITS_IN_JSAMPLE=10).
//
// Constants (CONST_BITS=13):
//   FIX(0.298631336)=2446  FIX(0.390180644)=3196  FIX(0.541196100)=4433
//   FIX(0.765366865)=6270  FIX(0.899976223)=7373   FIX(1.175875602)=9633
//   FIX(1.501321110)=12299 FIX(1.847759065)=15137  FIX(1.961570560)=16069
//   FIX(2.053119869)=16819 FIX(2.562915447)=20995  FIX(3.072711026)=25172
//
// Row pass: PASS1_BITS=1; DC/[4] output multiplied by 2 (no DESCALE),
//           AC outputs DESCALEd by CONST_BITS-PASS1_BITS = 12.
// Column pass: outputs DESCALEd by CONST_BITS+OUT_SHIFT = 13+2 = 15.
//
// DESCALE(x, n) = (x + (1 << (n-1))) >> n   (half-up rounding)
// ---------------------------------------------------------------------------

// Row pass: v[8] are raw 10-bit pixel values (0..1023), in-place → row DCT.
// Output is scaled ×2 for even terms (v[0], v[4]) and DESCALEd for the rest.
__device__ __forceinline__ void dct_row_pass(int32_t *v)
{
    int32_t tmp0 = v[0]+v[7], tmp7 = v[0]-v[7];
    int32_t tmp1 = v[1]+v[6], tmp6 = v[1]-v[6];
    int32_t tmp2 = v[2]+v[5], tmp5 = v[2]-v[5];
    int32_t tmp3 = v[3]+v[4], tmp4 = v[3]-v[4];

    int32_t tmp10 = tmp0+tmp3, tmp13 = tmp0-tmp3;
    int32_t tmp11 = tmp1+tmp2, tmp12 = tmp1-tmp2;

    // Even part: multiply by 2^PASS1_BITS = 2 (no DESCALE needed)
    v[0] = (tmp10 + tmp11) << 1;
    v[4] = (tmp10 - tmp11) << 1;
    int32_t ze = (tmp12 + tmp13) * 4433; // FIX_0_541196100
    v[2] = (ze + tmp13 * 6270  + (1 << 11)) >> 12; // DESCALE(,12)
    v[6] = (ze - tmp12 * 15137 + (1 << 11)) >> 12;

    // Odd part (DESCALE shift = 12)
    int32_t z1 = (tmp4 + tmp7) * (-7373);  // -FIX_0_899976223
    int32_t z2 = (tmp5 + tmp6) * (-20995); // -FIX_2_562915447
    int32_t z3 = (tmp4 + tmp6) * (-16069); // -FIX_1_961570560
    int32_t z4 = (tmp5 + tmp7) * (-3196);  // -FIX_0_390180644
    int32_t z5 = (tmp4 + tmp5 + tmp6 + tmp7) * 9633; // FIX_1_175875602
    z3 += z5;
    z4 += z5;
    v[7] = (tmp4 *  2446 + z1 + z3 + (1 << 11)) >> 12; // FIX_0_298631336
    v[5] = (tmp5 * 16819 + z2 + z4 + (1 << 11)) >> 12; // FIX_2_053119869
    v[3] = (tmp6 * 25172 + z2 + z3 + (1 << 11)) >> 12; // FIX_3_072711026
    v[1] = (tmp7 * 12299 + z1 + z4 + (1 << 11)) >> 12; // FIX_1_501321110
}

// Column pass: v[8] are row-pass outputs; in-place → column DCT.
// Outputs DESCALEd by CONST_BITS + OUT_SHIFT = 15.
__device__ __forceinline__ void dct_col_pass(int32_t *v)
{
    int32_t tmp0 = v[0]+v[7], tmp7 = v[0]-v[7];
    int32_t tmp1 = v[1]+v[6], tmp6 = v[1]-v[6];
    int32_t tmp2 = v[2]+v[5], tmp5 = v[2]-v[5];
    int32_t tmp3 = v[3]+v[4], tmp4 = v[3]-v[4];

    int32_t tmp10 = tmp0+tmp3, tmp13 = tmp0-tmp3;
    int32_t tmp11 = tmp1+tmp2, tmp12 = tmp1-tmp2;

    // Even part (DESCALE shift = OUT_SHIFT = 2 for DC/[4],
    //            CONST_BITS + OUT_SHIFT = 15 for v[2]/v[6])
    v[0] = (tmp10 + tmp11 + 2) >> 2;
    v[4] = (tmp10 - tmp11 + 2) >> 2;
    int32_t ze = (tmp12 + tmp13) * 4433;
    v[2] = (ze + tmp13 * 6270  + (1 << 14)) >> 15;
    v[6] = (ze - tmp12 * 15137 + (1 << 14)) >> 15;

    // Odd part (DESCALE shift = 15)
    int32_t z1 = (tmp4 + tmp7) * (-7373);
    int32_t z2 = (tmp5 + tmp6) * (-20995);
    int32_t z3 = (tmp4 + tmp6) * (-16069);
    int32_t z4 = (tmp5 + tmp7) * (-3196);
    int32_t z5 = (tmp4 + tmp5 + tmp6 + tmp7) * 9633;
    z3 += z5;
    z4 += z5;
    v[7] = (tmp4 *  2446 + z1 + z3 + (1 << 14)) >> 15;
    v[5] = (tmp5 * 16819 + z2 + z4 + (1 << 14)) >> 15;
    v[3] = (tmp6 * 25172 + z2 + z3 + (1 << 14)) >> 15;
    v[1] = (tmp7 * 12299 + z1 + z4 + (1 << 14)) >> 15;
}

// ---------------------------------------------------------------------------
// DCT + quantise kernel
// One thread block = one 8×8 block.
// blockIdx.x = linear block index (luma blocks first, then chroma).
// ---------------------------------------------------------------------------
__global__ void k_dct_quantise(
    const int16_t * __restrict__ d_plane,  // input plane (Y, Cb, or Cr)
    int16_t       * __restrict__ d_out_coeffs, // output [num_blocks][64]
    int plane_width,   // luma: width, chroma: width/2
    int plane_height,  // same for luma and chroma
    int q_scale,       // adaptive quality scale [1..31]; 1 = best quality
    int profile,       // ProResProfile index
    bool is_chroma,
    bool is_interlaced)
{
    // Each thread handles one coefficient position within the 8×8 block.
    const int tid   = threadIdx.x; // 0..63
    const int blk_x = blockIdx.x;  // block column in plane
    const int blk_y = blockIdx.y;  // block row in plane

    const int blocks_per_row = plane_width  / 8;
    const int blk_idx        = blk_y * blocks_per_row + blk_x;

    // Shared memory for the 8×8 block
    __shared__ int32_t s_block[64];

    const int px = blk_x * 8 + (tid % 8);
    const int py = blk_y * 8 + (tid / 8);

    // Load pixel — NO level shift; raw 10-bit value [0, 1023].
    // FFmpeg prores_fdct copies uint16_t directly into int16_t without shifting.
    // The DC bias (0x4000) is subtracted in encode_dcs, not here.
    if (px < plane_width && py < plane_height)
        s_block[tid] = (int32_t)d_plane[py * plane_width + px];
    else
        s_block[tid] = 0; // zero-pad partial blocks at frame edge
    __syncthreads();

    // Row DCT: thread i handles row i (using dct_row_pass)
    if (tid < 8) {
        int32_t row[8];
        for (int c = 0; c < 8; c++) row[c] = s_block[tid * 8 + c];
        dct_row_pass(row);
        for (int c = 0; c < 8; c++) s_block[tid * 8 + c] = row[c];
    }
    __syncthreads();

    // Column DCT: thread i handles column i (using dct_col_pass)
    if (tid < 8) {
        int32_t col[8];
        for (int r = 0; r < 8; r++) col[r] = s_block[r * 8 + tid];
        dct_col_pass(col);
        for (int r = 0; r < 8; r++) s_block[r * 8 + tid] = col[r];
    }
    __syncthreads();

    // Quantise and scan-reorder — look up quant table from __constant__ memory.
    // Direct constant memory access is correct here; passing a host-side pointer
    // to a __constant__ symbol as a kernel parameter would give a wrong address.
    uint8_t  q_val = (is_chroma ? c_quant_chroma[profile][tid]
                                : c_quant_luma  [profile][tid]);
    int32_t  denom = (int32_t)q_val * q_scale;
    // ProRes quantisation: round-half-away-from-zero, then clamp to int16_t
    int32_t  raw   = s_block[(is_interlaced ? c_scan_order_interlaced : c_scan_order)[tid]]; // scan reorder
    // Subtract the DC bias before quantising the DC coefficient (tid==0,
    // c_scan_order[0]==0).  FFmpeg's encode_dcs does (blocks[0] - 0x4000) / scale;
    // a flat 512-value block produces DCT DC = 512*32 = 16384 = 0x4000, so
    // bias removal yields DC=0 for a neutral-grey block.
    if (tid == 0) raw -= 0x4000;
    int32_t  qcoef;
    if (denom == 0) {
        qcoef = 0;
    } else {
        qcoef = (raw + (raw >= 0 ? denom / 2 : -(denom / 2))) / denom;
    }
    qcoef = max(-32768, min(32767, qcoef));

    d_out_coeffs[blk_idx * 64 + tid] = (int16_t)qcoef;
}

// Launcher — call for luma plane then each chroma plane separately.
// The kernel looks up the quant table from __constant__ memory directly.
inline cudaError_t launch_dct_quantise(
    const int16_t  *d_plane,
    int16_t        *d_out_coeffs,
    int plane_width, int plane_height,
    int q_scale, int profile, bool is_chroma,
    bool is_interlaced,
    cudaStream_t stream)
{
    dim3 threads(64);
    // Use ceiling division for block rows so that a partial last block row
    // (e.g. field_height=540: 540/8=67 full rows but the 68th covers lines
    // 536-543 with lines 540-543 zero-padded by the kernel's bounds check)
    // is correctly processed.  For heights divisible by 8 (e.g. 1080, 720)
    // this produces the same result as floor division.
    dim3 blocks(plane_width / 8, (plane_height + 7) / 8);
    k_dct_quantise<<<blocks, threads, 0, stream>>>(
        d_plane, d_out_coeffs,
        plane_width, plane_height, q_scale, profile, is_chroma, is_interlaced);
    return cudaGetLastError();
}

// ---------------------------------------------------------------------------
// Adaptive q_scale: binary search targeting profile bitrate.
// Runs k_count_bits-equivalent logic (via cuda_prores_entropy) to find
// q_scale in [1..31] that hits ≤ target_bytes per frame.
// The actual binary search is done on the CPU after one counting pass;
// described in cuda_prores_frame.cu which coordinates the multi-pass loop.
// ---------------------------------------------------------------------------
static constexpr int PRORES_QSCALE_MIN = 1;
static constexpr int PRORES_QSCALE_MAX = 31;
