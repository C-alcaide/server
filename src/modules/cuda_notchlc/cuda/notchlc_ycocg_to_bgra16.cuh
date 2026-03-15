// notchlc_ycocg_to_bgra16.cuh
// CUDA kernel: NotchLC YCbCr inverse transform → BGRA16.
//
// NotchLC bitstreams store standard YCbCr (BT.601 coefficients by default),
// NOT YCoCg-R as the codec name might imply.  The decoded planes are:
//   Y  (12-bit [0,4095]) — luma
//   Cb (12-bit, center 2048) — blue-difference chroma  (= U plane)
//   Cr (12-bit, center 2048) — red-difference chroma   (= V plane)
//
// Inverse matrix (BT.601 full-range):
//   Cb = U - 2048,  Cr = V - 2048
//   R  = Y + 1.402000 * Cr
//   G  = Y - 0.344136 * Cb - 0.714136 * Cr
//   B  = Y + 1.772000 * Cb
//
// The resulting R, G, B are 12-bit gamma-encoded display values [0,4095].
// They are scaled to 16-bit for output (shift by 4), with no additional
// gamma correction needed (the signal is already in display gamma space).
// For LINEAR mode, the stored gamma-encoded values are expanded to linear
// light via the Rec.709 EOTF applied inline.
// ---------------------------------------------------------------------------
#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

namespace caspar { namespace cuda_notchlc {

// COLOR_MATRIX constants (matching prores_producer convention):
//   -1 = AUTO (default; for NotchLC: same as 709 since no in-band metadata)
//    1 = BT.709
//    6 = BT.601
//    9 = BT.2020
//  100 = LINEAR  (no gamma, just 12→16 bit shift)
static constexpr int NOTCHLC_CM_AUTO    = -1;
static constexpr int NOTCHLC_CM_709     =  1;
static constexpr int NOTCHLC_CM_601     =  6;
static constexpr int NOTCHLC_CM_2020    =  9;
static constexpr int NOTCHLC_CM_LINEAR  = 100;

// ---------------------------------------------------------------------------
// Rec.709 EOTF (encoded → linear light) applied inline for LINEAR mode.
// Input: 12-bit gamma-encoded sample [0, 4095].
// Output: 16-bit linear-light value  [0, 65535].
//
// Formula (Rec.709 EOTF, inverse of OETF):
//   ef = in / 4095.0
//   if ef <= 0.081:  lf = ef / 4.500
//   else:            lf = ((ef + 0.099) / 1.099) ^ (1 / 0.45)
//   out_16 = round(lf * 65535)
// ---------------------------------------------------------------------------

#ifdef __CUDACC__

// Rec.709 EOTF applied inline for LINEAR COLOR_MATRIX mode.
// Input v: 12-bit gamma-encoded [0, 4095].  Output: 16-bit linear [0, 65535].
__device__ __forceinline__ uint16_t apply_eotf(int v)
{
    if (v <= 0)    return 0;
    if (v >= 4095) return 65535;
    const float ef = (float)v * (1.0f / 4095.0f);
    float lf;
    if (ef <= 0.081f)
        lf = ef * (1.0f / 4.500f);
    else
        lf = powf((ef + 0.099f) * (1.0f / 1.099f), 1.0f / 0.45f);
    int out = __float2int_rn(lf * 65535.0f);
    return (uint16_t)(out < 0 ? 0 : (out > 65535 ? 65535 : out));
}

// notchlc_upload_gamma_lut() is kept as a no-op so callers do not need
// changing; the LUT constant is no longer used in the main decode path.
inline cudaError_t notchlc_upload_gamma_lut()
{
    return cudaSuccess;
}

#endif // __CUDACC__

// ---------------------------------------------------------------------------
// k_notch_ycocg_to_bgra16
//
// d_y, d_u, d_v, d_a — decoded planes, uint16_t, 12-bit values
//   d_y = luma Y12,  d_u = Cb12 (centered at 2048),  d_v = Cr12 (centered)
// width, height       — frame dimensions
// color_matrix        — one of NOTCHLC_CM_* constants
// d_out_bgra16        — output: uint16_t[height × width × 4]
//                       channel order: B, G, R, A (BGRA interleaved)
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
__global__ void k_notch_ycocg_to_bgra16(
    const uint16_t* __restrict__ d_y,
    const uint16_t* __restrict__ d_u,
    const uint16_t* __restrict__ d_v,
    const uint16_t* __restrict__ d_a,
    int      width,
    int      height,
    int      color_matrix,
    uint16_t* __restrict__ d_out_bgra16)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= width * height) return;

    // YCbCr inverse transform.
    // The U plane stores Cb and the V plane stores Cr, both centered at 2048.
    const float Y12 = (float)(int)d_y[idx];
    const float Cb  = (float)(int)d_u[idx] - 2048.0f;
    const float Cr  = (float)(int)d_v[idx] - 2048.0f;

    // Select matrix coefficients based on color_matrix.
    // Default / AUTO / BT.601: crv=1.402, cgu=0.344136, cgv=0.714136, cbu=1.772
    // BT.709:                  crv=1.5748, cgu=0.187324, cgv=0.468124, cbu=1.8556
    // BT.2020 (NCL):           crv=1.4746, cgu=0.164553, cgv=0.571353, cbu=1.8814
    float crv, cgu, cgv, cbu;
    if (color_matrix == NOTCHLC_CM_709) {
        crv = 1.5748f; cgu = 0.187324f; cgv = 0.468124f; cbu = 1.8556f;
    } else if (color_matrix == NOTCHLC_CM_2020) {
        crv = 1.4746f; cgu = 0.164553f; cgv = 0.571353f; cbu = 1.8814f;
    } else {
        // BT.601, AUTO, and any unrecognised value → BT.601 (matches FFmpeg default)
        crv = 1.402f; cgu = 0.344136f; cgv = 0.714136f; cbu = 1.772f;
    }

    int R = __float2int_rn(Y12 + crv * Cr);
    int G = __float2int_rn(Y12 - cgu * Cb - cgv * Cr);
    int B = __float2int_rn(Y12 + cbu * Cb);

    // Clamp to 12-bit
    if (R < 0) R = 0; if (R > 4095) R = 4095;
    if (G < 0) G = 0; if (G > 4095) G = 4095;
    if (B < 0) B = 0; if (B > 4095) B = 4095;
    int A = (int)d_a[idx];
    if (A < 0) A = 0; if (A > 4095) A = 4095;

    uint16_t r16, g16, b16, a16;

    if (color_matrix == NOTCHLC_CM_LINEAR) {
        // LINEAR: expand gamma-encoded 12-bit → linear-light 16-bit via EOTF.
        // The stored NotchLC data is in display gamma space; applying the EOTF
        // here gives un-gammaed linear-light values for downstream mixing.
        r16 = apply_eotf(R);
        g16 = apply_eotf(G);
        b16 = apply_eotf(B);
        a16 = (uint16_t)(A << 4);
    } else {
        // Standard (BT.601/709/2020/AUTO): output is already in gamma-encoded
        // display space after the YCbCr inverse.  Simply scale 12→16 bit.
        r16 = (uint16_t)(R << 4);
        g16 = (uint16_t)(G << 4);
        b16 = (uint16_t)(B << 4);
        a16 = (uint16_t)(A << 4);
    }

    // Write BGRA16 interleaved
    const int out_base = idx * 4;
    d_out_bgra16[out_base + 0] = b16;
    d_out_bgra16[out_base + 1] = g16;
    d_out_bgra16[out_base + 2] = r16;
    d_out_bgra16[out_base + 3] = a16;
}

#endif // __CUDACC__

}} // namespace caspar::cuda_notchlc
