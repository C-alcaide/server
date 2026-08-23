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

// cuda_prores_tables.cuh
// ProRes quantisation matrices, scan order, and bitrate targets for all profiles.
//
// Quantisation matrices derived from:
//   FFmpeg libavcodec/proresenc_kostya.c (LGPL 2.1+, Kostya Shishkov)
//   which in turn implements the Apple ProRes White Paper specification.
//
// The scan order is NOT JPEG zigzag — it is the ProRes-specific order defined
// in the Apple ProRes White Paper Table 2.
//
// Data is uploaded to CUDA __constant__ memory by prores_tables_upload() and
// should be called once at module/encoder init.
#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

// ---------------------------------------------------------------------------
// ProRes profile indices
// ---------------------------------------------------------------------------
enum ProResProfile : int {
    PRORES_PROXY    = 0,  // apco  ~45 Mb/s @ 1080p29.97
    PRORES_LT       = 1,  // apcl  ~102 Mb/s
    PRORES_STANDARD = 2,  // apcn  ~147 Mb/s
    PRORES_HQ       = 3,  // apch  ~220 Mb/s
    PRORES_4444     = 4,  // ap4h  ~330 Mb/s
    PRORES_4444_XQ  = 5,  // ap4x  ~500 Mb/s
    PRORES_PROFILE_COUNT = 6
};

// Four-character codes for use in container stsd entries.
static constexpr uint32_t PRORES_TAG[PRORES_PROFILE_COUNT] = {
    0x6170636F, // 'apco'
    0x6170636C, // 'apcl'
    0x6170636E, // 'apcn'
    0x61706368, // 'apch'
    0x61703468, // 'ap4h'
    0x61703478, // 'ap4x'
};

// ---------------------------------------------------------------------------
// Quantisation matrices (8×8, luma and chroma, per profile)
// Source: FFmpeg proresenc_kostya.c and Apple ProRes White Paper Table 7.
// ---------------------------------------------------------------------------

// Each matrix is listed in natural raster order (not scan order).
// The DCT kernel applies both the scan permutation and quantisation together.

static constexpr uint8_t PRORES_QUANT_LUMA[PRORES_PROFILE_COUNT][64] = {
    // PROXY (apco) – aggressive quantisation
    {
         4,  7,  9, 11, 13, 14, 15, 63,
         7,  7, 11, 12, 14, 15, 63, 63,
         9, 11, 13, 14, 15, 63, 63, 63,
        11, 11, 13, 14, 63, 63, 63, 63,
        11, 13, 14, 63, 63, 63, 63, 63,
        13, 14, 63, 63, 63, 63, 63, 63,
        13, 63, 63, 63, 63, 63, 63, 63,
        63, 63, 63, 63, 63, 63, 63, 63,
    },
    // LT (apcl)
    {
         4,  5,  6,  7,  9, 11, 13, 15,
         5,  5,  7,  8, 11, 13, 15, 17,
         6,  7,  9, 11, 13, 15, 15, 17,
         7,  7,  9, 11, 13, 15, 17, 19,
         7,  9, 11, 13, 14, 16, 19, 23,
         9, 11, 13, 14, 16, 19, 23, 29,
         9, 11, 13, 15, 17, 21, 28, 35,
        11, 13, 16, 17, 21, 28, 35, 41,
    },
    // STANDARD (apcn) — the Apple "reference" matrix
    {
         4,  4,  5,  5,  6,  7,  7,  8,
         4,  4,  5,  6,  7,  7,  8,  9,
         5,  5,  6,  7,  7,  8,  9, 10,
         5,  5,  6,  7,  8,  9, 10, 11,
         5,  6,  7,  8,  9, 10, 11, 12,
         6,  7,  8,  9, 10, 11, 12, 13,
         6,  7,  8,  9, 10, 11, 13, 14,
         7,  8,  9, 10, 11, 12, 14, 15,
    },
    // HQ (apch) — fine quantisation
    {
         4,  4,  4,  4,  4,  4,  4,  4,
         4,  4,  4,  4,  4,  4,  4,  4,
         4,  4,  4,  4,  4,  4,  4,  4,
         4,  4,  4,  4,  4,  4,  4,  5,
         4,  4,  4,  4,  4,  4,  5,  5,
         4,  4,  4,  4,  4,  5,  5,  6,
         4,  4,  4,  4,  5,  5,  6,  7,
         4,  4,  4,  5,  5,  6,  7,  7,
    },
    // 4444 (ap4h) — same as HQ for 422 path
    {
         4,  4,  4,  4,  4,  4,  4,  4,
         4,  4,  4,  4,  4,  4,  4,  4,
         4,  4,  4,  4,  4,  4,  4,  4,
         4,  4,  4,  4,  4,  4,  4,  5,
         4,  4,  4,  4,  4,  4,  5,  5,
         4,  4,  4,  4,  4,  5,  5,  6,
         4,  4,  4,  4,  5,  5,  6,  7,
         4,  4,  4,  5,  5,  6,  7,  7,
    },
    // 4444 XQ (ap4x) — the reference encoder uses the HQ matrix here too
    // (proresenc_kostya_common.c carries a 'Fix me : use QUANT_MAT_XQ_LUMA' on this
    // very entry, so matching it is matching the only shipping behaviour there is).
    {
         4,  4,  4,  4,  4,  4,  4,  4,
         4,  4,  4,  4,  4,  4,  4,  4,
         4,  4,  4,  4,  4,  4,  4,  4,
         4,  4,  4,  4,  4,  4,  4,  5,
         4,  4,  4,  4,  4,  4,  5,  5,
         4,  4,  4,  4,  4,  5,  5,  6,
         4,  4,  4,  4,  5,  5,  6,  7,
         4,  4,  4,  5,  5,  6,  7,  7,
    },
};

static constexpr uint8_t PRORES_QUANT_CHROMA[PRORES_PROFILE_COUNT][64] = {
    // PROXY
    {
         4,  7,  9, 11, 13, 14, 63, 63,
         7,  7, 11, 12, 14, 63, 63, 63,
         9, 11, 13, 14, 63, 63, 63, 63,
        11, 11, 13, 63, 63, 63, 63, 63,
        11, 13, 63, 63, 63, 63, 63, 63,
        13, 63, 63, 63, 63, 63, 63, 63,
        63, 63, 63, 63, 63, 63, 63, 63,
        63, 63, 63, 63, 63, 63, 63, 63,
    },
    // LT
    {
         4,  5,  6,  7,  9, 11, 13, 15,
         5,  5,  7,  8, 11, 13, 15, 17,
         6,  7,  9, 11, 13, 15, 15, 17,
         7,  7,  9, 11, 13, 15, 17, 19,
         7,  9, 11, 13, 14, 16, 19, 23,
         9, 11, 13, 14, 16, 19, 23, 29,
         9, 11, 13, 15, 17, 21, 28, 35,
        11, 13, 16, 17, 21, 28, 35, 41,
    },
    // STANDARD
    {
         4,  4,  5,  5,  6,  7,  7,  8,
         4,  4,  5,  6,  7,  7,  8,  9,
         5,  5,  6,  7,  7,  8,  9, 10,
         5,  5,  6,  7,  8,  9, 10, 11,
         5,  6,  7,  8,  9, 10, 11, 12,
         6,  7,  8,  9, 10, 11, 12, 13,
         6,  7,  8,  9, 10, 11, 13, 14,
         7,  8,  9, 10, 11, 12, 14, 15,
    },
    // HQ
    {
         4,  4,  4,  4,  4,  4,  4,  4,
         4,  4,  4,  4,  4,  4,  4,  4,
         4,  4,  4,  4,  4,  4,  4,  4,
         4,  4,  4,  4,  4,  4,  4,  5,
         4,  4,  4,  4,  4,  4,  5,  5,
         4,  4,  4,  4,  4,  5,  5,  6,
         4,  4,  4,  4,  5,  5,  6,  7,
         4,  4,  4,  5,  5,  6,  7,  7,
    },
    // 4444
    {
         4,  4,  4,  4,  4,  4,  4,  4,
         4,  4,  4,  4,  4,  4,  4,  4,
         4,  4,  4,  4,  4,  4,  4,  4,
         4,  4,  4,  4,  4,  4,  4,  5,
         4,  4,  4,  4,  4,  4,  5,  5,
         4,  4,  4,  4,  4,  5,  5,  6,
         4,  4,  4,  4,  5,  5,  6,  7,
         4,  4,  4,  5,  5,  6,  7,  7,
    },
    // 4444 XQ — HQ chroma, as above
    {
         4,  4,  4,  4,  4,  4,  4,  4,
         4,  4,  4,  4,  4,  4,  4,  4,
         4,  4,  4,  4,  4,  4,  4,  4,
         4,  4,  4,  4,  4,  4,  4,  5,
         4,  4,  4,  4,  4,  4,  5,  5,
         4,  4,  4,  4,  4,  5,  5,  6,
         4,  4,  4,  4,  5,  5,  6,  7,
         4,  4,  4,  5,  5,  6,  7,  7,
    },
};

// ---------------------------------------------------------------------------
// ProRes scan orders
// Source: FFmpeg libavcodec/proresdata.c — ff_prores_progressive_scan and
//         ff_prores_interlaced_scan.  These are NOT the JPEG zigzag.
// Maps output-scan-index → natural-raster-index (r*8+c).
// ---------------------------------------------------------------------------

// Progressive scan (default for all CasparCG frames).
static constexpr uint8_t PRORES_SCAN_ORDER[64] = {
     0,  1,  8,  9,  2,  3, 10, 11,
    16, 17, 24, 25, 18, 19, 26, 27,
     4,  5, 12, 20, 13,  6,  7, 14,
    21, 28, 29, 22, 15, 23, 30, 31,
    32, 33, 40, 48, 41, 34, 35, 42,
    49, 56, 57, 50, 43, 36, 37, 44,
    51, 58, 59, 52, 45, 38, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63,
};

// Interlaced scan — used when the frame-flags byte signals interlaced DCT.
static constexpr uint8_t PRORES_SCAN_ORDER_INTERLACED[64] = {
     0,  8,  1,  9, 16, 24, 17, 25,
     2, 10,  3, 11, 18, 26, 19, 27,
    32, 40, 33, 34, 41, 48, 56, 49,
    42, 35, 43, 50, 57, 58, 51, 59,
     4, 12,  5,  6, 13, 20, 28, 21,
    14,  7, 15, 22, 29, 36, 44, 37,
    30, 23, 31, 38, 45, 52, 60, 53,
    46, 39, 47, 54, 61, 62, 55, 63,
};

// ---------------------------------------------------------------------------
// Target data rate per profile, in BITS PER MACROBLOCK
//
// This is how ProRes rates are actually specified, and why the same number covers 1080p
// and 4K: the rate is per picture area, so a raster with four times the macroblocks gets
// four times the bits at the same quality. Apple publishes Mbit/s figures for named
// formats; the per-macroblock form is what an encoder can aim at.
//
// Source: FFmpeg libavcodec/proresenc_kostya_common.c, `prores_profile_info[].br_tab`
// and `prores_mb_limits` — the reference encoder's targets, which sit about 5.6% above
// Apple's nominal figures (950 bits/MB is 193.8 Mbit/s at 1080p25 against Apple's 183.5).
//
// Four buckets by picture size. The bucket is chosen by the FIRST limit that is >= the
// macroblock count, and the loop stops at index 2 — so index 3 covers everything above
// 6075 macroblocks, which is every raster from 1440x1080 upward including 4K.
// ---------------------------------------------------------------------------
static constexpr int PRORES_MB_LIMITS[4] = {
    1620,  // up to 720x576
    2700,  // up to 960x720
    6075,  // up to 1440x1080
    9216,  // up to 2048x1152 -- and, because of the loop bound, everything larger
};

static constexpr int PRORES_BR_TAB[PRORES_PROFILE_COUNT][4] = {
    {  300,  242,  220,  194 },  // PROXY
    {  720,  560,  490,  440 },  // LT
    { 1050,  808,  710,  632 },  // STANDARD
    { 1566, 1216, 1070,  950 },  // HQ
    { 2350, 1828, 1600, 1425 },  // 4444
    { 3525, 2742, 2400, 2137 },  // 4444 XQ
};

// Target bits per macroblock for a given profile and picture geometry.
// `mbs_per_picture` counts macroblocks in ONE picture; interlaced frames carry two.
static inline int prores_target_bits_per_mb(int profile, int mbs_per_picture,
                                           int pictures_per_frame)
{
    if (profile < 0) profile = 0;
    if (profile >= PRORES_PROFILE_COUNT) profile = PRORES_PROFILE_COUNT - 1;
    const int total = mbs_per_picture * pictures_per_frame;
    int i = 0;
    for (; i < 3; i++)
        if (PRORES_MB_LIMITS[i] >= total)
            break;
    return PRORES_BR_TAB[profile][i];
}
// ---------------------------------------------------------------------------
// CUDA constant memory (uploaded at encoder init)
//
// Define PRORES_TABLES_DEFINE_CONSTANTS before including this header in the
// ONE translation unit that should own the definitions (cuda_prores_entropy.cu).
// All other TUs get extern declarations only.
// ---------------------------------------------------------------------------
#ifdef PRORES_TABLES_DEFINE_CONSTANTS
__constant__ uint8_t c_quant_luma  [PRORES_PROFILE_COUNT][64];
__constant__ uint8_t c_quant_chroma[PRORES_PROFILE_COUNT][64];
__constant__ uint8_t c_scan_order            [64]; // progressive
__constant__ uint8_t c_scan_order_interlaced [64]; // interlaced
#else
extern __constant__ uint8_t c_quant_luma  [PRORES_PROFILE_COUNT][64];
extern __constant__ uint8_t c_quant_chroma[PRORES_PROFILE_COUNT][64];
extern __constant__ uint8_t c_scan_order            [64];
extern __constant__ uint8_t c_scan_order_interlaced [64];
#endif

// Call once per CUDA context (before any encode kernel launch).
inline cudaError_t prores_tables_upload()
{
    cudaError_t e;
    e = cudaMemcpyToSymbol(c_quant_luma,   PRORES_QUANT_LUMA,
                           sizeof(PRORES_QUANT_LUMA));
    if (e != cudaSuccess) return e;
    e = cudaMemcpyToSymbol(c_quant_chroma, PRORES_QUANT_CHROMA,
                           sizeof(PRORES_QUANT_CHROMA));
    if (e != cudaSuccess) return e;
    e = cudaMemcpyToSymbol(c_scan_order, PRORES_SCAN_ORDER,
                           sizeof(PRORES_SCAN_ORDER));
    if (e != cudaSuccess) return e;
    e = cudaMemcpyToSymbol(c_scan_order_interlaced, PRORES_SCAN_ORDER_INTERLACED,
                           sizeof(PRORES_SCAN_ORDER_INTERLACED));
    return e;
}
