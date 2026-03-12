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
    PRORES_4444     = 4,  // ap4h  ~330 Mb/s (422 variant used here, not XQ)
    PRORES_PROFILE_COUNT = 5
};

// Four-character codes for use in container stsd entries.
static constexpr uint32_t PRORES_TAG[PRORES_PROFILE_COUNT] = {
    0x6170636F, // 'apco'
    0x6170636C, // 'apcl'
    0x6170636E, // 'apcn'
    0x61706368, // 'apch'
    0x61703468, // 'ap4h'
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
// Target bitrate per profile (Mbit/s) at 4K (3840×2160) 25p
// Used for adaptive q_scale selection.
// Values based on Apple ProRes White Paper Table 1 scaled to 4K.
// ---------------------------------------------------------------------------
static constexpr int PRORES_TARGET_MBPS[PRORES_PROFILE_COUNT] = {
    180,   // PROXY   (~45 Mb/s at 1080p → ×4 for 4K pixel count)
    410,   // LT
    590,   // STANDARD
    880,   // HQ
    1300,  // 4444
};

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
