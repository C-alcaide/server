// cuda_prores_frame.cu / .h
// ProRes 422 frame-level encoder: orchestrates V210 unpack → DCT/quant →
// entropy coding → ProRes picture + slice table header assembly.
//
// Reference: Apple ProRes White Paper §§ "Frame Layout", "Picture Header",
//            "Slice Data", "Slice Header".
// ---------------------------------------------------------------------------
#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

// ---------------------------------------------------------------------------
// ProRes frame layout (simplified, 422 progressive)
// ─────────────────────────────────────────────────────────────────────────────
//  [0..3]   Frame size (big-endian uint32, includes these 4 bytes)
//  [4..7]   'icpf' magic
//  [8..9]   Picture header size (= header_size, big-endian uint16)
//  Picture header (variable, starts at offset 8):
//    [0..1]  header_size (big-endian uint16, size of picture header incl. size field)
//    [2..3]  version (= 0)
//    [4..7]  encoder ID ('acfd' typical; we use 'CUDA')
//    [8..9]  width  (pixels, big-endian uint16)
//    [10..11] height (pixels, big-endian uint16)
//    [12]    chroma format:  0=4444, 2=422
//    [13]    interlace flags: 0=progressive
//    [14]    aspect ratio (0=square pixels)
//    [15]    color primaries (1=Rec.709, 9=BT.2020)
//    [16]    transfer function (1=Rec.709, 14=HLG, 16=PQ)
//    [17]    color matrix (1=Rec.709, 9=BT.2020-NCL)
//    [18]    source format: 5=4K
//    [19]    alpha channel: 0=none
//    [20]    reserved[0]
//    [21]    luma quant matrix present flag (1 byte)
//    [22..85] luma quant matrix (64 bytes, if flag=1)
//    [86]    chroma quant matrix present flag
//    [87..150] chroma quant matrix (64 bytes, if flag=1)
//  Slice table: uint32_t[num_slices-1] offsets relative to slice data start
//  Slice data: concatenated Rice-coded slices
// ---------------------------------------------------------------------------

#include "cuda_prores_tables.cuh"

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Color metadata for HDR/HLG/SDR
// ---------------------------------------------------------------------------
struct ProResColorDesc {
    uint8_t color_primaries;    // 1=Rec.709, 9=BT.2020
    uint8_t transfer_function;  // 1=Rec.709, 14=HLG, 16=PQ
    uint8_t color_matrix;       // 1=Rec.709, 9=BT.2020-NCL

    // HDR mastering display (ST 2086) — zero if SDR
    uint16_t mdcv_primaries_x[3]; // in units of 0.00002 (CIE 1931)
    uint16_t mdcv_primaries_y[3];
    uint16_t mdcv_white_x, mdcv_white_y;
    uint32_t mdcv_max_lum;  // in units of 0.0001 cd/m²
    uint32_t mdcv_min_lum;
    uint16_t clli_max_cll;  // MaxCLL  (nits)
    uint16_t clli_max_fall; // MaxFALL (nits)
};

// Standard SDR Rec.709 preset
static const ProResColorDesc COLOR_DESC_SDR_709 = {
    .color_primaries   = 1,
    .transfer_function = 1,
    .color_matrix      = 1,
};

// HDR HLG BT.2020
static const ProResColorDesc COLOR_DESC_HDR_HLG = {
    .color_primaries   = 9,
    .transfer_function = 14,
    .color_matrix      = 9,
};

// HDR PQ (HDR10)
static const ProResColorDesc COLOR_DESC_HDR_PQ = {
    .color_primaries   = 9,
    .transfer_function = 16,
    .color_matrix      = 9,
};

// ---------------------------------------------------------------------------
// Frame encoder context (GPU-side resources)
// ---------------------------------------------------------------------------
struct ProResFrameCtx {
    // Frame dimensions
    int width, height;
    int profile;          // ProResProfile enum
    int slices_per_row;   // typically 4 for 4K (configurable)
    int num_slices;       // = slices_per_row * (height / 8)  [macroblock rows]
    int blocks_per_slice; // luma + chroma blocks per slice

    // Device buffers (all allocated by prores_frame_ctx_create)
    int16_t  *d_y,  *d_cb, *d_cr;    // unpacked planar input
    int16_t  *d_coeffs_y;             // DCT coefficients luma  [nblocks_y * 64]
    int16_t  *d_coeffs_cb;            // DCT coefficients Cb    [nblocks_c * 64]
    int16_t  *d_coeffs_cr;            // DCT coefficients Cr    [nblocks_c * 64]
    int16_t  *d_coeffs_slice;         // interleaved slice layout [num_slices][bps*64]
    uint8_t  *d_bitstream;            // output encoded slice data (worst-case sized)
    uint32_t *d_slice_offsets;        // [num_slices + 1] byte offsets
    uint32_t *d_bit_counts;           // [num_slices] temp for CUB scan
    void     *d_cub_temp;
    size_t    cub_temp_bytes;

    // Pinned host output buffer (receives completed frame for async copy-out)
    uint8_t  *h_frame_buf;
    size_t    h_frame_buf_size;

    int q_scale; // current adaptive quality scale [1..31]
};

#ifdef __cplusplus
}
#endif
