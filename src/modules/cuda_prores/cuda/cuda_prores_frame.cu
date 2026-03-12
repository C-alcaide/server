// cuda_prores_frame.cu
// Frame-level orchestration: V210 unpack → DCT+quant → entropy → header build
// ---------------------------------------------------------------------------
// This TU owns the __constant__ table definitions for the entire encoder.
#define PRORES_TABLES_DEFINE_CONSTANTS
#include "cuda_prores_frame.h"
#include "cuda_prores_v210_unpack.cuh"
#include "cuda_prores_dct_quant.cuh"
#include "cuda_prores_entropy.cu"    // includes kernel definitions
#include "cuda_bgra_to_yuva444p10.cuh"  // direct BGRA->4444P10 for ProRes 4444

#include <cub/device/device_scan.cuh>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>

// ---------------------------------------------------------------------------
// Big-endian write helpers (host side, for header construction)
// ---------------------------------------------------------------------------
static inline void write_u8 (uint8_t *p, uint8_t  v)  { p[0] = v; }
static inline void write_u16(uint8_t *p, uint16_t v)  { p[0] = v >> 8; p[1] = v & 0xFF; }
static inline void write_u32(uint8_t *p, uint32_t v)  {
    p[0] = (v >> 24) & 0xFF; p[1] = (v >> 16) & 0xFF;
    p[2] = (v >>  8) & 0xFF; p[3] =  v        & 0xFF;
}

// ---------------------------------------------------------------------------
// Build the ProRes FRAME HEADER into dst.
//
// The frame header starts immediately after 'icpf' and contains:
//   [0..1]  frame_hdr_size (big-endian uint16, includes these 2 bytes)
//   [2..3]  version = 0
//   [4..7]  encoder tag (4 chars)
//   [8..9]  width  (big-endian uint16)
//   [10..11] height (big-endian uint16)
//   [12]    frame_flags (0x80 = 422 progressive)
//   [13]    reserved = 0
//   [14]    color_primaries
//   [15]    color_trc (transfer function)
//   [16]    colorspace (matrix coefficients)
//   [17]    alpha_bits >> 3 (0 = no alpha)
//   [18]    reserved = 0
//   [19]    matrix_flags (0x03 = both luma+chroma present)
//   [20..83] luma quantisation matrix (64 bytes, raster order)
//   [84..147] chroma quantisation matrix (64 bytes)
//   Total: 148 bytes.
//
// Returns bytes written.
// ---------------------------------------------------------------------------
static int build_frame_header(
    uint8_t               *dst,
    int                    width,
    int                    height,
    bool                   is_4444,
    bool                   has_alpha,
    bool                   is_interlaced,
    const ProResColorDesc *color,
    const uint8_t         *quant_luma,
    const uint8_t         *quant_chroma)
{
    uint8_t *p = dst;
    uint8_t *hdr_size_ptr = p; p += 2;   // [0..1] filled at end
    write_u16(p, 0); p += 2;             // [2..3] version = 0
    p[0]='C'; p[1]='U'; p[2]='D'; p[3]='A'; p += 4;  // [4..7] encoder tag
    write_u16(p, (uint16_t)width);  p += 2;  // [8..9]
    write_u16(p, (uint16_t)height); p += 2;  // [10..11]
    // [12] frame_flags: bits[7:6] = chroma format (0b10=4:2:2, 0b00=4:4:4)
    //                   bits[1:0] = frame type (0=progressive, 1=tff interlaced, 2=bff interlaced)
    uint8_t frame_flags = is_4444 ? 0x00u : 0x80u;
    if (is_interlaced) frame_flags |= 0x01u;  // top-field-first for 1080i
    write_u8(p++, frame_flags);
    write_u8(p++, 0);                    // [13] reserved
    write_u8(p++, color->color_primaries);   // [14]
    write_u8(p++, color->transfer_function); // [15]
    write_u8(p++, color->color_matrix);      // [16]
    // [17] alpha_channel_type: 0=none, 1=8-bit, 2=16-bit (ProRes 4444 standard)
    write_u8(p++, (is_4444 && has_alpha) ? 2u : 0u);
    write_u8(p++, 0);                    // [18] reserved
    write_u8(p++, 0x03);                 // [19] matrix_flags: both matrices present
    memcpy(p, quant_luma,   64); p += 64;
    memcpy(p, quant_chroma, 64); p += 64;

    uint16_t hdr_bytes = (uint16_t)(p - dst);
    write_u16(hdr_size_ptr, hdr_bytes);
    return (int)hdr_bytes;
}

// ---------------------------------------------------------------------------
// Build the ProRes PICTURE HEADER (8 bytes, fixed for progressive 422).
//
//   [0]    picture_hdr_size_bits = 0x40 (= 8 bytes Ã— 8 bits)
//   [1..4] picture_data_size â€” big-endian uint32, patched by caller
//   [5..6] slices_per_picture â€” big-endian uint16
//   [7]    log2(mbs_per_slice) << 4  (e.g. 8 MBs/slice â†’ log2=3 â†’ 0x30)
//
// Returns 8.
// ---------------------------------------------------------------------------
static int build_picture_header(
    uint8_t *dst,
    int      num_slices,
    int      mbs_per_slice)
{
    uint8_t log2_mb = 0;
    for (int m = mbs_per_slice; m > 1; m >>= 1) log2_mb++;

    dst[0] = 0x40;                              // pic hdr size = 8 bytes
    write_u32(dst + 1, 0);                      // picture_data_size â€” patched later
    write_u16(dst + 5, (uint16_t)num_slices);   // slices_per_picture
    dst[7] = (uint8_t)(log2_mb << 4);           // log2(mbs_per_slice) in upper nibble
    return 8;
}

// ---------------------------------------------------------------------------
// Interleave luma blocks into per-slice layout (one thread per 8Ã—8 block).
//
// ProRes 422 macroblock = 16Ã—16 pixels â†’ 4 luma 8Ã—8 blocks ordered TL,TR,BL,BR.
// For slice s at MB row y_mb, MB column range [x*mbs..(x+1)*mbs-1]:
//   Slice s's Y region starts at d_out + s * 8*mbs * 64
//   TL block of MB m â†’ d_out[...  + (m*4 + 0) * 64]
//   TR block of MB m â†’ d_out[...  + (m*4 + 1) * 64]
//   BL block of MB m â†’ d_out[...  + (m*4 + 2) * 64]
//   BR block of MB m â†’ d_out[...  + (m*4 + 3) * 64]
// ---------------------------------------------------------------------------
__global__ void k_interleave_luma(
    const int16_t *d_src,           // [height/8][width/8][64]
    int16_t       *d_out,           // [num_slices][8*mbs_per_slice][64]
    int            blk_cols,        // = width / 8
    int            blk_rows,        // = height / 8
    int            mbs_per_slice,
    int            slices_per_row)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= blk_rows * blk_cols) return;

    int blk_row = idx / blk_cols;
    int blk_col = idx % blk_cols;

    // Each ProRes 16Ã—16 MB maps to a 2Ã—2 block grid
    int mb_row = blk_row / 2;
    int mb_col = blk_col / 2;
    int row_in_mb = blk_row & 1;  // 0=top, 1=bottom
    int col_in_mb = blk_col & 1;  // 0=left, 1=right
    int block_in_mb = row_in_mb * 2 + col_in_mb; // TL=0, TR=1, BL=2, BR=3

    int slice_col   = mb_col / mbs_per_slice;
    int mb_in_slice = mb_col % mbs_per_slice;
    int slice_idx   = mb_row * slices_per_row + slice_col;

    // Total blocks per slice: 4Y + 2Cb + 2Cr = 8*mbs
    int dst_block = slice_idx * 8 * mbs_per_slice + mb_in_slice * 4 + block_in_mb;

    const int16_t *src = d_src + (ptrdiff_t)idx * 64;
    int16_t       *dst = d_out + (ptrdiff_t)dst_block * 64;
    for (int i = 0; i < 64; i++) dst[i] = src[i];
}

// ---------------------------------------------------------------------------
// Interleave chroma (Cb or Cr) blocks into per-slice layout.
//
// For 422: each 16Ã—16 MB has 2 chroma blocks (T=top 8 rows, B=bottom 8 rows).
// Chroma block width = luma_width/2 â†’ chroma blk_cols = width/16.
// ---------------------------------------------------------------------------
__global__ void k_interleave_chroma(
    const int16_t *d_src,           // [height/8][width/16][64]
    int16_t       *d_out,           // points to start of Cb (or Cr) region of slice 0
    int            blk_cols,        // = width / 16
    int            blk_rows,        // = height / 8
    int            mbs_per_slice,
    int            slices_per_row)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= blk_rows * blk_cols) return;

    int blk_row = idx / blk_cols;
    int blk_col = idx % blk_cols;  // blk_col = MB column (chroma col = MB col)

    int mb_row      = blk_row / 2;
    int block_in_mb = blk_row & 1;  // 0=top, 1=bottom
    int mb_col      = blk_col;

    int slice_col   = mb_col / mbs_per_slice;
    int mb_in_slice = mb_col % mbs_per_slice;
    int slice_idx   = mb_row * slices_per_row + slice_col;

    // Within the slice: chroma region (after 4*mbs Y blocks), 2 blocks per MB
    int dst_block_in_chroma = mb_in_slice * 2 + block_in_mb;  // 0..2*mbs-1
    int dst_block = slice_idx * 8 * mbs_per_slice + dst_block_in_chroma;

    const int16_t *src = d_src + (ptrdiff_t)idx * 64;
    int16_t       *dst = d_out + (ptrdiff_t)dst_block * 64;
    for (int i = 0; i < 64; i++) dst[i] = src[i];
}

// ---------------------------------------------------------------------------
// Encode one frame from raw V210 device buffer to pinned host frame buffer.
// ---------------------------------------------------------------------------
cudaError_t prores_encode_frame(
    ProResFrameCtx       *ctx,
    const uint32_t       *d_v210,
    uint8_t              *h_out,
    size_t               *out_size,
    cudaStream_t          stream,
    const ProResColorDesc *color)
{
    cudaError_t err;

    // 1. V210 unpack
    err = launch_v210_unpack(d_v210, ctx->d_y, ctx->d_cb, ctx->d_cr,
                             ctx->width, ctx->height, stream);
    if (err != cudaSuccess) return err;

    // 2. DCT + quantise â€” luma
    err = launch_dct_quantise(
        ctx->d_y, ctx->d_coeffs_y,
        ctx->width, ctx->height,
        ctx->q_scale, ctx->profile, false, ctx->is_interlaced, stream);
    if (err != cudaSuccess) return err;

    // 3. DCT + quantise — Cb
    err = launch_dct_quantise(
        ctx->d_cb, ctx->d_coeffs_cb,
        ctx->width / 2, ctx->height,
        ctx->q_scale, ctx->profile, true, ctx->is_interlaced, stream);
    if (err != cudaSuccess) return err;

    // 4. DCT + quantise — Cr
    err = launch_dct_quantise(
        ctx->d_cr, ctx->d_coeffs_cr,
        ctx->width / 2, ctx->height,
        ctx->q_scale, ctx->profile, true, ctx->is_interlaced, stream);
    if (err != cudaSuccess) return err;

    // 5. Interleave block coefficients into per-slice layout
    const int mbs  = ctx->mbs_per_slice;
    const int spr  = ctx->slices_per_row;
    const int T    = 256;
    // Use the number of MB rows that fit in the allocated slice buffer.
    // kHeight=1080 gives 135 DCT rows but only 67 full MB rows (1072 px);
    // the 68th MB row would overflow the slice buffer, so cap at 134 DCT rows.
    const int mb_rows_enc  = ctx->num_slices / spr;
    const int luma_blocks  = mb_rows_enc * 2 * (ctx->width  / 8);
    const int chroma_blocks= mb_rows_enc * 2 * (ctx->width  / 16);

    // Luma: coeffs_slice[0..4*mbs-1 per slice]
    k_interleave_luma<<<(luma_blocks + T - 1) / T, T, 0, stream>>>(
        ctx->d_coeffs_y, ctx->d_coeffs_slice,
        ctx->width / 8, mb_rows_enc * 2, mbs, spr);

    // Cb: coeffs_slice[4*mbs..6*mbs-1 per slice]
    // Each slice's Cb region starts 4*mbs blocks after the slice's Y region.
    // So d_cb_base = d_coeffs_slice + 4*mbs (offset for slice 0's Cb region).
    int16_t *d_cb_base = ctx->d_coeffs_slice + (ptrdiff_t)4 * mbs * 64;
    k_interleave_chroma<<<(chroma_blocks + T - 1) / T, T, 0, stream>>>(
        ctx->d_coeffs_cb, d_cb_base,
        ctx->width / 16, mb_rows_enc * 2, mbs, spr);

    // Cr: coeffs_slice[6*mbs..8*mbs-1 per slice]
    int16_t *d_cr_base = ctx->d_coeffs_slice + (ptrdiff_t)6 * mbs * 64;
    k_interleave_chroma<<<(chroma_blocks + T - 1) / T, T, 0, stream>>>(
        ctx->d_coeffs_cr, d_cr_base,
        ctx->width / 16, mb_rows_enc * 2, mbs, spr);

    // 6. Entropy coding (two-pass: count â†’ prefix-sum â†’ write)
    cuda_prores_enc_frame_raw(
        ctx->d_coeffs_slice,
        ctx->num_slices,
        mbs,
        ctx->q_scale,
        ctx->d_bitstream,
        ctx->d_slice_offsets,
        ctx->d_bit_counts,
        ctx->d_slice_sizes,
        ctx->d_cub_temp,
        ctx->cub_temp_bytes,
        stream);

    // 7. Sync and copy slice offsets to host (needed for seek table)
    uint32_t *h_offsets = nullptr;
    cudaMallocHost(&h_offsets, (ctx->num_slices + 1) * sizeof(uint32_t));
    cudaMemcpyAsync(h_offsets, ctx->d_slice_offsets,
                    (ctx->num_slices + 1) * sizeof(uint32_t),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    const uint32_t total_slice_bytes = h_offsets[ctx->num_slices];

    // 8. Build host-side ProRes frame container
    // Layout: [4 frame_size][4 'icpf'][frame_hdr][pic_hdr][seek_table][slices]

    uint8_t *p = h_out;

    // (a) 4-byte frame size placeholder â€” patched at the end
    uint8_t *frame_size_ptr = p; p += 4;

    // (b) 'icpf' magic
    p[0]='i'; p[1]='c'; p[2]='p'; p[3]='f'; p += 4;

    // (c) Frame header (150 bytes with quant matrices)
    const uint8_t *ql = PRORES_QUANT_LUMA[ctx->profile];
    const uint8_t *qc = PRORES_QUANT_CHROMA[ctx->profile];
    int fhdr_bytes = build_frame_header(p, ctx->width, ctx->height, false, false, ctx->is_interlaced, color, ql, qc);
    p += fhdr_bytes;

    // (d) Picture header (8 bytes)
    uint8_t *pic_hdr_ptr = p;
    int phdr_bytes = build_picture_header(p, ctx->num_slices, mbs);
    p += phdr_bytes;

    // (e) Slice seek table: ProRes stores (slice_bytes / 2) as uint16.
    //     Decoders (FFmpeg, QuickTime) read the entry and shift left 1 to recover
    //     the byte count.  k_bits_to_bytes already rounds each component to an even
    //     byte count, so slice sizes are always even and >> 1 is exact.
    for (int i = 0; i < ctx->num_slices; i++) {
        uint32_t sz = h_offsets[i + 1] - h_offsets[i];
        write_u16(p, (uint16_t)(sz >> 1));
        p += 2;
    }
    cudaFreeHost(h_offsets);

    // (f) Slice data: async copy from device
    cudaMemcpyAsync(p, ctx->d_bitstream, total_slice_bytes,
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    p += total_slice_bytes;

    // (g) Patch picture_data_size: covers from pic_hdr[1] to end of slices
    //     = phdr_bytes(=8) - 1 (skip first byte) + seek_table + slices
    uint32_t pic_data_size = (uint32_t)((p - pic_hdr_ptr));
    write_u32(pic_hdr_ptr + 1, pic_data_size);

    // (h) Patch frame size
    uint32_t frame_bytes = (uint32_t)(p - h_out);
    write_u32(frame_size_ptr, frame_bytes);

    *out_size = (size_t)frame_bytes;
    return cudaSuccess;
}

// ---------------------------------------------------------------------------
// prores_encode_frame_444 — ProRes 4444 / 4444 XQ from a BGRA8 device buffer.
//
// ctx->is_4444 must be true.  ctx->has_alpha controls whether the alpha channel
// in the BGRA input is DCT-encoded into the slice data.  Cb/Cr planes are
// processed at full resolution (4:4:4 — no horizontal subsampling).
//
// Slice format (ProRes 4444, 8-byte header per Apple White Paper):
//   [0x40]         1 byte  header_bits = 64
//   [q_scale]      1 byte
//   [Y_size  BE16] 2 bytes
//   [Cb_size BE16] 2 bytes
//   [A_size  BE16] 2 bytes  (0 when !has_alpha)
//   Data order: Y bytes, Cb bytes, Cr bytes (implicit), A bytes
// ---------------------------------------------------------------------------
cudaError_t prores_encode_frame_444(
    ProResFrameCtx        *ctx,
    const uint8_t         *d_bgra,
    uint8_t               *h_out,
    size_t                *out_size,
    cudaStream_t           stream,
    const ProResColorDesc *color)
{
    cudaError_t err;

    // 1. BGRA8 -> planar YUVA 4:4:4 10-bit  (skips V210 intermediate entirely)
    err = launch_bgra_to_yuva444p10(
        d_bgra,
        ctx->d_y, ctx->d_cb, ctx->d_cr,
        ctx->has_alpha ? ctx->d_alpha : nullptr,
        ctx->width, ctx->height, stream);
    if (err != cudaSuccess) return err;

    // 2. DCT + quantise — luma
    err = launch_dct_quantise(
        ctx->d_y, ctx->d_coeffs_y,
        ctx->width, ctx->height,
        ctx->q_scale, ctx->profile, false, false, stream);
    if (err != cudaSuccess) return err;

    // 3. DCT + quantise — Cb  (4444: FULL width, chroma quant table)
    err = launch_dct_quantise(
        ctx->d_cb, ctx->d_coeffs_cb,
        ctx->width, ctx->height,    // full width — no /2
        ctx->q_scale, ctx->profile, true, false, stream);
    if (err != cudaSuccess) return err;

    // 4. DCT + quantise — Cr  (4444: FULL width, chroma quant table)
    err = launch_dct_quantise(
        ctx->d_cr, ctx->d_coeffs_cr,
        ctx->width, ctx->height,    // full width
        ctx->q_scale, ctx->profile, true, false, stream);
    if (err != cudaSuccess) return err;

    // 5. DCT + quantise — Alpha  (uses luma quant table, per Apple spec)
    if (ctx->has_alpha) {
        err = launch_dct_quantise(
            ctx->d_alpha, ctx->d_coeffs_alpha,
            ctx->width, ctx->height,
            ctx->q_scale, ctx->profile, false, false, stream);
        if (err != cudaSuccess) return err;
    }

    // 6. Interleave block coefficients into per-slice layout.
    //    For 4444, all planes are full resolution, so k_interleave_luma can be
    //    reused for Cb, Cr and Alpha (same 2x2 block-per-16x16-MB layout as Y).
    const int mbs = ctx->mbs_per_slice;
    const int spr = ctx->slices_per_row;
    const int T   = 256;
    const int all_blocks = (ctx->height / 8) * (ctx->width / 8);

    // Y:     offsets [0       .. 4*mbs-1 ] per slice
    k_interleave_luma<<<(all_blocks + T-1)/T, T, 0, stream>>>(
        ctx->d_coeffs_y, ctx->d_coeffs_slice,
        ctx->width / 8, ctx->height / 8, mbs, spr);

    // Cb:    offsets [4*mbs   .. 8*mbs-1 ] per slice
    int16_t *d_cb_base = ctx->d_coeffs_slice + (ptrdiff_t)4 * mbs * 64;
    k_interleave_luma<<<(all_blocks + T-1)/T, T, 0, stream>>>(
        ctx->d_coeffs_cb, d_cb_base,
        ctx->width / 8, ctx->height / 8, mbs, spr);

    // Cr:    offsets [8*mbs   .. 12*mbs-1] per slice
    int16_t *d_cr_base = ctx->d_coeffs_slice + (ptrdiff_t)8 * mbs * 64;
    k_interleave_luma<<<(all_blocks + T-1)/T, T, 0, stream>>>(
        ctx->d_coeffs_cr, d_cr_base,
        ctx->width / 8, ctx->height / 8, mbs, spr);

    // Alpha: offsets [12*mbs  .. 16*mbs-1] per slice  (only when has_alpha)
    if (ctx->has_alpha) {
        int16_t *d_alpha_base = ctx->d_coeffs_slice + (ptrdiff_t)12 * mbs * 64;
        k_interleave_luma<<<(all_blocks + T-1)/T, T, 0, stream>>>(
            ctx->d_coeffs_alpha, d_alpha_base,
            ctx->width / 8, ctx->height / 8, mbs, spr);
    }

    // 7. Two-pass entropy coding (4444)
    cuda_prores_enc_frame_raw_444(
        ctx->d_coeffs_slice,
        ctx->num_slices,
        mbs,
        ctx->q_scale,
        ctx->d_bitstream,
        ctx->d_slice_offsets,
        ctx->d_bit_counts,
        ctx->d_slice_sizes,
        ctx->d_cub_temp,
        ctx->cub_temp_bytes,
        stream,
        ctx->has_alpha);

    // 8. Sync and transfer slice offsets to host for seek table
    uint32_t *h_offsets = nullptr;
    cudaMallocHost(&h_offsets, (ctx->num_slices + 1) * sizeof(uint32_t));
    cudaMemcpyAsync(h_offsets, ctx->d_slice_offsets,
                    (ctx->num_slices + 1) * sizeof(uint32_t),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    const uint32_t total_slice_bytes = h_offsets[ctx->num_slices];

    // 9. Assemble ProRes frame container  [frame_size][icpf][hdr][pic_hdr][seek][slices]
    uint8_t *p = h_out;
    uint8_t *frame_size_ptr = p; p += 4;
    p[0]='i'; p[1]='c'; p[2]='p'; p[3]='f'; p += 4;

    const uint8_t *ql = PRORES_QUANT_LUMA[ctx->profile];
    const uint8_t *qc = PRORES_QUANT_CHROMA[ctx->profile];
    int fhdr_bytes = build_frame_header(p, ctx->width, ctx->height,
                                         true, ctx->has_alpha, false, color, ql, qc);
    p += fhdr_bytes;

    uint8_t *pic_hdr_ptr = p;
    int phdr_bytes = build_picture_header(p, ctx->num_slices, mbs);
    p += phdr_bytes;

    // ProRes seek table stores size/2; decoder reads entry << 1.
    for (int i = 0; i < ctx->num_slices; i++) {
        uint32_t sz = h_offsets[i + 1] - h_offsets[i];
        write_u16(p, (uint16_t)(sz >> 1)); p += 2;
    }
    cudaFreeHost(h_offsets);

    cudaMemcpyAsync(p, ctx->d_bitstream, total_slice_bytes,
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    p += total_slice_bytes;

    uint32_t pic_data_size = (uint32_t)(p - pic_hdr_ptr);
    write_u32(pic_hdr_ptr + 1, pic_data_size);

    uint32_t frame_bytes = (uint32_t)(p - h_out);
    write_u32(frame_size_ptr, frame_bytes);

    *out_size = (size_t)frame_bytes;
    return cudaSuccess;
}
