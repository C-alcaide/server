// cuda_prores_frame.cu
// Frame-level orchestration: V210 unpack → DCT+quant → entropy → header build
// ---------------------------------------------------------------------------
#include "cuda_prores_frame.h"
#include "cuda_prores_v210_unpack.cuh"
#include "cuda_prores_dct_quant.cuh"
#include "cuda_prores_entropy.cu"    // includes kernel definitions

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
// Build ProRes picture header into dst, returns bytes written.
// ---------------------------------------------------------------------------
static int build_picture_header(
    uint8_t            *dst,
    int                 width,
    int                 height,
    ProResProfile       profile,
    const ProResColorDesc *color,
    const uint8_t      *quant_luma,
    const uint8_t      *quant_chroma)
{
    uint8_t *p = dst;

    // [0..1] header size (filled at end)
    uint8_t *hdr_size_ptr = p; p += 2;
    // [2..3] version = 0
    write_u16(p, 0); p += 2;
    // [4..7] encoder tag 'CUDA'
    p[0]='C'; p[1]='U'; p[2]='D'; p[3]='A'; p += 4;
    // [8..9] width
    write_u16(p, (uint16_t)width);  p += 2;
    // [10..11] height
    write_u16(p, (uint16_t)height); p += 2;
    // [12] chroma format: 2 = 4:2:2
    write_u8(p++, 0x82); // bits 7:4 = chroma fmt (2), bits 3:0 = 0
    // [13] frame type: 0 = progressive
    write_u8(p++, 0);
    // [14] aspect ratio: 0 = square
    write_u8(p++, 0);
    // [15] color primaries
    write_u8(p++, color->color_primaries);
    // [16] transfer function
    write_u8(p++, color->transfer_function);
    // [17] color matrix
    write_u8(p++, color->color_matrix);
    // [18] source format (5 = 4K)
    uint8_t src_fmt = (width >= 3840) ? 5 : (width >= 1920 ? 3 : 1);
    write_u8(p++, src_fmt);
    // [19] alpha channel depth: 0 = none
    write_u8(p++, 0);
    // [20] reserved
    write_u8(p++, 0);
    // [21] luma quant matrix flag
    write_u8(p++, 1);
    memcpy(p, quant_luma, 64); p += 64;
    // [86] chroma quant matrix flag
    write_u8(p++, 1);
    memcpy(p, quant_chroma, 64); p += 64;

    uint16_t hdr_bytes = (uint16_t)(p - dst);
    write_u16(hdr_size_ptr, hdr_bytes);
    return (int)hdr_bytes;
}

// ---------------------------------------------------------------------------
// Interleave block coefficients into per-slice layout
//
// ProRes 422 macroblock layout per slice (1 slice = slices_per_row columns):
//   Each macroblock column has: 2 Y blocks (left/right 8px), 1 Cb, 1 Cr
//   For slices_per_row=4 and width=3840: each slice spans 3840/4=960px = 120 MB columns
//   blocks_per_slice = 120 * 4 = 480
//
// The interleave kernel reorders from plane-layout into slice-layout so that
// the entropy kernel can process each slice independently.
// ---------------------------------------------------------------------------
__global__ void k_interleave_slices(
    const int16_t *d_y,
    const int16_t *d_cb,
    const int16_t *d_cr,
    int16_t       *d_out,    // [num_slices][blocks_per_slice][64]
    int width, int height,
    int slices_per_row)
{
    // Each thread handles one macroblock column (4 blocks: 2Y + Cb + Cr)
    int mb_cols   = width / 8;   // total macroblock columns in frame
    int mb_rows   = height / 8;
    int mb_total  = mb_cols * mb_rows;

    int mb_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (mb_idx >= mb_total) return;

    int mb_row = mb_idx / mb_cols;
    int mb_col = mb_idx % mb_cols;

    int slice_col_width = mb_cols / slices_per_row;
    int slice_idx       = mb_row * slices_per_row + mb_col / slice_col_width;
    int local_mb        = mb_col % slice_col_width;
    int blocks_per_slice = slice_col_width * 4; // 2Y + Cb + Cr

    // Destination: block positions within slice
    // order per spec: Y0, Y1, Cb0, Cr0 for each MB column
    int base_out = slice_idx * blocks_per_slice * 64 + local_mb * 4 * 64;

    const int16_t *src_y0 = d_y  + (mb_row * (width / 8) + mb_col * 2    ) * 64; // left 8px
    const int16_t *src_y1 = d_y  + (mb_row * (width / 8) + mb_col * 2 + 1) * 64; // right 8px
    const int16_t *src_cb = d_cb + (mb_row * (width / 16) + mb_col       ) * 64;
    const int16_t *src_cr = d_cr + (mb_row * (width / 16) + mb_col       ) * 64;

    for (int i = 0; i < 64; i++) {
        d_out[base_out +            i] = src_y0[i];
        d_out[base_out +    64 + i] = src_y1[i];
        d_out[base_out + 2 * 64 + i] = src_cb[i];
        d_out[base_out + 3 * 64 + i] = src_cr[i];
    }
}

// ---------------------------------------------------------------------------
// Encode one frame from raw V210 device buffer to pinned host frame buffer.
//
// d_v210          device pointer to V210 frame (from DeckLink or test data)
// h_out           pinned host output (caller provides, sized to h_frame_buf_size)
// out_size        [out] actual bytes written to h_out including all headers
// stream          CUDA stream
// color           HDR/SDR metadata
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

    // 2. DCT + quantise — luma plane
    err = launch_dct_quantise(
        ctx->d_y, ctx->d_coeffs_y,
        c_quant_luma + ctx->profile * 64,
        ctx->width, ctx->height,
        ctx->q_scale, ctx->profile, false, stream);
    if (err != cudaSuccess) return err;

    // 3. DCT + quantise — Cb
    err = launch_dct_quantise(
        ctx->d_cb, ctx->d_coeffs_cb,
        c_quant_chroma + ctx->profile * 64,
        ctx->width / 2, ctx->height,
        ctx->q_scale, ctx->profile, true, stream);
    if (err != cudaSuccess) return err;

    // 4. DCT + quantise — Cr
    err = launch_dct_quantise(
        ctx->d_cr, ctx->d_coeffs_cr,
        c_quant_chroma + ctx->profile * 64,
        ctx->width / 2, ctx->height,
        ctx->q_scale, ctx->profile, true, stream);
    if (err != cudaSuccess) return err;

    // 5. Interleave coefficients into slice layout
    {
        int mb_total = (ctx->width / 8) * (ctx->height / 8);
        int T = 256;
        k_interleave_slices<<<(mb_total + T - 1) / T, T, 0, stream>>>(
            ctx->d_coeffs_y, ctx->d_coeffs_cb, ctx->d_coeffs_cr,
            ctx->d_coeffs_slice,
            ctx->width, ctx->height, ctx->slices_per_row);
    }

    // 6. Entropy coding (two-pass: count → prefix-sum → write)
    cuda_prores_enc_frame_raw(
        ctx->d_coeffs_slice, ctx->num_slices, ctx->blocks_per_slice,
        ctx->d_bitstream, ctx->d_slice_offsets, ctx->d_bit_counts,
        ctx->d_cub_temp, ctx->cub_temp_bytes, stream);

    // 7. Sync to get slice offsets to host for header construction
    // (only the offsets array — bitstream stays on GPU until cudaMemcpyAsync)
    uint32_t *h_offsets = nullptr;
    cudaMallocHost(&h_offsets, (ctx->num_slices + 1) * sizeof(uint32_t));
    cudaMemcpyAsync(h_offsets, ctx->d_slice_offsets,
                    (ctx->num_slices + 1) * sizeof(uint32_t),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    uint32_t total_slice_bytes = h_offsets[ctx->num_slices];

    // 8. Async copy encoded bitstream to pinned host buffer after the headers
    // Header layout in h_out:
    //   [0..3]   frame size (patched at end)
    //   [4..7]   'icpf'
    //   [8..]    picture header
    //   [pic_hdr_end..slice_table_end]  slice offset table (uint16_t per slice-1)
    //   [slice_data_start..]            entropy-coded slice data
    uint8_t *p = h_out;

    // Frame size placeholder
    uint8_t *frame_size_ptr = p; p += 4;
    // 'icpf' magic
    p[0]='i'; p[1]='c'; p[2]='p'; p[3]='f'; p += 4;

    // Picture header
    const uint8_t *ql = PRORES_QUANT_LUMA[ctx->profile];
    const uint8_t *qc = PRORES_QUANT_CHROMA[ctx->profile];
    int hdr_bytes = build_picture_header(p, ctx->width, ctx->height,
                                         (ProResProfile)ctx->profile,
                                         color, ql, qc);
    p += hdr_bytes;

    // Slice offset table: offsets[1..num_slices-1] relative to slice data start
    // (first slice is always at offset 0, not stored)
    for (int i = 1; i < ctx->num_slices; i++) {
        // ProRes stores offsets as uint32_t (big-endian) per slice
        write_u32(p, h_offsets[i]);
        p += 4;
    }
    cudaFreeHost(h_offsets);

    // Entropy-coded slice data: copy from device
    size_t header_total = (size_t)(p - h_out);
    cudaMemcpyAsync(p, ctx->d_bitstream, total_slice_bytes,
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    p += total_slice_bytes;

    // Patch frame size
    uint32_t frame_bytes = (uint32_t)(p - h_out);
    write_u32(frame_size_ptr, frame_bytes);

    *out_size = frame_bytes;
    return cudaSuccess;
}

// Internal: called from prores_encode_frame to run the two-pass entropy.
void cuda_prores_enc_frame_raw(
    const int16_t *d_coeffs,
    int num_slices, int blocks_per_slice,
    uint8_t *d_output,
    uint32_t *d_slice_offsets,
    uint32_t *d_bit_counts,
    void *d_cub_temp, size_t cub_temp_bytes,
    cudaStream_t stream)
{
    const int T = 128;
    const int G = (num_slices + T - 1) / T;

    k_count_bits<<<G, T, 0, stream>>>(
        d_coeffs, d_bit_counts, blocks_per_slice, num_slices);
    k_bits_to_bytes<<<G, T, 0, stream>>>(d_bit_counts, num_slices);
    cub::DeviceScan::ExclusiveSum(
        d_cub_temp, cub_temp_bytes,
        d_bit_counts, d_slice_offsets, num_slices, stream);
    k_encode_slices<<<G, T, 0, stream>>>(
        d_coeffs, d_slice_offsets, d_output, blocks_per_slice, num_slices);
}
