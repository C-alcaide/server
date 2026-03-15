// notchlc_decode.cu
// NotchLC GPU decode orchestration.
//
// High-level pipeline per frame:
//   1. Upload compressed payload (h→d async via pinned staging)
//   2. nvcomp LZ4 decompress (or direct copy if Uncompressed format)
//   3. Sync stream, read block header (first 256 bytes from d_uncompressed → host)
//   4. Parse 10 LE u32 block-header fields to derive all data section offsets
//   5. Launch k_notch_y_decode   (1 thread per 4×4 Y block)
//   6. Launch k_notch_uv_decode  (1 thread per 16×16 UV block)
//   7. Launch k_notch_a_decode or fill_opaque (1 thread per 16×16 block)
//   8. Launch k_notch_ycocg_to_bgra16 (1 thread per pixel)
//   9. cudaMemcpy2DToArrayAsync → GL texture (or memcpy → host for to_host variant)
//  10. Synchronise
// ---------------------------------------------------------------------------

#include "notchlc_decode.h"
#include "notchlc_y_decode.cuh"
#include "notchlc_uv_decode.cuh"
#include "notchlc_a_decode.cuh"
#include "notchlc_ycocg_to_bgra16.cuh"

#if __has_include(<common/log.h>)
#  include <common/log.h>
#  define NOTCH_LOG_ERROR(msg)   CASPAR_LOG(error)   << L"" msg
#  define NOTCH_LOG_WARNING(msg) CASPAR_LOG(warning) << L"" msg
#else
#  include <cstdio>
#  define NOTCH_LOG_ERROR(msg)   fprintf(stderr, "[cuda_notchlc] ERROR: %s\n",   msg)
#  define NOTCH_LOG_WARNING(msg) fprintf(stderr, "[cuda_notchlc] WARNING: %s\n", msg)
#endif

#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>

using caspar::cuda_notchlc::NotchLCFormat;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call)                                              \
    do {                                                              \
        cudaError_t _e = (call);                                      \
        if (_e != cudaSuccess) return _e;                             \
    } while (0)

#define NVCOMP_CHECK(call)                                            \
    do {                                                              \
        nvcompStatus_t _s = (call);                                   \
        if (_s != nvcompSuccess) return cudaErrorUnknown;             \
    } while (0)

static void safe_cuda_free(void*& p)
{
    if (p) { cudaFree(p); p = nullptr; }
}
static void safe_cuda_free_host(void*& p)
{
    if (p) { cudaFreeHost(p); p = nullptr; }
}

// Read a LE u32 from a byte array at given byte offset.
static inline uint32_t read_le32h(const uint8_t* b, int off)
{
    return (uint32_t)b[off]
         | ((uint32_t)b[off+1] << 8)
         | ((uint32_t)b[off+2] << 16)
         | ((uint32_t)b[off+3] << 24);
}

// ---------------------------------------------------------------------------
// Block header layout (notchlc.c :: decode_blocks)
//
// Byte 0  of decompressed data (start of header, 10 × LE u32 = 40 bytes):
//   [00] texture_size_x
//   [04] texture_size_y
//   [08] uv_offset_data_offset_raw  → *4 = absolute byte offset of UV offset table
//   [12] y_control_data_offset_raw  → *4 = absolute byte offset of Y control words
//   [16] a_control_word_offset_raw  → *4 = absolute byte offset of alpha control words
//   [20] uv_data_offset_raw         → *4 = absolute byte offset of UV data blob
//   [24] y_data_size_raw            (raw bytes; NOT *4)
//   [28] a_data_offset_raw          → *4 = offset added to uv_data to find alpha data
//   [32] a_count_size_raw           → *4 (informational; not used in decode)
//   [36] data_end_raw               (raw, = abs byte offset of end of all data)
//
// Byte 40 onwards: y_data_row_offsets (one LE u32 per group of 4 rows)
//
// Derived:
//   y_data_offset   = data_end - y_data_size          (abs byte offset of Y bitstream)
//   uv_count_offset = y_data_offset - a_data_offset   (relative; also used as "no alpha" test)
//   No alpha if:    uv_count_offset == a_control_word_offset
// ---------------------------------------------------------------------------

// NotchBlockHeader is now defined in notchlc_decode.h

static cudaError_t parse_block_header(
    const uint8_t* h_header_buf,   // must be at least 256 bytes
    uint32_t        uncompressed_size,
    NotchBlockHeader& out)
{
    out.width                  = read_le32h(h_header_buf,  0);
    out.height                 = read_le32h(h_header_buf,  4);
    out.uv_offset_data_offset  = read_le32h(h_header_buf,  8) * 4u;
    out.y_control_data_offset  = read_le32h(h_header_buf, 12) * 4u;
    out.a_control_word_offset  = read_le32h(h_header_buf, 16) * 4u;
    out.uv_data_offset         = read_le32h(h_header_buf, 20) * 4u;
    out.y_data_size            = read_le32h(h_header_buf, 24);   // NOT *4
    out.a_data_offset          = read_le32h(h_header_buf, 28) * 4u;
    out.a_count_size           = read_le32h(h_header_buf, 32) * 4u;
    out.data_end               = read_le32h(h_header_buf, 36);   // NOT *4
    out.y_data_row_offsets     = 40u;

    if (out.data_end > uncompressed_size)
        return cudaErrorInvalidValue;
    if (out.data_end <= out.y_data_size)
        return cudaErrorInvalidValue;

    out.y_data_offset    = out.data_end - out.y_data_size;
    out.uv_count_offset  = out.y_data_offset - out.a_data_offset;
    out.has_alpha        = (out.uv_count_offset != out.a_control_word_offset);

    return cudaSuccess;
}

// ---------------------------------------------------------------------------
// Kernel launch helpers (simple wrappers to keep orchestration readable)
// ---------------------------------------------------------------------------

// ── Two-pass Y decode: compute per-block bit widths then per-row prefix scan ──
// Pass 1 (k_y_compute_widths):  1 thread per block, O(1) per thread.
// Pass 2 (k_y_prefix_rows):     1 thread per row-group, O(blocks_x) per row,
//   but fully parallel across blocks_y rows.  No external library needed.
// Pass 3 (k_notch_y_decode):    uses precomputed bit offsets, O(1) per thread.
static cudaError_t launch_y_precompute_offsets(
    const NotchBlockHeader& hdr,
    NotchLCDecodeCtx* ctx,
    cudaStream_t s)
{
    const int blocks_x = hdr.width  / 4;
    const int blocks_y = hdr.height / 4;
    const int total    = blocks_x * blocks_y;
    const int tpb256 = 256;

    // Pass 1: compute bit-widths for every block in parallel.
    caspar::cuda_notchlc::k_y_compute_widths<<<(total + tpb256 - 1) / tpb256, tpb256, 0, s>>>(
        ctx->d_uncompressed, hdr.y_control_data_offset, total, ctx->d_y_bit_widths);
    CUDA_CHECK(cudaGetLastError());

    // Pass 2: per-row exclusive prefix scan (1 thread per row-group).
    const int tpb64 = 64;
    caspar::cuda_notchlc::k_y_prefix_rows<<<(blocks_y + tpb64 - 1) / tpb64, tpb64, 0, s>>>(
        ctx->d_y_bit_widths, ctx->d_y_bit_offsets, blocks_x, blocks_y);
    CUDA_CHECK(cudaGetLastError());

    return cudaSuccess;
}

static cudaError_t launch_y_decode(
    const NotchBlockHeader& hdr,
    uint8_t*        d_uncompressed,
    uint16_t*       d_y,
    const uint32_t* d_bit_offsets,
    cudaStream_t    s)
{
    const int blocks_x = hdr.width  / 4;
    const int blocks_y = hdr.height / 4;
    const int total = blocks_x * blocks_y;
    const int tpb = 128;
    const int grid = (total + tpb - 1) / tpb;

    caspar::cuda_notchlc::k_notch_y_decode<<<grid, tpb, 0, s>>>(
        d_uncompressed,
        hdr.y_control_data_offset,
        hdr.y_data_row_offsets,
        hdr.y_data_offset,
        (int)hdr.width, (int)hdr.height,
        d_bit_offsets,
        d_y);
    return cudaGetLastError();
}

static cudaError_t launch_uv_decode(
    const NotchBlockHeader& hdr,
    uint8_t*   d_uncompressed,
    uint16_t*  d_u,
    uint16_t*  d_v,
    cudaStream_t s)
{
    const int blocks_x = (hdr.width  + 15) / 16;
    const int blocks_y = (hdr.height + 15) / 16;
    const int total = blocks_x * blocks_y;
    const int tpb = 64;
    const int grid = (total + tpb - 1) / tpb;

    caspar::cuda_notchlc::k_notch_uv_decode<<<grid, tpb, 0, s>>>(
        d_uncompressed,
        hdr.uv_offset_data_offset,
        hdr.uv_data_offset,
        (int)hdr.width, (int)hdr.height,
        d_u, d_v);
    return cudaGetLastError();
}

static cudaError_t launch_a_decode_or_fill(
    const NotchBlockHeader& hdr,
    uint8_t*   d_uncompressed,
    uint16_t*  d_a,
    int        width,
    int        height,
    cudaStream_t s)
{
    const int n_pixels = width * height;

    if (!hdr.has_alpha) {
        // Fill all pixels with 4095 (fully opaque).
        const int tpb = 256;
        const int grid = (n_pixels + tpb - 1) / tpb;
        caspar::cuda_notchlc::k_notch_a_fill_opaque<<<grid, tpb, 0, s>>>(d_a, n_pixels);
    } else {
        const int blocks_x = (width  + 15) / 16;
        const int blocks_y = (height + 15) / 16;
        const int total = blocks_x * blocks_y;
        const int tpb = 64;
        const int grid = (total + tpb - 1) / tpb;

        caspar::cuda_notchlc::k_notch_a_decode<<<grid, tpb, 0, s>>>(
            d_uncompressed,
            hdr.a_control_word_offset,
            hdr.uv_data_offset,
            hdr.a_data_offset,          // already *4 from parse step
            width, height,
            d_a);
    }
    return cudaGetLastError();
}

static cudaError_t launch_color_convert(
    uint16_t*  d_y, uint16_t* d_u, uint16_t* d_v, uint16_t* d_a,
    uint16_t*  d_bgra16,
    int width, int height, int color_matrix,
    cudaStream_t s)
{
    const int n_pixels = width * height;
    const int tpb = 256;
    const int grid = (n_pixels + tpb - 1) / tpb;

    caspar::cuda_notchlc::k_notch_ycocg_to_bgra16<<<grid, tpb, 0, s>>>(
        d_y, d_u, d_v, d_a,
        width, height, color_matrix,
        d_bgra16);
    return cudaGetLastError();
}

// ---------------------------------------------------------------------------
// notchlc_decode_ctx_create
// ---------------------------------------------------------------------------

cudaError_t notchlc_decode_ctx_create(NotchLCDecodeCtx* ctx,
                                       int    width,
                                       int    height,
                                       size_t max_compressed_bytes,
                                       size_t max_uncompressed_bytes)
{
    memset(ctx, 0, sizeof(*ctx));
    ctx->width                = width;
    ctx->height               = height;
    ctx->max_compressed_bytes = max_compressed_bytes;

    const size_t n_pix = (size_t)width * height;

    // ── Pinned staging — compressed input ─────────────────────────────────
    CUDA_CHECK(cudaMallocHost((void**)&ctx->h_compressed, max_compressed_bytes));

    // ── Pinned staging — CPU-decompressed uncompressed payload ────────────
    // The LZ4 path decompresses on CPU into this buffer, then the result is
    // uploaded to device via async DMA.  Eliminates the ~1700 ms / frame
    // bottleneck caused by nvcomp BatchedLZ4 on a single monolithic chunk.
    ctx->max_uncompressed_bytes = max_uncompressed_bytes;
    CUDA_CHECK(cudaMallocHost((void**)&ctx->h_uncompressed, max_uncompressed_bytes));

    // ── Heap staging for LZ4 output (cache-friendly for LZ4 back-references) ──
    ctx->h_task_buf = (uint8_t*)std::malloc(max_uncompressed_bytes);
    if (!ctx->h_task_buf) return cudaErrorMemoryAllocation;

    // ── Device: uncompressed buffer ───────────────────────────────────────
    // Add 16-byte padding so the Y-decode bit-reader can over-read safely.
    ctx->d_uncompressed_alloc  = max_uncompressed_bytes + 16;
    CUDA_CHECK(cudaMalloc(&ctx->d_uncompressed, ctx->d_uncompressed_alloc));
    CUDA_CHECK(cudaMemset(ctx->d_uncompressed, 0, ctx->d_uncompressed_alloc));

    // NOTE: d_compressed and the nvcomp batch API arrays are not allocated.
    // GPU-side LZ4 decompression is no longer used (replaced by CPU path).

    // ── Decoded planes ─────────────────────────────────────────────────────
    CUDA_CHECK(cudaMalloc((void**)&ctx->d_y,      n_pix * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc((void**)&ctx->d_u,      n_pix * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc((void**)&ctx->d_v,      n_pix * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc((void**)&ctx->d_a,      n_pix * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc((void**)&ctx->d_bgra16, n_pix * 4 * sizeof(uint16_t)));

    // ── Y-decode prefix-sum scratch ────────────────────────────────────────
    {
        const size_t n_blocks = ((size_t)width / 4) * ((size_t)height / 4);
        CUDA_CHECK(cudaMalloc((void**)&ctx->d_y_bit_widths,  n_blocks * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc((void**)&ctx->d_y_bit_offsets, n_blocks * sizeof(uint32_t)));
    }

    // ── Stream ────────────────────────────────────────────────────────────
    CUDA_CHECK(cudaStreamCreate(&ctx->stream));

    // ── Upload gamma LUT (idempotent — same data every time) ──────────────
    CUDA_CHECK(caspar::cuda_notchlc::notchlc_upload_gamma_lut());

    return cudaSuccess;
}

// ---------------------------------------------------------------------------
// notchlc_decode_ctx_destroy
// ---------------------------------------------------------------------------

void notchlc_decode_ctx_destroy(NotchLCDecodeCtx* ctx)
{
    if (!ctx) return;
    if (ctx->stream) { cudaStreamDestroy(ctx->stream); ctx->stream = nullptr; }

    safe_cuda_free_host((void*&)ctx->h_compressed);
    safe_cuda_free_host((void*&)ctx->h_uncompressed);
    std::free(ctx->h_task_buf); ctx->h_task_buf = nullptr;
    safe_cuda_free((void*&)ctx->d_compressed);
    safe_cuda_free((void*&)ctx->d_uncompressed);
    safe_cuda_free((void*&)ctx->d_nvcomp_temp);
    safe_cuda_free((void*&)ctx->d_compressed_ptrs);
    safe_cuda_free((void*&)ctx->d_compressed_bytes_arr);
    safe_cuda_free((void*&)ctx->d_uncompressed_buffer_bytes);
    safe_cuda_free((void*&)ctx->d_uncompressed_actual_bytes);
    safe_cuda_free((void*&)ctx->d_uncompressed_ptrs);
    safe_cuda_free((void*&)ctx->d_nvcomp_statuses);
    safe_cuda_free((void*&)ctx->d_y);
    safe_cuda_free((void*&)ctx->d_u);
    safe_cuda_free((void*&)ctx->d_v);
    safe_cuda_free((void*&)ctx->d_a);
    safe_cuda_free((void*&)ctx->d_bgra16);
    safe_cuda_free((void*&)ctx->d_y_bit_widths);
    safe_cuda_free((void*&)ctx->d_y_bit_offsets);
}

// ---------------------------------------------------------------------------
// CPU LZ4 block decompressor — NotchLC variant.
//
// NotchLC uses the standard LZ4 block format with a 64 KB history window.
// End of stream is signalled by a match block with delta == 0.
//
// This is a direct-output implementation: literals and match copies go
// straight into dst, using (d - delta) as the match source pointer exactly
// as standard LZ4 does.  The original ring-buffer variant from FFmpeg
// notchlc.c wrote every byte twice (ring → flush to WC pinned memory) and
// was hitting slow WC-read stalls on back-references; this version keeps all
// hot data in the CPU cache (dst is a regular malloc'd buffer) and achieves
// 5-8× higher throughput.
//
// Returns the actual decompressed byte count, or 0 on error.
// ---------------------------------------------------------------------------
static size_t lz4_decompress_cpu(
    const uint8_t* src, size_t src_size,
    uint8_t*       dst, size_t dst_max)
{
    const uint8_t* s   = src;
    const uint8_t* end = src + src_size;
    uint8_t*       d   = dst;
    uint8_t* const de  = dst + dst_max;

    while (s < end) {
        const uint8_t token = *s++;

        // ── Literals ─────────────────────────────────────────────────────
        unsigned int num_lit = (unsigned int)(token >> 4);
        if (num_lit == 15) {
            uint8_t c;
            do { if (s >= end) return 0; c = *s++; num_lit += (unsigned int)c; } while (c == 255);
        }
        if (s + num_lit > end || d + num_lit > de) return 0;
        std::memcpy(d, s, num_lit);
        d += num_lit;
        s += num_lit;

        if (s >= end) break;  // last sequence has literals only, no match

        // ── Match offset ──────────────────────────────────────────────────
        if (s + 2 > end) return 0;
        const uint16_t delta = (uint16_t)s[0] | ((uint16_t)s[1] << 8); s += 2;
        if (delta == 0) break;  // NotchLC end-of-stream marker

        // ── Match length ──────────────────────────────────────────────────
        unsigned int match_len = 4u + (unsigned int)(token & 0x0Fu);
        if ((token & 0x0Fu) == 0x0Fu) {
            uint8_t c;
            do { if (s >= end) return 0; c = *s++; match_len += (unsigned int)c; } while (c == 255);
        }

        if ((size_t)delta > (size_t)(d - dst) || d + match_len > de) return 0;

        // Standard LZ4 back-reference: copy match_len bytes from (d - delta).
        // When match_len > delta the copy intentionally overlaps (run-length
        // encoding pattern); byte-by-byte via index preserves this correctly.
        const uint8_t* match = d - delta;
        if (match_len <= delta) {
            // Non-overlapping: single memcpy is safe and fast.
            std::memcpy(d, match, match_len);
        } else {
            // Overlapping: copy byte-by-byte using a fixed source pointer so
            // already-written bytes are re-read as the pattern propagates.
            for (unsigned int i = 0; i < match_len; i++)
                d[i] = match[i];
        }
        d += match_len;
    }

    return (size_t)(d - dst);
}


// ---------------------------------------------------------------------------
// notchlc_decode_cpu_phase — LZ4 decompress + header parse (CPU, any thread)
// ---------------------------------------------------------------------------
cudaError_t notchlc_decode_cpu_phase(
    NotchLCDecodeCtx*              ctx,
    const uint8_t*                 h_compressed,
    size_t                         compressed_size,
    uint32_t                       uncompressed_size,
    caspar::cuda_notchlc::NotchLCFormat format,
    NotchBlockHeader&              hdr_out,
    size_t&                        actual_uncompressed_out)
{
    if (format == caspar::cuda_notchlc::NotchLCFormat::LZ4) {
        actual_uncompressed_out = lz4_decompress_cpu(
            h_compressed, compressed_size,
            ctx->h_task_buf, ctx->max_uncompressed_bytes);
        if (actual_uncompressed_out == 0) {
            NOTCH_LOG_ERROR("[notchlc_decode] CPU LZ4 decompression failed");
            return cudaErrorInvalidValue;
        }
        if (parse_block_header(ctx->h_task_buf, (uint32_t)actual_uncompressed_out, hdr_out) != cudaSuccess) {
            NOTCH_LOG_ERROR("[notchlc_decode] parse_block_header failed (LZ4)");
            return cudaErrorInvalidValue;
        }
        std::memcpy(ctx->h_uncompressed, ctx->h_task_buf, actual_uncompressed_out);
    } else if (format == caspar::cuda_notchlc::NotchLCFormat::Uncompressed) {
        if (compressed_size > ctx->max_compressed_bytes) {
            NOTCH_LOG_ERROR("[notchlc_decode] compressed_size exceeds max");
            return cudaErrorInvalidValue;
        }
        std::memcpy(ctx->h_compressed, h_compressed, compressed_size);
        actual_uncompressed_out = (size_t)uncompressed_size;
        uint8_t h_hdr_buf[256] = {};
        const size_t hdr_read = (actual_uncompressed_out >= 256u) ? 256u : actual_uncompressed_out;
        std::memcpy(h_hdr_buf, ctx->h_compressed, hdr_read);
        if (parse_block_header(h_hdr_buf, (uint32_t)actual_uncompressed_out, hdr_out) != cudaSuccess) {
            NOTCH_LOG_ERROR("[notchlc_decode] parse_block_header failed (Uncompressed)");
            return cudaErrorInvalidValue;
        }
        std::memcpy(ctx->h_uncompressed, ctx->h_compressed, actual_uncompressed_out);
    } else {
        NOTCH_LOG_ERROR("[notchlc_decode] LZF format not supported");
        return cudaErrorNotSupported;
    }
    return cudaSuccess;
}

// ---------------------------------------------------------------------------
// notchlc_decode_gpu_phase — H→D upload + GPU kernels (CUDA/GL thread)
// ---------------------------------------------------------------------------
cudaError_t notchlc_decode_gpu_phase(
    NotchLCDecodeCtx*        ctx,
    const NotchBlockHeader&  hdr,
    size_t                   actual_uncompressed,
    int                      color_matrix,
    cudaArray_t              d_gl_array)
{
    cudaStream_t s = ctx->stream;

    if ((int)hdr.width != ctx->width || (int)hdr.height != ctx->height)
        NOTCH_LOG_WARNING("[notchlc_decode] block header dimensions mismatch context");

    CUDA_CHECK(cudaMemcpyAsync(ctx->d_uncompressed, ctx->h_uncompressed,
                               actual_uncompressed, cudaMemcpyHostToDevice, s));
    CUDA_CHECK(launch_y_precompute_offsets(hdr, ctx, s));
    CUDA_CHECK(launch_y_decode(hdr, ctx->d_uncompressed, ctx->d_y, ctx->d_y_bit_offsets, s));
    CUDA_CHECK(launch_uv_decode(hdr, ctx->d_uncompressed, ctx->d_u, ctx->d_v, s));
    CUDA_CHECK(launch_a_decode_or_fill(hdr, ctx->d_uncompressed, ctx->d_a,
                                       ctx->width, ctx->height, s));
    CUDA_CHECK(launch_color_convert(ctx->d_y, ctx->d_u, ctx->d_v, ctx->d_a,
                                    ctx->d_bgra16, ctx->width, ctx->height, color_matrix, s));

    if (d_gl_array) {
        const size_t row_bytes = (size_t)ctx->width * 4 * sizeof(uint16_t);
        CUDA_CHECK(cudaMemcpy2DToArrayAsync(
            d_gl_array, 0, 0,
            ctx->d_bgra16, row_bytes,
            row_bytes, (size_t)ctx->height,
            cudaMemcpyDeviceToDevice, s));
    }

    cudaStreamSynchronize(s);
    return cudaSuccess;
}

// ---------------------------------------------------------------------------
// Internal: decompress + decode (shared by both decode_frame variants).
// Does NOT include the final D→D or D→H copy of d_bgra16.
// On exit the CUDA stream has NOT been synchronised — callers sync after
// submitting the final copy (Memcpy2DToArray or MemcpyAsync).
// ---------------------------------------------------------------------------
static cudaError_t decode_internal(
    NotchLCDecodeCtx*  ctx,
    const uint8_t*     h_compressed,
    size_t             compressed_size,
    uint32_t           uncompressed_size,
    NotchLCFormat      format,
    int                color_matrix)
{
    cudaStream_t s = ctx->stream;

    // 2. CPU decompress + parse header.
    NotchBlockHeader hdr;
    size_t actual_uncompressed = 0;
    const uint8_t* h_upload_src = nullptr;

    auto t0 = std::chrono::high_resolution_clock::now();

    if (format == NotchLCFormat::LZ4) {
        // Decompress from the caller's buffer (heap, cache-friendly reads) directly
        // into ctx->h_task_buf (regular malloc, cache-friendly writes + back-refs).
        // Skips the 69 MB WC-pinned copy that triggers slow non-temporal reads/writes.
        actual_uncompressed = lz4_decompress_cpu(
            h_compressed, compressed_size,
            ctx->h_task_buf, ctx->max_uncompressed_bytes);
        if (actual_uncompressed == 0) {
            NOTCH_LOG_ERROR("[notchlc_decode] CPU LZ4 decompression failed");
            return cudaErrorInvalidValue;
        }
        // Parse header from the heap buffer (cache-friendly).
        if (parse_block_header(ctx->h_task_buf, (uint32_t)actual_uncompressed, hdr) != cudaSuccess) {
            NOTCH_LOG_ERROR("[notchlc_decode] parse_block_header failed (LZ4)");
            return cudaErrorInvalidValue;
        }
        // Sequential memcpy heap→WC-pinned: fast (write-combining buffers).
        std::memcpy(ctx->h_uncompressed, ctx->h_task_buf, actual_uncompressed);
        h_upload_src = ctx->h_uncompressed;
    } else if (format == NotchLCFormat::Uncompressed) {
        // Copy compressed payload to pinned staging for direct GPU DMA.
        if (compressed_size > ctx->max_compressed_bytes) {
            NOTCH_LOG_ERROR("[notchlc_decode] compressed_size exceeds max_compressed_bytes");
            return cudaErrorInvalidValue;
        }
        std::memcpy(ctx->h_compressed, h_compressed, compressed_size);
        actual_uncompressed = (size_t)uncompressed_size;
        uint8_t h_hdr_buf[256] = {};
        const size_t hdr_read = (actual_uncompressed >= 256u) ? 256u : actual_uncompressed;
        std::memcpy(h_hdr_buf, ctx->h_compressed, hdr_read);
        if (parse_block_header(h_hdr_buf, (uint32_t)actual_uncompressed, hdr) != cudaSuccess) {
            NOTCH_LOG_ERROR("[notchlc_decode] parse_block_header failed (Uncompressed)");
            return cudaErrorInvalidValue;
        }
        h_upload_src = ctx->h_compressed;
    } else {
        NOTCH_LOG_ERROR("[notchlc_decode] LZF format not supported");
        return cudaErrorNotSupported;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    float ms_cpu = (float)std::chrono::duration<double, std::milli>(t1 - t0).count();

    if ((int)hdr.width != ctx->width || (int)hdr.height != ctx->height)
        NOTCH_LOG_WARNING("[notchlc_decode] block header dimensions mismatch context");

    // 3. H->D upload + decode kernels, all on the CUDA stream.
    cudaEvent_t ev[7];
    for (int i = 0; i < 7; i++) cudaEventCreate(&ev[i]);

    cudaEventRecord(ev[0], s);
    CUDA_CHECK(cudaMemcpyAsync(ctx->d_uncompressed, h_upload_src,
                               actual_uncompressed, cudaMemcpyHostToDevice, s));
    cudaEventRecord(ev[1], s);
    CUDA_CHECK(launch_y_precompute_offsets(hdr, ctx, s));
    cudaEventRecord(ev[2], s);
    CUDA_CHECK(launch_y_decode(hdr, ctx->d_uncompressed, ctx->d_y, ctx->d_y_bit_offsets, s));
    cudaEventRecord(ev[3], s);
    CUDA_CHECK(launch_uv_decode(hdr, ctx->d_uncompressed, ctx->d_u, ctx->d_v, s));
    cudaEventRecord(ev[4], s);
    CUDA_CHECK(launch_a_decode_or_fill(hdr, ctx->d_uncompressed, ctx->d_a,
                                       ctx->width, ctx->height, s));
    cudaEventRecord(ev[5], s);
    CUDA_CHECK(launch_color_convert(ctx->d_y, ctx->d_u, ctx->d_v, ctx->d_a,
                                    ctx->d_bgra16, ctx->width, ctx->height, color_matrix, s));
    cudaEventRecord(ev[6], s);

    cudaStreamSynchronize(s);

    float ms_upload=0, ms_y_pre=0, ms_y_dec=0, ms_uv=0, ms_a=0, ms_cc=0, ms_gpu=0;
    cudaEventElapsedTime(&ms_upload, ev[0], ev[1]);
    cudaEventElapsedTime(&ms_y_pre,  ev[1], ev[2]);
    cudaEventElapsedTime(&ms_y_dec,  ev[2], ev[3]);
    cudaEventElapsedTime(&ms_uv,     ev[3], ev[4]);
    cudaEventElapsedTime(&ms_a,      ev[4], ev[5]);
    cudaEventElapsedTime(&ms_cc,     ev[5], ev[6]);
    cudaEventElapsedTime(&ms_gpu,    ev[0], ev[6]);
    for (int i = 0; i < 7; i++) cudaEventDestroy(ev[i]);

    static int _log_count = 0;
    if (++_log_count % 25 == 1) {
#if __has_include(<common/log.h>)
        CASPAR_LOG(info)
            << L"[notchlc_decode] STAGE ms:  cpu_lz4=" << ms_cpu
            << L"  upload(H>D)=" << ms_upload
            << L"  y_precompute=" << ms_y_pre
            << L"  y_decode=" << ms_y_dec
            << L"  uv_decode=" << ms_uv
            << L"  a_fill=" << ms_a
            << L"  color_convert=" << ms_cc
            << L"  gpu_total=" << ms_gpu;
#else
        fprintf(stderr, "[notchlc_decode] STAGE ms: cpu=%.1f upload=%.1f y_pre=%.1f y_dec=%.1f uv=%.1f a=%.1f cc=%.1f gpu=%.1f\n",
                ms_cpu, ms_upload, ms_y_pre, ms_y_dec, ms_uv, ms_a, ms_cc, ms_gpu);
#endif
    }

    return cudaSuccess;
}

// ---------------------------------------------------------------------------
// notchlc_decode_frame — result goes to GL texture (zero-copy device→device)
// ---------------------------------------------------------------------------

cudaError_t notchlc_decode_frame(
    NotchLCDecodeCtx*  ctx,
    const uint8_t*     h_compressed,
    size_t             compressed_size,
    uint32_t           uncompressed_size,
    NotchLCFormat      format,
    int                color_matrix,
    cudaArray_t        d_gl_array)
{
    // Run kernels (stream is synchronised before returning from decode_internal).
    CUDA_CHECK(decode_internal(ctx, h_compressed, compressed_size,
                                uncompressed_size, format, color_matrix));

    // ── Copy BGRA16 rows into the GL texture array ─────────────────────────
    cudaStream_t s = ctx->stream;
    const size_t row_bytes = (size_t)ctx->width * 4 * sizeof(uint16_t);

    CUDA_CHECK(cudaMemcpy2DToArrayAsync(
        d_gl_array,
        /*wOffset=*/0, /*hOffset=*/0,
        ctx->d_bgra16,
        /*spitch=*/row_bytes,
        /*width=*/row_bytes,
        /*height=*/(size_t)ctx->height,
        cudaMemcpyDeviceToDevice,
        s));

    CUDA_CHECK(cudaStreamSynchronize(s));
    return cudaSuccess;
}

// ---------------------------------------------------------------------------
// notchlc_decode_frame_to_host — result goes to a plain host buffer
// ---------------------------------------------------------------------------

cudaError_t notchlc_decode_frame_to_host(
    NotchLCDecodeCtx*  ctx,
    const uint8_t*     h_compressed,
    size_t             compressed_size,
    uint32_t           uncompressed_size,
    NotchLCFormat      format,
    int                color_matrix,
    uint16_t*          h_bgra16_out)
{
    CUDA_CHECK(decode_internal(ctx, h_compressed, compressed_size,
                                uncompressed_size, format, color_matrix));

    cudaStream_t s = ctx->stream;
    const size_t out_bytes = (size_t)ctx->width * ctx->height * 4 * sizeof(uint16_t);

    CUDA_CHECK(cudaMemcpyAsync(h_bgra16_out, ctx->d_bgra16, out_bytes,
                               cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
    return cudaSuccess;
}
