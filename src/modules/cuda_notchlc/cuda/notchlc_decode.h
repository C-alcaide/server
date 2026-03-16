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
 * NotchLC is a codec specification by Derivative Inc., available under the
 * Creative Commons Attribution 4.0 International License.
 */

// notchlc_decode.h
// GPU-side decode context and entry points for the NotchLC CUDA decoder.
// ---------------------------------------------------------------------------
#pragma once

#include <cuda_runtime.h>
#include <nvcomp/lz4.h>
#include <stdint.h>
#include <stddef.h>

// Must include producer header for NotchLCFormat enum.
// (Alternatively the enum could live here; keeping it in the demuxer header
//  is fine because the decoder only needs the value, not avformat types.)
#include "../producer/notchlc_demuxer.h"

// ---------------------------------------------------------------------------
// NotchLCDecodeCtx — all device-side resources for one decode slot.
// Each slot is exclusively used by one in-flight decode at a time.
// ---------------------------------------------------------------------------
struct NotchLCDecodeCtx {
    // ── Geometry ─────────────────────────────────────────────────────────
    int    width  = 0;
    int    height = 0;

    // ── Compressed input staging ──────────────────────────────────────────
    // Pinned host memory for async H→D transfer.
    uint8_t* h_compressed = nullptr;              // cudaMallocHost
    size_t   max_compressed_bytes = 0;

    // Pinned host memory for CPU-decompressed uncompressed payload.
    // The LZ4 path decompresses h_compressed → h_uncompressed on CPU,
    // then uploads h_uncompressed → d_uncompressed via async DMA.
    // This is ~100× faster than GPU nvcomp on a single monolithic chunk.
    uint8_t* h_uncompressed = nullptr;            // cudaMallocHost
    size_t   max_uncompressed_bytes = 0;

    // Temporary heap buffer for LZ4 decompression output (non-pinned, cacheable).
    // LZ4 writes here first (back-references stay in CPU cache), then the result
    // is memcpy'd into h_uncompressed (pinned/WC) for fast GPU DMA.
    uint8_t* h_task_buf = nullptr;                // malloc (regular heap)

    // Device buffer for the compressed data (kept for potential future use;
    // not populated on the CPU-LZ4 path).
    uint8_t* d_compressed = nullptr;              // cudaMalloc (may be null)
    size_t   d_compressed_alloc = 0;

    // ── Decompressed output ───────────────────────────────────────────────
    uint8_t* d_uncompressed = nullptr;            // cudaMalloc, nvcomp writes here
    size_t   d_uncompressed_alloc = 0;

    // ── nvcomp temp workspace ─────────────────────────────────────────────
    void*    d_nvcomp_temp       = nullptr;
    size_t   d_nvcomp_temp_bytes = 0;

    // ── nvcomp batch API (batch_size = 1) — device-side arrays ────────────
    // nvcompBatchedLZ4DecompressAsync takes arrays-of-pointers on the device.
    const void** d_compressed_ptrs           = nullptr;  // [1] → d_compressed
    size_t*       d_compressed_bytes_arr      = nullptr;  // [1] compressed_size
    size_t*       d_uncompressed_buffer_bytes = nullptr;  // [1] max_uncompressed_bytes
    size_t*       d_uncompressed_actual_bytes = nullptr;  // [1] output: actual bytes written
    void**        d_uncompressed_ptrs         = nullptr;  // [1] → d_uncompressed
    nvcompStatus_t* d_nvcomp_statuses         = nullptr;  // [1]

    // ── Decoded planes (12-bit values in uint16_t) ────────────────────────
    uint16_t* d_y     = nullptr;  // [height × width]
    uint16_t* d_u     = nullptr;  // [height × width]  (Co chroma)
    uint16_t* d_v     = nullptr;  // [height × width]  (Cg chroma)
    uint16_t* d_a     = nullptr;  // [height × width]  (alpha)

    // ── Final BGRA16 output ────────────────────────────────────────────────
    uint16_t* d_bgra16 = nullptr; // [height × width × 4]

    // ── Y-decode prefix-sum scratch ─────────────────────────────────────
    // Pass 1 (k_y_compute_widths):  write per-block bit widths in parallel.
    // Pass 2 (k_y_prefix_rows):     1 thread per row-group does a serial
    //   prefix scan across its row.  Parallel across rows eliminates the
    //   original O(blocks_x^2 × blocks_y) bottleneck; total work is O(W×H/16).
    uint32_t* d_y_bit_widths   = nullptr;   // [blocks_y × blocks_x]
    uint32_t* d_y_bit_offsets  = nullptr;   // [blocks_y × blocks_x]

    // ── CUDA stream ───────────────────────────────────────────────────────
    cudaStream_t stream = nullptr;
};

// ---------------------------------------------------------------------------
// notchlc_decode_ctx_create / destroy
// Must be called from the CUDA device thread (cudaSetDevice already called).
// max_compressed_bytes    — worst-case compressed payload size (no outer hdr)
// max_uncompressed_bytes  — == uncompressed_size from packet header
// ---------------------------------------------------------------------------
cudaError_t notchlc_decode_ctx_create(NotchLCDecodeCtx* ctx,
                                       int    width,
                                       int    height,
                                       size_t max_compressed_bytes,
                                       size_t max_uncompressed_bytes);

void notchlc_decode_ctx_destroy(NotchLCDecodeCtx* ctx);

// ---------------------------------------------------------------------------
// notchlc_decode_frame
//
// Decompresses and decodes one NotchLC video frame.
//
// Parameters:
//   ctx              — decode context (must already be created)
//   h_compressed     — host pointer to compressed payload bytes
//                      (the raw avpacket data MINUS the 16-byte outer header)
//   compressed_size  — byte count of the compressed payload
//   uncompressed_size— from packet header field 1 (bytes 4-7)
//   format           — from packet header field 3 (bytes 12-15)
//   color_matrix     — see NOTCHLC_CM_* constants in notchlc_ycocg_to_bgra16.cuh
//   d_gl_array       — mapped cudaArray from CudaGLTexture::map(); may NOT be null
//
// The caller must map the GL texture before this call and unmap after.
// ---------------------------------------------------------------------------
cudaError_t notchlc_decode_frame(
    NotchLCDecodeCtx*           ctx,
    const uint8_t*              h_compressed,
    size_t                      compressed_size,
    uint32_t                    uncompressed_size,
    caspar::cuda_notchlc::NotchLCFormat format,
    int                         color_matrix,
    cudaArray_t                 d_gl_array);

// ---------------------------------------------------------------------------
// notchlc_decode_frame_to_host
//
// Same as notchlc_decode_frame but writes to a plain host buffer instead of
// a GL texture. Useful for unit tests and headless processing.
// h_bgra16_out must be at least width*height*4*sizeof(uint16_t) bytes.
// ---------------------------------------------------------------------------
cudaError_t notchlc_decode_frame_to_host(
    NotchLCDecodeCtx*           ctx,
    const uint8_t*              h_compressed,
    size_t                      compressed_size,
    uint32_t                    uncompressed_size,
    caspar::cuda_notchlc::NotchLCFormat format,
    int                         color_matrix,
    uint16_t*                   h_bgra16_out);

// ---------------------------------------------------------------------------
// NotchBlockHeader — parsed from the first 40 bytes of the decompressed data.
// Exposed here so the producer's LZ4-worker threads can parse it and pass to
// notchlc_decode_frame_gpu_phase without needing an internal reparse.
// ---------------------------------------------------------------------------
struct NotchBlockHeader {
    uint32_t width;
    uint32_t height;
    uint32_t uv_offset_data_offset;
    uint32_t y_control_data_offset;
    uint32_t a_control_word_offset;
    uint32_t uv_data_offset;
    uint32_t y_data_size;
    uint32_t a_data_offset;
    uint32_t a_count_size;
    uint32_t data_end;
    uint32_t y_data_row_offsets;
    uint32_t y_data_offset;
    uint32_t uv_count_offset;
    bool     has_alpha;
};

// ---------------------------------------------------------------------------
// Two-phase decode API for parallel LZ4 + GPU pipelining
//
// Phase 1 (CPU, any thread):
//   Decompresses the LZ4 payload, parses the block header.
//   On success: ctx->h_uncompressed contains the full decompressed frame,
//   hdr_out is populated, actual_uncompressed_out is set.
// ---------------------------------------------------------------------------
cudaError_t notchlc_decode_cpu_phase(
    NotchLCDecodeCtx*              ctx,
    const uint8_t*                 h_compressed,
    size_t                         compressed_size,
    uint32_t                       uncompressed_size,
    caspar::cuda_notchlc::NotchLCFormat format,
    NotchBlockHeader&              hdr_out,
    size_t&                        actual_uncompressed_out);

// ---------------------------------------------------------------------------
// Phase 2 (GPU / CUDA-GL thread):
//   Uploads ctx->h_uncompressed → device, launches all GPU kernels,
//   synchronises stream, runs color convert.
//   Pass d_gl_array != nullptr to write into GL texture (zero-copy path),
//   or null to leave result in ctx->d_bgra16.
// ---------------------------------------------------------------------------
cudaError_t notchlc_decode_gpu_phase(
    NotchLCDecodeCtx*        ctx,
    const NotchBlockHeader&  hdr,
    size_t                   actual_uncompressed,
    int                      color_matrix,
    cudaArray_t              d_gl_array);   // nullable: null → write to ctx->d_bgra16

