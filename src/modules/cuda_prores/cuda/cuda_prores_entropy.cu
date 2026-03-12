// cuda_prores_entropy.cu
// ProRes 422 two-pass slice entropy encoding on the GPU.
//
// Architecture
// ─────────────────────────────────────────────────────────────────────────────
//  Pass 1  k_count_bits    1 thread/slice  →  uint32_t bit_counts[num_slices]
//  Scan    CUB ExclusiveSum               →  uint32_t byte_offsets[num_slices+1]
//  Pass 2  k_encode_slices 1 thread/slice  →  write bitstream at correct offset
//
// Each slice is fully independent: Rice state is initialised at the start of
// every slice, so there is zero inter-slice dependency.
//
// Reference: Apple ProRes White Paper §"Entropy Coding";
//            FFmpeg libavcodec/proresenc_kostya.c (LGPL 2.1+)
// ---------------------------------------------------------------------------

#include "cuda_prores_rice.cuh"
#include "cuda_prores_tables.cuh"    // PRORES_SCAN, per-profile quant tables

#include <cub/device/device_scan.cuh>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstdint>

// ─── Per-slice Rice encoder ─────────────────────────────────────────────────
//
// coeffs        quantised, scan-reordered DCT coefficients for this slice
//               layout: [blocks_per_slice][64], int16_t
// blocks_per_slice  number of 8×8 blocks in the slice (luma + chroma mixed)
// bp            nullptr → counting pass (returns bit count, no writes)
//               non-null → encoding pass (writes bitstream, returns bit count)
//
// ProRes entropy coding per block:
//   DC  : horizontal predictor residual encoded as Rice(k_dc) + 1 sign bit
//   AC  : run/level pairs: run = Rice(k_run), |level|-1 = Rice(k_ac) + 1 sign
//   EOB : trailing-zeros run is emitted; the decoder infers EOB when run
//         consumes the remaining coefficients in the block.
// ─────────────────────────────────────────────────────────────────────────────
__device__ int encode_slice(
    const int16_t *coeffs,
    int            blocks_per_slice,
    BitPacker     *bp)
{
    int total_bits = 0;

    // ProRes uses per-component (Y/Cb/Cr) k-state reset at component boundaries.
    // For 422 the slice interleaves: Y Y Y Y Cb Cr (4 luma + 1 Cb + 1 Cr per
    // macroblock column).  A full implementation tracks component transitions.
    // This implementation uses a single shared state per slice — good enough
    // for Phase 1 SSIM validation; refine component-aware state in Phase 2.
    int k_dc  = 5;
    int k_ac  = 2;
    int k_run = 3;
    int prev_dc = 0;

    for (int b = 0; b < blocks_per_slice; b++) {
        const int16_t *blk = coeffs + b * 64;

        // ── DC residual ────────────────────────────────────────────────────
        int dc_diff = (int)blk[0] - prev_dc;
        prev_dc     = (int)blk[0];

        // Unsigned magnitude + explicit sign bit (ProRes DC coding)
        unsigned dc_abs = (unsigned)(dc_diff < 0 ? -dc_diff : dc_diff);
        if (bp) {
            rice_encode(bp, dc_abs, k_dc);
            bp_put(bp, dc_diff < 0 ? 1u : 0u, 1);
        }
        total_bits += rice_count(dc_abs, k_dc) + 1;
        rice_adapt_k(&k_dc, dc_abs);

        // ── AC coefficients ────────────────────────────────────────────────
        int run = 0;

        for (int i = 1; i < 64; i++) {
            int c = (int)blk[i]; // already scan-reordered by quant kernel

            if (c == 0) {
                run++;
            } else {
                // Encode run of preceding zeros
                if (bp) rice_encode(bp, (unsigned)run, k_run);
                total_bits += rice_count((unsigned)run, k_run);
                rice_adapt_k(&k_run, (unsigned)run);

                // Encode level: |c|-1 in Rice + 1 sign bit
                unsigned lv = (unsigned)(c < 0 ? -c : c) - 1u;
                if (bp) {
                    rice_encode(bp, lv, k_ac);
                    bp_put(bp, c < 0 ? 1u : 0u, 1);
                }
                total_bits += rice_count(lv, k_ac) + 1;
                rice_adapt_k(&k_ac, lv);
                run = 0;
            }
        }

        // Trailing zero run (EOB): emit the remaining run count so the decoder
        // can advance past it.  The decoder uses run==remaining to signal EOB.
        if (run > 0) {
            if (bp) rice_encode(bp, (unsigned)run, k_run);
            total_bits += rice_count((unsigned)run, k_run);
        }
    }

    return total_bits;
}

// ─── Pass 1: bit-count kernel ────────────────────────────────────────────────

__global__ void k_count_bits(
    const int16_t *d_coeffs,
    uint32_t      *d_bit_counts,
    int            blocks_per_slice,
    int            num_slices)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= num_slices) return;

    d_bit_counts[s] = (uint32_t)encode_slice(
        d_coeffs + (size_t)s * blocks_per_slice * 64,
        blocks_per_slice,
        nullptr);
}

// ─── Helper: convert bit counts to byte counts (ceil) ────────────────────────

__global__ void k_bits_to_bytes(uint32_t *v, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) v[i] = (v[i] + 7u) >> 3;
}

// ─── Pass 2: encode-and-write kernel ─────────────────────────────────────────

__global__ void k_encode_slices(
    const int16_t  *d_coeffs,
    const uint32_t *d_byte_offsets,
    uint8_t        *d_output,
    int             blocks_per_slice,
    int             num_slices)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= num_slices) return;

    BitPacker bp;
    uint8_t  *start = d_output + d_byte_offsets[s];
    bp_init(&bp, start);

    encode_slice(
        d_coeffs + (size_t)s * blocks_per_slice * 64,
        blocks_per_slice,
        &bp);

    bp_pad_byte(&bp);
}

// ─── Host context ────────────────────────────────────────────────────────────

struct CudaProResEnc {
    uint32_t *d_bit_counts;
    uint32_t *d_byte_offsets; // [max_slices + 1]
    void     *d_cub_temp;
    size_t    cub_temp_bytes;
    int       max_slices;
};

CudaProResEnc* cuda_prores_enc_create(int max_slices)
{
    auto *ctx = (CudaProResEnc*)malloc(sizeof(CudaProResEnc));
    if (!ctx) return nullptr;

    ctx->max_slices = max_slices;

    cudaMalloc(&ctx->d_bit_counts,   (size_t)max_slices * sizeof(uint32_t));
    cudaMalloc(&ctx->d_byte_offsets, (size_t)(max_slices + 1) * sizeof(uint32_t));

    // Dry-run to determine CUB temp storage requirement.
    ctx->d_cub_temp     = nullptr;
    ctx->cub_temp_bytes = 0;
    cub::DeviceScan::ExclusiveSum(
        ctx->d_cub_temp, ctx->cub_temp_bytes,
        ctx->d_bit_counts, ctx->d_byte_offsets, max_slices);
    cudaMalloc(&ctx->d_cub_temp, ctx->cub_temp_bytes);

    return ctx;
}

void cuda_prores_enc_destroy(CudaProResEnc *ctx)
{
    if (!ctx) return;
    cudaFree(ctx->d_bit_counts);
    cudaFree(ctx->d_byte_offsets);
    cudaFree(ctx->d_cub_temp);
    free(ctx);
}

// Encode one frame's worth of pre-quantised scan-reordered coefficients.
//
// d_coeffs           device: [num_slices][blocks_per_slice][64] int16_t
// d_output           device: output bitstream buffer (must be large enough)
// d_slice_offsets    device: [num_slices+1] uint32_t, filled by this call;
//                    element [num_slices] equals total output bytes
// stream             CUDA stream for all kernel and CUB launches
//
// Returns 0 on success.
int cuda_prores_enc_frame(
    CudaProResEnc  *ctx,
    const int16_t  *d_coeffs,
    int             num_slices,
    int             blocks_per_slice,
    uint8_t        *d_output,
    uint32_t       *d_slice_offsets,
    cudaStream_t    stream)
{
    const int T = 128;
    const int G = (num_slices + T - 1) / T;

    // Pass 1: count bits per slice
    k_count_bits<<<G, T, 0, stream>>>(
        d_coeffs, ctx->d_bit_counts, blocks_per_slice, num_slices);

    // Convert bits → bytes (ceil division)
    k_bits_to_bytes<<<G, T, 0, stream>>>(ctx->d_bit_counts, num_slices);

    // Prefix-sum → byte offset per slice
    cub::DeviceScan::ExclusiveSum(
        ctx->d_cub_temp, ctx->cub_temp_bytes,
        ctx->d_bit_counts, d_slice_offsets,
        num_slices, stream);

    // Pass 2: encode into output at computed offsets
    k_encode_slices<<<G, T, 0, stream>>>(
        d_coeffs, d_slice_offsets, d_output, blocks_per_slice, num_slices);

    return 0;
}
