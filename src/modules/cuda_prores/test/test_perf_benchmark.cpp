// test_perf_benchmark.cpp
// GPU ProRes encode throughput benchmark.
//
// Measures the end-to-end throughput of the CUDA ProRes pipeline:
//   V210 upload → BGRA→V210 (or direct V210) → DCT/quant → entropy → host copy
//
// Reports frames-per-second and encoded Mbps for each profile at each resolution.
// This test does NOT write files to disk to isolate pure GPU encode cost.
//
// Usage
// ─────────────────────────────────────────────────────────────────────────────
//   test_perf_benchmark.exe [width height] [--frames N] [--profiles 0-4]
//
//   Default: 3840x2160, 100 frames, profiles 0-4
//
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "../cuda/cuda_prores_frame.h"
#include "../cuda/cuda_prores_tables.cuh"

// ---------------------------------------------------------------------------
// CUDA helper
// ---------------------------------------------------------------------------
static void cce(cudaError_t e, const char *ctx)
{
    if (e != cudaSuccess) {
        char msg[256];
        snprintf(msg, sizeof(msg), "CUDA error [%s]: %s", ctx, cudaGetErrorString(e));
        throw std::runtime_error(msg);
    }
}

// ---------------------------------------------------------------------------
// ProResFrameCtx RAII wrapper
// ---------------------------------------------------------------------------
struct FrameCtxGuard {
    ProResFrameCtx ctx = {};

    FrameCtxGuard(int w, int h, int profile, int slices_per_row = 4)
    {
        ctx.width          = w;
        ctx.height         = h;
        ctx.profile        = profile;
        ctx.slices_per_row = slices_per_row;
        ctx.q_scale        = 8;
        const int mb_w = w / 8, mb_h = h / 8;
        ctx.num_slices       = (mb_w / slices_per_row) * mb_h;
        ctx.blocks_per_slice = slices_per_row * 6;

        const int yp = w * h, cp = (w / 2) * h;
        cce(cudaMalloc(&ctx.d_y,  yp * sizeof(int16_t)), "d_y");
        cce(cudaMalloc(&ctx.d_cb, cp * sizeof(int16_t)), "d_cb");
        cce(cudaMalloc(&ctx.d_cr, cp * sizeof(int16_t)), "d_cr");
        cce(cudaMalloc(&ctx.d_coeffs_y,  yp * sizeof(int16_t)), "d_cy");
        cce(cudaMalloc(&ctx.d_coeffs_cb, cp * sizeof(int16_t)), "d_ccb");
        cce(cudaMalloc(&ctx.d_coeffs_cr, cp * sizeof(int16_t)), "d_ccr");
        const size_t se = (size_t)ctx.num_slices * ctx.blocks_per_slice * 64;
        cce(cudaMalloc(&ctx.d_coeffs_slice, se * sizeof(int16_t)), "d_cs");
        const size_t bs = se * sizeof(int16_t) * 2 + ctx.num_slices * 32;
        cce(cudaMalloc(&ctx.d_bitstream,     bs), "d_bs");
        cce(cudaMalloc(&ctx.d_slice_offsets, (ctx.num_slices + 1) * sizeof(uint32_t)), "d_so");
        cce(cudaMalloc(&ctx.d_bit_counts,     ctx.num_slices * sizeof(uint32_t)), "d_bc");
        ctx.cub_temp_bytes = 8 * 1024 * 1024;
        cce(cudaMalloc(&ctx.d_cub_temp, ctx.cub_temp_bytes), "d_ct");
        const size_t fb = (size_t)w * h * 4;
        ctx.h_frame_buf_size = fb;
        cce(cudaMallocHost(&ctx.h_frame_buf, fb), "h_fb");
    }

    ~FrameCtxGuard() {
        cudaFree(ctx.d_y);   cudaFree(ctx.d_cb);  cudaFree(ctx.d_cr);
        cudaFree(ctx.d_coeffs_y); cudaFree(ctx.d_coeffs_cb); cudaFree(ctx.d_coeffs_cr);
        cudaFree(ctx.d_coeffs_slice);
        cudaFree(ctx.d_bitstream); cudaFree(ctx.d_slice_offsets); cudaFree(ctx.d_bit_counts);
        cudaFree(ctx.d_cub_temp);
        cudaFreeHost(ctx.h_frame_buf);
    }

    FrameCtxGuard(const FrameCtxGuard &) = delete;
};

// ---------------------------------------------------------------------------
// Run a benchmark for one (width, height, profile, num_frames) combination
// ---------------------------------------------------------------------------
static void bench(int width, int height, int profile, int num_frames)
{
    const size_t v210_bytes = (size_t)((width + 5) / 6) * 16 * height;

    // Allocate a constant-grey V210 device frame
    void *d_v210 = nullptr;
    cce(cudaMalloc(&d_v210, v210_bytes), "d_v210 bench");
    cudaMemset(d_v210, 0x20, v210_bytes); // arbitrary grey fill

    FrameCtxGuard g(width, height, profile);
    cudaStream_t stream;
    cce(cudaStreamCreate(&stream), "bench stream");

    static const ProResColorDesc K_SDR = {1, 1, 1, {},{},0,0,0,0,0,0};

    // Warm up (GPU jit + pipeline fill)
    for (int i = 0; i < 5; i++) {
        size_t sz = 0;
        prores_encode_frame(&g.ctx, (const uint32_t *)d_v210,
                            g.ctx.h_frame_buf, &sz, stream, &K_SDR);
    }
    cudaStreamSynchronize(stream);

    // Timed loop
    LARGE_INTEGER freq, t0, t1;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t0);

    size_t total_bytes = 0;
    for (int i = 0; i < num_frames; i++) {
        size_t sz = 0;
        prores_encode_frame(&g.ctx, (const uint32_t *)d_v210,
                            g.ctx.h_frame_buf, &sz, stream, &K_SDR);
        total_bytes += sz;
    }
    cudaStreamSynchronize(stream);

    QueryPerformanceCounter(&t1);
    double elapsed_ms = (double)(t1.QuadPart - t0.QuadPart) / freq.QuadPart * 1000.0;
    double fps   = num_frames / (elapsed_ms / 1000.0);
    double mbps  = (double)total_bytes * 8 / (elapsed_ms / 1000.0) / 1e6;
    double avg_frame_kb = (double)total_bytes / num_frames / 1024.0;

    static const char *PROFILE_NAMES[] = {"Proxy","LT","Std","HQ","4444"};
    fprintf(stdout,
        "  %-4s  %dx%d  %d frames  %.1f ms  "
        "→  %.1f fps  |  %.0f Mbps  |  ~%.0f KB/frame\n",
        PROFILE_NAMES[profile], width, height, num_frames,
        elapsed_ms, fps, mbps, avg_frame_kb);

    cudaFree(d_v210);
    cudaStreamDestroy(stream);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char *argv[])
{
    int width  = 3840;
    int height = 2160;
    int num_frames = 100;
    // Profiles to test (default: all)
    int profile_mask = 0x1F; // bit 0-4

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--frames") == 0 && i+1 < argc)
            num_frames = atoi(argv[++i]);
        else if (strcmp(argv[i], "--profiles") == 0 && i+1 < argc) {
            profile_mask = 0;
            int p = atoi(argv[++i]);
            // Support "0-4" range or single integer
            profile_mask = 1 << std::max(0, std::min(p, 4));
        } else if (i+1 < argc && atoi(argv[i]) > 0 && atoi(argv[i+1]) > 0) {
            width  = atoi(argv[i]);
            height = atoi(argv[i+1]);
            i++;
        }
    }

    int dev_count = 0;
    cudaGetDeviceCount(&dev_count);
    if (dev_count == 0) { fprintf(stderr, "No CUDA devices\n"); return 1; }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    fprintf(stdout, "CUDA device: %s  (sm_%d%d)  %d GB VRAM\n",
            prop.name, prop.major, prop.minor,
            (int)(prop.totalGlobalMem / (1024*1024*1024)));

    prores_tables_upload();

    fprintf(stdout, "\nEncode throughput — %d frames per run:\n", num_frames);
    fprintf(stdout, "%-6s  %-12s  %s  %s  %s  %s\n",
            "Prof", "Resolution", "Frames", "Time", "FPS", "Mbps");

    try {
        for (int p = 0; p <= 4; p++) {
            if (!(profile_mask & (1 << p))) continue;
            bench(width, height, p, num_frames);
        }

        // Also run 1080p25 for comparison
        if (width != 1920 || height != 1080) {
            fprintf(stdout, "\n1920x1080 comparison:\n");
            for (int p = 0; p <= 4; p++) {
                if (!(profile_mask & (1 << p))) continue;
                bench(1920, 1080, p, num_frames);
            }
        }
    } catch (const std::exception &e) {
        fprintf(stderr, "Exception: %s\n", e.what());
        return 1;
    }

    return 0;
}
