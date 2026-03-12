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

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/opt.h>
#include <libavutil/imgutils.h>
}

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
        const bool is_4444 = (profile == 4);

        ctx.width          = w;
        ctx.height         = h;
        ctx.profile        = profile;
        ctx.is_4444        = is_4444;
        ctx.has_alpha      = false;
        ctx.q_scale        = 8;

        // Macroblock geometry  (MB = 16×16 pixels)
        ctx.slices_per_row = slices_per_row;
        ctx.mbs_per_slice  = (w / 16) / slices_per_row;
        ctx.num_slices     = slices_per_row * ((h + 15) / 16);
        const int bpm      = is_4444 ? 12 : 8;  // DCT blocks per MB
        ctx.blocks_per_slice = ctx.mbs_per_slice * bpm;

        const int yp = w * h;
        const int cp = is_4444 ? yp : (w / 2) * h;  // 4444 chroma = full-res
        cce(cudaMalloc(&ctx.d_y,  yp * sizeof(int16_t)), "d_y");
        cce(cudaMalloc(&ctx.d_cb, cp * sizeof(int16_t)), "d_cb");
        cce(cudaMalloc(&ctx.d_cr, cp * sizeof(int16_t)), "d_cr");
        cce(cudaMalloc(&ctx.d_coeffs_y,  yp * sizeof(int16_t)), "d_cy");
        cce(cudaMalloc(&ctx.d_coeffs_cb, cp * sizeof(int16_t)), "d_ccb");
        cce(cudaMalloc(&ctx.d_coeffs_cr, cp * sizeof(int16_t)), "d_ccr");
        const size_t se = (size_t)ctx.num_slices * ctx.blocks_per_slice * 64;
        cce(cudaMalloc(&ctx.d_coeffs_slice, se * sizeof(int16_t)), "d_cs");
        const size_t bs = se * sizeof(int16_t) * 2 + ctx.num_slices * 64;
        cce(cudaMalloc(&ctx.d_bitstream,     bs), "d_bs");
        cce(cudaMalloc(&ctx.d_slice_offsets, (ctx.num_slices + 1) * sizeof(uint32_t)), "d_so");
        cce(cudaMalloc(&ctx.d_slice_sizes,    ctx.num_slices      * sizeof(uint32_t)), "d_sz");
        cce(cudaMalloc(&ctx.d_bit_counts,     ctx.num_slices * 3  * sizeof(uint32_t)), "d_bc");
        ctx.cub_temp_bytes = 8 * 1024 * 1024;
        cce(cudaMalloc(&ctx.d_cub_temp, ctx.cub_temp_bytes), "d_ct");
        const size_t fb = (size_t)w * h * 8;  // headroom for 4444
        ctx.h_frame_buf_size = fb;
        cce(cudaMallocHost(&ctx.h_frame_buf, fb), "h_fb");

        ctx.d_alpha        = nullptr;
        ctx.d_coeffs_alpha = nullptr;
        if (is_4444)
            cce(cudaMalloc(&ctx.d_alpha, yp * sizeof(int16_t)), "d_alpha");
    }

    ~FrameCtxGuard() {
        cudaFree(ctx.d_y);   cudaFree(ctx.d_cb);  cudaFree(ctx.d_cr);
        cudaFree(ctx.d_coeffs_y); cudaFree(ctx.d_coeffs_cb); cudaFree(ctx.d_coeffs_cr);
        cudaFree(ctx.d_coeffs_slice);
        cudaFree(ctx.d_bitstream); cudaFree(ctx.d_slice_offsets);  cudaFree(ctx.d_slice_sizes);
        cudaFree(ctx.d_bit_counts); cudaFree(ctx.d_cub_temp);
        cudaFreeHost(ctx.h_frame_buf);
        if (ctx.d_alpha)        cudaFree(ctx.d_alpha);
        if (ctx.d_coeffs_alpha) cudaFree(ctx.d_coeffs_alpha);
    }

    FrameCtxGuard(const FrameCtxGuard &) = delete;
};

// ---------------------------------------------------------------------------
// CPU baseline: encode num_frames through FFmpeg prores_ks.
// Returns fps (frames per second).  Prints one line to stdout.
// ---------------------------------------------------------------------------
static double bench_cpu(int width, int height, int profile_idx, int num_frames)
{
    static const char *NAMES[]   = {"proxy","lt","standard","hq","4444"};
    static const char *DISPLAY[] = {"Proxy","LT","Std","HQ","4444"};
    const bool is_4444 = (profile_idx == 4);

    const AVCodec *codec = avcodec_find_encoder_by_name("prores_ks");
    if (!codec) {
        fprintf(stderr, "bench_cpu: prores_ks not found\n");
        return 0.0;
    }
    AVCodecContext *enc = avcodec_alloc_context3(codec);
    enc->width     = width;
    enc->height    = height;
    enc->pix_fmt   = is_4444 ? AV_PIX_FMT_YUVA444P10LE : AV_PIX_FMT_YUV422P10LE;
    AVRational tb  = {1,25};
    enc->time_base = tb;
    av_opt_set(enc->priv_data, "profile", NAMES[profile_idx], 0);
    av_opt_set(enc->priv_data, "vendor",  "apl0", 0);
    if (avcodec_open2(enc, codec, nullptr) < 0) {
        fprintf(stderr, "bench_cpu: avcodec_open2 failed\n");
        avcodec_free_context(&enc);
        return 0.0;
    }

    AVFrame *frame = av_frame_alloc();
    frame->format = enc->pix_fmt;
    frame->width  = width;
    frame->height = height;
    av_frame_get_buffer(frame, 0);
    av_frame_make_writable(frame);
    // Fill with 50% grey (Y≈512, Cb/Cr≈512 in 10-bit narrow range)
    for (int p = 0; p < (is_4444 ? 4 : 3); p++) {
        const int stride = frame->linesize[p];
        const int rows   = (p == 3) ? height : ((p == 0) ? height :
                           ((enc->pix_fmt == AV_PIX_FMT_YUV422P10LE) ? height : height));
        // Write 0x0200 (= 512 in LE uint16) into every 2-byte sample
        uint8_t *row_ptr = frame->data[p];
        for (int r = 0; r < rows; r++) {
            uint16_t *s = (uint16_t *)row_ptr;
            const int cols = stride / 2;
            for (int c = 0; c < cols; c++) s[c] = (p == 3) ? 1023 : 512;
            row_ptr += stride;
        }
    }

    AVPacket *pkt = av_packet_alloc();

    // Warm-up (prores_ks is intra-only, no flush needed)
    for (int i = 0; i < 3; i++) {
        frame->pts = i;
        avcodec_send_frame(enc, frame);
        while (avcodec_receive_packet(enc, pkt) == 0) av_packet_unref(pkt);
    }
    // No avcodec_flush_buffers — prores_ks doesn't support it

    // Timed run
    LARGE_INTEGER freq, t0, t1;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t0);

    size_t total_bytes = 0;
    for (int i = 0; i < num_frames; i++) {
        frame->pts = i;
        avcodec_send_frame(enc, frame);
        while (avcodec_receive_packet(enc, pkt) == 0) {
            total_bytes += (size_t)pkt->size;
            av_packet_unref(pkt);
        }
    }
    // Drain
    avcodec_send_frame(enc, nullptr);
    while (avcodec_receive_packet(enc, pkt) == 0) {
        total_bytes += (size_t)pkt->size;
        av_packet_unref(pkt);
    }

    QueryPerformanceCounter(&t1);
    const double elapsed_ms = (double)(t1.QuadPart - t0.QuadPart) / freq.QuadPart * 1000.0;
    const double fps  = num_frames / (elapsed_ms / 1000.0);
    const double mbps = (double)total_bytes * 8.0 / (elapsed_ms / 1000.0) / 1e6;
    const double avg_kb = (double)total_bytes / num_frames / 1024.0;

    fprintf(stdout,
        "  CPU  %-5s  %dx%d  %d frames  %.1f ms  "
        "->  %.1f fps  |  %.0f Mbps  |  ~%.0f KB/frame\n",
        DISPLAY[profile_idx], width, height, num_frames,
        elapsed_ms, fps, mbps, avg_kb);

    av_frame_free(&frame);
    av_packet_free(&pkt);
    avcodec_free_context(&enc);
    return fps;
}

// ---------------------------------------------------------------------------
// Run a benchmark for one (width, height, profile, num_frames) combination
// Returns fps.
// ---------------------------------------------------------------------------
static double bench(int width, int height, int profile, int num_frames)
{
    const bool is_4444 = (profile == 4);
    // 4444 expects BGRA8; 422 expects V210
    const size_t input_bytes = is_4444
        ? (size_t)width * height * 4
        : (size_t)((width + 5) / 6) * 16 * height;

    void *d_input = nullptr;
    cce(cudaMalloc(&d_input, input_bytes), "d_input bench");
    cudaMemset(d_input, 0x40, input_bytes);

    FrameCtxGuard g(width, height, profile);
    cudaStream_t stream;
    cce(cudaStreamCreate(&stream), "bench stream");

    static const ProResColorDesc K_SDR = {1, 1, 1, {},{},0,0,0,0,0,0};

    // Warm up (GPU JIT + pipeline fill)
    for (int i = 0; i < 5; i++) {
        size_t sz = 0;
        if (is_4444)
            prores_encode_frame_444(&g.ctx, (const uint8_t *)d_input,
                                    g.ctx.h_frame_buf, &sz, stream, &K_SDR);
        else
            prores_encode_frame(&g.ctx, (const uint32_t *)d_input,
                                g.ctx.h_frame_buf, &sz, stream, &K_SDR);
    }
    // Check warm-up sync — catches sticky errors from kernel launches
    cce(cudaStreamSynchronize(stream), "warm-up sync");

    // Timed loop
    LARGE_INTEGER freq, t0, t1;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t0);

    size_t total_bytes = 0;
    for (int i = 0; i < num_frames; i++) {
        size_t sz = 0;
        if (is_4444)
            prores_encode_frame_444(&g.ctx, (const uint8_t *)d_input,
                                    g.ctx.h_frame_buf, &sz, stream, &K_SDR);
        else
            prores_encode_frame(&g.ctx, (const uint32_t *)d_input,
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
        "  GPU  %-5s  %dx%d  %d frames  %.1f ms  "
        "->  %.1f fps  |  %.0f Mbps  |  ~%.0f KB/frame\n",
        PROFILE_NAMES[profile], width, height, num_frames,
        elapsed_ms, fps, mbps, avg_frame_kb);

    cudaFree(d_input);
    cudaStreamDestroy(stream);
    return fps;
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

    static const char *PROF_NAMES[] = {"Proxy","LT","Std","HQ","4444"};

    // Helper lambda: run GPU+CPU for one resolution, fill result arrays.
    // gpu_fps[5] and cpu_fps[5] indexed by profile.
    auto run_comparison = [&](int W, int H) {
        double gpu_fps[5] = {}, cpu_fps[5] = {};

        fprintf(stdout, "\n=== GPU encode throughput  %dx%d  (%d frames) ===\n", W, H, num_frames);
        try {
            for (int p = 0; p <= 4; p++) {
                if (!(profile_mask & (1 << p))) continue;
                gpu_fps[p] = bench(W, H, p, num_frames);
            }
        } catch (const std::exception &e) {
            fprintf(stderr, "GPU bench exception: %s\n", e.what());
        }

        fprintf(stdout, "\n=== CPU encode throughput  %dx%d  (%d frames, FFmpeg prores_ks) ===\n", W, H, num_frames);
        for (int p = 0; p <= 4; p++) {
            if (!(profile_mask & (1 << p))) continue;
            cpu_fps[p] = bench_cpu(W, H, p, num_frames);
        }

        // Comparison table
        fprintf(stdout, "\n  GPU vs CPU comparison  %dx%d:\n", W, H);
        fprintf(stdout, "  %-7s  %9s  %9s  %9s\n", "Profile", "GPU fps", "CPU fps", "Speedup");
        fprintf(stdout, "  %-7s  %9s  %9s  %9s\n", "-------", "-------", "-------", "-------");
        for (int p = 0; p <= 4; p++) {
            if (!(profile_mask & (1 << p))) continue;
            if (cpu_fps[p] > 0.0) {
                fprintf(stdout, "  %-7s  %9.1f  %9.1f  %8.1fx\n",
                    PROF_NAMES[p], gpu_fps[p], cpu_fps[p],
                    gpu_fps[p] / cpu_fps[p]);
            } else {
                fprintf(stdout, "  %-7s  %9.1f  %9s  %9s\n",
                    PROF_NAMES[p], gpu_fps[p], "n/a", "n/a");
            }
        }
        fprintf(stdout, "\n");
    };

    run_comparison(width, height);

    if (width != 1920 || height != 1080)
        run_comparison(1920, 1080);

    return 0;
}
