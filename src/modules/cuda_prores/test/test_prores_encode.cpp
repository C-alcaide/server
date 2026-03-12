// test_prores_encode.cpp
// Phase 1f standalone validation test.
//
// Reads a raw 4K 10-bit V210 file (one or more frames), runs the CUDA ProRes
// encoder pipeline, writes a .mov file with SMPTE timecode, then invokes
// ffprobe to confirm the output is a valid QuickTime/ProRes container.
//
// Usage
// ─────────────────────────────────────────────────────────────────────────────
//   test_prores_encode.exe  <input.v210>  <width>  <height>  <profile>  <output.mov>
//
//   profile: 0=Proxy, 1=LT, 2=Standard, 3=HQ, 4=4444
//
// Validation
// ─────────────────────────────────────────────────────────────────────────────
//   To compare encoded output against FFmpeg ProRes reference:
//     ffmpeg -s 3840x2160 -pix_fmt yuv422p10le -i input.yuv -c:v prores_ks \
//            -profile:v 3 reference.mov
//     ffmpeg -i output.mov -i reference.mov \
//            -lavfi "[0:v][1:v]ssim=stats_file=ssim.txt" -f null -
//
//   To verify timecode track:
//     ffprobe -v quiet -print_format json -show_streams <output.mov>
//     # Look for codec_type="data" with "timecode" tag in the tags section.
//
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
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
#include "../muxer/mov_muxer.h"
#include "../timecode.h"

// ---------------------------------------------------------------------------
// CUDA helper
// ---------------------------------------------------------------------------
static void cuda_check(cudaError_t e, const char *ctx)
{
    if (e != cudaSuccess) {
        char msg[256];
        snprintf(msg, sizeof(msg), "CUDA error [%s]: %s", ctx, cudaGetErrorString(e));
        throw std::runtime_error(msg);
    }
}

// ---------------------------------------------------------------------------
// ProResFrameCtx helpers
// ---------------------------------------------------------------------------
static void frame_ctx_create(ProResFrameCtx &ctx,
                              int width, int height, int profile,
                              int slices_per_row = 4)
{
    ctx.width         = width;
    ctx.height        = height;
    ctx.profile       = profile;
    ctx.slices_per_row = slices_per_row;
    ctx.q_scale       = 8; // mid-range quality

    int mb_w = width  / 8;
    int mb_h = height / 8;
    ctx.num_slices       = (mb_w / slices_per_row) * mb_h;
    ctx.blocks_per_slice = slices_per_row * 6; // 4 luma + 1 Cb + 1 Cr per MB

    // Planar input buffers
    int y_pixels   = width * height;
    int c_pixels   = (width / 2) * height;
    cuda_check(cudaMalloc(&ctx.d_y,  y_pixels * sizeof(int16_t)), "d_y");
    cuda_check(cudaMalloc(&ctx.d_cb, c_pixels * sizeof(int16_t)), "d_cb");
    cuda_check(cudaMalloc(&ctx.d_cr, c_pixels * sizeof(int16_t)), "d_cr");

    // DCT coefficient buffers
    cuda_check(cudaMalloc(&ctx.d_coeffs_y,      y_pixels * sizeof(int16_t)), "d_coeffs_y");
    cuda_check(cudaMalloc(&ctx.d_coeffs_cb,     c_pixels * sizeof(int16_t)), "d_coeffs_cb");
    cuda_check(cudaMalloc(&ctx.d_coeffs_cr,     c_pixels * sizeof(int16_t)), "d_coeffs_cr");

    // Interleaved slice layout: [num_slices * blocks_per_slice * 64]
    size_t slice_elems = (size_t)ctx.num_slices * ctx.blocks_per_slice * 64;
    cuda_check(cudaMalloc(&ctx.d_coeffs_slice, slice_elems * sizeof(int16_t)), "d_coeffs_slice");

    // Bitstream buffer: worst case = slice_elems * 2 bytes (entirely uncompressible coefficients)
    size_t bs_size = slice_elems * sizeof(int16_t) * 2 + ctx.num_slices * 32;
    cuda_check(cudaMalloc(&ctx.d_bitstream,     bs_size),                 "d_bitstream");
    cuda_check(cudaMalloc(&ctx.d_slice_offsets, (ctx.num_slices + 1) * sizeof(uint32_t)), "d_offsets");
    cuda_check(cudaMalloc(&ctx.d_bit_counts,     ctx.num_slices * sizeof(uint32_t)),      "d_bitcounts");

    // CUB temp storage: query size then allocate
    ctx.d_cub_temp     = nullptr;
    ctx.cub_temp_bytes = 0;
    // Pass nullptr first call to query required temp size — done via entropy.cu API.
    // For test we over-allocate 8 MB.
    ctx.cub_temp_bytes = 8 * 1024 * 1024;
    cuda_check(cudaMalloc(&ctx.d_cub_temp, ctx.cub_temp_bytes), "d_cub_temp");

    // Pinned output buffer
    size_t frame_buf_size = (size_t)width * height * 4; // generous upper bound
    ctx.h_frame_buf_size  = frame_buf_size;
    cuda_check(cudaMallocHost(&ctx.h_frame_buf, frame_buf_size), "h_frame_buf");
}

static void frame_ctx_destroy(ProResFrameCtx &ctx)
{
    cudaFree(ctx.d_y);            cudaFree(ctx.d_cb);          cudaFree(ctx.d_cr);
    cudaFree(ctx.d_coeffs_y);     cudaFree(ctx.d_coeffs_cb);   cudaFree(ctx.d_coeffs_cr);
    cudaFree(ctx.d_coeffs_slice);
    cudaFree(ctx.d_bitstream);    cudaFree(ctx.d_slice_offsets); cudaFree(ctx.d_bit_counts);
    cudaFree(ctx.d_cub_temp);
    cudaFreeHost(ctx.h_frame_buf);
    memset(&ctx, 0, sizeof(ctx));
}

// ---------------------------------------------------------------------------
// Load one V210 frame from file into device memory
// ---------------------------------------------------------------------------
// V210 frame bytes: each row is ceil(width/6)*16 bytes (16 bytes = 4 words of 3 packed 10-bit samples)
static size_t v210_row_bytes(int width) {
    return static_cast<size_t>((width + 5) / 6) * 16;
}

static size_t v210_frame_bytes(int width, int height) {
    return v210_row_bytes(width) * height;
}

static bool load_v210_frame(FILE *fp, int width, int height,
                             void **d_v210_out, size_t frame_bytes)
{
    // Allocate or reuse host staging
    static std::vector<uint8_t> s_host_buf;
    if (s_host_buf.size() < frame_bytes) s_host_buf.resize(frame_bytes);

    size_t n = fread(s_host_buf.data(), 1, frame_bytes, fp);
    if (n != frame_bytes) return false;

    // Upload to device
    static void *s_d_v210 = nullptr;
    static size_t s_d_v210_size = 0;
    if (s_d_v210_size < frame_bytes) {
        if (s_d_v210) cudaFree(s_d_v210);
        cuda_check(cudaMalloc(&s_d_v210, frame_bytes), "d_v210");
        s_d_v210_size = frame_bytes;
    }
    cuda_check(cudaMemcpy(s_d_v210, s_host_buf.data(), frame_bytes, cudaMemcpyHostToDevice),
               "upload v210");
    *d_v210_out = s_d_v210;
    return true;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char *argv[])
{
    if (argc < 6) {
        fprintf(stderr,
            "Usage: %s <input.v210> <width> <height> <profile 0-4> <output.mov>\n"
            "  profile: 0=Proxy 1=LT 2=Standard 3=HQ 4=4444\n", argv[0]);
        return 1;
    }

    const char *input_path  = argv[1];
    int          width      = atoi(argv[2]);
    int          height     = atoi(argv[3]);
    int          profile    = atoi(argv[4]);
    const char *output_path = argv[5];

    if (width <= 0 || height <= 0 || profile < 0 || profile > 4) {
        fprintf(stderr, "Invalid parameters\n");
        return 1;
    }

    // CUDA device sanity
    int dev_count = 0;
    cuda_check(cudaGetDeviceCount(&dev_count), "cudaGetDeviceCount");
    if (dev_count == 0) {
        fprintf(stderr, "No CUDA devices found\n");
        return 1;
    }
    cudaDeviceProp prop;
    cuda_check(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
    fprintf(stdout, "[Test] CUDA device 0: %s (sm_%d%d)\n",
            prop.name, prop.major, prop.minor);

    // Upload ProRes quantisation tables to constant memory
    prores_tables_upload();

    // Create encoder context
    ProResFrameCtx ctx = {};
    frame_ctx_create(ctx, width, height, profile);

    // Open input V210 file
    FILE *fp = fopen(input_path, "rb");
    if (!fp) {
        fprintf(stderr, "Cannot open input file: %s\n", input_path);
        frame_ctx_destroy(ctx);
        return 1;
    }

    // Open MOV muxer
    const uint32_t prores_fourccs[] = {
        0x6170636F, // 'apco' Proxy
        0x6170636C, // 'apcl' LT
        0x6170636E, // 'apcn' Standard
        0x61706368, // 'apch' HQ
        0x61703468, // 'ap4h' 4444
    };

    MovVideoTrackInfo vinfo = {};
    vinfo.width         = width;
    vinfo.height        = height;
    vinfo.timebase_num  = 1;
    vinfo.timebase_den  = 25; // 4K25p
    vinfo.prores_fourcc = prores_fourccs[profile];
    vinfo.color.color_primaries   = 1;
    vinfo.color.transfer_function = 1;
    vinfo.color.color_matrix      = 1;
    vinfo.color.has_hdr           = false;

    MovAudioTrackInfo ainfo = {};
    ainfo.channels    = 2;
    ainfo.sample_rate = 48000;

    // Convert output path to wide string
    std::wstring wout;
    {
        size_t len = strlen(output_path);
        wout.resize(len);
        for (size_t i = 0; i < len; i++) wout[i] = (wchar_t)output_path[i];
    }

    MovMuxer muxer;
    if (!muxer.open(wout, vinfo, ainfo)) {
        fprintf(stderr, "Failed to open output .mov: %s\n", output_path);
        fclose(fp);
        frame_ctx_destroy(ctx);
        return 1;
    }

    cudaStream_t stream;
    cuda_check(cudaStreamCreate(&stream), "cudaStreamCreate");

    const size_t frame_bytes = v210_frame_bytes(width, height);
    fprintf(stdout, "[Test] Frame size: %zu bytes (V210 %dx%d)\n", frame_bytes, width, height);

    size_t     out_size    = 0;
    int        frame_count = 0;
    void      *d_v210      = nullptr;

    // Encode frames until EOF
    while (load_v210_frame(fp, width, height, &d_v210, frame_bytes)) {
        cudaError_t e = prores_encode_frame(
            &ctx,
            reinterpret_cast<const uint32_t*>(d_v210),
            ctx.h_frame_buf,
            &out_size,
            stream,
            &COLOR_DESC_SDR_709);

        if (e != cudaSuccess) {
            fprintf(stderr, "prores_encode_frame failed frame %d: %s\n",
                    frame_count, cudaGetErrorString(e));
            break;
        }

        if (!muxer.write_video(ctx.h_frame_buf, out_size, (uint64_t)frame_count)) {
            fprintf(stderr, "write_video failed at frame %d\n", frame_count);
            break;
        }

        // Write SMPTE timecode for this frame (start at 10:00:00:00)
        {
            SmpteTimecode tc;
            tc.valid      = true;
            tc.drop_frame = false;
            const uint32_t fps = 25;
            const int64_t  fn  = 10 * 3600 * fps + frame_count; // start offset at 10hr
            tc.frames  = (uint8_t)(fn % fps);
            const int64_t ts = fn / fps;
            tc.seconds = (uint8_t)(ts % 60);
            tc.minutes = (uint8_t)((ts / 60) % 60);
            tc.hours   = (uint8_t)((ts / 3600) % 24);
            muxer.write_timecode(tc);
        }

        if (frame_count % 25 == 0) {
            fprintf(stdout, "[Test] Encoded frame %d  (%zu bytes, %.2f Mbps)\n",
                    frame_count, out_size,
                    (out_size * 8 * 25) / 1e6);
        }
        ++frame_count;
    }

    fclose(fp);
    muxer.close();

    // Clean up
    if (d_v210) cudaFree(d_v210);
    cudaStreamDestroy(stream);
    frame_ctx_destroy(ctx);

    fprintf(stdout, "[Test] Done — encoded %d frames → %s\n", frame_count, output_path);

    // -------------------------------------------------------------------
    // Validation: run ffprobe to check container integrity + timecode track
    // -------------------------------------------------------------------
    char cmd[512];
    snprintf(cmd, sizeof(cmd),
             "ffprobe -v quiet -print_format json -show_streams \"%s\"",
             output_path);
    fprintf(stdout, "[Test] Running: %s\n", cmd);
    int ret = system(cmd);
    if (ret == 0)
        fprintf(stdout, "[Test] ffprobe OK — output file is a valid container\n");
    else
        fprintf(stderr, "[Test] ffprobe reported errors (exit %d)\n", ret);

    // Additional: show timecode tag
    snprintf(cmd, sizeof(cmd),
             "ffprobe -v quiet -print_format json -show_streams "
             "-show_format \"%s\" 2>&1 | findstr /i timecode",
             output_path);
    fprintf(stdout, "[Test] Checking timecode: %s\n", cmd);
    system(cmd); // output is informational

    return (ret == 0 && frame_count > 0) ? 0 : 1;
}
