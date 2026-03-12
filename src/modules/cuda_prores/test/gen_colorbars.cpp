// gen_colorbars.cpp
// Generates a visually rich 1920×1080 25 fps ProRes 422 HQ .mov file
// for manual quality inspection.
//
// Frame layout
// ─────────────────────────────────────────────────────────────────────────────
//
//  ┌──────────────────────────────────────────────────────────────────────────┐
//  │  EBU 75% colour bars  (White │ Yellow │ Cyan │ Green │ Magenta │ Red │ Blue)  rows 0–719  │
//  │  + rotating white circle (one rotation/sec) as motion indicator              │
//  ├──────────────────────────────────────────────────────────────────────────┤
//  │  Left-to-right sweep bar (resets every second, white on black)          rows 720–899  │
//  │  — use to confirm frame rate and playback speed                              │
//  ├──────────────────────────────────────────────────────────────────────────┤
//  │  Full luma ramp  Y=64 (black) → Y=940 (white), Cb=Cr=512               rows 900–1079 │
//  │  — use to check quantisation / banding                                       │
//  └──────────────────────────────────────────────────────────────────────────┘
//
// Build:
//   Part of the gen_colorbars target in tests_standalone/CMakeLists.txt
// Run:
//   gen_colorbars.exe [output_dir]
//   Output: <output_dir>\colorbars_1080p25_hq.mov
//
// Timecode: starts at 01:00:00:00 (tmcd track in .mov)
//
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#define _USE_MATH_DEFINES
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "../muxer/mov_muxer.h"
#include "../timecode.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/frame.h>
#include <libavutil/opt.h>
}

// ---------------------------------------------------------------------------
// Colour values for EBU 75% colour bars in 10-bit limited-range BT.709 YCbCr
// Formula:  Y  = 64 + 876 * (0.2126R + 0.7152G + 0.0722B)
//           Cb = 512 + 896 * (B – Y_lin) / 1.8556
//           Cr = 512 + 896 * (R – Y_lin) / 1.5748
//           where R,G,B ∈ {0.0, 0.75}
// ---------------------------------------------------------------------------
struct YCbCr10 {
    uint16_t y, cb, cr;
};

static constexpr YCbCr10 BARS[7] = {
    {721, 512, 512},   // 75% White
    {674, 176, 543},   // 75% Yellow
    {581, 589, 176},   // 75% Cyan
    {534, 253, 207},   // 75% Green
    {251, 771, 817},   // 75% Magenta
    {204, 435, 848},   // 75% Red
    {111, 848, 481},   // 75% Blue
};

// 100% White and Black for sweep/ramp sections
static constexpr uint16_t Y_WHITE = 940;
static constexpr uint16_t Y_BLACK = 64;
static constexpr uint16_t C_NEUTRAL = 512;

// ---------------------------------------------------------------------------
// Fill one YUV422P10LE frame with the test pattern.
//   frame        — pre-allocated AVFrame (av_frame_get_buffer called by caller)
//   frame_index  — 0-based frame number (drives animation)
//   fps          — frames per second (25)
// ---------------------------------------------------------------------------
static void fill_frame(AVFrame *frame, int frame_index, int fps)
{
    const int W = frame->width;
    const int H = frame->height;

    // --- Section boundaries ---
    const int top_end = (H * 2) / 3;   // row 720 for 1080 → colour bars
    const int mid_end = (H * 5) / 6;   // row 900 for 1080 → sweep bar

    // --- Rotating ball (one revolution per second) ---
    const double angle  = frame_index * 2.0 * M_PI / fps;
    const double radius = H * 0.18;    // orbit radius (~194 px for 1080)
    const int ball_cx   = W / 2 + (int)(radius * std::cos(angle));
    const int ball_cy   = top_end / 2 + (int)(radius * std::sin(angle));
    const int ball_r    = (int)(H * 0.04);  // ball radius (~43 px for 1080)
    const int ball_r2   = ball_r * ball_r;

    // --- Sweep bar progress within the current second ---
    // Position advances from 0 → W across one second, then resets.
    const int sweep_x = ((frame_index % fps) * W) / fps;

    for (int y = 0; y < H; ++y) {
        uint16_t *Y_row  = reinterpret_cast<uint16_t *>(frame->data[0] + y * frame->linesize[0]);
        uint16_t *Cb_row = reinterpret_cast<uint16_t *>(frame->data[1] + y * frame->linesize[1]);
        uint16_t *Cr_row = reinterpret_cast<uint16_t *>(frame->data[2] + y * frame->linesize[2]);

        if (y < top_end) {
            // ── Colour bars + rotating ball overlay ──────────────────────────
            const int dy = y - ball_cy;
            for (int x = 0; x < W; ++x) {
                const int  bar = (x * 7) / W;
                const int  dx  = x - ball_cx;
                const bool in_ball = (dx * dx + dy * dy) <= ball_r2;

                Y_row[x] = in_ball ? Y_WHITE : BARS[bar].y;

                // Chroma samples are at even x positions (YUV422 = 1 Cb/Cr per 2 Y)
                if ((x & 1) == 0) {
                    const int cx = x >> 1;
                    // Check both luma positions that share this chroma sample
                    const int dx1 = (x + 1) - ball_cx;
                    const bool in_ball_r = (dx1 * dx1 + dy * dy) <= ball_r2;
                    if (in_ball || in_ball_r) {
                        Cb_row[cx] = C_NEUTRAL;
                        Cr_row[cx] = C_NEUTRAL;
                    } else {
                        // Average chroma when the pair straddles a bar boundary
                        const int bar_r = ((x + 1) * 7) / W;
                        if (bar == bar_r) {
                            Cb_row[cx] = BARS[bar].cb;
                            Cr_row[cx] = BARS[bar].cr;
                        } else {
                            Cb_row[cx] = (uint16_t)((BARS[bar].cb + BARS[bar_r].cb) / 2);
                            Cr_row[cx] = (uint16_t)((BARS[bar].cr + BARS[bar_r].cr) / 2);
                        }
                    }
                }
            }
        } else if (y < mid_end) {
            // ── Sweep bar (white left of sweep_x, black right) ───────────────
            for (int x = 0; x < W; ++x) {
                Y_row[x] = (x < sweep_x) ? Y_WHITE : Y_BLACK;
                if ((x & 1) == 0) {
                    Cb_row[x >> 1] = C_NEUTRAL;
                    Cr_row[x >> 1] = C_NEUTRAL;
                }
            }
        } else {
            // ── Full luma ramp  Y=64 → Y=940 ──────────────────────────────────
            for (int x = 0; x < W; ++x) {
                Y_row[x] = (uint16_t)(Y_BLACK + (uint32_t)(x) * (Y_WHITE - Y_BLACK) / (uint32_t)(W - 1));
                if ((x & 1) == 0) {
                    Cb_row[x >> 1] = C_NEUTRAL;
                    Cr_row[x >> 1] = C_NEUTRAL;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char **argv)
{
    const std::string out_dir = (argc > 1) ? argv[1] : ".";
    const std::string out_path = out_dir + "\\colorbars_1080p25_hq.mov";
    const std::wstring wout_path(out_path.begin(), out_path.end());

    const int WIDTH       = 1920;
    const int HEIGHT      = 1080;
    const int FPS         = 25;
    const int NUM_FRAMES  = 5 * FPS;  // 5 seconds

    // ── Open the encoder ────────────────────────────────────────────────────
    const AVCodec *codec = avcodec_find_encoder_by_name("prores_ks");
    if (!codec) {
        fprintf(stderr, "[gen_colorbars] prores_ks encoder not found\n");
        return 1;
    }
    AVCodecContext *enc = avcodec_alloc_context3(codec);
    enc->width     = WIDTH;
    enc->height    = HEIGHT;
    enc->pix_fmt   = AV_PIX_FMT_YUV422P10LE;
    AVRational tb  = {1, FPS};
    enc->time_base = tb;
    av_opt_set(enc->priv_data, "profile", "hq", 0);  // ProRes 422 HQ → 'apch'
    av_opt_set(enc->priv_data, "vendor",  "apl0", 0); // Apple vendor tag (cosmetic)

    if (avcodec_open2(enc, codec, nullptr) < 0) {
        fprintf(stderr, "[gen_colorbars] avcodec_open2 failed\n");
        avcodec_free_context(&enc);
        return 1;
    }

    // ── Open the muxer ──────────────────────────────────────────────────────
    MovVideoTrackInfo vi{};
    vi.width         = WIDTH;
    vi.height        = HEIGHT;
    vi.timebase_num  = 1;
    vi.timebase_den  = (uint32_t)FPS;
    vi.prores_fourcc = 0x61706368u;  // 'apch' — ProRes 422 HQ
    vi.color.color_primaries   = 1;  // BT.709
    vi.color.transfer_function = 1;  // BT.709
    vi.color.color_matrix      = 1;  // BT.709

    MovAudioTrackInfo ai{};
    ai.channels    = 2;
    ai.sample_rate = 48000;

    MovMuxer muxer;
    if (!muxer.open(wout_path, vi, ai)) {
        fprintf(stderr, "[gen_colorbars] Failed to open muxer: %s\n", out_path.c_str());
        avcodec_free_context(&enc);
        return 1;
    }

    // ── Allocate reusable frame ─────────────────────────────────────────────
    AVFrame *frame = av_frame_alloc();
    frame->format = enc->pix_fmt;
    frame->width  = WIDTH;
    frame->height = HEIGHT;
    av_frame_get_buffer(frame, 0);

    AVPacket *pkt = av_packet_alloc();

    const int64_t start_frame_count = 1 * 3600 * FPS;  // 01:00:00:00

    for (int f = 0; f < NUM_FRAMES; ++f) {
        av_frame_make_writable(frame);
        frame->pts = f;

        fill_frame(frame, f, FPS);

        if (avcodec_send_frame(enc, frame) < 0) {
            fprintf(stderr, "[gen_colorbars] avcodec_send_frame failed at frame %d\n", f);
            break;
        }
        if (avcodec_receive_packet(enc, pkt) < 0) {
            fprintf(stderr, "[gen_colorbars] avcodec_receive_packet failed at frame %d\n", f);
            break;
        }

        if (!muxer.write_video(pkt->data, (size_t)pkt->size, (uint64_t)f)) {
            fprintf(stderr, "[gen_colorbars] write_video failed at frame %d\n", f);
            av_packet_unref(pkt);
            break;
        }
        av_packet_unref(pkt);

        // Write SMPTE timecode
        SmpteTimecode tc{};
        tc.valid      = true;
        tc.drop_frame = false;
        const int64_t fn = start_frame_count + f;
        tc.frames  = (uint8_t)(fn % FPS);
        const int64_t ts = fn / FPS;
        tc.seconds = (uint8_t)(ts % 60);
        tc.minutes = (uint8_t)((ts / 60) % 60);
        tc.hours   = (uint8_t)((ts / 3600) % 24);
        muxer.write_timecode(tc);

        if (f % FPS == 0)
            fprintf(stdout, "[gen_colorbars] Frame %d/%d (%02d:%02d:%02d:%02d)\n",
                    f, NUM_FRAMES, tc.hours, tc.minutes, tc.seconds, tc.frames);
    }

    av_packet_free(&pkt);
    av_frame_free(&frame);
    avcodec_free_context(&enc);

    if (!muxer.close()) {
        fprintf(stderr, "[gen_colorbars] muxer close() failed\n");
        return 1;
    }

    fprintf(stdout, "[gen_colorbars] Done  →  %s\n", out_path.c_str());
    return 0;
}
