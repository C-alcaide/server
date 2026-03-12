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
//   gen_colorbars.exe [output_dir] [profile]
//   profile: proxy | lt | standard | hq (default) | 4444
//   Output: <output_dir>\colorbars_1080p25_<profile>.mov
//
// Timecode: starts at 01:00:00:00 (tmcd track in .mov)
//
// Apple ProRes target data rates for 1920x1080 (Apple ProRes White Paper):
//   proxy  :  45 Mb/s   lt      : 102 Mb/s
//   standard: 147 Mb/s  hq      : 220 Mb/s  (all @ 29.97 fps)
// NOTE: ProRes is constant-quality VBR.  Stated rates apply to typical broadcast
// content.  Flat synthetic frames compress smaller; film grain is applied here
// to drive AC energy toward representative broadcast complexity levels.
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
// Apple ProRes profile table.
//
// bits_per_mb is derived from the Apple ProRes White Paper target data rates
// for 1920x1080 @ 29.97 fps:  target_Mbps * 1e6 / (29.97 * 8160 MBs/frame).
//
// This constant is RESOLUTION- and FPS-INDEPENDENT: the number of MBs in a
// frame scales with resolution, so the correct bitrate results automatically
// for any resolution / frame-rate combination.
//
// Example: HQ bits_per_mb=899, 1920x1080 @ 25fps
//   8160 MBs * 899 bits/MB * 25 fps / 1e6 = 183.6 Mb/s  (Apple spec: 220*25/29.97=183.5)
// ---------------------------------------------------------------------------
struct ProResProfileInfo {
    const char *name;        // prores_ks option value
    uint32_t    fourcc;      // .mov codec tag
    int         bits_per_mb; // Apple quantisation budget per 16x16 MB
    int         grain_amp;   // film grain amplitude for this profile
    // Lower-quality profiles need more grain to reach Apple target bitrates.
    // Higher-quality profiles encode faster and look better with less grain.
    // Scale: 0 = no grain, 40 = ~1.5 IRE (4.6% of 10-bit swing)
};

static const ProResProfileInfo PROFILE_INFO[] = {
    //   name         fourcc      bits_per_mb  grain_amp   apple_target
    { "proxy",    0x6170636Fu, 184,  40 },  // apco  45 Mb/s — heavy grain to drive bitrate
    { "lt",       0x61706373u, 417,  30 },  // apcs 102 Mb/s
    { "standard", 0x6170636Eu, 601,  15 },  // apcn 147 Mb/s — moderate; faster encoder
    { "hq",       0x61706368u, 899,   8 },  // apch 220 Mb/s — clean ref; fine quant dominates
    { "4444",     0x61703468u, 1350,  6 },  // ap4h ~330 Mb/s
};
static const int N_PROFILES = 5;

// ---------------------------------------------------------------------------
// Film grain — fast deterministic per-pixel noise (xorshift hash).
// Produces the same pattern every run (reproducible), varies per frame.
// Drives AC DCT energy so the encoded file has complexity representative of
// typical broadcast content, rather than the near-zero AC of flat test bars.
// ---------------------------------------------------------------------------
static inline int grain_pixel(int x, int y, int frame_idx, int amplitude)
{
    uint32_t h = (uint32_t)x;
    h ^= (uint32_t)y          * 2654435761u;
    h ^= (uint32_t)frame_idx  * 1234567891u;
    h ^= h >> 16; h *= 0x45d9f3bu; h ^= h >> 16;
    return ((int)(h & 0xFF) - 128) * amplitude / 128;
}

static inline uint16_t apply_grain(uint16_t base, int x, int y, int frame_idx, int amplitude)
{
    if (amplitude == 0) return base;
    int v = (int)base + grain_pixel(x, y, frame_idx, amplitude);
    return (uint16_t)(v < 64 ? 64 : v > 940 ? 940 : v);
}

// ---------------------------------------------------------------------------
// Fill one YUV422P10LE frame with the test pattern.
//   frame        — pre-allocated AVFrame (av_frame_get_buffer called by caller)
//   frame_index  — 0-based frame number (drives animation)
//   fps          — frames per second (25)
// ---------------------------------------------------------------------------
static void fill_frame(AVFrame *frame, int frame_index, int fps, int grain_amplitude)
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

                Y_row[x] = apply_grain(in_ball ? Y_WHITE : BARS[bar].y, x, y, frame_index, grain_amplitude);

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
                Y_row[x] = apply_grain((x < sweep_x) ? Y_WHITE : Y_BLACK, x, y, frame_index, grain_amplitude);
                if ((x & 1) == 0) {
                    Cb_row[x >> 1] = C_NEUTRAL;
                    Cr_row[x >> 1] = C_NEUTRAL;
                }
            }
        } else {
            // ── Full luma ramp  Y=64 → Y=940 ──────────────────────────────────
            for (int x = 0; x < W; ++x) {
                const uint16_t ramp_y = (uint16_t)(Y_BLACK + (uint32_t)(x) * (Y_WHITE - Y_BLACK) / (uint32_t)(W - 1));
                Y_row[x] = apply_grain(ramp_y, x, y, frame_index, grain_amplitude);
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
    const std::string out_dir  = (argc > 1) ? argv[1] : ".";
    const char *profile_arg      = (argc > 2) ? argv[2] : "hq";

    // Resolve profile (default to hq)
    const ProResProfileInfo *prof = &PROFILE_INFO[3];
    for (int i = 0; i < N_PROFILES; ++i) {
        if (std::string(PROFILE_INFO[i].name) == profile_arg) {
            prof = &PROFILE_INFO[i];
            break;
        }
    }

    const std::string out_path = out_dir + "\\colorbars_1080p25_" + prof->name + ".mov";
    const std::wstring wout_path(out_path.begin(), out_path.end());

    const int WIDTH      = 1920;
    const int HEIGHT     = 1080;
    const int FPS        = 25;
    const int NUM_FRAMES = 3 * FPS;   // 3 seconds — enough for visual inspection
    // Film grain amplitude varies by profile:
    // Lower-quality profiles (proxy/lt) need more grain to drive toward Apple target
    // bitrates; higher-quality profiles (standard/hq) encode clean bars faster and
    // bits_per_mb alone provides the spec-appropriate quality floor.
    const int GRAIN_AMP = prof->grain_amp;

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
    av_opt_set(enc->priv_data, "profile", prof->name, 0);
    // Set quantisation budget from Apple's ProRes White Paper target bitrate.
    // For typical broadcast content this produces approximately the stated Mb/s.
    av_opt_set_int(enc->priv_data, "bits_per_mb", prof->bits_per_mb, 0);
    av_opt_set(enc->priv_data, "vendor", "apl0", 0); // Apple vendor tag

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
    vi.prores_fourcc = prof->fourcc;  // set per-profile (apco/apcs/apcn/apch/ap4h)
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

        fill_frame(frame, f, FPS, GRAIN_AMP);

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

        if (f % FPS == 0 || f == NUM_FRAMES - 1)
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
