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
 * ProRes format reference: Apple Inc. "ProRes RAW White Paper" (public documentation).
 */

// prores_bypass_consumer.cpp
// ProRes recording direct from DeckLink SDI capture — bypasses the CasparCG
// GPU mixer entirely.  DecklinkCapture delivers raw V210 device buffers that
// are encoded straight to ProRes 422 (or 422 HQ) without any BGRA conversion.
//
// Frame flow
// ─────────────────────────────────────────────────────────────────────────────
// DeckLink driver thread → VideoInputFrameArrived (inside DecklinkCapture)
//   → synchronous H2D memcpy (copy_stream synced before callback returns)
//   → CaptureToken queued (d_vram guaranteed valid, host buffer already released)
// Encode thread:
//   1. cudaStreamSynchronize(token.copy_stream)   — no-op guard (copy already done)
//   Progressive:
//   2. prores_encode_frame(ctx, d_v210, ...)       — V210 → ProRes on encode_stream_
//   Interlaced:
//   2. launch_v210_unpack_field(d_v210, field_a_planes, parity_a) — extract field A
//   3. launch_v210_unpack_field(d_v210, ctx.d_y/cb/cr, parity_b) — extract field B
//   4. prores_encode_from_yuv_fields_422(...)
//   5. muxer.write_video() + write_timecode()
// ─────────────────────────────────────────────────────────────────────────────

#include "prores_bypass_consumer.h"

#include <common/except.h>
#include <common/log.h>
#include <common/param.h>
#include <common/memory.h>

#include <core/consumer/frame_consumer.h>
#include <core/frame/frame.h>
#include <core/video_format.h>
#include <core/monitor/monitor.h>
#include <core/consumer/channel_info.h>

#include <common/diagnostics/graph.h>

#include <boost/property_tree/ptree.hpp>
#include <boost/algorithm/string.hpp>

#include <cuda_runtime.h>

#include "../cuda/cuda_prores_frame.h"
#include "../cuda/cuda_prores_tables.cuh"
#include "../muxer/mov_muxer.h"
#include "../muxer/mxf_muxer.h"
#include "../timecode.h"

// DeckLink capture front-end (wraps IDeckLinkInputCallback + VRAM ring)
#include "../input/decklink_capture.h"

// get_decklink_video_format() and related helpers
#include <util/util.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

namespace caspar { namespace cuda_prores {

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------
static void cuda_check_bypass(cudaError_t e, const char *ctx)
{
    if (e != cudaSuccess) {
        CASPAR_LOG(error) << L"[cuda_prores_bypass] CUDA error in " << ctx << ": "
                          << cudaGetErrorString(e);
        throw std::runtime_error(std::string("[cuda_prores_bypass] CUDA error: ") + ctx);
    }
}

// Returns the V210 row stride for a given width.
static size_t v210_row_bytes_bp(int w) {
    return static_cast<size_t>((w + 5) / 6) * 16;
}
static size_t v210_frame_bytes_bp(int w, int h) {
    return v210_row_bytes_bp(w) * h;
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------
struct bypass_config {
    std::wstring output_path;
    std::wstring filename_pattern;
    int          profile       = 3;     // 3=HQ, others=422
    int          hdr_mode      = 0;     // 0=SDR_709
    uint16_t     hdr_max_cll   = 1000;
    uint16_t     hdr_max_fall  = 400;
    bool         use_mxf       = false;
    int          device_index  = 1;     // DeckLink device (1-based, per SDK convention)
    int          cuda_device   = 0;     // CUDA device index
    int          slices_per_row = 4;    // horizontal slices per MB row (1/2/4/8)
    int          q_scale        = 8;    // quantization scale [1..31]; 1=best quality
};

// ---------------------------------------------------------------------------
// BypassJob — one entry in the encode queue (one raw V210 frame)
// ---------------------------------------------------------------------------
struct BypassJob {
    CaptureToken           token;
    int64_t                frame_number;
    std::vector<int32_t>   audio;          // interleaved PCM32 samples captured from DeckLink
    int                    audio_channels = 0;
    std::promise<bool>     done;
};

// ---------------------------------------------------------------------------
// Buffer management (shared with prores_consumer.cpp via copy-paste to avoid
// header coupling — both consumers live in the same module, same translation
// unit is fine).
// ---------------------------------------------------------------------------
static void alloc_bypass_ctx(ProResFrameCtx &ctx,
                              int width, int height, int profile,
                              int slices_per_row,
                              bool is_interlaced)
{
    ctx.width          = width;
    ctx.height         = height;
    ctx.profile        = profile;
    ctx.is_4444        = false;
    ctx.has_alpha      = false;
    ctx.q_scale        = 8;

    // slices_per_row parameter carries mbs_per_slice (power of 2: 1,2,4,8);
    // actual slices_per_row is derived as (width_mbs / mbs_per_slice).
    ctx.mbs_per_slice  = slices_per_row;  // user SLICES param = MBs per slice
    ctx.slices_per_row = (width / 16) / ctx.mbs_per_slice;
    if ((width / 16) % ctx.mbs_per_slice != 0) ctx.slices_per_row++; // partial last slice
    ctx.num_slices     = ctx.slices_per_row * ((height + 15) / 16);
    ctx.blocks_per_slice = ctx.mbs_per_slice * 8; // 422

    const int y_px = width * height;
    const int c_px = (width / 2) * height;

    cuda_check_bypass(cudaMalloc(&ctx.d_y,  y_px * sizeof(int16_t)), "d_y");
    cuda_check_bypass(cudaMalloc(&ctx.d_cb, c_px * sizeof(int16_t)), "d_cb");
    cuda_check_bypass(cudaMalloc(&ctx.d_cr, c_px * sizeof(int16_t)), "d_cr");
    cuda_check_bypass(cudaMalloc(&ctx.d_coeffs_y,  y_px * sizeof(int16_t)), "d_coeffs_y");
    cuda_check_bypass(cudaMalloc(&ctx.d_coeffs_cb, c_px * sizeof(int16_t)), "d_coeffs_cb");
    cuda_check_bypass(cudaMalloc(&ctx.d_coeffs_cr, c_px * sizeof(int16_t)), "d_coeffs_cr");

    const size_t slice_elems = (size_t)ctx.num_slices * ctx.blocks_per_slice * 64;
    cuda_check_bypass(cudaMalloc(&ctx.d_coeffs_slice, slice_elems * sizeof(int16_t)), "d_coeffs_slice");

    const size_t bs_size = slice_elems * sizeof(int16_t) * 2 + ctx.num_slices * 64;
    cuda_check_bypass(cudaMalloc(&ctx.d_bitstream,     bs_size),                                   "d_bitstream");
    cuda_check_bypass(cudaMalloc(&ctx.d_slice_offsets, (ctx.num_slices + 1) * sizeof(uint32_t)), "d_slice_offsets");
    cuda_check_bypass(cudaMalloc(&ctx.d_slice_sizes,    ctx.num_slices      * sizeof(uint32_t)), "d_slice_sizes");
    cuda_check_bypass(cudaMalloc(&ctx.d_bit_counts, ctx.num_slices * 3 * sizeof(uint32_t)), "d_bit_counts");

    ctx.cub_temp_bytes = 8 * 1024 * 1024;
    cuda_check_bypass(cudaMalloc(&ctx.d_cub_temp, ctx.cub_temp_bytes), "d_cub_temp");

    // h_frame_buf must hold two picture headers + fields when interlaced.
    const size_t buf_height  = is_interlaced ? (size_t)height * 2 : (size_t)height;
    const size_t frame_buf_sz = (size_t)width * buf_height * 8;
    ctx.h_frame_buf_size = frame_buf_sz;
    cuda_check_bypass(cudaMallocHost(&ctx.h_frame_buf, frame_buf_sz), "h_frame_buf");

    ctx.d_alpha        = nullptr;
    ctx.d_coeffs_alpha = nullptr;
}

static void free_bypass_ctx(ProResFrameCtx &ctx)
{
    cudaFree(ctx.d_y);   cudaFree(ctx.d_cb);  cudaFree(ctx.d_cr);
    cudaFree(ctx.d_coeffs_y); cudaFree(ctx.d_coeffs_cb); cudaFree(ctx.d_coeffs_cr);
    cudaFree(ctx.d_coeffs_slice);
    cudaFree(ctx.d_bitstream);
    cudaFree(ctx.d_slice_offsets);
    cudaFree(ctx.d_slice_sizes);
    cudaFree(ctx.d_bit_counts);
    cudaFree(ctx.d_cub_temp);
    cudaFreeHost(ctx.h_frame_buf);
    std::memset(&ctx, 0, sizeof(ctx));
}

// Returns true if the channel format is top-field-first.
// PAL and NTSC SD interlaced are BFF; all 1080i formats are TFF.
static bool bypass_format_is_tff(const core::video_format_desc &fmt)
{
    using vf = core::video_format;
    if (fmt.format == vf::pal || fmt.format == vf::ntsc)
        return false;
    return true;
}

// ---------------------------------------------------------------------------
// Consumer implementation
// ---------------------------------------------------------------------------
class prores_bypass_consumer_impl : public core::frame_consumer {
public:
    prores_bypass_consumer_impl(bypass_config cfg, int consumer_index)
        : cfg_(std::move(cfg))
        , index_(consumer_index)
    {        graph_->set_color("encode-time",   diagnostics::color(0.1f, 0.7f, 0.2f));
        graph_->set_color("queue-depth",   diagnostics::color(0.8f, 0.6f, 0.2f));
        graph_->set_color("dropped-frame", diagnostics::color(0.9f, 0.2f, 0.1f));
        graph_->set_color("encode-error",  diagnostics::color(0.9f, 0.3f, 0.7f));
        graph_->set_text(print());
        diagnostics::register_graph(graph_);
        CASPAR_LOG(info) << L"[cuda_prores_bypass] Created bypass consumer #" << index_
                         << L" device=" << cfg_.device_index
                         << L" profile=" << cfg_.profile
                         << L" qscale=" << cfg_.q_scale
                         << L" slices=" << cfg_.slices_per_row
                         << L" " << (cfg_.use_mxf ? L"MXF" : L"MOV")
                         << L" → " << cfg_.output_path;
    }

    ~prores_bypass_consumer_impl() override
    {
        stop();
    }

    // ── frame_consumer interface ──────────────────────────────────────────

    void initialize(const core::video_format_desc& format_desc,
                    const core::channel_info& /*channel_info*/,
                    int /*port_index*/) override
    {
        format_desc_ = format_desc; // kept for hz (diagnostics timing) and initial BMD mode hint

        prores_tables_upload();

        // Select CUDA device and create encode stream
        cuda_check_bypass(cudaSetDevice(cfg_.cuda_device), "cudaSetDevice");
        cuda_check_bypass(cudaStreamCreateWithFlags(&encode_stream_, cudaStreamNonBlocking),
                          "cudaStreamCreateWithFlags");

        // ProRes fourcc from profile — stored for use at first-frame init
        static const uint32_t PRORES_FOURCC[] = {
            0x6170636Fu, // 'apco' Proxy
            0x61706373u, // 'apcs' LT
            0x6170636Eu, // 'apcn' Standard
            0x61706368u, // 'apch' HQ
            0x61703468u, // 'ap4h' 4444
            0x61703478u, // 'ap4x' 4444 XQ
        };
        pending_fourcc_ = PRORES_FOURCC[std::min(cfg_.profile, 5)];

        // Derive output path — generate a timestamped default when no pattern is specified
        std::wstring filename;
        if (!cfg_.filename_pattern.empty()) {
            filename = cfg_.filename_pattern;
        } else {
            const wchar_t *ext = cfg_.use_mxf ? L".mxf" : L".mov";
            std::time_t now = std::time(nullptr);
            struct tm t;
#ifdef _MSC_VER
            localtime_s(&t, &now);
#else
            localtime_r(&now, &t);
#endif
            wchar_t buf[32];
            std::swprintf(buf, 32, L"prores_%04d%02d%02d_%02d%02d%02d",
                          t.tm_year + 1900, t.tm_mon + 1, t.tm_mday,
                          t.tm_hour, t.tm_min, t.tm_sec);
            filename = std::wstring(buf) + ext;
        }
        pending_full_path_ = cfg_.output_path + L"\\" + filename;

        // Ensure output directory exists
        std::error_code ec;
        std::filesystem::create_directories(cfg_.output_path, ec);
        if (ec)
            CASPAR_LOG(warning) << L"[cuda_prores_bypass] Could not create directory: " << cfg_.output_path << L" (" << ec.message().c_str() << L")";

        // HDR color info — stored for use at first-frame muxer open
        pending_color_info_ = {};
        switch (cfg_.hdr_mode) {
            default:
                pending_color_info_ = { 1, 1, 1, false };
                break;
            case 1:
                pending_color_info_ = { 9, 14, 9, false };
                break;
            case 2: {
                pending_color_info_.color_primaries   = 9;
                pending_color_info_.transfer_function = 16;
                pending_color_info_.color_matrix      = 9;
                pending_color_info_.has_hdr           = true;
                pending_color_info_.mdcv_primaries_x[0] = 17000;
                pending_color_info_.mdcv_primaries_y[0] = 34000;
                pending_color_info_.mdcv_primaries_x[1] =  7500;
                pending_color_info_.mdcv_primaries_y[1] =  2500;
                pending_color_info_.mdcv_primaries_x[2] = 35400;
                pending_color_info_.mdcv_primaries_y[2] = 14600;
                pending_color_info_.mdcv_white_x = 15635;
                pending_color_info_.mdcv_white_y = 16450;
                pending_color_info_.mdcv_max_lum = (uint32_t)cfg_.hdr_max_cll  * 10000;
                pending_color_info_.mdcv_min_lum = 50;
                pending_color_info_.clli_max_cll  = cfg_.hdr_max_cll;
                pending_color_info_.clli_max_fall = cfg_.hdr_max_fall;
                break;
            }
        }

        // Start encode thread before DeckLink streaming begins
        encode_thread_ = std::thread(&prores_bypass_consumer_impl::encode_loop, this);

        // Use the CasparCG channel format as the initial DeckLink mode hint.
        // bmdVideoInputEnableFormatDetection (set in DecklinkCapture::start) will
        // override this with the actual SDI input format before any frames arrive,
        // so the encoder and muxer are initialized with the real capture dimensions.
        BMDDisplayMode bmd_mode = caspar::decklink::get_decklink_video_format(format_desc_.format);
        if (bmd_mode == (BMDDisplayMode)ULONG_MAX)
            bmd_mode = bmdModeHD1080i50; // safe fallback; format detection overrides anyway

        capture_ = std::make_unique<DecklinkCapture>(
            cfg_.device_index, bmd_mode, 0 /*no sync group*/,
            [this](CaptureToken token, AudioCapturePacket audio) {
                // Called on DeckLink driver thread — must return fast.
                BypassJob job;
                job.token        = std::move(token);
                job.frame_number = frame_number_++;
                // Copy audio PCM32 samples from DeckLink callback
                if (audio.samples && audio.sample_count > 0 && audio.channel_count > 0) {
                    const int total = audio.sample_count * audio.channel_count;
                    job.audio.assign(audio.samples, audio.samples + total);
                    job.audio_channels = audio.channel_count;
                }
                enqueue_bypass_job(std::move(job));
            });

        if (!capture_->start()) {
            CASPAR_LOG(error) << L"[cuda_prores_bypass] DecklinkCapture::start() failed";
            capture_.reset();
        } else {
            CASPAR_LOG(info) << L"[cuda_prores_bypass] DeckLink capture started on device "
                             << cfg_.device_index;
        }
    }

    // send() is called by the CasparCG channel output loop on every mixer frame.
    // Bypass consumer ignores mixer frames entirely — recording is driven by
    // DeckLink VideoInputFrameArrived callbacks.
    std::future<bool> send(const core::video_field, core::const_frame) override
    {
        return caspar::make_ready_future(static_cast<bool>(running_));
    }

    std::future<bool> call(const std::vector<std::wstring>& params) override
    {
        if (!params.empty()) {
            auto cmd = boost::to_upper_copy(params[0]);
            if (cmd == L"STOP") {
                stop();
                return caspar::make_ready_future(true);
            }
        }
        return caspar::make_ready_future(false);
    }

    core::monitor::state state() const override
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        return state_;
    }

    std::wstring name()  const override { return L"cuda_prores_bypass"; }
    int          index() const override { return index_; }
    std::wstring print() const override
    {
        return L"cuda_prores_bypass[" + std::to_wstring(index_) + L"|dev"
               + std::to_wstring(cfg_.device_index) + L"]";
    }

private:

    void enqueue_bypass_job(BypassJob job)
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        if (frame_queue_.size() >= kMaxQueueDepth) {
            // Drop oldest — release its DeckLink frame reference
            auto &front = frame_queue_.front();
            if (front.token.release_fn) front.token.release_fn();
            front.done.set_value(false);
            frame_queue_.pop();
            graph_->set_tag(diagnostics::tag_severity::WARNING, "dropped-frame");
            CASPAR_LOG(warning) << L"[cuda_prores_bypass] Queue overflow \u2014 dropped oldest frame";
        }
        graph_->set_value("queue-depth", static_cast<double>(frame_queue_.size()) / kMaxQueueDepth);
        frame_queue_.push(std::move(job));
        queue_cv_.notify_one();
    }

    // -------------------------------------------------------------------------
    // init_encoder_and_muxer — called lazily on first frame with actual capture
    // dimensions from the CaptureToken (not from the CasparCG channel format).
    // -------------------------------------------------------------------------
    void init_encoder_and_muxer(uint32_t w, uint32_t h,
                                 bool interlaced, bool tff,
                                 int tb_num, int tb_den)
    {
        // Free any resources from a previous init
        if (frame_ctx_.d_y) { free_bypass_ctx(frame_ctx_); }
        if (d_field_a_y_)  { cudaFree(d_field_a_y_);  d_field_a_y_  = nullptr; }
        if (d_field_a_cb_) { cudaFree(d_field_a_cb_); d_field_a_cb_ = nullptr; }
        if (d_field_a_cr_) { cudaFree(d_field_a_cr_); d_field_a_cr_ = nullptr; }
        if (mov_muxer_) { mov_muxer_->close(); mov_muxer_.reset(); }
        if (mxf_muxer_) { mxf_muxer_->close(); mxf_muxer_.reset(); }

        if (w == 0 || h == 0) {
            CASPAR_LOG(error) << L"[cuda_prores_bypass] Invalid capture dimensions: " << w << L"x" << h;
            return;
        }
        if (w % 16 != 0) {
            CASPAR_LOG(error) << L"[cuda_prores_bypass] Capture width " << w
                              << L" is not a multiple of 16 — cannot encode ProRes";
            return;
        }

        is_interlaced_ = interlaced;
        is_tff_        = tff;

        const int enc_height = interlaced ? static_cast<int>(h) / 2 : static_cast<int>(h);

        alloc_bypass_ctx(frame_ctx_, static_cast<int>(w), enc_height,
                         cfg_.profile, cfg_.slices_per_row, interlaced);
        frame_ctx_.q_scale = cfg_.q_scale;

        if (interlaced) {
            frame_ctx_.is_interlaced = true;
            frame_ctx_.is_tff        = tff;
            frame_ctx_.field_height  = enc_height;
            frame_ctx_.height        = static_cast<int>(h); // full height for ProRes frame header
            const int fh = enc_height;
            cuda_check_bypass(cudaMalloc(&d_field_a_y_,  (size_t)w       * fh * sizeof(int16_t)), "d_field_a_y_");
            cuda_check_bypass(cudaMalloc(&d_field_a_cb_, (size_t)(w / 2) * fh * sizeof(int16_t)), "d_field_a_cb_");
            cuda_check_bypass(cudaMalloc(&d_field_a_cr_, (size_t)(w / 2) * fh * sizeof(int16_t)), "d_field_a_cr_");
        } else {
            frame_ctx_.field_height = static_cast<int>(h);
        }

        // Open muxer with actual capture dimensions + frame rate
        if (cfg_.use_mxf) {
            MxfVideoTrackInfo vi{};
            vi.width         = static_cast<int>(w);
            vi.height        = static_cast<int>(h);
            vi.frame_rate    = { tb_num, tb_den };
            vi.prores_fourcc = pending_fourcc_;
            vi.color = (cfg_.hdr_mode == 1) ? MXF_COLOR_HDR_HLG
                     : (cfg_.hdr_mode == 2) ? MXF_COLOR_HDR_PQ
                     : MXF_COLOR_SDR_709;
            MxfAudioTrackInfo ai{};
            ai.channels    = 16;
            ai.sample_rate = 48000;
            mxf_muxer_ = std::make_unique<MxfMuxer>();
            if (!mxf_muxer_->open(pending_full_path_.c_str(), vi, ai)) {
                CASPAR_LOG(error) << L"[cuda_prores_bypass] Failed to open MXF: " << pending_full_path_;
                mxf_muxer_.reset();
            }
        } else {
            MovVideoTrackInfo vi{};
            vi.width         = static_cast<int>(w);
            vi.height        = static_cast<int>(h);
            vi.timebase_num  = static_cast<uint32_t>(tb_num);
            vi.timebase_den  = static_cast<uint32_t>(tb_den);
            vi.prores_fourcc = pending_fourcc_;
            vi.is_interlaced = interlaced;
            vi.is_tff        = tff;
            vi.color         = pending_color_info_;
            MovAudioTrackInfo ai{};
            ai.channels    = 16;
            ai.sample_rate = 48000;
            mov_muxer_ = std::make_unique<MovMuxer>();
            if (!mov_muxer_->open(pending_full_path_, vi, ai)) {
                CASPAR_LOG(error) << L"[cuda_prores_bypass] Failed to open MOV: " << pending_full_path_;
                mov_muxer_.reset();
            }
        }

        capture_hz_ = static_cast<double>(tb_den) / static_cast<double>(tb_num);

        CASPAR_LOG(info) << L"[cuda_prores_bypass] Encoder initialized: "
                         << w << L"x" << h << L" "
                         << (interlaced ? (tff ? L"interlaced TFF" : L"interlaced BFF") : L"progressive")
                         << L" @ " << capture_hz_ << L" fps"
                         << L" -> " << pending_full_path_;
        encoder_ready_ = true;
    }

    void encode_loop()
    {
        bool first_frame = true;
        int  encode_errors     = 0;
        int  encode_errors_log = 0; // last count that was logged

        while (running_ || !queue_empty()) {
            BypassJob job;
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                queue_cv_.wait_for(lock, std::chrono::milliseconds(50),
                    [this]{ return !frame_queue_.empty() || !running_; });
                if (frame_queue_.empty()) continue;
                job = std::move(frame_queue_.front());
                frame_queue_.pop();
            }

            auto t0 = std::chrono::high_resolution_clock::now();
            bool ok = encode_one(job, first_frame);
            first_frame = false;
            auto t1 = std::chrono::high_resolution_clock::now();
            double encode_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            // Use actual capture fps (set after first-frame init); fall back to channel hz
            // until the encoder is ready.
            const double hz = (capture_hz_ > 0.0) ? capture_hz_ : format_desc_.hz;
            double frame_ms  = 1000.0 / hz;
            graph_->set_value("encode-time", encode_ms / frame_ms);

            // Update graph label with elapsed recording time
            {
                int64_t frames_done = job.frame_number + 1;
                double elapsed_sec   = static_cast<double>(frames_done) / hz;
                std::wostringstream label;
                label << print() << L" | " << frames_done << L" fr ("
                      << std::fixed << std::setprecision(1) << elapsed_sec << L"s)";
                graph_->set_text(label.str());
            }

            if (!ok) {
                ++encode_errors;
                graph_->set_tag(diagnostics::tag_severity::WARNING, "encode-error");
                // Log on first failure, then every 100 to avoid spam
                if (encode_errors == 1 || (encode_errors - encode_errors_log) >= 100) {
                    CASPAR_LOG(error) << L"[cuda_prores_bypass] Encode failed (frame "
                                      << job.frame_number << L", total errors: " << encode_errors
                                      << L") — check width/format compatibility. "
                                      << L"Channel: " << format_desc_.width << L"x" << format_desc_.height
                                      << L" " << format_desc_.name;
                    encode_errors_log = encode_errors;
                }
            } else {
                encode_errors = 0;
                encode_errors_log = 0;
            }
            job.done.set_value(ok);
        }

        if (mov_muxer_) { mov_muxer_->close(); mov_muxer_.reset(); }
        if (mxf_muxer_) { mxf_muxer_->close(); mxf_muxer_.reset(); }

        CASPAR_LOG(info) << L"[cuda_prores_bypass] Encode thread exited cleanly.";
    }

    bool encode_one(BypassJob &job, bool first_frame)
    {
        const uint32_t *d_v210 = static_cast<const uint32_t*>(job.token.d_vram);

        // Wait for DeckLink H→D async copy to complete before reading VRAM.
        cudaError_t err = cudaStreamSynchronize(job.token.copy_stream);
        if (err != cudaSuccess) {
            CASPAR_LOG(error) << L"[cuda_prores_bypass] Stream sync failed: " << cudaGetErrorString(err);
            if (job.token.release_fn) job.token.release_fn();
            return false;
        }

        // On the first frame, initialize the encoder context and open the muxer
        // using the actual SDI capture dimensions from the token rather than the
        // CasparCG channel format.
        if (first_frame) {
            init_encoder_and_muxer(job.token.width, job.token.height,
                                   job.token.is_interlaced, job.token.is_tff,
                                   job.token.timebase_num, job.token.timebase_den);
            if (!encoder_ready_) {
                if (job.token.release_fn) job.token.release_fn();
                return false;
            }
        }

        static const ProResColorDesc k_sdr_709    = {1, 1, 1, {}, {}, 0, 0, 0, 0, 0, 0};
        static const ProResColorDesc k_hlg_bt2020 = {9, 14, 9, {}, {}, 0, 0, 0, 0, 0, 0};
        const ProResColorDesc *color_desc = (cfg_.hdr_mode == 1) ? &k_hlg_bt2020 : &k_sdr_709;

        size_t encoded_size = 0;

        if (is_interlaced_) {
            // Interlaced: split V210 full frame into two field planes, then encode
            const int parity_a = is_tff_ ? 0 : 1;  // TFF: field::a = even rows
            const int parity_b = is_tff_ ? 1 : 0;

            err = prores_launch_v210_unpack_field(d_v210, d_field_a_y_, d_field_a_cb_, d_field_a_cr_,
                                           frame_ctx_.width, frame_ctx_.height, parity_a, encode_stream_);
            if (err != cudaSuccess) {
                CASPAR_LOG(error) << L"[cuda_prores_bypass] V210 unpack field A failed: " << cudaGetErrorString(err);
                if (job.token.release_fn) job.token.release_fn();
                return false;
            }

            err = prores_launch_v210_unpack_field(d_v210, frame_ctx_.d_y, frame_ctx_.d_cb, frame_ctx_.d_cr,
                                           frame_ctx_.width, frame_ctx_.height, parity_b, encode_stream_);
            if (err != cudaSuccess) {
                CASPAR_LOG(error) << L"[cuda_prores_bypass] V210 unpack field B failed: " << cudaGetErrorString(err);
                if (job.token.release_fn) job.token.release_fn();
                return false;
            }

            // The DeckLink frame VRAM is no longer needed — release it so the ring slot recycles.
            if (job.token.release_fn) { job.token.release_fn(); job.token.release_fn = nullptr; }

            err = prores_encode_from_yuv_fields_422(
                &frame_ctx_,
                d_field_a_y_,   d_field_a_cb_,   d_field_a_cr_,   // picture 0 (temporal first)
                frame_ctx_.d_y, frame_ctx_.d_cb, frame_ctx_.d_cr,  // picture 1 (temporal second)
                frame_ctx_.h_frame_buf, &encoded_size,
                encode_stream_, color_desc);
        } else {
            // Progressive: encode raw V210 directly
            // DeckLink frame is still in use during encode; release after sync.
            err = prores_encode_frame(&frame_ctx_, d_v210,
                                      frame_ctx_.h_frame_buf, &encoded_size,
                                      encode_stream_, color_desc);
            if (job.token.release_fn) { job.token.release_fn(); job.token.release_fn = nullptr; }
        }

        if (err != cudaSuccess) {
            CASPAR_LOG(error) << L"[cuda_prores_bypass] Encode failed: " << cudaGetErrorString(err);
            return false;
        }

        // Set start timecode before first write
        if (first_frame && mxf_muxer_ && job.token.tc.valid)
            mxf_muxer_->set_start_timecode(job.token.tc);

        if (mov_muxer_) {
            if (!mov_muxer_->write_video(frame_ctx_.h_frame_buf, encoded_size, job.frame_number))
                return false;
            if (job.token.tc.valid)
                mov_muxer_->write_timecode(job.token.tc);
            if (!job.audio.empty())
                mov_muxer_->write_audio(job.audio.data(), job.audio.size() / job.audio_channels);
        }
        if (mxf_muxer_) {
            if (!mxf_muxer_->write_video(frame_ctx_.h_frame_buf, encoded_size, job.frame_number))
                return false;
            if (!job.audio.empty())
                mxf_muxer_->write_audio(job.audio.data(), job.audio.size() / job.audio_channels);
        }

        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            state_[std::string("frame")]        = job.frame_number;
            state_[std::string("profile")]      = cfg_.profile;
            state_[std::string("device_index")] = cfg_.device_index;
        }

        return true;
    }

    bool queue_empty() const {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        return frame_queue_.empty();
    }

    void stop()
    {
        if (!running_.exchange(false)) return;

        // Stop new frames from arriving.  DecklinkCapture::stop() syncs the copy
        // stream so any in-flight HtoD copies complete before we proceed.
        if (capture_) capture_->stop();

        // Wake the encode thread and let it drain the remaining queue
        queue_cv_.notify_all();
        if (encode_thread_.joinable())
            encode_thread_.join();

        // Safe to destroy capture now — encode thread is done using copy_stream
        capture_.reset();

        if (encode_stream_) {
            cudaStreamSynchronize(encode_stream_);
            cudaStreamDestroy(encode_stream_);
            encode_stream_ = nullptr;
        }
        if (frame_ctx_.d_y) free_bypass_ctx(frame_ctx_);

        if (d_field_a_y_)  { cudaFree(d_field_a_y_);  d_field_a_y_  = nullptr; }
        if (d_field_a_cb_) { cudaFree(d_field_a_cb_); d_field_a_cb_ = nullptr; }
        if (d_field_a_cr_) { cudaFree(d_field_a_cr_); d_field_a_cr_ = nullptr; }
    }

    // ── Configuration ─────────────────────────────────────────────────────
    bypass_config            cfg_;
    int                      index_;
    core::video_format_desc  format_desc_;  // channel format — used for hz (timing) only

    // ── Pending parameters for lazy encoder/muxer init ────────────────────
    // Set in initialize(), consumed on first frame in encode_one().
    std::wstring             pending_full_path_;
    uint32_t                 pending_fourcc_     = 0;
    MovColorInfo             pending_color_info_ = {};
    bool                     encoder_ready_      = false;
    double                   capture_hz_         = 0.0; // set from token on first frame

    // ── Format properties (set from CaptureToken on first frame) ──────────
    bool                     is_interlaced_ = false;
    bool                     is_tff_        = true;

    // ── DeckLink capture ──────────────────────────────────────────────────
    std::unique_ptr<DecklinkCapture> capture_;

    // ── CUDA resources ────────────────────────────────────────────────────
    cudaStream_t             encode_stream_ = nullptr;
    ProResFrameCtx           frame_ctx_     = {};

    // Interlaced-only: field A YUV422P10 planes (field B reuses ctx.d_y/cb/cr)
    int16_t                 *d_field_a_y_   = nullptr;
    int16_t                 *d_field_a_cb_  = nullptr;
    int16_t                 *d_field_a_cr_  = nullptr;

    // ── Muxers ────────────────────────────────────────────────────────────
    std::unique_ptr<MovMuxer>  mov_muxer_;
    std::unique_ptr<MxfMuxer>  mxf_muxer_;

    // ── Encode queue (bypass jobs = CaptureTokens) ────────────────────────
    // Queue depth is capped at DecklinkCapture::kVramRingSize so we never hold
    // more pending jobs than there are VRAM ring slots — otherwise an overflow
    // drop could recycle a slot while it's still being read by the encoder.
    static constexpr size_t kMaxQueueDepth = DecklinkCapture::kVramRingSize;
    mutable std::mutex       queue_mutex_;
    std::condition_variable  queue_cv_;
    std::queue<BypassJob>    frame_queue_;
    std::thread              encode_thread_;

    // ── Diagnostics ───────────────────────────────────────────────────────
    spl::shared_ptr<diagnostics::graph> graph_;

    // ── State ─────────────────────────────────────────────────────────────
    std::atomic<bool>        running_{true};
    int64_t                  frame_number_ = 0;
    mutable std::mutex       state_mutex_;
    core::monitor::state     state_;
};

// ---------------------------------------------------------------------------
// Factory helpers
// ---------------------------------------------------------------------------
static bypass_config parse_bypass_params(const std::vector<std::wstring>& params)
{
    bypass_config cfg;
    cfg.output_path   = caspar::get_param(L"PATH",    params, L".");
    // Strip control characters: AMCP processes C escape sequences in parameter
    // strings, so Windows paths with backslashes arrive mangled
    // (e.g. D:\recordings → D:<CR>ecordings).  Use forward slashes instead: D:/recordings
    {
        std::wstring clean;
        clean.reserve(cfg.output_path.size());
        bool had_ctrl = false;
        for (wchar_t c : cfg.output_path) { if (c < L' ') had_ctrl = true; else clean += c; }
        if (had_ctrl) {
            CASPAR_LOG(warning) << L"[cuda_prores_bypass] PATH contained control characters — "
                                   L"AMCP processes \\r, \\n etc. as C escape sequences. "
                                   L"Use forward slashes in paths: D:/recordings";
            cfg.output_path = std::move(clean);
        }
    }
    cfg.profile       = caspar::get_param(L"PROFILE", params, 3);
    cfg.device_index  = caspar::get_param(L"DEVICE",  params, 1);
    auto codec        = caspar::get_param(L"CODEC",   params, std::wstring(L"MOV"));
    cfg.use_mxf       = boost::iequals(codec, L"MXF");
    auto hdr          = boost::to_upper_copy(caspar::get_param(L"HDR", params, std::wstring(L"SDR")));
    cfg.hdr_mode      = boost::iequals(hdr, L"HLG") ? 1 : boost::iequals(hdr, L"PQ") ? 2 : 0;
    cfg.hdr_max_cll   = (uint16_t)caspar::get_param(L"MAXCLL",  params, 1000);
    cfg.hdr_max_fall  = (uint16_t)caspar::get_param(L"MAXFALL", params, 400);
    cfg.cuda_device   = caspar::get_param(L"CUDA_DEVICE", params, 0);
    cfg.filename_pattern = caspar::get_param(L"FILENAME", params, std::wstring(L""));
    // QSCALE 1..31 (1=maximum quality/largest file; default 8 matches Apple reference)
    int qscale   = caspar::get_param(L"QSCALE", params, 8);
    cfg.q_scale  = std::max(1, std::min(31, qscale));
    // SLICES: parallel horizontal slices per MB row — must be 1, 2, 4, or 8
    int slices = caspar::get_param(L"SLICES", params, 4);
    cfg.slices_per_row = (slices >= 8) ? 8 : (slices >= 4) ? 4 : (slices >= 2) ? 2 : 1;
    return cfg;
}

static bypass_config parse_bypass_xml(const boost::property_tree::wptree& elem)
{
    bypass_config cfg;
    cfg.output_path      = elem.get(L"path",     L".");
    cfg.filename_pattern = elem.get(L"filename", L"");
    cfg.profile          = elem.get(L"profile",  3);
    cfg.device_index     = elem.get(L"device",   1);
    auto codec = elem.get(L"codec", std::wstring(L"mov"));
    cfg.use_mxf = boost::iequals(codec, L"mxf");
    auto hdr = boost::to_upper_copy(elem.get(L"hdr", std::wstring(L"SDR")));
    cfg.hdr_mode    = boost::iequals(hdr, L"HLG") ? 1 : boost::iequals(hdr, L"PQ") ? 2 : 0;
    cfg.hdr_max_cll  = (uint16_t)elem.get(L"max_cll",  1000);
    cfg.hdr_max_fall = (uint16_t)elem.get(L"max_fall", 400);
    cfg.cuda_device  = elem.get(L"cuda_device", 0);
    int qscale = elem.get(L"qscale", 8);
    cfg.q_scale        = std::max(1, std::min(31, qscale));
    int slices = elem.get(L"slices", 4);
    cfg.slices_per_row = (slices >= 8) ? 8 : (slices >= 4) ? 4 : (slices >= 2) ? 2 : 1;
    return cfg;
}

// ---------------------------------------------------------------------------
// Exported factory functions
// ---------------------------------------------------------------------------
spl::shared_ptr<core::frame_consumer>
create_bypass_consumer(const std::vector<std::wstring>& params,
                       const core::video_format_repository& /*format_repository*/,
                       const std::vector<spl::shared_ptr<core::video_channel>>& /*channels*/,
                       const core::channel_info& /*channel_info*/)
{
    if (boost::to_upper_copy(params.at(0)) != L"CUDA_PRORES_BYPASS")
        return core::frame_consumer::empty();

    return spl::make_shared<prores_bypass_consumer_impl>(parse_bypass_params(params), 2);
}

spl::shared_ptr<core::frame_consumer>
create_preconfigured_bypass_consumer(const boost::property_tree::wptree& element,
                                     const core::video_format_repository& /*format_repository*/,
                                     const std::vector<spl::shared_ptr<core::video_channel>>& /*channels*/,
                                     const core::channel_info& /*channel_info*/)
{
    return spl::make_shared<prores_bypass_consumer_impl>(parse_bypass_xml(element), 2);
}

}} // namespace caspar::cuda_prores
