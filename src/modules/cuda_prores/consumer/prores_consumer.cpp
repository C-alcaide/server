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

// prores_consumer.cpp
// CasparCG frame_consumer implementation for the CUDA ProRes recording pipeline.
//
// Frame flow
// ─────────────────────────────────────────────────────────────────────────────
// Mixer thread → send() → FrameJob enqueued → Encode thread:
//   1. Wait on frame_ready condition
//   2. Copy BGRA pixels to pinned staging buffer (host side)
//   3. cudaMemcpyAsync(d_bgra, h_bgra, HostToDevice, stream)
//   4. k_bgra_to_v210  (CUDA, in-stream)
//   5. prores_encode_frame (CUDA, in-stream)
//   6. cudaStreamSynchronize
//   7. muxer.write_video() + write_timecode()
//   8. muxer.write_audio() if audio present
//   9. promise.set_value(true)
//
// On the first frame, set_start_timecode() is called on the muxer before the
// first write so that MXF timecode metadata is embedded in the file header.
// ─────────────────────────────────────────────────────────────────────────────

#include "prores_consumer.h"

#include <common/except.h>
#include <common/log.h>
#include <common/param.h>
#include <common/memory.h>
#include <common/diagnostics/graph.h>

#include <core/consumer/frame_consumer.h>
#include <core/frame/frame.h>
#include <core/video_format.h>
#include <core/monitor/monitor.h>
#include <core/consumer/channel_info.h>

#include <boost/property_tree/ptree.hpp>
#include <boost/algorithm/string.hpp>

#include <cuda_runtime.h>

#include "../cuda/cuda_prores_frame.h"
#include "../cuda/cuda_prores_tables.cuh"
#include "../cuda/cuda_bgra_to_v210.cuh"
#include "../cuda/cuda_bgra_to_field422p10.cuh"
#include "../muxer/mov_muxer.h"
#include "../muxer/mxf_muxer.h"
#include "../timecode.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <ctime>
#include <future>
#include <iomanip>
#include <memory>
#include <mutex>
#include <queue>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace caspar { namespace cuda_prores {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static void cuda_check_consumer(cudaError_t e, const char *ctx)
{
    if (e != cudaSuccess) {
        CASPAR_LOG(error) << L"[cuda_prores] CUDA error in " << ctx << ": "
                          << cudaGetErrorString(e);
        throw std::runtime_error(std::string("[cuda_prores] CUDA error: ") + ctx);
    }
}

// Returns the V210 row stride for a given width (DeckLink convention).
static size_t v210_row_bytes(int w) {
    return static_cast<size_t>((w + 5) / 6) * 16;
}
static size_t v210_frame_bytes(int w, int h) {
    return v210_row_bytes(w) * h;
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------
struct prores_config {
    std::wstring output_path;
    std::wstring filename_pattern; // e.g. L"prores_%04d.mov"
    int          profile        = 3;     // 3=HQ, 4=4444
    bool         has_alpha      = true;  // encode alpha plane for 4444
    int          hdr_mode       = 0;     // 0=SDR_709, 1=HLG_BT2020, 2=PQ_HDR10
    uint16_t     hdr_max_cll    = 1000;  // MaxCLL  nits (PQ only)
    uint16_t     hdr_max_fall   = 400;   // MaxFALL nits (PQ only)
    bool         use_mxf        = false; // false=MOV
    int          device_index   = 0;     // CUDA device
    int          slices_per_row = 4;     // horizontal slices per MB row (1/2/4/8)
    int          q_scale        = 8;     // quantization scale [1..31]; 1=best quality
};

// ---------------------------------------------------------------------------
// FrameJob — one entry in the encode queue
// ---------------------------------------------------------------------------
struct FrameJob {
    core::const_frame      frame;          // field::a frame (interlaced) or full frame (progressive)
    core::const_frame      frame_b;        // field::b frame (interlaced only; empty for progressive)
    bool                   is_interlaced = false;
    int64_t                frame_number;
    SmpteTimecode          tc;             // synthetic TC from frame counter
    std::vector<int32_t>   audio;          // interleaved PCM32 samples for this frame
    std::promise<bool>     done;
};

// ---------------------------------------------------------------------------
// ProRes frame context helpers — allocate all device/host buffers
// ---------------------------------------------------------------------------
static void alloc_frame_ctx(ProResFrameCtx &ctx,
                             int width, int height, int profile,
                             int slices_per_row,
                             bool is_4444, bool has_alpha,
                             bool is_interlaced = false)
{
    ctx.width          = width;
    ctx.height         = height;
    ctx.profile        = profile;
    ctx.is_4444        = is_4444;
    ctx.has_alpha      = has_alpha;
    ctx.q_scale        = 8;

    // Derive macroblock geometry.
    // slices_per_row parameter carries mbs_per_slice (power of 2: 1,2,4,8);
    // actual slices_per_row is derived as (width_mbs / mbs_per_slice).
    ctx.mbs_per_slice  = slices_per_row;  // user SLICES param = MBs per slice
    ctx.slices_per_row = (width / 16) / ctx.mbs_per_slice;
    if ((width / 16) % ctx.mbs_per_slice != 0) ctx.slices_per_row++; // partial last slice
    ctx.num_slices     = ctx.slices_per_row * ((height + 15) / 16);  // total slices

    // blocks_per_slice: 422=8*mbs, 4444=12*mbs, 4444+alpha=16*mbs
    const int bpm = is_4444 ? (has_alpha ? 16 : 12) : 8;
    ctx.blocks_per_slice = ctx.mbs_per_slice * bpm;

    const int y_px = width * height;
    const int c_px = is_4444 ? y_px : (width / 2) * height;  // 4444 chroma = full-res

    cuda_check_consumer(cudaMalloc(&ctx.d_y,  y_px * sizeof(int16_t)), "d_y");
    cuda_check_consumer(cudaMalloc(&ctx.d_cb, c_px * sizeof(int16_t)), "d_cb");
    cuda_check_consumer(cudaMalloc(&ctx.d_cr, c_px * sizeof(int16_t)), "d_cr");
    cuda_check_consumer(cudaMalloc(&ctx.d_coeffs_y,  y_px * sizeof(int16_t)), "d_coeffs_y");
    cuda_check_consumer(cudaMalloc(&ctx.d_coeffs_cb, c_px * sizeof(int16_t)), "d_coeffs_cb");
    cuda_check_consumer(cudaMalloc(&ctx.d_coeffs_cr, c_px * sizeof(int16_t)), "d_coeffs_cr");

    const size_t slice_elems = (size_t)ctx.num_slices * ctx.blocks_per_slice * 64;
    cuda_check_consumer(cudaMalloc(&ctx.d_coeffs_slice, slice_elems * sizeof(int16_t)), "d_coeffs_slice");

    const size_t bs_size = slice_elems * sizeof(int16_t) * 2 + ctx.num_slices * 64;
    cuda_check_consumer(cudaMalloc(&ctx.d_bitstream,     bs_size),                                   "d_bitstream");
    cuda_check_consumer(cudaMalloc(&ctx.d_slice_offsets, (ctx.num_slices + 1) * sizeof(uint32_t)), "d_slice_offsets");
    cuda_check_consumer(cudaMalloc(&ctx.d_slice_sizes,    ctx.num_slices      * sizeof(uint32_t)), "d_slice_sizes");

    // 4 bit-count entries per slice for 4444+alpha, 3 otherwise
    const int bcp = (is_4444 && has_alpha) ? 4 : 3;
    cuda_check_consumer(cudaMalloc(&ctx.d_bit_counts, ctx.num_slices * bcp * sizeof(uint32_t)), "d_bit_counts");

    ctx.cub_temp_bytes = 8 * 1024 * 1024;
    cuda_check_consumer(cudaMalloc(&ctx.d_cub_temp, ctx.cub_temp_bytes), "d_cub_temp");

    // For interlaced, h_frame_buf must hold both picture headers + two fields' data.
    // Use full-frame height (height*2 when height=field_height) for sizing.
    const size_t buf_height = is_interlaced ? (size_t)height * 2 : (size_t)height;
    const size_t frame_buf_size = (size_t)width * buf_height * 8; // headroom for 4444 HQ
    ctx.h_frame_buf_size = frame_buf_size;
    cuda_check_consumer(cudaMallocHost(&ctx.h_frame_buf, frame_buf_size), "h_frame_buf");

    // 4444-specific planes
    ctx.d_alpha        = nullptr;
    ctx.d_coeffs_alpha = nullptr;
    if (is_4444) {
        cuda_check_consumer(cudaMalloc(&ctx.d_alpha, y_px * sizeof(int16_t)), "d_alpha");
        if (has_alpha) {
            const size_t alpha_elems = (size_t)ctx.num_slices * 4 * ctx.mbs_per_slice * 64;
            cuda_check_consumer(cudaMalloc(&ctx.d_coeffs_alpha,
                                           alpha_elems * sizeof(int16_t)), "d_coeffs_alpha");
        }
    }
}

static void free_frame_ctx(ProResFrameCtx &ctx)
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
    if (ctx.d_alpha)        cudaFree(ctx.d_alpha);
    if (ctx.d_coeffs_alpha) cudaFree(ctx.d_coeffs_alpha);
    std::memset(&ctx, 0, sizeof(ctx));
}

// Returns true if the channel format is top-field-first interlaced.
// All 1080i formats are TFF; SD (PAL/NTSC) are BFF.
// Progressive formats return true (is_tff is irrelevant for them).
static bool format_is_tff(const core::video_format_desc &fmt)
{
    using vf = core::video_format;
    // Known BFF formats: PAL and NTSC SD interlaced
    if (fmt.format == vf::pal || fmt.format == vf::ntsc)
        return false;
    // All standard 1080i and any other interlaced formats default to TFF
    return true;
}

// ---------------------------------------------------------------------------
// Consumer implementation
// ---------------------------------------------------------------------------
class prores_consumer_impl : public core::frame_consumer {
public:
    prores_consumer_impl(prores_config cfg, int consumer_index)
        : cfg_(std::move(cfg))
        , index_(consumer_index)
    {
        graph_->set_color("encode-time",   diagnostics::color(0.1f, 0.7f, 0.2f));
        graph_->set_color("queue-depth",   diagnostics::color(0.8f, 0.6f, 0.2f));
        graph_->set_color("dropped-frame", diagnostics::color(0.9f, 0.2f, 0.1f));
        graph_->set_color("encode-error",  diagnostics::color(0.9f, 0.3f, 0.7f));
        graph_->set_text(print());
        diagnostics::register_graph(graph_);
        CASPAR_LOG(info) << L"[cuda_prores] Created consumer #" << index_
                         << L" profile=" << cfg_.profile
                         << L" qscale=" << cfg_.q_scale
                         << L" slices=" << cfg_.slices_per_row
                         << L" " << (cfg_.use_mxf ? L"MXF" : L"MOV")
                         << L" → " << cfg_.output_path;
    }

    ~prores_consumer_impl() override
    {
        stop();
    }

    // ── frame_consumer interface ──────────────────────────────────────────

    void initialize(const core::video_format_desc& format_desc,
                    const core::channel_info& /*channel_info*/,
                    int /*port_index*/) override
    {
        format_desc_ = format_desc;

        // ProRes requires width divisible by 16 (macroblock boundary).
        // V210 with widths that are not multiples of 6 (e.g. 1280 for 720p) are handled
        // correctly — the frame planes are pre-zeroed before unpack so edge pixels are clean.
        if (format_desc_.width % 16 != 0)
            throw std::runtime_error(
                "[cuda_prores] Incompatible channel format: width "
                + std::to_string(format_desc_.width)
                + " is not a multiple of 16. ProRes requires macroblock-aligned width.");

        prores_tables_upload();

        // Detect interlaced and field dominance from the channel format
        is_interlaced_ = (format_desc_.field_count == 2);
        is_tff_        = format_is_tff(format_desc_);

        // For interlaced, allocate the encoder context for field dimensions.
        // alloc_frame_ctx sets ctx.height = enc_height; we restore full height
        // afterward for correct ProRes frame_header dimensions.
        const int enc_height = is_interlaced_ ? format_desc_.height / 2 : format_desc_.height;

        // Select CUDA device
        cuda_check_consumer(cudaSetDevice(cfg_.device_index), "cudaSetDevice");
        cuda_check_consumer(cudaStreamCreateWithFlags(&encode_stream_, cudaStreamNonBlocking),
                            "cudaStreamCreateWithFlags");

        // Allocate GPU resources
        const bool is_4444   = (cfg_.profile >= 4);
        const bool has_alpha = (is_4444 && cfg_.has_alpha);
        alloc_frame_ctx(frame_ctx_, format_desc_.width, enc_height,
                        cfg_.profile, cfg_.slices_per_row, is_4444, has_alpha,
                        is_interlaced_);
        frame_ctx_.q_scale = cfg_.q_scale;

        // Patch the context for interlaced so frame headers carry correct full dimensions
        if (is_interlaced_) {
            frame_ctx_.is_interlaced = true;
            frame_ctx_.field_height  = enc_height;
            frame_ctx_.height        = format_desc_.height;  // full height for ProRes frame header
            frame_ctx_.is_tff        = is_tff_;
        } else {
            frame_ctx_.field_height  = format_desc_.height;
        }

        // Pinned staging buffers: BGRA (input) and V210 (intermediate for progressive 422)
        const size_t bgra_bytes = (size_t)format_desc_.width * format_desc_.height * 4;
        const size_t v210_bytes = v210_frame_bytes(format_desc_.width, format_desc_.height);
        cuda_check_consumer(cudaMallocHost(&h_bgra_, bgra_bytes),  "h_bgra_");
        cuda_check_consumer(cudaMalloc(&d_bgra_, bgra_bytes),      "d_bgra_");
        if (!is_4444 && !is_interlaced_)
            cuda_check_consumer(cudaMalloc(&d_v210_, v210_bytes),  "d_v210_");

        // Interlaced: additional staging for field A BGRA and YUV422P10 planes
        if (is_interlaced_) {
            const int fh = enc_height;
            cuda_check_consumer(cudaMallocHost(&h_bgra_a_, bgra_bytes),                           "h_bgra_a_");
            cuda_check_consumer(cudaMalloc(&d_bgra_a_,     bgra_bytes),                           "d_bgra_a_");
            cuda_check_consumer(cudaMalloc(&d_field_a_y_,  (size_t)format_desc_.width       * fh * sizeof(int16_t)), "d_field_a_y_");
            cuda_check_consumer(cudaMalloc(&d_field_a_cb_, (size_t)(format_desc_.width / 2) * fh * sizeof(int16_t)), "d_field_a_cb_");
            cuda_check_consumer(cudaMalloc(&d_field_a_cr_, (size_t)(format_desc_.width / 2) * fh * sizeof(int16_t)), "d_field_a_cr_");
        }

        // Determine ProRes fourcc from profile
        // Note: LT uses 'apcs' (0x61706373), NOT 'apcl'.
        static const uint32_t PRORES_FOURCC[] = {
            0x6170636Fu, // 'apco' Proxy
            0x61706373u, // 'apcs' LT        (corrected from 'apcl')
            0x6170636Eu, // 'apcn' Standard
            0x61706368u, // 'apch' HQ
            0x61703468u, // 'ap4h' 4444
            0x61703478u, // 'ap4x' 4444 XQ
        };
        const uint32_t fourcc = PRORES_FOURCC[std::min(cfg_.profile, 5)];

        // Build HDR color info for MOV/MXF muxer
        MovColorInfo color_info{};
        switch (cfg_.hdr_mode) {
            default: // SDR Rec.709
                color_info = { 1, 1, 1, false };
                break;
            case 1: // HLG BT.2020
                color_info = { 9, 14, 9, false };
                break;
            case 2: { // PQ HDR10 — BT.2020 primaries, ST 2086
                color_info.color_primaries   = 9;
                color_info.transfer_function = 16;
                color_info.color_matrix      = 9;
                color_info.has_hdr           = true;
                // BT.2020 display primaries (G, B, R) in units of 0.00002
                color_info.mdcv_primaries_x[0] = 17000; // G x=0.170
                color_info.mdcv_primaries_y[0] = 34000; // G y=0.340  (approx BT.2020 G)
                color_info.mdcv_primaries_x[1] =  7500; // B x=0.0750
                color_info.mdcv_primaries_y[1] =  2500; // B y=0.0250
                color_info.mdcv_primaries_x[2] = 35400; // R x=0.708
                color_info.mdcv_primaries_y[2] = 14600; // R y=0.292
                color_info.mdcv_white_x = 15635; // D65 x=0.3127
                color_info.mdcv_white_y = 16450; // D65 y=0.3290
                color_info.mdcv_max_lum = (uint32_t)cfg_.hdr_max_cll  * 10000;
                color_info.mdcv_min_lum = 50;    // 0.005 cd/m²
                color_info.clli_max_cll  = cfg_.hdr_max_cll;
                color_info.clli_max_fall = cfg_.hdr_max_fall;
                break;
            }
        }

        // Derive output file path
        const std::wstring filename =
            cfg_.filename_pattern.empty()
                ? build_filename(frame_number_)
                : cfg_.filename_pattern;
        const std::wstring full_path = cfg_.output_path + L"\\" + filename;

        // Open the muxer
        if (cfg_.use_mxf) {
            MxfVideoTrackInfo vi{};
            vi.width  = format_desc_.width;
            vi.height = format_desc_.height;
            vi.frame_rate = { format_desc_.fps_den, format_desc_.fps_num };
            vi.prores_fourcc = fourcc;
            vi.color = (cfg_.hdr_mode == 1) ? MXF_COLOR_HDR_HLG
                     : (cfg_.hdr_mode == 2) ? MXF_COLOR_HDR_PQ
                     : MXF_COLOR_SDR_709;

            MxfAudioTrackInfo ai{};
            ai.channels    = 16;
            ai.sample_rate = 48000;

            mxf_muxer_ = std::make_unique<MxfMuxer>();
            if (!mxf_muxer_->open(full_path.c_str(), vi, ai)) {
                CASPAR_LOG(error) << L"[cuda_prores] Failed to open MXF: " << full_path;
                mxf_muxer_.reset();
            }
        } else {
            MovVideoTrackInfo vi{};
            vi.width  = format_desc_.width;
            vi.height = format_desc_.height;
            vi.timebase_num = (uint32_t)format_desc_.fps_den; // duration per frame
            vi.timebase_den = (uint32_t)format_desc_.fps_num; // timescale
            vi.prores_fourcc = fourcc;
            vi.is_interlaced = is_interlaced_;
            vi.is_tff        = is_tff_;
            vi.color = color_info;

            MovAudioTrackInfo ai{};
            ai.channels    = 16;
            ai.sample_rate = 48000;

            mov_muxer_ = std::make_unique<MovMuxer>();
            if (!mov_muxer_->open(full_path, vi, ai)) {
                CASPAR_LOG(error) << L"[cuda_prores] Failed to open MOV: " << full_path;
                mov_muxer_.reset();
            }
        }

        // Start encode thread
        encode_thread_ = std::thread(&prores_consumer_impl::encode_loop, this);

        CASPAR_LOG(info) << L"[cuda_prores] Recording → " << full_path;
    }

    std::future<bool> send(const core::video_field field, core::const_frame frame) override
    {
        if (!frame) return caspar::make_ready_future(false);
        if (!running_) return caspar::make_ready_future(false);

        if (is_interlaced_) {
            if (field == core::video_field::a) {
                // Buffer field A (and its audio); encode on field B arrival
                const auto &aud = frame.audio_data();
                pending_audio_a_.assign(aud.begin(), aud.end());
                pending_field_a_ = std::move(frame);
                return caspar::make_ready_future(true);
            }
            // field::b — build and enqueue the pair job
            const uint32_t fps = (uint32_t)(format_desc_.hz + 0.5); // e.g. 25 for 1080i50
            const int64_t  fn  = frame_number_++;
            SmpteTimecode tc = build_smpte_tc(fn, fps);

            FrameJob job;
            job.frame         = std::move(pending_field_a_);
            job.frame_b       = std::move(frame);
            job.is_interlaced = true;
            job.frame_number  = fn;
            job.tc            = tc;
            // Combine audio from both fields: field::a samples were stashed when
            // it arrived; field::b samples are appended now to form the full frame.
            job.audio.insert(job.audio.end(),
                             pending_audio_a_.begin(), pending_audio_a_.end());
            {
                const auto &aud = job.frame_b.audio_data();
                job.audio.insert(job.audio.end(), aud.begin(), aud.end());
            }
            auto future = job.done.get_future();
            enqueue_job(std::move(job));
            return future;
        }

        // ── Progressive path ──────────────────────────────────────────────
        const uint32_t fps = (uint32_t)(format_desc_.fps + 0.5);
        const int64_t  fn  = frame_number_++;
        SmpteTimecode tc = build_smpte_tc(fn, fps);

        FrameJob job;
        job.frame         = std::move(frame);
        job.is_interlaced = false;
        job.frame_number  = fn;
        job.tc            = tc;
        // Copy audio samples from the mixer frame
        {
            const auto &aud = job.frame.audio_data();
            job.audio.assign(aud.begin(), aud.end());
        }
        auto future = job.done.get_future();
        enqueue_job(std::move(job));
        return future;
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

    std::wstring name()  const override { return L"cuda_prores"; }
    int          index() const override { return index_; }
    std::wstring print() const override
    {
        return L"cuda_prores[" + std::to_wstring(index_) + L"|"
               + std::to_wstring(cfg_.profile) + L"]";
    }

private:
    // ── Helpers ───────────────────────────────────────────────────────────
    static SmpteTimecode build_smpte_tc(int64_t fn, uint32_t fps)
    {
        SmpteTimecode tc{};
        tc.valid      = true;
        tc.drop_frame = false;
        tc.frames     = (uint8_t)(fn % fps);
        const int64_t total_sec = fn / fps;
        tc.seconds = (uint8_t)(total_sec % 60);
        tc.minutes = (uint8_t)((total_sec / 60) % 60);
        tc.hours   = (uint8_t)((total_sec / 3600) % 24);
        return tc;
    }

    void enqueue_job(FrameJob job)
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        if (frame_queue_.size() >= kMaxQueueDepth) {
            frame_queue_.front().done.set_value(false);
            frame_queue_.pop();
            graph_->set_tag(diagnostics::tag_severity::WARNING, "dropped-frame");
            CASPAR_LOG(warning) << L"[cuda_prores] Queue overflow — dropped oldest frame";
        }
        graph_->set_value("queue-depth", static_cast<double>(frame_queue_.size()) / kMaxQueueDepth);
        frame_queue_.push(std::move(job));
        queue_cv_.notify_one();
    }

    // ── Encode loop (runs on encode_thread_) ─────────────────────────────
    void encode_loop()
    {
        bool first_frame = true;

        while (running_ || !queue_empty()) {
            FrameJob job;
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
            double frame_ms  = 1000.0 / format_desc_.hz;
            graph_->set_value("encode-time", encode_ms / frame_ms);

            // Update graph label with elapsed recording time
            {
                int64_t frames_done = job.frame_number + 1;
                double elapsed_sec  = static_cast<double>(frames_done) / format_desc_.hz;
                std::wostringstream label;
                label << print() << L" | " << frames_done << L" fr ("
                      << std::fixed << std::setprecision(1) << elapsed_sec << L"s)";
                graph_->set_text(label.str());
            }

            if (!ok)
                graph_->set_tag(diagnostics::tag_severity::WARNING, "encode-error");

            job.done.set_value(ok);
        }

        // Flush and close muxers
        if (mov_muxer_) { mov_muxer_->close(); mov_muxer_.reset(); }
        if (mxf_muxer_) { mxf_muxer_->close(); mxf_muxer_.reset(); }

        CASPAR_LOG(info) << L"[cuda_prores] Encode thread exited cleanly.";
    }

    bool encode_one(FrameJob &job, bool first_frame)
    {
        const auto &fmt = format_desc_;

        static const ProResColorDesc k_sdr_709    = {1, 1, 1, {}, {}, 0, 0, 0, 0, 0, 0};
        static const ProResColorDesc k_hlg_bt2020 = {9, 14, 9, {}, {}, 0, 0, 0, 0, 0, 0};
        const ProResColorDesc *color_desc = (cfg_.hdr_mode == 1) ? &k_hlg_bt2020 : &k_sdr_709;

        size_t encoded_size = 0;
        cudaError_t err;

        if (job.is_interlaced) {
            // ── Interlaced path ───────────────────────────────────────────
            // Field A BGRA → pinned h_bgra_a_ → device d_bgra_a_
            const uint8_t *px_a = job.frame.image_data(0).begin();
            const size_t bgra_sz = (size_t)fmt.width * fmt.height * 4;
            std::memcpy(h_bgra_a_, px_a, std::min(bgra_sz, job.frame.image_data(0).size()));
            err = cudaMemcpyAsync(d_bgra_a_, h_bgra_a_, bgra_sz, cudaMemcpyHostToDevice, encode_stream_);
            if (err != cudaSuccess) {
                CASPAR_LOG(error) << L"[cuda_prores] H→D field_a failed: " << cudaGetErrorString(err);
                return false;
            }

            // Field B BGRA → pinned h_bgra_ → device d_bgra_
            const uint8_t *px_b = job.frame_b.image_data(0).begin();
            std::memcpy(h_bgra_, px_b, std::min(bgra_sz, job.frame_b.image_data(0).size()));
            err = cudaMemcpyAsync(d_bgra_, h_bgra_, bgra_sz, cudaMemcpyHostToDevice, encode_stream_);
            if (err != cudaSuccess) {
                CASPAR_LOG(error) << L"[cuda_prores] H→D field_b failed: " << cudaGetErrorString(err);
                return false;
            }

            // Determine yadif output field parity for each send() field:
            // TFF: field::a has real data at even rows (parity 0), field::b at odd (parity 1)
            // BFF: field::a has real data at odd  rows (parity 1), field::b at even (parity 0)
            const int parity_a = is_tff_ ? 0 : 1;
            const int parity_b = is_tff_ ? 1 : 0;

            // Extract field A → d_field_a_y/cb/cr (picture 0 = temporal first)
            err = launch_bgra8_to_field422p10(d_bgra_a_, d_field_a_y_, d_field_a_cb_, d_field_a_cr_,
                                              fmt.width, fmt.height, parity_a, encode_stream_);
            if (err != cudaSuccess) {
                CASPAR_LOG(error) << L"[cuda_prores] BGRA→field_a failed: " << cudaGetErrorString(err);
                return false;
            }

            // Extract field B → ctx.d_y/cb/cr (picture 1 = temporal second)
            err = launch_bgra8_to_field422p10(d_bgra_, frame_ctx_.d_y, frame_ctx_.d_cb, frame_ctx_.d_cr,
                                              fmt.width, fmt.height, parity_b, encode_stream_);
            if (err != cudaSuccess) {
                CASPAR_LOG(error) << L"[cuda_prores] BGRA→field_b failed: " << cudaGetErrorString(err);
                return false;
            }

            // Encode both fields as interlaced ProRes (two picture headers in icpf box)
            err = prores_encode_from_yuv_fields_422(
                &frame_ctx_,
                d_field_a_y_,  d_field_a_cb_,  d_field_a_cr_,   // picture 0
                frame_ctx_.d_y, frame_ctx_.d_cb, frame_ctx_.d_cr, // picture 1
                frame_ctx_.h_frame_buf, &encoded_size,
                encode_stream_, color_desc);
        } else if (frame_ctx_.is_4444) {
            // ── Progressive 4444 path ─────────────────────────────────────
            const uint8_t *frame_pixels = job.frame.image_data(0).begin();
            const size_t bgra_bytes = (size_t)fmt.width * fmt.height * 4;
            std::memcpy(h_bgra_, frame_pixels, std::min(bgra_bytes, job.frame.image_data(0).size()));
            err = cudaMemcpyAsync(d_bgra_, h_bgra_, bgra_bytes, cudaMemcpyHostToDevice, encode_stream_);
            if (err != cudaSuccess) {
                CASPAR_LOG(error) << L"[cuda_prores] cudaMemcpyAsync(bgra) failed: " << cudaGetErrorString(err);
                return false;
            }
            err = prores_encode_frame_444(&frame_ctx_, d_bgra_, frame_ctx_.h_frame_buf,
                                          &encoded_size, encode_stream_, color_desc);
        } else {
            // ── Progressive 422 path ──────────────────────────────────────
            const uint8_t *frame_pixels = job.frame.image_data(0).begin();
            const size_t bgra_bytes = (size_t)fmt.width * fmt.height * 4;
            std::memcpy(h_bgra_, frame_pixels, std::min(bgra_bytes, job.frame.image_data(0).size()));
            err = cudaMemcpyAsync(d_bgra_, h_bgra_, bgra_bytes, cudaMemcpyHostToDevice, encode_stream_);
            if (err != cudaSuccess) {
                CASPAR_LOG(error) << L"[cuda_prores] cudaMemcpyAsync(bgra) failed: " << cudaGetErrorString(err);
                return false;
            }
            err = launch_bgra_to_v210(d_bgra_, d_v210_, fmt.width, fmt.height, encode_stream_);
            if (err != cudaSuccess) {
                CASPAR_LOG(error) << L"[cuda_prores] launch_bgra_to_v210 failed: " << cudaGetErrorString(err);
                return false;
            }
            err = prores_encode_frame(&frame_ctx_, (const uint32_t *)d_v210_,
                                      frame_ctx_.h_frame_buf, &encoded_size,
                                      encode_stream_, color_desc);
        }

        if (err != cudaSuccess) {
            CASPAR_LOG(error) << L"[cuda_prores] Encode failed: " << cudaGetErrorString(err);
            return false;
        }

        // Set MXF start timecode before first write
        if (first_frame && mxf_muxer_ && job.tc.valid)
            mxf_muxer_->set_start_timecode(job.tc);

        // 4. Mux video frame
        if (mov_muxer_) {
            if (!mov_muxer_->write_video(frame_ctx_.h_frame_buf, encoded_size, job.frame_number))
                return false;
            if (job.tc.valid)
                mov_muxer_->write_timecode(job.tc);
            if (!job.audio.empty())
                mov_muxer_->write_audio(job.audio.data(),
                                        (int)(job.audio.size() / format_desc_.audio_channels));
        }
        if (mxf_muxer_) {
            if (!mxf_muxer_->write_video(frame_ctx_.h_frame_buf, encoded_size, job.frame_number))
                return false;
            if (!job.audio.empty())
                mxf_muxer_->write_audio(job.audio.data(),
                                        (int)(job.audio.size() / format_desc_.audio_channels));
        }

        // 6. Update diagnostics state
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            state_[L"frame"]   = job.frame_number;
            state_[L"profile"] = cfg_.profile;
        }

        return true;
    }

    bool queue_empty() const {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        return frame_queue_.empty();
    }

    void stop()
    {
        if (!running_.exchange(false)) return; // already stopped
        queue_cv_.notify_all();
        if (encode_thread_.joinable())
            encode_thread_.join();

        // GPU resource cleanup
        if (encode_stream_) {
            cudaStreamSynchronize(encode_stream_);
            cudaStreamDestroy(encode_stream_);
            encode_stream_ = nullptr;
        }
        if (frame_ctx_.d_y) free_frame_ctx(frame_ctx_);
        if (h_bgra_) { cudaFreeHost(h_bgra_); h_bgra_ = nullptr; }
        if (d_bgra_) { cudaFree(d_bgra_);     d_bgra_ = nullptr; }
        if (d_v210_) { cudaFree(d_v210_);     d_v210_ = nullptr; }

        // Interlaced staging buffers
        if (h_bgra_a_)     { cudaFreeHost(h_bgra_a_);     h_bgra_a_     = nullptr; }
        if (d_bgra_a_)     { cudaFree(d_bgra_a_);         d_bgra_a_     = nullptr; }
        if (d_field_a_y_)  { cudaFree(d_field_a_y_);      d_field_a_y_  = nullptr; }
        if (d_field_a_cb_) { cudaFree(d_field_a_cb_);     d_field_a_cb_ = nullptr; }
        if (d_field_a_cr_) { cudaFree(d_field_a_cr_);     d_field_a_cr_ = nullptr; }
    }

    std::wstring build_filename(int64_t /*fn*/) const
    {
        // Generate a timestamp-based filename so successive recordings don't
        // overwrite each other: prores_20260313_142305.mov
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
        return std::wstring(buf) + ext;
    }

    // ── Configuration ─────────────────────────────────────────────────────
    prores_config            cfg_;
    int                      index_;
    core::video_format_desc  format_desc_;

    // ── Format properties ─────────────────────────────────────────────────
    bool                     is_interlaced_ = false;
    bool                     is_tff_        = true;
    core::const_frame        pending_field_a_;  // buffered field::a for interlaced
    std::vector<int32_t>     pending_audio_a_;  // audio samples from field::a

    // ── CUDA resources ────────────────────────────────────────────────────
    cudaStream_t             encode_stream_ = nullptr;
    ProResFrameCtx           frame_ctx_     = {};
    uint8_t                 *h_bgra_        = nullptr; // pinned staging (field B / progressive)
    uint8_t                 *d_bgra_        = nullptr; // device BGRA    (field B / progressive)
    uint32_t                *d_v210_        = nullptr; // device V210    (progressive 422 only)

    // Interlaced-only: field A BGRA staging and YUV422P10 output planes
    uint8_t                 *h_bgra_a_      = nullptr;
    uint8_t                 *d_bgra_a_      = nullptr;
    int16_t                 *d_field_a_y_   = nullptr;
    int16_t                 *d_field_a_cb_  = nullptr;
    int16_t                 *d_field_a_cr_  = nullptr;

    // ── Muxers (only one is active at a time) ─────────────────────────────
    std::unique_ptr<MovMuxer>  mov_muxer_;
    std::unique_ptr<MxfMuxer>  mxf_muxer_;

    // ── Encode queue ──────────────────────────────────────────────────────
    static constexpr size_t kMaxQueueDepth = 8;
    mutable std::mutex       queue_mutex_;
    std::condition_variable  queue_cv_;
    std::queue<FrameJob>     frame_queue_;
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
static prores_config parse_params(const std::vector<std::wstring>& params)
{
    prores_config cfg;
    cfg.output_path = caspar::get_param(L"PATH", params, L".");
    // Strip control characters: AMCP processes C escape sequences in parameter
    // strings, so Windows paths with backslashes arrive mangled
    // (e.g. D:\recordings → D:<CR>ecordings).  Use forward slashes instead: D:/recordings
    {
        std::wstring clean;
        clean.reserve(cfg.output_path.size());
        bool had_ctrl = false;
        for (wchar_t c : cfg.output_path) { if (c < L' ') had_ctrl = true; else clean += c; }
        if (had_ctrl) {
            CASPAR_LOG(warning) << L"[cuda_prores] PATH contained control characters — "
                                   L"AMCP processes \\r, \\n etc. as C escape sequences. "
                                   L"Use forward slashes in paths: D:/recordings";
            cfg.output_path = std::move(clean);
        }
    }
    cfg.profile     = caspar::get_param(L"PROFILE", params, 3);
    auto codec      = caspar::get_param(L"CODEC", params, std::wstring(L"MOV"));
    cfg.use_mxf     = boost::iequals(codec, L"MXF");
    // HDR: HDR SDR|HLG|PQ  (default SDR)
    auto hdr = boost::to_upper_copy(caspar::get_param(L"HDR", params, std::wstring(L"SDR")));
    cfg.hdr_mode = boost::iequals(hdr, L"HLG") ? 1 : boost::iequals(hdr, L"PQ") ? 2 : 0;
    cfg.hdr_max_cll  = (uint16_t)caspar::get_param(L"MAXCLL",  params, 1000);
    cfg.hdr_max_fall = (uint16_t)caspar::get_param(L"MAXFALL", params, 400);
    // ALPHA: 1|0 (default 1 for profile 4444)
    cfg.has_alpha        = (caspar::get_param(L"ALPHA", params, 1) != 0);
    cfg.filename_pattern = caspar::get_param(L"FILENAME", params, std::wstring(L""));
    // QSCALE 1..31 (1=maximum quality/largest file; default 8 matches Apple reference)
    int qscale   = caspar::get_param(L"QSCALE", params, 8);
    cfg.q_scale  = std::max(1, std::min(31, qscale));
    // SLICES: parallel horizontal slices per MB row — must be 1, 2, 4, or 8
    int slices = caspar::get_param(L"SLICES", params, 4);
    cfg.slices_per_row = (slices >= 8) ? 8 : (slices >= 4) ? 4 : (slices >= 2) ? 2 : 1;
    return cfg;
}

static prores_config parse_xml(const boost::property_tree::wptree& elem)
{
    prores_config cfg;
    cfg.output_path      = elem.get(L"path",    L".");
    cfg.filename_pattern = elem.get(L"filename", L"");
    cfg.profile          = elem.get(L"profile",  3);
    auto codec = elem.get(L"codec", std::wstring(L"mov"));
    cfg.use_mxf = boost::iequals(codec, L"mxf");
    auto hdr = boost::to_upper_copy(elem.get(L"hdr", std::wstring(L"SDR")));
    cfg.hdr_mode = boost::iequals(hdr, L"HLG") ? 1 : boost::iequals(hdr, L"PQ") ? 2 : 0;
    cfg.hdr_max_cll    = (uint16_t)elem.get(L"max_cll",  1000);
    cfg.hdr_max_fall   = (uint16_t)elem.get(L"max_fall", 400);
    cfg.has_alpha      = (elem.get(L"alpha", 1) != 0);
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
create_consumer(const std::vector<std::wstring>& params,
                const core::video_format_repository& /*format_repository*/,
                const std::vector<spl::shared_ptr<core::video_channel>>& /*channels*/,
                const core::channel_info& /*channel_info*/)
{
    if (boost::to_upper_copy(params.at(0)) != L"CUDA_PRORES")
        return core::frame_consumer::empty();

    return spl::make_shared<prores_consumer_impl>(parse_params(params), 1);
}

spl::shared_ptr<core::frame_consumer>
create_preconfigured_consumer(const boost::property_tree::wptree& element,
                              const core::video_format_repository& /*format_repository*/,
                              const std::vector<spl::shared_ptr<core::video_channel>>& /*channels*/,
                              const core::channel_info& /*channel_info*/)
{
    return spl::make_shared<prores_consumer_impl>(parse_xml(element), 1);
}

}} // namespace caspar::cuda_prores
