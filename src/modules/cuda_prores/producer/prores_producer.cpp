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

// prores_producer.cpp
// CasparCG frame_producer for CUDA ProRes decode.
// On Windows: zero-copy CUDA-GL interop using a shared WGL context owned by
// read_thread_, avoiding any OGL-thread involvement during map/decode/unmap.
// On other platforms: falls back to prores_decode_frame_to_host + PBO upload.
#include "prores_producer.h"

#include "prores_demuxer.h"
#include "../cuda/cuda_prores_decode.h"
#include "../util/cuda_gl_texture.h"

#include <accelerator/ogl/image/image_mixer.h>
#include <accelerator/ogl/util/device.h>
#include <accelerator/ogl/util/texture.h>

#include <common/array.h>
#include <common/bit_depth.h>
#include <common/diagnostics/graph.h>
#include <common/env.h>
#include <common/filesystem.h>
#include <common/log.h>
#include <common/memory.h>
#include <common/os/thread.h>
#include <common/timer.h>
#include <common/utf.h>

#include <core/frame/draw_frame.h>
#include <core/frame/frame.h>
#include <core/frame/frame_factory.h>
#include <core/frame/pixel_format.h>
#include <core/monitor/monitor.h>
#include <core/producer/frame_producer.h>
#include <core/module_dependencies.h>
#include <core/producer/frame_producer_registry.h>
#include <core/video_format.h>

#include <cuda_runtime.h>

#include <boost/algorithm/string.hpp>

#ifdef WIN32
// wglMakeCurrent/wglCreateContext etc. are standard WGL functions in wingdi.h.
// Do NOT include wglew.h here -- it redefines HGPUNV which conflicts with
// cuda_gl_interop.h (already included transitively via cuda_gl_texture.h).
#include <GL/glew.h>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cwctype>
#include <future>
#include <iomanip>
#include <memory>
#include <mutex>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <cmath>
#include <thread>

namespace caspar { namespace cuda_prores {

// 5 slots: with MAX_QUEUED=2, a slot is reused only after >=3 consumer pops
// (~120ms at 25fps), ensuring OGL has finished rendering before CUDA maps it.
// cudaGraphicsMapResources also provides implicit driver-level GL serialization.
static constexpr int NUM_SLOTS  = 5;
static constexpr int MAX_QUEUED = 2;

struct prores_producer_impl final : public core::frame_producer
{
    const std::wstring                    path_;
    const int                             cuda_device_;
    bool                                  loop_          = false;
    // -1 = use per-frame metadata; 1 = force BT.709; 6 = force BT.601; 9 = force BT.2020
    int                                   color_matrix_override_ = -1;
    spl::shared_ptr<core::frame_factory>  frame_factory_;
    std::shared_ptr<accelerator::ogl::device> ogl_device_;
    core::video_format_desc               format_desc_;
    int                                   audio_channels_ = 0;

    std::unique_ptr<ProResDemuxer> demuxer_;
    ProResFrameInfo                frame_info_;

    ProResDecodeCtx slots_     [NUM_SLOTS];
    bool            slots_init_[NUM_SLOTS] = {};

    // GL textures created on OGL thread, shared with read_thread_ via shared WGL
    // context so CUDA can write directly into them (zero PCIe DÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢H copy).
    // Essential for 12K throughput where host-copy would saturate the PCIe bus.
    std::shared_ptr<accelerator::ogl::texture> gl_tex_[NUM_SLOTS];
    std::shared_ptr<CudaGLTexture>             cgt_   [NUM_SLOTS];

    // Shared WGL context (Windows only): owned by read_thread_, shares object
    // names with CasparCG's main GL context so CUDA-GL interop works there.
#ifdef WIN32
    HDC   hdc_          = nullptr;
    HGLRC shared_hglrc_ = nullptr;
#endif

    // Host-copy fallback buffers (used when shared WGL context setup fails).
    uint16_t* h_bgra16_[NUM_SLOTS] = {};
    bool      use_host_copy_       = false;

    // Diagnostics
    spl::shared_ptr<diagnostics::graph> graph_;

    // Field deduplication: 25p content on 50i channel must serve the same frame
    // for both fields (video_field::a and video_field::b).
    core::draw_frame cached_frame_;

    std::queue<core::draw_frame> ready_queue_;
    std::mutex                   queue_mutex_;
    std::condition_variable      queue_cv_;

    std::thread      read_thread_;
    std::atomic_bool stop_flag_{false};

    std::atomic<int64_t>  frame_count_{0};
    int64_t               total_frames_ = -1;   // from container metadata, -1 if unknown
    double                total_seconds_ = 0.0;  // duration_us / 1e6
    caspar::timer         frame_timer_; // measures inter-receive latency for DIAG
    // Live FPS counter: count frames delivered in the last second
    caspar::timer         fps_window_timer_;
    int                   fps_frame_acc_ = 0;    // frames counted in current window
    double                fps_display_   = 0.0;  // last computed fps value
    // Seek: written by receive/call threads, consumed by read_loop.
    std::atomic<int64_t>  seek_request_{-1LL}; // -1 = none pending; >=0 = target frame
    int64_t               in_frame_  = 0;       // initial play position (PLAY ... SEEK/IN n)
    int64_t               out_frame_ = -1;      // exclusive stop frame  (-1 = play to EOF)

    std::atomic<double>   speed_{1.0};
    std::atomic<bool>     pingpong_{false};
    double                speed_accum_{0.0};

    mutable core::monitor::state monitor_state_;

    prores_producer_impl(const std::wstring& path, int cuda_device, bool loop,
                         int color_matrix_override, int64_t in_frame, int64_t out_frame,
                         const core::frame_producer_dependencies& deps)
        : path_(path), cuda_device_(cuda_device), loop_(loop)
        , color_matrix_override_(color_matrix_override)
        , in_frame_(in_frame), out_frame_(out_frame)
        , frame_factory_(deps.frame_factory)
    {
        auto* ogl_mixer = dynamic_cast<accelerator::ogl::image_mixer*>(frame_factory_.get());
        if (!ogl_mixer)
            CASPAR_THROW_EXCEPTION(std::runtime_error("[prores_producer] frame_factory is not ogl::image_mixer"));

        ogl_device_  = ogl_mixer->get_ogl_device();
        format_desc_ = deps.format_desc;

        // Must be called before any CUDA resource alloc on this thread.
        cudaSetDevice(cuda_device_);

        demuxer_ = std::make_unique<ProResDemuxer>(path_);
        if (!demuxer_->valid())
            CASPAR_THROW_EXCEPTION(std::runtime_error("[prores_producer] Cannot open: " + u8(path_)));

        auto pkt = demuxer_->read_packet();
        if (pkt.is_eof || pkt.data.empty())
            CASPAR_THROW_EXCEPTION(std::runtime_error("[prores_producer] Empty file"));

        if (!ProResDemuxer::parse_frame_info(pkt.data.data(), (int)pkt.data.size(), frame_info_))
            CASPAR_THROW_EXCEPTION(std::runtime_error("[prores_producer] Bad ProRes header"));

        const auto matrix_name = [](int m) -> const wchar_t* {
            switch (m) {
                case 1: return L"BT.709";
                case 5: case 6: return L"BT.601";
                case 9: return L"BT.2020";
                case 0: return L"unspecified (ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢BT.709)";
                default: return L"unknown (ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢BT.709)";
            }
        };
        CASPAR_LOG(info) << L"[prores_producer] " << frame_info_.width << L"x" << frame_info_.height
                         << L" profile=" << frame_info_.profile
                         << L" color_matrix=" << (int)frame_info_.color_matrix
                         << L" (" << matrix_name((int)frame_info_.color_matrix) << L")"
                         << (color_matrix_override_ >= 0
                             ? (std::wstring(L" [OVERRIDDEN ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ ") + matrix_name(color_matrix_override_) + L"]")
                             : std::wstring())
                         << (loop_      ? L" LOOP" : L"")
                         << (in_frame_  > 0  ? (L" IN="  + std::to_wstring(in_frame_))  : L"")
                         << (out_frame_ >= 0 ? (L" OUT=" + std::to_wstring(out_frame_)) : L"");

        if (frame_info_.width  != format_desc_.width ||
            frame_info_.height != format_desc_.height) {
            CASPAR_LOG(warning) << L"[prores_producer] File resolution " << frame_info_.width
                                << L"x" << frame_info_.height << L" != channel "
                                << format_desc_.width << L"x" << format_desc_.height
                                << L" -- compositor will scale.";
        }

        // ~~ Diagnostics graph ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        graph_ = spl::make_shared<diagnostics::graph>();
        graph_->set_text(print());
        graph_->set_color("frame-time", diagnostics::color(0.0f, 1.0f, 0.0f));
        graph_->set_color("decode-time", diagnostics::color(0.2f, 0.9f, 0.2f));
        graph_->set_color("queue-fill",  diagnostics::color(0.5f, 0.5f, 1.0f));
        graph_->set_color("dropped",     diagnostics::color(1.0f, 0.3f, 0.3f));
        graph_->auto_reset();
        diagnostics::register_graph(graph_);

        // ~~ CUDA decode contexts ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        const size_t max_frame = std::max(pkt.data.size() * 2 + 65536,
                                          (size_t)frame_info_.width * frame_info_.height);
        for (int i = 0; i < NUM_SLOTS; i++) {
            cudaError_t e = prores_decode_ctx_create(&slots_[i],
                frame_info_.width, frame_info_.height, frame_info_.profile,
                frame_info_.mbs_per_slice, frame_info_.slices_per_row,
                frame_info_.num_slices, max_frame);
            if (e != cudaSuccess)
                CASPAR_THROW_EXCEPTION(std::runtime_error(
                    std::string("[prores_producer] prores_decode_ctx_create: ") + cudaGetErrorString(e)));
            slots_init_[i] = true;
        }

        // ~~ GL textures + shared WGL context (Windows zero-copy path) ~~~~~~~~~~~~~~~~~
        // Create NUM_SLOTS GL textures on the OGL thread, then capture its WGL handles
        // and create a second HGLRC that shares object names with the main one.
        // read_thread_ makes this shared context current so CUDA-GL interop
        // (Register / Map / Unmap) runs entirely there, with no OGL-thread involvement
        // at decode time.  This is the zero-copy path required for 12K ProRes HQ.
        {
            int fw = frame_info_.width, fh = frame_info_.height;
            ogl_device_->dispatch_sync([this, fw, fh]() {
                for (int i = 0; i < NUM_SLOTS; i++)
                    gl_tex_[i] = ogl_device_->create_texture(fw, fh, 4, common::bit_depth::bit16);

#ifdef WIN32
                HGLRC main_hglrc = wglGetCurrentContext();
                hdc_             = wglGetCurrentDC();
                if (main_hglrc && hdc_) {
                    // Both contexts must be non-current for wglShareLists to succeed.
                    wglMakeCurrent(nullptr, nullptr);
                    shared_hglrc_ = wglCreateContext(hdc_);
                    if (shared_hglrc_) {
                        if (!wglShareLists(main_hglrc, shared_hglrc_)) {
                            CASPAR_LOG(warning) << L"[prores_producer] wglShareLists failed"
                                                << L" -- falling back to host-copy path";
                            wglDeleteContext(shared_hglrc_);
                            shared_hglrc_ = nullptr;
                        }
                    }
                    wglMakeCurrent(hdc_, main_hglrc);  // restore main OGL context
                }
#endif
            });
        }

        // ~~ Host-copy fallback buffers ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#ifdef WIN32
        use_host_copy_ = (shared_hglrc_ == nullptr);
#else
        use_host_copy_ = true;
#endif
        if (use_host_copy_) {
            CASPAR_LOG(info) << L"[prores_producer] Using host-copy upload path";
            const size_t out_bytes = (size_t)frame_info_.width * frame_info_.height * 4 * sizeof(uint16_t);
            for (int i = 0; i < NUM_SLOTS; i++) {
                cudaError_t e = cudaMallocHost(&h_bgra16_[i], out_bytes);
                if (e != cudaSuccess)
                    CASPAR_THROW_EXCEPTION(std::runtime_error(
                        std::string("[prores_producer] cudaMallocHost: ") + cudaGetErrorString(e)));
            }
        }

        demuxer_ = std::make_unique<ProResDemuxer>(path_);  // reopen from start
        audio_channels_ = demuxer_->audio_channels();

        // Capture total-frame / duration info for DIAG display and OSC state.
        total_frames_  = demuxer_->total_frames();   // -1 if container doesn't report it
        const int64_t dur_us = demuxer_->duration_us();
        total_seconds_ = (dur_us > 0) ? dur_us / 1e6 : 0.0;

        if (audio_channels_ > 0 && demuxer_->audio_sample_rate() != 48000) {
            CASPAR_LOG(warning) << L"[prores_producer] Audio sample rate is "
                                << demuxer_->audio_sample_rate()
                                << L" Hz (CasparCG expects 48000 Hz) -- audio may pitch/speed incorrectly.";
        }

        // Apply PLAY ... SEEK/IN start position before the reading thread begins.
        if (in_frame_ > 0) {
            demuxer_->seek_to_frame(in_frame_);
            frame_count_ = in_frame_;  // DIAG / OSC shows correct position from start
        }
        read_thread_ = std::thread([this] { read_loop(); });
    }

    ~prores_producer_impl() override {
        stop_flag_ = true;
        queue_cv_.notify_all();
        if (read_thread_.joinable()) read_thread_.join();
        // cgt_[] are unregistered and shared_hglrc_ deleted by read_thread_ at exit.
        for (int i = 0; i < NUM_SLOTS; i++) {
            cudaFreeHost(h_bgra16_[i]);
            if (slots_init_[i]) prores_decode_ctx_destroy(&slots_[i]);
        }
    }

    void read_loop() {
        cudaSetDevice(cuda_device_);
        set_thread_name(L"prores-read");

#ifdef WIN32
        // Make the shared GL context current here so CUDA-GL interop
        // (Register, Map, Unmap) runs on this thread with no involvement
        // from the single-threaded OGL dispatch thread.
        if (!use_host_copy_ && shared_hglrc_) {
            if (!wglMakeCurrent(hdc_, shared_hglrc_)) {
                CASPAR_LOG(error) << L"[prores_producer] wglMakeCurrent on read_thread_ failed"
                                  << L" -- switching to host-copy fallback";
                use_host_copy_ = true;
            } else {
                try {
                    for (int i = 0; i < NUM_SLOTS; i++)
                        cgt_[i] = std::make_shared<CudaGLTexture>(gl_tex_[i]);
                    CASPAR_LOG(info) << L"[prores_producer] CUDA-GL interop active";
                } catch (const std::exception& ex) {
                    CASPAR_LOG(error) << L"[prores_producer] CUDA-GL register: " << ex.what()
                                      << L" -- switching to host-copy fallback";
                    for (int i = 0; i < NUM_SLOTS; i++) cgt_[i].reset();
                    use_host_copy_ = true;
                }
            }
        }
#endif

        std::vector<int32_t> audio_accum;
        int                  audio_frame_idx   = 0;
        bool                 warned_interlaced = false;
        int     slot              = 0;
        int64_t video_frame_count = in_frame_;  // frames pushed to queue; used for OUT check

        while (true) {
            {
                std::unique_lock<std::mutex> lk(queue_mutex_);
                queue_cv_.wait(lk, [this] {
                    return stop_flag_ || (int)ready_queue_.size() < MAX_QUEUED || seek_request_ >= 0;
                });

                // A seek request takes priority over everything ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â even a pending stop from EOF.
                const int64_t seek_target = seek_request_.exchange(-1LL);
                if (seek_target >= 0) {
                    while (!ready_queue_.empty()) ready_queue_.pop();  // flush stale frames
                    stop_flag_ = false;  // cancel any EOF-induced stop
                    lk.unlock();
                    queue_cv_.notify_all();
                    demuxer_->seek_to_frame(seek_target);
                    frame_count_        = seek_target;
                    video_frame_count   = seek_target;
                    fps_frame_acc_      = 0;
                    fps_window_timer_.restart();
                    audio_accum.clear();
                    audio_frame_idx     = 0;
                    continue;
                }

                if (stop_flag_) break;
            }

            auto pkt = demuxer_->read_packet();

            double current_speed = speed_.load();

            // ~~ Loop / EOF ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if (pkt.is_eof) {
                if (pingpong_.load()) {
                    speed_.store(-current_speed);
                    video_frame_count = std::max(0LL, frame_count_.load() - 1);
                    demuxer_->seek_to_frame(video_frame_count);
                    audio_accum.clear();
                    audio_frame_idx   = 0;
                    continue;
                } else if (loop_) {
                    demuxer_->seek_to_frame(in_frame_);
                    video_frame_count = in_frame_;
                    frame_count_      = in_frame_;
                    audio_accum.clear();
                    audio_frame_idx   = 0;
                    fps_frame_acc_    = 0;
                    fps_window_timer_.restart();
                } else {
                    stop_flag_ = true;
                }
                continue;
            }
            if (pkt.data.empty()) continue;

            ProResFrameInfo fi;
            if (!ProResDemuxer::parse_frame_info(pkt.data.data(), (int)pkt.data.size(), fi)) continue;

            if (fi.frame_type != 0 && !warned_interlaced) {
                warned_interlaced = true;
                CASPAR_LOG(warning) << L"[prores_producer] interlaced content (frame_type="
                                    << fi.frame_type << L") decoded as progressive";
            }

            ProResDecodeCtx& ctx = slots_[slot];

            // ~~ Decode ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            caspar::timer decode_timer;
            cudaError_t   err;

            if (!use_host_copy_ && cgt_[slot]) {
                // Zero-copy: CUDA writes directly into the shared GL texture.
                // prores_decode_frame already calls cudaStreamSynchronize internally,
                // so by the time it returns the GPU write is complete and we can unmap.
                CudaGLTexture& cgt = *cgt_[slot];
                cudaArray_t arr;
                try { arr = cgt.map(ctx.stream); }
                catch (const std::exception& ex) {
                    CASPAR_LOG(error) << L"[prores_producer] GL map failed: " << ex.what();
                    continue;
                }
                const int cm = (color_matrix_override_ >= 0) ? color_matrix_override_ : (int)fi.color_matrix;
                err = prores_decode_frame(&ctx, pkt.data.data(), pkt.data.size(),
                                          cm, fi.frame_type != 0, arr);
                cgt.unmap(ctx.stream);
            } else {
                // Host-copy: GPU -> pinned buffer -> GL texture via PBO.
                const int cm = (color_matrix_override_ >= 0) ? color_matrix_override_ : (int)fi.color_matrix;
                err = prores_decode_frame_to_host(&ctx, pkt.data.data(), pkt.data.size(),
                                                  cm, fi.frame_type != 0,
                                                  h_bgra16_[slot]);
            }

            if (err != cudaSuccess) {
                CASPAR_LOG(warning) << L"[prores_producer] decode: " << cudaGetErrorString(err);
                graph_->set_tag(diagnostics::tag_severity::WARNING, "dropped");
                continue;
            }

            graph_->set_value("decode-time", decode_timer.elapsed() * format_desc_.fps * 0.5);
            graph_->set_value("queue-fill",
                static_cast<double>(ready_queue_.size() + 1) / (MAX_QUEUED + 1));

            // ~~ Audio ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if (!pkt.audio_samples.empty())
                audio_accum.insert(audio_accum.end(),
                                   pkt.audio_samples.begin(), pkt.audio_samples.end());
            const int cadence_len    = (int)format_desc_.audio_cadence.size();
            const int samples_per_ch = cadence_len > 0
                ? format_desc_.audio_cadence[audio_frame_idx % cadence_len]
                : 1920;
            ++audio_frame_idx;
            const int out_ch    = format_desc_.audio_channels;
            const int out_total = samples_per_ch * out_ch;
            std::vector<int32_t> frame_audio(out_total, 0);
            if (audio_channels_ > 0 && !audio_accum.empty()) {
                const int avail_samp = (int)audio_accum.size() / audio_channels_;
                const int take_samp  = std::min(avail_samp, samples_per_ch);
                const int copy_ch    = std::min(audio_channels_, out_ch);
                for (int s = 0; s < take_samp; s++)
                    for (int c = 0; c < copy_ch; c++)
                        frame_audio[s * out_ch + c] = audio_accum[s * audio_channels_ + c];
                audio_accum.erase(audio_accum.begin(),
                                  audio_accum.begin() + take_samp * audio_channels_);
            }

            // ~~ Build frame ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            core::draw_frame df;
            if (!use_host_copy_ && cgt_[slot]) {
                // Zero-copy path: CUDA writes raw BGRA16 bytes via cudaMemcpy2DToArrayAsync
                // directly into the GL_RGBA16 cudaArray with no format conversion.
                // GL therefore stores B_val in the R slot, R_val in the B slot.
                // pixel_format::rgba (case 2, identity .rgba swizzle) passes the
                // sampled (B,G,R,A) through unchanged, which is the BGRA convention
                // expected by the rest of the CasparCG pipeline.
                // (pixel_format::bgra would apply an extra .bgra swizzle that causes
                // the RÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬ÂB swap the user observes on the zero-copy path.)
                core::pixel_format_desc pfd(core::pixel_format::rgba);
                // We still need one plane entry so the planes.empty() guard in
                // image_mixer::visit does not skip us.  A matching dummy (zero-byte)
                // image_data entry satisfies the const_frame invariant
                // desc.planes.size() == image_data.size().
                pfd.planes.push_back(core::pixel_format_desc::plane(fi.width, fi.height, 4, common::bit_depth::bit16));

                auto empty_store = std::make_shared<std::vector<uint8_t>>(0);
                array<const uint8_t> dummy_img(empty_store->data(), 0, std::move(empty_store));
                std::vector<array<const uint8_t>> img_vec;
                img_vec.push_back(std::move(dummy_img));

                auto audio_store = std::make_shared<std::vector<int32_t>>(std::move(frame_audio));
                array<const int32_t> audio_arr(audio_store->data(), audio_store->size(), std::move(audio_store));

                df = core::draw_frame(core::const_frame(
                    this,
                    std::move(img_vec),
                    std::move(audio_arr),
                    pfd,
                    cgt_[slot]->gl_texture()));     // shared_ptr<ogl::texture> -> shared_ptr<core::texture>
            } else {
                // Host-copy: upload BGRA16 through standard PBO path.
                // glTextureSubImage2D is called with FORMAT[4]=GL_BGRA, which makes GL
                // swap BÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬ÂR when storing into GL_RGBA16, so the texture has correct
                // (R,G,B,A) slots.  pixel_format::bgra (case 1, .bgra swizzle) then
                // re-swaps back to (B,G,R,A) = BGRA convention for the pipeline.
                core::pixel_format_desc pfd(core::pixel_format::bgra);
                pfd.planes.push_back(core::pixel_format_desc::plane(fi.width, fi.height, 4, common::bit_depth::bit16));
                auto mf = frame_factory_->create_frame(this, pfd);
                std::memcpy(mf.image_data(0).begin(), h_bgra16_[slot], mf.image_data(0).size());
                mf.audio_data() = std::move(frame_audio);
                df = core::draw_frame(std::move(mf));
            }

            { std::lock_guard<std::mutex> lk(queue_mutex_); ready_queue_.push(std::move(df)); }
            queue_cv_.notify_one();
            
            if (current_speed < 0.0) {
                --video_frame_count;
                demuxer_->seek_to_frame(video_frame_count);
            } else {
                ++video_frame_count;
            }
            
            slot = (slot + 1) % NUM_SLOTS;

            // ~~ OUT / LENGTH check ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if (current_speed >= 0.0 && out_frame_ >= 0 && video_frame_count >= out_frame_) {
                if (pingpong_.load()) {
                    speed_.store(-current_speed);
                    video_frame_count = out_frame_ - 1;
                    demuxer_->seek_to_frame(video_frame_count);
                    audio_accum.clear();
                    audio_frame_idx   = 0;
                } else if (loop_) {
                    demuxer_->seek_to_frame(in_frame_);
                    video_frame_count = in_frame_;
                    frame_count_      = in_frame_;
                    audio_accum.clear();
                    audio_frame_idx   = 0;
                    fps_frame_acc_    = 0;
                    fps_window_timer_.restart();
                } else {
                    stop_flag_ = true;
                }
                queue_cv_.notify_all();
            } else if (current_speed < 0.0 && video_frame_count < in_frame_) {
                if (pingpong_.load()) {
                    speed_.store(-current_speed);
                    video_frame_count = in_frame_;
                    demuxer_->seek_to_frame(video_frame_count);
                    audio_accum.clear();
                    audio_frame_idx   = 0;
                } else if (loop_) {
                    int64_t target = (out_frame_ >= 0) ? out_frame_ - 1 : std::max(0LL, total_frames_ - 1);
                    demuxer_->seek_to_frame(target);
                    video_frame_count = target;
                    frame_count_      = target;
                    audio_accum.clear();
                    audio_frame_idx   = 0;
                    fps_frame_acc_    = 0;
                    fps_window_timer_.restart();
                } else {
                    stop_flag_ = true;
                }
                queue_cv_.notify_all();
            }
        }

#ifdef WIN32
        // Cleanup: unregister CUDA-GL resources while the shared context is current,
        // then release the context so it can be deleted.
        for (int i = 0; i < NUM_SLOTS; i++) cgt_[i].reset();
        if (shared_hglrc_) {
            wglMakeCurrent(nullptr, nullptr);
            wglDeleteContext(shared_hglrc_);
            shared_hglrc_ = nullptr;
        }
#endif
    }

    std::future<std::wstring> call(const std::vector<std::wstring>& params) override
    {
        std::wstring result;
        if (!params.empty()) {
            const auto& cmd   = params[0];
            const std::wstring val = params.size() > 1 ? params[1] : L"";

            if (boost::iequals(cmd, L"loop")) {
                if (!val.empty())
                    loop_ = boost::lexical_cast<bool>(val);
                result = loop_ ? L"1" : L"0";

            } else if (boost::iequals(cmd, L"pingpong")) {
                bool pp = pingpong_.load();
                if (!val.empty())
                    pp = boost::lexical_cast<bool>(val);
                pingpong_.store(pp);
                result = pp ? L"1" : L"0";

            } else if (boost::iequals(cmd, L"speed")) {
                double spd = speed_.load();
                if (!val.empty())
                    spd = boost::lexical_cast<double>(val);
                speed_.store(spd);
                result = boost::lexical_cast<std::wstring>(spd);

            } else if (boost::iequals(cmd, L"seek") && !val.empty()) {
                int64_t target;
                if      (boost::iequals(val, L"rel")   || boost::iequals(val, L"current"))
                    target = frame_count_.load();
                else if (boost::iequals(val, L"start") || boost::iequals(val, L"in"))
                    target = 0;
                else if (boost::iequals(val, L"end"))
                    target = std::max(0LL, total_frames_ - 1);
                else
                    target = boost::lexical_cast<int64_t>(val);

                // Optional relative offset: CALL 1-10 seek 100 +50
                if (params.size() > 2)
                    target += boost::lexical_cast<int64_t>(params[2]);

                target = std::max(0LL, target);
                if (total_frames_ > 0)
                    target = std::min(target, total_frames_ - 1);

                seek_request_ = target;
                queue_cv_.notify_one();
                result = std::to_wstring(target);

            } else {
                CASPAR_LOG(warning) << L"[prores_producer] CALL: unknown command '" << cmd
                                    << L"'. Supported: loop, seek";
            }
        }

        std::promise<std::wstring> p;
        p.set_value(result);
        return p.get_future();
    }

    core::draw_frame receive_impl(const core::video_field field, int) override {
        // Interlaced output channels (e.g. 1080i50) call receive_impl 50x/sec,
        // once per field (video_field::a = first, ::b = second).
        // A 25p source must return the same frame for both fields, otherwise
        // it will play at 2x speed on a 50i channel.
        //
        // Record frame-time on every call (including B field) so the DIAG graph
        // shows the inter-receive cadence normalized to [0,1] for both interlaced
        // and progressive channels (same convention as ffmpeg av_producer).
        graph_->set_value("frame-time", frame_timer_.elapsed() * format_desc_.hz * 0.5);
        frame_timer_.restart();

        if (field == core::video_field::b)
            return cached_frame_ ? cached_frame_ : core::draw_frame{};

        // First field or progressive: fetch the next decoded frame.
        std::unique_lock<std::mutex> lk(queue_mutex_);
        queue_cv_.wait_for(lk, std::chrono::milliseconds(40),
                           [this] { return !ready_queue_.empty() || stop_flag_; });
        if (ready_queue_.empty()) return cached_frame_;

        double spd = speed_.load();
        speed_accum_ += std::abs(spd);
        int frames_to_advance = static_cast<int>(speed_accum_);
        speed_accum_ -= static_cast<double>(frames_to_advance);

        if (frames_to_advance == 0) {
            lk.unlock();
            // Mute the audio of duplicated frames to prevent ear-bleeding repetition
            return cached_frame_ ? core::draw_frame::still(cached_frame_) : core::draw_frame{};
        }

        for (int i = 1; i < frames_to_advance && ready_queue_.size() > 1; ++i) {
            ready_queue_.pop();
        }

        cached_frame_ = std::move(ready_queue_.front());
        ready_queue_.pop();
        lk.unlock();
        queue_cv_.notify_all();

        auto fc = (frame_count_ += (spd >= 0.0) ? frames_to_advance : -frames_to_advance);

        // Live FPS: count frames delivered; recompute every ~1 second
        ++fps_frame_acc_;
        const double fps_elapsed = fps_window_timer_.elapsed();
        if (fps_elapsed >= 1000.0) {  // caspar::timer returns milliseconds
            fps_display_   = fps_frame_acc_ * 1000.0 / fps_elapsed;
            fps_frame_acc_ = 0;
            fps_window_timer_.restart();
        }

        // Graph title: "clip.mov  125 / 700  |  5.0s / 28.0s  |  25.0fps"
        const double cur_sec = static_cast<double>(fc) / format_desc_.fps;
        std::wstring title = boost::filesystem::path(path_).filename().wstring();
        title += L"  " + std::to_wstring(fc);
        if (total_frames_ > 0)
            title += L" / " + std::to_wstring(total_frames_);
        title += L"  |  " + [&]{ std::wostringstream s; s << std::fixed << std::setprecision(1) << cur_sec; return s.str(); }();
        if (total_seconds_ > 0.0)
            title += L"s / " + [&]{ std::wostringstream s; s << std::fixed << std::setprecision(1) << total_seconds_; return s.str(); }() + L"s";
        else
            title += L"s";
        if (fps_display_ > 0.0)
            title += L"  |  " + [&]{ std::wostringstream s; s << std::fixed << std::setprecision(1) << fps_display_; return s.str(); }() + L"fps";
        graph_->set_text(title);

        return cached_frame_;
    }

    bool is_ready() override { return !ready_queue_.empty(); }

    core::monitor::state state() const override {
        const double cur_sec = static_cast<double>(frame_count_) / format_desc_.fps;
        monitor_state_["file/name"]     = u8(boost::filesystem::path(path_).filename().wstring());
        monitor_state_["file/path"]     = u8(path_);
        // file/time mirrors ffmpeg av_producer: {current_seconds, total_seconds}
        monitor_state_["file/time"]     = {cur_sec, total_seconds_};
        monitor_state_["file/loop"]     = loop_;
        monitor_state_["width"]         = frame_info_.width;
        monitor_state_["height"]        = frame_info_.height;
        return monitor_state_;
    }
    std::wstring print() const override { return L"cuda_prores[" + path_ + L"]"; }
    std::wstring name()  const override { return L"cuda-prores"; }
};


spl::shared_ptr<core::frame_producer>
create_prores_producer(const core::frame_producer_dependencies& deps, const std::vector<std::wstring>& params) {
    // Require the "CUDA_PRORES" keyword as the first parameter so this factory
    // only activates for explicit PLAY commands, not general .mov playback.
    // Without this guard the ffmpeg producer (registered earlier) would win for
    // all .mov files since CasparCG tries factories in registration order.
    //
    //   PLAY 1-1 CUDA_PRORES <filename>
    //   PLAY 1-1 CUDA_PRORES FILE <filename> [DEVICE <index>]
    if (params.empty() || !boost::iequals(params[0], L"CUDA_PRORES"))
        return core::frame_producer::empty();

    std::wstring path; int cuda_device = 0; bool loop = false; int color_matrix_override = -1;
    int64_t start_frame = 0, out_frame = -1, length_param = -1;
    for (size_t i = 1; i < params.size(); ++i) {  // start at 1 ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â skip "CUDA_PRORES"
        const auto& p = params[i];
        if      (boost::iequals(p, L"LOOP"))                              { loop = true; }
        else if (boost::iequals(p, L"FILE")   && i+1 < params.size()) path        = params[++i];
        else if (boost::iequals(p, L"DEVICE") && i+1 < params.size()) cuda_device = std::stoi(params[++i]);
        else if ((boost::iequals(p, L"SEEK") || boost::iequals(p, L"IN") || boost::iequals(p, L"START"))
                 && i+1 < params.size())
            start_frame = boost::lexical_cast<int64_t>(params[++i]);
        else if (boost::iequals(p, L"LENGTH") && i+1 < params.size())
            length_param = boost::lexical_cast<int64_t>(params[++i]);
        else if (boost::iequals(p, L"OUT") && i+1 < params.size())
            out_frame = boost::lexical_cast<int64_t>(params[++i]);
        else if (boost::iequals(p, L"COLOR_MATRIX") && i+1 < params.size()) {
            const auto& val = params[++i];
            if      (boost::iequals(val, L"709")   || boost::iequals(val, L"BT709"))  color_matrix_override = 1;
            else if (boost::iequals(val, L"2020")  || boost::iequals(val, L"BT2020")) color_matrix_override = 9;
            else if (boost::iequals(val, L"601")   || boost::iequals(val, L"BT601"))  color_matrix_override = 6;
            else if (boost::iequals(val, L"AUTO"))                                     color_matrix_override = -1;
            else CASPAR_LOG(warning) << L"[prores_producer] Unknown COLOR_MATRIX value '" << val
                                     << L"' ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â valid: 709, 2020, 601, AUTO. Using file metadata.";
        }
        else if (path.empty() && !p.empty() && p[0] != L'-')               path = p;
    }
    // LENGTH is a frame count from start_frame; convert to absolute exclusive out frame.
    if (length_param >= 0 && out_frame < 0)
        out_frame = start_frame + length_param;
    if (path.empty()) return core::frame_producer::empty();

    // Resolve relative path against the CasparCG media folder,
    // probing ProRes-compatible container extensions if no extension given.
    {
        auto is_valid = [](const boost::filesystem::path& p) {
            auto ext = boost::to_lower_copy(p.extension().wstring());
            return ext == L".mov" || ext == L".mxf" || ext == L".mkv" || ext == L".mp4";
        };
        auto resolved = find_file_within_dir_or_absolute(env::media_folder(), path, is_valid);
        if (!resolved) {
            CASPAR_LOG(error) << L"[prores_producer] File not found: " << path;
            return core::frame_producer::empty();
        }
        path = resolved->wstring();
    }

    try { return spl::make_shared<prores_producer_impl>(path, cuda_device, loop, color_matrix_override, start_frame, out_frame, deps); }
    catch (const std::exception& ex) { CASPAR_LOG(error) << L"[prores_producer] " << ex.what(); return core::frame_producer::empty(); }
}

void register_prores_producer(const core::module_dependencies& module_deps) {
    module_deps.producer_registry->register_producer_factory(L"CUDA_PRORES Producer", &create_prores_producer);
}

}} // namespace caspar::cuda_prores
