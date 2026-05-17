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
 */

// hap_producer.cpp
// CasparCG frame_producer for HAP decode.
// Architecture mirrors cuda_notchlc/producer/notchlc_producer.cpp;
// differences: GL compressed textures instead of CUDA kernels, Snappy instead of LZ4.
#include "hap_producer.h"

#include "hap_demuxer.h"
#include "../cpu/hap_cpu_decode.h"
#include "../gl/hap_gl_decode.h"
#include "../util/hap_frame_parser.h"

#include <accelerator/ogl/image/image_mixer.h>
#include <accelerator/ogl/util/device.h>
#include <accelerator/ogl/util/texture.h>

#ifdef ENABLE_VULKAN
#include <accelerator/vulkan/image/image_mixer.h>
#include <accelerator/vulkan/util/device.h>
#include <accelerator/vulkan/util/texture.h>
#include <accelerator/vulkan/util/texture_wrapper.h>
#endif

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

#include <boost/algorithm/string.hpp>

#ifdef WIN32
#include <GL/glew.h>
#include <GL/wglew.h>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cmath>
#include <future>
#include <iomanip>
#include <memory>
#include <mutex>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

namespace caspar { namespace hap {

static constexpr int NUM_SLOTS     = 5;
static constexpr int MAX_QUEUED    = 2;
static constexpr int IO_QUEUE_CAP  = 4;

// Raw packet from I/O thread → Snappy workers.
struct RawPacket {
    HapPacket pkt;
    uint64_t  seq   = 0;
    uint32_t  epoch = 0;
};

// Decompressed DXT data from Snappy workers → GL thread.
struct DecompressedItem {
    uint64_t               seq   = 0;
    uint32_t               epoch = 0;
    HapFrameResult         result;
    std::vector<int32_t>   audio_samples;
};

// Min-heap on sequence number for reordering.
struct DecompressedItemCmp {
    bool operator()(const DecompressedItem& a, const DecompressedItem& b) const
    { return a.seq > b.seq; }
};

struct hap_producer_impl final : public core::frame_producer
{
    const std::wstring                        path_;
    std::atomic<bool>                            loop_{false};
    spl::shared_ptr<core::frame_factory>      frame_factory_;
    std::shared_ptr<accelerator::ogl::device> ogl_device_; // nullptr when using CPU decode
#ifdef ENABLE_VULKAN
    std::shared_ptr<accelerator::vulkan::device> vk_device_; // non-null when VK mixer detected
#endif
    bool                                      use_cpu_decode_ = false;
    bool                                      use_vk_upload_  = false;
    core::video_format_desc                   format_desc_;
    int                                       audio_channels_ = 0;

    std::unique_ptr<HapDemuxer>               demuxer_;
    HapFrameInfo                              frame_info_;

    HapDecodeSlot                             decode_slots_[NUM_SLOTS];
    std::unique_ptr<HapGLDecoder>             gl_decoder_;

#ifdef WIN32
    HDC   hdc_          = nullptr;
    HGLRC shared_hglrc_ = nullptr;
#endif

    spl::shared_ptr<diagnostics::graph>       graph_;
    core::draw_frame                          cached_frame_;

    std::queue<core::draw_frame>              ready_queue_;
    std::mutex                                queue_mutex_;
    std::condition_variable                   queue_cv_;

    std::thread                               gl_thread_;
    std::thread                               io_thread_;
    std::thread                               snappy_thread_a_;
    std::thread                               snappy_thread_b_;
    std::thread                               snappy_thread_c_;
    std::thread                               snappy_thread_d_;

    // io_thread → Snappy workers
    std::queue<RawPacket>                     raw_queue_;
    std::mutex                                raw_mutex_;
    std::condition_variable                   raw_cv_;

    // Snappy workers → GL thread: priority queue for reordering
    std::priority_queue<DecompressedItem, std::vector<DecompressedItem>, DecompressedItemCmp>
                                              done_pq_;
    std::mutex                                done_mutex_;
    std::condition_variable                   done_cv_;

    std::atomic<uint32_t>                     seek_epoch_{0};
    std::atomic_bool                          stop_flag_{false};
    std::atomic_bool                          seek_done_{false};
    std::atomic_bool                          eof_paused_{false};

    std::atomic<int64_t>                      frame_count_{0};
    int64_t                                   total_frames_   = -1;
    double                                    total_seconds_  = 0.0;
    double                                    file_fps_       = 0.0;
    caspar::timer                             frame_timer_;
    caspar::timer                             fps_window_timer_;
    std::atomic<int>                             fps_frame_acc_{0};
    double                                    fps_display_    = 0.0;

    std::atomic<int64_t>                      seek_request_{-1LL};
    int64_t                                   in_frame_          = 0;
    int64_t                                   out_frame_         = -1;
    int64_t                                   video_frame_start_ = 0;

    std::atomic<double>                       speed_{1.0};
    std::atomic<bool>                         pingpong_{false};
    double                                    speed_accum_{0.0};

    mutable core::monitor::state              monitor_state_;

    // Detected HAP variant from first frame (used to init GL textures).
    HapVariant                                detected_variant_ = HapVariant::Unknown;

    hap_producer_impl(const std::wstring& path, bool loop, bool pingpong,
                      int64_t in_frame, int64_t out_frame, double initial_speed,
                      const core::frame_producer_dependencies& deps)
        : path_(path), loop_(loop)
        , in_frame_(in_frame), out_frame_(out_frame)
        , speed_(initial_speed), pingpong_(pingpong)
        , frame_factory_(deps.frame_factory)
    {
        auto* ogl_mixer = dynamic_cast<accelerator::ogl::image_mixer*>(frame_factory_.get());
#ifdef ENABLE_VULKAN
        auto* vk_mixer  = dynamic_cast<accelerator::vulkan::image_mixer*>(frame_factory_.get());
#else
        void* vk_mixer  = nullptr;
#endif
        if (ogl_mixer) {
            ogl_device_     = ogl_mixer->get_ogl_device();
            use_cpu_decode_ = false;
        } else if (vk_mixer) {
#ifdef ENABLE_VULKAN
            vk_device_      = vk_mixer->get_vk_device();
#endif
            use_vk_upload_  = true;
            use_cpu_decode_ = false;
            CASPAR_LOG(info) << L"[hap_producer] Vulkan mixer detected, using direct BC texture upload.";
        } else {
            use_cpu_decode_ = true;
            CASPAR_LOG(info) << L"[hap_producer] Non-OGL mixer detected, using CPU DXT decompression.";
        }
        format_desc_ = deps.format_desc;

        demuxer_ = std::make_unique<HapDemuxer>(path_);
        if (!demuxer_->valid())
            CASPAR_THROW_EXCEPTION(std::runtime_error("[hap_producer] Cannot open: " + u8(path_)));

        frame_info_ = demuxer_->frame_info();
        if (frame_info_.width == 0 || frame_info_.height == 0)
            CASPAR_THROW_EXCEPTION(std::runtime_error("[hap_producer] Invalid frame dimensions"));

        CASPAR_LOG(info) << L"[hap_producer] " << frame_info_.width << L"x" << frame_info_.height
                         << (loop_ ? L" LOOP" : L"")
                         << (in_frame_ > 0  ? (L" IN="  + std::to_wstring(in_frame_))  : L"")
                         << (out_frame_ >= 0 ? (L" OUT=" + std::to_wstring(out_frame_)) : L"");

        if (frame_info_.width  != format_desc_.width ||
            frame_info_.height != format_desc_.height) {
            CASPAR_LOG(warning) << L"[hap_producer] File resolution " << frame_info_.width
                                << L"x" << frame_info_.height << L" != channel "
                                << format_desc_.width << L"x" << format_desc_.height
                                << L" -- compositor will scale.";
        }

        // Diagnostics
        graph_ = spl::make_shared<diagnostics::graph>();
        graph_->set_text(print());
        graph_->set_color("frame-time",  diagnostics::color(0.0f, 1.0f, 0.0f));
        graph_->set_color("decode-time", diagnostics::color(0.2f, 0.9f, 0.2f));
        graph_->set_color("queue-fill",  diagnostics::color(0.5f, 0.5f, 1.0f));
        graph_->set_color("fps",         diagnostics::color(1.0f, 0.8f, 0.0f));
        graph_->set_color("dropped",     diagnostics::color(1.0f, 0.3f, 0.3f));
        graph_->auto_reset();
        diagnostics::register_graph(graph_);

        // Probe first frame to detect variant
        {
            auto probe = demuxer_->read_packet();
            if (!probe.is_eof && probe.payload_size() > 0) {
                HapFrameResult probe_result;
                if (parse_hap_frame(probe.payload_data(), probe.payload_size(), probe_result))
                    detected_variant_ = probe_result.variant;
            }
            demuxer_->seek_to_frame(0);
        }

        CASPAR_LOG(info) << L"[hap_producer] Detected variant: "
                         << [&] {
                                switch (detected_variant_) {
                                case HapVariant::Hap:      return L"HAP (DXT1)";
                                case HapVariant::HapAlpha: return L"HAP Alpha (DXT5)";
                                case HapVariant::HapQ:     return L"HAP Q (YCoCg DXT5)";
                                case HapVariant::HapQAlpha:return L"HAP Q Alpha";
                                case HapVariant::HapR:     return L"HAP R (BC7)";
                                default:                   return L"Unknown";
                                }
                            }();

        // Create GL textures on the OGL thread (skip for CPU decode and VK upload)
        if (!use_cpu_decode_ && !use_vk_upload_) {
            int fw = frame_info_.width, fh = frame_info_.height;
            ogl_device_->dispatch_sync([this, fw, fh]() {
                for (int i = 0; i < NUM_SLOTS; i++)
                    decode_slots_[i].output_tex = ogl_device_->create_texture(fw, fh, 4, common::bit_depth::bit8);

#ifdef WIN32
                HGLRC main_hglrc = wglGetCurrentContext();
                hdc_             = wglGetCurrentDC();
                if (main_hglrc && hdc_) {
                    wglMakeCurrent(nullptr, nullptr);
                    shared_hglrc_ = wglCreateContext(hdc_);
                    if (shared_hglrc_) {
                        if (!wglShareLists(main_hglrc, shared_hglrc_)) {
                            CASPAR_LOG(warning) << L"[hap_producer] wglShareLists failed";
                            wglDeleteContext(shared_hglrc_);
                            shared_hglrc_ = nullptr;
                        }
                    }
                    wglMakeCurrent(hdc_, main_hglrc);
                }
#endif
            });
        }

#ifdef WIN32
        if (!use_cpu_decode_ && !use_vk_upload_ && !shared_hglrc_)
            CASPAR_THROW_EXCEPTION(std::runtime_error("[hap_producer] Cannot create shared GL context"));
#endif

        // Reopen demuxer
        demuxer_     = std::make_unique<HapDemuxer>(path_);
        audio_channels_ = demuxer_->audio_channels();
        total_frames_   = demuxer_->total_frames();
        const int64_t dur_us = demuxer_->duration_us();
        total_seconds_ = (dur_us > 0) ? dur_us / 1e6 : 0.0;

        {
            int file_fps_num = 0, file_fps_den = 1;
            demuxer_->frame_rate(file_fps_num, file_fps_den);
            if (file_fps_num > 0 && file_fps_den > 0)
                file_fps_ = static_cast<double>(file_fps_num) / file_fps_den;
            else
                file_fps_ = format_desc_.fps;
            CASPAR_LOG(info) << L"[hap_producer] file_fps=" << file_fps_num << L"/" << file_fps_den
                             << L"  channel_fps=" << format_desc_.fps
                             << L"  total_frames=" << total_frames_
                             << L"  total_seconds=" << total_seconds_;
        }

        if (audio_channels_ > 0 && demuxer_->audio_sample_rate() != 48000) {
            CASPAR_LOG(warning) << L"[hap_producer] Audio sample rate is "
                                << demuxer_->audio_sample_rate()
                                << L" Hz (CasparCG expects 48000 Hz) -- audio may pitch/speed incorrectly.";
        }

        if (in_frame_ > 0) {
            demuxer_->seek_to_frame(in_frame_);
            frame_count_ = in_frame_;
        }

        video_frame_start_ = in_frame_;
        if (speed_.load() < 0.0 && in_frame_ == 0 && total_frames_ > 0)
            video_frame_start_ = total_frames_ - 1;
        if (video_frame_start_ != in_frame_) {
            demuxer_->seek_to_frame(video_frame_start_);
            frame_count_ = video_frame_start_;
        }

        io_thread_      = std::thread([this] { io_loop(); });
        snappy_thread_a_ = std::thread([this] { snappy_loop(); });
        snappy_thread_b_ = std::thread([this] { snappy_loop(); });
        snappy_thread_c_ = std::thread([this] { snappy_loop(); });
        snappy_thread_d_ = std::thread([this] { snappy_loop(); });
        gl_thread_       = std::thread([this] { gl_loop(); });
    }

    ~hap_producer_impl() override
    {
        stop_flag_ = true;
        eof_paused_.store(false);
        queue_cv_.notify_all();
        raw_cv_.notify_all();
        done_cv_.notify_all();
        if (io_thread_.joinable())      io_thread_.join();
        if (snappy_thread_a_.joinable()) snappy_thread_a_.join();
        if (snappy_thread_b_.joinable()) snappy_thread_b_.join();
        if (snappy_thread_c_.joinable()) snappy_thread_c_.join();
        if (snappy_thread_d_.joinable()) snappy_thread_d_.join();
        if (gl_thread_.joinable())       gl_thread_.join();

        // Clean up compressed textures on the OGL thread (only if GL path was used)
        if (ogl_device_) {
            ogl_device_->dispatch_sync([this]() {
                for (int i = 0; i < NUM_SLOTS; i++) {
                    if (decode_slots_[i].compressed_tex)
                        glDeleteTextures(1, &decode_slots_[i].compressed_tex);
                    if (decode_slots_[i].compressed_alpha)
                        glDeleteTextures(1, &decode_slots_[i].compressed_alpha);
                }
            });
        }
    }

    // ── I/O thread: reads packets from disk ──
    void io_loop()
    {
        set_thread_name(L"hap-io");
        try {
        int64_t  io_frame_count = video_frame_start_;
        uint64_t pkt_seq        = 0;

        while (true) {
            const int64_t seek_target = seek_request_.exchange(-1LL);
            if (seek_target >= 0) {
                eof_paused_.store(false, std::memory_order_release);
                {
                    std::lock_guard<std::mutex> lk(raw_mutex_);
                    while (!raw_queue_.empty()) raw_queue_.pop();
                }
                demuxer_->seek_to_frame(seek_target);
                io_frame_count = seek_target;
                pkt_seq        = 0;
                frame_count_.store(seek_target, std::memory_order_release);
                fps_frame_acc_ = 0;
                seek_epoch_.fetch_add(1u, std::memory_order_release);
                raw_cv_.notify_all();
                seek_done_.store(true, std::memory_order_release);
                continue;
            }
            if (stop_flag_) break;

            // Backpressure
            {
                std::unique_lock<std::mutex> lk(raw_mutex_);
                raw_cv_.wait(lk, [this] {
                    return stop_flag_ || (int)raw_queue_.size() < IO_QUEUE_CAP || seek_request_ >= 0;
                });
            }
            if (stop_flag_) break;
            if (seek_request_ >= 0) continue;

            auto pkt = demuxer_->read_packet();
            double current_speed = speed_.load();

            if (pkt.is_eof) {
                if (pingpong_.load()) {
                    speed_.store(-current_speed);
                    io_frame_count = std::max(0LL, frame_count_.load() - 1);
                    demuxer_->seek_to_frame(io_frame_count);
                    continue;
                } else if (loop_) {
                    demuxer_->seek_to_frame(in_frame_);
                    io_frame_count = in_frame_;
                    frame_count_.store(in_frame_, std::memory_order_release);
                    fps_frame_acc_ = 0;
                } else {
                    eof_paused_.store(true, std::memory_order_release);
                    queue_cv_.notify_all();
                    std::unique_lock<std::mutex> lk(raw_mutex_);
                    raw_cv_.wait(lk, [this] { return stop_flag_ || seek_request_ >= 0; });
                }
                continue;
            }
            if (pkt.payload_size() == 0) continue;

            {
                const uint32_t cur_epoch = seek_epoch_.load(std::memory_order_relaxed);
                std::lock_guard<std::mutex> lk(raw_mutex_);
                raw_queue_.push({std::move(pkt), pkt_seq++, cur_epoch});
            }
            raw_cv_.notify_one();

            if (current_speed < 0.0) {
                --io_frame_count;
                if (io_frame_count >= in_frame_)
                    demuxer_->seek_to_frame(io_frame_count);
            } else {
                ++io_frame_count;
            }

            if (current_speed >= 0.0 && out_frame_ >= 0 && io_frame_count >= out_frame_) {
                if (pingpong_.load()) {
                    speed_.store(-current_speed);
                    io_frame_count = out_frame_ - 1;
                    demuxer_->seek_to_frame(io_frame_count);
                } else if (loop_) {
                    demuxer_->seek_to_frame(in_frame_);
                    io_frame_count = in_frame_;
                    frame_count_.store(in_frame_, std::memory_order_release);
                    fps_frame_acc_ = 0;
                } else {
                    eof_paused_.store(true, std::memory_order_release);
                    queue_cv_.notify_all();
                    std::unique_lock<std::mutex> lk(raw_mutex_);
                    raw_cv_.wait(lk, [this] { return stop_flag_ || seek_request_ >= 0; });
                }
            } else if (current_speed < 0.0 && io_frame_count < in_frame_) {
                if (pingpong_.load()) {
                    speed_.store(-current_speed);
                    io_frame_count = in_frame_;
                    demuxer_->seek_to_frame(io_frame_count);
                } else if (loop_) {
                    int64_t target = (out_frame_ >= 0) ? out_frame_ - 1 : std::max(0LL, total_frames_ - 1);
                    demuxer_->seek_to_frame(target);
                    io_frame_count = target;
                    frame_count_.store(target, std::memory_order_release);
                    fps_frame_acc_ = 0;
                } else {
                    eof_paused_.store(true, std::memory_order_release);
                    queue_cv_.notify_all();
                    std::unique_lock<std::mutex> lk(raw_mutex_);
                    raw_cv_.wait(lk, [this] { return stop_flag_ || seek_request_ >= 0; });
                }
            }
        }
        } catch (const std::exception& e) {
            CASPAR_LOG(error) << L"[hap_producer] io_loop exception: " << e.what();
            stop_flag_ = true;
            raw_cv_.notify_all();
            done_cv_.notify_all();
            queue_cv_.notify_all();
        }
    }

    // ── Snappy worker: parse HAP frame + decompress (4 instances) ──
    void snappy_loop()
    {
        set_thread_name(L"hap-snappy");
        try {
        while (true) {
            RawPacket rp;
            {
                std::unique_lock<std::mutex> lk(raw_mutex_);
                raw_cv_.wait(lk, [this] { return stop_flag_ || !raw_queue_.empty(); });
                if (stop_flag_) break;
                rp = std::move(raw_queue_.front());
                raw_queue_.pop();
            }
            raw_cv_.notify_all();

            if (rp.pkt.is_eof) break;

            // Epoch check: discard stale packets
            if (rp.epoch != seek_epoch_.load(std::memory_order_acquire))
                continue;

            // Parse HAP frame header + Snappy decompress
            DecompressedItem item;
            item.seq   = rp.seq;
            item.epoch = rp.epoch;
            item.audio_samples = std::move(rp.pkt.audio_samples);

            if (!parse_hap_frame(rp.pkt.payload_data(), rp.pkt.payload_size(), item.result)) {
                CASPAR_LOG(warning) << L"[hap_producer] Failed to parse/decompress HAP frame";
                continue;
            }

            // Second epoch check after decompression
            if (rp.epoch != seek_epoch_.load(std::memory_order_acquire))
                continue;

            {
                std::lock_guard<std::mutex> lk(done_mutex_);
                done_pq_.push(std::move(item));
            }
            done_cv_.notify_one();
        }
        } catch (const std::exception& e) {
            CASPAR_LOG(error) << L"[hap_producer] snappy_loop exception: " << e.what();
            stop_flag_ = true;
            raw_cv_.notify_all();
            done_cv_.notify_all();
            queue_cv_.notify_all();
        }
    }

    // ── GL thread: upload compressed data + render pass + build frame ──
    void gl_loop()
    {
        set_thread_name(use_vk_upload_ ? L"hap-vk" : (use_cpu_decode_ ? L"hap-cpu" : L"hap-gl"));
        try {

        if (!use_cpu_decode_ && !use_vk_upload_) {
#ifdef WIN32
        if (shared_hglrc_) {
            if (!wglMakeCurrent(hdc_, shared_hglrc_)) {
                CASPAR_LOG(error) << L"[hap_producer] wglMakeCurrent failed";
                return;
            }
        }
#endif

        // Create GL decoder (shaders, FBO, VAO)
        gl_decoder_ = std::make_unique<HapGLDecoder>();

        // Initialize compressed textures for each slot based on detected variant
        {
            HapTextureFormat primary_fmt = HapTextureFormat::RGB_DXT1;
            HapTextureFormat alpha_fmt   = HapTextureFormat::RGB_DXT1;
            switch (detected_variant_) {
            case HapVariant::Hap:       primary_fmt = HapTextureFormat::RGB_DXT1;   break;
            case HapVariant::HapAlpha:  primary_fmt = HapTextureFormat::RGBA_DXT5;  break;
            case HapVariant::HapQ:      primary_fmt = HapTextureFormat::YCoCg_DXT5; break;
            case HapVariant::HapQAlpha:
                primary_fmt = HapTextureFormat::YCoCg_DXT5;
                alpha_fmt   = HapTextureFormat::A_RGTC1;
                break;
            case HapVariant::HapR:      primary_fmt = HapTextureFormat::RGBA_BC7;   break;
            default: break;
            }
            for (int i = 0; i < NUM_SLOTS; i++) {
                gl_decoder_->init_slot(decode_slots_[i],
                                       frame_info_.width, frame_info_.height,
                                       primary_fmt, alpha_fmt);
            }
        }
        } // end if (!use_cpu_decode_ && !use_vk_upload_)

        int                  gl_slot         = 0;
        std::vector<int32_t> audio_accum;
        int                  audio_frame_idx = 0;
        uint32_t             seen_epoch = seek_epoch_.load(std::memory_order_acquire);
        uint64_t             next_seq   = 0;

        while (!stop_flag_) {
            bool        epoch_changed = false;
            DecompressedItem item;
            {
                std::unique_lock<std::mutex> lk(done_mutex_);
                done_cv_.wait(lk, [&] {
                    if (stop_flag_) return true;
                    if (seek_epoch_.load(std::memory_order_acquire) != seen_epoch)
                        return true;
                    // Drain stale-epoch items
                    while (!done_pq_.empty() &&
                           done_pq_.top().epoch != seen_epoch) {
                        const_cast<DecompressedItem&>(done_pq_.top()) = {};
                        done_pq_.pop();
                    }
                    return !done_pq_.empty() &&
                           done_pq_.top().epoch == seen_epoch &&
                           done_pq_.top().seq   == next_seq;
                });
                if (stop_flag_) break;

                const uint32_t cur_epoch = seek_epoch_.load(std::memory_order_acquire);
                if (cur_epoch != seen_epoch) {
                    // Drain entire pq, keep new-epoch items
                    std::priority_queue<DecompressedItem, std::vector<DecompressedItem>, DecompressedItemCmp> fresh_pq;
                    while (!done_pq_.empty()) {
                        auto e = std::move(const_cast<DecompressedItem&>(done_pq_.top()));
                        done_pq_.pop();
                        if (e.epoch == cur_epoch)
                            fresh_pq.push(std::move(e));
                    }
                    done_pq_ = std::move(fresh_pq);
                    seen_epoch = cur_epoch;
                    next_seq   = 0;
                    epoch_changed = true;
                } else {
                    item = std::move(const_cast<DecompressedItem&>(done_pq_.top()));
                    done_pq_.pop();
                }
            }
            done_cv_.notify_all();

            if (epoch_changed) {
                audio_accum.clear();
                audio_frame_idx = 0;
                {
                    std::lock_guard<std::mutex> rlk(queue_mutex_);
                    while (!ready_queue_.empty()) ready_queue_.pop();
                }
                queue_cv_.notify_all();
                continue;
            }

            ++next_seq;

            // Backpressure: wait for mixer
            {
                std::unique_lock<std::mutex> lk(queue_mutex_);
                queue_cv_.wait(lk, [this] {
                    return stop_flag_ || (int)ready_queue_.size() < MAX_QUEUED;
                });
                if (stop_flag_) break;
            }

            // ═══ DECODE ═══
            caspar::timer decode_timer;
            core::draw_frame df;

#ifdef ENABLE_VULKAN
            if (use_vk_upload_) {
                // ── Vulkan direct BC upload path ──
                // HapQAlpha needs two textures which the current zero-copy path
                // doesn't support, so fall through to CPU decode for that variant.
                bool vk_handled = false;

                if (item.result.variant != HapVariant::HapQAlpha) {
                    vk::Format vk_fmt;
                    core::pixel_format pix_fmt;
                    switch (item.result.variant) {
                    case HapVariant::Hap:
                        vk_fmt  = vk::Format::eBc1RgbaUnormBlock;
                        pix_fmt = core::pixel_format::rgba;
                        break;
                    case HapVariant::HapAlpha:
                        vk_fmt  = vk::Format::eBc3UnormBlock;
                        pix_fmt = core::pixel_format::rgba;
                        break;
                    case HapVariant::HapQ:
                        vk_fmt  = vk::Format::eBc3UnormBlock;
                        pix_fmt = core::pixel_format::ycocg_dxt5;
                        break;
                    case HapVariant::HapR:
                        vk_fmt  = vk::Format::eBc7UnormBlock;
                        pix_fmt = core::pixel_format::rgba;
                        break;
                    default:
                        vk_fmt  = vk::Format::eBc1RgbaUnormBlock;
                        pix_fmt = core::pixel_format::rgba;
                        break;
                    }

                    // Upload compressed texture data to Vulkan BC image
                    auto tex_store = std::make_shared<std::vector<uint8_t>>(
                        item.result.texture_data.begin(), item.result.texture_data.end());
                    array<const uint8_t> tex_data(tex_store->data(), tex_store->size(), std::move(tex_store));

                    auto tex_future = vk_device_->copy_compressed_async(
                        tex_data, frame_info_.width, frame_info_.height, vk_fmt);
                    auto vk_tex = tex_future.get();

                    const double ms_decode = decode_timer.elapsed();
                    graph_->set_value("decode-time", ms_decode * format_desc_.fps * 0.5);
                    graph_->set_value("queue-fill",
                        static_cast<double>(ready_queue_.size() + 1) / (MAX_QUEUED + 1));

                    // Audio
                    if (!item.audio_samples.empty())
                        audio_accum.insert(audio_accum.end(),
                                           item.audio_samples.begin(), item.audio_samples.end());
                    const int cadence_len    = (int)format_desc_.audio_cadence.size();
                    const int samples_per_ch = (cadence_len > 0)
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

                    // Build frame with VK texture (zero-copy path via texture_wrapper)
                    core::pixel_format_desc pfd(pix_fmt);
                    pfd.planes.push_back(core::pixel_format_desc::plane(
                        frame_info_.width, frame_info_.height, 4, common::bit_depth::bit8));

                    auto empty_store = std::make_shared<std::vector<uint8_t>>(0);
                    array<const uint8_t> dummy_img(empty_store->data(), 0, std::move(empty_store));
                    std::vector<array<const uint8_t>> img_vec;
                    img_vec.push_back(std::move(dummy_img));

                    auto audio_store = std::make_shared<std::vector<int32_t>>(std::move(frame_audio));
                    array<const int32_t> audio_arr(audio_store->data(), audio_store->size(), std::move(audio_store));

                    auto wrapper = std::make_shared<accelerator::vulkan::texture_wrapper>(vk_tex);

                    df = core::draw_frame(core::const_frame(
                        this,
                        std::move(img_vec),
                        std::move(audio_arr),
                        pfd,
                        wrapper));
                    vk_handled = true;
                }

                if (!vk_handled) {
                    // Fall back to CPU decode for unsupported variants (HapQAlpha)
                    std::vector<uint8_t> bgra_pixels;
                    bool ok = cpu_decode_hap_to_bgra(
                        item.result.variant,
                        item.result.texture_data.data(),
                        item.result.texture_data.size(),
                        item.result.alpha_data.empty() ? nullptr : item.result.alpha_data.data(),
                        item.result.alpha_data.size(),
                        frame_info_.width, frame_info_.height,
                        bgra_pixels);
                    if (!ok) {
                        CASPAR_LOG(warning) << L"[hap_producer] VK fallback CPU decode failed";
                        continue;
                    }

                    const double ms_decode = decode_timer.elapsed();
                    graph_->set_value("decode-time", ms_decode * format_desc_.fps * 0.5);
                    graph_->set_value("queue-fill",
                        static_cast<double>(ready_queue_.size() + 1) / (MAX_QUEUED + 1));

                    if (!item.audio_samples.empty())
                        audio_accum.insert(audio_accum.end(),
                                           item.audio_samples.begin(), item.audio_samples.end());
                    const int cadence_len    = (int)format_desc_.audio_cadence.size();
                    const int samples_per_ch = (cadence_len > 0)
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

                    core::pixel_format_desc pfd(core::pixel_format::bgra);
                    pfd.planes.push_back(core::pixel_format_desc::plane(
                        frame_info_.width, frame_info_.height, 4, common::bit_depth::bit8));

                    auto pixel_store = std::make_shared<std::vector<uint8_t>>(std::move(bgra_pixels));
                    array<const uint8_t> img_data(pixel_store->data(), pixel_store->size(), std::move(pixel_store));
                    std::vector<array<const uint8_t>> img_vec;
                    img_vec.push_back(std::move(img_data));

                    auto audio_store = std::make_shared<std::vector<int32_t>>(std::move(frame_audio));
                    array<const int32_t> audio_arr(audio_store->data(), audio_store->size(), std::move(audio_store));

                    df = core::draw_frame(core::const_frame(
                        this,
                        std::move(img_vec),
                        std::move(audio_arr),
                        pfd));
                }
            } else
#endif // ENABLE_VULKAN
            if (use_cpu_decode_) {
                // ── CPU decode path ──
                std::vector<uint8_t> bgra_pixels;
                bool ok = cpu_decode_hap_to_bgra(
                    item.result.variant,
                    item.result.texture_data.data(),
                    item.result.texture_data.size(),
                    item.result.alpha_data.empty() ? nullptr : item.result.alpha_data.data(),
                    item.result.alpha_data.size(),
                    frame_info_.width,
                    frame_info_.height,
                    bgra_pixels);

                if (!ok) {
                    CASPAR_LOG(warning) << L"[hap_producer] CPU decode failed for frame";
                    continue;
                }

                const double ms_decode = decode_timer.elapsed();
                graph_->set_value("decode-time", ms_decode * format_desc_.fps * 0.5);
                graph_->set_value("queue-fill",
                    static_cast<double>(ready_queue_.size() + 1) / (MAX_QUEUED + 1));

                // Audio
                if (!item.audio_samples.empty())
                    audio_accum.insert(audio_accum.end(),
                                       item.audio_samples.begin(), item.audio_samples.end());
                const int cadence_len    = (int)format_desc_.audio_cadence.size();
                const int samples_per_ch = (cadence_len > 0)
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

                // Build frame with pixel data (standard path)
                core::pixel_format_desc pfd(core::pixel_format::bgra);
                pfd.planes.push_back(core::pixel_format_desc::plane(
                    frame_info_.width, frame_info_.height, 4, common::bit_depth::bit8));

                auto pixel_store = std::make_shared<std::vector<uint8_t>>(std::move(bgra_pixels));
                array<const uint8_t> img_data(pixel_store->data(), pixel_store->size(), std::move(pixel_store));
                std::vector<array<const uint8_t>> img_vec;
                img_vec.push_back(std::move(img_data));

                auto audio_store = std::make_shared<std::vector<int32_t>>(std::move(frame_audio));
                array<const int32_t> audio_arr(audio_store->data(), audio_store->size(), std::move(audio_store));

                df = core::draw_frame(core::const_frame(
                    this,
                    std::move(img_vec),
                    std::move(audio_arr),
                    pfd));
            } else {
                // ── GL decode path ──
                HapDecodeSlot& slot = decode_slots_[gl_slot];

                // Validate decompressed data size before GL upload
                {
                    size_t expected_tex = dxt_data_size(slot.width, slot.height, slot.compressed_format);
                    if (item.result.texture_data.size() != expected_tex) {
                        CASPAR_LOG(warning) << L"[hap_producer] Texture data size mismatch: got "
                                            << item.result.texture_data.size() << L", expected " << expected_tex;
                        continue;
                    }
                    if (item.result.variant == HapVariant::HapQAlpha && slot.compressed_alpha) {
                        size_t expected_alpha = dxt_data_size(slot.width, slot.height, slot.alpha_format);
                        if (item.result.alpha_data.size() != expected_alpha) {
                            CASPAR_LOG(warning) << L"[hap_producer] Alpha data size mismatch: got "
                                                << item.result.alpha_data.size() << L", expected " << expected_alpha;
                            continue;
                        }
                    }
                }

                gl_decoder_->decode(slot, item.result.variant,
                                    item.result.texture_data.data(),
                                    item.result.texture_data.size(),
                                    item.result.alpha_data.empty() ? nullptr : item.result.alpha_data.data(),
                                    item.result.alpha_data.size());

                const double ms_decode = decode_timer.elapsed();
                graph_->set_value("decode-time", ms_decode * format_desc_.fps * 0.5);
                graph_->set_value("queue-fill",
                    static_cast<double>(ready_queue_.size() + 1) / (MAX_QUEUED + 1));

                // Audio
                if (!item.audio_samples.empty())
                    audio_accum.insert(audio_accum.end(),
                                       item.audio_samples.begin(), item.audio_samples.end());
                const int cadence_len    = (int)format_desc_.audio_cadence.size();
                const int samples_per_ch = (cadence_len > 0)
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

                // Build frame with GL texture (zero-copy path)
                core::pixel_format_desc pfd(core::pixel_format::rgba);
                pfd.planes.push_back(core::pixel_format_desc::plane(
                    frame_info_.width, frame_info_.height, 4, common::bit_depth::bit8));

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
                    slot.output_tex));

                gl_slot = (gl_slot + 1) % NUM_SLOTS;
            }

            { std::lock_guard<std::mutex> lk(queue_mutex_); ready_queue_.push(std::move(df)); }
            queue_cv_.notify_one();
        }

        // Cleanup
        gl_decoder_.reset();

#ifdef WIN32
        if (shared_hglrc_) {
            wglMakeCurrent(nullptr, nullptr);
            wglDeleteContext(shared_hglrc_);
            shared_hglrc_ = nullptr;
        }
#endif
        } catch (const std::exception& e) {
            CASPAR_LOG(error) << L"[hap_producer] gl_loop exception: " << e.what();
            stop_flag_ = true;
            raw_cv_.notify_all();
            done_cv_.notify_all();
            queue_cv_.notify_all();
            gl_decoder_.reset();
#ifdef WIN32
            if (shared_hglrc_) {
                wglMakeCurrent(nullptr, nullptr);
                wglDeleteContext(shared_hglrc_);
                shared_hglrc_ = nullptr;
            }
#endif
        }
    }

    // ── AMCP call handler ──
    std::future<std::wstring> call(const std::vector<std::wstring>& params) override
    {
        std::wstring result;
        if (!params.empty()) {
            const auto& cmd = params[0];
            const std::wstring val = params.size() > 1 ? params[1] : L"";

            if (boost::iequals(cmd, L"loop")) {
                if (!val.empty()) loop_ = boost::lexical_cast<bool>(val);
                result = loop_ ? L"1" : L"0";

            } else if (boost::iequals(cmd, L"pingpong")) {
                bool pp = pingpong_.load();
                if (!val.empty()) pp = boost::lexical_cast<bool>(val);
                pingpong_.store(pp);
                result = pp ? L"1" : L"0";

            } else if (boost::iequals(cmd, L"speed")) {
                double spd = speed_.load();
                if (!val.empty()) spd = boost::lexical_cast<double>(val);
                speed_.store(spd);
                result = boost::lexical_cast<std::wstring>(spd);
                if (eof_paused_.load() && spd != 0.0) {
                    const int64_t resume_at = frame_count_.load();
                    seek_request_ = resume_at;
                    raw_cv_.notify_one();
                }

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

                if (params.size() > 2)
                    target += boost::lexical_cast<int64_t>(params[2]);

                target = std::max(0LL, target);
                if (total_frames_ > 0)
                    target = std::min(target, total_frames_ - 1);

                seek_request_ = target;
                queue_cv_.notify_one();
                raw_cv_.notify_one();
                result = std::to_wstring(target);

            } else {
                CASPAR_LOG(warning) << L"[hap_producer] CALL: unknown '" << cmd
                                    << L"'. Supported: loop, pingpong, speed, seek";
            }
        }
        std::promise<std::wstring> p;
        p.set_value(result);
        return p.get_future();
    }

    // ── receive_impl: called by mixer at channel frame rate ──
    core::draw_frame receive_impl(const core::video_field field, int) override
    {
        graph_->set_value("frame-time", frame_timer_.elapsed() * format_desc_.hz * 0.5);
        frame_timer_.restart();

        if (field == core::video_field::b)
            return cached_frame_ ? cached_frame_ : core::draw_frame{};

        std::unique_lock<std::mutex> lk(queue_mutex_);
        if (!eof_paused_) {
            queue_cv_.wait_for(lk, std::chrono::milliseconds(40),
                               [this] { return !ready_queue_.empty() || stop_flag_ || eof_paused_; });
        }
        if (ready_queue_.empty()) return cached_frame_;

        double spd = speed_.load();
        const double fps_ratio = (file_fps_ > 0.0 && format_desc_.fps > 0.0)
                                     ? file_fps_ / format_desc_.fps
                                     : 1.0;
        speed_accum_ += std::abs(spd) * fps_ratio;
        int frames_to_advance = static_cast<int>(speed_accum_);
        speed_accum_ -= static_cast<double>(frames_to_advance);

        if (frames_to_advance == 0) {
            if (seek_done_.exchange(false, std::memory_order_relaxed) && !ready_queue_.empty()) {
                cached_frame_ = std::move(ready_queue_.front());
                ready_queue_.pop();
                lk.unlock();
                queue_cv_.notify_all();
                return cached_frame_;
            }
            lk.unlock();
            return cached_frame_ ? core::draw_frame::still(cached_frame_) : core::draw_frame{};
        }

        int actually_consumed = 1;
        for (int i = 1; i < frames_to_advance && ready_queue_.size() > 1; ++i) {
            ready_queue_.pop();
            ++actually_consumed;
        }

        cached_frame_ = std::move(ready_queue_.front());
        ready_queue_.pop();
        lk.unlock();
        queue_cv_.notify_all();

        auto fc = (frame_count_ += (spd >= 0.0) ? actually_consumed : -actually_consumed);

        fps_frame_acc_ += actually_consumed;
        const double fps_elapsed = fps_window_timer_.elapsed();
        if (fps_elapsed >= 1.0) {
            fps_display_   = fps_frame_acc_ / fps_elapsed;
            fps_frame_acc_ = 0;
            fps_window_timer_.restart();
        }

        const double cur_sec = (file_fps_ > 0.0) ? static_cast<double>(fc) / file_fps_ : 0.0;
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

        const double target_fps = file_fps_ * std::abs(spd);
        if (target_fps > 0.0 && fps_display_ > 0.0)
            graph_->set_value("fps", std::min(1.0, fps_display_ / target_fps));

        return cached_frame_;
    }

    bool is_ready() override { std::lock_guard<std::mutex> lk(queue_mutex_); return !ready_queue_.empty(); }

    // Called by the layer on every tick when the layer is paused
    // (paused_=true in layer.cpp).  In that state, receive_impl() is NEVER
    // called, so cached_frame_ is never populated through normal playback.
    //
    // We pop from ready_queue_ in two cases:
    //  1. seek_done_ is true  – after an explicit SEEK while paused.
    //  2. cached_frame_ is empty – initial LOAD/preview: the decode thread
    //     has filled ready_queue_ but receive_impl() was never called because
    //     the layer went straight to paused.  Without this, LOAD produces a
    //     black frame since cached_frame_ starts empty.
    core::draw_frame last_frame(const core::video_field) override
    {
        if (seek_done_.load(std::memory_order_relaxed) || !cached_frame_) {
            std::unique_lock<std::mutex> lk(queue_mutex_);
            if (!ready_queue_.empty()) {
                seek_done_.store(false, std::memory_order_relaxed);
                cached_frame_ = std::move(ready_queue_.front());
                ready_queue_.pop();
                lk.unlock();
                queue_cv_.notify_all();

                auto fc = frame_count_.load(std::memory_order_relaxed);
                const double cur_sec = (file_fps_ > 0.0) ? static_cast<double>(fc) / file_fps_ : 0.0;
                std::wstring title = boost::filesystem::path(path_).filename().wstring();
                title += L"  " + std::to_wstring(fc);
                if (total_frames_ > 0)
                    title += L" / " + std::to_wstring(total_frames_);
                title += L"  |  " + [&]{ std::wostringstream s; s << std::fixed << std::setprecision(1) << cur_sec; return s.str(); }();
                if (total_seconds_ > 0.0)
                    title += L"s / " + [&]{ std::wostringstream s; s << std::fixed << std::setprecision(1) << total_seconds_; return s.str(); }() + L"s";
                else
                    title += L"s";
                graph_->set_text(title);
            }
        }
        if (!cached_frame_) return core::draw_frame{};
        return core::draw_frame::still(cached_frame_);
    }

    uint32_t frame_number() const override
    {
        if (file_fps_ > 0.0 && format_desc_.fps > 0.0 && file_fps_ != format_desc_.fps) {
            double scaled = static_cast<double>(frame_count_.load()) * (format_desc_.fps / file_fps_);
            return static_cast<uint32_t>(std::llround(scaled));
        }
        return static_cast<uint32_t>(frame_count_.load());
    }

    core::monitor::state state() const override
    {
        const double cur_sec = (file_fps_ > 0.0) ? static_cast<double>(frame_count_) / file_fps_ : 0.0;
        monitor_state_["file/name"]  = u8(boost::filesystem::path(path_).filename().wstring());
        monitor_state_["file/path"]  = u8(path_);
        monitor_state_["file/time"]  = {cur_sec, total_seconds_};
        monitor_state_["file/loop"]  = loop_;
        monitor_state_["width"]      = frame_info_.width;
        monitor_state_["height"]     = frame_info_.height;
        return monitor_state_;
    }

    std::wstring print() const override { return L"hap[" + path_ + L"]"; }
    std::wstring name()  const override { return L"hap"; }
};

// ---------------------------------------------------------------------------
// Factory / registration
// ---------------------------------------------------------------------------

spl::shared_ptr<core::frame_producer>
create_hap_producer(const core::frame_producer_dependencies& deps,
                    const std::vector<std::wstring>& params)
{
    if (params.empty() || !boost::iequals(params[0], L"HAP"))
        return core::frame_producer::empty();

    std::wstring path;
    bool    loop          = false;
    bool    pingpong_flag = false;
    double  initial_speed = 1.0;
    int64_t start_frame   = 0, out_frame = -1, length_param = -1;

    for (size_t i = 1; i < params.size(); ++i) {
        const auto& p = params[i];
        if      (boost::iequals(p, L"LOOP"))                              { loop = true; }
        else if (boost::iequals(p, L"PINGPONG"))                          { pingpong_flag = true; }
        else if (boost::iequals(p, L"SPEED") && i+1 < params.size())
            initial_speed = boost::lexical_cast<double>(params[++i]);
        else if (boost::iequals(p, L"FILE") && i+1 < params.size()) path = params[++i];
        else if ((boost::iequals(p, L"SEEK") || boost::iequals(p, L"IN") || boost::iequals(p, L"START"))
                 && i+1 < params.size())
            start_frame = boost::lexical_cast<int64_t>(params[++i]);
        else if (boost::iequals(p, L"LENGTH") && i+1 < params.size())
            length_param = boost::lexical_cast<int64_t>(params[++i]);
        else if (boost::iequals(p, L"OUT") && i+1 < params.size())
            out_frame = boost::lexical_cast<int64_t>(params[++i]);
        else if (path.empty() && !p.empty() && p[0] != L'-')
            path = p;
    }

    if (length_param >= 0 && out_frame < 0)
        out_frame = start_frame + length_param;
    if (path.empty()) return core::frame_producer::empty();

    {
        auto is_valid = [](const boost::filesystem::path& p) {
            auto ext = boost::to_lower_copy(p.extension().wstring());
            return ext == L".mov" || ext == L".mp4" || ext == L".avi";
        };
        auto resolved = find_file_within_dir_or_absolute(env::media_folder(), path, is_valid);
        if (!resolved) {
            CASPAR_LOG(error) << L"[hap_producer] File not found: " << path;
            return core::frame_producer::empty();
        }
        path = resolved->wstring();
    }

    try {
        return spl::make_shared<hap_producer_impl>(
            path, loop, pingpong_flag,
            start_frame, out_frame, initial_speed, deps);
    } catch (const std::exception& ex) {
        CASPAR_LOG(error) << L"[hap_producer] " << ex.what();
        return core::frame_producer::empty();
    }
}

void register_hap_producer(const core::module_dependencies& module_deps)
{
    module_deps.producer_registry->register_producer_factory(
        L"HAP Producer", &create_hap_producer);
}

}} // namespace caspar::hap
