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
 * NotchLC is a codec specification by Derivative Inc., available under the
 * Creative Commons Attribution 4.0 International License.
 */

// notchlc_producer.cpp
// CasparCG frame_producer for CUDA NotchLC decode.
// Architecture mirrors cuda_prores/producer/prores_producer.cpp exactly;
// differences are documented inline.
#include "notchlc_producer.h"

#include "notchlc_demuxer.h"
#include "../cuda/notchlc_decode.h"
#include "../cuda/notchlc_ycocg_to_bgra16.cuh"   // NOTCHLC_CM_* constants
#include "../util/cuda_gl_texture.h"
#include "../../cuda_gl_interop_lock.h"

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

namespace caspar { namespace cuda_notchlc {

// 5 slots: perfectly balances 2 slots in ready_queue_, 1 in active render, and 2 available for LZ4 threads,
// keeping memory footprints low to prevent Windows SysMem paging when multiple playbacks overlap.
static constexpr int NUM_SLOTS     = 5;
static constexpr int MAX_QUEUED    = 2;   // decoded frames buffered for renderer
static constexpr int IO_QUEUE_CAP  = 2;  // raw packets buffered from disk

// Item passed from io_loop  LZ4 workers  GPU thread.
struct RawPacket {
    NotchLCPacket pkt;
    uint64_t      seq     = 0;    // monotonic sequence number (reset on seek)
    uint32_t      epoch   = 0;    // seek epoch when packet was queued
};

// Item passed from LZ4 workers  GPU thread after CPU decompression.
struct Lz4DoneItem {
    uint64_t         seq     = 0;    // sequence number for reorder buffer
    uint32_t         epoch   = 0;    // seek epoch
    int              slot    = -1;
    NotchBlockHeader hdr     = {};
    size_t           actual_uncompressed = 0;
    std::vector<int32_t> audio_samples;
};

// Min-heap comparator: ensures GPU thread always gets the lowest-seq item first.
struct Lz4DoneItemCmp {
    bool operator()(const Lz4DoneItem& a, const Lz4DoneItem& b) const
    { return a.seq > b.seq; }
};

struct notchlc_producer_impl final : public core::frame_producer
{
    const std::wstring                        path_;
    const int                                 cuda_device_;
    bool                                      loop_                   = false;
    // -1=AUTO(709); 1=BT.709; 6=BT.601; 9=BT.2020; 100=LINEAR
    int                                       color_matrix_override_  = -1;
    spl::shared_ptr<core::frame_factory>      frame_factory_;
    std::shared_ptr<accelerator::ogl::device> ogl_device_;
    core::video_format_desc                   format_desc_;
    int                                       audio_channels_         = 0;

    std::unique_ptr<NotchLCDemuxer>           demuxer_;
    NotchLCFrameInfo                          frame_info_;
    int                                       num_slots_              = NUM_SLOTS;  // runtime slot count

    NotchLCDecodeCtx                          slots_     [NUM_SLOTS];
    bool                                      slots_init_[NUM_SLOTS] = {};

    std::shared_ptr<accelerator::ogl::texture> gl_tex_[NUM_SLOTS];
    std::shared_ptr<CudaGLTexture>             cgt_   [NUM_SLOTS];

#ifdef WIN32
    HDC   hdc_          = nullptr;
    HGLRC shared_hglrc_ = nullptr;
#endif

    uint16_t* h_bgra16_[NUM_SLOTS]  = {};
    bool      use_host_copy_        = false;

    spl::shared_ptr<diagnostics::graph>       graph_;
    core::draw_frame                          cached_frame_;

    std::queue<core::draw_frame>              ready_queue_;
    std::mutex                                queue_mutex_;
    std::condition_variable                   queue_cv_;

    std::thread                               read_thread_;
    std::thread                               io_thread_;
    std::thread                               lz4_thread_a_;
    std::thread                               lz4_thread_b_;
    std::thread                               lz4_thread_c_;
    std::thread                               lz4_thread_d_;

    // io_thread  LZ4 workers
    std::queue<RawPacket>                     raw_queue_;
    std::mutex                                raw_mutex_;
    std::condition_variable                   raw_cv_;
    static constexpr int                      RAW_QUEUE_CAP = 6;

    // LZ4 workers  GPU thread: priority queue (min-heap on seq) acts as
    // reorder buffer  GPU thread consumes frames strictly in sequence order
    // regardless of which worker finishes LZ4 first.
    std::priority_queue<Lz4DoneItem, std::vector<Lz4DoneItem>, Lz4DoneItemCmp>
                                              lz4_done_pq_;
    std::mutex                                lz4_done_mutex_;
    std::condition_variable                   lz4_done_cv_;

    // Available slot pool: LZ4 workers pick a slot, GPU thread returns it.
    std::queue<int>                           slot_pool_;
    std::mutex                                slot_pool_mutex_;
    std::condition_variable                   slot_pool_cv_;

    std::atomic<uint32_t>                     seek_epoch_{0};
    std::atomic_bool                          stop_flag_{false};
    std::atomic_bool                          seek_done_{false};   // force pop in last_frame()/receive_impl after seek while paused
    std::atomic_bool                          eof_paused_{false};  // set at EOF boundaries; keeps threads alive waiting for seek

    std::atomic<int64_t>                      frame_count_{0};
    int64_t                                   total_frames_   = -1;
    double                                    total_seconds_  = 0.0;
    double                                    file_fps_       = 0.0;  // native file frame rate
    caspar::timer                             frame_timer_;
    caspar::timer                             fps_window_timer_;
    int                                       fps_frame_acc_  = 0;
    double                                    fps_display_    = 0.0;

    std::atomic<int64_t>                      seek_request_{-1LL};
    int64_t                                   in_frame_          = 0;
    int64_t                                   out_frame_         = -1;
    int64_t                                   video_frame_start_ = 0;  // initial read position (may differ from in_frame_ for negative speed)

    std::atomic<double>                       speed_{1.0};
    std::atomic<bool>                         pingpong_{false};
    double                                    speed_accum_{0.0};

    mutable core::monitor::state              monitor_state_;

    notchlc_producer_impl(const std::wstring& path, int cuda_device, bool loop,
                          bool pingpong, int color_matrix_override,
                          int64_t in_frame, int64_t out_frame, double initial_speed,
                          const core::frame_producer_dependencies& deps)
        : path_(path), cuda_device_(cuda_device), loop_(loop)
        , color_matrix_override_(color_matrix_override)
        , in_frame_(in_frame), out_frame_(out_frame)
        , speed_(initial_speed), pingpong_(pingpong)
        , frame_factory_(deps.frame_factory)
    {
        auto* ogl_mixer = dynamic_cast<accelerator::ogl::image_mixer*>(frame_factory_.get());
        if (!ogl_mixer)
            CASPAR_THROW_EXCEPTION(std::runtime_error("[notchlc_producer] frame_factory is not ogl::image_mixer"));

        ogl_device_  = ogl_mixer->get_ogl_device();
        format_desc_ = deps.format_desc;

        cudaSetDevice(cuda_device_);

        demuxer_ = std::make_unique<NotchLCDemuxer>(path_);
        if (!demuxer_->valid())
            CASPAR_THROW_EXCEPTION(std::runtime_error("[notchlc_producer] Cannot open: " + u8(path_)));

        frame_info_ = demuxer_->frame_info();
        if (frame_info_.width == 0 || frame_info_.height == 0)
            CASPAR_THROW_EXCEPTION(std::runtime_error("[notchlc_producer] Invalid frame dimensions"));

        CASPAR_LOG(info) << L"[notchlc_producer] " << frame_info_.width << L"x" << frame_info_.height
                         << L" color_matrix=" << color_matrix_override_
                         << (loop_      ? L" LOOP" : L"")
                         << (in_frame_  > 0  ? (L" IN="  + std::to_wstring(in_frame_))  : L"")
                         << (out_frame_ >= 0 ? (L" OUT=" + std::to_wstring(out_frame_)) : L"");

        if (frame_info_.width  != format_desc_.width ||
            frame_info_.height != format_desc_.height) {
            CASPAR_LOG(warning) << L"[notchlc_producer] File resolution " << frame_info_.width
                                << L"x" << frame_info_.height << L" != channel "
                                << format_desc_.width << L"x" << format_desc_.height
                                << L" -- compositor will scale.";
        }

        //  Diagnostics 
        graph_ = spl::make_shared<diagnostics::graph>();
        graph_->set_text(print());
        graph_->set_color("frame-time",  diagnostics::color(0.0f, 1.0f, 0.0f));
        graph_->set_color("decode-time", diagnostics::color(0.2f, 0.9f, 0.2f));
        graph_->set_color("queue-fill",  diagnostics::color(0.5f, 0.5f, 1.0f));
        graph_->set_color("fps",         diagnostics::color(1.0f, 0.8f, 0.0f));
        graph_->set_color("dropped",     diagnostics::color(1.0f, 0.3f, 0.3f));
        graph_->auto_reset();
        diagnostics::register_graph(graph_);

        //  CUDA decode contexts 
        // Probe the first packet to get accurate compressed / uncompressed sizes
        // instead of guessing from frame dimensions.  The naive guess (max_pixels*6)
        // produces 453 MB per slot for 12K, causing WDDM GPU-memory paging and
        // catastrophic performance (100 slowdown) on GPUs with <16 GB VRAM.
        const size_t max_pixels = (size_t)frame_info_.width * frame_info_.height;
        size_t max_compressed   = max_pixels * 2;   // fallback (2 bytes/pixel)
        size_t max_uncompressed = max_pixels * 2;   // fallback
        {
            auto probe = demuxer_->read_packet();
            if (!probe.is_eof && probe.payload_size() > 0 && probe.uncompressed_size > 0) {
                max_compressed   = probe.payload_size()            * 12 / 10 + 65536;  // +20% + 64 KB
                max_uncompressed = (size_t)probe.uncompressed_size * 12 / 10 + 65536;
            }
            demuxer_->seek_to_frame(0);  // rewind so the read loop starts from the beginning
        }

        //  Slot Configuration 
        
        num_slots_ = NUM_SLOTS;

        CASPAR_LOG(info) << L"[notchlc_producer] max_compressed="
                         << max_compressed / (1024*1024) << L" MB  max_uncompressed="
                         << max_uncompressed / (1024*1024) << L" MB  slots="
                         << num_slots_;

        for (int i = 0; i < num_slots_; i++) {
            cudaError_t e = notchlc_decode_ctx_create(&slots_[i],
                frame_info_.width, frame_info_.height,
                max_compressed, max_uncompressed);
            if (e != cudaSuccess)
                CASPAR_THROW_EXCEPTION(std::runtime_error(
                    std::string("[notchlc_producer] notchlc_decode_ctx_create: ") + cudaGetErrorString(e)));
            slots_init_[i] = true;
        }

        // Pre-fill the slot pool with all created slot indices.
        for (int i = 0; i < num_slots_; i++) slot_pool_.push(i);

        //  GL textures + shared WGL context (Windows zero-copy path) 
        {
            int fw = frame_info_.width, fh = frame_info_.height;
            ogl_device_->dispatch_sync([this, fw, fh]() {
                for (int i = 0; i < num_slots_; i++)
                    gl_tex_[i] = ogl_device_->create_texture(fw, fh, 4, common::bit_depth::bit16);

#ifdef WIN32
                HGLRC main_hglrc = wglGetCurrentContext();
                hdc_             = wglGetCurrentDC();
                if (main_hglrc && hdc_) {
                    wglMakeCurrent(nullptr, nullptr);
                    shared_hglrc_ = wglCreateContext(hdc_);
                    if (shared_hglrc_) {
                        if (!wglShareLists(main_hglrc, shared_hglrc_)) {
                            CASPAR_LOG(warning) << L"[notchlc_producer] wglShareLists failed"
                                                << L" -- falling back to host-copy path";
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
        use_host_copy_ = (shared_hglrc_ == nullptr);
#else
        use_host_copy_ = true;
#endif
        if (use_host_copy_) {
            CASPAR_LOG(info) << L"[notchlc_producer] Using host-copy upload path";
            const size_t out_bytes = max_pixels * 4 * sizeof(uint16_t);
            for (int i = 0; i < num_slots_; i++) {
                cudaError_t e = cudaMallocHost(&h_bgra16_[i], out_bytes);
                if (e != cudaSuccess)
                    CASPAR_THROW_EXCEPTION(std::runtime_error(
                        std::string("[notchlc_producer] cudaMallocHost: ") + cudaGetErrorString(e)));
            }
        }

        // Reopen demuxer from beginning, then apply IN seek if requested.
        demuxer_     = std::make_unique<NotchLCDemuxer>(path_);
        audio_channels_ = demuxer_->audio_channels();

        total_frames_  = demuxer_->total_frames();
        const int64_t dur_us = demuxer_->duration_us();
        total_seconds_ = (dur_us > 0) ? dur_us / 1e6 : 0.0;

        {
            int file_fps_num = 0, file_fps_den = 1;
            demuxer_->frame_rate(file_fps_num, file_fps_den);
            if (file_fps_num > 0 && file_fps_den > 0)
                file_fps_ = static_cast<double>(file_fps_num) / file_fps_den;
            else
                file_fps_ = format_desc_.fps;
            CASPAR_LOG(info) << L"[notchlc_producer] file_fps=" << file_fps_num << L"/" << file_fps_den
                             << L"  channel_fps=" << format_desc_.fps
                             << L"  total_frames=" << total_frames_
                             << L"  total_seconds=" << total_seconds_;
        }

        if (audio_channels_ > 0 && demuxer_->audio_sample_rate() != 48000) {
            CASPAR_LOG(warning) << L"[notchlc_producer] Audio sample rate is "
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

        io_thread_    = std::thread([this] { io_loop(); });
        lz4_thread_a_ = std::thread([this] { lz4_loop(); });
        lz4_thread_b_ = std::thread([this] { lz4_loop(); });
        lz4_thread_c_ = std::thread([this] { lz4_loop(); });
        lz4_thread_d_ = std::thread([this] { lz4_loop(); });
        read_thread_  = std::thread([this] { read_loop(); });
    }

    ~notchlc_producer_impl() override {
        stop_flag_ = true;
        eof_paused_.store(false);
        queue_cv_.notify_all();
        raw_cv_.notify_all();
        lz4_done_cv_.notify_all();
        slot_pool_cv_.notify_all();
        if (io_thread_.joinable())    io_thread_.join();
        if (lz4_thread_a_.joinable()) lz4_thread_a_.join();
        if (lz4_thread_b_.joinable()) lz4_thread_b_.join();
        if (lz4_thread_c_.joinable()) lz4_thread_c_.join();
        if (lz4_thread_d_.joinable()) lz4_thread_d_.join();
        if (read_thread_.joinable())  read_thread_.join();
        for (int i = 0; i < num_slots_; i++) {
            cudaFreeHost(h_bgra16_[i]);
            if (slots_init_[i]) notchlc_decode_ctx_destroy(&slots_[i]);
        }
    }

    //  I/O thread: reads packets from disk into raw_queue_ 
    void io_loop()
    {
        set_thread_name(L"notchlc-io");
        int64_t  io_frame_count = video_frame_start_;
        uint64_t pkt_seq        = 0;  // sequence number assigned to each outgoing packet

        while (true) {
            //  External seek 
            const int64_t seek_target = seek_request_.exchange(-1LL);
            if (seek_target >= 0) {
                eof_paused_.store(false, std::memory_order_release);
                {
                    std::lock_guard<std::mutex> lk(raw_mutex_);
                    while (!raw_queue_.empty()) raw_queue_.pop();
                }
                demuxer_->seek_to_frame(seek_target);
                io_frame_count = seek_target;
                pkt_seq        = 0;  // reset seq after seek
                frame_count_.store(seek_target, std::memory_order_release);
                fps_frame_acc_ = 0;
                seek_epoch_.fetch_add(1u, std::memory_order_release);
                raw_cv_.notify_all();
                seek_done_.store(true, std::memory_order_release);
                continue;
            }
            if (stop_flag_) break;

            //  Backpressure: wait while raw_queue_ is full 
            {
                std::unique_lock<std::mutex> lk(raw_mutex_);
                raw_cv_.wait(lk, [this] {
                    return stop_flag_ || (int)raw_queue_.size() < RAW_QUEUE_CAP || seek_request_ >= 0;
                });
            }
            if (stop_flag_) break;
            if (seek_request_ >= 0) continue;

            //  Read packet from disk 
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
                    // NOTE: pkt_seq NOT reset on loop; stays monotonic
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
                    // NOTE: pkt_seq NOT reset on loop; stays monotonic
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
    }

    //  LZ4 worker: CPU decompress + header parse (3 instances run in parallel) 
    void lz4_loop()
    {
        set_thread_name(L"notchlc-lz4");

        while (true) {
            //  Pop a raw packet (earliest-woken worker wins the race for oldest packet) 
            RawPacket rp;
            {
                std::unique_lock<std::mutex> lk(raw_mutex_);
                raw_cv_.wait(lk, [this] {
                    return stop_flag_ || !raw_queue_.empty();
                });
                if (stop_flag_) break;
                rp = std::move(raw_queue_.front());
                raw_queue_.pop();
            }
            raw_cv_.notify_one();  // wake io_thread: space in raw_queue_

            //  EOF propagation (no slot needed) 
            if (rp.pkt.is_eof) {
                Lz4DoneItem eof_item;
                eof_item.seq   = rp.seq;
                eof_item.epoch = rp.epoch;
                eof_item.slot  = -1;
                {
                    std::lock_guard<std::mutex> lk(lz4_done_mutex_);
                    if (!stop_flag_) lz4_done_pq_.push(std::move(eof_item));
                }
                lz4_done_cv_.notify_one();
                break;
            }

            //  Claim a slot 
            int slot = -1;
            {
                std::unique_lock<std::mutex> lk(slot_pool_mutex_);
                slot_pool_cv_.wait(lk, [this] {
                    return stop_flag_ || !slot_pool_.empty();
                });
                if (stop_flag_) break;
                slot = slot_pool_.front();
                slot_pool_.pop();
            }

            //  LZ4 decompression (pure CPU, this slot's h_task_buf / h_uncompressed) 
            NotchLCDecodeCtx& ctx = slots_[slot];
            NotchBlockHeader hdr = {};
            size_t actual_uncompressed = 0;

            cudaError_t err = notchlc_decode_cpu_phase(
                &ctx,
                rp.pkt.payload_data(), rp.pkt.payload_size(),
                rp.pkt.uncompressed_size, rp.pkt.format,
                hdr, actual_uncompressed);

            if (err != cudaSuccess) {
                CASPAR_LOG(warning) << L"[notchlc_producer] LZ4 cpu_phase failed slot=" << slot
                                    << L": " << cudaGetErrorString(err);
                // Return slot to pool (frame dropped).
                { std::lock_guard<std::mutex> slk(slot_pool_mutex_); slot_pool_.push(slot); }
                slot_pool_cv_.notify_one();
                continue;
            }

            //  Epoch check: discard stale frames from before a seek 
            if (rp.epoch != seek_epoch_.load(std::memory_order_acquire)) {
                std::lock_guard<std::mutex> slk(slot_pool_mutex_);
                slot_pool_.push(slot);
                slot_pool_cv_.notify_one();
                continue;  // packet is from old epoch; drop it
            }

            //  Push to reorder buffer (min-heap on seq) 
            // NOTE: No capacity limit here  blocking while holding a slot while the
            // GPU thread waits for a different seq causes a deadlock.  The slot pool
            // (num_slots_ entries) is the natural backpressure: when all slots are
            // in use the lz4 workers block on slot_pool_cv_ before even starting work.
            {
                std::unique_lock<std::mutex> lk(lz4_done_mutex_);
                if (stop_flag_) {
                    std::lock_guard<std::mutex> slk(slot_pool_mutex_);
                    slot_pool_.push(slot);
                    slot_pool_cv_.notify_one();
                    break;
                }
                Lz4DoneItem item;
                item.seq                 = rp.seq;
                item.epoch               = rp.epoch;
                item.slot                = slot;
                item.hdr                 = hdr;
                item.actual_uncompressed = actual_uncompressed;
                item.audio_samples       = std::move(rp.pkt.audio_samples);
                lz4_done_pq_.push(std::move(item));
            }
            lz4_done_cv_.notify_one();
        }
    }

    //  GPU thread: HD upload + GPU kernels + frame build 
    void read_loop()
    {
        cudaSetDevice(cuda_device_);
        set_thread_name(L"notchlc-dec");

#ifdef WIN32
        if (!use_host_copy_ && shared_hglrc_) {
            if (!wglMakeCurrent(hdc_, shared_hglrc_)) {
                CASPAR_LOG(error) << L"[notchlc_producer] wglMakeCurrent on read_thread_ failed"
                                  << L" -- switching to host-copy fallback";
                use_host_copy_ = true;
            } else {
                try {
                    std::lock_guard<std::mutex> gl_lk(caspar::cuda_gl_interop_mutex());
                    for (int i = 0; i < num_slots_; i++)
                        cgt_[i] = std::make_shared<CudaGLTexture>(gl_tex_[i]);
                    CASPAR_LOG(info) << L"[notchlc_producer] CUDA-GL interop active";
                } catch (const std::exception& ex) {
                    CASPAR_LOG(error) << L"[notchlc_producer] CUDA-GL register: " << ex.what()
                                      << L" -- switching to host-copy fallback";
                    for (int i = 0; i < num_slots_; i++) cgt_[i].reset();
                    use_host_copy_ = true;
                }
            }
        }
#endif

        int                  gl_slot           = 0;
        std::vector<int32_t> audio_accum;
        int                  audio_frame_idx   = 0;
        int64_t              video_frame_count = in_frame_;
        uint32_t             seen_epoch  = seek_epoch_.load(std::memory_order_acquire);
        uint64_t             next_seq    = 0;  // monotonically increasing; resets only on user seek

        while (!stop_flag_) {
            //  Wait for next in-sequence item from reorder buffer 
            bool        epoch_changed = false;
            Lz4DoneItem item;
            {
                std::unique_lock<std::mutex> lk(lz4_done_mutex_);
                lz4_done_cv_.wait(lk, [&] {
                    if (stop_flag_) return true;
                    // Unblock immediately when a user-seek changes the epoch.
                    if (seek_epoch_.load(std::memory_order_acquire) != seen_epoch)
                        return true;
                    // Drain any stale-epoch items that snuck in before the epoch check
                    // in lz4_loop fired (tiny race window).
                    while (!lz4_done_pq_.empty() &&
                           lz4_done_pq_.top().epoch != seen_epoch) {
                        auto stale = std::move(
                            const_cast<Lz4DoneItem&>(lz4_done_pq_.top()));
                        lz4_done_pq_.pop();
                        if (stale.slot >= 0) {
                            std::lock_guard<std::mutex> slk(slot_pool_mutex_);
                            slot_pool_.push(stale.slot);
                            slot_pool_cv_.notify_one();
                        }
                        lz4_done_cv_.notify_all();  // space freed
                    }
                    // Ready when the next-in-sequence item is at the top.
                    return !lz4_done_pq_.empty() &&
                           lz4_done_pq_.top().epoch == seen_epoch &&
                           lz4_done_pq_.top().seq   == next_seq;
                });
                if (stop_flag_) break;

                const uint32_t cur_epoch =
                    seek_epoch_.load(std::memory_order_acquire);
                if (cur_epoch != seen_epoch) {
                    // User seek detected.  Drain entire pq:
                    // keep new-epoch items, return slots for old-epoch items.
                    std::priority_queue<Lz4DoneItem, std::vector<Lz4DoneItem>,
                                        Lz4DoneItemCmp> fresh_pq;
                    while (!lz4_done_pq_.empty()) {
                        auto e = std::move(
                            const_cast<Lz4DoneItem&>(lz4_done_pq_.top()));
                        lz4_done_pq_.pop();
                        if (e.epoch == cur_epoch) {
                            fresh_pq.push(std::move(e));
                        } else if (e.slot >= 0) {
                            std::lock_guard<std::mutex> slk(slot_pool_mutex_);
                            slot_pool_.push(e.slot);
                            slot_pool_cv_.notify_one();
                        }
                    }
                    lz4_done_pq_ = std::move(fresh_pq);
                    lz4_done_cv_.notify_all();
                    seen_epoch = cur_epoch;
                    next_seq   = 0;
                    epoch_changed = true;
                } else {
                    item = std::move(
                        const_cast<Lz4DoneItem&>(lz4_done_pq_.top()));
                    lz4_done_pq_.pop();
                }
            }
            lz4_done_cv_.notify_all();  // wake lz4 workers: space freed

            if (epoch_changed) {
                video_frame_count = video_frame_start_;
                audio_accum.clear();
                audio_frame_idx   = 0;
                {
                    std::lock_guard<std::mutex> rlk(queue_mutex_);
                    while (!ready_queue_.empty()) ready_queue_.pop();
                }
                queue_cv_.notify_all();
                continue;
            }

            ++next_seq;

            //  EOF 
            if (item.slot < 0) {
                // Safety net: EOF sentinel reached GPU thread (should not happen with eof_paused_ model,
                // but handle gracefully by pausing rather than terminating.)
                eof_paused_.store(true, std::memory_order_release);
                queue_cv_.notify_all();
                continue;
            }

            //  Backpressure: wait for renderer 
            {
                std::unique_lock<std::mutex> lk(queue_mutex_);
                queue_cv_.wait(lk, [this] {
                    return stop_flag_ || (int)ready_queue_.size() < MAX_QUEUED;
                });
                if (stop_flag_) {
                    std::lock_guard<std::mutex> slk(slot_pool_mutex_);
                    slot_pool_.push(item.slot);
                    slot_pool_cv_.notify_one();
                    break;
                }
            }

            const int      cur_slot = item.slot;
            NotchLCDecodeCtx& ctx   = slots_[cur_slot];
            const int cm = (color_matrix_override_ != -1) ? color_matrix_override_ : NOTCHLC_CM_709;

            //  GPU phase 
            caspar::timer decode_timer;
            cudaError_t   err = cudaSuccess;

            if (!use_host_copy_ && cgt_[gl_slot]) {
                CudaGLTexture& cgt = *cgt_[gl_slot];
                cudaArray_t arr;
                try { arr = cgt.map(ctx.stream); }
                catch (const std::exception& ex) {
                    CASPAR_LOG(error) << L"[notchlc_producer] GL map failed: " << ex.what();
                    std::lock_guard<std::mutex> slk(slot_pool_mutex_);
                    slot_pool_.push(cur_slot);
                    slot_pool_cv_.notify_one();
                    continue;
                }
                err = notchlc_decode_gpu_phase(&ctx, item.hdr, item.actual_uncompressed, cm, arr);
                cgt.unmap(ctx.stream);
            } else {
                err = notchlc_decode_gpu_phase(&ctx, item.hdr, item.actual_uncompressed, cm, nullptr);
                if (err == cudaSuccess) {
                    const size_t out_bytes = (size_t)ctx.width * ctx.height * 4 * sizeof(uint16_t);
                    std::memcpy(h_bgra16_[cur_slot], ctx.d_bgra16, out_bytes);
                }
            }

            if (err != cudaSuccess) {
                CASPAR_LOG(warning) << L"[notchlc_producer] gpu_phase: " << cudaGetErrorString(err);
                graph_->set_tag(diagnostics::tag_severity::WARNING, "dropped");
                std::lock_guard<std::mutex> slk(slot_pool_mutex_);
                slot_pool_.push(cur_slot);
                slot_pool_cv_.notify_one();
                continue;
            }

            const double ms_decode = decode_timer.elapsed();
            graph_->set_value("decode-time", ms_decode * format_desc_.fps * 0.5);
            graph_->set_value("queue-fill",
                static_cast<double>(ready_queue_.size() + 1) / (MAX_QUEUED + 1));

            // --- Audio ---
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

            //  Build frame 
            core::draw_frame df;
            if (!use_host_copy_ && cgt_[gl_slot]) {
                core::pixel_format_desc pfd(core::pixel_format::rgba);
                pfd.planes.push_back(core::pixel_format_desc::plane(
                    frame_info_.width, frame_info_.height, 4, common::bit_depth::bit16));

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
                    cgt_[gl_slot]->gl_texture()));
            } else {
                core::pixel_format_desc pfd(core::pixel_format::bgra);
                pfd.planes.push_back(core::pixel_format_desc::plane(
                    frame_info_.width, frame_info_.height, 4, common::bit_depth::bit16));
                auto mf = frame_factory_->create_frame(this, pfd);
                std::memcpy(mf.image_data(0).begin(), h_bgra16_[cur_slot], mf.image_data(0).size());
                mf.audio_data() = std::move(frame_audio);
                df = core::draw_frame(std::move(mf));
            }

            { std::lock_guard<std::mutex> lk(queue_mutex_); ready_queue_.push(std::move(df)); }
            queue_cv_.notify_one();
            ++video_frame_count;

            // WGL handles are decoupled from LZ4 slots. 
            // We advance the WGL ring index ensuring we don't clobber what CasparCG is currently rendering.
            gl_slot = (gl_slot + 1) % NUM_SLOTS;

            // Return LZ4 cpu-decoder slot to pool AFTER WGL copy is fully completed.
            {
                std::lock_guard<std::mutex> slk(slot_pool_mutex_);
                slot_pool_.push(cur_slot);
            }
            slot_pool_cv_.notify_one();


        }

#ifdef WIN32
        {
            std::lock_guard<std::mutex> gl_lk(caspar::cuda_gl_interop_mutex());
            for (int i = 0; i < num_slots_; i++) cgt_[i].reset();
        }
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
            const auto& cmd = params[0];
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
                raw_cv_.notify_one();   // wake io_loop which owns the demuxer
                result = std::to_wstring(target);

            } else {
                CASPAR_LOG(warning) << L"[notchlc_producer] CALL: unknown command '" << cmd
                                    << L"'. Supported: loop, seek";
            }
        }
        std::promise<std::wstring> p;
        p.set_value(result);
        return p.get_future();
    }

    core::draw_frame receive_impl(const core::video_field field, int) override
    {
        graph_->set_value("frame-time", frame_timer_.elapsed() * format_desc_.hz * 0.5);
        frame_timer_.restart();

        if (field == core::video_field::b)
            return cached_frame_ ? cached_frame_ : core::draw_frame{};

        std::unique_lock<std::mutex> lk(queue_mutex_);
        queue_cv_.wait_for(lk, std::chrono::milliseconds(40),
                           [this] { return !ready_queue_.empty() || stop_flag_; });
        if (ready_queue_.empty()) return cached_frame_;

        double spd = speed_.load();
        speed_accum_ += std::abs(spd);
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

        // Count how many file frames are actually consumed from the queue.
        // frames_to_advance is what the speed requests; the queue may not always
        // have that many ready (decoder underflow).  Overcounting here causes the
        // OSC/diag time counter to run ahead of the real file position.
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

        // Accumulate file frames consumed (not channel ticks) so fps_display_
        // reflects the real decode throughput at any speed.
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

        // Bar fills to 1.0 when the decoder keeps up with channel_fps * |speed|.
        const double target_fps = format_desc_.fps * std::abs(spd);
        if (target_fps > 0.0 && fps_display_ > 0.0)
            graph_->set_value("fps", std::min(1.0, fps_display_ / target_fps));

        return cached_frame_;
    }

    bool is_ready() override { return !ready_queue_.empty(); }

    // We override last_frame() so that a post-seek frame is shown and the
    // diag title updates immediately, without needing RESUME first.
    core::draw_frame last_frame(const core::video_field) override
    {
        if (seek_done_.load(std::memory_order_relaxed)) {
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
            // else: io_thread hasn't decoded the first post-seek frame yet.
            // Keep seek_done_=true and retry on the next tick (~40ms later).
        }
        // Guard: draw_frame::still() dereferences impl_  crash if frame is empty.
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
        monitor_state_["file/name"]     = u8(boost::filesystem::path(path_).filename().wstring());
        monitor_state_["file/path"]     = u8(path_);
        monitor_state_["file/time"]     = {cur_sec, total_seconds_};
        monitor_state_["file/loop"]     = loop_;
        monitor_state_["width"]         = frame_info_.width;
        monitor_state_["height"]        = frame_info_.height;
        monitor_state_["color_matrix"]  = color_matrix_override_;
        return monitor_state_;
    }

    std::wstring print() const override { return L"cuda_notchlc[" + path_ + L"]"; }
    std::wstring name()  const override { return L"cuda-notchlc"; }
};

// ---------------------------------------------------------------------------
// Factory / registration
// ---------------------------------------------------------------------------

spl::shared_ptr<core::frame_producer>
create_notchlc_producer(const core::frame_producer_dependencies& deps,
                        const std::vector<std::wstring>& params)
{
    // Require "CUDA_NOTCHLC" keyword as the first parameter.
    //   PLAY 1-1 CUDA_NOTCHLC <filename>
    //   PLAY 1-1 CUDA_NOTCHLC FILE <filename> [DEVICE <index>] [LOOP] ...
    if (params.empty() || !boost::iequals(params[0], L"CUDA_NOTCHLC"))
        return core::frame_producer::empty();

    std::wstring path;
    int     cuda_device             = 0;
    bool    loop                    = false;
    bool    pingpong_flag           = false;
    double  initial_speed           = 1.0;
    int     color_matrix_override   = -1;   // -1 = AUTO  709
    int64_t start_frame = 0, out_frame = -1, length_param = -1;

    for (size_t i = 1; i < params.size(); ++i) {
        const auto& p = params[i];
        if      (boost::iequals(p, L"LOOP"))                                 { loop = true; }
        else if (boost::iequals(p, L"PINGPONG"))                             { pingpong_flag = true; }
        else if (boost::iequals(p, L"SPEED") && i+1 < params.size())
            initial_speed = boost::lexical_cast<double>(params[++i]);
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
            if      (boost::iequals(val, L"709")    || boost::iequals(val, L"BT709"))   color_matrix_override = NOTCHLC_CM_709;
            else if (boost::iequals(val, L"2020")   || boost::iequals(val, L"BT2020"))  color_matrix_override = NOTCHLC_CM_2020;
            else if (boost::iequals(val, L"601")    || boost::iequals(val, L"BT601"))   color_matrix_override = NOTCHLC_CM_601;
            else if (boost::iequals(val, L"LINEAR"))                                    color_matrix_override = NOTCHLC_CM_LINEAR;
            else if (boost::iequals(val, L"AUTO"))                                      color_matrix_override = NOTCHLC_CM_AUTO;
            else CASPAR_LOG(warning) << L"[notchlc_producer] Unknown COLOR_MATRIX value '" << val
                                     << L"' -- valid: 709, 601, 2020, LINEAR, AUTO. Using AUTO.";
        }
        else if (path.empty() && !p.empty() && p[0] != L'-')                    path = p;
    }

    if (length_param >= 0 && out_frame < 0)
        out_frame = start_frame + length_param;
    if (path.empty()) return core::frame_producer::empty();

    {
        auto is_valid = [](const boost::filesystem::path& p) {
            auto ext = boost::to_lower_copy(p.extension().wstring());
            return ext == L".mov" || ext == L".mxf" || ext == L".mkv"
                || ext == L".mp4" || ext == L".avi";
        };
        auto resolved = find_file_within_dir_or_absolute(env::media_folder(), path, is_valid);
        if (!resolved) {
            CASPAR_LOG(error) << L"[notchlc_producer] File not found: " << path;
            return core::frame_producer::empty();
        }
        path = resolved->wstring();
    }

    try {
        return spl::make_shared<notchlc_producer_impl>(
            path, cuda_device, loop, pingpong_flag, color_matrix_override,
            start_frame, out_frame, initial_speed, deps);
    } catch (const std::exception& ex) {
        CASPAR_LOG(error) << L"[notchlc_producer] " << ex.what();
        return core::frame_producer::empty();
    }
}

void register_notchlc_producer(const core::module_dependencies& module_deps)
{
    module_deps.producer_registry->register_producer_factory(
        L"CUDA_NOTCHLC Producer", &create_notchlc_producer);
}

}} // namespace caspar::cuda_notchlc
