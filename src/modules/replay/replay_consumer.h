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
 * This module uses libvmx (https://github.com/openmediatransport/libvmx),
 * licensed under MIT, which is compatible with GPL-3.
 *
 * Derived from the CasparCG replay module
 * (https://github.com/krzyc/CasparCG-Server/tree/master/src/modules/replay).
 * Copyright (c) 2011 Sveriges Television AB <info@casparcg.com>
 * Copyright (c) 2013 Technical University of Lodz Multimedia Centre <office@cm.p.lodz.pl>
 * Authors: Robert Nagy <ronag89@gmail.com>,
 *          Jan Starzak <jan@ministryofgoodsteps.com>,
 *          Krzysztof Pyrkosz <pyrkosz@o2.pl>
 */

#pragma once

#include <core/consumer/frame_consumer.h>
#include <core/consumer/channel_info.h>
#include <core/frame/frame.h>
#include <core/video_format.h>
#include <common/utf.h>
#include <string>
#include <mutex>
#include <atomic>
#include <thread>
#include <vector>
#include <condition_variable>
#include <deque>
#include "replay_segmented_storage.h"
#include <common/diagnostics/graph.h>

// VMX Header
#include "vmxcodec.h" 

// Link libvmx
#pragma comment(lib, "libvmx.lib")

namespace caspar { namespace replay {

struct replay_consumer : public core::frame_consumer
{
    core::monitor::state    state_;
    mutable std::mutex      state_mutex_;
    std::string             path_;
    int                     channel_index_ = -1;
    
    std::unique_ptr<ReplaySegmentedWriter> writer_;
    int                     max_duration_sec_;
    int                     segment_duration_sec_;

    VMX_INSTANCE*           vmx_ = nullptr;
    int                     width_ = 0;
    int                     height_ = 0;
    
    // FPS counter
    std::chrono::steady_clock::time_point last_fps_update_;
    int                     frames_since_update_ = 0;
    double                  current_fps_ = 0.0;
    
    // Stats
    int64_t                 frames_written_ = 0;
    double                  fps_ = 25.0;
    
    VMX_PROFILE             quality_ = VMX_PROFILE_SQ;
    
    // Audio buffer
    std::vector<int32_t>    audio_buffer_;
    
    // Diagnostics
    spl::shared_ptr<diagnostics::graph> graph_;
    
    // Pre-allocated encode buffer to avoid per-frame allocation
    std::vector<uint8_t>    encode_buffer_;
    /// One line per consumer for a rejected frame, not one per frame: a systematic
    /// encode failure is a wall of identical lines at the channel's frame rate.
    bool                    encode_failure_warned_ = false;

    // ── THE RECORD PATH RUNS ON A WORKER, NOT ON THE CHANNEL ────────────────────────────
    //
    // `send()` used to encode the frame and call `WriteFrame` -- `fwrite` + `fflush` for the
    // payload and again for the index -- INLINE, and `output.cpp` waits on the consumer's
    // future and measures that wait as the channel's remaining headroom. Measured 2026-09-14
    // at 1080p50: `consume_load` **0.0003 idle against 0.127 recording**, so one replay
    // recording spent an eighth of the channel's frame budget on the thread that paces every
    // layer on it. A slow or busy volume stalled the whole channel, not only the recording.
    //
    // Same shape as `ffmpeg_consumer`, which the module's own gap note cites: a BOUNDED queue
    // and a worker. Bounded is the point -- an unbounded one converts a disk that cannot keep
    // up into memory growth, which fails later and worse.
    //
    // **A full queue drops a frame rather than blocking, and NEVER returns `false`**: a false
    // future makes `output::do_send` erase the consumer silently, which is exactly what an
    // instant-replay buffer armed for hours must not do. A dropped frame is a gap of one frame
    // in the recording; a stalled channel is every layer on air.
    struct pending_frame
    {
        std::vector<uint8_t> bgra;      ///< a copy, because the const_frame's host buffer is
                                        ///< recycled as soon as the channel moves on
        int                  stride = 0;
        std::vector<int32_t> audio;
    };

    std::deque<pending_frame> queue_;
    std::mutex                queue_mutex_;
    std::condition_variable   queue_cv_;
    std::thread               worker_;
    std::atomic<bool>         worker_stop_{false};
    std::atomic<int64_t>      frames_dropped_{0};
    bool                      drop_warned_ = false;

    /// Four frames at 1080p BGRA is ~33 MB, and four frames of slack absorbs a segment
    /// rotation (two `fopen`s and a header write) without the channel ever seeing it.
    static constexpr std::size_t queue_capacity_ = 4;

    void worker_loop();
    void write_one(pending_frame& f);

public:
    replay_consumer(std::string path, VMX_PROFILE quality);
    ~replay_consumer();

    void initialize(const core::video_format_desc& format_desc,
                    const core::channel_info&      channel_info,
                    int                            port_index) override;

    std::future<bool> send(core::video_field field, core::const_frame frame) override;

    std::wstring print() const override { return L"vmx[" + u16(path_) + L"]"; }
    std::wstring name() const override { return L"vmx"; }
    bool has_synchronization_clock() const override { return false; }
    int index() const override { return 200000 + channel_index_; }
    core::monitor::state state() const override;
};
}}
