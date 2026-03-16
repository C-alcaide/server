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

#include <core/producer/frame_producer.h>
#include <core/frame/frame_factory.h>
#include <core/video_format.h>
#include <common/diagnostics/graph.h>
#include <string>
#include <mutex>
#include <atomic>
#include <thread>
#include <vector>
#include "replay_segmented_storage.h"
#include "vmxcodec.h"

// Link libvmx
#pragma comment(lib, "libvmx.lib")

namespace caspar { namespace replay {

struct replay_producer : public core::frame_producer
{
    core::monitor::state    state_;
    mutable std::mutex      state_mutex_;

    std::string             path_;
    int                     channel_index_ = -1;
    
    std::unique_ptr<ReplaySegmentedReader> reader_;

    VMX_INSTANCE*           vmx_ = nullptr;
    int                     width_ = 0;
    int                     height_ = 0;
    double                  fps_ = 25.0;
    
    // FPS counter
    std::chrono::steady_clock::time_point last_fps_update_;
    int                     frames_since_update_ = 0;
    int                     refresh_skip_counter_ = 0;
    double                  current_fps_ = 0.0;
    
    // Diagnostics
    spl::shared_ptr<diagnostics::graph> graph_;
    
    // Playback state
    std::atomic<int64_t>    frame_num_ = 0;
    std::atomic<double>     speed_ = 1.0;
    double                  fractional_frame_ = 0.0;
    std::atomic<bool>       loop_ = false;
    std::atomic<int64_t>    duration_ = 0;
    std::atomic<int64_t>    in_point_ = 0;
    std::atomic<int64_t>    out_point_ = -1; // -1 means end of file

    // Decoding
    std::vector<uint8_t>    read_buffer_;
    core::draw_frame        current_frame_;

public:
    replay_producer(std::string path, spl::shared_ptr<core::frame_factory> frame_factory);
    ~replay_producer();

    core::draw_frame receive_impl(core::video_field field, int nb_samples) override;
    std::future<std::wstring> call(const std::vector<std::wstring>& params) override;
    bool is_ready() override;
    
    // Configures producer from PLAY command arguments
    void configure(const std::vector<std::wstring>& params);

    std::wstring         print() const override { return L"vmx[" + u16(path_) + L"]"; }
    std::wstring         name() const override { return L"vmx"; }
    core::monitor::state state() const override;

private:
    spl::shared_ptr<core::frame_factory> frame_factory_;
};

}}
