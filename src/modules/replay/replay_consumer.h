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
