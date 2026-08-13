/*
 * Copyright (c) 2011 Sveriges Television AB <info@casparcg.com>
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
 * Author: Robert Nagy, ronag89@gmail.com
 */

#pragma once

#include "../fwd.h"
#include "../monitor/monitor.h"

#include <common/bit_depth.h>
#include <common/memory.h>

#include <common/future.h>
#include <core/video_format.h>

#include <boost/property_tree/ptree_fwd.hpp>

#include <functional>
#include <future>
#include <string>
#include <utility>
#include <vector>

namespace caspar { namespace core {

// Pipeline depth information for A/V sync advisor.
// Each consumer overrides av_pipeline() to report its contribution
// to the overall audio/video pipeline latency on a channel.
struct av_pipeline_info
{
    bool   has_audio              = false; // Consumer outputs audio
    bool   has_video              = false; // Consumer outputs video
    bool   audio_is_embedded      = false; // Audio & video share physical path (e.g. SDI)
    int    video_depth_frames     = 0;     // Video pipeline depth in whole frames
    int    audio_depth_frames     = 0;     // Audio pipeline depth in whole frames (including delay)
    double video_delay_ms         = 0.0;   // Sub-frame video delay (user-configured)
    double audio_device_latency_ms = 0.0;  // Hardware-reported audio device latency
    bool   video_delay_adjustable = false; // Consumer supports <delay>/<delay-ms> parameters
    bool   audio_delay_adjustable = false; // Consumer supports audio <delay> parameter
    int    audio_delay_frames     = 0;     // Current user-configured audio delay (delay_frames)
};

class frame_consumer
{
    frame_consumer(const frame_consumer&);
    frame_consumer& operator=(const frame_consumer&);

  public:
    static const spl::shared_ptr<frame_consumer>& empty();

    frame_consumer() {}
    virtual ~frame_consumer() {}

    virtual std::future<bool> send(const core::video_field field, const_frame frame) = 0;
    virtual void
    initialize(const video_format_desc& format_desc, const core::channel_info& channel_info, int port_index) = 0;
    virtual std::future<bool> call(const std::vector<std::wstring>& params) { return caspar::make_ready_future(false); }

    virtual core::monitor::state state() const = 0;

    virtual std::wstring print() const = 0;
    virtual std::wstring name() const  = 0;
    virtual bool         has_synchronization_clock() const { return false; }
    virtual bool         needs_cpu_frame_data() const { return true; }

    /// Which OCIO display/view this consumer wants, if not the channel's own.
    ///
    /// Empty -- the default -- means "whatever the channel is showing", which is what every
    /// consumer got before this existed. A non-empty pair makes the mixer render one extra
    /// post-composite pass from the same working-space composite and hand THIS consumer the
    /// result, so a channel can feed an LED processor and an SDI monitor different views of
    /// one composite.
    ///
    /// Ignored unless the channel has `<working-space-composite>`: without it there is no
    /// working-space composite to fan out from. Declared here rather than configured
    /// centrally for the same reason `needs_cpu_frame_data()` is -- it is a property of the
    /// consumer, and the output stage is what has to act on it.
    virtual std::pair<std::string, std::string> ocio_view() const { return {}; }
    virtual int          index() const = 0;

    virtual av_pipeline_info av_pipeline() const { return {}; }
};

}} // namespace caspar::core
