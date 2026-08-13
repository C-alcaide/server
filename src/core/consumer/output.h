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

#include <common/memory.h>
#include <core/mixer/image/image_mixer.h>
#include <core/video_format.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace caspar::diagnostics {
class graph;
}

namespace caspar { namespace core {

class output final
{
  public:
    explicit output(const spl::shared_ptr<diagnostics::graph>& graph,
                    const video_format_desc&                   format_desc,
                    const core::channel_info&                  channel_info);

    output(const output&)            = delete;
    output& operator=(const output&) = delete;
    ~output();

    /// One rendered view and the frame it produced.
    using view_frames = std::vector<std::pair<ocio_view_key, const_frame>>;

    // Send a frame to the output. If running an interlaced channel, two frames will be provided.
    //
    // `views` carries the extra per-consumer views the mixer rendered this tick. A consumer
    // that declared one is handed its own frame; every other consumer gets `frame`, exactly
    // as before. Empty on any channel without `<working-space-composite>`.
    void operator()(const const_frame&       frame,
                    const const_frame&       frame2,
                    const video_format_desc& format_desc,
                    const view_frames&       views  = {},
                    const view_frames&       views2 = {});

    /// The distinct views this output's consumers want, for the mixer to render.
    ///
    /// Asked once per tick by `video_channel`. Deduplicated here rather than in the mixer
    /// because the consumers live here, and two consumers asking for the same view must
    /// cost one pass, not two.
    std::vector<ocio_view_key> distinct_consumer_views() const;

    void add(const spl::shared_ptr<frame_consumer>& consumer);
    void add(int index, const spl::shared_ptr<frame_consumer>& consumer);
    bool remove(const spl::shared_ptr<frame_consumer>& consumer);
    bool remove(int index);

    std::future<bool> call(int index, const std::vector<std::wstring>& params);

    size_t consumer_count() const;
    bool   any_consumer_needs_cpu_data() const;

    core::monitor::state state() const;

  private:
    struct impl;
    std::unique_ptr<impl> impl_;
};

}} // namespace caspar::core
