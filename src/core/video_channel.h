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

#include "fwd.h"
#include "input/input_event.h"
#include "video_format.h"

#include "frame/pixel_format.h"

#include "monitor/monitor.h"

#include <common/memory.h>

#include <boost/signals2.hpp>

#include <functional>
#include <memory>

namespace caspar { namespace core {

enum route_mode
{
    foreground,
    background,
    next, // background if any, otherwise foreground
};

struct route_id
{
    int        index;
    route_mode mode;
    bool       raw = false; // if true, frames are routed before mixer transforms are applied

    bool const operator==(const route_id& o) { return index == o.index && mode == o.mode && raw == o.raw; }
};

struct route
{
    route()             = default;
    route(const route&) = delete;
    route(route&&)      = default;

    route& operator=(const route&) = delete;
    route& operator=(route&&)      = default;

    boost::signals2::signal<void(class draw_frame, class draw_frame)> signal;
    video_format_desc                                                 format_desc;
    std::wstring                                                      name;
};

/// The per-tick state callback. Carries the snapshot as a shared immutable object
/// rather than by value: the state map is rebuilt every frame and copying it per
/// subscriber was pure waste, and a pointer is what lets a subscriber hold the frame it
/// was handed while the next one is being built.
using video_channel_tick_t = std::function<void(const std::shared_ptr<const core::monitor::state>&)>;

class video_channel final
{
    video_channel(const video_channel&);
    video_channel& operator=(const video_channel&);

  public:
    explicit video_channel(int                                       index,
                           const video_format_desc&                  format_desc,
                           color_space                               default_color_space,
                           std::unique_ptr<image_mixer>              image_mixer,
                           video_channel_tick_t                      on_tick,
                           color_transfer                            default_color_transfer = color_transfer::sdr,
                           bool                                      auto_color_convert     = true,
                           int                                       auto_tone_map          = 0,
                           float                                     display_peak_luminance = 1000.0f,
                           float                                     sdr_reference_white    = 100.0f,
                           bool                                      auto_gamut_compress    = false,
                           bool                                      straight_alpha_grading = false,
                           bool                                      working_space_composite = false);
    ~video_channel();

    /// A copy of the last published snapshot. Kept for `INFO`, which wants a value.
    core::monitor::state state() const;

    /// The last published snapshot, without copying it.
    ///
    /// `state()` returns a copy of a `flat_map` whose vectors each heap-allocate, and it
    /// used to read a member the tick thread was concurrently overwriting -- fine while
    /// the only reader was `INFO` (rare, and on the same thread as nothing else), and a
    /// real data race the moment anything polls. The snapshot is now published as an
    /// immutable object under an atomic, so a reader takes a pointer and the writer never
    /// mutates what a reader can see.
    std::shared_ptr<const core::monitor::state> state_snapshot() const;

    const std::shared_ptr<core::stage>& stage() const;
    std::shared_ptr<core::stage>&       stage();
    const core::mixer&                  mixer() const;
    core::mixer&                        mixer();
    const core::output&                 output() const;
    core::output&                       output();

    spl::shared_ptr<core::frame_factory> frame_factory();

    int index() const;

    [[nodiscard]] channel_info get_channel_info() const;

    std::shared_ptr<core::route> route(int index = -1, route_mode mode = route_mode::foreground, bool raw = false);

    /// Offer a pointer or keyboard event to this channel.
    ///
    /// THE ONE DISPATCH POINT, and `video_channel` is where it belongs because it is the only
    /// object that owns both halves: the image mixer (where previz renders) and the stage
    /// (where layers and their producers live). It is also in `core`, which the accelerator
    /// and the modules may both depend on.
    ///
    /// The mixer is offered the event FIRST. When previz is active it is the whole picture on
    /// this channel -- it replaces the 2D output -- so a click belongs to the 3D view and not
    /// to a layer the operator cannot see. If the mixer declines, the stage hit-tests its
    /// layers (see `stage_base::input`).
    ///
    /// This deliberately does NOT reinstate the 2013-2018 shape, where an `interaction_sink*`
    /// was threaded through EVERY consumer factory -- seven modules had to declare and ignore
    /// the argument to serve one real user, which is visibly why it was deleted. A consumer
    /// that wants to produce events already receives the channel vector and its own
    /// `channel_info.index`; it needs nothing from this interface but this method.
    void input(const input_event& event);

  private:
    struct impl;
    spl::unique_ptr<impl> impl_;
};

}} // namespace caspar::core
