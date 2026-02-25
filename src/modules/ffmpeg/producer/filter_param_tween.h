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
 */

#pragma once

#include <common/tweener.h>

#include <algorithm>
#include <string>

namespace caspar { namespace ffmpeg {

/**
 * Animates a single numeric FFmpeg filter parameter over a number of frames.
 *
 * Mirrors the tweened_transform / do_tween pattern used by the MIXER system:
 *   - duration and time are integer frame counts (not wall-clock time)
 *   - source is captured at set_target() to allow smooth mid-animation
 *     retargeting without a visible jump
 *   - the tweener is called as tweener(t, b, c, d) where c = dest - source
 *
 * Usage (per frame in the decode thread):
 *   tween.tick();
 *   double current = tween.fetch();
 *   avfilter_graph_send_command(graph, filter_name, param_name,
 *                               std::to_string(current).c_str(), ...);
 */
struct FilterParamTween
{
    double          source   = 0.0;
    double          dest     = 0.0;
    int             duration = 0; // total frames (0 = instant)
    int             time     = 0; // current frame in [0 .. duration]
    caspar::tweener tweener{L"linear"};

    /** Advance the animation by one frame. */
    void tick() { time = std::min(time + 1, duration); }

    /**
     * Return the current interpolated value.
     * When duration == 0 or time >= duration, returns dest exactly.
     */
    double fetch() const
    {
        if (duration <= 0 || time >= duration)
            return dest;
        return tweener(
            static_cast<double>(time),
            source,
            dest - source,              // c = delta, not absolute destination
            static_cast<double>(duration));
    }

    /**
     * Set a new destination target, starting from the current interpolated
     * position.  Issuing a new command mid-animation will therefore always
     * animate smoothly from wherever the parameter currently is.
     *
     * @param new_dest        The target value to reach.
     * @param new_duration    Number of frames over which to animate (0 = instant).
     * @param tween_name      Easing function name (e.g. L"linear", L"easeinsine").
     */
    void set_target(double new_dest, int new_duration, const std::wstring& tween_name)
    {
        source   = fetch(); // capture current interpolated value as new start
        dest     = new_dest;
        duration = new_duration;
        time     = 0;
        tweener  = caspar::tweener(tween_name);
    }

    /** Returns true once the animation has reached its destination. */
    bool is_done() const { return time >= duration; }
};

}} // namespace caspar::ffmpeg
