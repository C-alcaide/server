/*
 * Copyright (c) 2026 CasparCG Contributors
 *
 * This file is part of CasparCG (www.casparcg.com).
 *
 * CasparCG is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#pragma once

// THE ONE TIME BASE for the timeline: a 64-bit integer count of FLICKS, 1/705 600 000 s.
//
// Why an integer at all. A timeline resolver compares times constantly -- "does this object's
// start equal that object's end", "which of two instances on a layer started later" -- and
// last-started-wins collision is decided by exactly such a comparison. With `double` seconds
// (the unit the old KEYFRAMES module used) equality is an epsilon question, and 0.48 s on a 25p
// channel is frame 11.9999. With frames at the channel rate, one document cannot play on a 25p
// channel and a 50p one, and a 59.94 end cannot be compared to a 25 start without rounding.
//
// Why THIS integer. 705 600 000 = 2^9 * 3^2 * 5^5 * 7^2. It is exactly divisible by the frame
// period of every video format this server ships -- 24, 25, 30, 48, 50, 60, 100, 120, and every
// x1000/1001 rate (59.94 is 11 771 760 flicks per frame, an integer) -- and by every audio rate
// (44.1k, 48k, 96k). So a time expressed in any of those rasters round-trips through a flick
// count without loss, and a comparison between two of them is an integer compare. That is
// rational time with the denominator chosen once, which is what OpenTimelineIO's `RationalTime`
// becomes when every rate you meet divides one number: export is `RationalTime(flicks,
// 705600000)`, exactly. Nanoseconds were the alternative considered and are inexact at every
// NTSC frame boundary.
//
// What is NOT a time base here. Frames, timecode, seconds and bars are all PRESENTATION RASTERS
// derived from flicks by the functions below. Drop-frame is a labelling of a frame count and
// never a time base. Tempo is a REMAP -- bars -> flicks through a `tempo` -- so a musical
// timeline and a video timeline can share one document.
//
// The wire keeps seconds: a JSON number, rounded to the nearest flick on the way in. Every time
// whose denominator divides the flick rate survives that round trip exactly.
//
// Range: +/- 414 years. Enough.

#include <boost/rational.hpp>

#include <cstdint>
#include <string>

namespace caspar { namespace core { namespace timeline {

using flicks = std::int64_t;

constexpr flicks flicks_per_second = 705'600'000;

/// The length of one frame at `fps`, in flicks. EXACT for every rate in the format table; the
/// self-test at boot proves the remainder is zero for each one, so a rate that is not exact fails
/// at startup rather than as a drifting playhead. `fps` is a FRAME rate (`video_format_desc::
/// framerate`), not the field rate.
flicks flicks_per_frame(const boost::rational<int>& fps);

/// Seconds -> flicks, rounded to nearest. This is the wire conversion.
flicks from_seconds(double seconds);

/// Flicks -> seconds. Exact to double precision for any realistic show length.
double to_seconds(flicks t);

/// Flicks -> whole frames at `fps`, FLOOR (a time part-way through frame 12 is frame 12, also
/// for negative times).
std::int64_t to_frames(flicks t, const boost::rational<int>& fps);

/// Whole frames at `fps` -> flicks. Exact.
flicks from_frames(std::int64_t frames, const boost::rational<int>& fps);

/// `hh:mm:ss:ff`, or `hh:mm:ss;ff` when `drop_frame` and the rate is a x1000/1001 one (SMPTE
/// 12M drop-frame labelling: frame numbers 0 and 1 are skipped at the start of every minute
/// except every tenth, so the LABEL tracks wall-clock time while the frame COUNT stays honest).
/// `drop_frame` is ignored for any other rate, because there is nothing to drop.
std::string to_timecode(flicks t, const boost::rational<int>& fps, bool drop_frame);

/// A musical tempo. `bpm` beats per minute, `beats_per_bar` the time signature's numerator.
struct tempo
{
    double bpm           = 120.0;
    int    beats_per_bar = 4;
};

/// Bars -> flicks through a tempo. At 120 bpm, 4/4, one bar is two seconds.
flicks from_bars(double bars, const tempo& t);

/// Flicks -> bars through a tempo.
double to_bars(flicks t, const tempo& tp);

/// Returns the number of FAILURES and logs each. Same contract as `binding_math_self_test` and
/// `compose_self_test`: a divergence is named at boot rather than found in a picture. What it
/// checks is the one property everything above rests on -- that `flicks_per_frame` is exact for
/// every frame rate this server can be configured to -- plus the round trips and the two SMPTE
/// drop-frame anchors (`00:01:00;02` at frame 1800, `00:10:00;00` at frame 17982).
int time_self_test();

}}} // namespace caspar::core::timeline
