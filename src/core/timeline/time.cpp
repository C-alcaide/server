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

#include "../StdAfx.h"

#include "time.h"

#include <common/log.h>

#include <cmath>
#include <cstdio>
#include <string>

namespace caspar { namespace core { namespace timeline {

flicks flicks_per_frame(const boost::rational<int>& fps)
{
    // fps = time_scale / duration, so one frame is duration / time_scale seconds. Computed as
    // (flicks_per_second / time_scale) * duration so the intermediate never overflows and so the
    // exactness the header promises is visible: the first division has no remainder for any rate
    // in the format table, and the self-test asserts exactly that.
    const std::int64_t num = fps.numerator();
    const std::int64_t den = fps.denominator();
    if (num <= 0 || den <= 0)
        return flicks_per_second; // a degenerate rate reads as 1 fps rather than dividing by zero
    return (flicks_per_second * den) / num;
}

flicks from_seconds(double seconds) { return static_cast<flicks>(std::llround(seconds * flicks_per_second)); }

double to_seconds(flicks t) { return static_cast<double>(t) / flicks_per_second; }

namespace {

/// Floor division for signed 64-bit, because `/` truncates toward zero and a time one flick
/// before frame 0 is frame -1, not frame 0.
std::int64_t floor_div(std::int64_t a, std::int64_t b)
{
    const std::int64_t q = a / b;
    return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q;
}

} // namespace

std::int64_t to_frames(flicks t, const boost::rational<int>& fps) { return floor_div(t, flicks_per_frame(fps)); }

flicks from_frames(std::int64_t frames, const boost::rational<int>& fps) { return frames * flicks_per_frame(fps); }

std::string to_timecode(flicks t, const boost::rational<int>& fps, bool drop_frame)
{
    std::int64_t n = to_frames(t, fps);
    const bool   negative = n < 0;
    if (negative)
        n = -n;

    // The NOMINAL integer rate the label counts in: 30 for 29.97, 60 for 59.94, and the rate
    // itself for every integer rate. `fps` rounded up gives exactly that.
    const std::int64_t nominal = (fps.numerator() + fps.denominator() - 1) / fps.denominator();
    const bool         is_1001 = fps.denominator() == 1001;

    if (drop_frame && is_1001) {
        // SMPTE 12M. Two frame NUMBERS (four at 59.94) are skipped at the start of every minute
        // except every tenth, so that 00:10:00;00 lands on frame 17982 -- ten minutes of 29.97 --
        // rather than on frame 18000. The frame COUNT is untouched; only the label jumps.
        const std::int64_t drop  = nominal * 2 / 30;         // 2 or 4
        const std::int64_t fp10m = nominal * 600 - 9 * drop; // 17982 at 30
        const std::int64_t fpm   = nominal * 60 - drop;      // 1798 at 30
        const std::int64_t d     = n / fp10m;
        const std::int64_t m     = n % fp10m;
        n += drop * 9 * d;
        if (m >= drop)
            n += drop * ((m - drop) / fpm);
    }

    const std::int64_t ff = n % nominal;
    const std::int64_t ss = (n / nominal) % 60;
    const std::int64_t mm = (n / (nominal * 60)) % 60;
    const std::int64_t hh = n / (nominal * 3600);

    char buf[40];
    std::snprintf(buf, sizeof buf, "%s%02lld:%02lld:%02lld%c%02lld", negative ? "-" : "", static_cast<long long>(hh),
                  static_cast<long long>(mm), static_cast<long long>(ss), (drop_frame && is_1001) ? ';' : ':',
                  static_cast<long long>(ff));
    return buf;
}

flicks from_bars(double bars, const tempo& t)
{
    if (t.bpm <= 0.0 || t.beats_per_bar <= 0)
        return 0;
    const double seconds = bars * t.beats_per_bar * 60.0 / t.bpm;
    return from_seconds(seconds);
}

double to_bars(flicks t, const tempo& tp)
{
    if (tp.bpm <= 0.0 || tp.beats_per_bar <= 0)
        return 0.0;
    return to_seconds(t) * tp.bpm / (60.0 * tp.beats_per_bar);
}

// ---------------------------------------------------------------------------------------------
// Self-test

namespace {

int failures = 0;

void check(bool ok, const std::string& what, const std::string& detail = "")
{
    if (ok)
        return;
    ++failures;
    CASPAR_LOG(error) << L"[timeline-time] SELF-TEST FAILED: " << u16(what)
                      << (detail.empty() ? L"" : (L" -- " + u16(detail)));
}

std::string rate_name(const boost::rational<int>& r)
{
    return std::to_string(r.numerator()) + "/" + std::to_string(r.denominator());
}

} // namespace

int time_self_test()
{
    failures = 0;

    // Every frame rate a channel can be configured to (the distinct time_scale/duration pairs in
    // video_format.cpp's table), plus the two NTSC rates the table does not carry today but LTC
    // chase and imported media will: 29.97 and 48. THE PROPERTY: one frame is a whole number of
    // flicks. A rate that fails here would drift the playhead by a fraction of a flick per frame,
    // silently, forever -- which is why it is checked at boot and not discovered in a picture.
    const boost::rational<int> rates[] = {
        {24000, 1000},  {24000, 1001}, {25000, 1000},  {30000, 1000},  {30000, 1001}, {48000, 1000},
        {50000, 1000},  {60000, 1000}, {60000, 1001},  {100000, 1000}, {120000, 1000},
    };
    for (const auto& r : rates) {
        const std::int64_t num = r.numerator();
        const std::int64_t den = r.denominator();
        check((flicks_per_second * den) % num == 0, "exact/" + rate_name(r),
              "remainder " + std::to_string((flicks_per_second * den) % num));
        // And the round trip through frames is the identity at several frame counts, including
        // ones past an hour, so an off-by-one at the floor or a truncation shows.
        for (const std::int64_t f : {std::int64_t{0}, std::int64_t{1}, std::int64_t{12}, std::int64_t{1439},
                                     std::int64_t{1'000'000}}) {
            check(to_frames(from_frames(f, r), r) == f, "roundtrip/frames/" + rate_name(r),
                  "frame " + std::to_string(f));
            // One flick short of the next frame is still this frame -- floor, not round.
            check(to_frames(from_frames(f + 1, r) - 1, r) == f, "floor/" + rate_name(r), "frame " + std::to_string(f));
        }
    }

    // The specific numbers the header promises.
    check(flicks_per_frame({60000, 1001}) == 11'771'760, "59.94-is-11771760",
          std::to_string(flicks_per_frame({60000, 1001})));
    check(flicks_per_frame({25000, 1000}) == 28'224'000, "25-is-28224000",
          std::to_string(flicks_per_frame({25000, 1000})));

    // The wire example: 0.48 s on a 25p channel is frame 12 exactly, not 11.9999.
    check(to_frames(from_seconds(0.48), {25000, 1000}) == 12, "wire/0.48s-at-25p-is-frame-12",
          std::to_string(to_frames(from_seconds(0.48), {25000, 1000})));
    // Seconds round-trip to within one flick.
    for (const double s : {0.0, 0.04, 1.0, 59.999, 3600.5, 86399.0}) {
        const double back = to_seconds(from_seconds(s));
        check(std::abs(back - s) <= 1.0 / flicks_per_second, "roundtrip/seconds", std::to_string(s));
    }
    // Negative time floors, as the header says.
    check(to_frames(-1, {25000, 1000}) == -1, "floor/negative", std::to_string(to_frames(-1, {25000, 1000})));

    // Timecode labels. Non-drop: an hour at 25 is 01:00:00:00.
    check(to_timecode(from_frames(90000, {25000, 1000}), {25000, 1000}, false) == "01:00:00:00", "tc/25/hour",
          to_timecode(from_frames(90000, {25000, 1000}), {25000, 1000}, false));
    check(to_timecode(from_frames(1449, {25000, 1000}), {25000, 1000}, false) == "00:00:57:24", "tc/25/1449",
          to_timecode(from_frames(1449, {25000, 1000}), {25000, 1000}, false));
    // Drop-frame, the two SMPTE 12M anchors: frame 1800 labels 00:01:00;02 (frames 0 and 1 of
    // minute 1 are skipped) and frame 17982 labels 00:10:00;00 (minute 10 skips nothing).
    const boost::rational<int> ntsc{30000, 1001};
    check(to_timecode(from_frames(1800, ntsc), ntsc, true) == "00:01:00;02", "tc/df/1800",
          to_timecode(from_frames(1800, ntsc), ntsc, true));
    check(to_timecode(from_frames(17982, ntsc), ntsc, true) == "00:10:00;00", "tc/df/17982",
          to_timecode(from_frames(17982, ntsc), ntsc, true));
    check(to_timecode(from_frames(1799, ntsc), ntsc, true) == "00:00:59;29", "tc/df/1799",
          to_timecode(from_frames(1799, ntsc), ntsc, true));
    // drop_frame on an integer rate is ignored -- the separator stays ':' and nothing is skipped.
    check(to_timecode(from_frames(1800, {30000, 1000}), {30000, 1000}, true) == "00:01:00:00", "tc/df-ignored-at-30",
          to_timecode(from_frames(1800, {30000, 1000}), {30000, 1000}, true));

    // Tempo is a remap: 120 bpm 4/4, one bar is two seconds, and it round-trips.
    const tempo t120{120.0, 4};
    check(from_bars(1.0, t120) == 2 * flicks_per_second, "tempo/one-bar-is-2s", std::to_string(from_bars(1.0, t120)));
    check(std::abs(to_bars(from_bars(3.25, t120), t120) - 3.25) < 1e-9, "tempo/roundtrip",
          std::to_string(to_bars(from_bars(3.25, t120), t120)));

    if (failures == 0)
        CASPAR_LOG(info) << L"[timeline-time] self-test: all checks passed";
    else
        CASPAR_LOG(error) << L"[timeline-time] self-test: " << failures << L" FAILURE(S)";
    return failures;
}

}}} // namespace caspar::core::timeline
