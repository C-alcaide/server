/*
 * Copyright (c) 2026 CasparCG Contributors
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

#include "../StdAfx.h"

#include "caption_pacer.h"

#include <algorithm>

namespace caspar { namespace gstreamer {

namespace {

/// The per-frame `cc_data` budget, by frame rate.
///
/// **Copied from the standard's reference implementation, not derived here.** FFmpeg's
/// `libavutil/ccfifo.c` carries this table for the A/53 caption path, and it is what any
/// decoder on the other end will have been written against. A value computed from first
/// principles that disagreed by one would be indistinguishable from correct until a caption
/// went missing on a long line.
///
/// Note 24p carries the MOST (25) and 60p the least (10): the budget is bytes per second
/// divided by frames per second, so a faster picture rate means fewer triplets per frame, not
/// more.
struct rate_budget
{
    int         num;
    int         den;
    std::size_t triplets;
};

constexpr rate_budget kBudgets[] = {
    {24000, 1001, 25},
    {24, 1, 25},
    {25, 1, 24},
    {30000, 1001, 20},
    {30, 1, 20},
    {50, 1, 12},
    {60000, 1001, 10},
    {60, 1, 10},
};

/// Roughly two seconds of the most generous rate. A queue longer than this is not a backlog
/// the channel will drain, it is a source permanently faster than its sink -- and captions are
/// timed to pictures, so text that arrives two seconds late is worse than text that does not
/// arrive. Dropping the OLDEST keeps the stream current.
constexpr std::size_t kMaxBacklog = 64;

} // namespace

std::size_t caption_pacer::cc_count_for(int fps_numerator, int fps_denominator)
{
    if (fps_denominator <= 0 || fps_numerator <= 0)
        return 0;

    for (const auto& b : kBudgets) {
        if (b.num == fps_numerator && b.den == fps_denominator)
            return b.triplets;
    }

    // Also match a rate given in different terms -- 50/1 and 50000/1000 are the same picture
    // rate, and a channel description is not obliged to be in lowest terms.
    for (const auto& b : kBudgets) {
        if (static_cast<std::int64_t>(b.num) * fps_denominator ==
            static_cast<std::int64_t>(fps_numerator) * b.den)
            return b.triplets;
    }

    return 0;
}

std::vector<std::uint8_t> caption_pacer::pace(const void*                      frame_key,
                                              const std::vector<std::uint8_t>& arriving,
                                              int                              fps_numerator,
                                              int                              fps_denominator)
{
    const auto budget = cc_count_for(fps_numerator, fps_denominator);

    // An unrecognised rate, or a payload that is not whole triplets. Pass it through: that is
    // exactly the behaviour this class replaces, so falling back to it can only be as wrong as
    // before and never more so. Guessing a budget would be the worse failure -- it would drop
    // real caption bytes on a rate the standard simply does not list.
    if (budget == 0 || arriving.size() % 3 != 0)
        return arriving;

    // ── Enqueue, ONCE per distinct frame ───────────────────────────────────────────────
    // A rate conversion repeats pictures: 25p into a 50p channel delivers each frame twice.
    // The key is the address of the frame's shared `frame_metadata`, which survives copies of
    // one frame and differs between frames, so the second delivery of the same picture adds
    // nothing. Without this, every control code in the stream would be issued twice -- and
    // CEA-708 is a command stream, so a doubled `RollUp` is a visible fault rather than
    // redundancy.
    // Counted BEFORE the guard, deliberately. Counting enqueues instead made the two totals
    // move together -- disable the guard and both double, so the ratio stays 1.00 and the
    // battery case reported a pass for a channel duplicating every control code. Measured by
    // mutation 2026-08-25, which is the only reason it was noticed.
    const auto arriving_triplets = arriving.size() / 3;
    in_ += arriving_triplets;

    if (frame_key != last_key_) {
        last_key_ = frame_key;
        for (std::size_t i = 0; i + 2 < arriving.size(); i += 3)
            queue_.push_back({arriving[i], arriving[i + 1], arriving[i + 2]});

        while (queue_.size() > kMaxBacklog) {
            queue_.pop_front();
            ++dropped_;
        }
    } else {
        // A repeat of a picture already seen. This is the whole guard, and the counter is what
        // makes its absence visible.
        suppressed_ += arriving_triplets;
    }

    // ── Take what this frame may carry ─────────────────────────────────────────────────
    const auto take = std::min(budget, queue_.size());
    std::vector<std::uint8_t> out;
    out.reserve(take * 3);
    for (std::size_t i = 0; i < take; ++i) {
        const auto& t = queue_.front();
        out.insert(out.end(), t.begin(), t.end());
        queue_.pop_front();
        ++out_;
    }
    return out;
}

}} // namespace caspar::gstreamer
