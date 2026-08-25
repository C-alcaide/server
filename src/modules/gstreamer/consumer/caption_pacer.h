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

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <vector>

namespace caspar { namespace gstreamer {

/**
 * Re-paces CEA-708 `cc_data` from the source's frame rate to the channel's.
 *
 * ── The bug this removes ────────────────────────────────────────────────────
 * The consumer used to re-emit whatever caption bytes arrived on the frame, unchanged. That
 * is correct **only when the source and channel frame rates match**, and the end-to-end
 * measurement that validated it was 50 fps into a 50 fps channel — so it could not have
 * caught this.
 *
 * Two things go wrong otherwise, and they are different faults:
 *
 *  1. **The per-frame budget is rate-dependent.** CEA-708 carries a fixed number of `cc_data`
 *     triplets per frame, and the number falls as the rate rises: 25 at 24p, 24 at 25p, 20 at
 *     30p, 12 at 50p, 10 at 60p. A 25p source's 24 triplets pushed onto a 50p channel's frame
 *     is double what the frame may carry, and a decoder is entitled to discard the overflow —
 *     silently, because captions have no picture to look wrong.
 *
 *  2. **A repeated frame repeats its captions.** A 25p source in a 50p channel delivers each
 *     picture twice, and pass-through emitted the same triplets both times. Duplicated
 *     control codes are not a cosmetic problem: CEA-708 is a command stream, so a repeated
 *     `RollUp` or `SetPenLocation` changes what the viewer sees.
 *
 * A queue fixes both: triplets go in once per distinct frame and come out at whatever the
 * channel's budget allows, in order.
 *
 * ── Which formats ───────────────────────────────────────────────────────────
 * `GstVideoCaptionType` 3 (`CEA708_RAW`) is `cc_data`, a whole number of 3-byte triplets, and
 * is what this paces. Everything else — 608 raw, 608-in-S334-1A, 708-in-CDP — is passed
 * through untouched and counted separately, because their pacing is a different question:
 * 608 is two bytes per field per frame and a CDP carries its own framing and sequence
 * counter, so re-packing one means rewriting a header rather than moving triplets.
 *
 * That is a real limit, not an oversight, and it is stated here rather than discovered:
 * **a 608 source into a channel at a different rate is still unpaced.**
 *
 * ── The budget table ────────────────────────────────────────────────────────
 * Taken from FFmpeg's `libavutil/ccfifo.c`, which is the reference implementation the A/53
 * side-data path uses, rather than derived here. See `cc_count_for`.
 *
 * ── INTERIM ─────────────────────────────────────────────────────────────────
 * Upstream `CasparCG/server#1637` implements this properly as part of a general side-data
 * system, with an `a53_cc_queue` that is aware of interlacing as well as rate, and per-layer
 * priority for choosing the caption source. This is the smaller fix that removes the
 * correctness bug without deepening the divergence; it should be replaced by #1637's design
 * rather than developed further. See `core/frame/frame_metadata.h`.
 */
class caption_pacer
{
  public:
    /// One 3-byte `cc_data` triplet.
    using triplet = std::array<std::uint8_t, 3>;

    /**
     * How many `cc_data` triplets one frame may carry at this rate.
     *
     * From FFmpeg's `libavutil/ccfifo.c`. The rate is matched on the exact rational, and an
     * unrecognised one returns 0 — see `pace`, where 0 means "do not pace", because inventing
     * a budget for a rate the standard does not list would be worse than passing through.
     */
    static std::size_t cc_count_for(int fps_numerator, int fps_denominator);

    /**
     * Enqueue this frame's triplets, then take what the channel's frame may carry.
     *
     * `frame_key` identifies the frame the captions arrived on — the address of its
     * `frame_metadata`, which is shared across copies of one frame and therefore stable
     * across the repeats a rate conversion produces. Passing the same key twice enqueues
     * nothing the second time, which is what stops a repeated frame repeating its captions.
     *
     * Returns the bytes to attach, or empty. When the rate is unrecognised or the payload is
     * not a whole number of triplets, the input is returned unchanged: pass-through is the
     * behaviour this replaces and it is the right fallback for anything not understood.
     */
    std::vector<std::uint8_t> pace(const void*                      frame_key,
                                   const std::vector<std::uint8_t>& arriving,
                                   int                              fps_numerator,
                                   int                              fps_denominator);

    /// Triplets still waiting. Non-zero means the source is producing faster than the channel
    /// can carry, which is a real condition worth reporting rather than an error.
    std::size_t backlog() const { return queue_.size(); }

    /// Triplets that ARRIVED, triplets emitted, and triplets the duplicate guard rejected.
    ///
    /// **`in_` counts arrivals, not enqueues, and that distinction is the whole measurement.**
    /// Counting enqueues made the two totals move together: disable the guard and both double,
    /// the ratio stays 1.00, and a battery case reports a pass for a channel duplicating every
    /// control code. Found by mutating the guard and watching the case still pass -- which is
    /// the only way that class of mistake is ever found.
    ///
    /// With R frames repeated out of N, a working channel emits `(N-R)/N` of what arrives and
    /// a broken one emits all of it.
    std::uint64_t triplets_in() const { return in_; }
    std::uint64_t triplets_out() const { return out_; }
    std::uint64_t triplets_suppressed() const { return suppressed_; }

    /// Triplets dropped because the queue grew past anything a channel could drain. Captions
    /// are timed, so an unbounded queue would emit text minutes after the picture it belongs
    /// to -- worse than losing it.
    std::uint64_t dropped() const { return dropped_; }

  private:
    std::deque<triplet> queue_;
    const void*         last_key_ = nullptr;
    std::uint64_t       dropped_  = 0;
    std::uint64_t       in_         = 0;
    std::uint64_t       out_        = 0;
    std::uint64_t       suppressed_ = 0;
};

}} // namespace caspar::gstreamer
