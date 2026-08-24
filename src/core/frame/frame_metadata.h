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

#include <cstdint>
#include <vector>

namespace caspar { namespace core {

/// Ancillary data that travels **with** a frame rather than in its pixels.
///
/// ── Why this exists ─────────────────────────────────────────────────────────
/// A frame has carried a picture and audio and nothing else. That is fine until a source
/// carries something a broadcaster is obliged to preserve, and the obvious one is **closed
/// captions**: a grep of this tree finds no caption support anywhere, so a compliance
/// requirement had no answer at all.
///
/// Captions cannot be carried out of band. They are timed to individual frames -- CEA-708
/// packs a fixed number of caption bytes per frame and the decoder reconstructs the text from
/// their order -- so a side channel that arrives "about now" is not a smaller version of the
/// feature, it is a different and broken one. They have to ride the frame through the mixer to
/// whichever consumer emits them.
///
/// ── Why a struct rather than the existing `opaque()` ────────────────────────
/// `const_frame::opaque()` is a `std::any` and looks like the obvious home. It is already
/// taken: both mixers put their pre-staged textures there and `any_cast` them back, so a
/// second occupant would be a silent type collision on the frame path. A named field also says
/// what it holds, which an `any` cannot.
///
/// ── What goes in here, and what does not ────────────────────────────────────
/// Things timed to a frame and opaque to the mixer. Captions today; SCTE-35 splice points and
/// analytics results are the same shape and would fit.
///
/// **Measurements do not belong here.** Loudness, SRT link statistics and the like are already
/// carried by `core::monitor::state`, they update on their own clock rather than per frame, and
/// duplicating them onto every frame would cost the frame path to publish a number that a poll
/// already answers.
struct frame_metadata
{
    /// One caption packet, exactly as it arrived, with the format it arrived in.
    ///
    /// **Not decoded, and deliberately not.** The bytes are passed through so a consumer can
    /// re-emit them -- into VANC, or back into an encoded stream -- and every decode-and-
    /// re-encode step is a chance to change what the broadcaster is obliged to preserve. A
    /// caption renderer would decode these; a caption *pass-through* must not.
    struct caption
    {
        /// The wire format, kept as the numeric value the source used rather than re-coded
        /// into an enum of our own: the producer and the consumer are then talking about the
        /// same thing without this header having to track every format either end learns.
        /// GStreamer's `GstVideoCaptionType` is the reference — 1 = CEA-608 raw,
        /// 2 = CEA-608 in S334-1A, 3 = CEA-708 raw, 4 = CEA-708 in CDP.
        int                       format = 0;
        std::vector<std::uint8_t> data;
    };

    /// Usually empty, occasionally one, rarely more than one — a frame may carry both a 608
    /// and a 708 packet where a source is serving both.
    std::vector<caption> captions;

    bool empty() const { return captions.empty(); }
};

}} // namespace caspar::core
