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

// THE TRANSPORT CONTRACT -- the parameters a producer that plays a timeline declares.
//
// WHY A SHARED TABLE RATHER THAN EACH PRODUCER NAMING ITS OWN. Counted 2026-09-14: thirteen
// producers implement `frame_producer::call()` and two implement `parameters()`, and EIGHT of
// the thirteen implement the same transport verbs -- `loop`, `speed`, `in`, `out`, `length`,
// `seek`, `pingpong` -- written out eight times by hand. There was already a de-facto interface
// here; it had simply never been declared, so nothing could address it generically: not a
// client, not a timeline, not a binding.
//
// Eight producers declaring eight spellings of "speed" would reproduce that one layer up, with
// a client special-casing per producer -- which is the thing being fixed. So the NAME, TYPE,
// UNIT and SEMANTICS live here once, and a producer supplies only the two closures and the
// range it actually knows.
//
// WHAT DECLARING ONE BUYS, with no protocol-layer work at all: publication in the control API's
// tree, a typed and range-checked write, a BINDING target and a TIMELINE KEYFRAME target -- the
// machinery `producer_params.h` describes, and which all four producer wrappers
// (`frame_producer_registry`, `separated_producer`, `sting_producer`, `transition_producer`)
// already forward.
//
// A PRODUCER DECLARES ONLY WHAT IT IMPLEMENTS. `replay` has no `pingpong`;
// `image_scroll_producer` has only `speed`. A contract that forced every row onto every
// producer would produce parameters that accept a write and do nothing, which is the
// `MIXER EXPOSURE` class -- a command that returns success and changes no pixel.
//
// NOTHING HERE DEPRECATES A `CALL` VERB. Every producer keeps its `call()`; this is a second,
// typed route to the same members, and a producer should delegate both to one place so the two
// cannot drift.
//
// Plan: `docs/plans/PRODUCER_TRANSPORT_API_PLAN.md`.

#include <core/producer/producer_params.h>

#include <cstdint>
#include <string>

namespace caspar { namespace core {

/// The canonical transport parameters. The enumerator name IS the published path segment.
enum class transport_param
{
    /// The playhead, in frames from the start of the MATERIAL, 0-based.
    ///
    /// **This is `SEEK` as state rather than as an action**, and the distinction is the whole
    /// reason it is here: writing a frame number IS setting the playhead, and the getter
    /// already existed as the published `file/frame` leaf. As a parameter it is scrubbable,
    /// bindable and keyframable; as an action it would be none of those.
    ///
    /// A read_write playhead on a producer that also advances by itself is a genuine feedback
    /// hazard -- a timeline driving `position` fights the producer's own per-frame advance.
    /// That is also exactly what a scrub is, so it is resolved by OWNERSHIP rather than
    /// refused: while a timeline or binding owns the parameter it is authoritative, and on
    /// release the producer resumes **from where the playhead was left** rather than from the
    /// value captured on entry. `position` is the first parameter for which those two differ,
    /// which is why this paragraph exists rather than a line in a table.
    position,

    /// Playback rate, 1.0 nominal. Negative is reverse where the producer supports it; a
    /// producer that does not should refuse the write from its `set` rather than declare a
    /// minimum of 0, so a client can tell "not supported" from "out of range".
    speed,

    /// Wrap at `out` back to `in` instead of ending.
    loop,

    /// First frame of the playable range, in frames from the start of the material.
    in,

    /// End of the playable range, in frames from the start of the material.
    ///
    /// `in` and `out` fully determine the range, which is why there is no settable
    /// range-length row -- see `length`.
    out,

    /// The length of the MATERIAL, in frames. **Read-only**, and not the length of the
    /// playable range.
    ///
    /// The two were worth separating because AMCP conflates them: `ffmpeg_producer`'s `CALL
    /// LENGTH` sets the CLIP duration while its published `file/frame` reports the FILE
    /// duration, and both producers already publish the material length as `file/length`. So
    /// `length` here means what the published leaf already means in both, and the range is
    /// expressed only through `in`/`out`. `CALL LENGTH` keeps working and simply has no
    /// parameter twin.
    ///
    /// For a growing recording this changes under the reader, which is the point.
    length,

    /// Reverse at `out` instead of wrapping, and again at `in`.
    pingpong,
};

struct transport_param_meta
{
    const char*        name;
    fields::value_type type;
    fields::access_t   access;
    const char*        unit;
    const char*        description;
};

/// The contract itself. Order here is the published `index` order for a producer that declares
/// the whole set; a producer declaring a subset passes its own running index (see below).
inline const transport_param_meta& transport_meta(transport_param which)
{
    using fields::access_t;
    using fields::value_type;

    static const transport_param_meta rows[] = {
        {"position",
         value_type::integer,
         access_t::read_write,
         "frames",
         "the playhead, in frames from the start of the material. Writing it seeks"},
        {"speed",
         value_type::real,
         access_t::read_write,
         "x",
         "playback rate; 1.0 is nominal. Negative is reverse where the producer supports it"},
        {"loop", value_type::boolean, access_t::read_write, "", "wrap at out back to in instead of ending"},
        {"in", value_type::integer, access_t::read_write, "frames", "first frame of the playable range"},
        {"out", value_type::integer, access_t::read_write, "frames", "end of the playable range"},
        {"length",
         value_type::integer,
         access_t::read,
         "frames",
         "the length of the MATERIAL, not of the playable range. Read-only, and it changes "
         "under a growing recording"},
        {"pingpong", value_type::boolean, access_t::read_write, "", "reverse at out instead of wrapping"},
    };

    return rows[static_cast<std::size_t>(which)];
}

/// A `param_desc` pre-filled from the contract. The caller supplies `get`, `set`, and `min`/`max`
/// where the producer actually knows them -- an absent range is the honest answer for an
/// unbounded parameter, as `param_desc` says, and inventing 0..1 for one would be a lie a
/// control surface then draws a slider from.
///
/// `index` is the caller's running position, because a producer declaring a subset must still
/// publish a contiguous order: the tree's `CONTENTS` is a JSON object and therefore unordered by
/// definition, so the number is the only structure a client can lay out from.
inline param_desc make_transport_param(transport_param which, int index)
{
    const auto& m = transport_meta(which);

    param_desc p;
    p.name        = m.name;
    p.label       = m.name;
    p.type        = m.type;
    p.access      = m.access;
    p.unit        = m.unit;
    p.description = m.description;
    p.index       = index;
    p.arity       = 1;
    p.bounding    = fields::bounding_t::refuse;
    return p;
}

}} // namespace caspar::core
