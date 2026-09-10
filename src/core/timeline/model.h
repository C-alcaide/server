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

// WHAT A TIMELINE IS, as data. No behaviour, no JSON, no stage.
//
// The shape is supertimeline's, which the design study settled (`L171`): a tree of objects, each
// with an `enable` expression rather than a start and a duration, so "this lower third begins
// when the interview ends" is a property of the document instead of an arithmetic result the
// author has to maintain. Groups nest and carry their own time remapping, so a sequence can be
// slowed without touching its contents.
//
// Two things in here are deliberately NOT what a naive port of a video NLE would produce:
//
//   AN OBJECT'S LAYER MAY BE EMPTY. A group -- or a bare object -- with no layer is a
//   TRANSPARENT ANCHOR: it occupies time, other objects can reference its start and end, and it
//   writes nothing. That is what makes `#interview.end` expressible without inventing a layer to
//   hang the interview off. O22 of the study asks for it and the cue-stack work (D8) needs it.
//
//   TIME IS LOCAL AND COMPOSED. A key inside a group nested in a group is at its own local time;
//   the two groups' `time_remap`s compose outermost-first to place it. Nothing stores an absolute
//   time except the resolver's output, which is why moving a group moves its contents for free.

#include "curve.h"
#include "expression.h"
#include "time.h"

#include <core/monitor/monitor.h>

#include <boost/rational.hpp>

#include <map>
#include <optional>
#include <string>
#include <vector>

namespace caspar { namespace core { namespace timeline {

/// How an object's own time relates to its parent's.
///
/// `rate` 2 means the object's content plays at twice the speed -- one second of parent time is
/// two seconds of local time. A rational rather than a double so nesting composes exactly:
/// 1/3 inside 3/1 is exactly 1.
struct time_remap
{
    boost::rational<std::int64_t> rate{1};
    flicks                        offset = 0;

    /// Compose this remap INSIDE `outer`. Outermost-first, so `parent.then(child)` is the total.
    time_remap then(const time_remap& inner) const
    {
        time_remap out;
        out.rate   = rate * inner.rate;
        out.offset = offset + inner.offset;
        return out;
    }
};

/// A group's cue-stack behaviour (D8). Every field is off by default, which makes a plain group a
/// plain container -- a cue stack is a group that opted in.
struct group_play
{
    bool one_at_a_time = false; //< children run in sequence, each starting where the last ended
    bool auto_play     = false; //< the next child starts without waiting for a GO
    bool loop          = false; //< after the last child, back to the first
};

/// A step-valued keyframe (D7): enum, boolean, string or blob, merged rather than interpolated.
///
/// Separate from `curve` because these cannot be lerped -- half-way between `screen` and `add` is
/// a number naming neither -- and because their values are not doubles.
struct content_keyframe
{
    std::string                                    id;
    enable_spec                                    enable;
    std::map<std::string, monitor::vector_t>       content;
};

/// What happens to a parameter when the object animating it ends (D4).
enum class on_end_t
{
    release, //< the constant the operator last wrote comes back. Lossless, and the default.
    commit   //< the curve's final value is baked into the constant first
};

/// What an object does to its layer's producer, if anything.
enum class layer_action
{
    none,
    play,
    load,
    pause,
    resume,
    stop,
    clear
};

struct timeline_object
{
    /// STABLE, and the client's to choose (`L48`). Every reference, every published ownership
    /// record and every `resolved` entry keys on this, so a renamed object is a different object
    /// and a reordered document changes nothing.
    std::string id;

    /// When this object is active. More than one spec means more than one occurrence -- the same
    /// object appearing twice in a show is one object with two enables, not two objects, so its
    /// curves and content are authored once.
    std::vector<enable_spec> enable;

    /// `"1-10"` style, or EMPTY for a transparent anchor. See the header comment.
    std::string layer;

    /// Within one layer at one instant, higher wins. Equal priority falls through to
    /// last-started-wins, which is what every product surveyed does.
    int priority = 0;

    /// Authored but inert. Distinct from deleting it: a disabled object still holds its curves
    /// and still resolves its references, so switching it off does not break `#its.end`.
    bool disabled = false;

    /// Names for `.class.start` / `.class.end` references. An object may carry several.
    std::vector<std::string> classes;

    bool                         is_group = false;
    group_play                   play;
    std::vector<timeline_object> children;

    time_remap            remap;
    std::optional<tempo>  tempo_; //< a musical island inside a video document (D1)

    /// Step values applied on entry, before any keyframe.
    std::map<std::string, monitor::vector_t> content;

    std::vector<content_keyframe> keyframes;
    curve                         curves;

    layer_action                 action = layer_action::none;
    std::optional<std::wstring>  clip;
    int                          preroll_frames = 25;

    /// At entry, start the first segment from the LIVE resolved value rather than from the
    /// authored one (ossia's Tweening, Hippotizer's floating keyframe).
    bool rebase = false;

    on_end_t on_end = on_end_t::release;
};

struct timeline_document
{
    std::string          name;
    int                  channel  = 1;
    std::int64_t         revision = 0;
    boost::rational<int> rate{25, 1};
    std::optional<tempo> tempo_;

    struct
    {
        std::string easing = "linear";
    } defaults;

    std::vector<timeline_object> objects;
};

}}} // namespace caspar::core::timeline
