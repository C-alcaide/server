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

// EXPRESSIONS TO INSTANCES: the one function that turns an authored document into absolute times.
//
// `resolve` is PURE. Same document and same trigger log in, same instances out, no clock, no
// stage, no channel. That is deliberate and it is what makes two things possible that the old
// keyframe engine could not do:
//
//   `GET /v1/timeline/{name}/resolved?at=` can answer for any position without running the show,
//   so the client DRAWS WHAT THE SERVER COMPUTED rather than reimplementing the collision rules
//   in JavaScript and disagreeing (`L28`).
//
//   The whole thing is testable at boot, in microseconds, against a table of documents whose
//   answers were worked out by hand. `resolver_self_test` is that table.
//
// The tick calls it once per document per re-resolve -- on a PUT, or when a trigger fires -- and
// then only evaluates the cached result per frame. Resolution is not a per-frame cost.

#include "model.h"

#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace caspar { namespace core { namespace timeline {

/// One occurrence of one object, in absolute document time.
struct instance
{
    std::string object_id;
    std::string layer;      //< empty for a transparent anchor
    flicks      start = 0;

    /// OPEN when the object waits for a trigger, or when `while` names something still open.
    /// An open end is not "very long": last-started-wins has to distinguish "runs until told" from
    /// "runs until 10:00", because a later object with a definite end should not be shadowed
    /// forever by an earlier one that is merely unfinished.
    std::optional<flicks> end;

    int        repeat_index = 0; //< 0 for the first occurrence of a `repeating` spec
    int        priority     = 0;
    time_remap composed;         //< every ancestor's remap, outermost-first

    /// Absolute -> local time, through the composed remap. The inverse of what placed it.
    ///
    /// The composed OFFSET is already inside `start` -- that is what placed the instance -- so
    /// subtracting it again here double-counted it and put a child of a group offset by ten
    /// seconds at local time -18 s. Only the RATE remains to apply, and it multiplies going
    /// inwards: one second of document time is `rate` seconds of the object's own.
    flicks local_at(flicks absolute) const
    {
        const auto rel = absolute - start;
        return static_cast<flicks>(static_cast<double>(rel) *
                                   boost::rational_cast<double>(composed.rate));
    }

    bool contains(flicks t) const { return t >= start && (!end || t < *end); }
};

/// Which triggers have fired, and when. Supplied by the transport, so `resolve` stays pure.
struct trigger_log
{
    std::map<std::string, std::vector<flicks>> fired;

    /// The first firing at-or-after `after`, if any.
    std::optional<flicks> next_after(const std::string& name, flicks after) const
    {
        const auto it = fired.find(name);
        if (it == fired.end())
            return std::nullopt;
        for (const auto t : it->second)
            if (t >= after)
                return t;
        return std::nullopt;
    }
};

struct resolve_error
{
    std::string object;     //< the id whose expression failed, or the cycle's entry point
    std::string expression; //< the text, verbatim, so the client can highlight it
    std::string reason;
};

struct resolved_timeline
{
    std::int64_t revision = 0;

    /// Every instance, sorted by start then by the order the document declared them. The sort is
    /// what lets the per-frame evaluation binary-search instead of scanning.
    std::vector<instance> instances;

    /// By object id, for `#a.end` and for the client's own drawing.
    std::unordered_map<std::string, std::vector<std::size_t>> by_object;

    /// By layer, sorted as `instances` is. A transparent anchor appears in neither.
    std::unordered_map<std::string, std::vector<std::size_t>> by_layer;

    /// Triggers the document is waiting on, so the transport knows which GO means something.
    std::vector<std::string> pending_triggers;

    std::vector<resolve_error> errors;

    bool ok() const { return errors.empty(); }

    /// WHICH INSTANCE OWNS `layer` AT `t`, by the document's own precedence: higher `priority`
    /// first, then LAST-STARTED, then the later declaration. Returns null if none is active.
    ///
    /// Last-started rather than earliest: an operator firing a cue expects it to take over from
    /// whatever was running, and every product surveyed agrees. Earliest-start would make a long
    /// background object permanently shadow every cue fired during it.
    const instance* active_on(const std::string& layer, flicks t) const;
};

/// Turn a document into instances. Never throws; failures land in `errors`.
resolved_timeline resolve(const timeline_document& doc, const trigger_log& triggers);

/// Aborts on a resolver disagreement. Called at boot.
void resolver_self_test();

}}} // namespace caspar::core::timeline
