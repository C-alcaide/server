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

// WHEN an object is active, as an expression rather than a number.
//
// This is the half of supertimeline's model that makes a document maintainable: "the lower third
// starts five seconds after the interview does" is written down as that, so moving the interview
// moves the lower third and nobody edits two numbers. A media server's show is mostly this kind
// of relation -- `L27` of the study, and every product surveyed that has a timeline has some
// form of it.
//
// THE GRAMMAR, v1 and deliberately small (D12 of the plan):
//
//   12.5                a time literal in SECONDS
//   {"frames": 300}     ...or frames, timecode or bars -- but those are JSON literals converted
//                       at PUT against the document's rate and tempo, so by the time a string
//                       reaches this parser it is seconds. `12:00:04:10` and `4bars` are also
//                       accepted here as a convenience for AMCP, which has no JSON.
//   #interview          another object's START -- the bare form means `.start`
//   #interview.start    the same, explicit
//   #interview.end      its end. Open if that object's end is open.
//   #interview.duration end minus start
//   .lowerthird.start   the EARLIEST start of every object carrying that class
//   .lowerthird.end     the LATEST end of them
//   #interview.end + 5  one offset, a literal, `+` or `-`. Deliberately one: `#a.end - #b.start`
//                       is a duration expression and v1 has no arithmetic between references.
//   (#a.end) + 5        parentheses, which change nothing but let a client round-trip its own
//                       formatting
//   1                   in a `while` position only: always active
//   #interview          in a `while` position: active exactly while that object is
//
// WHAT IS DEFERRED, each because it needs something v1 does not have: `$layer` references (the
// resolver would have to know what is playing), reference-to-reference arithmetic (needs an
// expression tree rather than one offset), boolean `& | !` in `while` (needs a predicate
// evaluator), `*` and `/` (need units), and `seamless`.

#include "time.h"

#include <boost/rational.hpp>

#include <optional>
#include <string>
#include <string_view>

namespace caspar { namespace core { namespace timeline {

struct time_expr
{
    enum class kind
    {
        literal,      //< `literal` holds the time
        ref_start,    //< `ref` names an object; `literal` is the offset
        ref_end,      //< ...
        ref_duration, //< ...
        class_start,  //< `ref` names a class: the earliest start of its members
        class_end,    //< ...the latest end
        trigger,      //< `trigger` names a cue; the time is whenever it fires
        always        //< a `while` that is always true
    };

    kind        k       = kind::literal;
    flicks      literal = 0;
    std::string ref;
    std::string trigger;

    bool is_reference() const
    {
        return k == kind::ref_start || k == kind::ref_end || k == kind::ref_duration ||
               k == kind::class_start || k == kind::class_end;
    }
};

/// The document's units, needed to turn `300f`, `00:00:12:00` or `4bars` into flicks.
struct parse_context
{
    boost::rational<int> rate{25, 1};
    tempo                tempo_{};
};

/// Parse one TIME expression. Returns false and fills `error` with something an author can act on.
///
/// NEVER throws and never guesses. An expression it cannot parse is a document defect the client
/// must be told about, by object and by expression -- which is what `timeline_invalid` carries.
bool parse_time_expr(std::string_view text, const parse_context& ctx, time_expr& out,
                     std::string& error);

/// Parse a `while` expression, where the vocabulary is different and deliberately narrower:
/// `1` or `true` means always, and anything else must be an object reference.
///
/// SEPARATE FROM `parse_time_expr` BECAUSE `1` IS AMBIGUOUS. In a time position `1` is one
/// second; in a `while` position it is "always". One parser accepting both read every `end: "1"`
/// as always-on, which resolved to time zero and gave every repeating object a zero-length span
/// -- caught by `resolver_self_test`'s repeating case, which is a good deal further from the
/// cause than a reader would like. Two entry points, two vocabularies.
bool parse_while_expr(std::string_view text, const parse_context& ctx, time_expr& out,
                      std::string& error);

/// One occurrence's activity window, as expressions.
///
/// The three time fields are over-determined on purpose: a client may give any two of start, end
/// and duration and the resolver derives the third. All three given and disagreeing is an error
/// rather than a precedence rule nobody would remember.
struct enable_spec
{
    std::optional<time_expr> start;
    std::optional<time_expr> end;
    std::optional<time_expr> duration;

    /// Active exactly while another object is, or always. Mutually exclusive with start/end.
    std::optional<time_expr> while_;

    struct
    {
        bool   on     = false;
        flicks period = 0;
        int    count  = 0; //< 0 means forever
    } repeating;

    bool empty() const { return !start && !end && !duration && !while_ && !repeating.on; }
};

/// Aborts on a grammar disagreement. Called at boot, from `resolver_self_test`.
void expression_self_test();

}}} // namespace caspar::core::timeline
