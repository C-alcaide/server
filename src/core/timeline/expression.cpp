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

#include "expression.h"

#include <common/log.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <stdexcept>
#include <vector>

namespace caspar { namespace core { namespace timeline {

namespace {

std::string_view trim(std::string_view s)
{
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front())))
        s.remove_prefix(1);
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back())))
        s.remove_suffix(1);
    return s;
}

/// Strip one balanced pair of outer parentheses, repeatedly. `((#a.end))` is `#a.end`.
///
/// Balanced, not merely first-and-last: `(#a.end) + (5)` starts with `(` and ends with `)` and
/// stripping them would leave `#a.end) + (5`, which then fails to parse -- a valid expression
/// reported as a document defect.
std::string_view unwrap(std::string_view s, bool& ok)
{
    ok = true;
    for (;;) {
        s = trim(s);
        if (s.size() < 2 || s.front() != '(' || s.back() != ')')
            return s;
        int depth = 0;
        for (std::size_t i = 0; i < s.size(); ++i) {
            if (s[i] == '(')
                ++depth;
            else if (s[i] == ')') {
                if (--depth == 0 && i + 1 != s.size())
                    return s; // the opening paren closes before the end: not an outer pair
            }
        }
        if (depth != 0) {
            ok = false;
            return s;
        }
        s = s.substr(1, s.size() - 2);
    }
}

bool all_digits(std::string_view s)
{
    return !s.empty() && std::all_of(s.begin(), s.end(), [](char c) {
        return std::isdigit(static_cast<unsigned char>(c)) != 0;
    });
}

/// A time LITERAL, in any of the four forms AMCP can express without JSON.
///
///   12.5              seconds
///   300f              frames, against the document's rate
///   00:00:12:00       timecode, hh:mm:ss:ff -- and `;` before the frames means drop-frame,
///                     which is the SMPTE convention and the only place drop-frame appears
///   4bars / 4b        bars, against the document's tempo
bool parse_literal(std::string_view s, const parse_context& ctx, flicks& out, std::string& error)
{
    s = trim(s);
    if (s.empty()) {
        error = "empty time literal";
        return false;
    }

    const bool negative = s.front() == '-';
    if (negative || s.front() == '+')
        s.remove_prefix(1);

    const auto finish = [&](flicks v) {
        out = negative ? -v : v;
        return true;
    };

    // Timecode: exactly three separators, and the last may be `;` for drop-frame.
    if (s.find(':') != std::string_view::npos) {
        int  hh = 0, mm = 0, ss = 0, ff = 0;
        bool drop  = false;
        auto parts = std::vector<std::string>{};
        std::string cur;
        for (char c : s) {
            if (c == ':' || c == ';') {
                drop = drop || c == ';';
                parts.push_back(cur);
                cur.clear();
            } else
                cur += c;
        }
        parts.push_back(cur);
        if (parts.size() != 4 || !std::all_of(parts.begin(), parts.end(), [](const std::string& p) {
                return all_digits(p);
            })) {
            error = "timecode must be hh:mm:ss:ff (or ;ff for drop-frame), got '" + std::string(s) + "'";
            return false;
        }
        hh = std::atoi(parts[0].c_str());
        mm = std::atoi(parts[1].c_str());
        ss = std::atoi(parts[2].c_str());
        ff = std::atoi(parts[3].c_str());

        const auto fps_rounded = static_cast<int>(
            (boost::rational_cast<double>(boost::rational<int>(ctx.rate)) + 0.5));
        std::int64_t frame = static_cast<std::int64_t>(hh) * 3600 * fps_rounded +
                             static_cast<std::int64_t>(mm) * 60 * fps_rounded +
                             static_cast<std::int64_t>(ss) * fps_rounded + ff;
        if (drop) {
            // The 12M rule: two frames dropped every minute except every tenth. Applied as a
            // SUBTRACTION from the label's nominal frame, which is what `to_timecode` inverts.
            const std::int64_t total_minutes = static_cast<std::int64_t>(hh) * 60 + mm;
            const int          per_minute    = fps_rounded == 60 ? 4 : 2;
            frame -= per_minute * (total_minutes - total_minutes / 10);
        }
        return finish(from_frames(frame, ctx.rate));
    }

    // Frames, bars, or plain seconds, by suffix.
    if (s.size() > 1 && (s.back() == 'f' || s.back() == 'F')) {
        const auto num = s.substr(0, s.size() - 1);
        if (!all_digits(num)) {
            error = "frame literal must be digits followed by 'f', got '" + std::string(s) + "'";
            return false;
        }
        return finish(from_frames(std::atoll(std::string(num).c_str()), ctx.rate));
    }

    const auto ends_with = [&](std::string_view suffix) {
        return s.size() > suffix.size() &&
               s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
    };
    if (ends_with("bars") || ends_with("bar") || ends_with("b") || ends_with("B")) {
        const auto cut = ends_with("bars") ? 4 : ends_with("bar") ? 3 : 1;
        const auto num = std::string(s.substr(0, s.size() - cut));
        try {
            std::size_t used = 0;
            const auto  bars = std::stod(num, &used);
            if (used != num.size())
                throw std::invalid_argument("trailing");
            return finish(from_bars(bars, ctx.tempo_));
        } catch (...) {
            error = "bar literal must be a number followed by 'bars', got '" + std::string(s) + "'";
            return false;
        }
    }

    try {
        std::size_t used = 0;
        const auto  secs = std::stod(std::string(s), &used);
        if (used != s.size())
            throw std::invalid_argument("trailing");
        return finish(from_seconds(secs));
    } catch (...) {
        error = "not a time: '" + std::string(s) + "'";
        return false;
    }
}

} // namespace

bool parse_time_expr(std::string_view text, const parse_context& ctx, time_expr& out,
                     std::string& error)
{
    out   = time_expr{};
    error.clear();

    auto s = trim(text);
    if (s.empty()) {
        error = "empty expression";
        return false;
    }

    // THE OFFSET COMES OFF FIRST, before anything decides what kind of expression this is.
    //
    // It used to be split only after the leading `#` or `.` was seen, which made
    // `(#a.end) + 5` unparseable: the string starts with `(` and ends with `5`, so the
    // paren-stripper left it alone and the classifier then read the whole thing as a literal.
    // A client that round-trips its own parenthesised formatting would have had every such
    // expression rejected as a document defect.
    //
    // A sign is an operator only when SPACE-SEPARATED from what precedes it, and only at paren
    // depth 0. Without the space rule `#lower-third` parses as `#lower` minus `third`, and an id
    // with a hyphen in it is the most natural id there is.
    std::string_view head   = s;
    std::string_view offset = {};
    {
        int depth = 0;
        for (std::size_t i = s.size(); i-- > 0;) {
            const char c = s[i];
            if (c == ')')
                ++depth;
            else if (c == '(')
                --depth;
            else if (depth == 0 && (c == '+' || c == '-') && i > 0 &&
                     std::isspace(static_cast<unsigned char>(s[i - 1]))) {
                head   = s.substr(0, i);
                offset = s.substr(i);
                break;
            }
        }
    }

    bool balanced = true;
    head          = unwrap(head, balanced);
    if (!balanced) {
        error = "unbalanced parentheses";
        return false;
    }
    if (head.empty()) {
        error = "empty expression";
        return false;
    }

    flicks offset_flicks = 0;
    if (!offset.empty()) {
        bool ok2 = true;
        offset   = unwrap(offset, ok2);
        if (!ok2) {
            error = "unbalanced parentheses in the offset";
            return false;
        }
        if (!parse_literal(offset, ctx, offset_flicks, error))
            return false;
    }

    if (head.front() != '#' && head.front() != '.') {
        if (!offset.empty()) {
            // `5 + 3` is arithmetic between literals, which v1 does not have. Refusing it is
            // better than silently answering 5: two numbers with a plus between them is
            // something the author expects to be added.
            error = "arithmetic between two literals is not supported in v1: '" +
                    std::string(text) + "'";
            return false;
        }
        return parse_literal(head, ctx, out.literal, error);
    }

    out.literal = offset_flicks;

    const bool is_class = head.front() == '.';
    head.remove_prefix(1);
    if (head.empty()) {
        error = is_class ? "a class reference needs a name" : "an object reference needs an id";
        return false;
    }

    // The trailing `.start` / `.end` / `.duration`. Found from the right, and only if it is one
    // of those three words -- an id may contain dots.
    std::string_view field;
    const auto       dot = head.rfind('.');
    if (dot != std::string_view::npos) {
        const auto tail = head.substr(dot + 1);
        if (tail == "start" || tail == "end" || tail == "duration") {
            field = tail;
            head  = head.substr(0, dot);
        }
    }
    if (head.empty()) {
        error = "reference has a field but no name";
        return false;
    }

    out.ref = std::string(head);

    if (is_class) {
        if (field == "end")
            out.k = time_expr::kind::class_end;
        else if (field.empty() || field == "start")
            out.k = time_expr::kind::class_start;
        else {
            // `.class.duration` has no defensible meaning: the earliest start and the latest end
            // belong to different objects, so their difference is not any object's duration.
            error = "a class reference takes .start or .end, not ." + std::string(field);
            return false;
        }
        return true;
    }

    if (field == "end")
        out.k = time_expr::kind::ref_end;
    else if (field == "duration")
        out.k = time_expr::kind::ref_duration;
    else
        out.k = time_expr::kind::ref_start; //< the bare `#id` form means `.start`
    return true;
}

bool parse_while_expr(std::string_view text, const parse_context& ctx, time_expr& out,
                      std::string& error)
{
    out   = time_expr{};
    error.clear();

    const auto s = trim(text);
    if (s == "1" || s == "true") {
        out.k = time_expr::kind::always;
        return true;
    }
    if (s.empty()) {
        error = "empty while expression";
        return false;
    }
    if (s.front() != '#') {
        // Not a literal, and not a class either: `while` means "for as long as THAT object is
        // active", and a class has no single span to be active for -- its earliest start and its
        // latest end belong to different objects.
        error = "a while expression is `1` or an object reference such as `#interview`, not '" +
                std::string(s) + "'";
        return false;
    }
    if (!parse_time_expr(s, ctx, out, error))
        return false;
    if (out.k != time_expr::kind::ref_start) {
        error = "a while expression names an object, without .start, .end or .duration";
        return false;
    }
    return true;
}

void expression_self_test()
{
    const auto req = [](bool ok, const char* what) {
        if (!ok) {
            CASPAR_LOG(fatal) << L"timeline::expression_self_test: " << what;
            std::abort();
        }
    };

    parse_context ctx;
    ctx.rate = boost::rational<int>(25, 1);
    time_expr e;
    std::string err;

    // ---- literals, in all four forms ----------------------------------------------------
    req(parse_time_expr("12.5", ctx, e, err), "a seconds literal parses");
    req(e.k == time_expr::kind::literal && e.literal == from_seconds(12.5), "and is 12.5 s");

    req(parse_time_expr("300f", ctx, e, err), "a frame literal parses");
    req(e.literal == from_frames(300, ctx.rate), "and is frame 300 at the document's rate");

    req(parse_time_expr("00:00:12:00", ctx, e, err), "a timecode literal parses");
    req(e.literal == from_frames(300, ctx.rate), "12 s at 25p is frame 300");

    {
        // Drop-frame, at 29.97. The two anchors `time.cpp` already checks, read the other way.
        parse_context df;
        df.rate = boost::rational<int>(30000, 1001);
        req(parse_time_expr("00:01:00;02", df, e, err), "a drop-frame label parses");
        req(e.literal == from_frames(1800, df.rate), "00:01:00;02 is frame 1800");
        req(parse_time_expr("00:10:00;00", df, e, err), "and the tenth-minute anchor");
        req(e.literal == from_frames(17982, df.rate), "00:10:00;00 is frame 17982");
    }

    req(parse_time_expr("4bars", ctx, e, err), "a bar literal parses");
    req(e.literal == from_bars(4.0, ctx.tempo_), "against the document's tempo");

    req(!parse_time_expr("", ctx, e, err), "the empty expression is refused");
    req(!parse_time_expr("banana", ctx, e, err), "and so is a word");
    req(!parse_time_expr("00:12:00", ctx, e, err), "and a three-part timecode");
    req(!parse_time_expr("12.5.5", ctx, e, err), "and a number with two points");

    // ---- references ---------------------------------------------------------------------
    req(parse_time_expr("#interview", ctx, e, err), "a bare object reference parses");
    req(e.k == time_expr::kind::ref_start && e.ref == "interview" && e.literal == 0,
        "and means its START with no offset");

    req(parse_time_expr("#interview.end", ctx, e, err), "an explicit .end parses");
    req(e.k == time_expr::kind::ref_end && e.ref == "interview", "as a ref_end");

    req(parse_time_expr("#interview.duration", ctx, e, err), ".duration parses");
    req(e.k == time_expr::kind::ref_duration, "as a ref_duration");

    req(parse_time_expr("#interview.end + 5", ctx, e, err), "one offset parses");
    req(e.k == time_expr::kind::ref_end && e.literal == from_seconds(5.0), "and is +5 s");

    req(parse_time_expr("#interview.end - 2.5", ctx, e, err), "a negative offset parses");
    req(e.literal == from_seconds(-2.5), "and is -2.5 s");

    req(parse_time_expr("(#interview.end) + 5", ctx, e, err), "parentheses are tolerated");
    req(e.k == time_expr::kind::ref_end && e.literal == from_seconds(5.0), "and change nothing");
    req(!parse_time_expr("(#a.end + 5", ctx, e, err), "an unbalanced paren is refused");
    req(parse_time_expr("((#a.end)) - 1", ctx, e, err), "nested parentheses are tolerated");
    req(e.k == time_expr::kind::ref_end && e.literal == from_seconds(-1.0), "and still offset");
    req(!parse_time_expr("5 + 3", ctx, e, err),
        "arithmetic between two literals is REFUSED rather than silently answering 5");

    // THE HYPHEN RULE. An id with a hyphen is the most natural id there is, and finding the
    // offset sign without requiring a space made `#lower-third` parse as `#lower` minus `third`.
    req(parse_time_expr("#lower-third.end", ctx, e, err), "a hyphenated id parses whole");
    req(e.ref == "lower-third", "as `lower-third`, not `lower`");
    req(parse_time_expr("#lower-third.end - 1", ctx, e, err), "and still takes an offset");
    req(e.ref == "lower-third" && e.literal == from_seconds(-1.0), "with the id intact");

    // ---- class references ---------------------------------------------------------------
    req(parse_time_expr(".lowerthird.start + 5", ctx, e, err), "a class reference parses");
    req(e.k == time_expr::kind::class_start && e.ref == "lowerthird", "as a class_start");
    req(e.literal == from_seconds(5.0), "with its offset");
    req(parse_time_expr(".lowerthird.end", ctx, e, err), "and .end");
    req(e.k == time_expr::kind::class_end, "as a class_end");
    req(parse_time_expr(".lowerthird", ctx, e, err), "a bare class reference parses");
    req(e.k == time_expr::kind::class_start, "and means the earliest start");
    req(!parse_time_expr(".lowerthird.duration", ctx, e, err),
        "but .duration on a class is refused -- the earliest start and the latest end belong to "
        "different objects, so their difference is nobody's duration");
    req(!parse_time_expr(".", ctx, e, err), "a class reference needs a name");
    req(!parse_time_expr("#", ctx, e, err), "and so does an object reference");

    // ---- the `while` forms, which are a DIFFERENT vocabulary --------------------------
    req(parse_while_expr("1", ctx, e, err), "`1` parses in a while position");
    req(e.k == time_expr::kind::always, "as always-active");
    req(parse_while_expr("true", ctx, e, err), "and so does `true`");

    req(parse_while_expr("#interview", ctx, e, err), "an object reference parses");
    req(e.k == time_expr::kind::ref_start && e.ref == "interview", "as that object");

    // AND `1` IN A TIME POSITION IS ONE SECOND, not "always". One parser accepting both read
    // every `end: "1"` as always-on, which resolved to time zero and gave every repeating object
    // a zero-length span -- found by the resolver's repeating case, a long way from the cause.
    req(parse_time_expr("1", ctx, e, err), "`1` parses in a TIME position");
    req(e.k == time_expr::kind::literal && e.literal == from_seconds(1.0), "as one second");
    req(!parse_while_expr("12.5", ctx, e, err), "a literal is refused in a while position");
    req(!parse_while_expr(".lt", ctx, e, err),
        "and so is a class -- its earliest start and latest end belong to different objects, so "
        "there is no single span to be active for");
    req(!parse_while_expr("#a.end", ctx, e, err), "and so is a field reference");
}

}}} // namespace caspar::core::timeline
