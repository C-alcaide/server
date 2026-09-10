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

#include "api_timeline.h"

#include "json_state.h"

#include <core/timeline/curve.h>
#include <core/timeline/timeline_store.h>

#include <common/log.h>

#include <algorithm>
#include <cmath>

namespace caspar { namespace protocol { namespace http {

namespace tl = core::timeline;

namespace {

// =======================================================================================
// Decoding
// =======================================================================================

/// A parse failure that names the object it happened in.
struct decode_error
{
    std::string object;
    std::string expression;
    std::string reason;

    bool bad() const { return !reason.empty(); }
};

const json::value* member(const json::object& o, const char* key)
{
    const auto it = o.find(key);
    return it == o.end() ? nullptr : &it->value();
}

bool as_double(const json::value& v, double& out)
{
    if (v.is_double()) {
        out = v.as_double();
        return true;
    }
    if (v.is_int64()) {
        out = static_cast<double>(v.as_int64());
        return true;
    }
    if (v.is_uint64()) {
        out = static_cast<double>(v.as_uint64());
        return true;
    }
    return false;
}

/// A TIME, in any of the four wire forms.
///
///   12.5                     seconds, the canonical form
///   {"frames": 300}          against the document's rate
///   {"tc": "00:00:12:00"}    ...or its drop-frame spelling, `00:01:00;02`
///   {"bars": 4}              against the document's tempo
///
/// A STRING is not a time here -- it is an expression, and goes to the grammar. That split is
/// what keeps `"#a.end + 5"` and `12.5` from needing to be told apart by inspection.
bool decode_time(const json::value& v, const tl::parse_context& ctx, tl::flicks& out,
                 std::string& reason)
{
    double d = 0;
    if (as_double(v, d)) {
        out = tl::from_seconds(d);
        return true;
    }
    if (v.is_object()) {
        const auto& o = v.as_object();
        if (const auto* f = member(o, "frames")) {
            if (!as_double(*f, d)) {
                reason = "\"frames\" takes a number";
                return false;
            }
            out = tl::from_frames(static_cast<std::int64_t>(std::llround(d)), ctx.rate);
            return true;
        }
        if (const auto* b = member(o, "bars")) {
            if (!as_double(*b, d)) {
                reason = "\"bars\" takes a number";
                return false;
            }
            out = tl::from_bars(d, ctx.tempo_);
            return true;
        }
        if (const auto* t = member(o, "tc")) {
            if (!t->is_string()) {
                reason = "\"tc\" takes a timecode string";
                return false;
            }
            tl::time_expr e;
            if (!tl::parse_time_expr(t->as_string().c_str(), ctx, e, reason))
                return false;
            if (e.k != tl::time_expr::kind::literal) {
                reason = "\"tc\" takes a timecode, not an expression";
                return false;
            }
            out = e.literal;
            return true;
        }
        reason = "a time object takes \"frames\", \"tc\" or \"bars\"";
        return false;
    }
    reason = "a time is a number of seconds, or {\"frames\"|\"tc\"|\"bars\": ...}";
    return false;
}

/// A time EXPRESSION: a string goes to the grammar, anything else is a literal time.
bool decode_expr(const json::value& v, const tl::parse_context& ctx, tl::time_expr& out,
                 std::string& reason)
{
    if (v.is_string())
        return tl::parse_time_expr(v.as_string().c_str(), ctx, out, reason);

    if (v.is_object()) {
        // `{"trigger": "go"}` -- an end that waits for a cue. The only expression form that is
        // an object rather than a string, because a trigger name is not a time and giving it
        // string syntax (`"@go"`) would add a sigil to a grammar that has two already.
        const auto& o = v.as_object();
        if (const auto* t = member(o, "trigger")) {
            if (!t->is_string()) {
                reason = "\"trigger\" takes a name";
                return false;
            }
            out         = tl::time_expr{};
            out.k       = tl::time_expr::kind::trigger;
            out.trigger = t->as_string().c_str();
            return true;
        }
    }

    out = tl::time_expr{};
    return decode_time(v, ctx, out.literal, reason);
}

bool decode_enable(const json::value& v, const tl::parse_context& ctx, tl::enable_spec& out,
                   std::string& reason)
{
    if (!v.is_object()) {
        reason = "an enable is an object with start/end/duration, or while, or repeating";
        return false;
    }
    const auto& o = v.as_object();

    const auto one = [&](const char* key, std::optional<tl::time_expr>& dst) {
        if (const auto* m = member(o, key)) {
            tl::time_expr e;
            if (!decode_expr(*m, ctx, e, reason)) {
                reason = std::string(key) + ": " + reason;
                return false;
            }
            dst = e;
        }
        return true;
    };

    if (!one("start", out.start) || !one("end", out.end) || !one("duration", out.duration))
        return false;

    if (const auto* w = member(o, "while")) {
        tl::time_expr e;
        if (w->is_string()) {
            if (!tl::parse_while_expr(w->as_string().c_str(), ctx, e, reason)) {
                reason = "while: " + reason;
                return false;
            }
        } else if (w->is_bool() && w->as_bool()) {
            e.k = tl::time_expr::kind::always;
        } else if (as_double(*w, *(new double(0)))) {
            // Deliberately not accepted: `while: 1` as a NUMBER is the ambiguity that
            // `parse_while_expr` exists to remove. A client means "always", and a number in
            // every other position in this document is a time -- so it is refused and the
            // message says what to send.
            reason = "while: send `true` or `\"1\"` for always, or `\"#id\"` -- a bare number "
                     "would be a time everywhere else in this document";
            return false;
        } else {
            reason = "while: takes true, \"1\" or \"#id\"";
            return false;
        }
        out.while_ = e;
    }

    if (const auto* r = member(o, "repeating")) {
        if (!r->is_object()) {
            reason = "repeating takes {\"period\": <time>, \"count\": <n>}";
            return false;
        }
        const auto& ro = r->as_object();
        out.repeating.on = true;
        if (const auto* p = member(ro, "period")) {
            if (!decode_time(*p, ctx, out.repeating.period, reason)) {
                reason = "repeating.period: " + reason;
                return false;
            }
        } else {
            reason = "repeating needs a period";
            return false;
        }
        if (const auto* c = member(ro, "count")) {
            double d = 0;
            if (!as_double(*c, d) || d < 0) {
                reason = "repeating.count takes a non-negative number; 0 means forever";
                return false;
            }
            out.repeating.count = static_cast<int>(d);
        }
    }

    if (out.while_ && (out.start || out.end)) {
        reason = "an enable takes either while, or start/end -- not both";
        return false;
    }
    return true;
}

bool decode_values(const json::object& o, std::map<std::string, core::monitor::vector_t>& out,
                   std::string& reason)
{
    for (const auto& kv : o) {
        core::monitor::vector_t vec;
        const auto&             v = kv.value();
        const auto              push = [&](const json::value& e) {
            double d = 0;
            if (e.is_bool())
                vec.push_back(e.as_bool());
            else if (e.is_string())
                vec.push_back(std::string(e.as_string().c_str()));
            else if (as_double(e, d))
                vec.push_back(d);
            else
                return false;
            return true;
        };
        if (v.is_array()) {
            for (const auto& e : v.as_array())
                if (!push(e)) {
                    reason = std::string(kv.key()) + ": array elements must be numbers, booleans "
                                                     "or names";
                    return false;
                }
        } else if (!push(v)) {
            reason = std::string(kv.key()) + ": a value is a number, a boolean, a name, or an "
                                             "array of those";
            return false;
        }
        out[std::string(kv.key())] = std::move(vec);
    }
    return true;
}

bool decode_curve(const json::value& v, const tl::parse_context& ctx, const std::string& default_easing,
                  tl::curve& out, std::string& reason)
{
    if (!v.is_array()) {
        reason = "keys takes an array";
        return false;
    }
    for (const auto& kv : v.as_array()) {
        if (!kv.is_object()) {
            reason = "each key is an object with a time and values";
            return false;
        }
        const auto& o = kv.as_object();

        tl::curve_key key;
        const auto*   t = member(o, "at");
        if (!t)
            t = member(o, "time");
        if (!t) {
            reason = "a key needs \"at\"";
            return false;
        }
        if (!decode_time(*t, ctx, key.time, reason))
            return false;

        // THE DOCUMENT'S `defaults.easing` IS RESOLVED INTO EACH KEY HERE, at PUT (D11). The
        // alternative -- carrying the default and consulting it at evaluation -- means the
        // engine has to know about documents, and means a client that reads a key back cannot
        // see what it will actually do.
        key.easing_name = default_easing;
        if (const auto* e = member(o, "easing")) {
            if (!e->is_string()) {
                reason = "easing takes a name";
                return false;
            }
            key.easing_name = e->as_string().c_str();
        }
        if (!tl::easing_exists(key.easing_name)) {
            reason = "no such easing: '" + key.easing_name + "'";
            return false;
        }
        key.ease = tl::easing_from_name(key.easing_name);

        const auto* vals = member(o, "values");
        if (!vals || !vals->is_object()) {
            reason = "a key needs \"values\": {\"<path>\": <number>}";
            return false;
        }
        for (const auto& pv : vals->as_object()) {
            double d = 0;
            if (!as_double(pv.value(), d)) {
                reason = std::string(pv.key()) +
                         ": a curve value is a number. Step values -- enums, booleans, names -- "
                         "go in \"content\" or \"keyframes\", which merge rather than interpolate";
                return false;
            }
            key.values[std::string(pv.key())] = d;
        }
        out.add(std::move(key));
    }
    return true;
}

decode_error decode_object(const json::value& v, const tl::parse_context& parent_ctx,
                           const std::string& default_easing, tl::timeline_object& out);

decode_error decode_children(const json::value& v, const tl::parse_context& ctx,
                             const std::string& default_easing, tl::timeline_object& out)
{
    if (!v.is_array())
        return {out.id, "", "children takes an array"};
    for (const auto& c : v.as_array()) {
        tl::timeline_object child;
        if (const auto e = decode_object(c, ctx, default_easing, child); e.bad())
            return e;
        out.children.push_back(std::move(child));
    }
    return {};
}

decode_error decode_object(const json::value& v, const tl::parse_context& parent_ctx,
                           const std::string& default_easing, tl::timeline_object& out)
{
    if (!v.is_object())
        return {"", "", "each object in \"objects\" is a JSON object"};
    const auto& o = v.as_object();

    const auto* id = member(o, "id");
    if (!id || !id->is_string() || id->as_string().empty())
        return {"", "", "every object needs a non-empty string \"id\" -- references and published "
                        "ownership both key on it"};
    out.id = id->as_string().c_str();

    // A group's own tempo applies to ITS contents, so the context is extended before the
    // children and the enables are read. That is what makes a musical island inside a video
    // document work: `4bars` inside the group means four of the group's bars.
    auto ctx = parent_ctx;

    if (const auto* r = member(o, "remap")) {
        if (!r->is_object())
            return {out.id, "", "remap takes {\"rate\": <n>, \"offset\": <time>}"};
        const auto& ro = r->as_object();
        if (const auto* rate = member(ro, "rate")) {
            double d = 0;
            if (!as_double(*rate, d) || d == 0.0)
                return {out.id, "", "remap.rate takes a non-zero number"};
            // A rational from a double, at 1/1000 resolution, so nesting composes exactly and
            // `0.5` inside `2` is exactly 1 rather than 0.99999999999999989.
            out.remap.rate = boost::rational<std::int64_t>(static_cast<std::int64_t>(std::llround(d * 1000)), 1000);
        }
        if (const auto* off = member(ro, "offset")) {
            std::string reason;
            if (!decode_time(*off, ctx, out.remap.offset, reason))
                return {out.id, "", "remap.offset: " + reason};
        }
    }

    if (const auto* t = member(o, "tempo")) {
        if (!t->is_object())
            return {out.id, "", "tempo takes {\"bpm\": <n>, \"beats_per_bar\": <n>}"};
        tl::tempo tm;
        double    d = 0;
        if (const auto* bpm = member(t->as_object(), "bpm")) {
            if (!as_double(*bpm, d) || d <= 0)
                return {out.id, "", "tempo.bpm takes a positive number"};
            tm.bpm = d;
        }
        if (const auto* bpb = member(t->as_object(), "beats_per_bar")) {
            if (!as_double(*bpb, d) || d < 1)
                return {out.id, "", "tempo.beats_per_bar takes a number of at least 1"};
            tm.beats_per_bar = static_cast<int>(d);
        }
        out.tempo_  = tm;
        ctx.tempo_  = tm;
    }

    if (const auto* l = member(o, "layer")) {
        if (!l->is_string())
            return {out.id, "", "layer takes a string such as \"1-10\", or is absent for a "
                                "transparent anchor"};
        out.layer = l->as_string().c_str();
    }

    if (const auto* p = member(o, "priority")) {
        double d = 0;
        if (!as_double(*p, d))
            return {out.id, "", "priority takes a number"};
        out.priority = static_cast<int>(d);
    }

    if (const auto* d = member(o, "disabled")) {
        if (!d->is_bool())
            return {out.id, "", "disabled takes a boolean"};
        out.disabled = d->as_bool();
    }

    if (const auto* r = member(o, "rebase")) {
        if (!r->is_bool())
            return {out.id, "", "rebase takes a boolean"};
        out.rebase = r->as_bool();
    }

    if (const auto* oe = member(o, "on_end")) {
        if (!oe->is_string())
            return {out.id, "", "on_end takes \"release\" or \"commit\""};
        const std::string s = oe->as_string().c_str();
        if (s == "release")
            out.on_end = tl::on_end_t::release;
        else if (s == "commit")
            out.on_end = tl::on_end_t::commit;
        else
            return {out.id, "", "on_end takes \"release\" or \"commit\", not \"" + s + "\""};
    }

    if (const auto* c = member(o, "classes")) {
        if (!c->is_array())
            return {out.id, "", "classes takes an array of names"};
        for (const auto& e : c->as_array()) {
            if (!e.is_string())
                return {out.id, "", "a class is a name"};
            out.classes.emplace_back(e.as_string().c_str());
        }
    }

    if (const auto* e = member(o, "enable")) {
        const auto one = [&](const json::value& spec) -> decode_error {
            tl::enable_spec s;
            std::string     reason;
            if (!decode_enable(spec, ctx, s, reason))
                return {out.id, "", reason};
            out.enable.push_back(std::move(s));
            return {};
        };
        if (e->is_array()) {
            for (const auto& spec : e->as_array())
                if (const auto err = one(spec); err.bad())
                    return err;
        } else if (const auto err = one(*e); err.bad())
            return err;
    }

    if (const auto* c = member(o, "content")) {
        if (!c->is_object())
            return {out.id, "", "content takes {\"<path>\": <value>}"};
        std::string reason;
        if (!decode_values(c->as_object(), out.content, reason))
            return {out.id, "", "content: " + reason};
    }

    if (const auto* k = member(o, "keyframes")) {
        if (!k->is_array())
            return {out.id, "", "keyframes takes an array of step keyframes"};
        for (const auto& kf : k->as_array()) {
            if (!kf.is_object())
                return {out.id, "", "each step keyframe is an object"};
            tl::content_keyframe ck;
            const auto&          ko = kf.as_object();
            if (const auto* kid = member(ko, "id"); kid && kid->is_string())
                ck.id = kid->as_string().c_str();
            if (const auto* en = member(ko, "enable")) {
                std::string reason;
                if (!decode_enable(*en, ctx, ck.enable, reason))
                    return {out.id, "", "keyframes.enable: " + reason};
            } else if (const auto* at = member(ko, "at")) {
                tl::enable_spec s;
                tl::time_expr   e2;
                std::string     reason;
                if (!decode_time(*at, ctx, e2.literal, reason))
                    return {out.id, "", "keyframes.at: " + reason};
                s.start  = e2;
                ck.enable = s;
            } else {
                return {out.id, "", "a step keyframe needs \"at\" or \"enable\""};
            }
            const auto* vals = member(ko, "content");
            if (!vals || !vals->is_object())
                return {out.id, "", "a step keyframe needs \"content\""};
            std::string reason;
            if (!decode_values(vals->as_object(), ck.content, reason))
                return {out.id, "", "keyframes.content: " + reason};
            out.keyframes.push_back(std::move(ck));
        }
    }

    if (const auto* k = member(o, "keys")) {
        std::string reason;
        if (!decode_curve(*k, ctx, default_easing, out.curves, reason))
            return {out.id, "", reason};
    }

    if (const auto* a = member(o, "action")) {
        if (!a->is_string())
            return {out.id, "", "action takes a verb name"};
        const std::string s = a->as_string().c_str();
        if (s == "play")
            out.action = tl::layer_action::play;
        else if (s == "load")
            out.action = tl::layer_action::load;
        else if (s == "pause")
            out.action = tl::layer_action::pause;
        else if (s == "resume")
            out.action = tl::layer_action::resume;
        else if (s == "stop")
            out.action = tl::layer_action::stop;
        else if (s == "clear")
            out.action = tl::layer_action::clear;
        else if (s == "none")
            out.action = tl::layer_action::none;
        else
            return {out.id, "", "no such action: \"" + s + "\""};
    }

    if (const auto* c = member(o, "clip")) {
        if (!c->is_string())
            return {out.id, "", "clip takes a name"};
        out.clip = u16(std::string(c->as_string().c_str()));
    }

    if (const auto* p = member(o, "preroll_frames")) {
        double d = 0;
        if (!as_double(*p, d) || d < 0)
            return {out.id, "", "preroll_frames takes a non-negative number"};
        out.preroll_frames = static_cast<int>(d);
    }

    if (const auto* g = member(o, "group"); g && g->is_object()) {
        out.is_group  = true;
        const auto& go = g->as_object();
        const auto  flag = [&](const char* key, bool& dst) -> bool {
            if (const auto* m = member(go, key)) {
                if (!m->is_bool())
                    return false;
                dst = m->as_bool();
            }
            return true;
        };
        if (!flag("one_at_a_time", out.play.one_at_a_time) ||
            !flag("auto_play", out.play.auto_play) || !flag("loop", out.play.loop))
            return {out.id, "", "group flags take booleans"};
    }

    if (const auto* c = member(o, "children")) {
        out.is_group = true;
        if (const auto err = decode_children(*c, ctx, default_easing, out); err.bad())
            return err;
    }

    return {};
}

// =======================================================================================
// Encoding
// =======================================================================================

json::value encode_time(tl::flicks t) { return tl::to_seconds(t); }

json::value encode_expr(const tl::time_expr& e)
{
    // Re-encoded as a STRING wherever it was one, so a client's GET round-trips to the same
    // document it sent. A literal comes back as a plain number, which is the canonical form --
    // a client that sent `{"frames": 300}` gets `12` back, and the doc says so.
    switch (e.k) {
        case tl::time_expr::kind::literal:
            return encode_time(e.literal);
        case tl::time_expr::kind::always:
            return "1";
        case tl::time_expr::kind::trigger: {
            json::object o;
            o["trigger"] = e.trigger;
            return o;
        }
        default:
            break;
    }

    std::string s;
    switch (e.k) {
        case tl::time_expr::kind::ref_start:
            s = "#" + e.ref;
            break;
        case tl::time_expr::kind::ref_end:
            s = "#" + e.ref + ".end";
            break;
        case tl::time_expr::kind::ref_duration:
            s = "#" + e.ref + ".duration";
            break;
        case tl::time_expr::kind::class_start:
            s = "." + e.ref + ".start";
            break;
        case tl::time_expr::kind::class_end:
            s = "." + e.ref + ".end";
            break;
        default:
            break;
    }
    if (e.literal > 0)
        s += " + " + std::to_string(tl::to_seconds(e.literal));
    else if (e.literal < 0)
        s += " - " + std::to_string(-tl::to_seconds(e.literal));
    return json::value(s);
}

json::value encode_enable(const tl::enable_spec& s)
{
    json::object o;
    if (s.start)
        o["start"] = encode_expr(*s.start);
    if (s.end)
        o["end"] = encode_expr(*s.end);
    if (s.duration)
        o["duration"] = encode_expr(*s.duration);
    if (s.while_)
        o["while"] = encode_expr(*s.while_);
    if (s.repeating.on) {
        json::object r;
        r["period"]     = encode_time(s.repeating.period);
        r["count"]      = s.repeating.count;
        o["repeating"]  = r;
    }
    return o;
}

json::value encode_values(const std::map<std::string, core::monitor::vector_t>& m)
{
    json::object o;
    for (const auto& kv : m)
        o[kv.first] = vector_to_json(kv.second);
    return o;
}

json::value encode_object(const tl::timeline_object& o)
{
    json::object j;
    j["id"] = o.id;
    if (!o.layer.empty())
        j["layer"] = o.layer;
    if (o.priority != 0)
        j["priority"] = o.priority;
    if (o.disabled)
        j["disabled"] = true;
    if (o.rebase)
        j["rebase"] = true;
    if (o.on_end == tl::on_end_t::commit)
        j["on_end"] = "commit";

    if (!o.classes.empty()) {
        json::array c;
        for (const auto& s : o.classes)
            c.push_back(json::value(s));
        j["classes"] = c;
    }

    if (!o.enable.empty()) {
        json::array e;
        for (const auto& s : o.enable)
            e.push_back(encode_enable(s));
        j["enable"] = e;
    }

    if (o.remap.rate != 1 || o.remap.offset != 0) {
        json::object r;
        r["rate"]   = boost::rational_cast<double>(o.remap.rate);
        r["offset"] = encode_time(o.remap.offset);
        j["remap"]  = r;
    }
    if (o.tempo_) {
        json::object t;
        t["bpm"]           = o.tempo_->bpm;
        t["beats_per_bar"] = o.tempo_->beats_per_bar;
        j["tempo"]         = t;
    }

    if (!o.content.empty())
        j["content"] = encode_values(o.content);

    if (!o.keyframes.empty()) {
        json::array a;
        for (const auto& k : o.keyframes) {
            json::object kj;
            if (!k.id.empty())
                kj["id"] = k.id;
            kj["enable"]  = encode_enable(k.enable);
            kj["content"] = encode_values(k.content);
            a.push_back(kj);
        }
        j["keyframes"] = a;
    }

    if (!o.curves.empty()) {
        json::array a;
        for (const auto& k : o.curves.keys()) {
            json::object kj;
            kj["at"]     = encode_time(k.time);
            kj["easing"] = k.easing_name;
            json::object vals;
            for (const auto& pv : k.values)
                vals[pv.first] = pv.second;
            kj["values"] = vals;
            a.push_back(kj);
        }
        j["keys"] = a;
    }

    if (o.action != tl::layer_action::none) {
        const char* n = "none";
        switch (o.action) {
            case tl::layer_action::play: n = "play"; break;
            case tl::layer_action::load: n = "load"; break;
            case tl::layer_action::pause: n = "pause"; break;
            case tl::layer_action::resume: n = "resume"; break;
            case tl::layer_action::stop: n = "stop"; break;
            case tl::layer_action::clear: n = "clear"; break;
            default: break;
        }
        j["action"] = n;
    }
    if (o.clip)
        j["clip"] = u8(*o.clip);
    if (o.preroll_frames != 25)
        j["preroll_frames"] = o.preroll_frames;

    if (o.is_group) {
        json::object g;
        g["one_at_a_time"] = o.play.one_at_a_time;
        g["auto_play"]     = o.play.auto_play;
        g["loop"]          = o.play.loop;
        j["group"]         = g;

        json::array kids;
        for (const auto& c : o.children)
            kids.push_back(encode_object(c));
        j["children"] = kids;
    }
    return j;
}

json::value encode_document(const tl::timeline_document& d)
{
    json::object j;
    j["name"]     = d.name;
    j["channel"]  = d.channel;
    j["revision"] = d.revision;
    j["rate"]     = boost::rational_cast<double>(d.rate);
    if (d.tempo_) {
        json::object t;
        t["bpm"]           = d.tempo_->bpm;
        t["beats_per_bar"] = d.tempo_->beats_per_bar;
        j["tempo"]         = t;
    }
    json::object defaults;
    defaults["easing"] = d.defaults.easing;
    j["defaults"]      = defaults;

    json::array objs;
    for (const auto& o : d.objects)
        objs.push_back(encode_object(o));
    j["objects"] = objs;
    return j;
}

json::array encode_faults(const tl::resolved_timeline& r)
{
    json::array a;
    for (const auto& e : r.errors) {
        json::object o;
        o["object"]     = e.object;
        o["expression"] = e.expression;
        o["reason"]     = e.reason;
        a.push_back(o);
    }
    return a;
}

json::value encode_resolution(const tl::resolved_timeline& r)
{
    json::object j;
    j["revision"] = r.revision;
    j["ok"]       = r.ok();

    json::array insts;
    for (const auto& i : r.instances) {
        json::object o;
        o["object"] = i.object_id;
        if (!i.layer.empty())
            o["layer"] = i.layer;
        o["start"] = encode_time(i.start);
        // An OPEN end is `null`, not a large number. A client drawing a bar has to know the
        // difference between "runs until told" and "runs until 10:00" -- see `resolver.h`.
        o["end"]      = i.end ? json::value(encode_time(*i.end)) : json::value(nullptr);
        o["repeat"]   = i.repeat_index;
        o["priority"] = i.priority;
        insts.push_back(o);
    }
    j["instances"] = insts;

    json::array pending;
    for (const auto& t : r.pending_triggers)
        pending.push_back(json::value(t));
    j["pending_triggers"] = pending;

    if (!r.ok())
        j["faults"] = encode_faults(r);
    return j;
}

/// The object an instance came from, by id, depth-first.
const tl::timeline_object* find_object(const std::vector<tl::timeline_object>& objs, const std::string& id)
{
    for (const auto& o : objs) {
        if (o.id == id)
            return &o;
        if (!o.children.empty())
            if (const auto* c = find_object(o.children, id))
                return c;
    }
    return nullptr;
}

std::string query_value(const std::string& query, const std::string& key)
{
    // `at=12.5`, `?at=12.5`, or one of several separated by `&`. Hand-parsed because this is the
    // only endpoint in the API with a query parameter that carries a value, and pulling in a URL
    // parser for one key would be the larger change.
    auto q = query;
    if (!q.empty() && q.front() == '?')
        q.erase(0, 1);
    std::size_t pos = 0;
    while (pos < q.size()) {
        const auto amp  = q.find('&', pos);
        const auto part = q.substr(pos, amp == std::string::npos ? std::string::npos : amp - pos);
        const auto eq   = part.find('=');
        if (eq != std::string::npos && part.substr(0, eq) == key)
            return part.substr(eq + 1);
        if (amp == std::string::npos)
            break;
        pos = amp + 1;
    }
    return {};
}

} // namespace

api_reply parse_timeline_document(const std::string& body, const std::string& name,
                                  tl::timeline_document& out)
{
    json::error_code ec;
    const auto       parsed = json::parse(body, ec);
    if (ec || !parsed.is_object())
        return api_reply::fail(api_code::bad_request,
                               "a timeline document is a JSON object: " + ec.message());

    const auto& o = parsed.as_object();

    out      = tl::timeline_document{};
    out.name = name;

    if (const auto* n = member(o, "name"); n && n->is_string()) {
        const std::string in_body = n->as_string().c_str();
        if (!name.empty() && in_body != name)
            return api_reply::fail(api_code::bad_request,
                                   "the document's \"name\" is \"" + in_body +
                                       "\" and the path says \"" + name +
                                       "\" -- one of them is wrong, and guessing which would put "
                                       "the document under a name the client does not expect");
        out.name = in_body;
    }
    if (out.name.empty())
        return api_reply::fail(api_code::bad_request, "a timeline needs a name");

    if (const auto* c = member(o, "channel")) {
        double d = 0;
        if (!as_double(*c, d) || d < 1)
            return api_reply::fail(api_code::bad_request, "channel takes an index of at least 1");
        out.channel = static_cast<int>(d);
    }

    if (const auto* r = member(o, "rate")) {
        double d = 0;
        if (as_double(*r, d) && d > 0) {
            // A rational at 1/1001 resolution, so `29.97` becomes exactly 30000/1001 rather than
            // a value that misses a frame boundary every few minutes. `to_frames` floors, so a
            // rate that is a hair low would put a key one frame early once an hour.
            const auto num = static_cast<int>(std::llround(d * 1001));
            out.rate       = boost::rational<int>(num, 1001);
        } else if (r->is_object()) {
            const auto* n = member(r->as_object(), "num");
            const auto* dd = member(r->as_object(), "den");
            double      nv = 0, dv = 0;
            if (!n || !dd || !as_double(*n, nv) || !as_double(*dd, dv) || dv == 0)
                return api_reply::fail(api_code::bad_request,
                                        "rate takes a number, or {\"num\": n, \"den\": d}");
            out.rate = boost::rational<int>(static_cast<int>(nv), static_cast<int>(dv));
        } else {
            return api_reply::fail(api_code::bad_request, "rate takes a positive number");
        }
    }

    tl::parse_context ctx;
    ctx.rate = out.rate;

    if (const auto* t = member(o, "tempo")) {
        if (!t->is_object())
            return api_reply::fail(api_code::bad_request,
                                    "tempo takes {\"bpm\": n, \"beats_per_bar\": n}");
        tl::tempo tm;
        double    d = 0;
        if (const auto* bpm = member(t->as_object(), "bpm")) {
            if (!as_double(*bpm, d) || d <= 0)
                return api_reply::fail(api_code::bad_request, "tempo.bpm takes a positive number");
            tm.bpm = d;
        }
        if (const auto* bpb = member(t->as_object(), "beats_per_bar")) {
            if (!as_double(*bpb, d) || d < 1)
                return api_reply::fail(api_code::bad_request,
                                        "tempo.beats_per_bar takes a number of at least 1");
            tm.beats_per_bar = static_cast<int>(d);
        }
        out.tempo_ = tm;
        ctx.tempo_ = tm;
    }

    if (const auto* d = member(o, "defaults"); d && d->is_object()) {
        if (const auto* e = member(d->as_object(), "easing")) {
            if (!e->is_string())
                return api_reply::fail(api_code::bad_request, "defaults.easing takes a name");
            out.defaults.easing = e->as_string().c_str();
            if (!tl::easing_exists(out.defaults.easing))
                return api_reply::fail(api_code::bad_request,
                                        "no such easing: '" + out.defaults.easing + "'");
        }
    }

    const auto* objs = member(o, "objects");
    if (!objs || !objs->is_array())
        return api_reply::fail(api_code::bad_request, "a timeline needs an \"objects\" array");

    for (const auto& ov : objs->as_array()) {
        tl::timeline_object obj;
        if (const auto e = decode_object(ov, ctx, out.defaults.easing, obj); e.bad()) {
            json::object d;
            d["object"]     = e.object;
            d["expression"] = e.expression;
            d["reason"]     = e.reason;
            return api_reply::fail(api_code::bad_request,
                                    (e.object.empty() ? std::string("object") : "object '" + e.object + "'") +
                                        ": " + e.reason,
                                    json::array{d});
        }
        out.objects.push_back(std::move(obj));
    }
    return api_reply{};
}

api_reply put_timeline(const api_context& ctx, const std::string& name, const std::string& body)
{
    if (!ctx.timelines)
        return api_reply::fail(api_code::internal, "no timeline store is wired into this build");

    tl::timeline_document doc;
    if (auto r = parse_timeline_document(body, name, doc); r.code != api_code::ok)
        return r;

    const auto entry = ctx.timelines->put(std::move(doc));

    json::object out;
    out["name"]     = entry->document.name;
    out["revision"] = entry->document.revision;
    out["resolved"] = encode_resolution(entry->resolved);

    if (!entry->resolved.ok())
        return api_reply::fail(api_code::timeline_invalid,
                               "the document was stored and does not resolve: " +
                                   entry->resolved.errors.front().reason,
                               encode_faults(entry->resolved));

    return api_reply::ok_with(std::move(out));
}

api_reply get_timeline(const api_context& ctx, const std::string& name)
{
    if (!ctx.timelines)
        return api_reply::fail(api_code::internal, "no timeline store is wired into this build");

    if (name.empty()) {
        json::array a;
        for (const auto& n : ctx.timelines->names()) {
            const auto e = ctx.timelines->get(n);
            if (!e)
                continue;
            json::object o;
            o["name"]      = n;
            o["channel"]   = e->document.channel;
            o["revision"]  = e->document.revision;
            o["objects"]   = static_cast<std::int64_t>(e->document.objects.size());
            o["instances"] = static_cast<std::int64_t>(e->resolved.instances.size());
            o["ok"]        = e->resolved.ok();
            a.push_back(o);
        }
        json::object out;
        out["timelines"]      = a;
        out["store_revision"] = ctx.timelines->revision();
        return api_reply::ok_with(std::move(out));
    }

    const auto e = ctx.timelines->get(name);
    if (!e)
        return api_reply::fail(api_code::timeline_not_found, "no timeline named '" + name + "'");

    json::object out;
    out["document"] = encode_document(e->document);
    out["resolved"] = encode_resolution(e->resolved);
    return api_reply::ok_with(std::move(out));
}

api_reply delete_timeline(const api_context& ctx, const std::string& name)
{
    if (!ctx.timelines)
        return api_reply::fail(api_code::internal, "no timeline store is wired into this build");
    if (name.empty())
        return api_reply::fail(api_code::bad_request, "DELETE /v1/timeline/{name} takes a name");
    if (!ctx.timelines->erase(name))
        return api_reply::fail(api_code::timeline_not_found, "no timeline named '" + name + "'");

    json::object out;
    out["removed"]        = name;
    out["store_revision"] = ctx.timelines->revision();
    return api_reply::ok_with(std::move(out));
}

api_reply get_timeline_resolved(const api_context& ctx, const std::string& name,
                                const std::string& query)
{
    if (!ctx.timelines)
        return api_reply::fail(api_code::internal, "no timeline store is wired into this build");

    const auto e = ctx.timelines->get(name);
    if (!e)
        return api_reply::fail(api_code::timeline_not_found, "no timeline named '" + name + "'");

    auto out = encode_resolution(e->resolved).as_object();

    const auto at_text = query_value(query, "at");
    if (!at_text.empty()) {
        double at = 0;
        try {
            std::size_t used = 0;
            at               = std::stod(at_text, &used);
            if (used != at_text.size())
                throw std::invalid_argument("trailing");
        } catch (...) {
            return api_reply::fail(api_code::bad_request,
                                    "at= takes a position in seconds, got '" + at_text + "'");
        }
        const auto t = tl::from_seconds(at);

        // WHO OWNS EACH LAYER AT `at`, decided HERE by the same `active_on` the tick uses. That
        // is the whole point of the endpoint: the client draws this rather than reimplementing
        // priority and last-started-wins and disagreeing with the server about which cue is on
        // air (`L28`).
        json::object active;
        for (const auto& kv : e->resolved.by_layer) {
            if (const auto* inst = e->resolved.active_on(kv.first, t)) {
                json::object o;
                o["object"] = inst->object_id;
                o["start"]  = encode_time(inst->start);
                o["end"]    = inst->end ? json::value(encode_time(*inst->end)) : json::value(nullptr);
                const auto local = inst->local_at(t);
                o["local"]       = encode_time(local);

                // THE VALUES, which is what makes this endpoint worth having rather than a
                // convenience. Without them a client knows WHICH object owns a layer at a
                // position and has to interpolate the curve itself to draw the parameter -- and
                // a client's own interpolation is a second implementation of the easing, the
                // per-kind angular modulus and the discrete hold, which is where it comes to
                // disagree with the server about what is on air.
                //
                // Computed by the SAME `curve::interpolate` and the same `kind_of` the tick
                // uses, on the same local time. `timeline-resolved` gates the two against each
                // other for a spread of positions.
                if (const auto* obj = find_object(e->document.objects, inst->object_id)) {
                    json::object values;
                    for (const auto& sv : obj->content)
                        values[sv.first] = vector_to_json(sv.second);
                    for (const auto& pv : obj->curves.interpolate(local, tl::kind_of))
                        values[pv.first] = pv.second;
                    if (!values.empty())
                        o["values"] = values;
                }
                active[kv.first] = o;
            }
        }
        out["at"]     = at;
        out["active"] = active;
    }

    return api_reply::ok_with(std::move(out));
}

}}} // namespace caspar::protocol::http
