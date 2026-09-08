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

#include "api_value.h"

#include <core/stage/stage_fields.h>
#include "json_state.h"

#include <common/log.h>
#include <common/tweener.h>
#include <common/utf.h>

#include <boost/variant/static_visitor.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <memory>

namespace caspar { namespace protocol { namespace http {

namespace fields = core::fields;

namespace {

std::vector<std::string> split_path(std::string_view p)
{
    std::vector<std::string> out;
    size_t                   i = 0;
    while (i <= p.size()) {
        const auto j = p.find('/', i);
        const auto n = (j == std::string_view::npos ? p.size() : j) - i;
        if (n > 0)
            out.emplace_back(p.substr(i, n));
        if (j == std::string_view::npos)
            break;
        i = j + 1;
    }
    return out;
}

bool is_number(const std::string& s) { return !s.empty() && s.find_first_not_of("0123456789") == std::string::npos; }

/// A published datum as a double, where that is meaningful. `false` for a string.
struct as_double_visitor : boost::static_visitor<bool>
{
    double& out;
    explicit as_double_visitor(double& o)
        : out(o)
    {
    }
    bool operator()(bool v) const
    {
        out = v ? 1.0 : 0.0;
        return true;
    }
    bool operator()(std::int32_t v) const { return (out = static_cast<double>(v)), true; }
    bool operator()(std::int64_t v) const { return (out = static_cast<double>(v)), true; }
    bool operator()(std::uint32_t v) const { return (out = static_cast<double>(v)), true; }
    bool operator()(std::uint64_t v) const { return (out = static_cast<double>(v)), true; }
    bool operator()(float v) const { return (out = static_cast<double>(v)), true; }
    bool operator()(double v) const { return (out = v), true; }
    bool operator()(const std::string&) const { return false; }
    bool operator()(const std::wstring&) const { return false; }
};

bool as_double(const core::monitor::data_t& d, double& out)
{
    return boost::apply_visitor(as_double_visitor(out), d);
}

struct as_bool_visitor : boost::static_visitor<bool>
{
    bool& out;
    explicit as_bool_visitor(bool& o)
        : out(o)
    {
    }
    bool operator()(bool v) const { return (out = v), true; }
    bool operator()(std::int32_t v) const { return (out = v != 0), true; }
    bool operator()(std::int64_t v) const { return (out = v != 0), true; }
    bool operator()(std::uint32_t v) const { return (out = v != 0), true; }
    bool operator()(std::uint64_t v) const { return (out = v != 0), true; }
    bool operator()(float) const { return false; }
    bool operator()(double) const { return false; }
    bool operator()(const std::string&) const { return false; }
    bool operator()(const std::wstring&) const { return false; }
};

bool as_bool(const core::monitor::data_t& d, bool& out) { return boost::apply_visitor(as_bool_visitor(out), d); }

/// The JSON elements of a value, whatever shape it arrived in.
///
/// A scalar field accepts a bare value OR a one-element array, and a vector field accepts an
/// array. Accepting both for the scalar case is deliberate: `/v1/value` REPORTS a scalar
/// bare and the OSCQuery tree reports it as a one-element array, so a client that writes
/// back what it read from either place is right, and neither has to know which it used.
std::vector<const json::value*> elements(const json::value& v, std::vector<const json::value*>& storage)
{
    storage.clear();
    if (v.is_array()) {
        for (const auto& e : v.as_array())
            storage.push_back(&e);
    } else {
        storage.push_back(&v);
    }
    return storage;
}

json::object range_detail(const fields::field_meta& f, std::size_t component, double got)
{
    json::object d;
    d["component"] = static_cast<std::int64_t>(component);
    if (f.range) {
        d["min"] = f.range->lo;
        d["max"] = f.range->hi;
    }
    d["got"]  = got;
    d["path"] = f.path;
    return d;
}

} // namespace

bool is_stage_path(const std::string& path)
{
    const auto seg = split_path(path);
    return seg.size() >= 5 && seg[0] == "channel" && seg[2] == "mixer" && seg[3] == "previz";
}

api_reply resolve_stage_write_target(const std::string& path, stage_write_target& out)
{
    const auto seg = split_path(path);
    if (!is_stage_path(path))
        return api_reply::fail(api_code::unknown_path, "not a stage path: " + path);

    if (!is_number(seg[1]))
        return api_reply::fail(api_code::channel_not_found, "channel index is not a number: " + seg[1]);
    out.channel = std::atoi(seg[1].c_str());

    // channel N mixer previz camera FIELD          (6)
    // channel N mixer previz view_camera FIELD     (6)
    // channel N mixer previz screen NAME FIELD     (7)
    if (seg.size() == 6 && (seg[4] == "camera" || seg[4] == "view_camera")) {
        out.object = seg[4];
        out.field  = core::fields::find_camera_field(seg[5]);
        if (!out.field)
            return api_reply::fail(api_code::unknown_path, "no such camera field: " + seg[5]);
    } else if (seg.size() == 7 && seg[4] == "screen") {
        // `unknown_path` rather than a "screen not found" of its own: whether the screen exists
        // is decided by the renderer, on the channel, and this layer cannot see it. The write
        // below reports it, and reporting it twice with two different codes would be worse.
        out.object = "screen/" + seg[5];
        out.field  = core::fields::find_screen_field(seg[6]);
        if (!out.field)
            return api_reply::fail(api_code::unknown_path, "no such screen field: " + seg[6]);
    } else {
        return api_reply::fail(api_code::unknown_path,
                               "a writable stage path is /channel/{n}/mixer/previz/camera/{field}, "
                               ".../view_camera/{field} or .../screen/{name}/{field}, not " +
                                   path);
    }

    if ((static_cast<uint8_t>(out.field->access) & static_cast<uint8_t>(core::fields::access_t::write)) == 0)
        return api_reply::fail(api_code::not_writable,
                               std::string("field is derived and cannot be set: ") + out.field->path);
    return api_reply{};
}

api_reply resolve_write_target(const std::string& path, write_target& out)
{
    const auto seg = split_path(path);
    // channel N stage layer M mixer field
    if (seg.size() != 7 || seg[0] != "channel" || seg[2] != "stage" || seg[3] != "layer" || seg[5] != "mixer")
        return api_reply::fail(api_code::unknown_path,
                               "a writable path is /channel/{n}/stage/layer/{m}/mixer/{field}, not " + path);

    if (!is_number(seg[1]))
        return api_reply::fail(api_code::channel_not_found, "channel index is not a number: " + seg[1]);
    if (!is_number(seg[4]))
        return api_reply::fail(api_code::layer_not_found, "layer index is not a number: " + seg[4]);

    out.channel = std::atoi(seg[1].c_str());
    out.layer   = std::atoi(seg[4].c_str());
    out.field   = fields::find(seg[6]);
    if (!out.field)
        return api_reply::fail(api_code::unknown_path, "no such mixer field: " + seg[6]);

    if ((static_cast<uint8_t>(out.field->access) & static_cast<uint8_t>(fields::access_t::write)) == 0)
        return api_reply::fail(api_code::not_writable,
                               std::string("field is read-only in this build: ") + out.field->path);
    return api_reply{};
}

api_reply json_to_value(const fields::field_meta& f, const json::value& v, core::monitor::vector_t& out)
{
    using fields::value_type;

    std::vector<const json::value*> storage;
    const auto                      els = elements(v, storage);

    if (els.size() != f.arity)
        return api_reply::fail(api_code::field_wrong_type,
                               std::string("field ") + f.path + " takes " + std::to_string(f.arity) +
                                   " component(s), got " + std::to_string(els.size()));

    out.clear();
    for (const auto* e : els) {
        switch (f.type) {
            case value_type::boolean:
                if (!e->is_bool())
                    return api_reply::fail(api_code::field_wrong_type,
                                           std::string("field ") + f.path + " takes a boolean");
                out.push_back(e->as_bool());
                break;

            case value_type::enumeration:
                // A name or its ordinal. A client that already holds the number should not
                // have to look the word up, and one showing a menu should not have to hold
                // the numbers.
                if (e->is_string())
                    out.push_back(std::string(e->as_string().c_str()));
                else if (e->is_int64() || e->is_uint64() || e->is_double())
                    out.push_back(e->to_number<double>());
                else
                    return api_reply::fail(api_code::field_wrong_type,
                                           std::string("field ") + f.path + " takes a name or an index");
                break;

            case value_type::integer:
            case value_type::real:
            case value_type::vec2:
            case value_type::vec3:
            case value_type::vec4:
                if (!e->is_int64() && !e->is_uint64() && !e->is_double())
                    return api_reply::fail(api_code::field_wrong_type,
                                           std::string("field ") + f.path + " takes a number");
                out.push_back(e->to_number<double>());
                break;

            default:
                return api_reply::fail(api_code::not_writable,
                                       std::string("field ") + f.path + " cannot be set by value");
        }
    }
    return api_reply{};
}

api_reply check_and_bound(const fields::field_meta& f, core::monitor::vector_t& v)
{
    if (!f.range)
        return api_reply{};

    json::array details;
    for (std::size_t i = 0; i < v.size(); ++i) {
        double d = 0.0;
        if (!as_double(v[i], d))
            continue; // an enumeration by name: its own set() validates it

        if (f.bounding == fields::bounding_t::wrap) {
            // For a periodic quantity, 400 degrees is a legal way to say 40 rather than an
            // error, so wrap BEFORE the range test rather than reporting a violation the
            // field does not actually have. Live for `hue_shift`, `blur_angle`,
            // `qual_hue_offset` and `shape_gradient_angle`; the projection angles declare
            // `wrap` with NO range, so nothing normalises them and a yaw of 7.5 rad stays
            // 7.5 rad -- which is what AMCP stores too, so the two facades agree.
            const auto span = f.range->hi - f.range->lo;
            if (span > 0.0) {
                d = f.range->lo + std::fmod(std::fmod(d - f.range->lo, span) + span, span);
                v[i] = d;
            }
        }

        if (!f.range->contains(d))
            details.push_back(range_detail(f, i, d));
    }

    if (!details.empty())
        return api_reply::fail(api_code::field_out_of_range,
                               std::string("value out of range for ") + f.path,
                               std::move(details));
    return api_reply{};
}

api_reply prepare_set(const std::string& path, const json::object& op, prepared_set& out)
{
    if (auto r = resolve_write_target(path, out.target); r.code != api_code::ok)
        return r;

    const auto* v = op.if_contains("value");
    if (!v)
        return api_reply::fail(api_code::field_missing, "no value for " + path);
    if (auto r = json_to_value(*out.target.field, *v, out.value); r.code != api_code::ok)
        return r;
    if (auto r = check_and_bound(*out.target.field, out.value); r.code != api_code::ok)
        return r;

    if (auto* d = op.if_contains("duration")) {
        if (!d->is_int64() && !d->is_uint64() && !d->is_double())
            return api_reply::fail(api_code::field_wrong_type, "duration must be a number of frames");
        out.duration = static_cast<unsigned>(std::max(0.0, d->to_number<double>()));
    }
    if (auto* t = op.if_contains("tween")) {
        if (!t->is_string())
            return api_reply::fail(api_code::field_wrong_type, "tween must be a name");
        try {
            out.tween = caspar::tweener(u16(std::string(t->as_string().c_str())));
        } catch (...) {
            return api_reply::fail(api_code::bad_request, std::string("no such tween: ") + t->as_string().c_str());
        }
    }
    return api_reply{};
}

core::stage_base::transform_func_t set_closure(const prepared_set& p)
{
    const auto* f = p.target.field;
    auto        v = p.value;
    return [f, v](core::frame_transform t) {
        f->set(t.image_transform, v);
        core::fields::apply_enables(t.image_transform, *f);
        return t;
    };
}

namespace {

/// PUT one field of one screen or camera.
///
/// `set` ONLY, and that is a scope decision rather than an oversight. `toggle`, `add` and `cas`
/// get their atomicity on the mixer path from running inside one closure on the stage executor,
/// where no other client can interleave between the read and the write. The previz renderer has
/// no equivalent: its API is a set of per-property setters, so a read-modify-write here would
/// span two calls and two acquisitions of the scene lock. Offering a `toggle` that is not atomic
/// would be worse than not offering one, so the other three ops are refused by name and the
/// reason is in the message.
api_reply write_stage_value(const api_context& ctx,
                            const std::string& path,
                            const std::string& body,
                            const std::string& peer)
{
    stage_write_target target;
    if (auto r = resolve_stage_write_target(path, target); r.code != api_code::ok)
        return r;

    json::value doc;
    try {
        doc = body.empty() ? json::value(json::object()) : json::parse(body);
    } catch (...) {
        return api_reply::fail(api_code::bad_request, "body is not valid JSON");
    }
    if (!doc.is_object())
        return api_reply::fail(api_code::bad_request, "body must be a JSON object");
    const auto& o = doc.as_object();

    const std::string op =
        o.if_contains("op") && o.at("op").is_string() ? std::string(o.at("op").as_string().c_str()) : "set";
    if (op != "set")
        return api_reply::fail(api_code::bad_request,
                               "stage fields take op 'set' only; " + op +
                                   " needs a read-modify-write the previz renderer cannot make "
                                   "atomic, and a non-atomic one would be worse than none");

    if (o.if_contains("duration") || o.if_contains("tween"))
        return api_reply::fail(api_code::bad_request,
                               "stage fields are not tweenable: KEYFRAMES is bound to "
                               "image_transform, so a screen cannot be animated in this build");

    const auto& f = *target.field;

    const auto* v = o.if_contains("value");
    if (!v)
        return api_reply::fail(api_code::field_missing, "no value in the body");

    core::monitor::vector_t operand;
    if (auto r = json_to_value(f, *v, operand); r.code != api_code::ok)
        return r;
    if (auto r = check_and_bound(f, operand); r.code != api_code::ok)
        return r;

    if (!ctx.set_stage_field)
        return api_reply::fail(api_code::internal, "the API was built without access to the stage");

    const auto res = ctx.set_stage_field(target.channel, target.object, f.path, operand);
    if (!res.applied)
        return api_reply::fail(api_code::unknown_path,
                               res.reason.empty() ? std::string("the stage write did not apply")
                                                  : res.reason);

    // The bridge compared what it INTENDED against what the renderer now HOLDS, so this catches
    // a mutator that silently declined -- and one does. `set_screen_eye_mode` writes
    // `design_eye_*` only when the mode is already FIXED, so setting a design eye on a
    // camera-mode screen stores nothing. Reporting that as success would be the "202 and no
    // change" failure this whole registry exists to prevent.
    if (res.declined) {
        json::object d;
        d["requested"] = vector_to_json(res.intended);
        d["actual"]    = vector_to_json(res.written);
        d["path"]      = f.path;
        return api_reply::fail(api_code::field_conflict,
                               std::string("the renderer declined the value for ") + f.path +
                                   "; it still holds what it held",
                               json::array{std::move(d)});
    }

    CASPAR_LOG(info) << L"[http] " << u16(peer) << L" set " << u16(path);

    json::object r;
    r["path"]     = path;
    r["previous"] = vector_to_json(res.previous);
    r["value"]    = vector_to_json(res.written);
    return api_reply::ok_with(std::move(r));
}

} // namespace

/// Is this the path of a producer parameter?
///
/// `channel/{n}/stage/layer/{m}/foreground/params/{name}` -- which is where the producer's own
/// `state()` publishes it, so reads need nothing new and this exists only for the write.
///
/// `foreground/params/` rather than a `producer/` node of its own, and that is worth a line: a
/// layer's producer state is already nested under `foreground` by `layer::impl`, so publishing
/// the values there costs no new machinery and puts the read and the write at the SAME path. A
/// separate `producer/` node would have needed its own publication and would have left reads
/// and writes at two different addresses for one value.
bool is_param_path(const std::string& path)
{
    const auto seg = split_path(path);
    return seg.size() == 8 && seg[0] == "channel" && seg[2] == "stage" && seg[3] == "layer" &&
           seg[5] == "foreground" && seg[6] == "params";
}

/// A producer-parameter write. Validated against the producer's OWN descriptor, fetched first.
///
/// Two executor round trips per write -- describe, then set -- and that is the right trade for a
/// PUT. The alternative is to trust the body's type and arity and let the producer's setter
/// refuse, which loses the reason: `set` returns a bool, and "false" cannot say whether the name
/// was wrong, the arity was wrong or the value was out of range. A control surface needs the
/// distinction; a frame path would not, and this is not one.
api_reply write_param_value(const api_context& ctx,
                            const std::string& path,
                            const std::string& body,
                            const std::string& peer)
{
    const auto seg     = split_path(path);
    const int  channel = is_number(seg[1]) ? std::atoi(seg[1].c_str()) : 0;
    const int  layer   = is_number(seg[4]) ? std::atoi(seg[4].c_str()) : -1;
    const auto name    = seg[7];

    if (channel <= 0)
        return api_reply::fail(api_code::channel_not_found, "channel index is not a number: " + seg[1]);
    if (layer < 0)
        return api_reply::fail(api_code::unknown_path, "layer index is not a number: " + seg[4]);

    json::value doc;
    try {
        doc = body.empty() ? json::value(json::object()) : json::parse(body);
    } catch (...) {
        return api_reply::fail(api_code::bad_request, "body is not valid JSON");
    }
    if (!doc.is_object())
        return api_reply::fail(api_code::bad_request, "body must be a JSON object");
    const auto& o = doc.as_object();

    if (o.if_contains("duration") || o.if_contains("tween"))
        return api_reply::fail(api_code::bad_request,
                               "producer parameters are not tweenable: KEYFRAMES is bound to "
                               "image_transform end to end, so a producer parameter cannot be "
                               "animated by it. The OFX module has its own keyframe engine, "
                               "reachable through CALL ... OFX KEY");

    const std::string op =
        o.if_contains("op") && o.at("op").is_string() ? std::string(o.at("op").as_string().c_str()) : "set";
    if (op != "set")
        return api_reply::fail(api_code::bad_request,
                               "producer parameters take op: set only. `toggle`, `add` and `cas` get "
                               "their atomicity on the mixer path from running inside one closure on "
                               "the stage executor; a producer parameter's read and write are two "
                               "separate executor calls, so a read-modify-write here would not be "
                               "atomic -- and a non-atomic toggle is worse than none");

    const auto* v = o.if_contains("value");
    if (!v)
        return api_reply::fail(api_code::field_missing, "no value in the body");

    if (!ctx.stage)
        return api_reply::fail(api_code::internal, "the API was built without access to the channels");
    const auto stage = ctx.stage(channel);
    if (!stage)
        return api_reply::fail(api_code::channel_not_found, "no channel " + std::to_string(channel));

    std::vector<core::param_snapshot> params;
    try {
        params = stage->describe_params(layer).get();
    } catch (...) {
        return api_reply::fail(api_code::internal, "the parameter query threw");
    }

    const core::param_snapshot* p = nullptr;
    for (const auto& q : params) {
        if (q.name == name) {
            p = &q;
            break;
        }
    }
    if (!p) {
        std::string known;
        for (const auto& q : params)
            known += (known.empty() ? "" : ", ") + q.name;
        return api_reply::fail(api_code::unknown_path,
                               "no such producer parameter: " + name +
                                   (known.empty() ? " (this layer's producer has none)"
                                                  : " (this layer has: " + known + ")"));
    }

    if ((static_cast<uint8_t>(p->access) & static_cast<uint8_t>(core::fields::access_t::write)) == 0)
        return api_reply::fail(api_code::not_writable, "parameter is read-only: " + name);

    // The producer's descriptor stands in for a `field_meta`. It carries the same four things
    // the validation needs -- type, arity, range and bounding -- so the JSON conversion and the
    // range check are the same two steps a mixer field goes through.
    core::fields::field_meta meta{};
    meta.path     = p->name.c_str();
    meta.type     = p->type;
    meta.access   = p->access;
    meta.bounding = p->bounding;
    meta.arity    = p->arity;
    meta.values   = p->values.empty() ? nullptr : p->values.c_str();
    if (p->min && p->max)
        meta.range = core::grade_range{*p->min, *p->max};

    core::monitor::vector_t operand;
    if (auto r = json_to_value(meta, *v, operand); r.code != api_code::ok)
        return r;
    if (auto r = check_and_bound(meta, operand); r.code != api_code::ok)
        return r;

    bool applied = false;
    try {
        applied = stage->set_param(layer, name, operand).get();
    } catch (...) {
        return api_reply::fail(api_code::internal, "the parameter write threw");
    }

    if (!applied)
        return api_reply::fail(api_code::unknown_path,
                               "the producer declined the write to " + name +
                                   ". The descriptor came from the producer itself, so this is the "
                                   "producer refusing a value it described as legal rather than a "
                                   "path or type error");

    CASPAR_LOG(info) << L"[api] " << u16(peer) << L" PUT " << u16(path);

    json::object r;
    r["path"]  = path;
    r["param"] = name;
    return api_reply::ok_with(std::move(r));
}

api_reply write_value(const api_context& ctx,
                      const state_hub&,
                      const std::string& path,
                      const std::string& body,
                      const std::string& peer)
{
    if (is_stage_path(path))
        return write_stage_value(ctx, path, body, peer);
    if (is_param_path(path))
        return write_param_value(ctx, path, body, peer);

    write_target target;
    if (auto r = resolve_write_target(path, target); r.code != api_code::ok)
        return r;

    json::value doc;
    try {
        doc = body.empty() ? json::value(json::object()) : json::parse(body);
    } catch (...) {
        return api_reply::fail(api_code::bad_request, "body is not valid JSON");
    }
    if (!doc.is_object())
        return api_reply::fail(api_code::bad_request, "body must be a JSON object");
    const auto& o = doc.as_object();

    const std::string op =
        o.if_contains("op") && o.at("op").is_string() ? std::string(o.at("op").as_string().c_str()) : "set";

    unsigned duration = 0;
    if (auto* d = o.if_contains("duration")) {
        if (!d->is_int64() && !d->is_uint64() && !d->is_double())
            return api_reply::fail(api_code::field_wrong_type, "duration must be a number of frames");
        duration = static_cast<unsigned>(std::max(0.0, d->to_number<double>()));
    }

    caspar::tweener tween;
    if (auto* t = o.if_contains("tween")) {
        if (!t->is_string())
            return api_reply::fail(api_code::field_wrong_type, "tween must be a name");
        try {
            tween = caspar::tweener(u16(std::string(t->as_string().c_str())));
        } catch (...) {
            return api_reply::fail(api_code::bad_request,
                                   std::string("no such tween: ") + t->as_string().c_str());
        }
    }

    const std::string label =
        o.if_contains("label") && o.at("label").is_string() ? std::string(o.at("label").as_string().c_str()) : "";

    const auto& f = *target.field;

    // The operand, for the forms that take one. `toggle` takes none.
    core::monitor::vector_t operand;
    core::monitor::vector_t expect;
    if (op == "set" || op == "add" || op == "cas") {
        const auto* v = o.if_contains("value");
        if (!v)
            return api_reply::fail(api_code::field_missing, "no value in the body");
        if (auto r = json_to_value(f, *v, operand); r.code != api_code::ok)
            return r;
        if (op == "set")
            if (auto r = check_and_bound(f, operand); r.code != api_code::ok)
                return r;
    }
    if (op == "cas") {
        const auto* e = o.if_contains("expect");
        if (!e)
            return api_reply::fail(api_code::field_missing, "cas needs an expect");
        if (auto r = json_to_value(f, *e, expect); r.code != api_code::ok)
            return r;
    }
    if (op != "set" && op != "add" && op != "cas" && op != "toggle")
        return api_reply::fail(api_code::bad_request, "unknown op: " + op);

    if (!ctx.stage)
        return api_reply::fail(api_code::internal, "the API was built without access to the channels");
    const auto stage = ctx.stage(target.channel);
    if (!stage)
        return api_reply::fail(api_code::channel_not_found, "no channel " + std::to_string(target.channel));

    // Everything the closure has to report back. It runs on the STAGE executor, so it
    // cannot return a status -- it fills this in and the caller reads it after the future
    // settles.
    struct outcome
    {
        bool                    applied  = false;
        bool                    conflict = false;
        bool                    type_err = false;
        core::monitor::vector_t previous;
        core::monitor::vector_t written;
        json::array             range_details;
    };
    auto out = std::make_shared<outcome>();

    const auto* fp = &f;
    stage
        ->apply_transform(
            target.layer,
            [fp, op, operand, expect, out](core::frame_transform t) {
                auto& it = t.image_transform;

                // Read INSIDE the closure, on the stage executor, against the same
                // transform the write lands on. That is what makes toggle and cas atomic:
                // no other client's write can interleave between the read and the write,
                // because there is only one executor and this is one task on it.
                out->previous = fp->get(it);

                core::monitor::vector_t next = out->previous;

                if (op == "set") {
                    next = operand;
                } else if (op == "toggle") {
                    bool b = false;
                    if (!out->previous.empty() && as_bool(out->previous[0], b))
                        next = core::monitor::vector_t{!b};
                    else {
                        out->type_err = true;
                        return t;
                    }
                } else if (op == "add") {
                    if (next.size() != operand.size()) {
                        out->type_err = true;
                        return t;
                    }
                    for (std::size_t i = 0; i < next.size(); ++i) {
                        double a = 0.0, b = 0.0;
                        if (!as_double(next[i], a) || !as_double(operand[i], b)) {
                            out->type_err = true;
                            return t;
                        }
                        next[i] = a + b;
                    }
                } else if (op == "cas") {
                    if (!(out->previous == expect)) {
                        out->conflict = true;
                        return t;
                    }
                    next = operand;
                }

                if (op != "set") {
                    // Range-check the COMPUTED value. `add` and `toggle` are the forms
                    // where a client cannot have checked it itself -- it does not know
                    // what the value was.
                    auto v = next;
                    if (auto r = check_and_bound(*fp, v); r.code != api_code::ok) {
                        out->range_details = r.details;
                        return t;
                    }
                    next = std::move(v);
                }

                if (!fp->set(it, next)) {
                    out->type_err = true;
                    return t;
                }
                // Read back what the field now HOLDS rather than echoing what arrived.
                // They differ wherever the descriptor canonicalises: an enumeration set by
                // ordinal reports its name, which is what a client should store and show.
                next = fp->get(it);
                // Whatever subsystem this field belongs to, switched on the way a
                // KEYFRAMES write switches it on -- so the two agree and a `blur_radius`
                // set over either route actually blurs.
                core::fields::apply_enables(it, *fp);

                out->written = std::move(next);
                out->applied = true;
                return t;
            },
            duration,
            tween)
        .get();

    if (out->conflict) {
        json::object d;
        d["expected"] = vector_to_json(expect);
        d["actual"]   = vector_to_json(out->previous);
        return api_reply::fail(api_code::field_conflict,
                               std::string("compare-and-set on ") + f.path + " did not match",
                               json::array{std::move(d)});
    }
    if (!out->range_details.empty())
        return api_reply::fail(api_code::field_out_of_range,
                               std::string("computed value out of range for ") + f.path,
                               out->range_details);
    if (out->type_err)
        return api_reply::fail(api_code::field_wrong_type,
                               std::string("op ") + op + " does not apply to " + f.path);
    if (!out->applied)
        return api_reply::fail(api_code::internal, "the write did not run");

    // One line per write, with the client and the client's own label. A show that goes
    // wrong is reconstructed from this, and a label the operator chose is worth more than
    // any identifier the server could invent.
    CASPAR_LOG(info) << L"[api] " << u16(peer) << L" PUT " << u16(path) << L" op=" << u16(op)
                     << (label.empty() ? L"" : (L" label=\"" + u16(label) + L"\""));

    json::object r;
    r["path"]     = path;
    r["value"]    = vector_to_json(out->written);
    r["previous"] = vector_to_json(out->previous);
    if (duration > 0)
        r["tweening"] = static_cast<std::int64_t>(duration);
    return api_reply::ok_with(std::move(r));
}

}}} // namespace caspar::protocol::http
