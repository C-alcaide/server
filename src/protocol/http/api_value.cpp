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
    // The AUDIO half lives under the same prefix, because a layer's volume is a mixer property
    // to everyone except this codebase's struct layout. Tried second so the image table stays
    // the fast path and so a name can never resolve to both.
    if (!out.field)
        out.audio = fields::find_audio_field(seg[6]);
    if (!out.field && !out.audio)
        return api_reply::fail(api_code::unknown_path, "no such mixer field: " + seg[6]);

    const auto* m = out.meta();
    if ((static_cast<uint8_t>(m->access) & static_cast<uint8_t>(fields::access_t::write)) == 0)
        return api_reply::fail(api_code::not_writable,
                               std::string("field is read-only in this build: ") + m->path);
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

        // NOT IMPLEMENTED HERE, DELIBERATELY, and the reason is a facade split rather than an
        // oversight -- see `docs/features/control-api.md` §bounding. 74 of the 115 ranged fields
        // declare `bounding = clip`, and clamping them HERE alone would make the two facades
        // disagree on every one of them: AMCP's shared `grade_param` REFUSES at 81 call sites
        // (with a message naming the range), and only `MIXER CDL`'s four limits clamp, at 10.
        // So the descriptor's `clip` is a promise neither facade keeps, and making this one keep
        // it is a divergence rather than a fix. Tracked, not silently left.

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
    if (auto r = json_to_value(*out.target.meta(), *v, out.value); r.code != api_code::ok)
        return r;
    if (auto r = check_and_bound(*out.target.meta(), out.value); r.code != api_code::ok)
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
        } catch (const std::exception&) {
            return api_reply::fail(api_code::bad_request, std::string("no such tween: ") + t->as_string().c_str());
        }
    }
    return api_reply{};
}

core::stage_base::transform_func_t set_closure(const prepared_set& p)
{
    const auto* f = p.target.field;
    const auto* a = p.target.audio;
    auto        v = p.value;
    return [f, a, v](core::frame_transform t) {
        // One of the two, never both -- `resolve_write_target` tries the image table first and
        // only reaches the audio one when that misses, so a name cannot resolve to both.
        if (f) {
            f->set(t.image_transform, v);
            core::fields::apply_enables(t.image_transform, *f);
        } else if (a) {
            // No `apply_enables`: there is no audio subsystem gate to switch on, and the audio
            // rows declare `enables == nullptr` accordingly.
            a->set(t.audio_transform, v);
        }
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
    } catch (const std::exception&) {
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
                               "stage fields are not tweenable through this endpoint: the previz "
                               "renderer's mutators are called from the http executor and cannot be "
                               "driven per tick from here. A timeline document animates them -- "
                               "PUT /v1/timeline/{name} with a `previz/screen/...` path");

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
/// `channel/N/stage/layer/M/mixer/node/<id>/<param>` -- nine segments.
///
/// Tested BEFORE the seven-segment `resolve_write_target`, which would otherwise try to read
/// `node` as a field name and answer `unknown_path` for a parameter that exists.
bool is_node_path(const std::string& path)
{
    const auto seg = split_path(path);
    return seg.size() == 9 && seg[0] == "channel" && seg[2] == "stage" && seg[3] == "layer" &&
           seg[5] == "mixer" && seg[6] == "node";
}

/// A NODE-PARAMETER write, validated against the ATTACHED DOCUMENT's descriptor.
///
/// `write_param_value`'s shape, and the resemblance is the point: a node parameter and a producer
/// parameter are both LIVE registries, so both fetch the descriptor from the stage first and then
/// validate the body against it. Two executor round trips per write -- describe, then set -- which
/// is the right trade for a PUT: trusting the body and letting the setter refuse loses the reason,
/// and a control surface needs to know whether the name, the arity or the range was wrong.
///
/// WHAT IS DIFFERENT FROM A PRODUCER PARAMETER, and it is the whole reason the node graph was
/// designed this way: a node parameter has a CONSTANT. The attached document holds it, so a write
/// during a ramp is REMEMBERED rather than refused -- `effective: false` with `shadowed_by`, and
/// the value lands when the driver ends. `faults.yaml` calls the producer-parameter refusal "a gap
/// in the ownership stack rather than a policy"; this is that gap closed for node parameters.
api_reply write_node_value(const api_context& ctx,
                           const std::string& path,
                           const std::string& body,
                           const std::string& peer)
{
    const auto seg = split_path(path);
    if (!is_number(seg[1]) || !is_number(seg[4]))
        return api_reply::fail(api_code::bad_request, "channel and layer must be numbers: " + path);
    const auto channel = std::atoi(seg[1].c_str());
    const auto layer   = std::atoi(seg[4].c_str());
    const auto node_path = "node/" + seg[7] + "/" + seg[8];

    if (!ctx.stage)
        return api_reply::fail(api_code::internal, "no stage bridge is wired into this build");
    auto stage = ctx.stage(channel);
    if (!stage)
        return api_reply::fail(api_code::channel_not_found,
                               "no channel " + std::to_string(channel));

    const auto gname = stage->graph_of(layer);
    if (gname.empty())
        return api_reply::fail(api_code::unknown_path,
                               "layer " + std::to_string(layer) +
                                   " has no node graph attached. `GRAPH " +
                                   std::to_string(channel) + "-" + std::to_string(layer) +
                                   " ATTACH <name>` puts one on it");

    json::value doc;
    try {
        doc = body.empty() ? json::value(json::object()) : json::parse(body);
    } catch (const std::exception&) {
        return api_reply::fail(api_code::bad_request, "body is not valid JSON");
    }
    if (!doc.is_object())
        return api_reply::fail(api_code::bad_request, "body must be a JSON object");
    const auto& op = doc.as_object();

    // `duration`/`tween` REFUSED, with the producer-parameter message and for a related reason:
    // `MIXER <duration>` tweens a `frame_transform` and a node parameter is not in one. The
    // answer is a timeline, which is a better one -- it can ease, it can be scheduled, and it
    // publishes what it is doing.
    if (op.if_contains("duration") || op.if_contains("tween"))
        return api_reply::fail(api_code::bad_request,
                               "a node parameter cannot be tweened by a write. `MIXER <duration>` "
                               "interpolates the frame transform and a node parameter is not on "
                               "it -- animate it with a timeline, which can also ease it, "
                               "schedule it and publish what it is doing");

    std::vector<core::param_snapshot> params;
    try {
        params = stage->describe_graph(layer).get();
    } catch (const std::exception&) {
        return api_reply::fail(api_code::internal, "describing the graph threw");
    }
    const auto p = std::find_if(params.begin(), params.end(),
                                [&](const core::param_snapshot& x) { return x.name == node_path; });
    if (p == params.end())
        return api_reply::fail(api_code::unknown_path,
                               "graph '" + gname + "' has no parameter '" + node_path +
                                   "'. Its ports are in the tree under "
                                   "channel/N/stage/layer/M/mixer/node/");

    // The descriptor stands in for a `field_meta`, exactly as a producer parameter's does: the
    // same four things the validation needs, so the JSON conversion and the range check are the
    // same two steps a mixer field goes through.
    core::fields::field_meta meta{};
    meta.path     = p->name.c_str();
    meta.type     = p->type;
    meta.access   = p->access;
    meta.bounding = p->bounding;
    meta.arity    = p->arity;
    meta.values   = p->values.empty() ? nullptr : p->values.c_str();
    if (p->min && p->max)
        meta.range = core::grade_range{*p->min, *p->max};

    // A HOLD OR RELEASE WITH NO VALUE, handled BEFORE the value is required -- exactly as the
    // mixer-field path does, and for the same reason: `PUT {"hold": true}` on a ramping
    // parameter means "stop there", and requiring a value would make the client read the
    // position first and race the next tick. `{"hold": false}` is the release and has no value
    // to send at all.
    //
    // FOUND BY `graph-stack` ON ITS FIRST RUN, and it is worth recording how: the release check
    // came back `field_missing: no value`, and the four checks after it failed as CASCADES --
    // the hold never let go, so the binding, the document and a later write all read the held
    // number. One defect, five red checks, and reading them in order was the only way to see
    // that. A battery that had stopped at the first failure would have reported four defects.
    const auto* v = op.if_contains("value");
    if (const auto* h = op.if_contains("hold"); h && !v) {
        if (!h->is_bool())
            return api_reply::fail(api_code::bad_request, "\"hold\" takes a boolean");
        const bool want   = h->as_bool();
        const auto before = stage->describe_graph(layer).get();
        const auto bp     = std::find_if(before.begin(), before.end(),
                                     [&](const core::param_snapshot& x) { return x.name == node_path; });

        bool ok = false;
        try {
            ok = want ? stage->hold_field(layer, node_path).get()
                      : stage->release_field(layer, node_path).get();
        } catch (const std::exception&) {
            ok = false;
        }
        if (want && !ok)
            return api_reply::fail(api_code::not_writable,
                                   node_path + " cannot be held");

        CASPAR_LOG(info) << L"[api] " << u16(peer) << (want ? L" HOLD " : L" RELEASE ")
                         << u16(path);

        json::object r;
        r["path"]  = path;
        r["graph"] = gname;
        r["param"] = node_path;
        r["hold"]  = want;
        if (bp != before.end())
            r["value"] = vector_to_json(bp->value);
        // A RELEASE THAT REMOVED NOTHING IS `ok` with `hold: false`, not an error: "make sure
        // this is not held" is idempotent, and a client tidying up should not have to know
        // whether it had held anything.
        r["changed"] = ok;
        if (!want) {
            // WHO HAS IT NOW, which is the half a client needs to know whether releasing gave
            // the parameter to the operator or to a document underneath.
            const auto after = stage->driver_of(layer, node_path);
            if (!after.first.empty())
                r["shadowed_by"] = after.first;
        }
        return api_reply::ok_with(std::move(r));
    }

    if (!v)
        return api_reply::fail(api_code::field_missing, "no value for " + path);

    core::monitor::vector_t operand;
    if (auto r = json_to_value(meta, *v, operand); r.code != api_code::ok)
        return r;
    if (auto r = check_and_bound(meta, operand); r.code != api_code::ok)
        return r;

    // HOLD-AND-WRITE, in that order, so `{"value":..,"hold":true}` is one round trip and the
    // operator owns the parameter from the moment the value lands rather than one frame later.
    bool held = false;
    if (const auto* h = op.if_contains("hold"); h && h->is_bool() && h->as_bool()) {
        try {
            held = stage->hold_field(layer, node_path).get();
        } catch (const std::exception&) {
            held = false;
        }
    }

    std::string label;
    if (const auto* l = op.if_contains("label"); l && l->is_string())
        label = l->as_string().c_str();

    bool applied = false;
    try {
        applied = stage->set_node_param(layer, node_path, operand, label).get();
    } catch (const std::exception&) {
        return api_reply::fail(api_code::internal, "the node parameter write threw");
    }
    if (!applied)
        return api_reply::fail(api_code::unknown_path,
                               "the write to " + node_path + " did not land. The descriptor came "
                               "from the attached document, so this is the store refusing a value "
                               "it described as legal rather than a path or type error");

    CASPAR_LOG(info) << L"[api] " << u16(peer) << L" PUT " << u16(path);

    // WHO HAS IT NOW, in the same shape every mixer field's reply carries. `effective: false`
    // with `shadowed_by` is what makes a write during a ramp REMEMBERED instead of refused: the
    // value is in the document, and it is what the parameter returns to when the driver ends.
    const auto who = stage->driver_of(layer, node_path);
    json::object r;
    r["path"]  = path;
    r["graph"] = gname;
    r["param"] = node_path;
    if (held)
        r["held"] = true;
    if (!who.first.empty()) {
        // `hold` puts THIS write on top, so it is effective even though a driver exists.
        r["effective"] = held || who.first == "hold";
        if (!r["effective"].as_bool())
            r["shadowed_by"] = who.first;
        r["stack"] = who.second;
    } else {
        r["effective"] = true;
    }
    return api_reply::ok_with(std::move(r));
}

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
    } catch (const std::exception&) {
        return api_reply::fail(api_code::bad_request, "body is not valid JSON");
    }
    if (!doc.is_object())
        return api_reply::fail(api_code::bad_request, "body must be a JSON object");
    const auto& o = doc.as_object();

    if (o.if_contains("duration") || o.if_contains("tween"))
        return api_reply::fail(api_code::bad_request,
                               "producer parameters are not tweenable through this endpoint: a "
                               "producer's setter is not a transform and has no tween. A timeline "
                               "document animates them -- PUT /v1/timeline/{name} with a "
                               "`producer/<name>` path");

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
    // ── `std::exception` AND NOT `...`, FOR THE REASON THIS EXACT CALL ALREADY PROVED ──
    //
    // This tree is built with `/EHa`, under which a bare `catch (...)` also catches STRUCTURED
    // exceptions -- an access violation included. `describe_params` is the call that faulted on
    // every OFX plugin with a 2D parameter; `api_tree.cpp`'s copy ate it, answered with an empty
    // parameter node, and the reading was "OFX plugins do not group their parameters" rather
    // than "the server is corrupting memory here". It cost most of a session.
    //
    // THAT FIX WAS APPLIED WHERE THE BUG WAS FOUND AND NOWHERE ELSE. This file had twelve
    // `catch (...)` and no `catch (std::exception)` at all, six of them wrapping stage calls --
    // including this one, the same function. Memory corruption must reach a crash dump; only a
    // C++ exception is the benign failure these handlers are written for.
    try {
        params = stage->describe_params(layer).get();
    } catch (const std::exception&) {
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

    // A DRIVEN PRODUCER PARAMETER IS THE ONE CASE STILL REFUSED, and the reason is different
    // from the one `field_bound` used to give. A producer parameter has no constant on the
    // stage -- the value lives inside the producer, and a binding writes it through the
    // producer's own setter. There is nowhere to remember an operator's write, so it really
    // would be applied and overwritten. That is a gap in the ownership stack rather than a
    // policy: it closes when producer parameters get their own overlay.
    if (stage->is_bound(layer, "producer/" + name))
        return api_reply::fail(api_code::field_bound,
                               "producer/" + name + " is driven by a binding on this layer, and "
                               "a producer parameter has no constant on the stage to remember a "
                               "write in -- so unlike a mixer field this one is refused. `UNBIND " +
                                   std::to_string(channel) + "-" + std::to_string(layer) +
                                   " producer/" + name + "` hands the parameter back");

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
    } catch (const std::exception&) {
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
    // BEFORE `resolve_write_target`, which splits a seven-segment path and would read `node` as
    // a field name -- answering `unknown_path` for a parameter that exists.
    if (is_node_path(path))
        return write_node_value(ctx, path, body, peer);

    write_target target;
    if (auto r = resolve_write_target(path, target); r.code != api_code::ok)
        return r;

    json::value doc;
    try {
        doc = body.empty() ? json::value(json::object()) : json::parse(body);
    } catch (const std::exception&) {
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
        } catch (const std::exception&) {
            return api_reply::fail(api_code::bad_request,
                                   std::string("no such tween: ") + t->as_string().c_str());
        }
    }

    const std::string label =
        o.if_contains("label") && o.at("label").is_string() ? std::string(o.at("label").as_string().c_str()) : "";

    const auto& f = *target.meta();

    const auto stage = ctx.stage(target.channel);
    if (!stage)
        return api_reply::fail(api_code::channel_not_found, "no channel " + std::to_string(target.channel));

    // `{"hold": true}` / `{"hold": false}` -- take the path for the operator, above every other
    // rank, or give it back. PIXERA's Dominant, and the answer to "a document is driving the
    // thing I need to fix on air".
    //
    // On this endpoint rather than its own, because the operator's intent is one action: set it
    // and keep it. A separate `POST .../hold` would make a client send two requests and handle
    // the case where the first landed and the second did not.
    //
    // A HOLD WITH NO VALUE holds what is on air at that moment, which is why it is checked
    // before the value is required: `PUT {"hold": true}` on a ramping parameter means "stop
    // there", and requiring a value would make the client read the position first and race the
    // next tick.
    if (const auto* h = o.if_contains("hold")) {
        if (!h->is_bool())
            return api_reply::fail(api_code::bad_request, "\"hold\" takes a boolean");
        const bool want = h->as_bool();

        core::monitor::vector_t before;
        if (const auto* fi = target.field)
            before = fi->get(stage->get_current_transform(target.layer).get().image_transform);
        else if (const auto* ai = target.audio)
            before = ai->get(stage->get_current_transform(target.layer).get().audio_transform);

        const bool ok = want ? stage->hold_field(target.layer, std::string(f.path)).get()
                             : stage->release_field(target.layer, std::string(f.path)).get();
        if (want && !ok)
            return api_reply::fail(api_code::not_writable,
                                   std::string(f.path) + " cannot be held: it has no setter");

        CASPAR_LOG(info) << L"[api] " << u16(peer) << (want ? L" HOLD " : L" RELEASE ") << u16(path);

        json::object r;
        r["path"]  = path;
        r["hold"]  = want;
        r["value"] = vector_to_json(before);
        // A RELEASE that removed nothing is `ok` with `hold: false`, not an error: "make sure
        // this is not held" is idempotent, and a client tidying up after a session should not
        // have to know whether it had held anything.
        r["changed"] = ok;
        if (!want) {
            const auto after = stage->driver_of(target.layer, std::string(f.path));
            if (!after.first.empty())
                r["shadowed_by"] = after.first;
        }
        return api_reply::ok_with(std::move(r));
    }

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

    // A WRITE TO A DRIVEN FIELD IS REMEMBERED, NOT REFUSED, and that is the whole change
    // `field_bound` used to represent.
    //
    // It used to be refused, with this reasoning: a write would be applied and then overwritten
    // on the next tick, which succeeds and does not last, and that is worse than a refusal. The
    // reasoning was right about the old write path and wrong about what to do. On the old path
    // the write went into the layer's tween, which is also where a binding wrote, so the two
    // genuinely could not coexist -- one had to lose, and losing silently was the bad outcome.
    //
    // With the ownership stack they do not share a place. The operator's value goes into the
    // CONSTANT, which nothing else touches, and a driver's value goes into an overlay above it.
    // So the write lands, is kept, and takes effect the moment the driver ends -- and the reply
    // says so: `effective: false` with `shadowed_by` naming who has it. A control surface can
    // show the slider where the operator put it, the value on air beside it, and who to take it
    // from.
    //
    // `field_bound` therefore has no site left. It stays in `api_code` for one release, marked
    // deprecated in `faults.yaml`, because a client that switches on it should keep compiling.
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
    // WHICH HALF of the frame transform this field lives in, as an accessor pair, so every op
    // below -- set, toggle, add, cas -- has one implementation over both. The alternative was a
    // second copy of the closure for audio, and a second copy is where `toggle` would quietly
    // work on one half and not the other.
    const auto* img = target.field;
    const auto* aud = target.audio;
    auto        rd  = [img, aud](core::frame_transform& t) {
        return img ? img->get(t.image_transform) : aud->get(t.audio_transform);
    };
    auto wr = [img, aud](core::frame_transform& t, const core::monitor::vector_t& v) {
        if (img) {
            if (!img->set(t.image_transform, v))
                return false;
            // Whatever subsystem this field belongs to, switched on the way a KEYFRAMES write
            // switches it on -- so the two agree and a `blur_radius` set over either route
            // actually blurs. The audio rows declare no `enables`, so there is nothing to do
            // on that side.
            core::fields::apply_enables(t.image_transform, *img);
            return true;
        }
        return aud->set(t.audio_transform, v);
    };

    stage
        ->apply_transform(
            target.layer,
            [fp, rd, wr, op, operand, expect, out](core::frame_transform t) {
                // Read INSIDE the closure, on the stage executor, against the same
                // transform the write lands on. That is what makes toggle and cas atomic:
                // no other client's write can interleave between the read and the write,
                // because there is only one executor and this is one task on it.
                out->previous = rd(t);

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

                if (!wr(t, next)) {
                    out->type_err = true;
                    return t;
                }
                // Read back what the field now HOLDS rather than echoing what arrived.
                // They differ wherever the descriptor canonicalises: an enumeration set by
                // ordinal reports its name, which is what a client should store and show.
                next = rd(t);

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

    // WHETHER THE WRITE IS WHAT IS ON AIR. `value` is what the field now holds -- the operator's
    // constant -- and if something above it in the ownership stack is driving the same path,
    // that is NOT what the picture shows. Saying so is the difference between a slider that
    // snaps back for no visible reason and one whose panel reads "held by timeline:show/lt1".
    //
    // The write was still kept: it takes effect the moment the driver ends. `effective` is
    // absent rather than true when nothing shadows it, so a client that does not look for it
    // behaves exactly as before.
    const auto owner_now = stage->driver_of(target.layer, f.path);
    if (!owner_now.first.empty()) {
        r["effective"]   = false;
        r["shadowed_by"] = owner_now.first;
        if (owner_now.second != owner_now.first)
            r["stack"] = owner_now.second;
    }
    return api_reply::ok_with(std::move(r));
}

}}} // namespace caspar::protocol::http
