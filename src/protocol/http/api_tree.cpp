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

#include "api_tree.h"
#include "json_state.h"

#include <core/stage/stage_fields.h>

#include <boost/variant/get.hpp>

#include <common/utf.h>

#include <algorithm>
#include <cstdlib>

#if defined(_WIN32)
#include <process.h>
#define CASPAR_GETPID() _getpid()
#else
#include <unistd.h>
#define CASPAR_GETPID() getpid()
#endif

namespace caspar { namespace protocol { namespace http {

namespace fields = core::fields;

json::value to_json(const core::monitor::vector_t& v) { return vector_to_json(v); }

std::string osc_type_tags(const core::monitor::vector_t& v) { return osc_tags(v); }

namespace {

/// Split a `/`-separated path into segments, dropping empties. `//a` and `/a/` therefore
/// name the same node as `a`, which is what OSCQuery clients in the wild actually send.
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

/// Walk to a child node, creating the `CONTENTS` chain as it goes.
///
/// Every node it creates gets its own `FULL_PATH`, because OSCQuery containers are
/// addressable in their own right: a client asking for `/channel/1/stage/layer/1` must get
/// a node back, not a hole between two leaves.
json::object& descend(json::object& parent, const std::string& segment, std::string& full_path)
{
    full_path += "/";
    full_path += segment;

    auto& contents_val = parent["CONTENTS"];
    if (!contents_val.is_object())
        contents_val = json::object();
    auto& contents = contents_val.as_object();

    auto it = contents.find(segment);
    if (it == contents.end()) {
        json::object node;
        node["FULL_PATH"] = full_path;
        it                = contents.emplace(segment, std::move(node)).first;
    }
    return it->value().as_object();
}

/// The node at `segments`, relative to `root` (whose own path is `base`), creating
/// containers along the way.
json::object& node_at(json::object& root, const std::string& base, const std::vector<std::string>& segments)
{
    json::object* cur  = &root;
    std::string   full = base;
    for (const auto& s : segments)
        cur = &descend(*cur, s, full);
    return *cur;
}

json::object make_root()
{
    json::object root;
    root["FULL_PATH"] = "/";
    root["ACCESS"]    = 0;
    root["CONTENTS"]  = json::object();
    return root;
}

/// A read-only leaf carrying a published value. This is everything OSC has ever exposed:
/// the type comes from the value, because nothing else describes it.
void fill_telemetry_leaf(json::object& node, const core::monitor::vector_t& v)
{
    node["ACCESS"] = 1;
    node["TYPE"]   = osc_tags(v);
    node["VALUE"]  = vector_to_oscquery_value(v);
}

/// OSCQuery's `RANGE` is an array parallel to `VALUE`, one entry per component -- so a
/// vec3 gets three, even when all three share one limit.
json::array range_for(const fields::field_meta& f)
{
    json::array ranges;
    if (!f.range && !(f.type == fields::value_type::enumeration && f.values))
        return ranges;

    ranges.reserve(f.arity);
    for (uint8_t c = 0; c < f.arity; ++c) {
        json::object r;
        if (f.range) {
            r["MIN"] = f.range->lo;
            r["MAX"] = f.range->hi;
        }
        if (f.type == fields::value_type::enumeration && f.values) {
            json::array vals;
            for (auto name : fields::split_list(f.values))
                vals.push_back(json::value(std::string(name)));
            r["VALS"] = std::move(vals);
        }
        ranges.push_back(std::move(r));
    }
    return ranges;
}

/// Everything about a field that OSCQuery has no key for, under one vendor key.
///
/// The alternative -- inventing top-level keys -- makes this server's tree invalid against
/// the spec, and a strict client is entitled to reject the whole node. ossia and Vezér
/// both extend the format exactly this way.
/// The default arrives as a VALUE rather than being read from the descriptor, because
/// `defaults()` is the one accessor that depends on which struct the field belongs to. Passing it
/// in is what keeps this function usable for any table rather than only the transform's.
json::object vendor_block(const fields::field_meta& f, const core::monitor::vector_t& defaults)
{
    json::object c;
    c["type"]     = type_name(f.type);
    c["bounding"] = bounding_name(f.bounding);
    c["compose"]  = compose_name(f.compose);
    c["kind"]     = kind_name(f.kind);
    c["step"]     = f.step;
    c["arity"]    = static_cast<int>(f.arity);
    c["default"]  = vector_to_oscquery_value(defaults);
    c["writable"] = (static_cast<uint8_t>(f.access) & static_cast<uint8_t>(fields::access_t::write)) != 0;

    if (f.unit)
        c["unit"] = f.unit;
    if (f.enables)
        c["enables"] = f.enables;

    if (f.kf_names) {
        json::array kf;
        for (auto name : fields::split_list(f.kf_names))
            kf.push_back(json::value(std::string(name)));
        c["kf"] = std::move(kf);
    } else {
        // Explicitly null rather than absent: "this field cannot be animated" is a fact a
        // control surface needs, and an absent key is indistinguishable from a tree built
        // by a server that did not know about keyframes at all.
        c["kf"] = nullptr;
    }
    return c;
}

/// One descriptor leaf, from a `field_meta` plus its default.
///
/// The default arrives separately for the same reason `vendor_block` takes it separately:
/// `defaults()` is the one accessor that depends on which struct owns the field, and this
/// function is used for three different structs.
json::object descriptor_leaf(const fields::field_meta& f, const core::monitor::vector_t& def)
{
    json::object leaf;
    leaf["FULL_PATH"] = "";
    leaf["ACCESS"]    = static_cast<int>(static_cast<uint8_t>(f.access));
    leaf["TYPE"]      = osc_tags_for(f);
    leaf["VALUE"]     = vector_to_oscquery_value(def);
    // A field with neither limits nor a value list gets no `RANGE` key at all.
    // Emitting `[{}]` would be worse than saying nothing: a client reads the key's
    // presence as "this is bounded" and then finds no MIN or MAX to bound it with.
    if (auto range = range_for(f); !range.empty())
        leaf["RANGE"] = std::move(range);
    // OMITTED for a `refuse` field rather than filled with a value that would mislead --
    // see `clipmode_name`. A standard client reads absence as "no clipping", which is wrong
    // in a harmless direction; `"both"` was wrong in a harmful one.
    if (const char* cm = clipmode_name(f.bounding))
        leaf["CLIPMODE"] = cm;
    if (f.description)
        leaf["DESCRIPTION"] = f.description;
    leaf["casparcg"] = vendor_block(f, def);
    return leaf;
}

/// One producer parameter as an OSCQuery leaf.
///
/// **Built by borrowing `descriptor_leaf`** rather than by assembling the keys again, and that
/// is the point of doing it this way: a producer parameter then carries EXACTLY the key set a
/// mixer field carries -- TYPE, RANGE, CLIPMODE, VALUE, DESCRIPTION, ACCESS and the `casparcg`
/// vendor block -- so a client that can draw a mixer field can draw an ISF input with no new
/// code, and no future key can be added to one and forgotten on the other.
///
/// The `field_meta` here is a temporary VIEW of the snapshot, not a registry row: the `const
/// char*` members point into the snapshot's own strings and are used only inside this call.
/// That is safe because `descriptor_leaf` copies everything it reads into JSON, and it is why
/// this takes the snapshot by const reference and returns before it can outlive it.
json::object param_leaf(const core::param_snapshot& p, const std::string& full_path)
{
    fields::field_meta meta{};
    meta.path     = p.name.c_str();
    meta.type     = p.type;
    meta.access   = p.access;
    meta.bounding = p.bounding;
    meta.arity    = p.arity;
    meta.step     = p.step;
    meta.unit     = p.unit.empty() ? nullptr : p.unit.c_str();
    meta.values   = p.values.empty() ? nullptr : p.values.c_str();
    // The parameter's own description if the format gave it one, else its label -- which every
    // ISF input and every OFX parameter has. Never empty, so a generated control always has
    // something to put next to the slider.
    meta.description = p.description.empty() ? (p.label.empty() ? nullptr : p.label.c_str())
                                             : p.description.c_str();
    // RANGE only when the FORMAT declared one. ISF's MIN/MAX are optional and OFX's are not, so
    // inventing 0..1 for an unbounded parameter would put a wrong slider on a control surface --
    // and `descriptor_leaf` omits the key entirely when there is no range, which says so.
    if (p.min && p.max)
        meta.range = core::grade_range{*p.min, *p.max};
    // kf_names stays null: KEYFRAMES is bound to `image_transform` end to end, so `descriptor_leaf`
    // emits an explicit `"kf": null` and a client is told the parameter cannot be animated rather
    // than left to guess from an absent key.

    auto leaf         = descriptor_leaf(meta, p.default_value);
    leaf["FULL_PATH"] = full_path;
    // GROUP goes in the vendor block rather than at the top level, for the reason
    // `vendor_block` states: inventing top-level keys makes the tree invalid against the
    // OSCQuery spec. Absent -- not empty-string -- when the format declared none, so a client
    // can tell "ungrouped" from "grouped under the empty name".
    if (!p.group.empty()) {
        auto& c = leaf["casparcg"].as_object();
        c["group"] = p.group;
    }
    // ORDER, as a number. `CONTENTS` is a JSON object and a JSON object is an unordered
    // collection by definition, so the declaration order a client sees today is whatever its
    // parser happens to preserve. For ISF that is the only structure the format has.
    if (p.index >= 0) {
        auto& c = leaf["casparcg"].as_object();
        c["index"] = p.index;
    }
    // The DEFAULT is what `descriptor_leaf` put in VALUE; the live value overwrites it, exactly
    // as the mixer pass overwrites a published field's descriptor default below.
    if (!p.value.empty())
        leaf["VALUE"] = vector_to_oscquery_value(p.value);
    return leaf;
}

/// A whole object's descriptor node, from any of the three tables.
template <class T>
json::object object_template(const std::vector<fields::typed_field<T>>& table)
{
    json::object node;
    node["FULL_PATH"] = "";
    node["ACCESS"]    = 0;
    json::object contents;
    for (const auto& f : table)
        contents.emplace(f.path, descriptor_leaf(f, f.defaults()));
    node["CONTENTS"] = std::move(contents);
    return node;
}

const json::object& screen_template()
{
    static const json::object t = object_template(fields::screen_fields());
    return t;
}

const json::object& camera_template()
{
    static const json::object t = object_template(fields::camera_fields());
    return t;
}

/// The per-layer `mixer` sub-tree. Built once and copied per layer: the descriptor set is
/// the bulk of the tree and is identical for every layer on every channel.
const json::object& mixer_template()
{
    static const json::object tmpl = [] {
        json::object node;
        node["FULL_PATH"] = "";
        node["ACCESS"]    = 0;
        json::object contents;

        for (const auto& f : fields::all())
            contents.emplace(f.path, descriptor_leaf(f, f.defaults()));

        node["CONTENTS"] = std::move(contents);
        return node;
    }();
    return tmpl;
}

/// Stamp the copied template with this layer's addresses -- the only per-layer difference.
void instantiate_mixer(json::object& dst, const std::string& base)
{
    dst["FULL_PATH"] = base;
    auto& contents   = dst["CONTENTS"].as_object();
    for (auto& kv : contents)
        kv.value().as_object()["FULL_PATH"] = base + "/" + std::string(kv.key());
}

bool starts_with(const std::string& s, const std::string& p)
{
    return s.size() >= p.size() && s.compare(0, p.size(), p) == 0;
}

/// The layer indices a snapshot mentions, from keys shaped `stage/layer/<n>/...`.
///
/// Every existing layer writes at least `foreground/*` every tick, so scanning is complete
/// -- there is no separate list of layers to consult, and a layer that exists but publishes
/// nothing does not exist as far as any consumer is concerned.
std::vector<int> layers_in(const core::monitor::state& snap)
{
    std::vector<int>        out;
    static const std::string prefix = "stage/layer/";
    for (const auto& kv : snap) {
        const auto& key = kv.first;
        if (!starts_with(key, prefix))
            continue;
        const auto rest = key.find('/', prefix.size());
        if (rest == std::string::npos)
            continue;
        const auto digits = key.substr(prefix.size(), rest - prefix.size());
        if (digits.empty() || digits.find_first_not_of("0123456789") != std::string::npos)
            continue;
        const int n = std::atoi(digits.c_str());
        if (std::find(out.begin(), out.end(), n) == out.end())
            out.push_back(n);
    }
    std::sort(out.begin(), out.end());
    return out;
}

} // namespace

json::object build_tree(const state_hub& hub, const http_config& cfg, const api_context& ctx)
{
    const bool with_mixer = cfg.extent != L"state";

    auto root = make_root();

    for (const int ch : hub.channels()) {
        const auto snap = hub.get(ch);
        if (!snap)
            continue;

        const std::string ch_base = "/channel/" + std::to_string(ch);

        auto& ch_node    = node_at(root, "", {"channel", std::to_string(ch)});
        ch_node["ACCESS"] = 0;

        // Source 2: every key the tick actually published, as read-only telemetry.
        for (const auto& kv : *snap) {
            auto segments = split_path(kv.first);
            if (segments.empty())
                continue;
            auto& leaf = node_at(ch_node, ch_base, segments);
            fill_telemetry_leaf(leaf, kv.second);
        }

        if (!with_mixer)
            continue;

        // Source 3a: the STAGE registry, for the objects this channel's snapshot says exist.
        //
        // Named dynamic children, unlike a layer: a screen is created at runtime and addressed
        // by name, so the set comes out of the always-published `screens` list rather than being
        // scanned for. A screen every one of whose fields sits at its default publishes no field
        // keys at all, which is exactly why that list is published unconditionally.
        {
            std::vector<std::string> screens;
            bool                     has_previz = false;
            for (const auto& kv : *snap) {
                if (!starts_with(kv.first, "mixer/previz/"))
                    continue;
                has_previz = true;
                if (kv.first != "mixer/previz/screens")
                    continue;
                for (const auto& d : kv.second)
                    if (const auto* nm = boost::get<std::string>(&d))
                        screens.push_back(*nm);
            }

            if (has_previz) {
                const std::string previz_base = ch_base + "/mixer/previz";
                auto&             previz_node = node_at(ch_node, ch_base, {"mixer", "previz"});
                previz_node["ACCESS"]         = 0;
                auto& previz_contents         = previz_node["CONTENTS"].is_object()
                                                    ? previz_node["CONTENTS"].as_object()
                                                    : (previz_node["CONTENTS"] = json::object()).as_object();

                const auto graft = [&](json::object&                    parent,
                                       const std::string&               base,
                                       const std::string&               name,
                                       const json::object&              tmpl,
                                       const std::string&               published_prefix) {
                    json::object obj = tmpl;
                    instantiate_mixer(obj, base + "/" + name);
                    // A published value overwrites the descriptor default, so a moved screen
                    // reads back where it is while its untouched neighbours still describe
                    // themselves. Merged rather than replaced: the published leaf knows the
                    // value, the descriptor knows everything else about it.
                    auto& oc = obj["CONTENTS"].as_object();
                    for (const auto& kv : *snap) {
                        if (!starts_with(kv.first, published_prefix))
                            continue;
                        auto it = oc.find(kv.first.substr(published_prefix.size()));
                        if (it != oc.end())
                            it->value().as_object()["VALUE"] = vector_to_oscquery_value(kv.second);
                    }
                    parent[name] = std::move(obj);
                };

                graft(previz_contents, previz_base, "camera", camera_template(),
                      "mixer/previz/camera/");
                graft(previz_contents, previz_base, "view_camera", camera_template(),
                      "mixer/previz/view_camera/");

                if (!screens.empty()) {
                    json::object screen_node;
                    screen_node["FULL_PATH"] = previz_base + "/screen";
                    screen_node["ACCESS"]    = 0;
                    json::object screen_contents;
                    for (const auto& name : screens)
                        graft(screen_contents, previz_base + "/screen", name, screen_template(),
                              "mixer/previz/screen/" + name + "/");
                    screen_node["CONTENTS"] = std::move(screen_contents);
                    previz_contents["screen"] = std::move(screen_node);
                }
            }
        }

        // Source 3: the registry, per existing layer. This is the part a snapshot cannot
        // supply -- a parameter at its default is not published, so without this pass a
        // client discovers only the parameters somebody has already changed.
        for (const int layer : layers_in(*snap)) {
            const std::string layer_base = ch_base + "/stage/layer/" + std::to_string(layer);
            const std::string base       = layer_base + "/mixer";

            auto& layer_node    = node_at(ch_node, ch_base, {"stage", "layer", std::to_string(layer)});
            layer_node["ACCESS"] = 0;

            json::object mixer = mixer_template();
            instantiate_mixer(mixer, base);

            // Any mixer key the tick DID publish overwrites the descriptor default, so a
            // graded layer reads back its real value while its untouched neighbours still
            // describe themselves. Merged rather than replaced: the published leaf knows
            // the value, and the descriptor knows everything else about it.
            const std::string published_prefix = "stage/layer/" + std::to_string(layer) + "/mixer/";
            auto&             mixer_contents   = mixer["CONTENTS"].as_object();
            for (const auto& kv : *snap) {
                if (!starts_with(kv.first, published_prefix))
                    continue;
                auto it = mixer_contents.find(kv.first.substr(published_prefix.size()));
                if (it != mixer_contents.end())
                    it->value().as_object()["VALUE"] = vector_to_oscquery_value(kv.second);
            }

            auto& contents_val = layer_node["CONTENTS"];
            if (!contents_val.is_object())
                contents_val = json::object();
            contents_val.as_object()["mixer"] = std::move(mixer);

            // Source 4: the layer's PRODUCER parameters, queried from the producer itself.
            //
            // The one part of the tree that can come from neither a table nor the snapshot. An
            // ISF shader declares its own `INPUTS` in its own header, so two layers running two
            // shaders have two different parameter sets and there is nothing static to describe
            // them with; and a parameter at its default is not published, exactly as for a mixer
            // field, so the snapshot alone would show only the ones somebody already changed.
            //
            // Costs one stage-executor round trip per layer per tree request. `/v1/tree` is a
            // discovery call, not a frame path, and `/v1/value` -- the hot one -- does not come
            // through here. A producer with no parameters returns an empty vector and no node is
            // created, so an ordinary channel pays the round trip and nothing else.
            if (ctx.stage) {
                std::vector<core::param_snapshot> params;
                if (auto stage = ctx.stage(ch)) {
                    try {
                        params = stage->describe_params(layer).get();
                    } catch (const std::exception&) {
                        // A layer that went away between the snapshot and this query. Not an
                        // error: the tree describes a moving system, and this is the one node
                        // built by asking rather than by reading.
                        //
                        // `std::exception` and NOT `...`, which is not a style preference here:
                        // this tree is built with /EHa, so a bare `catch (...)` also swallows
                        // STRUCTURED exceptions -- an access violation included. One did, for
                        // every OFX plugin carrying a 2D parameter: `parameters()` faulted, this
                        // handler ate it, and the tree quietly fell back to the read-only value
                        // leaves the state snapshot had already put there. The result looked
                        // exactly like a producer that declares no parameters, so the reading was
                        // "OFX plugins do not group their parameters" rather than "the server is
                        // crashing here". Memory corruption must reach a crash dump; only a
                        // C++ exception is the benign race this comment is about.
                    }
                }

                if (!params.empty()) {
                    const std::string params_base = layer_base + "/foreground/params";

                    json::object params_node;
                    params_node["FULL_PATH"] = params_base;
                    params_node["ACCESS"]    = 0;
                    json::object param_contents;
                    for (const auto& p : params)
                        param_contents.emplace(p.name, param_leaf(p, params_base + "/" + p.name));
                    params_node["CONTENTS"] = std::move(param_contents);

                    // Merged under the EXISTING `foreground` node, which source 2 has already
                    // created from the snapshot -- the producer publishes its name, transport and
                    // frame count there. Replacing it would drop all of that.
                    auto& fg = contents_val.as_object()["foreground"];
                    if (!fg.is_object())
                        fg = json::object();
                    auto& fg_obj = fg.as_object();
                    if (!fg_obj.if_contains("FULL_PATH"))
                        fg_obj["FULL_PATH"] = layer_base + "/foreground";
                    if (!fg_obj.if_contains("ACCESS"))
                        fg_obj["ACCESS"] = 0;
                    auto& fg_contents = fg_obj["CONTENTS"];
                    if (!fg_contents.is_object())
                        fg_contents = json::object();
                    fg_contents.as_object()["params"] = std::move(params_node);
                }
            }
        }
    }

    return root;
}

api_reply tree_at(const state_hub& hub, const http_config& cfg, const std::string& path, const api_context& ctx)
{
    auto       root     = build_tree(hub, cfg, ctx);
    const auto segments = split_path(path);

    json::value* cur = nullptr;
    json::object* obj = &root;
    for (const auto& s : segments) {
        auto contents = obj->find("CONTENTS");
        if (contents == obj->end() || !contents->value().is_object())
            return api_reply::fail(api_code::unknown_path, "no such path: " + path);
        auto& c  = contents->value().as_object();
        auto  it = c.find(s);
        if (it == c.end())
            return api_reply::fail(api_code::unknown_path, "no such path: " + path);
        cur = &it->value();
        obj = &cur->as_object();
    }

    return api_reply::ok_with(cur ? std::move(*cur) : json::value(std::move(root)));
}

json::object host_info(const http_config& cfg, int subscriptions)
{
    json::object ext;
    ext["ACCESS"]      = true;
    ext["VALUE"]       = true;
    ext["RANGE"]       = true;
    // DERIVED, not asserted. This read `true` while all 177 macro rows passed `nullptr` for
    // `description`, so the guard at the leaf never fired and the server advertised an extension
    // it never once used -- an OSCQuery client that branches on this flag took the branch that
    // finds nothing. Computing it from the table means it says what is true on the day it is
    // asked, and turns itself on when the first described field lands rather than needing anyone
    // to remember.
    const auto described = [](const auto& table) {
        return std::any_of(table.begin(), table.end(), [](const auto& f) { return f.description != nullptr; });
    };
    ext["DESCRIPTION"] = described(fields::all()) || described(fields::screen_fields()) ||
                         described(fields::camera_fields());
    ext["CLIPMODE"]    = true;
    ext["TAGS"]        = false;
    // Live values arrive on `/v1/events` as a prefix subscription, which is a different
    // mechanism from OSCQuery's per-path LISTEN. Advertising LISTEN would make a standard
    // client wait forever for updates on a socket that is never going to send any.
    ext["LISTEN"]       = false;
    ext["PATH_CHANGED"] = false;

    json::object info;
    info["NAME"]          = u8(cfg.name);
    info["PID"]           = static_cast<std::int64_t>(CASPAR_GETPID());
    info["SUBSCRIPTIONS"] = subscriptions;
    info["EXTENSIONS"]    = std::move(ext);
    info["PORT"]          = static_cast<int>(cfg.port);
    info["EXTENT"]        = u8(cfg.extent);
    return info;
}

api_reply read_value(const state_hub& hub, const std::string& path)
{
    const auto segments = split_path(path);
    if (segments.size() < 3 || segments[0] != "channel")
        return api_reply::fail(api_code::bad_request, "a value path starts with /channel/{index}/");

    if (segments[1].empty() || segments[1].find_first_not_of("0123456789") != std::string::npos)
        return api_reply::fail(api_code::channel_not_found, "channel index is not a number: " + segments[1]);

    const int  ch   = std::atoi(segments[1].c_str());
    const auto snap = hub.get(ch);
    if (!snap)
        return api_reply::fail(api_code::channel_not_found, "no channel " + segments[1]);

    std::string key;
    for (size_t i = 2; i < segments.size(); ++i) {
        if (i > 2)
            key += "/";
        key += segments[i];
    }

    const std::string full = "/channel/" + segments[1] + "/" + key;

    for (const auto& kv : *snap) {
        if (kv.first == key) {
            json::object r;
            r["path"]  = full;
            r["value"] = vector_to_json(kv.second);
            r["type"]  = osc_tags(kv.second);
            return api_reply::ok_with(std::move(r));
        }
    }

    // Not published. If it names a registry field on a layer that exists, the value IS the
    // descriptor default -- sparse publication omits defaults, so "absent" and "at its
    // default" are the same state. Answering `unknown_path` here would make an untouched
    // parameter indistinguishable from a typo, which is the one thing this reply exists to
    // tell apart.
    static const std::string layer_prefix = "stage/layer/";
    if (starts_with(key, layer_prefix)) {
        const auto slash = key.find('/', layer_prefix.size());
        if (slash != std::string::npos) {
            const auto layer = key.substr(layer_prefix.size(), slash - layer_prefix.size());
            const auto rest  = key.substr(slash + 1);
            if (starts_with(rest, "mixer/")) {
                const auto* f = fields::find(rest.substr(std::string("mixer/").size()));
                if (f) {
                    const std::string this_layer = layer_prefix + layer + "/";
                    bool              exists     = false;
                    for (const auto& kv : *snap) {
                        if (starts_with(kv.first, this_layer)) {
                            exists = true;
                            break;
                        }
                    }
                    if (!exists)
                        return api_reply::fail(api_code::layer_not_found, "no layer " + layer);

                    json::object r;
                    r["path"]       = full;
                    r["value"]      = vector_to_json(f->defaults());
                    r["type"]       = osc_tags_for(*f);
                    r["is_default"] = true;
                    return api_reply::ok_with(std::move(r));
                }
            }
        }
    }

    // The same rule for the stage. `mixer/previz/camera/{field}`,
    // `mixer/previz/view_camera/{field}` and `mixer/previz/screen/{name}/{field}` are published
    // sparsely too, so an absent one is at its default rather than unknown -- and the object has
    // to EXIST for that to be true, which is what distinguishes an untouched screen from a
    // misspelt one.
    static const std::string previz_prefix = "mixer/previz/";
    if (starts_with(key, previz_prefix)) {
        const auto rest = key.substr(previz_prefix.size());

        // The DEFAULT is carried alongside the descriptor rather than read from it, because
        // `defaults()` is the one accessor that depends on which struct the field belongs to --
        // a camera field and a screen field are different `typed_field<T>`s over one
        // `field_meta`. Capturing it here is what lets the rest of this block stay type-blind.
        const fields::field_meta* f = nullptr;
        core::monitor::vector_t   def;
        bool                      exists = false;

        auto object_published = [&](const std::string& obj_prefix) {
            for (const auto& kv : *snap)
                if (starts_with(kv.first, previz_prefix + obj_prefix))
                    return true;
            return false;
        };

        if (starts_with(rest, "camera/") || starts_with(rest, "view_camera/")) {
            const auto slash = rest.find('/');
            if (const auto* cf = fields::find_camera_field(rest.substr(slash + 1))) {
                f   = cf;
                def = cf->defaults();
            }
            // A camera always exists once previz does, so `previz/screens` -- which is always
            // published -- is the presence test rather than the camera's own keys, which are
            // sparse and can all be absent on an untouched camera.
            exists = object_published("screens");
            (void)slash;
        } else if (starts_with(rest, "screen/")) {
            const auto body  = rest.substr(std::string("screen/").size());
            const auto slash = body.find('/');
            if (slash != std::string::npos) {
                const auto name = body.substr(0, slash);
                if (const auto* sf = fields::find_screen_field(body.substr(slash + 1))) {
                    f   = sf;
                    def = sf->defaults();
                }
                // A screen with every field at its default publishes no field keys at all, so
                // its existence is read out of the always-published `screens` list.
                for (const auto& kv : *snap) {
                    if (kv.first != previz_prefix + "screens")
                        continue;
                    for (const auto& d : kv.second) {
                        const auto* nm = boost::get<std::string>(&d);
                        if (nm && *nm == name)
                            exists = true;
                    }
                }
            }
        }

        if (f) {
            if (!exists)
                return api_reply::fail(api_code::unknown_path,
                                       "no such stage object on channel " + segments[1] + ": " + path);
            json::object r;
            r["path"]       = full;
            r["value"]      = vector_to_json(def);
            r["type"]       = osc_tags_for(*f);
            r["is_default"] = true;
            return api_reply::ok_with(std::move(r));
        }
    }

    return api_reply::fail(api_code::unknown_path, "no such path: " + path);
}

}}} // namespace caspar::protocol::http
