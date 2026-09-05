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

/// The per-layer `mixer` sub-tree. Built once and copied per layer: the descriptor set is
/// the bulk of the tree and is identical for every layer on every channel.
const json::object& mixer_template()
{
    static const json::object tmpl = [] {
        json::object node;
        node["FULL_PATH"] = "";
        node["ACCESS"]    = 0;
        json::object contents;

        for (const auto& f : fields::all()) {
            json::object leaf;
            leaf["FULL_PATH"] = "";
            leaf["ACCESS"]    = static_cast<int>(static_cast<uint8_t>(f.access));
            leaf["TYPE"]      = osc_tags_for(f);
            leaf["VALUE"]     = vector_to_oscquery_value(f.defaults());
            // A field with neither limits nor a value list gets no `RANGE` key at all.
            // Emitting `[{}]` would be worse than saying nothing: a client reads the key's
            // presence as "this is bounded" and then finds no MIN or MAX to bound it with.
            if (auto range = range_for(f); !range.empty())
                leaf["RANGE"] = std::move(range);
            leaf["CLIPMODE"] = clipmode_name(f.bounding, f.range.has_value());
            if (f.description)
                leaf["DESCRIPTION"] = f.description;
            leaf["casparcg"] = vendor_block(f, f.defaults());
            contents.emplace(f.path, std::move(leaf));
        }

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

json::object build_tree(const state_hub& hub, const http_config& cfg)
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
        }
    }

    return root;
}

api_reply tree_at(const state_hub& hub, const http_config& cfg, const std::string& path)
{
    auto       root     = build_tree(hub, cfg);
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
    const auto& all         = fields::all();
    ext["DESCRIPTION"]      = std::any_of(all.begin(), all.end(),
                                     [](const fields::field_desc& f) { return f.description != nullptr; });
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

    return api_reply::fail(api_code::unknown_path, "no such path: " + path);
}

}}} // namespace caspar::protocol::http
