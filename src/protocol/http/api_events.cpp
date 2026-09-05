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

#include "api_events.h"
#include "json_state.h"

#include <core/frame/transform_fields.h>

#include <algorithm>
#include <cstdlib>
#include <set>

namespace caspar { namespace protocol { namespace http {

namespace fields = core::fields;

bool prefix_matches(const std::string& path, const std::string& prefix)
{
    if (prefix.empty() || prefix == "/")
        return true;
    if (path.size() < prefix.size() || path.compare(0, prefix.size(), prefix) != 0)
        return false;
    // Exactly the prefix, or the prefix followed by a separator. A prefix that already
    // ends in `/` has consumed the boundary itself.
    return path.size() == prefix.size() || prefix.back() == '/' || path[prefix.size()] == '/';
}

namespace {

/// The channel indices a set of prefixes can possibly touch. Every prefix is required to
/// start `/channel/<n>`, so this is exact rather than a filter -- and it is what stops a
/// subscription to one channel from walking every other channel's snapshot.
std::set<int> channels_in(const std::vector<std::string>& prefixes)
{
    std::set<int>     out;
    const std::string head = "/channel/";
    for (const auto& p : prefixes) {
        if (p.size() <= head.size() || p.compare(0, head.size(), head) != 0)
            continue;
        const auto end = p.find('/', head.size());
        const auto num = p.substr(head.size(), end == std::string::npos ? std::string::npos : end - head.size());
        if (!num.empty() && num.find_first_not_of("0123456789") == std::string::npos)
            out.insert(std::atoi(num.c_str()));
    }
    return out;
}

bool any_prefix_matches(const std::string& path, const std::vector<std::string>& prefixes)
{
    for (const auto& p : prefixes)
        if (prefix_matches(path, p))
            return true;
    return false;
}

/// The value a path reverts TO when it stops being published.
///
/// A mixer field that is no longer in the sparse set is at its declared default -- that is
/// what "absent" means on this wire. Anything else that disappears (a layer that was
/// cleared, a producer key that stopped being written) has no default to report, so the
/// event carries `null` and says `reverted`, which is honest about the difference.
json::value reverted_value(const std::string& path)
{
    const auto pos = path.find("/mixer/");
    if (pos == std::string::npos)
        return nullptr;
    const auto* f = fields::find(path.substr(pos + 7));
    return f ? vector_to_json(f->defaults()) : json::value(nullptr);
}

} // namespace

json::object collect_events(subscription& sub, const state_hub& hub, const std::string& server_name)
{
    const auto now = std::chrono::steady_clock::now();
    if (sub.throttle_ms > 0) {
        const auto since = std::chrono::duration_cast<std::chrono::milliseconds>(now - sub.last_sent).count();
        if (since < sub.throttle_ms)
            return {};
    }

    json::array  events;
    json::object frames;

    // What this subscriber currently sees. Built fresh so a key that VANISHED is detectable
    // as "in last_values, not in seen" -- which is the half a naive diff misses, and the
    // half that matters most here because a field returning to its default is exactly that.
    std::map<std::string, core::monitor::vector_t> seen;

    for (const int ch : channels_in(sub.prefixes)) {
        const auto snap = hub.get(ch);
        if (!snap)
            continue;

        const std::string base = "/channel/" + std::to_string(ch);

        for (const auto& kv : *snap) {
            const auto path = base + "/" + kv.first;
            if (!any_prefix_matches(path, sub.prefixes))
                continue;

            seen.emplace(path, kv.second);

            const auto prev = sub.last_values.find(path);
            if (prev != sub.last_values.end() && sub.repetition_filter && prev->second == kv.second)
                continue;

            json::object e;
            e["path"]  = path;
            e["value"] = vector_to_json(kv.second);
            events.push_back(std::move(e));
        }

        // The channel's own frame number, whether or not it is subscribed: an event is
        // worth much less without the frame it belongs to, and a client should not have to
        // subscribe to `/channel/N/frame` to correlate.
        for (const auto& kv : *snap) {
            if (kv.first == "frame" && !kv.second.empty()) {
                frames[std::to_string(ch)] = data_to_json(kv.second.front());
                break;
            }
        }
    }

    for (const auto& kv : sub.last_values) {
        if (seen.find(kv.first) != seen.end())
            continue;
        json::object e;
        e["path"]     = kv.first;
        e["value"]    = reverted_value(kv.first);
        e["reverted"] = true;
        events.push_back(std::move(e));
    }

    sub.last_values = std::move(seen);

    if (events.empty())
        return {};

    sub.last_sent = now;

    json::object msg;
    msg["id"]     = sub.id;
    msg["server"] = server_name;
    msg["frame"]  = std::move(frames);
    msg["events"] = std::move(events);
    return msg;
}

api_reply parse_subscribe(const json::value& msg, int max_prefixes, subscription& out)
{
    if (!msg.is_object())
        return api_reply::fail(api_code::bad_request, "a message must be a JSON object");
    const auto& o = msg.as_object();

    if (auto* id = o.if_contains("id")) {
        if (!id->is_string())
            return api_reply::fail(api_code::field_wrong_type, "id must be a string");
        out.id = id->as_string().c_str();
    }
    if (out.id.empty())
        out.id = "s1";

    const auto* prefixes = o.if_contains("prefixes");
    if (!prefixes || !prefixes->is_array() || prefixes->as_array().empty())
        return api_reply::fail(api_code::field_missing, "subscribe needs a non-empty prefixes array");

    const auto& arr = prefixes->as_array();
    if (static_cast<int>(arr.size()) > max_prefixes)
        return api_reply::fail(api_code::bad_request,
                               "at most " + std::to_string(max_prefixes) + " prefixes per subscription");

    out.prefixes.clear();
    for (const auto& p : arr) {
        if (!p.is_string())
            return api_reply::fail(api_code::field_wrong_type, "every prefix must be a string");
        std::string s = p.as_string().c_str();
        // Required, and not merely conventional: the channel index in the prefix is what
        // bounds the scan to the channels a subscriber actually asked for. Without it a
        // single subscription walks every channel's whole snapshot on every tick.
        if (s.compare(0, 9, "/channel/") != 0)
            return api_reply::fail(api_code::bad_request, "a prefix must start with /channel/{index}: " + s);
        out.prefixes.push_back(std::move(s));
    }

    if (auto* t = o.if_contains("throttle_ms")) {
        if (!t->is_int64() && !t->is_uint64() && !t->is_double())
            return api_reply::fail(api_code::field_wrong_type, "throttle_ms must be a number");
        out.throttle_ms = std::max(0, static_cast<int>(t->to_number<double>()));
    }
    if (auto* r = o.if_contains("repetition_filter")) {
        if (!r->is_bool())
            return api_reply::fail(api_code::field_wrong_type, "repetition_filter must be a boolean");
        out.repetition_filter = r->as_bool();
    }

    out.last_values.clear();
    out.last_sent = {};
    return api_reply{};
}

}}} // namespace caspar::protocol::http
