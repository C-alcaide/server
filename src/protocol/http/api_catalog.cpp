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

#include "api_catalog.h"

#include "boost_prelude.h"

#include <algorithm>
#include <map>

namespace caspar { namespace protocol { namespace http {

namespace {

/// The formats this endpoint knows, and the only values `kind` accepts.
///
/// A fixed list rather than "whatever the entries happen to carry", so that a server with no
/// OFX plug-ins installed still answers `/v1/catalog/ofx` with an empty list instead of
/// `unknown_path`. Those two answers mean completely different things to a client: one is
/// "nothing is installed", the other is "this server does not do OFX at all", and a client
/// that cannot tell them apart shows the operator the wrong message.
const char* const kKinds[] = {"ofx", "isf"};

json::object entry_json(const catalog_entry& e)
{
    json::object o;
    o["id"]    = e.id;
    o["label"] = e.label.empty() ? e.id : e.label;
    // Absent, not empty, when the format declared none -- the same rule `param_leaf` follows
    // for `group`, so a client can tell "ungrouped" from "grouped under the empty name".
    if (!e.group.empty())
        o["group"] = e.group;
    for (const auto& kv : e.properties)
        if (!kv.second.empty())
            o[kv.first] = kv.second;
    return o;
}

} // namespace

api_reply catalog(const api_context& ctx, const std::string& kind)
{
    const bool known = kind.empty() ||
                       std::any_of(std::begin(kKinds), std::end(kKinds),
                                   [&](const char* k) { return kind == k; });
    if (!known)
        return api_reply::fail(api_code::unknown_path,
                               "no such catalogue: " + kind + ". Known: ofx, isf");

    // NOT an error when the shell wired nothing. This endpoint is answered by a build that has
    // the modules and by one that does not, and a build without them has an empty catalogue
    // rather than a broken one.
    std::vector<catalog_entry> all;
    if (ctx.catalog) {
        try {
            all = ctx.catalog();
        } catch (const std::exception& e) {
            // `std::exception` and not `...`: this tree is built with /EHa, where a bare
            // `catch (...)` also swallows an access violation and turns memory corruption into
            // an empty list. That exact handler hid a fault in `describe_params` for a session.
            return api_reply::fail(api_code::internal,
                                   std::string("the catalogue could not be built: ") + e.what());
        }
    }

    std::map<std::string, json::array> by_kind;
    for (const char* k : kKinds)
        if (kind.empty() || kind == k)
            by_kind.emplace(k, json::array{});

    for (const auto& e : all) {
        auto it = by_kind.find(e.kind);
        if (it != by_kind.end())
            it->second.push_back(entry_json(e));
    }

    json::object result;
    for (auto& [k, arr] : by_kind) {
        json::object section;
        // The COUNT alongside the array, because a client paging a long list or checking
        // whether a refresh changed anything should not have to measure the array to find out.
        section["count"]   = static_cast<std::int64_t>(arr.size());
        section["entries"] = std::move(arr);
        result[k]          = std::move(section);
    }

    return api_reply::ok_with(std::move(result));
}

}}} // namespace caspar::protocol::http
