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

#include "keyframe_json.h"
#include "keyframe_fields.h"

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include <sstream>
#include <stdexcept>

namespace caspar { namespace keyframes {

namespace pt = boost::property_tree;

// ═══════════════════════════════════════════════════════════════════════════
//  Serialize: timeline → JSON (sparse — only non-default values)
// ═══════════════════════════════════════════════════════════════════════════

std::string timeline_to_json(const keyframe_timeline& tl)
{
    pt::ptree root;
    pt::ptree kf_array;

    for (const auto& kf : tl.keyframes()) {
        pt::ptree node;
        node.put("time_secs", kf.time_secs);
        node.put("easing",    kf.easing_name);

        // Emit only the fields that are present (sparse)
        for (const auto& [name, value] : kf.values)
            node.put(name, value);

        kf_array.push_back({"", node});
    }

    root.add_child("keyframes", kf_array);

    std::ostringstream oss;
    oss.precision(17);  // full double round-trip fidelity
    pt::write_json(oss, root);
    return oss.str();
}

// ═══════════════════════════════════════════════════════════════════════════
//  Deserialize: JSON → timeline
// ═══════════════════════════════════════════════════════════════════════════

static constexpr size_t MAX_KEYFRAMES = 10000;
static constexpr size_t MAX_JSON_SIZE = 1024 * 1024; // 1 MB

keyframe_timeline json_to_timeline(const std::string& json)
{
    keyframe_timeline tl;

    if (json.size() > MAX_JSON_SIZE)
        throw std::runtime_error("KEYFRAMES JSON too large (max 1 MB)");

    pt::ptree root;
    try {
        std::istringstream iss(json);
        pt::read_json(iss, root);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("KEYFRAMES JSON parse error: ") + e.what());
    }

    auto child = root.get_child_optional("keyframes");
    if (!child)
        return tl;

    size_t count = 0;
    for (const auto& kv : *child) {
        if (++count > MAX_KEYFRAMES)
            throw std::runtime_error("KEYFRAMES: too many keyframes (max 10000)");

        const pt::ptree& node = kv.second;
        keyframe_t kf;

        kf.time_secs = node.get<double>("time_secs", 0.0);
        if (!std::isfinite(kf.time_secs) || kf.time_secs < 0.0)
            throw std::runtime_error("KEYFRAMES: invalid time_secs value");

        kf.easing_name = node.get<std::string>("easing", "LINEAR");
        kf.easing_fn   = resolve_easing(kf.easing_name);

        // Parse all other keys as field values (sparse)
        for (const auto& field_kv : node) {
            const std::string& key = field_kv.first;
            if (key == "time_secs" || key == "easing")
                continue;

            // Only accept keys that match known field descriptors
            if (kf_find_field(key)) {
                try {
                    kf.values[key] = field_kv.second.get_value<double>();
                } catch (...) {
                    // Skip non-numeric values silently
                }
            }
            // Unknown keys are silently ignored (forward compatibility)
        }

        tl.add(std::move(kf));
    }

    return tl;
}

// ═══════════════════════════════════════════════════════════════════════════
//  Parse partial JSON → kf_values (for PATCH command)
// ═══════════════════════════════════════════════════════════════════════════

kf_values parse_kf_values(const std::string& json_str)
{
    if (json_str.size() > MAX_JSON_SIZE)
        throw std::runtime_error("KEYFRAMES PATCH JSON too large (max 1 MB)");

    kf_values vals;

    pt::ptree root;
    std::istringstream iss(json_str);
    pt::read_json(iss, root);

    for (const auto& kv : root) {
        const std::string& key = kv.first;
        if (kf_find_field(key)) {
            try {
                vals[key] = kv.second.get_value<double>();
            } catch (...) {
                // Skip non-numeric
            }
        }
    }

    return vals;
}

}} // namespace caspar::keyframes
