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

#include "json_state.h"

#include <common/utf.h>

#include <boost/variant/static_visitor.hpp>

namespace caspar { namespace protocol { namespace http {

namespace {

struct to_json_visitor : boost::static_visitor<json::value>
{
    json::value operator()(bool v) const { return json::value(v); }
    json::value operator()(std::int32_t v) const { return json::value(static_cast<std::int64_t>(v)); }
    json::value operator()(std::int64_t v) const { return json::value(v); }
    json::value operator()(std::uint32_t v) const { return json::value(static_cast<std::uint64_t>(v)); }
    json::value operator()(std::uint64_t v) const { return json::value(v); }
    json::value operator()(float v) const { return json::value(static_cast<double>(v)); }
    json::value operator()(double v) const { return json::value(v); }
    json::value operator()(const std::string& v) const { return json::value(v); }
    json::value operator()(const std::wstring& v) const { return json::value(u8(v)); }
};

struct tag_visitor : boost::static_visitor<char>
{
    // OSC 1.0 encodes a boolean in the tag itself -- `T` and `F` are distinct types with
    // no payload -- so the tag depends on the VALUE, not only on its type. A client that
    // reads `T` knows both that the field is boolean and that it is currently true.
    char operator()(bool v) const { return v ? 'T' : 'F'; }
    char operator()(std::int32_t) const { return 'i'; }
    char operator()(std::int64_t) const { return 'h'; }
    char operator()(std::uint32_t) const { return 'i'; }
    char operator()(std::uint64_t) const { return 'h'; }
    char operator()(float) const { return 'f'; }
    char operator()(double) const { return 'd'; }
    char operator()(const std::string&) const { return 's'; }
    char operator()(const std::wstring&) const { return 's'; }
};

} // namespace

json::value data_to_json(const core::monitor::data_t& d) { return boost::apply_visitor(to_json_visitor(), d); }

char osc_tag(const core::monitor::data_t& d) { return boost::apply_visitor(tag_visitor(), d); }

json::value vector_to_json(const core::monitor::vector_t& v)
{
    if (v.size() == 1)
        return data_to_json(v.front());

    json::array a;
    a.reserve(v.size());
    for (const auto& d : v)
        a.push_back(data_to_json(d));
    return json::value(std::move(a));
}

std::string osc_tags(const core::monitor::vector_t& v)
{
    std::string s;
    s.reserve(v.size());
    for (const auto& d : v)
        s.push_back(osc_tag(d));
    return s;
}

json::array vector_to_oscquery_value(const core::monitor::vector_t& v)
{
    json::array a;
    a.reserve(v.size());
    for (const auto& d : v)
        a.push_back(data_to_json(d));
    return a;
}

std::string osc_tags_for(const core::fields::field_desc& f)
{
    using core::fields::value_type;

    char c = 'd';
    switch (f.type) {
        case value_type::boolean:
        // A blob reports PRESENCE, and presence is a boolean -- `lut3d` reads back `false`
        // when no LUT is loaded. This said `s` on its first outing, which put a string tag
        // on a boolean value in the same node: the one thing `TYPE` exists to prevent.
        case value_type::blob:
            // A descriptor cannot know whether the value is true or false, and OSC has no
            // type-agnostic boolean tag. `F` is the honest answer for a type declaration:
            // every boolean field in this table defaults to false, so it is also correct
            // for an unpublished one, and `VALUE` carries the truth wherever it differs.
            c = 'F';
            break;
        case value_type::integer:
        case value_type::enumeration: c = 'i'; break;
        case value_type::string: c = 's'; break;
        default: c = 'd'; break;
    }
    return std::string(f.arity, c);
}

const char* type_name(core::fields::value_type t)
{
    using core::fields::value_type;
    switch (t) {
        case value_type::boolean: return "boolean";
        case value_type::integer: return "integer";
        case value_type::real: return "real";
        case value_type::string: return "string";
        case value_type::enumeration: return "enumeration";
        case value_type::vec2: return "vec2";
        case value_type::vec3: return "vec3";
        case value_type::vec4: return "vec4";
        case value_type::blob: return "blob";
    }
    return "real";
}

const char* bounding_name(core::fields::bounding_t b)
{
    using core::fields::bounding_t;
    switch (b) {
        case bounding_t::free: return "free";
        case bounding_t::clip: return "clip";
        case bounding_t::wrap: return "wrap";
        case bounding_t::fold: return "fold";
    }
    return "free";
}

const char* compose_name(core::fields::compose_t c)
{
    using core::fields::compose_t;
    switch (c) {
        case compose_t::none: return "none";
        case compose_t::multiply: return "multiply";
        case compose_t::add: return "add";
        case compose_t::min_: return "min";
        case compose_t::max_: return "max";
        case compose_t::or_: return "or";
        case compose_t::xor_: return "xor";
        case compose_t::innermost_wins: return "innermost_wins";
        case compose_t::custom: return "custom";
    }
    return "none";
}

const char* kind_name(core::fields::kf_kind k)
{
    using core::fields::kf_kind;
    switch (k) {
        case kf_kind::continuous: return "continuous";
        case kf_kind::angular: return "angular";
        case kf_kind::angular_rad: return "angular";
        case kf_kind::discrete: return "discrete";
    }
    return "continuous";
}

const char* clipmode_name(core::fields::bounding_t b, bool has_range)
{
    using core::fields::bounding_t;
    if (!has_range)
        return "none";
    // Only `clip` maps. `wrap` and `fold` are ossia's vocabulary, not OSCQuery's -- see
    // the header for why they deliberately report `none` here.
    return b == bounding_t::clip ? "both" : "none";
}

}}} // namespace caspar::protocol::http
