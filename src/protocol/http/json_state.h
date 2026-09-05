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

#pragma once

#include "boost_prelude.h"

#include <core/frame/transform_fields.h>
#include <core/monitor/monitor.h>

#include <string>

namespace caspar { namespace protocol { namespace http {

/// One published datum as JSON, preserving which of the nine `data_t` alternatives it is.
///
/// `wstring` is converted to UTF-8 rather than escaped as UTF-16: JSON strings are Unicode
/// and a client should never have to know that the server's file paths are wide.
json::value data_to_json(const core::monitor::data_t& d);

/// A whole published vector. A one-element vector yields the bare value; anything else an
/// array. The arity is fixed per path by the code that publishes it, so this shape is
/// stable for a given path even though the rule is per-value.
json::value vector_to_json(const core::monitor::vector_t& v);

/// The OSC type tag for one datum -- `T`/`F` for a bool (OSC has no `b`), `i`, `h`, `f`,
/// `d`, `s`. Concatenated over a vector, this is OSCQuery's `TYPE`.
char osc_tag(const core::monitor::data_t& d);
std::string osc_tags(const core::monitor::vector_t& v);

/// OSCQuery's `VALUE`, which is ALWAYS an array -- including for a scalar. Kept separate
/// from `vector_to_json` on purpose: the tree speaks OSCQuery and `/v1/value` speaks the
/// shape a client would write back, and conflating them would make one of the two wrong.
json::array vector_to_oscquery_value(const core::monitor::vector_t& v);

/// The type tag a descriptor declares, before any value exists. Needed because the tree
/// describes fields that have never been published.
std::string osc_tags_for(const core::fields::field_meta& f);

/// The wire name of a `value_type`, for the vendor block.
const char* type_name(core::fields::value_type t);
const char* bounding_name(core::fields::bounding_t b);
const char* compose_name(core::fields::compose_t c);
const char* kind_name(core::fields::kf_kind k);
/// OSCQuery's CLIPMODE, which is `none|low|high|both` -- NOT the same vocabulary as
/// `bounding`. `wrap` and `fold` have no CLIPMODE, and saying `both` for them would tell a
/// standard client to clamp a hue rotation. They report `none` here and their real rule in
/// the vendor block.
const char* clipmode_name(core::fields::bounding_t b, bool has_range);

}}} // namespace caspar::protocol::http
