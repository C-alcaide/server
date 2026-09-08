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

#include "api_context.h"
#include "api_status.h"
#include "http_config.h"
#include "state_hub.h"

#include <core/monitor/monitor.h>

#include <string>
#include <vector>

namespace caspar { namespace protocol { namespace http {

/// A `monitor::data_t` vector as JSON, preserving the type it was published with.
///
/// A scalar field returns a bare value and a vector field returns an array: the arity is
/// fixed per path by the code that produces it, so a client can rely on the shape without
/// asking. (OSCQuery's own `VALUE` is always an array; that is a separate representation
/// and `tree_node` emits it that way.)
json::value to_json(const core::monitor::vector_t& v);

/// The OSC type-tag string for a published vector -- `d`, `ff`, `s`, `T`. This is
/// OSCQuery's `TYPE`, and it is what tells a standard client how to read `VALUE`.
std::string osc_type_tags(const core::monitor::vector_t& v);

/// Build the OSCQuery tree.
///
/// Three sources are joined, and the reason there are three is that none alone describes
/// the address space:
///
///   * the CHANNELS and LAYERS that exist, from the hub's snapshots -- every existing
///     layer is written every tick, so this is complete even though individual keys are
///     published sparsely;
///   * every key the snapshot actually carries, as a read-only node -- this is the
///     telemetry, and it is what OSC has always exposed;
///   * the transform field REGISTRY, per layer -- which is the part a snapshot cannot
///     supply, because a parameter at its default is not published and would otherwise be
///     invisible to a client trying to discover what it can set.
///
/// `extent` decides whether the third source is included at all.
/// `ctx` may be default-constructed; its `stage` is checked before use. It is here for one
/// thing only -- a layer's PRODUCER parameters, which are per-instance and therefore in no
/// static table and not fully in the published snapshot either. Everything else the tree
/// needs comes from `hub` and the three field registries.
json::object build_tree(const state_hub& hub, const http_config& cfg, const api_context& ctx = {});

/// A sub-tree, addressed the way OSCQuery addresses one: `/channel/1/stage/layer/1/mixer`.
/// Returns `unknown_path` when the path names nothing.
api_reply tree_at(const state_hub& hub,
                  const http_config& cfg,
                  const std::string& path,
                  const api_context& ctx = {});

/// OSCQuery's `?HOST_INFO`. Also the capability probe: a client asks what this server can
/// do rather than deriving it from a version number.
json::object host_info(const http_config& cfg, int subscriptions);

/// Read one value. `/channel/1/stage/layer/1/foreground/file/time`.
///
/// A registry path that the snapshot does not carry reads as the descriptor's default,
/// because a parameter at its default is not published -- the alternative would be
/// `unknown_path` for every untouched parameter, which is indistinguishable from a typo.
api_reply read_value(const state_hub& hub, const std::string& path);

}}} // namespace caspar::protocol::http
