/*
 * Copyright (c) 2026 CasparCG Contributors
 *
 * This file is part of CasparCG (www.casparcg.com).
 *
 * CasparCG is free software: you can redistribute it and/or modify it under the terms of the GNU
 * General Public License as published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 */

#pragma once

#include <core/graph/isf_render.h>
#include <core/graph/registry.h>

#include <core/module_dependencies.h>

namespace caspar { namespace isf {

void init(const core::module_dependencies& dependencies);

/// Map one ISF shader's declared INPUTS onto node-graph ports.
///
/// Registered by the shell as the `isf` class's port resolver, so `core/graph` can describe a
/// node whose ports come from a file without linking this module -- the same injection the
/// stage uses for its producer factory.
///
/// `selector` is the node's `path` parameter. Returns an empty list and sets `out_reason` when
/// the shader cannot be read or its header will not parse; the validator turns that into ONE
/// fault naming `path`, rather than one per port that went missing with it.
std::vector<core::graph::port_desc> resolve_node_ports(const std::string& selector, std::string& out_reason);

/// Draw one `isf` node on the mixer's own GL context. Injected into core at boot via
/// `core::graph::set_isf_node_renderer`, for the same reason `resolve_node_ports` is: the
/// dependency runs module -> core and never the other way.
///
/// Returns false for anything that stops it drawing -- a shader that will not compile, a path
/// that does not resolve, a bad texture. The evaluator renders the node's input UNCHANGED in
/// that case rather than black.
bool render_node(const core::graph::isf_node_request& req);

/// Generate one ISF shader as Vulkan GLSL. Injected into core at boot, and the reason it is a
/// seam rather than a call is the same one as above: core cannot link this module.
core::graph::isf_vulkan_source vulkan_source_for(const std::string& path);

/// This shader's IMPORTED images, decoded to RGBA8, in the header's declaration order.
std::vector<core::graph::isf_imported_image> imported_images_for(const std::string& path);

/// A shader's passes, sized for one frame. Injected at boot; see `core::graph::set_isf_pass_planner`.
std::vector<core::graph::isf_pass_plan>
plan_passes_for(const std::string&                                     path,
                int                                                    render_w,
                int                                                    render_h,
                const std::function<bool(const std::string&, double&)>& value,
                std::string&                                           out_error);

}} // namespace caspar::isf
