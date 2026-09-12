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

#include <cstdint>
#include <functional>
#include <string>

namespace caspar { namespace core { namespace graph {

/// One ISF node's draw, as the evaluator hands it to `modules/isf`.
///
/// GL TEXTURE IDS RATHER THAN A TEXTURE TYPE, because this struct crosses a link boundary that
/// has no shared texture class: `core` cannot link `modules/isf` (the same inversion
/// `set_port_resolver` exists for), and `modules/isf` must not link either accelerator's texture.
/// An unsigned id is the one thing both sides already agree on -- it is what
/// `isf::shader::render_into_shared` takes today, in production, for the Vulkan producer route.
struct isf_node_request
{
    /// The shader file, exactly as the node's `path` parameter carries it.
    std::string path;

    /// The node's value slots, and THEIR NAMES.
    ///
    /// MATCHED BY NAME, NOT BY POSITION, and that is load-bearing rather than fastidious. The
    /// compiler packs slots from `instance_ports`, whose dynamic half this module itself
    /// produced -- so the two orders agree today by construction and would go on agreeing right
    /// up until either side inserted a port. A positional mapping that drifts sends every
    /// parameter to the WRONG INPUT: the shader still compiles, still renders, and is wrong with
    /// nothing to report it. That is the failure mode this codebase has paid for twice.
    ///
    /// `names[k]` names `values[k]`. Both point into the plan, which is immutable and shared, so
    /// neither is copied per frame.
    const double*      values      = nullptr;
    const std::string* names       = nullptr;
    std::uint32_t      value_count = 0;

    /// The input picture, and where the result goes. Both are GL textures on the current
    /// context, RGBA, top-down, BGRA-labelled -- the mixer's own convention.
    unsigned int src_tex = 0;
    unsigned int dst_tex = 0;
    int          width   = 0;
    int          height  = 0;

    /// The channel's clock: `TIME`, `TIMEDELTA` and `FRAMEINDEX`. See
    /// `core::image_mixer::set_frame_number` for why these come from the channel rather than
    /// from a wall clock or from the transform.
    double time       = 0.0;
    double time_delta = 0.0;
    int    frame_index = 0;
};

/// Renders one ISF node. Returns false if the shader could not be compiled or drawn, which the
/// evaluator treats as a DEAD pass: the node renders its input unchanged rather than black,
/// because a shader that fails to compile mid-show must not take the layer off air.
using isf_node_renderer = std::function<bool(const isf_node_request&)>;

/// Injected once at boot by `shell/server.cpp`, for the same reason `set_port_resolver` is: the
/// dependency runs module -> core, so core holds a hook and the module fills it.
void set_isf_node_renderer(isf_node_renderer r);

/// Null until the module registers itself, and the evaluator checks rather than assumes -- a
/// build with the ISF module disabled must still run a document that mentions an `isf` node.
const isf_node_renderer& get_isf_node_renderer();

}}} // namespace caspar::core::graph
