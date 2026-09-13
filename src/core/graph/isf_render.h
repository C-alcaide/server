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
#include <memory>
#include <string>
#include <vector>

namespace caspar { namespace core { namespace graph {

/// Whatever `modules/isf` needs to keep for ONE `isf` node between frames.
///
/// ── WHY PER INSTANCE, AND WHY THIS IS AN OPAQUE BASE ────────────────────────────────────────
///
/// The ISF specification says a persistent buffer is *"saved across frames, and stays with your
/// effect until its deletion"* -- per EFFECT INSTANCE, not per shader file. Every engine that has
/// this feature agrees: TouchDesigner scopes it to the operator, Smode to the modifier, vvvv to
/// the process node, OpenFX to `kOfxPropInstanceData` (*"unique to each plug-in instance, so two
/// instances of the same plug-in do not share the same data pointer"*).
///
/// Two nodes running one shader file must therefore NOT share buffers: each would accumulate into
/// the other's history. The node path keyed its compiled shaders by PATH, and its own comment
/// promised this would become per instance in the commit that landed persistent buffers.
///
/// OPAQUE, because `core` cannot see a GL object or a `vk::Image`. The evaluator owns the slot and
/// its lifetime; the module owns what is in it.
struct isf_node_state
{
    virtual ~isf_node_state() = default;
};

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

    /// This node's own state slot, owned by the mixer and keyed per instance. The module fills
    /// it on first draw and finds its own object there on every later one. Never null.
    std::shared_ptr<isf_node_state>* state = nullptr;

    /// True while the node's `reset` port is held. Re-seeds the instance: `FRAMEINDEX` back to
    /// 0 and every persistent buffer re-blackened, every frame it stays true.
    ///
    /// LEVEL, NOT EDGE, which is TouchDesigner's shape -- it ships a latching `Reset` *and* a
    /// one-frame `Reset Pulse`, and a level port is both: hold it for the first, write 1 then 0
    /// for the second. No edge detection means the two backends cannot disagree about when a
    /// pulse was seen.
    bool reset = false;

    /// The channel's clock: `TIME`, `TIMEDELTA` and `FRAMEINDEX`. See
    /// `core::image_mixer::set_frame_number` for why these come from the channel rather than
    /// from a wall clock or from the transform.
    /// Which way to convert around the shader, from the node's `space` port against the
    /// graph's `stage`: +1 = the pass is WORKING and the shader wants DISPLAY, -1 = the
    /// opposite, 0 = they already agree and nothing is applied.
    ///
    /// Zero for `space: match` and for every agreeing combination, which is the overwhelming
    /// majority -- at zero the shader is bit-identical to one built before this existed.
    int to_display = 0;

    double time       = 0.0;
    double time_delta = 0.0;

    /// ISF's `FRAMEINDEX`: **0 on this INSTANCE's first drawn frame**, not the channel's.
    ///
    /// The spec defines it that way -- *"this value is 0 when the first frame is rendered"* --
    /// and the reset idiom every published feedback shader uses depends on it:
    /// `if (FRAMEINDEX < 1 || resetEvent) { seed the buffer }`. Fed the channel counter, a node
    /// attached mid-show never sees 0 and such a shader never initialises.
    ///
    /// `TIME` and `TIMEDELTA` stay on the CHANNEL clock, deliberately: they are what keeps two
    /// ISF nodes on a channel in step with each other and with the timeline. The two answer
    /// different questions and every reference host separates them.
    int    frame_index = 0;
};

/// Renders one ISF node. Returns false if the shader could not be compiled or drawn, which the
/// evaluator treats as a DEAD pass: the node renders its input unchanged rather than black,
/// because a shader that fails to compile mid-show must not take the layer off air.
using isf_node_renderer = std::function<bool(const isf_node_request&)>;

/// One ISF shader as Vulkan GLSL, ready to compile to SPIR-V.
///
/// The Vulkan mixer draws an ISF node as a per-layer VARIANT PIPELINE -- the same mechanism an
/// OCIO transform already uses -- rather than by running the author's GLSL on a GL context. It
/// cannot use the GL route: a node's input is an attachment inside a renderpass that commits once
/// per frame, so there is nothing for a GL context to import, and forcing the issue by committing
/// mid-accumulation is measured at 241 device losses.
struct isf_vulkan_source
{
    /// The generated GLSL. Empty when `error` is set.
    std::string source;
    /// Stable id for the pipeline cache. Two nodes naming the same file share one pipeline.
    std::string cache_id;
    /// How many value components the author's inputs consume, so the caller packs exactly that
    /// many into the uniform array.
    int value_count = 0;
    /// Why nothing was generated -- a missing file, an unparseable header, more parameters than
    /// the uniform array holds. Empty on success.
    std::string error;
};

/// One pass of a multi-pass ISF shader, as the evaluator needs it.
struct isf_pass_plan
{
    /// `TARGET`, or empty for the pass that renders the node's output.
    std::string target;
    bool        persistent = false;
    /// This pass's extent, already evaluated from its `WIDTH`/`HEIGHT` expressions.
    int         width  = 0;
    int         height = 0;
};

/// A shader's passes, sized for one draw of one node.
///
/// `render_w`/`render_h` are what the node is being asked to produce; `value` resolves a `$name`
/// in a size expression to the node's current parameter value. **Evaluated ONCE PER FRAME**, as
/// the spec requires: *"this equation is evaluated once per frame ... it's not evaluated multiple
/// times if the ISF file describes multiple rendering passes"*.
using isf_pass_plan_fn = std::function<std::vector<isf_pass_plan>(
    const std::string&                                     path,
    int                                                    render_w,
    int                                                    render_h,
    const std::function<bool(const std::string&, double&)>& value,
    std::string&                                           out_error)>;

void                      set_isf_pass_planner(isf_pass_plan_fn f);
const isf_pass_plan_fn&   get_isf_pass_planner();

/// Generates the above for a shader path. Null until the ISF module registers itself.
using isf_vulkan_source_fn = std::function<isf_vulkan_source(const std::string& path)>;

void                        set_isf_vulkan_source(isf_vulkan_source_fn f);
const isf_vulkan_source_fn& get_isf_vulkan_source();

/// Injected once at boot by `shell/server.cpp`, for the same reason `set_port_resolver` is: the
/// dependency runs module -> core, so core holds a hook and the module fills it.
void set_isf_node_renderer(isf_node_renderer r);

/// Null until the module registers itself, and the evaluator checks rather than assumes -- a
/// build with the ISF module disabled must still run a document that mentions an `isf` node.
const isf_node_renderer& get_isf_node_renderer();

}}} // namespace caspar::core::graph
