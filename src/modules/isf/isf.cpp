/*
 * Copyright (c) 2026 CasparCG Contributors
 *
 * This file is part of CasparCG (www.casparcg.com).
 *
 * CasparCG is free software: you can redistribute it and/or modify it under the terms of the GNU
 * General Public License as published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 */

#include "isf.h"
#include "isf_producer.h"
#include "isf_shader.h"

#include <algorithm>
#include "isf_vulkan_glsl.h"

#include <common/log.h>

#include <core/graph/isf_render.h>
#include <core/producer/frame_producer_registry.h>

#include <map>
#include <memory>

namespace caspar { namespace isf {

void init(const core::module_dependencies& dependencies)
{
    dependencies.producer_registry->register_producer_factory(L"ISF Producer", create_producer);
    CASPAR_LOG(info) << L"[isf] ISF shader producer registered ( [ISF] <shader-file> ).";
}

namespace {

/// One ISF input as a node port.
///
/// THE MAPPING IS THE INTERESTING PART, and every line of it is a decision about what a client
/// sees. An ISF `float` with MIN/MAX is a slider; one without is unbounded and must NOT be given
/// an invented range, because a client would draw a slider over a lie. A `long` is ISF's pop-up
/// menu and becomes an enumeration with its own labels. `event` is momentary: it is a boolean
/// here, because the graph has no trigger kind and a boolean a timeline can step is closer to
/// the intent than a number.
///
/// `bounding` is REFUSE for anything with a declared range, which matches every other port in
/// this registry: a value outside the range is a client error worth reporting, not something to
/// clamp silently.
core::graph::port_desc port_from_input(const input& in)
{
    namespace gr = core::graph;
    gr::port_desc p;
    p.param.name        = in.name;
    p.param.description = in.label.empty() ? in.name : in.label;
    p.param.access      = core::fields::access_t::read_write;
    p.direction         = gr::port_direction::input;
    p.flow              = gr::port_flow::signal;

    if (in.is_image) {
        // An image input is an IMAGE PORT, so the compiler routes a texture to it rather than a
        // value -- which is what makes a shader's second input connectable to another node.
        p.param.type = core::fields::value_type::blob;
        p.param.arity = 1;
        p.domain      = gr::port_domain::image;
        p.space       = gr::image_space::working;
        p.alpha       = gr::image_alpha::premultiplied;
        // NOT required: an ISF filter names its primary input `inputImage`, and the graph
        // already routes the node's `in` there. A second image left unconnected must render the
        // layer unchanged rather than black -- the rule every optional image port here follows.
        p.required             = false;
        p.disconnected_default = 1.0;
        return p;
    }

    p.domain = gr::port_domain::value;

    if (in.type == "bool" || in.type == "event") {
        p.param.type  = core::fields::value_type::boolean;
        p.param.arity = 1;
        p.param.default_value.push_back(!in.default_value.empty() && in.default_value.front() != 0.0);
        p.param.value = p.param.default_value;
        return p;
    }

    if (in.type == "long") {
        p.param.type  = core::fields::value_type::enumeration;
        p.param.arity = 1;
        std::string vals;
        for (std::size_t i = 0; i < in.labels.size(); ++i)
            vals += (i ? "," : "") + in.labels[i];
        p.param.values = vals;
        p.param.default_value.push_back(
            static_cast<std::int64_t>(in.default_value.empty() ? 0 : in.default_value.front()));
        p.param.value    = p.param.default_value;
        p.param.bounding = core::fields::bounding_t::refuse;
        return p;
    }

    const auto arity = static_cast<std::uint8_t>(in.type == "color"    ? 4
                                                 : in.type == "point2D" ? 2
                                                                        : 1);
    p.param.type  = arity == 4   ? core::fields::value_type::vec4
                    : arity == 2 ? core::fields::value_type::vec2
                                 : core::fields::value_type::real;
    p.param.arity = arity;
    for (std::uint8_t i = 0; i < arity; ++i)
        p.param.default_value.push_back(i < in.default_value.size() ? in.default_value[i]
                                        : in.default_value.empty()  ? 0.0
                                                                    : in.default_value.front());
    p.param.value = p.param.default_value;

    // A RANGE ONLY IF THE SHADER DECLARED ONE. Inventing 0..1 for an unbounded float would have
    // a client draw a slider that refuses the values the shader was written for.
    if (!in.min_value.empty() && !in.max_value.empty()) {
        p.param.min      = in.min_value.front();
        p.param.max      = in.max_value.front();
        p.param.bounding = core::fields::bounding_t::refuse;
    }
    return p;
}

} // namespace

std::vector<core::graph::port_desc> resolve_node_ports(const std::string& selector, std::string& out_reason)
{
    // ── WHAT A NODE CANNOT DO YET IS REFUSED HERE, AT PUT, ON BOTH BACKENDS ─────────────
    //
    // THROUGH THE PORT RESOLVER rather than through a new seam, because this already reaches
    // the validator: a resolver that sets `out_reason` produces ONE fault naming `path`, which
    // is exactly the message an editor wants -- the shader is unusable, rather than a dozen
    // parameters having gone missing.
    //
    // **AND ON BOTH BACKENDS, EVEN THOUGH OPENGL COULD RUN IT.** `isf::shader` has handled
    // PASSES, persistent buffers and IMPORTED images for years, so the OpenGL node path would
    // render a multi-pass shader correctly today. The Vulkan node path is a single generated
    // fragment shader and would silently render only the LAST pass -- a plausible picture, on
    // one backend, from a document that is valid on the other. The operator's rule for this
    // feature is that a document must not render differently depending on the mixer, so the
    // answer is to refuse it everywhere until the Vulkan pass loop exists, not to let OpenGL
    // race ahead.
    //
    // This costs nothing that ever worked: the `isf` NODE class is new, and multi-pass has
    // never rendered through it on either mixer. The PRODUCER is untouched and still runs all
    // of this.
    std::string feat_error;
    const auto  feats = describe_features(u16(selector), feat_error);
    if (feat_error.empty()) {
        // `multipass` IS NO LONGER REFUSED -- both backends draw N passes now. `persistent`
        // and `imported` still are, so nothing is ever half-rendered: a shader declaring either
        // is unusable as a node until the commit that implements it.
        const char* missing = feats.persistent ? "a PERSISTENT buffer"
                              : feats.imported ? "an IMPORTED image"
                                               : nullptr;
        if (missing) {
            out_reason = std::string("'") + selector + "' declares " + missing +
                         ", which an ISF NODE does not implement yet -- the ISF PRODUCER does, so "
                         "`[ISF] " + selector +
                         "` still plays it. Refused rather than half-rendered: the OpenGL node "
                         "path would run it and the Vulkan one would silently render only the "
                         "last pass, so the same document would look different on the two mixers";
            return {};
        }
    }

    // ── AND THE BINDING LIMIT, REFUSED ON BOTH BACKENDS FOR THE SAME REASON ─────────────
    //
    // A node's pass targets bind into descriptor set 1, which carries `OCIO_MAX_TEXTURES`
    // sampler bindings a variant pipeline may use as it likes. That is a real limit of the
    // pipeline layout rather than a policy -- and it is generous in practice: Vidvox's own
    // Gaussian blur needs six.
    //
    // **The OpenGL node path has NO such limit**, because `isf::shader` binds targets as plain
    // GL textures and would render a nine-target shader correctly. Refusing there too is the
    // whole point: a document that renders on one mixer and not the other is exactly the fault
    // this class is arranged to prevent, and the generator's own refusal would only be reached
    // on the Vulkan backend -- too late, and only for half the operators.
    {
        std::string pass_error;
        const auto  passes = describe_passes(u16(selector), pass_error);
        if (pass_error.empty()) {
            std::vector<std::string> targets;
            for (const auto& pi : passes) {
                if (pi.target.empty())
                    continue;
                if (std::find(targets.begin(), targets.end(), pi.target) == targets.end())
                    targets.push_back(pi.target);
            }
            if (static_cast<int>(targets.size()) > max_isf_targets) {
                out_reason = std::string("'") + selector + "' declares " +
                             std::to_string(targets.size()) + " PASSES targets and a node can " +
                             "sample " + std::to_string(max_isf_targets) +
                             " -- the mixer binds them into one descriptor set. The ISF PRODUCER "
                             "has no such limit, so `[ISF] " + selector + "` still plays it";
                return {};
            }
        }
    }

    // ── A SIBLING `.vs` IS REFUSED, AND THIS ONE IS NOT A PARITY FAULT ─────────────────
    //
    // Both node paths ignore a custom vertex shader today -- the OpenGL one constructs
    // `isf::shader` without a vertex source, and the Vulkan generator emits its own -- so the
    // two mixers AGREE. They agree on the wrong picture.
    //
    // A `.vs` is where the ISF specification's own primer puts per-vertex work: chapter 6
    // computes a convolution's eight neighbour coordinates as `vec2 d = 1.0/RENDERSIZE` in the
    // vertex stage and interpolates them. Drop it and the fragment shader reads varyings that
    // were never written -- a shader that compiles, runs, and renders something plausible and
    // wrong. Measured over Vidvox's collection: 38 of 327 shaders ship one.
    //
    // So the rule here is the rule everywhere else in this resolver -- **unchanged or refused,
    // never wrong** -- and the PRODUCER, which honours `.vs` and has for years, still plays it.
    {
        std::string  vs_source;
        std::wstring vs_base;
        std::string  vs_error;
        bool         has_vs = false;
        if (load_shader_source(u16(selector), vs_source, vs_base, vs_error, &has_vs) && has_vs) {
            out_reason = std::string("'") + selector +
                         "' ships a sibling `.vs`, which an ISF NODE does not run yet. Its "
                         "fragment shader would read varyings nothing wrote -- a plausible wrong "
                         "picture rather than a failure. The ISF PRODUCER honours it, so `[ISF] " +
                         selector + "` still plays it";
            return {};
        }
    }

    const auto inputs = describe_inputs(u16(selector), out_reason);
    if (!out_reason.empty())
        return {};

    std::vector<core::graph::port_desc> out;
    out.reserve(inputs.size());
    for (const auto& in : inputs) {
        // `inputImage` IS the node's `in`, which the class already declares. Emitting it again
        // would give the node two ports for one thing and let a document connect both.
        if (in.name == "inputImage")
            continue;
        out.push_back(port_from_input(in));
    }

    // A shader with no INPUTS at all is legal -- a plain GLSL fragment is playable -- and it
    // still has the class's static ports. Returning an empty list here would read as failure to
    // `instance_ports`, so say so with one port that is always true.
    if (out.empty() && out_reason.empty())
        out.push_back(port_from_input(input{"_isf_no_inputs", "bool", {0.0}, {}, {}, "this shader declares no inputs", false}));
    return out;
}

namespace {

/// One `isf` node's GL state, living in the slot the mixer keeps for that node.
///
/// ── PER NODE INSTANCE, WHICH THE PATH-KEYED CACHE THIS REPLACES COULD NOT BE ────────────────
///
/// The map that used to be here was a process-wide `static` keyed by shader PATH. Its own comment
/// said what was wrong with it: *"NOT SAFE FOR PERSISTENT BUFFERS, which are the ISF spec's
/// per-SHADER-INSTANCE state ... two nodes sharing this entry would accumulate into one buffer and
/// each would see the other's history ... this cache becomes per node instance in the same commit
/// that lands it."* This is that commit.
///
/// It is also what every other host does. ISF: a persistent buffer *"stays with your effect until
/// its deletion"*. TouchDesigner scopes feedback to the operator, Smode to the modifier, vvvv to
/// the process node, OpenFX to `kOfxPropInstanceData`, *"unique to each plug-in instance"*.
///
/// **A SHADER IS STILL COMPILED PER INSTANCE, not shared per path.** The producer already builds
/// one `isf::shader` per producer, and splitting the compiled program from the per-instance
/// buffers means threading a shared-program concept through a 1795-line class for a cost nobody
/// has measured. `grade-graph-cost`'s ISF arms -- sixteen nodes on ONE file -- are what would show
/// it, and they read 0 late on OpenGL. If that changes, the split is the fix and this is the note
/// that says so.
struct node_state final : public core::graph::isf_node_state
{
    std::unique_ptr<shader> sh;
    /// Tried and failed: a missing file or a shader that would not compile. Remembered so the
    /// cost is one attempt rather than one per frame, and one log line rather than fifty a
    /// second -- the node renders its input through either way.
    bool failed = false;

    ~node_state() override
    {
        // THE MIXER DESTROYS THIS STORE INSIDE A `dispatch_sync` ON ITS RENDER THREAD, which is
        // the only place the right GL context is current. `~shader` would otherwise free nothing
        // at all here: it releases through the device it was constructed with, and a node's
        // shader is built without one.
        if (sh)
            sh->release_gl_on_current_context();
    }
};

/// Draw one ISF node on the mixer's own GL context.
bool render_isf_node(const core::graph::isf_node_request& req)
{
    if (req.path.empty() || req.src_tex == 0 || req.dst_tex == 0 || req.width <= 0 || req.height <= 0)
        return false;

    if (!req.state)
        return false;

    // THE SLOT IS THE MIXER'S, keyed per node instance; what is in it is this module's. Created
    // on the node's first drawn frame and destroyed when the node leaves the attached document.
    if (!*req.state)
        *req.state = std::make_shared<node_state>();
    auto& st = static_cast<node_state&>(**req.state);

    if (st.failed)
        return false;

    if (!st.sh) {
        std::string  source;
        std::wstring base;
        std::string  error;
        if (!load_shader_source(u16(req.path), source, base, error)) {
            // ONCE, NOT EVERY FRAME. A missing shader at 50 fps is 50 identical log lines a
            // second, which buries everything else in the log -- and the node is already
            // rendering its input unchanged, so the picture is not the thing in doubt.
            CASPAR_LOG(warning) << L"[isf] node shader: " << u16(error)
                                << L" -- the node passes its input through";
            st.failed = true;
            return false;
        }
        try {
            st.sh = std::make_unique<shader>(source, base);
        } catch (const std::exception& e) {
            CASPAR_LOG(warning) << L"[isf] node shader '" << u16(req.path) << L"' would not compile: "
                                << u16(e.what()) << L" -- the node passes its input through";
            st.failed = true;
            return false;
        }
    }

    auto& sh = *st.sh;

    // THE `reset` PORT, applied before anything is drawn: every persistent buffer back to black,
    // for as long as the port is held. `FRAMEINDEX` is zeroed by the evaluator, which owns the
    // count -- see `isf_node_request::frame_index`.
    if (req.reset)
        sh.reset_persistent_buffers();

    // ── THE NODE'S VALUES ONTO THE SHADER'S INPUTS, BY NAME ─────────────────────────────────
    //
    // `names[k]` names `values[k]`, and an arity > 1 port repeats its name across its
    // components -- so a run of equal names IS one input's value list. Walking runs rather than
    // indexing by position is what makes this immune to either side inserting a port: a name
    // that no longer exists is skipped by `set_value`, which is a parameter that does nothing
    // rather than a parameter on the wrong input.
    for (std::uint32_t k = 0; k < req.value_count;) {
        std::uint32_t n = 1;
        while (k + n < req.value_count && req.names[k + n] == req.names[k])
            ++n;
        // The class's OWN ports -- `bypass`, `mix`, `space` -- are the evaluator's to act on and
        // are not shader inputs. `set_value` returns false for them, which is exactly right and
        // is why this does not check.
        sh.set_value(req.names[k], std::vector<double>(req.values + k, req.values + k + n));
        k += n;
    }

    // fp16 attachments all the way down the node chain, so the final pass must not quantise to
    // 8 bits on the way out -- which is what an ISF ramp measured before `set_output_depth`
    // existed.
    sh.set_output_depth(common::bit_depth::bit16);
    // fp16 for every pass buffer and the final target: the node graph's intermediates are fp16
    // everywhere, and the Vulkan node path has no other option. See `set_node_buffer_format`.
    sh.set_node_buffer_format(true);

    // The `space` port, resolved against the graph's stage by the evaluator. 0 for every
    // agreeing combination, which leaves the shader bit-identical to one built before this.
    sh.set_space_conversion(req.to_display);

    image_binding in;
    in.name   = "inputImage";
    in.tex_id = req.src_tex;
    in.width  = req.width;
    in.height = req.height;
    // THE MIXER'S CONVENTION -- and the colour half of it is the OPPOSITE of what it looks
    // like, which cost a measurement to establish.
    //
    // `flip` is TRUE: the mixer's attachments are top-down and GL is bottom-up.
    //
    // `bgra` is FALSE, and reasoning said true. The OpenGL mixer carries the pixel in BGR order
    // through the grading chain, so a bgra-labelled PRODUCER plane really does hold BGRA bytes
    // -- but a node-graph ATTACHMENT is written by the kernel's head pass as `col.bgra`, from
    // that already-BGR `col`, so the bytes in an attachment are RGBA. `apply_node` swizzles
    // back on read, which is the other half of the same convention.
    //
    // Measured: with both this and the output swizzle wrongly on, the two exchanges did not
    // cancel -- they compose into a picture where green is correct and red and blue are
    // exchanged. That is why the fixtures are asymmetric; a grey one passes this silently.
    in.flip = true;
    in.bgra = false;

    const bool ok = sh.render_into_current(req.width,
                                           req.height,
                                           req.time,
                                           req.time_delta,
                                           req.frame_index,
                                           {in},
                                           req.dst_tex,
                                           /*finish*/ false,
                                           // RGBA, because the destination is an ATTACHMENT and
                                           // not a producer's frame. See `in.bgra` above.
                                           /*swap_rb*/ false);
    // ── ISF `event` INPUTS ARE LEVEL HERE, ON BOTH BACKENDS ─────────────────────────────
    //
    // `reset_events()` used to run here, clearing every `event` input after each draw -- which
    // made an event fire exactly once on OpenGL while the Vulkan path, which copies the
    // document's value into its uniform block every frame, kept it set until the client wrote 0.
    // **The same document behaved differently on the two mixers**, which is the one thing this
    // feature's rules forbid.
    //
    // LEVEL on both is the answer that needs no per-instance bookkeeping of a shader input and
    // no edge detection that the two backends could disagree about: an event is true while the
    // document says true, and the client pulses it -- write 1, write 0 on the next tick, exactly
    // as a timeline key or the `reset` port does. It is also what makes `reset` and an `event`
    // input behave identically, which is one rule for a client to learn instead of two.
    //
    // The PRODUCER still calls `reset_events()` on its own path and is untouched.
    return ok;
}

} // namespace

bool render_node(const core::graph::isf_node_request& req) { return render_isf_node(req); }

std::vector<core::graph::isf_pass_plan>
plan_passes_for(const std::string&                                     path,
                int                                                    render_w,
                int                                                    render_h,
                const std::function<bool(const std::string&, double&)>& value,
                std::string&                                           out_error)
{
    std::vector<core::graph::isf_pass_plan> out;
    const auto                              passes = describe_passes(u16(path), out_error);
    if (!out_error.empty())
        return out;

    for (const auto& pi : passes) {
        core::graph::isf_pass_plan pp;
        pp.target     = pi.target;
        pp.persistent = pi.persistent;
        // ONCE PER FRAME, per the spec, and per pass only because each pass has its own
        // expressions -- the VALUES they read do not change between passes of one frame.
        pp.width  = eval_pass_size(pi.w_expr, render_w, render_w, render_h, value);
        pp.height = eval_pass_size(pi.h_expr, render_h, render_w, render_h, value);
        out.push_back(std::move(pp));
    }
    return out;
}

core::graph::isf_vulkan_source vulkan_source_for(const std::string& path)
{
    const auto r = build_vulkan_fragment_for(path);
    core::graph::isf_vulkan_source out;
    out.source      = r.source;
    out.cache_id    = r.cache_id;
    out.value_count = r.value_count;
    out.error       = r.error;
    return out;
}

}} // namespace caspar::isf
