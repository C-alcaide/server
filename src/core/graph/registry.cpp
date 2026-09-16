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

#include <utility>
#include <mutex>
#include <map>
#include "registry.h"

#include "isf_render.h"

#include <common/except.h>
#include <common/log.h>

#include <algorithm>
#include <set>

namespace caspar { namespace core { namespace graph {

namespace {

// ---------------------------------------------------------------------------------------
// Port builders
//
// Small and repetitive on purpose: each class's port list should read as a declaration of what
// the class IS, not as a paragraph of struct initialisation. Every default that matters is set
// here so a class author cannot forget one.
// ---------------------------------------------------------------------------------------

port_desc value_in(const char* name, double def, double lo, double hi, const char* unit,
                   const char* description, std::uint8_t arity = 1)
{
    port_desc p;
    p.param.name        = name;
    p.param.type        = arity == 1   ? fields::value_type::real
                          : arity == 2 ? fields::value_type::vec2
                          : arity == 3 ? fields::value_type::vec3
                                       : fields::value_type::vec4;
    p.param.access      = fields::access_t::read_write;
    p.param.min         = lo;
    p.param.max         = hi;
    p.param.bounding    = fields::bounding_t::refuse;
    p.param.unit        = unit;
    p.param.description = description;
    p.param.arity       = arity;
    for (std::uint8_t i = 0; i < arity; ++i)
        p.param.default_value.push_back(def);
    p.param.value  = p.param.default_value;
    p.direction    = port_direction::input;
    p.flow         = port_flow::signal;
    p.domain       = port_domain::value;
    return p;
}

port_desc value_in3(const char* name, double def, double lo, double hi, const char* unit,
                    const char* description)
{
    return value_in(name, def, lo, hi, unit, description, 3);
}

port_desc bool_in(const char* name, bool def, const char* description)
{
    port_desc p;
    p.param.name        = name;
    p.param.type        = fields::value_type::boolean;
    p.param.access      = fields::access_t::read_write;
    p.param.bounding    = fields::bounding_t::refuse;
    p.param.description = description;
    p.param.arity       = 1;
    p.param.default_value.push_back(def);
    p.param.value = p.param.default_value;
    p.direction   = port_direction::input;
    // DISCRETE, so `animatable` comes out as `"step"` and a timeline switches AT a key rather
    // than sliding through 0.5 of a boolean. Derived by `fields::animatable_of` from this.
    p.flow   = port_flow::signal;
    p.domain = port_domain::value;
    return p;
}

port_desc enum_in(const char* name, const char* values, int def, const char* description,
                  port_flow flow)
{
    port_desc p;
    p.param.name        = name;
    p.param.type        = fields::value_type::enumeration;
    p.param.access      = fields::access_t::read_write;
    p.param.bounding    = fields::bounding_t::refuse;
    p.param.values      = values;
    p.param.description = description;
    p.param.arity       = 1;
    p.param.default_value.push_back(static_cast<std::int64_t>(def));
    p.param.value = p.param.default_value;
    p.direction   = port_direction::input;
    p.flow        = flow;
    p.domain      = port_domain::value;
    return p;
}

port_desc image_port(const char* name, port_direction dir, bool required = false)
{
    port_desc p;
    p.param.name        = name;
    p.param.type        = fields::value_type::blob;
    p.param.access      = fields::access_t::read;
    p.param.arity       = 1;
    p.param.description = dir == port_direction::input ? "an image" : "the image this node makes";
    p.direction         = dir;
    p.flow              = port_flow::signal;
    p.domain            = port_domain::image;
    // WORKING SPACE ON EVERY IMAGE PORT IN v1, including on a `display`-stage graph's ports. The
    // stage is a property of the DOCUMENT, not of a port: a display-stage graph's head pass hands
    // it display-encoded pixels through the same ports. Tagging the port `display` per document
    // would make the registry depend on a document, which is backwards. `space_of` on the plan is
    // where the stage is read; the self-test asserts no port declares `display`.
    p.space             = image_space::working;
    p.alpha             = image_alpha::premultiplied;
    p.required          = required;
    return p;
}

port_desc mask_port(const char* name, port_direction dir, bool required = false)
{
    port_desc p;
    p.param.name        = name;
    p.param.type        = fields::value_type::blob;
    p.param.access      = fields::access_t::read;
    p.param.arity       = 1;
    p.param.description = "a single-component mask, 0..1";
    p.direction         = dir;
    p.flow              = port_flow::signal;
    p.domain            = port_domain::mask;
    p.required          = required;
    // 1.0 -- an unconnected mask means "everywhere", so a node with no mask grades the whole
    // image. The alternative, 0.0, would make a node with a forgotten mask a silent no-op.
    p.disconnected_default = 1.0;
    return p;
}

/// The `mix` amount that every grading class carries, so a node can be dialled back without a
/// `mix` node behind it. `col = mix(col, graded, mask * mix)`.
port_desc mix_amount()
{
    return value_in("mix", 1.0, 0.0, 1.0, "",
                    "how much of this node's result is used, multiplied by its mask");
}

/// A grading class: one image in, an optional mask, one image out, plus `mix`.
node_class grade_class(const char* id, const char* label, const char* description,
                       std::vector<port_desc> params)
{
    node_class c;
    c.id             = id;
    c.label          = label;
    c.group          = "grade";
    c.description    = description;
    c.preview        = true;
    c.produces_image = true;
    c.ports.push_back(image_port("in", port_direction::input, /*required*/ true));
    c.ports.push_back(mask_port("mask", port_direction::input));
    for (auto& p : params)
        c.ports.push_back(std::move(p));
    c.ports.push_back(mix_amount());
    c.ports.push_back(image_port("out", port_direction::output));
    return c;
}

std::vector<node_class> build_classes()
{
    std::vector<node_class> cs;

    // ---- roots ------------------------------------------------------------------------
    //
    // IMPLICIT AND EXACTLY ONE EACH, and the validator enforces that. `input` is the layer as the
    // head pass produced it; `output` is what the tail pass draws. They are nodes rather than
    // magic edges so that a client draws them, an author can see where the picture enters and
    // leaves, and `last_use` refcounting has a first and last step with no special cases.
    {
        node_class c;
        c.id             = "input";
        c.label          = "Input";
        c.group          = "root";
        c.description    = "the layer's picture, as it arrives at the graph";
        c.produces_image = true;
        c.preview        = true;
        c.ports.push_back(image_port("out", port_direction::output));
        cs.push_back(std::move(c));
    }
    {
        node_class c;
        c.id          = "output";
        c.label       = "Output";
        c.group       = "root";
        c.description = "what the layer draws. Exactly one, and it must be reachable";
        // NOT preview-able: its output IS the layer, so a client asking for a picture of it
        // should capture the channel.
        c.preview        = false;
        c.produces_image = false;
        c.ports.push_back(image_port("in", port_direction::input, /*required*/ true));
        cs.push_back(std::move(c));
    }

    // ---- grade ------------------------------------------------------------------------
    cs.push_back(grade_class(
        "exposure", "Exposure", "a stop of light -- a uniform scale on all three channels",
        {value_in("gain", 1.0, 0.0, 64.0, "x",
                  "linear multiplier. 2.0 is one stop up in working space")}));

    cs.push_back(grade_class(
        "cdl", "ASC CDL",
        "the published ASC CDL: pow(max(c * slope + offset, 0), power), then a mix toward luma",
        {value_in3("slope", 1.0, 0.0, 10.0, "", "per-channel multiplier"),
         value_in3("offset", 0.0, -1.0, 1.0, "", "per-channel offset, added after slope"),
         value_in3("power", 1.0, 0.01, 10.0, "", "per-channel gamma, applied last"),
         value_in("saturation", 1.0, 0.0, 10.0, "",
                  "0 is neutral. Mixes toward the working-space luma")}));

    // ---- mask -------------------------------------------------------------------------
    //
    // Only the ELLIPSE in v1, because it is the one the prototype already computes and the one
    // commit 4's evaluator can fuse into its consumer's uniforms with no extra pass. The rect,
    // gradient, qualifier and combine classes arrive with the shader that implements them --
    // declaring them here first would put classes in the catalogue that a PUT accepts and the
    // renderer ignores, which is the failure the timeline's path validation just closed.
    {
        node_class c;
        c.id          = "mask_ellipse";
        c.label       = "Ellipse Mask";
        c.group       = "mask";
        c.description = "a soft-edged ellipse. 1 inside, 0 outside, feathered isotropically";
        c.preview     = true;
        // Produces a MASK, not an image, so it does not cost an image pass when it is fused.
        c.produces_image = false;
        c.ports.push_back(value_in("center", 0.5, 0.0, 1.0, "",
                                   "centre, in the space `space` names", 2));
        c.ports.push_back(value_in("radius", 0.25, 0.0, 1.0, "",
                                   "the two radii, in the space `space` names", 2));
        c.ports.push_back(value_in("feather", 0.2, 0.0, 1.0, "",
                                   "edge softness as a FRACTION of the radius, so it is "
                                   "isotropic on any raster"));
        c.ports.push_back(bool_in("invert", false, "1 selects everything outside the ellipse"));
        // AN ATTRIBUTE, not a signal: the frame-space and source-space forms need different
        // uniforms in the pass (source space needs the layer's inverse affine), so changing it
        // changes the compiled plan.
        c.ports.push_back(enum_in("space", "frame,source", 0,
                                  "`frame` is the raster; `source` follows the layer's own "
                                  "geometry, so the mask moves with the picture",
                                  port_flow::attribute));
        c.ports.push_back(mask_port("out", port_direction::output));
        cs.push_back(std::move(c));
    }

    // ---- combine ----------------------------------------------------------------------
    //
    // The two fan-in classes, and they are what makes this a GRAPH rather than a chain. `mix`
    // needs two image inputs live at once, which the prototype's ping-pong cannot express at all.
    {
        node_class c;
        c.id          = "mix";
        c.label       = "Mix";
        c.group       = "combine";
        c.description = "mix(a, b, amount). A `switch` is this with a stepped 0/1 amount";
        c.preview     = true;
        c.produces_image = true;
        c.ports.push_back(image_port("a", port_direction::input, /*required*/ true));
        // B IS NOT REQUIRED, and its disconnected default returns `a`. A muted edge into `b`
        // must leave the picture alone rather than mixing toward black -- the one failure mode
        // that is unacceptable during a show.
        c.ports.push_back(image_port("b", port_direction::input));
        c.ports.push_back(value_in("amount", 0.5, 0.0, 1.0, "",
                                   "0 is all `a`, 1 is all `b`"));
        c.ports.push_back(mask_port("mask", port_direction::input));
        c.ports.push_back(image_port("out", port_direction::output));
        cs.push_back(std::move(c));
    }
    {
        node_class c;
        c.id          = "over";
        c.label       = "Over";
        c.group       = "combine";
        c.description = "premultiplied source-over: a + (1 - a.alpha) * b";
        c.preview     = true;
        c.produces_image = true;
        c.ports.push_back(image_port("a", port_direction::input, /*required*/ true));
        c.ports.push_back(image_port("b", port_direction::input));
        c.ports.push_back(image_port("out", port_direction::output));
        cs.push_back(std::move(c));
    }


    // ---- the rest of the mask families --------------------------------------------------
    //
    // APPENDED after `over` rather than filed beside `mask_ellipse`, because the op constants
    // are indices into this table and inserting here would renumber `mix` and `over` -- which
    // two kernels and two shaders hard-code. See the note in registry.h.
    //
    // All three GENERATORS share `mask_ellipse`'s parameter layout up to `invert`, and that is
    // deliberate rather than convenient: the kernels upload one set of mask uniforms and the
    // shaders evaluate one shape per `gn_op`, so a new generator costs a shader case and no new
    // plumbing. The layout is `bypass, center.0, center.1, radius.0, radius.1, feather, invert,
    // space` -- `radius` reads as the half-extent for a rect and as the gradient's run.
    {
        node_class c;
        c.id          = "mask_rect";
        c.label       = "Rectangle Mask";
        c.group       = "mask";
        c.description = "a soft-edged rectangle. 1 inside, 0 outside, feathered on both axes";
        c.preview     = true;
        c.produces_image = false;
        c.ports.push_back(value_in("center", 0.5, 0.0, 1.0, "",
                                   "centre, in the space `space` names", 2));
        c.ports.push_back(value_in("radius", 0.25, 0.0, 1.0, "",
                                   "HALF-extent on each axis, so the rect spans 2x this", 2));
        c.ports.push_back(value_in("feather", 0.2, 0.0, 1.0, "",
                                   "edge softness as a FRACTION of the half-extent, so it is "
                                   "the same proportion on both axes"));
        c.ports.push_back(bool_in("invert", false, "1 selects everything outside the rect"));
        c.ports.push_back(enum_in("space", "frame,source", 0,
                                  "`frame` is the raster; `source` follows the layer's own "
                                  "geometry, so the mask moves with the picture",
                                  port_flow::attribute));
        c.ports.push_back(mask_port("out", port_direction::output));
        cs.push_back(std::move(c));
    }
    {
        node_class c;
        c.id          = "mask_gradient";
        c.label       = "Gradient Mask";
        c.group       = "mask";
        c.description = "a linear ramp from 0 to 1 across `radius`, at `angle`";
        c.preview     = true;
        c.produces_image = false;
        c.ports.push_back(value_in("center", 0.5, 0.0, 1.0, "",
                                   "where the ramp reaches 0.5, in the space `space` names", 2));
        // `radius.0` IS THE RUN and `radius.1` is unused, so the layout stays identical to the
        // other generators. Stated because a reader will otherwise wonder which axis applies:
        // the ramp is one-dimensional and `angle` is what orients it.
        c.ports.push_back(value_in("radius", 0.25, 0.0, 1.0, "",
                                   "the RUN: half the 0-to-1 distance. Only the first component "
                                   "is read; the second keeps the layout shared with the other "
                                   "generators", 2));
        c.ports.push_back(value_in("feather", 0.2, 0.0, 1.0, "",
                                   "eases the two ends of the ramp. 0 is a hard linear ramp"));
        c.ports.push_back(bool_in("invert", false, "1 runs the ramp the other way"));
        c.ports.push_back(enum_in("space", "frame,source", 0,
                                  "`frame` is the raster; `source` follows the layer's own "
                                  "geometry",
                                  port_flow::attribute));
        c.ports.push_back(value_in("angle", 0.0, -6.2831853071795865, 6.2831853071795865, "rad",
                                   "0 runs the ramp along +x. RADIANS, like every other angle in "
                                   "the registry -- the address space writes registry units and "
                                   "the degrees-to-radians conversion the old keyframe path did "
                                   "is exactly what the address grammar removed"));
        c.ports.push_back(mask_port("out", port_direction::output));
        cs.push_back(std::move(c));
    }
    {
        node_class c;
        c.id          = "mask_qualifier";
        c.label       = "Qualifier Mask";
        c.group       = "mask";
        c.description = "keys on hue, saturation and luma -- the selection half of "
                        "`MIXER QUALIFIER`, as a mask other nodes can share";
        c.preview     = true;
        // TAKES AN IMAGE, which makes it the only mask generator that is not analytic: it reads
        // the pixel it is keying. So it is never fusable into a consumer's uniforms -- a fused
        // mask is evaluated from uniforms alone -- and `compile()` materialises it because its
        // `in` edge gives it a fan-in the analytic generators do not have.
        c.produces_image = false;
        c.ports.push_back(image_port("in", port_direction::input, /*required*/ true));
        c.ports.push_back(value_in("hue", 0.0, 0.0, 360.0, "deg",
                                   "centre of the keyed hue band"));
        c.ports.push_back(value_in("hue_width", 60.0, 0.0, 360.0, "deg",
                                   "full width of the band, so +/- half this from `hue`"));
        c.ports.push_back(value_in("sat_low", 0.1, 0.0, 1.0, "",
                                   "saturation below this is not keyed"));
        c.ports.push_back(value_in("sat_high", 1.0, 0.0, 1.0, "",
                                   "saturation above this is not keyed"));
        c.ports.push_back(value_in("luma_low", 0.0, 0.0, 1.0, "",
                                   "luma below this is not keyed"));
        c.ports.push_back(value_in("luma_high", 1.0, 0.0, 1.0, "",
                                   "luma above this is not keyed"));
        c.ports.push_back(value_in("softness", 0.1, 0.0, 1.0, "",
                                   "eases every one of the three bands' edges"));
        c.ports.push_back(bool_in("invert", false, "1 selects everything NOT keyed"));
        c.ports.push_back(mask_port("out", port_direction::output));
        cs.push_back(std::move(c));
    }
    {
        node_class c;
        c.id          = "mask_combine";
        c.label       = "Combine Masks";
        c.group       = "mask";
        c.description = "union, intersect or subtract two masks";
        c.preview     = true;
        c.produces_image = false;
        // TWO MASK INPUTS, so this class can NEVER be fused whatever its fan-out: a fused mask
        // is one a consumer evaluates from its own uniforms, and this one has to read two
        // textures. `graph_plan_self_test` asserts that, because "fused" here would mean
        // silently ignoring both inputs -- the same shape as the defect commit 8a fixed.
        c.ports.push_back(mask_port("a", port_direction::input, /*required*/ true));
        c.ports.push_back(mask_port("b", port_direction::input, /*required*/ true));
        c.ports.push_back(enum_in("op", "union,intersect,subtract", 0,
                                  "`union` is max(a, b), `intersect` is a*b, `subtract` is "
                                  "a*(1-b). Multiplicative rather than min/max for intersect "
                                  "and subtract so a FEATHERED edge stays feathered -- min "
                                  "would give the harder of the two edges",
                                  port_flow::attribute));
        c.ports.push_back(bool_in("invert", false, "1 inverts the result, after `op`"));
        c.ports.push_back(mask_port("out", port_direction::output));
        cs.push_back(std::move(c));
    }

    // ---- an ISF shader, whose ports come from the FILE --------------------------------
    //
    // THE FIRST CLASS WHOSE PORTS ARE NOT KNOWN HERE. `path` selects a `.fs` under the media
    // folder and the shader's own INPUTS become the rest of this node's ports, resolved per
    // instance by the resolver the shell injects -- see `dynamic_ports`.
    //
    // AND IT IS AN ORDINARY NODE, placeable anywhere in a chain. That is worth stating because
    // the first design restricted it to the head: an ISF shader is GLSL plus a JSON header, so
    // it is a SHADER PASS, which is exactly what this evaluator already draws. It needs no
    // pre-pass, no interop and no pass splitting. A foreign RENDERER -- an OFX plug-in with its
    // own GL or CUDA context -- is the thing that cannot live mid-pass, and conflating the two
    // nearly cost this feature the workflow that matters: every peer host (Resolume, Natron,
    // TouchDesigner) lets an effect sit anywhere, and reordering a stack is the creative act.
    {
        node_class c;
        c.id             = "isf";
        c.label          = "ISF Shader";
        c.group          = "effect";
        c.description    = "an Interactive Shader Format shader, from the media folder. "
                           "With no `path` chosen it passes its input through, so a node can be "
                           "placed before a shader is picked";
        c.preview        = true;
        c.produces_image = true;
        c.dynamic_ports  = true;
        c.ports_selector = "path";

        // THE STATIC PART: what every instance has whatever the file says. A shader declaring
        // its own `path` input cannot shadow this one -- `instance_ports` keeps the static
        // list first for exactly that reason.
        {
            port_desc path;
            path.param.name        = "path";
            path.param.type        = fields::value_type::string;
            path.param.access      = fields::access_t::read_write;
            path.param.arity       = 1;
            path.param.description = "the shader file, relative to the media folder";
            // AN EMPTY STRING, not nothing. `node_registry_self_test` refuses to boot a port
            // whose default component count does not match its arity -- and it caught this
            // exact omission, which is what a boot self-test is for: the server would not
            // start rather than serve a malformed port to a catalogue.
            //
            // Empty is also the right VALUE: a node with no path yet is a node whose ports are
            // not known, which `instance_ports` reports as "needs 'path' before its ports are
            // known" rather than as a missing shader.
            path.param.default_value.push_back(std::string{});
            path.param.value       = path.param.default_value;
            path.direction         = port_direction::input;
            path.flow              = port_flow::signal;
            path.domain            = port_domain::value;
            c.ports.push_back(std::move(path));
        }
        c.ports.push_back(image_port("in", port_direction::input, /*required*/ true));
        c.ports.push_back(mask_port("mask", port_direction::input));

        // WHICH COLOUR SPACE THE SHADER RUNS IN, and `display` is the default deliberately.
        //
        // A node pass carries SCENE-LINEAR fp16 in the working gamut, unbounded above 1.0.
        // Every ISF shader published anywhere -- Vidvox's collection, anything written for
        // Resolume or VDMX -- was authored against a DISPLAY-REFERRED 0..1 buffer. A blur
        // survives the difference; a threshold, a hue rotate, a `pow` or any hand-tuned
        // constant does not, and the result renders, looks plausible and is wrong with nothing
        // to report it.
        //
        // So `display` converts to display-referred, runs the shader, and converts back -- two
        // extra conversions for a picture that matches what its author saw. `working` runs it
        // raw and free, for a shader written for this server.
        //
        // THE DEFAULT IS THE COMPATIBLE ONE because the alternative default is the one that
        // fails SILENTLY, and because the population of shaders that exists today was written
        // for display-referred values. The same shape as `mask_ellipse`'s `space` and the
        // document's `stage`, so a client already reads this pattern.
        // ── AND `match` IS THE DEFAULT, WHICH THE FIRST VERSION GOT WRONG ───────────────
        //
        // The port started as `display,working` defaulting to `display`, on the reasoning that
        // every published ISF shader was authored against display-referred values. That is true
        // and it made the class default UN-PUT-ABLE: the two crossed combinations of `space`
        // and the graph's `stage` need a colour conversion that is not built and are refused,
        // and `display` in a `working`-stage graph is one of them. A client fetching
        // `/v1/catalog/node/isf/default` and PUTting it got a fault.
        //
        // `api-graph` caught it on the first run after the refusal landed -- it walks every
        // class's own default and asserts it validates, which is exactly the check for "the
        // catalogue offers something the PUT refuses". A class whose default is refused is
        // broken however good the reason.
        //
        // `match` needs no conversion by construction, so it is always valid: the shader runs in
        // whatever encoding the pass already carries. For a `display`-stage graph -- the normal
        // case, and what a published shader wants -- that IS display-referred, so the default
        // still does the compatible thing without a special case.
        c.ports.push_back(enum_in("space", "match,display,working", 0,
                                  "`match` runs the shader in whatever encoding the graph's "
                                  "`stage` already carries, which needs no conversion and is "
                                  "always valid. `display` and `working` ask for a specific "
                                  "encoding, and are refused when that would need a conversion "
                                  "around the shader -- which is not implemented yet",
                                  port_flow::attribute));

        // ── `reset`: RE-SEED THIS NODE'S STATE ───────────────────────────────────────────
        //
        // A LEVEL, not an edge, and that is TouchDesigner's shape rather than an invention: its
        // Feedback and Cache TOPs ship a latching `Reset` ("while On this will empty the cache")
        // AND a one-frame `Reset Pulse`. One level port is both -- hold it for the first, write
        // 1 then 0 for the second -- and it needs no edge detection, so the two mixers cannot
        // disagree about when a pulse was seen.
        //
        // While it is held: `FRAMEINDEX` reads 0 and every persistent buffer is re-blackened
        // each frame. That makes the ISF idiom every published feedback shader uses --
        // `if (FRAMEINDEX < 1 || resetEvent) { seed }` -- work through this port as well as
        // through a shader's own `event` input.
        //
        // A BOOLEAN SIGNAL, not `port_flow::event`: that flow is declared and refused in v1.
        {
            port_desc r;
            r.param.name        = "reset";
            r.param.type        = fields::value_type::boolean;
            r.param.access      = fields::access_t::read_write;
            r.param.arity       = 1;
            r.param.description = "while true, re-seed this node: FRAMEINDEX back to 0 and every "
                                  "persistent buffer cleared. Hold it to latch, pulse it (1 then "
                                  "0) for a one-frame reset";
            r.param.default_value.push_back(false);
            r.param.value       = r.param.default_value;
            r.direction         = port_direction::input;
            r.flow              = port_flow::attribute;
            r.domain            = port_domain::value;
            c.ports.push_back(std::move(r));
        }

        c.ports.push_back(mix_amount());
        c.ports.push_back(image_port("out", port_direction::output));
        cs.push_back(std::move(c));
    }

    // ---- the implicit `bypass`, added HERE so no class author can forget it -------------
    //
    // A bypassed node aliases its primary input and costs no draw, so a graph with a bypassed
    // node is byte-identical to the graph without it -- which is a gate in `grade-graph` rather
    // than a claim. Boolean and therefore a STEP target.
    for (auto& c : cs) {
        if (c.id == "input" || c.id == "output")
            continue; // the roots are not bypassable: there is nothing to alias
        c.ports.insert(c.ports.begin(),
                       bool_in("bypass", false,
                               "pass the primary input through untouched, with no draw"));
    }

    return cs;
}

} // namespace

const std::vector<node_class>& node_classes()
{
    static const std::vector<node_class> table = build_classes();
    return table;
}

namespace {

/// The injected resolvers, and the cache of what they answered.
///
/// A MUTEX RATHER THAN A THREAD-LOCAL OR AN ATOMIC: registration happens once at boot on one
/// thread, and lookup happens on the API executor (a PUT) and on the stage (a compile). Those
/// are different threads, and the cache is written on first use rather than at registration, so
/// the write is on the read path and needs guarding.
std::mutex&                                        resolver_lock()
{
    static std::mutex m;
    return m;
}
std::map<std::string, port_resolver>&              resolvers()
{
    static std::map<std::string, port_resolver> r;
    return r;
}
std::map<std::pair<std::string, std::string>, std::vector<port_desc>>& port_cache()
{
    static std::map<std::pair<std::string, std::string>, std::vector<port_desc>> c;
    return c;
}

} // namespace

namespace {
/// The ISF node renderer, injected at boot. A plain global for the same reason the port
/// resolvers are: it is written once before any channel ticks and read from the render thread
/// thereafter, so there is nothing to synchronise.
isf_node_renderer& isf_renderer_slot()
{
    static isf_node_renderer r;
    return r;
}
} // namespace

void set_isf_node_renderer(isf_node_renderer r) { isf_renderer_slot() = std::move(r); }

namespace {
isf_vulkan_source_fn& isf_vulkan_slot()
{
    static isf_vulkan_source_fn f;
    return f;
}
} // namespace

namespace {
isf_pass_plan_fn& isf_planner_slot()
{
    static isf_pass_plan_fn f;
    return f;
}
} // namespace

void set_isf_pass_planner(isf_pass_plan_fn f) { isf_planner_slot() = std::move(f); }

const isf_pass_plan_fn& get_isf_pass_planner() { return isf_planner_slot(); }

void set_isf_vulkan_source(isf_vulkan_source_fn f) { isf_vulkan_slot() = std::move(f); }

const isf_vulkan_source_fn& get_isf_vulkan_source() { return isf_vulkan_slot(); }

namespace {
isf_imported_fn& isf_imported_slot()
{
    static isf_imported_fn f;
    return f;
}
} // namespace

void set_isf_imported_images(isf_imported_fn f) { isf_imported_slot() = std::move(f); }

const isf_imported_fn& get_isf_imported_images() { return isf_imported_slot(); }

const isf_node_renderer& get_isf_node_renderer() { return isf_renderer_slot(); }

void set_port_resolver(const std::string& class_id, port_resolver resolver)
{
    std::lock_guard<std::mutex> lock(resolver_lock());
    resolvers()[class_id] = std::move(resolver);
}

std::vector<port_desc> instance_ports(const node_class& cls, const std::string& selector, std::string& out_reason)
{
    if (!cls.dynamic_ports)
        return cls.ports;

    if (selector.empty()) {
        // NO SELECTOR YET IS NOT A FAULT. An editor drops an `isf` node and then chooses a
        // shader, and reporting an error in between would make the whole document invalid --
        // so dropping a node mid-edit would stop the layer rendering.
        //
        // That is the failure mode this registry already refuses everywhere else: "a muted edge
        // or a dead branch must render the layer UNCHANGED, because a muted edge during a show
        // blacking a layer is the one failure mode nobody would forgive." A node with no shader
        // selected is a dead branch by exactly that definition, and the evaluator treats it as
        // bypassed -- it aliases its primary input and costs no draw.
        //
        // A WRONG path is a different thing entirely and still faults, because that is a
        // mistake rather than an unfinished edit.
        return cls.ports;
    }

    const auto key = std::make_pair(cls.id, selector);
    {
        std::lock_guard<std::mutex> lock(resolver_lock());
        const auto                  hit = port_cache().find(key);
        if (hit != port_cache().end())
            return hit->second;
    }

    port_resolver fn;
    {
        std::lock_guard<std::mutex> lock(resolver_lock());
        const auto                  it = resolvers().find(cls.id);
        if (it != resolvers().end())
            fn = it->second;
    }
    if (!fn) {
        // A dynamic class with nothing registered: the build has the class but not the module
        // that reads its files. Say that, rather than reporting every port as missing.
        out_reason = "'" + cls.id + "' is not wired into this build, so its ports cannot be read";
        return cls.ports;
    }

    std::string reason;
    auto        resolved = fn(selector, reason);
    if (resolved.empty()) {
        out_reason = reason.empty() ? ("'" + cls.ports_selector + "' could not be read") : reason;
        return cls.ports;
    }

    // THE STATIC PART FIRST, then what the file declared. The order is the one a client sees in
    // the catalogue, and a file cannot shadow a port the class guarantees: a shader declaring
    // its own `path` input must not replace the one that selects it.
    auto out = cls.ports;
    for (auto& p : resolved) {
        const auto exists = std::any_of(out.begin(), out.end(), [&](const port_desc& e) {
            return e.param.name == p.param.name && e.direction == p.direction;
        });
        if (!exists)
            out.push_back(std::move(p));
    }

    std::lock_guard<std::mutex> lock(resolver_lock());
    port_cache()[key] = out;
    return out;
}

const port_desc* find_port_in(const std::vector<port_desc>& ports, std::string_view name)
{
    const auto it =
        std::find_if(ports.begin(), ports.end(), [&](const port_desc& p) { return p.param.name == name; });
    return it == ports.end() ? nullptr : &*it;
}

const node_class* find_node_class(std::string_view id)
{
    const auto& t  = node_classes();
    const auto  it = std::find_if(t.begin(), t.end(), [&](const node_class& c) { return c.id == id; });
    return it == t.end() ? nullptr : &*it;
}

const port_desc* find_port(const node_class& cls, std::string_view name)
{
    const auto it = std::find_if(cls.ports.begin(), cls.ports.end(),
                                 [&](const port_desc& p) { return p.param.name == name; });
    return it == cls.ports.end() ? nullptr : &*it;
}

const char* domain_name(port_domain d)
{
    switch (d) {
        case port_domain::value: return "value";
        case port_domain::image: return "image";
        case port_domain::mask: return "mask";
        case port_domain::ref_layer: return "ref_layer";
        case port_domain::ref_channel: return "ref_channel";
        case port_domain::ref_lut: return "ref_lut";
    }
    return "?";
}

const char* flow_name(port_flow f)
{
    switch (f) {
        case port_flow::signal: return "signal";
        case port_flow::attribute: return "attribute";
        case port_flow::event: return "event";
    }
    return "?";
}

const char* direction_name(port_direction d)
{
    return d == port_direction::input ? "input" : "output";
}

const char* space_name(image_space s)
{
    return s == image_space::working ? "working" : "display";
}

// ---------------------------------------------------------------------------------------
// THE COERCION TABLE
//
// One function, three callers -- `validate` at PUT, `connections/preview` before a client commits
// a gesture, and `suggest` when a client asks what may be joined. They must agree, and the only
// way to make that structural rather than a convention is for all three to call this.
// ---------------------------------------------------------------------------------------

coercion coerce(const port_desc& from, const port_desc& to)
{
    coercion c;
    c.from = domain_name(from.domain);
    c.to   = domain_name(to.domain);

    // The three reference domains are declared so a client can see them refused, rather than
    // discovering the concept is missing. Checked first, so the message is about the domain and
    // not about a mismatch.
    const auto is_ref = [](port_domain d) {
        return d == port_domain::ref_layer || d == port_domain::ref_channel ||
               d == port_domain::ref_lut;
    };
    if (is_ref(from.domain) || is_ref(to.domain)) {
        c.legal = false;
        c.note  = "reference ports are declared and not implemented in v1";
        return c;
    }
    if (from.flow == port_flow::event || to.flow == port_flow::event) {
        c.legal = false;
        c.note  = "`event` flow is reserved and not implemented in v1";
        return c;
    }

    if (from.domain == to.domain) {
        // An image into an image still has to agree about its SPACE. Nothing declares `display`
        // in v1, so this cannot currently fire -- and it is here rather than added later because
        // the whole reason the tag exists is that a mismatch must be a refusal at PUT and never
        // an assumption at draw time.
        if (from.domain == port_domain::image && from.space != to.space) {
            c.legal = false;
            c.note  = std::string("an image in ") + space_name(from.space) +
                     " space cannot feed a " + space_name(to.space) +
                     " port. Use the document's `stage` rather than crossing spaces inside one "
                     "graph";
            return c;
        }
        if (from.domain == port_domain::value) {
            // Value to value: exact when the types are the same, reported when a conversion
            // loses something.
            const auto ft = from.param.type;
            const auto tt = to.param.type;
            c.legal       = true;
            c.from        = "value";
            c.to          = "value";
            if (ft == tt)
                return c;
            const auto is_num = [](fields::value_type t) {
                return t == fields::value_type::real || t == fields::value_type::integer ||
                       t == fields::value_type::boolean || t == fields::value_type::enumeration;
            };
            if (!is_num(ft) || !is_num(tt)) {
                c.legal = false;
                c.note  = "only numeric values convert; a string or a blob does not";
                return c;
            }
            if (tt == fields::value_type::integer || tt == fields::value_type::enumeration) {
                c.exact = false;
                c.note  = "rounded to the nearest whole value";
            } else if (tt == fields::value_type::boolean) {
                c.exact = false;
                c.note  = "true for any non-zero value";
            }
            return c;
        }
        c.legal = true;
        return c;
    }

    // mask -> image: replicate the single component across RGB with alpha 1. Legal and exact --
    // no information is lost, the mask simply becomes visible.
    if (from.domain == port_domain::mask && to.domain == port_domain::image) {
        c.legal = true;
        c.note  = "the mask is replicated across RGB with alpha 1, so it can be looked at";
        return c;
    }
    // image -> mask: a REDUCTION, so it is reported. The coefficients are named because a luma
    // is not a single agreed quantity: this is the working-space luma the shader's own
    // `working_luma` uses, which is the only one consistent with the rest of the chain.
    if (from.domain == port_domain::image && to.domain == port_domain::mask) {
        c.legal = true;
        c.exact = false;
        c.note  = "reduced to the WORKING-SPACE luma (the shader's own `working_luma`), so "
                  "three components become one";
        return c;
    }
    // value -> mask: a constant everywhere. Useful and exact.
    if (from.domain == port_domain::value && to.domain == port_domain::mask) {
        if (from.param.arity != 1) {
            c.legal = false;
            c.note  = "only a single value becomes a constant mask";
            return c;
        }
        c.legal = true;
        c.note  = "a constant mask over the whole raster";
        return c;
    }
    // Everything else is refused, and the note says what the client should reach for instead of
    // leaving them to guess which half was wrong.
    c.legal = false;
    if (from.domain == port_domain::image && to.domain == port_domain::value)
        c.note = "an image cannot become a value: there are no reduction nodes in v1";
    else if (from.domain == port_domain::mask && to.domain == port_domain::value)
        c.note = "a mask cannot become a value: there are no reduction nodes in v1";
    else
        c.note = std::string("no conversion from ") + c.from + " to " + c.to;
    return c;
}

// ---------------------------------------------------------------------------------------
// THE BOOT SELF-TEST
//
// A hand-written table gets a specific set of things wrong, and these are them. It aborts the
// boot rather than logging, for the reason `target_self_test` does: a registry that disagrees
// with its own rules produces a wrong ANSWER to a client's question about what it may build, and
// there is no later point at which that gets noticed.
// ---------------------------------------------------------------------------------------

void node_registry_self_test()
{
    const auto fail = [](const std::string& what) {
        CASPAR_THROW_EXCEPTION(caspar_exception()
                               << msg_info("node registry self-test: " + what));
    };

    const auto& classes = node_classes();
    if (classes.empty())
        fail("the table is empty");

    std::set<std::string> ids;
    for (const auto& c : classes) {
        if (c.id.empty())
            fail("a class has no id");
        if (!ids.insert(c.id).second)
            fail("two classes share the id '" + c.id + "'");
        if (c.id.find('/') != std::string::npos)
            fail("class id '" + c.id + "' contains '/', which the address grammar splits on");
        if (c.label.empty() || c.group.empty() || c.description.empty())
            fail("class '" + c.id + "' is missing a label, group or description");

        std::set<std::string> ports;
        int                   outputs = 0, image_outputs = 0;
        bool                  has_bypass = false, has_primary_image_in = false;
        for (const auto& p : c.ports) {
            if (p.param.name.empty())
                fail("class '" + c.id + "' has an unnamed port");
            if (!ports.insert(p.param.name).second)
                fail("class '" + c.id + "' has two ports named '" + p.param.name + "'");
            if (p.param.name.find('/') != std::string::npos ||
                p.param.name.find('.') != std::string::npos)
                fail("port '" + c.id + "." + p.param.name +
                     "' contains '/' or '.', which the address grammar splits on");

            if (p.direction == port_direction::output) {
                ++outputs;
                if (p.required)
                    fail("output '" + c.id + "." + p.param.name +
                         "' is marked required, which means nothing for an output");
                if (p.domain == port_domain::image)
                    ++image_outputs;
            }

            // NOTHING MAY DECLARE `display` SPACE. The stage is a property of the document, and a
            // port that claimed otherwise would make the registry depend on which document it is
            // in. This is the check that would have caught the prototype's placement if it had
            // ever been expressed as a port.
            if (p.domain == port_domain::image && p.space != image_space::working)
                fail("port '" + c.id + "." + p.param.name +
                     "' declares display space; the STAGE is a property of the document, not of "
                     "a port");

            if (p.domain == port_domain::value) {
                const auto arity_of = [](fields::value_type t) -> std::uint8_t {
                    switch (t) {
                        case fields::value_type::vec2: return 2;
                        case fields::value_type::vec3: return 3;
                        case fields::value_type::vec4: return 4;
                        default: return 1;
                    }
                };
                if (arity_of(p.param.type) != p.param.arity)
                    fail("port '" + c.id + "." + p.param.name + "' declares arity " +
                         std::to_string(p.param.arity) + " for a type whose arity is " +
                         std::to_string(arity_of(p.param.type)));
                if (p.param.default_value.size() != p.param.arity)
                    fail("port '" + c.id + "." + p.param.name + "' has " +
                         std::to_string(p.param.default_value.size()) +
                         " default components for arity " + std::to_string(p.param.arity));
                if (p.direction == port_direction::input &&
                    p.param.access == fields::access_t::read)
                    fail("value input '" + c.id + "." + p.param.name +
                         "' is read-only, so nothing could ever set it");
            }

            if (p.param.name == "bypass") {
                has_bypass = true;
                if (p.param.type != fields::value_type::boolean)
                    fail("'" + c.id + ".bypass' is not a boolean");
            }
            if (p.direction == port_direction::input && p.domain == port_domain::image &&
                p.required)
                has_primary_image_in = true;
        }

        if (outputs > 1)
            fail("class '" + c.id + "' has " + std::to_string(outputs) +
                 " outputs; v1 allows one, because a step produces one texture");
        if (c.id != "output" && outputs == 0)
            fail("class '" + c.id + "' has no output");
        if (c.produces_image && image_outputs != 1)
            fail("class '" + c.id + "' claims produces_image and has " +
                 std::to_string(image_outputs) + " image outputs");
        if (c.id == "input" || c.id == "output") {
            if (has_bypass)
                fail("root class '" + c.id + "' has a bypass, which has nothing to alias");
            continue;
        }
        if (!has_bypass)
            fail("class '" + c.id +
                 "' has no `bypass`; the registry adds it, so this means the loop missed it");
        // A BYPASSABLE CLASS NEEDS A PRIMARY INPUT TO ALIAS. Without one, bypass has no defined
        // meaning and the evaluator would have to invent something -- which for `mask_ellipse`
        // it does: see below.
        // A BYPASSABLE CLASS NEEDS SOMETHING DEFINED TO DO WHEN BYPASSED, and for the two
        // groups that is two different things. An image class aliases its primary input, so it
        // needs one. A MASK GENERATOR has no input to alias: bypassed, it emits its own port's
        // `disconnected_default` -- 1.0, "everywhere" -- so a bypassed mask means "no mask"
        // rather than "mask nothing", which is the same choice as an unconnected mask input and
        // for the same reason. Stated here because it is the one asymmetry in bypass's meaning.
        if (!has_primary_image_in && c.group != "mask")
            fail("class '" + c.id +
                 "' is bypassable but has no required image input to alias when bypassed");
        if (c.group == "mask" && !has_primary_image_in) {
            const auto* out = find_port(c, "out");
            if (!out || out->domain != port_domain::mask)
                fail("mask generator '" + c.id +
                     "' must have a mask output, since bypass emits that port's disconnected "
                     "default");
        }
    }

    if (!find_node_class("input") || !find_node_class("output"))
        fail("the two root classes must exist");

    // ---- THE OP INDICES AGAINST THE TABLE -------------------------------------------
    //
    // `node_step::cls` is an index into this table and it reaches the shader as `gn_op`, so the
    // same number is read by the table, two kernels and two shaders. A reordering of
    // `build_classes()` compiles perfectly and makes an `exposure` run the CDL's code. This is
    // the only thing that would catch it.
    {
        const std::pair<std::int32_t, const char*> ops[] = {
            {op_input, "input"},   {op_output, "output"},           {op_exposure, "exposure"},
            {op_cdl, "cdl"},       {op_mask_ellipse, "mask_ellipse"}, {op_mix, "mix"},
            {op_over, "over"},     {op_mask_rect, "mask_rect"},
            {op_mask_gradient, "mask_gradient"},
            {op_mask_qualifier, "mask_qualifier"},
            {op_mask_combine, "mask_combine"},
            {op_isf, "isf"},
        };
        for (const auto& o : ops) {
            if (static_cast<std::size_t>(o.first) >= classes.size())
                fail(std::string("op index for '") + o.second + "' is past the end of the table");
            if (classes[o.first].id != o.second)
                fail(std::string("op index ") + std::to_string(o.first) + " is '" +
                     classes[o.first].id + "' and the shaders expect '" + o.second +
                     "' -- `build_classes()` was reordered, and every gn_op in both shaders is "
                     "now wrong");
        }
        if (classes.size() != sizeof(ops) / sizeof(ops[0]))
            fail("the table has " + std::to_string(classes.size()) +
                 " classes and the op enum names " + std::to_string(sizeof(ops) / sizeof(ops[0])) +
                 " -- a class was added without an op constant, so the shaders cannot switch on "
                 "it and it would render as whatever the default case does");
    }

    // ---- the coercion table's own rules ----------------------------------------------
    const auto* expo = find_node_class("exposure");
    const auto* mixc = find_node_class("mix");
    if (!expo || !mixc)
        fail("exposure and mix must exist");

    const auto* img_out  = find_port(*expo, "out");
    const auto* img_in   = find_port(*mixc, "a");
    const auto* mask_in  = find_port(*expo, "mask");
    const auto* val_in   = find_port(*expo, "gain");
    const auto* mask_out = find_port(*find_node_class("mask_ellipse"), "out");
    if (!img_out || !img_in || !mask_in || !val_in || !mask_out)
        fail("the ports the coercion checks need are missing");

    if (!coerce(*img_out, *img_in).legal || !coerce(*img_out, *img_in).exact)
        fail("image -> image must be legal and exact");
    if (!coerce(*mask_out, *mask_in).legal || !coerce(*mask_out, *mask_in).exact)
        fail("mask -> mask must be legal and exact");
    // ASYMMETRIC ON PURPOSE, and asserted in both directions because a table that made these
    // symmetric would silently allow an image to be used as a mask with no warning.
    {
        const auto i2m = coerce(*img_out, *mask_in);
        if (!i2m.legal || i2m.exact)
            fail("image -> mask must be legal and REPORTED as lossy: it is a luma reduction");
        const auto m2i = coerce(*mask_out, *img_in);
        if (!m2i.legal || !m2i.exact)
            fail("mask -> image must be legal and exact: it is a replication");
        const auto i2v = coerce(*img_out, *val_in);
        if (i2v.legal)
            fail("image -> value must be refused: there are no reduction nodes in v1");
        const auto v2m = coerce(*val_in, *mask_in);
        if (!v2m.legal)
            fail("value -> mask must be legal: it is a constant mask");
    }
    // A value into an output, or an output into an output, is a DIRECTION error rather than a
    // domain one, and `validate` owns that -- `coerce` is asked only about domains. Asserted so
    // nobody later moves the direction check in here and makes two places answer it.
    if (!coerce(*val_in, *val_in).legal)
        fail("coerce must answer about DOMAINS only; direction is validate's question");

    std::size_t ports = 0;
    for (const auto& c : classes)
        ports += c.ports.size();
    CASPAR_LOG(info) << L"[graph-registry] self-test: all checks passed, over " << classes.size()
                     << L" node classes and " << ports << L" ports";
}

}}} // namespace caspar::core::graph
