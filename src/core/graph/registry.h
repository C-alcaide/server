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

// WHAT NODE CLASSES EXIST, and the one table that says whether two ports may be joined.
//
// A FIXED table, like `fields::mixer_fields()`, and for the same reason: a client needs to know
// what it may build BEFORE it builds it, and the only honest way to answer that is a registry the
// server compiled. `/v1/catalog/node` renders this; `validate` consults it; the compiler reads the
// step shape off it. Three readers, one table.
//
// THERE IS DELIBERATELY NO NEW TYPE SYSTEM, which is `transform_fields.h`'s rule and it applies
// here with more force. A port's VALUE half is a `param_snapshot` -- the same struct a producer
// parameter is described by -- so `api_tree.cpp`'s `param_leaf` and `api_value.cpp`'s
// `json_to_value`/`check_and_bound` describe and validate a node parameter with no new descriptor
// code at all, and a node parameter carries exactly a mixer field's key set including
// `animatable`. What a port adds is three enums, and only three:
//
//   direction   input or output. Not derivable from anything.
//   flow        signal (a value that changes per tick) or attribute (a value that changes the
//               TOPOLOGY or the compiled plan). The split is the whole reason a timeline can ramp
//               a node parameter at 50 Hz without reallocating a graph -- see `plan.h`.
//   domain      value, image or mask in v1. Three more are DECLARED and refused, because a
//               grammar that has no word for "a reference to a layer" cannot be extended to one
//               without a wire change, and a client needs to see the refusal rather than discover
//               the concept is absent.
//
// WHY `image` CARRIES TAGS NOW. `space` (working or display) and `alpha` (premultiplied or
// straight) are on the port from the first commit, when nothing needs them, because the
// alternative is a conversion that gets ASSUMED. This tree has already paid for that twice: the
// YCbCr decode counted in 8-bit codes for every bit depth, and `apply_transform_colour_values`
// silently drops a field nobody added to it. A tagged handle makes a mismatched join either an
// inserted conversion or a refusal at PUT, never a wrong picture. Blender's explicit conversion
// nodes and Aximmetry's colour-space pins are both this decision.
//
// WHAT IS NOT HERE, and each has its reason:
//
//   * NO SCALAR/MATH NODES. Scalar ports ARE parameters, and the server already has two scalar
//     dataflow engines addressing them -- bindings (LFO, audio, OSC, tracker) and the timeline
//     (curves, steps). A third inside the graph would duplicate both. A node parameter is an
//     ADDRESS, so a binding drives it and a document keys it with no new mechanism.
//   * NO `blur`/`sharpen`/`grain`. A windowed neighbourhood operator has to define its edge
//     behaviour, and doing that per node with pooled attachments is a separate design.
//   * NO `lut3d`, though the design lists it. Its LUT input needs a `ref_lut` port and reference
//     ports are refused in v1, so the class would be a node that can only be a pass-through with
//     a `strength` nobody can apply. Declaring an unusable class is the 202-and-no-picture shape
//     from the other end: the catalogue would promise something a PUT cannot connect.
//   * NO `group`. The model carries one (a group is inlined at compile), the evaluator never sees
//     one, and v1 refuses it.
//
// Every class gets an implicit `bypass` input from the registry's own constructor rather than from
// each class's author, so no class can forget it. It is a `discrete` boolean, which makes it a
// STEP target: a timeline switches a chain on at a key rather than sliding through it.

#include <core/producer/producer_params.h>

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace caspar { namespace core { namespace graph {

/// Which side of the node a port is on. Not derivable: `mix.amount` and `mix.out` are both
/// values and only one may be written.
enum class port_direction : std::uint8_t
{
    input,
    output,
};

/// WHAT A CHANGE TO THIS PORT INVALIDATES, which is the distinction the frame path is built on.
///
/// A `signal` port's value is compared BY VALUE and lives in a flat array the resolver writes
/// into; changing it costs nothing beyond the write. An `attribute` port's value is part of the
/// compiled plan -- it changes the step list, the pass count or a mask's space -- so changing it
/// requires a re-PUT and a new plan pointer.
///
/// Getting this wrong in the cheap direction (declaring a signal as an attribute) makes a
/// timeline ramp allocate a graph fifty times a second. Getting it wrong the other way makes a
/// change silently not take effect. So it is declared per port rather than inferred from the type.
///
/// `event` is reserved: ISF declares `event` inputs and OFX has push-button parameters, and both
/// are a trigger rather than a value. Refused in v1.
enum class port_flow : std::uint8_t
{
    signal,
    attribute,
    event,
};

/// WHAT FLOWS. `value` is a number or a name; `image` and `mask` are textures.
///
/// The last three are declared and REFUSED, which is deliberate. A client can see that the
/// concept exists and is not available, which is a different message from the concept being
/// absent -- and adding one later is a table entry rather than a grammar change.
enum class port_domain : std::uint8_t
{
    value,
    image,
    mask,
    ref_layer,   ///< v2 -- another layer's output as an input
    ref_channel, ///< v2 -- `route://`, with a latency tag
    ref_lut,     ///< v2 -- a LUT file as a node input; what blocks `lut3d`
};

/// WHICH SPACE an `image` port's texture is in. On the port, from the first commit, so a
/// mismatched join is a refusal or an inserted conversion and never an assumption.
enum class image_space : std::uint8_t
{
    working, ///< scene-linear, the working gamut. Where the design puts node passes.
    display, ///< output-encoded. Where the PROTOTYPE's nodes ran, measured 42 LSB away.
};

/// ...and whether its alpha is premultiplied. The legacy path is premultiplied and
/// `<straight-alpha-grading>` is not, so this is per configuration rather than a constant.
enum class image_alpha : std::uint8_t
{
    premultiplied,
    straight,
};

/// One port. A `param_snapshot` plus the three enums, and nothing else.
struct port_desc
{
    /// Name, type, arity, range, bounding, default, description -- the same descriptor a producer
    /// parameter carries, so every reader of that already reads this.
    param_snapshot param;

    port_direction direction = port_direction::input;
    port_flow      flow      = port_flow::signal;
    port_domain    domain    = port_domain::value;

    /// Only meaningful for `domain == image`. A `mask` is single-component and unencoded by
    /// construction, so it carries neither.
    image_space space = image_space::working;
    image_alpha alpha = image_alpha::premultiplied;

    /// A required input that nothing is connected to is a validation ERROR, not a default. `mix`
    /// needs both of its images; `exposure`'s mask is optional and defaults to 1.
    bool required = false;

    /// WHAT AN UNCONNECTED optional input CONTRIBUTES, and it is never transparent black.
    ///
    /// A muted edge or a dead branch must render the layer UNCHANGED, because a muted edge during
    /// a show blacking a layer is the one failure mode nobody would forgive. So an image input
    /// with no live route aliases the node's primary input, a mask defaults to 1.0, and `mix.b`
    /// returns `a`. Recorded per port so the evaluator does not have to know class by class.
    double disconnected_default = 1.0;
};

/// One node class: an id, presentation, and its ports.
struct node_class
{
    std::string id;    ///< `exposure`, `mask_ellipse`. The wire name; never changes.
    std::string label; ///< for a client's palette
    std::string group; ///< `grade`, `mask`, `combine`, `root` -- how a palette is organised
    std::string description;

    std::vector<port_desc> ports;

    /// Can a client ask for a picture of this node's output? False for `output` (it is the
    /// layer) and for anything that produces no image.
    bool preview = false;

    /// Does this class produce an IMAGE, and therefore cost a pass? Read off the ports, cached
    /// here because the compiler asks it per step.
    bool produces_image = false;
};

/// THE CLASS INDICES THE KERNELS AND THE SHADERS SWITCH ON.
///
/// `node_step::cls` is an index into `node_classes()`, and it reaches the shader as `gn_op`. So
/// the same number is read in four places -- the table, two kernels and two shaders -- and a
/// reordering of `build_classes()` would silently make an `exposure` run the CDL's code.
///
/// Named here, and `node_registry_self_test` asserts every one of them against the table. That
/// assertion is the only thing standing between a reorder and a wrong picture, because nothing
/// about the reorder itself would fail to compile.
enum : std::int32_t
{
    op_input        = 0,
    op_output       = 1,
    op_exposure     = 2,
    op_cdl          = 3,
    op_mask_ellipse = 4,
    op_mix          = 5,
    op_over         = 6,
};

/// The table. Built once, on first call, and never mutated.
const std::vector<node_class>& node_classes();

const node_class* find_node_class(std::string_view id);
const port_desc*  find_port(const node_class& cls, std::string_view name);

/// Is joining these two ports legal, and is it lossless?
///
/// ONE TABLE, and that is the point rather than an implementation detail: `validate` (at PUT),
/// `connections/preview` (before the client commits a gesture) and `suggest` (what a client is
/// offered) must give the same answer to "may these be joined", and the only way to guarantee
/// that is for all three to call this. Two coercion tables would let a client be offered an edge
/// that a PUT then refuses, which is worse than not offering it.
struct coercion
{
    bool        legal = false;
    /// Legal but LOSSY -- reported to the client as a `coercion` fault rather than an error, so
    /// the document stores and the picture renders while the editor can show a warning on the edge.
    bool        exact = true;
    std::string from;
    std::string to;
    std::string note;
};

coercion coerce(const port_desc& from, const port_desc& to);

const char* domain_name(port_domain);
const char* flow_name(port_flow);
const char* direction_name(port_direction);
const char* space_name(image_space);

/// Aborts the boot on a disagreement between this table and its own rules.
///
/// Called from `server.cpp` beside the other self-tests. What it checks is what a hand-written
/// table gets wrong: a class with no output, a duplicate port name, an `image` output tagged
/// `display` (nothing may declare that in v1 -- the prototype's placement is what the graph
/// exists to fix), a `required` output, a value port whose arity disagrees with its type, a
/// missing implicit `bypass`, and every coercion being symmetric where it claims to be.
void node_registry_self_test();

}}} // namespace caspar::core::graph
