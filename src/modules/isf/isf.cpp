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

#include <common/log.h>

#include <core/producer/frame_producer_registry.h>

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

}} // namespace caspar::isf
