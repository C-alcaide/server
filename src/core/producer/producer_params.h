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

// A producer's parameters, described the way `transform_fields.h` describes a mixer field.
//
// WHY THIS IS NOT `typed_field<T>`. That row type is a static table over a struct: the
// accessors are plain function pointers and the strings are `const char*` literals, because
// an `image_transform` has the same 178 fields on every layer forever. A producer's
// parameters are neither static nor shared -- an ISF shader declares its own `INPUTS` in its
// own JSON header, and two layers running two shaders have two different sets. So the
// descriptions are owned strings and the accessors are closures bound to the instance.
//
// WHAT THE SHAPE BUYS. ISF and OFX parameters were reachable only as `CALL` strings:
// `CALL 1-10 ISF SET brightness 0.5` with `CALL 1-10 ISF LIST` to discover them, and a
// reply parsed out of formatted text. They were not in the control API's tree, not
// publishable, not addressable by path and not keyframable. Declaring them here makes them
// describable and writable through the same machinery a mixer field uses, which is also the
// prerequisite for BINDING one to a live source -- a binding resolves a target path to a
// descriptor and a setter, and neither existed for a producer.
//
// THREADING. `get` and `set` are called ON THE STAGE EXECUTOR and nowhere else, because a
// producer is not generally thread-safe and `CALL` already reaches it that way. That is why
// the stage exposes `describe_params` and `set_param` -- pure data in and out -- rather than
// handing a `param_desc` to a caller whose thread nobody chose.

#include <core/frame/transform_fields.h>
#include <core/monitor/monitor.h>

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace caspar { namespace core {

/// One parameter of one producer INSTANCE, described and accessible.
struct param_desc
{
    /// The path segment, under `.../foreground/params/`. An ISF input's own name.
    std::string name;

    fields::value_type type   = fields::value_type::real;
    fields::access_t   access = fields::access_t::read_write;

    /// The declared range, when the format supplies one. ISF's `MIN`/`MAX` are optional and
    /// OFX's are not, so this is genuinely absent for some parameters rather than defaulted --
    /// a control surface that invented 0..1 for an unbounded parameter would be lying.
    std::optional<double> min;
    std::optional<double> max;

    fields::bounding_t bounding = fields::bounding_t::refuse;

    /// Human-facing. `label` is the format's own display name where it has one; ISF calls it
    /// LABEL and OFX calls it the label too.
    std::string label;
    std::string unit;

    /// For `enumeration`: the value names in order, comma-separated -- ISF's `LABELS` for a
    /// `long` input, OFX's choice options.
    std::string values;

    std::string description;

    double  step  = 0.0;
    uint8_t arity = 1;

    /// The format's own default, so a reader can tell a value at its default from an absent one
    /// exactly as it can for a mixer field.
    monitor::vector_t default_value;

    /// Read the current value. Stage executor only.
    std::function<monitor::vector_t()> get;

    /// Write it. Returns false for a wrong type or arity, or for a value the producer refuses.
    /// Does NOT range-check -- the caller does that against `min`/`max` and `bounding`, the
    /// same division of labour `typed_field::set` has. Stage executor only.
    std::function<bool(const monitor::vector_t&)> set;
};

/// A parameter as pure data: everything above except the closures, plus the current value.
///
/// This is what crosses a thread or a protocol boundary. Handing out a `param_desc` would
/// hand out two closures holding a raw producer pointer, whose validity ends the moment the
/// layer is cleared -- and the API's reader threads are exactly the callers who would keep one.
struct param_snapshot
{
    std::string           name;
    fields::value_type    type   = fields::value_type::real;
    fields::access_t      access = fields::access_t::read_write;
    std::optional<double> min;
    std::optional<double> max;
    fields::bounding_t    bounding = fields::bounding_t::refuse;
    std::string           label;
    std::string           unit;
    std::string           values;
    std::string           description;
    double                step  = 0.0;
    uint8_t               arity = 1;
    monitor::vector_t     default_value;
    monitor::vector_t     value;
};

/// `param_desc` -> `param_snapshot`, reading the value through `get`. Stage executor only.
param_snapshot snapshot_of(const param_desc& p);

/// One `monitor::data_t` as a double, whatever arithmetic type it actually holds.
///
/// Exported because every producer implementing `parameters()` needs it and there is no
/// reason for each to carry its own copy. `monitor::data_t` is a `boost::variant` over nine
/// types, so a setter that only accepted `double` would refuse a boolean sent as a bool and
/// an enumeration index sent as an int64 -- which is what a JSON body and an OSC packet
/// respectively produce. `transform_fields.cpp` has had a private one of these since the
/// registry was written; this is the same function, visible.
bool as_number(const monitor::data_t& d, double& out);

/// A whole `vector_t` as doubles. False if any element is not arithmetic, so a partially
/// converted write is never applied.
bool as_numbers(const monitor::vector_t& v, std::vector<double>& out);

}} // namespace caspar::core
