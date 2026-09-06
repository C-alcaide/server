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
#include "state_hub.h"

#include <common/tweener.h>
#include <core/frame/transform_fields.h>
#include <core/producer/stage.h>

#include <string>

namespace caspar { namespace protocol { namespace http {

/// A resolved write target: which channel, which layer, which descriptor.
struct write_target
{
    int                            channel = 0;
    int                            layer   = 0;
    const core::fields::field_desc* field   = nullptr;
};

/// Parse `/channel/1/stage/layer/10/mixer/opacity`. Anything else is refused with the code
/// that says WHICH part did not resolve, because "bad request" leaves a client guessing
/// between a typo, a channel that is not configured and a field this build does not have.
api_reply resolve_write_target(const std::string& path, write_target& out);

/// Where a stage write lands: which channel, which object, which field.
struct stage_write_target
{
    int                            channel = 0;
    /// "camera", "view_camera", or "screen/<name>" -- what `api_context::set_stage_field` takes.
    std::string                    object;
    const core::fields::field_meta* field = nullptr;
};

/// Parse `/channel/1/mixer/previz/camera/fov`, `/channel/1/mixer/previz/view_camera/position`
/// or `/channel/1/mixer/previz/screen/back/position`.
///
/// A SIBLING of `resolve_write_target` rather than a branch inside it. The two address spaces
/// have different shapes -- a layer is an integer and a screen is a runtime name -- and the
/// codes they return for a miss are different in a way a client acts on: an unknown layer is
/// `layer_not_found`, an unknown screen is `unknown_path`, because a screen that does not exist
/// is a typo rather than a channel that was not configured.
///
/// Returns `unknown_path` with no diagnosis for anything that is not a previz path at all, so
/// the caller can fall through to the mixer resolver and let THAT one produce the message.
api_reply resolve_stage_write_target(const std::string& path, stage_write_target& out);

/// Is this path shaped like a stage write? Cheap enough to test before either resolver runs.
bool is_stage_path(const std::string& path);

/// A JSON value as the descriptor's `vector_t`, checked for type and arity.
///
/// This is where a client's `"0.5"` is refused rather than quietly parsed: a control
/// surface that sends a string for a real number has a bug, and accepting it moves the
/// failure to somewhere much harder to find.
api_reply json_to_value(const core::fields::field_meta& f, const json::value& v, core::monitor::vector_t& out);

/// Range-check against the descriptor, component by component.
///
/// Out of range is REFUSED, not clipped -- the same rule `grade_param` applies to every
/// MIXER command, so the two façades cannot disagree about what is legal. `wrap` fields are
/// normalised into range first, because for a periodic quantity 400 degrees is a legal way
/// to say 40, not an error. The failure carries the component index and the limits, so a
/// generated control can put its slider back where it was and say why.
api_reply check_and_bound(const core::fields::field_meta& f, core::monitor::vector_t& v);

/// A validated `set`, ready to apply to any stage.
///
/// Split out of `write_value` so a batch can validate every op BEFORE applying any of
/// them. That ordering is the batch's whole guarantee, and it is only possible if
/// validation and application are separable -- which they were not while the two were one
/// function.
struct prepared_set
{
    write_target            target;
    core::monitor::vector_t value;
    unsigned                duration = 0;
    caspar::tweener         tween;
};

/// Validate one `{"path": ..., "value": ..., "duration": ..., "tween": ...}` object.
/// Touches no stage and has no side effects.
api_reply prepare_set(const std::string& path, const json::object& op, prepared_set& out);

/// The closure a prepared set applies. Handed to `apply_transform` on whichever stage --
/// the channel's own, or a `stage_delayed` standing in for it inside a batch.
core::stage_base::transform_func_t set_closure(const prepared_set& p);

/// `PUT /v1/value/{path}`.
///
/// Body: `{"value": ..., "duration": 25, "tween": "easeoutsine", "label": "..."}`, or an
/// in-place operation `{"op":"toggle"|"add"|"cas", ...}`.
///
/// Every form runs as ONE closure on the stage executor, so a read-modify-write cannot
/// interleave with another client's -- which is what makes `toggle` and `cas` meaningful
/// rather than approximate.
api_reply write_value(const api_context& ctx,
                      const state_hub&   hub,
                      const std::string& path,
                      const std::string& body,
                      const std::string& peer);

}}} // namespace caspar::protocol::http
