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

#include <string>
#include <vector>

namespace caspar { namespace core { namespace fields {

struct stage_math_report
{
    int                      checks = 0;
    std::vector<std::string> failures; // "name: got X want Y (tol T)", empty when clean
};

/// Property checks on `compute_frustum`, with no GL device and no display.
///
/// `compute_frustum` has no published standard behind it -- it is this fork's own convention
/// for turning a screen's placement into the orientation a channel must render at -- so there is
/// no authority to compare a number against. What there IS is a set of properties that any
/// correct implementation has and a wrong one breaks, and a genuine second implementation.
///
/// Both are used. The properties (a screen facing the camera projects at yaw 0; rotating it
/// about Y rotates the projection with it; the field of view follows the PERPENDICULAR distance;
/// a fixed design eye ignores the camera; the documented degenerate cases return what they are
/// documented to return) fail fast at boot and name the property. The ICVFX quad is additionally
/// intersected a second time from a closed-form camera basis -- the same expressions
/// `casparcg-360-client`'s pure-numpy `frustum_check.py` uses -- which catches an
/// agreed-but-wrong convention that properties and a single implementation would share.
///
/// Deliberately NOT reusing `mat4`: `compute_frustum` builds its bases by multiplying `mat4`
/// rotations and reading columns out by index, so a check written the same way would compare the
/// implementation against itself and pass for any self-consistent rotation order. That is the
/// same failure the ACEScg gamut matrices had -- two of them round-tripped to exactly the
/// identity while both were wrong.
stage_math_report stage_math_self_test();

/// Run it and log the outcome, in the shape `run_compose_self_test` established.
void run_stage_math_self_test();

}}} // namespace caspar::core::fields
