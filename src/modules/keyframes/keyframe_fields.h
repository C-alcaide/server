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

#include "keyframe_data.h"

#include <core/frame/frame_transform.h>

#include <string>
#include <vector>

namespace caspar { namespace keyframes {

// ---------------------------------------------------------------------------
// Field kind — controls interpolation behavior
// ---------------------------------------------------------------------------

enum class field_kind
{
    continuous, // standard linear interpolation
    angular,    // shortest-path wrapping at 360° (value in degrees)
    discrete    // snaps at source keyframe boundary, not interpolated
};

// ---------------------------------------------------------------------------
// kf_field — descriptor for one animatable image_transform property
//
// Adding a new MIXER property to keyframes requires adding ONE entry to the
// field table in keyframe_fields.cpp.  No other code changes needed.
// ---------------------------------------------------------------------------

struct kf_field
{
    const char* name;
    double (*get)(const core::image_transform&);
    void (*set)(core::image_transform&, double);
    double     default_val;
    field_kind kind;
};

// ---------------------------------------------------------------------------
// API
// ---------------------------------------------------------------------------

/// All registered field descriptors.
const std::vector<kf_field>& kf_all_fields();

/// Look up a field descriptor by JSON key name.  Returns nullptr if unknown.
const kf_field* kf_find_field(const std::string& name);

/// Apply sparse keyframe values to an image_transform.
///
/// Only the fields present in `vals` are written.  Angular fields use the
/// setter's degree→radian conversion.  After setting fields, subsystem
/// enable flags are auto-activated (e.g. enable_geometry_modifiers if any
/// geometry field was set, blur.enable if blur_radius > 0, etc.).
void apply_kf_to_transform(const kf_values& vals, core::image_transform& tf);

/// Capture the current image_transform values into sparse kf_values.
/// If only_non_default is true, only fields that differ from their default
/// are included.  If false, all fields are captured.
kf_values capture_from_transform(const core::image_transform& tf, bool only_non_default = true);

}} // namespace caspar::keyframes
