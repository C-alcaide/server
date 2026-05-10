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
#include <string>

namespace caspar { namespace keyframes {

/// Serialize a keyframe_timeline to a compact JSON string.
/// Format:
///  { "keyframes": [
///      { "time_secs": 0.0, "easing": "LINEAR", "opacity": 1.0, ... },
///      ...
///  ]}
std::string timeline_to_json(const keyframe_timeline& tl);

/// Deserialize a keyframe_timeline from a JSON string.
/// Throws std::runtime_error on parse failure.
keyframe_timeline json_to_timeline(const std::string& json);

/// Apply a partial JSON object (only the fields present) on top of *base*.
/// Fields not mentioned in the JSON retain their values from *base*.
/// Throws std::runtime_error on parse failure.
kf_state patch_state_from_json(const kf_state& base, const std::string& json);

}} // namespace caspar::keyframes
