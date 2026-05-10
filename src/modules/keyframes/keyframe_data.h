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

#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

namespace caspar { namespace keyframes {

// ---------------------------------------------------------------------------
// Easing function type and resolver
// ---------------------------------------------------------------------------

using easing_fn_t = double (*)(double);

/// Resolve an easing name (case-insensitive) to a function pointer.
/// Unknown names log a warning (once per name) and return linear.
easing_fn_t resolve_easing(const std::string& name);

// ---------------------------------------------------------------------------
// Keyframe status snapshot (returned by get_keyframe_status)
// ---------------------------------------------------------------------------

struct kf_status
{
    bool   has_timeline   = false;
    bool   armed          = false;
    size_t keyframe_count = 0;
};

// ---------------------------------------------------------------------------
// Sparse keyframe values — only the animated fields
// ---------------------------------------------------------------------------

using kf_values = std::unordered_map<std::string, double>;

// ---------------------------------------------------------------------------
// keyframe_t — a single keyframe: time + easing + sparse field values
// ---------------------------------------------------------------------------

struct keyframe_t
{
    double      time_secs   = 0.0;
    easing_fn_t easing_fn   = nullptr; // resolved at parse time
    std::string easing_name = "LINEAR"; // kept for serialization
    kf_values   values;                 // only the fields that are set
};

// ---------------------------------------------------------------------------
// keyframe_timeline — ordered collection of keyframes with per-field
//                     interpolation and hold semantics.
//
// Each field is interpolated independently: between the two nearest
// keyframes that CONTAIN that field.  Fields absent from a keyframe are
// not affected — they hold their last known value or revert to the mixer
// base state.  This gives NLE-like per-property animation on a unified
// timeline.
// ---------------------------------------------------------------------------

class keyframe_timeline
{
  public:
    void add(keyframe_t kf);

    /// Remove the nearest keyframe to time_secs, within max_distance.
    /// Returns true if a keyframe was removed.
    bool remove(double time_secs, double max_distance = 0.5);

    void   clear();
    bool   empty() const;
    size_t size() const;

    const std::vector<keyframe_t>& keyframes() const;

    /// Overwrite specific fields of the keyframe nearest to time_secs
    /// (within 1 ms).  Only fields present in `patch` are changed;
    /// existing fields not in `patch` are preserved.
    /// Returns true if a matching keyframe was found and patched.
    bool patch_at_time(double time_secs, const kf_values& patch);

    /// Per-field interpolation at the given time.
    ///
    /// For each field that appears in any keyframe:
    ///  - Find the last keyframe at-or-before `time` containing the field.
    ///  - Find the first keyframe after `time` containing the field.
    ///  - If both exist: lerp (with the "before" keyframe's easing).
    ///  - If only "before": hold at its value.
    ///  - If only "after": hold at its value (pre-roll).
    kf_values interpolate(double time_secs) const;

  private:
    std::vector<keyframe_t>  kfs_;             // sorted by time_secs
    std::vector<std::string> all_field_names_; // unique, rebuilt on mutation

    void rebuild_field_index();
};

}} // namespace caspar::keyframes
