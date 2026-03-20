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

#include <core/frame/frame_transform.h>

#include <string>
#include <vector>

namespace caspar { namespace keyframes {

// ---------------------------------------------------------------------------
// kf_state — flat numeric snapshot of all keyframeable image_transform fields
//
// All fields have the same defaults as their image_transform counterparts so
// that a "zero" kf_state represents the untouched mixer state.
// ---------------------------------------------------------------------------

struct kf_state
{
    // ── Basic ───────────────────────────────────────────────────────
    double opacity    = 1.0;
    double contrast   = 1.0;
    double brightness = 1.0;
    double saturation = 1.0;

    // ── Geometry ────────────────────────────────────────────────────
    double anchor_x  = 0.0, anchor_y  = 0.0;
    double fill_x    = 0.0, fill_y    = 0.0;
    double fill_sx   = 1.0, fill_sy   = 1.0;
    double angle     = 0.0;
    double crop_ul_x = 0.0, crop_ul_y = 0.0;
    double crop_lr_x = 1.0, crop_lr_y = 1.0;

    // ── Perspective corners (UL UR LR LL, each xy) ──────────────────
    double persp_ul_x = 0.0, persp_ul_y = 0.0;
    double persp_ur_x = 1.0, persp_ur_y = 0.0;
    double persp_lr_x = 1.0, persp_lr_y = 1.0;
    double persp_ll_x = 0.0, persp_ll_y = 1.0;

    // ── Projection (360°) ────────────────────────────────────────────
    double proj_enable   = 0.0;
    double proj_yaw      = 0.0;   // degrees (apply_state converts to radians)
    double proj_pitch    = 0.0;   // degrees
    double proj_roll     = 0.0;   // degrees
    double proj_fov      = 90.0;  // degrees (90° default = π/2 rad; apply_state converts)
    double proj_offset_x = 0.0;
    double proj_offset_y = 0.0;

    // ── White balance ────────────────────────────────────────────────
    double temperature = 0.0;
    double tint        = 0.0;

    // ── Tone balance ─────────────────────────────────────────────────
    double shadows    = 0.0;
    double highlights = 0.0;

    // ── 3-Way colour corrector ───────────────────────────────────────
    double lift_r = 0.0, lift_g = 0.0, lift_b = 0.0;
    double mid_r  = 1.0, mid_g  = 1.0, mid_b  = 1.0;
    double gain_r = 1.0, gain_g = 1.0, gain_b = 1.0;

    // ── Hue shift (degrees) ──────────────────────────────────────────
    double hue_shift = 0.0;

    // ── Invert ───────────────────────────────────────────────────────
    double invert = 0.0;

    // ── Levels (master) ──────────────────────────────────────────────
    double levels_min_in  = 0.0, levels_max_in  = 1.0;
    double levels_gamma   = 1.0;
    double levels_min_out = 0.0, levels_max_out = 1.0;

    // ── Per-channel RGB levels (R / G / B, each 5 scalars) ───────────
    double rgb_r_min_in = 0.0, rgb_r_max_in = 1.0, rgb_r_gamma = 1.0,
           rgb_r_min_out = 0.0, rgb_r_max_out = 1.0;
    double rgb_g_min_in = 0.0, rgb_g_max_in = 1.0, rgb_g_gamma = 1.0,
           rgb_g_min_out = 0.0, rgb_g_max_out = 1.0;
    double rgb_b_min_in = 0.0, rgb_b_max_in = 1.0, rgb_b_gamma = 1.0,
           rgb_b_min_out = 0.0, rgb_b_max_out = 1.0;

    // ── Blur ─────────────────────────────────────────────────────────
    double blur_radius   = 0.0;
    double blur_angle    = 0.0;
    double blur_center_x = 0.5, blur_center_y = 0.5;
    double blur_tilt_y   = 0.5, blur_tilt_h   = 0.2;
};

// ---------------------------------------------------------------------------
// keyframe_t — a single keyframe: time + easing + all mixer values
// ---------------------------------------------------------------------------

struct keyframe_t
{
    double      time_secs = 0.0;   // position in file time (seconds)
    std::string easing    = "LINEAR";
    kf_state    state;
};

// ---------------------------------------------------------------------------
// Helper: apply an easing function to a normalised [0,1] progress value
// ---------------------------------------------------------------------------
double ease(double t, const std::string& name);

// ---------------------------------------------------------------------------
// Lerp between two kf_state values (field-by-field linear, t in [0,1])
// ---------------------------------------------------------------------------
kf_state lerp_state(const kf_state& a, const kf_state& b, double t);

// ---------------------------------------------------------------------------
// Apply a kf_state to an image_transform reference in-place
// ---------------------------------------------------------------------------
void apply_state(const kf_state& s, core::image_transform& tf);

// ---------------------------------------------------------------------------
// Capture the current image_transform values into a kf_state
// ---------------------------------------------------------------------------
kf_state capture_state(const core::image_transform& tf);

// ---------------------------------------------------------------------------
// keyframe_timeline — ordered collection of keyframes with interpolation
// ---------------------------------------------------------------------------

class keyframe_timeline
{
  public:
    void   add(keyframe_t kf);
    void   remove(double time_secs);   // removes nearest keyframe
    void   clear();
    bool   empty() const;
    size_t size() const;

    const std::vector<keyframe_t>& keyframes() const;

    /// Overwrite the state of the keyframe nearest to *time_secs* (within 1ms).
    /// Returns true if a matching keyframe was found and patched.
    bool patch_at_time(double time_secs, const kf_state& patched_state);

    // Interpolate at the given file time.
    // Before the first keyframe: returns first keyframe's state.
    // After the last keyframe:   returns last keyframe's state.
    // Between keyframes A and B: applies A's easing curve to the lerp.
    kf_state interpolate(double time_secs) const;

  private:
    std::vector<keyframe_t> kfs_; // sorted by time_secs
};

}} // namespace caspar::keyframes
