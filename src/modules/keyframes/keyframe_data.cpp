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

#include "keyframe_data.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#ifndef M_PI
#define M_PI 3.141592653589793
#endif

namespace caspar { namespace keyframes {

// ---------------------------------------------------------------------------
// Easing functions — maps CasparCG tween names to f(t) → t′
// All take t ∈ [0,1] and return a value that is also typically in [0,1]
// (bounce/elastic may overshoot slightly).
// ---------------------------------------------------------------------------

static double ease_linear(double t) { return t; }

static double ease_in_sine(double t)  { return 1.0 - std::cos(t * M_PI * 0.5); }
static double ease_out_sine(double t) { return std::sin(t * M_PI * 0.5); }
static double ease_inout_sine(double t) { return -(std::cos(M_PI * t) - 1.0) * 0.5; }

static double ease_in_quad(double t)     { return t * t; }
static double ease_out_quad(double t)    { return 1.0 - (1.0 - t) * (1.0 - t); }
static double ease_inout_quad(double t)  { return t < 0.5 ? 2.0*t*t : 1.0 - std::pow(-2.0*t + 2.0, 2) * 0.5; }

static double ease_in_cubic(double t)    { return t * t * t; }
static double ease_out_cubic(double t)   { return 1.0 - std::pow(1.0 - t, 3); }
static double ease_inout_cubic(double t) { return t < 0.5 ? 4.0*t*t*t : 1.0 - std::pow(-2.0*t + 2.0, 3) * 0.5; }

static double ease_in_quart(double t)    { return t*t*t*t; }
static double ease_out_quart(double t)   { return 1.0 - std::pow(1.0 - t, 4); }
static double ease_inout_quart(double t) { return t < 0.5 ? 8.0*t*t*t*t : 1.0 - std::pow(-2.0*t + 2.0, 4) * 0.5; }

static double ease_in_quint(double t)    { return t*t*t*t*t; }
static double ease_out_quint(double t)   { return 1.0 - std::pow(1.0 - t, 5); }
static double ease_inout_quint(double t) { return t < 0.5 ? 16.0*t*t*t*t*t : 1.0 - std::pow(-2.0*t+2.0,5)*0.5; }

static double ease_in_expo(double t)    { return t == 0.0 ? 0.0 : std::pow(2.0, 10.0*t - 10.0); }
static double ease_out_expo(double t)   { return t == 1.0 ? 1.0 : 1.0 - std::pow(2.0, -10.0*t); }
static double ease_inout_expo(double t) {
    if (t == 0.0) return 0.0;
    if (t == 1.0) return 1.0;
    return t < 0.5 ? std::pow(2.0, 20.0*t - 10.0) * 0.5
                   : (2.0 - std::pow(2.0, -20.0*t + 10.0)) * 0.5;
}

static double ease_in_circ(double t)    { return 1.0 - std::sqrt(1.0 - t*t); }
static double ease_out_circ(double t)   { return std::sqrt(1.0 - std::pow(t - 1.0, 2)); }
static double ease_inout_circ(double t) {
    return t < 0.5 ? (1.0 - std::sqrt(1.0 - 4.0*t*t)) * 0.5
                   : (std::sqrt(1.0 - std::pow(-2.0*t + 2.0, 2)) + 1.0) * 0.5;
}

static double ease_in_back(double t) {
    const double c1 = 1.70158, c3 = c1 + 1.0;
    return c3*t*t*t - c1*t*t;
}
static double ease_out_back(double t) {
    const double c1 = 1.70158, c3 = c1 + 1.0;
    return 1.0 + c3*std::pow(t-1.0,3) + c1*std::pow(t-1.0,2);
}
static double ease_inout_back(double t) {
    const double c1 = 1.70158, c2 = c1*1.525;
    return t < 0.5 ? (std::pow(2.0*t,2)*((c2+1.0)*2.0*t - c2)) * 0.5
                   : (std::pow(2.0*t-2.0,2)*((c2+1.0)*(2.0*t-2.0) + c2) + 2.0) * 0.5;
}

static double ease_out_bounce(double t) {
    const double n1 = 7.5625, d1 = 2.75;
    if (t < 1.0/d1)        return n1*t*t;
    if (t < 2.0/d1)       { t -= 1.5/d1;   return n1*t*t + 0.75; }
    if (t < 2.5/d1)       { t -= 2.25/d1;  return n1*t*t + 0.9375; }
    t -= 2.625/d1; return n1*t*t + 0.984375;
}
static double ease_in_bounce(double t)    { return 1.0 - ease_out_bounce(1.0 - t); }
static double ease_inout_bounce(double t) {
    return t < 0.5 ? (1.0 - ease_out_bounce(1.0 - 2.0*t)) * 0.5
                   : (1.0 + ease_out_bounce(2.0*t - 1.0)) * 0.5;
}

static double ease_in_elastic(double t) {
    if (t == 0.0 || t == 1.0) return t;
    const double c4 = 2.0*M_PI / 3.0;
    return -std::pow(2.0, 10.0*t-10.0) * std::sin((t*10.0-10.75)*c4);
}
static double ease_out_elastic(double t) {
    if (t == 0.0 || t == 1.0) return t;
    const double c4 = 2.0*M_PI / 3.0;
    return std::pow(2.0, -10.0*t) * std::sin((t*10.0-0.75)*c4) + 1.0;
}
static double ease_inout_elastic(double t) {
    if (t == 0.0 || t == 1.0) return t;
    const double c5 = 2.0*M_PI / 4.5;
    return t < 0.5
        ? -(std::pow(2.0, 20.0*t-10.0) * std::sin((20.0*t-11.125)*c5)) * 0.5
        : (std::pow(2.0, -20.0*t+10.0) * std::sin((20.0*t-11.125)*c5)) * 0.5 + 1.0;
}

// "EASE" = css ease ≈ cubic-bezier(0.25, 0.1, 0.25, 1.0) — approximate with ease_inout_cubic
static double ease_ease(double t) { return ease_inout_cubic(t); }
// "EASEIN" = css ease-in ≈ cubic-bezier(0.42, 0, 1.0, 1.0)
static double ease_easein(double t)  { return ease_in_cubic(t); }
// "EASEOUT" = css ease-out ≈ cubic-bezier(0, 0, 0.58, 1.0)
static double ease_easeout(double t) { return ease_out_cubic(t); }

double ease(double t, const std::string& name)
{
    // Clamp to [0,1] for safety
    t = std::max(0.0, std::min(1.0, t));

    // Convert to uppercase for comparison
    std::string n = name;
    for (auto& c : n) c = (char)::toupper((unsigned char)c);

    if (n == "LINEAR")                return ease_linear(t);
    if (n == "EASE")                  return ease_ease(t);
    if (n == "EASEIN")                return ease_easein(t);
    if (n == "EASEOUT")               return ease_easeout(t);
    if (n == "EASEINQUAD")            return ease_in_quad(t);
    if (n == "EASEOUTQUAD")           return ease_out_quad(t);
    if (n == "EASEINOUTQUAD")         return ease_inout_quad(t);
    if (n == "EASEINCUBIC")           return ease_in_cubic(t);
    if (n == "EASEOUTCUBIC")          return ease_out_cubic(t);
    if (n == "EASEINOUTCUBIC")        return ease_inout_cubic(t);
    if (n == "EASEINQUART")           return ease_in_quart(t);
    if (n == "EASEOUTQUART")          return ease_out_quart(t);
    if (n == "EASEINOUTQUART")        return ease_inout_quart(t);
    if (n == "EASEINQUINT")           return ease_in_quint(t);
    if (n == "EASEOUTQUINT")          return ease_out_quint(t);
    if (n == "EASEINOUTQUINT")        return ease_inout_quint(t);
    if (n == "EASEINSINE")            return ease_in_sine(t);
    if (n == "EASEOUTSINE")           return ease_out_sine(t);
    if (n == "EASEINOUTSINE")         return ease_inout_sine(t);
    if (n == "EASEINEXPO")            return ease_in_expo(t);
    if (n == "EASEOUTEXPO")           return ease_out_expo(t);
    if (n == "EASEINOUTEXPO")         return ease_inout_expo(t);
    if (n == "EASEINCIRC")            return ease_in_circ(t);
    if (n == "EASEOUTCIRC")           return ease_out_circ(t);
    if (n == "EASEINOUTCIRC")         return ease_inout_circ(t);
    if (n == "EASEINBACK")            return ease_in_back(t);
    if (n == "EASEOUTBACK")           return ease_out_back(t);
    if (n == "EASEINOUTBACK")         return ease_inout_back(t);
    if (n == "EASEINBOUNCE")          return ease_in_bounce(t);
    if (n == "EASEOUTBOUNCE")         return ease_out_bounce(t);
    if (n == "EASEINOUTBOUNCE")       return ease_inout_bounce(t);
    // CasparCG uses EASEINELESTIC (typo in original), handle both spellings
    if (n == "EASEINELESTIC" || n == "EASEINELASTIC")   return ease_in_elastic(t);
    if (n == "EASEOUTELASTIC")        return ease_out_elastic(t);
    if (n == "EASEINOUTELASTIC")      return ease_inout_elastic(t);

    // Unknown easing — fall back to linear
    return t;
}

// ---------------------------------------------------------------------------
// Helper macro to lerp a single double field
// ---------------------------------------------------------------------------
#define LERP_F(field) a.field + (b.field - a.field) * t

kf_state lerp_state(const kf_state& a, const kf_state& b, double t)
{
    // t is raw progress [0,1]; easing is applied by the caller before lerp_state
    kf_state r;
    r.opacity    = LERP_F(opacity);
    r.contrast   = LERP_F(contrast);
    r.brightness = LERP_F(brightness);
    r.saturation = LERP_F(saturation);

    r.anchor_x  = LERP_F(anchor_x);   r.anchor_y  = LERP_F(anchor_y);
    r.fill_x    = LERP_F(fill_x);     r.fill_y    = LERP_F(fill_y);
    r.fill_sx   = LERP_F(fill_sx);    r.fill_sy   = LERP_F(fill_sy);
    r.angle     = LERP_F(angle);
    r.crop_ul_x = LERP_F(crop_ul_x);  r.crop_ul_y = LERP_F(crop_ul_y);
    r.crop_lr_x = LERP_F(crop_lr_x);  r.crop_lr_y = LERP_F(crop_lr_y);

    r.persp_ul_x = LERP_F(persp_ul_x); r.persp_ul_y = LERP_F(persp_ul_y);
    r.persp_ur_x = LERP_F(persp_ur_x); r.persp_ur_y = LERP_F(persp_ur_y);
    r.persp_lr_x = LERP_F(persp_lr_x); r.persp_lr_y = LERP_F(persp_lr_y);
    r.persp_ll_x = LERP_F(persp_ll_x); r.persp_ll_y = LERP_F(persp_ll_y);

    // Projection: lerp numeric values (enabled state is also smoothed)
    r.proj_enable   = LERP_F(proj_enable);
    r.proj_yaw      = LERP_F(proj_yaw);
    r.proj_pitch    = LERP_F(proj_pitch);
    r.proj_roll     = LERP_F(proj_roll);
    r.proj_fov      = LERP_F(proj_fov);
    r.proj_offset_x = LERP_F(proj_offset_x);
    r.proj_offset_y = LERP_F(proj_offset_y);

    r.temperature = LERP_F(temperature); r.tint       = LERP_F(tint);
    r.shadows     = LERP_F(shadows);     r.highlights = LERP_F(highlights);

    r.lift_r = LERP_F(lift_r); r.lift_g = LERP_F(lift_g); r.lift_b = LERP_F(lift_b);
    r.mid_r  = LERP_F(mid_r);  r.mid_g  = LERP_F(mid_g);  r.mid_b  = LERP_F(mid_b);
    r.gain_r = LERP_F(gain_r); r.gain_g = LERP_F(gain_g); r.gain_b = LERP_F(gain_b);

    r.hue_shift = LERP_F(hue_shift);
    r.invert    = LERP_F(invert);

    r.levels_min_in  = LERP_F(levels_min_in);  r.levels_max_in  = LERP_F(levels_max_in);
    r.levels_gamma   = LERP_F(levels_gamma);
    r.levels_min_out = LERP_F(levels_min_out); r.levels_max_out = LERP_F(levels_max_out);

    r.rgb_r_min_in = LERP_F(rgb_r_min_in); r.rgb_r_max_in = LERP_F(rgb_r_max_in);
    r.rgb_r_gamma  = LERP_F(rgb_r_gamma);
    r.rgb_r_min_out= LERP_F(rgb_r_min_out);r.rgb_r_max_out= LERP_F(rgb_r_max_out);

    r.rgb_g_min_in = LERP_F(rgb_g_min_in); r.rgb_g_max_in = LERP_F(rgb_g_max_in);
    r.rgb_g_gamma  = LERP_F(rgb_g_gamma);
    r.rgb_g_min_out= LERP_F(rgb_g_min_out);r.rgb_g_max_out= LERP_F(rgb_g_max_out);

    r.rgb_b_min_in = LERP_F(rgb_b_min_in); r.rgb_b_max_in = LERP_F(rgb_b_max_in);
    r.rgb_b_gamma  = LERP_F(rgb_b_gamma);
    r.rgb_b_min_out= LERP_F(rgb_b_min_out);r.rgb_b_max_out= LERP_F(rgb_b_max_out);

    r.blur_radius   = LERP_F(blur_radius);
    r.blur_angle    = LERP_F(blur_angle);
    r.blur_center_x = LERP_F(blur_center_x); r.blur_center_y = LERP_F(blur_center_y);
    r.blur_tilt_y   = LERP_F(blur_tilt_y);   r.blur_tilt_h   = LERP_F(blur_tilt_h);

    return r;
#undef LERP_F
}

// ---------------------------------------------------------------------------
// apply_state  — write kf_state values into an image_transform
// ---------------------------------------------------------------------------

void apply_state(const kf_state& s, core::image_transform& tf)
{
    static const double DEG2RAD = 3.141592653589793 / 180.0;

    tf.opacity    = s.opacity;
    tf.contrast   = s.contrast;
    tf.brightness = s.brightness;
    tf.saturation = s.saturation;

    tf.enable_geometry_modifiers = true;

    tf.anchor           = {s.anchor_x, s.anchor_y};
    tf.fill_translation = {s.fill_x,   s.fill_y};
    tf.fill_scale       = {s.fill_sx,  s.fill_sy};
    tf.angle            = s.angle * DEG2RAD;  // JSON stores degrees; image_transform expects radians

    tf.crop.ul = {s.crop_ul_x, s.crop_ul_y};
    tf.crop.lr = {s.crop_lr_x, s.crop_lr_y};

    tf.perspective.ul = {s.persp_ul_x, s.persp_ul_y};
    tf.perspective.ur = {s.persp_ur_x, s.persp_ur_y};
    tf.perspective.lr = {s.persp_lr_x, s.persp_lr_y};
    tf.perspective.ll = {s.persp_ll_x, s.persp_ll_y};

    // Projection: AMCP MIXER PROJECTION uses degrees; image_transform stores radians.
    tf.projection.enable   = s.proj_enable >= 0.5;
    tf.projection.yaw      = s.proj_yaw   * DEG2RAD;
    tf.projection.pitch    = s.proj_pitch * DEG2RAD;
    tf.projection.roll     = s.proj_roll  * DEG2RAD;
    tf.projection.fov      = s.proj_fov   * DEG2RAD;
    tf.projection.offset_x = s.proj_offset_x;
    tf.projection.offset_y = s.proj_offset_y;

    tf.temperature = s.temperature;
    tf.tint        = s.tint;
    tf.shadows     = s.shadows;
    tf.highlights  = s.highlights;

    tf.lift    = {s.lift_r, s.lift_g, s.lift_b};
    tf.midtone = {s.mid_r,  s.mid_g,  s.mid_b};
    tf.gain    = {s.gain_r, s.gain_g, s.gain_b};

    tf.hue_shift = s.hue_shift;
    tf.invert    = s.invert >= 0.5;

    tf.levels.min_input  = s.levels_min_in;
    tf.levels.max_input  = s.levels_max_in;
    tf.levels.gamma      = s.levels_gamma;
    tf.levels.min_output = s.levels_min_out;
    tf.levels.max_output = s.levels_max_out;

    tf.per_channel_levels.enable    = true;
    tf.per_channel_levels.r.min_input  = s.rgb_r_min_in;
    tf.per_channel_levels.r.max_input  = s.rgb_r_max_in;
    tf.per_channel_levels.r.gamma      = s.rgb_r_gamma;
    tf.per_channel_levels.r.min_output = s.rgb_r_min_out;
    tf.per_channel_levels.r.max_output = s.rgb_r_max_out;

    tf.per_channel_levels.g.min_input  = s.rgb_g_min_in;
    tf.per_channel_levels.g.max_input  = s.rgb_g_max_in;
    tf.per_channel_levels.g.gamma      = s.rgb_g_gamma;
    tf.per_channel_levels.g.min_output = s.rgb_g_min_out;
    tf.per_channel_levels.g.max_output = s.rgb_g_max_out;

    tf.per_channel_levels.b.min_input  = s.rgb_b_min_in;
    tf.per_channel_levels.b.max_input  = s.rgb_b_max_in;
    tf.per_channel_levels.b.gamma      = s.rgb_b_gamma;
    tf.per_channel_levels.b.min_output = s.rgb_b_min_out;
    tf.per_channel_levels.b.max_output = s.rgb_b_max_out;

    tf.blur.enable   = (s.blur_radius > 0.0);
    tf.blur.radius   = s.blur_radius;
    tf.blur.angle    = s.blur_angle;
    tf.blur.center   = {s.blur_center_x, s.blur_center_y};
    tf.blur.tilt_y   = s.blur_tilt_y;
    tf.blur.tilt_h   = s.blur_tilt_h;
}

// ---------------------------------------------------------------------------
// capture_state — read image_transform values into a kf_state
// ---------------------------------------------------------------------------

kf_state capture_state(const core::image_transform& tf)
{
    static const double RAD2DEG = 180.0 / 3.141592653589793;

    kf_state s;
    s.opacity    = tf.opacity;
    s.contrast   = tf.contrast;
    s.brightness = tf.brightness;
    s.saturation = tf.saturation;

    s.anchor_x  = tf.anchor[0];           s.anchor_y  = tf.anchor[1];
    s.fill_x    = tf.fill_translation[0]; s.fill_y    = tf.fill_translation[1];
    s.fill_sx   = tf.fill_scale[0];       s.fill_sy   = tf.fill_scale[1];
    s.angle     = tf.angle * RAD2DEG;  // image_transform stores radians; JSON uses degrees

    s.crop_ul_x = tf.crop.ul[0]; s.crop_ul_y = tf.crop.ul[1];
    s.crop_lr_x = tf.crop.lr[0]; s.crop_lr_y = tf.crop.lr[1];

    s.persp_ul_x = tf.perspective.ul[0]; s.persp_ul_y = tf.perspective.ul[1];
    s.persp_ur_x = tf.perspective.ur[0]; s.persp_ur_y = tf.perspective.ur[1];
    s.persp_lr_x = tf.perspective.lr[0]; s.persp_lr_y = tf.perspective.lr[1];
    s.persp_ll_x = tf.perspective.ll[0]; s.persp_ll_y = tf.perspective.ll[1];

    s.proj_enable   = tf.projection.enable ? 1.0 : 0.0;
    // image_transform stores projection angles in radians; JSON uses degrees (matching AMCP)
    s.proj_yaw      = tf.projection.yaw   * RAD2DEG;
    s.proj_pitch    = tf.projection.pitch * RAD2DEG;
    s.proj_roll     = tf.projection.roll  * RAD2DEG;
    s.proj_fov      = tf.projection.fov   * RAD2DEG;
    s.proj_offset_x = tf.projection.offset_x;
    s.proj_offset_y = tf.projection.offset_y;

    s.temperature = tf.temperature; s.tint       = tf.tint;
    s.shadows     = tf.shadows;     s.highlights = tf.highlights;

    s.lift_r = tf.lift[0]; s.lift_g = tf.lift[1]; s.lift_b = tf.lift[2];
    s.mid_r  = tf.midtone[0]; s.mid_g = tf.midtone[1]; s.mid_b = tf.midtone[2];
    s.gain_r = tf.gain[0]; s.gain_g = tf.gain[1]; s.gain_b = tf.gain[2];

    s.hue_shift = tf.hue_shift;
    s.invert    = tf.invert ? 1.0 : 0.0;

    s.levels_min_in  = tf.levels.min_input;  s.levels_max_in  = tf.levels.max_input;
    s.levels_gamma   = tf.levels.gamma;
    s.levels_min_out = tf.levels.min_output; s.levels_max_out = tf.levels.max_output;

    s.rgb_r_min_in  = tf.per_channel_levels.r.min_input;
    s.rgb_r_max_in  = tf.per_channel_levels.r.max_input;
    s.rgb_r_gamma   = tf.per_channel_levels.r.gamma;
    s.rgb_r_min_out = tf.per_channel_levels.r.min_output;
    s.rgb_r_max_out = tf.per_channel_levels.r.max_output;

    s.rgb_g_min_in  = tf.per_channel_levels.g.min_input;
    s.rgb_g_max_in  = tf.per_channel_levels.g.max_input;
    s.rgb_g_gamma   = tf.per_channel_levels.g.gamma;
    s.rgb_g_min_out = tf.per_channel_levels.g.min_output;
    s.rgb_g_max_out = tf.per_channel_levels.g.max_output;

    s.rgb_b_min_in  = tf.per_channel_levels.b.min_input;
    s.rgb_b_max_in  = tf.per_channel_levels.b.max_input;
    s.rgb_b_gamma   = tf.per_channel_levels.b.gamma;
    s.rgb_b_min_out = tf.per_channel_levels.b.min_output;
    s.rgb_b_max_out = tf.per_channel_levels.b.max_output;

    s.blur_radius   = tf.blur.radius;
    s.blur_angle    = tf.blur.angle;
    s.blur_center_x = tf.blur.center[0]; s.blur_center_y = tf.blur.center[1];
    s.blur_tilt_y   = tf.blur.tilt_y;    s.blur_tilt_h   = tf.blur.tilt_h;

    return s;
}

// ---------------------------------------------------------------------------
// keyframe_timeline
// ---------------------------------------------------------------------------

void keyframe_timeline::add(keyframe_t kf)
{
    // Remove any existing keyframe at the exact same time (±1ms)
    kfs_.erase(std::remove_if(kfs_.begin(), kfs_.end(),
        [&](const keyframe_t& k){ return std::abs(k.time_secs - kf.time_secs) < 0.001; }),
        kfs_.end());
    kfs_.push_back(std::move(kf));
    std::sort(kfs_.begin(), kfs_.end(),
        [](const keyframe_t& a, const keyframe_t& b){ return a.time_secs < b.time_secs; });
}

void keyframe_timeline::remove(double time_secs)
{
    if (kfs_.empty()) return;
    // Find nearest keyframe
    auto it = std::min_element(kfs_.begin(), kfs_.end(),
        [&](const keyframe_t& a, const keyframe_t& b){
            return std::abs(a.time_secs - time_secs) < std::abs(b.time_secs - time_secs);
        });
    kfs_.erase(it);
}

void keyframe_timeline::clear()   { kfs_.clear(); }
bool keyframe_timeline::empty()  const { return kfs_.empty(); }
size_t keyframe_timeline::size() const { return kfs_.size(); }
const std::vector<keyframe_t>& keyframe_timeline::keyframes() const { return kfs_; }

bool keyframe_timeline::patch_at_time(double time_secs, const kf_state& patched_state)
{
    for (auto& kf : kfs_) {
        if (std::abs(kf.time_secs - time_secs) < 0.001) {
            kf.state = patched_state;
            return true;
        }
    }
    return false;
}

kf_state keyframe_timeline::interpolate(double time_secs) const
{
    if (kfs_.empty())
        return kf_state{};

    // Before first keyframe — hold first
    if (time_secs <= kfs_.front().time_secs)
        return kfs_.front().state;

    // After last keyframe — hold last
    if (time_secs >= kfs_.back().time_secs)
        return kfs_.back().state;

    // Find bracket [A, B] where A.time <= time < B.time
    auto it_b = std::upper_bound(kfs_.begin(), kfs_.end(), time_secs,
        [](double t, const keyframe_t& k){ return t < k.time_secs; });
    auto it_a = std::prev(it_b);

    const keyframe_t& a = *it_a;
    const keyframe_t& b = *it_b;

    double seg_dur = b.time_secs - a.time_secs;
    double t_raw   = (seg_dur > 0.0) ? (time_secs - a.time_secs) / seg_dur : 0.0;

    // Apply easing from keyframe A's easing property
    double t_eased = ease(t_raw, a.easing);

    return lerp_state(a.state, b.state, t_eased);
}

}} // namespace caspar::keyframes
