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
#include "keyframe_fields.h"

#include <common/log.h>

#include <algorithm>
#include <cmath>
#include <mutex>
#include <set>
#include <unordered_map>
#include <unordered_set>

#ifndef M_PI
#define M_PI 3.141592653589793
#endif

namespace caspar { namespace keyframes {

// ═══════════════════════════════════════════════════════════════════════════
//  Easing functions
// ═══════════════════════════════════════════════════════════════════════════

static double ease_linear(double t) { return t; }

static double ease_in_sine(double t)     { return 1.0 - std::cos(t * M_PI * 0.5); }
static double ease_out_sine(double t)    { return std::sin(t * M_PI * 0.5); }
static double ease_inout_sine(double t)  { return -(std::cos(M_PI * t) - 1.0) * 0.5; }

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

static double ease_in_expo(double t)     { return t == 0.0 ? 0.0 : std::pow(2.0, 10.0*t - 10.0); }
static double ease_out_expo(double t)    { return t == 1.0 ? 1.0 : 1.0 - std::pow(2.0, -10.0*t); }
static double ease_inout_expo(double t)  {
    if (t == 0.0) return 0.0;
    if (t == 1.0) return 1.0;
    return t < 0.5 ? std::pow(2.0, 20.0*t - 10.0) * 0.5
                   : (2.0 - std::pow(2.0, -20.0*t + 10.0)) * 0.5;
}

static double ease_in_circ(double t)     { return 1.0 - std::sqrt(1.0 - t*t); }
static double ease_out_circ(double t)    { return std::sqrt(1.0 - std::pow(t - 1.0, 2)); }
static double ease_inout_circ(double t)  {
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

// ── Easing name → function pointer map (built once) ────────────────────

static const std::unordered_map<std::string, easing_fn_t>& easing_map()
{
    static const std::unordered_map<std::string, easing_fn_t> m = {
        {"LINEAR",           ease_linear},
        {"EASE",             ease_inout_cubic},
        {"EASEIN",           ease_in_cubic},
        {"EASEOUT",          ease_out_cubic},
        {"EASEINQUAD",       ease_in_quad},
        {"EASEOUTQUAD",      ease_out_quad},
        {"EASEINOUTQUAD",    ease_inout_quad},
        {"EASEINCUBIC",      ease_in_cubic},
        {"EASEOUTCUBIC",     ease_out_cubic},
        {"EASEINOUTCUBIC",   ease_inout_cubic},
        {"EASEINQUART",      ease_in_quart},
        {"EASEOUTQUART",     ease_out_quart},
        {"EASEINOUTQUART",   ease_inout_quart},
        {"EASEINQUINT",      ease_in_quint},
        {"EASEOUTQUINT",     ease_out_quint},
        {"EASEINOUTQUINT",   ease_inout_quint},
        {"EASEINSINE",       ease_in_sine},
        {"EASEOUTSINE",      ease_out_sine},
        {"EASEINOUTSINE",    ease_inout_sine},
        {"EASEINEXPO",       ease_in_expo},
        {"EASEOUTEXPO",      ease_out_expo},
        {"EASEINOUTEXPO",    ease_inout_expo},
        {"EASEINCIRC",       ease_in_circ},
        {"EASEOUTCIRC",      ease_out_circ},
        {"EASEINOUTCIRC",    ease_inout_circ},
        {"EASEINBACK",       ease_in_back},
        {"EASEOUTBACK",      ease_out_back},
        {"EASEINOUTBACK",    ease_inout_back},
        {"EASEINBOUNCE",     ease_in_bounce},
        {"EASEOUTBOUNCE",    ease_out_bounce},
        {"EASEINOUTBOUNCE",  ease_inout_bounce},
        {"EASEINELASTIC",    ease_in_elastic},
        {"EASEINELESTIC",    ease_in_elastic},  // legacy CasparCG typo
        {"EASEOUTELASTIC",   ease_out_elastic},
        {"EASEINOUTELASTIC", ease_inout_elastic},
    };
    return m;
}

easing_fn_t resolve_easing(const std::string& name)
{
    // Uppercase the name
    std::string upper;
    upper.reserve(name.size());
    for (char c : name)
        upper += static_cast<char>(::toupper(static_cast<unsigned char>(c)));

    const auto& m = easing_map();
    auto it = m.find(upper);
    if (it != m.end())
        return it->second;

    // Unknown easing — warn once per name, fall back to linear
    static std::mutex warned_mutex;
    static std::unordered_set<std::string> warned;
    {
        std::lock_guard<std::mutex> lk(warned_mutex);
        if (warned.insert(upper).second)
            CASPAR_LOG(warning) << L"[keyframes] Unknown easing \"" << std::wstring(upper.begin(), upper.end())
                                << L"\" — using LINEAR";
    }
    return ease_linear;
}

// ═══════════════════════════════════════════════════════════════════════════
//  keyframe_timeline
// ═══════════════════════════════════════════════════════════════════════════

void keyframe_timeline::add(keyframe_t kf)
{
    // Remove any existing keyframe at the exact same time (±1ms)
    kfs_.erase(std::remove_if(kfs_.begin(), kfs_.end(),
        [&](const keyframe_t& k) { return std::abs(k.time_secs - kf.time_secs) < 0.001; }),
        kfs_.end());
    kfs_.push_back(std::move(kf));
    std::sort(kfs_.begin(), kfs_.end(),
        [](const keyframe_t& a, const keyframe_t& b) { return a.time_secs < b.time_secs; });
    rebuild_field_index();
}

bool keyframe_timeline::remove(double time_secs, double max_distance)
{
    if (kfs_.empty())
        return false;
    auto it = std::min_element(kfs_.begin(), kfs_.end(),
        [&](const keyframe_t& a, const keyframe_t& b) {
            return std::abs(a.time_secs - time_secs) < std::abs(b.time_secs - time_secs);
        });
    if (std::abs(it->time_secs - time_secs) > max_distance)
        return false;
    kfs_.erase(it);
    rebuild_field_index();
    return true;
}

void   keyframe_timeline::clear()         { kfs_.clear(); all_field_names_.clear(); }
bool   keyframe_timeline::empty()  const  { return kfs_.empty(); }
size_t keyframe_timeline::size()   const  { return kfs_.size(); }
const std::vector<keyframe_t>& keyframe_timeline::keyframes() const { return kfs_; }

bool keyframe_timeline::patch_at_time(double time_secs, const kf_values& patch)
{
    for (auto& kf : kfs_) {
        if (std::abs(kf.time_secs - time_secs) < 0.001) {
            for (const auto& [k, v] : patch)
                kf.values[k] = v;
            rebuild_field_index();
            return true;
        }
    }
    return false;
}

kf_values keyframe_timeline::interpolate(double time_secs) const
{
    if (kfs_.empty())
        return {};

    kf_values result;

    for (const auto& field_name : all_field_names_) {
        // Find the last keyframe at-or-before `time_secs` that contains this field
        const keyframe_t* kf_before = nullptr;
        // Find the first keyframe after `time_secs` that contains this field
        const keyframe_t* kf_after = nullptr;

        for (const auto& kf : kfs_) {
            auto it = kf.values.find(field_name);
            if (it == kf.values.end())
                continue;
            if (kf.time_secs <= time_secs)
                kf_before = &kf;
            else if (!kf_after)
                kf_after = &kf;
        }

        if (kf_before && kf_after) {
            double a_val = kf_before->values.at(field_name);
            double b_val = kf_after->values.at(field_name);

            // Look up field kind for specialized interpolation
            const auto* fd = kf_find_field(field_name);
            field_kind  fk = fd ? fd->kind : field_kind::continuous;

            if (fk == field_kind::discrete) {
                // Discrete fields hold the source value for the entire segment
                result[field_name] = a_val;
            } else {
                double seg_dur = kf_after->time_secs - kf_before->time_secs;
                double t_raw = (seg_dur > 0.0) ? (time_secs - kf_before->time_secs) / seg_dur : 0.0;
                t_raw = std::max(0.0, std::min(1.0, t_raw));

                // Apply easing from the "before" keyframe
                easing_fn_t efn = kf_before->easing_fn ? kf_before->easing_fn : ease_linear;
                double t_eased = efn(t_raw);

                if (fk == field_kind::angular) {
                    // Shortest-path wrapping at 360°
                    double diff = std::fmod(b_val - a_val, 360.0);
                    if (diff > 180.0)
                        diff -= 360.0;
                    else if (diff < -180.0)
                        diff += 360.0;
                    result[field_name] = a_val + diff * t_eased;
                } else {
                    // Standard linear interpolation
                    result[field_name] = a_val + (b_val - a_val) * t_eased;
                }
            }
        } else if (kf_before) {
            // Hold at last known value
            result[field_name] = kf_before->values.at(field_name);
        } else if (kf_after) {
            // Pre-roll: hold at first known value
            result[field_name] = kf_after->values.at(field_name);
        }
    }

    return result;
}

void keyframe_timeline::rebuild_field_index()
{
    std::set<std::string> names;
    for (const auto& kf : kfs_)
        for (const auto& [k, v] : kf.values)
            names.insert(k);
    all_field_names_.assign(names.begin(), names.end());
}

}} // namespace caspar::keyframes
