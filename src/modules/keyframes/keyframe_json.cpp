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

#include "keyframe_json.h"

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include <sstream>
#include <stdexcept>

namespace caspar { namespace keyframes {

namespace pt = boost::property_tree;

// ── Helpers ─────────────────────────────────────────────────────────────────

static double get_d(const pt::ptree& node, const std::string& key, double def)
{
    return node.get<double>(key, def);
}

static void put_d(pt::ptree& node, const std::string& key, double val)
{
    node.put(key, val);
}

// ── kf_state → ptree ────────────────────────────────────────────────────────

static pt::ptree state_to_ptree(const kf_state& s)
{
    pt::ptree n;
    // Basic
    put_d(n, "opacity",    s.opacity);
    put_d(n, "contrast",   s.contrast);
    put_d(n, "brightness", s.brightness);
    put_d(n, "saturation", s.saturation);
    // Geometry
    put_d(n, "anchor_x",  s.anchor_x);  put_d(n, "anchor_y",  s.anchor_y);
    put_d(n, "fill_x",    s.fill_x);    put_d(n, "fill_y",    s.fill_y);
    put_d(n, "fill_sx",   s.fill_sx);   put_d(n, "fill_sy",   s.fill_sy);
    put_d(n, "angle",     s.angle);
    put_d(n, "crop_ul_x", s.crop_ul_x); put_d(n, "crop_ul_y", s.crop_ul_y);
    put_d(n, "crop_lr_x", s.crop_lr_x); put_d(n, "crop_lr_y", s.crop_lr_y);
    // Perspective
    put_d(n, "persp_ul_x", s.persp_ul_x); put_d(n, "persp_ul_y", s.persp_ul_y);
    put_d(n, "persp_ur_x", s.persp_ur_x); put_d(n, "persp_ur_y", s.persp_ur_y);
    put_d(n, "persp_lr_x", s.persp_lr_x); put_d(n, "persp_lr_y", s.persp_lr_y);
    put_d(n, "persp_ll_x", s.persp_ll_x); put_d(n, "persp_ll_y", s.persp_ll_y);
    // Projection
    put_d(n, "proj_enable",   s.proj_enable);
    put_d(n, "proj_yaw",      s.proj_yaw);      put_d(n, "proj_pitch", s.proj_pitch);
    put_d(n, "proj_roll",     s.proj_roll);      put_d(n, "proj_fov",   s.proj_fov);
    put_d(n, "proj_offset_x", s.proj_offset_x); put_d(n, "proj_offset_y", s.proj_offset_y);
    // White / Tone balance
    put_d(n, "temperature", s.temperature); put_d(n, "tint",       s.tint);
    put_d(n, "shadows",     s.shadows);     put_d(n, "highlights", s.highlights);
    // 3-Way
    put_d(n, "lift_r",  s.lift_r);  put_d(n, "lift_g",  s.lift_g);  put_d(n, "lift_b",  s.lift_b);
    put_d(n, "mid_r",   s.mid_r);   put_d(n, "mid_g",   s.mid_g);   put_d(n, "mid_b",   s.mid_b);
    put_d(n, "gain_r",  s.gain_r);  put_d(n, "gain_g",  s.gain_g);  put_d(n, "gain_b",  s.gain_b);
    // Hue / Invert
    put_d(n, "hue_shift", s.hue_shift);
    put_d(n, "invert",    s.invert);
    // Levels (master)
    put_d(n, "levels_min_in",  s.levels_min_in);  put_d(n, "levels_max_in",  s.levels_max_in);
    put_d(n, "levels_gamma",   s.levels_gamma);
    put_d(n, "levels_min_out", s.levels_min_out); put_d(n, "levels_max_out", s.levels_max_out);
    // RGB levels
    put_d(n, "rgb_r_min_in",  s.rgb_r_min_in);  put_d(n, "rgb_r_max_in",  s.rgb_r_max_in);
    put_d(n, "rgb_r_gamma",   s.rgb_r_gamma);
    put_d(n, "rgb_r_min_out", s.rgb_r_min_out); put_d(n, "rgb_r_max_out", s.rgb_r_max_out);
    put_d(n, "rgb_g_min_in",  s.rgb_g_min_in);  put_d(n, "rgb_g_max_in",  s.rgb_g_max_in);
    put_d(n, "rgb_g_gamma",   s.rgb_g_gamma);
    put_d(n, "rgb_g_min_out", s.rgb_g_min_out); put_d(n, "rgb_g_max_out", s.rgb_g_max_out);
    put_d(n, "rgb_b_min_in",  s.rgb_b_min_in);  put_d(n, "rgb_b_max_in",  s.rgb_b_max_in);
    put_d(n, "rgb_b_gamma",   s.rgb_b_gamma);
    put_d(n, "rgb_b_min_out", s.rgb_b_min_out); put_d(n, "rgb_b_max_out", s.rgb_b_max_out);
    // Blur
    put_d(n, "blur_radius",   s.blur_radius);
    put_d(n, "blur_angle",    s.blur_angle);
    put_d(n, "blur_center_x", s.blur_center_x); put_d(n, "blur_center_y", s.blur_center_y);
    put_d(n, "blur_tilt_y",   s.blur_tilt_y);   put_d(n, "blur_tilt_h",   s.blur_tilt_h);
    return n;
}

// ── ptree → kf_state ────────────────────────────────────────────────────────

static kf_state ptree_to_state(const pt::ptree& n)
{
    kf_state s;
    // Basic
    s.opacity    = get_d(n, "opacity",    1.0);
    s.contrast   = get_d(n, "contrast",   1.0);
    s.brightness = get_d(n, "brightness", 1.0);
    s.saturation = get_d(n, "saturation", 1.0);
    // Geometry
    s.anchor_x  = get_d(n, "anchor_x",  0.0); s.anchor_y  = get_d(n, "anchor_y",  0.0);
    s.fill_x    = get_d(n, "fill_x",    0.0); s.fill_y    = get_d(n, "fill_y",    0.0);
    s.fill_sx   = get_d(n, "fill_sx",   1.0); s.fill_sy   = get_d(n, "fill_sy",   1.0);
    s.angle     = get_d(n, "angle",     0.0);
    s.crop_ul_x = get_d(n, "crop_ul_x", 0.0); s.crop_ul_y = get_d(n, "crop_ul_y", 0.0);
    s.crop_lr_x = get_d(n, "crop_lr_x", 1.0); s.crop_lr_y = get_d(n, "crop_lr_y", 1.0);
    // Perspective
    s.persp_ul_x = get_d(n, "persp_ul_x", 0.0); s.persp_ul_y = get_d(n, "persp_ul_y", 0.0);
    s.persp_ur_x = get_d(n, "persp_ur_x", 1.0); s.persp_ur_y = get_d(n, "persp_ur_y", 0.0);
    s.persp_lr_x = get_d(n, "persp_lr_x", 1.0); s.persp_lr_y = get_d(n, "persp_lr_y", 1.0);
    s.persp_ll_x = get_d(n, "persp_ll_x", 0.0); s.persp_ll_y = get_d(n, "persp_ll_y", 1.0);
    // Projection
    s.proj_enable   = get_d(n, "proj_enable", 0.0);
    s.proj_yaw      = get_d(n, "proj_yaw",      0.0); s.proj_pitch = get_d(n, "proj_pitch", 0.0);
    s.proj_roll     = get_d(n, "proj_roll",     0.0); s.proj_fov   = get_d(n, "proj_fov", 90.0);  // degrees
    s.proj_offset_x = get_d(n, "proj_offset_x", 0.0); s.proj_offset_y = get_d(n, "proj_offset_y", 0.0);
    // White / Tone balance
    s.temperature = get_d(n, "temperature", 0.0); s.tint       = get_d(n, "tint",       0.0);
    s.shadows     = get_d(n, "shadows",     0.0); s.highlights = get_d(n, "highlights", 0.0);
    // 3-Way
    s.lift_r = get_d(n, "lift_r", 0.0); s.lift_g = get_d(n, "lift_g", 0.0); s.lift_b = get_d(n, "lift_b", 0.0);
    s.mid_r  = get_d(n, "mid_r",  1.0); s.mid_g  = get_d(n, "mid_g",  1.0); s.mid_b  = get_d(n, "mid_b",  1.0);
    s.gain_r = get_d(n, "gain_r", 1.0); s.gain_g = get_d(n, "gain_g", 1.0); s.gain_b = get_d(n, "gain_b", 1.0);
    // Hue / Invert
    s.hue_shift = get_d(n, "hue_shift", 0.0);
    s.invert    = get_d(n, "invert",    0.0);
    // Levels (master)
    s.levels_min_in  = get_d(n, "levels_min_in",  0.0); s.levels_max_in  = get_d(n, "levels_max_in",  1.0);
    s.levels_gamma   = get_d(n, "levels_gamma",   1.0);
    s.levels_min_out = get_d(n, "levels_min_out", 0.0); s.levels_max_out = get_d(n, "levels_max_out", 1.0);
    // RGB levels
    s.rgb_r_min_in  = get_d(n, "rgb_r_min_in",  0.0); s.rgb_r_max_in  = get_d(n, "rgb_r_max_in",  1.0);
    s.rgb_r_gamma   = get_d(n, "rgb_r_gamma",   1.0);
    s.rgb_r_min_out = get_d(n, "rgb_r_min_out", 0.0); s.rgb_r_max_out = get_d(n, "rgb_r_max_out", 1.0);
    s.rgb_g_min_in  = get_d(n, "rgb_g_min_in",  0.0); s.rgb_g_max_in  = get_d(n, "rgb_g_max_in",  1.0);
    s.rgb_g_gamma   = get_d(n, "rgb_g_gamma",   1.0);
    s.rgb_g_min_out = get_d(n, "rgb_g_min_out", 0.0); s.rgb_g_max_out = get_d(n, "rgb_g_max_out", 1.0);
    s.rgb_b_min_in  = get_d(n, "rgb_b_min_in",  0.0); s.rgb_b_max_in  = get_d(n, "rgb_b_max_in",  1.0);
    s.rgb_b_gamma   = get_d(n, "rgb_b_gamma",   1.0);
    s.rgb_b_min_out = get_d(n, "rgb_b_min_out", 0.0); s.rgb_b_max_out = get_d(n, "rgb_b_max_out", 1.0);
    // Blur
    s.blur_radius   = get_d(n, "blur_radius",   0.0);
    s.blur_angle    = get_d(n, "blur_angle",    0.0);
    s.blur_center_x = get_d(n, "blur_center_x", 0.5); s.blur_center_y = get_d(n, "blur_center_y", 0.5);
    s.blur_tilt_y   = get_d(n, "blur_tilt_y",   0.5); s.blur_tilt_h   = get_d(n, "blur_tilt_h",   0.2);
    return s;
}

// ── Public API ───────────────────────────────────────────────────────────────

std::string timeline_to_json(const keyframe_timeline& tl)
{
    pt::ptree root;
    pt::ptree kf_array;

    for (const auto& kf : tl.keyframes()) {
        pt::ptree node = state_to_ptree(kf.state);
        node.put("time_secs", kf.time_secs);
        node.put("easing",    kf.easing);
        kf_array.push_back({"", node});
    }

    root.add_child("keyframes", kf_array);

    std::ostringstream oss;
    pt::write_json(oss, root);
    return oss.str();
}

keyframe_timeline json_to_timeline(const std::string& json)
{
    keyframe_timeline tl;

    // Boost.PropertyTree's JSON parser rejects bare empty arrays ("[]").
    // If the keyframes list is empty there is nothing to do — return early
    // rather than letting Boost throw "expected key string".
    if (json.find("\"keyframes\"") == std::string::npos ||
        json.find("\"keyframes\":[]") != std::string::npos ||
        json.find("\"keyframes\": []") != std::string::npos) {
        return tl;
    }

    try {
        pt::ptree root;
        std::istringstream iss(json);
        pt::read_json(iss, root);

        // get_child_optional avoids a throw when the array is absent / empty
        auto child = root.get_child_optional("keyframes");
        if (!child)
            return tl;

        std::size_t count = 0;
        for (const auto& kv : *child) {
            const pt::ptree& node = kv.second;
            keyframe_t kf;
            kf.time_secs = node.get<double>("time_secs", 0.0);
            kf.easing    = node.get<std::string>("easing", "LINEAR");
            kf.state     = ptree_to_state(node);
            tl.add(std::move(kf));
            ++count;
        }
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("KEYFRAMES JSON parse error: ") + e.what());
    }

    return tl;
}

kf_state patch_state_from_json(const kf_state& base, const std::string& json_str)
{
    kf_state s = base;  // start from existing values; only overwrite what's in the JSON

    pt::ptree root;
    std::istringstream iss(json_str);
    pt::read_json(iss, root);

    for (const auto& kv : root) {
        const std::string& k = kv.first;
        double v = kv.second.get_value<double>(0.0);
        // Basic
        if      (k == "opacity")        s.opacity        = v;
        else if (k == "contrast")       s.contrast       = v;
        else if (k == "brightness")     s.brightness     = v;
        else if (k == "saturation")     s.saturation     = v;
        // Geometry
        else if (k == "anchor_x")       s.anchor_x       = v;
        else if (k == "anchor_y")       s.anchor_y       = v;
        else if (k == "fill_x")         s.fill_x         = v;
        else if (k == "fill_y")         s.fill_y         = v;
        else if (k == "fill_sx")        s.fill_sx        = v;
        else if (k == "fill_sy")        s.fill_sy        = v;
        else if (k == "angle")          s.angle          = v;
        else if (k == "crop_ul_x")      s.crop_ul_x      = v;
        else if (k == "crop_ul_y")      s.crop_ul_y      = v;
        else if (k == "crop_lr_x")      s.crop_lr_x      = v;
        else if (k == "crop_lr_y")      s.crop_lr_y      = v;
        // Perspective
        else if (k == "persp_ul_x")     s.persp_ul_x     = v;
        else if (k == "persp_ul_y")     s.persp_ul_y     = v;
        else if (k == "persp_ur_x")     s.persp_ur_x     = v;
        else if (k == "persp_ur_y")     s.persp_ur_y     = v;
        else if (k == "persp_lr_x")     s.persp_lr_x     = v;
        else if (k == "persp_lr_y")     s.persp_lr_y     = v;
        else if (k == "persp_ll_x")     s.persp_ll_x     = v;
        else if (k == "persp_ll_y")     s.persp_ll_y     = v;
        // Projection
        else if (k == "proj_enable")    s.proj_enable    = v;
        else if (k == "proj_yaw")       s.proj_yaw       = v;
        else if (k == "proj_pitch")     s.proj_pitch     = v;
        else if (k == "proj_roll")      s.proj_roll      = v;
        else if (k == "proj_fov")       s.proj_fov       = v;
        else if (k == "proj_offset_x")  s.proj_offset_x  = v;
        else if (k == "proj_offset_y")  s.proj_offset_y  = v;
        // White / Tone balance
        else if (k == "temperature")    s.temperature    = v;
        else if (k == "tint")           s.tint           = v;
        else if (k == "shadows")        s.shadows        = v;
        else if (k == "highlights")     s.highlights     = v;
        // 3-Way
        else if (k == "lift_r")         s.lift_r         = v;
        else if (k == "lift_g")         s.lift_g         = v;
        else if (k == "lift_b")         s.lift_b         = v;
        else if (k == "mid_r")          s.mid_r          = v;
        else if (k == "mid_g")          s.mid_g          = v;
        else if (k == "mid_b")          s.mid_b          = v;
        else if (k == "gain_r")         s.gain_r         = v;
        else if (k == "gain_g")         s.gain_g         = v;
        else if (k == "gain_b")         s.gain_b         = v;
        // Hue / Invert
        else if (k == "hue_shift")      s.hue_shift      = v;
        else if (k == "invert")         s.invert         = v;
        // Levels (master)
        else if (k == "levels_min_in")  s.levels_min_in  = v;
        else if (k == "levels_max_in")  s.levels_max_in  = v;
        else if (k == "levels_gamma")   s.levels_gamma   = v;
        else if (k == "levels_min_out") s.levels_min_out = v;
        else if (k == "levels_max_out") s.levels_max_out = v;
        // RGB levels R
        else if (k == "rgb_r_min_in")   s.rgb_r_min_in   = v;
        else if (k == "rgb_r_max_in")   s.rgb_r_max_in   = v;
        else if (k == "rgb_r_gamma")    s.rgb_r_gamma    = v;
        else if (k == "rgb_r_min_out")  s.rgb_r_min_out  = v;
        else if (k == "rgb_r_max_out")  s.rgb_r_max_out  = v;
        // RGB levels G
        else if (k == "rgb_g_min_in")   s.rgb_g_min_in   = v;
        else if (k == "rgb_g_max_in")   s.rgb_g_max_in   = v;
        else if (k == "rgb_g_gamma")    s.rgb_g_gamma    = v;
        else if (k == "rgb_g_min_out")  s.rgb_g_min_out  = v;
        else if (k == "rgb_g_max_out")  s.rgb_g_max_out  = v;
        // RGB levels B
        else if (k == "rgb_b_min_in")   s.rgb_b_min_in   = v;
        else if (k == "rgb_b_max_in")   s.rgb_b_max_in   = v;
        else if (k == "rgb_b_gamma")    s.rgb_b_gamma    = v;
        else if (k == "rgb_b_min_out")  s.rgb_b_min_out  = v;
        else if (k == "rgb_b_max_out")  s.rgb_b_max_out  = v;
        // Blur
        else if (k == "blur_radius")    s.blur_radius    = v;
        else if (k == "blur_angle")     s.blur_angle     = v;
        else if (k == "blur_center_x")  s.blur_center_x  = v;
        else if (k == "blur_center_y")  s.blur_center_y  = v;
        else if (k == "blur_tilt_y")    s.blur_tilt_y    = v;
        else if (k == "blur_tilt_h")    s.blur_tilt_h    = v;
        // unknown keys are silently ignored
    }

    return s;
}

}} // namespace caspar::keyframes
