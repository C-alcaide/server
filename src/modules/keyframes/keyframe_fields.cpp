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

#include "keyframe_fields.h"

#include <cmath>
#include <unordered_map>

namespace caspar { namespace keyframes {

using IT = core::image_transform;

static constexpr double DEG2RAD = 3.141592653589793 / 180.0;
static constexpr double RAD2DEG = 180.0 / 3.141592653589793;

// ---------------------------------------------------------------------------
// Helper macros for compact field definitions
// ---------------------------------------------------------------------------

#define KF_D(n, member, def) \
    {n, [](const IT& t) -> double { return t.member; }, [](IT& t, double v) { t.member = v; }, def, field_kind::continuous}

#define KF_A2(n, member, idx, def) \
    {n, [](const IT& t) -> double { return t.member[idx]; }, [](IT& t, double v) { t.member[idx] = v; }, def, field_kind::continuous}

#define KF_A3(n, member, idx, def) \
    {n, [](const IT& t) -> double { return t.member[idx]; }, [](IT& t, double v) { t.member[idx] = v; }, def, field_kind::continuous}

#define KF_B(n, member) \
    {n, [](const IT& t) -> double { return t.member ? 1.0 : 0.0; }, [](IT& t, double v) { t.member = (v >= 0.5); }, 0.0, field_kind::discrete}

#define KF_B1(n, member, def) \
    {n, [](const IT& t) -> double { return t.member ? 1.0 : 0.0; }, [](IT& t, double v) { t.member = (v >= 0.5); }, def, field_kind::discrete}

#define KF_RAD(n, member, def_deg) \
    {n, [](const IT& t) -> double { return t.member * RAD2DEG; }, [](IT& t, double v) { t.member = v * DEG2RAD; }, def_deg, field_kind::angular}

#define KF_F(n, member, def) \
    {n, [](const IT& t) -> double { return static_cast<double>(t.member); }, [](IT& t, double v) { t.member = static_cast<float>(v); }, def, field_kind::continuous}

#define KF_I(n, member, def) \
    {n, [](const IT& t) -> double { return static_cast<double>(t.member); }, [](IT& t, double v) { t.member = static_cast<int>(v + 0.5); }, def, field_kind::discrete}

// ---------------------------------------------------------------------------
// The field descriptor table — single source of truth
//
// To add a new animatable property: add ONE line here.
// ---------------------------------------------------------------------------

static const std::vector<kf_field>& build_field_table()
{
    static const std::vector<kf_field> fields = {
        // ── Basic ───────────────────────────────────────────────────
        KF_D("opacity",    opacity,    1.0),
        KF_D("contrast",   contrast,   1.0),
        KF_D("brightness", brightness, 1.0),
        KF_D("saturation", saturation, 1.0),

        // ── Geometry ────────────────────────────────────────────────
        KF_A2("anchor_x",  anchor, 0, 0.0),
        KF_A2("anchor_y",  anchor, 1, 0.0),
        KF_A2("fill_x",    fill_translation, 0, 0.0),
        KF_A2("fill_y",    fill_translation, 1, 0.0),
        KF_A2("fill_sx",   fill_scale, 0, 1.0),
        KF_A2("fill_sy",   fill_scale, 1, 1.0),
        KF_A2("clip_x",    clip_translation, 0, 0.0),
        KF_A2("clip_y",    clip_translation, 1, 0.0),
        KF_A2("clip_sx",   clip_scale, 0, 1.0),
        KF_A2("clip_sy",   clip_scale, 1, 1.0),

        // angle: stored as radians in image_transform, degrees in JSON
        KF_RAD("angle", angle, 0.0),

        // ── Crop ────────────────────────────────────────────────────
        KF_A2("crop_ul_x", crop.ul, 0, 0.0),
        KF_A2("crop_ul_y", crop.ul, 1, 0.0),
        KF_A2("crop_lr_x", crop.lr, 0, 1.0),
        KF_A2("crop_lr_y", crop.lr, 1, 1.0),

        // ── Perspective ─────────────────────────────────────────────
        KF_A2("persp_ul_x", perspective.ul, 0, 0.0),
        KF_A2("persp_ul_y", perspective.ul, 1, 0.0),
        KF_A2("persp_ur_x", perspective.ur, 0, 1.0),
        KF_A2("persp_ur_y", perspective.ur, 1, 0.0),
        KF_A2("persp_lr_x", perspective.lr, 0, 1.0),
        KF_A2("persp_lr_y", perspective.lr, 1, 1.0),
        KF_A2("persp_ll_x", perspective.ll, 0, 0.0),
        KF_A2("persp_ll_y", perspective.ll, 1, 1.0),

        // ── Projection (degrees in JSON, radians in IT) ─────────────
        KF_B("proj_enable", projection.enable),
        KF_RAD("proj_yaw",   projection.yaw,   0.0),
        KF_RAD("proj_pitch", projection.pitch,  0.0),
        KF_RAD("proj_roll",  projection.roll,   0.0),
        KF_RAD("proj_fov",   projection.fov,   90.0),
        KF_D("proj_offset_x", projection.offset_x, 0.0),
        KF_D("proj_offset_y", projection.offset_y, 0.0),
        KF_D("proj_frustum_h", projection.frustum_h, 0.0),
        KF_D("proj_frustum_v", projection.frustum_v, 0.0),
        KF_D("proj_lens_k1", projection.lens_k1, 0.0),
        KF_D("proj_lens_k2", projection.lens_k2, 0.0),
        KF_D("proj_lens_k3", projection.lens_k3, 0.0),
        KF_D("proj_lens_p1", projection.lens_p1, 0.0),
        KF_D("proj_lens_p2", projection.lens_p2, 0.0),
        KF_RAD("proj_screen_arc", projection.screen_arc, 0.0),
        KF_RAD("proj_screen_arc_v", projection.screen_arc_v, 0.0),
        KF_D("proj_eye_distance", projection.eye_distance, 1.0),
        KF_B("proj_curve_enable", projection.curve_enable),
        KF_B("proj_curve_auto", projection.curve_auto),
        KF_D("proj_edge_blend_left",   projection.edge_blend_left,   0.0),
        KF_D("proj_edge_blend_right",  projection.edge_blend_right,  0.0),
        KF_D("proj_edge_blend_top",    projection.edge_blend_top,    0.0),
        KF_D("proj_edge_blend_bottom", projection.edge_blend_bottom, 0.0),
        KF_D("proj_edge_blend_gamma",  projection.edge_blend_gamma,  2.2),

        // ── ICVFX inner/outer frustum ───────────────────────────────
        KF_B("proj_icvfx_enable",     projection.icvfx_enable),
        KF_RAD("proj_inner_fov",      projection.inner_fov,  90.0),
        KF_D("proj_icvfx_feather",    projection.icvfx_feather,   0.05),
        KF_D("proj_icvfx_outer_dim",  projection.icvfx_outer_dim, 1.0),

        // ── White balance ───────────────────────────────────────────
        KF_D("temperature", temperature, 0.0),
        KF_D("tint",        tint,        0.0),

        // ── Tone balance ────────────────────────────────────────────
        KF_D("shadows",    shadows,    0.0),
        KF_D("highlights", highlights, 0.0),

        // ── 3-Way colour corrector ──────────────────────────────────
        KF_A3("lift_r", lift, 0, 0.0),
        KF_A3("lift_g", lift, 1, 0.0),
        KF_A3("lift_b", lift, 2, 0.0),
        KF_A3("mid_r",  midtone, 0, 1.0),
        KF_A3("mid_g",  midtone, 1, 1.0),
        KF_A3("mid_b",  midtone, 2, 1.0),
        KF_A3("gain_r", gain, 0, 1.0),
        KF_A3("gain_g", gain, 1, 1.0),
        KF_A3("gain_b", gain, 2, 1.0),

        // ── Hue / Invert / Flip ─────────────────────────────────────
        {"hue_shift",
         [](const IT& t) -> double { return t.hue_shift; },
         [](IT& t, double v) { t.hue_shift = v; },
         0.0, field_kind::angular},
        KF_B("invert", invert),
        KF_B("flip_h", flip_h),
        KF_B("flip_v", flip_v),

        // ── Linear saturation ───────────────────────────────────────
        KF_D("linear_saturation", linear_saturation, 1.0),

        // ── Levels (master) ─────────────────────────────────────────
        KF_D("levels_min_in",  levels.min_input,  0.0),
        KF_D("levels_max_in",  levels.max_input,  1.0),
        KF_D("levels_gamma",   levels.gamma,      1.0),
        KF_D("levels_min_out", levels.min_output,  0.0),
        KF_D("levels_max_out", levels.max_output,  1.0),

        // ── Per-channel RGB levels ──────────────────────────────────
        KF_D("rgb_r_min_in",  per_channel_levels.r.min_input,  0.0),
        KF_D("rgb_r_max_in",  per_channel_levels.r.max_input,  1.0),
        KF_D("rgb_r_gamma",   per_channel_levels.r.gamma,      1.0),
        KF_D("rgb_r_min_out", per_channel_levels.r.min_output, 0.0),
        KF_D("rgb_r_max_out", per_channel_levels.r.max_output, 1.0),
        KF_D("rgb_g_min_in",  per_channel_levels.g.min_input,  0.0),
        KF_D("rgb_g_max_in",  per_channel_levels.g.max_input,  1.0),
        KF_D("rgb_g_gamma",   per_channel_levels.g.gamma,      1.0),
        KF_D("rgb_g_min_out", per_channel_levels.g.min_output, 0.0),
        KF_D("rgb_g_max_out", per_channel_levels.g.max_output, 1.0),
        KF_D("rgb_b_min_in",  per_channel_levels.b.min_input,  0.0),
        KF_D("rgb_b_max_in",  per_channel_levels.b.max_input,  1.0),
        KF_D("rgb_b_gamma",   per_channel_levels.b.gamma,      1.0),
        KF_D("rgb_b_min_out", per_channel_levels.b.min_output, 0.0),
        KF_D("rgb_b_max_out", per_channel_levels.b.max_output, 1.0),

        // ── Blur ────────────────────────────────────────────────────
        KF_D("blur_radius",   blur.radius,    0.0),
        {"blur_angle", [](const IT& t) -> double { return t.blur.angle; },
                       [](IT& t, double v) { t.blur.angle = v; }, 0.0, field_kind::angular},
        KF_A2("blur_center_x", blur.center, 0, 0.5),
        KF_A2("blur_center_y", blur.center, 1, 0.5),
        KF_D("blur_tilt_y", blur.tilt_y, 0.5),
        KF_D("blur_tilt_h", blur.tilt_h, 0.2),

        // ── ASC CDL ─────────────────────────────────────────────────
        KF_A3("cdl_slope_r",  cdl_slope,  0, 1.0),
        KF_A3("cdl_slope_g",  cdl_slope,  1, 1.0),
        KF_A3("cdl_slope_b",  cdl_slope,  2, 1.0),
        KF_A3("cdl_offset_r", cdl_offset, 0, 0.0),
        KF_A3("cdl_offset_g", cdl_offset, 1, 0.0),
        KF_A3("cdl_offset_b", cdl_offset, 2, 0.0),
        KF_A3("cdl_power_r",  cdl_power,  0, 1.0),
        KF_A3("cdl_power_g",  cdl_power,  1, 1.0),
        KF_A3("cdl_power_b",  cdl_power,  2, 1.0),
        KF_D("cdl_saturation", cdl_saturation, 1.0),

        // ── Split toning ────────────────────────────────────────────
        KF_A3("split_shadow_r",    split_shadow_color,    0, 0.0),
        KF_A3("split_shadow_g",    split_shadow_color,    1, 0.0),
        KF_A3("split_shadow_b",    split_shadow_color,    2, 0.0),
        KF_A3("split_highlight_r", split_highlight_color, 0, 0.0),
        KF_A3("split_highlight_g", split_highlight_color, 1, 0.0),
        KF_A3("split_highlight_b", split_highlight_color, 2, 0.0),
        KF_D("split_balance", split_balance, 0.5),

        // ── Gamut compression ───────────────────────────────────────
        KF_B("gamut_compress", gamut_compress),
        KF_D("gc_cyan",    gc_cyan,    1.147),
        KF_D("gc_magenta", gc_magenta, 1.264),
        KF_D("gc_yellow",  gc_yellow,  1.312),

        // ── LUT strength ────────────────────────────────────────────
        KF_F("lut3d_strength", lut3d_strength, 1.0),

        // ── Sharpening ──────────────────────────────────────────────
        KF_D("sharpen_amount", sharpen_amount, 0.0),
        KF_D("sharpen_radius", sharpen_radius, 1.0),

        // ── Film grain ──────────────────────────────────────────────
        KF_D("grain_intensity", grain_intensity, 0.0),
        KF_D("grain_size",      grain_size,      1.0),

        // ── Secondary qualifier ─────────────────────────────────────
        KF_B("qualifier_enable", qualifier_enable),
        KF_D("qual_target_hue", qual_target_hue, 0.0),
        KF_D("qual_hue_width",  qual_hue_width,  0.1),
        KF_D("qual_min_sat",    qual_min_sat,     0.2),
        KF_D("qual_max_sat",    qual_max_sat,     1.0),
        KF_D("qual_min_lum",    qual_min_lum,     0.0),
        KF_D("qual_max_lum",    qual_max_lum,     1.0),
        KF_D("qual_softness",   qual_softness,    0.1),
        KF_D("qual_exposure",   qual_exposure,    0.0),
        KF_D("qual_sat_offset", qual_sat_offset,  0.0),
        KF_D("qual_hue_offset", qual_hue_offset,  0.0),

        // ── Color grade ─────────────────────────────────────────────
        KF_B("color_grade_enable", color_grade.enable),
        KF_F("color_grade_exposure", color_grade.exposure, 1.0),
        KF_I("color_grade_input_transfer",  color_grade.input_transfer,  0),
        KF_I("color_grade_input_gamut",     color_grade.input_gamut,     0),
        KF_I("color_grade_tone_mapping",    color_grade.tone_mapping,    0),
        KF_I("color_grade_output_gamut",    color_grade.output_gamut,    0),
        KF_I("color_grade_output_transfer", color_grade.output_transfer, 0),

        // ── Shape ───────────────────────────────────────────────────
        KF_B("shape_enable", shape.enable),
        KF_A2("shape_center_x", shape.center, 0, 0.5),
        KF_A2("shape_center_y", shape.center, 1, 0.5),
        KF_A2("shape_size_x",   shape.size,   0, 0.5),
        KF_A2("shape_size_y",   shape.size,   1, 0.5),
        KF_D("shape_corner_radius", shape.corner_radius, 0.0),
        KF_D("shape_edge_softness", shape.edge_softness, 0.005),
        {"shape_gradient_angle",
         [](const IT& t) -> double { return t.shape.gradient_angle; },
         [](IT& t, double v) { t.shape.gradient_angle = v; }, 0.0, field_kind::angular},
        KF_A2("shape_gradient_cx", shape.gradient_center, 0, 0.5),
        KF_A2("shape_gradient_cy", shape.gradient_center, 1, 0.5),
        KF_D("shape_stroke_width", shape.stroke_width, 0.0),
        KF_B("shape_stroke_enable", shape.stroke_enable),
        KF_A2("shape_color1_r", shape.color1, 0, 1.0),  // Note: array<double,4>
        KF_A2("shape_color1_g", shape.color1, 1, 1.0),
        KF_A2("shape_color1_b", shape.color1, 2, 1.0),
        KF_A2("shape_color1_a", shape.color1, 3, 1.0),
        KF_A2("shape_color2_r", shape.color2, 0, 0.0),
        KF_A2("shape_color2_g", shape.color2, 1, 0.0),
        KF_A2("shape_color2_b", shape.color2, 2, 0.0),
        KF_A2("shape_color2_a", shape.color2, 3, 0.0),
        KF_A2("shape_stroke_r", shape.stroke_color, 0, 1.0),
        KF_A2("shape_stroke_g", shape.stroke_color, 1, 1.0),
        KF_A2("shape_stroke_b", shape.stroke_color, 2, 1.0),
        KF_A2("shape_stroke_a", shape.stroke_color, 3, 1.0),

        // ── Chroma ──────────────────────────────────────────────────
        KF_B("chroma_enable",    chroma.enable),
        KF_D("chroma_target_hue",   chroma.target_hue,   0.0),
        KF_D("chroma_hue_width",    chroma.hue_width,    0.0),
        KF_D("chroma_min_sat",      chroma.min_saturation, 0.0),
        KF_D("chroma_min_bright",   chroma.min_brightness, 0.0),
        KF_D("chroma_softness",     chroma.softness,     0.0),
        KF_D("chroma_spill",        chroma.spill_suppress, 0.0),
        KF_D("chroma_spill_sat",    chroma.spill_suppress_saturation, 1.0),

        // ── Subsystem enables ───────────────────────────────────────
        KF_B1("enable_geometry", enable_geometry_modifiers, 0.0),
        KF_B("blur_enable",  blur.enable),
        KF_B("rgb_enable",   per_channel_levels.enable),
        KF_B("curves_enable", curves.enable),
        KF_B("chroma_show_mask", chroma.show_mask),

        // ── Enum type selectors (discrete) ──────────────────────────
        {"blur_type",
         [](const IT& t) -> double { return static_cast<double>(static_cast<int>(t.blur.type)); },
         [](IT& t, double v) { t.blur.type = static_cast<core::blur_type>(static_cast<int>(v + 0.5)); },
         0.0, field_kind::discrete},
        {"shape_type",
         [](const IT& t) -> double { return static_cast<double>(static_cast<int>(t.shape.type)); },
         [](IT& t, double v) { t.shape.type = static_cast<core::shape_type>(static_cast<int>(v + 0.5)); },
         0.0, field_kind::discrete},
        {"shape_fill_type",
         [](const IT& t) -> double { return static_cast<double>(static_cast<int>(t.shape.fill_type)); },
         [](IT& t, double v) { t.shape.fill_type = static_cast<core::shape_fill_type>(static_cast<int>(v + 0.5)); },
         0.0, field_kind::discrete},
        {"proj_curve_type",
         [](const IT& t) -> double { return static_cast<double>(static_cast<int>(t.projection.curve_type)); },
         [](IT& t, double v) { t.projection.curve_type = static_cast<core::screen_curve_type>(static_cast<int>(v + 0.5)); },
         0.0, field_kind::discrete},
        {"proj_source_lens",
         [](const IT& t) -> double { return static_cast<double>(static_cast<int>(t.projection.source_lens)); },
         [](IT& t, double v) { t.projection.source_lens = static_cast<core::screen_curve_type>(static_cast<int>(v + 0.5)); },
         0.0, field_kind::discrete},
        {"blend_mode",
         [](const IT& t) -> double { return static_cast<double>(static_cast<int>(t.blend_mode)); },
         [](IT& t, double v) { t.blend_mode = static_cast<core::blend_mode>(static_cast<int>(v + 0.5)); },
         static_cast<double>(static_cast<int>(core::blend_mode::normal)), field_kind::discrete},
    };
    return fields;
}

#undef KF_D
#undef KF_A2
#undef KF_A3
#undef KF_B
#undef KF_B1
#undef KF_RAD
#undef KF_F
#undef KF_I

// ---------------------------------------------------------------------------
// Name → field descriptor lookup (built once)
// ---------------------------------------------------------------------------

static const std::unordered_map<std::string, size_t>& build_name_index()
{
    static const auto idx = [] {
        std::unordered_map<std::string, size_t> m;
        const auto& fields = build_field_table();
        for (size_t i = 0; i < fields.size(); ++i)
            m[fields[i].name] = i;
        return m;
    }();
    return idx;
}

const std::vector<kf_field>& kf_all_fields() { return build_field_table(); }

const kf_field* kf_find_field(const std::string& name)
{
    const auto& idx = build_name_index();
    auto it = idx.find(name);
    if (it == idx.end())
        return nullptr;
    return &build_field_table()[it->second];
}

// ---------------------------------------------------------------------------
// Geometry field names (for auto-enabling enable_geometry_modifiers)
// ---------------------------------------------------------------------------

static bool is_geometry_field(const std::string& name)
{
    static const std::unordered_map<std::string, bool> geo = {
        {"anchor_x", true}, {"anchor_y", true},
        {"fill_x", true},   {"fill_y", true},
        {"fill_sx", true},  {"fill_sy", true},
        {"clip_x", true},   {"clip_y", true},
        {"clip_sx", true},  {"clip_sy", true},
        {"angle", true},
        {"crop_ul_x", true}, {"crop_ul_y", true},
        {"crop_lr_x", true}, {"crop_lr_y", true},
        {"persp_ul_x", true}, {"persp_ul_y", true},
        {"persp_ur_x", true}, {"persp_ur_y", true},
        {"persp_lr_x", true}, {"persp_lr_y", true},
        {"persp_ll_x", true}, {"persp_ll_y", true},
    };
    return geo.count(name) > 0;
}

static bool is_rgb_levels_field(const std::string& name)
{
    return name.size() >= 5 && name.compare(0, 4, "rgb_") == 0;
}

// ---------------------------------------------------------------------------
// apply_kf_to_transform — apply sparse values via field descriptors
// ---------------------------------------------------------------------------

void apply_kf_to_transform(const kf_values& vals, core::image_transform& tf)
{
    bool has_geometry = false;
    bool has_rgb      = false;
    bool has_blur     = false;
    bool explicit_geo_enable = false;
    bool explicit_blur_enable = false;
    bool explicit_rgb_enable = false;

    for (const auto& [name, value] : vals) {
        // Check for explicit enable overrides
        if (name == "enable_geometry") explicit_geo_enable = true;
        if (name == "blur_enable")     explicit_blur_enable = true;
        if (name == "rgb_enable")      explicit_rgb_enable = true;

        // Track field groups
        if (is_geometry_field(name))   has_geometry = true;
        if (is_rgb_levels_field(name)) has_rgb = true;
        if (name.size() >= 4 && name.compare(0, 4, "blur") == 0 && name != "blur_enable") has_blur = true;

        const kf_field* f = kf_find_field(name);
        if (f)
            f->set(tf, value);
    }

    // Auto-enable subsystems if fields were set but enable wasn't explicit
    if (has_geometry && !explicit_geo_enable)
        tf.enable_geometry_modifiers = true;
    if (has_blur && !explicit_blur_enable)
        tf.blur.enable = (tf.blur.radius > 0.0);
    if (has_rgb && !explicit_rgb_enable)
        tf.per_channel_levels.enable = true;
}

// ---------------------------------------------------------------------------
// capture_from_transform — read image_transform into sparse values
// ---------------------------------------------------------------------------

kf_values capture_from_transform(const core::image_transform& tf, bool only_non_default)
{
    kf_values vals;
    const auto& fields = kf_all_fields();
    for (const auto& f : fields) {
        double v = f.get(tf);
        if (!only_non_default || std::abs(v - f.default_val) > 1e-9)
            vals[f.name] = v;
    }
    return vals;
}

}} // namespace caspar::keyframes
