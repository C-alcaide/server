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

#include <core/frame/transform_fields.h>

#include <common/log.h>

#include <algorithm>
#include <cmath>
#include <unordered_map>

namespace caspar { namespace keyframes {

namespace fields = core::fields;

static constexpr double RAD2DEG = 180.0 / 3.141592653589793;
static constexpr double DEG2RAD = 3.141592653589793 / 180.0;

// ---------------------------------------------------------------------------
// The table is DERIVED from `core::fields::all()` rather than written here.
//
// It used to be the source of truth for what a keyframe can animate, and it was
// one of four hand-written descriptions of `image_transform` that had to be kept
// aligned by hand. It is now a projection of the registry: one flat entry per
// (field, component), with the component names the registry declares.
//
// Two conversions happen in this projection and nowhere else, which is why the
// projection exists at all rather than KEYFRAMES using the registry directly:
//
//   * ANGLES. The registry stores what `image_transform` stores -- radians for
//     `angle` and the projection fields. KEYFRAMES has always spoken degrees, and
//     saved timelines are full of degrees, so `angular_rad` fields convert on the
//     way in and out. A saved timeline from before this change animates exactly
//     what it animated before.
//   * ARITY. The registry declares `lift` once with three components; KEYFRAMES
//     addresses `lift_r`, `lift_g`, `lift_b` separately, because a keyframe track
//     is a scalar.
//
// The names are not derivable from the members -- `fill_x` is
// `fill_translation[0]`, `mid_r` is `midtone[0]`, `rgb_r_min_in` is
// `per_channel_levels.r.min_input` -- so the registry carries them explicitly and
// this file reads them.
// ---------------------------------------------------------------------------

namespace {

/// The KEYFRAMES names as they stood when this table was hand-written, in table order.
/// Checked against what the registry generates at startup: a rename, a dropped row or a
/// mistyped alias changes what a SAVED TIMELINE animates, which is the one failure here
/// that is silent and irreversible. 193 names -- the figure is not 205; that count
/// included the twelve geometry names which also appear in `is_geometry_field`'s map.
constexpr const char* FROZEN_KF_NAMES[] = {
    "opacity", "contrast", "brightness", "saturation", "anchor_x", "anchor_y", "fill_x", "fill_y", "fill_sx",
    "fill_sy", "clip_x", "clip_y", "clip_sx", "clip_sy", "angle", "crop_ul_x", "crop_ul_y", "crop_lr_x", "crop_lr_y",
    "persp_ul_x", "persp_ul_y", "persp_ur_x", "persp_ur_y", "persp_lr_x", "persp_lr_y", "persp_ll_x", "persp_ll_y",
    "proj_enable", "proj_yaw", "proj_pitch", "proj_roll", "proj_fov", "proj_offset_x", "proj_offset_y",
    "proj_frustum_h", "proj_frustum_v", "proj_lens_k1", "proj_lens_k2", "proj_lens_k3", "proj_lens_p1",
    "proj_lens_p2", "proj_screen_arc", "proj_screen_arc_v", "proj_eye_distance", "proj_curve_enable",
    "proj_curve_auto", "proj_edge_blend_left", "proj_edge_blend_right", "proj_edge_blend_top",
    "proj_edge_blend_bottom", "proj_edge_blend_gamma", "proj_icvfx_enable", "proj_inner_fov", "proj_icvfx_feather",
    "proj_icvfx_outer_dim", "proj_icvfx_inner_dim", "proj_icvfx_inner_gain_r", "proj_icvfx_inner_gain_g",
    "proj_icvfx_inner_gain_b", "proj_icvfx_outer_gain_r", "proj_icvfx_outer_gain_g", "proj_icvfx_outer_gain_b",
    "temperature", "tint", "shadows", "highlights", "lift_r", "lift_g", "lift_b", "mid_r", "mid_g", "mid_b",
    "gain_r", "gain_g", "gain_b", "hue_shift", "invert", "flip_h", "flip_v", "linear_saturation", "levels_min_in",
    "levels_max_in", "levels_gamma", "levels_min_out", "levels_max_out", "rgb_r_min_in", "rgb_r_max_in",
    "rgb_r_gamma", "rgb_r_min_out", "rgb_r_max_out", "rgb_g_min_in", "rgb_g_max_in", "rgb_g_gamma", "rgb_g_min_out",
    "rgb_g_max_out", "rgb_b_min_in", "rgb_b_max_in", "rgb_b_gamma", "rgb_b_min_out", "rgb_b_max_out", "blur_radius",
    "blur_angle", "blur_center_x", "blur_center_y", "blur_tilt_y", "blur_tilt_h", "cdl_slope_r", "cdl_slope_g",
    "cdl_slope_b", "cdl_offset_r", "cdl_offset_g", "cdl_offset_b", "cdl_power_r", "cdl_power_g", "cdl_power_b",
    "cdl_saturation", "split_shadow_r", "split_shadow_g", "split_shadow_b", "split_highlight_r",
    "split_highlight_g", "split_highlight_b", "split_balance", "gamut_compress", "gc_cyan", "gc_magenta",
    "gc_yellow", "lut3d_strength", "sharpen_amount", "sharpen_radius", "grain_intensity", "grain_size",
    "qualifier_enable", "qual_target_hue", "qual_hue_width", "qual_min_sat", "qual_max_sat", "qual_min_lum",
    "qual_max_lum", "qual_softness", "qual_exposure", "qual_sat_offset", "qual_hue_offset", "color_grade_enable",
    "color_grade_exposure", "color_grade_input_transfer", "color_grade_input_gamut", "color_grade_tone_mapping",
    "color_grade_output_gamut", "color_grade_output_transfer", "shape_enable", "shape_center_x", "shape_center_y",
    "shape_size_x", "shape_size_y", "shape_corner_radius", "shape_edge_softness", "shape_gradient_angle",
    "shape_gradient_cx", "shape_gradient_cy", "shape_stroke_width", "shape_stroke_enable", "shape_color1_r",
    "shape_color1_g", "shape_color1_b", "shape_color1_a", "shape_color2_r", "shape_color2_g", "shape_color2_b",
    "shape_color2_a", "shape_stroke_r", "shape_stroke_g", "shape_stroke_b", "shape_stroke_a", "chroma_enable",
    "chroma_target_hue", "chroma_hue_width", "chroma_min_sat", "chroma_min_bright", "chroma_softness",
    "chroma_spill", "chroma_spill_sat", "enable_geometry", "blur_enable", "rgb_enable", "curves_enable",
    "chroma_show_mask", "blur_type", "shape_type", "shape_fill_type", "proj_curve_type", "proj_source_lens",
    "blend_mode",
};

/// The projection's per-entry state. A `kf_field` carries only function pointers, so the
/// field and component a generated accessor refers to have to be reachable from a
/// non-capturing lambda -- hence a side table indexed by the entry's position.
struct kf_binding
{
    const fields::field_desc* field;
    uint8_t                   component;
    bool                      to_degrees;
};

std::vector<kf_binding>& bindings()
{
    static std::vector<kf_binding> b;
    return b;
}

/// Read component `c` of a field as a double. Enum and blob fields read as their ordinal
/// or presence, which is what the previous hand-written table did.
double read_component(const kf_binding& b, const core::image_transform& t)
{
    const auto v = b.field->get(t);
    if (b.component >= v.size())
        return 0.0;

    const auto& e = v[b.component];

    // An enum reads back by name; KEYFRAMES wants the ordinal it has always used.
    if (const auto* s = boost::get<std::string>(&e)) {
        const auto names = fields::split_list(b.field->values);
        for (std::size_t i = 0; i < names.size(); ++i)
            if (names[i] == *s)
                return static_cast<double>(i);
        return 0.0;
    }

    double d = 0.0;
    if (const auto* pd = boost::get<double>(&e))
        d = *pd;
    else if (const auto* pf = boost::get<float>(&e))
        d = *pf;
    else if (const auto* pi = boost::get<int32_t>(&e))
        d = *pi;
    else if (const auto* pb = boost::get<bool>(&e))
        d = *pb ? 1.0 : 0.0;

    return b.to_degrees ? d * RAD2DEG : d;
}

void write_component(const kf_binding& b, core::image_transform& t, double value)
{
    auto v = b.field->get(t);
    if (b.component >= v.size())
        return;

    const double d = b.to_degrees ? value * DEG2RAD : value;

    switch (b.field->type) {
        case fields::value_type::boolean: v[b.component] = (d >= 0.5); break;
        case fields::value_type::integer: v[b.component] = static_cast<int32_t>(d + (d < 0 ? -0.5 : 0.5)); break;
        case fields::value_type::enumeration: {
            // Written by ordinal, snapped -- the setter accepts a number as well as a name.
            const auto names = fields::split_list(b.field->values);
            const auto n     = static_cast<int>(d + 0.5);
            if (n < 0 || static_cast<std::size_t>(n) >= names.size())
                return;
            v[b.component] = static_cast<int32_t>(n);
            break;
        }
        default: v[b.component] = d; break;
    }

    b.field->set(t, v);
}

field_kind to_field_kind(fields::kf_kind k)
{
    switch (k) {
        case fields::kf_kind::continuous: return field_kind::continuous;
        case fields::kf_kind::angular:
        case fields::kf_kind::angular_rad: return field_kind::angular;
        case fields::kf_kind::discrete: return field_kind::discrete;
    }
    return field_kind::continuous;
}

/// Non-capturing trampolines. `kf_field` predates this projection and holds raw function
/// pointers, so the binding is looked up by index at call time. The index is baked in by
/// the generator below through a small dispatch table.
constexpr std::size_t MAX_KF_FIELDS = 256;

template <std::size_t I>
double kf_get(const core::image_transform& t)
{
    return read_component(bindings()[I], t);
}

template <std::size_t I>
void kf_set(core::image_transform& t, double v)
{
    write_component(bindings()[I], t, v);
}

template <std::size_t... Is>
constexpr auto make_getters(std::index_sequence<Is...>)
{
    return std::array<double (*)(const core::image_transform&), sizeof...(Is)>{&kf_get<Is>...};
}

template <std::size_t... Is>
constexpr auto make_setters(std::index_sequence<Is...>)
{
    return std::array<void (*)(core::image_transform&, double), sizeof...(Is)>{&kf_set<Is>...};
}

const auto& getters()
{
    static const auto g = make_getters(std::make_index_sequence<MAX_KF_FIELDS>{});
    return g;
}

const auto& setters()
{
    static const auto s = make_setters(std::make_index_sequence<MAX_KF_FIELDS>{});
    return s;
}

const std::vector<kf_field>& build_field_table()
{
    static const std::vector<kf_field> table = [] {
        std::vector<kf_field> out;
        auto&                 binds = bindings();
        binds.clear();

        // `kf_field::name` is a raw `const char*`, so the strings it points at must never
        // move. A `vector` that grows reallocates and invalidates every pointer already
        // handed out -- which is not a crash, it is silently wrong names, and the frozen
        // check caught exactly that on the first run of this table. Reserving to the
        // bound the loop already enforces makes the addresses stable.
        static std::vector<std::string> storage;
        storage.clear();
        storage.reserve(MAX_KF_FIELDS);
        binds.reserve(MAX_KF_FIELDS);

        for (const auto& f : fields::all()) {
            const auto names = fields::split_list(f.kf_names);
            for (std::size_t c = 0; c < names.size(); ++c) {
                if (binds.size() >= MAX_KF_FIELDS) {
                    CASPAR_LOG(error) << L"[keyframes] more than " << MAX_KF_FIELDS
                                      << L" animatable components; raise MAX_KF_FIELDS";
                    break;
                }

                const auto idx = binds.size();
                binds.push_back(kf_binding{&f, static_cast<uint8_t>(c), f.kind == fields::kf_kind::angular_rad});

                // The name must outlive the table. The registry's `kf_names` is a literal
                // with static storage, but `split_list` gives a view into it that is not
                // null-terminated at the component boundary -- so this owns a copy.
                storage.emplace_back(names[c]);

                double def = 0.0;
                {
                    const auto dv = f.defaults();
                    if (c < dv.size()) {
                        core::image_transform probe;
                        f.set(probe, dv);
                        def = read_component(binds.back(), probe);
                    }
                }

                out.push_back(kf_field{storage.back().c_str(),
                                       getters()[idx],
                                       setters()[idx],
                                       def,
                                       to_field_kind(f.kind)});
            }
        }
        return out;
    }();
    return table;
}

const std::unordered_map<std::string, std::size_t>& build_name_index()
{
    static const auto idx = [] {
        std::unordered_map<std::string, std::size_t> m;
        const auto&                                  fields = build_field_table();
        for (std::size_t i = 0; i < fields.size(); ++i)
            m[fields[i].name] = i;
        return m;
    }();
    return idx;
}

} // namespace

const std::vector<kf_field>& kf_all_fields() { return build_field_table(); }

const kf_field* kf_find_field(const std::string& name)
{
    const auto& idx = build_name_index();
    const auto  it  = idx.find(name);
    if (it == idx.end())
        return nullptr;
    return &build_field_table()[it->second];
}

bool kf_verify_frozen_names(std::vector<std::string>& missing, std::vector<std::string>& added)
{
    const auto& idx = build_name_index();

    std::unordered_map<std::string, bool> frozen;
    for (const char* n : FROZEN_KF_NAMES) {
        frozen.emplace(n, true);
        if (idx.find(n) == idx.end())
            missing.emplace_back(n);
    }
    for (const auto& f : build_field_table())
        if (frozen.find(f.name) == frozen.end())
            added.emplace_back(f.name);

    std::sort(missing.begin(), missing.end());
    std::sort(added.begin(), added.end());
    return missing.empty();
}

// ---------------------------------------------------------------------------
// Geometry / RGB / blur grouping, for the auto-enable rule below.
//
// These read the registry's `enables` column rather than a second hand-written
// list of names, so a field that gains a subsystem enable gets the auto-enable
// behaviour without this file being touched.
// ---------------------------------------------------------------------------

namespace {

/// Which subsystem enable, if any, a KEYFRAMES name belongs to.
const char* enables_for(const std::string& name)
{
    const auto ref = fields::find_kf(name);
    if (!ref)
        return nullptr;
    return ref->field->enables;
}

} // namespace

void apply_kf_to_transform(const kf_values& vals, core::image_transform& tf)
{
    bool has_geometry         = false;
    bool has_rgb              = false;
    bool has_blur             = false;
    bool explicit_geo_enable  = false;
    bool explicit_blur_enable = false;
    bool explicit_rgb_enable  = false;

    for (const auto& [name, value] : vals) {
        // An explicit enable in the same value set wins over the inference below: a
        // timeline that animates `blur_enable` to 0 while animating `blur_radius` means it.
        if (name == "enable_geometry")
            explicit_geo_enable = true;
        if (name == "blur_enable")
            explicit_blur_enable = true;
        if (name == "rgb_enable")
            explicit_rgb_enable = true;

        if (const char* e = enables_for(name)) {
            const std::string_view ev{e};
            if (ev == "enable_geometry_modifiers")
                has_geometry = true;
            else if (ev == "per_channel_levels.enable")
                has_rgb = true;
            else if (ev == "blur.enable" && name != "blur_enable")
                has_blur = true;
        }

        if (const kf_field* f = kf_find_field(name))
            f->set(tf, value);
    }

    if (has_geometry && !explicit_geo_enable)
        tf.enable_geometry_modifiers = true;
    if (has_blur && !explicit_blur_enable)
        tf.blur.enable = (tf.blur.radius > 0.0);
    if (has_rgb && !explicit_rgb_enable)
        tf.per_channel_levels.enable = true;
}

kf_values capture_from_transform(const core::image_transform& tf, bool only_non_default)
{
    kf_values   vals;
    const auto& fields = kf_all_fields();
    for (const auto& f : fields) {
        const double v = f.get(tf);
        if (!only_non_default || std::abs(v - f.default_val) > 1e-9)
            vals[f.name] = v;
    }
    return vals;
}

}} // namespace caspar::keyframes
