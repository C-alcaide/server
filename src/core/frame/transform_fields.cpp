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

#include "transform_fields.h"

#include <core/mixer/image/blend_modes.h>

#include <common/except.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <unordered_map>

namespace caspar { namespace core { namespace fields {

using IT = image_transform;
namespace lim = grade_limits;

// ---------------------------------------------------------------------------------------
// Value helpers
//
// The carrier is `monitor::vector_t` throughout, so a read is directly publishable into the
// state tree and a write is directly what arrived off the wire. `as_num` accepts any
// arithmetic alternative because JSON gives no way to distinguish 1 from 1.0, and a client
// sending an integer for a double parameter is not making a mistake.
// ---------------------------------------------------------------------------------------

namespace {

bool as_num(const monitor::data_t& d, double& out)
{
    if (const auto* p = boost::get<double>(&d)) { out = *p; return true; }
    if (const auto* p = boost::get<float>(&d)) { out = *p; return true; }
    if (const auto* p = boost::get<int32_t>(&d)) { out = *p; return true; }
    if (const auto* p = boost::get<int64_t>(&d)) { out = static_cast<double>(*p); return true; }
    if (const auto* p = boost::get<uint32_t>(&d)) { out = *p; return true; }
    if (const auto* p = boost::get<uint64_t>(&d)) { out = static_cast<double>(*p); return true; }
    if (const auto* p = boost::get<bool>(&d)) { out = *p ? 1.0 : 0.0; return true; }
    return false;
}

bool as_bool(const monitor::data_t& d, bool& out)
{
    if (const auto* p = boost::get<bool>(&d)) { out = *p; return true; }
    double v = 0.0;
    if (as_num(d, v)) { out = v >= 0.5; return true; }
    return false;
}

bool as_str(const monitor::data_t& d, std::string& out)
{
    if (const auto* p = boost::get<std::string>(&d)) { out = *p; return true; }
    return false;
}

/// Clamp only where the field declares a range. Folded into every arithmetic composition,
/// which is what stops two layers at the edge of legal reaching a value no single command
/// could set: lift 0.9 over lift 0.9 is 1.8, and midtone 0.2 under 0.2 is an exponent of 25.
double clamp_opt(const std::optional<grade_range>& r, double v) { return r ? r->clamp(v) : v; }

} // namespace

std::vector<std::string_view> split_list(const char* csv)
{
    std::vector<std::string_view> out;
    if (!csv)
        return out;
    std::string_view s{csv};
    while (!s.empty()) {
        const auto comma = s.find(',');
        out.push_back(s.substr(0, comma == std::string_view::npos ? s.size() : comma));
        if (comma == std::string_view::npos)
            break;
        s.remove_prefix(comma + 1);
    }
    return out;
}

// ---------------------------------------------------------------------------------------
// Enum value names, in enum order. These ARE the descriptor's `RANGE.VALS`, so a generated
// control offers a menu of names rather than integers, and a client can write "multiply"
// instead of 3.
// ---------------------------------------------------------------------------------------

namespace {

// core/mixer/image/blend_modes.h. `blend_mode_count` is a sentinel, not a value.
constexpr const char* BLEND_MODES =
    "normal,lighten,darken,multiply,average,add,subtract,difference,negation,exclusion,screen,overlay,"
    "soft_light,hard_light,color_dodge,color_burn,linear_dodge,linear_burn,linear_light,vivid_light,"
    "pin_light,hard_mix,reflect,glow,phoenix,contrast,saturation,color,luminosity,mix";

constexpr const char* BLUR_TYPES  = "gaussian,box,directional,zoom,tilt_shift,lens";
constexpr const char* SHAPE_TYPES = "rect,rounded_rect,circle,ellipse";
constexpr const char* FILL_TYPES  = "solid,linear,radial,conic";
constexpr const char* CURVE_TYPES = "flat,cylinder,sphere,fisheye";

/// Each guard is the exact predicate the hand-written composition uses, named once here
/// rather than repeated at every site.
bool guard_holds(guard_t g, const IT& o)
{
    const auto is_set = [](double v) { return v != 0.0; };
    switch (g) {
        case guard_t::none: return true;
        case guard_t::projection_enable: return o.projection.enable;
        case guard_t::curve_enable: return o.projection.curve_enable;
        case guard_t::edge_blend_any:
            return o.projection.edge_blend_left > 0.0 || o.projection.edge_blend_right > 0.0 ||
                   o.projection.edge_blend_top > 0.0 || o.projection.edge_blend_bottom > 0.0;
        case guard_t::icvfx_enable: return o.projection.icvfx_enable;
        case guard_t::color_grade_enable: return o.color_grade.enable;
        case guard_t::ocio_enable: return o.ocio.enable;
        case guard_t::blur_enable: return o.blur.enable;
        case guard_t::shape_enable: return o.shape.enable;
        case guard_t::curves_enable: return o.curves.enable;
        case guard_t::rgb_levels_enable: return o.per_channel_levels.enable;
        case guard_t::qualifier_enable: return o.qualifier_enable;
        case guard_t::gamut_compress: return o.gamut_compress;
        case guard_t::lut3d_present: return static_cast<bool>(o.lut3d);
        case guard_t::hue_curves_present: return static_cast<bool>(o.hue_curves);
        case guard_t::blend_mask_present: return static_cast<bool>(o.blend_mask);
        case guard_t::grade_nodes_present: return static_cast<bool>(o.grade_nodes);
        case guard_t::split_active:
            return std::any_of(o.split_shadow_color.begin(), o.split_shadow_color.end(), is_set) ||
                   std::any_of(o.split_highlight_color.begin(), o.split_highlight_color.end(), is_set);
        case guard_t::sharpen_radius_set: return o.sharpen_radius != 1.0;
        case guard_t::grain_size_set: return o.grain_size != 1.0;
    }
    return true;
}

} // namespace

// ---------------------------------------------------------------------------------------
// Table macros
//
// One line per field. Each expands to a `field_desc` with generated get/set/defaults. The
// composition is NOT generated per field: `compose_field` below dispatches once at runtime
// on `compose`, `range`, `arity` and `guard`, which is a fraction of the code and removes
// a whole class of macro bug.
// ---------------------------------------------------------------------------------------

/// A plain `double` member. `F` composes with clamping; `FU` does not -- see
/// `field_desc::compose_clamps`, and use `FU` only where a mixer demonstrably does not clamp.
#define F(NAME, MEMBER, DEF, RANGE, BOUND, RULE, GUARD, ENABLES, UNIT, KF, KIND)                                       \
    F_(NAME, MEMBER, DEF, RANGE, BOUND, RULE, GUARD, ENABLES, UNIT, KF, KIND, true)

#define FU(NAME, MEMBER, DEF, RANGE, BOUND, RULE, GUARD, ENABLES, UNIT, KF, KIND)                                      \
    F_(NAME, MEMBER, DEF, RANGE, BOUND, RULE, GUARD, ENABLES, UNIT, KF, KIND, false)

#define F_(NAME, MEMBER, DEF, RANGE, BOUND, RULE, GUARD, ENABLES, UNIT, KF, KIND, CLAMPS)                                       \
    field_desc                                                                                                         \
    {                                                                                                                  \
        NAME, value_type::real, access_t::read_write, RANGE, bounding_t::BOUND, compose_t::RULE, guard_t::GUARD,        \
            ENABLES, UNIT, nullptr, nullptr, KF, kf_kind::KIND, 0.0, 1, CLAMPS,                                         \
            [](const IT& t) { return monitor::vector_t{t.MEMBER}; },                                                    \
            [](IT& t, const monitor::vector_t& v) {                                                                     \
                double d;                                                                                              \
                if (v.size() != 1 || !as_num(v[0], d))                                                                 \
                    return false;                                                                                      \
                t.MEMBER = d;                                                                                          \
                return true;                                                                                           \
            },                                                                                                         \
            []() { return monitor::vector_t{DEF}; }                                                                     \
    }

/// A `float` member (`lut3d_strength`, `color_grade.exposure`).
#define FL(NAME, MEMBER, DEF, RANGE, BOUND, RULE, GUARD, ENABLES, UNIT, KF, KIND)                                      \
    field_desc                                                                                                         \
    {                                                                                                                  \
        NAME, value_type::real, access_t::read_write, RANGE, bounding_t::BOUND, compose_t::RULE, guard_t::GUARD,        \
            ENABLES, UNIT, nullptr, nullptr, KF, kf_kind::KIND, 0.0, 1, true,                                           \
            [](const IT& t) { return monitor::vector_t{static_cast<double>(t.MEMBER)}; },                               \
            [](IT& t, const monitor::vector_t& v) {                                                                     \
                double d;                                                                                              \
                if (v.size() != 1 || !as_num(v[0], d))                                                                 \
                    return false;                                                                                      \
                t.MEMBER = static_cast<float>(d);                                                                      \
                return true;                                                                                           \
            },                                                                                                         \
            []() { return monitor::vector_t{static_cast<double>(DEF)}; }                                                \
    }

/// A `bool` member.
#define B(NAME, MEMBER, DEF, RULE, GUARD, ENABLES, KF)                                                                 \
    field_desc                                                                                                         \
    {                                                                                                                  \
        NAME, value_type::boolean, access_t::read_write, std::nullopt, bounding_t::free, compose_t::RULE,               \
            guard_t::GUARD, ENABLES, "", nullptr, nullptr, KF, kf_kind::discrete, 1.0, 1, true,                         \
            [](const IT& t) { return monitor::vector_t{t.MEMBER}; },                                                    \
            [](IT& t, const monitor::vector_t& v) {                                                                     \
                bool b;                                                                                                \
                if (v.size() != 1 || !as_bool(v[0], b))                                                                \
                    return false;                                                                                      \
                t.MEMBER = b;                                                                                          \
                return true;                                                                                           \
            },                                                                                                         \
            []() { return monitor::vector_t{DEF}; }                                                                     \
    }

/// A `bool` member with a composition GUARD -- the same body as `B`, and the only reason
/// it is separate is that `B` predates any boolean needing one.
#define B_G(NAME, MEMBER, DEF, RULE, GUARD, ENABLES, KF)                                                               \
    field_desc                                                                                                         \
    {                                                                                                                  \
        NAME, value_type::boolean, access_t::read_write, std::nullopt, bounding_t::free, compose_t::RULE,               \
            guard_t::GUARD, ENABLES, "", nullptr, nullptr, KF, kf_kind::discrete, 1.0, 1, true,                         \
            [](const IT& t) { return monitor::vector_t{t.MEMBER}; },                                                    \
            [](IT& t, const monitor::vector_t& v) {                                                                     \
                bool b;                                                                                                \
                if (v.size() != 1 || !as_bool(v[0], b))                                                                 \
                    return false;                                                                                      \
                t.MEMBER = b;                                                                                          \
                return true;                                                                                           \
            },                                                                                                         \
            []() { return monitor::vector_t{DEF}; }                                                                     \
    }

/// An `std::array<double, N>` member. Composed element-wise under one rule.
#define A(NAME, MEMBER, N, DEFS, RANGE, BOUND, RULE, GUARD, ENABLES, UNIT, KF, KIND)                                    \
    field_desc                                                                                                         \
    {                                                                                                                  \
        NAME, value_type::vec##N, access_t::read_write, RANGE, bounding_t::BOUND, compose_t::RULE, guard_t::GUARD,      \
            ENABLES, UNIT, nullptr, nullptr, KF, kf_kind::KIND, 0.0, N, true,                                           \
            [](const IT& t) {                                                                                          \
                monitor::vector_t r;                                                                                   \
                for (std::size_t i = 0; i < (N); ++i)                                                                  \
                    r.push_back(t.MEMBER[i]);                                                                          \
                return r;                                                                                              \
            },                                                                                                         \
            [](IT& t, const monitor::vector_t& v) {                                                                     \
                if (v.size() != (N))                                                                                   \
                    return false;                                                                                      \
                double tmp[(N)];                                                                                       \
                for (std::size_t i = 0; i < (N); ++i)                                                                  \
                    if (!as_num(v[i], tmp[i]))                                                                         \
                        return false;                                                                                  \
                for (std::size_t i = 0; i < (N); ++i)                                                                  \
                    t.MEMBER[i] = tmp[i];                                                                              \
                return true;                                                                                           \
            },                                                                                                         \
            []() { return monitor::vector_t DEFS; }                                                                     \
    }

/// An enum member stored as `ENUMT`, read and written by NAME -- with the integer accepted
/// too, because a client that already has the number should not have to look up the word.
#define E(NAME, MEMBER, ENUMT, DEF, NAMES, RULE, GUARD, KF)                                                            \
    field_desc                                                                                                         \
    {                                                                                                                  \
        NAME, value_type::enumeration, access_t::read_write, std::nullopt, bounding_t::clip, compose_t::RULE,           \
            guard_t::GUARD, nullptr, "", NAMES, nullptr, KF, kf_kind::discrete, 1.0, 1, true,                           \
            [](const IT& t) {                                                                                          \
                const auto names = split_list(NAMES);                                                                  \
                const auto idx   = static_cast<std::size_t>(static_cast<int>(t.MEMBER));                               \
                return monitor::vector_t{idx < names.size() ? std::string(names[idx]) : std::to_string(idx)};           \
            },                                                                                                         \
            [](IT& t, const monitor::vector_t& v) {                                                                     \
                if (v.size() != 1)                                                                                     \
                    return false;                                                                                      \
                const auto  names = split_list(NAMES);                                                                 \
                std::string s;                                                                                         \
                if (as_str(v[0], s)) {                                                                                 \
                    for (std::size_t i = 0; i < names.size(); ++i)                                                     \
                        if (names[i] == s) {                                                                           \
                            t.MEMBER = static_cast<ENUMT>(static_cast<int>(i));                                        \
                            return true;                                                                               \
                        }                                                                                             \
                    return false;                                                                                      \
                }                                                                                                      \
                double d;                                                                                              \
                if (!as_num(v[0], d))                                                                                  \
                    return false;                                                                                      \
                const auto n = static_cast<int>(d + 0.5);                                                              \
                if (n < 0 || static_cast<std::size_t>(n) >= names.size())                                              \
                    return false;                                                                                      \
                t.MEMBER = static_cast<ENUMT>(n);                                                                      \
                return true;                                                                                           \
            },                                                                                                         \
            []() {                                                                                                     \
                const auto names = split_list(NAMES);                                                                  \
                const auto idx   = static_cast<std::size_t>(static_cast<int>(DEF));                                    \
                return monitor::vector_t{idx < names.size() ? std::string(names[idx]) : std::string()};                 \
            }                                                                                                          \
    }

/// An `int` member with no name table -- the colour-grade selectors index tables the OCIO
/// layer owns rather than a closed enum this file can name.
#define I(NAME, MEMBER, DEF, RULE, GUARD, KF)                                                                          \
    field_desc                                                                                                         \
    {                                                                                                                  \
        NAME, value_type::integer, access_t::read_write, std::nullopt, bounding_t::free, compose_t::RULE,               \
            guard_t::GUARD, nullptr, "", nullptr, nullptr, KF, kf_kind::discrete, 1.0, 1, true,                         \
            [](const IT& t) { return monitor::vector_t{static_cast<int32_t>(t.MEMBER)}; },                              \
            [](IT& t, const monitor::vector_t& v) {                                                                     \
                double d;                                                                                              \
                if (v.size() != 1 || !as_num(v[0], d))                                                                 \
                    return false;                                                                                      \
                t.MEMBER = static_cast<int>(d + (d < 0 ? -0.5 : 0.5));                                                 \
                return true;                                                                                           \
            },                                                                                                         \
            []() { return monitor::vector_t{static_cast<int32_t>(DEF)}; }                                               \
    }

/// A `std::string` member. Read-only over the API for now: `ocio.source_space` is validated
/// against the loaded OCIO config, which lives in the accelerator layer, so the AMCP command
/// stays the only writer until that validation is reachable from here.
#define S(NAME, MEMBER, GUARD)                                                                                         \
    field_desc                                                                                                         \
    {                                                                                                                  \
        NAME, value_type::string, access_t::read, std::nullopt, bounding_t::free, compose_t::innermost_wins,            \
            guard_t::GUARD, nullptr, "", nullptr, nullptr, nullptr, kf_kind::discrete, 0.0, 1, true,                    \
            [](const IT& t) { return monitor::vector_t{t.MEMBER}; },                                                    \
            [](IT&, const monitor::vector_t&) { return false; },                                                        \
            []() { return monitor::vector_t{std::string()}; }                                                           \
    }

/// An owned blob (a LUT, a curve set, a mask, a grade graph): reported present or absent,
/// never set by value -- loading stays the MIXER command that can parse the file. Composed
/// innermost-wins for the reason the hand-written version gives: two of them cannot be
/// composed without resampling one onto the other's raster, and choosing a resampling rule
/// silently is worse than choosing the layer's own.
#define BLOB(NAME, MEMBER, GUARD)                                                                                      \
    field_desc                                                                                                         \
    {                                                                                                                  \
        NAME, value_type::blob, access_t::read, std::nullopt, bounding_t::free, compose_t::innermost_wins,              \
            guard_t::GUARD, nullptr, "", nullptr, nullptr, nullptr, kf_kind::discrete, 0.0, 1, true,                    \
            [](const IT& t) { return monitor::vector_t{static_cast<bool>(t.MEMBER)}; },                                 \
            [](IT&, const monitor::vector_t&) { return false; },                                                        \
            []() { return monitor::vector_t{false}; }                                                                   \
    }

// ---------------------------------------------------------------------------------------
// THE TABLE
//
// Ordered as the grading chain reads rather than as the struct is laid out, because this is
// the order a generated control surface shows an operator. Geometry carries
// `compose_t::none` throughout: it follows a separate flow through `combine_transform`, and
// the composition function this file mirrors says so explicitly.
// ---------------------------------------------------------------------------------------

const std::vector<field_desc>& all()
{
    // clang-format off
    static const std::vector<field_desc> table = {
        // ---- basic ---------------------------------------------------------------------
        F("opacity",           opacity,           1.0, std::nullopt,        free, multiply, none, nullptr, "", "opacity",           continuous),
        F("brightness",        brightness,        1.0, std::nullopt,        free, multiply, none, nullptr, "", "brightness",        continuous),
        F("contrast",          contrast,          1.0, std::nullopt,        free, multiply, none, nullptr, "", "contrast",          continuous),
        F("saturation",        saturation,        1.0, std::nullopt,        free, multiply, none, nullptr, "", "saturation",        continuous),
        F("exposure",          exposure,          1.0, lim::exposure,       clip, multiply, none, nullptr, "", nullptr,             continuous),

        // ---- geometry (composed elsewhere; here to be described and animated) -----------
        A("anchor",            anchor,            2, ({0.0, 0.0}), std::nullopt, free, none, none, "enable_geometry_modifiers", "", "anchor_x,anchor_y",       continuous),
        A("fill_translation",  fill_translation,  2, ({0.0, 0.0}), std::nullopt, free, none, none, "enable_geometry_modifiers", "", "fill_x,fill_y",           continuous),
        A("fill_scale",        fill_scale,        2, ({1.0, 1.0}), std::nullopt, free, none, none, "enable_geometry_modifiers", "", "fill_sx,fill_sy",         continuous),
        A("clip_translation",  clip_translation,  2, ({0.0, 0.0}), std::nullopt, free, none, none, "enable_geometry_modifiers", "", "clip_x,clip_y",           continuous),
        A("clip_scale",        clip_scale,        2, ({1.0, 1.0}), std::nullopt, free, none, none, "enable_geometry_modifiers", "", "clip_sx,clip_sy",         continuous),
        F("angle",             angle,             0.0, std::nullopt,        wrap, none, none, "enable_geometry_modifiers", "rad", "angle",                     angular_rad),
        A("crop_ul",           crop.ul,           2, ({0.0, 0.0}), std::nullopt, free, none, none, "enable_geometry_modifiers", "", "crop_ul_x,crop_ul_y",     continuous),
        A("crop_lr",           crop.lr,           2, ({1.0, 1.0}), std::nullopt, free, none, none, "enable_geometry_modifiers", "", "crop_lr_x,crop_lr_y",     continuous),
        A("perspective_ul",    perspective.ul,    2, ({0.0, 0.0}), std::nullopt, free, none, none, "enable_geometry_modifiers", "", "persp_ul_x,persp_ul_y",   continuous),
        A("perspective_ur",    perspective.ur,    2, ({1.0, 0.0}), std::nullopt, free, none, none, "enable_geometry_modifiers", "", "persp_ur_x,persp_ur_y",   continuous),
        A("perspective_lr",    perspective.lr,    2, ({1.0, 1.0}), std::nullopt, free, none, none, "enable_geometry_modifiers", "", "persp_lr_x,persp_lr_y",   continuous),
        A("perspective_ll",    perspective.ll,    2, ({0.0, 1.0}), std::nullopt, free, none, none, "enable_geometry_modifiers", "", "persp_ll_x,persp_ll_y",   continuous),
        // NOT composed here. It is a geometry flag, and `apply_transform_colour_values`
        // never touches it -- `combine_transform` reads it to decide whether to apply the
        // crop and perspective, which is the separate flow the header describes. Declared
        // `or_` at first, and the self-test disagreed on every iteration.
        B("enable_geometry_modifiers", enable_geometry_modifiers, false, none, none, nullptr, "enable_geometry"),

        // ---- levels (master): ranges intersect, gamma multiplies ------------------------
        F("levels_min_input",  levels.min_input,  0.0, lim::level,          clip, max_,     none, nullptr, "", "levels_min_in",     continuous),
        F("levels_max_input",  levels.max_input,  1.0, lim::level,          clip, min_,     none, nullptr, "", "levels_max_in",     continuous),
        FU("levels_gamma",      levels.gamma,      1.0, lim::level_gamma,    clip, multiply, none, nullptr, "", "levels_gamma",      continuous),
        F("levels_min_output", levels.min_output, 0.0, lim::level,          clip, max_,     none, nullptr, "", "levels_min_out",    continuous),
        F("levels_max_output", levels.max_output, 1.0, lim::level,          clip, min_,     none, nullptr, "", "levels_max_out",    continuous),

        // ---- white balance / tone -------------------------------------------------------
        F("temperature",       temperature,       0.0, lim::temperature,    clip, add,      none, nullptr, "", "temperature",       continuous),
        F("tint",              tint,              0.0, lim::tint,           clip, add,      none, nullptr, "", "tint",              continuous),
        F("shadows",           shadows,           0.0, lim::tone,           clip, add,      none, nullptr, "", "shadows",           continuous),
        F("highlights",        highlights,        0.0, lim::tone,           clip, add,      none, nullptr, "", "highlights",        continuous),

        // ---- lift / midtone / gain ------------------------------------------------------
        A("lift",              lift,              3, ({0.0, 0.0, 0.0}), lim::lift,    clip, add,      none, nullptr, "", "lift_r,lift_g,lift_b",    continuous),
        A("midtone",           midtone,           3, ({1.0, 1.0, 1.0}), lim::midtone, clip, multiply, none, nullptr, "", "mid_r,mid_g,mid_b",       continuous),
        A("gain",              gain,              3, ({1.0, 1.0, 1.0}), lim::gain,    clip, multiply, none, nullptr, "", "gain_r,gain_g,gain_b",    continuous),

        // Hue rotation WRAPS rather than clamping: 200 degrees is -160, not 180. The shader
        // rotates with fract() so rotation is already periodic; wrapping is what bounds the
        // accumulation, which keeps precision and keeps "is this effect active" honest --
        // an accumulated 360 is exactly identity but reads as active and pays for the branch.
        F("hue_shift",         hue_shift,         0.0, lim::hue_shift,      wrap, add,      none, nullptr, "deg", "hue_shift",      angular),
        F("linear_saturation", linear_saturation, 1.0, lim::cdl_saturation, clip, multiply, none, nullptr, "", "linear_saturation", continuous),

        // ---- ASC CDL ---------------------------------------------------------------------
        A("cdl_slope",         cdl_slope,         3, ({1.0, 1.0, 1.0}), lim::cdl_slope,  clip, multiply, none, nullptr, "", "cdl_slope_r,cdl_slope_g,cdl_slope_b",    continuous),
        A("cdl_offset",        cdl_offset,        3, ({0.0, 0.0, 0.0}), lim::cdl_offset, clip, add,      none, nullptr, "", "cdl_offset_r,cdl_offset_g,cdl_offset_b", continuous),
        A("cdl_power",         cdl_power,         3, ({1.0, 1.0, 1.0}), lim::cdl_power,  clip, multiply, none, nullptr, "", "cdl_power_r,cdl_power_g,cdl_power_b",    continuous),
        F("cdl_saturation",    cdl_saturation,    1.0, lim::cdl_saturation, clip, multiply, none, nullptr, "", "cdl_saturation",   continuous),

        // ---- split toning: colours add, the balance is a group rule (see compose_colour) --
        A("split_shadow_color",    split_shadow_color,    3, ({0.0, 0.0, 0.0}), lim::split_color, clip, add, none, nullptr, "", "split_shadow_r,split_shadow_g,split_shadow_b",          continuous),
        A("split_highlight_color", split_highlight_color, 3, ({0.0, 0.0, 0.0}), lim::split_color, clip, add, none, nullptr, "", "split_highlight_r,split_highlight_g,split_highlight_b", continuous),
        F("split_balance",     split_balance,     0.5, lim::split_balance,  clip, custom, split_active, nullptr, "", "split_balance", continuous),

        // ---- gamut compression -------------------------------------------------------------
        B("gamut_compress",    gamut_compress,    false, or_, none, nullptr, "gamut_compress"),
        F("gc_cyan",           gc_cyan,           1.147, lim::gamut_limit,  clip, innermost_wins, gamut_compress, nullptr, "", "gc_cyan",    continuous),
        F("gc_magenta",        gc_magenta,        1.264, lim::gamut_limit,  clip, innermost_wins, gamut_compress, nullptr, "", "gc_magenta", continuous),
        F("gc_yellow",         gc_yellow,         1.312, lim::gamut_limit,  clip, innermost_wins, gamut_compress, nullptr, "", "gc_yellow",  continuous),

        // ---- image effects -------------------------------------------------------------------
        FU("sharpen_amount",    sharpen_amount,    0.0, lim::sharpen_amount,  clip, add,            none,               nullptr, "",   "sharpen_amount",  continuous),
        F("sharpen_radius",    sharpen_radius,    1.0, lim::sharpen_radius,  clip, innermost_wins, sharpen_radius_set, nullptr, "px", "sharpen_radius",  continuous),
        FU("grain_intensity",   grain_intensity,   0.0, lim::grain_intensity, clip, add,            none,               nullptr, "",   "grain_intensity", continuous),
        F("grain_size",        grain_size,        1.0, lim::grain_size,      clip, innermost_wins, grain_size_set,     nullptr, "",   "grain_size",      continuous),

        // ---- blur --------------------------------------------------------------------------
        // The whole blur struct is replaced as a group (see compose_colour); the rows exist
        // so each parameter can be described, animated and written individually.
        B("blur_enable",       blur.enable,       false, or_, none, nullptr, "blur_enable"),
        F("blur_radius",       blur.radius,       0.0, lim::blur_radius,    clip, innermost_wins, blur_enable, "blur.enable", "px",  "blur_radius", continuous),
        E("blur_type",         blur.type,         blur_type, blur_type::gaussian, BLUR_TYPES, innermost_wins, blur_enable, "blur_type"),
        F("blur_angle",        blur.angle,        0.0, lim::blur_angle,     wrap, innermost_wins, blur_enable, "blur.enable", "deg", "blur_angle",  angular),
        A("blur_center",       blur.center,       2, ({0.5, 0.5}), lim::unit, clip, innermost_wins, blur_enable, "blur.enable", "",  "blur_center_x,blur_center_y", continuous),
        F("blur_tilt_y",       blur.tilt_y,       0.5, lim::unit,           clip, innermost_wins, blur_enable, "blur.enable", "",    "blur_tilt_y", continuous),
        F("blur_tilt_h",       blur.tilt_h,       0.2, lim::unit,           clip, innermost_wins, blur_enable, "blur.enable", "",    "blur_tilt_h", continuous),

        // ---- per-channel RGB levels ----------------------------------------------------------
        // The whole per-channel block is composed by ONE group rule in `compose_colour`,
        // mirroring the mixers' `if (other.per_channel_levels.enable) { merge_ch(...) }`.
        // Every row below therefore declares `none`: leaving them with individual rules
        // applied the merge TWICE, and while max/min are idempotent the gamma multiply is
        // not -- it came out squared. 116 of 256 iterations, on both mixers, and only the
        // three gamma rows named, which is exactly what a double-apply looks like.
        B("rgb_levels_enable", per_channel_levels.enable, false, none, none, nullptr, "rgb_enable"),
        F("rgb_r_min_input",   per_channel_levels.r.min_input,  0.0, lim::level,       clip, none,     rgb_levels_enable, "per_channel_levels.enable", "", "rgb_r_min_in",  continuous),
        F("rgb_r_max_input",   per_channel_levels.r.max_input,  1.0, lim::level,       clip, none,     rgb_levels_enable, "per_channel_levels.enable", "", "rgb_r_max_in",  continuous),
        FU("rgb_r_gamma",       per_channel_levels.r.gamma,      1.0, lim::level_gamma, clip, none,     rgb_levels_enable, "per_channel_levels.enable", "", "rgb_r_gamma",   continuous),
        F("rgb_r_min_output",  per_channel_levels.r.min_output, 0.0, lim::level,       clip, none,     rgb_levels_enable, "per_channel_levels.enable", "", "rgb_r_min_out", continuous),
        F("rgb_r_max_output",  per_channel_levels.r.max_output, 1.0, lim::level,       clip, none,     rgb_levels_enable, "per_channel_levels.enable", "", "rgb_r_max_out", continuous),
        F("rgb_g_min_input",   per_channel_levels.g.min_input,  0.0, lim::level,       clip, none,     rgb_levels_enable, "per_channel_levels.enable", "", "rgb_g_min_in",  continuous),
        F("rgb_g_max_input",   per_channel_levels.g.max_input,  1.0, lim::level,       clip, none,     rgb_levels_enable, "per_channel_levels.enable", "", "rgb_g_max_in",  continuous),
        FU("rgb_g_gamma",       per_channel_levels.g.gamma,      1.0, lim::level_gamma, clip, none,     rgb_levels_enable, "per_channel_levels.enable", "", "rgb_g_gamma",   continuous),
        F("rgb_g_min_output",  per_channel_levels.g.min_output, 0.0, lim::level,       clip, none,     rgb_levels_enable, "per_channel_levels.enable", "", "rgb_g_min_out", continuous),
        F("rgb_g_max_output",  per_channel_levels.g.max_output, 1.0, lim::level,       clip, none,     rgb_levels_enable, "per_channel_levels.enable", "", "rgb_g_max_out", continuous),
        F("rgb_b_min_input",   per_channel_levels.b.min_input,  0.0, lim::level,       clip, none,     rgb_levels_enable, "per_channel_levels.enable", "", "rgb_b_min_in",  continuous),
        F("rgb_b_max_input",   per_channel_levels.b.max_input,  1.0, lim::level,       clip, none,     rgb_levels_enable, "per_channel_levels.enable", "", "rgb_b_max_in",  continuous),
        FU("rgb_b_gamma",       per_channel_levels.b.gamma,      1.0, lim::level_gamma, clip, none,     rgb_levels_enable, "per_channel_levels.enable", "", "rgb_b_gamma",   continuous),
        F("rgb_b_min_output",  per_channel_levels.b.min_output, 0.0, lim::level,       clip, none,     rgb_levels_enable, "per_channel_levels.enable", "", "rgb_b_min_out", continuous),
        F("rgb_b_max_output",  per_channel_levels.b.max_output, 1.0, lim::level,       clip, none,     rgb_levels_enable, "per_channel_levels.enable", "", "rgb_b_max_out", continuous),

        // ---- secondary qualifier (a group; see compose_colour) ---------------------------------
        B("qualifier_enable",  qualifier_enable,  false, or_, none, nullptr, "qualifier_enable"),
        F("qual_target_hue",   qual_target_hue,   0.0, lim::unit,      clip, innermost_wins, qualifier_enable, nullptr, "",    "qual_target_hue", continuous),
        F("qual_hue_width",    qual_hue_width,    0.1, lim::unit,      clip, innermost_wins, qualifier_enable, nullptr, "",    "qual_hue_width",  continuous),
        F("qual_min_sat",      qual_min_sat,      0.2, lim::unit,      clip, innermost_wins, qualifier_enable, nullptr, "",    "qual_min_sat",    continuous),
        F("qual_max_sat",      qual_max_sat,      1.0, lim::unit,      clip, innermost_wins, qualifier_enable, nullptr, "",    "qual_max_sat",    continuous),
        F("qual_min_lum",      qual_min_lum,      0.0, lim::unit,      clip, innermost_wins, qualifier_enable, nullptr, "",    "qual_min_lum",    continuous),
        F("qual_max_lum",      qual_max_lum,      1.0, lim::unit,      clip, innermost_wins, qualifier_enable, nullptr, "",    "qual_max_lum",    continuous),
        F("qual_softness",     qual_softness,     0.1, lim::unit,      clip, innermost_wins, qualifier_enable, nullptr, "",    "qual_softness",   continuous),
        F("qual_exposure",     qual_exposure,     0.0, lim::offset,    clip, innermost_wins, qualifier_enable, nullptr, "",    "qual_exposure",   continuous),
        F("qual_sat_offset",   qual_sat_offset,   0.0, lim::offset,    clip, innermost_wins, qualifier_enable, nullptr, "",    "qual_sat_offset", continuous),
        F("qual_hue_offset",   qual_hue_offset,   0.0, lim::hue_shift, wrap, innermost_wins, qualifier_enable, nullptr, "deg", "qual_hue_offset", angular),

        // ---- chroma key --------------------------------------------------------------------------
        B("chroma_enable",     chroma.enable,     false, or_, none, nullptr, "chroma_enable"),
        B("chroma_show_mask",  chroma.show_mask,  false, or_, none, nullptr, "chroma_show_mask"),
        F("chroma_target_hue",     chroma.target_hue,     0.0, lim::hue_degrees, clip, max_, none, nullptr, "deg", "chroma_target_hue", continuous),
        F("chroma_hue_width",      chroma.hue_width,      0.0, lim::hue_width,   clip, max_, none, nullptr, "deg", "chroma_hue_width",  continuous),
        F("chroma_min_saturation", chroma.min_saturation, 0.0, lim::unit,        clip, max_, none, nullptr, "",    "chroma_min_sat",    continuous),
        F("chroma_min_brightness", chroma.min_brightness, 0.0, lim::unit,        clip, max_, none, nullptr, "",    "chroma_min_bright", continuous),
        F("chroma_softness",       chroma.softness,       0.0, lim::unit,        clip, max_, none, nullptr, "",    "chroma_softness",   continuous),
        F("chroma_spill_suppress", chroma.spill_suppress, 0.0, lim::unit,        clip, max_, none, nullptr, "",    "chroma_spill",      continuous),
        F("chroma_spill_suppress_saturation", chroma.spill_suppress_saturation, 1.0, lim::unit, clip, min_, none, nullptr, "", "chroma_spill_sat", continuous),

        // ---- shape mask ---------------------------------------------------------------------------
        B("shape_enable",      shape.enable,      false, or_, none, nullptr, "shape_enable"),
        E("shape_type",        shape.type,        shape_type,      shape_type::rect,       SHAPE_TYPES, innermost_wins, shape_enable, "shape_type"),
        E("shape_fill_type",   shape.fill_type,   shape_fill_type, shape_fill_type::solid, FILL_TYPES,  innermost_wins, shape_enable, "shape_fill_type"),
        A("shape_center",      shape.center,      2, ({0.5, 0.5}), lim::unit, clip, innermost_wins, shape_enable, "shape.enable", "", "shape_center_x,shape_center_y", continuous),
        A("shape_size",        shape.size,        2, ({0.5, 0.5}), lim::unit, clip, innermost_wins, shape_enable, "shape.enable", "", "shape_size_x,shape_size_y",     continuous),
        F("shape_corner_radius",  shape.corner_radius,  0.0,   lim::unit,       clip, innermost_wins, shape_enable, "shape.enable", "",    "shape_corner_radius",  continuous),
        F("shape_edge_softness",  shape.edge_softness,  0.005, lim::unit,       clip, innermost_wins, shape_enable, "shape.enable", "",    "shape_edge_softness",  continuous),
        F("shape_gradient_angle", shape.gradient_angle, 0.0,   lim::blur_angle, wrap, innermost_wins, shape_enable, "shape.enable", "deg", "shape_gradient_angle", angular),
        A("shape_gradient_center", shape.gradient_center, 2, ({0.5, 0.5}), lim::unit, clip, innermost_wins, shape_enable, "shape.enable", "", "shape_gradient_cx,shape_gradient_cy", continuous),
        // Follows the rest of the shape block rather than OR-ing on its own: the mixers
        // replace `shape` wholesale when the INNER layer has `shape.enable`, so a stroke
        // flag on a layer whose shape is off must not leak upward. `shape_enable` itself
        // can stay `or_` because OR and the whole-struct replacement agree for it; this one
        // is where they part company, which is what the self-test reported.
        B_G("shape_stroke_enable", shape.stroke_enable, false, innermost_wins, shape_enable, nullptr, "shape_stroke_enable"),
        F("shape_stroke_width",   shape.stroke_width,   0.0, lim::unit, clip, innermost_wins, shape_enable, "shape.enable", "", "shape_stroke_width", continuous),
        A("shape_color1",      shape.color1,      4, ({1.0, 1.0, 1.0, 1.0}), lim::unit, clip, innermost_wins, shape_enable, "shape.enable", "", "shape_color1_r,shape_color1_g,shape_color1_b,shape_color1_a", continuous),
        A("shape_color2",      shape.color2,      4, ({0.0, 0.0, 0.0, 0.0}), lim::unit, clip, innermost_wins, shape_enable, "shape.enable", "", "shape_color2_r,shape_color2_g,shape_color2_b,shape_color2_a", continuous),
        A("shape_stroke_color", shape.stroke_color, 4, ({1.0, 1.0, 1.0, 1.0}), lim::unit, clip, innermost_wins, shape_enable, "shape.enable", "", "shape_stroke_r,shape_stroke_g,shape_stroke_b,shape_stroke_a", continuous),

        // ---- projection: 360 virtual camera -----------------------------------------------------------
        B("proj_enable",       projection.enable, false, or_, none, nullptr, "proj_enable"),
        F("proj_yaw",          projection.yaw,     0.0,           std::nullopt, wrap, innermost_wins, projection_enable, nullptr, "rad", "proj_yaw",       angular_rad),
        F("proj_pitch",        projection.pitch,   0.0,           std::nullopt, wrap, innermost_wins, projection_enable, nullptr, "rad", "proj_pitch",     angular_rad),
        F("proj_roll",         projection.roll,    0.0,           std::nullopt, wrap, innermost_wins, projection_enable, nullptr, "rad", "proj_roll",      angular_rad),
        F("proj_fov",          projection.fov,     1.57079632679, std::nullopt, free, innermost_wins, projection_enable, nullptr, "rad", "proj_fov",       angular_rad),
        F("proj_offset_x",     projection.offset_x,  0.0, std::nullopt, free, innermost_wins, projection_enable, nullptr, "", "proj_offset_x",  continuous),
        F("proj_offset_y",     projection.offset_y,  0.0, std::nullopt, free, innermost_wins, projection_enable, nullptr, "", "proj_offset_y",  continuous),
        F("proj_frustum_h",    projection.frustum_h, 0.0, std::nullopt, free, innermost_wins, projection_enable, nullptr, "", "proj_frustum_h", continuous),
        F("proj_frustum_v",    projection.frustum_v, 0.0, std::nullopt, free, innermost_wins, projection_enable, nullptr, "", "proj_frustum_v", continuous),
        F("proj_lens_k1",      projection.lens_k1,   0.0, std::nullopt, free, innermost_wins, projection_enable, nullptr, "", "proj_lens_k1",   continuous),
        F("proj_lens_k2",      projection.lens_k2,   0.0, std::nullopt, free, innermost_wins, projection_enable, nullptr, "", "proj_lens_k2",   continuous),
        F("proj_lens_k3",      projection.lens_k3,   0.0, std::nullopt, free, innermost_wins, projection_enable, nullptr, "", "proj_lens_k3",   continuous),
        F("proj_lens_p1",      projection.lens_p1,   0.0, std::nullopt, free, innermost_wins, projection_enable, nullptr, "", "proj_lens_p1",   continuous),
        F("proj_lens_p2",      projection.lens_p2,   0.0, std::nullopt, free, innermost_wins, projection_enable, nullptr, "", "proj_lens_p2",   continuous),
        E("proj_source_lens",  projection.source_lens, screen_curve_type, screen_curve_type::flat, CURVE_TYPES, innermost_wins, projection_enable, "proj_source_lens"),

        // ---- projection: curved screen compensation (merges independently of 360) ---------------------
        B("proj_curve_enable", projection.curve_enable, false, or_, none, nullptr, "proj_curve_enable"),
        E("proj_curve_type",   projection.curve_type, screen_curve_type, screen_curve_type::flat, CURVE_TYPES, innermost_wins, curve_enable, "proj_curve_type"),
        F("proj_screen_arc",   projection.screen_arc,   0.0, std::nullopt, free, innermost_wins, curve_enable, nullptr, "rad", "proj_screen_arc",   angular_rad),
        F("proj_screen_arc_v", projection.screen_arc_v, 0.0, std::nullopt, free, innermost_wins, curve_enable, nullptr, "rad", "proj_screen_arc_v", angular_rad),
        F("proj_eye_distance", projection.eye_distance, 1.0, std::nullopt, free, innermost_wins, curve_enable, nullptr, "",    "proj_eye_distance", continuous),
        B("proj_curve_auto",   projection.curve_auto, false, innermost_wins, curve_enable, nullptr, "proj_curve_auto"),
        // Deliberately NOT animatable, unlike `proj_curve_auto` beside it: "this block is
        // owned by auto-projection" is a fact about ownership, not a quantity to tween. The
        // kf slot is `nullptr` rather than a name, so the keyframes module's frozen list needs
        // no entry and a saved timeline cannot come to depend on it.
        B("proj_icvfx_auto",   projection.icvfx_auto, true,  innermost_wins, icvfx_enable, nullptr, nullptr),

        // ---- projection: edge blending (a GROUP gate; see compose_colour) -----------------------------
        F("proj_edge_blend_left",   projection.edge_blend_left,   0.0, lim::unit,    clip, innermost_wins, edge_blend_any, nullptr, "", "proj_edge_blend_left",   continuous),
        F("proj_edge_blend_right",  projection.edge_blend_right,  0.0, lim::unit,    clip, innermost_wins, edge_blend_any, nullptr, "", "proj_edge_blend_right",  continuous),
        F("proj_edge_blend_top",    projection.edge_blend_top,    0.0, lim::unit,    clip, innermost_wins, edge_blend_any, nullptr, "", "proj_edge_blend_top",    continuous),
        F("proj_edge_blend_bottom", projection.edge_blend_bottom, 0.0, lim::unit,    clip, innermost_wins, edge_blend_any, nullptr, "", "proj_edge_blend_bottom", continuous),
        F("proj_edge_blend_gamma",  projection.edge_blend_gamma,  2.2, std::nullopt, free, innermost_wins, edge_blend_any, nullptr, "", "proj_edge_blend_gamma",  continuous),

        // ---- projection: ICVFX inner/outer frustum ------------------------------------------------------
        // The gains are per-channel and asymmetric by nature. A red/blue exchange here was
        // invisible for months because the natural test -- equal gains -- is invariant under it.
        B("proj_icvfx_enable", projection.icvfx_enable, false, or_, none, nullptr, "proj_icvfx_enable"),
        F("proj_inner_yaw",          projection.inner_yaw,          0.0,           std::nullopt, wrap, innermost_wins, icvfx_enable, nullptr, "rad", nullptr,            angular_rad),
        F("proj_inner_pitch",        projection.inner_pitch,        0.0,           std::nullopt, wrap, innermost_wins, icvfx_enable, nullptr, "rad", nullptr,            angular_rad),
        F("proj_inner_roll",         projection.inner_roll,         0.0,           std::nullopt, wrap, innermost_wins, icvfx_enable, nullptr, "rad", nullptr,            angular_rad),
        F("proj_inner_fov",          projection.inner_fov,          1.57079632679, std::nullopt, free, innermost_wins, icvfx_enable, nullptr, "rad", "proj_inner_fov",   angular_rad),
        F("proj_inner_eye_distance", projection.inner_eye_distance, 1.0,           std::nullopt, free, innermost_wins, icvfx_enable, nullptr, "",    nullptr,            continuous),
        F("proj_inner_offset_x",     projection.inner_offset_x,     0.0,           std::nullopt, free, innermost_wins, icvfx_enable, nullptr, "",    nullptr,            continuous),
        F("proj_inner_offset_y",     projection.inner_offset_y,     0.0,           std::nullopt, free, innermost_wins, icvfx_enable, nullptr, "",    nullptr,            continuous),
        F("proj_icvfx_q0x",          projection.icvfx_q0x,         -1.0,           std::nullopt, free, innermost_wins, icvfx_enable, nullptr, "",    nullptr,            continuous),
        F("proj_icvfx_q0y",          projection.icvfx_q0y,          1.0,           std::nullopt, free, innermost_wins, icvfx_enable, nullptr, "",    nullptr,            continuous),
        F("proj_icvfx_q1x",          projection.icvfx_q1x,          1.0,           std::nullopt, free, innermost_wins, icvfx_enable, nullptr, "",    nullptr,            continuous),
        F("proj_icvfx_q1y",          projection.icvfx_q1y,          1.0,           std::nullopt, free, innermost_wins, icvfx_enable, nullptr, "",    nullptr,            continuous),
        F("proj_icvfx_q2x",          projection.icvfx_q2x,          1.0,           std::nullopt, free, innermost_wins, icvfx_enable, nullptr, "",    nullptr,            continuous),
        F("proj_icvfx_q2y",          projection.icvfx_q2y,         -1.0,           std::nullopt, free, innermost_wins, icvfx_enable, nullptr, "",    nullptr,            continuous),
        F("proj_icvfx_q3x",          projection.icvfx_q3x,         -1.0,           std::nullopt, free, innermost_wins, icvfx_enable, nullptr, "",    nullptr,            continuous),
        F("proj_icvfx_q3y",          projection.icvfx_q3y,         -1.0,           std::nullopt, free, innermost_wins, icvfx_enable, nullptr, "",    nullptr,            continuous),
        F("proj_icvfx_feather",      projection.icvfx_feather,      0.05, lim::unit, clip, innermost_wins, icvfx_enable, nullptr, "", "proj_icvfx_feather",   continuous),
        F("proj_icvfx_outer_dim",    projection.icvfx_outer_dim,    1.0,  lim::unit, clip, innermost_wins, icvfx_enable, nullptr, "", "proj_icvfx_outer_dim", continuous),
        F("proj_icvfx_inner_dim",    projection.icvfx_inner_dim,    1.0,  lim::unit, clip, innermost_wins, icvfx_enable, nullptr, "", "proj_icvfx_inner_dim", continuous),
        F("proj_icvfx_inner_gain_r", projection.icvfx_inner_gain_r, 1.0,  lim::gain, clip, innermost_wins, icvfx_enable, nullptr, "", "proj_icvfx_inner_gain_r", continuous),
        F("proj_icvfx_inner_gain_g", projection.icvfx_inner_gain_g, 1.0,  lim::gain, clip, innermost_wins, icvfx_enable, nullptr, "", "proj_icvfx_inner_gain_g", continuous),
        F("proj_icvfx_inner_gain_b", projection.icvfx_inner_gain_b, 1.0,  lim::gain, clip, innermost_wins, icvfx_enable, nullptr, "", "proj_icvfx_inner_gain_b", continuous),
        F("proj_icvfx_outer_gain_r", projection.icvfx_outer_gain_r, 1.0,  lim::gain, clip, innermost_wins, icvfx_enable, nullptr, "", "proj_icvfx_outer_gain_r", continuous),
        F("proj_icvfx_outer_gain_g", projection.icvfx_outer_gain_g, 1.0,  lim::gain, clip, innermost_wins, icvfx_enable, nullptr, "", "proj_icvfx_outer_gain_g", continuous),
        F("proj_icvfx_outer_gain_b", projection.icvfx_outer_gain_b, 1.0,  lim::gain, clip, innermost_wins, icvfx_enable, nullptr, "", "proj_icvfx_outer_gain_b", continuous),

        // ---- colour management -------------------------------------------------------------------------
        B("color_grade_enable", color_grade.enable, false, or_, none, nullptr, "color_grade_enable"),
        I("color_grade_input_transfer",  color_grade.input_transfer,  0, innermost_wins, color_grade_enable, "color_grade_input_transfer"),
        I("color_grade_input_gamut",     color_grade.input_gamut,     0, innermost_wins, color_grade_enable, "color_grade_input_gamut"),
        I("color_grade_tone_mapping",    color_grade.tone_mapping,    0, innermost_wins, color_grade_enable, "color_grade_tone_mapping"),
        I("color_grade_output_gamut",    color_grade.output_gamut,    0, innermost_wins, color_grade_enable, "color_grade_output_gamut"),
        I("color_grade_output_transfer", color_grade.output_transfer, 0, innermost_wins, color_grade_enable, "color_grade_output_transfer"),
        FL("color_grade_exposure", color_grade.exposure, 1.0f, lim::exposure, clip, innermost_wins, color_grade_enable, nullptr, "", "color_grade_exposure", continuous),
        B("ocio_enable",       ocio.enable,       false, or_, none, nullptr, nullptr),
        S("ocio_source_space", ocio.source_space, ocio_enable),

        // ---- tone curves ---------------------------------------------------------------------------------
        B("curves_enable",     curves.enable,     false, or_, none, nullptr, "curves_enable"),

        // ---- flags ---------------------------------------------------------------------------------------
        B("is_key",            is_key,            false, or_,  none, nullptr, nullptr),
        B("is_mix",            is_mix,            false, or_,  none, nullptr, nullptr),
        B("invert",            invert,            false, or_,  none, nullptr, "invert"),
        // Flips XOR: two mirrors cancel.
        B("flip_h",            flip_h,            false, xor_, none, nullptr, "flip_h"),
        B("flip_v",            flip_v,            false, xor_, none, nullptr, "flip_v"),
        E("blend_mode",        blend_mode,        blend_mode, blend_mode::normal, BLEND_MODES, max_, none, "blend_mode"),
        I("layer_depth",       layer_depth,       0, add, none, nullptr),

        // ---- owned blobs: present/absent, loaded by their own commands --------------------------------------
        BLOB("lut3d",          lut3d,             lut3d_present),
        FL("lut3d_strength",   lut3d_strength,    1.0f, lim::lut3d_strength, clip, innermost_wins, lut3d_present, nullptr, "", "lut3d_strength", continuous),
        BLOB("hue_curves",     hue_curves,        hue_curves_present),
        BLOB("blend_mask",     blend_mask,        blend_mask_present),
        BLOB("grade_nodes",    grade_nodes,       grade_nodes_present),
    };
    // clang-format on

    // A `clip` with nothing to clip against is not a rule, it is a gap -- and a generated
    // control surface would read it as "clamp to [?]" and either invent limits or refuse
    // the field. Seventeen rows declared exactly that on their first outing (opacity,
    // brightness, contrast, saturation and the whole projection block), because `clip`
    // reads as the safe default when writing a table row. It is not: the server accepts
    // any value for those fields, and `free` is what says so.
    //
    // Enumerations are the deliberate exception: their bound is the `values` list, which
    // is a range in every sense except MIN/MAX.
    for (const auto& f : table) {
        const bool bounded = f.range.has_value() || (f.type == value_type::enumeration && f.values);
        if (f.bounding != bounding_t::free && f.bounding != bounding_t::wrap && !bounded)
            CASPAR_THROW_EXCEPTION(programming_error()
                                   << msg_info(std::string("transform field '") + f.path +
                                               "' declares a bounding rule with no range to bound it"));
    }

    return table;
}

#undef F
#undef F_
#undef FU
#undef FL
#undef B
#undef B_G
#undef A
#undef E
#undef I
#undef S
#undef BLOB

// ---------------------------------------------------------------------------------------
// Lookup
// ---------------------------------------------------------------------------------------

namespace {

const std::unordered_map<std::string_view, std::size_t>& path_index()
{
    static const auto idx = [] {
        std::unordered_map<std::string_view, std::size_t> m;
        const auto&                                       t = all();
        for (std::size_t i = 0; i < t.size(); ++i)
            m.emplace(std::string_view{t[i].path}, i);
        return m;
    }();
    return idx;
}

const std::unordered_map<std::string_view, kf_ref>& kf_index()
{
    static const auto idx = [] {
        std::unordered_map<std::string_view, kf_ref> m;
        for (const auto& f : all()) {
            const auto names = split_list(f.kf_names);
            for (std::size_t c = 0; c < names.size(); ++c)
                m.emplace(names[c], kf_ref{&f, static_cast<uint8_t>(c)});
        }
        return m;
    }();
    return idx;
}

} // namespace

const field_desc* find(std::string_view path)
{
    const auto& idx = path_index();
    const auto  it  = idx.find(path);
    return it == idx.end() ? nullptr : &all()[it->second];
}

std::optional<kf_ref> find_kf(std::string_view kf_name)
{
    const auto& idx = kf_index();
    const auto  it  = idx.find(kf_name);
    if (it == idx.end())
        return std::nullopt;
    return it->second;
}

std::vector<std::string> all_kf_names()
{
    std::vector<std::string> out;
    for (const auto& f : all())
        for (const auto n : split_list(f.kf_names))
            out.emplace_back(n);
    return out;
}

// ---------------------------------------------------------------------------------------
// Composition
// ---------------------------------------------------------------------------------------

void compose_field(const field_desc& f, image_transform& self, const image_transform& other)
{
    if (f.compose == compose_t::none || f.compose == compose_t::custom)
        return;
    if (!guard_holds(f.guard, other))
        return;

    const auto sv = f.get(self);
    const auto ov = f.get(other);
    if (sv.size() != ov.size())
        return;

    // Flags and enums do not go through the numeric path: OR on a bool is not the same
    // operation as max on a double, and an enum's "max" is over its ordinal.
    if (f.type == value_type::boolean) {
        bool s = false, o = false;
        if (!as_bool(sv[0], s) || !as_bool(ov[0], o))
            return;
        switch (f.compose) {
            case compose_t::or_: s = s || o; break;
            case compose_t::xor_: s = s != o; break;
            case compose_t::innermost_wins: s = o; break;
            default: return;
        }
        f.set(self, monitor::vector_t{s});
        return;
    }

    if (f.type == value_type::string || f.type == value_type::blob || f.type == value_type::enumeration) {
        // These carry no arithmetic. `innermost_wins` and the enum `max_` are handled by
        // the whole-struct rules in compose_colour (blur, shape, ocio) or, for blend_mode
        // and the blobs, below -- but their setters are read-only or name-based, so the
        // assignment happens on the members directly rather than through `set`.
        return;
    }

    monitor::vector_t out;
    for (std::size_t i = 0; i < sv.size(); ++i) {
        double s = 0.0, o = 0.0;
        if (!as_num(sv[i], s) || !as_num(ov[i], o))
            return;
        switch (f.compose) {
            case compose_t::multiply: s = f.compose_clamps ? clamp_opt(f.range, s * o) : s * o; break;
            case compose_t::add:
                s = f.bounding == bounding_t::wrap && f.range
                        ? std::remainder(s + o, f.range->hi - f.range->lo)
                        : (f.compose_clamps ? clamp_opt(f.range, s + o) : s + o);
                break;
            case compose_t::min_: s = std::min(s, o); break;
            case compose_t::max_: s = std::max(s, o); break;
            case compose_t::innermost_wins: s = o; break;
            default: return;
        }
        out.push_back(s);
    }
    f.set(self, out);
}

void compose_colour(image_transform& self, const image_transform& other)
{
    // Geometry is deliberately absent: it follows a separate flow through
    // `combine_transform`, and every geometry row carries `compose_t::none`.
    for (const auto& f : all())
        compose_field(f, self, other);

    // ---- the rules the per-field table cannot express -----------------------------------

    // Enums and integers whose composition is ordinal rather than arithmetic.
    self.blend_mode = std::max(self.blend_mode, other.blend_mode);

    // split_balance is a crossover POSITION in luma, not a strength, so no arithmetic
    // composition of two of them means anything: adding 0.5 and 0.5 gives 1.0 (everything
    // is shadow), multiplying gives 0.25. The shader has one crossover and one colour pair,
    // so two stacked split tones cannot both be represented however they are combined --
    // summing the colours (done per-field above) and keeping a single crossover is the
    // least-lossy approximation available. "Is other active" is the same test the mixer
    // uses to decide whether to enable the effect at all; comparing `other.split_balance`
    // against its 0.5 default instead would misread a tweened 0.4999999 as a setting.
    if (guard_holds(guard_t::split_active, other))
        self.split_balance = lim::split_balance.clamp(other.split_balance);

    // Edge blending is a GROUP: the four edges and the gamma move together or not at all,
    // gated on any edge being set. Per-field innermost-wins would let one layer's left edge
    // combine with another's right, which is not a state any single command can produce.
    if (guard_holds(guard_t::edge_blend_any, other)) {
        self.projection.edge_blend_left   = other.projection.edge_blend_left;
        self.projection.edge_blend_right  = other.projection.edge_blend_right;
        self.projection.edge_blend_top    = other.projection.edge_blend_top;
        self.projection.edge_blend_bottom = other.projection.edge_blend_bottom;
        self.projection.edge_blend_gamma  = other.projection.edge_blend_gamma;
    }

    // Whole-struct replacements, for structs carrying members that are not individually
    // addressable (curve control points, an OCIO cache id, an enum inside a group).
    if (other.color_grade.enable)
        self.color_grade = other.color_grade;
    if (other.ocio.enable)
        self.ocio = other.ocio;
    if (other.blur.enable)
        self.blur = other.blur;
    if (other.shape.enable)
        self.shape = other.shape;
    if (other.curves.enable)
        self.curves = other.curves;

    // Per-channel RGB levels intersect input and output ranges and multiply gamma, as the
    // master levels do -- but only when the inner layer has them enabled at all.
    if (other.per_channel_levels.enable) {
        self.per_channel_levels.enable = true;
        const auto merge_ch            = [](rgb_levels_channel& s, const rgb_levels_channel& o) {
            s.min_input  = std::max(s.min_input, o.min_input);
            s.max_input  = std::min(s.max_input, o.max_input);
            s.gamma *= o.gamma;
            s.min_output = std::max(s.min_output, o.min_output);
            s.max_output = std::min(s.max_output, o.max_output);
        };
        merge_ch(self.per_channel_levels.r, other.per_channel_levels.r);
        merge_ch(self.per_channel_levels.g, other.per_channel_levels.g);
        merge_ch(self.per_channel_levels.b, other.per_channel_levels.b);
    }

    // The qualifier is a group for the same reason edge blending is: its ten parameters
    // describe one selection, and mixing two selections describes neither.
    if (other.qualifier_enable) {
        self.qualifier_enable = true;
        self.qual_target_hue  = other.qual_target_hue;
        self.qual_hue_width   = other.qual_hue_width;
        self.qual_min_sat     = other.qual_min_sat;
        self.qual_max_sat     = other.qual_max_sat;
        self.qual_min_lum     = other.qual_min_lum;
        self.qual_max_lum     = other.qual_max_lum;
        self.qual_softness    = other.qual_softness;
        self.qual_exposure    = other.qual_exposure;
        self.qual_sat_offset  = other.qual_sat_offset;
        self.qual_hue_offset  = other.qual_hue_offset;
    }

    // Projection groups: the 360 camera, the curve compensation and the ICVFX frustum each
    // move as a unit, gated on their own enable.
    if (other.projection.enable) {
        self.projection.enable      = true;
        self.projection.yaw         = other.projection.yaw;
        self.projection.pitch       = other.projection.pitch;
        self.projection.roll        = other.projection.roll;
        self.projection.fov         = other.projection.fov;
        self.projection.offset_x    = other.projection.offset_x;
        self.projection.offset_y    = other.projection.offset_y;
        self.projection.frustum_h   = other.projection.frustum_h;
        self.projection.frustum_v   = other.projection.frustum_v;
        self.projection.lens_k1     = other.projection.lens_k1;
        self.projection.lens_k2     = other.projection.lens_k2;
        self.projection.lens_k3     = other.projection.lens_k3;
        self.projection.lens_p1     = other.projection.lens_p1;
        self.projection.lens_p2     = other.projection.lens_p2;
        self.projection.source_lens = other.projection.source_lens;
    }
    if (other.projection.curve_enable) {
        self.projection.curve_type   = other.projection.curve_type;
        self.projection.screen_arc   = other.projection.screen_arc;
        self.projection.screen_arc_v = other.projection.screen_arc_v;
        self.projection.eye_distance = other.projection.eye_distance;
        self.projection.curve_auto   = other.projection.curve_auto;
    }
    self.projection.curve_enable = self.projection.curve_enable || other.projection.curve_enable;
    if (other.projection.icvfx_enable) {
        self.projection.icvfx_enable       = true;
        self.projection.icvfx_auto         = other.projection.icvfx_auto;
        self.projection.inner_yaw          = other.projection.inner_yaw;
        self.projection.inner_pitch        = other.projection.inner_pitch;
        self.projection.inner_roll         = other.projection.inner_roll;
        self.projection.inner_fov          = other.projection.inner_fov;
        self.projection.inner_eye_distance = other.projection.inner_eye_distance;
        self.projection.inner_offset_x     = other.projection.inner_offset_x;
        self.projection.inner_offset_y     = other.projection.inner_offset_y;
        self.projection.icvfx_q0x          = other.projection.icvfx_q0x;
        self.projection.icvfx_q0y          = other.projection.icvfx_q0y;
        self.projection.icvfx_q1x          = other.projection.icvfx_q1x;
        self.projection.icvfx_q1y          = other.projection.icvfx_q1y;
        self.projection.icvfx_q2x          = other.projection.icvfx_q2x;
        self.projection.icvfx_q2y          = other.projection.icvfx_q2y;
        self.projection.icvfx_q3x          = other.projection.icvfx_q3x;
        self.projection.icvfx_q3y          = other.projection.icvfx_q3y;
        self.projection.icvfx_feather      = other.projection.icvfx_feather;
        self.projection.icvfx_outer_dim    = other.projection.icvfx_outer_dim;
        self.projection.icvfx_inner_dim    = other.projection.icvfx_inner_dim;
        self.projection.icvfx_inner_gain_r = other.projection.icvfx_inner_gain_r;
        self.projection.icvfx_inner_gain_g = other.projection.icvfx_inner_gain_g;
        self.projection.icvfx_inner_gain_b = other.projection.icvfx_inner_gain_b;
        self.projection.icvfx_outer_gain_r = other.projection.icvfx_outer_gain_r;
        self.projection.icvfx_outer_gain_g = other.projection.icvfx_outer_gain_g;
        self.projection.icvfx_outer_gain_b = other.projection.icvfx_outer_gain_b;
    }

    // The blobs: innermost wins, and the LUT carries its strength with it.
    if (other.lut3d) {
        self.lut3d          = other.lut3d;
        self.lut3d_strength = other.lut3d_strength;
    }
    if (other.hue_curves)
        self.hue_curves = other.hue_curves;
    if (other.blend_mask)
        self.blend_mask = other.blend_mask;
    if (other.grade_nodes)
        self.grade_nodes = other.grade_nodes;
}

void apply_enables(image_transform& tf, const field_desc& field)
{
    if (!field.enables)
        return;

    const std::string_view e{field.enables};
    if (e == "enable_geometry_modifiers")
        tf.enable_geometry_modifiers = true;
    else if (e == "blur.enable")
        tf.blur.enable = tf.blur.radius > 0.0;
    else if (e == "per_channel_levels.enable")
        tf.per_channel_levels.enable = true;
    else if (e == "shape.enable")
        tf.shape.enable = true;
}

// ---------------------------------------------------------------------------------------
// The self-test
//
// This is the whole justification for shipping a generated composition beside two
// hand-written ones rather than replacing them outright: it says, on this build and this
// backend, whether the generated rule agrees with the rule that actually renders. Zero
// means the swap is safe; a named field means that row is wrong.
// ---------------------------------------------------------------------------------------

self_test_report compose_self_test(void (*reference)(image_transform&, const image_transform&),
                                   unsigned iterations,
                                   uint32_t seed)
{
    self_test_report rep;
    rep.iterations = iterations;
    rep.fields     = static_cast<unsigned>(all().size());
    if (!reference)
        return rep;

    std::mt19937                           rng{seed};
    std::uniform_real_distribution<double> unit{0.0, 1.0};
    std::bernoulli_distribution            coin{0.5};

    // A random IN-RANGE value per field: inside the declared range when there is one, and a
    // small spread around zero when there is not. The values must be legal, or the clamps
    // fire on both sides and hide a genuine difference in the rule.
    const auto randomise = [&](image_transform& t) {
        for (const auto& f : all()) {
            if (f.type == value_type::blob || f.type == value_type::string)
                continue;
            if (f.type == value_type::boolean) {
                f.set(t, monitor::vector_t{coin(rng)});
                continue;
            }
            if (f.type == value_type::enumeration) {
                const auto names = split_list(f.values);
                if (!names.empty())
                    f.set(t, monitor::vector_t{static_cast<int32_t>(rng() % names.size())});
                continue;
            }
            if (f.type == value_type::integer) {
                f.set(t, monitor::vector_t{static_cast<int32_t>(rng() % 5)});
                continue;
            }
            monitor::vector_t v;
            for (uint8_t i = 0; i < f.arity; ++i) {
                const double lo = f.range ? f.range->lo : -1.0;
                const double hi = f.range ? f.range->hi : 1.0;
                v.push_back(lo + unit(rng) * (hi - lo));
            }
            f.set(t, v);
        }
    };

    std::vector<bool> already(all().size(), false);

    for (unsigned i = 0; i < iterations; ++i) {
        image_transform a;
        image_transform b;
        randomise(a);
        randomise(b);

        image_transform generated  = a;
        image_transform referenced = a;
        compose_colour(generated, b);
        reference(referenced, b);

        if (generated == referenced)
            continue;

        ++rep.divergences;

        // Name the field rather than only the fact. A report that says "they differ" sends
        // the next reader to a 250-line diff; one that says "gc_cyan" sends them to a line.
        const auto& t = all();
        for (std::size_t k = 0; k < t.size(); ++k) {
            if (already[k])
                continue;
            if (t[k].get(generated) == t[k].get(referenced))
                continue;
            already[k] = true;
            rep.diverged.emplace_back(t[k].path);
        }
    }

    return rep;
}

}}} // namespace caspar::core::fields
