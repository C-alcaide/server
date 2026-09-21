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

#include "stage_fields.h"

#include <common/except.h>
#include <common/log.h>

#include <boost/variant.hpp>

#include <algorithm>
#include <cstdint>
#include <string>

namespace caspar { namespace core { namespace fields {

namespace {

// Local copies rather than exports from `transform_fields.cpp`: they are four lines each and
// live in that file's anonymous namespace, and widening their linkage to save eight lines
// would make two tables share a private helper for no reason.
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

// ---- Row macros ---------------------------------------------------------------------------
//
// `compose`, `guard`, `enables` and `kf_names` are `none`/`nullptr` in every row, so unlike the
// transform table's macros these do not take them as parameters. A screen does not compose and
// cannot be keyframed; passing the same two literals 40 times would only invite someone to pass
// a different one.

/// A `float` member carried as a real.
#define SF(T, NAME, MEMBER, DEF, RANGE, BOUND, UNIT, DESC)                                                             \
    typed_field<T>                                                                                                     \
    {                                                                                                                  \
        NAME, value_type::real, access_t::read_write, RANGE, bounding_t::BOUND, compose_t::none, guard_t::none,         \
            nullptr, UNIT, nullptr, DESC, nullptr, kf_kind::continuous, 0.0, 1, false,                                  \
            [](const T& s) { return monitor::vector_t{static_cast<double>(s.MEMBER)}; },                                \
            [](T& s, const monitor::vector_t& v) {                                                                     \
                double d;                                                                                              \
                if (v.size() != 1 || !as_num(v[0], d))                                                                 \
                    return false;                                                                                      \
                s.MEMBER = static_cast<float>(d);                                                                      \
                return true;                                                                                           \
            },                                                                                                         \
            []() { return monitor::vector_t{static_cast<double>(DEF)}; }                                                \
    }

/// A `float` member that is DERIVED and therefore read-only.
#define SF_RO(T, NAME, MEMBER, DEF, UNIT, DESC)                                                                        \
    typed_field<T>                                                                                                     \
    {                                                                                                                  \
        NAME, value_type::real, access_t::read, std::nullopt, bounding_t::free, compose_t::none, guard_t::none,         \
            nullptr, UNIT, nullptr, DESC, nullptr, kf_kind::continuous, 0.0, 1, false,                                  \
            [](const T& s) { return monitor::vector_t{static_cast<double>(s.MEMBER)}; },                                \
            [](T&, const monitor::vector_t&) { return false; },                                                        \
            []() { return monitor::vector_t{static_cast<double>(DEF)}; }                                                \
    }

/// Two or three `float` members as one semantic tuple.
///
/// A tuple rather than three scalars, and that is not cosmetic. The renderer's mutators are
/// GROUPED -- `set_screen_position(name, x, y, z)` -- and each also re-applies the mesh
/// transform and calls `update_projections()`. A tuple row maps 1:1 onto a mutator; three
/// scalar rows would need a read-modify-write and could miss the recompute.
#define SV3(T, NAME, A, B, C, DA, DB, DC, RANGE, BOUND, UNIT, DESC)                                                    \
    typed_field<T>                                                                                                     \
    {                                                                                                                  \
        NAME, value_type::vec3, access_t::read_write, RANGE, bounding_t::BOUND, compose_t::none, guard_t::none,         \
            nullptr, UNIT, nullptr, DESC, nullptr, kf_kind::continuous, 0.0, 3, false,                                  \
            [](const T& s) {                                                                                           \
                return monitor::vector_t{static_cast<double>(s.A), static_cast<double>(s.B),                            \
                                         static_cast<double>(s.C)};                                                    \
            },                                                                                                         \
            [](T& s, const monitor::vector_t& v) {                                                                     \
                double a, b, c;                                                                                        \
                if (v.size() != 3 || !as_num(v[0], a) || !as_num(v[1], b) || !as_num(v[2], c))                          \
                    return false;                                                                                      \
                s.A = static_cast<float>(a);                                                                           \
                s.B = static_cast<float>(b);                                                                           \
                s.C = static_cast<float>(c);                                                                           \
                return true;                                                                                           \
            },                                                                                                         \
            []() {                                                                                                     \
                return monitor::vector_t{static_cast<double>(DA), static_cast<double>(DB),                              \
                                         static_cast<double>(DC)};                                                     \
            }                                                                                                          \
    }

#define SV2F(T, NAME, A, B, DA, DB, RANGE, BOUND, UNIT, DESC)                                                          \
    typed_field<T>                                                                                                     \
    {                                                                                                                  \
        NAME, value_type::vec2, access_t::read_write, RANGE, bounding_t::BOUND, compose_t::none, guard_t::none,         \
            nullptr, UNIT, nullptr, DESC, nullptr, kf_kind::continuous, 0.0, 2, false,                                  \
            [](const T& s) {                                                                                           \
                return monitor::vector_t{static_cast<double>(s.A), static_cast<double>(s.B)};                           \
            },                                                                                                         \
            [](T& s, const monitor::vector_t& v) {                                                                     \
                double a, b;                                                                                           \
                if (v.size() != 2 || !as_num(v[0], a) || !as_num(v[1], b))                                             \
                    return false;                                                                                      \
                s.A = static_cast<float>(a);                                                                           \
                s.B = static_cast<float>(b);                                                                           \
                return true;                                                                                           \
            },                                                                                                         \
            []() { return monitor::vector_t{static_cast<double>(DA), static_cast<double>(DB)}; }                        \
    }

/// Two `int` members as one tuple.
///
/// Published as DOUBLES, not int32, although the members are ints. `value_type::vec2` maps to
/// the OSC tag string "dd", and emitting an int pair under it made the tree contradict itself:
/// TYPE said two doubles and VALUE held two integers. There is no integer-vector `value_type`,
/// and inventing one to describe a single field would be worse than saying what every other
/// vector field says. `kind: discrete` and `step: 1.0` in the vendor block are how a client
/// knows the quantity is integral, the same way it knows for any other stepped control.
#define SV2I(T, NAME, A, B, DA, DB, RANGE, BOUND, UNIT, DESC)                                                          \
    typed_field<T>                                                                                                     \
    {                                                                                                                  \
        NAME, value_type::vec2, access_t::read_write, RANGE, bounding_t::BOUND, compose_t::none, guard_t::none,         \
            nullptr, UNIT, nullptr, DESC, nullptr, kf_kind::discrete, 1.0, 2, false,                                    \
            [](const T& s) {                                                                                           \
                return monitor::vector_t{static_cast<double>(s.A), static_cast<double>(s.B)};                         \
            },                                                                                                         \
            [](T& s, const monitor::vector_t& v) {                                                                     \
                double a, b;                                                                                           \
                if (v.size() != 2 || !as_num(v[0], a) || !as_num(v[1], b))                                             \
                    return false;                                                                                      \
                s.A = static_cast<int>(a + 0.5);                                                                       \
                s.B = static_cast<int>(b + 0.5);                                                                       \
                return true;                                                                                           \
            },                                                                                                         \
            []() { return monitor::vector_t{static_cast<double>(DA), static_cast<double>(DB)}; }                      \
    }

#define SI(T, NAME, MEMBER, DEF, DESC)                                                                                 \
    typed_field<T>                                                                                                     \
    {                                                                                                                  \
        NAME, value_type::integer, access_t::read_write, std::nullopt, bounding_t::free, compose_t::none,               \
            guard_t::none, nullptr, "", nullptr, DESC, nullptr, kf_kind::discrete, 1.0, 1, false,                       \
            [](const T& s) { return monitor::vector_t{static_cast<int32_t>(s.MEMBER)}; },                               \
            [](T& s, const monitor::vector_t& v) {                                                                     \
                double d;                                                                                              \
                if (v.size() != 1 || !as_num(v[0], d))                                                                 \
                    return false;                                                                                      \
                s.MEMBER = static_cast<int>(d + (d < 0 ? -0.5 : 0.5));                                                 \
                return true;                                                                                           \
            },                                                                                                         \
            []() { return monitor::vector_t{static_cast<int32_t>(DEF)}; }                                               \
    }

#define SB(T, NAME, MEMBER, DEF, DESC)                                                                                 \
    typed_field<T>                                                                                                     \
    {                                                                                                                  \
        NAME, value_type::boolean, access_t::read_write, std::nullopt, bounding_t::free, compose_t::none,               \
            guard_t::none, nullptr, "", nullptr, DESC, nullptr, kf_kind::discrete, 1.0, 1, false,                       \
            [](const T& s) { return monitor::vector_t{s.MEMBER}; },                                                    \
            [](T& s, const monitor::vector_t& v) {                                                                     \
                bool b;                                                                                                \
                if (v.size() != 1 || !as_bool(v[0], b))                                                                \
                    return false;                                                                                      \
                s.MEMBER = b;                                                                                          \
                return true;                                                                                           \
            },                                                                                                         \
            []() { return monitor::vector_t{DEF}; }                                                                     \
    }

/// A closed enumeration carried as an int member.
///
/// Closed is the point. `PREVIZ SCREEN <name> EYEMODE` tests `== "FIXED"` and treats EVERYTHING
/// else -- including a typo -- as CAMERA, so `EYEMODE FIXXED 0 1.5 3` answers 202 and silently
/// selects the other mode. A write through this row refuses an unknown name.
#define SE(T, NAME, MEMBER, DEF, NAMES, DESC)                                                                          \
    typed_field<T>                                                                                                     \
    {                                                                                                                  \
        NAME, value_type::enumeration, access_t::read_write, std::nullopt, bounding_t::refuse, compose_t::none,           \
            guard_t::none, nullptr, "", NAMES, DESC, nullptr, kf_kind::discrete, 1.0, 1, false,                         \
            [](const T& s) {                                                                                           \
                const auto names = split_list(NAMES);                                                                  \
                const auto idx   = static_cast<std::size_t>(s.MEMBER);                                                 \
                return monitor::vector_t{idx < names.size() ? std::string(names[idx]) : std::to_string(idx)};           \
            },                                                                                                         \
            [](T& s, const monitor::vector_t& v) {                                                                     \
                if (v.size() != 1)                                                                                     \
                    return false;                                                                                      \
                const auto  names = split_list(NAMES);                                                                 \
                std::string str;                                                                                       \
                if (as_str(v[0], str)) {                                                                               \
                    for (std::size_t i = 0; i < names.size(); ++i)                                                     \
                        if (names[i] == str) {                                                                         \
                            s.MEMBER = static_cast<int>(i);                                                            \
                            return true;                                                                               \
                        }                                                                                              \
                    return false;                                                                                      \
                }                                                                                                      \
                double d;                                                                                              \
                if (!as_num(v[0], d))                                                                                  \
                    return false;                                                                                      \
                const auto n = static_cast<int>(d + 0.5);                                                              \
                if (n < 0 || static_cast<std::size_t>(n) >= names.size())                                              \
                    return false;                                                                                      \
                s.MEMBER = n;                                                                                          \
                return true;                                                                                           \
            },                                                                                                         \
            []() {                                                                                                     \
                const auto names = split_list(NAMES);                                                                  \
                const auto idx   = static_cast<std::size_t>(DEF);                                                      \
                return monitor::vector_t{idx < names.size() ? std::string(names[idx]) : std::string()};                 \
            }                                                                                                          \
    }

constexpr grade_range R_SIZE{0.001, 1000.0};
constexpr grade_range R_POS{-10000.0, 10000.0};
constexpr grade_range R_ANG{-360.0, 360.0};
constexpr grade_range R_ARC{0.0, 360.0};
constexpr grade_range R_ARC_V{0.0, 180.0};
constexpr grade_range R_RES{0.0, 32768.0};
constexpr grade_range R_FOV{1.0, 170.0};
constexpr grade_range R_CLIP{0.0001, 100000.0};

using S = screen_meta;
using C = previz_camera;

/// Every row carries a DESCRIPTION, and that is a deliberate departure from the transform
/// table, where all 177 pass `nullptr`. `HOST_INFO` advertises `EXTENSIONS.DESCRIPTION` and
/// until these rows landed the flag had to report `false`, because not one field emitted one.
/// A generated control surface shows this text to an operator; a screen property whose
/// AMCP form does something surprising is exactly where that matters.
const std::vector<screen_field>& screen_table()
{
    static const std::vector<screen_field> t = {
        SV2F(S, "size", width_m, height_m, 1.0, 1.0, R_SIZE, refuse, "m",
             "Physical width and height of the screen surface."),
        SV3(S, "position", pos_x, pos_y, pos_z, 0.0, 0.0, 0.0, R_POS, refuse, "m",
            "World position of the screen's origin, which is its CENTRE-BOTTOM rather than its "
            "centre: a screen standing on the floor has y = 0."),
        SV3(S, "rotation", rot_yaw, rot_pitch, rot_roll, 0.0, 0.0, 0.0, R_ANG, wrap, "deg",
            "Yaw, pitch and roll, applied Ry*Rx*Rz. An unrotated screen faces +Z."),
        SF_RO(S, "radius", radius_m, 0.0, "m",
              "DERIVED for a curved screen and not settable: the server recomputes it as "
              "width / 2 / sin(arc / 2). The radius argument to PREVIZ SCREEN ADD ... CURVED is "
              "accepted and then discarded, so a layout file round-trips through a re-derived "
              "value. 0 means flat."),
        SF(S, "arc", arc_deg, 0.0, R_ARC, refuse, "deg",
           "Horizontal arc the screen subtends. Non-zero with a radius makes it curved."),
        SF(S, "arc_v", arc_v_deg, 0.0, R_ARC_V, refuse, "deg",
           "Vertical arc. 0 leaves a single-curved cylinder; non-zero makes it doubly curved."),
        SV2I(S, "resolution", res_w, res_h, 0, 0, R_RES, refuse, "px",
             "Pixel resolution of the panel. Drives `pixel_pitch_h` / `pixel_pitch_v` below, and "
             "since 2026-09-21 the PANEL BOUNDARIES too -- with `tile` it derives how many panels "
             "the wall is, and the renderer draws their edges. No projection calculation reads it. "
             "0 means unset."),
        SF_RO(S, "pixel_pitch_h", pitch_h_mm, 0.0, "mm",
              "DERIVED, horizontal: size / resolution, in millimetres per pixel -- the number a "
              "wall is actually specified by. Recomputed whenever either input changes, so it "
              "moves under a SIZE or a RESOLUTION write. 0 while the resolution is unset."),
        SF_RO(S, "pixel_pitch_v", pitch_v_mm, 0.0, "mm",
              "DERIVED, vertical. Published beside the horizontal one rather than assumed equal: "
              "a non-square pixel is a real (if unusual) panel, and a single figure would be a "
              "claim nothing checks."),
        SI(S, "channel", channel, -1,
           "Which channel feeds this screen. -1 is unmapped. A reference to a channel rather "
           "than a plain integer -- and note that PREVIZ MAP writes the mesh mapping WITHOUT "
           "writing this, so a screen mapped that way is textured but never auto-projected."),
        SE(S, "eye_mode", eye_mode, 0, "camera,fixed",
           "Where the eye sits for curve compensation and field of view. 'camera' follows the "
           "production virtual camera (in-camera VFX); 'fixed' sits at design_eye."),
        SV3(S, "design_eye", design_eye_x, design_eye_y, design_eye_z, 0.0, 1.5, 3.0, R_POS, refuse, "m",
            "The audience design eye position, used only when eye_mode is 'fixed'. It is also "
            "only WRITABLE then: the renderer's mutator stores these three components solely "
            "when the mode is already 'fixed', so set eye_mode first or the write is declined."),
        SV2I(S, "tile", tile_w_px, tile_h_px, 0, 0, R_RES, refuse, "px",
             "LED panel size in PIXELS. 0 clears it and stops the panel boundaries being drawn. "
             "With `resolution` it derives `tile_cols` / `tile_rows` below -- which read 0 when "
             "the two do not divide evenly, and that is worth looking at: a wall whose resolution "
             "is not a whole number of panels is a build error rather than a rounding one."),
        SF_RO(S, "tile_cols", tile_cols, 0, "",
              "DERIVED panel count across, from resolution / tile. 0 when either is unset or the "
              "division is not whole -- which is what a screen drawing no panel boundaries says "
              "for itself, rather than leaving a stored-and-inert tile size to be guessed at."),
        SF_RO(S, "tile_rows", tile_rows, 0, "",
              "DERIVED panel count down. See `tile_cols`."),
        SB(S, "icvfx", icvfx_enable, false,
           "Compute a camera-eye inner frustum and a feathered camera-frustum mask for this "
           "screen. Also reachable as MIXER PROJECTION_ICVFX, and auto-projection overwrites "
           "that one on every recompute."),
    };
    return t;
}

const std::vector<camera_field>& camera_table()
{
    static const std::vector<camera_field> t = {
        SV3(C, "position", x, y, z, 0.0, 1.5, 5.0, R_POS, refuse, "m", "World position of the camera."),
        SV3(C, "rotation", yaw, pitch, roll, 0.0, 0.0, 0.0, R_ANG, wrap, "deg",
            "Yaw, pitch and roll, applied Ry*Rx*Rz. An unrotated camera looks along -Z."),
        SF(C, "fov", fov, 60.0, R_FOV, refuse, "deg", "Vertical field of view."),
        SF_RO(C, "near_clip", near_clip, 0.1, "m",
              "Near clip plane. Read-only: hard-coded by set_camera and not carried in the "
              "layout file, so there is nothing that could persist a change."),
        SF_RO(C, "far_clip", far_clip, 100.0, "m",
              "Far clip plane. Read-only for the same reason as near_clip."),
    };
    return t;
}

/// The same shape as `fields::all()`'s guard, and it caught the same class of mistake there:
/// 17 transform rows declared `bounding: clip` with no range, which reached OSCQuery as an
/// empty `RANGE: [{}]` -- a client reads the key's presence as "this is bounded" and then
/// finds nothing to bound it with.
template <class T>
void validate(const std::vector<typed_field<T>>& t, const char* what)
{
    for (const auto& f : t) {
        // Enumerations are the same deliberate exception `fields::all()` makes: their bound is
        // the `values` list, which is a range in every sense except MIN/MAX. `wrap` likewise --
        // an angle that wraps is bounded by the period, not by a pair of limits.
        const bool bounded = f.range.has_value() || (f.type == value_type::enumeration && f.values);
        if (f.bounding != bounding_t::free && f.bounding != bounding_t::wrap && !bounded)
            CASPAR_THROW_EXCEPTION(programming_error()
                                   << msg_info(std::string(what) + " field '" + f.path +
                                               "' declares a bounding rule with no range to bound"));
        if (f.description == nullptr)
            CASPAR_THROW_EXCEPTION(programming_error()
                                   << msg_info(std::string(what) + " field '" + f.path +
                                               "' has no description; every stage row carries one"));
    }
}

} // namespace

const std::vector<screen_field>& screen_fields()
{
    static const bool ok = (validate(screen_table(), "screen"), true);
    (void)ok;
    return screen_table();
}

const std::vector<camera_field>& camera_fields()
{
    static const bool ok = (validate(camera_table(), "camera"), true);
    (void)ok;
    return camera_table();
}

namespace {

bool same_camera(const previz_camera& a, const previz_camera& b)
{
    for (const auto& f : camera_fields())
        if (f.get(a) != f.get(b))
            return false;
    return true;
}

bool same_screen(const screen_meta& a, const screen_meta& b)
{
    // THE PARENT IS PART OF THE POSE, so a re-parent is a change even when every number on the
    // screen is untouched -- which is exactly what re-parenting looks like, since the local
    // offset is deliberately left alone. Without this line the world pose moves and the tree
    // never says so.
    if (a.parent != b.parent)
        return false;

    // AND SO IS THE DERIVED WORLD POSE, for a reason the local fields cannot cover: when a
    // PARENT moves, nothing on the child changes. Its position, rotation and parent are all
    // identical, so without these two loops the publisher decides nothing happened and keeps
    // publishing the old world pose -- measured, with a venue mesh moved three times under a
    // screen hung off it and the tree reporting the same coordinates every time.
    //
    // That is this file's own rule met from a new direction: a property absent from the
    // comparison is invisible to change detection, and a DERIVED property is absent by default
    // because the thing it derives from lives somewhere else entirely.
    for (int i = 0; i < 3; ++i)
        if (a.world_pos[i] != b.world_pos[i] || a.world_rot[i] != b.world_rot[i])
            return false;

    // The coverage is derived from the CAMERA, so it moves while every field on the screen holds
    // still -- the same reason the world pose is compared above, from a different direction.
    if (a.icvfx_coverage != b.icvfx_coverage)
        return false;

    for (const auto& f : screen_fields())
        if (f.get(a) != f.get(b))
            return false;
    return true;
}

} // namespace

bool same_stage(const stage_snapshot& a, const stage_snapshot& b)
{
    const auto& fa = a.flags;
    const auto& fb = b.flags;
    if (fa.active != fb.active || fa.auto_projection != fb.auto_projection ||
        fa.show_grid != fb.show_grid || fa.show_wireframe != fb.show_wireframe ||
        fa.show_gizmo != fb.show_gizmo || fa.show_camera != fb.show_camera ||
        fa.camera_locked != fb.camera_locked || fa.snap_m != fb.snap_m ||
        fa.snap_deg != fb.snap_deg ||
        fa.has_view_override != fb.has_view_override)
        return false;

    if (a.scene_path != b.scene_path || a.selected != b.selected || a.hover != b.hover ||
        a.selected_mesh != b.selected_mesh || a.previz_bridge != b.previz_bridge ||
        a.selection != b.selection)
        return false;

    // THE MESHES, compared as a SET for the reason the screens are: they are named and created at
    // runtime, so a scene load that changes the key set is a change even when every surviving mesh
    // is untouched.
    if (a.meshes.size() != b.meshes.size())
        return false;
    for (const auto& [name, ma] : a.meshes) {
        const auto it = b.meshes.find(name);
        if (it == b.meshes.end())
            return false;
        const auto& mb = it->second;
        if (ma.parent != mb.parent || ma.visible != mb.visible || ma.touched != mb.touched)
            return false;
        for (int i = 0; i < 3; ++i)
            if (ma.pos[i] != mb.pos[i] || ma.rot[i] != mb.rot[i] ||
                ma.world_pos[i] != mb.world_pos[i] || ma.world_rot[i] != mb.world_rot[i])
                return false;
    }

    if (!same_camera(a.camera, b.camera) || !same_camera(a.view_camera, b.view_camera))
        return false;

    // Compared as a SET, not pairwise by position: screens are named and created at runtime, so
    // a rename or a removal changes the key set and must count as a change even when every
    // remaining screen is untouched.
    if (a.screens.size() != b.screens.size())
        return false;
    for (const auto& [name, s] : a.screens) {
        const auto it = b.screens.find(name);
        if (it == b.screens.end() || !same_screen(s, it->second))
            return false;
    }
    return true;
}

namespace {

/// Write one object's fields under `prefix`, omitting any that sit at their default.
///
/// "At its default" is decided against a DEFAULT-CONSTRUCTED OBJECT, not against `f.defaults()`,
/// and the difference is not pedantic. Every member here is a `float` and the accessors widen to
/// `double`, so a descriptor default of `0.1` compares unequal to the stored `0.1f` widened --
/// 0.1 against 0.10000000149011612. Measured: `near_clip` and `far_clip` published on every
/// camera and every view camera while sitting untouched at their defaults, because the test could
/// never be true for them. Reading the fresh object through the same accessor makes the
/// comparison exact for any member type, and keeps the descriptor's default a readable `0.1`
/// rather than forcing the float's decimal expansion into the tree.
template <class T>
void publish_object(monitor::state&                       st,
                    const std::string&                    prefix,
                    const T&                              obj,
                    const std::vector<typed_field<T>>&    table)
{
    static const T fresh{};
    for (const auto& f : table) {
        auto v = f.get(obj);
        if (v == f.get(fresh))
            continue;
        st[prefix + "/" + f.path] = std::move(v);
    }
}

} // namespace

void stage_publisher::refresh(const stage_snapshot& snap)
{
    if (have_ && same_stage(last_, snap))
        return;

    last_ = snap;
    have_ = true;
    ++rebuilds_;

    monitor::state st;

    // AN UNTOUCHED STAGE PUBLISHES NOTHING, and this is what makes the two backends agree.
    //
    // The OpenGL mixer holds its `previz_renderer` BY VALUE, so one exists from construction and
    // its snapshot is available on every tick of every channel. The Vulkan mixer builds one
    // lazily, on the first `PREVIZ` command. Without this test the OpenGL backend carried a
    // previz sub-tree on every idle channel and the Vulkan one did not -- measured, and caught by
    // `api-stage`'s first check on the first run of the ogl arm.
    //
    // "Untouched" is decided by comparing against a default-constructed snapshot through
    // `same_stage`, so it means exactly what the registry says it means and needs no second list
    // of what counts as interesting.
    static const stage_snapshot fresh{};
    if (same_stage(snap, fresh)) {
        state_ = std::move(st);
        return;
    }

    // Always published: no descriptor table stands behind these, so an absent key would leave a
    // reader with nothing to fall back on.
    st["active"]          = snap.flags.active;
    st["auto_projection"] = snap.flags.auto_projection;
    st["show_grid"]       = snap.flags.show_grid;
    st["show_wireframe"]  = snap.flags.show_wireframe;
    st["show_gizmo"]      = snap.flags.show_gizmo;
    st["show_camera"]     = snap.flags.show_camera;
    st["camera_locked"]   = snap.flags.camera_locked;
    st["snap_m"]          = snap.flags.snap_m;
    st["snap_deg"]        = snap.flags.snap_deg;
    st["view_override"]   = snap.flags.has_view_override;
    st["scene_path"]      = snap.scene_path;
    st["selected"]        = snap.selected;
    // The whole set. `selected` is the ACTIVE one -- what the rings and corners act on -- and is
    // always a member of this. A client drawing outlines wants this; a client drawing a gizmo
    // wants `selected`.
    st["selection"]       = snap.selection;
    st["selected_mesh"]   = snap.selected_mesh;
    st["bridge"]          = snap.previz_bridge;
    st["hover"]           = snap.hover;

    // The enumeration of named children. Always published, including when empty: a screen every
    // one of whose fields happened to sit at its default would otherwise not appear at all.
    monitor::vector_t names;
    names.reserve(snap.screens.size());
    for (const auto& [name, sm] : snap.screens)
        names.emplace_back(name);
    st["screens"] = std::move(names);

    publish_object(st, "camera", snap.camera, camera_fields());
    publish_object(st, "view_camera", snap.view_camera, camera_fields());
    // A client drawing a stage needs where a screen IS; an operator editing one needs what they
    // typed. Publishing only the first would make the tree disagree with the command, and only
    // the second leaves a client unable to place a parented screen at all. Both, then -- and the
    // world pair arrives RESOLVED on the snapshot rather than being derived here, because a
    // parent can be a venue mesh and this function cannot see one.
    for (const auto& [name, sm] : snap.screens) {
        publish_object(st, "screen/" + name, sm, screen_fields());

        // OUTSIDE THE DESCRIPTOR TABLE, deliberately, and this is the one place in this file
        // that is. The table drives the generic write path, which coerces a number into a field;
        // `parent` is a NAME whose legality depends on the rest of the document -- it must exist,
        // and it must not close a cycle -- and a generic coerce cannot ask either question. It
        // has its own command for that reason, so exposing it as a writable leaf would offer a
        // route that accepts documents the command refuses. Published read-only instead.
        st["screen/" + name + "/parent"] = sm.parent;

        // Derived, and published even for an unparented screen, where it equals the local pose.
        // An absent key would make a client test for the parent first and then choose which of
        // two shapes to read -- and the first client to forget would place every root screen at
        // the origin.
        st["screen/" + name + "/world_position"] = {sm.world_pos[0], sm.world_pos[1], sm.world_pos[2]};
        st["screen/" + name + "/world_rotation"] = {sm.world_rot[0], sm.world_rot[1], sm.world_rot[2]};
        // 1.0 means the camera's inner frustum is entirely on this wall; below that, the shot is
        // running off its edge and the rest of the frame is looking at whatever is behind it.
        st["screen/" + name + "/icvfx_coverage"] = sm.icvfx_coverage;
    }

    // THE MESH NAMES, always -- a client resolving a screen's `parent` may find it naming a venue
    // mesh, and until this list existed there was no way to learn such a thing was there. That gap
    // was self-inflicted: the hierarchy made `parent` able to name a mesh before anything published
    // one.
    {
        std::vector<std::string> mesh_names;
        mesh_names.reserve(snap.meshes.size());
        for (const auto& [name, _] : snap.meshes)
            mesh_names.push_back(name);
        st["meshes"] = std::move(mesh_names);
    }

    // THE DETAIL, only for the ones anyone has touched. See `mesh_info::touched`: a venue can
    // carry dozens of parts at six leaves apiece, against a measured ceiling of about 600 leaves
    // per channel per tick.
    for (const auto& [name, mi] : snap.meshes) {
        if (!mi.touched)
            continue;
        st["mesh/" + name + "/position"]       = {mi.pos[0], mi.pos[1], mi.pos[2]};
        st["mesh/" + name + "/rotation"]       = {mi.rot[0], mi.rot[1], mi.rot[2]};
        st["mesh/" + name + "/world_position"] = {mi.world_pos[0], mi.world_pos[1], mi.world_pos[2]};
        st["mesh/" + name + "/world_rotation"] = {mi.world_rot[0], mi.world_rot[1], mi.world_rot[2]};
        st["mesh/" + name + "/parent"]         = mi.parent;
        st["mesh/" + name + "/visible"]        = mi.visible;
    }

    state_ = std::move(st);
}

void log_stage_fields()
{
    CASPAR_LOG(info) << L"[core] stage fields: " << screen_fields().size() << L" screen, "
                     << camera_fields().size() << L" camera, all described.";
}

const screen_field* find_screen_field(std::string_view path)
{
    const auto& t = screen_fields();
    const auto  i = std::find_if(t.begin(), t.end(), [&](const screen_field& f) { return path == f.path; });
    return i == t.end() ? nullptr : &*i;
}

const camera_field* find_camera_field(std::string_view path)
{
    const auto& t = camera_fields();
    const auto  i = std::find_if(t.begin(), t.end(), [&](const camera_field& f) { return path == f.path; });
    return i == t.end() ? nullptr : &*i;
}

}}} // namespace caspar::core::fields
