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
        NAME, value_type::enumeration, access_t::read_write, std::nullopt, bounding_t::clip, compose_t::none,           \
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
        SV2F(S, "size", width_m, height_m, 1.0, 1.0, R_SIZE, clip, "m",
             "Physical width and height of the screen surface."),
        SV3(S, "position", pos_x, pos_y, pos_z, 0.0, 0.0, 0.0, R_POS, clip, "m",
            "World position of the screen's origin, which is its CENTRE-BOTTOM rather than its "
            "centre: a screen standing on the floor has y = 0."),
        SV3(S, "rotation", rot_yaw, rot_pitch, rot_roll, 0.0, 0.0, 0.0, R_ANG, wrap, "deg",
            "Yaw, pitch and roll, applied Ry*Rx*Rz. An unrotated screen faces +Z."),
        SF_RO(S, "radius", radius_m, 0.0, "m",
              "DERIVED for a curved screen and not settable: the server recomputes it as "
              "width / 2 / sin(arc / 2). The radius argument to PREVIZ SCREEN ADD ... CURVED is "
              "accepted and then discarded, so a layout file round-trips through a re-derived "
              "value. 0 means flat."),
        SF(S, "arc", arc_deg, 0.0, R_ARC, clip, "deg",
           "Horizontal arc the screen subtends. Non-zero with a radius makes it curved."),
        SF(S, "arc_v", arc_v_deg, 0.0, R_ARC_V, clip, "deg",
           "Vertical arc. 0 leaves a single-curved cylinder; non-zero makes it doubly curved."),
        SV2I(S, "resolution", res_w, res_h, 0, 0, R_RES, clip, "px",
             "Pixel resolution of the panel. INERT: stored, persisted to the layout file, and "
             "read by no renderer path and no projection calculation. 0 means unset."),
        SI(S, "channel", channel, -1,
           "Which channel feeds this screen. -1 is unmapped. A reference to a channel rather "
           "than a plain integer -- and note that PREVIZ MAP writes the mesh mapping WITHOUT "
           "writing this, so a screen mapped that way is textured but never auto-projected."),
        SE(S, "eye_mode", eye_mode, 0, "camera,fixed",
           "Where the eye sits for curve compensation and field of view. 'camera' follows the "
           "production virtual camera (in-camera VFX); 'fixed' sits at design_eye."),
        SV3(S, "design_eye", design_eye_x, design_eye_y, design_eye_z, 0.0, 1.5, 3.0, R_POS, clip, "m",
            "The audience design eye position, used only when eye_mode is 'fixed'. It is also "
            "only WRITABLE then: the renderer's mutator stores these three components solely "
            "when the mode is already 'fixed', so set eye_mode first or the write is declined."),
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
        SV3(C, "position", x, y, z, 0.0, 1.5, 5.0, R_POS, clip, "m", "World position of the camera."),
        SV3(C, "rotation", yaw, pitch, roll, 0.0, 0.0, 0.0, R_ANG, wrap, "deg",
            "Yaw, pitch and roll, applied Ry*Rx*Rz. An unrotated camera looks along -Z."),
        SF(C, "fov", fov, 60.0, R_FOV, clip, "deg", "Vertical field of view."),
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
        fa.show_gizmo != fb.show_gizmo || fa.camera_locked != fb.camera_locked ||
        fa.has_view_override != fb.has_view_override)
        return false;

    if (a.scene_path != b.scene_path || a.selected != b.selected || a.hover != b.hover)
        return false;

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
    st["camera_locked"]   = snap.flags.camera_locked;
    st["view_override"]   = snap.flags.has_view_override;
    st["scene_path"]      = snap.scene_path;
    st["selected"]        = snap.selected;
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
    for (const auto& [name, sm] : snap.screens)
        publish_object(st, "screen/" + name, sm, screen_fields());

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
