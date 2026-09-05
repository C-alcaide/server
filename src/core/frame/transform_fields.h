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

// The transform field registry: every parameter of `image_transform` declared ONCE, with
// everything a reader or a writer needs to know about it -- its type and arity, its legal
// range, what happens outside that range, how two layers' values COMPOSE, which subsystem
// it switches on, and the KEYFRAMES names it was already known by.
//
// Why this exists. `image_transform` has been described in four separate hand-written
// lists that nobody could keep aligned: the composition allowlist in BOTH mixers'
// `apply_transform_colour_values` (`accelerator/{ogl,vulkan}/util/transforms.cpp`), the
// KEYFRAMES table (`modules/keyframes/keyframe_fields.cpp`), `operator==` in
// `frame_transform.cpp`, and the argument parsing of every MIXER command. A field added to
// the struct but missed in the composition list is accepted, reported back correctly on
// query, and never reaches the kernel -- `MIXER EXPOSURE` did exactly that for a whole
// session. This table is the one declaration the other four are derived from or checked
// against.
//
// What it does NOT do yet: the two mixers still carry their hand-written composition.
// `compose_colour()` below is the GENERATED equivalent, and `compose_self_test()` proves
// the two agree on randomised inputs. Replacing the hand-written bodies with a call to
// `compose_colour` is deliberately a later change, because it touches every drawn layer
// on every frame and deserves its own measured commit.

#include "frame_transform.h"

#include <core/monitor/monitor.h>

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace caspar { namespace core { namespace fields {

/// The wire type of a field. There is deliberately no new type system: scalars are the
/// `monitor::data_t` alternatives, vectors are short arrays of them, and `blob` is an
/// opaque owned object (a LUT, a curve set) that can be reported as present but not set
/// by value.
enum class value_type : uint8_t
{
    boolean,
    integer,
    real,
    string,
    enumeration,
    vec2,
    vec3,
    vec4,
    blob,
};

/// OSCQuery's ACCESS mask, and ossia's three-state access: a trigger is write-only, and
/// a boolean `writable` cannot say so.
enum class access_t : uint8_t
{
    read       = 1,
    write      = 2,
    read_write = 3,
};

/// What happens to a value outside [min, max]. `free` has no bounds; `clip` saturates;
/// `wrap` is periodic (a hue rotation of 200 is -160); `fold` reflects. The fork's
/// parameters are clip or wrap; fold is here because ossia has it and a generated control
/// must be able to say which one it is.
enum class bounding_t : uint8_t
{
    free,
    clip,
    wrap,
    fold,
};

/// How a layer's value combines with the accumulated one during composition. This is the
/// column that closes the allowlist trap: a field with a compose rule is composed, and a
/// field without one is `none` by declaration rather than by omission.
///
///   none            geometry, which follows a separate flow; not composed here
///   multiply        opacity, brightness, gain, exposure -- the product of two gains
///   add             lift, temperature, hue rotation -- offsets sum
///   min_ / max_     level ranges intersect: input minima take the max, maxima the min
///   or_ / xor_      flags: enables OR, flips XOR (two flips cancel)
///   innermost_wins  the inner layer's value replaces the outer's when a guard holds --
///                   a LUT, a projection, a qualifier; two of them cannot be composed
///                   without resampling one onto the other, so the layer's own wins
///   custom          the two rules that fit no column: split-tone balance, edge blending
enum class compose_t : uint8_t
{
    none,
    multiply,
    add,
    min_,
    max_,
    or_,
    xor_,
    innermost_wins,
    custom,
};

/// How the KEYFRAMES module interpolates the field. `angular_rad` is stored in radians
/// and shown in degrees; `angular` is stored in degrees already.
enum class kf_kind : uint8_t
{
    continuous,
    angular,
    angular_rad,
    discrete,
};

/// The predicate on `other` that a guarded composition tests. Named rather than a
/// function pointer so the descriptor can report it ("projection.enable") and so the
/// table stays one line per field.
enum class guard_t : uint8_t
{
    none,
    projection_enable,
    curve_enable,
    edge_blend_any, // any of the four edges > 0 -- the group gate
    icvfx_enable,
    color_grade_enable,
    ocio_enable,
    blur_enable,
    shape_enable,
    curves_enable,
    rgb_levels_enable,
    qualifier_enable,
    gamut_compress,
    lut3d_present,
    hue_curves_present,
    blend_mask_present,
    grade_nodes_present,
    split_active,       // any split-tone colour component non-zero
    sharpen_radius_set, // != 1.0
    grain_size_set,     // != 1.0
};

struct field_desc
{
    /// The state key under `channel/N/stage/layer/M/mixer/`, and the API path segment.
    const char* path;
    value_type  type;
    access_t    access;
    /// The legal range, when there is one -- names the same `grade_limits` constant the
    /// MIXER command validates against, so the two cannot drift. nullopt is `free`.
    std::optional<grade_range> range;
    bounding_t                 bounding;
    compose_t                  compose;
    guard_t                    guard;
    /// A subsystem enable that a write to this field switches on. Carries the auto-enable
    /// rule from `apply_kf_to_transform`, so PUT and KEYFRAMES agree. nullptr = none.
    const char* enables;
    const char* unit;
    /// For `enumeration`: the value names in enum order, comma-separated.
    const char* values;
    const char* description;
    /// The KEYFRAMES names, one per component, comma-separated. nullptr = not animatable.
    /// Explicit because they are not derivable from members: `fill_x` is
    /// `fill_translation[0]`, `mid_r` is `midtone[0]`.
    const char* kf_names;
    kf_kind     kind;
    double      step;
    uint8_t     arity;

    /// Does COMPOSITION clamp the combined value back into `range`?
    ///
    /// Separate from `bounding`, which is about a WRITE and is what a control surface
    /// applies to its own slider. Composition is a different question, and the two mixers
    /// answer it per field: they clamp the grading block -- "two layers at the edge of
    /// legal would otherwise reach a value no single command could set" -- and do NOT clamp
    /// `levels.gamma`, the three per-channel gammas, `sharpen_amount` or `grain_intensity`.
    ///
    /// This column exists because `compose_self_test` FOUND that difference, not because
    /// anyone chose it: all six were declared clamped, all six diverged on every one of 256
    /// iterations, and both mixers agreed with each other against the table. Whether those
    /// six SHOULD be clamped is a real question and a rendered-output change; this table's
    /// job is to describe what the mixers do, so the exception is declared here and the
    /// question is recorded in the feature document.
    bool compose_clamps;

    // Type-erased accessors. The value carrier IS `monitor::vector_t`, so a read is
    // directly publishable and a write is directly what arrived off the wire.
    monitor::vector_t (*get)(const image_transform&);
    /// Returns false if the value has the wrong type or arity. Does not range-check --
    /// that is the caller's job, using `range` and `bounding`.
    bool (*set)(image_transform&, const monitor::vector_t&);
    monitor::vector_t (*defaults)();
};

/// Every field, in table order.
const std::vector<field_desc>& all();

/// Look a field up by its path segment. nullptr if unknown.
const field_desc* find(std::string_view path);

/// The GENERATED composition: every field's declared rule applied in table order, plus
/// the group rules the table cannot express. Semantically the same as both mixers'
/// hand-written `apply_transform_colour_values`, which `compose_self_test` checks rather
/// than assumes.
void compose_colour(image_transform& self, const image_transform& other);

/// Compose one field, by its declared rule. Exposed so the self-test can attribute a
/// divergence to a single row rather than to the whole struct.
void compose_field(const field_desc& f, image_transform& self, const image_transform& other);

/// Switch on whatever subsystem `field` belongs to, the way `apply_kf_to_transform`
/// does: a blur radius write enables blur (if the radius is non-zero), a geometry write
/// enables the geometry modifiers, an RGB-levels write enables per-channel levels.
void apply_enables(image_transform& tf, const field_desc& field);

/// The name→index of a KEYFRAMES name: which field, which component. Lets the keyframes
/// module build its flat per-component table from this one.
struct kf_ref
{
    const field_desc* field;
    uint8_t           component;
};
std::optional<kf_ref> find_kf(std::string_view kf_name);

/// Every KEYFRAMES name this table generates, in table order. The keyframes module
/// compares it against its frozen list so a rename or a dropped entry fails at startup
/// rather than silently changing what a saved timeline animates.
std::vector<std::string> all_kf_names();

/// Split a comma-separated `values` / `kf_names` list. Empty for nullptr.
std::vector<std::string_view> split_list(const char* csv);

/// Randomised agreement test between `compose_colour` and a hand-written reference (a
/// mixer's `apply_transform_colour_values`). Every field is driven through in-range random
/// values, guards and enables are toggled at random, and the two results are compared with
/// `operator==`. On a mismatch the diverging fields are named, so the report says WHICH
/// rule disagrees rather than only that one does.
struct self_test_report
{
    unsigned                 iterations  = 0;
    unsigned                 fields      = 0;
    unsigned                 divergences = 0;
    std::vector<std::string> diverged; // field paths, unique, in table order
};
self_test_report compose_self_test(void (*reference)(image_transform&, const image_transform&),
                                   unsigned iterations = 256,
                                   uint32_t seed       = 0x5EED5EEDu);

}}} // namespace caspar::core::fields
