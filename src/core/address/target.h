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

// ONE grammar for "which parameter", and one place the registries are consulted.
//
// Three subsystems name an animatable parameter and each of them learned to do it separately.
// `BIND` takes `brightness`, `midtone.1` or `producer/<name>` and validates them with an
// if/else inside `stage::impl::add_binding`. KEYFRAMES took its own 193-name frozen table with
// degrees baked in. The HTTP write path splits `/channel/1/stage/layer/10/mixer/opacity` into
// segments and consults `fields::find`, then `fields::find_audio_field`, itself. Three parsers
// for one question, and the timeline about to become a fourth -- which is what this file is for.
//
// The grammar is the BINDING one, extended, because it is the one an authored document wants: a
// short path with an optional `.N` component suffix, not a channel-qualified URL. The channel
// and layer come from the object that owns the target, so they are not in the string.
//
//   opacity                        an image_transform field, whole
//   fill_translation.0             one component of one
//   volume                         the audio half of the same transform
//   producer/brightness            a parameter of whatever is playing on the layer
//   previz/camera/position.1       the production camera; view_camera for the viewport one
//   previz/screen/wall/pos_x       a named screen on the channel's previz stage
//
// A leading `mixer/` is accepted and ignored, so an address copied out of the published state
// tree or out of an HTTP path resolves without editing.
//
// WHAT THIS FILE CAN AND CANNOT DECIDE. Two of the five registries are static tables and are
// resolved here to a `field_meta*`: the image transform and its audio half. The other three
// need live state this header must not depend on -- a producer parameter exists only while that
// producer is on that layer, and a screen exists only if the channel's previz renderer has one.
// So `parse` reports the KIND and the key, and the caller (the stage, which holds both) does the
// live half. Reporting a screen path as valid here and having the write fail later would put two
// different answers to one question in two places, which is the mistake this file exists to undo.

#include <core/frame/transform_fields.h>

#include <cstdint>
#include <string>
#include <string_view>

namespace caspar { namespace core { namespace address {

/// WHICH registry describes this target. `none` is a parse failure, not a sixth registry.
enum class target_kind
{
    none,
    image,           ///< `fields::find` -- image_transform
    audio,           ///< `fields::find_audio_field` -- audio_transform
    producer_param,  ///< `frame_producer::parameters()` on the layer's foreground
    screen,          ///< `fields::find_screen_field` on the named screen
    camera,          ///< `fields::find_camera_field`, production camera
    view_camera      ///< the same table, viewport camera
};

const char* kind_name(target_kind k);

struct target
{
    target_kind kind = target_kind::none;

    /// The path as given, `mixer/` stripped and the `.N` suffix still attached. This is what a
    /// document stored, what an error message should quote and what a publication keys on.
    std::string path;

    /// The registry key alone: `opacity`, `volume`, `brightness`, `pos_x`, `position`.
    std::string field;

    /// The screen name, for `kind::screen`. Empty for every other kind.
    std::string object;

    /// Which component of an arity>1 field, 0 for the whole thing. Only 0..3 parse.
    std::uint8_t component = 0;

    /// The descriptor, for the two static registries. Null for the three live ones -- a
    /// producer parameter is described by the producer, a stage field by the caller's lookup.
    const fields::field_meta* meta = nullptr;

    explicit operator bool() const { return kind != target_kind::none; }
};

/// Classify `path` and resolve it against every registry that is a static table.
///
/// Never throws and never touches the stage. An unparseable or unknown path comes back with
/// `kind == none`; a live-registry path comes back classified with `meta == nullptr`.
target parse(std::string_view path);

/// Split a target into its field and component, the way `binding::split_target` does.
///
/// Kept as its own entry point because `remove_bindings` and `is_bound` want only this half and
/// have no layer to resolve a producer parameter against.
void split(std::string_view path, std::string& field, std::uint8_t& component);

/// Aborts on a disagreement between this grammar and the registries. Called at boot.
void target_self_test();

}}} // namespace caspar::core::address
