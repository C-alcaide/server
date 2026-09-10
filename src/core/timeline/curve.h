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

// The keyframe INTERPOLATION ENGINE, moved into core and re-pointed at the address space.
//
// This is `modules/keyframes`' `keyframe_timeline` with three things changed and the rest kept.
// The study's L26 says keep the engine and L171 says break the commands, and this is where that
// line falls: the per-path index with its binary search, the hold-before/hold-after rules, the
// segment easing and the shortest-path angular wrap are all the same algorithm, because they were
// right. What changes:
//
//   TIME is `flicks`, not `double` seconds. A segment's fraction is computed from two integers,
//   so `local == key.time` is exact and a key placed at frame 12 of a 59.94 channel is at frame
//   12 forever. The old engine compared doubles with a 1 ms tolerance and called two keys 0.9 ms
//   apart the same key.
//
//   KEYS ARE ADDRESS-SPACE PATHS -- `opacity`, `fill_translation.0`, `volume`,
//   `producer/brightness`, `previz/screen/wall/position.0` -- not the 193 frozen KEYFRAMES names.
//   So a curve can drive anything `core::address::parse` resolves, and nothing has to be added to
//   a second table first. The degrees-to-radians conversion that the frozen table performed dies
//   with it: a path writes the registry's own units, exactly as `PUT` does.
//
//   KIND COMES FROM THE REGISTRY, through a lookup the caller supplies. `curve` must interpolate
//   a producer parameter and a screen property as readily as a mixer field, and those live in
//   registries `core/timeline` should not have to know about -- so the question "is this path
//   angular" is asked of a function, not answered by an `#include`. `kind_of` is the default
//   implementation, over `core::address`.
//
// One easing table. `common/tweener`'s 43 names are authoritative; the old keyframe map's 35 go,
// and its four names that tweener does not carry survive as ALIASES (see `easing_from_name`).

#include "time.h"

#include <core/frame/transform_fields.h>

#include <common/tweener.h>

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace caspar { namespace core { namespace timeline {

/// Resolve an easing name to a tweener. Case-insensitive, `:param` suffixes allowed.
///
/// REFUSES an unknown name (throws `user_error`) rather than falling back to linear. The old
/// engine warned once and used linear, which means a document with a typo animated differently
/// from the one the author wrote, forever, with nothing in the log after the first time.
tweener easing_from_name(const std::string& name);

/// Is `name` an easing this build knows? For validation at PUT, where the answer is a 400 rather
/// than an exception.
bool easing_exists(const std::string& name);

/// What kind of quantity a path holds. Asked of the caller so `curve` need not know the
/// registries -- see the header comment.
using kind_lookup = std::function<fields::kf_kind(const std::string& path)>;

/// The default lookup: `core::address::parse`, with `discrete` DERIVED for a boolean, an integer
/// or an enumeration rather than declared a second time. A path that does not resolve is
/// `continuous`, which is the only answer that cannot make a resolvable path behave differently.
fields::kf_kind kind_of(const std::string& path);

/// One keyframe: a time, an easing that governs the segment STARTING here, and the sparse set of
/// paths this key pins. A path absent from a key is not animated by it -- that is what makes the
/// per-path index necessary and what lets one curve carry independent per-parameter animation.
struct curve_key
{
    flicks      time = 0;
    std::string easing_name = "linear";
    tweener     ease{};                                 //< resolved once, at authoring time
    std::unordered_map<std::string, double> values;     //< path -> value, in registry units
};

/// Default tolerance for "the same key": one millisecond, in flicks. Kept from the old engine
/// because it is a human tolerance -- a client sending 1.0 and 1.0009 means one key.
constexpr flicks same_key_tolerance = flicks_per_second / 1000;

class curve
{
  public:
    /// Add a key, REPLACING any within `tol`. The replacement is whole: a client that wants to
    /// merge into an existing key calls `patch_at`.
    void add(curve_key k, flicks tol = same_key_tolerance);

    /// Remove the key nearest `at` if it is within `tol`. Returns whether one went.
    bool remove(flicks at, flicks tol = flicks_per_second / 2);

    /// Merge `patch` into the key nearest `at` within `tol`, leaving its other paths alone.
    bool patch_at(flicks at, const std::unordered_map<std::string, double>& patch,
                  flicks tol = same_key_tolerance);

    void        clear();
    bool        empty() const { return keys_.empty(); }
    std::size_t size() const { return keys_.size(); }

    const std::vector<curve_key>& keys() const { return keys_; }

    /// Every path any key pins, sorted, deduplicated.
    const std::vector<std::string>& paths() const { return paths_; }

    /// The last time any key sits at. 0 for an empty curve.
    flicks duration() const { return keys_.empty() ? 0 : keys_.back().time; }

    /// Every animated path's value at `local`.
    ///
    /// Per path, independently: the last key at-or-before `local` that pins it and the first key
    /// after it that does. Both -> interpolate with the BEFORE key's easing. Only before -> hold.
    /// Only after -> hold at it (pre-roll). Neither -> the path is absent from the result, which
    /// is how a caller distinguishes "not animated" from "animated to its default".
    std::unordered_map<std::string, double> interpolate(flicks local, const kind_lookup& kind) const;

  private:
    std::vector<curve_key>   keys_;   //< sorted by time
    std::vector<std::string> paths_;  //< sorted, unique

    /// Per path, the indices into `keys_` of the keys that pin it -- in the same (time) order, so
    /// `interpolate` binary-searches per path instead of scanning every key for every path.
    /// Rebuilt on every mutation, which is the right trade: mutation is authoring, interpolation
    /// is every frame.
    std::unordered_map<std::string, std::vector<std::size_t>> index_;

    void reindex();
};

/// Aborts on a disagreement between the engine and its stated rules. Called at boot.
void curve_self_test();

}}} // namespace caspar::core::timeline
