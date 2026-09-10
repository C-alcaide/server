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

#include "curve.h"

#include <core/address/target.h>

#include <common/except.h>
#include <common/log.h>
#include <common/utf.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <set>

namespace caspar { namespace core { namespace timeline {

namespace {

/// The names the old KEYFRAMES map carried that `common/tweener` does not.
///
/// MEASURED rather than assumed: of the 35 names in `modules/keyframes`' easing map, 31 are
/// tweener names already and four are not -- three shorthands and one typo that has been in
/// CasparCG's vocabulary long enough that a saved document may hold it. They are aliases here so
/// no document that worked stops working, and so there is still only ONE table.
const std::unordered_map<std::string, std::string>& easing_aliases()
{
    static const std::unordered_map<std::string, std::string> m = {
        {"ease", "easeinoutcubic"},      // the keyframe map's shorthand
        {"easein", "easeincubic"},       // ...
        {"easeout", "easeoutcubic"},     // ...
        {"easeinelestic", "easeinelastic"}, // a long-standing CasparCG typo, kept working
    };
    return m;
}

std::string lower(const std::string& s)
{
    std::string out;
    out.reserve(s.size());
    for (char c : s)
        out += static_cast<char>(::tolower(static_cast<unsigned char>(c)));
    return out;
}

std::string canonical_easing(const std::string& name)
{
    const auto  key = lower(name);
    const auto& al  = easing_aliases();
    const auto  it  = al.find(key);
    return it == al.end() ? key : it->second;
}

} // namespace

tweener easing_from_name(const std::string& name)
{
    // `tweener` throws `user_error` on a name it does not know, which is the behaviour wanted: an
    // unknown easing is a document defect the author must see, not something to paper over with
    // linear. Its constructor also lowercases and accepts `:param` suffixes, so
    // `easeoutelastic:1.5` reaches the right function with the right parameter.
    return tweener(u16(canonical_easing(name)));
}

bool easing_exists(const std::string& name)
{
    const auto canon = canonical_easing(name);
    // The parameter suffix, if any, is not part of the name.
    const auto colon = canon.find(':');
    const auto bare  = colon == std::string::npos ? canon : canon.substr(0, colon);
    if (bare == "linear")
        return true;
    for (const auto& n : tweener::names())
        if (u8(n) == bare)
            return true;
    return false;
}

fields::kf_kind kind_of(const std::string& path)
{
    const auto t = address::parse(path);
    if (!t || !t.meta) {
        // Either it does not resolve, or it is one of the three live registries whose descriptor
        // this function cannot reach. `continuous` is the safe answer: it is what an unlabelled
        // number is, and it is the only choice that cannot make a resolvable path behave
        // differently from how its own table says it should.
        return fields::kf_kind::continuous;
    }

    // DERIVED, not declared twice. A boolean, an integer or an enumeration cannot be half-way
    // between two values, so it holds across a segment whatever its `kind` column says -- and
    // the column exists to distinguish angular from plain, which is a question only reals have.
    switch (t.meta->type) {
        case fields::value_type::boolean:
        case fields::value_type::integer:
        case fields::value_type::enumeration:
            return fields::kf_kind::discrete;
        default:
            break;
    }
    return t.meta->kind;
}

void curve::add(curve_key k, flicks tol)
{
    keys_.erase(std::remove_if(keys_.begin(), keys_.end(),
                               [&](const curve_key& e) { return std::llabs(e.time - k.time) <= tol; }),
                keys_.end());
    keys_.push_back(std::move(k));
    std::stable_sort(keys_.begin(), keys_.end(),
                     [](const curve_key& a, const curve_key& b) { return a.time < b.time; });
    reindex();
}

bool curve::remove(flicks at, flicks tol)
{
    if (keys_.empty())
        return false;
    auto it = std::min_element(keys_.begin(), keys_.end(), [&](const curve_key& a, const curve_key& b) {
        return std::llabs(a.time - at) < std::llabs(b.time - at);
    });
    if (std::llabs(it->time - at) > tol)
        return false;
    keys_.erase(it);
    reindex();
    return true;
}

bool curve::patch_at(flicks at, const std::unordered_map<std::string, double>& patch, flicks tol)
{
    for (auto& k : keys_) {
        if (std::llabs(k.time - at) <= tol) {
            for (const auto& kv : patch)
                k.values[kv.first] = kv.second;
            reindex();
            return true;
        }
    }
    return false;
}

void curve::clear()
{
    keys_.clear();
    paths_.clear();
    index_.clear();
}

std::unordered_map<std::string, double> curve::interpolate(
    flicks local, const kind_lookup& kind,
    const std::unordered_map<std::string, double>* rebase_from) const
{
    std::unordered_map<std::string, double> out;
    if (keys_.empty())
        return out;

    out.reserve(paths_.size());

    for (const auto& path : paths_) {
        const auto idx_it = index_.find(path);
        if (idx_it == index_.end())
            continue;
        const auto& idx = idx_it->second;

        // The first key pinning this path whose time is strictly AFTER `local`. Everything else
        // follows from that one search: its predecessor, if any, is the `before` key.
        const auto after_it =
            std::upper_bound(idx.begin(), idx.end(), local,
                             [this](flicks t, std::size_t i) { return t < keys_[i].time; });

        const curve_key* before = after_it != idx.begin() ? &keys_[*std::prev(after_it)] : nullptr;
        const curve_key* after  = after_it != idx.end() ? &keys_[*after_it] : nullptr;

        if (before && after) {
            double       a = before->values.at(path);
            const double b = after->values.at(path);

            // REBASE, and only on the FIRST segment. `after_it == idx.begin() + 1` means the
            // `before` key is this path's first, which is the only place a captured entry value
            // belongs -- applied everywhere it would leave a rebased object permanently offset
            // from what its author wrote.
            if (rebase_from && after_it == std::next(idx.begin())) {
                const auto r = rebase_from->find(path);
                if (r != rebase_from->end())
                    a = r->second;
            }
            const auto   k = kind ? kind(path) : fields::kf_kind::continuous;

            if (k == fields::kf_kind::discrete) {
                // A step: the segment holds the value it STARTED at, and changes at the next
                // key. Interpolating a blend mode between `screen` and `add` would give a number
                // that names neither.
                out[path] = a;
                continue;
            }

            const flicks span = after->time - before->time;
            // Integer arithmetic to the last possible moment. `span` is exact and non-zero (two
            // keys closer than the tolerance were merged at `add`), so the fraction is the only
            // floating-point step and it is computed once.
            const double t_raw = span > 0 ? static_cast<double>(local - before->time) /
                                                static_cast<double>(span)
                                          : 0.0;
            const double t = std::max(0.0, std::min(1.0, t_raw));

            // `tweener(t, b, c, d)` wants the destination DELTA, so the wrap is applied to the
            // delta and the easing shapes the journey rather than the endpoints.
            double delta = b - a;
            if (k == fields::kf_kind::angular || k == fields::kf_kind::angular_rad) {
                // Shortest path round the circle. The MODULUS is the registry's own unit: an
                // `angular` row holds degrees and an `angular_rad` row holds radians, and using
                // 360 for both is how a rotation in radians would take the long way round
                // 57 times. The old engine only had degrees, because the frozen table converted
                // on the way in -- with paths writing registry units, this has to be per-kind.
                const double full = k == fields::kf_kind::angular ? 360.0 : 2.0 * 3.14159265358979323846;
                const double half = full * 0.5;
                delta             = std::fmod(delta, full);
                if (delta > half)
                    delta -= full;
                else if (delta < -half)
                    delta += full;
            }

            out[path] = before->ease(t, a, delta, 1.0);
        } else if (before) {
            out[path] = before->values.at(path); //< after the last key: hold
        } else if (after) {
            out[path] = after->values.at(path); //< before the first key: pre-roll
        }
    }

    return out;
}

void curve::reindex()
{
    std::set<std::string> names;
    for (const auto& k : keys_)
        for (const auto& kv : k.values)
            names.insert(kv.first);
    paths_.assign(names.begin(), names.end());

    index_.clear();
    index_.reserve(paths_.size());
    for (std::size_t i = 0; i < keys_.size(); ++i)
        for (const auto& kv : keys_[i].values)
            index_[kv.first].push_back(i);
}

// =======================================================================================
// Self-test
// =======================================================================================

namespace {

curve_key key(flicks t, const char* easing, std::initializer_list<std::pair<const char*, double>> vals)
{
    curve_key k;
    k.time        = t;
    k.easing_name = easing;
    k.ease        = easing_from_name(easing);
    for (const auto& v : vals)
        k.values[v.first] = v.second;
    return k;
}

} // namespace

void curve_self_test()
{
    const auto req = [](bool ok, const char* what) {
        if (!ok) {
            CASPAR_LOG(fatal) << L"timeline::curve_self_test: " << what;
            std::abort();
        }
    };
    // NOT called `near`: `near` is a macro in windef.h and this failed to compile as one.
    const auto close_to = [](double a, double b, double tol = 1e-9) { return std::abs(a - b) <= tol; };

    // ---- the easing table, and the four aliases that keep old documents working ----------
    req(easing_exists("linear"), "linear exists");
    req(easing_exists("easeOutSine"), "names are case-insensitive");
    req(easing_exists("ease"), "`ease` is an alias");
    req(easing_exists("easein") && easing_exists("easeout"), "so are `easein` and `easeout`");
    req(easing_exists("EASEINELESTIC"), "the legacy typo still resolves");
    req(!easing_exists("easeinelestik"), "and a real typo does not");
    req(tweener::names().size() == 43, "43 easing names -- MEASURED at boot, not predicted");

    // ---- per-path independence, and the hold rules --------------------------------------
    {
        curve c;
        c.add(key(0, "linear", {{"opacity", 0.0}}));
        c.add(key(from_seconds(1.0), "linear", {{"opacity", 1.0}, {"brightness", 0.5}}));
        c.add(key(from_seconds(2.0), "linear", {{"brightness", 1.5}}));

        req(c.size() == 3, "three keys");
        req(c.paths().size() == 2, "two paths");
        req(c.duration() == from_seconds(2.0), "duration is the last key");

        // Half-way along opacity's only segment.
        auto v = c.interpolate(from_seconds(0.5), kind_of);
        req(close_to(v.at("opacity"), 0.5), "opacity halfway is 0.5");
        // brightness has no key at-or-before 0.5 s, so it PRE-ROLLS at its first value rather
        // than appearing at a default nobody wrote.
        req(close_to(v.at("brightness"), 0.5), "brightness pre-rolls to its first key");

        // Past opacity's last key it HOLDS, while brightness is still moving. This is the whole
        // point of the per-path index: two paths on one curve with different extents.
        v = c.interpolate(from_seconds(1.5), kind_of);
        req(close_to(v.at("opacity"), 1.0), "opacity holds after its last key");
        req(close_to(v.at("brightness"), 1.0), "brightness is halfway through its own segment");

        // A path nobody pinned is ABSENT, not zero.
        req(v.find("hue_shift") == v.end(), "an unanimated path is absent from the result");

        // Exactly ON a key, and exactly at the ends.
        v = c.interpolate(from_seconds(1.0), kind_of);
        req(close_to(v.at("opacity"), 1.0), "on the key, exactly its value");
        req(close_to(c.interpolate(0, kind_of).at("opacity"), 0.0), "at zero, the first value");
    }

    // ---- ANGULAR wrap, per registry unit -------------------------------------------------
    //
    // The two kinds differ only in the modulus, and getting that wrong is silent: a radian
    // rotation wrapped at 360 never wraps, so it takes the long way round and looks like a
    // deliberate spin.
    {
        // `rotation` on a screen is degrees (kf_kind::angular). 350 -> 10 is +20, not -340.
        const auto deg = [](const std::string&) { return fields::kf_kind::angular; };
        curve      c;
        c.add(key(0, "linear", {{"a", 350.0}}));
        c.add(key(from_seconds(1.0), "linear", {{"a", 10.0}}));
        req(close_to(c.interpolate(from_seconds(0.5), deg).at("a"), 360.0),
            "350 -> 10 degrees passes through 360, not 180");
    }
    {
        // A transform angle is RADIANS (kf_kind::angular_rad). 6.0 -> 0.2 must wrap the short
        // way, a delta of about +0.483, so the midpoint is about 6.24 -- NOT 3.1.
        const auto rad = [](const std::string&) { return fields::kf_kind::angular_rad; };
        const double two_pi = 2.0 * 3.14159265358979323846;
        curve        c;
        c.add(key(0, "linear", {{"a", 6.0}}));
        c.add(key(from_seconds(1.0), "linear", {{"a", 0.2}}));
        const auto mid = c.interpolate(from_seconds(0.5), rad).at("a");
        req(close_to(mid, 6.0 + (0.2 + two_pi - 6.0) * 0.5, 1e-9), "6.0 -> 0.2 radians wraps forwards");
        req(mid > 6.0, "which means it goes UP through 2pi, not down through 3.1");
    }
    {
        // And the registry's own answer for a real angular_rad row, so the two halves of this
        // are connected rather than both asserted against a lambda. `angle` is the layer
        // rotation and the table declares it in radians.
        req(kind_of("angle") == fields::kf_kind::angular_rad,
            "`angle` is angular_rad in the registry");
    }

    // ---- DISCRETE hold, derived from the value type --------------------------------------
    {
        // `blend_mode` is an enumeration. It must STEP at the next key, not slide through
        // ordinals that name a third mode.
        req(kind_of("blend_mode") == fields::kf_kind::discrete,
            "an enumeration is discrete without the table saying so");
        req(kind_of("is_key") == fields::kf_kind::discrete, "so is a boolean");
        req(kind_of("opacity") == fields::kf_kind::continuous, "and a plain real is not");

        curve c;
        c.add(key(0, "linear", {{"blend_mode", 0.0}}));
        c.add(key(from_seconds(1.0), "linear", {{"blend_mode", 5.0}}));
        req(close_to(c.interpolate(from_seconds(0.99), kind_of).at("blend_mode"), 0.0),
            "a discrete path holds its segment's start value");
        req(close_to(c.interpolate(from_seconds(1.0), kind_of).at("blend_mode"), 5.0),
            "and changes exactly at the next key");
    }

    // ---- easing shapes the segment, and it is the BEFORE key's easing --------------------
    {
        curve c;
        c.add(key(0, "easeinquad", {{"x", 0.0}}));
        c.add(key(from_seconds(1.0), "linear", {{"x", 1.0}}));
        c.add(key(from_seconds(2.0), "linear", {{"x", 2.0}}));
        // easeinquad at t = 0.5 is 0.25, so the first segment is BELOW its linear midpoint...
        req(close_to(c.interpolate(from_seconds(0.5), kind_of).at("x"), 0.25, 1e-9),
            "the first segment eases in");
        // ...and the second, whose before-key is linear, is exactly at its midpoint.
        req(close_to(c.interpolate(from_seconds(1.5), kind_of).at("x"), 1.5, 1e-9),
            "the second segment does not inherit the first key's easing");
    }

    // ---- REBASE: the first segment starts from a captured value -------------------------
    {
        curve c;
        c.add(key(0, "linear", {{"x", 0.2}}));
        c.add(key(from_seconds(1.0), "linear", {{"x", 0.8}}));
        c.add(key(from_seconds(2.0), "linear", {{"x", 0.3}}));

        const std::unordered_map<std::string, double> from = {{"x", 0.5}};

        req(close_to(c.interpolate(0, kind_of, &from).at("x"), 0.5),
            "at entry a rebased path starts from the CAPTURED value, not the authored 0.2");
        req(close_to(c.interpolate(from_seconds(0.5), kind_of, &from).at("x"), 0.65),
            "and ramps from it to the second key -- halfway between 0.5 and 0.8");
        req(close_to(c.interpolate(from_seconds(1.0), kind_of, &from).at("x"), 0.8),
            "arriving exactly at the second key, which is where the author's curve resumes");
        req(close_to(c.interpolate(from_seconds(1.5), kind_of, &from).at("x"), 0.55),
            "and the SECOND segment is the authored one -- 0.8 to 0.3, halfway is 0.55. A rebase "
            "applied to every segment would leave the object permanently offset from what its "
            "author wrote, which is a different feature");
        req(close_to(c.interpolate(from_seconds(0.5), kind_of).at("x"), 0.5),
            "and with no capture the authored curve is unchanged: 0.2 to 0.8, halfway is 0.5");

        // A path the capture does not name is untouched, so one rebased object may carry a
        // mixture -- which is what happens when a parameter had no live value to capture.
        curve d;
        d.add(key(0, "linear", {{"x", 0.2}, {"y", 0.1}}));
        d.add(key(from_seconds(1.0), "linear", {{"x", 0.8}, {"y", 0.9}}));
        const auto mixed = d.interpolate(from_seconds(0.5), kind_of, &from);
        req(close_to(mixed.at("x"), 0.65), "the named path is rebased");
        req(close_to(mixed.at("y"), 0.5), "and an unnamed one is not");
    }

    // ---- mutation: add replaces within tolerance, patch merges, remove is nearest --------
    {
        curve c;
        c.add(key(0, "linear", {{"x", 1.0}}));
        c.add(key(same_key_tolerance / 2, "linear", {{"x", 2.0}}));
        req(c.size() == 1, "a key inside the tolerance REPLACES rather than crowding");
        req(close_to(c.interpolate(0, kind_of).at("x"), 2.0), "and the later write wins");

        c.add(key(from_seconds(1.0), "linear", {{"x", 3.0}}));
        req(c.patch_at(from_seconds(1.0), {{"y", 9.0}}), "patch finds the key");
        req(c.keys().back().values.size() == 2, "and merges rather than replacing");
        req(close_to(c.keys().back().values.at("x"), 3.0), "leaving the other path alone");

        req(!c.remove(from_seconds(5.0), flicks_per_second / 2), "remove refuses a distant time");
        req(c.remove(from_seconds(1.0)), "and takes the near one");
        req(c.size() == 1, "one left");
        req(c.paths().size() == 1, "and the index shrank with it");
    }

    // ---- integer time is exact where doubles were not ------------------------------------
    {
        // 1/3 s at 59.94 is not representable as a double, and the old engine's 1 ms tolerance
        // would have merged two keys 0.9 ms apart. In flicks both are exact integers.
        const auto  fps = boost::rational<int>(60000, 1001);
        const flicks f1  = flicks_per_frame(fps);
        curve       c;
        c.add(key(0, "linear", {{"x", 0.0}}));
        c.add(key(f1 * 100, "linear", {{"x", 1.0}}));
        req(close_to(c.interpolate(f1 * 50, kind_of).at("x"), 0.5, 1e-15),
            "frame 50 of 100 is exactly half, at 59.94");
        req(close_to(c.interpolate(f1 * 100, kind_of).at("x"), 1.0), "and frame 100 is exactly the end");
    }

    CASPAR_LOG(info) << L"[timeline-curve] self-test: all checks passed, over "
                     << tweener::names().size() << L" easing names and "
                     << easing_aliases().size() << L" legacy aliases";
}

}}} // namespace caspar::core::timeline
