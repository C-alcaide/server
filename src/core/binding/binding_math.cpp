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

#include "../StdAfx.h"

#include "binding_math.h"

#include <common/log.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>

namespace caspar { namespace core { namespace binding {

namespace {

double clamp01(double x) { return x < 0.0 ? 0.0 : (x > 1.0 ? 1.0 : x); }

/// A deterministic hash of an integer, in 0..1. For `wave_t::noise`.
///
/// A hash rather than a PRNG with state, because an LFO must give the same value for the same
/// phase whatever order it is sampled in -- otherwise two bindings on one source disagree, and
/// a seek changes the waveform. This is the integer mix from Thomas Wang, whose only property
/// that matters here is that consecutive inputs give unrelated outputs.
double hash01(int64_t n)
{
    uint64_t x = static_cast<uint64_t>(n) * 0x9E3779B97F4A7C15ull;
    x ^= x >> 30;
    x *= 0xBF58476D1CE4E5B9ull;
    x ^= x >> 27;
    x *= 0x94D049BB133111EBull;
    x ^= x >> 31;
    // The top 53 bits, so the result is exactly representable and uniform in [0, 1).
    return static_cast<double>(x >> 11) / static_cast<double>(1ull << 53);
}

bool iequal(std::string_view a, std::string_view b)
{
    if (a.size() != b.size())
        return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const auto ca = static_cast<char>(std::tolower(static_cast<unsigned char>(a[i])));
        const auto cb = static_cast<char>(std::tolower(static_cast<unsigned char>(b[i])));
        if (ca != cb)
            return false;
    }
    return true;
}

} // namespace

double apply_curve(curve_t c, double x)
{
    x = clamp01(x);
    switch (c) {
        case curve_t::linear:
            return x;
        case curve_t::ease_in:
            return x * x;
        case curve_t::ease_out:
            return 1.0 - (1.0 - x) * (1.0 - x);
        case curve_t::ease:
            // Smoothstep. Chosen over a cosine ease because its derivative is zero at BOTH
            // ends exactly, which is the property that makes a parameter stop cleanly.
            return x * x * (3.0 - 2.0 * x);
        case curve_t::step:
            return x >= 0.5 ? 1.0 : 0.0;
        case curve_t::invert:
            return 1.0 - x;
    }
    return x;
}

double map_value(const transform& t, double source)
{
    const double span = t.in_hi - t.in_lo;

    // A zero-width input range gives 0 rather than a division by zero or a NaN. A NaN reaching
    // a mixer field turns every pixel it touches black -- `grade_range::contains` is written as
    // a positive test for exactly that reason -- so it must not be reachable from here.
    double x = span == 0.0 ? 0.0 : (source - t.in_lo) / span;

    x = apply_curve(t.curve, x);
    x *= t.gain;

    // Clamped AFTER the gain, so a gain of 2 saturates rather than overshooting the declared
    // output range. The alternative -- letting it overshoot -- means `MIN`/`MAX` do not bound
    // what they say they bound, and every target's own range check would then have to catch it.
    x = clamp01(x);

    return t.out_lo + x * (t.out_hi - t.out_lo);
}

double apply_lag(double target, double previous, double lag_ms, double dt_ms)
{
    if (!(lag_ms > 0.0) || !(dt_ms > 0.0))
        return target;

    // One-pole, with the coefficient derived from the time constant rather than chosen: after
    // `lag_ms` the output has closed 1 - 1/e = 63.2% of the gap. So an operator who types 200
    // gets a settle they can predict, and the behaviour does not change with the frame rate --
    // which a bare `previous * 0.9 + target * 0.1` would.
    const double alpha = 1.0 - std::exp(-dt_ms / lag_ms);
    return previous + (target - previous) * alpha;
}

double wave_sample(wave_t w, double phase)
{
    // Wrapped into [0, 1) with a positive modulus, so a negative phase behaves.
    double p = std::fmod(phase, 1.0);
    if (p < 0.0)
        p += 1.0;

    switch (w) {
        case wave_t::sine:
            // 0 at phase 0, 1 at 0.25, 0 at 0.5 -- a UNIPOLAR sine, because a parameter range
            // is expressed by MIN/MAX and a bipolar source would halve the usable range.
            return 0.5 - 0.5 * std::cos(2.0 * 3.14159265358979323846 * p);
        case wave_t::triangle:
            return p < 0.5 ? p * 2.0 : 2.0 - p * 2.0;
        case wave_t::saw:
            return p;
        case wave_t::square:
            return p < 0.5 ? 0.0 : 1.0;
        case wave_t::noise: {
            // Value noise: a hash per integer turn, smoothstep-interpolated between. Not white
            // noise -- a parameter driven by white noise is unusable, and every product's
            // "noise" source is this.
            const double f = std::floor(phase);
            const double t = phase - f;
            const auto   n = static_cast<int64_t>(f);
            const double a = hash01(n);
            const double b = hash01(n + 1);
            return a + (b - a) * (t * t * (3.0 - 2.0 * t));
        }
    }
    return 0.0;
}

bool parse_curve(std::string_view name, curve_t& out)
{
    if (iequal(name, "linear")) { out = curve_t::linear; return true; }
    if (iequal(name, "ease_in") || iequal(name, "easein")) { out = curve_t::ease_in; return true; }
    if (iequal(name, "ease_out") || iequal(name, "easeout")) { out = curve_t::ease_out; return true; }
    if (iequal(name, "ease") || iequal(name, "smooth")) { out = curve_t::ease; return true; }
    if (iequal(name, "step")) { out = curve_t::step; return true; }
    if (iequal(name, "invert")) { out = curve_t::invert; return true; }
    return false;
}

bool parse_wave(std::string_view name, wave_t& out)
{
    if (iequal(name, "sine") || iequal(name, "sin")) { out = wave_t::sine; return true; }
    if (iequal(name, "triangle") || iequal(name, "tri")) { out = wave_t::triangle; return true; }
    if (iequal(name, "saw") || iequal(name, "ramp")) { out = wave_t::saw; return true; }
    if (iequal(name, "square") || iequal(name, "sqr")) { out = wave_t::square; return true; }
    if (iequal(name, "noise")) { out = wave_t::noise; return true; }
    return false;
}

const char* curve_name(curve_t c)
{
    switch (c) {
        case curve_t::linear: return "linear";
        case curve_t::ease_in: return "ease_in";
        case curve_t::ease_out: return "ease_out";
        case curve_t::ease: return "ease";
        case curve_t::step: return "step";
        case curve_t::invert: return "invert";
    }
    return "linear";
}

const char* wave_name(wave_t w)
{
    switch (w) {
        case wave_t::sine: return "sine";
        case wave_t::triangle: return "triangle";
        case wave_t::saw: return "saw";
        case wave_t::square: return "square";
        case wave_t::noise: return "noise";
    }
    return "sine";
}

// ---------------------------------------------------------------------------------------
// Self-test
// ---------------------------------------------------------------------------------------

namespace {

int failures = 0;

void check(bool ok, const std::string& what, const std::string& detail = "")
{
    if (ok)
        return;
    ++failures;
    CASPAR_LOG(error) << L"[binding-math] SELF-TEST FAILED: " << u16(what)
                      << (detail.empty() ? L"" : (L" -- " + u16(detail)));
}

void close(double got, double want, const std::string& what, double tol = 1e-9)
{
    check(std::abs(got - want) <= tol, what,
          "got " + std::to_string(got) + ", want " + std::to_string(want));
}

} // namespace

int binding_math_self_test()
{
    failures = 0;

    // ---- the curves, at the three points that define each --------------------------------
    for (auto c : {curve_t::linear, curve_t::ease_in, curve_t::ease_out, curve_t::ease}) {
        close(apply_curve(c, 0.0), 0.0, std::string("curve/") + curve_name(c) + "/at-0");
        close(apply_curve(c, 1.0), 1.0, std::string("curve/") + curve_name(c) + "/at-1");
    }
    close(apply_curve(curve_t::linear, 0.25), 0.25, "curve/linear/quarter");
    close(apply_curve(curve_t::ease_in, 0.5), 0.25, "curve/ease_in/half");
    close(apply_curve(curve_t::ease_out, 0.5), 0.75, "curve/ease_out/half");
    close(apply_curve(curve_t::ease, 0.5), 0.5, "curve/ease/half");
    close(apply_curve(curve_t::step, 0.4999), 0.0, "curve/step/below");
    close(apply_curve(curve_t::step, 0.5), 1.0, "curve/step/at");
    close(apply_curve(curve_t::invert, 0.25), 0.75, "curve/invert");

    // Every curve is MONOTONE except `invert`, which is monotone decreasing. Checked as a
    // property rather than at sample points, because a sign error inside one of the eases
    // would satisfy the endpoints above and break here.
    for (auto c : {curve_t::linear, curve_t::ease_in, curve_t::ease_out, curve_t::ease}) {
        bool mono = true;
        for (int i = 1; i <= 64; ++i) {
            const double a = apply_curve(c, (i - 1) / 64.0);
            const double b = apply_curve(c, i / 64.0);
            if (b < a - 1e-12)
                mono = false;
        }
        check(mono, std::string("curve/") + curve_name(c) + "/monotone");
    }

    // Clamped OUTSIDE 0..1 rather than extrapolating, on every curve.
    for (auto c : {curve_t::linear, curve_t::ease_in, curve_t::ease_out, curve_t::ease,
                   curve_t::step, curve_t::invert}) {
        const double lo = apply_curve(c, -3.0);
        const double hi = apply_curve(c, 4.0);
        check(lo >= -1e-12 && lo <= 1.0 + 1e-12 && hi >= -1e-12 && hi <= 1.0 + 1e-12,
              std::string("curve/") + curve_name(c) + "/clamped-outside");
    }

    // ---- the range map -------------------------------------------------------------------
    {
        transform t;                        // 0..1 in, 0..1 out, gain 1, linear
        close(map_value(t, 0.0), 0.0, "map/identity/lo");
        close(map_value(t, 0.5), 0.5, "map/identity/mid");
        close(map_value(t, 1.0), 1.0, "map/identity/hi");

        // Asymmetric on BOTH sides, so an in/out confusion fails: a transform that swapped the
        // two ranges would still pass a symmetric test.
        t.in_lo = -10.0;
        t.in_hi = 30.0;
        t.out_lo = 2.0;
        t.out_hi = 6.0;
        // Source 0 is a QUARTER of the way through [-10, 30], which is 3.0 out of [2, 6].
        //
        // The first version of this line used source 10 and expected 3.0, calling 10 "a quarter
        // of the way in" -- it is the MIDPOINT of that range, so the correct answer was 4.0 and
        // the self-test failed on its first boot. The failure was in the expectation, not in
        // `map_value`, and it is worth leaving the arithmetic written out: an asymmetric range
        // is the whole point of this check and it is exactly where an off-by-a-factor hides.
        close(map_value(t, 0.0), 3.0, "map/asymmetric/quarter");
        close(map_value(t, 10.0), 4.0, "map/asymmetric/midpoint");
        close(map_value(t, -10.0), 2.0, "map/asymmetric/lo");
        close(map_value(t, 30.0), 6.0, "map/asymmetric/hi");
        close(map_value(t, -999.0), 2.0, "map/below-clamps");
        close(map_value(t, 999.0), 6.0, "map/above-clamps");

        // A DESCENDING output range is legal and must invert: MIN 1 MAX 0 is how an operator
        // says "louder means dimmer", and refusing it would send them to CURVE invert for
        // something the range already expresses.
        t.out_lo = 6.0;
        t.out_hi = 2.0;
        close(map_value(t, -10.0), 6.0, "map/descending/lo");
        close(map_value(t, 30.0), 2.0, "map/descending/hi");
    }

    // Gain saturates INTO the output range rather than overshooting it.
    {
        transform t;
        t.gain = 2.0;
        close(map_value(t, 0.25), 0.5, "map/gain/inside");
        close(map_value(t, 0.75), 1.0, "map/gain/saturates");
        t.gain = 0.0;
        close(map_value(t, 1.0), 0.0, "map/gain/zero");
    }

    // A zero-width input range gives the output floor, never a NaN. A NaN in a mixer field
    // turns every pixel it touches black, so this is the check that keeps it unreachable.
    {
        transform t;
        t.in_lo = t.in_hi = 5.0;
        const double v = map_value(t, 5.0);
        check(!std::isnan(v) && v == 0.0, "map/zero-width-range", std::to_string(v));
    }

    // ---- the lag -------------------------------------------------------------------------
    close(apply_lag(1.0, 0.0, 0.0, 40.0), 1.0, "lag/off-is-off");
    close(apply_lag(1.0, 0.0, 100.0, 0.0), 1.0, "lag/zero-dt");

    // One time constant closes 63.2% of the gap. This is what makes the number an operator
    // types mean something -- a hand-picked coefficient would be frame-rate dependent.
    close(apply_lag(1.0, 0.0, 100.0, 100.0), 1.0 - std::exp(-1.0), "lag/one-time-constant", 1e-12);

    // Frame-rate independence: the same wall time reaches the same value whatever the tick.
    {
        double a = 0.0;
        for (int i = 0; i < 4; ++i)
            a = apply_lag(1.0, a, 100.0, 25.0);   // 4 x 25 ms
        double b = 0.0;
        for (int i = 0; i < 10; ++i)
            b = apply_lag(1.0, b, 100.0, 10.0);   // 10 x 10 ms
        close(a, b, "lag/frame-rate-independent", 1e-12);
    }

    // Converges, and never overshoots.
    {
        double v = 0.0;
        bool   over = false;
        for (int i = 0; i < 1000; ++i) {
            v = apply_lag(1.0, v, 50.0, 40.0);
            if (v > 1.0 + 1e-12)
                over = true;
        }
        check(!over, "lag/no-overshoot");
        close(v, 1.0, "lag/converges", 1e-9);
    }

    // ---- the waveforms -------------------------------------------------------------------
    close(wave_sample(wave_t::sine, 0.0), 0.0, "wave/sine/0");
    close(wave_sample(wave_t::sine, 0.25), 0.5, "wave/sine/quarter", 1e-12);
    close(wave_sample(wave_t::sine, 0.5), 1.0, "wave/sine/half", 1e-12);
    close(wave_sample(wave_t::sine, 0.75), 0.5, "wave/sine/three-quarters", 1e-12);

    close(wave_sample(wave_t::triangle, 0.0), 0.0, "wave/tri/0");
    close(wave_sample(wave_t::triangle, 0.5), 1.0, "wave/tri/half");
    close(wave_sample(wave_t::triangle, 0.25), 0.5, "wave/tri/quarter");

    close(wave_sample(wave_t::saw, 0.0), 0.0, "wave/saw/0");
    close(wave_sample(wave_t::saw, 0.99), 0.99, "wave/saw/near-1");

    close(wave_sample(wave_t::square, 0.25), 0.0, "wave/square/first-half");
    close(wave_sample(wave_t::square, 0.75), 1.0, "wave/square/second-half");

    // Every waveform stays inside 0..1 over two full turns, INCLUDING noise -- whose
    // interpolation is the one place an out-of-range value could appear.
    for (auto w : {wave_t::sine, wave_t::triangle, wave_t::saw, wave_t::square, wave_t::noise}) {
        bool inside = true;
        for (int i = 0; i < 512; ++i) {
            const double v = wave_sample(w, i / 256.0);
            if (v < -1e-12 || v > 1.0 + 1e-12)
                inside = false;
        }
        check(inside, std::string("wave/") + wave_name(w) + "/in-range");
    }

    // Periodic with period 1, so a phase that has wrapped a hundred times is the same sample.
    // This is what lets the phase be accumulated rather than recomputed, which matters because
    // a long-running LFO's phase is a large number.
    for (auto w : {wave_t::sine, wave_t::triangle, wave_t::saw, wave_t::square}) {
        close(wave_sample(w, 0.3), wave_sample(w, 100.3),
              std::string("wave/") + wave_name(w) + "/periodic", 1e-9);
        close(wave_sample(w, 0.3), wave_sample(w, -99.7),
              std::string("wave/") + wave_name(w) + "/negative-phase", 1e-9);
    }

    // Noise is DETERMINISTIC in the phase and NOT constant. Both halves matter: a hash that
    // returned the same value everywhere would satisfy determinism, and a stateful PRNG would
    // satisfy variation while making two bindings on one source disagree.
    {
        close(wave_sample(wave_t::noise, 7.25), wave_sample(wave_t::noise, 7.25),
              "wave/noise/deterministic");
        double lo = 2.0, hi = -1.0;
        for (int i = 0; i < 256; ++i) {
            const double v = wave_sample(wave_t::noise, i * 0.37);
            lo = std::min(lo, v);
            hi = std::max(hi, v);
        }
        check(hi - lo > 0.5, "wave/noise/varies",
              "range " + std::to_string(lo) + ".." + std::to_string(hi));
    }

    // ---- the parsers: every name round-trips, and a typo is REFUSED --------------------
    for (auto c : {curve_t::linear, curve_t::ease_in, curve_t::ease_out, curve_t::ease,
                   curve_t::step, curve_t::invert}) {
        curve_t got{};
        check(parse_curve(curve_name(c), got) && got == c,
              std::string("parse/curve/") + curve_name(c));
    }
    for (auto w : {wave_t::sine, wave_t::triangle, wave_t::saw, wave_t::square, wave_t::noise}) {
        wave_t got{};
        check(parse_wave(wave_name(w), got) && got == w,
              std::string("parse/wave/") + wave_name(w));
    }
    {
        curve_t c{};
        wave_t  w{};
        // Refused, not defaulted. A `CURVE eas` that silently became linear is a mapping the
        // operator did not ask for, applied on every frame, with a 202 in the log.
        check(!parse_curve("eas", c), "parse/curve/typo-refused");
        check(!parse_wave("sawtooth", w), "parse/wave/typo-refused");
        check(parse_curve("EASE_IN", c) && c == curve_t::ease_in, "parse/curve/case-insensitive");
        check(parse_wave("SINE", w) && w == wave_t::sine, "parse/wave/case-insensitive");
    }

    if (failures == 0)
        CASPAR_LOG(info) << L"[binding-math] self-test: all checks passed";

    return failures;
}

}}} // namespace caspar::core::binding
