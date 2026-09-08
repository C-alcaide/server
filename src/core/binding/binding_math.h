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

// The arithmetic a binding does between a source and a target, and the waveforms an LFO
// generates. No I/O, no state beyond what is passed in, no dependency on a channel -- so it is
// checkable at every server start and a divergence attributable to one line.
//
// THE SHAPE IS RESOLUME'S ANIMATION MENU AND NOTHING MORE.
//
//     value = curve( clamp01( (source - in_lo) / (in_hi - in_lo) ) ) * gain
//     out   = out_lo + value * (out_hi - out_lo)
//     held  = lag(out, previous, dt)
//
// Resolume gives every parameter an animation source -- Timeline, BPM, FFT with a band and a
// gain and a "Fall", Dashboard, Clip position, Crossfader -- and an envelope over any of them.
// TouchDesigner's Export mode plus a Lag CHOP is the same four operations. Hippotizer's pins
// are the same idea again. Three products converged on: input range, output range, gain, a
// one-pole lag, and a curve.
//
// WHAT IS DELIBERATELY ABSENT: an expression language. TouchDesigner's expressions are Python
// and that is a rabbit hole with no bottom; the four operations above cover what a VJ parameter
// mapping actually does, and a fifth can be argued for on evidence rather than added on
// speculation.

#include <cstdint>
#include <string_view>

namespace caspar { namespace core { namespace binding {

/// The response curve applied to the NORMALISED source value, before the output range.
enum class curve_t : uint8_t
{
    linear,   ///< identity
    ease_in,  ///< x^2 -- slow to start
    ease_out, ///< 1-(1-x)^2 -- slow to finish
    ease,     ///< smoothstep, slow at both ends
    step,     ///< 0 below 0.5, 1 at or above it. For a discrete target
    invert,   ///< 1-x. Not a curve, and here because every product's menu has it
};

/// An LFO waveform. Phase is in turns (0..1), not radians -- a musician thinks in bars.
enum class wave_t : uint8_t
{
    sine,
    triangle,
    saw,     ///< ramps 0 -> 1 and resets
    square,  ///< 0 for the first half turn, 1 for the second
    noise,   ///< value noise: a deterministic hash per integer turn, interpolated
};

/// The whole transform, as declared by a `BIND` command.
struct transform
{
    /// The source range mapped to 0..1. `in_hi == in_lo` is not an error: it makes the
    /// normalised value 0, which is the same thing a zero-width slider does.
    double in_lo = 0.0;
    double in_hi = 1.0;

    /// Where the normalised, curved, gained value lands. `MIN` and `MAX` in the command.
    double out_lo = 0.0;
    double out_hi = 1.0;

    double gain = 1.0;

    /// One-pole smoothing time in milliseconds -- Resolume's "Fall". 0 is no smoothing.
    ///
    /// Defined as the time to close 63.2% of the gap (one time constant), so a number an
    /// operator types has a meaning they can predict rather than being a coefficient.
    double lag_ms = 0.0;

    curve_t curve = curve_t::linear;
};

/// Apply a curve to a value already in 0..1.
double apply_curve(curve_t c, double x);

/// The transform WITHOUT the lag: range, curve, gain, output range.
///
/// Separate from the lag because the lag needs the previous output and the elapsed time, and
/// separating them is what lets the self-test check the static part against closed-form values
/// with no state at all.
double map_value(const transform& t, double source);

/// One step of the one-pole lag. `previous` is the last held value, `dt_ms` the tick length.
///
/// `lag_ms <= 0` returns `target` unchanged, so the smoothing is genuinely off rather than
/// approximately off -- an operator who did not ask for smoothing must not get a frame of it.
double apply_lag(double target, double previous, double lag_ms, double dt_ms);

/// One LFO sample. `phase` in TURNS; values outside 0..1 wrap.
///
/// Deterministic in the phase alone, with no internal state, which is what makes an LFO
/// reproducible across a seek and checkable at boot. The caller advances the phase.
double wave_sample(wave_t w, double phase);

/// Parse a curve name, case-insensitively. False for an unknown name -- so a typo is refused
/// rather than silently becoming linear.
bool parse_curve(std::string_view name, curve_t& out);

/// Parse a waveform name, case-insensitively. False for an unknown name.
bool parse_wave(std::string_view name, wave_t& out);

/// The names, for a descriptor and for an error message.
const char* curve_name(curve_t c);
const char* wave_name(wave_t w);

/// Property checks over the whole file, run at every server start.
///
/// Returns the number of FAILURES and logs each. Same contract as `compose_self_test` and
/// `stage_math_self_test`: a divergence is named at boot rather than found in a picture.
int binding_math_self_test();

}}} // namespace caspar::core::binding
