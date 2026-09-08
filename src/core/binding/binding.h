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

// A parameter that is not a constant but is DRIVEN by a named live source through a transform.
//
// WHY THIS IS THE KEYSTONE AND NOT ANOTHER FEATURE. The fork already had the description half
// of a reactive engine: 178 registry rows carrying type, range, composition, bounding and
// keyframe names, three façades derived from them, and -- since the commit before this one --
// producer parameters in the same shape. What it had none of was the other half: no modulation
// sources at all (no LFO, no audio bands, no MIDI, no OSC input, no mouse), and no binding
// layer. Every parameter was addressable and every one of them could only be a constant or a
// keyframe timeline.
//
// TouchDesigner gives every parameter four modes -- Constant, Expression, EXPORT, Bind -- and
// Export is a CHOP channel continuously overriding the value. Resolume gives every parameter an
// animation source with an envelope over it. Hippotizer's pins are the same idea. All three
// converge on one sentence: **a parameter is either a constant or is driven by a named live
// source through a transform.** That sentence is this file.
//
// TWO DECISIONS THAT ARE NOT NEGOTIABLE, both from measurement:
//
//   * **Evaluated IN THE TICK, never through the API.** The control API's write path is a
//     single serial `http-api` executor blocking on `stage->apply_transform(...).get()` per
//     write -- tens of writes a second, which is fine for an operator and useless for something
//     continuous. Bindings are applied on the stage executor at the top of the channel's tick,
//     where the value is needed and where nothing has to queue.
//   * **A bound field is OWNED by its binding.** This is the `icvfx_auto` lesson stated as a
//     rule: `PREVIZ AUTOPROJECTION` wrote the ICVFX block on every recompute with no ownership
//     guard, so a hand-set `MIXER PROJECTION_ICVFX` survived exactly until the next camera
//     move. Nobody had chosen that precedence -- it was simply not thought about. Here it is
//     chosen and it is reported: a write to a bound field comes back `field_bound` rather than
//     succeeding and being silently overwritten on the next frame.
//
// WHAT IS DELIBERATELY NOT HERE. A node graph -- bindings are edges, and the client draws them;
// putting a graph in the server duplicates the client's own job. An expression language -- see
// `binding_math.h`. And layer post-effects, which are a mixer architecture change; `route://`
// plus a wrapping producer already composes.

#include "binding_math.h"

#include <core/input/input_event.h>
#include <core/monitor/monitor.h>

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace caspar { namespace core { namespace binding {

/// A named producer of float channels, sampled once per tick.
///
/// `value` returns false for an unknown channel rather than 0, because a binding to a channel
/// that does not exist must be visible as such: a source silently answering 0 for a
/// misspelled channel name is a binding that runs forever and does nothing, with a 202 behind
/// it -- the failure mode this whole registry exists to make impossible.
class source
{
  public:
    virtual ~source() = default;

    /// Advance any internal phase. Called once per channel tick, on the stage executor.
    virtual void tick(double dt_ms) {}

    /// Sample one channel. False if this source has no such channel.
    virtual bool value(const std::string& channel, double& out) const = 0;

    /// Every channel this source offers, for the tree and for an error message.
    virtual std::vector<std::string> channels() const = 0;

    /// `lfo`, `input`, `audio`, `osc`, ... For the published state and for `SOURCE LIST`.
    virtual std::string kind() const = 0;

    /// How the source describes itself in `SOURCE LIST` -- rate, waveform, port.
    virtual std::string describe() const { return kind(); }

    /// Offer an input event. Only `input_source` cares; the default ignores it, so the channel
    /// can hand every event to every source without asking what each one is.
    virtual void feed(const input_event&) {}
};

/// A sine/triangle/saw/square/noise oscillator. One channel, `value`.
///
/// The phase is ACCUMULATED rather than derived from a frame count, so a rate change does not
/// jump the output -- and `wave_sample` is periodic with period 1 and checked to be, which is
/// what makes accumulating safe over a long run.
class lfo_source final : public source
{
  public:
    lfo_source(wave_t wave, double rate_hz, double phase0 = 0.0);

    void                     tick(double dt_ms) override;
    bool                     value(const std::string& channel, double& out) const override;
    std::vector<std::string> channels() const override;
    std::string              kind() const override { return "lfo"; }
    std::string              describe() const override;

    void set_rate(double rate_hz) { rate_hz_ = rate_hz; }
    double phase() const { return phase_; }

  private:
    wave_t wave_;
    double rate_hz_;
    double phase_ = 0.0;
};

/// The pointer and keyboard, latched per tick. Channels: `x y buttons wheel`.
///
/// LATCHED, not queued: a binding wants the pointer's CURRENT position, and a source that
/// replayed every intermediate move would make one gesture produce a burst of writes. This is
/// TouchDesigner's Mouse In CHOP, which reports a position rather than a stream of events.
///
/// `wheel` ACCUMULATES and is not reset, so it behaves as a continuous encoder rather than as
/// an impulse -- a binding to a wheel wants a dial, and an impulse would need the binding to
/// integrate it, which is arithmetic the transform deliberately does not have.
class input_source final : public source
{
  public:
    void                     feed(const input_event& e) override;
    bool                     value(const std::string& channel, double& out) const override;
    std::vector<std::string> channels() const override;
    std::string              kind() const override { return "input"; }
    std::string              describe() const override { return "input (x y buttons wheel)"; }

  private:
    double   x_       = 0.0;
    double   y_       = 0.0;
    double   wheel_   = 0.0;
    uint32_t buttons_ = 0;
};

/// One binding: a target, a source channel, and the transform between them.
struct binding_def
{
    int id = 0;

    /// The layer the target lives on. A producer parameter and a mixer field are both per-layer;
    /// there is no channel-level target in v1.
    int layer = 0;

    /// A mixer field's registry path (`opacity`, `brightness`, `sat`) or `producer/<name>`.
    std::string target;

    /// Which component of a multi-component target, or 0. A binding drives ONE number: driving
    /// a whole vec3 from one scalar source is a different feature (a colour ramp), and pretending
    /// a scalar can fill three components would need a rule nobody has chosen.
    uint8_t component = 0;

    std::string source_name;
    std::string source_channel;

    transform tf;

    /// The last value written, for the lag. Not published: it is `held` only between ticks.
    double held    = 0.0;
    bool   primed  = false;

    /// Set when the last evaluation could not find the source or the channel. Published, so a
    /// misspelled name is visible rather than being a binding that quietly does nothing.
    bool broken = false;
};

/// Split `<source>/<channel>` -- `lfo1/value`, `input/x`. A missing channel is an error rather
/// than a default: `input` has four and picking one for the operator would be a guess.
bool split_source_ref(const std::string& ref, std::string& source, std::string& channel);

/// Split a target into a field path and a component index: `midtone.1`, `fill_x`, `opacity`.
///
/// The `.N` suffix is how a component is named, and it is chosen rather than inherited: the
/// registry's own KEYFRAMES names already provide per-component names for the fields that have
/// them (`mid_r`, `fill_x`), but only for animatable fields -- so a suffix that works for every
/// field is needed anyway, and having one form is better than two.
void split_target(const std::string& spec, std::string& field, uint8_t& component);

}}} // namespace caspar::core::binding
