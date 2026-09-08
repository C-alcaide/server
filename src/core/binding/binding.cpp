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

#include "binding.h"

#include <algorithm>
#include <cstdlib>
#include <string>

namespace caspar { namespace core { namespace binding {

// ---------------------------------------------------------------------------------------
// lfo_source
// ---------------------------------------------------------------------------------------

lfo_source::lfo_source(wave_t wave, double rate_hz, double phase0)
    : wave_(wave)
    , rate_hz_(rate_hz)
    , phase_(phase0)
{
}

void lfo_source::tick(double dt_ms)
{
    phase_ += rate_hz_ * dt_ms * 0.001;

    // Wrapped back into [0, 1) once it has gone round. Not because `wave_sample` needs it -- it
    // is periodic and the self-test checks that -- but because the phase is PUBLISHED, and an
    // operator reading 41823.7 learns nothing where 0.7 tells them where in the cycle it is.
    if (phase_ >= 1.0 || phase_ < 0.0)
        phase_ -= std::floor(phase_);
}

bool lfo_source::value(const std::string& channel, double& out) const
{
    if (channel == "value") {
        out = wave_sample(wave_, phase_);
        return true;
    }
    if (channel == "phase") {
        out = phase_;
        return true;
    }
    return false;
}

std::vector<std::string> lfo_source::channels() const { return {"value", "phase"}; }

std::string lfo_source::describe() const
{
    return std::string("lfo ") + wave_name(wave_) + " " + std::to_string(rate_hz_) + " Hz";
}

// ---------------------------------------------------------------------------------------
// input_source
// ---------------------------------------------------------------------------------------

void input_source::feed(const input_event& e)
{
    if (e.has_position()) {
        x_ = e.x;
        y_ = e.y;
    }

    if (e.type == input_event::kind::button) {
        const uint32_t bit = e.button == 0   ? mod_left_button
                             : e.button == 1 ? mod_middle_button
                             : e.button == 2 ? mod_right_button
                                             : 0u;
        if (e.pressed)
            buttons_ |= bit;
        else
            buttons_ &= ~bit;
    }

    if (e.type == input_event::kind::wheel) {
        // ACCUMULATED and never reset -- a wheel binding wants an encoder, not an impulse. The
        // transform has no integrator (deliberately: see binding_math.h), so if this reported a
        // per-tick delta a wheel could only ever drive a value that snapped back to zero.
        // A `MIN`/`MAX` on the binding is what turns the running total into a usable range.
        wheel_ += e.wheel_dy;
    }
}

bool input_source::value(const std::string& channel, double& out) const
{
    if (channel == "x") {
        out = x_;
        return true;
    }
    if (channel == "y") {
        out = y_;
        return true;
    }
    if (channel == "wheel") {
        out = wheel_;
        return true;
    }
    if (channel == "buttons") {
        // The mask as a number, so `buttons` is one channel rather than three. A binding that
        // wants "is the left button down" uses `MIN 16 MAX 16` -- clumsy, and the alternative
        // (three boolean channels) was rejected because it doubles the channel list for a
        // source whose main use is x and y.
        out = static_cast<double>(buttons_);
        return true;
    }
    return false;
}

std::vector<std::string> input_source::channels() const { return {"x", "y", "buttons", "wheel"}; }

// ---------------------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------------------

bool split_source_ref(const std::string& ref, std::string& source, std::string& channel)
{
    const auto slash = ref.rfind('/');
    if (slash == std::string::npos || slash == 0 || slash + 1 >= ref.size())
        return false;
    source  = ref.substr(0, slash);
    channel = ref.substr(slash + 1);
    return true;
}

void split_target(const std::string& spec, std::string& field, uint8_t& component)
{
    component = 0;
    field     = spec;

    // `producer/<name>` may legitimately contain a dot in the parameter's own name, so the
    // suffix is only recognised when what follows it is entirely digits. `producer/foo.bar`
    // is one name; `midtone.1` is a component.
    const auto dot = spec.rfind('.');
    if (dot == std::string::npos || dot + 1 >= spec.size())
        return;

    const auto tail = spec.substr(dot + 1);
    if (tail.find_first_not_of("0123456789") != std::string::npos)
        return;

    const auto n = std::atoi(tail.c_str());
    if (n < 0 || n > 3)
        return;

    field     = spec.substr(0, dot);
    component = static_cast<uint8_t>(n);
}

}}} // namespace caspar::core::binding
