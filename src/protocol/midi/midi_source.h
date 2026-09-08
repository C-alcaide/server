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

// A MIDI input device, as a binding source.
//
// Every VJ controller speaks MIDI and nothing else -- a bank of knobs and faders sends control
// change, a pad grid sends note on/off, and that is the entire vocabulary this needs. Resolume,
// TouchDesigner and Hippotizer all take MIDI as a first-class animation source, and it is the
// one input in this family an operator is likely to already own hardware for.
//
// CHANNELS, and they are named after what the wire calls them so a controller's own manual is
// the documentation:
//
//     cc/<n>        control change 0..127, normalised to 0..1
//     note/<n>      note on/off velocity, normalised to 0..1 (0 on note off)
//     pitch         pitch bend, 14-bit, normalised to 0..1 with 0.5 at centre
//     messages      how many messages have arrived. Telemetry, for the same reason the OSC
//                   receiver publishes it: "nothing is connected" and "nothing is being turned"
//                   are different faults
//
// NORMALISED TO 0..1 rather than reported raw, because that is the range every source in this
// build produces and it is what a binding's default `IN 0 1` expects. A raw 0..127 would make
// every MIDI binding need an explicit `IN 0 127`, which is a papercut on every single one.
//
// WINDOWS ONLY in this build, via `winmm`'s `midiInOpen`. That is a real limit and it is stated
// rather than hidden: on other platforms the source constructs, binds nothing, and reports every
// channel absent -- so a binding to it reads BROKEN, which is visible. ALSA is the Linux
// equivalent and is not written.
//
// THE MIDI CALLBACK RUNS ON A SYSTEM THREAD, not one of ours, and it must return promptly: the
// documented consequence of blocking in it is dropped input. So it does nothing but take a
// mutex and store a number.

#include <core/binding/binding.h>

#include <memory>
#include <string>
#include <vector>

namespace caspar { namespace protocol { namespace midi {

/// Every MIDI input device the system reports, in the order `midiInGetNumDevs` gives them.
///
/// The INDEX is what `SOURCE ADD ... MIDI <n>` takes, and the name is what makes it findable.
/// Enumerated rather than matched by name because two identical controllers report identical
/// names, and an operator with two of the same device has to be able to say which.
std::vector<std::string> input_devices();

/// One MIDI input device, open and listening.
class midi_source final : public core::binding::source
{
  public:
    explicit midi_source(int device_index);
    ~midi_source() override;

    bool                     value(const std::string& channel, double& out) const override;
    std::vector<std::string> channels() const override;
    std::string              kind() const override { return "midi"; }
    std::string              describe() const override;

    /// Did the device open? A source that did not still exists and reports nothing, so a
    /// binding to it is visibly BROKEN rather than silently inert.
    bool listening() const;

  private:
    struct impl;
    std::shared_ptr<impl> impl_;
};

}}} // namespace caspar::protocol::midi
