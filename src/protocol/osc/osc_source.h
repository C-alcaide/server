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

// An OSC RECEIVER, as a binding source.
//
// The server has spoken OSC outbound since forever -- `protocol/osc/client.cpp` sends the whole
// channel state to any number of subscribers -- and has never listened. That asymmetry is the
// reason this file exists: every product this feature is modelled on takes OSC IN as its
// primary remote-control surface (Resolume, TouchDesigner, ossia, Hippotizer), and a control
// surface, a phone app or a lighting desk speaks it without being told to.
//
// WHAT IT IS NOT. Not a second control API: it does not run commands, it does not address
// layers, and it cannot start a clip. It turns `/casparcg/source/<name> <float>` into a named
// channel on a source, and that is the whole of it. Anything larger belongs to the control API,
// which already exists and is authenticated.
//
// AND IT IS NOT THE TRACKING MODULE'S RECEIVER. `modules/tracking` has an OSC receiver of its
// own, and it stays: it is camera-locked by design (it parses pose into a `camera_data`) and
// merging the two would give one of them a job it was not shaped for. Two receivers on two
// ports is the cheaper answer than one receiver with a mode flag.
//
// ADDRESS SHAPE, and why it is a prefix rather than a free-for-all:
//
//     /casparcg/source/<name>              one float, or the first of several
//     /casparcg/source/<name>/<channel>    a named channel on that source
//
// A source created as `SOURCE 1 ADD knobs OSC 7400` answers to `/casparcg/source/knobs/...`.
// Restricting the prefix means a packet meant for something else on the same port cannot
// silently populate a channel, and it means a misaddressed packet is DROPPED and counted rather
// than half-applied -- `packets` and `dropped` are both published, so "nothing is arriving" and
// "something is arriving at the wrong address" are different readings.

#include <core/binding/binding.h>

#include <memory>
#include <string>

namespace caspar { namespace protocol { namespace osc {

/// A UDP listener on one port, exposing whatever arrives under `/casparcg/source/<name>/`.
///
/// Owns its own thread and socket. Constructed by `SOURCE ADD ... OSC <port>`; destroyed when
/// the source is replaced or removed, which closes the socket and joins the thread -- so a
/// second `SOURCE ADD` on the same port works rather than failing to bind.
class osc_source final : public core::binding::source
{
  public:
    /// `name` is the address segment this source answers to, not the source's registry key --
    /// they are the same by construction in `SOURCE ADD`, and separate here so a caller could
    /// register one receiver under two names if that ever makes sense.
    osc_source(std::string name, unsigned short port);
    ~osc_source() override;

    bool                     value(const std::string& channel, double& out) const override;
    std::vector<std::string> channels() const override;
    std::string              kind() const override { return "osc"; }
    std::string              describe() const override;

    /// Did the socket bind? A source that could not bind still exists and reports every channel
    /// as absent, so a binding to it goes `broken` -- which is visible, unlike a source that
    /// silently never receives.
    bool listening() const;

  private:
    struct impl;
    std::shared_ptr<impl> impl_;
};

}}} // namespace caspar::protocol::osc
