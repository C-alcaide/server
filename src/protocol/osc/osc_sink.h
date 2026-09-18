/*
 * Copyright (c) 2011 Sveriges Television AB <info@casparcg.com>
 *
 * This file is part of CasparCG (www.casparcg.com).
 *
 * CasparCG is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * CasparCG is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with CasparCG. If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <core/binding/binding.h>

#include <memory>
#include <string>

namespace caspar { namespace protocol { namespace osc {

/// An OUTBOUND binding's other end: one UDP endpoint that receives an OSC message per change.
///
/// The inverse of `osc_source`, and the server half of DECISIONS L392 -- a parameter published
/// OUTBOUND over a protocol, so a client's binding matrix can carry an out-arrow. The server
/// has always broadcast its whole state over OSC to subscribed endpoints (`osc::client`); this
/// is different in kind: ONE address, ONE value, sent only when it changes, to an endpoint an
/// operator named. A lighting desk that wants layer 10's opacity on `/fader/3` does not want
/// 600 leaves per tick.
///
/// The value goes as a single OSC float32 (`f`), which is what desks, TouchOSC and the like
/// expect; a double would be dropped by most of them. Sending is a non-blocking `send_to` on
/// the stage tick: a full socket buffer drops the message and counts it rather than stalling a
/// frame, and `describe()` reports both counts so "nothing arrived" is distinguishable from
/// "nothing was sent".
class osc_sink final : public core::binding::sink
{
  public:
    /// `host` must be a numeric address (v4 or v6); a hostname is refused by the command that
    /// constructs this, so a typo in it cannot silently resolve to nothing.
    osc_sink(std::string name, const std::string& host, unsigned short port);
    ~osc_sink() override;

    bool        send(const std::string& address, double value) override;
    std::string kind() const override { return "osc"; }
    std::string describe() const override;
    bool        ready() const override;

  private:
    struct impl;
    std::shared_ptr<impl> impl_;
};

}}} // namespace caspar::protocol::osc
