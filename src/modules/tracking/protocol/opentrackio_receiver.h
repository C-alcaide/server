/*
 * Copyright (c) 2025 CasparCG Contributors
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

#include "receiver_base.h"

#include <cstdint>
#include <memory>
#include <string>

namespace caspar { namespace tracking {

/// SMPTE RIS OSVP OpenTrackIO receiver (JSON-over-UDP-multicast transport).
///
/// Listens on a UDP multicast group for OpenTrackIO samples and decodes the
/// camera transform (translation/rotation) and lens encoders (focus/iris/zoom),
/// forwarding normalised camera_data to tracker_registry::on_data().
///
/// The receiver accepts both raw JSON datagrams and "OTrk"-framed datagrams: it
/// extracts the JSON object from the payload, tolerating a transport header and
/// trailing checksum bytes. Each sample must fit in a single UDP datagram
/// (JSON-first; multi-segment reassembly and CBOR are not handled here).
///
/// Coordinate mapping: OpenTrackIO uses a right-handed metric world. Translation
/// (metres) is mapped to camera_data (millimetres) via a documented axis map;
/// the named Euler angles pan/tilt/roll (degrees) map directly to radians.
///
/// \param port            UDP port to bind (default 55555).
/// \param multicast_addr  Multicast group to join (default "239.135.1.100").
class opentrackio_receiver : public receiver_base
{
  public:
    explicit opentrackio_receiver(uint16_t port = 55555, std::string multicast_addr = "239.135.1.100");
    ~opentrackio_receiver() override;

    void        start() override;
    void        stop() override;
    std::string info() const override;

  private:
    struct impl;
    std::unique_ptr<impl> impl_;
};

}} // namespace caspar::tracking
