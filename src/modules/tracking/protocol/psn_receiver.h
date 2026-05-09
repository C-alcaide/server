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
 *
 * PosiStageNet (PSN) protocol specification by VYV Corporation.
 * C++ reference implementation licensed under the MIT License.
 * See vendor/psn/ for the original source and license.
 */

#pragma once

#include "receiver_base.h"

#include <cstdint>
#include <memory>
#include <string>

namespace caspar { namespace tracking {

/// PosiStageNet (PSN) UDP multicast receiver.
///
/// Listens on a UDP multicast group for PSN v2.0 packets and decodes them
/// using the reference psn_decoder from VYV Corporation. Decoded tracker
/// data (position, orientation) is forwarded to tracker_registry::on_data().
///
/// PSN is an open protocol for live 3D position data on stage, commonly
/// used with BlackTrax, MA Lighting grandMA, and other stage-tracking
/// systems. Each PSN frame may carry multiple named trackers; each tracker
/// ID is mapped to a camera_id in camera_data.
///
/// PSN does not carry lens data (zoom/focus) — those fields remain at 0.
/// Position is converted from metres (PSN convention) to millimetres
/// (camera_data convention). Orientation is converted from degrees to
/// radians.
///
/// \param port            UDP port to bind (default 56565).
/// \param multicast_addr  Multicast group to join (default "236.10.10.10").
class psn_receiver : public receiver_base
{
  public:
    explicit psn_receiver(uint16_t port = 56565, std::string multicast_addr = "236.10.10.10");
    ~psn_receiver() override;

    void        start() override;
    void        stop() override;
    std::string info() const override;

  private:
    struct impl;
    std::unique_ptr<impl> impl_;
};

}} // namespace caspar::tracking
