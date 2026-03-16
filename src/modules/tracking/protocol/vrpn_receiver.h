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
 * This file optionally links against VRPN (https://github.com/vrpn/vrpn),
 * licensed under the Boost Software License 1.0 (BSL-1.0),
 * which is compatible with GPL-3. VRPN is only required when building
 * with -DBUILD_TRACKING_VRPN=ON.
 */

#pragma once

#include "receiver_base.h"

#include <cstdint>
#include <memory>
#include <string>

namespace caspar { namespace tracking {

/// VRPN pose tracker receiver.
///
/// Connects to a VRPN server as a tracker client (vrpn_Tracker_Remote).
/// Quaternion poses from the tracker are converted to yaw/pitch/roll (Euler
/// angles, ZYX convention, right-hand / Y-up, matching CasparVP's 360
/// coordinate system).
///
/// Zoom is read from a VRPN analogue channel (vrpn_Analog_Remote) on channel
/// index 0 of the same device name, and forwarded as a 16-bit raw zoom value
/// by scaling the normalised [0,1] analogue output by 65535.
///
/// This receiver is only compiled when CASPAR_TRACKING_WITH_VRPN is defined
/// (set automatically by CMake when BUILD_TRACKING_VRPN=ON).
///
/// \param host_url  VRPN URL, e.g. "Tracker0@localhost" or "MyDevice@192.168.1.100"
/// \param camera_id Camera ID to stamp on forwarded camera_data packets.
/// \param sensor    VRPN sensor index (default 0).
class vrpn_receiver : public receiver_base
{
  public:
    explicit vrpn_receiver(std::string host_url, int camera_id = 0, int sensor = 0);
    ~vrpn_receiver() override;

    void        start() override;
    void        stop() override;
    std::string info() const override;

  private:
    struct impl;
    std::unique_ptr<impl> impl_;
};

}} // namespace caspar::tracking
