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

#include <chrono>
#include <cstdint>

namespace caspar { namespace tracking {

/// Normalised camera tracking data as received from any protocol.
/// Angles are in radians; position is in millimetres.
///
/// THE FRAME IS LEFT-HANDED: x right, y up, z towards the subject, with a tracker at rest facing
/// +z. Every receiver converts its protocol's own frame into this one, and every mode converts out
/// of it -- `mode_previz` into the previz world, which is RIGHT-handed and looks along -Z (see
/// `tracker_registry::inject_transform`). `set_position_z_up` below is the FreeD/OpenTrackIO half.
struct camera_data
{
    int camera_id = 0; ///< Camera / device identifier (from packet or binding config)

    // Orientation (radians)
    double pan  = 0.0; ///< Yaw:   right is positive (clockwise seen from above)
    double tilt = 0.0; ///< Pitch: up is positive
    double roll = 0.0; ///< Roll:  clockwise seen from BEHIND the camera is positive

    // Position (mm)
    double x = 0.0; ///< Right
    double y = 0.0; ///< Up
    double z = 0.0; ///< Towards subject

    // Optics (raw 16-bit, vendor-calibrated)
    uint16_t zoom  = 0; ///< 0 = max telephoto, 65535 = widest (de-facto convention)
    uint16_t focus = 0;
    uint16_t iris  = 0; ///< Aperture/iris (0 = closed, 65535 = open). OSC/VRPN only; FreeD leaves 0.

    std::chrono::steady_clock::time_point timestamp = std::chrono::steady_clock::now();

    /// WHICH RECEIVER produced this sample: `tracking_protocol` as an integer and the UDP port.
    /// Stamped by `tracker_registry::on_data(data, protocol, port)`. -1 means "not stamped",
    /// which matches any binding -- the behaviour before sources existed, kept for any caller
    /// that has not been told. See `tracker_registry::on_data` for why this exists at all.
    int source_protocol = -1;
    int source_port     = -1;
};

/// FreeD and OpenTrackIO share one native frame, and it is not camera_data's.
///
/// Both define RIGHT-HANDED axes fixed to the studio: X and Y horizontal, Z vertical and positive
/// upwards, and a camera at zero pan and tilt looking along +Y -- which puts +X at the camera's
/// right. Vinten free-d Installation Manual V1.4.4, Appendix B.1-B.2; SMPTE RIS OSVP OpenTrackIO
/// ("right-handed ... the Z-axis points upwards ... Y points in the forward camera direction").
/// Their ANGLES already carry camera_data's senses -- pan right, tilt up and roll clockwise from
/// behind are positive (free-d B.2-B.4), and SMPTE's own camdkit copies a Mo-Sys F4 pan, tilt and
/// roll into OpenTrackIO unchanged -- so only the position needs mapping: the protocol's forward
/// (Y) is camera_data's z and its height (Z) is camera_data's y.
///
/// The receivers copied X/Y/Z straight across until 2026-09-24, which put the camera's HEIGHT on the
/// forward axis and its travel towards the subject on the up axis.
inline void set_position_z_up(camera_data& d, double x_right_mm, double y_forward_mm, double z_up_mm)
{
    d.x = x_right_mm;
    d.y = z_up_mm;
    d.z = y_forward_mm;
}

}} // namespace caspar::tracking
