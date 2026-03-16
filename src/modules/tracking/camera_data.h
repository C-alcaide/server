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
struct camera_data
{
    int camera_id = 0; ///< Camera / device identifier (from packet or binding config)

    // Orientation (radians)
    double pan  = 0.0; ///< Yaw:   right is positive
    double tilt = 0.0; ///< Pitch: up is positive
    double roll = 0.0; ///< Roll:  clockwise is positive

    // Position (mm)
    double x = 0.0; ///< Right
    double y = 0.0; ///< Up
    double z = 0.0; ///< Towards subject

    // Optics (raw 16-bit, vendor-calibrated)
    uint16_t zoom  = 0; ///< 0 = max telephoto, 65535 = widest (de-facto convention)
    uint16_t focus = 0;

    std::chrono::steady_clock::time_point timestamp = std::chrono::steady_clock::now();
};

}} // namespace caspar::tracking
