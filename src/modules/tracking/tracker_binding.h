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

#include "camera_data.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#ifndef M_PI
#define M_PI 3.141592653589793
#endif

namespace caspar { namespace core {
class stage_base;
}} // namespace caspar::core

namespace caspar { namespace tracking {

enum class tracking_mode
{
    mode_360, ///< 360° equirectangular — injects into projection.{yaw,pitch,roll,fov}
    mode_2d,  ///< 2D layer — injects into fill_translation, angle, fill_scale
};

enum class tracking_protocol
{
    freed,      ///< FreeD D-type (D1) UDP, the broadcast industry standard
    freed_plus, ///< FreeD+ / Stype extended-precision variant
    osc,        ///< OSC 1.0 UDP (schema: /camera/{id}/pan|tilt|roll|zoom|x|y|z)
    vrpn,       ///< VRPN pose tracker + analogue zoom channel
};

/// Identifies which receiver a binding was created on (needed for clean release).
struct receiver_handle
{
    tracking_protocol protocol  = tracking_protocol::freed;
    int               port      = 6301;
    std::string       host; ///< Only used by VRPN
};

/// Optional per-lens zoom calibration entry.
struct zoom_entry
{
    uint16_t raw_value = 0;   ///< FreeD raw zoom value
    double   fov_rad   = 0.0; ///< Corresponding FOV in radians
};

/// Per-layer binding: maps a camera_id to a CasparCG layer with transform parameters.
struct tracker_binding
{
    int camera_id = 0; ///< Camera ID to accept. -1 = accept any camera on this receiver.

    std::weak_ptr<core::stage_base> stage;
    int                             layer_index = 0;

    tracking_mode mode = tracking_mode::mode_360;

    // ------- Angular offsets (radians, added after scaling) --------
    double pan_offset  = 0.0;
    double tilt_offset = 0.0;
    double roll_offset = 0.0;

    // ------- Axis scale factors ------------------------------------
    // Multiply raw decoded radians before applying offset.
    // Use negative values to flip an axis (e.g. counter-tracking).
    double pan_scale  = 1.0;
    double tilt_scale = 1.0;

    // ------- Zoom mapping ------------------------------------------
    // Lens formula: fov = 2*atan(tan(zoom_default_fov/2) * zoom_full_range / max(zoom_raw,1))
    // zoom_full_range should match the raw value the vendor sends for the widest angle.
    double zoom_full_range  = 65535.0;
    double zoom_default_fov = M_PI / 2.0; ///< FOV (rad) when zoom == zoom_full_range (90° default)

    /// Optional lookup table (sorted ascending by raw_value).
    /// If non-empty the formula above is replaced by linear interpolation.
    std::vector<zoom_entry> zoom_lookup;

    // ------- Position mapping --------------------------------------
    // Scale factor applied to raw X/Y/Z (mm) to produce NDC units.
    //
    // 360 mode:  X → projection.offset_x,  Y → projection.offset_y  (horizontal/vertical lens shift)
    //            Z is not mapped (no depth concept inside an equirectangular sphere).
    // 2D mode:   X → additional fill_translation.x parallax offset
    //            Y → additional fill_translation.y parallax offset (inverted so up = up)
    //            Z is not mapped (interacts poorly with zoom-based scale).
    //
    // Default 0.001 NDC/mm means 1 metre of camera travel ≈ 1.0 NDC of offset.
    double position_scale = 0.001;

    /// Tracks which receiver this binding was registered on (for reference counting).
    receiver_handle receiver;
};

}} // namespace caspar::tracking
