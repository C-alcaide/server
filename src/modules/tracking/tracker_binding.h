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

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace caspar { namespace core {
class stage_base;
}} // namespace caspar::core

namespace caspar { namespace tracking {

class lens_profile;

enum class tracking_mode
{
    mode_360,   ///< 360° equirectangular — injects into projection.{yaw,pitch,roll,fov}
    mode_2d,    ///< 2D layer — injects into fill_translation, angle, fill_scale
    mode_previz,///< Previz — drives PREVIZ channel virtual camera
    mode_target,///< Track-target — projects a tracked SUBJECT world position through a
                ///< static virtual camera to drive the layer's screen position (AR follow).
};

enum class tracking_protocol
{
    freed,      ///< FreeD D-type (D1) UDP, the broadcast industry standard
    freed_plus, ///< FreeD+ / Stype extended-precision variant
    osc,        ///< OSC 1.0 UDP (schema: /camera/{id}/pan|tilt|roll|zoom|x|y|z)
    vrpn,       ///< VRPN pose tracker + analogue zoom channel
    psn,        ///< PosiStageNet v2.0 UDP multicast (stage/performer tracking)
    opentrackio,///< SMPTE RIS OSVP OpenTrackIO (JSON over UDP multicast)
};

/// Identifies which receiver a binding was created on (needed for clean release).
struct receiver_handle
{
    tracking_protocol protocol  = tracking_protocol::freed;
    int               port      = 6301;
    std::string       host; ///< VRPN: server URL; PSN: multicast group address
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

    // ------- Entrance-pupil (nodal) offset -------------------------
    // Lens-local offset (metres) of the optical entrance pupil from the tracked
    // origin. Expressed in the camera frame: forward (+along view), right, up.
    // Produces correct parallax as the camera rotates about an off-axis pupil.
    // All 0 = disabled (no position shift).
    double nodal_forward_m = 0.0;
    double nodal_right_m   = 0.0;
    double nodal_up_m      = 0.0;

    // ------- Rigid tracker→world alignment (xR survey) -------------
    // Maps the tracker's own coordinate frame to the previz/LED-wall world
    // frame, solved from a survey of known points (client-side Umeyama):
    //     world_metres = align_scale · R · tracker_millimetres + align_t
    // R is row-major (tracker→world); align_t is in metres; align_scale folds
    // the millimetre→metre unit change into the fit. Orientation is aligned by
    // composing R with the tracker rotation (see tracker_registry::inject_transform).
    // Only applied in mode_previz. align_enable = false leaves the legacy
    // per-axis offset / position_scale path untouched.
    bool   align_enable = false;
    double align_r[9]   = {1, 0, 0, 0, 1, 0, 0, 0, 1}; ///< row-major rotation, tracker→world
    double align_t[3]   = {0, 0, 0};                    ///< translation, metres
    double align_scale  = 1.0;                          ///< metres per millimetre

    // ------- Tracking latency compensation -------------------------
    // Delays the applied camera pose by this many milliseconds, interpolating
    // between buffered samples. Used to time-align tracking data with delayed
    // video (e.g. genlock/processing latency on the camera feed).
    // 0 = disabled (raw newest sample injected immediately, no buffering).
    double delay_ms = 0.0;

    // ------- Genlock / LTC frame-anchored latency compensation -----
    // Frame-native alternative to delay_ms: holds the pose back by genlock_frames
    // frames of the channel's frame rate, and — when a valid house LTC signal is
    // present — snaps the sampled time to the house frame grid so pose updates
    // align to video frame boundaries (true genlock behaviour).
    // Takes precedence over delay_ms when genlock_enable is true.
    bool   genlock_enable = false;
    double genlock_frames = 0.0;  ///< number of channel frames to delay the pose by
    double channel_fps    = 0.0;  ///< channel frame rate captured at bind (0 = unknown)

    // ------- Depth-of-field (faked rack-focus) ---------------------
    // Maps the decoded lens focus value to a blur radius (lens-bokeh type).
    // NON-PHYSICAL: there is no depth buffer for 2D/360 layers; this is an
    // operator-calibrated creative defocus driven by the focus channel.
    //   radius = max_radius * clamp((focus - near_raw) / (far_raw - near_raw), 0, 1)
    // dof_enable = false leaves the layer's blur untouched (manual MIXER BLUR wins).
    bool   dof_enable         = false;
    double dof_focus_near_raw = 0.0;     ///< focus raw value mapped to 0 blur (sharp)
    double dof_focus_far_raw  = 65535.0; ///< focus raw value mapped to max blur
    double dof_max_radius     = 0.0;     ///< blur radius at the far cal point

    // ------- Dynamic lens distortion profile -----------------------
    // Optional lens-calibration grid. When set, the binding samples it by
    // (zoom, focus) each packet to drive projection FOV + distortion k1..k3 /
    // p1/p2 + entrance-pupil forward offset. Null = disabled.
    std::shared_ptr<lens_profile> lens;

    // ------- Track-target virtual camera (mode_target) -------------
    // A static virtual camera used to project a tracked SUBJECT world position
    // (data.x/y/z, scaled mm→m by position_scale) into screen space so a graphic
    // FOLLOWS the subject. Only used when mode == mode_target.
    //   Camera position in metres; orientation in radians; fov is vertical (rad).
    //   The view convention matches previz_renderer: Rz(-roll)·Rx(-pitch)·Ry(-yaw)·T(-C).
    // target_enable = false (or an all-zero camera) makes mode_target a no-op.
    bool   target_enable   = false;  ///< mode_target writes no transform until a camera is set
    double target_cam_x    = 0.0;    ///< camera position X (metres)
    double target_cam_y    = 0.0;    ///< camera position Y (metres)
    double target_cam_z    = 0.0;    ///< camera position Z (metres)
    double target_cam_yaw  = 0.0;    ///< camera yaw   (radians)
    double target_cam_pitch= 0.0;    ///< camera pitch (radians)
    double target_cam_roll = 0.0;    ///< camera roll  (radians)
    double target_cam_fov  = M_PI / 2.0; ///< camera vertical FOV (radians, 90° default)
    double target_aspect   = 16.0 / 9.0; ///< frame aspect ratio (W/H) for horizontal NDC
    double target_gain     = 0.5;    ///< NDC[-1,1] → fill_translation (0.5: NDC edge → frame edge)
    double target_ref_dist_m = 0.0;  ///< 0 = no distance scaling; else fill_scale = ref/dist

    /// Tracks which receiver this binding was registered on (for reference counting).
    receiver_handle receiver;

    /// Previz mode callback: (x, y, z, yaw_deg, pitch_deg, roll_deg, fov_deg) → void.
    /// Set by the AMCP layer when MODE PREVIZ is selected.
    std::function<void(float, float, float, float, float, float, float)> previz_camera_fn;
};

}} // namespace caspar::tracking
