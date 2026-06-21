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

#include <memory>
#include <string>
#include <vector>

namespace caspar { namespace tracking {

/// A single lens-calibration entry: distortion + field-of-view + nodal point
/// at one (zoom, focus) grid node. Compatible with the OpenLensIO distortion
/// model (Brown-Conrady radial k1..k3 + tangential p1/p2).
struct lens_sample
{
    double k1              = 0.0; ///< radial distortion coefficient
    double k2              = 0.0;
    double k3              = 0.0;
    double p1              = 0.0; ///< tangential (decentering) coefficient
    double p2              = 0.0;
    double fov_rad         = 0.0; ///< horizontal field of view (radians); 0 = leave unchanged
    double nodal_forward_m = 0.0; ///< entrance-pupil forward offset (metres)
};

/// Lens-calibration profile: a 1-D (zoom) or 2-D (zoom × focus) grid of
/// lens_samples, queried with bilinear interpolation. Loaded from a JSON file.
///
/// JSON schema:
///   {
///     "name": "Canon CN7x17",
///     "fov_unit": "deg",                 // "deg" (default) or "rad"
///     "zoom_axis":  [0, 16384, 32768, 49152, 65535],
///     "focus_axis": [0, 65535],          // optional; omit for a 1-D (zoom-only) profile
///     "samples": [                       // 2-D: rows = zoom_axis, cols = focus_axis
///       [ {"k1":-0.02,"fov":4.5,"nodal_forward_m":0.05}, { ... } ],
///       ...
///     ]
///     // For a 1-D profile, "samples" is a flat array (one entry per zoom node).
///   }
class lens_profile
{
  public:
    /// Loads and validates a profile from a JSON file. Throws std::runtime_error
    /// on malformed input. Never returns null on success.
    static std::shared_ptr<lens_profile> load(const std::string& path);

    const std::string& name() const { return name_; }
    const std::string& path() const { return path_; }

    /// Bilinearly interpolates a lens_sample for the given raw zoom and focus
    /// values, clamping to the calibration range at the edges.
    lens_sample sample(double zoom_raw, double focus_raw) const;

  private:
    std::string                            name_;
    std::string                            path_;
    std::vector<double>                    zoom_axis_;
    std::vector<double>                    focus_axis_;  ///< always size >= 1 (1-D uses a single node)
    std::vector<std::vector<lens_sample>>  grid_;        ///< grid_[zoom_index][focus_index]
};

}} // namespace caspar::tracking
