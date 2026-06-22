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
#include "tracker_binding.h"

#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace caspar { namespace tracking {

/// Thread-safe registry that holds all per-layer tracking bindings and serves
/// as the single injection point for every protocol receiver.
class tracker_registry
{
  public:
    static tracker_registry& instance();

    // ----- Binding management (typically called from AMCP commands) -----------

    void bind(int channel, int layer, tracker_binding binding);
    void unbind(int channel, int layer);
    bool has_binding(int channel, int layer) const;
    std::optional<tracker_binding> get_binding(int channel, int layer) const;
    std::vector<std::pair<std::pair<int, int>, tracker_binding>> get_all_bindings() const;

    void update_offset(int channel, int layer, double pan_rad, double tilt_rad, double roll_rad);
    void update_scale(int channel, int layer, double pan_scale, double tilt_scale, double zoom_full_range);
    void update_zoom_default_fov(int channel, int layer, double fov_rad);
    void update_zoom_lut(int channel, int layer, std::vector<zoom_entry> lut);
    void update_position_scale(int channel, int layer, double scale);
    void update_delay(int channel, int layer, double delay_ms);
    void update_genlock(int channel, int layer, bool enable, double frames);
    void update_nodal(int channel, int layer, double forward_m, double right_m, double up_m);
    void update_world_align(int channel, int layer, bool enable, const double r[9], const double t[3], double scale);
    void update_dof(int channel, int layer, bool enable, double near_raw, double far_raw, double max_radius);
    void update_lens(int channel, int layer, std::shared_ptr<lens_profile> lens);
    void update_target_camera(int channel, int layer, bool enable,
                              double x, double y, double z,
                              double yaw_rad, double pitch_rad, double roll_rad, double fov_rad);
    void update_target_map(int channel, int layer, double gain, double ref_dist_m, double aspect);

    // ----- Called by protocol receivers when a packet arrives -----------------

    /// Process a decoded camera_data packet and inject transforms into all
    /// matching bindings.
    void on_data(const camera_data& data);

    // ----- Diagnostics --------------------------------------------------------

    /// Returns the latest received data for the given camera_id, if any.
    std::optional<camera_data> get_latest_data(int camera_id) const;

  private:
    tracker_registry()  = default;
    ~tracker_registry() = default;

    mutable std::mutex                               mutex_;
    std::map<std::pair<int, int>, tracker_binding>   bindings_;
    std::map<int, camera_data>                       latest_data_;

    /// Per-camera_id time-ordered history of recent samples, used to reconstruct
    /// a delayed pose for latency compensation (see tracker_binding::delay_ms).
    std::map<int, std::deque<camera_data>>           sample_history_;

    void inject_transform(const tracker_binding& binding, const camera_data& data);
};

}} // namespace caspar::tracking
