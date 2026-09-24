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

#include <atomic>
#include <chrono>
#include <deque>
#include <map>
#include <tuple>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
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
    void update_axes(int channel, int layer, const int src[3], const int sign[3]);
    void update_invert(int channel, int layer, bool pan, bool tilt, bool roll);

    /// A sample with the binding's AXES and INVERT applied -- the first thing done to every
    /// sample, so everything after it (and ZERO, which captures a reference) sees one frame.
    static camera_data remap(const tracker_binding& binding, const camera_data& data);

    // ----- Called by protocol receivers when a packet arrives -----------------

    /// Process a decoded camera_data packet and inject transforms into all
    /// matching bindings.
    void on_data(const camera_data& data);

    /// The same, STAMPED with the receiver it came from, so it drives only the bindings made
    /// on that receiver. Every receiver in this module calls this form.
    void on_data(camera_data data, tracking_protocol protocol, int port);

    // ----- Diagnostics --------------------------------------------------------

    /// Returns the latest received data for the given camera_id, if any.
    std::optional<camera_data> get_latest_data(int camera_id) const;

    /// The latest sample from ONE receiver for `camera_id` -- or, for the wildcard -1, the
    /// newest sample that receiver produced for any camera, which is what an `ANY` binding
    /// is driven by and so what its INFO and ZERO must read.
    std::optional<camera_data> get_latest_data(const receiver_handle& source, int camera_id) const;

    /// Stop the delay pump thread. Called at module uninit; idempotent.
    void stop_pump();

  private:
    tracker_registry() = default;
    ~tracker_registry();

    mutable std::mutex                               mutex_;
    std::map<std::pair<int, int>, tracker_binding>   bindings_;
    /// Keyed by (protocol, port, camera_id) -- a camera ID is only unique WITHIN a receiver.
    using source_key = std::tuple<int, int, int>;
    std::map<source_key, camera_data>                latest_data_;

    /// Per-camera_id time-ordered history of recent samples, used to reconstruct
    /// a delayed pose for latency compensation (see tracker_binding::delay_ms).
    std::map<source_key, std::deque<camera_data>>    sample_history_;

    void inject_transform(const tracker_binding& binding, const camera_data& data);

    // ── THE DELAY PUMP ──────────────────────────────────────────────────────────────────
    // A DELAY or GENLOCK binding shows the pose from `delay` ago, and until 2026-09-24 it was
    // only ever computed when a PACKET arrived -- so a tracker that sends on change, as most OSC
    // and many FreeD sources do when the camera is still, never landed its final pose: the last
    // packet injected the pose from `delay` before it, and nothing ever injected the last sample
    // itself. The pump re-injects the delayed pose on a clock between packets, and stops once the
    // newest sample has been shown (`settled`). Packet-driven injection is unchanged.
    struct pump_state
    {
        std::tuple<int, int, int> source;          ///< the source_key the binding last heard from
        bool                      settled = false; ///< the newest sample has been injected
    };
    std::map<std::pair<int, int>, pump_state> pump_;
    std::thread                               pump_thread_;
    std::atomic<bool>                         pump_run_{false};
    void                                      pump_loop();
};

}} // namespace caspar::tracking
