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

#include "tracker_registry.h"

#include "lens_profile.h"

#include "../ltc/ltc_input.h"

#include <core/frame/frame_transform.h>
#include <core/producer/stage.h>
#include <common/tweener.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace caspar { namespace tracking {

namespace {
// History retention bounds for the latency-compensation buffer.
constexpr int HISTORY_WINDOW_MS   = 2000; ///< Drop samples older than this (relative to newest).
constexpr int HISTORY_MAX_SAMPLES = 512;  ///< Hard cap on buffered samples per camera.

// --- Rigid tracker→world alignment helpers (mirror client tracker_align.py) ---
// Camera→world rotation convention matches previz_renderer:
//   R_cam = Ry(yaw)·Rx(pitch)·Rz(roll)   (view = Rz(-roll)·Rx(-pitch)·Ry(-yaw)·T(-C)).

/// Build R = Ry(yaw)·Rx(pitch)·Rz(roll) into a row-major 3×3 (radians).
inline void rot_ypr(double yaw, double pitch, double roll, double out[9])
{
    const double cy = std::cos(yaw),   sy = std::sin(yaw);
    const double cp = std::cos(pitch), sp = std::sin(pitch);
    const double cr = std::cos(roll),  sr = std::sin(roll);
    // Ry(yaw):  [ cy 0 sy ; 0 1 0 ; -sy 0 cy ]
    // Rx(pitch):[ 1 0 0 ; 0 cp -sp ; 0 sp cp ]
    // Rz(roll): [ cr -sr 0 ; sr cr 0 ; 0 0 1 ]
    // M = Rx·Rz:
    const double m00 = cr,        m01 = -sr,       m02 = 0.0;
    const double m10 = cp * sr,   m11 = cp * cr,   m12 = -sp;
    const double m20 = sp * sr,   m21 = sp * cr,   m22 = cp;
    // R = Ry·M:
    out[0] = cy * m00 + sy * m20; out[1] = cy * m01 + sy * m21; out[2] = cy * m02 + sy * m22;
    out[3] = m10;                 out[4] = m11;                 out[5] = m12;
    out[6] = -sy * m00 + cy * m20; out[7] = -sy * m01 + cy * m21; out[8] = -sy * m02 + cy * m22;
}

/// Row-major 3×3 product: out = a·b.
inline void mat3_mul(const double a[9], const double b[9], double out[9])
{
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            out[r * 3 + c] = a[r * 3 + 0] * b[0 * 3 + c]
                           + a[r * 3 + 1] * b[1 * 3 + c]
                           + a[r * 3 + 2] * b[2 * 3 + c];
}

/// Decompose R = Ry(yaw)·Rx(pitch)·Rz(roll) (row-major) → yaw/pitch/roll radians.
inline void euler_yxz(const double R[9], double& yaw, double& pitch, double& roll)
{
    double sp = -R[5]; // -R[1][2]
    sp = std::clamp(sp, -1.0, 1.0);
    pitch = std::asin(sp);
    const double cp = std::cos(pitch);
    if (std::abs(cp) > 1e-6) {
        yaw  = std::atan2(R[2], R[8]);  // atan2(R[0][2], R[2][2])
        roll = std::atan2(R[3], R[4]);  // atan2(R[1][0], R[1][1])
    } else {
        roll = 0.0;
        yaw  = std::atan2(-R[6], R[0]); // atan2(-R[2][0], R[0][0])
    }
}
} // namespace

tracker_registry& tracker_registry::instance()
{
    static tracker_registry inst;
    return inst;
}

// ---- Binding management ----------------------------------------------------

void tracker_registry::bind(int channel, int layer, tracker_binding binding)
{
    std::lock_guard<std::mutex> lk(mutex_);
    bindings_[{channel, layer}] = std::move(binding);
}

void tracker_registry::unbind(int channel, int layer)
{
    std::lock_guard<std::mutex> lk(mutex_);
    bindings_.erase({channel, layer});
}

bool tracker_registry::has_binding(int channel, int layer) const
{
    std::lock_guard<std::mutex> lk(mutex_);
    return bindings_.count({channel, layer}) > 0;
}

std::optional<tracker_binding> tracker_registry::get_binding(int channel, int layer) const
{
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = bindings_.find({channel, layer});
    if (it == bindings_.end())
        return std::nullopt;
    return it->second;
}

std::vector<std::pair<std::pair<int, int>, tracker_binding>> tracker_registry::get_all_bindings() const
{
    std::lock_guard<std::mutex> lk(mutex_);
    return std::vector<std::pair<std::pair<int, int>, tracker_binding>>(bindings_.begin(), bindings_.end());
}

void tracker_registry::update_offset(int channel, int layer, double pan_rad, double tilt_rad, double roll_rad)
{
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = bindings_.find({channel, layer});
    if (it == bindings_.end())
        throw std::runtime_error("No binding at channel/layer");
    it->second.pan_offset  = pan_rad;
    it->second.tilt_offset = tilt_rad;
    it->second.roll_offset = roll_rad;
}

void tracker_registry::update_scale(int channel, int layer, double pan_scale, double tilt_scale, double zoom_full_range)
{
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = bindings_.find({channel, layer});
    if (it == bindings_.end())
        throw std::runtime_error("No binding at channel/layer");
    it->second.pan_scale       = pan_scale;
    it->second.tilt_scale      = tilt_scale;
    it->second.zoom_full_range = zoom_full_range;
}

void tracker_registry::update_zoom_default_fov(int channel, int layer, double fov_rad)
{
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = bindings_.find({channel, layer});
    if (it == bindings_.end())
        throw std::runtime_error("No binding at channel/layer");
    it->second.zoom_default_fov = fov_rad;
}

void tracker_registry::update_zoom_lut(int channel, int layer, std::vector<zoom_entry> lut)
{
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = bindings_.find({channel, layer});
    if (it == bindings_.end())
        throw std::runtime_error("No binding at channel/layer");
    // Keep the table sorted ascending by raw_value so compute_fov() can bracket.
    std::sort(lut.begin(), lut.end(),
              [](const zoom_entry& a, const zoom_entry& b) { return a.raw_value < b.raw_value; });
    it->second.zoom_lookup = std::move(lut);
}

void tracker_registry::update_position_scale(int channel, int layer, double scale)
{
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = bindings_.find({channel, layer});
    if (it == bindings_.end())
        throw std::runtime_error("No binding at channel/layer");
    it->second.position_scale = scale;
}

void tracker_registry::update_delay(int channel, int layer, double delay_ms)
{
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = bindings_.find({channel, layer});
    if (it == bindings_.end())
        throw std::runtime_error("No binding at channel/layer");
    it->second.delay_ms = delay_ms;
}

void tracker_registry::update_genlock(int channel, int layer, bool enable, double frames)
{
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = bindings_.find({channel, layer});
    if (it == bindings_.end())
        throw std::runtime_error("No binding at channel/layer");
    it->second.genlock_enable = enable;
    it->second.genlock_frames = frames;
}

void tracker_registry::update_nodal(int channel, int layer, double forward_m, double right_m, double up_m)
{
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = bindings_.find({channel, layer});
    if (it == bindings_.end())
        throw std::runtime_error("No binding at channel/layer");
    it->second.nodal_forward_m = forward_m;
    it->second.nodal_right_m   = right_m;
    it->second.nodal_up_m      = up_m;
}

void tracker_registry::update_world_align(
    int channel, int layer, bool enable, const double r[9], const double t[3], double scale)
{
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = bindings_.find({channel, layer});
    if (it == bindings_.end())
        throw std::runtime_error("No binding at channel/layer");
    it->second.align_enable = enable;
    if (enable) {
        for (int i = 0; i < 9; ++i)
            it->second.align_r[i] = r[i];
        for (int i = 0; i < 3; ++i)
            it->second.align_t[i] = t[i];
        it->second.align_scale = scale;
    }
}

void tracker_registry::update_dof(int channel, int layer, bool enable, double near_raw, double far_raw, double max_radius)
{
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = bindings_.find({channel, layer});
    if (it == bindings_.end())
        throw std::runtime_error("No binding at channel/layer");
    it->second.dof_enable         = enable;
    it->second.dof_focus_near_raw = near_raw;
    it->second.dof_focus_far_raw  = far_raw;
    it->second.dof_max_radius     = max_radius;
}

void tracker_registry::update_lens(int channel, int layer, std::shared_ptr<lens_profile> lens)
{
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = bindings_.find({channel, layer});
    if (it == bindings_.end())
        throw std::runtime_error("No binding at channel/layer");
    it->second.lens = std::move(lens);
}

void tracker_registry::update_target_camera(int channel, int layer, bool enable,
                                            double x, double y, double z,
                                            double yaw_rad, double pitch_rad, double roll_rad, double fov_rad)
{
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = bindings_.find({channel, layer});
    if (it == bindings_.end())
        throw std::runtime_error("No binding at channel/layer");
    it->second.target_enable    = enable;
    it->second.target_cam_x     = x;
    it->second.target_cam_y     = y;
    it->second.target_cam_z     = z;
    it->second.target_cam_yaw   = yaw_rad;
    it->second.target_cam_pitch = pitch_rad;
    it->second.target_cam_roll  = roll_rad;
    it->second.target_cam_fov   = fov_rad;
}

void tracker_registry::update_target_map(int channel, int layer, double gain, double ref_dist_m, double aspect)
{
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = bindings_.find({channel, layer});
    if (it == bindings_.end())
        throw std::runtime_error("No binding at channel/layer");
    it->second.target_gain       = gain;
    it->second.target_ref_dist_m = ref_dist_m;
    if (aspect > 0.0)
        it->second.target_aspect = aspect;
}

// ---- Data injection --------------------------------------------------------

/// Compute effective FOV in radians from a raw zoom value and the binding params.
static double compute_fov(const tracker_binding& b, uint16_t zoom_raw)
{
    if (!b.zoom_lookup.empty()) {
        const auto& lut = b.zoom_lookup;
        if (zoom_raw <= lut.front().raw_value)
            return lut.front().fov_rad;
        if (zoom_raw >= lut.back().raw_value)
            return lut.back().fov_rad;

        auto it = std::lower_bound(lut.begin(), lut.end(), zoom_raw,
                                   [](const zoom_entry& e, uint16_t v) { return e.raw_value < v; });
        const auto& hi = *it;
        const auto& lo = *std::prev(it);
        double t       = double(zoom_raw - lo.raw_value) / double(hi.raw_value - lo.raw_value);
        return lo.fov_rad + t * (hi.fov_rad - lo.fov_rad);
    }

    // Realistic lens formula: FOV(z) = 2*atan(tan(fov_wide/2) * full_range / z)
    // At zoom_raw == zoom_full_range → FOV == zoom_default_fov (widest angle baseline).
    // As zoom_raw decreases towards 0 → FOV increases (wider).
    // As zoom_raw increases beyond full_range → FOV narrows (telephoto).
    // Note: many vendors send 0 = widest. Guard against division by zero.
    const double z = (zoom_raw < 1) ? 1.0 : static_cast<double>(zoom_raw);
    return 2.0 * std::atan(std::tan(b.zoom_default_fov * 0.5) * b.zoom_full_range / z);
}

/// Shortest-path angular interpolation (handles wrap-around at ±π).
static double lerp_angle(double a, double b, double t)
{
    double diff = b - a;
    while (diff > M_PI)
        diff -= 2.0 * M_PI;
    while (diff < -M_PI)
        diff += 2.0 * M_PI;
    return a + diff * t;
}

/// Reconstruct a camera pose at target_time by interpolating between buffered
/// samples. `history` must be ordered oldest→newest by timestamp.
/// Clamps to the oldest/newest sample when target_time falls outside the buffer.
static camera_data interpolate_history(const std::deque<camera_data>&        history,
                                       std::chrono::steady_clock::time_point target_time)
{
    if (history.empty())
        return camera_data{};
    if (history.size() == 1 || target_time <= history.front().timestamp)
        return history.front();
    if (target_time >= history.back().timestamp)
        return history.back();

    // First sample with timestamp >= target_time.
    auto hi = std::lower_bound(history.begin(),
                               history.end(),
                               target_time,
                               [](const camera_data& s, std::chrono::steady_clock::time_point t) {
                                   return s.timestamp < t;
                               });
    auto lo = std::prev(hi);

    const double span = std::chrono::duration<double>(hi->timestamp - lo->timestamp).count();
    const double t    = (span > 0.0)
                         ? std::chrono::duration<double>(target_time - lo->timestamp).count() / span
                         : 0.0;

    camera_data out;
    out.camera_id = lo->camera_id;
    out.pan       = lerp_angle(lo->pan, hi->pan, t);
    out.tilt      = lerp_angle(lo->tilt, hi->tilt, t);
    out.roll      = lerp_angle(lo->roll, hi->roll, t);
    out.x         = lo->x + (hi->x - lo->x) * t;
    out.y         = lo->y + (hi->y - lo->y) * t;
    out.z         = lo->z + (hi->z - lo->z) * t;
    out.zoom      = static_cast<uint16_t>(std::lround(lo->zoom + (static_cast<double>(hi->zoom) - lo->zoom) * t));
    out.focus     = static_cast<uint16_t>(std::lround(lo->focus + (static_cast<double>(hi->focus) - lo->focus) * t));
    out.iris      = static_cast<uint16_t>(std::lround(lo->iris + (static_cast<double>(hi->iris) - lo->iris) * t));
    out.timestamp = target_time;
    return out;
}

void tracker_registry::inject_transform(const tracker_binding& binding, const camera_data& data)
{
    auto stage = binding.stage.lock();
    if (!stage)
        return; // channel has been destroyed

    const int    layer     = binding.layer_index;
    const double pan       = data.pan * binding.pan_scale + binding.pan_offset;
    const double tilt      = data.tilt * binding.tilt_scale + binding.tilt_offset;
    const double roll      = data.roll + binding.roll_offset;

    // Dynamic lens profile: sample distortion / FOV / nodal forward by (zoom, focus).
    lens_sample lens{};
    const bool  has_lens = static_cast<bool>(binding.lens);
    if (has_lens)
        lens = binding.lens->sample(static_cast<double>(data.zoom), static_cast<double>(data.focus));

    const double fov = (has_lens && lens.fov_rad > 0.0) ? lens.fov_rad : compute_fov(binding, data.zoom);

    // Entrance-pupil (nodal) offset: shift the tracked position by the lens-local
    // nodal vector expressed in world space, so an off-axis pupil produces correct
    // parallax as the camera rotates. Offsets are in metres; positions in mm.
    // The lens profile's forward offset augments any manual NODAL forward setting.
    const double nodal_fwd_m = binding.nodal_forward_m + (has_lens ? lens.nodal_forward_m : 0.0);
    double pos_x = data.x;
    double pos_y = data.y;
    double pos_z = data.z;
    if (nodal_fwd_m != 0.0 || binding.nodal_right_m != 0.0 || binding.nodal_up_m != 0.0) {
        const double cp = std::cos(pan),  sp = std::sin(pan);
        const double ct = std::cos(tilt), st = std::sin(tilt);
        // Camera-frame basis (right-handed: at rest forward=+Z, right=+X, up=+Y).
        const double fwd[3]   = {sp * ct, st, cp * ct};
        const double right[3] = {cp, 0.0, -sp};
        const double up[3]    = {-sp * st, ct, -cp * st};
        const double nf       = nodal_fwd_m * 1000.0; // m → mm
        const double nr       = binding.nodal_right_m * 1000.0;
        const double nu       = binding.nodal_up_m * 1000.0;
        pos_x += nf * fwd[0] + nr * right[0] + nu * up[0];
        pos_y += nf * fwd[1] + nr * right[1] + nu * up[1];
        pos_z += nf * fwd[2] + nr * right[2] + nu * up[2];
    }

    const double offset_x  = pos_x * binding.position_scale;
    const double offset_y  = pos_y * binding.position_scale;

    // Faked depth-of-field: map the decoded focus value to a lens-bokeh blur radius.
    const bool   dof_on     = binding.dof_enable;
    double       dof_radius = 0.0;
    if (dof_on) {
        const double span = binding.dof_focus_far_raw - binding.dof_focus_near_raw;
        const double t    = (span != 0.0)
                             ? (static_cast<double>(data.focus) - binding.dof_focus_near_raw) / span
                             : 0.0;
        dof_radius = binding.dof_max_radius * std::clamp(t, 0.0, 1.0);
    }

    if (binding.mode == tracking_mode::mode_360) {
        // 360° equirectangular: set yaw / pitch / roll / fov on the projection struct.
        // X→offset_x and Y→offset_y provide horizontal/vertical lens-shift for
        // camera parallax correction inside the sphere.
        stage->apply_transform(
            layer,
            [pan, tilt, roll, fov, offset_x, offset_y, dof_on, dof_radius, has_lens, lens](core::frame_transform t) -> core::frame_transform {
                t.image_transform.projection.enable   = (fov > 0.0);
                t.image_transform.projection.yaw      = pan;
                t.image_transform.projection.pitch    = tilt;
                t.image_transform.projection.roll     = roll;
                t.image_transform.projection.fov      = fov;
                t.image_transform.projection.offset_x = offset_x;
                t.image_transform.projection.offset_y = offset_y;
                if (has_lens) {
                    // Lens profile drives the Brown-Conrady distortion coefficients.
                    t.image_transform.projection.lens_k1 = lens.k1;
                    t.image_transform.projection.lens_k2 = lens.k2;
                    t.image_transform.projection.lens_k3 = lens.k3;
                    t.image_transform.projection.lens_p1 = lens.p1;
                    t.image_transform.projection.lens_p2 = lens.p2;
                }
                if (dof_on) {
                    t.image_transform.blur.enable = dof_radius > 0.0;
                    t.image_transform.blur.radius = dof_radius;
                    t.image_transform.blur.type   = core::blur_type::lens;
                }
                return t;
            },
            0,
            tweener(L"linear"));
    } else if (binding.mode == tracking_mode::mode_previz) {
        // Previz: drive the previz renderer's virtual camera.
        // Angles are in radians from the tracker; convert to degrees for the previz API.
        // Positions are in mm from the tracker; convert to metres for the previz API.
        if (binding.previz_camera_fn) {
            double cam_x_m, cam_y_m, cam_z_m;
            double cam_yaw = pan, cam_pitch = tilt, cam_roll = roll;

            if (binding.align_enable) {
                // Rigid tracker→world alignment (survey). Position uses the raw
                // tracker millimetres directly (align_scale folds mm→m):
                //   world_m = align_scale · R_align · (pos_x,pos_y,pos_z) + align_t
                const double* R = binding.align_r;
                const double  s = binding.align_scale;
                cam_x_m = s * (R[0] * pos_x + R[1] * pos_y + R[2] * pos_z) + binding.align_t[0];
                cam_y_m = s * (R[3] * pos_x + R[4] * pos_y + R[5] * pos_z) + binding.align_t[1];
                cam_z_m = s * (R[6] * pos_x + R[7] * pos_y + R[8] * pos_z) + binding.align_t[2];

                // Orientation: camera→world = R_align · R_track, decomposed back
                // into the previz yaw/pitch/roll convention.
                double R_track[9];
                rot_ypr(pan, tilt, roll, R_track);
                double R_world[9];
                mat3_mul(R, R_track, R_world);
                euler_yxz(R_world, cam_yaw, cam_pitch, cam_roll);
            } else {
                cam_x_m = pos_x * binding.position_scale;
                cam_y_m = pos_y * binding.position_scale;
                cam_z_m = pos_z * binding.position_scale;
            }

            binding.previz_camera_fn(
                static_cast<float>(cam_x_m),
                static_cast<float>(cam_y_m),
                static_cast<float>(cam_z_m),
                static_cast<float>(cam_yaw   * 180.0 / M_PI),   // rad → degrees
                static_cast<float>(cam_pitch * 180.0 / M_PI),
                static_cast<float>(cam_roll  * 180.0 / M_PI),
                static_cast<float>(fov   * 180.0 / M_PI));
        }
    } else if (binding.mode == tracking_mode::mode_target) {
        // Track-target: project the tracked SUBJECT world position through a static
        // virtual camera to drive this layer's screen position (AR follow). The
        // subject position is data.x/y/z scaled mm→m by position_scale; orientation
        // and zoom from the tracker are intentionally ignored here.
        // target_enable == false (no camera configured) writes no transform.
        if (binding.target_enable) {
            // Subject world position (metres).
            const double sx = data.x * binding.position_scale;
            const double sy = data.y * binding.position_scale;
            const double sz = data.z * binding.position_scale;
            // Vector from camera to subject (world).
            const double dx = sx - binding.target_cam_x;
            const double dy = sy - binding.target_cam_y;
            const double dz = sz - binding.target_cam_z;
            // World→view: apply inverse camera rotation Rz(-roll)·Rx(-pitch)·Ry(-yaw).
            const double cy = std::cos(-binding.target_cam_yaw),   sy_ = std::sin(-binding.target_cam_yaw);
            const double cp = std::cos(-binding.target_cam_pitch), sp_ = std::sin(-binding.target_cam_pitch);
            const double cr = std::cos(-binding.target_cam_roll),  sr_ = std::sin(-binding.target_cam_roll);
            // v1 = Ry(-yaw)·d
            const double v1x =  cy * dx + sy_ * dz;
            const double v1y =  dy;
            const double v1z = -sy_ * dx + cy * dz;
            // v2 = Rx(-pitch)·v1
            const double v2x =  v1x;
            const double v2y =  cp * v1y - sp_ * v1z;
            const double v2z =  sp_ * v1y + cp * v1z;
            // v3 = Rz(-roll)·v2
            const double vx =  cr * v2x - sr_ * v2y;
            const double vy =  sr_ * v2x + cr * v2y;
            const double vz =  v2z;

            constexpr double kNear = 1e-3;
            const bool visible = vz > kNear;
            double fill_x = 0.0, fill_y = 0.0, t_scale = 1.0;
            if (visible) {
                const double f      = 1.0 / std::tan(binding.target_cam_fov * 0.5);
                const double ndc_x  = (vx / vz) * f / binding.target_aspect;
                const double ndc_y  = (vy / vz) * f;
                fill_x =  ndc_x * binding.target_gain;
                fill_y = -ndc_y * binding.target_gain; // screen Y is down
                t_scale = (binding.target_ref_dist_m > 0.0) ? (binding.target_ref_dist_m / vz) : 1.0;
            }
            stage->apply_transform(
                layer,
                [visible, fill_x, fill_y, t_scale](core::frame_transform t) -> core::frame_transform {
                    t.image_transform.enable_geometry_modifiers = true;
                    // Behind the camera → hide via opacity rather than smearing across frame.
                    t.image_transform.opacity = visible ? 1.0 : 0.0;
                    if (visible) {
                        t.image_transform.fill_translation[0] = fill_x;
                        t.image_transform.fill_translation[1] = fill_y;
                        t.image_transform.fill_scale[0]       = t_scale;
                        t.image_transform.fill_scale[1]       = t_scale;
                    }
                    return t;
                },
                0,
                tweener(L"linear"));
        }
    } else {
        // 2D mode: pan → fill_translation X, tilt → fill_translation Y (inverted so up = up),
        // roll → angle (radians), zoom → fill_scale (uniform).
        // Physical X/Y are added on top of the pan/tilt translation as position parallax:
        //   camera moves right (+X) → content shifts right (+fill_translation.x)
        //   camera moves up   (+Y) → camera Y up = same sign as tilt up → -fill_translation.y
        const double scale = (fov > 0.0) ? (binding.zoom_default_fov / fov) : 1.0;
        stage->apply_transform(
            layer,
            [pan, tilt, roll, scale, offset_x, offset_y, dof_on, dof_radius](core::frame_transform t) -> core::frame_transform {
                t.image_transform.enable_geometry_modifiers = true;
                t.image_transform.fill_translation[0]       = pan + offset_x;
                t.image_transform.fill_translation[1]       = -tilt - offset_y;
                t.image_transform.angle                     = roll;
                t.image_transform.fill_scale[0]             = scale;
                t.image_transform.fill_scale[1]             = scale;
                if (dof_on) {
                    t.image_transform.blur.enable = dof_radius > 0.0;
                    t.image_transform.blur.radius = dof_radius;
                    t.image_transform.blur.type   = core::blur_type::lens;
                }
                return t;
            },
            0,
            tweener(L"linear"));
    }
}

void tracker_registry::on_data(const camera_data& data)
{
    std::vector<tracker_binding> matched;
    std::deque<camera_data>      history_snapshot;
    bool                         any_delay = false;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        latest_data_[data.camera_id] = data;

        // Append to per-camera history and trim to the retention window/cap.
        auto& hist = sample_history_[data.camera_id];
        hist.push_back(data);
        const auto cutoff = data.timestamp - std::chrono::milliseconds(HISTORY_WINDOW_MS);
        while (hist.size() > 1 && hist.front().timestamp < cutoff)
            hist.pop_front();
        while (hist.size() > HISTORY_MAX_SAMPLES)
            hist.pop_front();

        for (auto& [key, binding] : bindings_) {
            if (binding.camera_id == -1 || binding.camera_id == data.camera_id) {
                matched.push_back(binding);
                if (binding.delay_ms > 0.0 || binding.genlock_enable)
                    any_delay = true;
            }
        }

        if (any_delay)
            history_snapshot = hist; // copy for lock-free interpolation below
    }

    for (auto& binding : matched) {
        if (binding.genlock_enable && binding.channel_fps > 0.0) {
            // Frame-native genlock: hold the pose back by N channel frames, and
            // snap the sampled time to the house frame grid when LTC is valid so
            // pose updates align to video frame boundaries.
            const double fps = binding.channel_fps;
            auto to_dur = [](double seconds) {
                return std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                    std::chrono::duration<double>(seconds));
            };
            auto target = data.timestamp - to_dur(binding.genlock_frames / fps);

            uint32_t                              ltc_frame = 0;
            std::chrono::steady_clock::time_point ltc_time;
            if (ltc::LTCInput::instance().get_timecode_anchor(
                    static_cast<int>(std::lround(fps)), ltc_frame, ltc_time)) {
                const double rel    = std::chrono::duration<double>(target - ltc_time).count();
                const double k      = std::round(static_cast<double>(ltc_frame) + rel * fps);
                target = ltc_time + to_dur((k - static_cast<double>(ltc_frame)) / fps);
            }
            inject_transform(binding, interpolate_history(history_snapshot, target));
        } else if (binding.delay_ms > 0.0) {
            const auto target =
                data.timestamp - std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                                     std::chrono::duration<double, std::milli>(binding.delay_ms));
            inject_transform(binding, interpolate_history(history_snapshot, target));
        } else {
            inject_transform(binding, data);
        }
    }
}

std::optional<camera_data> tracker_registry::get_latest_data(int camera_id) const
{
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = latest_data_.find(camera_id);
    if (it == latest_data_.end())
        return std::nullopt;
    return it->second;
}

}} // namespace caspar::tracking
