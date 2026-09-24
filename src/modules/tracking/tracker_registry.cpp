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

void tracker_registry::update_axes(int channel, int layer, const int src[3], const int sign[3])
{
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = bindings_.find({channel, layer});
    if (it == bindings_.end())
        throw std::runtime_error("No binding at channel/layer");
    for (int i = 0; i < 3; ++i) {
        it->second.axis_src[i]  = src[i];
        it->second.axis_sign[i] = sign[i];
    }
}

void tracker_registry::update_invert(int channel, int layer, bool pan, bool tilt, bool roll)
{
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = bindings_.find({channel, layer});
    if (it == bindings_.end())
        throw std::runtime_error("No binding at channel/layer");
    it->second.invert_pan  = pan;
    it->second.invert_tilt = tilt;
    it->second.invert_roll = roll;
}

camera_data tracker_registry::remap(const tracker_binding& binding, const camera_data& data)
{
    camera_data out = data;
    const double src[3] = {data.x, data.y, data.z};
    out.x = src[binding.axis_src[0]] * binding.axis_sign[0];
    out.y = src[binding.axis_src[1]] * binding.axis_sign[1];
    out.z = src[binding.axis_src[2]] * binding.axis_sign[2];
    if (binding.invert_pan)
        out.pan = -data.pan;
    if (binding.invert_tilt)
        out.tilt = -data.tilt;
    if (binding.invert_roll)
        out.roll = -data.roll;
    return out;
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

// DOF OFF and LENS CLEAR TAKE WHAT THEY WROTE BACK OFF THE LAYER. Both only stop WRITING, and
// until 2026-09-24 that was all they did: the last tracking-driven blur, and the last profile's
// distortion coefficients, stayed on the layer's transform for as long as nothing else touched
// it -- so turning DOF off left the picture exactly as soft as it was. One write clears them, and
// only in the modes that set them (blur: 360 and 2D; distortion: 360), so an operator's own
// MIXER BLUR or PROJECTION_DISTORTION on a layer in another mode is not touched.
void tracker_registry::update_dof(int channel, int layer, bool enable, double near_raw, double far_raw, double max_radius)
{
    std::weak_ptr<core::stage_base> stage;
    bool                            turned_off = false;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        auto it = bindings_.find({channel, layer});
        if (it == bindings_.end())
            throw std::runtime_error("No binding at channel/layer");
        turned_off = it->second.dof_enable && !enable &&
                     (it->second.mode == tracking_mode::mode_360 || it->second.mode == tracking_mode::mode_2d);
        it->second.dof_enable         = enable;
        it->second.dof_focus_near_raw = near_raw;
        it->second.dof_focus_far_raw  = far_raw;
        it->second.dof_max_radius     = max_radius;
        stage                         = it->second.stage;
    }
    if (turned_off)
        if (auto s = stage.lock())
            s->apply_transform(
                layer,
                [](core::frame_transform t) {
                    t.image_transform.blur.enable = false;
                    t.image_transform.blur.radius = 0.0;
                    return t;
                },
                0,
                tweener(L"linear"));
}

void tracker_registry::update_lens(int channel, int layer, std::shared_ptr<lens_profile> lens)
{
    std::weak_ptr<core::stage_base> stage;
    bool                            cleared = false;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        auto it = bindings_.find({channel, layer});
        if (it == bindings_.end())
            throw std::runtime_error("No binding at channel/layer");
        cleared            = it->second.lens && !lens && it->second.mode == tracking_mode::mode_360;
        it->second.lens    = std::move(lens);
        stage              = it->second.stage;
    }
    if (cleared)
        if (auto s = stage.lock())
            s->apply_transform(
                layer,
                [](core::frame_transform t) {
                    auto& p   = t.image_transform.projection;
                    p.lens_k1 = p.lens_k2 = p.lens_k3 = p.lens_p1 = p.lens_p2 = 0.0;
                    return t;
                },
                0,
                tweener(L"linear"));
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
    // a/b come from wire-supplied pose data (OSC/OpenTrackIO); the previous
    // while-loop normalization could spin proportionally to |diff|/2π on a
    // garbage or absurd value (e.g. 1e300), hanging the receiver thread
    // indefinitely. std::remainder normalizes in one step regardless of
    // magnitude, and is exact for finite inputs.
    double diff = std::remainder(b - a, 2.0 * M_PI);
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

void tracker_registry::inject_transform(const tracker_binding& binding, const camera_data& raw)
{
    auto stage = binding.stage.lock();
    if (!stage)
        return; // channel has been destroyed

    // AXES and INVERT first: everything below is written in camera_data's frame.
    const camera_data data = remap(binding, raw);

    const int    layer     = binding.layer_index;
    const double pan       = data.pan * binding.pan_scale + binding.pan_offset;
    const double tilt      = data.tilt * binding.tilt_scale + binding.tilt_offset;
    const double roll      = data.roll + binding.roll_offset;

    // Dynamic lens profile: sample distortion / FOV / nodal forward by (zoom, focus).
    lens_sample lens{};
    const bool  has_lens = static_cast<bool>(binding.lens);
    if (has_lens)
        lens = binding.lens->sample(static_cast<double>(data.zoom), static_cast<double>(data.focus));

    // A lens profile's fov is HORIZONTAL unless it says otherwise, and every consumer below takes a
    // VERTICAL one -- the rectilinear 360 projection (latitude = atan(ndc.y tan(fov/2))) and the
    // previz camera. Until 2026-09-24 the horizontal value went straight through, so a 16:9
    // channel drew 1.78x the vertical angle the lens has. DEFAULT_FOV and ZOOM_LUT are vertical.
    double fov = compute_fov(binding, data.zoom);
    if (has_lens && lens.fov_rad > 0.0) {
        fov = lens.fov_rad;
        if (binding.lens->fov_horizontal() && binding.channel_aspect > 0.0)
            fov = 2.0 * std::atan(std::tan(fov * 0.5) / binding.channel_aspect);
    }

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
        // Camera-frame basis in camera_data's frame -- LEFT-handed: at rest forward=+Z, right=+X,
        // up=+Y (this comment said right-handed, which that triple is not). mode_previz converts
        // the result into the previz world's right-handed frame afterwards, with the position.
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
            // FROM camera_data's frame INTO the previz world's. camera_data is LEFT-handed
            // (x right, y up, z towards the subject; pan right, tilt up, roll clockwise from
            // behind are positive). The previz world is RIGHT-handed, its camera looks along
            // -Z, and R_cam = Ry(yaw)·Rx(pitch)·Rz(roll) (previz_renderer; `rot_ypr` above) --
            // under which a positive YAW turns the camera LEFT and a positive ROLL is
            // anticlockwise from behind. So the forward axis and those two angles change sign:
            //
            //     (x, y, z) -> (x, y, -z)      yaw = -pan    pitch = tilt    roll = -roll
            //
            // Checked by forward vectors rather than by sign: camera_data forward for a pan p is
            // (sin p, 0, cos p); previz forward Ry(-p)·(0,0,-1) is (sin p, 0, -cos p) -- the same
            // direction with z mirrored, as the position is. Until 2026-09-24 all six passed
            // through unchanged, so a tracked camera that panned right turned left in previz,
            // moved backwards when it dollied in, and rolled the wrong way.
            //
            // The conversion comes BEFORE the survey alignment: a mirror is not a rotation, so a
            // survey fitted from the left-handed frame straight to the right-handed one (Umeyama
            // returns det +1) could never fit, only be least-squared towards the reflection.
            const double rh_x = pos_x, rh_y = pos_y, rh_z = -pos_z;
            const double rh_yaw = -pan, rh_pitch = tilt, rh_roll = -roll;

            double cam_x_m, cam_y_m, cam_z_m;
            double cam_yaw = rh_yaw, cam_pitch = rh_pitch, cam_roll = rh_roll;

            if (binding.align_enable) {
                // Rigid tracker→world alignment (survey), in the RIGHT-handed frame above.
                // align_scale folds mm→m:
                //   world_m = align_scale · R_align · (x, y, -z) + align_t
                const double* R = binding.align_r;
                const double  s = binding.align_scale;
                cam_x_m = s * (R[0] * rh_x + R[1] * rh_y + R[2] * rh_z) + binding.align_t[0];
                cam_y_m = s * (R[3] * rh_x + R[4] * rh_y + R[5] * rh_z) + binding.align_t[1];
                cam_z_m = s * (R[6] * rh_x + R[7] * rh_y + R[8] * rh_z) + binding.align_t[2];

                // Orientation: camera→world = R_align · R_track, decomposed back
                // into the previz yaw/pitch/roll convention.
                double R_track[9];
                rot_ypr(rh_yaw, rh_pitch, rh_roll, R_track);
                double R_world[9];
                mat3_mul(R, R_track, R_world);
                euler_yxz(R_world, cam_yaw, cam_pitch, cam_roll);
            } else {
                cam_x_m = rh_x * binding.position_scale;
                cam_y_m = rh_y * binding.position_scale;
                cam_z_m = rh_z * binding.position_scale;
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
                        // Scale about the frame CENTRE: fill_scale pivots about the top-left
                        // corner, so without the 0.5·(1-s) the graphic slid towards the corner as
                        // the subject walked away, instead of shrinking in place.
                        t.image_transform.fill_translation[0] = fill_x + 0.5 * (1.0 - t_scale);
                        t.image_transform.fill_translation[1] = fill_y + 0.5 * (1.0 - t_scale);
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
                // Zoom scales about the frame CENTRE, not fill_scale's top-left pivot -- a zoom-in
                // used to slide the picture towards the corner. The roll below still turns about
                // the corner; changing the anchor would change what every translation means.
                t.image_transform.fill_translation[0]       = pan + offset_x + 0.5 * (1.0 - scale);
                t.image_transform.fill_translation[1]       = -tilt - offset_y + 0.5 * (1.0 - scale);
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

void tracker_registry::on_data(camera_data data, tracking_protocol protocol, int port)
{
    data.source_protocol = static_cast<int>(protocol);
    data.source_port     = port;
    on_data(data);
}

// ── A SAMPLE DRIVES THE BINDINGS OF THE RECEIVER THAT PRODUCED IT ────────────────────────
//
// Matched on camera_id ALONE until 2026-09-23, so every receiver drove every binding sharing
// the ID -- across ports AND protocols. Two FreeD cameras on 6301 and 6302 both sending the
// common default ID 1 drove each other's layers; an OSC tracker and a FreeD tracker on one ID
// fought over one binding. And the caches were keyed by ID alone too, so INFO and the delay
// history read whichever receiver spoke last. The key is (protocol, port, camera_id) now,
// and a sample matches a binding only if it came from that binding's receiver.
//
// NOT DISTINGUISHED: two multicast receivers of one protocol on one PORT but different GROUPS
// (PSN, OpenTrackIO), and VRPN, which has no port. Those need the host in the key; the
// binding's `host` is empty when the default group is used while the receiver's is not, so
// comparing it naively would match nothing.
/// When a delayed binding should be showing, for a wall-clock instant: `delay_ms` before it, or
/// `genlock_frames` channel frames before it snapped to the house LTC frame grid.
static std::chrono::steady_clock::time_point delayed_target(const tracker_binding&                  binding,
                                                            std::chrono::steady_clock::time_point at)
{
    auto to_dur = [](double seconds) {
        return std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(seconds));
    };
    if (binding.genlock_enable && binding.channel_fps > 0.0) {
        const double fps    = binding.channel_fps;
        auto         target = at - to_dur(binding.genlock_frames / fps);

        uint32_t                              ltc_frame = 0;
        std::chrono::steady_clock::time_point ltc_time;
        if (ltc::LTCInput::instance().get_timecode_anchor(static_cast<int>(std::lround(fps)), ltc_frame, ltc_time)) {
            const double rel = std::chrono::duration<double>(target - ltc_time).count();
            const double k   = std::round(static_cast<double>(ltc_frame) + rel * fps);
            target           = ltc_time + to_dur((k - static_cast<double>(ltc_frame)) / fps);
        }
        return target;
    }
    return at - to_dur(binding.delay_ms / 1000.0);
}

tracker_registry::~tracker_registry() { stop_pump(); }

void tracker_registry::stop_pump()
{
    pump_run_ = false;
    if (pump_thread_.joinable())
        pump_thread_.join();
}

void tracker_registry::pump_loop()
{
    while (pump_run_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));

        std::vector<std::pair<tracker_binding, std::deque<camera_data>>> work;
        {
            std::lock_guard<std::mutex> lk(mutex_);
            for (auto it = pump_.begin(); it != pump_.end();) {
                const auto b = bindings_.find(it->first);
                if (b == bindings_.end() || !(b->second.delay_ms > 0.0 || b->second.genlock_enable)) {
                    it = pump_.erase(it); // unbound, or no longer delayed
                    continue;
                }
                if (!it->second.settled) {
                    const auto h = sample_history_.find(it->second.source);
                    if (h != sample_history_.end() && !h->second.empty()) {
                        const auto target = delayed_target(b->second, std::chrono::steady_clock::now());
                        work.emplace_back(b->second, h->second);
                        if (target >= h->second.back().timestamp)
                            it->second.settled = true; // this injection shows the newest sample
                    }
                }
                ++it;
            }
        }
        for (auto& [binding, hist] : work)
            inject_transform(binding, interpolate_history(hist, delayed_target(binding, std::chrono::steady_clock::now())));
    }
}

void tracker_registry::on_data(const camera_data& data)
{
    std::vector<tracker_binding> matched;
    std::deque<camera_data>      history_snapshot;
    bool                         any_delay = false;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        const source_key skey{data.source_protocol, data.source_port, data.camera_id};
        latest_data_[skey] = data;

        // Append to per-camera history and trim to the retention window/cap.
        auto& hist = sample_history_[skey];
        hist.push_back(data);
        const auto cutoff = data.timestamp - std::chrono::milliseconds(HISTORY_WINDOW_MS);
        while (hist.size() > 1 && hist.front().timestamp < cutoff)
            hist.pop_front();
        while (hist.size() > HISTORY_MAX_SAMPLES)
            hist.pop_front();

        for (auto& [key, binding] : bindings_) {
            const bool from_its_receiver =
                data.source_protocol < 0 ||
                (static_cast<int>(binding.receiver.protocol) == data.source_protocol &&
                 (data.source_port < 0 || binding.receiver.port == data.source_port));
            if (from_its_receiver && (binding.camera_id == -1 || binding.camera_id == data.camera_id)) {
                matched.push_back(binding);
                if (binding.delay_ms > 0.0 || binding.genlock_enable) {
                    any_delay  = true;
                    pump_[key] = pump_state{skey, false}; // a new sample to land
                }
            }
        }

        if (any_delay) {
            history_snapshot = hist; // copy for lock-free interpolation below
            if (!pump_run_.exchange(true))
                pump_thread_ = std::thread([this] { pump_loop(); });
        }
    }

    for (auto& binding : matched) {
        if (binding.genlock_enable || binding.delay_ms > 0.0) {
            // Frame-native genlock holds the pose back by N channel frames and snaps to the house
            // LTC grid; a plain DELAY holds it back by milliseconds. Both via delayed_target,
            // which the pump uses too, so the two paths cannot disagree about the rule.
            inject_transform(binding, interpolate_history(history_snapshot, delayed_target(binding, data.timestamp)));
        } else {
            inject_transform(binding, data);
        }
    }
}

std::optional<camera_data> tracker_registry::get_latest_data(int camera_id) const
{
    // Any receiver: the newest sample for this ID. Kept for callers with no binding to hand.
    std::lock_guard<std::mutex>  lk(mutex_);
    std::optional<camera_data>   best;
    for (const auto& [key, d] : latest_data_)
        if (std::get<2>(key) == camera_id && (!best || d.timestamp > best->timestamp))
            best = d;
    return best;
}

std::optional<camera_data> tracker_registry::get_latest_data(const receiver_handle& source, int camera_id) const
{
    std::lock_guard<std::mutex> lk(mutex_);
    std::optional<camera_data>  best;
    for (const auto& [key, d] : latest_data_) {
        const bool same_source = std::get<0>(key) < 0 ||
                                 (std::get<0>(key) == static_cast<int>(source.protocol) &&
                                  (std::get<1>(key) < 0 || std::get<1>(key) == source.port));
        // The wildcard reads what an ANY binding is actually driven by: the newest sample from
        // its receiver. It used to look up camera -1 literally, find nothing, and have INFO
        // report no data and ZERO refuse on a binding that was moving the picture.
        const bool same_camera = camera_id == -1 || std::get<2>(key) == camera_id;
        if (same_source && same_camera && (!best || d.timestamp > best->timestamp))
            best = d;
    }
    return best;
}

}} // namespace caspar::tracking
