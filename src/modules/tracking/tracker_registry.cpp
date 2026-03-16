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

#include <core/frame/frame_transform.h>
#include <core/producer/stage.h>
#include <common/tweener.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace caspar { namespace tracking {

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

void tracker_registry::inject_transform(const tracker_binding& binding, const camera_data& data)
{
    auto stage = binding.stage.lock();
    if (!stage)
        return; // channel has been destroyed

    const int    layer = binding.layer_index;
    const double pan   = data.pan * binding.pan_scale + binding.pan_offset;
    const double tilt  = data.tilt * binding.tilt_scale + binding.tilt_offset;
    const double roll  = data.roll + binding.roll_offset;
    const double fov   = compute_fov(binding, data.zoom);

    if (binding.mode == tracking_mode::mode_360) {
        // 360° equirectangular: set yaw / pitch / roll / fov on the projection struct.
        // This mirrors exactly what mixer_projection_command does in AMCPCommandsImpl.cpp.
        stage->apply_transform(
            layer,
            [pan, tilt, roll, fov](core::frame_transform t) -> core::frame_transform {
                t.image_transform.projection.enable = (fov > 0.0);
                t.image_transform.projection.yaw    = pan;
                t.image_transform.projection.pitch  = tilt;
                t.image_transform.projection.roll   = roll;
                t.image_transform.projection.fov    = fov;
                return t;
            },
            0,
            tweener(L"linear"));
    } else {
        // 2D mode: pan → fill_translation X, tilt → fill_translation Y (inverted so up = up),
        // roll → angle (radians), zoom → fill_scale (uniform).
        // CasparCG MIXER FILL space: {0,0} with scale {1,1} = full screen (no offset).
        // Positive pan moves the layer RIGHT; use negative pan_scale for counter-tracking.
        // scale < 1.0 zooms out; use zoom_default_fov / current_fov so narrower = larger.
        const double scale = (fov > 0.0) ? (binding.zoom_default_fov / fov) : 1.0;
        stage->apply_transform(
            layer,
            [pan, tilt, roll, scale](core::frame_transform t) -> core::frame_transform {
                t.image_transform.enable_geometry_modifiers = true;
                t.image_transform.fill_translation[0]       = pan;
                t.image_transform.fill_translation[1]       = -tilt; // invert: camera up → layer up
                t.image_transform.angle                     = roll;
                t.image_transform.fill_scale[0]             = scale;
                t.image_transform.fill_scale[1]             = scale;
                return t;
            },
            0,
            tweener(L"linear"));
    }
}

void tracker_registry::on_data(const camera_data& data)
{
    std::lock_guard<std::mutex> lk(mutex_);
    latest_data_[data.camera_id] = data;

    for (auto& [key, binding] : bindings_) {
        if (binding.camera_id == -1 || binding.camera_id == data.camera_id)
            inject_transform(binding, data);
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
