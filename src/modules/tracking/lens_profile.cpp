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

#include "lens_profile.h"

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace caspar { namespace tracking {

namespace {

constexpr double DEG2RAD = 3.14159265358979323846 / 180.0;

/// Reads one lens_sample node, converting FOV per the profile's unit.
lens_sample read_sample(const boost::property_tree::ptree& n, double fov_to_rad)
{
    lens_sample s;
    s.k1              = n.get("k1", 0.0);
    s.k2              = n.get("k2", 0.0);
    s.k3              = n.get("k3", 0.0);
    s.p1              = n.get("p1", 0.0);
    s.p2              = n.get("p2", 0.0);
    s.fov_rad         = n.get("fov", 0.0) * fov_to_rad;
    s.nodal_forward_m = n.get("nodal_forward_m", 0.0);
    return s;
}

/// Finds the bracketing indices and interpolation fraction for `v` within the
/// ascending `axis`. Clamps to the ends when out of range.
void bracket(const std::vector<double>& axis, double v, size_t& i0, size_t& i1, double& t)
{
    if (axis.size() == 1 || v <= axis.front()) {
        i0 = i1 = 0;
        t       = 0.0;
        return;
    }
    if (v >= axis.back()) {
        i0 = i1 = axis.size() - 1;
        t       = 0.0;
        return;
    }
    auto hi = std::lower_bound(axis.begin(), axis.end(), v);
    i1      = static_cast<size_t>(std::distance(axis.begin(), hi));
    i0      = i1 - 1;
    const double span = axis[i1] - axis[i0];
    t                 = (span > 0.0) ? (v - axis[i0]) / span : 0.0;
}

lens_sample lerp_sample(const lens_sample& a, const lens_sample& b, double t)
{
    lens_sample s;
    s.k1              = a.k1 + (b.k1 - a.k1) * t;
    s.k2              = a.k2 + (b.k2 - a.k2) * t;
    s.k3              = a.k3 + (b.k3 - a.k3) * t;
    s.p1              = a.p1 + (b.p1 - a.p1) * t;
    s.p2              = a.p2 + (b.p2 - a.p2) * t;
    s.fov_rad         = a.fov_rad + (b.fov_rad - a.fov_rad) * t;
    s.nodal_forward_m = a.nodal_forward_m + (b.nodal_forward_m - a.nodal_forward_m) * t;
    return s;
}

} // namespace

std::shared_ptr<lens_profile> lens_profile::load(const std::string& path)
{
    namespace pt = boost::property_tree;
    pt::ptree root;
    pt::read_json(path, root);

    auto profile   = std::make_shared<lens_profile>();
    profile->path_ = path;
    profile->name_ = root.get<std::string>("name", "lens");

    const std::string fov_unit   = root.get<std::string>("fov_unit", "deg");
    const double      fov_to_rad = (fov_unit == "rad") ? 1.0 : DEG2RAD;

    auto zoom_axis = root.get_child_optional("zoom_axis");
    if (!zoom_axis || zoom_axis->empty())
        throw std::runtime_error("lens profile: missing or empty 'zoom_axis'");
    for (const auto& [_, z] : *zoom_axis)
        profile->zoom_axis_.push_back(z.get_value<double>());

    // focus_axis is optional; absence means a 1-D (zoom-only) profile.
    bool is_2d = false;
    if (auto focus_axis = root.get_child_optional("focus_axis")) {
        for (const auto& [_, f] : *focus_axis)
            profile->focus_axis_.push_back(f.get_value<double>());
    }
    if (profile->focus_axis_.size() >= 2)
        is_2d = true;
    if (profile->focus_axis_.empty())
        profile->focus_axis_.push_back(0.0); // single dummy node for 1-D lookups

    // Axes must be strictly ascending for bracketing to work.
    auto check_ascending = [](const std::vector<double>& a, const char* what) {
        for (size_t i = 1; i < a.size(); ++i)
            if (a[i] <= a[i - 1])
                throw std::runtime_error(std::string("lens profile: '") + what + "' must be ascending");
    };
    check_ascending(profile->zoom_axis_, "zoom_axis");
    if (is_2d)
        check_ascending(profile->focus_axis_, "focus_axis");

    auto samples = root.get_child_optional("samples");
    if (!samples || samples->empty())
        throw std::runtime_error("lens profile: missing or empty 'samples'");

    const size_t n_zoom  = profile->zoom_axis_.size();
    const size_t n_focus = profile->focus_axis_.size();
    profile->grid_.assign(n_zoom, std::vector<lens_sample>(n_focus));

    if (is_2d) {
        // 2-D: 'samples' is an array of rows; each row an array of focus columns.
        size_t zi = 0;
        for (const auto& [_, row] : *samples) {
            if (zi >= n_zoom)
                throw std::runtime_error("lens profile: more sample rows than zoom_axis entries");
            size_t fi = 0;
            for (const auto& [__, cell] : row) {
                if (fi >= n_focus)
                    throw std::runtime_error("lens profile: more sample columns than focus_axis entries");
                profile->grid_[zi][fi] = read_sample(cell, fov_to_rad);
                ++fi;
            }
            if (fi != n_focus)
                throw std::runtime_error("lens profile: sample row size != focus_axis size");
            ++zi;
        }
        if (zi != n_zoom)
            throw std::runtime_error("lens profile: sample row count != zoom_axis size");
    } else {
        // 1-D: 'samples' is a flat array, one entry per zoom node.
        size_t zi = 0;
        for (const auto& [_, cell] : *samples) {
            if (zi >= n_zoom)
                throw std::runtime_error("lens profile: more samples than zoom_axis entries");
            profile->grid_[zi][0] = read_sample(cell, fov_to_rad);
            ++zi;
        }
        if (zi != n_zoom)
            throw std::runtime_error("lens profile: sample count != zoom_axis size");
    }

    return profile;
}

lens_sample lens_profile::sample(double zoom_raw, double focus_raw) const
{
    if (grid_.empty())
        return lens_sample{};

    size_t z0, z1, f0, f1;
    double tz, tf;
    bracket(zoom_axis_, zoom_raw, z0, z1, tz);
    bracket(focus_axis_, focus_raw, f0, f1, tf);

    // Bilinear: interpolate along focus at each zoom row, then along zoom.
    const lens_sample s0 = lerp_sample(grid_[z0][f0], grid_[z0][f1], tf);
    const lens_sample s1 = lerp_sample(grid_[z1][f0], grid_[z1][f1], tf);
    return lerp_sample(s0, s1, tz);
}

}} // namespace caspar::tracking
