/*
 * Copyright (c) 2011 Sveriges Television AB <info@casparcg.com>
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
 *
 * Author: Eliyah Sundström eliyah@sundstroem.com
 */

#include "fixture_geometry.h"

#include <algorithm>

namespace caspar { namespace dmx {

namespace {

constexpr double kPi = 3.14159265358979323846;

/// Horizontal span of the quad on the sample line `yc`, found by intersecting that
/// line with all four edges and taking the extremes.
///
/// This replaces an active-edge-pair table indexed by which of the three vertical
/// bands `y` fell into. On the last scanline of an axis-aligned rect -- the common
/// case, since rotation defaults to 0 -- that table's third entry selected two edges
/// sharing the same endpoint, the slope guard skipped the interpolation, and the span
/// silently kept its initialisers of 0 and width-1. That fixture's bottom row was
/// therefore sampled across the full width of the frame instead of across the
/// fixture. Measured on the shipped example geometry (ten 50x100 fixtures): 7020
/// pixels accumulated where 5151 are covered, and a region of pure red averaged to
/// 181 instead of 247 -- 27% of the value coming from outside the fixture.
///
/// `xs`/`ys` must be in winding order (p1->p2->p3->p4), not sorted -- the edge walk
/// depends on consecutive entries being actual edges of the quad.
bool scanline_span(const float (&xs)[4], const float (&ys)[4], float yc, float& x_lo, float& x_hi)
{
    bool any = false;

    for (int e = 0; e < 4; ++e) {
        const int   f  = (e + 1) & 3;
        const float y0 = ys[e];
        const float y1 = ys[f];

        // A horizontal edge contributes no crossing; its endpoints are picked up by
        // the two edges adjoining it, which is what makes the degenerate case above
        // resolve correctly instead of falling through.
        if (y0 == y1)
            continue;
        if (yc < std::min(y0, y1) || yc > std::max(y0, y1))
            continue;

        const float t = (yc - y0) / (y1 - y0);
        const float x = xs[e] + t * (xs[f] - xs[e]);

        x_lo = any ? std::min(x_lo, x) : x;
        x_hi = any ? std::max(x_hi, x) : x;
        any  = true;
    }

    return any;
}

} // namespace

rect compute_rect(box fixtureBox, int index, int count)
{
    // Calculates the corners of a rectangle that is part of a fixture
    // The count represents how many fixtures exist in the box and the index which one to calculate

    auto f_count = (float)count;
    auto f_index = (float)index;

    float x = fixtureBox.x;
    float y = fixtureBox.y;

    float width  = fixtureBox.width;
    float height = fixtureBox.height;

    float  rotation = fixtureBox.rotation;
    double angle    = kPi * rotation / 180.0;

    double sin_ = sin(angle);
    double cos_ = cos(angle);

    // Half width and height of the rectangle for this fixture
    float hx = width / (2 * f_count);
    float hy = height / 2.0f;

    // Offset distance from the center of the box to the center of the fixture
    float od = (2 * f_index - f_count + 1) * hx;

    // Center of the fixture
    double ox = x + od * cos_;
    double oy = y + od * sin_;

    // Calculate the corners of the rectangle, by offsetting the center with the half width and height
    // in the direction of the corners and the box's rotation

    point p1{
        static_cast<float>(ox + -hx * cos_ + -hy * -sin_),
        static_cast<float>(oy + -hx * sin_ + -hy * cos_),
    };

    point p2{
        static_cast<float>(ox + hx * cos_ + -hy * -sin_),
        static_cast<float>(oy + hx * sin_ + -hy * cos_),
    };

    point p3{
        static_cast<float>(ox + hx * cos_ + hy * -sin_),
        static_cast<float>(oy + hx * sin_ + hy * cos_),
    };

    point p4{
        static_cast<float>(ox + -hx * cos_ + hy * -sin_),
        static_cast<float>(oy + -hx * sin_ + hy * cos_),
    };

    rect rectangle{p1, p2, p3, p4};

    return rectangle;
}

color average_color(const std::uint8_t* data,
                    int                 width,
                    int                 height,
                    const rect&         rectangle,
                    float               scale_x,
                    float               scale_y)
{
    if (data == nullptr || width <= 0 || height <= 0)
        return color{0, 0, 0};

    // Winding order, scaled into the resolution of `data`.
    const float xs[4] = {rectangle.p1.x * scale_x,
                         rectangle.p2.x * scale_x,
                         rectangle.p3.x * scale_x,
                         rectangle.p4.x * scale_x};
    const float ys[4] = {rectangle.p1.y * scale_y,
                         rectangle.p2.y * scale_y,
                         rectangle.p3.y * scale_y,
                         rectangle.p4.y * scale_y};

    const float y_lo = std::min(std::min(ys[0], ys[1]), std::min(ys[2], ys[3]));
    const float y_hi = std::max(std::max(ys[0], ys[1]), std::max(ys[2], ys[3]));

    const int y_min = std::max(0, std::min(height - 1, (int)y_lo));
    const int y_max = std::max(0, std::min(height - 1, (int)y_hi));

    // Totals are accumulated at full precision and averaged once, so a large
    // fixture does not lose the low bits to repeated rounding.
    unsigned long long tr = 0;
    unsigned long long tg = 0;
    unsigned long long tb = 0;

    unsigned long long count = 0;

    for (int y = y_min; y <= y_max; y++) {
        float x_lo = 0.0f;
        float x_hi = 0.0f;

        // Sampled on the integer grid, matching the original rasteriser -- the only
        // intended behavioural change here is the degenerate-span fix, not a
        // re-specification of which pixels a fixture covers.
        if (!scanline_span(xs, ys, (float)y, x_lo, x_hi))
            continue;

        const int x1 = std::max(0, std::min(width - 1, (int)x_lo));
        const int x2 = std::max(0, std::min(width - 1, (int)x_hi));

        const int min_x = std::min(x1, x2);
        const int max_x = std::max(x1, x2);

        for (int x = min_x; x <= max_x; x++) {
            const int           pos      = y * width + x;
            const std::uint8_t* base_ptr = data + pos * 4;

            float a = (float)base_ptr[3] / 255.0f;

            float r = (float)base_ptr[2] * a;
            float g = (float)base_ptr[1] * a;
            float b = (float)base_ptr[0] * a;

            tr += (unsigned long long)(r);
            tg += (unsigned long long)(g);
            tb += (unsigned long long)(b);

            count++;
        }
    }

    // A fixture entirely outside the frame now yields no samples at all, where the
    // old rasteriser would clamp to a single scanline and return a full-width average
    // of a row the fixture does not touch. So this guard is load-bearing rather than
    // defensive: the zero case was unreachable before and is reachable now. (sacn's
    // copy already had the check, artnet's did not, and neither could trigger it.)
    if (count == 0)
        return color{0, 0, 0};

    return color{(std::uint8_t)(tr / count), (std::uint8_t)(tg / count), (std::uint8_t)(tb / count)};
}

color average_color(const core::const_frame& frame, const rect& rectangle)
{
    if (!frame.has_host_image())
        return color{0, 0, 0};

    return average_color(
        frame.image_data(0).data(), (int)frame.width(), (int)frame.height(), rectangle);
}

int pick_reduction_levels(const std::vector<computed_fixture>& fixtures, int max_levels, int min_texels)
{
    if (fixtures.empty())
        return 0;

    int levels = max_levels;
    for (const auto& cf : fixtures) {
        const auto& r  = cf.rectangle;
        const float xs[4] = {r.p1.x, r.p2.x, r.p3.x, r.p4.x};
        const float ys[4] = {r.p1.y, r.p2.y, r.p3.y, r.p4.y};

        // Bounding box, not edge lengths: a rotated fixture is rasterised against
        // the texel grid, so it is the axis-aligned extent that has to stay
        // resolvable.
        const float w = *std::max_element(xs, xs + 4) - *std::min_element(xs, xs + 4);
        const float h = *std::max_element(ys, ys + 4) - *std::min_element(ys, ys + 4);

        int l = 0;
        while (l < max_levels && std::min(w, h) / static_cast<float>(1 << (l + 1)) >= static_cast<float>(min_texels))
            ++l;

        levels = std::min(levels, l);
    }
    return levels;
}

sampling_source::sampling_source(const core::const_frame& frame, int levels)
{
    if (auto tex = frame.texture()) {
        int w = 0, h = 0;
        owned_ = tex->read_pixels_reduced(levels, w, h);
        if (!owned_.empty() && w > 0 && h > 0 &&
            owned_.size() >= static_cast<std::size_t>(w) * h * 4) {
            data_    = owned_.data();
            width_   = w;
            height_  = h;
            reduced_ = true;
            // Fixture coordinates are always in the channel's pixel space, so map
            // them onto the reduced grid. Derived from the actual returned size
            // rather than 1<<levels, because each halving floors.
            const auto fw = static_cast<float>(frame.width());
            const auto fh = static_cast<float>(frame.height());
            scale_x_      = fw > 0.0f ? static_cast<float>(w) / fw : 1.0f;
            scale_y_      = fh > 0.0f ? static_cast<float>(h) / fh : 1.0f;
            return;
        }
        owned_.clear();
    }

    if (frame.has_host_image()) {
        data_   = frame.image_data(0).data();
        width_  = static_cast<int>(frame.width());
        height_ = static_cast<int>(frame.height());
    }
}

color sampling_source::average(const rect& rectangle) const
{
    if (!data_)
        return color{0, 0, 0};
    return average_color(data_, width_, height_, rectangle, scale_x_, scale_y_);
}

}} // namespace caspar::dmx
