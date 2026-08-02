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

#pragma once

#include <core/frame/frame.h>

#include <cmath>
#include <cstdint>
#include <vector>

// Fixture geometry and colour sampling shared by the artnet and sacn consumers.
//
// These lived as two near-identical copies under artnet/util and sacn/util. They
// drifted -- sacn gained a divide-by-zero guard artnet never got -- and a
// rasteriser bug present in both had to be found twice. One copy now.
namespace caspar { namespace dmx {

enum FixtureType
{
    DIMMER = 1,
    RGB    = 3,
    RGBW   = 4,
};

struct point
{
    float x;
    float y;
};

struct rect
{
    point p1;
    point p2;
    point p3;
    point p4;
};

struct computed_fixture
{
    FixtureType    type;
    unsigned short address;

    rect rectangle;
};

struct color
{
    std::uint8_t r;
    std::uint8_t g;
    std::uint8_t b;
};

struct box
{
    float x;
    float y;

    float width;
    float height;

    float rotation; // degrees
};

struct fixture
{
    FixtureType    type;
    unsigned short startAddress;    // DMX address of the first channel in the fixture
    unsigned short fixtureCount;    // number of fixtures in the chain, dividing along the width
    unsigned short fixtureChannels; // number of channels per fixture

    box fixtureBox;
};

/// Corners of the `index`th of `count` fixtures tiled along the box's local X axis,
/// rotated about the box centre. Returned in winding order p1->p2->p3->p4.
rect compute_rect(box fixtureBox, int index, int count);

/// Alpha-weighted average colour of the pixels covered by `rectangle`.
///
/// `data` is packed 8-bit BGRA with stride == width * 4 -- what the mixer emits for
/// an 8-bit channel (see core::mixer). `scale_x`/`scale_y` map rectangle coordinates,
/// which are always in the channel's own pixel space, onto `data`'s resolution: 1.0
/// for a full-resolution frame, out_width/channel_width for a downscaled one.
///
/// Returns black when the rectangle covers no pixels of the image.
color average_color(const std::uint8_t* data,
                    int                 width,
                    int                 height,
                    const rect&         rectangle,
                    float               scale_x = 1.0f,
                    float               scale_y = 1.0f);

/// Full-resolution convenience overload. Returns black for a frame with no host
/// pixels (a GPU-only frame whose readback was skipped) rather than dereferencing null.
color average_color(const core::const_frame& frame, const rect& rectangle);

/// The coarsest reduction that still leaves every fixture at least `min_texels`
/// across, clamped to [0, max_levels].
///
/// Below about 4 texels the error from a partially covered texel at the region
/// border starts to dominate the average. For the shipped example geometry
/// (50x100 fixtures) this returns 3, i.e. 1/8 per axis: 240x135 instead of
/// 1920x1080, 129 KB instead of 8.29 MB.
///
/// Fixture geometry is static configuration, so call this once when the fixture
/// list is built, not per frame.
int pick_reduction_levels(const std::vector<computed_fixture>& fixtures, int max_levels = 4, int min_texels = 4);

/**
 * A frame's pixels prepared for fixture sampling, at whatever resolution was
 * cheapest to get them at.
 *
 * The DMX consumers need a few averaged colours, not a picture. Declaring
 * needs_cpu_frame_data() to obtain them makes the channel read the whole
 * composited frame back to host memory every tick -- 8.29 MB at 1080p, at the
 * channel's rate, not the sender's 10-30 Hz -- to compute a handful of bytes.
 *
 * So this asks the frame's GPU texture for a box-filtered reduction first, and
 * only falls back to the frame's host pixels if that is unavailable. Which one
 * happened is reported by reduced(), and the consumer feeds that back through
 * needs_cpu_frame_data() so the readback is re-armed if the GPU path ever stops
 * working.
 *
 * Sampling on the reduced image is a box filter over a box filter, which is the
 * same box filter -- exact for a fully covered region, with error only at the
 * border texels. One documented difference: RGB and alpha are averaged
 * independently there, giving avg(rgb)*avg(a) rather than the full-resolution
 * sum(rgb*a)/N. Those agree whenever alpha is constant over the fixture, which
 * is the normal composited-output case.
 */
class sampling_source
{
  public:
    sampling_source(const core::const_frame& frame, int levels);

    /// False when neither a reduction nor host pixels were available -- an
    /// entirely GPU-resident frame on a mixer with no reduction support. The
    /// consumer should emit nothing rather than black, so a transient failure
    /// does not blink the fixtures.
    explicit operator bool() const { return data_ != nullptr; }

    /// True when this came from the GPU reduction rather than a full readback.
    bool reduced() const { return reduced_; }

    color average(const rect& rectangle) const;

  private:
    std::vector<std::uint8_t> owned_; ///< populated only on the reduced path
    const std::uint8_t*       data_   = nullptr;
    int                       width_  = 0;
    int                       height_ = 0;
    float                     scale_x_ = 1.0f;
    float                     scale_y_ = 1.0f;
    bool                      reduced_ = false;
};

}} // namespace caspar::dmx
