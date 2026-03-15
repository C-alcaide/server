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
 * Author: Robert Nagy, ronag89@gmail.com
 */

#pragma once

#include <common/tweener.h>

#include <core/mixer/image/blend_modes.h>

#include <array>
#include <optional>

namespace caspar { namespace core {

struct chroma
{
    enum class legacy_type
    {
        none,
        green,
        blue
    };

    bool   enable                    = false;
    bool   show_mask                 = false;
    double target_hue                = 0.0;
    double hue_width                 = 0.0;
    double min_saturation            = 0.0;
    double min_brightness            = 0.0;
    double softness                  = 0.0;
    double spill_suppress            = 0.0;
    double spill_suppress_saturation = 1.0;
};

struct levels final
{
    double min_input  = 0.0;
    double max_input  = 1.0;
    double gamma      = 1.0;
    double min_output = 0.0;
    double max_output = 1.0;
};

enum class screen_curve_type { flat = 0, cylinder = 1, sphere = 2 };

struct projection final
{
    bool              enable       = false;
    double            yaw          = 0.0;
    double            pitch        = 0.0;
    double            roll         = 0.0;
    double            fov          = 1.57079632679;
    double            offset_x     = 0.0;  // NDC lens-shift: +1 = pan right, -1 = pan left
    double            offset_y     = 0.0;  // NDC lens-shift: +1 = pan up,    -1 = pan down
    // Curved screen compensation — independent of 360 mode
    bool              curve_enable = false;
    screen_curve_type curve_type   = screen_curve_type::flat;
    double            screen_arc   = 0.0;  // total arc in radians (horizontal for cylinder, radial for sphere)
};

// Transfer: 0=linear,1=srgb,2=rec709,3=pq(st2084),4=hlg,5=logc3(arri),6=slog3(sony)
// Gamut:    0=bt709,1=bt2020,2=dcip3_d65,3=aces_ap0,4=aces_ap1(acescg),5=arri_wg3,6=sgamut3_cine
// Tonemapping: 0=none,1=reinhard,2=aces_filmic,3=aces_rrt
struct color_grade final
{
    bool  enable          = false;
    int   input_transfer  = 0;
    int   input_gamut     = 0;
    int   tone_mapping    = 0;
    int   output_gamut    = 0;
    int   output_transfer = 0;
    float exposure        = 1.0f;
};

// ---- Per-channel RGB Levels ------------------------------------------------
struct rgb_levels_channel final
{
    double min_input  = 0.0;
    double max_input  = 1.0;
    double gamma      = 1.0;
    double min_output = 0.0;
    double max_output = 1.0;
};

struct rgb_levels final
{
    bool               enable = false;
    rgb_levels_channel r;
    rgb_levels_channel g;
    rgb_levels_channel b;
};

// ---- Tone Curves (per-channel + master, up to 16 control points each) ------
struct curve_point final
{
    double x = 0.0;
    double y = 0.0;
};

// count == 0 means "identity" — the LUT builder returns a linear ramp.
struct curve_channel final
{
    int                         count = 0;
    std::array<curve_point, 16> points{};
};

struct tone_curves final
{
    bool          enable = false;
    curve_channel master;  // applied as global tone to every channel after per-channel
    curve_channel red;
    curve_channel green;
    curve_channel blue;
};

struct corners final
{
    std::array<double, 2> ul = {0.0, 0.0};
    std::array<double, 2> ur = {1.0, 0.0};
    std::array<double, 2> lr = {1.0, 1.0};
    std::array<double, 2> ll = {0.0, 1.0};
};

struct rectangle final
{
    std::array<double, 2> ul = {0.0, 0.0};
    std::array<double, 2> lr = {1.0, 1.0};
};

enum class blur_type : int
{
    gaussian    = 0,
    box         = 1,
    directional = 2,
    zoom        = 3,
    tilt_shift  = 4,
    lens        = 5
};

struct blur_config final
{
    bool                  enable = false;
    double                radius = 0.0;
    blur_type             type   = blur_type::gaussian;
    double                angle  = 0.0;
    std::array<double, 2> center = {0.5, 0.5};
    double                tilt_y = 0.5;
    double                tilt_h = 0.2;
};

// Shape types for MIXER SHAPE
enum class shape_type : int
{
    rect         = 0,
    rounded_rect = 1,
    circle       = 2,
    ellipse      = 3
};

// Fill types for MIXER SHAPE
enum class shape_fill_type : int
{
    solid   = 0,
    linear  = 1,
    radial  = 2,
    conic   = 3
};

struct shape_config final
{
    bool                  enable          = false;
    shape_type            type            = shape_type::rect;
    std::array<double, 2> center          = {0.5, 0.5};   // normalised 0-1
    std::array<double, 2> size            = {0.5, 0.5};   // normalised 0-1
    double                corner_radius   = 0.0;          // normalised; used by rounded_rect
    double                edge_softness   = 0.005;        // AA feather width
    shape_fill_type       fill_type       = shape_fill_type::solid;
    std::array<double, 4> color1          = {1.0, 1.0, 1.0, 1.0}; // RGBA
    std::array<double, 4> color2          = {0.0, 0.0, 0.0, 0.0}; // RGBA
    double                gradient_angle  = 0.0;          // degrees; for linear fill
    std::array<double, 2> gradient_center = {0.5, 0.5};  // normalised; for radial/conic
    bool                  stroke_enable   = false;
    double                stroke_width    = 0.0;          // normalised
    std::array<double, 4> stroke_color    = {1.0, 1.0, 1.0, 1.0}; // RGBA
};

struct image_transform final
{
    double opacity    = 1.0;
    double contrast   = 1.0;
    double brightness = 1.0;
    double saturation = 1.0;

    /**
     * This enables the clip/crop/perspective fields.
     * It is often desirable to have this disabled, to avoid cropping/clipping unnecessarily
     */
    bool enable_geometry_modifiers = false;

    std::array<double, 2> anchor           = {0.0, 0.0};
    std::array<double, 2> fill_translation = {0.0, 0.0};
    std::array<double, 2> fill_scale       = {1.0, 1.0};
    std::array<double, 2> clip_translation = {0.0, 0.0};
    std::array<double, 2> clip_scale       = {1.0, 1.0};
    double                angle            = 0.0;
    rectangle             crop;
    corners               perspective;
    core::levels          levels;
    core::chroma          chroma;
    core::projection      projection;
    core::color_grade     color_grade;

    // White balance (temperature/tint)
    double temperature = 0.0;  // -1..+1  (neg=cool/blue, pos=warm/orange)
    double tint        = 0.0;  // -1..+1  (neg=magenta, pos=green)

    // Lift / Midtone / Gain — per-channel 3-way color corrector (DaVinci-style)
    std::array<double, 3> lift    = {0.0, 0.0, 0.0};  // shadow offset,      default 0
    std::array<double, 3> midtone = {1.0, 1.0, 1.0};  // midtone power,      default 1
    std::array<double, 3> gain    = {1.0, 1.0, 1.0};  // highlight mult,     default 1

    // Hue shift (degrees, -180..+180)
    double hue_shift = 0.0;

    // Tonal balance (shadows / highlights separation)
    double shadows    = 0.0;  // -1..+1
    double highlights = 0.0;  // -1..+1

    // Per-channel RGB levels and tone curves
    core::rgb_levels  per_channel_levels;
    core::tone_curves curves;
    blur_config       blur;
    shape_config      shape;

    bool             is_key      = false;
    bool             invert      = false;
    bool             flip_h      = false;  // horizontal mirror (left ↔ right)
    bool             flip_v      = false;  // vertical mirror   (top ↔ bottom)
    bool             is_mix      = false;
    core::blend_mode blend_mode  = blend_mode::normal;
    int              layer_depth = 0;

    static image_transform tween(double                 time,
                                 const image_transform& source,
                                 const image_transform& dest,
                                 double                 duration,
                                 const tweener&         tween);
};

bool operator==(const image_transform& lhs, const image_transform& rhs);
bool operator!=(const image_transform& lhs, const image_transform& rhs);

struct audio_transform final
{
    double volume           = 1.0;
    bool   immediate_volume = false; // When false, intra-frame samples are ramped from previous volume in audio mixer

    audio_transform& operator*=(const audio_transform& other);
    audio_transform  operator*(const audio_transform& other) const;

    static audio_transform tween(double                 time,
                                 const audio_transform& source,
                                 const audio_transform& dest,
                                 double                 duration,
                                 const tweener&         tween);
};

bool operator==(const audio_transform& lhs, const audio_transform& rhs);
bool operator!=(const audio_transform& lhs, const audio_transform& rhs);

struct frame_transform final
{
  public:
    frame_transform();

    core::image_transform image_transform;
    core::audio_transform audio_transform;

    static frame_transform tween(double                 time,
                                 const frame_transform& source,
                                 const frame_transform& dest,
                                 double                 duration,
                                 const tweener&         tween);
};

bool operator==(const frame_transform& lhs, const frame_transform& rhs);
bool operator!=(const frame_transform& lhs, const frame_transform& rhs);

class tweened_transform
{
    frame_transform source_;
    frame_transform dest_;
    int             duration_ = 0;
    int             time_     = 0;
    tweener         tweener_;

  public:
    tweened_transform() = default;

    tweened_transform(const frame_transform& source, const frame_transform& dest, int duration, tweener tween);

    const frame_transform& dest() const;

    frame_transform fetch();
    void            tick(int num);
};

std::optional<chroma::legacy_type> get_chroma_mode(const std::wstring& str);

}} // namespace caspar::core
