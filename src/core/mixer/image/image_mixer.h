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

#include <core/frame/frame.h>
#include <core/frame/frame_factory.h>
#include <core/frame/frame_visitor.h>
#include <core/frame/pixel_format.h>

#include <cstdint>
#include <future>
#include <memory>
#include <string>

namespace caspar { namespace core {

struct lut3d_data; // fwd (defined in core/frame/frame_transform.h)

/// Snapshot of the channel-master calibration LUT state, reported by INFO.
struct calibration_lut_state
{
    bool         enabled  = false;   // a calibration LUT is loaded
    bool         bypass   = false;   // temporarily bypassed (e.g. while shooting patches)
    int          size     = 0;       // LUT cube dimension (e.g. 33)
    float        strength = 1.0f;    // 0..1 blend factor
    std::wstring path;               // source .cube path (for diagnostics)
};

class image_mixer
    : public frame_visitor
    , public frame_factory
{
    image_mixer(const image_mixer&);
    image_mixer& operator=(const image_mixer&);

  public:
    image_mixer() {}
    virtual ~image_mixer() {}

    void push(const struct frame_transform& frame) override = 0;
    void visit(const class const_frame& frame) override     = 0;
    void pop() override                                     = 0;

    virtual void update_aspect_ratio(double aspect_ratio) = 0;

    virtual std::future<std::tuple<std::shared_future<array<const std::uint8_t>>, std::shared_ptr<texture>>>
    render(const struct video_format_desc& format_desc) = 0;

    class mutable_frame create_frame(const void* tag, const struct pixel_format_desc& desc) override = 0;
    class mutable_frame create_frame(const void*                     video_stream_tag,
                                     const struct pixel_format_desc& desc,
                                     common::bit_depth               depth) override                               = 0;

    virtual common::bit_depth depth() const = 0;

    virtual bool is_vulkan() const { return false; }

    /// Return the native GL context handle for context sharing (nullptr if not applicable).
    virtual void* native_gl_context() const { return nullptr; }

    /// Return the native EGL display for context ops (nullptr on non-EGL platforms).
    virtual void* native_egl_display() const { return nullptr; }

    virtual void set_cpu_readback_needed(bool needed) { (void)needed; }

    virtual void set_target_color(color_space cs, color_transfer ct, bool auto_convert, int auto_tone_map = 0, float peak_luminance = 1000.0f, float sdr_reference_white = 100.0f, bool auto_gamut_compress = false)
    {
        (void)cs;
        (void)ct;
        (void)auto_convert;
        (void)auto_tone_map;
        (void)peak_luminance;
        (void)sdr_reference_white;
        (void)auto_gamut_compress;
    }

    /// Channel-master LED-wall calibration LUT. Applied to the final composited
    /// frame (channel→output, post-grade) so every consumer receives the
    /// corrected output. Pass nullptr to clear.
    virtual void set_calibration_lut(std::shared_ptr<const lut3d_data> lut, float strength, const std::wstring& path)
    {
        (void)lut;
        (void)strength;
        (void)path;
    }

    /// Temporarily bypass the calibration LUT without unloading it (e.g. while
    /// shooting calibration patches).
    virtual void set_calibration_bypass(bool bypass) { (void)bypass; }

    virtual calibration_lut_state get_calibration_state() const { return {}; }
};

}} // namespace caspar::core
