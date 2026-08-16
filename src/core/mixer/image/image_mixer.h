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
#include <utility>
#include <vector>

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

/// A channel-level OCIO display/view transform: what screen the composite is going to.
///
/// Channel-level rather than per layer, and that is not a simplification. An INPUT transform
/// describes where pixels came from, which is a property of each layer; a DISPLAY transform
/// describes what screen they are going to, and every layer in a channel goes to the same
/// screen. Two layers with different display transforms would blend a PQ-encoded layer with
/// a Rec.709-encoded one, and that composite is not in any space.
struct ocio_display_state
{
    bool        enabled = false;
    std::string display;
    std::string view;
};

/// A (display, view) pair, as a map key. Empty means "the channel's own view".
struct ocio_view_key
{
    std::string display;
    std::string view;

    bool empty() const { return display.empty() || view.empty(); }
    bool operator==(const ocio_view_key& o) const { return display == o.display && view == o.view; }
    bool operator<(const ocio_view_key& o) const
    {
        return display != o.display ? display < o.display : view < o.view;
    }
};

/// One rendered output: the lazy CPU readback and the GPU texture, exactly the pair
/// `render()` has always returned.
struct render_result
{
    std::shared_future<array<const std::uint8_t>> image;
    std::shared_ptr<class texture>                texture;
};

/// What one tick produces.
///
/// `primary` is the channel's own view and is what every consumer gets unless it asked for
/// something else. `views` carries one extra result per DISTINCT consumer view, rendered
/// from the same working-space composite -- which is the whole reason the fan-out lives in
/// the mixer: a display transform is not invertible, so a second view cannot be derived
/// from the first once one has been applied.
///
/// `views` is empty unless `set_consumer_views()` asked for something, so a channel with no
/// per-consumer views produces exactly what it produced before this type existed.
struct render_output
{
    render_result                                        primary;
    std::vector<std::pair<ocio_view_key, render_result>> views;
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

    virtual std::future<render_output> render(const struct video_format_desc& format_desc) = 0;

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

    virtual void set_target_color(color_space cs, color_transfer ct, bool auto_convert, int auto_tone_map = 0, float peak_luminance = 1000.0f, float sdr_reference_white = 100.0f, bool auto_gamut_compress = false, bool straight_alpha_grading = false, bool working_space_composite = false)
    {
        (void)cs;
        (void)ct;
        (void)auto_convert;
        (void)auto_tone_map;
        (void)peak_luminance;
        (void)sdr_reference_white;
        (void)auto_gamut_compress;
        (void)straight_alpha_grading;
        (void)working_space_composite;
    }

    /// The channel's OCIO display/view transform, applied in the post-composite stage.
    ///
    /// Requires `<working-space-composite>`: a display transform consumes WORKING-space
    /// pixels, and without it the composite is already display-encoded by the time this
    /// stage runs. `set_ocio_display` is refused in that case rather than rendering
    /// something plausible from the wrong input.
    ///
    /// Empty display or view clears it.
    virtual void set_ocio_display(const std::string& display, const std::string& view)
    {
        (void)display;
        (void)view;
    }

    virtual ocio_display_state get_ocio_display() const { return {}; }

    /// The channel's LOOK (LMT) -- a creative or technical transform applied in the
    /// working space BEFORE the display rendering. The show LUT of an ACES pipeline.
    ///
    /// Composed into the display processor rather than spliced separately, so it applies
    /// to the primary AND to every consumer view: a look is creative intent, a view is
    /// the screen it goes to, and a consumer asking for a different view still wants the
    /// show's look. It therefore requires a display transform to ride on -- the command
    /// refuses without one rather than silently doing nothing.
    ///
    /// Empty clears it. The string is OCIO's look EXPRESSION, so `-name` inverts and a
    /// comma-separated list applies several in order.
    virtual void set_ocio_look(const std::string& look) { (void)look; }

    virtual std::string get_ocio_look() const { return {}; }

    /// The distinct views this channel's consumers asked for, beyond the channel's own.
    ///
    /// Set once per tick by `mixer`, from what the consumers declare. One post-composite
    /// pass runs per entry over the SAME working-space composite -- one extra full-screen
    /// draw plus one resolve each, and nothing recomposites.
    ///
    /// Ignored unless the channel composites in the working space: without one there is no
    /// composite to fan out from, because every layer has already been converted to the
    /// channel's display space.
    virtual void set_consumer_views(std::vector<ocio_view_key> views) { (void)views; }

    /// Build an OCIO program NOW, off the frame path.
    ///
    /// Called when the AMCP command that selects a transform is accepted, where the OCIO
    /// processor has already been built for validation. Without it the GPU program is
    /// generated, its LUTs uploaded and its GLSL compiled on the first draw that needs it:
    /// ~1.2 s and a dropped frame for an input transform, and worse for a display
    /// transform, whose source is ten times larger.
    ///
    /// Fire and forget: it dispatches to the device thread and does not wait. A caller that
    /// blocked would move the stall from the frame path onto the command, which is the same
    /// stall wearing a different hat.
    virtual void prewarm_ocio(const std::string& source_space,
                              const std::string& display,
                              const std::string& view,
                              const std::string& look = "")
    {
        (void)source_space;
        (void)display;
        (void)view;
        (void)look;
    }

    /// Does this mixer composite in the working space? The AMCP layer asks before
    /// accepting a display transform, so the refusal names the real reason.
    virtual bool composites_in_working_space() const { return false; }

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
