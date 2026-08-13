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
#include <algorithm>
#include "image_mixer.h"

#include "image_kernel.h"
#include "previz_renderer.h"
#include "previz_scene.h"

#include "../util/buffer.h"
#include "../util/device.h"
#include "../util/texture.h"

#include <boost/align/aligned_allocator.hpp>

#include <common/array.h>
#include <common/bit_depth.h>
#include <common/future.h>
#include <common/log.h>
#include <common/render_format.h>

#include <core/frame/frame.h>
#include <core/frame/frame_transform.h>
#include <core/frame/geometry.h>
#include <core/frame/pixel_format.h>
#include <core/video_format.h>

#include <GL/glew.h>

#include <any>
#include <atomic>
#include <vector>

namespace caspar { namespace accelerator { namespace ogl {

using future_texture = std::shared_future<std::shared_ptr<texture>>;

/// Upload futures the frame factory started for a mutable_frame, carried on
/// const_frame::opaque().
///
/// `owner_device` is essential, not decorative: the payload is produced by
/// whichever channel's mixer created the frame, and a routed frame is visited by
/// a *different* channel's mixer -- which may hold another ogl::device (and so
/// another GL context). It used to be a bare vector, so the receiving mixer
/// trusted it blindly.
struct staged_textures
{
    const void*                 owner_device = nullptr;
    std::vector<future_texture> textures;
};

struct item
{
    core::pixel_format_desc     pix_desc = core::pixel_format_desc(core::pixel_format::invalid);
    std::vector<future_texture> textures;
    draw_transforms             transforms;
    core::frame_geometry        geometry = core::frame_geometry::get_default();
};

struct layer
{
    std::vector<layer> sublayers;
    std::vector<item>  items;
    core::blend_mode   blend_mode;

    explicit layer(core::blend_mode blend_mode)
        : blend_mode(blend_mode)
    {
    }
};

// ── Still-frame cache fingerprint ──────────────────────────────────────────
// Everything the composition result depends on must appear here. A missing
// field means a change to it does not invalidate the cache, and the channel
// keeps sending a stale frame — the worst failure this cache can produce.
//
// Holding shared_ptr<texture> (rather than a raw pointer) also keeps the old
// textures alive, so the texture pool cannot recycle an address and make two
// different textures compare equal (ABA).
struct item_fingerprint
{
    std::vector<std::shared_ptr<texture>> textures; // all planes, not just plane 0
    core::image_transform                 transform;
    core::frame_geometry                  geometry   = core::frame_geometry::get_default();
    core::pixel_format_desc               pix_desc   = core::pixel_format_desc(core::pixel_format::invalid);
    core::blend_mode                      blend_mode = core::blend_mode::normal;
    int                                   layer_path = 0; // position in the layer/sublayer tree

    bool operator==(const item_fingerprint& other) const
    {
        return textures == other.textures && transform == other.transform && geometry == other.geometry &&
               pix_desc == other.pix_desc && blend_mode == other.blend_mode && layer_path == other.layer_path;
    }
    bool operator!=(const item_fingerprint& other) const { return !(*this == other); }
};

struct render_fingerprint
{
    std::vector<item_fingerprint> items;

    // True only when every texture future was already resolved. An unresolved
    // future reads as nullptr, so two different frames could otherwise compare
    // equal while still uploading; an incomplete fingerprint never matches.
    bool complete = false;

    // Channel-wide state the kernel binds. Changing any of these changes the
    // output for identical inputs.
    int                  target_width           = 0;
    int                  target_height          = 0;
    core::color_space    target_color_space     = core::color_space::bt709;
    core::color_transfer target_color_transfer  = core::color_transfer::sdr;
    bool                 auto_color_convert     = true;
    int                  auto_tone_map          = 0;
    float                display_peak_luminance = 0.0f;
    float                sdr_reference_white    = 0.0f;
    bool                 auto_gamut_compress    = false;
    bool                 straight_alpha_grading = false;
    bool                 working_space_composite = false;
    std::string          ocio_display;
    std::string          ocio_view;
    const void*          calibration_lut        = nullptr;
    float                calibration_strength   = 0.0f;
    bool                 calibration_bypass     = false;

    bool matches(const render_fingerprint& other) const
    {
        return complete && other.complete && items == other.items && target_width == other.target_width &&
               target_height == other.target_height && target_color_space == other.target_color_space &&
               target_color_transfer == other.target_color_transfer &&
               auto_color_convert == other.auto_color_convert && auto_tone_map == other.auto_tone_map &&
               display_peak_luminance == other.display_peak_luminance &&
               sdr_reference_white == other.sdr_reference_white && auto_gamut_compress == other.auto_gamut_compress &&
               straight_alpha_grading == other.straight_alpha_grading &&
               working_space_composite == other.working_space_composite &&
               ocio_display == other.ocio_display && ocio_view == other.ocio_view &&
               calibration_lut == other.calibration_lut && calibration_strength == other.calibration_strength &&
               calibration_bypass == other.calibration_bypass;
    }
};

class image_renderer
{
    spl::shared_ptr<device> ogl_;
    image_kernel            kernel_;
    const size_t            max_frame_size_;
    common::bit_depth       depth_;
    std::atomic<bool>       cpu_readback_needed_{true};

    // The numeric format of every intermediate this renderer composites into. Distinct
    // from depth_, which stays the channel's *output* depth and is what consumers are told
    // (see A0.2's resolve pass): the float working space lives inside the mixer only.
    // unorm keeps every existing configuration bit-identical.
    common::render_format render_format_ = common::render_format::unorm;

    // Still-frame cache: skip GPU composition when inputs are unchanged.
    render_fingerprint                            prev_fingerprint_;
    std::shared_ptr<core::texture>                cached_texture_;
    std::shared_future<array<const std::uint8_t>> cached_cpu_;
    //: The cached tick's extra views, cached alongside the primary rather than recomputed.
    //: Compared by COUNT before the cache is trusted: a consumer attaching or detaching
    //: changes the view set without changing a single layer, so the fingerprint alone would
    //: happily replay a tick that no longer has the right outputs in it.
    std::vector<std::pair<core::ocio_view_key, core::render_result>> cached_views_;

  public:
    core::color_space    target_color_space    = core::color_space::bt709;
    core::color_transfer target_color_transfer = core::color_transfer::sdr;
    bool                 auto_color_convert    = true;
    int                  auto_tone_map         = 0;
    float                display_peak_luminance = 1000.0f;
    float                sdr_reference_white    = 100.0f;
    bool                 auto_gamut_compress    = false;
    bool                 straight_alpha_grading = false;
    bool                 working_space_composite = false;

    // The channel's OCIO display/view transform, applied in the post-composite stage.
    // Empty means none. Guarded by working_space_composite: a display transform consumes
    // working-space pixels.
    std::string          ocio_display;
    std::string          ocio_view;

    //: Distinct views the consumers asked for, beyond the channel's own. Set once per tick.
    std::vector<core::ocio_view_key> consumer_views_;

    // Channel-master LED-wall calibration LUT, applied as a final full-screen
    // pass over the composited frame (output-agnostic — every consumer sees it).
    std::shared_ptr<const core::lut3d_data> calibration_lut_;
    float                                   calibration_strength_ = 1.0f;
    bool                                    calibration_bypass_   = false;

    void set_calibration_lut(std::shared_ptr<const core::lut3d_data> lut, float strength)
    {
        calibration_lut_      = std::move(lut);
        calibration_strength_ = strength;
        // Invalidate the still-frame cache so the new LUT takes effect immediately.
        prev_fingerprint_ = {};
        cached_texture_.reset();
        cached_cpu_ = {};
    }

    void set_calibration_bypass(bool bypass)
    {
        calibration_bypass_ = bypass;
        prev_fingerprint_ = {};
        cached_texture_.reset();
        cached_cpu_ = {};
    }

    explicit image_renderer(const spl::shared_ptr<device>& ogl,
                            const size_t                   max_frame_size,
                            common::bit_depth              depth,
                            common::render_format          render_format = common::render_format::unorm)
        : ogl_(ogl)
        , kernel_(ogl_)
        , max_frame_size_(max_frame_size)
        , depth_(depth)
        , render_format_(render_format)
    {
        if (render_format_ != common::render_format::unorm) {
            CASPAR_LOG(info) << L"[ogl_renderer] compositing into fp16 render targets";
        }
    }

    common::render_format render_format() const { return render_format_; }

    void set_cpu_readback_needed(bool needed)
    {
        bool was = cpu_readback_needed_.exchange(needed, std::memory_order_relaxed);
        // When transitioning from GPU-only to CPU-needed (e.g. IMAGE consumer
        // added dynamically), invalidate the still-frame cache so the next
        // render actually performs the GPU→CPU readback instead of returning
        // the stale empty buffer from the previous cached result.
        if (needed && !was) {
            prev_fingerprint_ = {};
        }
    }

    std::future<core::render_output>
    operator()(std::vector<layer> layers, const core::video_format_desc& format_desc)
    {
        if (layers.empty()) { // Bypass GPU with empty frame.
            // Release cached textures so VRAM from the last rendered frame is freed.
            prev_fingerprint_ = {};
            cached_texture_.reset();
            cached_cpu_ = {};

            static const std::vector<uint8_t, boost::alignment::aligned_allocator<uint8_t, 32>> buffer(max_frame_size_,
                                                                                                       0);
            auto ready = make_ready_future<array<const std::uint8_t>>(
                array<const std::uint8_t>(buffer.data(), format_desc.size, true));
            core::render_output empty;
            empty.primary = {ready.share(), nullptr};
            return make_ready_future<core::render_output>(std::move(empty));
        }

        // ── Still-frame cache ──────────────────────────────────────────────
        // When every input to composition is identical to the previous tick,
        // skip the GPU work entirely and reuse the cached output.
        {
            auto fingerprint = build_fingerprint(layers, format_desc);

            // The cache holds the WHOLE tick, views included. Caching only the primary
            // and replaying it for every view would hand one view another's picture --
            // silently, and only on the ticks where nothing moved.
            if (!fingerprint.items.empty() && fingerprint.matches(prev_fingerprint_) && cached_texture_ &&
                cached_views_.size() == consumer_views_.size()) {
                layers.clear();
                core::render_output cached;
                cached.primary = {cached_cpu_, cached_texture_};
                cached.views   = cached_views_;
                return make_ready_future<core::render_output>(std::move(cached));
            }
            prev_fingerprint_ = std::move(fingerprint);
        }

        auto needs_cpu = cpu_readback_needed_.load(std::memory_order_relaxed);

        // Snapshot the calibration LUT state for this tick (stable across the
        // async render).
        auto cal_lut      = calibration_lut_;
        auto cal_strength = calibration_strength_;
        auto cal_bypass   = calibration_bypass_;
        auto ws_composite = working_space_composite;
        // Only meaningful under a working-space composite: without one there is nothing to
        // fan out FROM, because every layer has already been converted to the channel's
        // display space and a second view would be a second encoding of an encoded picture.
        auto views        = ws_composite ? consumer_views_ : std::vector<core::ocio_view_key>{};
        auto ch_display   = ocio_display;
        auto ch_view      = ocio_view;

        auto f = std::move(
            ogl_->dispatch_async([=, layers = std::move(layers)]() mutable -> core::render_output {
                auto target_texture =
                    ogl_->create_texture(format_desc.width, format_desc.height, 4, depth_, true, render_format_);
                draw(target_texture, std::move(layers), format_desc);

                // THE WORKING-SPACE COMPOSITE, before anything encodes it. Every view below
                // starts from this same texture, which is the whole reason the fan-out lives
                // here: a display transform is not invertible, so a second view cannot be
                // derived from the first once one has been applied.
                auto composite = target_texture;

                // Everything after the composite, for ONE view. The calibration LUT is
                // inside because it is authored against display values, so each view needs
                // its own; the resolve is inside because it is per texture.
                // `texture` unqualified is ogl::texture here, which is what the three
                // helpers below take by reference; core::render_result stores the base
                // pointer, so the conversion happens once, on return.
                auto finish = [&](std::shared_ptr<texture> tex,
                                  const std::string& disp,
                                  const std::string& vw) -> core::render_result {
                    if (ws_composite) {
                        auto oc = ogl_->create_texture(format_desc.width, format_desc.height, 4,
                                                       depth_, true, render_format_);
                        apply_output_convert(tex, oc, format_desc, disp, vw);
                        tex = oc;
                    }
                    if (cal_lut && !cal_bypass && cal_lut->size > 0) {
                        auto cal = ogl_->create_texture(format_desc.width, format_desc.height, 4,
                                                        depth_, true, render_format_);
                        apply_calibration_lut(tex, cal, format_desc, cal_lut, cal_strength);
                        tex = cal;
                    }
                    if (render_format_ != common::render_format::unorm) {
                        tex = resolve_to_output(tex, format_desc);
                    }
                    if (!needs_cpu) {
                        auto e = make_ready_future<array<const std::uint8_t>>(
                            array<const std::uint8_t>(nullptr, 0, true));
                        return {e.share(), tex};
                    }
                    return {ogl_->copy_async(tex).share(), tex};
                };

                core::render_output out;
                out.primary = finish(composite, ch_display, ch_view);
                for (const auto& v : views)
                    out.views.emplace_back(v, finish(composite, v.display, v.view));
                return out;

            }));

        return std::async(
            std::launch::deferred,
            [this, f = std::move(f)]() mutable -> core::render_output {
                auto out = std::move(f.get());
                // Update the still-frame cache with the freshly rendered result -- views
                // included, so a cached tick replays every view rather than the primary
                // alone.
                cached_cpu_     = out.primary.image;
                cached_texture_ = out.primary.texture;
                cached_views_   = out.views;
                return out;
            });
    }

    common::bit_depth depth() const { return depth_; }

    /// Forward to the kernel, which owns the variant cache. Called on the GL thread.
    void prewarm_ocio(const std::string& source_space, const std::string& display, const std::string& view)
    {
        kernel_.prewarm_ocio(source_space, display, view);
    }

  private:
    /// Collects the full description of what composition would draw, including
    /// sublayers (which the previous fingerprint ignored entirely, so a change
    /// inside a sublayer left a stale frame on air).
    render_fingerprint build_fingerprint(const std::vector<layer>&      layers,
                                        const core::video_format_desc& format_desc) const
    {
        render_fingerprint fp;
        fp.complete               = true;
        fp.target_width           = format_desc.square_width;
        fp.target_height          = format_desc.square_height;
        fp.target_color_space     = target_color_space;
        fp.target_color_transfer  = target_color_transfer;
        fp.auto_color_convert     = auto_color_convert;
        fp.auto_tone_map          = auto_tone_map;
        fp.display_peak_luminance = display_peak_luminance;
        fp.sdr_reference_white    = sdr_reference_white;
        fp.auto_gamut_compress    = auto_gamut_compress;
        fp.straight_alpha_grading = straight_alpha_grading;
        fp.working_space_composite = working_space_composite;
        fp.ocio_display           = ocio_display;
        fp.ocio_view              = ocio_view;
        fp.calibration_lut        = calibration_lut_.get();
        fp.calibration_strength   = calibration_strength_;
        fp.calibration_bypass     = calibration_bypass_;

        int path = 0;
        std::function<void(const std::vector<layer>&)> collect = [&](const std::vector<layer>& ls) {
            for (auto& l : ls) {
                ++path;
                collect(l.sublayers);
                for (auto& itm : l.items) {
                    item_fingerprint ifp;
                    ifp.transform  = itm.transforms.image_transform;
                    ifp.geometry   = itm.geometry;
                    ifp.pix_desc   = itm.pix_desc;
                    ifp.blend_mode = l.blend_mode;
                    ifp.layer_path = path;

                    for (auto& tex : itm.textures) {
                        if (tex.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
                            // Still uploading: an unresolved future would read as
                            // nullptr and could match a different frame.
                            fp.complete = false;
                            ifp.textures.clear();
                            break;
                        }
                        ifp.textures.push_back(tex.get());
                    }

                    fp.items.push_back(std::move(ifp));
                }
            }
        };
        collect(layers);

        return fp;
    }

    void draw(std::shared_ptr<texture>&      target_texture,
              std::vector<layer>             layers,
              const core::video_format_desc& format_desc)
    {
        std::shared_ptr<texture> layer_key_texture;

        for (auto& layer : layers) {
            draw(target_texture, layer.sublayers, format_desc);
            draw(target_texture, std::move(layer), layer_key_texture, format_desc);
        }
    }

    void draw(std::shared_ptr<texture>&      target_texture,
              layer                          layer,
              std::shared_ptr<texture>&      layer_key_texture,
              const core::video_format_desc& format_desc)
    {
        if (layer.items.empty())
            return;

        std::shared_ptr<texture> local_key_texture;
        std::shared_ptr<texture> local_mix_texture;

        if (layer.blend_mode != core::blend_mode::normal) {
            auto layer_texture = ogl_->create_texture(
                target_texture->width(), target_texture->height(), 4, depth_, true, render_format_);

            for (auto& item : layer.items)
                draw(layer_texture,
                     std::move(item),
                     layer_key_texture,
                     local_key_texture,
                     local_mix_texture,
                     format_desc);

            draw(layer_texture, std::move(local_mix_texture), format_desc, core::blend_mode::normal);
            draw(target_texture, std::move(layer_texture), format_desc, layer.blend_mode);
        } else // fast path
        {
            for (auto& item : layer.items)
                draw(target_texture,
                     std::move(item),
                     layer_key_texture,
                     local_key_texture,
                     local_mix_texture,
                     format_desc);

            draw(target_texture, std::move(local_mix_texture), format_desc, core::blend_mode::normal);
        }

        layer_key_texture = std::move(local_key_texture);
    }

    void draw(std::shared_ptr<texture>&      target_texture,
              item                           item,
              std::shared_ptr<texture>&      layer_key_texture,
              std::shared_ptr<texture>&      local_key_texture,
              std::shared_ptr<texture>&      local_mix_texture,
              const core::video_format_desc& format_desc)
    {
        draw_params draw_params;
        draw_params.target_width  = format_desc.square_width;
        // A custom channel format is an LED wall or projector, not an SD broadcast
        // destination; the kernel uses this to stop a small raster implying BT.601.
        draw_params.target_is_custom_format =
            format_desc.format == core::video_format::custom;
        draw_params.target_height = format_desc.square_height;
        draw_params.target_color_space    = target_color_space;
        draw_params.target_color_transfer = target_color_transfer;
        draw_params.auto_color_convert    = auto_color_convert;
        draw_params.auto_tone_map         = auto_tone_map;
        draw_params.display_peak_luminance = display_peak_luminance;
        draw_params.sdr_reference_white    = sdr_reference_white;
        draw_params.auto_gamut_compress    = auto_gamut_compress;
        draw_params.straight_alpha_grading = straight_alpha_grading;
        draw_params.working_space_composite = working_space_composite;

        draw_params.pix_desc   = std::move(item.pix_desc);
        draw_params.transforms = std::move(item.transforms);
        draw_params.geometry   = std::move(item.geometry);
        draw_params.aspect_ratio =
            static_cast<double>(format_desc.square_width) / static_cast<double>(format_desc.square_height);

        for (auto& future_texture : item.textures) {
            draw_params.textures.push_back(spl::make_shared_ptr(future_texture.get()));
        }

        if (draw_params.transforms.image_transform
                .is_key) { // A key means we will use it for the next non-key item as a mask
            local_key_texture =
                local_key_texture ? local_key_texture
                                  : ogl_->create_texture(
                                        target_texture->width(), target_texture->height(), 1, depth_, true, render_format_);

            draw_params.background = local_key_texture;
            draw_params.local_key  = nullptr;
            draw_params.layer_key  = nullptr;

            kernel_.draw(std::move(draw_params));
        } else if (draw_params.transforms.image_transform
                       .is_mix) { // A mix means precomp the items to a texture, before drawing to the channel
            local_mix_texture =
                local_mix_texture ? local_mix_texture
                                  : ogl_->create_texture(
                                        target_texture->width(), target_texture->height(), 4, depth_, true, render_format_);

            draw_params.background = local_mix_texture;
            draw_params.local_key  = std::move(local_key_texture); // Use and reset the key
            draw_params.layer_key  = layer_key_texture;

            draw_params.keyer = keyer::additive;

            kernel_.draw(std::move(draw_params));
        } else {
            // If there is a mix, this is the end so draw it and reset
            draw(target_texture, std::move(local_mix_texture), format_desc, core::blend_mode::normal);

            draw_params.background = target_texture;
            draw_params.local_key  = std::move(local_key_texture);
            draw_params.layer_key  = layer_key_texture;

            kernel_.draw(std::move(draw_params));
        }
    }

    void draw(std::shared_ptr<texture>&  target_texture,
              std::shared_ptr<texture>&& source_texture,
              core::video_format_desc    format_desc,
              core::blend_mode           blend_mode = core::blend_mode::normal)
    {
        if (!source_texture)
            return;

        draw_params draw_params;
        draw_params.target_width    = format_desc.square_width;
        // A custom channel format is an LED wall or projector, not an SD broadcast
        // destination; the kernel uses this to stop a small raster implying BT.601.
        draw_params.target_is_custom_format =
            format_desc.format == core::video_format::custom;
        draw_params.target_height   = format_desc.square_height;
        draw_params.pix_desc.format = core::pixel_format::bgra;
        draw_params.pix_desc.planes = {core::pixel_format_desc::plane(
            source_texture->width(), source_texture->height(), 4, source_texture->depth())};
        draw_params.textures        = {spl::make_shared_ptr(source_texture)};
        draw_params.blend_mode      = blend_mode;
        draw_params.background      = target_texture;
        draw_params.geometry        = core::frame_geometry::get_default();

        kernel_.draw(std::move(draw_params));
    }

    // Channel-master calibration LUT: full-screen pass that copies the composited
    // frame through a 3D LUT into a fresh texture. The source is tagged with the
    // channel's output colour space so the kernel performs NO colour conversion —
    // only the calibration LUT runs (display-to-display correction).
    void apply_calibration_lut(std::shared_ptr<texture>&                      source_texture,
                               std::shared_ptr<texture>&                      target_texture,
                               const core::video_format_desc&                 format_desc,
                               const std::shared_ptr<const core::lut3d_data>& lut,
                               float                                          strength)
    {
        if (!source_texture || !lut)
            return;

        draw_params draw_params;
        draw_params.target_width    = format_desc.square_width;
        // A custom channel format is an LED wall or projector, not an SD broadcast
        // destination; the kernel uses this to stop a small raster implying BT.601.
        draw_params.target_is_custom_format =
            format_desc.format == core::video_format::custom;
        draw_params.target_height   = format_desc.square_height;
        draw_params.pix_desc.format = core::pixel_format::bgra;
        draw_params.pix_desc.planes = {core::pixel_format_desc::plane(
            source_texture->width(), source_texture->height(), 4, source_texture->depth())};
        draw_params.pix_desc.color_space    = target_color_space;
        draw_params.pix_desc.color_transfer = target_color_transfer;
        draw_params.target_color_space      = target_color_space;
        draw_params.target_color_transfer   = target_color_transfer;
        draw_params.auto_color_convert      = false;
        draw_params.auto_tone_map           = 0;
        draw_params.textures                = {spl::make_shared_ptr(source_texture)};
        draw_params.blend_mode              = core::blend_mode::normal;
        draw_params.background              = target_texture;
        draw_params.geometry               = core::frame_geometry::get_default();
        draw_params.transforms.image_transform.lut3d          = lut;
        draw_params.transforms.image_transform.lut3d_strength = strength;

        kernel_.draw(std::move(draw_params));
    }

    /// The channel's post-composite output conversion.
    ///
    /// The other half of `working_space_composite`: every layer reached the composite in
    /// scene-linear ACEScg with its output half suppressed, so the display encoding -- tone
    /// map, k_to_output, clamp, OETF -- is applied ONCE here, to the composite.
    ///
    /// Same shape as `apply_calibration_lut` above, which is the precedent this follows
    /// rather than invents. `output_convert_only` routes the kernel into the branch the OCIO
    /// input transform already uses, which is exactly this configuration.
    void apply_output_convert(std::shared_ptr<texture>&      source_texture,
                              std::shared_ptr<texture>&      target_texture,
                              const core::video_format_desc& format_desc,
                              const std::string&             display,
                              const std::string&             view)
    {
        if (!source_texture)
            return;

        draw_params draw_params;
        draw_params.target_width  = format_desc.square_width;
        draw_params.target_height = format_desc.square_height;
        draw_params.target_is_custom_format = format_desc.format == core::video_format::custom;
        draw_params.pix_desc.format = core::pixel_format::bgra;
        draw_params.pix_desc.planes = {core::pixel_format_desc::plane(
            source_texture->width(), source_texture->height(), 4, source_texture->depth())};
        draw_params.pix_desc.color_space    = target_color_space;
        draw_params.pix_desc.color_transfer = target_color_transfer;
        draw_params.target_color_space      = target_color_space;
        draw_params.target_color_transfer   = target_color_transfer;
        draw_params.auto_tone_map           = auto_tone_map;
        draw_params.display_peak_luminance  = display_peak_luminance;
        draw_params.sdr_reference_white     = sdr_reference_white;
        // Off, both of them: this draw is the output half and nothing else. auto_color_convert
        // would try to own both halves, and working_space_composite belongs to LAYER draws --
        // setting it here would suppress the very conversion this pass exists to apply.
        draw_params.auto_color_convert      = false;
        draw_params.working_space_composite = false;
        draw_params.output_convert_only     = true;
        // The display transform, if any, owns the output half of THIS pass: the kernel's
        // `ocio_out` check clears do_output_convert and splices the generated program in
        // its place. Set here and nowhere else -- a layer draw with a display transform
        // would encode each layer separately, which is the per-layer arrangement this
        // stage exists to replace.
        // Whichever view THIS pass is for -- the channel's own, or a consumer's. Empty
        // leaves the built-in output conversion in charge, which is the no-display-transform
        // case that `output_convert_only` above already set up.
        draw_params.ocio_display            = display;
        draw_params.ocio_view               = view;
        // Composes with the alpha-domain fix: with straight-alpha grading on, the output
        // encoding is applied to the straight colour and re-premultiplied, which is the
        // same rule every layer draw follows.
        draw_params.straight_alpha_grading  = straight_alpha_grading;
        draw_params.textures                = {spl::make_shared_ptr(source_texture)};
        draw_params.blend_mode              = core::blend_mode::normal;
        draw_params.background              = target_texture;
        draw_params.geometry                = core::frame_geometry::get_default();

        kernel_.draw(std::move(draw_params));
    }

    /// Convert a float render target into the channel's output depth.
    ///
    /// Mandatory whenever render_format_ is not unorm, and not an optimisation: the
    /// channel hands consumers `array<const uint8_t>` from copy_async() and a
    /// core::texture they may read directly, and both of those interfaces mean integer.
    /// A half-float target reaching a DeckLink, screen, ffmpeg or GPU-direct consumer
    /// would be reinterpreted as unsigned shorts -- not clipped, not dim, but garbage,
    /// because the bit patterns are unrelated.
    ///
    /// The conversion itself is a straight blit through the kernel with no colour work.
    /// That is correct rather than lazy: the OGL chain applies the OETF at the end of the
    /// same fragment pass that composites a layer (COLOR_GRADING.md steps 27-29), so the
    /// composite target already holds display-encoded values. All the float target adds is
    /// that whatever fell outside [0,1] survived being written down. Writing into a
    /// normalized target clamps it here, which is exactly where the display range should
    /// be imposed.
    ///
    /// This is also where an OCIO display/view transform belongs, so that a channel has
    /// one ordered output chain (composite -> calibration -> display transform -> resolve)
    /// rather than several competing full-screen passes.
    std::shared_ptr<texture> resolve_to_output(std::shared_ptr<texture>&      source_texture,
                                               const core::video_format_desc& format_desc)
    {
        auto out = ogl_->create_texture(
            format_desc.width, format_desc.height, 4, depth_, true, common::render_format::unorm);

        draw_params draw_params;
        draw_params.target_width  = format_desc.square_width;
        draw_params.target_height = format_desc.square_height;
        draw_params.target_is_custom_format = format_desc.format == core::video_format::custom;
        draw_params.pix_desc.format = core::pixel_format::bgra;
        draw_params.pix_desc.planes = {core::pixel_format_desc::plane(
            source_texture->width(), source_texture->height(), 4, source_texture->depth())};
        draw_params.pix_desc.color_space    = target_color_space;
        draw_params.pix_desc.color_transfer = target_color_transfer;
        draw_params.target_color_space      = target_color_space;
        draw_params.target_color_transfer   = target_color_transfer;
        // No colour conversion and no tone mapping: the picture is already in the
        // channel's output encoding by this point. Enabling either here would apply a
        // second transform to an already-encoded image.
        draw_params.auto_color_convert = false;
        draw_params.auto_tone_map      = 0;
        draw_params.textures           = {spl::make_shared_ptr(source_texture)};
        draw_params.blend_mode         = core::blend_mode::normal;
        draw_params.background         = out;
        draw_params.geometry           = core::frame_geometry::get_default();

        kernel_.draw(std::move(draw_params));

        return out;
    }
};

struct image_mixer::impl
    : public core::frame_factory
    , public std::enable_shared_from_this<impl>
{
    spl::shared_ptr<device>      ogl_;
    int                          channel_id_;
    image_renderer               renderer_;
    previz_renderer              previz_renderer_;
    std::shared_ptr<channel_texture_store> channel_tex_store_;
    std::vector<draw_transforms> transform_stack_;
    std::vector<layer>           layers_; // layer/stream/items
    std::vector<layer*>          layer_stack_;

    // One-shot warnings: these paths cost a readback, or drop an item, on every
    // frame -- so they must be visible, but not once per frame.
    std::atomic<bool> foreign_texture_logged_{false};
    std::atomic<bool> no_source_logged_{false};

    double aspect_ratio_ = 1.0;

  public:
    impl(const spl::shared_ptr<device>& ogl,
         const int                      channel_id,
         const size_t                   max_frame_size,
         common::bit_depth              depth,
         common::render_format          render_format)
        : ogl_(ogl)
        , channel_id_(channel_id)
        , renderer_(ogl, max_frame_size, depth, render_format)
        , previz_renderer_(ogl)
        , transform_stack_(1)
    {
        CASPAR_LOG(info) << L"Initialized OpenGL Accelerated GPU Image Mixer for channel " << channel_id;
    }

    void update_aspect_ratio(double aspect_ratio) { aspect_ratio_ = aspect_ratio; }

    void set_target_color(core::color_space cs, core::color_transfer ct, bool auto_convert, int auto_tone_map, float peak_luminance, float sdr_ref_white, bool gamut_compress, bool straight_alpha, bool ws_composite)
    {
        CASPAR_LOG(trace) << L"[ogl_mixer] set_target_color cs=" << static_cast<int>(cs)
                          << L" ct=" << static_cast<int>(ct) << L" auto=" << auto_convert
                          << L" tone_map=" << auto_tone_map << L" peak_lum=" << peak_luminance
                          << L" sdr_ref_white=" << sdr_ref_white << L" gamut_compress=" << gamut_compress
                          << L" straight_alpha_grading=" << straight_alpha;
        renderer_.target_color_space    = cs;
        renderer_.target_color_transfer = ct;
        renderer_.auto_color_convert    = auto_convert;
        renderer_.auto_tone_map         = auto_tone_map;
        renderer_.display_peak_luminance = peak_luminance;
        renderer_.sdr_reference_white    = sdr_ref_white;
        renderer_.auto_gamut_compress    = gamut_compress;
        renderer_.straight_alpha_grading = straight_alpha;
        renderer_.working_space_composite = ws_composite;
    }

    void set_ocio_display(const std::string& display, const std::string& view)
    {
        CASPAR_LOG(info) << L"[mixer] set_ocio_display display=\"" << u16(display)
                         << L"\" view=\"" << u16(view) << L"\"";
        renderer_.ocio_display = display;
        renderer_.ocio_view    = view;
    }

    void set_consumer_views(std::vector<core::ocio_view_key> views)
    {
        // Pre-warm anything NEW since the last tick.
        //
        // This is called every tick, so the comparison is what stops it dispatching a build
        // 25 times a second -- the build itself would be a cache hit, but the dispatch is
        // not free and the log line would be.
        //
        // Without it a consumer's view is the one path left compiling on the frame path:
        // measured 2026-08-13, `OCIO_DISPLAY` pre-warmed correctly while a consumer's view
        // still logged "compiling an OCIO program ON THE FRAME PATH" -- the channel had a
        // command to hang the pre-warm on and the consumer had none.
        for (const auto& v : views) {
            if (std::find(prewarmed_views_.begin(), prewarmed_views_.end(), v) == prewarmed_views_.end()) {
                prewarm_ocio("", v.display, v.view);
                prewarmed_views_.push_back(v);
            }
        }
        renderer_.consumer_views_ = std::move(views);
    }

    //: Views already pre-warmed. Grows only; a view that goes away has its program cached in
    //: the kernel anyway, so re-warming it later would be a no-op and forgetting it would
    //: cost a frame if the consumer came back.
    std::vector<core::ocio_view_key> prewarmed_views_;

    void prewarm_ocio(const std::string& source_space, const std::string& display, const std::string& view)
    {
        // On the GL thread, because it creates textures and compiles a program -- and
        // asynchronously, so the AMCP command returns immediately. The compile still costs
        // what it costs; the point is that it no longer costs a FRAME.
        ogl_->dispatch_async([this, source_space, display, view] {
            renderer_.prewarm_ocio(source_space, display, view);
        });
    }

    core::ocio_display_state get_ocio_display() const
    {
        core::ocio_display_state st;
        st.display = renderer_.ocio_display;
        st.view    = renderer_.ocio_view;
        st.enabled = !st.display.empty() && !st.view.empty();
        return st;
    }

    std::wstring calibration_path_;

    void set_calibration_lut(std::shared_ptr<const core::lut3d_data> lut, float strength, const std::wstring& path)
    {
        CASPAR_LOG(info) << L"[ogl_mixer] set_calibration_lut size="
                         << (lut ? lut->size : 0) << L" strength=" << strength
                         << L" path=" << path;
        renderer_.set_calibration_lut(std::move(lut), strength);
        calibration_path_ = path;
    }

    void set_calibration_bypass(bool bypass)
    {
        CASPAR_LOG(info) << L"[ogl_mixer] set_calibration_bypass " << bypass;
        renderer_.set_calibration_bypass(bypass);
    }

    core::calibration_lut_state get_calibration_state() const
    {
        core::calibration_lut_state s;
        s.enabled  = static_cast<bool>(renderer_.calibration_lut_) && renderer_.calibration_lut_->size > 0;
        s.bypass   = renderer_.calibration_bypass_;
        s.size     = renderer_.calibration_lut_ ? renderer_.calibration_lut_->size : 0;
        s.strength = renderer_.calibration_strength_;
        s.path     = calibration_path_;
        return s;
    }

    void push(const core::frame_transform& transform)
    {
        auto previous_layer_depth = transform_stack_.back().image_transform.layer_depth;

        transform_stack_.push_back(transform_stack_.back().combine_transform(transform.image_transform, aspect_ratio_));

        auto new_layer_depth = transform_stack_.back().image_transform.layer_depth;

        if (previous_layer_depth < new_layer_depth) {
            layer new_layer(transform_stack_.back().image_transform.blend_mode);

            if (layer_stack_.empty()) {
                layers_.push_back(std::move(new_layer));
                layer_stack_.push_back(&layers_.back());
            } else {
                layer_stack_.back()->sublayers.push_back(std::move(new_layer));
                layer_stack_.push_back(&layer_stack_.back()->sublayers.back());
            }
        }
    }

    /// Resolves the source textures for `item`, in one fixed order shared by both
    /// mixer backends:
    ///
    ///   1. a GPU texture owned by *this* device      -> zero copy
    ///   2. a GPU texture owned by another device     -> import not implemented;
    ///                                                   falls through to host
    ///   3. pre-staged upload futures on opaque()     -> upload already started
    ///   4. host planes                               -> upload now
    ///   5. nothing usable                            -> drop the item
    ///
    /// Returns false when nothing usable was found, in which case the item must
    /// be dropped rather than pushed with an empty texture list.
    ///
    /// This replaced three divergent branches per backend, each with its own
    /// (and differently worded) fallback behaviour. Keeping the decision order in
    /// one place per backend is what makes "why did this frame not draw?"
    /// answerable.
    bool resolve_item_textures(item& item, const core::const_frame& frame)
    {
        const auto host_state = frame.host_image_state();

        if (!frame.textures().empty()) {
            // Every plane must belong to this device; a partially-usable set is
            // not usable at all, so check them all before binding any.
            bool all_mine = true;
            for (auto& core_tex : frame.textures()) {
                auto native = std::dynamic_pointer_cast<texture>(core_tex);
                if (!native || core_tex->owner_device() == nullptr ||
                    core_tex->owner_device() != static_cast<const void*>(ogl_.get())) {
                    all_mine = false;
                    break;
                }
            }

            if (all_mine) {
                for (auto& core_tex : frame.textures()) {
                    auto native = std::static_pointer_cast<texture>(core_tex);
                    item.textures.emplace_back(make_ready_future(std::shared_ptr<texture>(native)).share());
                }
                return true;
            }

            // (2) would import the foreign allocation via external memory. Not
            // implemented; until it is, fall through to the host path so the
            // frame still draws, and say so once because it costs a readback.
            if (!foreign_texture_logged_.exchange(true)) {
                CASPAR_LOG(warning) << L"[image_mixer]" L" frame GPU planes are not usable on this device "
                                       L"(different ogl::device or backend); falling back to a host upload.";
            }
        }

        // (3) The frame factory already started uploads for this frame.
        if (frame.opaque().has_value()) {
            // any_cast to a pointer type yields nullptr on a type mismatch rather
            // than throwing, which is what we want when the payload came from the
            // other backend.
            if (auto staged = std::any_cast<std::shared_ptr<staged_textures>>(&frame.opaque())) {
                if (*staged && (*staged)->owner_device == static_cast<const void*>(ogl_.get())) {
                    item.textures = (*staged)->textures;
                    return true;
                }
                if (!foreign_texture_logged_.exchange(true)) {
                    CASPAR_LOG(warning) << L"[image_mixer] pre-staged upload belongs to another device; "
                                           L"falling back to a host upload.";
                }
            }
        }

        // (4) Host planes. `unavailable` means there are none and none coming --
        // typically a GPU-only frame whose readback was skipped -- so uploading
        // would sample a null pointer.
        if (host_state == core::host_image_availability::unavailable) {
            if (!no_source_logged_.exchange(true)) {
                CASPAR_LOG(warning) << L"[image_mixer]" L" item has no usable GPU texture and no host pixels "
                                       L"(readback was skipped); dropping it.";
            }
            return false;
        }

        for (int n = 0; n < static_cast<int>(item.pix_desc.planes.size()); ++n) {
            const auto& plane = frame.image_data(n);
            if (plane.size() == 0 || plane.data() == nullptr) {
                if (!no_source_logged_.exchange(true)) {
                    CASPAR_LOG(warning) << L"[image_mixer]" L" host plane " << n << L" is empty; dropping this item.";
                }
                item.textures.clear();
                return false;
            }
            item.textures.emplace_back(ogl_->copy_async(plane,
                                                         item.pix_desc.planes[n].width,
                                                         item.pix_desc.planes[n].height,
                                                         item.pix_desc.planes[n].stride,
                                                         item.pix_desc.planes[n].depth));
        }

        return true;
    }

    void visit(const core::const_frame& frame)
    {
        if (frame.pixel_format_desc().format == core::pixel_format::invalid)
            return;

        if (frame.pixel_format_desc().planes.empty())
            return;

        item item;
        item.pix_desc   = frame.pixel_format_desc();
        item.transforms = transform_stack_.back();
        item.geometry   = item.transforms.image_transform.geometry_override.has_value()
                              ? *item.transforms.image_transform.geometry_override
                              : frame.geometry();

        if (!resolve_item_textures(item, frame))
            return;

        layer_stack_.back()->items.push_back(item);
    }

    void pop()
    {
        transform_stack_.pop_back();
        layer_stack_.resize(transform_stack_.back().image_transform.layer_depth);
    }

    std::future<core::render_output>
    render(const core::video_format_desc& format_desc)
    {
        // If previz is active, first do normal 2D compositing to post this
        // channel's output to the texture store (so other screens — including
        // screens mapped to *this* channel — can sample it), then render the
        // 3D scene for the previz viewport.
        if (previz_renderer_.active() && channel_tex_store_) {
            auto  store  = channel_tex_store_;
            auto  ch_id  = channel_id_;

            // Normal compositing first — produces this channel's flat output.
            // renderer_() internally dispatches GL work via ogl_, so we must
            // resolve its future BEFORE entering another ogl_->dispatch_async
            // to avoid deadlocking the single GL thread.
            auto composited = renderer_(std::move(layers_), format_desc);

            return std::async(
                std::launch::deferred,
                [this, composited = std::move(composited), store, ch_id, format_desc]() mutable
                -> core::render_output {
                    // Resolve composited result (runs the queued GL work)
                    auto comp_out  = composited.get();
                    auto& comp_tex = comp_out.primary.texture;
                    if (comp_tex) {
                        auto* ogl_tex = dynamic_cast<ogl::texture*>(comp_tex.get());
                        if (ogl_tex)
                            store->update(ch_id, comp_tex,
                                          static_cast<unsigned int>(ogl_tex->id()),
                                          ogl_tex->width(), ogl_tex->height());
                    }

                    // Now dispatch the previz 3D render on the GL thread
                    // Previz replaces the 2D output entirely and has no working-space
                    // composite of its own, so it carries no per-view outputs. A consumer
                    // that asked for a view gets the primary, which is the honest answer:
                    // there is nothing else to give it.
                    auto f = ogl_->dispatch_async(
                        [this, store, format_desc]() mutable -> core::render_output {
                            auto target_texture =
                                ogl_->create_texture(format_desc.width,
                                                     format_desc.height,
                                                     4,
                                                     renderer_.depth(),
                                                     true,
                                                     // Previz bypasses the 2D composite and is read back
                                                     // directly, so it has no resolve pass to convert a
                                                     // float target. Keep it unorm.
                                                     common::render_format::unorm);
                            previz_renderer_.render(target_texture, *store, format_desc.width, format_desc.height);
                            core::render_output pv;
                            pv.primary = {ogl_->copy_async(target_texture).share(), target_texture};
                            return pv;
                        });

                    return std::move(f.get());
                });
        }

        // Normal 2D compositing
        auto result = renderer_(std::move(layers_), format_desc);

        // Post output texture to the store for previz channels to sample
        if (channel_tex_store_) {
            auto ch_id = channel_id_;
            auto store = channel_tex_store_;
            return std::async(
                std::launch::deferred,
                [result = std::move(result), ch_id, store]() mutable -> core::render_output {
                    auto out  = result.get();
                    auto& tex = out.primary.texture;
                    if (tex) {
                        auto* ogl_tex = dynamic_cast<ogl::texture*>(tex.get());
                        if (ogl_tex)
                            store->update(ch_id, tex, static_cast<unsigned int>(ogl_tex->id()), ogl_tex->width(), ogl_tex->height());
                    }
                    return out;
                });
        }

        return result;
    }

    core::mutable_frame create_frame(const void* tag, const core::pixel_format_desc& desc) override
    {
        return create_frame(tag, desc, common::bit_depth::bit8);
    }

    core::mutable_frame
    create_frame(const void* tag, const core::pixel_format_desc& desc, common::bit_depth depth) override
    {
        std::vector<array<std::uint8_t>> image_data;
        for (std::size_t n = 0; n < desc.planes.size(); ++n) {
            const auto& plane           = desc.planes[n];
            auto        bytes_per_pixel = depth == common::bit_depth::bit8 ? 1 : 2;

            // An aliasing plane shares the buffer of the plane it names instead of
            // getting its own: same bytes, a different sampling rate on the GPU side.
            // Each plane still gets its own texture below; only the staging buffer and
            // the producer's memcpy into it are shared.
            if (plane.alias_of >= 0 && plane.alias_of < static_cast<int>(n)) {
                image_data.push_back(image_data[plane.alias_of].alias());
                continue;
            }

            image_data.push_back(ogl_->create_array(plane.size * bytes_per_pixel));
        }

        std::weak_ptr<image_mixer::impl> weak_self = shared_from_this();
        return core::mutable_frame(tag,
                                   std::move(image_data),
                                   array<int32_t>{},
                                   desc,
                                   [weak_self, desc](std::vector<array<const std::uint8_t>> image_data) -> std::any {
                                       auto self = weak_self.lock();
                                       if (!self) {
                                           return std::any{};
                                       }
                                       std::vector<future_texture> textures;
                                       for (int n = 0; n < static_cast<int>(desc.planes.size()); ++n) {
                                           textures.emplace_back(self->ogl_->copy_async(image_data[n],
                                                                                        desc.planes[n].width,
                                                                                        desc.planes[n].height,
                                                                                        desc.planes[n].stride,
                                                                                        desc.planes[n].depth));
                                       }
                                       auto staged          = std::make_shared<staged_textures>();
                                       staged->owner_device = static_cast<const void*>(self->ogl_.get());
                                       staged->textures     = std::move(textures);
                                       return staged;
                                   });
    }

    common::bit_depth depth() const { return renderer_.depth(); }

    void* gpu_device_handle() const override { return ogl_.get(); }

    core::gpu_backend gpu_device_backend() const override { return core::gpu_backend::opengl; }
};

image_mixer::image_mixer(const spl::shared_ptr<device>& ogl,
                         const int                      channel_id,
                         const size_t                   max_frame_size,
                         common::bit_depth              depth,
                         common::render_format          render_format)
    : impl_(std::make_unique<impl>(ogl, channel_id, max_frame_size, depth, render_format))
{
}
image_mixer::~image_mixer() {}
void image_mixer::push(const core::frame_transform& transform) { impl_->push(transform); }
void image_mixer::visit(const core::const_frame& frame) { impl_->visit(frame); }
void image_mixer::pop() { impl_->pop(); }
void image_mixer::update_aspect_ratio(double aspect_ratio) { impl_->update_aspect_ratio(aspect_ratio); }
std::future<core::render_output> image_mixer::render(const core::video_format_desc& format_desc)
{
    return impl_->render(format_desc);
}
core::mutable_frame image_mixer::create_frame(const void* tag, const core::pixel_format_desc& desc)
{
    return impl_->create_frame(tag, desc);
}
core::mutable_frame
image_mixer::create_frame(const void* tag, const core::pixel_format_desc& desc, common::bit_depth depth)
{
    return impl_->create_frame(tag, desc, depth);
}

common::bit_depth image_mixer::depth() const { return impl_->depth(); }

std::shared_ptr<device> image_mixer::get_ogl_device() const { return impl_->ogl_; }
void* image_mixer::gpu_device_handle() const { return impl_->gpu_device_handle(); }

core::gpu_backend image_mixer::gpu_device_backend() const { return impl_->gpu_device_backend(); }

void* image_mixer::native_gl_context() const { return impl_->ogl_->native_gl_context(); }

void* image_mixer::native_egl_display() const { return impl_->ogl_->native_egl_display(); }

void image_mixer::set_cpu_readback_needed(bool needed) { impl_->renderer_.set_cpu_readback_needed(needed); }

previz_renderer& image_mixer::get_previz_renderer() { return impl_->previz_renderer_; }

void image_mixer::set_channel_texture_store(std::shared_ptr<channel_texture_store> store)
{
    impl_->channel_tex_store_ = std::move(store);
}

void image_mixer::set_target_color(core::color_space cs, core::color_transfer ct, bool auto_convert, int auto_tone_map, float peak_luminance, float sdr_reference_white, bool auto_gamut_compress, bool straight_alpha_grading, bool working_space_composite)
{
    impl_->set_target_color(cs, ct, auto_convert, auto_tone_map, peak_luminance, sdr_reference_white, auto_gamut_compress, straight_alpha_grading, working_space_composite);
}

void image_mixer::set_calibration_lut(std::shared_ptr<const core::lut3d_data> lut, float strength, const std::wstring& path)
{
    impl_->set_calibration_lut(std::move(lut), strength, path);
}

void image_mixer::set_calibration_bypass(bool bypass) { impl_->set_calibration_bypass(bypass); }

core::calibration_lut_state image_mixer::get_calibration_state() const { return impl_->get_calibration_state(); }

void image_mixer::set_ocio_display(const std::string& display, const std::string& view)
{
    impl_->set_ocio_display(display, view);
}

core::ocio_display_state image_mixer::get_ocio_display() const { return impl_->get_ocio_display(); }

void image_mixer::set_consumer_views(std::vector<core::ocio_view_key> views)
{
    impl_->set_consumer_views(std::move(views));
}

void image_mixer::prewarm_ocio(const std::string& source_space,
                               const std::string& display,
                               const std::string& view)
{
    impl_->prewarm_ocio(source_space, display, view);
}

bool image_mixer::composites_in_working_space() const
{
    return impl_->renderer_.working_space_composite;
}

}}} // namespace caspar::accelerator::ogl
