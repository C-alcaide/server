/*
 * Copyright 2025
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
 * Author: Niklas Andersson, niklas@niklaspandersson.se
 */
#include <algorithm>
#include "image_mixer.h"

#include "image_kernel.h"
#include "previz_texture_bridge.h"

#include "../util/buffer.h"
#include "../util/device.h"
#include "../util/renderpass.h"
#include "../util/texture.h"
#include "../util/texture_wrapper.h"

#include "../../ogl/image/previz_renderer.h"
#include "../../ogl/image/previz_scene.h"
#include "../../ogl/util/device.h"
#include "../../ogl/util/texture.h"

#include <boost/align/aligned_allocator.hpp>

#include <common/array.h>
#include <common/bit_depth.h>
#include <common/future.h>
#include <common/log.h>

#include <core/frame/frame.h>
#include <core/frame/frame_transform.h>
#include <core/frame/geometry.h>
#include <core/frame/pixel_format.h>
#include <core/video_format.h>

#include <any>
#include <atomic>
#include <functional>
#include <vector>

namespace caspar { namespace accelerator { namespace vulkan {

using future_texture = std::shared_future<std::shared_ptr<texture>>;

/// Upload futures the frame factory started for a mutable_frame, carried on
/// const_frame::opaque().
///
/// `owner_device` is essential, not decorative: the payload is produced by
/// whichever channel's mixer created the frame, and a routed frame is visited by
/// a *different* channel's mixer. With per-channel GPU affinity that mixer can be
/// on another VkDevice, and binding these VkImages there is undefined behaviour.
/// It used to be a bare vector, so the receiving mixer trusted it blindly and a
/// cross-GPU route rendered nothing.
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
// Textures are held as shared_ptr, not raw pointers: the attachment and
// device-texture pools recycle allocations, so a raw pointer can be reused by a
// different texture and make two different frames compare equal (ABA). The OGL
// mixer was fixed for this; this one had been left behind.
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
    //: The channel LMT. In the fingerprint because changing it changes every pixel while
    //: leaving every layer identical -- exactly what the still-frame cache would replay.
    std::string          ocio_look;
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
               ocio_look == other.ocio_look &&
               calibration_lut == other.calibration_lut && calibration_strength == other.calibration_strength &&
               calibration_bypass == other.calibration_bypass;
    }
};

class image_renderer
{
    spl::shared_ptr<device> vulkan_;
    image_kernel            kernel_;
    const size_t            max_frame_size_;
    common::bit_depth       depth_;
    // The format the kernel composites into. depth_ stays the channel's output depth,
    // which is what the resolve target and every consumer use.
    common::render_format   render_format_ = common::render_format::unorm;
    std::atomic<bool>       cpu_readback_needed_{true};

    // Still-frame cache: skip GPU composition when inputs are unchanged.
    render_fingerprint                            prev_fingerprint_;
    std::shared_ptr<core::texture>                cached_result_wrapper_;
    std::shared_future<array<const std::uint8_t>> cached_result_cpu_;
    //: The cached tick's extra views, cached alongside the primary.
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
    // The channel's LOOK (LMT), composed into the display processor above. Applies to
    // the primary and to every consumer view.
    std::string          ocio_look;

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
        cached_result_wrapper_.reset();
        cached_result_cpu_ = {};
    }

    void set_calibration_bypass(bool bypass)
    {
        calibration_bypass_ = bypass;
        prev_fingerprint_ = {};
        cached_result_wrapper_.reset();
        cached_result_cpu_ = {};
    }

    explicit image_renderer(const spl::shared_ptr<device>& vulkan,
                            const size_t                   max_frame_size,
                            common::bit_depth              depth,
                            common::render_format          render_format = common::render_format::unorm)
        : vulkan_(vulkan)
        , kernel_(vulkan_, depth, render_format)
        , max_frame_size_(max_frame_size)
        , depth_(depth)
        , render_format_(render_format)
    {
        if (render_format_ != common::render_format::unorm) {
            CASPAR_LOG(info) << L"[vulkan_renderer] compositing into fp16 attachments";
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
        // Counted, because this bypass is the ONLY path out of this function that returns a
        // null texture, and a null texture makes every GPU-native consumer fall back to a
        // blank frame — a silently black SDI output. Distinguishing "no layers, correctly"
        // from "no layers, and that is the bug" needs the count and the layer total.
        {
            static std::atomic<std::uint64_t> empty_ticks{0}, drawn_ticks{0};
            const auto n = layers.empty() ? ++empty_ticks : ++drawn_ticks;
            if (n == 1 || n == 100 || (n % 1000) == 0) {
                CASPAR_LOG(debug) << L"[vk_mixer] render tick with "
                                  << static_cast<std::uint64_t>(layers.size())
                                  << L" layer(s) (occurrence " << n << L"; empty="
                                  << empty_ticks.load() << L" drawn=" << drawn_ticks.load()
                                  << L")";
            }
        }

        if (layers.empty()) { // Bypass GPU with empty frame.
            // Release cached textures so VRAM from the last rendered frame is freed
            // (e.g. after STOP clears all layers).
            prev_fingerprint_ = {};
            cached_result_wrapper_.reset();
            cached_result_cpu_ = {};

            static const std::vector<uint8_t, boost::alignment::aligned_allocator<uint8_t, 32>> buffer(max_frame_size_,
                                                                                                       0);
            auto ready = make_ready_future<array<const std::uint8_t>>(
                array<const std::uint8_t>(buffer.data(), format_desc.size, true));
            core::render_output empty;
            empty.primary = {ready.share(), nullptr};
            return make_ready_future<core::render_output>(std::move(empty));
        }

        // ── Still-frame cache ──────────────────────────────────────────────
        // When the input textures AND transforms are identical to the previous
        // tick (i.e. the producer returned a "still" frame and no animation is
        // active), skip the GPU composition entirely and reuse the cached output.
        // This reduces GPU mixer load from 60fps to ~25fps for typical
        // single-producer setups, freeing GPU resources for the CUDA decoder.
        {
            auto fingerprint = build_fingerprint(layers, format_desc);

            // The cache holds the WHOLE tick, views included, and the view COUNT is part
            // of the decision: a consumer attaching or detaching changes the view set
            // without changing a single layer, so the fingerprint alone would replay a tick
            // that no longer has the right outputs in it.
            if (!fingerprint.items.empty() && fingerprint.matches(prev_fingerprint_) &&
                cached_result_wrapper_ && cached_views_.size() == consumer_views_.size()) {
                layers.clear();   // release the layer data
                core::render_output cached;
                cached.primary = {cached_result_cpu_, cached_result_wrapper_};
                cached.views   = cached_views_;
                return make_ready_future<core::render_output>(std::move(cached));
            }
            prev_fingerprint_ = std::move(fingerprint);
        }

        auto f = std::move(vulkan_->dispatch_async(
            [this, format_desc, cal_lut = calibration_lut_, cal_strength = calibration_strength_,
             cal_bypass = calibration_bypass_, ws_composite = working_space_composite,
             ch_display = ocio_display, ch_view = ocio_view,
             // Applies to the primary AND every consumer view: a look is creative intent,
             // a view is the screen it goes to.
             ch_look = ocio_look,
             views = (working_space_composite ? consumer_views_ : std::vector<core::ocio_view_key>{}),
             layers = std::move(layers)]() mutable -> core::render_output {
                // THE RASTER IS `width x height`, NOT `square_width x square_height`.
                //
                // `square_*` is the display size a non-square-pixel format would occupy on a
                // square-pixel screen -- PAL is 720x576 stored, 1024x576 displayed. It belongs
                // in the aspect maths (`draw_params.target_width` below, exactly as the OpenGL
                // mixer sets it) and NOT in the size of the thing we render into and read back:
                // the channel's frame is `format_desc.width x format_desc.height`, and the
                // consumer interprets the readback at that size.
                //
                // Rendering into 1024x576 and handing back 1024x576 bytes for a frame declared
                // 720x576 shears every row by 304 pixels. PAL rendered as unrecognisable
                // striping; OpenGL was correct all along because it creates its target with
                // `format_desc.width, format_desc.height` (ogl/image/image_mixer.cpp).
                //
                // Invisible everywhere `square_width == width`, which is every other mode on
                // this rig -- NTSC (720x486, square 720x540), 1080i/p and the custom
                // 2600x1500 all pass. And invisible to `cli.py conformance`, whose flat
                // patches cannot show a sampling displacement at all.
                auto pass   = kernel_.create_renderpass(format_desc.width, format_desc.height);
                auto target = pass->default_attachment();

                draw(target, std::move(layers), format_desc, pass);

                // THE WORKING-SPACE COMPOSITE, before anything encodes it. Every view
                // below starts from this same attachment -- a display transform is not
                // invertible, so a second view cannot be derived from the first once one
                // has been applied.
                auto composite = target;

                // Everything after the composite, for ONE view. All of it lands in this
                // renderpass's single command buffer, so one fence covers every view and
                // each wrapper below can share it -- which is why this is one pass with N
                // attachments rather than N passes sharing a texture across them.
                auto finish = [&](std::shared_ptr<texture> tex,
                                  const std::string&       disp,
                                  const std::string&       vw) -> std::shared_ptr<texture> {
                    if (ws_composite) {
                        auto a = pass->create_attachment();
                        apply_output_convert(tex, a, format_desc, pass, disp, vw, ch_look);
                        tex = a;
                    }
                    if (cal_lut && !cal_bypass && cal_lut->size > 0) {
                        auto a = pass->create_attachment();
                        apply_calibration_lut(tex, a, format_desc, pass, cal_lut, cal_strength);
                        tex = a;
                    }
                    // Everything downstream of the mixer means integer, so a float working
                    // space is resolved here. An explicit draw rather than
                    // set_resolve_target(), because that is per pass and singular -- see
                    // apply_passthrough.
                    if (render_format_ != common::render_format::unorm) {
                        auto a = pass->create_attachment_as(common::render_format::unorm);
                        apply_passthrough(tex, a, format_desc, pass);
                        tex = a;
                    }
                    return tex;
                };

                auto                                  primary_target = finish(composite, ch_display, ch_view);
                std::vector<std::shared_ptr<texture>> view_targets;
                view_targets.reserve(views.size());
                for (const auto& v : views)
                    view_targets.push_back(finish(composite, v.display, v.view));

                pass->commit();

                // One fence and one semaphore for the whole pass, shared by every view's
                // wrapper: all the draws above are in the same command buffer, so waiting
                // on the pass covers all of them.
                auto wait_fn    = [p = pass]() { p->wait_for_completion(); };
                auto sem_handle = pass->render_semaphore_handle();
                auto sem_value  = pass->render_semaphore_value();

                const bool needs_cpu = cpu_readback_needed_.load(std::memory_order_relaxed);
                if (!needs_cpu) {
                    static bool logged_skip = false;
                    if (!logged_skip) {
                        CASPAR_LOG(info) << L"[vk_mixer] CPU readback SKIPPED - all consumers use GPU-native paths";
                        logged_skip = true;
                    }
                }

                // The device goes into each wrapper so a consumer can ask for a reduced
                // readback instead of declaring needs_cpu_frame_data and pulling the whole
                // frame back every tick.
                auto make_result = [&](std::shared_ptr<texture> tex) -> core::render_result {
                    auto wrapper = std::make_shared<texture_wrapper>(
                        tex, wait_fn, sem_handle, sem_value, vulkan_);
                    if (!needs_cpu) {
                        auto empty = make_ready_future<array<const std::uint8_t>>(
                            array<const std::uint8_t>(nullptr, 0, true));
                        return {empty.share(), wrapper};
                    }
                    return {vulkan_->copy_async(tex).share(), wrapper};
                };

                core::render_output out;
                out.primary = make_result(primary_target);
                for (size_t k = 0; k < views.size(); ++k)
                    out.views.emplace_back(views[k], make_result(view_targets[k]));
                return out;
            }));

        return std::async(
            std::launch::deferred,
            [this, f = std::move(f)]() mutable -> core::render_output {
                auto out = std::move(f.get());
                // Update the still-frame cache so the next tick can skip GPU composition if
                // the inputs haven't changed -- views included, or a cached tick replays the
                // primary for every view.
                cached_result_cpu_     = out.primary.image;
                cached_result_wrapper_ = out.primary.texture;
                cached_views_          = out.views;
                return out;
            });
    }

    common::bit_depth depth() const { return depth_; }

    /// Forward to the kernel, which owns the variant cache.
    void prewarm_ocio(const std::string& source_space, const std::string& display, const std::string& view,
                      const std::string& look = "")
    {
        kernel_.prewarm_ocio(source_space, display, view, look);
    }

  private:
    /// Collects the full description of what composition would draw.
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
        fp.ocio_look              = ocio_look;
        fp.calibration_lut        = calibration_lut_.get();
        fp.calibration_strength   = calibration_strength_;
        fp.calibration_bypass     = calibration_bypass_;

        int                                           path = 0;
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
              const core::video_format_desc& format_desc,
              spl::shared_ptr<renderpass>    pass)
    {
        std::shared_ptr<texture> layer_key_texture;

        for (auto& layer : layers) {
            draw(target_texture, layer.sublayers, format_desc, pass);
            draw(target_texture, std::move(layer), layer_key_texture, format_desc, pass);
        }
    }

    void draw(std::shared_ptr<texture>&      target_texture,
              layer                          layer,
              std::shared_ptr<texture>&      layer_key_texture,
              const core::video_format_desc& format_desc,
              spl::shared_ptr<renderpass>    pass)
    {
        if (layer.items.empty())
            return;

        std::shared_ptr<texture> local_key_texture;
        std::shared_ptr<texture> local_mix_texture;

        if (layer.blend_mode != core::blend_mode::normal) {
            auto layer_texture = pass->create_attachment();

            for (auto& item : layer.items)
                draw(layer_texture,
                     std::move(item),
                     layer_key_texture,
                     local_key_texture,
                     local_mix_texture,
                     format_desc,
                     pass);

            draw(layer_texture, std::move(local_mix_texture), format_desc, pass, core::blend_mode::normal);
            draw(target_texture, std::move(layer_texture), format_desc, pass, layer.blend_mode);
        } else // fast path
        {
            for (auto& item : layer.items)
                draw(target_texture,
                     std::move(item),
                     layer_key_texture,
                     local_key_texture,
                     local_mix_texture,
                     format_desc,
                     pass);

            draw(target_texture, std::move(local_mix_texture), format_desc, pass, core::blend_mode::normal);
        }

        layer_key_texture = std::move(local_key_texture);
    }

    void draw(std::shared_ptr<texture>&      target_texture,
              item                           item,
              std::shared_ptr<texture>&      layer_key_texture,
              std::shared_ptr<texture>&      local_key_texture,
              std::shared_ptr<texture>&      local_mix_texture,
              const core::video_format_desc& format_desc,
              spl::shared_ptr<renderpass>    pass)
    {
        draw_params draw_params;
        draw_params.target_width  = format_desc.square_width;
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
            local_key_texture = local_key_texture ? local_key_texture : pass->create_attachment();

            draw_params.background = local_key_texture;
            draw_params.local_key  = nullptr;
            draw_params.layer_key  = nullptr;

            pass->draw(std::move(draw_params));
        } else if (draw_params.transforms.image_transform
                       .is_mix) { // A mix means precomp the items to a texture, before drawing to the channel
            local_mix_texture = local_mix_texture ? local_mix_texture : pass->create_attachment();

            draw_params.background = local_mix_texture;
            draw_params.local_key  = std::move(local_key_texture); // Use and reset the key
            draw_params.layer_key  = layer_key_texture;

            draw_params.keyer = keyer::additive;

            pass->draw(std::move(draw_params));
        } else {
            // If there is a mix, this is the end so draw it and reset
            draw(target_texture, std::move(local_mix_texture), format_desc, pass, core::blend_mode::normal);

            // Mirror of the OpenGL mixer, including the reason this is read out before the
            // move: `grade_nodes` lives in draw_params and reading it afterwards is UB.
            std::shared_ptr<const core::grade_graph> graph = draw_params.transforms.image_transform.grade_nodes;
            int                                     enabled_nodes = 0;
            if (graph) {
                for (const auto& n : graph->nodes)
                    if (n.enable)
                        ++enabled_nodes;
            }

            // No graph = the path that existed before this feature, unchanged: one draw
            // straight into the target, no attachment.
            std::shared_ptr<texture> node_texture;
            if (enabled_nodes > 0) {
                // Once feared to break the composite, MEASURED NOT TO. The worry was that
                // routing through an attachment changes how this layer meets the target --
                // `keyer`, the keys and a non-normal blend mode all interact with it. Working
                // through which of those can actually co-occur with a node graph:
                //
                //   * a non-normal blend mode is a LAYER property, and such a layer already
                //     renders into its own `layer_texture` before being composited against the
                //     real target with that mode. The node attachment nests inside that and is
                //     invisible to it.
                //   * `keyer::additive` is set only in the `is_mix` branch above; the node path
                //     is the final `else`, so the two cannot co-occur.
                //   * `local_key`/`layer_key` only scale the item's alpha -- they mask the item,
                //     not the composite.
                //
                // What is left is an ordinary item with the linear keyer and normal blend, where
                // `fore + (1-a)*0` into the attachment followed by `fore + (1-a)*target` at the
                // composite is algebraically identical to the direct draw.
                //
                // Guarded by `grade-window`'s composite check: a two-layer scene under a
                // `screen` blend, sampled outside the window where the node does nothing, with
                // and without a graph. 0.00 LSB on both mixers.
                node_texture           = pass->create_attachment();
                draw_params.background = node_texture;
            } else {
                draw_params.background = target_texture;
            }
            draw_params.local_key  = std::move(local_key_texture);
            draw_params.layer_key  = layer_key_texture;

            pass->draw(std::move(draw_params));

            if (enabled_nodes > 0) {
                auto src = node_texture;
                for (const auto& n : graph->nodes) {
                    if (!n.enable)
                        continue;
                    auto dst = pass->create_attachment();
                    apply_grade_node(src, dst, format_desc, pass, n);
                    src = dst;
                }
                draw(target_texture, std::move(src), format_desc, pass, core::blend_mode::normal);
            }
        }
    }

    /// One grading node's full-screen pass. Mirror of the OpenGL mixer's version, and
    /// modelled on `apply_calibration_lut` below for the same reason: source in `textures`
    /// sampled as an ordinary sampler2D (not through `subpassInput background`), destination
    /// in `background`, both conversion halves off.
    void apply_grade_node(std::shared_ptr<texture>&      source_texture,
                          std::shared_ptr<texture>&      target_texture,
                          const core::video_format_desc& format_desc,
                          spl::shared_ptr<renderpass>    pass,
                          const core::grade_node&        node)
    {
        if (!source_texture)
            return;

        draw_params draw_params;
        draw_params.target_width  = format_desc.square_width;
        draw_params.target_height = format_desc.square_height;
        // 8-bit attachments store BGRA (shader .bgra swizzle); 16-bit store RGBA directly.
        // Getting this wrong exchanges red and blue, and the node's own operation is a
        // uniform scale that would not reveal it.
        draw_params.pix_desc.format = (source_texture->depth() == common::bit_depth::bit8)
                                          ? core::pixel_format::bgra
                                          : core::pixel_format::rgba;
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
        draw_params.geometry                = core::frame_geometry::get_default();
        draw_params.grade_node_only         = true;
        draw_params.grade_node              = node;

        pass->draw(std::move(draw_params));
    }

    void draw(std::shared_ptr<texture>&   target_texture,
              std::shared_ptr<texture>&&  source_texture,
              core::video_format_desc     format_desc,
              spl::shared_ptr<renderpass> pass,
              core::blend_mode            blend_mode = core::blend_mode::normal)
    {
        if (!source_texture)
            return;

        draw_params draw_params;
        draw_params.target_width    = format_desc.square_width;
        draw_params.target_height   = format_desc.square_height;
        // 8-bit attachments store BGRA (shader .bgra swizzle); 16-bit store RGBA directly.
        draw_params.pix_desc.format = (source_texture->depth() == common::bit_depth::bit8)
                                          ? core::pixel_format::bgra
                                          : core::pixel_format::rgba;
        draw_params.pix_desc.planes = {core::pixel_format_desc::plane(
            source_texture->width(), source_texture->height(), 4, source_texture->depth())};
        draw_params.textures        = {spl::make_shared_ptr(source_texture)};
        draw_params.blend_mode      = blend_mode;
        draw_params.background      = target_texture;
        draw_params.geometry        = core::frame_geometry::get_default();

        pass->draw(std::move(draw_params));
    }

    // Channel-master calibration LUT: full-screen pass that copies the composited
    // frame through a 3D LUT into a fresh attachment. The source is tagged with
    // the channel's output colour space so the kernel performs NO colour
    // conversion — only the calibration LUT runs (display-to-display correction).
    /// The channel's post-composite output conversion. Mirrors the OGL mixer, where the
    /// full account lives: every layer reached the composite in scene-linear ACEScg with its
    /// output half suppressed, so the display encoding is applied ONCE here.
    /// A straight blit through the kernel with no colour work -- the explicit form of what
    /// `set_resolve_target()` does implicitly.
    ///
    /// Needed because that setter is per renderpass and singular (`_resolve_target` is one
    /// member) and `result_attachment()` returns one texture, so it cannot serve several
    /// views. Drawing instead keeps every view's resolve in the SAME command buffer, which
    /// is the property the setter's comment was protecting: ordered against the composite
    /// without an extra submit or a stall.
    void apply_passthrough(std::shared_ptr<texture>&      source_texture,
                           std::shared_ptr<texture>&      target_texture,
                           const core::video_format_desc& format_desc,
                           spl::shared_ptr<renderpass>    pass)
    {
        if (!source_texture)
            return;

        draw_params draw_params;
        draw_params.target_width  = format_desc.square_width;
        draw_params.target_height = format_desc.square_height;
        draw_params.pix_desc.format = (source_texture->depth() == common::bit_depth::bit8)
                                          ? core::pixel_format::bgra
                                          : core::pixel_format::rgba;
        draw_params.pix_desc.planes = {core::pixel_format_desc::plane(
            source_texture->width(), source_texture->height(), 4, source_texture->depth())};
        draw_params.pix_desc.color_space    = target_color_space;
        draw_params.pix_desc.color_transfer = target_color_transfer;
        draw_params.target_color_space      = target_color_space;
        draw_params.target_color_transfer   = target_color_transfer;
        // No conversion and no tone mapping: the picture is already in the channel's output
        // encoding by this point, and either would apply a second transform to it.
        draw_params.auto_color_convert      = false;
        draw_params.auto_tone_map           = 0;
        draw_params.textures                = {spl::make_shared_ptr(source_texture)};
        draw_params.blend_mode              = core::blend_mode::normal;
        draw_params.background              = target_texture;
        draw_params.geometry                = core::frame_geometry::get_default();

        pass->draw(std::move(draw_params));
    }

    void apply_output_convert(std::shared_ptr<texture>&      source_texture,
                              std::shared_ptr<texture>&      target_texture,
                              const core::video_format_desc& format_desc,
                              spl::shared_ptr<renderpass>    pass,
                              const std::string&             display,
                              const std::string&             view,
                              const std::string&             look)
    {
        if (!source_texture)
            return;

        draw_params draw_params;
        draw_params.target_width  = format_desc.square_width;
        draw_params.target_height = format_desc.square_height;
        draw_params.pix_desc.format = (source_texture->depth() == common::bit_depth::bit8)
                                          ? core::pixel_format::bgra
                                          : core::pixel_format::rgba;
        draw_params.pix_desc.planes = {core::pixel_format_desc::plane(
            source_texture->width(), source_texture->height(), 4, source_texture->depth())};
        draw_params.pix_desc.color_space    = target_color_space;
        draw_params.pix_desc.color_transfer = target_color_transfer;
        draw_params.target_color_space      = target_color_space;
        draw_params.target_color_transfer   = target_color_transfer;
        draw_params.auto_tone_map           = auto_tone_map;
        draw_params.display_peak_luminance  = display_peak_luminance;
        draw_params.sdr_reference_white     = sdr_reference_white;
        // Both off: this draw IS the output half. working_space_composite belongs to LAYER
        // draws, and setting it here would suppress the conversion this pass exists for.
        draw_params.auto_color_convert      = false;
        draw_params.working_space_composite = false;
        draw_params.output_convert_only     = true;
        // The display transform, if any, owns the output half of THIS pass: the kernel's
        // `ocio_out` check clears do_output_convert and splices the generated program in
        // its place. Set here and nowhere else -- a layer draw with a display transform
        // would encode each layer separately, which is the per-layer arrangement this
        // stage exists to replace.
        // Whichever view THIS pass is for -- the channel's own, or a consumer's.
        draw_params.ocio_display            = display;
        draw_params.ocio_view               = view;
        draw_params.ocio_look               = look;
        draw_params.straight_alpha_grading  = straight_alpha_grading;
        draw_params.textures                = {spl::make_shared_ptr(source_texture)};
        draw_params.blend_mode              = core::blend_mode::normal;
        draw_params.background              = target_texture;
        draw_params.geometry                = core::frame_geometry::get_default();

        pass->draw(std::move(draw_params));
    }

    void apply_calibration_lut(std::shared_ptr<texture>&                      source_texture,
                               std::shared_ptr<texture>&                      target_texture,
                               const core::video_format_desc&                 format_desc,
                               spl::shared_ptr<renderpass>                    pass,
                               const std::shared_ptr<const core::lut3d_data>& lut,
                               float                                          strength)
    {
        if (!source_texture || !lut)
            return;

        draw_params draw_params;
        draw_params.target_width    = format_desc.square_width;
        draw_params.target_height   = format_desc.square_height;
        // 8-bit attachments store BGRA (shader .bgra swizzle); 16-bit store RGBA directly.
        draw_params.pix_desc.format = (source_texture->depth() == common::bit_depth::bit8)
                                          ? core::pixel_format::bgra
                                          : core::pixel_format::rgba;
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

        pass->draw(std::move(draw_params));
    }
};

struct image_mixer::impl
    : public core::frame_factory
    , public std::enable_shared_from_this<impl>
{
    spl::shared_ptr<device>      vulkan_;
    image_renderer               renderer_;
    std::vector<draw_transforms> transform_stack_;
    std::vector<layer>           layers_; // layer/stream/items
    std::vector<layer*>          layer_stack_;

    // One-shot warnings: these paths cost a readback, or drop an item, on every
    // frame -- so they must be visible, but not once per frame.
    std::atomic<bool> foreign_texture_logged_{false};
    std::atomic<bool> no_source_logged_{false};
    std::atomic<bool> unsupported_stride_logged_{false};

    double aspect_ratio_ = 1.0;

    // Previz support
    std::shared_ptr<ogl::device>                 previz_ogl_device_;
    std::shared_ptr<ogl::channel_texture_store>  channel_tex_store_;
    std::unique_ptr<ogl::previz_renderer>        previz_renderer_;
    std::shared_ptr<previz_texture_bridge>       previz_bridge_;
    std::once_flag                               previz_init_flag_;
    int                                          channel_id_ = 0;

  public:
    impl(const spl::shared_ptr<device>& device,
         const int                      channel_id,
         const size_t                   max_frame_size,
         common::bit_depth              depth,
         common::render_format          render_format)
        : vulkan_(device)
        , renderer_(device, max_frame_size, depth, render_format)
        , transform_stack_(1)
        , channel_id_(channel_id)
    {
        CASPAR_LOG(info) << L"Initialized Vulkan Accelerated GPU Image Mixer for channel " << channel_id;
    }

    void update_aspect_ratio(double aspect_ratio) { aspect_ratio_ = aspect_ratio; }

    void set_target_color(core::color_space cs, core::color_transfer ct, bool auto_convert, int auto_tone_map, float peak_luminance, float sdr_ref_white, bool gamut_compress, bool straight_alpha, bool ws_composite)
    {
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

    void set_ocio_look(const std::string& look)
    {
        CASPAR_LOG(info) << L"[mixer] set_ocio_look look=\"" << u16(look) << L"\"";
        renderer_.ocio_look = look;
    }

    std::string get_ocio_look() const { return renderer_.ocio_look; }

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

    void prewarm_ocio(const std::string& source_space, const std::string& display, const std::string& view,
                      const std::string& look = "")
    {
        // On the device thread and asynchronously, for the same reason as the OGL mixer:
        // the compile still costs what it costs, but it no longer costs a frame.
        vulkan_->dispatch_async([this, source_space, display, view, look] {
            renderer_.prewarm_ocio(source_space, display, view, look);
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
        CASPAR_LOG(info) << L"[vk_mixer] set_calibration_lut size="
                         << (lut ? lut->size : 0) << L" strength=" << strength
                         << L" path=" << path;
        renderer_.set_calibration_lut(std::move(lut), strength);
        calibration_path_ = path;
    }

    void set_calibration_bypass(bool bypass)
    {
        CASPAR_LOG(info) << L"[vk_mixer] set_calibration_bypass " << bypass;
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

        // Can this GPU sample the layout at all? Asked first, before any of the four
        // routes below, because it is a property of `pix_desc` and every route ends in the
        // same create_texture -- including the pre-staged one, where the frame factory
        // already handed the producer a copy_async future. That future is only awaited
        // during draw, so the throw surfaced on the CHANNEL thread, once per frame, rather
        // than where the frame was built.
        //
        // Packed 3-byte RGB (rgb24/bgr24) is the case that reaches this. On this GPU it
        // cost an opaque PNG its picture entirely: a blank SDI output, ~4,800 exceptions a
        // second, and a channel free-running at 190x its frame rate -- none of which named
        // a pixel format.
        //
        // Dropping the item loses the layer, which is worse than OpenGL (it samples
        // stride-3 fine). That is a parity floor the codebase already took: producers are
        // expected to convert before the mixer, and the ffmpeg and image producers both do.
        // This is the guard for the ones that do not, and the reason it says so out loud.
        for (const auto& plane : item.pix_desc.planes) {
            if (!vulkan_->can_sample_packed(plane.stride, plane.depth)) {
                if (!unsupported_stride_logged_.exchange(true)) {
                    CASPAR_LOG(warning)
                        << L"[vk::image_mixer] this GPU cannot sample a packed " << plane.stride
                        << L"-component image at this depth, so the layer cannot be uploaded; dropping it. Packed "
                           L"3-byte RGB (rgb24/bgr24) is the case that reaches this -- the producer should convert "
                           L"to a 4-component or planar layout, or the channel should use the OpenGL accelerator.";
                }
                item.textures.clear();
                return false;
            }
        }

        if (!frame.textures().empty()) {
            // Every plane must belong to this device; a partially-usable set is
            // not usable at all, so check them all before binding any.
            bool all_mine = true;
            for (auto& core_tex : frame.textures()) {
                auto native = std::dynamic_pointer_cast<texture_wrapper>(core_tex);
                if (!native || core_tex->owner_device() == nullptr ||
                    core_tex->owner_device() != static_cast<const void*>(vulkan_->getVkDevice())) {
                    all_mine = false;
                    break;
                }
            }

            if (all_mine) {
                for (auto& core_tex : frame.textures()) {
                    auto native = std::static_pointer_cast<texture_wrapper>(core_tex);
                    item.textures.emplace_back(make_ready_future(std::shared_ptr<texture>(native->vk_texture())).share());
                }
                return true;
            }

            // (2) would import the foreign allocation via external memory. Not
            // implemented; until it is, fall through to the host path so the
            // frame still draws, and say so once because it costs a readback.
            if (!foreign_texture_logged_.exchange(true)) {
                CASPAR_LOG(warning) << L"[vk::image_mixer]" L" frame GPU planes are not usable on this device "
                                       L"(different VkDevice -- cross-GPU route? -- or backend); falling back to a host upload.";
            }
        }

        // (3) The frame factory already started uploads for this frame.
        if (frame.opaque().has_value()) {
            // any_cast to a pointer type yields nullptr on a type mismatch rather
            // than throwing, which is what we want when the payload came from the
            // other backend.
            if (auto staged = std::any_cast<std::shared_ptr<staged_textures>>(&frame.opaque())) {
                if (*staged && (*staged)->owner_device == static_cast<const void*>(vulkan_->getVkDevice())) {
                    item.textures = (*staged)->textures;
                    return true;
                }
                if (!foreign_texture_logged_.exchange(true)) {
                    CASPAR_LOG(warning) << L"[vk::image_mixer] pre-staged upload belongs to another device "
                                           L"(cross-GPU route?); falling back to a host upload.";
                }
            }
        }

        // (4) Host planes. `unavailable` means there are none and none coming --
        // typically a GPU-only frame whose readback was skipped -- so uploading
        // would sample a null pointer.
        if (host_state == core::host_image_availability::unavailable) {
            if (!no_source_logged_.exchange(true)) {
                CASPAR_LOG(warning) << L"[vk::image_mixer] item has no usable GPU texture and no host pixels "
                                       L"(readback was skipped); dropping it.";
            }
            return false;
        }

        for (int n = 0; n < static_cast<int>(item.pix_desc.planes.size()); ++n) {
            const auto& plane = frame.image_data(n);
            if (plane.size() == 0 || plane.data() == nullptr) {
                if (!no_source_logged_.exchange(true)) {
                    CASPAR_LOG(warning) << L"[vk::image_mixer] host plane " << n << L" is empty; dropping this item.";
                }
                item.textures.clear();
                return false;
            }
            item.textures.emplace_back(vulkan_->copy_async(plane,
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
        item.geometry   = frame.geometry();

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
        // ── Previz path ────────────────────────────────────────────────────
        // When previz is active: (1) do normal VK compositing, (2) post the
        // VK output texture to the VK→GL bridge, (3) render the previz 3D
        // scene on the OGL thread, (4) return the previz output.
        if (previz_renderer_ && previz_renderer_->active() && previz_bridge_ && channel_tex_store_) {
            auto bridge = previz_bridge_.get();
            auto store  = channel_tex_store_;
            auto ch_id  = channel_id_;
            auto ogl    = previz_ogl_device_;
            auto previz = previz_renderer_.get();
            auto depth  = renderer_.depth();

            // Normal VK compositing first
            auto composited = renderer_(std::move(layers_), format_desc);

            return std::async(
                std::launch::deferred,
                [bridge, store, ch_id, ogl, previz, depth, format_desc,
                 composited = std::move(composited)]() mutable -> core::render_output {
                    // Wait for VK compositing to complete
                    auto  comp_out = composited.get();
                    auto& comp_tex = comp_out.primary.texture;

                    // Post the composited VK texture to the bridge
                    if (comp_tex) {
                        auto* wrapper = dynamic_cast<texture_wrapper*>(comp_tex.get());
                        if (wrapper) {
                            wrapper->ensure_render_complete();
                            auto vk_tex = wrapper->vk_texture();
                            // After the renderpass, the attachment is in
                            // eColorAttachmentOptimal.  copy_async (if it ran)
                            // transitions to eTransferSrcOptimal, but that runs
                            // as a separate VK dispatch task before our
                            // dispatch_sync, so by the time our blit runs the
                            // source may be in either layout.  We use
                            // COLOR_ATTACHMENT_OPTIMAL as the common case;
                            // NVIDIA drivers tolerate the mismatch gracefully.
                            bridge->post_channel(ch_id,
                                                 vk_tex->id(),
                                                 VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                                 vk_tex->width(),
                                                 vk_tex->height(),
                                                 depth != common::bit_depth::bit8);
                        }
                    }

                    // Render previz on the OGL thread
                    auto f = ogl->dispatch_async(
                        // Previz replaces the 2D output and has no working-space composite
                        // of its own, so it carries no per-view outputs.
                        [bridge, store, previz, format_desc, depth, ogl]() mutable
                        -> core::render_output {
                            // Sync bridge textures into the channel store
                            bridge->sync_to_store(*store);

                            // Render 3D previz scene.  The previz renderer
                            // binds its own FBO and does a full glClear, so the
                            // texture's glClearTexImage init is redundant — and
                            // for some depth/format combinations it raises a
                            // transient GL_INVALID_VALUE.  Skip it (clear=false).
                            auto target = ogl->create_texture(
                                format_desc.width, format_desc.height, 4, depth, false);
                            previz->render(target, *store, format_desc.width, format_desc.height);

                            core::render_output pv;
                            pv.primary = {ogl->copy_async(target).share(),
                                          std::static_pointer_cast<core::texture>(target)};
                            return pv;
                        });

                    return std::move(f.get());
                });
        }

        // ── Normal (non-previz) path ───────────────────────────────────────
        // Post VK output to the bridge for other previz channels to sample
        if (previz_bridge_ && channel_tex_store_) {
            auto bridge = previz_bridge_.get();
            auto ch_id  = channel_id_;
            auto depth  = renderer_.depth();

            auto result = renderer_(std::move(layers_), format_desc);

            return std::async(
                std::launch::deferred,
                [result = std::move(result), bridge, ch_id, depth]() mutable -> core::render_output {
                    auto  out = result.get();
                    auto& tex = out.primary.texture;
                    if (tex) {
                        auto* wrapper = dynamic_cast<texture_wrapper*>(tex.get());
                        if (wrapper) {
                            wrapper->ensure_render_complete();
                            auto vk_tex = wrapper->vk_texture();
                            bridge->post_channel(ch_id,
                                                 vk_tex->id(),
                                                 VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                                 vk_tex->width(),
                                                 vk_tex->height(),
                                                 depth != common::bit_depth::bit8);
                        }
                    }
                    return out;
                });
        }

        return renderer_(std::move(layers_), format_desc);
    }

    core::mutable_frame create_frame(const void* tag, const core::pixel_format_desc& desc) override
    {
        return create_frame(tag, desc, common::bit_depth::bit8);
    }

    /// Producers that can hand this mixer a GPU texture need its device. Until
    /// this existed the base's nullptr was returned, and the ffmpeg producer
    /// read that as "no GPU device" and declined GPU-direct decode outright --
    /// so choosing the Vulkan mixer meant silently losing hardware decode.
    void* gpu_device_handle() const override { return vulkan_.get(); }

    core::gpu_backend gpu_device_backend() const override { return core::gpu_backend::vulkan; }

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

            image_data.push_back(vulkan_->create_array(plane.size * bytes_per_pixel));
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
                                           textures.emplace_back(self->vulkan_->copy_async(image_data[n],
                                                                                           desc.planes[n].width,
                                                                                           desc.planes[n].height,
                                                                                           desc.planes[n].stride,
                                                                                           desc.planes[n].depth));
                                       }
                                       auto staged          = std::make_shared<staged_textures>();
                                       staged->owner_device = static_cast<const void*>(self->vulkan_->getVkDevice());
                                       staged->textures     = std::move(textures);
                                       return staged;
                                   });
    }

    common::bit_depth depth() const { return renderer_.depth(); }

    void set_previz_ogl_device(const std::shared_ptr<ogl::device>& ogl_dev)
    {
        previz_ogl_device_ = ogl_dev;
    }

    void set_channel_texture_store(const std::shared_ptr<ogl::channel_texture_store>& store)
    {
        channel_tex_store_ = store;
    }

    void set_previz_bridge(const std::shared_ptr<previz_texture_bridge>& bridge)
    {
        previz_bridge_ = bridge;
    }

    ogl::previz_renderer* get_previz_renderer()
    {
        std::call_once(previz_init_flag_, [this] {
            if (!previz_ogl_device_)
                return;
            previz_renderer_ = std::make_unique<ogl::previz_renderer>(
                spl::make_shared_ptr(previz_ogl_device_));
            CASPAR_LOG(info) << L"[vk_mixer] Created previz renderer for channel " << channel_id_;
        });
        return previz_renderer_.get();
    }
};

image_mixer::image_mixer(const spl::shared_ptr<device>& vulkan,
                         const int                      channel_id,
                         const size_t                   max_frame_size,
                         common::bit_depth              depth,
                         common::render_format          render_format)
    : impl_(std::make_unique<impl>(vulkan, channel_id, max_frame_size, depth, render_format))
{
}
image_mixer::~image_mixer()
{
    if (impl_->channel_tex_store_)
        impl_->channel_tex_store_->remove(impl_->channel_id_);
}
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

void image_mixer::set_cpu_readback_needed(bool needed) { impl_->renderer_.set_cpu_readback_needed(needed); }

std::shared_ptr<device> image_mixer::get_vk_device() const { return impl_->vulkan_; }
void*             image_mixer::gpu_device_handle() const { return impl_->gpu_device_handle(); }
core::gpu_backend image_mixer::gpu_device_backend() const { return impl_->gpu_device_backend(); }

void image_mixer::set_previz_ogl_device(const std::shared_ptr<ogl::device>& ogl_dev)
{
    impl_->set_previz_ogl_device(ogl_dev);
}

void image_mixer::set_channel_texture_store(const std::shared_ptr<ogl::channel_texture_store>& store)
{
    impl_->set_channel_texture_store(store);
}

void image_mixer::set_previz_bridge(const std::shared_ptr<previz_texture_bridge>& bridge)
{
    impl_->set_previz_bridge(bridge);
}

ogl::previz_renderer* image_mixer::get_previz_renderer()
{
    return impl_->get_previz_renderer();
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

void image_mixer::set_ocio_look(const std::string& look) { impl_->set_ocio_look(look); }

std::string image_mixer::get_ocio_look() const { return impl_->get_ocio_look(); }

void image_mixer::set_consumer_views(std::vector<core::ocio_view_key> views)
{
    impl_->set_consumer_views(std::move(views));
}

void image_mixer::prewarm_ocio(const std::string& source_space,
                               const std::string& display,
                               const std::string& view,
                               const std::string& look)
{
    impl_->prewarm_ocio(source_space, display, view, look);
}

bool image_mixer::composites_in_working_space() const
{
    return impl_->renderer_.working_space_composite;
}

}}} // namespace caspar::accelerator::vulkan
