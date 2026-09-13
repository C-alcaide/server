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

#include <core/graph/plan.h>
#include <core/graph/registry.h>
#include "previz_texture_bridge.h"

#include "../util/buffer.h"
#include "../util/device.h"
#include "../util/renderpass.h"
#include "../util/texture.h"
#include "../util/texture_wrapper.h"

#include "../../ogl/image/previz_renderer.h"
#include "../../ogl/image/previz_scene.h"

#include <core/stage/stage_fields.h>
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
  public:
    // ── A PENDING NODE PREVIEW ──────────────────────────────────────────────────────
    //
    // IN THE RENDERER, because this is where it is SERVED: a node's attachment is alive only
    // inside the evaluator's loop, between the draw that writes it and the `last_use` release
    // that returns it to the pool. `impl` forwards the arming call here rather than holding the
    // state -- the first version put it on `impl` and compiled nowhere, because the serve site
    // could not see it. The promise belongs next to the only code that can keep it.
    //
    // AT MOST ONE AT A TIME, and that is a decision rather than a simplification. A preview is a
    // client hovering over a node in an editor: there is one cursor, the answer arrives in a
    // frame or two, and a queue would mostly hold requests nobody is waiting for any more. A
    // second request REPLACES the first and the first is ANSWERED -- a promise destroyed without
    // a value makes its future throw `broken_promise`, which reaches a client as an internal
    // error for something that is not an error at all.
    //
    // Its own mutex: written from the API executor, read from the render thread.
    struct pending_preview
    {
        //: BY DOCUMENT NAME, not by layer. The mixer sees a tree of layers and items and carries
        //: no stage layer index at all, so `node_plan::document_name` is the only thing a
        //: request can be matched against -- and addressing by document is the better interface
        //: anyway, for the reason `detach` takes no layer: a client knows the look's name.
        //: Unique for the life of the process, so "is this the request I saw last frame" is
        //: answerable without a pointer that the next allocation may reuse.
        std::uint64_t                          id = 0;
        std::string                                         graph_name;
        //: IN THE ORDER ASKED, and answered in that order, so a client can zip the replies onto
        //: its own list of nodes without matching on ids it already knows.
        std::vector<std::string>                            node_ids;
        //: Cap on the longest edge of each picture; 0 means the layer's raster.
        int                                                 max_edge = 0;
        std::promise<std::vector<core::node_preview_image>> promise;
    };

    //: The size a preview should be read back at, given the layer's raster and the client's cap.
    //:
    //: THE CAP IS ON THE LONGEST EDGE and the aspect is kept, because a thumbnail that changed
    //: shape would be useless for judging a grade. Never enlarges: asking for 4096 of a 1080p
    //: layer gets 1080p, not an upscale nobody asked to pay for.
    static void preview_extent(int src_w, int src_h, int max_edge, int& out_w, int& out_h)
    {
        out_w = src_w;
        out_h = src_h;
        if (max_edge <= 0 || src_w <= 0 || src_h <= 0)
            return;
        const int longest = std::max(src_w, src_h);
        if (longest <= max_edge)
            return;
        const double k = static_cast<double>(max_edge) / static_cast<double>(longest);
        out_w          = std::max(1, static_cast<int>(src_w * k + 0.5));
        out_h          = std::max(1, static_cast<int>(src_h * k + 0.5));
    }

    //: The channel's frame number and rate, set once per tick by `video_channel`. An ISF node
    //: derives `TIME`, `TIMEDELTA` and `FRAMEINDEX` from these. Plain members rather than part
    //: of any transform, so they never reach the still-frame fingerprint -- see
    //: `core::image_mixer::set_frame_number`.
    std::atomic<std::uint64_t>       channel_frame_{0};
    std::atomic<double>              channel_fps_{0.0};

    std::mutex                       preview_lock_;
    std::unique_ptr<pending_preview> preview_;
    //: A MONOTONIC ID, not a pointer, and the difference is a real defect rather than taste.
    //:
    //: The first version of this compared `preview_.get()` against the raw pointer seen at the
    //: top of the previous frame, to answer "has this request survived a whole frame unclaimed".
    //: But the object it named is DESTROYED as soon as the evaluator claims and fulfils it, so
    //: the stored pointer dangles -- and the very next request is a fresh allocation that the
    //: allocator is free to place at exactly that address. Under sustained polling that is not a
    //: remote possibility, it is the common case: same size, same thread, immediately after the
    //: free. The comparison then says "you have been here a whole frame" about a request that
    //: arrived microseconds ago, and refuses it with `no layer on this channel is rendering a
    //: graph named ...` while the graph is right there.
    //:
    //: A counter cannot be recycled, so the question it answers stays the question it was asked.
    std::uint64_t                    preview_next_id_ = 1;
    std::uint64_t                    preview_seen_id_ = 0;

    //: "Drop the still-frame cache on your next tick", set by the API executor and acted on by
    //: the render thread -- because `prev_fingerprint_` owns a vector and is the render
    //: thread's alone. An atomic flag rather than a lock: the render path would otherwise take
    //: a mutex every frame to answer a question that is almost always "no".
    std::atomic<bool>                preview_drop_cache_{false};

    //: A preview that has been DRAWN this frame and not yet copied back. Resolved after the
    //: frame's single `commit()`, because a mid-frame commit corrupts the command stream --
    //: see the serve site. Render-thread only, so no lock.
    //: Previews DRAWN this frame and not yet copied back, resolved after the frame's single
    //: `commit()` -- a mid-frame commit corrupts the command stream, see the serve site.
    //: `deferred_preview_slots_` maps the pass's readbacks, in registration order, onto the
    //: slots of `deferred_preview_out_`. Render-thread only, so no lock.
    std::unique_ptr<pending_preview>      deferred_preview_;
    std::vector<std::size_t>              deferred_preview_slots_;
    std::vector<std::shared_ptr<texture>> deferred_preview_targets_;
    std::vector<core::node_preview_image> deferred_preview_out_;

    std::future<std::vector<core::node_preview_image>>
    arm_node_preview(const std::string& graph_name, const std::vector<std::string>& node_ids, int max_edge)
    {
        auto req        = std::make_unique<pending_preview>();
        req->graph_name = graph_name;
        req->node_ids   = node_ids;
        req->max_edge   = max_edge;
        auto fut        = req->promise.get_future();

        std::lock_guard<std::mutex> lock(preview_lock_);
        req->id = preview_next_id_++;
        if (preview_) {
            std::vector<core::node_preview_image> superseded(preview_->node_ids.size());
            for (auto& o : superseded)
                o.reason = "superseded by a later preview request";
            preview_->promise.set_value(std::move(superseded));
        }
        preview_ = std::move(req);

        // AND THE STILL-FRAME CACHE HAS TO BE DROPPED, or the preview can only ever be served
        // on a frame that was going to be composited anyway.
        //
        // MEASURED, and it is the whole reason this exists: a static colour under a static graph
        // is exactly the cached case, so the evaluator never ran, the request was never seen,
        // and every preview timed out at two seconds while the channel was demonstrably running
        // and the graph demonstrably attached. The symptom -- "is the channel running?" --
        // pointed at everything except the cache.
        //
        // ASKED FOR WITH A FLAG, NOT DONE HERE, and the first version did it here: a bare
        // `prev_fingerprint_ = {}` on this line. THIS FUNCTION RUNS ON THE API EXECUTOR THREAD
        // and `prev_fingerprint_` belongs to the RENDER thread, which reads it through
        // `matches()` and overwrites it through `std::move` on every single frame. It holds a
        // `std::vector<item_fingerprint>`, so assigning an empty one from here frees that
        // buffer while the render thread may be walking it or assigning over it -- a data race
        // on a heap-owning object, once per preview request.
        //
        // `preview_lock_` did NOT make it safe, which is what made it easy to write: the lock is
        // held here, and the render thread never takes it for the fingerprint, so it guarded
        // nothing at all.
        //
        // WHAT IT COST, because the symptoms pointed everywhere but here. Under sustained
        // polling it corrupted the heap, and corruption surfaces wherever it lands rather than
        // where it was caused: `std::bad_array_new_length` from a vector with a garbage length
        // on the OpenGL mixer, 16157 access violations at heap addresses on Vulkan, four
        // channels ceasing to tick, and -- the one that cost the most -- `grade-graph`'s mask
        // checks failing with masks that graded the WHOLE IMAGE, in code nothing had touched
        // for two commits. That last symptom was first attributed to the stale-vtable build
        // trap and "fixed" by a full rebuild; the rebuild changed nothing and the run that
        // followed it simply did not hit the race. See CLAUDE.md, where that misattribution is
        // now corrected.
        //
        // The render thread clears it at the top of its own tick. Dropping the FINGERPRINT
        // rather than the texture: the next tick composites once, serves the request, and
        // re-caches -- one extra composition per preview, for something a human asked for.
        preview_drop_cache_.store(true, std::memory_order_release);
        return fut;
    }

  private:
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
        // ── A PREVIEW THIS CHANNEL CANNOT SERVE IS REFUSED, AFTER ONE FULL FRAME ───────
        //
        // The shell resolves a preview by ARMING EVERY CHANNEL and waiting on each one's
        // future in turn, so a channel that is not drawing the requested document has to SAY
        // so. Before this it simply never answered, and the wait is two seconds -- so a graph
        // on channel 4 of 4 cost SIX SECONDS before the first byte while channels 1-3 timed
        // out in series. Measured at 6099 ms against 79 ms on channel 1 (`node-preview-cost`),
        // which is the difference between an editor that feels live and one that looks broken.
        //
        // REFUSED ON THE FRAME AFTER THE ONE THAT SAW IT, never on the frame it arrived: a
        // request armed while this frame was already mid-draw has not had its chance yet, and
        // rejecting it immediately would fail a perfectly good request that merely raced the
        // tick. Comparing the POINTER is what makes "has this survived a whole frame"
        // answerable at all -- a new request is a new object, so every request gets a frame of
        // its own whatever arrives after it.
        //
        // AT THE TOP, BEFORE EVERY EARLY RETURN, and that is the reason it is here rather than
        // at the end of the draw: this function returns early for an empty layer list and
        // again for a still-frame cache hit, and an idle channel is exactly the one that will
        // never claim a request. Putting the refusal after the evaluator would leave the two
        // commonest cases waiting the full two seconds.
        // THE STILL-FRAME CACHE, DROPPED HERE RATHER THAN BY THE THREAD THAT ASKED.
        // `prev_fingerprint_` owns a vector and is read and overwritten by this thread every
        // frame; clearing it from the API executor was a data race that corrupted the heap.
        // See `arm_node_preview`.
        if (preview_drop_cache_.exchange(false, std::memory_order_acq_rel)) {
            prev_fingerprint_ = {};
        }

        {
            std::lock_guard<std::mutex> lock(preview_lock_);
            if (preview_ && preview_->id == preview_seen_id_) {
                std::vector<core::node_preview_image> out(preview_->node_ids.size());
                for (auto& o : out)
                    o.reason = "no layer on this channel is rendering a graph named '" +
                               preview_->graph_name + "'";
                preview_->promise.set_value(std::move(out));
                preview_.reset();
            }
            preview_seen_id_ = preview_ ? preview_->id : 0;
        }

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

                // ── DEFERRED NODE PREVIEWS, now that the pass has been issued ──────────
                //
                // The copies could not happen inside the evaluator: `renderpass::draw` only
                // queues, and committing mid-frame to make a target readable corrupts the
                // stream the rest of the frame is still being written into. Here the pass is
                // issued, its command buffer already contains every copy, and all that is left
                // is to hand out futures that wait on it.
                if (deferred_preview_) {
                    for (std::size_t n = 0; n < deferred_preview_slots_.size(); ++n) {
                        const auto slot = deferred_preview_slots_[n];
                        if (n >= deferred_preview_targets_.size() || slot >= deferred_preview_out_.size())
                            continue;
                        auto& out = deferred_preview_out_[slot];
                        // SCALED ON THE WAY OUT, which is where the cost is: the readback is
                        // priced in pixels. `copy_async_scaled` owns its whole command buffer,
                        // so the layouts are its own to reason about -- an earlier version
                        // recorded the blit into the FRAME's command buffer to save a submit
                        // and produced undefined contents (right, dark and black on successive
                        // runs). The submit was measured not to be the cost anyway.
                        out.pending =
                            vulkan_->copy_async_scaled(deferred_preview_targets_[n], out.width, out.height)
                                .share();
                    }
                    deferred_preview_->promise.set_value(std::move(deferred_preview_out_));
                    deferred_preview_.reset();
                    deferred_preview_slots_.clear();
                    deferred_preview_targets_.clear();
                    deferred_preview_out_.clear();
                }

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

            // Mirror of the OpenGL mixer, including the reason both members are read out
            // BEFORE the move: they live in `draw_params` and reading them afterwards is UB.
            std::shared_ptr<const core::graph::node_plan> plan =
                draw_params.transforms.image_transform.node_plan;
            const auto values = draw_params.transforms.image_transform.node_values;

            // WHICH STEPS ARE LIVE, decided here and not in the compiler, because `bypass` and
            // `mute` are VALUES: the plan is identical whether a node is bypassed or not, which
            // is what lets a timeline step one without reallocating a graph mid-show.
            //
            // A bypassed step ALIASES its primary input -- no draw, no attachment -- so a graph
            // with a bypassed node renders byte-identically to the graph without it. A step
            // whose primary input is dead is itself dead, and "dead" resolves to the INPUT
            // rather than to black.
            std::vector<int> alias;
            int              live_passes = 0;
            if (plan && !plan->steps.empty()) {
                alias.resize(plan->steps.size(), -1);
                for (std::size_t i = 0; i < plan->steps.size(); ++i) {
                    const auto& st = plan->steps[i];
                    // `bypass` is every class's FIRST value port by construction, so its slot
                    // is the step's own offset.
                    const bool bypassed = st.values_count > 0 &&
                                          st.values_offset < values.size() &&
                                          values[st.values_offset] != 0.0;
                    const int primary = st.in0 >= 0 ? alias[st.in0] : -1;

                    // A MATERIALISED MASK IS A LIVE PASS -- see the OpenGL mixer for the full
                    // account. It produces no image, so `produces_image` does not count it, but
                    // it costs a draw and owns an attachment.
                    if (st.produces_mask && !bypassed) {
                        alias[i] = static_cast<int>(i);
                        ++live_passes;
                        continue;
                    }
                    if (!st.produces_image) {
                        alias[i] = st.in0 < 0 ? static_cast<int>(i) : primary;
                        continue;
                    }
                    if (bypassed || primary < 0) {
                        alias[i] = primary < 0 ? 0 : primary;
                        continue;
                    }
                    alias[i] = static_cast<int>(i);
                    ++live_passes;
                }
            }

            // No graph = the path that existed before this feature, unchanged: one draw
            // straight into the target, no attachment. `live_passes == 0` covers no graph, an
            // empty graph and a graph with everything bypassed.
            // ── THE ITEM'S PLACEMENT, INVERTED, FOR SOURCE-SPACE MASKS ──────────────
            //
            // Computed ONCE per graphed layer rather than per node: every node pass of one
            // layer masks through the same geometry, and the matrix costs a 3x3 inverse.
            //
            // `apply_geometry_scale_mode` FIRST, because the kernel applies it inside `draw()`
            // and the mask has to agree with where the picture actually landed -- a 4K clip
            // `fit` into an HD channel is placed by that scale and by nothing in
            // `item.transforms`.
            //
            // A layer whose placement is not invertible (a corner-pin `perspective`, a zero
            // scale) leaves `node_uv_valid` false and its masks stay in FRAME space. That is
            // deliberately a visible answer rather than an approximation: a mask that plainly
            // did not follow the picture is diagnosable, and one that followed it to somewhere
            // plausible and wrong is not.
            // FROM `draw_params`, NOT `item`: `item.transforms`, `item.geometry` and
            // `item.pix_desc` were all MOVED into `draw_params` above, so reading them here
            // would read moved-from objects -- an empty plane list and an identity placement,
            // which is a mask that silently never follows anything.
            std::array<float, 9> node_uv_inv{1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f};
            const bool           node_uv_valid = source_uv_inverse(
                apply_geometry_scale_mode(draw_params.transforms,
                                          draw_params.geometry,
                                          draw_params.target_width,
                                          draw_params.target_height,
                                          draw_params.pix_desc.planes.empty()
                                              ? 0
                                              : draw_params.pix_desc.planes[0].width,
                                          draw_params.pix_desc.planes.empty()
                                              ? 0
                                              : draw_params.pix_desc.planes[0].height,
                                          draw_params.aspect_ratio),
                node_uv_inv);

            // THE ITEM'S PARAMS, COPIED BEFORE THE HEAD DRAW MOVES THEM. The tail needs the
            // item's colour configuration -- that is the correction over the first attempt at
            // this commit, which built fresh params and so got the channel's conversion instead
            // of the layer's.
            struct draw_params item_params_for_tail;
            const bool         working_stage =
                plan && plan->stage == core::graph::graph_stage::working;

            std::shared_ptr<texture> head_texture;
            if (live_passes > 0) {
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
                // fp16, REGARDLESS OF THE CHANNEL'S FORMAT -- see the OpenGL mixer for the
                // full account. A node's intermediate is a value on its way to the next node,
                // and those legitimately exceed 1.0; a unorm attachment clips them, which is
                // the same reason `<working-space-composite>` refuses to run on one.
                //
                // AND ON THIS BACKEND THE ATTACHMENT FORMAT IS HALF THE CHANGE: the pipeline
                // carries the colour-attachment format in its own creation info, so writing
                // fp16 through a unorm pipeline is a format mismatch rather than a conversion.
                // `apply_node` sets `draw_params.node_fp16` and the kernel hands back the
                // matching pipeline through the existing per-layer hook.
                head_texture           = pass->create_attachment_as(common::render_format::fp16);
                draw_params.node_fp16  = true;
                draw_params.background = head_texture;
            } else {
                draw_params.background = target_texture;
            }
            draw_params.local_key  = std::move(local_key_texture);
            draw_params.layer_key  = layer_key_texture;

            // ── WORKING SPACE: the head stops at the boundary ───────────────────────
            //
            // `stage: working` is the default and the design: the nodes then run where
            // `MIXER CDL` already does, and the output half is applied ONCE by the tail
            // against the real target. `stage: display` sets nothing here, so that graph
            // keeps the prototype's placement byte for byte -- which is what makes the stage
            // a compatibility guarantee rather than a label.
            if (live_passes > 0 && working_stage) {
                draw_params.graph_head = true;
                item_params_for_tail   = draw_params;
            }

            pass->draw(std::move(draw_params));

            if (live_passes > 0) {
                // ONE TEXTURE PER STEP, and an ALIAS is a `shared_ptr` copy rather than a draw
                // -- so fan-out is free: two consumers of one output hold the same attachment.
                //
                // NOTE FOR THE VULKAN SIDE SPECIFICALLY: `renderpass::commit()` transitions
                // each attachment to `eShaderReadOnlyOptimal` as the pass writing it ends, so
                // an earlier attachment a later step READS is already in the right layout. What
                // is NOT proven is a RECYCLED one -- `last_use` returns an attachment to the
                // pool and the next `create_attachment()` may hand the same object back to be
                // written as a colour attachment, with nothing transitioning it out of
                // shader-read. Flagged rather than fixed here (the plan's G1), and
                // `vk-validation` cannot help: it reports clean whatever you do.
                std::vector<std::shared_ptr<texture>> outputs(plan->steps.size());
                outputs[0] = head_texture;

                // WHICH STEPS THE PENDING PREVIEW WANTS, resolved once before the loop.
                //
                // ONE PASS OVER THE REQUEST, producing a slot per asked-for node with a reason
                // already filled in for the ones this graph cannot answer. Claimed here so the
                // request is taken exactly once whatever the graph looks like, and so a node
                // this document does not have is refused immediately rather than after a whole
                // frame of drawing.
                //
                // A BAD NODE ID REFUSES ITS OWN THUMBNAIL, NOT THE STRIP. An editor asking for
                // sixteen previews while the operator deletes one node should get fifteen
                // pictures and one reason.
                std::unique_ptr<pending_preview>      preview_req;
                std::vector<std::int32_t>             preview_want;
                std::vector<core::node_preview_image> preview_out;
                {
                    std::lock_guard<std::mutex> lock(preview_lock_);
                    if (preview_ && preview_->graph_name == plan->document_name)
                        preview_req = std::move(preview_);
                }
                if (preview_req) {
                    preview_want.assign(preview_req->node_ids.size(), -1);
                    preview_out.resize(preview_req->node_ids.size());
                    for (std::size_t q = 0; q < preview_req->node_ids.size(); ++q) {
                        const auto& id = preview_req->node_ids[q];
                        // FROM THE CHANNEL THAT HAS THE DOCUMENT, so the shell keeps these
                        // reasons over other channels' "not here".
                        preview_out[q].from_matching_graph = true;
                        std::int32_t found = -1;
                        for (std::size_t k = 0; k < plan->steps.size(); ++k)
                            if (plan->steps[k].id == id)
                                found = static_cast<std::int32_t>(k);
                        if (found < 0) {
                            preview_out[q].reason = "no node '" + id + "' in the attached graph";
                        } else if (alias[found] != found) {
                            // BYPASSED OR DEAD: it aliases something else and has no output of
                            // its own. Saying so beats previewing what it aliases.
                            preview_out[q].reason = "node '" + id +
                                                    "' is bypassed or unreachable, so it has no "
                                                    "output this frame";
                        } else {
                            preview_want[q] = found;
                        }
                    }
                    if (std::none_of(preview_want.begin(), preview_want.end(),
                                     [](std::int32_t w) { return w >= 0; })) {
                        // Nothing servable: answer now rather than hold the request for a frame
                        // that cannot help it.
                        preview_req->promise.set_value(std::move(preview_out));
                        preview_req.reset();
                    }
                }


                for (std::size_t i = 0; i < plan->steps.size(); ++i) {
                    const auto& st = plan->steps[i];
                    if (alias[i] != static_cast<int>(i)) {
                        if (alias[i] >= 0)
                            outputs[i] = outputs[alias[i]];
                        continue;
                    }
                    if (!st.produces_image && !st.produces_mask)
                        continue;

                    core::graph::node_draw nd;
                    nd.op           = st.cls;
                    nd.values       = st.values_count ? values.data() + st.values_offset : nullptr;
                    nd.values_count = st.values_count;
                    nd.has_in1 = st.in1 >= 0 && alias[st.in1] >= 0 && outputs[alias[st.in1]];
                    // THE THREE SHAPES A MASK CAN REACH A CONSUMER IN, the third of which was
                    // declared in `node_draw.h` and read by nothing -- so a mask with two
                    // consumers was neither inlined nor sampled and both consumers graded the
                    // WHOLE image. `graph_plan_self_test` now asserts fused and materialised are
                    // exhaustive.
                    std::shared_ptr<texture> mask_texture;
                    if (st.mask >= 0) {
                        const auto& mst = plan->steps[st.mask];
                        if (mst.fused_mask && mst.values_count) {
                            nd.mask_values = values.data() + mst.values_offset;
                            // WHICH SHAPE, because the consumer evaluates it and `nd.op` is the
                            // consumer's own class. Without this every fused mask is an ellipse.
                            nd.mask_op = mst.cls;
                        }
                        else if (alias[st.mask] >= 0 && outputs[alias[st.mask]]) {
                            mask_texture        = outputs[alias[st.mask]];
                            nd.has_mask_texture = true;
                        }
                    }

                    auto dst = pass->create_attachment_as(common::render_format::fp16);
                    // A MASK GENERATOR HAS NO IMAGE INPUT, so it is handed the head texture as
                    // a source it never samples: the pass computes its value from its own
                    // uniforms, and `apply_node` returns early on a null source.
                    // `in0 < 0` RATHER THAN `produces_mask`, and the difference is two classes:
                    // `mask_combine` reads two masks and `mask_qualifier` reads the pixel it is
                    // keying, so both have a real `in0`. Substituting the head texture for every
                    // mask pass fed the PICTURE in as `a` -- a combine then read the layer's red
                    // channel as a mask, and two of its three operators passed for the wrong
                    // reason. Only a generator with NO input needs a source it will not sample.
                    const auto& src0 =
                        st.in0 < 0 ? head_texture : outputs[alias[st.in0]];

                    // ── AN ISF NODE NAMES ITS SHADER, AND CARRIES THE CHANNEL'S CLOCK ────
                    //
                    // A POINTER INTO THE PLAN's own string table, which is immutable and shared
                    // for the plan's whole life -- so this costs nothing per frame and stays
                    // valid for as long as the draw does.
                    //
                    // The kernel turns it into a pipeline. Not here: the kernel already owns the
                    // OCIO variant cache and the device handle that builds one, and a second
                    // cache would be a second answer to "have I compiled this shader".
                    const std::string* isf_path = nullptr;
                    if (st.cls == core::graph::op_isf && st.string_index >= 0 &&
                        static_cast<std::size_t>(st.string_index) < plan->strings.size())
                        isf_path = &plan->strings[st.string_index];

                    // WHICH WAY TO CONVERT, from `space` against the graph's stage. BY NAME
                    // through `value_names`, for the reason the OpenGL evaluator gives at the
                    // same spot: the last time this counted ports instead of asking, every
                    // parameter landed three slots early and still rendered.
                    int space_idx = -1;
                    if (isf_path && plan->value_names.size() >= values.size())
                        for (std::uint32_t q = 0; q < st.values_count; ++q)
                            if (plan->value_names[st.values_offset + q] == "space")
                                space_idx = static_cast<int>(st.values_offset + q);
                    const int space_v =
                        space_idx >= 0 && static_cast<std::size_t>(space_idx) < values.size()
                            ? static_cast<int>(values[space_idx])
                            : 0;
                    const bool stage_display = plan->stage == core::graph::graph_stage::display;
                    const int  isf_to_display = space_v == 1 && !stage_display   ?  1
                                                : space_v == 2 && stage_display  ? -1
                                                                                 :  0;

                    const auto fr  = channel_frame_.load(std::memory_order_relaxed);
                    const auto fps = channel_fps_.load(std::memory_order_relaxed);

                    apply_node(src0,
                               nd.has_in1 ? outputs[alias[st.in1]] : src0,
                               dst, format_desc, pass, nd, node_uv_inv, node_uv_valid,
                               mask_texture, isf_path,
                               fps > 0.0 ? static_cast<double>(fr) / fps : 0.0,
                               fps > 0.0 ? 1.0 / fps : 0.0,
                               static_cast<int>(fr), isf_to_display);
                    outputs[i] = dst;

                    // ── SERVE EVERY SLOT THAT WANTED THIS STEP ────────────────────
                    //
                    // DRAWN AND REGISTERED NOW, COPIED WHEN THE FRAME COMMITS. An earlier
                    // version called `pass->commit()` right here, which is what a mid-frame
                    // commit costs on this backend: the renderpass is still ACCUMULATING
                    // layers, so committing and then continuing to draw into it corrupts the
                    // command stream -- 241 `vk::Queue::submit: ErrorDeviceLost` in one run,
                    // after a standalone run had passed, which is the kind of luck that ships.
                    //
                    // Each target is registered with the pass, which records the copy -- and
                    // the downscaling blit -- into the frame's own command buffer.
                    if (preview_req) {
                        for (std::size_t q = 0; q < preview_want.size(); ++q) {
                            if (preview_want[q] != static_cast<std::int32_t>(i))
                                continue;
                            auto pdst = pass->create_attachment();
                            apply_graph_tail(outputs[i], pdst, format_desc, pass, item_params_for_tail);
                            int pw = 0, ph = 0;
                            preview_extent(static_cast<int>(pdst->width()),
                                           static_cast<int>(pdst->height()),
                                           preview_req->max_edge, pw, ph);
                            // The size that COMES BACK is the blit's target, not the
                            // attachment's -- a client decodes what it was sent.
                            preview_out[q].width  = pw;
                            preview_out[q].height = ph;
                            deferred_preview_slots_.push_back(q);
                            deferred_preview_targets_.push_back(pdst);
                        }
                    }


                    for (std::size_t j = 0; j < i; ++j)
                        if (plan->steps[j].last_use == static_cast<std::int32_t>(i))
                            outputs[j].reset();
                }

                // The set moves to the deferred state; the copies land when the pass commits.
                if (preview_req) {
                    deferred_preview_out_ = std::move(preview_out);
                    deferred_preview_     = std::move(preview_req);
                }

                const auto& out_step  = plan->steps.back();
                auto        final_tex = out_step.in0 >= 0 && alias[out_step.in0] >= 0
                                            ? outputs[alias[out_step.in0]]
                                            : head_texture;
                if (working_stage)
                    apply_graph_tail(final_tex, target_texture, format_desc, pass,
                                     item_params_for_tail);
                else
                    draw(target_texture, std::move(final_tex), format_desc, pass,
                         core::blend_mode::normal);
            }
        }
    }

    /// One node's full-screen pass. Mirror of the OpenGL mixer's version, and modelled on
    /// `apply_calibration_lut` below for the same reason: sources in `textures` sampled as
    /// ordinary sampler2Ds (not through `subpassInput background`), destination in
    /// `background`, both conversion halves off.
    ///
    /// TWO SOURCES, because `mix` and `over` need them. A class with one input gets the same
    /// texture twice rather than a null second slot: the shader reads the second only when
    /// `has_in1` says so, and binding nothing would make an unconnected `mix.b` sample garbage
    /// instead of returning `a`.
    void apply_node(const std::shared_ptr<texture>& source_a,
                    const std::shared_ptr<texture>& source_b,
                    std::shared_ptr<texture>&       target_texture,
                    const core::video_format_desc&  format_desc,
                    spl::shared_ptr<renderpass>     pass,
                    const core::graph::node_draw&   nd,
                    const std::array<float, 9>&     node_uv_inv,
                    bool                            node_uv_valid,
                    const std::shared_ptr<texture>& mask_texture = nullptr,
                    /// Set only for an `isf` node: the shader file its `path` names. The kernel
                    /// compiles it to a variant pipeline and caches it by this string.
                    const std::string*              isf_path     = nullptr,
                    double                          isf_time     = 0.0,
                    double                          isf_dt       = 0.0,
                    int                             isf_frame    = 0,
                    int                             isf_to_display = 0)
    {
        if (!source_a)
            return;

        draw_params draw_params;
        draw_params.target_width  = format_desc.square_width;
        draw_params.target_height = format_desc.square_height;
        // 8-bit attachments store BGRA (shader .bgra swizzle); 16-bit store RGBA directly.
        // Getting this wrong exchanges red and blue, and the node's own operation is a
        // uniform scale that would not reveal it.
        draw_params.pix_desc.format = (source_a->depth() == common::bit_depth::bit8)
                                          ? core::pixel_format::bgra
                                          : core::pixel_format::rgba;
        draw_params.pix_desc.planes = {core::pixel_format_desc::plane(
            source_a->width(), source_a->height(), 4, source_a->depth())};
        draw_params.pix_desc.color_space    = target_color_space;
        draw_params.pix_desc.color_transfer = target_color_transfer;
        draw_params.target_color_space      = target_color_space;
        draw_params.target_color_transfer   = target_color_transfer;
        draw_params.auto_color_convert      = false;
        draw_params.auto_tone_map           = 0;
        // THIRD SLOT IS THE MASK: `in0` -> PLANE0, `in1` -> PLANE1, `mask` -> PLANE2, each
        // sampled raw. Bound unconditionally so the slot never holds a previous draw's image;
        // `shader_flags2::node_mask_tex` is what decides whether the shader reads it.
        draw_params.textures                = {spl::make_shared_ptr(source_a),
                                               spl::make_shared_ptr(nd.has_in1 ? source_b : source_a),
                                               spl::make_shared_ptr(mask_texture ? mask_texture
                                                                                 : source_a)};
        draw_params.blend_mode              = core::blend_mode::normal;
        draw_params.background              = target_texture;
        draw_params.geometry                = core::frame_geometry::get_default();
        draw_params.node                    = nd;
        draw_params.isf_path                = isf_path;
        draw_params.isf_time                = isf_time;
        draw_params.isf_time_delta          = isf_dt;
        draw_params.isf_frame               = isf_frame;
        draw_params.isf_to_display          = isf_to_display;
        // The destination is an fp16 attachment, so this draw needs the fp16 pipeline. See the
        // head pass above for why the format is not just a property of the image here.
        draw_params.node_fp16               = true;
        // Per-LAYER, the same for every node pass of this layer. The kernel ANDs it with the
        // mask's own `space` port, so a `frame`-space mask is unaffected by its presence.
        draw_params.node_uv_inv             = node_uv_inv;
        draw_params.node_uv_valid           = node_uv_valid;

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

    /// The TAIL of a working-space graph: the layer's output half, applied once.
    ///
    /// Mirror of the OpenGL mixer's version, where the full account lives. The one thing worth
    /// repeating here because it is the correction over the first attempt at this commit: it
    /// takes the ITEM'S OWN `draw_params` rather than building fresh ones. The kernel decides
    /// what to convert from `transforms.image_transform.color_grade`, `auto_color_convert` and
    /// `pix_desc`'s colour fields, so a tail carrying none of them gets the CHANNEL's conversion
    /// instead of the LAYER's -- which converts a layer that converts nothing, and uses the
    /// wrong transfer for a layer under `MIXER COLORSPACE`.
    ///
    /// A DEFAULT `image_transform` carrying only `color_grade`, so the grading chain does not run
    /// a second time: every operator's enable is at its default. `color_grade` is not an operator
    /// -- it is the conversion configuration -- which is why it is the one field copied.
    void apply_graph_tail(const std::shared_ptr<texture>& source_texture,
                          std::shared_ptr<texture>&       target_texture,
                          const core::video_format_desc&  format_desc,
                          spl::shared_ptr<renderpass>     pass,
                          const draw_params&              item)
    {
        if (!source_texture)
            return;

        draw_params tail;
        tail.target_width  = format_desc.square_width;
        tail.target_height = format_desc.square_height;

        // The PLANE geometry is the attachment's; the COLOUR metadata is the item's, so every
        // branch in the kernel decides exactly as it did for the head. `node_fp16` stays FALSE:
        // the tail READS the fp16 attachment and WRITES the channel's own format, and on this
        // backend that flag selects the pipeline's colour-attachment format rather than the
        // sampler's.
        tail.pix_desc.format = (source_texture->depth() == common::bit_depth::bit8)
                                   ? core::pixel_format::bgra
                                   : core::pixel_format::rgba;
        tail.pix_desc.planes = {core::pixel_format_desc::plane(
            source_texture->width(), source_texture->height(), 4, source_texture->depth())};
        tail.pix_desc.color_space    = item.pix_desc.color_space;
        tail.pix_desc.color_transfer = item.pix_desc.color_transfer;

        tail.target_color_space      = item.target_color_space;
        tail.target_color_transfer   = item.target_color_transfer;
        tail.auto_color_convert      = item.auto_color_convert;
        tail.auto_tone_map           = item.auto_tone_map;
        tail.display_peak_luminance  = item.display_peak_luminance;
        tail.sdr_reference_white     = item.sdr_reference_white;
        tail.working_space_composite = item.working_space_composite;
        tail.straight_alpha_grading  = item.straight_alpha_grading;
        tail.ocio_display            = item.ocio_display;
        tail.ocio_view               = item.ocio_view;
        tail.ocio_look               = item.ocio_look;

        // ONLY `color_grade`. Everything else defaults, so no grading operator runs twice.
        tail.transforms.image_transform.color_grade = item.transforms.image_transform.color_grade;

        // THE KEYS ARE NOT RE-APPLIED. The head pass already consumed them -- they mask the
        // ITEM -- and applying them twice would double-darken every soft edge. `grade-window`'s
        // composite check and `alpha-domain` are what adjudicate that.
        tail.graph_tail = true;
        tail.textures   = {spl::make_shared_ptr(source_texture)};
        tail.blend_mode = core::blend_mode::normal;
        tail.background = target_texture;
        tail.geometry   = core::frame_geometry::get_default();

        pass->draw(std::move(tail));
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
    std::atomic<ogl::previz_renderer*>           previz_ready_{nullptr};
    mutable core::fields::stage_publisher        previz_publisher_;
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

    /// Forwarded to the RENDERER, which is where the request can actually be served.
    void set_frame_number(std::uint64_t frame, double fps)
    {
        renderer_.channel_frame_.store(frame, std::memory_order_relaxed);
        renderer_.channel_fps_.store(fps, std::memory_order_relaxed);
    }

    std::future<std::vector<core::node_preview_image>>
    arm_node_preview(const std::string& graph_name, const std::vector<std::string>& node_ids, int max_edge)
    {
        return renderer_.arm_node_preview(graph_name, node_ids, max_edge);
    }

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
            // Published AFTER the unique_ptr is set, and read by `stage_state()` below from a
            // different thread. `previz_renderer_` itself cannot be read from there: this
            // `call_once` runs on whichever thread asked for the renderer first, so a plain read
            // of the pointer racing this write is undefined.
            previz_ready_.store(previz_renderer_.get(), std::memory_order_release);
            CASPAR_LOG(info) << L"[vk_mixer] Created previz renderer for channel " << channel_id_;
        });
        return previz_renderer_.get();
    }

    bool stage_input(const core::input_event& event)
    {
        // Deliberately NOT `get_previz_renderer()`, for the same reason `stage_state()` avoids
        // it: that call CREATES the renderer, and a stray mouse move must not bring a GL device
        // into existence on a Vulkan channel that never uses previz.
        auto* r = previz_ready_.load(std::memory_order_acquire);
        if (!r)
            return false;

        // `aspect_ratio_` is maintained per tick by `update_aspect_ratio`, so it is both
        // correct and already there.
        return r->input(event, aspect_ratio_);
    }

    core::monitor::state stage_state() const
    {
        // Deliberately NOT `get_previz_renderer()`. That CREATES the renderer on first call, and
        // a per-tick publisher must not be the thing that brings a GL device into existence on a
        // Vulkan channel that never uses previz. Until somebody asks for previz, this publishes
        // nothing and costs an atomic load.
        auto* r = previz_ready_.load(std::memory_order_acquire);
        if (!r)
            return {};
        previz_publisher_.refresh(r->stage_snapshot());
        return previz_publisher_.published();
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

void image_mixer::set_frame_number(std::uint64_t frame, double fps) { impl_->set_frame_number(frame, fps); }

std::future<std::vector<core::node_preview_image>>
image_mixer::arm_node_preview(const std::string&              graph_name,
                              const std::vector<std::string>& node_ids,
                              int                             max_edge)
{
    return impl_->arm_node_preview(graph_name, node_ids, max_edge);
}
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

core::monitor::state image_mixer::state() const { return impl_->stage_state(); }

bool image_mixer::input(const core::input_event& event) { return impl_->stage_input(event); }

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
