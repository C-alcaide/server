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
#include <core/input/input_event.h>
#include <core/monitor/monitor.h>

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

/// ONE NODE'S OUTPUT, read back to the host.
///
/// `data` is 8-bit BGRA at `width` x `height`, which is the layout every readback in this server
/// already produces -- the same `array<const uint8_t>` a consumer receives. Empty when the
/// request could not be served, and the CALLER is told why rather than left to guess: an empty
/// buffer with a `reason` is the difference between "that node is not in the graph" and "the
/// frame did not arrive in time", which a client can act on differently.
/// The longest edge a preview is capped to when a client does not say otherwise.
///
/// 512 is an editor thumbnail: big enough to judge a grade on, and a fortieth of a 4K frame to
/// copy back. See `arm_node_preview` for why the DEFAULT is the cheap one.
constexpr int default_preview_edge = 512;

struct node_preview_image
{
    int         width  = 0;
    int         height = 0;
    std::string reason; ///< empty on success

    /// Did THIS channel have the document at all?
    ///
    /// The distinction exists because the shell asks EVERY channel and only one can be right,
    /// so the refusals it collects are not equal in value: "no node `e9` in the attached graph"
    /// comes from the channel that HAS the graph and is the answer the client needs, while "no
    /// layer on this channel is rendering a graph named `look`" is three other channels saying
    /// nothing useful. Without this flag the last channel's generic miss overwrites the one
    /// specific message, and a typo reports as a missing graph.
    bool from_matching_graph = false;

    /// THE READBACK, STILL PENDING -- and it has to be, which cost a deadlock to learn.
    ///
    /// The obvious shape is for this struct to carry the bytes, with the mixer calling `.get()`
    /// on the readback before fulfilling its promise. That SELF-DEADLOCKS: the serve site runs
    /// ON THE RENDER THREAD, inside the evaluator, and the readback is completed by that same
    /// thread -- so waiting for it there waits for work only the waiter can do. Measured as a
    /// preview that timed out after two seconds while the trace showed the request had been
    /// matched and served, which pointed at everything except the thread it was on.
    ///
    /// So the future is handed OUT. The shell resolves it on the API executor, where blocking
    /// is correct: that thread has nothing else to do until the preview arrives, and the render
    /// thread is never asked to wait on itself.
    std::shared_future<array<const std::uint8_t>> pending;

    bool ok() const { return width > 0 && height > 0 && reason.empty() && pending.valid(); }
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

    /// ARM A PREVIEW of one node's output, fulfilled on a later frame.
    ///
    /// ADDRESSED BY DOCUMENT NAME, not by layer, and that is forced as well as preferable: the
    /// mixer sees a tree of layers and items and carries no stage layer index at all, so the
    /// only thing it can match a request against is `node_plan::document_name`. It is also the
    /// better interface, for the reason `detach` takes no layer -- a client knows the look's
    /// name, and making it also remember where the look is attached gives it something to get
    /// wrong.
    ///
    /// A VIRTUAL ON THE BASE, so the control API reaches it through
    /// `mixer().get_image_mixer()` with no `dynamic_cast` and no dependency on either backend.
    /// `previz` is reached by a cast precisely because it is not on this interface, and the
    /// asymmetry is worth noticing: a cast works until somebody adds a third backend.
    ///
    /// ARMED RATHER THAN SYNCHRONOUS, because the attachment it wants does not exist yet. A
    /// node's output lives for the span of one `draw()` and is returned to the pool the moment
    /// nothing reads it, so the only place it can be copied is inside the frame that produces
    /// it. The request is recorded, the next frame that draws that layer's graph serves it, and
    /// the future is fulfilled from the render thread.
    ///
    /// WHAT IT RETURNS IS AFTER THE OUTPUT HALF, not the raw working-space attachment. A node's
    /// intermediate is scene-linear in the working gamut, and handing a client that as a PNG
    /// would show them a picture nothing will ever look like. The preview runs the same tail the
    /// layer's own draw runs, so what a client sees is what that node contributes to the
    /// finished frame.
    ///
    /// The default is a refusal, so a backend that has not implemented it says so rather than
    /// hanging: an unimplemented preview must not look like a slow one.
    /// The channel's frame number and its rate, set once per tick.
    ///
    /// FOR `TIME`, WHICH AN ISF SHADER ANIMATES FROM. The spec gives a shader `TIME`,
    /// `TIMEDELTA` and `FRAMEINDEX`, and all three derive from the channel's own counter -- the
    /// same one the timeline uses -- so two ISF nodes on a channel agree and a shader stays in
    /// step with everything else that is animating.
    ///
    /// **NOT ON `image_transform`, and that is the point of this seam.** A transform is compared
    /// field by field for the STILL-FRAME CACHE: a value that changes every frame would make
    /// every fingerprint unique, disable the cache for every channel, and cost a full
    /// composition per tick on layers that have not moved -- a serious regression to plumb one
    /// uniform. `set_consumer_views` beside it exists for the same reason and says so.
    ///
    /// A SETTER RATHER THAN AN ARGUMENT because `operator()` is the mixer's interface to the
    /// channel and every backend implements it; this is state the evaluator reads, not an input
    /// to a composition.
    virtual void set_frame_number(std::uint64_t /*frame*/, double /*fps*/) {}

    /// Arm a preview of one or more nodes of `graph_name`, served TOGETHER on the next frame
    /// that draws it.
    ///
    /// A SET RATHER THAN ONE NODE, because the frame is the scarce thing. A node's attachment
    /// exists only for the span of one `draw()`, so one request per node means one FRAME per
    /// node -- and the client this exists for is an editor showing what every step of a chain
    /// does, which wants all of them at once. Measured before this took a set: a 16-node strip
    /// cost 1294 ms because it was sixteen separate frames, and asking in parallel did not help
    /// at all, because the API executor owns a single thread and serialises them anyway.
    ///
    /// Answers come back IN THE ORDER ASKED, one per requested node, each with its own
    /// `reason` -- so one bad node id refuses its own thumbnail rather than the whole strip.
    ///
    /// `max_edge` caps the LONGEST SIDE of each picture in pixels, and is THE WHOLE OF THE
    /// COST. A preview is a readback and an encode, both priced in pixels: measured at 2160p50
    /// a full-raster preview copies 33 MB and costs about one late frame on the Vulkan mixer
    /// even at one request a second, while 512 costs 0.15% of frames at twenty-five a second.
    /// Folding the copy into the frame's own command buffer -- removing a whole queue submit --
    /// moved the number by less than run-to-run noise, which is what proved the volume was the
    /// cost rather than the mechanism.
    ///
    /// **0 MEANS THE LAYER'S OWN RASTER**, and the DEFAULT IS A THUMBNAIL rather than the
    /// raster. That is deliberate and was the other way round first: it defaulted to full
    /// raster "so a client that does not ask keeps what it had", which protected nobody --
    /// nothing consumes this API yet -- and left the common case paying the expensive path by
    /// not thinking about it. The cheap answer is the one you get for free; the expensive one
    /// is available by asking.
    virtual std::future<std::vector<node_preview_image>>
    arm_node_preview(const std::string&              /*graph_name*/,
                     const std::vector<std::string>& node_ids,
                     //: NO DEFAULT ARGUMENT, deliberately. One existed and was DEAD: the HTTP
                     //: route passes this explicitly on every call, so its value won -- and it
                     //: passed 0, meaning every request without `?max=` did a full-raster
                     //: readback while this header claimed a 512 thumbnail. A 16-node strip at
                     //: 2160p50 read 1001 ms that way against about 45 ms with the default
                     //: actually in force, and both the battery and the feature doc repeated
                     //: the slow number. **A default expressed in two places is a default in
                     //: neither**, so the only one lives at the route, which is where the
                     //: client's absence of a parameter is observed.
                     int                             /*max_edge*/)
    {
        std::vector<node_preview_image> out(node_ids.size());
        for (auto& o : out)
            o.reason = "this mixer does not implement node previews";
        std::promise<std::vector<node_preview_image>> p;
        p.set_value(std::move(out));
        return p.get_future();
    }

    /// What this backend wants published under `channel/{n}/mixer/`, every tick.
    ///
    /// A VIRTUAL rather than a `dynamic_cast` in the channel tick, and that is forced rather than
    /// preferred: AMCP reaches the previz renderer with `dynamic_cast<ogl::image_mixer*>` because
    /// `protocol` links `accelerator`. `core` does NOT and must not, so the same trick in
    /// `video_channel::impl::tick()` would invert the dependency. This interface is already the
    /// one thing both backends implement and `core` may name.
    ///
    /// Default is empty, so a backend with nothing to say costs nothing: assigning an empty state
    /// writes no keys at all.
    virtual monitor::state state() const { return {}; }

    /// Offer this backend a pointer or keyboard event. Returns true if it consumed it.
    ///
    /// The WRITE direction of the `state()` seam directly above, and it exists for the same
    /// reason: previz lives inside the mixer, `core` may not see the accelerator, and a
    /// `dynamic_cast` in the channel tick would invert the dependency. A backend that returns
    /// false has not looked at the event and the channel passes it on to the stage, so a
    /// channel with no previz behaves exactly as it did before this existed.
    ///
    /// Called on the CONSUMER's thread, not the channel tick -- the implementation is
    /// responsible for its own locking. `previz_renderer` already takes a scene mutex for
    /// every mutator, which is what makes that safe here.
    virtual bool input(const input_event&) { return false; }

    virtual std::future<render_output> render(const struct video_format_desc& format_desc) = 0;

    class mutable_frame create_frame(const void* tag, const struct pixel_format_desc& desc) override = 0;
    class mutable_frame create_frame(const void*                     video_stream_tag,
                                     const struct pixel_format_desc& desc,
                                     common::bit_depth               depth) override                               = 0;

// DEFERRED: upstream's CEF shared-texture path (2427604fa) declares
// frame_factory::import_d3d_texture and implements it here. This fork carries its own
// GPU-direct HTML path instead (html_gpu_bridge + ogl/util/dx_interop), so the base
// method is not declared and this override cannot exist. Reconciling the two is
// tracked; see docs/audits/UPSTREAM_SYNC_2026-08-18.md.

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
