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

#include "../StdAfx.h"

#include "stage.h"

#include "layer.h"

#include "../frame/draw_frame.h"

#include <common/diagnostics/graph.h>
#include <common/executor.h>
#include <common/future.h>

#include <core/address/target.h>
#include <core/timeline/resolver.h>
#include <core/timeline/timeline_store.h>
#include <core/timeline/transport.h>
#include <core/frame/frame_transform.h>
#include <core/frame/transform_fields.h>
#include <core/producer/route/route_producer.h>

#include <boost/algorithm/string/predicate.hpp>
#include <boost/range/adaptors.hpp>

#include <algorithm>
#include <chrono>
#include <functional>
#include <future>
#include <iterator>
#include <map>
#include <set>
#include <vector>

namespace caspar { namespace core {

struct stage::impl : public std::enable_shared_from_this<impl>
{
    int                                 channel_index_;
    spl::shared_ptr<diagnostics::graph> graph_;
    monitor::state                      state_;
    std::map<int, layer>                layers_;
    std::map<int, tweened_transform>    tweens_;
    std::set<int>                       routeSources;

    mutable std::mutex      format_desc_mutex_;
    core::video_format_desc format_desc_;

    // ── Keyframe state (all accessed only on the stage executor — no mutex) ──

    // -- Bindings and sources ------------------------------------------------------------
    //
    // Mutated and evaluated ON THE STAGE EXECUTOR. `binding_lock_` guards ONLY the two reads
    // that come from another thread -- `is_bound`, which a write path calls synchronously
    // because it has to refuse rather than be overwritten a frame later, and `feed_sources`,
    // which arrives on the consumer's render thread. Everything else is executor-only, like
    // the timeline state above.
    std::map<std::string, std::shared_ptr<binding::source>> sources_;
    std::vector<binding::binding_def>                       bindings_;
    int                                                     next_binding_id_ = 1;
    mutable std::mutex                                      binding_lock_;

    // -- Structural revision (stage executor only) --
    //
    // A MONOTONIC COUNTER A CLIENT COMPARES TO KNOW THE ADDRESS SPACE GREW OR SHRANK, published
    // as `channel/N/stage/structure_revision`. It exists because nothing else says so: the tree
    // is dynamic -- `PLAY` grows `layer/M/*` and a producer's `params/*`, `SOURCE ADD` grows
    // `source/*`, `BIND` grows `binding/*` -- and `/v1/events` carries VALUE changes only. So a
    // client either re-walks the tree speculatively or shows a stale one.
    //
    // WHY A COUNTER AND NOT OSCQuery's `PATH_CHANGED`, which is the standard answer: that is a
    // per-path WebSocket command, and emitting it needs a hand-maintained hook at every site
    // that creates or destroys a subtree -- the same shape as `apply_transform_colour_values`'s
    // allowlist, silently incomplete the moment someone adds a new dynamic subtree. A missed
    // bump here leaves a client stale; a missed hook there makes the server ASSERT that nothing
    // changed. Stale is recoverable, a false assertion is not, so `EXTENSIONS.PATH_CHANGED`
    // stays `false` and this is the fork's own signal.
    //
    // WHY IT IS DERIVED FROM A FINGERPRINT RATHER THAN BUMPED AT THE MUTATION SITES: the same
    // argument, one level down. `structure_fingerprint()` is computed in the publish pass from
    // what actually exists, so a new dynamic subtree is covered by editing one function that is
    // obviously about this, rather than by remembering a call in a command handler.
    //
    // AND WHY IT IS NOT A HASH OF EVERY PUBLISHED PATH, which would be self-maintaining and
    // wrong: mixer fields are published SPARSELY -- `if (v != defaults[i])` -- so a field
    // returning to its default vanishes from the state, and a client would be told the structure
    // changed every time an operator set opacity back to 1. The fingerprint therefore names the
    // containers that are genuinely dynamic and nothing else.
    std::size_t  structure_hash_     = 0;
    std::int64_t structure_revision_ = 0;

    /// Server-wide, injected by the shell.
    std::shared_ptr<timeline::timeline_store> timelines_;

    /// How a previz screen or camera property is written. Injected by the shell; see stage.h.
    stage::stage_field_writer stage_field_writer_;

    /// The house timecode. Injected by the shell; asked once per tick per CHASING document, so
    /// a server with no LTC and no chase pays nothing.
    stage::timecode_source timecode_source_;

    /// The last value written through each of the two LIVE registries, so a tick that changes
    /// nothing writes nothing.
    ///
    /// For the previz half that is not an optimisation but a correctness question the plan flagged
    /// (F1): every screen mutator also re-applies the mesh transform and calls
    /// `update_projections()`, so writing an unchanged value every tick would recompute the
    /// projection fifty times a second for nothing. For the producer half it keeps a `set_param`
    /// -- which may reach a shader upload -- off the tick when the value has not moved.
    std::map<std::string, monitor::vector_t> last_live_write_;

    /// What a producer parameter or stage field held when the object driving it took over.
    ///
    /// THESE TWO REGISTRIES HAVE NO CONSTANT. A mixer field's operator value lives in `tweens_`
    /// and nothing else touches it, so releasing a driver is free. A producer parameter's value
    /// lives inside the producer and a screen's inside the renderer -- there is nowhere for an
    /// overlay to sit above, so the only way to be lossless is to remember what was there and
    /// put it back. Keyed `<layer>|<path>`.
    std::map<std::string, monitor::vector_t> live_captures_;

    // ── THE TIMELINE, IN THE TICK ───────────────────────────────────────────────────────
    //
    // A DRIVER OVERLAY per layer rather than a write into the layer's tween, which is D3 of the
    // plan and the whole reason this shape differs from the engine it replaces. `KEYFRAMES`
    // wrote by REPLACING `tweens_[layer]` with a zero-duration tween built from the interpolated
    // values, and that made four things impossible at once: an in-flight `MIXER <duration>` was
    // collapsed, an operator's explicit write during an animation was lost rather than
    // remembered, a holding timeline leaked its last value forever after it ended, and `UNBIND`
    // could not give a field back because nothing remembered what it had been.
    //
    // Here the operator's constant lives in `tweens_` and NOTHING ELSE EVER TOUCHES IT. The
    // timeline publishes into `drivers_[layer].timeline`, one `resolve_drivers()` composes the
    // two into `resolved_`, and that is what is pushed, published and read. Ending a driver is
    // then just clearing an overlay -- the constant is still there, untouched, and comes back by
    // construction rather than by being restored.
    struct layer_overlay
    {
        /// Registry path (with its `.N` component suffix, if any) -> value, in registry units.
        std::unordered_map<std::string, double> values;
        /// Step values -- an enum, a boolean, a name -- which cannot be interpolated (D7).
        std::map<std::string, monitor::vector_t> steps;
        /// `timeline:<document>/<object>`, `binding:<id>` or `hold` -- which is what a write
        /// reply's `shadowed_by` reports and what `driver/<path>` publishes.
        std::string owner;

        bool empty() const { return values.empty() && steps.empty(); }
    };
    /// THE OWNERSHIP STACK, one overlay per rank. The operator's constant is NOT here: it is
    /// `tweens_`, and nothing in this file writes it but the operator.
    ///
    /// Named in rank order, highest first, because that is the order `resolve_drivers` applies
    /// them in -- a later write wins, so the strongest owner goes last. Before this the rank was
    /// expressed as the order of two loops in the tick, which is a rule nobody can state without
    /// reading both of them.
    struct layer_drivers
    {
        layer_overlay dominant; //< rank 1 -- HOLD; beats everything
        layer_overlay binding;  //< rank 2 -- a live source
        layer_overlay timeline; //< rank 3 -- an authored document
    };
    std::map<int, layer_drivers> drivers_;

    /// Which instance was active on a layer last tick, and what the layer's parameters were
    /// when it took over.
    ///
    /// The identity is `<document>/<object>#<repeat>` rather than a pointer, because the
    /// resolution is rebuilt on every re-resolve and a pointer into it would dangle. It changing
    /// is what "an object took over" MEANS, and it is the only signal the tick gets: nothing
    /// tells it an object began, so entry is detected by comparing this to the current answer.
    struct layer_entry
    {
        std::string                             active;
        std::string                             document; //< which document's object it is
        std::string                             object;   //< and which object, for `on_end`
        std::unordered_map<std::string, double> captured; //< for `rebase`
    };
    std::map<int, layer_entry> entries_;

    /// The composed transform per DRIVEN layer. Absent means "nothing drives this layer", and
    /// the tween is the answer -- which keeps the cost proportional to what is animated rather
    /// than to the number of layers.
    std::map<int, frame_transform> resolved_;

    /// One playhead per document this channel owns. Keyed by document name because that is what
    /// a client addresses, and because a document may be re-PUT without losing its position.
    std::map<std::string, timeline::transport>                     transports_;
    std::map<std::string, std::vector<timeline::transport_command>> pending_transport_;

    /// ONE CLIP BUILD PER INSTANCE, keyed `<document>/<object>#<repeat>` -- the same identity
    /// string entry detection uses, and for the same reason: the resolution is rebuilt on every
    /// re-resolve, so a pointer into it would dangle.
    ///
    /// A REPEAT GETS ITS OWN BUILD. Sharing one producer across repeats would share its
    /// playhead, so the second pass would start wherever the first one finished -- which is
    /// exactly the class of defect the timeline exists to remove.
    struct media_build
    {
        /// A PLAIN `shared_ptr` INSIDE THE FUTURE TOO, and it has to be. MSVC's
        /// `_Associated_state<T>` DEFAULT-CONSTRUCTS its result type, and `spl::shared_ptr`'s
        /// default constructor calls `spl::make_shared<T>()` -- which cannot compile for an
        /// abstract `frame_producer`. The error arrives as "cannot instantiate abstract class"
        /// from inside `<future>` with nothing in this file named in it.
        std::shared_future<std::shared_ptr<frame_producer>> pending;
        /// A PLAIN `shared_ptr`, not `spl::shared_ptr`: the not-null wrapper cannot be
        /// default-constructed for an abstract type, and "nothing built yet" is exactly the
        /// state this struct spends most of its life in.
        std::shared_ptr<frame_producer>                     producer;
        bool                                                ready  = false;
        bool                                                failed = false;
        bool                                                fired  = false; //< the action has run
        std::string                                         error;
        std::string                                         clip;

        /// HOW LONG THE BUILD TOOK, published rather than gated.
        ///
        /// This is F10 of the timeline plan -- "producer build latency on the API executor for
        /// objects declared within a few frames of play" -- and it was carried as unmeasured
        /// because nothing measured it. A number is only useful if it is on the machine the show
        /// runs on, so the server reports it and the operator compares it with the preroll
        /// window they chose. Gating it here would be gating this box's disk.
        std::chrono::steady_clock::time_point started;
        double                                              build_ms = 0.0;
    };
    std::map<std::string, media_build> media_;
    stage::clip_factory                clip_factory_;
    std::map<std::string, timeline::trigger_log>                   trigger_logs_;

    /// The frame number of the last tick, so `timeline_state` can answer a position without
    /// inventing a frame the channel has not reached.
    std::uint64_t last_frame_number_ = 0;

    /// What `evaluate_timelines` published this tick, held for the state builder below. A member
    /// rather than a return value because the state is assembled on a later pass and the
    /// evaluation must not run twice: a second `position_at` for the same frame would be equal,
    /// but a second `retrigger` would not.
    monitor::state timeline_state_;

    // -- Input routing (stage executor only, no mutex) --
    //
    // `input_capture_` latches the layer a drag belongs to. Once a button goes down on a layer,
    // every move and the release go to THAT layer even after the pointer leaves its rectangle --
    // otherwise a drag that starts on a slider and travels off it stops mid-gesture, which is not
    // what any real drag does. The 2013 interaction API had this right and it is worth not losing.
    //
    // `input_focus_` is where a KEY or TEXT event goes: keys carry no position, so the only honest
    // target is whatever last accepted a pointer event. The 2013 API had no keyboard at all, so
    // there is no precedent to follow for this one.
    int      input_capture_ = -1;
    int      input_focus_   = -1;
    uint32_t input_buttons_ = 0;

    executor   executor_{L"stage " + std::to_wstring(channel_index_)};
    std::mutex lock_;

    // ── Per-layer failure tracking (stage executor only — no mutex) ──
    // Consecutive receive() failures per layer. Used to rate-limit logging and
    // to give up on a layer that fails every tick instead of paying for it
    // forever.
    static constexpr int max_consecutive_layer_failures = 25;
    std::map<int, int>   layer_failures_;

    // ── Per-layer transform publication (stage executor only) ──────────────
    //
    // The keys a layer's transform contributes to the channel state: its
    // non-default mixer fields, and its projection block once it has ever been
    // non-default. Rebuilt only when the transform CHANGES, and written into
    // the state on EVERY tick.
    //
    // That split is the whole point, and it was got wrong first. The
    // projection block this replaces published on change and on a periodic
    // refresh, and skipped the ticks in between -- which is correct for a
    // stream, because an OSC receiver holds the last value it was sent, and
    // WRONG for a snapshot, because a reader of one tick's state sees only what
    // that tick published. Measured: `MIXER 1-10 OPACITY 0.5` reached the
    // control API on the tick it changed and vanished on the next, so the same
    // value read as 0.5 or as "absent, therefore default" depending on which
    // frame the request landed in. A per-frame state has to be COMPLETE; only
    // the work of building it may be skipped.
    //
    // So the cache holds the built keys, and the tick pays only the inserts.
    // For an untouched layer that is nothing at all; for a graded one it is the
    // handful of fields that are actually set.
    //
    // A field at its default is still absent from the wire -- that is what
    // keeps this cheap -- so absence means "at its default", never "unknown".
    // The control API's `/v1/value` falls back to the descriptor default for
    // exactly this reason, and flags such a read `is_default`.
    struct layer_publication
    {
        core::image_transform built_from;
        //: The audio half, compared the same way. `frame_transform` has always carried both and
        //: only the image one was ever published, so a layer's volume was reachable by
        //: `MIXER VOLUME` and by nothing else -- not readable, not bindable, not animatable.
        core::audio_transform built_from_audio;
        bool                  valid = false;
        bool                  projection_ever = false;
        //: relative key ("mixer/opacity", "projection/yaw") and its value
        std::vector<std::pair<std::string, monitor::vector_t>> keys;
    };
    std::map<int, layer_publication> layer_publications_;

    // ── Per-layer receive timing (stage executor only) ──────────────────────
    // Layers are pulled sequentially below, so a producer that *blocks* inside
    // receive() delays every later layer and the channel's whole tick. Note that
    // a producer returning an *empty* frame is already harmless: layer::receive
    // substitutes last_frame(). The hazard is specifically a producer that
    // waits -- a stalled network read, a CEF paint, a lock held by a decoder.
    //
    // Whether any producer in this tree actually does that is an empirical
    // question, and it decides whether a prefetch decorator plus parallel layer
    // fan-out is worth its costs (a frame of added latency per decorated
    // producer, and giving up the deliberate sources-before-routes ordering).
    // So measure it before building it.
    //
    // Published under "receive" on the refresh tick only. Publishing per-tick
    // timings every tick would defeat the change-driven projection publication
    // directly above, which exists precisely to keep this state cheap.
    struct receive_timing
    {
        uint64_t count    = 0;
        uint64_t total_us = 0;
        uint64_t peak_us  = 0;
    };
    std::map<int, receive_timing> receive_timings_;
    uint64_t                      receive_tick_us_      = 0; // current tick, all layers
    uint64_t                      receive_tick_peak_us_ = 0; // worst tick in the window
    uint64_t                      receive_window_us_    = 0; // summed across ticks
    uint64_t                      receive_window_ticks_ = 0;

    // Last computed figures. `state_` is rebuilt from scratch every tick, so the
    // values have to be re-emitted every tick even though they are only
    // recomputed once a second -- otherwise they are visible in one tick out of
    // 25 and any reader sees them only by luck.
    struct receive_published
    {
        bool         valid = false;
        uint64_t     tick_avg_us  = 0;
        uint64_t     tick_peak_us = 0;
        double       budget_percent      = 0.0;
        double       peak_budget_percent = 0.0;
        int32_t      layers              = 0;
        int          slowest_layer       = -1;
        uint64_t     slowest_avg_us      = 0;
        uint64_t     slowest_peak_us     = 0;
        std::wstring slowest_producer;
    };
    receive_published receive_published_;

  private:
    /// Records a failure for `index` and logs it (first failure, then every
    /// 25th) so a permanently-broken producer cannot flood the log.
    /// Returns true when the layer has failed too many times in a row and
    /// should be dropped.
    bool note_layer_failure(int index)
    {
        auto count = ++layer_failures_[index];

        if (count == 1 || count % max_consecutive_layer_failures == 0) {
            CASPAR_LOG_CURRENT_EXCEPTION();
            CASPAR_LOG(error) << L"stage[" << channel_index_ << L"] layer " << index << L" failed to produce a frame ("
                              << count << L" consecutive). Other layers are unaffected.";
        }

        if (count >= max_consecutive_layer_failures) {
            CASPAR_LOG(error) << L"stage[" << channel_index_ << L"] removing layer " << index << L" after " << count
                              << L" consecutive failures.";
            return true;
        }

        return false;
    }

    void orderSourceLayers(std::vector<std::pair<int, bool>>&        layerVec,
                           const std::map<int, std::pair<int, int>>& routed_layers,
                           int                                       l,
                           int                                       depth)
    {
        if (0 == depth)
            routeSources.clear();

        if (std::find_if(layerVec.begin(), layerVec.end(), [l](std::pair<int, bool> p) { return p.first == l; }) !=
            layerVec.end()) {
            return;
        }

        auto routeIt = routed_layers.find(l);
        if (routed_layers.end() == routeIt) {
            layerVec.push_back(std::make_pair(l, true));
            return;
        }

        std::pair<int, int> routeSrc(routeIt->second);
        if (channel_index_ != routeSrc.first) {
            layerVec.push_back(std::make_pair(l, true));
            return;
        }

        // check for circular route setup - skip recursion if found
        routeSources.emplace(routeSrc.second);
        bool layerOK = true;
        if (routeSources.find(l) == routeSources.end()) {
            orderSourceLayers(layerVec, routed_layers, routeSrc.second, ++depth);
        } else {
            layerOK = false;
        }

        if (std::find_if(layerVec.begin(), layerVec.end(), [l](std::pair<int, bool> p) { return p.first == l; }) ==
            layerVec.end()) {
            layerVec.push_back(std::make_pair(l, layerOK));
        }
    }

    layer& get_layer(int index)
    {
        auto it = layers_.find(index);
        if (it == std::end(layers_)) {
            it = layers_.emplace(index, layer(video_format_desc())).first;
        }
        return it->second;
    }

  public:
    impl(int channel_index, spl::shared_ptr<diagnostics::graph> graph, const core::video_format_desc& format_desc)
        : channel_index_(channel_index)
        , graph_(std::move(graph))
        , format_desc_(format_desc)
    {
    }

    const stage_frames operator()(uint64_t                                     frame_number,
                                  std::vector<int>&                            fetch_background,
                                  std::function<void(int, const layer_frame&)> routesCb)
    {
        return executor_.invoke([=, this] {
            std::map<int, layer_frame> frames;
            stage_frames               result = {};

            result.format_desc = video_format_desc();
            result.nb_samples =
                result.format_desc.audio_cadence[frame_number % result.format_desc.audio_cadence.size()];

            auto is_interlaced = format_desc_.field_count == 2;
            auto field1        = is_interlaced ? video_field::a : video_field::progressive;

            try {
                for (auto& t : tweens_)
                    t.second.tick(1);

                // -- Bindings, before the timeline and before any layer is pulled -----------
                //
                // Before the timeline deliberately. Both write the same parameters, and a
                // parameter cannot honestly be driven by two things at once -- so the order is
                // chosen rather than left to whichever runs first. The TIMELINE wins: it is an
                // explicit authored value for this frame and a binding is a standing rule, which
                // is rank 3 over rank 2 of the ownership stack in `docs/features/timeline.md`.
                //
                // In this build that ordering is what implements the rank: a binding writes the
                // CONSTANT through `tweened_transform::patch`, and the timeline's overlay is
                // composed on top of it a few lines below. When bindings move to their own
                // overlay the rank moves into `resolve_drivers` and this comment goes with it.
                //
                // Before the layers are pulled, because the value has to be in the transform
                // when compositing reads it. Applied after, it would land one frame late -- and
                // for an audio-reactive parameter, a frame of lag is the artefact the whole
                // feature exists to avoid.
                {
                    double fps_for_dt = 25.0;
                    if (format_desc_.framerate.numerator() > 0 && format_desc_.framerate.denominator() > 0)
                        fps_for_dt = static_cast<double>(format_desc_.framerate.numerator()) /
                                     format_desc_.framerate.denominator();
                    // The NOMINAL tick, not a measured wall-clock delta. A source driven by
                    // real elapsed time would run faster during a dropped frame and produce a
                    // waveform that is not reproducible -- and every gate on this feature reads
                    // the frame clock, per the harness's own rule.
                    evaluate_bindings(1000.0 / (fps_for_dt > 0.0 ? fps_for_dt : 25.0));
                }

                // -- The TIMELINE, then the composition of the two --------------------------
                //
                // After the bindings and before the layers are pulled. After, because a
                // timeline is an explicit authored value for THIS frame and a binding is a
                // standing rule -- and because in this build a binding still writes the
                // constant directly, so the timeline overlay composed on top of it is what
                // makes the timeline outrank it (D3's ranks 3 over 4). Before the layers,
                // because the value has to be in the transform when compositing reads it: one
                // frame late is the artefact the whole feature exists to avoid.
                //
                // `timeline_state_` is filled here and consumed by the publication below, so
                // the evaluation happens once per tick rather than once per reader.
                last_frame_number_ = frame_number;
                timeline_state_    = monitor::state{};
                evaluate_timelines(frame_number, timeline_state_);
                resolve_drivers();
                // The two LIVE registries, which are not part of the frame transform: a
                // producer's own parameters and the previz stage. Same overlays, same rank, a
                // different destination -- and write-on-change, because a screen mutator
                // recomputes the projection.
                apply_live_targets();

                // build a map of layers that are sourced from route producers
                std::map<int, std::pair<int, int>> routed_layers;
                for (auto& p : layers_) {
                    auto producer = std::move(p.second.foreground());
                    if (0 == producer->name().compare(L"route")) {
                        try {
                            auto rc       = spl::dynamic_pointer_cast<core::route_control>(producer);
                            auto srcChan  = rc->get_source_channel();
                            auto srcLayer = rc->get_source_layer();
                            routed_layers.emplace(p.first, std::make_pair(srcChan, srcLayer));
                            rc->set_cross_channel(channel_index_ != srcChan);
                        } catch (std::bad_cast) {
                            CASPAR_LOG(error) << "Failed to cast route producer";
                        }
                    }
                }

                // sort layer order so that sources get pulled before routes
                std::vector<std::pair<int, bool>> layerVec;
                for (auto& p : layers_)
                    orderSourceLayers(layerVec, routed_layers, p.first, 0);

                // when running interlaced, both fields are be pulled at once.
                // This will risk some stutter for freshly created producers, but it lets us tick at 25hz and avoids
                // amcp changes starting on the second field

                receive_tick_us_ = 0;

                for (auto& l : layerVec) {
                    auto p = layers_.find(l.first);
                    if (p == layers_.end())
                        continue;

                    auto& layer = p->second;
                    auto& tween = tweens_[p->first];
                    // FETCHED ONCE per layer per tick, and it is the RESOLVED transform where
                    // anything drives the layer. Both fields of an interlaced frame get the same
                    // one: they are one tick, and sampling the tween twice would have advanced
                    // it between them.
                    const auto effective = effective_transform(p->first);
                    (void)tween;

                    auto has_background_route =
                        std::find(fetch_background.begin(), fetch_background.end(), p->first) != fetch_background.end();

                    // ── Per-layer fault isolation ──────────────────────────
                    // A throwing producer must not take the rest of the
                    // channel down with it. Previously a single exception here
                    // propagated to the outer handler, which cleared EVERY
                    // layer on the channel — a broadcast-hostile failure mode.
                    // Now the offending layer alone degrades (to black, or to
                    // its last good frame via layer::receive) and is dropped
                    // entirely after too many consecutive failures.
                    const auto recv_start = std::chrono::steady_clock::now();

                    layer_frame res = {};
                    try {
                        if (l.second) {
                            res.foreground1_raw = layer.receive(field1, result.nb_samples);
                            res.foreground1     = draw_frame::push(res.foreground1_raw, effective);
                            res.foreground1.transform().image_transform.enable_geometry_modifiers = true;
                        }

                        res.has_background = layer.has_background();
                        if (has_background_route)
                            res.background1 = layer.receive_background(field1, result.nb_samples);

                        if (is_interlaced) {
                            res.is_interlaced = true;
                            if (l.second) {
                                res.foreground2_raw = layer.receive(video_field::b, result.nb_samples);
                                res.foreground2     = draw_frame::push(res.foreground2_raw, effective);
                                res.foreground2.transform().image_transform.enable_geometry_modifiers = true;
                            }
                            if (has_background_route)
                                res.background2 = layer.receive_background(video_field::b, result.nb_samples);
                        }

                        layer_failures_.erase(p->first);
                    } catch (...) {
                        res = layer_frame{};
                        if (is_interlaced)
                            res.is_interlaced = true;

                        if (note_layer_failure(p->first)) {
                            // Too many consecutive failures — drop just this
                            // layer so it stops costing a frame every tick.
                            layers_.erase(p->first);
                            layer_failures_.erase(l.first);
                            receive_timings_.erase(p->first);
                            continue;
                        }
                    }

                    {
                        const auto recv_us = static_cast<uint64_t>(
                            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() -
                                                                                 recv_start)
                                .count());
                        auto& t = receive_timings_[p->first];
                        t.count++;
                        t.total_us += recv_us;
                        t.peak_us = std::max(t.peak_us, recv_us);
                        receive_tick_us_ += recv_us;
                    }

                    frames[p->first] = res;

                    // push received foreground frame to any configured route producer
                    try {
                        routesCb(p->first, res);
                    } catch (...) {
                        if (note_layer_failure(p->first)) {
                            layers_.erase(p->first);
                            layer_failures_.erase(l.first);
                        }
                    }
                }

                for (auto& p : frames) {
                    result.frames.push_back(p.second.foreground1);
                    if (is_interlaced)
                        result.frames2.push_back(p.second.foreground2);
                }

                {
                    // push stage_frames to support any channel routes that have been set
                    layer_frame chan_lf   = {};
                    chan_lf.is_interlaced = is_interlaced;
                    chan_lf.foreground1   = wrap_layer_frames_for_route(result.frames);
                    if (is_interlaced)
                        chan_lf.foreground2 = wrap_layer_frames_for_route(result.frames2);

                    routesCb(-1, chan_lf);
                }

                // The receive-timing block below publishes on this tick only; publishing
                // per-tick timings every tick would make the state churn for values nobody
                // reads at that rate.
                //
                // It used to serve a second purpose -- a periodic republication of the
                // per-layer transform keys, so an OSC subscriber joining mid-show
                // converged on values that were not changing. That is gone because it is
                // no longer needed: those keys are now written on EVERY tick from a cache,
                // so every snapshot is complete and there is nothing to converge to.
                const auto refresh_ticks =
                    static_cast<uint64_t>(std::max(1, static_cast<int>(result.format_desc.hz)));
                const bool osc_refresh_due = (frame_number % refresh_ticks) == 0;

                receive_window_us_ += receive_tick_us_;
                receive_window_ticks_++;
                receive_tick_peak_us_ = std::max(receive_tick_peak_us_, receive_tick_us_);

                monitor::state state;

                // Receive timing, on the refresh tick only. The whole-tick figure
                // against the frame budget is the number that matters: if pulling
                // every layer costs a few percent of a frame, sequential receive is
                // not hurting anyone and a prefetch decorator would only add
                // latency.
                if (osc_refresh_due && receive_window_ticks_ > 0) {
                    // Drop timings for layers that no longer exist. CLEAR removes
                    // from layers_ but said nothing about this map, so stale entries
                    // survived and were counted -- reporting eight layers on a
                    // three-layer channel, and potentially naming a departed
                    // producer as the slowest.
                    for (auto it = receive_timings_.begin(); it != receive_timings_.end();)
                        it = layers_.find(it->first) == layers_.end() ? receive_timings_.erase(it) : std::next(it);

                    const double period_us = result.format_desc.hz > 0.0 ? 1000000.0 / result.format_desc.hz : 0.0;
                    const auto   avg_us    = receive_window_us_ / receive_window_ticks_;

                    auto& pub          = receive_published_;
                    pub.valid          = true;
                    pub.tick_avg_us    = avg_us;
                    pub.tick_peak_us   = receive_tick_peak_us_;
                    pub.budget_percent = period_us > 0.0 ? (static_cast<double>(avg_us) * 100.0) / period_us : 0.0;
                    pub.peak_budget_percent =
                        period_us > 0.0 ? (static_cast<double>(receive_tick_peak_us_) * 100.0) / period_us : 0.0;
                    pub.layers = static_cast<int32_t>(receive_timings_.size());

                    // Name the worst layer, so a blocking producer can be found
                    // rather than merely suspected.
                    pub.slowest_layer   = -1;
                    pub.slowest_peak_us = 0;
                    pub.slowest_avg_us  = 0;
                    pub.slowest_producer.clear();
                    for (const auto& [index, t] : receive_timings_) {
                        if (t.peak_us > pub.slowest_peak_us) {
                            pub.slowest_peak_us = t.peak_us;
                            pub.slowest_layer   = index;
                            pub.slowest_avg_us  = t.count ? t.total_us / t.count : 0;
                        }
                    }
                    if (pub.slowest_layer >= 0) {
                        const auto it = layers_.find(pub.slowest_layer);
                        if (it != layers_.end())
                            pub.slowest_producer = it->second.foreground()->name();
                    }

                    receive_window_us_    = 0;
                    receive_window_ticks_ = 0;
                    receive_tick_peak_us_ = 0;
                    for (auto& [index, t] : receive_timings_)
                        t = receive_timing{};
                }

                if (receive_published_.valid) {
                    const auto& pub                      = receive_published_;
                    state["receive"]["tick_avg_us"]       = static_cast<int64_t>(pub.tick_avg_us);
                    state["receive"]["tick_peak_us"]      = static_cast<int64_t>(pub.tick_peak_us);
                    state["receive"]["budget_percent"]    = pub.budget_percent;
                    state["receive"]["peak_budget_percent"] = pub.peak_budget_percent;
                    state["receive"]["layers"]            = pub.layers;
                    if (pub.slowest_layer >= 0) {
                        state["receive"]["slowest_layer"]   = pub.slowest_layer;
                        state["receive"]["slowest_avg_us"]  = static_cast<int64_t>(pub.slowest_avg_us);
                        state["receive"]["slowest_peak_us"] = static_cast<int64_t>(pub.slowest_peak_us);
                        if (!pub.slowest_producer.empty())
                            state["receive"]["slowest_producer"] = pub.slowest_producer;
                    }
                }

                // THE PLAYHEADS, one sub-tree per document this channel owns. Merged rather
                // than rebuilt: `evaluate_timelines` ran at the top of this tick and is the only
                // thing allowed to advance a transport or fire a trigger, so the state builder
                // reads what it produced instead of asking again.
                state.merge(timeline_state_);

                for (auto& p : layers_) {
                    state["layer"][p.first] = p.second.state();
                    publish_layer_transform(state, p.first);
                }

                // -- Structural revision -------------------------------------------------
                //
                // Four inputs, and each one is a container that appears or disappears rather
                // than a value that moves:
                //
                //   * WHICH LAYERS EXIST      `PLAY`/`STOP` grow and prune `layer/M/*`
                //   * EACH LAYER'S PRODUCER   a swap changes the producer's own sub-tree, and
                //                             carries its `params/*` set with it. The producer
                //                             NAME is a sound proxy for that set: an ISF or OFX
                //                             parameter list is fixed per shader or plugin, so
                //                             params cannot change under a stable producer
                //   * BINDING IDS             `BIND`/`UNBIND` grow and prune `binding/{id}/*`
                //   * SOURCE NAMES            `SOURCE ADD`/`REMOVE` grow and prune `source/*`
                //
                // Deliberately NOT included: any field VALUE, any timing figure, the frame
                // number. A revision that moved every tick would be a revision a client had to
                // ignore, which is the same as not having one.
                {
                    std::string fp;
                    fp.reserve(256);
                    for (const auto& p : layers_) {
                        fp += std::to_string(p.first);
                        fp += ':';
                        // `foreground()` is never null -- an empty layer holds an empty
                        // producer whose name is stable -- so this needs no guard, and a
                        // producer swap changes the string.
                        //
                        // Hashed rather than appended because `name()` is a `std::wstring` and
                        // this fingerprint never needs to be READ: it only needs to differ when
                        // the producer differs. That avoids pulling `common/utf.h` in here and
                        // avoids an encoding question in a function that has no business having
                        // one.
                        fp += std::to_string(std::hash<std::wstring>{}(p.second.foreground()->name()));
                        fp += ';';
                    }
                    fp += '|';
                    for (const auto& b : bindings_) {
                        fp += std::to_string(b.id);
                        fp += ';';
                    }
                    fp += '|';
                    {
                        std::lock_guard<std::mutex> lock(binding_lock_);
                        for (const auto& kv : sources_) {
                            fp += kv.first;
                            fp += ';';
                        }
                    }
                    fp += '|';
                    // THE TIMELINE STORE'S REVISION, and it is what makes a `PUT
                    // /v1/timeline/{name}` observable from outside this process. A document
                    // appearing, changing or being deleted moves this counter, which moves the
                    // fingerprint, which bumps `structure_revision` -- and re-walking the tree
                    // on that is exactly what a client already does.
                    //
                    // The counter rather than the documents themselves: the question is "has
                    // anything changed since I looked", and hashing a large document every tick
                    // on every channel to answer it would be the wrong trade. One mutex
                    // acquisition per tick per channel is the whole cost.
                    if (timelines_)
                        fp += std::to_string(timelines_->revision());

                    const auto h = std::hash<std::string>{}(fp);
                    if (h != structure_hash_) {
                        structure_hash_ = h;
                        // FIRST PASS BUMPS TO 1 rather than staying at 0, deliberately: a
                        // client that reads 0 cannot tell "nothing has been published yet"
                        // from "the structure is empty". Every published revision is >= 1.
                        ++structure_revision_;
                    }
                    state["structure_revision"] = structure_revision_;
                }

                // -- Bindings and sources, published every tick ----------------------------
                //
                // Published UNCONDITIONALLY when they exist, including `broken` and `value`,
                // because there is no descriptor default for these to fall back to: a binding
                // is not a field with a known default, and an absent key would be read as "no
                // such binding" rather than "at its default". That is the rule the previz stage
                // publication learned the hard way.
                //
                // `value` is the last value WRITTEN, after the transform and the lag -- not the
                // source's raw sample. It is what a client needs to draw the parameter moving,
                // and it is the number a battery can fit a waveform to.
                for (const auto& b : bindings_) {
                    auto node = state["binding"][std::to_string(b.id)];
                    node["layer"]   = b.layer;
                    node["target"]  = b.target;
                    node["component"] = static_cast<int>(b.component);
                    node["source"]  = b.source_name + "/" + b.source_channel;
                    node["min"]     = b.tf.out_lo;
                    node["max"]     = b.tf.out_hi;
                    node["in_min"]  = b.tf.in_lo;
                    node["in_max"]  = b.tf.in_hi;
                    node["gain"]    = b.tf.gain;
                    node["lag"]     = b.tf.lag_ms;
                    node["curve"]   = std::string(binding::curve_name(b.tf.curve));
                    node["value"]   = b.held;
                    node["broken"]  = b.broken;
                }

                {
                    std::lock_guard<std::mutex> lock(binding_lock_);
                    for (const auto& kv : sources_) {
                        auto node = state["source"][kv.first];
                        node["kind"] = kv.second->kind();
                        // WHERE IT LISTENS AND WHETHER IT IS LISTENING, which `kind` cannot say
                        // and which a client cannot otherwise find out. `describe()` has existed
                        // on every source since they were introduced and was reachable only
                        // through `SOURCE LIST` over AMCP:
                        //
                        //   osc udp/7411 /casparcg/source/knobs/... (listening, 42 accepted, 0 dropped)
                        //   midi 1 "nanoKONTROL2" (listening, 1180 messages)
                        //   lfo sine 0.5 Hz
                        //
                        // So an OSC source's PORT was undiscoverable over the control API: an
                        // operator had to be told out of band where to send. Note this is
                        // deliberately NOT OSCQuery's `HOST_INFO.OSC_PORT` -- that assumes ONE
                        // OSC server for the whole address space, and there are N receivers here,
                        // one per `SOURCE ADD`, per channel, on arbitrary ports. A single
                        // `OSC_PORT` would have to name one of them arbitrarily.
                        node["describe"] = kv.second->describe();
                        // Every channel's CURRENT value, so a client can show a source moving
                        // before anything is bound to it -- which is how an operator checks a
                        // MIDI knob or an audio band is arriving at all.
                        for (const auto& ch : kv.second->channels()) {
                            double v = 0.0;
                            if (kv.second->value(ch, v))
                                node[ch] = v;
                        }
                    }
                }

                state_ = std::move(state);
            } catch (...) {
                // Per-layer faults are handled inside the loop above; anything
                // reaching here is a stage-wide fault (tween/timeline/route
                // bookkeeping). Do NOT clear layers_ — losing every layer on
                // the channel is far worse than dropping one frame.
                CASPAR_LOG_CURRENT_EXCEPTION();
            }

            return result;
        });
    }

    /// Write this layer's transform keys into the tick's state.
    ///
    /// Two jobs with very different costs, and keeping them apart is the point:
    ///
    ///   * REBUILD -- work out which fields differ from their declared default and format
    ///     their keys. Done only when the transform changed, because it walks ~180
    ///     descriptors and allocates a value vector per non-default field;
    ///   * WRITE -- put the cached keys into this tick's state. Done EVERY tick, because a
    ///     per-frame snapshot has to be a complete description of the frame. Cost is one
    ///     flat_map insert per key that is actually set: nothing for an untouched layer,
    ///     a handful for a graded one.
    ///
    /// The projection block used to skip the write as well as the rebuild, publishing on
    /// change and on a periodic refresh only. For OSC that is invisible -- a receiver holds
    /// the last value it was sent -- and for anything reading one tick's state it is a
    /// value that blinks: measured on the control API, `MIXER 1-10 OPACITY 0.5` was
    /// readable on the tick it changed and gone on the next, so the same request returned
    /// 0.5 or "absent, therefore default" depending on which frame it landed in.
    ///
    /// A field AT its default is still absent from the wire, which is what keeps the write
    /// cheap. Absence therefore means "at its default", never "unknown" -- a consumer that
    /// reads it the other way shows a stale value forever after a reset.
    // ── The timeline's tick ────────────────────────────────────────────────────────────

    /// `"1-10"` or `"10"` -> the layer index on THIS channel, or nothing.
    ///
    /// A channel-qualified layer that names a different channel is not an error here and not
    /// this channel's business: cross-channel evaluation is commit 17, and until then a document
    /// addressing another channel is simply not evaluated by this one. Silently, because it is
    /// the document's declared intent rather than a mistake.
    /// `"1-10"` or `"10"` -> this channel's layer index, or nothing if it is not this channel's.
    ///
    /// A BARE layer number means "the document's own channel", so it resolves here only on the
    /// home channel -- `home` is false when this stage is a guest. Without that a document
    /// declaring channel 1 and writing `"10"` would drive layer 10 on every channel it also
    /// addresses, which is the opposite of what an author writing an unqualified layer means.
    std::optional<int> layer_index_for(const std::string& spec, bool home = true) const
    {
        const auto dash = spec.find('-');
        if (dash == std::string::npos) {
            if (!home)
                return std::nullopt;
            if (spec.empty() || spec.find_first_not_of("0123456789") != std::string::npos)
                return std::nullopt;
            return std::atoi(spec.c_str());
        }
        const auto ch = spec.substr(0, dash);
        const auto ly = spec.substr(dash + 1);
        if (ch.empty() || ly.empty() || ch.find_first_not_of("0123456789") != std::string::npos ||
            ly.find_first_not_of("0123456789") != std::string::npos)
            return std::nullopt;
        if (std::atoi(ch.c_str()) != channel_index_)
            return std::nullopt;
        return std::atoi(ly.c_str());
    }

    static const timeline::timeline_object* find_object(const std::vector<timeline::timeline_object>& objs,
                                                        const std::string&                            id)
    {
        for (const auto& o : objs) {
            if (o.id == id)
                return &o;
            if (!o.children.empty())
                if (const auto* c = find_object(o.children, id))
                    return c;
        }
        return nullptr;
    }

    /// Does this object ask for its final value to be kept?
    bool object_on_end_commit(const std::string& doc_name, const std::string& object_id)
    {
        if (!timelines_ || doc_name.empty())
            return false;
        const auto entry = timelines_->get(doc_name);
        if (!entry)
            return false;
        const auto* obj = find_object(entry->document.objects, object_id);
        return obj && obj->on_end == timeline::on_end_t::commit;
    }

    /// Write an object's curve value AT ITS END into the layer's constant.
    ///
    /// At its end rather than at the position the tick has reached: an object that ended two
    /// frames ago should commit the value it finished on, not the value it would have had if it
    /// had kept running. Through `tweened_transform::patch`, so an in-flight `MIXER <duration>`
    /// on another field of the same layer keeps interpolating -- which is the whole reason
    /// `patch` exists.
    void commit_final_values(int layer, const std::string& doc_name, const std::string& object_id)
    {
        if (!timelines_)
            return;
        const auto entry = timelines_->get(doc_name);
        if (!entry)
            return;
        const auto* obj = find_object(entry->document.objects, object_id);
        if (!obj || obj->curves.empty())
            return;

        const auto at_end = obj->curves.interpolate(obj->curves.duration(), timeline::kind_of);
        if (at_end.empty())
            return;

        tweens_[layer].patch([&](frame_transform& t) {
            for (const auto& pv : at_end) {
                const auto target = address::parse(pv.first);
                if (target.kind == address::target_kind::image) {
                    const auto* f = fields::find(target.field);
                    if (!f || !f->set || !f->get)
                        continue;
                    auto v = f->get(t.image_transform);
                    if (v.size() != f->arity)
                        continue;
                    v[std::min<std::size_t>(target.component, v.size() - 1)] = pv.second;
                    if (f->set(t.image_transform, v))
                        fields::apply_enables(t.image_transform, *f);
                } else if (target.kind == address::target_kind::audio) {
                    const auto* a = fields::find_audio_field(target.field);
                    if (!a || !a->set || !a->get)
                        continue;
                    auto v = a->get(t.audio_transform);
                    if (v.size() != a->arity)
                        continue;
                    v[std::min<std::size_t>(target.component, v.size() - 1)] = pv.second;
                    a->set(t.audio_transform, v);
                }
            }
        });
    }

    /// SEEK COMPILATION: make the constants say what they would say if the show had been played
    /// up to `to` instead of jumped to it.
    ///
    /// This is the whole compilation, and it is short for a reason worth stating: a `release`
    /// object leaves NO state behind, so only `commit` objects have anything to replay. Replayed
    /// in END order, because two objects committing the same path must land in the order they
    /// would have.
    ///
    /// Without it, a seek forwards past a committed cue leaves the parameter wherever it was --
    /// so the same position reached by playing and by seeking would look different, and an
    /// operator checking a cue by seeking to it would be looking at the wrong picture.
    void compile_seek(const std::string& doc_name, timeline::flicks to)
    {
        if (!timelines_)
            return;
        const auto entry = timelines_->get(doc_name);
        if (!entry || !entry->resolved.ok())
            return;

        std::vector<const timeline::instance*> done;
        for (const auto& in : entry->resolved.instances)
            if (in.end && *in.end <= to && !in.layer.empty())
                done.push_back(&in);

        std::stable_sort(done.begin(), done.end(),
                         [](const timeline::instance* a, const timeline::instance* b) {
                             return *a->end < *b->end;
                         });

        for (const auto* in : done) {
            const auto layer = layer_index_for(in->layer);
            if (!layer)
                continue;
            if (object_on_end_commit(doc_name, in->object_id))
                commit_final_values(*layer, doc_name, in->object_id);
        }
    }

    /// Drive this channel's layers from one document at one position.
    ///
    /// SHARED BY THE HOME CHANNEL AND EVERY GUEST, and that sharing is the whole of cross-channel
    /// rather than an economy. A guest that ran its own copy of this would be a second
    /// implementation of entry detection, rebase capture, step-keyframe precedence and the curve
    /// evaluation -- four rules that have to agree across channels for a show to look like one
    /// show, and that would agree only by being re-read every time either was edited.
    ///
    /// `home` is passed down to `layer_index_for`, which is the ONLY difference between the two
    /// callers: a bare layer number means the document's own channel.
    void drive_layers(const std::string&               name,
                      const timeline::stored_timeline& entry,
                      timeline::flicks                 pos,
                      monitor::state&                  state,
                      std::set<int>&                   owned,
                      bool                             home)
    {
        auto ts = state["timeline"][name];

        for (const auto& kv : entry.resolved.by_layer) {
            const auto layer = layer_index_for(kv.first, home);
            if (!layer)
                continue;
            const auto* inst = entry.resolved.active_on(kv.first, pos);
            if (!inst)
                continue; //< the sweep in `evaluate_timelines` handles the release
            const auto* obj = find_object(entry.document.objects, inst->object_id);
            if (!obj)
                continue;

            auto& ov = drivers_[*layer].timeline;
            ov.owner = "timeline:" + name + "/" + inst->object_id;

            const auto local = inst->local_at(pos);

            // ENTRY DETECTION, and it has to be done here because nothing tells the tick
            // that an object began. The identity is `<document>/<object>#<repeat>`: a
            // string rather than a pointer, because the resolution is rebuilt on every
            // re-resolve and a pointer into it would dangle.
            const auto identity =
                name + "/" + inst->object_id + "#" + std::to_string(inst->repeat_index);
            auto&      ent = entries_[*layer];
            const bool entering = ent.active != identity;
            // THE LAYER ACTION, EVERY TICK THE INSTANCE IS ACTIVE, not only on entry. It
            // self-guards with `fired`, so it runs exactly once per instance -- but WHICH tick
            // that is cannot be decided in advance: a build takes as long as it takes, and a
            // clip that is not ready on the cue frame has to get its action on the tick it
            // becomes ready.
            //
            // The first version called this inside the `entering` branch below, with a comment
            // claiming a late build would still fire on a later tick. That comment was false and
            // `timeline-media` proved it: with `preroll_frames: 0` the clip never reached the
            // layer at all, because entry had already passed by the time the producer existed.
            //
            // BEFORE the capture below, and that ordering matters: a `play` that swaps the
            // producer makes the previous producer's parameter values irrelevant, so capturing
            // them after the swap is the only way to capture the right layer's state.
            fire_layer_action(*layer, identity, *obj);

            if (entering) {
                ent.active   = identity;
                ent.document = name;
                ent.object   = inst->object_id;
                ent.captured.clear();

                // THE LIVE REGISTRIES ARE CAPTURED ON ENTRY WHATEVER `rebase` SAYS, and for
                // a different reason from the rebase below. A producer parameter's value
                // lives inside the producer and a screen's inside the renderer -- there is
                // no constant for an overlay to sit above, so the only way releasing them
                // can be lossless is to remember what was there. `rebase` is about where a
                // ramp STARTS; this is about what is given back when it ends.
                for (const auto& path : obj->curves.paths()) {
                    const auto t = address::parse(path);
                    if (t.kind != address::target_kind::producer_param)
                        continue;
                    const auto key = std::to_string(*layer) + "|" + path;
                    if (live_captures_.count(key))
                        continue;
                    auto v = read_live_target(*layer, t);
                    if (!v.empty())
                        live_captures_[key] = std::move(v);
                }

                // WHAT THE PARAMETERS WERE WHEN THIS OBJECT TOOK OVER. Captured for
                // `rebase` -- and captured from the EFFECTIVE transform, so an object
                // taking over from another driver starts from what was on air rather than
                // from the operator's constant underneath it.
                if (obj->rebase) {
                    const auto eff = effective_transform(*layer);
                    for (const auto& path : obj->curves.paths()) {
                        const auto t = address::parse(path);
                        monitor::vector_t v;
                        if (const auto* f = fields::find(t.field)) {
                            if (f->get)
                                v = f->get(eff.image_transform);
                        } else if (const auto* a = fields::find_audio_field(t.field)) {
                            if (a->get)
                                v = a->get(eff.audio_transform);
                        }
                        if (v.empty())
                            continue;
                        const auto  idx = std::min<std::size_t>(t.component, v.size() - 1);
                        if (const auto* d = boost::get<double>(&v[idx]))
                            ent.captured[path] = *d;
                        else if (const auto* i32 = boost::get<int32_t>(&v[idx]))
                            ent.captured[path] = *i32;
                    }
                }
            }

            // Step values, in three passes and in this order:
            //
            //   1. the object's `content` -- what it sets on entry and holds throughout;
            //   2. its `keyframes` -- STEP values that change at a time, the ones that
            //      cannot be interpolated (an enum, a boolean, a name, a file). The LAST
            //      one whose start has passed wins, which is what "step" means: it changes
            //      AT the key and holds until the next;
            //   3. the curves, so a path in both has the CURVE win (D7).
            for (const auto& c : obj->content)
                ov.steps[c.first] = c.second;

            for (const auto& kf : obj->keyframes) {
                // Literal times only, and the PUT refuses anything else -- a step keyframe
                // referencing another object would need the resolver, and the resolver works
                // on objects rather than on keys inside them.
                for (const auto& spec : {kf.enable}) {
                    if (!spec.start || spec.start->k != timeline::time_expr::kind::literal)
                        continue;
                    if (local < spec.start->literal)
                        continue;
                    if (spec.end && spec.end->k == timeline::time_expr::kind::literal &&
                        local >= spec.end->literal)
                        continue;
                    for (const auto& c : kf.content)
                        ov.steps[c.first] = c.second;
                }
            }

            for (const auto& pv : obj->curves.interpolate(
                     local, timeline::kind_of, obj->rebase ? &ent.captured : nullptr))
                ov.values[pv.first] = pv.second;

            owned.insert(*layer);
            ts["active"][kv.first] = inst->object_id;
        }
    }

    /// BUILD AHEAD OF THE CUE, and publish whether each build is ready.
    ///
    /// Walks the instances of every object carrying a `clip` and starts a build for any whose
    /// start is within `preroll_frames` of now. Off the executor, so the decode that a build
    /// performs never lands in the frame path -- which is the whole reason the factory is a
    /// bridge rather than a call into the registry from here.
    ///
    /// WHY IT LOOKS AHEAD RATHER THAN BUILDING ON ENTRY. A build takes as long as it takes: a
    /// local file is milliseconds and a network source is not. Building on entry would put that
    /// latency between the cue and the picture, with nothing the operator could do about it. A
    /// preroll window is the operator SAYING how much warning the source needs, and it defaults
    /// to 25 frames because one second is the answer for a local file.
    ///
    /// A BUILD IS NEVER RETRIED inside one instance. A clip that cannot be built will not
    /// build a second time either, and retrying once per tick would hammer a missing path fifty
    /// times a second and fill the log. The fault is published and the instance stays failed;
    /// a re-PUT is how an operator fixes it, which also re-resolves and gives new keys.
    void preroll_media(const std::string&               name,
                       const timeline::stored_timeline& entry,
                       timeline::flicks                 pos,
                       monitor::state&                  state,
                       bool                             home)
    {
        if (!clip_factory_)
            return;

        const auto per_frame = timeline::flicks_per_frame(format_desc_.framerate);
        auto       ts        = state["timeline"][name];

        for (const auto& kv : entry.resolved.by_object) {
            const auto* obj = find_object(entry.document.objects, kv.first);
            if (!obj || !obj->clip || obj->layer.empty())
                continue;
            if (!layer_index_for(obj->layer, home))
                continue;

            const auto window = per_frame * std::max(0, obj->preroll_frames);

            for (const auto idx : kv.second) {
                if (idx >= entry.resolved.instances.size())
                    continue;
                const auto& in = entry.resolved.instances[idx];
                const auto key = name + "/" + in.object_id + "#" + std::to_string(in.repeat_index);
                auto&      mb  = media_[key];

                if (mb.clip.empty())
                    mb.clip = u8(*obj->clip);

                // NOT YET IN THE WINDOW, and nothing is started. Checked against the instance's
                // own start rather than the object's, so a repeating object prerolls each pass.
                //
                // THE UPPER BOUND IS THE INSTANCE'S END, not its start, and the first version
                // used the start -- which made `preroll_frames: 0` a feature that never fired.
                // With a zero window the only tick that could start a build was the one where
                // `pos == start` exactly; the build then took a frame, and by the next tick
                // `pos > start` closed the door on it forever. `timeline-media` caught it on its
                // first run. Ending the window at the instance's end also means a document
                // seeked into the middle of a cue still builds its clip, which is what an
                // operator scrubbing a show expects.
                if (!mb.ready && !mb.failed && !mb.pending.valid()) {
                    if (in.start - pos > window || (in.end && pos >= *in.end))
                        continue;
                    const auto clip    = *obj->clip;
                    auto       factory = clip_factory_;
                    mb.started         = std::chrono::steady_clock::now();
                    mb.pending         = std::async(std::launch::async,
                                            [factory, clip]() -> std::shared_ptr<frame_producer> {
                                                return factory(clip);
                                            })
                                     .share();
                    CASPAR_LOG(debug) << L"[timeline] prerolling " << clip << L" for " << u16(key);
                }

                if (mb.pending.valid() && !mb.ready && !mb.failed &&
                    mb.pending.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                    // COLLECTED IN THE TICK WITHOUT BLOCKING: `wait_for(0)` first, so a build
                    // that is still running costs one atomic read and the tick moves on. A bare
                    // `get()` here would make a slow source stall the channel, which is the
                    // failure this whole pass exists to avoid.
                    mb.build_ms = std::chrono::duration<double, std::milli>(
                                      std::chrono::steady_clock::now() - mb.started)
                                      .count();
                    try {
                        mb.producer = mb.pending.get();
                        mb.ready    = true;
                    } catch (const std::exception& e) {
                        mb.failed = true;
                        mb.error  = e.what();
                        CASPAR_LOG(error) << L"[timeline] " << u16(key) << L": could not build '"
                                          << u16(mb.clip) << L"': " << u16(mb.error);
                    } catch (...) {
                        mb.failed = true;
                        mb.error  = "the producer factory threw";
                    }
                }

                auto ms     = ts["media"][in.object_id];
                ms["clip"]  = mb.clip;
                ms["ready"] = mb.ready;
                if (mb.build_ms > 0.0)
                    ms["build_ms"] = mb.build_ms;
                if (mb.failed)
                    ms["error"] = mb.error;
                if (mb.fired)
                    ms["on_air"] = true;
            }
        }
    }

    /// The object's `layer_action` at the moment its instance begins.
    ///
    /// Called from the entry branch of `drive_layers`, so it runs exactly once per instance --
    /// the identity string is what makes that true, and it is the same one `preroll_media` keys
    /// its builds by.
    void fire_layer_action(int layer, const std::string& key, const timeline::timeline_object& obj)
    {
        auto& mb = media_[key];
        if (mb.fired)
            return;

        // A CLIP THAT IS NOT READY DOES NOT BLOCK THE TICK. The instance starts with nothing on
        // the layer and the fault is already published -- late is a visible mistake the operator
        // can see and fix, and a stalled channel is not. `fired` is NOT set, so a build that
        // lands a few frames later still gets its action on the next tick.
        if (obj.clip) {
            if (mb.failed) {
                mb.fired = true; //< it will never be ready; stop looking
                return;
            }
            if (!mb.ready)
                return;
            get_layer(layer).load(spl::make_shared_ptr(mb.producer), /* preview */ false,
                                  obj.action == timeline::layer_action::play);
        }

        switch (obj.action) {
            case timeline::layer_action::play:
                // `load(auto_play)` above already started it when there was a clip. Without one
                // this is a PLAY of whatever the operator had loaded, which is the form a
                // document uses to start a clip somebody else cued.
                if (!obj.clip)
                    get_layer(layer).play();
                break;
            case timeline::layer_action::load:
                break; //< `load(auto_play=false)` is the whole action
            case timeline::layer_action::pause:
                get_layer(layer).pause();
                break;
            case timeline::layer_action::resume:
                get_layer(layer).resume();
                break;
            case timeline::layer_action::stop:
                get_layer(layer).stop();
                break;
            case timeline::layer_action::clear:
                layers_.erase(layer);
                break;
            case timeline::layer_action::none:
                break;
        }
        mb.fired = true;
    }

    /// Does this document address any layer on THIS channel?
    ///
    /// Asked before a guest does any work, so a four-channel server does not walk every
    /// document's resolution on every channel once per frame. A document that names no layer here
    /// is not this channel's business at all.
    bool document_addresses_this_channel(const timeline::stored_timeline& entry) const
    {
        for (const auto& kv : entry.resolved.by_layer)
            if (layer_index_for(kv.first, false))
                return true;
        return false;
    }

    /// A document whose HOME is another channel, evaluated here at the home channel's position.
    ///
    /// Everything transport-shaped is absent by design and not by omission: no command queue (a
    /// `TIMELINE 2 PLAY show` for a document declaring channel 1 is refused by
    /// `timeline_command`, so a guest has nothing queued), no seek compilation (the constants a
    /// seek replays are the HOME channel's -- a guest's own constants are compiled by its own
    /// release path), and no re-resolve on a GO (the home channel does it and swaps the store's
    /// pointer, so a guest picks up the new resolution on its next tick, within one frame).
    ///
    /// What it publishes is deliberately a SUBSET: `follows`, `state` and `active`, and NOT
    /// `position` or `rate`. One playhead per show means one place publishing it -- two channels
    /// each publishing a `position` for one document would be two numbers a client has to choose
    /// between, and they would differ by up to a frame for reasons that are not a fault.
    void evaluate_guest_timeline(const std::string&               name,
                                 const timeline::stored_timeline& entry,
                                 monitor::state&                  state,
                                 std::set<int>&                   owned)
    {
        auto ts = state["timeline"][name];
        ts["follows"] = entry.document.channel;

        if (!entry.resolved.ok()) {
            ts["ok"]     = false;
            ts["faults"] = static_cast<std::int64_t>(entry.resolved.errors.size());
            return;
        }
        ts["ok"] = true;

        const auto ph = timelines_->playhead_of(name);
        if (!ph) {
            // NOTHING HAS PUBLISHED A PLAYHEAD YET, which means the home channel has not ticked
            // since this document was stored. Reported as `stopped` and NOT as a state of its
            // own, and the distinction was measured away rather than reasoned away:
            // `timeline-crosschannel` asserted a separate `unstarted` on its first run and got
            // `stopped`, because the home channel publishes every tick including while stopped.
            // So "never played" is a transient of at most one frame that no client can rely on
            // seeing, and a state name for it would be a state name that is always wrong.
            //
            // Either way the guest drives NOTHING -- not position zero, which would put every
            // guest layer under a driver the moment a document was stored and take it from the
            // operator who had set it by hand.
            ts["state"] = std::string(timeline::to_string(timeline::transport_state::stopped));
            return;
        }

        ts["state"]    = std::string(timeline::to_string(ph->state));
        ts["revision"] = entry.document.revision;
        if (ph->chasing)
            ts["chasing"] = true;

        if (ph->state == timeline::transport_state::stopped)
            return; //< the release sweep at the end of `evaluate_timelines` gives the layers back

        preroll_media(name, entry, ph->position, state, /* home */ false);
        drive_layers(name, entry, ph->position, state, owned, /* home */ false);
    }

    /// Every document this channel owns, advanced one tick and evaluated into the overlays.
    void evaluate_timelines(std::uint64_t frame_number, monitor::state& state)
    {
        // Only THIS rank is cleared. The binding overlay is rebuilt by `evaluate_bindings` and
        // the dominant one persists until `RELEASE` -- clearing all three here would make a HOLD
        // last exactly one tick.
        for (auto& kv : drivers_)
            kv.second.timeline = layer_overlay{};
        if (!timelines_)
            return;

        const auto per_frame = timeline::flicks_per_frame(format_desc_.framerate);

        // WHICH LAYERS STILL HAVE AN OWNER after this pass. Gathered across every document
        // rather than per document, because the restore below has to fire when a document is
        // STOPPED, deleted, invalid, or simply has no object active -- and only three of those
        // four reach the per-layer loop at all. Doing it per document meant `STOP` restored
        // nothing, which `timeline-targets` caught on its first run.
        std::set<int> owned;

        for (const auto& name : timelines_->names()) {
            auto entry = timelines_->get(name);
            if (!entry)
                continue;

            // HOME OR GUEST. The document declares ONE channel and that channel owns its
            // transport -- it takes the commands, advances off its own frame counter, and chases
            // house timecode. Every other channel it addresses is a guest: it reads the home
            // channel's published playhead and drives only its own layers.
            //
            // Why not give every channel its own transport for the same document? Because then
            // two channels would each have a position and a run state for one show, and any
            // difference between them -- a chase correction, a dropped frame, a command that
            // reached one executor first -- would be a split-brain a client could see and
            // nothing could reconcile. There is one playhead per show by construction.
            const bool home = entry->document.channel == channel_index_;
            if (!home && !document_addresses_this_channel(*entry))
                continue;

            if (!home) {
                evaluate_guest_timeline(name, *entry, state, owned);
                continue;
            }

            // AFTER the guest branch, because `operator[]` would otherwise give every guest
            // channel an empty transport for a document it does not own -- and `timeline_state`
            // would then answer "stopped at zero" for a running show instead of refusing.
            auto& tr = transports_[name];

            // A SEEK IN THIS TICK'S COMMANDS needs its compilation run, and the target has to
            // be read BEFORE `apply_all` consumes them. A stop counts too: it rewinds to zero,
            // which is a seek to zero as far as the constants are concerned.
            std::optional<timeline::flicks> seek_to;
            for (const auto& c : pending_transport_[name]) {
                if (c.v == timeline::transport_command::verb::seek)
                    seek_to = c.at;
                else if (c.v == timeline::transport_command::verb::stop)
                    seek_to = 0;
            }

            // The pending commands for this tick, in the SAFE order rather than arrival order:
            // stop outranks pause outranks play. See `transport::apply_all`.
            tr.apply_all(pending_transport_[name], frame_number, per_frame);

            if (seek_to)
                compile_seek(name, *seek_to);

            // A GO fired: the document has to be RE-RESOLVED, because an object whose end waits
            // on a trigger has no end until it fires, and everything downstream of it has no
            // start. Re-resolving the whole document rather than patching it -- they are small,
            // this happens on an operator action rather than per frame, and a partial
            // re-resolution is where the collision semantics would start to diverge from
            // `/resolved`, which is the one thing that must not happen.
            if (!tr.fired().empty()) {
                auto& log = trigger_logs_[name];
                for (const auto& f : tr.fired())
                    log.fired[f.first].push_back(f.second);
                tr.clear_fired();
                if (auto re = timelines_->retrigger(name, log))
                    entry = re;
            }

            auto ts = state["timeline"][name];
            ts["state"]    = std::string(timeline::to_string(tr.state()));
            ts["revision"] = entry->document.revision;
            ts["ok"]       = entry->resolved.ok();

            if (!entry->resolved.ok()) {
                // An invalid document is INERT rather than dangerous: it is stored so the client
                // can show the author the faults, and evaluating a half-authored show would be
                // the wrong reading of "stored".
                ts["faults"] = static_cast<std::int64_t>(entry->resolved.errors.size());
                continue;
            }

            // THE POSITION, and chase is a CORRECTION on top of the frame counter rather than a
            // different clock. `chase_position` is the only per-tick call that may mutate the
            // transport, which is why it is called here and exactly once.
            timeline::flicks pos;
            if (tr.chase().enabled) {
                std::optional<timeline::flicks> house;
                if (timecode_source_) {
                    const auto fps = static_cast<int>(
                        std::lround(boost::rational_cast<double>(format_desc_.framerate)));
                    if (const auto f = timecode_source_(fps))
                        house = timeline::from_frames(static_cast<std::int64_t>(*f),
                                                      format_desc_.framerate);
                }
                pos = tr.chase_position(frame_number, per_frame, house);
                ts["chasing"]     = tr.chasing();
                ts["freewheeled"] = static_cast<std::int64_t>(tr.freewheeled());
            } else {
                pos = tr.position_at(frame_number, per_frame);
            }
            ts["position"] = timeline::to_seconds(pos);
            ts["rate"]     = boost::rational_cast<double>(tr.rate());
            ts["home"]     = channel_index_;

            // WHERE THE GUESTS READ FROM. Published every tick including while stopped, because
            // a guest has to be able to tell "stopped" from "never played" -- the first releases
            // its layers and the second drives nothing at all, and they are not the same state.
            timelines_->publish_playhead(name, timeline::playhead{pos, tr.state(), frame_number,
                                                                  entry->document.revision,
                                                                  tr.chasing()});

            if (tr.state() == timeline::transport_state::stopped) {
                // STOPPED RELEASES. A stopped document owns nothing, so every layer it was
                // driving falls back to its constant on this very tick -- which is the
                // difference between stop and pause (D4), and the reason `pause` re-anchors
                // instead of clearing.
                continue;
            }

            preroll_media(name, *entry, pos, state, /* home */ true);
            drive_layers(name, *entry, pos, state, owned, /* home */ true);
        }

        // ---- the release sweep -------------------------------------------------------
        //
        // Every layer that had an owner and does not now. This is the only place that knows a
        // driver ENDED, and it has to be here rather than inside the per-document loop because
        // three of the four ways a driver can end -- the document stopped, deleted, or turned
        // invalid -- never reach that loop.
        //
        // A mixer field needs nothing: its release is the overlay being cleared, which happens
        // at the top of this function. The two LIVE registries do: a producer parameter's value
        // lives inside the producer, so it has to be written back.
        // A BUILD FOR A DOCUMENT THAT IS GONE. Dropped here rather than left, because `media_`
        // is keyed by `<document>/<object>#<repeat>` and a deleted-then-re-PUT document reuses
        // those keys: a stale `fired` would make the new document's first cue a no-op, and a
        // stale producer would put the OLD clip on air. Cheap -- one pass over a map with one
        // entry per media instance, and only the names still loaded survive.
        if (timelines_ && !media_.empty()) {
            const auto live = timelines_->names();
            for (auto it = media_.begin(); it != media_.end();) {
                const auto slash = it->first.find('/');
                const auto doc   = slash == std::string::npos ? it->first : it->first.substr(0, slash);
                if (std::find(live.begin(), live.end(), doc) == live.end())
                    it = media_.erase(it);
                else
                    ++it;
            }
        }

        std::vector<int> ended;
        for (const auto& e : entries_)
            if (!owned.count(e.first))
                ended.push_back(e.first);

        for (const auto layer : ended) {
            const auto& ent = entries_[layer];

            // `on_end: commit` -- BAKE the object's final value into the constant instead of
            // releasing it. `release` (the default) is lossless because the constant was never
            // touched; `commit` is the opposite intent: "this is the new normal", which is what
            // an operator means by a cue that moves a grade and leaves it there.
            //
            // Committed BEFORE the live-target restore below, and the two do not overlap: a
            // committed mixer field writes the constant, and a producer parameter or previz
            // field has no constant to write -- so `commit` on one of those is a no-op and is
            // documented as one rather than silently doing something else.
            if (auto committed = object_on_end_commit(ent.document, ent.object))
                commit_final_values(layer, ent.document, ent.object);

            const auto               prefix = std::to_string(layer) + "|";
            std::vector<std::string> paths;
            for (const auto& c : live_captures_)
                if (c.first.compare(0, prefix.size(), prefix) == 0)
                    paths.push_back(c.first.substr(prefix.size()));
            for (const auto& path : paths)
                restore_live_target(layer, path);
            entries_.erase(layer);
        }
    }

    /// Compose the constant and the overlays into `resolved_`, once per driven layer per tick.
    ///
    /// The rank is the ownership stack of D3, and only two of its five levels exist yet: the
    /// timeline (3) over the constant (4). Bindings still write the constant directly through
    /// `tweened_transform::patch`, and move up to their own overlay in commit 9 -- until then a
    /// binding and a timeline on the SAME field would fight, which nothing does yet and which
    /// the plan's own sequence puts at commit 9.
    void resolve_drivers()
    {
        resolved_.clear();
        if (drivers_.empty())
            return;

        for (const auto& kv : drivers_) {
            const auto& d = kv.second;
            if (d.timeline.empty() && d.binding.empty() && d.dominant.empty())
                continue;

            const auto tw = tweens_.find(kv.first);
            if (tw == tweens_.end())
                continue;

            auto t = tw->second.fetch();

            const auto write = [&t](const std::string& path, const monitor::vector_t& value) {
                const auto target = address::parse(path);
                if (!target.meta)
                    return; // a producer parameter or a stage field: they have their own writers
                if (target.kind == address::target_kind::image) {
                    const auto* f = fields::find(target.field);
                    if (!f || !f->set)
                        return;
                    if (value.size() == f->arity) {
                        if (f->set(t.image_transform, value))
                            fields::apply_enables(t.image_transform, *f);
                        return;
                    }
                    // ONE COMPONENT of a wider field: read-modify-write, because the setter's
                    // arity check refuses a one-element vector and inventing the other three
                    // would be a guess.
                    auto v = f->get(t.image_transform);
                    if (v.size() != f->arity || value.size() != 1)
                        return;
                    v[std::min<std::size_t>(target.component, v.size() - 1)] = value.front();
                    if (f->set(t.image_transform, v))
                        fields::apply_enables(t.image_transform, *f);
                } else if (target.kind == address::target_kind::audio) {
                    const auto* a = fields::find_audio_field(target.field);
                    if (!a || !a->set)
                        return;
                    if (value.size() == a->arity) {
                        a->set(t.audio_transform, value);
                        return;
                    }
                    auto v = a->get(t.audio_transform);
                    if (v.size() != a->arity || value.size() != 1)
                        return;
                    v[std::min<std::size_t>(target.component, v.size() - 1)] = value.front();
                    a->set(t.audio_transform, v);
                }
            };

            // THE RANK, and it is these three lines. Weakest first, so the strongest owner's
            // write is the one that survives: timeline (3), then binding (2), then dominant (1).
            //
            // A binding over a timeline rather than the other way round, because a binding is a
            // LIVE input -- an audio level, a tracker, a fader -- and a document is authored
            // ahead of time. Every product surveyed gives the live thing the parameter, and an
            // operator whose fader stops working because a show is running would not accept the
            // opposite. `HOLD` is above both because it is the operator saying "this one is
            // mine now", which nothing else may override.
            const auto apply = [&](const layer_overlay& ov) {
                for (const auto& sv : ov.steps)
                    write(sv.first, sv.second);
                for (const auto& pv : ov.values)
                    write(pv.first, monitor::vector_t{pv.second});
            };
            apply(d.timeline);
            apply(d.binding);
            apply(d.dominant);

            resolved_[kv.first] = std::move(t);
        }
    }

    /// Write the overlay entries that are NOT part of the frame transform: producer parameters
    /// and previz screen/camera properties.
    ///
    /// Separate from `resolve_drivers` because those compose into a `frame_transform` and these
    /// do not -- a producer parameter is set through the producer's own setter and a screen
    /// property through the renderer's mutator. Same overlays, same rank, a different
    /// destination.
    ///
    /// WRITE-ON-CHANGE, which for the previz half answers F1 of the plan conservatively rather
    /// than measuring it later: every screen mutator re-applies the mesh transform and calls
    /// `update_projections()`, so writing an unchanged value every tick would recompute the
    /// projection fifty times a second for nothing. The comparison is against what THIS code
    /// last wrote, not against what the renderer holds -- reading the renderer back per tick is
    /// the round trip the bridge exists to avoid.
    void apply_live_targets()
    {
        for (const auto& kv : drivers_) {
            const auto layer = kv.first;
            const auto& d    = kv.second;

            // The rank again, weakest first, so the strongest owner's value is the one that
            // survives into `wanted`.
            std::map<std::string, monitor::vector_t> wanted;
            for (const auto* ov : {&d.timeline, &d.binding, &d.dominant}) {
                for (const auto& sv : ov->steps)
                    wanted[sv.first] = sv.second;
                for (const auto& pv : ov->values)
                    wanted[pv.first] = monitor::vector_t{pv.second};
            }

            for (const auto& w : wanted) {
                const auto t = address::parse(w.first);
                if (t.kind != address::target_kind::producer_param &&
                    t.kind != address::target_kind::screen && t.kind != address::target_kind::camera &&
                    t.kind != address::target_kind::view_camera)
                    continue;

                const auto key = std::to_string(layer) + "|" + w.first;
                const auto it  = last_live_write_.find(key);
                if (it != last_live_write_.end() && it->second == w.second)
                    continue;

                if (write_live_target(layer, t, w.second))
                    last_live_write_[key] = w.second;
            }
        }
    }

    /// One write into a live registry. Returns whether it landed.
    bool write_live_target(int layer, const address::target& t, const monitor::vector_t& value)
    {
        if (t.kind == address::target_kind::producer_param) {
            const auto it = layers_.find(layer);
            if (it == layers_.end())
                return false;
            auto producer = it->second.foreground();
            if (producer == frame_producer::empty())
                return false;
            for (const auto& p : producer->parameters()) {
                if (p.name != t.field || !p.set)
                    continue;
                // ONE COMPONENT of a wider parameter: read-modify-write, for the reason
                // `apply_binding` gives -- the setter's arity check refuses a one-element vector
                // and inventing the rest would be a guess.
                if (p.arity > 1 && value.size() == 1) {
                    if (!p.get)
                        return false;
                    auto v = p.get();
                    if (v.size() != p.arity)
                        return false;
                    v[std::min<std::size_t>(t.component, v.size() - 1)] = value.front();
                    return p.set(v);
                }
                return p.set(value);
            }
            return false;
        }

        if (!stage_field_writer_)
            return false;
        const std::string object = t.kind == address::target_kind::screen
                                       ? ("screen/" + t.object)
                                       : (t.kind == address::target_kind::view_camera ? "view_camera" : "camera");
        // A wider stage field driven one component at a time needs the whole vector, and the
        // renderer is the only thing that knows the other components -- so a partial write is
        // refused rather than guessed. `previz/screen/wall/position` with all three values in a
        // document works; `position.0` alone does not, and that is stated in timeline.md.
        if (t.meta && value.size() != t.meta->arity)
            return false;
        return stage_field_writer_(object, t.field, value);
    }

    /// Give a live registry back what it held before a driver took it.
    void restore_live_target(int layer, const std::string& path)
    {
        const auto key = std::to_string(layer) + "|" + path;
        const auto cap = live_captures_.find(key);
        if (cap == live_captures_.end())
            return;
        const auto t = address::parse(path);
        write_live_target(layer, t, cap->second);
        last_live_write_[key] = cap->second;
        live_captures_.erase(cap);
    }

    /// Read a live registry's current value, for the capture.
    monitor::vector_t read_live_target(int layer, const address::target& t)
    {
        if (t.kind == address::target_kind::producer_param) {
            const auto it = layers_.find(layer);
            if (it == layers_.end())
                return {};
            auto producer = it->second.foreground();
            if (producer == frame_producer::empty())
                return {};
            for (const auto& p : producer->parameters())
                if (p.name == t.field && p.get)
                    return p.get();
            return {};
        }
        // No reader for a stage field from here: the renderer is behind a one-way bridge, and
        // adding a read to it would be the synchronous round trip per tick that the bridge's
        // whole shape avoids. A previz field is restored to its DOCUMENT-declared value at
        // release instead, which is what `timeline-previz` asserts and what timeline.md says.
        return {};
    }

    /// What a layer's frame is actually drawn with: the resolved transform if anything drives it,
    /// the operator's constant if not.
    frame_transform effective_transform(int layer)
    {
        const auto r = resolved_.find(layer);
        if (r != resolved_.end())
            return r->second;
        return tweens_[layer].fetch();
    }

    void publish_layer_transform(monitor::state& state, int layer)
    {
        const auto tw = tweens_.find(layer);
        if (tw == tweens_.end())
            return;

        // THE EFFECTIVE TRANSFORM, not the constant. `mixer/*` is what a client reads to draw a
        // slider, and publishing the constant while the picture showed the driver's value would
        // make the API disagree with the screen -- which is the one thing a control surface
        // cannot recover from. The constant is still reachable, under `constant/*` below, and a
        // client that wants to show "you set 0.5, the timeline is at 0.2" has both.
        const auto  ft  = effective_transform(layer);
        const auto& tf  = ft.image_transform;
        const auto& at  = ft.audio_transform;
        auto&       pub = layer_publications_[layer];

        // Both halves in the change test. `audio_transform` has no `operator==`, and adding one
        // would be a public API change for two members -- so they are compared here, where the
        // only question is "has anything this publication reads moved".
        const bool audio_same =
            pub.built_from_audio.volume == at.volume && pub.built_from_audio.immediate_volume == at.immediate_volume;
        if (!pub.valid || !(pub.built_from == tf) || !audio_same) {
            pub.built_from       = tf;
            pub.built_from_audio = at;
            pub.valid            = true;
            rebuild_layer_publication(pub, tf, at);
        }

        if (pub.keys.empty())
            return;

        auto ls = state["layer"][layer];
        for (const auto& kv : pub.keys)
            ls[kv.first] = kv.second;

        // ── WHO OWNS WHAT, published rather than inferable ─────────────────────────────
        //
        // `driver/<path>` names what is writing a field, and `constant/<path>` carries the
        // operator's own value where it DIFFERS from what is on air. Published unconditionally
        // for a driven layer -- there is no descriptor default for "who owns this", so an absent
        // key means "nobody", which is exactly the fact a client needs.
        //
        // Without this a client can see that a value moved and not what moved it, which is how
        // an operator ends up dragging a slider that snaps back every frame with no explanation.
        const auto dit = drivers_.find(layer);
        if (dit == drivers_.end())
            return;
        const auto& d = dit->second;
        if (d.timeline.empty() && d.binding.empty() && d.dominant.empty())
            return;

        const auto constant = tweens_[layer].fetch();

        // WHO OWNS A PATH, and WHO ELSE WANTED IT. `driver/<path>` is the effective owner --
        // the strongest rank writing that path -- and `stack/<path>` is every rank that wanted
        // it, strongest first, as a comma-separated list.
        //
        // The stack is not decoration. Without it, `UNBIND` looks like it will hand the field
        // back to the operator when in fact a document underneath will take it, and a client
        // cannot warn anybody. With it, "binding:3,timeline:show/lt1" says exactly what happens
        // next. It is the same question `field_bound` used to answer with a refusal, answered
        // with information instead.
        const auto rank_of = [&](const std::string& path) {
            std::string effective;
            std::string stack;
            const layer_overlay* ranked[3] = {&d.dominant, &d.binding, &d.timeline};
            for (const auto* ov : ranked) {
                if (ov->owner.empty())
                    continue;
                if (ov->values.find(path) == ov->values.end() && ov->steps.find(path) == ov->steps.end())
                    continue;
                if (effective.empty())
                    effective = ov->owner;
                if (!stack.empty())
                    stack += ",";
                stack += ov->owner;
            }
            return std::make_pair(effective, stack);
        };

        const auto report = [&](const std::string& path) {
            const auto who = rank_of(path);
            if (who.first.empty())
                return;
            ls["driver"][path] = who.first;
            ls["stack"][path]  = who.second;

            const auto target = address::parse(path);
            if (!target.meta)
                return;
            monitor::vector_t was, now;
            if (target.kind == address::target_kind::image) {
                const auto* f = fields::find(target.field);
                if (!f || !f->get)
                    return;
                was = f->get(constant.image_transform);
                now = f->get(tf);
            } else if (target.kind == address::target_kind::audio) {
                const auto* a = fields::find_audio_field(target.field);
                if (!a || !a->get)
                    return;
                was = a->get(constant.audio_transform);
                now = a->get(at);
            } else {
                return;
            }
            if (was != now)
                ls["constant"][target.field] = was;
        };

        // Every path any rank touches, once. A path two ranks write is reported once with both
        // of them in its stack, rather than twice with the weaker one overwriting the answer.
        std::set<std::string> paths;
        for (const auto* ov : {&d.dominant, &d.binding, &d.timeline}) {
            for (const auto& pv : ov->values)
                paths.insert(pv.first);
            for (const auto& sv : ov->steps)
                paths.insert(sv.first);
        }
        for (const auto& path : paths)
            report(path);
    }

    /// The expensive half: which keys this transform contributes, and their values.
    void rebuild_layer_publication(layer_publication&           pub,
                                   const core::image_transform& tf,
                                   const core::audio_transform& at)
    {
        pub.keys.clear();

        // The audio rows, sparse on the same rule as the image ones: a field at its declared
        // default is absent, and `/v1/value` falls back to the descriptor for it.
        {
            static const std::vector<monitor::vector_t> adefaults = [] {
                std::vector<monitor::vector_t> d;
                for (const auto& f : core::fields::audio_fields())
                    d.push_back(f.defaults());
                return d;
            }();
            const auto& afs = core::fields::audio_fields();
            for (std::size_t i = 0; i < afs.size(); ++i) {
                auto v = afs[i].get(at);
                if (v != adefaults[i])
                    pub.keys.emplace_back(std::string("mixer/") + afs[i].path, std::move(v));
            }
        }

        // `defaults()` builds a fresh vector per call, so reading all ~180 of them per
        // rebuild would be most of the cost of the feature. They never change.
        static const std::vector<monitor::vector_t> defaults = [] {
            std::vector<monitor::vector_t> d;
            d.reserve(core::fields::all().size());
            for (const auto& f : core::fields::all())
                d.push_back(f.defaults());
            return d;
        }();

        const auto& fs = core::fields::all();
        for (std::size_t i = 0; i < fs.size(); ++i) {
            auto v = fs[i].get(tf);
            if (v != defaults[i])
                pub.keys.emplace_back(std::string("mixer/") + fs[i].path, std::move(v));
        }

        // The projection block keeps its own rule rather than following the sparse one
        // above: once a layer has ever had a non-default projection it publishes the WHOLE
        // block, defaults included. That is what the existing OSC consumers of
        // `layer/N/projection/*` were written against -- a projection is read as a
        // coherent set of angles and offsets, and half of one is worse than none.
        static const core::projection projection_defaults{};
        const auto&                   pr = tf.projection;
        if (!pub.projection_ever) {
            if (pr == projection_defaults)
                return;
            pub.projection_ever = true;
        }

        const auto add = [&pub](const char* key, monitor::vector_t v) {
            pub.keys.emplace_back(std::string("projection/") + key, std::move(v));
        };
        add("enable", {pr.enable});
        add("yaw", {pr.yaw});
        add("pitch", {pr.pitch});
        add("roll", {pr.roll});
        add("fov", {pr.fov});
        add("offset_x", {pr.offset_x});
        add("offset_y", {pr.offset_y});
        add("frustum_h", {pr.frustum_h});
        add("frustum_v", {pr.frustum_v});
        add("lens_k1", {pr.lens_k1});
        add("lens_k2", {pr.lens_k2});
        add("lens_k3", {pr.lens_k3});
        add("lens_p1", {pr.lens_p1});
        add("lens_p2", {pr.lens_p2});
        add("source_lens", {static_cast<int32_t>(pr.source_lens)});
        add("curve_enable", {pr.curve_enable});
        add("curve_auto", {pr.curve_auto});
        add("icvfx_auto", {pr.icvfx_auto});
        add("curve_type", {static_cast<int32_t>(pr.curve_type)});
        add("screen_arc", {pr.screen_arc});
        add("screen_arc_v", {pr.screen_arc_v});
        add("eye_distance", {pr.eye_distance});
        add("edge_blend",
            {pr.edge_blend_left,
             pr.edge_blend_right,
             pr.edge_blend_top,
             pr.edge_blend_bottom,
             pr.edge_blend_gamma});
        add("icvfx_enable", {pr.icvfx_enable});
        add("inner_fov", {pr.inner_fov});
        add("icvfx_feather", {pr.icvfx_feather});
        add("icvfx_outer_dim", {pr.icvfx_outer_dim});
        add("icvfx_inner_dim", {pr.icvfx_inner_dim});
        add("icvfx_inner_gain", {pr.icvfx_inner_gain_r, pr.icvfx_inner_gain_g, pr.icvfx_inner_gain_b});
        add("icvfx_outer_gain", {pr.icvfx_outer_gain_r, pr.icvfx_outer_gain_g, pr.icvfx_outer_gain_b});
    }

    core::draw_frame wrap_layer_frames_for_route(std::vector<core::draw_frame> frames)
    {
        // Note: this must not mutate the vector used for the layer
        for (auto& frame : frames) {
            // Tell the compositor that these are layers, matching what normal rendering does
            frame.transform().image_transform.layer_depth = 1;
        }
        return core::draw_frame(frames);
    }

    std::future<void>
    apply_transforms(const std::vector<std::tuple<int, stage::transform_func_t, unsigned int, tweener>>& transforms)
    {
        return executor_.begin_invoke([=, this] {
            for (auto& transform : transforms) {
                auto& tween = tweens_[std::get<0>(transform)];
                auto  src   = tween.fetch();
                auto  dst   = std::get<1>(transform)(tween.dest());
                tweens_[std::get<0>(transform)] =
                    tweened_transform(src, dst, std::get<2>(transform), std::get<3>(transform));
            }
        });
    }

    std::future<void> apply_transform(int                            index,
                                      const stage::transform_func_t& transform,
                                      unsigned int                   mix_duration,
                                      const tweener&                 tween)
    {
        return executor_.begin_invoke([=, this] {
            auto src       = tweens_[index].fetch();
            auto dst       = transform(src);
            tweens_[index] = tweened_transform(src, dst, mix_duration, tween);
        });
    }

    std::future<void> clear_transforms(int index)
    {
        return executor_.begin_invoke([=, this] { tweens_.erase(index); });
    }

    std::future<void> clear_transforms()
    {
        return executor_.begin_invoke([=, this] { tweens_.clear(); });
    }

    // ---------------------------------------------------------------------------------
    // Bindings
    // ---------------------------------------------------------------------------------

    /// Advance every source and apply every binding. Called at the TOP of the tick.
    ///
    /// At the top rather than the bottom, because the value has to be in the transform before
    /// the layers are pulled and composited -- applied afterwards it would land one frame late,
    /// which for an audio-reactive parameter is exactly the artefact the feature exists to
    /// avoid.
    void evaluate_bindings(double dt_ms)
    {
        // Rebuilt from scratch every tick, like the timeline's. A binding whose source went
        // BROKEN writes nothing this tick, and clearing first is what makes that fall through to
        // whatever is underneath rather than freeze at the last good value -- which is D3's
        // "a broken binding falls through", and which a persistent overlay would silently break.
        for (auto& kv : drivers_)
            kv.second.binding = layer_overlay{};

        {
            std::lock_guard<std::mutex> lock(binding_lock_);
            for (auto& kv : sources_)
                kv.second->tick(dt_ms);
        }

        if (bindings_.empty())
            return;

        for (auto& b : bindings_) {
            double raw = 0.0;
            {
                std::lock_guard<std::mutex> lock(binding_lock_);
                auto it = sources_.find(b.source_name);
                // BROKEN rather than silently 0. A source removed under a live binding, or a
                // channel name that no longer exists, must be visible in the published state --
                // a binding that runs forever writing zero is the 202-and-no-change failure
                // this registry exists to prevent, and it would be indistinguishable from a
                // source legitimately sitting at zero.
                if (it == sources_.end() || !it->second->value(b.source_channel, raw)) {
                    b.broken = true;
                    continue;
                }
            }
            b.broken = false;

            const double target = binding::map_value(b.tf, raw);

            // The lag is primed on the FIRST evaluation rather than smoothed from zero. Without
            // this, every binding with a lag ramps up from 0 on its first frame, which for an
            // opacity binding is a visible flash and for a position binding is a jump.
            if (!b.primed) {
                b.held   = target;
                b.primed = true;
            } else {
                b.held = binding::apply_lag(target, b.held, b.tf.lag_ms, dt_ms);
            }

            apply_binding(b, b.held);
        }
    }

    /// Write one binding's value to its target.
    void apply_binding(const binding::binding_def& b, double value)
    {
        // A producer parameter. The target names the parameter; the producer's own setter takes
        // it, so a bound ISF input goes through exactly the path `PUT` and `CALL ISF SET` use.
        // NOT `address::parse` here, deliberately, and this is a per-tick path: `b.target` is
        // ALREADY the bare field name -- the `.N` suffix was split off at BIND time -- so a
        // parse would allocate two or three strings per binding per tick to recover a prefix
        // test. `add_binding` does the full resolution once, where it is free.
        if (b.target.rfind("producer/", 0) == 0) {
            const auto name = b.target.substr(9);

            auto it = layers_.find(b.layer);
            if (it == layers_.end())
                return;
            auto producer = it->second.foreground();
            if (producer == frame_producer::empty())
                return;

            for (const auto& p : producer->parameters()) {
                if (p.name != name || !p.set || !p.get)
                    continue;

                // Read-modify-write for a multi-component parameter, because a binding drives
                // ONE number: writing a one-element vector into a vec4 would be refused by the
                // setter's arity check, and inventing three more values would be a guess.
                monitor::vector_t v = p.arity > 1 ? p.get() : monitor::vector_t{};
                if (p.arity > 1) {
                    if (v.size() != p.arity)
                        return; // the producer's read and its declared arity disagree
                    v[std::min<std::size_t>(b.component, v.size() - 1)] = value;
                } else {
                    v.push_back(value);
                }
                p.set(v);
                return;
            }
            return;
        }

        // A MIXER FIELD, and it goes into the binding OVERLAY rather than into the layer's tween.
        //
        // That is what makes `UNBIND` lossless: the operator's constant is untouched for the
        // whole life of the binding, so handing the field back is clearing an overlay rather
        // than restoring a value nobody kept. It also puts the RANK in one place -- a binding
        // used to outrank the timeline by being written first and overwritten, which is a rank
        // expressed as the order of two loops and impossible to state without reading both.
        // `resolve_drivers` now applies timeline, then binding, then dominant, and the rank is
        // that line.
        //
        // Both halves of the transform, through their own tables. `tweened_transform::patch` is
        // still what a `MIXER` write uses and is still why it exists; it is no longer on this
        // path, and the collapse it fixed cannot happen here at all -- an overlay does not touch
        // the tween.
        const auto* f = fields::find(b.target);
        const auto* a = f ? nullptr : fields::find_audio_field(b.target);
        if (!f && !a)
            return;
        if (f ? (!f->set || !f->get) : (!a->set || !a->get))
            return;

        const auto arity = f ? f->arity : a->arity;
        auto&      ov    = drivers_[b.layer].binding;
        ov.owner         = "binding:" + std::to_string(b.id);
        // The `.N` suffix is re-attached for an arity>1 field, because `resolve_drivers` reads
        // the path and needs to know which component -- `b.target` had it split off at BIND time.
        if (arity > 1)
            ov.values[b.target + "." + std::to_string(static_cast<int>(b.component))] = value;
        else
            ov.values[b.target] = value;
    }

    std::future<void> add_source(const std::string& name, std::shared_ptr<binding::source> src)
    {
        return executor_.begin_invoke([this, name, src] {
            std::lock_guard<std::mutex> lock(binding_lock_);
            sources_[name] = src;
        });
    }

    std::future<bool> remove_source(const std::string& name)
    {
        return executor_.begin_invoke([this, name] {
            std::lock_guard<std::mutex> lock(binding_lock_);
            return sources_.erase(name) > 0;
        });
    }

    std::future<std::vector<stage_base::source_info>> list_sources()
    {
        return executor_.begin_invoke([this] {
            std::vector<stage_base::source_info> out;
            std::lock_guard<std::mutex>          lock(binding_lock_);
            for (const auto& kv : sources_)
                out.push_back({kv.first, kv.second->kind(), kv.second->describe(),
                               kv.second->channels()});
            return out;
        });
    }

    std::future<int> add_binding(binding::binding_def def)
    {
        return executor_.begin_invoke([this, def]() mutable {
            // The target must RESOLVE now. A binding to a misspelled field would otherwise be
            // accepted, evaluated on every tick, and do nothing -- with a 202 behind it.
            //
            // Through `address::parse`, which is the one place the registries are consulted.
            // This used to be an if/else that knew about two of them, which is why `volume` was
            // unbindable: it is not in the image table and nothing here looked anywhere else.
            const auto t = address::parse(def.target);
            switch (t.kind) {
                case address::target_kind::producer_param: {
                    auto it = layers_.find(def.layer);
                    if (it == layers_.end())
                        return 0;
                    auto producer = it->second.foreground();
                    if (producer == frame_producer::empty())
                        return 0;
                    bool found = false;
                    for (const auto& p : producer->parameters())
                        if (p.name == t.field)
                            found = true;
                    if (!found)
                        return 0;
                    break;
                }
                case address::target_kind::image:
                case address::target_kind::audio: {
                    // `set` is the arm a binding needs: a derived or blob field is describable
                    // and readable and cannot take a number every tick.
                    const auto* img = fields::find(t.field);
                    if (img ? !img->set : !fields::find_audio_field(t.field)->set)
                        return 0;
                    if (def.component >= t.meta->arity)
                        return 0;
                    break;
                }
                default:
                    // The previz registries resolve as targets (commit 10 gives them a writer);
                    // a binding to one is refused here rather than accepted and dropped.
                    return 0;
            }

            {
                std::lock_guard<std::mutex> lock(binding_lock_);
                if (sources_.find(def.source_name) == sources_.end())
                    return 0;
            }

            // ONE binding per target, replacing any earlier one. Two bindings writing the same
            // number would give whichever ran last, which is an ordering nobody chose -- the
            // same class of accident as auto-projection overwriting a hand-set ICVFX block.
            bindings_.erase(std::remove_if(bindings_.begin(), bindings_.end(),
                                           [&](const binding::binding_def& b) {
                                               return b.layer == def.layer &&
                                                      b.target == def.target &&
                                                      b.component == def.component;
                                           }),
                            bindings_.end());

            def.id     = next_binding_id_++;
            def.held   = 0.0;
            def.primed = false;
            bindings_.push_back(def);
            return def.id;
        });
    }

    std::future<int> remove_bindings(int layer, const std::string& target)
    {
        return executor_.begin_invoke([this, layer, target] {
            const auto before = bindings_.size();
            bindings_.erase(std::remove_if(bindings_.begin(), bindings_.end(),
                                           [&](const binding::binding_def& b) {
                                               if (layer >= 0 && b.layer != layer)
                                                   return false;
                                               if (target.empty())
                                                   return true;
                                               std::string field;
                                               uint8_t     comp = 0;
                                               address::split(target, field, comp);
                                               return b.target == field;
                                           }),
                            bindings_.end());
            return static_cast<int>(before - bindings_.size());
        });
    }

    /// Seek to the start of the next or previous instance in the document.
    ///
    /// NOT a transport verb, because the transport is pure and knows nothing about a document's
    /// contents -- it takes a position. So this reads the resolution, computes the position, and
    /// issues an ordinary seek. That keeps `transport` testable at boot against a table of
    /// numbers rather than against a document.
    ///
    /// "The next instance" is the earliest start strictly after the playhead, over every layer
    /// AND the transparent anchors: an anchor is a cue point that writes nothing, which is
    /// exactly the thing an operator wants to jump to.
    std::future<bool> timeline_chase(const std::string& name, const timeline::chase_config& cfg)
    {
        return executor_.begin_invoke([this, name, cfg] {
            if (!timelines_)
                return false;
            const auto entry = timelines_->get(name);
            if (!entry || entry->document.channel != channel_index_)
                return false;
            transports_[name].set_chase(cfg);
            return true;
        });
    }

    std::future<timeline::chase_config> timeline_chase_config(const std::string& name)
    {
        return executor_.begin_invoke([this, name] {
            const auto it = transports_.find(name);
            return it == transports_.end() ? timeline::chase_config{} : it->second.chase();
        });
    }

    std::future<bool> timeline_seek_relative(const std::string& name, bool forward)
    {
        return executor_.begin_invoke([this, name, forward] {
            if (!timelines_)
                return false;
            const auto entry = timelines_->get(name);
            if (!entry || entry->document.channel != channel_index_ || !entry->resolved.ok())
                return false;

            const auto per_frame = timeline::flicks_per_frame(format_desc_.framerate);
            auto&      tr        = transports_[name];
            const auto pos       = tr.position_at(last_frame_number_, per_frame);

            // STRICTLY after / strictly before, with a one-frame guard on the backwards case.
            // Without the guard, `PREVIOUS` pressed just after a cue started lands on that same
            // cue and looks like it did nothing -- which is the one behaviour an operator will
            // press twice and then report as broken.
            std::optional<timeline::flicks> target;
            for (const auto& in : entry->resolved.instances) {
                if (forward) {
                    if (in.start > pos && (!target || in.start < *target))
                        target = in.start;
                } else {
                    if (in.start < pos - per_frame && (!target || in.start > *target))
                        target = in.start;
                }
            }
            if (!target)
                return false;

            timeline::transport_command c;
            c.v  = timeline::transport_command::verb::seek;
            c.at = *target;
            pending_transport_[name].push_back(c);
            return true;
        });
    }

    std::future<bool> timeline_command(const std::string& name, const timeline::transport_command& cmd)
    {
        return executor_.begin_invoke([this, name, cmd] {
            if (!timelines_)
                return false;
            const auto entry = timelines_->get(name);
            // A document this channel does not own is not this channel's to drive. Refused
            // rather than queued and ignored: a `TIMELINE 1 PLAY show` against a document that
            // declares channel 2 is a mistake worth an error, not a no-op with a 202 behind it.
            if (!entry || entry->document.channel != channel_index_)
                return false;
            pending_transport_[name].push_back(cmd);
            return true;
        });
    }

    std::future<stage::timeline_status> timeline_state(const std::string& name)
    {
        return executor_.begin_invoke([this, name] {
            stage::timeline_status out;
            if (!timelines_)
                return out;
            const auto entry = timelines_->get(name);
            if (!entry || entry->document.channel != channel_index_)
                return out;
            out.exists    = true;
            out.ok        = entry->resolved.ok();
            out.faults    = entry->resolved.errors.size();
            out.instances = entry->resolved.instances.size();

            const auto it = transports_.find(name);
            if (it == transports_.end())
                return out;
            out.state = timeline::to_string(it->second.state());
            out.rate  = boost::rational_cast<double>(it->second.rate());
            // The position at the LAST tick, not at now: the playhead is derived from the frame
            // counter and asking for a frame that has not been ticked would report a position
            // the channel has not reached.
            out.position = timeline::to_seconds(
                it->second.position_at(last_frame_number_, timeline::flicks_per_frame(format_desc_.framerate)));
            return out;
        });
    }

    std::pair<std::string, std::string> driver_of(int layer, const std::string& path) const
    {
        // NOT on the executor: the caller is a write path that has to answer in its reply. The
        // overlays are written only on the executor, so this reads under the same lock the
        // binding mutators take -- the identical arrangement `is_bound` uses and for the
        // identical reason.
        std::lock_guard<std::mutex> lock(binding_lock_);
        const auto                  it = drivers_.find(layer);
        if (it == drivers_.end())
            return {};

        // The `.N` form is what an overlay keys on for an arity>1 field, and a caller asking
        // about `fill_translation` means the whole field -- so a prefix match on the component
        // suffix counts. Otherwise a write to a vector field whose X is bound would report
        // nobody.
        const auto touches = [&path](const layer_overlay& ov) {
            if (ov.values.find(path) != ov.values.end() || ov.steps.find(path) != ov.steps.end())
                return true;
            for (const auto& pv : ov.values) {
                if (pv.first.size() > path.size() + 1 && pv.first.compare(0, path.size(), path) == 0 &&
                    pv.first[path.size()] == '.')
                    return true;
            }
            return false;
        };

        const layer_overlay* ranked[3] = {&it->second.dominant, &it->second.binding,
                                          &it->second.timeline};
        std::string          effective, stack;
        for (const auto* ov : ranked) {
            if (ov->owner.empty() || !touches(*ov))
                continue;
            if (effective.empty())
                effective = ov->owner;
            if (!stack.empty())
                stack += ",";
            stack += ov->owner;
        }
        return {effective, stack};
    }

    std::future<bool> hold_field(int layer, const std::string& path)
    {
        return executor_.begin_invoke([this, layer, path] {
            const auto t = address::parse(path);
            if (t.kind != address::target_kind::image && t.kind != address::target_kind::audio)
                return false;

            // THE HELD VALUE IS WHAT IS ON AIR AT THIS MOMENT, not the constant. An operator
            // holding a parameter a document is driving means "keep it where it is and let me
            // move it from here" -- snapping to whatever they last typed, possibly minutes ago,
            // is the opposite of what they asked for.
            const auto     eff = effective_transform(layer);
            monitor::vector_t v;
            if (const auto* f = fields::find(t.field)) {
                if (!f->get || !f->set)
                    return false;
                v = f->get(eff.image_transform);
            } else if (const auto* a = fields::find_audio_field(t.field)) {
                if (!a->get || !a->set)
                    return false;
                v = a->get(eff.audio_transform);
            } else {
                return false;
            }

            std::lock_guard<std::mutex> lock(binding_lock_);
            auto&                       ov = drivers_[layer].dominant;
            ov.owner                       = "hold";
            // A single NUMBER goes in `values`; anything else -- a name, a boolean, a vector --
            // goes in `steps`, which is the same split the timeline uses and for the same
            // reason: only a number can be interpolated, and the overlays are read by one
            // writer that has to know which it has.
            if (v.size() == 1 && (boost::get<double>(&v[0]) || boost::get<int32_t>(&v[0]) ||
                                  boost::get<int64_t>(&v[0]))) {
                double d = 0;
                if (const auto* dd = boost::get<double>(&v[0]))
                    d = *dd;
                else if (const auto* ii = boost::get<int32_t>(&v[0]))
                    d = *ii;
                else
                    d = static_cast<double>(*boost::get<int64_t>(&v[0]));
                ov.values[t.path] = d;
            } else {
                ov.steps[t.path] = v;
            }
            return true;
        });
    }

    std::future<bool> release_field(int layer, const std::string& path)
    {
        return executor_.begin_invoke([this, layer, path] {
            std::lock_guard<std::mutex> lock(binding_lock_);
            const auto                  it = drivers_.find(layer);
            if (it == drivers_.end())
                return false;
            auto&      ov      = it->second.dominant;
            const auto removed = ov.values.erase(path) + ov.steps.erase(path);
            if (ov.empty())
                ov.owner.clear();
            return removed > 0;
        });
    }

    std::future<std::vector<std::pair<std::string, stage::timeline_status>>> timeline_list()
    {
        return executor_.begin_invoke([this] {
            std::vector<std::pair<std::string, stage::timeline_status>> out;
            if (!timelines_)
                return out;
            const auto per_frame = timeline::flicks_per_frame(format_desc_.framerate);
            for (const auto& name : timelines_->names()) {
                const auto entry = timelines_->get(name);
                if (!entry || entry->document.channel != channel_index_)
                    continue;
                stage::timeline_status st;
                st.exists    = true;
                st.ok        = entry->resolved.ok();
                st.faults    = entry->resolved.errors.size();
                st.instances = entry->resolved.instances.size();
                const auto it = transports_.find(name);
                if (it != transports_.end()) {
                    st.state    = timeline::to_string(it->second.state());
                    st.rate     = boost::rational_cast<double>(it->second.rate());
                    st.position = timeline::to_seconds(it->second.position_at(last_frame_number_, per_frame));
                }
                out.emplace_back(name, st);
            }
            return out;
        });
    }

    std::future<std::vector<binding::binding_def>> list_bindings()
    {
        return executor_.begin_invoke([this] { return bindings_; });
    }

    bool is_bound(int layer, const std::string& target) const
    {
        // NOT on the executor: the caller is a write path that has to answer now. `bindings_` is
        // a vector mutated only on the executor, so this reads under the same lock the mutators
        // take -- which is why they take it at all.
        std::string field;
        uint8_t     comp = 0;
        address::split(target, field, comp);

        std::lock_guard<std::mutex> lock(binding_lock_);
        for (const auto& b : bindings_)
            if (b.layer == layer && b.target == field)
                return true;
        return false;
    }

    void feed_sources(const input_event& event)
    {
        std::lock_guard<std::mutex> lock(binding_lock_);
        for (auto& kv : sources_)
            kv.second->feed(event);
    }

    std::future<frame_transform> get_current_transform(int index)
    {
        // THE EFFECTIVE TRANSFORM, not the constant, because every caller is a READER: the
        // `MIXER ... ` read forms and `MIXER FIELD <name>` with no value. A read that answered
        // the operator's constant while the picture showed a driver's value would make the two
        // facades disagree about what is on air -- and `binding-lfo`'s "both facades agree"
        // check found exactly that the moment bindings moved into an overlay, because until
        // then the binding WAS the constant.
        //
        // A writer that needs the constant does not come through here: `apply_transform` gets
        // the tween handed to its closure, which is the value it is composing onto.
        return executor_.begin_invoke([=, this] { return effective_transform(index); });
    }

    /// Offer an event to whichever layer wants it, topmost first.
    ///
    /// Posted rather than blocking: the caller is the screen consumer's render thread, or the
    /// protocol thread for `INPUT`, and neither can afford to wait on the stage executor.
    void input(const input_event& event)
    {
        executor_.begin_invoke([this, event] { input_on_executor(event); });
    }

    /// The targeted form: one layer, no hit-test, no rectangle check.
    ///
    /// It still undoes the layer's fill transform, so a client sending 0.5 0.5 means the middle of
    /// the LAYER either way -- what it skips is the rejection, not the conversion.
    void input(int layer, const input_event& event)
    {
        executor_.begin_invoke([this, layer, event] { offer(layer, event, false); });
    }

    std::future<std::vector<param_snapshot>> describe_params(int layer)
    {
        return executor_.begin_invoke([this, layer] {
            std::vector<param_snapshot> out;
            auto                        it = layers_.find(layer);
            if (it == layers_.end())
                return out;
            auto producer = it->second.foreground();
            if (producer == frame_producer::empty())
                return out;
            for (const auto& p : producer->parameters())
                out.push_back(snapshot_of(p));
            return out;
        });
    }

    std::future<bool> set_param(int layer, const std::string& name, const monitor::vector_t& value)
    {
        return executor_.begin_invoke([this, layer, name, value] {
            auto it = layers_.find(layer);
            if (it == layers_.end())
                return false;
            auto producer = it->second.foreground();
            if (producer == frame_producer::empty())
                return false;
            for (const auto& p : producer->parameters()) {
                if (p.name != name)
                    continue;
                if (p.access == fields::access_t::read)
                    return false;
                return p.set ? p.set(value) : false;
            }
            return false;
        });
    }

    /// Deliver to one layer, converting the channel-relative point into the layer's own space.
    ///
    /// Returns whether the producer consumed it. The conversion is the INVERSE of what the mixer
    /// does with `fill_translation` and `fill_scale`: a layer drawn at translation `t` with scale
    /// `s` occupies `[t, t + s]` of the channel, so a channel point `x` is `(x - t) / s` in the
    /// layer. `reject_outside` is false for a captured drag, which is the whole point of capture.
    ///
    /// Rotation, perspective corner-pin and crop are NOT inverted -- the same limit the 2013 API
    /// had. A rotated layer hit-tests as its unrotated rectangle, which is wrong and is documented
    /// as wrong rather than silently approximated.
    bool offer(int index, const input_event& event, bool reject_outside)
    {
        auto it = layers_.find(index);
        if (it == layers_.end())
            return false;

        auto producer = it->second.foreground();
        if (producer == frame_producer::empty())
            return false;

        auto local = event;

        if (event.has_position()) {
            const auto& img = tweens_[index].fetch().image_transform;

            const double sx = img.fill_scale[0];
            const double sy = img.fill_scale[1];
            if (sx == 0.0 || sy == 0.0)
                return false; // a layer scaled to nothing has no interior to hit

            local.x = (event.x - img.fill_translation[0]) / sx;
            local.y = (event.y - img.fill_translation[1]) / sy;

            if (reject_outside && !local.on_surface())
                return false;
        }

        return producer->input(local);
    }

    void input_on_executor(const input_event& event)
    {
        // Held buttons are tracked HERE rather than read from the event's modifier bits. The
        // source fills those from the platform's own state, and a press whose release never
        // arrived -- a window losing capture, a caller posting a DOWN and then giving up -- would
        // leave a drag latched forever with nothing to clear it.
        if (event.type == input_event::kind::button) {
            const uint32_t bit = event.button == 0   ? mod_left_button
                                 : event.button == 1 ? mod_middle_button
                                 : event.button == 2 ? mod_right_button
                                                     : 0u;
            if (event.pressed)
                input_buttons_ |= bit;
            else
                input_buttons_ &= ~bit;
        }

        // A leave ends nothing. A drag that travels off the window is still a drag and its release
        // will arrive, because the window holds capture. Only the focused layer is told, so a page
        // can clear its own hover state.
        if (event.type == input_event::kind::leave || event.type == input_event::kind::key ||
            event.type == input_event::kind::text) {
            if (input_focus_ >= 0)
                offer(input_focus_, event, false);
            return;
        }

        // A drag in progress belongs to the layer it started on, wherever the pointer now is.
        if (input_capture_ >= 0) {
            offer(input_capture_, event, false);
            if (input_buttons_ == 0)
                input_capture_ = -1;
            return;
        }

        // Topmost first. `layers_` is ordered ascending by index and the mixer draws in that
        // order, so the LAST layer is the one on top.
        //
        // A producer returning false does not end the search: geometry decides the ORDER, and the
        // producer decides whether it consumes. So a colour layer sitting above an HTML page does
        // not swallow every click for being on top -- which is what the 2013 API's "topmost hit
        // layer wins" would have done, since every consumer carried a sink whether it wanted one
        // or not.
        for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
            if (!offer(it->first, event, true))
                continue;

            input_focus_ = it->first;
            if (event.type == input_event::kind::button && event.pressed)
                input_capture_ = it->first;
            return;
        }
    }

    std::future<void> load(int index, const spl::shared_ptr<frame_producer>& producer, bool preview, bool auto_play)
    {
        return executor_.begin_invoke([=, this] { get_layer(index).load(producer, preview, auto_play); });
    }

    std::future<void> preview(int index)
    {
        return executor_.begin_invoke([=, this] { get_layer(index).preview(); });
    }

    std::future<void> pause(int index)
    {
        return executor_.begin_invoke([=, this] { get_layer(index).pause(); });
    }

    std::future<void> resume(int index)
    {
        return executor_.begin_invoke([=, this] { get_layer(index).resume(); });
    }

    std::future<void> play(int index)
    {
        return executor_.begin_invoke([=, this] { get_layer(index).play(); });
    }

    std::future<void> stop(int index)
    {
        return executor_.begin_invoke([=, this] { get_layer(index).stop(); });
    }

    std::future<void> clear(int index)
    {
        return executor_.begin_invoke([=, this] { layers_.erase(index); });
    }

    std::future<void> clear()
    {
        return executor_.begin_invoke([=, this] { layers_.clear(); });
    }

    std::future<void> swap_layers(const std::shared_ptr<stage>& other, bool swap_transforms)
    {
        auto other_impl = other->impl_;

        if (other_impl.get() == this) {
            return make_ready_future();
        }

        auto func = [=, this] {
            auto layers       = layers_ | boost::adaptors::map_values;
            auto other_layers = other_impl->layers_ | boost::adaptors::map_values;

            std::swap(layers_, other_impl->layers_);

            if (swap_transforms)
                std::swap(tweens_, other_impl->tweens_);
        };

        return invoke_both(other, func);
    }

    std::future<void> swap_layer(int index, int other_index, bool swap_transforms)
    {
        return executor_.begin_invoke([=, this] {
            std::swap(get_layer(index), get_layer(other_index));

            if (swap_transforms)
                std::swap(tweens_[index], tweens_[other_index]);
        });
    }

    std::future<void> swap_layer(int index, int other_index, const std::shared_ptr<stage>& other, bool swap_transforms)
    {
        auto other_impl = other->impl_;

        if (other_impl.get() == this)
            return swap_layer(index, other_index, swap_transforms);
        auto func = [=, this] {
            auto& my_layer    = get_layer(index);
            auto& other_layer = other_impl->get_layer(other_index);

            std::swap(my_layer, other_layer);

            if (swap_transforms) {
                auto& my_tween    = tweens_[index];
                auto& other_tween = other_impl->tweens_[other_index];
                std::swap(my_tween, other_tween);
            }
        };

        return invoke_both(other, func);
    }

    std::future<void> invoke_both(const std::shared_ptr<stage>& other, std::function<void()> func)
    {
        auto other_impl = other->impl_;

        if (other_impl->channel_index_ < channel_index_) {
            return other_impl->executor_.begin_invoke([=, this] { executor_.invoke(func); });
        }

        return executor_.begin_invoke([=, this] { other_impl->executor_.invoke(func); });
    }

    std::future<std::shared_ptr<frame_producer>> foreground(int index)
    {
        return executor_.begin_invoke(
            [=, this]() -> std::shared_ptr<frame_producer> { return get_layer(index).foreground(); });
    }

    std::future<std::shared_ptr<frame_producer>> background(int index)
    {
        return executor_.begin_invoke(
            [=, this]() -> std::shared_ptr<frame_producer> { return get_layer(index).background(); });
    }

    std::future<std::wstring> call(int index, const std::vector<std::wstring>& params)
    {
        return flatten(executor_.begin_invoke([=, this] {
            auto result = get_layer(index).foreground()->call(params).share();


            return result;
        }));
    }
    std::future<std::wstring> callbg(int index, const std::vector<std::wstring>& params)
    {
        return flatten(
            executor_.begin_invoke([=, this] { return get_layer(index).background()->call(params).share(); }));
    }

    std::unique_lock<std::mutex> get_lock() { return std::move(std::unique_lock<std::mutex>(lock_)); }

    // ── Keyframe management (all on executor) ─────────────────────────────

    core::video_format_desc video_format_desc() const
    {
        std::lock_guard<std::mutex> lock(format_desc_mutex_);
        return format_desc_;
    }

    std::future<void> video_format_desc(const core::video_format_desc& format_desc)
    {
        return executor_.begin_invoke([=, this] {
            {
                std::lock_guard<std::mutex> lock(format_desc_mutex_);
                format_desc_ = format_desc;
            }

            layers_.clear();
        });
    }
};

stage::stage(int channel_index, spl::shared_ptr<diagnostics::graph> graph, const core::video_format_desc& format_desc)
    : impl_(new impl(channel_index, std::move(graph), format_desc))
{
}
std::future<std::wstring> stage::call(int index, const std::vector<std::wstring>& params)
{
    return impl_->call(index, params);
}
std::future<std::wstring> stage::callbg(int index, const std::vector<std::wstring>& params)
{
    return impl_->callbg(index, params);
}
std::future<void> stage::apply_transforms(const std::vector<stage::transform_tuple_t>& transforms)
{
    return impl_->apply_transforms(transforms);
}
std::future<void> stage::apply_transform(int                                                                index,
                                         const std::function<core::frame_transform(core::frame_transform)>& transform,
                                         unsigned int   mix_duration,
                                         const tweener& tween)
{
    return impl_->apply_transform(index, transform, mix_duration, tween);
}
std::future<void>            stage::clear_transforms(int index) { return impl_->clear_transforms(index); }
std::future<void>            stage::clear_transforms() { return impl_->clear_transforms(); }
std::future<frame_transform> stage::get_current_transform(int index) { return impl_->get_current_transform(index); }
std::future<void> stage::load(int index, const spl::shared_ptr<frame_producer>& producer, bool preview, bool auto_play)
{
    return impl_->load(index, producer, preview, auto_play);
}
std::future<void> stage::preview(int index) { return impl_->preview(index); }
std::future<void> stage::pause(int index) { return impl_->pause(index); }
std::future<void> stage::resume(int index) { return impl_->resume(index); }
std::future<void> stage::play(int index) { return impl_->play(index); }
std::future<void> stage::stop(int index) { return impl_->stop(index); }
std::future<void> stage::clear(int index) { return impl_->clear(index); }
std::future<void> stage::clear() { return impl_->clear(); }
std::future<void> stage::swap_layers(const std::shared_ptr<stage_base>& other, bool swap_transforms)
{
    const auto other2 = std::static_pointer_cast<stage>(other);
    return impl_->swap_layers(other2, swap_transforms);
}
std::future<void> stage::swap_layer(int index, int other_index, bool swap_transforms)
{
    return impl_->swap_layer(index, other_index, swap_transforms);
}
std::future<void>
stage::swap_layer(int index, int other_index, const std::shared_ptr<stage_base>& other, bool swap_transforms)
{
    const auto other2 = std::static_pointer_cast<stage>(other);
    return impl_->swap_layer(index, other_index, other2, swap_transforms);
}
std::future<std::shared_ptr<frame_producer>> stage::foreground(int index) { return impl_->foreground(index); }
std::future<std::shared_ptr<frame_producer>> stage::background(int index) { return impl_->background(index); }
const stage_frames                           stage::operator()(uint64_t                                     frame_number,
                                     std::vector<int>&                            fetch_background,
                                     std::function<void(int, const layer_frame&)> routesCb)
{
    return (*impl_)(frame_number, fetch_background, routesCb);
}
core::monitor::state    stage::state() const { return impl_->state_; }
core::video_format_desc stage::video_format_desc() const { return impl_->video_format_desc(); }
std::future<void>       stage::video_format_desc(const core::video_format_desc& format_desc)
{
    return impl_->video_format_desc(format_desc);
}
std::unique_lock<std::mutex> stage::get_lock() const { return impl_->get_lock(); }
std::future<void>            stage::execute(std::function<void()> func)
{
    func();
    return make_ready_future();
}
void stage::input(const input_event& event) { impl_->input(event); }
void stage::input(int layer, const input_event& event) { impl_->input(layer, event); }
std::future<std::vector<param_snapshot>> stage::describe_params(int layer)
{
    return impl_->describe_params(layer);
}
std::future<bool> stage::set_param(int layer, const std::string& name, const monitor::vector_t& value)
{
    return impl_->set_param(layer, name, value);
}
std::future<void> stage::add_source(const std::string& name, std::shared_ptr<binding::source> src)
{
    return impl_->add_source(name, std::move(src));
}
std::future<bool> stage::remove_source(const std::string& name) { return impl_->remove_source(name); }
std::future<std::vector<stage_base::source_info>> stage::list_sources() { return impl_->list_sources(); }
std::future<bool> stage::timeline_command(const std::string&                 name,
                                          const timeline::transport_command& cmd)
{
    return impl_->timeline_command(name, cmd);
}

std::future<bool> stage::timeline_seek_relative(const std::string& name, bool forward)
{
    return impl_->timeline_seek_relative(name, forward);
}

std::future<stage::timeline_status> stage::timeline_state(const std::string& name)
{
    return impl_->timeline_state(name);
}

std::future<std::vector<std::pair<std::string, stage::timeline_status>>> stage::timeline_list()
{
    return impl_->timeline_list();
}

std::pair<std::string, std::string> stage::driver_of(int layer, const std::string& path) const
{
    return impl_->driver_of(layer, path);
}

std::future<bool> stage::hold_field(int layer, const std::string& path)
{
    return impl_->hold_field(layer, path);
}

std::future<bool> stage::release_field(int layer, const std::string& path)
{
    return impl_->release_field(layer, path);
}

void stage::set_stage_field_writer(stage::stage_field_writer w)
{
    impl_->stage_field_writer_ = std::move(w);
}

void stage::set_timecode_source(stage::timecode_source src)
{
    impl_->timecode_source_ = std::move(src);
}

std::future<bool> stage::timeline_chase(const std::string& name, const timeline::chase_config& cfg)
{
    return impl_->timeline_chase(name, cfg);
}

std::future<timeline::chase_config> stage::timeline_chase_config(const std::string& name)
{
    return impl_->timeline_chase_config(name);
}

void stage::set_producer_factory(stage::clip_factory factory)
{
    impl_->clip_factory_ = std::move(factory);
}

void stage::set_timeline_store(std::shared_ptr<timeline::timeline_store> store)
{
    // NOT on the executor, and it does not need to be: this is called once, from the shell,
    // before the channel ticks. Posting it would make the first frame's fingerprint depend on
    // whether the task had run yet.
    impl_->timelines_ = std::move(store);
}

std::future<int>  stage::add_binding(const binding::binding_def& def) { return impl_->add_binding(def); }
std::future<int>  stage::remove_bindings(int layer, const std::string& target)
{
    return impl_->remove_bindings(layer, target);
}
std::future<std::vector<binding::binding_def>> stage::list_bindings() { return impl_->list_bindings(); }
bool stage::is_bound(int layer, const std::string& target) const { return impl_->is_bound(layer, target); }
void stage::feed_sources(const input_event& event) { impl_->feed_sources(event); }

// ── Keyframe management (stage wrappers) ─────────────────────────────────

// STAGE DELAYED (For batching operations)
stage_delayed::stage_delayed(const std::shared_ptr<stage>& st, int index)
    : executor_{L"batch stage " + boost::lexical_cast<std::wstring>(index)}
    , stage_(st)
{
    // Start the executor blocked on a future that will complete when we are ready for it to execute
    executor_.begin_invoke([=, this]() -> void { waiter_.get_future().get(); });
}

std::future<std::wstring> stage_delayed::call(int index, const std::vector<std::wstring>& params)
{
    return executor_.begin_invoke([=, this]() -> std::wstring { return stage_->call(index, params).get(); });
}
std::future<std::wstring> stage_delayed::callbg(int index, const std::vector<std::wstring>& params)
{
    return executor_.begin_invoke([=, this]() -> std::wstring { return stage_->callbg(index, params).get(); });
}
std::future<void> stage_delayed::apply_transforms(const std::vector<stage_delayed::transform_tuple_t>& transforms)
{
    return executor_.begin_invoke([=, this]() { return stage_->apply_transforms(transforms).get(); });
}
std::future<void>
stage_delayed::apply_transform(int                                                                index,
                               const std::function<core::frame_transform(core::frame_transform)>& transform,
                               unsigned int                                                       mix_duration,
                               const tweener&                                                     tween)
{
    return executor_.begin_invoke(
        [=, this]() { return stage_->apply_transform(index, transform, mix_duration, tween).get(); });
}
std::future<void> stage_delayed::clear_transforms(int index)
{
    return executor_.begin_invoke([=, this]() { return stage_->clear_transforms(index).get(); });
}
std::future<void> stage_delayed::clear_transforms()
{
    return executor_.begin_invoke([=, this]() { return stage_->clear_transforms().get(); });
}
std::future<frame_transform> stage_delayed::get_current_transform(int index)
{
    return executor_.begin_invoke([=, this]() { return stage_->get_current_transform(index).get(); });
}
std::future<void>
stage_delayed::load(int index, const spl::shared_ptr<frame_producer>& producer, bool preview, bool auto_play)
{
    return executor_.begin_invoke([=, this]() { return stage_->load(index, producer, preview, auto_play).get(); });
}
std::future<void> stage_delayed::preview(int index)
{
    return executor_.begin_invoke([=, this]() { return stage_->preview(index).get(); });
}
std::future<void> stage_delayed::pause(int index)
{
    return executor_.begin_invoke([=, this]() { return stage_->pause(index).get(); });
}
std::future<void> stage_delayed::resume(int index)
{
    return executor_.begin_invoke([=, this]() { return stage_->resume(index).get(); });
}
std::future<void> stage_delayed::play(int index)
{
    return executor_.begin_invoke([=, this]() { return stage_->play(index).get(); });
}
std::future<void> stage_delayed::stop(int index)
{
    return executor_.begin_invoke([=, this]() { return stage_->stop(index).get(); });
}
std::future<void> stage_delayed::clear(int index)
{
    return executor_.begin_invoke([=, this]() { return stage_->clear(index).get(); });
}
std::future<void> stage_delayed::clear()
{
    return executor_.begin_invoke([=, this]() { return stage_->clear().get(); });
}
std::future<void> stage_delayed::swap_layers(const std::shared_ptr<stage_base>& other, bool swap_transforms)
{
    const auto other2 = std::static_pointer_cast<stage_delayed>(other);
    return executor_.begin_invoke([=, this]() { return stage_->swap_layers(other2->stage_, swap_transforms).get(); });
}
std::future<void> stage_delayed::swap_layer(int index, int other_index, bool swap_transforms)
{
    return executor_.begin_invoke(
        [=, this]() { return stage_->swap_layer(index, other_index, swap_transforms).get(); });
}
std::future<void>
stage_delayed::swap_layer(int index, int other_index, const std::shared_ptr<stage_base>& other, bool swap_transforms)
{
    const auto other2 = std::static_pointer_cast<stage_delayed>(other);

    // Something so that we know to lock the channel
    other2->executor_.begin_invoke([]() {});

    return executor_.begin_invoke(
        [=, this]() { return stage_->swap_layer(index, other_index, other2->stage_, swap_transforms).get(); });
}

std::future<std::shared_ptr<frame_producer>> stage_delayed::foreground(int index)
{
    return executor_.begin_invoke(
        [=, this]() -> std::shared_ptr<frame_producer> { return stage_->foreground(index).get(); });
}
std::future<std::shared_ptr<frame_producer>> stage_delayed::background(int index)
{
    return executor_.begin_invoke(
        [=, this]() -> std::shared_ptr<frame_producer> { return stage_->background(index).get(); });
}

std::future<void> stage_delayed::execute(std::function<void()> func)
{
    return executor_.begin_invoke([=, this]() { return stage_->execute(func).get(); });
}

std::future<bool> stage_delayed::timeline_command(const std::string&                       name,
                                                  const timeline::transport_command& cmd)
{
    // Queued against the DELAYED executor like every other op, so a `{"op": "timeline"}` in a
    // batch reaches the stage in the same released burst as the batch's field writes and both
    // land on the same tick. Calling the real stage from here instead would deadlock: the
    // delayed stage is holding that executor.
    return executor_.begin_invoke([=, this]() { return stage_->timeline_command(name, cmd).get(); });
}

// ── Keyframe management (stage_delayed forwarding) ───────────────────────

}} // namespace caspar::core
