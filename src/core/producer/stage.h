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

#include "../fwd.h"
#include "../input/input_event.h"
#include "../monitor/monitor.h"
#include "producer_params.h"

#include <core/binding/binding.h>

#include <common/executor.h>
#include <common/future.h>
#include <common/memory.h>
#include <common/tweener.h>

#include <core/frame/draw_frame.h>
#include <core/video_format.h>

#include <functional>
#include <future>
#include <map>
#include <mutex>
#include <tuple>
#include <vector>

namespace caspar::diagnostics {
class graph;
}

namespace caspar { namespace core {

namespace timeline {
class timeline_store;
struct transport_command;
struct chase_config;
}

namespace graph {
class graph_store;
}


struct layer_frame
{
    bool       is_interlaced;
    draw_frame foreground1;
    draw_frame foreground1_raw; // raw producer output before mixer transforms are applied
    draw_frame background1;
    draw_frame foreground2;
    draw_frame foreground2_raw; // raw producer output before mixer transforms are applied (field 2)
    draw_frame background2;
    bool       has_background;
};

struct stage_frames
{
    core::video_format_desc format_desc;
    int                     nb_samples;
    std::vector<draw_frame> frames;
    std::vector<draw_frame> frames2;
};

/**
 * Base class for the stage. Should be used when either stage or stage_delayed may be used
 */
class stage_base
{
  public:
    using transform_func_t  = std::function<struct frame_transform(struct frame_transform)>;
    using transform_tuple_t = std::tuple<int, transform_func_t, unsigned int, tweener>;

    virtual ~stage_base() {}

    // Methods
    virtual std::future<void> apply_transforms(const std::vector<transform_tuple_t>& transforms) = 0;
    virtual std::future<void>
    apply_transform(int index, const transform_func_t& transform, unsigned int mix_duration, const tweener& tween) = 0;
    virtual std::future<void>            clear_transforms(int index)                                               = 0;
    virtual std::future<void>            clear_transforms()                                                        = 0;
    virtual std::future<frame_transform> get_current_transform(int index)                                          = 0;
    virtual std::future<void>
    load(int index, const spl::shared_ptr<frame_producer>& producer, bool preview = false, bool auto_play = false) = 0;
    virtual std::future<void>         preview(int index)                                                           = 0;
    virtual std::future<void>         pause(int index)                                                             = 0;
    virtual std::future<void>         resume(int index)                                                            = 0;
    virtual std::future<void>         play(int index)                                                              = 0;
    virtual std::future<void>         stop(int index)                                                              = 0;
    virtual std::future<std::wstring> call(int index, const std::vector<std::wstring>& params)                     = 0;
    virtual std::future<std::wstring> callbg(int index, const std::vector<std::wstring>& params)                   = 0;
    virtual std::future<void>         clear(int index)                                                             = 0;
    virtual std::future<void>         clear()                                                                      = 0;
    virtual std::future<void>         swap_layers(const std::shared_ptr<stage_base>& other, bool swap_transforms)  = 0;
    virtual std::future<void>         swap_layer(int index, int other_index, bool swap_transforms)                 = 0;
    virtual std::future<void>
    swap_layer(int index, int other_index, const std::shared_ptr<stage_base>& other, bool swap_transforms) = 0;

    virtual std::future<void> execute(std::function<void()> k) = 0;

    /// Offer a pointer or keyboard event to whichever layer wants it.
    ///
    /// Default is a no-op, so a stage implementation that does not route input -- and
    /// `stage_delayed`, which exists only to defer writes inside a batch -- needs nothing. The
    /// real implementation hit-tests layers topmost-first; see `stage::impl::input`.
    virtual void input(const input_event& event) {}

    /// A layer's producer parameters, as pure data. Empty for a layer with no producer or a
    /// producer with no parameters -- the two are indistinguishable here, deliberately: a
    /// caller wanting to know whether a layer exists has `foreground()`.
    ///
    /// Snapshots rather than `param_desc`, because a `param_desc` carries two closures bound
    /// to the producer and the callers are the API's reader threads -- exactly the ones who
    /// would keep one past a `CLEAR`.
    virtual std::future<std::vector<param_snapshot>> describe_params(int layer)
    {
        return make_ready_future(std::vector<param_snapshot>());
    }

    /// Write one parameter by name. False for an unknown name, a wrong type or arity, or a
    /// value the producer refuses. Runs on the stage executor, like `call`.
    virtual std::future<bool> set_param(int layer, const std::string& name, const monitor::vector_t& value)
    {
        return make_ready_future(false);
    }

    // -- Bindings: a target driven by a live source, evaluated in the tick --------------
    //
    // Every one of these runs on the stage executor, because that is where the transform and
    // the producers are and because the API's write path is too slow for anything continuous
    // (a single serial thread blocking on a stage round trip per write). See
    // `core/binding/binding.h`.

    /// Create or replace a source. `SOURCE ADD`.
    virtual std::future<void> add_source(const std::string& name, std::shared_ptr<binding::source> src)
    {
        return make_ready_future();
    }

    /// Remove one. Bindings that referenced it go `broken` rather than disappearing, so an
    /// operator who removes the wrong source sees why their parameter stopped moving.
    virtual std::future<bool> remove_source(const std::string& name) { return make_ready_future(false); }

    /// Every source, as `{name, kind, description, channels}`. For `SOURCE LIST` and the tree.
    struct source_info
    {
        std::string              name;
        std::string              kind;
        std::string              description;
        std::vector<std::string> channels;
    };
    virtual std::future<std::vector<source_info>> list_sources()
    {
        return make_ready_future(std::vector<source_info>());
    }

    /// Create a binding. Returns its id, or 0 if the target or the source does not resolve.
    ///
    /// Resolved ONCE, here, rather than on every tick: a target becomes a descriptor plus a
    /// component index and a source becomes a pointer, so the per-tick cost is arithmetic and
    /// a write. It also means a bad target is refused at `BIND` time with a reason instead of
    /// being a binding that runs forever and does nothing.
    virtual std::future<int> add_binding(const binding::binding_def& def) { return make_ready_future(0); }

    /// Remove the binding on a target, or every binding if `target` is empty. Returns how many.
    virtual std::future<int> remove_bindings(int layer, const std::string& target)
    {
        return make_ready_future(0);
    }

    virtual std::future<std::vector<binding::binding_def>> list_bindings()
    {
        return make_ready_future(std::vector<binding::binding_def>());
    }

    /// Is this exact target owned by a binding? Synchronous, for a write path that has to
    /// REFUSE rather than be silently overwritten one frame later.
    virtual bool is_bound(int layer, const std::string& target) const { return false; }

    /// WHO IS DRIVING `path` on `layer`, and everyone else who wanted it.
    ///
    /// `first` is the effective owner -- the strongest rank writing that path -- and `second` is
    /// every rank that wanted it, strongest first, comma-separated. Both empty means nobody.
    ///
    /// Defaulted to "nobody" rather than pure, so a stage with no drivers needs no override --
    /// which is what `stage_delayed` is until a batch grows an ownership question of its own.
    virtual std::pair<std::string, std::string> driver_of(int layer, const std::string& path) const
    {
        return {};
    }

    /// Take a path for the operator, above every other rank, until `release_field`.
    virtual std::future<bool> hold_field(int layer, const std::string& path)
    {
        return make_ready_future(false);
    }

    /// Give it back. Whatever was underneath takes it again on the next tick.
    virtual std::future<bool> release_field(int layer, const std::string& path)
    {
        return make_ready_future(false);
    }

    // -- The node graph: attach a document to a layer, and read or write its parameters -----
    //
    // On the BASE for the same reason `timeline_command` is: a `/v1/batch` op has to reach
    // these through its `stage_delayed`, and reaching past it to the real stage would block the
    // HTTP thread against a lock it is itself holding. Defaulted rather than pure, so a stage
    // with no graph support needs no override.

    /// What happened to an `ATTACH`. Three outcomes rather than a bool, because a client acts
    /// on them differently: `graph_not_found` is a name to fix, `already_attached` is a document
    /// in use somewhere else, and `layer_busy` is this layer already holding another.
    enum class attach_result
    {
        ok,
        no_such_graph,
        already_attached,
        layer_busy,
    };

    /// Attach `name` to `layer`. Re-attaching the same document to the same layer is `ok` --
    /// idempotent, which a client retrying after a timeout depends on.
    virtual std::future<attach_result> attach_graph(int layer, const std::string& name)
    {
        return make_ready_future(attach_result::no_such_graph);
    }

    /// Detach whatever is on `layer`. False if there was nothing.
    virtual std::future<bool> detach_graph(int layer) { return make_ready_future(false); }

    /// Which document `layer` has, or empty. Synchronous, for a write path that has to answer
    /// before it can validate anything.
    virtual std::string graph_of(int layer) const { return {}; }

    /// ONE node parameter, described. The registry's port descriptor with the DOCUMENT's value
    /// in it -- which is the only place those two meet, and the reason this is a stage call
    /// rather than a registry lookup: the registry has the type and the document has the value.
    ///
    /// The `name` field is the ADDRESS (`node/<id>/<param>`) rather than the bare port name, so
    /// what a caller publishes, writes and reads back is one string.
    virtual std::future<std::vector<param_snapshot>> describe_graph(int layer)
    {
        return make_ready_future(std::vector<param_snapshot>());
    }

    /// Write one node parameter, by address. False for a layer with no graph, an unknown node
    /// or port, or a value the validator refuses.
    ///
    /// `label` names the GESTURE for the store's undo history, so a slider drag under one label
    /// is one undo rather than fifty.
    /// One graph VERB, by document name: `attach`, `detach`, `undo` or `redo`.
    ///
    /// ON `stage_base` RATHER THAN ONLY ON `stage`, for the reason `timeline_command` is: a
    /// batch applies against a `stage_delayed`, and reaching past it to the real stage would
    /// both miss the batch's frame and deadlock the HTTP thread on an executor the delayed
    /// stage is holding.
    ///
    /// `layer` is read only by `attach`; the other three take it and ignore it, because where a
    /// document is attached is the STORE's business and a client asking to undo should not have
    /// to remember where the look happens to be.
    virtual std::future<bool> graph_command(const std::string& /*name*/,
                                            const std::string& /*verb*/,
                                            int /*layer*/)
    {
        return make_ready_future(false);
    }

    virtual std::future<bool> set_node_param(int                      layer,
                                             const std::string&       path,
                                             const monitor::vector_t& value,
                                             const std::string&       label)
    {
        return make_ready_future(false);
    }

    /// Queue one transport command for a document this stage's channel owns.
    ///
    /// ON THE BASE, and it has to be: a `/v1/batch` holding this channel's executor blocked
    /// must still be able to start a document. A batch reaching past its `stage_delayed` to the
    /// real stage would not merely land on the wrong frame -- the delayed stage holds that
    /// executor, so the call would block the HTTP thread against a lock it is itself holding.
    /// Going through the base means one implementation of the verb table serves the route, the
    /// batch and AMCP.
    ///
    /// Defaulted to "no such document" rather than pure, for the same reason as `hold_field`.
    virtual std::future<bool> timeline_command(const std::string&                 name,
                                               const timeline::transport_command& cmd)
    {
        return make_ready_future(false);
    }

    /// Hand an input event to every source that wants one. Called from `video_channel::input`.
    virtual void feed_sources(const input_event&) {}

    /// Deliver to ONE layer, with no hit-test and no rectangle check.
    ///
    /// For a caller that already knows its target -- `INPUT 1-10 ...`, or a client driving a
    /// template it just loaded. A client that knows which layer it wants should not have its
    /// event silently dropped because the layer happens to be scaled or translated away from
    /// where the client thinks it is; that is the hit-test's job, not this one's.
    virtual void input(int layer, const input_event& event) {}

    // Keyframe management (type-erased: void* wraps module types)

    // Properties
    virtual std::future<std::shared_ptr<frame_producer>> foreground(int index) = 0;
    virtual std::future<std::shared_ptr<frame_producer>> background(int index) = 0;
};

/**
 * The normal stage implementation.
 */
class stage final : public stage_base
{
    stage(const stage&);
    stage& operator=(const stage&);

  public:
    explicit stage(int                                         channel_index,
                   spl::shared_ptr<caspar::diagnostics::graph> graph,
                   const core::video_format_desc&              format_desc);

    const stage_frames operator()(uint64_t                                     frame_number,
                                  std::vector<int>&                            fetch_background,
                                  std::function<void(int, const layer_frame&)> routesCb);

    std::future<void>            apply_transforms(const std::vector<transform_tuple_t>& transforms) override;
    std::future<void>            apply_transform(int                     index,
                                                 const transform_func_t& transform,
                                                 unsigned int            mix_duration,
                                                 const tweener&          tween) override;
    std::future<void>            clear_transforms(int index) override;
    std::future<void>            clear_transforms() override;
    std::future<frame_transform> get_current_transform(int index) override;
    std::future<void>            load(int                                    index,
                                      const spl::shared_ptr<frame_producer>& producer,
                                      bool                                   preview   = false,
                                      bool                                   auto_play = false) override;
    std::future<void>            preview(int index) override;
    std::future<void>            pause(int index) override;
    std::future<void>            resume(int index) override;
    std::future<void>            play(int index) override;
    std::future<void>            stop(int index) override;
    std::future<std::wstring>    call(int index, const std::vector<std::wstring>& params) override;
    std::future<std::wstring>    callbg(int index, const std::vector<std::wstring>& params) override;
    std::future<void>            clear(int index) override;
    std::future<void>            clear() override;
    std::future<void>            swap_layers(const std::shared_ptr<stage_base>& other, bool swap_transforms) override;
    std::future<void>            swap_layer(int index, int other_index, bool swap_transforms) override;
    std::future<void>
    swap_layer(int index, int other_index, const std::shared_ptr<stage_base>& other, bool swap_transforms) override;

    core::monitor::state state() const;

    std::future<std::shared_ptr<frame_producer>> foreground(int index) override;
    std::future<std::shared_ptr<frame_producer>> background(int index) override;

    std::future<void>            execute(std::function<void()> k) override;

    /// Route an event to the topmost layer that consumes it. See `stage::impl::input`.
    void                         input(const input_event& event) override;
    void                         input(int layer, const input_event& event) override;

    std::future<std::vector<param_snapshot>> describe_params(int layer) override;
    std::future<bool> set_param(int layer, const std::string& name, const monitor::vector_t& value) override;

    std::future<void>                    add_source(const std::string& name,
                                                    std::shared_ptr<binding::source> src) override;
    std::future<bool>                    remove_source(const std::string& name) override;
    std::future<std::vector<source_info>> list_sources() override;
    std::future<int>                     add_binding(const binding::binding_def& def) override;
    std::future<int>                     remove_bindings(int layer, const std::string& target) override;
    std::future<std::vector<binding::binding_def>> list_bindings() override;
    bool                                 is_bound(int layer, const std::string& target) const override;
    void                                 feed_sources(const input_event& event) override;


    /// The server-wide timeline documents, injected by the shell.
    ///
    /// The stage does not OWN it -- one document may drive several channels -- and this build
    /// only reads its revision, which is mixed into the structure fingerprint so a client
    /// re-walks the tree when a document appears or changes. The tick evaluation arrives with
    /// the next commit; the pointer is here now because the fingerprint is what makes a PUT
    /// observable from outside the process at all.
    void set_timeline_store(std::shared_ptr<timeline::timeline_store> store);

    /// The node graphs. Injected by the shell for the same reason the timeline store is:
    /// one store for the server, and the stage needs its STRUCTURE revision in the
    /// fingerprint so a PUT is observable to a client walking the tree.
    void set_graph_store(std::shared_ptr<graph::graph_store> store);

    /// HOW THE TICK BUILDS A CLIP, injected by the shell.
    ///
    /// A bridge for the same reason the previz writer and the timecode source are: building a
    /// producer needs `frame_producer_registry` and a `frame_producer_dependencies` carrying the
    /// channel's frame factory, format and channel_info -- which the shell assembles and the
    /// stage has no business assembling for itself.
    ///
    /// CALLED OFF THE STAGE EXECUTOR, on a worker thread, because a build opens a file and
    /// decodes: doing it in the tick would drop frames on every cue. It may throw, and the
    /// caller turns that into a published fault rather than letting it reach the frame path.
    using clip_factory = std::function<spl::shared_ptr<frame_producer>(const std::wstring& clip)>;
    void set_producer_factory(clip_factory factory);

    /// How the stage writes a previz screen or camera property, injected by the shell.
    ///
    /// A BRIDGE for the reason `api_context::set_stage_field` is one: the previz renderer lives
    /// in `accelerator`, which `core` does not link and must not. `object` is `"camera"`,
    /// `"view_camera"` or `"screen/<name>"`, and `field` is a row's `path` out of
    /// `fields::screen_fields()` / `camera_fields()`.
    ///
    /// Called from the STAGE executor, which the http path was not -- see the write-on-change
    /// note in `apply_stage_overlay`. Without this the timeline can address a previz path (the
    /// address space resolves it) and cannot write it, which is the shape of gap that gets
    /// recorded as "implemented" and is not.
    using stage_field_writer =
        std::function<bool(const std::string& object, const std::string& field, const monitor::vector_t& value)>;
    void set_stage_field_writer(stage_field_writer w);

    /// The HOUSE TIMECODE, as an absolute frame at the channel's rate, or nothing when there is
    /// no valid signal. Injected by the shell.
    ///
    /// A BRIDGE for the reason the previz writer is one: `LTCInput` lives in `modules/ltc` and
    /// `core` does not link the modules -- that inversion is exactly what forced the ten
    /// `shared_ptr<void>` virtuals the `KEYFRAMES` module needed, and they are gone.
    ///
    /// Asked once per tick per chasing document. A document that is not chasing never calls it,
    /// so a server with no LTC pays nothing.
    using timecode_source = std::function<std::optional<std::uint32_t>(int fps)>;
    void set_timecode_source(timecode_source src);

    /// Configure timecode chase for a document. False if it is not one this channel owns.
    std::future<bool> timeline_chase(const std::string& name, const timeline::chase_config& cfg);

    /// The chase settings as they stand, so a caller can change ONE of them.
    ///
    /// Read-modify-write rather than a setter per field: the four settings are set independently
    /// in practice -- an operator adds a hot region without restating the offset -- and four
    /// setters would be four places for the executor hop.
    std::future<timeline::chase_config> timeline_chase_config(const std::string& name);

    /// Queue one transport command for a document, to take effect on the next tick.
    ///
    /// QUEUED rather than applied, and that is what makes several clients acting in the same
    /// frame well-defined: the tick applies a whole frame's worth at once under `stop > pause >
    /// run`, so a stop can never lose to a play that happened to arrive a microsecond later.
    /// Returns false only if the document is not one this channel owns.
    ///
    /// A command carrying `at_frame` is held in the pending list until this channel's counter
    /// reaches that frame -- see `transport::apply_all`, which is where the hold lives so the
    /// rank cannot see a command that is not due.
    std::future<bool> timeline_command(const std::string&                 name,
                                       const timeline::transport_command& cmd) override;

    /// Seek to the start of the next (`forward`) or previous instance in the document.
    ///
    /// Here rather than on `transport` because the transport is pure and takes a position -- it
    /// knows nothing about a document's contents, which is what keeps it testable at boot
    /// against a table of numbers. False if there is no such instance.
    std::future<bool> timeline_seek_relative(const std::string& name, bool forward);

    /// A document's playhead as this channel sees it: state, position in seconds, rate.
    struct timeline_status
    {
        bool        exists   = false;
        std::string state    = "stopped";
        double      position = 0.0;
        double      rate     = 1.0;
        bool        ok       = true;   //< does the document resolve
        std::size_t faults   = 0;
        std::size_t instances = 0;
    };
    std::future<timeline_status> timeline_state(const std::string& name);

    /// WHO IS DRIVING `path` on `layer`, and everyone else who wanted it.
    ///
    /// `first` is the effective owner -- the strongest rank writing that path -- and `second` is
    /// every rank that wanted it, strongest first, comma-separated. Both empty means nobody.
    ///
    /// NOT on the executor: the caller is a write path that has to answer now, in the reply. The
    /// same reasoning as `is_bound`, and it reads the same lock.
    std::pair<std::string, std::string> driver_of(int layer, const std::string& path) const override;

    /// Take a path for the operator, above every other rank, until `release_field`.
    ///
    /// `HOLD` is the operator saying "this one is mine now" -- PIXERA's Dominant, and the answer
    /// to "a document is driving the thing I need to fix on air". Returns false if the path does
    /// not resolve.
    std::future<bool> hold_field(int layer, const std::string& path) override;
    std::future<bool> release_field(int layer, const std::string& path) override;

    /// The node graph, attached per layer. See `stage_base` for what each of these answers.
    std::future<attach_result>              attach_graph(int layer, const std::string& name) override;
    std::future<bool>                       detach_graph(int layer) override;
    std::string                             graph_of(int layer) const override;
    std::future<std::vector<param_snapshot>> describe_graph(int layer) override;
    std::future<bool>                       graph_command(const std::string& name,
                                                          const std::string& verb,
                                                          int                layer) override;

    std::future<bool>                       set_node_param(int                      layer,
                                                           const std::string&       path,
                                                           const monitor::vector_t& value,
                                                           const std::string&       label) override;

    /// The documents this channel owns, with their playheads. For `TIMELINE <ch> LIST`.
    std::future<std::vector<std::pair<std::string, timeline_status>>> timeline_list();

    std::unique_lock<std::mutex> get_lock() const;

    core::video_format_desc video_format_desc() const;
    std::future<void>       video_format_desc(const core::video_format_desc& format_desc);

  private:
    struct impl;
    spl::shared_ptr<impl> impl_;
};

/**
 * A stage wrapper, that queues up stage operations until release() is called.
 * This is useful for batching commands.
 */
class stage_delayed final : public stage_base
{
  public:
    stage_delayed(const std::shared_ptr<stage>& st, int index);

    int64_t count_queued() const { return executor_.size(); }
    void    release() { waiter_.set_value(); }
    void    abort() { executor_.clear(); }
    void    wait() { executor_.stop_and_wait(); }

    std::future<void>            apply_transforms(const std::vector<transform_tuple_t>& transforms) override;
    std::future<void>            apply_transform(int                     index,
                                                 const transform_func_t& transform,
                                                 unsigned int            mix_duration,
                                                 const tweener&          tween) override;
    std::future<void>            clear_transforms(int index) override;
    std::future<void>            clear_transforms() override;
    std::future<frame_transform> get_current_transform(int index) override;
    std::future<void>            load(int                                    index,
                                      const spl::shared_ptr<frame_producer>& producer,
                                      bool                                   preview   = false,
                                      bool                                   auto_play = false) override;
    std::future<void>            preview(int index) override;
    std::future<void>            pause(int index) override;
    std::future<void>            resume(int index) override;
    std::future<void>            play(int index) override;
    std::future<void>            stop(int index) override;
    std::future<std::wstring>    call(int index, const std::vector<std::wstring>& params) override;
    std::future<std::wstring>    callbg(int index, const std::vector<std::wstring>& params) override;
    std::future<void>            clear(int index) override;
    std::future<void>            clear() override;
    std::future<void>            swap_layers(const std::shared_ptr<stage_base>& other, bool swap_transforms) override;
    std::future<void>            swap_layer(int index, int other_index, bool swap_transforms) override;
    std::future<void>
    swap_layer(int index, int other_index, const std::shared_ptr<stage_base>& other, bool swap_transforms) override;

    // Properties

    std::future<std::shared_ptr<frame_producer>> foreground(int index) override;
    std::future<std::shared_ptr<frame_producer>> background(int index) override;

    std::future<void>            execute(std::function<void()> k) override;
    std::future<bool>            timeline_command(const std::string&                 name,
                                                  const timeline::transport_command& cmd) override;

    /// FORWARDED, and it has to be. `stage_base`'s default returns `false` without doing
    /// anything, so a node-parameter write inside a batch would be accepted by the route,
    /// counted by the batch, and then silently do nothing -- which is exactly the shape the
    /// timeline's own delayed-stage defect took, and that one cost two red checks to find.
    std::future<bool>            set_node_param(int                      layer,
                                                const std::string&       path,
                                                const monitor::vector_t& value,
                                                const std::string&       label) override;

    /// The same for a graph VERB -- attach, detach, undo, redo -- so `{"op": "graph"}` lands on
    /// the batch's frame beside the field writes rather than whenever HTTP got to it.
    std::future<bool>            graph_command(const std::string& name,
                                               const std::string& verb,
                                               int                layer) override;


    std::unique_lock<std::mutex> get_lock() const { return stage_->get_lock(); }

  private:
    std::promise<void>     waiter_;
    // BY VALUE, and it has to be. This was a reference to the caller's `shared_ptr`, which
    // is safe only while that particular variable outlives the batch -- true for
    // `AMCPCommandQueue`, which passes a channel's own member, and false for any caller
    // holding the pointer in a local. It failed as `resource_deadlock_would_occur` from
    // `get_lock()` on a two-channel batch: the dangling reference read whatever was left on
    // the stack, both delayed stages resolved to the same `stage`, and the second lock hit
    // a mutex this thread already held. A reference member that only works for one caller
    // is a trap for the second; the cost of owning it is one atomic increment per batch.
    std::shared_ptr<stage> stage_;
    executor               executor_;
};

}} // namespace caspar::core
