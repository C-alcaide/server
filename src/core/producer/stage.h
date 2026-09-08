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
    virtual std::future<void>                  set_keyframe_data(int layer, std::shared_ptr<void> data)                      = 0;
    virtual std::future<bool>                  arm_keyframes(int layer)                                                     = 0;
    virtual std::future<void>                  disarm_keyframes(int layer)                                                  = 0;
    virtual std::future<void>                  clear_keyframes(int layer)                                                   = 0;
    virtual std::future<std::shared_ptr<void>> get_keyframe_data(int layer)                                                 = 0;
    virtual std::future<bool>                  has_keyframe_data(int layer)                                                 = 0;
    virtual std::future<bool>                  is_keyframes_armed(int layer)                                                = 0;
    virtual std::future<bool>                  patch_keyframe(int layer, double time_secs, std::shared_ptr<void> patch_data) = 0;
    virtual std::future<void>                  set_media_time_override(int layer, double time_secs)                         = 0;
    virtual std::future<std::shared_ptr<void>> get_keyframe_status(int layer)                                               = 0;

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

    // Keyframe management
    std::future<void>                  set_keyframe_data(int layer, std::shared_ptr<void> data) override;
    std::future<bool>                  arm_keyframes(int layer) override;
    std::future<void>                  disarm_keyframes(int layer) override;
    std::future<void>                  clear_keyframes(int layer) override;
    std::future<std::shared_ptr<void>> get_keyframe_data(int layer) override;
    std::future<bool>                  has_keyframe_data(int layer) override;
    std::future<bool>                  is_keyframes_armed(int layer) override;
    std::future<bool>                  patch_keyframe(int layer, double time_secs, std::shared_ptr<void> patch_data) override;
    std::future<void>                  set_media_time_override(int layer, double time_secs) override;
    std::future<std::shared_ptr<void>> get_keyframe_status(int layer) override;

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

    // Keyframe management
    std::future<void>                  set_keyframe_data(int layer, std::shared_ptr<void> data) override;
    std::future<bool>                  arm_keyframes(int layer) override;
    std::future<void>                  disarm_keyframes(int layer) override;
    std::future<void>                  clear_keyframes(int layer) override;
    std::future<std::shared_ptr<void>> get_keyframe_data(int layer) override;
    std::future<bool>                  has_keyframe_data(int layer) override;
    std::future<bool>                  is_keyframes_armed(int layer) override;
    std::future<bool>                  patch_keyframe(int layer, double time_secs, std::shared_ptr<void> patch_data) override;
    std::future<void>                  set_media_time_override(int layer, double time_secs) override;
    std::future<std::shared_ptr<void>> get_keyframe_status(int layer) override;

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
