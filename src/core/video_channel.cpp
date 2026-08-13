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

#include "StdAfx.h"

#include "common/os/thread.h"
#include "video_channel.h"

#include <chrono>
#include <sstream>

#include "video_format.h"

#include "consumer/channel_info.h"
#include "consumer/output.h"
#include "frame/draw_frame.h"
#include "frame/frame.h"
#include "frame/frame_factory.h"
#include "mixer/mixer.h"
#include "producer/stage.h"

#include <common/diagnostics/graph.h>
#include <common/executor.h>
#include <common/timer.h>

#include <core/diagnostics/call_context.h>
#include <core/mixer/image/image_mixer.h>

#include <chrono>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>

namespace caspar { namespace core {

bool operator<(const route_id& a, const route_id& b)
{
    return std::tie(a.index, a.mode, a.raw) < std::tie(b.index, b.mode, b.raw);
}

struct video_channel::impl final
{
    monitor::state state_;

    const channel_info channel_info_;

    const spl::shared_ptr<caspar::diagnostics::graph> graph_ = [](int index) {
        core::diagnostics::scoped_call_context save;
        core::diagnostics::call_context::for_thread().video_channel = index;
        return spl::make_shared<caspar::diagnostics::graph>();
    }(channel_info_.index);

    caspar::core::output         output_;
    spl::shared_ptr<image_mixer> image_mixer_;
    caspar::core::mixer          mixer_;
    std::shared_ptr<core::stage> stage_;

    uint64_t frame_counter_ = 0;
    bool     had_consumers_ = false;

    std::chrono::steady_clock::time_point last_fps_update_ = std::chrono::steady_clock::now();
    int                                   frames_since_update_ = 0;
    double                                current_fps_ = 0.0;

    // ── Tick decomposition ────────────────────────────────────────────────
    // The tick below already times produce / mix / consume / osc, but only into
    // the diagnostics graph, which is a picture. So when a channel misses its
    // rate there was no way to ask *which phase* took the time without attaching
    // a debugger — and the honest answer to "why is this channel at half rate"
    // was a guess. These are the same measurements, published under "tick" in
    // channel state so they can be read over AMCP and asserted on.
    //
    // Accumulated every tick, summarised once a second, and re-emitted every
    // tick (state_ is rebuilt from scratch, so a value written only on the
    // summary tick would be visible in one tick out of 25).
    struct phase_stats
    {
        double sum_ms  = 0.0;
        double peak_ms = 0.0;

        void add(double ms)
        {
            sum_ms += ms;
            peak_ms = std::max(peak_ms, ms);
        }
        void reset() { *this = phase_stats{}; }
    };
    struct tick_phases
    {
        phase_stats produce, mix, consume, osc, total;
        uint64_t    ticks = 0;

        void reset()
        {
            produce.reset();
            mix.reset();
            consume.reset();
            osc.reset();
            total.reset();
            ticks = 0;
        }
    };
    tick_phases tick_window_;
    tick_phases tick_published_;
    bool        tick_published_valid_ = false;

    std::function<void(core::monitor::state)> tick_;

    std::map<route_id, std::weak_ptr<core::route>> routes_;
    std::mutex                                     routes_mutex_;

    std::atomic<bool> abort_request_{false};
    std::thread       thread_;

    std::function<void(int, const layer_frame&)> routesCb = [&](int layer, const layer_frame& layer_frame) {
        std::lock_guard<std::mutex> lock(routes_mutex_);
        for (auto& r : routes_) {
            // if this layer is the source for this route, push the frame to the route producers
            if (layer == r.first.index) {
                auto route = r.second.lock();
                if (!route)
                    continue;

                if (r.first.index == -1) {
                    route->signal(layer_frame.foreground1, layer_frame.foreground2);
                } else if (r.first.raw) {
                    // PREMIX route: deliver raw producer output before any mixer transforms
                    route->signal(layer_frame.foreground1_raw, layer_frame.foreground2_raw);
                } else if (r.first.mode == route_mode::background ||
                           (r.first.mode == route_mode::next && layer_frame.has_background)) {
                    route->signal(draw_frame::pop(layer_frame.background1), draw_frame::pop(layer_frame.background2));
                } else {
                    route->signal(draw_frame::pop(layer_frame.foreground1), draw_frame::pop(layer_frame.foreground2));
                }
            }
        }
    };

  public:
    impl(int                                       index,
         const core::video_format_desc&            format_desc,
         color_space                               default_color_space,
         std::unique_ptr<image_mixer>              image_mixer,
         std::function<void(core::monitor::state)> tick,
         color_transfer                            default_color_transfer = color_transfer::sdr,
         bool                                      auto_color_convert     = true,
         int                                       auto_tone_map          = 0,
         float                                     display_peak_luminance = 1000.0f,
         float                                     sdr_reference_white    = 100.0f,
         bool                                      auto_gamut_compress    = false,
         bool                                      straight_alpha_grading = false,
         bool                                      working_space_composite = false)
        : channel_info_(index, image_mixer->depth(), default_color_space, default_color_transfer, image_mixer->is_vulkan(), image_mixer->native_gl_context(), auto_color_convert, image_mixer->native_egl_display())
        , output_(graph_, format_desc, channel_info_)
        , image_mixer_(std::move(image_mixer))
        , mixer_(index, graph_, image_mixer_, default_color_space, default_color_transfer, auto_color_convert, auto_tone_map, display_peak_luminance, sdr_reference_white, auto_gamut_compress, straight_alpha_grading, working_space_composite)
        , stage_(std::make_shared<core::stage>(index, graph_, format_desc))
        , tick_(std::move(tick))
    {
        graph_->set_color("produce-time", caspar::diagnostics::color(0.0f, 1.0f, 0.0f));
        graph_->set_color("mix-time", caspar::diagnostics::color(1.0f, 0.0f, 0.9f, 0.8f));
        graph_->set_color("consume-time", caspar::diagnostics::color(1.0f, 0.4f, 0.0f, 0.8f));
        graph_->set_color("frame-time", caspar::diagnostics::color(1.0f, 0.4f, 0.4f, 0.8f));
        graph_->set_color("osc-time", caspar::diagnostics::color(0.3f, 0.4f, 0.0f, 0.8f));
        graph_->set_text(print_graph());
        caspar::diagnostics::register_graph(graph_);

        CASPAR_LOG(info) << print() << " Successfully Initialized.";

        thread_ = std::thread([=] {
            set_thread_realtime_priority();
            set_thread_name(L"channel-" + std::to_wstring(channel_info_.index));

            while (!abort_request_) {
                // Started outside the try because the catch needs it: the loop's
                // pacing comes from the consumers, and a phase that throws before
                // output_() is reached skips that wait entirely. See the catch.
                caspar::timer tick_timer;
                try {
                    frames_since_update_++;
                    auto   now          = std::chrono::steady_clock::now();
                    double duration_sec = std::chrono::duration_cast<std::chrono::duration<double>>(now - last_fps_update_).count();

                    if (duration_sec >= 1.0) {
                        current_fps_         = frames_since_update_ / duration_sec;
                        frames_since_update_ = 0;
                        last_fps_update_     = now;
                    }

                    graph_->set_text(print_graph());

                    frame_counter_ += 1;

                    caspar::timer frame_timer;

                    // Determine all layers that need a frame from the background producer
                    std::vector<int> background_routes = {};
                    {
                        std::lock_guard<std::mutex> lock(routes_mutex_);

                        for (auto& r : routes_) {
                            // Ensure pointer is still valid
                            if (!r.second.lock())
                                continue;

                            if (r.first.mode != route_mode::foreground) {
                                background_routes.push_back(r.first.index);
                            }
                        }
                    }

                    // Produce
                    caspar::timer produce_timer;
                    auto          stage_frames = (*stage_)(frame_counter_, background_routes, routesCb);
                    const auto produce_elapsed = produce_timer.elapsed();
                    graph_->set_value("produce-time", produce_elapsed * format_desc.hz * 0.5);

                    // This is a little race prone, but at worst a new consumer will start with a frame of black
                    bool has_consumers = output_.consumer_count() > 0;

                    // When consumers reappear after an idle period, flush the
                    // mixer's 1-frame deferred buffer so the new consumer gets a
                    // freshly rendered frame instead of a stale one left over
                    // from the previous producer.
                    if (has_consumers && !had_consumers_) {
                        mixer_.flush();
                    }
                    had_consumers_ = has_consumers;

                    // Tell the mixer whether any consumer needs CPU pixel data.
                    // When false, the VK mixer can skip the GPU→CPU readback entirely.
                    image_mixer_->set_cpu_readback_needed(
                        has_consumers && output_.any_consumer_needs_cpu_data());

                    // Tell the mixer which views the consumers want, BEFORE mixing.
                    //
                    // Asked every tick rather than cached: consumers come and go, and the
                    // set is what decides how many post-composite passes run. The mixer's
                    // still-frame cache compares the count for the same reason -- a
                    // consumer attaching changes nothing about the layers.
                    image_mixer_->set_consumer_views(output_.distinct_consumer_views());

                    // Mix
                    caspar::timer      mix_timer;
                    mixer::output_frames mixed_frame =
                        has_consumers ? mixer_(stage_frames.frames, stage_frames.format_desc, stage_frames.nb_samples)
                                      : mixer::output_frames{};
                    mixer::output_frames mixed_frame2 =
                        has_consumers && stage_frames.format_desc.field_count == 2
                            ? mixer_(stage_frames.frames2, stage_frames.format_desc, stage_frames.nb_samples)
                            : mixer::output_frames{};
                    const auto mix_elapsed = mix_timer.elapsed();
                    graph_->set_value("mix-time", mix_elapsed * format_desc.hz * 0.5);

                    // Consume
                    caspar::timer consume_timer;
                    output_(mixed_frame.primary, mixed_frame2.primary, stage_frames.format_desc,
                            mixed_frame.views, mixed_frame2.views);
                    auto consume_elapsed = consume_timer.elapsed();
                    graph_->set_value("consume-time", consume_elapsed * stage_frames.format_desc.hz * 0.5);

                    const auto frame_elapsed = frame_timer.elapsed();
                    graph_->set_value("frame-time", frame_elapsed * stage_frames.format_desc.hz * 0.5);

                    // Accumulate the phase timings taken above. osc is folded in
                    // one tick late, because it is only known after this state has
                    // been built and handed to tick_().
                    {
                        auto& w = tick_window_;
                        w.ticks++;
                        // Each of these was captured where its phase ended.
                        // Reading .elapsed() here instead would measure from that
                        // timer's construction to now -- so "produce" would include
                        // mix and consume and come out equal to the whole tick,
                        // which is exactly what the first version of this reported
                        // (19.9 ms of "produce" on an empty channel).
                        w.produce.add(produce_elapsed * 1000.0);
                        w.mix.add(mix_elapsed * 1000.0);
                        w.consume.add(consume_elapsed * 1000.0);
                        w.total.add(frame_elapsed * 1000.0);

                        const auto summary_ticks =
                            static_cast<uint64_t>(std::max(1, static_cast<int>(stage_frames.format_desc.hz)));
                        if (w.ticks >= summary_ticks) {
                            tick_published_       = w;
                            tick_published_valid_ = true;
                            w.reset();
                        }
                    }

                    monitor::state state = {};
                    state["stage"]       = stage_->state();
                    state["mixer"]       = mixer_.state();
                    state["output"]      = output_.state();

                    if (tick_published_valid_ && tick_published_.ticks > 0) {
                        const auto&  p         = tick_published_;
                        const auto   n         = static_cast<double>(p.ticks);
                        const double period_ms = stage_frames.format_desc.hz > 0.0
                                                     ? 1000.0 / stage_frames.format_desc.hz
                                                     : 0.0;

                        const auto emit = [&](const char* name, const phase_stats& s) {
                            state["tick"][name]["avg_ms"]  = s.sum_ms / n;
                            state["tick"][name]["peak_ms"] = s.peak_ms;
                            state["tick"][name]["percent"] =
                                period_ms > 0.0 ? (s.sum_ms / n) * 100.0 / period_ms : 0.0;
                        };
                        emit("produce", p.produce);
                        emit("mix", p.mix);
                        emit("consume", p.consume);
                        emit("osc", p.osc);
                        emit("total", p.total);

                        // What produce+mix+consume+osc does not account for: waiting
                        // on the tick's own pacing, plus anything else in the loop.
                        // A large unaccounted share means the channel is idle-waiting
                        // (healthy) or blocked somewhere not timed here (not).
                        const double accounted =
                            (p.produce.sum_ms + p.mix.sum_ms + p.consume.sum_ms + p.osc.sum_ms) / n;
                        state["tick"]["unaccounted"]["avg_ms"] = std::max(0.0, p.total.sum_ms / n - accounted);
                        state["tick"]["nominal_ms"]            = period_ms;
                    }
                    state["framerate"]   = {stage_frames.format_desc.framerate.numerator() *
                                                stage_frames.format_desc.field_count,
                                            stage_frames.format_desc.framerate.denominator()};
                    state["format"]      = stage_frames.format_desc.name;
                    state_               = state;

                    caspar::timer osc_timer;
                    tick_(state_);
                    const auto osc_elapsed = osc_timer.elapsed();
                    graph_->set_value("osc-time", osc_elapsed * stage_frames.format_desc.hz * 0.5);
                    // Folded into the window a tick late, by construction: it is not
                    // known until after the state above was published.
                    tick_window_.osc.add(osc_elapsed * 1000.0);
                } catch (...) {
                    CASPAR_LOG_CURRENT_EXCEPTION();

                    // Nothing paces this loop except the consumers. output_() is what
                    // blocks on the DeckLink clock (or on a file consumer's queue), and
                    // it is the last phase — so an exception thrown in produce or mix
                    // returns straight to the top of the while and the channel free-runs
                    // as fast as the GPU will take work.
                    //
                    // Measured, Vulkan mixer, 1080p25, one still-image layer whose rgb24
                    // upload throws in the mixer every frame: 28,997 exceptions in 6.0 s
                    // = ~4,800 ticks/s against a nominal 25. That turned one unsupported
                    // pixel format into a pegged CPU core, ~5,000 log lines a second, and
                    // 90x the compositing the channel consumes -- all of which read as
                    // separate faults. Sleeping out the remainder of the frame period
                    // bounds a persistent failure to one occurrence per frame.
                    //
                    // Only the REMAINDER, so a tick that already overran is not delayed
                    // further, and a transient exception costs nothing.
                    const auto hz = stage_->video_format_desc().hz;
                    if (hz > 0.0) {
                        const auto remaining = 1.0 / hz - tick_timer.elapsed();
                        if (remaining > 0.0) {
                            std::this_thread::sleep_for(std::chrono::duration<double>(remaining));
                        }
                    }
                }
            }
        });
    }

    ~impl()
    {
        CASPAR_LOG(info) << print() << " Uninitializing.";
        abort_request_ = true;
        thread_.join();
    }

    std::shared_ptr<core::route> route(int index = -1, route_mode mode = route_mode::foreground, bool raw = false)
    {
        std::lock_guard<std::mutex> lock(routes_mutex_);

        route_id id = {};
        id.index    = index;
        id.mode     = mode;
        id.raw      = raw;

        auto route = routes_[id].lock();
        if (!route) {
            route              = std::make_shared<core::route>();
            route->format_desc = stage_->video_format_desc(); // TODO this needs updating whenever the videomode changes
            route->name        = std::to_wstring(channel_info_.index);
            if (index != -1) {
                route->name += L"/" + std::to_wstring(index);
            }
            if (mode == route_mode::background) {
                route->name += L"/background";
            } else if (mode == route_mode::next) {
                route->name += L"/next";
            }
            if (raw) {
                route->name += L"/premix";
            }
            routes_[id] = route;
        }

        return route;
    }

    std::wstring print() const
    {
        return L"video_channel[" + std::to_wstring(channel_info_.index) + L"|" + stage_->video_format_desc().name +
               L"]";
    }

    std::wstring print_graph() const
    {
        std::wstringstream stats;
        stats.precision(2);
        stats << std::fixed;
        stats << print() << L" fps: " << current_fps_;
        return stats.str();
    }

    int index() const { return channel_info_.index; }

    channel_info get_consumer_channel_info() const { return channel_info_; }
};

video_channel::video_channel(int                                       index,
                             const core::video_format_desc&            format_desc,
                             color_space                               default_color_space,
                             std::unique_ptr<image_mixer>              image_mixer,
                             std::function<void(core::monitor::state)> tick,
                             color_transfer                            default_color_transfer,
                             bool                                      auto_color_convert,
                             int                                       auto_tone_map,
                             float                                     display_peak_luminance,
                             float                                     sdr_reference_white,
                             bool                                      auto_gamut_compress,
                             bool                                      straight_alpha_grading,
                             bool                                      working_space_composite)
    : impl_(new impl(index, format_desc, default_color_space, std::move(image_mixer), std::move(tick), default_color_transfer, auto_color_convert, auto_tone_map, display_peak_luminance, sdr_reference_white, auto_gamut_compress, straight_alpha_grading, working_space_composite))
{
}
video_channel::~video_channel() {}
const std::shared_ptr<core::stage>& video_channel::stage() const { return impl_->stage_; }
std::shared_ptr<core::stage>&       video_channel::stage() { return impl_->stage_; }
const mixer&                        video_channel::mixer() const { return impl_->mixer_; }
mixer&                              video_channel::mixer() { return impl_->mixer_; }
const output&                       video_channel::output() const { return impl_->output_; }
output&                             video_channel::output() { return impl_->output_; }
spl::shared_ptr<frame_factory>      video_channel::frame_factory() { return impl_->image_mixer_; }
int                                 video_channel::index() const { return impl_->index(); }
channel_info         video_channel::get_consumer_channel_info() const { return impl_->get_consumer_channel_info(); };
core::monitor::state video_channel::state() const { return impl_->state_; }

std::shared_ptr<route> video_channel::route(int index, route_mode mode, bool raw) { return impl_->route(index, mode, raw); }

}} // namespace caspar::core
