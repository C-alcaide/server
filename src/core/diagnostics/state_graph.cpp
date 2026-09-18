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
 */

#include "state_graph.h"

#include <common/diagnostics/graph.h>
#include <common/log.h>
#include <common/memory.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace caspar { namespace core { namespace diagnostics { namespace publish {

namespace {

std::atomic<bool> g_enabled{false};

/// One sink per graph, exactly as the OSD and log sinks are: the SPI calls the factory once
/// per graph, so the sink IS the graph's identity and needs no keying.
class state_sink : public caspar::diagnostics::spi::graph_sink
{
    struct value
    {
        double last = 0.0;
        double max  = 0.0;
    };
    struct tag
    {
        std::int64_t count = 0;
        const char*  severity = "info";
    };

    const int                     id_;
    mutable std::mutex            mutex_;
    std::wstring                  text_;
    std::map<std::string, value>  values_;
    std::map<std::string, tag>    tags_;
    std::map<std::string, int>    colors_;

  public:
    explicit state_sink(int id)
        : id_(id)
    {
    }

    void activate() override {}

    void set_text(const std::wstring& value) override
    {
        std::lock_guard<std::mutex> lock(mutex_);
        text_ = value;
    }

    void set_value(const std::string& name, double v) override
    {
        // The frame path. One relaxed load and out when nobody is listening -- this is what
        // makes registering the sink unconditionally free.
        if (!g_enabled.load(std::memory_order_relaxed))
            return;
        std::lock_guard<std::mutex> lock(mutex_);
        auto&                       e = values_[name];
        e.last                        = v;
        if (v > e.max)
            e.max = v;
    }

    // Colour is set once, at construction, by every graph that sets it: kept whether or not
    // publication is on, so a client that enables it later still gets the palette.
    void set_color(const std::string& name, int color) override
    {
        std::lock_guard<std::mutex> lock(mutex_);
        colors_[name] = color;
    }

    void set_tag(caspar::diagnostics::tag_severity severity, const std::string& name) override
    {
        if (severity == caspar::diagnostics::tag_severity::SILENT)
            return;
        if (!g_enabled.load(std::memory_order_relaxed))
            return;
        std::lock_guard<std::mutex> lock(mutex_);
        auto&                       t = tags_[name];
        ++t.count;
        t.severity = severity == caspar::diagnostics::tag_severity::WARNING ? "warning" : "info";
    }

    void auto_reset() override {}

    /// Fold this graph into `out` under `{id}/` -- relative to the root, as a channel's
    /// snapshot is relative to `channel/{n}` -- and start the next interval: `max`
    /// collapses to `last`, counts persist. Called on the publication thread.
    void fold(monitor::state& out)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto                        node = out[id_];
        node["text"]                     = text_;
        for (auto& [name, v] : values_) {
            node[name] = monitor::vector_t{v.last, v.max};
            v.max      = v.last;
        }
        for (const auto& [name, t] : tags_)
            node["tag"][name] = monitor::vector_t{t.count, std::string(t.severity)};
        for (const auto& [name, c] : colors_)
            node["color"][name] = static_cast<std::int32_t>(c);
    }

    /// Forget what was counted, so a client that re-enables sees counters from zero
    /// (DECISIONS L76: drop counters are resettable). Values keep their last level.
    void reset_counts()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        tags_.clear();
        for (auto& [name, v] : values_)
            v.max = v.last;
    }
};

class registry
{
    std::mutex                             mutex_;
    std::vector<std::weak_ptr<state_sink>> sinks_;
    int                                    next_id_ = 1;

    publish_fn_t publish_;
    double       interval_ = 0.1;

    std::thread             thread_;
    std::condition_variable wake_;
    std::mutex              wake_mutex_;
    bool                    stop_ = false;
    bool                    was_enabled_ = false;

  public:
    ~registry()
    {
        {
            std::lock_guard<std::mutex> lock(wake_mutex_);
            stop_ = true;
        }
        wake_.notify_all();
        if (thread_.joinable())
            thread_.join();
    }

    void start(double interval, publish_fn_t publish)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        interval_ = interval;
        publish_  = std::move(publish);
        if (!thread_.joinable())
            thread_ = std::thread([this] { run(); });
    }

    double interval()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return interval_;
    }

    spl::shared_ptr<caspar::diagnostics::spi::graph_sink> make_sink()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto                        sink = std::make_shared<state_sink>(next_id_++);
        sinks_.push_back(sink);
        return spl::make_shared_ptr(std::shared_ptr<caspar::diagnostics::spi::graph_sink>(sink));
    }

    void poke() { wake_.notify_all(); }

  private:
    void run()
    {
        while (true) {
            {
                std::unique_lock<std::mutex> lock(wake_mutex_);
                const auto interval = std::chrono::duration<double>(this->interval());
                wake_.wait_for(lock, std::chrono::duration_cast<std::chrono::milliseconds>(interval));
                if (stop_)
                    return;
            }
            try {
                tick();
            } catch (const std::exception& e) {
                CASPAR_LOG(warning) << L"[diag] publication failed: " << e.what();
            }
        }
    }

    void tick()
    {
        const bool   enabled = g_enabled.load(std::memory_order_acquire);
        publish_fn_t publish;
        std::vector<std::shared_ptr<state_sink>> live;
        double                                   interval;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            publish  = publish_;
            interval = interval_;
            // Prune the graphs that died since the last fold: a sink lives exactly as long as
            // its graph, so a producer that stopped takes its sub-tree with it and an events
            // subscriber sees the leaves reverted.
            for (auto it = sinks_.begin(); it != sinks_.end();) {
                if (auto s = it->lock()) {
                    live.push_back(std::move(s));
                    ++it;
                } else {
                    it = sinks_.erase(it);
                }
            }
        }
        if (!publish)
            return;

        if (!enabled) {
            // One empty tree on the way off, so the sub-tree vanishes rather than freezing at
            // its last values; then nothing until re-enabled.
            if (was_enabled_) {
                was_enabled_ = false;
                for (auto& s : live)
                    s->reset_counts();
                publish(std::make_shared<const monitor::state>());
            }
            return;
        }
        was_enabled_ = true;

        auto out            = std::make_shared<monitor::state>();
        (*out)["interval"] = interval;
        for (auto& s : live)
            s->fold(*out);
        publish(std::move(out));
    }
};

registry& the_registry()
{
    static registry r;
    return r;
}

} // namespace

void register_sink(double interval_seconds, publish_fn_t publish)
{
    the_registry().start(interval_seconds, std::move(publish));
    caspar::diagnostics::spi::register_sink_factory([] { return the_registry().make_sink(); });
    CASPAR_LOG(info) << L"[diag] diagnostics publication registered, every " << interval_seconds << L" s, "
                     << (g_enabled.load() ? L"ON" : L"OFF (DIAG PUBLISH ON, or <publish-diagnostics>true)");
}

void set_enabled(bool enabled)
{
    const bool before = g_enabled.exchange(enabled, std::memory_order_acq_rel);
    if (before != enabled) {
        CASPAR_LOG(info) << L"[diag] diagnostics publication " << (enabled ? L"ON" : L"OFF");
        the_registry().poke();
    }
}

bool enabled() { return g_enabled.load(std::memory_order_acquire); }

double interval() { return the_registry().interval(); }

}}}} // namespace caspar::core::diagnostics::publish
