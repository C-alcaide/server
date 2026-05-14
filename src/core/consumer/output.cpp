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
#include "output.h"

#include "channel_info.h"
#include "frame_consumer.h"

#include "../frame/frame.h"
#include "../frame/pixel_format.h"

#include <common/bit_depth.h>
#include <common/diagnostics/graph.h>
#include <common/except.h>
#include <common/memory.h>

#include <atomic>
#include <chrono>
#include <iomanip>
#include <map>
#include <optional>
#include <thread>
#include <utility>

namespace caspar { namespace core {

using time_point_t = decltype(std::chrono::high_resolution_clock::now());

struct output::impl
{
    monitor::state                      state_;
    spl::shared_ptr<diagnostics::graph> graph_;
    const channel_info                  channel_info_;
    video_format_desc                   format_desc_;

    std::mutex                                     consumers_mutex_;
    std::map<int, spl::shared_ptr<frame_consumer>> consumers_;

    std::atomic<uint64_t>      tick_count_{0};

    std::optional<time_point_t> time_;

    // Channel-level periodic timing diagnostic
    std::chrono::steady_clock::time_point timing_start_{std::chrono::steady_clock::now()};
    std::chrono::steady_clock::time_point timing_last_frame_{};
    uint64_t timing_frames_ = 0;
    uint64_t timing_late_   = 0;

  public:
    impl(const spl::shared_ptr<diagnostics::graph>& graph,
         const video_format_desc&                   format_desc,
         const core::channel_info&                  channel_info)
        : graph_(graph)
        , channel_info_(channel_info)
        , format_desc_(format_desc)
    {
    }

    void add(int index, spl::shared_ptr<frame_consumer> consumer)
    {
        // Extract old consumer without destroying it under the lock.
        std::shared_ptr<frame_consumer> old;
        {
            std::lock_guard<std::mutex> lock(consumers_mutex_);
            auto it = consumers_.find(index);
            if (it != consumers_.end()) {
                old = static_cast<std::shared_ptr<frame_consumer>>(it->second);
                consumers_.erase(it);
            }
        }

        if (old) {
            // Wait for the tick loop to finish its current iteration so it
            // drops its shared_ptr copy of the old consumer.  At 25 fps a
            // tick takes ~40 ms; we wait up to 200 ms (5 frames).
            auto pre = tick_count_.load();
            for (int i = 0; i < 200 && tick_count_.load() == pre; ++i) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            // Destroy old consumer — joins its GL/render thread.
            old.reset();
        }

        consumer->initialize(format_desc_, channel_info_, index);

        std::lock_guard<std::mutex> lock(consumers_mutex_);
        consumers_.emplace(index, std::move(consumer));
    }

    void add(const spl::shared_ptr<frame_consumer>& consumer) { add(consumer->index(), consumer); }

    bool remove(int index)
    {
        std::shared_ptr<frame_consumer> old;
        {
            std::lock_guard<std::mutex> lock(consumers_mutex_);
            auto it = consumers_.find(index);
            if (it == consumers_.end())
                return false;
            old = static_cast<std::shared_ptr<frame_consumer>>(it->second);
            consumers_.erase(it);
        }

        // Wait for the tick loop to finish its current iteration so it
        // drops its shared_ptr snapshot of the old consumer.  Without this,
        // the old consumer may be destroyed later (when the tick snapshot
        // goes out of scope) and race with a newly-added consumer at the
        // same index.
        auto pre = tick_count_.load();
        for (int i = 0; i < 200 && tick_count_.load() == pre; ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        old.reset();
        return true;
    }

    bool remove(const spl::shared_ptr<frame_consumer>& consumer) { return remove(consumer->index()); }

    std::future<bool> call(int index, const std::vector<std::wstring>& params)
    {
        std::lock_guard<std::mutex> lock(consumers_mutex_);
        auto                        it = consumers_.find(index);
        if (it != consumers_.end()) {
            try {
                return it->second->call(params);
            } catch (...) {
                CASPAR_LOG_CURRENT_EXCEPTION();
            }
        } else {
            CASPAR_LOG(warning) << print() << L" No consumer found for index " << index << L".";
        }
        return caspar::make_ready_future(false);
    }

    size_t consumer_count()
    {
        std::lock_guard<std::mutex> lock(consumers_mutex_);
        return consumers_.size();
    }

    bool any_consumer_needs_cpu_data()
    {
        std::lock_guard<std::mutex> lock(consumers_mutex_);
        for (auto& p : consumers_) {
            if (p.second->needs_cpu_frame_data()) {
                static int log_count = 0;
                if (log_count < 5) {
                    CASPAR_LOG(info) << L"[output] Consumer forcing CPU readback: "
                                    << p.second->name() << L" (index=" << p.first << L")"
                                    << L" needs_cpu=" << p.second->needs_cpu_frame_data();
                    log_count++;
                }
                return true;
            }
        }
        static bool logged_skip = false;
        if (!logged_skip && !consumers_.empty()) {
            CASPAR_LOG(info) << L"[output] No consumer needs CPU readback (" << consumers_.size() << L" consumers)";
            logged_skip = true;
        }
        return false;
    }

    void operator()(const const_frame&             input_frame1,
                    const const_frame&             input_frame2,
                    const core::video_format_desc& format_desc)
    {
        // Channel-level timing diagnostic
        {
            auto now = std::chrono::steady_clock::now();
            timing_frames_++;
            if (timing_last_frame_.time_since_epoch().count() > 0) {
                double frame_ms = std::chrono::duration<double, std::milli>(now - timing_last_frame_).count();
                double expected_ms = 1000.0 / format_desc_.hz;
                if (frame_ms > expected_ms * 1.15)
                    timing_late_++;
            }
            timing_last_frame_ = now;

            auto elapsed = std::chrono::duration<double>(now - timing_start_).count();
            if (elapsed >= 5.0 && timing_frames_ > 1) {
                double avg_ms = elapsed * 1000.0 / timing_frames_;
                CASPAR_LOG(info) << L"[channel " << channel_info_.index << L"] TIMING: avg="
                                 << std::fixed << std::setprecision(1) << avg_ms
                                 << L"ms late=" << timing_late_
                                 << L" frames=" << timing_frames_;
                timing_start_  = now;
                timing_frames_ = 0;
                timing_late_   = 0;
            }
        }

        auto time = std::move(time_);

        if (format_desc_ != format_desc) {
            std::lock_guard<std::mutex> lock(consumers_mutex_);
            for (auto it = consumers_.begin(); it != consumers_.end();) {
                try {
                    it->second->initialize(format_desc, channel_info_, it->first);
                    ++it;
                } catch (...) {
                    CASPAR_LOG_CURRENT_EXCEPTION();
                    it = consumers_.erase(it);
                }
            }
            format_desc_ = format_desc;
            time_        = {};
            return;
        }

        // If no frame is provided, this should only happen when the channel has no consumers.
        // Take a shortcut and perform the sleep to let the channel tick correctly.
        if (!input_frame1) {
            if (!time) {
                time = std::chrono::high_resolution_clock::now();
            } else {
                std::this_thread::sleep_until(*time);
            }
            time_ = *time + std::chrono::microseconds(static_cast<int>(1e6 / format_desc_.hz));
            return;
        }

        const auto bytesPerComponent1 =
            input_frame1.pixel_format_desc().planes.at(0).depth == common::bit_depth::bit8 ? 1 : 2;
        if (input_frame1.size() != format_desc_.size * bytesPerComponent1) {
            CASPAR_LOG(warning) << print() << L" Invalid input frame size.";
            return;
        }

        if (input_frame2) {
            const auto bytesPerComponent2 =
                input_frame2.pixel_format_desc().planes.at(0).depth == common::bit_depth::bit8 ? 1 : 2;

            if (input_frame2.size() != format_desc_.size * bytesPerComponent2) {
                CASPAR_LOG(warning) << print() << L" Invalid input frame size.";
                return;
            }
        }

        decltype(consumers_) consumers;
        {
            std::lock_guard<std::mutex> lock(consumers_mutex_);
            consumers = consumers_;
        }

        auto do_send = [this, &consumers](core::video_field field, const core::const_frame& frame) {
            std::map<int, std::future<bool>> futures;

            for (auto it = consumers.begin(); it != consumers.end();) {
                try {
                    futures.emplace(it->first, it->second->send(field, frame));
                    ++it;
                } catch (...) {
                    CASPAR_LOG_CURRENT_EXCEPTION();
                    auto index  = it->first;
                    auto failed = it->second.get();
                    it          = consumers.erase(it);

                    std::lock_guard<std::mutex> lock(consumers_mutex_);
                    auto mit = consumers_.find(index);
                    if (mit != consumers_.end() && mit->second.get() == failed)
                        consumers_.erase(mit);
                }
            }

            for (auto& p : futures) {
                try {
                    if (!p.second.get()) {
                        auto fit    = consumers.find(p.first);
                        auto failed = fit != consumers.end() ? fit->second.get() : nullptr;
                        consumers.erase(p.first);

                        std::lock_guard<std::mutex> lock(consumers_mutex_);
                        auto mit = consumers_.find(p.first);
                        if (mit != consumers_.end() && mit->second.get() == failed)
                            consumers_.erase(mit);
                    }
                } catch (...) {
                    CASPAR_LOG_CURRENT_EXCEPTION();
                    auto fit    = consumers.find(p.first);
                    auto failed = fit != consumers.end() ? fit->second.get() : nullptr;
                    consumers.erase(p.first);

                    std::lock_guard<std::mutex> lock(consumers_mutex_);
                    auto mit = consumers_.find(p.first);
                    if (mit != consumers_.end() && mit->second.get() == failed)
                        consumers_.erase(mit);
                }
            }
        };

        if (format_desc_.field_count == 2) {
            do_send(core::video_field::a, input_frame1);
            do_send(core::video_field::b, input_frame2);
        } else {
            do_send(core::video_field::progressive, input_frame1);
        }

        monitor::state state;
        for (auto& p : consumers) {
            state["port"][p.first]             = p.second->state();
            state["port"][p.first]["consumer"] = p.second->name();
        }
        state_ = std::move(state);

        tick_count_.fetch_add(1, std::memory_order_release);

        const auto needs_sync = std::all_of(
            consumers.begin(), consumers.end(), [](auto& p) { return !p.second->has_synchronization_clock(); });

        if (needs_sync) {
            if (!time) {
                time = std::chrono::high_resolution_clock::now();
            } else {
                std::this_thread::sleep_until(*time);
            }
            time_ = *time + std::chrono::microseconds(static_cast<int>(1e6 / format_desc_.hz));
        } else {
            time_.reset();
        }
    }

    std::wstring print() const { return L"output[" + std::to_wstring(channel_info_.index) + L"]"; }
};

output::output(const spl::shared_ptr<diagnostics::graph>& graph,
               const video_format_desc&                   format_desc,
               const core::channel_info&                  channel_info)
    : impl_(new impl(graph, format_desc, channel_info))
{
}
output::~output() {}
void output::add(int index, const spl::shared_ptr<frame_consumer>& consumer) { impl_->add(index, consumer); }
void output::add(const spl::shared_ptr<frame_consumer>& consumer) { impl_->add(consumer); }
bool output::remove(int index) { return impl_->remove(index); }
bool output::remove(const spl::shared_ptr<frame_consumer>& consumer) { return impl_->remove(consumer); }
std::future<bool> output::call(int index, const std::vector<std::wstring>& params)
{
    return impl_->call(index, params);
}
size_t output::consumer_count() const { return impl_->consumer_count(); }
bool   output::any_consumer_needs_cpu_data() const { return impl_->any_consumer_needs_cpu_data(); }
void   output::operator()(const const_frame& frame, const const_frame& frame2, const video_format_desc& format_desc)
{
    return (*impl_)(frame, frame2, format_desc);
}
core::monitor::state output::state() const { return impl_->state_; }
}} // namespace caspar::core
