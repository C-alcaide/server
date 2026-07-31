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
#include <cmath>
#include <condition_variable>
#include <iomanip>
#include <map>
#include <optional>
#include <sstream>
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
    std::mutex                 tick_mutex_;
    std::condition_variable    tick_cv_;

    // Which consumer last forced a CPU readback, for transition-only logging.
    // Compared by address only — never dereferenced outside consumers_mutex_.
    const frame_consumer* last_cpu_requester_ = nullptr;

    std::optional<time_point_t> time_;

    // ── Channel timing telemetry ──────────────────────────────────────────
    // The channel is paced by consumer back-pressure (an audio ring buffer, a
    // genlocked SDI card) with a wall-clock sleep as the fallback. That design
    // is defensible, but it was entirely unmeasured: there was no way to tell a
    // healthy channel from one whose clock consumer is fighting another, or one
    // sitting permanently late. These figures are published in state() under
    // "timing" so the 360-client and the test runner can assert on them.
    std::chrono::steady_clock::time_point timing_start_{std::chrono::steady_clock::now()};
    std::chrono::steady_clock::time_point timing_last_frame_{};
    uint64_t timing_frames_ = 0;
    uint64_t timing_late_   = 0;

    // Rolling window, reset with the periodic report.
    double   period_sum_ms_  = 0.0;
    double   period_min_ms_  = 0.0;
    double   period_max_ms_  = 0.0;
    uint64_t period_samples_ = 0;

    // Last published values, so state() does not depend on where the window is.
    double   last_avg_ms_    = 0.0;
    double   last_min_ms_    = 0.0;
    double   last_max_ms_    = 0.0;
    double   last_jitter_ms_ = 0.0;
    uint64_t last_late_      = 0;
    uint64_t total_late_     = 0;
    uint64_t total_frames_   = 0;

    // How long the consumers' send() futures took to settle. With a hardware
    // clock this is the back-pressure itself, i.e. the channel's real pacing
    // signal, and its size relative to the frame period is the headroom left.
    double   last_consume_ms_ = 0.0;
    double   consume_max_ms_  = 0.0;

    // Set when more than one consumer claims to be the channel's clock. They
    // cannot both pace it: whichever blocks longest wins and the other drifts
    // until its buffer absorbs or breaks.
    bool     multi_clock_warned_ = false;
    int      clock_source_count_  = 0;

  public:
    impl(const spl::shared_ptr<diagnostics::graph>& graph,
         const video_format_desc&                   format_desc,
         const core::channel_info&                  channel_info)
        : graph_(graph)
        , channel_info_(channel_info)
        , format_desc_(format_desc)
    {
    }

    /// Blocks until the tick loop has certainly dropped any snapshot it holds
    /// of a consumer removed from consumers_ just now, so the caller's
    /// shared_ptr is the last reference and destruction happens here (on the
    /// calling AMCP thread) instead of on the realtime channel thread.
    ///
    /// operator() increments tick_count_ *before* its local `consumers`
    /// snapshot goes out of scope, so seeing one increment is not enough —
    /// two are needed to prove the snapshot is gone. The previous code slept in
    /// 1 ms steps and waited for a single increment, so it was both quantised
    /// and one epoch short of the guarantee it claimed.
    void wait_for_consumer_snapshot_release()
    {
        const auto target   = tick_count_.load(std::memory_order_acquire) + 2;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(200);

        std::unique_lock<std::mutex> lock(tick_mutex_);
        bool released = tick_cv_.wait_until(
            lock, deadline, [&] { return tick_count_.load(std::memory_order_acquire) >= target; });
        if (!released) {
            // Callers proceed as if the old consumer's snapshot on the tick
            // thread were guaranteed gone (see add()/remove()) — on a timeout
            // that guarantee silently doesn't hold, reintroducing the "consumer
            // destroyed on the realtime channel thread" hazard this exists to
            // prevent. A slow tick (blocking hardware consumer, GC pause, etc.)
            // is the only way to hit this in practice.
            CASPAR_LOG(warning) << L"Timed out waiting for consumer snapshot release after 200ms"
                                   L" — old consumer may be destroyed on the channel tick thread.";
        }
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
            wait_for_consumer_snapshot_release();
            // Destroy old consumer — joins its GL/render thread. Kept
            // synchronous so that REMOVE/ADD on the same hardware device
            // releases it before the replacement tries to claim it.
            old.reset();
        }

        consumer->initialize(format_desc_, channel_info_, index);

        {
            std::lock_guard<std::mutex> lock(consumers_mutex_);
            consumers_.emplace(index, std::move(consumer));
        }

        log_sync_recommendation();
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

        // Wait for the tick loop to drop its shared_ptr snapshot of the old
        // consumer. Without this, the old consumer may be destroyed later (when
        // the tick snapshot goes out of scope) and race with a newly-added
        // consumer at the same index.
        wait_for_consumer_snapshot_release();
        old.reset();
        log_sync_recommendation();
        return true;
    }

    bool remove(const spl::shared_ptr<frame_consumer>& consumer) { return remove(consumer->index()); }

    // ─── A/V Sync Advisor ──────────────────────────────────────────────────
    // Called after a consumer is added or removed. Analyzes pipeline depths
    // of all consumers on the channel and logs a sync recommendation.
    void log_sync_recommendation()
    {
        struct consumer_info {
            std::wstring      name;
            av_pipeline_info  pipeline;
        };

        std::vector<consumer_info> infos;
        {
            std::lock_guard<std::mutex> lock(consumers_mutex_);
            for (auto& p : consumers_) {
                infos.push_back({p.second->name(), p.second->av_pipeline()});
            }
        }

        // Collect consumers by role
        std::vector<consumer_info*> video_only;     // Video without embedded audio (Vulkan)
        std::vector<consumer_info*> audio_only;     // Audio only (PortAudio)
        std::vector<consumer_info*> combined;       // Video + embedded audio (DeckLink)

        for (auto& ci : infos) {
            if (!ci.pipeline.has_video && !ci.pipeline.has_audio)
                continue; // Consumer doesn't participate (screen, FFmpeg, etc.)

            if (ci.pipeline.has_video && ci.pipeline.has_audio && ci.pipeline.audio_is_embedded) {
                combined.push_back(&ci);
            } else if (ci.pipeline.has_video && !ci.pipeline.has_audio) {
                video_only.push_back(&ci);
            } else if (ci.pipeline.has_audio && !ci.pipeline.has_video) {
                audio_only.push_back(&ci);
            } else if (ci.pipeline.has_video) {
                video_only.push_back(&ci);
            }
        }

        // Nothing to advise if we don't have at least one video and one separate audio
        bool has_separate_av  = !video_only.empty() && !audio_only.empty();
        bool has_cross_video  = !combined.empty() && !video_only.empty();

        if (!has_separate_av && !has_cross_video)
            return;

        const double frame_ms = 1000.0 / format_desc_.fps;
        const int    ch_idx   = channel_info_.index;

        // Build consumer list string
        std::wostringstream consumer_list;
        bool first = true;
        for (auto& ci : infos) {
            if (!ci.pipeline.has_video && !ci.pipeline.has_audio)
                continue;
            if (!first) consumer_list << L", ";
            consumer_list << ci.name;
            if (ci.pipeline.has_video && ci.pipeline.has_audio && ci.pipeline.audio_is_embedded)
                consumer_list << L" (video+audio)";
            else if (ci.pipeline.has_video)
                consumer_list << L" (video)";
            else if (ci.pipeline.has_audio)
                consumer_list << L" (audio)";
            first = false;
        }

        CASPAR_LOG(info) << L"";
        CASPAR_LOG(info) << L"===== A/V Sync Recommendation (channel " << ch_idx << L") =====";
        CASPAR_LOG(info) << L"  Consumers: " << consumer_list.str();
        CASPAR_LOG(info) << L"  Frame rate: " << format_desc_.fps << L" fps (" << std::fixed
                         << std::setprecision(1) << frame_ms << L" ms/frame)";

        int section = 1;

        // Section: Combined A/V consumers (DeckLink with embedded audio)
        for (auto* c : combined) {
            CASPAR_LOG(info) << L"  [" << section++ << L"] " << c->name
                             << L" A/V: hardware-synced (embedded audio), no adjustment needed.";
        }

        // Section: Separate video ↔ audio sync
        for (auto* vid : video_only) {
            for (auto* aud : audio_only) {
                double video_ms = vid->pipeline.video_depth_frames * frame_ms + vid->pipeline.video_delay_ms;
                double audio_ms = aud->pipeline.audio_depth_frames * frame_ms + aud->pipeline.audio_device_latency_ms;
                double offset   = audio_ms - video_ms;

                CASPAR_LOG(info) << L"  [" << section++ << L"] " << vid->name << L" + " << aud->name << L" A/V sync:";
                CASPAR_LOG(info) << L"      Audio pipeline: " << std::fixed << std::setprecision(1) << audio_ms
                                 << L" ms  (" << aud->pipeline.audio_depth_frames << L" frames"
                                 << (aud->pipeline.audio_device_latency_ms > 0.0
                                         ? L" + " + std::to_wstring(static_cast<int>(std::round(aud->pipeline.audio_device_latency_ms))) + L" ms device"
                                         : L"")
                                 << L")";
                CASPAR_LOG(info) << L"      Video pipeline: " << std::fixed << std::setprecision(1) << video_ms
                                 << L" ms  (delay=" << (vid->pipeline.video_depth_frames - 1)
                                 << L", delay-ms=" << std::setprecision(1) << vid->pipeline.video_delay_ms << L")";

                if (std::abs(offset) <= 5.0) {
                    CASPAR_LOG(info) << L"      OK - offset " << std::setprecision(1) << offset
                                     << L" ms (within +/-5 ms)";
                } else {
                    CASPAR_LOG(warning) << L"      Video " << (offset > 0 ? L"leads" : L"lags")
                                        << L" audio by " << std::setprecision(1) << std::abs(offset)
                                        << L" ms" << (std::abs(offset) > 40.0 ? L" (outside EBU R37 +/-40 ms)" : L"");

                    if (vid->pipeline.video_delay_adjustable) {
                        // Suggest adjusting the video consumer's delay (Vulkan)
                        int    rec_delay_frames = aud->pipeline.audio_depth_frames - 1;
                        double rec_delay_ms     = aud->pipeline.audio_device_latency_ms;
                        if (rec_delay_frames < 0) rec_delay_frames = 0;
                        if (rec_delay_ms < 0.0)   rec_delay_ms = 0.0;

                        double rec_video_ms = (rec_delay_frames + 1) * frame_ms + rec_delay_ms;
                        double rec_offset   = audio_ms - rec_video_ms;

                        CASPAR_LOG(info) << L"      Suggested " << vid->name << L" config:";
                        CASPAR_LOG(info) << L"        <delay>" << rec_delay_frames << L"</delay>"
                                         << (rec_delay_ms > 0.5 ? L"  <delay-ms>" + std::to_wstring(static_cast<int>(std::round(rec_delay_ms))) + L"</delay-ms>" : L"");
                        CASPAR_LOG(info) << L"        Predicted offset after adjustment: "
                                         << std::setprecision(1) << rec_offset << L" ms";
                    } else if (aud->pipeline.audio_delay_adjustable) {
                        // Adjust audio delay to match video (DeckLink + PortAudio case)
                        int    inherent_audio    = aud->pipeline.audio_depth_frames - aud->pipeline.audio_delay_frames;
                        double inherent_audio_ms = inherent_audio * frame_ms + aud->pipeline.audio_device_latency_ms;
                        double target_audio_ms   = vid->pipeline.video_depth_frames * frame_ms;
                        int    rec_audio_delay   = std::max(0, static_cast<int>(std::round((target_audio_ms - inherent_audio_ms) / frame_ms)));

                        double rec_audio_pipeline = (inherent_audio + rec_audio_delay) * frame_ms
                                                    + aud->pipeline.audio_device_latency_ms;
                        double rec_offset = rec_audio_pipeline - video_ms;

                        if (rec_audio_delay != aud->pipeline.audio_delay_frames) {
                            CASPAR_LOG(info) << L"      Suggested " << aud->name << L" config:";
                            CASPAR_LOG(info) << L"        <delay>" << rec_audio_delay << L"</delay>";
                            CASPAR_LOG(info) << L"        Predicted offset after adjustment: "
                                             << std::setprecision(1) << rec_offset << L" ms";
                        } else if (offset > 0) {
                            CASPAR_LOG(info) << L"      Audio pipeline is inherently deeper than "
                                             << vid->name << L" video. Consider reducing buffer-frames if stable.";
                        } else {
                            CASPAR_LOG(info) << L"      Video pipeline is inherently deeper than "
                                             << aud->name << L". Cannot fully compensate with audio delay.";
                        }
                    } else {
                        CASPAR_LOG(info) << L"      Neither consumer supports delay adjustment.";
                        CASPAR_LOG(info) << L"      Consider adjusting buffer-depth settings.";
                    }
                }
            }
        }

        // Section: Cross-output video alignment (combined vs video-only)
        for (auto* emb : combined) {
            for (auto* vid : video_only) {
                int    emb_depth  = emb->pipeline.video_depth_frames;
                int    vid_depth  = vid->pipeline.video_depth_frames;
                int    diff       = emb_depth - vid_depth;
                double diff_ms   = diff * frame_ms - vid->pipeline.video_delay_ms;

                if (std::abs(diff_ms) <= frame_ms * 0.25)
                    continue; // Close enough

                CASPAR_LOG(info) << L"  [" << section++ << L"] Cross-output alignment: "
                                 << emb->name << L" vs " << vid->name;
                CASPAR_LOG(info) << L"      " << emb->name << L" video: " << emb_depth << L" frames ahead";
                CASPAR_LOG(info) << L"      " << vid->name << L" video: " << vid_depth << L" frames ahead"
                                 << (vid->pipeline.video_delay_ms > 0.0
                                         ? L" + " + std::to_wstring(static_cast<int>(std::round(vid->pipeline.video_delay_ms))) + L" ms"
                                         : L"");

                if (diff > 0 && vid->pipeline.video_delay_adjustable) {
                    CASPAR_LOG(info) << L"      " << vid->name << L" leads " << emb->name
                                     << L" by ~" << std::setprecision(0) << std::abs(diff_ms) << L" ms";
                    CASPAR_LOG(info) << L"      To align, set " << vid->name << L" <delay>"
                                     << (emb_depth - 1) << L"</delay>";
                } else if (diff > 0) {
                    CASPAR_LOG(info) << L"      " << vid->name << L" leads " << emb->name
                                     << L" by ~" << std::setprecision(0) << std::abs(diff_ms) << L" ms";
                } else {
                    CASPAR_LOG(info) << L"      " << emb->name << L" leads " << vid->name
                                     << L" by ~" << std::setprecision(0) << std::abs(diff_ms) << L" ms";
                }

                // Note conflict when lip-sync and alignment suggest different delays
                if (!audio_only.empty() && vid->pipeline.video_delay_adjustable) {
                    int lip_sync_delay  = audio_only[0]->pipeline.audio_depth_frames - 1;
                    int alignment_delay = emb_depth - 1;
                    if (lip_sync_delay != alignment_delay) {
                        CASPAR_LOG(info) << L"      NOTE: Lip-sync suggests <delay>"
                                         << lip_sync_delay << L"</delay>, alignment suggests <delay>"
                                         << alignment_delay << L"</delay>.";
                        CASPAR_LOG(info) << L"      Prioritize lip-sync; the alignment difference is cosmetic.";
                    }
                }
            }
        }

        CASPAR_LOG(info) << L"  NOTE: This compensates for internal pipeline depth only.";
        CASPAR_LOG(info) << L"  Adjust further for external latency (display, scaler, amplifier).";
        CASPAR_LOG(info) << L"  Use <delay-ms> for fine-tuning after visual/audible check.";
        CASPAR_LOG(info) << L"======================================================";
        CASPAR_LOG(info) << L"";
    }

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

    // Evaluated every tick on purpose: a consumer's answer can change at
    // runtime (a GPU strategy failing over to CPU, a Spout/ProRes shared
    // context dropping out), and the mixer must follow it. The cost is one
    // uncontended lock plus a couple of virtual calls; only the logging is
    // worth keeping out of the steady state, which is done by reporting
    // transitions rather than counting calls.
    bool any_consumer_needs_cpu_data()
    {
        std::lock_guard<std::mutex> lock(consumers_mutex_);

        const frame_consumer* requester = nullptr;
        for (auto& p : consumers_) {
            if (p.second->needs_cpu_frame_data()) {
                requester = p.second.get();
                break;
            }
        }

        // Log only when the answer changes. These used to be function-level
        // statics, so the diagnostic was shared across every channel in the
        // process and went silent after the first few lines server-wide.
        if (requester != last_cpu_requester_) {
            if (requester != nullptr) {
                CASPAR_LOG(info) << print() << L" CPU readback required by consumer " << requester->name() << L".";
            } else if (!consumers_.empty()) {
                CASPAR_LOG(info) << print() << L" No consumer needs CPU readback (" << consumers_.size()
                                 << L" consumers); mixer readback skipped.";
            }
            last_cpu_requester_ = requester;
        }

        return requester != nullptr;
    }

    void operator()(const const_frame&             input_frame1,
                    const const_frame&             input_frame2,
                    const core::video_format_desc& format_desc)
    {
        // Channel timing telemetry
        {
            auto now = std::chrono::steady_clock::now();
            timing_frames_++;
            total_frames_++;
            const double expected_ms = 1000.0 / format_desc_.hz;

            if (timing_last_frame_.time_since_epoch().count() > 0) {
                double frame_ms = std::chrono::duration<double, std::milli>(now - timing_last_frame_).count();
                if (frame_ms > expected_ms * 1.15) {
                    timing_late_++;
                    total_late_++;
                }
                period_sum_ms_ += frame_ms;
                period_min_ms_ = period_samples_ == 0 ? frame_ms : std::min(period_min_ms_, frame_ms);
                period_max_ms_ = period_samples_ == 0 ? frame_ms : std::max(period_max_ms_, frame_ms);
                period_samples_++;
            }
            timing_last_frame_ = now;

            auto elapsed = std::chrono::duration<double>(now - timing_start_).count();
            if (elapsed >= 5.0 && period_samples_ > 1) {
                last_avg_ms_    = period_sum_ms_ / static_cast<double>(period_samples_);
                last_min_ms_    = period_min_ms_;
                last_max_ms_    = period_max_ms_;
                last_jitter_ms_ = period_max_ms_ - period_min_ms_;
                last_late_      = timing_late_;

                CASPAR_LOG(trace) << L"[channel " << channel_info_.index << L"] TIMING: avg=" << std::fixed
                                  << std::setprecision(2) << last_avg_ms_ << L"ms (nominal " << expected_ms
                                  << L") jitter=" << last_jitter_ms_ << L"ms late=" << last_late_ << L"/"
                                  << period_samples_ << L" consume_max=" << consume_max_ms_ << L"ms";

                timing_start_   = now;
                timing_frames_  = 0;
                timing_late_    = 0;
                period_sum_ms_  = 0.0;
                period_samples_ = 0;
                consume_max_ms_ = 0.0;
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

        // Count the consumers claiming to pace this channel. More than one cannot
        // be true at once: whichever blocks longest wins, and the others drift
        // until their buffers absorb the difference -- or stop absorbing it. That
        // is a supported configuration (a genlocked SDI card alongside an audio
        // device, say), but it is worth stating once, because when it does go
        // wrong the symptom is a slow drift with no obvious cause.
        clock_source_count_ = 0;
        for (auto& p : consumers) {
            if (p.second->has_synchronization_clock())
                ++clock_source_count_;
        }

        if (clock_source_count_ > 1 && !multi_clock_warned_) {
            multi_clock_warned_ = true;
            std::wostringstream names;
            bool                first_name = true;
            for (auto& p : consumers) {
                if (!p.second->has_synchronization_clock())
                    continue;
                if (!first_name)
                    names << L", ";
                names << p.second->name();
                first_name = false;
            }
            CASPAR_LOG(info) << print() << L" " << clock_source_count_
                             << L" consumers claim a synchronization clock (" << names.str()
                             << L"). They pace the channel jointly: the slowest wins each tick and the others rely "
                                L"on their buffers to absorb the difference. Watch timing/jitter_ms and "
                                L"timing/consume_load if playout drifts.";
        }


        const auto consume_start = std::chrono::steady_clock::now();

        if (format_desc_.field_count == 2) {
            do_send(core::video_field::a, input_frame1);
            do_send(core::video_field::b, input_frame2);
        } else {
            do_send(core::video_field::progressive, input_frame1);
        }

        // With a hardware clock, this is where the channel is actually paced:
        // send() blocks until the device accepts the frame. Its size relative to
        // the frame period is the remaining headroom.
        last_consume_ms_ =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - consume_start).count();
        consume_max_ms_ = std::max(consume_max_ms_, last_consume_ms_);

        monitor::state state;
        for (auto& p : consumers) {
            state["port"][p.first]             = p.second->state();
            state["port"][p.first]["consumer"] = p.second->name();
        }

        // The A/V pipeline model each consumer declares. log_sync_recommendation()
        // already computes the full picture from these, then only prints it -- so
        // nothing could assert on it. Publishing the inputs lets the 360-client
        // and the test runner check lip-sync and cross-output alignment directly.
        for (auto& p : consumers) {
            const auto info = p.second->av_pipeline();
            auto       node = state["port"][p.first]["pipeline"];
            node["has_video"]           = info.has_video;
            node["has_audio"]           = info.has_audio;
            node["audio_embedded"]      = info.audio_is_embedded;
            node["video_depth_frames"]  = static_cast<int32_t>(info.video_depth_frames);
            node["video_delay_ms"]      = info.video_delay_ms;
            node["video_delay_settable"] = info.video_delay_adjustable;
            node["audio_depth_frames"]  = static_cast<int32_t>(info.audio_depth_frames);
            node["audio_delay_frames"]  = static_cast<int32_t>(info.audio_delay_frames);
            node["audio_device_ms"]     = info.audio_device_latency_ms;
            node["audio_delay_settable"] = info.audio_delay_adjustable;
            node["clock"]               = p.second->has_synchronization_clock();
        }

        const double nominal_ms = 1000.0 / format_desc_.hz;
        state["timing"]["nominal_ms"]    = nominal_ms;
        state["timing"]["period_avg_ms"] = last_avg_ms_;
        state["timing"]["period_min_ms"] = last_min_ms_;
        state["timing"]["period_max_ms"] = last_max_ms_;
        state["timing"]["jitter_ms"]     = last_jitter_ms_;
        // Signed: positive means the channel is running slower than nominal.
        state["timing"]["drift_ms"]      = last_avg_ms_ > 0.0 ? last_avg_ms_ - nominal_ms : 0.0;
        state["timing"]["late_frames"]   = static_cast<int64_t>(total_late_);
        state["timing"]["frames"]        = static_cast<int64_t>(total_frames_);
        state["timing"]["consume_ms"]    = last_consume_ms_;
        // The share of the frame budget spent waiting for consumers to accept the
        // frame. Approaching 1.0 means there is no headroom left.
        state["timing"]["consume_load"]  = nominal_ms > 0.0 ? last_consume_ms_ / nominal_ms : 0.0;
        state["timing"]["clock_sources"] = static_cast<int32_t>(clock_source_count_);

        state_ = std::move(state);

        tick_count_.fetch_add(1, std::memory_order_release);
        {
            // Publish the new epoch to any add()/remove() waiting for this
            // tick's consumer snapshot to be released. Taking the mutex
            // (uncontended, ~tens of ns) closes the lost-wakeup window between
            // a waiter's predicate check and its wait.
            std::lock_guard<std::mutex> tick_lock(tick_mutex_);
        }
        tick_cv_.notify_all();

        const auto needs_sync = clock_source_count_ == 0;

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
