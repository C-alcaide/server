/*
 * Copyright (c) 2024 CasparCG contributors
 *
 * CasparCG is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#include "content_sync.h"

#include <common/log.h>
#include <core/video_channel.h>
#include <core/producer/stage.h>
#include <core/producer/frame_producer.h>

#include <chrono>
#include <limits>

namespace caspar { namespace cluster { namespace sync {

content_sync::content_sync(std::shared_ptr<frame_clock>                       clock,
                           std::vector<std::shared_ptr<core::video_channel>>   channels,
                           int                                                 drift_threshold)
    : clock_(std::move(clock))
    , channels_(std::move(channels))
    , drift_threshold_(drift_threshold)
{
}

content_sync::~content_sync()
{
    stop();
}

void content_sync::start()
{
    if (running_.exchange(true)) {
        return;
    }
    watchdog_thread_ = std::thread([this] { watchdog_loop(); });
    CASPAR_LOG(info) << L"[cluster] Content sync watchdog started, threshold=" << drift_threshold_ << L" frames";
}

void content_sync::stop()
{
    if (!running_.exchange(false)) {
        return;
    }
    if (watchdog_thread_.joinable()) {
        watchdog_thread_.join();
    }
    CASPAR_LOG(info) << L"[cluster] Content sync watchdog stopped, corrections=" << total_corrections_.load();
}

void content_sync::track_producer(int channel_index, int layer_index,
                                  int64_t start_frame, int64_t duration_frames, bool looping)
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto key = make_key(channel_index, layer_index);

    tracked_producer tp;
    tp.channel_index   = channel_index;
    tp.layer_index     = layer_index;
    tp.start_frame     = start_frame;
    tp.duration_frames = duration_frames;
    tp.looping         = looping;
    tp.drift_frames    = 0;
    tp.corrections     = 0;

    tracked_[key] = tp;

    CASPAR_LOG(debug) << L"[cluster] Tracking producer ch=" << channel_index
                      << L" layer=" << layer_index
                      << L" start_frame=" << start_frame
                      << L" duration=" << duration_frames
                      << L" loop=" << looping;
}

void content_sync::untrack_producer(int channel_index, int layer_index)
{
    std::lock_guard<std::mutex> lock(mutex_);
    tracked_.erase(make_key(channel_index, layer_index));
}

void content_sync::track_channel(int channel_index, int max_layer)
{
    std::lock_guard<std::mutex> lock(mutex_);
    tracked_channels_[channel_index] = max_layer;
    CASPAR_LOG(info) << L"[cluster] Tracking entire channel " << channel_index
                     << L", scanning layers 0-" << max_layer;
}

void content_sync::untrack_channel(int channel_index)
{
    std::lock_guard<std::mutex> lock(mutex_);
    tracked_channels_.erase(channel_index);
    for (auto it = tracked_.begin(); it != tracked_.end();) {
        if (it->second.channel_index == channel_index) {
            it = tracked_.erase(it);
        } else {
            ++it;
        }
    }
}

std::vector<channel_sync_status> content_sync::status() const
{
    std::lock_guard<std::mutex> lock(mutex_);

    // Group by channel
    std::unordered_map<int, channel_sync_status> by_channel;
    for (const auto& [key, tp] : tracked_) {
        auto& s = by_channel[tp.channel_index];
        s.channel_index = tp.channel_index;
        s.active_layers++;
        s.total_corrections += tp.corrections;
        if (std::abs(tp.drift_frames) > s.max_drift_frames) {
            s.max_drift_frames = std::abs(tp.drift_frames);
        }
        if (std::abs(tp.drift_frames) > drift_threshold_) {
            s.synced = false;
        }
    }

    std::vector<channel_sync_status> result;
    result.reserve(by_channel.size());
    for (auto& [ch, s] : by_channel) {
        result.push_back(s);
    }
    return result;
}

void content_sync::watchdog_loop()
{
    // Run at approximately frame rate — check once per frame period
    // We use frame_clock to determine timing rather than a fixed sleep
    int64_t last_frame = clock_->current_frame();

    while (running_) {
        int64_t current_frame = clock_->current_frame();

        // Only check on new frames (avoid redundant checks within same frame)
        if (current_frame > last_frame) {
            last_frame = current_frame;

            std::lock_guard<std::mutex> lock(mutex_);

            // Periodic scan for new layers on tracked channels (~every 15 frames ≈ 0.3s at 50fps)
            scan_counter_++;
            if (scan_counter_ % 15 == 0) {
                for (const auto& [ch_idx, max_layer] : tracked_channels_) {
                    scan_channel_layers(ch_idx, max_layer);
                }
            }

            for (auto& [key, tp] : tracked_) {
                check_producer(tp);
            }
        }

        // Sleep ~2ms — well under one frame period (16-20ms at 50-60fps)
        // This gives us sub-frame detection latency with negligible CPU
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
}

void content_sync::scan_channel_layers(int channel_index, int max_layer)
{
    // mutex_ is already held by watchdog_loop
    if (channel_index < 0 || channel_index >= static_cast<int>(channels_.size())) {
        return;
    }

    auto& channel = channels_[channel_index];

    // Limit futures per scan pass to avoid heap allocation storms
    static constexpr int max_probes_per_scan = 10;
    int probes_this_scan = 0;

    for (int layer = 0; layer <= max_layer; ++layer) {
        auto key = make_key(channel_index, layer);
        if (tracked_.count(key)) {
            continue; // Already tracked
        }

        if (++probes_this_scan > max_probes_per_scan) {
            break; // Resume remaining layers next scan cycle
        }

        // Probe this layer for an active producer
        try {
            auto future = channel->stage()->foreground(layer);
            if (future.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready) {
                continue; // Don't block
            }
            auto producer = future.get();
            if (!producer || producer->frame_number() == 0) {
                // Check nb_frames to distinguish empty producer from just-started producer
                if (!producer || producer->nb_frames() == 0) {
                    continue; // Empty/default producer
                }
            }

            // Found an active producer — auto-track it
            tracked_producer tp;
            tp.channel_index = channel_index;
            tp.layer_index   = layer;

            uint32_t nb = producer->nb_frames();
            if (nb != std::numeric_limits<uint32_t>::max()) {
                tp.duration_frames = static_cast<int64_t>(nb);
            }
            tp.last_nb_frames = nb;

            // Anchor start_frame based on current position
            int64_t current_global = clock_->current_frame();
            uint32_t current_fn    = producer->frame_number();
            tp.start_frame      = current_global - static_cast<int64_t>(current_fn);
            tp.last_frame_number = current_fn;

            // Query loop state
            try {
                auto loop_future = channel->stage()->call(layer, {L"loop"});
                if (loop_future.wait_for(std::chrono::milliseconds(5)) == std::future_status::ready) {
                    auto result = loop_future.get();
                    tp.looping = (result == L"1" || result == L"true");
                }
            } catch (...) {}

            tp.requery_cooldown = 5; // Let it settle

            tracked_[key] = tp;

            CASPAR_LOG(info) << L"[cluster] Auto-tracking layer " << channel_index << L"-" << layer
                             << L" duration=" << tp.duration_frames
                             << L" loop=" << tp.looping;
        } catch (...) {
            continue;
        }
    }
}

void content_sync::check_producer(tracked_producer& tp)
{
    // Skip foreground() query on most ticks — producer pointer rarely changes
    // Only re-query every 10 ticks (~20ms) to reduce heap allocations from std::future
    if (++tp.query_skip_counter < 10) {
        // Still use cached producer if available
        if (!tp.cached_producer) {
            return;
        }
        // Use cached frame_number for drift check
        auto fn = tp.cached_producer->frame_number();
        if (fn != tp.last_frame_number) {
            tp.last_frame_number = fn;
            tp.stall_count       = 0;
        }
        return;
    }
    tp.query_skip_counter = 0;

    // Get actual producer from the channel's stage
    if (tp.channel_index < 0 || tp.channel_index >= static_cast<int>(channels_.size())) {
        return;
    }

    auto& channel = channels_[tp.channel_index];
    std::shared_ptr<core::frame_producer> producer;

    try {
        auto future = channel->stage()->foreground(tp.layer_index);
        // Use wait_for(0) to avoid blocking — just check if result is ready
        if (future.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready) {
            return; // Stage busy, skip this tick
        }
        producer = future.get();
        tp.cached_producer = producer; // Cache to avoid future allocs on skip ticks
    } catch (...) {
        return; // Layer doesn't exist or stage error
    }

    if (!producer) {
        // Layer cleared or empty — mark paused, skip drift check
        if (!tp.paused) {
            tp.paused       = true;
            tp.drift_frames = 0;
            CASPAR_LOG(debug) << L"[cluster] Layer " << tp.channel_index << L"-" << tp.layer_index
                              << L" empty, watchdog paused";
        }
        return;
    }

    // Detect content change: nb_frames() changed or producer restarted (frame_number < expected)
    uint32_t current_nb = producer->nb_frames();
    uint32_t current_fn = producer->frame_number();

    bool content_changed = false;
    if (tp.paused) {
        // Was paused, new content appeared
        content_changed = true;
        tp.paused = false;
    } else if (tp.last_nb_frames != 0 && current_nb != tp.last_nb_frames) {
        // Duration changed — different content loaded
        content_changed = true;
    } else if (current_fn == 0 && tp.drift_frames != 0) {
        // Producer reset to frame 0 — content restarted or new clip
        content_changed = true;
    }

    if (content_changed) {
        // Re-anchor: new start_frame, re-query duration and loop
        int64_t current_global = clock_->current_frame();
        tp.start_frame     = current_global - static_cast<int64_t>(current_fn); // Account for frames already played
        tp.last_nb_frames  = current_nb;
        tp.drift_frames    = 0;
        tp.requery_cooldown = 5; // Wait 5 frames before drift-checking (let producer settle)

        // Update duration
        if (current_nb != std::numeric_limits<uint32_t>::max()) {
            tp.duration_frames = static_cast<int64_t>(current_nb);
        } else {
            tp.duration_frames = 0; // Infinite
        }

        // Re-query loop state
        try {
            auto loop_future = channel->stage()->call(tp.layer_index, {L"loop"});
            if (loop_future.wait_for(std::chrono::milliseconds(10)) == std::future_status::ready) {
                auto result = loop_future.get();
                tp.looping = (result == L"1" || result == L"true");
            }
        } catch (...) {
            // Keep previous loop state
        }

        CASPAR_LOG(info) << L"[cluster] Content change detected on " << tp.channel_index << L"-" << tp.layer_index
                         << L", re-anchored: duration=" << tp.duration_frames
                         << L" loop=" << tp.looping;
        return;
    }

    // Cooldown after content change — let producer settle
    if (tp.requery_cooldown > 0) {
        tp.requery_cooldown--;
        tp.last_nb_frames    = current_nb;
        tp.last_frame_number = current_fn;
        return;
    }

    // Detect paused/stalled producer: frame_number() not advancing
    if (current_fn == tp.last_frame_number && current_fn > 0) {
        tp.stall_count++;
        if (tp.stall_count >= 3) {
            // Producer is paused (LOAD preview, PAUSE, or end of non-looping clip)
            if (!tp.paused) {
                tp.paused = true;
                CASPAR_LOG(debug) << L"[cluster] Layer " << tp.channel_index << L"-" << tp.layer_index
                                  << L" stalled at frame " << current_fn << L", drift-check suspended";
            }
            tp.last_nb_frames    = current_nb;
            tp.last_frame_number = current_fn;
            return;
        }
    } else {
        // Producer is advancing — clear stall/pause state
        if (tp.paused && tp.stall_count >= 3) {
            // Was paused, now resumed (e.g., PLAY after LOAD preview)
            int64_t current_global = clock_->current_frame();
            tp.start_frame = current_global - static_cast<int64_t>(current_fn);
            tp.paused      = false;
            tp.stall_count = 0;
            CASPAR_LOG(info) << L"[cluster] Layer " << tp.channel_index << L"-" << tp.layer_index
                             << L" resumed, re-anchored at global frame " << current_global;
            tp.last_nb_frames    = current_nb;
            tp.last_frame_number = current_fn;
            return;
        }
        tp.stall_count = 0;
        tp.paused      = false;
    }

    tp.last_nb_frames    = current_nb;
    tp.last_frame_number = current_fn;

    int64_t current_global = clock_->current_frame();
    int64_t expected       = compute_expected_frame(tp, current_global);

    if (expected < 0) {
        return; // Not yet started or invalid
    }

    int64_t actual = static_cast<int64_t>(current_fn);
    int64_t drift  = expected - actual;

    tp.drift_frames = static_cast<int>(drift);

    // Only correct if drift exceeds threshold
    if (std::abs(drift) <= drift_threshold_) {
        return;
    }

    // Cooldown after a correction: wait before issuing another seek
    // This prevents seek storms when the decoder can't land precisely on the target frame
    if (tp.requery_cooldown > 0) {
        tp.requery_cooldown--;
        return;
    }

    // Issue seek correction
    try {
        std::vector<std::wstring> seek_params = {L"seek", std::to_wstring(expected)};
        channel->stage()->call(tp.layer_index, seek_params);

        tp.corrections++;
        total_corrections_.fetch_add(1, std::memory_order_relaxed);

        // Re-anchor start_frame after correction to avoid immediate re-triggering
        tp.start_frame = current_global - expected;
        tp.requery_cooldown = 15; // Wait ~15 frames (0.3s at 50fps) before next correction

        CASPAR_LOG(warning) << L"[cluster] Drift correction: ch=" << tp.channel_index
                            << L" layer=" << tp.layer_index
                            << L" drift=" << drift << L" frames"
                            << L" -> seek to " << expected;
    } catch (const std::exception& e) {
        CASPAR_LOG(error) << L"[cluster] Seek correction failed: " << e.what();
    }
}

int64_t content_sync::compute_expected_frame(const tracked_producer& tp, int64_t current_global_frame) const
{
    int64_t elapsed = current_global_frame - tp.start_frame;

    if (elapsed < 0) {
        return -1; // Not started yet
    }

    if (tp.duration_frames <= 0) {
        // Unknown/infinite duration (live source or unknown file) — just use elapsed
        return elapsed;
    }

    if (tp.looping) {
        // Looping: expected = elapsed % duration
        return elapsed % tp.duration_frames;
    }

    // Non-looping: clamp to duration
    if (elapsed >= tp.duration_frames) {
        return tp.duration_frames - 1; // Stopped at last frame
    }

    return elapsed;
}

}}} // namespace caspar::cluster::sync
