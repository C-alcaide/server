/*
 * Copyright (c) 2024 CasparCG contributors
 *
 * CasparCG is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#pragma once

#include "frame_clock.h"

#include <core/fwd.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace caspar { namespace cluster { namespace sync {

/// Tracks a single producer's expected position for drift detection
struct tracked_producer
{
    int      channel_index   = 0;
    int      layer_index     = 0;
    int64_t  start_frame     = 0;  // Global frame when current content started
    int64_t  duration_frames = 0;  // Total content frames (0 = unknown/infinite)
    bool     looping         = false;
    int      drift_frames    = 0;  // Current accumulated drift
    int      corrections     = 0;  // Number of seek corrections issued
    uint32_t last_nb_frames  = 0;  // Cached nb_frames() to detect content change
    uint32_t last_frame_number = 0; // Cached frame_number() to detect paused/stalled producers
    int      stall_count     = 0;  // Consecutive frames where frame_number() didn't advance
    bool     paused          = false; // True if layer has no active producer or is stalled
    int64_t  requery_cooldown = 0; // Frames to wait before re-querying after content change
};

/// Per-channel drift metrics reported by the watchdog
struct channel_sync_status
{
    int     channel_index    = 0;
    int     active_layers    = 0;
    int     max_drift_frames = 0;
    int     total_corrections = 0;
    bool    synced           = true;
};

/// Content sync watchdog: detects drift between expected and actual producer frame
/// position, issuing seek corrections when divergence exceeds threshold.
///
/// Design:
/// - Attaches to video channels via their stage
/// - Each tick (driven by its own lightweight thread at frame rate),
///   compares expected content frame vs actual frame_number() for tracked producers
/// - If |drift| > threshold, issues a CALL SEEK to snap back
/// - Costs: 1 integer comparison per tracked layer per frame (~2ns each)
class content_sync
{
  public:
    content_sync(std::shared_ptr<frame_clock>                                clock,
                 std::vector<std::shared_ptr<core::video_channel>>              channels,
                 int                                                            drift_threshold = 2);

    ~content_sync();

    content_sync(const content_sync&)            = delete;
    content_sync& operator=(const content_sync&) = delete;

    /// Start the watchdog
    void start();

    /// Stop the watchdog
    void stop();

    /// Register a producer for tracking (called when PLAY is scheduled)
    void track_producer(int channel_index, int layer_index, int64_t start_frame,
                        int64_t duration_frames, bool looping);

    /// Track an entire channel — auto-discovers layers with active producers
    void track_channel(int channel_index, int max_layer = 100);

    /// Unregister a producer (called on STOP/CLEAR)
    void untrack_producer(int channel_index, int layer_index);

    /// Untrack all producers on a channel (also removes channel-level tracking)
    void untrack_channel(int channel_index);

    /// Get sync status for all channels
    std::vector<channel_sync_status> status() const;

    /// Get total corrections issued since start
    int total_corrections() const { return total_corrections_.load(std::memory_order_relaxed); }

  private:
    void watchdog_loop();
    void check_producer(tracked_producer& tp);
    void scan_channel_layers(int channel_index, int max_layer);

    int64_t compute_expected_frame(const tracked_producer& tp, int64_t current_global_frame) const;

    std::shared_ptr<frame_clock>                              clock_;
    std::vector<std::shared_ptr<core::video_channel>>         channels_;
    int                                                       drift_threshold_;

    mutable std::mutex                                        mutex_;
    std::unordered_map<uint64_t, tracked_producer>            tracked_; // key = channel<<32 | layer

    // Channel-level tracking: channel_index → max_layer to scan
    std::unordered_map<int, int>                              tracked_channels_;
    int64_t                                                   scan_counter_{0}; // Frame counter for periodic scans

    std::atomic<int>  total_corrections_{0};
    std::atomic<bool> running_{false};
    std::thread       watchdog_thread_;

    static uint64_t make_key(int channel, int layer) {
        return (static_cast<uint64_t>(channel) << 32) | static_cast<uint64_t>(layer);
    }
};

}}} // namespace caspar::cluster::sync
