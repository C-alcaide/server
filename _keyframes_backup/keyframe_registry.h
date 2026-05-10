/*
 * Copyright (c) 2026 CasparCG Contributors
 *
 * This file is part of CasparCG (www.casparcg.com).
 *
 * CasparCG is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#pragma once

#include "keyframe_data.h"

#include <core/producer/stage.h>

#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>

namespace caspar { namespace keyframes {

struct kf_binding
{
    bool              armed = false;
    keyframe_timeline timeline;
    // Weak reference to the stage for this channel
    std::weak_ptr<core::stage_base> stage;
    int layer_index = 0;

    uint64_t last_frame_number = 0;
    double   last_time_secs = 0.0;
    // Last time explicitly reported by the client via KEYFRAMES TICK.
    // Used as fallback in auto_tick_all when frame_number==0 (e.g. paused layer).
    double   client_time_secs = -1.0;
};

/// Thread-safe per-(channel, layer) keyframe registry.
/// Receives TICK events (current file time) from the AMCP command handler and
/// injects the interpolated transform directly into the stage — identical
/// architecture to the tracking module.
class keyframe_registry
{
  public:
    static keyframe_registry& instance();

    // ─── Binding management ─────────────────────────────────────────────────

    void set_timeline(int channel, int layer, keyframe_timeline tl,
                      std::shared_ptr<core::stage_base> stage_ptr);
    void arm(int channel, int layer);
    void disarm(int channel, int layer);
    void clear(int channel, int layer);
    bool has_binding(int channel, int layer) const;

    std::optional<kf_binding> get_binding(int channel, int layer) const;

    // Retrieve the current timeline as-is (for GET command serialization).
    std::optional<keyframe_timeline> get_timeline(int channel, int layer) const;

    // Patch a single keyframe's state (by time) without replacing the whole timeline.
    // Returns false if no binding or no keyframe matches the given time.
    bool patch_timeline(int channel, int layer, double time_secs, const kf_state& patched_state);

    // ─── Called by AMCP KEYFRAMES TICK ─────────────────────────────────────

    /// Interpolate the timeline at file_time_secs and call stage->apply_transform.
    /// duration_frames: pass 1 for normal playback, 0 for a seek snap.
    void tick(int channel, int layer, double file_time_secs, unsigned int duration_frames = 1);

    /// Called every stage frame from within the stage executor.
    /// apply_cb(layer_index, interpolated_state, duration_frames) is invoked directly
    /// on the caller's thread — no begin_invoke, no weak_ptr indirection.
    /// This guarantees the tween is updated BEFORE the frame is rendered.
    using apply_callback_t = std::function<void(int layer, const kf_state&, unsigned int duration)>;
    void auto_tick_all(int channel,
                       const std::map<int, uint64_t>& layer_frame_numbers,
                       const core::video_format_desc& format_desc,
                       const apply_callback_t& apply_cb);
  private:
    keyframe_registry()  = default;
    ~keyframe_registry() = default;

    mutable std::mutex                               mutex_;
    std::map<std::pair<int,int>, kf_binding>         bindings_;
};

}} // namespace caspar::keyframes
