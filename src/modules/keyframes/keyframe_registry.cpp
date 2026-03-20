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

#include "keyframe_registry.h"

#include <common/tweener.h>
#include <common/log.h>

namespace caspar { namespace keyframes {

keyframe_registry& keyframe_registry::instance()
{
    static keyframe_registry inst;
    return inst;
}

void keyframe_registry::set_timeline(int channel, int layer, keyframe_timeline tl,
                                     std::shared_ptr<core::stage_base> stage_ptr)
{
    std::lock_guard<std::mutex> lk(mutex_);
    auto& b        = bindings_[{channel, layer}];
    b.timeline     = std::move(tl);
    b.stage        = stage_ptr;
    b.layer_index  = layer;
}

void keyframe_registry::arm(int channel, int layer)
{
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = bindings_.find({channel, layer});
    if (it != bindings_.end()) {
        it->second.armed = true;
        // Force is_seek=true on the very first auto_tick_all after arming so
        // the KF state is applied immediately (duration=0) rather than
        // tweening from whatever stale last_time_secs was stored.
        it->second.last_time_secs    = -1.0;
        it->second.last_frame_number = 0;
    } else
        CASPAR_LOG(warning) << L"[keyframes] ARM on non-existent binding ch="
                            << channel << L" lay=" << layer;
}

void keyframe_registry::disarm(int channel, int layer)
{
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = bindings_.find({channel, layer});
    if (it != bindings_.end())
        it->second.armed = false;
}

void keyframe_registry::clear(int channel, int layer)
{
    std::lock_guard<std::mutex> lk(mutex_);
    bindings_.erase({channel, layer});
}

bool keyframe_registry::has_binding(int channel, int layer) const
{
    std::lock_guard<std::mutex> lk(mutex_);
    return bindings_.count({channel, layer}) > 0;
}

std::optional<kf_binding> keyframe_registry::get_binding(int channel, int layer) const
{
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = bindings_.find({channel, layer});
    if (it == bindings_.end()) return std::nullopt;
    return it->second;
}

std::optional<keyframe_timeline> keyframe_registry::get_timeline(int channel, int layer) const
{
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = bindings_.find({channel, layer});
    if (it == bindings_.end()) return std::nullopt;
    return it->second.timeline;
}

bool keyframe_registry::patch_timeline(int channel, int layer,
                                       double time_secs, const kf_state& patched_state)
{
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = bindings_.find({channel, layer});
    if (it == bindings_.end()) return false;
    return it->second.timeline.patch_at_time(time_secs, patched_state);
}

void keyframe_registry::tick(int channel, int layer,
                             double file_time_secs, unsigned int duration_frames)
{
    // Store the client-reported time so auto_tick_all can use it as a
    // fallback when frame_number()==0 (e.g. the layer is paused or the clip
    // has not yet produced a frame with a non-zero PTS).
    {
        std::lock_guard<std::mutex> lk(mutex_);
        auto it = bindings_.find({channel, layer});
        if (it != bindings_.end())
            it->second.client_time_secs = file_time_secs;
    }
    // auto_tick_all owns tween updates; do not also call apply_transform here
    // or it will fight auto_tick_all every render frame.
}

void keyframe_registry::auto_tick_all(int channel,
                                      const std::map<int, uint64_t>& layer_frame_numbers,
                                      const core::video_format_desc& format_desc,
                                      const apply_callback_t& apply_cb)
{
    double fps = 25.0;
    if (format_desc.framerate.numerator() > 0 && format_desc.framerate.denominator() > 0)
        fps = static_cast<double>(format_desc.framerate.numerator()) / format_desc.framerate.denominator();

    for (const auto& frame_pair : layer_frame_numbers) {
        int      layer_index = frame_pair.first;
        uint64_t frame_number = frame_pair.second;

        // Snapshot the binding under the lock
        kf_binding b;
        {
            std::lock_guard<std::mutex> lk(mutex_);
            auto it = bindings_.find({channel, layer_index});
            if (it == bindings_.end() || !it->second.armed)
                continue;
            b = it->second;  // deep copy: timeline, last_time_secs, etc.
        }

        if (b.timeline.empty()) {
            CASPAR_LOG(warning) << L"[keyframes] auto_tick: binding has EMPTY timeline ch="
                                << channel << L" lay=" << layer_index;
            continue;
        }

        // Use client_time_secs (from KEYFRAMES TICK) as the authoritative time
        // whenever the producer's frame_number() is unreliable:
        //   • frame_number == 0:  av_producer before it has produced a frame,
        //                         or empty/initial producer state.
        //   • frame_number == last_frame_number (frozen):  the producer's
        //     base-class frame_number_ is only incremented by receive(), which
        //     is skipped when the CasparCG layer is paused (layer.cpp calls
        //     last_frame() instead).  A SEEK on a paused layer never triggers
        //     receive() either, so after seeking while paused frame_number()
        //     stays stuck at its pre-seek value.  prores_producer has no
        //     frame_number() override and always falls into this category.
        double time_secs;
        bool frame_frozen = (frame_number == 0 || frame_number == b.last_frame_number);
        if (frame_frozen && b.client_time_secs >= 0.0) {
            time_secs = b.client_time_secs;
        } else {
            time_secs = static_cast<double>(frame_number) / fps;
        }

        // Detect seeking: backward jump or time delta > 2.5 frames.
        // Also treat delta == 0.0 (paused, same frame every tick) as an
        // immediate-apply case (duration=0) so that SET commands sent while
        // the clip is paused are visible on the VERY NEXT rendered frame.
        // Without this, a fresh tweened_transform(src, dst, 1) created by
        // auto_tick_all would always be fetched at t=0 (returning src, not
        // dst) because the tween is replaced again before the next tick,
        // making KF state changes invisible during paused editing.
        bool is_seek = false;
        if (time_secs < b.last_time_secs) {
            is_seek = true;
        } else {
            double delta = time_secs - b.last_time_secs;
            if (delta == 0.0 || delta > 2.5 / fps)
                is_seek = true;
        }

        unsigned int duration = is_seek ? 0u : 1u;

        // Update tracking state under lock
        {
            std::lock_guard<std::mutex> lk(mutex_);
            auto it = bindings_.find({channel, layer_index});
            if (it != bindings_.end()) {
                it->second.last_frame_number = frame_number;
                it->second.last_time_secs = time_secs;
            }
        }

        // Interpolate and call the apply callback directly on the caller's thread.
        // This avoids executor::begin_invoke and guarantees the tween is updated
        // BEFORE the frame is rendered (since we are already on the stage executor).
        kf_state s = b.timeline.interpolate(time_secs);

        apply_cb(layer_index, s, duration);
    }
}

}} // namespace caspar::keyframes
