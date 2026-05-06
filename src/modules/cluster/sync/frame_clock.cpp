/*
 * Copyright (c) 2024 CasparCG contributors
 *
 * CasparCG is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#include "frame_clock.h"

namespace caspar { namespace cluster { namespace sync {

frame_clock::frame_clock(std::shared_ptr<ptp::ptp_clock> clock,
                         int64_t                         epoch_origin_ns,
                         int                             fps_num,
                         int                             fps_den)
    : clock_(std::move(clock))
    , epoch_origin_ns_(epoch_origin_ns)
    , fps_num_(fps_num)
    , fps_den_(fps_den)
{
}

int64_t frame_clock::current_frame() const
{
    return ptp_ns_to_frame(clock_->now_ns());
}

int64_t frame_clock::frame_to_ptp_ns(int64_t frame) const
{
    int num = fps_num_.load(std::memory_order_relaxed);
    int den = fps_den_.load(std::memory_order_relaxed);
    // ptp_ns = epoch_origin + frame * den * 1e9 / num
    // Use integer arithmetic to avoid floating point drift
    return epoch_origin_ns_ + (frame * den * 1'000'000'000LL) / num;
}

int64_t frame_clock::ptp_ns_to_frame(int64_t ptp_ns) const
{
    int num = fps_num_.load(std::memory_order_relaxed);
    int den = fps_den_.load(std::memory_order_relaxed);
    int64_t elapsed = ptp_ns - epoch_origin_ns_;
    if (elapsed < 0) {
        return -1;
    }
    // frame = floor(elapsed_ns * fps_num / (fps_den * 1e9))
    return (elapsed * num) / (static_cast<int64_t>(den) * 1'000'000'000LL);
}

int64_t frame_clock::ns_until_frame(int64_t target_frame) const
{
    int64_t target_ns = frame_to_ptp_ns(target_frame);
    int64_t now_ns    = clock_->now_ns();
    return target_ns - now_ns;
}

void frame_clock::set_framerate(int fps_num, int fps_den)
{
    fps_num_.store(fps_num, std::memory_order_relaxed);
    fps_den_.store(fps_den, std::memory_order_relaxed);
}

}}} // namespace caspar::cluster::sync
