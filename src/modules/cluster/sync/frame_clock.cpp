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
    , fps_packed_(pack_fps(fps_num, fps_den))
{
}

int64_t frame_clock::current_frame() const
{
    return ptp_ns_to_frame(clock_->now_ns());
}

int64_t frame_clock::frame_to_ptp_ns(int64_t frame) const
{
    auto [num, den] = unpack_fps(fps_packed_.load(std::memory_order_relaxed));
    // ptp_ns = epoch_origin + frame * den * 1e9 / num
    // Split to avoid overflow: frame * den * 1e9 overflows int64 at ~42h for 59.94fps
    // frame_ns = (frame / num) * den * 1e9 + ((frame % num) * den * 1e9) / num
    int64_t whole_sec_frames = frame / num;          // How many whole "seconds worth" of frames
    int64_t remainder_frames = frame % num;          // Leftover frames within the second
    int64_t whole_ns = whole_sec_frames * den * 1'000'000'000LL; // safe: whole_sec_frames * den fits for centuries
    int64_t frac_ns  = (remainder_frames * den * 1'000'000'000LL) / num; // safe: remainder < num, so remainder*den*1e9 < num*den*1e9 ≈ 6e13
    return epoch_origin_ns_ + whole_ns + frac_ns;
}

int64_t frame_clock::ptp_ns_to_frame(int64_t ptp_ns) const
{
    auto [num, den] = unpack_fps(fps_packed_.load(std::memory_order_relaxed));
    int64_t elapsed = ptp_ns - epoch_origin_ns_;
    if (elapsed < 0) {
        return -1;
    }
    // Split computation to avoid overflow of (elapsed * num):
    // frame = elapsed_ns * num / (den * 1e9)
    // Split elapsed into whole seconds and remaining nanoseconds
    int64_t elapsed_sec = elapsed / 1'000'000'000LL;
    int64_t elapsed_rem = elapsed % 1'000'000'000LL;
    // whole_frames = elapsed_sec * num / den (safe: elapsed_sec * num fits int64 for centuries)
    int64_t whole_frames = (elapsed_sec * num) / den;
    // frac_frames = elapsed_rem * num / (den * 1e9) (safe: elapsed_rem < 1e9, * num < 6e13)
    int64_t frac_frames = (elapsed_rem * num) / (static_cast<int64_t>(den) * 1'000'000'000LL);
    return whole_frames + frac_frames;
}

int64_t frame_clock::ns_until_frame(int64_t target_frame) const
{
    int64_t target_ns = frame_to_ptp_ns(target_frame);
    int64_t now_ns    = clock_->now_ns();
    return target_ns - now_ns;
}

void frame_clock::set_framerate(int fps_num, int fps_den)
{
    if (fps_den <= 0 || fps_num <= 0) {
        return;
    }
    fps_packed_.store(pack_fps(fps_num, fps_den), std::memory_order_relaxed);
}

}}} // namespace caspar::cluster::sync
