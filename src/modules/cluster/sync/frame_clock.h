/*
 * Copyright (c) 2024 CasparCG contributors
 *
 * CasparCG is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#pragma once

#include "../ptp/ptp_clock.h"

#include <atomic>
#include <cstdint>

namespace caspar { namespace cluster { namespace sync {

/// Derives a monotonically increasing global frame number from PTP time.
/// Formula: global_frame = floor((ptp_time_ns - epoch_origin_ns) * fps_num / (fps_den * 1e9))
class frame_clock
{
  public:
    frame_clock(std::shared_ptr<ptp::ptp_clock> clock,
                int64_t                         epoch_origin_ns,
                int                             fps_num,
                int                             fps_den);

    /// Get the current global frame number
    int64_t current_frame() const;

    /// Get PTP time for a given frame number
    int64_t frame_to_ptp_ns(int64_t frame) const;

    /// Get the frame number for a given PTP time
    int64_t ptp_ns_to_frame(int64_t ptp_ns) const;

    /// Get time until next frame boundary (for precise scheduling)
    int64_t ns_until_frame(int64_t target_frame) const;

    /// Update framerate (e.g., on channel format change)
    void set_framerate(int fps_num, int fps_den);

    /// Get epoch origin
    int64_t epoch_origin_ns() const { return epoch_origin_ns_; }

  private:
    std::shared_ptr<ptp::ptp_clock> clock_;
    int64_t                         epoch_origin_ns_;
    std::atomic<int>                fps_num_;
    std::atomic<int>                fps_den_;
};

}}} // namespace caspar::cluster::sync
