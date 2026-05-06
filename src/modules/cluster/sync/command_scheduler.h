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

#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <atomic>

namespace caspar { namespace cluster { namespace sync {

/// A scheduled command with a target frame for execution
struct scheduled_command
{
    int64_t      target_frame;
    std::wstring command_text;    // Raw AMCP command text
    int          priority = 0;   // Higher = execute first at same frame

    bool operator>(const scheduled_command& other) const
    {
        if (target_frame != other.target_frame)
            return target_frame > other.target_frame; // Min-heap by frame
        return priority < other.priority;             // Higher priority first
    }
};

/// Callback type for executing a scheduled command
using command_executor = std::function<void(const std::wstring& command_text)>;

/// Priority queue that executes commands at their target frame.
/// Runs a tight loop checking frame_clock and dispatching commands.
class command_scheduler
{
  public:
    command_scheduler(std::shared_ptr<frame_clock> clock, command_executor executor);
    ~command_scheduler();

    command_scheduler(const command_scheduler&)            = delete;
    command_scheduler& operator=(const command_scheduler&) = delete;

    /// Start the scheduler dispatch loop
    void start();

    /// Stop the scheduler
    void stop();

    /// Schedule a command for execution at target_frame
    void schedule(int64_t target_frame, std::wstring command_text, int priority = 0);

    /// Schedule a command with automatic sync_margin offset
    void schedule_now(std::wstring command_text, int sync_margin);

    /// Get number of pending commands
    size_t pending_count() const;

    /// Get the current frame from the underlying clock
    int64_t current_frame() const { return clock_->current_frame(); }

  private:
    void dispatch_loop();

    std::shared_ptr<frame_clock> clock_;
    command_executor             executor_;

    mutable std::mutex                                                              queue_mutex_;
    std::priority_queue<scheduled_command, std::vector<scheduled_command>,
                        std::greater<scheduled_command>>                             queue_;

    std::atomic<bool> running_{false};
    std::thread       dispatch_thread_;
};

}}} // namespace caspar::cluster::sync
