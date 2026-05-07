/*
 * Copyright (c) 2024 CasparCG contributors
 *
 * CasparCG is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#include "command_scheduler.h"

#include <common/log.h>

#include <chrono>

#ifdef _MSC_VER
#include <intrin.h>
#define SPIN_PAUSE() _mm_pause()
#else
#define SPIN_PAUSE() ((void)0)
#endif

namespace caspar { namespace cluster { namespace sync {

command_scheduler::command_scheduler(std::shared_ptr<frame_clock> clock, command_executor executor)
    : clock_(std::move(clock))
    , executor_(std::move(executor))
{
}

command_scheduler::~command_scheduler()
{
    stop();
}

void command_scheduler::start()
{
    if (running_.exchange(true)) {
        return;
    }
    dispatch_thread_ = std::thread([this] { dispatch_loop(); });
    CASPAR_LOG(info) << L"[cluster] Command scheduler started";
}

void command_scheduler::stop()
{
    if (!running_.exchange(false)) {
        return;
    }
    if (dispatch_thread_.joinable()) {
        dispatch_thread_.join();
    }
    CASPAR_LOG(info) << L"[cluster] Command scheduler stopped";
}

void command_scheduler::schedule(int64_t target_frame, std::wstring command_text, int priority)
{
    std::lock_guard<std::mutex> lock(queue_mutex_);
    queue_.push({target_frame, std::move(command_text), priority});
}

void command_scheduler::schedule_now(std::wstring command_text, int sync_margin)
{
    int64_t target = clock_->current_frame() + sync_margin;
    schedule(target, std::move(command_text));
}

size_t command_scheduler::pending_count() const
{
    std::lock_guard<std::mutex> lock(queue_mutex_);
    return queue_.size();
}

void command_scheduler::dispatch_loop()
{
    while (running_) {
        int64_t now_frame = clock_->current_frame();

        // Dispatch all commands due at or before current frame
        while (true) {
            scheduled_command cmd;
            {
                std::lock_guard<std::mutex> lock(queue_mutex_);
                if (queue_.empty() || queue_.top().target_frame > now_frame) {
                    break;
                }
                cmd = queue_.top();
                queue_.pop();
            }

            // Check if command is late
            int64_t lateness = now_frame - cmd.target_frame;
            if (lateness > 1) {
                CASPAR_LOG(warning) << L"[cluster] Command late by " << lateness
                                    << L" frames: " << cmd.command_text.substr(0, 60);
            }

            // Execute
            try {
                executor_(cmd.command_text);
            } catch (const std::exception& e) {
                CASPAR_LOG(error) << L"[cluster] Command execution failed: "
                                  << std::string(e.what()).c_str();
            }
        }

        // Sleep until close to next frame boundary
        // Use sub-millisecond sleep to avoid busy-wait but maintain precision
        int64_t ns_until = -1;
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            if (!queue_.empty()) {
                int64_t next_frame = queue_.top().target_frame;
                ns_until = clock_->ns_until_frame(next_frame);
            }
        }

        if (ns_until >= 0) {
            if (ns_until > 2'000'000) { // More than 2ms away
                std::this_thread::sleep_for(std::chrono::microseconds(500));
            } else if (ns_until > 100'000) { // More than 100µs away
                std::this_thread::sleep_for(std::chrono::microseconds(50));
            } else {
                // Sub-100µs: spin with pause to reduce power and pipeline stalls
                SPIN_PAUSE();
            }
        } else {
            // No pending commands, sleep longer
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
}

}}} // namespace caspar::cluster::sync
