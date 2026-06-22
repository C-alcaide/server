/*
 * Copyright (c) 2026 CasparCG Contributors
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
 */

#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>

namespace caspar { namespace vulkan_output {

// Lightweight startup gate that prevents present loops from running until all
// consumers on this module have finished heavy initialization (swapchain + window
// creation). This avoids TDR when multiple consumers create swapchains simultaneously.
class startup_gate
{
  public:
    static startup_gate& instance()
    {
        static startup_gate g;
        return g;
    }

    void set_expected(int n) { expected_.store(n); ready_.store(0); }

    void signal_ready()
    {
        ++ready_;
        cv_.notify_all();
    }

    bool wait_all_ready(std::chrono::seconds timeout = std::chrono::seconds(30))
    {
        std::unique_lock<std::mutex> lock(mtx_);
        return cv_.wait_for(lock, timeout, [this] {
            return ready_.load() >= expected_.load() || expected_.load() == 0;
        });
    }

    void reset()
    {
        expected_.store(0);
        ready_.store(0);
    }

  private:
    startup_gate() = default;

    std::atomic<int>        expected_{0};
    std::atomic<int>        ready_{0};
    std::mutex              mtx_;
    std::condition_variable cv_;
};

}} // namespace caspar::vulkan_output
