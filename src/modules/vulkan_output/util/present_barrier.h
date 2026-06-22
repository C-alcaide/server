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
#include <map>
#include <memory>
#include <mutex>

namespace caspar { namespace vulkan_output {

// Software present barrier for frame-locking multiple consumers without
// Quadro Sync hardware. Consumers in the same sync_group all wait at the
// barrier before presenting, ensuring synchronized frame output.
//
// Works across different VkDevices and even different physical GPUs.
// Typical latency overhead: <1ms per barrier wait.

class present_barrier
{
  public:
    static present_barrier& instance()
    {
        static present_barrier b;
        return b;
    }

    // Register a consumer in a sync group. Returns a token — barrier is active
    // as long as at least one token exists for the group.
    struct token
    {
        int group_id;
        token(int g) : group_id(g) {}
        ~token() { present_barrier::instance().leave(group_id); }
        token(const token&) = delete;
        token& operator=(const token&) = delete;
    };

    std::shared_ptr<token> join(int group_id)
    {
        std::lock_guard<std::mutex> lock(mtx_);
        groups_[group_id].member_count++;
        return std::make_shared<token>(group_id);
    }

    // Block until all members of the group have called wait() for this generation.
    // Returns false if timed out (safety against deadlock on shutdown).
    bool wait(int group_id, std::chrono::milliseconds timeout = std::chrono::milliseconds(100))
    {
        std::unique_lock<std::mutex> lock(mtx_);
        auto& g = groups_[group_id];

        auto my_gen = g.generation;
        g.arrived++;

        if (g.arrived >= g.member_count) {
            // Last to arrive — release all
            g.arrived = 0;
            g.generation++;
            lock.unlock();
            cv_.notify_all();
            return true;
        }

        // Wait for generation to advance (meaning all arrived)
        return cv_.wait_for(lock, timeout, [&] {
            return g.generation != my_gen;
        });
    }

  private:
    present_barrier() = default;

    void leave(int group_id)
    {
        std::lock_guard<std::mutex> lock(mtx_);
        auto it = groups_.find(group_id);
        if (it != groups_.end()) {
            it->second.member_count--;
            if (it->second.member_count <= 0)
                groups_.erase(it);
        }
    }

    struct group_state
    {
        int      member_count = 0;
        int      arrived      = 0;
        uint64_t generation   = 0;
    };

    std::mutex                   mtx_;
    std::condition_variable      cv_;
    std::map<int, group_state>   groups_;
};

}} // namespace caspar::vulkan_output
