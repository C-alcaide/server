/*
 * Copyright (c) 2025 CasparCG Contributors
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

#include "tracker_binding.h"
#include "protocol/receiver_base.h"

#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>

namespace caspar { namespace tracking {

/// Manages the lifecycle of protocol receivers.
/// Receivers are reference-counted by (protocol, port/host) key so that
/// multiple bindings sharing the same source do not create duplicate sockets.
class receiver_manager
{
  public:
    static receiver_manager& instance();

    /// Ensure a receiver is running for the given protocol/port.
    /// Increments the reference count; no-op if already running.
    void ensure_receiver(tracking_protocol protocol, int port, const std::string& host = "");

    /// Decrement reference count. Stops and destroys the receiver when it
    /// reaches zero.
    void release_receiver(tracking_protocol protocol, int port, const std::string& host = "");

    void stop_all();

    std::string info() const;

  private:
    receiver_manager()  = default;
    ~receiver_manager() = default;

    struct entry
    {
        std::unique_ptr<receiver_base> receiver;
        int                            ref_count = 0;
    };

    using key_t = std::pair<std::pair<tracking_protocol, int>, std::string>;

    mutable std::mutex          mutex_;
    std::map<key_t, entry>      receivers_;

    static key_t make_key(tracking_protocol p, int port, const std::string& host)
    {
        return {{p, port}, host};
    }
};

}} // namespace caspar::tracking
