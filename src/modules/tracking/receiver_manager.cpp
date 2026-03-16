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

#include "receiver_manager.h"

#include "protocol/freed_receiver.h"
#include "protocol/freed_plus_receiver.h"
#include "protocol/osc_receiver.h"
#include "protocol/vrpn_receiver.h"

#include <iostream>
#include <sstream>
#include <stdexcept>

namespace caspar { namespace tracking {

receiver_manager& receiver_manager::instance()
{
    static receiver_manager inst;
    return inst;
}

void receiver_manager::ensure_receiver(tracking_protocol protocol, int port, const std::string& host)
{
    std::lock_guard<std::mutex> lk(mutex_);
    auto key = make_key(protocol, port, host);
    auto& e  = receivers_[key];

    if (e.ref_count == 0) {
        // Create and start a new receiver
        switch (protocol) {
        case tracking_protocol::freed:
            e.receiver = std::make_unique<freed_receiver>(static_cast<uint16_t>(port));
            break;
        case tracking_protocol::freed_plus:
            e.receiver = std::make_unique<freed_plus_receiver>(static_cast<uint16_t>(port));
            break;
        case tracking_protocol::osc:
            e.receiver = std::make_unique<osc_receiver>(static_cast<uint16_t>(port));
            break;
        case tracking_protocol::vrpn:
            e.receiver = std::make_unique<vrpn_receiver>(host, 0 /*camera_id override via binding*/, 0);
            break;
        default:
            throw std::runtime_error("Unknown tracking protocol");
        }
        e.receiver->start();
    }
    ++e.ref_count;
}

void receiver_manager::release_receiver(tracking_protocol protocol, int port, const std::string& host)
{
    std::lock_guard<std::mutex> lk(mutex_);
    auto key = make_key(protocol, port, host);
    auto it  = receivers_.find(key);
    if (it == receivers_.end())
        return;

    --it->second.ref_count;
    if (it->second.ref_count <= 0) {
        it->second.receiver->stop();
        receivers_.erase(it);
    }
}

void receiver_manager::stop_all()
{
    std::lock_guard<std::mutex> lk(mutex_);
    for (auto& [key, entry] : receivers_)
        entry.receiver->stop();
    receivers_.clear();
}

std::string receiver_manager::info() const
{
    std::lock_guard<std::mutex> lk(mutex_);
    std::ostringstream ss;
    ss << receivers_.size() << " active receiver(s)";
    for (auto& [key, e] : receivers_) {
        ss << "\n  [refs=" << e.ref_count << "] " << e.receiver->info();
    }
    return ss.str();
}

}} // namespace caspar::tracking
