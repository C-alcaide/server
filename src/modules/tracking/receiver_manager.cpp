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
#include "protocol/opentrackio_receiver.h"
#include "protocol/osc_receiver.h"
#include "protocol/psn_receiver.h"
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
    if (auto it = receivers_.find(key); it != receivers_.end() && it->second.ref_count > 0) {
        ++it->second.ref_count;
        return;
    }

    // ONE UNICAST RECEIVER PER PORT. FREED, FREED_PLUS and OSC are unicast and every receiver sets
    // SO_REUSEADDR, so a second protocol on a port another already holds -- FREED and FREED_PLUS
    // both default to 6301 -- opened a second socket, and on Windows the two then split the
    // incoming packets unpredictably: each tracker delivered roughly half its samples to the
    // wrong decoder, which drops them as malformed. Refused now, naming the holder. Multicast
    // receivers (PSN, OpenTrackIO) are exempt: every member of a group gets its own copy.
    const auto unicast = [](tracking_protocol p) {
        return p == tracking_protocol::freed || p == tracking_protocol::freed_plus || p == tracking_protocol::osc;
    };
    for (const auto& [k, other] : receivers_) {
        if (other.ref_count > 0 && k.first.second == port && k.first.first != protocol &&
            (unicast(protocol) || unicast(k.first.first)))
            throw std::runtime_error("UDP port " + std::to_string(port) +
                                     " is already held by another tracking protocol's receiver; "
                                     "two sockets on one port split its packets between them");
    }

    // BUILD AND START FIRST, INSERT AFTER. The entry used to be created by `receivers_[key]` before
    // start(), so a start() that threw -- a port another process holds -- left an entry at ref
    // count 0 with a constructed receiver in it: bound, never serviced, and holding the port.
    entry e;
    {
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
        case tracking_protocol::psn:
            e.receiver = std::make_unique<psn_receiver>(
                static_cast<uint16_t>(port),
                host.empty() ? "236.10.10.10" : host);
            break;
        case tracking_protocol::opentrackio:
            e.receiver = std::make_unique<opentrackio_receiver>(
                static_cast<uint16_t>(port),
                host.empty() ? "239.135.1.100" : host);
            break;
        default:
            throw std::runtime_error("Unknown tracking protocol");
        }
        e.receiver->start();
    }
    e.ref_count = 1;
    receivers_[key] = std::move(e);
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
