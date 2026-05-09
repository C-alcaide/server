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
 *
 * PosiStageNet (PSN) protocol specification by VYV Corporation.
 * C++ reference implementation licensed under the MIT License.
 * See vendor/psn/ for the original source and license.
 */

#include "psn_receiver.h"

#include "../camera_data.h"
#include "../tracker_registry.h"

// PSN header-only library (vendored, MIT license)
#include <psn_lib.hpp>

#include <boost/asio.hpp>

#include <array>
#include <atomic>
#include <cmath>
#include <iostream>
#include <string>
#include <thread>

namespace caspar { namespace tracking {

static constexpr double PSN_DEG2RAD = 3.141592653589793 / 180.0;
static constexpr double PSN_M_TO_MM = 1000.0; // PSN positions are in metres; camera_data uses mm

// ---------------------------------------------------------------------------

struct psn_receiver::impl
{
    const uint16_t    port_;
    const std::string multicast_addr_;
    std::atomic<bool> running_{false};

    boost::asio::io_context        io_;
    boost::asio::ip::udp::socket   socket_{io_};
    boost::asio::ip::udp::endpoint sender_endpoint_;

    std::array<uint8_t, ::psn::MAX_UDP_PACKET_SIZE * 2> recv_buf_{};
    std::thread io_thread_;

    ::psn::psn_decoder decoder_;
    uint8_t            last_frame_id_{0};
    bool               has_received_frame_{false};

    explicit impl(uint16_t port, std::string multicast_addr)
        : port_(port)
        , multicast_addr_(std::move(multicast_addr))
    {
    }

    ~impl() { stop(); }

    void start()
    {
        if (running_.exchange(true))
            return;

        socket_.open(boost::asio::ip::udp::v4());
        socket_.set_option(boost::asio::socket_base::reuse_address(true));
        socket_.bind(boost::asio::ip::udp::endpoint(boost::asio::ip::udp::v4(), port_));

        // Join the PSN multicast group
        socket_.set_option(
            boost::asio::ip::multicast::join_group(
                boost::asio::ip::make_address_v4(multicast_addr_)));

        start_receive();

        io_thread_ = std::thread([this] { io_.run(); });
    }

    void stop()
    {
        if (!running_.exchange(false))
            return;
        boost::asio::post(io_, [this] { socket_.close(); });
        if (io_thread_.joinable())
            io_thread_.join();
        io_.restart();
    }

    void start_receive()
    {
        socket_.async_receive_from(
            boost::asio::buffer(recv_buf_),
            sender_endpoint_,
            [this](const boost::system::error_code& ec, std::size_t bytes_transferred) {
                if (ec) {
                    if (ec != boost::asio::error::operation_aborted)
                        std::cerr << "[tracking/psn] receive error: " << ec.message() << "\n";
                    return;
                }
                process_packet(bytes_transferred);
                if (running_)
                    start_receive();
            });
    }

    void process_packet(std::size_t len)
    {
        if (len == 0 || len > ::psn::MAX_UDP_PACKET_SIZE)
            return;

        decoder_.decode(reinterpret_cast<const char*>(recv_buf_.data()), len);

        // Only emit data when a new frame has been committed
        const auto& data = decoder_.get_data();
        if (has_received_frame_ && data.header.frame_id == last_frame_id_)
            return;
        last_frame_id_      = data.header.frame_id;
        has_received_frame_ = true;

        for (const auto& [id, trk] : data.trackers) {
            camera_data d;
            d.camera_id = static_cast<int>(id);

            if (trk.is_pos_set()) {
                ::psn::float3 pos = trk.get_pos();
                d.x = static_cast<double>(pos.x) * PSN_M_TO_MM;
                d.y = static_cast<double>(pos.y) * PSN_M_TO_MM;
                d.z = static_cast<double>(pos.z) * PSN_M_TO_MM;
            }

            if (trk.is_ori_set()) {
                ::psn::float3 ori = trk.get_ori();
                d.pan  = static_cast<double>(ori.x) * PSN_DEG2RAD;
                d.tilt = static_cast<double>(ori.y) * PSN_DEG2RAD;
                d.roll = static_cast<double>(ori.z) * PSN_DEG2RAD;
            }

            // PSN has no zoom/focus — leave at defaults (0)

            d.timestamp = std::chrono::steady_clock::now();

            tracker_registry::instance().on_data(d);
        }
    }
};

// ---------------------------------------------------------------------------

psn_receiver::psn_receiver(uint16_t port, std::string multicast_addr)
    : impl_(std::make_unique<impl>(port, std::move(multicast_addr)))
{
}

psn_receiver::~psn_receiver() = default;

void psn_receiver::start()
{
    impl_->start();
}

void psn_receiver::stop()
{
    impl_->stop();
}

std::string psn_receiver::info() const
{
    return "PSN multicast " + impl_->multicast_addr_ + " port " + std::to_string(impl_->port_);
}

}} // namespace caspar::tracking
