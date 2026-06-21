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
 * OpenTrackIO is part of the SMPTE Rapid Industry Solutions (RIS) On-Set
 * Virtual Production (OSVP) initiative. Implemented from the public schema.
 */

#include "opentrackio_receiver.h"

#include "../camera_data.h"
#include "../tracker_registry.h"

#include <boost/asio.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace caspar { namespace tracking {

namespace {
constexpr double DEG2RAD_OTRK = 3.14159265358979323846 / 180.0;
constexpr double M_TO_MM      = 1000.0;
} // namespace

struct opentrackio_receiver::impl
{
    const uint16_t    port_;
    const std::string multicast_addr_;
    std::atomic<bool> running_{false};

    boost::asio::io_context        io_;
    boost::asio::ip::udp::socket   socket_{io_};
    boost::asio::ip::udp::endpoint sender_endpoint_;
    std::vector<uint8_t>           recv_buf_;
    std::thread                    io_thread_;

    explicit impl(uint16_t port, std::string multicast_addr)
        : port_(port)
        , multicast_addr_(std::move(multicast_addr))
    {
        recv_buf_.resize(65536); // OpenTrackIO samples are small but allow headroom
    }

    ~impl() { stop(); }

    void start()
    {
        if (running_.exchange(true))
            return;

        socket_.open(boost::asio::ip::udp::v4());
        socket_.set_option(boost::asio::socket_base::reuse_address(true));
        socket_.bind(boost::asio::ip::udp::endpoint(boost::asio::ip::udp::v4(), port_));

        // Join the OpenTrackIO multicast group.
        socket_.set_option(
            boost::asio::ip::multicast::join_group(boost::asio::ip::make_address_v4(multicast_addr_)));

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
            [this](const boost::system::error_code& ec, std::size_t bytes) {
                if (ec) {
                    if (ec != boost::asio::error::operation_aborted)
                        std::cerr << "[tracking/opentrackio] receive error: " << ec.message() << "\n";
                    return;
                }
                process_packet(reinterpret_cast<const char*>(recv_buf_.data()), bytes);
                if (running_)
                    start_receive();
            });
    }

    /// Extract the JSON object from a datagram that may carry an "OTrk" transport
    /// header and/or a trailing checksum: take the span from the first '{' to the
    /// matching last '}'.
    static bool extract_json(const char* data, std::size_t len, std::string& out)
    {
        std::size_t begin = 0;
        while (begin < len && data[begin] != '{')
            ++begin;
        if (begin >= len)
            return false;
        std::size_t end = len;
        while (end > begin && data[end - 1] != '}')
            --end;
        if (end <= begin)
            return false;
        out.assign(data + begin, data + end);
        return true;
    }

    void process_packet(const char* data, std::size_t len)
    {
        if (len == 0)
            return;

        std::string json;
        if (!extract_json(data, len, json))
            return;

        namespace pt = boost::property_tree;
        pt::ptree root;
        try {
            std::istringstream is(json);
            pt::read_json(is, root);
        } catch (const std::exception& e) {
            std::cerr << "[tracking/opentrackio] JSON parse error: " << e.what() << "\n";
            return;
        }

        camera_data d;
        d.camera_id = root.get<int>("sourceNumber", 0);

        // ── Transform: translation (metres) + rotation (degrees) ──────────────
        // OpenTrackIO carries a 'transforms' array; use the first camera transform.
        bool have_transform = false;
        if (auto transforms = root.get_child_optional("transforms")) {
            for (const auto& [_, xf] : *transforms) {
                if (auto tr = xf.get_child_optional("translation")) {
                    // Axis map: OpenTrackIO right-handed metric world → camera_data
                    // (x = right, y = up, z = toward subject). Adjust here if a
                    // source uses a different handedness/orientation.
                    const double tx = tr->get("x", 0.0);
                    const double ty = tr->get("y", 0.0);
                    const double tz = tr->get("z", 0.0);
                    d.x = tx * M_TO_MM;
                    d.y = ty * M_TO_MM;
                    d.z = tz * M_TO_MM;
                }
                if (auto rot = xf.get_child_optional("rotation")) {
                    d.pan  = rot->get("pan", 0.0) * DEG2RAD_OTRK;
                    d.tilt = rot->get("tilt", 0.0) * DEG2RAD_OTRK;
                    d.roll = rot->get("roll", 0.0) * DEG2RAD_OTRK;
                }
                have_transform = true;
                break; // first transform only
            }
        }

        // ── Lens encoders: normalised 0..1 → raw 16-bit ───────────────────────
        if (auto lens = root.get_child_optional("lens")) {
            if (auto enc = lens->get_child_optional("encoders")) {
                auto to_raw = [](double v) {
                    if (v < 0.0) v = 0.0;
                    if (v > 1.0) v = 1.0;
                    return static_cast<uint16_t>(v * 65535.0);
                };
                if (auto z = enc->get_optional<double>("zoom"))
                    d.zoom = to_raw(*z);
                if (auto f = enc->get_optional<double>("focus"))
                    d.focus = to_raw(*f);
                if (auto i = enc->get_optional<double>("iris"))
                    d.iris = to_raw(*i);
            }
        }

        if (!have_transform)
            return; // nothing usable in this sample

        d.timestamp = std::chrono::steady_clock::now();
        tracker_registry::instance().on_data(d);
    }
};

// ---------------------------------------------------------------------------

opentrackio_receiver::opentrackio_receiver(uint16_t port, std::string multicast_addr)
    : impl_(std::make_unique<impl>(port, std::move(multicast_addr)))
{
}

opentrackio_receiver::~opentrackio_receiver() = default;

void opentrackio_receiver::start() { impl_->start(); }
void opentrackio_receiver::stop()  { impl_->stop(); }

std::string opentrackio_receiver::info() const
{
    return "OpenTrackIO multicast " + impl_->multicast_addr_ + " port " + std::to_string(impl_->port_);
}

}} // namespace caspar::tracking
