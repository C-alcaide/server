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
 * Open Sound Control (OSC) 1.0 specification by CNMAT/UC Berkeley.
 * Implemented from the publicly available specification.
 */

#include "osc_receiver.h"

#include "../camera_data.h"
#include "../tracker_registry.h"

#include <boost/asio.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace caspar { namespace tracking {

// ---------------------------------------------------------------------------
// Minimal OSC 1.0 parser
// ---------------------------------------------------------------------------
// OSC packets start with the address pattern (null-padded to 4-byte boundary),
// then the type tag string (starting with ','), then arguments.
// We only handle ,f (float32) and ,d (float64) and ,i (int32) arguments.
// Integers are converted to float for uniformity.
// ---------------------------------------------------------------------------

static constexpr double DEG2RAD_OSC = 3.141592653589793 / 180.0;

/// Returns the 4-byte-aligned offset after a null-terminated string
static std::size_t osc_string_end(const uint8_t* data, std::size_t offset, std::size_t len)
{
    while (offset < len && data[offset] != 0)
        ++offset;
    ++offset; // skip null
    // pad to 4-byte boundary
    return (offset + 3) & ~std::size_t{3};
}

static uint32_t read_u32(const uint8_t* p)
{
    return (static_cast<uint32_t>(p[0]) << 24) | (static_cast<uint32_t>(p[1]) << 16) |
           (static_cast<uint32_t>(p[2]) <<  8)  |  static_cast<uint32_t>(p[3]);
}

static float read_f32(const uint8_t* p)
{
    uint32_t u = read_u32(p);
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

static uint64_t read_u64(const uint8_t* p)
{
    return (static_cast<uint64_t>(read_u32(p)) << 32) | read_u32(p + 4);
}

static double read_f64(const uint8_t* p)
{
    uint64_t u = read_u64(p);
    double d;
    std::memcpy(&d, &u, sizeof(d));
    return d;
}

/// Try to parse a single OSC message at [data+offset, data+len).
/// Returns true if matched; value is set to the first numeric argument.
static bool parse_osc_message(const uint8_t* data, std::size_t offset, std::size_t len,
                               std::string& out_addr, double& out_value)
{
    if (offset >= len)
        return false;

    // Address string
    const char* addr_c = reinterpret_cast<const char*>(data + offset);
    out_addr = addr_c;
    std::size_t tag_offset = osc_string_end(data, offset, len);

    if (tag_offset >= len || data[tag_offset] != ',')
        return false;

    // Type tag string
    const char* tags = reinterpret_cast<const char*>(data + tag_offset);
    std::size_t args_offset = osc_string_end(data, tag_offset, len);

    if (args_offset > len)
        return false;

    // Read first argument
    if (tags[1] == 'f' && args_offset + 4 <= len) {
        out_value = static_cast<double>(read_f32(data + args_offset));
        return true;
    } else if (tags[1] == 'd' && args_offset + 8 <= len) {
        out_value = read_f64(data + args_offset);
        return true;
    } else if (tags[1] == 'i' && args_offset + 4 <= len) {
        out_value = static_cast<double>(static_cast<int32_t>(read_u32(data + args_offset)));
        return true;
    }
    return false;
}

// ---------------------------------------------------------------------------

struct osc_receiver::impl
{
    const uint16_t    port_;
    std::atomic<bool> running_{false};

    boost::asio::io_context       io_;
    boost::asio::ip::udp::socket  socket_{io_};
    boost::asio::ip::udp::endpoint sender_endpoint_;
    std::vector<uint8_t>          recv_buf_;
    std::thread                   io_thread_;

    // Partial camera data accumulation: keyed by camera_id
    std::mutex                   partial_mutex_;
    std::map<int, camera_data>   partial_data_;

    explicit impl(uint16_t port)
        : port_(port)
    {
        recv_buf_.resize(8192);
    }

    ~impl() { stop(); }

    void start()
    {
        if (running_.exchange(true))
            return;

        socket_.open(boost::asio::ip::udp::v4());
        socket_.set_option(boost::asio::socket_base::reuse_address(true));
        socket_.bind(boost::asio::ip::udp::endpoint(boost::asio::ip::udp::v4(), port_));

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
                        std::cerr << "[tracking/osc] receive error: " << ec.message() << "\n";
                    return;
                }

                const uint8_t* data = recv_buf_.data();

                // OSC bundle starts with "#bundle\0"
                if (bytes > 8 && std::memcmp(data, "#bundle\0", 8) == 0) {
                    process_bundle(data, bytes);
                } else {
                    std::string addr;
                    double      val = 0.0;
                    if (parse_osc_message(data, 0, bytes, addr, val))
                        dispatch(addr, val);
                }

                if (running_)
                    start_receive();
            });
    }

    void process_bundle(const uint8_t* data, std::size_t len)
    {
        // Skip "#bundle\0" (8) + timetag (8) = offset 16
        std::size_t offset = 16;
        while (offset + 4 <= len) {
            uint32_t msg_size = read_u32(data + offset);
            offset += 4;
            if (offset + msg_size > len)
                break;
            std::string addr;
            double      val = 0.0;
            if (parse_osc_message(data, offset, offset + msg_size, addr, val))
                dispatch(addr, val);
            offset += msg_size;
        }
    }

    /// Parse addresses of the form /camera/{id}/pan and accumulate into partial_data_.
    /// When a /camera/{id}/x message arrives (assumed last in sequence), fire on_data.
    void dispatch(const std::string& addr, double val)
    {
        // Expected format: /camera/<id>/<field>
        if (addr.rfind("/camera/", 0) != 0)
            return;

        const std::string rest = addr.substr(8); // strip "/camera/"
        const auto slash = rest.find('/');
        if (slash == std::string::npos)
            return;

        int         camera_id = 0;
        std::string field;
        try {
            camera_id = std::stoi(rest.substr(0, slash));
            field     = rest.substr(slash + 1);
        } catch (...) {
            return;
        }

        std::lock_guard<std::mutex> lk(partial_mutex_);
        camera_data& d = partial_data_[camera_id];
        d.camera_id    = camera_id;

        if      (field == "pan")   d.pan   = val * DEG2RAD_OSC;
        else if (field == "tilt")  d.tilt  = val * DEG2RAD_OSC;
        else if (field == "roll")  d.roll  = val * DEG2RAD_OSC;
        else if (field == "x")     d.x     = val;
        else if (field == "y")     d.y     = val;
        else if (field == "z")     {
            d.z         = val;
            d.timestamp = std::chrono::steady_clock::now();
            // Fire after z — assume a sender bundles all values together;
            // sending z last is the trigger to push the complete frame.
            tracker_registry::instance().on_data(d);
        }
        else if (field == "zoom")  {
            // Accept 0-1 normalised or raw 0-65535
            double zoom = (val <= 1.0 && val >= 0.0) ? val * 65535.0 : val;
            d.zoom = static_cast<uint16_t>(std::clamp(zoom, 0.0, 65535.0));
        }
        else if (field == "focus") {
            double focus = (val <= 1.0 && val >= 0.0) ? val * 65535.0 : val;
            d.focus = static_cast<uint16_t>(std::clamp(focus, 0.0, 65535.0));
        }
    }
};

// ---------------------------------------------------------------------------

osc_receiver::osc_receiver(uint16_t port)
    : impl_(std::make_unique<impl>(port))
{
}

osc_receiver::~osc_receiver() = default;

void osc_receiver::start() { impl_->start(); }
void osc_receiver::stop()  { impl_->stop(); }

std::string osc_receiver::info() const
{
    return "OSC 1.0 UDP port " + std::to_string(impl_->port_);
}

}} // namespace caspar::tracking
