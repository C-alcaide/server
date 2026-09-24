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
 * FreeD+ (Stype extended FreeD) protocol implemented from the
 * publicly available Stype specification document.
 */

#include "freed_plus_receiver.h"

#include "../camera_data.h"
#include "../tracker_registry.h"

#include <boost/asio.hpp>

#include <array>
#include <atomic>
#include <cmath>
#include <iostream>
#include <string>
#include <thread>

namespace caspar { namespace tracking {

// ---------------------------------------------------------------------------
// FreeD+ / Stype extended-precision packet (41 bytes, big-endian)
// ---------------------------------------------------------------------------
// Bytes 0-26   : Identical to standard FreeD D1 packet
// Bytes 27-30  : High-res pan   (32-bit signed, 1/(32768 * 256) deg/unit)
// Bytes 31-34  : High-res tilt  (same scale)
// Bytes 35-38  : High-res roll  (same scale)
// Byte  39     : User / equipment bits (high nibble = device ID)
// Byte  40     : Checksum — byte such that XOR of all 41 bytes == 0
// ---------------------------------------------------------------------------

static constexpr uint8_t  FREED_MSG_TYPE           = 0xD1;
static constexpr size_t   FREED_PLUS_PACKET_LEN    = 41;
static constexpr size_t   FREED_STD_PACKET_LEN     = 29;
static constexpr double   FREED_ANGLE_SCALE        = 1.0 / 32768.0;
static constexpr double   FREED_PLUS_ANGLE_SCALE   = 1.0 / (32768.0 * 256.0);
static constexpr double   FREED_POSITION_SCALE     = 1.0 / 64.0;
static constexpr double   DEG2RAD                  = 3.141592653589793 / 180.0;

static int32_t decode_24bit_signed_fp(const uint8_t* p)
{
    int32_t v = (static_cast<int32_t>(p[0]) << 16) |
                (static_cast<int32_t>(p[1]) <<  8) |
                (static_cast<int32_t>(p[2])      );
    if (v & 0x800000)
        v |= 0xFF000000;
    return v;
}

static int32_t decode_32bit_signed_fp(const uint8_t* p)
{
    return static_cast<int32_t>( (static_cast<uint32_t>(p[0]) << 24) |
                                  (static_cast<uint32_t>(p[1]) << 16) |
                                  (static_cast<uint32_t>(p[2]) <<  8) |
                                  (static_cast<uint32_t>(p[3])      ) );
}

static uint32_t decode_24bit_unsigned_fp(const uint8_t* p)
{
    return (static_cast<uint32_t>(p[0]) << 16) | (static_cast<uint32_t>(p[1]) << 8) |
           static_cast<uint32_t>(p[2]);
}
// ---------------------------------------------------------------------------

struct freed_plus_receiver::impl
{
    const uint16_t    port_;
    std::atomic<bool> running_{false};

    boost::asio::io_context     io_;
    boost::asio::ip::udp::socket socket_{io_};
    boost::asio::ip::udp::endpoint sender_endpoint_;
    std::array<uint8_t, FREED_PLUS_PACKET_LEN * 2> recv_buf_{};
    std::thread io_thread_;

    explicit impl(uint16_t port)
        : port_(port)
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
                    if (ec != boost::asio::error::operation_aborted) {
                        std::cerr << "[tracking/freed+] receive error: " << ec.message() << "\n";
                        // Recoverable error — re-arm rather than permanently
                        // ending tracking over a single bad datagram.
                        if (running_)
                            start_receive();
                    }
                    return;
                }

                if (bytes_transferred == FREED_PLUS_PACKET_LEN)
                    process_plus_packet(bytes_transferred);
                else if (bytes_transferred == FREED_STD_PACKET_LEN)
                    process_std_packet(bytes_transferred);

                if (running_)
                    start_receive();
            });
    }

    // Standard FreeD D1 fallback — no extended precision
    void process_std_packet(std::size_t)
    {
        if (recv_buf_[0] != FREED_MSG_TYPE)
            return;
        // The plain D1 CHECKSUM, which this fallback never verified while the 41-byte path verifies
        // its own: (sum of all 29 bytes) & 0xFF == 0x40. A corrupt datagram of the right length
        // used to drive the camera.
        unsigned sum = 0;
        for (std::size_t i = 0; i < FREED_STD_PACKET_LEN; ++i)
            sum += recv_buf_[i];
        if ((sum & 0xFF) != 0x40)
            return;

        camera_data d;
        // The plain D1 layout -- see freed_receiver.cpp, where the account of what this read
        // before, and why nothing caught it, is written once.
        d.pan  = decode_24bit_signed_fp(&recv_buf_[2]) * FREED_ANGLE_SCALE * DEG2RAD;
        d.tilt = decode_24bit_signed_fp(&recv_buf_[5]) * FREED_ANGLE_SCALE * DEG2RAD;
        d.roll = decode_24bit_signed_fp(&recv_buf_[8]) * FREED_ANGLE_SCALE * DEG2RAD;

        set_position_z_up(d,
                          decode_24bit_signed_fp(&recv_buf_[11]) * FREED_POSITION_SCALE,
                          decode_24bit_signed_fp(&recv_buf_[14]) * FREED_POSITION_SCALE,
                          decode_24bit_signed_fp(&recv_buf_[17]) * FREED_POSITION_SCALE);

        d.zoom  = static_cast<uint16_t>(decode_24bit_unsigned_fp(&recv_buf_[20]) >> 8);
        d.focus = static_cast<uint16_t>(decode_24bit_unsigned_fp(&recv_buf_[23]) >> 8);

        d.camera_id = recv_buf_[1];
        d.timestamp = std::chrono::steady_clock::now();

        tracker_registry::instance().on_data(d, tracking_protocol::freed_plus, port_);
    }

    // Full FreeD+ packet with 32-bit high-precision angles
    void process_plus_packet(std::size_t)
    {
        if (recv_buf_[0] != FREED_MSG_TYPE)
            return;

        // Verify XOR checksum: XOR of all 41 bytes == 0
        uint8_t xor_val = 0;
        for (size_t i = 0; i < FREED_PLUS_PACKET_LEN; ++i)
            xor_val ^= recv_buf_[i];
        if (xor_val != 0) {
            std::cerr << "[tracking/freed+] bad XOR checksum - packet dropped\n";
            return;
        }

        camera_data d;

        // Position / zoom / focus from the standard D1 header -- which this file's own layout
        // note says bytes 0-26 ARE, so they are read with the D1 offsets and widths. They were
        // read with the same misaligned 32/16-bit layout as the plain decoder.
        // FreeD's frame, not camera_data's: see camera_data.h.
        set_position_z_up(d,
                          decode_24bit_signed_fp(&recv_buf_[11]) * FREED_POSITION_SCALE,
                          decode_24bit_signed_fp(&recv_buf_[14]) * FREED_POSITION_SCALE,
                          decode_24bit_signed_fp(&recv_buf_[17]) * FREED_POSITION_SCALE);
        d.zoom  = static_cast<uint16_t>(decode_24bit_unsigned_fp(&recv_buf_[20]) >> 8);
        d.focus = static_cast<uint16_t>(decode_24bit_unsigned_fp(&recv_buf_[23]) >> 8);

        // High-precision angles from the extension block
        d.pan  = decode_32bit_signed_fp(&recv_buf_[27]) * FREED_PLUS_ANGLE_SCALE * DEG2RAD;
        d.tilt = decode_32bit_signed_fp(&recv_buf_[31]) * FREED_PLUS_ANGLE_SCALE * DEG2RAD;
        d.roll = decode_32bit_signed_fp(&recv_buf_[35]) * FREED_PLUS_ANGLE_SCALE * DEG2RAD;

        d.camera_id = (recv_buf_[39] >> 4) & 0x0F;
        d.timestamp = std::chrono::steady_clock::now();

        tracker_registry::instance().on_data(d, tracking_protocol::freed_plus, port_);
    }
};

// ---------------------------------------------------------------------------

freed_plus_receiver::freed_plus_receiver(uint16_t port)
    : impl_(std::make_unique<impl>(port))
{
}

freed_plus_receiver::~freed_plus_receiver() = default;

void freed_plus_receiver::start() { impl_->start(); }
void freed_plus_receiver::stop()  { impl_->stop(); }

std::string freed_plus_receiver::info() const
{
    return "FreeD+ (Stype) UDP port " + std::to_string(impl_->port_);
}

}} // namespace caspar::tracking
