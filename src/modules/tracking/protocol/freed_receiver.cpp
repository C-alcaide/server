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
 * FreeD protocol specification by Mark Roberts Motion Control Ltd.
 * Implemented from the publicly available specification document.
 */

#include "freed_receiver.h"

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
// FreeD D1 packet layout (29 bytes, network byte order / big-endian)
// ---------------------------------------------------------------------------
//  Byte 0      : Message type (must be 0xD1)
//  Bytes  1-3  : Pan  (24-bit signed, 1/32768 degree per unit)
//  Bytes  4-6  : Tilt (24-bit signed, 1/32768 degree per unit)
//  Bytes  7-9  : Roll (24-bit signed, 1/32768 degree per unit)
//  Bytes 10-13 : X position (32-bit signed, 1/64 mm per unit)
//  Bytes 14-17 : Y position (32-bit signed, 1/64 mm per unit)
//  Bytes 18-21 : Z position (32-bit signed, 1/64 mm per unit)
//  Bytes 22-23 : Zoom  (16-bit unsigned)
//  Bytes 24-25 : Focus (16-bit unsigned)
//  Byte  26    : User bits — high nibble is device/camera ID (bits 7-4)
//  Bytes 27-28 : Checksum pad (byte 28 = 0x40 minus low byte of sum of bytes 0-27)
//
// The standard checksum formula used by most vendors:
//   (sum of all 29 bytes) & 0xFF == 0x40
// ---------------------------------------------------------------------------

static constexpr uint8_t  FREED_MSG_TYPE  = 0xD1;
static constexpr size_t   FREED_PACKET_LEN = 29;
static constexpr double   FREED_ANGLE_SCALE    = 1.0 / 32768.0; // degrees per unit  →  × (π/180) for radians
static constexpr double   FREED_POSITION_SCALE = 1.0 / 64.0;    // mm per unit
static constexpr double   DEG2RAD              = 3.141592653589793 / 180.0;

static int32_t decode_24bit_signed(const uint8_t* p)
{
    // Big-endian 24-bit → sign-extended 32-bit
    int32_t v = (static_cast<int32_t>(p[0]) << 16) |
                (static_cast<int32_t>(p[1]) <<  8) |
                (static_cast<int32_t>(p[2])      );
    if (v & 0x800000) // sign-extend
        v |= 0xFF000000;
    return v;
}

static uint32_t decode_24bit_unsigned(const uint8_t* p)
{
    return (static_cast<uint32_t>(p[0]) << 16) | (static_cast<uint32_t>(p[1]) << 8) |
           static_cast<uint32_t>(p[2]);
}
static bool verify_checksum(const std::array<uint8_t, FREED_PACKET_LEN>& buf)
{
    // (sum of all 29 bytes) & 0xFF == 0x40
    uint32_t sum = 0;
    for (uint8_t b : buf)
        sum += b;
    return (sum & 0xFF) == 0x40;
}

// ---------------------------------------------------------------------------

struct freed_receiver::impl
{
    const uint16_t port_;
    std::atomic<bool> running_{false};

    boost::asio::io_context                    io_;
    boost::asio::ip::udp::socket               socket_{io_};
    boost::asio::ip::udp::endpoint             sender_endpoint_;
    std::array<uint8_t, FREED_PACKET_LEN * 4>  recv_buf_{}; // slightly oversized to detect bad packets
    std::thread                                io_thread_;

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
                        std::cerr << "[tracking/freed] receive error: " << ec.message() << "\n";
                        // Recoverable errors (e.g. WSAEMSGSIZE from an oversized
                        // stray packet) must not permanently end the receive loop —
                        // re-arm so a single bad datagram doesn't kill tracking for
                        // the rest of the session.
                        if (running_)
                            start_receive();
                    }
                    return;
                }
                if (bytes_transferred == FREED_PACKET_LEN)
                    process_packet(bytes_transferred);
                if (running_)
                    start_receive();
            });
    }

    void process_packet(std::size_t len)
    {
        if (len != FREED_PACKET_LEN)
            return;
        if (recv_buf_[0] != FREED_MSG_TYPE)
            return;

        // Copy into fixed-size array for checksum check
        std::array<uint8_t, FREED_PACKET_LEN> pkt;
        std::copy_n(recv_buf_.begin(), FREED_PACKET_LEN, pkt.begin());

        if (!verify_checksum(pkt)) {
            std::cerr << "[tracking/freed] bad checksum - packet dropped\n";
            return;
        }

        camera_data d;

        // ── THE PUBLISHED D1 LAYOUT (FreeD, 29 bytes, big-endian) ─────────────────────────
        //
        //   [0]      0xD1                    [11..13] X     24-bit signed, 1/64 mm
        //   [1]      CAMERA ID, whole byte   [14..16] Y     24-bit signed, 1/64 mm
        //   [2..4]   pan   24-bit, 1/32768 deg  [17..19] Z  24-bit signed, 1/64 mm
        //   [5..7]   tilt                    [20..22] zoom  24-bit unsigned
        //   [8..10]  roll                    [23..25] focus 24-bit unsigned
        //   [26..27] spare / user            [28]     checksum
        //
        // THIS DECODER DID NOT READ THAT. It had no camera-ID byte, so pan started on the ID and
        // every field after byte 0 was misaligned; it read the positions as 32-bit words at
        // 10/14/18 and zoom and focus as 16-bit at 22/24, and took the camera ID from a nibble of
        // a spare byte. Self-consistent -- it adds up to 29 -- and matching no real tracker: a
        // Mo-Sys, Stype, Ncam or PTZ FreeD output decoded to garbage angles and positions, and
        // its camera ID came out 0 whatever the device sent. It went unnoticed because the
        // harness's packet builder was written FROM this decoder, so the two agreed byte for byte
        // -- the "wrong in the same direction" trap CLAUDE.md records for LTC. The field widths
        // are fixed independently by TouchDesigner's FreeD CHOP reference (camera ID 0-255,
        // positions +/-131.07 m = 2^23/64 mm, zoom and focus 24-bit), which no 32-bit position
        // or 16-bit zoom can produce.
        //
        // ZOOM AND FOCUS are narrowed 24 -> 16 bits, the width `camera_data` carries, by dropping
        // the low byte: the normalised range is kept (0..0xFFFFFF -> 0..0xFFFF), which is what
        // `ZOOM_LUT` and `SCALE`'s full-range calibrate against.
        //
        // FreeD's third position is HEIGHT (free-d manual Appendix B.7) and its second is the
        // direction a camera at zero pan faces, so the three go through `set_position_z_up`
        // rather than straight into x, y, z -- see camera_data.h for the frame and its sources.
        d.pan  = decode_24bit_signed(&pkt[2]) * FREED_ANGLE_SCALE * DEG2RAD;
        d.tilt = decode_24bit_signed(&pkt[5]) * FREED_ANGLE_SCALE * DEG2RAD;
        d.roll = decode_24bit_signed(&pkt[8]) * FREED_ANGLE_SCALE * DEG2RAD;

        set_position_z_up(d,
                          decode_24bit_signed(&pkt[11]) * FREED_POSITION_SCALE,
                          decode_24bit_signed(&pkt[14]) * FREED_POSITION_SCALE,
                          decode_24bit_signed(&pkt[17]) * FREED_POSITION_SCALE);

        d.zoom  = static_cast<uint16_t>(decode_24bit_unsigned(&pkt[20]) >> 8);
        d.focus = static_cast<uint16_t>(decode_24bit_unsigned(&pkt[23]) >> 8);

        d.camera_id = pkt[1];

        d.timestamp = std::chrono::steady_clock::now();

        tracker_registry::instance().on_data(d, tracking_protocol::freed, port_);
    }
};

// ---------------------------------------------------------------------------

freed_receiver::freed_receiver(uint16_t port)
    : impl_(std::make_unique<impl>(port))
{
}

freed_receiver::~freed_receiver() = default;

void freed_receiver::start()
{
    impl_->start();
}

void freed_receiver::stop()
{
    impl_->stop();
}

std::string freed_receiver::info() const
{
    return "FreeD D1 UDP port " + std::to_string(impl_->port_);
}

}} // namespace caspar::tracking
