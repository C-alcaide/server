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

#include "sacn_consumer.h"

#undef NOMINMAX

#include <common/future.h>
#include <common/log.h>
#include <common/ptree.h>

#include <core/consumer/channel_info.h>

#include <boost/algorithm/string.hpp>
#include <boost/asio.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/uuid/random_generator.hpp>
#include <boost/uuid/uuid.hpp>

#include <array>
#include <cstring>
#include <thread>
#include <utility>
#include <vector>

using namespace boost::asio;
using namespace boost::asio::ip;

namespace caspar { namespace sacn {

// ANSI E1.31 (sACN) packet constants
static const uint8_t ACN_PACKET_IDENTIFIER[12] = {
    0x41, 0x53, 0x43, 0x2d, 0x45, 0x31, 0x2e, 0x31, 0x37, 0x00, 0x00, 0x00 // "ASC-E1.17\0\0\0"
};
static const uint32_t VECTOR_ROOT_E131_DATA    = 0x00000004;
static const uint32_t VECTOR_E131_DATA_PACKET  = 0x00000002;
static const uint8_t  VECTOR_DMP_SET_PROPERTY  = 0x02;
static const uint8_t  DMP_ADDRESS_DATA_TYPE    = 0xa1; // Range address, type 0xa1

// Multicast base address for sACN: 239.255.x.y where x.y = high/low byte of universe
static std::string multicast_address_for_universe(int universe)
{
    int hi = (universe >> 8) & 0xFF;
    int lo = universe & 0xFF;
    return "239.255." + std::to_string(hi) + "." + std::to_string(lo);
}

struct configuration
{
    int            universe    = 1;
    std::wstring   host        = L""; // empty = use multicast
    unsigned short port        = 5568;
    uint8_t        priority    = 100;
    int            multicastTtl = 10;
    int            refreshRate = 10;

    std::vector<fixture> fixtures;
};

struct sacn_consumer : public core::frame_consumer
{
    const configuration           config_;
    std::vector<computed_fixture> computed_fixtures_;

  public:
    explicit sacn_consumer(configuration config)
        : config_(std::move(config))
        , io_context_()
        , socket_(io_context_)
        , sequence_number_(0)
    {
        // Generate a random CID (Component Identifier — UUID v4) for this source instance
        boost::uuids::random_generator gen;
        boost::uuids::uuid             uuid = gen();
        std::copy(uuid.begin(), uuid.end(), cid_.begin());

        socket_.open(udp::v4());

        // Determine target endpoint: unicast if host is specified, multicast otherwise
        std::string host;
        if (!config_.host.empty()) {
            host = u8(config_.host);
        } else {
            host = multicast_address_for_universe(config_.universe);
            // Set multicast TTL so packets can traverse managed switches in a studio
            socket_.set_option(ip::multicast::hops(config_.multicastTtl));
        }

        remote_endpoint_ = udp::endpoint(make_address(host), config_.port);

        compute_fixtures();
    }

    void initialize(const core::video_format_desc& /*format_desc*/,
                    const core::channel_info& channel_info,
                    int                       port_index) override
    {
        thread_ = std::thread([this] {
            long long time      = 1000 / config_.refreshRate;
            auto      last_send = std::chrono::system_clock::now();

            while (!abort_request_) {
                try {
                    auto                          now             = std::chrono::system_clock::now();
                    std::chrono::duration<double> elapsed_seconds = now - last_send;
                    long long                     elapsed_ms =
                        std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_seconds).count();

                    long long sleep_time = time - elapsed_ms * 1000;
                    if (sleep_time > 0)
                        std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time));

                    last_send = now;

                    frame_mutex_.lock();
                    auto frame = last_frame_;
                    frame_mutex_.unlock();

                    if (!frame)
                        continue;

                    uint8_t dmx_data[512];
                    memset(dmx_data, 0, 512);

                    for (auto& cf : computed_fixtures_) {
                        auto     color = average_color(frame, cf.rectangle);
                        uint8_t* ptr   = dmx_data + cf.address;

                        switch (cf.type) {
                            case FixtureType::DIMMER:
                                ptr[0] = (uint8_t)(0.279f * color.r + 0.547f * color.g + 0.106f * color.b);
                                break;
                            case FixtureType::RGB:
                                ptr[0] = color.r;
                                ptr[1] = color.g;
                                ptr[2] = color.b;
                                break;
                            case FixtureType::RGBW: {
                                uint8_t w = std::min({color.r, color.g, color.b});
                                ptr[0]    = color.r - w;
                                ptr[1]    = color.g - w;
                                ptr[2]    = color.b - w;
                                ptr[3]    = w;
                                break;
                            }
                        }
                    }

                    send_sacn_data(dmx_data, 512);
                } catch (...) {
                    CASPAR_LOG_CURRENT_EXCEPTION();
                }
            }
        });
    }

    ~sacn_consumer()
    {
        abort_request_ = true;
        if (thread_.joinable())
            thread_.join();
    }

    std::future<bool> send(core::video_field field, core::const_frame frame) override
    {
        std::lock_guard<std::mutex> lock(frame_mutex_);
        last_frame_ = frame;
        return make_ready_future(true);
    }

    std::wstring print() const override { return L"sacn[universe=" + std::to_wstring(config_.universe) + L"]"; }

    std::wstring name() const override { return L"sacn"; }

    int index() const override { return 1338; }

    core::monitor::state state() const override
    {
        core::monitor::state state;
        state["sacn/universe"]     = config_.universe;
        state["sacn/host"]         = config_.host;
        state["sacn/port"]         = config_.port;
        state["sacn/priority"]     = (int)config_.priority;
        state["sacn/refresh-rate"] = config_.refreshRate;
        state["sacn/fixtures"]     = config_.fixtures.size();
        return state;
    }

  private:
    core::const_frame last_frame_;
    std::mutex        frame_mutex_;

    std::thread       thread_;
    std::atomic<bool> abort_request_{false};

    io_context    io_context_;
    udp::socket   socket_;
    udp::endpoint remote_endpoint_;

    std::array<uint8_t, 16> cid_;
    uint8_t                 sequence_number_;

    void compute_fixtures()
    {
        computed_fixtures_.clear();
        for (auto& f : config_.fixtures) {
            for (unsigned short i = 0; i < f.fixtureCount; i++) {
                computed_fixture cf{};
                cf.type      = f.type;
                cf.address   = f.startAddress + i * f.fixtureChannels;
                cf.rectangle = compute_rect(f.fixtureBox, i, f.fixtureCount);
                computed_fixtures_.push_back(cf);
            }
        }
    }

    // Build and send an E1.31 (sACN) Data Packet per ANSI E1.31-2018
    void send_sacn_data(const uint8_t* dmx_data, std::size_t dmx_length)
    {
        // Packet layout (fixed at 638 bytes for 512 DMX channels):
        //
        //  Bytes   0-15  : Preamble (2) + Postamble (2) + ACN Identifier (12)
        //  Bytes  16-37  : Root PDU  — Flags&Length (2) + Vector (4) + CID (16)  [64 + 16 = 22 bytes payload]
        //  Bytes  38-114 : Framing PDU — Flags&Length (2) + Vector (4) + SourceName (64) + Priority (1)
        //                              + SyncAddr (2) + Sequence (1) + Options (1) + Universe (2)  = 77 bytes
        //  Bytes 115-124 : DMP PDU header — Flags&Length (2) + Vector (1) + AddrType (1)
        //                                  + FirstAddr (2) + Increment (2) + PropCount (2)  = 10 bytes
        //  Bytes 125-637 : Start code (1) + 512 DMX bytes = 513 bytes
        //
        // PDU "length" field includes the 2-byte Flags&Length field itself.
        //   DMP PDU length     = 10 + 513 = 523
        //   Framing PDU length = 77 + 523 = 600
        //   Root PDU length    = 22 + 600 = 622

        static const std::size_t SACN_PACKET_SIZE = 638;
        static const std::size_t PROP_COUNT       = 513; // start code + 512 channels
        static const uint16_t    DMP_PDU_LEN      = 523;
        static const uint16_t    FRAMING_PDU_LEN  = 600;
        static const uint16_t    ROOT_PDU_LEN     = 622;

        uint8_t buffer[SACN_PACKET_SIZE];
        memset(buffer, 0, SACN_PACKET_SIZE);

        int off = 0;

        // --- Preamble Size (2 bytes) ---
        buffer[off++] = 0x00;
        buffer[off++] = 0x10;

        // --- Postamble Size (2 bytes) ---
        buffer[off++] = 0x00;
        buffer[off++] = 0x00;

        // --- ACN Packet Identifier (12 bytes) ---
        memcpy(buffer + off, ACN_PACKET_IDENTIFIER, 12);
        off += 12;

        // --- Root PDU: Flags & Length (2 bytes) ---
        // Upper nibble 0x7 = length is 2 bytes; lower 12 bits = length
        buffer[off++] = static_cast<uint8_t>(0x70 | ((ROOT_PDU_LEN >> 8) & 0x0F));
        buffer[off++] = static_cast<uint8_t>(ROOT_PDU_LEN & 0xFF);

        // --- Root PDU: Vector (4 bytes) = VECTOR_ROOT_E131_DATA = 0x00000004 ---
        buffer[off++] = 0x00;
        buffer[off++] = 0x00;
        buffer[off++] = 0x00;
        buffer[off++] = 0x04;

        // --- Root PDU: CID (16 bytes) ---
        memcpy(buffer + off, cid_.data(), 16);
        off += 16;

        // --- Framing PDU: Flags & Length (2 bytes) ---
        buffer[off++] = static_cast<uint8_t>(0x70 | ((FRAMING_PDU_LEN >> 8) & 0x0F));
        buffer[off++] = static_cast<uint8_t>(FRAMING_PDU_LEN & 0xFF);

        // --- Framing PDU: Vector (4 bytes) = VECTOR_E131_DATA_PACKET = 0x00000002 ---
        buffer[off++] = 0x00;
        buffer[off++] = 0x00;
        buffer[off++] = 0x00;
        buffer[off++] = 0x02;

        // --- Framing PDU: Source Name (64 bytes, null-terminated UTF-8) ---
        static const char SOURCE_NAME[] = "CasparCG sACN";
        memcpy(buffer + off, SOURCE_NAME, sizeof(SOURCE_NAME) - 1);
        off += 64;

        // --- Framing PDU: Priority (1 byte) ---
        buffer[off++] = config_.priority;

        // --- Framing PDU: Synchronization Address (2 bytes) = 0 (no sync) ---
        buffer[off++] = 0x00;
        buffer[off++] = 0x00;

        // --- Framing PDU: Sequence Number (1 byte, wraps 0-255) ---
        buffer[off++] = sequence_number_++;

        // --- Framing PDU: Options (1 byte) = 0 ---
        buffer[off++] = 0x00;

        // --- Framing PDU: Universe (2 bytes, big-endian) ---
        buffer[off++] = static_cast<uint8_t>((config_.universe >> 8) & 0xFF);
        buffer[off++] = static_cast<uint8_t>(config_.universe & 0xFF);

        // --- DMP PDU: Flags & Length (2 bytes) ---
        buffer[off++] = static_cast<uint8_t>(0x70 | ((DMP_PDU_LEN >> 8) & 0x0F));
        buffer[off++] = static_cast<uint8_t>(DMP_PDU_LEN & 0xFF);

        // --- DMP PDU: Vector (1 byte) = VECTOR_DMP_SET_PROPERTY = 0x02 ---
        buffer[off++] = VECTOR_DMP_SET_PROPERTY;

        // --- DMP PDU: Address Type & Data Type (1 byte) = 0xa1 ---
        buffer[off++] = DMP_ADDRESS_DATA_TYPE;

        // --- DMP PDU: First Property Address (2 bytes) = 0x0000 ---
        buffer[off++] = 0x00;
        buffer[off++] = 0x00;

        // --- DMP PDU: Address Increment (2 bytes) = 0x0001 ---
        buffer[off++] = 0x00;
        buffer[off++] = 0x01;

        // --- DMP PDU: Property Count (2 bytes) = 513 (start code + 512 channels) ---
        buffer[off++] = static_cast<uint8_t>((PROP_COUNT >> 8) & 0xFF);
        buffer[off++] = static_cast<uint8_t>(PROP_COUNT & 0xFF);

        // --- DMX Start Code (1 byte) = 0x00 ---
        buffer[off++] = 0x00;

        // --- DMX Data (512 bytes) ---
        memcpy(buffer + off, dmx_data, dmx_length);
        off += static_cast<int>(dmx_length);

        // off == 638 at this point

        boost::system::error_code err;
        socket_.send_to(boost::asio::buffer(buffer, SACN_PACKET_SIZE), remote_endpoint_, 0, err);
        if (err)
            CASPAR_THROW_EXCEPTION(io_error() << msg_info(err.message()));
    }
};

static std::vector<fixture> get_fixtures_ptree(const boost::property_tree::wptree& ptree)
{
    std::vector<fixture> fixtures;

    using boost::property_tree::wptree;

    for (auto& xml_channel : ptree | witerate_children(L"fixtures") | welement_context_iteration) {
        ptree_verify_element_name(xml_channel, L"fixture");

        fixture f{};

        int startAddress = xml_channel.second.get(L"start-address", 0);
        if (startAddress < 1)
            CASPAR_THROW_EXCEPTION(user_error() << msg_info(L"Fixture start address must be specified"));

        f.startAddress = (unsigned short)startAddress - 1;

        int fixtureCount = xml_channel.second.get(L"fixture-count", -1);
        if (fixtureCount < 1)
            CASPAR_THROW_EXCEPTION(user_error() << msg_info(L"Fixture count must be specified"));

        f.fixtureCount = (unsigned short)fixtureCount;

        std::wstring type = xml_channel.second.get(L"type", L"");
        if (type.empty())
            CASPAR_THROW_EXCEPTION(user_error() << msg_info(L"Fixture type must be specified"));

        if (boost::iequals(type, L"DIMMER")) {
            f.type = FixtureType::DIMMER;
        } else if (boost::iequals(type, L"RGB")) {
            f.type = FixtureType::RGB;
        } else if (boost::iequals(type, L"RGBW")) {
            f.type = FixtureType::RGBW;
        } else {
            CASPAR_THROW_EXCEPTION(user_error() << msg_info(L"Unknown fixture type"));
        }

        int fixtureChannels = xml_channel.second.get(L"fixture-channels", -1);
        if (fixtureChannels < 0)
            fixtureChannels = f.type;
        if (fixtureChannels < f.type)
            CASPAR_THROW_EXCEPTION(
                user_error() << msg_info(
                    L"Fixture channel count must be at least enough channels for current color mode"));

        f.fixtureChannels = (unsigned short)fixtureChannels;

        box b{};

        b.x        = xml_channel.second.get(L"x", 0.0f);
        b.y        = xml_channel.second.get(L"y", 0.0f);
        b.width    = xml_channel.second.get(L"width", 0.0f);
        b.height   = xml_channel.second.get(L"height", 0.0f);
        b.rotation = xml_channel.second.get(L"rotation", 0.0f);

        f.fixtureBox = b;

        fixtures.push_back(f);
    }

    return fixtures;
}

spl::shared_ptr<core::frame_consumer>
create_preconfigured_consumer(const boost::property_tree::wptree&                      ptree,
                              const core::video_format_repository&                     format_repository,
                              const std::vector<spl::shared_ptr<core::video_channel>>& channels,
                              const core::channel_info&                                channel_info)
{
    if (channel_info.depth != common::bit_depth::bit8)
        CASPAR_THROW_EXCEPTION(caspar_exception() << msg_info("sACN consumer only supports 8-bit color depth."));

    configuration config;

    config.universe     = ptree.get(L"universe", config.universe);
    config.host         = ptree.get(L"host", config.host);
    config.port         = ptree.get(L"port", config.port);
    config.priority     = static_cast<uint8_t>(ptree.get(L"priority", (int)config.priority));
    config.multicastTtl = ptree.get(L"multicast-ttl", config.multicastTtl);
    config.refreshRate  = ptree.get(L"refresh-rate", config.refreshRate);

    if (config.universe < 1 || config.universe > 63999)
        CASPAR_THROW_EXCEPTION(user_error() << msg_info(L"sACN universe must be between 1 and 63999"));

    if (config.priority < 1 || config.priority > 200)
        CASPAR_THROW_EXCEPTION(user_error() << msg_info(L"sACN priority must be between 1 and 200"));

    if (config.refreshRate < 1)
        CASPAR_THROW_EXCEPTION(user_error() << msg_info(L"Refresh rate must be at least 1"));

    config.fixtures = get_fixtures_ptree(ptree);

    return spl::make_shared<sacn_consumer>(config);
}

}} // namespace caspar::sacn
