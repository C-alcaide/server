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

#include "dmx_producer.h"

#undef NOMINMAX

#include <common/env.h>
#include <common/except.h>
#include <common/filesystem.h>
#include <common/log.h>
#include <common/param.h>
#include <common/utf.h>

#include <core/frame/draw_frame.h>
#include <core/monitor/monitor.h>
#include <core/video_format.h>

#include <boost/algorithm/string.hpp>
#include <boost/asio.hpp>
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

using namespace boost::asio;
using namespace boost::asio::ip;

namespace caspar { namespace artnet {

// ─── NDJSON data model ───────────────────────────────────────────────

struct dmx_frame
{
    int64_t              dt;        // microseconds from recording start
    std::vector<uint8_t> data;      // 512 channel values
    int                  universe;  // universe number (extension)
};

struct dmx_recording
{
    std::string              copyright;
    std::string              version;
    std::vector<int>         channels;
    std::vector<dmx_frame>   frames;
    int64_t                  duration_us = 0; // cached
};

// ─── NDJSON loader ───────────────────────────────────────────────────

static dmx_recording load_ndjson(const std::wstring& path)
{
    dmx_recording recording;
    std::ifstream file(u8(path));
    if (!file.is_open()) {
        CASPAR_THROW_EXCEPTION(file_not_found() << msg_info(u8(L"Cannot open DMX file: " + path)));
    }

    std::string line;
    int         line_no = 0;

    while (std::getline(file, line)) {
        if (line.empty())
            continue;

        // Parse JSON using boost property_tree
        std::istringstream          iss(line);
        boost::property_tree::ptree pt;
        try {
            boost::property_tree::read_json(iss, pt);
        } catch (const std::exception& e) {
            CASPAR_LOG(warning) << L"[dmx_producer] Skipping malformed JSON at line " << line_no << L": " << e.what();
            line_no++;
            continue;
        }

        if (line_no == 0) {
            // Header line
            auto copyright_opt = pt.get_optional<std::string>("copyright");
            auto version_opt   = pt.get_optional<std::string>("version");

            if (copyright_opt) {
                recording.copyright = *copyright_opt;
                recording.version   = version_opt.value_or("v01");

                // Parse channels array
                auto channels_node = pt.get_child_optional("channels");
                if (channels_node) {
                    for (auto& ch : *channels_node) {
                        recording.channels.push_back(ch.second.get_value<int>());
                    }
                }

                line_no++;
                continue;
            }
            // If first line has no copyright, treat it as a frame (non-WATCHOUT format)
        }

        // Frame line
        auto dt_opt   = pt.get_optional<int64_t>("dt");
        auto data_opt = pt.get_child_optional("data");

        if (dt_opt && data_opt) {
            dmx_frame frame;
            frame.dt       = *dt_opt;
            frame.universe = pt.get<int>("universe", 0);

            frame.data.reserve(512);
            for (auto& val : *data_opt) {
                frame.data.push_back(static_cast<uint8_t>(
                    std::clamp(val.second.get_value<int>(), 0, 255)));
            }
            // Pad to 512
            frame.data.resize(512, 0);

            recording.frames.push_back(std::move(frame));
        }

        line_no++;
    }

    // Sort by timestamp (should already be sorted, but be safe)
    std::sort(recording.frames.begin(), recording.frames.end(),
              [](const dmx_frame& a, const dmx_frame& b) { return a.dt < b.dt; });

    // Cache duration
    if (!recording.frames.empty()) {
        recording.duration_us = recording.frames.back().dt - recording.frames.front().dt;
    }

    return recording;
}

// ─── Art-Net packet builder ──────────────────────────────────────────

static void build_artnet_packet(uint8_t* buffer, int universe, const uint8_t* data, int length, uint8_t sequence)
{
    // Art-Net header: "Art-Net\0"
    buffer[0] = 'A'; buffer[1] = 'r'; buffer[2] = 't';
    buffer[3] = '-'; buffer[4] = 'N'; buffer[5] = 'e';
    buffer[6] = 't'; buffer[7] = 0;

    // OpCode: OpDmx (0x5000) little-endian
    buffer[8]  = 0x00;
    buffer[9]  = 0x50;

    // Protocol version (14) big-endian
    buffer[10] = 0x00;
    buffer[11] = 14;

    // Sequence
    buffer[12] = sequence;

    // Physical port
    buffer[13] = 0;

    // Universe (SubUni + Net)
    buffer[14] = static_cast<uint8_t>(universe & 0xFF);
    buffer[15] = static_cast<uint8_t>((universe >> 8) & 0x7F);

    // Length big-endian
    buffer[16] = static_cast<uint8_t>((length >> 8) & 0xFF);
    buffer[17] = static_cast<uint8_t>(length & 0xFF);

    // DMX data
    std::memcpy(buffer + 18, data, std::min(length, 512));
}

// ─── sACN / E1.31 packet builder ─────────────────────────────────────

static const uint8_t ACN_PACKET_IDENTIFIER[12] = {
    0x41, 0x53, 0x43, 0x2d, 0x45, 0x31, 0x2e, 0x31, 0x37, 0x00, 0x00, 0x00 // "ASC-E1.17\0\0\0"
};

static std::string sacn_multicast_address(int universe)
{
    return "239.255." + std::to_string((universe >> 8) & 0xFF) + "." + std::to_string(universe & 0xFF);
}

static void build_sacn_packet(uint8_t* buffer, int universe, const uint8_t* data, int length,
                              uint8_t sequence, const uint8_t* cid, uint8_t priority = 100)
{
    static const std::size_t SACN_PACKET_SIZE = 638;
    static const uint16_t    PROP_COUNT       = 513;
    static const uint16_t    DMP_PDU_LEN      = 523;
    static const uint16_t    FRAMING_PDU_LEN  = 600;
    static const uint16_t    ROOT_PDU_LEN     = 622;

    std::memset(buffer, 0, SACN_PACKET_SIZE);
    int off = 0;

    // Preamble
    buffer[off++] = 0x00; buffer[off++] = 0x10;
    // Postamble
    buffer[off++] = 0x00; buffer[off++] = 0x00;
    // ACN Packet Identifier
    std::memcpy(buffer + off, ACN_PACKET_IDENTIFIER, 12); off += 12;

    // Root PDU: Flags & Length
    buffer[off++] = static_cast<uint8_t>(0x70 | ((ROOT_PDU_LEN >> 8) & 0x0F));
    buffer[off++] = static_cast<uint8_t>(ROOT_PDU_LEN & 0xFF);
    // Root PDU: Vector (VECTOR_ROOT_E131_DATA = 0x00000004)
    buffer[off++] = 0x00; buffer[off++] = 0x00; buffer[off++] = 0x00; buffer[off++] = 0x04;
    // Root PDU: CID (16 bytes)
    std::memcpy(buffer + off, cid, 16); off += 16;

    // Framing PDU: Flags & Length
    buffer[off++] = static_cast<uint8_t>(0x70 | ((FRAMING_PDU_LEN >> 8) & 0x0F));
    buffer[off++] = static_cast<uint8_t>(FRAMING_PDU_LEN & 0xFF);
    // Framing PDU: Vector (VECTOR_E131_DATA_PACKET = 0x00000002)
    buffer[off++] = 0x00; buffer[off++] = 0x00; buffer[off++] = 0x00; buffer[off++] = 0x02;
    // Framing PDU: Source Name (64 bytes)
    static const char SRC_NAME[] = "CasparCG DMX Producer";
    std::memcpy(buffer + off, SRC_NAME, sizeof(SRC_NAME) - 1); off += 64;
    // Priority
    buffer[off++] = priority;
    // Sync address (0 = no sync)
    buffer[off++] = 0x00; buffer[off++] = 0x00;
    // Sequence
    buffer[off++] = sequence;
    // Options
    buffer[off++] = 0x00;
    // Universe (big-endian)
    buffer[off++] = static_cast<uint8_t>((universe >> 8) & 0xFF);
    buffer[off++] = static_cast<uint8_t>(universe & 0xFF);

    // DMP PDU: Flags & Length
    buffer[off++] = static_cast<uint8_t>(0x70 | ((DMP_PDU_LEN >> 8) & 0x0F));
    buffer[off++] = static_cast<uint8_t>(DMP_PDU_LEN & 0xFF);
    // DMP Vector
    buffer[off++] = 0x02;
    // Address & Data Type
    buffer[off++] = 0xa1;
    // First Property Address
    buffer[off++] = 0x00; buffer[off++] = 0x00;
    // Address Increment
    buffer[off++] = 0x00; buffer[off++] = 0x01;
    // Property Count
    buffer[off++] = static_cast<uint8_t>((PROP_COUNT >> 8) & 0xFF);
    buffer[off++] = static_cast<uint8_t>(PROP_COUNT & 0xFF);
    // Start code
    buffer[off++] = 0x00;
    // DMX data
    std::memcpy(buffer + off, data, std::min(length, 512));
}

// ─── Protocol enum ──────────────────────────────────────────────────

enum class dmx_protocol { artnet, sacn };

// ─── DMX File Producer ───────────────────────────────────────────────

struct dmx_file_producer : public core::frame_producer
{
    const std::wstring                         filename_;
    const spl::shared_ptr<core::frame_factory> frame_factory_;
    const core::video_format_desc              format_desc_;

    dmx_recording recording_;

    // Network output
    boost::asio::io_context io_context_;
    udp::socket             socket_;
    udp::endpoint           remote_endpoint_;
    uint8_t                 sequence_     = 1;
    dmx_protocol            protocol_     = dmx_protocol::artnet;
    uint8_t                 sacn_cid_[16] = {};
    uint8_t                 sacn_priority_ = 100;

    // Playback state
    int64_t  base_dt_         = 0;  // timestamp of first frame
    int      last_idx_        = -1; // last sent frame index (to avoid redundant sends)
    uint32_t total_frames_    = 0;  // total producer frames (based on channel fps)
    int64_t  position_us_     = 0;  // current playback position (microseconds)
    int64_t  seek_offset_us_  = 0;  // offset applied by seek commands
    uint32_t frame_at_seek_   = 0;  // frame_number() at last seek

    // Configuration
    int          universe_filter_ = -1; // -1 = send all universes
    std::wstring host_;
    unsigned short port_;

    dmx_file_producer(const core::frame_producer_dependencies& dependencies,
                      const std::wstring&                      path,
                      const std::wstring&                      host,
                      unsigned short                           port,
                      int                                      universe_filter,
                      dmx_protocol                             protocol,
                      uint8_t                                  sacn_priority)
        : frame_factory_(dependencies.frame_factory)
        , format_desc_(dependencies.format_desc)
        , filename_(path)
        , socket_(io_context_)
        , host_(host)
        , port_(port)
        , universe_filter_(universe_filter)
        , protocol_(protocol)
        , sacn_priority_(sacn_priority)
    {
        // Load NDJSON file
        recording_ = load_ndjson(path);

        if (recording_.frames.empty()) {
            CASPAR_THROW_EXCEPTION(user_error() << msg_info(u8(L"DMX file contains no frames: " + path)));
        }

        base_dt_ = recording_.frames.front().dt;

        // Calculate total producer frames based on channel framerate
        double duration_s    = static_cast<double>(recording_.duration_us) / 1000000.0;
        double channel_fps   = format_desc_.hz; // progressive tick rate
        total_frames_        = static_cast<uint32_t>(std::ceil(duration_s * channel_fps));

        // Setup UDP socket
        socket_.open(udp::v4());
        socket_.set_option(udp::socket::reuse_address(true));
        socket_.set_option(boost::asio::socket_base::broadcast(true));

        std::string host_str = u8(host_);

        if (protocol_ == dmx_protocol::sacn) {
            // Generate a static CID for this producer instance
            // Use a deterministic hash from filename so the same file always gets the same CID
            std::hash<std::string> hasher;
            auto h = hasher(u8(path));
            std::memset(sacn_cid_, 0, 16);
            std::memcpy(sacn_cid_, &h, std::min(sizeof(h), std::size_t(16)));
            sacn_cid_[6] = (sacn_cid_[6] & 0x0F) | 0x40; // UUID v4
            sacn_cid_[8] = (sacn_cid_[8] & 0x3F) | 0x80; // variant 1

            // sACN: if no host specified, use multicast for the universe
            if (host_str.empty() || host_str == "255.255.255.255") {
                int uni = universe_filter_ >= 0 ? universe_filter_ : 1;
                host_str = sacn_multicast_address(uni);
                socket_.set_option(boost::asio::ip::multicast::hops(10));
            }
        }

        remote_endpoint_ = udp::endpoint(boost::asio::ip::make_address(host_str), port_);

        std::wstring proto_name = protocol_ == dmx_protocol::artnet ? L"Art-Net" : L"sACN";
        CASPAR_LOG(info) << print() << L" Loaded " << recording_.frames.size()
                         << L" DMX frames, duration " << std::fixed << std::setprecision(1) << duration_s
                         << L"s, " << total_frames_ << L" channel frames @ " << channel_fps
                         << L" Hz, output: " << proto_name << L" → " << u16(host_str) << L":" << port_;
    }

    ~dmx_file_producer()
    {
        // Send blackout on stop
        try {
            uint8_t zeros[512] = {};
            send_dmx(universe_filter_ >= 0 ? universe_filter_ : 0, zeros, 512);
        } catch (...) {
        }
    }

    // ── frame_producer interface ──

    core::draw_frame receive_impl(const core::video_field /*field*/, int /*nb_samples*/) override
    {
        // Calculate current time position using own tracking (survives seek)
        uint32_t frames_since_seek = frame_number() - frame_at_seek_;
        double   channel_fps       = format_desc_.hz;
        int64_t  current_us        = seek_offset_us_
                                   + static_cast<int64_t>((static_cast<double>(frames_since_seek) / channel_fps) * 1000000.0);

        // Clamp to recording duration
        if (current_us > recording_.duration_us)
            current_us = recording_.duration_us;

        position_us_ = current_us;

        // Find the NDJSON frame for this time position using binary search
        int64_t target_dt = base_dt_ + current_us;

        // Binary search: find last frame with dt <= target_dt
        auto it = std::upper_bound(
            recording_.frames.begin(), recording_.frames.end(), target_dt,
            [](int64_t t, const dmx_frame& f) { return t < f.dt; });

        int idx = 0;
        if (it != recording_.frames.begin()) {
            idx = static_cast<int>(std::distance(recording_.frames.begin(), it) - 1);
        }

        // Only send if we moved to a new NDJSON frame
        if (idx != last_idx_) {
            last_idx_ = idx;
            const auto& frame = recording_.frames[idx];

            // Send Art-Net packet(s)
            if (universe_filter_ < 0) {
                // Send this frame's universe
                send_dmx(frame.universe, frame.data.data(), static_cast<int>(frame.data.size()));
            } else if (frame.universe == universe_filter_) {
                send_dmx(frame.universe, frame.data.data(), static_cast<int>(frame.data.size()));
            }
        }

        // Return empty/transparent frame — DMX producer has no visual output
        return core::draw_frame::empty();
    }

    core::draw_frame last_frame(const core::video_field /*field*/) override
    {
        return core::draw_frame::empty();
    }

    core::draw_frame first_frame(const core::video_field /*field*/) override
    {
        return core::draw_frame::empty();
    }

    bool is_ready() override { return true; }

    uint32_t nb_frames() const override { return total_frames_; }

    std::future<std::wstring> call(const std::vector<std::wstring>& params) override
    {
        std::wstring result;
        std::wstring cmd = params.at(0);

        if (boost::iequals(cmd, L"seek") && params.size() > 1) {
            int64_t seek_frame = 0;
            std::wstring value = params.at(1);

            if (boost::iequals(value, L"rel")) {
                // Seek relative to current position (expects offset in params[2])
                double channel_fps = format_desc_.hz;
                seek_frame = static_cast<int64_t>(position_us_ / 1000000.0 * channel_fps);
            } else if (boost::iequals(value, L"in")) {
                seek_frame = 0;
            } else if (boost::iequals(value, L"out") || boost::iequals(value, L"end")) {
                seek_frame = total_frames_;
            } else {
                seek_frame = boost::lexical_cast<int64_t>(value);
            }

            if (params.size() > 2) {
                seek_frame += boost::lexical_cast<int64_t>(params.at(2));
            }

            // Convert frame number to microseconds
            double channel_fps = format_desc_.hz;
            seek_offset_us_ = static_cast<int64_t>((static_cast<double>(seek_frame) / channel_fps) * 1000000.0);
            seek_offset_us_ = std::clamp(seek_offset_us_, int64_t(0), recording_.duration_us);
            frame_at_seek_  = frame_number();
            last_idx_       = -1; // force re-send at new position

            result = std::to_wstring(seek_frame);

            CASPAR_LOG(info) << print() << L" Seek to frame " << seek_frame
                             << L" (" << std::fixed << std::setprecision(1)
                             << (seek_offset_us_ / 1000000.0) << L"s)";
        } else {
            CASPAR_THROW_EXCEPTION(not_implemented());
        }

        auto promise = std::promise<std::wstring>();
        promise.set_value(result);
        return promise.get_future();
    }

    std::wstring print() const override
    {
        return L"dmx[" + boost::filesystem::path(filename_).filename().wstring() + L"]";
    }

    std::wstring name() const override { return L"dmx"; }

    core::monitor::state state() const override
    {
        core::monitor::state state;
        state["dmx/file"]      = filename_;
        state["dmx/frames"]    = static_cast<int>(recording_.frames.size());
        state["dmx/duration"]  = static_cast<double>(recording_.duration_us) / 1000000.0;
        state["dmx/universe"]  = universe_filter_;
        state["dmx/protocol"]  = protocol_ == dmx_protocol::artnet ? std::wstring(L"artnet") : std::wstring(L"sacn");
        state["dmx/host"]      = host_;
        state["dmx/port"]      = static_cast<int>(port_);
        state["dmx/last-idx"]  = last_idx_;
        state["dmx/position"]  = static_cast<double>(position_us_) / 1000000.0;
        return state;
    }

  private:
    void send_dmx(int universe, const uint8_t* data, int length)
    {
        boost::system::error_code err;

        if (protocol_ == dmx_protocol::artnet) {
            uint8_t buffer[18 + 512] = {};
            int     send_len = std::min(length, 512);
            build_artnet_packet(buffer, universe, data, send_len, sequence_);
            sequence_ = (sequence_ + 1) | 1; // keep 1-255, skip 0
            socket_.send_to(boost::asio::buffer(buffer, 18 + send_len), remote_endpoint_, 0, err);
        } else {
            uint8_t buffer[638] = {};
            build_sacn_packet(buffer, universe, data, std::min(length, 512),
                              sequence_++, sacn_cid_, sacn_priority_);
            socket_.send_to(boost::asio::buffer(buffer, 638), remote_endpoint_, 0, err);
        }

        if (err) {
            CASPAR_LOG(warning) << print() << L" DMX send error: " << err.message();
        }
    }
};

// ─── File validation ─────────────────────────────────────────────────

static bool is_valid_ndjson(const boost::filesystem::path& path)
{
    auto ext = path.extension().wstring();
    boost::to_lower(ext);
    return ext == L".ndjson";
}

// ─── Factory function ────────────────────────────────────────────────

spl::shared_ptr<core::frame_producer> create_dmx_producer(const core::frame_producer_dependencies& dependencies,
                                                          const std::vector<std::wstring>&         params)
{
    if (params.empty())
        return core::frame_producer::empty();

    // Check for [DMX] prefix or .ndjson file extension
    bool explicit_dmx = boost::iequals(params.at(0), L"[DMX]");

    std::wstring path_param;
    if (explicit_dmx) {
        if (params.size() < 2)
            return core::frame_producer::empty();
        path_param = params.at(1);
    } else {
        path_param = params.at(0);
    }

    // Try to find the file
    auto filename = find_file_within_dir_or_absolute(env::media_folder(), path_param, is_valid_ndjson);
    if (!filename && !explicit_dmx) {
        return core::frame_producer::empty();
    }
    if (!filename) {
        // Try with .ndjson extension appended
        filename = find_file_within_dir_or_absolute(env::media_folder(), path_param + L".ndjson", is_valid_ndjson);
    }
    if (!filename) {
        if (explicit_dmx) {
            CASPAR_THROW_EXCEPTION(file_not_found() << msg_info(u8(L"DMX file not found: " + path_param)));
        }
        return core::frame_producer::empty();
    }

    // Parse optional parameters
    auto host     = get_param(L"HOST", params, L"255.255.255.255");
    auto port_val  = get_param(L"PORT", params, static_cast<unsigned short>(0));
    auto universe  = get_param(L"UNIVERSE", params, -1);
    auto priority  = static_cast<uint8_t>(get_param(L"PRIORITY", params, 100));

    // Protocol selection
    auto proto_str = get_param(L"PROTOCOL", params, L"ARTNET");
    dmx_protocol protocol = dmx_protocol::artnet;
    if (boost::iequals(proto_str, L"SACN") || boost::iequals(proto_str, L"E131")
        || boost::iequals(proto_str, L"E1.31")) {
        protocol = dmx_protocol::sacn;
    }

    // Default port based on protocol
    unsigned short port = port_val > 0 ? port_val
                        : (protocol == dmx_protocol::sacn ? static_cast<unsigned short>(5568)
                                                          : static_cast<unsigned short>(6454));

    return spl::make_shared<dmx_file_producer>(dependencies, filename->wstring(), host, port,
                                               universe, protocol, priority);
}

}} // namespace caspar::artnet
