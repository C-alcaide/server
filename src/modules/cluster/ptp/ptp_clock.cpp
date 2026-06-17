/*
 * Copyright (c) 2024 CasparCG contributors
 *
 * CasparCG is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#include "ptp_clock.h"

#include <common/log.h>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <WinSock2.h>
#include <WS2tcpip.h>
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>
#include <cerrno>
#endif

#include <algorithm>
#include <cstring>
#include <random>

namespace caspar { namespace cluster { namespace ptp {

namespace {

// ─── Platform socket abstraction ────────────────────────────────────────────
#ifdef _WIN32
using socket_t = SOCKET;
constexpr socket_t kInvalidSocket = INVALID_SOCKET;
constexpr int      kSocketError   = SOCKET_ERROR;
inline int         close_socket(socket_t s) { return closesocket(s); }
inline int         last_error() { return WSAGetLastError(); }
#else
using socket_t = int;
constexpr socket_t kInvalidSocket = -1;
constexpr int      kSocketError   = -1;
inline int         close_socket(socket_t s) { return ::close(s); }
inline int         last_error() { return errno; }
#endif

constexpr uint16_t PTP_EVENT_PORT   = 319;
constexpr uint16_t PTP_GENERAL_PORT = 320;
constexpr double   OFFSET_FILTER_ALPHA = 0.125; // RFC-like EMA constant

int64_t steady_clock_ns()
{
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
}

int64_t system_clock_ns()
{
    auto now = std::chrono::system_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
}

port_identity generate_identity()
{
    port_identity id;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> dist(0, 255);
    for (auto& b : id.clock_identity) {
        b = static_cast<uint8_t>(dist(gen));
    }
    id.port_number = htons(1); // Stored in network byte order (PTP wire format)
    return id;
}

socket_t create_udp_socket(const std::string& bind_addr, uint16_t port, const std::string& multicast_group)
{
    socket_t sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (sock == kInvalidSocket) {
        return kInvalidSocket;
    }

    // Allow address reuse
    int reuse = 1;
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&reuse), sizeof(reuse));

    // Bind
    sockaddr_in addr = {};
    addr.sin_family      = AF_INET;
    addr.sin_port        = htons(port);
    inet_pton(AF_INET, bind_addr.c_str(), &addr.sin_addr);
    if (bind(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == kSocketError) {
        close_socket(sock);
        return kInvalidSocket;
    }

    // Join multicast group
    ip_mreq mreq = {};
    inet_pton(AF_INET, multicast_group.c_str(), &mreq.imr_multiaddr);
    inet_pton(AF_INET, bind_addr.c_str(), &mreq.imr_interface);
    if (setsockopt(sock, IPPROTO_IP, IP_ADD_MEMBERSHIP, reinterpret_cast<const char*>(&mreq), sizeof(mreq)) ==
        kSocketError) {
        CASPAR_LOG(warning) << L"[cluster] Failed to join PTP multicast group";
    }

    // Set multicast interface
    in_addr iface = {};
    inet_pton(AF_INET, bind_addr.c_str(), &iface);
    setsockopt(sock, IPPROTO_IP, IP_MULTICAST_IF, reinterpret_cast<const char*>(&iface), sizeof(iface));

    // Set TTL
    int ttl = 1;
    setsockopt(sock, IPPROTO_IP, IP_MULTICAST_TTL, reinterpret_cast<const char*>(&ttl), sizeof(ttl));

    // Set receive timeout (50ms)
#ifdef _WIN32
    DWORD timeout = 50;
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&timeout), sizeof(timeout));
#else
    struct timeval tv = {0, 50000}; // 50ms
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
#endif

    return sock;
}

void send_to_multicast(socket_t sock, const std::string& group, uint16_t port, const void* data, int len)
{
    sockaddr_in dest = {};
    dest.sin_family  = AF_INET;
    dest.sin_port    = htons(port);
    inet_pton(AF_INET, group.c_str(), &dest.sin_addr);
    sendto(sock, reinterpret_cast<const char*>(data), len, 0, reinterpret_cast<sockaddr*>(&dest), sizeof(dest));
}

uint16_t net_to_host16(uint16_t v)
{
    return ntohs(v);
}

uint16_t host_to_net16(uint16_t v)
{
    return htons(v);
}

uint32_t net_to_host32(uint32_t v)
{
    return ntohl(v);
}

uint32_t host_to_net32(uint32_t v)
{
    return htonl(v);
}

} // anonymous namespace

ptp_clock::ptp_clock(clock_mode         mode,
                     const std::string& bind_address,
                     const std::string& multicast_group,
                     uint8_t            domain,
                     int                sync_interval_ms)
    : mode_(mode)
    , bind_address_(bind_address)
    , multicast_group_(multicast_group)
    , domain_(domain)
    , sync_interval_ms_(sync_interval_ms)
    , identity_(generate_identity())
{
}

ptp_clock::~ptp_clock()
{
    stop();
}

void ptp_clock::start()
{
    if (running_.exchange(true)) {
        return; // Already running
    }

#ifdef _WIN32
    // Initialize Winsock
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
        CASPAR_LOG(error) << L"[cluster] WSAStartup failed";
        running_ = false;
        return;
    }
#endif

    // Create sockets
    event_socket_   = static_cast<uintptr_t>(create_udp_socket(bind_address_, PTP_EVENT_PORT, multicast_group_));
    general_socket_ = static_cast<uintptr_t>(create_udp_socket(bind_address_, PTP_GENERAL_PORT, multicast_group_));

    if (static_cast<socket_t>(event_socket_) == kInvalidSocket ||
        static_cast<socket_t>(general_socket_) == kInvalidSocket) {
        CASPAR_LOG(error) << L"[cluster] Failed to create PTP sockets";
        if (static_cast<socket_t>(event_socket_) != kInvalidSocket) {
            close_socket(static_cast<socket_t>(event_socket_));
            event_socket_ = static_cast<uintptr_t>(kInvalidSocket);
        }
        if (static_cast<socket_t>(general_socket_) != kInvalidSocket) {
            close_socket(static_cast<socket_t>(general_socket_));
            general_socket_ = static_cast<uintptr_t>(kInvalidSocket);
        }
#ifdef _WIN32
        WSACleanup();
#endif
        running_ = false;
        return;
    }

    if (mode_ == clock_mode::master) {
        state_ = clock_state::locked; // Master is always "locked" (authoritative)
        worker_thread_ = std::thread([this] { master_loop(); });
    } else {
        worker_thread_ = std::thread([this] { client_loop(); });
    }

    CASPAR_LOG(info) << L"[cluster] PTP clock started, mode="
                     << (mode_ == clock_mode::master ? L"master" : L"client")
                     << L", domain=" << static_cast<int>(domain_);
}

void ptp_clock::stop()
{
    if (!running_.exchange(false)) {
        return;
    }

    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }

    if (static_cast<socket_t>(event_socket_) != kInvalidSocket) {
        close_socket(static_cast<socket_t>(event_socket_));
        event_socket_ = static_cast<uintptr_t>(kInvalidSocket);
    }
    if (static_cast<socket_t>(general_socket_) != kInvalidSocket) {
        close_socket(static_cast<socket_t>(general_socket_));
        general_socket_ = static_cast<uintptr_t>(kInvalidSocket);
    }

#ifdef _WIN32
    WSACleanup();
#endif

    CASPAR_LOG(info) << L"[cluster] PTP clock stopped";
}

int64_t ptp_clock::now_ns() const
{
    if (mode_ == clock_mode::master) {
        return system_clock_ns();
    }
    // Client: local system time + corrected offset
    return system_clock_ns() + offset_ns_.load(std::memory_order_relaxed);
}

clock_status ptp_clock::status() const
{
    clock_status s;
    s.state       = state_.load(std::memory_order_relaxed);
    s.offset_ns   = offset_ns_.load(std::memory_order_relaxed);
    s.delay_ns    = delay_ns_.load(std::memory_order_relaxed);
    s.sequence_id = sequence_id_.load(std::memory_order_relaxed);
    s.ptp_time_ns = now_ns();
    return s;
}

int64_t ptp_clock::local_time_ns() const
{
    return system_clock_ns();
}

// ─── Master loop ────────────────────────────────────────────────────────────

void ptp_clock::master_loop()
{
    auto last_sync     = std::chrono::steady_clock::now();
    auto last_announce = last_sync;

#ifdef _WIN32
    WSAPOLLFD pfd = {};
#else
    struct pollfd pfd = {};
#endif
    pfd.fd     = static_cast<socket_t>(event_socket_);
    pfd.events = POLLIN;

    while (running_) {
        auto now = std::chrono::steady_clock::now();

        // Send Sync + Follow_Up at configured interval
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_sync).count() >= sync_interval_ms_) {
            send_sync();
            last_sync = now;
        }

        // Send Announce every 2 seconds
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_announce).count() >= 2000) {
            send_announce();
            last_announce = now;
        }

        // Poll for Delay_Req from clients (non-blocking with short timeout)
#ifdef _WIN32
        int poll_result = WSAPoll(&pfd, 1, 10);
#else
        int poll_result = poll(&pfd, 1, 10);
#endif
        if (poll_result <= 0) {
            continue;
        }

        char buffer[256];
        sockaddr_in sender_addr = {};
        socklen_t sender_len = sizeof(sender_addr);
        int received = recvfrom(static_cast<socket_t>(event_socket_), buffer, sizeof(buffer), 0,
                                reinterpret_cast<sockaddr*>(&sender_addr), &sender_len);

        if (received >= static_cast<int>(sizeof(ptp_header))) {
            auto* hdr = reinterpret_cast<const ptp_header*>(buffer);
            if (hdr->domain_number == domain_ && hdr->get_type() == message_type::delay_req) {
                // Respond with Delay_Resp
                auto recv_time = ptp_timestamp::from_nanoseconds(system_clock_ns());

                delay_resp_message resp = {};
                resp.header.set_type(message_type::delay_resp);
                resp.header.version_ptp    = 0x02;
                resp.header.message_length = host_to_net16(sizeof(delay_resp_message));
                resp.header.domain_number  = domain_;
                resp.header.source_port_identity = identity_;
                resp.header.sequence_id    = hdr->sequence_id; // Echo sequence
                resp.header.control_field  = 3;
                resp.header.log_message_interval = 0;
                resp.receive_timestamp     = recv_time;
                resp.requesting_port_identity = hdr->source_port_identity;

                send_to_multicast(static_cast<socket_t>(general_socket_), multicast_group_,
                                  PTP_GENERAL_PORT, &resp, sizeof(resp));
            }
        }
    }
}

void ptp_clock::send_sync()
{
    uint16_t seq = sequence_id_.fetch_add(1, std::memory_order_relaxed);

    // Send Sync (event port)
    sync_message sync = {};
    sync.header.set_type(message_type::sync);
    sync.header.version_ptp    = 0x02;
    sync.header.message_length = host_to_net16(sizeof(sync_message));
    sync.header.domain_number  = domain_;
    sync.header.flags          = host_to_net16(0x0200); // TWO_STEP flag
    sync.header.source_port_identity = identity_;
    sync.header.sequence_id    = host_to_net16(seq);
    sync.header.control_field  = 0;
    sync.header.log_message_interval = -3; // 125ms

    auto t1 = ptp_timestamp::from_nanoseconds(system_clock_ns());
    sync.origin_timestamp = t1;

    send_to_multicast(static_cast<socket_t>(event_socket_), multicast_group_,
                      PTP_EVENT_PORT, &sync, sizeof(sync));

    // Send Follow_Up with precise timestamp (general port)
    auto precise_t1 = ptp_timestamp::from_nanoseconds(system_clock_ns());

    follow_up_message fup = {};
    fup.header.set_type(message_type::follow_up);
    fup.header.version_ptp    = 0x02;
    fup.header.message_length = host_to_net16(sizeof(follow_up_message));
    fup.header.domain_number  = domain_;
    fup.header.source_port_identity = identity_;
    fup.header.sequence_id    = host_to_net16(seq);
    fup.header.control_field  = 2;
    fup.header.log_message_interval = -3;
    fup.precise_origin_timestamp = precise_t1;

    send_to_multicast(static_cast<socket_t>(general_socket_), multicast_group_,
                      PTP_GENERAL_PORT, &fup, sizeof(fup));
}

void ptp_clock::send_announce()
{
    announce_message ann = {};
    ann.header.set_type(message_type::announce);
    ann.header.version_ptp    = 0x02;
    ann.header.message_length = host_to_net16(sizeof(announce_message));
    ann.header.domain_number  = domain_;
    ann.header.source_port_identity = identity_;
    ann.header.sequence_id    = host_to_net16(sequence_id_.load(std::memory_order_relaxed));
    ann.header.control_field  = 5;
    ann.header.log_message_interval = 1; // 2 seconds

    ann.grandmaster_priority1 = 128;
    ann.grandmaster_priority2 = 128;
    ann.grandmaster_clock_quality = host_to_net32(0x60FE00FF); // class 248, accuracy 0xFE, variance 0x00FF
    ann.grandmaster_identity  = identity_.clock_identity;
    ann.steps_removed         = 0;
    ann.time_source           = 0xA0; // Internal oscillator

    send_to_multicast(static_cast<socket_t>(general_socket_), multicast_group_,
                      PTP_GENERAL_PORT, &ann, sizeof(ann));
}

// ─── Client loop ────────────────────────────────────────────────────────────

void ptp_clock::client_loop()
{
    auto last_delay_req = std::chrono::steady_clock::now();

#ifdef _WIN32
    WSAPOLLFD fds[2];
#else
    struct pollfd fds[2];
#endif
    fds[0].fd     = static_cast<socket_t>(event_socket_);
    fds[0].events = POLLIN;
    fds[1].fd     = static_cast<socket_t>(general_socket_);
    fds[1].events = POLLIN;

    while (running_) {
        // Poll both sockets with 100ms timeout
#ifdef _WIN32
        int poll_result = WSAPoll(fds, 2, 100);
#else
        int poll_result = poll(fds, 2, 100);
#endif
        if (poll_result < 0) {
            CASPAR_LOG(warning) << L"[ptp_clock] poll failed: " << last_error();
            continue;
        }

        int64_t recv_time = system_clock_ns();

        // Check event port (Sync messages)
        if (fds[0].revents & POLLIN) {
            char buffer[256];
            int  received = recv(fds[0].fd, buffer, sizeof(buffer), 0);

            if (received >= static_cast<int>(sizeof(ptp_header))) {
                auto* hdr = reinterpret_cast<const ptp_header*>(buffer);
                if (hdr->domain_number == domain_ && !is_self_message(*hdr)) {
                    if (hdr->get_type() == message_type::sync &&
                        received >= static_cast<int>(sizeof(sync_message))) {
                        auto& sync_msg = *reinterpret_cast<const sync_message*>(buffer);
                        if (accept_master(sync_msg.header.source_port_identity)) {
                            handle_sync(sync_msg, recv_time);
                            last_sync_received_ = std::chrono::steady_clock::now();
                        }
                    }
                }
            }
        }

        // Check general port (Follow_Up, Delay_Resp)
        if (fds[1].revents & POLLIN) {
            char buffer[256];
            int  received = recv(fds[1].fd, buffer, sizeof(buffer), 0);

            if (received >= static_cast<int>(sizeof(ptp_header))) {
                auto* hdr = reinterpret_cast<const ptp_header*>(buffer);
                if (hdr->domain_number == domain_ && !is_self_message(*hdr)) {
                    if (hdr->get_type() == message_type::follow_up &&
                        received >= static_cast<int>(sizeof(follow_up_message))) {
                        auto& fup = *reinterpret_cast<const follow_up_message*>(buffer);
                        if (accept_master(fup.header.source_port_identity)) {
                            handle_follow_up(fup);
                        }
                    } else if (hdr->get_type() == message_type::delay_resp &&
                               received >= static_cast<int>(sizeof(delay_resp_message))) {
                        handle_delay_resp(*reinterpret_cast<const delay_resp_message*>(buffer));
                    }
                }
            }
        }

        // Send Delay_Req periodically (every ~1 second)
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_delay_req).count() >= 1000) {
            send_delay_req();
            last_delay_req = now;
        }

        // Detect master disappearance: if no Sync received for 5 seconds, go free-running
        if (state_.load(std::memory_order_relaxed) == clock_state::locked) {
            auto since_last_sync = std::chrono::duration_cast<std::chrono::seconds>(now - last_sync_received_).count();
            if (since_last_sync >= 5) {
                CASPAR_LOG(warning) << L"[ptp_clock] No Sync from master for 5s - transitioning to free-running";
                state_.store(clock_state::free_running, std::memory_order_relaxed);
                locked_master_ = {};
                first_measurement_ = true;
            }
        }
    }
}

// Fix #20: Skip messages we sent ourselves
bool ptp_clock::is_self_message(const ptp_header& hdr) const
{
    return std::memcmp(&hdr.source_port_identity, &identity_, sizeof(port_identity)) == 0;
}

// Fix #18: Accept only one master once locked; accept any when not locked
bool ptp_clock::accept_master(const port_identity& source)
{
    if (state_.load(std::memory_order_relaxed) == clock_state::locked) {
        if (std::memcmp(&source, &locked_master_, sizeof(port_identity)) != 0) {
            return false; // Ignore Sync from a different master
        }
    } else {
        // When not locked, track whichever master sends Sync
        locked_master_ = source;
    }
    return true;
}

void ptp_clock::handle_sync(const sync_message& msg, int64_t recv_time_ns)
{
    // Store T2 (receive timestamp) and sequence for correlation with Follow_Up
    last_sync_recv_time_ns_ = recv_time_ns;
    last_sync_sequence_     = net_to_host16(msg.header.sequence_id);
}

void ptp_clock::handle_follow_up(const follow_up_message& msg)
{
    uint16_t seq = net_to_host16(msg.header.sequence_id);
    if (seq != last_sync_sequence_) {
        return; // Not correlated with last Sync
    }

    // T1 = precise origin timestamp from Follow_Up
    int64_t t1 = msg.precise_origin_timestamp.to_nanoseconds();
    // T2 = local receive time of Sync
    int64_t t2 = last_sync_recv_time_ns_;

    // offset = T1 - T2 + delay
    // We compute a raw offset; delay is added from Delay_Resp
    int64_t raw_offset = t1 - t2 + delay_ns_.load(std::memory_order_relaxed);
    update_offset(raw_offset);
}

void ptp_clock::handle_delay_resp(const delay_resp_message& msg)
{
    uint16_t seq = net_to_host16(msg.header.sequence_id);
    if (seq != delay_req_sequence_) {
        return; // Not our request
    }

    // Fix #8: Verify the response is actually for us
    if (std::memcmp(&msg.requesting_port_identity, &identity_, sizeof(port_identity)) != 0) {
        return; // Response for a different node
    }

    // T3 = our Delay_Req send time
    int64_t t3 = delay_req_send_time_ns_;
    // T4 = receive timestamp from master
    int64_t t4 = msg.receive_timestamp.to_nanoseconds();

    // Mean path delay = (T4 - T3 - offset) / 2
    // Simplified: delay = (T4 - T3) / 2 (first order)
    int64_t delay = (t4 - t3) / 2;
    if (delay > 0) {
        delay_ns_.store(delay, std::memory_order_relaxed);
    }
}

void ptp_clock::send_delay_req()
{
    delay_req_sequence_ = sequence_id_.fetch_add(1, std::memory_order_relaxed);

    delay_req_message req = {};
    req.header.set_type(message_type::delay_req);
    req.header.version_ptp    = 0x02;
    req.header.message_length = host_to_net16(sizeof(delay_req_message));
    req.header.domain_number  = domain_;
    req.header.source_port_identity = identity_;
    req.header.sequence_id    = host_to_net16(delay_req_sequence_);
    req.header.control_field  = 1;
    req.header.log_message_interval = 0;

    delay_req_send_time_ns_ = system_clock_ns();
    req.origin_timestamp = ptp_timestamp::from_nanoseconds(delay_req_send_time_ns_);

    send_to_multicast(static_cast<socket_t>(event_socket_), multicast_group_,
                      PTP_EVENT_PORT, &req, sizeof(req));
}

void ptp_clock::update_offset(int64_t measured_offset)
{
    if (first_measurement_) {
        filtered_offset_ns_ = static_cast<double>(measured_offset);
        first_measurement_  = false;
    } else {
        filtered_offset_ns_ = OFFSET_FILTER_ALPHA * measured_offset +
                              (1.0 - OFFSET_FILTER_ALPHA) * filtered_offset_ns_;
    }

    offset_ns_.store(static_cast<int64_t>(filtered_offset_ns_), std::memory_order_relaxed);

    // Consider locked if offset is below 1ms
    if (std::abs(filtered_offset_ns_) < 1'000'000.0) {
        state_.store(clock_state::locked, std::memory_order_relaxed);
    } else if (state_.load(std::memory_order_relaxed) == clock_state::locked) {
        // Lost lock threshold: 10ms
        if (std::abs(filtered_offset_ns_) > 10'000'000.0) {
            state_.store(clock_state::free_running, std::memory_order_relaxed);
            CASPAR_LOG(warning) << L"[cluster] PTP clock lost sync, offset="
                                << static_cast<int64_t>(filtered_offset_ns_) / 1000 << L"µs";
        }
    }
}

}}} // namespace caspar::cluster::ptp
