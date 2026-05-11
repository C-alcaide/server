/*
 * Copyright (c) 2024 CasparCG contributors
 *
 * CasparCG is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#include "command_relay.h"

#include <common/log.h>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <WinSock2.h>
#include <WS2tcpip.h>

#include <chrono>
#include <sstream>

namespace caspar { namespace cluster { namespace relay {

namespace {

// Narrow ASCII wstring to string (safe for AMCP commands and addresses)
std::string narrow(const std::wstring& ws)
{
    if (ws.empty()) return {};
    int len = WideCharToMultiByte(CP_UTF8, 0, ws.data(), static_cast<int>(ws.size()), nullptr, 0, nullptr, nullptr);
    if (len <= 0) return {};
    std::string s(static_cast<size_t>(len), '\0');
    WideCharToMultiByte(CP_UTF8, 0, ws.data(), static_cast<int>(ws.size()), s.data(), len, nullptr, nullptr);
    return s;
}

std::wstring widen(const std::string& s)
{
    if (s.empty()) return {};
    int len = MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), nullptr, 0);
    if (len <= 0) return {};
    std::wstring ws(static_cast<size_t>(len), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), ws.data(), len);
    return ws;
}

// Wire protocol: "FRAME:target_frame COMMAND_TEXT\r\n"
// Simple text protocol for easy debugging and interop
// Protocol version handshake: master sends "CASPAR_CLUSTER/1\r\n" immediately after connect

static constexpr const char* PROTOCOL_HANDSHAKE = "CASPAR_CLUSTER/1\r\n";
static constexpr int         PROTOCOL_VERSION   = 1;

std::string encode_relay_message(int64_t target_frame, const std::wstring& command)
{
    std::string cmd_utf8 = narrow(command);
    return "FRAME:" + std::to_string(target_frame) + " " + cmd_utf8 + "\r\n";
}

bool parse_relay_message(const std::string& line, int64_t& out_frame, std::wstring& out_command)
{
    if (line.substr(0, 6) != "FRAME:") {
        return false;
    }
    auto space_pos = line.find(' ', 6);
    if (space_pos == std::string::npos) {
        return false;
    }
    try {
        out_frame = std::stoll(line.substr(6, space_pos - 6));
    } catch (...) {
        return false;
    }
    std::string cmd = line.substr(space_pos + 1);
    // Trim trailing \r\n
    while (!cmd.empty() && (cmd.back() == '\r' || cmd.back() == '\n')) {
        cmd.pop_back();
    }
    out_command = widen(cmd);
    return true;
}

} // anonymous namespace

command_relay::command_relay(const virtual_channel_map& channel_map, int sync_margin)
    : channel_map_(channel_map)
    , sync_margin_(sync_margin)
{
}

command_relay::~command_relay()
{
    stop();
}

void command_relay::set_command_handler(relay_command_handler handler)
{
    command_handler_ = std::move(handler);
}

void command_relay::connect_members(const std::vector<std::string>& member_addresses)
{
    {
        std::lock_guard<std::mutex> lock(members_mutex_);
        members_.clear();
        for (const auto& addr : member_addresses) {
            member_info m;
            // Parse "host:port"
            auto colon = addr.rfind(':');
            if (colon != std::string::npos) {
                m.host = addr.substr(0, colon);
                m.port = static_cast<uint16_t>(std::stoi(addr.substr(colon + 1)));
            } else {
                m.host = addr;
                m.port = 5250;
            }
            members_.push_back(m);
        }
    }

    running_ = true;
    connection_thread_ = std::thread([this] { master_connection_loop(); });
    CASPAR_LOG(info) << L"[cluster] Relay connecting to " << member_addresses.size() << L" member(s)";
}

void command_relay::route_command(int64_t target_frame, int virtual_channel, const std::wstring& command)
{
    std::string host = channel_map_.get_host(virtual_channel);
    if (host == "local") {
        return; // Local commands go directly to scheduler, not relay
    }

    // Rewrite channel reference
    std::wstring rewritten = channel_map_.rewrite_command(command, virtual_channel);
    std::string  msg       = encode_relay_message(target_frame, rewritten);

    std::lock_guard<std::mutex> lock(members_mutex_);
    for (auto& member : members_) {
        std::string member_addr = member.host + ":" + std::to_string(member.port);
        if (member_addr == host || member.host == host) {
            send_to_member(member, msg);
            return;
        }
    }

    CASPAR_LOG(warning) << L"[cluster] No member found for host: "
                        << std::wstring(host.begin(), host.end());
}

void command_relay::broadcast_command(int64_t target_frame, const std::wstring& command)
{
    std::string msg = encode_relay_message(target_frame, command);

    std::lock_guard<std::mutex> lock(members_mutex_);
    for (auto& member : members_) {
        send_to_member(member, msg);
    }
}

void command_relay::start_client_listener(const std::string& bind_address, uint16_t port)
{
    client_bind_address_ = bind_address;
    client_port_         = port;
    running_             = true;
    listener_thread_     = std::thread([this] { client_listener_loop(); });
    CASPAR_LOG(info) << L"[cluster] Client relay listening on port " << port;
}

void command_relay::stop()
{
    if (!running_.exchange(false)) {
        return;
    }

    // Wake the master_connection_loop if it's sleeping on the CV
    stop_cv_.notify_all();

    // Close all member sockets
    {
        std::lock_guard<std::mutex> lock(members_mutex_);
        for (auto& m : members_) {
            if (static_cast<SOCKET>(m.socket) != INVALID_SOCKET) {
                closesocket(static_cast<SOCKET>(m.socket));
                m.socket = static_cast<uintptr_t>(INVALID_SOCKET);
            }
        }
    }

    // Close listen/master sockets
    if (static_cast<SOCKET>(listen_socket_) != INVALID_SOCKET) {
        closesocket(static_cast<SOCKET>(listen_socket_));
        listen_socket_ = static_cast<uintptr_t>(INVALID_SOCKET);
    }
    if (static_cast<SOCKET>(master_socket_) != INVALID_SOCKET) {
        closesocket(static_cast<SOCKET>(master_socket_));
        master_socket_ = static_cast<uintptr_t>(INVALID_SOCKET);
    }

    if (connection_thread_.joinable()) connection_thread_.join();
    if (listener_thread_.joinable()) listener_thread_.join();
    if (receive_thread_.joinable()) receive_thread_.join();

    CASPAR_LOG(info) << L"[cluster] Command relay stopped";
}

std::vector<member_info> command_relay::get_members() const
{
    std::lock_guard<std::mutex> lock(members_mutex_);
    return members_;
}

// ─── Master: connection management ─────────────────────────────────────────

void command_relay::master_connection_loop()
{
    while (running_) {
        // Collect indices of members that need (re)connection
        std::vector<size_t> to_connect;
        {
            std::lock_guard<std::mutex> lock(members_mutex_);
            for (size_t i = 0; i < members_.size(); ++i) {
                if (!running_) break;
                if (members_[i].state == member_state::disconnected || members_[i].state == member_state::error) {
                    to_connect.push_back(i);
                }
            }
        }
        // Connect one member at a time, releasing the lock between attempts
        // so route_command/broadcast aren't blocked for the entire batch
        for (auto idx : to_connect) {
            if (!running_) break;
            std::lock_guard<std::mutex> lock(members_mutex_);
            if (members_[idx].state == member_state::disconnected || members_[idx].state == member_state::error) {
                connect_to_member(members_[idx]);
            }
        }
        // Wait up to 2s but wake immediately on stop()
        std::unique_lock<std::mutex> lock(members_mutex_);
        stop_cv_.wait_for(lock, std::chrono::seconds(2), [this] { return !running_.load(); });
    }
}

bool command_relay::connect_to_member(member_info& member)
{
    member.state = member_state::connecting;

    SOCKET sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (sock == INVALID_SOCKET) {
        member.state = member_state::error;
        return false;
    }

    // Set non-blocking for connect with timeout
    u_long non_blocking = 1;
    ioctlsocket(sock, FIONBIO, &non_blocking);

    sockaddr_in addr = {};
    addr.sin_family  = AF_INET;
    addr.sin_port    = htons(member.port);
    inet_pton(AF_INET, member.host.c_str(), &addr.sin_addr);

    int result = connect(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
    if (result == SOCKET_ERROR) {
        int err = WSAGetLastError();
        if (err == WSAEWOULDBLOCK) {
            // Wait for connection with 2-second timeout
            fd_set write_set, except_set;
            FD_ZERO(&write_set);
            FD_ZERO(&except_set);
            FD_SET(sock, &write_set);
            FD_SET(sock, &except_set);
            timeval tv = {2, 0}; // 2 seconds
            int sel = select(0, nullptr, &write_set, &except_set, &tv);
            if (sel <= 0 || FD_ISSET(sock, &except_set)) {
                closesocket(sock);
                member.state = member_state::error;
                CASPAR_LOG(debug) << L"[cluster] Connect timeout to "
                                  << std::wstring(member.host.begin(), member.host.end())
                                  << L":" << member.port;
                return false;
            }
        } else {
            closesocket(sock);
            member.state = member_state::error;
            CASPAR_LOG(debug) << L"[cluster] Failed to connect to "
                              << std::wstring(member.host.begin(), member.host.end())
                              << L":" << member.port;
            return false;
        }
    }

    // Restore blocking mode
    u_long blocking = 0;
    ioctlsocket(sock, FIONBIO, &blocking);

    // Set send timeout
    DWORD timeout = 1000;
    setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, reinterpret_cast<const char*>(&timeout), sizeof(timeout));

    // Disable Nagle for low-latency command delivery
    int nodelay = 1;
    setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<const char*>(&nodelay), sizeof(nodelay));

    member.socket = static_cast<uintptr_t>(sock);
    member.state  = member_state::connected;

    // Send protocol version handshake
    send(sock, PROTOCOL_HANDSHAKE, static_cast<int>(strlen(PROTOCOL_HANDSHAKE)), 0);

    CASPAR_LOG(info) << L"[cluster] Connected to member "
                     << std::wstring(member.host.begin(), member.host.end())
                     << L":" << member.port;
    return true;
}

bool command_relay::send_to_member(member_info& member, const std::string& data)
{
    if (member.state != member_state::connected) {
        return false;
    }

    SOCKET sock = static_cast<SOCKET>(member.socket);
    int    total_sent = 0;
    int    remaining  = static_cast<int>(data.size());

    while (remaining > 0) {
        int sent = send(sock, data.c_str() + total_sent, remaining, 0);
        if (sent == SOCKET_ERROR) {
            CASPAR_LOG(warning) << L"[cluster] Lost connection to "
                                << std::wstring(member.host.begin(), member.host.end());
            closesocket(sock);
            member.socket = static_cast<uintptr_t>(INVALID_SOCKET);
            member.state  = member_state::disconnected;
            return false;
        }
        total_sent += sent;
        remaining  -= sent;
    }
    return true;
}

// ─── Client: listener ───────────────────────────────────────────────────────

void command_relay::client_listener_loop()
{
    SOCKET listen_sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (listen_sock == INVALID_SOCKET) {
        CASPAR_LOG(error) << L"[cluster] Failed to create listener socket";
        return;
    }

    int reuse = 1;
    setsockopt(listen_sock, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&reuse), sizeof(reuse));

    sockaddr_in addr = {};
    addr.sin_family  = AF_INET;
    addr.sin_port    = htons(client_port_);
    inet_pton(AF_INET, client_bind_address_.c_str(), &addr.sin_addr);

    if (bind(listen_sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == SOCKET_ERROR) {
        CASPAR_LOG(error) << L"[cluster] Failed to bind relay listener on port " << client_port_;
        closesocket(listen_sock);
        return;
    }

    if (listen(listen_sock, 1) == SOCKET_ERROR) {
        CASPAR_LOG(error) << L"[cluster] Failed to listen on relay port " << client_port_;
        closesocket(listen_sock);
        return;
    }
    listen_socket_ = static_cast<uintptr_t>(listen_sock);

    // Set accept timeout so we can check running_ flag
    DWORD timeout = 1000;
    setsockopt(listen_sock, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&timeout), sizeof(timeout));

    while (running_) {
        sockaddr_in client_addr = {};
        int         client_len  = sizeof(client_addr);
        SOCKET      client_sock = accept(listen_sock, reinterpret_cast<sockaddr*>(&client_addr), &client_len);

        if (client_sock == INVALID_SOCKET) {
            continue; // Timeout or error, retry
        }

        // Disable Nagle
        int nodelay = 1;
        setsockopt(client_sock, IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<const char*>(&nodelay), sizeof(nodelay));

        // Set receive timeout on client socket so receive thread doesn't block forever
        DWORD recv_timeout = 2000;
        setsockopt(client_sock, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&recv_timeout), sizeof(recv_timeout));

        master_socket_ = static_cast<uintptr_t>(client_sock);
        CASPAR_LOG(info) << L"[cluster] Master connected to client relay";

        // Start receive loop in separate thread (replace if reconnected)
        if (receive_thread_.joinable()) {
            receive_thread_.join();
        }
        receive_thread_ = std::thread([this, client_sock] {
            client_receive_loop(static_cast<uintptr_t>(client_sock));
        });
    }
}

void command_relay::client_receive_loop(uintptr_t client_socket)
{
    SOCKET sock = static_cast<SOCKET>(client_socket);
    std::string buffer;
    char        recv_buf[4096];
    constexpr size_t MAX_BUFFER_SIZE = 1024 * 1024; // 1 MB max
    bool handshake_validated = false;

    while (running_) {
        int received = recv(sock, recv_buf, sizeof(recv_buf), 0);
        if (received <= 0) {
            if (received == 0) {
                CASPAR_LOG(info) << L"[cluster] Master disconnected gracefully";
            } else if (running_) {
                CASPAR_LOG(warning) << L"[cluster] Lost connection to master";
            }
            break;
        }

        buffer.append(recv_buf, received);

        // Protect against unbounded buffer growth (no newline in sight)
        if (buffer.size() > MAX_BUFFER_SIZE) {
            CASPAR_LOG(error) << L"[cluster] TCP receive buffer exceeded 1 MB without newline, dropping data";
            buffer.clear();
            continue;
        }

        // Process complete lines
        size_t newline;
        while ((newline = buffer.find('\n')) != std::string::npos) {
            std::string line = buffer.substr(0, newline);
            buffer.erase(0, newline + 1);

            // Trim \r
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }

            // First line must be the protocol handshake
            if (!handshake_validated) {
                if (line.find("CASPAR_CLUSTER/") == 0) {
                    int version = 0;
                    try { version = std::stoi(line.substr(15)); } catch (...) {}
                    if (version != PROTOCOL_VERSION) {
                        CASPAR_LOG(error) << L"[cluster] Protocol version mismatch: master="
                                          << version << L" local=" << PROTOCOL_VERSION;
                        return; // Disconnect
                    }
                    handshake_validated = true;
                    CASPAR_LOG(debug) << L"[cluster] Protocol handshake OK (v" << PROTOCOL_VERSION << L")";
                } else {
                    CASPAR_LOG(error) << L"[cluster] Missing protocol handshake from master, disconnecting";
                    return;
                }
                continue;
            }

            parse_incoming_command(line);
        }
    }
}

void command_relay::parse_incoming_command(const std::string& line)
{
    int64_t      target_frame;
    std::wstring command;

    if (!parse_relay_message(line, target_frame, command)) {
        CASPAR_LOG(warning) << L"[cluster] Invalid relay message: "
                            << std::wstring(line.begin(), line.end());
        return;
    }

    if (command_handler_) {
        command_handler_(target_frame, command);
    }
}

}}} // namespace caspar::cluster::relay
