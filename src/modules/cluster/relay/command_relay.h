/*
 * Copyright (c) 2024 CasparCG contributors
 *
 * CasparCG is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#pragma once

#include "virtual_channel_map.h"
#include "../sync/command_scheduler.h"

#include <atomic>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace caspar { namespace cluster { namespace relay {

enum class member_state
{
    disconnected,
    connecting,
    connected,
    error,
};

struct member_info
{
    std::string  host;
    uint16_t     port         = 5250;
    member_state state        = member_state::disconnected;
    uintptr_t    socket       = ~0ULL;
    int64_t      last_seen_ns = 0;
};

/// Callback for commands received from master (on client nodes)
using relay_command_handler = std::function<void(int64_t target_frame, const std::wstring& command)>;

/// Manages TCP connections to cluster members and routes commands.
/// Master: connects to all members, stamps commands with target frame, forwards.
/// Client: accepts connection from master, feeds received commands to scheduler.
class command_relay
{
  public:
    command_relay(const virtual_channel_map& channel_map, int sync_margin);
    ~command_relay();

    command_relay(const command_relay&)            = delete;
    command_relay& operator=(const command_relay&) = delete;

    /// Set the handler for received commands (client mode)
    void set_command_handler(relay_command_handler handler);

    // ─── Master mode ─────────────────────────────────────────────────────

    /// Connect to all remote members
    void connect_members(const std::vector<std::string>& member_addresses);

    /// Route a command to the appropriate member(s) based on virtual channel
    /// Stamps with target_frame and sends over TCP
    void route_command(int64_t target_frame, int virtual_channel, const std::wstring& command);

    /// Broadcast a command to ALL members (e.g., for BEGIN/COMMIT sync)
    void broadcast_command(int64_t target_frame, const std::wstring& command);

    // ─── Client mode ─────────────────────────────────────────────────────

    /// Start listening for master connection
    void start_client_listener(const std::string& bind_address, uint16_t port);

    // ─── Common ──────────────────────────────────────────────────────────

    /// Stop all connections
    void stop();

    /// Get status of all members
    std::vector<member_info> get_members() const;

  private:
    void master_connection_loop();
    void client_listener_loop();
    void client_receive_loop(uintptr_t client_socket);

    bool send_to_member(member_info& member, const std::string& data);
    bool connect_to_member(member_info& member);

    void parse_incoming_command(const std::string& line);

    const virtual_channel_map& channel_map_;
    int                        sync_margin_;

    relay_command_handler command_handler_;

    // Master: outgoing connections to members
    mutable std::mutex              members_mutex_;
    std::vector<member_info>        members_;
    std::thread                     connection_thread_;

    // Client: incoming connection from master
    std::string   client_bind_address_;
    uint16_t      client_port_ = 0;
    uintptr_t     listen_socket_ = ~0ULL;
    uintptr_t     master_socket_ = ~0ULL;
    std::thread   listener_thread_;
    std::thread   receive_thread_;

    std::atomic<bool> running_{false};
};

}}} // namespace caspar::cluster::relay
