/*
 * Copyright (c) 2024 CasparCG contributors
 *
 * CasparCG is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#pragma once

#include "ptp_messages.h"

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <thread>

namespace caspar { namespace cluster { namespace ptp {

enum class clock_mode
{
    master,   // This instance is the PTP grandmaster
    client,   // This instance syncs to a master
    external, // Lock to an external grandmaster on the network
};

enum class clock_state
{
    initializing,
    locked,       // Synchronized to master (client/external) or running (master)
    free_running, // Lost sync, using local clock
};

struct clock_status
{
    clock_state state         = clock_state::initializing;
    int64_t     offset_ns     = 0;   // Measured offset from master
    int64_t     delay_ns      = 0;   // Mean path delay
    uint16_t    sequence_id   = 0;   // Last sync sequence
    int64_t     ptp_time_ns   = 0;   // Current PTP time in nanoseconds
};

class ptp_clock
{
  public:
    ptp_clock(clock_mode          mode,
              const std::string&  bind_address,
              const std::string&  multicast_group,
              uint8_t             domain,
              int                 sync_interval_ms = 125);

    ~ptp_clock();

    ptp_clock(const ptp_clock&)            = delete;
    ptp_clock& operator=(const ptp_clock&) = delete;

    /// Start the PTP clock (launches network threads)
    void start();

    /// Stop the PTP clock
    void stop();

    /// Get current PTP time in nanoseconds since epoch
    int64_t now_ns() const;

    /// Get current clock status
    clock_status status() const;

    /// Get the clock mode
    clock_mode mode() const { return mode_; }

  private:
    void master_loop();
    void client_loop();
    void send_sync();
    void send_announce();
    void handle_sync(const sync_message& msg, int64_t recv_time_ns);
    void handle_follow_up(const follow_up_message& msg);
    void handle_delay_resp(const delay_resp_message& msg);
    void send_delay_req();

    int64_t local_time_ns() const;
    void    update_offset(int64_t measured_offset);

    clock_mode  mode_;
    std::string bind_address_;
    std::string multicast_group_;
    uint8_t     domain_;
    int         sync_interval_ms_;

    // Network
    uintptr_t event_socket_   = ~0ULL; // SOCKET for event port 319
    uintptr_t general_socket_ = ~0ULL; // SOCKET for general port 320

    // Clock state
    std::atomic<int64_t>     offset_ns_{0};
    std::atomic<int64_t>     delay_ns_{0};
    std::atomic<clock_state> state_{clock_state::initializing};
    std::atomic<uint16_t>    sequence_id_{0};

    // Delay measurement state
    int64_t  delay_req_send_time_ns_ = 0;
    uint16_t delay_req_sequence_     = 0;

    // Follow-up correlation
    int64_t  last_sync_recv_time_ns_ = 0;
    uint16_t last_sync_sequence_     = 0;

    // Offset filter (simple exponential moving average)
    double filtered_offset_ns_ = 0.0;
    bool   first_measurement_  = true;

    // Identity
    port_identity identity_;

    // Threading
    std::atomic<bool> running_{false};
    std::thread       worker_thread_;
};

}}} // namespace caspar::cluster::ptp
