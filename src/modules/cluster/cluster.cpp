/*
 * Copyright (c) 2024 CasparCG contributors
 *
 * CasparCG is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#include "cluster.h"
#include "cluster_config.h"
#include "ptp/ptp_clock.h"
#include "sync/frame_clock.h"
#include "sync/command_scheduler.h"
#include "sync/content_sync.h"
#include "relay/command_relay.h"
#include "relay/virtual_channel_map.h"

#include <common/env.h>
#include <common/log.h>
#include <protocol/amcp/amcp_command_repository_wrapper.h>
#include <protocol/amcp/amcp_shared.h>

#include <memory>
#include <mutex>
#include <sstream>

namespace caspar { namespace cluster {

using protocol::amcp::channel_context;
using protocol::amcp::command_context;

namespace {

// ─── Singleton cluster state ────────────────────────────────────────────────

struct cluster_state
{
    cluster_config                              config;
    std::shared_ptr<ptp::ptp_clock>             ptp;
    std::shared_ptr<sync::frame_clock>          frame_clock;
    std::shared_ptr<sync::command_scheduler>    scheduler;
    std::shared_ptr<sync::content_sync>         watchdog;
    std::unique_ptr<relay::virtual_channel_map> channel_map;
    std::unique_ptr<relay::command_relay>       relay;
    bool                                        active          = false;
    bool                                        watchdog_armed  = false; // true once channels are captured
};

std::mutex    g_state_mutex;
cluster_state g_state;

// Forward declaration
void arm_watchdog(const spl::shared_ptr<std::vector<channel_context>>& channels);

// ─── AMCP command: CLUSTER STATUS ───────────────────────────────────────────

std::wstring cluster_status_command(protocol::amcp::command_context& ctx)
{
    std::lock_guard<std::mutex> lock(g_state_mutex);

    if (!g_state.active) {
        return L"201 CLUSTER STATUS OK\r\nDISABLED\r\n";
    }

    // Deferred arm: channels are available now
    arm_watchdog(ctx.channels);

    std::wostringstream reply;
    reply << L"201 CLUSTER STATUS OK\r\n";

    // Mode
    switch (g_state.config.mode) {
        case cluster_mode::master:   reply << L"MODE: master\r\n"; break;
        case cluster_mode::client:   reply << L"MODE: client\r\n"; break;
        case cluster_mode::external: reply << L"MODE: external\r\n"; break;
        default:                     reply << L"MODE: disabled\r\n"; break;
    }

    // PTP status
    if (g_state.ptp) {
        auto status = g_state.ptp->status();
        reply << L"PTP-STATE: ";
        switch (status.state) {
            case ptp::clock_state::initializing: reply << L"initializing"; break;
            case ptp::clock_state::locked:       reply << L"locked"; break;
            case ptp::clock_state::free_running: reply << L"free-running"; break;
        }
        reply << L"\r\n";
        reply << L"PTP-OFFSET-US: " << (status.offset_ns / 1000) << L"\r\n";
        reply << L"PTP-DELAY-US: " << (status.delay_ns / 1000) << L"\r\n";
    }

    // Frame clock
    if (g_state.frame_clock) {
        reply << L"FRAME: " << g_state.frame_clock->current_frame() << L"\r\n";
    }

    // Scheduler
    if (g_state.scheduler) {
        reply << L"PENDING-COMMANDS: " << g_state.scheduler->pending_count() << L"\r\n";
    }

    // Members (master mode)
    if (g_state.relay && g_state.config.mode == cluster_mode::master) {
        auto members = g_state.relay->get_members();
        for (const auto& m : members) {
            std::wstring host_w(m.host.begin(), m.host.end());
            reply << L"MEMBER: " << host_w << L":" << m.port << L" ";
            switch (m.state) {
                case relay::member_state::connected:    reply << L"connected"; break;
                case relay::member_state::connecting:   reply << L"connecting"; break;
                case relay::member_state::disconnected: reply << L"disconnected"; break;
                case relay::member_state::error:        reply << L"error"; break;
            }
            reply << L"\r\n";
        }
    }

    // Content sync watchdog
    if (g_state.watchdog) {
        reply << L"SYNC-CORRECTIONS: " << g_state.watchdog->total_corrections() << L"\r\n";
        auto sync_status = g_state.watchdog->status();
        for (const auto& s : sync_status) {
            reply << L"SYNC-CH" << s.channel_index << L": "
                  << (s.synced ? L"synced" : L"DRIFTED")
                  << L" layers=" << s.active_layers
                  << L" max-drift=" << s.max_drift_frames
                  << L" corrections=" << s.total_corrections
                  << L"\r\n";
        }
    }

    reply << L"\r\n";
    return reply.str();
}

// ─── AMCP command: SCHEDULE ─────────────────────────────────────────────────

std::wstring cluster_schedule_command(protocol::amcp::command_context& ctx)
{
    std::lock_guard<std::mutex> lock(g_state_mutex);

    if (!g_state.active || !g_state.scheduler) {
        return L"501 CLUSTER SCHEDULE FAILED\r\n";
    }

    // Deferred arm: channels are available now
    arm_watchdog(ctx.channels);

    // Parameters: SCHEDULE <channel-layer> <command> AT <frame>
    // Or: SCHEDULE <channel-layer> <command>  (uses sync_margin)
    if (ctx.parameters.size() < 2) {
        return L"400 CLUSTER SCHEDULE ERROR\r\n";
    }

    // Find "AT" keyword
    int64_t      target_frame = -1;
    std::wstring command_text;

    auto at_it = std::find(ctx.parameters.begin(), ctx.parameters.end(), L"AT");
    if (at_it != ctx.parameters.end() && (at_it + 1) != ctx.parameters.end()) {
        // Explicit frame target
        try {
            target_frame = std::stoll(*(at_it + 1));
        } catch (...) {
            return L"400 CLUSTER SCHEDULE ERROR - INVALID FRAME\r\n";
        }
        // Command is everything before "AT"
        std::wostringstream cmd;
        for (auto it = ctx.parameters.begin(); it != at_it; ++it) {
            if (it != ctx.parameters.begin()) cmd << L" ";
            cmd << *it;
        }
        command_text = cmd.str();
    } else {
        // No explicit frame - use sync_margin
        std::wostringstream cmd;
        for (const auto& p : ctx.parameters) {
            if (&p != &ctx.parameters[0]) cmd << L" ";
            cmd << p;
        }
        command_text = cmd.str();
        target_frame = g_state.frame_clock->current_frame() + g_state.config.sync_margin;
    }

    // Parse virtual channel from command (first token like "1-10")
    int virtual_channel = 0;
    if (!command_text.empty()) {
        // Commands start with verb then channel: "PLAY 2-1 clip"
        // We need to extract channel from the command text
        std::wistringstream iss(command_text);
        std::wstring verb, channel_str;
        iss >> verb >> channel_str;
        auto dash = channel_str.find(L'-');
        if (dash != std::wstring::npos) {
            try { virtual_channel = std::stoi(channel_str.substr(0, dash)); } catch (...) {}
        } else {
            try { virtual_channel = std::stoi(channel_str); } catch (...) {}
        }
    }

    // Route: local or remote?
    if (g_state.channel_map && virtual_channel > 0 && !g_state.channel_map->is_local(virtual_channel)) {
        // Forward to remote member
        g_state.relay->route_command(target_frame, virtual_channel, command_text);
    } else {
        // Schedule locally
        g_state.scheduler->schedule(target_frame, command_text);
    }

    std::wostringstream reply;
    reply << L"202 CLUSTER SCHEDULE OK " << target_frame << L"\r\n";
    return reply.str();
}

// ─── Arm watchdog (deferred until channels are available) ───────────────────

void arm_watchdog(const spl::shared_ptr<std::vector<channel_context>>& channels)
{
    // Already armed or no cluster active
    if (g_state.watchdog_armed || !g_state.active || !g_state.frame_clock) {
        return;
    }

    std::vector<std::shared_ptr<core::video_channel>> channel_vec;
    for (const auto& ch_ctx : *channels) {
        if (ch_ctx.raw_channel) {
            channel_vec.push_back(ch_ctx.raw_channel);
        }
    }

    if (channel_vec.empty()) {
        return;
    }

    g_state.watchdog = std::make_shared<sync::content_sync>(
        g_state.frame_clock, channel_vec, g_state.config.content_sync_threshold);
    g_state.watchdog->start();
    g_state.watchdog_armed = true;

    // If content-sync is enabled in config, auto-track all channels
    if (g_state.config.content_sync_enabled) {
        for (int i = 0; i < static_cast<int>(channel_vec.size()); ++i) {
            g_state.watchdog->track_channel(i, g_state.config.content_sync_max_layer);
        }
        CASPAR_LOG(info) << L"[cluster] Content sync auto-tracking all " << channel_vec.size()
                         << L" channel(s), max_layer=" << g_state.config.content_sync_max_layer;
    }

    CASPAR_LOG(info) << L"[cluster] Content sync watchdog armed with " << channel_vec.size() << L" channel(s)";
}

// ─── AMCP command: CLUSTER TRACK ────────────────────────────────────────────
// Usage: CLUSTER TRACK <channel>[-<layer>] [<duration> [LOOP]]
//   CLUSTER TRACK 1       — track entire channel 1 (auto-discover layers)
//   CLUSTER TRACK 1-10    — track specific layer 10 on channel 1
// If duration/loop omitted, queries the producer for its own values.

std::wstring cluster_track_command(protocol::amcp::command_context& ctx)
{
    std::lock_guard<std::mutex> lock(g_state_mutex);

    if (!g_state.active) {
        return L"501 CLUSTER TRACK FAILED\r\n";
    }

    // Arm watchdog on first use
    arm_watchdog(ctx.channels);

    if (!g_state.watchdog) {
        return L"501 CLUSTER TRACK FAILED - NO WATCHDOG\r\n";
    }

    if (ctx.parameters.empty()) {
        return L"400 CLUSTER TRACK ERROR\r\n";
    }

    // Parse channel[-layer]
    int  channel_index = 0;
    int  layer_index   = -1; // -1 = whole channel
    auto dash = ctx.parameters[0].find(L'-');
    if (dash != std::wstring::npos) {
        try {
            channel_index = std::stoi(ctx.parameters[0].substr(0, dash)) - 1; // 1-based to 0-based
            layer_index   = std::stoi(ctx.parameters[0].substr(dash + 1));
        } catch (...) {
            return L"400 CLUSTER TRACK ERROR - INVALID CHANNEL\r\n";
        }
    } else {
        // No dash = whole channel
        try {
            channel_index = std::stoi(ctx.parameters[0]) - 1;
        } catch (...) {
            return L"400 CLUSTER TRACK ERROR - INVALID CHANNEL\r\n";
        }
    }

    // Whole-channel tracking
    if (layer_index < 0) {
        g_state.watchdog->track_channel(channel_index);
        std::wostringstream reply;
        reply << L"202 CLUSTER TRACK OK channel=" << (channel_index + 1) << L" (auto-discover)\r\n";
        return reply.str();
    }

    // Optional explicit duration and LOOP override
    int64_t duration = 0;
    bool    looping  = false;
    bool    auto_query = true;

    if (ctx.parameters.size() >= 2) {
        try {
            duration = std::stoll(ctx.parameters[1]);
            auto_query = false;
        } catch (...) {
            // Not a number — might be "LOOP" flag directly
            if (ctx.parameters[1] == L"LOOP" || ctx.parameters[1] == L"loop") {
                looping = true;
                auto_query = true;
            }
        }
    }
    if (ctx.parameters.size() > 2 && (ctx.parameters[2] == L"LOOP" || ctx.parameters[2] == L"loop")) {
        looping = true;
    }

    // Auto-query producer for duration and loop state
    if (auto_query && channel_index < static_cast<int>(ctx.channels->size())) {
        try {
            auto& ch_ctx = ctx.channels->at(channel_index);
            if (ch_ctx.raw_channel) {
                auto producer_future = ch_ctx.raw_channel->stage()->foreground(layer_index);
                if (producer_future.wait_for(std::chrono::milliseconds(100)) == std::future_status::ready) {
                    auto producer = producer_future.get();
                    if (producer) {
                        auto nb = producer->nb_frames();
                        if (nb != std::numeric_limits<uint32_t>::max()) {
                            duration = static_cast<int64_t>(nb);
                        }
                        // Query loop state via call
                        auto loop_future = ch_ctx.raw_channel->stage()->call(layer_index, {L"loop"});
                        if (loop_future.wait_for(std::chrono::milliseconds(50)) == std::future_status::ready) {
                            auto result = loop_future.get();
                            looping = (result == L"1" || result == L"true");
                        }
                    }
                }
            }
        } catch (...) {
            // Failed to query — use provided values or defaults
        }
    }

    // Start frame = current global frame (producer starts now)
    int64_t start_frame = g_state.frame_clock->current_frame();

    g_state.watchdog->track_producer(channel_index, layer_index, start_frame, duration, looping);

    std::wostringstream reply;
    reply << L"202 CLUSTER TRACK OK duration=" << duration
          << L" loop=" << (looping ? L"true" : L"false") << L"\r\n";
    return reply.str();
}

// ─── AMCP command: CLUSTER UNTRACK ──────────────────────────────────────────
// Usage: CLUSTER UNTRACK <channel>[-<layer>]
//   CLUSTER UNTRACK 1     — untrack entire channel 1 (all layers + auto-discover)
//   CLUSTER UNTRACK 1-10  — untrack specific layer

std::wstring cluster_untrack_command(protocol::amcp::command_context& ctx)
{
    std::lock_guard<std::mutex> lock(g_state_mutex);

    if (!g_state.active || !g_state.watchdog) {
        return L"501 CLUSTER UNTRACK FAILED\r\n";
    }

    if (ctx.parameters.empty()) {
        return L"400 CLUSTER UNTRACK ERROR\r\n";
    }

    int channel_index = 0;
    int layer_index   = -1;
    auto dash = ctx.parameters[0].find(L'-');
    if (dash != std::wstring::npos) {
        try {
            channel_index = std::stoi(ctx.parameters[0].substr(0, dash)) - 1;
            layer_index   = std::stoi(ctx.parameters[0].substr(dash + 1));
        } catch (...) {
            return L"400 CLUSTER UNTRACK ERROR - INVALID CHANNEL\r\n";
        }
    } else {
        try {
            channel_index = std::stoi(ctx.parameters[0]) - 1;
        } catch (...) {
            return L"400 CLUSTER UNTRACK ERROR - INVALID CHANNEL\r\n";
        }
    }

    if (layer_index < 0) {
        g_state.watchdog->untrack_channel(channel_index);
    } else {
        g_state.watchdog->untrack_producer(channel_index, layer_index);
    }
    return L"202 CLUSTER UNTRACK OK\r\n";
}

// ─── Startup logic ──────────────────────────────────────────────────────────

void start_cluster(const cluster_config& config,
                   const std::shared_ptr<protocol::amcp::amcp_command_repository_wrapper>& command_repo)
{
    std::lock_guard<std::mutex> lock(g_state_mutex);

    g_state.config = config;

    // 1. Create PTP clock
    auto ptp_mode = (config.mode == cluster_mode::master) ? ptp::clock_mode::master :
                    (config.mode == cluster_mode::external) ? ptp::clock_mode::external :
                    ptp::clock_mode::client;

    g_state.ptp = std::make_shared<ptp::ptp_clock>(
        ptp_mode, config.bind_address, config.multicast_group,
        config.ptp_domain, config.sync_interval_ms);
    g_state.ptp->start();

    // 2. Create frame clock (default 50fps, will be updated from channel config)
    g_state.frame_clock = std::make_shared<sync::frame_clock>(
        g_state.ptp, config.epoch_origin_ns, 50, 1);

    // 3. Create command scheduler with executor that parses AMCP
    // The executor feeds commands back through the AMCP parser
    sync::command_executor executor = [command_repo](const std::wstring& cmd) {
        // For now, log execution. Full integration requires parser access.
        CASPAR_LOG(debug) << L"[cluster] Execute@frame: " << cmd;
        // TODO: Wire to amcp_command_repository::parse_and_execute()
    };

    g_state.scheduler = std::make_shared<sync::command_scheduler>(g_state.frame_clock, executor);
    g_state.scheduler->start();

    // 4. Setup channel map and relay
    g_state.channel_map = std::make_unique<relay::virtual_channel_map>();
    for (const auto& ch : config.channels) {
        g_state.channel_map->add_mapping(ch.virtual_channel, ch.host, ch.physical_channel);
    }

    g_state.relay = std::make_unique<relay::command_relay>(*g_state.channel_map, config.sync_margin);

    if (config.mode == cluster_mode::master) {
        // Master: connect to all members
        if (!config.members.empty()) {
            g_state.relay->connect_members(config.members);
        }
    } else if (config.mode == cluster_mode::client) {
        // Client: listen for master connection and feed commands to scheduler
        g_state.relay->set_command_handler(
            [](int64_t target_frame, const std::wstring& command) {
                std::lock_guard<std::mutex> lock(g_state_mutex);
                if (g_state.scheduler) {
                    g_state.scheduler->schedule(target_frame, std::wstring(command));
                }
            });
        g_state.relay->start_client_listener(config.bind_address, config.relay_port);
    }

    g_state.active = true;

    CASPAR_LOG(info) << L"[cluster] Cluster started, mode="
                     << (config.mode == cluster_mode::master ? L"master" : L"client")
                     << L", sync_margin=" << config.sync_margin
                     << L", channels=" << config.channels.size()
                     << L", members=" << config.members.size();
}

} // anonymous namespace

// ─── Module init ────────────────────────────────────────────────────────────

void init(const core::module_dependencies& dependencies)
{
    // Register AMCP commands
    if (dependencies.command_repository) {
        dependencies.command_repository->register_command(
            L"Cluster Commands", L"CLUSTER STATUS", cluster_status_command, 0);
        dependencies.command_repository->register_command(
            L"Cluster Commands", L"CLUSTER SCHEDULE", cluster_schedule_command, 1);
        dependencies.command_repository->register_command(
            L"Cluster Commands", L"CLUSTER TRACK", cluster_track_command, 2);
        dependencies.command_repository->register_command(
            L"Cluster Commands", L"CLUSTER UNTRACK", cluster_untrack_command, 1);
    }

    // Parse config and start cluster if configured
    auto config = parse_cluster_config(env::properties());

    if (config.mode != cluster_mode::disabled) {
        start_cluster(config, dependencies.command_repository);
    } else {
        CASPAR_LOG(info) << L"[cluster] No <cluster> block in config, module idle";
    }
}

}} // namespace caspar::cluster
