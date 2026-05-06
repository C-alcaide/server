/*
 * Copyright (c) 2024 CasparCG contributors
 *
 * CasparCG is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#pragma once

#include <boost/property_tree/ptree.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace caspar { namespace cluster {

enum class cluster_mode
{
    disabled, // No cluster functionality
    master,   // This instance is the cluster master (PTP master + command relay)
    client,   // This instance syncs to a master
    external, // Lock to external PTP grandmaster, but run as independent node
};

struct channel_config
{
    int         virtual_channel  = 0;
    std::string host             = "local";
    int         physical_channel = 0;
};

struct cluster_config
{
    cluster_mode mode = cluster_mode::disabled;

    // Sync settings
    int         sync_margin      = 3;      // Frames of margin for command scheduling
    std::string bind_address     = "0.0.0.0";
    std::string multicast_group  = "224.0.1.129";
    uint8_t     ptp_domain       = 0;
    int64_t     epoch_origin_ns  = 0;      // Nanoseconds since Unix epoch (parsed from ISO datetime)
    int         sync_interval_ms = 125;    // PTP sync message interval

    // Master-specific
    std::vector<channel_config> channels;
    std::vector<std::string>    members;   // "host:port" of cluster members

    // Client-specific
    std::string master_address;            // "host:port" of the master
    uint16_t    relay_port = 5250;         // Port for relay listener

    // Content sync
    bool content_sync_enabled  = false;    // Auto-discover all layers on all channels
    int  content_sync_threshold = 2;       // Drift threshold in frames before correction
    int  content_sync_max_layer = 100;     // Max layer index to scan per channel

    // Diagnostics
    bool log_ptp_status = false;           // Periodic PTP status logging
};

/// Parse <cluster> block from the already-loaded config property tree
cluster_config parse_cluster_config(const boost::property_tree::wptree& properties);

/// Parse epoch origin from ISO 8601 datetime string to nanoseconds since Unix epoch
int64_t parse_epoch_origin(const std::string& iso_datetime);

}} // namespace caspar::cluster
