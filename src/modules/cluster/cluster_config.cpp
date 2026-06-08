/*
 * Copyright (c) 2024 CasparCG contributors
 *
 * CasparCG is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#include "cluster_config.h"

#include <common/log.h>
#include <common/utf.h>

#include <algorithm>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>

namespace caspar { namespace cluster {

namespace {

std::string narrow(const std::wstring& ws)
{
    return u8(ws);
}

} // anonymous namespace

int64_t parse_epoch_origin(const std::string& iso_datetime)
{
    if (iso_datetime.empty()) {
        return 0;
    }

    std::tm tm = {};
    std::istringstream ss(iso_datetime);
    ss >> std::get_time(&tm, "%Y-%m-%dT%H:%M:%S");
    if (ss.fail()) {
        CASPAR_LOG(warning) << L"[cluster] Failed to parse epoch-origin: "
                            << std::wstring(iso_datetime.begin(), iso_datetime.end());
        return 0;
    }

    // Convert to time_t (UTC)
    #ifdef _WIN32
    time_t t = _mkgmtime(&tm);
    #else
    time_t t = timegm(&tm);
    #endif

    return static_cast<int64_t>(t) * 1'000'000'000LL;
}

cluster_config parse_cluster_config(const boost::property_tree::wptree& properties)
{
    cluster_config cfg;

    try {
        auto cluster_node = properties.get_child_optional(L"configuration.cluster");
        if (!cluster_node) {
            return cfg; // No cluster block = disabled
        }

        auto& node = *cluster_node;

        // Mode
        auto mode_str = node.get<std::wstring>(L"mode", L"disabled");
        if (mode_str == L"master") cfg.mode = cluster_mode::master;
        else if (mode_str == L"client") cfg.mode = cluster_mode::client;
        else if (mode_str == L"external") cfg.mode = cluster_mode::external;
        else cfg.mode = cluster_mode::disabled;

        if (cfg.mode == cluster_mode::disabled) {
            return cfg;
        }

        // Common settings
        cfg.sync_margin = node.get<int>(L"sync-margin", 3);
        auto bind_w = node.get<std::wstring>(L"bind", L"0.0.0.0");
        cfg.bind_address = narrow(bind_w);

        auto mcast_w = node.get<std::wstring>(L"multicast-group", L"224.0.1.129");
        cfg.multicast_group = narrow(mcast_w);

        cfg.ptp_domain = static_cast<uint8_t>(node.get<int>(L"ptp-domain", 0));
        cfg.sync_interval_ms = node.get<int>(L"sync-interval-ms", 125);
        cfg.log_ptp_status = node.get<bool>(L"log-ptp-status", false);

        // Content sync
        cfg.content_sync_enabled   = node.get<bool>(L"content-sync", false);
        cfg.content_sync_threshold = node.get<int>(L"content-sync-threshold", 2);
        cfg.content_sync_max_layer = node.get<int>(L"content-sync-max-layer", 100);

        // Epoch origin
        auto epoch_w = node.get<std::wstring>(L"epoch-origin", L"");
        if (!epoch_w.empty()) {
            cfg.epoch_origin_ns = parse_epoch_origin(narrow(epoch_w));
        }

        // Master: channel mappings
        if (auto channels_node = node.get_child_optional(L"channels")) {
            for (auto& [key, child] : *channels_node) {
                if (key == L"channel") {
                    channel_config ch;
                    ch.virtual_channel  = child.get<int>(L"<xmlattr>.virtual", 0);
                    auto host_w         = child.get<std::wstring>(L"<xmlattr>.host", L"local");
                    ch.host             = narrow(host_w);
                    ch.physical_channel = child.get<int>(L"<xmlattr>.physical", ch.virtual_channel);
                    if (ch.virtual_channel > 0) {
                        cfg.channels.push_back(ch);
                    }
                }
            }
        }

        // Master: member list
        if (auto members_node = node.get_child_optional(L"members")) {
            for (auto& [key, child] : *members_node) {
                if (key == L"member") {
                    auto val_w = child.get_value<std::wstring>();
                    cfg.members.push_back(narrow(val_w));
                }
            }
        }

        // Client: master address
        auto master_w = node.get<std::wstring>(L"master", L"");
        cfg.master_address = narrow(master_w);
        cfg.relay_port = static_cast<uint16_t>(node.get<int>(L"relay-port", 5250));

    } catch (const std::exception& e) {
        CASPAR_LOG(error) << L"[cluster] Config parse error: " << e.what();
    }

    // Validate ranges
    if (cfg.sync_margin < 0 || cfg.sync_margin > 100) {
        CASPAR_LOG(warning) << L"[cluster] sync-margin " << cfg.sync_margin << L" out of range [0..100], clamping";
        cfg.sync_margin = std::clamp(cfg.sync_margin, 0, 100);
    }
    if (cfg.sync_interval_ms < 1 || cfg.sync_interval_ms > 10000) {
        CASPAR_LOG(warning) << L"[cluster] sync-interval-ms " << cfg.sync_interval_ms
                            << L" out of range [1..10000], clamping";
        cfg.sync_interval_ms = std::clamp(cfg.sync_interval_ms, 1, 10000);
    }
    if (cfg.content_sync_threshold < 1 || cfg.content_sync_threshold > 50) {
        CASPAR_LOG(warning) << L"[cluster] content-sync-threshold " << cfg.content_sync_threshold
                            << L" out of range [1..50], clamping";
        cfg.content_sync_threshold = std::clamp(cfg.content_sync_threshold, 1, 50);
    }
    if (cfg.content_sync_max_layer < 1 || cfg.content_sync_max_layer > 10000) {
        CASPAR_LOG(warning) << L"[cluster] content-sync-max-layer " << cfg.content_sync_max_layer
                            << L" out of range [1..10000], clamping";
        cfg.content_sync_max_layer = std::clamp(cfg.content_sync_max_layer, 1, 10000);
    }
    if (cfg.ptp_domain > 127) {
        CASPAR_LOG(warning) << L"[cluster] ptp-domain " << static_cast<int>(cfg.ptp_domain)
                            << L" out of range [0..127], clamping";
        cfg.ptp_domain = 127;
    }

    return cfg;
}

}} // namespace caspar::cluster
