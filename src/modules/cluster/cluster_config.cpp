/*
 * Copyright (c) 2024 CasparCG contributors
 *
 * CasparCG is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#include "cluster_config.h"

#include <common/except.h>
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
        // A WORD THAT NAMES NO MODE IS REFUSED. It used to fall to `disabled`, so a node whose
        // config said `<mode>Master</mode>` or `<mode>clinet</mode>` started standalone and said
        // nothing -- a cluster member quietly not in the cluster.
        auto mode_str = node.get<std::wstring>(L"mode", L"disabled");
        if (mode_str == L"master") cfg.mode = cluster_mode::master;
        else if (mode_str == L"client") cfg.mode = cluster_mode::client;
        else if (mode_str == L"external") cfg.mode = cluster_mode::external;
        else if (mode_str == L"disabled") cfg.mode = cluster_mode::disabled;
        else
            CASPAR_THROW_EXCEPTION(user_error() << msg_info(
                L"<cluster><mode> must be master, client, external or disabled, not '" + mode_str + L"'"));

        if (cfg.mode == cluster_mode::disabled) {
            return cfg;
        }

        // Common settings
        cfg.sync_margin = node.get<int>(L"sync-margin", 3);
        auto bind_w = node.get<std::wstring>(L"bind", L"0.0.0.0");
        cfg.bind_address = narrow(bind_w);

        auto mcast_w = node.get<std::wstring>(L"multicast-group", L"224.0.1.129");
        cfg.multicast_group = narrow(mcast_w);

        // VALIDATED BEFORE IT IS NARROWED. `static_cast<uint8_t>` ran first, so 256 became
        // domain 0 -- the default domain, every other PTP device's -- and passed the range check.
        const int ptp_domain = node.get<int>(L"ptp-domain", 0);
        if (ptp_domain < 0 || ptp_domain > 127)
            CASPAR_THROW_EXCEPTION(user_error() << msg_info(
                "<cluster><ptp-domain> must be 0..127, not " + std::to_string(ptp_domain)));
        cfg.ptp_domain = static_cast<uint8_t>(ptp_domain);
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
        // 5252, NOT 5250. The old default was AMCP's own port, so a client node that did not
        // set this explicitly asked its relay to bind the port AMCP already had -- which on
        // Windows succeeded, silently, and left the relay eating AMCP's clients. 5253 is the
        // config's own second AMCP listener, so 5252 is the first free neighbour.
        const int relay_port = node.get<int>(L"relay-port", 5252);
        if (relay_port < 1 || relay_port > 65535)   // 70000 became port 4464
            CASPAR_THROW_EXCEPTION(user_error() << msg_info(
                "<cluster><relay-port> must be 1..65535, not " + std::to_string(relay_port)));
        cfg.relay_port = static_cast<uint16_t>(relay_port);

        // `<master>` IS NOT READ BY ANYTHING: a client is reached by its master through the
        // master's `<members>`, never the other way round. Said, because CLUSTER_SYNC.md
        // documented it as the client's master address.
        if (!cfg.master_address.empty() && cfg.mode == cluster_mode::client)
            CASPAR_LOG(warning) << L"[cluster] <master> is not used: a client is reached by the master "
                                   L"through the master's <members> list. Remove it, or list this node there.";

    } catch (const user_error&) {
        throw;
    } catch (const std::exception& e) {
        // A PARSE ERROR PART-WAY THROUGH is refused too. It was logged and the half-filled config
        // RETURNED -- a master running with some of its channels and none of its members.
        CASPAR_THROW_EXCEPTION(user_error() << msg_info(
            std::string("[cluster] <cluster> could not be parsed: ") + e.what()));
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

    return cfg;
}

}} // namespace caspar::cluster
