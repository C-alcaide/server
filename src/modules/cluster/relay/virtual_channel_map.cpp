/*
 * Copyright (c) 2024 CasparCG contributors
 *
 * CasparCG is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#include "virtual_channel_map.h"

#include <algorithm>
#include <regex>
#include <set>
#include <sstream>

namespace caspar { namespace cluster { namespace relay {

void virtual_channel_map::add_mapping(int virtual_channel, const std::string& host, int physical_channel)
{
    channel_mapping m;
    m.virtual_channel  = virtual_channel;
    m.host             = host;
    m.physical_channel = physical_channel;
    mappings_[virtual_channel] = m;
}

void virtual_channel_map::clear()
{
    mappings_.clear();
}

std::string virtual_channel_map::get_host(int virtual_channel) const
{
    auto it = mappings_.find(virtual_channel);
    if (it == mappings_.end()) {
        return "local";
    }
    return it->second.host;
}

int virtual_channel_map::get_physical_channel(int virtual_channel) const
{
    auto it = mappings_.find(virtual_channel);
    if (it == mappings_.end()) {
        return virtual_channel; // Identity mapping if not configured
    }
    return it->second.physical_channel;
}

bool virtual_channel_map::is_local(int virtual_channel) const
{
    return get_host(virtual_channel) == "local";
}

std::wstring virtual_channel_map::rewrite_command(const std::wstring& command, int virtual_channel) const
{
    int physical = get_physical_channel(virtual_channel);
    if (physical == virtual_channel) {
        return command; // No rewrite needed
    }

    // AMCP commands use format: "COMMAND channel-layer ..."
    // Match pattern like "2-10" or just "2" and replace the channel number
    std::wstring vchan_str = std::to_wstring(virtual_channel);
    std::wstring pchan_str = std::to_wstring(physical);

    // Replace first occurrence of "vchan-" with "pchan-" or " vchan " with " pchan "
    std::wstring result = command;

    // Pattern: space + virtual_channel + dash + layer
    std::wstring pattern = L" " + vchan_str + L"-";
    std::wstring replacement = L" " + pchan_str + L"-";
    auto pos = result.find(pattern);
    if (pos != std::wstring::npos) {
        result.replace(pos, pattern.size(), replacement);
    }

    return result;
}

std::vector<int> virtual_channel_map::channels_for_host(const std::string& host) const
{
    std::vector<int> result;
    for (const auto& [vch, mapping] : mappings_) {
        if (mapping.host == host) {
            result.push_back(vch);
        }
    }
    return result;
}

std::vector<std::string> virtual_channel_map::remote_hosts() const
{
    std::set<std::string> hosts;
    for (const auto& [vch, mapping] : mappings_) {
        if (mapping.host != "local") {
            hosts.insert(mapping.host);
        }
    }
    return {hosts.begin(), hosts.end()};
}

}}} // namespace caspar::cluster::relay
