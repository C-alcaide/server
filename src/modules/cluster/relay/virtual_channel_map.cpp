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

    // THE CHANNEL IS ALWAYS THE SECOND TOKEN. AMCP's command repository pops the command's FIRST
    // word and parses the channel from the token right after it (`parse_channel_id`, which takes
    // `tokens.front()`), and only then matches the sub-verb: `MIXER 3-10 OPACITY`, `CG 3-10 ADD`,
    // `PREVIZ 3 SCREEN ...`, `MIXER 3 COMMIT`. This used to treat DATA, CG, THUMBNAIL and MIXER as
    // two-word verbs and look for the channel in the THIRD token -- so for MIXER and CG it
    // inspected the sub-verb, found no channel, and forwarded the command with the VIRTUAL number.
    // Every MIXER and CG command to a remapped channel ran on the wrong channel of the remote node.
    // (DATA and THUMBNAIL carry no channel at all, so the list was wrong for every entry.)
    //
    // SWAP's second channel is NOT rewritten: a swap between two channels is only meaningful on
    // one node, and a virtual pair that maps to two different hosts has no swap to perform.
    std::wstring vchan_str = std::to_wstring(virtual_channel);
    std::wstring pchan_str = std::to_wstring(physical);

    // Find the first token (verb)
    auto first_space = command.find(L' ');
    if (first_space == std::wstring::npos) {
        return command; // Single-word command, nothing to rewrite
    }

    const int tokens_to_skip = 1;

    // Skip the specified number of tokens to reach the channel token
    size_t pos = 0;
    for (int i = 0; i < tokens_to_skip; ++i) {
        pos = command.find(L' ', pos);
        if (pos == std::wstring::npos) {
            return command;
        }
        pos = command.find_first_not_of(L' ', pos);
        if (pos == std::wstring::npos) {
            return command;
        }
    }

    auto token_start = pos;
    auto token_end   = command.find(L' ', token_start);
    if (token_end == std::wstring::npos) {
        token_end = command.size();
    }

    std::wstring token = command.substr(token_start, token_end - token_start);

    // Check if token starts with the virtual channel number followed by '-' or end
    if (token == vchan_str) {
        // Bare channel number: "COMMAND 2" → "COMMAND 1"
        return command.substr(0, token_start) + pchan_str + command.substr(token_end);
    }

    if (token.size() > vchan_str.size() && token.substr(0, vchan_str.size()) == vchan_str &&
        token[vchan_str.size()] == L'-') {
        // Channel-layer: "COMMAND 2-10" → "COMMAND 1-10"
        std::wstring layer_part = token.substr(vchan_str.size()); // "-10"
        return command.substr(0, token_start) + pchan_str + layer_part + command.substr(token_end);
    }

    return command; // Token doesn't match virtual channel
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
