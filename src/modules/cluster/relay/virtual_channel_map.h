/*
 * Copyright (c) 2024 CasparCG contributors
 *
 * CasparCG is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#pragma once

#include <map>
#include <string>
#include <vector>

namespace caspar { namespace cluster { namespace relay {

/// A mapping from virtual channel/layer to a physical channel on a specific host
struct channel_mapping
{
    int         virtual_channel = 0;
    std::string host;          // "local" or "ip:port"
    int         physical_channel = 0;
};

/// Manages virtual→physical channel mapping for the cluster.
/// On master: rewrites channel references in commands before forwarding.
class virtual_channel_map
{
  public:
    virtual_channel_map() = default;

    /// Add a channel mapping
    void add_mapping(int virtual_channel, const std::string& host, int physical_channel);

    /// Clear all mappings
    void clear();

    /// Get the host for a virtual channel
    std::string get_host(int virtual_channel) const;

    /// Get the physical channel for a virtual channel
    int get_physical_channel(int virtual_channel) const;

    /// Check if a virtual channel maps to local
    bool is_local(int virtual_channel) const;

    /// Rewrite a command's channel reference from virtual to physical
    /// e.g., "PLAY 2-1 clip" with virtual=2→physical=1 becomes "PLAY 1-1 clip"
    std::wstring rewrite_command(const std::wstring& command, int virtual_channel) const;

    /// Get all virtual channels that map to a specific host
    std::vector<int> channels_for_host(const std::string& host) const;

    /// Get all unique remote hosts
    std::vector<std::string> remote_hosts() const;

    /// Get total number of mappings
    size_t size() const { return mappings_.size(); }

  private:
    std::map<int, channel_mapping> mappings_; // keyed by virtual_channel
};

}}} // namespace caspar::cluster::relay
