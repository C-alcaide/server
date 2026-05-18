/*
 * Copyright (c) 2025 CasparCG Contributors
 *
 * This file is part of CasparCG (www.casparcg.com).
 *
 * CasparCG is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * CasparCG is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with CasparCG. If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <mutex>

namespace caspar { namespace portaudio {

class shared_portaudio_capture;

enum class host_api_preference
{
    auto_select, // ASIO if available, then WASAPI, then default
    asio,
    wasapi,
    directsound
};

struct device_info
{
    int              index;
    std::string      name;
    std::string      host_api_name;
    int              host_api_type;
    int              max_input_channels;
    int              max_output_channels;
    double           default_sample_rate;
    double           default_low_input_latency;
    double           default_low_output_latency;
};

class portaudio_device_manager
{
  public:
    static portaudio_device_manager& instance();

    void initialize();
    void shutdown();
    bool is_initialized() const;

    /// List all available output devices, optionally filtered by host API.
    std::vector<device_info> enumerate_output_devices(host_api_preference preference = host_api_preference::auto_select) const;

    /// List all available input (capture) devices, optionally filtered by host API.
    std::vector<device_info> enumerate_input_devices(host_api_preference preference = host_api_preference::auto_select) const;

    /// Find an output device by name (partial match). Respects host API preference.
    /// Returns -1 if not found.
    int find_output_device(const std::string& device_name, host_api_preference preference = host_api_preference::auto_select) const;

    /// Find an input device by name (partial match). Respects host API preference.
    /// Returns -1 if not found.
    int find_input_device(const std::string& device_name, host_api_preference preference = host_api_preference::auto_select) const;

    /// Get the preferred host API index based on preference. Returns -1 on failure.
    int get_preferred_host_api(host_api_preference preference) const;

    /// Get the default output device for a given host API preference.
    int get_default_output_device(host_api_preference preference = host_api_preference::auto_select) const;

    /// Get the default input device for a given host API preference.
    int get_default_input_device(host_api_preference preference = host_api_preference::auto_select) const;

    /// Parse host API preference from string (e.g., "asio", "wasapi", "auto").
    static host_api_preference parse_host_api(const std::string& str);

    /// Get or create a shared capture stream for the given device.
    /// Multiple producers targeting the same device share a single PaStream.
    /// The shared_ptr ref-counting manages the lifecycle automatically.
    std::shared_ptr<shared_portaudio_capture> get_shared_capture(int device_index, int channels, int sample_rate);

  private:
    portaudio_device_manager() = default;
    ~portaudio_device_manager();

    portaudio_device_manager(const portaudio_device_manager&)            = delete;
    portaudio_device_manager& operator=(const portaudio_device_manager&) = delete;

    bool matches_host_api(int device_api, host_api_preference preference) const;

    mutable std::mutex mutex_;
    bool               initialized_ = false;

    // Shared capture streams: one per device_index, weak_ptr so they auto-expire
    std::map<int, std::weak_ptr<shared_portaudio_capture>> shared_captures_;
};

}} // namespace caspar::portaudio
