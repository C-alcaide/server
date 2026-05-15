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

#include "portaudio_device.h"

#include <common/log.h>
#include <common/utf.h>

#include <portaudio.h>

#include <algorithm>

namespace caspar { namespace portaudio {

portaudio_device_manager& portaudio_device_manager::instance()
{
    static portaudio_device_manager inst;
    return inst;
}

portaudio_device_manager::~portaudio_device_manager()
{
    shutdown();
}

void portaudio_device_manager::initialize()
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (initialized_)
        return;

    PaError err = Pa_Initialize();
    if (err != paNoError) {
        CASPAR_LOG(error) << L"[portaudio] Failed to initialize PortAudio: " << Pa_GetErrorText(err);
        return;
    }

    initialized_ = true;

    // Log available host APIs and devices
    int api_count = Pa_GetHostApiCount();
    CASPAR_LOG(info) << L"[portaudio] Initialized. Host APIs: " << api_count;
    for (int i = 0; i < api_count; ++i) {
        const PaHostApiInfo* api_info = Pa_GetHostApiInfo(i);
        if (api_info) {
            CASPAR_LOG(info) << L"[portaudio]   API " << i << L": " << api_info->name
                            << L" (devices: " << api_info->deviceCount << L")";
        }
    }

    int device_count = Pa_GetDeviceCount();
    CASPAR_LOG(info) << L"[portaudio] Total devices: " << device_count;
    for (int i = 0; i < device_count; ++i) {
        const PaDeviceInfo* dev_info = Pa_GetDeviceInfo(i);
        if (dev_info) {
            const PaHostApiInfo* api_info = Pa_GetHostApiInfo(dev_info->hostApi);
            CASPAR_LOG(debug) << L"[portaudio]   Device " << i << L": " << dev_info->name
                             << L" [" << (api_info ? api_info->name : "?") << L"]"
                             << L" in=" << dev_info->maxInputChannels
                             << L" out=" << dev_info->maxOutputChannels;
        }
    }
}

void portaudio_device_manager::shutdown()
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized_)
        return;

    Pa_Terminate();
    initialized_ = false;
    CASPAR_LOG(info) << L"[portaudio] Shutdown.";
}

bool portaudio_device_manager::is_initialized() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return initialized_;
}

bool portaudio_device_manager::matches_host_api(int device_api, host_api_preference preference) const
{
    switch (preference) {
        case host_api_preference::asio:        return device_api == paASIO;
        case host_api_preference::wasapi:      return device_api == paWASAPI;
        case host_api_preference::directsound: return device_api == paDirectSound;
        case host_api_preference::auto_select: return true;
    }
    return true;
}

std::vector<device_info> portaudio_device_manager::enumerate_output_devices(host_api_preference preference) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<device_info> result;
    if (!initialized_)
        return result;

    int device_count = Pa_GetDeviceCount();
    for (int i = 0; i < device_count; ++i) {
        const PaDeviceInfo* dev_info = Pa_GetDeviceInfo(i);
        if (!dev_info || dev_info->maxOutputChannels <= 0)
            continue;

        const PaHostApiInfo* api_info = Pa_GetHostApiInfo(dev_info->hostApi);
        PaHostApiTypeId api_type = api_info ? api_info->type : paInDevelopment;

        if (!matches_host_api(api_type, preference))
            continue;

        device_info info;
        info.index                    = i;
        info.name                     = dev_info->name;
        info.host_api_name            = api_info ? api_info->name : "Unknown";
        info.host_api_type            = api_type;
        info.max_input_channels       = dev_info->maxInputChannels;
        info.max_output_channels      = dev_info->maxOutputChannels;
        info.default_sample_rate      = dev_info->defaultSampleRate;
        info.default_low_input_latency  = dev_info->defaultLowInputLatency;
        info.default_low_output_latency = dev_info->defaultLowOutputLatency;
        result.push_back(info);
    }
    return result;
}

std::vector<device_info> portaudio_device_manager::enumerate_input_devices(host_api_preference preference) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<device_info> result;
    if (!initialized_)
        return result;

    int device_count = Pa_GetDeviceCount();
    for (int i = 0; i < device_count; ++i) {
        const PaDeviceInfo* dev_info = Pa_GetDeviceInfo(i);
        if (!dev_info || dev_info->maxInputChannels <= 0)
            continue;

        const PaHostApiInfo* api_info = Pa_GetHostApiInfo(dev_info->hostApi);
        PaHostApiTypeId api_type = api_info ? api_info->type : paInDevelopment;

        if (!matches_host_api(api_type, preference))
            continue;

        device_info info;
        info.index                    = i;
        info.name                     = dev_info->name;
        info.host_api_name            = api_info ? api_info->name : "Unknown";
        info.host_api_type            = api_type;
        info.max_input_channels       = dev_info->maxInputChannels;
        info.max_output_channels      = dev_info->maxOutputChannels;
        info.default_sample_rate      = dev_info->defaultSampleRate;
        info.default_low_input_latency  = dev_info->defaultLowInputLatency;
        info.default_low_output_latency = dev_info->defaultLowOutputLatency;
        result.push_back(info);
    }
    return result;
}

int portaudio_device_manager::find_output_device(const std::string& device_name, host_api_preference preference) const
{
    auto devices = enumerate_output_devices(preference);

    // Exact match first
    for (const auto& dev : devices) {
        if (dev.name == device_name)
            return dev.index;
    }

    // Partial match (case-insensitive substring)
    std::string lower_search = device_name;
    std::transform(lower_search.begin(), lower_search.end(), lower_search.begin(), ::tolower);

    for (const auto& dev : devices) {
        std::string lower_name = dev.name;
        std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(), ::tolower);
        if (lower_name.find(lower_search) != std::string::npos)
            return dev.index;
    }

    return -1;
}

int portaudio_device_manager::find_input_device(const std::string& device_name, host_api_preference preference) const
{
    auto devices = enumerate_input_devices(preference);

    // Exact match first
    for (const auto& dev : devices) {
        if (dev.name == device_name)
            return dev.index;
    }

    // Partial match (case-insensitive substring)
    std::string lower_search = device_name;
    std::transform(lower_search.begin(), lower_search.end(), lower_search.begin(), ::tolower);

    for (const auto& dev : devices) {
        std::string lower_name = dev.name;
        std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(), ::tolower);
        if (lower_name.find(lower_search) != std::string::npos)
            return dev.index;
    }

    return -1;
}

int portaudio_device_manager::get_preferred_host_api(host_api_preference preference) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized_)
        return -1;

    if (preference == host_api_preference::auto_select) {
        // Try ASIO first, then WASAPI, then default
        PaHostApiIndex asio_idx = Pa_HostApiTypeIdToHostApiIndex(paASIO);
        if (asio_idx >= 0)
            return asio_idx;

        PaHostApiIndex wasapi_idx = Pa_HostApiTypeIdToHostApiIndex(paWASAPI);
        if (wasapi_idx >= 0)
            return wasapi_idx;

        return Pa_GetDefaultHostApi();
    }

    PaHostApiTypeId type;
    switch (preference) {
        case host_api_preference::asio:        type = paASIO; break;
        case host_api_preference::wasapi:      type = paWASAPI; break;
        case host_api_preference::directsound: type = paDirectSound; break;
        default:                               return Pa_GetDefaultHostApi();
    }

    return Pa_HostApiTypeIdToHostApiIndex(type);
}

int portaudio_device_manager::get_default_output_device(host_api_preference preference) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized_)
        return -1;

    PaHostApiIndex api_idx = -1;

    if (preference == host_api_preference::auto_select) {
        // Try ASIO first
        api_idx = Pa_HostApiTypeIdToHostApiIndex(paASIO);
        if (api_idx >= 0) {
            const PaHostApiInfo* api_info = Pa_GetHostApiInfo(api_idx);
            if (api_info && api_info->defaultOutputDevice >= 0)
                return Pa_HostApiDeviceIndexToDeviceIndex(api_idx, api_info->defaultOutputDevice);
        }
        // Then WASAPI
        api_idx = Pa_HostApiTypeIdToHostApiIndex(paWASAPI);
        if (api_idx >= 0) {
            const PaHostApiInfo* api_info = Pa_GetHostApiInfo(api_idx);
            if (api_info && api_info->defaultOutputDevice >= 0)
                return Pa_HostApiDeviceIndexToDeviceIndex(api_idx, api_info->defaultOutputDevice);
        }
        // Fallback to system default
        return Pa_GetDefaultOutputDevice();
    }

    PaHostApiTypeId type;
    switch (preference) {
        case host_api_preference::asio:        type = paASIO; break;
        case host_api_preference::wasapi:      type = paWASAPI; break;
        case host_api_preference::directsound: type = paDirectSound; break;
        default:                               return Pa_GetDefaultOutputDevice();
    }

    api_idx = Pa_HostApiTypeIdToHostApiIndex(type);
    if (api_idx >= 0) {
        const PaHostApiInfo* api_info = Pa_GetHostApiInfo(api_idx);
        if (api_info && api_info->defaultOutputDevice >= 0)
            return Pa_HostApiDeviceIndexToDeviceIndex(api_idx, api_info->defaultOutputDevice);
    }

    return -1;
}

int portaudio_device_manager::get_default_input_device(host_api_preference preference) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized_)
        return -1;

    PaHostApiIndex api_idx = -1;

    if (preference == host_api_preference::auto_select) {
        api_idx = Pa_HostApiTypeIdToHostApiIndex(paASIO);
        if (api_idx >= 0) {
            const PaHostApiInfo* api_info = Pa_GetHostApiInfo(api_idx);
            if (api_info && api_info->defaultInputDevice >= 0)
                return Pa_HostApiDeviceIndexToDeviceIndex(api_idx, api_info->defaultInputDevice);
        }
        api_idx = Pa_HostApiTypeIdToHostApiIndex(paWASAPI);
        if (api_idx >= 0) {
            const PaHostApiInfo* api_info = Pa_GetHostApiInfo(api_idx);
            if (api_info && api_info->defaultInputDevice >= 0)
                return Pa_HostApiDeviceIndexToDeviceIndex(api_idx, api_info->defaultInputDevice);
        }
        return Pa_GetDefaultInputDevice();
    }

    PaHostApiTypeId type;
    switch (preference) {
        case host_api_preference::asio:        type = paASIO; break;
        case host_api_preference::wasapi:      type = paWASAPI; break;
        case host_api_preference::directsound: type = paDirectSound; break;
        default:                               return Pa_GetDefaultInputDevice();
    }

    api_idx = Pa_HostApiTypeIdToHostApiIndex(type);
    if (api_idx >= 0) {
        const PaHostApiInfo* api_info = Pa_GetHostApiInfo(api_idx);
        if (api_info && api_info->defaultInputDevice >= 0)
            return Pa_HostApiDeviceIndexToDeviceIndex(api_idx, api_info->defaultInputDevice);
    }

    return -1;
}

host_api_preference portaudio_device_manager::parse_host_api(const std::string& str)
{
    std::string lower = str;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

    if (lower == "asio")        return host_api_preference::asio;
    if (lower == "wasapi")      return host_api_preference::wasapi;
    if (lower == "directsound") return host_api_preference::directsound;
    if (lower == "ds")          return host_api_preference::directsound;
    return host_api_preference::auto_select;
}

}} // namespace caspar::portaudio
