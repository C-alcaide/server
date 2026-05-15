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
 *
 * This module links against libltc (https://github.com/x42/libltc),
 * licensed under LGPL-2.1+, which is compatible with GPL-3.
 */

#include <ltc.h>
#include <decoder.h>
#include "ltc_input.h"
#include "ltc.h"
#include <mutex>
#include <atomic>
#include <string>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <chrono>
#include <ctime>

#include <common/env.h>
#include <common/log.h>
#include <common/utf.h>
#include <boost/property_tree/ptree.hpp>

#include <portaudio.h>

namespace caspar { namespace ltc {
    
static const wchar_t* LTC_CONFIG_ROOT = L"configuration.ltc";

class LTCInputImpl {
    PaStream*   stream_  = nullptr;
    LTCDecoder* decoder_ = nullptr;
    
    // Atomic double-buffer for timecode to avoid mutex in audio callback
    struct TimecodeSlot {
        SMPTETimecode tc = {0};
        std::chrono::steady_clock::time_point signal_time;
        bool has_data = false;
    };
    std::atomic<int> active_slot_{0};       // Index of the slot being READ by consumers
    TimecodeSlot slots_[2];                 // Double-buffer: audio writes to !active, then swaps
    std::atomic<bool> valid_signal{false};
    std::atomic<bool> running{false};
    
    bool pa_initialized_ = false;

    // Device Management
    std::string current_device_name;
    int         current_device_index_ = -1; // -1 = default
    std::mutex  device_mutex;

    static int stream_callback(const void*                     input,
                               void*                           /*output*/,
                               unsigned long                   frameCount,
                               const PaStreamCallbackTimeInfo* /*timeInfo*/,
                               PaStreamCallbackFlags           /*statusFlags*/,
                               void*                           userData)
    {
        auto* self = static_cast<LTCInputImpl*>(userData);
        if (!self || !self->decoder_ || !input) return paContinue;

        const float* src = static_cast<const float*>(input);
        ltc_decoder_write_float(self->decoder_, const_cast<float*>(src), static_cast<size_t>(frameCount), 0);
        
        LTCFrameExt ltc_frame;
        bool got_frame = false;
        SMPTETimecode temp_tc = {0};
        while(ltc_decoder_read(self->decoder_, &ltc_frame)) {
           ltc_frame_to_time(&temp_tc, &ltc_frame.ltc, 0);
           got_frame = true;
        }

        if (got_frame) {
            // Write to the inactive slot, then swap atomically
            int write_slot = 1 - self->active_slot_.load(std::memory_order_acquire);
            self->slots_[write_slot].tc = temp_tc;
            self->slots_[write_slot].signal_time = std::chrono::steady_clock::now();
            self->slots_[write_slot].has_data = true;
            self->active_slot_.store(write_slot, std::memory_order_release);
            self->valid_signal = true;
        }
        return paContinue;
    }
    
    // Find device index by name. Returns -1 if not found (use default).
    int find_device_by_name(const std::string& name) {
        if (name.empty()) return -1;
        int count = Pa_GetDeviceCount();
        for (int i = 0; i < count; ++i) {
            const PaDeviceInfo* info = Pa_GetDeviceInfo(i);
            if (info && info->maxInputChannels > 0 && name == info->name) {
                return i;
            }
        }
        return -1;
    }

    // Internal start logic — caller must hold device_mutex
    bool start_unlocked() {
        if (running) return true;
        
        if (!decoder_) decoder_ = ltc_decoder_create(48000, 25);

        int device_idx = current_device_index_;
        if (device_idx < 0) {
            device_idx = Pa_GetDefaultInputDevice();
        }
        if (device_idx == paNoDevice) {
            CASPAR_LOG(error) << "LTC: No input device available";
            return false;
        }

        const PaDeviceInfo* dev_info = Pa_GetDeviceInfo(device_idx);
        if (!dev_info || dev_info->maxInputChannels < 1) {
            CASPAR_LOG(error) << "LTC: Device " << device_idx << " has no input channels";
            return false;
        }

        PaStreamParameters input_params{};
        input_params.device                    = device_idx;
        input_params.channelCount              = 1;
        input_params.sampleFormat              = paFloat32;
        input_params.suggestedLatency          = dev_info->defaultLowInputLatency;
        input_params.hostApiSpecificStreamInfo = nullptr;

        PaError err = Pa_OpenStream(&stream_,
                                    &input_params,
                                    nullptr,        // no output
                                    48000,          // sample rate
                                    paFramesPerBufferUnspecified,
                                    paNoFlag,
                                    stream_callback,
                                    this);
        if (err != paNoError) {
            CASPAR_LOG(warning) << "LTC: Failed to open device " << device_idx 
                                << " (" << Pa_GetErrorText(err) << "), trying default.";
            if (device_idx != Pa_GetDefaultInputDevice()) {
                device_idx = Pa_GetDefaultInputDevice();
                if (device_idx == paNoDevice) {
                    CASPAR_LOG(error) << "LTC: No default input device";
                    return false;
                }
                dev_info = Pa_GetDeviceInfo(device_idx);
                input_params.device           = device_idx;
                input_params.suggestedLatency  = dev_info ? dev_info->defaultLowInputLatency : 0.0;
                err = Pa_OpenStream(&stream_, &input_params, nullptr, 48000,
                                    paFramesPerBufferUnspecified, paNoFlag, stream_callback, this);
                if (err != paNoError) {
                    CASPAR_LOG(error) << "LTC: Failed to open default device: " << Pa_GetErrorText(err);
                    return false;
                }
            } else {
                return false;
            }
        }

        err = Pa_StartStream(stream_);
        if (err != paNoError) {
            CASPAR_LOG(error) << "LTC: Failed to start stream: " << Pa_GetErrorText(err);
            Pa_CloseStream(stream_);
            stream_ = nullptr;
            return false;
        }

        running = true;
        CASPAR_LOG(info) << "LTC: Capturing from device " << device_idx 
                         << " (" << (dev_info ? dev_info->name : "unknown") << ")";
        return true;
    }

    void stop_unlocked() {
        if (stream_) {
            Pa_StopStream(stream_);
            Pa_CloseStream(stream_);
            stream_ = nullptr;
        }
        running = false;
    }

public:
    static LTCInputImpl& instance() {
        static LTCInputImpl instance;
        return instance;
    }
    
    LTCInputImpl() {
        PaError err = Pa_Initialize();
        if (err == paNoError) {
            pa_initialized_ = true;
        } else {
            CASPAR_LOG(error) << "LTC: Failed to init PortAudio: " << Pa_GetErrorText(err);
        }
    }
    
    void shutdown() {
        std::lock_guard<std::mutex> lock(device_mutex);
        stop_unlocked();
        if (decoder_) {
            ltc_decoder_free(decoder_);
            decoder_ = nullptr;
        }
        if (pa_initialized_) {
            Pa_Terminate();
            pa_initialized_ = false;
        }
    }

    ~LTCInputImpl() {
        // shutdown() should be called by ltc::uninit() before PortAudio terminates.
        // Safety fallback: free decoder if shutdown() was not called.
        if (decoder_) {
            ltc_decoder_free(decoder_);
            decoder_ = nullptr;
        }
    }

    void start() {
        std::lock_guard<std::mutex> lock(device_mutex);
        if (running) return;
        if (!pa_initialized_) return;
        
        // Read configuration
        try {
            auto& pt = caspar::env::properties();
            boost::optional<std::wstring> dev = pt.get_optional<std::wstring>(L"configuration.ltc.device");
            if (dev && current_device_name.empty()) {
                 current_device_name = caspar::u8(*dev);
                 current_device_index_ = find_device_by_name(current_device_name);
            }
        } catch (...) {}

        start_unlocked();
    }

    std::vector<std::string> get_capture_devices() {
        std::lock_guard<std::mutex> lock(device_mutex);
        std::vector<std::string> devices;
        if (!pa_initialized_) return devices;
        
        int count = Pa_GetDeviceCount();
        for (int i = 0; i < count; ++i) {
            const PaDeviceInfo* info = Pa_GetDeviceInfo(i);
            if (info && info->maxInputChannels > 0) {
                devices.push_back(info->name);
            }
        }
        return devices;
    }

    bool set_capture_device(const std::string& name) {
        std::lock_guard<std::mutex> lock(device_mutex);
        
        stop_unlocked();
        
        // Free old decoder and create fresh one
        if (decoder_) {
            ltc_decoder_free(decoder_);
            decoder_ = nullptr;
        }
        
        current_device_name = name;
        current_device_index_ = find_device_by_name(name);
        return start_unlocked();
    }

    std::string get_current_device_name() {
         return current_device_name.empty() ? "Default" : current_device_name;
    }
    
    std::string get_current_timecode_string() {
        int slot = active_slot_.load(std::memory_order_acquire);
        bool use_fallback = !valid_signal;
        
        if (valid_signal) {
             auto now = std::chrono::steady_clock::now();
             if (std::chrono::duration_cast<std::chrono::milliseconds>(now - slots_[slot].signal_time).count() > 1000) {
                 use_fallback = true;
             }
        }

        if (use_fallback) {
             time_t now = time(0);
             struct tm tstruct;
#ifdef _WIN32
             localtime_s(&tstruct, &now);
#else
             localtime_r(&now, &tstruct);
#endif
             char buffer[16];
             snprintf(buffer, sizeof(buffer), "%02d:%02d:%02d:00", tstruct.tm_hour, tstruct.tm_min, tstruct.tm_sec);
             return std::string(buffer);
        }

        char buffer[16];
        snprintf(buffer, sizeof(buffer), "%02d:%02d:%02d:%02d", 
            slots_[slot].tc.hours, slots_[slot].tc.mins, 
            slots_[slot].tc.secs, slots_[slot].tc.frame);
        return std::string(buffer);
    }
    
    uint32_t get_current_frame_number(int fps) {
         int slot = active_slot_.load(std::memory_order_acquire);
         bool use_fallback = !valid_signal;
        
         if (valid_signal) {
             auto now = std::chrono::steady_clock::now();
             if (std::chrono::duration_cast<std::chrono::milliseconds>(now - slots_[slot].signal_time).count() > 1000) {
                 use_fallback = true;
             }
         }

         if (use_fallback) {
             time_t now = time(0);
             struct tm tstruct;
#ifdef _WIN32
             localtime_s(&tstruct, &now);
#else
             localtime_r(&now, &tstruct);
#endif
             return (tstruct.tm_hour * 3600 + tstruct.tm_min * 60 + tstruct.tm_sec) * fps;
         }

         return (slots_[slot].tc.hours * 3600 + slots_[slot].tc.mins * 60 + slots_[slot].tc.secs) * fps + slots_[slot].tc.frame;
    }
    
    bool is_valid() {
        if (!valid_signal) return false;
        int slot = active_slot_.load(std::memory_order_acquire);
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - slots_[slot].signal_time).count() > 1000) {
             return false;
        }
        return true;
    }
    
    bool is_using_system_clock() {
        return !is_valid();
    }
};

LTCInput& LTCInput::instance() {
    static LTCInput wrapper;
    return wrapper;
}
void LTCInput::start() { LTCInputImpl::instance().start(); }
std::string LTCInput::get_current_timecode_string() { return LTCInputImpl::instance().get_current_timecode_string(); }
uint32_t LTCInput::get_current_frame_number(int fps) { return LTCInputImpl::instance().get_current_frame_number(fps); }
bool LTCInput::is_valid() { return LTCInputImpl::instance().is_valid(); }

// New methods bindings
std::vector<std::string> LTCInput::get_capture_devices() { return LTCInputImpl::instance().get_capture_devices(); }
bool LTCInput::set_capture_device(const std::string& name) { return LTCInputImpl::instance().set_capture_device(name); }
std::string LTCInput::get_current_device_name() { return LTCInputImpl::instance().get_current_device_name(); }
bool LTCInput::is_using_system_clock() { return LTCInputImpl::instance().is_using_system_clock(); }
void LTCInput::shutdown() { LTCInputImpl::instance().shutdown(); }

void init(const core::module_dependencies&) {
    LTCInput::instance().start();
}

void uninit() {
    LTCInput::instance().shutdown();
}

}}
