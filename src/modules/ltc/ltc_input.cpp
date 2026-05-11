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
#include <common/utf.h>
#include <boost/property_tree/ptree.hpp>

#pragma warning(push)
#pragma warning(disable : 4244)
#include "miniaudio.h"
#pragma warning(pop)

namespace caspar { namespace ltc {
    
static const wchar_t* LTC_CONFIG_ROOT = L"configuration.ltc";

class LTCInputImpl {
    ma_context context;
    ma_device device;
    ma_device_config deviceConfig;
    LTCDecoder* decoder = nullptr;
    
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
    
    bool context_initialized_ = false;
    bool device_initialized_  = false;

    // Device Management
    std::string current_device_name;
    std::mutex device_mutex;

    static void data_callback(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount) {
        LTCInputImpl* self = (LTCInputImpl*)pDevice->pUserData;
        if (!self || !self->decoder || !pInput) return;

        // Copy input to local buffer to avoid const_cast
        const float* src = static_cast<const float*>(pInput);
        std::vector<float> buf(src, src + frameCount);
        ltc_decoder_write_float(self->decoder, buf.data(), frameCount, 0);
        
        LTCFrameExt ltc_frame;
        bool got_frame = false;
        SMPTETimecode temp_tc = {0};
        while(ltc_decoder_read(self->decoder, &ltc_frame)) {
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
    }
    
    // Internal start logic — caller must hold device_mutex
    bool start_unlocked() {
        if (running) return true;
        
        if (!decoder) decoder = ltc_decoder_create(48000, 25);
        
        deviceConfig = ma_device_config_init(ma_device_type_capture);
        deviceConfig.capture.format = ma_format_f32;
        deviceConfig.capture.channels = 1;
        deviceConfig.sampleRate = 48000;
        deviceConfig.dataCallback = data_callback;
        deviceConfig.pUserData = this;

        // Find device ID if name is set
        ma_device_id* pDeviceID = NULL;
        if (!current_device_name.empty()) {
             ma_device_info* pCaptureInfos = nullptr;
             ma_uint32 captureCount = 0;
             ma_device_info* pPlaybackInfos = nullptr;
             ma_uint32 playbackCount = 0;
             
             if (ma_context_get_devices(&context, &pPlaybackInfos, &playbackCount, &pCaptureInfos, &captureCount) == MA_SUCCESS) {
                 for (ma_uint32 i = 0; i < captureCount; ++i) {
                     if (current_device_name == pCaptureInfos[i].name) {
                         pDeviceID = &pCaptureInfos[i].id;
                         break;
                     }
                 }
             }
        }
        
        // Assign found device ID to config
        if (pDeviceID != NULL) {
            deviceConfig.capture.pDeviceID = pDeviceID;
        }

        if (ma_device_init(&context, &deviceConfig, &device) != MA_SUCCESS) {
            if (pDeviceID != NULL) {
                 // Fallback to default
                 CASPAR_LOG(warning) << "LTC: Failed to init specific device, trying default.";
                 deviceConfig.capture.pDeviceID = NULL;
                 if (ma_device_init(&context, &deviceConfig, &device) != MA_SUCCESS) {
                     CASPAR_LOG(error) << "LTC: Failed to init miniaudio device";
                     return false;
                 }
            } else {
                CASPAR_LOG(error) << "LTC: Failed to init miniaudio device";
                return false;
            }
        }
        device_initialized_ = true;

        if (ma_device_start(&device) == MA_SUCCESS) {
            running = true;
            return true;
        }
        // Start failed — clean up the initialized device
        ma_device_uninit(&device);
        device_initialized_ = false;
        return false;
    }

public:
    static LTCInputImpl& instance() {
        static LTCInputImpl instance;
        return instance;
    }
    
    LTCInputImpl() {
        if (ma_context_init(NULL, 0, NULL, &context) == MA_SUCCESS) {
            context_initialized_ = true;
        } else {
            CASPAR_LOG(error) << "LTC: Failed to init miniaudio context";
        }
    }
    
    ~LTCInputImpl() {
        running = false;
        if (device_initialized_) {
             ma_device_uninit(&device);
        }
        if (context_initialized_) {
            ma_context_uninit(&context);
        }
        if (decoder) {
            ltc_decoder_free(decoder);
            decoder = nullptr;
        }
    }

    void start() {
        std::lock_guard<std::mutex> lock(device_mutex);
        if (running) return;
        
        // Read configuration
        try {
            auto& pt = caspar::env::properties();
            boost::optional<std::wstring> dev = pt.get_optional<std::wstring>(L"configuration.ltc.device");
            if (dev && current_device_name.empty()) {
                 current_device_name = caspar::u8(*dev);
            }
        } catch (...) {}

        start_unlocked();
    }

    std::vector<std::string> get_capture_devices() {
        std::lock_guard<std::mutex> lock(device_mutex);
        ma_device_info* pCaptureInfos = nullptr;
        ma_uint32 captureCount = 0;
        ma_device_info* pPlaybackInfos = nullptr;
        ma_uint32 playbackCount = 0;
        
        // Always re-enumerate to pick up newly connected devices
        std::vector<std::string> devices;
        if (ma_context_get_devices(&context, &pPlaybackInfos, &playbackCount, &pCaptureInfos, &captureCount) == MA_SUCCESS) {
            for (ma_uint32 i = 0; i < captureCount; ++i) {
                devices.push_back(pCaptureInfos[i].name);
            }
        }
        return devices;
    }

    bool set_capture_device(const std::string& name) {
        std::lock_guard<std::mutex> lock(device_mutex);
        
        if (running) {
             ma_device_stop(&device);
             ma_device_uninit(&device);
             device_initialized_ = false;
             running = false;
        }
        
        // Free old decoder and create fresh one
        if (decoder) {
            ltc_decoder_free(decoder);
            decoder = nullptr;
        }
        
        current_device_name = name;
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

void init(const core::module_dependencies&) {
    LTCInput::instance().start();
}

}} 
