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
 * This module uses miniaudio (https://miniaud.io), dual-licensed under
 * MIT and public domain (Unlicense), both compatible with GPL-3.
 */

#include <core/frame/frame_factory.h>
#include <core/producer/frame_producer.h>
#include <core/video_format.h>
#include <core/frame/pixel_format.h>
#include <core/frame/draw_frame.h>
#include <core/frame/frame.h>
#include <common/executor.h>
#include <common/memory.h>
#include <common/array.h>
#include <common/log.h>
#include <common/utf.h>

#include "system_audio_producer.h"

#pragma warning(push)
#pragma warning(disable : 4244)
#include "miniaudio.h"
#pragma warning(pop)

#include <iostream>
#include <mutex>
#include <atomic>
#include <vector>
#include <queue>
#include <cstring>
#include <algorithm>
#include <string>

namespace caspar { namespace system_audio {

static const std::wstring PRODUCER_NAME = L"system_audio";

// Maximum audio buffer: 5 seconds worth of interleaved samples.
// Prevents unbounded growth if receive_impl is not called (e.g. channel paused).
static constexpr size_t MAX_BUFFER_SECONDS = 5;

spl::shared_ptr<core::frame_producer> create_producer(const core::frame_producer_dependencies& dependencies,
                                                      const std::vector<std::wstring>&         params)
{
    if (params.empty() || params[0] != L"system_audio") {
        return core::frame_producer::empty();
    }

    std::wstring device_name = L"";
    if (params.size() > 1 && params[1] != L"EMPTY") {
        device_name = params[1];
    }
    
    // Check parameters for DEVICE=... 
    for(size_t i=1; i<params.size(); ++i) {
        if(params[i].rfind(L"DEVICE=", 0) == 0) {
            device_name = params[i].substr(7);
        }
    }
    
    return spl::make_shared<system_audio_producer>(dependencies, device_name, params);
}

class system_audio_producer::impl
{
    ma_context context;
    ma_device device;
    ma_device_config deviceConfig;
    
    std::string device_name_u8;
    std::wstring device_display_name_;
    
    std::mutex audio_mutex;
    std::vector<int32_t> audio_buffer; // Interleaved S32 samples
    size_t max_buffer_samples_ = 0;   // Computed cap for buffer size
    
    core::video_format_desc format_desc;
    spl::shared_ptr<core::frame_factory> frame_factory_;
    
    std::atomic<bool> running{false};
    bool context_initialized_ = false;
    bool device_initialized_  = false;
    
    // Stats for OSC state
    std::atomic<int64_t> underrun_count_{0};
    std::atomic<int64_t> buffer_fill_{0};
    
    core::monitor::state state_;
    mutable std::mutex   state_mutex_;

    static void data_callback(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount) {
        impl* self = (impl*)pDevice->pUserData;
        if (!self || !pInput) return;

        std::lock_guard<std::mutex> lock(self->audio_mutex);
        
        const int32_t* input_samples = (const int32_t*)pInput;
        size_t sample_count = frameCount * self->format_desc.audio_channels;
        
        self->audio_buffer.insert(self->audio_buffer.end(), input_samples, input_samples + sample_count);
        
        // Cap buffer to prevent unbounded growth
        if (self->max_buffer_samples_ > 0 && self->audio_buffer.size() > self->max_buffer_samples_) {
            size_t excess = self->audio_buffer.size() - self->max_buffer_samples_;
            self->audio_buffer.erase(self->audio_buffer.begin(), self->audio_buffer.begin() + excess);
        }
    }
    
public:
    impl(const core::frame_producer_dependencies& dependencies, const std::wstring& device_name)
        : frame_factory_(dependencies.frame_factory)
        , format_desc(dependencies.format_desc)
    {
        max_buffer_samples_ = (size_t)format_desc.audio_sample_rate * format_desc.audio_channels * MAX_BUFFER_SECONDS;
        
        if (!device_name.empty()) {
            device_name_u8 = caspar::u8(device_name);
            device_display_name_ = device_name;
        }

        if (ma_context_init(NULL, 0, NULL, &context) != MA_SUCCESS) {
            CASPAR_LOG(error) << "system_audio: Failed to init miniaudio context";
            return;
        }
        context_initialized_ = true;

        deviceConfig = ma_device_config_init(ma_device_type_capture);
        deviceConfig.capture.format   = ma_format_s32;
        deviceConfig.capture.channels = format_desc.audio_channels;
        deviceConfig.sampleRate       = format_desc.audio_sample_rate;
        deviceConfig.dataCallback     = data_callback;
        deviceConfig.pUserData        = this;
        
        // Select device if specified
        if (!device_name_u8.empty()) {
             ma_device_info* pPlaybackInfos;
             ma_uint32 playbackCount;
             ma_device_info* pCaptureInfos;
             ma_uint32 captureCount;
             
             if (ma_context_get_devices(&context, &pPlaybackInfos, &playbackCount, &pCaptureInfos, &captureCount) == MA_SUCCESS) {
                 bool found = false;
                 for (ma_uint32 i = 0; i < captureCount; ++i) {
                     std::string name(pCaptureInfos[i].name);
                     if (name == device_name_u8 || name.find(device_name_u8) != std::string::npos) {
                         deviceConfig.capture.pDeviceID = &pCaptureInfos[i].id;
                         device_display_name_ = u16(name);
                         CASPAR_LOG(info) << "system_audio: Selected device: " << name;
                         found = true;
                         break;
                     }
                 }
                 if (!found) {
                     CASPAR_LOG(warning) << "system_audio: Device not found: " << device_name_u8 << ". Using default.";
                     device_display_name_ = L"Default";
                 }
             }
        } else {
             CASPAR_LOG(info) << "system_audio: Using default capture device";
             device_display_name_ = L"Default";
        }

        if (ma_device_init(&context, &deviceConfig, &device) != MA_SUCCESS) {
            CASPAR_LOG(error) << "system_audio: Failed to init device";
            return;
        }
        device_initialized_ = true;

        if (ma_device_start(&device) != MA_SUCCESS) {
             CASPAR_LOG(error) << "system_audio: Failed to start device";
             ma_device_uninit(&device);
             device_initialized_ = false;
             return;
        }
        
        running = true;
        
        // Initial state
        std::lock_guard<std::mutex> slock(state_mutex_);
        state_["device"] = device_display_name_;
        state_["running"] = std::to_wstring(running ? 1 : 0);
    }
    
    ~impl() {
        running = false;
        if (device_initialized_) {
            ma_device_uninit(&device);
        }
        if (context_initialized_) {
            ma_context_uninit(&context);
        }
    }
    
    core::draw_frame get_frame(int nb_samples) {
        size_t samples_per_channel;
        if (nb_samples > 0) {
            samples_per_channel = nb_samples;
        } else {
            samples_per_channel = static_cast<size_t>(format_desc.audio_sample_rate * format_desc.duration / format_desc.time_scale); 
        }
        size_t total_samples_needed = samples_per_channel * format_desc.audio_channels;
        
        std::vector<int32_t> output_samples;
        output_samples.resize(total_samples_needed, 0); // Silence by default
        
        bool underrun = false;
        size_t current_fill = 0;
        {
            std::lock_guard<std::mutex> lock(audio_mutex);
            current_fill = audio_buffer.size();
            if (audio_buffer.size() >= total_samples_needed) {
                std::copy(audio_buffer.begin(), audio_buffer.begin() + total_samples_needed, output_samples.begin());
                audio_buffer.erase(audio_buffer.begin(), audio_buffer.begin() + total_samples_needed);
            } else {
                underrun = true;
                if (!audio_buffer.empty()) {
                    size_t to_copy = std::min(audio_buffer.size(), total_samples_needed);
                    std::copy(audio_buffer.begin(), audio_buffer.begin() + to_copy, output_samples.begin());
                    audio_buffer.clear();
                }
            }
        }
        
        if (underrun) {
            underrun_count_++;
        }
        buffer_fill_ = (int64_t)current_fill;
        
        // Update OSC state
        {
            std::lock_guard<std::mutex> slock(state_mutex_);
            state_["buffer/fill"] = std::to_wstring(current_fill);
            state_["buffer/underruns"] = std::to_wstring(underrun_count_.load());
        }
        
        // Reuse cached black frame, just attach new audio
        core::pixel_format_desc pix_desc(core::pixel_format::bgra);
        pix_desc.planes.push_back(core::pixel_format_desc::plane(format_desc.width, format_desc.height, 4));
        auto frame = frame_factory_->create_frame(this, pix_desc);
        
        if (!output_samples.empty()) {
            frame.audio_data() = caspar::array<int32_t>(std::move(output_samples));
        }
        
        return core::draw_frame(std::move(frame));
    }
    
    core::monitor::state get_state() const {
        std::lock_guard<std::mutex> slock(state_mutex_);
        return state_;
    }
};

system_audio_producer::system_audio_producer(const core::frame_producer_dependencies& dependencies,
                                             const std::wstring&                      device_name,
                                             const std::vector<std::wstring>&         params)
    : impl_(new impl(dependencies, device_name))
{
}

system_audio_producer::~system_audio_producer() = default;

core::draw_frame system_audio_producer::receive_impl(const core::video_field /*field*/, int nb_samples)
{
    return impl_->get_frame(nb_samples);
}

std::wstring system_audio_producer::print() const { return L"system_audio[" + name() + L"]"; }
std::wstring system_audio_producer::name() const { return PRODUCER_NAME; }
core::monitor::state system_audio_producer::state() const { return impl_->get_state(); }

}} // namespace caspar::system_audio