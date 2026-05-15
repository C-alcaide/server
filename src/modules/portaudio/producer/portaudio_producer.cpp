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

#include "portaudio_producer.h"

#include "../util/portaudio_device.h"

#include <common/array.h>
#include <common/log.h>
#include <common/utf.h>

#include <core/frame/draw_frame.h>
#include <core/frame/frame.h>
#include <core/frame/frame_factory.h>
#include <core/frame/pixel_format.h>
#include <core/producer/frame_producer.h>
#include <core/video_format.h>

#include <portaudio.h>

#include "../util/spsc_ring_buffer.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <mutex>
#include <vector>

namespace caspar { namespace portaudio {

static const std::wstring PRODUCER_NAME = L"portaudio";

// Maximum audio buffer: 5 seconds worth of interleaved samples.
static constexpr size_t MAX_BUFFER_SECONDS = 5;

class portaudio_producer_impl
{
    PaStream*    stream_          = nullptr;
    int          device_index_    = -1;

    std::string  device_name_;
    host_api_preference host_api_pref_ = host_api_preference::auto_select;
    int          input_channels_  = 2;

    std::unique_ptr<spsc_ring_buffer> capture_ring_;
    std::atomic<int64_t>               overflow_count_{0};

    core::video_format_desc                format_desc_;
    spl::shared_ptr<core::frame_factory>   frame_factory_;

    std::atomic<bool>    running_{false};
    std::atomic<int64_t> underrun_count_{0};
    std::atomic<int64_t> buffer_fill_{0};

    core::monitor::state state_;
    mutable std::mutex   state_mutex_;

    // PortAudio capture callback
    static int stream_callback(const void*                     input,
                               void*                           /*output*/,
                               unsigned long                   frame_count,
                               const PaStreamCallbackTimeInfo* /*time_info*/,
                               PaStreamCallbackFlags           /*status_flags*/,
                               void*                           user_data)
    {
        auto* self = static_cast<portaudio_producer_impl*>(user_data);
        if (!self || !input)
            return paContinue;

        const auto* input_samples = static_cast<const int32_t*>(input);
        size_t sample_count = frame_count * self->input_channels_;

        // Lock-free write into SPSC ring buffer
        size_t written = self->capture_ring_->write(input_samples, sample_count);
        if (written < sample_count)
            self->overflow_count_.fetch_add(1, std::memory_order_relaxed);

        return self->running_.load(std::memory_order_relaxed) ? paContinue : paComplete;
    }

  public:
    portaudio_producer_impl(const core::frame_producer_dependencies& dependencies,
                            std::string                              device_name,
                            host_api_preference                      host_api,
                            int                                      channels)
        : device_name_(std::move(device_name))
        , host_api_pref_(host_api)
        , input_channels_(channels)
        , format_desc_(dependencies.format_desc)
        , frame_factory_(dependencies.frame_factory)
    {
        size_t ring_capacity = static_cast<size_t>(format_desc_.audio_sample_rate) *
                              input_channels_ * MAX_BUFFER_SECONDS;
        capture_ring_ = std::make_unique<spsc_ring_buffer>(ring_capacity);

        auto& mgr = portaudio_device_manager::instance();

        // Find device
        if (!device_name_.empty()) {
            device_index_ = mgr.find_input_device(device_name_, host_api_pref_);
            if (device_index_ < 0) {
                CASPAR_LOG(warning) << L"[portaudio-producer] Device not found: "
                                   << u16(device_name_) << L". Using default.";
            }
        }

        if (device_index_ < 0) {
            device_index_ = mgr.get_default_input_device(host_api_pref_);
        }

        if (device_index_ < 0) {
            CASPAR_LOG(error) << L"[portaudio-producer] No input device available.";
            return;
        }

        const PaDeviceInfo* dev_info = Pa_GetDeviceInfo(device_index_);
        if (!dev_info) {
            CASPAR_LOG(error) << L"[portaudio-producer] Failed to get device info.";
            return;
        }

        // Clamp channels to device capability
        if (input_channels_ > dev_info->maxInputChannels) {
            CASPAR_LOG(warning) << L"[portaudio-producer] Requested " << input_channels_
                               << L" channels, device supports " << dev_info->maxInputChannels
                               << L". Clamping.";
            input_channels_ = dev_info->maxInputChannels;
        }

        // Open capture stream
        PaStreamParameters input_params = {};
        input_params.device                    = device_index_;
        input_params.channelCount              = input_channels_;
        input_params.sampleFormat              = paInt32;
        input_params.suggestedLatency          = dev_info->defaultLowInputLatency;
        input_params.hostApiSpecificStreamInfo = nullptr;

        int frames_per_buffer = format_desc_.audio_sample_rate / static_cast<int>(format_desc_.fps);

        PaError err = Pa_OpenStream(&stream_,
                                    &input_params,
                                    nullptr,   // no output
                                    format_desc_.audio_sample_rate,
                                    frames_per_buffer,
                                    paClipOff,
                                    stream_callback,
                                    this);

        if (err != paNoError) {
            CASPAR_LOG(error) << L"[portaudio-producer] Failed to open stream: " << Pa_GetErrorText(err);
            return;
        }

        err = Pa_StartStream(stream_);
        if (err != paNoError) {
            CASPAR_LOG(error) << L"[portaudio-producer] Failed to start stream: " << Pa_GetErrorText(err);
            Pa_CloseStream(stream_);
            stream_ = nullptr;
            return;
        }

        running_ = true;

        const PaHostApiInfo* api_info = Pa_GetHostApiInfo(dev_info->hostApi);
        CASPAR_LOG(info) << L"[portaudio-producer] Opened capture device: " << dev_info->name
                        << L" [" << (api_info ? api_info->name : "?") << L"]"
                        << L" channels=" << input_channels_
                        << L" rate=" << format_desc_.audio_sample_rate;

        std::lock_guard<std::mutex> slock(state_mutex_);
        state_["device"] = u16(std::string(dev_info->name));
        state_["running"] = std::to_wstring(1);
    }

    ~portaudio_producer_impl()
    {
        running_ = false;
        if (stream_) {
            Pa_StopStream(stream_);
            Pa_CloseStream(stream_);
            stream_ = nullptr;
        }
    }

    core::draw_frame get_frame(int nb_samples)
    {
        size_t samples_per_channel;
        if (nb_samples > 0) {
            samples_per_channel = static_cast<size_t>(nb_samples);
        } else {
            samples_per_channel = static_cast<size_t>(format_desc_.audio_sample_rate) *
                                  format_desc_.duration / format_desc_.time_scale;
        }
        size_t total_samples_needed = samples_per_channel * input_channels_;

        std::vector<int32_t> output_samples(total_samples_needed, 0);

        // Lock-free read from SPSC ring buffer
        size_t current_fill = capture_ring_->read_available();
        size_t samples_read = capture_ring_->read(output_samples.data(), total_samples_needed);
        bool underrun = (samples_read < total_samples_needed);

        if (underrun)
            underrun_count_++;
        buffer_fill_ = static_cast<int64_t>(current_fill);

        {
            std::lock_guard<std::mutex> slock(state_mutex_);
            state_["buffer/fill"]     = std::to_wstring(current_fill);
            state_["buffer/underruns"] = std::to_wstring(underrun_count_.load());
        }

        // Create audio-only frame with minimal 1x1 pixel plane (audio producers don't need video data)
        core::pixel_format_desc pix_desc(core::pixel_format::bgra);
        pix_desc.planes.push_back(core::pixel_format_desc::plane(1, 1, 4));
        auto frame = frame_factory_->create_frame(this, pix_desc);

        // Zero the single pixel for transparency
        if (frame.image_data(0).size() >= 4)
            std::memset(frame.image_data(0).data(), 0, 4);

        if (!output_samples.empty()) {
            frame.audio_data() = caspar::array<int32_t>(std::move(output_samples));
        }

        return core::draw_frame(std::move(frame));
    }

    core::monitor::state get_state() const
    {
        std::lock_guard<std::mutex> slock(state_mutex_);
        return state_;
    }
};

class portaudio_producer : public core::frame_producer
{
  public:
    portaudio_producer(const core::frame_producer_dependencies& dependencies,
                       std::string                              device_name,
                       host_api_preference                      host_api,
                       int                                      channels)
        : impl_(std::make_unique<portaudio_producer_impl>(dependencies, std::move(device_name), host_api, channels))
    {
    }

    ~portaudio_producer() override = default;

    core::draw_frame receive_impl(const core::video_field /*field*/, int nb_samples) override
    {
        return impl_->get_frame(nb_samples);
    }

    std::wstring         print() const override { return L"portaudio[" + name() + L"]"; }
    std::wstring         name() const override { return PRODUCER_NAME; }
    core::monitor::state state() const override { return impl_->get_state(); }
    bool                 is_ready() override { return true; }

  private:
    std::unique_ptr<portaudio_producer_impl> impl_;
};

// --- Factory ---

spl::shared_ptr<core::frame_producer> create_producer(const core::frame_producer_dependencies& dependencies,
                                                      const std::vector<std::wstring>&         params)
{
    if (params.empty() || params[0] != L"portaudio")
        return core::frame_producer::empty();

    std::string device_name;
    host_api_preference host_api = host_api_preference::auto_select;
    int channels = 2;

    for (size_t i = 1; i < params.size(); ++i) {
        if (params[i].rfind(L"DEVICE=", 0) == 0) {
            device_name = u8(params[i].substr(7));
        } else if (params[i].rfind(L"API=", 0) == 0) {
            host_api = portaudio_device_manager::parse_host_api(u8(params[i].substr(4)));
        } else if (params[i].rfind(L"CHANNELS=", 0) == 0) {
            try { channels = std::stoi(u8(params[i].substr(9))); }
            catch (...) {}
        } else if (i == 1 && params[i] != L"EMPTY") {
            device_name = u8(params[i]);
        }
    }

    return spl::make_shared<portaudio_producer>(dependencies, device_name, host_api, channels);
}

}} // namespace caspar::portaudio
