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
#include "../util/shared_capture.h"

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
#include <sstream>
#include <vector>

namespace caspar { namespace portaudio {

static const std::wstring PRODUCER_NAME = L"portaudio";

// Maximum audio buffer: 5 seconds worth of interleaved samples.
static constexpr size_t MAX_BUFFER_SECONDS = 5;

/// Parse a comma-separated list of non-negative integers (e.g. "0,3,7").
static std::vector<int> parse_int_list(const std::string& str)
{
    std::vector<int> result;
    std::istringstream ss(str);
    std::string token;
    while (std::getline(ss, token, ',')) {
        try {
            int v = std::stoi(token);
            if (v >= 0)
                result.push_back(v);
        } catch (...) {}
    }
    return result;
}

class portaudio_producer_impl : public capture_listener
{
    PaStream*    stream_          = nullptr;
    int          device_index_    = -1;

    std::string  device_name_;
    host_api_preference host_api_pref_ = host_api_preference::auto_select;
    int          device_channels_ = 2;  // actual channels opened on device
    int          output_channels_ = 2;  // channels delivered to CasparCG mixer
    bool         use_shared_      = false;

    // Shared capture (Phase 2): when multiple producers target same device
    std::shared_ptr<shared_portaudio_capture> shared_capture_;

    // Channel selection: which device channels to extract.
    // If empty, uses contiguous range [from_channel_ .. from_channel_+output_channels_).
    int              from_channel_ = 0;
    std::vector<int> channel_map_;       // explicit non-contiguous map (MAP= param)

    // Delay compensation
    int              delay_ms_     = 0;  // DELAY= in milliseconds
    size_t           delay_samples_ = 0; // computed: delay in interleaved samples (output_channels_ wide)
    std::unique_ptr<spsc_ring_buffer> delay_ring_;  // optional delay buffer

    std::unique_ptr<spsc_ring_buffer> capture_ring_;
    std::atomic<int64_t>               overflow_count_{0};

    core::video_format_desc                format_desc_;
    spl::shared_ptr<core::frame_factory>   frame_factory_;

    std::atomic<bool>    running_{false};
    std::atomic<bool>    disconnected_{false};
    bool                 disconnect_logged_{false};
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
        size_t sample_count = frame_count * self->device_channels_;

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
                            int                                      channels,
                            int                                      from_channel,
                            int                                      count,
                            std::vector<int>                         channel_map,
                            int                                      delay_ms,
                            bool                                     use_shared)
        : device_name_(std::move(device_name))
        , host_api_pref_(host_api)
        , device_channels_(channels)
        , use_shared_(use_shared)
        , from_channel_(from_channel)
        , channel_map_(std::move(channel_map))
        , delay_ms_(delay_ms)
        , format_desc_(dependencies.format_desc)
        , frame_factory_(dependencies.frame_factory)
    {
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

        // Auto-detect: force shared mode for ASIO (exclusive access)
        const PaHostApiInfo* api_info = Pa_GetHostApiInfo(dev_info->hostApi);
        if (api_info && api_info->type == paASIO)
            use_shared_ = true;

        // Determine device_channels_: open the full device width (or CHANNELS= if specified)
        if (device_channels_ <= 0)
            device_channels_ = dev_info->maxInputChannels;
        if (device_channels_ > dev_info->maxInputChannels) {
            CASPAR_LOG(warning) << L"[portaudio-producer] Requested " << device_channels_
                               << L" channels, device supports " << dev_info->maxInputChannels
                               << L". Clamping.";
            device_channels_ = dev_info->maxInputChannels;
        }

        // Resolve output channel count and validate channel selection
        if (!channel_map_.empty()) {
            // MAP= mode: explicit non-contiguous channel picks
            for (auto& ch : channel_map_) {
                if (ch >= device_channels_) {
                    CASPAR_LOG(warning) << L"[portaudio-producer] MAP channel " << ch
                                       << L" exceeds device channels (" << device_channels_ << L"). Clamping.";
                    ch = device_channels_ - 1;
                }
            }
            output_channels_ = static_cast<int>(channel_map_.size());
        } else {
            // FROM=/COUNT= mode: contiguous range
            if (from_channel_ >= device_channels_)
                from_channel_ = 0;
            int available = device_channels_ - from_channel_;
            if (count > 0 && count <= available)
                output_channels_ = count;
            else
                output_channels_ = available;
        }

        // Allocate capture ring buffer (stores full device-width interleaved samples)
        size_t ring_capacity = static_cast<size_t>(format_desc_.audio_sample_rate) *
                              device_channels_ * MAX_BUFFER_SECONDS;
        capture_ring_ = std::make_unique<spsc_ring_buffer>(ring_capacity);

        // Delay compensation
        if (delay_ms_ > 0) {
            delay_samples_ = static_cast<size_t>(format_desc_.audio_sample_rate) *
                             output_channels_ * delay_ms_ / 1000;
            // Delay ring sized for delay amount + 1 second headroom
            size_t delay_ring_capacity = delay_samples_ +
                static_cast<size_t>(format_desc_.audio_sample_rate) * output_channels_;
            delay_ring_ = std::make_unique<spsc_ring_buffer>(delay_ring_capacity);

            // Pre-fill with silence to introduce the delay
            std::vector<int32_t> silence(delay_samples_, 0);
            delay_ring_->write(silence.data(), delay_samples_);
        }

        if (use_shared_) {
            // Shared capture path: get or create a shared stream for this device
            shared_capture_ = mgr.get_shared_capture(
                device_index_, device_channels_, format_desc_.audio_sample_rate);
            if (!shared_capture_) {
                CASPAR_LOG(error) << L"[portaudio-producer] Failed to get shared capture.";
                return;
            }
            shared_capture_->add_listener(this);
            running_ = true;

            CASPAR_LOG(info) << L"[portaudio-producer] Using shared capture for device: " << dev_info->name
                            << L" [" << (api_info ? api_info->name : "?") << L"]"
                            << L" device_ch=" << device_channels_
                            << L" output_ch=" << output_channels_
                            << L" from=" << from_channel_
                            << L" map=" << (channel_map_.empty() ? "none" : "custom")
                            << L" delay=" << delay_ms_ << L"ms"
                            << L" listeners=" << shared_capture_->listener_count();
        } else {
            // Direct stream path: open our own PaStream
            PaStreamParameters input_params = {};
            input_params.device                    = device_index_;
            input_params.channelCount              = device_channels_;
            input_params.sampleFormat              = paInt32;
            input_params.suggestedLatency          = dev_info->defaultLowInputLatency;
            input_params.hostApiSpecificStreamInfo = nullptr;

            int frames_per_buffer = format_desc_.audio_sample_rate / static_cast<int>(format_desc_.fps);

            PaError err = Pa_OpenStream(&stream_,
                                        &input_params,
                                        nullptr,
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

            CASPAR_LOG(info) << L"[portaudio-producer] Opened direct capture: " << dev_info->name
                            << L" [" << (api_info ? api_info->name : "?") << L"]"
                            << L" device_ch=" << device_channels_
                            << L" output_ch=" << output_channels_
                            << L" from=" << from_channel_
                            << L" map=" << (channel_map_.empty() ? "none" : "custom")
                            << L" delay=" << delay_ms_ << L"ms"
                            << L" rate=" << format_desc_.audio_sample_rate;
        }

        std::lock_guard<std::mutex> slock(state_mutex_);
        state_["device"]  = u16(std::string(dev_info->name));
        state_["running"] = std::to_wstring(1);
        state_["shared"]  = std::to_wstring(use_shared_ ? 1 : 0);
    }

    ~portaudio_producer_impl() override
    {
        running_ = false;
        if (shared_capture_) {
            shared_capture_->remove_listener(this);
            shared_capture_.reset();
        }
        if (stream_) {
            Pa_StopStream(stream_);
            Pa_CloseStream(stream_);
            stream_ = nullptr;
        }
    }

    // --- capture_listener interface (shared capture mode) ---
    void on_captured_audio(const int32_t* interleaved, size_t frame_count, int channels) override
    {
        if (!running_)
            return;
        size_t sample_count = frame_count * channels;
        size_t written = capture_ring_->write(interleaved, sample_count);
        if (written < sample_count)
            overflow_count_.fetch_add(1, std::memory_order_relaxed);
    }

    void on_capture_disconnected() override
    {
        disconnected_ = true;
        CASPAR_LOG(warning) << L"[portaudio-producer] Shared capture disconnected for device "
                           << device_index_;
    }

    core::draw_frame get_frame(int nb_samples)
    {
        if (disconnected_) {
            if (!disconnect_logged_) {
                disconnect_logged_ = true;
                CASPAR_LOG(warning) << L"[portaudio-producer] Device " << device_index_
                                   << L" disconnected — returning silence.";
            }
            return core::draw_frame::empty();
        }

        size_t samples_per_channel;
        if (nb_samples > 0) {
            samples_per_channel = static_cast<size_t>(nb_samples);
        } else {
            samples_per_channel = static_cast<size_t>(format_desc_.audio_sample_rate) *
                                  format_desc_.duration / format_desc_.time_scale;
        }

        // Read from capture ring (full device-width interleaved)
        size_t total_device_samples = samples_per_channel * device_channels_;
        std::vector<int32_t> device_buffer(total_device_samples, 0);

        size_t current_fill = capture_ring_->read_available();
        size_t samples_read = capture_ring_->read(device_buffer.data(), total_device_samples);
        bool underrun = (samples_read < total_device_samples);

        if (underrun)
            underrun_count_++;
        buffer_fill_ = static_cast<int64_t>(current_fill);

        // Extract selected channels from the device-width buffer
        size_t total_output_samples = samples_per_channel * output_channels_;
        std::vector<int32_t> output_samples(total_output_samples, 0);

        bool has_map = !channel_map_.empty();
        for (size_t s = 0; s < samples_per_channel; ++s) {
            for (int c = 0; c < output_channels_; ++c) {
                int src_ch = has_map ? channel_map_[c] : (from_channel_ + c);
                if (src_ch >= 0 && src_ch < device_channels_) {
                    output_samples[s * output_channels_ + c] =
                        device_buffer[s * device_channels_ + src_ch];
                }
            }
        }

        // Apply delay compensation if configured
        if (delay_ring_) {
            // Push extracted samples into delay ring, pull delayed samples out
            delay_ring_->write(output_samples.data(), total_output_samples);
            size_t delayed_read = delay_ring_->read(output_samples.data(), total_output_samples);
            if (delayed_read < total_output_samples) {
                std::memset(output_samples.data() + delayed_read, 0,
                            (total_output_samples - delayed_read) * sizeof(int32_t));
            }
        }

        {
            std::lock_guard<std::mutex> slock(state_mutex_);
            state_["buffer/fill"]     = std::to_wstring(current_fill);
            state_["buffer/underruns"] = std::to_wstring(underrun_count_.load());
        }

        // Create audio-only frame with minimal 1x1 pixel plane
        core::pixel_format_desc pix_desc(core::pixel_format::bgra);
        pix_desc.planes.push_back(core::pixel_format_desc::plane(1, 1, 4));
        auto frame = frame_factory_->create_frame(this, pix_desc);

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
                       int                                      channels,
                       int                                      from_channel,
                       int                                      count,
                       std::vector<int>                         channel_map,
                       int                                      delay_ms,
                       bool                                     use_shared)
        : impl_(std::make_unique<portaudio_producer_impl>(
              dependencies, std::move(device_name), host_api, channels,
              from_channel, count, std::move(channel_map), delay_ms, use_shared))
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
    int device_channels = 0;  // 0 = use full device width (CHANNELS= param)
    int from_channel    = 0;
    int count           = 0;  // 0 = derive from device_channels - from_channel
    int delay_ms        = 0;
    bool use_shared     = false;
    std::vector<int> channel_map;

    for (size_t i = 1; i < params.size(); ++i) {
        if (params[i].rfind(L"DEVICE=", 0) == 0) {
            device_name = u8(params[i].substr(7));
        } else if (params[i].rfind(L"API=", 0) == 0) {
            host_api = portaudio_device_manager::parse_host_api(u8(params[i].substr(4)));
        } else if (params[i].rfind(L"CHANNELS=", 0) == 0) {
            try { device_channels = std::stoi(u8(params[i].substr(9))); }
            catch (...) {}
        } else if (params[i].rfind(L"FROM=", 0) == 0) {
            try { from_channel = std::stoi(u8(params[i].substr(5))); }
            catch (...) {}
        } else if (params[i].rfind(L"COUNT=", 0) == 0) {
            try { count = std::stoi(u8(params[i].substr(6))); }
            catch (...) {}
        } else if (params[i].rfind(L"MAP=", 0) == 0) {
            channel_map = parse_int_list(u8(params[i].substr(4)));
        } else if (params[i].rfind(L"DELAY=", 0) == 0) {
            try { delay_ms = std::stoi(u8(params[i].substr(6))); }
            catch (...) {}
        } else if (params[i].rfind(L"DELAY_FRAMES=", 0) == 0) {
            // Convert video frames to milliseconds
            try {
                int frames = std::stoi(u8(params[i].substr(13)));
                delay_ms = static_cast<int>(frames * 1000.0 / dependencies.format_desc.fps + 0.5);
            } catch (...) {}
        } else if (params[i] == L"SHARED") {
            use_shared = true;
        } else if (i == 1 && params[i] != L"EMPTY") {
            device_name = u8(params[i]);
        }
    }

    if (delay_ms < 0)
        delay_ms = 0;

    return spl::make_shared<portaudio_producer>(
        dependencies, device_name, host_api, device_channels,
        from_channel, count, std::move(channel_map), delay_ms, use_shared);
}

}} // namespace caspar::portaudio
