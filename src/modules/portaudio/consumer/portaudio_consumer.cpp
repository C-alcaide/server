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

#include "portaudio_consumer.h"

#include "../util/portaudio_device.h"
#include "../util/spsc_ring_buffer.h"

#include <common/diagnostics/graph.h>
#include <common/env.h>
#include <common/except.h>
#include <common/log.h>
#include <common/param.h>
#include <common/timer.h>
#include <common/utf.h>

#include <core/consumer/channel_info.h>
#include <core/consumer/frame_consumer.h>
#include <core/frame/frame.h>
#include <core/video_format.h>

#include <boost/algorithm/string/predicate.hpp>
#include <boost/property_tree/ptree.hpp>

#include <portaudio.h>

#include <atomic>
#include <chrono>
#include <future>
#include <memory>
#include <thread>
#include <vector>

namespace caspar { namespace portaudio {

struct portaudio_consumer : public core::frame_consumer
{
    spl::shared_ptr<diagnostics::graph> graph_;
    caspar::timer                       perf_timer_;
    int                                 channel_index_ = -1;

    core::video_format_desc format_desc_;

    // Configuration
    std::string          device_name_;
    host_api_preference  host_api_pref_  = host_api_preference::auto_select;
    int                  output_channels_ = 2;
    int                  buffer_frames_   = 4;    // Ring buffer depth in video frames
    int                  delay_frames_    = 0;    // External pipeline delay compensation

    // PortAudio stream
    PaStream*            stream_         = nullptr;
    int                  device_index_   = -1;
    int                  samples_per_frame_ = 0;  // audio samples per video frame (per channel)

    // Ring buffer: bridges push (send) to pull (PA callback)
    std::unique_ptr<spsc_ring_buffer> ring_buffer_;

    // Sync control
    std::atomic<bool>    stop_{false};
    std::atomic<bool>    started_{false};

    // Stats
    std::atomic<int64_t> underrun_count_{0};
    std::atomic<int64_t> overflow_count_{0};

    explicit portaudio_consumer(std::string  device_name,
                                host_api_preference host_api,
                                int          output_channels,
                                int          buffer_frames,
                                int          delay_frames)
        : device_name_(std::move(device_name))
        , host_api_pref_(host_api)
        , output_channels_(output_channels)
        , buffer_frames_(buffer_frames)
        , delay_frames_(delay_frames)
    {
        graph_->set_color("tick-time", diagnostics::color(0.0f, 0.6f, 0.9f));
        graph_->set_color("buffer-fill", diagnostics::color(0.2f, 0.8f, 0.2f));
        graph_->set_color("underrun", diagnostics::color(0.9f, 0.2f, 0.2f));
        graph_->set_color("overflow", diagnostics::color(0.9f, 0.6f, 0.0f));
        diagnostics::register_graph(graph_);
    }

    ~portaudio_consumer() override
    {
        stop_ = true;
        if (stream_) {
            Pa_StopStream(stream_);
            Pa_CloseStream(stream_);
            stream_ = nullptr;
        }
    }

    // --- PortAudio stream callback (called from audio hardware thread) ---
    static int stream_callback(const void*                     /*input*/,
                               void*                           output,
                               unsigned long                   frame_count,
                               const PaStreamCallbackTimeInfo* /*time_info*/,
                               PaStreamCallbackFlags           /*status_flags*/,
                               void*                           user_data)
    {
        auto* self = static_cast<portaudio_consumer*>(user_data);
        auto* out = static_cast<int32_t*>(output);

        size_t samples_needed = frame_count * self->output_channels_;
        size_t samples_read = self->ring_buffer_->read(out, samples_needed);

        // If underrun, fill remainder with silence
        if (samples_read < samples_needed) {
            std::memset(out + samples_read, 0, (samples_needed - samples_read) * sizeof(int32_t));
            self->underrun_count_.fetch_add(1, std::memory_order_relaxed);
        }

        return self->stop_.load(std::memory_order_relaxed) ? paComplete : paContinue;
    }

    // --- frame_consumer interface ---

    void initialize(const core::video_format_desc& format_desc,
                    const core::channel_info&      channel_info,
                    int                            /*port_index*/) override
    {
        format_desc_   = format_desc;
        channel_index_ = channel_info.index;
        graph_->set_text(print());

        // Calculate samples per video frame
        // Use minimum cadence value for consistent buffer sizing
        samples_per_frame_ = *std::min_element(format_desc_.audio_cadence.begin(),
                                               format_desc_.audio_cadence.end());

        // Allocate ring buffer: enough for buffer_frames_ worth of audio
        size_t ring_capacity = static_cast<size_t>(samples_per_frame_) * output_channels_ * (buffer_frames_ + 2);
        ring_buffer_ = std::make_unique<spsc_ring_buffer>(ring_capacity);

        // Find device
        auto& mgr = portaudio_device_manager::instance();
        if (!device_name_.empty()) {
            device_index_ = mgr.find_output_device(device_name_, host_api_pref_);
            if (device_index_ < 0) {
                CASPAR_LOG(warning) << print() << L" Device not found: " << u16(device_name_)
                                   << L". Using default.";
            }
        }

        if (device_index_ < 0) {
            device_index_ = mgr.get_default_output_device(host_api_pref_);
        }

        if (device_index_ < 0) {
            CASPAR_THROW_EXCEPTION(invalid_operation()
                << msg_info("No PortAudio output device available."));
        }

        const PaDeviceInfo* dev_info = Pa_GetDeviceInfo(device_index_);
        if (!dev_info) {
            CASPAR_THROW_EXCEPTION(invalid_operation()
                << msg_info("Failed to get PortAudio device info."));
        }

        // Clamp output channels to device capability
        if (output_channels_ > dev_info->maxOutputChannels) {
            CASPAR_LOG(warning) << print() << L" Requested " << output_channels_
                               << L" channels, device supports " << dev_info->maxOutputChannels
                               << L". Clamping.";
            output_channels_ = dev_info->maxOutputChannels;
        }

        // Open stream
        PaStreamParameters output_params = {};
        output_params.device                    = device_index_;
        output_params.channelCount              = output_channels_;
        output_params.sampleFormat              = paInt32;  // CasparCG native format
        output_params.suggestedLatency          = dev_info->defaultLowOutputLatency;
        output_params.hostApiSpecificStreamInfo = nullptr;

        PaError err = Pa_OpenStream(&stream_,
                                    nullptr,           // no input
                                    &output_params,
                                    format_desc_.audio_sample_rate,
                                    samples_per_frame_, // frames per buffer = one video frame's worth
                                    paClipOff,
                                    stream_callback,
                                    this);

        if (err != paNoError) {
            CASPAR_THROW_EXCEPTION(invalid_operation()
                << msg_info(std::string("Failed to open PortAudio stream: ") + Pa_GetErrorText(err)));
        }

        // Pre-fill ring buffer with silence (delay compensation + prevent initial underrun)
        int silence_frames = delay_frames_ + 1;
        size_t silence_samples = static_cast<size_t>(samples_per_frame_) * output_channels_ * silence_frames;
        std::vector<int32_t> silence(silence_samples, 0);
        ring_buffer_->write(silence.data(), silence_samples);

        // Start stream
        err = Pa_StartStream(stream_);
        if (err != paNoError) {
            Pa_CloseStream(stream_);
            stream_ = nullptr;
            CASPAR_THROW_EXCEPTION(invalid_operation()
                << msg_info(std::string("Failed to start PortAudio stream: ") + Pa_GetErrorText(err)));
        }

        started_ = true;

        const PaHostApiInfo* api_info = Pa_GetHostApiInfo(dev_info->hostApi);
        CASPAR_LOG(info) << print() << L" Opened device: " << dev_info->name
                        << L" [" << (api_info ? api_info->name : "?") << L"]"
                        << L" channels=" << output_channels_
                        << L" rate=" << format_desc_.audio_sample_rate
                        << L" buffer_frames=" << buffer_frames_
                        << L" delay=" << delay_frames_;
    }

    std::future<bool> send(core::video_field field, core::const_frame frame) override
    {
        // Skip field B for interlaced (audio is same for both fields)
        if (field == core::video_field::b)
            return make_ready_future(true);

        if (!started_)
            return make_ready_future(true);

        // Audio-as-master-clock: return a deferred future that blocks until
        // the ring buffer has space. This makes the ASIO hardware clock
        // pace the channel tick.
        return std::async(std::launch::deferred, [this, frame]() -> bool {
            if (stop_)
                return false;

            const auto& audio = frame.audio_data();
            if (!audio)
                return true;

            const int32_t* audio_ptr = audio.data();
            size_t audio_size = audio.size();

            // Determine how many samples to write (interleaved, output_channels_ wide)
            // CasparCG provides format_desc_.audio_channels channels interleaved as int32_t.
            // We may need to select a subset or all channels.
            int src_channels = format_desc_.audio_channels;
            int src_samples_per_channel = static_cast<int>(audio_size) / src_channels;
            size_t total_output_samples = static_cast<size_t>(src_samples_per_channel) * output_channels_;

            // Prepare output buffer with channel mapping
            // Simple mapping: take first output_channels_ from source, or zero-pad if source has fewer
            std::vector<int32_t> output_buffer(total_output_samples);
            for (int s = 0; s < src_samples_per_channel; ++s) {
                for (int c = 0; c < output_channels_; ++c) {
                    if (c < src_channels) {
                        // CasparCG audio is 16.16 fixed-point in int32_t
                        // PortAudio paInt32 expects full-scale 32-bit
                        int32_t sample = audio_ptr[s * src_channels + c];
                        output_buffer[s * output_channels_ + c] = sample;
                    } else {
                        output_buffer[s * output_channels_ + c] = 0;
                    }
                }
            }

            // Block until ring buffer has space — this is the master clock mechanism.
            // The ring buffer drains at the rate the ASIO/WASAPI hardware consumes samples.
            size_t written = 0;
            while (written < total_output_samples && !stop_) {
                size_t w = ring_buffer_->write(output_buffer.data() + written,
                                               total_output_samples - written);
                written += w;

                if (written < total_output_samples) {
                    // Buffer full — wait for hardware to consume. Poll at 100µs.
                    std::this_thread::sleep_for(std::chrono::microseconds(100));
                }
            }

            if (written < total_output_samples) {
                overflow_count_.fetch_add(1, std::memory_order_relaxed);
            }

            // Update diagnostics
            double fill_ratio = static_cast<double>(ring_buffer_->read_available()) /
                               static_cast<double>(ring_buffer_->capacity());
            graph_->set_value("buffer-fill", fill_ratio);
            graph_->set_value("tick-time", perf_timer_.elapsed() * format_desc_.fps * 0.5);
            perf_timer_.restart();

            return true;
        });
    }

    std::wstring print() const override
    {
        return L"portaudio[" + std::to_wstring(channel_index_) + L"|" + format_desc_.name + L"]";
    }

    std::wstring name() const override { return L"portaudio"; }

    bool has_synchronization_clock() const override { return true; }

    int index() const override { return 510; }

    core::monitor::state state() const override
    {
        core::monitor::state s;
        s["buffer/fill"]    = static_cast<int64_t>(ring_buffer_ ? ring_buffer_->read_available() : 0);
        s["buffer/underruns"] = underrun_count_.load(std::memory_order_relaxed);
        s["buffer/overflows"] = overflow_count_.load(std::memory_order_relaxed);
        return s;
    }
};

// --- Factory functions ---

spl::shared_ptr<core::frame_consumer> create_consumer(const std::vector<std::wstring>&     params,
                                                      const core::video_format_repository& /*format_repository*/,
                                                      const std::vector<spl::shared_ptr<core::video_channel>>& /*channels*/,
                                                      const core::channel_info& /*channel_info*/)
{
    if (params.empty() || !boost::iequals(params.at(0), L"PORTAUDIO"))
        return core::frame_consumer::empty();

    std::string device_name;
    if (params.size() > 1)
        device_name = u8(params.at(1));

    int channels = 2;
    if (params.size() > 2) {
        try { channels = std::stoi(u8(params.at(2))); }
        catch (...) {}
    }

    return spl::make_shared<portaudio_consumer>(
        device_name,
        host_api_preference::auto_select,
        channels,
        4,   // default buffer depth
        0);  // default delay
}

spl::shared_ptr<core::frame_consumer>
create_preconfigured_consumer(const boost::property_tree::wptree&                      element,
                              const core::video_format_repository&                     /*format_repository*/,
                              const std::vector<spl::shared_ptr<core::video_channel>>& /*channels*/,
                              const core::channel_info&                                /*channel_info*/)
{
    std::string device_name = u8(element.get(L"device", L""));
    std::string host_api_str = u8(element.get(L"host-api", L"auto"));
    int output_channels = element.get(L"channels", 2);
    int buffer_frames   = element.get(L"buffer-size", 4);
    int delay_frames    = element.get(L"delay", 0);

    auto host_api = portaudio_device_manager::parse_host_api(host_api_str);

    return spl::make_shared<portaudio_consumer>(
        device_name,
        host_api,
        output_channels,
        buffer_frames,
        delay_frames);
}

}} // namespace caspar::portaudio
