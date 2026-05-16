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

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <future>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>

namespace caspar { namespace portaudio {

/// Parse a comma-separated list of integers (e.g. "0,1,0,1,-1,-1").
static std::vector<int> parse_int_list(const std::string& str)
{
    std::vector<int> result;
    std::istringstream ss(str);
    std::string token;
    while (std::getline(ss, token, ',')) {
        try { result.push_back(std::stoi(token)); }
        catch (...) {}
    }
    return result;
}

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

    // Channel routing matrix: channel_map_[output_ch] = source_ch (or -1 for silence)
    // If empty, uses default 1:1 mapping with zero-pad.
    std::vector<int>     channel_map_;

    // PortAudio stream
    PaStream*            stream_         = nullptr;
    int                  device_index_   = -1;

    // Ring buffer: bridges push (write thread) to pull (PA callback)
    std::unique_ptr<spsc_ring_buffer> ring_buffer_;

    // Drain notification: PA callback signals after consuming samples from ring buffer
    std::mutex              drain_mutex_;
    std::condition_variable drain_cv_;

    // Write thread: takes pending writes from queue, writes to ring buffer (may block)
    std::thread             write_thread_;
    std::mutex              queue_mutex_;
    std::condition_variable queue_cv_;
    struct pending_write {
        std::vector<int32_t> samples;
        std::promise<bool>   promise;
    };
    std::deque<pending_write> write_queue_;

    // Sync control
    std::atomic<bool>    stop_{false};
    std::atomic<bool>    started_{false};

    // Hardware-reported device output latency (set after Pa_OpenStream)
    double device_latency_ms_ = 0.0;

    // Stats
    std::atomic<int64_t> underrun_count_{0};
    std::atomic<int64_t> overflow_count_{0};

    explicit portaudio_consumer(std::string  device_name,
                                host_api_preference host_api,
                                int          output_channels,
                                int          buffer_frames,
                                int          delay_frames,
                                std::vector<int> channel_map)
        : device_name_(std::move(device_name))
        , host_api_pref_(host_api)
        , output_channels_(output_channels)
        , buffer_frames_(buffer_frames)
        , delay_frames_(delay_frames)
        , channel_map_(std::move(channel_map))
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
        queue_cv_.notify_one();
        drain_cv_.notify_all();
        if (write_thread_.joinable())
            write_thread_.join();
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

        // Signal write thread that buffer space is now available
        self->drain_cv_.notify_one();

        return self->stop_.load(std::memory_order_relaxed) ? paComplete : paContinue;
    }

    // --- Write thread: drains queue into ring buffer, blocking on drain_cv_ ---
    void write_thread_func()
    {
        while (!stop_) {
            pending_write pw;
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                queue_cv_.wait(lock, [&] { return !write_queue_.empty() || stop_; });
                if (stop_ && write_queue_.empty())
                    break;
                pw = std::move(write_queue_.front());
                write_queue_.pop_front();
            }

            // Check device health (hot-unplug detection)
            if (stream_ && !Pa_IsStreamActive(stream_) && !stop_) {
                CASPAR_LOG(warning) << print() << L" Audio device disconnected or stream error.";
                pw.promise.set_value(false);
                continue;
            }

            // Write to ring buffer, blocking via condvar until space is available.
            // This is the master clock mechanism: the ring buffer drains at the
            // rate the hardware consumes samples (ASIO/WASAPI clock).
            const int32_t* data = pw.samples.data();
            size_t total = pw.samples.size();
            size_t written = 0;

            while (written < total && !stop_) {
                size_t w = ring_buffer_->write(data + written, total - written);
                written += w;
                if (written < total) {
                    std::unique_lock<std::mutex> lock(drain_mutex_);
                    drain_cv_.wait_for(lock, std::chrono::milliseconds(5));
                }
            }

            if (written < total && !stop_) {
                overflow_count_.fetch_add(1, std::memory_order_relaxed);
            }

            // Update diagnostics
            double fill_ratio = static_cast<double>(ring_buffer_->read_available()) /
                               static_cast<double>(ring_buffer_->capacity());
            graph_->set_value("buffer-fill", fill_ratio);
            graph_->set_value("tick-time", perf_timer_.elapsed() * format_desc_.fps * 0.5);
            perf_timer_.restart();

            pw.promise.set_value(!stop_);
        }

        // Drain remaining promises on shutdown
        std::lock_guard<std::mutex> lock(queue_mutex_);
        for (auto& pw : write_queue_)
            pw.promise.set_value(false);
        write_queue_.clear();
    }

    // --- frame_consumer interface ---

    void initialize(const core::video_format_desc& format_desc,
                    const core::channel_info&      channel_info,
                    int                            /*port_index*/) override
    {
        format_desc_   = format_desc;
        channel_index_ = channel_info.index;
        graph_->set_text(print());

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
        if (!channel_map_.empty()) {
            // MAP= determines output channel count
            output_channels_ = static_cast<int>(channel_map_.size());
        }
        if (output_channels_ > dev_info->maxOutputChannels) {
            CASPAR_LOG(warning) << print() << L" Requested " << output_channels_
                               << L" channels, device supports " << dev_info->maxOutputChannels
                               << L". Clamping.";
            output_channels_ = dev_info->maxOutputChannels;
            if (!channel_map_.empty())
                channel_map_.resize(output_channels_);
        }

        // Calculate average samples per video frame for ring buffer sizing.
        // Using average (not min) prevents drift on variable-cadence formats like NTSC 29.97.
        // NOTE: Must be done AFTER channel_map_ adjusts output_channels_.
        int total_cadence = 0;
        for (auto c : format_desc_.audio_cadence)
            total_cadence += c;
        int avg_samples = total_cadence / static_cast<int>(format_desc_.audio_cadence.size());

        // Allocate ring buffer: enough for buffer_frames_ worth of audio + headroom
        size_t ring_capacity = static_cast<size_t>(avg_samples) * output_channels_ * (buffer_frames_ + 2);
        ring_buffer_ = std::make_unique<spsc_ring_buffer>(ring_capacity);

        // Open stream — let hardware choose optimal callback buffer size
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
                                    paFramesPerBufferUnspecified,
                                    paClipOff,
                                    stream_callback,
                                    this);

        if (err != paNoError) {
            CASPAR_THROW_EXCEPTION(invalid_operation()
                << msg_info(std::string("Failed to open PortAudio stream: ") + Pa_GetErrorText(err)));
        }

        // Query actual device output latency
        const PaStreamInfo* stream_info = Pa_GetStreamInfo(stream_);
        if (stream_info) {
            device_latency_ms_ = stream_info->outputLatency * 1000.0;
        }

        // Pre-fill ring buffer with silence (delay compensation + prevent initial underrun)
        int silence_frames = delay_frames_ + 1;
        size_t silence_samples = static_cast<size_t>(avg_samples) * output_channels_ * silence_frames;
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

        // Start write thread
        write_thread_ = std::thread(&portaudio_consumer::write_thread_func, this);

        started_ = true;

        const PaHostApiInfo* api_info = Pa_GetHostApiInfo(dev_info->hostApi);
        CASPAR_LOG(info) << print() << L" Opened device: " << dev_info->name
                        << L" [" << (api_info ? api_info->name : "?") << L"]"
                        << L" channels=" << output_channels_
                        << L" rate=" << format_desc_.audio_sample_rate
                        << L" buffer_frames=" << buffer_frames_
                        << L" delay=" << delay_frames_
                        << L" map=" << (channel_map_.empty() ? "default" : "custom");
    }

    std::future<bool> send(core::video_field field, core::const_frame frame) override
    {
        // Skip field B for interlaced (audio is same for both fields)
        if (field == core::video_field::b)
            return make_ready_future(true);

        if (!started_)
            return make_ready_future(true);

        if (stop_)
            return make_ready_future(false);

        const auto& audio = frame.audio_data();
        if (!audio)
            return make_ready_future(true);

        const int32_t* audio_ptr = audio.data();
        size_t audio_size = audio.size();

        // Determine how many samples to write (interleaved, output_channels_ wide)
        int src_channels = format_desc_.audio_channels;
        int src_samples_per_channel = static_cast<int>(audio_size) / src_channels;
        size_t total_output_samples = static_cast<size_t>(src_samples_per_channel) * output_channels_;

        // Channel mapping: use explicit map if configured, otherwise default 1:1 + zero-pad
        std::vector<int32_t> output_buffer(total_output_samples);
        bool has_map = !channel_map_.empty() && static_cast<int>(channel_map_.size()) == output_channels_;
        for (int s = 0; s < src_samples_per_channel; ++s) {
            for (int c = 0; c < output_channels_; ++c) {
                int src_ch = has_map ? channel_map_[c] : c;
                output_buffer[s * output_channels_ + c] =
                    (src_ch >= 0 && src_ch < src_channels) ? audio_ptr[s * src_channels + src_ch] : 0;
            }
        }

        // Post to write thread — returns future that completes when ring buffer write is done.
        // The write thread blocks until the hardware drains enough space, providing the
        // master clock mechanism. Because the work runs on a dedicated thread, this
        // consumer's future can be awaited in parallel with other consumers (e.g. decklink).
        std::promise<bool> promise;
        auto future = promise.get_future();
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            write_queue_.push_back({std::move(output_buffer), std::move(promise)});
        }
        queue_cv_.notify_one();
        return future;
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

    core::av_pipeline_info av_pipeline() const override
    {
        core::av_pipeline_info info;
        info.has_audio               = true;
        info.audio_depth_frames      = buffer_frames_ + delay_frames_ + 1;
        info.audio_device_latency_ms = device_latency_ms_;
        info.audio_delay_adjustable  = true;
        info.audio_delay_frames      = delay_frames_;
        return info;
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
    host_api_preference host_api = host_api_preference::auto_select;
    int channels      = 2;
    int buffer_frames  = 4;
    int delay_frames   = 0;
    std::vector<int> channel_map;

    for (size_t i = 1; i < params.size(); ++i) {
        if (boost::istarts_with(params[i], L"DEVICE=")) {
            device_name = u8(params[i].substr(7));
        } else if (boost::istarts_with(params[i], L"API=")) {
            host_api = portaudio_device_manager::parse_host_api(u8(params[i].substr(4)));
        } else if (boost::istarts_with(params[i], L"CHANNELS=")) {
            try { channels = std::stoi(u8(params[i].substr(9))); }
            catch (...) {}
        } else if (boost::istarts_with(params[i], L"BUFFER=")) {
            try { buffer_frames = std::stoi(u8(params[i].substr(7))); }
            catch (...) {}
        } else if (boost::istarts_with(params[i], L"DELAY=")) {
            try { delay_frames = std::stoi(u8(params[i].substr(6))); }
            catch (...) {}
        } else if (boost::istarts_with(params[i], L"MAP=")) {
            channel_map = parse_int_list(u8(params[i].substr(4)));
        } else if (i == 1 && !params[i].empty()) {
            device_name = u8(params[i]);
        }
    }

    return spl::make_shared<portaudio_consumer>(
        device_name,
        host_api,
        channels,
        buffer_frames,
        delay_frames,
        std::move(channel_map));
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

    // Parse channel-map: comma-separated ints, e.g. "0,1,0,1,-1,-1"
    std::vector<int> channel_map;
    auto map_str = u8(element.get(L"channel-map", L""));
    if (!map_str.empty())
        channel_map = parse_int_list(map_str);

    auto host_api = portaudio_device_manager::parse_host_api(host_api_str);

    return spl::make_shared<portaudio_consumer>(
        device_name,
        host_api,
        output_channels,
        buffer_frames,
        delay_frames,
        std::move(channel_map));
}

}} // namespace caspar::portaudio
