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

#include "spsc_ring_buffer.h"

#include <common/log.h>

#include <portaudio.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace caspar { namespace portaudio {

/// Interface for objects that want to receive captured audio from a shared stream.
class capture_listener
{
  public:
    virtual ~capture_listener() = default;

    /// Called from the pump thread (NOT the PA callback) with the full interleaved
    /// capture buffer. Listeners should copy the data they need quickly.
    /// @param interleaved  Full device-width interleaved int32_t samples
    /// @param frame_count  Number of audio frames (samples per channel)
    /// @param channels     Number of interleaved channels
    virtual void on_captured_audio(const int32_t* interleaved, size_t frame_count, int channels) = 0;

    /// Called when the shared stream detects a device disconnection or fatal error.
    virtual void on_capture_disconnected() = 0;
};

/// Owns a single PaStream* for a capture device and distributes captured audio
/// to all registered listeners. Multiple producers targeting the same device
/// share one SharedPortAudioCapture via the device manager.
///
/// Lifecycle: managed by shared_ptr. When the last producer releases its
/// shared_ptr, the stream is stopped and closed automatically.
class shared_portaudio_capture : public std::enable_shared_from_this<shared_portaudio_capture>
{
  public:
    shared_portaudio_capture(int device_index, int channels, int sample_rate)
        : device_index_(device_index)
        , channels_(channels)
        , sample_rate_(sample_rate)
    {
        // Ring buffer: 2 seconds of full-width capture
        size_t ring_cap = static_cast<size_t>(sample_rate_) * channels_ * 2;
        ring_ = std::make_unique<spsc_ring_buffer>(ring_cap);
    }

    ~shared_portaudio_capture()
    {
        stop();
    }

    /// Start the PA stream. Returns true on success.
    bool start()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stream_)
            return true;  // already running

        const PaDeviceInfo* dev_info = Pa_GetDeviceInfo(device_index_);
        if (!dev_info)
            return false;

        PaStreamParameters input_params = {};
        input_params.device            = device_index_;
        input_params.channelCount      = channels_;
        input_params.sampleFormat      = paInt32;
        input_params.suggestedLatency  = dev_info->defaultLowInputLatency;

        int frames_per_buffer = sample_rate_ / 50;  // ~20ms chunks

        PaError err = Pa_OpenStream(&stream_,
                                    &input_params,
                                    nullptr,
                                    sample_rate_,
                                    frames_per_buffer,
                                    paClipOff,
                                    pa_callback,
                                    this);
        if (err != paNoError) {
            CASPAR_LOG(error) << L"[shared-capture] Failed to open stream: " << Pa_GetErrorText(err);
            stream_ = nullptr;
            return false;
        }

        running_ = true;

        // Start pump thread: reads from ring buffer and distributes to listeners
        pump_thread_ = std::thread(&shared_portaudio_capture::pump_thread_func, this);

        err = Pa_StartStream(stream_);
        if (err != paNoError) {
            CASPAR_LOG(error) << L"[shared-capture] Failed to start stream: " << Pa_GetErrorText(err);
            running_ = false;
            if (pump_thread_.joinable())
                pump_thread_.join();
            Pa_CloseStream(stream_);
            stream_ = nullptr;
            return false;
        }

        CASPAR_LOG(info) << L"[shared-capture] Started device " << device_index_
                        << L" channels=" << channels_ << L" rate=" << sample_rate_;
        return true;
    }

    /// Stop the PA stream.
    void stop()
    {
        running_ = false;
        if (pump_thread_.joinable())
            pump_thread_.join();

        std::lock_guard<std::mutex> lock(mutex_);
        if (stream_) {
            Pa_StopStream(stream_);
            Pa_CloseStream(stream_);
            stream_ = nullptr;
            CASPAR_LOG(info) << L"[shared-capture] Stopped device " << device_index_;
        }
    }

    void add_listener(capture_listener* listener)
    {
        std::lock_guard<std::mutex> lock(listeners_mutex_);
        listeners_.push_back(listener);
    }

    void remove_listener(capture_listener* listener)
    {
        std::lock_guard<std::mutex> lock(listeners_mutex_);
        listeners_.erase(
            std::remove(listeners_.begin(), listeners_.end(), listener),
            listeners_.end());
    }

    int  device_index() const { return device_index_; }
    int  channels()     const { return channels_; }
    int  sample_rate()  const { return sample_rate_; }
    bool is_running()   const { return running_.load(std::memory_order_relaxed); }

    bool is_disconnected() const { return disconnected_.load(std::memory_order_relaxed); }

    size_t listener_count() const
    {
        std::lock_guard<std::mutex> lock(listeners_mutex_);
        return listeners_.size();
    }

  private:
    static int pa_callback(const void*                     input,
                           void*                           /*output*/,
                           unsigned long                   frame_count,
                           const PaStreamCallbackTimeInfo* /*time_info*/,
                           PaStreamCallbackFlags           /*status_flags*/,
                           void*                           user_data)
    {
        auto* self = static_cast<shared_portaudio_capture*>(user_data);
        if (!self || !input)
            return paContinue;

        const auto* samples = static_cast<const int32_t*>(input);
        size_t count = frame_count * self->channels_;

        self->ring_->write(samples, count);

        return self->running_.load(std::memory_order_relaxed) ? paContinue : paComplete;
    }

    void pump_thread_func()
    {
        // Temporary buffer for reading from ring
        const size_t chunk_frames = static_cast<size_t>(sample_rate_) / 50;  // ~20ms
        const size_t chunk_samples = chunk_frames * channels_;
        std::vector<int32_t> buf(chunk_samples);

        while (running_) {
            size_t avail = ring_->read_available();
            if (avail < chunk_samples) {
                // Check device health
                if (stream_ && !Pa_IsStreamActive(stream_) && running_) {
                    disconnected_ = true;
                    CASPAR_LOG(warning) << L"[shared-capture] Device " << device_index_
                                       << L" disconnected.";
                    std::lock_guard<std::mutex> lock(listeners_mutex_);
                    for (auto* l : listeners_)
                        l->on_capture_disconnected();
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }

            size_t read = ring_->read(buf.data(), chunk_samples);
            if (read == 0)
                continue;

            size_t frames = read / channels_;
            std::lock_guard<std::mutex> lock(listeners_mutex_);
            for (auto* l : listeners_)
                l->on_captured_audio(buf.data(), frames, channels_);
        }
    }

    int device_index_;
    int channels_;
    int sample_rate_;

    PaStream*       stream_  = nullptr;
    std::mutex      mutex_;
    std::thread     pump_thread_;

    std::unique_ptr<spsc_ring_buffer> ring_;

    std::atomic<bool> running_{false};
    std::atomic<bool> disconnected_{false};

    mutable std::mutex              listeners_mutex_;
    std::vector<capture_listener*>  listeners_;
};

}} // namespace caspar::portaudio
