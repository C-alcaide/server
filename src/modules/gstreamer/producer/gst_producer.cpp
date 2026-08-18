/*
 * Copyright (c) 2011 Sveriges Television AB <info@casparcg.com>
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

#include "../StdAfx.h"

#include "gst_producer.h"

#include "../util/gst_frame.h"
#include "../util/gst_runtime.h"

#ifdef _WIN32
#include "../util/gst_d3d11.h"
#endif

#include <common/array.h>

#include <core/frame/draw_frame.h>
#include <core/frame/frame_factory.h>
#include <core/frame/pixel_format.h>
#include <core/monitor/monitor.h>
#include <core/video_format.h>

#include <common/bit_depth.h>

#include <common/diagnostics/graph.h>
#include <common/except.h>
#include <common/future.h>
#include <common/log.h>
#include <common/param.h>
#include <common/timer.h>
#include <common/utf.h>

#include <ffmpeg/util/av_util.h>

extern "C" {
#include <libavutil/channel_layout.h>
#include <libavutil/frame.h>
#include <libavutil/samplefmt.h>
}

#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>

#include <gst/app/gstappsink.h>
#include <gst/gst.h>

#include <atomic>
#include <chrono>
#include <limits>
#include <deque>
#include <mutex>
#include <queue>
#include <shared_mutex>
#include <thread>

namespace caspar { namespace gstreamer {

namespace {

/// How many decoded frames may sit between the pipeline and the channel. Deep enough to
/// absorb a decode that is briefly late, shallow enough that a producer switched away from
/// and back does not present stale pictures.
constexpr std::size_t max_queued_frames = 4;

/// How long to wait before rebuilding a pipeline that errored, and the ceiling that backoff
/// climbs to. A live source that drops — a sender restarted, a network that blinked — must not
/// take the layer down with it, and must not spin on a source that is simply gone either.
constexpr auto reconnect_delay_initial = std::chrono::milliseconds(500);
constexpr auto reconnect_delay_max     = std::chrono::seconds(5);

/// The names the module looks for after parsing, and writes into the description when it
/// appends the sink itself.
constexpr const char* video_sink_name = "caspar_video";
constexpr const char* audio_sink_name = "caspar_audio";

/// What gets appended when the description names no sink of its own. The GPU tail keeps the
/// frame in video memory from the decoder to the mixer; the host tail is the portable one and
/// stays the default.
constexpr const char* host_tail = " ! videoconvert ! videorate ! appsink name=";
constexpr const char* gpu_tail =
    " ! d3d11upload ! d3d11convert ! video/x-raw(memory:D3D11Memory),format=BGRA ! appsink name=";

/// Reads whichever of the two a message actually carries. Parsing a WARNING with
/// gst_message_parse_error leaves the GError null and prints "unknown error", which is how
/// every warning this producer logged came out saying nothing at all.
std::wstring describe_message(GstMessage* message)
{
    GError* error = nullptr;
    gchar*  debug = nullptr;

    if (GST_MESSAGE_TYPE(message) == GST_MESSAGE_WARNING)
        gst_message_parse_warning(message, &error, &debug);
    else
        gst_message_parse_error(message, &error, &debug);

    std::wstring text = error ? u16(std::string(error->message)) : L"unknown error";
    if (debug != nullptr)
        text += L" (" + u16(std::string(debug)) + L")";

    if (error != nullptr)
        g_error_free(error);
    g_free(debug);

    return text;
}

} // namespace

struct gst_producer : public core::frame_producer
{
    static std::atomic<int> instances_;
    const int               instance_no_;
    const std::wstring      description_;

    spl::shared_ptr<core::frame_factory> frame_factory_;
    const core::video_format_desc        format_desc_;

    // Guards the three pointers below against a rebuild happening under the pull threads.
    // Shared, not exclusive, for the pulls: both threads block in try_pull_sample for up to
    // 100 ms at a time and serialising them would halve the throughput of a producer that is
    // working perfectly well, to protect against something that happens when it is not.
    mutable std::shared_mutex pipeline_mutex_;
    GstElement*               pipeline_   = nullptr;
    GstElement*               video_sink_ = nullptr;
    GstElement*               audio_sink_ = nullptr;

    spl::shared_ptr<diagnostics::graph> graph_;
    timer                               tick_timer_;
    timer                               frame_timer_;

    std::queue<core::draw_frame> frames_;
    mutable std::mutex           frames_mutex_;
    core::draw_frame             last_frame_;

    // Audio arrives on its own sink and its own thread, and is only paired with a picture when
    // one is built — the channel takes audio attached to a frame, at a cadence the format
    // decides, not at whatever rate the pipeline happens to deliver it.
    std::deque<int32_t> audio_samples_;
    mutable std::mutex  audio_mutex_;
    std::atomic<int>    audio_channels_{0};
    std::size_t         cadence_counter_ = 0;

    const bool want_gpu_;
#ifdef _WIN32
    std::unique_ptr<d3d11_bridge> gpu_bridge_;
#endif
    std::atomic<uint64_t> frames_on_gpu_{0};

    std::atomic<bool>     is_running_{true};
    std::atomic<bool>     is_eos_{false};
    std::atomic<uint64_t> frames_received_{0};
    std::atomic<uint64_t> frames_dropped_{0};
    std::atomic<uint64_t> audio_underruns_{0};
    std::atomic<uint64_t> restarts_{0};

    // Whether the pipeline is keeping ahead of the channel, which "frames received" cannot
    // say: a source delivering 25 fps into a 50 fps channel still counts every frame it sent.
    // `starved` counts ticks that found the queue empty and had to repeat the last picture —
    // the direct symptom — and `queue_peak` says how much slack there was when it did not.
    std::atomic<uint64_t> frames_starved_{0};
    std::atomic<uint64_t> queue_peak_{0};
    std::atomic<bool>     is_failed_{false};
    std::thread           video_thread_;
    std::thread           audio_thread_;

  public:
    gst_producer(spl::shared_ptr<core::frame_factory> frame_factory,
                 core::video_format_desc              format_desc,
                 std::wstring                         description,
                 bool                                 want_gpu)
        : instance_no_(instances_++)
        , description_(std::move(description))
        , frame_factory_(std::move(frame_factory))
        , format_desc_(std::move(format_desc))
        , want_gpu_(want_gpu)
    {
        graph_->set_text(print());
        graph_->set_color("frame-time", diagnostics::color(0.5f, 1.0f, 0.2f));
        graph_->set_color("tick-time", diagnostics::color(0.0f, 0.6f, 0.9f));
        graph_->set_color("dropped-frame", diagnostics::color(0.3f, 0.6f, 0.3f));
        graph_->set_color("audio-underrun", diagnostics::color(0.6f, 0.3f, 0.3f));
        diagnostics::register_graph(graph_);

        build_pipeline();

        // set_state returning ASYNC says only that the change was accepted. A pipeline that
        // cannot link — the commonest description mistake — fails afterwards, on the bus, and
        // reporting 202 for it puts the error in the log minutes after the operator stopped
        // looking. So wait for the change to settle, and treat the failure as a failed PLAY.
        auto change = gst_element_set_state(pipeline_, GST_STATE_PLAYING);
        if (change == GST_STATE_CHANGE_ASYNC) {
            GstState state = GST_STATE_NULL;
            change         = gst_element_get_state(pipeline_, &state, nullptr, 5 * GST_SECOND);

            if (change == GST_STATE_CHANGE_ASYNC) {
                // Still settling after 5 s. A slow network source is a legitimate reason, so
                // this is a warning and the producer carries on.
                CASPAR_LOG(warning) << print() << L" Still preparing after 5 s; continuing.";
                change = GST_STATE_CHANGE_SUCCESS;
            }
        }

        if (change == GST_STATE_CHANGE_FAILURE) {
            const auto reason = drain_bus_error();
            destroy_pipeline();
            CASPAR_LOG(error) << print() << L" Failed to start: " << reason;
            CASPAR_THROW_EXCEPTION(user_error() << msg_info(u8(L"Failed to start GStreamer pipeline: " + reason)));
        }

#ifdef _WIN32
        // Audio is no longer a reason to refuse: import_d3d_texture takes the samples and puts
        // them on the frame it builds, so a pipeline with both keeps its picture on the GPU.
        if (want_gpu_)
            gpu_bridge_ = std::make_unique<d3d11_bridge>();
#endif

        video_thread_ = std::thread([this] { run_video(); });
        if (audio_sink_ != nullptr)
            audio_thread_ = std::thread([this] { run_audio(); });

        CASPAR_LOG(info) << print() << L" Initialized" << (audio_sink_ != nullptr ? L" with audio." : L".");
    }

    ~gst_producer()
    {
        is_running_ = false;
        if (video_thread_.joinable())
            video_thread_.join();
        if (audio_thread_.joinable())
            audio_thread_.join();

        std::unique_lock<std::shared_mutex> lock(pipeline_mutex_);
        destroy_pipeline();
    }

    // The description is gst-launch syntax, and it is parsed as such — which is what makes a
    // dynamic source work: gst_parse_launch installs delayed links, so "filesrc ! decodebin"
    // links to our sink when decodebin produces its pad, minutes of code later avoided.
    //
    // A description that does not name caspar_video gets the sink appended, which covers the
    // single-chain case. A description that does name it is taken as written, so a second
    // chain can be routed to caspar_audio.
    void build_pipeline()
    {
        auto launch = u8(description_);

        if (launch.find(video_sink_name) == std::string::npos)
            launch += std::string(want_gpu_ ? gpu_tail : host_tail) + video_sink_name;

        GError* error = nullptr;
        pipeline_     = gst_parse_launch(launch.c_str(), &error);

        if (pipeline_ == nullptr) {
            const std::wstring reason = error ? u16(std::string(error->message)) : L"unknown error";
            if (error != nullptr)
                g_error_free(error);
            CASPAR_LOG(error) << L"[gstreamer] Failed to parse pipeline: " << reason;
            CASPAR_THROW_EXCEPTION(user_error() << msg_info(u8(L"Failed to parse GStreamer pipeline: " + reason)));
        }
        if (error != nullptr) {
            // gst_parse_launch reports a recoverable parse problem this way while still
            // returning a pipeline — an element that could not be created, or a link it had to
            // omit. The pipeline that comes back is missing whatever the message names, so it
            // is reported as an error rather than a warning: a description whose link was
            // dropped runs, reaches EOS and delivers nothing, which is the hardest kind of
            // failure to read from the outside.
            const auto reason = u16(std::string(error->message));
            g_error_free(error);
            destroy_pipeline();
            CASPAR_LOG(error) << L"[gstreamer] Incomplete pipeline: " << reason;
            CASPAR_THROW_EXCEPTION(user_error() << msg_info(u8(L"Incomplete GStreamer pipeline: " + reason)));
        }

        video_sink_ = gst_bin_get_by_name(GST_BIN(pipeline_), video_sink_name);
        audio_sink_ = gst_bin_get_by_name(GST_BIN(pipeline_), audio_sink_name);

        if (video_sink_ == nullptr) {
            destroy_pipeline();
            CASPAR_THROW_EXCEPTION(user_error() << msg_info(
                                       "The GStreamer description names no appsink called caspar_video. Either end "
                                       "the description before the sink and let it be appended, or include "
                                       "'appsink name=caspar_video' yourself."));
        }

        configure_video_sink();
        if (audio_sink_ != nullptr)
            configure_audio_sink();
    }

    void configure_video_sink()
    {
        const auto rate = std::string(", framerate=(fraction)") +
                          std::to_string(format_desc_.framerate.numerator()) + "/" +
                          std::to_string(format_desc_.framerate.denominator());

        // On the GPU path both are offered, GPU first: a source that cannot reach D3D11 memory
        // then negotiates the host caps and the producer keeps working rather than failing to
        // link, which is the difference between an optimisation and a restriction.
        const auto caps_text =
            (want_gpu_ ? std::string("video/x-raw(memory:D3D11Memory), format=(string)BGRA") + rate + "; "
                       : std::string("")) +
            std::string("video/x-raw, format=(string){ ") + supported_caps_formats() + " }" + rate;

        auto* caps = gst_caps_from_string(caps_text.c_str());
        g_object_set(G_OBJECT(video_sink_),
                     "emit-signals",
                     FALSE,
                     "sync",
                     TRUE, // the pipeline paces itself on its own timestamps; we pull what it produces
                     "max-buffers",
                     static_cast<guint>(max_queued_frames),
                     "drop",
                     TRUE,
                     "caps",
                     caps,
                     nullptr);
        gst_caps_unref(caps);
    }

    void configure_audio_sink()
    {
        // Interleaved S32 at the channel's rate, which is what core::mutable_frame carries.
        // Channel count is left free: whatever the source has, make_frame maps into the
        // channel's 16.
        const auto caps_text = std::string("audio/x-raw, format=(string)S32LE, layout=(string)interleaved, "
                                           "rate=(int)") +
                               std::to_string(format_desc_.audio_sample_rate);

        auto* caps = gst_caps_from_string(caps_text.c_str());
        g_object_set(G_OBJECT(audio_sink_),
                     "emit-signals",
                     FALSE,
                     "sync",
                     TRUE,
                     "max-buffers",
                     static_cast<guint>(64),
                     "drop",
                     TRUE,
                     "caps",
                     caps,
                     nullptr);
        gst_caps_unref(caps);
    }

    void destroy_pipeline()
    {
        if (video_sink_ != nullptr) {
            gst_object_unref(video_sink_);
            video_sink_ = nullptr;
        }
        if (audio_sink_ != nullptr) {
            gst_object_unref(audio_sink_);
            audio_sink_ = nullptr;
        }
        if (pipeline_ != nullptr) {
            gst_element_set_state(pipeline_, GST_STATE_NULL);
            gst_object_unref(pipeline_);
            pipeline_ = nullptr;
        }
    }

    std::wstring drain_bus_error()
    {
        std::wstring reason = L"no error on the bus";

        auto* bus = gst_element_get_bus(pipeline_);
        if (bus == nullptr)
            return reason;

        while (auto* message = gst_bus_pop_filtered(bus, GST_MESSAGE_ERROR)) {
            reason = describe_message(message);
            gst_message_unref(message);
        }
        gst_object_unref(bus);

        return reason;
    }

    void poll_bus()
    {
        std::shared_lock<std::shared_mutex> lock(pipeline_mutex_);
        if (pipeline_ == nullptr)
            return;

        auto* bus = gst_element_get_bus(pipeline_);
        if (bus == nullptr)
            return;

        while (auto* message = gst_bus_pop_filtered(
                   bus, static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_WARNING | GST_MESSAGE_EOS))) {
            switch (GST_MESSAGE_TYPE(message)) {
                case GST_MESSAGE_ERROR:
                    // Not is_running_ = false. That was the original behaviour and it is the
                    // wrong one for everything this module is for: a live source that drops
                    // took the layer down permanently, with the producer still on air showing
                    // its last frame and no way back short of a re-PLAY.
                    CASPAR_LOG(error) << print() << L" " << describe_message(message);
                    is_failed_ = true;
                    break;
                case GST_MESSAGE_WARNING:
                    CASPAR_LOG(warning) << print() << L" " << describe_message(message);
                    break;
                case GST_MESSAGE_EOS:
                    CASPAR_LOG(info) << print() << L" End of stream.";
                    is_eos_ = true;
                    break;
                default:
                    break;
            }
            gst_message_unref(message);
        }

        gst_object_unref(bus);
    }

    /// One frame's worth of audio, at the format's rotating cadence, **already widened to the
    /// channel's 16**. Short reads are padded with silence and counted rather than stalling the
    /// picture.
    ///
    /// The widening is here rather than left to `ffmpeg::make_frame` because only one of the
    /// two frame paths goes through make_frame. Handing the source's own channel count to
    /// `import_d3d_texture` produced a frame whose audio was read as 16-channel interleaved:
    /// measured, a mono 1 kHz sine came back with no peak at 1 kHz at all and energy smeared
    /// across every channel, because sample n of a mono source was being read as channel n of
    /// sixteen. Doing it once, here, means both paths carry the same thing.
    std::vector<int32_t> take_audio_samples()
    {
        const auto channels = audio_channels_.load();
        if (audio_sink_ == nullptr || channels <= 0)
            return {};

        const auto cadence  = format_desc_.audio_cadence[cadence_counter_++ % format_desc_.audio_cadence.size()];
        const auto out_channels = format_desc_.audio_channels;
        const auto wanted   = static_cast<std::size_t>(cadence) * channels;

        std::vector<int32_t> source(wanted, 0);
        {
            std::lock_guard<std::mutex> lock(audio_mutex_);
            const auto                  available = std::min(wanted, audio_samples_.size());
            std::copy_n(audio_samples_.begin(), available, source.begin());
            audio_samples_.erase(audio_samples_.begin(), audio_samples_.begin() + available);

            if (available < wanted) {
                ++audio_underruns_;
                graph_->set_tag(diagnostics::tag_severity::WARNING, "audio-underrun");
            }
        }

        if (channels == out_channels)
            return source;

        std::vector<int32_t> widened(static_cast<std::size_t>(cadence) * out_channels, 0);
        const auto           copy_channels = std::min(channels, out_channels);
        for (int frame = 0; frame < cadence; ++frame) {
            for (int ch = 0; ch < copy_channels; ++ch) {
                widened[static_cast<std::size_t>(frame) * out_channels + ch] =
                    source[static_cast<std::size_t>(frame) * channels + ch];
            }
        }

        return widened;
    }

    /// An AVFrame view over samples the caller owns, for the host path. The samples are
    /// already at the channel's width, so make_frame's own mapping becomes the identity. The
    /// vector must outlive the returned frame; `ffmpeg::make_frame` copies out of it before
    /// returning, so in practice that means the same statement.
    std::shared_ptr<AVFrame> as_av_audio(std::vector<int32_t>& samples)
    {
        if (samples.empty())
            return nullptr;

        auto frame = ffmpeg::alloc_frame();
        av_channel_layout_default(&frame->ch_layout, format_desc_.audio_channels);
        frame->format      = AV_SAMPLE_FMT_S32;
        frame->sample_rate = format_desc_.audio_sample_rate;
        frame->nb_samples  = static_cast<int>(samples.size() / format_desc_.audio_channels);
        frame->data[0]     = reinterpret_cast<uint8_t*>(samples.data());

        return frame;
    }

    /// Tear the pipeline down and build it again. Called only from the video thread, which
    /// is also the one that reads the bus, so there is exactly one restarter.
    void restart()
    {
        auto delay = reconnect_delay_initial * (1 << std::min<uint64_t>(restarts_.load(), 3));
        delay      = std::min(std::chrono::duration_cast<std::chrono::milliseconds>(delay),
                              std::chrono::duration_cast<std::chrono::milliseconds>(reconnect_delay_max));

        CASPAR_LOG(warning) << print() << L" restarting in " << delay.count() << L" ms (attempt "
                            << (restarts_.load() + 1) << L").";

#ifdef _WIN32
        // The ring holds textures created on GStreamer's D3D11 device, and a rebuilt pipeline
        // may come up on a different one. Copying into a texture that belongs to another
        // device is not a slow path, it is undefined behaviour, so the bridge goes with the
        // pipeline it was built for.
        gpu_bridge_.reset();
#endif

        {
            std::unique_lock<std::shared_mutex> lock(pipeline_mutex_);
            destroy_pipeline();
        }

        // Outside the lock: the pull threads must be able to time out and notice the sinks are
        // gone rather than block on a rebuild that is deliberately slow.
        std::this_thread::sleep_for(delay);
        if (!is_running_)
            return;

        try {
            std::unique_lock<std::shared_mutex> lock(pipeline_mutex_);
            build_pipeline();
            if (gst_element_set_state(pipeline_, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
                destroy_pipeline();
                CASPAR_LOG(warning) << print() << L" restart failed to start the pipeline.";
                return;
            }
        } catch (...) {
            CASPAR_LOG(warning) << print() << L" restart could not build the pipeline.";
            return;
        }

#ifdef _WIN32
        if (want_gpu_)
            gpu_bridge_ = std::make_unique<d3d11_bridge>();
#endif

        ++restarts_;
        is_failed_ = false;
        CASPAR_LOG(info) << print() << L" restarted.";
    }

    void run_video()
    {
        while (is_running_) {
            poll_bus();

            if (is_failed_) {
                restart();
                continue;
            }

            // 100 ms, so a stalled pipeline still lets the loop see is_running_ and the bus.
            GstSample* sample = nullptr;
            {
                std::shared_lock<std::shared_mutex> lock(pipeline_mutex_);
                if (video_sink_ == nullptr)
                    continue;
                sample = gst_app_sink_try_pull_sample(GST_APP_SINK(video_sink_), 100 * GST_MSECOND);
            }

            if (sample == nullptr)
                continue;

            try {
                frame_timer_.restart();

                core::draw_frame frame;

                // Drawn once, before either path, so the cadence advances exactly once per
                // picture no matter which route the picture takes.
                auto audio_samples = take_audio_samples();

#ifdef _WIN32
                if (gpu_bridge_ && d3d11_bridge::handles(sample)) {
                    if (auto texture = gpu_bridge_->import(sample)) {
                        try {
                            frame = core::draw_frame(
                                frame_factory_->import_shared_texture(this,
                                                                      texture.handle,
                                                                      texture.width,
                                                                      texture.height,
                                                                      core::pixel_format::bgra,
                                                                      common::bit_depth::bit8,
                                                                      caspar::array<std::int32_t>(audio_samples)));
                            ++frames_on_gpu_;
                        } catch (...) {
                            // The Vulkan mixer throws here by design — it has no D3D11 import.
                            // One attempt, one message, then the host path for good.
                            CASPAR_LOG(info)
                                << print()
                                << L" the mixer would not take a D3D11 texture; using host memory instead.";
                            CASPAR_LOG_CURRENT_EXCEPTION();
                            gpu_bridge_.reset();
                        }
                    }
                }
#endif

                if (!frame)
                    frame = make_frame(this, *frame_factory_, sample, as_av_audio(audio_samples));

                if (frame) {
                    std::lock_guard<std::mutex> lock(frames_mutex_);
                    frames_.push(std::move(frame));
                    while (frames_.size() > max_queued_frames) {
                        frames_.pop();
                        ++frames_dropped_;
                        graph_->set_tag(diagnostics::tag_severity::WARNING, "dropped-frame");
                    }
                    ++frames_received_;
                }

                graph_->set_value("frame-time", frame_timer_.elapsed() * format_desc_.fps * 0.5);
            } catch (...) {
                CASPAR_LOG_CURRENT_EXCEPTION();
            }

            gst_sample_unref(sample);
        }
    }

    void run_audio()
    {
        // A second of audio at 16 channels. Past that the picture is not keeping up, and old
        // audio is worth less than bounded memory.
        const std::size_t max_samples = static_cast<std::size_t>(format_desc_.audio_sample_rate) * 16;

        while (is_running_) {
            GstSample* sample = nullptr;
            {
                std::shared_lock<std::shared_mutex> lock(pipeline_mutex_);
                if (audio_sink_ == nullptr) {
                    lock.unlock();
                    std::this_thread::sleep_for(std::chrono::milliseconds(50));
                    continue;
                }
                sample = gst_app_sink_try_pull_sample(GST_APP_SINK(audio_sink_), 100 * GST_MSECOND);
            }

            if (sample == nullptr)
                continue;

            auto* caps   = gst_sample_get_caps(sample);
            auto* buffer = gst_sample_get_buffer(sample);

            gint channels = 0;
            if (caps != nullptr && gst_caps_get_size(caps) > 0)
                gst_structure_get_int(gst_caps_get_structure(caps, 0), "channels", &channels);

            if (buffer != nullptr && channels > 0) {
                GstMapInfo map;
                if (gst_buffer_map(buffer, &map, GST_MAP_READ)) {
                    const auto* data  = reinterpret_cast<const int32_t*>(map.data);
                    const auto  count = map.size / sizeof(int32_t);

                    audio_channels_ = channels;
                    {
                        std::lock_guard<std::mutex> lock(audio_mutex_);
                        audio_samples_.insert(audio_samples_.end(), data, data + count);
                        if (audio_samples_.size() > max_samples) {
                            audio_samples_.erase(audio_samples_.begin(),
                                                 audio_samples_.begin() + (audio_samples_.size() - max_samples));
                        }
                    }
                    gst_buffer_unmap(buffer, &map);
                }
            }

            gst_sample_unref(sample);
        }
    }

    // ---------------------------------------------------------------- transport

    /// Frames, from GStreamer's nanoseconds. Returns 0 when the pipeline cannot answer —
    /// a live source has no duration, and reporting one would be worse than reporting none.
    uint32_t query_duration_frames() const
    {
        std::shared_lock<std::shared_mutex> lock(pipeline_mutex_);
        if (pipeline_ == nullptr)
            return 0;

        gint64 duration_ns = 0;
        if (!gst_element_query_duration(pipeline_, GST_FORMAT_TIME, &duration_ns) || duration_ns <= 0)
            return 0;

        return static_cast<uint32_t>(gst_util_uint64_scale(static_cast<guint64>(duration_ns),
                                                           format_desc_.framerate.numerator(),
                                                           format_desc_.framerate.denominator() * GST_SECOND));
    }

    uint32_t query_position_frames() const
    {
        std::shared_lock<std::shared_mutex> lock(pipeline_mutex_);
        if (pipeline_ == nullptr)
            return 0;

        gint64 position_ns = 0;
        if (!gst_element_query_position(pipeline_, GST_FORMAT_TIME, &position_ns) || position_ns < 0)
            return 0;

        return static_cast<uint32_t>(gst_util_uint64_scale(static_cast<guint64>(position_ns),
                                                           format_desc_.framerate.numerator(),
                                                           format_desc_.framerate.denominator() * GST_SECOND));
    }

    /// A flushing, accurate seek.
    ///
    /// FLUSH so the queued frames go with it — without it the channel keeps showing what was
    /// already buffered for up to four frames, which reads as the seek having been ignored.
    ///
    /// ACCURATE rather than KEY_UNIT because an operator who asks for frame 300 means frame
    /// 300. Measured with KEY_UNIT on a clip with keyframes every 250: SEEK 300 landed at 250
    /// and the position read back 324 after a second and a half of play — plausible enough to
    /// be believed and wrong by two seconds. The cost is a slower seek on a long GOP, which is
    /// the right trade for a command whose whole purpose is to land somewhere exact.
    bool seek_to_frame(uint32_t frame)
    {
        const auto position_ns = gst_util_uint64_scale(frame,
                                                       GST_SECOND * format_desc_.framerate.denominator(),
                                                       format_desc_.framerate.numerator());

        bool ok = false;
        {
            std::shared_lock<std::shared_mutex> lock(pipeline_mutex_);
            if (pipeline_ == nullptr)
                return false;

            ok = gst_element_seek_simple(pipeline_,
                                         GST_FORMAT_TIME,
                                         static_cast<GstSeekFlags>(GST_SEEK_FLAG_FLUSH | GST_SEEK_FLAG_ACCURATE),
                                         static_cast<gint64>(position_ns));
        }

        if (ok) {
            std::lock_guard<std::mutex> lock(frames_mutex_);
            std::queue<core::draw_frame>().swap(frames_);
        }

        return ok;
    }

    bool set_paused(bool paused)
    {
        std::shared_lock<std::shared_mutex> lock(pipeline_mutex_);
        if (pipeline_ == nullptr)
            return false;

        const auto target = paused ? GST_STATE_PAUSED : GST_STATE_PLAYING;
        return gst_element_set_state(pipeline_, target) != GST_STATE_CHANGE_FAILURE;
    }

    // frame_producer

    /// PAUSE, RESUME, SEEK <frame>, LENGTH, POSITION. Deliberately a subset of what the
    /// FFmpeg producer answers: LOOP, IN and OUT are properties of a *file* producer that
    /// owns its own reader, and this one owns a pipeline whose source may be a socket. A
    /// command that cannot mean anything here is refused rather than silently accepted.
    std::future<std::wstring> call(const std::vector<std::wstring>& params) override
    {
        if (params.empty())
            CASPAR_THROW_EXCEPTION(user_error() << msg_info("CALL needs a command."));

        const auto  cmd   = params.at(0);
        const auto  value = params.size() > 1 ? params.at(1) : std::wstring();
        std::wstring result;

        if (boost::iequals(cmd, L"pause")) {
            result = set_paused(true) ? L"true" : L"false";
        } else if (boost::iequals(cmd, L"resume") || boost::iequals(cmd, L"play")) {
            result = set_paused(false) ? L"true" : L"false";
        } else if (boost::iequals(cmd, L"seek")) {
            if (value.empty())
                CASPAR_THROW_EXCEPTION(user_error() << msg_info("SEEK needs a frame number."));
            result = seek_to_frame(boost::lexical_cast<uint32_t>(value)) ? L"true" : L"false";
        } else if (boost::iequals(cmd, L"length")) {
            result = std::to_wstring(query_duration_frames());
        } else if (boost::iequals(cmd, L"position") || boost::iequals(cmd, L"time")) {
            result = std::to_wstring(query_position_frames());
        } else {
            CASPAR_THROW_EXCEPTION(user_error() << msg_info(
                                       u8(L"The GStreamer producer does not answer '" + cmd +
                                          L"'. It takes PAUSE, RESUME, SEEK <frame>, LENGTH and POSITION.")));
        }

        return make_ready_future(std::move(result));
    }

    uint32_t nb_frames() const override
    {
        const auto frames = query_duration_frames();
        // The base class's answer for "unbounded", which is the honest one for a live source.
        return frames > 0 ? frames : std::numeric_limits<uint32_t>::max();
    }

    core::draw_frame receive_impl(const core::video_field field, int nb_samples) override
    {
        graph_->set_value("tick-time", tick_timer_.elapsed() * format_desc_.fps * 0.5);
        tick_timer_.restart();

        std::lock_guard<std::mutex> lock(frames_mutex_);

        queue_peak_ = std::max(queue_peak_.load(), static_cast<uint64_t>(frames_.size()));

        if (!frames_.empty()) {
            last_frame_ = frames_.front();
            frames_.pop();
        } else if (last_frame_) {
            // Only once a picture has been shown: the ticks before the first frame arrives are
            // startup, not starvation, and counting them would make every producer look bad
            // for the first second of its life.
            ++frames_starved_;
        }

        return last_frame_;
    }

    bool is_ready() override
    {
        std::lock_guard<std::mutex> lock(frames_mutex_);
        return !frames_.empty() || static_cast<bool>(last_frame_);
    }

    core::draw_frame last_frame(const core::video_field field) override
    {
        if (!last_frame_) {
            last_frame_ = receive_impl(field, 0);
        }
        return core::draw_frame::still(last_frame_);
    }

    std::wstring print() const override
    {
        return L"gst[" + boost::lexical_cast<std::wstring>(instance_no_) + L"|" + description_ + L"]";
    }

    std::wstring name() const override { return L"gstreamer"; }

    core::monitor::state state() const override
    {
        core::monitor::state state;
        state["gstreamer/pipeline"]  = u8(description_);
        state["gstreamer/received"]  = static_cast<int64_t>(frames_received_.load());
        state["gstreamer/dropped"]   = static_cast<int64_t>(frames_dropped_.load());
        state["gstreamer/eos"]       = is_eos_.load();
        state["gstreamer/audio"]     = audio_channels_.load();
        state["gstreamer/underruns"] = static_cast<int64_t>(audio_underruns_.load());
        state["gstreamer/gpu-frames"] = static_cast<int64_t>(frames_on_gpu_.load());
        state["gstreamer/restarts"]   = static_cast<int64_t>(restarts_.load());
        state["gstreamer/starved"]    = static_cast<int64_t>(frames_starved_.load());
        state["gstreamer/queue-peak"] = static_cast<int64_t>(queue_peak_.load());
        state["gstreamer/failed"]     = is_failed_.load();
        state["gstreamer/position"]   = static_cast<int64_t>(query_position_frames());
        state["gstreamer/length"]     = static_cast<int64_t>(query_duration_frames());
        {
            std::lock_guard<std::mutex> lock(frames_mutex_);
            state["gstreamer/queue"] = static_cast<int64_t>(frames_.size());
        }
        return state;
    }
};

std::atomic<int> gst_producer::instances_(0);

spl::shared_ptr<core::frame_producer> create_producer(const core::frame_producer_dependencies& dependencies,
                                                      const std::vector<std::wstring>&         params)
{
    if (params.empty())
        return core::frame_producer::empty();

    std::wstring description;

    if (boost::iequals(params.at(0), L"[GSTREAMER]")) {
        if (params.size() < 2)
            CASPAR_THROW_EXCEPTION(user_error() << msg_info("[GSTREAMER] needs a pipeline description."));

        description = params.at(1);
    } else if (boost::algorithm::istarts_with(params.at(0), L"gst://")) {
        description = params.at(0).substr(6);
    } else {
        return core::frame_producer::empty();
    }

    boost::trim(description);
    if (description.empty())
        CASPAR_THROW_EXCEPTION(user_error() << msg_info("The GStreamer pipeline description is empty."));

    runtime::ensure_initialized();

    // Opt-in rather than automatic. The GPU path changes which elements the pipeline has to
    // negotiate with, and a source that cannot reach D3D11 memory pays an upload for nothing —
    // so it is asked for, per PLAY, by a caller who knows the source is decoded on the GPU.
    const bool want_gpu = contains_param(L"GPU", params);

    return spl::make_shared<gst_producer>(
        dependencies.frame_factory, dependencies.format_desc, description, want_gpu);
}

}} // namespace caspar::gstreamer
