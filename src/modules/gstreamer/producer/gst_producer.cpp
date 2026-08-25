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

// The GPU route is guarded rather than stubbed. Upstream's version of it is built on
// `frame_factory::import_shared_texture`, which this tree deliberately does not have (see the
// note at core/mixer/image/image_mixer.h), so it is not portable a line at a time — the
// mixer-facing half has to be rewritten against `d3d11_import_bridge` and `dx_interop`, and
// it should carry NV12 planes rather than converting to BGRA, because this tree has a
// semi-planar pixel format and upstream does not.
//
// A stub that silently fell back to host memory would be worse than nothing: a GPU path that
// never engages looks exactly like one that is switched off, which is how this went wrong
// twice while the upstream module was being built. Asking for GPU without the bridge compiled
// in refuses with a message instead.
#ifdef CASPAR_GST_GPU_BRIDGE
#include "gst_gpu_bridge.h"
#endif

#include <common/array.h>

#include <core/frame/draw_frame.h>
#include <core/frame/frame_factory.h>
#include <core/frame/pixel_format.h>
#include <core/monitor/monitor.h>
#include <core/video_format.h>

#include <common/bit_depth.h>

#include <chrono>
#include <future>
#include <thread>

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
#include <gst/video/video.h>

#include <atomic>
#include <chrono>
#include <limits>
#include <deque>
#include <cstring>
#include <map>
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
#ifdef CASPAR_GST_GPU_BRIDGE
/// **NV12, not BGRA**, which is what makes this tree's GPU route cheaper than upstream's: the
/// mixer has a semi-planar pixel format, so the decoder's own two planes go straight to the
/// shader and the mixer's colour management does the conversion. Asking for BGRA here would
/// insert a full-frame conversion pass AND move the YCbCr maths into `d3d11convert`, where it
/// has to agree with the mixer's by luck rather than by construction.
///
/// `d3d11convert` is still in the chain, and that is not a contradiction. It is a
/// GstBaseTransform: when the decoder already produces NV12 in D3D11 memory -- the case this
/// route exists for -- input and output caps are identical and it passes buffers through
/// untouched. When they are not, it is the difference between a pipeline that works and one
/// that refuses to link, which is what a software `decodebin` feeding `d3d11upload` does:
/// upload changes the memory type, never the format.
/// **Both depths, and P010 first.** Pinning this to NV12 silently threw away 10-bit: a
/// `d3d11h265dec` producing P010 would have `d3d11convert` narrow it to 8 bits to satisfy the
/// filter -- or, as measured, fail to reach D3D11 memory at all, so the GPU route never engaged
/// and the producer fell back to host transfer with no reason logged anywhere. Listing both
/// lets each source keep the depth it decoded at, and the bridge reads the surface's real
/// format per sample rather than trusting this string.
///
/// No spaces inside the braces: `gst_parse_launch` splits the description on whitespace.
constexpr const char* gpu_tail =
    " ! d3d11upload ! d3d11convert ! video/x-raw(memory:D3D11Memory),format={P010_10LE,NV12} ! "
    "appsink name=";
#endif

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

    // The video format the appsink actually settled on, read off each sample's caps. Its own
    // mutex rather than pipeline_mutex_: it is written on the video thread and read on the
    // channel thread for `state()`, and neither should wait on a pipeline rebuild.
    mutable std::mutex format_mutex_;
    std::string        negotiated_format_;

    /// Whatever the pipeline's elements have reported about themselves, flattened. See
    /// `note_element_message`. A map rather than named fields because the whole point is that
    /// an element nobody anticipated still reports.
    mutable std::mutex                     element_mutex_;
    std::map<std::string, double>          element_state_;

    const bool want_gpu_;
#ifdef CASPAR_GST_GPU_BRIDGE
    std::unique_ptr<gst_gpu_bridge> gpu_bridge_;

    /// Set only when the pipeline's move to PLAYING did not finish in time -- i.e. a source is
    /// blocking inside its own state change. Its presence is what tells teardown it must not
    /// join. Null in every ordinary case.
    std::shared_ptr<std::future<GstStateChangeReturn>> starting_;

    /// Rate limit for `poll_element_stats`. Not atomic: it is only touched from `state()`,
    /// and a duplicated poll would be harmless anyway.
    std::chrono::steady_clock::time_point last_stats_poll_{};
#endif
    std::atomic<uint64_t> frames_on_gpu_{0};
    /// Caption packets lifted off incoming buffers. Symmetric with the consumer's `captions`,
    /// and the pair is what makes a caption path debuggable: in-but-not-out is a carry
    /// problem, neither is a pipeline that never attached any in the first place, and without
    /// both you cannot tell those apart -- the picture is identical either way.
    std::atomic<uint64_t> captions_in_{0};

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
        // How much runway is left between the pipeline and the channel, normalised to the
        // queue's depth. The single most useful line on a live source: it falls before the
        // producer starves, so a sender that is slipping shows here while the picture is still
        // perfect. `buffer` rather than `queue` to match the ffmpeg producer's line of the
        // same meaning -- two names for one quantity is worse than either name.
        graph_->set_color("buffer", diagnostics::color(1.0f, 1.0f, 0.0f));
        // Ticks that found the queue empty and repeated the last picture. Frame counts cannot
        // tell a healthy producer from a frozen one -- a source that stops leaves `received`
        // sitting still while the channel keeps ticking -- so this is the line that shows a
        // dead source, and it had no line at all despite the README calling it the one worth
        // knowing.
        graph_->set_color("starved", diagnostics::color(0.9f, 0.2f, 0.2f));
        // A pipeline rebuild. Tagged rather than plotted: it is an event, and on a live source
        // it is the event you want to be able to point at afterwards.
        graph_->set_color("restart", diagnostics::color(1.0f, 0.5f, 0.0f));
        diagnostics::register_graph(graph_);

        build_pipeline();

        // set_state returning ASYNC says only that the change was accepted. A pipeline that
        // cannot link — the commonest description mistake — fails afterwards, on the bus, and
        // reporting 202 for it puts the error in the log minutes after the operator stopped
        // looking. So wait for the change to settle, and treat the failure as a failed PLAY.
        //
        // **ON A WORKER THREAD, and that is not a refinement.** `gst_element_set_state` itself
        // blocks for a source that waits inside NULL->READY: `srtsrc mode=listener` sits in
        // accept() until a caller arrives, which is a completely ordinary thing to ask for --
        // you listen, and the encoder connects when it is ready. Called on the AMCP thread it
        // never returns, so the settle below was unreachable and the SERVER stopped answering
        // commands entirely: process alive, port still accepting, every PLAY, INFO and CLEAR
        // unanswered from then on. Measured 2026-08-25 with a one-line pipeline.
        auto* pipeline  = pipeline_;
        auto  start_job = std::make_shared<std::future<GstStateChangeReturn>>(
            std::async(std::launch::async, [pipeline] {
                auto change = gst_element_set_state(pipeline, GST_STATE_PLAYING);
                if (change == GST_STATE_CHANGE_ASYNC) {
                    GstState state = GST_STATE_NULL;
                    change = gst_element_get_state(pipeline, &state, nullptr, 5 * GST_SECOND);
                }
                return change;
            }));

        auto change = GST_STATE_CHANGE_SUCCESS;
        // 3 s, not the 5 this used to imply. The wait is paid by the AMCP client waiting for
        // its `202 PLAY OK`, and a 6 s version measured as a client-side timeout on a PLAY the
        // server had in fact accepted -- the same unresponsive-looking symptom, moved. Link
        // failures, which are what this wait is for, surface immediately.
        if (start_job->wait_for(std::chrono::seconds(3)) == std::future_status::ready) {
            change = start_job->get();
            if (change == GST_STATE_CHANGE_ASYNC) {
                // Still settling. A slow network source is a legitimate reason, so this is a
                // warning and the producer carries on.
                CASPAR_LOG(warning) << print() << L" Still preparing; continuing.";
                change = GST_STATE_CHANGE_SUCCESS;
            }
        } else {
            // Blocked in the state change rather than merely slow. The layer is live and will
            // deliver as soon as the source does; what must not happen is this thread waiting
            // for it. Kept so teardown can avoid joining a thread that may never return.
            starting_ = start_job;
            CASPAR_LOG(info) << print()
                             << L" still connecting; the layer is live and will show frames when "
                                L"the source does. A listening `srtsrc` blocks here until a "
                                L"caller arrives — pass `wait-for-connection=false` if you would "
                                L"rather it gave up.";
        }

        if (change == GST_STATE_CHANGE_FAILURE) {
            const auto reason = drain_bus_error();
            destroy_pipeline();
            CASPAR_LOG(error) << print() << L" Failed to start: " << reason;
            CASPAR_THROW_EXCEPTION(user_error() << msg_info(u8(L"Failed to start GStreamer pipeline: " + reason)));
        }

#ifdef CASPAR_GST_GPU_BRIDGE
        // Audio is no reason to refuse: the samples are taken once per picture and put on
        // whichever frame the route produces, so a pipeline with both stays on the GPU.
        if (want_gpu_)
            gpu_bridge_ = std::make_unique<gst_gpu_bridge>();
#else
        // Refuse out loud. A GPU path that silently falls back to host memory is
        // indistinguishable from one that is working, which is exactly how two defects
        // survived in the upstream module: the producer counted frames either way.
        if (want_gpu_)
            CASPAR_LOG(warning) << print()
                                << L" GPU was requested, but this build has no GStreamer GPU "
                                   L"bridge compiled in. Using host memory; the picture is "
                                   L"correct and the transfer is not.";
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

        if (launch.find(video_sink_name) == std::string::npos) {
#ifdef CASPAR_GST_GPU_BRIDGE
            const char* tail = want_gpu_ ? gpu_tail : host_tail;
#else
            // want_gpu_ is still honoured as far as it can be: the refusal was logged at
            // construction, and the host tail is what a pipeline without the bridge must get.
            const char* tail = host_tail;
#endif
            launch += std::string(tail) + video_sink_name;
        }

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
#ifdef CASPAR_GST_GPU_BRIDGE
        const auto gpu_caps =
            want_gpu_ ? std::string("video/x-raw(memory:D3D11Memory), format=(string){ P010_10LE, NV12 }") +
                            rate + "; "
                      : std::string("");
#else
        const std::string gpu_caps;
#endif
        const auto caps_text = gpu_caps +
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
            if (starting_ && starting_->valid() &&
                starting_->wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
                // The start is still blocked inside the source. Waiting for it here would hang
                // the CHANNEL as well as the layer, which is the fault this whole change exists
                // to remove -- so the pipeline is handed to a detached janitor and this returns
                // immediately.
                //
                // `set_state(NULL)` is what asks a blocked source to give up; it may itself
                // wait on the state lock the stuck transition holds, which is exactly why it is
                // not run here. If the source never unblocks the janitor never finishes and one
                // pipeline is leaked -- bounded, pathological, and a great deal better than an
                // unresponsive server.
                auto* orphan = pipeline_;
                auto  job    = starting_;
                std::thread([orphan, job] {
                    gst_element_set_state(orphan, GST_STATE_NULL);
                    job->wait();
                    gst_element_set_state(orphan, GST_STATE_NULL);
                    gst_object_unref(orphan);
                }).detach();
                pipeline_ = nullptr;
                starting_.reset();
                return;
            }
            gst_element_set_state(pipeline_, GST_STATE_NULL);
            gst_object_unref(pipeline_);
            pipeline_ = nullptr;
        }
        starting_.reset();
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
                   bus,
                   static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_WARNING | GST_MESSAGE_EOS |
                                               GST_MESSAGE_ELEMENT))) {
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
                case GST_MESSAGE_ELEMENT:
                    note_element_message(message);
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

#ifdef CASPAR_GST_GPU_BRIDGE
        // The bridge holds textures created on GStreamer's D3D11 device, and a rebuilt pipeline
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

#ifdef CASPAR_GST_GPU_BRIDGE
        // Rebuilt with the pipeline: the bridge holds textures belonging to the D3D11 device
        // of the elements that just went away.
        if (want_gpu_)
            gpu_bridge_ = std::make_unique<gst_gpu_bridge>();
#endif

        ++restarts_;
        graph_->set_tag(diagnostics::tag_severity::WARNING, "restart");
        is_failed_ = false;
        CASPAR_LOG(info) << print() << L" restarted.";
    }

    /// An element's own message, flattened into this producer's monitor state.
    ///
    /// **This is how a whole class of GStreamer element reports at all.** `level` and
    /// `ebur128level` publish loudness this way; `spectrum` its bands; `videoframe-audiolevel`,
    /// the ONVIF elements and the analytics overlays likewise. None of them touch the video
    /// they pass through, so before this the only way to see their output was the log -- which
    /// meant the module could carry them and tell you nothing.
    ///
    /// Everything lands under `gstreamer/element/<name>/<field>`, so it reaches INFO and OSC
    /// with no per-element code. That genericity is the point: an element nobody anticipated
    /// still reports, and adding support for a new one is a pipeline change rather than a
    /// server change.
    ///
    /// Only scalar fields are taken. A GstStructure can hold arrays -- `level` sends one peak
    /// per channel -- and those are reduced to their first value with the channel count
    /// alongside, because monitor state is a flat key/value space and a per-channel fan-out
    /// would be unbounded on a 16-channel bus.
    void note_element_message(GstMessage* message)
    {
        const auto* st = gst_message_get_structure(message);
        if (st == nullptr)
            return;

        const std::string name = gst_structure_get_name(st);
        // A handful of elements post high-rate structural messages that are not measurements
        // and would only add noise -- and one of them, `GstNavigation`, arrives per mouse move.
        if (name.rfind("Gst", 0) == 0 && name != "GstMultiFileSink")
            return;

        std::lock_guard<std::mutex> lock(element_mutex_);
        flatten_structure(st, name);
    }

    /// Flattens a GstStructure's scalar fields into `element_state_` under `prefix`.
    /// Caller holds `element_mutex_`.
    void flatten_structure(const GstStructure* st, const std::string& prefix)
    {
        const int fields = gst_structure_n_fields(st);
        for (int i = 0; i < fields; ++i) {
            const char*  field = gst_structure_nth_field_name(st, i);
            const GValue* v    = gst_structure_get_value(st, field);
            if (field == nullptr || v == nullptr)
                continue;

            const std::string key = prefix + "/" + field;
            if (G_VALUE_HOLDS_DOUBLE(v))
                element_state_[key] = g_value_get_double(v);
            else if (G_VALUE_HOLDS_INT(v))
                element_state_[key] = static_cast<double>(g_value_get_int(v));
            else if (G_VALUE_HOLDS_UINT(v))
                element_state_[key] = static_cast<double>(g_value_get_uint(v));
            else if (G_VALUE_HOLDS_INT64(v))
                element_state_[key] = static_cast<double>(g_value_get_int64(v));
            else if (G_VALUE_HOLDS_UINT64(v))
                element_state_[key] = static_cast<double>(g_value_get_uint64(v));
            else if (G_VALUE_HOLDS_BOOLEAN(v))
                element_state_[key] = g_value_get_boolean(v) ? 1.0 : 0.0;
            else {
                // `GST_VALUE_HOLDS_ARRAY`/`_LIST` expand to a comparison against a DATA symbol
                // (`_gst_value_array_type`), and a data import cannot be delay-loaded:
                // `LNK1194: cannot delay-load gstreamer-1.0-0.dll due to the import of data
                // symbol`. This module is delay-loaded on purpose, so that the server still
                // starts on a machine with no GStreamer -- which makes the type NAME, resolved
                // through a function, the only way to ask.
                const char* type_name = G_VALUE_TYPE_NAME(v);
                if (type_name == nullptr)
                    continue;
                const bool is_array = std::strcmp(type_name, "GstValueArray") == 0;
                const bool is_list  = std::strcmp(type_name, "GstValueList") == 0;
                if (!is_array && !is_list)
                    continue;

                const guint n_vals = is_array ? gst_value_array_get_size(v) : gst_value_list_get_size(v);
                if (n_vals == 0)
                    continue;
                const GValue* first = is_array ? gst_value_array_get_value(v, 0)
                                               : gst_value_list_get_value(v, 0);
                if (first != nullptr && G_VALUE_HOLDS_DOUBLE(first))
                    element_state_[key] = g_value_get_double(first);
                element_state_[key + "-count"] = static_cast<double>(n_vals);
            }
        }
    }

    /// Reads a `stats` property off every element that has one, once per state() call.
    ///
    /// **This is the other half of how GStreamer reports about itself**, and it is not the bus:
    /// `srtsrc` and `srtsink` publish RTT, bandwidth, packet loss and retransmission counts as
    /// a GstStructure on a property, and nothing is ever posted about them. Without this the
    /// module could carry an SRT link and say nothing at all about its health -- "the picture
    /// is breaking up" would stay an opinion when the transport has the number.
    ///
    /// Generic on purpose, like the bus handler: any element with a `stats` property reports,
    /// so `rtspsrc`, `rtpbin` and the RIST elements come along without another line of code.
    void poll_element_stats()
    {
        // **Twice a second, not fifty times.** `state()` is called by the monitor at frame
        // rate, so "once per state() call" meant a recursive bin walk plus a property read on
        // every element, 50 times a second, to publish numbers that -- as the comment above
        // says -- change far more slowly than that.
        //
        // It was not only waste. `srtsrc` logs a warning from inside the property read when
        // its socket has no connection, so an SRT source that is waiting, retrying, or
        // standing by as a `fallbacksrc` primary filled the server log at 50 lines a second:
        // measured 2026-08-25, several thousand identical `failed to retrieve stats for
        // socket ... Connection does not exist` lines, which is both the log unusable and
        // synchronous file I/O on the path that publishes channel state.
        {
            const auto now = std::chrono::steady_clock::now();
            if (now - last_stats_poll_ < std::chrono::milliseconds(500))
                return;
            last_stats_poll_ = now;
        }

        std::shared_lock<std::shared_mutex> lock(pipeline_mutex_);
        if (pipeline_ == nullptr)
            return;

        auto* it = gst_bin_iterate_recurse(GST_BIN(pipeline_));
        if (it == nullptr)
            return;

        GValue item = G_VALUE_INIT;
        while (gst_iterator_next(it, &item) == GST_ITERATOR_OK) {
            auto* element = static_cast<GstElement*>(g_value_get_object(&item));
            if (element != nullptr &&
                g_object_class_find_property(G_OBJECT_GET_CLASS(element), "stats") != nullptr) {
                GstStructure* stats = nullptr;
                g_object_get(G_OBJECT(element), "stats", &stats, nullptr);
                if (stats != nullptr) {
                    auto* name = gst_element_get_name(element);
                    {
                        std::lock_guard<std::mutex> guard(element_mutex_);
                        flatten_structure(stats, name ? std::string(name) : std::string("stats"));
                    }
                    g_free(name);
                    gst_structure_free(stats);
                }
            }
            g_value_unset(&item);
        }
        gst_iterator_free(it);
    }

    /// Records what the sink negotiated, including the memory type, so `state()` can answer
    /// "NV12 in D3D11 memory" rather than leaving a test to infer it from the pipeline text.
    /// Cheap enough to run per sample -- a caps pointer comparison in the common case -- and
    /// per-sample is the point: a renegotiation mid-stream is exactly the event that would
    /// otherwise go unnoticed.
    void note_negotiated_format(GstSample* sample)
    {
        auto* caps = gst_sample_get_caps(sample);
        if (caps == nullptr || gst_caps_get_size(caps) == 0)
            return;

        auto*       s      = gst_caps_get_structure(caps, 0);
        const char* format = gst_structure_get_string(s, "format");
        if (format == nullptr)
            return;

        std::string text(format);
        if (auto* features = gst_caps_get_features(caps, 0)) {
            if (gst_caps_features_contains(features, "memory:D3D11Memory"))
                text += " (D3D11)";
            else if (gst_caps_features_contains(features, "memory:CUDAMemory"))
                text += " (CUDA)";
        }

        std::lock_guard<std::mutex> lock(format_mutex_);
        if (negotiated_format_ != text)
            negotiated_format_ = std::move(text);
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

                note_negotiated_format(sample);

                core::draw_frame frame;

                // Drawn once, before either path, so the cadence advances exactly once per
                // picture no matter which route the picture takes.
                auto audio_samples = take_audio_samples();

#ifdef CASPAR_GST_GPU_BRIDGE
                if (gpu_bridge_ && gst_gpu_bridge::handles(sample)) {
                    frame = gpu_bridge_->to_frame(this, *frame_factory_, sample, audio_samples);
                    if (frame)
                        ++frames_on_gpu_;
                }
#endif

                if (!frame)
                    frame = make_frame(this, *frame_factory_, sample, as_av_audio(audio_samples));

                if (auto* buffer = gst_sample_get_buffer(sample)) {
                    gpointer st = nullptr;
                    while (gst_buffer_iterate_meta_filtered(buffer, &st, GST_VIDEO_CAPTION_META_API_TYPE))
                        ++captions_in_;
                }

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
        graph_->set_value("buffer", static_cast<double>(frames_.size()) /
                                        static_cast<double>(max_queued_frames));

        if (!frames_.empty()) {
            last_frame_ = frames_.front();
            frames_.pop();
        } else if (last_frame_) {
            // Only once a picture has been shown: the ticks before the first frame arrives are
            // startup, not starvation, and counting them would make every producer look bad
            // for the first second of its life.
            ++frames_starved_;
            graph_->set_tag(diagnostics::tag_severity::WARNING, "starved");
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
        state["gstreamer/captions-in"] = static_cast<int64_t>(captions_in_.load());
        // Which format the sink actually negotiated. Not decoration: the sink offers a list,
        // and `videoconvert` will happily satisfy it by converting -- so "the pipeline names
        // d3d11h264dec, therefore the mixer got NV12" is an assumption, not an observation.
        // Reporting it is what lets a test tell a working semi-planar path from a BGRA
        // conversion that produces the same picture at more cost.
        {
            std::lock_guard<std::mutex> lock(format_mutex_);
            state["gstreamer/format"] = negotiated_format_.empty() ? std::string("none") : negotiated_format_;
        }
        // Properties are pulled here rather than on the video thread: they are a read of live
        // element state, and doing it per frame would cost a recursive bin walk fifty times a
        // second to publish numbers that update far more slowly than that.
        const_cast<gst_producer*>(this)->poll_element_stats();
        {
            std::lock_guard<std::mutex> lock(element_mutex_);
            for (const auto& [key, value] : element_state_)
                state["gstreamer/element/" + key] = value;
        }
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
