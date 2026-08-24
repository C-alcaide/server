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

#include "gst_consumer.h"

#include "../util/gst_runtime.h"

#include <core/consumer/channel_info.h>
#include <core/consumer/frame_consumer.h>
#include <core/frame/frame.h>
#include <core/frame/pixel_format.h>
#include <core/monitor/monitor.h>
#include <core/video_format.h>

#include <common/diagnostics/graph.h>
#include <common/except.h>
#include <common/future.h>
#include <common/log.h>
#include <common/param.h>
#include <common/timer.h>
#include <common/utf.h>

#include <boost/algorithm/string.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/lexical_cast.hpp>

#include <gst/app/gstappsrc.h>
#include <gst/video/video.h>
#include <gst/gst.h>

#include <algorithm>
#include <atomic>
#include <mutex>

namespace caspar { namespace gstreamer {

namespace {

/// The names the consumer looks for after parsing, and writes into the description when it
/// prepends the source itself. The mirror of the producer's sink names, deliberately: an
/// operator who has learned one has learned the other.
constexpr const char* video_src_name = "caspar_video";
constexpr const char* audio_src_name = "caspar_audio";

/// What gets prepended when the description names no source of its own. `is-live=true` and
/// `format=time` because the frames arrive at the channel's rate carrying real timestamps,
/// and an appsrc left on its defaults would let a muxer treat them as a file being written as
/// fast as it can be read.
constexpr const char* video_head = "appsrc name=caspar_video is-live=true format=time ! ";

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

struct gst_consumer : public core::frame_consumer
{
    const std::wstring description_;
    int                channel_index_ = -1;

    core::video_format_desc format_desc_;

    GstElement* pipeline_  = nullptr;
    GstElement* video_src_ = nullptr;
    GstElement* audio_src_ = nullptr;

    spl::shared_ptr<diagnostics::graph> graph_;
    timer                               frame_timer_;

    std::atomic<uint64_t> frames_sent_{0};
    std::atomic<uint64_t> captions_sent_{0};
    std::atomic<uint64_t> frames_dropped_{0};
    std::atomic<bool>     is_failed_{false};
    mutable std::mutex    mutex_;

  public:
    explicit gst_consumer(std::wstring description)
        : description_(std::move(description))
    {
        graph_->set_color("frame-time", diagnostics::color(0.5f, 1.0f, 0.2f));
        graph_->set_color("dropped-frame", diagnostics::color(0.3f, 0.6f, 0.3f));
        // How full appsrc's own queue is, normalised to the limit it was configured with.
        // Every other consumer in the tree graphs its input buffer under this name, and it is
        // what distinguishes "the encoder is keeping up" from "it is about to start dropping".
        graph_->set_color("input", diagnostics::color(0.7f, 0.4f, 0.4f));
    }

    ~gst_consumer()
    {
        std::lock_guard<std::mutex> lock(mutex_);

        if (pipeline_ != nullptr) {
            // End of stream first, and then wait for it to come back round the bus. A muxer
            // writing a file finalises on EOS; going straight to NULL leaves a container with
            // no index, which plays in some tools and not others — the worst kind of broken.
            if (video_src_ != nullptr)
                gst_app_src_end_of_stream(GST_APP_SRC(video_src_));
            if (audio_src_ != nullptr)
                gst_app_src_end_of_stream(GST_APP_SRC(audio_src_));

            if (auto* bus = gst_element_get_bus(pipeline_)) {
                auto* message = gst_bus_timed_pop_filtered(
                    bus, 5 * GST_SECOND, static_cast<GstMessageType>(GST_MESSAGE_EOS | GST_MESSAGE_ERROR));
                if (message == nullptr)
                    CASPAR_LOG(warning) << print() << L" no end-of-stream within 5 s; the output may be unfinished.";
                else
                    gst_message_unref(message);
                gst_object_unref(bus);
            }
        }

        destroy_pipeline();
        CASPAR_LOG(info) << print() << L" Uninitialized.";
    }

    void initialize(const core::video_format_desc& format_desc,
                    const core::channel_info&      channel_info,
                    int                            port_index) override
    {
        std::lock_guard<std::mutex> lock(mutex_);

        format_desc_   = format_desc;
        channel_index_ = channel_info.index;

        runtime::ensure_initialized();
        build_pipeline();

        if (gst_element_set_state(pipeline_, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
            const auto reason = drain_bus_error();
            destroy_pipeline();
            CASPAR_LOG(error) << print() << L" Failed to start: " << reason;
            CASPAR_THROW_EXCEPTION(user_error()
                                   << msg_info(u8(L"Failed to start GStreamer consumer pipeline: " + reason)));
        }

        graph_->set_text(print());
        diagnostics::register_graph(graph_);

        CASPAR_LOG(info) << print() << L" Initialized.";
    }

    void build_pipeline()
    {
        auto launch = u8(description_);

        if (launch.find(video_src_name) == std::string::npos)
            launch = std::string(video_head) + launch;

        GError* error = nullptr;
        pipeline_     = gst_parse_launch(launch.c_str(), &error);

        if (pipeline_ == nullptr) {
            const std::wstring reason = error ? u16(std::string(error->message)) : L"unknown error";
            if (error != nullptr)
                g_error_free(error);
            CASPAR_LOG(error) << L"[gstreamer] Failed to parse consumer pipeline: " << reason;
            CASPAR_THROW_EXCEPTION(user_error()
                                   << msg_info(u8(L"Failed to parse GStreamer pipeline: " + reason)));
        }
        if (error != nullptr) {
            // A partial pipeline: an element that could not be created, or a link dropped.
            // Same reasoning as the producer — it runs, writes nothing and says nothing.
            const auto reason = u16(std::string(error->message));
            g_error_free(error);
            destroy_pipeline();
            CASPAR_THROW_EXCEPTION(user_error() << msg_info(u8(L"Incomplete GStreamer pipeline: " + reason)));
        }

        video_src_ = gst_bin_get_by_name(GST_BIN(pipeline_), video_src_name);
        audio_src_ = gst_bin_get_by_name(GST_BIN(pipeline_), audio_src_name);

        if (video_src_ == nullptr) {
            destroy_pipeline();
            CASPAR_THROW_EXCEPTION(user_error() << msg_info(
                                       "The GStreamer description names no appsrc called caspar_video. Either start "
                                       "the description after the source and let it be prepended, or include "
                                       "'appsrc name=caspar_video' yourself."));
        }

        configure_video_src();
        if (audio_src_ != nullptr)
            configure_audio_src();
    }

    void configure_video_src()
    {
        // BGRA because that is what a channel frame carries; converting here rather than in
        // the pipeline would be this module deciding something the operator wrote a pipeline
        // to decide.
        const auto caps_text = std::string("video/x-raw, format=(string)BGRA, width=(int)") +
                               std::to_string(format_desc_.width) + ", height=(int)" +
                               std::to_string(format_desc_.height) + ", framerate=(fraction)" +
                               std::to_string(format_desc_.framerate.numerator()) + "/" +
                               std::to_string(format_desc_.framerate.denominator());

        auto* caps = gst_caps_from_string(caps_text.c_str());
        g_object_set(G_OBJECT(video_src_),
                     "caps",
                     caps,
                     "is-live",
                     TRUE,
                     "format",
                     GST_FORMAT_TIME,
                     // Never block the channel. A consumer that stalls its pipeline would
                     // stall playout, which is a far worse failure than a dropped frame on an
                     // output that cannot keep up — and the drop is counted rather than
                     // hidden.
                     "block",
                     FALSE,
                     "max-bytes",
                     static_cast<guint64>(format_desc_.size) * 4,
                     nullptr);
        gst_caps_unref(caps);
    }

    void configure_audio_src()
    {
        const auto caps_text = std::string("audio/x-raw, format=(string)S32LE, layout=(string)interleaved, "
                                           "rate=(int)") +
                               std::to_string(format_desc_.audio_sample_rate) + ", channels=(int)" +
                               std::to_string(format_desc_.audio_channels);

        auto* caps = gst_caps_from_string(caps_text.c_str());
        g_object_set(G_OBJECT(audio_src_),
                     "caps", caps,
                     "is-live", TRUE,
                     "format", GST_FORMAT_TIME,
                     "block", FALSE,
                     nullptr);
        gst_caps_unref(caps);
    }

    void destroy_pipeline()
    {
        if (video_src_ != nullptr) {
            gst_object_unref(video_src_);
            video_src_ = nullptr;
        }
        if (audio_src_ != nullptr) {
            gst_object_unref(audio_src_);
            audio_src_ = nullptr;
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

    /// How full appsrc's queue is, as a fraction of the limit it was configured with.
    ///
    /// `max-bytes` is set to one frame's BGRA size times four, so this reads 0.25 per frame
    /// still in flight. A consumer whose encoder is keeping up sits near zero; one about to
    /// start dropping climbs first, which is the whole point of plotting it rather than only
    /// counting the drops after they happen.
    double appsrc_fill() const
    {
        if (video_src_ == nullptr)
            return 0.0;
        const auto limit = static_cast<guint64>(format_desc_.size) * 4;
        if (limit == 0)
            return 0.0;
        const auto level = gst_app_src_get_current_level_bytes(GST_APP_SRC(video_src_));
        return std::min(1.0, static_cast<double>(level) / static_cast<double>(limit));
    }

    void poll_bus()
    {
        auto* bus = gst_element_get_bus(pipeline_);
        if (bus == nullptr)
            return;

        while (auto* message = gst_bus_pop_filtered(
                   bus, static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_WARNING))) {
            if (GST_MESSAGE_TYPE(message) == GST_MESSAGE_ERROR) {
                CASPAR_LOG(error) << print() << L" " << describe_message(message);
                is_failed_ = true;
            } else {
                CASPAR_LOG(warning) << print() << L" " << describe_message(message);
            }
            gst_message_unref(message);
        }

        gst_object_unref(bus);
    }

    /// Wraps the frame's own pixels — no copy. The `const_frame` is captured by the buffer's
    /// destroy notify, so the memory outlives the push for exactly as long as GStreamer holds
    /// it and not one frame longer.
    GstBuffer* wrap(const core::const_frame& frame)
    {
        auto* held = new core::const_frame(frame);
        auto  data = frame.image_data(0);

        return gst_buffer_new_wrapped_full(GST_MEMORY_FLAG_READONLY,
                                           const_cast<uint8_t*>(data.data()),
                                           data.size(),
                                           0,
                                           data.size(),
                                           held,
                                           [](gpointer p) { delete static_cast<core::const_frame*>(p); });
    }

    // frame_consumer

    std::future<bool> send(const core::video_field field, core::const_frame frame) override
    {
        std::lock_guard<std::mutex> lock(mutex_);

        if (pipeline_ == nullptr || is_failed_)
            return make_ready_future(false);

        poll_bus();
        frame_timer_.restart();

        if (frame.pixel_format_desc().format != core::pixel_format::bgra) {
            CASPAR_LOG(warning) << print() << L" received a frame that is not BGRA; dropping it.";
            ++frames_dropped_;
            return make_ready_future(true);
        }

        const auto index = frames_sent_.load();
        auto*      buffer = wrap(frame);

        // Captions the source attached, put back on the way out. Pass-through: the bytes are
        // the ones that arrived, because every decode-and-re-encode step is a chance to change
        // what a broadcaster is obliged to preserve.
        //
        // This is what makes the channel a caption-transparent path rather than a place
        // captions go to die -- `h264ccinserter`, `cccombiner` or a mux downstream can now put
        // them back into a stream.
        for (const auto& cc : frame.metadata().captions) {
            if (cc.data.empty())
                continue;
            gst_buffer_add_video_caption_meta(buffer,
                                              static_cast<GstVideoCaptionType>(cc.format),
                                              cc.data.data(),
                                              cc.data.size());
            ++captions_sent_;
        }

        GST_BUFFER_PTS(buffer)      = gst_util_uint64_scale(index, GST_SECOND * format_desc_.framerate.denominator(),
                                                            format_desc_.framerate.numerator());
        GST_BUFFER_DURATION(buffer) = gst_util_uint64_scale(1, GST_SECOND * format_desc_.framerate.denominator(),
                                                            format_desc_.framerate.numerator());

        if (gst_app_src_push_buffer(GST_APP_SRC(video_src_), buffer) != GST_FLOW_OK) {
            ++frames_dropped_;
            graph_->set_tag(diagnostics::tag_severity::WARNING, "dropped-frame");
        } else {
            ++frames_sent_;
        }

        if (audio_src_ != nullptr)
            push_audio(frame, index);

        graph_->set_value("frame-time", frame_timer_.elapsed() * format_desc_.fps * 0.5);
        graph_->set_value("input", appsrc_fill());

        return make_ready_future(true);
    }

    void push_audio(const core::const_frame& frame, uint64_t index)
    {
        const auto& samples = frame.audio_data();
        if (samples.size() == 0)
            return;

        const auto bytes  = samples.size() * sizeof(int32_t);
        auto*      held   = new core::const_frame(frame);
        auto*      buffer = gst_buffer_new_wrapped_full(GST_MEMORY_FLAG_READONLY,
                                                        const_cast<int32_t*>(samples.data()),
                                                        bytes,
                                                        0,
                                                        bytes,
                                                        held,
                                                        [](gpointer p) { delete static_cast<core::const_frame*>(p); });

        GST_BUFFER_PTS(buffer) = gst_util_uint64_scale(index, GST_SECOND * format_desc_.framerate.denominator(),
                                                       format_desc_.framerate.numerator());

        if (gst_app_src_push_buffer(GST_APP_SRC(audio_src_), buffer) != GST_FLOW_OK)
            ++frames_dropped_;
    }

    std::wstring print() const override
    {
        return L"gst-out[" + boost::lexical_cast<std::wstring>(channel_index_) + L"|" + description_ + L"]";
    }

    std::wstring name() const override { return L"gstreamer"; }

    bool has_synchronization_clock() const override { return false; }

    /// Constant, and that is not laziness: output::add keys its map on index() **before**
    /// calling initialize(), so an index derived from anything initialize() sets is registered
    /// under one number and reported under another, and REMOVE by index never finds it. The
    /// map is per-channel, so a constant is unique where it needs to be. 110000 keeps this
    /// clear of the FFmpeg consumer's 100000 block.
    int index() const override { return 110000; }

    core::monitor::state state() const override
    {
        core::monitor::state state;
        state["gstreamer/pipeline"] = u8(description_);
        state["gstreamer/sent"]     = static_cast<int64_t>(frames_sent_.load());
        // Not decoration: a caption path that silently carries nothing looks exactly like one
        // that is working, because the picture is identical either way.
        state["gstreamer/captions"] = static_cast<int64_t>(captions_sent_.load());
        state["gstreamer/dropped"]  = static_cast<int64_t>(frames_dropped_.load());
        state["gstreamer/failed"]   = is_failed_.load();
        return state;
    }
};

spl::shared_ptr<core::frame_consumer>
create_consumer(const std::vector<std::wstring>&                        params,
                const core::video_format_repository&                    format_repository,
                const std::vector<spl::shared_ptr<core::video_channel>>& channels,
                const core::channel_info&                               channel_info)
{
    if (params.size() < 2 || !boost::iequals(params.at(0), L"GSTREAMER"))
        return core::frame_consumer::empty();

    auto description = params.at(1);
    boost::trim(description);

    if (description.empty())
        CASPAR_THROW_EXCEPTION(user_error() << msg_info("The GStreamer pipeline description is empty."));

    return spl::make_shared<gst_consumer>(std::move(description));
}

spl::shared_ptr<core::frame_consumer>
create_preconfigured_consumer(const boost::property_tree::wptree&                      ptree,
                              const core::video_format_repository&                     format_repository,
                              const std::vector<spl::shared_ptr<core::video_channel>>& channels,
                              const core::channel_info&                                channel_info)
{
    auto description = ptree.get<std::wstring>(L"pipeline", L"");
    boost::trim(description);

    if (description.empty())
        CASPAR_THROW_EXCEPTION(user_error() << msg_info(
                                   "<gstreamer> consumer needs a <pipeline> element: the gst-launch "
                                   "description to send this channel through."));

    return spl::make_shared<gst_consumer>(std::move(description));
}

}} // namespace caspar::gstreamer
