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

#include <core/frame/draw_frame.h>
#include <core/frame/frame_factory.h>
#include <core/monitor/monitor.h>
#include <core/video_format.h>

#include <common/diagnostics/graph.h>
#include <common/except.h>
#include <common/log.h>
#include <common/param.h>
#include <common/timer.h>
#include <common/utf.h>

#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>

#include <gst/app/gstappsink.h>
#include <gst/gst.h>

#include <atomic>
#include <mutex>
#include <queue>
#include <thread>

namespace caspar { namespace gstreamer {

namespace {

/// How many decoded frames may sit between the pipeline and the channel. Deep enough to
/// absorb a decode that is briefly late, shallow enough that a producer switched away from
/// and back does not present stale pictures.
constexpr std::size_t max_queued_frames = 4;

std::wstring describe_error(GstMessage* message)
{
    GError* error = nullptr;
    gchar*  debug = nullptr;
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

    GstElement* pipeline_ = nullptr;
    GstElement* appsink_  = nullptr;

    spl::shared_ptr<diagnostics::graph> graph_;
    timer                               tick_timer_;
    timer                               frame_timer_;

    std::queue<core::draw_frame> frames_;
    mutable std::mutex           frames_mutex_;
    core::draw_frame             last_frame_;

    std::atomic<bool>     is_running_{true};
    std::atomic<bool>     is_eos_{false};
    std::atomic<uint64_t> frames_received_{0};
    std::atomic<uint64_t> frames_dropped_{0};
    std::thread           thread_;

  public:
    gst_producer(spl::shared_ptr<core::frame_factory> frame_factory,
                 core::video_format_desc              format_desc,
                 std::wstring                         description)
        : instance_no_(instances_++)
        , description_(std::move(description))
        , frame_factory_(std::move(frame_factory))
        , format_desc_(std::move(format_desc))
    {
        graph_->set_text(print());
        graph_->set_color("frame-time", diagnostics::color(0.5f, 1.0f, 0.2f));
        graph_->set_color("tick-time", diagnostics::color(0.0f, 0.6f, 0.9f));
        graph_->set_color("dropped-frame", diagnostics::color(0.3f, 0.6f, 0.3f));
        diagnostics::register_graph(graph_);

        build_pipeline();

        if (gst_element_set_state(pipeline_, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
            const auto reason = drain_bus_error();
            destroy_pipeline();
            CASPAR_LOG(error) << print() << L" Failed to start: " << reason;
            CASPAR_THROW_EXCEPTION(user_error() << msg_info(u8(L"Failed to start GStreamer pipeline: " + reason)));
        }

        thread_ = std::thread([this] { run(); });

        CASPAR_LOG(info) << print() << L" Initialized.";
    }

    ~gst_producer()
    {
        is_running_ = false;
        if (thread_.joinable())
            thread_.join();
        destroy_pipeline();
    }

    // The description names everything up to but not including the sink, so the sink is ours:
    // a bin with a ghosted src pad, then videoconvert/videorate, then an appsink whose caps
    // pin the formats gst_frame can map and the channel's rate.
    void build_pipeline()
    {
        GError* error = nullptr;
        auto*   bin   = gst_parse_bin_from_description(u8(description_).c_str(), TRUE, &error);

        if (bin == nullptr) {
            const std::wstring reason = error ? u16(std::string(error->message)) : L"unknown error";
            if (error != nullptr)
                g_error_free(error);
            CASPAR_LOG(error) << L"[gstreamer] Failed to parse pipeline: " << reason;
            CASPAR_THROW_EXCEPTION(user_error() << msg_info(u8(L"Failed to parse GStreamer pipeline: " + reason)));
        }
        if (error != nullptr)
            g_error_free(error);

        auto* convert  = gst_element_factory_make("videoconvert", "caspar_convert");
        auto* rate     = gst_element_factory_make("videorate", "caspar_rate");
        auto* appsink  = gst_element_factory_make("appsink", "caspar_sink");
        pipeline_      = gst_pipeline_new(("caspar_gst_" + std::to_string(instance_no_)).c_str());

        if (convert == nullptr || rate == nullptr || appsink == nullptr || pipeline_ == nullptr) {
            gst_object_unref(bin);
            destroy_pipeline();
            CASPAR_THROW_EXCEPTION(not_supported() << msg_info(
                                       "GStreamer is missing videoconvert, videorate or appsink — "
                                       "the base plugin set is not installed or not on the plugin path."));
        }

        const auto caps_text = std::string("video/x-raw, format=(string){ ") + supported_caps_formats() +
                               " }, framerate=(fraction)" + std::to_string(format_desc_.framerate.numerator()) + "/" +
                               std::to_string(format_desc_.framerate.denominator());
        auto* caps = gst_caps_from_string(caps_text.c_str());

        g_object_set(G_OBJECT(appsink),
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

        gst_bin_add_many(GST_BIN(pipeline_), bin, convert, rate, appsink, nullptr);

        if (!gst_element_link_many(bin, convert, rate, appsink, nullptr)) {
            destroy_pipeline();
            CASPAR_THROW_EXCEPTION(user_error() << msg_info(
                                       "Failed to link the GStreamer pipeline to the sink. The description must "
                                       "end in an element with an unlinked video src pad — no sink of its own."));
        }

        appsink_ = appsink;
    }

    void destroy_pipeline()
    {
        if (pipeline_ != nullptr) {
            gst_element_set_state(pipeline_, GST_STATE_NULL);
            gst_object_unref(pipeline_);
            pipeline_ = nullptr;
        }
        appsink_ = nullptr;
    }

    std::wstring drain_bus_error()
    {
        std::wstring reason = L"no error on the bus";

        auto* bus = gst_element_get_bus(pipeline_);
        if (bus == nullptr)
            return reason;

        while (auto* message = gst_bus_pop_filtered(bus, GST_MESSAGE_ERROR)) {
            reason = describe_error(message);
            gst_message_unref(message);
        }
        gst_object_unref(bus);

        return reason;
    }

    void poll_bus()
    {
        auto* bus = gst_element_get_bus(pipeline_);
        if (bus == nullptr)
            return;

        while (auto* message = gst_bus_pop_filtered(
                   bus, static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_WARNING | GST_MESSAGE_EOS))) {
            switch (GST_MESSAGE_TYPE(message)) {
                case GST_MESSAGE_ERROR:
                    CASPAR_LOG(error) << print() << L" " << describe_error(message);
                    is_running_ = false;
                    break;
                case GST_MESSAGE_WARNING:
                    CASPAR_LOG(warning) << print() << L" " << describe_error(message);
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

    void run()
    {
        while (is_running_) {
            poll_bus();

            // 100 ms, so a stalled pipeline still lets the loop see is_running_ and the bus.
            auto* sample = gst_app_sink_try_pull_sample(GST_APP_SINK(appsink_), 100 * GST_MSECOND);

            if (sample == nullptr)
                continue;

            try {
                frame_timer_.restart();

                auto frame = make_frame(this, *frame_factory_, sample);

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

    // frame_producer

    core::draw_frame receive_impl(const core::video_field field, int nb_samples) override
    {
        graph_->set_value("tick-time", tick_timer_.elapsed() * format_desc_.fps * 0.5);
        tick_timer_.restart();

        std::lock_guard<std::mutex> lock(frames_mutex_);
        if (!frames_.empty()) {
            last_frame_ = frames_.front();
            frames_.pop();
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
        state["gstreamer/pipeline"] = u8(description_);
        state["gstreamer/received"] = static_cast<int64_t>(frames_received_.load());
        state["gstreamer/dropped"]  = static_cast<int64_t>(frames_dropped_.load());
        state["gstreamer/eos"]      = is_eos_.load();
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

    return spl::make_shared<gst_producer>(dependencies.frame_factory, dependencies.format_desc, description);
}

}} // namespace caspar::gstreamer
