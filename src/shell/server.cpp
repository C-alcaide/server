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
 *
 * Author: Robert Nagy, ronag89@gmail.com
 */
#include "included_modules.h"

#include "server.h"

#include <accelerator/accelerator.h>
#include <accelerator/ocio/ocio_config.h>

#include <common/bit_depth.h>
#include <common/render_format.h>
#include <common/env.h>
#include <common/except.h>
#include <common/memory.h>
#include <common/ptree.h>
#include <common/utf.h>

#include <core/consumer/output.h>
#include <core/diagnostics/call_context.h>
#include <core/diagnostics/log_graph.h>
#include <core/diagnostics/osd_graph.h>
#include <core/frame/pixel_format.h>
#include <core/mixer/image/image_mixer.h>
#include <core/producer/cg_proxy.h>
#include <core/producer/color/color_producer.h>
#include <core/producer/frame_producer.h>
#include <core/video_channel.h>

#include <accelerator/compose_self_test.h>
#include <core/stage/stage_fields.h>
#include <accelerator/ogl/image/image_mixer.h>
#include <accelerator/ogl/image/previz_renderer.h>
#include <core/stage/stage_math_self_test.h>

#include <protocol/http/http_server.h>
#include <protocol/http/state_hub.h>
#include <core/video_format.h>

#include <modules/image/consumer/image_consumer.h>

#ifdef ENABLE_VULKAN
#include <accelerator/vulkan/image/image_mixer.h>
#include <modules/vulkan_output/util/vk_device_manager.h>
#endif

#include <protocol/amcp/AMCPCommandsImpl.h>
#include <protocol/amcp/AMCPProtocolStrategy.h>
#include <protocol/amcp/amcp_command_repository.h>
#include <protocol/amcp/amcp_shared.h>
#include <protocol/osc/client.h>
#include <protocol/util/AsyncEventServer.h>
#include <protocol/util/strategy_adapters.h>
#include <protocol/util/tokenize.h>

#include <boost/algorithm/string.hpp>
#include <boost/asio.hpp>
#include <boost/format.hpp>
#include <boost/property_tree/ptree.hpp>

#include <set>
#include <thread>
#include <utility>

namespace caspar {
using namespace core;
using namespace protocol;

/// The default `<name>` of an `<http>` controller.
///
/// A client attached to two servers has to tell their replies and their events apart, and
/// the hostname is the one identifier that is already unique on a network and already
/// means something to the operator reading it. An operator running two servers on one box
/// gives at least one of them an explicit `<name>`.
std::wstring default_server_name()
{
    boost::system::error_code ec;
    const auto                host = boost::asio::ip::host_name(ec);
    return ec || host.empty() ? std::wstring(L"casparcg") : u16(host);
}

std::shared_ptr<boost::asio::io_context> create_io_context_with_running_service()
{
    auto io_context = std::make_shared<boost::asio::io_context>();
    // To keep the io_context::run() running although no pending async
    // operations are posted.
    auto work = std::make_shared<boost::asio::executor_work_guard<boost::asio::io_context::executor_type>>(
        boost::asio::make_work_guard(*io_context));
    auto weak_work = std::weak_ptr<boost::asio::executor_work_guard<boost::asio::io_context::executor_type>>(work);
    auto thread    = std::make_shared<std::thread>([io_context, weak_work] {
        while (auto strong = weak_work.lock()) {
            try {
                io_context->run();
            } catch (...) {
                CASPAR_LOG_CURRENT_EXCEPTION();
            }
        }

        CASPAR_LOG(info) << "[asio] Global io_context uninitialized.";
    });

    return std::shared_ptr<boost::asio::io_context>(io_context.get(), [io_context, work, thread](void*) mutable {
        CASPAR_LOG(info) << "[asio] Shutting down global io_context.";
        work.reset();
        io_context->stop();
        if (thread->get_id() != std::this_thread::get_id())
            thread->join();
        else
            thread->detach();
    });
}

struct server::impl
{
    std::shared_ptr<boost::asio::io_context>               io_context_ = create_io_context_with_running_service();
    video_format_repository                                video_format_repository_;
    accelerator::accelerator                               accelerator_;
    std::shared_ptr<amcp::amcp_command_repository>         amcp_command_repo_;
    std::shared_ptr<amcp::amcp_command_repository_wrapper> amcp_command_repo_wrapper_;
    std::shared_ptr<amcp::command_context_factory>         amcp_context_factory_;
    std::vector<spl::shared_ptr<IO::AsyncEventServer>>     async_servers_;
    std::shared_ptr<IO::AsyncEventServer>                  primary_amcp_server_;
    std::shared_ptr<osc::client>                           osc_client_ = std::make_shared<osc::client>(io_context_);
    // The control API's view of the channel snapshots. Constructed unconditionally and
    // fed by every channel tick whether or not an <http> controller exists: the cost is
    // one atomic pointer store per channel per frame, and making it conditional would
    // mean the channels had to be built after the controllers, which they are not.
    std::shared_ptr<http::state_hub>                       state_hub_  = std::make_shared<http::state_hub>();
    std::shared_ptr<http::http_server>                     http_server_;
    std::vector<std::shared_ptr<void>>                     predefined_osc_subscriptions_;
    spl::shared_ptr<std::vector<protocol::amcp::channel_context>> channels_;
    spl::shared_ptr<core::cg_producer_registry>                   cg_registry_;
    spl::shared_ptr<core::frame_producer_registry>                producer_registry_;
    spl::shared_ptr<core::frame_consumer_registry>                consumer_registry_;
    std::function<void(bool)>                                     shutdown_server_now_;

    impl(const impl&)            = delete;
    impl& operator=(const impl&) = delete;

    explicit impl(std::function<void(bool)> shutdown_server_now)
        : video_format_repository_()
        , accelerator_(video_format_repository_)
        , producer_registry_(spl::make_shared<core::frame_producer_registry>())
        , consumer_registry_(spl::make_shared<core::frame_consumer_registry>())
        , shutdown_server_now_(std::move(shutdown_server_now))
    {
        caspar::core::diagnostics::osd::register_sink();
        // Same metrics, to the log as well, for anything that cannot read a graph -- a test
        // battery, or a post-mortem. Off by default: it is per-graph output on the frame path.
        if (env::properties().get(L"configuration.log-diagnostics", false)) {
            caspar::core::diagnostics::log::register_sink();
        }
    }

    void start()
    {
        setup_video_modes(env::properties());
        CASPAR_LOG(info) << L"Initialized video modes.";

        setup_accelerator(env::properties());
        CASPAR_LOG(info) << L"Initialized accelerator.";

        // Both backends' hand-written layer composition, checked against the registry's
        // generated one on randomised transform pairs. Neither mixer CALLS the generated
        // version yet -- this is what has to be green before one does, and running it every
        // start is what stops the two drifting in the meantime.
        //
        // Run for BOTH backends regardless of which one this server configured. A
        // divergence in the backend nobody selected is still a divergence, and a check that
        // only runs on the configured mixer would report parity that was never tested.
        core::fields::log_stage_fields();
        core::fields::run_stage_math_self_test();
        accelerator::ogl::run_compose_self_test();
        accelerator::vulkan::run_compose_self_test();

        // Before the channels: a channel may carry <ocio-display>/<ocio-view> on a consumer,
        // and those are validated against the loaded config.
        setup_ocio(env::properties());

        auto xml_channels = setup_channels(env::properties());
        CASPAR_LOG(info) << L"Initialized channels.";

        setup_amcp_command_repo();
        CASPAR_LOG(info) << L"Initialized command repository.";

        module_dependencies dependencies(
            cg_registry_, producer_registry_, consumer_registry_, amcp_command_repo_wrapper_, channels_);
        initialize_modules(dependencies);
        CASPAR_LOG(info) << L"Initialized modules.";

        setup_channel_producers_and_consumers(xml_channels);
        CASPAR_LOG(info) << L"Initialized startup producers.";

        setup_controllers(env::properties());
        CASPAR_LOG(info) << L"Initialized controllers.";

        setup_osc(env::properties());
        CASPAR_LOG(info) << L"Initialized osc.";
    }

    ~impl()
    {
        std::weak_ptr<boost::asio::io_context> weak_io_context = io_context_;
        io_context_.reset();
        predefined_osc_subscriptions_.clear();
        osc_client_.reset();

        http_server_.reset();

        amcp_command_repo_wrapper_.reset();
        amcp_command_repo_.reset();
        amcp_context_factory_.reset();

        primary_amcp_server_.reset();
        async_servers_.clear();

        destroy_producers_synchronously();
        destroy_consumers_synchronously();
        channels_->clear();

        while (weak_io_context.lock())
            std::this_thread::sleep_for(std::chrono::milliseconds(100));

        uninitialize_modules();
        core::diagnostics::osd::shutdown();
    }

    /// `<ocio-config>` — the OpenColorIO config the whole server resolves colour space,
    /// display and view names against.
    ///
    /// Absent, which is the default, leaves the pinned built-in
    /// `ocio://studio-config-v4.0.0_aces-v2.0_ocio-v2.5`: nothing to install, and a colour
    /// space name means the same thing on every machine running the same build. Present, it
    /// takes either another `ocio://` built-in URI or a path to a config file.
    ///
    /// Loading here is deliberately eager, against `ocio_config.cpp`'s load-on-first-use
    /// rule. That rule exists so a server with no OCIO channel does not pay to parse a
    /// config; an operator who named one has asked for it, and a bad path should stop the
    /// server now rather than surface as a `MIXER OCIO` failing hours later.
    ///
    /// A failure throws rather than warns. `load_config` keeps the previous config on
    /// failure, so warning would leave the server running the built-in config while the
    /// operator believes their own is loaded — every look silently wrong, nothing in the log
    /// after startup to say so.
    void setup_ocio(const boost::property_tree::wptree& pt)
    {
        const auto uri = pt.get(L"configuration.ocio-config", L"");
        if (uri.empty()) {
            // Say which it is, either way. Without this a default build logged NOTHING about
            // OCIO at startup, and an operator learned it was missing only from a `501` on
            // the first `MIXER OCIO` -- or, worse, never asked and assumed it was there.
            //
            // Neither call loads a config: the pinned built-in is deliberately loaded on
            // first use, so a server with no OCIO channel does not pay to parse one.
            if (accelerator::ocio::available())
                CASPAR_LOG(info) << L"[server] OpenColorIO " << u16(accelerator::ocio::version())
                                 << L" available; the pinned built-in config loads on first use. "
                                    L"Set <ocio-config> to use your own.";
            else
                CASPAR_LOG(info) << L"[server] OpenColorIO NOT available -- this build was made "
                                    L"with ENABLE_OCIO=OFF. MIXER OCIO and OCIO_DISPLAY will "
                                    L"answer 501, and INFO OCIO reports ocio.available=false.";
            return;
        }

        if (!accelerator::ocio::available())
            CASPAR_THROW_EXCEPTION(user_error() << msg_info(
                L"<ocio-config> is set but this build has no OpenColorIO support "
                L"(built with ENABLE_OCIO=OFF)"));

        if (!accelerator::ocio::load_config(u8(uri)))
            CASPAR_THROW_EXCEPTION(user_error() << msg_info(
                L"could not load <ocio-config> " + uri + L" -- see the log line above for why"));

        CASPAR_LOG(info) << L"[server] OCIO config " << uri << L" (" << accelerator::ocio::colorspaces().size()
                         << L" colour spaces, " << accelerator::ocio::displays().size() << L" displays)";
    }

    void setup_video_modes(const boost::property_tree::wptree& pt)
    {
        using boost::property_tree::wptree;

        auto videomodes_config = pt.get_child_optional(L"configuration.video-modes");
        if (videomodes_config) {
            for (auto& xml_channel :
                 pt | witerate_children(L"configuration.video-modes") | welement_context_iteration) {
                ptree_verify_element_name(xml_channel, L"video-mode");

                const std::wstring id = xml_channel.second.get(L"id", L"");
                if (id == L"")
                    CASPAR_THROW_EXCEPTION(user_error() << msg_info(L"Invalid video-mode id: " + id));

                const int width  = xml_channel.second.get<int>(L"width", 0);
                const int height = xml_channel.second.get<int>(L"height", 0);
                if (width == 0 || height == 0)
                    CASPAR_THROW_EXCEPTION(user_error() << msg_info(L"Invalid dimensions: " +
                                                                    boost::lexical_cast<std::wstring>(width) + L"x" +
                                                                    boost::lexical_cast<std::wstring>(height)));

                const int field_count = xml_channel.second.get<int>(L"field-count", 1);
                if (field_count != 1 && field_count != 2)
                    CASPAR_THROW_EXCEPTION(user_error() << msg_info(L"Invalid field-count: " +
                                                                    boost::lexical_cast<std::wstring>(field_count)));

                const int timescale = xml_channel.second.get<int>(L"time-scale", 60000);
                const int duration  = xml_channel.second.get<int>(L"duration", 1000);
                if (timescale == 0 || duration == 0)
                    CASPAR_THROW_EXCEPTION(
                        user_error() << msg_info(L"Invalid framerate: " + boost::lexical_cast<std::wstring>(timescale) +
                                                 L"/" + boost::lexical_cast<std::wstring>(duration)));

                std::vector<int> cadence;
                int              cadence_sum = 0;

                const std::wstring      cadence_str = xml_channel.second.get(L"cadence", L"");
                std::list<std::wstring> cadence_parts;
                boost::split(cadence_parts, cadence_str, boost::is_any_of(L", "));

                for (auto& cad : cadence_parts) {
                    if (cad.empty())
                        continue;

                    const int c = std::stoi(cad);
                    cadence.push_back(c);
                    cadence_sum += c;
                }

                if (cadence.empty()) {
                    // Attempt to calculate in the cadence for integer formats
                    const int c = static_cast<int>(48000 / (static_cast<double>(timescale) / duration) + 0.5);
                    cadence.push_back(c);
                    cadence_sum += c;
                }

                if (cadence_sum * timescale != 48000 * duration * cadence.size()) {
                    auto samples_per_second =
                        static_cast<double>(cadence_sum * timescale) / (duration * cadence.size());
                    CASPAR_THROW_EXCEPTION(user_error() << msg_info(L"Incorrect cadence in video-mode " + id +
                                                                    L". Got " + std::to_wstring(samples_per_second) +
                                                                    L" samples per second, expected 48000"));
                }

                const auto new_format = video_format_desc(
                    video_format::custom, field_count, width, height, width, height, timescale, duration, id, cadence);

                const auto existing = video_format_repository_.find(id);
                if (existing.format != video_format::invalid)
                    CASPAR_THROW_EXCEPTION(user_error() << msg_info(L"Video-mode already exists: " + id));

                video_format_repository_.store(new_format);
            }
        }
    }

    void setup_accelerator(const boost::property_tree::wptree& pt)
    {
        using boost::property_tree::wptree;
        using namespace boost::asio::ip;

#ifdef ENABLE_VULKAN
        caspar::accelerator::accelerator_backend backend = caspar::accelerator::accelerator_backend::invalid;
        auto accelerator = boost::to_lower_copy(pt.get(L"configuration.accelerator", L"auto"));
        if (accelerator == L"auto") {
            backend = caspar::accelerator::accelerator_backend::opengl;
        } else if (accelerator == L"opengl") {
            backend = caspar::accelerator::accelerator_backend::opengl;
        } else if (accelerator == L"vulkan") {
            backend = caspar::accelerator::accelerator_backend::vulkan;
        } else {
            CASPAR_THROW_EXCEPTION(user_error() << msg_info(L"Invalid accelerator: " + accelerator));
        }
#else
        caspar::accelerator::accelerator_backend backend = caspar::accelerator::accelerator_backend::opengl;
#endif

        accelerator_.set_backend(backend);
    }

    std::vector<boost::property_tree::wptree> setup_channels(const boost::property_tree::wptree& pt)
    {
        using boost::property_tree::wptree;

        std::vector<wptree> xml_channels;

        for (auto& xml_channel : pt | witerate_children(L"configuration.channels") | welement_context_iteration) {
            xml_channels.push_back(xml_channel.second);
            ptree_verify_element_name(xml_channel, L"channel");

            auto format_desc_str = xml_channel.second.get(L"video-mode", L"PAL");
            auto format_desc     = video_format_repository_.find(format_desc_str);
            auto color_depth     = xml_channel.second.get<unsigned char>(L"color-depth", 8);
            if (color_depth != 8 && color_depth != 16)
                CASPAR_THROW_EXCEPTION(user_error()
                                       << msg_info(L"Invalid color-depth: " + std::to_wstring(color_depth)));

            auto color_space_str = boost::to_lower_copy(xml_channel.second.get(L"color-space", L"bt709"));
            if (color_space_str != L"bt709" && color_space_str != L"bt2020" &&
                color_space_str != L"p3-d65" && color_space_str != L"p3-dci" && color_space_str != L"adobe-rgb")
                CASPAR_THROW_EXCEPTION(user_error() << msg_info(L"Invalid color-space, must be bt709, bt2020, p3-d65, p3-dci or adobe-rgb"));

            auto color_transfer_str = boost::to_lower_copy(xml_channel.second.get(L"color-transfer", L"sdr"));
            if (color_transfer_str != L"sdr" && color_transfer_str != L"pq" && color_transfer_str != L"hlg" &&
                color_transfer_str != L"linear" && color_transfer_str != L"gamma24" && color_transfer_str != L"gamma26")
                CASPAR_THROW_EXCEPTION(user_error() << msg_info(L"Invalid color-transfer, must be sdr, pq, hlg, linear, gamma24 or gamma26"));

            if (format_desc.format == video_format::invalid)
                CASPAR_THROW_EXCEPTION(user_error() << msg_info(L"Invalid video-mode: " + format_desc_str));

            auto weak_client = std::weak_ptr<osc::client>(osc_client_);
            auto weak_hub    = std::weak_ptr<http::state_hub>(state_hub_);
            auto channel_id  = static_cast<int>(channels_->size() + 1);
            auto depth       = color_depth == 16 ? common::bit_depth::bit16 : common::bit_depth::bit8;

            // <render-format>unorm|fp16</render-format> -- the numeric format of the
            // mixer's *internal* render targets, independent of <color-depth>, which stays
            // the channel's output depth.
            //
            // unorm (default) is bit-identical to the behaviour before this option existed.
            // fp16 lets the composite carry negative values and values above 1.0, which is
            // what a scene-referred linear working space requires; a final resolve pass
            // converts back to the output depth, so consumers are unaffected either way.
            //
            // fp16 is not a free upgrade: near 1.0 its ulp is ~32x coarser than unorm16's,
            // so it is the right format for a linear buffer and the wrong one for a
            // display-encoded one. See docs/architecture/OCIO_INTEGRATION_STUDY.md section 4.3.
            auto render_format_str = boost::to_lower_copy(xml_channel.second.get(L"render-format", L"unorm"));
            if (render_format_str != L"unorm" && render_format_str != L"fp16")
                CASPAR_THROW_EXCEPTION(user_error()
                                       << msg_info(L"Invalid render-format, must be unorm or fp16"));
            auto render_format = render_format_str == L"fp16" ? common::render_format::fp16
                                                             : common::render_format::unorm;
            if (render_format != common::render_format::unorm) {
                CASPAR_LOG(info) << L"[server] Channel " << channel_id << L" render-format "
                                 << render_format_str << L" (float working space).";
            }
            auto default_color_space =
                color_space_str == L"bt2020"    ? core::color_space::bt2020
              : color_space_str == L"p3-d65"   ? core::color_space::p3_d65
              : color_space_str == L"p3-dci"   ? core::color_space::p3_dci
              : color_space_str == L"adobe-rgb" ? core::color_space::adobe_rgb
                                               : core::color_space::bt709;
            auto default_color_transfer = color_transfer_str == L"pq"      ? core::color_transfer::pq
                                        : color_transfer_str == L"hlg"     ? core::color_transfer::hlg
                                        : color_transfer_str == L"linear"  ? core::color_transfer::linear
                                        : color_transfer_str == L"gamma24" ? core::color_transfer::gamma24
                                        : color_transfer_str == L"gamma26" ? core::color_transfer::gamma26
                                                                           : core::color_transfer::sdr;
            auto auto_color_convert = xml_channel.second.get(L"auto-color-convert", true);

            // Parse auto-tone-map: none(0), reinhard(1), aces_filmic(2), aces_rrt(3), hlg_ootf(7)
            auto auto_tone_map_str = boost::to_lower_copy(xml_channel.second.get(L"auto-tone-map", L"none"));
            int  auto_tone_map     = 0;
            if (auto_tone_map_str == L"reinhard")
                auto_tone_map = 1;
            else if (auto_tone_map_str == L"aces_filmic")
                auto_tone_map = 2;
            else if (auto_tone_map_str == L"aces_rrt")
                auto_tone_map = 3;
            else if (auto_tone_map_str == L"hlg_ootf")
                auto_tone_map = 7;
            else if (auto_tone_map_str != L"none")
                CASPAR_THROW_EXCEPTION(user_error() << msg_info(L"Invalid auto-tone-map, must be none, reinhard, aces_filmic, aces_rrt or hlg_ootf"));

            auto display_peak_luminance = xml_channel.second.get<float>(L"display-peak-luminance", 1000.0f);

            // BT.2408 Amd.4: configurable SDR reference white level for PQ mapping.
            // 100 = traditional SDR white (100 cd/m²). 203 = PQ media white per BT.2408 Amd.4.
            // Affects SDR→PQ scale factor: scale = sdr_ref_white / 10000.
            auto sdr_reference_white = xml_channel.second.get<float>(L"sdr-reference-white-nits", 100.0f);
            if (sdr_reference_white <= 0.0f || sdr_reference_white > 1000.0f)
                CASPAR_THROW_EXCEPTION(user_error() << msg_info(L"sdr-reference-white-nits must be between 1 and 1000"));

            // Auto gamut compression: soft-compress out-of-gamut values in auto_color_convert path.
            // When converting wide→narrow gamut (e.g. BT.2020→BT.709), saturated colors can
            // produce negative RGB values. Default (false) = hard-clip (broadcast standard).
            // true = ACES-style soft compression (prevents visible clipping on saturated content).
            auto auto_gamut_compress = xml_channel.second.get(L"auto-gamut-compress", false);

            // Run the colour chain on STRAIGHT (unpremultiplied) RGB, as OCIO documents and
            // as OIIO's `unpremult` on `colorconvert` exists for, rather than on
            // premultiplied RGB as both mixers have always done. C(a*c) != a*C(c) for any
            // non-linear C, so a transform belongs on the surface colour with the coverage
            // reapplied after.
            //
            // Default false because it CHANGES RENDERED OUTPUT wherever content has soft
            // edges and any non-linear transform is configured -- which is most lower
            // thirds. Opaque content is bit-identical either way, which is exactly why no
            // flat-patch battery could see the difference until one drove partial alpha.
            //
            // Measured on both mixers 2026-08-12:
            // CasparCG-TestRunner/docs/alpha_domain_2026-08-12.md.
            auto straight_alpha_grading = xml_channel.second.get(L"straight-alpha-grading", false);

            // Composite in the WORKING space (scene-linear ACEScg) instead of in display
            // space. Every layer converts INTO ACEScg and none of them out of it; the
            // channel applies the display encoding once, after the composite.
            //
            // Blend modes then operate on scene-linear values rather than on 0-1 display
            // values -- which is what the shader comment beside the output block warns
            // about, and is the point rather than a side effect. Default false: it changes
            // every composite of two or more layers.
            //
            // Two hard preconditions, refused rather than warned about, because either one
            // missing turns this into a silently wrong picture:
            //
            //   fp16              ACEScg carries values above 1.0 and below 0. On a unorm
            //                     target the composite clamps them away and the output
            //                     conversion then works from clipped data.
            //   auto-color-convert  every layer needs a defined route INTO the working
            //                     space. Without it a layer with no MIXER COLORSPACE and no
            //                     OCIO transform would enter an ACEScg composite still
            //                     display-encoded, and nothing downstream could tell.
            auto working_space_composite = xml_channel.second.get(L"working-space-composite", false);
            if (working_space_composite) {
                if (render_format != common::render_format::fp16)
                    CASPAR_THROW_EXCEPTION(user_error() << msg_info(
                        L"working-space-composite requires <render-format>fp16</render-format>: "
                        L"a scene-linear ACEScg composite carries values outside [0,1] and a "
                        L"unorm target would clamp them."));
                if (!auto_color_convert)
                    CASPAR_THROW_EXCEPTION(user_error() << msg_info(
                        L"working-space-composite requires <auto-color-convert>true</auto-color-convert>: "
                        L"every layer needs a defined conversion into the working space, or it "
                        L"reaches the composite still display-encoded."));
                CASPAR_LOG(info) << L"[server] Channel " << channel_id
                                 << L" working-space-composite ON: layers blend in scene-linear "
                                    L"ACEScg and the display encoding is applied once, after the "
                                    L"composite. Existing looks that rely on display-space "
                                    L"blending will change.";
            }
            if (straight_alpha_grading) {
                CASPAR_LOG(info) << L"[server] Channel " << channel_id
                                 << L" straight-alpha-grading ON: the colour chain runs on "
                                    L"unpremultiplied RGB. Partial-alpha content renders "
                                    L"differently from the default.";
            }

            // Resolve which physical GPU this channel's mixer should run on.
            // Priority: explicit <gpu> at channel level, else inherit from the
            // channel's first <vulkan-output> consumer's <gpu>, else default 0.
            // Keeping the mixer on the same GPU as the vulkan-output consumer
            // avoids cross-GPU PCIe copies on the hot output path.
            int  gpu_index          = xml_channel.second.get(L"gpu", -1);
            bool gpu_index_explicit = gpu_index >= 0;
            if (gpu_index < 0) {
                gpu_index = 0;
                if (auto consumers = xml_channel.second.get_child_optional(L"consumers")) {
                    for (auto& consumer : *consumers) {
                        if (consumer.first == L"vulkan-output") {
                            gpu_index = consumer.second.get(L"gpu", 0);
                            break;
                        }
                    }
                }
            }
            CASPAR_LOG(info) << L"[server] Channel " << channel_id << L" mixer assigned to GPU " << gpu_index << L".";

            auto channel =
                spl::make_shared<video_channel>(channel_id,
                                                format_desc,
                                                default_color_space,
                                                accelerator_.create_image_mixer(
                                                    channel_id, depth, gpu_index, gpu_index_explicit, render_format),
                                                [channel_id, weak_client, weak_hub](
                                                    const std::shared_ptr<const core::monitor::state>& channel_state) {
                                                    // The API's copy is a pointer store; the
                                                    // OSC copy below is the one that costs,
                                                    // and it is unchanged.
                                                    if (auto hub = weak_hub.lock())
                                                        hub->publish(channel_id, channel_state);

                                                    monitor::state state;
                                                    state[""]["channel"][channel_id] = *channel_state;
                                                    auto client                      = weak_client.lock();
                                                    if (client) {
                                                        client->send(std::move(state));
                                                    }
                                                },
                                                default_color_transfer,
                                                auto_color_convert,
                                                auto_tone_map,
                                                display_peak_luminance,
                                                sdr_reference_white,
                                                auto_gamut_compress,
                                                straight_alpha_grading,
                                                working_space_composite);

            const std::wstring lifecycle_key = L"lock" + std::to_wstring(channel_id);
            channels_->emplace_back(channel, channel->stage(), lifecycle_key);
        }

        return xml_channels;
    }

    void setup_osc(const boost::property_tree::wptree& pt)
    {
        using boost::property_tree::wptree;
        using namespace boost::asio::ip;

        auto default_port                 = pt.get<unsigned short>(L"configuration.osc.default-port", 6250);
        auto disable_send_to_amcp_clients = pt.get(L"configuration.osc.disable-send-to-amcp-clients", false);
        auto predefined_clients           = pt.get_child_optional(L"configuration.osc.predefined-clients");

        if (predefined_clients) {
            for (auto& predefined_client :
                 pt | witerate_children(L"configuration.osc.predefined-clients") | welement_context_iteration) {
                ptree_verify_element_name(predefined_client, L"predefined-client");

                const auto address = ptree_get<std::wstring>(predefined_client.second, L"address");
                const auto port    = ptree_get<unsigned short>(predefined_client.second, L"port");

                boost::system::error_code ec;
                auto                      ipaddr = make_address_v4(u8(address), ec);
                if (!ec)
                    predefined_osc_subscriptions_.push_back(
                        osc_client_->get_subscription_token(udp::endpoint(ipaddr, port)));
                else
                    CASPAR_LOG(warning) << "Invalid OSC client. Must be valid ipv4 address: " << address;
            }
        }

        if (!disable_send_to_amcp_clients && primary_amcp_server_)
            primary_amcp_server_->add_client_lifecycle_object_factory(
                [=, this](const std::string& ipv4_address) -> std::pair<std::wstring, std::shared_ptr<void>> {
                    using namespace boost::asio::ip;

                    return std::make_pair(std::wstring(L"osc_subscribe"),
                                          osc_client_->get_subscription_token(
                                              udp::endpoint(make_address_v4(ipv4_address), default_port)));
                });
    }

    void setup_channel_producers_and_consumers(const std::vector<boost::property_tree::wptree>& xml_channels)
    {
        auto console_client = spl::make_shared<IO::ConsoleClientInfo>();

        std::vector<spl::shared_ptr<core::video_channel>> channels_vec;
        for (auto& cc : *channels_) {
            channels_vec.emplace_back(cc.raw_channel);
        }

        // Count Vulkan consumers per GPU for the startup gate, and pre-create
        // VkDevices if needed.
#ifdef ENABLE_VULKAN
        {
            std::set<int> gpu_indices;
            std::map<int, int> gpu_consumer_counts;  // gpu_index → number of consumers
            for (const auto& ch : xml_channels) {
                if (auto consumers = ch.get_child_optional(L"consumers")) {
                    for (const auto& c : *consumers) {
                        if (c.first == L"vulkan-output") {
                            int gpu = c.second.get(L"gpu", 0);
                            gpu_indices.insert(gpu);
                            gpu_consumer_counts[gpu]++;
                        }
                    }
                }
            }
            if (!gpu_indices.empty()) {
                vulkan_output::vk_device_manager::set_expected_consumers(gpu_consumer_counts);
                // NOTE: warm-up (pre-creating VkDevices) is intentionally omitted.
                // On driver 582.53+ the Vulkan loader dispatches vkCreateDevice
                // through both ICD entries for each GPU (the driver exposes 4
                // VkPhysicalDevices — 2 per GPU).  The failing ICD call hangs
                // the GPU for ~2 seconds, triggering TDR.  The startup gate
                // alone is sufficient: it defers all vkQueueSubmit calls until
                // every consumer has finished init (including vkCreateDevice).
            }
        }
#endif

        for (auto& channel : *channels_) {
            core::diagnostics::scoped_call_context save;
            core::diagnostics::call_context::for_thread().video_channel = channel.raw_channel->index();

            auto xml_channel = xml_channels.at(channel.raw_channel->index() - 1);

            // Consumers
            if (xml_channel.get_child_optional(L"consumers")) {
                for (auto& xml_consumer : xml_channel | witerate_children(L"consumers") | welement_context_iteration) {
                    auto name = xml_consumer.first;

                    try {
                        if (name != L"<xmlcomment>")
                            channel.raw_channel->output().add(
                                consumer_registry_->create_consumer(name,
                                                                    xml_consumer.second,
                                                                    video_format_repository_,
                                                                    channels_vec,
                                                                    channel.raw_channel->get_channel_info()));
                    } catch (...) {
                        CASPAR_LOG_CURRENT_EXCEPTION();
                    }
                }
            }

            // Producers
            if (xml_channel.get_child_optional(L"producers")) {
                for (auto& xml_producer : xml_channel | witerate_children(L"producers") | welement_context_iteration) {
                    ptree_verify_element_name(xml_producer, L"producer");

                    const std::wstring command = xml_producer.second.get_value(L"");
                    const auto         attrs   = xml_producer.second.get_child(L"<xmlattr>");
                    const int          id      = attrs.get(L"id", -1);

                    try {
                        std::list<std::wstring> tokens{
                            L"PLAY", (boost::wformat(L"%i-%i") % channel.raw_channel->index() % id).str()};
                        IO::tokenize(command, tokens);
                        auto cmd = amcp_command_repo_->parse_command(console_client, tokens, L"");

                        if (cmd) {
                            std::wstring res = cmd->Execute(channels_).get();
                            console_client->send(std::move(res), false);
                        }
                    } catch (const user_error&) {
                        CASPAR_LOG(error) << "Failed to parse command: " << command;
                    } catch (...) {
                        CASPAR_LOG_CURRENT_EXCEPTION();
                    }
                }
            }
        }
    }

    void setup_amcp_command_repo()
    {
        amcp_command_repo_ = std::make_shared<amcp::amcp_command_repository>(channels_);

        auto accelerator_device = accelerator_.get_device();
        auto ctx        = std::make_shared<amcp::amcp_command_static_context>(
            video_format_repository_,
            cg_registry_,
            producer_registry_,
            consumer_registry_,
            amcp_command_repo_,
            shutdown_server_now_,
            u8(caspar::env::properties().get(L"configuration.amcp.media-server.host", L"127.0.0.1")),
            u8(caspar::env::properties().get(L"configuration.amcp.media-server.port", L"8000")),
            accelerator_device,
            spl::make_shared_ptr(osc_client_));

        amcp_context_factory_ = std::make_shared<amcp::command_context_factory>(ctx);

        amcp_command_repo_wrapper_ =
            std::make_shared<amcp::amcp_command_repository_wrapper>(amcp_command_repo_, amcp_context_factory_);

        amcp::register_commands(amcp_command_repo_wrapper_);
    }

    void setup_controllers(const boost::property_tree::wptree& pt)
    {
        using boost::property_tree::wptree;
        for (auto& xml_controller : pt | witerate_children(L"configuration.controllers") | welement_context_iteration) {
            auto name = xml_controller.first;

            if (name == L"tcp") {
                // Read INSIDE the branch. `<protocol>` is a TCP-controller element, and
                // reading it before the name was tested made every other controller throw
                // "Missing parameter: protocol" -- a message naming a parameter that the
                // controller in question does not have.
                auto protocol = ptree_get<std::wstring>(xml_controller.second, L"protocol");
                auto port     = ptree_get<unsigned int>(xml_controller.second, L"port");
                auto host = xml_controller.second.get(L"host", L"");

                try {
                    auto asyncbootstrapper = spl::make_shared<IO::AsyncEventServer>(
                        io_context_,
                        create_protocol(protocol, L"TCP Port " + std::to_wstring(port)),
                        static_cast<short>(port),
                        host);
                    async_servers_.push_back(asyncbootstrapper);

                    if (!primary_amcp_server_ && boost::iequals(protocol, L"AMCP"))
                        primary_amcp_server_ = asyncbootstrapper;
                } catch (...) {
                    CASPAR_LOG(fatal) << L"Failed to setup " << protocol << L" controller on port "
                                      << boost::lexical_cast<std::wstring>(port) << L". It is likely already in use";
                    throw;
                    // CASPAR_LOG_CURRENT_EXCEPTION();
                }
            } else if (name == L"http") {
                http::http_config cfg;
                cfg.port         = xml_controller.second.get<unsigned short>(L"port", 5254);
                cfg.host         = xml_controller.second.get(L"host", L"0.0.0.0");
                cfg.name         = xml_controller.second.get(L"name", default_server_name());
                cfg.auth         = boost::to_lower_copy(xml_controller.second.get(L"auth", L"off"));
                cfg.password     = xml_controller.second.get(L"password", L"");
                cfg.extent       = boost::to_lower_copy(xml_controller.second.get(L"extent", L"mixer"));
                cfg.max_prefixes = xml_controller.second.get(L"max-prefixes", 32);

                // Every refusal below logs a FATAL naming the element before it throws.
                // Without that the server exits with the config error nowhere in the log:
                // the shutdown path's own exceptions are logged after it, so the last thing
                // a reader sees is an unrelated failure in a module that was being torn
                // down. Measured -- `<extent></extent>` from a template presented as
                // "[cef_executor] Could not post task", and the harness's startup hint
                // reported that, correctly and uselessly.
                if (cfg.extent != L"state" && cfg.extent != L"mixer") {
                    CASPAR_LOG(fatal) << L"[http-api] <extent>" << cfg.extent
                                      << L"</extent> is not valid; use state or mixer, or omit it.";
                    CASPAR_THROW_EXCEPTION(user_error()
                                           << msg_info(L"Invalid <extent>, must be state or mixer: " + cfg.extent));
                }

                if (cfg.auth != L"off" && cfg.auth != L"password") {
                    CASPAR_LOG(fatal) << L"[http-api] <auth>" << cfg.auth
                                      << L"</auth> is not valid; use off or password.";
                    CASPAR_THROW_EXCEPTION(user_error()
                                           << msg_info(L"Invalid <auth>, must be off or password: " + cfg.auth));
                }

                // A password mode with no password is refused rather than silently letting
                // everyone in: an operator who wrote `<auth>password</auth>` and left
                // `<password>` empty has asked for authentication, and starting anyway
                // would give them a wide-open port they believe is closed.
                if (cfg.auth == L"password" && cfg.password.empty()) {
                    CASPAR_LOG(fatal) << L"[http-api] <auth>password</auth> with an empty <password>.";
                    CASPAR_THROW_EXCEPTION(user_error() << msg_info(L"<auth>password</auth> needs a <password>"));
                }

                if (cfg.auth == L"off")
                    CASPAR_LOG(warning) << L"[http-api] No authentication. Do not bind this port to an interface "
                                           L"reachable off-segment.";

                // A lookup rather than the channel vector itself: `protocol_http` links
                // core and common only, and taking `amcp::channel_context` would put the
                // whole AMCP command layer on its include path to obtain one stage pointer.
                http::api_context api_ctx;
                auto              channels = channels_;
                api_ctx.stage = [channels](int index) -> std::shared_ptr<core::stage_base> {
                    if (index < 1 || index > static_cast<int>(channels->size()))
                        return nullptr;
                    return channels->at(static_cast<std::size_t>(index - 1)).stage;
                };
                api_ctx.concrete_stage = [channels](int index) -> std::shared_ptr<core::stage> {
                    if (index < 1 || index > static_cast<int>(channels->size()))
                        return nullptr;
                    return channels->at(static_cast<std::size_t>(index - 1)).raw_channel->stage();
                };
                api_ctx.channel_count = [channels] { return static_cast<int>(channels->size()); };

                // The re-entrant AMCP bridge, for the two actions that need a producer
                // built from a string. Same shape as `cluster.cpp`'s scheduler executor:
                // tokenize, parse against an internal client, execute, read the reply.
                auto repo   = amcp_command_repo_;
                auto client = std::make_shared<IO::ConsoleClientInfo>();
                api_ctx.amcp =
                    [repo, client](const std::wstring& line) -> http::api_context::amcp_reply {
                    http::api_context::amcp_reply out;
                    try {
                        std::list<std::wstring> tokens;
                        IO::tokenize(line, tokens);
                        auto command = repo->parse_command(
                            spl::shared_ptr<IO::client_connection<wchar_t>>(
                                std::static_pointer_cast<IO::client_connection<wchar_t>>(client)),
                            std::move(tokens),
                            L"");
                        if (!command) {
                            out.code = 400;
                            out.text = L"could not be parsed as an AMCP command";
                            return out;
                        }
                        out.text = command->Execute(repo->channels()).get();
                        // "202 PLAY OK" -- the leading integer is the status.
                        out.code = std::wcstol(out.text.c_str(), nullptr, 10);
                    } catch (const caspar::file_not_found&) {
                        // AMCP's own handlers turn this into `404 ... ERROR` before the
                        // reply string exists, so a bridge that only reads the string sees
                        // nothing and reports `internal` -- which tells a client to retry
                        // something that will never work.
                        out.code = 404;
                        out.text = L"file not found";
                    } catch (const caspar::user_error& e) {
                        out.code = 400;
                        out.text = u16(e.what());
                    } catch (const std::exception& e) {
                        out.code = 500;
                        out.text = u16(e.what());
                    } catch (...) {
                        out.code = 500;
                        out.text = L"unhandled exception";
                    }
                    return out;
                };

                // The stage bridge. `protocol_http` cannot see `accelerator`, and this is
                // where the `dynamic_cast` that reaches the previz renderer belongs -- the
                // same one `AMCPCommandsImpl.cpp` already does, for the same reason.
                api_ctx.set_stage_field =
                    [channels](int                            index,
                               const std::string&             object,
                               const std::string&             path,
                               const core::monitor::vector_t& value) -> http::api_context::stage_write {
                    http::api_context::stage_write out;

                    if (index < 1 || index > static_cast<int>(channels->size())) {
                        out.reason = "no channel " + std::to_string(index);
                        return out;
                    }
                    auto img = channels->at(static_cast<std::size_t>(index - 1)).raw_channel->mixer().get_image_mixer();

                    accelerator::ogl::previz_renderer* previz = nullptr;
                    if (auto* o = dynamic_cast<accelerator::ogl::image_mixer*>(img.get()))
                        previz = &o->get_previz_renderer();
#ifdef ENABLE_VULKAN
                    else if (auto* v = dynamic_cast<accelerator::vulkan::image_mixer*>(img.get()))
                        previz = v->get_previz_renderer();
#endif
                    if (!previz) {
                        out.reason = "channel " + std::to_string(index) + " has no previz renderer";
                        return out;
                    }

                    // Read the CURRENT object first: every camera mutator takes all seven
                    // components at once, so setting `fov` alone means re-sending the other six.
                    // Reading here rather than in the protocol layer is also what makes
                    // `previous` in the reply the value that was actually replaced.
                    const auto snap = previz->stage_snapshot();

                    if (object == "camera" || object == "view_camera") {
                        const auto* cf = core::fields::find_camera_field(path);
                        if (!cf) {
                            out.reason = "no such camera field: " + path;
                            return out;
                        }
                        const bool view = object == "view_camera";
                        auto       cam  = view ? snap.view_camera : snap.camera;
                        out.previous    = cf->get(cam);
                        if (!cf->set(cam, value)) {
                            out.reason = "value does not fit camera field " + path;
                            return out;
                        }
                        if (view)
                            previz->set_view_camera(cam.x, cam.y, cam.z, cam.yaw, cam.pitch, cam.roll, cam.fov);
                        else
                            previz->set_camera(cam.x, cam.y, cam.z, cam.yaw, cam.pitch, cam.roll, cam.fov);
                        // READ BACK FROM THE RENDERER, not from the local copy. Echoing `cam`
                        // would report what was INTENDED; the point of a reply is what the
                        // server now holds. See the screen branch below, where the difference
                        // is not hypothetical.
                        const auto post = previz->stage_snapshot();
                        out.intended    = cf->get(cam);
                        out.written     = cf->get(view ? post.view_camera : post.camera);
                        out.declined    = !(out.written == out.intended);
                        out.applied     = true;
                        return out;
                    }

                    const std::string prefix = "screen/";
                    if (object.rfind(prefix, 0) != 0) {
                        out.reason = "unknown stage object: " + object;
                        return out;
                    }
                    const auto name = object.substr(prefix.size());
                    const auto it   = snap.screens.find(name);
                    if (it == snap.screens.end()) {
                        out.reason = "no screen '" + name + "' on channel " + std::to_string(index);
                        return out;
                    }

                    const auto* sf = core::fields::find_screen_field(path);
                    if (!sf) {
                        out.reason = "no such screen field: " + path;
                        return out;
                    }
                    auto sm      = it->second;
                    out.previous = sf->get(sm);
                    if (!sf->set(sm, value)) {
                        out.reason = "value does not fit screen field " + path;
                        return out;
                    }

                    // Dispatch to the mutator that carries this field. Going through them rather
                    // than writing `sm` back is the whole point: each one also re-applies the
                    // mesh transform and calls `update_projections()`.
                    if (path == "position")
                        previz->set_screen_position(name, sm.pos_x, sm.pos_y, sm.pos_z);
                    else if (path == "rotation")
                        previz->set_screen_rotation(name, sm.rot_yaw, sm.rot_pitch, sm.rot_roll);
                    else if (path == "resolution")
                        previz->set_screen_resolution(name, sm.res_w, sm.res_h);
                    else if (path == "channel")
                        previz->set_screen_channel(name, sm.channel);
                    else if (path == "eye_mode" || path == "design_eye")
                        previz->set_screen_eye_mode(
                            name, sm.eye_mode, sm.design_eye_x, sm.design_eye_y, sm.design_eye_z);
                    else if (path == "arc_v")
                        previz->set_screen_arc_v(name, sm.arc_v_deg);
                    else if (path == "icvfx")
                        previz->set_screen_icvfx(name, sm.icvfx_enable);
                    else {
                        // `size` and `arc` reach here. The renderer has no mutator for either:
                        // both are set only by `add_screen_flat`/`add_screen_curved`, which build
                        // a FRESH screen_meta and would silently discard position, rotation, eye
                        // mode and ICVFX. Refusing is the honest answer until the renderer grows
                        // a resize that regenerates the mesh in place.
                        out.reason = "screen field '" + path +
                                     "' has no mutator in the renderer: it is set only when the "
                                     "screen is created, and re-creating it would discard every "
                                     "other property";
                        return out;
                    }

                    // READ BACK FROM THE RENDERER rather than echoing the local copy, and this
                    // is where it earns itself. `set_screen_eye_mode` writes `design_eye_*` ONLY
                    // when the mode is FIXED, so a `design_eye` write while the screen is in
                    // CAMERA mode stores nothing at all -- and echoing `sm` reported the new
                    // value with `applied: true`. That is precisely the "202 and no change" shape
                    // this whole registry exists to make impossible.
                    const auto after = previz->stage_snapshot();
                    const auto ait   = after.screens.find(name);
                    if (ait == after.screens.end()) {
                        out.reason = "screen '" + name + "' disappeared during the write";
                        return out;
                    }
                    // INTENDED vs ACTUAL, both through `sf->get`, so canonicalisation and float
                    // widening cancel and only a genuine refusal survives the comparison.
                    out.intended = sf->get(sm);
                    out.written  = sf->get(ait->second);
                    out.declined = !(out.written == out.intended);
                    out.applied  = true;
                    return out;
                };

                try {
                    http_server_ = std::make_shared<http::http_server>(io_context_, state_hub_, cfg, api_ctx);
                    CASPAR_LOG(info) << L"[http-api] Listening on " << cfg.host << L":" << cfg.port << L" as '"
                                     << cfg.name << L"' (extent " << cfg.extent << L", auth " << cfg.auth << L").";
                } catch (...) {
                    CASPAR_LOG(fatal) << L"Failed to setup control API on port "
                                      << boost::lexical_cast<std::wstring>(cfg.port)
                                      << L". It is likely already in use";
                    throw;
                }
            } else
                CASPAR_LOG(warning) << "Invalid controller: " << name;
        }
    }

    IO::protocol_strategy_factory<char>::ptr create_protocol(const std::wstring& name,
                                                             const std::wstring& port_description) const
    {
        using namespace IO;

        if (boost::iequals(name, L"AMCP"))
            return amcp::create_char_amcp_strategy_factory(port_description, spl::make_shared_ptr(amcp_command_repo_));

        CASPAR_THROW_EXCEPTION(user_error() << msg_info(L"Invalid protocol: " + name));
    }
};

server::server(std::function<void(bool)> shutdown_server_now)
    : impl_(new impl(std::move(shutdown_server_now)))
{
}
void                                                     server::start() { impl_->start(); }
spl::shared_ptr<protocol::amcp::amcp_command_repository> server::get_amcp_command_repository() const
{
    return spl::make_shared_ptr(impl_->amcp_command_repo_);
}

} // namespace caspar
