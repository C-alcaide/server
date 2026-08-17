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
 * Author: Nicklas P Andersson
 */

#include "../StdAfx.h"

#if defined(_MSC_VER)
#pragma warning(push, 1) // TODO: Legacy code, just disable warnings
#endif

#include "AMCPCommandsImpl.h"

#include "../util/http_request.h"
#include "AMCPCommandQueue.h"
#include "amcp_args.h"

#include "../../modules/ltc/ltc_input.h"
#include "../../modules/portaudio/util/portaudio_device.h"

#include <common/env.h>

#include <common/base64.h>
#include <common/filesystem.h>
#include <common/future.h>
#include <common/log.h>
#include <accelerator/ocio/ocio_config.h>
#include <common/os/filesystem.h>
#include <common/param.h>

#include <core/consumer/frame_consumer.h>
#include <core/consumer/frame_consumer_registry.h>
#include <core/consumer/output.h>
#include <core/diagnostics/call_context.h>
#include <core/diagnostics/osd_graph.h>
#include <core/frame/frame_transform.h>
#include <core/frame/frame_visitor.h>
#include <core/frame/mesh_loader.h>
#include <core/frame/blend_mask_loader.h>
#include <core/frame/write_frame.h>
#include <core/mixer/mixer.h>

#include <accelerator/ogl/image/image_mixer.h>
#include <accelerator/ogl/image/previz_renderer.h>
#include <accelerator/ogl/image/previz_scene.h>
#ifdef ENABLE_VULKAN
#include <accelerator/vulkan/image/image_mixer.h>
#endif
#include <core/producer/cg_proxy.h>
#include <core/producer/color/color_producer.h>
#include <core/producer/frame_producer.h>
#include <core/producer/frame_producer_registry.h>
#include <core/producer/stage.h>
#include <core/producer/transition/sting_producer.h>
#include <core/producer/transition/transition_producer.h>
#include <core/video_format.h>
#include <core/video_channel.h>

#include <protocol/osc/client.h>

#include <algorithm>
#include <fstream>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>

#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/regex.hpp>
#include <boost/archive/iterators/base64_from_binary.hpp>
#include <boost/archive/iterators/insert_linebreaks.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/locale.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/range/adaptor/transformed.hpp>
#include <boost/range/algorithm/copy.hpp>
#include <boost/regex.hpp>

#include <tbb/concurrent_unordered_map.h>

/* Return codes

102 [action]			Information that [action] has happened
101 [action]			Information that [action] has happened plus one row of data

202 [command] OK		[command] has been executed
201 [command] OK		[command] has been executed, plus one row of data
200 [command] OK		[command] has been executed, plus multiple lines of data. ends with an empty line

400 ERROR				the command could not be understood
401 [command] ERROR		invalid/missing channel
402 [command] ERROR		parameter missing
403 [command] ERROR		invalid parameter
404 [command] ERROR		file not found

500 FAILED						internal error
501 [command] FAILED			internal error
502 [command] FAILED			could not read file
503 [command] FAILED			access denied
504 [command] QUEUE OVERFLOW	command queue overflow

600 [command] FAILED	[command] not implemented
*/

namespace caspar { namespace protocol { namespace amcp {

using namespace core;
namespace pt = boost::property_tree;

std::wstring read_utf8_file(const boost::filesystem::path& file)
{
    std::wstringstream           result;
    boost::filesystem::wifstream filestream(file);

    if (filestream) {
        // Consume BOM first
        filestream.get();
        // read all data
        result << filestream.rdbuf();
    }

    return result.str();
}

std::wstring read_latin1_file(const boost::filesystem::path& file)
{
    boost::locale::generator gen;
    gen.locale_cache_enabled(true);
    gen.categories(boost::locale::category_t::codepage);

    std::stringstream           result_stream;
    boost::filesystem::ifstream filestream(file);
    filestream.imbue(gen("en_US.ISO8859-1"));

    if (filestream) {
        // read all data
        result_stream << filestream.rdbuf();
    }

    std::string  result = result_stream.str();
    std::wstring widened_result;

    // The first 255 codepoints in unicode is the same as in latin1
    boost::copy(result | boost::adaptors::transformed([](char c) { return static_cast<unsigned char>(c); }),
                std::back_inserter(widened_result));

    return widened_result;
}

std::wstring read_file(const boost::filesystem::path& file)
{
    static const uint8_t BOM[] = {0xef, 0xbb, 0xbf};

    if (!boost::filesystem::exists(file)) {
        return L"";
    }

    if (boost::filesystem::file_size(file) >= 3) {
        boost::filesystem::ifstream bom_stream(file);

        char header[3];
        bom_stream.read(header, 3);
        bom_stream.close();

        if (std::memcmp(BOM, header, 3) == 0)
            return read_utf8_file(file);
    }

    return read_latin1_file(file);
}

std::wstring get_sub_directory(const std::wstring& base_folder, const std::wstring& sub_directory)
{
    if (sub_directory.empty())
        return base_folder;

    auto found = find_case_insensitive(base_folder + L"/" + sub_directory);

    if (!found)
        CASPAR_THROW_EXCEPTION(file_not_found() << msg_info(L"Sub directory " + sub_directory + L" not found."));

    return *found;
}

std::vector<spl::shared_ptr<core::video_channel>> get_channels(const command_context& ctx)
{
    std::vector<spl::shared_ptr<core::video_channel>> result;
    for (auto& cc : *ctx.channels) {
        result.emplace_back(cc.raw_channel);
    }
    return result;
}

core::frame_producer_dependencies get_producer_dependencies(const std::shared_ptr<core::video_channel>& channel,
                                                            const command_context&                      ctx)
{
    return core::frame_producer_dependencies(channel->frame_factory(),
                                             get_channels(ctx),
                                             ctx.static_context->format_repository,
                                             channel->stage()->video_format_desc(),
                                             ctx.static_context->producer_registry,
                                             ctx.static_context->cg_registry);
}

bool try_match_sting(const std::vector<std::wstring>& params, sting_info& stingInfo)
{
    auto match = std::find_if(params.begin(), params.end(), param_comparer(L"STING"));
    if (match == params.end())
        return false;

    auto start_ind = static_cast<int>(match - params.begin());

    if (params.size() <= start_ind + 1) {
        // No mask filename
        return false;
    }

    auto params_token = params.at(start_ind + 1);
    if (is_args_token(params_token)) {
        auto args = tokenize_args(params_token);

        std::wstring val;
        if (!get_arg_value(args, L"MASK", val)) {
            // TODO - throw error?
            // No mask filename
            return false;
        }
        stingInfo.mask_filename = val;

        if (get_arg_value(args, L"trigger_point", val)) {
            int val2 = boost::lexical_cast<int>(val);
            if (val2 > 0) {
                stingInfo.trigger_point = val2;
            }
        }
        if (get_arg_value(args, L"overlay", val)) {
            stingInfo.overlay_filename = val;
        }

        if (get_arg_value(args, L"audio_fade_start", val)) {
            int val2 = boost::lexical_cast<int>(val);
            if (val2 > 0) {
                stingInfo.audio_fade_start = val2;
            }
        }
        if (get_arg_value(args, L"audio_fade_duration", val)) {
            int val2 = boost::lexical_cast<int>(val);
            if (val2 > 0) {
                stingInfo.audio_fade_duration = val2;
            }
        }

    } else {
        stingInfo.mask_filename = params.at(start_ind + 1);

        if (params.size() > start_ind + 2) {
            stingInfo.trigger_point = boost::lexical_cast<int>(params.at(start_ind + 2));
        }

        if (params.size() > start_ind + 3) {
            stingInfo.overlay_filename = params.at(start_ind + 3);
        }
    }

    return true;
}

// Basic Commands

std::wstring loadbg_command(command_context& ctx)
{
    // Perform loading of the clip
    core::diagnostics::scoped_call_context save;
    core::diagnostics::call_context::for_thread().video_channel = ctx.channel_index + 1;
    core::diagnostics::call_context::for_thread().layer         = ctx.layer_index();

    auto channel   = ctx.channel.raw_channel;
    bool auto_play = contains_param(L"AUTO", ctx.parameters);

    try {
        auto new_producer = ctx.static_context->producer_registry->create_producer(
            get_producer_dependencies(channel, ctx), ctx.parameters);

        if (new_producer == frame_producer::empty())
            CASPAR_THROW_EXCEPTION(file_not_found() << msg_info(!ctx.parameters.empty() ? ctx.parameters[0] : L""));

        spl::shared_ptr<frame_producer> transition_producer = frame_producer::empty();
        transition_info                 transitionInfo;
        sting_info                      stingInfo;

        if (try_match_sting(ctx.parameters, stingInfo)) {
            transition_producer =
                create_sting_producer(get_producer_dependencies(channel, ctx), new_producer, stingInfo);
        } else {
            std::wstring message;
            for (std::wstring& parameter : ctx.parameters) {
                message += boost::to_upper_copy(parameter) + L" ";
            }

            // Try other transitions
            try_match_transition(message, transitionInfo);
            transition_producer = create_transition_producer(new_producer, transitionInfo);
        }

        // TODO - we should pass the format into load(), so that we can catch it having changed since the producer was
        // initialised
        ctx.channel.stage->load(ctx.layer_index(), transition_producer, false, auto_play); // TODO: LOOP
    } catch (file_not_found&) {
        if (contains_param(L"CLEAR_ON_404", ctx.parameters)) {
            ctx.channel.stage->load(
                ctx.layer_index(), core::create_color_producer(channel->frame_factory(), 0), false, auto_play);
        }
        throw;
    }

    return L"202 LOADBG OK\r\n";
}

std::wstring load_command(command_context& ctx)
{
    core::diagnostics::scoped_call_context save;
    core::diagnostics::call_context::for_thread().video_channel = ctx.channel_index + 1;
    core::diagnostics::call_context::for_thread().layer         = ctx.layer_index();

    if (ctx.parameters.empty()) {
        // Must be a promoting load
        ctx.channel.stage->preview(ctx.layer_index());
    } else {
        try {
            auto new_producer = ctx.static_context->producer_registry->create_producer(
                get_producer_dependencies(ctx.channel.raw_channel, ctx), ctx.parameters);
            auto transition_producer = create_transition_producer(new_producer, transition_info{});

            ctx.channel.stage->load(ctx.layer_index(), transition_producer, true);
        } catch (file_not_found&) {
            if (contains_param(L"CLEAR_ON_404", ctx.parameters)) {
                ctx.channel.stage->load(
                    ctx.layer_index(), core::create_color_producer(ctx.channel.raw_channel->frame_factory(), 0), true);
            }
            throw;
        }
    }

    return L"202 LOAD OK\r\n";
}

std::wstring play_command(command_context& ctx)
{
    try {
        if (!ctx.parameters.empty())
            loadbg_command(ctx);
    } catch (file_not_found&) {
        if (contains_param(L"CLEAR_ON_404", ctx.parameters)) {
            ctx.channel.stage->play(ctx.layer_index());
        }
        throw;
    }

    ctx.channel.stage->play(ctx.layer_index());

    return L"202 PLAY OK\r\n";
}

std::wstring pause_command(command_context& ctx)
{
    ctx.channel.stage->pause(ctx.layer_index());
    return L"202 PAUSE OK\r\n";
}

std::wstring resume_command(command_context& ctx)
{
    ctx.channel.stage->resume(ctx.layer_index());
    return L"202 RESUME OK\r\n";
}

std::wstring stop_command(command_context& ctx)
{
    ctx.channel.stage->stop(ctx.layer_index());
    return L"202 STOP OK\r\n";
}

std::wstring clear_command(command_context& ctx)
{
    int index = ctx.layer_index(std::numeric_limits<int>::min());
    if (index != std::numeric_limits<int>::min())
        ctx.channel.stage->clear(index);
    else
        ctx.channel.stage->clear();

    return L"202 CLEAR OK\r\n";
}

std::wstring clear_all_command(command_context& ctx)
{
    for (auto& ch : *ctx.channels) {
        ch.stage->clear();
    }

    return L"202 CLEAR ALL OK\r\n";
}

std::future<std::wstring> callbg_command(command_context& ctx)
{
    const auto result = ctx.channel.stage->callbg(ctx.layer_index(), ctx.parameters).share();

    // TODO: because of std::async deferred timed waiting does not work

    /*auto wait_res = result.wait_for(std::chrono::seconds(2));
    if (wait_res == std::future_status::timeout)
    CASPAR_THROW_EXCEPTION(timed_out());*/

    return std::async(std::launch::deferred, [result]() -> std::wstring {
        std::wstring res = result.get();

        std::wstringstream replyString;
        if (res.empty())
            replyString << L"202 CALLBG OK\r\n";
        else
            replyString << L"201 CALLBG OK\r\n" << res << L"\r\n";

        return replyString.str();
    });
}

std::future<std::wstring> call_command(command_context& ctx)
{
    const auto result = ctx.channel.stage->call(ctx.layer_index(), ctx.parameters).share();

    // TODO: because of std::async deferred timed waiting does not work

    /*auto wait_res = result.wait_for(std::chrono::seconds(2));
    if (wait_res == std::future_status::timeout)
    CASPAR_THROW_EXCEPTION(timed_out());*/

    return std::async(std::launch::deferred, [result]() -> std::wstring {
        std::wstring res = result.get();

        std::wstringstream replyString;
        if (res.empty())
            replyString << L"202 CALL OK\r\n";
        else
            replyString << L"201 CALL OK\r\n" << res << L"\r\n";

        return replyString.str();
    });
}

std::wstring swap_command(command_context& ctx)
{
    bool swap_transforms = ctx.parameters.size() > 1 && boost::iequals(ctx.parameters.at(1), L"TRANSFORMS");

    if (ctx.layer_index(-1) != -1) {
        std::vector<std::wstring> strs;
        boost::split(strs, ctx.parameters[0], boost::is_any_of(L"-"));

        auto ch2 = ctx.channels->at(std::stoi(strs.at(0)) - 1);

        int l1 = ctx.layer_index();
        int l2 = std::stoi(strs.at(1));

        ctx.channel.stage->swap_layer(l1, l2, ch2.stage, swap_transforms);
    } else {
        auto ch2 = ctx.channels->at(std::stoi(ctx.parameters[0]) - 1);
        ctx.channel.stage->swap_layers(ch2.stage, swap_transforms);
    }

    return L"202 SWAP OK\r\n";
}

std::future<std::wstring> apply_command(command_context& ctx)
{
    const auto result = ctx.channel.raw_channel->output().call(ctx.layer_index(), ctx.parameters).share();

    // TODO: because of std::async deferred timed waiting does not work

    /*auto wait_res = result.wait_for(std::chrono::seconds(2));
    if (wait_res == std::future_status::timeout)
    CASPAR_THROW_EXCEPTION(timed_out());*/

    return std::async(std::launch::deferred, [result]() -> std::wstring {
        bool res = result.get();

        std::wstringstream replyString;
        if (res)
            replyString << L"202 APPLY OK\r\n";
        else
            replyString << L"403 APPLY FAILED\r\n";

        return replyString.str();
    });
}

std::wstring add_command(command_context& ctx)
{
    replace_placeholders(L"<CLIENT_IP_ADDRESS>", ctx.client->address(), ctx.parameters);

    core::diagnostics::scoped_call_context save;
    core::diagnostics::call_context::for_thread().video_channel = ctx.channel_index + 1;

    auto consumer =
        ctx.static_context->consumer_registry->create_consumer(ctx.parameters,
                                                               ctx.static_context->format_repository,
                                                               get_channels(ctx),
                                                               ctx.channel.raw_channel->get_consumer_channel_info());
    ctx.channel.raw_channel->output().add(ctx.layer_index(consumer->index()), consumer);

    return L"202 ADD OK\r\n";
}

std::wstring remove_command(command_context& ctx)
{
    auto index = ctx.layer_index(std::numeric_limits<int>::min());

    if (index == std::numeric_limits<int>::min()) {
        replace_placeholders(L"<CLIENT_IP_ADDRESS>", ctx.client->address(), ctx.parameters);

        if (ctx.parameters.size() == 0) {
            return L"402 REMOVE FAILED\r\n";
        }

        index = ctx.static_context->consumer_registry
                    ->create_consumer(ctx.parameters,
                                      ctx.static_context->format_repository,
                                      get_channels(ctx),
                                      ctx.channel.raw_channel->get_consumer_channel_info())
                    ->index();
    }

    if (!ctx.channel.raw_channel->output().remove(index)) {
        return L"404 REMOVE FAILED\r\n";
    }

    return L"202 REMOVE OK\r\n";
}

std::wstring print_command(command_context& ctx)
{
    std::vector<std::wstring> params = {L"IMAGE"};
    if (!ctx.parameters.empty()) {
        params.resize(ctx.parameters.size() + 1);
        std::copy(std::cbegin(ctx.parameters), std::cend(ctx.parameters), params.begin() + 1);
    }

    ctx.channel.raw_channel->output().add(
        ctx.static_context->consumer_registry->create_consumer(params,
                                                               ctx.static_context->format_repository,
                                                               get_channels(ctx),
                                                               ctx.channel.raw_channel->get_consumer_channel_info()));

    return L"202 PRINT OK\r\n";
}

std::wstring print_raw_command(command_context& ctx)
{
    // PRINT 1-10 RAW [filename]
    // Captures layer's raw producer frame (decoded output before mixer transforms)
    // and writes it to <media>/_raw/<filename>.png

    int layer_index = ctx.layer_index(1);

    std::wstring filename;
    if (!ctx.parameters.empty()) {
        filename = ctx.parameters.at(0);
    } else {
        filename = L"raw_" + std::to_wstring(ctx.channel_index + 1) + L"-" + std::to_wstring(layer_index);
    }

    // Resolve the output path relative to <media>/_raw and verify it can't
    // escape that folder via "..", an absolute path, etc. — mirrors the
    // containment check used by MIXER MESH / PROJECTION_BLEND_MASK.
    auto                       raw_base = boost::filesystem::canonical(env::media_folder()) / L"_raw";
    boost::system::error_code ec;
    boost::filesystem::create_directories(raw_base, ec);
    raw_base = boost::filesystem::canonical(raw_base);
    auto resolved = boost::filesystem::weakly_canonical(raw_base / (filename + L".png"));
    if (!is_within_base(resolved, raw_base)) {
        CASPAR_LOG(warning) << L"PRINT RAW: rejected path escaping _raw folder: " << filename;
        return L"403 PRINT RAW FORBIDDEN\r\n";
    }
    std::wstring output_path = resolved.wstring();

    // Get the foreground producer for this layer
    auto producer = ctx.channel.stage->foreground(layer_index).get();
    if (!producer || producer == core::frame_producer::empty()) {
        CASPAR_LOG(warning) << L"PRINT RAW: no foreground producer on layer " << layer_index
                            << L" (producer=" << (producer ? L"empty" : L"null") << L")";
        return L"404 PRINT RAW FAILED\r\n";
    }

    CASPAR_LOG(debug) << L"PRINT RAW: producer=" << producer->name() << L" on layer " << layer_index;

    // Get the last produced frame (raw decode output, no transforms)
    auto raw_frame = producer->last_frame(core::video_field::progressive);
    if (!raw_frame) {
        CASPAR_LOG(warning) << L"PRINT RAW: last_frame returned empty/blank frame from producer " << producer->name();
        return L"404 PRINT RAW FAILED\r\n";
    }

    // Extract the const_frame from the draw_frame tree via visitor
    struct frame_extractor final : public core::frame_visitor
    {
        core::const_frame result;
        void push(const core::frame_transform&) override {}
        void visit(const core::const_frame& f) override {
            if (!result)
                result = f;
        }
        void pop() override {}
    };

    frame_extractor extractor;
    raw_frame.accept(extractor);

    if (!extractor.result) {
        CASPAR_LOG(warning) << L"PRINT RAW: frame_extractor found no const_frame in draw_frame from " << producer->name();
        return L"404 PRINT RAW FAILED\r\n";
    }

    auto frame = extractor.result;

    // Write synchronously, and report what actually happened.
    //
    // This was a detached thread whose result nobody could observe, so the command
    // answered `202 OK` before the write was attempted and every failure inside
    // `write_frame_png` was invisible. Measured: on the GPU-direct path it wrote no file
    // at all for months while returning OK on every call, and the test harness — which
    // has no other way to know — recorded the missing decoder check as simply absent
    // rather than as a failure.
    //
    // A 1080p PNG encode costs tens of milliseconds and PRINT RAW is an explicitly
    // invoked debug command, not a per-frame path. Blocking the AMCP thread for that is
    // a better trade than a status code that cannot be wrong.
    if (!core::write_frame_png(frame, output_path)) {
        CASPAR_LOG(warning) << L"PRINT RAW: write_frame_png failed for " << output_path;
        return L"404 PRINT RAW FAILED\r\n";
    }

    return L"202 PRINT RAW OK\r\n";
}

std::wstring log_level_command(command_context& ctx)
{
    if (ctx.parameters.size() == 0) {
        std::wstringstream replyString;
        replyString << L"201 LOG OK\r\n" << boost::to_upper_copy(log::get_log_level()) << L"\r\n";

        return replyString.str();
    }

    if (!log::set_log_level(ctx.parameters.at(0))) {
        return L"403 LOG FAILED\r\n";
    }

    return L"202 LOG OK\r\n";
}

std::wstring set_command(command_context& ctx)
{
    std::wstring name  = boost::to_upper_copy(ctx.parameters[0]);
    std::wstring value = boost::to_upper_copy(ctx.parameters[1]);

    if (name == L"MODE") {
        auto format_desc = ctx.static_context->format_repository.find(value);
        if (format_desc.format != core::video_format::invalid) {
            ctx.channel.raw_channel->stage()->video_format_desc(format_desc);
            return L"202 SET MODE OK\r\n";
        }

        CASPAR_THROW_EXCEPTION(user_error() << msg_info(L"Invalid video mode"));
    }

    CASPAR_THROW_EXCEPTION(user_error() << msg_info(L"Invalid channel variable"));
}

std::wstring data_store_command(command_context& ctx)
{
    std::wstring filename = env::data_folder();
    filename.append(ctx.parameters[0]);
    filename.append(L".ftd");

    auto data_path       = boost::filesystem::path(filename).parent_path().wstring();
    auto found_data_path = find_case_insensitive(data_path);

    if (found_data_path)
        data_path = *found_data_path;

    if (!boost::filesystem::exists(data_path))
        boost::filesystem::create_directories(data_path);

    auto found_filename = find_case_insensitive(filename);

    if (found_filename)
        filename = *found_filename; // Overwrite case insensitive.

    boost::filesystem::wofstream datafile(filename);
    if (!datafile)
        CASPAR_THROW_EXCEPTION(caspar_exception() << msg_info(L"Could not open file " + filename));

    datafile << static_cast<wchar_t>(65279); // UTF-8 BOM character
    datafile << ctx.parameters[1] << std::flush;
    datafile.close();

    return L"202 DATA STORE OK\r\n";
}

std::wstring data_retrieve_command(command_context& ctx)
{
    std::wstring filename = env::data_folder();
    filename.append(ctx.parameters[0]);
    filename.append(L".ftd");

    std::wstring file_contents;

    auto found_file = find_case_insensitive(filename);

    if (found_file)
        file_contents = read_file(boost::filesystem::path(*found_file));

    if (file_contents.empty())
        CASPAR_THROW_EXCEPTION(file_not_found() << msg_info(filename + L" not found"));

    std::wstringstream reply;
    reply << L"201 DATA RETRIEVE OK\r\n";

    std::wstringstream file_contents_stream(file_contents);
    std::wstring       line;

    bool firstLine = true;
    while (std::getline(file_contents_stream, line)) {
        if (firstLine)
            firstLine = false;
        else
            reply << "\n";

        reply << line;
    }

    reply << "\r\n";
    return reply.str();
}

std::wstring data_list_command(command_context& ctx)
{
    std::wstring sub_directory;

    if (!ctx.parameters.empty())
        sub_directory = ctx.parameters.at(0);

    std::wstringstream replyString;
    replyString << L"200 DATA LIST OK\r\n";

    for (boost::filesystem::recursive_directory_iterator itr(get_sub_directory(env::data_folder(), sub_directory)), end;
         itr != end;
         ++itr) {
        if (boost::filesystem::is_regular_file(itr->path())) {
            if (!boost::iequals(itr->path().extension().wstring(), L".ftd"))
                continue;

            auto relativePath = get_relative_without_extension(itr->path(), env::data_folder());
            auto str          = relativePath.generic_wstring();

            if (str[0] == L'\\' || str[0] == L'/')
                str = std::wstring(str.begin() + 1, str.end());

            replyString << str << L"\r\n";
        }
    }

    replyString << L"\r\n";

    return boost::to_upper_copy(replyString.str());
}

std::wstring data_remove_command(command_context& ctx)
{
    std::wstring filename = env::data_folder();
    filename.append(ctx.parameters[0]);
    filename.append(L".ftd");

    if (!boost::filesystem::exists(filename))
        CASPAR_THROW_EXCEPTION(file_not_found() << msg_info(filename + L" not found"));

    if (!boost::filesystem::remove(filename))
        CASPAR_THROW_EXCEPTION(caspar_exception() << msg_info(filename + L" could not be removed"));

    return L"202 DATA REMOVE OK\r\n";
}

// Template Graphics Commands

std::wstring cg_add_command(command_context& ctx)
{
    // CG 1 ADD 0 "template_folder/templatename" [STARTLABEL] 0/1 [DATA]

    int          layer = std::stoi(ctx.parameters.at(0));
    std::wstring label;             //_parameters[2]
    bool         bDoStart  = false; //_parameters[2] alt. _parameters[3]
    unsigned int dataIndex = 3;

    if (ctx.parameters.at(2).length() > 1) { // read label
        label = ctx.parameters.at(2);
        ++dataIndex;

        if (ctx.parameters.at(3).length() > 0) // read play-on-load-flag
            bDoStart = ctx.parameters.at(3).at(0) == L'1' ? true : false;
    } else { // read play-on-load-flag
        bDoStart = ctx.parameters.at(2).at(0) == L'1' ? true : false;
    }

    const wchar_t* pDataString = nullptr;
    std::wstring   dataFromFile;
    if (ctx.parameters.size() > dataIndex) { // read data
        const std::wstring& dataString = ctx.parameters.at(dataIndex);

        if (dataString.at(0) == L'<' || dataString.at(0) == L'{') // the data is XML or Json
            pDataString = dataString.c_str();
        else {
            // The data is not an XML-string, it must be a filename
            std::wstring filename = env::data_folder();
            filename.append(dataString);
            filename.append(L".ftd");

            auto found_file = find_case_insensitive(filename);

            if (found_file) {
                dataFromFile = read_file(boost::filesystem::path(*found_file));
                pDataString  = dataFromFile.c_str();
            }
        }
    }

    auto filename = ctx.parameters.at(1);
    auto proxy =
        ctx.static_context->cg_registry->get_or_create_proxy(spl::make_shared_ptr(ctx.channel.raw_channel),
                                                             get_producer_dependencies(ctx.channel.raw_channel, ctx),
                                                             ctx.layer_index(core::cg_proxy::DEFAULT_LAYER),
                                                             filename);

    if (proxy == core::cg_proxy::empty())
        CASPAR_THROW_EXCEPTION(file_not_found() << msg_info(L"Could not find template " + filename));
    else
        proxy->add(layer, filename, bDoStart, label, pDataString != nullptr ? pDataString : L"");

    return L"202 CG OK\r\n";
}

std::wstring cg_play_command(command_context& ctx)
{
    int layer = std::stoi(ctx.parameters.at(0));
    ctx.static_context->cg_registry
        ->get_proxy(spl::make_shared_ptr(ctx.channel.raw_channel), ctx.layer_index(core::cg_proxy::DEFAULT_LAYER))
        ->play(layer);

    return L"202 CG OK\r\n";
}

spl::shared_ptr<core::cg_proxy> get_expected_cg_proxy(command_context& ctx)
{
    auto proxy = ctx.static_context->cg_registry->get_proxy(spl::make_shared_ptr(ctx.channel.raw_channel),
                                                            ctx.layer_index(core::cg_proxy::DEFAULT_LAYER));

    if (proxy == cg_proxy::empty())
        CASPAR_THROW_EXCEPTION(expected_user_error() << msg_info(L"No CG proxy running on layer"));

    return proxy;
}

std::wstring cg_stop_command(command_context& ctx)
{
    int layer = std::stoi(ctx.parameters.at(0));
    get_expected_cg_proxy(ctx)->stop(layer);

    return L"202 CG OK\r\n";
}

std::wstring cg_next_command(command_context& ctx)
{
    int layer = std::stoi(ctx.parameters.at(0));
    get_expected_cg_proxy(ctx)->next(layer);

    return L"202 CG OK\r\n";
}

std::wstring cg_remove_command(command_context& ctx)
{
    int layer = std::stoi(ctx.parameters.at(0));
    get_expected_cg_proxy(ctx)->remove(layer);

    return L"202 CG OK\r\n";
}

std::wstring cg_clear_command(command_context& ctx)
{
    ctx.channel.stage->clear(ctx.layer_index(core::cg_proxy::DEFAULT_LAYER));

    return L"202 CG OK\r\n";
}

std::wstring cg_update_command(command_context& ctx)
{
    int layer = std::stoi(ctx.parameters.at(0));

    std::wstring dataString = ctx.parameters.at(1);
    if (dataString.at(0) != L'<' && dataString.at(0) != L'{') {
        // The data is not XML or Json, it must be a filename
        std::wstring filename = env::data_folder();
        filename.append(dataString);
        filename.append(L".ftd");

        dataString = read_file(boost::filesystem::path(filename));
    }

    get_expected_cg_proxy(ctx)->update(layer, dataString);

    return L"202 CG OK\r\n";
}

std::wstring cg_invoke_command(command_context& ctx)
{
    std::wstringstream replyString;
    replyString << L"201 CG OK\r\n";
    int  layer  = std::stoi(ctx.parameters.at(0));
    auto result = get_expected_cg_proxy(ctx)->invoke(layer, ctx.parameters.at(1));
    replyString << result << L"\r\n";

    return replyString.str();
}

// Mixer Commands

std::future<core::frame_transform> get_current_transform(command_context& ctx)
{
    return ctx.channel.stage->get_current_transform(ctx.layer_index());
}

template <typename Func>
std::future<std::wstring> reply_value(command_context& ctx, const Func& extractor)
{
    auto transform = get_current_transform(ctx).share();

    return std::async(std::launch::deferred, [transform, extractor]() -> std::wstring {
        auto value = extractor(transform.get());
        return L"201 MIXER OK\r\n" + boost::lexical_cast<std::wstring>(value) + L"\r\n";
    });
}

class transforms_applier
{
    static tbb::concurrent_unordered_map<int, std::vector<stage::transform_tuple_t>> deferred_transforms_;

    std::vector<stage::transform_tuple_t> transforms_;
    command_context&                      ctx_;
    bool                                  defer_;

  public:
    explicit transforms_applier(command_context& ctx)
        : ctx_(ctx)
    {
        defer_ = !ctx.parameters.empty() && boost::iequals(ctx.parameters.back(), L"DEFER");

        if (defer_)
            ctx.parameters.pop_back();
    }

    void add(stage::transform_tuple_t&& transform) { transforms_.push_back(std::move(transform)); }

    std::future<void> commit_deferred()
    {
        const int  channel_index = ctx_.channel_index;
        const auto f             = ctx_.channel.stage->apply_transforms(deferred_transforms_[channel_index]).share();

        return std::async(std::launch::deferred, [=]() {
            f.get();
            deferred_transforms_[channel_index].clear();
        });
    }

    void apply()
    {
        if (defer_) {
            auto& defer_tranforms = deferred_transforms_[ctx_.channel_index];
            defer_tranforms.insert(defer_tranforms.end(), transforms_.begin(), transforms_.end());
        } else
            ctx_.channel.stage->apply_transforms(transforms_);
    }
};
tbb::concurrent_unordered_map<int, std::vector<stage::transform_tuple_t>> transforms_applier::deferred_transforms_;

std::future<std::wstring> mixer_keyer_command(command_context& ctx)
{
    if (ctx.parameters.empty())
        return reply_value(ctx, [](const frame_transform& t) { return t.image_transform.is_key ? 1 : 0; });

    transforms_applier transforms(ctx);
    bool               value = std::stoi(ctx.parameters.at(0));
    transforms.add(stage::transform_tuple_t(
        ctx.layer_index(),
        [=](frame_transform transform) -> frame_transform {
            transform.image_transform.is_key = value;
            return transform;
        },
        0,
        tweener(L"linear")));
    transforms.apply();

    return make_ready_future<std::wstring>(L"202 MIXER OK\r\n");
}

std::future<std::wstring> mixer_invert_command(command_context& ctx)
{
    if (ctx.parameters.empty())
        return reply_value(ctx, [](const frame_transform& t) { return t.image_transform.invert ? 1 : 0; });

    transforms_applier transforms(ctx);
    bool               value = std::stoi(ctx.parameters.at(0));
    transforms.add(stage::transform_tuple_t(
        ctx.layer_index(),
        [=](frame_transform transform) -> frame_transform {
            transform.image_transform.invert = value;
            return transform;
        },
        0,
        tweener(L"linear")));
    transforms.apply();

    return make_ready_future<std::wstring>(L"202 MIXER OK\r\n");
}

std::future<std::wstring> mixer_chroma_command(command_context& ctx)
{
    if (ctx.parameters.empty()) {
        auto chroma2 = get_current_transform(ctx).share();

        return std::async(std::launch::deferred, [chroma2]() -> std::wstring {
            auto chroma = chroma2.get().image_transform.chroma;
            return L"201 MIXER OK\r\n" + std::wstring(chroma.enable ? L"1 " : L"0 ") +
                   std::to_wstring(chroma.target_hue) + L" " + std::to_wstring(chroma.hue_width) + L" " +
                   std::to_wstring(chroma.min_saturation) + L" " + std::to_wstring(chroma.min_brightness) + L" " +
                   std::to_wstring(chroma.softness) + L" " + std::to_wstring(chroma.spill_suppress) + L" " +
                   std::to_wstring(chroma.spill_suppress_saturation) + L" " +
                   std::wstring(chroma.show_mask ? L"1" : L"0") + L"\r\n";
        });
    }

    transforms_applier transforms(ctx);
    core::chroma       chroma;

    int          duration;
    std::wstring tween;

    auto legacy_mode = core::get_chroma_mode(ctx.parameters.at(0));

    if (legacy_mode) {
        duration = ctx.parameters.size() > 4 ? std::stoi(ctx.parameters.at(4)) : 0;
        tween    = ctx.parameters.size() > 5 ? ctx.parameters.at(5) : L"linear";

        if (*legacy_mode == chroma::legacy_type::none) {
            chroma.enable = false;
        } else {
            chroma.enable                    = true;
            chroma.hue_width                 = 0.5 - std::stod(ctx.parameters.at(1)) * 0.5;
            chroma.min_brightness            = std::stod(ctx.parameters.at(1));
            chroma.min_saturation            = std::stod(ctx.parameters.at(1));
            chroma.softness                  = std::stod(ctx.parameters.at(2)) - std::stod(ctx.parameters.at(1));
            chroma.spill_suppress            = 180.0 - std::stod(ctx.parameters.at(3)) * 180.0;
            chroma.spill_suppress_saturation = 1;

            if (*legacy_mode == chroma::legacy_type::green)
                chroma.target_hue = 120;
            else if (*legacy_mode == chroma::legacy_type::blue)
                chroma.target_hue = 240;
        }
    } else {
        duration = ctx.parameters.size() > 9 ? std::stoi(ctx.parameters.at(9)) : 0;
        tween    = ctx.parameters.size() > 10 ? ctx.parameters.at(10) : L"linear";

        chroma.enable = ctx.parameters.at(0) == L"1";

        if (chroma.enable) {
            chroma.target_hue                = std::stod(ctx.parameters.at(1));
            chroma.hue_width                 = std::stod(ctx.parameters.at(2));
            chroma.min_saturation            = std::stod(ctx.parameters.at(3));
            chroma.min_brightness            = std::stod(ctx.parameters.at(4));
            chroma.softness                  = std::stod(ctx.parameters.at(5));
            chroma.spill_suppress            = std::stod(ctx.parameters.at(6));
            chroma.spill_suppress_saturation = std::stod(ctx.parameters.at(7));
            chroma.show_mask                 = std::stod(ctx.parameters.at(8));
        }
    }

    transforms.add(stage::transform_tuple_t(
        ctx.layer_index(),
        [=](frame_transform transform) -> frame_transform {
            transform.image_transform.chroma = chroma;
            return transform;
        },
        duration,
        tween));
    transforms.apply();

    return make_ready_future<std::wstring>(L"202 MIXER OK\r\n");
}

std::future<std::wstring> mixer_blend_command(command_context& ctx)
{
    if (ctx.parameters.empty())
        return reply_value(ctx, [](const frame_transform& t) { return get_blend_mode(t.image_transform.blend_mode); });

    transforms_applier transforms(ctx);
    auto               value = get_blend_mode(ctx.parameters.at(0));
    transforms.add(stage::transform_tuple_t(
        ctx.layer_index(),
        [=](frame_transform transform) -> frame_transform {
            transform.image_transform.blend_mode = value;
            return transform;
        },
        0,
        tweener(L"linear")));
    transforms.apply();

    return make_ready_future<std::wstring>(L"202 MIXER OK\r\n");
}

// Refuse an unrecognised enumeration token, naming the ones that would have worked.
//
// The parsers below used to `return <first enum value>` for anything they did not
// recognise, so a typo selected a default and the command answered 202. `MIXER BLUR 10
// lenz` rendered a gaussian; `MIXER COLORSPACE ... SDR ...` asked for a transfer that does
// not exist and got LINEAR. Both are indistinguishable, from the client's side, from
// having worked.
//
// THE PART THAT MADE THIS MORE THAN A ONE-LINE CHANGE: every one of those defaults was
// reachable ONLY through the fallback. There was no `LINEAR` case in parse_transfer_fn, no
// `BT709` in parse_gamut_fn, no `NONE` in parse_tonemapping_fn and no `GAUSSIAN` in
// get_blur_type -- so simply erroring on the unknown would have refused the perfectly
// ordinary commands that name a default explicitly, which is what the test harness sends.
// Each default is given a spelling here first, and only then is the rest rejected.
[[noreturn]] void reject_token(const std::wstring& got, const wchar_t* what, const wchar_t* valid)
{
    CASPAR_THROW_EXCEPTION(user_error() << msg_info(std::wstring(what) + L" must be one of " +
                                                    valid + L", got: " + got));
}

core::blur_type get_blur_type(const std::wstring& str)
{
    if (boost::iequals(str, L"gaussian"))
        return core::blur_type::gaussian;
    if (boost::iequals(str, L"box"))
        return core::blur_type::box;
    if (boost::iequals(str, L"directional"))
        return core::blur_type::directional;
    if (boost::iequals(str, L"zoom"))
        return core::blur_type::zoom;
    if (boost::iequals(str, L"tilt_shift") || boost::iequals(str, L"tilt-shift"))
        return core::blur_type::tilt_shift;
    if (boost::iequals(str, L"lens"))
        return core::blur_type::lens;
    reject_token(str, L"blur type",
                 L"GAUSSIAN, BOX, DIRECTIONAL, ZOOM, TILT_SHIFT or LENS");
}

std::wstring get_blur_type_string(core::blur_type type)
{
    switch (type) {
        case core::blur_type::box:
            return L"box";
        case core::blur_type::directional:
            return L"directional";
        case core::blur_type::zoom:
            return L"zoom";
        case core::blur_type::tilt_shift:
            return L"tilt_shift";
        case core::blur_type::lens:
            return L"lens";
        case core::blur_type::gaussian:
        default:
            return L"gaussian";
    }
}

// Parse one grading argument and refuse it if it is outside its range.
//
// `std::stod("nan")` SUCCEEDS, so a command could otherwise put a NaN into the transform
// and every pixel it touched would go black. core::grade_range::contains is written as a
// positive test precisely so NaN falls outside it -- see frame_transform.h.
//
// Defined here rather than beside the grading commands because MIXER BLUR is the first
// command in this file that needs it, several hundred lines above the rest.
double grade_param(const std::wstring& raw, core::grade_range range, const wchar_t* name)
{
    double value = 0.0;
    try {
        value = std::stod(raw);
    } catch (...) {
        CASPAR_THROW_EXCEPTION(user_error()
                               << msg_info(std::wstring(name) + L" is not a number: " + raw));
    }
    if (!range.contains(value)) {
        CASPAR_THROW_EXCEPTION(user_error() << msg_info(std::wstring(name) + L" must be between " +
                                                       std::to_wstring(range.lo) + L" and " +
                                                       std::to_wstring(range.hi) + L", got " + raw));
    }
    return value;
}

// Refuse a short parameter list by name. Without this the .at() below throws
// std::out_of_range, which surfaces as a generic failure rather than saying which
// command wanted how many arguments.
void grade_require(const command_context& ctx, size_t count, const wchar_t* usage)
{
    if (ctx.parameters.size() < count) {
        CASPAR_THROW_EXCEPTION(user_error() << msg_info(std::wstring(L"expected at least ") +
                                                       std::to_wstring(count) + L" parameters: " + usage));
    }
}

std::future<std::wstring> mixer_blur_command(command_context& ctx)
{
    if (ctx.parameters.empty()) {
        auto transform2 = get_current_transform(ctx).share();
        return std::async(std::launch::deferred, [transform2]() -> std::wstring {
            auto blur = transform2.get().image_transform.blur;
            return L"201 MIXER OK\r\n" + std::to_wstring(blur.radius) + L" " + get_blur_type_string(blur.type) + L" " +
                   std::to_wstring(blur.angle) + L" " + std::to_wstring(blur.center[0]) + L" " +
                   std::to_wstring(blur.center[1]) + L" " + std::to_wstring(blur.tilt_y) + L" " +
                   std::to_wstring(blur.tilt_h) + L"\r\n";
        });
    }

    transforms_applier transforms(ctx);
    core::blur_config  blur;

    int          duration = 0;
    std::wstring tween    = L"linear";

    // Format: MIXER 1-10 BLUR <radius> [type] [angle] [center_x] [center_y] [tilt_y] [tilt_h] [duration] [tween]

    blur.radius = grade_param(ctx.parameters.at(0), core::grade_limits::blur_radius, L"radius");
    blur.enable = blur.radius > 0.001; // Enable if radius > 0
    blur.type   = ctx.parameters.size() > 1 ? get_blur_type(ctx.parameters[1]) : core::blur_type::gaussian;
    blur.angle  = ctx.parameters.size() > 2
                      ? grade_param(ctx.parameters[2], core::grade_limits::blur_angle, L"angle")
                      : 0.0;
    blur.center = {ctx.parameters.size() > 3
                       ? grade_param(ctx.parameters[3], core::grade_limits::unit, L"center_x")
                       : 0.5,
                   ctx.parameters.size() > 4
                       ? grade_param(ctx.parameters[4], core::grade_limits::unit, L"center_y")
                       : 0.5};
    blur.tilt_y = ctx.parameters.size() > 5
                      ? grade_param(ctx.parameters[5], core::grade_limits::unit, L"tilt_y")
                      : 0.5;
    blur.tilt_h = ctx.parameters.size() > 6
                      ? grade_param(ctx.parameters[6], core::grade_limits::unit, L"tilt_h")
                      : 0.2;

    duration = ctx.parameters.size() > 7 ? std::stoi(ctx.parameters[7]) : 0;
    tween    = ctx.parameters.size() > 8 ? ctx.parameters[8] : L"linear";

    transforms.add(stage::transform_tuple_t(
        ctx.layer_index(),
        [=](frame_transform transform) -> frame_transform {
            transform.image_transform.blur = blur;
            return transform;
        },
        duration,
        tween));
    transforms.apply();

    return make_ready_future<std::wstring>(L"202 MIXER OK\r\n");
}

// ---------------------------------------------------------------------------
// MIXER SHAPE helpers
// ---------------------------------------------------------------------------

static core::shape_type parse_shape_type(const std::wstring& s)
{
    if (boost::iequals(s, L"RECT") || boost::iequals(s, L"RECTANGLE")) return core::shape_type::rect;
    if (boost::iequals(s, L"ROUNDED_RECT") || boost::iequals(s, L"ROUNDEDRECT")) return core::shape_type::rounded_rect;
    if (boost::iequals(s, L"CIRCLE"))   return core::shape_type::circle;
    if (boost::iequals(s, L"ELLIPSE"))  return core::shape_type::ellipse;
    reject_token(s, L"shape type", L"RECT, ROUNDED_RECT, CIRCLE or ELLIPSE");
}

static std::wstring shape_type_to_string(core::shape_type t)
{
    switch (t) {
        case core::shape_type::rounded_rect: return L"ROUNDED_RECT";
        case core::shape_type::circle:       return L"CIRCLE";
        case core::shape_type::ellipse:      return L"ELLIPSE";
        default:                             return L"RECT";
    }
}

static core::shape_fill_type parse_fill_type(const std::wstring& s)
{
    if (boost::iequals(s, L"SOLID"))  return core::shape_fill_type::solid;
    if (boost::iequals(s, L"LINEAR")) return core::shape_fill_type::linear;
    if (boost::iequals(s, L"RADIAL")) return core::shape_fill_type::radial;
    if (boost::iequals(s, L"CONIC"))  return core::shape_fill_type::conic;
    reject_token(s, L"shape fill type", L"SOLID, LINEAR, RADIAL or CONIC");
}

static std::wstring fill_type_to_string(core::shape_fill_type t)
{
    switch (t) {
        case core::shape_fill_type::linear: return L"LINEAR";
        case core::shape_fill_type::radial: return L"RADIAL";
        case core::shape_fill_type::conic:  return L"CONIC";
        default:                            return L"SOLID";
    }
}

// Parse #RRGGBBAA or #RRGGBB hex string into RGBA doubles [0,1].
static std::array<double, 4> parse_hex_color(const std::wstring& hex)
{
    std::wstring s = hex;
    if (!s.empty() && s[0] == L'#') s = s.substr(1);
    if (s.size() == 6) s += L"FF";
    if (s.size() < 8) return {1.0, 1.0, 1.0, 1.0};
    auto h2d = [&](int pos) -> double {
        return static_cast<double>(std::stoul(s.substr(pos, 2), nullptr, 16)) / 255.0;
    };
    return {h2d(0), h2d(2), h2d(4), h2d(6)};
}

// Format RGBA doubles back to #RRGGBBAA
static std::wstring rgba_to_hex(const std::array<double, 4>& c)
{
    auto d2b = [](double v) -> unsigned int { return static_cast<unsigned int>(std::round(v * 255.0)); };
    wchar_t buf[16];
    swprintf(buf, 16, L"#%02X%02X%02X%02X", d2b(c[0]), d2b(c[1]), d2b(c[2]), d2b(c[3]));
    return buf;
}

std::future<std::wstring> mixer_shape_command(command_context& ctx)
{
    // --- Query mode ---
    if (ctx.parameters.empty()) {
        auto transform2 = get_current_transform(ctx).share();
        return std::async(std::launch::deferred, [transform2]() -> std::wstring {
            auto sh = transform2.get().image_transform.shape;
            if (!sh.enable)
                return L"201 MIXER OK\r\nNONE\r\n";
            return L"201 MIXER OK\r\n" +
                   shape_type_to_string(sh.type) + L" " +
                   std::to_wstring(sh.center[0]) + L" " + std::to_wstring(sh.center[1]) + L" " +
                   std::to_wstring(sh.size[0])   + L" " + std::to_wstring(sh.size[1])   + L" " +
                   L"CORNER_RADIUS "   + std::to_wstring(sh.corner_radius)    + L" " +
                   L"SOFTNESS "        + std::to_wstring(sh.edge_softness)    + L" " +
                   L"FILL "            + fill_type_to_string(sh.fill_type)    + L" " +
                   L"COLOR1 "          + rgba_to_hex(sh.color1)               + L" " +
                   L"COLOR2 "          + rgba_to_hex(sh.color2)               + L" " +
                   L"ANGLE "           + std::to_wstring(sh.gradient_angle)   + L" " +
                   L"GRADIENT_CENTER " + std::to_wstring(sh.gradient_center[0]) + L" "
                                       + std::to_wstring(sh.gradient_center[1]) + L" " +
                   L"STROKE "          + std::to_wstring(sh.stroke_width) + L" " + rgba_to_hex(sh.stroke_color) +
                   L"\r\n";
        });
    }

    // --- Disable ---
    if (boost::iequals(ctx.parameters.at(0), L"NONE")) {
        transforms_applier transforms(ctx);
        transforms.add(stage::transform_tuple_t(
            ctx.layer_index(),
            [](frame_transform transform) -> frame_transform {
                transform.image_transform.shape = core::shape_config{};
                return transform;
            },
            0, L"linear"));
        transforms.apply();
        return make_ready_future<std::wstring>(L"202 MIXER OK\r\n");
    }

    // --- Set mode ---
    // Minimum: MIXER 1-1 SHAPE <type> <cx> <cy> <w> <h> [keywords...] [DURATION d] [TWEEN t]
    if (ctx.parameters.size() < 5)
        CASPAR_THROW_EXCEPTION(user_error() << msg_info(L"MIXER SHAPE requires at least: type cx cy w h"));

    core::shape_config sh;
    sh.enable = true;
    sh.type   = parse_shape_type(ctx.parameters.at(0));
    sh.center = { std::stod(ctx.parameters.at(1)), std::stod(ctx.parameters.at(2)) };
    sh.size   = { std::stod(ctx.parameters.at(3)), std::stod(ctx.parameters.at(4)) };

    int          duration = 0;
    std::wstring tween    = L"linear";

    // Parse keyword arguments (order-independent, start at index 5)
    for (std::size_t i = 5; i < ctx.parameters.size(); ++i) {
        const auto& kw = ctx.parameters[i];
        if (boost::iequals(kw, L"CORNER_RADIUS") && i + 1 < ctx.parameters.size())
            sh.corner_radius = std::stod(ctx.parameters[++i]);
        else if (boost::iequals(kw, L"SOFTNESS") && i + 1 < ctx.parameters.size())
            sh.edge_softness = std::stod(ctx.parameters[++i]);
        else if (boost::iequals(kw, L"FILL") && i + 1 < ctx.parameters.size())
            sh.fill_type = parse_fill_type(ctx.parameters[++i]);
        else if (boost::iequals(kw, L"COLOR1") && i + 1 < ctx.parameters.size())
            sh.color1 = parse_hex_color(ctx.parameters[++i]);
        else if (boost::iequals(kw, L"COLOR2") && i + 1 < ctx.parameters.size())
            sh.color2 = parse_hex_color(ctx.parameters[++i]);
        else if (boost::iequals(kw, L"ANGLE") && i + 1 < ctx.parameters.size())
            sh.gradient_angle = std::stod(ctx.parameters[++i]);
        else if (boost::iequals(kw, L"GRADIENT_CENTER") && i + 2 < ctx.parameters.size()) {
            sh.gradient_center[0] = std::stod(ctx.parameters[++i]);
            sh.gradient_center[1] = std::stod(ctx.parameters[++i]);
        } else if (boost::iequals(kw, L"STROKE") && i + 2 < ctx.parameters.size()) {
            sh.stroke_enable = true;
            sh.stroke_width  = std::stod(ctx.parameters[++i]);
            sh.stroke_color  = parse_hex_color(ctx.parameters[++i]);
        } else if (boost::iequals(kw, L"DURATION") && i + 1 < ctx.parameters.size())
            duration = std::stoi(ctx.parameters[++i]);
        else if (boost::iequals(kw, L"TWEEN") && i + 1 < ctx.parameters.size())
            tween = ctx.parameters[++i];
    }

    transforms_applier transforms(ctx);
    transforms.add(stage::transform_tuple_t(
        ctx.layer_index(),
        [sh](frame_transform transform) -> frame_transform {
            transform.image_transform.shape = sh;
            return transform;
        },
        duration,
        tween));
    transforms.apply();

    return make_ready_future<std::wstring>(L"202 MIXER OK\r\n");
}

/// `range` is OPTIONAL, and that is deliberate. This helper is shared with MIXER OPACITY,
/// BRIGHTNESS, SATURATION, CONTRAST, ROTATION and VOLUME, which have never validated
/// anything -- `MIXER OPACITY -5` is accepted today. Retrofitting them changes behaviour
/// for existing clients, so they keep the bare `std::stod` and only the grading commands
/// pass a range. Upstream drew the same line (CasparCG/server#1765).
template <typename Getter, typename Setter>
std::future<std::wstring>
single_double_animatable_mixer_command(command_context&                 ctx,
                                       const Getter&                    getter,
                                       const Setter&                    setter,
                                       std::optional<core::grade_range> range = std::nullopt,
                                       const wchar_t*                   name  = L"value")
{
    if (ctx.parameters.empty())
        return reply_value(ctx, getter);

    transforms_applier transforms(ctx);
    double             value    = range ? grade_param(ctx.parameters.at(0), *range, name)
                                        : std::stod(ctx.parameters.at(0));
    int                duration = ctx.parameters.size() > 1 ? std::stoi(ctx.parameters[1]) : 0;
    std::wstring       tween    = ctx.parameters.size() > 2 ? ctx.parameters[2] : L"linear";

    transforms.add(stage::transform_tuple_t(
        ctx.layer_index(),
        [=](frame_transform transform) -> frame_transform {
            setter(transform, value);
            return transform;
        },
        duration,
        tween));
    transforms.apply();

    return make_ready_future<std::wstring>(L"202 MIXER OK\r\n");
}

std::future<std::wstring> mixer_opacity_command(command_context& ctx)
{
    return single_double_animatable_mixer_command(
        ctx,
        [](const frame_transform& t) { return t.image_transform.opacity; },
        [](frame_transform& t, double value) { t.image_transform.opacity = value; });
}

std::future<std::wstring> mixer_brightness_command(command_context& ctx)
{
    return single_double_animatable_mixer_command(
        ctx,
        [](const frame_transform& t) { return t.image_transform.brightness; },
        [](frame_transform& t, double value) { t.image_transform.brightness = value; });
}

std::future<std::wstring> mixer_saturation_command(command_context& ctx)
{
    return single_double_animatable_mixer_command(
        ctx,
        [](const frame_transform& t) { return t.image_transform.saturation; },
        [](frame_transform& t, double value) { t.image_transform.saturation = value; });
}

std::future<std::wstring> mixer_contrast_command(command_context& ctx)
{
    return single_double_animatable_mixer_command(
        ctx,
        [](const frame_transform& t) { return t.image_transform.contrast; },
        [](frame_transform& t, double value) { t.image_transform.contrast = value; });
}

std::future<std::wstring> mixer_levels_command(command_context& ctx)
{
    if (ctx.parameters.empty()) {
        auto levels2 = get_current_transform(ctx).share();
        return std::async(std::launch::deferred, [levels2]() -> std::wstring {
            auto levels = levels2.get().image_transform.levels;
            return L"201 MIXER OK\r\n" + std::to_wstring(levels.min_input) + L" " + std::to_wstring(levels.max_input) +
                   L" " + std::to_wstring(levels.gamma) + L" " + std::to_wstring(levels.min_output) + L" " +
                   std::to_wstring(levels.max_output) + L"\r\n";
        });
    }

    transforms_applier transforms(ctx);
    levels             value;
    value.min_input       = std::stod(ctx.parameters.at(0));
    value.max_input       = std::stod(ctx.parameters.at(1));
    value.gamma           = std::stod(ctx.parameters.at(2));
    value.min_output      = std::stod(ctx.parameters.at(3));
    value.max_output      = std::stod(ctx.parameters.at(4));
    int          duration = ctx.parameters.size() > 5 ? std::stoi(ctx.parameters[5]) : 0;
    std::wstring tween    = ctx.parameters.size() > 6 ? ctx.parameters[6] : L"linear";

    transforms.add(stage::transform_tuple_t(
        ctx.layer_index(),
        [=](frame_transform transform) -> frame_transform {
            transform.image_transform.levels = value;
            return transform;
        },
        duration,
        tween));
    transforms.apply();

    return make_ready_future<std::wstring>(L"202 MIXER OK\r\n");
}

std::future<std::wstring> mixer_fill_command(command_context& ctx)
{
    if (ctx.parameters.empty()) {
        auto transform2 = get_current_transform(ctx).share();
        return std::async(std::launch::deferred, [transform2]() -> std::wstring {
            auto transform   = transform2.get().image_transform;
            auto translation = transform.fill_translation;
            auto scale       = transform.fill_scale;
            return L"201 MIXER OK\r\n" + std::to_wstring(translation[0]) + L" " + std::to_wstring(translation[1]) +
                   L" " + std::to_wstring(scale[0]) + L" " + std::to_wstring(scale[1]) + L"\r\n";
        });
    }

    transforms_applier transforms(ctx);
    int                duration = ctx.parameters.size() > 4 ? std::stoi(ctx.parameters[4]) : 0;
    std::wstring       tween    = ctx.parameters.size() > 5 ? ctx.parameters[5] : L"linear";
    double             x        = std::stod(ctx.parameters.at(0));
    double             y        = std::stod(ctx.parameters.at(1));
    double             x_s      = std::stod(ctx.parameters.at(2));
    double             y_s      = std::stod(ctx.parameters.at(3));

    transforms.add(stage::transform_tuple_t(
        ctx.layer_index(),
        [=](frame_transform transform) mutable -> frame_transform {
            transform.image_transform.fill_translation[0] = x;
            transform.image_transform.fill_translation[1] = y;
            transform.image_transform.fill_scale[0]       = x_s;
            transform.image_transform.fill_scale[1]       = y_s;
            return transform;
        },
        duration,
        tween));
    transforms.apply();

    return make_ready_future<std::wstring>(L"202 MIXER OK\r\n");
}

std::future<std::wstring> mixer_clip_command(command_context& ctx)
{
    if (ctx.parameters.empty()) {
        auto transform2 = get_current_transform(ctx).share();
        return std::async(std::launch::deferred, [transform2]() -> std::wstring {
            auto transform   = transform2.get().image_transform;
            auto translation = transform.clip_translation;
            auto scale       = transform.clip_scale;

            return L"201 MIXER OK\r\n" + std::to_wstring(translation[0]) + L" " + std::to_wstring(translation[1]) +
                   L" " + std::to_wstring(scale[0]) + L" " + std::to_wstring(scale[1]) + L"\r\n";
        });
    }

    transforms_applier transforms(ctx);
    int                duration = ctx.parameters.size() > 4 ? std::stoi(ctx.parameters[4]) : 0;
    std::wstring       tween    = ctx.parameters.size() > 5 ? ctx.parameters[5] : L"linear";
    double             x        = std::stod(ctx.parameters.at(0));
    double             y        = std::stod(ctx.parameters.at(1));
    double             x_s      = std::stod(ctx.parameters.at(2));
    double             y_s      = std::stod(ctx.parameters.at(3));

    transforms.add(stage::transform_tuple_t(
        ctx.layer_index(),
        [=](frame_transform transform) -> frame_transform {
            transform.image_transform.clip_translation[0] = x;
            transform.image_transform.clip_translation[1] = y;
            transform.image_transform.clip_scale[0]       = x_s;
            transform.image_transform.clip_scale[1]       = y_s;
            return transform;
        },
        duration,
        tween));
    transforms.apply();

    return make_ready_future<std::wstring>(L"202 MIXER OK\r\n");
}

std::future<std::wstring> mixer_anchor_command(command_context& ctx)
{
    if (ctx.parameters.empty()) {
        auto transform2 = get_current_transform(ctx).share();
        return std::async(std::launch::deferred, [transform2]() -> std::wstring {
            auto transform = transform2.get().image_transform;
            auto anchor    = transform.anchor;
            return L"201 MIXER OK\r\n" + std::to_wstring(anchor[0]) + L" " + std::to_wstring(anchor[1]) + L"\r\n";
        });
    }

    transforms_applier transforms(ctx);
    int                duration = ctx.parameters.size() > 2 ? std::stoi(ctx.parameters[2]) : 0;
    std::wstring       tween    = ctx.parameters.size() > 3 ? ctx.parameters[3] : L"linear";
    double             x        = std::stod(ctx.parameters.at(0));
    double             y        = std::stod(ctx.parameters.at(1));

    transforms.add(stage::transform_tuple_t(
        ctx.layer_index(),
        [=](frame_transform transform) mutable -> frame_transform {
            transform.image_transform.anchor[0] = x;
            transform.image_transform.anchor[1] = y;
            return transform;
        },
        duration,
        tween));
    transforms.apply();

    return make_ready_future<std::wstring>(L"202 MIXER OK\r\n");
}

std::future<std::wstring> mixer_crop_command(command_context& ctx)
{
    if (ctx.parameters.empty()) {
        auto transform2 = get_current_transform(ctx).share();
        return std::async(std::launch::deferred, [transform2]() -> std::wstring {
            auto crop = transform2.get().image_transform.crop;
            return L"201 MIXER OK\r\n" + std::to_wstring(crop.ul[0]) + L" " + std::to_wstring(crop.ul[1]) + L" " +
                   std::to_wstring(crop.lr[0]) + L" " + std::to_wstring(crop.lr[1]) + L"\r\n";
        });
    }

    transforms_applier transforms(ctx);
    int                duration = ctx.parameters.size() > 4 ? std::stoi(ctx.parameters[4]) : 0;
    std::wstring       tween    = ctx.parameters.size() > 5 ? ctx.parameters[5] : L"linear";
    double             ul_x     = std::stod(ctx.parameters.at(0));
    double             ul_y     = std::stod(ctx.parameters.at(1));
    double             lr_x     = std::stod(ctx.parameters.at(2));
    double             lr_y     = std::stod(ctx.parameters.at(3));

    transforms.add(stage::transform_tuple_t(
        ctx.layer_index(),
        [=](frame_transform transform) -> frame_transform {
            transform.image_transform.crop.ul[0] = ul_x;
            transform.image_transform.crop.ul[1] = ul_y;
            transform.image_transform.crop.lr[0] = lr_x;
            transform.image_transform.crop.lr[1] = lr_y;
            return transform;
        },
        duration,
        tween));
    transforms.apply();

    return make_ready_future<std::wstring>(L"202 MIXER OK\r\n");
}

std::future<std::wstring> mixer_rotation_command(command_context& ctx)
{
    static const double PI = 3.141592653589793;

    return single_double_animatable_mixer_command(
        ctx,
        [](const frame_transform& t) { return t.image_transform.angle / PI * 180.0; },
        [](frame_transform& t, double value) { t.image_transform.angle = value * PI / 180.0; });
}

std::future<std::wstring> mixer_perspective_command(command_context& ctx)
{
    if (ctx.parameters.empty()) {
        auto transform2 = get_current_transform(ctx).share();
        return std::async(std::launch::deferred, [transform2]() -> std::wstring {
            auto perspective = transform2.get().image_transform.perspective;
            return L"201 MIXER OK\r\n" + std::to_wstring(perspective.ul[0]) + L" " +
                   std::to_wstring(perspective.ul[1]) + L" " + std::to_wstring(perspective.ur[0]) + L" " +
                   std::to_wstring(perspective.ur[1]) + L" " + std::to_wstring(perspective.lr[0]) + L" " +
                   std::to_wstring(perspective.lr[1]) + L" " + std::to_wstring(perspective.ll[0]) + L" " +
                   std::to_wstring(perspective.ll[1]) + L"\r\n";
        });
    }

    transforms_applier transforms(ctx);
    int                duration = ctx.parameters.size() > 8 ? std::stoi(ctx.parameters[8]) : 0;
    std::wstring       tween    = ctx.parameters.size() > 9 ? ctx.parameters[9] : L"linear";
    double             ul_x     = std::stod(ctx.parameters.at(0));
    double             ul_y     = std::stod(ctx.parameters.at(1));
    double             ur_x     = std::stod(ctx.parameters.at(2));
    double             ur_y     = std::stod(ctx.parameters.at(3));
    double             lr_x     = std::stod(ctx.parameters.at(4));
    double             lr_y     = std::stod(ctx.parameters.at(5));
    double             ll_x     = std::stod(ctx.parameters.at(6));
    double             ll_y     = std::stod(ctx.parameters.at(7));

    transforms.add(stage::transform_tuple_t(
        ctx.layer_index(),
        [=](frame_transform transform) -> frame_transform {
            transform.image_transform.perspective.ul[0] = ul_x;
            transform.image_transform.perspective.ul[1] = ul_y;
            transform.image_transform.perspective.ur[0] = ur_x;
            transform.image_transform.perspective.ur[1] = ur_y;
            transform.image_transform.perspective.lr[0] = lr_x;
            transform.image_transform.perspective.lr[1] = lr_y;
            transform.image_transform.perspective.ll[0] = ll_x;
            transform.image_transform.perspective.ll[1] = ll_y;
            return transform;
        },
        duration,
        tween));
    transforms.apply();

    return make_ready_future<std::wstring>(L"202 MIXER OK\r\n");
}

std::future<std::wstring> mixer_projection_command(command_context& ctx)
{
    static const double PI = 3.141592653589793;
    static const double DEG2RAD = PI / 180.0;
    static const double RAD2DEG = 180.0 / PI;

    if (ctx.parameters.empty()) {
        auto transform2 = get_current_transform(ctx).share();
        return std::async(std::launch::deferred, [transform2]() -> std::wstring {
            auto projection = transform2.get().image_transform.projection;
            return L"201 MIXER OK\r\n" + std::to_wstring(projection.yaw * RAD2DEG) + L" " +
                   std::to_wstring(projection.pitch * RAD2DEG) + L" " +
                   std::to_wstring(projection.roll * RAD2DEG) + L" " +
                   std::to_wstring(projection.fov * RAD2DEG) + L"\r\n";
        });
    }

    transforms_applier transforms(ctx);
    int                duration = ctx.parameters.size() > 4 ? std::stoi(ctx.parameters[4]) : 0;
    std::wstring       tween    = ctx.parameters.size() > 5 ? ctx.parameters[5] : L"linear";
    double             yaw      = std::stod(ctx.parameters.at(0)) * DEG2RAD;
    double             pitch    = std::stod(ctx.parameters.at(1)) * DEG2RAD;
    double             roll     = std::stod(ctx.parameters.at(2)) * DEG2RAD;
    double             fov      = std::stod(ctx.parameters.at(3)) * DEG2RAD;

    transforms.add(stage::transform_tuple_t(
        ctx.layer_index(),
        [=](frame_transform transform) -> frame_transform {
            transform.image_transform.projection.enable = (fov > 0.0);
            transform.image_transform.projection.yaw    = yaw;
            transform.image_transform.projection.pitch  = pitch;
            transform.image_transform.projection.roll   = roll;
            transform.image_transform.projection.fov    = fov;
            return transform;
        },
        duration,
        tween));
    transforms.apply();

    return make_ready_future<std::wstring>(L"202 MIXER OK\r\n");
}

std::future<std::wstring> mixer_projection_offset_command(command_context& ctx)
{
    if (ctx.parameters.empty()) {
        auto transform2 = get_current_transform(ctx).share();
        return std::async(std::launch::deferred, [transform2]() -> std::wstring {
            auto proj = transform2.get().image_transform.projection;
            return L"201 MIXER OK\r\n" + std::to_wstring(proj.offset_x) + L" " +
                   std::to_wstring(proj.offset_y) + L"\r\n";
        });
    }

    transforms_applier transforms(ctx);
    int          duration = ctx.parameters.size() > 2 ? std::stoi(ctx.parameters[2]) : 0;
    std::wstring tween    = ctx.parameters.size() > 3 ? ctx.parameters[3] : L"linear";
    double       offset_x = std::stod(ctx.parameters.at(0));
    double       offset_y = std::stod(ctx.parameters.at(1));

    transforms.add(stage::transform_tuple_t(
        ctx.layer_index(),
        [=](frame_transform transform) -> frame_transform {
            transform.image_transform.projection.offset_x = offset_x;
            transform.image_transform.projection.offset_y = offset_y;
            return transform;
        },
        duration,
        tween));
    transforms.apply();

    return make_ready_future<std::wstring>(L"202 MIXER OK\r\n");
}

std::future<std::wstring> mixer_projection_curve_command(command_context& ctx)
{
    static const double PI = 3.141592653589793;
    static const double DEG2RAD = PI / 180.0;
    static const double RAD2DEG = 180.0 / PI;

    if (ctx.parameters.empty()) {
        auto transform2 = get_current_transform(ctx).share();
        return std::async(std::launch::deferred, [transform2]() -> std::wstring {
            auto& proj = transform2.get().image_transform.projection;
            std::wstring type_str = L"FLAT";
            if (proj.curve_type == core::screen_curve_type::cylinder)
                type_str = L"CYLINDER";
            else if (proj.curve_type == core::screen_curve_type::sphere)
                type_str = L"SPHERE";
            else if (proj.curve_type == core::screen_curve_type::fisheye)
                type_str = L"FISHEYE";
            return L"201 MIXER OK\r\n" + type_str + L" " +
                   std::to_wstring(proj.screen_arc * RAD2DEG) + L" " +
                   std::to_wstring(proj.screen_arc_v * RAD2DEG) + L" " +
                   std::to_wstring(proj.eye_distance) + L"\r\n";
        });
    }

    using core::screen_curve_type;
    transforms_applier transforms(ctx);
    const auto&  type_arg = ctx.parameters.at(0);
    screen_curve_type curve_type = screen_curve_type::flat;
    if      (boost::iequals(type_arg, L"CYLINDER")) curve_type = screen_curve_type::cylinder;
    else if (boost::iequals(type_arg, L"SPHERE"))   curve_type = screen_curve_type::sphere;
    else if (boost::iequals(type_arg, L"FISHEYE"))  curve_type = screen_curve_type::fisheye;
    double screen_arc    = std::stod(ctx.parameters.at(1)) * DEG2RAD;
    // Optional: type arc [arc_v] [eye_distance] [duration] [tween]
    double       screen_arc_v = ctx.parameters.size() > 2 ? std::stod(ctx.parameters[2]) * DEG2RAD : 0.0;
    double       eye_distance = ctx.parameters.size() > 3 ? std::stod(ctx.parameters[3]) : 1.0;
    int          duration     = ctx.parameters.size() > 4 ? std::stoi(ctx.parameters[4]) : 0;
    std::wstring tween        = ctx.parameters.size() > 5 ? ctx.parameters[5] : L"linear";
    bool         curve_enable = (curve_type != screen_curve_type::flat && screen_arc != 0.0);

    transforms.add(stage::transform_tuple_t(
        ctx.layer_index(),
        [=](frame_transform transform) -> frame_transform {
            transform.image_transform.projection.curve_type   = curve_type;
            transform.image_transform.projection.screen_arc   = screen_arc;
            transform.image_transform.projection.screen_arc_v = screen_arc_v;
            transform.image_transform.projection.eye_distance = eye_distance;
            transform.image_transform.projection.curve_enable = curve_enable;
            transform.image_transform.projection.curve_auto   = false; // explicit set locks against auto-projection
            return transform;
        },
        duration,
        tween));
    transforms.apply();

    return make_ready_future<std::wstring>(L"202 MIXER OK\r\n");
}

std::future<std::wstring> mixer_projection_lens_command(command_context& ctx)
{
    if (ctx.parameters.empty()) {
        auto transform2 = get_current_transform(ctx).share();
        return std::async(std::launch::deferred, [transform2]() -> std::wstring {
            auto& proj = transform2.get().image_transform.projection;
            std::wstring lens_str = L"RECTILINEAR";
            if (proj.source_lens == core::screen_curve_type::cylinder)
                lens_str = L"CYLINDER";
            else if (proj.source_lens == core::screen_curve_type::sphere)
                lens_str = L"SPHERE";
            else if (proj.source_lens == core::screen_curve_type::fisheye)
                lens_str = L"FISHEYE";
            return L"201 MIXER OK\r\n" + lens_str + L"\r\n";
        });
    }

    using core::screen_curve_type;
    transforms_applier transforms(ctx);
    const auto&  lens_arg  = ctx.parameters.at(0);
    screen_curve_type lens = screen_curve_type::flat;  // flat = rectilinear
    if      (boost::iequals(lens_arg, L"CYLINDER")) lens = screen_curve_type::cylinder;
    else if (boost::iequals(lens_arg, L"SPHERE"))   lens = screen_curve_type::sphere;
    else if (boost::iequals(lens_arg, L"FISHEYE"))  lens = screen_curve_type::fisheye;

    transforms.add(stage::transform_tuple_t(
        ctx.layer_index(),
        [=](frame_transform transform) -> frame_transform {
            transform.image_transform.projection.source_lens = lens;
            return transform;
        },
        0,
        L"linear"));
    transforms.apply();

    return make_ready_future<std::wstring>(L"202 MIXER OK\r\n");
}

std::future<std::wstring> mixer_projection_icvfx_command(command_context& ctx)
{
    if (ctx.parameters.empty()) {
        auto transform2 = get_current_transform(ctx).share();
        return std::async(std::launch::deferred, [transform2]() -> std::wstring {
            auto proj = transform2.get().image_transform.projection;
            return L"201 MIXER OK\r\n" +
                   std::wstring(proj.icvfx_enable ? L"1" : L"0") + L" " +
                   std::to_wstring(proj.inner_fov) + L" " +
                   std::to_wstring(proj.icvfx_feather) + L" " +
                   std::to_wstring(proj.icvfx_outer_dim) + L" " +
                   std::to_wstring(proj.icvfx_inner_dim) + L"\r\n";
        });
    }

    // MIXER <ch>-<l> PROJECTION_ICVFX <enable> [inner_fov_rad] [feather] [outer_dim] [inner_dim] [dur] [tween]
    transforms_applier transforms(ctx);
    bool         enable    = (ctx.parameters.at(0) == L"1" ||
                              boost::iequals(ctx.parameters.at(0), L"true"));
    bool         has_fov   = ctx.parameters.size() > 1;
    double       inner_fov = has_fov ? std::stod(ctx.parameters[1]) : 0.0;
    bool         has_feat  = ctx.parameters.size() > 2;
    double       feather   = has_feat ? std::stod(ctx.parameters[2]) : 0.0;
    bool         has_dim   = ctx.parameters.size() > 3;
    double       outer_dim = has_dim ? std::stod(ctx.parameters[3]) : 1.0;
    bool         has_inner = ctx.parameters.size() > 4;
    double       inner_dim = has_inner ? std::stod(ctx.parameters[4]) : 1.0;
    int          duration  = ctx.parameters.size() > 5 ? std::stoi(ctx.parameters[5]) : 0;
    std::wstring tween     = ctx.parameters.size() > 6 ? ctx.parameters[6] : L"linear";

    transforms.add(stage::transform_tuple_t(
        ctx.layer_index(),
        [=](frame_transform transform) -> frame_transform {
            auto& p = transform.image_transform.projection;
            p.icvfx_enable = enable;
            if (has_fov && inner_fov > 0.0)
                p.inner_fov = inner_fov;
            if (has_feat)
                p.icvfx_feather = feather;
            if (has_dim)
                p.icvfx_outer_dim = outer_dim;
            if (has_inner)
                p.icvfx_inner_dim = inner_dim;
            return transform;
        },
        duration,
        tween));
    transforms.apply();

    return make_ready_future<std::wstring>(L"202 MIXER OK\r\n");
}

std::future<std::wstring> mixer_projection_icvfx_color_command(command_context& ctx)
{
    if (ctx.parameters.empty()) {
        auto transform2 = get_current_transform(ctx).share();
        return std::async(std::launch::deferred, [transform2]() -> std::wstring {
            auto proj = transform2.get().image_transform.projection;
            return L"201 MIXER OK\r\n" +
                   std::to_wstring(proj.icvfx_inner_gain_r) + L" " +
                   std::to_wstring(proj.icvfx_inner_gain_g) + L" " +
                   std::to_wstring(proj.icvfx_inner_gain_b) + L" " +
                   std::to_wstring(proj.icvfx_outer_gain_r) + L" " +
                   std::to_wstring(proj.icvfx_outer_gain_g) + L" " +
                   std::to_wstring(proj.icvfx_outer_gain_b) + L"\r\n";
        });
    }

    // MIXER <ch>-<l> PROJECTION_ICVFX_COLOR <ir> <ig> <ib> <or> <og> <ob> [dur] [tween]
    transforms_applier transforms(ctx);
    double       ir       = std::stod(ctx.parameters.at(0));
    double       ig       = std::stod(ctx.parameters.at(1));
    double       ib       = std::stod(ctx.parameters.at(2));
    double       orr      = std::stod(ctx.parameters.at(3));
    double       og       = std::stod(ctx.parameters.at(4));
    double       ob       = std::stod(ctx.parameters.at(5));
    int          duration = ctx.parameters.size() > 6 ? std::stoi(ctx.parameters[6]) : 0;
    std::wstring tween    = ctx.parameters.size() > 7 ? ctx.parameters[7] : L"linear";

    transforms.add(stage::transform_tuple_t(
        ctx.layer_index(),
        [=](frame_transform transform) -> frame_transform {
            auto& p = transform.image_transform.projection;
            p.icvfx_inner_gain_r = ir;
            p.icvfx_inner_gain_g = ig;
            p.icvfx_inner_gain_b = ib;
            p.icvfx_outer_gain_r = orr;
            p.icvfx_outer_gain_g = og;
            p.icvfx_outer_gain_b = ob;
            return transform;
        },
        duration,
        tween));
    transforms.apply();

    return make_ready_future<std::wstring>(L"202 MIXER OK\r\n");
}

std::future<std::wstring> mixer_projection_frustum_command(command_context& ctx)
{
    if (ctx.parameters.empty()) {
        auto transform2 = get_current_transform(ctx).share();
        return std::async(std::launch::deferred, [transform2]() -> std::wstring {
            auto proj = transform2.get().image_transform.projection;
            return L"201 MIXER OK\r\n" + std::to_wstring(proj.frustum_h) + L" " +
                   std::to_wstring(proj.frustum_v) + L"\r\n";
        });
    }

    transforms_applier transforms(ctx);
    int          duration  = ctx.parameters.size() > 2 ? std::stoi(ctx.parameters[2]) : 0;
    std::wstring tween     = ctx.parameters.size() > 3 ? ctx.parameters[3] : L"linear";
    double       frustum_h = std::stod(ctx.parameters.at(0));
    double       frustum_v = std::stod(ctx.parameters.at(1));

    transforms.add(stage::transform_tuple_t(
        ctx.layer_index(),
        [=](frame_transform transform) -> frame_transform {
            transform.image_transform.projection.frustum_h = frustum_h;
            transform.image_transform.projection.frustum_v = frustum_v;
            return transform;
        },
        duration,
        tween));
    transforms.apply();

    return make_ready_future<std::wstring>(L"202 MIXER OK\r\n");
}

std::future<std::wstring> mixer_projection_distortion_command(command_context& ctx)
{
    if (ctx.parameters.empty()) {
        auto transform2 = get_current_transform(ctx).share();
        return std::async(std::launch::deferred, [transform2]() -> std::wstring {
            auto proj = transform2.get().image_transform.projection;
            return L"201 MIXER OK\r\n" + std::to_wstring(proj.lens_k1) + L" " +
                   std::to_wstring(proj.lens_k2) + L" " + std::to_wstring(proj.lens_k3) + L" " +
                   std::to_wstring(proj.lens_p1) + L" " + std::to_wstring(proj.lens_p2) + L"\r\n";
        });
    }

    transforms_applier transforms(ctx);
    // Canonical form: k1 k2 k3 [p1 p2] [duration] [tween]
    double       k1       = std::stod(ctx.parameters.at(0));
    double       k2       = std::stod(ctx.parameters.at(1));
    double       k3       = std::stod(ctx.parameters.at(2));
    double       p1       = ctx.parameters.size() > 3 ? std::stod(ctx.parameters[3]) : 0.0;
    double       p2       = ctx.parameters.size() > 4 ? std::stod(ctx.parameters[4]) : 0.0;
    int          duration = ctx.parameters.size() > 5 ? std::stoi(ctx.parameters[5]) : 0;
    std::wstring tween    = ctx.parameters.size() > 6 ? ctx.parameters[6] : L"linear";

    transforms.add(stage::transform_tuple_t(
        ctx.layer_index(),
        [=](frame_transform transform) -> frame_transform {
            transform.image_transform.projection.lens_k1 = k1;
            transform.image_transform.projection.lens_k2 = k2;
            transform.image_transform.projection.lens_k3 = k3;
            transform.image_transform.projection.lens_p1 = p1;
            transform.image_transform.projection.lens_p2 = p2;
            return transform;
        },
        duration,
        tween));
    transforms.apply();

    return make_ready_future<std::wstring>(L"202 MIXER OK\r\n");
}

std::future<std::wstring> mixer_projection_blend_command(command_context& ctx)
{
    if (ctx.parameters.empty()) {
        auto transform2 = get_current_transform(ctx).share();
        return std::async(std::launch::deferred, [transform2]() -> std::wstring {
            auto proj = transform2.get().image_transform.projection;
            return L"201 MIXER OK\r\n" +
                   std::to_wstring(proj.edge_blend_left) + L" " +
                   std::to_wstring(proj.edge_blend_right) + L" " +
                   std::to_wstring(proj.edge_blend_top) + L" " +
                   std::to_wstring(proj.edge_blend_bottom) + L" " +
                   std::to_wstring(proj.edge_blend_gamma) + L"\r\n";
        });
    }

    transforms_applier transforms(ctx);
    double left   = std::stod(ctx.parameters.at(0));
    double right  = std::stod(ctx.parameters.at(1));
    double top    = ctx.parameters.size() > 2 ? std::stod(ctx.parameters[2]) : 0.0;
    double bottom = ctx.parameters.size() > 3 ? std::stod(ctx.parameters[3]) : 0.0;
    double gamma  = ctx.parameters.size() > 4 ? std::stod(ctx.parameters[4]) : 2.2;
    int    duration = ctx.parameters.size() > 5 ? std::stoi(ctx.parameters[5]) : 0;
    std::wstring tween = ctx.parameters.size() > 6 ? ctx.parameters[6] : L"linear";

    transforms.add(stage::transform_tuple_t(
        ctx.layer_index(),
        [=](frame_transform transform) -> frame_transform {
            transform.image_transform.projection.edge_blend_left   = left;
            transform.image_transform.projection.edge_blend_right  = right;
            transform.image_transform.projection.edge_blend_top    = top;
            transform.image_transform.projection.edge_blend_bottom = bottom;
            transform.image_transform.projection.edge_blend_gamma  = gamma;
            return transform;
        },
        duration,
        tween));
    transforms.apply();

    return make_ready_future<std::wstring>(L"202 MIXER OK\r\n");
}

std::future<std::wstring> mixer_mesh_command(command_context& ctx)
{
    // Query mode: MIXER ch-layer MESH
    if (ctx.parameters.empty()) {
        auto transform2 = get_current_transform(ctx).share();
        return std::async(std::launch::deferred, [transform2]() -> std::wstring {
            auto& t = transform2.get().image_transform;
            if (t.geometry_override.has_value()) {
                auto tri_count = t.geometry_override->data().size() / 3;
                return L"201 MIXER OK\r\nMESH " + std::to_wstring(tri_count) + L" triangles\r\n";
            }
            return L"201 MIXER OK\r\nNONE\r\n";
        });
    }

    // Clear mode: MIXER ch-layer MESH NONE
    if (boost::iequals(ctx.parameters.at(0), L"NONE")) {
        transforms_applier transforms(ctx);
        transforms.add(stage::transform_tuple_t(
            ctx.layer_index(),
            [](frame_transform transform) -> frame_transform {
                transform.image_transform.geometry_override.reset();
                return transform;
            },
            0,
            L"linear"));
        transforms.apply();

        return make_ready_future<std::wstring>(L"202 MIXER OK\r\n");
    }

    // Set mode: MIXER ch-layer MESH <path.glb|gltf|obj>
    auto mesh_path = ctx.parameters.at(0);

    // Resolve path relative to media folder (prevent path traversal)
    auto media_base = boost::filesystem::canonical(env::media_folder());
    auto resolved   = media_base / mesh_path;

    // Canonicalize and verify the path stays inside media folder
    if (!boost::filesystem::exists(resolved)) {
        return make_ready_future<std::wstring>(L"404 MIXER ERROR\r\n");
    }
    resolved = boost::filesystem::canonical(resolved);
    if (!is_within_base(resolved, media_base)) {
        return make_ready_future<std::wstring>(L"403 MIXER FORBIDDEN\r\n");
    }

    try {
        auto geometry = core::load_mesh(resolved.wstring());

        transforms_applier transforms(ctx);
        transforms.add(stage::transform_tuple_t(
            ctx.layer_index(),
            [geom = std::move(geometry)](frame_transform transform) mutable -> frame_transform {
                transform.image_transform.geometry_override = std::move(geom);
                return transform;
            },
            0,
            L"linear"));
        transforms.apply();

        return make_ready_future<std::wstring>(L"202 MIXER OK\r\n");
    } catch (const std::exception& e) {
        CASPAR_LOG(error) << L"[MIXER MESH] " << e.what();
        return make_ready_future<std::wstring>(L"502 MIXER FAILED\r\n");
    }
}

std::future<std::wstring> mixer_projection_blend_mask_command(command_context& ctx)
{
    // Query mode: MIXER ch-layer PROJECTION_BLEND_MASK
    if (ctx.parameters.empty()) {
        auto transform2 = get_current_transform(ctx).share();
        return std::async(std::launch::deferred, [transform2]() -> std::wstring {
            auto& mask = transform2.get().image_transform.blend_mask;
            if (mask && mask->width > 0 && mask->height > 0) {
                return L"201 MIXER OK\r\nMASK " + std::to_wstring(mask->width) + L"x" +
                       std::to_wstring(mask->height) + L"\r\n";
            }
            return L"201 MIXER OK\r\nNONE\r\n";
        });
    }

    // Clear mode: MIXER ch-layer PROJECTION_BLEND_MASK NONE
    if (boost::iequals(ctx.parameters.at(0), L"NONE")) {
        transforms_applier transforms(ctx);
        transforms.add(stage::transform_tuple_t(
            ctx.layer_index(),
            [](frame_transform transform) -> frame_transform {
                transform.image_transform.blend_mask.reset();
                return transform;
            },
            0,
            L"linear"));
        transforms.apply();

        return make_ready_future<std::wstring>(L"202 MIXER OK\r\n");
    }

    // Set mode: MIXER ch-layer PROJECTION_BLEND_MASK <path.png>
    auto mask_path = ctx.parameters.at(0);

    // Resolve path relative to media folder (prevent path traversal)
    auto media_base = boost::filesystem::canonical(env::media_folder());
    auto resolved   = media_base / mask_path;

    if (!boost::filesystem::exists(resolved)) {
        return make_ready_future<std::wstring>(L"404 MIXER ERROR\r\n");
    }
    resolved = boost::filesystem::canonical(resolved);
    if (!is_within_base(resolved, media_base)) {
        return make_ready_future<std::wstring>(L"403 MIXER FORBIDDEN\r\n");
    }

    try {
        auto mask = core::load_blend_mask(resolved.wstring());

        transforms_applier transforms(ctx);
        transforms.add(stage::transform_tuple_t(
            ctx.layer_index(),
            [mask = std::move(mask)](frame_transform transform) mutable -> frame_transform {
                transform.image_transform.blend_mask = std::move(mask);
                return transform;
            },
            0,
            L"linear"));
        transforms.apply();

        return make_ready_future<std::wstring>(L"202 MIXER OK\r\n");
    } catch (const std::exception& e) {
        CASPAR_LOG(error) << L"[MIXER PROJECTION_BLEND_MASK] " << e.what();
        return make_ready_future<std::wstring>(L"502 MIXER FAILED\r\n");
    }
}

std::future<std::wstring> mixer_flip_command(command_context& ctx)
{
    if (ctx.parameters.empty()) {
        auto transform2 = get_current_transform(ctx).share();
        return std::async(std::launch::deferred, [transform2]() -> std::wstring {
            auto& t = transform2.get().image_transform;
            std::wstring val = L"NONE";
            if (t.flip_h && t.flip_v) val = L"HV";
            else if (t.flip_h)        val = L"H";
            else if (t.flip_v)        val = L"V";
            return L"201 MIXER OK\r\n" + val + L"\r\n";
        });
    }

    bool flip_h = false;
    bool flip_v = false;
    const auto& arg = ctx.parameters.at(0);
    if (boost::iequals(arg, L"H"))                                    { flip_h = true; }
    else if (boost::iequals(arg, L"V"))                               { flip_v = true; }
    else if (boost::iequals(arg, L"HV") || boost::iequals(arg, L"VH")) { flip_h = true; flip_v = true; }
    else if (arg == L"1")                                             { flip_h = true; }  // 1 = H-flip (mirror)
    // else NONE / 0 / anything unrecognised -> both false

    transforms_applier transforms(ctx);
    transforms.add(stage::transform_tuple_t(
        ctx.layer_index(),
        [=](frame_transform transform) -> frame_transform {
            transform.image_transform.flip_h = flip_h;
            transform.image_transform.flip_v = flip_v;
            return transform;
        },
        0,
        tweener(L"linear")));
    transforms.apply();

    return make_ready_future<std::wstring>(L"202 MIXER OK\r\n");
}

// `LINEAR` is spelled out rather than being the fallback -- see reject_token. It is the
// transfer the test harness names for an EXR and for the manual path's linear input, so it
// has to be a token in its own right before anything else can be refused.
static int parse_transfer_fn(const std::wstring& s)
{
    if (boost::iequals(s, L"LINEAR")) return 0;
    if (boost::iequals(s, L"SRGB"))   return 1;
    if (boost::iequals(s, L"REC709")) return 2;
    if (boost::iequals(s, L"PQ"))     return 3;
    if (boost::iequals(s, L"HLG"))    return 4;
    if (boost::iequals(s, L"LOGC3"))  return 5;
    if (boost::iequals(s, L"SLOG3"))  return 6;
    reject_token(s, L"transfer function",
                 L"LINEAR, SRGB, REC709, PQ, HLG, LOGC3 or SLOG3");
}

// `REC709` is accepted alongside `BT709` because the two name the same primaries and both
// spellings appear in the wild. It is unambiguous even though `REC709` is also a transfer
// token: these parse different argument positions.
static int parse_gamut_fn(const std::wstring& s)
{
    if (boost::iequals(s, L"BT709") || boost::iequals(s, L"REC709")) return 0;
    if (boost::iequals(s, L"BT2020"))        return 1;
    if (boost::iequals(s, L"DCIP3"))         return 2;
    if (boost::iequals(s, L"ACES_AP0"))      return 3;
    if (boost::iequals(s, L"ACES_AP1"))      return 4;
    if (boost::iequals(s, L"ACESCG"))        return 4;
    if (boost::iequals(s, L"ARRI_WG3"))      return 5;
    if (boost::iequals(s, L"SGAMUT3_CINE"))  return 6;
    reject_token(s, L"gamut",
                 L"BT709, BT2020, DCIP3, ACES_AP0, ACES_AP1 (ACESCG), ARRI_WG3 or SGAMUT3_CINE");
}

// Every ACES operator here is ACES **1.x** -- `ACES_RRT` is Stephen Hill's approximation,
// `ACES_FILMIC` is Narkowicz's, and the three RRT+ODT operators use 1.x segmented splines.
// ACES 2.0 lives on the OCIO path (`OCIO_DISPLAY` with an ACES 2.0 view) and renders
// visibly differently; see COLOR_GRADING.md.
//
// The `ACES1_` spellings are ALIASES, accepted so a command or a show file can say which
// generation it meant. The query direction keeps emitting the original names: nothing in
// tree reads a COLORSPACE reply back, but an unknown client might, and making a name
// self-documenting is not worth changing what a query returns.
static int parse_tonemapping_fn(const std::wstring& s)
{
    if (boost::iequals(s, L"NONE"))            return 0;
    if (boost::iequals(s, L"REINHARD"))        return 1;
    if (boost::iequals(s, L"ACES_FILMIC") || boost::iequals(s, L"ACES1_FILMIC"))         return 2;
    if (boost::iequals(s, L"ACES_RRT") || boost::iequals(s, L"ACES1_RRT"))               return 3;
    if (boost::iequals(s, L"ACES_RRT_709") || boost::iequals(s, L"ACES1_RRT_709"))       return 4;
    if (boost::iequals(s, L"ACES_RRT_P3") || boost::iequals(s, L"ACES1_RRT_P3"))         return 5;
    if (boost::iequals(s, L"ACES_RRT_2020_PQ") || boost::iequals(s, L"ACES1_RRT_2020_PQ")) return 6;
    reject_token(s, L"tone mapping operator",
                 L"NONE, REINHARD, ACES_FILMIC, ACES_RRT, ACES_RRT_709, ACES_RRT_P3 or "
                 L"ACES_RRT_2020_PQ (each ACES operator also accepts an ACES1_ spelling)");
}

static std::wstring to_wstring_transfer(int t) {
    switch(t) {
        case 1: return L"SRGB";
        case 2: return L"REC709";
        case 3: return L"PQ";
        case 4: return L"HLG";
        case 5: return L"LOGC3";
        case 6: return L"SLOG3";
        default: return L"LINEAR";
    }
}
static std::wstring to_wstring_gamut(int g) {
    switch(g) {
        case 1: return L"BT2020";
        case 2: return L"DCIP3";
        case 3: return L"ACES_AP0";
        case 4: return L"ACES_AP1"; // ACEScg
        case 5: return L"ARRI_WG3";
        case 6: return L"SGAMUT3_CINE";
        default: return L"BT709";
    }
}
static std::wstring to_wstring_tonemap(int tm) {
    switch(tm) {
        case 1: return L"REINHARD";
        case 2: return L"ACES_FILMIC";
        case 3: return L"ACES_RRT";
        case 4: return L"ACES_RRT_709";
        case 5: return L"ACES_RRT_P3";
        case 6: return L"ACES_RRT_2020_PQ";
        default: return L"NONE";
    }
}

// MIXER <ch>-<layer> OCIO                  -> query
// MIXER <ch>-<layer> OCIO <source-space>   -> convert this layer's source encoding to the
//                                             mixer's ACEScg working space via OpenColorIO
// MIXER <ch>-<layer> OCIO NONE             -> back to the built-in path
//
// ⚠ QUOTE THE COLOUR SPACE NAME. AMCP tokenizes on whitespace, and 40 of the 55 spaces in
// the pinned studio config contain spaces or parentheses:
//
//     MIXER 1-1 OCIO "ARRI LogC3 (EI800)"     -> 202
//     MIXER 1-1 OCIO ARRI LogC3 (EI800)       -> 404, having looked for a space named "ARRI"
//
// The unquoted form fails cleanly rather than doing something surprising, but the 404 says
// nothing about quoting, so clients must quote and operators must be told to. The tokenizer
// already handles quotes; this needs no special support here.
//
// The alternative front end to MIXER COLORSPACE. Both write the same stage of the chain
// (COLOR_GRADING.md steps 4-7), so they are mutually exclusive and this command refuses
// rather than silently overriding -- an operator who set a COLORSPACE and then an OCIO space
// has contradicted themselves, and quietly picking one produces a look nobody chose. The
// same check exists in mixer_colorspace_command for the other direction.
//
// The creative grading tools in the middle of the chain are unaffected either way: they
// operate on scene-linear ACEScg regardless of which front end produced it, so a CDL or a
// curve keeps its meaning across this switch.
std::future<std::wstring> mixer_ocio_command(command_context& ctx)
{
    if (ctx.parameters.empty()) {
        auto transform2 = get_current_transform(ctx).share();
        return std::async(std::launch::deferred, [transform2]() -> std::wstring {
            const auto o = transform2.get().image_transform.ocio;
            return L"201 MIXER OK\r\n" + (o.enable ? u16(o.source_space) : std::wstring(L"NONE")) + L"\r\n";
        });
    }

    transforms_applier transforms(ctx);

    if (boost::iequals(ctx.parameters.at(0), L"NONE") || boost::iequals(ctx.parameters.at(0), L"OFF")) {
        transforms.add(stage::transform_tuple_t(
            ctx.layer_index(),
            [](frame_transform t) {
                t.image_transform.ocio.enable = false;
                t.image_transform.ocio.source_space.clear();
                t.image_transform.ocio.cache_id.clear();
                return t;
            },
            0,
            L"linear"));
        transforms.apply();
        return make_ready_future<std::wstring>(L"202 MIXER OK\r\n");
    }

    if (!accelerator::ocio::available()) {
        CASPAR_LOG(warning) << L"[ocio] MIXER OCIO refused: this server was built without OCIO support";
        return make_ready_future<std::wstring>(L"501 MIXER FAILED\r\n");
    }

    const auto source_space = u8(ctx.parameters.at(0));

    // Validated against the loaded config at command time so a wrong name fails the command,
    // never the frame. A colour space that does not exist must not become a black layer or a
    // silently untransformed one mid-show.
    if (!accelerator::ocio::has_colorspace(source_space)) {
        CASPAR_LOG(warning) << L"[ocio] MIXER OCIO refused: '" << ctx.parameters.at(0)
                            << L"' is not a colour space in " << u16(accelerator::ocio::config_uri())
                            << L". Use INFO OCIO COLORSPACES to list them.";
        return make_ready_future<std::wstring>(L"404 MIXER ERROR\r\n");
    }

    // Build the transform now, rather than only checking that the name exists. A space can
    // be present in the config and still fail to produce a processor -- a missing LUT file
    // referenced by a FileTransform, an unresolvable role -- and the difference matters,
    // because by the time the mixer needs it there is no way to report the failure except by
    // rendering something wrong. Discarding the result here is deliberate: this is
    // validation, and the mixer rebuilds it on the GL thread where the textures belong. The
    // second build is a cache hit inside OCIO.
    accelerator::ocio::gpu_shader probe;
    if (!accelerator::ocio::build_input_transform(source_space, probe)) {
        CASPAR_LOG(warning) << L"[ocio] MIXER OCIO refused: '" << ctx.parameters.at(0)
                            << L"' exists but no GPU transform could be built from it";
        return make_ready_future<std::wstring>(L"404 MIXER ERROR\r\n");
    }

    // Blocking read of the layer's current transform, to enforce the exclusion. Safe here:
    // the stage runs on its own thread, so there is no self-wait, and the query branch above
    // already depends on the same future.
    if (get_current_transform(ctx).get().image_transform.color_grade.enable) {
        CASPAR_LOG(warning) << L"[ocio] MIXER OCIO refused: MIXER COLORSPACE is active on this layer. "
                               L"They are mutually exclusive -- clear it with MIXER COLORSPACE NONE first.";
        return make_ready_future<std::wstring>(L"403 MIXER ERROR\r\n");
    }

    transforms.add(stage::transform_tuple_t(
        ctx.layer_index(),
        [source_space](frame_transform transform) -> frame_transform {
            auto& o = transform.image_transform.ocio;
            o.enable       = true;
            o.source_space = source_space;
            // cache_id is filled in by the mixer once it has built the processor: only it
            // can ask OCIO, and the value depends on the config's contents rather than on
            // the name typed here.
            o.cache_id.clear();
            return transform;
        },
        0,
        L"linear"));
    transforms.apply();

    // Same pre-warm as OCIO_DISPLAY, and for the same reason: the processor is built above
    // for validation, but the GPU program -- LUT upload, GLSL compile, driver pipeline
    // build -- happens on the first draw unless it is asked for here. ~1.2 s and one
    // dropped frame, on the tick where an operator selects a look.
    ctx.channel.raw_channel->mixer().get_image_mixer()->prewarm_ocio(source_space, "", "");

    return make_ready_future<std::wstring>(L"202 MIXER OK\r\n");
}

std::future<std::wstring> mixer_colorspace_command(command_context& ctx)
{
    if (ctx.parameters.empty()) {
        auto transform2 = get_current_transform(ctx).share();
        return std::async(std::launch::deferred, [transform2]() -> std::wstring {
            auto cg = transform2.get().image_transform.color_grade;
            return L"201 MIXER OK\r\n" +
                   (cg.enable ? (
                       to_wstring_transfer(cg.input_transfer) + L" " + 
                       to_wstring_gamut(cg.input_gamut) + L" " +
                       to_wstring_tonemap(cg.tone_mapping) + L" " +
                       to_wstring_gamut(cg.output_gamut) + L" " +
                       to_wstring_transfer(cg.output_transfer) + L" " + 
                       std::to_wstring(cg.exposure)
                   ) : std::wstring(L"NONE")) + L"\r\n";
        });
    }

    // MIXER 1-1 COLORSPACE [input_transfer] [input_gamut] [tonemapping] [output_gamut] [output_transfer] [exposure]
    // Disable with: MIXER 1-1 COLORSPACE NONE
    transforms_applier transforms(ctx);

    // Mutually exclusive with MIXER OCIO -- see mixer_ocio_command. Refusing NONE would be
    // unhelpful, so only an enabling form is blocked.
    if (!boost::iequals(ctx.parameters.at(0), L"NONE") &&
        get_current_transform(ctx).get().image_transform.ocio.enable) {
        CASPAR_LOG(warning) << L"[ocio] MIXER COLORSPACE refused: MIXER OCIO is active on this layer. "
                               L"They are mutually exclusive -- clear it with MIXER OCIO NONE first.";
        return make_ready_future<std::wstring>(L"403 MIXER ERROR\r\n");
    }

    if (boost::iequals(ctx.parameters.at(0), L"NONE")) {
        transforms.add(stage::transform_tuple_t(
            ctx.layer_index(),
            [](frame_transform t) { t.image_transform.color_grade.enable = false; return t; },
            0,
            L"linear"));
        transforms.apply();
        return make_ready_future<std::wstring>(L"202 MIXER OK\r\n");
    }

    int   it       = parse_transfer_fn(ctx.parameters.at(0));
    int   ig       = ctx.parameters.size() > 1 ? parse_gamut_fn(ctx.parameters.at(1))       : 0;
    int   tm       = ctx.parameters.size() > 2 ? parse_tonemapping_fn(ctx.parameters.at(2)) : 0;
    int   og       = ctx.parameters.size() > 3 ? parse_gamut_fn(ctx.parameters.at(3))       : 0;
    int   ot       = ctx.parameters.size() > 4 ? parse_transfer_fn(ctx.parameters.at(4))    : 1;
    float exposure = ctx.parameters.size() > 5
                         ? static_cast<float>(grade_param(ctx.parameters.at(5),
                                                          core::grade_limits::exposure, L"exposure"))
                         : 1.0f;

    transforms.add(stage::transform_tuple_t(
        ctx.layer_index(),
        [=](frame_transform transform) -> frame_transform {
            auto& cg           = transform.image_transform.color_grade;
            cg.enable          = true;
            cg.input_transfer  = it;
            cg.input_gamut     = ig;
            cg.tone_mapping    = tm;
            cg.output_gamut    = og;
            cg.output_transfer = ot;
            cg.exposure        = exposure;
            return transform;
        },
        0,
        L"linear"));
    transforms.apply();

    return make_ready_future<std::wstring>(L"202 MIXER OK\r\n");
}

// ---------- Per-channel triple-value animatable helper ----------------------

/// `range` and `name` are required here: this helper serves only LIFT, MIDTONE and GAIN,
/// all three of which are in the validated set. They need different ranges rather than a
/// shared one -- lift is an offset, midtone an exponent, gain a multiplier -- so the
/// range is threaded through rather than assumed.
template <typename Getter, typename Setter>
std::future<std::wstring> triple_double_animatable_mixer_command(command_context&  ctx,
                                                                  const Getter&    getter,
                                                                  const Setter&    setter,
                                                                  core::grade_range range,
                                                                  const wchar_t*    name)
{
    if (ctx.parameters.empty()) {
        auto transform2 = get_current_transform(ctx).share();
        return std::async(std::launch::deferred, [transform2, getter]() -> std::wstring {
            auto arr = getter(transform2.get());
            return L"201 MIXER OK\r\n" +
                   std::to_wstring(arr[0]) + L" " +
                   std::to_wstring(arr[1]) + L" " +
                   std::to_wstring(arr[2]) + L"\r\n";
        });
    }

    grade_require(ctx, 3, L"expects r g b [duration] [tween]");
    transforms_applier transforms(ctx);
    double       r        = grade_param(ctx.parameters.at(0), range, name);
    double       g        = grade_param(ctx.parameters.at(1), range, name);
    double       b        = grade_param(ctx.parameters.at(2), range, name);
    int          duration = ctx.parameters.size() > 3 ? std::stoi(ctx.parameters[3]) : 0;
    std::wstring tween    = ctx.parameters.size() > 4 ? ctx.parameters[4] : L"linear";

    transforms.add(stage::transform_tuple_t(
        ctx.layer_index(),
        [=](frame_transform transform) -> frame_transform {
            setter(transform, r, g, b);
            return transform;
        },
        duration,
        tween));
    transforms.apply();

    return make_ready_future<std::wstring>(L"202 MIXER OK\r\n");
}

// ---------- New color-grading commands (DaVinci Resolve-style) ---------------

std::future<std::wstring> mixer_whitebalance_command(command_context& ctx)
{
    if (ctx.parameters.empty()) {
        auto transform2 = get_current_transform(ctx).share();
        return std::async(std::launch::deferred, [transform2]() -> std::wstring {
            auto t = transform2.get().image_transform;
            return L"201 MIXER OK\r\n" + std::to_wstring(t.temperature) + L" " +
                   std::to_wstring(t.tint) + L"\r\n";
        });
    }

    grade_require(ctx, 2, L"MIXER WHITEBALANCE temperature tint [duration] [tween]");
    transforms_applier transforms(ctx);
    double temperature = grade_param(ctx.parameters.at(0), core::grade_limits::temperature, L"temperature");
    double tint        = grade_param(ctx.parameters.at(1), core::grade_limits::tint, L"tint");
    int          duration    = ctx.parameters.size() > 2 ? std::stoi(ctx.parameters[2]) : 0;
    std::wstring tween       = ctx.parameters.size() > 3 ? ctx.parameters[3] : L"linear";

    transforms.add(stage::transform_tuple_t(
        ctx.layer_index(),
        [=](frame_transform transform) -> frame_transform {
            transform.image_transform.temperature = temperature;
            transform.image_transform.tint        = tint;
            return transform;
        },
        duration,
        tween));
    transforms.apply();

    return make_ready_future<std::wstring>(L"202 MIXER OK\r\n");
}

// MIXER LIFT r g b [duration tween]  -- per-channel shadow offset (-0.5..+0.5)
std::future<std::wstring> mixer_lift_command(command_context& ctx)
{
    return triple_double_animatable_mixer_command(
        ctx,
        [](const frame_transform& t) { return t.image_transform.lift; },
        [](frame_transform& t, double r, double g, double b) {
            t.image_transform.lift = {r, g, b};
        },
        core::grade_limits::lift,
        L"lift");
}

// MIXER MIDTONE r g b [duration tween]  -- per-channel midtone power (0.1..4, DaVinci "Gamma" wheel)
std::future<std::wstring> mixer_midtone_command(command_context& ctx)
{
    return triple_double_animatable_mixer_command(
        ctx,
        [](const frame_transform& t) { return t.image_transform.midtone; },
        [](frame_transform& t, double r, double g, double b) {
            t.image_transform.midtone = {r, g, b};
        },
        core::grade_limits::midtone,
        L"midtone");
}

// MIXER GAIN r g b [duration tween]  -- per-channel highlight multiplier (0..4, DaVinci "Gain" wheel)
std::future<std::wstring> mixer_gain_command(command_context& ctx)
{
    return triple_double_animatable_mixer_command(
        ctx,
        [](const frame_transform& t) { return t.image_transform.gain; },
        [](frame_transform& t, double r, double g, double b) {
            t.image_transform.gain = {r, g, b};
        },
        core::grade_limits::gain,
        L"gain");
}

// MIXER HUESHIFT degrees [duration tween]  -- global hue rotation (-180..+180)
std::future<std::wstring> mixer_hueshift_command(command_context& ctx)
{
    return single_double_animatable_mixer_command(
        ctx,
        [](const frame_transform& t) { return t.image_transform.hue_shift; },
        [](frame_transform& t, double value) { t.image_transform.hue_shift = value; },
        core::grade_limits::hue_shift,
        L"hue shift");
}

// MIXER LINEARSATURATION val [duration tween]  -- scene-linear saturation (0=mono, 1=normal, >1=boost)
std::future<std::wstring> mixer_linearsaturation_command(command_context& ctx)
{
    return single_double_animatable_mixer_command(
        ctx,
        [](const frame_transform& t) { return t.image_transform.linear_saturation; },
        [](frame_transform& t, double value) { t.image_transform.linear_saturation = value; },
        core::grade_limits::cdl_saturation,
        L"linear saturation");
}

// MIXER CDL sR sG sB oR oG oB pR pG pB [sat] [duration tween]  -- ASC CDL
std::future<std::wstring> mixer_cdl_command(command_context& ctx)
{
    if (ctx.parameters.empty()) {
        auto transform2 = get_current_transform(ctx).share();
        return std::async(std::launch::deferred, [transform2]() -> std::wstring {
            auto t = transform2.get().image_transform;
            auto f = [](double v) { return std::to_wstring(v); };
            return L"201 MIXER OK\r\n" +
                   f(t.cdl_slope[0])  + L" " + f(t.cdl_slope[1])  + L" " + f(t.cdl_slope[2])  + L" " +
                   f(t.cdl_offset[0]) + L" " + f(t.cdl_offset[1]) + L" " + f(t.cdl_offset[2]) + L" " +
                   f(t.cdl_power[0])  + L" " + f(t.cdl_power[1])  + L" " + f(t.cdl_power[2])  + L" " +
                   f(t.cdl_saturation) + L"\r\n";
        });
    }

    if (boost::iequals(ctx.parameters.at(0), L"RESET")) {
        transforms_applier transforms(ctx);
        transforms.add(stage::transform_tuple_t(
            ctx.layer_index(),
            [](frame_transform t) {
                t.image_transform.cdl_slope      = {1.0, 1.0, 1.0};
                t.image_transform.cdl_offset     = {0.0, 0.0, 0.0};
                t.image_transform.cdl_power      = {1.0, 1.0, 1.0};
                t.image_transform.cdl_saturation = 1.0;
                return t;
            },
            0, L"linear"));
        transforms.apply();
        return make_ready_future<std::wstring>(L"202 MIXER OK\r\n");
    }

    // ASC CDL requires slope >= 0, power > 0 and saturation >= 0; the ranges follow the
    // standard rather than taste.
    grade_require(ctx, 9, L"MIXER CDL sR sG sB oR oG oB pR pG pB [sat] [duration] [tween]");
    transforms_applier transforms(ctx);
    double sR = grade_param(ctx.parameters.at(0), core::grade_limits::cdl_slope, L"slope");
    double sG = grade_param(ctx.parameters.at(1), core::grade_limits::cdl_slope, L"slope");
    double sB = grade_param(ctx.parameters.at(2), core::grade_limits::cdl_slope, L"slope");
    double oR = grade_param(ctx.parameters.at(3), core::grade_limits::cdl_offset, L"offset");
    double oG = grade_param(ctx.parameters.at(4), core::grade_limits::cdl_offset, L"offset");
    double oB = grade_param(ctx.parameters.at(5), core::grade_limits::cdl_offset, L"offset");
    double pR = grade_param(ctx.parameters.at(6), core::grade_limits::cdl_power, L"power");
    double pG = grade_param(ctx.parameters.at(7), core::grade_limits::cdl_power, L"power");
    double pB = grade_param(ctx.parameters.at(8), core::grade_limits::cdl_power, L"power");
    double sat = ctx.parameters.size() > 9
                     ? grade_param(ctx.parameters[9], core::grade_limits::cdl_saturation, L"saturation")
                     : 1.0;
    int    dur = ctx.parameters.size() > 10 ? std::stoi(ctx.parameters[10]) : 0;
    std::wstring tw = ctx.parameters.size() > 11 ? ctx.parameters[11] : L"linear";

    transforms.add(stage::transform_tuple_t(
        ctx.layer_index(),
        [=](frame_transform transform) -> frame_transform {
            transform.image_transform.cdl_slope      = {sR, sG, sB};
            transform.image_transform.cdl_offset     = {oR, oG, oB};
            transform.image_transform.cdl_power      = {pR, pG, pB};
            transform.image_transform.cdl_saturation = sat;
            return transform;
        },
        dur,
        tw));
    transforms.apply();
    return make_ready_future<std::wstring>(L"202 MIXER OK\r\n");
}

// MIXER <ch>-<layer> CDL_FILE "<path>" ["<id>"] [duration] [tween]
//
// Load an ASC CDL grade from a `.cdl`, `.ccc` or `.cc` file. `<id>` selects one correction
// from a collection by its `id` attribute; omit it when the file holds exactly one.
//
// THE FILE IS PARSED, NOT INTERPRETED. It sets exactly the state `MIXER CDL` sets, so the
// grade runs in the same shader block and the two commands are required to render
// identically -- which is how the harness gates this, without needing a colour model.
//
// The path is tried as given and then under <media-path>, matching CALIBRATION LUT — so an
// operator does not have to learn two rules for the two file-taking commands.
std::future<std::wstring> mixer_cdl_file_command(command_context& ctx)
{
    if (!accelerator::ocio::available()) {
        CASPAR_LOG(warning) << L"[ocio] MIXER CDL_FILE refused: this server was built without OCIO "
                               L"support, which is what parses ASC CDL files";
        return make_ready_future<std::wstring>(L"501 MIXER FAILED\r\n");
    }

    if (ctx.parameters.empty()) {
        CASPAR_LOG(warning) << L"[ocio] MIXER CDL_FILE needs a path. "
                               L"MIXER <ch>-<layer> CDL_FILE \"<file.cdl>\" [\"<id>\"]";
        return make_ready_future<std::wstring>(L"400 MIXER FAILED\r\n");
    }

    // As-is first, then under <media-path>: the same resolution CALIBRATION LUT performs.
    std::wstring path = ctx.parameters.at(0);
    if (!std::ifstream(path).is_open())
        path = caspar::env::media_folder() + L"/" + path;

    // A second parameter that parses as a number is a duration, not an id: `MIXER CDL`'s
    // trailing [duration] [tween] are the convention here, and an ASC id is a string.
    std::wstring id;
    size_t       tail = 1;
    if (ctx.parameters.size() > 1) {
        try {
            std::stoi(ctx.parameters.at(1));
        } catch (...) {
            id   = ctx.parameters.at(1);
            tail = 2;
        }
    }

    double slope[3]{1.0, 1.0, 1.0};
    double offset[3]{0.0, 0.0, 0.0};
    double power[3]{1.0, 1.0, 1.0};
    double sat = 1.0;
    if (!accelerator::ocio::load_cdl(u8(path), u8(id), slope, offset, power, sat)) {
        return make_ready_future<std::wstring>(L"404 MIXER ERROR\r\n");
    }

    // Clamped through the SAME limits MIXER CDL uses. A file is operator-supplied and can
    // hold a negative slope or a zero power, and the shader's guarantees are written against
    // those ranges -- so a file cannot reach a state the numeric command refuses.
    const double sR = core::grade_limits::cdl_slope.clamp(slope[0]);
    const double sG = core::grade_limits::cdl_slope.clamp(slope[1]);
    const double sB = core::grade_limits::cdl_slope.clamp(slope[2]);
    const double oR = core::grade_limits::cdl_offset.clamp(offset[0]);
    const double oG = core::grade_limits::cdl_offset.clamp(offset[1]);
    const double oB = core::grade_limits::cdl_offset.clamp(offset[2]);
    const double pR = core::grade_limits::cdl_power.clamp(power[0]);
    const double pG = core::grade_limits::cdl_power.clamp(power[1]);
    const double pB = core::grade_limits::cdl_power.clamp(power[2]);
    const double st = core::grade_limits::cdl_saturation.clamp(sat);

    int          dur = ctx.parameters.size() > tail ? std::stoi(ctx.parameters[tail]) : 0;
    std::wstring tw  = ctx.parameters.size() > tail + 1 ? ctx.parameters[tail + 1] : L"linear";

    transforms_applier transforms(ctx);
    transforms.add(stage::transform_tuple_t(
        ctx.layer_index(),
        [=](frame_transform transform) -> frame_transform {
            transform.image_transform.cdl_slope      = {sR, sG, sB};
            transform.image_transform.cdl_offset     = {oR, oG, oB};
            transform.image_transform.cdl_power      = {pR, pG, pB};
            transform.image_transform.cdl_saturation = st;
            return transform;
        },
        dur,
        tw));
    transforms.apply();
    return make_ready_future<std::wstring>(L"202 MIXER OK\r\n");
}

// MIXER SPLITTONE shad_r shad_g shad_b hi_r hi_g hi_b [balance] [duration tween]
std::future<std::wstring> mixer_splittone_command(command_context& ctx)
{
    if (ctx.parameters.empty()) {
        auto transform2 = get_current_transform(ctx).share();
        return std::async(std::launch::deferred, [transform2]() -> std::wstring {
            auto t = transform2.get().image_transform;
            auto f = [](double v) { return std::to_wstring(v); };
            return L"201 MIXER OK\r\n" +
                   f(t.split_shadow_color[0])    + L" " + f(t.split_shadow_color[1])    + L" " + f(t.split_shadow_color[2])    + L" " +
                   f(t.split_highlight_color[0]) + L" " + f(t.split_highlight_color[1]) + L" " + f(t.split_highlight_color[2]) + L" " +
                   f(t.split_balance) + L"\r\n";
        });
    }

    if (boost::iequals(ctx.parameters.at(0), L"RESET")) {
        transforms_applier transforms(ctx);
        transforms.add(stage::transform_tuple_t(
            ctx.layer_index(),
            [](frame_transform t) {
                t.image_transform.split_shadow_color    = {0.0, 0.0, 0.0};
                t.image_transform.split_highlight_color = {0.0, 0.0, 0.0};
                t.image_transform.split_balance         = 0.5;
                return t;
            },
            0, L"linear"));
        transforms.apply();
        return make_ready_future<std::wstring>(L"202 MIXER OK\r\n");
    }

    grade_require(ctx, 6, L"MIXER SPLITTONE sR sG sB hR hG hB [balance] [duration] [tween]");
    transforms_applier transforms(ctx);
    double sr = grade_param(ctx.parameters.at(0), core::grade_limits::split_color, L"shadow colour");
    double sg = grade_param(ctx.parameters.at(1), core::grade_limits::split_color, L"shadow colour");
    double sb = grade_param(ctx.parameters.at(2), core::grade_limits::split_color, L"shadow colour");
    double hr = grade_param(ctx.parameters.at(3), core::grade_limits::split_color, L"highlight colour");
    double hg = grade_param(ctx.parameters.at(4), core::grade_limits::split_color, L"highlight colour");
    double hb = grade_param(ctx.parameters.at(5), core::grade_limits::split_color, L"highlight colour");
    double bal = ctx.parameters.size() > 6
                     ? grade_param(ctx.parameters[6], core::grade_limits::split_balance, L"balance")
                     : 0.5;
    int    dur = ctx.parameters.size() > 7  ? std::stoi(ctx.parameters[7])  : 0;
    std::wstring tw = ctx.parameters.size() > 8 ? ctx.parameters[8] : L"linear";

    transforms.add(stage::transform_tuple_t(
        ctx.layer_index(),
        [=](frame_transform transform) -> frame_transform {
            transform.image_transform.split_shadow_color    = {sr, sg, sb};
            transform.image_transform.split_highlight_color = {hr, hg, hb};
            transform.image_transform.split_balance         = bal;
            return transform;
        },
        dur,
        tw));
    transforms.apply();
    return make_ready_future<std::wstring>(L"202 MIXER OK\r\n");
}

// MIXER EXPOSURE <stops-as-linear-gain>
// MIXER EXPOSURE — query
//
// A linear gain in the WORKING space, applied after the conversion into it and before the
// grade. Distinct from `MIXER COLORSPACE`'s 6th argument, which lives in the color_grade
// struct and is mutually exclusive with `MIXER OCIO`; this one applies on any route into
// the working space, so it is the only exposure an OCIO layer can be given. Where both are
// set they multiply -- see image_transform::exposure.
//
// Inert on a layer that reaches no working space, for the same reason MIXER GAMUTCOMPRESS
// is: an un-converted pixel is still display-encoded and a "linear" gain on it is not a
// linear gain on light.
std::future<std::wstring> mixer_exposure_command(command_context& ctx)
{
    if (ctx.parameters.empty()) {
        auto transform2 = get_current_transform(ctx).share();
        return std::async(std::launch::deferred, [transform2]() -> std::wstring {
            return L"201 MIXER OK\r\n" +
                   std::to_wstring(transform2.get().image_transform.exposure) + L"\r\n";
        });
    }

    grade_require(ctx, 1, L"MIXER EXPOSURE gain [duration] [tween]");
    transforms_applier transforms(ctx);
    // Was a hand-rolled finite/non-negative test, which was right as far as it went but
    // had no upper bound. It has to share `grade_limits::exposure` with the combine-side
    // clamp: a command that accepts 1e6 while composition clamps to 16 makes a value
    // reachable by stacking that no single command would set, which is the mismatch the
    // one-table design exists to prevent. NaN is still refused -- `contains` is a positive
    // test, so NaN falls outside it for the same reason `>= 0.0` caught it here.
    double value    = grade_param(ctx.parameters.at(0), core::grade_limits::exposure, L"exposure");
    int  duration = ctx.parameters.size() > 1 ? std::stoi(ctx.parameters[1]) : 0;
    auto tween    = ctx.parameters.size() > 2 ? ctx.parameters[2] : L"linear";

    transforms.add(stage::transform_tuple_t(
        ctx.layer_index(),
        [=](frame_transform transform) -> frame_transform {
            transform.image_transform.exposure = value;
            return transform;
        },
        duration,
        tween));
    transforms.apply();
    return make_ready_future<std::wstring>(L"202 MIXER OK\r\n");
}

// MIXER GAMUTCOMPRESS <0|1> [cyan_limit] [magenta_limit] [yellow_limit]
std::future<std::wstring> mixer_gamutcompress_command(command_context& ctx)
{
    if (ctx.parameters.empty()) {
        auto transform2 = get_current_transform(ctx).share();
        return std::async(std::launch::deferred, [transform2]() -> std::wstring {
            auto t = transform2.get().image_transform;
            return L"201 MIXER OK\r\n" +
                   std::to_wstring(t.gamut_compress ? 1 : 0) + L" " +
                   std::to_wstring(t.gc_cyan)    + L" " +
                   std::to_wstring(t.gc_magenta) + L" " +
                   std::to_wstring(t.gc_yellow)  + L"\r\n";
        });
    }

    transforms_applier transforms(ctx);
    bool   enable  = std::stoi(ctx.parameters.at(0)) != 0;
    double cyan    = ctx.parameters.size() > 1
                         ? grade_param(ctx.parameters[1], core::grade_limits::gamut_limit, L"cyan limit")
                         : 1.147;
    double magenta = ctx.parameters.size() > 2
                         ? grade_param(ctx.parameters[2], core::grade_limits::gamut_limit, L"magenta limit")
                         : 1.264;
    double yellow  = ctx.parameters.size() > 3
                         ? grade_param(ctx.parameters[3], core::grade_limits::gamut_limit, L"yellow limit")
                         : 1.312;

    transforms.add(stage::transform_tuple_t(
        ctx.layer_index(),
        [=](frame_transform transform) -> frame_transform {
            transform.image_transform.gamut_compress = enable;
            transform.image_transform.gc_cyan        = cyan;
            transform.image_transform.gc_magenta     = magenta;
            transform.image_transform.gc_yellow      = yellow;
            return transform;
        },
        0,
        L"linear"));
    transforms.apply();
    return make_ready_future<std::wstring>(L"202 MIXER OK\r\n");
}

// MIXER LUT3D <path.cube> [strength] — load a .cube 3D LUT file
// MIXER LUT3D NONE — disable 3D LUT
// MIXER LUT3D — query current state
// The largest LUT_3D_SIZE we will accept. 128 costs 128^3 * 3 floats = 25 MB, which is
// already far beyond anything a grading tool emits (Resolve tops out at 65); the point of
// the cap is that `size` arrives from the file and is cubed, so an unbounded value is a
// memory-exhaustion primitive. "LUT_3D_SIZE 2000" used to ask reserve() for ~96 GB. The
// AMCP dispatcher does catch the resulting bad_alloc, so this was never a crash -- but a
// live server could be walked into an allocation storm by a single malformed file.
static constexpr int MAX_LUT_3D_SIZE = 128;

// LUT strength arrives as an operator-typed AMCP argument. std::stof throws on anything
// non-numeric, and neither LUT call site handled it -- the AMCP dispatcher's catch-all
// turned "MIXER 1-1 LUT3D my.cube abc" into an opaque failure with no indication of which
// argument was at fault. Returns false so the caller can answer 400 instead.
static bool parse_lut_strength(const std::wstring& text, float& out)
{
    try {
        size_t consumed = 0;
        const float value = std::stof(text, &consumed);
        if (consumed != text.size())
            return false;
        out = std::clamp(value, 0.0f, 1.0f);
        return true;
    } catch (...) {
        return false;
    }
}

static std::shared_ptr<const core::lut3d_data> parse_cube_file(const std::wstring& path)
{
    std::ifstream file(path);
    if (!file.is_open())
        return nullptr;

    auto        lut = std::make_shared<core::lut3d_data>();
    std::string line;

    // DOMAIN_MIN/DOMAIN_MAX declare the input range the table is indexed over. Both
    // shaders index with clamp(c, 0, 1), so a non-unit domain would need the lookup
    // coordinate rescaled before the fetch -- it cannot be baked into the table data.
    // These lines used to be skipped outright, which silently misapplied every LUT
    // authored over a non-unit domain (routine for log LUTs). Refusing is not a
    // regression: such a LUT was never being honoured, it was being applied wrongly.
    float domain_min[3] = {0.0f, 0.0f, 0.0f};
    float domain_max[3] = {1.0f, 1.0f, 1.0f};

    while (std::getline(file, line)) {
        // Skip comments and empty lines
        if (line.empty() || line[0] == '#')
            continue;
        // Remove leading whitespace
        size_t start = line.find_first_not_of(" \t\r\n");
        if (start == std::string::npos)
            continue;
        line = line.substr(start);

        if (line.rfind("TITLE", 0) == 0)
            continue;

        if (line.rfind("DOMAIN_MIN", 0) == 0) {
            if (sscanf(line.c_str(), "DOMAIN_MIN %f %f %f", &domain_min[0], &domain_min[1], &domain_min[2]) != 3) {
                CASPAR_LOG(warning) << L"[lut3d] " << path << L": malformed DOMAIN_MIN";
                return nullptr;
            }
            continue;
        }
        if (line.rfind("DOMAIN_MAX", 0) == 0) {
            if (sscanf(line.c_str(), "DOMAIN_MAX %f %f %f", &domain_max[0], &domain_max[1], &domain_max[2]) != 3) {
                CASPAR_LOG(warning) << L"[lut3d] " << path << L": malformed DOMAIN_MAX";
                return nullptr;
            }
            continue;
        }

        if (line.rfind("LUT_3D_SIZE", 0) == 0) {
            if (lut->size != 0) {
                CASPAR_LOG(warning) << L"[lut3d] " << path << L": more than one LUT_3D_SIZE";
                return nullptr;
            }

            int size = 0;
            if (sscanf(line.c_str(), "LUT_3D_SIZE %d", &size) != 1) {
                CASPAR_LOG(warning) << L"[lut3d] " << path << L": malformed LUT_3D_SIZE";
                return nullptr;
            }
            if (size < 2 || size > MAX_LUT_3D_SIZE) {
                CASPAR_LOG(warning) << L"[lut3d] " << path << L": LUT_3D_SIZE " << size
                                    << L" out of range (2.." << MAX_LUT_3D_SIZE << L")";
                return nullptr;
            }

            lut->size = size;
            lut->data.reserve(static_cast<size_t>(size) * size * size * 3);
            continue;
        }
        if (line.rfind("LUT_1D_SIZE", 0) == 0)
            continue; // skip 1D LUT sections

        // Try to parse as R G B data line
        if (lut->size > 0) {
            float r, g, b;
            if (sscanf(line.c_str(), "%f %f %f", &r, &g, &b) == 3) {
                lut->data.push_back(r);
                lut->data.push_back(g);
                lut->data.push_back(b);
            }
        }
    }

    for (int i = 0; i < 3; ++i) {
        if (domain_min[i] != 0.0f || domain_max[i] != 1.0f) {
            CASPAR_LOG(warning) << L"[lut3d] " << path
                                << L": non-unit domain is not supported -- the lookup is indexed over [0,1]. "
                                   L"Re-export the LUT over a 0..1 domain.";
            return nullptr;
        }
    }

    size_t expected = static_cast<size_t>(lut->size) * lut->size * lut->size * 3;
    if (lut->size <= 0 || lut->data.size() != expected) {
        CASPAR_LOG(warning) << L"[lut3d] " << path << L": expected " << expected << L" values, read "
                            << lut->data.size();
        return nullptr;
    }

    return lut;
}

std::future<std::wstring> mixer_lut3d_command(command_context& ctx)
{
    if (ctx.parameters.empty()) {
        auto transform2 = get_current_transform(ctx).share();
        return std::async(std::launch::deferred, [transform2]() -> std::wstring {
            auto t = transform2.get().image_transform;
            if (!t.lut3d)
                return L"201 MIXER OK\r\nNONE\r\n";
            return L"201 MIXER OK\r\nACTIVE " + std::to_wstring(t.lut3d->size) +
                   L" " + std::to_wstring(t.lut3d_strength) + L"\r\n";
        });
    }

    if (boost::iequals(ctx.parameters.at(0), L"NONE")) {
        transforms_applier transforms(ctx);
        transforms.add(stage::transform_tuple_t(
            ctx.layer_index(),
            [](frame_transform t) {
                t.image_transform.lut3d          = nullptr;
                t.image_transform.lut3d_strength = 1.0f;
                return t;
            },
            0, L"linear"));
        transforms.apply();
        return make_ready_future<std::wstring>(L"202 MIXER OK\r\n");
    }

    // Resolve path: try as-is first, then relative to media folder
    std::wstring path = ctx.parameters.at(0);
    if (!std::ifstream(path).is_open()) {
        auto media = caspar::env::media_folder();
        path = media + L"/" + path;
    }

    auto lut = parse_cube_file(path);
    if (!lut)
        return make_ready_future<std::wstring>(L"404 LUT3D LOAD FAILED\r\n");

    float strength = 1.0f;
    if (ctx.parameters.size() > 1 && !parse_lut_strength(ctx.parameters[1], strength))
        return make_ready_future<std::wstring>(L"400 MIXER FAILED\r\n");

    transforms_applier transforms(ctx);
    transforms.add(stage::transform_tuple_t(
        ctx.layer_index(),
        [=](frame_transform transform) -> frame_transform {
            transform.image_transform.lut3d          = lut;
            transform.image_transform.lut3d_strength = strength;
            return transform;
        },
        0, L"linear"));
    transforms.apply();
    return make_ready_future<std::wstring>(L"202 MIXER OK\r\n");
}

// OCIO_DISPLAY <channel> "<display>" "<view>"   set the channel's display transform
// OCIO_DISPLAY <channel> NONE                    clear it
// OCIO_DISPLAY <channel> [INFO]                  query
//
// CHANNEL-LEVEL, and unlike `MIXER OCIO` that is not a simplification. An INPUT transform
// describes where the pixels came from, which is a property of each layer. A DISPLAY
// transform describes what screen they are going to, and every layer in a channel goes to
// the same screen -- two layers with different display transforms would blend a PQ-encoded
// layer with a Rec.709-encoded one, and that composite is not in any space.
//
// Applied in the post-composite stage, so it REQUIRES <working-space-composite>. A display
// transform consumes working-space pixels; without it the composite is already
// display-encoded by the time this stage runs, and applying a display transform to it would
// encode twice. Refused rather than rendered.
//
// Quote both arguments. Every display and view name in the bundled ACES config contains
// spaces, and AMCP tokenizes on whitespace.
std::future<std::wstring> ocio_display_command(command_context& ctx)
{
    auto image_mixer = ctx.channel.raw_channel->mixer().get_image_mixer();

    if (ctx.parameters.empty() || boost::iequals(ctx.parameters.at(0), L"INFO")) {
        auto st = image_mixer->get_ocio_display();
        std::wstringstream result;
        result << L"201 OCIO_DISPLAY OK\r\n";
        if (st.enabled)
            result << u16(st.display) << L" / " << u16(st.view) << L"\r\n";
        else
            result << L"NONE\r\n";
        return make_ready_future<std::wstring>(result.str());
    }

    if (boost::iequals(ctx.parameters.at(0), L"NONE") || boost::iequals(ctx.parameters.at(0), L"OFF")) {
        image_mixer->set_ocio_display("", "");
        return make_ready_future<std::wstring>(L"202 OCIO_DISPLAY OK\r\n");
    }

    if (!accelerator::ocio::available()) {
        CASPAR_LOG(warning) << L"[ocio] OCIO_DISPLAY refused: this server was built without OCIO support";
        return make_ready_future<std::wstring>(L"501 OCIO_DISPLAY FAILED\r\n");
    }

    if (ctx.parameters.size() < 2) {
        CASPAR_LOG(warning) << L"[ocio] OCIO_DISPLAY needs a display AND a view, both quoted. "
                               L"Use INFO OCIO DISPLAYS to list them.";
        return make_ready_future<std::wstring>(L"400 OCIO_DISPLAY FAILED\r\n");
    }

    const auto display = u8(ctx.parameters.at(0));
    const auto view    = u8(ctx.parameters.at(1));

    if (!accelerator::ocio::has_display_view(display, view)) {
        CASPAR_LOG(warning) << L"[ocio] OCIO_DISPLAY refused: '" << ctx.parameters.at(0) << L"' / '"
                            << ctx.parameters.at(1) << L"' is not a display/view pair in this config. "
                            << L"Use INFO OCIO DISPLAYS to list them.";
        return make_ready_future<std::wstring>(L"404 OCIO_DISPLAY ERROR\r\n");
    }

    // The precondition, checked before the build so the operator gets the real reason rather
    // than a shader that renders a double-encoded picture.
    if (!image_mixer->composites_in_working_space()) {
        CASPAR_LOG(warning) << L"[ocio] OCIO_DISPLAY refused: this channel does not composite in the "
                               L"working space. A display transform consumes working-space pixels; add "
                               L"<working-space-composite>true</working-space-composite> (and the fp16 "
                               L"render format it requires) to the channel.";
        return make_ready_future<std::wstring>(L"403 OCIO_DISPLAY ERROR\r\n");
    }

    // Build it now rather than only checking that the names exist. A pair can be present in
    // the config and still fail to produce a processor -- a missing LUT file behind a
    // FileTransform, an unresolvable role -- and by the time the mixer needs it there is no
    // way to report that except by rendering something wrong.
    //
    // It is also the pre-warm: OCIO's own guidance is that building a processor is expensive
    // and thread-blocking and should happen "as infrequently as is sensible". Doing it here
    // puts the cost on the command rather than on the first composited frame, and the
    // mixer's own build is then a cache hit inside OCIO.
    accelerator::ocio::gpu_shader probe;
    if (!accelerator::ocio::build_display_transform(display, view, probe,
                                                    accelerator::ocio::gpu_target::opengl)) {
        CASPAR_LOG(warning) << L"[ocio] OCIO_DISPLAY refused: '" << ctx.parameters.at(0) << L"' / '"
                            << ctx.parameters.at(1)
                            << L"' exists but no GPU transform could be built from it";
        return make_ready_future<std::wstring>(L"404 OCIO_DISPLAY ERROR\r\n");
    }

    image_mixer->set_ocio_display(display, view);

    // Build the GPU program now, off the frame path. The processor above is already a cache
    // hit inside OCIO; what this adds is the LUT upload and the GLSL compile, which are what
    // actually cost a frame. Measured 2026-08-13: a capture 1.6 s after this command
    // returned NO FRAME AT ALL, because the ACES 2.0 program is ~15 KB of GLSL.
    // With the channel look included: the look is composed INTO this processor, so
    // warming without it would compile a program the draw never asks for.
    image_mixer->prewarm_ocio("", display, view, image_mixer->get_ocio_look());

    return make_ready_future<std::wstring>(L"202 OCIO_DISPLAY OK\r\n");
}

// OCIO_LOOK <channel> "<look>"     apply an LMT in the working space, before the view
// OCIO_LOOK <channel> NONE          clear it
// OCIO_LOOK <channel> [INFO]        query
//
// A look is the show LUT of an ACES pipeline: a creative or technical transform applied to
// the scene-referred image BEFORE the display rendering. Channel-level, for the same reason
// the display transform is -- every layer of one composite belongs to one show -- and it
// applies to the primary AND to every consumer view, because a consumer asking for a
// different view still wants the show's look.
//
// It is COMPOSED INTO the display processor rather than spliced separately. That keeps the
// shader-variant cache key the (input, output) pair it already is, and lets OCIO optimise
// the look and the view together. The consequence is the refusal below: without a display
// transform there is nothing for the look to ride on.
std::future<std::wstring> ocio_look_command(command_context& ctx)
{
    auto image_mixer = ctx.channel.raw_channel->mixer().get_image_mixer();

    // Query: OCIO_LOOK <ch>  or  OCIO_LOOK <ch> INFO
    if (ctx.parameters.empty() || boost::iequals(ctx.parameters.at(0), L"INFO")) {
        const auto look = image_mixer->get_ocio_look();
        std::wstringstream result;
        result << L"201 OCIO_LOOK OK\r\n";
        result << (look.empty() ? std::wstring(L"NONE") : u16(look)) << L"\r\n";
        return make_ready_future<std::wstring>(result.str());
    }

    if (boost::iequals(ctx.parameters.at(0), L"NONE") || boost::iequals(ctx.parameters.at(0), L"OFF")) {
        image_mixer->set_ocio_look("");
        const auto cleared = image_mixer->get_ocio_display();
        if (cleared.enabled)
            image_mixer->prewarm_ocio("", cleared.display, cleared.view, "");
        return make_ready_future<std::wstring>(L"202 OCIO_LOOK OK\r\n");
    }

    if (!accelerator::ocio::available()) {
        CASPAR_LOG(warning) << L"[ocio] OCIO_LOOK refused: this server was built without OCIO support";
        return make_ready_future<std::wstring>(L"501 OCIO_LOOK FAILED\r\n");
    }

    const auto look = u8(ctx.parameters.at(0));

    // A single name is validated here; a look EXPRESSION (`-name` inverts, commas chain
    // several) is left to the build below, which is the only thing that can judge it.
    // Checking the simple case means the common typo gets the useful error.
    if (look.find(',') == std::string::npos && look.front() != '-' && !accelerator::ocio::has_look(look)) {
        CASPAR_LOG(warning) << L"[ocio] OCIO_LOOK refused: '" << ctx.parameters.at(0) << L"' is not a look in "
                            << u16(accelerator::ocio::config_uri()) << L". Use INFO OCIO LOOKS to list them.";
        return make_ready_future<std::wstring>(L"404 OCIO_LOOK ERROR\r\n");
    }

    if (!image_mixer->composites_in_working_space()) {
        CASPAR_LOG(warning) << L"[ocio] OCIO_LOOK refused: this channel does not composite in the "
                               L"working space. A look acts on scene-referred pixels; add "
                               L"<working-space-composite>true</working-space-composite> (and the fp16 "
                               L"render format it requires) to the channel.";
        return make_ready_future<std::wstring>(L"403 OCIO_LOOK ERROR\r\n");
    }

    // The look rides on the display processor, so there has to be one. Refused rather than
    // stored for later: a command that returns 202 and changes nothing is the exact failure
    // this tree has hit repeatedly.
    const auto disp = image_mixer->get_ocio_display();
    if (!disp.enabled) {
        CASPAR_LOG(warning) << L"[ocio] OCIO_LOOK refused: no display transform is set on this channel. "
                               L"A look is composed into the display rendering, so set OCIO_DISPLAY first.";
        return make_ready_future<std::wstring>(L"403 OCIO_LOOK ERROR\r\n");
    }

    // Build it now: a look can exist and still fail to produce a processor -- a missing LUT
    // file behind a FileTransform, a process space that does not resolve -- and by the time
    // the mixer needs it there is no way to report that except by rendering something wrong.
    accelerator::ocio::gpu_shader probe;
    if (!accelerator::ocio::build_display_transform(disp.display, disp.view, probe,
                                                    accelerator::ocio::gpu_target::opengl, look)) {
        CASPAR_LOG(warning) << L"[ocio] OCIO_LOOK refused: '" << ctx.parameters.at(0)
                            << L"' exists but no GPU transform could be built with it";
        return make_ready_future<std::wstring>(L"404 OCIO_LOOK ERROR\r\n");
    }

    image_mixer->set_ocio_look(look);
    image_mixer->prewarm_ocio("", disp.display, disp.view, look);

    return make_ready_future<std::wstring>(L"202 OCIO_LOOK OK\r\n");
}

// AMF <channel>-<layer> "<file.amf>"
//
// Configure a layer and its channel from an ACES Metadata File: the document a show carries
// to say which input transform, look and output transform its pipeline uses.
//
// It applies exactly what the three existing commands apply, and nothing else --
// `MIXER <ch>-<layer> OCIO`, `OCIO_LOOK <ch>` and `OCIO_DISPLAY <ch>` -- so an AMF is a way
// of *addressing* those, not a fourth colour path. That is also how it is gated: applying an
// AMF must render byte-identically to issuing the three by hand.
//
// The mapping is mechanical rather than a table. OCIO configs carry the AMF transform ids
// under `interchange: amf_transform_ids`, so an id resolves through the loaded config and a
// config change moves the ids with the transforms they name. See docs/AMF_SUPPORT_STUDY.md.
//
// RESOLVE EVERYTHING, THEN APPLY. Three settings from one file must not leave a channel half
// configured because the third id was unknown -- the operator would be looking at a picture
// that is neither the old look nor the new one.
std::future<std::wstring> amf_command(command_context& ctx)
{
    if (!accelerator::ocio::available()) {
        CASPAR_LOG(warning) << L"[ocio] AMF refused: this server was built without OCIO support";
        return make_ready_future<std::wstring>(L"501 AMF FAILED\r\n");
    }
    if (ctx.parameters.empty()) {
        CASPAR_LOG(warning) << L"[ocio] AMF needs a file. AMF <ch>-<layer> \"<file.amf>\"";
        return make_ready_future<std::wstring>(L"400 AMF FAILED\r\n");
    }

    auto image_mixer = ctx.channel.raw_channel->mixer().get_image_mixer();

    // As given, then under <media-path> -- the same resolution CALIBRATION LUT and
    // MIXER CDL_FILE use.
    std::wstring path = ctx.parameters.at(0);
    if (!std::ifstream(path).is_open())
        path = caspar::env::media_folder() + L"/" + path;

    pt::wptree amf;
    try {
        boost::filesystem::wifstream file(path);
        pt::read_xml(file, amf, pt::xml_parser::trim_whitespace | pt::xml_parser::no_comments);
    } catch (...) {
        CASPAR_LOG(warning) << L"[ocio] AMF refused: could not read " << path << L" as XML";
        return make_ready_future<std::wstring>(L"404 AMF ERROR\r\n");
    }

    // The AMF namespace prefix is `aces:` by convention but is not guaranteed, and
    // property_tree does not resolve namespaces -- it keeps the prefix in the key. So walk
    // for a node whose LOCAL name matches rather than assuming `aces:`. A file using a
    // different prefix is still a valid AMF and would otherwise be rejected as malformed.
    std::function<std::wstring(const pt::wptree&, const std::wstring&)> find_id =
        [&](const pt::wptree& node, const std::wstring& local) -> std::wstring {
        for (const auto& child : node) {
            const auto  key   = child.first;
            const auto  colon = key.find_last_of(L':');
            const auto  name  = colon == std::wstring::npos ? key : key.substr(colon + 1);
            if (name == local) {
                // The id may be this node's own text or a child <transformId>.
                auto tid = child.second.get_optional<std::wstring>(L"aces:transformId");
                if (!tid) {
                    for (const auto& g : child.second) {
                        const auto k = g.first;
                        const auto c = k.find_last_of(L':');
                        if ((c == std::wstring::npos ? k : k.substr(c + 1)) == L"transformId")
                            tid = g.second.get_value<std::wstring>();
                    }
                }
                if (tid && !tid->empty())
                    return *tid;
            }
            auto deeper = find_id(child.second, local);
            if (!deeper.empty())
                return deeper;
        }
        return {};
    };

    const auto in_id   = find_id(amf, L"inputTransform");
    const auto look_id = find_id(amf, L"lookTransform");
    const auto out_id  = find_id(amf, L"outputTransform");

    if (in_id.empty() && look_id.empty() && out_id.empty()) {
        CASPAR_LOG(warning) << L"[ocio] AMF refused: " << path
                            << L" carries no inputTransform, lookTransform or outputTransform";
        return make_ready_future<std::wstring>(L"404 AMF ERROR\r\n");
    }

    // ---- resolve everything first -------------------------------------------------
    accelerator::ocio::amf_resolution in_res, look_res, out_res;
    auto resolve = [&](const std::wstring& id, accelerator::ocio::amf_resolution& out,
                       const wchar_t* what) -> bool {
        if (id.empty())
            return true;
        if (accelerator::ocio::resolve_amf_transform_id(u8(id), out))
            return true;
        CASPAR_LOG(warning) << L"[ocio] AMF refused: its " << what << L" id '" << id
                            << L"' does not resolve against " << u16(accelerator::ocio::config_uri());
        return false;
    };
    if (!resolve(in_id, in_res, L"inputTransform") || !resolve(look_id, look_res, L"lookTransform") ||
        !resolve(out_id, out_res, L"outputTransform"))
        return make_ready_future<std::wstring>(L"404 AMF ERROR\r\n");

    // An output transform must yield BOTH halves of the display/view pair. The same id sits
    // on a colour space (whose name is the display) and on a view transform (the view); a
    // file that resolves only one of them cannot drive OCIO_DISPLAY and is refused rather
    // than half-applied.
    if (!out_id.empty() && (out_res.colorspace.empty() || out_res.view_transform.empty())) {
        CASPAR_LOG(warning) << L"[ocio] AMF refused: its outputTransform resolved to display='"
                            << u16(out_res.colorspace) << L"' view='" << u16(out_res.view_transform)
                            << L"' -- both halves are needed to set a display transform";
        return make_ready_future<std::wstring>(L"404 AMF ERROR\r\n");
    }
    if (!out_res.colorspace.empty() &&
        !accelerator::ocio::has_display_view(out_res.colorspace, out_res.view_transform)) {
        CASPAR_LOG(warning) << L"[ocio] AMF refused: '" << u16(out_res.colorspace) << L"' / '"
                            << u16(out_res.view_transform) << L"' is not a display/view pair "
                            << L"in this config";
        return make_ready_future<std::wstring>(L"404 AMF ERROR\r\n");
    }
    if (!look_id.empty() && look_res.look.empty()) {
        CASPAR_LOG(warning) << L"[ocio] AMF refused: its lookTransform id resolved to no look";
        return make_ready_future<std::wstring>(L"404 AMF ERROR\r\n");
    }
    if (!in_id.empty() && in_res.colorspace.empty()) {
        CASPAR_LOG(warning) << L"[ocio] AMF refused: its inputTransform id resolved to no colour space";
        return make_ready_future<std::wstring>(L"404 AMF ERROR\r\n");
    }

    // ---- then apply, display before look ------------------------------------------
    // Ordered, not arbitrary: OCIO_LOOK is composed into the display processor and refuses
    // when no display transform is set, so a look applied first would be rejected by the
    // very state this command is about to establish.
    if (!out_res.colorspace.empty()) {
        image_mixer->set_ocio_display(out_res.colorspace, out_res.view_transform);
        image_mixer->prewarm_ocio("", out_res.colorspace, out_res.view_transform, look_res.look);
    }
    if (!look_id.empty())
        image_mixer->set_ocio_look(look_res.look);

    if (!in_res.colorspace.empty()) {
        const auto space = in_res.colorspace;
        transforms_applier transforms(ctx);
        transforms.add(stage::transform_tuple_t(
            ctx.layer_index(),
            [space](frame_transform t) -> frame_transform {
                t.image_transform.ocio.enable       = true;
                t.image_transform.ocio.source_space = space;
                return t;
            },
            0,
            L"linear"));
        transforms.apply();
        // The input transform's own pre-warm, as MIXER OCIO does: the display half above is
        // warmed separately, and this one is a different program.
        image_mixer->prewarm_ocio(space, "", "");
    }

    CASPAR_LOG(info) << L"[ocio] AMF " << path << L" -> input '" << u16(in_res.colorspace)
                     << L"' look '" << u16(look_res.look) << L"' display '"
                     << u16(out_res.colorspace) << L"' / '" << u16(out_res.view_transform) << L"'";

    std::wstringstream result;
    result << L"202 AMF OK\r\n";
    return make_ready_future<std::wstring>(result.str());
}

// CALIBRATION <channel> LUT <file.cube> [strength]   load a channel-master LED calibration LUT
// CALIBRATION <channel> CLEAR                         remove the calibration LUT
// CALIBRATION <channel> BYPASS <0|1>                  temporarily bypass without unloading
// CALIBRATION <channel> [INFO]                        query current state
//
// Unlike MIXER LUT3D (which is per-layer), CALIBRATION applies a single
// display-to-display 3D LUT to the final composited channel output, so every
// consumer (SDI/NDI/screen/file) receives the corrected pixels. It is intended
// for whole LED-wall colour calibration (e.g. LUTs solved by OpenVPCal).
std::future<std::wstring> calibration_command(command_context& ctx)
{
    auto image_mixer = ctx.channel.raw_channel->mixer().get_image_mixer();

    // Query: CALIBRATION <ch>  or  CALIBRATION <ch> INFO
    if (ctx.parameters.empty() || boost::iequals(ctx.parameters.at(0), L"INFO")) {
        auto              st = image_mixer->get_calibration_state();
        std::wstringstream result;
        result << L"201 CALIBRATION OK\r\n";
        result << (st.enabled ? L"ENABLED" : L"NONE");
        if (st.enabled) {
            result << L" SIZE " << st.size << L" STRENGTH " << st.strength << L" BYPASS "
                   << (st.bypass ? 1 : 0);
            if (!st.path.empty())
                result << L" PATH " << st.path;
        }
        result << L"\r\n";
        return make_ready_future<std::wstring>(result.str());
    }

    if (boost::iequals(ctx.parameters.at(0), L"CLEAR")) {
        image_mixer->set_calibration_lut(nullptr, 1.0f, L"");
        return make_ready_future<std::wstring>(L"202 CALIBRATION OK\r\n");
    }

    if (boost::iequals(ctx.parameters.at(0), L"BYPASS")) {
        bool bypass = ctx.parameters.size() > 1 &&
                      (ctx.parameters.at(1) == L"1" || boost::iequals(ctx.parameters.at(1), L"TRUE"));
        image_mixer->set_calibration_bypass(bypass);
        return make_ready_future<std::wstring>(L"202 CALIBRATION OK\r\n");
    }

    if (boost::iequals(ctx.parameters.at(0), L"LUT")) {
        if (ctx.parameters.size() < 2)
            return make_ready_future<std::wstring>(L"403 CALIBRATION ERROR\r\n");

        // Resolve path: try as-is first, then relative to media folder
        std::wstring path = ctx.parameters.at(1);
        if (!std::ifstream(path).is_open()) {
            auto media = caspar::env::media_folder();
            path       = media + L"/" + path;
        }

        auto lut = parse_cube_file(path);
        if (!lut)
            return make_ready_future<std::wstring>(L"404 CALIBRATION LOAD FAILED\r\n");

        float strength = 1.0f;
        if (ctx.parameters.size() > 2 && !parse_lut_strength(ctx.parameters[2], strength))
            return make_ready_future<std::wstring>(L"403 CALIBRATION ERROR\r\n");

        image_mixer->set_calibration_lut(lut, strength, path);
        return make_ready_future<std::wstring>(L"202 CALIBRATION OK\r\n");
    }

    return make_ready_future<std::wstring>(L"403 CALIBRATION ERROR\r\n");
}

// MIXER HUECURVE <HUE_HUE|HUE_SAT|HUE_LUM|SAT_SAT> <h1> <offset1> <h2> <offset2> ...
// MIXER HUECURVE RESET
// MIXER HUECURVE — query
static std::shared_ptr<core::hue_curve_data> build_hue_curve_lut(
    const std::vector<std::pair<float, float>>& points, int channel)
{
    // Build 256-entry LUT from control points using linear interpolation
    // Channel: 0=HvH, 1=HvS, 2=HvL, 3=SvS
    auto data = std::make_shared<core::hue_curve_data>();
    data->data.resize(256 * 4, 0.0f);

    // Set defaults: HvH=0 (no offset), HvS=1 (no change), HvL=0, SvS=1
    for (int i = 0; i < 256; ++i) {
        data->data[i * 4 + 0] = 0.0f;  // HvH offset
        data->data[i * 4 + 1] = 1.0f;  // HvS multiplier
        data->data[i * 4 + 2] = 0.0f;  // HvL offset
        data->data[i * 4 + 3] = 1.0f;  // SvS multiplier
    }

    if (points.size() < 2)
        return data;

    // Sort points by x (hue position 0..1)
    auto sorted = points;
    std::sort(sorted.begin(), sorted.end());

    // Linear interpolation between control points, wrapping at edges
    for (int i = 0; i < 256; ++i) {
        float x = static_cast<float>(i) / 255.0f;
        float val = 0.0f;

        // Find surrounding control points
        if (x <= sorted.front().first) {
            val = sorted.front().second;
        } else if (x >= sorted.back().first) {
            val = sorted.back().second;
        } else {
            for (size_t j = 0; j + 1 < sorted.size(); ++j) {
                if (x >= sorted[j].first && x <= sorted[j + 1].first) {
                    float t = (x - sorted[j].first) / (sorted[j + 1].first - sorted[j].first);
                    val = sorted[j].second + t * (sorted[j + 1].second - sorted[j].second);
                    break;
                }
            }
        }
        data->data[i * 4 + channel] = val;
    }
    return data;
}

std::future<std::wstring> mixer_huecurve_command(command_context& ctx)
{
    if (ctx.parameters.empty()) {
        auto transform2 = get_current_transform(ctx).share();
        return std::async(std::launch::deferred, [transform2]() -> std::wstring {
            auto t = transform2.get().image_transform;
            if (!t.hue_curves)
                return L"201 MIXER OK\r\nDISABLED\r\n";
            return L"201 MIXER OK\r\nACTIVE\r\n";
        });
    }

    if (boost::iequals(ctx.parameters.at(0), L"RESET")) {
        transforms_applier transforms(ctx);
        transforms.add(stage::transform_tuple_t(
            ctx.layer_index(),
            [](frame_transform t) { t.image_transform.hue_curves = nullptr; return t; },
            0, L"linear"));
        transforms.apply();
        return make_ready_future<std::wstring>(L"202 MIXER OK\r\n");
    }

    // Determine channel
    int channel = -1;
    if      (boost::iequals(ctx.parameters.at(0), L"HUE_HUE")) channel = 0;
    else if (boost::iequals(ctx.parameters.at(0), L"HUE_SAT")) channel = 1;
    else if (boost::iequals(ctx.parameters.at(0), L"HUE_LUM")) channel = 2;
    else if (boost::iequals(ctx.parameters.at(0), L"SAT_SAT")) channel = 3;
    if (channel < 0)
        return make_ready_future<std::wstring>(L"400 ERROR\r\n");

    int n_params = static_cast<int>(ctx.parameters.size()) - 1;
    if (n_params < 4 || n_params % 2 != 0)
        return make_ready_future<std::wstring>(L"400 ERROR\r\n");

    // TWO ranges, chosen by curve type, and this is not a detail. HUE_HUE and HUE_LUM
    // carry signed OFFSETS (-1..1); HUE_SAT and SAT_SAT carry MULTIPLIERS. Validating all
    // four against the offset range refuses ordinary commands -- a saturation boost of
    // 1.45 is entirely normal and sits outside -1..1.
    const auto value_range = (channel == 1 || channel == 3) ? core::grade_limits::hue_curve_scale
                                                            : core::grade_limits::hue_curve_offset;
    std::vector<std::pair<float, float>> points;
    for (int i = 0; i < n_params / 2; ++i) {
        float h = static_cast<float>(grade_param(ctx.parameters.at(1 + i * 2),
                                                 core::grade_limits::curve_coord, L"curve position"));
        float v = static_cast<float>(grade_param(ctx.parameters.at(2 + i * 2),
                                                 value_range, L"curve value"));
        points.emplace_back(h, v);
    }

    auto lut = build_hue_curve_lut(points, channel);

    transforms_applier transforms(ctx);
    transforms.add(stage::transform_tuple_t(
        ctx.layer_index(),
        [=](frame_transform transform) -> frame_transform {
            // Merge with existing hue curves if present
            if (transform.image_transform.hue_curves) {
                auto merged = std::make_shared<core::hue_curve_data>(*transform.image_transform.hue_curves);
                for (int i = 0; i < 256; ++i) {
                    merged->data[i * 4 + channel] = lut->data[i * 4 + channel];
                }
                transform.image_transform.hue_curves = merged;
            } else {
                transform.image_transform.hue_curves = lut;
            }
            return transform;
        },
        0, L"linear"));
    transforms.apply();
    return make_ready_future<std::wstring>(L"202 MIXER OK\r\n");
}

// MIXER TONEBALANCE shadows highlights [duration tween]  -- shadow/highlight tonal separation
std::future<std::wstring> mixer_tonebalance_command(command_context& ctx)
{
    if (ctx.parameters.empty()) {
        auto transform2 = get_current_transform(ctx).share();
        return std::async(std::launch::deferred, [transform2]() -> std::wstring {
            auto t = transform2.get().image_transform;
            return L"201 MIXER OK\r\n" + std::to_wstring(t.shadows) + L" " +
                   std::to_wstring(t.highlights) + L"\r\n";
        });
    }

    transforms_applier transforms(ctx);
    grade_require(ctx, 2, L"MIXER TONEBALANCE shadows highlights [duration] [tween]");
    double shadows    = grade_param(ctx.parameters.at(0), core::grade_limits::tone, L"shadows");
    double highlights = grade_param(ctx.parameters.at(1), core::grade_limits::tone, L"highlights");
    int          duration   = ctx.parameters.size() > 2 ? std::stoi(ctx.parameters[2]) : 0;
    std::wstring tween      = ctx.parameters.size() > 3 ? ctx.parameters[3] : L"linear";

    transforms.add(stage::transform_tuple_t(
        ctx.layer_index(),
        [=](frame_transform transform) -> frame_transform {
            transform.image_transform.shadows    = shadows;
            transform.image_transform.highlights = highlights;
            return transform;
        },
        duration,
        tween));
    transforms.apply();

    return make_ready_future<std::wstring>(L"202 MIXER OK\r\n");
}

// MIXER SHARPEN amount [radius] [duration tween]
std::future<std::wstring> mixer_sharpen_command(command_context& ctx)
{
    if (ctx.parameters.empty()) {
        auto transform2 = get_current_transform(ctx).share();
        return std::async(std::launch::deferred, [transform2]() -> std::wstring {
            auto t = transform2.get().image_transform;
            return L"201 MIXER OK\r\n" + std::to_wstring(t.sharpen_amount) + L" " +
                   std::to_wstring(t.sharpen_radius) + L"\r\n";
        });
    }

    transforms_applier transforms(ctx);
    grade_require(ctx, 1, L"MIXER SHARPEN amount [radius] [duration] [tween]");
    double amount = grade_param(ctx.parameters.at(0), core::grade_limits::sharpen_amount, L"amount");
    double radius = ctx.parameters.size() > 1
                        ? grade_param(ctx.parameters[1], core::grade_limits::sharpen_radius, L"radius")
                        : 1.0;
    int          duration = ctx.parameters.size() > 2 ? std::stoi(ctx.parameters[2]) : 0;
    std::wstring tween    = ctx.parameters.size() > 3 ? ctx.parameters[3] : L"linear";

    transforms.add(stage::transform_tuple_t(
        ctx.layer_index(),
        [=](frame_transform transform) -> frame_transform {
            transform.image_transform.sharpen_amount = amount;
            transform.image_transform.sharpen_radius = radius;
            return transform;
        },
        duration,
        tween));
    transforms.apply();
    return make_ready_future<std::wstring>(L"202 MIXER OK\r\n");
}

// MIXER GRAIN intensity [size] [duration tween]
std::future<std::wstring> mixer_grain_command(command_context& ctx)
{
    if (ctx.parameters.empty()) {
        auto transform2 = get_current_transform(ctx).share();
        return std::async(std::launch::deferred, [transform2]() -> std::wstring {
            auto t = transform2.get().image_transform;
            return L"201 MIXER OK\r\n" + std::to_wstring(t.grain_intensity) + L" " +
                   std::to_wstring(t.grain_size) + L"\r\n";
        });
    }

    transforms_applier transforms(ctx);
    grade_require(ctx, 1, L"MIXER GRAIN intensity [size] [duration] [tween]");
    double intensity = grade_param(ctx.parameters.at(0), core::grade_limits::grain_intensity, L"intensity");
    double size      = ctx.parameters.size() > 1
                           ? grade_param(ctx.parameters[1], core::grade_limits::grain_size, L"size")
                           : 1.0;
    int          duration  = ctx.parameters.size() > 2 ? std::stoi(ctx.parameters[2]) : 0;
    std::wstring tween     = ctx.parameters.size() > 3 ? ctx.parameters[3] : L"linear";

    transforms.add(stage::transform_tuple_t(
        ctx.layer_index(),
        [=](frame_transform transform) -> frame_transform {
            transform.image_transform.grain_intensity = intensity;
            transform.image_transform.grain_size      = size;
            return transform;
        },
        duration,
        tween));
    transforms.apply();
    return make_ready_future<std::wstring>(L"202 MIXER OK\r\n");
}

// MIXER QUALIFIER <target_hue> <hue_width> <min_sat> <max_sat> <min_lum> <max_lum>
//                 <softness> <exp_offset> <sat_offset> <hue_offset> [duration tween]
// MIXER QUALIFIER 0 — disable
// MIXER QUALIFIER — query
std::future<std::wstring> mixer_qualifier_command(command_context& ctx)
{
    if (ctx.parameters.empty()) {
        auto transform2 = get_current_transform(ctx).share();
        return std::async(std::launch::deferred, [transform2]() -> std::wstring {
            auto t = transform2.get().image_transform;
            if (!t.qualifier_enable)
                return L"201 MIXER OK\r\nDISABLED\r\n";
            auto f = [](double v) { return std::to_wstring(v); };
            return L"201 MIXER OK\r\n" +
                   f(t.qual_target_hue)  + L" " + f(t.qual_hue_width)  + L" " +
                   f(t.qual_min_sat)     + L" " + f(t.qual_max_sat)    + L" " +
                   f(t.qual_min_lum)     + L" " + f(t.qual_max_lum)    + L" " +
                   f(t.qual_softness)    + L" " + f(t.qual_exposure)   + L" " +
                   f(t.qual_sat_offset)  + L" " + f(t.qual_hue_offset) + L"\r\n";
        });
    }

    // Single param "0" = disable
    if (ctx.parameters.size() == 1 && ctx.parameters.at(0) == L"0") {
        transforms_applier transforms(ctx);
        transforms.add(stage::transform_tuple_t(
            ctx.layer_index(),
            [](frame_transform t) { t.image_transform.qualifier_enable = false; return t; },
            0, L"linear"));
        transforms.apply();
        return make_ready_future<std::wstring>(L"202 MIXER OK\r\n");
    }

    transforms_applier transforms(ctx);
    grade_require(ctx, 10, L"MIXER QUALIFIER hue width minSat maxSat minLum maxLum softness exposure satOffset hueOffset [duration] [tween]");
    double tgt_hue   = grade_param(ctx.parameters.at(0), core::grade_limits::hue_degrees, L"target hue");
    double hue_w     = grade_param(ctx.parameters.at(1), core::grade_limits::hue_width, L"hue width");
    double min_sat   = grade_param(ctx.parameters.at(2), core::grade_limits::unit, L"min saturation");
    double max_sat   = grade_param(ctx.parameters.at(3), core::grade_limits::unit, L"max saturation");
    double min_lum   = grade_param(ctx.parameters.at(4), core::grade_limits::unit, L"min luminance");
    double max_lum   = grade_param(ctx.parameters.at(5), core::grade_limits::unit, L"max luminance");
    double softness  = grade_param(ctx.parameters.at(6), core::grade_limits::unit, L"softness");
    double exp_off   = grade_param(ctx.parameters.at(7), core::grade_limits::offset, L"exposure offset");
    double sat_off   = grade_param(ctx.parameters.at(8), core::grade_limits::offset, L"saturation offset");
    double hue_off   = grade_param(ctx.parameters.at(9), core::grade_limits::hue_shift, L"hue offset");
    int    duration  = ctx.parameters.size() > 10 ? std::stoi(ctx.parameters[10]) : 0;
    std::wstring tw  = ctx.parameters.size() > 11 ? ctx.parameters[11] : L"linear";

    transforms.add(stage::transform_tuple_t(
        ctx.layer_index(),
        [=](frame_transform transform) -> frame_transform {
            transform.image_transform.qualifier_enable = true;
            transform.image_transform.qual_target_hue  = tgt_hue;
            transform.image_transform.qual_hue_width   = hue_w;
            transform.image_transform.qual_min_sat     = min_sat;
            transform.image_transform.qual_max_sat     = max_sat;
            transform.image_transform.qual_min_lum     = min_lum;
            transform.image_transform.qual_max_lum     = max_lum;
            transform.image_transform.qual_softness    = softness;
            transform.image_transform.qual_exposure    = exp_off;
            transform.image_transform.qual_sat_offset  = sat_off;
            transform.image_transform.qual_hue_offset  = hue_off;
            return transform;
        },
        duration,
        tw));
    transforms.apply();
    return make_ready_future<std::wstring>(L"202 MIXER OK\r\n");
}

// MIXER GRADE — windowed grading node chain (PROTOTYPE)
//
// One window shape (soft-edged ellipse in FRAME space) and one operation (exposure).
// The narrowest surface that exercises the node PASS and the variable-length data model
// end to end; design study in docs/GRADING_NODE_GRAPH_STUDY.md.
//
//   MIXER <ch>-<layer> GRADE NODE <n> <cx> <cy> <rx> <ry> <feather> <exposure> [<invert>]
//   MIXER <ch>-<layer> GRADE CLEAR    — drop the whole chain
//   MIXER <ch>-<layer> GRADE          — query
//
// No DURATION/TWEEN. Tweening would have to address node[n].window.field, which the
// tween system cannot express -- named as an open question in the study rather than
// quietly half-built here.
std::future<std::wstring> mixer_grade_command(command_context& ctx)
{
    if (ctx.parameters.empty()) {
        auto transform2 = get_current_transform(ctx).share();
        return std::async(std::launch::deferred, [transform2]() -> std::wstring {
            auto graph = transform2.get().image_transform.grade_nodes;
            if (!graph || graph->nodes.empty())
                return L"201 MIXER OK\r\nDISABLED\r\n";
            auto         f = [](double v) { return std::to_wstring(v); };
            std::wstring out = L"201 MIXER OK\r\n";
            for (size_t i = 0; i < graph->nodes.size(); ++i) {
                const auto& n = graph->nodes[i];
                out += std::to_wstring(i) + L" " + (n.enable ? L"1" : L"0") + L" " +
                       f(n.window.center[0]) + L" " + f(n.window.center[1]) + L" " +
                       f(n.window.radius[0]) + L" " + f(n.window.radius[1]) + L" " +
                       f(n.window.feather) + L" " + f(n.exposure) + L" " +
                       (n.window.invert ? L"1" : L"0") + L"\r\n";
            }
            return out;
        });
    }

    if (boost::iequals(ctx.parameters.at(0), L"CLEAR")) {
        transforms_applier transforms(ctx);
        transforms.add(stage::transform_tuple_t(
            ctx.layer_index(),
            [](frame_transform t) {
                t.image_transform.grade_nodes = nullptr;
                return t;
            },
            0,
            L"linear"));
        transforms.apply();
        return make_ready_future<std::wstring>(L"202 MIXER OK\r\n");
    }

    if (!boost::iequals(ctx.parameters.at(0), L"NODE"))
        CASPAR_THROW_EXCEPTION(user_error() << msg_info(
                                   "MIXER GRADE NODE <n> cx cy rx ry feather exposure [invert] | CLEAR"));

    grade_require(ctx, 8, L"MIXER GRADE NODE <n> cx cy rx ry feather exposure [invert]");
    const int index = std::stoi(ctx.parameters.at(1));
    if (index < 0 || index > 15)
        CASPAR_THROW_EXCEPTION(user_error() << msg_info("MIXER GRADE node index must be 0-15"));

    const double cx      = grade_param(ctx.parameters.at(2), core::grade_limits::unit, L"centre x");
    const double cy      = grade_param(ctx.parameters.at(3), core::grade_limits::unit, L"centre y");
    const double rx      = grade_param(ctx.parameters.at(4), core::grade_limits::unit, L"radius x");
    const double ry      = grade_param(ctx.parameters.at(5), core::grade_limits::unit, L"radius y");
    const double feather = grade_param(ctx.parameters.at(6), core::grade_limits::unit, L"feather");
    const double expos   = grade_param(ctx.parameters.at(7), core::grade_limits::exposure, L"exposure");
    const bool   invert  = ctx.parameters.size() > 8 && ctx.parameters.at(8) != L"0";

    transforms_applier transforms(ctx);
    transforms.add(stage::transform_tuple_t(
        ctx.layer_index(),
        [=](frame_transform transform) -> frame_transform {
            // Copy-on-write, and it is load-bearing rather than tidy: composition
            // (apply_transform_colour_values) and equality (image_transform::operator==,
            // which is what the still-frame cache compares) both use POINTER identity. A
            // graph mutated in place would compare equal to itself and the cache would
            // replay the previous frame -- the exact defect the fingerprint exists to
            // prevent. Every mutation therefore allocates.
            auto next = std::make_shared<core::grade_graph>();
            if (transform.image_transform.grade_nodes)
                next->nodes = transform.image_transform.grade_nodes->nodes;
            if (next->nodes.size() <= static_cast<size_t>(index))
                next->nodes.resize(static_cast<size_t>(index) + 1);

            auto& n            = next->nodes[static_cast<size_t>(index)];
            n.enable           = true;
            n.window.center    = {cx, cy};
            n.window.radius    = {rx, ry};
            n.window.feather   = feather;
            n.window.invert    = invert;
            n.exposure         = expos;

            transform.image_transform.grade_nodes = next;
            return transform;
        },
        0,
        L"linear"));
    transforms.apply();
    return make_ready_future<std::wstring>(L"202 MIXER OK\r\n");
}

// MIXER RGBLEVELS — per-channel independent levels
// Query:  MIXER 1-1 RGBLEVELS
// Reset:  MIXER 1-1 RGBLEVELS RESET
// Set:    MIXER 1-1 RGBLEVELS r_min_in r_max_in r_gamma r_min_out r_max_out
//                             g_min_in g_max_in g_gamma g_min_out g_max_out
//                             b_min_in b_max_in b_gamma b_min_out b_max_out  [dur tween]
std::future<std::wstring> mixer_rgblevels_command(command_context& ctx)
{
    if (ctx.parameters.empty()) {
        auto t2 = get_current_transform(ctx).share();
        return std::async(std::launch::deferred, [t2]() -> std::wstring {
            const auto& rl = t2.get().image_transform.per_channel_levels;
            if (!rl.enable)
                return L"201 MIXER OK\r\nDISABLED\r\n";
            auto f   = [](double v) { return std::to_wstring(v); };
            auto row = [&](const core::rgb_levels_channel& c) {
                return f(c.min_input) + L" " + f(c.max_input) + L" " +
                       f(c.gamma)     + L" " + f(c.min_output) + L" " + f(c.max_output);
            };
            return L"201 MIXER OK\r\n" + row(rl.r) + L" " + row(rl.g) + L" " + row(rl.b) + L"\r\n";
        });
    }

    if (boost::iequals(ctx.parameters.at(0), L"RESET")) {
        transforms_applier transforms(ctx);
        transforms.add(stage::transform_tuple_t(
            ctx.layer_index(),
            [](frame_transform t) { t.image_transform.per_channel_levels = core::rgb_levels{}; return t; },
            0, L"linear"));
        transforms.apply();
        return make_ready_future<std::wstring>(L"202 MIXER OK\r\n");
    }

    transforms_applier transforms(ctx);
    core::rgb_levels rl;
    rl.enable       = true;
    grade_require(ctx, 15, L"MIXER RGBLEVELS <min_in max_in gamma min_out max_out> x3 [duration] [tween]");
    rl.r.min_input  = grade_param(ctx.parameters.at(0), core::grade_limits::level, L"R min input");
    rl.r.max_input  = grade_param(ctx.parameters.at(1), core::grade_limits::level, L"R max input");
    rl.r.gamma      = grade_param(ctx.parameters.at(2), core::grade_limits::level_gamma, L"R gamma");
    rl.r.min_output = grade_param(ctx.parameters.at(3), core::grade_limits::level, L"R min output");
    rl.r.max_output = grade_param(ctx.parameters.at(4), core::grade_limits::level, L"R max output");
    rl.g.min_input  = grade_param(ctx.parameters.at(5), core::grade_limits::level, L"G min input");
    rl.g.max_input  = grade_param(ctx.parameters.at(6), core::grade_limits::level, L"G max input");
    rl.g.gamma      = grade_param(ctx.parameters.at(7), core::grade_limits::level_gamma, L"G gamma");
    rl.g.min_output = grade_param(ctx.parameters.at(8), core::grade_limits::level, L"G min output");
    rl.g.max_output = grade_param(ctx.parameters.at(9), core::grade_limits::level, L"G max output");
    rl.b.min_input  = grade_param(ctx.parameters.at(10), core::grade_limits::level, L"B min input");
    rl.b.max_input  = grade_param(ctx.parameters.at(11), core::grade_limits::level, L"B max input");
    rl.b.gamma      = grade_param(ctx.parameters.at(12), core::grade_limits::level_gamma, L"B gamma");
    rl.b.min_output = grade_param(ctx.parameters.at(13), core::grade_limits::level, L"B min output");
    rl.b.max_output = grade_param(ctx.parameters.at(14), core::grade_limits::level, L"B max output");
    int          duration = ctx.parameters.size() > 15 ? std::stoi(ctx.parameters[15]) : 0;
    std::wstring tween    = ctx.parameters.size() > 16 ? ctx.parameters[16]            : L"linear";

    transforms.add(stage::transform_tuple_t(
        ctx.layer_index(),
        [=](frame_transform transform) -> frame_transform {
            transform.image_transform.per_channel_levels = rl;
            return transform;
        },
        duration,
        tween));
    transforms.apply();
    return make_ready_future<std::wstring>(L"202 MIXER OK\r\n");
}

// MIXER CURVES — per-channel + master tone curves via Catmull-Rom spline control points
// Query all:      MIXER 1-1 CURVES
// Query channel:  MIXER 1-1 CURVES R|G|B|MASTER
// Reset all:      MIXER 1-1 CURVES RESET
// Set channel:    MIXER 1-1 CURVES R|G|B|MASTER x1 y1 x2 y2 [...up to 16 pairs]
//                 (min 2 pairs; x values must be in 0..1 ascending order)
std::future<std::wstring> mixer_curves_command(command_context& ctx)
{
    if (ctx.parameters.empty()) {
        auto t2 = get_current_transform(ctx).share();
        return std::async(std::launch::deferred, [t2]() -> std::wstring {
            const auto& cv = t2.get().image_transform.curves;
            if (!cv.enable)
                return L"201 MIXER OK\r\nDISABLED\r\n";
            auto dump = [](const core::curve_channel& c) -> std::wstring {
                std::wstring s;
                for (int i = 0; i < c.count; ++i)
                    s += std::to_wstring(c.points[i].x) + L" " + std::to_wstring(c.points[i].y) + L" ";
                return s;
            };
            return L"201 MIXER OK\r\nMASTER " + dump(cv.master) +
                   L"\r\nR "     + dump(cv.red)   +
                   L"\r\nG "     + dump(cv.green) +
                   L"\r\nB "     + dump(cv.blue)  + L"\r\n";
        });
    }

    if (boost::iequals(ctx.parameters.at(0), L"RESET")) {
        transforms_applier transforms(ctx);
        transforms.add(stage::transform_tuple_t(
            ctx.layer_index(),
            [](frame_transform t) { t.image_transform.curves = core::tone_curves{}; return t; },
            0, L"linear"));
        transforms.apply();
        return make_ready_future<std::wstring>(L"202 MIXER OK\r\n");
    }

    const std::wstring& ch_str = ctx.parameters.at(0);
    int ch = -1;
    if      (boost::iequals(ch_str, L"MASTER"))                                ch = 0;
    else if (boost::iequals(ch_str, L"R") || boost::iequals(ch_str, L"RED"))   ch = 1;
    else if (boost::iequals(ch_str, L"G") || boost::iequals(ch_str, L"GREEN")) ch = 2;
    else if (boost::iequals(ch_str, L"B") || boost::iequals(ch_str, L"BLUE"))  ch = 3;

    if (ch < 0)
        return make_ready_future<std::wstring>(L"400 ERROR\r\n");

    if (ctx.parameters.size() == 1) {
        auto t2 = get_current_transform(ctx).share();
        return std::async(std::launch::deferred, [t2, ch]() -> std::wstring {
            const auto& cv = t2.get().image_transform.curves;
            const core::curve_channel& cc = (ch == 0) ? cv.master
                                          : (ch == 1) ? cv.red
                                          : (ch == 2) ? cv.green
                                          :              cv.blue;
            std::wstring s = L"201 MIXER OK\r\n";
            for (int i = 0; i < cc.count; ++i)
                s += std::to_wstring(cc.points[i].x) + L" " + std::to_wstring(cc.points[i].y) + L" ";
            return s + L"\r\n";
        });
    }

    int n_params = static_cast<int>(ctx.parameters.size()) - 1;
    if (n_params < 4 || n_params % 2 != 0 || n_params / 2 > 16)
        return make_ready_future<std::wstring>(L"400 ERROR\r\n");

    core::curve_channel new_cc;
    new_cc.count = n_params / 2;
    for (int i = 0; i < new_cc.count; ++i) {
        new_cc.points[i].x = grade_param(ctx.parameters.at(1 + i * 2),
                                         core::grade_limits::curve_coord, L"curve x");
        new_cc.points[i].y = grade_param(ctx.parameters.at(2 + i * 2),
                                         core::grade_limits::curve_coord, L"curve y");
    }

    transforms_applier transforms(ctx);
    transforms.add(stage::transform_tuple_t(
        ctx.layer_index(),
        [=](frame_transform transform) -> frame_transform {
            auto& cv = transform.image_transform.curves;
            cv.enable = true;
            switch (ch) {
                case 0: cv.master = new_cc; break;
                case 1: cv.red    = new_cc; break;
                case 2: cv.green  = new_cc; break;
                case 3: cv.blue   = new_cc; break;
            }
            return transform;
        },
        0, L"linear"));
    transforms.apply();
    return make_ready_future<std::wstring>(L"202 MIXER OK\r\n");
}

std::future<std::wstring> mixer_volume_command(command_context& ctx)
{
    return single_double_animatable_mixer_command(
        ctx,
        [](const frame_transform& t) { return t.audio_transform.volume; },
        [](frame_transform& t, double value) { t.audio_transform.volume = value; });
}

std::wstring mixer_mastervolume_command(command_context& ctx)
{
    if (ctx.parameters.empty()) {
        auto volume = ctx.channel.raw_channel->mixer().get_master_volume();
        return L"201 MIXER OK\r\n" + std::to_wstring(volume) + L"\r\n";
    }

    float master_volume = boost::lexical_cast<float>(ctx.parameters.at(0));
    ctx.channel.raw_channel->mixer().set_master_volume(master_volume);

    return L"202 MIXER OK\r\n";
}

std::wstring mixer_grid_command(command_context& ctx)
{
    transforms_applier transforms(ctx);
    int                duration = ctx.parameters.size() > 1 ? std::stoi(ctx.parameters[1]) : 0;
    std::wstring       tween    = ctx.parameters.size() > 2 ? ctx.parameters[2] : L"linear";
    int                n        = std::stoi(ctx.parameters.at(0));
    double             delta    = 1.0 / static_cast<double>(n);
    for (int x = 0; x < n; ++x) {
        for (int y = 0; y < n; ++y) {
            int index = x + y * n + 1;
            transforms.add(stage::transform_tuple_t(
                index,
                [=](frame_transform transform) -> frame_transform {
                    transform.image_transform.fill_translation[0] = x * delta;
                    transform.image_transform.fill_translation[1] = y * delta;
                    transform.image_transform.fill_scale[0]       = delta;
                    transform.image_transform.fill_scale[1]       = delta;
                    transform.image_transform.clip_translation[0] = x * delta;
                    transform.image_transform.clip_translation[1] = y * delta;
                    transform.image_transform.clip_scale[0]       = delta;
                    transform.image_transform.clip_scale[1]       = delta;
                    return transform;
                },
                duration,
                tween));
        }
    }
    transforms.apply();

    return L"202 MIXER OK\r\n";
}

std::future<std::wstring> mixer_commit_command(command_context& ctx)
{
    transforms_applier transforms(ctx);
    const auto         r = transforms.commit_deferred().share();
    return std::async(std::launch::deferred, [r]() -> std::wstring {
        r.get();
        return L"202 MIXER OK\r\n";
    });
}

std::wstring mixer_clear_command(command_context& ctx)
{
    int layer = ctx.layer_id;

    if (layer == -1)
        ctx.channel.stage->clear_transforms();
    else
        ctx.channel.stage->clear_transforms(layer);

    return L"202 MIXER OK\r\n";
}

std::wstring channel_grid_command(command_context& ctx)
{
    int   index = 1;
    auto& self  = ctx.channels->back();

    core::diagnostics::scoped_call_context save;
    core::diagnostics::call_context::for_thread().video_channel = ctx.channels->size();

    std::vector<std::wstring> params;
    params.emplace_back(L"SCREEN");
    params.emplace_back(L"0");
    params.emplace_back(L"NAME");
    params.emplace_back(L"Channel Grid Window");
    auto screen =
        ctx.static_context->consumer_registry->create_consumer(params,
                                                               ctx.static_context->format_repository,
                                                               get_channels(ctx),
                                                               self.raw_channel->get_consumer_channel_info());

    self.raw_channel->output().add(screen);

    for (auto& ch : *ctx.channels) {
        if (ch.raw_channel != self.raw_channel) {
            core::diagnostics::call_context::for_thread().layer = index;
            auto producer = ctx.static_context->producer_registry->create_producer(
                get_producer_dependencies(self.raw_channel, ctx),
                L"route://" + std::to_wstring(ch.raw_channel->index()));
            self.stage->load(index, producer, false);
            self.stage->play(index);
            index++;
        }
    }

    auto num_channels       = ctx.channels->size() - 1;
    int  square_side_length = std::ceil(std::sqrt(num_channels));

    auto ctx2 =
        command_context(ctx.static_context, ctx.channels, ctx.client, self, self.raw_channel->index(), ctx.layer_id);
    ctx2.parameters.push_back(std::to_wstring(square_side_length));
    mixer_grid_command(ctx2);

    return L"202 CHANNEL_GRID OK\r\n";
}

// Thumbnail Commands

std::wstring make_request(command_context& ctx, const std::string& path, const std::wstring& default_response)
{
    auto res = http::request(ctx.static_context->proxy_host, ctx.static_context->proxy_port, path);
    if (res.status_code >= 500 || res.body.size() == 0) {
        CASPAR_LOG(error) << "Failed to connect to media-scanner. Is it running? \nReason: " << res.status_message;
        return default_response;
    }
    return u16(res.body);
}

std::wstring thumbnail_list_command(command_context& ctx)
{
    return make_request(ctx, "/thumbnail", L"501 THUMBNAIL LIST FAILED\r\n");
}

std::wstring thumbnail_retrieve_command(command_context& ctx)
{
    return make_request(
        ctx, "/thumbnail/" + http::url_encode(u8(ctx.parameters.at(0))), L"501 THUMBNAIL RETRIEVE FAILED\r\n");
}

std::wstring thumbnail_generate_command(command_context& ctx)
{
    return make_request(
        ctx, "/thumbnail/generate/" + http::url_encode(u8(ctx.parameters.at(0))), L"501 THUMBNAIL GENERATE FAILED\r\n");
}

std::wstring thumbnail_generateall_command(command_context& ctx)
{
    return make_request(ctx, "/thumbnail/generate", L"501 THUMBNAIL GENERATE_ALL FAILED\r\n");
}

// Query Commands

std::wstring cinf_command(command_context& ctx)
{
    return make_request(ctx, "/cinf/" + http::url_encode(u8(ctx.parameters.at(0))), L"501 CINF FAILED\r\n");
}

std::wstring cls_command(command_context& ctx) { return make_request(ctx, "/cls", L"501 CLS FAILED\r\n"); }

std::wstring fls_command(command_context& ctx) { return make_request(ctx, "/fls", L"501 FLS FAILED\r\n"); }

std::wstring tls_command(command_context& ctx) { return make_request(ctx, "/tls", L"501 TLS FAILED\r\n"); }

std::wstring version_command(command_context& ctx) { return L"201 VERSION OK\r\n" + env::version() + L"\r\n"; }

struct param_visitor : public boost::static_visitor<void>
{
    std::wstring path;
    pt::wptree&  o;

    template <typename T>
    param_visitor(std::string path, T& o)
        : path(u16(path))
        , o(o)
    {
    }

    void operator()(const bool value) { o.add(path, value); }

    void operator()(const int32_t value) { o.add(path, value); }

    void operator()(const uint32_t value) { o.add(path, value); }

    void operator()(const int64_t value) { o.add(path, value); }

    void operator()(const uint64_t value) { o.add(path, value); }

    void operator()(const float value) { o.add(path, value); }

    void operator()(const double value) { o.add(path, value); }

    void operator()(const std::string& value) { o.add(path, u16(value)); }

    void operator()(const std::wstring& value) { o.add(path, value); }
};

std::wstring info_channel_command(command_context& ctx)
{
    pt::wptree info;
    pt::wptree channel_info;

    auto state = ctx.channel.raw_channel->state();
    for (const auto& p : state) {
        const auto replaced = boost::algorithm::replace_all_copy(p.first, "/", ".");
        // avoid digit-only nodes in XML
        const auto path = boost::algorithm::replace_all_regex_copy(
            replaced, boost::regex("\\.(.*?)\\.([0-9]*?)\\."), std::string(".$1.$1_$2."));
        param_visitor param_visitor(path, channel_info);
        for (const auto& element : p.second) {
            boost::apply_visitor(param_visitor, element);
        }
    }

    info.add_child(L"channel", channel_info);

    std::wstringstream replyString;
    // This is needed for backwards compatibility with old clients
    replyString << L"201 INFO OK\r\n";

    pt::xml_writer_settings<std::wstring> w(' ', 3);
    pt::xml_parser::write_xml(replyString, info, w);

    replyString << L"\r\n";
    return replyString.str();
}

std::wstring info_command(command_context& ctx)
{
    std::wstringstream replyString;
    // This is needed for backwards compatibility with old clients
    replyString << L"200 INFO OK\r\n";

    for (auto& ch : *ctx.channels) {
        replyString << ch.raw_channel->index() << L" " << ch.raw_channel->stage()->video_format_desc().name
                    << L" PLAYING\r\n";
    }
    replyString << L"\r\n";
    return replyString.str();
}

std::wstring info_config_command(command_context& ctx)
{
    std::wstringstream replyString;
    // This is needed for backwards compatibility with old clients
    replyString << L"201 INFO CONFIG OK\r\n";

    pt::xml_writer_settings<std::wstring> w(' ', 3);
    pt::xml_parser::write_xml(replyString, caspar::env::properties(), w);

    replyString << L"\r\n";
    return replyString.str();
}

std::wstring info_paths_command(command_context& ctx)
{
    boost::property_tree::wptree info;

    info.add(L"paths.media-path", caspar::env::media_folder());
    info.add(L"paths.log-path", caspar::env::log_folder());
    info.add(L"paths.data-path", caspar::env::data_folder());
    info.add(L"paths.template-path", caspar::env::template_folder());
    info.add(L"paths.initial-path", caspar::env::initial_folder() + L"/");

    std::wstringstream replyString;
    // This is needed for backwards compatibility with old clients
    replyString << L"201 INFO PATHS OK\r\n";

    pt::xml_writer_settings<std::wstring> w(' ', 3);
    pt::xml_parser::write_xml(replyString, info, w);

    replyString << L"\r\n";
    return replyString.str();
}

// INFO OCIO               -> whether OCIO is available, its version, the loaded config URI
// INFO OCIO COLORSPACES   -> every colour space name in the loaded config
// INFO OCIO DISPLAYS      -> every display, with the views available for each
// INFO OCIO LOOKS         -> every look (LMT) name in the loaded config
//
// One command branching on its argument rather than three registered commands, because
// AMCP's dispatcher resolves exactly one level of subcommand: find_command() tries
// `name + " " + tokens.front()` and nothing deeper, so "INFO OCIO COLORSPACES" registered
// as a name is unreachable -- it matches "INFO OCIO" with COLORSPACES arriving as a
// parameter. Registering it anyway silently does nothing, which is how this was first
// written and only testing caught it.
//
// These exist so a client can populate its controls from the server rather than from a
// hardcoded list. The operator-facing surface goes from ~20 documented enums to hundreds of
// config-defined strings, and the mitigation is that the operator never types one -- which
// only holds if the client can discover what THIS server actually has. Without them, a
// client's lists drift from the server's config and a show file silently references a
// colour space that no longer exists.
//
// The version is reported alongside the config URI on purpose: together they determine what
// a colour space name means, so a client can warn when they are not what a stored look was
// approved against.
std::wstring info_ocio_command(command_context& ctx)
{
    const auto what =
        ctx.parameters.empty() ? std::wstring() : boost::to_upper_copy(ctx.parameters.at(0));

    boost::property_tree::wptree info;
    std::wstring                 reply_name = L"INFO OCIO";

    if (what.empty()) {
        const bool have = accelerator::ocio::available();
        info.add(L"ocio.available", have ? L"true" : L"false");

        if (have) {
            info.add(L"ocio.version", u16(accelerator::ocio::version()));
            info.add(L"ocio.config", u16(accelerator::ocio::config_uri()));
            info.add(L"ocio.colorspace-count", std::to_wstring(accelerator::ocio::colorspaces().size()));
            info.add(L"ocio.display-count", std::to_wstring(accelerator::ocio::displays().size()));
        }
    } else if (what == L"COLORSPACES") {
        reply_name = L"INFO OCIO COLORSPACES";
        for (const auto& name : accelerator::ocio::colorspaces())
            info.add(L"colorspaces.colorspace", u16(name));
    } else if (what == L"DISPLAYS") {
        reply_name = L"INFO OCIO DISPLAYS";
        for (const auto& display : accelerator::ocio::displays()) {
            boost::property_tree::wptree node;
            node.add(L"name", u16(display));
            // Views are nested under their display rather than listed flat: the same view
            // name can appear under several displays and means a different transform in
            // each, so a flat list would be ambiguous for exactly the client that has to
            // build a menu from it.
            for (const auto& view : accelerator::ocio::views(display))
                node.add(L"views.view", u16(view));

            info.add_child(L"displays.display", node);
        }
    } else if (what == L"LOOKS") {
        reply_name = L"INFO OCIO LOOKS";
        // Flat, unlike displays: a look name is unique across the config, and OCIO_LOOK
        // takes exactly this string.
        for (const auto& name : accelerator::ocio::looks())
            info.add(L"looks.look", u16(name));
    } else {
        return L"403 INFO OCIO ERROR\r\n";
    }

    std::wstringstream replyString;
    replyString << L"201 " << reply_name << L" OK\r\n";

    pt::xml_writer_settings<std::wstring> w(' ', 3);
    pt::xml_parser::write_xml(replyString, info, w);

    replyString << L"\r\n";
    return replyString.str();
}

std::wstring info_ltc_command(command_context& ctx)
{
    boost::property_tree::wptree info;

    info.add(L"ltc.timecode", caspar::u16(caspar::ltc::LTCInput::instance().get_current_timecode_string()));
    
    // Convert bool manually to string, property tree might output 0/1 or true/false depending on locale
    info.add(L"ltc.valid", caspar::ltc::LTCInput::instance().is_valid() ? L"true" : L"false");
    info.add(L"ltc.source", caspar::ltc::LTCInput::instance().is_using_system_clock() ? L"System Clock" : L"LTC");
    info.add(L"ltc.device", caspar::u16(caspar::ltc::LTCInput::instance().get_current_device_name()));
    
    std::vector<std::string> devices = caspar::ltc::LTCInput::instance().get_capture_devices();
    for (const auto& dev : devices) {
        info.add(L"ltc.devices.device", caspar::u16(dev));
    }

    std::wstringstream replyString;
    replyString << L"201 INFO LTC OK\r\n";

    pt::xml_writer_settings<std::wstring> w(' ', 3);
    pt::xml_parser::write_xml(replyString, info, w);

    replyString << L"\r\n";
    return replyString.str();
}

std::wstring info_portaudio_command(command_context& ctx)
{
    boost::property_tree::wptree info;

    auto& mgr = caspar::portaudio::portaudio_device_manager::instance();
    if (!mgr.is_initialized()) {
        info.add(L"portaudio.status", L"not initialized");
    } else {
        info.add(L"portaudio.status", L"initialized");

        auto outputs = mgr.enumerate_output_devices();
        for (const auto& dev : outputs) {
            boost::property_tree::wptree device_node;
            device_node.put(L"index", dev.index);
            device_node.put(L"name", caspar::u16(dev.name));
            device_node.put(L"host-api", caspar::u16(dev.host_api_name));
            device_node.put(L"channels", dev.max_output_channels);
            device_node.put(L"sample-rate", dev.default_sample_rate);
            device_node.put(L"latency", dev.default_low_output_latency);
            info.add_child(L"portaudio.output-devices.device", device_node);
        }

        auto inputs = mgr.enumerate_input_devices();
        for (const auto& dev : inputs) {
            boost::property_tree::wptree device_node;
            device_node.put(L"index", dev.index);
            device_node.put(L"name", caspar::u16(dev.name));
            device_node.put(L"host-api", caspar::u16(dev.host_api_name));
            device_node.put(L"channels", dev.max_input_channels);
            device_node.put(L"sample-rate", dev.default_sample_rate);
            device_node.put(L"latency", dev.default_low_input_latency);
            info.add_child(L"portaudio.input-devices.device", device_node);
        }
    }

    std::wstringstream replyString;
    replyString << L"201 INFO PORTAUDIO OK\r\n";

    pt::xml_writer_settings<std::wstring> w(' ', 3);
    pt::xml_parser::write_xml(replyString, info, w);

    replyString << L"\r\n";
    return replyString.str();
}

std::wstring ltc_load_command(command_context& ctx)
{
    if (ctx.parameters.size() < 1)
        return L"400 ERROR\r\n";

    std::string device_name = caspar::u8(ctx.parameters[0]);
    
    if (caspar::ltc::LTCInput::instance().set_capture_device(device_name)) {
        return L"202 LTC LOAD OK\r\n";
    }
    
    return L"404 LTC LOAD ERROR\r\n";
}

std::wstring diag_command(command_context& ctx)
{
    core::diagnostics::osd::show_graphs(true);

    return L"202 DIAG OK\r\n";
}

std::wstring bye_command(command_context& ctx)
{
    ctx.client->disconnect();
    return L"";
}

std::wstring kill_command(command_context& ctx)
{
    ctx.static_context->shutdown_server_now(false); // false for not attempting to restart
    return L"202 KILL OK\r\n";
}

std::wstring restart_command(command_context& ctx)
{
    ctx.static_context->shutdown_server_now(true); // true for attempting to restart
    return L"202 RESTART OK\r\n";
}

std::wstring lock_command(command_context& ctx)
{
    int  channel_index = std::stoi(ctx.parameters.at(0)) - 1;
    auto lock          = ctx.channels->at(channel_index).lock;
    auto command       = boost::to_upper_copy(ctx.parameters.at(1));

    if (command == L"ACQUIRE") {
        std::wstring lock_phrase = ctx.parameters.at(2);

        // TODO: read options

        // just lock one channel
        if (!lock->try_lock(lock_phrase, ctx.client))
            return L"503 LOCK ACQUIRE FAILED\r\n";

        return L"202 LOCK ACQUIRE OK\r\n";
    }
    if (command == L"RELEASE") {
        lock->release_lock(ctx.client);
        return L"202 LOCK RELEASE OK\r\n";
    }
    if (command == L"CLEAR") {
        std::wstring override_phrase = env::properties().get(L"configuration.lock-clear-phrase", L"");
        std::wstring client_override_phrase;

        if (!override_phrase.empty())
            client_override_phrase = ctx.parameters.at(2);

        // just clear one channel
        if (client_override_phrase != override_phrase)
            return L"503 LOCK CLEAR FAILED\r\n";

        lock->clear_locks();

        return L"202 LOCK CLEAR OK\r\n";
    }

    CASPAR_THROW_EXCEPTION(file_not_found() << msg_info(L"Unknown LOCK command " + command));
}

std::wstring gl_info_command(command_context& ctx)
{
    auto device = ctx.static_context->ogl_device.lock();
    if (!device)
        CASPAR_THROW_EXCEPTION(not_supported() << msg_info("GL command only supported with OpenGL accelerator."));

    std::wstringstream result;
    result << L"201 GL INFO OK\r\n";

    pt::xml_writer_settings<std::wstring> w(' ', 3);
    pt::xml_parser::write_xml(result, device->info(), w);
    result << L"\r\n";

    return result.str();
}

std::wstring gl_gc_command(command_context& ctx)
{
    auto device = ctx.static_context->ogl_device.lock();
    if (!device)
        CASPAR_THROW_EXCEPTION(not_supported() << msg_info("GL command only supported with OpenGL accelerator."));

    device->gc().wait();

    return L"202 GL GC OK\r\n";
}

std::wstring get_osc_subscription_token(unsigned short port)
{
    std::wstringstream token;
    token << "osc-sub-" << port;
    return token.str();
}

std::wstring osc_subscribe_command(command_context& ctx)
{
    using namespace boost::asio::ip;

    unsigned short port = 0;
    try {
        port = std::stoi(ctx.parameters.at(0));
    } catch (...) {
        return L"403 OSC SUBSCRIBE BAD PORT\r\n";
    }

    auto subscription = ctx.static_context->osc_client->get_subscription_token(
        udp::endpoint(make_address_v4(u8(ctx.client->address())), port));

    ctx.client->add_lifecycle_bound_object(get_osc_subscription_token(port), subscription);

    return L"202 OSC SUBSCRIBE OK\r\n";
}

std::wstring osc_unsubscribe_command(command_context& ctx)
{
    unsigned short port = 0;
    try {
        port = std::stoi(ctx.parameters.at(0));
    } catch (...) {
        return L"403 OSC UNSUBSCRIBE BAD PORT\r\n";
    }

    ctx.client->remove_lifecycle_bound_object(get_osc_subscription_token(port));

    return L"202 OSC UNSUBSCRIBE OK\r\n";
}

// -------- Previz commands --------------------------------------------------

static accelerator::ogl::previz_renderer* get_previz_renderer(command_context& ctx)
{
    auto img = ctx.channel.raw_channel->mixer().get_image_mixer();

    // Try OGL mixer first
    auto* ogl_mix = dynamic_cast<accelerator::ogl::image_mixer*>(img.get());
    if (ogl_mix)
        return &ogl_mix->get_previz_renderer();

#ifdef ENABLE_VULKAN
    // Try VK mixer
    auto* vk_mix = dynamic_cast<accelerator::vulkan::image_mixer*>(img.get());
    if (vk_mix)
        return vk_mix->get_previz_renderer();
#endif

    return nullptr;
}

// Backward-compat alias — existing command functions use this name
static accelerator::ogl::previz_renderer* get_ogl_mixer(command_context& ctx)
{
    return get_previz_renderer(ctx);
}

std::wstring previz_scene_command(command_context& ctx)
{
    auto* ogl_mix = get_ogl_mixer(ctx);
    if (!ogl_mix)
        return L"501 PREVIZ FAILED\r\n";

    if (ctx.parameters.empty()) {
        // Query
        auto sc = ogl_mix->scene();
        if (sc.scene_path.empty())
            return L"201 PREVIZ OK\r\nNONE\r\n\r\n";
        return L"201 PREVIZ OK\r\n" + u16(sc.scene_path) + L"\r\n\r\n";
    }

    auto path_param = ctx.parameters.at(0);

    // PREVIZ ch SCENE NONE — clear scene
    if (boost::iequals(path_param, L"NONE")) {
        ogl_mix->load_scene("");
        return L"202 PREVIZ OK\r\n";
    }

    // PREVIZ ch SCENE SAVE <path> / LOAD <path> — JSON stage-layout persistence
    if (boost::iequals(path_param, L"SAVE") || boost::iequals(path_param, L"LOAD")) {
        if (ctx.parameters.size() < 2)
            return L"400 PREVIZ ERROR usage: SCENE SAVE|LOAD <path>\r\n";
        bool is_save     = boost::iequals(path_param, L"SAVE");
        auto media_base  = boost::filesystem::canonical(env::media_folder());
        auto resolved    = boost::filesystem::path(media_base) / ctx.parameters.at(1);
        // Guard against path traversal outside the media folder. weakly_canonical
        // (not lexically_normal) because SAVE's target need not exist yet, but only
        // a real canonicalization resolves a symlink planted inside the media
        // folder -- which is what otherwise makes this a write-anywhere. Same
        // treatment as PRINT RAW, whose output file likewise does not exist yet.
        resolved = boost::filesystem::weakly_canonical(resolved);
        if (!is_within_base(resolved, media_base))
            return L"403 PREVIZ FORBIDDEN\r\n";
        try {
            if (is_save) {
                ogl_mix->save_layout(u8(resolved.wstring()));
            } else {
                if (!boost::filesystem::exists(resolved))
                    return L"404 PREVIZ ERROR\r\n";
                ogl_mix->load_layout(u8(resolved.wstring()));
            }
            return L"202 PREVIZ OK\r\n";
        } catch (const std::exception& e) {
            CASPAR_LOG(error) << L"[PREVIZ SCENE " << path_param << L"] " << e.what();
            return L"502 PREVIZ FAILED\r\n";
        }
    }

    // Resolve path relative to media folder (prevent path traversal)
    auto media_base = boost::filesystem::canonical(env::media_folder());
    auto resolved   = media_base / path_param;
    if (!boost::filesystem::exists(resolved))
        return L"404 PREVIZ ERROR\r\n";
    resolved = boost::filesystem::canonical(resolved);
    if (!is_within_base(resolved, media_base))
        return L"403 PREVIZ FORBIDDEN\r\n";

    try {
        ogl_mix->load_scene(u8(resolved.wstring()));
        return L"202 PREVIZ OK\r\n";
    } catch (const std::exception& e) {
        CASPAR_LOG(error) << L"[PREVIZ SCENE] " << e.what();
        return L"502 PREVIZ FAILED\r\n";
    }
}

std::wstring previz_map_command(command_context& ctx)
{
    auto* ogl_mix = get_ogl_mixer(ctx);
    if (!ogl_mix)
        return L"501 PREVIZ FAILED\r\n";

    if (ctx.parameters.size() < 2)
        return L"400 PREVIZ ERROR\r\n";

    auto mesh_name   = u8(ctx.parameters.at(0));
    auto channel_str = ctx.parameters.at(1);

    try {
        int  target_ch   = std::stoi(u8(channel_str));
        ogl_mix->map_mesh(mesh_name, target_ch);
        return L"202 PREVIZ OK\r\n";
    } catch (const std::exception& e) {
        CASPAR_LOG(error) << L"[PREVIZ MAP] " << e.what();
        return L"502 PREVIZ FAILED\r\n";
    }
}

std::wstring previz_unmap_command(command_context& ctx)
{
    auto* ogl_mix = get_ogl_mixer(ctx);
    if (!ogl_mix)
        return L"501 PREVIZ FAILED\r\n";

    if (ctx.parameters.empty())
        return L"400 PREVIZ ERROR\r\n";

    try {
        ogl_mix->unmap_mesh(u8(ctx.parameters.at(0)));
        return L"202 PREVIZ OK\r\n";
    } catch (const std::exception& e) {
        CASPAR_LOG(error) << L"[PREVIZ UNMAP] " << e.what();
        return L"502 PREVIZ FAILED\r\n";
    }
}

std::wstring previz_camera_command(command_context& ctx)
{
    auto* ogl_mix = get_ogl_mixer(ctx);
    if (!ogl_mix)
        return L"501 PREVIZ FAILED\r\n";

    // Query
    if (ctx.parameters.empty()) {
        auto cam = ogl_mix->scene().camera;
        std::wostringstream os;
        os << L"201 PREVIZ OK\r\n"
           << cam.x << L" " << cam.y << L" " << cam.z << L" "
           << cam.yaw << L" " << cam.pitch << L" " << cam.roll << L" "
           << cam.fov << L"\r\n\r\n";
        return os.str();
    }

    // RESET
    if (boost::iequals(ctx.parameters.at(0), L"RESET")) {
        ogl_mix->reset_camera();
        return L"202 PREVIZ OK\r\n";
    }

    // OVERRIDE 1|0 — freeze/unfreeze tracker control of the production camera
    if (boost::iequals(ctx.parameters.at(0), L"OVERRIDE")) {
        bool lock = (ctx.parameters.size() < 2) || ctx.parameters.at(1) != L"0";
        ogl_mix->set_camera_locked(lock);
        return L"202 PREVIZ OK\r\n";
    }

    // SET: PREVIZ ch CAMERA x y z yaw pitch roll fov
    if (ctx.parameters.size() < 7)
        return L"400 PREVIZ ERROR\r\n";

    try {
        float x     = std::stof(u8(ctx.parameters.at(0)));
        float y     = std::stof(u8(ctx.parameters.at(1)));
        float z     = std::stof(u8(ctx.parameters.at(2)));
        float yaw   = std::stof(u8(ctx.parameters.at(3)));
        float pitch = std::stof(u8(ctx.parameters.at(4)));
        float roll  = std::stof(u8(ctx.parameters.at(5)));
        float fov   = std::stof(u8(ctx.parameters.at(6)));

        ogl_mix->set_camera(x, y, z, yaw, pitch, roll, fov);
        return L"202 PREVIZ OK\r\n";
    } catch (const std::exception& e) {
        CASPAR_LOG(error) << L"[PREVIZ CAMERA] " << e.what();
        return L"502 PREVIZ FAILED\r\n";
    }
}

std::wstring previz_view_command(command_context& ctx)
{
    auto* ogl_mix = get_ogl_mixer(ctx);
    if (!ogl_mix)
        return L"501 PREVIZ FAILED\r\n";

    // Query
    if (ctx.parameters.empty()) {
        auto cam = ogl_mix->scene().view_camera;
        bool ov  = ogl_mix->scene().has_view_override;
        std::wostringstream os;
        os << L"201 PREVIZ OK\r\n"
           << (ov ? 1 : 0) << L" "
           << cam.x << L" " << cam.y << L" " << cam.z << L" "
           << cam.yaw << L" " << cam.pitch << L" " << cam.roll << L" "
           << cam.fov << L"\r\n\r\n";
        return os.str();
    }

    // CLEAR / RESET: drop the viewport override (render follows production camera)
    if (boost::iequals(ctx.parameters.at(0), L"CLEAR") ||
        boost::iequals(ctx.parameters.at(0), L"RESET")) {
        ogl_mix->clear_view_camera();
        return L"202 PREVIZ OK\r\n";
    }

    // SET: PREVIZ ch VIEW x y z yaw pitch roll fov
    if (ctx.parameters.size() < 7)
        return L"400 PREVIZ ERROR\r\n";

    try {
        float x     = std::stof(u8(ctx.parameters.at(0)));
        float y     = std::stof(u8(ctx.parameters.at(1)));
        float z     = std::stof(u8(ctx.parameters.at(2)));
        float yaw   = std::stof(u8(ctx.parameters.at(3)));
        float pitch = std::stof(u8(ctx.parameters.at(4)));
        float roll  = std::stof(u8(ctx.parameters.at(5)));
        float fov   = std::stof(u8(ctx.parameters.at(6)));

        ogl_mix->set_view_camera(x, y, z, yaw, pitch, roll, fov);
        return L"202 PREVIZ OK\r\n";
    } catch (const std::exception& e) {
        CASPAR_LOG(error) << L"[PREVIZ VIEW] " << e.what();
        return L"502 PREVIZ FAILED\r\n";
    }
}

std::wstring previz_info_command(command_context& ctx)
{
    auto* ogl_mix = get_ogl_mixer(ctx);
    if (!ogl_mix)
        return L"501 PREVIZ FAILED\r\n";

    auto sc = ogl_mix->scene();
    std::wostringstream os;
    os << L"201 PREVIZ OK\r\n";
    os << L"active: "    << (sc.active ? L"true" : L"false") << L"\r\n";
    os << L"scene: "     << (sc.scene_path.empty() ? L"NONE" : u16(sc.scene_path)) << L"\r\n";
    os << L"meshes: "    << sc.meshes.size() << L"\r\n";
    for (auto& m : sc.meshes) {
        os << L"  " << u16(m.name);
        auto it = sc.mesh_to_channel.find(m.name);
        if (it != sc.mesh_to_channel.end())
            os << L" -> channel " << it->second;
        os << L"\r\n";
    }
    os << L"\r\n";   // blank line terminates the multi-line response
    return os.str();
}

// ---------------------------------------------------------------------------
// PREVIZ SHOW <mesh_name> [1|0]
// ---------------------------------------------------------------------------
std::wstring previz_show_command(command_context& ctx)
{
    auto* ogl_mix = get_ogl_mixer(ctx);
    if (!ogl_mix)
        return L"501 PREVIZ FAILED\r\n";
    if (ctx.parameters.empty())
        return L"400 PREVIZ ERROR missing mesh name\r\n";

    std::string mesh_name = u8(ctx.parameters.at(0));
    bool visible = ctx.parameters.size() >= 2 ? (ctx.parameters.at(1) != L"0") : true;
    ogl_mix->set_mesh_visible(mesh_name, visible);
    return L"202 PREVIZ OK\r\n";
}

// ---------------------------------------------------------------------------
// PREVIZ GRID [1|0]
// ---------------------------------------------------------------------------
std::wstring previz_grid_command(command_context& ctx)
{
    auto* ogl_mix = get_ogl_mixer(ctx);
    if (!ogl_mix)
        return L"501 PREVIZ FAILED\r\n";
    bool on = ctx.parameters.empty() || ctx.parameters.at(0) != L"0";
    ogl_mix->set_grid(on);
    return L"202 PREVIZ OK\r\n";
}

// ---------------------------------------------------------------------------
// PREVIZ WIREFRAME [1|0]
// ---------------------------------------------------------------------------
std::wstring previz_wireframe_command(command_context& ctx)
{
    auto* ogl_mix = get_ogl_mixer(ctx);
    if (!ogl_mix)
        return L"501 PREVIZ FAILED\r\n";
    bool on = ctx.parameters.empty() || ctx.parameters.at(0) != L"0";
    ogl_mix->set_wireframe(on);
    return L"202 PREVIZ OK\r\n";
}

// ---------------------------------------------------------------------------
// PREVIZ GIZMO [1|0]
// ---------------------------------------------------------------------------
std::wstring previz_gizmo_command(command_context& ctx)
{
    auto* ogl_mix = get_ogl_mixer(ctx);
    if (!ogl_mix)
        return L"501 PREVIZ FAILED\r\n";
    bool on = ctx.parameters.empty() || ctx.parameters.at(0) != L"0";
    ogl_mix->set_gizmo(on);
    return L"202 PREVIZ OK\r\n";
}

// ---------------------------------------------------------------------------
// PREVIZ PRESET SAVE <name>
// PREVIZ PRESET RECALL <name>
// PREVIZ PRESET LIST
// ---------------------------------------------------------------------------
std::wstring previz_preset_command(command_context& ctx)
{
    auto* ogl_mix = get_ogl_mixer(ctx);
    if (!ogl_mix)
        return L"501 PREVIZ FAILED\r\n";
    if (ctx.parameters.empty())
        return L"400 PREVIZ ERROR missing subcommand\r\n";

    auto sub = boost::to_upper_copy(ctx.parameters.at(0));
    if (sub == L"SAVE") {
        if (ctx.parameters.size() < 2)
            return L"400 PREVIZ ERROR missing preset name\r\n";
        ogl_mix->save_camera_preset(u8(ctx.parameters.at(1)));
        return L"202 PREVIZ OK\r\n";
    } else if (sub == L"RECALL") {
        if (ctx.parameters.size() < 2)
            return L"400 PREVIZ ERROR missing preset name\r\n";
        ogl_mix->recall_camera_preset(u8(ctx.parameters.at(1)));
        return L"202 PREVIZ OK\r\n";
    } else if (sub == L"LIST") {
        auto names = ogl_mix->list_camera_presets();
        std::wostringstream os;
        os << L"201 PREVIZ OK\r\n";
        for (auto& n : names)
            os << u16(n) << L"\r\n";
        os << L"\r\n";   // blank line terminates the multi-line response
        return os.str();
    }
    return L"400 PREVIZ ERROR unknown preset subcommand\r\n";
}

// ---------------------------------------------------------------------------
// PREVIZ SCREEN ADD <name> FLAT <width_m> <height_m>
// PREVIZ SCREEN ADD <name> CURVED <width_m> <height_m> <radius_m> <arc_deg>
// PREVIZ SCREEN <name> POSITION <x> <y> <z>
// PREVIZ SCREEN <name> ROTATION <yaw> <pitch> <roll>
// PREVIZ SCREEN <name> RESOLUTION <width_px> <height_px>
// PREVIZ SCREEN <name> CHANNEL <ch_num>
// PREVIZ SCREEN <name> REMOVE
// PREVIZ SCREEN LIST
// ---------------------------------------------------------------------------
std::wstring previz_screen_command(command_context& ctx)
{
    auto* ogl_mix = get_ogl_mixer(ctx);
    if (!ogl_mix)
        return L"501 PREVIZ FAILED\r\n";
    if (ctx.parameters.empty())
        return L"400 PREVIZ ERROR missing subcommand\r\n";

    try {
        auto first = boost::to_upper_copy(ctx.parameters.at(0));

        if (first == L"LIST") {
            auto names = ogl_mix->list_screens();
            auto sc   = ogl_mix->scene();
            std::wostringstream os;
            os << L"201 PREVIZ OK\r\n";
            for (auto& n : names) {
                auto it = sc.screens.find(n);
                if (it != sc.screens.end()) {
                    auto& s = it->second;
                    os << u16(n) << L" " << s.width_m << L"x" << s.height_m << L"m";
                    if (s.radius_m > 0)
                        os << L" curved r=" << s.radius_m << L"m";
                    if (s.channel >= 0)
                        os << L" ch=" << s.channel;
                    os << L"\r\n";
                }
            }
            os << L"\r\n";   // blank line terminates the multi-line response
            return os.str();
        }

        if (first == L"ADD") {
            if (ctx.parameters.size() < 5)
                return L"400 PREVIZ ERROR usage: SCREEN ADD <name> FLAT|CURVED <params...>\r\n";

            std::string name = u8(ctx.parameters.at(1));
            auto type        = boost::to_upper_copy(ctx.parameters.at(2));

            if (type == L"FLAT") {
                float w = std::stof(u8(ctx.parameters.at(3)));
                float h = std::stof(u8(ctx.parameters.at(4)));
                ogl_mix->add_screen_flat(name, w, h);
                return L"202 PREVIZ OK\r\n";
            } else if (type == L"CURVED") {
                if (ctx.parameters.size() < 7)
                    return L"400 PREVIZ ERROR usage: SCREEN ADD <name> CURVED <w> <h> <radius> <arc_deg>\r\n";
                float w   = std::stof(u8(ctx.parameters.at(3)));
                float h   = std::stof(u8(ctx.parameters.at(4)));
                float r   = std::stof(u8(ctx.parameters.at(5)));
                float arc = std::stof(u8(ctx.parameters.at(6)));
                ogl_mix->add_screen_curved(name, w, h, r, arc);
                return L"202 PREVIZ OK\r\n";
            }
            return L"400 PREVIZ ERROR unknown screen type (FLAT or CURVED)\r\n";
        }

        // Commands that take <name> as first parameter
        if (ctx.parameters.size() < 2)
            return L"400 PREVIZ ERROR usage: SCREEN <name> <subcommand> [params...]\r\n";

        std::string name = u8(ctx.parameters.at(0));
        auto sub         = boost::to_upper_copy(ctx.parameters.at(1));

        if (sub == L"POSITION") {
            if (ctx.parameters.size() < 5)
                return L"400 PREVIZ ERROR usage: SCREEN <name> POSITION <x> <y> <z>\r\n";
            float x = std::stof(u8(ctx.parameters.at(2)));
            float y = std::stof(u8(ctx.parameters.at(3)));
            float z = std::stof(u8(ctx.parameters.at(4)));
            ogl_mix->set_screen_position(name, x, y, z);
            return L"202 PREVIZ OK\r\n";
        } else if (sub == L"ROTATION") {
            if (ctx.parameters.size() < 5)
                return L"400 PREVIZ ERROR usage: SCREEN <name> ROTATION <yaw> <pitch> <roll>\r\n";
            float yaw   = std::stof(u8(ctx.parameters.at(2)));
            float pitch = std::stof(u8(ctx.parameters.at(3)));
            float roll  = std::stof(u8(ctx.parameters.at(4)));
            ogl_mix->set_screen_rotation(name, yaw, pitch, roll);
            return L"202 PREVIZ OK\r\n";
        } else if (sub == L"RESOLUTION") {
            if (ctx.parameters.size() < 4)
                return L"400 PREVIZ ERROR usage: SCREEN <name> RESOLUTION <w> <h>\r\n";
            int w = std::stoi(u8(ctx.parameters.at(2)));
            int h = std::stoi(u8(ctx.parameters.at(3)));
            ogl_mix->set_screen_resolution(name, w, h);
            return L"202 PREVIZ OK\r\n";
        } else if (sub == L"CHANNEL") {
            if (ctx.parameters.size() < 3)
                return L"400 PREVIZ ERROR usage: SCREEN <name> CHANNEL <ch>\r\n";
            int ch = std::stoi(u8(ctx.parameters.at(2)));
            ogl_mix->set_screen_channel(name, ch);
            return L"202 PREVIZ OK\r\n";
        } else if (sub == L"REMOVE") {
            ogl_mix->remove_screen(name);
            return L"202 PREVIZ OK\r\n";
        } else if (sub == L"EYEMODE") {
            // SCREEN <name> EYEMODE CAMERA | FIXED [x y z]
            if (ctx.parameters.size() < 3)
                return L"400 PREVIZ ERROR usage: SCREEN <name> EYEMODE CAMERA|FIXED [x y z]\r\n";
            auto mode_arg = boost::to_upper_copy(ctx.parameters.at(2));
            int  mode     = (mode_arg == L"FIXED") ? 1 : 0;
            float x = 0.0f, y = 1.5f, z = 3.0f;
            if (mode == 1 && ctx.parameters.size() >= 6) {
                x = std::stof(u8(ctx.parameters.at(3)));
                y = std::stof(u8(ctx.parameters.at(4)));
                z = std::stof(u8(ctx.parameters.at(5)));
            }
            ogl_mix->set_screen_eye_mode(name, mode, x, y, z);
            return L"202 PREVIZ OK\r\n";
        } else if (sub == L"ARCV") {
            // SCREEN <name> ARCV <arc_v_deg>   (0 = single-curved cylinder)
            if (ctx.parameters.size() < 3)
                return L"400 PREVIZ ERROR usage: SCREEN <name> ARCV <arc_v_deg>\r\n";
            float arc_v = std::stof(u8(ctx.parameters.at(2)));
            ogl_mix->set_screen_arc_v(name, arc_v);
            return L"202 PREVIZ OK\r\n";
        } else if (sub == L"ICVFX") {
            // SCREEN <name> ICVFX 1|0   (enable inner/outer frustum in-camera VFX)
            if (ctx.parameters.size() < 3)
                return L"400 PREVIZ ERROR usage: SCREEN <name> ICVFX 1|0\r\n";
            bool enable = (ctx.parameters.at(2) == L"1" ||
                           boost::iequals(ctx.parameters.at(2), L"true"));
            ogl_mix->set_screen_icvfx(name, enable);
            return L"202 PREVIZ OK\r\n";
        }

        return L"400 PREVIZ ERROR unknown screen subcommand\r\n";
    } catch (const std::exception& e) {
        return L"502 PREVIZ FAILED " + u16(e.what()) + L"\r\n";
    }
}

// ---------------------------------------------------------------------------
// PREVIZ AUTOPROJECTION [1|0] [SOURCE <ch>-<layer>]
// Derive MIXER PROJECTION parameters from screen geometry + camera.
//
// Without SOURCE: applies projection to layer 0 of each mapped channel.
// With SOURCE:    routes the source layer (PREMIX) to each non-source
//                 channel on the same layer number, then applies projection.
//                 Decode once, project many.
// ---------------------------------------------------------------------------
std::wstring previz_autoprojection_command(command_context& ctx)
{
    auto* ogl_mix = get_ogl_mixer(ctx);
    if (!ogl_mix)
        return L"501 PREVIZ FAILED\r\n";

    bool on = ctx.parameters.empty() || ctx.parameters.at(0) != L"0";

    if (on) {
        // Parse optional SOURCE <ch>-<layer>
        int  source_channel = 0;
        int  source_layer   = 0;
        bool has_source     = false;

        for (size_t i = 0; i < ctx.parameters.size(); ++i) {
            if (boost::iequals(ctx.parameters[i], L"SOURCE") && i + 1 < ctx.parameters.size()) {
                auto& src = ctx.parameters[i + 1];
                auto  dash = src.find(L'-');
                if (dash != std::wstring::npos) {
                    source_channel = std::stoi(src.substr(0, dash));
                    source_layer   = std::stoi(src.substr(dash + 1));
                    has_source     = true;
                }
                break;
            }
        }

        // Capture weak_ptrs to all channel stages + video_channels
        auto channels = ctx.channels;
        std::map<int, std::weak_ptr<core::stage_base>> stages;
        for (size_t i = 0; i < channels->size(); ++i)
            stages[static_cast<int>(i + 1)] = (*channels)[i].stage;

        accelerator::ogl::projection_apply_fn apply_fn;

        if (has_source) {
            // SOURCE mode: route premix, apply projection on source_layer
            std::map<int, std::weak_ptr<core::video_channel>> video_channels;
            for (size_t i = 0; i < channels->size(); ++i)
                video_channels[static_cast<int>(i + 1)] = (*channels)[i].raw_channel;

            // Shared set to track which channels already have routes loaded
            auto routed_channels = std::make_shared<std::set<int>>();
            auto routed_mutex    = std::make_shared<std::mutex>();

            // Capture producer dependencies for route creation
            auto producer_registry = ctx.static_context->producer_registry;
            auto cg_registry       = ctx.static_context->cg_registry;
            auto format_repository = ctx.static_context->format_repository;

            apply_fn =
                [stages, video_channels, source_channel, source_layer,
                 routed_channels, routed_mutex,
                 producer_registry, cg_registry, format_repository]
                (int channel, const accelerator::ogl::screen_projection& proj) {
                    static const double DEG2RAD = 3.141592653589793 / 180.0;
                    const double yaw   = static_cast<double>(proj.yaw_deg)   * DEG2RAD;
                    const double pitch = static_cast<double>(proj.pitch_deg) * DEG2RAD;
                    const double roll  = static_cast<double>(proj.roll_deg)  * DEG2RAD;
                    const double fov   = static_cast<double>(proj.fov_deg)   * DEG2RAD;
                    try {
                        auto stage_it = stages.find(channel);
                        if (stage_it == stages.end())
                            return;
                        auto stage = stage_it->second.lock();
                        if (!stage)
                            return;

                        int target_layer = source_layer;

                        // For non-source channels, ensure route://src-layer PREMIX is loaded
                        if (channel != source_channel) {
                            bool need_route = false;
                            {
                                std::lock_guard<std::mutex> lock(*routed_mutex);
                                if (routed_channels->find(channel) == routed_channels->end()) {
                                    routed_channels->insert(channel);
                                    need_route = true;
                                }
                            }
                            if (need_route) {
                                // Route creation runs on a background thread to avoid
                                // blocking the previz renderer (which could deadlock
                                // with the stage executor).
                                std::thread([video_channels, source_channel, source_layer, target_layer,
                                             channel, stage, producer_registry, cg_registry, format_repository]() {
                                    try {
                                        auto src_it = video_channels.find(source_channel);
                                        auto dst_it = video_channels.find(channel);
                                        if (src_it == video_channels.end() || dst_it == video_channels.end())
                                            return;
                                        auto src_ch = src_it->second.lock();
                                        auto dst_ch = dst_it->second.lock();
                                        if (!src_ch || !dst_ch)
                                            return;

                                        std::vector<spl::shared_ptr<core::video_channel>> all_chs;
                                        for (auto& [id, wch] : video_channels) {
                                            auto ch = wch.lock();
                                            if (ch)
                                                all_chs.push_back(spl::make_shared_ptr(ch));
                                        }

                                        core::frame_producer_dependencies deps(
                                            dst_ch->frame_factory(),
                                            all_chs,
                                            format_repository,
                                            dst_ch->stage()->video_format_desc(),
                                            producer_registry,
                                            cg_registry);

                                        std::vector<std::wstring> route_params = {
                                            L"route://" + std::to_wstring(source_channel)
                                                + L"-" + std::to_wstring(source_layer),
                                            L"PREMIX"
                                        };

                                        auto producer = producer_registry->create_producer(deps, route_params);
                                        stage->load(target_layer, producer, false, false).get();
                                        stage->play(target_layer).get();

                                        CASPAR_LOG(info) << L"[previz] Auto-routed route://"
                                            << source_channel << L"-" << source_layer
                                            << L" PREMIX to channel " << channel
                                            << L" layer " << target_layer;
                                    } catch (const std::exception& e) {
                                        CASPAR_LOG(error) << L"[previz] Failed to create route for channel "
                                            << channel << L": " << e.what();
                                    } catch (...) {
                                        CASPAR_LOG(error) << L"[previz] Route creation unknown error for channel "
                                            << channel;
                                    }
                                }).detach();
                            }
                        }

                        // Apply projection on the target layer (always, even before
                        // the route finishes loading — the transform will be ready
                        // when frames start arriving).
                        stage->apply_transform(
                            target_layer,
                            [yaw, pitch, roll, fov, proj](core::frame_transform t) -> core::frame_transform {
                                t.image_transform.projection.enable = (fov > 0.0);
                                t.image_transform.projection.yaw    = yaw;
                                t.image_transform.projection.pitch  = pitch;
                                t.image_transform.projection.roll   = roll;
                                t.image_transform.projection.fov    = fov;
                                // Auto-projection owns the curve only while the
                                // layer has not been manually overridden.  An
                                // explicit MIXER PROJECTION_CURVE clears curve_auto
                                // and freezes the operator's values.
                                if (t.image_transform.projection.curve_auto ||
                                    !t.image_transform.projection.curve_enable) {
                                    static const double D2R = 3.141592653589793 / 180.0;
                                    auto& p = t.image_transform.projection;
                                    p.curve_type   = static_cast<core::screen_curve_type>(proj.curve_type);
                                    p.screen_arc   = static_cast<double>(proj.screen_arc_deg)   * D2R;
                                    p.screen_arc_v = static_cast<double>(proj.screen_arc_v_deg) * D2R;
                                    p.eye_distance = static_cast<double>(proj.eye_distance);
                                    p.curve_enable = (proj.curve_type != 0 && proj.screen_arc_deg != 0.0f);
                                    p.curve_auto   = true;
                                }
                                // ICVFX inner/outer frustum (auto path)
                                {
                                    static const double I2R = 3.141592653589793 / 180.0;
                                    auto& p = t.image_transform.projection;
                                    p.icvfx_enable = proj.icvfx_enable;
                                    if (proj.icvfx_enable) {
                                        p.inner_yaw          = static_cast<double>(proj.inner_yaw_deg)   * I2R;
                                        p.inner_pitch        = static_cast<double>(proj.inner_pitch_deg) * I2R;
                                        p.inner_roll         = static_cast<double>(proj.inner_roll_deg)  * I2R;
                                        p.inner_fov          = static_cast<double>(proj.inner_fov_deg)   * I2R;
                                        p.inner_eye_distance = static_cast<double>(proj.inner_eye_distance);
                                        p.icvfx_q0x = proj.icvfx_q[0]; p.icvfx_q0y = proj.icvfx_q[1];
                                        p.icvfx_q1x = proj.icvfx_q[2]; p.icvfx_q1y = proj.icvfx_q[3];
                                        p.icvfx_q2x = proj.icvfx_q[4]; p.icvfx_q2y = proj.icvfx_q[5];
                                        p.icvfx_q3x = proj.icvfx_q[6]; p.icvfx_q3y = proj.icvfx_q[7];
                                    }
                                }
                                return t;
                            },
                            0,
                            tweener(L"linear"));
                    } catch (const std::exception& e) {
                        CASPAR_LOG(error) << L"[previz] Auto-projection callback error for channel "
                            << channel << L": " << e.what();
                    } catch (...) {
                        CASPAR_LOG(error) << L"[previz] Auto-projection callback unknown error for channel "
                            << channel;
                    }
                };

            CASPAR_LOG(info) << L"[previz] Auto-projection SOURCE mode: route://"
                << source_channel << L"-" << source_layer << L" PREMIX";
        } else {
            // Legacy mode: just apply projection to layer 0, no routing
            apply_fn =
                [stages](int channel, const accelerator::ogl::screen_projection& proj) {
                    static const double DEG2RAD = 3.141592653589793 / 180.0;
                    const double yaw   = static_cast<double>(proj.yaw_deg)   * DEG2RAD;
                    const double pitch = static_cast<double>(proj.pitch_deg) * DEG2RAD;
                    const double roll  = static_cast<double>(proj.roll_deg)  * DEG2RAD;
                    const double fov   = static_cast<double>(proj.fov_deg)   * DEG2RAD;
                    auto it = stages.find(channel);
                    if (it == stages.end())
                        return;
                    auto stage = it->second.lock();
                    if (!stage)
                        return;
                    stage->apply_transform(
                        0,
                        [yaw, pitch, roll, fov, proj](core::frame_transform t) -> core::frame_transform {
                            t.image_transform.projection.enable = (fov > 0.0);
                            t.image_transform.projection.yaw    = yaw;
                            t.image_transform.projection.pitch  = pitch;
                            t.image_transform.projection.roll   = roll;
                            t.image_transform.projection.fov    = fov;
                            if (t.image_transform.projection.curve_auto ||
                                !t.image_transform.projection.curve_enable) {
                                static const double D2R = 3.141592653589793 / 180.0;
                                auto& p = t.image_transform.projection;
                                p.curve_type   = static_cast<core::screen_curve_type>(proj.curve_type);
                                p.screen_arc   = static_cast<double>(proj.screen_arc_deg)   * D2R;
                                p.screen_arc_v = static_cast<double>(proj.screen_arc_v_deg) * D2R;
                                p.eye_distance = static_cast<double>(proj.eye_distance);
                                p.curve_enable = (proj.curve_type != 0 && proj.screen_arc_deg != 0.0f);
                                p.curve_auto   = true;
                            }
                            // ICVFX inner/outer frustum (auto path)
                            {
                                static const double I2R = 3.141592653589793 / 180.0;
                                auto& p = t.image_transform.projection;
                                p.icvfx_enable = proj.icvfx_enable;
                                if (proj.icvfx_enable) {
                                    p.inner_yaw          = static_cast<double>(proj.inner_yaw_deg)   * I2R;
                                    p.inner_pitch        = static_cast<double>(proj.inner_pitch_deg) * I2R;
                                    p.inner_roll         = static_cast<double>(proj.inner_roll_deg)  * I2R;
                                    p.inner_fov          = static_cast<double>(proj.inner_fov_deg)   * I2R;
                                    p.inner_eye_distance = static_cast<double>(proj.inner_eye_distance);
                                    p.icvfx_q0x = proj.icvfx_q[0]; p.icvfx_q0y = proj.icvfx_q[1];
                                    p.icvfx_q1x = proj.icvfx_q[2]; p.icvfx_q1y = proj.icvfx_q[3];
                                    p.icvfx_q2x = proj.icvfx_q[4]; p.icvfx_q2y = proj.icvfx_q[5];
                                    p.icvfx_q3x = proj.icvfx_q[6]; p.icvfx_q3y = proj.icvfx_q[7];
                                }
                            }
                            return t;
                        },
                        0,
                        tweener(L"linear"));
                };
        }

        ogl_mix->set_projection_callback(std::move(apply_fn));
        ogl_mix->set_auto_projection(true);
    } else {
        ogl_mix->set_auto_projection(false);
        ogl_mix->set_projection_callback(nullptr);
    }

    return L"202 PREVIZ OK\r\n";
}

void register_commands(std::shared_ptr<amcp_command_repository_wrapper>& repo)
{
    repo->register_channel_command(L"Basic Commands", L"LOADBG", loadbg_command, 1);
    repo->register_channel_command(L"Basic Commands", L"CALLBG", callbg_command, 1);
    repo->register_channel_command(L"Basic Commands", L"LOAD", load_command, 0);
    repo->register_channel_command(L"Basic Commands", L"PLAY", play_command, 0);
    repo->register_channel_command(L"Basic Commands", L"PAUSE", pause_command, 0);
    repo->register_channel_command(L"Basic Commands", L"RESUME", resume_command, 0);
    repo->register_channel_command(L"Basic Commands", L"STOP", stop_command, 0);
    repo->register_channel_command(L"Basic Commands", L"CLEAR", clear_command, 0);
    repo->register_channel_command(L"Basic Commands", L"CALL", call_command, 1);
    repo->register_channel_command(L"Basic Commands", L"SWAP", swap_command, 1);
    repo->register_channel_command(L"Basic Commands", L"ADD", add_command, 1);
    repo->register_channel_command(L"Basic Commands", L"REMOVE", remove_command, 0);
    repo->register_channel_command(L"Basic Commands", L"APPLY", apply_command, 1);
    repo->register_channel_command(L"Basic Commands", L"PRINT", print_command, 0);
    repo->register_channel_command(L"Basic Commands", L"PRINT RAW", print_raw_command, 0);
    repo->register_command(L"Basic Commands", L"CLEAR ALL", clear_all_command, 0);
    repo->register_command(L"Basic Commands", L"LOG LEVEL", log_level_command, 0);
    repo->register_channel_command(L"Basic Commands", L"SET", set_command, 2);
    repo->register_command(L"Basic Commands", L"LOCK", lock_command, 2);

    repo->register_command(L"Data Commands", L"DATA STORE", data_store_command, 2);
    repo->register_command(L"Data Commands", L"DATA RETRIEVE", data_retrieve_command, 1);
    repo->register_command(L"Data Commands", L"DATA LIST", data_list_command, 0);
    repo->register_command(L"Data Commands", L"DATA REMOVE", data_remove_command, 1);

    repo->register_channel_command(L"Template Commands", L"CG ADD", cg_add_command, 3);
    repo->register_channel_command(L"Template Commands", L"CG PLAY", cg_play_command, 1);
    repo->register_channel_command(L"Template Commands", L"CG STOP", cg_stop_command, 1);
    repo->register_channel_command(L"Template Commands", L"CG NEXT", cg_next_command, 1);
    repo->register_channel_command(L"Template Commands", L"CG REMOVE", cg_remove_command, 1);
    repo->register_channel_command(L"Template Commands", L"CG CLEAR", cg_clear_command, 0);
    repo->register_channel_command(L"Template Commands", L"CG UPDATE", cg_update_command, 2);
    repo->register_channel_command(L"Template Commands", L"CG INVOKE", cg_invoke_command, 2);

    repo->register_channel_command(L"Mixer Commands", L"MIXER KEYER", mixer_keyer_command, 0);
    repo->register_channel_command(L"Mixer Commands", L"MIXER INVERT", mixer_invert_command, 0);
    repo->register_channel_command(L"Mixer Commands", L"MIXER CHROMA", mixer_chroma_command, 0);
    repo->register_channel_command(L"Mixer Commands", L"MIXER BLEND", mixer_blend_command, 0);
    repo->register_channel_command(L"Mixer Commands", L"MIXER BLUR", mixer_blur_command, 0);
    repo->register_channel_command(L"Mixer Commands", L"MIXER SHAPE", mixer_shape_command, 0);
    repo->register_channel_command(L"Mixer Commands", L"MIXER OPACITY", mixer_opacity_command, 0);
    repo->register_channel_command(L"Mixer Commands", L"MIXER BRIGHTNESS", mixer_brightness_command, 0);
    repo->register_channel_command(L"Mixer Commands", L"MIXER SATURATION", mixer_saturation_command, 0);
    repo->register_channel_command(L"Mixer Commands", L"MIXER CONTRAST", mixer_contrast_command, 0);
    repo->register_channel_command(L"Mixer Commands", L"MIXER LEVELS", mixer_levels_command, 0);
    repo->register_channel_command(L"Mixer Commands", L"MIXER FILL", mixer_fill_command, 0);
    repo->register_channel_command(L"Mixer Commands", L"MIXER CLIP", mixer_clip_command, 0);
    repo->register_channel_command(L"Mixer Commands", L"MIXER ANCHOR", mixer_anchor_command, 0);
    repo->register_channel_command(L"Mixer Commands", L"MIXER CROP", mixer_crop_command, 0);
    repo->register_channel_command(L"Mixer Commands", L"MIXER ROTATION", mixer_rotation_command, 0);
    repo->register_channel_command(L"Mixer Commands", L"MIXER PERSPECTIVE", mixer_perspective_command, 0);
    repo->register_channel_command(L"Mixer Commands", L"MIXER PROJECTION",        mixer_projection_command,        0);
    repo->register_channel_command(L"Mixer Commands", L"MIXER PROJECTION_OFFSET", mixer_projection_offset_command, 0);
    repo->register_channel_command(L"Mixer Commands", L"MIXER PROJECTION_CURVE",  mixer_projection_curve_command,  0);
    repo->register_channel_command(L"Mixer Commands", L"MIXER PROJECTION_LENS",   mixer_projection_lens_command,   0);
    repo->register_channel_command(L"Mixer Commands", L"MIXER PROJECTION_ICVFX",  mixer_projection_icvfx_command,  0);
    repo->register_channel_command(L"Mixer Commands", L"MIXER PROJECTION_ICVFX_COLOR", mixer_projection_icvfx_color_command, 0);
    repo->register_channel_command(L"Mixer Commands", L"MIXER PROJECTION_FRUSTUM",    mixer_projection_frustum_command,    0);
    repo->register_channel_command(L"Mixer Commands", L"MIXER PROJECTION_DISTORTION", mixer_projection_distortion_command, 0);
    repo->register_channel_command(L"Mixer Commands", L"MIXER PROJECTION_BLEND",      mixer_projection_blend_command,      0);
    repo->register_channel_command(L"Mixer Commands", L"MIXER PROJECTION_BLEND_MASK", mixer_projection_blend_mask_command, 0);
    repo->register_channel_command(L"Mixer Commands", L"MIXER MESH",                  mixer_mesh_command,                  0);
    repo->register_channel_command(L"Mixer Commands", L"MIXER FLIP",             mixer_flip_command,              0);
    repo->register_channel_command(L"Mixer Commands", L"MIXER COLORSPACE",        mixer_colorspace_command,        0);
    repo->register_channel_command(L"Mixer Commands", L"MIXER OCIO",              mixer_ocio_command,              0);
    repo->register_channel_command(L"Mixer Commands", L"MIXER WHITEBALANCE", mixer_whitebalance_command, 0);
    repo->register_channel_command(L"Mixer Commands", L"MIXER LIFT",         mixer_lift_command,         0);
    repo->register_channel_command(L"Mixer Commands", L"MIXER MIDTONE",      mixer_midtone_command,      0);
    repo->register_channel_command(L"Mixer Commands", L"MIXER GAIN",         mixer_gain_command,         0);
    repo->register_channel_command(L"Mixer Commands", L"MIXER HUESHIFT",     mixer_hueshift_command,     0);
    repo->register_channel_command(L"Mixer Commands", L"MIXER LINEARSATURATION", mixer_linearsaturation_command, 0);
    repo->register_channel_command(L"Mixer Commands", L"MIXER CDL",          mixer_cdl_command,          0);
    repo->register_channel_command(L"Mixer Commands", L"MIXER CDL_FILE",     mixer_cdl_file_command,     1);
    repo->register_channel_command(L"Mixer Commands", L"MIXER SPLITTONE",    mixer_splittone_command,    0);
    repo->register_channel_command(L"Mixer Commands", L"MIXER EXPOSURE", mixer_exposure_command, 0);
    repo->register_channel_command(L"Mixer Commands", L"MIXER GAMUTCOMPRESS", mixer_gamutcompress_command, 0);
    repo->register_channel_command(L"Mixer Commands", L"MIXER LUT3D",        mixer_lut3d_command,        0);
    repo->register_channel_command(L"Mixer Commands", L"MIXER HUECURVE",     mixer_huecurve_command,     0);
    repo->register_channel_command(L"Mixer Commands", L"MIXER TONEBALANCE",  mixer_tonebalance_command,  0);
    repo->register_channel_command(L"Mixer Commands", L"MIXER SHARPEN",      mixer_sharpen_command,      0);
    repo->register_channel_command(L"Mixer Commands", L"MIXER GRAIN",        mixer_grain_command,        0);
    repo->register_channel_command(L"Mixer Commands", L"MIXER QUALIFIER",    mixer_qualifier_command,    0);
    repo->register_channel_command(L"Mixer Commands", L"MIXER GRADE",        mixer_grade_command,        0);
    repo->register_channel_command(L"Mixer Commands", L"MIXER RGBLEVELS",    mixer_rgblevels_command,    0);
    repo->register_channel_command(L"Mixer Commands", L"MIXER CURVES",       mixer_curves_command,       0);
    repo->register_channel_command(L"Mixer Commands", L"MIXER VOLUME",      mixer_volume_command,       0);
    repo->register_channel_command(L"Mixer Commands", L"MIXER MASTERVOLUME", mixer_mastervolume_command, 0);
    repo->register_channel_command(L"Mixer Commands", L"MIXER GRID", mixer_grid_command, 1);
    repo->register_channel_command(L"Mixer Commands", L"MIXER COMMIT", mixer_commit_command, 0);
    repo->register_channel_command(L"Mixer Commands", L"MIXER CLEAR", mixer_clear_command, 0);
    repo->register_command(L"Mixer Commands", L"CHANNEL_GRID", channel_grid_command, 0);

    repo->register_channel_command(L"Calibration Commands", L"CALIBRATION", calibration_command, 0);
    repo->register_channel_command(L"Mixer Commands", L"OCIO_DISPLAY", ocio_display_command, 0);
    repo->register_channel_command(L"Mixer Commands", L"OCIO_LOOK", ocio_look_command, 0);
    repo->register_channel_command(L"Mixer Commands", L"AMF", amf_command, 1);

    repo->register_channel_command(L"Previz Commands", L"PREVIZ SCENE",     previz_scene_command,     0);
    repo->register_channel_command(L"Previz Commands", L"PREVIZ MAP",       previz_map_command,       2);
    repo->register_channel_command(L"Previz Commands", L"PREVIZ UNMAP",     previz_unmap_command,     1);
    repo->register_channel_command(L"Previz Commands", L"PREVIZ CAMERA",    previz_camera_command,    0);
    repo->register_channel_command(L"Previz Commands", L"PREVIZ VIEW",      previz_view_command,      0);
    repo->register_channel_command(L"Previz Commands", L"PREVIZ INFO",      previz_info_command,      0);
    repo->register_channel_command(L"Previz Commands", L"PREVIZ SHOW",      previz_show_command,      1);
    repo->register_channel_command(L"Previz Commands", L"PREVIZ GRID",      previz_grid_command,      0);
    repo->register_channel_command(L"Previz Commands", L"PREVIZ WIREFRAME", previz_wireframe_command, 0);
    repo->register_channel_command(L"Previz Commands", L"PREVIZ GIZMO",     previz_gizmo_command,     0);
    repo->register_channel_command(L"Previz Commands", L"PREVIZ PRESET",    previz_preset_command,    1);
    repo->register_channel_command(L"Previz Commands", L"PREVIZ SCREEN",    previz_screen_command,    1);
    repo->register_channel_command(L"Previz Commands", L"PREVIZ AUTOPROJECTION", previz_autoprojection_command, 0);

    repo->register_command(L"Thumbnail Commands", L"THUMBNAIL LIST", thumbnail_list_command, 0);
    repo->register_command(L"Thumbnail Commands", L"THUMBNAIL RETRIEVE", thumbnail_retrieve_command, 1);
    repo->register_command(L"Thumbnail Commands", L"THUMBNAIL GENERATE", thumbnail_generate_command, 1);
    repo->register_command(L"Thumbnail Commands", L"THUMBNAIL GENERATE_ALL", thumbnail_generateall_command, 0);

    repo->register_command(L"Query Commands", L"CINF", cinf_command, 1);
    repo->register_command(L"Query Commands", L"CLS", cls_command, 0);
    repo->register_command(L"Query Commands", L"FLS", fls_command, 0);
    repo->register_command(L"Query Commands", L"TLS", tls_command, 0);
    repo->register_command(L"Query Commands", L"VERSION", version_command, 0);
    repo->register_command(L"Query Commands", L"DIAG", diag_command, 0);
    repo->register_command(L"Query Commands", L"BYE", bye_command, 0);
    repo->register_command(L"Query Commands", L"KILL", kill_command, 0);
    repo->register_command(L"Query Commands", L"RESTART", restart_command, 0);
    repo->register_channel_command(L"Query Commands", L"INFO", info_channel_command, 0);
    repo->register_command(L"Query Commands", L"INFO", info_command, 0);
    repo->register_command(L"Query Commands", L"INFO CONFIG", info_config_command, 0);
    repo->register_command(L"Query Commands", L"INFO PATHS", info_paths_command, 0);
    repo->register_command(L"Query Commands", L"INFO OCIO", info_ocio_command, 0);
    repo->register_command(L"Query Commands", L"INFO LTC", info_ltc_command, 0);
    repo->register_command(L"Query Commands", L"INFO PORTAUDIO", info_portaudio_command, 0);
    repo->register_command(L"LTC Commands", L"LTC LOAD", ltc_load_command, 1);
    repo->register_command(L"Query Commands", L"GL INFO", gl_info_command, 0);
    repo->register_command(L"Query Commands", L"GL GC", gl_gc_command, 0);

    repo->register_command(L"Query Commands", L"OSC SUBSCRIBE", osc_subscribe_command, 1);
    repo->register_command(L"Query Commands", L"OSC UNSUBSCRIBE", osc_unsubscribe_command, 1);
}
}}} // namespace caspar::protocol::amcp
