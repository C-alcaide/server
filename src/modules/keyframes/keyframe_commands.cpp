/*
 * Copyright (c) 2026 CasparCG Contributors
 *
 * This file is part of CasparCG (www.casparcg.com).
 *
 * CasparCG is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#include "keyframe_commands.h"

#include "keyframe_data.h"
#include "keyframe_fields.h"
#include "keyframe_json.h"

#include <common/log.h>
#include <common/utf.h>
#include <protocol/amcp/amcp_command_context.h>
#include <protocol/amcp/amcp_command_repository_wrapper.h>

#include <sstream>
#include <stdexcept>
#include <string>

namespace caspar { namespace keyframes {

using namespace protocol::amcp;

// ---------------------------------------------------------------------------
// Helper: reconstruct JSON blob from AMCP parameters, stripping outer parens
// ---------------------------------------------------------------------------

static std::string extract_json(const std::vector<std::wstring>& params, size_t start_idx = 0)
{
    std::string json;
    for (size_t i = start_idx; i < params.size(); ++i)
        json += caspar::u8(params[i]) + " ";
    if (!json.empty())
        json.pop_back(); // trim trailing space

    // Strip the outer parentheses the client adds for AMCP tokeniser safety
    if (json.size() >= 2 && json.front() == '(' && json.back() == ')')
        json = json.substr(1, json.size() - 2);

    return json;
}

// ---------------------------------------------------------------------------
// Helper: parse time_secs with validation
// ---------------------------------------------------------------------------

static bool parse_time(const std::wstring& param, double& out)
{
    try {
        std::string s = caspar::u8(param);
        size_t pos = 0;
        out = std::stod(s, &pos);
        if (pos != s.size())
            return false; // trailing garbage
        if (!std::isfinite(out) || out < 0.0)
            return false;
        return true;
    } catch (...) {
        return false;
    }
}

// ---------------------------------------------------------------------------
// KEYFRAMES SET <json_blob>
//
// Uploads the keyframe timeline for the selected channel/layer.
// After SET, the binding is NOT armed.  Call ARM to activate.
// ---------------------------------------------------------------------------
static std::wstring keyframes_set_command(command_context& ctx)
{
    if (ctx.parameters.empty())
        return L"400 KEYFRAMES ERROR Missing JSON argument\r\n";

    std::string json = extract_json(ctx.parameters);

    try {
        auto tl     = json_to_timeline(json);
        auto tl_ptr = std::make_shared<keyframe_timeline>(std::move(tl));

        ctx.channel.stage->set_keyframe_data(
            ctx.layer_index(), std::static_pointer_cast<void>(tl_ptr)).get();

        CASPAR_LOG(info) << L"[keyframes] SET ch=" << ctx.channel_index
                         << L" lay=" << ctx.layer_index();
        return L"202 KEYFRAMES OK\r\n";
    } catch (const std::exception& e) {
        return L"400 KEYFRAMES ERROR " + caspar::u16(e.what()) + L"\r\n";
    }
}

// ---------------------------------------------------------------------------
// KEYFRAMES ARM
// ---------------------------------------------------------------------------
static std::wstring keyframes_arm_command(command_context& ctx)
{
    bool armed = ctx.channel.stage->arm_keyframes(ctx.layer_index()).get();
    if (!armed)
        return L"404 KEYFRAMES ERROR No timeline set for this layer \u2014 call SET first\r\n";

    CASPAR_LOG(info) << L"[keyframes] ARMED ch=" << ctx.channel_index
                     << L" lay=" << ctx.layer_index();
    return L"202 KEYFRAMES OK\r\n";
}

// ---------------------------------------------------------------------------
// KEYFRAMES DISARM
// ---------------------------------------------------------------------------
static std::wstring keyframes_disarm_command(command_context& ctx)
{
    ctx.channel.stage->disarm_keyframes(ctx.layer_index()).get();
    CASPAR_LOG(info) << L"[keyframes] DISARMED ch=" << ctx.channel_index
                     << L" lay=" << ctx.layer_index();
    return L"202 KEYFRAMES OK\r\n";
}

// ---------------------------------------------------------------------------
// KEYFRAMES CLEAR
// ---------------------------------------------------------------------------
static std::wstring keyframes_clear_command(command_context& ctx)
{
    ctx.channel.stage->clear_keyframes(ctx.layer_index()).get();
    CASPAR_LOG(info) << L"[keyframes] CLEAR ch=" << ctx.channel_index
                     << L" lay=" << ctx.layer_index();
    return L"202 KEYFRAMES OK\r\n";
}

// ---------------------------------------------------------------------------
// KEYFRAMES GET
// ---------------------------------------------------------------------------
static std::wstring keyframes_get_command(command_context& ctx)
{
    auto data = ctx.channel.stage->get_keyframe_data(ctx.layer_index()).get();
    if (!data)
        return L"404 KEYFRAMES ERROR No timeline set for this layer\r\n";

    try {
        auto tl = std::static_pointer_cast<keyframe_timeline>(data);
        std::string json = timeline_to_json(*tl);
        return L"201 KEYFRAMES OK\r\n" + caspar::u16(json) + L"\r\n";
    } catch (const std::exception& e) {
        return L"500 KEYFRAMES ERROR " + caspar::u16(e.what()) + L"\r\n";
    }
}

// ---------------------------------------------------------------------------
// KEYFRAMES PATCH <time_secs> <json_blob>
//
// Updates only the fields present in the JSON for the keyframe nearest to
// time_secs.  Atomic: runs entirely on the stage executor (no TOCTOU race).
// ---------------------------------------------------------------------------
static std::wstring keyframes_patch_command(command_context& ctx)
{
    if (ctx.parameters.size() < 2)
        return L"400 KEYFRAMES ERROR PATCH requires time_secs and JSON arguments\r\n";

    double time_secs = 0.0;
    if (!parse_time(ctx.parameters.at(0), time_secs))
        return L"400 KEYFRAMES ERROR PATCH: invalid time_secs\r\n";

    std::string json = extract_json(ctx.parameters, 1);

    try {
        auto vals     = parse_kf_values(json);
        auto vals_ptr = std::make_shared<kf_values>(std::move(vals));

        bool ok = ctx.channel.stage->patch_keyframe(
            ctx.layer_index(), time_secs, std::static_pointer_cast<void>(vals_ptr)).get();

        if (!ok)
            return L"404 KEYFRAMES ERROR KF not found at given time\r\n";

        return L"202 KEYFRAMES OK\r\n";
    } catch (const std::exception& e) {
        return L"400 KEYFRAMES ERROR " + caspar::u16(e.what()) + L"\r\n";
    }
}

// ---------------------------------------------------------------------------
// KEYFRAMES SEEK <time_secs>
//
// Sets the media time override for keyframe evaluation without requiring
// video seek.  Useful for scrubbing the keyframe timeline while paused.
// ---------------------------------------------------------------------------
static std::wstring keyframes_seek_command(command_context& ctx)
{
    if (ctx.parameters.empty())
        return L"400 KEYFRAMES ERROR Missing time_secs argument\r\n";

    double time_secs = 0.0;
    if (!parse_time(ctx.parameters.at(0), time_secs))
        return L"400 KEYFRAMES ERROR Invalid time_secs\r\n";

    ctx.channel.stage->set_media_time_override(ctx.layer_index(), time_secs).get();
    return L"202 KEYFRAMES OK\r\n";
}

// ---------------------------------------------------------------------------
// KEYFRAMES STATUS
//
// Returns JSON with the armed/disarmed state and keyframe count for the
// selected channel/layer.  Always succeeds (empty timeline = count 0).
// ---------------------------------------------------------------------------
static std::wstring keyframes_status_command(command_context& ctx)
{
    auto data = ctx.channel.stage->get_keyframe_status(ctx.layer_index()).get();
    auto status = std::static_pointer_cast<kf_status>(data);

    std::ostringstream oss;
    oss << "{\"armed\":" << (status->armed ? "true" : "false")
        << ",\"keyframe_count\":" << status->keyframe_count << "}";

    return L"201 KEYFRAMES OK\r\n" + caspar::u16(oss.str()) + L"\r\n";
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

void register_amcp_commands(
    const std::shared_ptr<protocol::amcp::amcp_command_repository_wrapper>& repo)
{
    repo->register_channel_command(L"Keyframe Commands", L"KEYFRAMES SET",    keyframes_set_command,    1);
    repo->register_channel_command(L"Keyframe Commands", L"KEYFRAMES ARM",    keyframes_arm_command,    0);
    repo->register_channel_command(L"Keyframe Commands", L"KEYFRAMES DISARM", keyframes_disarm_command, 0);
    repo->register_channel_command(L"Keyframe Commands", L"KEYFRAMES CLEAR",  keyframes_clear_command,  0);
    repo->register_channel_command(L"Keyframe Commands", L"KEYFRAMES GET",    keyframes_get_command,    0);
    repo->register_channel_command(L"Keyframe Commands", L"KEYFRAMES PATCH",  keyframes_patch_command,  2);
    repo->register_channel_command(L"Keyframe Commands", L"KEYFRAMES SEEK",   keyframes_seek_command,   1);
    repo->register_channel_command(L"Keyframe Commands", L"KEYFRAMES STATUS", keyframes_status_command, 0);
}

}} // namespace caspar::keyframes
