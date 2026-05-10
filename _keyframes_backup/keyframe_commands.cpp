/*
 * Copyright (c) 2026 CasparCG (www.casparcg.com).
 *
 * CasparCG is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#include "keyframe_commands.h"

#include "keyframe_data.h"
#include "keyframe_json.h"
#include "keyframe_registry.h"

#include <common/log.h>
#include <common/utf.h>
#include <protocol/amcp/amcp_command_context.h>
#include <protocol/amcp/amcp_command_repository_wrapper.h>

#include <boost/algorithm/string.hpp>

#include <sstream>
#include <stdexcept>
#include <string>

namespace caspar { namespace keyframes {

using namespace protocol::amcp;

// ---------------------------------------------------------------------------
// KEYFRAMES SET <json_blob>
//
// Uploads the keyframe timeline for the selected channel/layer.
// The JSON blob must be the entire "keyframes" JSON object produced by
// timeline_to_json().
//
// After SET, the binding exists but is NOT yet armed.  Call ARM to activate.
// ---------------------------------------------------------------------------
static std::wstring keyframes_set_command(command_context& ctx)
{
    if (ctx.parameters.empty())
        return L"400 KEYFRAMES ERROR Missing JSON argument\r\n";

    // Reconstruct the JSON from (possibly space-split) parameters
    std::string json;
    for (const auto& p : ctx.parameters)
        json += caspar::u8(p) + " ";
    if (!json.empty()) json.pop_back(); // trim trailing space

    // Strip the outer parentheses the client adds so the AMCP tokenizer
    // preserves the JSON blob verbatim: ({"keyframes":[...]}) → {"keyframes":[...]}
    if (json.size() >= 2 && json.front() == '(' && json.back() == ')')
        json = json.substr(1, json.size() - 2);

    try {
        auto tl = json_to_timeline(json);

        auto stage_ptr = ctx.channel.stage;
        if (!stage_ptr)
            return L"403 KEYFRAMES ERROR Invalid channel\r\n";

        keyframe_registry::instance().set_timeline(
            ctx.channel_index, ctx.layer_index(), std::move(tl),
            std::static_pointer_cast<core::stage_base>(stage_ptr));

        CASPAR_LOG(info) << L"[keyframes] SET ch=" << ctx.channel_index
                         << L" lay=" << ctx.layer_index();
        return L"202 KEYFRAMES OK\r\n";
    } catch (const std::exception& e) {
        return L"400 KEYFRAMES ERROR " + caspar::u16(e.what()) + L"\r\n";
    }
}

// ---------------------------------------------------------------------------
// KEYFRAMES ARM
//
// Enables keyframe injection.  Subsequent TICK calls will call
// stage->apply_transform().
// ---------------------------------------------------------------------------
static std::wstring keyframes_arm_command(command_context& ctx)
{
    if (!keyframe_registry::instance().has_binding(ctx.channel_index, ctx.layer_index()))
        return L"404 KEYFRAMES ERROR No timeline set for this layer — call SET first\r\n";

    keyframe_registry::instance().arm(ctx.channel_index, ctx.layer_index());
    CASPAR_LOG(info) << L"[keyframes] ARMED ch=" << ctx.channel_index
                     << L" lay=" << ctx.layer_index();
    return L"202 KEYFRAMES OK\r\n";
}

// ---------------------------------------------------------------------------
// KEYFRAMES DISARM
//
// Stops keyframe injection.  Static MIXER commands take over again.
// The timeline is preserved so ARM can re-enable without a new SET.
// ---------------------------------------------------------------------------
static std::wstring keyframes_disarm_command(command_context& ctx)
{
    keyframe_registry::instance().disarm(ctx.channel_index, ctx.layer_index());
    CASPAR_LOG(info) << L"[keyframes] DISARMED ch=" << ctx.channel_index
                     << L" lay=" << ctx.layer_index();
    return L"202 KEYFRAMES OK\r\n";
}

// ---------------------------------------------------------------------------
// KEYFRAMES CLEAR
//
// Removes the timeline and disarms.  The layer returns to full manual control.
// ---------------------------------------------------------------------------
static std::wstring keyframes_clear_command(command_context& ctx)
{
    keyframe_registry::instance().clear(ctx.channel_index, ctx.layer_index());
    CASPAR_LOG(info) << L"[keyframes] CLEAR ch=" << ctx.channel_index
                     << L" lay=" << ctx.layer_index();
    return L"202 KEYFRAMES OK\r\n";
}

// ---------------------------------------------------------------------------
// KEYFRAMES GET
//
// Returns the stored timeline as a JSON string.
// Response: 201 KEYFRAMES OK\r\n<json>\r\n
// ---------------------------------------------------------------------------
static std::wstring keyframes_get_command(command_context& ctx)
{
    auto tl = keyframe_registry::instance().get_timeline(
        ctx.channel_index, ctx.layer_index());

    if (!tl)
        return L"404 KEYFRAMES ERROR No timeline set for this layer\r\n";

    try {
        std::string json = timeline_to_json(*tl);
        return L"201 KEYFRAMES OK\r\n" + caspar::u16(json) + L"\r\n";
    } catch (const std::exception& e) {
        return L"500 KEYFRAMES ERROR " + caspar::u16(e.what()) + L"\r\n";
    }
}

// ---------------------------------------------------------------------------
// KEYFRAMES TICK <time_secs> [SEEK]
//
// Called by the client on every OSC file/time update.
// - time_secs : current file position in fractional seconds
// - SEEK      : optional flag — uses duration=0 (hard snap, for seek events)
//
// The TICK command has virtually no server-side overhead: it reads the
// pre-computed timeline, interpolates one kf_state, and calls apply_transform
// with duration=1 or 0.  The response is a single-line 202 OK.
// ---------------------------------------------------------------------------
static std::wstring keyframes_tick_command(command_context& ctx)
{
    if (ctx.parameters.empty())
        return L"400 KEYFRAMES ERROR Missing time_secs argument\r\n";

    double time_secs = 0.0;
    try {
        time_secs = std::stod(caspar::u8(ctx.parameters.at(0)));
    } catch (...) {
        return L"400 KEYFRAMES ERROR Invalid time_secs — expected a floating-point number\r\n";
    }

    bool is_seek = false;
    for (size_t i = 1; i < ctx.parameters.size(); ++i) {
        if (boost::iequals(ctx.parameters[i], L"SEEK"))
            is_seek = true;
    }

    unsigned int duration = is_seek ? 0u : 1u;
    keyframe_registry::instance().tick(
        ctx.channel_index, ctx.layer_index(), time_secs, duration);

    return L"202 KEYFRAMES OK\r\n";
}

// ---------------------------------------------------------------------------
// KEYFRAMES PATCH <time_secs> <json_blob>
//
// Updates only the fields present in the JSON for the keyframe nearest to
// time_secs (within 1 ms). Fields absent from the JSON keep their current
// values.  This avoids uploading the full ~50-field state on every slider move.
// ---------------------------------------------------------------------------
static std::wstring keyframes_patch_command(command_context& ctx)
{
    if (ctx.parameters.size() < 2)
        return L"400 KEYFRAMES ERROR PATCH requires time_secs and JSON arguments\r\n";

    double time_secs = 0.0;
    try {
        time_secs = std::stod(caspar::u8(ctx.parameters.at(0)));
    } catch (...) {
        return L"400 KEYFRAMES ERROR PATCH: invalid time_secs\r\n";
    }

    // Reconstruct JSON blob from remaining parameters
    std::string json;
    for (size_t i = 1; i < ctx.parameters.size(); ++i)
        json += caspar::u8(ctx.parameters[i]) + " ";
    if (!json.empty()) json.pop_back();

    // Strip outer parens added by the client for AMCP tokeniser safety
    if (json.size() >= 2 && json.front() == '(' && json.back() == ')')
        json = json.substr(1, json.size() - 2);

    auto binding_opt = keyframe_registry::instance().get_binding(
        ctx.channel_index, ctx.layer_index());
    if (!binding_opt)
        return L"404 KEYFRAMES ERROR No timeline set for this layer\r\n";

    // Find the base state of the KF at that time, then apply the partial patch
    try {
        kf_state base_state;
        bool found = false;
        for (const auto& kf : binding_opt->timeline.keyframes()) {
            if (std::abs(kf.time_secs - time_secs) < 0.001) {
                base_state = kf.state;
                found = true;
                break;
            }
        }
        if (!found)
            return L"404 KEYFRAMES ERROR KF not found at given time\r\n";

        kf_state patched = patch_state_from_json(base_state, json);
        bool ok = keyframe_registry::instance().patch_timeline(
            ctx.channel_index, ctx.layer_index(), time_secs, patched);
        if (!ok)
            return L"404 KEYFRAMES ERROR KF not found at given time\r\n";

        return L"202 KEYFRAMES OK\r\n";
    } catch (const std::exception& e) {
        return L"400 KEYFRAMES ERROR " + caspar::u16(e.what()) + L"\r\n";
    }
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
    repo->register_channel_command(L"Keyframe Commands", L"KEYFRAMES TICK",   keyframes_tick_command,   1);
    repo->register_channel_command(L"Keyframe Commands", L"KEYFRAMES PATCH",  keyframes_patch_command,  2);
}

}} // namespace caspar::keyframes
