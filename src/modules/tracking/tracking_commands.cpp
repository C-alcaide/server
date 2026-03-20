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

#include "tracking_commands.h"

#include "camera_data.h"
#include "receiver_manager.h"
#include "tracker_binding.h"
#include "tracker_registry.h"

#include <common/utf.h>
#include <protocol/amcp/amcp_command_context.h>
#include <protocol/amcp/amcp_command_repository_wrapper.h>

#include <boost/algorithm/string.hpp>

#include <sstream>
#include <stdexcept>
#include <string>

#ifndef M_PI
#define M_PI 3.141592653589793
#endif

static constexpr double DEG2RAD_CMD = M_PI / 180.0;
static constexpr double RAD2DEG_CMD = 180.0 / M_PI;

namespace caspar { namespace tracking {

using namespace protocol::amcp;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static tracking_protocol parse_protocol(const std::wstring& s)
{
    if (boost::iequals(s, L"FREED"))      return tracking_protocol::freed;
    if (boost::iequals(s, L"FREED_PLUS")) return tracking_protocol::freed_plus;
    if (boost::iequals(s, L"OSC"))        return tracking_protocol::osc;
    if (boost::iequals(s, L"VRPN"))       return tracking_protocol::vrpn;
    throw std::runtime_error("Unknown tracking protocol — use FREED, FREED_PLUS, OSC or VRPN");
}

static std::wstring protocol_name(tracking_protocol p)
{
    switch (p) {
    case tracking_protocol::freed:      return L"FREED";
    case tracking_protocol::freed_plus: return L"FREED_PLUS";
    case tracking_protocol::osc:        return L"OSC";
    case tracking_protocol::vrpn:       return L"VRPN";
    default:                            return L"UNKNOWN";
    }
}

static std::wstring mode_name(tracking_mode m)
{
    return m == tracking_mode::mode_360 ? L"360" : L"2D";
}

// Parse optional named keyword parameters from the rest of the params list.
// Returns the value if found, otherwise default.
static std::wstring kwparam(const std::vector<std::wstring>& params,
                             const std::wstring& key,
                             const std::wstring& fallback = L"")
{
    for (size_t i = 0; i + 1 < params.size(); ++i) {
        if (boost::iequals(params[i], key))
            return params[i + 1];
    }
    return fallback;
}

static bool kwflag(const std::vector<std::wstring>& params, const std::wstring& key)
{
    for (auto& p : params)
        if (boost::iequals(p, key))
            return true;
    return false;
}

// ---------------------------------------------------------------------------
// TRACKING BIND — create / replace a binding on a channel/layer
//
// Syntax:
//   TRACKING <ch>-<layer> BIND <FREED|FREED_PLUS|OSC|VRPN>
//              [PORT <port>]
//              [HOST <host>]     (VRPN only)
//              [CAMERA <id>]     (default 0)
//              [MODE <2D|360>]   (default 360)
// ---------------------------------------------------------------------------
static std::wstring tracking_bind_command(command_context& ctx)
{
    if (ctx.parameters.empty())
        return L"400 TRACKING ERROR Missing protocol argument\r\n";

    try {
        tracking_protocol proto = parse_protocol(ctx.parameters.at(0));

        int         port      = 6301;
        std::string host;
        int         camera_id = 0;
        tracking_mode mode    = tracking_mode::mode_360;

        std::wstring port_str = kwparam(ctx.parameters, L"PORT");
        if (!port_str.empty())
            port = std::stoi(port_str);

        std::wstring host_str = kwparam(ctx.parameters, L"HOST");
        if (!host_str.empty())
            host = caspar::u8(host_str);

        std::wstring cam_str = kwparam(ctx.parameters, L"CAMERA");
        if (!cam_str.empty())
            camera_id = std::stoi(cam_str);

        std::wstring mode_str = kwparam(ctx.parameters, L"MODE", L"360");
        if (boost::iequals(mode_str, L"2D"))
            mode = tracking_mode::mode_2d;

        // Grab stage from the current channel context
        auto stage_ptr = ctx.channel.stage;
        if (!stage_ptr)
            return L"403 TRACKING ERROR Invalid channel\r\n";

        // Release old receiver reference if there was a previous binding
        int ch    = ctx.channel_index;
        int layer = ctx.layer_index();

        if (tracker_registry::instance().has_binding(ch, layer)) {
            auto old_b = tracker_registry::instance().get_binding(ch, layer);
            if (old_b)
                receiver_manager::instance().release_receiver(
                    old_b->receiver.protocol, old_b->receiver.port, old_b->receiver.host);
        }

        // Ensure receiver is running
        receiver_manager::instance().ensure_receiver(proto, port, host);

        // Create binding
        tracker_binding b;
        b.camera_id      = camera_id;
        b.stage          = stage_ptr;
        b.layer_index    = layer;
        b.mode           = mode;
        b.receiver.protocol = proto;
        b.receiver.port     = port;
        b.receiver.host     = host;

        tracker_registry::instance().bind(ch, layer, std::move(b));

        return L"202 TRACKING OK\r\n";
    } catch (const std::exception& e) {
        return L"400 TRACKING ERROR " + caspar::u16(e.what()) + L"\r\n";
    }
}

// ---------------------------------------------------------------------------
// TRACKING UNBIND — remove a binding
// ---------------------------------------------------------------------------
static std::wstring tracking_unbind_command(command_context& ctx)
{
    int ch    = ctx.channel_index;
    int layer = ctx.layer_index();

    auto b = tracker_registry::instance().get_binding(ch, layer);
    if (b)
        receiver_manager::instance().release_receiver(b->receiver.protocol, b->receiver.port, b->receiver.host);

    tracker_registry::instance().unbind(ch, layer);
    return L"202 TRACKING OK\r\n";
}

// ---------------------------------------------------------------------------
// TRACKING OFFSET — set angular offsets in degrees
//
// Syntax: TRACKING <ch>-<layer> OFFSET <pan_deg> <tilt_deg> <roll_deg>
// ---------------------------------------------------------------------------
static std::wstring tracking_offset_command(command_context& ctx)
{
    if (ctx.parameters.size() < 3)
        return L"400 TRACKING ERROR Expected: OFFSET <pan_deg> <tilt_deg> <roll_deg>\r\n";

    try {
        double pan  = std::stod(ctx.parameters.at(0)) * DEG2RAD_CMD;
        double tilt = std::stod(ctx.parameters.at(1)) * DEG2RAD_CMD;
        double roll = std::stod(ctx.parameters.at(2)) * DEG2RAD_CMD;

        tracker_registry::instance().update_offset(ctx.channel_index, ctx.layer_index(), pan, tilt, roll);
        return L"202 TRACKING OK\r\n";
    } catch (const std::exception& e) {
        return L"400 TRACKING ERROR " + caspar::u16(e.what()) + L"\r\n";
    }
}

// ---------------------------------------------------------------------------
// TRACKING SCALE — set per-axis scale factors + zoom calibration
//
// Syntax: TRACKING <ch>-<layer> SCALE <pan_scale> <tilt_scale> <zoom_full_range>
//         Use negative pan_scale / tilt_scale to flip an axis (counter-tracking).
//         zoom_full_range: raw value that represents the widest angle (default 65535).
// ---------------------------------------------------------------------------
static std::wstring tracking_scale_command(command_context& ctx)
{
    if (ctx.parameters.size() < 3)
        return L"400 TRACKING ERROR Expected: SCALE <pan_scale> <tilt_scale> <zoom_full_range>\r\n";

    try {
        double pan_scale       = std::stod(ctx.parameters.at(0));
        double tilt_scale      = std::stod(ctx.parameters.at(1));
        double zoom_full_range = std::stod(ctx.parameters.at(2));

        tracker_registry::instance().update_scale(
            ctx.channel_index, ctx.layer_index(), pan_scale, tilt_scale, zoom_full_range);
        return L"202 TRACKING OK\r\n";
    } catch (const std::exception& e) {
        return L"400 TRACKING ERROR " + caspar::u16(e.what()) + L"\r\n";
    }
}

// ---------------------------------------------------------------------------
// TRACKING INFO — report binding config + latest received data
// ---------------------------------------------------------------------------
static std::wstring tracking_info_command(command_context& ctx)
{
    int ch    = ctx.channel_index;
    int layer = ctx.layer_index();

    auto binding = tracker_registry::instance().get_binding(ch, layer);
    if (!binding)
        return L"404 TRACKING ERROR No binding on this channel/layer\r\n";

    auto latest = tracker_registry::instance().get_latest_data(binding->camera_id);

    std::wostringstream ss;
    ss << L"201 TRACKING OK\r\n";
    ss << L"PROTOCOL "     << protocol_name(binding->receiver.protocol)               << L"\r\n";
    ss << L"PORT "         << binding->receiver.port                                  << L"\r\n";
    ss << L"CAMERA "       << binding->camera_id                                      << L"\r\n";
    ss << L"MODE "         << mode_name(binding->mode)                                << L"\r\n";
    ss << L"PAN_SCALE "    << binding->pan_scale                                      << L"\r\n";
    ss << L"TILT_SCALE "   << binding->tilt_scale                                     << L"\r\n";
    ss << L"PAN_OFFSET "   << (binding->pan_offset  * RAD2DEG_CMD)                    << L"\r\n";
    ss << L"TILT_OFFSET "  << (binding->tilt_offset * RAD2DEG_CMD)                    << L"\r\n";
    ss << L"ROLL_OFFSET "  << (binding->roll_offset * RAD2DEG_CMD)                    << L"\r\n";
    ss << L"ZOOM_FULL_RANGE " << binding->zoom_full_range                             << L"\r\n";
    ss << L"ZOOM_DEFAULT_FOV " << (binding->zoom_default_fov * RAD2DEG_CMD)           << L"\r\n";
    ss << L"POSITION_SCALE "  << binding->position_scale                              << L"\r\n";

    if (latest) {
        ss << L"LAST_PAN "   << (latest->pan  * RAD2DEG_CMD) << L"\r\n";
        ss << L"LAST_TILT "  << (latest->tilt * RAD2DEG_CMD) << L"\r\n";
        ss << L"LAST_ROLL "  << (latest->roll * RAD2DEG_CMD) << L"\r\n";
        ss << L"LAST_X "     << latest->x                    << L"\r\n";
        ss << L"LAST_Y "     << latest->y                    << L"\r\n";
        ss << L"LAST_Z "     << latest->z                    << L"\r\n";
        ss << L"LAST_ZOOM "  << latest->zoom                 << L"\r\n";
        ss << L"LAST_FOCUS " << latest->focus                << L"\r\n";
    } else {
        ss << L"LAST_DATA NONE\r\n";
    }

    return ss.str();
}

// ---------------------------------------------------------------------------
// TRACKING LIST — list all active bindings
// ---------------------------------------------------------------------------
static std::wstring tracking_list_command(command_context& /*ctx*/)
{
    auto all = tracker_registry::instance().get_all_bindings();

    std::wostringstream ss;
    ss << L"200 TRACKING OK\r\n";
    for (auto& [key, b] : all) {
        ss << key.first << L"-" << key.second
           << L" PROTOCOL " << protocol_name(b.receiver.protocol)
           << L" PORT "     << b.receiver.port
           << L" CAMERA "   << b.camera_id
           << L" MODE "     << mode_name(b.mode)
           << L"\r\n";
    }
    ss << L"\r\n";
    return ss.str();
}

// ---------------------------------------------------------------------------
// TRACKING ZERO — capture current physical position as the new home
//
// Reads the latest received data for the bound camera and sets the pan/tilt/roll
// offsets so that the tracker's current pose yields (0, 0, 0) injected.
// i.e. "wherever the camera is pointing right now, that becomes the neutral view".
//
// For 360: the equirectangular frame will show whatever the camera currently
//   faces as the centre-of-frame (yaw=0, pitch=0, roll=0).
// For 2D:  the layer will return to its unshifted position (fill_translation=0,0).
//
// Syntax: TRACKING <ch>-<layer> ZERO
// ---------------------------------------------------------------------------
static std::wstring tracking_zero_command(command_context& ctx)
{
    int ch    = ctx.channel_index;
    int layer = ctx.layer_index();

    auto binding = tracker_registry::instance().get_binding(ch, layer);
    if (!binding)
        return L"404 TRACKING ERROR No binding on this channel/layer\r\n";

    auto latest = tracker_registry::instance().get_latest_data(binding->camera_id);
    if (!latest)
        return L"404 TRACKING ERROR No data received yet for camera "
               + std::to_wstring(binding->camera_id) + L"\r\n";

    // new_offset = -(decoded_angle)  so that next packet at the same position → 0
    // decoded = data.pan * pan_scale + pan_offset  →  set pan_offset = -(data.pan * pan_scale)
    double new_pan_offset  = -(latest->pan  * binding->pan_scale);
    double new_tilt_offset = -(latest->tilt * binding->tilt_scale);
    double new_roll_offset = -(latest->roll);

    tracker_registry::instance().update_offset(ch, layer, new_pan_offset, new_tilt_offset, new_roll_offset);
    return L"202 TRACKING OK\r\n";
}

// ---------------------------------------------------------------------------
// TRACKING DEFAULT_FOV — set the wide-end FOV used by the zoom lens formula
//
// Equivalent to setting zoom_default_fov in the binding. Live — no rebind needed.
// This is the FOV (degrees) the lens produces when zoom_raw == zoom_full_range.
//
// Syntax: TRACKING <ch>-<layer> DEFAULT_FOV <degrees>
// ---------------------------------------------------------------------------
static std::wstring tracking_default_fov_command(command_context& ctx)
{
    if (ctx.parameters.empty())
        return L"400 TRACKING ERROR Expected: DEFAULT_FOV <degrees>\r\n";

    try {
        double fov_deg = std::stod(ctx.parameters.at(0));
        if (fov_deg <= 0.0 || fov_deg >= 180.0)
            return L"400 TRACKING ERROR DEFAULT_FOV must be between 0 and 180 degrees\r\n";

        tracker_registry::instance().update_zoom_default_fov(
            ctx.channel_index, ctx.layer_index(), fov_deg * DEG2RAD_CMD);
        return L"202 TRACKING OK\r\n";
    } catch (const std::exception& e) {
        return L"400 TRACKING ERROR " + caspar::u16(e.what()) + L"\r\n";
    }
}

// ---------------------------------------------------------------------------
// TRACKING POSITION_SCALE — set the NDC-per-mm scale for X/Y camera position
//
// Controls how physical camera translation (in millimetres, from the tracking
// protocol) is mapped to CasparCG NDC units:
//   360 mode: X → projection.offset_x,  Y → projection.offset_y (lens-shift)
//   2D mode:  X/Y added to fill_translation as parallax offset
//
// Default: 0.001  (1 metre = 1.0 NDC unit)
// Use 0 to disable position translation while keeping pan/tilt/roll active.
//
// Syntax: TRACKING <ch>-<layer> POSITION_SCALE <scale>
// ---------------------------------------------------------------------------
static std::wstring tracking_position_scale_command(command_context& ctx)
{
    if (ctx.parameters.empty())
        return L"400 TRACKING ERROR Expected: POSITION_SCALE <scale>\r\n";

    try {
        double scale = std::stod(ctx.parameters.at(0));
        tracker_registry::instance().update_position_scale(
            ctx.channel_index, ctx.layer_index(), scale);
        return L"202 TRACKING OK\r\n";
    } catch (const std::exception& e) {
        return L"400 TRACKING ERROR " + caspar::u16(e.what()) + L"\r\n";
    }
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

void register_amcp_commands(
    const std::shared_ptr<protocol::amcp::amcp_command_repository_wrapper>& repo)
{
    // Channel commands (require ch-layer syntax)
    repo->register_channel_command(L"Tracking Commands", L"TRACKING BIND",   tracking_bind_command,   1);
    repo->register_channel_command(L"Tracking Commands", L"TRACKING UNBIND", tracking_unbind_command, 0);
    repo->register_channel_command(L"Tracking Commands", L"TRACKING OFFSET",      tracking_offset_command,      3);
    repo->register_channel_command(L"Tracking Commands", L"TRACKING SCALE",       tracking_scale_command,       3);
    repo->register_channel_command(L"Tracking Commands", L"TRACKING ZERO",        tracking_zero_command,        0);
    repo->register_channel_command(L"Tracking Commands", L"TRACKING DEFAULT_FOV", tracking_default_fov_command, 1);
    repo->register_channel_command(L"Tracking Commands", L"TRACKING POSITION_SCALE", tracking_position_scale_command, 1);
    repo->register_channel_command(L"Tracking Commands", L"TRACKING INFO",        tracking_info_command,        0);

    // Global command (no channel required)
    repo->register_command(L"Tracking Commands", L"TRACKING LIST", tracking_list_command, 0);
}

}} // namespace caspar::tracking
