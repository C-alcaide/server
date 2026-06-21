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
#include "lens_profile.h"
#include "receiver_manager.h"
#include "tracker_binding.h"
#include "tracker_registry.h"

#include "../ltc/ltc_input.h"

#include <common/env.h>
#include <common/utf.h>
#include <protocol/amcp/amcp_command_context.h>
#include <protocol/amcp/amcp_command_repository_wrapper.h>

#include <core/mixer/mixer.h>
#include <core/producer/stage.h>
#include <core/video_channel.h>
#include <core/video_format.h>
#include <accelerator/ogl/image/image_mixer.h>
#include <accelerator/ogl/image/previz_renderer.h>

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>

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
    if (boost::iequals(s, L"PSN"))        return tracking_protocol::psn;
    if (boost::iequals(s, L"OPENTRACKIO")) return tracking_protocol::opentrackio;
    throw std::runtime_error("Unknown tracking protocol \u2014 use FREED, FREED_PLUS, OSC, VRPN, PSN or OPENTRACKIO");
}

static std::wstring protocol_name(tracking_protocol p)
{
    switch (p) {
    case tracking_protocol::freed:      return L"FREED";
    case tracking_protocol::freed_plus: return L"FREED_PLUS";
    case tracking_protocol::osc:        return L"OSC";
    case tracking_protocol::vrpn:       return L"VRPN";
    case tracking_protocol::psn:        return L"PSN";
    case tracking_protocol::opentrackio: return L"OPENTRACKIO";
    default:                            return L"UNKNOWN";
    }
}

static std::wstring mode_name(tracking_mode m)
{
    switch (m) {
    case tracking_mode::mode_360:    return L"360";
    case tracking_mode::mode_2d:     return L"2D";
    case tracking_mode::mode_previz: return L"PREVIZ";
    case tracking_mode::mode_target: return L"TARGET";
    default:                         return L"UNKNOWN";
    }
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
//   TRACKING <ch>-<layer> BIND <FREED|FREED_PLUS|OSC|VRPN|PSN|OPENTRACKIO>
//              [PORT <port>]
//              [HOST <host>]     (VRPN: server URL; PSN/OPENTRACKIO: multicast group)
//              [CAMERA <id>]     (default 0)
//              [MODE <2D|360|PREVIZ|TARGET>]   (default 360)
//                  TARGET: project a tracked SUBJECT position through a static
//                  virtual camera (TARGET_CAMERA/TARGET_MAP) so a graphic follows it.
static std::wstring tracking_bind_command(command_context& ctx)
{
    if (ctx.parameters.empty())
        return L"400 TRACKING ERROR Missing protocol argument\r\n";

    try {
        tracking_protocol proto = parse_protocol(ctx.parameters.at(0));

        int         port      = (proto == tracking_protocol::psn)         ? 56565
                                : (proto == tracking_protocol::opentrackio) ? 55555
                                                                            : 6301;
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
        else if (boost::iequals(mode_str, L"PREVIZ"))
            mode = tracking_mode::mode_previz;
        else if (boost::iequals(mode_str, L"TARGET"))
            mode = tracking_mode::mode_target;

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

        // Capture the channel frame rate for frame-native genlock latency comp.
        if (auto raw_ch = ctx.channel.raw_channel)
            b.channel_fps = raw_ch->stage()->video_format_desc().fps;

        // For PREVIZ mode, set up the callback to drive the previz camera
        if (mode == tracking_mode::mode_previz) {
            auto raw_ch = ctx.channel.raw_channel;
            if (raw_ch) {
                auto img = raw_ch->mixer().get_image_mixer();
                auto* ogl_mix = dynamic_cast<accelerator::ogl::image_mixer*>(img.get());
                if (ogl_mix) {
                    // Capture a weak reference to the stage so we can detect channel destruction.
                    auto stage_weak = b.stage;
                    auto& renderer  = ogl_mix->get_previz_renderer();
                    b.previz_camera_fn = [&renderer, stage_weak](float x, float y, float z,
                                                                  float yaw, float pitch, float roll, float fov) {
                        // Skip while the operator has frozen the camera (OVERRIDE).
                        if (stage_weak.lock() && !renderer.is_camera_locked())
                            renderer.set_camera(x, y, z, yaw, pitch, roll, fov);
                    };
                }
            }
        }

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
    ss << L"DELAY "           << binding->delay_ms                                    << L"\r\n";
    ss << L"GENLOCK "         << (binding->genlock_enable ? 1 : 0) << L" " << binding->genlock_frames << L"\r\n";
    ss << L"NODAL "           << binding->nodal_forward_m << L" " << binding->nodal_right_m
                              << L" " << binding->nodal_up_m                            << L"\r\n";
    ss << L"DOF "             << (binding->dof_enable ? 1 : 0) << L" " << binding->dof_focus_near_raw
                              << L" " << binding->dof_focus_far_raw << L" " << binding->dof_max_radius << L"\r\n";
    ss << L"LENS "            << (binding->lens ? caspar::u16(binding->lens->name()) : std::wstring(L"NONE")) << L"\r\n";
    ss << L"TARGET_CAMERA "    << (binding->target_enable ? 1 : 0) << L" "
                              << binding->target_cam_x << L" " << binding->target_cam_y << L" " << binding->target_cam_z
                              << L" " << (binding->target_cam_yaw   * RAD2DEG_CMD)
                              << L" " << (binding->target_cam_pitch * RAD2DEG_CMD)
                              << L" " << (binding->target_cam_roll  * RAD2DEG_CMD)
                              << L" " << (binding->target_cam_fov   * RAD2DEG_CMD)                  << L"\r\n";
    ss << L"TARGET_MAP "       << binding->target_gain << L" " << binding->target_ref_dist_m
                              << L" " << binding->target_aspect                                     << L"\r\n";

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

    // House timecode diagnostics (shared LTC input).
    ss << L"LTC_VALID " << (caspar::ltc::LTCInput::instance().is_valid() ? 1 : 0) << L"\r\n";
    ss << L"LTC_TC "    << caspar::u16(caspar::ltc::LTCInput::instance().get_current_timecode_string()) << L"\r\n";

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
// TRACKING DELAY — latency compensation for the tracking pose
//
// Delays the applied camera pose by the given number of milliseconds,
// interpolating between buffered samples. Use this to time-align tracking data
// with a delayed video feed (genlock/processing latency).
//   0 = disabled (newest sample injected immediately — no buffering).
//
// Live — no rebind needed. With no argument, queries the current value.
//
// Syntax: TRACKING <ch>-<layer> DELAY [milliseconds]
// ---------------------------------------------------------------------------
static std::wstring tracking_delay_command(command_context& ctx)
{
    if (ctx.parameters.empty()) {
        auto binding = tracker_registry::instance().get_binding(ctx.channel_index, ctx.layer_index());
        if (!binding)
            return L"404 TRACKING ERROR No binding on this channel/layer\r\n";
        std::wostringstream ss;
        ss << L"201 TRACKING OK\r\n" << L"DELAY " << binding->delay_ms << L"\r\n";
        return ss.str();
    }

    try {
        double delay_ms = std::stod(ctx.parameters.at(0));
        if (delay_ms < 0.0)
            return L"400 TRACKING ERROR DELAY must be >= 0 milliseconds\r\n";

        tracker_registry::instance().update_delay(
            ctx.channel_index, ctx.layer_index(), delay_ms);
        return L"202 TRACKING OK\r\n";
    } catch (const std::exception& e) {
        return L"400 TRACKING ERROR " + caspar::u16(e.what()) + L"\r\n";
    }
}

// ---------------------------------------------------------------------------
// TRACKING GENLOCK — frame-native, LTC-anchored latency compensation
//
// A broadcast-native alternative to DELAY: holds the tracking pose back by a
// number of frames of the channel's frame rate. When a valid house LTC signal
// is present (shared LTC input), the sampled time is snapped to the house frame
// grid so pose updates align to video frame boundaries (true genlock). When no
// LTC is present it behaves as a frame-native delay (frames / channel_fps).
//
// Takes precedence over DELAY when enabled. Live — no rebind needed.
// With no argument, queries. `OFF` (or 0 frames with no ON) disables.
//
// Syntax: TRACKING <ch>-<layer> GENLOCK <frames> [ON|OFF]
//         TRACKING <ch>-<layer> GENLOCK OFF
// ---------------------------------------------------------------------------
static std::wstring tracking_genlock_command(command_context& ctx)
{
    if (ctx.parameters.empty()) {
        auto binding = tracker_registry::instance().get_binding(ctx.channel_index, ctx.layer_index());
        if (!binding)
            return L"404 TRACKING ERROR No binding on this channel/layer\r\n";
        std::wostringstream ss;
        ss << L"201 TRACKING OK\r\n"
           << L"GENLOCK " << (binding->genlock_enable ? 1 : 0) << L" " << binding->genlock_frames << L"\r\n";
        return ss.str();
    }

    try {
        auto first = ctx.parameters.at(0);

        // GENLOCK OFF — disable, keep the frame count.
        if (boost::iequals(first, L"OFF")) {
            auto binding = tracker_registry::instance().get_binding(ctx.channel_index, ctx.layer_index());
            double frames = binding ? binding->genlock_frames : 0.0;
            tracker_registry::instance().update_genlock(
                ctx.channel_index, ctx.layer_index(), false, frames);
            return L"202 TRACKING OK\r\n";
        }

        double frames = std::stod(first);
        if (frames < 0.0)
            return L"400 TRACKING ERROR GENLOCK frames must be >= 0\r\n";

        bool enable = true;
        if (ctx.parameters.size() > 1) {
            if (boost::iequals(ctx.parameters.at(1), L"OFF"))
                enable = false;
            else if (boost::iequals(ctx.parameters.at(1), L"ON"))
                enable = true;
        }

        tracker_registry::instance().update_genlock(
            ctx.channel_index, ctx.layer_index(), enable, frames);
        return L"202 TRACKING OK\r\n";
    } catch (const std::exception& e) {
        return L"400 TRACKING ERROR " + caspar::u16(e.what()) + L"\r\n";
    }
}

// ---------------------------------------------------------------------------
// TRACKING NODAL — entrance-pupil (nodal) offset for parallax correction
//
// Sets the lens-local offset (metres) of the optical entrance pupil from the
// tracked origin, in the camera frame: forward (along view), right, up.
// Produces correct parallax as the camera rotates about an off-axis pupil.
// All 0 = disabled. Live — no rebind needed. With no argument, queries.
//
// Syntax: TRACKING <ch>-<layer> NODAL [forward_m right_m up_m]
// ---------------------------------------------------------------------------
static std::wstring tracking_nodal_command(command_context& ctx)
{
    if (ctx.parameters.empty()) {
        auto binding = tracker_registry::instance().get_binding(ctx.channel_index, ctx.layer_index());
        if (!binding)
            return L"404 TRACKING ERROR No binding on this channel/layer\r\n";
        std::wostringstream ss;
        ss << L"201 TRACKING OK\r\n"
           << L"NODAL " << binding->nodal_forward_m << L" " << binding->nodal_right_m << L" "
           << binding->nodal_up_m << L"\r\n";
        return ss.str();
    }

    if (ctx.parameters.size() < 3)
        return L"400 TRACKING ERROR Expected: NODAL <forward_m> <right_m> <up_m>\r\n";

    try {
        double forward_m = std::stod(ctx.parameters.at(0));
        double right_m   = std::stod(ctx.parameters.at(1));
        double up_m      = std::stod(ctx.parameters.at(2));
        tracker_registry::instance().update_nodal(
            ctx.channel_index, ctx.layer_index(), forward_m, right_m, up_m);
        return L"202 TRACKING OK\r\n";
    } catch (const std::exception& e) {
        return L"400 TRACKING ERROR " + caspar::u16(e.what()) + L"\r\n";
    }
}

// ---------------------------------------------------------------------------
// TRACKING DOF — faked depth-of-field driven by the lens focus channel
//
// NON-PHYSICAL: 2D/360 layers have no depth buffer. This maps the decoded focus
// value to a lens-bokeh blur radius for an operator-calibrated rack-focus look:
//   radius = max_radius * clamp((focus - near_raw)/(far_raw - near_raw), 0, 1)
// When enabled, tracking drives the layer blur each packet (overrides MIXER BLUR).
// Disable to leave the layer blur untouched. With no argument, queries.
//
// Syntax: TRACKING <ch>-<layer> DOF <enable 0|1> [near_raw far_raw max_radius]
// ---------------------------------------------------------------------------
static std::wstring tracking_dof_command(command_context& ctx)
{
    if (ctx.parameters.empty()) {
        auto binding = tracker_registry::instance().get_binding(ctx.channel_index, ctx.layer_index());
        if (!binding)
            return L"404 TRACKING ERROR No binding on this channel/layer\r\n";
        std::wostringstream ss;
        ss << L"201 TRACKING OK\r\n"
           << L"DOF " << (binding->dof_enable ? 1 : 0) << L" " << binding->dof_focus_near_raw << L" "
           << binding->dof_focus_far_raw << L" " << binding->dof_max_radius << L"\r\n";
        return ss.str();
    }

    try {
        bool enable = std::stoi(ctx.parameters.at(0)) != 0;
        // Preserve existing calibration unless explicitly provided.
        auto binding = tracker_registry::instance().get_binding(ctx.channel_index, ctx.layer_index());
        if (!binding)
            return L"404 TRACKING ERROR No binding on this channel/layer\r\n";
        double near_raw   = ctx.parameters.size() > 1 ? std::stod(ctx.parameters[1]) : binding->dof_focus_near_raw;
        double far_raw    = ctx.parameters.size() > 2 ? std::stod(ctx.parameters[2]) : binding->dof_focus_far_raw;
        double max_radius = ctx.parameters.size() > 3 ? std::stod(ctx.parameters[3]) : binding->dof_max_radius;
        if (max_radius < 0.0)
            return L"400 TRACKING ERROR DOF max_radius must be >= 0\r\n";

        tracker_registry::instance().update_dof(
            ctx.channel_index, ctx.layer_index(), enable, near_raw, far_raw, max_radius);
        return L"202 TRACKING OK\r\n";
    } catch (const std::exception& e) {
        return L"400 TRACKING ERROR " + caspar::u16(e.what()) + L"\r\n";
    }
}

// ---------------------------------------------------------------------------
// TRACKING LENS — load/clear a dynamic lens-calibration profile
//
// When loaded, the binding samples the profile by (zoom, focus) each packet to
// drive the projection FOV + Brown-Conrady distortion (k1..k3, p1/p2) +
// entrance-pupil forward offset. Distortion is applied in 360 mode.
// The path is resolved relative to the media folder (path-traversal guarded).
// With no argument, queries the active profile name.
//
// Syntax: TRACKING <ch>-<layer> LENS LOAD <path>
//         TRACKING <ch>-<layer> LENS CLEAR
// ---------------------------------------------------------------------------
static std::wstring tracking_lens_command(command_context& ctx)
{
    if (ctx.parameters.empty()) {
        auto binding = tracker_registry::instance().get_binding(ctx.channel_index, ctx.layer_index());
        if (!binding)
            return L"404 TRACKING ERROR No binding on this channel/layer\r\n";
        std::wostringstream ss;
        ss << L"201 TRACKING OK\r\n";
        if (binding->lens)
            ss << L"LENS " << caspar::u16(binding->lens->name()) << L"\r\n";
        else
            ss << L"LENS NONE\r\n";
        return ss.str();
    }

    auto sub = ctx.parameters.at(0);

    if (boost::iequals(sub, L"CLEAR") || boost::iequals(sub, L"NONE")) {
        try {
            tracker_registry::instance().update_lens(ctx.channel_index, ctx.layer_index(), nullptr);
            return L"202 TRACKING OK\r\n";
        } catch (const std::exception& e) {
            return L"400 TRACKING ERROR " + caspar::u16(e.what()) + L"\r\n";
        }
    }

    if (boost::iequals(sub, L"LOAD")) {
        if (ctx.parameters.size() < 2)
            return L"400 TRACKING ERROR Expected: LENS LOAD <path>\r\n";
        try {
            auto media_base = boost::filesystem::canonical(env::media_folder());
            auto resolved   = boost::filesystem::path(media_base) / ctx.parameters.at(1);
            // Guard against path traversal outside the media folder.
            auto check = resolved.lexically_normal();
            if (check.wstring().find(media_base.wstring()) != 0)
                return L"403 TRACKING FORBIDDEN\r\n";
            if (!boost::filesystem::exists(resolved))
                return L"404 TRACKING ERROR Lens file not found\r\n";

            auto profile = lens_profile::load(caspar::u8(resolved.wstring()));
            tracker_registry::instance().update_lens(ctx.channel_index, ctx.layer_index(), profile);
            return L"202 TRACKING OK\r\n";
        } catch (const std::exception& e) {
            return L"400 TRACKING ERROR " + caspar::u16(e.what()) + L"\r\n";
        }
    }

    return L"400 TRACKING ERROR Expected: LENS LOAD <path> | LENS CLEAR\r\n";
}

// ---------------------------------------------------------------------------
// TRACKING TARGET_CAMERA — static virtual camera for mode TARGET
//
// Defines the virtual camera through which a tracked SUBJECT world position is
// projected to screen space (mode TARGET only). Position in metres, orientation
// in degrees, fov is the vertical field of view in degrees. Setting a camera
// enables target projection; with no argument, queries. Live — no rebind needed.
//
// Syntax: TRACKING <ch>-<layer> TARGET_CAMERA <x> <y> <z> <yaw> <pitch> <roll> <fov>
// ---------------------------------------------------------------------------
static std::wstring tracking_target_camera_command(command_context& ctx)
{
    if (ctx.parameters.empty()) {
        auto binding = tracker_registry::instance().get_binding(ctx.channel_index, ctx.layer_index());
        if (!binding)
            return L"404 TRACKING ERROR No binding on this channel/layer\r\n";
        std::wostringstream ss;
        ss << L"201 TRACKING OK\r\n"
           << L"TARGET_CAMERA " << (binding->target_enable ? 1 : 0) << L" "
           << binding->target_cam_x << L" " << binding->target_cam_y << L" " << binding->target_cam_z
           << L" " << (binding->target_cam_yaw   * RAD2DEG_CMD)
           << L" " << (binding->target_cam_pitch * RAD2DEG_CMD)
           << L" " << (binding->target_cam_roll  * RAD2DEG_CMD)
           << L" " << (binding->target_cam_fov   * RAD2DEG_CMD) << L"\r\n";
        return ss.str();
    }

    if (ctx.parameters.size() < 7)
        return L"400 TRACKING ERROR Expected: TARGET_CAMERA <x> <y> <z> <yaw> <pitch> <roll> <fov>\r\n";

    try {
        double x     = std::stod(ctx.parameters.at(0));
        double y     = std::stod(ctx.parameters.at(1));
        double z     = std::stod(ctx.parameters.at(2));
        double yaw   = std::stod(ctx.parameters.at(3)) * DEG2RAD_CMD;
        double pitch = std::stod(ctx.parameters.at(4)) * DEG2RAD_CMD;
        double roll  = std::stod(ctx.parameters.at(5)) * DEG2RAD_CMD;
        double fov   = std::stod(ctx.parameters.at(6)) * DEG2RAD_CMD;
        if (fov <= 0.0)
            return L"400 TRACKING ERROR TARGET_CAMERA fov must be > 0\r\n";
        tracker_registry::instance().update_target_camera(
            ctx.channel_index, ctx.layer_index(), true, x, y, z, yaw, pitch, roll, fov);
        return L"202 TRACKING OK\r\n";
    } catch (const std::exception& e) {
        return L"400 TRACKING ERROR " + caspar::u16(e.what()) + L"\r\n";
    }
}

// ---------------------------------------------------------------------------
// TRACKING TARGET_MAP — screen mapping for mode TARGET
//
// gain        : maps projected NDC[-1,1] to fill_translation (0.5 = NDC edge → frame edge).
// ref_dist_m  : optional. 0 = no distance scaling; otherwise fill_scale = ref_dist / view_z,
//               so the graphic shrinks as the subject moves away.
// aspect      : optional frame aspect ratio (W/H); <= 0 leaves the current value.
// With no argument, queries. Live — no rebind needed.
//
// Syntax: TRACKING <ch>-<layer> TARGET_MAP <gain> [ref_dist_m] [aspect]
// ---------------------------------------------------------------------------
static std::wstring tracking_target_map_command(command_context& ctx)
{
    if (ctx.parameters.empty()) {
        auto binding = tracker_registry::instance().get_binding(ctx.channel_index, ctx.layer_index());
        if (!binding)
            return L"404 TRACKING ERROR No binding on this channel/layer\r\n";
        std::wostringstream ss;
        ss << L"201 TRACKING OK\r\n"
           << L"TARGET_MAP " << binding->target_gain << L" " << binding->target_ref_dist_m
           << L" " << binding->target_aspect << L"\r\n";
        return ss.str();
    }

    try {
        auto binding = tracker_registry::instance().get_binding(ctx.channel_index, ctx.layer_index());
        if (!binding)
            return L"404 TRACKING ERROR No binding on this channel/layer\r\n";
        double gain      = std::stod(ctx.parameters.at(0));
        double ref_dist  = ctx.parameters.size() > 1 ? std::stod(ctx.parameters[1]) : binding->target_ref_dist_m;
        double aspect    = ctx.parameters.size() > 2 ? std::stod(ctx.parameters[2]) : binding->target_aspect;
        if (ref_dist < 0.0)
            return L"400 TRACKING ERROR TARGET_MAP ref_dist_m must be >= 0\r\n";
        tracker_registry::instance().update_target_map(
            ctx.channel_index, ctx.layer_index(), gain, ref_dist, aspect);
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
    repo->register_channel_command(L"Tracking Commands", L"TRACKING DELAY",          tracking_delay_command,          0);
    repo->register_channel_command(L"Tracking Commands", L"TRACKING GENLOCK",        tracking_genlock_command,        0);
    repo->register_channel_command(L"Tracking Commands", L"TRACKING NODAL",          tracking_nodal_command,          0);
    repo->register_channel_command(L"Tracking Commands", L"TRACKING DOF",            tracking_dof_command,            0);
    repo->register_channel_command(L"Tracking Commands", L"TRACKING LENS",           tracking_lens_command,           0);
    repo->register_channel_command(L"Tracking Commands", L"TRACKING TARGET_CAMERA",   tracking_target_camera_command,  0);
    repo->register_channel_command(L"Tracking Commands", L"TRACKING TARGET_MAP",      tracking_target_map_command,     0);
    repo->register_channel_command(L"Tracking Commands", L"TRACKING INFO",        tracking_info_command,        0);

    // Global command (no channel required)
    repo->register_command(L"Tracking Commands", L"TRACKING LIST", tracking_list_command, 0);
}

}} // namespace caspar::tracking
