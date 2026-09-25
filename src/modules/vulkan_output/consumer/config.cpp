/*
 * Copyright (c) 2026 CasparCG Contributors
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

#include "config.h"

#include <common/except.h>
#include <common/utf.h>

#include <boost/property_tree/ptree.hpp>

namespace caspar { namespace vulkan_output {

namespace {

/// Every word below either names a value or is refused. They used to fall through to a default,
/// which is the tree's most repeated defect: the consumer starts, says nothing, and outputs the
/// value nobody chose -- an SDR signal from `<transfer>PQ </transfer>`, a hold from a misspelled
/// `on-disconnect`, internal sync from `<reference>hous</reference>`.
[[noreturn]] void refuse(const wchar_t* key, const std::wstring& value, const wchar_t* valid)
{
    CASPAR_THROW_EXCEPTION(user_error() << msg_info(u8(std::wstring(L"vulkan-output <") + key + L"> must be " +
                                                       valid + L", not '" + value + L"'")));
}

} // namespace

configuration parse_config(const boost::property_tree::wptree& ptree)
{
    configuration config;

    config.gpu_index    = ptree.get(L"gpu", 0);
    config.output_index = ptree.get(L"device", 1);
    config.buffer_depth = ptree.get(L"buffer-depth", 3);
    config.sync_group   = ptree.get(L"sync-group", 0);
    config.video_mode   = ptree.get(L"video-mode", L"");
    // PARSED, STORED AND READ BY NOTHING -- the `MIXER EXPOSURE` class in a config element. The
    // output always takes the display's current mode, so an operator who set this believed they
    // had forced a mode they had not. Refused until something implements it.
    if (!config.video_mode.empty())
        CASPAR_THROW_EXCEPTION(user_error() << msg_info(
            "vulkan-output <video-mode> is not implemented: the output uses the display's current mode. "
            "Remove the element, or set the mode on the display."));
    config.identify_on_start = ptree.get(L"identify-on-start", false);

    // HDR
    auto transfer_str = ptree.get(L"transfer", L"sdr");
    if (transfer_str == L"pq")
        config.transfer = hdr_transfer::pq;
    else if (transfer_str == L"hlg")
        config.transfer = hdr_transfer::hlg;
    else if (transfer_str == L"sdr")
        config.transfer = hdr_transfer::sdr;
    else
        refuse(L"transfer", transfer_str, L"sdr, pq or hlg");

    // Color gamut (output primaries)
    auto gamut_str = ptree.get(L"gamut", L"");
    if (gamut_str == L"bt2020" || gamut_str == L"2020")
        config.gamut = output_gamut::bt2020;
    else if (gamut_str == L"p3-d65" || gamut_str == L"p3" || gamut_str == L"display-p3")
        config.gamut = output_gamut::p3_d65;
    else if (gamut_str == L"p3-dci" || gamut_str == L"dci-p3")
        config.gamut = output_gamut::p3_dci;
    else if (gamut_str == L"adobe-rgb" || gamut_str == L"adobergb")
        config.gamut = output_gamut::adobe_rgb;
    else if (gamut_str.empty() || gamut_str == L"bt709" || gamut_str == L"709")
        config.gamut = output_gamut::bt709;
    else
        refuse(L"gamut", gamut_str, L"bt709, bt2020, p3-d65, p3-dci or adobe-rgb");

    // Transfer function (EOTF) — if not explicitly set, infer from hdr_transfer
    auto eotf_str = ptree.get(L"eotf", L"");
    if (eotf_str == L"linear")
        config.eotf = output_eotf::linear;
    else if (eotf_str == L"pq" || eotf_str == L"st2084")
        config.eotf = output_eotf::pq;
    else if (eotf_str == L"hlg")
        config.eotf = output_eotf::hlg;
    else if (eotf_str == L"gamma24" || eotf_str == L"2.4")
        config.eotf = output_eotf::gamma24;
    else if (eotf_str == L"gamma26" || eotf_str == L"2.6")
        config.eotf = output_eotf::gamma26;
    else if (eotf_str == L"srgb")
        config.eotf = output_eotf::srgb;
    else if (!eotf_str.empty())
        refuse(L"eotf", eotf_str, L"linear, pq, hlg, gamma24, gamma26 or srgb");
    else {
        // Infer from legacy transfer setting
        if (config.transfer == hdr_transfer::pq)
            config.eotf = output_eotf::pq;
        else if (config.transfer == hdr_transfer::hlg)
            config.eotf = output_eotf::hlg;
        else
            config.eotf = output_eotf::srgb;
    }

    // If gamut not explicitly set, infer from transfer mode
    if (gamut_str.empty()) {
        if (config.transfer == hdr_transfer::pq || config.transfer == hdr_transfer::hlg)
            config.gamut = output_gamut::bt2020;
    }

    auto hdr_meta = ptree.get_child_optional(L"hdr-metadata");
    if (hdr_meta) {
        config.max_cll  = hdr_meta->get(L"max-cll", 1000);
        config.max_fall = hdr_meta->get(L"max-fall", 400);
        // Same spelling as the DeckLink and FFmpeg consumers: the same four numbers describe
        // the same mastering display, and an operator should not need three vocabularies.
        config.min_dml  = hdr_meta->get(L"min-dml", 0.005);
        config.max_dml  = hdr_meta->get(L"max-dml", 1000.0);

        auto t = hdr_meta->get(L"transfer", L"");
        if (t == L"pq")  config.transfer = hdr_transfer::pq;
        else if (t == L"hlg") config.transfer = hdr_transfer::hlg;
        else if (!t.empty()) refuse(L"hdr-metadata/transfer", t, L"pq or hlg");
    }

    // Subregion
    auto subregion = ptree.get_child_optional(L"subregion");
    if (subregion) {
        config.src_x    = subregion->get(L"src-x", 0);
        config.src_y    = subregion->get(L"src-y", 0);
        config.dest_x   = subregion->get(L"dest-x", 0);
        config.dest_y   = subregion->get(L"dest-y", 0);
        config.region_w = subregion->get(L"width", 0);
        config.region_h = subregion->get(L"height", 0);
    }

    // Disconnect behavior
    auto disconnect_str = ptree.get(L"on-disconnect", L"retry");
    if (disconnect_str == L"hold")
        config.on_disconnect = disconnect_behavior::hold;
    else if (disconnect_str == L"black")
        config.on_disconnect = disconnect_behavior::black;
    else if (disconnect_str == L"retry")
        config.on_disconnect = disconnect_behavior::retry;
    else
        refuse(L"on-disconnect", disconnect_str, L"retry, hold or black");

    // Presentation delay
    config.delay_frames = ptree.get(L"delay", 0);
    config.delay_ms     = ptree.get(L"delay-ms", 0.0);

    // Quadro Sync II (NvAPI)
    auto gsync = ptree.get_child_optional(L"gsync");
    if (gsync) {
        config.gsync_enabled = gsync->get(L"enabled", false);
        config.gsync_master  = gsync->get(L"master", false);
        auto ref_str         = gsync->get(L"reference", L"internal");
        if (ref_str == L"external" || ref_str == L"house")
            config.gsync_source = gsync_reference::external;
        else if (ref_str == L"internal")
            config.gsync_source = gsync_reference::internal;
        else
            refuse(L"gsync/reference", ref_str, L"internal, external or house");
    }

    // EDID auto-detection
    config.edid_auto_hdr = ptree.get(L"edid-auto-hdr", false);

    // EDID emulation for headless outputs
    config.edid_emulation = ptree.get(L"edid-emulation", false);

    // EDID persistence (lock connected monitor's EDID to survive cable disconnect)
    config.persist_edid = ptree.get(L"persist-edid", false);

    // Display name matching (overrides index-based selection)
    config.display_name = ptree.get(L"display-name", L"");

    // Display blanker companion process
    config.display_blanker = ptree.get(L"display-blanker", false);

    // Per-consumer tone mapping override
    auto tone_map_str = ptree.get(L"auto-tone-map", L"");
    if (!tone_map_str.empty()) {
        if (tone_map_str == L"none")
            config.tone_map_op = 0;
        else if (tone_map_str == L"reinhard")
            config.tone_map_op = 1;
        else if (tone_map_str == L"aces_filmic")
            config.tone_map_op = 2;
        else if (tone_map_str == L"aces_rrt")
            config.tone_map_op = 3;
        else if (tone_map_str == L"hlg_ootf")
            config.tone_map_op = 7;
        else
            refuse(L"auto-tone-map", tone_map_str, L"none, reinhard, aces_filmic, aces_rrt or hlg_ootf");
    }
    config.display_peak_luminance = ptree.get(L"display-peak-luminance", config.display_peak_luminance);

    return config;
}

}} // namespace caspar::vulkan_output
