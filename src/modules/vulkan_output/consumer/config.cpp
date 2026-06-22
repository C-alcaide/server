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

#include <boost/property_tree/ptree.hpp>

namespace caspar { namespace vulkan_output {

configuration parse_config(const boost::property_tree::wptree& ptree)
{
    configuration config;

    config.gpu_index    = ptree.get(L"gpu", 0);
    config.output_index = ptree.get(L"device", 1);
    config.buffer_depth = ptree.get(L"buffer-depth", 3);

    // Display name matching (overrides index-based selection)
    config.display_name = ptree.get(L"display-name", L"");

    // HDR transfer
    auto transfer_str = ptree.get(L"transfer", L"sdr");
    if (transfer_str == L"pq")
        config.transfer = hdr_transfer::pq;
    else if (transfer_str == L"hlg")
        config.transfer = hdr_transfer::hlg;
    else
        config.transfer = hdr_transfer::sdr;

    // Color gamut
    auto gamut_str = ptree.get(L"gamut", L"");
    if (gamut_str == L"bt2020" || gamut_str == L"2020")
        config.gamut = output_gamut::bt2020;
    else if (gamut_str == L"p3-d65" || gamut_str == L"p3" || gamut_str == L"display-p3")
        config.gamut = output_gamut::p3_d65;
    else
        config.gamut = output_gamut::bt709;

    // Transfer function (EOTF)
    auto eotf_str = ptree.get(L"eotf", L"");
    if (eotf_str == L"linear")
        config.eotf = output_eotf::linear;
    else if (eotf_str == L"pq" || eotf_str == L"st2084")
        config.eotf = output_eotf::pq;
    else if (eotf_str == L"hlg")
        config.eotf = output_eotf::hlg;
    else if (eotf_str == L"gamma24" || eotf_str == L"2.4")
        config.eotf = output_eotf::gamma24;
    else if (eotf_str == L"srgb")
        config.eotf = output_eotf::srgb;
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

    // Subregion
    auto subregion = ptree.get_child_optional(L"subregion");
    if (subregion) {
        config.src_x    = subregion->get(L"src-x", 0);
        config.src_y    = subregion->get(L"src-y", 0);
        config.region_w = subregion->get(L"width", 0);
        config.region_h = subregion->get(L"height", 0);
    }

    // NvAPI automation
    config.edid_emulation = ptree.get(L"edid-emulation", false);
    config.edid_auto_hdr  = ptree.get(L"edid-auto-hdr", false);
    config.persist_edid   = ptree.get(L"persist-edid", false);
    config.hardware_hdr   = ptree.get(L"hardware-hdr", false);
    config.max_cll        = ptree.get(L"max-cll", 1000);
    config.max_fall       = ptree.get(L"max-fall", 400);
    config.gsync_enabled  = ptree.get(L"gsync", false);
    config.gsync_master   = ptree.get(L"gsync-master", false);

    auto gsync_src_str = ptree.get(L"gsync-source", L"vsync");
    if (gsync_src_str == L"external" || gsync_src_str == L"house")
        config.gsync_source = gsync_reference::external;

    // Presentation timing
    config.delay_frames = ptree.get(L"delay-frames", 0);
    config.delay_ms     = ptree.get(L"delay-ms", 0.0);
    config.sync_group   = ptree.get(L"sync-group", 0);

    // Separate device mode (TDR isolation + multi-queue)
    config.separate_device = ptree.get(L"separate-device", false);

    // Display disconnect behavior
    auto disc_str = ptree.get(L"on-disconnect", L"retry");
    if (disc_str == L"hold")
        config.on_disconnect = disconnect_behavior::hold;
    else if (disc_str == L"black")
        config.on_disconnect = disconnect_behavior::black;
    else
        config.on_disconnect = disconnect_behavior::retry;

    return config;
}

}} // namespace caspar::vulkan_output
