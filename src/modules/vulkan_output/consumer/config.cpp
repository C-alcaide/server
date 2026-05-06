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
    config.sync_group   = ptree.get(L"sync-group", 0);
    config.video_mode   = ptree.get(L"video-mode", L"");
    config.identify_on_start = ptree.get(L"identify-on-start", false);

    // HDR
    auto transfer_str = ptree.get(L"transfer", L"sdr");
    if (transfer_str == L"pq")
        config.transfer = hdr_transfer::pq;
    else if (transfer_str == L"hlg")
        config.transfer = hdr_transfer::hlg;
    else
        config.transfer = hdr_transfer::sdr;

    auto hdr_meta = ptree.get_child_optional(L"hdr-metadata");
    if (hdr_meta) {
        config.max_cll  = hdr_meta->get(L"max-cll", 1000);
        config.max_fall = hdr_meta->get(L"max-fall", 400);

        auto t = hdr_meta->get(L"transfer", L"");
        if (t == L"pq")  config.transfer = hdr_transfer::pq;
        if (t == L"hlg") config.transfer = hdr_transfer::hlg;
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
    else
        config.on_disconnect = disconnect_behavior::retry;

    // Presentation delay
    config.delay_frames = ptree.get(L"delay", 0);

    // Quadro Sync II (NvAPI)
    auto gsync = ptree.get_child_optional(L"gsync");
    if (gsync) {
        config.gsync_enabled = gsync->get(L"enabled", false);
        config.gsync_master  = gsync->get(L"master", false);
        auto ref_str         = gsync->get(L"reference", L"internal");
        if (ref_str == L"external" || ref_str == L"house")
            config.gsync_source = gsync_reference::external;
        else
            config.gsync_source = gsync_reference::internal;
    }

    // EDID auto-detection
    config.edid_auto_hdr = ptree.get(L"edid-auto-hdr", false);

    return config;
}

}} // namespace caspar::vulkan_output
