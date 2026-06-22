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

#pragma once

#include <boost/property_tree/ptree_fwd.hpp>

#include <string>

namespace caspar { namespace vulkan_output {

enum class hdr_transfer
{
    sdr,
    pq,
    hlg
};

enum class output_gamut
{
    bt709,
    bt2020,
    p3_d65,
};

enum class output_eotf
{
    srgb,
    linear,
    pq,
    hlg,
    gamma24,
};

enum class gsync_reference
{
    vsync,
    external, // House sync
};

enum class disconnect_behavior
{
    hold,  // Hold last frame (silent)
    black, // Output black
    retry, // Periodically attempt swapchain recreation
};

struct configuration
{
    int          gpu_index    = 0;     // Physical GPU index (reserved for multi-GPU PR)
    int          output_index = 1;     // Display output index (1-based)
    int          buffer_depth = 3;     // Swapchain image count / pre-scheduled frames
    int          delay_frames = 0;     // Extra frames to hold before presenting (A/V sync)
    double       delay_ms     = 0.0;   // Sub-frame delay in ms (clamped to one frame period)
    int          sync_group   = 0;     // Present barrier group (0=disabled, >0=frame-lock with peers)
    bool         borderless   = true;  // Borderless fullscreen (always true for now)
    hdr_transfer transfer     = hdr_transfer::sdr;
    output_gamut gamut        = output_gamut::bt709;
    output_eotf  eotf         = output_eotf::srgb;

    // Subregion (crop from source frame)
    int src_x    = 0;
    int src_y    = 0;
    int region_w = 0; // 0 = full width
    int region_h = 0; // 0 = full height

    // Display name matching: if set, selects monitor by substring match on device name
    // instead of index. Example: "BNQ" matches BenQ monitors.
    std::wstring display_name;

    // NvAPI automation (requires CASPAR_NVAPI_ENABLED + professional GPU)
    bool edid_emulation    = false;  // Inject synthetic EDID on disconnected outputs
    bool edid_auto_hdr     = false;  // Read EDID to auto-detect HDR capabilities
    bool persist_edid      = false;  // Lock EDID so display survives cable disconnect
    bool hardware_hdr      = false;  // Use display engine PQ+BT.2020 (zero GPU cost)
    int  max_cll           = 1000;   // Peak content luminance (cd/m²)
    int  max_fall          = 400;    // Max frame-average luminance (cd/m²)
    bool gsync_enabled     = false;  // Enable Quadro Sync framelock
    bool gsync_master      = false;  // Configure this output as sync master
    gsync_reference gsync_source = gsync_reference::vsync;

    // Display disconnect recovery behavior
    disconnect_behavior on_disconnect = disconnect_behavior::retry;
};

configuration parse_config(const boost::property_tree::wptree& ptree);

}} // namespace caspar::vulkan_output
