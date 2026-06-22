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

struct configuration
{
    int          gpu_index    = 0;     // Physical GPU index (reserved for multi-GPU PR)
    int          output_index = 1;     // Display output index (1-based)
    int          buffer_depth = 3;     // Swapchain image count / pre-scheduled frames
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
};

configuration parse_config(const boost::property_tree::wptree& ptree);

}} // namespace caspar::vulkan_output
