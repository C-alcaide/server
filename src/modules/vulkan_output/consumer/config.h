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

#include <core/video_format.h>

#include <boost/property_tree/ptree_fwd.hpp>

#include <string>

namespace caspar { namespace vulkan_output {

enum class hdr_transfer
{
    sdr,
    pq,
    hlg
};

// Output color gamut (primaries)
enum class output_gamut
{
    bt709,     // ITU-R BT.709 / sRGB (default, no conversion needed)
    bt2020,    // ITU-R BT.2020 / BT.2100
    p3_d65,    // DCI-P3 with D65 white point (Display P3, Apple wide color)
    p3_dci,    // DCI-P3 with DCI white point (cinema projection, gamma 2.6)
    adobe_rgb, // Adobe RGB (1998)
};

// Output electro-optical transfer function
enum class output_eotf
{
    srgb,      // sRGB ~2.2 curve (default, matches mixer working space)
    linear,    // Linear light (1:1, no curve) — for compositing previews
    pq,        // SMPTE ST 2084 (Perceptual Quantizer) — HDR10, BT.2100
    hlg,       // ARIB STD-B67 (Hybrid Log-Gamma) — live broadcast HDR
    gamma24,   // Pure gamma 2.4 (EBU broadcast reference)
    gamma26,   // Pure gamma 2.6 (DCI cinema projection)
};

enum class disconnect_behavior
{
    hold,  // Hold last frame (black if no frame yet)
    black, // Clear to black on disconnect
    retry  // Continuously retry (default)
};

enum class gsync_reference
{
    internal, // Use GPU VSync as timing reference
    external  // Use external house sync (BNC input on Quadro Sync II card)
};

struct configuration
{
    int          gpu_index    = 0;     // Physical GPU index
    int          output_index = 1;     // Display output index (1-based, like DeckLink)
    int          buffer_depth = 3;     // Number of pre-scheduled frames
    int          sync_group   = 0;     // 0 = no sync, >0 = present barrier group
    bool         borderless   = true;  // For fallback FSE mode
    hdr_transfer transfer     = hdr_transfer::sdr;
    output_gamut gamut        = output_gamut::bt709;   // Output color gamut
    output_eotf  eotf         = output_eotf::srgb;    // Output transfer function
    int          max_cll      = 1000;
    int          max_fall     = 400;
    bool         identify_on_start = false;

    // Display disconnect behavior
    disconnect_behavior on_disconnect = disconnect_behavior::retry;

    // Subregion (same as DeckLink)
    int src_x    = 0;
    int src_y    = 0;
    int dest_x   = 0;
    int dest_y   = 0;
    int region_w = 0; // 0 = full width
    int region_h = 0; // 0 = full height

    // Optional explicit video mode for the output
    std::wstring video_mode;

    // Presentation delay (frames) — compensates for downstream pipeline latency
    // (e.g. scaler, audio de-embedder, LED processor). Video is held in the buffer
    // this many extra frames before being presented.
    int delay_frames = 0;

    // Quadro Sync II (NvAPI)
    bool             gsync_enabled  = false; // Enable GSync framelock
    bool             gsync_master   = false; // This output is the master (others slave to it)
    gsync_reference  gsync_source   = gsync_reference::internal;
    bool             edid_auto_hdr  = false; // Auto-detect HDR from EDID
};

configuration parse_config(const boost::property_tree::wptree& ptree);

}} // namespace caspar::vulkan_output
