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

#include <cstdint>
#include <string>
#include <vector>

namespace caspar { namespace vulkan_output {

// ─── EDID Data ──────────────────────────────────────────────────────────────

struct edid_info
{
    std::wstring manufacturer;
    std::wstring model;
    uint32_t     max_width       = 0;
    uint32_t     max_height      = 0;
    double       max_refresh     = 0.0;
    bool         supports_hdr    = false;
    uint32_t     max_luminance   = 0;   // cd/m² (from HDR static metadata block)
    uint32_t     min_luminance   = 0;   // 0.0001 cd/m² units
    bool         supports_10bit  = false;
    std::vector<uint8_t> raw_edid;
};

// ─── Quadro Sync Status ─────────────────────────────────────────────────────

enum class sync_source
{
    vsync,
    house_sync
};

enum class sync_role
{
    none,
    master,
    slave
};

struct gsync_status
{
    bool        available       = false;  // Is a GSync board detected?
    bool        synced          = false;  // Is timing currently locked?
    bool        signal_present  = false;  // Is sync signal available?
    bool        house_sync      = false;  // Is house sync connected?
    uint32_t    house_sync_freq = 0;      // Incoming house sync frequency (Hz)
    uint32_t    refresh_rate    = 0;      // Current refresh rate
    sync_source source          = sync_source::vsync;
    sync_role   role            = sync_role::none;
};

// ─── NvAPI Wrapper Class ────────────────────────────────────────────────────

class nvapi_helpers
{
  public:
    nvapi_helpers();
    ~nvapi_helpers();

    nvapi_helpers(const nvapi_helpers&)            = delete;
    nvapi_helpers& operator=(const nvapi_helpers&) = delete;

    // Returns true if NvAPI initialized successfully
    bool is_available() const { return available_; }

    // ─── EDID ────────────────────────────────────────────────────────────────

    // Read EDID from a display connected to the specified GPU
    edid_info read_edid(int gpu_index, int display_output_id);

    // ─── Quadro Sync ─────────────────────────────────────────────────────────

    // Query current sync status for the given GPU
    gsync_status get_sync_status(int gpu_index);

    // Configure sync: set the specified display as master, others as slaves
    bool configure_sync(int gpu_index, int master_display_id, sync_source source);

    // Disable sync for all displays on this sync device
    bool disable_sync();

    // Get number of detected GSync boards
    int gsync_device_count() const { return gsync_count_; }

  private:
    bool     available_    = false;
    int      gsync_count_  = 0;

#ifdef CASPAR_NVAPI_ENABLED
    struct impl;
    struct impl* impl_ = nullptr;
#endif
};

}} // namespace caspar::vulkan_output
