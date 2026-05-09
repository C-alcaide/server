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

    // ─── EDID Emulation ──────────────────────────────────────────────────────

    // Inject a synthetic EDID on an unconnected output.
    // Returns the NvAPI displayId used (needed for removal), or 0 on failure.
    // output_index is 1-based (same as config <device>).
    // NOTE: This makes the GPU report the connector as "connected" but does NOT
    // create a usable Windows desktop display. A physical dongle or IddCx virtual
    // display driver is needed for EnumDisplayMonitors to see the output.
    uint32_t inject_edid(int gpu_index, int output_index, uint32_t width, uint32_t height, double refresh_hz);

    // Remove a previously injected EDID, restoring the output to unconnected state.
    bool remove_edid(int gpu_index, uint32_t display_id);

    // Persist the EDID of a currently connected monitor. After this, if the cable
    // is disconnected, the GPU still reports the output as connected with the same
    // EDID — Windows keeps the display in the desktop topology. Persists across
    // reboots. Call remove_edid() to clear the override.
    // Requires Administrator privileges and a professional GPU (Quadro/RTX A-series).
    bool persist_edid(int gpu_index, int output_index);

    // ─── Dedicated Display ───────────────────────────────────────────────────

    // Information about a dedicated display managed by the NVIDIA driver.
    // Dedicated displays are NOT part of the Windows desktop — they remain detached
    // from DWM even when no application has acquired them (output stays black).
    struct dedicated_display_info
    {
        uint32_t     display_id  = 0;
        bool         is_acquired = false; // Already acquired by another process?
        bool         is_mosaic   = false; // Part of a Mosaic grid?
    };

    // Enumerate all dedicated displays managed by the NVIDIA driver.
    // Returns empty if the GPU/driver doesn't support dedicated displays.
    std::vector<dedicated_display_info> get_dedicated_displays();

    // Acquire exclusive access to a dedicated display. The display must already be
    // configured as "dedicated" in NVIDIA Control Panel (or by the driver).
    // Returns a DisplaySource handle on success, 0 on failure.
    // Once acquired, the display stays detached from DWM even if the acquiring
    // process crashes — this prevents the Windows desktop from appearing on the output.
    uint64_t acquire_dedicated_display(uint32_t display_id);

    // Release a previously acquired dedicated display. The display remains "dedicated"
    // (detached from DWM) but is available for other processes to acquire.
    bool release_dedicated_display(uint32_t display_id);

    // Find and acquire the dedicated display matching the given output index.
    // Convenience wrapper: enumerates dedicated displays, maps to the output index,
    // and acquires if found. Returns the NvAPI displayId on success, 0 if the output
    // is not a dedicated display or acquisition failed.
    uint32_t acquire_dedicated_display_by_output(int gpu_index, int output_index);

    // ─── Hardware HDR (Display Engine) ───────────────────────────────────────

    // Resolve a 1-based output index to an NvAPI displayId.
    // Returns 0 if the display is not found.
    uint32_t resolve_display_id(int gpu_index, int output_index);

    // Enable HDR output via the display engine hardware.
    // The GPU's display engine performs PQ EOTF encoding and BT.709→BT.2020 gamut
    // mapping in hardware — zero GPU shader cost, no extra frame latency.
    // Source must be scRGB FP16 (linear, sRGB primaries). The swapchain should use
    // VK_FORMAT_R16G16B16A16_SFLOAT with linear values where RGB(1,1,1) = 80 nits.
    // Returns true if the display engine accepted the HDR mode.
    bool enable_hdr_output(uint32_t display_id, int max_cll, int max_fall);

    // Disable hardware HDR output, returning the display to SDR mode.
    bool disable_hdr_output(uint32_t display_id);

    // Query whether the display supports hardware HDR (ST2084 PQ).
    bool supports_hdr_output(uint32_t display_id);

  private:
    bool     available_    = false;
    int      gsync_count_  = 0;

#ifdef CASPAR_NVAPI_ENABLED
    struct impl;
    struct impl* impl_ = nullptr;
#endif
};

}} // namespace caspar::vulkan_output
