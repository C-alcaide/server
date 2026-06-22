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

#include "nvapi_helpers.h"

#include <common/log.h>

#ifdef CASPAR_NVAPI_ENABLED

#include <windows.h>
#include <nvapi.h>

#include <algorithm>
#include <cstring>
#include <vector>

namespace caspar { namespace vulkan_output {

namespace {

std::wstring to_wstring(const char* str)
{
    if (!str)
        return L"";
    return std::wstring(str, str + strlen(str));
}

// Parse EDID manufacturer ID (3 chars from bytes 8-9)
std::wstring parse_edid_manufacturer(const uint8_t* edid)
{
    uint16_t id = (edid[8] << 8) | edid[9];
    wchar_t  mfr[4];
    mfr[0] = static_cast<wchar_t>(((id >> 10) & 0x1F) + L'A' - 1);
    mfr[1] = static_cast<wchar_t>(((id >> 5) & 0x1F) + L'A' - 1);
    mfr[2] = static_cast<wchar_t>((id & 0x1F) + L'A' - 1);
    mfr[3] = 0;
    return mfr;
}

// Parse EDID model name from descriptor blocks (bytes 54-125)
std::wstring parse_edid_model(const uint8_t* edid, uint32_t edid_size)
{
    // Look for Monitor Name descriptor (tag 0xFC)
    for (int base = 54; base <= 108 && base + 18 <= static_cast<int>(edid_size); base += 18) {
        if (edid[base] == 0 && edid[base + 1] == 0 && edid[base + 3] == 0xFC) {
            std::wstring name;
            for (int i = 5; i < 18; ++i) {
                if (edid[base + i] == 0x0A || edid[base + i] == 0)
                    break;
                name += static_cast<wchar_t>(edid[base + i]);
            }
            return name;
        }
    }
    return L"Unknown";
}

// Check for HDR Static Metadata Data Block in CTA-861 extension (EDID block 1+)
bool parse_edid_hdr_support(const uint8_t* edid, uint32_t edid_size, uint32_t& max_lum, uint32_t& min_lum)
{
    if (edid_size < 256)
        return false;

    // CTA extension block starts at byte 128
    const uint8_t* ext = edid + 128;
    if (ext[0] != 0x02) // Not a CTA-861 extension
        return false;

    uint8_t dtd_offset = ext[2];
    if (dtd_offset < 4)
        return false;

    // Parse data blocks in the CTA extension
    int pos = 4;
    while (pos < dtd_offset && pos < 127) {
        uint8_t tag    = (ext[pos] >> 5) & 0x07;
        uint8_t length = ext[pos] & 0x1F;

        if (pos + length + 1 > 128)
            break; // data block overflows the extension block

        if (tag == 7 && length >= 2) { // Extended tag
            uint8_t ext_tag = ext[pos + 1];
            if (ext_tag == 6 && length >= 3) { // HDR Static Metadata
                // Byte 3: EOTF support (bit 0=SDR, bit 1=HDR, bit 2=SMPTE ST2084, bit 3=HLG)
                bool has_pq = (ext[pos + 2] & 0x04) != 0;
                if (has_pq && length >= 5) {
                    // Byte 4: max luminance (raw value, cv = 50 * 2^(raw/32))
                    max_lum = static_cast<uint32_t>(50.0 * pow(2.0, ext[pos + 4] / 32.0));
                    // Byte 5: max frame-avg luminance
                    if (length >= 6)
                        min_lum = static_cast<uint32_t>(50.0 * pow(2.0, ext[pos + 5] / 32.0));
                }
                return has_pq;
            }
        }

        pos += length + 1;
    }

    return false;
}

} // namespace

// ─── Implementation struct holding NvAPI handles ────────────────────────────

struct nvapi_helpers::impl
{
    NvPhysicalGpuHandle gpus[NVAPI_MAX_PHYSICAL_GPUS] = {};
    NvU32               gpu_count                     = 0;
    NvGSyncDeviceHandle gsync_handles[NVAPI_MAX_GSYNC_DEVICES] = {};
    NvU32               gsync_count                   = 0;
};

nvapi_helpers::nvapi_helpers()
{
    auto status = NvAPI_Initialize();
    if (status != NVAPI_OK) {
        NvAPI_ShortString err;
        NvAPI_GetErrorMessage(status, err);
        CASPAR_LOG(warning) << L"[vulkan_output] NvAPI_Initialize failed: " << to_wstring(err);
        return;
    }

    impl_ = new impl();

    // Enumerate GPUs
    status = NvAPI_EnumPhysicalGPUs(impl_->gpus, &impl_->gpu_count);
    if (status != NVAPI_OK) {
        CASPAR_LOG(warning) << L"[vulkan_output] NvAPI_EnumPhysicalGPUs failed.";
        impl_->gpu_count = 0;
    }

    // Enumerate GSync devices
    status = NvAPI_GSync_EnumSyncDevices(impl_->gsync_handles, &impl_->gsync_count);
    if (status == NVAPI_NVIDIA_DEVICE_NOT_FOUND) {
        impl_->gsync_count = 0;
        CASPAR_LOG(info) << L"[vulkan_output] No Quadro Sync devices found.";
    } else if (status != NVAPI_OK) {
        impl_->gsync_count = 0;
    }

    gsync_count_ = static_cast<int>(impl_->gsync_count);
    available_   = true;

    CASPAR_LOG(info) << L"[vulkan_output] NvAPI initialized: " << impl_->gpu_count << L" GPU(s), "
                     << impl_->gsync_count << L" GSync device(s).";
}

nvapi_helpers::~nvapi_helpers()
{
    if (available_)
        NvAPI_Unload();
    delete impl_;
}

// ─── EDID ───────────────────────────────────────────────────────────────────

edid_info nvapi_helpers::read_edid(int gpu_index, int display_output_id)
{
    edid_info info{};

    if (!available_ || !impl_ || gpu_index >= static_cast<int>(impl_->gpu_count))
        return info;

    // Map our 1-based output index to an NvAPI display ID.
    // NvAPI_GPU_GetEDID requires a proper display ID from the enumeration API,
    // not a sequential index.
    NvU32 actual_display_id = 0;
    {
        NvU32 disp_id_count = 0;
        auto  st = NvAPI_GPU_GetConnectedDisplayIds(impl_->gpus[gpu_index], nullptr, &disp_id_count, 0);
        if (st != NVAPI_OK || disp_id_count == 0) {
            CASPAR_LOG(debug) << L"[vulkan_output] No connected displays for GPU " << gpu_index;
            return info;
        }

        std::vector<NV_GPU_DISPLAYIDS> display_ids(disp_id_count);
        for (auto& d : display_ids)
            d.version = NV_GPU_DISPLAYIDS_VER;

        st = NvAPI_GPU_GetConnectedDisplayIds(impl_->gpus[gpu_index], display_ids.data(), &disp_id_count, 0);
        if (st != NVAPI_OK) {
            CASPAR_LOG(debug) << L"[vulkan_output] GetConnectedDisplayIds failed for GPU " << gpu_index;
            return info;
        }

        int target_idx = display_output_id - 1; // Convert 1-based to 0-based
        if (target_idx < 0 || target_idx >= static_cast<int>(disp_id_count)) {
            CASPAR_LOG(debug) << L"[vulkan_output] Output index " << display_output_id
                              << L" out of range (GPU has " << disp_id_count << L" connected displays)";
            return info;
        }

        actual_display_id = display_ids[target_idx].displayId;
    }

    // Read EDID using NvAPI_GPU_GetEDID
    NV_EDID edid{};
    edid.version = NV_EDID_VER;
    edid.offset  = 0;

    auto status = NvAPI_GPU_GetEDID(impl_->gpus[gpu_index],
                                    actual_display_id,
                                    &edid);
    if (status != NVAPI_OK) {
        NvAPI_ShortString err;
        NvAPI_GetErrorMessage(status, err);
        CASPAR_LOG(debug) << L"[vulkan_output] GetEDID failed for gpu=" << gpu_index
                          << L" displayId=" << actual_display_id << L": " << to_wstring(err);
        return info;
    }

    // Copy first 256 bytes
    info.raw_edid.assign(edid.EDID_Data, edid.EDID_Data + (std::min)(edid.sizeofEDID, (NvU32)NV_EDID_DATA_SIZE));

    // Read additional pages if EDID is larger than 256 bytes
    NvU32 total_size = edid.sizeofEDID;
    NvU32 edid_id    = edid.edidId;
    NvU32 offset     = NV_EDID_DATA_SIZE;

    while (offset < total_size) {
        NV_EDID page{};
        page.version = NV_EDID_VER;
        page.offset  = offset;

        status = NvAPI_GPU_GetEDID(impl_->gpus[gpu_index],
                                   actual_display_id,
                                   &page);
        if (status != NVAPI_OK || page.edidId != edid_id)
            break;

        uint32_t chunk = (std::min)(page.sizeofEDID - offset, (NvU32)NV_EDID_DATA_SIZE);
        info.raw_edid.insert(info.raw_edid.end(), page.EDID_Data, page.EDID_Data + chunk);
        offset += NV_EDID_DATA_SIZE;
    }

    // Parse EDID fields
    if (info.raw_edid.size() >= 128) {
        const uint8_t* raw = info.raw_edid.data();
        info.manufacturer = parse_edid_manufacturer(raw);
        info.model        = parse_edid_model(raw, static_cast<uint32_t>(info.raw_edid.size()));

        // Preferred timing (first DTD at byte 54)
        info.max_width  = ((raw[58] & 0xF0) << 4) | raw[56];
        info.max_height = ((raw[61] & 0xF0) << 4) | raw[59];

        // Pixel clock in 10kHz units
        uint32_t pixel_clock = raw[54] | (raw[55] << 8);
        if (pixel_clock > 0 && info.max_width > 0 && info.max_height > 0) {
            uint32_t h_total = info.max_width + ((raw[58] & 0x0F) << 8 | raw[57]);
            uint32_t v_total = info.max_height + ((raw[61] & 0x0F) << 8 | raw[60]);
            if (h_total > 0 && v_total > 0)
                info.max_refresh = (pixel_clock * 10000.0) / (h_total * v_total);
        }

        // Check for 10-bit support in CTA extension
        if (info.raw_edid.size() >= 256) {
            const uint8_t* ext_block = raw + 128;
            if (ext_block[0] == 0x02) { // CTA-861
                // Byte 3 bit 4 indicates YCbCr 4:4:4 deep color 10-bit
                // More reliable: look for Video Capability Data Block
                info.supports_10bit = true; // CTA extension generally indicates advanced capabilities
            }
        }

        // HDR support
        uint32_t max_lum = 0, min_lum = 0;
        info.supports_hdr   = parse_edid_hdr_support(raw, static_cast<uint32_t>(info.raw_edid.size()), max_lum, min_lum);
        info.max_luminance  = max_lum;
        info.min_luminance  = min_lum;
    }

    CASPAR_LOG(info) << L"[vulkan_output] EDID: " << info.manufacturer << L" " << info.model
                     << L" " << info.max_width << L"x" << info.max_height
                     << L"@" << static_cast<int>(info.max_refresh) << L"Hz"
                     << (info.supports_hdr ? L" [HDR]" : L"")
                     << (info.supports_10bit ? L" [10-bit]" : L"");

    return info;
}

// ─── Quadro Sync ────────────────────────────────────────────────────────────

gsync_status nvapi_helpers::get_sync_status(int gpu_index)
{
    gsync_status result{};

    if (!available_ || !impl_ || impl_->gsync_count == 0)
        return result;

    if (gpu_index >= static_cast<int>(impl_->gpu_count))
        return result;

    result.available = true;

    // Find which GSync device this GPU is connected to
    for (NvU32 g = 0; g < impl_->gsync_count; ++g) {
        // Query topology
        NvU32 gpu_count = 0;
        auto  status    = NvAPI_GSync_GetTopology(impl_->gsync_handles[g], &gpu_count, nullptr, nullptr, nullptr);
        if (status != NVAPI_OK || gpu_count == 0)
            continue;

        std::vector<NV_GSYNC_GPU> gsync_gpus(gpu_count);
        gsync_gpus[0].version = NV_GSYNC_GPU_VER;

        NvU32 disp_count = 0;
        status = NvAPI_GSync_GetTopology(impl_->gsync_handles[g], &gpu_count, gsync_gpus.data(), &disp_count, nullptr);
        if (status != NVAPI_OK)
            continue;

        // Check if our GPU is in this topology
        bool found = false;
        for (NvU32 i = 0; i < gpu_count; ++i) {
            if (gsync_gpus[i].hPhysicalGpu == impl_->gpus[gpu_index]) {
                found = true;
                result.synced = gsync_gpus[i].isSynced != 0;
                break;
            }
        }

        if (!found)
            continue;

        // Get sync status
        NV_GSYNC_STATUS sync_st{};
        sync_st.version = NV_GSYNC_STATUS_VER;
        status = NvAPI_GSync_GetSyncStatus(impl_->gsync_handles[g], impl_->gpus[gpu_index], &sync_st);
        if (status == NVAPI_OK) {
            result.synced         = sync_st.bIsSynced != 0;
            result.signal_present = sync_st.bIsSyncSignalAvailable != 0;
        }

        // Get status parameters
        NV_GSYNC_STATUS_PARAMS params{};
        params.version = NV_GSYNC_STATUS_PARAMS_VER;
        status = NvAPI_GSync_GetStatusParameters(impl_->gsync_handles[g], &params);
        if (status == NVAPI_OK) {
            result.refresh_rate    = params.refreshRate;
            result.house_sync      = params.bHouseSync != 0;
            result.house_sync_freq = params.houseSyncIncoming;
        }

        // Get control parameters
        NV_GSYNC_CONTROL_PARAMS ctrl{};
        ctrl.version = NV_GSYNC_CONTROL_PARAMS_VER;
        status = NvAPI_GSync_GetControlParameters(impl_->gsync_handles[g], &ctrl);
        if (status == NVAPI_OK) {
            result.source = (ctrl.source == NVAPI_GSYNC_SYNC_SOURCE_HOUSESYNC)
                                ? sync_source::house_sync
                                : sync_source::vsync;
        }

        // Get display roles
        if (disp_count > 0) {
            std::vector<NV_GSYNC_DISPLAY> displays(disp_count);
            displays[0].version = NV_GSYNC_DISPLAY_VER;
            NvU32 gc = 0;
            status = NvAPI_GSync_GetTopology(impl_->gsync_handles[g], &gc, nullptr, &disp_count, displays.data());
            if (status == NVAPI_OK) {
                for (NvU32 d = 0; d < disp_count; ++d) {
                    NvPhysicalGpuHandle disp_gpu = nullptr;
                    NvAPI_SYS_GetPhysicalGpuFromDisplayId(displays[d].displayId, &disp_gpu);
                    if (disp_gpu == impl_->gpus[gpu_index]) {
                        if (displays[d].syncState == NVAPI_GSYNC_DISPLAY_SYNC_STATE_MASTER)
                            result.role = sync_role::master;
                        else if (displays[d].syncState == NVAPI_GSYNC_DISPLAY_SYNC_STATE_SLAVE)
                            result.role = sync_role::slave;
                        break;
                    }
                }
            }
        }

        break; // Found our GPU's sync device
    }

    return result;
}

bool nvapi_helpers::configure_sync(int gpu_index, int master_display_id, sync_source source)
{
    if (!available_ || !impl_ || impl_->gsync_count == 0)
        return false;

    // Find the GSync device for this GPU
    NvGSyncDeviceHandle target_gsync = nullptr;
    for (NvU32 g = 0; g < impl_->gsync_count; ++g) {
        NvU32 gpu_count = 0;
        auto  status    = NvAPI_GSync_GetTopology(impl_->gsync_handles[g], &gpu_count, nullptr, nullptr, nullptr);
        if (status != NVAPI_OK || gpu_count == 0)
            continue;

        std::vector<NV_GSYNC_GPU> gsync_gpus(gpu_count);
        gsync_gpus[0].version = NV_GSYNC_GPU_VER;
        status = NvAPI_GSync_GetTopology(impl_->gsync_handles[g], &gpu_count, gsync_gpus.data(), nullptr, nullptr);
        if (status != NVAPI_OK)
            continue;

        for (NvU32 i = 0; i < gpu_count; ++i) {
            if (gsync_gpus[i].hPhysicalGpu == impl_->gpus[gpu_index]) {
                target_gsync = impl_->gsync_handles[g];
                break;
            }
        }
        if (target_gsync)
            break;
    }

    if (!target_gsync) {
        CASPAR_LOG(warning) << L"[vulkan_output] No GSync device found for GPU " << gpu_index;
        return false;
    }

    // Get all displays in the topology
    NvU32 disp_count = 0;
    auto  status     = NvAPI_GSync_GetTopology(target_gsync, nullptr, nullptr, &disp_count, nullptr);
    if (status != NVAPI_OK || disp_count == 0)
        return false;

    std::vector<NV_GSYNC_DISPLAY> displays(disp_count);
    displays[0].version = NV_GSYNC_DISPLAY_VER;
    NvU32 gc = 0;
    status = NvAPI_GSync_GetTopology(target_gsync, &gc, nullptr, &disp_count, displays.data());
    if (status != NVAPI_OK)
        return false;

    // Configure: master_display_id becomes master, all others become slaves
    for (NvU32 i = 0; i < disp_count; ++i) {
        if (static_cast<int>(displays[i].displayId) == master_display_id)
            displays[i].syncState = NVAPI_GSYNC_DISPLAY_SYNC_STATE_MASTER;
        else
            displays[i].syncState = NVAPI_GSYNC_DISPLAY_SYNC_STATE_SLAVE;
    }

    NvU32 flags = 0;
    status = NvAPI_GSync_SetSyncStateSettings(disp_count, displays.data(), flags);
    if (status != NVAPI_OK) {
        NvAPI_ShortString err;
        NvAPI_GetErrorMessage(status, err);
        CASPAR_LOG(warning) << L"[vulkan_output] SetSyncStateSettings failed: " << to_wstring(err);
        return false;
    }

    // Set control parameters (sync source)
    NV_GSYNC_CONTROL_PARAMS ctrl{};
    ctrl.version = NV_GSYNC_CONTROL_PARAMS_VER;
    status = NvAPI_GSync_GetControlParameters(target_gsync, &ctrl);
    if (status == NVAPI_OK) {
        ctrl.source = (source == sync_source::house_sync)
                          ? NVAPI_GSYNC_SYNC_SOURCE_HOUSESYNC
                          : NVAPI_GSYNC_SYNC_SOURCE_VSYNC;
        status = NvAPI_GSync_SetControlParameters(target_gsync, &ctrl);
        if (status != NVAPI_OK) {
            NvAPI_ShortString err;
            NvAPI_GetErrorMessage(status, err);
            CASPAR_LOG(warning) << L"[vulkan_output] SetControlParameters failed: " << to_wstring(err);
        }
    }

    CASPAR_LOG(info) << L"[vulkan_output] Quadro Sync configured: master display=" << master_display_id
                     << L" source=" << (source == sync_source::house_sync ? L"house" : L"vsync");
    return true;
}

bool nvapi_helpers::disable_sync()
{
    if (!available_ || !impl_ || impl_->gsync_count == 0)
        return false;

    // Unsync all displays on all GSync devices
    for (NvU32 g = 0; g < impl_->gsync_count; ++g) {
        NvU32 disp_count = 0;
        auto  status = NvAPI_GSync_GetTopology(impl_->gsync_handles[g], nullptr, nullptr, &disp_count, nullptr);
        if (status != NVAPI_OK || disp_count == 0)
            continue;

        std::vector<NV_GSYNC_DISPLAY> displays(disp_count);
        displays[0].version = NV_GSYNC_DISPLAY_VER;
        NvU32 gc = 0;
        status = NvAPI_GSync_GetTopology(impl_->gsync_handles[g], &gc, nullptr, &disp_count, displays.data());
        if (status != NVAPI_OK)
            continue;

        // Set all to unsynced
        for (NvU32 i = 0; i < disp_count; ++i)
            displays[i].syncState = NVAPI_GSYNC_DISPLAY_SYNC_STATE_UNSYNCED;

        NvU32 flags = 0;
        NvAPI_GSync_SetSyncStateSettings(disp_count, displays.data(), flags);
    }

    CASPAR_LOG(info) << L"[vulkan_output] Quadro Sync disabled for all displays.";
    return true;
}

// ─── EDID Emulation ─────────────────────────────────────────────────────────

namespace {

// Build a minimal 128-byte EDID block for the given resolution and refresh rate.
// This is enough for Windows to enumerate the output as a connected display.
void build_synthetic_edid(uint8_t* edid, uint32_t width, uint32_t height, double refresh_hz)
{
    memset(edid, 0, 128);

    // Header
    edid[0] = 0x00; edid[1] = 0xFF; edid[2] = 0xFF; edid[3] = 0xFF;
    edid[4] = 0xFF; edid[5] = 0xFF; edid[6] = 0xFF; edid[7] = 0x00;

    // Manufacturer ID: "CSP" (CasparCG) — encoded as 3 x 5-bit letters
    // C=3, S=19, P=16 → (3<<10)|(19<<5)|16 = 0x0E70
    edid[8] = 0x0E; edid[9] = 0x70;

    // Product code
    edid[10] = 0x01; edid[11] = 0x00;

    // Serial number
    edid[12] = 0x01; edid[13] = 0x00; edid[14] = 0x00; edid[15] = 0x00;

    // Week/Year of manufacture (week 1, 2024)
    edid[16] = 1; edid[17] = 34; // 2024 = 1990 + 34

    // EDID version 1.4
    edid[18] = 1; edid[19] = 4;

    // Video input: digital, 8-bit color depth, DisplayPort
    edid[20] = 0xA5; // Digital, 8-bit, DP

    // Screen size (cm) — approximate from pixels at 96 DPI
    edid[21] = static_cast<uint8_t>((std::min)(width * 254 / 960 / 10, 255u));  // horizontal cm
    edid[22] = static_cast<uint8_t>((std::min)(height * 254 / 960 / 10, 255u)); // vertical cm

    // Gamma (2.2 = value 120, stored as (gamma*100)-100)
    edid[23] = 120;

    // Feature support: RGB color, preferred timing in DTD1
    edid[24] = 0x0A;

    // Chromaticity (sRGB defaults)
    edid[25] = 0xEE; edid[26] = 0x91; edid[27] = 0xA3; edid[28] = 0x54;
    edid[29] = 0x4C; edid[30] = 0x99; edid[31] = 0x26; edid[32] = 0x0F;
    edid[33] = 0x50; edid[34] = 0x54;

    // Established timings (none)
    edid[35] = 0; edid[36] = 0; edid[37] = 0;

    // Standard timings (unused)
    for (int i = 38; i < 54; i += 2) {
        edid[i] = 0x01; edid[i + 1] = 0x01;
    }

    // ─── Detailed Timing Descriptor #1 (preferred mode) ─────────────────────
    // Calculate timing parameters (CVT-like simplified)
    uint32_t h_active  = width;
    uint32_t v_active  = height;
    uint32_t h_blank   = (width < 1920) ? 160 : 280;
    uint32_t v_blank   = (height < 1080) ? 28 : 45;
    uint32_t h_front   = (width < 1920) ? 48 : 88;
    uint32_t h_sync    = (width < 1920) ? 32 : 44;
    uint32_t v_front   = 4;
    uint32_t v_sync    = 5;
    uint32_t h_total   = h_active + h_blank;
    uint32_t v_total   = v_active + v_blank;

    // Pixel clock in 10 kHz units
    double pixel_clock_mhz = (h_total * v_total * refresh_hz) / 1000000.0;
    uint32_t pixel_clock = static_cast<uint32_t>(pixel_clock_mhz * 100.0 + 0.5); // in 10 kHz

    edid[54] = pixel_clock & 0xFF;
    edid[55] = (pixel_clock >> 8) & 0xFF;

    // H active / H blanking
    edid[56] = h_active & 0xFF;
    edid[57] = h_blank & 0xFF;
    edid[58] = static_cast<uint8_t>(((h_active >> 8) & 0x0F) << 4 | ((h_blank >> 8) & 0x0F));

    // V active / V blanking
    edid[59] = v_active & 0xFF;
    edid[60] = v_blank & 0xFF;
    edid[61] = static_cast<uint8_t>(((v_active >> 8) & 0x0F) << 4 | ((v_blank >> 8) & 0x0F));

    // H front porch / H sync width
    edid[62] = h_front & 0xFF;
    edid[63] = h_sync & 0xFF;
    edid[64] = static_cast<uint8_t>((v_front & 0x0F) << 4 | (v_sync & 0x0F));
    edid[65] = static_cast<uint8_t>(((h_front >> 8) & 0x03) << 6 | ((h_sync >> 8) & 0x03) << 4 |
                                    ((v_front >> 4) & 0x03) << 2 | ((v_sync >> 4) & 0x03));

    // Image size (mm)
    uint32_t h_mm = width * 254 / 960;
    uint32_t v_mm = height * 254 / 960;
    edid[66] = h_mm & 0xFF;
    edid[67] = v_mm & 0xFF;
    edid[68] = static_cast<uint8_t>(((h_mm >> 8) & 0x0F) << 4 | ((v_mm >> 8) & 0x0F));

    // Borders
    edid[69] = 0; edid[70] = 0;

    // Features: non-interlaced, digital separate sync, H+ V+
    edid[71] = 0x1E;

    // ─── Descriptor #2: Monitor Name ────────────────────────────────────────
    edid[72] = 0; edid[73] = 0; edid[74] = 0;
    edid[75] = 0xFC; // Monitor name tag
    edid[76] = 0;
    const char* name = "CasparCG Out";
    for (int i = 0; i < 13; ++i) {
        edid[77 + i] = (i < static_cast<int>(strlen(name))) ? static_cast<uint8_t>(name[i]) : 0x0A;
    }

    // ─── Descriptor #3: Monitor Range Limits ────────────────────────────────
    edid[90] = 0; edid[91] = 0; edid[92] = 0;
    edid[93] = 0xFD; // Range limits tag
    edid[94] = 0;
    edid[95] = 23;   // Min V freq (Hz)
    edid[96] = static_cast<uint8_t>((std::min)(static_cast<uint32_t>(refresh_hz + 1), 255u)); // Max V freq
    edid[97] = 30;   // Min H freq (kHz)
    edid[98] = static_cast<uint8_t>((std::min)(h_total * static_cast<uint32_t>(refresh_hz + 1) / 1000 + 1, 255u)); // Max H freq
    edid[99] = static_cast<uint8_t>((std::min)(static_cast<uint32_t>(pixel_clock_mhz / 10) + 1, 255u)); // Max pixel clock / 10 MHz

    // GTF not supported
    edid[100] = 0x00;
    // Padding
    for (int i = 101; i < 108; ++i) edid[i] = 0x0A;

    // ─── Descriptor #4: Dummy (unused) ──────────────────────────────────────
    edid[108] = 0; edid[109] = 0; edid[110] = 0;
    edid[111] = 0x10; // Dummy descriptor tag
    edid[112] = 0;
    for (int i = 113; i < 126; ++i) edid[i] = 0;

    // Extension count: 0
    edid[126] = 0;

    // Checksum: byte 127 such that all 128 bytes sum to 0 mod 256
    uint8_t sum = 0;
    for (int i = 0; i < 127; ++i) sum += edid[i];
    edid[127] = static_cast<uint8_t>(256 - sum);
}

} // namespace

uint32_t nvapi_helpers::inject_edid(int gpu_index, int output_index, uint32_t width, uint32_t height, double refresh_hz)
{
    if (!available_ || !impl_ || gpu_index >= static_cast<int>(impl_->gpu_count))
        return 0;

    // Get ALL display IDs (including unconnected outputs)
    NvU32 count = 0;
    auto status = NvAPI_GPU_GetAllDisplayIds(impl_->gpus[gpu_index], nullptr, &count);
    if (status != NVAPI_OK || count == 0) {
        CASPAR_LOG(warning) << L"[vulkan_output] EDID emulation: GetAllDisplayIds failed or no outputs.";
        return 0;
    }

    std::vector<NV_GPU_DISPLAYIDS> all_ids(count);
    for (auto& d : all_ids) d.version = NV_GPU_DISPLAYIDS_VER;

    status = NvAPI_GPU_GetAllDisplayIds(impl_->gpus[gpu_index], all_ids.data(), &count);
    if (status != NVAPI_OK) {
        CASPAR_LOG(warning) << L"[vulkan_output] EDID emulation: GetAllDisplayIds enumeration failed.";
        return 0;
    }

    // Find the target output by sequential index (1-based).
    // We count ALL physical connectors (connected or not).
    int current = 0;
    NvU32 target_display_id = 0;
    for (NvU32 i = 0; i < count; ++i) {
        // Skip MST topology entries (virtual sinks)
        if (all_ids[i].isDynamic)
            continue;

        current++;
        if (current == output_index) {
            target_display_id = all_ids[i].displayId;

            if (all_ids[i].isConnected) {
                CASPAR_LOG(info) << L"[vulkan_output] EDID emulation: Output " << output_index
                                 << L" already connected (displayId=" << target_display_id
                                 << L"), skipping injection.";
                return target_display_id; // Already connected, nothing to do
            }
            break;
        }
    }

    if (target_display_id == 0) {
        CASPAR_LOG(warning) << L"[vulkan_output] EDID emulation: Output " << output_index
                            << L" not found (GPU has " << current << L" outputs).";
        return 0;
    }

    // Build synthetic EDID
    uint8_t edid_data[128];
    build_synthetic_edid(edid_data, width, height, refresh_hz);

    // Inject via NvAPI
    NV_EDID nv_edid{};
    nv_edid.version    = NV_EDID_VER;
    nv_edid.sizeofEDID = 128;
    memcpy(nv_edid.EDID_Data, edid_data, 128);

    status = NvAPI_GPU_SetEDID(impl_->gpus[gpu_index], target_display_id, &nv_edid);
    if (status != NVAPI_OK) {
        NvAPI_ShortString err;
        NvAPI_GetErrorMessage(status, err);
        CASPAR_LOG(warning) << L"[vulkan_output] EDID emulation: SetEDID failed for output " << output_index
                            << L" (displayId=" << target_display_id << L"): " << to_wstring(err)
                            << L". Requires Administrator privileges and a professional GPU.";
        return 0;
    }

    CASPAR_LOG(info) << L"[vulkan_output] EDID emulation: Injected " << width << L"x" << height
                     << L"@" << static_cast<int>(refresh_hz) << L"Hz on output " << output_index
                     << L" (displayId=" << target_display_id << L")";

    return target_display_id;
}

bool nvapi_helpers::remove_edid(int gpu_index, uint32_t display_id)
{
    if (!available_ || !impl_ || display_id == 0)
        return false;

    if (gpu_index >= static_cast<int>(impl_->gpu_count))
        return false;

    // Setting sizeofEDID to 0 removes the injected EDID
    NV_EDID nv_edid{};
    nv_edid.version    = NV_EDID_VER;
    nv_edid.sizeofEDID = 0;

    auto status = NvAPI_GPU_SetEDID(impl_->gpus[gpu_index], display_id, &nv_edid);
    if (status != NVAPI_OK) {
        NvAPI_ShortString err;
        NvAPI_GetErrorMessage(status, err);
        CASPAR_LOG(warning) << L"[vulkan_output] EDID emulation: RemoveEDID failed for displayId="
                            << display_id << L": " << to_wstring(err);
        return false;
    }

    CASPAR_LOG(info) << L"[vulkan_output] EDID emulation: Removed EDID from displayId=" << display_id;
    return true;
}

bool nvapi_helpers::persist_edid(int gpu_index, int output_index)
{
    if (!available_ || !impl_)
        return false;

    // Read the real EDID from the currently connected display
    auto info = read_edid(gpu_index, output_index);
    if (info.raw_edid.empty()) {
        CASPAR_LOG(warning) << L"[vulkan_output] persist_edid: No EDID available for gpu="
                            << gpu_index << L" output=" << output_index
                            << L" (is a monitor connected?)";
        return false;
    }

    // Get the display ID for this output using GetAllDisplayIds (includes disconnected)
    NvU32 count = 0;
    auto status = NvAPI_GPU_GetAllDisplayIds(impl_->gpus[gpu_index], nullptr, &count);
    if (status != NVAPI_OK || count == 0)
        return false;

    std::vector<NV_GPU_DISPLAYIDS> all_ids(count);
    for (auto& d : all_ids)
        d.version = NV_GPU_DISPLAYIDS_VER;

    status = NvAPI_GPU_GetAllDisplayIds(impl_->gpus[gpu_index], all_ids.data(), &count);
    if (status != NVAPI_OK)
        return false;

    int current = 0;
    NvU32 target_display_id = 0;
    for (NvU32 i = 0; i < count; ++i) {
        if (all_ids[i].isDynamic)
            continue;
        current++;
        if (current == output_index) {
            target_display_id = all_ids[i].displayId;
            break;
        }
    }

    if (target_display_id == 0)
        return false;

    // Write the EDID back as a persistent override
    NV_EDID nv_edid{};
    nv_edid.version    = NV_EDID_VER;
    nv_edid.sizeofEDID = static_cast<NvU32>((std::min)(info.raw_edid.size(), size_t(NV_EDID_DATA_SIZE)));
    memcpy(nv_edid.EDID_Data, info.raw_edid.data(), nv_edid.sizeofEDID);

    status = NvAPI_GPU_SetEDID(impl_->gpus[gpu_index], target_display_id, &nv_edid);
    if (status != NVAPI_OK) {
        NvAPI_ShortString err;
        NvAPI_GetErrorMessage(status, err);
        CASPAR_LOG(warning) << L"[vulkan_output] persist_edid: SetEDID failed: " << to_wstring(err);
        return false;
    }

    CASPAR_LOG(info) << L"[vulkan_output] persist_edid: Locked EDID for output " << output_index
                     << L" (" << info.manufacturer << L" " << info.model
                     << L" " << info.max_width << L"x" << info.max_height << L")"
                     << L" - display will remain active even if cable is disconnected.";
    return true;
}

// ─── Dedicated Display ──────────────────────────────────────────────────────

std::vector<nvapi_helpers::dedicated_display_info> nvapi_helpers::get_dedicated_displays()
{
    std::vector<dedicated_display_info> result;
    if (!available_ || !impl_)
        return result;

    NvU32 count = 0;
    auto status = NvAPI_DISP_GetNvManagedDedicatedDisplays(&count, nullptr);
    if (status != NVAPI_OK || count == 0) {
        if (status != NVAPI_OK && status != NVAPI_NO_IMPLEMENTATION) {
            NvAPI_ShortString err;
            NvAPI_GetErrorMessage(status, err);
            CASPAR_LOG(debug) << L"[vulkan_output] GetNvManagedDedicatedDisplays failed: " << to_wstring(err);
        }
        return result;
    }

    std::vector<NV_MANAGED_DEDICATED_DISPLAY_INFO> infos(count);
    for (auto& d : infos)
        d.version = NV_MANAGED_DEDICATED_DISPLAY_INFO_VER;

    status = NvAPI_DISP_GetNvManagedDedicatedDisplays(&count, infos.data());
    if (status != NVAPI_OK) {
        NvAPI_ShortString err;
        NvAPI_GetErrorMessage(status, err);
        CASPAR_LOG(warning) << L"[vulkan_output] GetNvManagedDedicatedDisplays enumerate failed: " << to_wstring(err);
        return result;
    }

    for (NvU32 i = 0; i < count; ++i) {
        dedicated_display_info ddi;
        ddi.display_id  = infos[i].displayId;
        ddi.is_acquired = infos[i].isAcquired != 0;
        ddi.is_mosaic   = infos[i].isMosaic != 0;
        result.push_back(ddi);
    }

    CASPAR_LOG(info) << L"[vulkan_output] Found " << result.size() << L" dedicated display(s).";
    for (const auto& d : result) {
        CASPAR_LOG(info) << L"[vulkan_output]   displayId=" << d.display_id
                         << (d.is_acquired ? L" [acquired by another process]" : L"")
                         << (d.is_mosaic ? L" [mosaic]" : L"");
    }

    return result;
}

uint64_t nvapi_helpers::acquire_dedicated_display(uint32_t display_id)
{
    if (!available_ || !impl_)
        return 0;

    NvU64 source_handle = 0;
    auto status = NvAPI_DISP_AcquireDedicatedDisplay(display_id, &source_handle);
    if (status != NVAPI_OK) {
        NvAPI_ShortString err;
        NvAPI_GetErrorMessage(status, err);
        CASPAR_LOG(warning) << L"[vulkan_output] AcquireDedicatedDisplay failed for displayId="
                            << display_id << L": " << to_wstring(err);
        return 0;
    }

    CASPAR_LOG(info) << L"[vulkan_output] Acquired dedicated display: displayId=" << display_id
                     << L" sourceHandle=0x" << std::hex << source_handle << std::dec;
    return source_handle;
}

bool nvapi_helpers::release_dedicated_display(uint32_t display_id)
{
    if (!available_ || !impl_)
        return false;

    auto status = NvAPI_DISP_ReleaseDedicatedDisplay(display_id);
    if (status != NVAPI_OK) {
        NvAPI_ShortString err;
        NvAPI_GetErrorMessage(status, err);
        CASPAR_LOG(warning) << L"[vulkan_output] ReleaseDedicatedDisplay failed for displayId="
                            << display_id << L": " << to_wstring(err);
        return false;
    }

    CASPAR_LOG(info) << L"[vulkan_output] Released dedicated display: displayId=" << display_id;
    return true;
}

uint32_t nvapi_helpers::acquire_dedicated_display_by_output(int gpu_index, int output_index)
{
    if (!available_ || !impl_ || gpu_index >= static_cast<int>(impl_->gpu_count))
        return 0;

    // Get all display IDs for this GPU (connected outputs)
    NvU32 disp_count = 0;
    auto status = NvAPI_GPU_GetConnectedDisplayIds(impl_->gpus[gpu_index], nullptr, &disp_count, 0);
    if (status != NVAPI_OK || disp_count == 0)
        return 0;

    std::vector<NV_GPU_DISPLAYIDS> display_ids(disp_count);
    for (auto& d : display_ids)
        d.version = NV_GPU_DISPLAYIDS_VER;

    status = NvAPI_GPU_GetConnectedDisplayIds(impl_->gpus[gpu_index], display_ids.data(), &disp_count, 0);
    if (status != NVAPI_OK)
        return 0;

    // Map 1-based output_index to NvAPI displayId
    int target_idx = output_index - 1;
    if (target_idx < 0 || target_idx >= static_cast<int>(disp_count))
        return 0;

    NvU32 target_display_id = display_ids[target_idx].displayId;

    // Check if this display is in the dedicated display list
    auto dedicated = get_dedicated_displays();
    bool found = false;
    for (const auto& d : dedicated) {
        if (d.display_id == target_display_id) {
            if (d.is_acquired) {
                CASPAR_LOG(warning) << L"[vulkan_output] Dedicated display " << target_display_id
                                    << L" already acquired by another process.";
                return 0;
            }
            found = true;
            break;
        }
    }

    if (!found) {
        CASPAR_LOG(info) << L"[vulkan_output] Output " << output_index << L" (displayId="
                         << target_display_id << L") is not a dedicated display. "
                         << L"Configure it as dedicated in NVIDIA Control Panel for crash-safe blanking.";
        return 0;
    }

    // Acquire it
    auto handle = acquire_dedicated_display(target_display_id);
    if (handle == 0)
        return 0;

    return target_display_id;
}

// ─── Display ID Resolution ──────────────────────────────────────────────────

uint32_t nvapi_helpers::resolve_display_id(int gpu_index, int output_index)
{
    if (!available_ || !impl_ || gpu_index >= static_cast<int>(impl_->gpu_count))
        return 0;

    NvU32 disp_id_count = 0;
    auto st = NvAPI_GPU_GetConnectedDisplayIds(impl_->gpus[gpu_index], nullptr, &disp_id_count, 0);
    if (st != NVAPI_OK || disp_id_count == 0)
        return 0;

    std::vector<NV_GPU_DISPLAYIDS> display_ids(disp_id_count);
    for (auto& d : display_ids)
        d.version = NV_GPU_DISPLAYIDS_VER;

    st = NvAPI_GPU_GetConnectedDisplayIds(impl_->gpus[gpu_index], display_ids.data(), &disp_id_count, 0);
    if (st != NVAPI_OK)
        return 0;

    int target_idx = output_index - 1;
    if (target_idx < 0 || target_idx >= static_cast<int>(disp_id_count))
        return 0;

    return display_ids[target_idx].displayId;
}

// ─── Hardware HDR (Display Engine) ──────────────────────────────────────────

bool nvapi_helpers::supports_hdr_output(uint32_t display_id)
{
    if (!available_ || display_id == 0)
        return false;

    NV_HDR_CAPABILITIES hdr_caps{};
    hdr_caps.version = NV_HDR_CAPABILITIES_VER;
    auto st = NvAPI_Disp_GetHdrCapabilities(display_id, &hdr_caps);
    if (st != NVAPI_OK)
        return false;

    return hdr_caps.isST2084EotfSupported != 0;
}

bool nvapi_helpers::enable_hdr_output(uint32_t display_id, int max_cll, int max_fall)
{
    if (!available_ || display_id == 0)
        return false;

    // First check capabilities
    NV_HDR_CAPABILITIES hdr_caps{};
    hdr_caps.version = NV_HDR_CAPABILITIES_VER;
    auto st = NvAPI_Disp_GetHdrCapabilities(display_id, &hdr_caps);
    if (st != NVAPI_OK) {
        NvAPI_ShortString err;
        NvAPI_GetErrorMessage(st, err);
        CASPAR_LOG(warning) << L"[vulkan_output] GetHdrCapabilities failed: " << to_wstring(err);
        return false;
    }

    if (!hdr_caps.isST2084EotfSupported) {
        CASPAR_LOG(warning) << L"[vulkan_output] Display does not support ST2084 (PQ) HDR.";
        return false;
    }

    CASPAR_LOG(info) << L"[vulkan_output] Display HDR caps: ST2084="
                     << hdr_caps.isST2084EotfSupported
                     << L" EDR=" << hdr_caps.isEdrSupported
                     << L" maxLum=" << hdr_caps.display_data.desired_content_max_luminance
                     << L" minLum=" << hdr_caps.display_data.desired_content_min_luminance;

    // Enable UHDA HDR mode (ST2084 PQ + BT.2020)
    // Source: scRGB FP16 linear (sRGB primaries, RGB(1,1,1) = 80 nits)
    // Output: HDR10 (PQ EOTF, BT.2020 primaries, 10-bit)
    // The display engine hardware performs PQ encoding + gamut mapping.
    NV_HDR_COLOR_DATA hdr_data{};
    hdr_data.version                    = NV_HDR_COLOR_DATA_VER;
    hdr_data.cmd                        = NV_HDR_CMD_SET;
    hdr_data.hdrMode                    = NV_HDR_MODE_UHDA;
    hdr_data.static_metadata_descriptor_id = NV_STATIC_METADATA_TYPE_1;

    // Mastering display metadata (BT.2020 primaries, D65 white)
    hdr_data.mastering_display_data.displayPrimary_x0 = 35400; // Red:   0.708
    hdr_data.mastering_display_data.displayPrimary_y0 = 14600; // Red:   0.292
    hdr_data.mastering_display_data.displayPrimary_x1 =  8500; // Green: 0.170
    hdr_data.mastering_display_data.displayPrimary_y1 = 39850; // Green: 0.797
    hdr_data.mastering_display_data.displayPrimary_x2 =  6550; // Blue:  0.131
    hdr_data.mastering_display_data.displayPrimary_y2 =  2300; // Blue:  0.046
    hdr_data.mastering_display_data.displayWhitePoint_x = 15635; // D65: 0.3127
    hdr_data.mastering_display_data.displayWhitePoint_y = 16450; // D65: 0.3290
    hdr_data.mastering_display_data.max_display_mastering_luminance =
        static_cast<NvU16>((std::min)(max_cll, 65535));
    hdr_data.mastering_display_data.min_display_mastering_luminance = 1; // 0.0001 cd/m²
    hdr_data.mastering_display_data.max_content_light_level =
        static_cast<NvU16>((std::min)(max_cll, 65535));
    hdr_data.mastering_display_data.max_frame_average_light_level =
        static_cast<NvU16>((std::min)(max_fall, 65535));

    // Let driver choose optimal wire format
    hdr_data.hdrColorFormat  = NV_COLOR_FORMAT_AUTO;
    hdr_data.hdrDynamicRange = NV_DYNAMIC_RANGE_AUTO;
    hdr_data.hdrBpc          = NV_BPC_10;

    st = NvAPI_Disp_HdrColorControl(display_id, &hdr_data);
    if (st != NVAPI_OK) {
        NvAPI_ShortString err;
        NvAPI_GetErrorMessage(st, err);
        CASPAR_LOG(warning) << L"[vulkan_output] HdrColorControl SET failed: " << to_wstring(err);
        return false;
    }

    CASPAR_LOG(info) << L"[vulkan_output] Hardware HDR enabled (NvAPI UHDA mode). "
                     << L"Display engine performs PQ + BT.2020 conversion. "
                     << L"MaxCLL=" << max_cll << L" MaxFALL=" << max_fall;
    return true;
}

bool nvapi_helpers::disable_hdr_output(uint32_t display_id)
{
    if (!available_ || display_id == 0)
        return false;

    NV_HDR_COLOR_DATA hdr_data{};
    hdr_data.version = NV_HDR_COLOR_DATA_VER;
    hdr_data.cmd     = NV_HDR_CMD_SET;
    hdr_data.hdrMode = NV_HDR_MODE_OFF;

    auto st = NvAPI_Disp_HdrColorControl(display_id, &hdr_data);
    if (st != NVAPI_OK) {
        NvAPI_ShortString err;
        NvAPI_GetErrorMessage(st, err);
        CASPAR_LOG(warning) << L"[vulkan_output] HdrColorControl OFF failed: " << to_wstring(err);
        return false;
    }

    CASPAR_LOG(info) << L"[vulkan_output] Hardware HDR disabled (SDR mode restored).";
    return true;
}

}} // namespace caspar::vulkan_output

#else // !CASPAR_NVAPI_ENABLED

namespace caspar { namespace vulkan_output {

nvapi_helpers::nvapi_helpers()
{
    CASPAR_LOG(info) << L"[vulkan_output] NvAPI not available (compiled without CASPAR_NVAPI_ENABLED).";
}

nvapi_helpers::~nvapi_helpers() = default;

edid_info nvapi_helpers::read_edid(int, int) { return {}; }

gsync_status nvapi_helpers::get_sync_status(int) { return {}; }

bool nvapi_helpers::configure_sync(int, int, sync_source) { return false; }

bool nvapi_helpers::disable_sync() { return false; }

uint32_t nvapi_helpers::inject_edid(int, int, uint32_t, uint32_t, double) { return 0; }

bool nvapi_helpers::remove_edid(int, uint32_t) { return false; }

bool nvapi_helpers::persist_edid(int, int) { return false; }

std::vector<nvapi_helpers::dedicated_display_info> nvapi_helpers::get_dedicated_displays() { return {}; }

uint64_t nvapi_helpers::acquire_dedicated_display(uint32_t) { return 0; }

bool nvapi_helpers::release_dedicated_display(uint32_t) { return false; }

uint32_t nvapi_helpers::acquire_dedicated_display_by_output(int, int) { return 0; }

uint32_t nvapi_helpers::resolve_display_id(int, int) { return 0; }

bool nvapi_helpers::supports_hdr_output(uint32_t) { return false; }

bool nvapi_helpers::enable_hdr_output(uint32_t, int, int) { return false; }

bool nvapi_helpers::disable_hdr_output(uint32_t) { return false; }

}} // namespace caspar::vulkan_output

#endif
