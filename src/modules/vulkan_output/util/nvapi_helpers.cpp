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

#include <nvapi.h>

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

}} // namespace caspar::vulkan_output

#endif
