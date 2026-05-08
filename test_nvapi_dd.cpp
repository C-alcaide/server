// Quick NvAPI dedicated display diagnostic
// Build: cl /EHsc /std:c++17 test_nvapi_dd.cpp /Fe:test_nvapi_dd.exe /link D:\Github\nvapi-main\amd64\nvapi64.lib
#include <cstdio>
#include <cstdint>
#include <cstring>
#include "D:\Github\nvapi-main\nvapi.h"

int main()
{
    auto status = NvAPI_Initialize();
    if (status != NVAPI_OK) {
        NvAPI_ShortString err;
        NvAPI_GetErrorMessage(status, err);
        printf("NvAPI_Initialize failed: %s\n", err);
        return 1;
    }
    printf("NvAPI_Initialize: OK\n");

    // Enumerate GPUs
    NvPhysicalGpuHandle gpus[NVAPI_MAX_PHYSICAL_GPUS] = {};
    NvU32 gpu_count = 0;
    NvAPI_EnumPhysicalGPUs(gpus, &gpu_count);
    printf("Physical GPUs: %u\n", gpu_count);

    for (NvU32 i = 0; i < gpu_count; ++i) {
        NvAPI_ShortString name;
        NvAPI_GPU_GetFullName(gpus[i], name);
        printf("  GPU %u: %s\n", i, name);
    }

    // Try NvAPI_DISP_GetNvManagedDedicatedDisplays
    NvU32 dd_count = 0;
    NV_MANAGED_DEDICATED_DISPLAY_INFO dd_infos[16] = {};
    
    status = NvAPI_DISP_GetNvManagedDedicatedDisplays(&dd_count, nullptr);
    printf("\nNvAPI_DISP_GetNvManagedDedicatedDisplays (count query): status=%d count=%u\n", status, dd_count);
    if (status != NVAPI_OK) {
        NvAPI_ShortString err;
        NvAPI_GetErrorMessage(status, err);
        printf("  Error: %s\n", err);
    }

    if (status == NVAPI_OK && dd_count > 0) {
        for (NvU32 i = 0; i < dd_count && i < 16; ++i)
            dd_infos[i].version = NV_MANAGED_DEDICATED_DISPLAY_INFO_VER;

        status = NvAPI_DISP_GetNvManagedDedicatedDisplays(&dd_count, dd_infos);
        if (status == NVAPI_OK) {
            for (NvU32 i = 0; i < dd_count; ++i) {
                printf("  Dedicated display %u: displayId=%u acquired=%u mosaic=%u\n",
                       i, dd_infos[i].displayId, dd_infos[i].isAcquired, dd_infos[i].isMosaic);
            }
        } else {
            NvAPI_ShortString err;
            NvAPI_GetErrorMessage(status, err);
            printf("  Full query error: %s\n", err);
        }
    }

    // Also enumerate all display IDs to compare
    printf("\nAll connected displays (active, flags=0):\n");
    for (NvU32 g = 0; g < gpu_count; ++g) {
        NvU32 display_count = 0;
        NV_GPU_DISPLAYIDS display_ids[32] = {};
        for (auto& d : display_ids)
            d.version = NV_GPU_DISPLAYIDS_VER;

        display_count = 32;
        status = NvAPI_GPU_GetConnectedDisplayIds(gpus[g], display_ids, &display_count, 0);
        if (status == NVAPI_OK) {
            for (NvU32 d = 0; d < display_count; ++d) {
                printf("  GPU %u display %u: displayId=%u connector=%u dynamic=%u multi=%u active=%u\n",
                       g, d, display_ids[d].displayId, display_ids[d].connectorType,
                       display_ids[d].isDynamic, display_ids[d].isMultiStreamRootNode,
                       display_ids[d].isActive);
            }
        }
    }

    // Enumerate ALL connected displays including inactive (uncached)
    printf("\nAll connected displays (uncached, flags=NV_GPU_CONNECTED_IDS_FLAG_UNCACHED):\n");
    for (NvU32 g = 0; g < gpu_count; ++g) {
        NvU32 display_count = 0;
        NV_GPU_DISPLAYIDS display_ids[32] = {};
        for (auto& d : display_ids)
            d.version = NV_GPU_DISPLAYIDS_VER;

        display_count = 32;
        status = NvAPI_GPU_GetConnectedDisplayIds(gpus[g], display_ids, &display_count, NV_GPU_CONNECTED_IDS_FLAG_UNCACHED);
        if (status == NVAPI_OK) {
            for (NvU32 d = 0; d < display_count; ++d) {
                printf("  GPU %u display %u: displayId=%u connector=%u dynamic=%u multi=%u active=%u\n",
                       g, d, display_ids[d].displayId, display_ids[d].connectorType,
                       display_ids[d].isDynamic, display_ids[d].isMultiStreamRootNode,
                       display_ids[d].isActive);

                // Try direct acquire on inactive displays
                if (!display_ids[d].isActive) {
                    NvU64 handle = 0;
                    auto acq_status = NvAPI_DISP_AcquireDedicatedDisplay(display_ids[d].displayId, &handle);
                    NvAPI_ShortString acq_err;
                    NvAPI_GetErrorMessage(acq_status, acq_err);
                    printf("    -> AcquireDedicatedDisplay: status=%d (%s)\n", acq_status, acq_err);
                    if (acq_status == NVAPI_OK) {
                        printf("    -> SUCCESS! handle=0x%llx. Releasing...\n", handle);
                        NvAPI_DISP_ReleaseDedicatedDisplay(display_ids[d].displayId);
                    }
                }
            }
        } else {
            NvAPI_ShortString err;
            NvAPI_GetErrorMessage(status, err);
            printf("  Error: %s\n", err);
        }
    }

    // Also try GetAllDisplayIds
    printf("\nNvAPI_GPU_GetAllDisplayIds:\n");
    for (NvU32 g = 0; g < gpu_count; ++g) {
        NvU32 display_count = 0;
        // First call: get count
        status = NvAPI_GPU_GetAllDisplayIds(gpus[g], nullptr, &display_count);
        if (status == NVAPI_OK && display_count > 0) {
            NV_GPU_DISPLAYIDS* all_ids = new NV_GPU_DISPLAYIDS[display_count];
            for (NvU32 i = 0; i < display_count; ++i)
                all_ids[i].version = NV_GPU_DISPLAYIDS_VER;

            status = NvAPI_GPU_GetAllDisplayIds(gpus[g], all_ids, &display_count);
            if (status == NVAPI_OK) {
                for (NvU32 d = 0; d < display_count; ++d) {
                    printf("  GPU %u display %u: displayId=%u connector=%u dynamic=%u multi=%u active=%u\n",
                           g, d, all_ids[d].displayId, all_ids[d].connectorType,
                           all_ids[d].isDynamic, all_ids[d].isMultiStreamRootNode,
                           all_ids[d].isActive);

                    // Try acquire on all inactive displays
                    if (!all_ids[d].isActive) {
                        NvU64 handle = 0;
                        auto acq_status = NvAPI_DISP_AcquireDedicatedDisplay(all_ids[d].displayId, &handle);
                        NvAPI_ShortString acq_err;
                        NvAPI_GetErrorMessage(acq_status, acq_err);
                        printf("    -> AcquireDedicatedDisplay: status=%d (%s)\n", acq_status, acq_err);
                        if (acq_status == NVAPI_OK) {
                            printf("    -> SUCCESS! handle=0x%llx. Releasing...\n", handle);
                            NvAPI_DISP_ReleaseDedicatedDisplay(all_ids[d].displayId);
                        }
                    }
                }
            }
            delete[] all_ids;
        } else {
            NvAPI_ShortString err;
            NvAPI_GetErrorMessage(status, err);
            printf("  GPU %u: count=%u status=%d (%s)\n", g, display_count, status, err);
        }
    }

    NvAPI_Unload();
    return 0;
}
