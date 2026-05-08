// Utility to remove persistent EDID overrides injected via NvAPI_GPU_SetEDID.
// Build: cl /EHsc /I"d:\Github\nvapi-main" clear_edid.cpp /link "d:\Github\nvapi-main\amd64\nvapi64.lib"
// Run as Administrator.

#include <cstdio>
#include <nvapi.h>

int main()
{
    if (NvAPI_Initialize() != NVAPI_OK) {
        printf("NvAPI_Initialize failed\n");
        return 1;
    }

    NvPhysicalGpuHandle gpus[NVAPI_MAX_PHYSICAL_GPUS] = {};
    NvU32 gpu_count = 0;
    NvAPI_EnumPhysicalGPUs(gpus, &gpu_count);
    printf("Found %u GPU(s)\n", gpu_count);

    for (NvU32 g = 0; g < gpu_count; ++g) {
        NvU32 count = 0;
        NvAPI_GPU_GetAllDisplayIds(gpus[g], nullptr, &count);
        if (count == 0) continue;

        NV_GPU_DISPLAYIDS* ids = new NV_GPU_DISPLAYIDS[count];
        for (NvU32 i = 0; i < count; ++i) ids[i].version = NV_GPU_DISPLAYIDS_VER;
        NvAPI_GPU_GetAllDisplayIds(gpus[g], ids, &count);

        printf("GPU %u: %u display outputs\n", g, count);
        for (NvU32 i = 0; i < count; ++i) {
            if (ids[i].isDynamic) continue;

            // Try to remove EDID override (setting sizeofEDID=0)
            NV_EDID nv_edid = {};
            nv_edid.version = NV_EDID_VER;
            nv_edid.sizeofEDID = 0;

            auto status = NvAPI_GPU_SetEDID(gpus[g], ids[i].displayId, &nv_edid);
            if (status == NVAPI_OK) {
                printf("  Output %u (displayId=%u): EDID override REMOVED %s\n",
                       i, ids[i].displayId,
                       ids[i].isConnected ? "(was connected)" : "(was disconnected)");
            } else {
                NvAPI_ShortString err;
                NvAPI_GetErrorMessage(status, err);
                printf("  Output %u (displayId=%u): %s (connected=%d)\n",
                       i, ids[i].displayId, err, ids[i].isConnected);
            }
        }
        delete[] ids;
    }

    NvAPI_Unload();
    printf("\nDone. Restart may be needed for changes to take effect.\n");
    return 0;
}
