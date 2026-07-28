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

#include "../StdAfx.h"

#include "dvp_support.h"

#include <common/log.h>

#ifdef DECKLINK_CUDA_DVP_ENABLED
#include <cuda.h>
#include <cuda_runtime.h>

#include "DVPAPI.h"
#include "dvpapi_cuda.h"
#endif

namespace caspar { namespace decklink {

#ifdef DECKLINK_CUDA_DVP_ENABLED

namespace {

// Try dvpInitCUDAContext on the current context; returns true and logs the DVP
// alignment constants on success. Closes the context again (probe only).
bool try_init_current(const wchar_t* who)
{
    // The header notes SHARE_APP_CONTEXT is the only CUDA flag; try plain first, then it.
    DVPStatus st = dvpInitCUDAContext(0);
    if (st != DVP_STATUS_OK)
        st = dvpInitCUDAContext(DVP_DEVICE_FLAGS_SHARE_APP_CONTEXT);
    if (st != DVP_STATUS_OK) {
        CASPAR_LOG(debug) << L"[decklink] DVP init failed on " << who << L" (status=" << static_cast<int>(st) << L")";
        return false;
    }
    uint32_t a = 0, b = 0, c = 0, d = 0, e = 0, f = 0;
    dvpGetRequiredConstantsCUDACtx(&a, &b, &c, &d, &e, &f);
    CASPAR_LOG(debug) << L"[decklink] DVP OK on " << who << L": bufAddrAlign=" << a << L" strideAlign=" << b
                      << L" semAllocSize=" << d;
    dvpCloseCUDAContext();
    return true;
}

bool probe_dvp()
{
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count <= 0)
        return false;

    for (int dev = 0; dev < count; ++dev) {
        cudaDeviceProp prop{};
        cudaGetDeviceProperties(&prop, dev);
        if (cudaSetDevice(dev) != cudaSuccess)
            continue;
        cudaFree(nullptr); // force this device's primary context

        CUcontext ctx = nullptr;
        if (cuCtxGetCurrent(&ctx) != CUDA_SUCCESS || ctx == nullptr)
            continue;

        std::wstring who = L"CUDA device " + std::to_wstring(dev);
        if (try_init_current(who.c_str())) {
            CASPAR_LOG(info) << L"[decklink] DVP available on CUDA device " << dev << L" (" << prop.name << L").";
            return true;
        }
    }
    return false;
}

} // namespace

bool dvp_available()
{
    static const bool available = probe_dvp();
    static const bool logged    = [] {
        CASPAR_LOG(info) << L"[decklink] NVIDIA DVP (GPUDirect for Video): "
                         << (available ? L"available (Tier-2 GPU-direct output enabled)"
                                       : L"not available on this GPU (falling back to pinned copy)");
        return true;
    }();
    (void)logged;
    return available;
}

#else // !DECKLINK_CUDA_DVP_ENABLED

bool dvp_available() { return false; }

#endif

}} // namespace caspar::decklink
