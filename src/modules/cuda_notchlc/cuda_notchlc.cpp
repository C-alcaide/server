/*
 * Copyright (c) 2025 CasparCG Contributors
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
 *
 * This module requires the NVIDIA CUDA Toolkit (https://developer.nvidia.com/cuda-toolkit).
 * NotchLC is a codec specification by Derivative Inc., available under the
 * Creative Commons Attribution 4.0 International License.
 */

// cuda_notchlc.cpp
// CasparCG module entry point — registered by casparcg_add_module_project(INIT_FUNCTION "cuda_notchlc::init").
#include "cuda_notchlc.h"
#include "producer/notchlc_producer.h"

#include <common/log.h>
#include <cuda_runtime.h>

namespace caspar { namespace cuda_notchlc {

void init(const core::module_dependencies& dependencies)
{
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0) {
        CASPAR_LOG(info) << L"[cuda_notchlc] CUDA devices found:";
        for (int i = 0; i < device_count; i++) {
            cudaDeviceProp p;
            if (cudaGetDeviceProperties(&p, i) == cudaSuccess)
                CASPAR_LOG(info) << L"  [" << i << L"] " << p.name
                                 << L" -- sm_" << p.major << p.minor
                                 << L" -- " << (p.totalGlobalMem / (1024 * 1024)) << L" MB";
        }
    } else {
        CASPAR_LOG(warning) << L"[cuda_notchlc] No CUDA devices found -- producer unavailable";
    }

    register_notchlc_producer(dependencies);

    CASPAR_LOG(info) << L"[cuda_notchlc] Module initialised";
}

}} // namespace caspar::cuda_notchlc
