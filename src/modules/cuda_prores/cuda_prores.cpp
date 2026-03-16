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
 * The CUDA Toolkit is proprietary software by NVIDIA Corporation, free to use
 * for application development. It must be installed separately by the end user.
 * ProRes format reference: Apple Inc. "ProRes RAW White Paper" (public documentation).
 */

// cuda_prores.cpp
// CasparCG module entry point — registered in CASPARCG_MODULE_INIT_STATEMENTS
// by casparcg_add_module_project(INIT_FUNCTION "cuda_prores::init").
#include "cuda_prores.h"

#include "consumer/prores_consumer.h"
#include "consumer/prores_bypass_consumer.h"
#include "producer/prores_producer.h"

#include <common/log.h>

#include <cuda_runtime.h>

namespace caspar { namespace cuda_prores {

void init(const core::module_dependencies& dependencies)
{
    // Log available CUDA devices
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0) {
        CASPAR_LOG(info) << L"[cuda_prores] CUDA devices found:";
        for (int i = 0; i < device_count; i++) {
            cudaDeviceProp p;
            if (cudaGetDeviceProperties(&p, i) == cudaSuccess) {
                CASPAR_LOG(info) << L"  [" << i << L"] " << p.name
                                 << L" -- sm_"
                                 << p.major << p.minor
                                 << L" -- "
                                 << (p.totalGlobalMem / (1024 * 1024))
                                 << L" MB";
            }
        }
    } else {
        CASPAR_LOG(warning) << L"[cuda_prores] No CUDA devices found -- consumer will be unavailable";
    }

    // Register AMCP consumer (responds to: ADD 1-10 CUDA_PRORES ...)
    dependencies.consumer_registry->register_consumer_factory(
        L"CUDA_PRORES Consumer", create_consumer);

    // Register XML preconfigured consumer (<cuda-prores> element in casparcg.config)
    dependencies.consumer_registry->register_preconfigured_consumer_factory(
        L"cuda-prores", create_preconfigured_consumer);

    // Register bypass consumer (ADD 1 CUDA_PRORES_BYPASS DEVICE 1 PATH ...)
    dependencies.consumer_registry->register_consumer_factory(
        L"CUDA_PRORES_BYPASS Consumer", create_bypass_consumer);
    dependencies.consumer_registry->register_preconfigured_consumer_factory(
        L"cuda-prores-bypass", create_preconfigured_bypass_consumer);

    // Register ProRes decoder producer (PLAY 1-1 CUDA_PRORES FILE x.mov)
    register_prores_producer(dependencies);

    CASPAR_LOG(info) << L"[cuda_prores] Module initialised";
}

}} // namespace caspar::cuda_prores
