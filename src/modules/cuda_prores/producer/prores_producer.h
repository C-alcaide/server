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
 * ProRes format reference: Apple Inc. "ProRes RAW White Paper" (public documentation).
 */

// prores_producer.h
// CasparCG frame_producer for CUDA-accelerated ProRes decode.
//
// AMCP command:
//   PLAY 1-10 CUDA_PRORES [FILE <path>] [DEVICE <cuda_device_index>]
//
// This producer:
//  - Demuxes raw icpf packets from MOV/MXF/MKV using libavformat (no avcodec).
//  - Decodes each frame entirely on the GPU via CUDA kernels.
//  - Writes the decoded BGRA16 output directly to an OpenGL texture via
//    CUDA-GL interop (zero host transfers for pixel data).
//  - Returns a core::const_frame with the GL texture pre-populated, so the
//    OGL image mixer can composite it without a PBO upload path.
// ---------------------------------------------------------------------------
#pragma once

#include <common/memory.h>
#include <core/fwd.h>
#include <core/module_dependencies.h>
#include <core/producer/frame_producer.h>

#include <string>
#include <vector>

namespace caspar { namespace cuda_prores {

// Factory function — registered with the producer registry at module init.
spl::shared_ptr<core::frame_producer>
create_prores_producer(const core::frame_producer_dependencies& dependencies,
                       const std::vector<std::wstring>&         params);

// Module-level registration helper (called from cuda_prores.cpp::init()).
void register_prores_producer(const core::module_dependencies& module_deps);

}} // namespace caspar::cuda_prores
