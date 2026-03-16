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

// notchlc_producer.h
// CasparCG frame_producer for CUDA-accelerated NotchLC decode.
//
// AMCP command:
//   PLAY 1-10 CUDA_NOTCHLC [FILE <path>] [DEVICE <cuda_device_index>]
//              [LOOP] [SEEK|IN <frame>] [OUT <frame>] [LENGTH <frames>]
//              [COLOR_MATRIX {709|601|2020|LINEAR|AUTO}]
//
// This producer:
//  - Demuxes raw NotchLC packets from MOV/MKV/AVI/MP4 using libavformat.
//  - Decompresses LZ4 payload on the GPU via nvcomp.
//  - Decodes the YCoCg-RT pixel data via CUDA kernels (Y/UV/Alpha planes).
//  - Converts to BGRA16 on the GPU and writes directly to an OpenGL texture
//    via CUDA-GL interop (zero host transfers on the zero-copy path).
// ---------------------------------------------------------------------------
#pragma once

#include <common/memory.h>
#include <core/fwd.h>
#include <core/module_dependencies.h>
#include <core/producer/frame_producer.h>

#include <string>
#include <vector>

namespace caspar { namespace cuda_notchlc {

// Factory function — registered with the producer registry at module init.
spl::shared_ptr<core::frame_producer>
create_notchlc_producer(const core::frame_producer_dependencies& dependencies,
                        const std::vector<std::wstring>&         params);

// Module-level registration helper (called from cuda_notchlc.cpp::init()).
void register_notchlc_producer(const core::module_dependencies& module_deps);

}} // namespace caspar::cuda_notchlc
