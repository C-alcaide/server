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

// prores_bypass_consumer.h
// CasparCG frame_consumer that drives ProRes recording directly from a
// DeckLink SDI input, bypassing the CasparCG GPU mixer entirely.
//
// AMCP usage:
//   ADD 1 CUDA_PRORES_BYPASS DEVICE 1 PATH "d:\clips" [PROFILE 3] [CODEC MOV|MXF]
//
// The consumer attaches to the channel only to receive format_desc via
// initialize(); it ignores all frames delivered by send().  Recording is
// driven by VideoInputFrameArrived callbacks from DecklinkCapture.
#pragma once

#include <core/consumer/frame_consumer.h>
#include <core/video_format.h>

#include <boost/property_tree/ptree_fwd.hpp>
#include <common/memory.h>

#include <vector>
#include <string>

namespace caspar { namespace cuda_prores {

// ─── Factory functions (registered in cuda_prores.cpp) ───────────────────────

/// Responds to: ADD 1 CUDA_PRORES_BYPASS DEVICE 1 PATH "d:\clips" ...
spl::shared_ptr<core::frame_consumer>
create_bypass_consumer(const std::vector<std::wstring>&                          params,
                       const core::video_format_repository&                     format_repository,
                       const std::vector<spl::shared_ptr<core::video_channel>>& channels,
                       const core::channel_info&                                channel_info);

/// Responds to <cuda-prores-bypass> XML elements in casparcg.config
spl::shared_ptr<core::frame_consumer>
create_preconfigured_bypass_consumer(const boost::property_tree::wptree&                       element,
                                     const core::video_format_repository&                     format_repository,
                                     const std::vector<spl::shared_ptr<core::video_channel>>& channels,
                                     const core::channel_info&                                channel_info);

}} // namespace caspar::cuda_prores
