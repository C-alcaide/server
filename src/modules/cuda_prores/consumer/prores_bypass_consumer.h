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
