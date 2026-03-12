// prores_consumer.h
// CasparCG frame_consumer implementation for the CUDA ProRes recording pipeline.
//
// AMCP command
// ─────────────────────────────────────────────────────────────────────────────
//   ADD 1-10 CUDA_PRORES [PATH <path>] [PROFILE <0-4>] [CODEC MOV|MXF]
//
//   PATH    output directory (default: cwd)
//   PROFILE 0=Proxy, 1=LT, 2=Standard, 3=HQ (default), 4=4444
//   CODEC   container format (default: MOV)
//
// XML preconfigured consumer (casparcg.config)
// ─────────────────────────────────────────────────────────────────────────────
//   <consumers>
//     <cuda-prores>
//       <path>D:\Recordings</path>
//       <filename>ch1_%(date)s_%(time)s.mov</filename>
//       <profile>3</profile>
//       <codec>mov</codec>
//     </cuda-prores>
//   </consumers>
#pragma once

#include <core/consumer/frame_consumer.h>
#include <core/video_format.h>

#include <boost/property_tree/ptree_fwd.hpp>
#include <common/memory.h>

#include <vector>
#include <string>

namespace caspar { namespace cuda_prores {

// Factory functions registered with the consumer registry.
spl::shared_ptr<core::frame_consumer>
create_consumer(const std::vector<std::wstring>&             params,
                const core::video_format_repository&         format_repository,
                const std::vector<spl::shared_ptr<core::video_channel>>& channels,
                const core::channel_info&                    channel_info);

spl::shared_ptr<core::frame_consumer>
create_preconfigured_consumer(const boost::property_tree::wptree&          element,
                              const core::video_format_repository&         format_repository,
                              const std::vector<spl::shared_ptr<core::video_channel>>& channels,
                              const core::channel_info&                    channel_info);

}} // namespace caspar::cuda_prores
