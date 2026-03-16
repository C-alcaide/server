#pragma once

#include <common/memory.h>
#include <core/fwd.h>
#include <string>
#include <vector>

namespace caspar { namespace spout {

spl::shared_ptr<core::frame_consumer> create_spout_consumer(
    const std::vector<std::wstring>&                         params,
    const core::video_format_repository&                     format_repository,
    const std::vector<spl::shared_ptr<core::video_channel>>& channels,
    const core::channel_info&                                channel_info);

}} // namespace caspar::spout
