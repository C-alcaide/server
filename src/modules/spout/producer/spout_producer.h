#pragma once

#include <common/memory.h>
#include <core/fwd.h>
#include <string>
#include <vector>

namespace caspar { namespace spout {

spl::shared_ptr<core::frame_producer> create_spout_producer(
    const core::frame_producer_dependencies& dependencies,
    const std::vector<std::wstring>&         params);

}} // namespace caspar::spout
