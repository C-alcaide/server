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

// remotewall_producer.h — native cloudXR tile-wall producer (PLAY 1-10 REMOTEWALL ...).
#pragma once

#include <core/fwd.h>
#include <core/module_dependencies.h>

#include <common/memory.h>

#include <string>
#include <vector>

namespace caspar { namespace remotewall {

/// Module-wide defaults, populated from <configuration><remotewall> in casparcg.config.
struct module_config
{
    int listen_port   = 9000; ///< default UDP port the receiver binds
    int buffer_frames = 4;    ///< receiver reorder/queue depth hint
    int cuda_device   = 0;    ///< CUDA device index for decode/convert
};

/// Process-wide module config accessor.
module_config& config();

/// AMCP producer factory: PLAY <ch-layer> REMOTEWALL [PORT <n>] [TILES <n>] [CODEC hevc|h264] ...
spl::shared_ptr<core::frame_producer> create_producer(const core::frame_producer_dependencies& dependencies,
                                                      const std::vector<std::wstring>&          params);

void register_remotewall_producer(const core::module_dependencies& dependencies);

}} // namespace caspar::remotewall
