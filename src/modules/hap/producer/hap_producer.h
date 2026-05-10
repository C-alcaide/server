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
 */

// hap_producer.h
// CasparCG frame_producer for GPU-accelerated HAP decode.
//
// AMCP command:
//   PLAY 1-10 HAP [FILE <path>] [LOOP] [SEEK|IN <frame>] [OUT <frame>]
//              [LENGTH <frames>] [SPEED <factor>] [PINGPONG]
//
// This producer:
//  - Demuxes raw HAP packets from MOV/AVI/MP4 using libavformat.
//  - Decompresses Snappy payload on CPU via worker threads.
//  - Uploads raw DXT/BC block data as GL compressed textures.
//  - Runs a lightweight GL render pass to decompress DXT→RGBA via the
//    GPU texture unit (zero compute cost) and writes to a standard
//    uncompressed texture for the mixer.
// ---------------------------------------------------------------------------
#pragma once

#include <common/memory.h>
#include <core/fwd.h>
#include <core/module_dependencies.h>
#include <core/producer/frame_producer.h>

#include <string>
#include <vector>

namespace caspar { namespace hap {

spl::shared_ptr<core::frame_producer>
create_hap_producer(const core::frame_producer_dependencies& deps,
                    const std::vector<std::wstring>& params);

void register_hap_producer(const core::module_dependencies& module_deps);

}} // namespace caspar::hap
