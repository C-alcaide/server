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
 *
 * This module ingests the cloudXR tile-wall stream (UDP/RTP+FEC HEVC tiles with
 * in-band SyncMeta) as a native CasparCG producer. It requires the NVIDIA CUDA
 * Toolkit and, for decode, the NVIDIA Video Codec SDK — both proprietary NVIDIA
 * software installed separately by the end user.
 */

// remotewall.h — CasparCG module entry point for the native tile-wall producer.
#pragma once

#include <core/module_dependencies.h>

namespace caspar { namespace remotewall {

void init(const core::module_dependencies& dependencies);

}} // namespace caspar::remotewall
