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
 * HAP is an open video codec designed for fast, GPU-accelerated playback
 * using S3TC/DXT compressed textures. https://hap.video/
 */

// hap.h
// CasparCG module entry point for the HAP decoder producer.
#pragma once
#include <core/module_dependencies.h>

namespace caspar { namespace hap {

void init(const core::module_dependencies& dependencies);

}} // namespace caspar::hap
