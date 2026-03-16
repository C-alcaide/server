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
 * This module uses miniaudio (https://miniaud.io), dual-licensed under
 * MIT and public domain (Unlicense), both compatible with GPL-3.
 */

#include "system_audio.h"

#include <core/module_dependencies.h>
#include <core/producer/frame_producer_registry.h>

namespace caspar { namespace system_audio {

void init(const core::module_dependencies& dependencies)
{
    dependencies.producer_registry->register_producer_factory(L"system_audio", create_producer);
}

void uninit()
{
}

}} // namespace caspar::system_audio