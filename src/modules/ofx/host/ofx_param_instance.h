/*
 * Copyright (c) 2011 Sveriges Television AB <info@casparcg.com>
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

#pragma once

#include "ofx_includes.h"

#include <string>

namespace caspar { namespace ofx {

/// Create a parameter instance for the given descriptor. Parameters are backed by simple
/// in-memory storage initialised from the descriptor's default value (Phase 2: no keyframe
/// animation yet). Returns nullptr for unsupported parameter types.
OFX::Host::Param::Instance* create_param_instance(OFX::Host::Param::SetInstance* effect,
                                                  const std::string&             name,
                                                  OFX::Host::Param::Descriptor&  descriptor);

}} // namespace caspar::ofx
