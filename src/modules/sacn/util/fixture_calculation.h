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

// The fixture geometry and rasteriser used to live here, duplicated almost verbatim
// from artnet/util/fixture_calculation.h. They now live in modules/dmx_common; this
// header just re-exports them under caspar::sacn so the consumer reads the same.
#include <modules/dmx_common/fixture_geometry.h>

namespace caspar { namespace sacn {

using dmx::box;
using dmx::color;
using dmx::computed_fixture;
using dmx::fixture;
using dmx::FixtureType;
using dmx::point;
using dmx::rect;

using dmx::average_color;
using dmx::compute_rect;

}} // namespace caspar::sacn
