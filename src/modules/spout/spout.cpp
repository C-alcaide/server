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
 * This module uses the Spout SDK (https://github.com/leadedge/Spout2),
 * licensed under BSD 2-Clause, which is compatible with GPL-3.
 */

#include "spout.h"
#include <core/module_dependencies.h>
#include "consumer/spout_consumer.h"
#include "producer/spout_producer.h"

namespace caspar { namespace spout {

void init(const core::module_dependencies& dependencies)
{
    dependencies.producer_registry->register_producer_factory(L"SPOUT", create_spout_producer);
    dependencies.consumer_registry->register_consumer_factory(L"SPOUT", create_spout_consumer);
}

}}
