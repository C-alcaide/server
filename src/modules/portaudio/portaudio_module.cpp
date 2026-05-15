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

#include "portaudio_module.h"

#include "consumer/portaudio_consumer.h"
#include "producer/portaudio_producer.h"
#include "util/portaudio_device.h"

#include <core/consumer/frame_consumer.h>
#include <core/producer/frame_producer.h>

#include <common/log.h>

namespace caspar { namespace portaudio {

void init(const core::module_dependencies& dependencies)
{
    portaudio_device_manager::instance().initialize();

    dependencies.consumer_registry->register_consumer_factory(L"PortAudio Consumer", create_consumer);
    dependencies.consumer_registry->register_preconfigured_consumer_factory(L"portaudio",
                                                                            create_preconfigured_consumer);

    dependencies.producer_registry->register_producer_factory(L"portaudio", create_producer);
}

void uninit()
{
    portaudio_device_manager::instance().shutdown();
}

}} // namespace caspar::portaudio
