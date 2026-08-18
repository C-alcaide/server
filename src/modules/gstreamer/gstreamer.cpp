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

#include "StdAfx.h"

#include "gstreamer.h"

#include "consumer/gst_consumer.h"
#include "producer/gst_producer.h"
#include "util/gst_runtime.h"

#include <core/consumer/frame_consumer.h>
#include <core/producer/frame_producer.h>

#include <common/env.h>
#include <common/log.h>

#include <boost/property_tree/ptree.hpp>

namespace caspar { namespace gstreamer {

void init(const core::module_dependencies& dependencies)
{
    dependencies.producer_registry->register_producer_factory(L"GStreamer Producer", create_producer);
    dependencies.consumer_registry->register_consumer_factory(L"GStreamer Consumer", create_consumer);

    // GStreamer is loaded on first use, so a server with no installation starts normally and
    // only a PLAY that asks for it fails. auto-load turns that into a startup diagnostic,
    // which is what you want on a machine that is supposed to have it.
    if (env::properties().get(L"configuration.gstreamer.auto-load", false)) {
        try {
            runtime::ensure_initialized();
        } catch (...) {
            CASPAR_LOG_CURRENT_EXCEPTION();
        }
    }
}

}} // namespace caspar::gstreamer
