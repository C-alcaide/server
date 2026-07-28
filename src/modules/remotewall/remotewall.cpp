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

// remotewall.cpp — CasparCG module entry point (registered via
// casparcg_add_module_project(INIT_FUNCTION "remotewall::init")).
#include "remotewall.h"

#include "producer/remotewall_producer.h"

#include <common/env.h>
#include <common/log.h>

#include <boost/property_tree/ptree.hpp>

namespace caspar { namespace remotewall {

void init(const core::module_dependencies& dependencies)
{
    // Optional defaults from <configuration><remotewall> in casparcg.config.
    try {
        if (auto rw = env::properties().get_child_optional(L"configuration.remotewall")) {
            config().listen_port   = rw->get(L"listen-port", config().listen_port);
            config().buffer_frames = rw->get(L"buffer-frames", config().buffer_frames);
            config().cuda_device   = rw->get(L"cuda-device", config().cuda_device);
        }
    } catch (...) {
        CASPAR_LOG(warning) << L"[remotewall] Could not read <remotewall> config; using defaults.";
    }

    register_remotewall_producer(dependencies);

    CASPAR_LOG(info) << L"[remotewall] Module initialised (default port " << config().listen_port << L").";
}

}} // namespace caspar::remotewall
