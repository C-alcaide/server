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

#include "tracking.h"

#include "receiver_manager.h"
#include "tracker_binding.h"
#include "tracking_commands.h"

#include <common/env.h>
#include <common/log.h>
#include <common/utf.h>

#include <boost/algorithm/string.hpp>
#include <boost/property_tree/ptree.hpp>

namespace caspar { namespace tracking {

// ---------------------------------------------------------------------------
// Config loading
//
// Reads the optional <tracking> block from casparcg.config and pre-starts
// any configured receivers so they are ready before AMCP clients connect.
//
// Example config block:
//
//   <tracking>
//     <!-- Pre-launch a FreeD receiver on the standard port. -->
//     <receiver>
//       <protocol>FREED</protocol>
//       <port>6301</port>
//     </receiver>
//     <!-- Pre-launch an OSC receiver on a custom port. -->
//     <receiver>
//       <protocol>OSC</protocol>
//       <port>9100</port>
//     </receiver>
//     <!-- Pre-launch a VRPN connection. -->
//     <receiver>
//       <protocol>VRPN</protocol>
//       <host>Tracker0@192.168.1.50</host>
//     </receiver>
//   </tracking>
//
// Channel/layer bindings are always established at runtime via AMCP:
//   TRACKING 1-1 BIND FREED PORT 6301 CAMERA 1 MODE 360
// ---------------------------------------------------------------------------

static void load_config_receivers()
{
    try {
        auto& pt = caspar::env::properties();

        for (auto& child : pt.get_child(L"configuration.tracking", boost::property_tree::wptree{})) {
            if (!boost::iequals(std::wstring(child.first), L"receiver"))
                continue;

            const auto& rec = child.second;

            std::wstring proto_str = rec.get(L"protocol", L"FREED");
            int          port      = rec.get(L"port",     6301);
            std::wstring host_w    = rec.get(L"host",     L"");
            std::string  host      = caspar::u8(host_w);
            tracking_protocol proto;
            try {
                if      (boost::iequals(proto_str, L"FREED"))      proto = tracking_protocol::freed;
                else if (boost::iequals(proto_str, L"FREED_PLUS")) proto = tracking_protocol::freed_plus;
                else if (boost::iequals(proto_str, L"OSC"))        proto = tracking_protocol::osc;
                else if (boost::iequals(proto_str, L"VRPN"))       proto = tracking_protocol::vrpn;
                else {
                    CASPAR_LOG(warning) << L"[tracking] Unknown protocol in config: " << proto_str;
                    continue;
                }
            } catch (...) {
                continue;
            }

            try {
                receiver_manager::instance().ensure_receiver(proto, port, host);
                CASPAR_LOG(info) << L"[tracking] Started " << proto_str
                                 << L" receiver on port " << port
                                 << (host.empty() ? L"" : L" host " + caspar::u16(host));
            } catch (const std::exception& e) {
                CASPAR_LOG(error) << L"[tracking] Failed to start receiver: " << caspar::u16(e.what());
            }
        }
    } catch (...) {
        // No <tracking> block in config — that's fine
    }
}

// ---------------------------------------------------------------------------

void init(const core::module_dependencies& dependencies)
{
    CASPAR_LOG(info) << L"[tracking] Initialising camera-tracking module";
    load_config_receivers();

    if (dependencies.command_repository)
        register_amcp_commands(dependencies.command_repository);
}

void uninit()
{
    CASPAR_LOG(info) << L"[tracking] Shutting down camera-tracking module";
    receiver_manager::instance().stop_all();
}

}} // namespace caspar::tracking
