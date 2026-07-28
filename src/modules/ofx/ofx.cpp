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

#include "ofx.h"

#include <common/log.h>

#ifdef CASPAR_OFX_ENABLED
#include "host/ofx_host.h"
#include "producer/ofx_producer.h"

#include <ofxCore.h>
#include <ofxImageEffect.h>

#include <common/env.h>
#include <common/utf.h>

#include <boost/property_tree/ptree.hpp>
#endif

namespace caspar { namespace ofx {

void init(const core::module_dependencies& dependencies)
{
#ifdef CASPAR_OFX_ENABLED
    try {
        auto& h = global_host();

        // Optional extra plug-in search paths + backend/blocklist settings from casparcg.config:
        //   <ofx>
        //     <plugin-path>C:\Path\To\OFX</plugin-path> ...
        //     <enable-opengl>true</enable-opengl>
        //     <enable-cuda>false</enable-cuda>
        //     <blocklist><plugin>com.vendor.bad</plugin> ...</blocklist>
        //   </ofx>
        try {
            if (auto ofx = env::properties().get_child_optional(L"configuration.ofx")) {
                const auto is_true = [](const std::wstring& s) {
                    return s == L"true" || s == L"1" || s == L"yes";
                };
                for (const auto& kv : *ofx) {
                    if (kv.first == L"plugin-path") {
                        const auto path = kv.second.get_value<std::wstring>();
                        if (!path.empty())
                            h.add_search_path(u8(path));
                    } else if (kv.first == L"enable-opengl") {
                        configure_opengl(is_true(kv.second.get_value<std::wstring>()));
                    } else if (kv.first == L"enable-cuda") {
                        configure_cuda(is_true(kv.second.get_value<std::wstring>()));
                    } else if (kv.first == L"blocklist") {
                        for (const auto& bp : kv.second) {
                            if (bp.first == L"plugin") {
                                const auto id = bp.second.get_value<std::wstring>();
                                if (!id.empty())
                                    blocklist_plugin(u8(id));
                            }
                        }
                    }
                }
            }
        } catch (...) {
        }

        h.scan();

        const auto& plugins = h.plugins();
        CASPAR_LOG(info) << L"[ofx] OpenFX host initialised (image-effect API v"
                         << kOfxImageEffectPluginApiVersion << L"); discovered " << plugins.size()
                         << L" plug-in(s).";

        for (const auto& p : plugins) {
            CASPAR_LOG(info) << L"[ofx]   " << u16(p.identifier) << L" (" << u16(p.label) << L") v"
                             << p.version_major << L"." << p.version_minor;
        }

        dependencies.producer_registry->register_producer_factory(L"OFX Producer", create_producer);
    } catch (...) {
        CASPAR_LOG(warning) << L"[ofx] OpenFX host initialisation failed; OFX plug-ins unavailable.";
    }
#else
    CASPAR_LOG(info) << L"[ofx] OpenFX module loaded (host support disabled at build time).";
#endif
}

}} // namespace caspar::ofx
