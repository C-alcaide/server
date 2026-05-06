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

#include "vulkan_output.h"

#include "consumer/vulkan_output_consumer.h"
#include "util/vulkan_device.h"

#include <core/consumer/frame_consumer.h>

#include <common/log.h>

#include <protocol/amcp/amcp_command_repository_wrapper.h>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>

#include <sstream>

namespace caspar { namespace vulkan_output {

namespace {

std::wstring info_vulkan_output_command(protocol::amcp::command_context& /*ctx*/)
{
    boost::property_tree::wptree info;

    auto displays = vulkan_device::enumerate_displays();

    int idx = 0;
    for (const auto& d : displays) {
        auto base = L"vulkan-outputs.output";
        boost::property_tree::wptree output;
        output.add(L"gpu-index", d.gpu_index);
        output.add(L"output-index", d.output_index);
        output.add(L"gpu-name", d.gpu_name);
        output.add(L"display-name", d.display_name);
        output.add(L"width", d.width);
        output.add(L"height", d.height);
        output.add(L"tier", d.tier == gpu_tier::pro ? L"pro" : L"consumer");
        info.add_child(base, output);
        ++idx;
    }

    std::wstringstream reply;
    reply << L"201 INFO VULKAN_OUTPUT OK\r\n";
    boost::property_tree::xml_writer_settings<std::wstring> w(L' ', 3);
    boost::property_tree::xml_parser::write_xml(reply, info, w);
    reply << L"\r\n";
    return reply.str();
}

} // namespace

void init(const core::module_dependencies& dependencies)
{
    dependencies.consumer_registry->register_consumer_factory(L"Vulkan Output", create_consumer);
    dependencies.consumer_registry->register_preconfigured_consumer_factory(L"vulkan-output",
                                                                            create_preconfigured_consumer);

    if (dependencies.command_repository) {
        dependencies.command_repository->register_command(
            L"Query Commands", L"INFO VULKAN_OUTPUT", info_vulkan_output_command, 0);
    }

    // Log available outputs at startup
    auto displays = vulkan_device::enumerate_displays();
    if (!displays.empty()) {
        CASPAR_LOG(info) << L"[vulkan_output] Available outputs:";
        for (const auto& d : displays) {
            CASPAR_LOG(info) << L"  GPU " << d.gpu_index << L" Output " << d.output_index << L": "
                             << d.gpu_name << L" - " << d.display_name
                             << L" [" << (d.tier == gpu_tier::pro ? L"Pro" : L"Consumer") << L"]";
        }
    }
}

}} // namespace caspar::vulkan_output
