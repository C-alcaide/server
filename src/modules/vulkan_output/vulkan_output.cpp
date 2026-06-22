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
#include "util/display_enum.h"

#include <core/consumer/frame_consumer.h>

#include <common/log.h>

#include <protocol/amcp/amcp_command_repository_wrapper.h>

#include <accelerator/vulkan/util/device.h>

#include <VkBootstrap.h>
#include <vulkan/vulkan.h>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>

#include <memory>
#include <sstream>

namespace caspar { namespace vulkan_output {

void init(const core::module_dependencies& dependencies)
{
#ifdef ENABLE_VULKAN
    auto vk_device = std::dynamic_pointer_cast<accelerator::vulkan::device>(dependencies.accelerator_device);
    if (!vk_device) {
        CASPAR_LOG(info) << L"[vulkan_output] Vulkan accelerator not active; module disabled.";
        return;
    }

    dependencies.consumer_registry->register_consumer_factory(
        L"Vulkan Output",
        [vk_device](const std::vector<std::wstring>&                         params,
                    const core::video_format_repository&                     format_repository,
                    const std::vector<spl::shared_ptr<core::video_channel>>& channels,
                    const core::channel_info& channel_info) -> spl::shared_ptr<core::frame_consumer> {
            return create_consumer(vk_device, params, format_repository, channels, channel_info);
        });

    dependencies.consumer_registry->register_preconfigured_consumer_factory(
        L"vulkan-output",
        [vk_device](const boost::property_tree::wptree&                      ptree,
                    const core::video_format_repository&                     format_repository,
                    const std::vector<spl::shared_ptr<core::video_channel>>& channels,
                    const core::channel_info& channel_info) -> spl::shared_ptr<core::frame_consumer> {
            return create_preconfigured_consumer(vk_device, ptree, format_repository, channels, channel_info);
        });

    if (dependencies.command_repository) {
        dependencies.command_repository->register_command(
            L"Query Commands",
            L"INFO VULKAN_OUTPUT",
            [](protocol::amcp::command_context& /*ctx*/) -> std::wstring {
                boost::property_tree::wptree info;

                auto displays = enumerate_displays();
                for (const auto& d : displays) {
                    boost::property_tree::wptree output;
                    output.add(L"gpu-index", d.gpu_index);
                    output.add(L"output-index", d.output_index);
                    output.add(L"gpu-name", d.gpu_name);
                    output.add(L"display-name", d.display_name);
                    output.add(L"width", d.width);
                    output.add(L"height", d.height);
                    info.add_child(L"vulkan-outputs.output", output);
                }

                std::wstringstream reply;
                reply << L"201 INFO VULKAN_OUTPUT OK\r\n";
                boost::property_tree::xml_writer_settings<std::wstring> w(L' ', 3);
                boost::property_tree::xml_parser::write_xml(reply, info, w);
                reply << L"\r\n";
                return reply.str();
            },
            0);
    }

    // Log available outputs at startup
    auto displays = enumerate_displays();
    if (!displays.empty()) {
        CASPAR_LOG(info) << L"[vulkan_output] Available outputs:";
        for (const auto& d : displays) {
            CASPAR_LOG(info) << L"  GPU " << d.gpu_index << L" Output " << d.output_index << L": " << d.gpu_name
                             << L" - " << d.display_name << L" (" << d.width << L"x" << d.height << L")";
        }
    } else {
        CASPAR_LOG(info) << L"[vulkan_output] No displays enumerated.";
    }
#else
    CASPAR_LOG(info) << L"[vulkan_output] Vulkan not enabled; module disabled.";
#endif
}

#ifdef ENABLE_VULKAN
void register_vulkan_requirements(vkb::PhysicalDevice& pd)
{
    pd.enable_extension_if_present(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
#ifdef _WIN32
    pd.enable_extension_if_present("VK_EXT_full_screen_exclusive");
    pd.enable_extension_if_present("VK_KHR_win32_surface");
#endif
    pd.enable_extension_if_present("VK_KHR_display_swapchain");
}
#endif

}} // namespace caspar::vulkan_output
