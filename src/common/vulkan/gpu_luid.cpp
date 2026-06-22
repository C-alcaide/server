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

#include "gpu_luid.h"

#include <common/log.h>

#include <algorithm>
#include <cstring>

namespace caspar { namespace vulkan_common {

bool query_device_luid(VkPhysicalDevice device, std::array<uint8_t, 8>& out_luid)
{
    VkPhysicalDeviceIDProperties id_props{};
    id_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
    VkPhysicalDeviceProperties2 props2{};
    props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props2.pNext = &id_props;
    vkGetPhysicalDeviceProperties2(device, &props2);
    if (!id_props.deviceLUIDValid)
        return false;
    std::memcpy(out_luid.data(), id_props.deviceLUID, 8);
    return true;
}

std::vector<VkPhysicalDevice> deduplicate_by_luid(const std::vector<VkPhysicalDevice>& devices)
{
    std::vector<std::array<uint8_t, 8>> seen_luids;
    std::vector<VkPhysicalDevice>       unique;

    for (auto dev : devices) {
        std::array<uint8_t, 8> luid{};
        if (query_device_luid(dev, luid)) {
            if (std::find(seen_luids.begin(), seen_luids.end(), luid) != seen_luids.end())
                continue;
            seen_luids.push_back(luid);
        }
        unique.push_back(dev);
    }

    if (unique.size() < devices.size()) {
        CASPAR_LOG(info) << L"[vulkan] Deduplicated " << devices.size() << L" physical devices to " << unique.size()
                         << L" unique GPU(s) by LUID";
    }

    return unique;
}

}} // namespace caspar::vulkan_common
