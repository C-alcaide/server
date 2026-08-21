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

#include "shared_device_info.h"

#include "device.h"

namespace caspar { namespace accelerator { namespace vulkan {

shared_device_info describe_shared_device(void* vk_device)
{
    shared_device_info info;
    auto*              dev = static_cast<device*>(vk_device);
    if (!dev)
        return info;

    info.valid           = true;
    info.instance        = static_cast<VkInstance>(dev->getVkInstance());
    info.physical_device = static_cast<VkPhysicalDevice>(dev->getVkPhysicalDevice());
    info.device          = static_cast<VkDevice>(dev->getVkDevice());
    info.get_proc_addr   = reinterpret_cast<void*>(dev->getInstanceProcAddr());
    info.graphics_qf     = dev->getGraphicsQueueFamily();
    info.decode_qf       = dev->getDecodeQueueFamily();
    info.decode_qf_isolated        = dev->hasDedicatedDecodeQueue();
    info.encode_qf                 = dev->getEncodeQueueFamily();
    info.encode_qf_present         = dev->hasEncodeQueue();
    info.enabled_device_extensions = dev->getEnabledDeviceExtensions();
    return info;
}

}}} // namespace caspar::accelerator::vulkan
