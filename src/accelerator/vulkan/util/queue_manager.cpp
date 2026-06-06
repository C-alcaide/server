/*
 * Copyright 2025
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
 * Author: Niklas Andersson, niklas@niklaspandersson.se
 */

#include "queue_manager.h"

#include "vulkan_queue.h"

#include <common/except.h>
#include <common/log.h>

#include <algorithm>

namespace caspar { namespace accelerator { namespace vulkan {

queue_manager::queue_manager(vk::PhysicalDevice physical_device, size_t max_queues)
{
    auto family_props = physical_device.getQueueFamilyProperties();

    uint32_t graphics_family = UINT32_MAX;
    for (uint32_t i = 0; i < family_props.size(); ++i) {
        if (family_props[i].queueFlags & vk::QueueFlagBits::eGraphics) {
            graphics_family = i;
            break;
        }
    }
    if (graphics_family == UINT32_MAX) {
        CASPAR_THROW_EXCEPTION(caspar_exception() << msg_info("No graphics-capable Vulkan queue family found"));
    }

    // Graphics family first (the primary queue), then the rest, capped.
    families_.push_back(graphics_family);
    for (uint32_t i = 0; i < family_props.size() && families_.size() < max_queues; ++i) {
        if (i != graphics_family)
            families_.push_back(i);
    }

    CASPAR_LOG(info) << "Vulkan: " << family_props.size() << " queue families available, creating " << families_.size()
                     << " queues.";
    for (uint32_t i = 0; i < family_props.size(); ++i) {
        bool created = std::find(families_.begin(), families_.end(), i) != families_.end();
        CASPAR_LOG(info) << "  queue family " << i << ": queueCount=" << family_props[i].queueCount
                         << " flags=" << vk::to_string(family_props[i].queueFlags).c_str()
                         << (created ? " [queue created]" : "");
    }
}

void queue_manager::create_queues(vk::Device device)
{
    for (auto family : families_) {
        queues_.push_back(std::make_shared<vulkan_queue>(device.getQueue(family, 0), family));
    }
}

std::shared_ptr<vulkan_queue> queue_manager::primary() const { return queues_.at(0); }

std::shared_ptr<vulkan_queue> queue_manager::acquire()
{
    // Round-robin over the secondary families (everything past queues_[0]),
    // sharing if there are more clients than secondaries. Single-family hardware
    // has none, so this returns the primary and the client aliases the render
    // queue (the distance-0, single-queue path).
    if (queues_.size() <= 1)
        return queues_.at(0);

    auto n = queues_.size() - 1;
    return queues_[1 + (cursor_++ % n)];
}

}}} // namespace caspar::accelerator::vulkan
