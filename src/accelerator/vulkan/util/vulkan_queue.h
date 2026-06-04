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

#pragma once

#include <cstdint>

#include <vulkan/vulkan.hpp>

namespace caspar { namespace accelerator { namespace vulkan {

// A single VkQueue together with the family it was created from. Mirrors
// vk::Queue::submit so call sites stop passing raw handles around. Single
// queue, single thread for now: no internal synchronization.
class vulkan_queue final
{
  public:
    vulkan_queue(vk::Queue queue, uint32_t family_index);

    vulkan_queue(const vulkan_queue&)            = delete;
    vulkan_queue& operator=(const vulkan_queue&) = delete;

    void submit(const vk::ArrayProxy<const vk::SubmitInfo>& submits, vk::Fence fence = {});

    uint32_t family_index() const { return family_index_; }

  private:
    vk::Queue queue_;
    uint32_t  family_index_;
};

}}} // namespace caspar::accelerator::vulkan
