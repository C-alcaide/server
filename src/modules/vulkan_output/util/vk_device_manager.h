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

#pragma once

#include <map>
#include <memory>
#include <mutex>

namespace caspar { namespace vulkan_output {

class vulkan_device;

// Singleton registry that shares one VkDevice per physical GPU across all
// vulkan_output_consumer instances.  This is required for VK_NV_present_barrier
// to frame-lock multiple swapchains — the driver only synchronises swapchains
// belonging to the same VkDevice.
//
// Usage:
//   auto dev = vk_device_manager::get(gpu_index);  // shared_ptr
//   // ... consumer uses dev until destruction
//   dev.reset();  // last consumer releases the device
class vk_device_manager
{
  public:
    // Returns a shared VkDevice for the given GPU index.  If no device exists
    // yet, one is created.  When the last shared_ptr is released the device
    // is destroyed and removed from the registry.
    static std::shared_ptr<vulkan_device> get(int gpu_index);

  private:
    static std::mutex                                    mutex_;
    static std::map<int, std::weak_ptr<vulkan_device>>   devices_;
};

}} // namespace caspar::vulkan_output
