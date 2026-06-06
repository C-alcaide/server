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

#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

#include <vulkan/vulkan.hpp>

namespace caspar { namespace accelerator { namespace vulkan {

class vulkan_queue;

// Picks which queue families to take queues from and owns the resulting queues.
//
// Deliberately small: queue count is frozen at vkCreateDevice, so this is a
// two-phase object spanning device creation. The ctor scans the families and
// selects which to take one queue from each (graphics first, capped); the device
// feeds families() into its custom queue setup, then calls create_queues() once
// the VkDevice exists. After that it just hands queues out — primary() for the
// render/transfer path, acquire() for a client that wants its own family.
//
// There is no capability scoring, no per-queue reclamation, and no ref-counting:
// the families this hardware exposes are taken in order, and acquire() shares
// (vulkan_queue is internally synchronized) so it can never exhaust.
class queue_manager final
{
  public:
    // Phase 1: scan + select (graphics family first, then the rest, capped at
    // max_queues). Logs the scan. create_queues() must be called before any
    // primary()/acquire().
    queue_manager(vk::PhysicalDevice physical_device, size_t max_queues);

    queue_manager(const queue_manager&)            = delete;
    queue_manager& operator=(const queue_manager&) = delete;

    // The families to create one queue each from, in selection order (families()[0]
    // is the graphics family). Feed these to the device's custom queue setup.
    const std::vector<uint32_t>& families() const { return families_; }

    // Phase 2: the VkDevice has been created with one queue per families() entry —
    // pull the handles into vulkan_queues.
    void create_queues(vk::Device device);

    // The primary graphics queue (families()[0]); the render/transfer path.
    std::shared_ptr<vulkan_queue> primary() const;

    // A queue on a family distinct from the primary, round-robin + shared across
    // the secondary families. Collapses to primary() on single-family hardware.
    std::shared_ptr<vulkan_queue> acquire();

  private:
    std::vector<uint32_t>                      families_;
    std::vector<std::shared_ptr<vulkan_queue>> queues_;
    std::atomic<size_t>                        cursor_{0};
};

}}} // namespace caspar::accelerator::vulkan
