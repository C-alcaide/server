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

#include "vk_device_manager.h"
#include "vulkan_device.h"

#include <common/log.h>

namespace caspar { namespace vulkan_output {

std::mutex                                  vk_device_manager::mutex_;
std::map<int, std::weak_ptr<vulkan_device>> vk_device_manager::devices_;

std::shared_ptr<vulkan_device> vk_device_manager::get(int gpu_index)
{
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = devices_.find(gpu_index);
    if (it != devices_.end()) {
        auto ptr = it->second.lock();
        if (ptr) {
            CASPAR_LOG(info) << L"[vulkan] Sharing existing VkDevice for GPU " << gpu_index
                             << L" (use_count=" << ptr.use_count() << L")";
            return ptr;
        }
        // Expired — remove stale entry
        devices_.erase(it);
    }

    // Create new device.  output_index=0 is unused by vulkan_device constructor.
    auto ptr = std::shared_ptr<vulkan_device>(
        new vulkan_device(gpu_index, 0),
        [gpu_index](vulkan_device* dev) {
            // Custom deleter: remove from registry before destroying
            {
                std::lock_guard<std::mutex> lock(mutex_);
                devices_.erase(gpu_index);
            }
            CASPAR_LOG(info) << L"[vulkan] Releasing shared VkDevice for GPU " << gpu_index;
            delete dev;
        });

    devices_[gpu_index] = ptr;
    CASPAR_LOG(info) << L"[vulkan] Created shared VkDevice for GPU " << gpu_index;
    return ptr;
}

void vk_device_manager::warm_up(const std::vector<int>& gpu_indices)
{
    // Pre-create all needed VkDevices sequentially before any consumer starts
    // its present thread.  This avoids TDR caused by creating a VkDevice on
    // GPU B while GPU A is already presenting (a known issue on older NVIDIA
    // drivers where vkCreateDevice stalls the entire GPU pipeline).
    //
    // The warm_up refs are stored locally to keep the devices alive until
    // consumers claim them via get().  Since consumers call get() during their
    // init_vulkan() (before the warm_up refs go out of scope), the devices will
    // be shared rather than destroyed.
    std::vector<std::shared_ptr<vulkan_device>> refs;
    refs.reserve(gpu_indices.size());

    for (int idx : gpu_indices) {
        try {
            refs.push_back(get(idx));
        } catch (const std::exception& e) {
            CASPAR_LOG(error) << L"[vulkan] Failed to pre-create VkDevice for GPU " << idx << L": " << e.what();
        }
    }

    CASPAR_LOG(info) << L"[vulkan] Warmed up " << refs.size() << L" VkDevice(s)";
    // refs go out of scope — but consumers will call get() soon, so weak_ptrs
    // may or may not expire.  We need to keep them alive.
    // Actually, let's store them in a static to keep alive until shutdown.
    static std::vector<std::shared_ptr<vulkan_device>> warmup_refs;
    warmup_refs = std::move(refs);
}

}} // namespace caspar::vulkan_output
