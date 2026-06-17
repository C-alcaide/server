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
std::vector<std::shared_ptr<vulkan_device>> vk_device_manager::warmup_refs_;
std::mutex                                  vk_device_manager::gate_mutex_;
std::condition_variable                     vk_device_manager::gate_cv_;
std::map<int, vk_device_manager::gate_state> vk_device_manager::gate_;
std::mutex                                  vk_device_manager::sync_mutex_;
std::condition_variable                     vk_device_manager::sync_cv_;
std::map<int, vk_device_manager::sync_state> vk_device_manager::sync_;

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
    // The warm_up refs keep devices alive until consumers claim them via get().
    // On subsequent calls (e.g. config reload), the old refs are merged/replaced
    // safely — devices still in use by consumers survive via their shared_ptrs.
    std::lock_guard<std::mutex> lock(mutex_);
    warmup_refs_.clear(); // Release previous warm-up refs (consumers hold their own)

    for (int idx : gpu_indices) {
        try {
            // Inline of get() logic — we already hold mutex_
            auto it = devices_.find(idx);
            std::shared_ptr<vulkan_device> ptr;
            if (it != devices_.end())
                ptr = it->second.lock();
            if (!ptr) {
                if (it != devices_.end())
                    devices_.erase(it);
                ptr = std::shared_ptr<vulkan_device>(
                    new vulkan_device(idx, 0),
                    [idx](vulkan_device* dev) {
                        {
                            std::lock_guard<std::mutex> lk(mutex_);
                            devices_.erase(idx);
                        }
                        CASPAR_LOG(info) << L"[vulkan] Releasing shared VkDevice for GPU " << idx;
                        delete dev;
                    });
                devices_[idx] = ptr;
                CASPAR_LOG(info) << L"[vulkan] Created shared VkDevice for GPU " << idx;
            }
            warmup_refs_.push_back(std::move(ptr));
        } catch (const std::exception& e) {
            CASPAR_LOG(error) << L"[vulkan] Failed to pre-create VkDevice for GPU " << idx << L": " << e.what();
        }
    }

    CASPAR_LOG(info) << L"[vulkan] Warmed up " << warmup_refs_.size() << L" VkDevice(s)";
}

void vk_device_manager::set_expected_consumers(const std::map<int, int>& gpu_counts)
{
    std::lock_guard<std::mutex> lock(gate_mutex_);
    gate_.clear();
    for (auto& [gpu_idx, count] : gpu_counts) {
        gate_[gpu_idx] = {count, 0};
        CASPAR_LOG(info) << L"[vulkan] Startup gate: expecting " << count
                         << L" consumer(s) on GPU " << gpu_idx;
    }
}

void vk_device_manager::consumer_ready(int gpu_index)
{
    std::lock_guard<std::mutex> lock(gate_mutex_);
    auto it = gate_.find(gpu_index);
    if (it == gate_.end())
        return;
    it->second.ready++;
    CASPAR_LOG(info) << L"[vulkan] Startup gate: GPU " << gpu_index << L" - "
                     << it->second.ready << L"/" << it->second.expected << L" consumers ready";
    if (it->second.ready >= it->second.expected)
        gate_cv_.notify_all();
}

void vk_device_manager::wait_all_ready(int gpu_index)
{
    std::unique_lock<std::mutex> lock(gate_mutex_);
    auto it = gate_.find(gpu_index);
    if (it == gate_.end())
        return;  // No expectation set — don't block
    // Wait with a generous timeout (30s) to avoid hanging forever if a
    // consumer fails to initialise.
    bool ok = gate_cv_.wait_for(lock, std::chrono::seconds(30), [&] {
        return it->second.ready >= it->second.expected;
    });
    if (!ok) {
        CASPAR_LOG(warning) << L"[vulkan] Startup gate timeout on GPU " << gpu_index
                            << L" (" << it->second.ready << L"/" << it->second.expected
                            << L" ready). Starting present thread anyway.";
    }
}

void vk_device_manager::release_warmup()
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (!warmup_refs_.empty()) {
        CASPAR_LOG(info) << L"[vulkan] Releasing " << warmup_refs_.size() << L" warm-up device ref(s)";
        warmup_refs_.clear();
    }
}

// ── Software present barrier ────────────────────────────────────────────────

std::shared_ptr<void> vk_device_manager::sync_group_join(int sync_group)
{
    if (sync_group == 0)
        return nullptr;

    {
        std::lock_guard<std::mutex> lock(sync_mutex_);
        sync_[sync_group].participants++;
        CASPAR_LOG(info) << L"[vulkan] Sync group " << sync_group
                         << L": participant joined (total=" << sync_[sync_group].participants << L")";
    }

    // Return a shared_ptr whose destructor removes the participant.
    return std::shared_ptr<void>(nullptr, [sync_group](void*) {
        std::lock_guard<std::mutex> lock(sync_mutex_);
        auto it = sync_.find(sync_group);
        if (it != sync_.end()) {
            it->second.participants--;
            CASPAR_LOG(info) << L"[vulkan] Sync group " << sync_group
                             << L": participant left (total=" << it->second.participants << L")";
            if (it->second.participants <= 0) {
                sync_.erase(it);
            }
            // Wake anyone blocked — participant count changed
            sync_cv_.notify_all();
        }
    });
}

bool vk_device_manager::sync_group_wait(int sync_group, int timeout_ms)
{
    std::unique_lock<std::mutex> lock(sync_mutex_);
    auto it = sync_.find(sync_group);
    if (it == sync_.end() || it->second.participants <= 1)
        return true; // Solo — no sync needed

    auto& state = it->second;
    auto  gen   = state.generation;
    state.waiting++;

    if (state.waiting >= state.participants) {
        // Last to arrive — open the barrier
        state.waiting = 0;
        state.generation++;
        sync_cv_.notify_all();
        return true;
    }

    // Wait for the barrier to open (generation to advance)
    bool ok = sync_cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [&] {
        // Re-lookup in case map changed (participant left)
        auto it2 = sync_.find(sync_group);
        if (it2 == sync_.end() || it2->second.participants <= 1)
            return true;
        return it2->second.generation != gen;
    });

    if (!ok) {
        // Timed out — decrement waiting count and proceed anyway
        auto it2 = sync_.find(sync_group);
        if (it2 != sync_.end() && it2->second.waiting > 0)
            it2->second.waiting--;
    }
    return ok;
}

}} // namespace caspar::vulkan_output
