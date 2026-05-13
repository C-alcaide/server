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

#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

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
//
// Startup gate:
//   Prevents TDR by holding present threads until all consumers on the same GPU
//   have finished heavy initialisation (window + swapchain creation).
//   1. server.cpp calls set_expected_consumers() before consumer init loop
//   2. Each consumer calls consumer_ready() after swapchain setup
//   3. Each consumer calls wait_all_ready() — blocks until all peers are ready
//   4. Only then does the consumer start its present thread
class vk_device_manager
{
  public:
    // Returns a shared VkDevice for the given GPU index.  If no device exists
    // yet, one is created.  When the last shared_ptr is released the device
    // is destroyed and removed from the registry.
    static std::shared_ptr<vulkan_device> get(int gpu_index);

    // Pre-create VkDevices for a set of GPU indices.  Call this before any
    // consumer starts its present loop to avoid creating a VkDevice on one
    // GPU while another GPU is already presenting — which can trigger TDR
    // on older NVIDIA drivers.
    static void warm_up(const std::vector<int>& gpu_indices);

    // ── Startup gate ────────────────────────────────────────────────────

    // Declare how many consumers will initialise on each GPU.
    // Must be called before any consumer's initialize().
    static void set_expected_consumers(const std::map<int, int>& gpu_counts);

    // A consumer calls this after finishing its heavy init (swapchain created).
    // Increments the ready counter for *gpu_index*.
    static void consumer_ready(int gpu_index);

    // Blocks until all expected consumers on *gpu_index* have called
    // consumer_ready().  Returns immediately if no expectation was set.
    static void wait_all_ready(int gpu_index);

    // Release warm-up refs — allows VkDevices no longer used by any consumer
    // to be destroyed. Call after all consumers have been initialized.
    static void release_warmup();

    // ── Software present barrier ────────────────────────────────────────
    // Lightweight frame-lock for consumers in the same sync group.
    // Each present thread calls join() before vkQueueSubmit; it blocks
    // until all registered participants arrive, then they all proceed.

    // Register a participant in a sync group.  Returns a token the caller
    // must keep alive; the participant is removed when the token is destroyed.
    static std::shared_ptr<void> sync_group_join(int sync_group);

    // Block until all participants in *sync_group* have called wait().
    // Returns false if timed out (e.g. peer died).
    static bool sync_group_wait(int sync_group, int timeout_ms = 50);

  private:
    static std::mutex                                    mutex_;
    static std::map<int, std::weak_ptr<vulkan_device>>   devices_;

    // Warm-up refs — keep VkDevices alive between warm_up() and consumer init
    static std::vector<std::shared_ptr<vulkan_device>>   warmup_refs_;

    // Startup gate state
    struct gate_state
    {
        int expected = 0;
        int ready    = 0;
    };
    static std::mutex                       gate_mutex_;
    static std::condition_variable          gate_cv_;
    static std::map<int, gate_state>        gate_;  // gpu_index → state

    // Software present barrier state
    struct sync_state
    {
        int          participants = 0; // number of registered consumers
        int          waiting      = 0; // number currently blocked in wait()
        unsigned     generation   = 0; // flips each time the barrier opens
    };
    static std::mutex                       sync_mutex_;
    static std::condition_variable          sync_cv_;
    static std::map<int, sync_state>        sync_;  // sync_group → state
};

}} // namespace caspar::vulkan_output
