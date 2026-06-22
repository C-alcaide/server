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

#include <vulkan/vulkan.h>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <vulkan/vulkan_win32.h>
#endif

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace caspar { namespace vulkan_output {

// Separate Vulkan device for presentation, isolated from the shared accelerator
// device used by the mixer. This provides:
//
// 1. TDR isolation: if the output device TDRs (e.g. driver bug in FSE acquire),
//    the mixer's rendering device is unaffected.
// 2. HIGH queue priority: output submission preempts DWM/other GPU clients.
// 3. Multi-queue pool: each consumer gets its own VkQueue for parallel present.
// 4. Full extension control: instance has all presentation extensions regardless
//    of what the shared accelerator requests.
//
// Trade-off: source frames must be imported via VK_KHR_external_memory (requires
// the mixer device to export its images). Without external memory, frames are
// copied through host memory (slower but still functional).
//
// This class is OPTIONAL — the consumer can also use the shared accelerator
// device directly (as commits 1-7 do). Enable this via config <separate-device>.

class output_device
{
  public:
    // Create a separate instance + device targeting the specified GPU.
    // gpu_index: 0-based physical device index (same ordering as DXGI adapters).
    explicit output_device(int gpu_index);
    ~output_device();

    output_device(const output_device&)            = delete;
    output_device& operator=(const output_device&) = delete;

    VkInstance       instance() const { return instance_; }
    VkPhysicalDevice physical_device() const { return physical_device_; }
    VkDevice         device() const { return device_; }
    uint32_t         queue_family() const { return queue_family_; }
    uint32_t         queue_count() const { return queue_count_; }

    // Acquire a queue index. Thread-safe, wraps around queue_count_.
    uint32_t acquire_queue_index();

    // Get queue by index (wraps if idx >= queue_count_).
    VkQueue  queue_at(uint32_t idx) const { return queues_[idx % queue_count_]; }

    // Get per-queue mutex for serialized submission.
    std::mutex& queue_mutex(uint32_t idx) { return queue_mutexes_[idx % queue_count_]; }

    // Surface creation (instance owns the surface lifetime).
#ifdef _WIN32
    VkSurfaceKHR create_win32_surface(HWND hwnd);
#endif
    VkSurfaceKHR create_display_surface(VkDisplayKHR display, uint32_t width, uint32_t height,
                                        uint32_t target_refresh_mhz = 0);
    void         destroy_surface(VkSurfaceKHR surface);

    // Check if device extension is enabled.
    bool has_extension(const char* name) const;

    // Device LUID (for cross-API GPU matching). Returns nullptr if unavailable.
    const uint8_t* luid() const { return luid_valid_ ? luid_ : nullptr; }

  private:
    void create_instance_();
    void select_physical_device_(int gpu_index);
    void create_device_();

    VkInstance               instance_        = VK_NULL_HANDLE;
    VkPhysicalDevice         physical_device_ = VK_NULL_HANDLE;
    VkDevice                 device_          = VK_NULL_HANDLE;
    uint32_t                 queue_family_    = 0;
    uint32_t                 queue_count_     = 0;
    std::vector<VkQueue>     queues_;
    std::vector<std::mutex>  queue_mutexes_;
    std::atomic<uint32_t>    next_queue_{0};
    std::vector<std::string> enabled_extensions_;
    uint8_t                  luid_[8]         = {};
    bool                     luid_valid_      = false;
    VkDebugUtilsMessengerEXT debug_messenger_ = VK_NULL_HANDLE;
};

}} // namespace caspar::vulkan_output
