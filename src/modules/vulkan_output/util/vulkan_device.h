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

#define VK_USE_PLATFORM_WIN32_KHR
#include <vulkan/vulkan.h>

#include <memory>
#include <string>
#include <vector>

namespace caspar { namespace vulkan_output {

// Capability tier detected at runtime
enum class gpu_tier
{
    pro,      // VK_KHR_display available (Quadro/RTX Pro)
    consumer, // VK_EXT_full_screen_exclusive only (GeForce)
    none      // No usable output path
};

struct display_info
{
    int          gpu_index;
    int          output_index;
    std::wstring gpu_name;
    std::wstring display_name;
    uint32_t     width;
    uint32_t     height;
    double       refresh_rate;
    gpu_tier     tier;
    VkDisplayKHR display_handle; // Only valid for Pro tier
};

// RAII wrapper for Vulkan instance + device
class vulkan_device
{
  public:
    vulkan_device(int gpu_index, int output_index);
    ~vulkan_device();

    vulkan_device(const vulkan_device&)            = delete;
    vulkan_device& operator=(const vulkan_device&) = delete;

    VkInstance       instance() const { return instance_; }
    VkPhysicalDevice physical_device() const { return physical_device_; }
    VkDevice         device() const { return device_; }
    uint32_t         present_queue_family() const { return present_queue_family_; }
    VkQueue          present_queue() const { return present_queue_; }
    gpu_tier         tier() const { return tier_; }
    int              gpu_index() const { return gpu_index_; }

    // Device LUID for cross-API GPU matching (OGL ↔ VK)
    const uint8_t*   device_luid() const { return device_luid_; }
    bool             device_luid_valid() const { return device_luid_valid_; }

    // Surface + swapchain creation
    VkSurfaceKHR create_display_surface(const display_info& info, uint32_t target_refresh_mhz = 0);
    VkSurfaceKHR create_win32_surface(HWND hwnd);

    // Check if a device extension is enabled
    bool has_extension(const char* name) const;

    // VBlank fence for Pro tier (VK_EXT_display_control)
    // Returns a fence that signals on the next first-pixel-out event.
    // Returns VK_NULL_HANDLE if display_control is not available.
    VkFence create_vblank_fence(VkDisplayKHR display);

    // Enumerate all available outputs across all GPUs (creates temporary instance)
    static std::vector<display_info> enumerate_displays();

    // Enumerate displays attached to THIS device's physical GPU (uses live instance — handles valid for device lifetime)
    std::vector<display_info> enumerate_displays_on_device() const;

  private:
    void create_instance();
    void select_physical_device(int gpu_index);
    void create_logical_device();

    VkInstance               instance_             = VK_NULL_HANDLE;
    VkPhysicalDevice         physical_device_      = VK_NULL_HANDLE;
    VkDevice                 device_               = VK_NULL_HANDLE;
    uint32_t                 present_queue_family_ = 0;
    VkQueue                  present_queue_        = VK_NULL_HANDLE;
    gpu_tier                 tier_                 = gpu_tier::none;
    int                      gpu_index_            = 0;
    uint8_t                  device_luid_[8]       = {};
    bool                     device_luid_valid_    = false;
    VkDebugUtilsMessengerEXT debug_messenger_      = VK_NULL_HANDLE;
    std::vector<std::string> enabled_extensions_;
};

}} // namespace caspar::vulkan_output
