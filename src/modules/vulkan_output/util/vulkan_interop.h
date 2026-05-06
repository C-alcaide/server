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

#include <cstdint>

namespace caspar { namespace vulkan_output {

class vulkan_device;

// Manages an OpenGL → Vulkan shared memory object for zero-copy frame transfer.
// Uses GL_EXT_memory_object + VK_KHR_external_memory_win32.
class vulkan_interop
{
  public:
    vulkan_interop(vulkan_device& device, uint32_t width, uint32_t height, VkFormat format);
    ~vulkan_interop();

    vulkan_interop(const vulkan_interop&)            = delete;
    vulkan_interop& operator=(const vulkan_interop&) = delete;

    // Import an OGL texture (already allocated with GL_EXT_memory_object) into VK.
    // Returns a VkImage backed by the shared memory.
    VkImage       vk_image() const { return image_; }
    VkImageView   vk_image_view() const { return image_view_; }
    VkDeviceMemory vk_memory() const { return memory_; }
    HANDLE        shared_handle() const { return shared_handle_; }

    // Create the Vulkan side from a Win32 HANDLE exported by OpenGL
    void import_from_handle(HANDLE handle);

  private:
    vulkan_device& device_;
    uint32_t       width_  = 0;
    uint32_t       height_ = 0;
    VkFormat       format_ = VK_FORMAT_B8G8R8A8_UNORM;

    VkImage        image_      = VK_NULL_HANDLE;
    VkImageView    image_view_ = VK_NULL_HANDLE;
    VkDeviceMemory memory_     = VK_NULL_HANDLE;
    HANDLE         shared_handle_ = nullptr;
};

}} // namespace caspar::vulkan_output
