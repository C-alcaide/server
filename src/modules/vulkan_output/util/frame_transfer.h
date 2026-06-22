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

#include <vulkan/vulkan.hpp>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <vulkan/vulkan_win32.h>
#endif

#include <cstdint>
#include <memory>

namespace caspar { namespace vulkan_output {

// Transfer strategy determined at construction time based on device capabilities.
enum class transfer_mode
{
    same_device,      // Source and dest on same VkDevice — zero-copy, use image directly
    external_memory,  // Cross-GPU via VK_KHR_external_memory (NVLink-accelerated when available)
    host_staging,     // Cross-GPU fallback: download to host, upload to dest device
};

// Handles getting a frame from the mixer device to the output device.
//
// When both devices are the same physical GPU (LUID match or same VkDevice),
// this is a no-op passthrough — the source VkImage is used directly.
//
// When devices differ, this class manages the transfer:
// - External memory: exports the source image's memory as an OS handle,
//   imports it on the output device. The GPU driver routes the DMA over
//   NVLink when the GPUs are NVLink-connected, otherwise over PCIe.
// - Host staging: allocates host-visible buffers on both sides, copies
//   src→host→dst. Slowest path but always available.
class frame_transfer
{
  public:
    // Create a frame transfer between source (mixer) and destination (output) devices.
    // If src_device == dst_device, mode is same_device (passthrough).
    // Otherwise probes for external memory support.
    frame_transfer(vk::Device         src_device,
                   vk::PhysicalDevice src_physical,
                   vk::Device         dst_device,
                   vk::PhysicalDevice dst_physical,
                   uint32_t           dst_queue_family,
                   uint32_t           width,
                   uint32_t           height,
                   vk::Format         format = vk::Format::eB8G8R8A8Unorm);

    ~frame_transfer();

    frame_transfer(const frame_transfer&)            = delete;
    frame_transfer& operator=(const frame_transfer&) = delete;

    transfer_mode mode() const { return mode_; }

    // Transfer the source image to the destination device.
    // Returns the VkImage on the destination device ready for use.
    //
    // For same_device mode: returns src_image unchanged.
    // For external_memory: returns the imported image on dst_device.
    // For host_staging: copies src→host→dst, returns dst landing image.
    //
    // The caller must ensure src_image is in TRANSFER_SRC_OPTIMAL layout
    // before calling this. The returned image will be in TRANSFER_SRC_OPTIMAL.
    vk::Image transfer(vk::Image         src_image,
                       vk::CommandBuffer  src_cmd,
                       vk::Queue          src_queue,
                       vk::CommandBuffer  dst_cmd,
                       vk::Queue          dst_queue);

    // For same_device mode, just returns the source directly.
    // For cross-GPU modes, returns the landing image on dst_device.
    vk::Image dst_image() const { return dst_image_; }

    uint32_t width() const { return width_; }
    uint32_t height() const { return height_; }

  private:
    void probe_transfer_mode();
    void create_external_memory_path();
    void create_host_staging_path();

    vk::Device         src_device_;
    vk::PhysicalDevice src_physical_;
    vk::Device         dst_device_;
    vk::PhysicalDevice dst_physical_;
    uint32_t           dst_queue_family_;
    uint32_t           width_;
    uint32_t           height_;
    vk::Format         format_;
    transfer_mode      mode_ = transfer_mode::same_device;

    // External memory path
    vk::Image          dst_image_      = nullptr;
    vk::DeviceMemory   dst_memory_     = nullptr;
#ifdef _WIN32
    HANDLE             shared_handle_  = nullptr;
#endif

    // Host staging path
    vk::Buffer       src_staging_buffer_ = nullptr;
    vk::DeviceMemory src_staging_memory_ = nullptr;
    vk::Buffer       dst_staging_buffer_ = nullptr;
    vk::DeviceMemory dst_staging_memory_ = nullptr;
    vk::Image        dst_landing_image_  = nullptr;
    vk::DeviceMemory dst_landing_memory_ = nullptr;

    // Sync
    vk::Fence src_fence_ = nullptr;
    vk::Fence dst_fence_ = nullptr;
};

// Check if two physical devices are the same GPU (LUID comparison).
bool same_physical_device(vk::PhysicalDevice a, vk::PhysicalDevice b);

}} // namespace caspar::vulkan_output
