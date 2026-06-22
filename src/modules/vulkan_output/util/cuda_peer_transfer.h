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

#ifdef CASPAR_CUDA_P2P_ENABLED

#include "platform_handles.h"

#include <vulkan/vulkan.hpp>

#include <cstdint>

// Forward declarations (avoid pulling CUDA headers into every TU)
struct CUstream_st;
typedef CUstream_st* cudaStream_t;

namespace caspar { namespace vulkan_output {

/**
 * GPU-to-GPU frame transfer using CUDA peer copy over NVLink/PCIe.
 *
 * Uses cudaMemcpyPeerAsync which leverages the dedicated GPU copy engine
 * for direct device-to-device DMA. When GPUs are NVLink-connected, this
 * provides up to 900 GB/s bidirectional bandwidth (NVLink 4).
 *
 * Architecture (Vulkan-native):
 *   1. Source VkImage → vkCmdCopyImageToBuffer (device-local buffer with exportable memory)
 *   2. CUDA imports both src and dst buffers via external memory handles
 *   3. cudaMemcpyPeerAsync: GPU A device memory → GPU B device memory (NVLink DMA)
 *   4. Destination VkBuffer → vkCmdCopyBufferToImage (landing image on dst device)
 *
 * The CUDA peer copy runs on a dedicated stream and can overlap with
 * graphics/compute work on both GPUs.
 */
class cuda_peer_transfer
{
  public:
    /// Construct a CUDA P2P transfer path between two Vulkan devices.
    /// Both devices must be NVIDIA GPUs visible to the CUDA runtime.
    cuda_peer_transfer(vk::Device         src_device,
                       vk::PhysicalDevice src_physical,
                       vk::Device         dst_device,
                       vk::PhysicalDevice dst_physical,
                       uint32_t           width,
                       uint32_t           height,
                       vk::Format         format = vk::Format::eB8G8R8A8Unorm);

    ~cuda_peer_transfer();

    cuda_peer_transfer(const cuda_peer_transfer&)            = delete;
    cuda_peer_transfer& operator=(const cuda_peer_transfer&) = delete;

    /// Returns true if CUDA P2P is available between the two physical devices.
    /// Call this before constructing to check feasibility.
    static bool is_available(vk::PhysicalDevice src, vk::PhysicalDevice dst);

    /// Returns true if peer access is direct (NVLink), false if routed through host.
    bool is_nvlink() const { return nvlink_; }

    /// Perform the peer copy. The source VkImage must be in TRANSFER_SRC_OPTIMAL.
    /// Returns the destination VkImage in TRANSFER_SRC_OPTIMAL layout.
    vk::Image transfer(vk::Image        src_image,
                       vk::CommandBuffer src_cmd,
                       vk::Queue         src_queue,
                       vk::CommandBuffer dst_cmd,
                       vk::Queue         dst_queue);

    /// The landing image on the destination device.
    vk::Image dst_image() const { return dst_landing_image_; }

  private:
    void create_exportable_buffers();
    void setup_cuda_peer_access();

    vk::Device         src_device_;
    vk::PhysicalDevice src_physical_;
    vk::Device         dst_device_;
    vk::PhysicalDevice dst_physical_;
    uint32_t           width_;
    uint32_t           height_;
    vk::Format         format_;
    vk::DeviceSize     image_size_;

    // Source side: device-local buffer with exportable memory
    vk::Buffer       src_buffer_ = nullptr;
    vk::DeviceMemory src_memory_ = nullptr;

    // Destination side: device-local buffer with exportable memory + landing image
    vk::Buffer       dst_buffer_        = nullptr;
    vk::DeviceMemory dst_memory_        = nullptr;
    vk::Image        dst_landing_image_ = nullptr;
    vk::DeviceMemory dst_image_memory_  = nullptr;

    // CUDA state
    int           src_cuda_device_ = -1;
    int           dst_cuda_device_ = -1;
    void*         src_cuda_ptr_    = nullptr;  // Device pointer on GPU A
    void*         dst_cuda_ptr_    = nullptr;  // Device pointer on GPU B
    cudaStream_t  copy_stream_     = nullptr;
    bool          nvlink_          = false;
    bool          peer_access_     = false;

    // Vulkan sync
    vk::Fence src_fence_ = nullptr;
    vk::Fence dst_fence_ = nullptr;
};

}} // namespace caspar::vulkan_output

#endif // CASPAR_CUDA_P2P_ENABLED
