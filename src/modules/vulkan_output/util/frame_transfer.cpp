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

#include "frame_transfer.h"

#include "platform_handles.h"

#include <common/log.h>

#include <cstring>

namespace caspar { namespace vulkan_output {

// ─── LUID comparison ────────────────────────────────────────────────────────

bool same_physical_device(vk::PhysicalDevice a, vk::PhysicalDevice b)
{
    if (a == b)
        return true;

    vk::PhysicalDeviceIDProperties id_a{};
    vk::PhysicalDeviceProperties2  props_a{};
    props_a.pNext = &id_a;
    a.getProperties2(&props_a);

    vk::PhysicalDeviceIDProperties id_b{};
    vk::PhysicalDeviceProperties2  props_b{};
    props_b.pNext = &id_b;
    b.getProperties2(&props_b);

    if (id_a.deviceLUIDValid && id_b.deviceLUIDValid)
        return std::memcmp(id_a.deviceLUID, id_b.deviceLUID, VK_LUID_SIZE) == 0;

    // Fallback: compare device UUIDs
    return std::memcmp(id_a.deviceUUID, id_b.deviceUUID, VK_UUID_SIZE) == 0;
}

// ─── Construction / Destruction ─────────────────────────────────────────────

frame_transfer::frame_transfer(vk::Device         src_device,
                               vk::PhysicalDevice src_physical,
                               vk::Device         dst_device,
                               vk::PhysicalDevice dst_physical,
                               uint32_t           dst_queue_family,
                               uint32_t           width,
                               uint32_t           height,
                               vk::Format         format)
    : src_device_(src_device)
    , src_physical_(src_physical)
    , dst_device_(dst_device)
    , dst_physical_(dst_physical)
    , dst_queue_family_(dst_queue_family)
    , width_(width)
    , height_(height)
    , format_(format)
{
    if (src_device_ == dst_device_ || same_physical_device(src_physical_, dst_physical_)) {
        mode_ = transfer_mode::same_device;
        CASPAR_LOG(info) << L"[frame_transfer] Same device — zero-copy passthrough.";
        return;
    }

    probe_transfer_mode();

    if (mode_ == transfer_mode::external_memory) {
        create_external_memory_path();
        CASPAR_LOG(info) << L"[frame_transfer] Cross-GPU via external memory (NVLink-accelerated if connected).";
    } else {
        create_host_staging_path();
        CASPAR_LOG(info) << L"[frame_transfer] Cross-GPU via host staging (PCIe bandwidth limited).";
    }
}

frame_transfer::~frame_transfer()
{
    if (mode_ == transfer_mode::same_device)
        return;

    if (src_fence_)
        src_device_.destroyFence(src_fence_);
    if (dst_fence_)
        dst_device_.destroyFence(dst_fence_);

    if (mode_ == transfer_mode::external_memory) {
        if (dst_image_)
            dst_device_.destroyImage(dst_image_);
        if (dst_memory_)
            dst_device_.freeMemory(dst_memory_);
        platform::close_handle(shared_handle_);
    } else {
        if (src_staging_buffer_)
            src_device_.destroyBuffer(src_staging_buffer_);
        if (src_staging_memory_)
            src_device_.freeMemory(src_staging_memory_);
        if (dst_staging_buffer_)
            dst_device_.destroyBuffer(dst_staging_buffer_);
        if (dst_staging_memory_)
            dst_device_.freeMemory(dst_staging_memory_);
        if (dst_landing_image_)
            dst_device_.destroyImage(dst_landing_image_);
        if (dst_landing_memory_)
            dst_device_.freeMemory(dst_landing_memory_);
    }
}

// ─── Transfer execution ─────────────────────────────────────────────────────

vk::Image frame_transfer::transfer(vk::Image         src_image,
                                   vk::CommandBuffer  src_cmd,
                                   vk::Queue          src_queue,
                                   vk::CommandBuffer  dst_cmd,
                                   vk::Queue          dst_queue)
{
    if (mode_ == transfer_mode::same_device)
        return src_image;

    if (mode_ == transfer_mode::external_memory) {
        // The imported image shares memory with the source — just return it.
        // Caller is responsible for proper synchronization (timeline semaphore
        // or fence between source render and dest present).
        return dst_image_;
    }

    // Host staging path: copy src → host buffer → dst buffer → dst image
    vk::DeviceSize image_size = static_cast<vk::DeviceSize>(width_) * height_ * 4; // BGRA8

    // Phase 1: Copy source image → source staging buffer (on src device)
    src_cmd.reset();
    src_cmd.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    vk::BufferImageCopy copy_region{};
    copy_region.bufferOffset      = 0;
    copy_region.bufferRowLength   = 0; // tightly packed
    copy_region.bufferImageHeight = 0;
    copy_region.imageSubresource  = vk::ImageSubresourceLayers{vk::ImageAspectFlagBits::eColor, 0, 0, 1};
    copy_region.imageOffset       = vk::Offset3D{0, 0, 0};
    copy_region.imageExtent       = vk::Extent3D{width_, height_, 1};

    src_cmd.copyImageToBuffer(src_image, vk::ImageLayout::eTransferSrcOptimal,
                              src_staging_buffer_, copy_region);
    src_cmd.end();

    src_device_.resetFences(src_fence_);
    vk::SubmitInfo src_submit{};
    src_submit.commandBufferCount = 1;
    src_submit.pCommandBuffers    = &src_cmd;
    src_queue.submit(src_submit, src_fence_);
    auto wait_result = src_device_.waitForFences(src_fence_, VK_TRUE, UINT64_MAX);
    (void)wait_result;

    // Phase 2: Memcpy host-visible src buffer → host-visible dst buffer
    void* src_mapped = src_device_.mapMemory(src_staging_memory_, 0, image_size);
    void* dst_mapped = dst_device_.mapMemory(dst_staging_memory_, 0, image_size);
    std::memcpy(dst_mapped, src_mapped, static_cast<size_t>(image_size));
    src_device_.unmapMemory(src_staging_memory_);
    dst_device_.unmapMemory(dst_staging_memory_);

    // Phase 3: Copy dst staging buffer → dst landing image (on dst device)
    dst_cmd.reset();
    dst_cmd.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    // Transition landing image: undefined → transfer dst
    vk::ImageMemoryBarrier bar_to_dst{};
    bar_to_dst.dstAccessMask    = vk::AccessFlagBits::eTransferWrite;
    bar_to_dst.oldLayout        = vk::ImageLayout::eUndefined;
    bar_to_dst.newLayout        = vk::ImageLayout::eTransferDstOptimal;
    bar_to_dst.image            = dst_landing_image_;
    bar_to_dst.subresourceRange = vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
    dst_cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
                            vk::PipelineStageFlagBits::eTransfer,
                            {}, nullptr, nullptr, bar_to_dst);

    dst_cmd.copyBufferToImage(dst_staging_buffer_, dst_landing_image_,
                              vk::ImageLayout::eTransferDstOptimal, copy_region);

    // Transition landing image: transfer dst → transfer src (ready for blit)
    vk::ImageMemoryBarrier bar_to_src{};
    bar_to_src.srcAccessMask    = vk::AccessFlagBits::eTransferWrite;
    bar_to_src.dstAccessMask    = vk::AccessFlagBits::eTransferRead;
    bar_to_src.oldLayout        = vk::ImageLayout::eTransferDstOptimal;
    bar_to_src.newLayout        = vk::ImageLayout::eTransferSrcOptimal;
    bar_to_src.image            = dst_landing_image_;
    bar_to_src.subresourceRange = vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
    dst_cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                            vk::PipelineStageFlagBits::eTransfer,
                            {}, nullptr, nullptr, bar_to_src);

    dst_cmd.end();

    dst_device_.resetFences(dst_fence_);
    vk::SubmitInfo dst_submit{};
    dst_submit.commandBufferCount = 1;
    dst_submit.pCommandBuffers    = &dst_cmd;
    dst_queue.submit(dst_submit, dst_fence_);
    auto dst_wait = dst_device_.waitForFences(dst_fence_, VK_TRUE, UINT64_MAX);
    (void)dst_wait;

    return dst_landing_image_;
}

// ─── Private: Probe transfer capabilities ───────────────────────────────────

void frame_transfer::probe_transfer_mode()
{
    // Check if dst device supports importing external memory from src device
    auto dst_ext_props = dst_physical_.enumerateDeviceExtensionProperties();

    bool has_external_memory = false;
    for (const auto& ext : dst_ext_props) {
        if (std::string(ext.extensionName.data()) == platform::kExtMemExtName)
            has_external_memory = true;
    }

    // Also need the src device to support exporting
    auto src_ext_props = src_physical_.enumerateDeviceExtensionProperties();
    bool src_can_export = false;
    for (const auto& ext : src_ext_props) {
        if (std::string(ext.extensionName.data()) == platform::kExtMemExtName)
            src_can_export = true;
    }

    if (has_external_memory && src_can_export)
        mode_ = transfer_mode::external_memory;
    else
        mode_ = transfer_mode::host_staging;
}

// ─── Private: External memory path ──────────────────────────────────────────

void frame_transfer::create_external_memory_path()
{
    // Create an image on dst_device backed by memory imported from src_device.
    // The actual GPU-to-GPU transfer happens via the Vulkan driver's DMA engine
    // which uses NVLink when available between the two GPUs.

    // For the external memory path to work in practice, the SOURCE image needs
    // to be allocated with VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT.
    // Since the niklas accelerator's texture allocator doesn't currently do this,
    // we fall back to host staging for now. This path is a placeholder that will
    // become functional when the accelerator exports its textures.
    //
    // TODO: When accelerator::vulkan::texture supports exportable allocation,
    // implement the import here:
    //   1. Get the HANDLE from src_device via vkGetMemoryWin32HandleKHR
    //   2. Import on dst_device via VkImportMemoryWin32HandleInfoKHR
    //   3. Bind to dst_image_

    CASPAR_LOG(warning) << L"[frame_transfer] External memory path not yet functional "
                           L"(accelerator textures not exportable). Falling back to host staging.";
    mode_ = transfer_mode::host_staging;
    create_host_staging_path();
}

// ─── Private: Host staging path ─────────────────────────────────────────────

namespace {

uint32_t find_memory_type(vk::PhysicalDevice physical, uint32_t type_bits, vk::MemoryPropertyFlags props)
{
    auto mem_props = physical.getMemoryProperties();
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; ++i) {
        if ((type_bits & (1u << i)) && (mem_props.memoryTypes[i].propertyFlags & props) == props)
            return i;
    }
    return 0;
}

} // namespace

void frame_transfer::create_host_staging_path()
{
    vk::DeviceSize image_size = static_cast<vk::DeviceSize>(width_) * height_ * 4; // BGRA8

    // Source staging buffer (host-visible, on src device)
    {
        vk::BufferCreateInfo ci{};
        ci.size  = image_size;
        ci.usage = vk::BufferUsageFlagBits::eTransferDst;
        src_staging_buffer_ = src_device_.createBuffer(ci);

        auto reqs = src_device_.getBufferMemoryRequirements(src_staging_buffer_);
        vk::MemoryAllocateInfo alloc{};
        alloc.allocationSize  = reqs.size;
        alloc.memoryTypeIndex = find_memory_type(src_physical_, reqs.memoryTypeBits,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
        src_staging_memory_ = src_device_.allocateMemory(alloc);
        src_device_.bindBufferMemory(src_staging_buffer_, src_staging_memory_, 0);
    }

    // Destination staging buffer (host-visible, on dst device)
    {
        vk::BufferCreateInfo ci{};
        ci.size  = image_size;
        ci.usage = vk::BufferUsageFlagBits::eTransferSrc;
        dst_staging_buffer_ = dst_device_.createBuffer(ci);

        auto reqs = dst_device_.getBufferMemoryRequirements(dst_staging_buffer_);
        vk::MemoryAllocateInfo alloc{};
        alloc.allocationSize  = reqs.size;
        alloc.memoryTypeIndex = find_memory_type(dst_physical_, reqs.memoryTypeBits,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
        dst_staging_memory_ = dst_device_.allocateMemory(alloc);
        dst_device_.bindBufferMemory(dst_staging_buffer_, dst_staging_memory_, 0);
    }

    // Landing image on dst device (receives uploaded pixels)
    {
        vk::ImageCreateInfo ci{};
        ci.imageType   = vk::ImageType::e2D;
        ci.format      = format_;
        ci.extent      = vk::Extent3D{width_, height_, 1};
        ci.mipLevels   = 1;
        ci.arrayLayers = 1;
        ci.samples     = vk::SampleCountFlagBits::e1;
        ci.tiling      = vk::ImageTiling::eOptimal;
        ci.usage       = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eTransferSrc;
        ci.sharingMode = vk::SharingMode::eExclusive;
        dst_landing_image_ = dst_device_.createImage(ci);

        auto reqs = dst_device_.getImageMemoryRequirements(dst_landing_image_);
        vk::MemoryAllocateInfo alloc{};
        alloc.allocationSize  = reqs.size;
        alloc.memoryTypeIndex = find_memory_type(dst_physical_, reqs.memoryTypeBits,
            vk::MemoryPropertyFlagBits::eDeviceLocal);
        dst_landing_memory_ = dst_device_.allocateMemory(alloc);
        dst_device_.bindImageMemory(dst_landing_image_, dst_landing_memory_, 0);
    }

    dst_image_ = dst_landing_image_;

    // Fences for synchronizing transfers
    src_fence_ = src_device_.createFence({});
    dst_fence_ = dst_device_.createFence({});
}

}} // namespace caspar::vulkan_output
