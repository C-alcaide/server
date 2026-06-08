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

#include "vulkan_interop.h"
#include "vulkan_device.h"
#include "platform_handles.h"

#include <common/except.h>
#include <common/log.h>

#ifndef _WIN32
#include <unistd.h>
#endif

namespace caspar { namespace vulkan_output {

namespace {

#define VK_CHECK(call)                                                                                                 \
    do {                                                                                                               \
        VkResult result_ = (call);                                                                                     \
        if (result_ != VK_SUCCESS) {                                                                                   \
            CASPAR_THROW_EXCEPTION(caspar_exception()                                                                  \
                                   << msg_info("Vulkan interop call failed: " #call " = " + std::to_string(result_))); \
        }                                                                                                              \
    } while (0)

} // namespace

vulkan_interop::vulkan_interop(vulkan_device& device, uint32_t width, uint32_t height, VkFormat format)
    : device_(device)
    , width_(width)
    , height_(height)
    , format_(format)
{
}

vulkan_interop::~vulkan_interop()
{
    auto dev = device_.device();
    if (image_view_ != VK_NULL_HANDLE)
        vkDestroyImageView(dev, image_view_, nullptr);
    if (image_ != VK_NULL_HANDLE)
        vkDestroyImage(dev, image_, nullptr);
    if (memory_ != VK_NULL_HANDLE)
        vkFreeMemory(dev, memory_, nullptr);
#ifdef _WIN32
    if (shared_handle_)
        CloseHandle(shared_handle_);
#else
    if (shared_handle_ != platform::kInvalidHandle) {
        close(shared_handle_);
        shared_handle_ = platform::kInvalidHandle;
    }
#endif
}

void vulkan_interop::import_from_handle(platform::native_handle_t handle)
{
    shared_handle_ = handle;

    // Create VkImage with external memory
    VkExternalMemoryImageCreateInfo ext_mem_info{};
    ext_mem_info.sType       = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
    ext_mem_info.handleTypes = platform::kExternalMemoryHandleType;

    VkImageCreateInfo image_info{};
    image_info.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image_info.pNext         = &ext_mem_info;
    image_info.imageType     = VK_IMAGE_TYPE_2D;
    image_info.format        = format_;
    image_info.extent        = {width_, height_, 1};
    image_info.mipLevels     = 1;
    image_info.arrayLayers   = 1;
    image_info.samples       = VK_SAMPLE_COUNT_1_BIT;
    image_info.tiling        = VK_IMAGE_TILING_OPTIMAL;
    image_info.usage         = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    image_info.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
    image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VK_CHECK(vkCreateImage(device_.device(), &image_info, nullptr, &image_));

    // Get memory requirements
    VkMemoryRequirements mem_reqs;
    vkGetImageMemoryRequirements(device_.device(), image_, &mem_reqs);

    // Import the platform handle as device memory
#ifdef _WIN32
    VkImportMemoryWin32HandleInfoKHR import_info{};
    import_info.sType      = VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_KHR;
    import_info.handleType = platform::kExternalMemoryHandleType;
    import_info.handle     = handle;
#else
    VkImportMemoryFdInfoKHR import_info{};
    import_info.sType      = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR;
    import_info.handleType = platform::kExternalMemoryHandleType;
    import_info.fd         = handle;
#endif

    // Find a suitable memory type
    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(device_.physical_device(), &mem_props);

    uint32_t memory_type_index = UINT32_MAX;
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; ++i) {
        if ((mem_reqs.memoryTypeBits & (1u << i)) &&
            (mem_props.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
            memory_type_index = i;
            break;
        }
    }

    if (memory_type_index == UINT32_MAX)
        CASPAR_THROW_EXCEPTION(caspar_exception() << msg_info("No suitable memory type for external import"));

    VkMemoryAllocateInfo alloc_info{};
    alloc_info.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.pNext           = &import_info;
    alloc_info.allocationSize  = mem_reqs.size;
    alloc_info.memoryTypeIndex = memory_type_index;

    VK_CHECK(vkAllocateMemory(device_.device(), &alloc_info, nullptr, &memory_));

#ifndef _WIN32
    // vkAllocateMemory with VkImportMemoryFdInfoKHR consumes the fd on success —
    // mark as invalid to prevent double-close in destructor.
    shared_handle_ = platform::kInvalidHandle;
#endif
    VK_CHECK(vkBindImageMemory(device_.device(), image_, memory_, 0));

    // Create image view
    VkImageViewCreateInfo view_info{};
    view_info.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_info.image                           = image_;
    view_info.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
    view_info.format                          = format_;
    view_info.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    view_info.subresourceRange.baseMipLevel   = 0;
    view_info.subresourceRange.levelCount     = 1;
    view_info.subresourceRange.baseArrayLayer = 0;
    view_info.subresourceRange.layerCount     = 1;

    VK_CHECK(vkCreateImageView(device_.device(), &view_info, nullptr, &image_view_));

    CASPAR_LOG(debug) << L"[vulkan_output] Imported shared memory handle -> VkImage "
                      << width_ << L"x" << height_;
}

}} // namespace caspar::vulkan_output
