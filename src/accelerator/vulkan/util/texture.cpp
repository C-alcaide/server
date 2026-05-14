/*
 * Copyright 2025
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
 *
 * Author: Niklas Andersson, niklas@niklaspandersson.se
 */

#include "texture.h"
#include "buffer.h"

#include <common/bit_depth.h>
#include <common/log.h>
#include <vulkan/vulkan.hpp>

#ifdef _WIN32
#include <vulkan/vulkan_win32.h>
#endif

namespace caspar { namespace accelerator { namespace vulkan {

struct texture::impl
{
    vk::Image         image_;
    vk::DeviceMemory  memory_;
    vk::ImageView     imageView_;
    vk::Device        device_;
    int               width_  = 0;
    int               height_ = 0;
    int               stride_ = 0;
    int               size_   = 0;
    vk::DeviceSize    alloc_size_ = 0;
    common::bit_depth depth_;
    uint8_t           device_luid_[8] = {};
    bool              has_luid_ = false;
    mutable HANDLE    win32_handle_ = nullptr; // Cached external memory handle

    impl(const impl&)            = delete;
    impl& operator=(const impl&) = delete;

  public:
    impl(int               width,
         int               height,
         int               stride,
         common::bit_depth depth,
         vk::Image         image,
         vk::DeviceMemory  memory,
         vk::ImageView     imageView,
         vk::Device        device,
         vk::DeviceSize    alloc_size = 0)
        : image_(image)
        , memory_(memory)
        , imageView_(imageView)
        , device_(device)
        , width_(width)
        , height_(height)
        , stride_(stride)
        , size_(width * height * stride * (depth == common::bit_depth::bit8 ? 1 : 2))
        , alloc_size_(alloc_size > 0 ? alloc_size : static_cast<vk::DeviceSize>(size_))
        , depth_(depth)
    {
    }

    ~impl()
    {
        if (win32_handle_) {
            CloseHandle(win32_handle_);
            win32_handle_ = nullptr;
        }
        device_.destroyImageView(imageView_);
        device_.freeMemory(memory_);
        device_.destroyImage(image_);
    }
};

texture::texture(int               width,
                 int               height,
                 int               stride,
                 common::bit_depth depth,
                 vk::Image         image,
                 vk::DeviceMemory  memory,
                 vk::ImageView     imageView,
                 vk::Device        device,
                 vk::DeviceSize    alloc_size)
    : impl_(new impl(width, height, stride, depth, image, memory, imageView, device, alloc_size))
{
}
texture::texture(texture&& other)
    : impl_(std::move(other.impl_))
{
}
texture::~texture() {}
texture& texture::operator=(texture&& other)
{
    impl_ = std::move(other.impl_);
    return *this;
}

vk::ImageView texture::view() const { return impl_->imageView_; }

int               texture::width() const { return impl_->width_; }
int               texture::height() const { return impl_->height_; }
int               texture::stride() const { return impl_->stride_; }
common::bit_depth texture::depth() const { return impl_->depth_; }
void              texture::set_depth(common::bit_depth depth) { impl_->depth_ = depth; }
int               texture::size() const { return impl_->size_; }
VkImage           texture::id() const { return impl_->image_; }
VkDeviceMemory    texture::memory() const { return impl_->memory_; }
vk::DeviceSize    texture::alloc_size() const { return impl_->alloc_size_; }

const uint8_t* texture::device_luid() const
{
    return impl_->has_luid_ ? impl_->device_luid_ : nullptr;
}

void texture::set_device_luid(const uint8_t* luid)
{
    if (luid) {
        std::memcpy(impl_->device_luid_, luid, 8);
        impl_->has_luid_ = true;
    }
}

HANDLE texture::export_win32_handle() const
{
    if (impl_->win32_handle_)
        return impl_->win32_handle_;

    auto pfn = reinterpret_cast<PFN_vkGetMemoryWin32HandleKHR>(
        impl_->device_.getProcAddr("vkGetMemoryWin32HandleKHR"));
    if (!pfn) {
        CASPAR_LOG(debug) << L"[vulkan] vkGetMemoryWin32HandleKHR not available";
        return nullptr;
    }

    VkMemoryGetWin32HandleInfoKHR handleInfo{};
    handleInfo.sType      = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
    handleInfo.memory     = impl_->memory_;
    handleInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;

    HANDLE handle = nullptr;
    VkResult result = pfn(static_cast<VkDevice>(impl_->device_), &handleInfo, &handle);
    if (result != VK_SUCCESS || !handle) {
        CASPAR_LOG(debug) << L"[vulkan] Failed to export Win32 handle (result=" << result << L")";
        return nullptr;
    }

    impl_->win32_handle_ = handle;
    return handle;
}

}}} // namespace caspar::accelerator::vulkan
