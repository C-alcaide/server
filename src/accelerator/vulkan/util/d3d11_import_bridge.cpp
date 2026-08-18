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

#include "d3d11_import_bridge.h"

#ifdef _WIN32

#include "device.h"
#include "texture.h"

#include <common/log.h>

#include <array>
#include <chrono>
#include <cstring>
#include <vector>


#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_win32.h>

namespace caspar { namespace accelerator { namespace vulkan {

namespace {

/// A D3D11 shared texture must be imported through one of the two
/// D3D11-specific handle types. The opaque type in platform_handles.h is for
/// memory Vulkan itself exported and is deliberately not used here: asked to
/// import a real shared D3D11 NV12 texture, `eOpaqueWin32` fails with
/// `vkGetMemoryWin32HandleProperties` = ErrorInitializationFailed and
/// `vkAllocateMemory` = ErrorOutOfDeviceMemory, even though
/// `getImageFormatProperties2` reports the combination as supported. A
/// capability query is not a feasibility test.
constexpr vk::ExternalMemoryHandleTypeFlagBits kD3D11HandleType =
    vk::ExternalMemoryHandleTypeFlagBits::eD3D11Texture;

/// The plane formats, chosen to be identical on both sides of the copy so it is
/// a straight memory move with no format conversion anywhere. `stride` is the
/// component count the mixer's texture pool indexes by: 1 for Y, 2 for CbCr.
vk::Format plane_format(int stride, common::bit_depth depth)
{
    if (depth == common::bit_depth::bit8)
        return stride == 1 ? vk::Format::eR8Unorm : vk::Format::eR8G8Unorm;
    return stride == 1 ? vk::Format::eR16Unorm : vk::Format::eR16G16Unorm;
}

} // namespace

struct d3d11_import_bridge::impl
{
    /// One imported D3D11 plane texture. The import is cached: the producer
    /// reuses the same two D3D11 textures for every frame, so re-importing per
    /// frame would allocate and free device memory 50 times a second for
    /// nothing.
    struct imported
    {
        void*            handle = nullptr;
        vk::Image        image;
        vk::DeviceMemory memory;
        int              width  = 0;
        int              height = 0;
        vk::Format       format = vk::Format::eUndefined;

        bool matches(void* h, int w, int ht, vk::Format f) const
        {
            return image && handle == h && width == w && height == ht && format == f;
        }
    };

    device*   dev_ = nullptr;
    vk::Device vk_device_;


    /// Imports for copy_texture(). A ring rather than a single slot because a caller
    /// typically rotates through a few staging textures to keep the GPU busy; sized to
    /// cover that without growing without bound if a caller does hand over fresh handles.
    static constexpr std::size_t kTexCacheSize = 8;
    std::vector<imported>        tex_cache_;
    std::size_t                  tex_cache_next_ = 0;

    vk::CommandBuffer cmd_;
    vk::Fence         fence_;
    bool              copy_pending_ = false;

    explicit impl(device* dev)
        : dev_(dev)
        , vk_device_(dev->getVkDevice())
    {
        // Allocated once, outside any dispatch: allocateCommandBuffers hops to
        // the device thread itself, and doing it per frame would defeat that
        // thread's own command-buffer recycling.
        cmd_   = dev_->allocateCommandBuffers(1).front();
        fence_ = vk_device_.createFence(vk::FenceCreateInfo{});
    }

    ~impl()
    {
        // The copy may still be in flight; freeing the images underneath it
        // would be a use-after-free on the GPU.
        wait_for_previous_copy();

        if (fence_)
            vk_device_.destroyFence(fence_);
        for (auto& im : tex_cache_)
            release(im);
        // cmd_ belongs to the device's command pool and is reclaimed with it.
    }

    /// The cached import for `handle`, importing it if this is the first sighting.
    /// Returns nullptr on failure.
    imported* find_or_import(void* handle, int width, int height, vk::Format format)
    {
        for (auto& im : tex_cache_) {
            if (im.matches(handle, width, height, format))
                return &im;
        }

        // A handle we have seen before but at a different size (the page resized)
        // has a stale import; drop it rather than adding a second entry for the
        // same handle.
        for (auto& im : tex_cache_) {
            if (im.handle == handle) {
                release(im);
                if (!ensure_import(im, handle, width, height, format))
                    return nullptr;
                return &im;
            }
        }

        if (tex_cache_.size() < kTexCacheSize) {
            tex_cache_.emplace_back();
            if (!ensure_import(tex_cache_.back(), handle, width, height, format)) {
                tex_cache_.pop_back();
                return nullptr;
            }
            return &tex_cache_.back();
        }

        // Full: evict round-robin. Safe because wait_for_previous_copy() has
        // already run, so nothing in flight is reading the victim.
        auto& victim = tex_cache_[tex_cache_next_];
        tex_cache_next_ = (tex_cache_next_ + 1) % kTexCacheSize;
        release(victim);
        if (!ensure_import(victim, handle, width, height, format))
            return nullptr;
        return &victim;
    }

    void release(imported& im)
    {
        if (im.image) {
            vk_device_.destroyImage(im.image);
            im.image = nullptr;
        }
        if (im.memory) {
            vk_device_.freeMemory(im.memory);
            im.memory = nullptr;
        }
        im = imported{};
    }

    std::int64_t wait_for_previous_copy()
    {
        if (!copy_pending_)
            return 0;

        const auto start = std::chrono::steady_clock::now();
        // A one-second cap rather than UINT64_MAX: a lost submit must not wedge
        // the producer's decode thread forever, it must fall back.
        const auto res = vk_device_.waitForFences(fence_, VK_TRUE, 1'000'000'000ull);
        copy_pending_  = false;
        if (res != vk::Result::eSuccess)
            CASPAR_LOG(warning) << L"[vk::d3d11_import] timed out waiting for the previous plane copy";

        return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count();
    }

    /// Imports `handle` as a single-plane VkImage, or reuses the cached import
    /// if nothing has changed. Must run on the device thread.
    bool ensure_import(imported& im, void* handle, int width, int height, vk::Format format)
    {
        if (im.matches(handle, width, height, format))
            return true;

        release(im);

        vk::ExternalMemoryImageCreateInfo ext{};
        ext.handleTypes = kD3D11HandleType;

        vk::ImageCreateInfo ii{};
        ii.pNext         = &ext;
        ii.imageType     = vk::ImageType::e2D;
        ii.format        = format;
        ii.extent        = vk::Extent3D{static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1};
        ii.mipLevels     = 1;
        ii.arrayLayers   = 1;
        ii.samples       = vk::SampleCountFlagBits::e1;
        ii.tiling        = vk::ImageTiling::eOptimal;
        ii.usage         = vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eSampled;
        ii.sharingMode   = vk::SharingMode::eExclusive;
        ii.initialLayout = vk::ImageLayout::eUndefined;

        try {
            im.image = vk_device_.createImage(ii);

            const auto req = vk_device_.getImageMemoryRequirements(im.image);

            // Which memory types can actually back this handle. Asking the
            // handle rather than only the image is what catches an unsupported
            // handle type here instead of as a driver error later.
            //
            // Resolved through getProcAddr and the C structs rather than
            // vulkan.hpp's Win32 wrappers, which is how the rest of this
            // directory does it: VK_USE_PLATFORM_WIN32_KHR is deliberately not
            // defined for vulkan.hpp in this build.
            uint32_t type_bits = req.memoryTypeBits;
            {
                auto pfn = reinterpret_cast<PFN_vkGetMemoryWin32HandlePropertiesKHR>(
                    vk_device_.getProcAddr("vkGetMemoryWin32HandlePropertiesKHR"));
                if (!pfn) {
                    CASPAR_LOG(warning) << L"[vk::d3d11_import] vkGetMemoryWin32HandlePropertiesKHR unavailable";
                    release(im);
                    return false;
                }

                VkMemoryWin32HandlePropertiesKHR props{};
                props.sType = VK_STRUCTURE_TYPE_MEMORY_WIN32_HANDLE_PROPERTIES_KHR;
                const auto r =
                    pfn(static_cast<VkDevice>(vk_device_),
                        static_cast<VkExternalMemoryHandleTypeFlagBits>(kD3D11HandleType), handle, &props);
                if (r != VK_SUCCESS) {
                    CASPAR_LOG(warning) << L"[vk::d3d11_import] this D3D11 handle is not importable (result=" << r
                                        << L")";
                    release(im);
                    return false;
                }
                type_bits &= props.memoryTypeBits;
            }

            const auto mem_props = dev_->getMemoryProperties();
            uint32_t   type_index = UINT32_MAX;
            for (uint32_t i = 0; i < mem_props.memoryTypeCount; ++i) {
                if ((type_bits & (1u << i)) &&
                    (mem_props.memoryTypes[i].propertyFlags & vk::MemoryPropertyFlagBits::eDeviceLocal)) {
                    type_index = i;
                    break;
                }
            }
            if (type_index == UINT32_MAX) {
                CASPAR_LOG(warning) << L"[vk::d3d11_import] no device-local memory type accepts this D3D11 handle";
                release(im);
                return false;
            }

            // A D3D11 texture is always a dedicated allocation; importing it
            // without saying so is rejected.
            vk::MemoryDedicatedAllocateInfo dedicated{};
            dedicated.image = im.image;

            VkImportMemoryWin32HandleInfoKHR import{};
            import.sType      = VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_KHR;
            import.pNext      = &dedicated;
            import.handleType = static_cast<VkExternalMemoryHandleTypeFlagBits>(kD3D11HandleType);
            import.handle     = handle;

            vk::MemoryAllocateInfo alloc{};
            alloc.pNext           = &import;
            alloc.allocationSize  = req.size;
            alloc.memoryTypeIndex = type_index;

            im.memory = vk_device_.allocateMemory(alloc);
            vk_device_.bindImageMemory(im.image, im.memory, 0);
        } catch (const vk::SystemError& e) {
            CASPAR_LOG(warning) << L"[vk::d3d11_import] import failed: " << u16(e.what());
            release(im);
            return false;
        }

        im.handle = handle;
        im.width  = width;
        im.height = height;
        im.format = format;

        // One transition out of eUndefined, now, before D3D11 has written
        // anything worth keeping. From here the image stays in eGeneral for
        // good: D3D11 writes it outside Vulkan's knowledge, so any later
        // transition would either be a lie about the previous layout or would
        // discard the picture D3D11 just put there.
        dev_->dispatch_sync([&] {
            cmd_.reset();
            cmd_.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

            vk::ImageMemoryBarrier2 b{};
            b.oldLayout           = vk::ImageLayout::eUndefined;
            b.newLayout           = vk::ImageLayout::eGeneral;
            b.srcQueueFamilyIndex = vk::QueueFamilyIgnored;
            b.dstQueueFamilyIndex = vk::QueueFamilyIgnored;
            b.image               = im.image;
            b.subresourceRange    = vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);
            b.srcStageMask        = vk::PipelineStageFlagBits2::eTopOfPipe;
            b.dstStageMask        = vk::PipelineStageFlagBits2::eTransfer;
            b.dstAccessMask       = vk::AccessFlagBits2::eTransferRead;

            vk::DependencyInfo dep;
            dep.setImageMemoryBarriers(b);
            cmd_.pipelineBarrier2(dep);
            cmd_.end();

            vk::SubmitInfo si{};
            si.setCommandBuffers(cmd_);
            vk_device_.resetFences(fence_);
            dev_->submit(si, fence_);
        });
        copy_pending_ = true;
        wait_for_previous_copy();

        CASPAR_LOG(debug) << L"[vk::d3d11_import] imported D3D11 plane " << width << L"x" << height
                          << L" as " << u16(vk::to_string(format));
        return true;
    }
};

d3d11_import_bridge::d3d11_import_bridge(void* vk_device)
{
    if (!vk_device)
        CASPAR_THROW_EXCEPTION(caspar_exception() << msg_info("d3d11_import_bridge needs a Vulkan device"));
    impl_ = std::make_unique<impl>(static_cast<device*>(vk_device));
}

d3d11_import_bridge::~d3d11_import_bridge() = default;

std::int64_t d3d11_import_bridge::wait_for_previous_copy() { return impl_->wait_for_previous_copy(); }

void d3d11_import_bridge::release_imports()
{
    impl_->wait_for_previous_copy();
    for (auto& im : impl_->tex_cache_)
        impl_->release(im);
    impl_->tex_cache_.clear();
    impl_->tex_cache_next_ = 0;
}

bool d3d11_import_bridge::copy_texture(void* handle, int width, int height, std::shared_ptr<texture>& out)
{
    auto& m = *impl_;

    // Anything in flight is reading an import; it must finish before this copy is
    // recorded and before the caller reuses the source.
    m.wait_for_previous_copy();

    // Always eR8G8B8A8Unorm -- see the header. The mixer texture matches, so the
    // copy is a straight memory move and channel order is the frame's business.
    constexpr auto kFormat = vk::Format::eR8G8B8A8Unorm;

    std::shared_ptr<texture> tex;

    const bool ok = m.dev_->dispatch_sync([&]() -> bool {
        auto* src = m.find_or_import(handle, width, height, kFormat);
        if (!src)
            return false;

        // Copy into a pooled mixer texture rather than binding the import: the
        // source belongs to the caller's staging ring and has to be free again by
        // the next frame.
        tex = m.dev_->create_texture(width, height, 4, common::bit_depth::bit8);
        if (!tex)
            return false;

        try {
            m.cmd_.reset();
            m.cmd_.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

            const auto range = vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);

            std::array<vk::ImageMemoryBarrier2, 2> barriers{};
            // Source stays in eGeneral (see ensure_import) -- eGeneral to eGeneral
            // makes the D3D11 write visible to the transfer stage without
            // discarding it.
            {
                auto& b               = barriers[0];
                b.oldLayout           = vk::ImageLayout::eGeneral;
                b.newLayout           = vk::ImageLayout::eGeneral;
                b.srcQueueFamilyIndex = vk::QueueFamilyIgnored;
                b.dstQueueFamilyIndex = vk::QueueFamilyIgnored;
                b.image               = src->image;
                b.subresourceRange    = range;
                b.srcStageMask        = vk::PipelineStageFlagBits2::eAllCommands;
                b.srcAccessMask       = vk::AccessFlagBits2::eMemoryWrite;
                b.dstStageMask        = vk::PipelineStageFlagBits2::eTransfer;
                b.dstAccessMask       = vk::AccessFlagBits2::eTransferRead;
            }
            // Destination is overwritten whole, so eUndefined is honest.
            {
                auto& b               = barriers[1];
                b.oldLayout           = vk::ImageLayout::eUndefined;
                b.newLayout           = vk::ImageLayout::eTransferDstOptimal;
                b.srcQueueFamilyIndex = vk::QueueFamilyIgnored;
                b.dstQueueFamilyIndex = vk::QueueFamilyIgnored;
                b.image               = tex->id();
                b.subresourceRange    = range;
                b.srcStageMask        = vk::PipelineStageFlagBits2::eTopOfPipe;
                b.dstStageMask        = vk::PipelineStageFlagBits2::eTransfer;
                b.dstAccessMask       = vk::AccessFlagBits2::eTransferWrite;
            }
            {
                vk::DependencyInfo dep;
                dep.setImageMemoryBarriers(barriers);
                m.cmd_.pipelineBarrier2(dep);
            }

            const auto      layers = vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, 0, 1);
            vk::ImageCopy   c(layers, vk::Offset3D{}, layers, vk::Offset3D{},
                              vk::Extent3D{static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1});
            m.cmd_.copyImage(src->image, vk::ImageLayout::eGeneral, tex->id(),
                             vk::ImageLayout::eTransferDstOptimal, c);

            vk::ImageMemoryBarrier2 to_read{};
            to_read.oldLayout           = vk::ImageLayout::eTransferDstOptimal;
            to_read.newLayout           = vk::ImageLayout::eShaderReadOnlyOptimal;
            to_read.srcQueueFamilyIndex = vk::QueueFamilyIgnored;
            to_read.dstQueueFamilyIndex = vk::QueueFamilyIgnored;
            to_read.image               = tex->id();
            to_read.subresourceRange    = range;
            to_read.srcStageMask        = vk::PipelineStageFlagBits2::eTransfer;
            to_read.srcAccessMask       = vk::AccessFlagBits2::eTransferWrite;
            to_read.dstStageMask        = vk::PipelineStageFlagBits2::eFragmentShader;
            to_read.dstAccessMask       = vk::AccessFlagBits2::eShaderRead;
            {
                vk::DependencyInfo dep;
                dep.setImageMemoryBarriers(to_read);
                m.cmd_.pipelineBarrier2(dep);
            }

            m.cmd_.end();

            vk::SubmitInfo si{};
            si.setCommandBuffers(m.cmd_);
            m.vk_device_.resetFences(m.fence_);
            m.dev_->submit(si, m.fence_);
            m.copy_pending_ = true;
        } catch (const vk::SystemError& e) {
            CASPAR_LOG(warning) << L"[vk::d3d11_import] texture copy failed: " << u16(e.what());
            return false;
        }

        return true;
    });

    if (!ok)
        return false;

    // The pooled texture goes back as it is. CasparVP wraps it in a VkReadableTextureWrapper
    // so the imported picture can also be read back to host memory — for PRINT, and for
    // filters with no GPU path. That wrapper has nothing to carry here: `core::texture` in
    // this tree is `bind`/`unbind` and nothing else, so there is no readback to enable and a
    // wrapper would be an empty layer claiming otherwise.
    out = std::move(tex);
    return true;
}

}}} // namespace caspar::accelerator::vulkan

#endif // _WIN32
