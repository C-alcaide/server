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

#include "av_vulkan_export.h"
#include "gpu_wait.h"

#include "device.h"
#include "texture.h"
#include "texture_wrapper.h"

#include <common/except.h>
#include <common/log.h>

#include <exception>

#include <vulkan/vulkan.hpp>

namespace caspar { namespace accelerator { namespace vulkan {

struct av_vulkan_exporter::impl
{
    device*           dev_ = nullptr;
    vk::Device        vk_device_;
    vk::CommandBuffer cmd_;
    vk::Fence         fence_;
    bool              copy_pending_ = false;
    bool              warned_       = false;

    explicit impl(device* dev)
        : dev_(dev)
        , vk_device_(dev->getVkDevice())
    {
        // Allocated once, outside any dispatch, for the same reasons as the importer:
        // allocateCommandBuffers hops to the device thread itself, and doing it per frame
        // would both leak the buffer and defeat that thread's own recycling.
        cmd_   = dev_->allocateCommandBuffers(1).front();
        fence_ = vk_device_.createFence(vk::FenceCreateInfo{});
    }

    ~impl()
    {
        // The copy may still be reading the mixer's attachment and writing FFmpeg's image;
        // tearing the fence down under it is a use-after-free on the GPU.
        wait_for_previous_copy();
        if (fence_)
            vk_device_.destroyFence(fence_);
    }

    void wait_for_previous_copy()
    {
        if (!copy_pending_ || !fence_)
            return;
        // Never returns on a timeout -- see gpu_wait.h for the TDR that taught this. Reusing
        // a command buffer whose submission is still executing is undefined behaviour.
        wait_for_fence(vk_device_, fence_, L"[vk::av_export] previous frame copy");
        copy_pending_ = false;
    }

    /// Log the first failure of a kind and then stay quiet: this runs per frame, and a
    /// per-frame warning turns a recording into an unreadable log.
    void warn_once(const std::wstring& what)
    {
        if (warned_)
            return;
        warned_ = true;
        CASPAR_LOG(warning) << L"[vk::av_export] " << what
                            << L" -- falling back to the host path for this recording";
    }
};

av_vulkan_exporter::av_vulkan_exporter(void* vk_device)
{
    auto* dev = static_cast<device*>(vk_device);
    if (!dev)
        CASPAR_THROW_EXCEPTION(caspar_exception() << msg_info("av_vulkan_exporter needs a Vulkan device"));
    impl_ = std::make_unique<impl>(dev);
}

av_vulkan_exporter::~av_vulkan_exporter() = default;

bool av_vulkan_exporter::copy_from_texture(const std::shared_ptr<core::texture>& src, av_plane_dest& dest)
{
    auto& m = *impl_;

    if (!dest.image || !dest.semaphore || dest.width <= 0 || dest.height <= 0) {
        m.warn_once(L"the destination frame is incompletely described");
        return false;
    }

    // A Vulkan texture is required, and a cast is the only way to ask: `core::texture` is the
    // backend-agnostic interface, so an OpenGL one would compile and then hand over a null
    // image. Refusing here is what keeps that from reaching vkCmdCopyImage.
    auto* wrapper = dynamic_cast<texture_wrapper*>(src.get());
    if (!wrapper) {
        m.warn_once(L"the composited frame is not a Vulkan texture");
        return false;
    }
    auto source = wrapper->vk_texture();
    if (!source) {
        m.warn_once(L"the Vulkan texture wrapper carries no image");
        return false;
    }

    // Sizes must match exactly. A blit could rescale, but silently rescaling a recording is
    // worse than refusing it: the operator asked for the channel's raster.
    if (source->width() != dest.width || source->height() != dest.height) {
        m.warn_once(L"the composite and the encoder frame differ in size");
        return false;
    }

    return submit(dest, wrapper, source.get());
}

bool av_vulkan_exporter::clear_to_black(av_plane_dest& dest)
{
    auto& m = *impl_;

    if (!dest.image || !dest.semaphore || dest.width <= 0 || dest.height <= 0) {
        m.warn_once(L"the destination frame is incompletely described");
        return false;
    }
    // No wrapper and no source image, which `submit` reads as "clear" rather than "copy".
    return submit(dest, nullptr, nullptr);
}

/// `source` null means fill with black instead of copying; `wrapper` is then null too, so there
/// is no render to wait for and no source layout to move or restore. Everything else -- the
/// destination barriers, the timeline signal, the fence, the exception handling -- is identical,
/// which is the point. A black frame and a copied one must be indistinguishable to FFmpeg's
/// bookkeeping, and that only stays true while there is one copy of the bookkeeping.
bool av_vulkan_exporter::submit(av_plane_dest& dest, void* wrapper_ptr, void* source_ptr)
{
    auto& m       = *impl_;
    auto* dev     = m.dev_;
    auto* wrapper = static_cast<texture_wrapper*>(wrapper_ptr);
    auto* source  = static_cast<texture*>(source_ptr);

    // FFmpeg does not promise a layout for a freshly-allocated frame, and UNDEFINED is the
    // normal state of one from `av_hwframe_get_buffer`. Unlike the importer -- which refuses
    // UNDEFINED because it would be COPYING FROM undefined contents -- here the whole image is
    // overwritten, so UNDEFINED is honest and cheaper than preserving pooled contents. The
    // post-barrier still reports the layout FFmpeg expects to find.
    const auto dest_old = static_cast<vk::ImageLayout>(dest.layout);

    const bool ok = dev->dispatch_sync([&]() -> bool {
        try {
            // The mixer's render must be complete before its attachment is read. The wrapper
            // carries the wait for exactly this, and it is a host wait: a queue wait would sit
            // on the mixer's single shared graphics queue and block every other channel behind
            // it, which is the head-of-line stall that cost a TDR on the decode side.
            //
            // Nothing to wait for when there is no source: a black frame does not read the
            // mixer's attachment at all.
            if (wrapper)
                wrapper->ensure_render_complete();

            // Re-recording a command buffer the GPU may still be executing is undefined
            // behaviour. Against the PREVIOUS frame's copy, so in steady state it is free.
            m.wait_for_previous_copy();

            auto cmd = m.cmd_;
            cmd.reset(vk::CommandBufferResetFlags{});
            cmd.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

            const auto range  = vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);
            const auto layers = vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, 0, 1);
            (void)layers; // used only by the copy branch below

            std::vector<vk::ImageMemoryBarrier2> pre;
            pre.reserve(2);
            if (source) {
                // The mixer's attachment: from shader-read (where the composite left it) to
                // transfer-read.
                vk::ImageMemoryBarrier2 b{};
                b.oldLayout           = vk::ImageLayout::eShaderReadOnlyOptimal;
                b.newLayout           = vk::ImageLayout::eTransferSrcOptimal;
                b.srcQueueFamilyIndex = vk::QueueFamilyIgnored;
                b.dstQueueFamilyIndex = vk::QueueFamilyIgnored;
                b.image               = source->id();
                b.subresourceRange    = range;
                b.srcStageMask        = vk::PipelineStageFlagBits2::eAllCommands;
                b.srcAccessMask       = vk::AccessFlagBits2::eMemoryWrite;
                b.dstStageMask        = vk::PipelineStageFlagBits2::eTransfer;
                b.dstAccessMask       = vk::AccessFlagBits2::eTransferRead;
                pre.push_back(b);
            }
            {
                // FFmpeg's image: whole-image overwrite, so eUndefined discards rather than
                // preserving contents nobody reads.
                vk::ImageMemoryBarrier2 b{};
                b.oldLayout           = vk::ImageLayout::eUndefined;
                b.newLayout           = vk::ImageLayout::eTransferDstOptimal;
                b.srcQueueFamilyIndex = vk::QueueFamilyIgnored;
                b.dstQueueFamilyIndex = vk::QueueFamilyIgnored;
                b.image               = static_cast<VkImage>(dest.image);
                b.subresourceRange    = range;
                b.srcStageMask        = vk::PipelineStageFlagBits2::eTopOfPipe;
                b.dstStageMask        = vk::PipelineStageFlagBits2::eTransfer;
                b.dstAccessMask       = vk::AccessFlagBits2::eTransferWrite;
                pre.push_back(b);
            }
            {
                vk::DependencyInfo dep;
                dep.setImageMemoryBarriers(pre);
                cmd.pipelineBarrier2(dep);
            }

            if (source) {
                vk::ImageCopy c(
                    layers,
                    vk::Offset3D{},
                    layers,
                    vk::Offset3D{},
                    vk::Extent3D{static_cast<uint32_t>(dest.width), static_cast<uint32_t>(dest.height), 1});
                cmd.copyImage(source->id(),
                              vk::ImageLayout::eTransferSrcOptimal,
                              static_cast<VkImage>(dest.image),
                              vk::ImageLayout::eTransferDstOptimal,
                              c);
            } else {
                // Opaque black. Byte order does not matter for it -- (0,0,0,1) reads the same
                // as RGBA or as BGRA -- which makes this the one place in the Vulkan path where
                // the mixer's channel order can be ignored rather than compensated for.
                const vk::ClearColorValue black(std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f});
                cmd.clearColorImage(static_cast<VkImage>(dest.image),
                                    vk::ImageLayout::eTransferDstOptimal,
                                    black,
                                    range);
            }

            std::vector<vk::ImageMemoryBarrier2> post;
            post.reserve(2);
            if (source) {
                // Put the mixer's attachment back where it was found, for the same reason the
                // importer restores FFmpeg's: a client that moves an image and does not say so
                // leaves the next barrier with a wrong oldLayout, which discards contents.
                vk::ImageMemoryBarrier2 b{};
                b.oldLayout           = vk::ImageLayout::eTransferSrcOptimal;
                b.newLayout           = vk::ImageLayout::eShaderReadOnlyOptimal;
                b.srcQueueFamilyIndex = vk::QueueFamilyIgnored;
                b.dstQueueFamilyIndex = vk::QueueFamilyIgnored;
                b.image               = source->id();
                b.subresourceRange    = range;
                b.srcStageMask        = vk::PipelineStageFlagBits2::eTransfer;
                b.srcAccessMask       = vk::AccessFlagBits2::eTransferRead;
                b.dstStageMask        = vk::PipelineStageFlagBits2::eFragmentShader;
                b.dstAccessMask       = vk::AccessFlagBits2::eShaderRead;
                post.push_back(b);
            }
            {
                // Leave FFmpeg's image in the layout its own bookkeeping already records, so
                // `AVVkFrame::layout[]` stays true without the caller writing anything back.
                // UNDEFINED there means "no promise", and a general layout is the safe answer.
                vk::ImageMemoryBarrier2 b{};
                b.oldLayout = vk::ImageLayout::eTransferDstOptimal;
                const auto dest_new =
                    dest_old == vk::ImageLayout::eUndefined ? vk::ImageLayout::eGeneral : dest_old;
                b.newLayout      = dest_new;
                dest.final_layout = static_cast<int>(dest_new);
                b.srcQueueFamilyIndex = vk::QueueFamilyIgnored;
                b.dstQueueFamilyIndex = vk::QueueFamilyIgnored;
                b.image               = static_cast<VkImage>(dest.image);
                b.subresourceRange    = range;
                b.srcStageMask        = vk::PipelineStageFlagBits2::eTransfer;
                b.srcAccessMask       = vk::AccessFlagBits2::eTransferWrite;
                b.dstStageMask        = vk::PipelineStageFlagBits2::eAllCommands;
                b.dstAccessMask       = vk::AccessFlagBits2::eMemoryRead;
                post.push_back(b);
            }
            {
                vk::DependencyInfo dep;
                dep.setImageMemoryBarriers(post);
                cmd.pipelineBarrier2(dep);
            }

            cmd.end();

            // Signal `sem_value + 1`, the AVVkFrame contract: FFmpeg waits on the recorded
            // value before reading the frame, so this is what tells the encoder the picture
            // has arrived. The WAIT half is on the host above rather than in this submission,
            // for the same reason as the importer -- an unsatisfiable wait here would sit on
            // the mixer's shared graphics queue.
            vk::Semaphore sem          = static_cast<VkSemaphore>(dest.semaphore);
            uint64_t      signal_value = dest.sem_value + 1;

            vk::TimelineSemaphoreSubmitInfo timeline{};
            timeline.setSignalSemaphoreValues(signal_value);

            vk::SubmitInfo si{};
            si.setCommandBuffers(cmd);
            si.setSignalSemaphores(sem);
            si.pNext = &timeline;

            // Report both back before the submit can be observed. FFmpeg reads `sem_value[0]`
            // to decide what to wait on and what to signal next, so a stale value there is the
            // VUID-VkSubmitInfo2-semaphore-03882 in the header.
            dest.signalled_value = signal_value;

            m.vk_device_.resetFences(m.fence_);
            dev->submit(si, m.fence_);
            m.copy_pending_ = true;
            return true;
        } catch (const std::exception& e) {
            // std::exception rather than vk::SystemError: the wait helpers throw
            // caspar_exception, and letting that escape the lambda would rethrow it out of
            // dispatch_sync into the consumer, where it reads as an unexplained recording
            // failure rather than a declined frame.
            CASPAR_LOG(warning) << L"[vk::av_export] frame copy failed: " << u16(e.what());
            return false;
        } catch (...) {
            CASPAR_LOG(warning) << L"[vk::av_export] frame copy failed with an unknown exception";
            return false;
        }
    });

    return ok;
}

}}} // namespace caspar::accelerator::vulkan
