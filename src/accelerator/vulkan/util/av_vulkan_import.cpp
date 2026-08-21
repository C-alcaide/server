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

#include "av_vulkan_import.h"

#include "device.h"
#include "texture.h"
#include "texture_wrapper.h"

#include <common/except.h>
#include <common/log.h>

#include <cstdlib>
#include <exception>
#include <vector>

namespace caspar { namespace accelerator { namespace vulkan {

struct av_vulkan_importer::impl
{
    device*           dev_ = nullptr;
    vk::Device        vk_device_;
    vk::CommandBuffer cmd_;
    vk::Fence         fence_;
    bool              copy_pending_ = false;

    explicit impl(device* dev)
        : dev_(dev)
        , vk_device_(dev->getVkDevice())
    {
        // Allocated once, and outside any dispatch: allocateCommandBuffers hops to the
        // device thread itself, and doing it per frame would both leak the buffer and
        // defeat that thread's own recycling. Same reasoning as d3d11_import_bridge.
        cmd_   = dev_->allocateCommandBuffers(1).front();
        fence_ = vk_device_.createFence(vk::FenceCreateInfo{});
    }

    ~impl()
    {
        // The copy may still be reading the decoder's images and writing the mixer's;
        // tearing the fence down under it would be a use-after-free on the GPU.
        wait_for_previous_copy();
        if (fence_)
            vk_device_.destroyFence(fence_);
    }

    /// Block until every plane's decode work has completed, on the HOST.
    ///
    /// `AVVkFrame::sem_value[i]` is the value at which image i becomes accessible, so this is
    /// a wait for work FFmpeg has already submitted. Returns false on timeout or error, and
    /// the caller then declines the frame -- which costs one software frame, where letting an
    /// unsatisfiable wait reach the shared graphics queue costs the device.
    bool wait_for_decoder(const std::vector<av_plane_source>& planes)
    {
        std::vector<vk::Semaphore> sems;
        std::vector<uint64_t>      values;
        sems.reserve(planes.size());
        values.reserve(planes.size());
        for (const auto& p : planes) {
            sems.push_back(static_cast<VkSemaphore>(p.semaphore));
            values.push_back(p.sem_value);
        }

        vk::SemaphoreWaitInfo wi{};
        wi.setSemaphores(sems);
        wi.setValues(values);

        // One second: long enough that a busy decoder is never mistaken for a stuck one,
        // short enough that a stuck one does not hold the channel's tick for a visible time.
        const auto res = vk_device_.waitSemaphores(wi, 1'000'000'000ull);
        if (res != vk::Result::eSuccess) {
            CASPAR_LOG(warning) << L"[vk::av_import] the decoder's frame never became ready ("
                                << u16(vk::to_string(res)) << L"); declining it";
            return false;
        }
        return true;
    }

    void wait_for_previous_copy()
    {
        if (!copy_pending_ || !fence_)
            return;
        const auto res = vk_device_.waitForFences(fence_, VK_TRUE, 1'000'000'000ull);
        if (res != vk::Result::eSuccess)
            CASPAR_LOG(warning) << L"[vk::av_import] waiting for the previous plane copy timed out";
        copy_pending_ = false;
    }
};

av_vulkan_importer::av_vulkan_importer(void* vk_device)
{
    auto* dev = static_cast<device*>(vk_device);
    if (!dev)
        CASPAR_THROW_EXCEPTION(caspar_exception() << msg_info("av_vulkan_importer needs a Vulkan device"));
    impl_ = std::make_unique<impl>(dev);
}

av_vulkan_importer::~av_vulkan_importer() = default;

bool av_vulkan_importer::copy_planes(const std::vector<av_plane_source>&          planes,
                                     common::bit_depth                            depth,
                                     std::vector<std::shared_ptr<core::texture>>& out)
{
    auto& m   = *impl_;
    auto* dev = m.dev_;
    if (planes.empty())
        return false;

    // AVVkFrame carries up to AV_NUM_DATA_POINTERS planes; a YCbCr(A) frame is 3 or 4.
    if (planes.size() > 4) {
        CASPAR_LOG(warning) << L"[vk::av_import] " << planes.size()
                            << L" planes is more than this path handles";
        return false;
    }

    for (const auto& p : planes) {
        if (!p.image || !p.semaphore || p.width <= 0 || p.height <= 0) {
            CASPAR_LOG(warning) << L"[vk::av_import] a plane is incompletely described";
            return false;
        }
        // VK_IMAGE_LAYOUT_UNDEFINED means the contents are undefined, so copying from it
        // would produce a picture out of nothing. Refusing also lets the barriers below
        // restore the incoming layout unconditionally, which is what keeps FFmpeg's
        // `AVVkFrame::layout[]` bookkeeping true without the caller writing anything back.
        if (p.layout == static_cast<int>(VK_IMAGE_LAYOUT_UNDEFINED)) {
            CASPAR_LOG(warning) << L"[vk::av_import] a plane is still in VK_IMAGE_LAYOUT_UNDEFINED";
            return false;
        }
    }

    std::vector<std::shared_ptr<texture>> dst;

    const bool ok = dev->dispatch_sync([&]() -> bool {
        try {
            for (const auto& p : planes) {
                auto t = dev->create_texture(p.width, p.height, p.components, depth);
                if (!t)
                    return false;
                dst.push_back(t);
            }

            // WAIT ON THE HOST, NOT ON THE QUEUE, and this is the difference between one
            // producer working and four hanging the GPU.
            //
            // The obvious form of this function put each plane's timeline semaphore in the
            // SubmitInfo's wait list. That is correct for a queue this code owns, and wrong
            // for this one: the copy runs on the mixer's SINGLE graphics queue, shared by
            // every channel and every other producer, so a wait that is not yet satisfiable
            // blocks the mixer and everything queued behind it -- head-of-line blocking on
            // the one queue that must never stall. Measured 2026-08-21: one ProRes producer
            // ran fine, four gave `VK_ERROR_DEVICE_LOST` on every submission within seconds,
            // with a matching `nvlddmkm stopped responding and recovered` TDR in the Windows
            // System log at the same second. A TDR at 4x25 fps is a deadlock, not overload --
            // the same decode sustains ~2100 fps standalone.
            //
            // So the wait happens here, on the host, before anything is recorded. In steady
            // state FFmpeg has already submitted the work these values belong to, so it
            // returns immediately; if it does not, the frame is DECLINED rather than allowed
            // to wedge the queue. The signal stays on the submission, because that is what
            // tells FFmpeg the copy is done reading.
            if (!m.wait_for_decoder(planes))
                return false;

            // Re-recording a command buffer the GPU may still be reading is undefined
            // behaviour, so the previous copy has to have retired first. Against a copy from
            // a PREVIOUS frame, so in steady state it does not block either.
            m.wait_for_previous_copy();

            auto cmd = m.cmd_;
            cmd.reset(vk::CommandBufferResetFlags{});
            cmd.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

            const auto range = vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);

            // Sources: from whatever layout FFmpeg left them in, to transfer-read. The
            // layout is carried per plane in AVVkFrame rather than assumed, because the
            // decoder does not promise a particular one.
            std::vector<vk::ImageMemoryBarrier2> pre;
            pre.reserve(planes.size() * 2);
            for (const auto& p : planes) {
                vk::ImageMemoryBarrier2 b{};
                b.oldLayout           = static_cast<vk::ImageLayout>(p.layout);
                b.newLayout           = vk::ImageLayout::eTransferSrcOptimal;
                b.srcQueueFamilyIndex = vk::QueueFamilyIgnored;
                b.dstQueueFamilyIndex = vk::QueueFamilyIgnored;
                b.image               = static_cast<VkImage>(p.image);
                b.subresourceRange    = range;
                b.srcStageMask        = vk::PipelineStageFlagBits2::eAllCommands;
                b.srcAccessMask       = vk::AccessFlagBits2::eMemoryWrite;
                b.dstStageMask        = vk::PipelineStageFlagBits2::eTransfer;
                b.dstAccessMask       = vk::AccessFlagBits2::eTransferRead;
                pre.push_back(b);
            }
            // Destinations: whole-image overwrite, so eUndefined is honest and avoids
            // preserving pooled contents nobody will read.
            for (const auto& t : dst) {
                vk::ImageMemoryBarrier2 b{};
                b.oldLayout           = vk::ImageLayout::eUndefined;
                b.newLayout           = vk::ImageLayout::eTransferDstOptimal;
                b.srcQueueFamilyIndex = vk::QueueFamilyIgnored;
                b.dstQueueFamilyIndex = vk::QueueFamilyIgnored;
                b.image               = t->id();
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

            const auto layers = vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, 0, 1);
            for (size_t i = 0; i < planes.size(); ++i) {
                vk::ImageCopy c(layers,
                                vk::Offset3D{},
                                layers,
                                vk::Offset3D{},
                                vk::Extent3D{static_cast<uint32_t>(planes[i].width),
                                             static_cast<uint32_t>(planes[i].height),
                                             1});
                cmd.copyImage(static_cast<VkImage>(planes[i].image),
                              vk::ImageLayout::eTransferSrcOptimal,
                              dst[i]->id(),
                              vk::ImageLayout::eTransferDstOptimal,
                              c);
            }

            std::vector<vk::ImageMemoryBarrier2> post;
            post.reserve(dst.size() + planes.size());
            // Put the sources back exactly where they were found. `AVVkFrame::layout[]` is
            // documented as updated after every barrier, and a client that moves an image
            // and does not say so leaves FFmpeg's next barrier with a wrong oldLayout --
            // which discards contents rather than transitioning them. Restoring here is
            // preferable to reporting the new layout back, because it cannot be forgotten
            // at a call site.
            for (const auto& p : planes) {
                vk::ImageMemoryBarrier2 b{};
                b.oldLayout           = vk::ImageLayout::eTransferSrcOptimal;
                b.newLayout           = static_cast<vk::ImageLayout>(p.layout);
                b.srcQueueFamilyIndex = vk::QueueFamilyIgnored;
                b.dstQueueFamilyIndex = vk::QueueFamilyIgnored;
                b.image               = static_cast<VkImage>(p.image);
                b.subresourceRange    = range;
                b.srcStageMask        = vk::PipelineStageFlagBits2::eTransfer;
                b.srcAccessMask       = vk::AccessFlagBits2::eTransferRead;
                b.dstStageMask        = vk::PipelineStageFlagBits2::eAllCommands;
                post.push_back(b);
            }
            for (const auto& t : dst) {
                vk::ImageMemoryBarrier2 b{};
                b.oldLayout           = vk::ImageLayout::eTransferDstOptimal;
                b.newLayout           = vk::ImageLayout::eShaderReadOnlyOptimal;
                b.srcQueueFamilyIndex = vk::QueueFamilyIgnored;
                b.dstQueueFamilyIndex = vk::QueueFamilyIgnored;
                b.image               = t->id();
                b.subresourceRange    = range;
                b.srcStageMask        = vk::PipelineStageFlagBits2::eTransfer;
                b.srcAccessMask       = vk::AccessFlagBits2::eTransferWrite;
                b.dstStageMask        = vk::PipelineStageFlagBits2::eFragmentShader;
                b.dstAccessMask       = vk::AccessFlagBits2::eShaderRead;
                post.push_back(b);
            }
            {
                vk::DependencyInfo dep;
                dep.setImageMemoryBarriers(post);
                cmd.pipelineBarrier2(dep);
            }

            cmd.end();

            // SIGNAL, per the AVVkFrame contract: wait on `sem_value`, signal an
            // incremented value, and let the caller record it. FFmpeg waits on the recorded
            // value before reusing a pooled frame, so this is what stops the decoder
            // overwriting an image the copy is still reading.
            //
            // Worth knowing that this was accused and acquitted. It was removed for a while
            // on the strength of an A/B that showed 356 device-lost errors with it and none
            // without -- and that A/B was invalid, because the arm without it had only one
            // of four producers ever active, so it was not doing concurrent work at all.
            // With the real cause fixed (one shared AVHWDeviceContext, see
            // make_vulkan_hwdevice_from_mixer) signalling is clean at eight concurrent
            // producers.
            //
            // The WAIT half is on the host rather than in this submission, and that is not
            // the contract's letter: a wait here would sit on the mixer's single shared
            // graphics queue, so an unsatisfiable value would block every other channel
            // behind it. `wait_for_decoder` above does it on the host instead, and declines
            // the frame on timeout rather than risking the queue.
            std::vector<vk::Semaphore> sems;
            std::vector<uint64_t>      signal_values;
            sems.reserve(planes.size());
            for (const auto& p : planes) {
                sems.push_back(static_cast<VkSemaphore>(p.semaphore));
                signal_values.push_back(p.sem_value + 1);
            }

            vk::TimelineSemaphoreSubmitInfo timeline{};
            timeline.setSignalSemaphoreValues(signal_values);

            vk::SubmitInfo si{};
            si.setCommandBuffers(cmd);
            si.setSignalSemaphores(sems);
            si.pNext = &timeline;

            // The fence is for buffer reuse, not for the mixer: this goes to the same
            // queue the mixer's draw does, through the same dispatch thread, so the copy
            // is already ordered before any later sampling of these textures.
            m.vk_device_.resetFences(m.fence_);
            dev->submit(si, m.fence_);
            m.copy_pending_ = true;
            return true;
        } catch (const std::exception& e) {
            // std::exception rather than vk::SystemError: create_texture throws a
            // caspar_exception for a layout this GPU cannot sample, and letting that
            // escape the lambda would rethrow it out of dispatch_sync into the producer,
            // where it reads as an unexplained decode failure.
            CASPAR_LOG(warning) << L"[vk::av_import] plane copy failed: " << u16(e.what());
            return false;
        } catch (...) {
            CASPAR_LOG(warning) << L"[vk::av_import] plane copy failed with an unknown exception";
            return false;
        }
    });

    if (!ok)
        return false;

    // vulkan::texture is not a core::texture; the mixer recognises a frame's planes by the
    // device the wrapper carries, and the wrapper is also what makes PRINT RAW work on this
    // path (see texture_wrapper::read_pixels, which keys off exactly this pointer).
    out.clear();
    out.reserve(dst.size());
    for (auto& t : dst)
        out.push_back(std::make_shared<VkReadableTextureWrapper>(std::move(t), dev->shared_from_this()));
    return true;
}

}}} // namespace caspar::accelerator::vulkan
