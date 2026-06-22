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

#include "vulkan_output_consumer.h"
#include "config.h"
#include "../util/color_convert_pipeline.h"
#include "../util/display_enum.h"
#include "../util/output_window.h"
#include "../util/presentation.h"

#include <accelerator/vulkan/util/device.h>
#include <accelerator/vulkan/util/texture.h>
#include <accelerator/vulkan/util/vulkan_queue.h>

#include <common/diagnostics/graph.h>
#include <common/except.h>
#include <common/future.h>
#include <common/log.h>
#include <common/memory.h>
#include <common/timer.h>

#include <core/consumer/channel_info.h>
#include <core/consumer/frame_consumer.h>
#include <core/frame/frame.h>
#include <core/video_format.h>

#include <boost/algorithm/string.hpp>
#include <boost/property_tree/ptree.hpp>

#include <tbb/concurrent_queue.h>

#include <vulkan/vulkan.hpp>

#ifdef _WIN32
#include <vulkan/vulkan_win32.h>
#endif

#include <atomic>
#include <limits>
#include <thread>

namespace caspar { namespace vulkan_output {

namespace {

static constexpr int MAX_FRAMES_IN_FLIGHT = 3;

struct frame_sync
{
    vk::Semaphore     image_available;
    vk::Semaphore     render_finished;
    vk::Fence         in_flight;
    vk::CommandBuffer cmd_buffer;
};

// ----------------------------------------------------------------------------
// Swapchain helper (minimal, blit-based — no render pass or shaders needed)
// ----------------------------------------------------------------------------

class present_swapchain
{
  public:
    present_swapchain(vk::Instance        instance,
                      vk::PhysicalDevice  physical,
                      vk::Device          device,
                      vk::Queue           queue,
                      uint32_t            queue_family,
                      vk::SurfaceKHR      surface,
                      int                 desired_images,
                      presentation_tier   tier,
                      vk::PresentModeKHR  present_mode)
        : instance_(instance)
        , physical_(physical)
        , device_(device)
        , queue_(queue)
        , queue_family_(queue_family)
        , surface_(surface)
        , tier_(tier)
        , present_mode_(present_mode)
    {
        create_swapchain(desired_images);
        create_sync_objects();
        create_command_pool();
    }

    ~present_swapchain()
    {
        device_.waitIdle();
        for (auto& f : frames_) {
            device_.destroySemaphore(f.image_available);
            device_.destroySemaphore(f.render_finished);
            device_.destroyFence(f.in_flight);
        }
        device_.destroyCommandPool(cmd_pool_);
        for (auto& iv : image_views_)
            device_.destroyImageView(iv);
        device_.destroySwapchainKHR(swapchain_);
    }

    present_swapchain(const present_swapchain&)            = delete;
    present_swapchain& operator=(const present_swapchain&) = delete;

    uint32_t width() const { return extent_.width; }
    uint32_t height() const { return extent_.height; }

    // Returns UINT32_MAX if swapchain needs recreation (shouldn't happen for FSE)
    uint32_t acquire_next_image()
    {
        auto result = device_.acquireNextImageKHR(
            swapchain_, std::numeric_limits<uint64_t>::max(), current_frame().image_available, nullptr);

        if (result.result == vk::Result::eErrorOutOfDateKHR || result.result == vk::Result::eSuboptimalKHR)
            return std::numeric_limits<uint32_t>::max();

        return result.value;
    }

    void wait_fence()
    {
        auto res = device_.waitForFences(current_frame().in_flight, VK_TRUE, std::numeric_limits<uint64_t>::max());
        (void)res;
        device_.resetFences(current_frame().in_flight);
    }

    // Record and submit a blit from src_image to the swapchain image at image_index.
    void blit_and_present(VkImage  src_image,
                          uint32_t src_width,
                          uint32_t src_height,
                          uint32_t image_index)
    {
        auto& sync = current_frame();
        auto  cmd  = sync.cmd_buffer;

        cmd.reset();
        cmd.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

        // Transition swapchain image: undefined -> transfer dst
        vk::ImageMemoryBarrier barrier_to_dst{};
        barrier_to_dst.srcAccessMask       = {};
        barrier_to_dst.dstAccessMask       = vk::AccessFlagBits::eTransferWrite;
        barrier_to_dst.oldLayout           = vk::ImageLayout::eUndefined;
        barrier_to_dst.newLayout           = vk::ImageLayout::eTransferDstOptimal;
        barrier_to_dst.image               = images_[image_index];
        barrier_to_dst.subresourceRange    = vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
                            vk::PipelineStageFlagBits::eTransfer,
                            {},
                            nullptr,
                            nullptr,
                            barrier_to_dst);

        // Transition source image: shader read -> transfer src
        vk::ImageMemoryBarrier barrier_src{};
        barrier_src.srcAccessMask    = vk::AccessFlagBits::eShaderRead;
        barrier_src.dstAccessMask    = vk::AccessFlagBits::eTransferRead;
        barrier_src.oldLayout        = vk::ImageLayout::eShaderReadOnlyOptimal;
        barrier_src.newLayout        = vk::ImageLayout::eTransferSrcOptimal;
        barrier_src.image            = src_image;
        barrier_src.subresourceRange = vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eFragmentShader,
                            vk::PipelineStageFlagBits::eTransfer,
                            {},
                            nullptr,
                            nullptr,
                            barrier_src);

        // Blit (handles scaling if source != swapchain size)
        vk::ImageBlit region{};
        region.srcSubresource = vk::ImageSubresourceLayers{vk::ImageAspectFlagBits::eColor, 0, 0, 1};
        region.srcOffsets[0]  = vk::Offset3D{0, 0, 0};
        region.srcOffsets[1]  = vk::Offset3D{static_cast<int32_t>(src_width), static_cast<int32_t>(src_height), 1};
        region.dstSubresource = vk::ImageSubresourceLayers{vk::ImageAspectFlagBits::eColor, 0, 0, 1};
        region.dstOffsets[0]  = vk::Offset3D{0, 0, 0};
        region.dstOffsets[1]  = vk::Offset3D{static_cast<int32_t>(extent_.width), static_cast<int32_t>(extent_.height), 1};
        cmd.blitImage(src_image,
                      vk::ImageLayout::eTransferSrcOptimal,
                      images_[image_index],
                      vk::ImageLayout::eTransferDstOptimal,
                      region,
                      vk::Filter::eLinear);

        // Transition source back: transfer src -> shader read
        vk::ImageMemoryBarrier barrier_src_back{};
        barrier_src_back.srcAccessMask    = vk::AccessFlagBits::eTransferRead;
        barrier_src_back.dstAccessMask    = vk::AccessFlagBits::eShaderRead;
        barrier_src_back.oldLayout        = vk::ImageLayout::eTransferSrcOptimal;
        barrier_src_back.newLayout        = vk::ImageLayout::eShaderReadOnlyOptimal;
        barrier_src_back.image            = src_image;
        barrier_src_back.subresourceRange = vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                            vk::PipelineStageFlagBits::eFragmentShader,
                            {},
                            nullptr,
                            nullptr,
                            barrier_src_back);

        // Transition swapchain image: transfer dst -> present
        vk::ImageMemoryBarrier barrier_to_present{};
        barrier_to_present.srcAccessMask    = vk::AccessFlagBits::eTransferWrite;
        barrier_to_present.dstAccessMask    = {};
        barrier_to_present.oldLayout        = vk::ImageLayout::eTransferDstOptimal;
        barrier_to_present.newLayout        = vk::ImageLayout::ePresentSrcKHR;
        barrier_to_present.image            = images_[image_index];
        barrier_to_present.subresourceRange = vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                            vk::PipelineStageFlagBits::eBottomOfPipe,
                            {},
                            nullptr,
                            nullptr,
                            barrier_to_present);

        cmd.end();

        // Submit
        vk::PipelineStageFlags wait_stage = vk::PipelineStageFlagBits::eTransfer;
        vk::SubmitInfo         submit{};
        submit.waitSemaphoreCount   = 1;
        submit.pWaitSemaphores      = &sync.image_available;
        submit.pWaitDstStageMask    = &wait_stage;
        submit.commandBufferCount   = 1;
        submit.pCommandBuffers      = &sync.cmd_buffer;
        submit.signalSemaphoreCount = 1;
        submit.pSignalSemaphores    = &sync.render_finished;
        queue_.submit(submit, sync.in_flight);

        // Present
        vk::PresentInfoKHR present{};
        present.waitSemaphoreCount = 1;
        present.pWaitSemaphores    = &sync.render_finished;
        present.swapchainCount     = 1;
        present.pSwapchains        = &swapchain_;
        present.pImageIndices      = &image_index;
        auto present_result = queue_.presentKHR(present);
        (void)present_result;

        current_frame_idx_ = (current_frame_idx_ + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    // Record and submit: blit src → intermediate, dispatch color compute, blit intermediate → swapchain.
    void blit_via_compute_and_present(VkImage                  src_image,
                                      uint32_t                 src_width,
                                      uint32_t                 src_height,
                                      color_convert_pipeline&  pipeline,
                                      uint32_t                 image_index)
    {
        auto& sync = current_frame();
        auto  cmd  = sync.cmd_buffer;

        cmd.reset();
        cmd.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

        // Transition intermediate: undefined → transfer dst
        vk::ImageMemoryBarrier bar_int_to_dst{};
        bar_int_to_dst.srcAccessMask    = {};
        bar_int_to_dst.dstAccessMask    = vk::AccessFlagBits::eTransferWrite;
        bar_int_to_dst.oldLayout        = vk::ImageLayout::eUndefined;
        bar_int_to_dst.newLayout        = vk::ImageLayout::eTransferDstOptimal;
        bar_int_to_dst.image            = pipeline.image();
        bar_int_to_dst.subresourceRange = vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
                            vk::PipelineStageFlagBits::eTransfer,
                            {}, nullptr, nullptr, bar_int_to_dst);

        // Transition source: shader read → transfer src
        vk::ImageMemoryBarrier bar_src_to_xfer{};
        bar_src_to_xfer.srcAccessMask    = vk::AccessFlagBits::eShaderRead;
        bar_src_to_xfer.dstAccessMask    = vk::AccessFlagBits::eTransferRead;
        bar_src_to_xfer.oldLayout        = vk::ImageLayout::eShaderReadOnlyOptimal;
        bar_src_to_xfer.newLayout        = vk::ImageLayout::eTransferSrcOptimal;
        bar_src_to_xfer.image            = src_image;
        bar_src_to_xfer.subresourceRange = vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eFragmentShader,
                            vk::PipelineStageFlagBits::eTransfer,
                            {}, nullptr, nullptr, bar_src_to_xfer);

        // Blit source → intermediate (scaled to intermediate size)
        vk::ImageBlit region_to_int{};
        region_to_int.srcSubresource = vk::ImageSubresourceLayers{vk::ImageAspectFlagBits::eColor, 0, 0, 1};
        region_to_int.srcOffsets[0]  = vk::Offset3D{0, 0, 0};
        region_to_int.srcOffsets[1]  = vk::Offset3D{static_cast<int32_t>(src_width), static_cast<int32_t>(src_height), 1};
        region_to_int.dstSubresource = vk::ImageSubresourceLayers{vk::ImageAspectFlagBits::eColor, 0, 0, 1};
        region_to_int.dstOffsets[0]  = vk::Offset3D{0, 0, 0};
        region_to_int.dstOffsets[1]  = vk::Offset3D{static_cast<int32_t>(pipeline.width()), static_cast<int32_t>(pipeline.height()), 1};
        cmd.blitImage(src_image,
                      vk::ImageLayout::eTransferSrcOptimal,
                      pipeline.image(),
                      vk::ImageLayout::eTransferDstOptimal,
                      region_to_int,
                      vk::Filter::eLinear);

        // Transition source back: transfer src → shader read
        vk::ImageMemoryBarrier bar_src_back{};
        bar_src_back.srcAccessMask    = vk::AccessFlagBits::eTransferRead;
        bar_src_back.dstAccessMask    = vk::AccessFlagBits::eShaderRead;
        bar_src_back.oldLayout        = vk::ImageLayout::eTransferSrcOptimal;
        bar_src_back.newLayout        = vk::ImageLayout::eShaderReadOnlyOptimal;
        bar_src_back.image            = src_image;
        bar_src_back.subresourceRange = vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                            vk::PipelineStageFlagBits::eFragmentShader,
                            {}, nullptr, nullptr, bar_src_back);

        // Transition intermediate: transfer dst → general (for compute)
        vk::ImageMemoryBarrier bar_int_to_general{};
        bar_int_to_general.srcAccessMask    = vk::AccessFlagBits::eTransferWrite;
        bar_int_to_general.dstAccessMask    = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite;
        bar_int_to_general.oldLayout        = vk::ImageLayout::eTransferDstOptimal;
        bar_int_to_general.newLayout        = vk::ImageLayout::eGeneral;
        bar_int_to_general.image            = pipeline.image();
        bar_int_to_general.subresourceRange = vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                            vk::PipelineStageFlagBits::eComputeShader,
                            {}, nullptr, nullptr, bar_int_to_general);

        // Dispatch color conversion compute shader
        pipeline.dispatch(cmd, pipeline.width(), pipeline.height());

        // Transition intermediate: general → transfer src
        vk::ImageMemoryBarrier bar_int_to_src{};
        bar_int_to_src.srcAccessMask    = vk::AccessFlagBits::eShaderWrite;
        bar_int_to_src.dstAccessMask    = vk::AccessFlagBits::eTransferRead;
        bar_int_to_src.oldLayout        = vk::ImageLayout::eGeneral;
        bar_int_to_src.newLayout        = vk::ImageLayout::eTransferSrcOptimal;
        bar_int_to_src.image            = pipeline.image();
        bar_int_to_src.subresourceRange = vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                            vk::PipelineStageFlagBits::eTransfer,
                            {}, nullptr, nullptr, bar_int_to_src);

        // Transition swapchain image: undefined → transfer dst
        vk::ImageMemoryBarrier bar_swap_to_dst{};
        bar_swap_to_dst.srcAccessMask    = {};
        bar_swap_to_dst.dstAccessMask    = vk::AccessFlagBits::eTransferWrite;
        bar_swap_to_dst.oldLayout        = vk::ImageLayout::eUndefined;
        bar_swap_to_dst.newLayout        = vk::ImageLayout::eTransferDstOptimal;
        bar_swap_to_dst.image            = images_[image_index];
        bar_swap_to_dst.subresourceRange = vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
                            vk::PipelineStageFlagBits::eTransfer,
                            {}, nullptr, nullptr, bar_swap_to_dst);

        // Blit intermediate → swapchain
        vk::ImageBlit region_to_swap{};
        region_to_swap.srcSubresource = vk::ImageSubresourceLayers{vk::ImageAspectFlagBits::eColor, 0, 0, 1};
        region_to_swap.srcOffsets[0]  = vk::Offset3D{0, 0, 0};
        region_to_swap.srcOffsets[1]  = vk::Offset3D{static_cast<int32_t>(pipeline.width()), static_cast<int32_t>(pipeline.height()), 1};
        region_to_swap.dstSubresource = vk::ImageSubresourceLayers{vk::ImageAspectFlagBits::eColor, 0, 0, 1};
        region_to_swap.dstOffsets[0]  = vk::Offset3D{0, 0, 0};
        region_to_swap.dstOffsets[1]  = vk::Offset3D{static_cast<int32_t>(extent_.width), static_cast<int32_t>(extent_.height), 1};
        cmd.blitImage(pipeline.image(),
                      vk::ImageLayout::eTransferSrcOptimal,
                      images_[image_index],
                      vk::ImageLayout::eTransferDstOptimal,
                      region_to_swap,
                      vk::Filter::eLinear);

        // Transition swapchain image: transfer dst → present
        vk::ImageMemoryBarrier bar_swap_to_present{};
        bar_swap_to_present.srcAccessMask    = vk::AccessFlagBits::eTransferWrite;
        bar_swap_to_present.dstAccessMask    = {};
        bar_swap_to_present.oldLayout        = vk::ImageLayout::eTransferDstOptimal;
        bar_swap_to_present.newLayout        = vk::ImageLayout::ePresentSrcKHR;
        bar_swap_to_present.image            = images_[image_index];
        bar_swap_to_present.subresourceRange = vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                            vk::PipelineStageFlagBits::eBottomOfPipe,
                            {}, nullptr, nullptr, bar_swap_to_present);

        cmd.end();

        // Submit
        vk::PipelineStageFlags wait_stage = vk::PipelineStageFlagBits::eTransfer;
        vk::SubmitInfo         submit{};
        submit.waitSemaphoreCount   = 1;
        submit.pWaitSemaphores      = &sync.image_available;
        submit.pWaitDstStageMask    = &wait_stage;
        submit.commandBufferCount   = 1;
        submit.pCommandBuffers      = &sync.cmd_buffer;
        submit.signalSemaphoreCount = 1;
        submit.pSignalSemaphores    = &sync.render_finished;
        queue_.submit(submit, sync.in_flight);

        // Present
        vk::PresentInfoKHR present{};
        present.waitSemaphoreCount = 1;
        present.pWaitSemaphores    = &sync.render_finished;
        present.swapchainCount     = 1;
        present.pSwapchains        = &swapchain_;
        present.pImageIndices      = &image_index;
        auto present_result = queue_.presentKHR(present);
        (void)present_result;

        current_frame_idx_ = (current_frame_idx_ + 1) % MAX_FRAMES_IN_FLIGHT;
    }

  private:
    frame_sync& current_frame() { return frames_[current_frame_idx_]; }

    void create_swapchain(int desired_images)
    {
        auto caps = physical_.getSurfaceCapabilitiesKHR(surface_);
        auto formats = physical_.getSurfaceFormatsKHR(surface_);

        // Prefer BGRA8 SRGB
        vk::SurfaceFormatKHR chosen = formats[0];
        for (const auto& f : formats) {
            if (f.format == vk::Format::eB8G8R8A8Srgb && f.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
                chosen = f;
                break;
            }
        }
        if (chosen.format == vk::Format::eUndefined)
            chosen.format = vk::Format::eB8G8R8A8Unorm;

        format_ = chosen.format;
        extent_ = caps.currentExtent;
        if (extent_.width == std::numeric_limits<uint32_t>::max()) {
            extent_.width  = 1920;
            extent_.height = 1080;
        }

        uint32_t image_count = std::max(static_cast<uint32_t>(desired_images), caps.minImageCount);
        if (caps.maxImageCount > 0)
            image_count = std::min(image_count, caps.maxImageCount);

        vk::SwapchainCreateInfoKHR ci{};
        ci.surface          = surface_;
        ci.minImageCount    = image_count;
        ci.imageFormat      = format_;
        ci.imageColorSpace  = chosen.colorSpace;
        ci.imageExtent      = extent_;
        ci.imageArrayLayers = 1;
        ci.imageUsage       = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eColorAttachment;
        ci.imageSharingMode = vk::SharingMode::eExclusive;
        ci.preTransform     = caps.currentTransform;
        ci.compositeAlpha   = vk::CompositeAlphaFlagBitsKHR::eOpaque;
        ci.presentMode      = present_mode_;
        ci.clipped          = VK_TRUE;

#ifdef _WIN32
        // Chain FSE info if using full-screen exclusive tier
        VkSurfaceFullScreenExclusiveInfoEXT fse_info{};
        bool fse_chained = false;
        if (tier_ == presentation_tier::full_screen_exclusive) {
            fse_info.sType               = VK_STRUCTURE_TYPE_SURFACE_FULL_SCREEN_EXCLUSIVE_INFO_EXT;
            fse_info.pNext               = nullptr;
            fse_info.fullScreenExclusive = VK_FULL_SCREEN_EXCLUSIVE_DEFAULT_EXT;
            ci.pNext                     = &fse_info;
            fse_chained = true;
        }
#endif

        swapchain_ = device_.createSwapchainKHR(ci);

#ifdef _WIN32
        if (!swapchain_ && fse_chained) {
            // FSE chain failed — retry without it
            CASPAR_LOG(warning) << L"[vulkan_output] Swapchain creation with FSE failed. Retrying without.";
            ci.pNext = nullptr;
            swapchain_ = device_.createSwapchainKHR(ci);
        }
#endif
        images_    = device_.getSwapchainImagesKHR(swapchain_);

        for (const auto& img : images_) {
            vk::ImageViewCreateInfo iv_ci{};
            iv_ci.image                           = img;
            iv_ci.viewType                        = vk::ImageViewType::e2D;
            iv_ci.format                          = format_;
            iv_ci.subresourceRange.aspectMask     = vk::ImageAspectFlagBits::eColor;
            iv_ci.subresourceRange.baseMipLevel   = 0;
            iv_ci.subresourceRange.levelCount     = 1;
            iv_ci.subresourceRange.baseArrayLayer = 0;
            iv_ci.subresourceRange.layerCount     = 1;
            image_views_.push_back(device_.createImageView(iv_ci));
        }
    }

    void create_sync_objects()
    {
        for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
            frames_[i].image_available = device_.createSemaphore({});
            frames_[i].render_finished = device_.createSemaphore({});
            frames_[i].in_flight       = device_.createFence({vk::FenceCreateFlagBits::eSignaled});
        }
    }

    void create_command_pool()
    {
        vk::CommandPoolCreateInfo pool_ci{};
        pool_ci.flags            = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
        pool_ci.queueFamilyIndex = queue_family_;
        cmd_pool_                = device_.createCommandPool(pool_ci);

        vk::CommandBufferAllocateInfo alloc{};
        alloc.commandPool        = cmd_pool_;
        alloc.level              = vk::CommandBufferLevel::ePrimary;
        alloc.commandBufferCount = MAX_FRAMES_IN_FLIGHT;
        auto buffers             = device_.allocateCommandBuffers(alloc);
        for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
            frames_[i].cmd_buffer = buffers[i];
    }

    vk::Instance               instance_;
    vk::PhysicalDevice         physical_;
    vk::Device                 device_;
    vk::Queue                  queue_;
    uint32_t                   queue_family_;
    vk::SurfaceKHR             surface_;
    presentation_tier          tier_;
    vk::PresentModeKHR         present_mode_;
    vk::SwapchainKHR           swapchain_;
    vk::Format                 format_;
    vk::Extent2D               extent_;
    std::vector<vk::Image>     images_;
    std::vector<vk::ImageView> image_views_;
    vk::CommandPool            cmd_pool_;
    frame_sync                 frames_[MAX_FRAMES_IN_FLIGHT];
    int                        current_frame_idx_ = 0;
};

// ----------------------------------------------------------------------------
// Consumer implementation
// ----------------------------------------------------------------------------

class vulkan_output_consumer_impl
{
  public:
    vulkan_output_consumer_impl(configuration                                config,
                                std::shared_ptr<accelerator::vulkan::device> device)
        : config_(std::move(config))
        , device_(std::move(device))
    {
    }

    ~vulkan_output_consumer_impl()
    {
        is_running_ = false;
        frame_buffer_.abort();
        if (thread_.joinable())
            thread_.join();
    }

    void initialize(const core::video_format_desc& format_desc, int channel_index)
    {
        format_desc_   = format_desc;
        channel_index_ = channel_index;

        graph_->set_color("tick-time", diagnostics::color(0.0f, 0.6f, 0.9f));
        graph_->set_color("frame-time", diagnostics::color(0.1f, 1.0f, 0.1f));
        graph_->set_color("dropped-frame", diagnostics::color(0.3f, 0.6f, 0.3f));
        graph_->set_text(print());
        diagnostics::register_graph(graph_);

        frame_buffer_.set_capacity(1);

        thread_ = std::thread([this] {
            try {
                run();
            } catch (tbb::user_abort&) {
                // Normal shutdown
            } catch (...) {
                CASPAR_LOG_CURRENT_EXCEPTION();
                is_running_ = false;
            }
        });
    }

    std::future<bool> send(core::video_field /*field*/, const core::const_frame& frame)
    {
        if (!frame_buffer_.try_push(frame)) {
            graph_->set_tag(diagnostics::tag_severity::WARNING, "dropped-frame");
        }
        return make_ready_future(is_running_.load());
    }

    std::wstring print() const
    {
        return L"vulkan_output[" + std::to_wstring(config_.output_index) + L"|ch" +
               std::to_wstring(channel_index_) + L"]";
    }

    bool is_running() const { return is_running_.load(); }

  private:
    void run()
    {
        // Find target display
        auto displays = enumerate_displays();
        auto* target  = find_display(displays, config_.output_index, config_.display_name);
        if (!target) {
            CASPAR_LOG(error) << print() << L" Display output " << config_.output_index << L" not found.";
            is_running_ = false;
            return;
        }

        auto vk_instance = device_->instance();
        auto physical    = device_->physical_device();
        auto vk_device   = device_->getVkDevice();
        auto queue_obj   = device_->queue();
        auto vk_queue    = queue_obj->vk_queue();

        // Determine presentation tier and create surface
        uint32_t target_refresh_mhz = static_cast<uint32_t>(format_desc_.fps * 1000.0 + 0.5);
        auto tier_result = create_tiered_surface(vk_instance, physical, vk_device, *target, target_refresh_mhz);

        vk::SurfaceKHR surface;
        presentation_tier active_tier = tier_result.tier;

        if (tier_result.surface) {
            // Direct display path (VK_KHR_display) — surface already created, no window needed
            surface = tier_result.surface;
            CASPAR_LOG(info) << print() << L" Using " << tier_name(active_tier) << L" (no window).";
        } else {
            // FSE or borderless — need a window for the surface
            window_ = std::make_unique<output_window>(*target);
            surface = window_->create_surface(vk_instance);
            CASPAR_LOG(info) << print() << L" Using " << tier_name(active_tier) << L" with window.";
        }

        // Select present mode
        auto present_mode = pick_present_mode(physical, surface);

        swapchain_ = std::make_unique<present_swapchain>(vk_instance,
                                                         physical,
                                                         vk_device,
                                                         vk_queue,
                                                         queue_obj->family_index(),
                                                         surface,
                                                         config_.buffer_depth,
                                                         active_tier,
                                                         present_mode);

        CASPAR_LOG(info) << print() << L" Started. Swapchain " << swapchain_->width() << L"x"
                         << swapchain_->height() << L" | " << tier_name(active_tier)
                         << L" | " << target->display_name;

        // Initialize color conversion pipeline (always created, conditionally dispatched)
        color_pipeline_ = std::make_unique<color_convert_pipeline>(
            vk_device, physical, swapchain_->width(), swapchain_->height());

        // Determine tone map op from transfer/EOTF combination
        int tone_map = 0;
        if (config_.eotf == output_eotf::hlg)
            tone_map = 7; // hlg_ootf
        else if (config_.gamut == output_gamut::bt2020 && config_.eotf == output_eotf::pq)
            tone_map = 3; // aces_rrt for HDR10

        color_pipeline_->update_config(config_.gamut, config_.eotf, 1000.0f, tone_map);

        if (color_pipeline_->is_active()) {
            CASPAR_LOG(info) << print() << L" Color pipeline active: gamut="
                             << static_cast<int>(config_.gamut)
                             << L" eotf=" << static_cast<int>(config_.eotf);
        }

        while (is_running_) {
            tick(queue_obj);
        }

        // Cleanup
        color_pipeline_.reset();
        swapchain_.reset();
        vk_instance.destroySurfaceKHR(surface);
        window_.reset();
    }

    void tick(const std::shared_ptr<accelerator::vulkan::vulkan_queue>& queue_obj)
    {
        core::const_frame frame;

        while (!frame_buffer_.try_pop(frame) && is_running_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        if (!frame || !is_running_)
            return;

        // Get the Vulkan texture from the composited frame
        auto src = std::dynamic_pointer_cast<accelerator::vulkan::texture>(frame.texture());
        if (!src) {
            // No Vulkan texture — likely OGL mixer or empty frame; skip
            return;
        }

        caspar::timer frame_timer;

        {
            auto lock = queue_obj->scoped_lock();

            swapchain_->wait_fence();

            uint32_t image_index = swapchain_->acquire_next_image();
            if (image_index == std::numeric_limits<uint32_t>::max()) {
                CASPAR_LOG(warning) << print() << L" Swapchain out of date.";
                return;
            }

            if (color_pipeline_ && color_pipeline_->is_active()) {
                swapchain_->blit_via_compute_and_present(
                    src->id(), static_cast<uint32_t>(src->width()), static_cast<uint32_t>(src->height()),
                    *color_pipeline_, image_index);
            } else {
                swapchain_->blit_and_present(
                    src->id(), static_cast<uint32_t>(src->width()), static_cast<uint32_t>(src->height()), image_index);
            }
        }

        graph_->set_value("frame-time", frame_timer.elapsed() * format_desc_.fps * 0.5);
        graph_->set_value("tick-time", tick_timer_.elapsed() * format_desc_.fps * 0.5);
        tick_timer_.restart();
    }

    configuration                                config_;
    std::shared_ptr<accelerator::vulkan::device> device_;
    core::video_format_desc                      format_desc_;
    int                                          channel_index_ = 0;

    std::unique_ptr<output_window>          window_;
    std::unique_ptr<present_swapchain>      swapchain_;
    std::unique_ptr<color_convert_pipeline> color_pipeline_;

    spl::shared_ptr<diagnostics::graph>            graph_ = spl::make_shared<diagnostics::graph>();
    caspar::timer                                  tick_timer_;
    tbb::concurrent_bounded_queue<core::const_frame> frame_buffer_;

    std::atomic<bool> is_running_{true};
    std::thread       thread_;
};

// ----------------------------------------------------------------------------
// Proxy (implements core::frame_consumer interface)
// ----------------------------------------------------------------------------

class vulkan_output_consumer_proxy : public core::frame_consumer
{
  public:
    vulkan_output_consumer_proxy(configuration                                config,
                                 std::shared_ptr<accelerator::vulkan::device> device)
        : config_(std::move(config))
        , device_(std::move(device))
    {
    }

    void initialize(const core::video_format_desc& format_desc,
                    const core::channel_info&      channel_info,
                    int                            /*port_index*/) override
    {
        impl_ = std::make_unique<vulkan_output_consumer_impl>(config_, device_);
        impl_->initialize(format_desc, channel_info.index);
    }

    std::future<bool> send(core::video_field field, core::const_frame frame) override
    {
        return impl_->send(field, frame);
    }

    std::wstring print() const override
    {
        return impl_ ? impl_->print() : L"vulkan_output[not initialized]";
    }

    std::wstring name() const override { return L"vulkan-output"; }

    bool has_synchronization_clock() const override { return false; }

    bool needs_host_frame() const override { return false; }

    int index() const override { return 700 + config_.output_index; }

    core::monitor::state state() const override
    {
        core::monitor::state s;
        s["vulkan-output/index"]   = config_.output_index;
        s["vulkan-output/gpu"]     = config_.gpu_index;
        s["vulkan-output/running"] = impl_ ? impl_->is_running() : false;
        return s;
    }

  private:
    configuration                                config_;
    std::shared_ptr<accelerator::vulkan::device> device_;
    std::unique_ptr<vulkan_output_consumer_impl> impl_;
};

} // anonymous namespace

// ----------------------------------------------------------------------------
// Factory functions
// ----------------------------------------------------------------------------

spl::shared_ptr<core::frame_consumer>
create_consumer(const std::shared_ptr<accelerator::vulkan::device>&      device,
                const std::vector<std::wstring>&                         params,
                const core::video_format_repository&                     /*format_repository*/,
                const std::vector<spl::shared_ptr<core::video_channel>>& /*channels*/,
                const core::channel_info&                                /*channel_info*/)
{
    if (params.empty() || !boost::iequals(params.at(0), L"VULKAN_OUTPUT"))
        return core::frame_consumer::empty();

    configuration config;

    // VULKAN_OUTPUT [output_index] [gpu_index]
    if (params.size() > 1) {
        try {
            config.output_index = std::stoi(params.at(1));
        } catch (...) {
        }
    }
    if (params.size() > 2) {
        try {
            config.gpu_index = std::stoi(params.at(2));
        } catch (...) {
        }
    }

    return spl::make_shared<vulkan_output_consumer_proxy>(std::move(config), device);
}

spl::shared_ptr<core::frame_consumer>
create_preconfigured_consumer(const std::shared_ptr<accelerator::vulkan::device>&      device,
                              const boost::property_tree::wptree&                      ptree,
                              const core::video_format_repository&                     /*format_repository*/,
                              const std::vector<spl::shared_ptr<core::video_channel>>& /*channels*/,
                              const core::channel_info&                                /*channel_info*/)
{
    auto config = parse_config(ptree);
    return spl::make_shared<vulkan_output_consumer_proxy>(std::move(config), device);
}

}} // namespace caspar::vulkan_output
