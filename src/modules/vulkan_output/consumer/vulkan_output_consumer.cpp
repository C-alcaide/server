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
#include "../util/nvapi_helpers.h"
#include "../util/output_window.h"
#include "../util/presentation.h"
#include "../util/startup_gate.h"

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
#include <mutex>
#include <thread>

namespace caspar { namespace vulkan_output {

namespace {

// ─── TDR Watchdog ───────────────────────────────────────────────────────────
// After a GPU reset (TDR), Vulkan calls may block indefinitely, creating zombie
// processes. This detached thread ensures forced termination after a grace period.
static std::once_flag tdr_watchdog_flag;

void start_tdr_watchdog(int grace_seconds = 10)
{
    std::call_once(tdr_watchdog_flag, [grace_seconds] {
        std::thread([grace_seconds] {
            CASPAR_LOG(error) << L"[vulkan_output] TDR detected (VK_ERROR_DEVICE_LOST). "
                                 L"Terminating in " << grace_seconds << L"s if shutdown does not complete.";
            std::this_thread::sleep_for(std::chrono::seconds(grace_seconds));
#ifdef _WIN32
            TerminateProcess(GetCurrentProcess(), 1);
#else
            kill(getpid(), SIGKILL);
#endif
        }).detach();
    });
}

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
                      vk::PresentModeKHR  present_mode,
                      vk::SurfaceFormatKHR preferred_format = {})
        : instance_(instance)
        , physical_(physical)
        , device_(device)
        , queue_(queue)
        , queue_family_(queue_family)
        , surface_(surface)
        , tier_(tier)
        , present_mode_(present_mode)
        , preferred_format_(preferred_format)
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
    vk::SwapchainKHR swapchain() const { return swapchain_; }

    // Returns UINT32_MAX if swapchain needs recreation (shouldn't happen for FSE)
    // Sets out_result to the acquire result for error detection.
    uint32_t acquire_next_image(vk::Result& out_result)
    {
        auto result = device_.acquireNextImageKHR(
            swapchain_, std::numeric_limits<uint64_t>::max(), current_frame().image_available, nullptr);

        out_result = result.result;
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
    // src_x/src_y define the top-left crop offset; src_width/src_height the crop size.
    // Returns the vk::Result from vkQueuePresentKHR.
    vk::Result blit_and_present(VkImage  src_image,
                          uint32_t src_x,
                          uint32_t src_y,
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
        region.srcOffsets[0]  = vk::Offset3D{static_cast<int32_t>(src_x), static_cast<int32_t>(src_y), 0};
        region.srcOffsets[1]  = vk::Offset3D{static_cast<int32_t>(src_x + src_width), static_cast<int32_t>(src_y + src_height), 1};
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

        current_frame_idx_ = (current_frame_idx_ + 1) % MAX_FRAMES_IN_FLIGHT;
        return present_result;
    }

    // Record and submit: blit src → intermediate, dispatch color compute, blit intermediate → swapchain.
    // Returns the vk::Result from vkQueuePresentKHR.
    vk::Result blit_via_compute_and_present(VkImage                  src_image,
                                      uint32_t                 src_x,
                                      uint32_t                 src_y,
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
        region_to_int.srcOffsets[0]  = vk::Offset3D{static_cast<int32_t>(src_x), static_cast<int32_t>(src_y), 0};
        region_to_int.srcOffsets[1]  = vk::Offset3D{static_cast<int32_t>(src_x + src_width), static_cast<int32_t>(src_y + src_height), 1};
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

        current_frame_idx_ = (current_frame_idx_ + 1) % MAX_FRAMES_IN_FLIGHT;
        return present_result;
    }

    // Recreate the swapchain (e.g., after display reconnect or OUT_OF_DATE).
    // Returns false if surface is no longer valid (display disconnected).
    bool recreate(int desired_images)
    {
        // Wait for all GPU work to finish
        auto wait_result = device_.waitForFences(
            {frames_[0].in_flight, frames_[1].in_flight, frames_[2].in_flight},
            VK_TRUE, std::numeric_limits<uint64_t>::max());
        if (static_cast<VkResult>(wait_result) == VK_ERROR_DEVICE_LOST)
            return false;

        // Verify surface is still valid
        try {
            auto caps = physical_.getSurfaceCapabilitiesKHR(surface_);
            (void)caps;
        } catch (const vk::SurfaceLostKHRError&) {
            return false;
        } catch (const vk::SystemError&) {
            return false;
        }

        // Destroy old swapchain resources
        for (auto& iv : image_views_)
            device_.destroyImageView(iv);
        image_views_.clear();
        images_.clear();
        device_.destroySwapchainKHR(swapchain_);
        swapchain_ = nullptr;

        // Recreate
        create_swapchain(desired_images);
        current_frame_idx_ = 0;
        return true;
    }

  private:
    frame_sync& current_frame() { return frames_[current_frame_idx_]; }

    void create_swapchain(int desired_images)
    {
        auto caps = physical_.getSurfaceCapabilitiesKHR(surface_);
        auto formats = physical_.getSurfaceFormatsKHR(surface_);

        // Use preferred format if it was explicitly set and is available
        vk::SurfaceFormatKHR chosen{};
        if (preferred_format_.format != vk::Format::eUndefined) {
            for (const auto& f : formats) {
                if (f.format == preferred_format_.format && f.colorSpace == preferred_format_.colorSpace) {
                    chosen = f;
                    break;
                }
            }
        }

        // Fallback: prefer BGRA8 SRGB
        if (chosen.format == vk::Format::eUndefined) {
            chosen = formats[0];
            for (const auto& f : formats) {
                if (f.format == vk::Format::eB8G8R8A8Srgb && f.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
                    chosen = f;
                    break;
                }
            }
            if (chosen.format == vk::Format::eUndefined)
                chosen.format = vk::Format::eB8G8R8A8Unorm;
        }

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
    vk::SurfaceFormatKHR       preferred_format_;
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
// HDR surface format selection + metadata
// ----------------------------------------------------------------------------

// Pick a surface format appropriate for the configured transfer/EOTF:
//   HDR (PQ/HLG): prefer RGBA16F + extended sRGB linear (scRGB),
//                  fall back to A2B10G10R10 + HDR10 ST2084
//   SDR:          returns default (empty) — swapchain picks BGRA8 SRGB
vk::SurfaceFormatKHR pick_hdr_surface_format(vk::PhysicalDevice physical,
                                             vk::SurfaceKHR     surface,
                                             hdr_transfer        transfer)
{
    if (transfer == hdr_transfer::sdr)
        return {}; // Use default SDR format

    auto formats = physical.getSurfaceFormatsKHR(surface);

    // Priority 1: RGBA16F + extended sRGB linear (scRGB — Windows HDR compositing)
    for (const auto& f : formats) {
        if (f.format == vk::Format::eR16G16B16A16Sfloat &&
            f.colorSpace == vk::ColorSpaceKHR::eExtendedSrgbLinearEXT) {
            return f;
        }
    }

    // Priority 2: A2B10G10R10 + HDR10 ST2084
    for (const auto& f : formats) {
        if (f.format == vk::Format::eA2B10G10R10UnormPack32 &&
            f.colorSpace == vk::ColorSpaceKHR::eHdr10St2084EXT) {
            return f;
        }
    }

    // Priority 3: RGBA16F with any HDR color space
    for (const auto& f : formats) {
        if (f.format == vk::Format::eR16G16B16A16Sfloat &&
            f.colorSpace != vk::ColorSpaceKHR::eSrgbNonlinear) {
            return f;
        }
    }

    CASPAR_LOG(warning) << L"[vulkan_output] No HDR-capable surface format found; falling back to SDR.";
    return {};
}

// Set HDR metadata (VK_EXT_hdr_metadata) on the swapchain.
// Only call after swapchain creation and only if HDR format was selected.
void set_hdr_metadata(vk::Device device, vk::SwapchainKHR swapchain, output_gamut gamut, float max_luminance)
{
    // Try to get the function pointer
    auto fn = reinterpret_cast<PFN_vkSetHdrMetadataEXT>(
        VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr(device, "vkSetHdrMetadataEXT"));
    if (!fn) {
        CASPAR_LOG(debug) << L"[vulkan_output] vkSetHdrMetadataEXT not available.";
        return;
    }

    VkHdrMetadataEXT metadata{};
    metadata.sType = VK_STRUCTURE_TYPE_HDR_METADATA_EXT;

    if (gamut == output_gamut::bt2020) {
        // BT.2020 primaries
        metadata.displayPrimaryRed   = {0.708f, 0.292f};
        metadata.displayPrimaryGreen = {0.170f, 0.797f};
        metadata.displayPrimaryBlue  = {0.131f, 0.046f};
    } else if (gamut == output_gamut::p3_d65) {
        // DCI-P3 (D65)
        metadata.displayPrimaryRed   = {0.680f, 0.320f};
        metadata.displayPrimaryGreen = {0.265f, 0.690f};
        metadata.displayPrimaryBlue  = {0.150f, 0.060f};
    } else {
        // BT.709
        metadata.displayPrimaryRed   = {0.640f, 0.330f};
        metadata.displayPrimaryGreen = {0.300f, 0.600f};
        metadata.displayPrimaryBlue  = {0.150f, 0.060f};
    }

    metadata.whitePoint         = {0.3127f, 0.3290f}; // D65
    metadata.maxLuminance       = max_luminance;
    metadata.minLuminance       = 0.001f;
    metadata.maxContentLightLevel      = max_luminance;
    metadata.maxFrameAverageLightLevel = max_luminance * 0.5f;

    VkSwapchainKHR sc = swapchain;
    fn(device, 1, &sc, &metadata);

    CASPAR_LOG(info) << L"[vulkan_output] HDR metadata set: max_luminance=" << max_luminance
                     << L" gamut=" << static_cast<int>(gamut);
}

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
        // ─── NvAPI pre-surface automation ───────────────────────────────────
        // EDID emulation: inject synthetic EDID on disconnected outputs so
        // Windows enumerates them (must happen BEFORE display enumeration).
        if (config_.edid_emulation) {
            if (!nvapi_)
                nvapi_ = std::make_unique<nvapi_helpers>();
            if (nvapi_->is_available()) {
                uint32_t edid_w = config_.region_w > 0 ? config_.region_w : format_desc_.width;
                uint32_t edid_h = config_.region_h > 0 ? config_.region_h : format_desc_.height;
                injected_edid_display_id_ = nvapi_->inject_edid(
                    config_.gpu_index, config_.output_index, edid_w, edid_h, format_desc_.fps);
                if (injected_edid_display_id_ != 0) {
                    // Give Windows time to enumerate the new display
                    std::this_thread::sleep_for(std::chrono::milliseconds(2000));
                    CASPAR_LOG(info) << print() << L" EDID emulation active for output "
                                     << config_.output_index;
                }
            }
        }

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

        // Select HDR surface format if configured
        auto hdr_format = pick_hdr_surface_format(physical, surface, config_.transfer);

        swapchain_ = std::make_unique<present_swapchain>(vk_instance,
                                                         physical,
                                                         vk_device,
                                                         vk_queue,
                                                         queue_obj->family_index(),
                                                         surface,
                                                         config_.buffer_depth,
                                                         active_tier,
                                                         present_mode,
                                                         hdr_format);

        CASPAR_LOG(info) << print() << L" Started. Swapchain " << swapchain_->width() << L"x"
                         << swapchain_->height() << L" | " << tier_name(active_tier)
                         << L" | " << target->display_name;

        // Set HDR metadata if we got an HDR surface format
        if (hdr_format.format != vk::Format::eUndefined) {
            float max_nits = static_cast<float>(config_.max_cll);
            set_hdr_metadata(vk_device, swapchain_->swapchain(), config_.gamut, max_nits);
        }

        // ─── NvAPI post-swapchain automation ────────────────────────────────
        setup_nvapi();

        // Initialize color conversion pipeline (always created, conditionally dispatched)
        color_pipeline_ = std::make_unique<color_convert_pipeline>(
            vk_device, physical, swapchain_->width(), swapchain_->height());

        // Determine tone map op from transfer/EOTF combination
        // If hardware HDR is active, the display engine handles PQ+BT.2020 —
        // the color pipeline should pass linear data through (identity).
        int tone_map = 0;
        output_gamut effective_gamut = config_.gamut;
        output_eotf  effective_eotf = config_.eotf;
        float        effective_nits = static_cast<float>(config_.max_cll);

        if (hw_hdr_active_) {
            // Hardware HDR: display engine performs PQ encoding + gamut mapping.
            // Source stays linear sRGB primaries — skip compute shader conversion.
            effective_gamut = output_gamut::bt709;
            effective_eotf  = output_eotf::linear;
            tone_map        = 0;
        } else {
            if (config_.eotf == output_eotf::hlg)
                tone_map = 7; // hlg_ootf
            else if (config_.gamut == output_gamut::bt2020 && config_.eotf == output_eotf::pq)
                tone_map = 3; // aces_rrt for HDR10
        }

        color_pipeline_->update_config(effective_gamut, effective_eotf, effective_nits, tone_map);

        if (color_pipeline_->is_active()) {
            CASPAR_LOG(info) << print() << L" Color pipeline active: gamut="
                             << static_cast<int>(effective_gamut)
                             << L" eotf=" << static_cast<int>(effective_eotf);
        }

        // ─── Startup gate ───────────────────────────────────────────────────
        // Signal that this consumer has finished heavy initialization.
        // Then wait for all other consumers before entering the present loop.
        startup_gate::instance().signal_ready();
        startup_gate::instance().wait_all_ready();

        while (is_running_ && !device_dead_) {
            tick(queue_obj);
        }

        // ─── Cleanup ────────────────────────────────────────────────────────
        shutdown_nvapi();
        color_pipeline_.reset();
        swapchain_.reset();
        vk_instance.destroySurfaceKHR(surface);
        window_.reset();
    }

    // ─── NvAPI automation helpers ──────────────────────────────────────────
    void setup_nvapi()
    {
        // Only proceed if NvAPI features are configured
        if (!config_.edid_auto_hdr && !config_.persist_edid &&
            !config_.gsync_enabled && !config_.hardware_hdr)
            return;

        if (!nvapi_)
            nvapi_ = std::make_unique<nvapi_helpers>();
        if (!nvapi_->is_available())
            return;

        // EDID auto-detection: read connected monitor's EDID to discover HDR capabilities
        if (config_.edid_auto_hdr) {
            auto edid = nvapi_->read_edid(config_.gpu_index, config_.output_index);
            if (edid.supports_hdr && config_.transfer == hdr_transfer::sdr) {
                config_.transfer = hdr_transfer::pq;
                config_.gamut    = output_gamut::bt2020;
                config_.eotf     = output_eotf::pq;
                if (edid.max_luminance > 0)
                    config_.max_cll = static_cast<int>(edid.max_luminance);
                CASPAR_LOG(info) << print() << L" EDID auto-detected HDR (PQ) display: "
                                 << edid.manufacturer << L" " << edid.model
                                 << L" MaxCLL=" << config_.max_cll << L" cd/m²";
            }
        }

        // EDID persistence: lock the monitor's EDID so the display survives cable disconnect
        if (config_.persist_edid) {
            nvapi_->persist_edid(config_.gpu_index, config_.output_index);
        }

        // Quadro Sync configuration
        if (config_.gsync_enabled && nvapi_->gsync_device_count() > 0) {
            auto source = (config_.gsync_source == gsync_reference::external)
                              ? sync_source::house_sync
                              : sync_source::vsync;

            if (config_.gsync_master) {
                nvapi_->configure_sync(config_.gpu_index, config_.output_index, source);
            }

            auto status = nvapi_->get_sync_status(config_.gpu_index);
            CASPAR_LOG(info) << print() << L" Quadro Sync: "
                             << (status.synced ? L"LOCKED" : L"UNLOCKED")
                             << L" role=" << (status.role == sync_role::master ? L"master" : L"slave")
                             << (status.house_sync
                                     ? L" house_sync=" + std::to_wstring(status.house_sync_freq) + L"Hz"
                                     : L"");
        }

        // Hardware HDR: let the display engine perform PQ encoding + gamut mapping.
        // Source stays linear scRGB FP16 — no compute shader EOTF needed.
        if (config_.hardware_hdr &&
            (config_.transfer == hdr_transfer::pq || config_.transfer == hdr_transfer::hlg)) {
            nvapi_display_id_ = nvapi_->resolve_display_id(config_.gpu_index, config_.output_index);
            if (nvapi_display_id_ != 0 && nvapi_->supports_hdr_output(nvapi_display_id_)) {
                hw_hdr_active_ = nvapi_->enable_hdr_output(nvapi_display_id_, config_.max_cll, config_.max_fall);
                if (hw_hdr_active_) {
                    CASPAR_LOG(info) << print()
                                     << L" Hardware HDR active — display engine handles PQ + BT.2020.";
                }
            }
        }
    }

    void shutdown_nvapi()
    {
        // Remove injected EDID
        if (injected_edid_display_id_ != 0 && nvapi_) {
            nvapi_->remove_edid(config_.gpu_index, injected_edid_display_id_);
            injected_edid_display_id_ = 0;
        }

        // Disable hardware HDR
        if (hw_hdr_active_ && nvapi_display_id_ != 0 && nvapi_) {
            nvapi_->disable_hdr_output(nvapi_display_id_);
            hw_hdr_active_    = false;
            nvapi_display_id_ = 0;
        }

        nvapi_.reset();
    }

    void tick(const std::shared_ptr<accelerator::vulkan::vulkan_queue>& queue_obj)
    {
        // ─── Display disconnect state machine ───────────────────────────────
        if (display_lost_) {
            switch (config_.on_disconnect) {
                case disconnect_behavior::hold:
                    // Silently hold last frame — sleep to avoid busy spin
                    std::this_thread::sleep_for(std::chrono::milliseconds(16));
                    return;
                case disconnect_behavior::black:
                case disconnect_behavior::retry:
                default:
                    if (++hotplug_retry_counter_ % 50 == 0) {
                        // Attempt swapchain recreation every ~50 ticks
                        if (swapchain_->recreate(config_.buffer_depth)) {
                            display_lost_ = false;
                            hotplug_retry_counter_ = 0;
                            CASPAR_LOG(info) << print() << L" Display reconnected — swapchain recreated.";
                        }
                    }
                    if (display_lost_) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(16));
                        return;
                    }
                    break;
            }
        }

        if (device_dead_)
            return;

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

        // Compute source region (subregion crop or full frame)
        uint32_t crop_x = static_cast<uint32_t>(config_.src_x);
        uint32_t crop_y = static_cast<uint32_t>(config_.src_y);
        uint32_t crop_w = config_.region_w > 0 ? static_cast<uint32_t>(config_.region_w)
                                               : static_cast<uint32_t>(src->width());
        uint32_t crop_h = config_.region_h > 0 ? static_cast<uint32_t>(config_.region_h)
                                               : static_cast<uint32_t>(src->height());

        vk::Result present_result = vk::Result::eSuccess;

        try {
            auto lock = queue_obj->scoped_lock();

            swapchain_->wait_fence();

            vk::Result acquire_result{};
            uint32_t image_index = swapchain_->acquire_next_image(acquire_result);

            if (acquire_result == vk::Result::eErrorSurfaceLostKHR) {
                display_lost_ = true;
                CASPAR_LOG(warning) << print() << L" Surface lost — display disconnected.";
                return;
            }

            if (image_index == std::numeric_limits<uint32_t>::max()) {
                // OUT_OF_DATE or SUBOPTIMAL — recreate swapchain
                if (!swapchain_->recreate(config_.buffer_depth)) {
                    display_lost_ = true;
                    CASPAR_LOG(warning) << print() << L" Swapchain recreation failed — display lost.";
                }
                return;
            }

            if (color_pipeline_ && color_pipeline_->is_active()) {
                present_result = swapchain_->blit_via_compute_and_present(
                    src->id(), crop_x, crop_y, crop_w, crop_h,
                    *color_pipeline_, image_index);
            } else {
                present_result = swapchain_->blit_and_present(
                    src->id(), crop_x, crop_y, crop_w, crop_h, image_index);
            }
        } catch (const vk::DeviceLostError&) {
            device_dead_ = true;
            start_tdr_watchdog();
            is_running_ = false;
            return;
        } catch (const vk::SystemError& e) {
            if (static_cast<VkResult>(e.code().value()) == VK_ERROR_DEVICE_LOST) {
                device_dead_ = true;
                start_tdr_watchdog();
                is_running_ = false;
                return;
            }
            throw;
        }

        // Handle present result
        switch (present_result) {
            case vk::Result::eSuccess:
            case vk::Result::eSuboptimalKHR:
                break;
            case vk::Result::eErrorOutOfDateKHR:
                if (!swapchain_->recreate(config_.buffer_depth))
                    display_lost_ = true;
                break;
            case vk::Result::eErrorSurfaceLostKHR:
                display_lost_ = true;
                CASPAR_LOG(warning) << print() << L" Surface lost during present.";
                break;
            case vk::Result::eErrorDeviceLost:
                device_dead_ = true;
                start_tdr_watchdog();
                is_running_ = false;
                return;
            default:
                // Includes eErrorFullScreenExclusiveModeLostEXT if available
                break;
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

    // NvAPI automation state
    std::unique_ptr<nvapi_helpers> nvapi_;
    uint32_t                      nvapi_display_id_          = 0;
    bool                          hw_hdr_active_             = false;
    uint32_t                      injected_edid_display_id_  = 0;

    // Display disconnect / TDR recovery state
    std::atomic<bool> display_lost_{false};
    std::atomic<bool> device_dead_{false};
    uint64_t          hotplug_retry_counter_ = 0;

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
