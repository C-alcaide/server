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

#pragma once

#include <accelerator/accelerator.h>
#include <common/array.h>
#include <common/bit_depth.h>
#include <common/render_format.h>
#include <vector>
#include <string>
#include <core/frame/geometry.h>

#include <functional>
#include <future>

#include <vulkan/vulkan.hpp>

namespace caspar { namespace accelerator { namespace vulkan {

struct draw_params;

class image_kernel;

class device final
    : public std::enable_shared_from_this<device>
    , public accelerator_device
{
  public:
    // gpu_index selects which physical GPU the mixer runs on (0-based, deduplicated
    // by LUID to match the indexing used by the vulkan_output consumer).
    explicit device(int gpu_index = 0);
    ~device();

    device(const device&) = delete;

    device& operator=(const device&) = delete;

    /// The pipeline built against the attachment format this (depth, render_format) pair
    /// implies. A pipeline is bound to one attachment format, so fp16 needs its own.
    /// The pipeline for a generated colour transform, built on first request.
    ///
    /// Keyed on (variant_id, attachment format): a Vulkan pipeline is bound to both its
    /// shader module and its attachment format, so the same transform on an 8-bit and an
    /// fp16 channel are two pipelines. An empty variant_id or empty SPIR-V returns the base
    /// pipeline rather than building a duplicate of it.
    ///
    /// Builds on the calling thread and must not be called from the frame path: this is the
    /// driver's SPIR-V-to-ISA step on top of shaderc's compile.
    std::shared_ptr<class pipeline> get_variant_pipeline(common::bit_depth            depth,
                                                        common::render_format        render_format,
                                                        const std::string&           variant_id,
                                                        const std::vector<uint32_t>& frag_spirv);

    std::shared_ptr<class pipeline> get_pipeline(common::bit_depth     depth,
                                                common::render_format render_format = common::render_format::unorm);
    std::pair<vk::Buffer, vk::DeviceMemory>
    upload_vertex_buffer(const std::vector<core::frame_geometry::coord>& coords);

    vk::PhysicalDeviceMemoryProperties getMemoryProperties();
    std::vector<vk::CommandBuffer>     allocateCommandBuffers(uint32_t count);
    void                               submit(const vk::SubmitInfo& submitInfo, vk::Fence fence);
    vk::Device                         getVkDevice() const;
    vk::PhysicalDevice                 getVkPhysicalDevice() const;

    /// Enough of this device to hand it to another API that wants to share it -- see
    /// device.cpp for what FFmpeg's `AVVulkanDeviceContext` asks for and why the single
    /// graphics queue family matters.
    vk::Instance                       getVkInstance() const;
    /// A queue reserved for another API (an FFmpeg Vulkan decoder) to submit on, so that
    /// sharing this device does not mean sharing a queue. Check
    /// `hasDedicatedDecodeQueue()` first: on a GPU with no separate compute family this
    /// falls back to the graphics queue's family, which is not isolation.
    /// The device extensions this device enabled. Another API sharing it must be told
    /// what it may rely on; FFmpeg's `enabled_dev_extensions` is a declaration by the
    /// application, not something it can query.
    const std::vector<std::string>&    getEnabledDeviceExtensions() const;
    vk::Queue                          getDecodeQueue() const;
    uint32_t                           getDecodeQueueFamily() const;
    bool                               hasDedicatedDecodeQueue() const;
    /// The VIDEO ENCODE queue family, for an FFmpeg Vulkan *encoder* sharing this device.
    ///
    /// Separate from the decode accessors above because it is a different thing: those hand
    /// out a COMPUTE family (which is all `prores_vulkan` and friends need), while
    /// `h264_vulkan`/`hevc_vulkan` are `VK_KHR_video_encode` codecs and need a family with
    /// `VK_QUEUE_VIDEO_ENCODE_BIT_KHR`. Declaring the compute family for those is what made
    /// VP9 *fault* rather than decline on the decode side.
    ///
    /// `hasEncodeQueue()` is false on a GPU with no such family, and the caller must then
    /// refuse rather than substitute another one.
    uint32_t                           getEncodeQueueFamily() const;
    bool                               hasEncodeQueue() const;
    uint32_t                           getGraphicsQueueFamily() const;
    PFN_vkGetInstanceProcAddr          getInstanceProcAddr() const;
    vk::CommandPool                    getCommandPool() const;

    /// `render_format` selects the attachment's numeric format. unorm is the historical
    /// behaviour; fp16 gives a render target that can carry negatives and values above
    /// 1.0, for a linear working space. It participates in the attachment pool key,
    /// because a VkImage's format is fixed at creation.
    std::shared_ptr<class texture> create_attachment(int                   width,
                                                     int                   height,
                                                     common::bit_depth     depth,
                                                     uint32_t              components_count,
                                                     common::render_format render_format = common::render_format::unorm);
    // Transitions an attachment texture to eRenderingLocalRead before it is
    // reused as a render target. create_attachment() does this internally for
    // every texture it returns (new or pooled); callers that keep their own
    // cache of attachments across frames (see image_kernel's per-slot pool)
    // must call this themselves on a cache hit, since they bypass create_attachment().
    void reset_attachment_layout(const std::shared_ptr<class texture>& tex);
    std::shared_ptr<class texture> create_texture(int width, int height, int stride, common::bit_depth depth);

    /// Can this GPU sample a packed `stride`-component image of `depth`, i.e. will
    /// create_texture() / copy_async() accept that layout?
    ///
    /// Ask before uploading a plane whose stride is 3. Vulkan does not oblige an
    /// implementation to support a 3-component format as a sampled image and this GPU does
    /// not, so create_texture() throws for one -- from inside the channel's tick, on every
    /// frame, which pinned a CPU core and blanked the output rather than dropping a layer.
    bool can_sample_packed(int stride, common::bit_depth depth) const;
    std::shared_ptr<class texture>
    create_exportable_texture(int width, int height, int stride, common::bit_depth depth);
    array<uint8_t>                 create_array(int size);

    std::future<std::shared_ptr<class texture>>
    copy_async(const array<const uint8_t>& source, int width, int height, int stride, common::bit_depth depth);
    std::future<std::shared_ptr<class texture>>
    copy_compressed_async(const array<const uint8_t>& source, int width, int height, vk::Format format);
    std::future<array<const uint8_t>> copy_async(const std::shared_ptr<class texture>& source);

    /// Box-filtered downscale of `source` by `levels` successive exact halvings,
    /// returned as a new texture. Blocking; safe to call from a consumer thread.
    ///
    /// Each pass is a vkCmdBlitImage with VK_FILTER_LINEAR into an exactly-halved
    /// target, which weights the four contributing texels equally -- a true 2x2 box
    /// average, so `levels` of them compose into a 2^levels box filter. Intended
    /// for a consumer that needs a summary of the picture rather than the picture;
    /// see core::texture::read_pixels_reduced().
    ///
    /// A blit maps components, not bytes, so the result carries the source's channel
    /// order into 8 bits unchanged -- BGRA from an 8-bit attachment, RGBA from a
    /// 16-bit one, since only the 8-bit shader path swizzles. read_pixels_reduced()
    /// normalises that to the BGRA it promises its callers; do not do it here, or the
    /// 8-bit case gets swapped twice.
    ///
    /// The result is allocated through create_attachment(), not create_texture():
    /// pooled textures carry only eTransferDst|eSampled, and a blit source and a
    /// readback source both need eTransferSrc. It is returned in
    /// eColorAttachmentOptimal, which is exactly what copy_async() above assumes,
    /// so the two compose without a further transition.
    ///
    /// `source` is left in eTransferSrcOptimal. That is safe for a mixer attachment
    /// because the attachment pool calls reset_attachment_layout() (from
    /// eUndefined) before reusing one -- the same reason copy_async() may leave it
    /// that way.
    ///
    /// Returns nullptr if the reduction fails.
    std::shared_ptr<class texture> reduce_texture(const std::shared_ptr<class texture>& source, int levels);

    template <typename Func>
    auto dispatch_async(Func&& func)
    {
        using result_type = decltype(func());
        using task_type   = std::packaged_task<result_type()>;

        auto task   = std::make_shared<task_type>(std::forward<Func>(func));
        auto future = task->get_future();
        dispatch([=] { (*task)(); });
        return future;
    }

    template <typename Func>
    auto dispatch_sync(Func&& func)
    {
        return dispatch_async(std::forward<Func>(func)).get();
    }

    std::wstring version() const;

    boost::property_tree::wptree info() const;
    std::future<void>            gc();

  private:
    void dispatch(std::function<void()> func);
    struct impl;
    std::shared_ptr<impl> impl_;
};

}}} // namespace caspar::accelerator::vulkan
