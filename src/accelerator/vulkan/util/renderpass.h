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

#include <common/bit_depth.h>
#include <common/memory.h>
#include <common/render_format.h>
#include <tbb/concurrent_queue.h>
#include <tbb/concurrent_unordered_map.h>
#include <vector>
#include <vulkan/vulkan.hpp>

#include "draw_params.h"
#include "uniform_block.h"

namespace caspar { namespace accelerator { namespace vulkan {

using draw_data = std::pair<std::vector<core::frame_geometry::coord>, uniform_block>;

/// Optional LUT image views returned by the kernel alongside draw_data.
/// These fill descriptor bindings 3 (3D LUT), 4 (hue curve), 5 (curve LUT),
/// 6 (blend mask).
struct lut_views
{
    vk::ImageView lut3d      = nullptr;
    vk::ImageView hue_curve  = nullptr;
    vk::ImageView curve_lut  = nullptr;
    vk::ImageView blend_mask = nullptr;
};
struct frame_context
{
    virtual vk::Buffer                      upload_vertex_data(const std::vector<float>& data) = 0;
    virtual draw_data                       create_draw_data(const draw_params& params)        = 0;
    virtual lut_views                       get_lut_views() const                              { return {}; }
    virtual void                            upload_pending_luts(vk::CommandBuffer cmd)          {}
    virtual std::shared_ptr<class pipeline> get_pipeline()                                     = 0;
    virtual vk::CommandBuffer               get_command_buffer()                               = 0;
    virtual void                            submit()                                           = 0;
    virtual void                            wait_for_completion()                              = 0;
    /// Returns a Win32 HANDLE for an exportable VkSemaphore that is signaled
    /// when the render command buffer completes on the GPU.  Returns nullptr
    /// if not available.  The handle is owned by the frame_context and must
    /// NOT be closed by the caller.
    virtual void*                           render_complete_semaphore_handle()                  { return nullptr; }
    /// Returns the timeline semaphore value signaled by the most recent submit().
    virtual uint64_t                        render_complete_semaphore_value()                   { return 0; }
    /// An attachment in whatever format this context composites in.
    virtual std::shared_ptr<class texture>
    create_attachment(uint32_t width, uint32_t height, uint32_t components_count) = 0;

    /// An attachment in an explicitly chosen format. The resolve target is the one caller
    /// that needs this, because it is by definition a different format from the working
    /// space it is resolving.
    virtual std::shared_ptr<class texture>
    create_attachment_as(uint32_t width, uint32_t height, uint32_t components_count, common::render_format format) = 0;
};

class renderpass
{
    frame_context*                  _ctx;
    std::shared_ptr<class pipeline> _pipeline;
    uint32_t                        _width;
    uint32_t                        _height;

    std::shared_ptr<class texture> _default_attachment;
    std::shared_ptr<class texture> _resolve_target;
    // Set by commit() to whichever attachment was rendered to last, so result_attachment()
    // is correct even when a post-process pass redirected the output.
    std::shared_ptr<class texture> _final_attachment;

    struct layer_info
    {
        std::shared_ptr<class texture>           attachment;
        std::shared_ptr<class texture>           local_key_attachment;
        std::shared_ptr<class texture>           layer_key_attachment;
        std::array<vk::ImageView, 11>            textures;
        std::vector<core::frame_geometry::coord> coords;
        uniform_block                            uniforms;
        uint32_t                                 vertex_buffer_offset = 0;
    };
    std::vector<layer_info> layers_;

  public:
    renderpass(frame_context* ctx, uint32_t width, uint32_t height);

    renderpass()                             = delete;
    renderpass(const renderpass&)            = delete;
    renderpass& operator=(const renderpass&) = delete;

    ~renderpass();
    std::shared_ptr<class texture> create_attachment(uint32_t components_count = 4);

    /// A full-size attachment in `format`, for use as commit()'s resolve target.
    std::shared_ptr<class texture> create_attachment_as(common::render_format format,
                                                        uint32_t              components_count = 4);

    /// Ask commit() to blit the final attachment into `target` inside the same command
    /// buffer, converting format on the way.
    ///
    /// Required when this pass composites into a float attachment: everything downstream
    /// of the mixer -- copy_async(), texture_wrapper, every consumer -- means integer, and
    /// a half-float image reaching them is reinterpreted as unsigned shorts, which is
    /// garbage rather than merely clipped.
    ///
    /// It has to happen here rather than in a separate submit. The composite's work is
    /// only ordered by this command buffer, so a standalone submitSingleTimeCommands()
    /// blit would race it, and waiting for the pass first would stall the pipeline every
    /// frame -- defeating the semaphore handoff that lets the channel tick continue while
    /// the previous frame is still in flight.
    void set_resolve_target(std::shared_ptr<class texture> target) { _resolve_target = std::move(target); }

    /// The image callers should treat as the pass's result: the resolve target when one
    /// was set, otherwise whichever attachment was rendered to last.
    std::shared_ptr<class texture> result_attachment() const;

    void                           draw(const draw_params& params);
    virtual void                   commit();
    void                           wait_for_completion() { _ctx->wait_for_completion(); }
    void*                          render_semaphore_handle() { return _ctx->render_complete_semaphore_handle(); }
    uint64_t                       render_semaphore_value() { return _ctx->render_complete_semaphore_value(); }

    std::shared_ptr<class texture> default_attachment() const { return _default_attachment; }
    size_t                         layer_count() const { return layers_.size(); }

  private:
    vk::Buffer upload_vertex_buffers();
};

}}} // namespace caspar::accelerator::vulkan
