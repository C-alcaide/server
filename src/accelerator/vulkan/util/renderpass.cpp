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

#include "renderpass.h"

#include <algorithm>
#include "../image/image_kernel.h"
#include "device.h"
#include "pipeline.h"
#include "texture.h"

namespace caspar { namespace accelerator { namespace vulkan {

vk::Buffer renderpass::upload_vertex_buffers()
{
    uint32_t total_coords = 0;
    for (auto& layer : layers_) {
        layer.vertex_buffer_offset = total_coords * 6 * sizeof(float);
        total_coords += static_cast<uint32_t>(layer.coords.size());
    }
    std::vector<float> fl(total_coords * 6);

    size_t idx = 0;
    for (auto& layer : layers_) {
        for (auto& c : layer.coords) {
            fl[idx * 6 + 0] = static_cast<float>(c.vertex_x);
            fl[idx * 6 + 1] = static_cast<float>(c.vertex_y);
            fl[idx * 6 + 2] = static_cast<float>(c.texture_x);
            fl[idx * 6 + 3] = static_cast<float>(c.texture_y);
            fl[idx * 6 + 4] = static_cast<float>(c.texture_r);
            fl[idx * 6 + 5] = static_cast<float>(c.texture_q);
            ++idx;
        }
    }

    return _ctx->upload_vertex_data(fl);
}

renderpass::renderpass(frame_context* ctx, uint32_t width, uint32_t height)
    : _ctx(ctx)
    , _pipeline(ctx->get_pipeline())
    , _width(width)
    , _height(height)
    , _default_attachment(ctx->create_attachment(width, height, 4))
{
}

renderpass::~renderpass() {}

std::shared_ptr<texture> renderpass::create_attachment(uint32_t components_count)
{
    return _ctx->create_attachment(_width, _height, components_count);
}

std::shared_ptr<texture> renderpass::create_attachment_as(common::render_format format, uint32_t components_count)
{
    return _ctx->create_attachment_as(_width, _height, components_count, format);
}

void renderpass::adopt_as_shader_read(const std::shared_ptr<texture>& tex)
{
    if (tex)
        adopted_shader_read_.push_back(tex);
}

std::shared_ptr<texture> renderpass::create_attachment_sized(uint32_t              width,
                                                            uint32_t              height,
                                                            common::render_format format,
                                                            uint32_t              components_count)
{
    // The `frame_context` virtual has always taken an explicit width and height, and
    // `frame_data` has always honoured them -- no caller ever passed anything but the channel
    // raster. An ISF `PASSES` buffer sized `"$WIDTH/16.0"` is the first that needs to.
    return _ctx->create_attachment_as(width, height, components_count, format);
}

std::shared_ptr<texture> renderpass::result_attachment() const
{
    if (_resolve_target)
        return _resolve_target;
    return _final_attachment ? _final_attachment : _default_attachment;
}

void renderpass::draw(const draw_params& params)
{
    auto attachment         = params.background;
    auto [coords, uniforms] = _ctx->create_draw_data(params);

    if (coords.empty()) {
        return;
    }

    std::array<vk::ImageView, 11> textures = {attachment->view(),
                                              nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                              nullptr, nullptr, nullptr, nullptr};

    for (int n = 0; n < params.textures.size(); ++n) {
        textures[1+n] = params.textures[n]->view();
    }
    if (params.local_key) {
        textures[5] = params.local_key->view();
    }
    if (params.layer_key) {
        textures[6] = params.layer_key->view();
    }

    // Fill LUT texture slots from the kernel's cached views
    auto luts = _ctx->get_lut_views();
    textures[7] = luts.lut3d;
    textures[8] = luts.hue_curve;
    textures[9] = luts.curve_lut;
    textures[10] = luts.blend_mask;

    layers_.push_back({
        attachment,
        params.local_key,
        params.layer_key,
        std::move(textures),
        luts.ocio,
        luts.ocio_nearest,
        _ctx->get_layer_pipeline(),
        std::move(coords),
        uniforms,
    });
}

void renderpass::commit()
{
    auto vertex_buffer = upload_vertex_buffers();

    auto cmd_buffer = _ctx->get_command_buffer();
    cmd_buffer.begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));

    // Upload any LUT textures that were updated during draw() calls.
    // Must happen before rendering starts (images need to reach ShaderReadOnlyOptimal).
    _ctx->upload_pending_luts(cmd_buffer);
    // Every attachment this frame took, transitioned in ONE place instead of one submit each.
    // Before any `beginRendering`, which is what makes the barriers ordered against the draws
    // that use those attachments.
    _ctx->apply_pending_layouts(cmd_buffer);

    vk::ClearValue clearColor{vk::ClearColorValue(std::array<float, 4>{0.0f, 0.0f, 0.0f, 0.0f})};

    // Viewport and scissor
    vk::Viewport viewport{0.0f, 0.0f, static_cast<float>(_width), static_cast<float>(_height), 0.0f, 1.0f};

    vk::Extent2D extent = {_width, _height};
    vk::Rect2D   scissor{{0, 0}, extent};

    // Tracks whichever attachment was actually rendered to last, so the closing
    // store barrier below transitions the real final image — not always
    // _default_attachment, which is wrong whenever a post-process pass (e.g.
    // calibration LUT) renders its result into a different attachment.
    std::shared_ptr<texture> previous_attachment = _default_attachment;

    if (layers_.empty()) {
        // No layers, just clear the default attachment
        vk::RenderingAttachmentInfo attachment_info{};
        attachment_info.imageView   = _default_attachment->view();
        attachment_info.imageLayout = vk::ImageLayout::eRenderingLocalRead;
        attachment_info.loadOp      = vk::AttachmentLoadOp::eClear;
        attachment_info.storeOp     = vk::AttachmentStoreOp::eStore;
        attachment_info.clearValue  = clearColor;

        vk::RenderingInfo rendering_info{};
        rendering_info.renderArea = scissor;
        rendering_info.layerCount = 1;
        rendering_info.setColorAttachments(attachment_info);

        cmd_buffer.beginRendering(rendering_info);
        cmd_buffer.setViewport(0, viewport);
        cmd_buffer.setScissor(0, scissor);
    } else {
        // create a renderpass for each layer
        bool default_cleared = false;
        previous_attachment  = nullptr; // force a fresh render pass to start below

        // ── ATTACHMENTS CURRENTLY HANDED TO THE SAMPLER ────────────────────────────────────
        //
        // An attachment that has been finished with is transitioned to `eShaderReadOnlyOptimal`
        // below so the next pass can sample it. Until this commit, nothing was ever RENDERED
        // INTO again afterwards -- the node evaluator took a fresh attachment per pass -- so the
        // reverse transition did not exist.
        //
        // It does now, because reusing attachments within a frame is the only way a chain's
        // memory stops growing with its length: the renderpass holds a reference to every
        // layer's attachment until `commit()`, so sixteen passes meant sixteen live 4K fp16
        // images per channel however promptly the evaluator dropped its own references.
        //
        // Re-entering one as a target needs BOTH halves: the layout back to
        // `eRenderingLocalRead`, and a write-after-read barrier against the sampling that the
        // earlier pass did. Neither is optional and neither is checkable by `vk-validation`,
        // which reports clean whatever you do -- `grade-graph`'s deep chain and diamond arms are
        // what actually adjudicate this.
        std::vector<const class texture*> in_shader_read;
        // A PERSISTENT buffer arrives already in that layout, from the frame before this one.
        // Seeding the list is what makes `take_back_for_writing` below emit the write-after-read
        // barrier for it instead of the `eUndefined` transition, which would discard exactly the
        // history a feedback effect exists to keep.
        for (const auto& adopted : adopted_shader_read_)
            in_shader_read.push_back(adopted.get());
        adopted_shader_read_.clear();
        const auto take_back_for_writing = [&](const std::shared_ptr<class texture>& tex) {
            const auto it = std::find(in_shader_read.begin(), in_shader_read.end(), tex.get());
            if (it == in_shader_read.end())
                return;
            in_shader_read.erase(it);

            vk::ImageMemoryBarrier2 b{};
            b.subresourceRange = vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);
            // WRITE AFTER READ: the earlier pass sampled this image in the fragment stage, and
            // this pass is about to write it as a colour attachment. Without the barrier the
            // write may land before that read has happened.
            b.srcStageMask  = vk::PipelineStageFlagBits2::eFragmentShader;
            b.srcAccessMask = vk::AccessFlagBits2::eShaderRead;
            b.dstStageMask  = vk::PipelineStageFlagBits2::eColorAttachmentOutput;
            b.dstAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite;
            b.oldLayout     = vk::ImageLayout::eShaderReadOnlyOptimal;
            b.newLayout     = vk::ImageLayout::eRenderingLocalRead;
            b.image         = tex->id();

            vk::DependencyInfo di{};
            di.setImageMemoryBarriers(b);
            cmd_buffer.pipelineBarrier2(di);
        };
        for (auto& layer : layers_) {
            if (layer.attachment != previous_attachment) {
                // We need to start a new render pass

                if (previous_attachment) {
                    // If this is not the first pass, end the previous render pass
                    cmd_buffer.endRendering();

                    if (previous_attachment != _default_attachment) {
                        // If we're done with a non-default attachment, we need to transition it to a shader read layout
                        vk::ImageMemoryBarrier2 memoryBarrier{};
                        auto range = vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);
                        memoryBarrier.subresourceRange = range;
                        memoryBarrier.srcStageMask     = vk::PipelineStageFlagBits2::eColorAttachmentOutput;
                        memoryBarrier.srcAccessMask    = vk::AccessFlagBits2::eColorAttachmentWrite;
                        memoryBarrier.dstStageMask     = vk::PipelineStageFlagBits2::eFragmentShader;
                        memoryBarrier.dstAccessMask    = vk::AccessFlagBits2::eInputAttachmentRead;
                        memoryBarrier.oldLayout        = vk::ImageLayout::eRenderingLocalRead;
                        memoryBarrier.newLayout        = vk::ImageLayout::eShaderReadOnlyOptimal;
                        memoryBarrier.image            = previous_attachment->id();

                        vk::DependencyInfo dependencyInfo{};
                        dependencyInfo.setImageMemoryBarriers(memoryBarrier);
                        cmd_buffer.pipelineBarrier2(dependencyInfo);

                        // Remember it is now the sampler's, so re-targeting it later takes it
                        // back rather than declaring a layout it is not in.
                        in_shader_read.push_back(previous_attachment.get());
                    }
                }

                // If this attachment was handed to the sampler earlier in the frame, take it
                // back before declaring `eRenderingLocalRead` for it.
                take_back_for_writing(layer.attachment);

                // We only want to clear the default attachment once
                bool do_clear = (layer.attachment != _default_attachment) || !default_cleared;

                vk::RenderingAttachmentInfo attachment_info{};
                attachment_info.imageView   = layer.attachment->view();
                attachment_info.imageLayout = vk::ImageLayout::eRenderingLocalRead;
                attachment_info.loadOp      = do_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad;
                attachment_info.storeOp     = vk::AttachmentStoreOp::eStore;
                attachment_info.clearValue  = clearColor;

                if (layer.attachment == _default_attachment) {
                    default_cleared = true;
                }

                previous_attachment = layer.attachment;

                // ── THE EXTENT IS THE ATTACHMENT'S, NOT THE CHANNEL'S ────────────────────
                //
                // Every layer used to be drawn with one viewport and one scissor taken from
                // `_width/_height`. That is right for every attachment the mixer had until now,
                // because `create_attachment*` could only make channel-sized ones -- and it is
                // wrong the moment a layer's target is smaller, which an ISF pass sized
                // `"$WIDTH/16.0"` is. A full-raster `renderArea` over a smaller attachment is
                // not a scaled draw, it is invalid.
                //
                // `_width/_height` still drive the empty-layers clear and the resolve blit
                // below, which really are channel-sized.
                const auto lw = layer.attachment ? static_cast<uint32_t>(layer.attachment->width())
                                                 : _width;
                const auto lh = layer.attachment ? static_cast<uint32_t>(layer.attachment->height())
                                                 : _height;
                const vk::Viewport layer_viewport{
                    0.0f, 0.0f, static_cast<float>(lw), static_cast<float>(lh), 0.0f, 1.0f};
                const vk::Rect2D layer_scissor{{0, 0}, vk::Extent2D{lw, lh}};

                vk::RenderingInfo rendering_info{};
                rendering_info.renderArea = layer_scissor;
                rendering_info.layerCount = 1;
                rendering_info.setColorAttachments(attachment_info);

                cmd_buffer.beginRendering(rendering_info);
                cmd_buffer.setViewport(0, layer_viewport);
                cmd_buffer.setScissor(0, layer_scissor);
            } else {
                // We are continuing in the same render pass, so we need a barrier to ensure the attachment is ready
                vk::MemoryBarrier2 memoryBarrier{};
                memoryBarrier.srcStageMask  = vk::PipelineStageFlagBits2::eColorAttachmentOutput;
                memoryBarrier.srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite;
                memoryBarrier.dstStageMask  = vk::PipelineStageFlagBits2::eFragmentShader;
                memoryBarrier.dstAccessMask = vk::AccessFlagBits2::eInputAttachmentRead;

                vk::DependencyInfo dependencyInfo{};
                dependencyInfo.dependencyFlags    = vk::DependencyFlagBits::eByRegion;
                dependencyInfo.memoryBarrierCount = 1;
                dependencyInfo.pMemoryBarriers    = &memoryBarrier;
                cmd_buffer.pipelineBarrier2(dependencyInfo);
            }

            // Per layer, not per pass. Layout-compatible with the base pipeline by
            // construction -- both declare the same two descriptor set layouts -- so
            // switching between them mid-pass needs no descriptor rebinding beyond what
            // draw() already does.
            const auto& layer_pipeline = layer.pipeline ? layer.pipeline : _pipeline;
            layer_pipeline->draw(cmd_buffer,
                                 vertex_buffer,
                                 static_cast<uint32_t>(layer.coords.size()),
                                 layer.vertex_buffer_offset,
                                 layer.uniforms,
                                 layer.textures,
                                 layer.ocio_textures,
                                 layer.ocio_nearest);
        }
    }

    cmd_buffer.endRendering();

    // Flush the render-pass store operation and transition the default
    // attachment out of eRenderingLocalRead (which may be tile-local on
    // some implementations) into a standard layout that later submits
    // can reliably transition from (e.g. for GPU→CPU readback or blit).
    {
        vk::ImageMemoryBarrier2 storeBarrier{};
        storeBarrier.oldLayout           = vk::ImageLayout::eRenderingLocalRead;
        storeBarrier.newLayout           = vk::ImageLayout::eColorAttachmentOptimal;
        storeBarrier.srcQueueFamilyIndex = vk::QueueFamilyIgnored;
        storeBarrier.dstQueueFamilyIndex = vk::QueueFamilyIgnored;
        storeBarrier.image               = previous_attachment->id();
        storeBarrier.subresourceRange =
            vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);
        storeBarrier.srcStageMask  = vk::PipelineStageFlagBits2::eColorAttachmentOutput;
        storeBarrier.srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite;
        storeBarrier.dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
        storeBarrier.dstAccessMask = vk::AccessFlagBits2::eMemoryRead;

        vk::DependencyInfo depInfo{};
        depInfo.setImageMemoryBarriers(storeBarrier);
        cmd_buffer.pipelineBarrier2(depInfo);
    }

    _final_attachment = previous_attachment;

    // Float working space -> the channel's output format, inside this same command buffer
    // so it is ordered after the composite without an extra submit or a stall. See
    // set_resolve_target().
    if (_resolve_target && _resolve_target != previous_attachment) {
        const auto src_img = previous_attachment->id();
        const auto dst_img = _resolve_target->id();

        // The store barrier above left the source in eColorAttachmentOptimal with
        // dstStageMask eAllCommands, which is what makes this transition legal here.
        {
            vk::ImageMemoryBarrier2 toSrc{};
            toSrc.oldLayout        = vk::ImageLayout::eColorAttachmentOptimal;
            toSrc.newLayout        = vk::ImageLayout::eTransferSrcOptimal;
            toSrc.srcQueueFamilyIndex = vk::QueueFamilyIgnored;
            toSrc.dstQueueFamilyIndex = vk::QueueFamilyIgnored;
            toSrc.image            = src_img;
            toSrc.subresourceRange = vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);
            toSrc.srcStageMask     = vk::PipelineStageFlagBits2::eColorAttachmentOutput;
            toSrc.srcAccessMask    = vk::AccessFlagBits2::eColorAttachmentWrite;
            toSrc.dstStageMask     = vk::PipelineStageFlagBits2::eTransfer;
            toSrc.dstAccessMask    = vk::AccessFlagBits2::eTransferRead;

            // The resolve target is overwritten whole, so eUndefined is honest and lets
            // the driver skip preserving contents. create_attachment hands it back in
            // eRenderingLocalRead.
            vk::ImageMemoryBarrier2 toDst{};
            toDst.oldLayout        = vk::ImageLayout::eUndefined;
            toDst.newLayout        = vk::ImageLayout::eTransferDstOptimal;
            toDst.srcQueueFamilyIndex = vk::QueueFamilyIgnored;
            toDst.dstQueueFamilyIndex = vk::QueueFamilyIgnored;
            toDst.image            = dst_img;
            toDst.subresourceRange = vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);
            toDst.srcStageMask     = vk::PipelineStageFlagBits2::eTopOfPipe;
            toDst.srcAccessMask    = vk::AccessFlagBits2::eNone;
            toDst.dstStageMask     = vk::PipelineStageFlagBits2::eTransfer;
            toDst.dstAccessMask    = vk::AccessFlagBits2::eTransferWrite;

            std::array<vk::ImageMemoryBarrier2, 2> barriers{toSrc, toDst};
            vk::DependencyInfo                     depInfo{};
            depInfo.setImageMemoryBarriers(barriers);
            cmd_buffer.pipelineBarrier2(depInfo);
        }

        const auto     layers = vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, 0, 1);
        vk::ImageBlit2 region{};
        region.srcSubresource = layers;
        region.dstSubresource = layers;
        region.srcOffsets[0]  = vk::Offset3D{0, 0, 0};
        region.srcOffsets[1]  = vk::Offset3D{static_cast<int32_t>(_width), static_cast<int32_t>(_height), 1};
        region.dstOffsets[0]  = vk::Offset3D{0, 0, 0};
        region.dstOffsets[1]  = vk::Offset3D{static_cast<int32_t>(_width), static_cast<int32_t>(_height), 1};

        vk::BlitImageInfo2 blit{};
        blit.srcImage       = src_img;
        blit.srcImageLayout = vk::ImageLayout::eTransferSrcOptimal;
        blit.dstImage       = dst_img;
        blit.dstImageLayout = vk::ImageLayout::eTransferDstOptimal;
        // eNearest, not eLinear: this is 1:1, so there is nothing to interpolate and
        // eLinear would only invite half-texel error. Conversion to a normalized
        // destination format clamps out-of-range values, which is where the display range
        // should be imposed.
        blit.filter = vk::Filter::eNearest;
        blit.setRegions(region);
        cmd_buffer.blitImage2(blit);

        // Hand the resolve target on in the layout copy_async() and texture_wrapper
        // assume, so callers chain the two without knowing anything about layouts --
        // the same contract reduce_texture() honours.
        {
            vk::ImageMemoryBarrier2 toColor{};
            toColor.oldLayout        = vk::ImageLayout::eTransferDstOptimal;
            toColor.newLayout        = vk::ImageLayout::eColorAttachmentOptimal;
            toColor.srcQueueFamilyIndex = vk::QueueFamilyIgnored;
            toColor.dstQueueFamilyIndex = vk::QueueFamilyIgnored;
            toColor.image            = dst_img;
            toColor.subresourceRange = vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);
            toColor.srcStageMask     = vk::PipelineStageFlagBits2::eTransfer;
            toColor.srcAccessMask    = vk::AccessFlagBits2::eTransferWrite;
            toColor.dstStageMask     = vk::PipelineStageFlagBits2::eAllCommands;
            toColor.dstAccessMask    = vk::AccessFlagBits2::eMemoryRead;

            vk::DependencyInfo depInfo{};
            depInfo.setImageMemoryBarriers(toColor);
            cmd_buffer.pipelineBarrier2(depInfo);
        }
    }

    cmd_buffer.end();

    _ctx->submit();
}

}}} // namespace caspar::accelerator::vulkan
