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

#include "image_kernel.h"

#include "../util/device.h"
#include "../util/pipeline.h"
#include "../util/renderpass.h"
#include "../util/texture.h"

#include <common/assert.h>

#ifdef WIN32
#include <vulkan/vulkan_win32.h>
#endif

#include <core/frame/frame_transform.h>
#include <core/frame/pixel_format.h>

#include <boost/algorithm/cxx11/all_of.hpp>
#include <boost/range/adaptor/transformed.hpp>

#include <array>
#include <cmath>

namespace caspar::accelerator::vulkan {

float get_precision_factor(common::bit_depth depth)
{
    switch (depth) {
        case common::bit_depth::bit8:
            return 1.0f;
        case common::bit_depth::bit10:
            return 64.0f;
        case common::bit_depth::bit12:
            return 16.0f;
        case common::bit_depth::bit16:
        default:
            return 1.0f;
    }
}

bool is_above_screen(double y) { return y < 0.0; }

bool is_below_screen(double y) { return y > 1.0; }

bool is_left_of_screen(double x) { return x < 0.0; }

bool is_right_of_screen(double x) { return x > 1.0; }

bool is_outside_screen(const std::vector<core::frame_geometry::coord>& coords)
{
    auto x_coords =
        coords | boost::adaptors::transformed([](const core::frame_geometry::coord& c) { return c.vertex_x; });
    auto y_coords =
        coords | boost::adaptors::transformed([](const core::frame_geometry::coord& c) { return c.vertex_y; });

    return boost::algorithm::all_of(x_coords, &is_left_of_screen) ||
           boost::algorithm::all_of(x_coords, &is_right_of_screen) ||
           boost::algorithm::all_of(y_coords, &is_above_screen) || boost::algorithm::all_of(y_coords, &is_below_screen);
}

static const double epsilon = 0.001;

static const uint32_t frame_buffer_size = 3;

struct image_kernel::impl
{
    spl::shared_ptr<device> vulkan_;
    common::bit_depth       depth_;
    int32_t                 frame_counter_ = 0;

    struct frame_data : public frame_context
    {
        image_kernel::impl* parent = nullptr;

        vk::Buffer       buffer = nullptr;
        void*            data   = nullptr;
        vk::DeviceMemory memory = nullptr;
        size_t           size   = 0;

        vk::CommandBuffer cmd_buffer = nullptr;
        vk::Fence         fence      = nullptr;
        vk::Semaphore     render_sem = nullptr;   // exportable timeline semaphore for GPU-side wait
        void*             render_sem_handle = nullptr; // cached Win32 HANDLE
        uint64_t          render_sem_value  = 0;       // current timeline value

        explicit frame_data(image_kernel::impl* parent)
            : parent(parent)
        {
        }

        virtual vk::Buffer upload_vertex_data(const std::vector<float>& src)
        {
            return parent->upload_vertex_buffer(*this, (void*)src.data(), src.size() * sizeof(float));
        }
        virtual draw_data create_draw_data(const draw_params& params) { return parent->draw(params); }
        virtual std::shared_ptr<class pipeline> get_pipeline() { return parent->vulkan_->get_pipeline(parent->depth_); }
        virtual vk::CommandBuffer               get_command_buffer() { return cmd_buffer; }
        virtual void                            submit()
        {
            auto vk_device = parent->vulkan_->getVkDevice();
            if (!fence)
                fence = vk_device.createFence({});

            // Create an exportable timeline semaphore for GPU-side wait by CUDA consumers.
            if (!render_sem) {
#ifdef WIN32
                vk::ExportSemaphoreCreateInfo exportInfo{};
                exportInfo.handleTypes = vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueWin32;
                vk::SemaphoreTypeCreateInfo typeInfo{};
                typeInfo.semaphoreType = vk::SemaphoreType::eTimeline;
                typeInfo.initialValue  = 0;
                typeInfo.pNext         = &exportInfo;
                vk::SemaphoreCreateInfo semInfo{};
                semInfo.pNext = &typeInfo;
                render_sem = vk_device.createSemaphore(semInfo);
#endif
            }

            render_sem_value++;

            vk::TimelineSemaphoreSubmitInfo timelineInfo{};
            uint64_t signalValue = render_sem_value;
            timelineInfo.signalSemaphoreValueCount = 1;
            timelineInfo.pSignalSemaphoreValues    = &signalValue;

            vk::SubmitInfo submitInfo{};
            submitInfo.setCommandBuffers(cmd_buffer);
            if (render_sem) {
                submitInfo.setSignalSemaphores(render_sem);
                submitInfo.pNext = &timelineInfo;
            }
            parent->vulkan_->submit(submitInfo, fence);
        }

        void*                           render_complete_semaphore_handle() override
        {
#ifdef WIN32
            if (render_sem && !render_sem_handle) {
                auto vk_device = parent->vulkan_->getVkDevice();
                auto pfn = reinterpret_cast<PFN_vkGetSemaphoreWin32HandleKHR>(
                    vk_device.getProcAddr("vkGetSemaphoreWin32HandleKHR"));
                if (!pfn) return nullptr;

                VkSemaphoreGetWin32HandleInfoKHR handleInfo{};
                handleInfo.sType     = VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR;
                handleInfo.semaphore = static_cast<VkSemaphore>(render_sem);
                handleInfo.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;

                HANDLE handle = nullptr;
                VkResult result = pfn(static_cast<VkDevice>(vk_device), &handleInfo, &handle);
                if (result == VK_SUCCESS && handle)
                    render_sem_handle = handle;
            }
            return render_sem_handle;
#else
            return nullptr;
#endif
        }

        uint64_t render_complete_semaphore_value() override { return render_sem_value; }
        virtual void                            wait_for_completion()
        {
            if (fence) {
                auto result = parent->vulkan_->getVkDevice().waitForFences(fence, true, 1000000000);
                if (result == vk::Result::eTimeout) {
                    CASPAR_LOG(warning) << L"[Vulkan image_kernel] Timeout waiting for render completion";
                }
            }
        }
        virtual std::shared_ptr<class texture>
        create_attachment(uint32_t width, uint32_t height, uint32_t components_count)
        {
            // Reuse an attachment texture from a previous frame on this slot
            // if the consumer has released its reference (use_count == 1 means
            // only our pool holds it).  This keeps the underlying VkDeviceMemory
            // and its exported Win32 HANDLE stable, which in turn keeps any
            // CUDA import on the consumer side valid — avoiding the extremely
            // expensive cudaImportExternalMemory call (~10-150 ms) every frame.
            for (auto& tex : attachment_pool_) {
                if (tex && tex.use_count() == 1 &&
                    static_cast<uint32_t>(tex->width()) == width &&
                    static_cast<uint32_t>(tex->height()) == height) {
                    return tex;
                }
            }
            auto tex = parent->vulkan_->create_attachment(width, height, parent->depth_, components_count);
            // Cap pool to prevent unbounded VRAM growth when consumers hold refs.
            static constexpr size_t MAX_ATTACHMENT_POOL = 4;
            if (attachment_pool_.size() < MAX_ATTACHMENT_POOL)
                attachment_pool_.push_back(tex);
            return tex;
        }

        // Pool of attachment textures for this slot.
        std::vector<std::shared_ptr<class texture>> attachment_pool_;
    };

    frame_data frames_[frame_buffer_size];
    uint32_t   current_frame_index_ = 0;

    explicit impl(const spl::shared_ptr<device>& vulkan, common::bit_depth depth)
        : vulkan_(vulkan)
        , depth_(depth)
        , frames_{frame_data{this}, frame_data{this}, frame_data{this}}
    {
        auto cmd_buffers = vulkan_->allocateCommandBuffers(frame_buffer_size);
        for (uint32_t i = 0; i < frame_buffer_size; ++i) {
            frames_[i].cmd_buffer = cmd_buffers[i];
        }
    }

    ~impl()
    {
        auto vk_device = vulkan_->getVkDevice();

        for (auto& frame : frames_) {
            if (frame.buffer) {
                vk_device.unmapMemory(frame.memory);
                vk_device.destroyBuffer(frame.buffer);
                vk_device.freeMemory(frame.memory);
                if (frame.fence) {
                    vk_device.destroyFence(frame.fence);
                }
                if (frame.render_sem_handle) {
                    CloseHandle(frame.render_sem_handle);
                }
                if (frame.render_sem) {
                    vk_device.destroySemaphore(frame.render_sem);
                }
            }
        }
    }

    spl::shared_ptr<renderpass> create_renderpass(uint32_t width, uint32_t height)
    {
        auto  device = vulkan_->getVkDevice();
        auto& ctx    = frames_[(++current_frame_index_) % frame_buffer_size];
        if (ctx.fence) {
            auto result = device.waitForFences(ctx.fence, true, 1000000000); // wait up to one second
            if (result == vk::Result::eTimeout) {
                CASPAR_LOG(warning) << L"[Vulkan image_kernel] Timeout waiting for fence";
            }
            device.resetFences(ctx.fence);
        }

        ctx.cmd_buffer.reset({});
        return spl::make_shared<renderpass>(&ctx, width, height);
    }

    uint32_t findDedicatedMemoryType(uint32_t typeMask, vk::MemoryPropertyFlags properties)
    {
        auto memProperties = vulkan_->getMemoryProperties();
        for (uint32_t i = 0; i < memProperties.memoryTypeCount; ++i) {
            if ((typeMask & (1 << i)) && ((memProperties.memoryTypes[i].propertyFlags & properties) == properties)) {
                return i;
            }
        }
        throw std::runtime_error("[Vulkan image_kernel] Failed to find suitable memory type");
    }

    vk::Buffer upload_vertex_buffer(frame_data& vb, void* data, size_t size)
    {
        if (vb.size < size) {
            auto vk_device = vulkan_->getVkDevice();

            if (vb.buffer) {
                vk_device.unmapMemory(vb.memory);
                vk_device.destroyBuffer(vb.buffer);
                vk_device.freeMemory(vb.memory);
            }

            // staging buffer
            vk::BufferCreateInfo stagingInfo{};
            stagingInfo.size        = size;
            stagingInfo.usage       = vk::BufferUsageFlagBits::eVertexBuffer;
            stagingInfo.sharingMode = vk::SharingMode::eExclusive;

            vb.buffer = vk_device.createBuffer(stagingInfo);

            auto stagingMemReq = vk_device.getBufferMemoryRequirements(vb.buffer);

            vk::MemoryAllocateInfo stagingAlloc{};
            stagingAlloc.allocationSize  = stagingMemReq.size;
            stagingAlloc.memoryTypeIndex = findDedicatedMemoryType(stagingMemReq.memoryTypeBits,
                                                                   vk::MemoryPropertyFlagBits::eHostVisible |
                                                                       vk::MemoryPropertyFlagBits::eHostCoherent);

            vb.memory = vk_device.allocateMemory(stagingAlloc);
            vk_device.bindBufferMemory(vb.buffer, vb.memory, 0);

            vb.data = vk_device.mapMemory(vb.memory, 0, size);
            vb.size = size;
        }
        memcpy(vb.data, data, size);

        return vb.buffer;
    }

    std::pair<std::vector<core::frame_geometry::coord>, uniform_block> draw(const draw_params& params)
    {
        CASPAR_ASSERT(params.pix_desc.planes.size() == params.textures.size());

        if (params.textures.empty() || !params.background) {
            return {};
        }

        if (params.transforms.image_transform.opacity < epsilon) {
            return {};
        }

        if (params.geometry.data().empty()) {
            return {};
        }

        auto coords     = params.geometry.data();
        auto transforms = params.transforms;

        auto const first_plane = params.pix_desc.planes.at(0);
        if (params.geometry.mode() != core::frame_geometry::scale_mode::stretch && first_plane.width > 0 &&
            first_plane.height > 0) {
            auto width_scale  = static_cast<double>(params.target_width) / static_cast<double>(first_plane.width);
            auto height_scale = static_cast<double>(params.target_height) / static_cast<double>(first_plane.height);

            core::image_transform transform;
            double                target_scale;
            switch (params.geometry.mode()) {
                case core::frame_geometry::scale_mode::fit:
                    target_scale = std::min(width_scale, height_scale);

                    transform.fill_scale[0] *= target_scale / width_scale;
                    transform.fill_scale[1] *= target_scale / height_scale;
                    break;

                case core::frame_geometry::scale_mode::fill:
                    target_scale = std::max(width_scale, height_scale);
                    transform.fill_scale[0] *= target_scale / width_scale;
                    transform.fill_scale[1] *= target_scale / height_scale;
                    break;

                case core::frame_geometry::scale_mode::original:
                    transform.fill_scale[0] /= width_scale;
                    transform.fill_scale[1] /= height_scale;
                    break;

                case core::frame_geometry::scale_mode::hfill:
                    transform.fill_scale[1] *= width_scale / height_scale;
                    break;

                case core::frame_geometry::scale_mode::vfill:
                    transform.fill_scale[0] *= height_scale / width_scale;
                    break;

                default:;
            }

            transforms = transforms.combine_transform(transform, params.aspect_ratio);
        }

        coords = transforms.transform_coords(coords);

        // Skip drawing if all the coordinates will be outside the screen.
        if (coords.size() < 3 || is_outside_screen(coords)) {
            return {};
        }

        uniform_block uniforms;

        for (int n = 0; n < params.textures.size(); ++n) {
            uniforms.precision_factor[n] = get_precision_factor(params.textures[n]->depth());
        }

        const auto is_hd           = params.pix_desc.planes.at(0).height > 700;
        const auto color_space     = is_hd ? params.pix_desc.color_space : core::color_space::bt601;
        uniforms.color_space_index = static_cast<uint32_t>(color_space);

        if (params.pix_desc.is_straight_alpha) {
            uniforms.flags |= static_cast<uint32_t>(shader_flags::is_straight_alpha);
        }

        if (static_cast<bool>(params.local_key)) {
            uniforms.flags |= static_cast<uint32_t>(shader_flags::has_local_key);
        }
        if (static_cast<bool>(params.layer_key)) {
            uniforms.flags |= static_cast<uint32_t>(shader_flags::has_layer_key);
        }
        uniforms.pixel_format = static_cast<uint32_t>(params.pix_desc.format);

        uniforms.opacity =
            transforms.image_transform.is_key ? 1.0f : static_cast<float>(transforms.image_transform.opacity);

        if (transforms.image_transform.chroma.enable) {
            uniforms.flags |= static_cast<uint32_t>(shader_flags::chroma);

            if (transforms.image_transform.chroma.show_mask)
                uniforms.flags |= static_cast<uint32_t>(shader_flags::chroma_show_mask);

            uniforms.chroma_target_hue     = static_cast<float>(transforms.image_transform.chroma.target_hue) / 360.0f;
            uniforms.chroma_hue_width      = static_cast<float>(transforms.image_transform.chroma.hue_width);
            uniforms.chroma_min_saturation = static_cast<float>(transforms.image_transform.chroma.min_saturation);
            uniforms.chroma_min_brightness = static_cast<float>(transforms.image_transform.chroma.min_brightness);
            uniforms.chroma_softness       = 1.0f + static_cast<float>(transforms.image_transform.chroma.softness);
            uniforms.chroma_spill_suppress =
                static_cast<float>(transforms.image_transform.chroma.spill_suppress) / 360.0f;
            uniforms.chroma_spill_suppress_saturation =
                static_cast<float>(transforms.image_transform.chroma.spill_suppress_saturation);
        }

        // Setup blend_func
        auto blend_mode = params.blend_mode;
        if (transforms.image_transform.is_key) {
            blend_mode = core::blend_mode::normal;
        }

        uniforms.blend_mode = static_cast<uint32_t>(blend_mode);
        uniforms.keyer      = static_cast<uint32_t>(params.keyer);

        if (transforms.image_transform.invert) {
            uniforms.flags |= static_cast<uint32_t>(shader_flags::invert);
        }

        if (transforms.image_transform.levels.min_input > epsilon ||
            transforms.image_transform.levels.max_input < 1.0 - epsilon ||
            transforms.image_transform.levels.min_output > epsilon ||
            transforms.image_transform.levels.max_output < 1.0 - epsilon ||
            std::abs(transforms.image_transform.levels.gamma - 1.0) > epsilon) {
            uniforms.flags |= static_cast<uint32_t>(shader_flags::levels);
            uniforms.min_input  = static_cast<float>(transforms.image_transform.levels.min_input);
            uniforms.max_input  = static_cast<float>(transforms.image_transform.levels.max_input);
            uniforms.min_output = static_cast<float>(transforms.image_transform.levels.min_output);
            uniforms.max_output = static_cast<float>(transforms.image_transform.levels.max_output);
            uniforms.gamma      = static_cast<float>(transforms.image_transform.levels.gamma);
        }

        if (std::abs(transforms.image_transform.brightness - 1.0) > epsilon ||
            std::abs(transforms.image_transform.saturation - 1.0) > epsilon ||
            std::abs(transforms.image_transform.contrast - 1.0) > epsilon) {
            uniforms.flags |= static_cast<uint32_t>(shader_flags::csb);

            uniforms.brt = static_cast<float>(transforms.image_transform.brightness);
            uniforms.sat = static_cast<float>(transforms.image_transform.saturation);
            uniforms.con = static_cast<float>(transforms.image_transform.contrast);
        }

        // ── Target size (needed by blur, sharpening, grain) ────────────
        uniforms.target_size[0] = static_cast<float>(params.target_width);
        uniforms.target_size[1] = static_cast<float>(params.target_height);
        uniforms.aspect_ratio   = static_cast<float>(params.aspect_ratio);

        // ── Flip H/V ──────────────────────────────────────────────────
        if (transforms.image_transform.flip_h)
            uniforms.flags |= static_cast<uint32_t>(shader_flags::flip_h);
        if (transforms.image_transform.flip_v)
            uniforms.flags |= static_cast<uint32_t>(shader_flags::flip_v);

        // ── 360° Projection ───────────────────────────────────────────
        if (transforms.image_transform.projection.enable) {
            uniforms.flags |= static_cast<uint32_t>(shader_flags::is_360);
            uniforms.view_yaw       = static_cast<float>(transforms.image_transform.projection.yaw);
            uniforms.view_pitch     = static_cast<float>(transforms.image_transform.projection.pitch);
            uniforms.view_roll      = static_cast<float>(transforms.image_transform.projection.roll);
            uniforms.view_fov       = static_cast<float>(transforms.image_transform.projection.fov);
            uniforms.view_offset_x  = static_cast<float>(transforms.image_transform.projection.offset_x);
            uniforms.view_offset_y  = static_cast<float>(transforms.image_transform.projection.offset_y);
            uniforms.frustum_h      = static_cast<float>(transforms.image_transform.projection.frustum_h);
            uniforms.frustum_v      = static_cast<float>(transforms.image_transform.projection.frustum_v);
            uniforms.lens_k1        = static_cast<float>(transforms.image_transform.projection.lens_k1);
            uniforms.lens_k2        = static_cast<float>(transforms.image_transform.projection.lens_k2);
            uniforms.lens_k3        = static_cast<float>(transforms.image_transform.projection.lens_k3);
        }

        // ── Curved Screen ─────────────────────────────────────────────
        if (transforms.image_transform.projection.curve_enable)
            uniforms.flags |= static_cast<uint32_t>(shader_flags::is_curved);
        uniforms.screen_curve_type = static_cast<int32_t>(transforms.image_transform.projection.curve_type);
        uniforms.screen_arc        = static_cast<float>(transforms.image_transform.projection.screen_arc);

        // ── Edge Blending ─────────────────────────────────────────────
        {
            float ebl = static_cast<float>(transforms.image_transform.projection.edge_blend_left);
            float ebr = static_cast<float>(transforms.image_transform.projection.edge_blend_right);
            float ebt = static_cast<float>(transforms.image_transform.projection.edge_blend_top);
            float ebb = static_cast<float>(transforms.image_transform.projection.edge_blend_bottom);
            if (ebl > epsilon || ebr > epsilon || ebt > epsilon || ebb > epsilon) {
                uniforms.flags |= static_cast<uint32_t>(shader_flags::edge_blend);
                uniforms.edge_blend_left   = ebl;
                uniforms.edge_blend_right  = ebr;
                uniforms.edge_blend_top    = ebt;
                uniforms.edge_blend_bottom = ebb;
                uniforms.edge_blend_gamma  = static_cast<float>(transforms.image_transform.projection.edge_blend_gamma);
            }
        }

        // ── Blur ──────────────────────────────────────────────────────
        if (transforms.image_transform.blur.enable) {
            uniforms.flags |= static_cast<uint32_t>(shader_flags::blur_enable);
            uniforms.blur_radius    = static_cast<float>(transforms.image_transform.blur.radius);
            uniforms.blur_type      = static_cast<int32_t>(transforms.image_transform.blur.type);
            uniforms.blur_angle     = static_cast<float>(transforms.image_transform.blur.angle);
            uniforms.blur_center[0] = static_cast<float>(transforms.image_transform.blur.center[0]);
            uniforms.blur_center[1] = static_cast<float>(transforms.image_transform.blur.center[1]);
            uniforms.blur_tilt[0]   = static_cast<float>(transforms.image_transform.blur.tilt_y);
            uniforms.blur_tilt[1]   = static_cast<float>(transforms.image_transform.blur.tilt_h);
        }

        // ── Color Grading ─────────────────────────────────────────────
        {
            static const float k_to_working[7][9] = {
                {0.6131516f,0.3395148f,0.0472947f, 0.0701011f,0.9162792f,0.0136197f, 0.0206177f,0.1095763f,0.8698060f},
                {0.7951281f,0.1643585f,0.0405134f, 0.0234399f,0.9415642f,0.0349959f, 0.0036186f,0.0613513f,0.9350301f},
                {0.8224549f,0.1774521f,-0.0000070f, 0.0332021f,0.9618927f,0.0049052f, 0.0170512f,0.0723025f,0.9106463f},
                {1.4514393f,-0.2365107f,-0.2149286f, -0.0765538f,1.1762297f,-0.0996759f, 0.0083161f,-0.0060324f,0.9977163f},
                {1.0f,0.0f,0.0f, 0.0f,1.0f,0.0f, 0.0f,0.0f,1.0f},
                {0.6954522f,0.1446577f,0.1598901f, 0.0439823f,0.8591788f,0.0968389f, -0.0055023f,0.0040678f,1.0014345f},
                {0.7112957f,0.1903613f,0.0983436f, 0.0406952f,0.8550396f,0.1042651f, -0.0025079f,0.0085993f,0.9939086f}
            };
            static const float k_to_output[7][9] = {
                {1.7050585f,-0.6217876f,-0.0832709f, -0.1302597f,1.1407927f,-0.0105330f, -0.0240003f,-0.1289711f,1.1529714f},
                {1.2746843f,-0.2692490f,-0.0054353f, -0.0293524f,1.0763680f,-0.0470156f, -0.0160993f,-0.0606079f,1.0767072f},
                {1.2239840f,-0.2239840f,0.0000000f, -0.0421197f,1.0421197f,0.0000000f, -0.0196576f,-0.0787093f,1.0983669f},
                {0.6954522f,0.1406787f,0.1638691f, 0.0447946f,0.8596711f,0.0955343f, -0.0055259f,0.0040252f,1.0015007f},
                {1.0f,0.0f,0.0f, 0.0f,1.0f,0.0f, 0.0f,0.0f,1.0f},
                {1.4516608f,-0.2434265f,-0.2082343f, -0.0752455f,1.1770530f,-0.1018075f, 0.0082817f,-0.0061186f,0.9978370f},
                {1.4235761f,-0.3158537f,-0.1077233f, -0.0682645f,1.1859178f,-0.1176531f, 0.0041827f,-0.0110575f,1.0068749f}
            };

            // Helper: expand mat3 (9 floats col-major) to 3×vec4 (12 floats, std140)
            auto set_mat3 = [](float dst[12], const float src[9]) {
                dst[0]=src[0]; dst[1]=src[1]; dst[2]=src[2]; dst[3]=0;
                dst[4]=src[3]; dst[5]=src[4]; dst[6]=src[5]; dst[7]=0;
                dst[8]=src[6]; dst[9]=src[7]; dst[10]=src[8]; dst[11]=0;
            };

            const auto& cg = transforms.image_transform.color_grade;
            if (cg.enable) {
                uniforms.flags |= static_cast<uint32_t>(shader_flags::color_grading);
                uniforms.input_transfer  = cg.input_transfer;
                uniforms.output_transfer = cg.output_transfer;
                uniforms.tone_mapping_op = cg.tone_mapping;
                uniforms.exposure        = static_cast<float>(cg.exposure);
                int ig = std::min(std::max(cg.input_gamut,  0), 6);
                int og = std::min(std::max(cg.output_gamut, 0), 6);
                set_mat3(uniforms.input_to_working,  k_to_working[ig]);
                set_mat3(uniforms.working_to_output, k_to_output[og]);
            } else if (params.auto_color_convert &&
                       (params.pix_desc.color_space != params.target_color_space ||
                        params.pix_desc.color_transfer != params.target_color_transfer)) {
                // Auto color conversion: source differs from channel output.
                auto gamut_index = [](core::color_space cs) -> int {
                    switch (cs) {
                        case core::color_space::bt2020: return 1;
                        default:                       return 0; // bt601/bt709 → shader index 0 (bt709)
                    }
                };
                auto transfer_index = [](core::color_transfer ct) -> int {
                    switch (ct) {
                        case core::color_transfer::pq:  return 3;
                        case core::color_transfer::hlg: return 4;
                        default:                        return 2; // sdr → rec709
                    }
                };
                int ig = gamut_index(params.pix_desc.color_space);
                int og = gamut_index(params.target_color_space);
                // Skip if the mapped indices are identical (e.g. bt601 source on bt709 channel)
                if (ig != og || params.pix_desc.color_transfer != params.target_color_transfer) {
                    int it = transfer_index(params.pix_desc.color_transfer);
                    int ot = transfer_index(params.target_color_transfer);
                    int tm = (it >= 3 && ot <= 2) ? 4 : 0; // ACES_RRT_709 for HDR→SDR
                    uniforms.flags |= static_cast<uint32_t>(shader_flags::color_grading);
                    uniforms.input_transfer  = it;
                    uniforms.output_transfer = ot;
                    uniforms.tone_mapping_op = tm;
                    uniforms.exposure        = 1.0f;
                    set_mat3(uniforms.input_to_working,  k_to_working[ig]);
                    set_mat3(uniforms.working_to_output, k_to_output[og]);
                }
            }
        }

        // ── White Balance ─────────────────────────────────────────────
        if (std::abs(transforms.image_transform.temperature) > epsilon ||
            std::abs(transforms.image_transform.tint) > epsilon) {
            uniforms.flags |= static_cast<uint32_t>(shader_flags::white_balance);
            uniforms.wb_temperature = static_cast<float>(transforms.image_transform.temperature);
            uniforms.wb_tint        = static_cast<float>(transforms.image_transform.tint);
        }

        // ── Lift / Midtone / Gain ─────────────────────────────────────
        {
            const auto& lift    = transforms.image_transform.lift;
            const auto& midtone = transforms.image_transform.midtone;
            const auto& gain    = transforms.image_transform.gain;
            bool lmg_active =
                std::abs(lift[0]) > epsilon || std::abs(lift[1]) > epsilon || std::abs(lift[2]) > epsilon ||
                std::abs(midtone[0]-1.0) > epsilon || std::abs(midtone[1]-1.0) > epsilon || std::abs(midtone[2]-1.0) > epsilon ||
                std::abs(gain[0]-1.0) > epsilon || std::abs(gain[1]-1.0) > epsilon || std::abs(gain[2]-1.0) > epsilon;
            if (lmg_active) {
                uniforms.flags |= static_cast<uint32_t>(shader_flags::lmg_enable);
                // R↔B swap for BGRA convention
                uniforms.lmg_lift[0] = static_cast<float>(lift[2]);
                uniforms.lmg_lift[1] = static_cast<float>(lift[1]);
                uniforms.lmg_lift[2] = static_cast<float>(lift[0]);
                uniforms.lmg_midtone[0] = static_cast<float>(midtone[2]);
                uniforms.lmg_midtone[1] = static_cast<float>(midtone[1]);
                uniforms.lmg_midtone[2] = static_cast<float>(midtone[0]);
                uniforms.lmg_gain[0] = static_cast<float>(gain[2]);
                uniforms.lmg_gain[1] = static_cast<float>(gain[1]);
                uniforms.lmg_gain[2] = static_cast<float>(gain[0]);
            }
        }

        // ── Hue Shift ─────────────────────────────────────────────────
        if (std::abs(transforms.image_transform.hue_shift) > epsilon) {
            uniforms.flags |= static_cast<uint32_t>(shader_flags::hue_shift_enable);
            uniforms.hue_shift_degrees = static_cast<float>(transforms.image_transform.hue_shift);
        }

        // ── Tonal Balance ─────────────────────────────────────────────
        if (std::abs(transforms.image_transform.shadows) > epsilon ||
            std::abs(transforms.image_transform.highlights) > epsilon) {
            uniforms.flags |= static_cast<uint32_t>(shader_flags::tonebalance_enable);
            uniforms.tb_shadows    = static_cast<float>(transforms.image_transform.shadows);
            uniforms.tb_highlights = static_cast<float>(transforms.image_transform.highlights);
        }

        // ── Linear Saturation ─────────────────────────────────────────
        if (std::abs(transforms.image_transform.linear_saturation - 1.0) > epsilon) {
            uniforms.flags |= static_cast<uint32_t>(shader_flags::linear_sat_enable);
            uniforms.linear_sat_value = static_cast<float>(transforms.image_transform.linear_saturation);
        }

        // ── ASC CDL ───────────────────────────────────────────────────
        {
            const auto& s  = transforms.image_transform.cdl_slope;
            const auto& o  = transforms.image_transform.cdl_offset;
            const auto& p  = transforms.image_transform.cdl_power;
            double      cs = transforms.image_transform.cdl_saturation;
            bool cdl_active =
                std::abs(s[0]-1.0) > epsilon || std::abs(s[1]-1.0) > epsilon || std::abs(s[2]-1.0) > epsilon ||
                std::abs(o[0]) > epsilon     || std::abs(o[1]) > epsilon     || std::abs(o[2]) > epsilon     ||
                std::abs(p[0]-1.0) > epsilon || std::abs(p[1]-1.0) > epsilon || std::abs(p[2]-1.0) > epsilon ||
                std::abs(cs-1.0) > epsilon;
            if (cdl_active) {
                uniforms.flags |= static_cast<uint32_t>(shader_flags::cdl_enable);
                // R↔B swap for BGRA convention
                uniforms.cdl_slope[0] = static_cast<float>(s[2]);
                uniforms.cdl_slope[1] = static_cast<float>(s[1]);
                uniforms.cdl_slope[2] = static_cast<float>(s[0]);
                uniforms.cdl_saturation = static_cast<float>(cs);
                uniforms.cdl_offset[0] = static_cast<float>(o[2]);
                uniforms.cdl_offset[1] = static_cast<float>(o[1]);
                uniforms.cdl_offset[2] = static_cast<float>(o[0]);
                uniforms.cdl_power[0] = static_cast<float>(p[2]);
                uniforms.cdl_power[1] = static_cast<float>(p[1]);
                uniforms.cdl_power[2] = static_cast<float>(p[0]);
            }
        }

        // ── Split Toning ──────────────────────────────────────────────
        {
            const auto& sc = transforms.image_transform.split_shadow_color;
            const auto& hc = transforms.image_transform.split_highlight_color;
            bool split_active =
                std::abs(sc[0]) > epsilon || std::abs(sc[1]) > epsilon || std::abs(sc[2]) > epsilon ||
                std::abs(hc[0]) > epsilon || std::abs(hc[1]) > epsilon || std::abs(hc[2]) > epsilon;
            if (split_active) {
                uniforms.flags |= static_cast<uint32_t>(shader_flags::split_tone_enable);
                // R↔B swap for BGRA convention
                uniforms.split_shadow_color[0] = static_cast<float>(sc[2]);
                uniforms.split_shadow_color[1] = static_cast<float>(sc[1]);
                uniforms.split_shadow_color[2] = static_cast<float>(sc[0]);
                uniforms.split_balance = static_cast<float>(transforms.image_transform.split_balance);
                uniforms.split_highlight_color[0] = static_cast<float>(hc[2]);
                uniforms.split_highlight_color[1] = static_cast<float>(hc[1]);
                uniforms.split_highlight_color[2] = static_cast<float>(hc[0]);
            }
        }

        // ── Gamut Compression ─────────────────────────────────────────
        if (transforms.image_transform.gamut_compress) {
            uniforms.flags |= static_cast<uint32_t>(shader_flags::gamut_compress);
            // BGRA order: .r=Blue(yellow), .g=Green(magenta), .b=Red(cyan)
            uniforms.gc_limit[0] = static_cast<float>(transforms.image_transform.gc_yellow);
            uniforms.gc_limit[1] = static_cast<float>(transforms.image_transform.gc_magenta);
            uniforms.gc_limit[2] = static_cast<float>(transforms.image_transform.gc_cyan);
        }

        // ── 3D LUT ───────────────────────────────────────────────────
        {
            const auto& lut = transforms.image_transform.lut3d;
            if (lut && lut->size > 0 && !lut->data.empty()) {
                uniforms.flags |= static_cast<uint32_t>(shader_flags::lut3d_enable);
                uniforms.lut3d_strength = static_cast<float>(transforms.image_transform.lut3d_strength);
                // Note: actual 3D texture view set in renderpass/draw_params (Phase 3C)
            }
        }

        // ── Hue Curves ───────────────────────────────────────────────
        {
            const auto& hc = transforms.image_transform.hue_curves;
            if (hc && !hc->data.empty()) {
                uniforms.flags |= static_cast<uint32_t>(shader_flags::hue_curve_enable);
                // Note: actual texture view set in renderpass/draw_params (Phase 3C)
            }
        }

        // ── Sharpening ───────────────────────────────────────────────
        if (std::abs(transforms.image_transform.sharpen_amount) > epsilon) {
            uniforms.flags |= static_cast<uint32_t>(shader_flags::sharpen_enable);
            uniforms.sharpen_amount = static_cast<float>(transforms.image_transform.sharpen_amount);
            uniforms.sharpen_radius = static_cast<float>(transforms.image_transform.sharpen_radius);
        }

        // ── Film Grain ───────────────────────────────────────────────
        if (std::abs(transforms.image_transform.grain_intensity) > epsilon) {
            uniforms.flags |= static_cast<uint32_t>(shader_flags::grain_enable);
            uniforms.grain_intensity = static_cast<float>(transforms.image_transform.grain_intensity);
            uniforms.grain_size      = static_cast<float>(transforms.image_transform.grain_size);
            uniforms.grain_frame     = frame_counter_++;
        }

        // ── Secondary Qualifier ──────────────────────────────────────
        if (transforms.image_transform.qualifier_enable) {
            uniforms.flags |= static_cast<uint32_t>(shader_flags::qualifier_enable);
            uniforms.qual_target_hue = static_cast<float>(transforms.image_transform.qual_target_hue);
            uniforms.qual_hue_width  = static_cast<float>(transforms.image_transform.qual_hue_width);
            uniforms.qual_min_sat    = static_cast<float>(transforms.image_transform.qual_min_sat);
            uniforms.qual_max_sat    = static_cast<float>(transforms.image_transform.qual_max_sat);
            uniforms.qual_min_lum    = static_cast<float>(transforms.image_transform.qual_min_lum);
            uniforms.qual_max_lum    = static_cast<float>(transforms.image_transform.qual_max_lum);
            uniforms.qual_softness   = static_cast<float>(transforms.image_transform.qual_softness);
            uniforms.qual_exposure   = static_cast<float>(transforms.image_transform.qual_exposure);
            uniforms.qual_sat_offset = static_cast<float>(transforms.image_transform.qual_sat_offset);
            uniforms.qual_hue_offset = static_cast<float>(transforms.image_transform.qual_hue_offset);
        }

        // ── Per-Channel RGB Levels ───────────────────────────────────
        {
            const auto& rl = transforms.image_transform.per_channel_levels;
            if (rl.enable) {
                uniforms.flags |= static_cast<uint32_t>(shader_flags::rgb_levels_enable);
                // R↔B swap for BGRA convention: [0]=Blue_displayed, [2]=Red_displayed
                uniforms.rgb_levels_min_input[0]  = static_cast<float>(rl.b.min_input);
                uniforms.rgb_levels_min_input[1]  = static_cast<float>(rl.g.min_input);
                uniforms.rgb_levels_min_input[2]  = static_cast<float>(rl.r.min_input);
                uniforms.rgb_levels_max_input[0]  = static_cast<float>(rl.b.max_input);
                uniforms.rgb_levels_max_input[1]  = static_cast<float>(rl.g.max_input);
                uniforms.rgb_levels_max_input[2]  = static_cast<float>(rl.r.max_input);
                uniforms.rgb_levels_gamma[0]      = static_cast<float>(rl.b.gamma);
                uniforms.rgb_levels_gamma[1]      = static_cast<float>(rl.g.gamma);
                uniforms.rgb_levels_gamma[2]      = static_cast<float>(rl.r.gamma);
                uniforms.rgb_levels_min_output[0] = static_cast<float>(rl.b.min_output);
                uniforms.rgb_levels_min_output[1] = static_cast<float>(rl.g.min_output);
                uniforms.rgb_levels_min_output[2] = static_cast<float>(rl.r.min_output);
                uniforms.rgb_levels_max_output[0] = static_cast<float>(rl.b.max_output);
                uniforms.rgb_levels_max_output[1] = static_cast<float>(rl.g.max_output);
                uniforms.rgb_levels_max_output[2] = static_cast<float>(rl.r.max_output);
            }
        }

        // ── Tone Curves ──────────────────────────────────────────────
        {
            const auto& cv = transforms.image_transform.curves;
            if (cv.enable) {
                uniforms.flags |= static_cast<uint32_t>(shader_flags::curves_enable);
                // Note: actual curve LUT texture set in renderpass/draw_params (Phase 3C)
            }
        }

        // ── Shape Overlay ────────────────────────────────────────────
        {
            const auto& sh = transforms.image_transform.shape;
            if (sh.enable) {
                uniforms.flags |= static_cast<uint32_t>(shader_flags::shape_enable);
                uniforms.shape_type      = static_cast<int32_t>(sh.type);
                uniforms.shape_fill_type = static_cast<int32_t>(sh.fill_type);
                uniforms.shape_center[0] = static_cast<float>(sh.center[0]);
                uniforms.shape_center[1] = static_cast<float>(sh.center[1]);
                uniforms.shape_size[0]   = static_cast<float>(sh.size[0]);
                uniforms.shape_size[1]   = static_cast<float>(sh.size[1]);
                uniforms.shape_corner_radius = static_cast<float>(sh.corner_radius);
                uniforms.shape_softness      = static_cast<float>(sh.edge_softness);
                uniforms.shape_color1[0] = static_cast<float>(sh.color1[0]);
                uniforms.shape_color1[1] = static_cast<float>(sh.color1[1]);
                uniforms.shape_color1[2] = static_cast<float>(sh.color1[2]);
                uniforms.shape_color1[3] = static_cast<float>(sh.color1[3]);
                uniforms.shape_color2[0] = static_cast<float>(sh.color2[0]);
                uniforms.shape_color2[1] = static_cast<float>(sh.color2[1]);
                uniforms.shape_color2[2] = static_cast<float>(sh.color2[2]);
                uniforms.shape_color2[3] = static_cast<float>(sh.color2[3]);
                uniforms.shape_gradient_angle     = static_cast<float>(sh.gradient_angle);
                uniforms.shape_gradient_center[0] = static_cast<float>(sh.gradient_center[0]);
                uniforms.shape_gradient_center[1] = static_cast<float>(sh.gradient_center[1]);
                if (sh.stroke_enable) {
                    uniforms.flags |= static_cast<uint32_t>(shader_flags::shape_stroke);
                    uniforms.shape_stroke_width    = static_cast<float>(sh.stroke_width);
                    uniforms.shape_stroke_color[0] = static_cast<float>(sh.stroke_color[0]);
                    uniforms.shape_stroke_color[1] = static_cast<float>(sh.stroke_color[1]);
                    uniforms.shape_stroke_color[2] = static_cast<float>(sh.stroke_color[2]);
                    uniforms.shape_stroke_color[3] = static_cast<float>(sh.stroke_color[3]);
                }
            }
        }

        return {std::move(coords), uniforms};
    }
};

image_kernel::image_kernel(const spl::shared_ptr<device>& device, common::bit_depth depth)
    : impl_(new impl(device, depth))
{
}
image_kernel::~image_kernel() {}

spl::shared_ptr<renderpass> image_kernel::create_renderpass(uint32_t width, uint32_t height)
{
    return impl_->create_renderpass(width, height);
}

} // namespace caspar::accelerator::vulkan
