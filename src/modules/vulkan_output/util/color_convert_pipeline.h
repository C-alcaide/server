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

#pragma once

#include "../consumer/config.h"

#include <vulkan/vulkan.hpp>

#include <cstdint>

namespace caspar { namespace vulkan_output {

// Push constants matching the compute shader layout (80 bytes).
struct color_convert_push_constants
{
    float gamut_matrix[16]; // 4×4 row-major (3×3 embedded)
    int   eotf_mode;        // 0=srgb, 1=linear, 2=pq, 3=hlg, 4=gamma24, 5=gamma26
    float max_luminance;    // PQ/HLG max nits
    int   tone_map_op;      // 0=none, 1=reinhard, 2=aces_filmic, 3=aces_rrt, 7=hlg_ootf
    float padding;
};
static_assert(sizeof(color_convert_push_constants) == 80, "Push constants must be 80 bytes");

// Color conversion compute pipeline.
//
// Processing flow (within a single command buffer):
// 1. Blit source texture → intermediate (RGBA16F, TRANSFER_DST)
// 2. Barrier intermediate: TRANSFER_DST → GENERAL
// 3. Dispatch compute shader (gamut + EOTF + tone map)
// 4. Barrier intermediate: GENERAL → TRANSFER_SRC
// 5. Blit intermediate → swapchain image
//
// The intermediate is only allocated/used when active (gamut != BT.709 or EOTF != sRGB).
class color_convert_pipeline
{
  public:
    color_convert_pipeline(vk::Device         device,
                           vk::PhysicalDevice physical,
                           uint32_t           width,
                           uint32_t           height);
    ~color_convert_pipeline();

    color_convert_pipeline(const color_convert_pipeline&)            = delete;
    color_convert_pipeline& operator=(const color_convert_pipeline&) = delete;

    // Update configuration. Sets active_ = true if conversion is needed.
    void update_config(output_gamut gamut, output_eotf eotf, float max_luminance = 1000.0f, int tone_map_op = 0);

    // Whether the pipeline should be dispatched (gamut/EOTF differs from identity).
    bool is_active() const { return active_; }

    // The intermediate RGBA16F image (storage + transfer src/dst).
    vk::Image image() const { return intermediate_image_; }

    uint32_t width() const { return width_; }
    uint32_t height() const { return height_; }

    // Record the compute dispatch into the command buffer.
    // The intermediate must be in GENERAL layout.
    void dispatch(vk::CommandBuffer cmd, uint32_t dispatch_width, uint32_t dispatch_height);

  private:
    void create_intermediate_image();
    void create_pipeline();
    void build_gamut_matrix(output_gamut gamut);

    vk::Device         device_;
    vk::PhysicalDevice physical_;
    uint32_t           width_;
    uint32_t           height_;
    bool               active_ = false;

    // Push constants (updated by update_config)
    color_convert_push_constants push_constants_{};

    // Intermediate image (RGBA16F)
    vk::Image        intermediate_image_;
    vk::DeviceMemory intermediate_memory_;
    vk::ImageView    intermediate_view_;

    // Compute pipeline
    vk::DescriptorSetLayout descriptor_set_layout_;
    vk::DescriptorPool      descriptor_pool_;
    vk::DescriptorSet       descriptor_set_;
    vk::PipelineLayout      pipeline_layout_;
    vk::Pipeline            pipeline_;
    vk::ShaderModule        shader_module_;
};

}} // namespace caspar::vulkan_output
