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

#define VK_USE_PLATFORM_WIN32_KHR
#include <vulkan/vulkan.h>

#include "../consumer/config.h"

#include <array>
#include <cstdint>

namespace caspar { namespace vulkan_output {

class vulkan_device;

/// Push constants matching the compute shader layout.
struct color_convert_push_constants
{
    float gamut_matrix[16]; // 4×4 (only 3×3 used, row-major)
    int   eotf_mode;       // 0=srgb, 1=linear, 2=pq, 3=hlg, 4=gamma24, 5=gamma26
    float max_luminance;   // PQ max nits
    int   tone_map_op;     // 0=none, 1=reinhard, 2=aces_filmic, 3=aces_rrt, 7=hlg_ootf
    float padding2;
};
static_assert(sizeof(color_convert_push_constants) == 80, "Push constants must be 80 bytes");

/// Manages the Vulkan compute pipeline for color space conversion.
/// Owns an intermediate RGBA16F image used as the compute work surface.
///
/// Usage in present path:
///   1. Blit source (shared_pool VkImage) → intermediate (TRANSFER_DST)
///   2. Barrier intermediate → GENERAL
///   3. dispatch(cmd_buffer, width, height)   — runs the compute shader
///   4. Barrier intermediate → TRANSFER_SRC
///   5. Blit intermediate → swapchain (TRANSFER_DST)
class color_convert_pipeline
{
  public:
    color_convert_pipeline(vulkan_device& device, uint32_t width, uint32_t height);
    ~color_convert_pipeline();

    color_convert_pipeline(const color_convert_pipeline&)            = delete;
    color_convert_pipeline& operator=(const color_convert_pipeline&) = delete;

    /// Record compute dispatch into the command buffer.
    /// The intermediate image must be in VK_IMAGE_LAYOUT_GENERAL before calling.
    void dispatch(VkCommandBuffer cmd, uint32_t width, uint32_t height);

    /// Update the push constants (gamut matrix + EOTF mode + tone map) from config.
    void update_config(output_gamut gamut, output_eotf eotf, float max_luminance, int tone_map_op = 0);

    /// Returns true if conversion is needed (gamut != bt709 or eotf != srgb)
    bool is_active() const { return active_; }

    /// The intermediate work image (RGBA16F, STORAGE | TRANSFER_SRC | TRANSFER_DST)
    VkImage     image() const { return image_; }
    VkImageView image_view() const { return image_view_; }

    uint32_t width() const { return width_; }
    uint32_t height() const { return height_; }

  private:
    void create_image();
    void create_pipeline();
    void create_descriptor();

    VkDevice device_ = VK_NULL_HANDLE;

    // Intermediate image
    VkImage        image_      = VK_NULL_HANDLE;
    VkImageView    image_view_ = VK_NULL_HANDLE;
    VkDeviceMemory memory_     = VK_NULL_HANDLE;
    uint32_t       width_      = 0;
    uint32_t       height_     = 0;

    // Pipeline
    VkShaderModule        shader_module_    = VK_NULL_HANDLE;
    VkDescriptorSetLayout desc_set_layout_  = VK_NULL_HANDLE;
    VkPipelineLayout      pipeline_layout_  = VK_NULL_HANDLE;
    VkPipeline            pipeline_         = VK_NULL_HANDLE;
    VkDescriptorPool      descriptor_pool_  = VK_NULL_HANDLE;
    VkDescriptorSet       descriptor_set_   = VK_NULL_HANDLE;

    // State
    color_convert_push_constants push_constants_{};
    bool                         active_ = false;

    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
};

/// Build 3×3 gamut conversion matrix (BT.709 → target) stored in 4×4 layout.
void build_gamut_matrix(output_gamut target, float out_matrix[16]);

}} // namespace caspar::vulkan_output
