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

#include "color_convert_pipeline.h"
#include "vulkan_device.h"
#include "../shaders/color_convert_spv.h"

#include <common/except.h>
#include <common/log.h>

#include <cstring>
#include <algorithm>

namespace caspar { namespace vulkan_output {

namespace {

#define VK_CHECK(call)                                                                                                 \
    do {                                                                                                               \
        VkResult vk_result_ = (call);                                                                                  \
        if (vk_result_ != VK_SUCCESS)                                                                                  \
            CASPAR_THROW_EXCEPTION(caspar_exception() << msg_info(#call " failed: " + std::to_string(vk_result_)));    \
    } while (0)

uint32_t find_memory_type(VkPhysicalDevice phys_dev, uint32_t type_filter, VkMemoryPropertyFlags properties)
{
    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(phys_dev, &mem_props);

    for (uint32_t i = 0; i < mem_props.memoryTypeCount; ++i) {
        if ((type_filter & (1u << i)) && (mem_props.memoryTypes[i].propertyFlags & properties) == properties)
            return i;
    }
    CASPAR_THROW_EXCEPTION(caspar_exception() << msg_info("Failed to find suitable memory type for intermediate image"));
}

// ─── Gamut matrices ────────────────────────────────────────────────────────
// All matrices are BT.709 linear RGB → target linear RGB (3×3, stored row-major in 4×4)
// Derived from CIE XYZ chromaticity coordinates via Bradford chromatic adaptation.

// Identity (BT.709 → BT.709)
constexpr float mat_identity[16] = {
    1.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 1.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 1.0f,
};

// BT.709 → BT.2020
constexpr float mat_709_to_2020[16] = {
    0.6274040f,  0.3292820f,  0.0433136f, 0.0f,
    0.0690970f,  0.9195400f,  0.0113612f, 0.0f,
    0.0163916f,  0.0880132f,  0.8955950f, 0.0f,
    0.0f,        0.0f,        0.0f,       1.0f,
};

// BT.709 → Display P3 (D65)
constexpr float mat_709_to_p3_d65[16] = {
    0.8224620f,  0.1775380f,  0.0000000f, 0.0f,
    0.0331942f,  0.9668058f,  0.0000000f, 0.0f,
    0.0170826f,  0.0723974f,  0.9105200f, 0.0f,
    0.0f,        0.0f,        0.0f,       1.0f,
};

// BT.709 → DCI-P3 (DCI white, D50-ish, uses Bradford adaptation from D65→DCI)
constexpr float mat_709_to_p3_dci[16] = {
    0.8685170f,  0.1283810f,  0.0031015f, 0.0f,
    0.0344530f,  0.9618840f,  0.0036629f, 0.0f,
    0.0167662f,  0.0710578f,  0.9121760f, 0.0f,
    0.0f,        0.0f,        0.0f,       1.0f,
};

// BT.709 → Adobe RGB (1998) (same white point D65, so just primaries transform)
constexpr float mat_709_to_adobe_rgb[16] = {
    0.7151583f,  0.2848417f,  0.0000000f, 0.0f,
    0.0000000f,  1.0000000f,  0.0000000f, 0.0f,
    0.0000000f,  0.0411539f,  0.9588461f, 0.0f,
    0.0f,        0.0f,        0.0f,       1.0f,
};

} // anonymous namespace

// ─── Public ────────────────────────────────────────────────────────────────

color_convert_pipeline::color_convert_pipeline(vulkan_device& device, uint32_t width, uint32_t height)
    : device_(device.device())
    , physical_device_(device.physical_device())
    , queue_(device.present_queue())
    , width_(width)
    , height_(height)
{
    // Validate push constant size against device limits
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(physical_device_, &props);
    if (props.limits.maxPushConstantsSize < sizeof(color_convert_push_constants)) {
        CASPAR_THROW_EXCEPTION(caspar_exception() << msg_info(
            "Device maxPushConstantsSize (" + std::to_string(props.limits.maxPushConstantsSize) +
            ") is less than required " + std::to_string(sizeof(color_convert_push_constants)) + " bytes"));
    }

    // Default: identity matrix, sRGB EOTF, inactive
    std::memcpy(push_constants_.gamut_matrix, mat_identity, sizeof(mat_identity));
    push_constants_.eotf_mode     = 0;
    push_constants_.max_luminance = 1000.0f;

    create_image();
    create_pipeline();
    create_descriptor();

    CASPAR_LOG(debug) << "[color_convert_pipeline] Created " << width_ << "x" << height_
                      << " intermediate image (RGBA16F)";
}

color_convert_pipeline::~color_convert_pipeline()
{
    if (device_ == VK_NULL_HANDLE)
        return;

    // Wait on the present queue where compute dispatches are submitted,
    // rather than stalling all queues with vkDeviceWaitIdle.
    if (queue_ != VK_NULL_HANDLE)
        vkQueueWaitIdle(queue_);
    else
        vkDeviceWaitIdle(device_);

    if (descriptor_pool_ != VK_NULL_HANDLE)
        vkDestroyDescriptorPool(device_, descriptor_pool_, nullptr);
    if (pipeline_ != VK_NULL_HANDLE)
        vkDestroyPipeline(device_, pipeline_, nullptr);
    if (pipeline_layout_ != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(device_, pipeline_layout_, nullptr);
    if (desc_set_layout_ != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(device_, desc_set_layout_, nullptr);
    if (shader_module_ != VK_NULL_HANDLE)
        vkDestroyShaderModule(device_, shader_module_, nullptr);
    if (image_view_ != VK_NULL_HANDLE)
        vkDestroyImageView(device_, image_view_, nullptr);
    if (image_ != VK_NULL_HANDLE)
        vkDestroyImage(device_, image_, nullptr);
    if (memory_ != VK_NULL_HANDLE)
        vkFreeMemory(device_, memory_, nullptr);
}

void color_convert_pipeline::dispatch(VkCommandBuffer cmd, uint32_t width, uint32_t height)
{
    if (!active_)
        return;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_, 0, 1, &descriptor_set_, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(color_convert_push_constants), &push_constants_);

    uint32_t gx = (width + 15) / 16;
    uint32_t gy = (height + 15) / 16;
    vkCmdDispatch(cmd, gx, gy, 1);
}

void color_convert_pipeline::update_config(output_gamut gamut, output_eotf eotf, float max_luminance)
{
    build_gamut_matrix(gamut, push_constants_.gamut_matrix);

    switch (eotf) {
        case output_eotf::srgb:    push_constants_.eotf_mode = 0; break;
        case output_eotf::linear:  push_constants_.eotf_mode = 1; break;
        case output_eotf::pq:      push_constants_.eotf_mode = 2; break;
        case output_eotf::hlg:     push_constants_.eotf_mode = 3; break;
        case output_eotf::gamma24: push_constants_.eotf_mode = 4; break;
        case output_eotf::gamma26: push_constants_.eotf_mode = 5; break;
        default:                   push_constants_.eotf_mode = 0; break;
    }

    push_constants_.max_luminance = max_luminance;
    push_constants_.padding1      = 0.0f;
    push_constants_.padding2      = 0.0f;

    // Active if any conversion is needed
    active_ = (gamut != output_gamut::bt709 || eotf != output_eotf::srgb);

    CASPAR_LOG(debug) << "[color_convert_pipeline] Config updated: gamut="
                      << static_cast<int>(gamut) << " eotf=" << static_cast<int>(eotf)
                      << " max_lum=" << max_luminance << " active=" << (active_ ? "yes" : "no");
}

// ─── Private ───────────────────────────────────────────────────────────────

void color_convert_pipeline::create_image()
{
    VkImageCreateInfo img_info{};
    img_info.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    img_info.imageType     = VK_IMAGE_TYPE_2D;
    img_info.format        = VK_FORMAT_R16G16B16A16_SFLOAT;
    img_info.extent        = {width_, height_, 1};
    img_info.mipLevels     = 1;
    img_info.arrayLayers   = 1;
    img_info.samples       = VK_SAMPLE_COUNT_1_BIT;
    img_info.tiling        = VK_IMAGE_TILING_OPTIMAL;
    img_info.usage         = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                             VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    img_info.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
    img_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VK_CHECK(vkCreateImage(device_, &img_info, nullptr, &image_));

    VkMemoryRequirements mem_reqs;
    vkGetImageMemoryRequirements(device_, image_, &mem_reqs);

    VkMemoryAllocateInfo alloc_info{};
    alloc_info.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize  = mem_reqs.size;
    alloc_info.memoryTypeIndex = find_memory_type(physical_device_, mem_reqs.memoryTypeBits,
                                                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    VK_CHECK(vkAllocateMemory(device_, &alloc_info, nullptr, &memory_));
    VK_CHECK(vkBindImageMemory(device_, image_, memory_, 0));

    // Create image view
    VkImageViewCreateInfo view_info{};
    view_info.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_info.image                           = image_;
    view_info.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
    view_info.format                          = VK_FORMAT_R16G16B16A16_SFLOAT;
    view_info.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    view_info.subresourceRange.baseMipLevel   = 0;
    view_info.subresourceRange.levelCount     = 1;
    view_info.subresourceRange.baseArrayLayer = 0;
    view_info.subresourceRange.layerCount     = 1;

    VK_CHECK(vkCreateImageView(device_, &view_info, nullptr, &image_view_));
}

void color_convert_pipeline::create_pipeline()
{
    // Shader module from embedded SPIR-V
    VkShaderModuleCreateInfo module_info{};
    module_info.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    module_info.codeSize = color_convert_comp_spv_size;
    module_info.pCode    = color_convert_comp_spv;

    VK_CHECK(vkCreateShaderModule(device_, &module_info, nullptr, &shader_module_));

    // Descriptor set layout: one storage image (read-write)
    VkDescriptorSetLayoutBinding binding{};
    binding.binding         = 0;
    binding.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    binding.descriptorCount = 1;
    binding.stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo layout_info{};
    layout_info.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.bindingCount = 1;
    layout_info.pBindings    = &binding;

    VK_CHECK(vkCreateDescriptorSetLayout(device_, &layout_info, nullptr, &desc_set_layout_));

    // Push constant range
    VkPushConstantRange push_range{};
    push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push_range.offset     = 0;
    push_range.size       = sizeof(color_convert_push_constants);

    // Pipeline layout
    VkPipelineLayoutCreateInfo pipe_layout_info{};
    pipe_layout_info.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipe_layout_info.setLayoutCount         = 1;
    pipe_layout_info.pSetLayouts            = &desc_set_layout_;
    pipe_layout_info.pushConstantRangeCount = 1;
    pipe_layout_info.pPushConstantRanges    = &push_range;

    VK_CHECK(vkCreatePipelineLayout(device_, &pipe_layout_info, nullptr, &pipeline_layout_));

    // Compute pipeline
    VkComputePipelineCreateInfo pipe_info{};
    pipe_info.sType              = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipe_info.stage.sType        = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipe_info.stage.stage        = VK_SHADER_STAGE_COMPUTE_BIT;
    pipe_info.stage.module       = shader_module_;
    pipe_info.stage.pName        = "main";
    pipe_info.layout             = pipeline_layout_;

    VK_CHECK(vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipe_info, nullptr, &pipeline_));
}

void color_convert_pipeline::create_descriptor()
{
    // Descriptor pool
    VkDescriptorPoolSize pool_size{};
    pool_size.type            = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    pool_size.descriptorCount = 1;

    VkDescriptorPoolCreateInfo pool_info{};
    pool_info.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.maxSets       = 1;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes    = &pool_size;

    VK_CHECK(vkCreateDescriptorPool(device_, &pool_info, nullptr, &descriptor_pool_));

    // Allocate descriptor set
    VkDescriptorSetAllocateInfo alloc_info{};
    alloc_info.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool     = descriptor_pool_;
    alloc_info.descriptorSetCount = 1;
    alloc_info.pSetLayouts        = &desc_set_layout_;

    VK_CHECK(vkAllocateDescriptorSets(device_, &alloc_info, &descriptor_set_));

    // Write descriptor — bind our intermediate image
    VkDescriptorImageInfo img_desc{};
    img_desc.imageView   = image_view_;
    img_desc.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkWriteDescriptorSet write{};
    write.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet          = descriptor_set_;
    write.dstBinding      = 0;
    write.descriptorCount = 1;
    write.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    write.pImageInfo      = &img_desc;

    vkUpdateDescriptorSets(device_, 1, &write, 0, nullptr);
}

// ─── Gamut matrix builder ──────────────────────────────────────────────────

void build_gamut_matrix(output_gamut target, float out_matrix[16])
{
    const float* src = nullptr;
    switch (target) {
        case output_gamut::bt2020:    src = mat_709_to_2020;     break;
        case output_gamut::p3_d65:    src = mat_709_to_p3_d65;   break;
        case output_gamut::p3_dci:    src = mat_709_to_p3_dci;   break;
        case output_gamut::adobe_rgb: src = mat_709_to_adobe_rgb; break;
        default:                      src = mat_identity;         break;
    }
    std::memcpy(out_matrix, src, 16 * sizeof(float));
}

}} // namespace caspar::vulkan_output
