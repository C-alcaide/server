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

#include <common/log.h>

#include <cstring>

// Embedded SPIR-V (compiled by CMake glslc)
#include "color_convert_comp.h"

namespace caspar { namespace vulkan_output {

// ─── Gamut Matrices (3×3 in row-major 4×4) ──────────────────────────────────
// These convert from BT.709 (sRGB) primaries to the target gamut in linear light.

static const float kIdentity[16] = {
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1,
};

// BT.709 → BT.2020 (Bradford chromatic adaptation)
static const float kBT709toBT2020[16] = {
    0.6274f,  0.3293f,  0.0433f, 0,
    0.0691f,  0.9195f,  0.0114f, 0,
    0.0164f,  0.0880f,  0.8956f, 0,
    0,        0,        0,       1,
};

// BT.709 → Display P3 (D65)
static const float kBT709toP3D65[16] = {
    0.8225f,  0.1774f,  0.0000f, 0,
    0.0332f,  0.9669f,  0.0000f, 0,
    0.0171f,  0.0724f,  0.9108f, 0,
    0,        0,        0,       1,
};

// ─── Implementation ─────────────────────────────────────────────────────────

color_convert_pipeline::color_convert_pipeline(vk::Device         device,
                                               vk::PhysicalDevice physical,
                                               uint32_t           width,
                                               uint32_t           height)
    : device_(device)
    , physical_(physical)
    , width_(width)
    , height_(height)
{
    create_intermediate_image();
    create_pipeline();

    // Default: identity gamut, sRGB EOTF (no-op)
    std::memcpy(push_constants_.gamut_matrix, kIdentity, sizeof(kIdentity));
    push_constants_.eotf_mode     = 0;
    push_constants_.max_luminance = 1000.0f;
    push_constants_.tone_map_op   = 0;
}

color_convert_pipeline::~color_convert_pipeline()
{
    device_.waitIdle();
    device_.destroyPipeline(pipeline_);
    device_.destroyPipelineLayout(pipeline_layout_);
    device_.destroyDescriptorPool(descriptor_pool_);
    device_.destroyDescriptorSetLayout(descriptor_set_layout_);
    device_.destroyShaderModule(shader_module_);
    device_.destroyImageView(intermediate_view_);
    device_.destroyImage(intermediate_image_);
    device_.freeMemory(intermediate_memory_);
}

void color_convert_pipeline::update_config(output_gamut gamut,
                                           output_eotf  eotf,
                                           float        max_luminance,
                                           int          tone_map_op)
{
    build_gamut_matrix(gamut);

    switch (eotf) {
    case output_eotf::srgb:    push_constants_.eotf_mode = 0; break;
    case output_eotf::linear:  push_constants_.eotf_mode = 1; break;
    case output_eotf::pq:      push_constants_.eotf_mode = 2; break;
    case output_eotf::hlg:     push_constants_.eotf_mode = 3; break;
    case output_eotf::gamma24: push_constants_.eotf_mode = 4; break;
    }

    push_constants_.max_luminance = max_luminance;
    push_constants_.tone_map_op   = tone_map_op;

    // Active if anything differs from passthrough
    active_ = (gamut != output_gamut::bt709 || eotf != output_eotf::srgb || tone_map_op != 0);
}

void color_convert_pipeline::dispatch(vk::CommandBuffer cmd, uint32_t dispatch_width, uint32_t dispatch_height)
{
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline_);
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipeline_layout_, 0, descriptor_set_, nullptr);
    cmd.pushConstants(pipeline_layout_, vk::ShaderStageFlagBits::eCompute, 0,
                      sizeof(push_constants_), &push_constants_);

    uint32_t gx = (dispatch_width + 15) / 16;
    uint32_t gy = (dispatch_height + 15) / 16;
    cmd.dispatch(gx, gy, 1);
}

void color_convert_pipeline::build_gamut_matrix(output_gamut gamut)
{
    switch (gamut) {
    case output_gamut::bt2020:
        std::memcpy(push_constants_.gamut_matrix, kBT709toBT2020, sizeof(kBT709toBT2020));
        break;
    case output_gamut::p3_d65:
        std::memcpy(push_constants_.gamut_matrix, kBT709toP3D65, sizeof(kBT709toP3D65));
        break;
    case output_gamut::bt709:
    default:
        std::memcpy(push_constants_.gamut_matrix, kIdentity, sizeof(kIdentity));
        break;
    }
}

void color_convert_pipeline::create_intermediate_image()
{
    // RGBA16F intermediate for compute read/write + transfer src/dst
    vk::ImageCreateInfo ci{};
    ci.imageType   = vk::ImageType::e2D;
    ci.format      = vk::Format::eR16G16B16A16Sfloat;
    ci.extent      = vk::Extent3D{width_, height_, 1};
    ci.mipLevels   = 1;
    ci.arrayLayers = 1;
    ci.samples     = vk::SampleCountFlagBits::e1;
    ci.tiling      = vk::ImageTiling::eOptimal;
    ci.usage       = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc |
                     vk::ImageUsageFlagBits::eTransferDst;
    ci.sharingMode = vk::SharingMode::eExclusive;

    intermediate_image_ = device_.createImage(ci);

    auto mem_reqs = device_.getImageMemoryRequirements(intermediate_image_);
    auto mem_props = physical_.getMemoryProperties();

    uint32_t mem_type_idx = 0;
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; ++i) {
        if ((mem_reqs.memoryTypeBits & (1u << i)) &&
            (mem_props.memoryTypes[i].propertyFlags & vk::MemoryPropertyFlagBits::eDeviceLocal)) {
            mem_type_idx = i;
            break;
        }
    }

    vk::MemoryAllocateInfo alloc_info{};
    alloc_info.allocationSize  = mem_reqs.size;
    alloc_info.memoryTypeIndex = mem_type_idx;
    intermediate_memory_       = device_.allocateMemory(alloc_info);
    device_.bindImageMemory(intermediate_image_, intermediate_memory_, 0);

    // Image view for descriptor binding
    vk::ImageViewCreateInfo iv_ci{};
    iv_ci.image                           = intermediate_image_;
    iv_ci.viewType                        = vk::ImageViewType::e2D;
    iv_ci.format                          = vk::Format::eR16G16B16A16Sfloat;
    iv_ci.subresourceRange.aspectMask     = vk::ImageAspectFlagBits::eColor;
    iv_ci.subresourceRange.baseMipLevel   = 0;
    iv_ci.subresourceRange.levelCount     = 1;
    iv_ci.subresourceRange.baseArrayLayer = 0;
    iv_ci.subresourceRange.layerCount     = 1;
    intermediate_view_ = device_.createImageView(iv_ci);
}

void color_convert_pipeline::create_pipeline()
{
    // Shader module from embedded SPIR-V
    vk::ShaderModuleCreateInfo shader_ci{};
    shader_ci.codeSize = sizeof(caspar::vulkan_output::color_convert_comp_spv);
    shader_ci.pCode    = reinterpret_cast<const uint32_t*>(caspar::vulkan_output::color_convert_comp_spv);
    shader_module_     = device_.createShaderModule(shader_ci);

    // Descriptor set layout: one storage image
    vk::DescriptorSetLayoutBinding binding{};
    binding.binding         = 0;
    binding.descriptorType  = vk::DescriptorType::eStorageImage;
    binding.descriptorCount = 1;
    binding.stageFlags      = vk::ShaderStageFlagBits::eCompute;

    vk::DescriptorSetLayoutCreateInfo dsl_ci{};
    dsl_ci.bindingCount = 1;
    dsl_ci.pBindings    = &binding;
    descriptor_set_layout_ = device_.createDescriptorSetLayout(dsl_ci);

    // Descriptor pool
    vk::DescriptorPoolSize pool_size{vk::DescriptorType::eStorageImage, 1};
    vk::DescriptorPoolCreateInfo pool_ci{};
    pool_ci.maxSets       = 1;
    pool_ci.poolSizeCount = 1;
    pool_ci.pPoolSizes    = &pool_size;
    descriptor_pool_      = device_.createDescriptorPool(pool_ci);

    // Allocate descriptor set
    vk::DescriptorSetAllocateInfo ds_alloc{};
    ds_alloc.descriptorPool     = descriptor_pool_;
    ds_alloc.descriptorSetCount = 1;
    ds_alloc.pSetLayouts        = &descriptor_set_layout_;
    descriptor_set_ = device_.allocateDescriptorSets(ds_alloc)[0];

    // Write descriptor (point at intermediate image)
    vk::DescriptorImageInfo img_info{};
    img_info.imageView   = intermediate_view_;
    img_info.imageLayout = vk::ImageLayout::eGeneral;

    vk::WriteDescriptorSet write{};
    write.dstSet          = descriptor_set_;
    write.dstBinding      = 0;
    write.descriptorCount = 1;
    write.descriptorType  = vk::DescriptorType::eStorageImage;
    write.pImageInfo      = &img_info;
    device_.updateDescriptorSets(write, nullptr);

    // Pipeline layout with push constants
    vk::PushConstantRange push_range{};
    push_range.stageFlags = vk::ShaderStageFlagBits::eCompute;
    push_range.offset     = 0;
    push_range.size       = sizeof(color_convert_push_constants);

    vk::PipelineLayoutCreateInfo layout_ci{};
    layout_ci.setLayoutCount         = 1;
    layout_ci.pSetLayouts            = &descriptor_set_layout_;
    layout_ci.pushConstantRangeCount = 1;
    layout_ci.pPushConstantRanges    = &push_range;
    pipeline_layout_ = device_.createPipelineLayout(layout_ci);

    // Compute pipeline
    vk::ComputePipelineCreateInfo pipe_ci{};
    pipe_ci.stage.stage  = vk::ShaderStageFlagBits::eCompute;
    pipe_ci.stage.module = shader_module_;
    pipe_ci.stage.pName  = "main";
    pipe_ci.layout       = pipeline_layout_;

    auto result = device_.createComputePipeline(nullptr, pipe_ci);
    pipeline_ = result.value;
}

}} // namespace caspar::vulkan_output
