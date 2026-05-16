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

#include "pipeline.h"
#include "../image/image_kernel.h"
#include "texture.h"

#include "vk_image_fragment.h"
#include "vk_image_vertex.h"
#include <core/frame/geometry.h>

#include <vulkan/vulkan.hpp>

#include <cstring>
#include <unordered_map>

namespace caspar { namespace accelerator { namespace vulkan {

std::vector<vk::PipelineShaderStageCreateInfo> create_shader_program(vk::Device device)
{
    // Helper to create shader module
    auto createShaderModule = [&](const uint8_t* code, size_t size) {
        vk::ShaderModuleCreateInfo createInfo{};
        createInfo.codeSize = size;
        createInfo.pCode    = reinterpret_cast<const uint32_t*>(code);
        return device.createShaderModule(createInfo);
    };

    auto vertShaderModule = createShaderModule(vertex_shader, sizeof(vertex_shader) - 1);
    auto fragShaderModule = createShaderModule(fragment_shader, sizeof(fragment_shader) - 1);

    vk::PipelineShaderStageCreateInfo vertShaderStageInfo;
    vertShaderStageInfo.stage  = vk::ShaderStageFlagBits::eVertex;
    vertShaderStageInfo.module = vertShaderModule;
    vertShaderStageInfo.pName  = "main";

    vk::PipelineShaderStageCreateInfo fragShaderStageInfo;
    fragShaderStageInfo.stage  = vk::ShaderStageFlagBits::eFragment;
    fragShaderStageInfo.module = fragShaderModule;
    fragShaderStageInfo.pName  = "main";

    return {vertShaderStageInfo, fragShaderStageInfo};
}

std::array<vk::VertexInputAttributeDescription, 2> get_attribute_descriptions(uint32_t binding)
{
    std::array<vk::VertexInputAttributeDescription, 2> attributeDescriptions{
        {{0, binding, vk::Format::eR32G32Sfloat, 0}, {1, binding, vk::Format::eR32G32B32A32Sfloat, 2 * sizeof(float)}}};

    return attributeDescriptions;
}

// Descriptor set ring buffer size. With ~12 draw calls per frame and 3 frames in flight,
// we consume ~36 sets before the GPU retires the oldest frame. 128 gives ~10 frames of
// headroom without any per-set fence tracking. A future improvement could add per-slot
// timeline semaphore stamps (vkGetSemaphoreCounterValue) to guarantee safety regardless
// of pool size, but the overhead is unnecessary while 128 >> typical in-flight usage.
const int DescriptorPoolSize = 128;
const int BindlessTextureCount = 8;
// UBO ring buffer: round sizeof(uniform_block) up to multiple of 256 for alignment
const vk::DeviceSize UBO_SLOT_SIZE  = (sizeof(uniform_block) + 255) & ~vk::DeviceSize(255);
const vk::DeviceSize UBO_TOTAL_SIZE = UBO_SLOT_SIZE * DescriptorPoolSize;

struct pipeline::impl
{
    vk::Device device_;
    vk::Format format_;
    vk::PhysicalDeviceMemoryProperties memProperties_;

    vk::Sampler                    textureSampler_;
    vk::Sampler                    keySampler_;
    vk::DescriptorSetLayout        descriptorSetLayout_;
    vk::DescriptorPool             descriptorPool_;
    std::vector<vk::DescriptorSet> descriptorSets_;

    vk::PipelineLayout pipelineLayout_;
    vk::Pipeline       pipeline_;

    // UBO ring buffer
    vk::Buffer       uboBuffer_;
    vk::DeviceMemory uboMemory_;
    uint8_t*         uboMapped_ = nullptr;

    size_t currentDescriptorSet_ = 0;

    impl(const impl&)            = delete;
    impl& operator=(const impl&) = delete;

    void setup_descriptors()
    {
        // Binding 0: bindless texture array for planes (up to 4), local_key, and layer_key
        vk::DescriptorSetLayoutBinding texturesLayoutBinding{};
        texturesLayoutBinding.binding         = 0;
        texturesLayoutBinding.descriptorType  = vk::DescriptorType::eCombinedImageSampler;
        texturesLayoutBinding.descriptorCount = BindlessTextureCount;
        texturesLayoutBinding.stageFlags      = vk::ShaderStageFlagBits::eFragment;

        // Binding 1: input attachment for background
        vk::DescriptorSetLayoutBinding backgroundLayoutBinding{};
        backgroundLayoutBinding.binding         = 1;
        backgroundLayoutBinding.descriptorType  = vk::DescriptorType::eInputAttachment;
        backgroundLayoutBinding.descriptorCount = 1;
        backgroundLayoutBinding.stageFlags      = vk::ShaderStageFlagBits::eFragment;

        // Binding 2: UBO
        vk::DescriptorSetLayoutBinding uboLayoutBinding{};
        uboLayoutBinding.binding         = 2;
        uboLayoutBinding.descriptorType  = vk::DescriptorType::eUniformBuffer;
        uboLayoutBinding.descriptorCount = 1;
        uboLayoutBinding.stageFlags      = vk::ShaderStageFlagBits::eFragment;

        // Binding 3: 3D LUT sampler3D
        vk::DescriptorSetLayoutBinding lut3dLayoutBinding{};
        lut3dLayoutBinding.binding         = 3;
        lut3dLayoutBinding.descriptorType  = vk::DescriptorType::eCombinedImageSampler;
        lut3dLayoutBinding.descriptorCount = 1;
        lut3dLayoutBinding.stageFlags      = vk::ShaderStageFlagBits::eFragment;

        // Binding 4: hue curve sampler2D
        vk::DescriptorSetLayoutBinding hueCurveLayoutBinding{};
        hueCurveLayoutBinding.binding         = 4;
        hueCurveLayoutBinding.descriptorType  = vk::DescriptorType::eCombinedImageSampler;
        hueCurveLayoutBinding.descriptorCount = 1;
        hueCurveLayoutBinding.stageFlags      = vk::ShaderStageFlagBits::eFragment;

        // Binding 5: curve LUT sampler2D
        vk::DescriptorSetLayoutBinding curveLutLayoutBinding{};
        curveLutLayoutBinding.binding         = 5;
        curveLutLayoutBinding.descriptorType  = vk::DescriptorType::eCombinedImageSampler;
        curveLutLayoutBinding.descriptorCount = 1;
        curveLutLayoutBinding.stageFlags      = vk::ShaderStageFlagBits::eFragment;

        vk::DescriptorSetLayoutCreateInfo layoutInfo{};
        std::array bindings{texturesLayoutBinding, backgroundLayoutBinding, uboLayoutBinding,
                            lut3dLayoutBinding, hueCurveLayoutBinding, curveLutLayoutBinding};
        layoutInfo.setBindings(bindings);

        std::array<vk::DescriptorBindingFlags, 6> bindingFlags{
            vk::DescriptorBindingFlagBits::ePartiallyBound, // 0: textures
            vk::DescriptorBindingFlags{},                   // 1: background
            vk::DescriptorBindingFlags{},                   // 2: UBO
            vk::DescriptorBindingFlagBits::ePartiallyBound, // 3: lut3d
            vk::DescriptorBindingFlagBits::ePartiallyBound, // 4: hue curve
            vk::DescriptorBindingFlagBits::ePartiallyBound  // 5: curve lut
        };
        vk::DescriptorSetLayoutBindingFlagsCreateInfo bindingFlagsInfo;
        bindingFlagsInfo.setBindingFlags(bindingFlags);
        layoutInfo.pNext = &bindingFlagsInfo;

        descriptorSetLayout_ = device_.createDescriptorSetLayout(layoutInfo);

        // Create descriptor pool
        vk::DescriptorPoolSize samplerPoolSize(vk::DescriptorType::eCombinedImageSampler,
                                               (BindlessTextureCount + 3) * DescriptorPoolSize);
        vk::DescriptorPoolSize inputAttachmentPoolSize(vk::DescriptorType::eInputAttachment,
                                                       1 * DescriptorPoolSize);
        vk::DescriptorPoolSize uboPoolSize(vk::DescriptorType::eUniformBuffer,
                                           1 * DescriptorPoolSize);

        std::array poolSizes{samplerPoolSize, inputAttachmentPoolSize, uboPoolSize};

        vk::DescriptorPoolCreateInfo poolInfo{};
        poolInfo.maxSets = DescriptorPoolSize;

        poolInfo.setPoolSizes(poolSizes);
        descriptorPool_ = device_.createDescriptorPool(poolInfo);

        // Allocate descriptor sets
        std::vector<vk::DescriptorSetLayout> layouts(DescriptorPoolSize, descriptorSetLayout_);
        vk::DescriptorSetAllocateInfo        allocInfo;
        allocInfo.descriptorPool = descriptorPool_;
        allocInfo.setSetLayouts(layouts);

        descriptorSets_ = device_.allocateDescriptorSets(allocInfo);
    }

    void setup_sampler()
    {
        vk::SamplerCreateInfo samplerInfo{};

        samplerInfo.magFilter               = vk::Filter::eLinear;
        samplerInfo.minFilter               = vk::Filter::eLinear;
        samplerInfo.mipmapMode              = vk::SamplerMipmapMode::eLinear;
        samplerInfo.addressModeU            = vk::SamplerAddressMode::eRepeat;
        samplerInfo.addressModeV            = vk::SamplerAddressMode::eRepeat;
        samplerInfo.addressModeW            = vk::SamplerAddressMode::eRepeat;
        samplerInfo.mipLodBias              = 0.0f;
        samplerInfo.anisotropyEnable        = VK_FALSE;
        samplerInfo.maxAnisotropy           = 2;
        samplerInfo.compareEnable           = VK_FALSE;
        samplerInfo.compareOp               = vk::CompareOp::eAlways;
        samplerInfo.minLod                  = 0.0f;
        samplerInfo.maxLod                  = 0.0f;
        samplerInfo.borderColor             = vk::BorderColor::eIntOpaqueBlack;
        samplerInfo.unnormalizedCoordinates = VK_FALSE;

        textureSampler_ = device_.createSampler(samplerInfo);

        samplerInfo.magFilter  = vk::Filter::eNearest;
        samplerInfo.minFilter  = vk::Filter::eNearest;
        samplerInfo.mipmapMode = vk::SamplerMipmapMode::eNearest;
        keySampler_            = device_.createSampler(samplerInfo);
    }

  public:
    uint32_t findMemoryType(uint32_t typeMask, vk::MemoryPropertyFlags properties)
    {
        for (uint32_t i = 0; i < memProperties_.memoryTypeCount; ++i) {
            if ((typeMask & (1 << i)) &&
                ((memProperties_.memoryTypes[i].propertyFlags & properties) == properties)) {
                return i;
            }
        }
        throw std::runtime_error("[Vulkan pipeline] Failed to find suitable memory type");
    }

    void setup_ubo_ring()
    {
        vk::BufferCreateInfo bufferInfo{};
        bufferInfo.size        = UBO_TOTAL_SIZE;
        bufferInfo.usage       = vk::BufferUsageFlagBits::eUniformBuffer;
        bufferInfo.sharingMode = vk::SharingMode::eExclusive;

        uboBuffer_ = device_.createBuffer(bufferInfo);

        auto memReq = device_.getBufferMemoryRequirements(uboBuffer_);

        vk::MemoryAllocateInfo allocInfo{};
        allocInfo.allocationSize  = memReq.size;
        allocInfo.memoryTypeIndex = findMemoryType(
            memReq.memoryTypeBits,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

        uboMemory_ = device_.allocateMemory(allocInfo);
        device_.bindBufferMemory(uboBuffer_, uboMemory_, 0);

        uboMapped_ = static_cast<uint8_t*>(device_.mapMemory(uboMemory_, 0, UBO_TOTAL_SIZE));
    }

    impl(vk::Device device, vk::Format format, vk::PhysicalDeviceMemoryProperties memProperties)
        : device_(device)
        , format_(format)
        , memProperties_(memProperties)
    {
        setup_descriptors();

        setup_sampler();

        setup_ubo_ring();

        // Vertex input
        auto attributeDescriptions = get_attribute_descriptions(0);

        auto vertexBindings = vk::VertexInputBindingDescription(0, sizeof(float) * 6, vk::VertexInputRate::eVertex);
        vk::PipelineVertexInputStateCreateInfo vertexInputInfo;
        vertexInputInfo.setVertexBindingDescriptions(vertexBindings);
        vertexInputInfo.setVertexAttributeDescriptions(attributeDescriptions);

        // Input assembly
        vk::PipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.topology               = vk::PrimitiveTopology::eTriangleFan;
        inputAssembly.primitiveRestartEnable = VK_TRUE;

        vk::PipelineViewportStateCreateInfo viewportState{};
        viewportState.scissorCount  = 1;
        viewportState.viewportCount = 1;
        vk::DynamicState dynamicStates[]{vk::DynamicState::eViewport, vk::DynamicState::eScissor};

        // Rasterizer
        vk::PipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.depthClampEnable        = VK_FALSE;
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        rasterizer.polygonMode             = vk::PolygonMode::eFill;
        rasterizer.cullMode                = vk::CullModeFlagBits::eNone;
        rasterizer.frontFace               = vk::FrontFace::eClockwise;
        rasterizer.depthBiasEnable         = VK_FALSE;
        rasterizer.lineWidth               = 1.0f;

        // Multisampling
        vk::PipelineMultisampleStateCreateInfo multisampling{};
        multisampling.rasterizationSamples = vk::SampleCountFlagBits::e1;
        multisampling.sampleShadingEnable  = VK_FALSE;

        // Color blending
        vk::PipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.blendEnable = vk::False;

        colorBlendAttachment.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                                              vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;

        vk::PipelineColorBlendStateCreateInfo colorBlending{};
        colorBlending.logicOpEnable = vk::False;
        colorBlending.logicOp       = vk::LogicOp::eCopy;
        colorBlending.setAttachments(colorBlendAttachment);

        // Pipeline layout (no push constants — we use UBO)
        vk::PipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.setSetLayouts(descriptorSetLayout_);

        pipelineLayout_ = device_.createPipelineLayout(pipelineLayoutInfo);

        vk::PipelineDynamicStateCreateInfo dynamicState{};
        dynamicState.setDynamicStates(dynamicStates);

        // Graphics pipeline
        vk::GraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.pVertexInputState   = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState      = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pDynamicState       = &dynamicState;
        pipelineInfo.pMultisampleState   = &multisampling;
        pipelineInfo.pColorBlendState    = &colorBlending;
        pipelineInfo.layout              = pipelineLayout_;
        pipelineInfo.renderPass          = nullptr;
        pipelineInfo.subpass             = 0;

        auto shaderStages = std::move(create_shader_program(device_));
        pipelineInfo.setStages(shaderStages);

        vk::PipelineRenderingCreateInfo rendering_info{};
        rendering_info.setColorAttachmentFormats({format});

        // VK_KHR_dynamic_rendering_local_read requires declaring which
        // color attachments are also used as input attachments.
        // Color attachment 0 maps to input attachment index 0.
        uint32_t colorInputIndex = 0;
        vk::RenderingInputAttachmentIndexInfoKHR inputAttachmentInfo{};
        inputAttachmentInfo.colorAttachmentCount        = 1;
        inputAttachmentInfo.pColorAttachmentInputIndices = &colorInputIndex;
        inputAttachmentInfo.pDepthInputAttachmentIndex   = nullptr;
        inputAttachmentInfo.pStencilInputAttachmentIndex = nullptr;

        rendering_info.pNext = &inputAttachmentInfo;
        pipelineInfo.pNext = &rendering_info;

        pipeline_ = device_.createGraphicsPipeline(nullptr, pipelineInfo).value;

        // Cleanup shader modules after pipeline creation
        for (auto& shaderStage : shaderStages) {
            device_.destroyShaderModule(shaderStage.module);
        }
    }

    vk::DescriptorSet acquire_descriptor_set(const uniform_block& params,
                                              const std::array<vk::ImageView, 10>& textures)
    {
        // C++ textures array layout:
        //   [0] = background attachment, [1..4] = planes, [5] = local_key, [6] = layer_key
        //   [7] = lut3d, [8] = hue_curve, [9] = curve_lut

        // Shader bindless textures[N] layout:
        //   [0..3] = planes, [4] = local_key, [5] = layer_key

        auto  setIndex        = currentDescriptorSet_;
        auto  descriptorSet   = descriptorSets_[setIndex];
        currentDescriptorSet_ = (currentDescriptorSet_ + 1) % DescriptorPoolSize;

        // Copy UBO data to the ring buffer slot
        std::memcpy(uboMapped_ + setIndex * UBO_SLOT_SIZE, &params, sizeof(uniform_block));

        // Bind planes, local_key, and layer_key to the bindless texture array
        std::array<vk::DescriptorImageInfo, 6> textureInfos;
        for (int i = 0; i < 6; ++i) {
            textureInfos[i].sampler     = textureSampler_;
            textureInfos[i].imageView   = textures[i + 1];
            textureInfos[i].imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        }

        // Override samplers for local_key and layer_key to use nearest filtering
        textureInfos[4].sampler = keySampler_;
        textureInfos[5].sampler = keySampler_;

        vk::WriteDescriptorSet texturesWrite{};
        texturesWrite.dstSet          = descriptorSet;
        texturesWrite.dstBinding      = 0;
        texturesWrite.dstArrayElement = 0;
        texturesWrite.descriptorType  = vk::DescriptorType::eCombinedImageSampler;
        texturesWrite.setImageInfo(textureInfos);
        texturesWrite.descriptorCount = 6;

        // Bind background attachment as input attachment
        vk::DescriptorImageInfo backgroundInfo{};
        backgroundInfo.imageLayout = vk::ImageLayout::eRenderingLocalRead;
        backgroundInfo.imageView   = textures[0];

        vk::WriteDescriptorSet backgroundWrite{};
        backgroundWrite.dstSet          = descriptorSet;
        backgroundWrite.dstBinding      = 1;
        backgroundWrite.dstArrayElement = 0;
        backgroundWrite.descriptorType  = vk::DescriptorType::eInputAttachment;
        backgroundWrite.setImageInfo(backgroundInfo);

        // Bind UBO
        vk::DescriptorBufferInfo uboInfo{};
        uboInfo.buffer = uboBuffer_;
        uboInfo.offset = setIndex * UBO_SLOT_SIZE;
        uboInfo.range  = sizeof(uniform_block);

        vk::WriteDescriptorSet uboWrite{};
        uboWrite.dstSet          = descriptorSet;
        uboWrite.dstBinding      = 2;
        uboWrite.dstArrayElement = 0;
        uboWrite.descriptorType  = vk::DescriptorType::eUniformBuffer;
        uboWrite.setBufferInfo(uboInfo);

        // Collect writes
        std::vector<vk::WriteDescriptorSet> writes{backgroundWrite, texturesWrite, uboWrite};

        // Binding 3: 3D LUT (if present)
        vk::DescriptorImageInfo lut3dInfo{};
        if (textures[7]) {
            lut3dInfo.sampler     = textureSampler_;
            lut3dInfo.imageView   = textures[7];
            lut3dInfo.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;

            vk::WriteDescriptorSet lut3dWrite{};
            lut3dWrite.dstSet          = descriptorSet;
            lut3dWrite.dstBinding      = 3;
            lut3dWrite.dstArrayElement = 0;
            lut3dWrite.descriptorType  = vk::DescriptorType::eCombinedImageSampler;
            lut3dWrite.setImageInfo(lut3dInfo);
            writes.push_back(lut3dWrite);
        }

        // Binding 4: hue curve (if present)
        vk::DescriptorImageInfo hueCurveInfo{};
        if (textures[8]) {
            hueCurveInfo.sampler     = textureSampler_;
            hueCurveInfo.imageView   = textures[8];
            hueCurveInfo.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;

            vk::WriteDescriptorSet hueCurveWrite{};
            hueCurveWrite.dstSet          = descriptorSet;
            hueCurveWrite.dstBinding      = 4;
            hueCurveWrite.dstArrayElement = 0;
            hueCurveWrite.descriptorType  = vk::DescriptorType::eCombinedImageSampler;
            hueCurveWrite.setImageInfo(hueCurveInfo);
            writes.push_back(hueCurveWrite);
        }

        // Binding 5: curve LUT (if present)
        vk::DescriptorImageInfo curveLutInfo{};
        if (textures[9]) {
            curveLutInfo.sampler     = textureSampler_;
            curveLutInfo.imageView   = textures[9];
            curveLutInfo.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;

            vk::WriteDescriptorSet curveLutWrite{};
            curveLutWrite.dstSet          = descriptorSet;
            curveLutWrite.dstBinding      = 5;
            curveLutWrite.dstArrayElement = 0;
            curveLutWrite.descriptorType  = vk::DescriptorType::eCombinedImageSampler;
            curveLutWrite.setImageInfo(curveLutInfo);
            writes.push_back(curveLutWrite);
        }

        device_.updateDescriptorSets(writes, nullptr);

        return descriptorSet;
    }

    void draw(vk::CommandBuffer                    commandBuffer,
              vk::Buffer                           vertexBuffer,
              uint32_t                             coords_count,
              uint32_t                             vertex_buffer_offset,
              const uniform_block&                 params,
              const std::array<vk::ImageView, 10>& textures)
    {
        auto descriptorSet = acquire_descriptor_set(params, textures);
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline_);
        commandBuffer.bindVertexBuffers(0, vertexBuffer, {vertex_buffer_offset});
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout_, 0, descriptorSet, nullptr);
        commandBuffer.draw(coords_count, 1, 0, 0);
    }

    ~impl()
    {
        if (uboMapped_) {
            device_.unmapMemory(uboMemory_);
        }
        if (uboBuffer_) {
            device_.destroyBuffer(uboBuffer_);
        }
        if (uboMemory_) {
            device_.freeMemory(uboMemory_);
        }

        device_.destroyDescriptorPool(descriptorPool_);
        device_.destroyDescriptorSetLayout(descriptorSetLayout_);
        device_.destroySampler(textureSampler_);
        device_.destroySampler(keySampler_);

        device_.destroyPipeline(pipeline_);
        device_.destroyPipelineLayout(pipelineLayout_);
    }
};

pipeline::pipeline(vk::Device device, vk::Format format, vk::PhysicalDeviceMemoryProperties memProperties)
    : impl_(new impl(device, format, memProperties))
{
}
pipeline::~pipeline() {}

void pipeline::draw(vk::CommandBuffer                    commandBuffer,
                    vk::Buffer                           vertexBuffer,
                    uint32_t                             coords_count,
                    uint32_t                             vertex_buffer_offset,
                    const uniform_block&                 params,
                    const std::array<vk::ImageView, 10>& textures)
{
    impl_->draw(commandBuffer, vertexBuffer, coords_count, vertex_buffer_offset, params, textures);
}

vk::Pipeline pipeline::id() const { return impl_->pipeline_; }

}}} // namespace caspar::accelerator::vulkan
