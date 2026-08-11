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

std::vector<vk::PipelineShaderStageCreateInfo> create_shader_program(vk::Device device,
                                                                    const std::vector<uint32_t>& frag_spirv)
{
    // Helper to create shader module
    auto createShaderModule = [&](const uint8_t* code, size_t size) {
        vk::ShaderModuleCreateInfo createInfo{};
        createInfo.codeSize = size;
        createInfo.pCode    = reinterpret_cast<const uint32_t*>(code);
        return device.createShaderModule(createInfo);
    };

    auto vertShaderModule = createShaderModule(vertex_shader, sizeof(vertex_shader) - 1);
    // The build-time SPIR-V unless a caller supplied its own. sizeof-1 drops bin2c's
    // terminating NUL, which is not part of the SPIR-V.
    auto fragShaderModule = frag_spirv.empty()
                                ? createShaderModule(fragment_shader, sizeof(fragment_shader) - 1)
                                : createShaderModule(reinterpret_cast<const uint8_t*>(frag_spirv.data()),
                                                     frag_spirv.size() * sizeof(uint32_t));

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

// Descriptor set ring buffer size, SHARED by every channel using this bit depth
// on this device (see device::get_pipeline) — not per-channel. With ~12 draw
// calls per frame and 3 frames in flight, one channel consumes ~36 sets before
// the GPU retires its oldest frame; N concurrent channels consume ~36*N. At the
// previous size of 128, two 8-bit channels with ~20 layers each (~132 sets)
// already exceeded it, silently rewriting a slot still bound by an in-flight
// command buffer. 2048 gives headroom for a much larger multi-channel rig
// without any per-set fence tracking. This is a mitigation, not a proof of
// safety — a future improvement should add per-slot timeline semaphore stamps
// (vkGetSemaphoreCounterValue) to make this correct regardless of pool size or
// channel count; there is currently no in-process access to the completion
// semaphore's value at the point acquire_descriptor_set() is called (frame_context
// only exposes an OS-exportable HANDLE, not the vk::Semaphore + value pair
// needed to wait in-process) without a larger cross-file change to that interface.
const int DescriptorPoolSize = 2048;
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
    vk::Sampler                    hueCurveSampler_;
    vk::DescriptorSetLayout        descriptorSetLayout_;
    /// Empty when this pipeline uses the SPIR-V built by glslc at configure time.
    std::vector<uint32_t>          frag_spirv_;
    vk::DescriptorPool             descriptorPool_;
    std::vector<vk::DescriptorSet> descriptorSets_;

    // ---- Descriptor set 1: a generated colour transform's resources ----------
    //
    // OCIO's Vulkan output declares its own resources with explicit
    // layout(set=N, binding=M), and its contract reserves binding 0 of that set for its
    // uniform buffer with textures from binding 1 upward
    // (GpuShaderDesc::setDescriptorSetIndex). The mixer's own set is full -- bindings 0..6
    // are all assigned -- so a generated transform gets a set of its own rather than
    // squeezing into this one.
    //
    // Declared and bound even when nothing uses it, so that the pipeline layout is the same
    // shape whether or not a variant is active. The alternative -- two layouts differing by
    // one set -- means the base and variant pipelines are not layout-compatible, and every
    // switch between them would need the descriptor sets rebound.
    vk::DescriptorSetLayout        ocioSetLayout_;
    std::vector<vk::DescriptorSet> ocioDescriptorSets_;
    /// OCIO's uniform buffer at set 1 binding 0, and it stays a zero-filled placeholder.
    ///
    /// getUniformBufferSize() answers 0 for every one of the 55 colour spaces in the pinned
    /// studio config: an input transform has no dynamic property, so OCIO declares no uniform
    /// block at all and the generated source never reads this. Written anyway, because a
    /// descriptor set may carry a binding the shader ignores but not one left unwritten, and
    /// because a display transform with a dynamic exposure -- A5 -- is what would fill it.
    /// 256 bytes is the Vulkan minimum guaranteed UBO range, so it fits anything that comes.
    vk::Buffer                     ocioUbo_;
    vk::DeviceMemory               ocioUboMemory_;
    static constexpr vk::DeviceSize OCIO_UBO_SIZE    = 256;

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

        // Binding 6: blend mask sampler2D
        vk::DescriptorSetLayoutBinding blendMaskLayoutBinding{};
        blendMaskLayoutBinding.binding         = 6;
        blendMaskLayoutBinding.descriptorType  = vk::DescriptorType::eCombinedImageSampler;
        blendMaskLayoutBinding.descriptorCount = 1;
        blendMaskLayoutBinding.stageFlags      = vk::ShaderStageFlagBits::eFragment;

        vk::DescriptorSetLayoutCreateInfo layoutInfo{};
        std::array bindings{texturesLayoutBinding, backgroundLayoutBinding, uboLayoutBinding,
                            lut3dLayoutBinding, hueCurveLayoutBinding, curveLutLayoutBinding,
                            blendMaskLayoutBinding};
        layoutInfo.setBindings(bindings);

        std::array<vk::DescriptorBindingFlags, 7> bindingFlags{
            vk::DescriptorBindingFlagBits::ePartiallyBound, // 0: textures
            vk::DescriptorBindingFlags{},                   // 1: background
            vk::DescriptorBindingFlags{},                   // 2: UBO
            vk::DescriptorBindingFlagBits::ePartiallyBound, // 3: lut3d
            vk::DescriptorBindingFlagBits::ePartiallyBound, // 4: hue curve
            vk::DescriptorBindingFlagBits::ePartiallyBound, // 5: curve lut
            vk::DescriptorBindingFlagBits::ePartiallyBound  // 6: blend mask
        };
        vk::DescriptorSetLayoutBindingFlagsCreateInfo bindingFlagsInfo;
        bindingFlagsInfo.setBindingFlags(bindingFlags);
        layoutInfo.pNext = &bindingFlagsInfo;

        descriptorSetLayout_ = device_.createDescriptorSetLayout(layoutInfo);

        // Set 1, for a generated colour transform. Binding 0 is OCIO's uniform buffer by its
        // own contract; bindings 1..OCIO_MAX_TEXTURES are its LUTs. Every sampler binding is
        // partially bound, because a transform uses as many as it needs and usually none:
        // camera log encodings generate no LUT at all, while ADX and display-referred spaces
        // generate one.
        {
            std::vector<vk::DescriptorSetLayoutBinding> ocioBindings;
            vk::DescriptorSetLayoutBinding             ocioUboBinding{};
            ocioUboBinding.binding         = 0;
            ocioUboBinding.descriptorType  = vk::DescriptorType::eUniformBuffer;
            ocioUboBinding.descriptorCount = 1;
            ocioUboBinding.stageFlags      = vk::ShaderStageFlagBits::eFragment;
            ocioBindings.push_back(ocioUboBinding);

            for (uint32_t i = 1; i <= OCIO_MAX_TEXTURES; ++i) {
                vk::DescriptorSetLayoutBinding b{};
                b.binding         = i;
                // eCombinedImageSampler covers both sampler2D and sampler3D -- the
                // descriptor type is the same, only the image view type differs, which
                // matters because OCIO emits either depending on the LUT.
                b.descriptorType  = vk::DescriptorType::eCombinedImageSampler;
                b.descriptorCount = 1;
                b.stageFlags      = vk::ShaderStageFlagBits::eFragment;
                ocioBindings.push_back(b);
            }

            std::vector<vk::DescriptorBindingFlags> ocioFlags(ocioBindings.size(),
                                                              vk::DescriptorBindingFlagBits::ePartiallyBound);
            ocioFlags[0] = vk::DescriptorBindingFlags{}; // the UBO is always written

            vk::DescriptorSetLayoutBindingFlagsCreateInfo ocioFlagsInfo;
            ocioFlagsInfo.setBindingFlags(ocioFlags);

            vk::DescriptorSetLayoutCreateInfo ocioLayoutInfo{};
            ocioLayoutInfo.setBindings(ocioBindings);
            ocioLayoutInfo.pNext = &ocioFlagsInfo;
            ocioSetLayout_       = device_.createDescriptorSetLayout(ocioLayoutInfo);
        }

        // Create descriptor pool
        vk::DescriptorPoolSize samplerPoolSize(vk::DescriptorType::eCombinedImageSampler,
                                               (BindlessTextureCount + 4) * DescriptorPoolSize);
        vk::DescriptorPoolSize inputAttachmentPoolSize(vk::DescriptorType::eInputAttachment,
                                                       1 * DescriptorPoolSize);
        vk::DescriptorPoolSize uboPoolSize(vk::DescriptorType::eUniformBuffer,
                                           1 * DescriptorPoolSize);

        // Capacity for set 1 too: one UBO and OCIO_MAX_TEXTURES samplers per slot.
        vk::DescriptorPoolSize ocioSamplerPoolSize(vk::DescriptorType::eCombinedImageSampler,
                                                   OCIO_MAX_TEXTURES * DescriptorPoolSize);
        vk::DescriptorPoolSize ocioUboPoolSize(vk::DescriptorType::eUniformBuffer,
                                                1 * DescriptorPoolSize);

        std::array poolSizes{samplerPoolSize, inputAttachmentPoolSize, uboPoolSize,
                             ocioSamplerPoolSize, ocioUboPoolSize};

        vk::DescriptorPoolCreateInfo poolInfo{};
        // Twice the sets: one per slot for the mixer's own bindings, one for set 1.
        poolInfo.maxSets = DescriptorPoolSize * 2;

        poolInfo.setPoolSizes(poolSizes);
        descriptorPool_ = device_.createDescriptorPool(poolInfo);

        // Allocate descriptor sets
        std::vector<vk::DescriptorSetLayout> layouts(DescriptorPoolSize, descriptorSetLayout_);
        vk::DescriptorSetAllocateInfo        allocInfo;
        allocInfo.descriptorPool = descriptorPool_;
        allocInfo.setSetLayouts(layouts);

        descriptorSets_ = device_.allocateDescriptorSets(allocInfo);

        // Set 1, one per slot so it can be rewritten per draw later without touching a set
        // still referenced by an in-flight command buffer -- the same reason the mixer's own
        // sets are a ring.
        std::vector<vk::DescriptorSetLayout> ocioLayouts(DescriptorPoolSize, ocioSetLayout_);
        vk::DescriptorSetAllocateInfo        ocioAllocInfo;
        ocioAllocInfo.descriptorPool = descriptorPool_;
        ocioAllocInfo.setSetLayouts(ocioLayouts);
        ocioDescriptorSets_ = device_.allocateDescriptorSets(ocioAllocInfo);

        // A placeholder UBO, written into every set-1 slot. The binding is not partially
        // bound, so a set bound with binding 0 unwritten is invalid -- and the validation
        // layers are entitled to reject the draw. Zero-filled and unread until A4e supplies
        // OCIO's real uniform data.
        {
            vk::BufferCreateInfo bufferInfo{};
            bufferInfo.size        = OCIO_UBO_SIZE;
            bufferInfo.usage       = vk::BufferUsageFlagBits::eUniformBuffer;
            bufferInfo.sharingMode = vk::SharingMode::eExclusive;
            ocioUbo_               = device_.createBuffer(bufferInfo);

            auto memReq = device_.getBufferMemoryRequirements(ocioUbo_);

            vk::MemoryAllocateInfo memAlloc{};
            memAlloc.allocationSize  = memReq.size;
            memAlloc.memoryTypeIndex = findMemoryType(
                memReq.memoryTypeBits,
                vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
            ocioUboMemory_ = device_.allocateMemory(memAlloc);
            device_.bindBufferMemory(ocioUbo_, ocioUboMemory_, 0);

            auto* mapped = device_.mapMemory(ocioUboMemory_, 0, OCIO_UBO_SIZE);
            std::memset(mapped, 0, static_cast<size_t>(OCIO_UBO_SIZE));
            device_.unmapMemory(ocioUboMemory_);

            std::vector<vk::DescriptorBufferInfo> bufferInfos(ocioDescriptorSets_.size());
            std::vector<vk::WriteDescriptorSet>   writes(ocioDescriptorSets_.size());
            for (size_t i = 0; i < ocioDescriptorSets_.size(); ++i) {
                bufferInfos[i] = vk::DescriptorBufferInfo(ocioUbo_, 0, OCIO_UBO_SIZE);
                writes[i]      = vk::WriteDescriptorSet(ocioDescriptorSets_[i], 0, 0, 1,
                                                        vk::DescriptorType::eUniformBuffer,
                                                        nullptr, &bufferInfos[i]);
            }
            device_.updateDescriptorSets(writes, nullptr);
        }
    }

    void setup_sampler()
    {
        vk::SamplerCreateInfo samplerInfo{};

        samplerInfo.magFilter  = vk::Filter::eLinear;
        samplerInfo.minFilter  = vk::Filter::eLinear;
        samplerInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;
        // Clamp, matching the OpenGL backend, where every texture is created
        // with GL_CLAMP_TO_EDGE (ogl/util/texture.cpp) and only the hue curve
        // departs from it.
        //
        // This was eRepeat on all three axes, which meant a bilinear tap at any
        // texture boundary pulled in the opposite edge instead of holding the
        // last texel. The interiors matched exactly, so it only showed at the
        // frame's outermost pixel column -- 0.1 % of the picture, and invisible
        // in a PSNR figure, but it is the column that has to line up when a
        // channel drives one segment of a video wall. borderColor below was
        // already set to opaque black, which only has meaning under
        // eClampToBorder: clamping was intended here and the address mode was
        // never brought into line with it.
        samplerInfo.addressModeU            = vk::SamplerAddressMode::eClampToEdge;
        samplerInfo.addressModeV            = vk::SamplerAddressMode::eClampToEdge;
        samplerInfo.addressModeW            = vk::SamplerAddressMode::eClampToEdge;
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

        // The hue curve is the one texture that must wrap, and only across S:
        // it is indexed by hue, so the far end of the curve is adjacent to the
        // near end and a red-region adjustment has to carry across the seam.
        // OpenGL sets exactly this -- GL_REPEAT on S, GL_CLAMP_TO_EDGE on T
        // (ogl/image/image_kernel.cpp). Sharing one repeating sampler with
        // everything else got this right by accident and everything else wrong.
        samplerInfo.magFilter    = vk::Filter::eLinear;
        samplerInfo.minFilter    = vk::Filter::eLinear;
        samplerInfo.mipmapMode   = vk::SamplerMipmapMode::eLinear;
        samplerInfo.addressModeU = vk::SamplerAddressMode::eRepeat;
        samplerInfo.addressModeV = vk::SamplerAddressMode::eClampToEdge;
        samplerInfo.addressModeW = vk::SamplerAddressMode::eClampToEdge;
        hueCurveSampler_         = device_.createSampler(samplerInfo);
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

    impl(vk::Device                         device,
         vk::Format                         format,
         vk::PhysicalDeviceMemoryProperties memProperties,
         std::vector<uint32_t>              frag_spirv)
        : device_(device)
        , format_(format)
        , memProperties_(memProperties)
        , frag_spirv_(std::move(frag_spirv))
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
        // Two sets: the mixer's own, then a generated colour transform's. Declared
        // unconditionally so base and variant pipelines share one layout and stay
        // layout-compatible -- see the ocioSetLayout_ member comment.
        std::array setLayouts{descriptorSetLayout_, ocioSetLayout_};
        vk::PipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.setSetLayouts(setLayouts);

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

        auto shaderStages = std::move(create_shader_program(device_, frag_spirv_));
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

    /// Returns the acquired set together with its ring slot.
    ///
    /// The slot matters because descriptor set 1 is allocated one-per-slot in the same ring:
    /// a draw has to bind set 1 from the SAME slot, or a later draw rewriting set 1 would
    /// disturb one still referenced by an in-flight command buffer. That became observable
    /// the moment a generated transform's LUTs started being written here, which is why it
    /// was worth wiring correctly while every slot still held the identical placeholder.
    std::pair<vk::DescriptorSet, size_t> acquire_descriptor_set(const uniform_block& params,
                                              const std::array<vk::ImageView, 11>& textures,
                                              const ocio_texture_views&            ocio_textures)
    {
        // C++ textures array layout:
        //   [0] = background attachment, [1..4] = planes, [5] = local_key, [6] = layer_key
        //   [7] = lut3d, [8] = hue_curve, [9] = curve_lut, [10] = blend_mask

        // Shader bindless textures[N] layout:
        //   [0..3] = planes, [4] = local_key, [5] = layer_key

        auto  setIndex        = currentDescriptorSet_;
        auto  descriptorSet   = descriptorSets_[setIndex];
        currentDescriptorSet_ = (currentDescriptorSet_ + 1) % DescriptorPoolSize;

        // Copy UBO data to the ring buffer slot
        std::memcpy(uboMapped_ + setIndex * UBO_SLOT_SIZE, &params, sizeof(uniform_block));

        // Bind planes, local_key, and layer_key to the bindless texture array.
        // Not every pixel format populates all 4 plane slots, and local_key/
        // layer_key are frequently absent — write only the slots that actually
        // have an image view. Writing a null VkImageView here is only valid
        // with VK_EXT_robustness2's nullDescriptor, which is merely opportunistic
        // (enable_extension_features_if_present) rather than guaranteed; relying
        // on ePartiallyBound (already set on this binding, see the LUT/hue-curve/
        // blend-mask bindings below which already skip absent entries the same way)
        // works on every device instead.
        std::array<vk::DescriptorImageInfo, 6> textureInfos;
        std::vector<vk::WriteDescriptorSet>    texture_writes;
        for (int i = 0; i < 6; ++i) {
            if (!textures[i + 1])
                continue;
            textureInfos[i].sampler     = (i == 4 || i == 5) ? keySampler_ : textureSampler_;
            textureInfos[i].imageView   = textures[i + 1];
            textureInfos[i].imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;

            vk::WriteDescriptorSet write{};
            write.dstSet          = descriptorSet;
            write.dstBinding      = 0;
            write.dstArrayElement = static_cast<uint32_t>(i);
            write.descriptorType  = vk::DescriptorType::eCombinedImageSampler;
            write.setImageInfo(textureInfos[i]);
            texture_writes.push_back(write);
        }

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
        std::vector<vk::WriteDescriptorSet> writes{backgroundWrite, uboWrite};
        writes.insert(writes.end(), texture_writes.begin(), texture_writes.end());

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
            // Wraps across S -- the only texture here that should. See setup_sampler.
            hueCurveInfo.sampler     = hueCurveSampler_;
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

        // Binding 6: blend mask (if present)
        vk::DescriptorImageInfo blendMaskInfo{};
        if (textures[10]) {
            blendMaskInfo.sampler     = textureSampler_;
            blendMaskInfo.imageView   = textures[10];
            blendMaskInfo.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;

            vk::WriteDescriptorSet blendMaskWrite{};
            blendMaskWrite.dstSet          = descriptorSet;
            blendMaskWrite.dstBinding      = 6;
            blendMaskWrite.dstArrayElement = 0;
            blendMaskWrite.descriptorType  = vk::DescriptorType::eCombinedImageSampler;
            blendMaskWrite.setImageInfo(blendMaskInfo);
            writes.push_back(blendMaskWrite);
        }

        // Descriptor set 1: a generated transform's LUTs, at the bindings OCIO declared.
        //
        // Written into this ring slot's set 1, the same slot the mixer's own set came from,
        // so an in-flight command buffer's set is never rewritten underneath it. Slot i of
        // ocio_textures is binding i+1; a null slot is a binding the transform did not
        // declare, left unwritten under ePartiallyBound rather than written null.
        // ocioInfos must outlive updateDescriptorSets: setImageInfo stores a pointer into it,
        // not a copy.
        std::array<vk::DescriptorImageInfo, OCIO_MAX_TEXTURES> ocioInfos;
        for (uint32_t i = 0; i < OCIO_MAX_TEXTURES; ++i) {
            if (!ocio_textures[i])
                continue;
            // OCIO's own LUTs are indexed by a computed coordinate and must clamp, never
            // wrap: textureSampler_, not hueCurveSampler_. A wrap here folds the top of a
            // 1D LUT onto its bottom and shows up only in the extreme highlights.
            ocioInfos[i].sampler     = textureSampler_;
            ocioInfos[i].imageView   = ocio_textures[i];
            ocioInfos[i].imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;

            vk::WriteDescriptorSet w{};
            w.dstSet          = ocioDescriptorSets_[setIndex];
            w.dstBinding      = i + 1;
            w.dstArrayElement = 0;
            w.descriptorType  = vk::DescriptorType::eCombinedImageSampler;
            w.setImageInfo(ocioInfos[i]);
            writes.push_back(w);
        }

        device_.updateDescriptorSets(writes, nullptr);

        return {descriptorSet, setIndex};
    }

    void draw(vk::CommandBuffer                    commandBuffer,
              vk::Buffer                           vertexBuffer,
              uint32_t                             coords_count,
              uint32_t                             vertex_buffer_offset,
              const uniform_block&                 params,
              const std::array<vk::ImageView, 11>& textures,
              const ocio_texture_views&            ocio_textures)
    {
        auto [descriptorSet, setIndex] = acquire_descriptor_set(params, textures, ocio_textures);
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline_);
        commandBuffer.bindVertexBuffers(0, vertexBuffer, {vertex_buffer_offset});
        // Both sets, always. Vulkan only requires binding what the pipeline statically uses,
        // but binding set 1 unconditionally means a switch to a variant pipeline needs no
        // extra bookkeeping, and an unbound-but-declared set is the sort of thing the
        // validation layers flag inconsistently across drivers.
        std::array boundSets{descriptorSet, ocioDescriptorSets_[setIndex]};
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout_, 0, boundSets, nullptr);
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
        device_.destroyDescriptorSetLayout(ocioSetLayout_);
        device_.destroyBuffer(ocioUbo_);
        device_.freeMemory(ocioUboMemory_);
        device_.destroySampler(textureSampler_);
        device_.destroySampler(keySampler_);
        device_.destroySampler(hueCurveSampler_);

        device_.destroyPipeline(pipeline_);
        device_.destroyPipelineLayout(pipelineLayout_);
    }
};

pipeline::pipeline(vk::Device                         device,
                   vk::Format                         format,
                   vk::PhysicalDeviceMemoryProperties memProperties,
                   const std::vector<uint32_t>&        frag_spirv)
    : impl_(new impl(device, format, memProperties, frag_spirv))
{
}
pipeline::~pipeline() {}

void pipeline::draw(vk::CommandBuffer                    commandBuffer,
                    vk::Buffer                           vertexBuffer,
                    uint32_t                             coords_count,
                    uint32_t                             vertex_buffer_offset,
                    const uniform_block&                 params,
                    const std::array<vk::ImageView, 11>& textures,
                    const ocio_texture_views&            ocio_textures)
{
    impl_->draw(commandBuffer, vertexBuffer, coords_count, vertex_buffer_offset, params, textures,
                ocio_textures);
}

vk::Pipeline pipeline::id() const { return impl_->pipeline_; }

}}} // namespace caspar::accelerator::vulkan
