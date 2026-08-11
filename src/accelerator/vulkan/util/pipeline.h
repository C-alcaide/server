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

#include "uniform_block.h"
#include <array>
#include <cstdint>
#include <vector>
#include <vulkan/vulkan.hpp>

namespace caspar { namespace accelerator { namespace vulkan {

/// LUT bindings reserved in descriptor set 1 for a generated colour transform. OCIO keeps
/// binding 0 of that set for its uniform buffer and declares its textures from binding 1
/// upward, so this is the highest binding the set layout provides.
///
/// Eight is generous on purpose: across all 55 colour spaces in the pinned studio config the
/// most any input transform emits is one. Display transforms, which A5 adds, are the case
/// that could need more.
inline constexpr uint32_t OCIO_MAX_TEXTURES = 8;

/// The image views a generated transform's samplers read, indexed so that slot i is
/// binding i+1 -- the binding OCIO wrote into the shader source, not one chosen here. A null
/// slot is a binding the transform did not declare and is simply not written; every sampler
/// binding in the set layout is ePartiallyBound for exactly that reason.
using ocio_texture_views = std::array<vk::ImageView, OCIO_MAX_TEXTURES>;

/// Which of those bindings must be point-sampled, same indexing.
///
/// OCIO says per texture whether it wants INTERP_NEAREST, and for an ACES display transform
/// two of its three tables do -- interpolating between their entries is wrong, not soft.
/// Every input-transform LUT was linear, so binding one sampler to all of them was correct
/// by accident until display transforms arrived.
using ocio_texture_filters = std::array<bool, OCIO_MAX_TEXTURES>;

class pipeline final
{
    pipeline(const pipeline&);
    pipeline& operator=(const pipeline&);

  public:
    /// `frag_spirv` overrides the fragment stage. Empty means use the SPIR-V glslc compiled
    /// at build time, which is the normal path and byte-identical to before this parameter
    /// existed.
    ///
    /// A non-empty vector is how a generated colour transform gets in: OCIO emits GLSL text,
    /// which is spliced into this shader's source and compiled at runtime. Passing the
    /// result here rather than reaching into the pipeline keeps SPIR-V production and
    /// pipeline construction separable -- and separately testable, which is why the runtime
    /// path can be exercised with the BASE shader before any OCIO code exists.
    pipeline(vk::Device                          device,
             vk::Format                          format,
             vk::PhysicalDeviceMemoryProperties  memProperties,
             const std::vector<uint32_t>&        frag_spirv = {});
    ~pipeline();

    /// `ocio_textures` fills descriptor set 1. Defaulted to all-null, which writes nothing
    /// there and leaves the set holding only its uniform buffer -- the state every draw was
    /// in before a generated transform existed.
    void         draw(vk::CommandBuffer                    commandBuffer,
                      vk::Buffer                           vertexBuffer,
                      uint32_t                             coords_count,
                      uint32_t                             vertex_buffer_offset,
                      const uniform_block&                 params,
                      const std::array<vk::ImageView, 11>& textures,
                      const ocio_texture_views&            ocio_textures = {},
                      const ocio_texture_filters&          ocio_nearest  = {});
    vk::Pipeline id() const;

  private:
    struct impl;
    std::unique_ptr<impl> impl_;
};

}}} // namespace caspar::accelerator::vulkan
