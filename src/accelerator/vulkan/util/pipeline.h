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
#include <vulkan/vulkan.hpp>
namespace caspar { namespace accelerator { namespace vulkan {

#include <vector>

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

    void         draw(vk::CommandBuffer                    commandBuffer,
                      vk::Buffer                           vertexBuffer,
                      uint32_t                             coords_count,
                      uint32_t                             vertex_buffer_offset,
                      const uniform_block&                 params,
                      const std::array<vk::ImageView, 11>& textures);
    vk::Pipeline id() const;

  private:
    struct impl;
    std::unique_ptr<impl> impl_;
};

}}} // namespace caspar::accelerator::vulkan
