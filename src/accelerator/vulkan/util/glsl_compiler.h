/*
 * Copyright (c) 2011 Sveriges Television AB <info@casparcg.com>
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

#include <cstdint>
#include <string>
#include <vector>

namespace caspar { namespace accelerator { namespace vulkan {

/// Compile Vulkan-flavoured GLSL to SPIR-V at runtime.
///
/// The mixer's own fragment shader is compiled at build time by glslc and embedded, and
/// that stays true. This exists because a generated colour transform cannot be: OCIO emits
/// GLSL *text* (GPU_LANGUAGE_GLSL_VK_4_6 for Vulkan), so the program has to be built after
/// the transform is known, which means at runtime.
///
/// Cost: a compile is tens to hundreds of milliseconds. Never call this from the frame
/// path. Compile when the configuration changes and keep serving the previous pipeline
/// until the new module is ready.
///
/// Returns an empty vector on failure, having logged the compiler's diagnostic. Callers
/// must treat that as "keep using what you had" rather than proceeding with no shader --
/// a colour transform that fails to compile is a reason to refuse the command, not to put
/// a black frame on air.
///
/// `name` appears in diagnostics only; pass something identifying the variant.
std::vector<uint32_t> compile_glsl_fragment_to_spirv(const std::string& source, const std::string& name);

}}} // namespace caspar::accelerator::vulkan
