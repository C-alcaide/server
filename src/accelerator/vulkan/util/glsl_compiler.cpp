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

#include "glsl_compiler.h"

#include <common/log.h>
#include <common/utf.h>

#include <shaderc/shaderc.hpp>

namespace caspar { namespace accelerator { namespace vulkan {

std::vector<uint32_t> compile_glsl_fragment_to_spirv(const std::string& source, const std::string& name)
{
    if (source.empty()) {
        CASPAR_LOG(warning) << L"[glsl_compiler] empty source for '" << u16(name) << L"'";
        return {};
    }

    // Constructed per call rather than kept as a static. A compile happens on a
    // configuration change, not per frame, so the setup cost is irrelevant -- and a shared
    // compiler would be one more thing needing a lock, for no gain.
    shaderc::Compiler       compiler;
    shaderc::CompileOptions options;

    // Must match the pipeline these modules are used by. Vulkan 1.3 is what the device is
    // created against, and it is also what OCIO's GPU_LANGUAGE_GLSL_VK_4_6 output assumes
    // -- descriptor sets and explicit binding indices rather than GL-style uniforms.
    options.SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_3);
    options.SetSourceLanguage(shaderc_source_language_glsl);
    // Optimisation level: measured, and left at `performance`.
    //
    // Level zero looked like an easy win for the frame-path stall and the data did not
    // support it. Both produce identical pixels (100/100 conversions within 1 LSB either
    // way, verified on the Vulkan mixer with the runtime path forced), but the totals go the
    // other way:
    //
    //                        shaderc      SPIR-V        driver -> 3 pipelines
    //     performance         1.16 s      80194 words   0.22 s
    //     zero                0.475 s     39879 words   1.51 s
    //
    // Level zero halves the shaderc time and halves the module, and the driver then spends
    // ~7x longer turning it into ISA -- the optimiser front-loads work the driver would
    // otherwise repeat, and its larger module is larger because it INLINED, producing more
    // but simpler instructions. Both halves are paid together when a variant compiles, so
    // the total is what matters and `performance` wins it.
    //
    // Single samples under differing machine load, so treat the pipeline-creation figures as
    // indicative rather than settled. Recorded because the intuition here is wrong: a
    // cheaper compile is not automatically a shorter stall.
    //
    // The real levers, if the stall needs reducing, are caching rather than tuning: a SPIR-V
    // cache keyed on the source hash so a restart pays nothing, and a serialised
    // VkPipelineCache for the driver half -- which these numbers show is comparable to or
    // larger than shaderc's share.
    options.SetOptimizationLevel(shaderc_optimization_level_performance);

    const auto result = compiler.CompileGlslToSpv(source, shaderc_glsl_fragment_shader, name.c_str(), options);

    if (result.GetCompilationStatus() != shaderc_compilation_status_success) {
        // The diagnostic is the whole value of this branch: a generated shader fails for
        // reasons that are not visible in the C++ that assembled it, so log what the
        // compiler actually said rather than a summary of it.
        CASPAR_LOG(error) << L"[glsl_compiler] '" << u16(name) << L"' failed to compile: "
                          << u16(result.GetErrorMessage());
        return {};
    }

    if (result.GetNumWarnings() > 0) {
        CASPAR_LOG(warning) << L"[glsl_compiler] '" << u16(name) << L"' compiled with warnings: "
                            << u16(result.GetErrorMessage());
    }

    return std::vector<uint32_t>(result.cbegin(), result.cend());
}

}}} // namespace caspar::accelerator::vulkan
