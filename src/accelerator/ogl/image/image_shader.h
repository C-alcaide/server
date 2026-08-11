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
 *
 * Author: Robert Nagy, ronag89@gmail.com
 */

#pragma once

#include <common/memory.h>

namespace caspar { namespace accelerator { namespace ogl {

class shader;
class device;

enum class texture_id
{
    plane0 = 0,
    plane1,
    plane2,
    plane3,
    local_key,
    layer_key,
    background,
    curve_lut_tex,
    lut3d_tex,
    hue_curve_tex,
    blend_mask_tex
};

/// Identifies one compiled variant of the mixer's fragment program.
///
/// The mixer used to have exactly one program for the whole process, because every colour
/// behaviour was expressible as a uniform. A generated colour transform is not: it arrives
/// as GLSL source, so the program itself becomes a function of the transform and there has
/// to be more than one.
///
/// An empty `id` means the base program -- the shader as embedded at build time, with no
/// generated code spliced in. That is the only variant anything asks for today.
struct shader_variant
{
    /// Cache key. Empty for the base program. A generated transform supplies its own
    /// stable identifier here (OCIO exposes one on the processor) rather than a hash of
    /// the source text, so the key survives cosmetic differences in code generation.
    std::string id;

    /// Declarations and functions to insert ahead of the shader's own body.
    std::string prologue;

    /// The expression spliced in at the transform point, e.g. a call into `prologue`.
    /// Empty leaves the base program's own conversion in place.
    std::string transform_call;

    bool is_base() const { return id.empty(); }
};

/// The compiled program for `variant`, compiling it on first request.
///
/// Compilation happens on the calling thread, so callers must not invoke this from the
/// frame path: a program build is tens to hundreds of milliseconds and would drop frames.
/// Ask for a variant when the configuration changes, hold the returned pointer, and keep
/// using the previous one until the new pointer is in hand.
///
/// The cache holds a bounded number of recently-requested programs alive so that a
/// configuration flipping between two transforms does not recompile on every switch;
/// beyond that bound the least-recently-requested entry is released, and a program with no
/// remaining users is destroyed on the GL thread.
std::shared_ptr<shader> get_image_shader(const spl::shared_ptr<device>& ogl,
                                         const shader_variant&         variant = shader_variant{});

/// Number of programs the cache is currently keeping alive. Diagnostics and tests only.
size_t image_shader_cache_size();

}}} // namespace caspar::accelerator::ogl
