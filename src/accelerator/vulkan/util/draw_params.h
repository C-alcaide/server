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

#include "transforms.h"
#include <common/memory.h>
#include <core/frame/frame_transform.h>
#include <core/frame/geometry.h>
#include <core/frame/pixel_format.h>
#include <vector>

namespace caspar { namespace accelerator { namespace vulkan {

enum class keyer
{
    linear = 0,
    additive,
};

struct draw_params final
{
    core::pixel_format_desc                     pix_desc = core::pixel_format_desc(core::pixel_format::invalid);
    std::vector<spl::shared_ptr<class texture>> textures;
    draw_transforms                             transforms;
    core::frame_geometry                        geometry   = core::frame_geometry::get_default();
    core::blend_mode                            blend_mode = core::blend_mode::normal;
    vulkan::keyer                               keyer      = vulkan::keyer::linear;
    std::shared_ptr<class texture>              background;
    std::shared_ptr<class texture>              local_key;
    std::shared_ptr<class texture>              layer_key;
    double                                      aspect_ratio = 1.0;
    int                                         target_width;
    int                                         target_height;
    core::color_space                           target_color_space    = core::color_space::bt709;
    core::color_transfer                        target_color_transfer = core::color_transfer::sdr;
    bool                                        auto_color_convert    = true;
    int                                         auto_tone_map         = 0;
    float                                       display_peak_luminance = 1000.0f;
    float                                       sdr_reference_white    = 100.0f;
    bool                                        auto_gamut_compress    = false;

    /// Run the colour chain on straight (unpremultiplied) RGB, as OCIO documents, rather
    /// than on premultiplied RGB. Channel-level and off by default: it changes rendered
    /// output wherever content has soft edges and any non-linear transform is configured.
    /// Measured account in CasparCG-TestRunner/docs/alpha_domain_2026-08-12.md.
    bool                                        straight_alpha_grading = false;

    /// Composite in the WORKING space (scene-linear ACEScg) rather than in display space.
    ///
    /// Set on every LAYER draw of a channel configured for it. Two effects in image_kernel,
    /// and they are the whole feature: every layer's input half is forced through the
    /// ACEScg route so the layers agree on a space, and every layer's OUTPUT half is
    /// suppressed -- the channel converts once, post-composite, via `output_convert_only`.
    ///
    /// Blend modes then operate on scene-linear values instead of 0-1 display values, which
    /// is exactly what the comment beside the output block warns about. That is the point
    /// rather than a side effect, and it is why the channel element is opt-in.
    ///
    /// Requires fp16 (ACEScg carries values above 1.0 and below 0) and auto-color-convert
    /// (every layer needs a defined route INTO the working space). server.cpp refuses the
    /// config otherwise.
    bool                                        working_space_composite = false;

    /// This draw IS the channel's post-composite output conversion.
    ///
    /// Input half off, output half on, driven by the channel's target -- and luminance_scale
    /// 1.0, because each layer's input half already moved the pixel into the target's
    /// luminance domain. That is exactly the configuration the OCIO input-transform branch
    /// sets up, so the kernel reuses that branch rather than growing a fourth.
    bool                                        output_convert_only = false;

    /// Channel-level OCIO display transform: the composited look. Empty means none.
    ///
    /// Channel-level rather than per layer, and stamped onto every draw by the mixer, because
    /// it describes the output the whole channel is being graded FOR. Applied where the
    /// built-in output conversion runs -- per layer, before the blend -- so that foreground
    /// and background reach the blend in the same display encoding, exactly as today. With
    /// one transform for the whole channel that is equivalent to transforming the composite.
    ///
    /// A consumer-level view needs the composite to still be in working space, which was the
    /// larger change this comment used to defer: `<working-space-composite>` plus one
    /// post-composite pass per distinct view. Both shipped -- see OCIO_USER_GUIDE.md §6.2.
    std::string                                 ocio_display;
    std::string                                 ocio_view;
    /// An LMT applied in the working space BEFORE the display rendering. Composed into
    /// the display processor rather than spliced separately, so it changes the OUTPUT
    /// half's cache id and the variant key stays the (input, output) pair it already is.
    /// Empty means none, and generates exactly what it generated before looks existed.
    std::string                                 ocio_look;
};

}}} // namespace caspar::accelerator::vulkan
