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
#include <array>

#include <core/graph/node_draw.h>

#include <string>
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

    /// This draw IS one grading node's pass. Mirror of the OpenGL flag of the same name;
    /// see accelerator/ogl/image/image_kernel.h for the full account.
    ///
    /// On this backend the source has to arrive in `textures` and be sampled as an ordinary
    /// sampler2D: the fragment shader reads the composite through `subpassInput background`,
    /// which must be an attachment of the same render pass read at the same pixel, and a
    /// node pass reads its predecessor instead. `apply_output_convert` and
    /// `apply_calibration_lut` already work this way, which is why this fits without a new
    /// pass structure.
    /// This draw IS one node's pass.
    ///
    /// Deliberately the same shape as `output_convert_only` above, which is the precedent this
    /// follows rather than invents: a full-screen draw through the ordinary kernel with the
    /// source in `textures` and the destination in `background`, tagged so that the
    /// colour-conversion halves do not run.
    ///
    /// It routes the shader past everything the layer pass already did -- both conversion
    /// halves, the whole primary grading chain, alpha handling, keying, blending, chroma, grain
    /// and the projection blend mask -- and runs this node's mask and operation only.
    /// Double-applying any of those is the failure mode this exists to prevent.
    ///
    /// ONE FIELD, not a flag plus a struct: `node.op >= 0` IS the flag, so the two cannot
    /// disagree. The prototype carried `bool grade_node_only` beside a `core::grade_node`, and
    /// a state where one said yes and the other was default was expressible.
    core::graph::node_draw node{};

    /// This node is an `isf` node, and this is the shader file its `path` parameter names.
    ///
    /// A POINTER INTO THE PLAN, which is immutable and shared, so nothing is copied per draw.
    /// Empty or null for every other class.
    ///
    /// The KERNEL turns this into a pipeline, not the mixer, and that is deliberate: the kernel
    /// already owns the OCIO variant cache and the device handle that builds one, so putting the
    /// ISF cache anywhere else would mean two caches and two answers to "have I compiled this".
    const std::string* isf_path = nullptr;

    /// The channel's clock for this node, feeding ISF's `TIME`, `TIMEDELTA` and `FRAMEINDEX`.
    /// From `core::image_mixer::set_frame_number` -- the channel's own counter, so two ISF nodes
    /// on a channel agree and a shader stays in step with the timeline.
    double isf_time       = 0.0;
    double isf_time_delta = 0.0;
    int    isf_frame      = 0;
    /// The node's `space` resolved against the graph's stage: +1, -1 or 0. See
    /// `core::graph::isf_node_request::to_display`.
    int    isf_to_display = 0;

    /// This LAYER draw feeds a node graph running in **working** space.
    ///
    /// The shader stops at the working-space boundary -- before tone-map, the gamut matrix and
    /// the OETF -- and the TAIL pass applies that half once, against the real target. Distinct
    /// from `node` above: this is set on the ordinary layer draw, not on a node pass.
    bool graph_head = false;

    /// This draw IS that tail: apply the output half **only**, using the configuration the
    /// head's own draw would have used.
    ///
    /// NOT the same as `output_convert_only`, and the difference is the whole reason the first
    /// attempt at this was reverted. `output_convert_only` forces the output half ON using the
    /// CHANNEL's target values; a layer under `MIXER COLORSPACE` has its own output transfer and
    /// leaves the gamut matrix in the input half, and a layer converting nothing must have its
    /// tail convert nothing. So this flag lets the kernel select its branch exactly as it would
    /// for the head, and then forces the INPUT half off.
    ///
    /// The grading chain does not run on a tail pass because the tail's transform is a DEFAULT
    /// `image_transform` carrying only `color_grade` -- every operator's enable is at its
    /// default, so there is nothing to double-apply.
    bool graph_tail = false;

    /// FRAME uv -> the ITEM's own uv, as three ROWS of a matrix the shader multiplies
    /// `vec3(uv, 1)` by. Set on a node pass so a `mask_ellipse` declaring `space = source`
    /// follows the layer's own geometry instead of the raster.
    ///
    /// Identity by default, and `node_uv_valid` is what gates it rather than comparing against
    /// the identity: a layer at its default fill HAS the identity here, and a mask must not
    /// behave differently depending on whether the matrix happened to be interesting.
    std::array<float, 9> node_uv_inv{1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f};

    /// False when the placement could not be inverted -- a corner-pin `perspective`, which is
    /// not a 3x3 by construction, or a singular scale. The pass then masks in FRAME space, which
    /// is a visible, explicable answer; an approximated corner pin would put the mask somewhere
    /// plausible and wrong, and nothing downstream could tell.
    bool node_uv_valid = false;

    /// This draw's destination is an **fp16** attachment.
    ///
    /// Vulkan-only, and it exists because a pipeline carries its colour-attachment format in
    /// its own creation info: writing fp16 through a unorm pipeline is a format MISMATCH rather
    /// than a conversion. The kernel reads this and hands back the matching pipeline through
    /// the existing per-layer hook, which OCIO already uses for the same reason.
    ///
    /// OpenGL needs no equivalent: a GL program does not carry its target's format.
    bool node_fp16 = false;
};

}}} // namespace caspar::accelerator::vulkan
