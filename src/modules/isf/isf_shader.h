/*
 * Copyright (c) 2026 CasparCG Contributors
 *
 * This file is part of CasparCG (www.casparcg.com).
 *
 * CasparCG is free software: you can redistribute it and/or modify it under the terms of the GNU
 * General Public License as published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 */

#pragma once

#include <common/bit_depth.h>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace caspar { namespace core { class texture; } }
namespace caspar { namespace accelerator { namespace ogl { class device; } } }
namespace caspar { namespace isf { class gl_context; } }

namespace caspar { namespace isf {

/// One declared ISF input (from the shader's JSON header).
struct input
{
    std::string         name;
    std::string         type;          ///< float | bool | event | long | color | point2D | image
    std::vector<double> default_value; ///< 1 value (float/bool/long) or 2/4 (point2D/color)
    std::vector<double> min_value;
    std::vector<double> max_value;
    std::string         label;
    bool                is_image = false;

    // "long" pop-up menu (optional).
    std::vector<long>        values;
    std::vector<std::string> labels;
};

/// ISF shader role, inferred from the declared image inputs (per the ISF conventions).
enum class shader_role
{
    generator,  ///< no image inputs
    filter,     ///< an image input named "inputImage"
    transition, ///< image inputs "startImage" + "endImage" + a float "progress"
};

/// One image bound for a render: either an existing GL texture (zero-copy) or a CPU RGBA buffer
/// (bottom-up, tightly packed width*height*4) that the shader uploads on the GL thread.
struct image_binding
{
    std::string          name;             ///< ISF image input name (e.g. "inputImage")
    unsigned int         tex_id = 0;       ///< non-zero: sample this GL texture directly
    const unsigned char* rgba   = nullptr; ///< else: upload this bottom-up RGBA buffer
    int                  width  = 0;
    int                  height = 0;
    bool                 flip   = false;   ///< sample vertically flipped (for top-down GL textures)
    bool                 bgra   = false;   ///< apply the mixer's .bgra swizzle (bgra-labelled texture)
};

/// A compiled ISF (Interactive Shader Format) shader that renders on CasparCG's OpenGL device.
///
/// Supports the ISF v2 single- and multi-pass subset: standard uniforms, float/bool/event/long/
/// color/point2D inputs, multiple image inputs, IMPORTED images, PASSES with persistent/float
/// buffers, and optional custom vertex shaders. Renders into a mixer-owned texture (zero-copy)
/// that is Y-flipped to the mixer's top-down, BGRA-convention orientation.
class shader
{
  public:
    /// base_path = directory of the shader file (for IMPORTED relative paths).
    /// vertex_source = optional custom ISF vertex shader (.vs) body; empty uses a generated one.
    explicit shader(const std::string& source, const std::wstring& base_path = {}, const std::string& vertex_source = {});
    ~shader();

    shader(const shader&)            = delete;
    shader& operator=(const shader&) = delete;

    const std::vector<input>& inputs() const;
    const std::string&        description() const;
    shader_role               role() const;

    /// Names of declared image inputs (order = declaration order).
    std::vector<std::string> image_input_names() const;

    /// Set an input value by name (1..4 scalars). Returns false if the input is unknown.
    bool set_value(const std::string& name, const std::vector<double>& values);

    /// Reset all `event`-type inputs to 0 (call once per rendered frame for momentary triggers).
    void reset_events();

    /// Bits per component for the FINAL PASS TARGET and every output route.
    ///
    /// Not merely the output texture: the final pass renders into `ensure_final`'s
    /// buffer, so a 16-bit output with an 8-bit final pass would blit an
    /// already-quantised result and deliver 256 levels -- which is exactly what an ISF
    /// ramp measured before this existed, through a Spout sender correctly advertising
    /// rgba16. Intermediate PASSES buffers keep their own ISF `FLOAT` attribute, which
    /// is a different question.
    ///
    /// Defaults to 8, so a caller that does not ask is byte-identical to before.
    void set_output_depth(common::bit_depth depth);

    /// Render one frame on the device's GL thread into a texture (top-down, BGRA-labelled).
    /// images binds declared image inputs by name. time/time_delta/frame_index feed the standard
    /// ISF uniforms. Returns nullptr on compile/render failure.
    std::shared_ptr<core::texture> render(const std::shared_ptr<accelerator::ogl::device>& device,
                                          int                                               width,
                                          int                                               height,
                                          double                                            time,
                                          double                                            time_delta,
                                          int                                               frame_index,
                                          const std::vector<image_binding>&                 images = {});

    /// Render on a self-contained GL context and read the result back into a tightly-packed,
    /// top-down BGRA CPU buffer (used when the active mixer is not the OpenGL mixer). Returns false
    /// on failure.
    bool render_readback(gl_context&                       ctx,
                         int                               width,
                         int                               height,
                         double                            time,
                         double                            time_delta,
                         int                               frame_index,
                         const std::vector<image_binding>& images,
                         /// Written directly, top-down BGRA. This is the frame's own
                         /// mapped memory: going via a vector and copying afterwards
                         /// meant a second full-frame copy every frame.
                         unsigned char*                    dst,
                         int                               dst_stride);

    /// Render on a self-contained GL context straight into `dst_gl_texture` -- a GL texture whose
    /// storage is a Vulkan image's memory (see accelerator/vulkan/util/gl_export_bridge.h), so the
    /// Vulkan mixer samples what this writes with no host round trip.
    ///
    /// The result is top-down and BGRA-ordered: byte-for-byte what render_readback puts in a CPU
    /// frame, which is what keeps the two paths interchangeable. Ends with glFinish(), so the
    /// pixels are complete for Vulkan when this returns. Returns false on failure.
    bool render_into_shared(gl_context&                       ctx,
                            int                               width,
                            int                               height,
                            double                            time,
                            double                            time_delta,
                            int                               frame_index,
                            const std::vector<image_binding>& images,
                            unsigned int                      dst_gl_texture);

  private:
    struct impl;
    std::unique_ptr<impl> impl_;
};

}} // namespace caspar::isf
