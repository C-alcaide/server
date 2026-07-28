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
                         std::vector<unsigned char>&       out_bgra_top_down);

  private:
    struct impl;
    std::unique_ptr<impl> impl_;
};

}} // namespace caspar::isf
