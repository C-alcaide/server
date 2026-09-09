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
/// Which kind of AUDIO texture an input wants, if it wants one.
///
/// ISF's `audio` and `audioFFT` are declared as INPUTS and consumed as sampler2D -- they are
/// image inputs whose pixels are audio rather than picture. So they take the same code path as
/// `image` all the way to the sampler, and the only thing that has to distinguish them is WHO
/// FILLS THEM: an `image` is fed by a producer the operator names on the command line, and these
/// two are fed by the channel itself.
///
/// That distinction is the whole reason this enum exists rather than a bool. `image_input_names()`
/// is what the producer uses to decide how many source producers to wire up, so an audio input
/// appearing in that list would make `[ISF] spectrum.fs` try to open a producer called
/// "audioFFT" and fail to load the shader at all.
enum class audio_input
{
    none,     ///< an ordinary input
    waveform, ///< ISF `audio`: raw samples, width = samples, height = channels
    fft,      ///< ISF `audioFFT`: per-bin magnitudes, width = bins, height = channels
};

struct input
{
    std::string         name;
    std::string         type;          ///< float | bool | event | long | color | point2D | image | audio | audioFFT
    std::vector<double> default_value; ///< 1 value (float/bool/long) or 2/4 (point2D/color)
    std::vector<double> min_value;
    std::vector<double> max_value;
    std::string         label;
    bool                is_image = false;

    /// `none` for everything except the two audio texture types.
    ///
    /// `max_value`'s first element carries ISF's `MAX` for these, which the specification defines
    /// as a CAP on the number of samples or bins rather than as a value limit. The producer
    /// treats it that way and nothing else reads it, so it does not need a field of its own.
    audio_input audio_kind = audio_input::none;

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
    ///
    /// EXCLUDES `audio` and `audioFFT`, which are image inputs in every other respect. This is
    /// the list the producer sizes its source-producer wiring from, so including them would make
    /// `[ISF] spectrum.fs` try to open a producer named "audioFFT".
    std::vector<std::string> image_input_names() const;

    /// The declared audio texture inputs: name, kind, and the `MAX` cap if one was given (0 if
    /// not). Empty for a shader that wants no audio, which is almost all of them.
    struct audio_input_desc
    {
        std::string name;
        audio_input kind = audio_input::none;
        int         max  = 0; ///< ISF `MAX`: a cap on samples or bins. 0 = unspecified
    };
    std::vector<audio_input_desc> audio_inputs() const;

    /// Set an input value by name (1..4 scalars). Returns false if the input is unknown.
    bool set_value(const std::string& name, const std::vector<double>& values);

    /// The CURRENT value of an input, or an empty vector if there is no such input.
    ///
    /// There was no read path at all until 2026-09-08: `ISF LIST` reports each input's
    /// declared default, min and max and never what it is set to now, so
    /// `CALL ... ISF SET brightness 0.5` was unverifiable through the server. A parameter that
    /// can be written and not read cannot be described to a control surface, cannot round-trip
    /// through a preset, and cannot be the target of a binding.
    std::vector<double> get_value(const std::string& name) const;

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

/// One ISF shader FOUND ON DISK, described without compiling it.
///
/// The counterpart of `ofx::plugin_info`, and it exists for the same reason: a client that has
/// to be told what it can run. `foreground/params` describes a producer that is ALREADY
/// PLAYING, which answers "what does this have" and not "what is there" -- so a control surface
/// could draw a panel for a shader an operator had already chosen and had no way to offer the
/// choice. There is no `CLS` for shaders: `CLS` lists media and `TLS` lists templates, and a
/// `.fs` is neither.
struct shader_info
{
    /// What `PLAY 1-1 [ISF] <this>` takes -- the file name, extension included. `load_shader`
    /// resolves an extensionless token too, but a name that is already resolvable is one less
    /// thing for a client to guess at.
    std::string name;

    /// Relative to the media folder, so a client can show a tree and a `.fs` in a subdirectory
    /// is distinguishable from one of the same name at the top.
    std::string path;

    std::string description; ///< the header's DESCRIPTION, or empty
    std::string credit;      ///< the header's CREDIT, or empty
    std::string isf_version; ///< the header's ISFVSN, or empty -- absent means ISF 1

    /// The header's CATEGORIES, which is how every published collection organises itself.
    std::vector<std::string> categories;

    int  inputs    = 0;     ///< how many INPUTS it declares, image inputs included
    bool multipass = false; ///< true when it declares PASSES
    bool has_vertex_shader = false; ///< a sibling `.vs` exists, so it overrides the pass-through

    /// Set when the file was read but its header could not be parsed as JSON, with the reason.
    /// A shader is REPORTED with this rather than dropped: a client showing 313 of 314 with no
    /// sign of the missing one is worse than showing it as broken, and the operator who wrote
    /// the header is the only person who can fix it.
    std::string error;
};

/// Every ISF shader under the media folder, recursively, sorted by path.
///
/// Parses the JSON header and NOTHING else -- no GL context, no compilation, no device. That is
/// what makes it callable from a protocol thread: compiling 327 shaders to answer a listing
/// would need the mixer's context and would take seconds.
std::vector<shader_info> discover_shaders();

}} // namespace caspar::isf
