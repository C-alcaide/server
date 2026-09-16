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
#include <functional>
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

/// The role a shader plays, from the image inputs it declares.
///
/// **ONE RULE, AND THAT IS THE POINT OF IT BEING A FUNCTION.** Two callers want this -- the
/// listing that answers `INFO ISF`, and anything that loads a shader for real -- and they parse
/// the header by different routes: `discover_shaders` reads the JSON and nothing else, while
/// `shader` builds a full input table. A second copy of the rule would drift, and the two would
/// then disagree about whether a shader is a filter, which is exactly the class of divergence
/// the pass-size evaluator already had to be made single-sourced to avoid.
///
/// `image_input_names` is the declared image inputs in declaration order.
shader_role role_of(const std::vector<std::string>& image_input_names);

/// `"generator"`, `"filter"` or `"transition"` -- the wire spelling, for `INFO ISF` and anything
/// else that reports it. Beside the classifier so the names cannot drift from the enum.
const char* role_name(shader_role r);

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

    /// Run every PASS buffer and the final pass target at fp16, rather than at the ISF
    /// spec's 8-bit / `FLOAT`-32 pair.
    ///
    /// FOR THE NODE PATH ONLY, and default off so the ISF PRODUCER is byte-identical. A node
    /// graph's intermediates are fp16 everywhere else, and the Vulkan node path has no other
    /// option -- its attachments come from the mixer's own pool. Matching here is what lets one
    /// document render the same on both mixers.
    ///
    /// ⚠ `FLOAT: true` therefore gets fp16, not 32-bit. A slow accumulator drifts differently
    /// from a reference host; stated in `node-graph.md` rather than left to be discovered.
    /// It also REMOVES a clip: the OpenGL final pass was `GL_RGBA16` UNORM, so a node's output
    /// could not exceed 1.0 at all.
    void set_node_buffer_format(bool fp16);

    /// Free every GL object this shader owns, on the CURRENTLY BOUND context.
    ///
    /// `~shader` frees through the device it was given, and a shader built for a NODE has no
    /// device -- so a node's shader leaked its program, its VAOs and its pass buffers. Harmless
    /// while there was one cache entry per path for the life of the process; not harmless once
    /// there is one per node instance and instances come and go with documents.
    ///
    /// The caller must have the right context current. The mixer's renderer destroys its store
    /// inside a `dispatch_sync`, which is where that is true.
    void release_gl_on_current_context();

    /// Re-blacken every persistent pass buffer, on the currently bound context.
    ///
    /// The `reset` port, and the same clear an allocation already does -- VVISF blackens a
    /// persistent buffer on creation *"as a persistent buffer the content of this frame will
    /// matter immediately and it will screw things up if it's anything but a black frame"*.
    void reset_persistent_buffers();

    /// Which way to convert around the author's body: +1 encodes to display-referred before
    /// the shader and decodes after, -1 the reverse, 0 applies nothing.
    ///
    /// BT.1886 (pure gamma 2.4), TRANSFER ONLY -- no gamut conversion and no tone map, because
    /// the mixer's output half bundles those with a clamp and the composition has no inverse.
    /// Defaults to 0, so the ISF producer and every agreeing node are unaffected.
    void set_space_conversion(int to_display);

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
    /// Render into `dst_gl_texture` on the CURRENTLY BOUND context, with no context switch.
    ///
    /// FOR A NODE ON THE OPENGL MIXER, where the mixer's own device is already current and
    /// owns both textures: taking a `gl_context` there would mean rendering on a second context
    /// that cannot see either of them. Otherwise identical to `render_into_shared` -- same
    /// output pass, same top-down BGRA result.
    ///
    /// `finish`: whether to `glFinish()` before returning. TRUE for the Vulkan route, where
    /// nothing else orders these writes against the mixer's read; FALSE on the OpenGL mixer,
    /// where the next draw is on this same context and is ordered by the driver already -- a
    /// full pipeline stall per node per frame, bought for nothing.
    ///
    /// ⚠ LEAVES GL STATE DISTURBED. The framebuffer binding, the viewport, the program and the
    /// vertex array are restored; the BLEND ENABLE, the active texture unit and the bindings on
    /// it are not. A caller that keeps cached state -- which the mixer's kernel does -- must
    /// re-establish it afterwards rather than trust this.
    bool render_into_current(int                               width,
                             int                               height,
                             double                            time,
                             double                            time_delta,
                             int                               frame_index,
                             const std::vector<image_binding>& images,
                             unsigned int                      dst_gl_texture,
                             bool                              finish  = false,
                             /// TRUE writes BGRA bytes, which is what a PRODUCER's frame must
                             /// hold. FALSE writes RGBA, which is what a node-graph ATTACHMENT
                             /// holds -- the kernel writes `col.bgra` into one from its internal
                             /// BGR convention. The two destinations are opposite, and neither
                             /// side of the boundary says so; the output program carries the
                             /// measurement that established it.
                             bool                              swap_rb = true);

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

    /// Generator, filter or transition -- see `role_of`. **This is the field a client sorts a
    /// shader list by**: a filter needs a layer beneath it, a generator does not, and a
    /// transition needs two sources, so it decides where a shader can be used at all rather
    /// than merely describing it.
    shader_role role = shader_role::generator;
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

/// The INPUTS one shader declares, parsed from its header and nothing else.
///
/// `discover_shaders()` answers "what is there" and reports a COUNT of inputs; this answers
/// "what does this one take", which is what a caller needs to build a parameter surface from.
///
/// NO GL CONTEXT, NO COMPILATION, NO DEVICE -- the same property that makes `discover_shaders`
/// callable from a protocol thread, and for the same reason: the node graph resolves a node's
/// ports on the API executor during a PUT, and compiling a shader there would put the mixer's
/// context on the write path.
///
/// `path` is resolved the way `PLAY 1-1 [ISF] <name>` resolves it, so a document may name a
/// shader the same way an operator does. Returns empty and sets `out_error` when the file
/// cannot be read or its header cannot be parsed -- an empty list is never a valid answer for a
/// shader that exists, because a shader with no INPUTS still has its image input.
/// Read a shader named the way a document names it: relative to the media folder, with `.fs`,
/// `.glsl` and `.frag` probed when no extension is given, and anything resolving outside the
/// media root refused.
///
/// SHARED BY THE PORT RESOLVER AND THE NODE RENDERER so a path cannot mean two different files
/// to the two of them -- which would give a node its parameters from one shader and its picture
/// from another. Returns false with `out_error` set.
///
/// `out_has_vertex_shader`, when given, reports whether the resolved `.fs` has a SIBLING `.vs`.
/// It is reported from HERE rather than probed by the caller for the reason above: a second
/// resolution of the same token is a second chance to disagree about which file it names.
bool load_shader_source(const std::wstring& path,
                        std::string&        out_source,
                        std::wstring&       out_base_path,
                        std::string&        out_error,
                        bool*               out_has_vertex_shader = nullptr);

std::vector<input> describe_inputs(const std::wstring& path, std::string& out_error);

/// What a shader asks for beyond a single pass of plain GLSL.
///
/// Exists so a NODE can be refused for something the node evaluator does not implement yet,
/// while the PRODUCER -- which has run all of this for years -- is untouched. The two have
/// different capabilities and the same file may be loaded by either.
struct shader_features
{
    /// `PASSES` with more than one entry.
    bool multipass = false;
    /// Any pass declaring `PERSISTENT`.
    bool persistent = false;
    /// An `IMPORTED` block.
    bool imported = false;
};

/// Parse-only, no GL. Empty `out_error` on success.
shader_features describe_features(const std::wstring& path, std::string& out_error);

/// One entry of a shader's `PASSES`, as the JSON header declares it.
struct pass_info
{
    /// `TARGET`, or empty for a pass that renders to the node's own output.
    std::string target;
    bool        persistent = false;
    /// `FLOAT`: the author asking for 32-bit precision. The node path runs every buffer at fp16
    /// (see `set_node_buffer_format`), so this is carried and not honoured -- stated in
    /// `node-graph.md` rather than silently ignored.
    bool        is_float = false;
    /// `WIDTH` / `HEIGHT` as written: expressions over `$WIDTH`, `$HEIGHT` and the shader's own
    /// inputs, or empty for "the size asked of the node".
    std::string w_expr;
    std::string h_expr;
};

/// A shader's passes, in declaration order. One entry for a single-pass shader.
std::vector<pass_info> describe_passes(const std::wstring& path, std::string& out_error);

/// One `IMPORTED` entry, as the JSON header declares it.
struct imported_info
{
    /// The sampler name the shader refers to it by.
    std::string name;
    /// `PATH`, as written -- relative to the shader's own directory, or absolute.
    std::string path;
};

/// A shader's `IMPORTED` images, in declaration order, with the directory to resolve them
/// against.
///
/// DECLARATION ORDER IS PART OF THE CONTRACT, not an implementation detail: the generated
/// Vulkan GLSL binds these into descriptor set 1 by index, and the mixer fills the same indices
/// from this list. Two independent orderings of the same JSON would put the wrong picture in the
/// wrong sampler -- and the shader would still compile and still render, which is the failure
/// this codebase has paid for twice.
std::vector<imported_info>
describe_imported(const std::wstring& path, std::wstring& out_base, std::string& out_error);

/// Evaluate one `WIDTH`/`HEIGHT` expression.
///
/// **THE SAME EVALUATOR BOTH BACKENDS USE.** ISF sizes are arbitrary arithmetic over `$WIDTH`,
/// `$HEIGHT` and the shader's declared inputs, so two implementations would round differently
/// and the two mixers would allocate different buffers for one document -- which is the failure
/// this whole feature's rules exist to prevent. `var` resolves a `$name` to its current value.
int eval_pass_size(const std::string&                              expr,
                   int                                             fallback,
                   int                                             render_w,
                   int                                             render_h,
                   const std::function<bool(const std::string&, double&)>& var);

}} // namespace caspar::isf
