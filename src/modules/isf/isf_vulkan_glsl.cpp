/*
 * Copyright (c) 2026 CasparCG Contributors
 *
 * This file is part of CasparCG (www.casparcg.com).
 *
 * CasparCG is free software: you can redistribute it and/or modify it under the terms of the GNU
 * General Public License as published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 */

#include "isf_vulkan_glsl.h"

#include <common/except.h>
#include <common/log.h>

#include <functional>
#include <algorithm>
#include <sstream>

namespace caspar { namespace isf {

namespace {

/// Byte offsets of the ISF fields in `ParamsBlock`, from
/// `accelerator/vulkan/util/uniform_block.h`.
///
/// ADDRESSED EXPLICITLY BECAUSE THE VARIANT DECLARES ONLY WHAT IT USES. The alternative is
/// emitting all three hundred fields of that block into generated text so the trailing ones land
/// at the right place, which would make every future edit to the mixer's uniforms a silent break
/// of every ISF shader.
///
/// **THIS IS THE ONE PLACE THE TWO SIDES CAN DISAGREE WITH NO COMPILE ERROR ON EITHER.** The
/// block is `layout(scalar)`; a wrong offset here reads a neighbouring float and renders a
/// plausible picture. `uniform_block.h` carries `static_assert`s pinning each of these, and
/// `isf_vulkan_self_test` re-checks the ones it can see, but the assertion that matters is the
/// PICTURE: `grade-graph`'s ISF arm gates the Vulkan result against the same closed-form model
/// the OpenGL arm uses, so a parameter arriving from the wrong offset fails at 1 LSB.
/// `flags2`, whose bit 0 is `output_bgra`. The variant must honour it -- see the wrapper at the
/// end of the generated source.
constexpr int OFF_FLAGS2         = 736;
/// `gn_isf_to_display`: which way to convert around the author's body. See the OpenGL preamble
/// in `isf_shader.cpp` -- the two must agree, because a document must render the same on either
/// mixer and this is a place where two implementations could quietly diverge.
constexpr int OFF_ISF_TODISPLAY  = 1212;
constexpr int OFF_ISF_VALUES     = 1056;
constexpr int OFF_ISF_COUNT      = 1184;
constexpr int OFF_ISF_TIME       = 1188;
constexpr int OFF_ISF_TIMEDELTA  = 1192;
constexpr int OFF_ISF_FRAME      = 1196;
constexpr int OFF_ISF_PASS       = 1200;
constexpr int OFF_ISF_RENDERSIZE = 1204;

/// Which texture slot an ISF image input lands in.
///
/// The node path already binds `in0` to slot 0, `in1` to 1 and a materialised mask to 2, and the
/// mixer declares eight. So an ISF shader's FIRST image input is the node's own input -- slot 0,
/// which is `inputImage` for the overwhelming majority of published shaders -- and further ones
/// take 1 and upward. A shader wanting more images than the array holds is refused rather than
/// silently given a texture belonging to something else.
constexpr int max_image_slots = 8;

/// The GLSL type for an ISF input. Mirrors `gl_type_of` in `isf_shader.cpp` deliberately: the two
/// backends must agree about what a `long` is, and a divergence here is a shader that compiles on
/// one mixer and not the other.
const char* glsl_type_of(const std::string& t)
{
    if (t == "float")
        return "float";
    if (t == "bool" || t == "event")
        return "bool";
    if (t == "long")
        return "int";
    if (t == "color")
        return "vec4";
    if (t == "point2D")
        return "vec2";
    return nullptr; // an image, or a type this subset does not carry
}

/// How many `gn_isf[]` components an input occupies. Matches the node compiler's `arity`, because
/// the array IS that compiler's value slots copied across -- see `build_vulkan_fragment`.
int components_of(const std::string& t)
{
    const auto* g = glsl_type_of(t);
    if (!g)
        return 0;
    if (t == "color")
        return 4;
    if (t == "point2D")
        return 2;
    return 1;
}

} // namespace

vulkan_source build_vulkan_fragment(const std::vector<input>&       inputs,
                                    const std::string&              body,
                                    const std::string&              cache_key,
                                    const std::vector<std::string>& targets)
{
    vulkan_source out;
    out.cache_id = "isf:" + cache_key;

    // ---- the image inputs, in declaration order ------------------------------------------
    std::vector<std::string> images;
    for (const auto& in : inputs) {
        if (!in.is_image && in.audio_kind == audio_input::none)
            continue;
        images.push_back(in.name);
    }
    if (static_cast<int>(images.size()) > max_image_slots) {
        out.error = "this shader declares " + std::to_string(images.size()) +
                    " image inputs and the mixer binds " + std::to_string(max_image_slots) +
                    ". Refused rather than bound to whatever is in the spare slots";
        return out;
    }

    if (static_cast<int>(targets.size()) > max_isf_targets) {
        out.error = "this shader declares " + std::to_string(targets.size()) +
                    " PASSES targets and the mixer can sample " + std::to_string(max_isf_targets) +
                    ". Refused rather than bound to whatever is in the spare slots";
        return out;
    }

    // ---- the scalar inputs, and their slots ----------------------------------------------
    //
    // IN DECLARATION ORDER, which is not a stylistic choice: the node compiler walks
    // `instance_ports` and allocates value slots in that same order, and `resolve_node_ports`
    // builds those ports from this same input list. So walking it here reproduces the compiler's
    // packing exactly, and the `#define`s below land on the right numbers by construction rather
    // than by a second table that could drift.
    std::ostringstream defines;
    int                slot = 0;
    for (const auto& in : inputs) {
        const auto* g = glsl_type_of(in.type);
        if (!g)
            continue;
        const int n = components_of(in.type);
        if (slot + n > max_isf_values) {
            out.error = "this shader's parameters need " + std::to_string(slot + n) +
                        " components and the mixer carries " + std::to_string(max_isf_values) +
                        ". Refused rather than truncated: a silently dropped parameter renders a "
                        "plausible picture and reports nothing";
            return out;
        }
        // A `#define`, NOT a `const` or a function -- so the author's body sees a plain name of
        // the right type wherever it used one, with no rewriting of the body at all.
        defines << "#define " << in.name << " ";
        if (n == 1) {
            if (std::string(g) == "bool")
                defines << "(gn_isf[" << slot << "] != 0.0)";
            else if (std::string(g) == "int")
                defines << "int(gn_isf[" << slot << "])";
            else
                defines << "gn_isf[" << slot << "]";
        } else if (n == 2) {
            defines << "vec2(gn_isf[" << slot << "], gn_isf[" << (slot + 1) << "])";
        } else {
            defines << "vec4(gn_isf[" << slot << "], gn_isf[" << (slot + 1) << "], gn_isf["
                    << (slot + 2) << "], gn_isf[" << (slot + 3) << "])";
        }
        defines << "\n";
        slot += n;
    }
    out.value_count = slot;

    std::ostringstream f;
    f << "#version 450\n"
         "#extension GL_EXT_scalar_block_layout : require\n"
         "\n"
         // The mixer's vertex stage, unchanged and shared: ISF's normalised coordinate is
         // `TexCoord.xy`. Declaring the whole vec4 rather than a vec2 because the location's
         // type has to match what the vertex shader writes.
         "layout(location = 0) in vec4 TexCoord;\n"
         "layout(location = 1) in vec4 TexCoord2;\n"
         "layout(location = 0) out vec4 isf_out_color;\n"
         "\n"
         "layout(binding = 0) uniform sampler2D textures["
      << max_image_slots
      << "];\n"
         "\n"
         // ONLY THE ISF TAIL OF THE BLOCK, by explicit offset. See the note on the offset
         // constants: this is the seam that fails silently if it ever disagrees.
         "layout(scalar, binding = 2) uniform ParamsBlock {\n"
         "    layout(offset = "
      << OFF_FLAGS2
      << ") uint gn_flags2;\n"
         "    layout(offset = "
      << OFF_ISF_VALUES
      << ") float gn_isf[" << max_isf_values
      << "];\n"
         "    layout(offset = "
      << OFF_ISF_COUNT
      << ") int gn_isf_count;\n"
         "    layout(offset = "
      << OFF_ISF_TIME
      << ") float gn_isf_time;\n"
         "    layout(offset = "
      << OFF_ISF_TIMEDELTA
      << ") float gn_isf_timedelta;\n"
         "    layout(offset = "
      << OFF_ISF_FRAME
      << ") int gn_isf_frame;\n"
         "    layout(offset = "
      << OFF_ISF_PASS
      << ") int gn_isf_pass;\n"
         "    layout(offset = "
      << OFF_ISF_RENDERSIZE
      << ") vec2 gn_isf_rendersize;\n"
         "    layout(offset = "
      << OFF_ISF_TODISPLAY
      << ") int gn_isf_to_display;\n"
         "};\n"
         "\n"
         // ---- ISF's standard uniforms, as the spec names them ----------------------------
         "#define RENDERSIZE gn_isf_rendersize\n"
         "#define TIME gn_isf_time\n"
         "#define TIMEDELTA gn_isf_timedelta\n"
         "#define FRAMEINDEX gn_isf_frame\n"
         "#define PASSINDEX gn_isf_pass\n"
         // DATE is in the spec and almost nothing uses it. Zero rather than absent, because a
         // shader referencing it must compile; a wrong date is a visible nothing, an undeclared
         // identifier is a shader that will not load.
         "const vec4 DATE = vec4(0.0, 0.0, 0.0, 0.0);\n"
         "\n"
         // THE PERSPECTIVE DIVIDE, which the base shader does and the first version of this did not:
         // `fragment_shader.frag` samples with `TexCoord.st / TexCoord.q`. With a default
         // geometry q is 1 and the divide is a no-op, which is exactly why omitting it is easy
         // and why it must not be omitted -- a layer with a corner-pin or any projective
         // placement would sample somewhere else entirely.
         // ISF'S Y IS UP, THE MIXER'S IS DOWN. The spec puts `isf_FragNormCoord` (0,0) at the
         // BOTTOM left; the mixer's texture coordinate has its origin at the top. A shader whose
         // output depends on position -- a gradient, a wipe, anything with a `step(0.5, y)` --
         // renders mirrored without this, and a shader whose output does not depend on position
         // cannot tell you it is wrong.
         "vec2 isf_uv = TexCoord.st / TexCoord.q;\n"
         "vec2 isf_FragNormCoord = vec2(isf_uv.x, 1.0 - isf_uv.y);\n"
         "#define vv_FragNormCoord isf_FragNormCoord\n"
         "#define isf_FragCoord (isf_FragNormCoord * RENDERSIZE)\n"
         "\n";

    // ---- the PASSES targets, in descriptor set 1 -----------------------------------------
    //
    // SET 1, NOT SET 0. Set 0's sampler array is the mixer's own plane slots and is full; set 1
    // carries `OCIO_MAX_TEXTURES` bindings that a variant pipeline may use as it likes, and a
    // node pass never has an OCIO transform. So multi-pass costs no descriptor layout change, no
    // new binding in set 0 and nothing in the uniform block -- which is the whole reason this
    // shape was chosen over a second descriptor set.
    //
    // Bound per LAYER by the kernel, through the same `lut_views.ocio` slots OCIO writes, with a
    // linear/clamp sampler -- which is what the OpenGL path gives a pass buffer too.
    for (std::size_t k = 0; k < targets.size(); ++k) {
        f << "layout(set = 1, binding = " << (k + 1) << ") uniform sampler2D _isf_tgt_" << k
          << ";\n";
        f << "#define " << targets[k] << " _isf_tgt_" << k << "\n";
        f << "#define _" << targets[k] << "_imgSize vec2(textureSize(_isf_tgt_" << k << ", 0))\n";
    }
    f << "\n";

    // ---- the image inputs as macros over the bound slots ---------------------------------
    //
    // A `#define` per name rather than a sampler variable, because a sampler cannot be assigned
    // in GLSL and the author's body names the input directly.
    for (std::size_t i = 0; i < images.size(); ++i) {
        f << "#define " << images[i] << " textures[" << i << "]\n";
        f << "#define _" << images[i] << "_imgSize vec2(textureSize(textures[" << i << "], 0))\n";
    }
    f << "\n";

    // THE MACRO FAMILY PORTS UNCHANGED from the OpenGL preamble -- it is GLSL text, not API --
    // except that the flip and swizzle flags are gone. On the OpenGL mixer those exist because a
    // producer's plane is bottom-up and BGRA; here the source is a node ATTACHMENT, which is
    // top-down and RGBA, so there is nothing to correct. That asymmetry is the whole of the
    // 2026-09-12 channel-order finding, and this is the other end of it.
    // ── THE FETCH UNDOES BOTH CONVENTIONS, AND MEASUREMENT IS WHAT ESTABLISHED THEM ──
    //
    // A node attachment on the Vulkan mixer is top-down and holds the mixer's own byte order --
    // the same order every write in `fragment_shader.frag` chooses at runtime from
    // `F2_OUTPUT_BGRA`. So a sampler reading one has to flip the coordinate back and apply the
    // same swizzle, or the author's shader sees a picture that is upside down and has its red
    // and blue exchanged.
    //
    // BOTH WERE MEASURED, SEPARATELY, because the fixture was built to separate them: with the
    // input reversed and flipped, the filter's TOP patch read [15, 40, 112] -- which is the
    // reversed source times the BOTTOM gains -- and its bottom patch read [46, 62, 35], the
    // reversed source times the TOP gains. Two exact permutations, two named checks.
    //
    // AND THEY DID NOT CANCEL, which is the whole reason the two gain sets are not red/blue
    // mirrors of each other. A mirror-symmetric fixture reports these two faults together as no
    // fault at all.
    // ── THE SPACE CONVERSION, IDENTICAL TO THE OPENGL PREAMBLE'S ────────────────────
    //
    // BT.1886 -- pure gamma 2.4, the curve both mixer shaders carry as `oetf_rec709` /
    // `eotf_rec709` for SDR. An exact inverse pair, so the round trip is lossless within 0..1.
    // TRANSFER ONLY: no gamut conversion and no tone map, because the output half bundles those
    // with a clamp and that composition has no inverse.
    //
    // THE TWO BACKENDS MUST AGREE HERE, and this is exactly the kind of place they would not:
    // the same three lines, written twice, in two languages. `grade-graph` gates both against
    // the same closed-form model rather than against each other, for that reason.
    f << "vec3 _isf_enc(vec3 c) { return pow(max(c, vec3(0.0)), vec3(1.0 / 2.4)); }\n"
         "vec3 _isf_dec(vec3 c) { return pow(max(c, vec3(0.0)), vec3(2.4)); }\n"
         "vec4 _isf_fetch(sampler2D s, vec2 nc) {\n"
         "  vec4 c = texture(s, vec2(nc.x, 1.0 - nc.y));\n"
         "  c = ((gn_flags2 & 1u) != 0u) ? c.bgra : c;\n"
         "  if (gn_isf_to_display > 0) c.rgb = _isf_enc(c.rgb);\n"
         "  else if (gn_isf_to_display < 0) c.rgb = _isf_dec(c.rgb);\n"
         "  return c;\n"
         "}\n"
         "#define IMG_SIZE(image) vec2(textureSize(image, 0))\n"
         "#define IMG_NORM_PIXEL(image, nc) _isf_fetch(image, vec2(nc))\n"
         "#define IMG_PIXEL(image, pc) IMG_NORM_PIXEL(image, (pc) / IMG_SIZE(image))\n"
         "#define IMG_THIS_NORM_PIXEL(image) IMG_NORM_PIXEL(image, isf_FragNormCoord)\n"
         "#define IMG_THIS_PIXEL(image) IMG_THIS_NORM_PIXEL(image)\n"
         "\n"
      << defines.str()
      << "\n"
         // -- THE AUTHOR WRITES TO A LOCAL, AND A WRAPPER APPLIES THE MIXER'S BYTE ORDER --
         //
         // The Vulkan mixer decides its output byte order at RUNTIME: every write in
         // `fragment_shader.frag` is `flag2(F2_OUTPUT_BGRA) ? col.bgra : col`. A variant that
         // wrote straight RGB would therefore be correct on some configurations and reversed on
         // others -- which is not a thing to leave to luck.
         //
         // MEASURED ON THE FIRST RUN THAT DREW: a shader emitting (0.10, 0.30, 0.40) rendered
         // [102, 76, 25] -- exactly reversed, with GREEN CORRECT. That is the same signature the
         // OpenGL path produced for its own version of this mistake, and it is why both
         // `grade-graph` ISF fixtures are asymmetric: a grey one passes this silently.
         //
         // `#define main isf_main` renames the author's entry point WITHOUT editing the body,
         // so the promise that the shader's own source passes through untouched still holds.
         // The wrapper then runs after it and applies the order the mixer asked for.
         "vec4 isf_color = vec4(0.0);\n"
         "#define gl_FragColor isf_color\n"
         "#define main isf_main\n"
         // `#line 1` so a compiler diagnostic names the line the AUTHOR wrote, not the line of
         // generated preamble it landed on. Without it every error in a shader file points
         // roughly forty lines past where it is.
         "#line 1\n"
      << body
      << "\n#undef main\n"
         "void main() {\n"
         "  isf_main();\n"
         "  vec4 o = isf_color;\n"
         "  if (gn_isf_to_display > 0) o.rgb = _isf_dec(o.rgb);\n"
         "  else if (gn_isf_to_display < 0) o.rgb = _isf_enc(o.rgb);\n"
         "  isf_out_color = ((gn_flags2 & 1u) != 0u) ? o.bgra : o;\n"
         "}\n";

    out.source = f.str();
    return out;
}

vulkan_source build_vulkan_fragment_for(const std::string& path)
{
    vulkan_source out;

    std::string  source;
    std::wstring base;
    std::string  err;
    if (!load_shader_source(u16(path), source, base, err)) {
        out.error = err;
        return out;
    }

    const auto inputs = describe_inputs(u16(path), err);
    if (!err.empty()) {
        out.error = err;
        return out;
    }

    // THE PASS TARGETS, so the generated shader can declare a sampler for each. Deduped and in
    // declaration order, which is the order the evaluator binds them in -- the two walk the same
    // list, so a mismatch is not possible rather than merely unlikely.
    std::vector<std::string> targets;
    for (const auto& pi : describe_passes(u16(path), err)) {
        if (pi.target.empty())
            continue;
        if (std::find(targets.begin(), targets.end(), pi.target) == targets.end())
            targets.push_back(pi.target);
    }
    if (!err.empty()) {
        out.error = err;
        return out;
    }

    return build_vulkan_fragment(inputs, source, path, targets);
}

void isf_vulkan_self_test()
{
    const auto fail = [](const std::string& why) {
        CASPAR_LOG(fatal) << L"[isf_vulkan_self_test] " << u16(why);
        CASPAR_THROW_EXCEPTION(caspar_exception() << msg_info("isf_vulkan_self_test: " + why));
    };

    // ── THE ARRAY THIS MODULE BELIEVES IN IS THE ONE THE MIXER DECLARES ──────────────
    //
    // `max_isf_values` is duplicated rather than included, because the dependency runs
    // module -> accelerator everywhere else here. A drift would refuse valid shaders, or let one
    // through that overruns the array. There is no compile-time way to check it from this side,
    // so it is checked at boot against the offsets it implies.
    if (OFF_ISF_COUNT - OFF_ISF_VALUES != max_isf_values * 4)
        fail("gn_isf's offsets imply " + std::to_string((OFF_ISF_COUNT - OFF_ISF_VALUES) / 4) +
             " components but this module is built for " + std::to_string(max_isf_values) +
             " -- the mixer's uniform block and this generator disagree about the array's size");

    // ── A SHADER WITH ONE OF EVERY DECLARABLE TYPE ───────────────────────────────────
    //
    // Not a real file: the generator is pure text by design so this needs no media folder, no
    // GL, no Vulkan and no channel. A boot self-test that needed any of those would not run.
    std::vector<input> ins;
    const auto         add = [&ins](const char* name, const char* type, std::vector<double> def) {
        input i;
        i.name          = name;
        i.type          = type;
        i.default_value = std::move(def);
        i.is_image      = std::string(type) == "image";
        ins.push_back(std::move(i));
    };
    add("inputImage", "image", {});
    add("level", "float", {0.5});
    add("enabled", "bool", {1.0});
    add("mode", "long", {0.0});
    add("centre", "point2D", {0.5, 0.5});
    add("tint", "color", {0.2, 0.6, 0.8, 1.0});

    const auto r = build_vulkan_fragment(ins, "void main() { gl_FragColor = vec4(level); }", "selftest");
    if (!r.error.empty())
        fail("the generator refused a shader of one input per type: " + r.error);

    // 1 + 1 + 1 + 2 + 4 = 9, and the IMAGE takes none: images are textures, not values.
    if (r.value_count != 9)
        fail("one input of each type packs to " + std::to_string(r.value_count) +
             " components; it must be 9 (float 1, bool 1, long 1, point2D 2, color 4, image 0). "
             "If this moved, the generator and the node compiler no longer agree about slots -- "
             "which puts every parameter on the wrong input while still rendering");

    // WHAT THE CONTRACT SAYS IT DECLARES. Asserted by NAME rather than by compiling, because a
    // shader that compiles is not the same claim as a shader an ISF author's source will find
    // what it expects in -- an omitted `TIMEDELTA` compiles perfectly until a shader uses it.
    const char* required[] = {"#version 450",
                              "GL_EXT_scalar_block_layout",
                              "RENDERSIZE",
                              "TIME",
                              "TIMEDELTA",
                              "FRAMEINDEX",
                              "PASSINDEX",
                              "DATE",
                              "isf_FragNormCoord",
                              "IMG_THIS_PIXEL",
                              "IMG_NORM_PIXEL",
                              "IMG_PIXEL",
                              "IMG_SIZE",
                              "gl_FragColor",
                              "gn_isf_to_display",
                              "_isf_enc",
                              "_isf_dec"};
    for (const auto* n : required)
        if (r.source.find(n) == std::string::npos)
            fail(std::string("the generated shader does not declare `") + n +
                 "`, which the ISF specification says every shader has");

    // THE AUTHOR'S BODY IS NOT REWRITTEN -- the promise that lets the two backends agree.
    if (r.source.find("void main() { gl_FragColor = vec4(level); }") == std::string::npos)
        fail("the author's body was altered on its way through the generator");

    // AND THE SLOTS THE DEFINES NAME ARE THE ONES THE COMPILER PACKS. Spot-checked on the two
    // that would be silently wrong rather than broken: `centre` at 3 and `tint` at 4.
    if (r.source.find("#define centre vec2(gn_isf[3], gn_isf[4])") == std::string::npos)
        fail("`centre` is not defined over slots 3..4; the generator's packing has drifted from "
             "the node compiler's");
    if (r.source.find("#define tint vec4(gn_isf[5], gn_isf[6], gn_isf[7], gn_isf[8])") ==
        std::string::npos)
        fail("`tint` is not defined over slots 5..8; the generator's packing has drifted from the "
             "node compiler's");

    // A SHADER THAT ASKS FOR MORE THAN THE ARRAY HOLDS IS REFUSED, NOT TRUNCATED.
    std::vector<input> many;
    for (int i = 0; i < max_isf_values + 1; ++i) {
        input in;
        in.name = "p" + std::to_string(i);
        in.type = "float";
        in.default_value.push_back(0.0);
        many.push_back(std::move(in));
    }
    const auto over = build_vulkan_fragment(many, "void main() {}", "overflow");
    if (over.error.empty())
        fail("a shader needing more than " + std::to_string(max_isf_values) +
             " components was accepted. It must be refused: truncating parameters renders a "
             "plausible picture and reports nothing");

    CASPAR_LOG(info) << L"[isf] Vulkan GLSL generator self-test passed.";
}

}} // namespace caspar::isf
