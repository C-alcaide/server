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

#include "isf_shader.h"

#include <string>
#include <vector>

namespace caspar { namespace isf {

/// Emit a **Vulkan GLSL** fragment shader for one ISF shader, to be compiled to SPIR-V and bound
/// as a per-layer VARIANT PIPELINE by the Vulkan mixer.
///
/// ── WHY THIS EXISTS AT ALL, GIVEN `isf::shader` ALREADY RUNS ISF ────────────────────────────
///
/// `isf::shader` runs the author's GLSL on an OpenGL context. That serves the OpenGL mixer and
/// the ISF producer, and it cannot serve a Vulkan NODE: a node's input is an attachment inside a
/// renderpass that accumulates every layer of the channel and commits once per frame, so there is
/// nothing for a GL context to import. Forcing the issue by committing mid-accumulation is
/// measured at 241 `ErrorDeviceLost` in one run. So the shader has to become an ordinary pipeline
/// in the same renderpass as every other node pass, which means SPIR-V, which means this.
///
/// ── WHAT IT EMITS, AND WHAT IT DELIBERATELY DOES NOT TOUCH ──────────────────────────────────
///
/// A STANDALONE fragment shader, not a splice into the mixer's own. A pipeline layout comes from
/// the layout object rather than from shader reflection, so a variant may declare only the
/// bindings it actually uses -- which avoids generating three hundred `ParamsBlock` fields into
/// text purely to reach the handful at the end.
///
/// **The shader BODY is the author's GLSL and is not rewritten.** That is the same promise the
/// OpenGL path keeps by running the source directly, and it is what makes the two backends able
/// to agree: the parts that differ are declarations, and declarations are what this file is.
///
/// The author's declared inputs become `#define`s onto `gn_isf[]`, in the SAME ORDER the node
/// compiler packs its value slots -- see `build_vulkan_fragment`.
struct vulkan_source
{
    /// The generated GLSL. Empty when `error` is set.
    std::string source;
    /// A stable id for the pipeline cache, derived from the shader's path and its declaration
    /// set. Two nodes running the same file share one compiled pipeline.
    std::string cache_id;
    /// How many `gn_isf[]` components the author's inputs consume. The caller packs exactly this
    /// many and refuses a shader needing more than the array holds.
    int value_count = 0;
    /// Why nothing was emitted. Empty on success.
    std::string error;
};

/// The size of `gn_isf[]` in `accelerator/vulkan/util/uniform_block.h`.
///
/// DUPLICATED DELIBERATELY, AND CHECKED AT BOOT rather than included. `modules/isf` must not
/// depend on an accelerator header for a number -- the dependency runs the other way everywhere
/// else in this module -- so `isf_vulkan_self_test` asserts the two agree and refuses to boot if
/// they drift. A silent disagreement would refuse valid shaders, or overrun the array.
constexpr int max_isf_values = 32;

/// Generate from an already-parsed shader. Pure text: no GL, no Vulkan, no file system, so a boot
/// self-test can drive it with a literal.
vulkan_source build_vulkan_fragment(const std::vector<input>& inputs,
                                    const std::string&        body,
                                    const std::string&        cache_key);

/// Generate from a shader named the way a document names it -- relative to the media folder,
/// through the same `load_shader_source` the port resolver uses, so a path cannot mean two
/// different files to the two of them.
vulkan_source build_vulkan_fragment_for(const std::string& path);

/// Aborts the boot if the generator's contract is broken. Called from `shell/server.cpp` beside
/// the other self-tests.
void isf_vulkan_self_test();

}} // namespace caspar::isf
