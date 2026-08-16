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

#pragma once

#include <string>
#include <vector>

namespace caspar { namespace accelerator { namespace ocio {

/// Process-wide access to the loaded OpenColorIO config.
///
/// Deliberately a pure-std interface with no OCIO types in it. Two reasons: OCIO's headers
/// stay confined to the one translation unit that implements this, and the rest of the tree
/// -- protocol in particular, which needs the discovery commands -- does not acquire a
/// compile-time dependency on the library. It also means a build with ENABLE_OCIO=OFF
/// compiles unchanged, with available() answering false.
///
/// The config is process-wide rather than per-channel because it is what makes a look
/// reproducible: pinning one built-in config URI for the whole server means a colour space
/// name resolves to the same transform on every channel and in every log line.

/// Was the server built with OCIO support?
bool available();

/// OCIO library version, e.g. "2.5.2". Empty when unavailable.
std::string version();

/// The URI of the currently loaded config. Loads the default on first call.
std::string config_uri();

/// Replace the loaded config.
///
/// Returns false on failure -- an unknown URI, a malformed file -- having logged why, and
/// leaves the previously loaded config in place. A failed load must not leave the server
/// with no colour config, because commands already accepted refer to it.
bool load_config(const std::string& uri);

/// Colour space names in the loaded config, in config order.
std::vector<std::string> colorspaces();

/// Display names in the loaded config.
std::vector<std::string> displays();

/// View names available for `display`. Empty if the display is unknown.
std::vector<std::string> views(const std::string& display);

/// Whether `name` names a colour space in the loaded config. For validating an AMCP
/// argument at command time, so a bad name fails the command rather than the frame.
bool has_colorspace(const std::string& name);

/// Whether `display` exists and offers `view`.
bool has_display_view(const std::string& display, const std::string& view);

/// Read an ASC CDL file -- `.cdl`, `.ccc` or `.cc` -- into the ten SOP+sat values.
///
/// Parsed by OCIO rather than by hand: it already implements the ASC schema for all three
/// container shapes, and the parsers were hardened in 2.5.2 (CVE-2026-42450). These files
/// are operator-supplied, so that is the half of the decision that matters.
///
/// `cccid` selects one correction from a `.ccc` collection (or a `.cdl` holding several) by
/// its `id` attribute. Empty means "the only one", and a file with several then fails rather
/// than silently taking the first.
///
/// Returns false having logged why. The values are untouched on failure, so a caller that
/// ignores the result renders what it had rather than a partly-applied grade.
///
/// NOTE: this reaches OCIO for PARSING only. The grade itself runs in the existing shader
/// CDL block, so the numbers here are exactly what `MIXER CDL` takes and the two paths are
/// required to render identically.
bool load_cdl(const std::string& path,
              const std::string& cccid,
              double             slope[3],
              double             offset[3],
              double             power[3],
              double&            saturation);

/// Look (LMT) names in the loaded config, in config order.
///
/// A look is a creative or technical transform applied in the working space *before* the
/// display rendering -- the show LUT of an ACES pipeline. The pinned built-in config defines
/// exactly one, `ACES 1.3 Reference Gamut Compression`; a studio config supplies its own,
/// which is what `<ocio-config>` is for.
std::vector<std::string> looks();

/// Whether `name` names a look in the loaded config. For validating an AMCP argument at
/// command time, so a bad name fails the command rather than the frame.
bool has_look(const std::string& name);

// ---- GPU shader generation -------------------------------------------------
//
// OCIO's GPU path is a shader *generator*, not a pixel processor. Once per transform it
// emits GLSL source plus the LUT textures and uniforms that source expects; per frame there
// is nothing to do but keep the uniforms current. That is why this belongs nowhere near an
// AVFrame, and why the previous attempt -- OCIO as a CPU filter in the producer's filter
// chain -- was the wrong shape rather than merely slow.

/// Which shading language to generate, and with it how resources are declared.
///
/// Not a cosmetic difference. GLSL 4.0 emits `uniform sampler2D name;` and leaves binding to
/// the application; Vulkan GLSL emits `layout(set=N, binding=M) uniform sampler2D name;` and
/// so decides its own bindings, which the pipeline layout then has to match. The two
/// generated programs are not interchangeable, and OCIO's cache ID leads with the language,
/// so a cache keyed on it cannot confuse them.
enum class gpu_target
{
    opengl, ///< GPU_LANGUAGE_GLSL_4_0, for the OGL mixer
    vulkan  ///< GPU_LANGUAGE_GLSL_VK_4_6 at descriptor set 1, textures from binding 1
};

/// Where each kind of transform's textures start in the Vulkan descriptor set.
///
/// A layer can carry an input transform and a display transform at once, spliced into the
/// same shader. Both would otherwise declare their first sampler at binding 1 and collide,
/// so they get disjoint ranges: **input 1..4, display 5..8**, inside the 8 bindings the
/// mixer's descriptor set 1 reserves. Binding 0 is OCIO's uniform buffer by its own
/// contract and belongs to neither.
///
/// The split is 4/4 because of what was measured across the pinned studio config: an input
/// transform emits at most **1** texture (55 colour spaces checked) and a display transform
/// at most **3** (all 41 display/view combinations checked). Both sides have headroom, and
/// exceeding a range is refused rather than silently overlapping.
///
/// Meaningless on the OpenGL target, which declares no bindings and takes texture units the
/// caller assigns.
constexpr unsigned INPUT_TEXTURE_BINDING_START   = 1;
constexpr unsigned DISPLAY_TEXTURE_BINDING_START = 5;

/// One LUT the generated shader samples. The application owns uploading it.
struct gpu_texture
{
    std::string sampler_name; ///< the sampler as declared in the generated source

    /// 1, 2 or 3 — and **1 really happens**. Display transforms declare `sampler1D` for
    /// their reach and gamut-cusp tables; input transforms never did, so a backend that
    /// treats "not 3D" as "2D" is wrong the moment a display transform is added. On Vulkan
    /// the image view type must match what the shader declares, so a 1D LUT needs an `e1D`
    /// image and view, not an Nx1 2D one.
    int         dimensions = 2;
    int         width      = 0;
    int         height     = 1;
    int         edge_len   = 0; ///< 3D only: values are edge_len^3 * 3
    int         channels   = 3; ///< 1 (red) or 3 (rgb)
    bool        interpolate_linear = true;

    /// The binding this sampler was declared at, from OCIO rather than inferred: the
    /// generated source hard-codes it, so a descriptor written anywhere else is read as
    /// whatever happens to be bound there. Meaningless on the OpenGL target, which declares
    /// no binding and takes a texture unit chosen by the caller.
    int binding = 0;

    std::vector<float> values;
};

/// Everything needed to compile and feed one generated transform.
struct gpu_shader
{
    /// OCIO's own identifier for this processor. Stable across cosmetic differences in code
    /// generation and sensitive to the config's contents, which is what makes it a better
    /// shader-cache key than a hash of the source text or the colour space name.
    std::string cache_id;

    /// Declarations plus the entry function, to be spliced ahead of the host shader's body.
    std::string source;

    /// Name of the generated entry point, as called from the splice site.
    std::string function_name;

    /// Bytes OCIO wants for the uniform buffer it declares at binding 0 of its descriptor
    /// set (Vulkan target only). **Zero for every input transform in the studio config** --
    /// measured across all 55 colour spaces, none has a dynamic property, so none declares
    /// a uniform block at all. The reserved binding stays declared and written anyway: a
    /// descriptor set may legally carry a binding the shader never reads, and a display
    /// transform with a dynamic exposure would need it.
    std::size_t uniform_buffer_size = 0;

    std::vector<gpu_texture> textures;
};

/// Build the GLSL for `source_space` -> the mixer's working space (ACEScg / ACES - ACEScg).
///
/// Returns nothing on failure, having logged why. Callers must treat that as a reason to
/// refuse a command, never to render without the transform.
///
/// Expensive: builds an OCIO processor and generates source. Call it when a configuration
/// changes, not per frame.
bool build_input_transform(const std::string& source_space,
                           gpu_shader&        out,
                           gpu_target         target = gpu_target::opengl);

/// Build the GLSL for the mixer's working space -> `display` / `view`.
///
/// The output half of an OCIO pipeline: tone mapping, gamut compression and the display's
/// own encoding, which together are what "an ACES look" means to an operator. It replaces
/// the mixer's built-in `working_to_output` matrix plus OETF, exactly as an input transform
/// replaces the built-in EOTF plus `input_to_working`.
///
/// Measured across all 41 display/view combinations in the pinned studio config: at most 3
/// textures, no 3D LUT, no dynamic uniform. So this needs nothing from the Vulkan side that
/// an input transform did not already need -- the same descriptor set 1, the same 2D image
/// views, the same unused uniform buffer at binding 0. The generated source is an order of
/// magnitude larger though (~16 KB against ~1.5 KB), which is paid at compile time.
///
/// Same failure contract as build_input_transform: nothing on failure, having logged why,
/// and a caller must refuse the command rather than render without it.
/// `looks` is an optional LMT applied in the working space BEFORE the display rendering,
/// composed into the same processor rather than spliced separately. Two reasons that
/// matters: the variant cache key stays the (input, output) pair it already is, and OCIO
/// gets to optimise the look and the view together instead of us emitting two programs.
/// Empty means no look, and generates byte-for-byte what this generated before it existed.
///
/// The string is OCIO's look *expression*, so `-name` inverts a look and a comma-separated
/// list applies several in order. It is passed through rather than parsed here.
bool build_display_transform(const std::string& display,
                             const std::string& view,
                             gpu_shader&        out,
                             gpu_target         target = gpu_target::opengl,
                             const std::string& looks  = "");

}}} // namespace caspar::accelerator::ocio
