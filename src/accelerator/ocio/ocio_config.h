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

// ---- GPU shader generation -------------------------------------------------
//
// OCIO's GPU path is a shader *generator*, not a pixel processor. Once per transform it
// emits GLSL source plus the LUT textures and uniforms that source expects; per frame there
// is nothing to do but keep the uniforms current. That is why this belongs nowhere near an
// AVFrame, and why the previous attempt -- OCIO as a CPU filter in the producer's filter
// chain -- was the wrong shape rather than merely slow.

/// One LUT the generated shader samples. The application owns uploading it.
struct gpu_texture
{
    std::string sampler_name; ///< the sampler as declared in the generated source
    int         dimensions = 2; ///< 1, 2 or 3
    int         width      = 0;
    int         height     = 1;
    int         edge_len   = 0; ///< 3D only: values are edge_len^3 * 3
    int         channels   = 3; ///< 1 (red) or 3 (rgb)
    bool        interpolate_linear = true;

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

    std::vector<gpu_texture> textures;
};

/// Build the GLSL for `source_space` -> the mixer's working space (ACEScg / ACES - ACEScg).
///
/// Returns nothing on failure, having logged why. Callers must treat that as a reason to
/// refuse a command, never to render without the transform.
///
/// Expensive: builds an OCIO processor and generates source. Call it when a configuration
/// changes, not per frame.
bool build_input_transform(const std::string& source_space, gpu_shader& out);

}}} // namespace caspar::accelerator::ocio
