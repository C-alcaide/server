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

#include <core/monitor/monitor.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace caspar { namespace core { namespace scopes {

/// THE FOUR REDUCTIONS BOTH MIXERS COMPUTE, in one compute pass over the finished frame, into
/// one buffer. DECISIONS L163/L166: a scope is a data visualisation of a picture that is
/// already in VRAM, so the reduction belongs on the GPU and only the RESULT is read back.
///
/// The layout is written down ONCE, here, because it is shared by two shaders, two readback
/// paths and the client-facing images -- and a second copy of an offset is a defect waiting for
/// a backend to disagree with itself.
///
///     histogram   3 x 256                  counts per 8-bit code, whole frame, per channel
///     waveform    256 columns x 256 levels counts of Rec.709 luma per output column
///     parade      3 x (256 x 256)          the same, per colour channel: R, G, B
///     vectorscope 256 x 256                a 2-D histogram over (Cb, Cr), Rec.709
///
/// A column is `x * 256 / width`, a level is the 8-bit code, so on a 1920-wide frame each
/// column is 7 or 8 pixels and every level is exactly one code. That makes a flat colour patch
/// an EXACT expectation rather than an approximate one -- every pixel of a column lands in one
/// cell -- which is what lets the battery gate these at no tolerance at all.
constexpr std::size_t histogram_bins    = 256;
constexpr std::size_t histogram_counts  = 3 * histogram_bins;
constexpr int         waveform_columns  = 256;
constexpr int         waveform_levels   = 256;
constexpr int         vector_size       = 256;

constexpr std::size_t waveform_cells = static_cast<std::size_t>(waveform_columns) * waveform_levels;
constexpr std::size_t vector_cells   = static_cast<std::size_t>(vector_size) * vector_size;

constexpr std::size_t histogram_offset = 0;
constexpr std::size_t waveform_offset  = histogram_offset + histogram_counts;
constexpr std::size_t parade_offset    = waveform_offset + waveform_cells;
constexpr std::size_t vector_offset    = parade_offset + 3 * waveform_cells;
constexpr std::size_t counts_total     = vector_offset + vector_cells;

/// One scope as an image a client can draw directly: BGRA, which is what
/// `image::encode_png_bgra8` takes and what both mixers' readbacks already produce, so nothing
/// on this path converts twice.
struct grid
{
    int                       width  = 0;
    int                       height = 0;
    std::vector<std::uint8_t> bgra;

    bool empty() const { return bgra.empty(); }
};

/// The kinds a client can ask for by name, so the HTTP route and the battery share one
/// vocabulary rather than two spellings of it.
inline bool is_scope_kind(const std::string& kind)
{
    return kind == "waveform" || kind == "parade" || kind == "vectorscope";
}

/// THE COMPUTE SHADER, GENERATED HERE so the offsets above are the only place they exist.
///
/// Both backends run the same body: OpenGL compiles this text directly, Vulkan runs it through
/// `glsl_compiler` to SPIR-V. They differ in two tokens and nothing else -- the layout
/// qualifiers, and the swizzle, because the Vulkan mixer's images carry BGRA bytes under an
/// RGBA format label while the OpenGL readback does not. That difference is the repository's
/// most-repeated defect (see CLAUDE.md, the channel-order trap) and it is spelled out at the
/// one call site each rather than hidden in a copy of the shader.
///
/// The conversions are Rec.709, from ITU-R BT.709-6 Table 3: luma is
/// `0.2126 R + 0.7152 G + 0.0722 B`, and Cb/Cr follow from it as `(B - Y) / 1.8556` and
/// `(R - Y) / 1.5748`, expanded here into the dot products they reduce to. The values are the
/// DISPLAY-ENCODED ones the consumer receives, which is what a scope shows.
inline std::string compute_glsl(const std::string& image_layout,
                                const std::string& buffer_layout,
                                const std::string& swizzle)
{
    const auto u = [](std::size_t v) { return std::to_string(v) + "u"; };
    return
        "#version 450\n"
        "layout(local_size_x = 16, local_size_y = 16) in;\n"
        "layout(" + image_layout + ") uniform sampler2D img;\n"
        "layout(std430, " + buffer_layout + ") buffer Counts { uint c[" + std::to_string(counts_total) + "]; };\n"
        "void main()\n"
        "{\n"
        "    ivec2 sz = textureSize(img, 0);\n"
        "    ivec2 p  = ivec2(gl_GlobalInvocationID.xy);\n"
        "    if (p.x >= sz.x || p.y >= sz.y)\n"
        "        return;\n"
        "    vec3  rgb = clamp(texelFetch(img, p, 0)." + swizzle + ", 0.0, 1.0);\n"
        "    uvec3 b   = uvec3(rgb * 255.0 + 0.5);\n"
        "    atomicAdd(c[" + u(histogram_offset) + " + b.r], 1u);\n"
        "    atomicAdd(c[" + u(histogram_offset + histogram_bins) + " + b.g], 1u);\n"
        "    atomicAdd(c[" + u(histogram_offset + 2 * histogram_bins) + " + b.b], 1u);\n"
        "    uint col = min(uint(p.x) * " + u(waveform_columns) + " / uint(sz.x), " +
            u(waveform_columns - 1) + ");\n"
        "    uint lv  = uint(clamp(dot(rgb, vec3(0.2126, 0.7152, 0.0722)), 0.0, 1.0) * 255.0 + 0.5);\n"
        "    uint base = col * " + u(waveform_levels) + ";\n"
        "    atomicAdd(c[" + u(waveform_offset) + " + base + lv], 1u);\n"
        "    atomicAdd(c[" + u(parade_offset) + " + base + b.r], 1u);\n"
        "    atomicAdd(c[" + u(parade_offset + waveform_cells) + " + base + b.g], 1u);\n"
        "    atomicAdd(c[" + u(parade_offset + 2 * waveform_cells) + " + base + b.b], 1u);\n"
        "    float cb = dot(rgb, vec3(-0.1146, -0.3854, 0.5));\n"
        "    float cr = dot(rgb, vec3(0.5, -0.4542, -0.0458));\n"
        "    uint  vb = uint(clamp(cb + 0.5, 0.0, 1.0) * " + std::to_string(vector_size - 1) + ".0 + 0.5);\n"
        "    uint  vr = uint(clamp(cr + 0.5, 0.0, 1.0) * " + std::to_string(vector_size - 1) + ".0 + 0.5);\n"
        "    atomicAdd(c[" + u(vector_offset) + " + vr * " + u(vector_size) + " + vb], 1u);\n"
        "}\n";
}

namespace detail {

/// Counts to intensity. LOG, because that is what every scope does and for the reason they all
/// do it: a frame of flat colour puts a whole column into one cell and everything a grade is
/// judged by -- the few pixels at the top of the highlights, the tail into the shadows -- is
/// three orders of magnitude below it and would be invisible under a linear map.
///
/// Normalised against the WHOLE grid's peak rather than per column, so the relative weight of
/// one column against another survives; a per-column normalisation makes an empty sky as bright
/// as a face. The peak is published beside the image so a client can recover counts.
inline std::uint8_t intensity(std::uint32_t count, double inv_log_peak)
{
    if (count == 0)
        return 0;
    const double v = std::log1p(static_cast<double>(count)) * inv_log_peak;
    return static_cast<std::uint8_t>(std::clamp(v, 0.0, 1.0) * 255.0 + 0.5);
}

inline double inverse_log_peak(const std::uint32_t* first, std::size_t n)
{
    std::uint32_t peak = 0;
    for (std::size_t i = 0; i < n; ++i)
        peak = std::max(peak, first[i]);
    return peak == 0 ? 0.0 : 1.0 / std::log1p(static_cast<double>(peak));
}

inline void put(grid& g, int x, int y, std::uint8_t b, std::uint8_t gr, std::uint8_t r)
{
    const auto o   = (static_cast<std::size_t>(y) * g.width + x) * 4;
    g.bgra[o + 0]  = b;
    g.bgra[o + 1]  = gr;
    g.bgra[o + 2]  = r;
    g.bgra[o + 3]  = 255;
}

inline grid blank(int w, int h)
{
    grid g;
    g.width  = w;
    g.height = h;
    g.bgra.assign(static_cast<std::size_t>(w) * h * 4, 0);
    for (std::size_t i = 3; i < g.bgra.size(); i += 4)
        g.bgra[i] = 255;
    return g;
}

} // namespace detail

/// The luma waveform: x is the output column, y is the code value with **0 at the BOTTOM**,
/// which is the orientation every scope in the industry uses and the opposite of an image
/// raster's. A client drawing this as a texture gets a waveform the right way up with no
/// transform of its own.
inline grid waveform_grid(const std::vector<std::uint32_t>& counts)
{
    if (counts.size() != counts_total)
        return {};
    const auto* first = counts.data() + waveform_offset;
    const auto  inv   = detail::inverse_log_peak(first, waveform_cells);
    auto        g     = detail::blank(waveform_columns, waveform_levels);
    for (int col = 0; col < waveform_columns; ++col) {
        for (int level = 0; level < waveform_levels; ++level) {
            const auto v = detail::intensity(first[static_cast<std::size_t>(col) * waveform_levels + level], inv);
            detail::put(g, col, waveform_levels - 1 - level, v, v, v);
        }
    }
    return g;
}

/// The RGB parade, as ONE image with the three waveforms in the three colour planes rather than
/// three images side by side. That is not a shortcut: a parade IS the three channels' waveforms
/// on one axis, and a client can draw this single texture as three panels, as an overlay, or
/// one plane at a time, where three separate images force the choice here. The peak is shared
/// across all three planes so their heights stay comparable.
inline grid parade_grid(const std::vector<std::uint32_t>& counts)
{
    if (counts.size() != counts_total)
        return {};
    const auto* first = counts.data() + parade_offset;
    const auto  inv   = detail::inverse_log_peak(first, 3 * waveform_cells);
    auto        g     = detail::blank(waveform_columns, waveform_levels);
    for (int col = 0; col < waveform_columns; ++col) {
        for (int level = 0; level < waveform_levels; ++level) {
            const auto cell = static_cast<std::size_t>(col) * waveform_levels + level;
            const auto r    = detail::intensity(first[0 * waveform_cells + cell], inv);
            const auto gr   = detail::intensity(first[1 * waveform_cells + cell], inv);
            const auto b    = detail::intensity(first[2 * waveform_cells + cell], inv);
            detail::put(g, col, waveform_levels - 1 - level, b, gr, r);
        }
    }
    return g;
}

/// The vectorscope: a 2-D histogram over (Cb, Cr) with **Cb to the right and Cr UP**, again the
/// industry orientation, so 75% bars land on the graticule targets a client draws over it.
inline grid vectorscope_grid(const std::vector<std::uint32_t>& counts)
{
    if (counts.size() != counts_total)
        return {};
    const auto* first = counts.data() + vector_offset;
    const auto  inv   = detail::inverse_log_peak(first, vector_cells);
    auto        g     = detail::blank(vector_size, vector_size);
    for (int cr = 0; cr < vector_size; ++cr) {
        for (int cb = 0; cb < vector_size; ++cb) {
            const auto v = detail::intensity(first[static_cast<std::size_t>(cr) * vector_size + cb], inv);
            detail::put(g, cb, vector_size - 1 - cr, v, v, v);
        }
    }
    return g;
}

inline grid grid_for(const std::string& kind, const std::vector<std::uint32_t>& counts)
{
    if (kind == "waveform")
        return waveform_grid(counts);
    if (kind == "parade")
        return parade_grid(counts);
    if (kind == "vectorscope")
        return vectorscope_grid(counts);
    return {};
}

/// Fold the HISTOGRAM into the tree a channel publishes under `mixer/scopes/`:
///
///     scopes/histogram/r   256 counts          scopes/pixels   width * height
///     scopes/histogram/g   256 counts          scopes/bins     256
///     scopes/histogram/b   256 counts          scopes/images   the pullable kinds
///
/// **Only the histogram.** The other three are 80 times the data and are PULLED as images
/// (`GET /v1/scopes/{ch}/{kind}`) rather than published, which is L166's "decide the bulk-data
/// rate before building" answered by not choosing a rate at all: a client reads them when it
/// draws, an idle client costs nothing, and the per-tick tree does not grow by one leaf.
inline monitor::state fold_histogram(const std::vector<std::uint32_t>& counts, int width, int height)
{
    monitor::state s;
    if (counts.size() < histogram_counts)
        return s;
    const char* names[3] = {"r", "g", "b"};
    for (std::size_t c = 0; c < 3; ++c) {
        monitor::vector_t bins;
        bins.reserve(histogram_bins);
        for (std::size_t i = 0; i < histogram_bins; ++i)
            bins.push_back(static_cast<std::int32_t>(counts[histogram_offset + c * histogram_bins + i]));
        s["histogram"][names[c]] = std::move(bins);
    }
    s["pixels"] = static_cast<std::int64_t>(width) * static_cast<std::int64_t>(height);
    s["bins"]   = static_cast<std::int32_t>(histogram_bins);
    s["images"] = std::string("waveform,parade,vectorscope");
    return s;
}

}}} // namespace caspar::core::scopes
