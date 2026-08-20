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
 *
 * Author: Robert Nagy, ronag89@gmail.com
 */

#pragma once

#include <vector>

#include <common/bit_depth.h>

namespace caspar { namespace core {

enum class pixel_format
{
    gray = 0,
    bgra,
    rgba,
    argb,
    abgr,
    ycbcr,
    ycbcra,
    luma,
    bgr,
    rgb,
    uyvy,
    gbrp,  // planar
    gbrap, // planar
    ycocg_dxt5,  // YCoCg-DXT5 (HAP Q) — BC3 hardware-decoded then shader YCoCg→RGB
    ycocg_dxt5a, // YCoCg-DXT5 + separate alpha (HAP Q Alpha) — two BC3 planes
    // Semi-planar YCbCr as hardware decoders produce it: plane 0 = Y (1 component),
    // plane 1 = Cb/Cr interleaved at half resolution (2 components). Covers NV12
    // (8-bit) and P010/P016 (16-bit) — the only difference is the plane depth, and
    // for P010 the 10 significant bits already sit in the high bits of each 16-bit
    // word, so no rescaling is needed.
    //
    // New values must be APPENDED: the mixer shaders switch on this enum's numeric
    // value, so inserting one silently reinterprets every format after it.
    nv12,
    count,
    invalid,
};

enum class color_space
{
    bt601,
    bt709,
    bt2020,
    p3_d65,    // DCI-P3 with D65 white point (Display P3)
    p3_dci,    // DCI-P3 with DCI white point (cinema)
    adobe_rgb, // Adobe RGB (1998)

    /// The source did not DECLARE a colour space. `decode_color_space` below resolves it
    /// for the YCbCr matrix; everywhere it is read as a source GAMUT it falls through to
    /// bt709, which is what the field defaulted to before this value existed.
    ///
    /// Must stay last. Both kernels index their matrix tables with this enum's numeric
    /// value, clamped as `> 2 ? 1 : value`, so an `unknown` that leaked would silently
    /// decode as BT.709 rather than reading past the table.
    unknown,
};

/// Whether the source's YCbCr codes use the full code range or studio swing.
///
/// Defaults to `limited` everywhere, which is what broadcast material is and what every
/// producer that does not know its range should say. `full` is the JPEG convention: black at
/// code 0 and white at code 255 rather than 16 and 235, so decoding it with the studio-swing
/// offsets expands it a second time -- measured at 55 where 64 was correct, a 14% error, with
/// blacks crushed below 0 and whites clipped above 255.
enum class color_range
{
    limited,
    full,
};

enum class color_transfer
{
    sdr,
    pq,
    hlg,
    linear,    // Linear light (no curve)
    gamma24,   // Pure gamma 2.4 (EBU broadcast reference)
    gamma26,   // Pure gamma 2.6 (DCI cinema projection)
};

enum class chroma_location
{
    unspecified,
    left,       // MPEG-2/4, H.264 default for 4:2:0/4:2:2
    center,     // MPEG-1, JPEG
    topleft,    // Co-sited with top-left luma sample
};

struct pixel_format_desc final
{
    struct plane
    {
        int               linesize = 0;
        int               width    = 0;
        int               height   = 0;
        int               size     = 0;
        int               stride   = 0;
        common::bit_depth depth    = common::bit_depth::bit8;

        /// Index of an earlier plane whose host bytes this plane shares, or -1 for
        /// its own buffer.
        ///
        /// Some packed formats are described to the mixer as several planes over the
        /// same bytes so the shader can sample them at different rates -- UYVY is one
        /// buffer presented as a full-rate 2-component luma view and a half-rate
        /// 4-component chroma view. Without this the frame factory allocated a staging
        /// buffer per plane and the producer memcpy'd the identical bytes into each,
        /// doubling the host cost of every UYVY frame (decklink and NDI input, and
        /// ffmpeg software decode).
        ///
        /// Only the host buffer is shared. Each plane still gets its own texture.
        int alias_of = -1;

        plane() = default;

        plane(int width, int height, int stride, common::bit_depth depth = common::bit_depth::bit8)
            : linesize(width * stride * (depth == common::bit_depth::bit8 ? 1 : 2))
            , width(width)
            , height(height)
            , size(width * height * stride * (depth == common::bit_depth::bit8 ? 1 : 2))
            , stride(stride)
            , depth(depth)
        {
        }
    };

    pixel_format_desc() = default;

    explicit pixel_format_desc(pixel_format          format,
                               core::color_space     color_space    = core::color_space::unknown,
                               core::color_transfer  color_transfer = core::color_transfer::sdr)
        : format(format)
        , color_space(color_space)
        , color_transfer(color_transfer)
    {
    }

    pixel_format          format            = pixel_format::invalid;
    bool                  is_straight_alpha = false;
    std::vector<plane>    planes;
    core::color_space     color_space      = core::color_space::unknown;
    core::color_transfer  color_transfer   = core::color_transfer::sdr;
    core::chroma_location chroma_location  = core::chroma_location::unspecified;
    /// Studio swing unless a producer says otherwise. Set post-construction, like
    /// `color_transfer` and `chroma_location`, so no existing call site changes.
    core::color_range     color_range      = core::color_range::limited;
};

inline bool operator==(const pixel_format_desc::plane& lhs, const pixel_format_desc::plane& rhs)
{
    return lhs.linesize == rhs.linesize && lhs.width == rhs.width && lhs.height == rhs.height &&
           lhs.size == rhs.size && lhs.stride == rhs.stride && lhs.depth == rhs.depth &&
           lhs.alias_of == rhs.alias_of;
}

inline bool operator!=(const pixel_format_desc::plane& lhs, const pixel_format_desc::plane& rhs)
{
    return !(lhs == rhs);
}

/// Every field is compared. Used by the mixers' still-frame cache: a field left
/// out here is a field whose change would not invalidate the cache, which shows
/// up on air as a frozen frame.
inline bool operator==(const pixel_format_desc& lhs, const pixel_format_desc& rhs)
{
    return lhs.format == rhs.format && lhs.is_straight_alpha == rhs.is_straight_alpha &&
           lhs.color_space == rhs.color_space && lhs.color_transfer == rhs.color_transfer &&
           lhs.chroma_location == rhs.chroma_location && lhs.color_range == rhs.color_range &&
           lhs.planes == rhs.planes;
}

inline bool operator!=(const pixel_format_desc& lhs, const pixel_format_desc& rhs) { return !(lhs == rhs); }

/// The matrix to decode this source's chroma with: whatever it declared, or a convention
/// when it declared nothing. Both mixers must pick the same one, so it lives here rather
/// than once per backend — the two copies of this rule had already drifted once.
///
/// Untagged sub-720 material is conventionally BT.601, and honouring that is right. What
/// was wrong was applying it as an OVERRIDE: `height > 700 ? declared : bt601` discarded
/// correct metadata, so a 960x540 clip tagged BT.709 — ordinary LED-panel content — was
/// matrixed as BT.601 on any channel. A 601/709 mismatch is ~12 dB PSNR: a visible hue
/// shift on saturated colour, invisible on greys, which is why no ramp-based check caught
/// it. Nothing downstream can repair it either; the matrix is applied in `ycbcra_to_rgba`
/// at texture-fetch time, before the colour-management block, so `auto-color-convert` and
/// `MIXER COLORSPACE` both act too late.
///
/// Nothing about the DESTINATION enters into it, and that is deliberate. This fork used to
/// carry a `target_is_custom_format` term that defeated the convention on a custom video
/// mode, reasoning that an LED wall's small raster is a panel size rather than SD broadcast
/// content. It was wrong twice over. Upstream rejected it (CasparCG/server#1775) because a
/// file's encoding matrix cannot depend on where it is shown — measured, an untagged BT.601
/// clip on a custom raster decoded as BT.709, i.e. wrongly, at 0.54 LSB against the model.
/// And it made this rule unimplementable outside the mixer: `write_frame_png` has no channel
/// to ask, so PRINT RAW decoded with BT.709 while the composite used BT.601, a disagreement
/// of up to 27.9 LSB on saturated colour. With the term gone the only input is the source
/// descriptor, so every path that resolves a decode matrix agrees by construction.
inline color_space decode_color_space(const pixel_format_desc& desc)
{
    if (desc.color_space != color_space::unknown) {
        return desc.color_space;
    }
    return desc.planes.at(0).height > 700 ? color_space::bt709 : color_space::bt601;
}

/// The source GAMUT, for the colour-management chain rather than the chroma decode.
///
/// A separate resolution from `decode_color_space` on purpose: the SD convention is about
/// which matrix encoded the chroma and says nothing about primaries, so an undeclared
/// source is BT.709 here at every raster. That is what the field defaulted to before
/// `unknown` existed, which is what keeps this behaviour-preserving.
inline color_space source_gamut(const pixel_format_desc& desc)
{
    return desc.color_space == color_space::unknown ? color_space::bt709 : desc.color_space;
}

}} // namespace caspar::core
