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
                               core::color_space     color_space    = core::color_space::bt709,
                               core::color_transfer  color_transfer = core::color_transfer::sdr)
        : format(format)
        , color_space(color_space)
        , color_transfer(color_transfer)
    {
    }

    pixel_format          format            = pixel_format::invalid;
    bool                  is_straight_alpha = false;
    std::vector<plane>    planes;
    core::color_space     color_space      = core::color_space::bt709;
    core::color_transfer  color_transfer   = core::color_transfer::sdr;
    core::chroma_location chroma_location  = core::chroma_location::unspecified;

    /// Did the source actually SAY what its colour space is, or is `color_space`
    /// just the default?
    ///
    /// `color_space` defaults to bt709, so a genuinely BT.709-tagged frame and a frame
    /// that declared nothing are indistinguishable without this. The mixers need to
    /// tell them apart: untagged sub-720 material is conventionally BT.601, but a file
    /// that explicitly says BT.709 must be honoured whatever its size.
    ///
    /// Before this flag existed the kernels applied the SD convention unconditionally —
    /// `height > 700 ? declared : bt601` — which silently discarded correct metadata
    /// for every sub-720 YCbCr source. Measured on the SDI rig, a 601/709 mismatch is
    /// ~12 dB PSNR: a visible hue shift on saturated colour, and invisible on greys,
    /// which is why it survived every ramp-based check. LED-panel content authored at
    /// odd small sizes (960x540, 1024x640) and tagged BT.709 was the case that made it
    /// matter.
    ///
    /// Producers that know set this; a producer that does not is treated exactly as it
    /// was before, so leaving one alone cannot regress it.
    bool color_space_specified = false;
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
           lhs.chroma_location == rhs.chroma_location &&
           lhs.color_space_specified == rhs.color_space_specified && lhs.planes == rhs.planes;
}

inline bool operator!=(const pixel_format_desc& lhs, const pixel_format_desc& rhs) { return !(lhs == rhs); }

}} // namespace caspar::core
