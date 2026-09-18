/*
 * Copyright (c) contributors to the CasparCG project
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

#include <common/param.h>

namespace caspar { namespace core {

/// Does this source's RGB carry a factor of its own alpha already, or not?
///
/// `pixel_format_desc::is_straight_alpha` answers that, the shader premultiplies when it
/// is set, and `blend()` requires premultiplied input. The field defaults to `false` —
/// "already premultiplied" — and **no call site in this tree or upstream ever set it
/// true**, so for every decoded file the premultiply was skipped and the blend was handed
/// straight RGB.
///
/// What that cost, measured 2026-08-20 on a 1080p qtrle clip with alpha strips at
/// 255/192/128/64/0 over an opaque background, both mixers:
///
///   * The composite matched `c + bg*(1-a)` on **100.0%** of 1 152 000 partial-alpha
///     pixels (mean 0.03, max 0.2) where the correct `c*a + bg*(1-a)` sat 43.05 away. At
///     alpha **0** the layer rendered as `[101,101,255]` against a background of
///     `[0,0,191]`: a fully transparent region showing as opaque colour ADDED to what was
///     behind it.
///   * The IMAGE consumer's un-premultiply then had nothing to undo, so 41.4% of
///     partial-alpha components clipped at white.
///
/// It survived because `col.rgb *= 1.0` is a no-op: content with no partial alpha is
/// completely unaffected, and until a fixture carried real partial alpha there was nothing
/// to see. HTML is unaffected for a different reason — CEF hands over genuinely
/// premultiplied BGRA, so `false` is the truth there and stays.
///
/// THE DEFAULT IS PREMULTIPLIED, and it is configurable. Nothing in ProRes 4444, QuickTime
/// Animation, Hap or NotchLC declares the mode, and both populations are real: After
/// Effects and this server's own recordings are premultiplied, Premiere and Media Encoder
/// are straight. Detection cannot settle it either -- a dark graphic satisfies
/// `rgb <= alpha` whichever convention it is -- so it is declared, never guessed.
///
/// Measured 2026-09-18, a premultiplied qtrle recorded out of this server and played back:
/// under the old straight default its partial-alpha pixels were wrong by mean 17.3 / max 53
/// codes, because the mixer's premultiplied output was premultiplied a second time. The
/// round trip closes at mean 0.58 with the default below.
///
///     <decode-alpha-mode>default [default|straight|premultiplied]</decode-alpha-mode>
///
///     PLAY 1-1 "clip"                    the configured default
///     PLAY 1-1 "clip" PREMULTIPLIED      the RGB already carries its alpha
///     PLAY 1-1 "clip" STRAIGHT           it does not; premultiply it before the blend
///
/// This matches upstream's `decode-alpha-mode` (CasparCG/server#1800) so the two do not
/// diverge, and the shape other systems use: Resolume's per-clip "Alpha Type" defaults to
/// premultiplied for the same reason, Disguise makes it a per-layer blend mode.
///
/// The convention to fall back on: `configuration.decode-alpha-mode`, premultiplied unless
/// it says otherwise. Defined in alpha_mode.cpp so the config is read, and warned about,
/// exactly once for all four producers that share this.
bool configured_default_straight();

template <class C>
bool source_is_straight_alpha(C&& params)
{
    if (contains_param(L"PREMULTIPLIED", params))
        return false;
    if (contains_param(L"STRAIGHT", params))
        return true;
    return configured_default_straight();
}

/// What the OPERATOR said, kept separate from what we would assume.
enum class alpha_declaration
{
    unspecified = 0, //< nothing was said; a source that knows may speak, else the convention
    straight,
    premultiplied,
};

/// The same three keywords, read as three answers rather than two.
///
/// `source_is_straight_alpha` collapses "nothing was said" into the configured default,
/// which is right for a producer with nothing better to consult. It is wrong for one that has: FFmpeg 8
/// carries `AVFrame.alpha_mode`, and a PNG or EXR or alpha-tagged Matroska now DECLARES its
/// mode. Collapsing first would let the fallback silently outrank the file.
///
/// So the precedence is: the operator, then the file, then the convention. The operator
/// stays on top because the override exists for content the file is wrong about -- some
/// Adobe ProRes 4444 exports -- and ProRes carries no declaration anyway.
template <class C>
alpha_declaration source_alpha_declaration(C&& params)
{
    if (contains_param(L"PREMULTIPLIED", params))
        return alpha_declaration::premultiplied;
    if (contains_param(L"STRAIGHT", params))
        return alpha_declaration::straight;
    return alpha_declaration::unspecified;
}

}} // namespace caspar::core
