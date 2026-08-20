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
/// STRAIGHT IS THE DEFAULT FOR DECODED MEDIA, because that is what the formats store:
/// FFmpeg has no premultiplied pixel format for these, and every NLE writes ProRes 4444,
/// QuickTime Animation, NotchLC and PNG with straight alpha. `docs/COLOR_GRADING.md` has
/// documented this as the intended behaviour all along — *"Premultiply if the source is
/// straight (default)"* — it was only ever the flag that went unset.
///
/// The override exists because the assumption is about a convention rather than a signal
/// in the file, and a premultiplied render does occasionally reach us — some Adobe
/// ProRes 4444 exports among them. There is nothing in the container to detect it with,
/// so it is the operator's to declare:
///
///     PLAY 1-1 "clip"                    straight (default, correct for anything an NLE wrote)
///     PLAY 1-1 "clip" PREMULTIPLIED      the RGB already carries its alpha
///     PLAY 1-1 "clip" STRAIGHT           explicit, for a config that wants to say so
///
/// `PREMULTIPLIED` also restores the pre-2026-08-20 rendering exactly, which is the
/// escape hatch for anyone whose content was authored against the old behaviour.
template <class C>
bool source_is_straight_alpha(C&& params, bool default_straight = true)
{
    if (contains_param(L"PREMULTIPLIED", params))
        return false;
    if (contains_param(L"STRAIGHT", params))
        return true;
    return default_straight;
}

}} // namespace caspar::core
