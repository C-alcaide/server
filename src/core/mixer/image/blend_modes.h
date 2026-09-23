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

#include <optional>

#include <string>

namespace caspar { namespace core {

enum class blend_mode
{
    normal = 0,
    lighten,
    darken,
    multiply,
    average,
    add,
    subtract,
    difference,
    negation,
    exclusion,
    screen,
    overlay,
    soft_light,
    hard_light,
    color_dodge,
    color_burn,
    linear_dodge,
    linear_burn,
    linear_light,
    vivid_light,
    pin_light,
    hard_mix,
    reflect,
    glow,
    phoenix,
    contrast,
    saturation,
    color,
    luminosity,
    mix,
    blend_mode_count
};

/// A NAME THIS SERVER CAN ACTUALLY COMPOSITE, or nothing.
///
/// `std::optional`, not a `blend_mode`, and that is the fix. It returned `blend_mode::normal`
/// for any name it did not recognise -- and `normal` is also the legitimate way to turn blending
/// OFF, so a typo was indistinguishable from an operator clearing the mode and `MIXER <ch> BLEND
/// bannana` answered `202 MIXER OK`. This tree's most repeated defect, in the one shape the
/// 2026-09-19 sweep could not see: no `if/else if` chain over `boost::iequals` with a missing
/// `else`, just a bare `return` at the bottom of a list of `if`s.
///
/// `mix` IS DELIBERATELY NOT ACCEPTED. It is the last enumerator of `blend_mode` and it is
/// implemented by neither shader, so it composited as `normal` while the control API -- whose
/// value list is generated from the same enum -- accepted it and read it back as "mix". A name
/// that three facades disagree about is worse than one that does not exist, so both refuse it
/// now and `BLEND_MODES` in `transform_fields.cpp` omits it. The enumerator stays where it is,
/// because removing it would renumber `blend_mode_count` and every value above it.
std::optional<blend_mode> get_blend_mode(const std::wstring& str);
std::wstring              get_blend_mode(blend_mode mode);

/// Every name `get_blend_mode` accepts, comma-separated, for an error message that tells the
/// operator what to type instead of only that they were wrong.
std::wstring blend_mode_names();

}} // namespace caspar::core