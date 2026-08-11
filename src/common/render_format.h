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

#include <cstdint>

namespace caspar { namespace common {

/// The numeric format of a mixer *render target*, as distinct from bit_depth.
///
/// bit_depth describes storage width and is shared with CPU frame buffers, which are fed
/// from integer AVFrames and must stay integer. This enum describes only what the mixer
/// composites into, which is a separate question:
///
///   unorm  - normalized integer, the historical and default behaviour. Values are clamped
///            to [0,1] on write, so super-white and negative channel values are destroyed.
///            Precision is uniform: 1/65535 everywhere at 16 bits.
///
///   fp16   - half float. Carries negatives and values above 1.0, which is what a
///            scene-referred ACEScg working space requires -- AP0/AP1 conversions produce
///            negatives routinely and linear scene data exceeds 1.0 by design.
///
/// fp16 is NOT a uniform upgrade over unorm16: near 1.0 its ulp is 4.88e-4 against
/// unorm16's 1.53e-5, so it is ~32x coarser in the highlights, crossing over at about
/// 1/64 of full scale and becoming finer below that. It is the correct format for a
/// *linear* buffer, where relative precision is what matters, and the wrong one for a
/// display-encoded buffer, where the OETF has already redistributed precision.
///
/// So the format follows the working space, and display-referred channels stay on unorm
/// and remain bit-identical to before this enum existed. See
/// docs/OCIO_INTEGRATION_STUDY.md section 4.3.
enum class render_format : uint8_t
{
    unorm = 0,
    fp16,
};

inline bool is_float(render_format f) { return f == render_format::fp16; }

}} // namespace caspar::common
