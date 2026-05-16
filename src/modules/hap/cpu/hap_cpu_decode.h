/*
 * Copyright (c) 2025 CasparCG Contributors
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

// hap_cpu_decode.h
// CPU-side BCn block decompression for HAP frames.
// Uses bcdec library to decompress DXT/BC blocks to BGRA pixels.
// This path is used when the GPU-accelerated GL/VK paths are unavailable.
// ---------------------------------------------------------------------------
#pragma once

#include "../util/hap_frame_parser.h"

#include <cstdint>
#include <vector>

namespace caspar { namespace hap {

/// Decompress a HAP frame's DXT/BC data to packed BGRA8 pixels.
///
/// @param variant        The HAP variant (determines BC format + color space)
/// @param texture_data   Primary compressed texture data (BC1/BC3/BC4/BC7)
/// @param texture_size   Size of primary texture data in bytes
/// @param alpha_data     Alpha texture data (BC4, only for HapQAlpha), nullptr if N/A
/// @param alpha_size     Size of alpha texture data
/// @param width          Image width in pixels (must be multiple of 4)
/// @param height         Image height in pixels (must be multiple of 4)
/// @param out_pixels     Output: BGRA8 pixel buffer (width * height * 4 bytes)
///
/// @returns true on success, false on unsupported variant or invalid data
bool cpu_decode_hap_to_bgra(HapVariant             variant,
                            const uint8_t*         texture_data,
                            size_t                 texture_size,
                            const uint8_t*         alpha_data,
                            size_t                 alpha_size,
                            int                    width,
                            int                    height,
                            std::vector<uint8_t>&  out_pixels);

}} // namespace caspar::hap
