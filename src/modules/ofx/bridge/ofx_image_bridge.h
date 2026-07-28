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

namespace caspar { namespace ofx {

/// Convert an 8-bit BGRA image (CasparCG mixer format, top-down rows) into an 8-bit RGBA
/// image (OpenFX, bottom-up rows). Swaps the R/B channels and flips vertically so the OFX
/// bottom-left origin matches CasparCG's top-left origin.
///
///   src        : source BGRA pixels, top row first
///   src_stride : bytes per source row
///   dst        : destination RGBA buffer, laid out bottom row first
///   dst_stride : bytes per destination row
///   width      : image width in pixels
///   height     : image height in pixels
void bgra_top_down_to_rgba_bottom_up(const std::uint8_t* src,
                                     int                 src_stride,
                                     std::uint8_t*       dst,
                                     int                 dst_stride,
                                     int                 width,
                                     int                 height);

/// Inverse of bgra_top_down_to_rgba_bottom_up: convert an 8-bit RGBA bottom-up OFX image
/// back to an 8-bit BGRA top-down CasparCG image.
void rgba_bottom_up_to_bgra_top_down(const std::uint8_t* src,
                                     int                 src_stride,
                                     std::uint8_t*       dst,
                                     int                 dst_stride,
                                     int                 width,
                                     int                 height);

/// RGBA source variant: convert an 8-bit RGBA image (CasparCG RGBA-native mixer, top-down rows)
/// into an 8-bit RGBA OFX image (bottom-up rows). Vertical flip only — no channel swap.
void rgba_top_down_to_rgba_bottom_up(const std::uint8_t* src,
                                     int                 src_stride,
                                     std::uint8_t*       dst,
                                     int                 dst_stride,
                                     int                 width,
                                     int                 height);

/// Inverse: 8-bit RGBA bottom-up OFX image back to 8-bit RGBA top-down CasparCG image
/// (vertical flip only, no channel swap).
void rgba_bottom_up_to_rgba_top_down(const std::uint8_t* src,
                                     int                 src_stride,
                                     std::uint8_t*       dst,
                                     int                 dst_stride,
                                     int                 width,
                                     int                 height);

/// 16-bit variant: BGRA16 (CasparCG, top-down) -> RGBA16 (OFX, bottom-up).
void bgra16_top_down_to_rgba16_bottom_up(const std::uint8_t* src,
                                         int                 src_stride,
                                         std::uint8_t*       dst,
                                         int                 dst_stride,
                                         int                 width,
                                         int                 height);

/// 16-bit variant: RGBA16 (OFX, bottom-up) -> BGRA16 (CasparCG, top-down).
void rgba16_bottom_up_to_bgra16_top_down(const std::uint8_t* src,
                                         int                 src_stride,
                                         std::uint8_t*       dst,
                                         int                 dst_stride,
                                         int                 width,
                                         int                 height);

/// Premultiply an 8-bit RGBA buffer in place (RGB *= A/255). Used to normalise straight-alpha
/// sources to the premultiplied convention the OFX clips advertise and the mixer expects.
void premultiply_rgba8(std::uint8_t* rgba, int stride, int width, int height);

/// Premultiply a 16-bit RGBA buffer in place (RGB *= A/65535).
void premultiply_rgba16(std::uint8_t* rgba, int stride, int width, int height);

/// Convert a BGRA source (8- or 16-bit int, top-down) to RGBA float32 (bottom-up), normalised
/// to 0..1. src_bytes is 1 (8-bit) or 2 (16-bit).
void bgra_to_rgbaf_bottom_up(const std::uint8_t* src,
                             int                 src_stride,
                             int                 src_bytes,
                             float*              dst,
                             int                 dst_stride,
                             int                 width,
                             int                 height);

/// Convert an RGBA float32 (bottom-up, 0..1) image back to a BGRA target (8- or 16-bit int,
/// top-down). dst_bytes is 1 (8-bit) or 2 (16-bit). Values are clamped to 0..1.
void rgbaf_bottom_up_to_bgra(const float*  src,
                             int           src_stride,
                             std::uint8_t* dst,
                             int           dst_stride,
                             int           dst_bytes,
                             int           width,
                             int           height);

/// Premultiply an RGBA float32 buffer in place (RGB *= A).
void premultiply_rgbaf(float* rgba, int stride, int width, int height);

}} // namespace caspar::ofx
