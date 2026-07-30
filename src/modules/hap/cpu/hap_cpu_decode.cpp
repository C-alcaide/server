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

// hap_cpu_decode.cpp
// CPU-side BCn decompression using bcdec (header-only, MIT/Unlicense).
// ---------------------------------------------------------------------------

// bcdec has int→char narrowing warnings that MSVC /WX would reject.
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4244) // conversion from 'int' to 'char/unsigned char'
#endif

#define BCDEC_IMPLEMENTATION
#include <bcdec.h>

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include "hap_cpu_decode.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace caspar { namespace hap {

// bcdec outputs RGBA (R in lowest byte on little-endian).
// CasparCG's mixer expects BGRA, so we swizzle R<->B in-place.
static void swizzle_rgba_to_bgra(uint8_t* pixels, int count)
{
    for (int i = 0; i < count; ++i) {
        std::swap(pixels[i * 4 + 0], pixels[i * 4 + 2]);
    }
}

// YCoCg-DXT5 → RGB conversion (same math as the GL shader).
// Input pixel layout after BC3 decompression: R=Co, G=Cg, B=scale_code, A=Y
// Output: BGRA (CasparCG convention)
static void convert_ycocg_to_bgra(uint8_t* pixels, int count)
{
    for (int i = 0; i < count; ++i) {
        float co    = (pixels[i * 4 + 0] - 128.0f);
        float cg    = (pixels[i * 4 + 1] - 128.0f);
        float scale = (pixels[i * 4 + 2] / 8.0f) + 1.0f;
        float y     = pixels[i * 4 + 3];

        co /= scale;
        cg /= scale;

        // Round rather than truncate. The GL shader that decodes the same data
        // on the other path ends in a normalised framebuffer write, which rounds
        // to nearest; a cast to int truncates, so every channel came out biased
        // half a level low and the two backends disagreed by ~6 dB on the same
        // file for no reason other than this.
        int r = static_cast<int>(std::lround(y + co - cg));
        int g = static_cast<int>(std::lround(y + cg));
        int b = static_cast<int>(std::lround(y - co - cg));

        // BGRA output
        pixels[i * 4 + 0] = static_cast<uint8_t>(std::clamp(b, 0, 255));
        pixels[i * 4 + 1] = static_cast<uint8_t>(std::clamp(g, 0, 255));
        pixels[i * 4 + 2] = static_cast<uint8_t>(std::clamp(r, 0, 255));
        pixels[i * 4 + 3] = 255;
    }
}

// Same as convert_ycocg_to_bgra but takes separate alpha channel data.
static void convert_ycocg_alpha_to_bgra(uint8_t* pixels, const uint8_t* alpha_pixels, int count)
{
    for (int i = 0; i < count; ++i) {
        float co    = (pixels[i * 4 + 0] - 128.0f);
        float cg    = (pixels[i * 4 + 1] - 128.0f);
        float scale = (pixels[i * 4 + 2] / 8.0f) + 1.0f;
        float y     = pixels[i * 4 + 3];

        co /= scale;
        cg /= scale;

        // Rounded, to match the shader -- see convert_ycocg_to_bgra.
        int r = static_cast<int>(std::lround(y + co - cg));
        int g = static_cast<int>(std::lround(y + cg));
        int b = static_cast<int>(std::lround(y - co - cg));

        // BGRA output with external alpha
        pixels[i * 4 + 0] = static_cast<uint8_t>(std::clamp(b, 0, 255));
        pixels[i * 4 + 1] = static_cast<uint8_t>(std::clamp(g, 0, 255));
        pixels[i * 4 + 2] = static_cast<uint8_t>(std::clamp(r, 0, 255));
        pixels[i * 4 + 3] = alpha_pixels[i]; // BC4 alpha channel (single component)
    }
}

bool cpu_decode_hap_to_bgra(HapVariant             variant,
                            const uint8_t*         texture_data,
                            size_t                 texture_size,
                            const uint8_t*         alpha_data,
                            size_t                 alpha_size,
                            int                    width,
                            int                    height,
                            std::vector<uint8_t>&  out_pixels)
{
    if (width <= 0 || height <= 0)
        return false;

    // Block-compressed textures cover a whole 4x4 grid, so a Hap file whose
    // dimensions are not multiples of four carries padding on the right and
    // bottom edges -- 973x734 is stored as 244x184 blocks, or 976x736 pixels.
    // This used to reject those outright, which took out every odd-sized clip on
    // the CPU path, and with it Hap Q Alpha on the Vulkan mixer, since the
    // two-texture variant has no zero-copy route and falls back to here.
    //
    // Decode the full block grid into scratch and copy the visible region out.
    // When the dimensions are already aligned nothing is padded and the decode
    // writes straight into the destination, as before.
    const int blocks_w    = (width  + 3) / 4;
    const int blocks_h    = (height + 3) / 4;
    const int padded_w    = blocks_w * 4;
    const int padded_h    = blocks_h * 4;
    const bool is_padded  = (padded_w != width) || (padded_h != height);

    const int pixel_count = padded_w * padded_h;   // what the decoders fill
    const int dst_pitch   = padded_w * 4;

    out_pixels.resize(static_cast<size_t>(width) * height * 4);

    std::vector<uint8_t> scratch;
    if (is_padded)
        scratch.resize(static_cast<size_t>(pixel_count) * 4);

    // Everything below writes through this; `out_pixels` is the same buffer when
    // no padding is involved.
    std::vector<uint8_t>& decode_target = is_padded ? scratch : out_pixels;

    switch (variant) {
        case HapVariant::Hap: {
            // BC1 (DXT1): 8 bytes per 4x4 block → RGBA
            if (texture_size < static_cast<size_t>(blocks_w * blocks_h * BCDEC_BC1_BLOCK_SIZE))
                return false;

            const uint8_t* src = texture_data;
            for (int by = 0; by < blocks_h; ++by) {
                for (int bx = 0; bx < blocks_w; ++bx) {
                    uint8_t* dst = decode_target.data() + (by * 4) * dst_pitch + (bx * 4) * 4;
                    bcdec_bc1(src, dst, dst_pitch);
                    src += BCDEC_BC1_BLOCK_SIZE;
                }
            }
            swizzle_rgba_to_bgra(decode_target.data(), pixel_count);
            break;
        }

        case HapVariant::HapAlpha: {
            // BC3 (DXT5): 16 bytes per 4x4 block → RGBA
            if (texture_size < static_cast<size_t>(blocks_w * blocks_h * BCDEC_BC3_BLOCK_SIZE))
                return false;

            const uint8_t* src = texture_data;
            for (int by = 0; by < blocks_h; ++by) {
                for (int bx = 0; bx < blocks_w; ++bx) {
                    uint8_t* dst = decode_target.data() + (by * 4) * dst_pitch + (bx * 4) * 4;
                    bcdec_bc3(src, dst, dst_pitch);
                    src += BCDEC_BC3_BLOCK_SIZE;
                }
            }
            swizzle_rgba_to_bgra(decode_target.data(), pixel_count);
            break;
        }

        case HapVariant::HapQ: {
            // BC3 (DXT5) with YCoCg color encoding: 16 bytes per block
            if (texture_size < static_cast<size_t>(blocks_w * blocks_h * BCDEC_BC3_BLOCK_SIZE))
                return false;

            const uint8_t* src = texture_data;
            for (int by = 0; by < blocks_h; ++by) {
                for (int bx = 0; bx < blocks_w; ++bx) {
                    uint8_t* dst = decode_target.data() + (by * 4) * dst_pitch + (bx * 4) * 4;
                    bcdec_bc3(src, dst, dst_pitch);
                    src += BCDEC_BC3_BLOCK_SIZE;
                }
            }
            // Raw decompressed: R=Co, G=Cg, B=scale, A=Y → convert to BGRA
            convert_ycocg_to_bgra(decode_target.data(), pixel_count);
            break;
        }

        case HapVariant::HapQAlpha: {
            // Primary: BC3 (YCoCg), Alpha: BC4 (single channel)
            if (texture_size < static_cast<size_t>(blocks_w * blocks_h * BCDEC_BC3_BLOCK_SIZE))
                return false;
            if (!alpha_data || alpha_size < static_cast<size_t>(blocks_w * blocks_h * BCDEC_BC4_BLOCK_SIZE))
                return false;

            // Decompress primary (YCoCg in BC3)
            const uint8_t* src = texture_data;
            for (int by = 0; by < blocks_h; ++by) {
                for (int bx = 0; bx < blocks_w; ++bx) {
                    uint8_t* dst = decode_target.data() + (by * 4) * dst_pitch + (bx * 4) * 4;
                    bcdec_bc3(src, dst, dst_pitch);
                    src += BCDEC_BC3_BLOCK_SIZE;
                }
            }

            // Decompress alpha (BC4 → single channel per pixel)
            std::vector<uint8_t> alpha_buf(pixel_count);
            const uint8_t* asrc = alpha_data;
            for (int by = 0; by < blocks_h; ++by) {
                for (int bx = 0; bx < blocks_w; ++bx) {
                    // One byte per pixel, so the row stride is padded_w, not
                    // padded_w * 4 -- and it must be the padded width to match
                    // the block grid this walks.
                    uint8_t* adst = alpha_buf.data() + (by * 4) * padded_w + (bx * 4);
                    bcdec_bc4(asrc, adst, padded_w);
                    asrc += BCDEC_BC4_BLOCK_SIZE;
                }
            }

            convert_ycocg_alpha_to_bgra(decode_target.data(), alpha_buf.data(), pixel_count);
            break;
        }

        case HapVariant::HapR: {
            // BC7: 16 bytes per 4x4 block → RGBA
            if (texture_size < static_cast<size_t>(blocks_w * blocks_h * BCDEC_BC7_BLOCK_SIZE))
                return false;

            const uint8_t* src = texture_data;
            for (int by = 0; by < blocks_h; ++by) {
                for (int bx = 0; bx < blocks_w; ++bx) {
                    uint8_t* dst = decode_target.data() + (by * 4) * dst_pitch + (bx * 4) * 4;
                    bcdec_bc7(src, dst, dst_pitch);
                    src += BCDEC_BC7_BLOCK_SIZE;
                }
            }
            swizzle_rgba_to_bgra(decode_target.data(), pixel_count);
            break;
        }

        default:
            return false;
    }

    // Drop the block padding, keeping the visible region only.
    if (is_padded) {
        const size_t out_pitch = static_cast<size_t>(width) * 4;
        for (int y = 0; y < height; ++y) {
            std::memcpy(out_pixels.data() + static_cast<size_t>(y) * out_pitch,
                        decode_target.data() + static_cast<size_t>(y) * dst_pitch,
                        out_pitch);
        }
    }

    return true;
}

}} // namespace caspar::hap
