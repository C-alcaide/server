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

#include "ofx_image_bridge.h"

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace caspar { namespace ofx {

void bgra_top_down_to_rgba_bottom_up(const std::uint8_t* src,
                                     int                 src_stride,
                                     std::uint8_t*       dst,
                                     int                 dst_stride,
                                     int                 width,
                                     int                 height)
{
    for (int y = 0; y < height; ++y) {
        const std::uint8_t* s = src + static_cast<std::size_t>(y) * src_stride;
        // OFX image is bottom-up: destination row (height-1-y) receives source row y.
        std::uint8_t* d = dst + static_cast<std::size_t>(height - 1 - y) * dst_stride;
        for (int x = 0; x < width; ++x) {
            const std::uint8_t b = s[0];
            const std::uint8_t g = s[1];
            const std::uint8_t r = s[2];
            const std::uint8_t a = s[3];
            d[0] = r;
            d[1] = g;
            d[2] = b;
            d[3] = a;
            s += 4;
            d += 4;
        }
    }
}

void rgba_bottom_up_to_bgra_top_down(const std::uint8_t* src,
                                     int                 src_stride,
                                     std::uint8_t*       dst,
                                     int                 dst_stride,
                                     int                 width,
                                     int                 height)
{
    for (int y = 0; y < height; ++y) {
        // Source OFX image is bottom-up: source row (height-1-y) maps to top-down row y.
        const std::uint8_t* s = src + static_cast<std::size_t>(height - 1 - y) * src_stride;
        std::uint8_t*       d = dst + static_cast<std::size_t>(y) * dst_stride;
        for (int x = 0; x < width; ++x) {
            const std::uint8_t r = s[0];
            const std::uint8_t g = s[1];
            const std::uint8_t b = s[2];
            const std::uint8_t a = s[3];
            d[0] = b;
            d[1] = g;
            d[2] = r;
            d[3] = a;
            s += 4;
            d += 4;
        }
    }
}

void rgba_top_down_to_rgba_bottom_up(const std::uint8_t* src,
                                     int                 src_stride,
                                     std::uint8_t*       dst,
                                     int                 dst_stride,
                                     int                 width,
                                     int                 height)
{
    const std::size_t row = static_cast<std::size_t>(width) * 4;
    for (int y = 0; y < height; ++y) {
        const std::uint8_t* s = src + static_cast<std::size_t>(y) * src_stride;
        // OFX image is bottom-up: destination row (height-1-y) receives source row y.
        std::uint8_t* d = dst + static_cast<std::size_t>(height - 1 - y) * dst_stride;
        std::memcpy(d, s, row);
    }
}

void rgba_bottom_up_to_rgba_top_down(const std::uint8_t* src,
                                     int                 src_stride,
                                     std::uint8_t*       dst,
                                     int                 dst_stride,
                                     int                 width,
                                     int                 height)
{
    const std::size_t row = static_cast<std::size_t>(width) * 4;
    for (int y = 0; y < height; ++y) {
        // Source OFX image is bottom-up: source row (height-1-y) maps to top-down row y.
        const std::uint8_t* s = src + static_cast<std::size_t>(height - 1 - y) * src_stride;
        std::uint8_t*       d = dst + static_cast<std::size_t>(y) * dst_stride;
        std::memcpy(d, s, row);
    }
}

void bgra16_top_down_to_rgba16_bottom_up(const std::uint8_t* src,                                         int                 src_stride,
                                         std::uint8_t*       dst,
                                         int                 dst_stride,
                                         int                 width,
                                         int                 height)
{
    for (int y = 0; y < height; ++y) {
        const auto* s = reinterpret_cast<const std::uint16_t*>(src + static_cast<std::size_t>(y) * src_stride);
        auto*       d = reinterpret_cast<std::uint16_t*>(dst + static_cast<std::size_t>(height - 1 - y) * dst_stride);
        for (int x = 0; x < width; ++x) {
            const std::uint16_t b = s[0];
            const std::uint16_t g = s[1];
            const std::uint16_t r = s[2];
            const std::uint16_t a = s[3];
            d[0]                  = r;
            d[1]                  = g;
            d[2]                  = b;
            d[3]                  = a;
            s += 4;
            d += 4;
        }
    }
}

void rgba16_bottom_up_to_bgra16_top_down(const std::uint8_t* src,
                                         int                 src_stride,
                                         std::uint8_t*       dst,
                                         int                 dst_stride,
                                         int                 width,
                                         int                 height)
{
    for (int y = 0; y < height; ++y) {
        const auto* s = reinterpret_cast<const std::uint16_t*>(src + static_cast<std::size_t>(height - 1 - y) * src_stride);
        auto*       d = reinterpret_cast<std::uint16_t*>(dst + static_cast<std::size_t>(y) * dst_stride);
        for (int x = 0; x < width; ++x) {
            const std::uint16_t r = s[0];
            const std::uint16_t g = s[1];
            const std::uint16_t b = s[2];
            const std::uint16_t a = s[3];
            d[0]                  = b;
            d[1]                  = g;
            d[2]                  = r;
            d[3]                  = a;
            s += 4;
            d += 4;
        }
    }
}

void premultiply_rgba8(std::uint8_t* rgba, int stride, int width, int height)
{
    for (int y = 0; y < height; ++y) {
        std::uint8_t* p = rgba + static_cast<std::size_t>(y) * stride;
        for (int x = 0; x < width; ++x) {
            const unsigned a = p[3];
            p[0]             = static_cast<std::uint8_t>((p[0] * a + 127) / 255);
            p[1]             = static_cast<std::uint8_t>((p[1] * a + 127) / 255);
            p[2]             = static_cast<std::uint8_t>((p[2] * a + 127) / 255);
            p += 4;
        }
    }
}

void premultiply_rgba16(std::uint8_t* rgba, int stride, int width, int height)
{
    for (int y = 0; y < height; ++y) {
        auto* p = reinterpret_cast<std::uint16_t*>(rgba + static_cast<std::size_t>(y) * stride);
        for (int x = 0; x < width; ++x) {
            const std::uint32_t a = p[3];
            p[0]                  = static_cast<std::uint16_t>((p[0] * a + 32767) / 65535);
            p[1]                  = static_cast<std::uint16_t>((p[1] * a + 32767) / 65535);
            p[2]                  = static_cast<std::uint16_t>((p[2] * a + 32767) / 65535);
            p += 4;
        }
    }
}

void bgra_to_rgbaf_bottom_up(const std::uint8_t* src,
                             int                 src_stride,
                             int                 src_bytes,
                             float*              dst,
                             int                 dst_stride,
                             int                 width,
                             int                 height)
{
    const float inv = src_bytes == 2 ? 1.0f / 65535.0f : 1.0f / 255.0f;
    for (int y = 0; y < height; ++y) {
        const std::uint8_t* srow = src + static_cast<std::size_t>(y) * src_stride;
        float*              d    = reinterpret_cast<float*>(reinterpret_cast<std::uint8_t*>(dst) +
                                              static_cast<std::size_t>(height - 1 - y) * dst_stride);
        if (src_bytes == 2) {
            const auto* s = reinterpret_cast<const std::uint16_t*>(srow);
            for (int x = 0; x < width; ++x) {
                d[0] = s[2] * inv; // R <- B index? no: BGRA -> b,g,r,a
                d[1] = s[1] * inv;
                d[2] = s[0] * inv;
                d[3] = s[3] * inv;
                s += 4;
                d += 4;
            }
        } else {
            const std::uint8_t* s = srow;
            for (int x = 0; x < width; ++x) {
                d[0] = s[2] * inv;
                d[1] = s[1] * inv;
                d[2] = s[0] * inv;
                d[3] = s[3] * inv;
                s += 4;
                d += 4;
            }
        }
    }
}

void rgbaf_bottom_up_to_bgra(const float*  src,
                             int           src_stride,
                             std::uint8_t* dst,
                             int           dst_stride,
                             int           dst_bytes,
                             int           width,
                             int           height)
{
    auto clamp01 = [](float v) { return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v); };
    for (int y = 0; y < height; ++y) {
        const float* s = reinterpret_cast<const float*>(reinterpret_cast<const std::uint8_t*>(src) +
                                                        static_cast<std::size_t>(height - 1 - y) * src_stride);
        std::uint8_t* drow = dst + static_cast<std::size_t>(y) * dst_stride;
        if (dst_bytes == 2) {
            auto* d = reinterpret_cast<std::uint16_t*>(drow);
            for (int x = 0; x < width; ++x) {
                d[0] = static_cast<std::uint16_t>(clamp01(s[2]) * 65535.0f + 0.5f); // B
                d[1] = static_cast<std::uint16_t>(clamp01(s[1]) * 65535.0f + 0.5f); // G
                d[2] = static_cast<std::uint16_t>(clamp01(s[0]) * 65535.0f + 0.5f); // R
                d[3] = static_cast<std::uint16_t>(clamp01(s[3]) * 65535.0f + 0.5f); // A
                s += 4;
                d += 4;
            }
        } else {
            std::uint8_t* d = drow;
            for (int x = 0; x < width; ++x) {
                d[0] = static_cast<std::uint8_t>(clamp01(s[2]) * 255.0f + 0.5f);
                d[1] = static_cast<std::uint8_t>(clamp01(s[1]) * 255.0f + 0.5f);
                d[2] = static_cast<std::uint8_t>(clamp01(s[0]) * 255.0f + 0.5f);
                d[3] = static_cast<std::uint8_t>(clamp01(s[3]) * 255.0f + 0.5f);
                s += 4;
                d += 4;
            }
        }
    }
}

void premultiply_rgbaf(float* rgba, int stride, int width, int height)
{
    for (int y = 0; y < height; ++y) {
        float* p = reinterpret_cast<float*>(reinterpret_cast<std::uint8_t*>(rgba) + static_cast<std::size_t>(y) * stride);
        for (int x = 0; x < width; ++x) {
            const float a = p[3];
            p[0] *= a;
            p[1] *= a;
            p[2] *= a;
            p += 4;
        }
    }
}

}} // namespace caspar::ofx
