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
 * Author: Carlos Fernandez
 */

#include "write_frame.h"

#include "frame.h"
#include "pixel_format.h"

#include <common/log.h>
#include <common/utf.h>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#include <lodepng.h>

namespace caspar { namespace core {

namespace {

// Scale value from native bit depth to 16-bit range (0–65535) using bit replication.
inline uint16_t scale_to_16(uint16_t val, common::bit_depth depth)
{
    switch (depth) {
    case common::bit_depth::bit10: return static_cast<uint16_t>((val << 6) | (val >> 4));
    case common::bit_depth::bit12: return static_cast<uint16_t>((val << 4) | (val >> 8));
    case common::bit_depth::bit16: return val;
    default: return static_cast<uint16_t>(val * 257); // bit8: 0x00→0x0000, 0xFF→0xFFFF
    }
}

// Write big-endian uint16 (PNG native byte order).
inline void write_be16(uint8_t* dst, uint16_t val)
{
    dst[0] = static_cast<uint8_t>(val >> 8);
    dst[1] = static_cast<uint8_t>(val & 0xFF);
}

bool is_hbd(const pixel_format_desc& desc)
{
    return !desc.planes.empty() && desc.planes[0].depth != common::bit_depth::bit8;
}

// YCbCr→RGB conversion coefficients per color space.
// cr_to_r = 2*(1-Kr), cb_to_b = 2*(1-Kb)
// cb_to_g = -Kb*(1-Kb)/Kg * 2, cr_to_g = -Kr*(1-Kr)/Kg * 2
// where Kg = 1 - Kr - Kb
struct ycbcr_coeffs {
    double cr_to_r, cb_to_g, cr_to_g, cb_to_b;
};

inline ycbcr_coeffs get_ycbcr_coeffs(core::color_space cs)
{
    switch (cs) {
    case core::color_space::bt2020: // Kr=0.2627, Kb=0.0593, Kg=0.6780
        return {1.4746, -0.16455, -0.57135, 1.8814};
    case core::color_space::bt601:  // Kr=0.299, Kb=0.114, Kg=0.587
        return {1.4020, -0.34414, -0.71414, 1.7720};
    default: // bt709: Kr=0.2126, Kb=0.0722, Kg=0.7152
        return {1.5748, -0.18732, -0.46812, 1.8556};
    }
}

} // anonymous namespace

bool write_frame_png(const const_frame& frame, const std::wstring& path)
{
    try {
        if (!frame) {
            CASPAR_LOG(warning) << L"write_frame_png: empty frame";
            return false;
        }

        const auto width  = static_cast<int>(frame.width());
        const auto height = static_cast<int>(frame.height());

        if (width <= 0 || height <= 0) {
            CASPAR_LOG(warning) << L"write_frame_png: invalid dimensions " << width << L"x" << height;
            return false;
        }

        // Get pixel data (triggers lazy GPU readback if needed)
        const auto& plane0 = frame.image_data(0);
        if (plane0.size() == 0) {
            // No CPU data — try on-demand GPU readback via the texture
            auto tex = frame.texture();
            if (tex) {
                auto gpu_pixels = tex->read_pixels();
                if (!gpu_pixels.empty()) {
                    CASPAR_LOG(debug) << L"write_frame_png: using GPU readback (" << gpu_pixels.size() << L" bytes)";
                    auto store = std::make_shared<std::vector<uint8_t>>(std::move(gpu_pixels));
                    array<const uint8_t> img_arr(store->data(), store->size(), store);
                    std::vector<array<const uint8_t>> img_vec;
                    img_vec.push_back(std::move(img_arr));
                    // Use the frame's own pixel format — it encodes the correct
                    // channel interpretation for whatever wrote the texture:
                    //   OGL+CUDA: glGetTextureImage(GL_BGRA) applies R↔B swap on
                    //             internal channels, producing RGBA in memory → frame says rgba.
                    //   Vulkan:   raw cudaMemcpy gives BGRA in memory → frame says bgra.
                    auto tex_depth = tex->tex_is_hbd() ? common::bit_depth::bit16 : common::bit_depth::bit8;
                    auto readback_fmt = frame.pixel_format_desc().format;
                    pixel_format_desc pfd(readback_fmt);
                    pfd.planes.push_back(pixel_format_desc::plane(width, height, 4, tex_depth));
                    const_frame readback_frame(frame.stream_tag(), std::move(img_vec),
                                              frame.audio_data(), pfd, nullptr);
                    return write_frame_png(readback_frame, path);
                }
            }
            CASPAR_LOG(warning) << L"write_frame_png: no image data";
            return false;
        }

        const auto  num_pixels = static_cast<size_t>(width) * static_cast<size_t>(height);
        const auto& desc       = frame.pixel_format_desc();
        const bool  hbd        = is_hbd(desc);
        const auto  depth      = hbd ? desc.planes[0].depth : common::bit_depth::bit8;

        // --- HBD path: output 16-bit RGBA PNG via lodepng ---
        if (hbd) {
            // 8 bytes per pixel: RGBA, 2 bytes each, big-endian
            std::vector<uint8_t> rgba16(num_pixels * 8);

            if (desc.format == pixel_format::bgra || desc.format == pixel_format::rgba ||
                desc.format == pixel_format::argb || desc.format == pixel_format::abgr) {
                const auto* src16 = reinterpret_cast<const uint16_t*>(plane0.data());
                for (size_t i = 0; i < num_pixels; ++i) {
                    uint16_t r, g, b, a;
                    if (desc.format == pixel_format::bgra) {
                        b = src16[i * 4 + 0]; g = src16[i * 4 + 1]; r = src16[i * 4 + 2]; a = src16[i * 4 + 3];
                    } else if (desc.format == pixel_format::rgba) {
                        r = src16[i * 4 + 0]; g = src16[i * 4 + 1]; b = src16[i * 4 + 2]; a = src16[i * 4 + 3];
                    } else if (desc.format == pixel_format::argb) {
                        a = src16[i * 4 + 0]; r = src16[i * 4 + 1]; g = src16[i * 4 + 2]; b = src16[i * 4 + 3];
                    } else { // abgr
                        a = src16[i * 4 + 0]; b = src16[i * 4 + 1]; g = src16[i * 4 + 2]; r = src16[i * 4 + 3];
                    }
                    auto* dst = &rgba16[i * 8];
                    write_be16(dst + 0, scale_to_16(r, depth));
                    write_be16(dst + 2, scale_to_16(g, depth));
                    write_be16(dst + 4, scale_to_16(b, depth));
                    write_be16(dst + 6, scale_to_16(a, depth));
                }
            } else if (desc.format == pixel_format::ycbcr && desc.planes.size() >= 3) {
                const auto& y_plane  = frame.image_data(0);
                const auto& cb_plane = frame.image_data(1);
                const auto& cr_plane = frame.image_data(2);
                const int   cb_w  = desc.planes[1].width;
                const int   cb_h  = desc.planes[1].height;
                const int   sub_x = (cb_w < width) ? (width / cb_w) : 1;
                const int   sub_y = (cb_h < height) ? (height / cb_h) : 1;

                // Limited-range offsets/ranges scaled to bit depth
                const int n = (depth == common::bit_depth::bit10) ? 10 :
                              (depth == common::bit_depth::bit12) ? 12 : 16;
                const double scale  = static_cast<double>(1 << (n - 8));
                const double y_off  = 16.0 * scale;
                const double y_rng  = 219.0 * scale;
                const double c_off  = 128.0 * scale;
                const double c_rng  = 224.0 * scale;
                const auto   coeff  = get_ycbcr_coeffs(desc.color_space);

                for (int row = 0; row < height; ++row) {
                    for (int col = 0; col < width; ++col) {
                        auto* y_ptr  = reinterpret_cast<const uint16_t*>(y_plane.data()) + row * desc.planes[0].width + col;
                        auto* cb_ptr = reinterpret_cast<const uint16_t*>(cb_plane.data()) + (row / sub_y) * cb_w + (col / sub_x);
                        auto* cr_ptr = reinterpret_cast<const uint16_t*>(cr_plane.data()) + (row / sub_y) * cb_w + (col / sub_x);

                        double y_n  = (*y_ptr - y_off) / y_rng;
                        double cb_n = (*cb_ptr - c_off) / c_rng;
                        double cr_n = (*cr_ptr - c_off) / c_rng;

                        double r = y_n + coeff.cr_to_r * cr_n;
                        double g = y_n + coeff.cb_to_g * cb_n + coeff.cr_to_g * cr_n;
                        double b = y_n + coeff.cb_to_b * cb_n;

                        auto idx  = static_cast<size_t>(row) * width + col;
                        auto* dst = &rgba16[idx * 8];
                        write_be16(dst + 0, static_cast<uint16_t>(std::clamp(r * 65535.0 + 0.5, 0.0, 65535.0)));
                        write_be16(dst + 2, static_cast<uint16_t>(std::clamp(g * 65535.0 + 0.5, 0.0, 65535.0)));
                        write_be16(dst + 4, static_cast<uint16_t>(std::clamp(b * 65535.0 + 0.5, 0.0, 65535.0)));
                        write_be16(dst + 6, 0xFFFF);
                    }
                }
            } else if ((desc.format == pixel_format::gbrap || desc.format == pixel_format::gbrp) &&
                       desc.planes.size() >= 3) {
                const auto* g16 = reinterpret_cast<const uint16_t*>(frame.image_data(0).data());
                const auto* b16 = reinterpret_cast<const uint16_t*>(frame.image_data(1).data());
                const auto* r16 = reinterpret_cast<const uint16_t*>(frame.image_data(2).data());
                const bool  has_alpha = (desc.format == pixel_format::gbrap && desc.planes.size() >= 4);
                const uint16_t* a16 = has_alpha
                    ? reinterpret_cast<const uint16_t*>(frame.image_data(3).data()) : nullptr;

                for (size_t i = 0; i < num_pixels; ++i) {
                    auto* dst = &rgba16[i * 8];
                    write_be16(dst + 0, scale_to_16(r16[i], depth));
                    write_be16(dst + 2, scale_to_16(g16[i], depth));
                    write_be16(dst + 4, scale_to_16(b16[i], depth));
                    write_be16(dst + 6, a16 ? scale_to_16(a16[i], depth) : uint16_t(0xFFFF));
                }
            } else if (desc.format == pixel_format::ycbcra && desc.planes.size() >= 4) {
                const auto& y_plane  = frame.image_data(0);
                const auto& cb_plane = frame.image_data(1);
                const auto& cr_plane = frame.image_data(2);
                const auto& a_plane  = frame.image_data(3);
                const int   cb_w  = desc.planes[1].width;
                const int   cb_h  = desc.planes[1].height;
                const int   sub_x = (cb_w < width) ? (width / cb_w) : 1;
                const int   sub_y = (cb_h < height) ? (height / cb_h) : 1;

                const int n = (depth == common::bit_depth::bit10) ? 10 :
                              (depth == common::bit_depth::bit12) ? 12 : 16;
                const double scale  = static_cast<double>(1 << (n - 8));
                const double y_off  = 16.0 * scale;
                const double y_rng  = 219.0 * scale;
                const double c_off  = 128.0 * scale;
                const double c_rng  = 224.0 * scale;
                const auto   coeff  = get_ycbcr_coeffs(desc.color_space);

                for (int row = 0; row < height; ++row) {
                    for (int col = 0; col < width; ++col) {
                        auto* y_ptr  = reinterpret_cast<const uint16_t*>(y_plane.data()) + row * desc.planes[0].width + col;
                        auto* cb_ptr = reinterpret_cast<const uint16_t*>(cb_plane.data()) + (row / sub_y) * cb_w + (col / sub_x);
                        auto* cr_ptr = reinterpret_cast<const uint16_t*>(cr_plane.data()) + (row / sub_y) * cb_w + (col / sub_x);
                        auto* a_ptr  = reinterpret_cast<const uint16_t*>(a_plane.data()) + row * desc.planes[3].width + col;

                        double y_n  = (*y_ptr - y_off) / y_rng;
                        double cb_n = (*cb_ptr - c_off) / c_rng;
                        double cr_n = (*cr_ptr - c_off) / c_rng;

                        double r = y_n + coeff.cr_to_r * cr_n;
                        double g = y_n + coeff.cb_to_g * cb_n + coeff.cr_to_g * cr_n;
                        double b = y_n + coeff.cb_to_b * cb_n;

                        auto idx  = static_cast<size_t>(row) * width + col;
                        auto* dst = &rgba16[idx * 8];
                        write_be16(dst + 0, static_cast<uint16_t>(std::clamp(r * 65535.0 + 0.5, 0.0, 65535.0)));
                        write_be16(dst + 2, static_cast<uint16_t>(std::clamp(g * 65535.0 + 0.5, 0.0, 65535.0)));
                        write_be16(dst + 4, static_cast<uint16_t>(std::clamp(b * 65535.0 + 0.5, 0.0, 65535.0)));
                        write_be16(dst + 6, scale_to_16(*a_ptr, depth));
                    }
                }
            } else {
                CASPAR_LOG(warning) << L"write_frame_png: unsupported HBD pixel format " << static_cast<int>(desc.format);
                return false;
            }

            // Write 16-bit PNG via lodepng (force RGBA output for viewer compatibility)
            auto output_path = std::filesystem::path(path);
            std::filesystem::create_directories(output_path.parent_path());
            auto path_u8 = u8(path);

            lodepng::State state;
            state.info_raw.colortype    = LCT_RGBA;
            state.info_raw.bitdepth     = 16;
            state.info_png.color.colortype = LCT_RGBA;
            state.info_png.color.bitdepth  = 16;
            state.encoder.auto_convert  = 0; // disable: always output RGBA16

            std::vector<unsigned char> png;
            unsigned error = lodepng::encode(png, rgba16,
                                            static_cast<unsigned>(width), static_cast<unsigned>(height),
                                            state);
            if (error) {
                CASPAR_LOG(warning) << L"write_frame_png: lodepng error " << error << L" for " << path;
                return false;
            }
            error = lodepng::save_file(png, path_u8);
            if (error) {
                CASPAR_LOG(warning) << L"write_frame_png: lodepng save error " << error << L" for " << path;
                return false;
            }

            CASPAR_LOG(info) << L"write_frame_png: saved " << width << L"x" << height
                             << L" 16-bit → " << path;
            return true;
        }

        // --- 8-bit path: output 8-bit RGBA PNG via stb ---
        std::vector<uint8_t> rgba(num_pixels * 4);

        if (desc.format == pixel_format::bgra || desc.format == pixel_format::rgba ||
            desc.format == pixel_format::argb || desc.format == pixel_format::abgr) {
            const auto* src = plane0.data();
            auto*       dst = rgba.data();
            if (desc.format == pixel_format::bgra) {
                for (size_t i = 0; i < num_pixels; ++i) {
                    dst[i * 4 + 0] = src[i * 4 + 2]; // R <- B
                    dst[i * 4 + 1] = src[i * 4 + 1]; // G
                    dst[i * 4 + 2] = src[i * 4 + 0]; // B <- R
                    dst[i * 4 + 3] = src[i * 4 + 3]; // A
                }
            } else if (desc.format == pixel_format::rgba) {
                std::memcpy(dst, src, num_pixels * 4);
            } else if (desc.format == pixel_format::argb) {
                for (size_t i = 0; i < num_pixels; ++i) {
                    dst[i * 4 + 0] = src[i * 4 + 1];
                    dst[i * 4 + 1] = src[i * 4 + 2];
                    dst[i * 4 + 2] = src[i * 4 + 3];
                    dst[i * 4 + 3] = src[i * 4 + 0];
                }
            } else { // abgr
                for (size_t i = 0; i < num_pixels; ++i) {
                    dst[i * 4 + 0] = src[i * 4 + 3];
                    dst[i * 4 + 1] = src[i * 4 + 2];
                    dst[i * 4 + 2] = src[i * 4 + 1];
                    dst[i * 4 + 3] = src[i * 4 + 0];
                }
            }
        } else if (desc.format == pixel_format::ycbcr && desc.planes.size() >= 3) {
            const auto& y_plane  = frame.image_data(0);
            const auto& cb_plane = frame.image_data(1);
            const auto& cr_plane = frame.image_data(2);
            const int   cb_w  = desc.planes[1].width;
            const int   cb_h  = desc.planes[1].height;
            const int   sub_x = (cb_w < width) ? (width / cb_w) : 1;
            const int   sub_y = (cb_h < height) ? (height / cb_h) : 1;
            const auto  coeff = get_ycbcr_coeffs(desc.color_space);

            auto* dst = rgba.data();
            for (int row = 0; row < height; ++row) {
                for (int col = 0; col < width; ++col) {
                    double y_val  = y_plane.data()[row * desc.planes[0].width + col];
                    double cb_val = cb_plane.data()[(row / sub_y) * cb_w + (col / sub_x)];
                    double cr_val = cr_plane.data()[(row / sub_y) * cb_w + (col / sub_x)];

                    // YCbCr → RGB (limited range)
                    double y_n  = (y_val - 16.0) / 219.0;
                    double cb_n = (cb_val - 128.0) / 224.0;
                    double cr_n = (cr_val - 128.0) / 224.0;
                    int    r    = static_cast<int>((y_n + coeff.cr_to_r * cr_n) * 255.0 + 0.5);
                    int    g    = static_cast<int>((y_n + coeff.cb_to_g * cb_n + coeff.cr_to_g * cr_n) * 255.0 + 0.5);
                    int    b    = static_cast<int>((y_n + coeff.cb_to_b * cb_n) * 255.0 + 0.5);
                    auto   idx  = static_cast<size_t>(row) * width + col;
                    dst[idx * 4 + 0] = static_cast<uint8_t>(std::clamp(r, 0, 255));
                    dst[idx * 4 + 1] = static_cast<uint8_t>(std::clamp(g, 0, 255));
                    dst[idx * 4 + 2] = static_cast<uint8_t>(std::clamp(b, 0, 255));
                    dst[idx * 4 + 3] = 255;
                }
            }
        } else if (desc.format == pixel_format::ycbcra && desc.planes.size() >= 4) {
            const auto& y_plane  = frame.image_data(0);
            const auto& cb_plane = frame.image_data(1);
            const auto& cr_plane = frame.image_data(2);
            const auto& a_plane  = frame.image_data(3);
            const int   cb_w  = desc.planes[1].width;
            const int   cb_h  = desc.planes[1].height;
            const int   sub_x = (cb_w < width) ? (width / cb_w) : 1;
            const int   sub_y = (cb_h < height) ? (height / cb_h) : 1;
            const auto  coeff = get_ycbcr_coeffs(desc.color_space);

            auto* dst = rgba.data();
            for (int row = 0; row < height; ++row) {
                for (int col = 0; col < width; ++col) {
                    double y_val  = y_plane.data()[row * desc.planes[0].width + col];
                    double cb_val = cb_plane.data()[(row / sub_y) * cb_w + (col / sub_x)];
                    double cr_val = cr_plane.data()[(row / sub_y) * cb_w + (col / sub_x)];

                    double y_n  = (y_val - 16.0) / 219.0;
                    double cb_n = (cb_val - 128.0) / 224.0;
                    double cr_n = (cr_val - 128.0) / 224.0;
                    int    r    = static_cast<int>((y_n + coeff.cr_to_r * cr_n) * 255.0 + 0.5);
                    int    g    = static_cast<int>((y_n + coeff.cb_to_g * cb_n + coeff.cr_to_g * cr_n) * 255.0 + 0.5);
                    int    b    = static_cast<int>((y_n + coeff.cb_to_b * cb_n) * 255.0 + 0.5);
                    auto   idx  = static_cast<size_t>(row) * width + col;
                    dst[idx * 4 + 0] = static_cast<uint8_t>(std::clamp(r, 0, 255));
                    dst[idx * 4 + 1] = static_cast<uint8_t>(std::clamp(g, 0, 255));
                    dst[idx * 4 + 2] = static_cast<uint8_t>(std::clamp(b, 0, 255));
                    dst[idx * 4 + 3] = a_plane.data()[row * desc.planes[3].width + col];
                }
            }
        } else if ((desc.format == pixel_format::gbrap || desc.format == pixel_format::gbrp) &&
                   desc.planes.size() >= 3) {
            const auto* g8 = frame.image_data(0).data();
            const auto* b8 = frame.image_data(1).data();
            const auto* r8 = frame.image_data(2).data();
            const bool  has_alpha = (desc.format == pixel_format::gbrap && desc.planes.size() >= 4);
            const uint8_t* a8 = has_alpha ? frame.image_data(3).data() : nullptr;

            auto* dst = rgba.data();
            for (size_t i = 0; i < num_pixels; ++i) {
                dst[i * 4 + 0] = r8[i];
                dst[i * 4 + 1] = g8[i];
                dst[i * 4 + 2] = b8[i];
                dst[i * 4 + 3] = a8 ? a8[i] : uint8_t(255);
            }
        } else {
            CASPAR_LOG(warning) << L"write_frame_png: unsupported pixel format " << static_cast<int>(desc.format);
            return false;
        }

        // Write 8-bit PNG via stb
        auto output_path = std::filesystem::path(path);
        std::filesystem::create_directories(output_path.parent_path());
        auto path_u8 = u8(path);
        int  result  = stbi_write_png(path_u8.c_str(), width, height, 4, rgba.data(), width * 4);

        if (result == 0) {
            CASPAR_LOG(warning) << L"write_frame_png: stbi_write_png failed for " << path;
            return false;
        }

        CASPAR_LOG(info) << L"write_frame_png: saved " << width << L"x" << height << L" → " << path;
        return true;

    } catch (const std::exception& e) {
        CASPAR_LOG(error) << L"write_frame_png: exception: " << e.what();
        return false;
    } catch (...) {
        CASPAR_LOG(error) << L"write_frame_png: unknown exception";
        return false;
    }
}

}} // namespace caspar::core
