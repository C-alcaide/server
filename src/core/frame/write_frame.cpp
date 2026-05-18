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

namespace caspar { namespace core {

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
            CASPAR_LOG(warning) << L"write_frame_png: no image data";
            return false;
        }

        const auto  num_pixels = static_cast<size_t>(width) * static_cast<size_t>(height);
        const auto& desc       = frame.pixel_format_desc();

        std::vector<uint8_t> rgba(num_pixels * 4);

        if (desc.format == pixel_format::bgra || desc.format == pixel_format::rgba ||
            desc.format == pixel_format::argb || desc.format == pixel_format::abgr) {
            // 4-component packed format
            if (plane0.size() < num_pixels * 4) {
                CASPAR_LOG(warning) << L"write_frame_png: buffer too small for BGRA";
                return false;
            }
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
                    dst[i * 4 + 0] = src[i * 4 + 1]; // R
                    dst[i * 4 + 1] = src[i * 4 + 2]; // G
                    dst[i * 4 + 2] = src[i * 4 + 3]; // B
                    dst[i * 4 + 3] = src[i * 4 + 0]; // A
                }
            } else { // abgr
                for (size_t i = 0; i < num_pixels; ++i) {
                    dst[i * 4 + 0] = src[i * 4 + 3]; // R
                    dst[i * 4 + 1] = src[i * 4 + 2]; // G
                    dst[i * 4 + 2] = src[i * 4 + 1]; // B
                    dst[i * 4 + 3] = src[i * 4 + 0]; // A
                }
            }
        } else if (desc.format == pixel_format::ycbcr && desc.planes.size() >= 3) {
            // YCbCr planar (YUV420P, YUV422P, YUV444P, or deinterleaved NV12/P010)
            const auto& y_plane  = frame.image_data(0);
            const auto& cb_plane = frame.image_data(1);
            const auto& cr_plane = frame.image_data(2);

            const bool is_10bit = (desc.planes[0].depth != common::bit_depth::bit8);
            const int  cb_w     = desc.planes[1].width;
            const int  cb_h     = desc.planes[1].height;
            const int  sub_x    = (cb_w < width) ? (width / cb_w) : 1;
            const int  sub_y    = (cb_h < height) ? (height / cb_h) : 1;

            auto* dst = rgba.data();
            for (int row = 0; row < height; ++row) {
                for (int col = 0; col < width; ++col) {
                    double y_val, cb_val, cr_val;
                    if (is_10bit) {
                        auto* y_ptr  = reinterpret_cast<const uint16_t*>(y_plane.data()) + row * desc.planes[0].width + col;
                        auto* cb_ptr = reinterpret_cast<const uint16_t*>(cb_plane.data()) + (row / sub_y) * cb_w + (col / sub_x);
                        auto* cr_ptr = reinterpret_cast<const uint16_t*>(cr_plane.data()) + (row / sub_y) * cb_w + (col / sub_x);
                        // 10-bit data always in low 10 bits of uint16 (P010 normalised during deinterleave)
                        y_val  = (*y_ptr) / 1023.0 * 255.0;
                        cb_val = (*cb_ptr) / 1023.0 * 255.0;
                        cr_val = (*cr_ptr) / 1023.0 * 255.0;
                    } else {
                        y_val  = y_plane.data()[row * desc.planes[0].width + col];
                        cb_val = cb_plane.data()[(row / sub_y) * cb_w + (col / sub_x)];
                        cr_val = cr_plane.data()[(row / sub_y) * cb_w + (col / sub_x)];
                    }
                    // BT.709 YCbCr -> RGB (limited range)
                    double y_n  = (y_val - 16.0) / 219.0;
                    double cb_n = (cb_val - 128.0) / 224.0;
                    double cr_n = (cr_val - 128.0) / 224.0;
                    int    r    = static_cast<int>((y_n + 1.5748 * cr_n) * 255.0 + 0.5);
                    int    g    = static_cast<int>((y_n - 0.1873 * cb_n - 0.4681 * cr_n) * 255.0 + 0.5);
                    int    b    = static_cast<int>((y_n + 1.8556 * cb_n) * 255.0 + 0.5);
                    auto   idx  = static_cast<size_t>(row) * width + col;
                    dst[idx * 4 + 0] = static_cast<uint8_t>(std::clamp(r, 0, 255));
                    dst[idx * 4 + 1] = static_cast<uint8_t>(std::clamp(g, 0, 255));
                    dst[idx * 4 + 2] = static_cast<uint8_t>(std::clamp(b, 0, 255));
                    dst[idx * 4 + 3] = 255;
                }
            }
        } else {
            CASPAR_LOG(warning) << L"write_frame_png: unsupported pixel format " << static_cast<int>(desc.format);
            return false;
        }

        // Ensure output directory exists
        auto output_path = std::filesystem::path(path);
        std::filesystem::create_directories(output_path.parent_path());

        // Write PNG
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
