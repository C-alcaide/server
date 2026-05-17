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

#include <common/log.h>
#include <common/utf.h>

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

        // Get BGRA pixel data (triggers lazy GPU readback if needed)
        const auto& bgra_data = frame.image_data(0);
        if (bgra_data.size() == 0) {
            CASPAR_LOG(warning) << L"write_frame_png: no image data";
            return false;
        }

        const auto num_pixels = static_cast<size_t>(width) * static_cast<size_t>(height);

        // Convert BGRA → RGBA for PNG output
        std::vector<uint8_t> rgba(num_pixels * 4);
        const auto*          src = bgra_data.data();
        auto*                dst = rgba.data();

        for (size_t i = 0; i < num_pixels; ++i) {
            dst[i * 4 + 0] = src[i * 4 + 2]; // R ← B
            dst[i * 4 + 1] = src[i * 4 + 1]; // G ← G
            dst[i * 4 + 2] = src[i * 4 + 0]; // B ← R
            dst[i * 4 + 3] = src[i * 4 + 3]; // A ← A
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
