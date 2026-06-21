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

#include "blend_mask_loader.h"

#include <common/except.h>
#include <common/log.h>
#include <common/utf.h>

#include <boost/filesystem.hpp>

#include <lodepng.h>

namespace caspar { namespace core {

std::shared_ptr<blend_mask_data> load_blend_mask(const std::wstring& filename)
{
    auto path_str = u8(filename);

    if (!boost::filesystem::exists(path_str))
        CASPAR_THROW_EXCEPTION(file_not_found() << msg_info("Blend mask file not found: " + path_str));

    // Decode the PNG into 8-bit RGB.
    std::vector<unsigned char> rgb;
    unsigned                   width  = 0;
    unsigned                   height = 0;

    unsigned error = lodepng::decode(rgb, width, height, path_str, LCT_RGB, 8);
    if (error) {
        CASPAR_THROW_EXCEPTION(file_read_error()
                               << msg_info("Failed to decode blend mask PNG '" + path_str + "': " +
                                           lodepng_error_text(error)));
    }

    if (width == 0 || height == 0)
        CASPAR_THROW_EXCEPTION(file_read_error() << msg_info("Blend mask PNG has zero size: " + path_str));

    auto result    = std::make_shared<blend_mask_data>();
    result->width  = static_cast<int>(width);
    result->height = static_cast<int>(height);
    result->data.resize(static_cast<size_t>(width) * height * 3);

    constexpr float inv255 = 1.0f / 255.0f;
    for (size_t i = 0; i < result->data.size(); ++i) {
        result->data[i] = static_cast<float>(rgb[i]) * inv255;
    }

    CASPAR_LOG(info) << L"[blend_mask_loader] Loaded blend mask: " << filename << L" (" << width << L"x" << height
                     << L")";

    return result;
}

}} // namespace caspar::core
