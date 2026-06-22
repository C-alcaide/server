/*
 * Copyright (c) 2026 CasparCG Contributors
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

#include <vulkan/vulkan.h>

#include <string>
#include <vector>

namespace caspar { namespace vulkan_output {

struct display_info
{
    int          gpu_index    = 0;
    int          output_index = 0;
    std::wstring gpu_name;
    std::wstring display_name;
    int          width  = 0;
    int          height = 0;
    int          pos_x  = 0;
    int          pos_y  = 0;

    // VK_KHR_display handle (Linux: always set; Windows: set for Pro tier only)
    VkDisplayKHR vk_display = VK_NULL_HANDLE;
};

// Enumerate all active displays. Each entry contains the display geometry
// and the GPU/output index needed to target it.
std::vector<display_info> enumerate_displays();

// Find a specific display by output_index (1-based) or by display_name substring.
// Returns nullptr if not found.
const display_info* find_display(const std::vector<display_info>& displays,
                                 int                              output_index,
                                 const std::wstring&              display_name = L"");

}} // namespace caspar::vulkan_output
