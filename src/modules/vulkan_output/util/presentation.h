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

#include "display_enum.h"

#include <vulkan/vulkan.hpp>

#include <string>

namespace caspar { namespace vulkan_output {

// Presentation tier — ordered from best to worst. The consumer tries each in
// order and falls back if the current tier is unavailable or fails.
enum class presentation_tier
{
    khr_display,              // VK_KHR_display: direct-to-hardware, no compositor
    full_screen_exclusive,    // VK_EXT_full_screen_exclusive: FSE window, DWM bypassed
    borderless,              // Borderless HWND: compositor may be in path
};

// Detect the best presentation tier available for the given physical device and display.
presentation_tier detect_presentation_tier(vk::Instance       instance,
                                           vk::PhysicalDevice physical,
                                           const display_info& display);

// Human-readable tier name for logging.
std::wstring tier_name(presentation_tier tier);

// Result of surface creation — carries the surface and the tier that was actually used.
struct surface_result
{
    vk::SurfaceKHR    surface = {};
    presentation_tier tier    = presentation_tier::borderless;
    uint32_t          width   = 0;
    uint32_t          height  = 0;
};

// Create a Vulkan surface using the best available tier. Tries tiers in order:
// khr_display → full_screen_exclusive → borderless. Falls back on failure.
// The output_window is only created if needed (FSE or borderless path).
// target_refresh_mhz: desired refresh rate in millihertz (e.g. 60000 for 60Hz), 0 for don't care.
surface_result create_tiered_surface(vk::Instance        instance,
                                     vk::PhysicalDevice  physical,
                                     vk::Device          device,
                                     const display_info& display,
                                     uint32_t            target_refresh_mhz);

// Pick the best present mode for externally-paced playout.
// MAILBOX > FIFO_RELAXED > FIFO (unless force_fifo is true for sync barriers).
vk::PresentModeKHR pick_present_mode(vk::PhysicalDevice physical,
                                      vk::SurfaceKHR    surface,
                                      bool              force_fifo = false);

}} // namespace caspar::vulkan_output
