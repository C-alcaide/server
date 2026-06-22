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

#include <memory>
#include <string>

namespace caspar { namespace vulkan_output {

// A borderless fullscreen window covering a specific display output.
// Used for Vulkan presentation without DWM composition artifacts.
class output_window
{
  public:
    // Create a borderless fullscreen window on the specified display.
    explicit output_window(const display_info& display);
    ~output_window();

    output_window(const output_window&)            = delete;
    output_window& operator=(const output_window&) = delete;

    int width() const;
    int height() const;

    // Create a Vulkan surface for this window.
    vk::SurfaceKHR create_surface(vk::Instance instance);

    // Returns true if the window received a close/destroy event.
    bool should_close() const;

  private:
    struct impl;
    std::unique_ptr<impl> impl_;
};

}} // namespace caspar::vulkan_output
