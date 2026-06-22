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

#include <string>

namespace caspar { namespace vulkan_output {

// Detach a display from the Windows desktop via CCD (Connecting and Configuring
// Displays) API. Required for VK_KHR_display on Windows 11 — the display must
// not be owned by DWM. Identifies the display by GDI device name (e.g. "\\\\.\\DISPLAY3").
// Returns true if successfully detached.
bool detach_display_from_desktop(const std::wstring& device_name);

// Reattach a previously detached display to the Windows desktop.
// Returns true if successfully reattached.
bool reattach_display_to_desktop(const std::wstring& device_name);

// Ensure VK_KHR_display outputs are exported on professional GPUs.
// Detects if a Pro GPU (Quadro/RTX A-series) reports zero KHR_display outputs,
// and if so, spawns configureDriver.exe --set 6 with UAC elevation to enable them.
// Blocks for up to 10 seconds waiting for the driver to reconfigure.
// Returns true if displays were found (either already present or after reconfigure).
bool ensure_vk_khr_display_exported();

}} // namespace caspar::vulkan_output
