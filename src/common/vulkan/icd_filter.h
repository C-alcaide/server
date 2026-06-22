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

namespace caspar { namespace vulkan_common {

/// Filter stale NVIDIA Vulkan ICDs on Windows.
///
/// After a driver upgrade, old driver entries may remain in the Windows
/// DriverStore.  The Vulkan loader discovers ALL ICD JSON files and
/// dispatches vkCreateDevice through each ICD.  A stale ICD (from the
/// previous driver) can hang the GPU when it tries to talk to the current
/// kernel-mode driver, causing TDR exactly 2 seconds later.
///
/// This function finds all NVIDIA ICD JSON files, keeps only the newest,
/// and sets VK_DRIVER_FILES to force the loader to use only the current ICD.
///
/// Must be called before any vkCreateInstance().  Thread-safe (internally
/// guarded by call_once).  No-op on non-Windows platforms.
void filter_stale_nvidia_icds();

}} // namespace caspar::vulkan_common
