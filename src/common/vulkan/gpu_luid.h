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

// Shared Vulkan GPU identification utilities used by both the mixer
// (accelerator::vulkan) and the output consumer (vulkan_output) to
// ensure consistent GPU indexing across subsystems.

#include <vulkan/vulkan.h>

#include <array>
#include <cstdint>
#include <vector>

namespace caspar { namespace vulkan_common {

/// Query the LUID of a physical device. Returns true if the LUID is valid.
bool query_device_luid(VkPhysicalDevice device, std::array<uint8_t, 8>& out_luid);

/// Deduplicate VkPhysicalDevices by LUID.
///
/// Newer NVIDIA drivers expose the same physical GPU as multiple
/// VkPhysicalDevice handles (graphics vs compute/video queue families).
/// This function keeps only the first occurrence of each unique LUID so
/// that user-facing GPU indices remain stable regardless of driver version.
///
/// Devices without a valid LUID are always kept (cannot be matched cross-API).
std::vector<VkPhysicalDevice> deduplicate_by_luid(const std::vector<VkPhysicalDevice>& devices);

}} // namespace caspar::vulkan_common
