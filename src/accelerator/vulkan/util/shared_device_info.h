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

#include <cstdint>
#include <string>
#include <vector>

namespace caspar { namespace accelerator { namespace vulkan {

/// Enough of the mixer's Vulkan device to hand it to another API, as plain types.
///
/// Deliberately a pure-std interface with no Vulkan types in it, for the same reason
/// `dxgi_adapter_for_vk_device` takes a `void*`: the consumer is the ffmpeg module, which
/// has FFmpeg's headers but not the Vulkan SDK's, and `device.h` pulls in `vulkan.hpp`.
/// Including it there fails with `C1083: vulkan.hpp` before anything else is even tried.
///
/// The handles are Vulkan dispatchable handles, which are pointers, so `void*` round-trips
/// them exactly. `get_proc_addr` is a function pointer and is carried the same way; the
/// consumer casts it back to `PFN_vkGetInstanceProcAddr`.
struct shared_device_info
{
    bool     valid              = false;
    void*    instance           = nullptr;  //< VkInstance
    void*    physical_device    = nullptr;  //< VkPhysicalDevice
    void*    device             = nullptr;  //< VkDevice
    void*    get_proc_addr      = nullptr;  //< PFN_vkGetInstanceProcAddr
    uint32_t graphics_qf        = 0;
    uint32_t decode_qf         = 0;
    /// False when this GPU has no compute family distinct from graphics, in which case
    /// `decode_qf == graphics_qf` and sharing would mean sharing a queue.
    bool     decode_qf_isolated = false;
    /// Device extensions the mixer enabled. Another API sharing the device may rely on
    /// these and nothing else.
    std::vector<std::string> enabled_device_extensions;
};

/// Describe the mixer's Vulkan device. `vk_device` is an `accelerator::vulkan::device*`,
/// as returned by `core::frame_factory::gpu_device_handle()` when the backend is Vulkan.
/// Returns `valid == false` for a null handle.
shared_device_info describe_shared_device(void* vk_device);

}}} // namespace caspar::accelerator::vulkan
