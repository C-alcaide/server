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

struct AVBufferRef;

namespace caspar { namespace ffmpeg {

/// Wrap the MIXER'S OWN Vulkan device in an FFmpeg hardware device context.
///
/// FFmpeg 8.1 decodes ProRes, ProRes RAW, FFV1 and DPX with Vulkan COMPUTE shaders, and
/// the only way to get those frames without a copy is for FFmpeg to allocate them on the
/// device the mixer already has. `AVVulkanDeviceContext` is designed for exactly this: the
/// application fills it in and FFmpeg adopts the device rather than creating one.
///
/// WHY NOT LET FFMPEG MAKE ITS OWN DEVICE. It happily will, and then every decoded frame
/// lives on a second logical device and has to be exported and imported to reach the
/// mixer. Measured standalone on this hardware, the readback that implies costs 78% of
/// decode throughput and half the CPU saving -- the whole point of the exercise.
///
/// QUEUES. FFmpeg is given the queue family `device::getComputeQueue()` reserves, never the
/// graphics one. The mixer submits on a single graphics queue with no mutex, relying on
/// everything going through its dispatch thread; FFmpeg submits from its own decode
/// threads. Two submitters on one queue is undefined behaviour, and FFmpeg's internal
/// queue mutexes only cover FFmpeg's own submissions. Handing over a different family
/// removes the question rather than synchronising it -- and if this GPU has no separate
/// compute family, this function refuses instead of quietly sharing.
///
/// Returns nullptr, having logged why, when the mixer is not Vulkan, when there is no
/// isolated queue family, or when FFmpeg rejects the device. A caller must treat that as a
/// reason to use the existing path, never to proceed without a device.
///
/// `vk_device_handle` is `core::frame_factory::gpu_device_handle()`, valid only when
/// `gpu_device_backend()` is Vulkan -- the two backends' device types are unrelated.
AVBufferRef* make_vulkan_hwdevice_from_mixer(void* vk_device_handle);

}} // namespace caspar::ffmpeg
