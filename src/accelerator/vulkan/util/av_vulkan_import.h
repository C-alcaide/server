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

#include <common/bit_depth.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace caspar { namespace core {
class texture;
}} // namespace caspar::core

namespace caspar { namespace accelerator { namespace vulkan {

/// One plane of an FFmpeg Vulkan frame, described without Vulkan types.
///
/// Pure-std for the same reason as `shared_device_info`: the caller is the ffmpeg module,
/// which has FFmpeg's headers but not the Vulkan SDK's. The handles are dispatchable
/// Vulkan handles, i.e. pointers, so `void*` round-trips them; `layout` carries a
/// `VkImageLayout` as its integer value.
struct av_plane_source
{
    void*    image     = nullptr; //< VkImage, from AVVkFrame::img[i]
    void*    semaphore = nullptr; //< VkSemaphore (timeline), from AVVkFrame::sem[i]
    uint64_t sem_value = 0;       //< AVVkFrame::sem_value[i] -- the value to WAIT on
    int      layout    = 0;       //< AVVkFrame::layout[i], as a VkImageLayout value
    int      width     = 0;
    int      height    = 0;
    int      components = 1;      //< 1 for Y/Cb/Cr planes; 2 for a semi-planar CbCr plane
};

/// Copies an FFmpeg Vulkan frame's planes into pooled mixer textures.
///
/// COPY RATHER THAN ALIAS, which is the same decision `d3d11_import_bridge` made and for the
/// same reason: the decoder's pool is shallow and stalls if its frames are held, and handing
/// a decoder-owned image to the mixer would tie the mixer's sampling to FFmpeg's frame
/// lifetime and layout transitions. This is a device-local copy -- nothing goes near host
/// memory, which is the point of the exercise.
///
/// A CLASS RATHER THAN A FUNCTION, and that is not decoration. The command buffer and fence
/// are allocated once and reused, exactly as `d3d11_import_bridge` does: `device`'s command
/// pool is externally synchronised and `allocateCommandBuffers` hops to the device thread,
/// so allocating per frame both leaks the buffer and defeats that thread's own recycling.
/// The first version of this was a free function that allocated one per call.
class av_vulkan_importer final
{
  public:
    /// `vk_device` is `core::frame_factory::gpu_device_handle()`, valid only when
    /// `gpu_device_backend()` is Vulkan -- the two backends' device types are unrelated.
    /// Throws if it is null.
    explicit av_vulkan_importer(void* vk_device);
    ~av_vulkan_importer();

    av_vulkan_importer(const av_vulkan_importer&)            = delete;
    av_vulkan_importer& operator=(const av_vulkan_importer&) = delete;

    /// Copy one frame's planes into freshly pooled mixer textures.
    ///
    /// SYNCHRONISATION, and it is not optional. `AVVkFrame`'s timeline semaphores carry a
    /// contract: a client must WAIT on `sem_value` in every submission that touches the
    /// image and SIGNAL an incremented value. This submission does both, per plane.
    /// Honouring the signal is what makes it safe for the caller to release the AVFrame
    /// immediately -- FFmpeg waits on the latest value before reusing a pooled frame -- so
    /// **the caller must add 1 to each `AVVkFrame::sem_value` after this returns true**, or
    /// FFmpeg will wait on a value that has already passed and reuse an image this copy is
    /// still reading.
    ///
    /// The work is recorded and submitted through the device's own dispatch thread, so it is
    /// serialised against the mixer's submissions on the same queue rather than racing them.
    /// Being on that queue is also what orders the copy before the mixer's later draw, so no
    /// host wait is needed to make the textures safe to sample. The fence exists for a
    /// different reason: the command buffer is reused, and re-recording one the GPU is still
    /// reading is undefined behaviour.
    ///
    /// LAYOUTS are restored: each source image is transitioned back to the layout it arrived
    /// in, so `AVVkFrame::layout[]` -- documented as updated after every barrier -- stays
    /// true with nothing for the caller to write back. A plane arriving in
    /// VK_IMAGE_LAYOUT_UNDEFINED is refused rather than copied, since its contents are
    /// undefined by definition.
    ///
    /// QUEUE FAMILY. The copy runs on the mixer's graphics queue, which is a different
    /// family from the one FFmpeg decodes on. That is only sound because the frames are
    /// allocated VK_SHARING_MODE_CONCURRENT -- `make_vulkan_hwdevice_from_mixer` names a
    /// second queue family for exactly this reason, and its comment carries the argument. An
    /// exclusively owned image would need a queue family ownership transfer whose release
    /// half can only be submitted on FFmpeg's own queue.
    ///
    /// Returns false having logged why, leaving `out` untouched. The caller must then fall
    /// back to a software path; there is no partial success.
    bool copy_planes(const std::vector<av_plane_source>&          planes,
                     common::bit_depth                            depth,
                     std::vector<std::shared_ptr<core::texture>>& out);

  private:
    struct impl;
    std::unique_ptr<impl> impl_;
};

}}} // namespace caspar::accelerator::vulkan
