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
    void*    image     = nullptr; //< VkImage, from AVVkFrame::img[image_index]
    void*    semaphore = nullptr; //< VkSemaphore (timeline), from AVVkFrame::sem[image_index]
    uint64_t sem_value = 0;       //< AVVkFrame::sem_value[image_index] -- the value to WAIT on
    int      layout    = 0;       //< AVVkFrame::layout[image_index], as a VkImageLayout value
    int      width     = 0;
    int      height    = 0;
    int      components = 1;      //< 1 for Y/Cb/Cr planes; 2 for a semi-planar CbCr plane

    /// WHICH IMAGE, because a plane does not always get one of its own.
    ///
    /// FFmpeg hands a frame over as either one image per plane or a single MULTI-PLANAR image
    /// whose planes are aspects of it, and which one is not ours to choose. The compute
    /// decoders can be asked for the first via `AV_VK_FRAME_FLAG_DISABLE_MULTIPLANE`; a
    /// `VK_KHR_video_decode` decoder cannot, because `avcodec_get_hw_frames_parameters`
    /// pre-sets `AVVulkanFramesContext::format[0]` to the format its decode profile requires
    /// and `vulkan_frames_init` then passes a literal 0 for `disable_multiplane`
    /// (hwcontext_vulkan.c, the `format[0] != VK_FORMAT_UNDEFINED` branch). So h264/hevc
    /// ALWAYS arrive as one image, whatever we ask for.
    ///
    /// Several plane sources may therefore name the same image, which is why the importer
    /// groups by this for anything that is per-image rather than per-plane: the layout
    /// transition, the timeline wait, and the signal. Transitioning one image twice in a
    /// batch is a hazard; signalling one timeline twice in a submit is invalid.
    int image_index = 0;
    /// -1 for a whole-image plane (`VK_IMAGE_ASPECT_COLOR_BIT`), else the aspect plane index
    /// 0..2 (`VK_IMAGE_ASPECT_PLANE_0/1/2_BIT`). This mirrors FFmpeg's own mapping in
    /// `libavutil/vulkan.c`, `ff_vk_aspect_flag`.
    int aspect_plane = -1;
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
    /// SYNCHRONISATION follows `AVVkFrame`'s contract, with one deliberate departure.
    ///
    /// SIGNAL, as documented: this submission signals `sem_value + 1` per plane, and **the
    /// caller must record that on the frame** (`av_producer.cpp` does, under `lock_frame`).
    /// FFmpeg waits on the recorded value before reusing a pooled frame, so it is what stops
    /// the decoder overwriting an image this copy is still reading.
    ///
    /// WAIT on the host instead of in the submission, which is the departure. A wait here
    /// would sit on the mixer's single shared graphics queue, so a value not yet signalled
    /// would block every other channel behind it -- head-of-line blocking on the one queue
    /// that must never stall. It happens on the host before anything is recorded, and a frame
    /// that is not ready within a second is DECLINED rather than allowed to wedge the queue.
    ///
    /// Worth knowing that the signal was accused of losing the GPU with concurrent producers
    /// and acquitted. The A/B that convicted it was invalid -- the arm without it had one of
    /// four producers active, so it was not doing concurrent work -- and repeats then failed
    /// in both arms. The real cause was one `AVHWDeviceContext` per producer, and therefore
    /// one copy of FFmpeg's queue mutex per producer guarding a single queue; see
    /// `make_vulkan_hwdevice_from_mixer`. Eight concurrent producers are clean.
    ///
    /// The work is submitted through the device's own dispatch thread, on the same queue the
    /// mixer draws with, which is what orders the copy before any later sampling of these
    /// textures without a host wait. The fence exists for a different reason again: the
    /// command buffer is reused, and re-recording one the GPU is still reading is undefined
    /// behaviour.
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
