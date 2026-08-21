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
#include <memory>

namespace caspar { namespace core {
class texture;
}} // namespace caspar::core

namespace caspar { namespace accelerator { namespace vulkan {

/// The destination plane of an FFmpeg Vulkan frame, described without Vulkan types.
///
/// Pure-std for the same reason as `av_plane_source` and `shared_device_info`: the caller is
/// the ffmpeg module, which has FFmpeg's headers but not the Vulkan SDK's. The handles are
/// dispatchable Vulkan handles, i.e. pointers, so `void*` round-trips them; `layout` carries a
/// `VkImageLayout` as its integer value.
///
/// ONE PLANE IS ENOUGH, and that is the whole reason this is simpler than the importer. The
/// mixer composites BGRA and every Vulkan encoder wants planar YUV -- ProRes accepts only
/// `YUV422P10 / YUV444P10 / YUVA444P10` -- but the conversion is not this class's job. FFmpeg's
/// `libplacebo` filter does it on the GPU, so the hand-off is BGRA to BGRA and the encoder's
/// format is negotiated downstream. Measured 2026-08-22: `scale_vulkan` cannot do it (RGB to
/// only NV12/YUV420P/YUV444P, all 8-bit) while `libplacebo=format=yuv422p10` produces valid
/// ProRes HQ.
struct av_plane_dest
{
    void*    image     = nullptr; //< VkImage, from AVVkFrame::img[0]
    void*    semaphore = nullptr; //< VkSemaphore (timeline), from AVVkFrame::sem[0]
    uint64_t sem_value = 0;       //< AVVkFrame::sem_value[0] -- the value BEFORE this copy
    int      layout    = 0;       //< AVVkFrame::layout[0], as a VkImageLayout value
    int      width     = 0;
    int      height    = 0;
};

/// Copies the mixer's composited image into a frame FFmpeg owns, for a Vulkan encoder.
///
/// The mirror of `av_vulkan_importer`, and deliberately built the same way: one command buffer
/// and one fence for the life of the object rather than per frame, the wait done on the host
/// before anything is recorded, layouts restored so FFmpeg's `AVVkFrame::layout[]` bookkeeping
/// stays true, and a refusal rather than a guess whenever the frame is not as expected.
///
/// ── Why it copies instead of wrapping ────────────────────────────────────────
/// Handing the encoder the mixer's own image would be zero-copy and wrong. The composite comes
/// from a small recycling attachment pool whose handles are deliberately stable across frames,
/// and an encoder holds a frame for several ticks -- so the mixer would overwrite a frame still
/// being encoded, intermittently, with no error anywhere. `cuda_vk_uploader` copies for exactly
/// this reason. Both sides live on one `VkDevice`, so this is a same-device image copy rather
/// than a transfer across a bus.
class av_vulkan_exporter
{
  public:
    /// `vk_device` is an `accelerator::vulkan::device*`, as `describe_shared_device` takes.
    explicit av_vulkan_exporter(void* vk_device);
    ~av_vulkan_exporter();

    av_vulkan_exporter(const av_vulkan_exporter&)            = delete;
    av_vulkan_exporter& operator=(const av_vulkan_exporter&) = delete;

    /// Copy `src` into `dest`, signalling `dest.sem_value + 1` when the copy completes.
    ///
    /// `src` must be a Vulkan `texture_wrapper`. Returns false -- having logged once -- on any
    /// mismatch or failure, and the caller must then fall back to the host path rather than
    /// send a frame that may hold anything.
    bool copy_from_texture(const std::shared_ptr<core::texture>& src, const av_plane_dest& dest);

  private:
    struct impl;
    std::unique_ptr<impl> impl_;
};

}}} // namespace caspar::accelerator::vulkan
