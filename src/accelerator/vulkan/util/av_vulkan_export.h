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
/// mixer composites packed RGBA and every Vulkan encoder wants planar YUV -- ProRes accepts only
/// `YUV422P10 / YUV444P10 / YUVA444P10` -- but the conversion is not this class's job. FFmpeg's
/// `libplacebo` filter does it on the GPU, so the hand-off is packed to packed and the encoder's
/// format is negotiated downstream. Measured 2026-08-22: `scale_vulkan` cannot do it (RGB to
/// only NV12/YUV420P/YUV444P, all 8-bit, and it mangles the levels) while
/// `libplacebo=format=yuv422p10` produces valid ProRes HQ.
///
/// THE COMPOSITE MUST BE THE 16-BIT ONE, which is enforced by the caller rather than here. The
/// mixer's 8-bit attachment holds BGRA bytes and `vf_libplacebo` exchanges red and blue on a
/// BGRA Vulkan frame; the 16-bit attachment holds RGBA, which it handles correctly. This class
/// copies bytes and does not care, so the guard lives where the decision is made --
/// `ffmpeg_consumer.cpp`, whose `make_vulkan_frames_ctx` carries the measurement.
struct av_plane_dest
{
    void*    image     = nullptr; //< VkImage, from AVVkFrame::img[0]
    void*    semaphore = nullptr; //< VkSemaphore (timeline), from AVVkFrame::sem[0]
    uint64_t sem_value = 0;       //< AVVkFrame::sem_value[0] -- the value BEFORE this copy
    int      layout    = 0;       //< AVVkFrame::layout[0], as a VkImageLayout value
    int      width     = 0;
    int      height    = 0;

    // ── Filled BY `copy_from_texture`, and the caller MUST write both back ──────────
    //
    // An AVVkFrame is shared bookkeeping, not just a handle: whoever signals the timeline
    // semaphore owns telling FFmpeg the new value, and whoever moves the image owns telling
    // FFmpeg the new layout. Leaving either stale is not cosmetic. Measured 2026-08-22 under
    // the validation layer, with the writeback missing:
    //
    //   VUID-VkSubmitInfo2-semaphore-03882, vkQueueSubmit2(): pSubmits[0]
    //     .pSignalSemaphoreInfos[1].semaphore signal value (1) must be greater than the
    //     current timeline semaphore value
    //
    // -- FFmpeg's own submit, signalling a value this copy had already consumed, on every
    // frame. It encoded a correct picture anyway on this driver, so nothing failed and nothing
    // was logged; signalling a non-increasing timeline value is undefined behaviour and the
    // next driver is entitled to deadlock on it.
    //
    // These are outputs rather than something the caller computes, so the rule stays in one
    // place: duplicating "signal is sem_value + 1, and UNDEFINED becomes GENERAL" in the
    // ffmpeg module would leave two copies to keep in step.
    uint64_t signalled_value = 0; //< write to AVVkFrame::sem_value[0]
    int      final_layout    = 0; //< write to AVVkFrame::layout[0]
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
    bool copy_from_texture(const std::shared_ptr<core::texture>& src, av_plane_dest& dest);

    /// Fill `dest` with opaque black, signalling `dest.sem_value + 1` the same way a copy does.
    ///
    /// FOR A CHANNEL THAT IS COMPOSITING NOTHING. `const_frame::empty()` carries no texture, so
    /// there is nothing to copy from -- and the consumer used to treat that as a GPU-direct
    /// FAILURE. A recording consumer declared in `casparcg.config` starts before anything is
    /// played, so the path was abandoned on its very first frame, every time -- and the channel
    /// then spent the rest of the recording doing a CPU readback nobody consumed, which is the
    /// one cost this whole class exists to remove. Measured 2026-08-23: the mixer logs "CPU
    /// readback SKIPPED" for a configuration that used to log "CPU readback required by
    /// consumer ffmpeg" four seconds in and never recover.
    ///
    /// Black is not a guess here. The Vulkan mixer attaches a texture to every frame it
    /// composites (`image_mixer.cpp`, `make_result`), so an absent one means no composition
    /// happened -- an idle channel, whose picture IS black. That is also exactly what the host
    /// path records for the same frame, so this keeps the two paths recording the same thing.
    bool clear_to_black(av_plane_dest& dest);

  private:
    /// The one submission both entry points use. `source` null means clear rather than copy;
    /// the arguments are `void*` so the header does not have to name the mixer's own types.
    bool submit(av_plane_dest& dest, void* wrapper_ptr, void* source_ptr);

    struct impl;
    std::unique_ptr<impl> impl_;
};

}}} // namespace caspar::accelerator::vulkan
