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

#ifdef CASPAR_GST_CUDA_EGRESS

#include <core/frame/frame.h>
#include <core/video_format.h>

#include <gst/gst.h>

#include <memory>
#include <string>

namespace caspar { namespace gstreamer {

/**
 * Hands the channel's composited Vulkan texture to GStreamer as CUDA memory, so an encoder
 * takes it without the frame going through host memory.
 *
 * ── IT COSTS MORE CPU THAN THE READBACK IT REPLACES. Read this before enabling it ──
 * This route is correct and it is not, on the reference machine, an optimisation. Measured
 * with `cli.py gst-consumer-cost --mixer vulkan`, 1080p50, one consumer, `nvh264enc` on both
 * consumer arms, three interleaved rounds:
 *
 *     no consumer                    1.84 cores
 *     host readback -> nvh264enc     1.97 cores   (+0.13)
 *     CUDA direct   -> nvh264enc     2.04 cores   (+0.20)
 *
 * So enabling `GPU` costs about 0.07 cores MORE than leaving it off. Late frames are within
 * noise on all three arms (0-2 per 1500) and do not separate them.
 *
 * That number is after four rounds of tuning it, each of which is still in the code because
 * each is right on its own terms:
 *
 *   | change                                            | cores |
 *   | :------------------------------------------------ | ----: |
 *   | first working version, allocate per frame          |  2.24 |
 *   | `GstCudaBufferPool` instead                        |  2.09 |
 *   | our own stream, not a device-wide sync             |  2.08 |
 *   | `cudaEventBlockingSync` instead of a spinning wait |  2.03 |
 *   | GPU-side semaphore ordering, no CPU fence wait     |  2.04 |
 *
 * Only the first two moved it much, and the remaining gap is unexplained -- the obvious
 * suspects (per-frame allocation, device-wide synchronisation, a spin, a CPU fence wait) were
 * each tested and each was either not the cost or not the whole of it.
 *
 * **So why is it here?** Because it is correct, opt-in, and the conditions this was measured
 * under are narrow ones: one consumer, 1080p, a machine doing nothing else. The readback
 * scales with pixels and with consumer count; this route's per-frame CPU largely does not. At
 * 4K, or with several consumers, or on a box already CPU-bound, the ordering may reverse. It
 * is off unless asked for, and `gstreamer/gpu-frames` in INFO says whether it engaged.
 *
 * ── What it is measured BY ──────────────────────────────────────────────────
 * The picture: `gst_cuda_egress` output downloaded through `cudadownload` and compared
 * against the channel's own IMAGE capture -- **max 0 LSB**, with an R/B-swapped control at
 * mean 97 so the comparison demonstrably has teeth, over 399/399 GPU-direct frames.
 *
 * The shared `cuda_vk_uploader` this leans on is gated by
 * `cli.py encode-parity --codec h264 --gpu-arm nvenc`, an arm that did not exist before this
 * work: every battery touching the FFmpeg consumer's GPU path ran the *Vulkan* encoder, which
 * reaches the encoder through entirely different code, so that file had no coverage at all.
 *
 * ── Why CUDA and not D3D11, which looked easier ─────────────────────────────
 * `nvd3d11h264enc` accepts D3D11 memory and would seem to let the existing
 * `d3d11_import_bridge` machinery run in reverse. It cannot: the Vulkan mixer exports its
 * textures as `VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32`, and
 * `ID3D11Device1::OpenSharedResource1` will not open an opaque Vulkan handle -- it wants one
 * D3D11 itself created. The ingest bridge works in its direction precisely because D3D11
 * allocates there and Vulkan imports; reversing the data flow does not reverse that.
 *
 * CUDA imports `OPAQUE_WIN32` without complaint, which is why `cuda_vk_strategy` in the
 * DeckLink consumer and `cuda_vk_uploader` in the FFmpeg consumer both already do it. This is
 * the third user of a route that is proven in this tree, not a new one.
 *
 * ── One context, and why that is a safety property ──────────────────────────
 * GStreamer creates the `GstCudaContext` and this takes its `CUcontext` for the uploader,
 * rather than the other way round. A CUDA device pointer is **not valid across contexts**, and
 * using one from the wrong context does not return an error -- it takes the process down with
 * an access violation (`cuda_vk_upload.h`). Sourcing both from one object makes the mismatch
 * unrepresentable instead of merely unlikely.
 *
 * ── Not zero-copy, and the name would be wrong ──────────────────────────────
 * One device-to-device copy per frame, from the mixer's attachment into a pitched buffer the
 * encoder owns. What it removes is the readback and the re-upload, not all copying. The
 * import itself is cached: `cudaImportExternalMemory` costs tens of milliseconds, more than a
 * frame, and the mixer's attachment pool recycles a small set of allocations so a handful of
 * cache entries cover a channel indefinitely.
 *
 * Vulkan mixer only. `ogl::texture` implements no `export_native_handle`, so an OpenGL channel
 * has nothing to import and falls back to the host path.
 */
class gst_cuda_egress
{
  public:
    gst_cuda_egress();
    ~gst_cuda_egress();

    gst_cuda_egress(const gst_cuda_egress&)            = delete;
    gst_cuda_egress& operator=(const gst_cuda_egress&) = delete;

    /// Is a CUDA device present at all? Cheap, and safe to call before constructing.
    static bool available();

    /**
     * Prepares the context and the buffer ring for this format. Returns false with `reason`
     * set if the route cannot be taken, which the caller must treat as "use host memory"
     * rather than as an error.
     */
    bool open(const core::video_format_desc& format_desc, std::wstring& reason);

    /**
     * A `GstBuffer` holding the frame's pixels as CUDA memory, or null.
     *
     * Null means this frame must go through the host path; the reason is logged once. A
     * caller that treats null as fatal will drop frames for a condition that is recoverable
     * by definition.
     */
    GstBuffer* wrap(const core::const_frame& frame);

    /// The caps an appsrc must offer for the buffers `wrap` returns.
    GstCaps* caps() const;

  private:
    struct impl;
    std::unique_ptr<impl> impl_;
};

}} // namespace caspar::gstreamer

#endif // CASPAR_GST_CUDA_EGRESS
