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

#ifdef CASPAR_GST_GPU_BRIDGE

#include <common/array.h>

#include <core/frame/draw_frame.h>
#include <core/frame/frame_factory.h>

#include <gst/gst.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace caspar { namespace gstreamer {

/**
 * Carries a `GstD3D11Memory` sample to the mixer as NV12 planes, without a host round trip.
 *
 * ── Why planes rather than BGRA ──────────────────────────────────────────────
 * The upstream module's version of this appends `d3d11convert` and hands the mixer one BGRA
 * texture. That works, and it pays twice: a full-frame conversion pass, and 4 bytes per pixel
 * across the bus where NV12 needs 1.5.
 *
 * The cost that matters more is not bandwidth. `d3d11convert` picks the YCbCr matrix and the
 * range itself, and the mixer then treats the result as already-RGB -- so the picture depends
 * on a *second* colour path agreeing with the mixer's. Handing over the two planes instead
 * means the mixer's own `ycbcra_to_rgba` runs, with the channel's colour management, and the
 * caps' colorimetry reaches the shader rather than being consumed on the way.
 *
 * It also means 10-bit survives: `d3d11h265dec` produces P010, carried here at bit16, where
 * BGRA8 would truncate it.
 *
 * ── Not zero-copy, and saying so ─────────────────────────────────────────────
 * The decoded surface belongs to GStreamer's decoder pool and is recycled, so it cannot simply
 * be handed over. One GPU-to-GPU extraction pass per frame writes the two planes into textures
 * this bridge owns; the mixer then copies those into pooled textures. Two full-frame passes at
 * 1.5 bytes/pixel, against three at 4 for the BGRA route.
 *
 * ── Two backends, two mechanisms ─────────────────────────────────────────────
 * OpenGL registers the plane textures with `WGL_NV_DX_interop2` through `ogl::dx_interop`;
 * Vulkan imports shared NT handles through `vulkan::d3d11_import_bridge::copy_planes`. The
 * plane textures are created with `SHARED | SHARED_NTHANDLE` only on the Vulkan path, because
 * GL interop needs neither and the flags are not free.
 *
 * `frame_factory::import_shared_texture` is deliberately absent from this tree, which is why
 * none of upstream's mixer-facing half is reused here.
 *
 * ── One fixed pair of plane textures, not a ring ─────────────────────────────
 * `d3d11_import_bridge::copy_planes` keys its imports on the handle and holds exactly two
 * slots, so rotating through a ring would re-run `OpenSharedResource1` and re-import a VkImage
 * every frame. Ordering comes from `wait_for_previous_copy()` before the extraction draw, not
 * from rotation -- the same shape `av_producer`'s bridge uses for the identical reason.
 *
 * ── The first sample is the test ─────────────────────────────────────────────
 * `html_gpu_bridge` builds its own D3D11 device and self-tests before any frame arrives. This
 * cannot: GStreamer's device only exists once a `GstD3D11Memory` turns up, so construction is
 * lazy and the give-up latch has to cover construction failure as well as per-frame failure.
 */
class gst_gpu_bridge
{
  public:
    gst_gpu_bridge();
    ~gst_gpu_bridge();

    gst_gpu_bridge(const gst_gpu_bridge&)            = delete;
    gst_gpu_bridge& operator=(const gst_gpu_bridge&) = delete;

    /// Is this sample D3D11 memory this bridge could take? Cheap, and does not touch the
    /// device -- the caller uses it to decide whether to try the GPU route at all.
    static bool handles(GstSample* sample);

    /**
     * Extracts the sample's two planes and builds a frame whose `pixel_format` is `nv12`.
     *
     * Returns an empty frame on any failure, having logged the reason ONCE. The caller must
     * then fall back to the host path -- which is not a disaster, only slower, and is why the
     * reason is logged rather than thrown.
     *
     * After `kMaxConsecutiveFailures` the bridge latches off and stops trying, so a permanent
     * fault costs one message rather than fifty a second.
     */
    core::draw_frame to_frame(void*                       tag,
                              core::frame_factory&        frame_factory,
                              GstSample*                  sample,
                              const std::vector<int32_t>& audio_samples);

  private:
    struct impl;
    std::unique_ptr<impl> impl_;
};

}} // namespace caspar::gstreamer

#endif // CASPAR_GST_GPU_BRIDGE
