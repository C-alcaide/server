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

#ifdef _WIN32

#include <common/bit_depth.h>
#include <core/frame/frame.h>

#include <memory>

namespace caspar { namespace accelerator { namespace vulkan {

/**
 * Imports D3D11 shared textures as VkImages on the mixer's device, and copies
 * them into pooled mixer textures the image_mixer can bind directly.
 *
 * This is the Vulkan half of the ffmpeg producer's GPU-direct decode path. The
 * OpenGL half is `WGL_NV_DX_interop2` in av_producer.cpp; this is the
 * equivalent through `VK_KHR_external_memory_win32`.
 *
 * ── Why two single-plane imports, not one multi-planar one ───────────────────
 * A hardware-decoded frame is NV12 (or P010/P016), so importing the decoded
 * surface as one `eG8B8R82Plane420Unorm` image and taking `ePlane0`/`ePlane1`
 * views looks like the obvious shape. It was measured and does not work: on the
 * reference GPU the two APIs disagree about where plane 1 lives inside a shared
 * allocation. Luma comes across byte-identical; chroma comes back with 844800
 * of 1036800 bytes wrong, in a tiling pattern rather than an offset, and there
 * is no arithmetic that recovers it. Isolated with a four-case control:
 * Vulkan-writes-Vulkan-reads is correct both on a native and on an imported
 * allocation, so Vulkan's plane machinery is sound and self-consistent -- it is
 * specifically D3D11-writes/Vulkan-reads that breaks. See
 * docs/GPU_INTEROP_PLAN.md item 1.
 *
 * So the producer extracts the two planes into two ordinary single-plane D3D11
 * textures first (R8 for Y, R8G8 for interleaved CbCr, which is exactly what
 * the OpenGL path already builds), and each is imported here as a plain
 * single-plane VkImage. That round-trips byte-identically, and it removes any
 * dependence on the two APIs agreeing about multi-planar layout -- a better
 * property to rely on than a fallback.
 *
 * ── Threading ────────────────────────────────────────────────────────────────
 * All Vulkan work is dispatched onto the device's thread, which is where the
 * command pool and queue are externally synchronised.
 */
class d3d11_import_bridge
{
  public:
    /// `vk_device` is an `accelerator::vulkan::device*`, as returned by
    /// `core::frame_factory::gpu_device_handle()` when the backend is Vulkan.
    /// Throws if it is null.
    explicit d3d11_import_bridge(void* vk_device);
    ~d3d11_import_bridge();

    d3d11_import_bridge(const d3d11_import_bridge&)            = delete;
    d3d11_import_bridge& operator=(const d3d11_import_bridge&) = delete;

    /**
     * Imports the two plane handles (once; cached until they change) and copies
     * both into freshly pooled mixer textures, which is what makes the shared
     * D3D11 textures free for the next frame.
     *
     * `depth` selects the plane formats: bit8 gives R8 / R8G8 (NV12), bit16
     * gives R16 / R16G16 (P010/P016).
     *
     * Returns false and leaves the outputs untouched on any failure; the caller
     * must then fall back to the host transfer path.
     */
    bool copy_planes(void*                           y_handle,
                     void*                           uv_handle,
                     int                             y_width,
                     int                             y_height,
                     int                             uv_width,
                     int                             uv_height,
                     common::bit_depth               depth,
                     std::shared_ptr<core::texture>& out_y,
                     std::shared_ptr<core::texture>& out_uv);

    /**
     * Single-plane variant, for a source that is one packed 4-byte surface rather
     * than two YCbCr planes -- a browser's composited output, say.
     *
     * The image is imported as `eR8G8B8A8Unorm` whether the source calls itself
     * BGRA or RGBA. The two are byte-compatible and the copy is a straight memory
     * move, so importing under a single format keeps `vkCmdCopyImage`'s
     * size-compatibility rule out of the picture; the byte order travels to the
     * mixer on the frame's `pixel_format` instead, where the shader swizzle costs
     * nothing.
     *
     * Imports are cached by handle, in a small ring. That assumes the caller
     * rotates through a *stable* set of handles -- its own staging textures, not
     * handles borrowed from someone else's pool. A handle whose underlying
     * resource can change behind the same value must not be passed here.
     *
     * Returns false and leaves `out` untouched on any failure.
     */
    bool copy_texture(void* handle, int width, int height, std::shared_ptr<core::texture>& out);

    /**
     * Blocks until the copy submitted by the previous `copy_planes()` or
     * `copy_texture()` has completed on the GPU. The caller must call this before
     * letting D3D11 overwrite the shared textures, because nothing else orders the
     * Vulkan read against the next D3D11 write.
     *
     * Returns the time spent waiting, in microseconds.
     */
    std::int64_t wait_for_previous_copy();

    /**
     * Waits for any copy in flight and frees the imported VkImages, without
     * destroying the bridge. The caller must do this before releasing the D3D11
     * textures the imports alias -- on a frame-size change, for instance, where
     * the textures are rebuilt but the bridge carries on.
     */
    void release_imports();

  private:
    struct impl;
    std::unique_ptr<impl> impl_;
};

/**
 * The DXGI adapter index whose LUID matches `vk_device`, or -1 if it cannot be
 * determined.
 *
 * A caller that creates a D3D11 device to feed this bridge has to put it on the
 * same physical adapter as the mixer, or the shared handles it produces are not
 * importable here ("this D3D11 handle is not importable"). The producer decides
 * that before any Vulkan type is in scope, so the lookup lives on this side of
 * the boundary and takes the device as void*, exactly like the constructor.
 *
 * `vk_device` is an `accelerator::vulkan::device*`, as returned by
 * `core::frame_factory::gpu_device_handle()` when the backend is Vulkan.
 */
int dxgi_adapter_for_vk_device(void* vk_device);

}}} // namespace caspar::accelerator::vulkan

#endif // _WIN32
