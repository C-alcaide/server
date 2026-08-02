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

#include <core/frame/frame.h>

#include <memory>
#include <string>

namespace caspar { namespace accelerator { namespace ogl {

/**
 * WGL_NV_DX_interop2, wrapped: registers D3D11 textures with the mixer's GL context
 * and copies them into pooled mixer textures.
 *
 * The OpenGL counterpart of vulkan::d3d11_import_bridge. Both exist so a producer
 * holding a D3D11 texture can reach either mixer without knowing which one it has;
 * the mechanisms are unrelated (Vulkan imports a shared NT handle through
 * VK_KHR_external_memory_win32, this registers the D3D11 object itself with GL) but
 * the shape of the operation is the same.
 *
 * ── Threading ────────────────────────────────────────────────────────────────
 * Every entry point here dispatches onto the mixer's GL thread. The GL names and
 * the interop device belong to that context, and a private shared context is not an
 * alternative: wglShareLists fails against a context that is current on another
 * thread, which the mixer's always is.
 *
 * ── Synchronisation ──────────────────────────────────────────────────────────
 * The lock/unlock pair is the ordering primitive -- wglDXLockObjectsNV makes the
 * D3D11 writes visible to GL and wglDXUnlockObjectsNV releases the object back. No
 * explicit fence is needed on this path, unlike the Vulkan one.
 */
class dx_interop
{
  public:
    /**
     * Opens an interop device for `d3d11_device` (an `ID3D11Device*`) against the GL
     * context of `ogl_device` -- an `accelerator::ogl::device*`, as returned by
     * `core::frame_factory::gpu_device_handle()` when the backend is OpenGL. Taken as
     * void* for the same reason d3d11_import_bridge does: it keeps GLEW, and the GL
     * headers generally, on this side of the boundary so a module can use the bridge
     * without linking them.
     *
     * Returns null with the reason in `reason` if the extension is missing or the
     * open fails.
     *
     * A failed open is also how the adapter is checked. WGL_NV_DX_interop2 only
     * works when the D3D11 device and the GL context are on the same physical
     * adapter, and wglDXOpenDeviceNV is the cheapest -- and only reliable -- way to
     * ask. There is no LUID query for a GL context to compare against.
     */
    static std::unique_ptr<dx_interop> create(void* ogl_device, void* d3d11_device, std::wstring& reason);

    ~dx_interop();

    dx_interop(const dx_interop&)            = delete;
    dx_interop& operator=(const dx_interop&) = delete;

    /**
     * Registers `d3d11_texture` (an `ID3D11Texture2D*`) for reading, once. Safe to
     * call repeatedly for the same texture; the registration is cached.
     *
     * Registration is not per frame on purpose: it allocates driver-side state, and
     * a caller rotating through a small ring of staging textures would otherwise pay
     * for it fifty times a second.
     */
    bool register_texture(void* d3d11_texture);

    /// Drops the registration for `d3d11_texture`. Must happen before the texture
    /// itself is released.
    void unregister_texture(void* d3d11_texture);

    /**
     * Locks the registered texture, copies it into a freshly pooled mixer texture,
     * and unlocks -- which is what frees the caller's staging texture for reuse.
     *
     * Returns null if the texture was never registered or the lock fails.
     */
    std::shared_ptr<core::texture> copy_to_pooled(void* d3d11_texture, int width, int height);

    /// Releases every registration, leaving the interop device open. For a caller
    /// rebuilding its staging ring on a size change.
    void release_registrations();

  private:
    dx_interop();
    struct impl;
    std::unique_ptr<impl> impl_;
};

}}} // namespace caspar::accelerator::ogl

#endif // _WIN32
