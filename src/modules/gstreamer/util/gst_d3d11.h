/*
 * Copyright (c) 2011 Sveriges Television AB <info@casparcg.com>
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

#include <memory>
#include <string>

typedef struct _GstSample GstSample;

namespace caspar { namespace gstreamer {

/// What the mixer needs to import a frame: a shared NT handle and the size behind it.
struct shared_texture
{
    void* handle = nullptr;
    int   width  = 0;
    int   height = 0;

    explicit operator bool() const { return handle != nullptr; }
};

/// Carries a decoded frame from GStreamer's D3D11 device to the mixer's, without it ever
/// touching host memory.
///
/// The route: a GStreamer element decodes or converts into a `GstD3D11Memory`, which owns an
/// `ID3D11Texture2D` on GStreamer's own device. That texture is not shareable, so it cannot be
/// handed to the mixer directly. What can be shared is a texture *we* create on GStreamer's
/// device with `MISC_SHARED_NTHANDLE`; a `CopySubresourceRegion` into it stays on the GPU, and
/// the mixer opens the result by handle and imports it exactly the way the CEF path already
/// does.
///
/// So this trades one host round trip — download, then upload — for one GPU-to-GPU copy on the
/// same adapter. It is not a true zero-copy handoff, and calling it one would be wrong: the
/// copy is real, it is just on the right side of the bus.
///
/// **Three things it will refuse rather than guess at:**
///
/// * A sample that is not D3D11 memory. The caller falls back to the CPU path.
/// * A GStreamer device on a different adapter from the mixer's. `OpenSharedResource1` fails
///   across adapters, and a cross-adapter copy would be slower than the host path it replaces.
/// * A mixer that cannot import a shared texture at all. The first failure is caught, logged
///   once, and the bridge disables itself for the rest of the producer's life.
class d3d11_bridge
{
  public:
    d3d11_bridge();
    ~d3d11_bridge();

    d3d11_bridge(const d3d11_bridge&)            = delete;
    d3d11_bridge& operator=(const d3d11_bridge&) = delete;

    /// True when the sample carries D3D11 memory this bridge could take.
    static bool handles(GstSample* sample);

    /// A shared handle holding this sample's picture, or an empty one — in which case the
    /// caller must use the CPU path. Never throws.
    ///
    /// A handle rather than an opened texture, because opening one is a thing only the OpenGL
    /// mixer can do: `d3d_texture2d` registers with WGL_NV_DX_interop2 on construction, and
    /// the device it needs does not exist on a Vulkan server. The handle is what both mixers
    /// accept.
    shared_texture import(GstSample* sample);

    /// Why the bridge gave up, for the log. Empty while it is still working.
    const std::wstring& disabled_reason() const;

  private:
    struct impl;
    std::unique_ptr<impl> impl_;
};

}} // namespace caspar::gstreamer

#endif // _WIN32
