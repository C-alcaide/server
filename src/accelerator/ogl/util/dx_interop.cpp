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

#ifdef _WIN32

#include "dx_interop.h"

#include "device.h"
#include "texture.h"

#include <common/log.h>

#include <GL/glew.h>
#include <GL/wglew.h>

#include <unordered_map>
#include <vector>

namespace caspar { namespace accelerator { namespace ogl {

struct dx_interop::impl
{
    std::shared_ptr<device> dev_;

    PFNWGLDXOPENDEVICENVPROC       wglDXOpenDeviceNV_       = nullptr;
    PFNWGLDXCLOSEDEVICENVPROC      wglDXCloseDeviceNV_      = nullptr;
    PFNWGLDXREGISTEROBJECTNVPROC   wglDXRegisterObjectNV_   = nullptr;
    PFNWGLDXUNREGISTEROBJECTNVPROC wglDXUnregisterObjectNV_ = nullptr;
    PFNWGLDXLOCKOBJECTSNVPROC      wglDXLockObjectsNV_      = nullptr;
    PFNWGLDXUNLOCKOBJECTSNVPROC    wglDXUnlockObjectsNV_    = nullptr;

    HANDLE interop_device_ = nullptr;

    struct registration
    {
        HANDLE object = nullptr;
        GLuint gl_tex = 0;
    };
    std::unordered_map<void*, registration> registered_;

    ~impl()
    {
        if (!dev_)
            return;
        // Unregistering and deleting GL names both need the context current, so
        // everything happens on the device thread -- including closing the interop
        // device, which the extension ties to the context that opened it.
        dev_->dispatch_sync([&] {
            for (auto& [tex, reg] : registered_) {
                if (reg.object)
                    wglDXUnregisterObjectNV_(interop_device_, reg.object);
                if (reg.gl_tex)
                    glDeleteTextures(1, &reg.gl_tex);
            }
            registered_.clear();
            if (interop_device_) {
                wglDXCloseDeviceNV_(interop_device_);
                interop_device_ = nullptr;
            }
        });
    }

    bool load_entry_points()
    {
        wglDXOpenDeviceNV_       = (PFNWGLDXOPENDEVICENVPROC)wglGetProcAddress("wglDXOpenDeviceNV");
        wglDXCloseDeviceNV_      = (PFNWGLDXCLOSEDEVICENVPROC)wglGetProcAddress("wglDXCloseDeviceNV");
        wglDXRegisterObjectNV_   = (PFNWGLDXREGISTEROBJECTNVPROC)wglGetProcAddress("wglDXRegisterObjectNV");
        wglDXUnregisterObjectNV_ = (PFNWGLDXUNREGISTEROBJECTNVPROC)wglGetProcAddress("wglDXUnregisterObjectNV");
        wglDXLockObjectsNV_      = (PFNWGLDXLOCKOBJECTSNVPROC)wglGetProcAddress("wglDXLockObjectsNV");
        wglDXUnlockObjectsNV_    = (PFNWGLDXUNLOCKOBJECTSNVPROC)wglGetProcAddress("wglDXUnlockObjectsNV");

        return wglDXOpenDeviceNV_ && wglDXCloseDeviceNV_ && wglDXRegisterObjectNV_ && wglDXUnregisterObjectNV_ &&
               wglDXLockObjectsNV_ && wglDXUnlockObjectsNV_;
    }
};

dx_interop::dx_interop()
    : impl_(new impl())
{
}

dx_interop::~dx_interop() = default;

std::unique_ptr<dx_interop> dx_interop::create(void* ogl_device, void* d3d11_device, std::wstring& reason)
{
    if (!ogl_device || !d3d11_device) {
        reason = L"no GL device or no D3D11 device";
        return nullptr;
    }

    // The mixer owns the device through a shared_ptr already, so shared_from_this is
    // valid; the bridge keeps it alive for as long as it holds GL names.
    auto dev = static_cast<device*>(ogl_device)->shared_from_this();

    std::unique_ptr<dx_interop> self(new dx_interop());
    auto&                       m = *self->impl_;
    m.dev_                        = dev;

    // wglGetProcAddress and wglDXOpenDeviceNV both need a current context, which
    // only exists on the device's own thread.
    const bool ok = dev->dispatch_sync([&]() -> bool {
        if (!m.load_entry_points())
            return false;
        m.interop_device_ = m.wglDXOpenDeviceNV_(d3d11_device);
        return m.interop_device_ != nullptr;
    });

    if (!ok) {
        if (!m.wglDXOpenDeviceNV_) {
            reason = L"WGL_NV_DX_interop2 is not available on this context";
        } else {
            // Almost always an adapter mismatch: the extension requires the D3D11
            // device and the GL context to be on the same physical GPU.
            reason = L"wglDXOpenDeviceNV failed -- the D3D11 device is probably on a "
                     L"different adapter than the mixer's GL context";
        }
        m.dev_.reset(); // nothing to tear down on the device thread
        return nullptr;
    }

    return self;
}

bool dx_interop::register_texture(void* d3d11_texture)
{
    auto& m = *impl_;
    if (!d3d11_texture || !m.interop_device_)
        return false;
    if (m.registered_.count(d3d11_texture))
        return true;

    return m.dev_->dispatch_sync([&]() -> bool {
        impl::registration reg;
        glGenTextures(1, &reg.gl_tex);
        reg.object = m.wglDXRegisterObjectNV_(m.interop_device_, d3d11_texture, reg.gl_tex, GL_TEXTURE_2D,
                                              WGL_ACCESS_READ_ONLY_NV);
        if (!reg.object) {
            glDeleteTextures(1, &reg.gl_tex);
            CASPAR_LOG(warning) << L"[ogl::dx_interop] wglDXRegisterObjectNV failed";
            return false;
        }
        m.registered_.emplace(d3d11_texture, reg);
        return true;
    });
}

void dx_interop::unregister_texture(void* d3d11_texture)
{
    auto& m  = *impl_;
    auto  it = m.registered_.find(d3d11_texture);
    if (it == m.registered_.end())
        return;

    auto reg = it->second;
    m.registered_.erase(it);
    m.dev_->dispatch_sync([&] {
        if (reg.object)
            m.wglDXUnregisterObjectNV_(m.interop_device_, reg.object);
        if (reg.gl_tex)
            glDeleteTextures(1, &reg.gl_tex);
    });
}

void dx_interop::release_registrations()
{
    auto& m = *impl_;
    if (m.registered_.empty())
        return;

    auto all = std::move(m.registered_);
    m.registered_.clear();
    m.dev_->dispatch_sync([&] {
        for (auto& [tex, reg] : all) {
            if (reg.object)
                m.wglDXUnregisterObjectNV_(m.interop_device_, reg.object);
            if (reg.gl_tex)
                glDeleteTextures(1, &reg.gl_tex);
        }
    });
}

std::shared_ptr<core::texture> dx_interop::copy_to_pooled(void* d3d11_texture, int width, int height)
{
    auto& m  = *impl_;
    auto  it = m.registered_.find(d3d11_texture);
    if (it == m.registered_.end())
        return nullptr;

    const auto reg = it->second;

    std::shared_ptr<texture> out;
    m.dev_->dispatch_sync([&] {
        HANDLE obj = reg.object;
        // The lock is what orders the D3D11 write against the GL read; there is no
        // separate fence on this path.
        if (!m.wglDXLockObjectsNV_(m.interop_device_, 1, &obj)) {
            CASPAR_LOG(warning) << L"[ogl::dx_interop] wglDXLockObjectsNV failed";
            return;
        }

        // Copy into a pooled mixer texture rather than handing the registered one
        // over: the source belongs to the caller's staging ring and has to be free
        // again by the next frame.
        out = m.dev_->create_texture(width, height, 4, common::bit_depth::bit8, false);
        if (out) {
            glCopyImageSubData(reg.gl_tex, GL_TEXTURE_2D, 0, 0, 0, 0, out->id(), GL_TEXTURE_2D, 0, 0, 0, 0, width,
                               height, 1);
        }

        m.wglDXUnlockObjectsNV_(m.interop_device_, 1, &obj);
    });

    return std::static_pointer_cast<core::texture>(out);
}

}}} // namespace caspar::accelerator::ogl

#endif // _WIN32
