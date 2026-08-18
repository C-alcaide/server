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

#include "../StdAfx.h"

#ifdef _WIN32

#include "gst_d3d11.h"

#include <common/log.h>
#include <common/utf.h>

#include <gst/d3d11/gstd3d11.h>
#include <gst/gst.h>

#include <atlcomcli.h>
#include <d3d11_1.h>
#include <dxgi1_2.h>

#include <array>
#include <vector>

namespace caspar { namespace gstreamer {

namespace {

/// How many textures the ring holds. The mixer takes its own copy of the texture it is given
/// — `import_d3d_texture` blits into a GL texture — but it does so on its own thread and on
/// its own schedule, so writing straight back into the texture it was handed last frame would
/// be a race. Three is one in flight, one being read, one spare.
constexpr std::size_t ring_size = 3;

/// Only this one. `d3d11convert` is asked for BGRA upstream, and the mixer's import path maps
/// exactly two DXGI formats — accepting anything else here would hand it a texture it would
/// then have to reject, further from the cause.
constexpr DXGI_FORMAT accepted_format = DXGI_FORMAT_B8G8R8A8_UNORM;

/// HRESULTs reach the log as hex, because that is the form every reference indexes them by
/// and the first failure here — E_INVALIDARG from an invalid MiscFlags combination — cost an
/// hour precisely because the message did not carry it.
std::wstring hresult_text(HRESULT hr)
{
    wchar_t buffer[16] = {};
    swprintf(buffer, sizeof(buffer) / sizeof(buffer[0]), L"0x%08lX", static_cast<unsigned long>(hr));
    return buffer;
}

GstD3D11Memory* d3d11_memory_of(GstSample* sample)
{
    auto* buffer = sample != nullptr ? gst_sample_get_buffer(sample) : nullptr;
    if (buffer == nullptr || gst_buffer_n_memory(buffer) == 0)
        return nullptr;

    auto* mem = gst_buffer_peek_memory(buffer, 0);
    if (mem == nullptr || !gst_is_d3d11_memory(mem))
        return nullptr;

    return GST_D3D11_MEMORY_CAST(mem);
}

} // namespace

struct d3d11_bridge::impl
{
    std::wstring disabled_reason;

    // The ring lives on GStreamer's device; each entry carries the shared handle the mixer
    // imports, so the per-frame cost is the copy and nothing else.
    struct slot
    {
        CComPtr<ID3D11Texture2D> source_side;
        HANDLE                   handle = nullptr;

        ~slot()
        {
            if (handle != nullptr)
                CloseHandle(handle);
        }
    };

    std::vector<std::unique_ptr<slot>> ring;
    std::size_t       next = 0;

    UINT                     width  = 0;
    UINT                     height = 0;
    CComPtr<ID3D11Query>     fence;
    CComPtr<ID3D11Device>    gst_device;

    void disable(const std::wstring& reason)
    {
        if (disabled_reason.empty()) {
            disabled_reason = reason;
            CASPAR_LOG(info) << L"[gstreamer] GPU frame path disabled: " << reason
                             << L" Falling back to host memory.";
        }
        ring.clear();
    }

    bool ensure_ring(GstD3D11Memory* mem, const D3D11_TEXTURE2D_DESC& src_desc)
    {
        if (!ring.empty() && width == src_desc.Width && height == src_desc.Height)
            return true;

        ring.clear();
        width  = src_desc.Width;
        height = src_desc.Height;

        gst_device = gst_d3d11_device_get_device_handle(mem->device);
        if (!gst_device) {
            disable(L"GStreamer's D3D11 device handle was null.");
            return false;
        }

        D3D11_TEXTURE2D_DESC desc = {};
        desc.Width                = width;
        desc.Height               = height;
        desc.MipLevels            = 1;
        desc.ArraySize            = 1;
        desc.Format               = accepted_format;
        desc.SampleDesc.Count     = 1;
        desc.Usage                = D3D11_USAGE_DEFAULT;
        desc.BindFlags            = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
        // NTHANDLE because OpenSharedResource1 — what the mixer's d3d_device uses — takes
        // nothing else. Paired with the legacy SHARED bit because NTHANDLE **is not valid on
        // its own**: D3D requires it alongside SHARED or SHARED_KEYEDMUTEX, and asking for it
        // alone fails CreateTexture2D with E_INVALIDARG. Not KEYEDMUTEX, because the mixer
        // side never acquires one and a mutex nobody acquires is a deadlock rather than a
        // synchronisation; the fence below is what orders the copy against the read.
        desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED | D3D11_RESOURCE_MISC_SHARED_NTHANDLE;

        for (std::size_t n = 0; n < ring_size; ++n) {
            auto s  = std::make_unique<slot>();
            auto hr = gst_device->CreateTexture2D(&desc, nullptr, &s->source_side);
            if (FAILED(hr)) {
                disable(L"could not create a shared texture on GStreamer's device (hr " +
                        hresult_text(hr) + L").");
                return false;
            }

            CComQIPtr<IDXGIResource1> resource(s->source_side.p);
            if (!resource) {
                disable(L"the texture does not implement IDXGIResource1, so it cannot be shared.");
                return false;
            }

            hr = resource->CreateSharedHandle(
                nullptr, DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE, nullptr, &s->handle);
            if (FAILED(hr)) {
                disable(L"could not create a shared handle for the texture (hr " + hresult_text(hr) + L").");
                return false;
            }

            ring.push_back(std::move(s));
        }

        D3D11_QUERY_DESC query_desc = {};
        query_desc.Query            = D3D11_QUERY_EVENT;
        if (FAILED(gst_device->CreateQuery(&query_desc, &fence))) {
            disable(L"could not create the fence used to order the copy against the read.");
            return false;
        }

        CASPAR_LOG(info) << L"[gstreamer] GPU frame path active: " << width << L"x" << height
                         << L" BGRA, " << ring_size << L" shared textures.";
        return true;
    }

    shared_texture import(GstSample* sample)
    {
        if (!disabled_reason.empty())
            return {};

        auto* mem = d3d11_memory_of(sample);
        if (mem == nullptr)
            return {};

        D3D11_TEXTURE2D_DESC src_desc = {};
        if (!gst_d3d11_memory_get_texture_desc(mem, &src_desc))
            return {};

        if (src_desc.Format != accepted_format) {
            disable(L"the pipeline delivered " + std::to_wstring(static_cast<int>(src_desc.Format)) +
                    L" rather than BGRA on the GPU; ask the pipeline for "
                    L"video/x-raw(memory:D3D11Memory),format=BGRA.");
            return {};
        }

        if (!ensure_ring(mem, src_desc))
            return {};

        auto* source = gst_d3d11_memory_get_resource_handle(mem);
        if (source == nullptr)
            return {};

        auto& s = *ring[next];
        next    = (next + 1) % ring.size();

        auto* context = gst_d3d11_device_get_device_context_handle(mem->device);
        if (context == nullptr) {
            disable(L"GStreamer's D3D11 context handle was null.");
            return {};
        }

        // GStreamer's device context is shared with its own streaming threads and is not
        // free-threaded, so every use of it goes inside the device lock. Skipping this is the
        // kind of defect that shows up as a corrupt frame once an hour.
        gst_d3d11_device_lock(mem->device);
        context->CopySubresourceRegion(
            s.source_side, 0, 0, 0, 0, source, gst_d3d11_memory_get_subresource_index(mem), nullptr);
        context->End(fence);
        gst_d3d11_device_unlock(mem->device);

        // Wait for the copy to retire before the mixer is allowed to read the texture. A Flush
        // alone only submits the work; without this the mixer can sample a half-written
        // surface, which reads as tearing that comes and goes with load.
        BOOL done = FALSE;
        for (int spins = 0; spins < 100000; ++spins) {
            gst_d3d11_device_lock(mem->device);
            const auto hr = context->GetData(fence, &done, sizeof(done), 0);
            gst_d3d11_device_unlock(mem->device);

            if (hr == S_OK && done)
                break;
            if (FAILED(hr)) {
                disable(L"the fence query failed while waiting for the GPU copy.");
                return {};
            }
        }

        if (!done) {
            disable(L"the GPU copy did not complete in time.");
            return {};
        }

        return shared_texture{s.handle, static_cast<int>(width), static_cast<int>(height)};
    }
};

d3d11_bridge::d3d11_bridge()
    : impl_(std::make_unique<impl>())
{
}

d3d11_bridge::~d3d11_bridge() = default;

bool d3d11_bridge::handles(GstSample* sample) { return d3d11_memory_of(sample) != nullptr; }

shared_texture d3d11_bridge::import(GstSample* sample)
{
    try {
        return impl_->import(sample);
    } catch (...) {
        impl_->disable(L"an exception escaped the GPU import path.");
        CASPAR_LOG_CURRENT_EXCEPTION();
        return {};
    }
}

const std::wstring& d3d11_bridge::disabled_reason() const { return impl_->disabled_reason; }

}} // namespace caspar::gstreamer

#endif // _WIN32
