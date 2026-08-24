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

#include "../StdAfx.h"

#ifdef CASPAR_GST_GPU_BRIDGE

#include "gst_gpu_bridge.h"

#include <common/except.h>
#include <common/log.h>
#include <common/utf.h>

#include <core/frame/frame.h>
#include <core/frame/pixel_format.h>

#include <accelerator/ogl/util/dx_interop.h>
#ifdef ENABLE_VULKAN
#include <accelerator/vulkan/util/d3d11_import_bridge.h>
#endif

#include <gst/d3d11/gstd3d11.h>
#include <gst/video/video.h>

#include <d3d10.h>       // ID3D10Multithread, which lives here and applies to D3D11
#include <d3d11_4.h>
#include <d3dcompiler.h>
#include <dxgi1_2.h>

#include <algorithm>
#include <string>

namespace caspar { namespace gstreamer {

namespace {

/// After this many consecutive failures the bridge gives up for good. A GPU route that cannot
/// work will not start working, and retrying costs a D3D11 pass per frame to learn nothing.
constexpr int kMaxConsecutiveFailures = 3;

/// Full-screen triangle from SV_VertexID. No vertex or index buffers.
constexpr const char* kVS = "struct VSOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };\n"
                            "VSOut main(uint vid : SV_VertexID)\n"
                            "{\n"
                            "    VSOut o;\n"
                            "    float2 t = float2((vid << 1) & 2, vid & 2);\n"
                            "    o.uv  = t;\n"
                            "    o.pos = float4(t * float2(2, -2) + float2(-1, 1), 0, 1);\n"
                            "    return o;\n"
                            "}\n";

/// Pass-through: whatever the plane SRV yields is written unchanged, and the SRV's format
/// decides which components mean anything (R8 for luma, R8G8 for interleaved chroma). No
/// arithmetic happens here by design -- the colour conversion belongs to the mixer's shader,
/// which is the entire point of carrying the planes separately.
///
/// `Load()`, not `Sample()`. A decoded surface is padded up to the codec's macroblock grid --
/// H.264 stores 1080 as 1088 -- so normalised coordinates span the padding too and silently
/// rescale the picture, about four rows of vertical shift at 1080p. Indexing texels directly
/// copies the cropped top-left region exactly and needs no sampler at all.
constexpr const char* kPS = "Texture2D src : register(t0);\n"
                            "struct VSOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };\n"
                            "float4 main(VSOut i) : SV_Target { return src.Load(int3(i.pos.xy, 0)); }\n";

struct plane_formats
{
    DXGI_FORMAT       y  = DXGI_FORMAT_UNKNOWN;
    DXGI_FORMAT       uv = DXGI_FORMAT_UNKNOWN;
    common::bit_depth depth = common::bit_depth::bit8;
};

/// Only the semi-planar layouts a hardware decoder produces. Anything else must take the host
/// path rather than be reinterpreted.
bool formats_for(DXGI_FORMAT surface, plane_formats& out)
{
    switch (surface) {
        case DXGI_FORMAT_NV12:
            out = {DXGI_FORMAT_R8_UNORM, DXGI_FORMAT_R8G8_UNORM, common::bit_depth::bit8};
            return true;
        case DXGI_FORMAT_P010:
        case DXGI_FORMAT_P016:
            out = {DXGI_FORMAT_R16_UNORM, DXGI_FORMAT_R16G16_UNORM, common::bit_depth::bit16};
            return true;
        default:
            return false;
    }
}

core::chroma_location chroma_location_of(const GstVideoInfo& info)
{
    const auto site = GST_VIDEO_INFO_CHROMA_SITE(&info);
    if (site == GST_VIDEO_CHROMA_SITE_UNKNOWN)
        return core::chroma_location::unspecified;
    if ((site & GST_VIDEO_CHROMA_SITE_H_COSITED) && (site & GST_VIDEO_CHROMA_SITE_V_COSITED))
        return core::chroma_location::topleft;
    if (site & GST_VIDEO_CHROMA_SITE_H_COSITED)
        return core::chroma_location::left;
    return core::chroma_location::center;
}

} // namespace

struct gst_gpu_bridge::impl
{
    // GStreamer's, not ours: borrowed from the first sample and released with the bridge.
    GstD3D11Device*      gst_device_ = nullptr;
    ID3D11Device*        device_     = nullptr;
    ID3D11DeviceContext* ctx_        = nullptr;

    /// A shader-readable copy of the decoded surface, allocated only when the decoder's own is
    /// not shader-readable -- which is the common case, not an exotic one.
    ///
    /// `d3d11h264dec` allocates its output pool for DECODING: `D3D11_BIND_DECODER`, no
    /// `BIND_SHADER_RESOURCE`. A texture without that bind flag cannot have a shader resource
    /// view at all, so `gst_d3d11_memory_get_shader_resource_view_size()` answers 0 and
    /// `CreateShaderResourceView1` fails, and the plane extraction has nothing to sample.
    ///
    /// It hid at first because the earliest pipeline fed `decodebin` a lossless clip, which
    /// decodes in SOFTWARE, so `d3d11convert` had real work to do and produced a fresh
    /// shader-readable texture as a side effect. Point the same route at a hardware decoder and
    /// the converter becomes passthrough, the decoder's own surface arrives, and it cannot be
    /// sampled. "It worked until we made it faster" is the shape to recognise here.
    ///
    /// One `CopySubresourceRegion` into a texture we own fixes it. That is a full-frame
    /// GPU-to-GPU copy, so the route costs three passes rather than two on such decoders --
    /// still well under the BGRA route's three at 4 bytes/pixel, and still no host round trip.
    ID3D11Texture2D* staging_    = nullptr;
    bool             need_stage_ = false;

    // Our plane textures. One fixed pair rather than a ring -- see the header.
    ID3D11Texture2D*        y_tex_  = nullptr;
    ID3D11Texture2D*        uv_tex_ = nullptr;
    ID3D11RenderTargetView* y_rtv_  = nullptr;
    ID3D11RenderTargetView* uv_rtv_ = nullptr;
    HANDLE                  y_share_  = nullptr;
    HANDLE                  uv_share_ = nullptr;

    ID3D11VertexShader* vs_ = nullptr;
    ID3D11PixelShader*  ps_ = nullptr;

    std::unique_ptr<accelerator::ogl::dx_interop> gl_interop_;
#ifdef ENABLE_VULKAN
    std::unique_ptr<accelerator::vulkan::d3d11_import_bridge> vk_import_;
#endif

    core::gpu_backend backend_ = core::gpu_backend::none;

    int           width_  = 0;
    int           height_ = 0;
    DXGI_FORMAT   surface_format_ = DXGI_FORMAT_UNKNOWN;
    plane_formats planes_;

    bool        ready_    = false;
    bool        disabled_ = false;
    int         failures_ = 0;

    ~impl() { teardown(); }

    /// Order matters and is not obvious: the Vulkan imports alias the D3D11 textures, so they
    /// must go first, and the GL registrations hold driver-side state keyed on the textures,
    /// so those must go before the textures are released.
    void teardown()
    {
#ifdef ENABLE_VULKAN
        if (vk_import_) {
            vk_import_->release_imports();
            vk_import_.reset();
        }
#endif
        if (gl_interop_) {
            gl_interop_->release_registrations();
            gl_interop_.reset();
        }
        release_planes();
        if (ps_) { ps_->Release(); ps_ = nullptr; }
        if (vs_) { vs_->Release(); vs_ = nullptr; }
        // ctx_ and device_ are GStreamer's; only the ref on the device is ours.
        ctx_    = nullptr;
        device_ = nullptr;
        if (gst_device_) { gst_object_unref(gst_device_); gst_device_ = nullptr; }
        ready_ = false;
    }

    void release_planes()
    {
        if (staging_) { staging_->Release(); staging_ = nullptr; }
        if (y_rtv_)  { y_rtv_->Release();  y_rtv_  = nullptr; }
        if (uv_rtv_) { uv_rtv_->Release(); uv_rtv_ = nullptr; }
        if (y_tex_)  { y_tex_->Release();  y_tex_  = nullptr; }
        if (uv_tex_) { uv_tex_->Release(); uv_tex_ = nullptr; }
        if (y_share_)  { CloseHandle(y_share_);  y_share_  = nullptr; }
        if (uv_share_) { CloseHandle(uv_share_); uv_share_ = nullptr; }
    }

    void give_up(const std::wstring& reason)
    {
        if (disabled_)
            return;
        disabled_ = true;
        CASPAR_LOG(info) << L"[gstreamer] GPU path off: " << reason
                         << L". Using host memory; the picture is correct and the transfer is not.";
        teardown();
    }

    void note_failure(const std::wstring& reason)
    {
        if (disabled_)
            return;
        if (++failures_ >= kMaxConsecutiveFailures)
            give_up(reason + L" (" + std::to_wstring(failures_) + L" consecutive)");
        else
            CASPAR_LOG(warning) << L"[gstreamer] GPU frame failed: " << reason;
    }

    bool compile_shaders()
    {
        auto compile = [](const char* src, const char* target, ID3DBlob** out) -> bool {
            ID3DBlob* err = nullptr;
            auto      hr  = D3DCompile(src, std::strlen(src), nullptr, nullptr, nullptr, "main", target,
                                       D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, out, &err);
            if (err)
                err->Release();
            return SUCCEEDED(hr);
        };

        ID3DBlob* vsb = nullptr;
        ID3DBlob* psb = nullptr;
        bool      ok  = compile(kVS, "vs_5_0", &vsb) && compile(kPS, "ps_5_0", &psb);
        if (ok)
            ok = SUCCEEDED(device_->CreateVertexShader(vsb->GetBufferPointer(), vsb->GetBufferSize(), nullptr, &vs_)) &&
                 SUCCEEDED(device_->CreatePixelShader(psb->GetBufferPointer(), psb->GetBufferSize(), nullptr, &ps_));
        if (vsb) vsb->Release();
        if (psb) psb->Release();
        return ok;
    }

    /// Same format and size as the decoder's surface, but shader-readable and ours.
    bool make_staging()
    {
        D3D11_TEXTURE2D_DESC td = {};
        td.Width                = width_;
        td.Height               = height_;
        td.MipLevels            = 1;
        td.ArraySize            = 1;
        td.Format               = surface_format_;
        td.SampleDesc.Count     = 1;
        td.Usage                = D3D11_USAGE_DEFAULT;
        td.BindFlags            = D3D11_BIND_SHADER_RESOURCE;
        return SUCCEEDED(device_->CreateTexture2D(&td, nullptr, &staging_));
    }

    bool make_planes()
    {
        release_planes();

        const bool shared = backend_ == core::gpu_backend::vulkan;

        auto make = [&](int w, int h, DXGI_FORMAT fmt, ID3D11Texture2D** tex, ID3D11RenderTargetView** rtv,
                        HANDLE* share) -> bool {
            D3D11_TEXTURE2D_DESC td = {};
            td.Width                = w;
            td.Height               = h;
            td.MipLevels            = 1;
            td.ArraySize            = 1;
            td.Format               = fmt;
            td.SampleDesc.Count     = 1;
            td.Usage                = D3D11_USAGE_DEFAULT;
            td.BindFlags            = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
            // The only difference between the two backends' plane textures.
            // WGL_NV_DX_interop2 shares a texture without a handle; Vulkan needs a real one to
            // import through VK_KHR_external_memory_win32. SHARED_NTHANDLE alone is
            // E_INVALIDARG -- it has to be paired with SHARED.
            if (shared)
                td.MiscFlags = D3D11_RESOURCE_MISC_SHARED | D3D11_RESOURCE_MISC_SHARED_NTHANDLE;
            if (FAILED(device_->CreateTexture2D(&td, nullptr, tex)))
                return false;
            if (FAILED(device_->CreateRenderTargetView(*tex, nullptr, rtv)))
                return false;
            if (!shared)
                return true;

            IDXGIResource1* res = nullptr;
            if (FAILED((*tex)->QueryInterface(__uuidof(IDXGIResource1), reinterpret_cast<void**>(&res))) || !res)
                return false;
            auto hr = res->CreateSharedHandle(nullptr, DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE, nullptr,
                                              share);
            res->Release();
            return SUCCEEDED(hr) && *share != nullptr;
        };

        return make(width_, height_, planes_.y, &y_tex_, &y_rtv_, &y_share_) &&
               make(width_ / 2, height_ / 2, planes_.uv, &uv_tex_, &uv_rtv_, &uv_share_);
    }
};

gst_gpu_bridge::gst_gpu_bridge()
    : impl_(new impl())
{
}

gst_gpu_bridge::~gst_gpu_bridge() = default;

bool gst_gpu_bridge::handles(GstSample* sample)
{
    auto* buffer = sample ? gst_sample_get_buffer(sample) : nullptr;
    if (buffer == nullptr || gst_buffer_n_memory(buffer) == 0)
        return false;
    auto* mem = gst_buffer_peek_memory(buffer, 0);
    return mem != nullptr && gst_is_d3d11_memory(mem);
}

core::draw_frame gst_gpu_bridge::to_frame(void*                       tag,
                                          core::frame_factory&        frame_factory,
                                          GstSample*                  sample,
                                          const std::vector<int32_t>& audio_samples)
{
    auto& m = *impl_;
    if (m.disabled_)
        return core::draw_frame{};

    auto* caps   = gst_sample_get_caps(sample);
    auto* buffer = gst_sample_get_buffer(sample);
    if (caps == nullptr || buffer == nullptr)
        return core::draw_frame{};

    GstVideoInfo info;
    if (!gst_video_info_from_caps(&info, caps)) {
        m.note_failure(L"the sample's caps are not raw video");
        return core::draw_frame{};
    }

    auto* mem = gst_buffer_peek_memory(buffer, 0);
    if (mem == nullptr || !gst_is_d3d11_memory(mem)) {
        m.note_failure(L"the sample is not D3D11 memory");
        return core::draw_frame{};
    }
    auto* d3d_mem = reinterpret_cast<GstD3D11Memory*>(mem);

    // ── One-time setup, on the first sample ────────────────────────────────────────────
    if (!m.ready_) {
        m.backend_ = frame_factory.gpu_device_backend();
        if (m.backend_ == core::gpu_backend::none) {
            m.give_up(L"the mixer exposes no GPU device");
            return core::draw_frame{};
        }

        m.gst_device_ = reinterpret_cast<GstD3D11Device*>(gst_object_ref(d3d_mem->device));
        m.device_     = gst_d3d11_device_get_device_handle(m.gst_device_);
        m.ctx_        = gst_d3d11_device_get_device_context_handle(m.gst_device_);
        if (m.device_ == nullptr || m.ctx_ == nullptr) {
            m.give_up(L"GStreamer's D3D11 device exposes no handle");
            return core::draw_frame{};
        }

        // ── Multithread protection on GStreamer's context ─────────────────────────────────
        // `gst_d3d11_device_lock` is GStreamer's own mutex and it is only mutual exclusion
        // among code that takes it. The D3D11 video decoder does not: it issues
        // `ID3D11VideoContext` calls on its own thread, and with the extraction draws running
        // on the producer's thread the debug layer says so exactly --
        //
        //   SwapDeviceContextState: Two threads were found to be executing functions
        //   associated with the same Device[Context] at the same time. This will cause
        //   corruption of memory. Appropriate thread synchronization needs to occur external
        //   to the Direct3D API (or through the ID3D10Multithread interface).
        //
        // -- and then the process takes an unhandled exception. Holding the GStreamer lock
        // across the whole extraction was necessary and did not fix it, because the other
        // party never takes that lock.
        //
        // `ID3D10Multithread` is the remedy the message names, and it is the same one FFmpeg's
        // D3D11VA path enables for the same reason. It makes the driver serialise context use
        // internally. Idempotent, and cheap relative to a decode.
        //
        // **Under GStreamer's lock, and that is not belt-and-braces.** `SetMultithreadProtected`
        // is itself a device-context call, so making it while the decoder is mid-frame on the
        // same context is the very race it exists to prevent -- the debug layer named this call
        // by name once it was added without the lock. It has to be the first thing serialised,
        // not the thing that starts serialising.
        {
            gst_d3d11_device_lock(m.gst_device_);
            ID3D10Multithread* mt = nullptr;
            const bool         got =
                SUCCEEDED(m.ctx_->QueryInterface(__uuidof(ID3D10Multithread),
                                                 reinterpret_cast<void**>(&mt))) && mt != nullptr;
            BOOL was = TRUE;
            if (got) {
                was = mt->SetMultithreadProtected(TRUE);
                mt->Release();
            }
            gst_d3d11_device_unlock(m.gst_device_);

            if (got) {
                if (!was)
                    CASPAR_LOG(info) << L"[gstreamer] enabled D3D11 multithread protection on "
                                        L"GStreamer's device context (it was off).";
            } else {
                m.give_up(L"GStreamer's D3D11 context does not expose ID3D10Multithread, so the "
                          L"extraction cannot be made safe against the decoder's own thread");
                return core::draw_frame{};
            }
        }

        auto* resource = gst_d3d11_memory_get_resource_handle(d3d_mem);
        ID3D11Texture2D* surface = nullptr;
        if (resource == nullptr ||
            FAILED(resource->QueryInterface(__uuidof(ID3D11Texture2D), reinterpret_cast<void**>(&surface))) ||
            !surface) {
            m.give_up(L"the sample's resource is not a 2D texture");
            return core::draw_frame{};
        }
        D3D11_TEXTURE2D_DESC sd = {};
        surface->GetDesc(&sd);
        surface->Release();

        // Can the decoder's own surface be sampled at all? If not, everything below has to
        // read a copy instead. Deciding once, from the surface description, rather than
        // discovering it as a view-creation failure per frame.
        m.need_stage_ = (sd.BindFlags & D3D11_BIND_SHADER_RESOURCE) == 0;

        if (!formats_for(sd.Format, m.planes_)) {
            m.give_up(L"the decoded surface format " + std::to_wstring(static_cast<int>(sd.Format)) +
                      L" is not one of NV12, P010 or P016");
            return core::draw_frame{};
        }

        m.width_          = GST_VIDEO_INFO_WIDTH(&info);
        m.height_         = GST_VIDEO_INFO_HEIGHT(&info);
        m.surface_format_ = sd.Format;

        if (!m.compile_shaders()) {
            m.give_up(L"the plane extraction shaders would not compile");
            return core::draw_frame{};
        }
        if (!m.make_planes()) {
            m.give_up(L"the plane textures could not be created");
            return core::draw_frame{};
        }
        if (m.need_stage_ && !m.make_staging()) {
            m.give_up(L"the decoder's surface is not shader-readable and a staging copy could "
                      L"not be allocated");
            return core::draw_frame{};
        }

        if (m.backend_ == core::gpu_backend::opengl) {
            // A failed open IS the adapter check. WGL_NV_DX_interop2 only works when the
            // D3D11 device and the GL context are on the same physical adapter, and there is no
            // LUID query for a GL context to compare against -- so this is the cheapest and the
            // only reliable way to ask, and it names the decoder when it says no.
            std::wstring reason;
            m.gl_interop_ =
                accelerator::ogl::dx_interop::create(frame_factory.gpu_device_handle(), m.device_, reason);
            if (!m.gl_interop_) {
                m.give_up(L"GL interop would not open against GStreamer's D3D11 device (" + reason +
                          L"). If the two are on different adapters, name the decoder explicitly -- "
                          L"d3d11h264device1dec selects the second one");
                return core::draw_frame{};
            }
            if (!m.gl_interop_->register_texture(m.y_tex_) || !m.gl_interop_->register_texture(m.uv_tex_)) {
                m.give_up(L"the plane textures could not be registered with GL");
                return core::draw_frame{};
            }
        } else {
#ifdef ENABLE_VULKAN
            try {
                m.vk_import_ = std::make_unique<accelerator::vulkan::d3d11_import_bridge>(
                    frame_factory.gpu_device_handle());
            } catch (...) {
                m.give_up(L"the Vulkan D3D11 import bridge would not open");
                return core::draw_frame{};
            }
#else
            m.give_up(L"this build has no Vulkan support");
            return core::draw_frame{};
#endif
        }

        m.ready_ = true;
        CASPAR_LOG(info) << L"[gstreamer] GPU path active: " << m.width_ << L"x" << m.height_
                         << L" semi-planar handed to the mixer, which performs the colour conversion.";
    }

    // A mid-stream renegotiation would need new plane textures, new imports and new
    // registrations. Refusing is honest and rare; silently rendering the wrong size is not.
    if (GST_VIDEO_INFO_WIDTH(&info) != m.width_ || GST_VIDEO_INFO_HEIGHT(&info) != m.height_) {
        m.give_up(L"the stream changed size mid-flight");
        return core::draw_frame{};
    }

    // ── Per-frame extraction ───────────────────────────────────────────────────────────
    auto*            resource = gst_d3d11_memory_get_resource_handle(d3d_mem);
    ID3D11Texture2D* surface  = nullptr;
    if (resource == nullptr ||
        FAILED(resource->QueryInterface(__uuidof(ID3D11Texture2D), reinterpret_cast<void**>(&surface))) || !surface) {
        m.note_failure(L"the sample's resource is not a 2D texture");
        return core::draw_frame{};
    }
    const UINT subresource = gst_d3d11_memory_get_subresource_index(d3d_mem);

#ifdef ENABLE_VULKAN
    // Vulkan may still be reading last frame's planes, and the draws below overwrite them;
    // nothing else orders the two queues. Deliberately BEFORE the lock below: this can block
    // for milliseconds, and the decoder needs the context during exactly that window.
    if (m.vk_import_)
        m.vk_import_->wait_for_previous_copy();
#endif

    ID3D11Device3* dev3 = nullptr;
    if (FAILED(m.device_->QueryInterface(__uuidof(ID3D11Device3), reinterpret_cast<void**>(&dev3))) || !dev3) {
        surface->Release();
        m.give_up(L"ID3D11Device3 is unavailable, so there are no plane views");
        return core::draw_frame{};
    }

    // The decoder's surface is not shader-readable, so sample a copy of it instead. The copy is
    // array slice 0 of a texture we own, which is also why `sample_slice` below is 0 rather
    // than the decoder's subresource index.
    ID3D11Texture2D* sample_from  = surface;
    UINT             sample_slice = subresource;
    // ── ONE critical section for the whole extraction ─────────────────────────────────────
    // This is GStreamer's immediate context, shared with its own streaming threads, and it is
    // not free-threaded. Taking the lock separately around the staging copy and around the
    // draws leaves a window between them in which the decoder can use -- and re-state -- the
    // same context.
    //
    // That window is not theoretical. It presented as an unhandled exception on the 10-bit
    // clip with the D3D11 debug layer naming it outright: "SwapDeviceContextState: Two threads
    // were found to be executing functions associated with the same Device[Context] at the
    // same time." The 8-bit path had the identical structure and had simply won the race every
    // time, which is the worst way for a threading defect to behave.
    //
    // So the lock is held across the staging copy, the view creation and both draws. The
    // sequence is one indivisible use of somebody else's context.
    gst_d3d11_device_lock(m.gst_device_);

    if (m.need_stage_) {
        m.ctx_->CopySubresourceRegion(m.staging_, 0, 0, 0, 0, surface, subresource, nullptr);
        sample_from  = m.staging_;
        sample_slice = 0;
    }

    // GStreamer publishes per-plane SRVs for a memory allocated with BIND_SHADER_RESOURCE, and
    // using them saves creating our own. It is not guaranteed to -- the size query answers 0
    // for a memory allocated without that bind flag -- so CreateShaderResourceView1 stays as
    // the fallback rather than as the only route.
    ID3D11ShaderResourceView* y_srv     = nullptr;
    ID3D11ShaderResourceView* uv_srv    = nullptr;
    bool                      own_views = false;

    if (!m.need_stage_ && gst_d3d11_memory_get_shader_resource_view_size(d3d_mem) >= 2) {
        y_srv  = gst_d3d11_memory_get_shader_resource_view(d3d_mem, 0);
        uv_srv = gst_d3d11_memory_get_shader_resource_view(d3d_mem, 1);
    }

    if (y_srv == nullptr || uv_srv == nullptr) {
        own_views = true;
        auto make_srv = [&](DXGI_FORMAT fmt, UINT plane, ID3D11ShaderResourceView** srv) -> bool {
            D3D11_SHADER_RESOURCE_VIEW_DESC1 sd = {};
            sd.Format                           = fmt;
            sd.ViewDimension                    = D3D11_SRV_DIMENSION_TEXTURE2DARRAY;
            sd.Texture2DArray.MipLevels         = 1;
            sd.Texture2DArray.FirstArraySlice   = sample_slice;
            sd.Texture2DArray.ArraySize         = 1;
            sd.Texture2DArray.PlaneSlice        = plane;
            ID3D11ShaderResourceView1* v        = nullptr;
            if (FAILED(dev3->CreateShaderResourceView1(sample_from, &sd, &v)) || !v)
                return false;
            *srv = v;
            return true;
        };
        if (!make_srv(m.planes_.y, 0, &y_srv) || !make_srv(m.planes_.uv, 1, &uv_srv)) {
            if (y_srv) y_srv->Release();
            if (uv_srv) uv_srv->Release();
            dev3->Release();
            surface->Release();
            gst_d3d11_device_unlock(m.gst_device_);   // still inside the critical section
            m.note_failure(L"the NV12 plane views could not be created (is the surface bound for shader access?)");
            return core::draw_frame{};
        }
    }


    auto draw = [&](ID3D11RenderTargetView* rtv, ID3D11ShaderResourceView* srv, int w, int h) {
        D3D11_VIEWPORT vp = {};
        vp.Width          = static_cast<float>(w);
        vp.Height         = static_cast<float>(h);
        vp.MaxDepth       = 1.0f;

        m.ctx_->OMSetRenderTargets(1, &rtv, nullptr);
        m.ctx_->RSSetViewports(1, &vp);
        m.ctx_->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        m.ctx_->IASetInputLayout(nullptr);
        m.ctx_->VSSetShader(m.vs_, nullptr, 0);
        m.ctx_->PSSetShader(m.ps_, nullptr, 0);
        m.ctx_->PSSetShaderResources(0, 1, &srv);
        m.ctx_->Draw(3, 0);

        ID3D11ShaderResourceView* none = nullptr;
        m.ctx_->PSSetShaderResources(0, 1, &none);
    };

    // GStreamer's immediate context is shared with its own streaming threads and is not
    // free-threaded. This is ITS device, not one we made, so the lock is GStreamer's.
    draw(m.y_rtv_, y_srv, m.width_, m.height_);
    draw(m.uv_rtv_, uv_srv, m.width_ / 2, m.height_ / 2);
    m.ctx_->Flush();

    gst_d3d11_device_unlock(m.gst_device_);

    if (own_views) {
        y_srv->Release();
        uv_srv->Release();
    }
    dev3->Release();
    surface->Release();

    // ── Hand the two planes to whichever mixer this is ─────────────────────────────────
    std::shared_ptr<core::texture> y_out;
    std::shared_ptr<core::texture> uv_out;

    if (m.backend_ == core::gpu_backend::opengl) {
        // Component counts, not byte counts: luma is one channel and interleaved chroma is
        // two, whatever the depth. `depth` carries the 8-vs-16-bit half separately.
        y_out  = m.gl_interop_->copy_to_pooled(m.y_tex_, m.width_, m.height_, 1, m.planes_.depth);
        uv_out = m.gl_interop_->copy_to_pooled(m.uv_tex_, m.width_ / 2, m.height_ / 2, 2, m.planes_.depth);
    } else {
#ifdef ENABLE_VULKAN
        if (!m.vk_import_->copy_planes(m.y_share_, m.uv_share_, m.width_, m.height_, m.width_ / 2, m.height_ / 2,
                                       m.planes_.depth, y_out, uv_out)) {
            y_out.reset();
            uv_out.reset();
        }
#endif
    }

    if (!y_out || !uv_out) {
        m.note_failure(L"the mixer would not take the semi-planar textures");
        return core::draw_frame{};
    }

    m.failures_ = 0;

    // ── Build the frame ────────────────────────────────────────────────────────────────
    auto desc = core::pixel_format_desc(core::pixel_format::nv12);
    desc.planes.push_back(core::pixel_format_desc::plane(m.width_, m.height_, 1, m.planes_.depth));
    desc.planes.push_back(
        core::pixel_format_desc::plane(m.width_ / 2, m.height_ / 2, 2, m.planes_.depth));

    switch (GST_VIDEO_INFO_COLORIMETRY(&info).matrix) {
        case GST_VIDEO_COLOR_MATRIX_BT601:  desc.color_space = core::color_space::bt601;  break;
        case GST_VIDEO_COLOR_MATRIX_BT2020: desc.color_space = core::color_space::bt2020; break;
        default:                            desc.color_space = core::color_space::bt709;  break;
    }
    desc.color_range = GST_VIDEO_INFO_COLORIMETRY(&info).range == GST_VIDEO_COLOR_RANGE_0_255
                           ? core::color_range::full
                           : core::color_range::limited;
    desc.chroma_location = chroma_location_of(info);

    // No host pixels. A consumer that needs them asks the channel, which reads back the
    // composited frame; host_image_state() reports this honestly as `unavailable` rather than
    // handing out an empty array that reads as black.
    std::vector<array<const std::uint8_t>> image_data;
    image_data.emplace_back(static_cast<std::size_t>(0));
    image_data.emplace_back(static_cast<std::size_t>(0));

    array<const std::int32_t> audio_data;
    if (!audio_samples.empty())
        audio_data = array<const std::int32_t>(audio_samples);

    std::vector<std::shared_ptr<core::texture>> textures;
    textures.push_back(std::move(y_out));
    textures.push_back(std::move(uv_out));

    return core::draw_frame(
        core::const_frame(tag, std::move(image_data), std::move(audio_data), desc, std::move(textures)));
}

}} // namespace caspar::gstreamer

#endif // CASPAR_GST_GPU_BRIDGE
