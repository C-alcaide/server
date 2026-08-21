#include "av_producer.h"

#include "av_input.h"
#include "filter_param_tween.h"

#include "../util/av_assert.h"
#include "../util/av_color.h"
#include "../util/av_util.h"
#include "../util/vulkan_hwdevice.h"

#include <boost/algorithm/string.hpp>
#include <boost/exception/exception.hpp>
#include <boost/format.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/range/algorithm/rotate.hpp>
#include <boost/rational.hpp>
#include <boost/thread.hpp>
#include <boost/thread/condition_variable.hpp>
#include <boost/thread/mutex.hpp>

#include <common/diagnostics/graph.h>
#include <common/env.h>
#include <common/except.h>
#include <common/executor.h>
#include <common/os/thread.h>
#include <common/scope_exit.h>
#include <common/timer.h>

#include <core/frame/draw_frame.h>
#include <core/frame/alpha_mode.h>
#include <core/frame/frame_factory.h>
#include <core/monitor/monitor.h>

#ifdef _WIN32
#include <d3d11.h>
#include <d3d11_3.h>
#include <d3d11_4.h>
#include <d3dcompiler.h>
#include <cstring>
#include <dxgi1_2.h>
#endif

#ifdef _WIN32
#include <GL/glew.h>
#include <GL/wglew.h>
#include <accelerator/ogl/util/device.h>
#include <accelerator/ogl/util/texture.h>
#ifdef ENABLE_VULKAN
// Header-only in the Vulkan sense: it exposes no Vulkan types, so this
// translation unit needs neither the Vulkan headers nor its dispatch loader.
#include <accelerator/vulkan/util/d3d11_import_bridge.h>
#include <accelerator/vulkan/util/av_vulkan_import.h>
#endif
#endif

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4244)
#endif
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavfilter/avfilter.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/channel_layout.h>
#include <libavutil/error.h>
#include <libavutil/imgutils.h>
#include <libavutil/mastering_display_metadata.h>
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>
#include <libavutil/pixfmt.h>
#include <libavutil/samplefmt.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_d3d11va.h>
#if defined(ENABLE_VULKAN) && LIBAVUTIL_VERSION_MAJOR >= 60
// FFmpeg 8's Vulkan compute decoders. The module gets the Vulkan SDK's include path for
// this one header alone (see src/modules/ffmpeg/CMakeLists.txt); it touches no Vulkan type
// of its own beyond what AVVkFrame exposes.
#include <libavutil/hwcontext_vulkan.h>
#endif
}

#include <algorithm>
#include <atomic>
#include <chrono>
#include <deque>
#include <iomanip>
#include <memory>
#include <queue>
#include <sstream>
#include <string>
#include <thread>

namespace caspar { namespace ffmpeg {

const AVRational TIME_BASE_Q = {1, AV_TIME_BASE};

// ── D3D11 → mixer GPU-direct bridge ─────────────────────────────────────────
// Avoids the GPU→CPU→GPU round trip for D3D11VA decoded frames by handing the
// mixer the decoded NV12 planes as GPU textures.
//
// The D3D11 half is shared by both mixers: a tiny pass-through pixel shader
// extracts the decoded surface's two planes into an R8 and an R8G8 texture (see
// setup_planes). Only the last step differs:
//
//   OpenGL — WGL_NV_DX_interop2 registers the two plane textures with GL and
//            they are copied into pooled GL textures.
//   Vulkan — the two plane textures carry D3D11 shared NT handles, which are
//            imported as VkImages and copied into pooled mixer textures by
//            accelerator::vulkan::d3d11_import_bridge.
//
// Vulkan imports the two single-plane textures rather than the decoded NV12
// surface as one multi-planar image, because the latter was measured not to
// work: D3D11 and the Vulkan driver disagree about where plane 1 lives inside a
// shared allocation. See docs/GPU_INTEROP_PLAN.md item 1 and the comment on
// d3d11_import_bridge.
#ifdef _WIN32
class d3d11_bridge
{
  public:
    enum class backend
    {
        opengl,
        vulkan
    };

  private:
    backend backend_ = backend::opengl;

    // D3D11 objects
    ID3D11Device*                       d3d11_device_      = nullptr;
    ID3D11DeviceContext*                d3d11_ctx_         = nullptr;
    ID3D11VideoDevice*                  video_device_      = nullptr;
    ID3D11VideoContext*                 video_ctx_         = nullptr;
    ID3D11VideoProcessor*               video_processor_   = nullptr;
    ID3D11VideoProcessorEnumerator*     vp_enum_           = nullptr;
    ID3D11VideoProcessorOutputView*     vp_output_view_    = nullptr;
    ID3D11Texture2D*                    bgra_texture_      = nullptr;

    // WGL_NV_DX_interop2
    PFNWGLDXOPENDEVICENVPROC            wglDXOpenDeviceNV_      = nullptr;
    PFNWGLDXCLOSEDEVICENVPROC           wglDXCloseDeviceNV_     = nullptr;
    PFNWGLDXREGISTEROBJECTNVPROC        wglDXRegisterObjectNV_  = nullptr;
    PFNWGLDXUNREGISTEROBJECTNVPROC      wglDXUnregisterObjectNV_ = nullptr;
    PFNWGLDXLOCKOBJECTSNVPROC           wglDXLockObjectsNV_     = nullptr;
    PFNWGLDXUNLOCKOBJECTSNVPROC         wglDXUnlockObjectsNV_   = nullptr;

    HANDLE interop_device_ = nullptr;
    HANDLE interop_object_ = nullptr;
    GLuint interop_gl_tex_ = 0;

    // ── Colour-exact plane path ──────────────────────────────────────────────
    // Instead of letting the VideoProcessor convert NV12->BGRA with a
    // driver-defined matrix and range, extract the two NV12 planes as-is and let
    // the mixer's shader convert them. A tiny pixel shader does the extraction:
    // it only moves texels between formats (R8 / R8G8), so it cannot alter
    // colour.
    ID3D11Texture2D*          y_texture_      = nullptr;   // R8_UNORM,   w x h
    ID3D11Texture2D*          uv_texture_     = nullptr;   // R8G8_UNORM, w/2 x h/2
    ID3D11RenderTargetView*   y_rtv_          = nullptr;
    ID3D11RenderTargetView*   uv_rtv_         = nullptr;
    ID3D11VertexShader*       plane_vs_       = nullptr;
    ID3D11PixelShader*        plane_ps_       = nullptr;
    ID3D11SamplerState*       plane_sampler_  = nullptr;
    // Per-surface-format view/target choice. NV12 planes are 8-bit (R8 / R8G8);
    // P010 and P016 are 16-bit (R16 / R16G16) with the significant bits
    // high-aligned, which the mixer handles by declaring the planes bit16.
    DXGI_FORMAT               expected_format_ = DXGI_FORMAT_NV12;
    DXGI_FORMAT               y_view_format_   = DXGI_FORMAT_R8_UNORM;
    DXGI_FORMAT               uv_view_format_  = DXGI_FORMAT_R8G8_UNORM;
    common::bit_depth         plane_depth_     = common::bit_depth::bit8;
    HANDLE                    y_interop_obj_  = nullptr;
    HANDLE                    uv_interop_obj_ = nullptr;
    GLuint                    y_gl_tex_       = 0;
    GLuint                    uv_gl_tex_      = 0;
    bool                      plane_path_ok_  = false;

    // ── Vulkan hand-off ──────────────────────────────────────────────────────
    // Shared NT handles onto the same two plane textures, and the importer that
    // turns them into mixer textures. Null on the OpenGL path.
    HANDLE y_shared_handle_  = nullptr;
    HANDLE uv_shared_handle_ = nullptr;
#ifdef ENABLE_VULKAN
    std::unique_ptr<accelerator::vulkan::d3d11_import_bridge> vk_import_;
#endif

    // The cross-API wait: D3D11 writes the planes on its own queue and Vulkan
    // reads them on another, so the CPU has to order them. Measured rather than
    // assumed, because a GPU-direct path that wins on decode and gives it back
    // on a sync stall looks identical to a win in a CPU figure -- and at four
    // layers this wait is milliseconds, not microseconds, so *how* it waits
    // matters as much as that it waits.
    //
    // An ID3D11Fence signalled on the immediate context and waited on through a
    // Win32 event blocks the thread. The ID3D11Query fallback below can only be
    // polled, which means spinning: at four layers that spin cost about 0.2
    // cores and gave back roughly a third of what GPU-direct saves. The fence
    // path is preferred wherever D3D11.4 is available, which is every OS this
    // runs on; the query remains for the case where it is not.
    ID3D11Fence*         sync_fence_    = nullptr;
    ID3D11DeviceContext4* d3d11_ctx4_   = nullptr;
    HANDLE               sync_event_    = nullptr;
    uint64_t             sync_value_    = 0;
    ID3D11Query*         sync_query_    = nullptr; // fallback: poll, no fence available
    int64_t              sync_wait_us_  = 0;       // D3D11 completion wait, accumulated
    int64_t              vk_wait_us_    = 0;       // waiting for the previous Vulkan copy
    int64_t              sync_frames_   = 0;
    bool                 sync_reported_ = false;

    // The mixer's GL device. All GL and interop work is dispatched onto its
    // thread, where its context is current: a private context is not an option
    // because wglShareLists fails against a context that is current on another
    // thread, which is exactly what the mixer's is.
    accelerator::ogl::device* ogl_dev_ = nullptr;

    // FFmpeg's D3D11 device lock. ID3D11DeviceContext is not free-threaded and
    // the decoder is using the same one, so every D3D11 call here must hold it.
    AVD3D11VADeviceContext* d3d11_hwctx_ = nullptr;

    int width_  = 0;
    int height_ = 0;
    bool active_ = false;

  public:
    bool init(AVBufferRef* hw_device_ctx, void* gpu_device_ptr, backend which)
    {
        if (!hw_device_ctx || !gpu_device_ptr) {
            CASPAR_LOG(warning) << L"[av_producer] bridge init: no hw_device_ctx or no mixer GPU device";
            return false;
        }

        backend_ = which;
        if (backend_ == backend::opengl)
            ogl_dev_ = static_cast<accelerator::ogl::device*>(gpu_device_ptr);

        // Get the D3D11 device from FFmpeg's hw_device_ctx
        auto* hwctx = reinterpret_cast<AVHWDeviceContext*>(hw_device_ctx->data);
        if (!hwctx || hwctx->type != AV_HWDEVICE_TYPE_D3D11VA) {
            CASPAR_LOG(warning) << L"[av_producer] bridge init: hw device is not D3D11VA";
            return false;
        }
        d3d11_hwctx_  = static_cast<AVD3D11VADeviceContext*>(hwctx->hwctx);
        d3d11_device_ = d3d11_hwctx_->device;
        d3d11_ctx_    = d3d11_hwctx_->device_context;
        if (!d3d11_device_ || !d3d11_ctx_) {
            CASPAR_LOG(warning) << L"[av_producer] bridge init: null D3D11 device/context";
            return false;
        }

        if (backend_ == backend::vulkan) {
#ifdef ENABLE_VULKAN
            try {
                vk_import_ = std::make_unique<accelerator::vulkan::d3d11_import_bridge>(gpu_device_ptr);
            } catch (...) {
                CASPAR_LOG(warning) << L"[av_producer] D3D11->Vulkan import bridge failed to initialise";
                return false;
            }
            CASPAR_LOG(info) << L"[av_producer] D3D11->Vulkan GPU-direct bridge initialized";
            active_ = true;
            return true;
#else
            CASPAR_LOG(warning) << L"[av_producer] built without Vulkan support";
            return false;
#endif
        }

        // Load the interop entry points and open the interop device on the
        // mixer's GL thread. wglGetProcAddress resolves nothing without a
        // current context, and the interop device must belong to the context
        // that will later lock the shared objects.
        bool ok = ogl_dev_->dispatch_sync([&]() -> bool {
            wglDXOpenDeviceNV_       = (PFNWGLDXOPENDEVICENVPROC)wglGetProcAddress("wglDXOpenDeviceNV");
            wglDXCloseDeviceNV_      = (PFNWGLDXCLOSEDEVICENVPROC)wglGetProcAddress("wglDXCloseDeviceNV");
            wglDXRegisterObjectNV_   = (PFNWGLDXREGISTEROBJECTNVPROC)wglGetProcAddress("wglDXRegisterObjectNV");
            wglDXUnregisterObjectNV_ = (PFNWGLDXUNREGISTEROBJECTNVPROC)wglGetProcAddress("wglDXUnregisterObjectNV");
            wglDXLockObjectsNV_      = (PFNWGLDXLOCKOBJECTSNVPROC)wglGetProcAddress("wglDXLockObjectsNV");
            wglDXUnlockObjectsNV_    = (PFNWGLDXUNLOCKOBJECTSNVPROC)wglGetProcAddress("wglDXUnlockObjectsNV");

            if (!wglDXOpenDeviceNV_ || !wglDXCloseDeviceNV_ || !wglDXRegisterObjectNV_ ||
                !wglDXUnregisterObjectNV_ || !wglDXLockObjectsNV_ || !wglDXUnlockObjectsNV_) {
                CASPAR_LOG(info) << L"[av_producer] WGL_NV_DX_interop2 not available - using CPU path";
                return false;
            }

            interop_device_ = wglDXOpenDeviceNV_(d3d11_device_);
            if (!interop_device_) {
                CASPAR_LOG(warning) << L"[av_producer] wglDXOpenDeviceNV failed - using CPU path";
                return false;
            }
            return true;
        });

        if (!ok)
            return false;

        CASPAR_LOG(info) << L"[av_producer] D3D11->GL GPU-direct bridge initialized";
        active_ = true;
        return true;
    }

    bool setup_for_size(int width, int height)
    {
        if (width == width_ && height == height_ && bgra_texture_)
            return true;

        // Tear down old resources
        teardown_interop();
        teardown_video_processor();

        width_  = width;
        height_ = height;

        // Create BGRA output texture
        D3D11_TEXTURE2D_DESC desc = {};
        desc.Width            = width;
        desc.Height           = height;
        desc.MipLevels        = 1;
        desc.ArraySize        = 1;
        desc.Format           = DXGI_FORMAT_B8G8R8A8_UNORM;
        desc.SampleDesc.Count = 1;
        desc.Usage            = D3D11_USAGE_DEFAULT;
        desc.BindFlags        = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
        desc.MiscFlags        = 0; // no sharing needed — interop handles it

        HRESULT hr = d3d11_device_->CreateTexture2D(&desc, nullptr, &bgra_texture_);
        if (FAILED(hr))
            return false;

        // Create video processor for NV12→BGRA
        HRESULT vhr;
        vhr = d3d11_device_->QueryInterface(__uuidof(ID3D11VideoDevice), (void**)&video_device_);
        if (FAILED(vhr))
            return false;

        d3d11_ctx_->QueryInterface(__uuidof(ID3D11VideoContext), (void**)&video_ctx_);

        D3D11_VIDEO_PROCESSOR_CONTENT_DESC vp_desc = {};
        vp_desc.InputFrameFormat            = D3D11_VIDEO_FRAME_FORMAT_PROGRESSIVE;
        vp_desc.InputWidth                  = width;
        vp_desc.InputHeight                 = height;
        vp_desc.OutputWidth                 = width;
        vp_desc.OutputHeight                = height;
        vp_desc.Usage                       = D3D11_VIDEO_USAGE_PLAYBACK_NORMAL;

        vhr = video_device_->CreateVideoProcessorEnumerator(&vp_desc, &vp_enum_);
        if (FAILED(vhr))
            return false;

        vhr = video_device_->CreateVideoProcessor(vp_enum_, 0, &video_processor_);
        if (FAILED(vhr))
            return false;

        // Create output view for the BGRA texture
        D3D11_VIDEO_PROCESSOR_OUTPUT_VIEW_DESC ov_desc = {};
        ov_desc.ViewDimension = D3D11_VPOV_DIMENSION_TEXTURE2D;
        vhr = video_device_->CreateVideoProcessorOutputView(bgra_texture_, vp_enum_, &ov_desc, &vp_output_view_);
        if (FAILED(vhr))
            return false;

        // Register BGRA texture with GL via WGL_NV_DX_interop2
        glGenTextures(1, &interop_gl_tex_);
        interop_object_ = wglDXRegisterObjectNV_(
            interop_device_, bgra_texture_, interop_gl_tex_, GL_TEXTURE_2D, WGL_ACCESS_READ_ONLY_NV);
        if (!interop_object_) {
            CASPAR_LOG(warning) << L"[av_producer] wglDXRegisterObjectNV failed";
            return false;
        }

        return true;
    }

    std::shared_ptr<accelerator::ogl::texture>
    convert(ID3D11Texture2D* nv12_tex, int array_index, accelerator::ogl::device& ogl_dev)
    {
        if (!active_ || !bgra_texture_)
            return nullptr;

        // Create input view for the NV12 texture array slice
        D3D11_VIDEO_PROCESSOR_INPUT_VIEW_DESC iv_desc = {};
        iv_desc.FourCC               = 0;
        iv_desc.ViewDimension        = D3D11_VPIV_DIMENSION_TEXTURE2D;
        iv_desc.Texture2D.ArraySlice = array_index;
        iv_desc.Texture2D.MipSlice   = 0;

        ID3D11VideoProcessorInputView* input_view = nullptr;
        HRESULT hr = video_device_->CreateVideoProcessorInputView(nv12_tex, vp_enum_, &iv_desc, &input_view);
        if (FAILED(hr))
            return nullptr;

        // Run the video processor: NV12→BGRA
        D3D11_VIDEO_PROCESSOR_STREAM stream = {};
        stream.Enable      = TRUE;
        stream.pInputSurface = input_view;

        video_ctx_->VideoProcessorBlt(video_processor_, vp_output_view_, 0, 1, &stream);
        input_view->Release();

        // Lock the D3D11 texture for GL access
        if (!wglDXLockObjectsNV_(interop_device_, 1, &interop_object_))
            return nullptr;

        // Create OGL texture and copy from the interop texture.
        // Must go through device::create_texture: it draws from the texture pool
        // (avoiding VRAM churn) and stamps the owning device onto the texture,
        // which the mixer checks before binding it. A directly constructed
        // texture has no owner and the mixer would reject it, silently falling
        // back to a host upload and undoing this whole GPU-direct path.
        auto ogl_tex = ogl_dev.dispatch_sync([&]() {
            auto tex = ogl_dev.create_texture(width_, height_, 4, common::bit_depth::bit8, false);
            tex->copy_from(static_cast<int>(interop_gl_tex_));
            return tex;
        });

        wglDXUnlockObjectsNV_(interop_device_, 1, &interop_object_);
        return ogl_tex;
    }

    /// Builds the plane-extraction resources for a given frame size. Returns
    /// false if anything is unavailable, in which case the caller falls back to
    /// the CPU transfer path -- this is opt-in and must never take a channel down.
    bool setup_planes(int width, int height, DXGI_FORMAT src_format)
    {
        if (plane_path_ok_ && width == width_ && height == height_ && src_format == expected_format_)
            return true;

        teardown_planes();
        width_  = width;
        height_ = height;

        // Only the semi-planar YCbCr layouts hardware decoders produce are
        // supported; anything else must take the CPU path rather than be
        // reinterpreted.
        expected_format_ = src_format;
        switch (src_format) {
            case DXGI_FORMAT_NV12:
                y_view_format_  = DXGI_FORMAT_R8_UNORM;
                uv_view_format_ = DXGI_FORMAT_R8G8_UNORM;
                plane_depth_    = common::bit_depth::bit8;
                break;
            case DXGI_FORMAT_P010:
            case DXGI_FORMAT_P016:
                y_view_format_  = DXGI_FORMAT_R16_UNORM;
                uv_view_format_ = DXGI_FORMAT_R16G16_UNORM;
                plane_depth_    = common::bit_depth::bit16;
                break;
            default:
                CASPAR_LOG(info) << L"[av_producer] GPU-direct: unsupported decoded surface format "
                                 << static_cast<int>(src_format) << L"; using the CPU path";
                return false;
        }

        // Full-screen triangle from SV_VertexID; no vertex or index buffers.
        static const char* kVS =
            "struct VSOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };\n"
            "VSOut main(uint vid : SV_VertexID)\n"
            "{\n"
            "    VSOut o;\n"
            "    float2 t = float2((vid << 1) & 2, vid & 2);\n"
            "    o.uv  = t;\n"
            "    o.pos = float4(t * float2(2, -2) + float2(-1, 1), 0, 1);\n"
            "    return o;\n"
            "}\n";

        // Pass-through. Whatever the plane SRV yields is written unchanged: the
        // SRV format decides which components are meaningful (R8 for Y, R8G8 for
        // CbCr). No arithmetic happens here, by design -- the colour conversion
        // belongs to the mixer's shader, which is the entire point of this path.
        //
        // Load(), not Sample(): the decoded surface is padded up to the codec's
        // macroblock grid (H.264 stores 1080 as 1088), so normalised coordinates
        // span the padding too and silently rescale the picture -- a ~4 row
        // vertical shift at 1080p. Indexing texels directly copies the cropped
        // top-left region exactly, and needs no sampler at all.
        static const char* kPS =
            "Texture2D src : register(t0);\n"
            "struct VSOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };\n"
            "float4 main(VSOut i) : SV_Target { return src.Load(int3(i.pos.xy, 0)); }\n";

        auto compile = [](const char* src, const char* target, ID3DBlob** out) -> bool {
            ID3DBlob* err = nullptr;
            auto      hr  = D3DCompile(src, std::strlen(src), nullptr, nullptr, nullptr, "main", target,
                                       D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, out, &err);
            if (err)
                err->Release();
            return SUCCEEDED(hr);
        };

        ID3DBlob* vs_blob = nullptr;
        ID3DBlob* ps_blob = nullptr;
        if (!compile(kVS, "vs_5_0", &vs_blob) || !compile(kPS, "ps_5_0", &ps_blob)) {
            if (vs_blob)
                vs_blob->Release();
            if (ps_blob)
                ps_blob->Release();
            return false;
        }

        bool ok = SUCCEEDED(d3d11_device_->CreateVertexShader(
                      vs_blob->GetBufferPointer(), vs_blob->GetBufferSize(), nullptr, &plane_vs_)) &&
                  SUCCEEDED(d3d11_device_->CreatePixelShader(
                      ps_blob->GetBufferPointer(), ps_blob->GetBufferSize(), nullptr, &plane_ps_));
        vs_blob->Release();
        ps_blob->Release();
        if (!ok)
            return false;

        D3D11_SAMPLER_DESC sd = {};
        // Point sampling: this pass copies texels, it must not resample them.
        sd.Filter   = D3D11_FILTER_MIN_MAG_MIP_POINT;
        sd.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
        sd.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
        sd.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
        if (FAILED(d3d11_device_->CreateSamplerState(&sd, &plane_sampler_)))
            return false;

        auto make_plane = [&](int w, int h, DXGI_FORMAT fmt, ID3D11Texture2D** tex,
                              ID3D11RenderTargetView** rtv) -> bool {
            D3D11_TEXTURE2D_DESC td = {};
            td.Width                = w;
            td.Height               = h;
            td.MipLevels            = 1;
            td.ArraySize            = 1;
            td.Format               = fmt;
            td.SampleDesc.Count     = 1;
            td.Usage                = D3D11_USAGE_DEFAULT;
            td.BindFlags            = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
            // The only difference between the two mixers' plane textures.
            // WGL_NV_DX_interop2 shares a texture without one; Vulkan needs a
            // real shared handle to import through VK_KHR_external_memory_win32.
            if (backend_ == backend::vulkan)
                td.MiscFlags = D3D11_RESOURCE_MISC_SHARED | D3D11_RESOURCE_MISC_SHARED_NTHANDLE;
            if (FAILED(d3d11_device_->CreateTexture2D(&td, nullptr, tex)))
                return false;
            return SUCCEEDED(d3d11_device_->CreateRenderTargetView(*tex, nullptr, rtv));
        };

        if (!make_plane(width, height, y_view_format_, &y_texture_, &y_rtv_) ||
            !make_plane(width / 2, height / 2, uv_view_format_, &uv_texture_, &uv_rtv_))
            return false;

        if (backend_ == backend::vulkan) {
            auto share = [&](ID3D11Texture2D* tex, HANDLE* out) -> bool {
                IDXGIResource1* res = nullptr;
                if (FAILED(tex->QueryInterface(__uuidof(IDXGIResource1), reinterpret_cast<void**>(&res))) || !res)
                    return false;
                auto hr = res->CreateSharedHandle(
                    nullptr, DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE, nullptr, out);
                res->Release();
                return SUCCEEDED(hr) && *out != nullptr;
            };
            if (!share(y_texture_, &y_shared_handle_) || !share(uv_texture_, &uv_shared_handle_)) {
                CASPAR_LOG(warning) << L"[av_producer] bridge: could not create shared handles for the NV12 planes";
                return false;
            }

            // Used once per frame to know the D3D11 extraction has actually
            // finished before Vulkan reads the planes. Prefer the fence, which
            // can be waited on rather than polled.
            ID3D11Device5* dev5 = nullptr;
            if (SUCCEEDED(d3d11_device_->QueryInterface(__uuidof(ID3D11Device5), reinterpret_cast<void**>(&dev5))) &&
                dev5) {
                if (FAILED(dev5->CreateFence(0, D3D11_FENCE_FLAG_NONE, __uuidof(ID3D11Fence),
                                             reinterpret_cast<void**>(&sync_fence_))))
                    sync_fence_ = nullptr;
                dev5->Release();
            }
            if (sync_fence_ &&
                FAILED(d3d11_ctx_->QueryInterface(__uuidof(ID3D11DeviceContext4),
                                                  reinterpret_cast<void**>(&d3d11_ctx4_)))) {
                sync_fence_->Release();
                sync_fence_ = nullptr;
            }
            if (sync_fence_) {
                sync_event_ = CreateEventW(nullptr, FALSE, FALSE, nullptr);
                if (!sync_event_) {
                    sync_fence_->Release();
                    sync_fence_ = nullptr;
                    d3d11_ctx4_->Release();
                    d3d11_ctx4_ = nullptr;
                }
            }

            if (!sync_fence_) {
                D3D11_QUERY_DESC qd = {D3D11_QUERY_EVENT, 0};
                if (FAILED(d3d11_device_->CreateQuery(&qd, &sync_query_))) {
                    CASPAR_LOG(warning) << L"[av_producer] bridge: no way to wait for the D3D11 plane extraction "
                                           L"(neither an ID3D11Fence nor a query could be created)";
                    return false;
                }
                CASPAR_LOG(info) << L"[av_producer] bridge: ID3D11Fence unavailable; polling a query instead, "
                                    L"which costs CPU while it waits";
            }

            plane_path_ok_ = true;
            return true;
        }

        // GL object creation and interop registration belong on the mixer's GL
        // thread, where the context these names live in is current.
        bool registered = ogl_dev_->dispatch_sync([&]() -> bool {
            glGenTextures(1, &y_gl_tex_);
            glGenTextures(1, &uv_gl_tex_);
            y_interop_obj_ = wglDXRegisterObjectNV_(interop_device_, y_texture_, y_gl_tex_, GL_TEXTURE_2D,
                                                    WGL_ACCESS_READ_ONLY_NV);
            uv_interop_obj_ = wglDXRegisterObjectNV_(interop_device_, uv_texture_, uv_gl_tex_, GL_TEXTURE_2D,
                                                     WGL_ACCESS_READ_ONLY_NV);
            return y_interop_obj_ != nullptr && uv_interop_obj_ != nullptr;
        });

        if (!registered) {
            CASPAR_LOG(warning) << L"[av_producer] bridge: could not register NV12 planes for GL interop";
            return false;
        }

        plane_path_ok_ = true;
        return true;
    }

    /// Extracts the decoded NV12 surface into two mixer textures (Y, and CbCr
    /// interleaved at half resolution), leaving colour conversion to the mixer.
    /// Returns a pair of nulls on any failure.
    std::pair<std::shared_ptr<core::texture>, std::shared_ptr<core::texture>>
    convert_planes(ID3D11Texture2D* nv12, int array_idx)
    {
        if (!plane_path_ok_ || !nv12) {
            CASPAR_LOG(warning) << L"[av_producer] extract: planes not set up";
            return {};
        }

        // What did the decoder actually give us? The plane views must match the
        // surface's real format, and an 8-bit view over a 10-bit surface is
        // rejected outright.
        D3D11_TEXTURE2D_DESC src_desc = {};
        nv12->GetDesc(&src_desc);
        if (src_desc.Format != expected_format_) {
            CASPAR_LOG(warning) << L"[av_producer] extract: surface format " << static_cast<int>(src_desc.Format)
                                << L" is not the expected " << static_cast<int>(expected_format_);
            return {};
        }

        // Plane SRVs require the D3D11.3 view descriptors (PlaneSlice). The
        // decoded surface is an array; index the slice this frame lives in.
        ID3D11Device3* dev3 = nullptr;
        if (FAILED(d3d11_device_->QueryInterface(__uuidof(ID3D11Device3), reinterpret_cast<void**>(&dev3))) || !dev3) {
            CASPAR_LOG(warning) << L"[av_producer] extract: ID3D11Device3 unavailable (no plane views)";
            return {};
        }

        auto make_srv = [&](DXGI_FORMAT fmt, UINT plane, ID3D11ShaderResourceView1** srv) -> bool {
            D3D11_SHADER_RESOURCE_VIEW_DESC1 sd = {};
            sd.Format                           = fmt;
            sd.ViewDimension                    = D3D11_SRV_DIMENSION_TEXTURE2DARRAY;
            sd.Texture2DArray.MostDetailedMip   = 0;
            sd.Texture2DArray.MipLevels         = 1;
            sd.Texture2DArray.FirstArraySlice   = static_cast<UINT>(array_idx);
            sd.Texture2DArray.ArraySize         = 1;
            sd.Texture2DArray.PlaneSlice        = plane;
            return SUCCEEDED(dev3->CreateShaderResourceView1(nv12, &sd, srv));
        };

        ID3D11ShaderResourceView1* y_srv  = nullptr;
        ID3D11ShaderResourceView1* uv_srv = nullptr;
        if (!make_srv(y_view_format_, 0, &y_srv) || !make_srv(uv_view_format_, 1, &uv_srv)) {
            CASPAR_LOG(warning) << L"[av_producer] extract: could not create NV12 plane views (surface not bound "
                                   L"for shader access?)";
            if (y_srv)
                y_srv->Release();
            if (uv_srv)
                uv_srv->Release();
            dev3->Release();
            return {};
        }

        auto draw_plane = [&](ID3D11RenderTargetView* rtv, ID3D11ShaderResourceView1* srv, int w, int h) {
            D3D11_VIEWPORT vp = {};
            vp.Width          = static_cast<float>(w);
            vp.Height         = static_cast<float>(h);
            vp.MaxDepth       = 1.0f;

            ID3D11ShaderResourceView* srv0 = srv;
            d3d11_ctx_->OMSetRenderTargets(1, &rtv, nullptr);
            d3d11_ctx_->RSSetViewports(1, &vp);
            d3d11_ctx_->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            d3d11_ctx_->IASetInputLayout(nullptr);
            d3d11_ctx_->VSSetShader(plane_vs_, nullptr, 0);
            d3d11_ctx_->PSSetShader(plane_ps_, nullptr, 0);
            d3d11_ctx_->PSSetShaderResources(0, 1, &srv0);
            d3d11_ctx_->Draw(3, 0);

            ID3D11ShaderResourceView* none = nullptr;
            d3d11_ctx_->PSSetShaderResources(0, 1, &none);
        };

        // Vulkan may still be reading last frame's planes; the D3D11 draws below
        // overwrite them, and nothing but this orders the two queues. Done
        // before taking the decoder's lock so the wait never blocks the decoder.
#ifdef ENABLE_VULKAN
        if (backend_ == backend::vulkan && vk_import_)
            vk_wait_us_ += vk_import_->wait_for_previous_copy();
#endif

        // The decoder shares this immediate context, and it is not free-threaded.
        if (d3d11_hwctx_ && d3d11_hwctx_->lock)
            d3d11_hwctx_->lock(d3d11_hwctx_->lock_ctx);

        draw_plane(y_rtv_, y_srv, width_, height_);
        draw_plane(uv_rtv_, uv_srv, width_ / 2, height_ / 2);
        d3d11_ctx_->Flush();

        // Vulkan reads these textures from a different queue, so "submitted" is
        // not good enough -- it has to be "finished". GL does not need this:
        // wglDXLockObjectsNV synchronises for us.
        //
        // Only the *signal* belongs inside the decoder's lock. The wait that
        // follows it must not hold the lock: it is milliseconds long at four
        // layers, and the decoder needs the context during exactly that window.
        const bool     use_fence  = backend_ == backend::vulkan && sync_fence_ != nullptr;
        const bool     use_query  = backend_ == backend::vulkan && !use_fence && sync_query_ != nullptr;
        const auto     sync_start = std::chrono::steady_clock::now();

        if (use_fence)
            d3d11_ctx4_->Signal(sync_fence_, ++sync_value_);

        if (use_query) {
            // No fence available. GetData is an immediate-context call, so this
            // has to poll with the decoder's lock held, and it burns a core
            // while it does. See the note on sync_fence_.
            d3d11_ctx_->End(sync_query_);
            BOOL done = FALSE;
            while (d3d11_ctx_->GetData(sync_query_, &done, sizeof(done), 0) != S_OK)
                std::this_thread::yield();
        }

        if (d3d11_hwctx_ && d3d11_hwctx_->unlock)
            d3d11_hwctx_->unlock(d3d11_hwctx_->lock_ctx);

        // Fence methods are free-threaded, so this half happens with the lock
        // released -- which is the point of using a fence at all.
        if (use_fence && sync_fence_->GetCompletedValue() < sync_value_ &&
            SUCCEEDED(sync_fence_->SetEventOnCompletion(sync_value_, sync_event_))) {
            // Bounded: a lost signal must fall the producer back, not wedge its
            // decode thread for good.
            if (WaitForSingleObject(sync_event_, 2000) != WAIT_OBJECT_0)
                CASPAR_LOG(warning) << L"[av_producer] timed out waiting for the D3D11 plane extraction";
        }

        if (use_fence || use_query)
            sync_wait_us_ += std::chrono::duration_cast<std::chrono::microseconds>(
                                 std::chrono::steady_clock::now() - sync_start)
                                 .count();

        y_srv->Release();
        uv_srv->Release();
        dev3->Release();

        if (backend_ == backend::vulkan) {
#ifdef ENABLE_VULKAN
            std::shared_ptr<core::texture> y_out;
            std::shared_ptr<core::texture> uv_out;
            if (!vk_import_ || !vk_import_->copy_planes(y_shared_handle_, uv_shared_handle_, width_, height_,
                                                        width_ / 2, height_ / 2, plane_depth_, y_out, uv_out))
                return {};
            report_sync_cost();
            return {std::move(y_out), std::move(uv_out)};
#else
            return {};
#endif
        }

        // Hand both planes to GL and copy into pooled textures, so the D3D11
        // surfaces are free for the next frame immediately.
        std::shared_ptr<accelerator::ogl::texture> y_out;
        std::shared_ptr<accelerator::ogl::texture> uv_out;

        ogl_dev_->dispatch_sync([&] {
            HANDLE objs[2] = {y_interop_obj_, uv_interop_obj_};
            if (!wglDXLockObjectsNV_(interop_device_, 2, objs)) {
                CASPAR_LOG(warning) << L"[av_producer] extract: wglDXLockObjectsNV failed";
                return;
            }

            y_out  = ogl_dev_->create_texture(width_, height_, 1, plane_depth_, false);
            uv_out = ogl_dev_->create_texture(width_ / 2, height_ / 2, 2, plane_depth_, false);
            if (y_out && uv_out) {
                glCopyImageSubData(y_gl_tex_, GL_TEXTURE_2D, 0, 0, 0, 0, y_out->id(), GL_TEXTURE_2D, 0, 0, 0, 0,
                                   width_, height_, 1);
                glCopyImageSubData(uv_gl_tex_, GL_TEXTURE_2D, 0, 0, 0, 0, uv_out->id(), GL_TEXTURE_2D, 0, 0, 0, 0,
                                   width_ / 2, height_ / 2, 1);
            }

            wglDXUnlockObjectsNV_(interop_device_, 2, objs);
        });

        if (!y_out || !uv_out)
            return {};

        return {std::static_pointer_cast<core::texture>(y_out), std::static_pointer_cast<core::texture>(uv_out)};
    }

    /// Reports what the cross-API synchronisation actually costs, once at info
    /// so it always lands in the log, then at debug so it stays measurable
    /// without becoming noise. Reported per frame, not per window, because that
    /// is the number that can be compared against the frame budget.
    void report_sync_cost()
    {
        constexpr int64_t kWindow = 250;
        if (++sync_frames_ < kWindow)
            return;

        const auto d3d11_us = static_cast<double>(sync_wait_us_) / static_cast<double>(sync_frames_);
        const auto vk_us    = static_cast<double>(vk_wait_us_) / static_cast<double>(sync_frames_);

        std::wostringstream msg;
        msg << L"[av_producer] GPU-direct (Vulkan) cross-API sync per frame: D3D11 completion wait "
            << std::fixed << std::setprecision(3) << d3d11_us / 1000.0 << L" ms, previous-copy wait "
            << vk_us / 1000.0 << L" ms, over " << sync_frames_ << L" frames.";
        if (!sync_reported_) {
            sync_reported_ = true;
            CASPAR_LOG(info) << msg.str();
        } else {
            CASPAR_LOG(debug) << msg.str();
        }

        sync_wait_us_ = 0;
        vk_wait_us_   = 0;
        sync_frames_  = 0;
    }

    void teardown_planes()
    {
#ifdef ENABLE_VULKAN
        // The importer holds VkImages aliasing the D3D11 textures released
        // below, and a copy may still be in flight reading them. Drop the
        // imports first -- but keep the importer itself, because this also runs
        // when the frame size changes and the path has to carry on afterwards.
        if (vk_import_)
            vk_import_->release_imports();
#endif
        if (sync_query_) {
            sync_query_->Release();
            sync_query_ = nullptr;
        }
        if (sync_fence_) {
            sync_fence_->Release();
            sync_fence_ = nullptr;
        }
        if (d3d11_ctx4_) {
            d3d11_ctx4_->Release();
            d3d11_ctx4_ = nullptr;
        }
        if (sync_event_) {
            CloseHandle(sync_event_);
            sync_event_ = nullptr;
        }
        sync_value_ = 0;
        if (y_shared_handle_) {
            CloseHandle(y_shared_handle_);
            y_shared_handle_ = nullptr;
        }
        if (uv_shared_handle_) {
            CloseHandle(uv_shared_handle_);
            uv_shared_handle_ = nullptr;
        }
        if (y_interop_obj_) {
            wglDXUnregisterObjectNV_(interop_device_, y_interop_obj_);
            y_interop_obj_ = nullptr;
        }
        if (uv_interop_obj_) {
            wglDXUnregisterObjectNV_(interop_device_, uv_interop_obj_);
            uv_interop_obj_ = nullptr;
        }
        if (y_gl_tex_) {
            glDeleteTextures(1, &y_gl_tex_);
            y_gl_tex_ = 0;
        }
        if (uv_gl_tex_) {
            glDeleteTextures(1, &uv_gl_tex_);
            uv_gl_tex_ = 0;
        }
        if (y_rtv_) { y_rtv_->Release(); y_rtv_ = nullptr; }
        if (uv_rtv_) { uv_rtv_->Release(); uv_rtv_ = nullptr; }
        if (y_texture_) { y_texture_->Release(); y_texture_ = nullptr; }
        if (uv_texture_) { uv_texture_->Release(); uv_texture_ = nullptr; }
        if (plane_vs_) { plane_vs_->Release(); plane_vs_ = nullptr; }
        if (plane_ps_) { plane_ps_->Release(); plane_ps_ = nullptr; }
        if (plane_sampler_) { plane_sampler_->Release(); plane_sampler_ = nullptr; }
        plane_path_ok_ = false;
    }

    bool is_active() const { return active_; }

    /// Depth of the extracted planes: bit8 for NV12, bit16 for P010/P016.
    common::bit_depth plane_depth() const { return plane_depth_; }

    void cleanup()
    {
        teardown_planes();
#ifdef ENABLE_VULKAN
        vk_import_.reset();
#endif
        teardown_interop();
        teardown_video_processor();

        // The interop device belongs to the mixer's GL context, so close it on
        // that thread. There is no private context or dummy window to tear down
        // any more -- the bridge borrows the mixer's.
        if (interop_device_ && ogl_dev_) {
            ogl_dev_->dispatch_sync([&] {
                wglDXCloseDeviceNV_(interop_device_);
                interop_device_ = nullptr;
            });
        }
        interop_device_ = nullptr;
        active_         = false;
    }

    ~d3d11_bridge() { cleanup(); }

  private:
    void teardown_interop()
    {
        if (interop_object_) {
            wglDXUnregisterObjectNV_(interop_device_, interop_object_);
            interop_object_ = nullptr;
        }
        if (interop_gl_tex_) {
            glDeleteTextures(1, &interop_gl_tex_);
            interop_gl_tex_ = 0;
        }
    }

    void teardown_video_processor()
    {
        if (vp_output_view_) { vp_output_view_->Release(); vp_output_view_ = nullptr; }
        if (video_processor_) { video_processor_->Release(); video_processor_ = nullptr; }
        if (vp_enum_) { vp_enum_->Release(); vp_enum_ = nullptr; }
        if (video_ctx_) { video_ctx_->Release(); video_ctx_ = nullptr; }
        if (video_device_) { video_device_->Release(); video_device_ = nullptr; }
        if (bgra_texture_) { bgra_texture_->Release(); bgra_texture_ = nullptr; }
    }
};
#endif // _WIN32

struct Frame
{
    std::shared_ptr<AVFrame> video;
    std::shared_ptr<AVFrame> audio;
    core::draw_frame         frame;
    int64_t                  start_time  = AV_NOPTS_VALUE;
    int64_t                  pts         = AV_NOPTS_VALUE;
    int64_t                  duration    = 0;
    int64_t                  frame_count = 0;
};

AVPixelFormat get_pix_fmt_with_alpha(AVPixelFormat fmt)
{
    switch (fmt) {
        case AV_PIX_FMT_YUV420P:
            return AV_PIX_FMT_YUVA420P;
        case AV_PIX_FMT_YUV422P:
            return AV_PIX_FMT_YUVA422P;
        case AV_PIX_FMT_YUV444P:
            return AV_PIX_FMT_YUVA444P;
        default:
            break;
    }
    return fmt;
}

/// What the decode adapter can do with AV1, as reported by D3D11 itself.
enum class av1_hw_support
{
    none,   ///< no AV1 decoder on this adapter
    main8,  ///< AV1 Profile 0, 8-bit only
    main10, ///< AV1 Profile 0, 8- and 10-bit
};

/// Which DXGI adapter the decoder should put its D3D11VA device on.
///
/// It used to be adapter 0 unconditionally, which was harmless while GPU-direct
/// was OpenGL-only -- the OpenGL mixer refuses gpu_index != 0, so the mixer was
/// always GPU 0 too. 489b02fbc brought GPU-direct to the Vulkan mixer, and
/// Vulkan *does* implement GPU affinity, so a channel on GPU 1 had its frames
/// decoded on adapter 0 and then handed to d3d11_import_bridge, which cannot
/// import across devices: "this D3D11 handle is not importable (result=-3)",
/// and the channel silently dropped to the host transfer path.
///
/// Returns -1 for "the default adapter", which is what a null device string
/// gives av_hwdevice_ctx_create.
int resolve_decode_adapter(const core::frame_factory* factory)
{
    // An explicit setting always wins -- it is the escape hatch when DXGI
    // enumeration and the mixer disagree, or when the decode should deliberately
    // run on a different card from the mixer.
    const int configured = env::properties().get(L"configuration.ffmpeg.producer.hardware-decode-adapter", -1);
    if (configured >= 0)
        return configured;

#ifdef _WIN32
    if (!factory || factory->gpu_device_backend() != core::gpu_backend::vulkan)
        return -1; // OpenGL has no affinity, so its mixer is always the default adapter

    // The lookup lives in the accelerator: matching a VkPhysicalDevice to a DXGI
    // adapter needs the Vulkan headers, which this module does not carry.
    const int index = accelerator::vulkan::dxgi_adapter_for_vk_device(factory->gpu_device_handle());
    if (index < 0) {
        CASPAR_LOG(warning) << L"[av_producer] no DXGI adapter matches the mixer's GPU; hardware decode will use "
                               L"the default adapter and GPU-direct may decline";
    }
    return index;
#else
    return -1;
#endif
}

#ifdef _WIN32
/// Ask the adapter the decoder will actually use whether it decodes AV1.
///
/// This has to be asked, not assumed, because FFmpeg's native `av1` decoder has
/// NO software path -- it is hwaccel-only, and on an adapter without an AV1
/// engine it does not degrade, it fails outright:
///
///     Your platform doesn't support hardware accelerated AV1 decoding.
///     Error submitting packet to decoder: Function not implemented
///
/// which for a playout server is a black channel. CheckVideoDecoderFormat is a
/// static query -- no decoding, no packets -- and it is the same question the
/// decoder will ask later, put to the same adapter.
///
/// Cached per adapter: one D3D11 device creation each, not one per file. The
/// adapter matters -- on a mixed rig the answer differs between cards, and this
/// machine is exactly that case (Ampere A4000 decodes AV1, Pascal P4000 does
/// not), so a process-wide answer would be wrong for one of them.
av1_hw_support probe_av1_hw_decode(int adapter)
{
    static std::mutex                    cache_mutex;
    static std::map<int, av1_hw_support> cache;
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        auto                        it = cache.find(adapter);
        if (it != cache.end())
            return it->second;
    }

    const auto probe = [&]() -> av1_hw_support {
        const std::string adapter_str = adapter >= 0 ? std::to_string(adapter) : std::string();

        AVBufferRef* hw_ctx = nullptr;
        if (av_hwdevice_ctx_create(
                &hw_ctx, AV_HWDEVICE_TYPE_D3D11VA, adapter >= 0 ? adapter_str.c_str() : nullptr, nullptr, 0) != 0 ||
            !hw_ctx) {
            CASPAR_LOG(info) << L"[av_producer] AV1 probe: no D3D11VA device; AV1 will decode in software";
            return av1_hw_support::none;
        }

        auto unref = [&] { av_buffer_unref(&hw_ctx); };

        auto* hwctx  = reinterpret_cast<AVHWDeviceContext*>(hw_ctx->data);
        auto* d3dctx = static_cast<AVD3D11VADeviceContext*>(hwctx->hwctx);
        if (!d3dctx || !d3dctx->device) {
            unref();
            return av1_hw_support::none;
        }

        ID3D11VideoDevice* video_device = nullptr;
        if (FAILED(d3dctx->device->QueryInterface(__uuidof(ID3D11VideoDevice),
                                                  reinterpret_cast<void**>(&video_device))) ||
            !video_device) {
            CASPAR_LOG(info) << L"[av_producer] AV1 probe: adapter exposes no ID3D11VideoDevice";
            unref();
            return av1_hw_support::none;
        }

        // Name the adapter. On a multi-GPU box the decoder takes DXGI adapter 0,
        // which need not be the card anyone expects, and "AV1 decodes in
        // software" is otherwise indistinguishable from "wrong card".
        std::wstring adapter_name = L"<unknown>";
        {
            IDXGIDevice*  dxgi_device  = nullptr;
            IDXGIAdapter* dxgi_adapter = nullptr;
            if (SUCCEEDED(d3dctx->device->QueryInterface(__uuidof(IDXGIDevice),
                                                         reinterpret_cast<void**>(&dxgi_device))) &&
                dxgi_device) {
                if (SUCCEEDED(dxgi_device->GetAdapter(&dxgi_adapter)) && dxgi_adapter) {
                    DXGI_ADAPTER_DESC desc = {};
                    if (SUCCEEDED(dxgi_adapter->GetDesc(&desc)))
                        adapter_name = desc.Description;
                    dxgi_adapter->Release();
                }
                dxgi_device->Release();
            }
        }

        const auto supports = [&](DXGI_FORMAT fmt) {
            BOOL ok = FALSE;
            return SUCCEEDED(video_device->CheckVideoDecoderFormat(&D3D11_DECODER_PROFILE_AV1_VLD_PROFILE0, fmt, &ok)) &&
                   ok;
        };

        // Profile 0 is 4:2:0 8- or 10-bit by specification; NV12 and P010 are the
        // two surface formats that can carry it.
        const bool nv12 = supports(DXGI_FORMAT_NV12);
        const bool p010 = supports(DXGI_FORMAT_P010);

        video_device->Release();
        unref();

        const auto result = p010 ? av1_hw_support::main10 : (nv12 ? av1_hw_support::main8 : av1_hw_support::none);

        CASPAR_LOG(info) << L"[av_producer] AV1 hardware decode on \"" << adapter_name << L"\": "
                         << (result == av1_hw_support::main10  ? L"Profile 0, 8- and 10-bit"
                             : result == av1_hw_support::main8 ? L"Profile 0, 8-bit only"
                                                               : L"not supported");
        return result;
    };

    const auto result = probe();
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        cache[adapter] = result;
    }
    return result;
}
#else
av1_hw_support probe_av1_hw_decode(int /*adapter*/) { return av1_hw_support::none; }
#endif

/// configuration.ffmpeg.producer.av1-decoder: auto | hardware | software.
/// `hardware` bypasses the capability gate -- see the warning where it is used.
std::wstring av1_decoder_mode()
{
    static const std::wstring value = [] {
        auto v = env::properties().get<std::wstring>(L"configuration.ffmpeg.producer.av1-decoder", L"auto");
        boost::to_lower(v);
        if (v != L"auto" && v != L"hardware" && v != L"software") {
            CASPAR_LOG(warning) << L"[av_producer] unknown av1-decoder value \"" << v << L"\"; using auto";
            return std::wstring(L"auto");
        }
        return v;
    }();
    return value;
}

/// Which AV1 decoder to open, and why. Split out so the reason can be logged
/// once per file rather than inferred from a black picture.
const AVCodec* choose_av1_decoder(const AVStream* stream, int adapter, std::wstring& reason)
{
    const auto* software = avcodec_find_decoder_by_name("libdav1d");
    const auto* hardware = avcodec_find_decoder_by_name("av1");

    // NOTE: avcodec_find_decoder(AV_CODEC_ID_AV1) resolves to libdav1d, which
    // advertises no hwaccel at all. That is why AV1 never reached D3D11VA and
    // why the hardware decoder has to be asked for by name.
    if (!hardware) {
        reason = L"the native av1 decoder is not in this build";
        return software;
    }
    if (!software) {
        // Nothing to fall back to; the gate below would be meaningless.
        reason = L"libdav1d is not in this build";
        return hardware;
    }

    const auto mode = av1_decoder_mode();
    if (mode == L"software") {
        reason = L"av1-decoder=software";
        return software;
    }
    if (mode == L"hardware") {
        reason = L"av1-decoder=hardware (capability gate bypassed; unsupported streams will fail to decode)";
        return hardware;
    }

    if (!stream || !stream->codecpar) {
        reason = L"no stream parameters to check";
        return software;
    }

    // Profile 0 (Main) is 4:2:0 8/10-bit. Profile 1 is 4:4:4 and Profile 2 is
    // 4:2:2/12-bit, neither of which NVDEC or DXVA decode -- and the native
    // decoder cannot fall back to software for them. Gate on the profile rather
    // than the pixel format: the profile is in the container, whereas
    // codecpar->format can still be NONE before the first frame.
    if (stream->codecpar->profile != AV_PROFILE_AV1_MAIN) {
        reason = L"stream is not AV1 Profile 0 (4:4:4 and 12-bit have no hardware decoder)";
        return software;
    }

    const bool needs_10bit = stream->codecpar->format == AV_PIX_FMT_YUV420P10LE ||
                             stream->codecpar->bits_per_raw_sample > 8;

    const auto support = probe_av1_hw_decode(adapter);
    if (support == av1_hw_support::none) {
        reason = L"this GPU has no AV1 decoder";
        return software;
    }
    if (needs_10bit && support != av1_hw_support::main10) {
        reason = L"this GPU decodes AV1 at 8-bit only and the stream is 10-bit";
        return software;
    }

    reason.clear();
    return hardware;
}

const AVCodec* get_decoder(AVCodecID codec_id, const AVStream* stream, int decode_adapter)
{
    // libvpx is the only VP8/VP9 decoder that reads the alpha channel out of a
    // WebM, so it used to be forced for every VP8 and VP9 file. That bought
    // alpha at the cost of hardware decoding for all of them, because libvpx is
    // a software-only external library: FFmpeg's own vp9 decoder advertises
    // dxva2, d3d11va, d3d12va, cuda and vaapi, and none of it was reachable.
    // GPU-direct decode declined every VP9 clip with "decoder is not using
    // D3D11VA", which is true but reads like a hardware limitation rather than
    // a choice made here.
    //
    // The choice only has to be made per file. Matroska records alpha on the
    // track and the demuxer copies it to stream metadata as alpha_mode (see
    // matroskadec.c), so it is known before a decoder is opened. Take libvpx
    // when the file actually declares alpha, and the native decoder -- with
    // whatever hardware support it has -- otherwise.
    const bool declares_alpha =
        stream != nullptr && av_dict_get(stream->metadata, "alpha_mode", nullptr, 0) != nullptr;

    if (declares_alpha) {
        const AVCodec* result = nullptr;
        if (codec_id == AV_CODEC_ID_VP9)
            result = avcodec_find_decoder_by_name("libvpx-vp9");
        else if (codec_id == AV_CODEC_ID_VP8)
            result = avcodec_find_decoder_by_name("libvpx");
        if (result != nullptr)
            return result;
    }

    // AV1 has the same shape of problem as VP9 did, with a sharper edge:
    // avcodec_find_decoder resolves to libdav1d, which is software-only, so the
    // hardware engine went unused and GPU-direct declined every AV1 clip with
    // "decoder is not using D3D11VA". Unlike VP9 the native decoder has no
    // software path to fall back to, so the choice is gated rather than simply
    // reversed. See choose_av1_decoder.
    if (codec_id == AV_CODEC_ID_AV1) {
        std::wstring reason;
        const auto*  chosen = choose_av1_decoder(stream, decode_adapter, reason);
        if (chosen != nullptr) {
            if (reason.empty()) {
                CASPAR_LOG(info) << L"[av_producer] AV1: using the hardware decoder";
            } else {
                CASPAR_LOG(info) << L"[av_producer] AV1: using " << u16(chosen->name) << L" -- " << reason;
            }
            return chosen;
        }
    }

    return avcodec_find_decoder(codec_id);
}

// TODO (fix) Handle ts discontinuities.
// TODO (feat) Forward options.

namespace {
/// Server-wide ceiling on memory held by resident loop ranges, across every producer.
///
/// The per-producer budget alone does not bound anything useful: twenty layers each holding
/// their own allowance is twenty times the number an operator thought they were setting, and on
/// the GPU-direct path those bytes are VRAM shared with the mixer. So a range is RESERVED from a
/// shared allowance before a single frame is held, and a producer that cannot reserve simply
/// does not cache -- it plays exactly as it did before, which is the right way to run out.
std::atomic<int64_t> g_loop_cache_bytes{0};

int64_t loop_cache_total_limit()
{
    static const int64_t value =
        static_cast<int64_t>(env::properties().get(L"configuration.ffmpeg.producer.loop-cache-total-mb", 1024))
        << 20;
    return value;
}

/// All-or-nothing, because a partial range is worse than none: it pins memory and can never be
/// served from, which is the failure this replaces.
bool loop_cache_reserve(int64_t bytes)
{
    if (bytes <= 0)
        return false;
    const int64_t limit = loop_cache_total_limit();
    int64_t       cur   = g_loop_cache_bytes.load(std::memory_order_relaxed);
    while (cur + bytes <= limit) {
        if (g_loop_cache_bytes.compare_exchange_weak(cur, cur + bytes, std::memory_order_relaxed))
            return true;
    }
    return false;
}

void loop_cache_release(int64_t bytes)
{
    if (bytes > 0)
        g_loop_cache_bytes.fetch_sub(bytes, std::memory_order_relaxed);
}

#if defined(ENABLE_VULKAN) && LIBAVUTIL_VERSION_MAJOR >= 60
/// Describe an AVVkFrame's planes for the mixer's importer, or return an empty vector.
///
/// The geometry comes from `sw_format` rather than from the VkImages, because a plane's
/// dimensions are the pixel format's business and the images carry no subsampling
/// information. `depth_out` and `alpha_out` describe what the mixer must be told.
std::vector<accelerator::vulkan::av_plane_source>
describe_av_vulkan_planes(const AVFrame* av, common::bit_depth& depth_out, bool& alpha_out)
{
    std::vector<accelerator::vulkan::av_plane_source> out;

    if (!av || !av->hw_frames_ctx)
        return out;

    auto* vkf = reinterpret_cast<AVVkFrame*>(av->data[0]);
    auto* fc  = reinterpret_cast<AVHWFramesContext*>(av->hw_frames_ctx->data);
    if (!vkf || !fc)
        return out;

    const auto* pd = av_pix_fmt_desc_get(fc->sw_format);
    if (!pd)
        return out;

    const int nb_planes = av_pix_fmt_count_planes(fc->sw_format);
    if (nb_planes != 3 && nb_planes != 4)
        return out; // only planar YCbCr(A) reaches the mixer's ycbcr shapes

    // Each plane's VkFormat is R8_UNORM or R16_UNORM (hwcontext_vulkan's format table), so
    // a 10-bit sample is a code 0..1023 sitting in the LOW bits of a 16-bit word. That is
    // exactly what bit10 means to the mixer: precision factor 64. It is also why this path
    // inherits the precision loss recorded above for yuv420p10le -- chroma is upsampled by
    // the texture unit before the multiply, so rounding is amplified 64-fold. The decoder's
    // output format is not ours to choose, and the alternative is a CPU round trip.
    switch (pd->comp[0].depth) {
        case 8:
            depth_out = common::bit_depth::bit8;
            break;
        case 10:
            depth_out = common::bit_depth::bit10;
            break;
        case 12:
            depth_out = common::bit_depth::bit12;
            break;
        case 16:
            depth_out = common::bit_depth::bit16;
            break;
        default:
            return out;
    }
    alpha_out = nb_planes == 4;

    for (int i = 0; i < nb_planes; ++i) {
        accelerator::vulkan::av_plane_source ps;
        ps.image     = vkf->img[i];
        ps.semaphore = vkf->sem[i];
        ps.sem_value = vkf->sem_value[i];
        ps.layout    = static_cast<int>(vkf->layout[i]);
        // Chroma planes only; alpha in yuva* is full resolution like luma.
        const bool sub = i == 1 || i == 2;
        ps.width      = sub ? AV_CEIL_RSHIFT(av->width, pd->log2_chroma_w) : av->width;
        ps.height     = sub ? AV_CEIL_RSHIFT(av->height, pd->log2_chroma_h) : av->height;
        ps.components = 1;
        out.push_back(ps);
    }

    return out;
}
#endif

/// Whether the GPU-direct decode path is enabled in configuration. Read once.
bool gpu_direct_decode_requested()
{
    static const bool value =
        env::properties().get(L"configuration.ffmpeg.producer.gpu-direct-decode", true);
    return value;
}
} // namespace

class Decoder
{
    static enum AVPixelFormat get_hw_format(AVCodecContext* ctx, const enum AVPixelFormat* pix_fmts)
    {
        const enum AVPixelFormat* p;

        // Vulkan first, when the decoder offers it. An FFmpeg Vulkan compute decoder on
        // the mixer's own device produces frames that need no copy, so there is nothing
        // for this callback to prepare -- FFmpeg allocates the pool on the device it was
        // handed. Everything below is D3D11-specific and does not apply.
        // ONLY when the attached device is actually a Vulkan one. FFmpeg 8 advertises a
        // Vulkan hwaccel for h264, hevc, av1 and vp9 as well as the compute decoders, so
        // AV_PIX_FMT_VULKAN appears in `pix_fmts` for those codecs on the ORDINARY D3D11VA
        // path too. Returning it there hands the decoder a format its device cannot back:
        // FFmpeg answers "Invalid setup for format vulkan: does not match the type of the
        // provided device context" plus "Device does not support the
        // VK_KHR_video_decode_queue extension", hardware decode fails, and the producer
        // limps to software with a fatal in the log.
        //
        // Measured 2026-08-21: this cost `flat-decoded --mixer vulkan` two of its six
        // formats -- and `<vulkan-decode>` was OFF for that run, which is the point. The
        // check below is what makes this branch belong to the opt-in path only.
        const bool vulkan_device_attached =
            ctx->hw_device_ctx != nullptr &&
            reinterpret_cast<AVHWDeviceContext*>(ctx->hw_device_ctx->data)->type == AV_HWDEVICE_TYPE_VULKAN;

        for (p = pix_fmts; vulkan_device_attached && *p != -1; p++) {
            if (*p != AV_PIX_FMT_VULKAN)
                continue;
#if defined(ENABLE_VULKAN) && LIBAVUTIL_VERSION_MAJOR >= 60
            // ONE PLANE PER VkImage, and the whole hand-over depends on it. FFmpeg
            // defaults to a single multi-planar VkImage where the format allows one --
            // yuv422p10le has VK_FORMAT_G16_B16_R16_3PLANE_422_UNORM, so ProRes would
            // arrive as `img[0]` alone with three disjoint aspect planes inside it.
            // The mixer samples one VkImage per plane and its importer copies with
            // VK_IMAGE_ASPECT_COLOR_BIT, so a multiplane frame would be read as if it
            // were a single colour image: wrong size, wrong contents, no error.
            //
            // Set on the frames context rather than the device because the device is
            // hand-filled here (av_hwdevice_ctx_alloc + _init), which never reads the
            // `disable_multiplane` option a device-creation string would carry.
            //
            // Failing is not fatal: without a frames context FFmpeg allocates its own
            // and the publish side declines, falling back to the CPU transfer path.
            if (!ctx->hw_frames_ctx && ctx->hw_device_ctx) {
                AVBufferRef* frames_ref = nullptr;
                if (avcodec_get_hw_frames_parameters(ctx, ctx->hw_device_ctx, AV_PIX_FMT_VULKAN, &frames_ref) >= 0 &&
                    frames_ref) {
                    auto* frames = reinterpret_cast<AVHWFramesContext*>(frames_ref->data);
                    auto* vk     = static_cast<AVVulkanFramesContext*>(frames->hwctx);
                    // `flags` is a C enum, so |= needs the cast back in C++.
                    vk->flags = static_cast<AVVkFrameFlags>(vk->flags | AV_VK_FRAME_FLAG_DISABLE_MULTIPLANE);
                    if (av_hwframe_ctx_init(frames_ref) >= 0) {
                        ctx->hw_frames_ctx = frames_ref;
                    } else {
                        CASPAR_LOG(warning) << L"[av_producer] could not initialise a single-plane Vulkan "
                                               L"frame pool; the decoded frames will not reach the mixer "
                                               L"directly";
                        av_buffer_unref(&frames_ref);
                    }
                }
            }
#endif
            CASPAR_LOG(debug) << L"[av_producer] decoder chose AV_PIX_FMT_VULKAN";
            return AV_PIX_FMT_VULKAN;
        }

        for (p = pix_fmts; *p != -1; p++) {
            if (*p != AV_PIX_FMT_D3D11)
                continue;

            // Allocate the frame pool here, where ffmpeg is ready for it, adding
            // shader-resource binding so the GPU-direct path can create plane
            // views over the decoded surfaces. FFmpeg's default pool binds them
            // for decode only. Failing is not fatal -- GPU-direct then declines
            // and the CPU transfer path runs.
            // Only when GPU-direct was actually requested. Adding
            // SHADER_RESOURCE unconditionally changes the pool for everyone, and
            // for P010 surfaces it made av_hwframe_transfer_data fail outright
            // ("Error transferring the data to system memory: Invalid argument")
            // -- i.e. it broke the ordinary CPU path for 10-bit HEVC.
            if (gpu_direct_decode_requested() && !ctx->hw_frames_ctx && ctx->hw_device_ctx) {
                AVBufferRef* frames_ref = nullptr;
                if (avcodec_get_hw_frames_parameters(ctx, ctx->hw_device_ctx, AV_PIX_FMT_D3D11, &frames_ref) >= 0 &&
                    frames_ref) {
                    auto* frames = reinterpret_cast<AVHWFramesContext*>(frames_ref->data);
                    auto* d3d11  = static_cast<AVD3D11VAFramesContext*>(frames->hwctx);
                    d3d11->BindFlags |= D3D11_BIND_SHADER_RESOURCE;

                    // The pool is sized for the decoder alone, but the GPU-direct
                    // path also holds surfaces while they queue for extraction.
                    // Without headroom the decoder hits "Static surface pool size
                    // exceeded" and stops producing frames -- a black channel.
                    if (frames->initial_pool_size > 0)
                        frames->initial_pool_size += 16;

                    if (av_hwframe_ctx_init(frames_ref) >= 0) {
                        ctx->hw_frames_ctx = frames_ref;
                    } else {
                        CASPAR_LOG(info)
                            << L"[av_producer] could not allocate a shader-readable decode pool; GPU-direct "
                               L"will decline";
                        av_buffer_unref(&frames_ref);
                    }
                }
            }

            return *p;
        }

        // D3D11VA cannot decode this stream -- H.264 High 10 and, on many GPUs,
        // VP9 are common cases. Returning AV_PIX_FMT_NONE here tells the decoder
        // that *no* offered format is acceptable, so it fails outright
        // ("decode_slice_header error") and the channel goes black with nothing
        // but an ffmpeg-level error to explain it.
        //
        // Accept the decoder's preferred software format instead: the stream then
        // decodes on the CPU, which is exactly what would have happened without
        // hardware acceleration configured.
        // Pick the first *software* format. The list can also contain other
        // hardware formats (vulkan, vaapi) which would be rejected with
        // "does not match the type of the provided device context", leaving us
        // no better off than returning NONE.
        for (p = pix_fmts; *p != AV_PIX_FMT_NONE; p++) {
            const auto* fmt_desc = av_pix_fmt_desc_get(*p);
            if (fmt_desc && !(fmt_desc->flags & AV_PIX_FMT_FLAG_HWACCEL)) {
                static std::once_flag once;
                std::call_once(once, [&] {
                    CASPAR_LOG(info) << L"[ffmpeg] hardware decoding unavailable for this stream; falling back to "
                                        L"software decoding.";
                });
                return *p;
            }
        }

        av_log(ctx, AV_LOG_ERROR, "Failed to get HW surface format.\n");
        return AV_PIX_FMT_NONE;
    }

    Decoder(const Decoder&)            = delete;
    Decoder& operator=(const Decoder&) = delete;

    AVStream*         st       = nullptr;
    int64_t           next_pts = AV_NOPTS_VALUE;
    std::atomic<bool> eof      = {false};

    std::queue<std::shared_ptr<AVPacket>> input;
    mutable boost::mutex                  input_mutex;
    boost::condition_variable             input_cond;
    // 4 pre-staged packets: reduces decoder input starvation between schedule()
    // iterations, especially at high bitrates (e.g. 12K ProRes).
    int                                   input_capacity = 4;

    std::queue<std::shared_ptr<AVFrame>> output;
    mutable boost::mutex                 output_mutex;
    boost::condition_variable            output_cond;
    // Will be raised to ctx->thread_count after avcodec_open2 for video streams
    // so the decode thread is never blocked by output backpressure when the full
    // frame-threading pool has frames ready (e.g. 16 threads, old cap was 8).
    int                                  output_capacity = 8;

    boost::thread             thread;
    // Set by the destructor before interrupting/joining. The decode loop tests
    // this instead of thread.interruption_requested(): the worker used to read
    // the `thread` member while that very member was still being assigned from
    // the boost::thread constructor's result -- a data race on the object the
    // lambda was being stored into. See
    // docs/CasparCG_HRC_Crash_Report_2026-06-17.md §9.1 fix 4a.
    std::atomic<bool>         abort_{false};
    std::atomic<bool>         flush_requested_{false};
    boost::mutex              flush_mutex_;
    boost::condition_variable flush_done_cond_;

  public:
    std::shared_ptr<AVCodecContext> ctx;
    // When HW decoding is active, pix_fmt on ctx becomes AV_PIX_FMT_D3D11 (a HW surface
    // format). The filter buffersrc needs a real CPU pixel format instead. This stores
    // the SW pixel format to use for both the buffersrc args and frame transfer.
    AVPixelFormat sw_pix_fmt = AV_PIX_FMT_NONE;

    // Colour properties as reported by decoded frames, for codecs that carry them
    // in the bitstream rather than the container. ProRes is the common case: its
    // codecpar says UNSPECIFIED, so the buffersrc gets configured "csp: unknown
    // range: unknown" and the first real frame trips FFmpeg's "Changing video
    // frame properties on the fly" warning -- the same shape of problem as
    // sw_pix_fmt being probed before the decoder had run. Learned here so a
    // filter rebuild can declare them up front instead.
    //
    // Written by the decode thread below, read by the run thread when it builds
    // the filter graph. Atomic because those are genuinely different threads.
    std::atomic<AVColorSpace> frame_colorspace{AVCOL_SPC_UNSPECIFIED};
    std::atomic<AVColorRange> frame_color_range{AVCOL_RANGE_UNSPECIFIED};
#if LIBAVUTIL_VERSION_MAJOR >= 60
    /// Whether the decoder DECLARED how its alpha relates to the colour values, which
    /// FFmpeg 8 is the first version able to say. Latched from the first frame that states
    /// it, exactly like the two above and for the same reason: the container often does not.
    std::atomic<int> frame_alpha_mode{AVALPHA_MODE_UNSPECIFIED};
#endif

    // Set if any decoded frame was actually flagged interlaced. The deinterlacer
    // is omitted for streams whose container declares them progressive (see the
    // filter spec), and containers do occasionally lie; this is the safety net,
    // consulted on every filter rebuild so a mis-declared file gets its
    // deinterlacer back rather than playing combed forever.
    std::atomic<bool> saw_interlaced_frame_{false};

#ifdef _WIN32
    // When true, D3D11 frames are kept as-is (no CPU transfer) and placed in hw_output.
    std::atomic<bool>                         gpu_direct_mode_{false};
    /// Armed by a flush (seek or loop wrap), cleared by the first frame that arrives. See the
    /// receive call: it selects AV_CODEC_RECEIVE_FRAME_FLAG_SYNCHRONOUS for exactly the
    /// frames a cue is waiting on, and nothing else.
    bool                                      sync_receive_ = false;
    // Set when the decoder emits an ordinary frame while it was asked to produce
    // hardware surfaces -- i.e. hardware decoding declined after the fact. The
    // producer watches this so it stops waiting for surfaces that never come.
    std::atomic<bool>                         saw_software_frame_{false};
    std::queue<std::shared_ptr<AVFrame>>      hw_output;
    mutable boost::mutex                      hw_output_mutex;
    boost::condition_variable                 hw_output_cond;
#endif

    /// DXGI adapter for the hardware decode device; -1 = default adapter.
    int decode_adapter_ = -1;
    /// The mixer's Vulkan device, when the mixer IS Vulkan, so an FFmpeg Vulkan compute
    /// decoder can allocate its frames there instead of on a device of its own. Null on
    /// the OpenGL mixer and on any build without Vulkan. Threaded like decode_adapter_
    /// rather than fetched from a global, for the same reason.
    void* vk_mixer_device_ = nullptr;

    Decoder() = default;

    /// `decode_adapter` is the DXGI adapter the hardware decoder should run on,
    /// or -1 for the default. It has to match the mixer's GPU or the decoded
    /// surfaces cannot be imported by the GPU-direct bridge -- see
    /// resolve_decode_adapter.
    explicit Decoder(AVStream* stream, int decode_adapter = -1, void* vk_mixer_device = nullptr)
        : st(stream)
        , decode_adapter_(decode_adapter)
        , vk_mixer_device_(vk_mixer_device)
    {
        const auto codec = get_decoder(stream->codecpar->codec_id, stream, decode_adapter);

        if (!codec) {
            FF_RET(AVERROR_DECODER_NOT_FOUND, "avcodec_find_decoder");
        }

        ctx = std::shared_ptr<AVCodecContext>(avcodec_alloc_context3(codec),
                                              [](AVCodecContext* ptr) { avcodec_free_context(&ptr); });

        if (!ctx) {
            FF_RET(AVERROR(ENOMEM), "avcodec_alloc_context3");
        }

        FF(avcodec_parameters_to_context(ctx.get(), stream->codecpar));

        if (stream->metadata != NULL) {
            auto entry = av_dict_get(stream->metadata, "alpha_mode", NULL, AV_DICT_MATCH_CASE);
            if (entry != NULL && entry->value != NULL && *entry->value == '1')
                ctx->pix_fmt = get_pix_fmt_with_alpha(ctx->pix_fmt);
        }

        // Auto-detect optimal thread count (0) rather than potentially starving the
        // mixer thread pool by overriding with arbitrarily high thread counts.
        int thread_count = 0;
        FF(av_opt_set_int(ctx.get(), "threads", thread_count, 0));

        ctx->pkt_timebase = stream->time_base;

        if (ctx->codec_type == AVMEDIA_TYPE_VIDEO) {
            ctx->framerate           = av_guess_frame_rate(nullptr, stream, nullptr);
            ctx->sample_aspect_ratio = av_guess_sample_aspect_ratio(nullptr, stream, nullptr);

            // Ask the *decoder* whether it can use D3D11VA, rather than guessing
            // from the codec ID.
            //
            // A codec ID does not identify a decoder: AV_CODEC_ID_VP9 resolves to
            // libvpx-vp9 in this build, an external-library wrapper with no
            // hardware support whatsoever. Attaching a hardware device context
            // and a get_format callback to it faulted inside the decode thread --
            // and because the project builds with /EHa, catch(...) swallowed the
            // access violation, so VP9 clips simply stopped producing frames with
            // "No diagnostic information available" and the channel sat black.
            //
            // avcodec_get_hw_config() answers the question properly and needs no
            // hardcoded codec list.
            // An ORDERED preference, still asking the decoder rather than guessing from a
            // codec list -- the comment above is why that matters.
            //
            // Vulkan first, but only when the mixer is Vulkan. FFmpeg 8 decodes ProRes,
            // ProRes RAW, FFV1 and DPX with compute shaders, and on the mixer's own device
            // those frames need no copy at all. Measured standalone on this hardware:
            // 64% less host CPU for 13% less decode throughput with the frames left on the
            // GPU, against 48%/78% once a readback is added -- so the copy is the whole
            // question, and sharing the device is how it is avoided.
            //
            // D3D11VA otherwise, unchanged: it is what h264/hevc/vp9/av1 use, it is what
            // the GPU-direct plane bridge is built on, and nothing here alters it.
            const AVCodecHWConfig* hw_cfg      = nullptr;
            bool                   want_vulkan = false;

            if (vk_mixer_device_) {
                for (int i = 0;; ++i) {
                    const auto* cfg = avcodec_get_hw_config(codec, i);
                    if (!cfg)
                        break;
                    if (cfg->device_type == AV_HWDEVICE_TYPE_VULKAN &&
                        (cfg->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX)) {
                        hw_cfg      = cfg;
                        want_vulkan = true;
                        break;
                    }
                }
            }

            if (!hw_cfg) {
                for (int i = 0;; ++i) {
                    const auto* cfg = avcodec_get_hw_config(codec, i);
                    if (!cfg)
                        break;
                    if (cfg->device_type == AV_HWDEVICE_TYPE_D3D11VA &&
                        (cfg->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX)) {
                        hw_cfg = cfg;
                        break;
                    }
                }
            }

            if (want_vulkan) {
                // Refuses and returns null rather than sharing the mixer's graphics queue;
                // falling through to D3D11VA (or software) is the correct outcome then.
                if (auto* vk_ctx = make_vulkan_hwdevice_from_mixer(vk_mixer_device_)) {
                    ctx->hw_device_ctx = vk_ctx;
                    ctx->get_format    = get_hw_format;
                    hw_cfg             = nullptr;   // handled; skip the D3D11 setup below
                } else {
                    want_vulkan = false;
                    hw_cfg      = nullptr;
                    for (int i = 0;; ++i) {
                        const auto* cfg = avcodec_get_hw_config(codec, i);
                        if (!cfg)
                            break;
                        if (cfg->device_type == AV_HWDEVICE_TYPE_D3D11VA &&
                            (cfg->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX)) {
                            hw_cfg = cfg;
                            break;
                        }
                    }
                }
            }

            if (hw_cfg) {
                // Put the decode device on the adapter the mixer lives on. A
                // null device string means adapter 0, which is only right when
                // the mixer happens to be there too.
                const std::string adapter_str =
                    decode_adapter_ >= 0 ? std::to_string(decode_adapter_) : std::string();

                AVBufferRef* hw_device_ctx = nullptr;
                if (av_hwdevice_ctx_create(&hw_device_ctx,
                                           AV_HWDEVICE_TYPE_D3D11VA,
                                           decode_adapter_ >= 0 ? adapter_str.c_str() : nullptr,
                                           nullptr,
                                           0) == 0) {
                    ctx->hw_device_ctx = hw_device_ctx;
                    ctx->get_format    = get_hw_format;
                }
            }
        } else if (ctx->codec_type == AVMEDIA_TYPE_AUDIO) {
        }

        FF(avcodec_open2(ctx.get(), codec, nullptr));

        // Resolve the format hardware-decoded frames will actually have, BEFORE
        // any decoding starts.
        //
        // ctx->sw_pix_fmt is only filled in during the decoder's first
        // get_format callback, i.e. once frames are already flowing -- so
        // reading it here always yielded AV_PIX_FMT_NONE and sw_pix_fmt stayed
        // unset. The filter graph is built from this value, so it was configured
        // for the *bitstream's* format (yuv420p) while the decoder went on to
        // emit NV12. For a passthrough graph ffmpeg only warns ("Changing video
        // frame properties on the fly"), but as soon as a real filter is in the
        // graph -- yadif, which auto-deinterlace inserts for every interlaced
        // clip -- it receives data in a layout it was configured against and the
        // process dies with an access violation. Interlaced H.264 with the
        // default settings crashed the server outright.
        //
        // avcodec_get_hw_frames_parameters() answers the same question up front,
        // so ask it now. The frames context it produces is also the one the
        // decoder will use, which lets us add SHADER_RESOURCE binding for the
        // GPU-direct path in the same place.
        if (ctx->hw_device_ctx) {
            // Only *read* the format here; do not take ownership of the pool.
            // Allocating it this early yielded a context that av_hwframe_ctx_init
            // would not accept with the extra bind flag, and ffmpeg then silently
            // built its own default pool -- leaving the surfaces unbindable for
            // shader access. The pool with SHADER_RESOURCE is created in
            // get_format instead, which is where it works.
            // Ask about the format this device actually produces. Probing for
            // AV_PIX_FMT_D3D11 against a Vulkan device simply fails, leaving sw_pix_fmt
            // unset -- harmless but pointless, and it hides the answer from the fallback
            // path that does want it.
            auto probe_fmt = AV_PIX_FMT_D3D11;
#if defined(ENABLE_VULKAN) && LIBAVUTIL_VERSION_MAJOR >= 60
            if (reinterpret_cast<AVHWDeviceContext*>(ctx->hw_device_ctx->data)->type == AV_HWDEVICE_TYPE_VULKAN)
                probe_fmt = AV_PIX_FMT_VULKAN;
#endif
            AVBufferRef* probe = nullptr;
            if (avcodec_get_hw_frames_parameters(ctx.get(), ctx->hw_device_ctx, probe_fmt, &probe) >= 0 && probe) {
                sw_pix_fmt = reinterpret_cast<AVHWFramesContext*>(probe->data)->sw_format;
                av_buffer_unref(&probe);
            }

            // Fall back to whatever the codec has resolved, if anything.
            if (sw_pix_fmt == AV_PIX_FMT_NONE && ctx->sw_pix_fmt != AV_PIX_FMT_NONE) {
                sw_pix_fmt = ctx->sw_pix_fmt;
            }
        }

        // For video with frame threading the codec resolves threads=0 to
        // hardware_concurrency (e.g. 16). Raise the output queue to match so
        // the decode thread is never blocked waiting for the filter to drain
        // when the entire thread pool finishes frames simultaneously.
        if (ctx->codec_type == AVMEDIA_TYPE_VIDEO && ctx->thread_count > output_capacity) {
            output_capacity = ctx->thread_count;
        }

        thread = boost::thread([=]() {
            while (!abort_.load(std::memory_order_relaxed)) {
                // Named so the catch below can say WHERE, which is the only reason the
                // prores_vulkan fault could be attributed at all: "Decoder thread
                // non-C++ exception" alone is true of the whole loop, and under /EHa an
                // access violation inside FFmpeg arrives here looking exactly like one
                // from our own code.
                const char* stage = "enter";
                try {
                    auto av_frame = alloc_frame();
                    stage     = "avcodec_receive_frame";
                    // AFTER A SEEK ONLY, bypass frame threading for the frames we are
                    // waiting on. `AV_CODEC_RECEIVE_FRAME_FLAG_SYNCHRONOUS` (lavc 62.22.101)
                    // makes the decoder "return the next frame as soon as possible ... may
                    // deliver frames earlier than the advertised AVCodecContext.delay", which
                    // is exactly the cue-latency problem: with threads=0 resolving to
                    // hardware_concurrency, a long-GOP decoder holds thread_count frames
                    // before emitting the first one, and every one of those is latency the
                    // operator waits through after a SEEK.
                    //
                    // NOT unconditionally. The flag defeats frame threading, which is what
                    // pays for steady-state throughput -- the whole reason `output_capacity`
                    // is raised to `thread_count` a few hundred lines up. So it is armed by a
                    // flush (i.e. a seek or a loop wrap) and disarmed by the first frame that
                    // actually arrives.
                    //
                    // WHAT THIS MEASURABLY FIXES: dropped frames at the LOOP WRAP on the
                    // Vulkan decode path. A wrap flushes the decoder, and the frame the
                    // channel needs is the next one -- so without this the channel starves
                    // for a frame or two every time round. Measured 2026-08-21 on a 3-second
                    // looping ProRes clip, 4 layers, 2 rounds, counting `fps` samples below
                    // nominal in the server's own diagnostics:
                    //
                    //     arm         with      without
                    //     vulkan         8           34      <- 4x, and the reason this exists
                    //     software       9           12      noise
                    //     cuda          14           12      noise, and a CONTROL: the CUDA
                    //                                        producer bypasses avcodec, so
                    //                                        this flag cannot reach it
                    //
                    // The Vulkan path is uniquely sensitive because its per-frame chain is
                    // longer -- a host wait on the decoder's semaphore plus a copy submitted
                    // through the device thread -- so the frame-threading delay is what
                    // pushed the post-wrap refill past the tick budget. Before this, that
                    // path dropped a frame every ~3 seconds and was the stated reason
                    // <vulkan-decode> could not be a default.
                    //
                    // NOT measured to improve CUE LATENCY, which is what it was adopted for:
                    // 78.7 vs 78.2 ms median over 39 interleaved seeks, every value pinned
                    // between 78 and 80 ms because the probe resolves only to a channel tick.
                    // So quote the loop-wrap result, not a cue-latency one.
                    //
                    // Safe on throughput: server CPU moved +4.1% one round and -4.6% the
                    // next -- a sign flip, so noise.
                    int ret = 0;
#if LIBAVCODEC_VERSION_INT >= AV_VERSION_INT(62, 22, 101)
                    if (sync_receive_) {
                        ret = avcodec_receive_frame_flags(ctx.get(), av_frame.get(),
                                                          AV_CODEC_RECEIVE_FRAME_FLAG_SYNCHRONOUS);
                        if (ret == 0)
                            sync_receive_ = false;
                    } else {
                        ret = avcodec_receive_frame(ctx.get(), av_frame.get());
                    }
#else
                    ret = avcodec_receive_frame(ctx.get(), av_frame.get());
#endif
                    stage     = "after receive_frame";

                    if (ret == AVERROR(EAGAIN)) {
                        std::shared_ptr<AVPacket> packet;
                        {
                            boost::unique_lock<boost::mutex> lock(input_mutex);
                            // Also wake on flush_requested_ so the flush is performed
                            // from this thread (avcodec_flush_buffers is not thread-safe
                            // with concurrent send/receive calls).
                            // abort_ is part of the predicate so teardown does not
                            // depend on a boost interruption point firing.
                            input_cond.wait(lock, [&]() {
                                return !input.empty() || flush_requested_.load() ||
                                       abort_.load(std::memory_order_relaxed);
                            });
                            if (abort_.load(std::memory_order_relaxed))
                                break;
                            if (flush_requested_.load()) {
                                // Perform the in-place flush from within the decode thread.
                                {
                                    boost::lock_guard<boost::mutex> out_lock(output_mutex);
                                    while (!output.empty())
                                        output.pop();
                                }
                                output_cond.notify_all();
#ifdef _WIN32
                                // The same, for the GPU-direct queue. It was left
                                // alone here, so a seek emptied the software frames
                                // and kept the hardware ones -- and every seek that
                                // matters is a loop point.
                                //
                                // What that looked like: with LOOP, playback reached
                                // the end and stopped dead, the position frozen a few
                                // frames short and the log repeating "Waiting for
                                // video frame...". The reader was being handed
                                // pre-seek surfaces whose timestamps sit past the
                                // point just seeked to, so it never accepted one and
                                // the position never moved again. On both mixers, and
                                // it also pinned decoder pool surfaces that the
                                // decoder needed back.
                                {
                                    boost::lock_guard<boost::mutex> hw_lock(hw_output_mutex);
                                    while (!hw_output.empty())
                                        hw_output.pop();
                                }
                                hw_output_cond.notify_all();
#endif
                                // Drop anything flush() cleared and a producer pushed back
                                // in before flush_requested_ was observed; those packets
                                // pre-date the seek.
                                while (!input.empty())
                                    input.pop();
                                avcodec_flush_buffers(ctx.get());
                                // The frame the operator is waiting for is the next one, so
                                // ask for it without the frame-threading delay. Disarms
                                // itself as soon as a frame arrives.
                                sync_receive_ = true;
                                next_pts = AV_NOPTS_VALUE;
                                eof      = false;
                                // Clear the flag and notify under flush_mutex_. Doing it
                                // without that lock let the notify slip between flush()'s
                                // predicate check and its wait, and the lost wakeup then
                                // cost the seek the full 500 ms timeout.
                                {
                                    boost::lock_guard<boost::mutex> flush_lock(flush_mutex_);
                                    flush_requested_ = false;
                                }
                                flush_done_cond_.notify_all();
                                continue;
                            }
                            packet = std::move(input.front());
                            input.pop();
                        }
                        FF(avcodec_send_packet(ctx.get(), packet.get()));
                    } else if (ret == AVERROR_EOF) {
                        avcodec_flush_buffers(ctx.get());
                        av_frame->pts = next_pts;
                        next_pts      = AV_NOPTS_VALUE;
                        eof           = true;

                        {
                            boost::unique_lock<boost::mutex> lock(output_mutex);
                            output_cond.wait(lock, [&]() {
                                return output.size() < output_capacity || flush_requested_.load() ||
                                       abort_.load(std::memory_order_relaxed);
                            });
                            if (!flush_requested_.load() && !abort_.load(std::memory_order_relaxed))
                                output.push(std::move(av_frame));
                        }
                    } else {
                        // Learn the colour properties the container did not state.
                        // Done for every decoded frame regardless of hardware or
                        // software path, since a hardware surface carries them too.
                        if (av_frame->colorspace != AVCOL_SPC_UNSPECIFIED &&
                            frame_colorspace.load(std::memory_order_relaxed) == AVCOL_SPC_UNSPECIFIED) {
                            frame_colorspace.store(static_cast<AVColorSpace>(av_frame->colorspace),
                                                   std::memory_order_relaxed);
                        }
                        if (av_frame->color_range != AVCOL_RANGE_UNSPECIFIED &&
                            frame_color_range.load(std::memory_order_relaxed) == AVCOL_RANGE_UNSPECIFIED) {
                            frame_color_range.store(static_cast<AVColorRange>(av_frame->color_range),
                                                    std::memory_order_relaxed);
                        }
#if LIBAVUTIL_VERSION_MAJOR >= 60
                        // Read PRE-filter, because this is the decoder's statement and the
                        // buffersrc is configured from it below. Only png, exr, jpegxl and
                        // Matroska set it in FFmpeg 8.1 -- prores and qtrle, which is most
                        // of the alpha content this fork sees, say nothing and fall back to
                        // the convention.
                        if (av_frame->alpha_mode != AVALPHA_MODE_UNSPECIFIED &&
                            frame_alpha_mode.load(std::memory_order_relaxed) == AVALPHA_MODE_UNSPECIFIED) {
                            frame_alpha_mode.store(av_frame->alpha_mode, std::memory_order_relaxed);
                        }
#endif

                        // Recorded, not judged: whether it matters depends on what the
                        // container claimed, which the filter graph decides and logs.
#if LIBAVCODEC_VERSION_MAJOR < 61
                        const bool frame_interlaced = av_frame->interlaced_frame != 0;
#else
                        const bool frame_interlaced = (av_frame->flags & AV_FRAME_FLAG_INTERLACED) != 0;
#endif
                        if (frame_interlaced)
                            saw_interlaced_frame_.store(true, std::memory_order_relaxed);

                        // A hardware surface is not a software frame, and there are two
                        // kinds now. Reading AV_PIX_FMT_VULKAN as "the decoder fell back to
                        // software" stood the path down on exactly the frames it exists for.
                        const bool hw_surface = av_frame->format == AV_PIX_FMT_D3D11
#if defined(ENABLE_VULKAN) && LIBAVUTIL_VERSION_MAJOR >= 60
                                                || av_frame->format == AV_PIX_FMT_VULKAN
#endif
                            ;

                        if (!hw_surface && gpu_direct_mode_.load())
                            saw_software_frame_.store(true, std::memory_order_relaxed);

                        // sw_pix_fmt is probed from the hardware frames context when the
                        // decoder is opened, before anyone knows whether hardware decoding
                        // will actually be used. When it declines -- H.264 High 10 and
                        // 4:4:4 both do -- the decoder emits its native software format
                        // while sw_pix_fmt still says NV12, and the buffersrc is then
                        // configured with NV12 for a 10-bit stream. FFmpeg only warns
                        // about that ("Changing video frame properties on the fly...")
                        // and copes, but it is a false declaration and it repeated on
                        // every filter rebuild. An ordinary frame is authoritative about
                        // its own layout, so take it.
                        if (!hw_surface && av_frame->format != AV_PIX_FMT_NONE &&
                            sw_pix_fmt != static_cast<AVPixelFormat>(av_frame->format)) {
                            const auto* was = av_get_pix_fmt_name(sw_pix_fmt);
                            const auto* now = av_get_pix_fmt_name(static_cast<AVPixelFormat>(av_frame->format));
                            CASPAR_LOG(info) << L"[ffmpeg] decoder software format is " << (now ? u16(now) : L"?")
                                             << L", not the " << (was ? u16(was) : L"unset")
                                             << L" advertised by the hardware frames context; correcting";
                            sw_pix_fmt = static_cast<AVPixelFormat>(av_frame->format);
                        }

                        // Handle HW frame transfer
                        if (hw_surface) {
                            // Resolve the actual SW pixel format from the HW frames context.
                            // ctx->sw_pix_fmt may not be set until after get_format is called
                            // during the first decode, so we update it here from the frame's
                            // hw_frames_ctx which always has the correct sw_format.
                            if (sw_pix_fmt == AV_PIX_FMT_NONE && av_frame->hw_frames_ctx) {
                                auto* frames_ctx = reinterpret_cast<AVHWFramesContext*>(av_frame->hw_frames_ctx->data);
                                sw_pix_fmt = frames_ctx->sw_format;
                            }
#ifdef _WIN32
                            if (gpu_direct_mode_.load()) {
                                // GPU-direct path: keep D3D11 frame, push to hw_output queue.
                                // The Impl thread will use d3d11_bridge to convert on GPU.
                                av_frame->pts = av_frame->best_effort_timestamp;
                                {
                                    boost::unique_lock<boost::mutex> lock(hw_output_mutex);
                                    // Same predicate as the software queue above,
                                    // abort_ included: this is where the thread
                                    // rests for most of its life in GPU-direct
                                    // mode, so it is the one that has to notice.
                                    hw_output_cond.wait(lock, [&]() {
                                        return hw_output.size() < output_capacity || flush_requested_.load() ||
                                               abort_.load(std::memory_order_relaxed);
                                    });
                                    if (!flush_requested_.load() && !abort_.load(std::memory_order_relaxed))
                                        hw_output.push(std::move(av_frame));
                                }
                                hw_output_cond.notify_all();
                                continue;
                            }
#endif
                            auto sw_frame = alloc_frame();
                            // Request the specific SW pixel format that was advertised to the filter's
                            // buffersrc (sw_pix_fmt, e.g. NV12). Width/height must also be set before
                            // the call so FFmpeg can allocate the destination CPU buffer correctly.
                            // Ask for the surface's OWN layout, taken from the frame's
                            // hw_frames_ctx. Leaving it unset lets
                            // av_hwframe_transfer_data pick, and it picks an 8-bit
                            // target for a 10-bit (P010) surface -- silently
                            // discarding the extra bits. Forcing a separately probed
                            // value is not safe either; the frame's own context is
                            // the authority.
                            sw_frame->format = AV_PIX_FMT_NONE;
                            if (av_frame->hw_frames_ctx) {
                                auto* fctx = reinterpret_cast<AVHWFramesContext*>(av_frame->hw_frames_ctx->data);
                                sw_frame->format = fctx->sw_format;
                            }
                            sw_frame->width  = av_frame->width;
                            sw_frame->height = av_frame->height;
                            int transfer_ret = av_hwframe_transfer_data(sw_frame.get(), av_frame.get(), 0);
                            if (transfer_ret < 0) {
                                char errbuf[AV_ERROR_MAX_STRING_SIZE];
                                av_strerror(transfer_ret, errbuf, AV_ERROR_MAX_STRING_SIZE);
                                CASPAR_LOG(error) << "Error transferring the data to system memory: " << errbuf;
                                continue;
                            }
                            // Copy all frame properties: pts, colorspace, color_range, side data, etc.
                            av_frame_copy_props(sw_frame.get(), av_frame.get());
                            av_frame = sw_frame;
                        }

                        FF_RET(ret, "avcodec_receive_frame");

                        // TODO: Maybe Fixed in:
                        // https://github.com/FFmpeg/FFmpeg/commit/33203a08e0a26598cb103508327a1dc184b27bc6
                        // NOTE This is a workaround for DVCPRO HD.
#if LIBAVCODEC_VERSION_MAJOR < 61
                        if (av_frame->width > 1024 && av_frame->interlaced_frame) {
                            av_frame->top_field_first = 1;
                        }
#else
                        if (av_frame->width > 1024 && (av_frame->flags & AV_FRAME_FLAG_INTERLACED)) {
                            av_frame->flags |= AV_FRAME_FLAG_TOP_FIELD_FIRST;
                        }
#endif

                        // TODO (fix) is this always best?
                        av_frame->pts = av_frame->best_effort_timestamp;

                        auto duration_pts = av_frame->duration;
                        if (duration_pts <= 0) {
                            if (ctx->codec_type == AVMEDIA_TYPE_VIDEO) {
#if LIBAVCODEC_VERSION_MAJOR < 62
                                const int ticks_per_frame = ctx->ticks_per_frame;
#else
                                // https://github.com/FFmpeg/FFmpeg/commit/e930b834a928546f9cbc937f6633709053448232#diff-115616f8a2b59cab3aac4e7f4c8c31e69e94e7fcfa339b9f65b0bf34308aa80fR682
                                const int ticks_per_frame =
                                    (ctx->codec_descriptor && (ctx->codec_descriptor->props & AV_CODEC_PROP_FIELDS))
                                        ? 2
                                        : 1;
#endif
                                const auto ticks = av_stream_get_parser(st) ? av_stream_get_parser(st)->repeat_pict + 1
                                                                            : ticks_per_frame;
                                duration_pts     = static_cast<int64_t>(AV_TIME_BASE) * ctx->framerate.den * ticks /
                                               ctx->framerate.num / ticks_per_frame;
                                duration_pts = av_rescale_q(duration_pts, {1, AV_TIME_BASE}, st->time_base);
                            } else if (ctx->codec_type == AVMEDIA_TYPE_AUDIO) {
                                duration_pts = av_rescale_q(av_frame->nb_samples, {1, ctx->sample_rate}, st->time_base);
                            }
                        }

                        if (duration_pts > 0) {
                            next_pts = av_frame->pts + duration_pts;
                        } else {
                            next_pts = AV_NOPTS_VALUE;
                        }

                        {
                            boost::unique_lock<boost::mutex> lock(output_mutex);
                            // Also wake on flush_requested_ so we don't deadlock if a
                            // flush arrives while the output queue is full.
                            output_cond.wait(lock, [&]() {
                                return output.size() < output_capacity || flush_requested_.load() ||
                                       abort_.load(std::memory_order_relaxed);
                            });
                            if (!flush_requested_.load() && !abort_.load(std::memory_order_relaxed))
                                output.push(std::move(av_frame));
                        }
                    }
                } catch (boost::thread_interrupted&) {
                    break;
                } catch (const std::exception& e) {
                    CASPAR_LOG(warning) << "Decoder thread exception (packet dropped) at " << stage << ": "
                                        << e.what();
                } catch (...) {
                    CASPAR_LOG(error) << "Decoder thread non-C++ exception at " << stage;
                    CASPAR_LOG_CURRENT_EXCEPTION();
                }
            }

            // Cleanup any dangling flush requests if thread exits
            if (flush_requested_.load()) {
                boost::lock_guard<boost::mutex> lock(flush_mutex_);
                flush_requested_ = false;
                flush_done_cond_.notify_all();
            }
        });
    }

    ~Decoder()
    {
        try {
            // Raise the flag before interrupting so the loop exits even if it is
            // between waits, and wake anything blocked on the queues.
            abort_.store(true, std::memory_order_relaxed);
            input_cond.notify_all();
            output_cond.notify_all();
#ifdef _WIN32
            hw_output_cond.notify_all();
#endif

            if (thread.joinable()) {
                thread.interrupt();
                thread.join();
            }
        } catch (boost::thread_interrupted&) {
            // Do nothing...
        }
    }

    // Flush the decoder in-place: clears queues, calls avcodec_flush_buffers from
    // within the decode thread (thread-safety requirement), and resets eof/next_pts.
    // The decode thread stays alive — for intra-only codecs (ProRes, NotchLC) the
    // very next packet produces a frame immediately, giving zero-stutter loop seeks.
    void flush()
    {
        // 1. Drop all pending input packets.
        {
            boost::lock_guard<boost::mutex> lock(input_mutex);
            while (!input.empty())
                input.pop();
        }
        // 2. Ask the decode thread to flush; wake it in case it is blocked waiting.
        flush_requested_ = true;
        eof              = false;
        input_cond.notify_all();
        output_cond.notify_all();
#ifdef _WIN32
        // Including the hardware queue, which this did not wake. In GPU-direct
        // mode the decode thread's usual resting place is the hw_output_cond
        // wait -- the queue runs full, because nothing pulls from it faster than
        // the decoder fills it -- and a flush that does not notify there is
        // simply not seen. flush() then spent its whole 500 ms timeout, logged
        // "Decoder flush timed out - continuing anyway", and cleared
        // flush_requested_ itself, so the flush never happened at all:
        // avcodec_flush_buffers was not called and hw_output kept its pre-seek
        // surfaces. The reader served those, so roughly a second of pre-seek
        // video played after every SEEK before the new position appeared.
        hw_output_cond.notify_all();
#endif
        // 3. Wait until the decode thread confirms the flush is done.
        boost::unique_lock<boost::mutex> lock(flush_mutex_);
        if (!flush_done_cond_.wait_for(lock, boost::chrono::milliseconds(500), [&]() { return !flush_requested_.load(); })) {
            CASPAR_LOG(warning) << "Decoder flush timed out - continuing anyway";
            flush_requested_ = false;
        }
    }

    bool want_packet() const
    {
        if (eof) {
            return false;
        }

        {
            boost::lock_guard<boost::mutex> lock(input_mutex);
            return input.size() < input_capacity;
        }
    }

    void push(std::shared_ptr<AVPacket> packet)
    {
        if (eof) {
            return;
        }

        {
            boost::lock_guard<boost::mutex> lock(input_mutex);
            input.push(std::move(packet));
        }

        input_cond.notify_all();
    }

    std::shared_ptr<AVFrame> pop()
    {
        std::shared_ptr<AVFrame> frame;

        {
            boost::lock_guard<boost::mutex> lock(output_mutex);

            if (!output.empty()) {
                frame = std::move(output.front());
                output.pop();
            }
        }

        if (frame) {
            output_cond.notify_all();
        } else if (eof) {
            frame = alloc_frame();
        }

        return frame;
    }

#ifdef _WIN32
    /// Whether the decoder has reached end of file.
    ///
    /// Public alongside pop_hw() because the GPU-direct reader needs both: the
    /// frames, and the fact that there will be no more. On the software path
    /// that second fact travels as a sentinel through the filter graph, which
    /// GPU-direct frames do not enter.
    bool at_eof() const { return eof.load(); }

    /// Whether a flush has been asked for but not yet carried out.
    ///
    /// The flush happens on the decode thread, so between a seek being requested
    /// and that thread acting on it the decoder still reports the end of file it
    /// reached before the seek. A reader that acts on eof during that window
    /// seeks again, and again.
    bool flush_pending() const { return flush_requested_.load(); }

    std::shared_ptr<AVFrame> pop_hw()
    {
        std::shared_ptr<AVFrame> frame;
        {
            boost::lock_guard<boost::mutex> lock(hw_output_mutex);
            if (!hw_output.empty()) {
                frame = std::move(hw_output.front());
                hw_output.pop();
            }
        }
        if (frame) {
            hw_output_cond.notify_all();
        }
        return frame;
    }
#endif
};

struct Filter
{
    std::shared_ptr<AVFilterGraph>  graph;
    AVFilterContext*                sink = nullptr;
    std::map<int, AVFilterContext*> sources;
    std::shared_ptr<AVFrame>        frame;
    bool                            eof = false;

    Filter() = default;

    Filter(std::string                    filter_spec,
           const Input&                   input,
           std::map<int, Decoder>&        streams,
           int64_t                        start_time,
           AVMediaType                    media_type,
           const core::video_format_desc& format_desc,
           int                            decode_adapter    = -1,
           void*                          vk_mixer_device   = nullptr)
    {
        // Whether bwdif ends up in the graph. The output format restriction below
        // needs to know, because bwdif's chroma handling is what makes interlaced
        // 4:2:0 delicate, and it must not be given a different chroma layout to
        // work on than it has always had.
        bool deinterlacing = false;

        if (media_type == AVMEDIA_TYPE_VIDEO) {
            if (filter_spec.empty()) {
                filter_spec = "null";
            }

            auto deint = u8(
                env::properties().get<std::wstring>(L"configuration.ffmpeg.producer.auto-deinterlace", L"interlaced"));

            // ── Skip the deinterlacer on declared-progressive streams ──────
            // With deint=interlaced (the default) bwdif is a pass-through on
            // progressive frames -- but it is still in the graph, so its format
            // constraints apply to everything. It has no semi-planar support, so a
            // hardware-decoded NV12 frame gets de-interleaved to yuv420p by
            // libswscale on every frame, for a filter that then does nothing.
            //
            // Measured on 12 layers of hardware-decoded 1080p25 H.264, three runs:
            // 3.85 -> 3.14, 3.84 -> 3.17 and 3.89 -> 3.17 CPU cores, about 18%.
            // NV12 also then reaches the mixer as two planes instead of three,
            // which is what the native semi-planar upload path was built for.
            //
            // Only for streams the container explicitly declares progressive --
            // "unknown" keeps the deinterlacer -- and only for deint=interlaced,
            // since deint=all means the caller wants every frame deinterlaced.
            // If any decoder has since seen a frame actually flagged interlaced,
            // the container lied and the deinterlacer goes back in.
            bool declared_progressive = false;
            bool observed_interlaced  = false;
            if (deint == "interlaced") {
                for (auto n = 0U; n < input->nb_streams; ++n) {
                    const auto* st = input->streams[n];
                    if (st->codecpar->codec_type != AVMEDIA_TYPE_VIDEO)
                        continue;
                    declared_progressive = st->codecpar->field_order == AV_FIELD_PROGRESSIVE;
                    const auto it        = streams.find(st->index);
                    if (it != streams.end())
                        observed_interlaced = it->second.saw_interlaced_frame_.load(std::memory_order_relaxed);
                    break;
                }
                if (declared_progressive && observed_interlaced) {
                    CASPAR_LOG(warning) << L"[ffmpeg] stream declares itself progressive but has delivered interlaced "
                                           L"frames; keeping the deinterlacer";
                }
            }

            const bool skip_deint = declared_progressive && !observed_interlaced;

            if (deint != "none" && !skip_deint) {
                deinterlacing = true;
                filter_spec += (boost::format(",bwdif=mode=send_field:parity=auto:deint=%s") % deint).str();
            }

            filter_spec += (boost::format(",fps=fps=%d/%d:start_time=%f") %
                            (format_desc.framerate.numerator() * format_desc.field_count) %
                            format_desc.framerate.denominator() % (static_cast<double>(start_time) / AV_TIME_BASE))
                               .str();
        } else if (media_type == AVMEDIA_TYPE_AUDIO) {
            if (filter_spec.empty()) {
                filter_spec = "anull";
            }

            // Find first audio stream to get a time_base for the first_pts calculation
            AVRational tb = {1, format_desc.audio_sample_rate};
            for (auto n = 0U; n < input->nb_streams; ++n) {
                const auto st             = input->streams[n];
                const auto codec_channels = st->codecpar->ch_layout.nb_channels;
                if (st->codecpar->codec_type == AVMEDIA_TYPE_AUDIO && codec_channels > 0) {
                    tb = {1, st->codecpar->sample_rate};
                    break;
                }
            }
            filter_spec += (boost::format(",aresample=async=1000:first_pts=%d:min_comp=0.01:osr=%d,"
                                          "asetnsamples=n=1024:p=0") %
                            av_rescale_q(start_time, TIME_BASE_Q, tb) % format_desc.audio_sample_rate)
                               .str();
        }

        AVFilterInOut* outputs = nullptr;
        AVFilterInOut* inputs  = nullptr;

        CASPAR_SCOPE_EXIT
        {
            avfilter_inout_free(&inputs);
            avfilter_inout_free(&outputs);
        };

        int video_input_count = 0;
        int audio_input_count = 0;
        {
            auto graph2 = avfilter_graph_alloc();
            if (!graph2) {
                FF_RET(AVERROR(ENOMEM), "avfilter_graph_alloc");
            }

            CASPAR_SCOPE_EXIT
            {
                avfilter_graph_free(&graph2);
                avfilter_inout_free(&inputs);
                avfilter_inout_free(&outputs);
            };

            FF(avfilter_graph_parse2(graph2, filter_spec.c_str(), &inputs, &outputs));

            for (auto cur = inputs; cur; cur = cur->next) {
                const auto type = avfilter_pad_get_type(cur->filter_ctx->input_pads, cur->pad_idx);
                if (type == AVMEDIA_TYPE_VIDEO) {
                    video_input_count += 1;
                } else if (type == AVMEDIA_TYPE_AUDIO) {
                    audio_input_count += 1;
                }
            }
        }

        std::vector<AVStream*> av_streams;
        for (auto n = 0U; n < input->nb_streams; ++n) {
            const auto st = input->streams[n];

            const auto codec_channels = st->codecpar->ch_layout.nb_channels;
            if (st->codecpar->codec_type == AVMEDIA_TYPE_AUDIO && codec_channels == 0) {
                continue;
            }

            auto disposition = st->disposition;
            if (!disposition || disposition == AV_DISPOSITION_DEFAULT) {
                av_streams.push_back(st);
            }
        }

        if (audio_input_count == 1) {
            auto count = std::count_if(av_streams.begin(), av_streams.end(), [](auto s) {
                return s->codecpar->codec_type == AVMEDIA_TYPE_AUDIO;
            });

            // TODO (fix) Use some form of stream meta data to do this.
            // https://github.com/CasparCG/server/issues/833
            if (count > 1) {
                filter_spec = (boost::format("amerge=inputs=%d,") % count).str() + filter_spec;
            }
        }

        if (video_input_count == 1) {
            std::stable_sort(av_streams.begin(), av_streams.end(), [](auto lhs, auto rhs) {
                return lhs->codecpar->codec_type == AVMEDIA_TYPE_VIDEO && lhs->codecpar->height > rhs->codecpar->height;
            });

            std::vector<AVStream*> video_av_streams;
            std::copy_if(av_streams.begin(), av_streams.end(), std::back_inserter(video_av_streams), [](auto s) {
                return s->codecpar->codec_type == AVMEDIA_TYPE_VIDEO;
            });

            // TODO (fix) Use some form of stream meta data to do this.
            // https://github.com/CasparCG/server/issues/832
            if (video_av_streams.size() >= 2 &&
                video_av_streams[0]->codecpar->height == video_av_streams[1]->codecpar->height) {
                filter_spec = "alphamerge," + filter_spec;
            }
        }

        graph = std::shared_ptr<AVFilterGraph>(avfilter_graph_alloc(),
                                               [](AVFilterGraph* ptr) { avfilter_graph_free(&ptr); });

        if (!graph) {
            FF_RET(AVERROR(ENOMEM), "avfilter_graph_alloc");
        }

        FF(avfilter_graph_parse2(graph.get(), filter_spec.c_str(), &inputs, &outputs));

        auto filter_src_fmt = AV_PIX_FMT_NONE;

        // inputs
        {
            for (auto cur = inputs; cur; cur = cur->next) {
                const auto type = avfilter_pad_get_type(cur->filter_ctx->input_pads, cur->pad_idx);
                if (type != AVMEDIA_TYPE_VIDEO && type != AVMEDIA_TYPE_AUDIO) {
                    CASPAR_THROW_EXCEPTION(ffmpeg_error_t() << boost::errinfo_errno(EINVAL)
                                                            << msg_info_t("only video and audio filters supported"));
                }

                unsigned index = 0;

                // TODO find stream based on link name
                while (true) {
                    if (index == av_streams.size()) {
                        graph = nullptr;
                        return;
                    }
                    if (av_streams.at(index)->codecpar->codec_type == type &&
                        sources.find(static_cast<int>(index)) == sources.end()) {
                        break;
                    }
                    index++;
                }

                index = av_streams.at(index)->index;

                auto it = streams.find(index);
                if (it == streams.end()) {
                    it = streams
                             .emplace(std::piecewise_construct,
                                      std::forward_as_tuple(index),
                                      std::forward_as_tuple(input->streams[index], decode_adapter, vk_mixer_device))
                             .first;
                }

                auto st = it->second.ctx;

                if (st->codec_type == AVMEDIA_TYPE_VIDEO) {
                    // If the decoder uses HW acceleration, ctx->pix_fmt is a HW surface format
                    // (e.g. AV_PIX_FMT_D3D11). We must configure the buffersrc with the real
                    // CPU pixel format that the decoder will produce after the HW->SW transfer.
                    // sw_pix_fmt is the resolved CPU format (e.g. NV12 for 8-bit, P010 for 10-bit).
                    const auto src_fmt = (it->second.sw_pix_fmt != AV_PIX_FMT_NONE)
                                             ? it->second.sw_pix_fmt
                                             : st->pix_fmt;
                    // "pix_fmt=-1" here means neither the decoder nor the stream has
                    // resolved a format yet; the buffersrc rejects it and the whole
                    // producer thread dies with an exception carrying no message.
                    // Say what we saw before that happens.
                    if (src_fmt == AV_PIX_FMT_NONE) {
                        CASPAR_LOG(warning) << L"[ffmpeg] cannot configure the video filter input: neither the "
                                               L"decoder nor the stream reports a pixel format yet (decoder "
                                            << u16(st->codec ? st->codec->name : "?") << L").";
                    }

                    // Remembered for the buffersink's format list below. Note this
                    // is the *stream's* declared format, not src_fmt: src_fmt comes
                    // from sw_pix_fmt, which is probed from the hardware frames
                    // context at open and stays at NV12 even when the decoder then
                    // declines hardware decoding and emits its native software
                    // format. Restricting against NV12 permits an 8-bit 4:2:0
                    // result, which is exactly the loss being prevented. codecpar
                    // describes what the content is and is not affected by how it
                    // gets decoded.
                    const auto declared = static_cast<AVPixelFormat>(input->streams[index]->codecpar->format);
                    filter_src_fmt      = declared != AV_PIX_FMT_NONE ? declared : src_fmt;

                    auto args = (boost::format("video_size=%dx%d:pix_fmt=%d:time_base=%d/%d") % st->width % st->height %
                                 src_fmt % st->pkt_timebase.num % st->pkt_timebase.den)
                                    .str();
                    auto name = (boost::format("in_%d") % index).str();

                    if (st->sample_aspect_ratio.num > 0 && st->sample_aspect_ratio.den > 0) {
                        args +=
                            (boost::format(":sar=%d/%d") % st->sample_aspect_ratio.num % st->sample_aspect_ratio.den)
                                .str();
                    }

                    if (st->framerate.num > 0 && st->framerate.den > 0) {
                        args += (boost::format(":frame_rate=%d/%d") % st->framerate.num % st->framerate.den).str();
                    }

                    AVFilterContext* source = nullptr;
                    FF(avfilter_graph_create_filter(
                        &source, avfilter_get_by_name("buffer"), name.c_str(), args.c_str(), nullptr, graph.get()));

                    // Set colorspace and color_range on the buffersrc so FFmpeg's filter
                    // graph knows the incoming frame properties upfront.  Without this the
                    // filter context starts as "csp: unknown range: unknown" and emits a
                    // warning on the first frame that carries proper metadata.
                    // Where the container says nothing, use what decoded frames
                    // reported. ProRes and friends carry colour in the bitstream, so
                    // codecpar is UNSPECIFIED and the buffersrc would start as
                    // "csp: unknown range: unknown" -- and then every real frame
                    // disagrees with it, which is what FFmpeg's "Changing video frame
                    // properties on the fly" warning was reporting. The decoder has
                    // seen frames by the time the graph is rebuilt, so by then this is
                    // known; on the very first build it is not, and the buffersrc is
                    // configured exactly as before.
                    auto src_csp = st->colorspace;
                    auto src_rng = st->color_range;
                    if (src_csp == AVCOL_SPC_UNSPECIFIED)
                        src_csp = it->second.frame_colorspace.load(std::memory_order_relaxed);
                    if (src_rng == AVCOL_RANGE_UNSPECIFIED)
                        src_rng = it->second.frame_color_range.load(std::memory_order_relaxed);

                    // Declared on the buffersrc so the graph carries it: AVFilterLink has an
                    // alpha_mode of its own, and a link left unspecified hands the sink
                    // frames that no longer say what the decoder said.
                    int src_alpha = AVALPHA_MODE_UNSPECIFIED;
#if LIBAVUTIL_VERSION_MAJOR >= 60
                    src_alpha = it->second.frame_alpha_mode.load(std::memory_order_relaxed);
#endif

                    if (src_csp != AVCOL_SPC_UNSPECIFIED || src_rng != AVCOL_RANGE_UNSPECIFIED ||
                        src_alpha != AVALPHA_MODE_UNSPECIFIED) {
                        AVBufferSrcParameters* par = av_buffersrc_parameters_alloc();
                        if (par) {
                            par->color_space = src_csp;
                            par->color_range = src_rng;
#if LIBAVUTIL_VERSION_MAJOR >= 60
                            par->alpha_mode = static_cast<AVAlphaMode>(src_alpha);
#endif
                            av_buffersrc_parameters_set(source, par);
                            av_free(par);
                        }
                    }

                    FF(avfilter_link(source, 0, cur->filter_ctx, cur->pad_idx));
                    sources.emplace(index, source);
                } else if (st->codec_type == AVMEDIA_TYPE_AUDIO) {
                    char channel_layout[128];
                    FF(av_channel_layout_describe(&st->ch_layout, channel_layout, sizeof(channel_layout)));

                    auto args = (boost::format("time_base=%d/%d:sample_rate=%d:sample_fmt=%s:channel_layout=%#x") %
                                 st->pkt_timebase.num % st->pkt_timebase.den % st->sample_rate %
                                 av_get_sample_fmt_name(st->sample_fmt) % channel_layout)
                                    .str();
                    auto name = (boost::format("in_%d") % index).str();

                    AVFilterContext* source = nullptr;
                    FF(avfilter_graph_create_filter(
                        &source, avfilter_get_by_name("abuffer"), name.c_str(), args.c_str(), nullptr, graph.get()));
                    FF(avfilter_link(source, 0, cur->filter_ctx, cur->pad_idx));
                    sources.emplace(index, source);
                } else {
                    CASPAR_THROW_EXCEPTION(ffmpeg_error_t() << boost::errinfo_errno(EINVAL)
                                                            << msg_info_t("invalid filter input media type"));
                }
            }
        }

        if (media_type == AVMEDIA_TYPE_VIDEO) {
            sink = FFMEM(avfilter_graph_alloc_filter(graph.get(), avfilter_get_by_name("buffersink"), "out"));

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4245)
#endif
            const AVPixelFormat pix_fmts[] = {AV_PIX_FMT_BGRA,
                                              AV_PIX_FMT_RGBA,
                                              // RGB24 and BGR24 are deliberately not offered. Vulkan
                                              // implementations are not required to support
                                              // 3-component 8-bit formats as sampled images, and this
                                              // one does not: the VK mixer throws as soon as a frame
                                              // of rgb24 reaches it, which an FFV1 RGB clip made it
                                              // do the moment a CPU consumer was attached. OpenGL
                                              // renders them correctly, so this is a parity floor
                                              // rather than a bug -- supporting them would mean
                                              // expanding to 4 components during upload.
                                              //
                                              // ARGB and ABGR are also not offered. The
                                              // mixer's shaders have cases for them
                                              // (pixel_format 3 and 4) but the swizzles are wrong:
                                              // a QuickTime RLE clip, whose decoder emits argb,
                                              // renders at 6.4 dB against a reference where every
                                              // other format scores 38 dB or better. Nothing
                                              // noticed because the filter graph always converted
                                              // packed alpha formats to planar before the mixer saw
                                              // them, so the path was never exercised. Excluding
                                              // them here keeps negotiation on a format known to
                                              // render correctly; fixing the shaders is a separate
                                              // job needing a parity test on both backends.
                                              // mixer's shaders have cases for them
                                              // (pixel_format 3 and 4) but the swizzles are wrong:
                                              // a QuickTime RLE clip, whose decoder emits argb,
                                              // renders at 6.4 dB against a reference where every
                                              // other format scores 38 dB or better. Nothing
                                              // noticed because the filter graph always converted
                                              // packed alpha formats to planar before the mixer saw
                                              // them, so the path was never exercised. Excluding
                                              // them here keeps negotiation on a format known to
                                              // render correctly; fixing the shaders is a separate
                                              // job needing a parity test on both backends.
                                              AV_PIX_FMT_YUV444P,
                                              AV_PIX_FMT_YUV444P10,
                                              AV_PIX_FMT_YUV444P12,
                                              AV_PIX_FMT_YUV422P,
                                              AV_PIX_FMT_YUV422P10,
                                              AV_PIX_FMT_YUV422P12,
                                              AV_PIX_FMT_YUV420P,
                                              AV_PIX_FMT_YUV420P10,
                                              AV_PIX_FMT_YUV420P12,
                                              AV_PIX_FMT_YUV410P,
                                              AV_PIX_FMT_YUVA444P,
                                              AV_PIX_FMT_YUVA422P,
                                              AV_PIX_FMT_YUVA420P,
                                              AV_PIX_FMT_YUVA444P10,
                                              AV_PIX_FMT_YUVA422P10,
                                              AV_PIX_FMT_YUVA420P10,
                                              AV_PIX_FMT_YUVA444P12,
                                              AV_PIX_FMT_YUVA422P12,
                                              // bwdif does not support 12-bit YUVA and promotes it to
                                              // 16-bit. Without these entries the only formats a
                                              // ProRes 4444 clip could negotiate were RGB, which cost
                                              // a full-frame YUV->RGB conversion on the host for no
                                              // gain -- same byte count, same picture.
                                              AV_PIX_FMT_YUVA420P16,
                                              AV_PIX_FMT_YUVA422P16,
                                              AV_PIX_FMT_YUVA444P16,
                                              AV_PIX_FMT_RGBA64LE,
                                              AV_PIX_FMT_BGRA64LE,
                                              AV_PIX_FMT_UYVY422,
                                              AV_PIX_FMT_NV12,
                                              AV_PIX_FMT_P010LE,
                                              // bwdif needs planar rgb
                                              AV_PIX_FMT_GBRP,
                                              AV_PIX_FMT_GBRP10,
                                              AV_PIX_FMT_GBRP12,
                                              AV_PIX_FMT_GBRP16,
                                              AV_PIX_FMT_GBRAP,
                                              AV_PIX_FMT_GBRAP16,
                                              AV_PIX_FMT_NONE};

            // ── Do not offer the sink a format that would lose picture ────
            // Offering this whole list let negotiation choose freely, and what it
            // chose was driven by the *filters'* format preferences rather than by
            // the source. `bwdif` is inserted into every video graph (it only acts
            // on frames flagged interlaced, but its format constraints apply to
            // all of them) and lists yuv420p first, so measured at the mixer:
            //
            //   H.264 High 10   yuv420p10le -> yuv420p   truncated to 8-bit
            //   H.264 High 4:4:4  yuv444p   -> yuv420p   chroma subsampled
            //   ProRes 4444    yuva444p12le -> gbrap16le YUV->RGB on the CPU
            //
            // The first two are silent quality losses on every such clip; they are
            // also why GPU-direct decode measured *better* than the software path
            // for 10-bit content, which had been recorded as an oddity of the
            // GPU path rather than a defect in this one.
            //
            // So restrict the offer to formats that cannot lose anything relative
            // to what the decoder produces: at least its bit depth, at least its
            // chroma resolution, and alpha if it has alpha. Negotiation is then
            // free to pick any of them and cannot pick a lossy one. Deinterlacing
            // is untouched -- bwdif stays in the graph for all content.
            std::vector<AVPixelFormat> allowed;
            const auto*                src_desc =
                filter_src_fmt != AV_PIX_FMT_NONE ? av_pix_fmt_desc_get(filter_src_fmt) : nullptr;
            if (src_desc != nullptr) {
                const int  src_depth  = src_desc->comp[0].depth;
                const bool src_alpha  = (src_desc->flags & AV_PIX_FMT_FLAG_ALPHA) != 0;
                for (auto p = pix_fmts; *p != AV_PIX_FMT_NONE; ++p) {
                    const auto* d = av_pix_fmt_desc_get(*p);
                    if (d == nullptr)
                        continue;
                    if (d->comp[0].depth < src_depth)
                        continue;
                    // Smaller log2_chroma means denser chroma. Never coarsen.
                    if (d->log2_chroma_w > src_desc->log2_chroma_w ||
                        d->log2_chroma_h > src_desc->log2_chroma_h)
                        continue;
                    if (src_alpha && (d->flags & AV_PIX_FMT_FLAG_ALPHA) == 0)
                        continue;
                    allowed.push_back(*p);
                }
            }

            // ── Prefer P010 for 10-bit 4:2:0 ─────────────────────────────────
            // Both yuv420p10le and p010le are lossless relative to a 10-bit
            // source, so the restriction above accepts either -- but they are not
            // equally accurate once the mixer samples them, and the difference is
            // in the *filtering*, not the arithmetic.
            //
            // yuv420p10le arrives as three planes of codes 0..1023 in 16-bit
            // words, so it is declared bit10 and the shader multiplies by a
            // precision factor of 64. The data therefore occupies only the bottom
            // 1/64 of the texture's UNORM range, and chroma is upsampled by the
            // texture unit before that multiply -- so whatever the filter rounds
            // away is amplified 64-fold. P010 carries the same codes high-aligned,
            // uses the full range, and needs no multiply.
            //
            // Measured on a static lossless 10-bit ramp, which should render
            // perfectly smooth: P010 gives roughness 0.220 against 0.246 and 184
            // backward steps in a monotone ramp against 229. And GPU-direct decode,
            // which hands over P010 by construction, is byte-identical to this path
            // once both take the same route -- against 69.64 dB before, which is
            // what led here.
            //
            // Offering p010le *first* does nothing -- negotiation is driven by the
            // filters' own preferences and ignores the sink list's order, measured.
            // So offer it as the only choice for this case and let libavfilter
            // insert the conversion, which it does.
            //
            // Progressive content only, and that restriction is not caution: with
            // bwdif in the graph, requiring p010le at the sink changes the chroma
            // layout the deinterlacer works on, and the deinterlaced picture then
            // moves by far more than this is trying to fix -- measured at 53.4 dB
            // with differences up to 83 code values on a 10-bit interlaced clip,
            // against the ±1 this addresses. Interlaced 4:2:0 chroma is delicate
            // and bwdif keeps the planar layout it has always had. GPU-direct
            // declines interlaced content anyway, so nothing is lost.
            if (src_desc != nullptr && !allowed.empty() && !deinterlacing && src_desc->comp[0].depth == 10 &&
                src_desc->log2_chroma_w == 1 && src_desc->log2_chroma_h == 1 &&
                (src_desc->flags & AV_PIX_FMT_FLAG_ALPHA) == 0 &&
                std::find(allowed.begin(), allowed.end(), AV_PIX_FMT_P010LE) != allowed.end()) {
                allowed.assign(1, AV_PIX_FMT_P010LE);
                CASPAR_LOG(info) << L"[ffmpeg] 10-bit 4:2:0 progressive source: requiring p010le, which the mixer "
                                    L"samples at full precision (yuv420p10le costs ~6 bits of chroma filtering "
                                    L"precision).";
            }

            // An empty or unusable restriction must never be worse than the old
            // behaviour, so fall back to the full list and say why.
            if (allowed.empty()) {
                CASPAR_LOG(info) << L"[ffmpeg] filter output unrestricted: source format "
                                 << (src_desc != nullptr ? u16(src_desc->name) : L"unresolved at graph build")
                                 << L" -- conversion is permitted and may lose precision";
#if LIBAVUTIL_VERSION_MAJOR >= 60 // FFmpeg 8
                FF(av_opt_set_array(sink,
                                    "pixel_formats",
                                    AV_OPT_SEARCH_CHILDREN | AV_OPT_ARRAY_REPLACE,
                                    0,
                                    FF_ARRAY_ELEMS(pix_fmts) - 1,
                                    AV_OPT_TYPE_PIXEL_FMT,
                                    pix_fmts));
#else
                FF(av_opt_set_int_list(sink, "pix_fmts", pix_fmts, -1, AV_OPT_SEARCH_CHILDREN));
#endif
            } else {
                CASPAR_LOG(info) << L"[ffmpeg] filter output restricted to " << allowed.size()
                                 << L" lossless format(s) for source " << u16(src_desc->name);
#if LIBAVUTIL_VERSION_MAJOR >= 60 // FFmpeg 8
                // The array form takes a COUNT, so it is read before the AV_PIX_FMT_NONE
                // terminator that the pre-8 int-list form needs is appended -- appending first
                // would offer the sink AV_PIX_FMT_NONE as if it were a real format.
                FF(av_opt_set_array(sink,
                                    "pixel_formats",
                                    AV_OPT_SEARCH_CHILDREN | AV_OPT_ARRAY_REPLACE,
                                    0,
                                    static_cast<unsigned int>(allowed.size()),
                                    AV_OPT_TYPE_PIXEL_FMT,
                                    allowed.data()));
#else
                allowed.push_back(AV_PIX_FMT_NONE);
                FF(av_opt_set_int_list(sink, "pix_fmts", allowed.data(), AV_PIX_FMT_NONE, AV_OPT_SEARCH_CHILDREN));
#endif
            }
#ifdef _MSC_VER
#pragma warning(pop)
#endif
        } else if (media_type == AVMEDIA_TYPE_AUDIO) {
            sink = FFMEM(avfilter_graph_alloc_filter(graph.get(), avfilter_get_by_name("abuffersink"), "out"));

            const AVSampleFormat sample_fmts[]  = {AV_SAMPLE_FMT_S32, AV_SAMPLE_FMT_NONE};
            const int            sample_rates[] = {format_desc.audio_sample_rate, -1};

            FF(av_opt_set_int(sink, "all_channel_counts", 1, AV_OPT_SEARCH_CHILDREN));

#if LIBAVUTIL_VERSION_MAJOR >= 60 // FFmpeg 8
            FF(av_opt_set_array(sink,
                                "sample_formats",
                                AV_OPT_SEARCH_CHILDREN | AV_OPT_ARRAY_REPLACE,
                                0,
                                FF_ARRAY_ELEMS(sample_fmts) - 1,
                                AV_OPT_TYPE_SAMPLE_FMT,
                                sample_fmts));
            FF(av_opt_set_array(sink,
                                "samplerates",
                                AV_OPT_SEARCH_CHILDREN | AV_OPT_ARRAY_REPLACE,
                                0,
                                FF_ARRAY_ELEMS(sample_rates) - 1,
                                AV_OPT_TYPE_INT,
                                sample_rates));
#else
            FF(av_opt_set_int_list(sink, "sample_fmts", sample_fmts, -1, AV_OPT_SEARCH_CHILDREN));
            FF(av_opt_set_int_list(sink, "sample_rates", sample_rates, -1, AV_OPT_SEARCH_CHILDREN));
#endif
        } else {
            CASPAR_THROW_EXCEPTION(ffmpeg_error_t()
                                   << boost::errinfo_errno(EINVAL) << msg_info_t("invalid output media type"));
        }

        FF(avfilter_init_str(sink, nullptr));

        // output
        {
            const auto cur = outputs;

            if (!cur || cur->next) {
                CASPAR_THROW_EXCEPTION(ffmpeg_error_t() << boost::errinfo_errno(EINVAL)
                                                        << msg_info_t("invalid filter graph output count"));
            }

            if (avfilter_pad_get_type(cur->filter_ctx->output_pads, cur->pad_idx) != media_type) {
                CASPAR_THROW_EXCEPTION(ffmpeg_error_t() << boost::errinfo_errno(EINVAL)
                                                        << msg_info_t("invalid filter output media type"));
            }

            FF(avfilter_link(cur->filter_ctx, cur->pad_idx, sink, 0));
        }

        FF(avfilter_graph_config(graph.get(), nullptr));

        CASPAR_LOG(debug) << avfilter_graph_dump(graph.get(), nullptr);
    }

    bool operator()(int nb_samples = -1)
    {
        if (frame || eof) {
            return false;
        }

        if (!sink || sources.empty()) {
            eof   = true;
            frame = nullptr;
            return true;
        }

        auto av_frame = alloc_frame();
        auto ret      = nb_samples >= 0 ? av_buffersink_get_samples(sink, av_frame.get(), nb_samples)
                                        : av_buffersink_get_frame(sink, av_frame.get());

        if (ret == AVERROR(EAGAIN)) {
            return false;
        }
        if (ret == AVERROR_EOF) {
            eof   = true;
            frame = nullptr;
            return true;
        }
        FF_RET(ret, "av_buffersink_get_frame");
        frame = std::move(av_frame);
        return true;
    }
};

struct AVProducer::Impl
{
    caspar::core::monitor::state state_;
    mutable boost::mutex         state_mutex_;

    spl::shared_ptr<diagnostics::graph> graph_;

    const std::shared_ptr<core::frame_factory> frame_factory_;
    const core::video_format_desc              format_desc_;
    const AVRational                           format_tb_;
    const std::string                          name_;
    const std::string                          path_;

    Input                  input_;
    std::map<int, Decoder> decoders_;
    Filter                 video_filter_;
    Filter                 audio_filter_;

    /// DXGI adapter the hardware decoder runs on, resolved once from the mixer's
    /// GPU. -1 = default adapter. See resolve_decode_adapter.
    const int decode_adapter_ = -1;
    /// The mixer's Vulkan device when the mixer is Vulkan, else null. Used only to let an
    /// FFmpeg Vulkan compute decoder allocate on it; see vulkan_hwdevice.h.
    void* const                                                  vk_mixer_device_ = nullptr;
    std::unique_ptr<accelerator::vulkan::av_vulkan_importer>     vk_importer_;

    std::map<int, std::vector<AVFilterContext*>> sources_;

    // Stream-level color metadata from the container/decoder, used as fallback
    // when individual decoded frames have UNSPECIFIED colorspace/transfer.
    AVColorSpace                     stream_color_space_ = AVCOL_SPC_UNSPECIFIED;
    AVColorTransferCharacteristic    stream_color_trc_   = AVCOL_TRC_UNSPECIFIED;
    AVChromaLocation                 stream_chroma_loc_  = AVCHROMA_LOC_UNSPECIFIED;

    //: SAY WHAT THIS SOURCE DECLARED, ONCE. The producer resolves a colour space and transfer
    //: for every frame and reported neither, so "the file/stream we ingest is tagged PQ
    //: BT.2020 and we saw that" was not an assertable statement -- only inferrable from the
    //: picture, which cannot separate a correct reading from a lucky one. The DeckLink
    //: producer grew the same line for the same reason; this is its FFmpeg counterpart, and it
    //: is what `cli.py signalling --stream` reads.
    bool logged_colour_ = false;

    //: HDR10 STATIC METADATA THE SOURCE DECLARES: ST 2086 mastering display volume and
    //: CTA-861.3 content light level, decoded from the bitstream's SEI.
    //:
    //: REPORTED, NOT PROPAGATED, and that is a decision rather than an omission. These
    //: describe the display a SOURCE was graded on. A channel's output is a composite of
    //: however many layers are on it, and there is no single mastering display for a
    //: composite -- picking one layer's would be inventing a claim about a picture nobody
    //: graded. So the values are surfaced here, on the producer, where they are true, and the
    //: OUTPUT's `<hdr-metadata>` stays what an operator declares about their own programme.
    //:
    //: Not on every frame either: the decoder attaches them once it has seen the SEI, so this
    //: keeps looking until it finds them rather than sampling the first frame -- the mistake
    //: that cost a day on the DeckLink side.
    std::atomic<bool> hdr10_seen_{false};
    double            hdr10_max_dml_  = 0.0;
    double            hdr10_min_dml_  = 0.0;
    int               hdr10_max_cll_  = 0;
    int               hdr10_max_fall_ = 0;

    void note_hdr10(const std::shared_ptr<AVFrame>& video)
    {
        if (hdr10_seen_.load(std::memory_order_relaxed) || !video) {
            return;
        }

        const auto* mdm_sd = av_frame_get_side_data(video.get(), AV_FRAME_DATA_MASTERING_DISPLAY_METADATA);
        const auto* cll_sd = av_frame_get_side_data(video.get(), AV_FRAME_DATA_CONTENT_LIGHT_LEVEL);
        if (mdm_sd == nullptr && cll_sd == nullptr) {
            return;
        }

        if (mdm_sd != nullptr) {
            const auto* mdm = reinterpret_cast<const AVMasteringDisplayMetadata*>(mdm_sd->data);
            if (mdm->has_luminance) {
                // Both are AVRational in cd/m2 already -- av_q2d rather than a scale factor,
                // because the units live in the denominator (10000000/10000 is 1000 nits) and
                // reading the numerator alone is wrong by four orders of magnitude.
                hdr10_max_dml_ = av_q2d(mdm->max_luminance);
                hdr10_min_dml_ = av_q2d(mdm->min_luminance);
            }
        }
        if (cll_sd != nullptr) {
            const auto* cll = reinterpret_cast<const AVContentLightMetadata*>(cll_sd->data);
            hdr10_max_cll_  = static_cast<int>(cll->MaxCLL);
            hdr10_max_fall_ = static_cast<int>(cll->MaxFALL);
        }

        hdr10_seen_.store(true, std::memory_order_relaxed);
        CASPAR_LOG(info) << print() << L" source hdr10: mastering display " << hdr10_max_dml_ << L"/"
                         << hdr10_min_dml_ << L" cd/m2, MaxCLL " << hdr10_max_cll_ << L", MaxFALL "
                         << hdr10_max_fall_;
    }

    /// Resolve the transfer, and name both halves the first time through.
    /// Does this frame's RGB already carry a factor of its own alpha?
    ///
    /// Three answers in precedence order, and the order is the point:
    ///
    ///   1. the OPERATOR, via PREMULTIPLIED / STRAIGHT. On top because the override exists
    ///      for content whose file is wrong -- some Adobe ProRes 4444 exports -- and
    ///      because it is also the escape hatch back to the pre-2026-08-20 rendering.
    ///   2. the FILE, via `AVFrame.alpha_mode`. New in FFmpeg 8, and the first time this
    ///      tree has had a signal rather than a convention. Only png, exr, jpegxl and
    ///      alpha-tagged Matroska set it in 8.1.
    ///   3. the CONVENTION: decoded media is straight, because that is what the formats
    ///      store and what every NLE writes. See core/frame/alpha_mode.h.
    ///
    /// Read from the frame that leaves the filter graph rather than the latched decoder
    /// value, so a graph that legitimately premultiplies is believed over the decoder.
    bool straight_alpha_for(const std::shared_ptr<AVFrame>& video)
    {
        const auto declared = static_cast<core::alpha_declaration>(alpha_declaration_.load());
        if (declared == core::alpha_declaration::premultiplied)
            return false;
        if (declared == core::alpha_declaration::straight)
            return true;

#if LIBAVUTIL_VERSION_MAJOR >= 60
        if (video && video->alpha_mode != AVALPHA_MODE_UNSPECIFIED) {
            const bool straight = video->alpha_mode == AVALPHA_MODE_STRAIGHT;
            if (!alpha_source_logged_.exchange(true)) {
                CASPAR_LOG(info) << print() << L" alpha is "
                                 << (straight ? L"STRAIGHT" : L"PREMULTIPLIED")
                                 << L" because the file says so, not by convention.";
            }
            return straight;
        }
#endif
        return straight_alpha_.load();
    }

    core::color_transfer note_colour(const std::shared_ptr<AVFrame>& video)
    {
        note_hdr10(video);

        const auto trc = get_color_transfer(video, stream_color_trc_);
        if (!logged_colour_) {
            logged_colour_ = true;
            const auto cs = get_color_space(video, stream_color_space_);
            const auto space_name = [&] {
                switch (cs) {
                    case core::color_space::bt601:  return L"bt601";
                    case core::color_space::bt709:  return L"bt709";
                    case core::color_space::bt2020: return L"bt2020";
                    case core::color_space::p3_d65: return L"p3-d65";
                    case core::color_space::p3_dci: return L"p3-dci";
                    default:                        return L"unknown";
                }
            }();
            const auto transfer_name = [&] {
                switch (trc) {
                    case core::color_transfer::pq:  return L"pq";
                    case core::color_transfer::hlg: return L"hlg";
                    default:                        return L"sdr";
                }
            }();
            CASPAR_LOG(info) << print() << L" source colour: colorspace=" << space_name
                             << L" transfer=" << transfer_name;
        }
        return trc;
    }

#ifdef _WIN32
    // D3D11→mixer GPU-direct video path (bypasses filter graph + CPU transfer)
    std::unique_ptr<d3d11_bridge>    d3d11_bridge_;
    // Cheap preconditions passed; the bridge is created lazily when the first
    // hardware surface arrives (see get_hw_format: sw_pix_fmt is not resolved
    // until then).
    bool                             gpu_direct_requested_ = false;
    bool                             gpu_direct_video_ = false;
    bool                             gpu_direct_failed_ = false;
    bool                             gpu_direct_logged_ = false;
    int                              gpu_direct_decoder_idx_ = -1;
#endif

    std::atomic<int64_t> start_{AV_NOPTS_VALUE};
    std::atomic<int64_t> duration_{AV_NOPTS_VALUE};
    std::atomic<int64_t> input_duration_{AV_NOPTS_VALUE};
    std::atomic<int64_t> seek_{AV_NOPTS_VALUE};
    std::atomic<bool>    loop_{false};
    std::atomic<bool>    pingpong_{false};  // ping-pong: auto-reverse at each end
    bool                 growing_{false};

    std::string afilter_;
    std::string vfilter_;

    // Per-parameter animation state for VFPARAM / AFPARAM CALL commands.
    // Keyed by [filter_type_name][param_name].  Accessed from the AMCP thread
    // (set_filter_param) and the decode thread (apply_filter_param_tweens),
    // so protected by param_tween_mutex_.
    std::map<std::string, std::map<std::string, FilterParamTween>> video_param_tweens_;
    std::map<std::string, std::map<std::string, FilterParamTween>> audio_param_tweens_;
    mutable boost::mutex                                           param_tween_mutex_;

    int                              seekable_ = 2;
    core::frame_geometry::scale_mode scale_mode_;
    /// What the operator declared, which outranks everything below.
    std::atomic<int> alpha_declaration_{static_cast<int>(core::alpha_declaration::unspecified)};
    /// The effective answer make_frame consumes: operator, then the file, then the
    /// convention that decoded media is straight. Computed rather than stored so a stream
    /// that only declares itself on frame 30 is still honoured from frame 30.
    std::atomic<bool> straight_alpha_{true};
    std::atomic<bool> alpha_source_logged_{false};
    int64_t                          frame_count_    = 0;
    bool                             frame_flush_    = true;
    int64_t                          frame_time_     = AV_NOPTS_VALUE;
    int64_t                          frame_duration_ = AV_NOPTS_VALUE;
    std::atomic<double>              speed_{1.0};    // playback rate: 0.5 = half-speed, 2.0 = double-speed
    double                           speed_accum_    = 0.0; // fractional accumulator, protected by buffer_mutex_
    std::deque<Frame>                rev_frames_;            // batch of decoded frames for reverse playback (served back→front)
    bool                             rev_active_     = false; // true once the initial reverse seek has been issued
    // Exclusive upper bound on the NEXT reverse batch: frames at or above this pts have already
    // been shown this sweep. INT64_MAX = unbounded. Needed because the near-IN seek below is
    // CLAMPED to the IN point, so the batch it fetches deliberately overlaps the previous one.
    int64_t                          rev_batch_top_  = INT64_MAX;
    // Frames already DELIVERED forward, oldest first, kept so a ping-pong turnaround can be
    // served from pictures already in hand instead of waiting for a seek and a decode.
    // Reversing away from OUT means showing OUT-1, OUT-2 ... which are precisely the frames
    // just displayed, so nothing needs decoding at the turn.
    std::deque<Frame>                shown_;
    // Once `shown_` holds EVERY frame of a looping range, contiguously, playback needs no
    // decoder at all and each boundary becomes an index step instead of a seek and a decode.
    int64_t                          cache_budget_   = 0;      // bytes; 0 disables the cache
    bool                             cache_complete_ = false;  // the whole range is resident
    bool                             cache_serving_  = false;  // playback is coming from it
    int                              cache_index_    = 0;      // cursor, in ascending pts order
    int64_t                          cache_reserved_ = 0;      // bytes held from the shared allowance
    bool                             cache_refused_  = false;  // decided once, not re-tried per frame
    core::draw_frame                 frame_;

    std::deque<Frame>         buffer_;
    mutable boost::mutex      buffer_mutex_;
    boost::condition_variable buffer_cond_;
    std::atomic<bool>         buffer_eof_{false};
    int                       buffer_capacity_ = 0; // set in constructor from config
    std::atomic<int64_t>      current_seek_target_{AV_NOPTS_VALUE};

    std::optional<caspar::executor> video_executor_;
    std::optional<caspar::executor> audio_executor_;

    int latency_ = 0;

    // Periodic producer timing diagnostics
    std::chrono::steady_clock::time_point diag_start_ = std::chrono::steady_clock::now();
    int diag_frames_    = 0;
    int diag_underflows_ = 0;

    std::chrono::steady_clock::time_point last_fps_update_;
    int                     frames_since_update_ = 0;
    double                  current_fps_ = 0.0;

    boost::thread thread_;

    Impl(std::shared_ptr<core::frame_factory> frame_factory,
         core::video_format_desc              format_desc,
         std::string                          name,
         std::string                          path,
         std::string                          vfilter,
         std::string                          afilter,
         std::optional<int64_t>               start,
         std::optional<int64_t>               seek,
         std::optional<int64_t>               duration,
         bool                                 loop,
         int                                  seekable,
         core::frame_geometry::scale_mode     scale_mode,
         bool                                 growing)
        : growing_(growing)
        , frame_factory_(frame_factory)
        , format_desc_(format_desc)
        , format_tb_({format_desc.duration, format_desc.time_scale * format_desc.field_count})
        , name_(name)
        , path_(path)
        , input_(path, graph_, seekable >= 0 && seekable < 2 ? std::optional<bool>(false) : std::optional<bool>())
        , decode_adapter_(resolve_decode_adapter(frame_factory.get()))
        // Opt-in, on the same grounds gpu-direct-decode was: it is a different route
        // through the driver and stays off until measured on this hardware.
        //
        // GATED ON gpu-direct-decode TOO, and that is not belt-and-braces. Without a
        // GPU-direct publish path the decoded AVVkFrame has to be read back to host memory,
        // and a Vulkan decode plus a readback was measured standalone at 78% BELOW software
        // decode throughput -- the worst of the three arms. Enabling one knob without the
        // other would make the server slower and look like the decoder's fault.
        , vk_mixer_device_(
              env::properties().get(L"configuration.ffmpeg.producer.vulkan-decode", false) &&
                      gpu_direct_decode_requested() && frame_factory &&
                      frame_factory->gpu_device_backend() == core::gpu_backend::vulkan
                  ? frame_factory->gpu_device_handle()
                  : nullptr)
        , start_(start ? av_rescale_q(*start, format_tb_, TIME_BASE_Q) : AV_NOPTS_VALUE)
        , duration_(duration ? av_rescale_q(*duration, format_tb_, TIME_BASE_Q) : AV_NOPTS_VALUE)
        , loop_(loop)
        , afilter_(afilter)
        , vfilter_(vfilter)
        , seekable_(seekable)
        , scale_mode_(scale_mode)
        , video_executor_(L"video-executor")
        , audio_executor_(L"audio-executor")
    {
        diagnostics::register_graph(graph_);
        graph_->set_color("underflow", diagnostics::color(0.6f, 0.3f, 0.9f));
        graph_->set_color("frame-time", diagnostics::color(0.0f, 1.0f, 0.0f));
        graph_->set_color("decode-time", diagnostics::color(0.0f, 1.0f, 1.0f));
        graph_->set_color("buffer", diagnostics::color(1.0f, 1.0f, 0.0f));
        graph_->set_color("fps", diagnostics::color(1.0f, 0.6f, 0.0f));

        const int default_buffer_depth = std::max(1, static_cast<int>(format_desc_.fps) / 4);
        // Operator-tunable again. This is the largest latency term in the producer path --
        // fps/4 is 12 frames at 50p, ~250 ms -- and it used to be settable. The config read
        // was lost at some point while the declaration kept saying "set in constructor from
        // config", so there was no way to trade latency against robustness at all.
        // -1 (the documented default) means "follow the video mode". Clamping a negative
        // to 1 instead would give a 1-frame buffer to anyone who wrote the documented
        // value, which is worse than not offering the option.
        const int configured =
            env::properties().get(L"configuration.ffmpeg.producer.buffer-depth", -1);
        buffer_capacity_ = configured > 0 ? configured : default_buffer_depth;

        // Memory a LOOP or PINGPONG range may keep resident so it can be replayed without
        // decoding. Where it lives depends on the decode path and is worth knowing: software
        // frames sit in host RAM and still cost an upload per show, while GPU-direct frames are
        // already mixer textures, so replay costs nothing at all. 0 disables it.
        cache_budget_ = static_cast<int64_t>(
                            env::properties().get(L"configuration.ffmpeg.producer.loop-cache-mb", 256))
                        << 20;
        CASPAR_LOG(debug) << print() << " buffer-depth: " << buffer_capacity_;

        state_["file/name"] = u8(name_);
        state_["file/path"] = u8(path_);
        state_["loop"]      = loop;
        update_state();

        CASPAR_LOG(debug) << print() << " seekable: " << seekable_;

        thread_ = boost::thread([=, this] {
            try {
                run(seek);
            } catch (boost::thread_interrupted&) {
                // Do nothing...
            } catch (ffmpeg::ffmpeg_error_t& ex) {
                if (auto errn = boost::get_error_info<ffmpeg_errn_info>(ex)) {
                    if (*errn == AVERROR_EXIT) {
                        return;
                    }
                }
                CASPAR_LOG_CURRENT_EXCEPTION();
            } catch (const std::exception& e) {
                // boost::diagnostic_information reports "No diagnostic information
                // available" for anything that is not a boost::exception, which
                // hides the one thing worth knowing. Name it.
                CASPAR_LOG(error) << print() << L" producer thread stopped: " << u16(typeid(e).name()) << L": "
                                  << u16(e.what());
                CASPAR_LOG_CURRENT_EXCEPTION();
            } catch (...) {
                CASPAR_LOG(error) << print() << L" producer thread stopped with a non-standard exception.";
                CASPAR_LOG_CURRENT_EXCEPTION();
            }
        });
    }

    ~Impl()
    {
        // Quiesce explicitly and in a defined order rather than relying on
        // implicit member-destruction order. The filter graphs, decoders and
        // sources reference each other, and the run loop plus both filter
        // executors touch all of them; freeing them in declaration order while
        // any of that is still live is how ffmpeg allocations end up being freed
        // twice or read after free. See
        // docs/CasparCG_HRC_Crash_Report_2026-06-17.md §9.1 fix 3.

        // 0. Give back the shared loop-cache allowance. Before anything else, because it is
        //    accounting rather than teardown and must not be skipped by an exception below.
        loop_cache_release(cache_reserved_);
        cache_reserved_ = 0;

        // 1. Stop feeding the pipeline.
        input_.abort();

        // 2. Join the run loop, so nothing else touches decoders_/filters while
        //    they are being torn down.
        try {
            if (thread_.joinable()) {
                thread_.interrupt();
                thread_.join();
            }
        } catch (boost::thread_interrupted&) {
            // Do nothing...
        }

        // 3. Stop the filter executors before freeing the graphs they reference.
        try {
            video_executor_.reset();
            audio_executor_.reset();
        } catch (...) {
            CASPAR_LOG_CURRENT_EXCEPTION();
        }

        // 4. Now fully quiesced: destroy the pipeline in dependency order.
        //    sources_ holds AVFilterContext* borrowed from the filter graphs, so
        //    it must go before the graphs; the graphs reference decoder contexts,
        //    so they go before the decoders (whose destructors join their
        //    worker threads).
        try {
            sources_.clear();
            video_filter_ = Filter{};
            audio_filter_ = Filter{};
            decoders_.clear();
        } catch (...) {
            CASPAR_LOG_CURRENT_EXCEPTION();
        }

        CASPAR_LOG(debug) << print() << " Joined";
    }

    void run(std::optional<int64_t> firstSeek)
    {
        std::vector<int> audio_cadence = format_desc_.audio_cadence;

        input_.reset();
        {
            core::monitor::state streams;
            for (auto n = 0UL; n < input_->nb_streams; ++n) {
                auto st                             = input_->streams[n];
                auto framerate                      = av_guess_frame_rate(nullptr, st, nullptr);
                streams[std::to_string(n) + "/fps"] = {framerate.num, framerate.den};
            }

            boost::lock_guard<boost::mutex> lock(state_mutex_);
            state_["file/streams"] = streams;
        }

        if (input_duration_ == AV_NOPTS_VALUE) {
            int64_t v_dur = AV_NOPTS_VALUE;
            for (auto n = 0UL; n < input_->nb_streams; ++n) {
                auto st = input_->streams[n];
                if (st->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
                    // Capture stream-level color metadata as fallback for frames
                    // where the decoder leaves colorspace/transfer unspecified.
                    stream_color_space_ = static_cast<AVColorSpace>(st->codecpar->color_space);
                    stream_color_trc_   = static_cast<AVColorTransferCharacteristic>(st->codecpar->color_trc);
                    stream_chroma_loc_  = static_cast<AVChromaLocation>(st->codecpar->chroma_location);
                    if (st->duration != AV_NOPTS_VALUE) {
                        v_dur = av_rescale_q(st->duration, st->time_base, {1, AV_TIME_BASE});
                    } else if (input_->duration != AV_NOPTS_VALUE) {
                        // Some formats (like MXF) don't have stream duration, use global.
                        v_dur = input_->duration;
                    }
                    break;
                }
            }
            if (v_dur != AV_NOPTS_VALUE && v_dur > 0) {
                input_duration_ = v_dur;
            } else {
                input_duration_ = input_->duration;
            }
        }

        {
            const auto start = start_.load();
            if (duration_ == AV_NOPTS_VALUE && input_duration_.load() > 0) {
                if (start != AV_NOPTS_VALUE) {
                    duration_ = input_duration_.load() - start;
                } else {
                    duration_ = input_duration_.load();
                }
            }

            const auto firstStart = firstSeek ? av_rescale_q(*firstSeek, format_tb_, TIME_BASE_Q) : start;
            if (firstStart != AV_NOPTS_VALUE) {
                seek_internal(firstStart);
            } else {
                reset(input_->start_time != AV_NOPTS_VALUE ? input_->start_time : 0);
            }
        }

        set_thread_name(L"[ffmpeg::av_producer]");

#ifdef _WIN32
        // ── Initialize D3D11→mixer GPU-direct path ───────────────────────
        // Conditions: D3D11VA active, no user vfilter, progressive content,
        // matching framerate, and an interop route to the mixer's GPU device
        // (WGL_NV_DX_interop2 on OpenGL, VK_KHR_external_memory_win32 on Vulkan).
        {
            // Every branch below reports why the path was or was not taken. It
            // used to fail silently, so there was no way to tell "GPU-direct is
            // running" from "GPU-direct quietly declined" -- which matters
            // because a silent fallback and a path with no benefit look
            // identical in a CPU figure. Knowing whether it is live is a
            // prerequisite for trusting the picture.
            const auto declined = [&](const std::wstring& why) {
                CASPAR_LOG(info) << print() << L" D3D11 GPU-direct video not used: " << why << L".";
            };

            // Opt-in. The extraction pass is arithmetic-free and the mixer's
            // shader does the colour conversion, so the picture matches the
            // software path byte for byte -- but it is still a different route
            // through the driver, and it stays opt-in until each new backend has
            // been verified against that standard.
            const bool gpu_direct_enabled =
                env::properties().get(L"configuration.ffmpeg.producer.gpu-direct-decode", true);

            void*      gpu_dev     = frame_factory_->gpu_device_handle();
            const auto gpu_backend = frame_factory_->gpu_device_backend();

            auto bridge_backend = d3d11_bridge::backend::opengl;
            if (gpu_backend == core::gpu_backend::vulkan)
                bridge_backend = d3d11_bridge::backend::vulkan;

            if (!gpu_direct_enabled) {
                declined(L"disabled (configuration.ffmpeg.producer.gpu-direct-decode is false)");
            } else if (!gpu_dev || gpu_backend == core::gpu_backend::none) {
                declined(L"the mixer exposes no GPU device (GPU affinity moved it, or this is a CPU mixer)");
            } else if (!vfilter_.empty()) {
                declined(L"a video filter is set (" + u16(vfilter_) + L"), which requires CPU frames");
            } else {
                // The handle is passed straight to the bridge, which is the only
                // thing that knows which backend it belongs to and how to cast it.

                // Find the video decoder with D3D11VA active
                bool found_video = false;
                for (auto& [idx, dec] : decoders_) {
                    if (!dec.ctx || dec.ctx->codec_type != AVMEDIA_TYPE_VIDEO)
                        continue;

                    found_video = true;

                    if (!dec.ctx->hw_device_ctx) {
                        declined(L"the decoder has no hardware device (codec not hardware-accelerated here)");
                        break;
                    }

                    // NOTE: dec.sw_pix_fmt is deliberately NOT checked here. A
                    // D3D11VA decoder only resolves ctx->sw_pix_fmt in its first
                    // get_format callback, i.e. once decoding has started, so at
                    // this point it is always AV_PIX_FMT_NONE. Requiring it here
                    // is why this path never actually ran.

                    // Check progressive (field_order or container flags)
                    auto* st = input_->streams[idx];
                    bool is_progressive = (st->codecpar->field_order == AV_FIELD_PROGRESSIVE ||
                                           st->codecpar->field_order == AV_FIELD_UNKNOWN);

                    // Check framerate match
                    auto content_fr = av_guess_frame_rate(nullptr, st, nullptr);
                    bool fps_match = (content_fr.num * format_desc_.framerate.denominator() ==
                                      content_fr.den * format_desc_.framerate.numerator());

                    // Check auto-deinterlace setting
                    auto deint = u8(env::properties().get<std::wstring>(
                        L"configuration.ffmpeg.producer.auto-deinterlace", L"interlaced"));
                    bool deint_none = (deint == "none") || is_progressive;

                    if (!is_progressive) {
                        declined(L"content is interlaced");
                    } else if (!fps_match) {
                        declined(L"content framerate does not match the channel");
                    } else if (!deint_none) {
                        declined(L"auto-deinterlace is active");
                    } else {
                        // Everything knowable up front passes. Let the decoder
                        // emit hardware surfaces; the bridge is created when the
                        // first one arrives, which is the earliest moment the
                        // surface format and dimensions are actually known. If
                        // that fails we fall back to the transfer path and say so.
                        // init() only needs the D3D11 device and the mixer's GPU
                        // device, both known now. The size-dependent resources
                        // are built on the first hardware frame, which is also
                        // the first moment the surface format is resolved.
#if defined(ENABLE_VULKAN) && LIBAVUTIL_VERSION_MAJOR >= 60
                        // A Vulkan decoder needs no bridge at all: its frames are already
                        // on the mixer's device, so there is nothing to import and the
                        // D3D11 bridge would be handed a VkDevice and fail. Detected from
                        // the hardware context's own type rather than from configuration,
                        // because the ordered hwaccel preference in create_decoder may have
                        // fallen back to D3D11VA after asking for Vulkan.
                        const bool vulkan_hwaccel =
                            reinterpret_cast<AVHWDeviceContext*>(dec.ctx->hw_device_ctx->data)->type ==
                            AV_HWDEVICE_TYPE_VULKAN;
                        if (vulkan_hwaccel) {
                            gpu_direct_requested_   = true;
                            gpu_direct_video_       = true;
                            gpu_direct_decoder_idx_ = idx;
                            dec.gpu_direct_mode_    = true;
                            CASPAR_LOG(info) << print()
                                             << L" Vulkan GPU-direct video eligible: the decoder allocates on "
                                                L"the mixer's own device, so no import bridge is needed.";
                            break; // Only one video decoder
                        }
#endif

                        d3d11_bridge_ = std::make_unique<d3d11_bridge>();
                        if (!d3d11_bridge_->init(dec.ctx->hw_device_ctx, gpu_dev, bridge_backend)) {
                            d3d11_bridge_.reset();
                            declined(bridge_backend == d3d11_bridge::backend::vulkan
                                         ? L"the D3D11->Vulkan import bridge failed to initialise"
                                         : L"the WGL_NV_DX_interop2 bridge failed to initialise");
                        } else {
                            gpu_direct_requested_ = true;
                            // The decoder must start emitting hardware surfaces
                            // now, otherwise the first one never arrives and there
                            // is nothing to decide on. If plane extraction then
                            // fails, the producer falls back to CPU transfer.
                            gpu_direct_video_       = true;
                            gpu_direct_decoder_idx_ = idx;
                            dec.gpu_direct_mode_    = true;
                            CASPAR_LOG(info)
                                << print() << L" D3D11 GPU-direct video eligible on the "
                                << (bridge_backend == d3d11_bridge::backend::vulkan ? L"Vulkan" : L"OpenGL")
                                << L" mixer; plane extraction starts on the first hardware frame.";
                        }
                    }
                    break; // Only one video decoder
                }

                if (!found_video)
                    declined(L"no video decoder was created");
            }
        }
#endif

        boost::range::rotate(audio_cadence, std::end(audio_cadence) - 1);

        Frame frame;
        Frame last_dropped_frame;
        std::shared_ptr<AVFrame> last_valid_video;
        timer frame_timer;
        timer decode_timer;

        int warning_debounce = 0;

        while (!thread_.interruption_requested()) {
            try {
                {
                    const auto seek = seek_.exchange(AV_NOPTS_VALUE);

                    if (seek != AV_NOPTS_VALUE) {
                        try {
                            seek_internal(seek);
                            current_seek_target_ = seek;
                            last_dropped_frame   = Frame{};
                            last_valid_video.reset();
                            frame                = Frame{};
                        } catch (const std::exception& e) {
                            CASPAR_LOG(warning) << print() << " Seek (graph rebuild) failed: " << e.what() << " - retrying";
                            // Restore the seek command so the next loop iteration retries it,
                            // unless another newer seek has already been enqueued!
                            int64_t expected = AV_NOPTS_VALUE;
                            seek_.compare_exchange_strong(expected, seek);
                            std::this_thread::sleep_for(std::chrono::milliseconds(20));
                        }
                        continue;
                    }
                }

                {
                    // TODO (perf) seek as soon as input is past duration or eof.

                    auto start    = start_.load();
                    auto duration = duration_.load();

                    start       = start != AV_NOPTS_VALUE ? start : 0;
                    auto end    = duration != AV_NOPTS_VALUE ? start + duration : INT64_MAX;
                    auto time   = frame.pts != AV_NOPTS_VALUE ? frame.pts + frame.duration : 0;
                    
                    if (frame.frame_count == 0 && frame_count_ == 0 && current_seek_target_.load() != AV_NOPTS_VALUE) {
                        // Special case: we are just starting (or just seeked) and haven't decoded any frames yet.
                        // We must NOT calculate EOF based on `frame` because `frame` is empty/zero!
                        // This prevents an immediate EOF triggering before the decode pipeline produces the first result.
                        buffer_eof_ = false;
                    } else {
                        buffer_eof_ = !growing_ && ((video_filter_.eof && audio_filter_.eof) ||
                                      av_rescale_q(time, TIME_BASE_Q, format_tb_) >= av_rescale_q(end, TIME_BASE_Q, format_tb_));
                    }

                    if (buffer_eof_) {
                        if (current_seek_target_.load() != AV_NOPTS_VALUE && (last_dropped_frame.video || last_dropped_frame.audio)) {
                            // We hit EOF while fast-forwarding to a seek target (the target was beyond the video).
                            // Render and push the very last dropped frame so we don't output a black screen.
                            last_dropped_frame.frame = core::draw_frame(
                                make_frame(this, *frame_factory_, last_dropped_frame.video, last_dropped_frame.audio, get_color_space(last_dropped_frame.video, stream_color_space_), scale_mode_, straight_alpha_for(last_dropped_frame.video), get_color_transfer(last_dropped_frame.video, stream_color_trc_)));
                            last_dropped_frame.frame_count = frame_count_++;

                            boost::unique_lock<boost::mutex> buffer_lock(buffer_mutex_);
                            buffer_cond_.wait(buffer_lock, [&] { return buffer_.size() < buffer_capacity_; });
                            if (seek_.load() == AV_NOPTS_VALUE) {
                                buffer_.push_back(std::move(last_dropped_frame));
                                buffer_cond_.notify_all();
                            }
                            current_seek_target_ = AV_NOPTS_VALUE;
                        }

                        if (loop_ && !pingpong_ && frame_count_ > 2 && seek_.load() == AV_NOPTS_VALUE && speed_.load() >= 0.0) {
                            // Normal loop — seek back to IN point.
                            // Only auto-loop if no user seek is pending; if seek_ is set the
                            // next iteration will consume it and we must not override it with start.
                            // ALSO disabled if playing in reverse, let frontend logic handle reverse boundary.
                            //
                            // Let the consumer take the decoded tail FIRST. seek_internal() clears
                            // buffer_ -- correct for a user seek, which must discard, and wrong
                            // here: at a loop boundary those frames are the clip's last frames and
                            // clearing them loses one iteration's tail every time round.
                            //
                            // MEASURED 2026-08-19 from the picture, not the counter: a 4-frame loop
                            // (SEEK 40 LENGTH 4 LOOP) over 26 captures showed markers {40, 41, 43}
                            // -- frame 42 never reached the screen. The HAP producer, which flushes
                            // nothing at its wrap, showed all four.
                            //
                            // Hence the drain is HERE and not in seek_internal, which is the mirror
                            // image of speed_accum_: that belongs inside seek_internal because every
                            // caller invalidates it, this belongs only at the loop caller because
                            // the explicit-seek caller must still discard.
                            //
                            // Unbounded in time on purpose, with two escapes that always release
                            // it: a pending user seek, and shutdown -- boost's wait_for is itself
                            // an interruption point, so the thread_interrupted this loop already
                            // catches breaks it without needing a flag. A paused channel does not
                            // drain, so the wrap waits for it and the clip holds its last frame,
                            // which is what a paused layer should do.
                            {
                                boost::unique_lock<boost::mutex> buffer_lock(buffer_mutex_);
                                while (!buffer_.empty() && seek_.load() == AV_NOPTS_VALUE) {
                                    buffer_cond_.wait_for(buffer_lock,
                                                          boost::chrono::milliseconds(10));
                                }
                            }
                            frame = Frame{};
                            seek_internal(start);
                        } else {
                            // ping-pong, non-looping, or a user seek is pending:
                            // stall here so next_frame() / the top-of-loop seek_ check can act.
                            std::this_thread::sleep_for(std::chrono::milliseconds(10));
                        }
                        // TODO (fix) Limit live polling due to bugs.
                        continue;
                    }
                }

                bool progress = false;
                {
                    progress |= schedule();

                    std::vector<std::future<bool>> futures;

#ifdef _WIN32
                    if (gpu_direct_video_) {
                        // Hardware decoding can decline after the fact: H.264 High 10
                        // and VP9 fall back to software here. The decoder then emits
                        // ordinary frames while this path waits for surfaces that
                        // never arrive, and the channel stalls on "Waiting for video
                        // frame...". Stand down as soon as that is observed. This has
                        // to happen here, where the waiting is: a check further
                        // downstream is never reached by a stalled producer.
                        {
                            auto it = decoders_.find(gpu_direct_decoder_idx_);
                            if (it != decoders_.end() &&
                                it->second.saw_software_frame_.load(std::memory_order_relaxed)) {
                                gpu_direct_video_          = false;
                                it->second.gpu_direct_mode_ = false;
                                CASPAR_LOG(info) << print()
                                                 << L" D3D11 GPU-direct video stood down: this stream decodes in "
                                                    L"software.";
                            }
                        }
                    }

                    if (gpu_direct_video_) {
                        // GPU-direct: video comes from decoder hw_output, not filter graph
                        if (!video_filter_.frame) {
                            // find(), not at(): every other access to this map
                            // is written this way, and the one that was not
                            // threw on every iteration when the decoder went
                            // missing. If it is gone, stand down to the software
                            // path rather than throw -- the frame is still
                            // delivered, just not by this route.
                            auto it = decoders_.find(gpu_direct_decoder_idx_);
                            if (it == decoders_.end()) {
                                gpu_direct_video_ = false;
                                CASPAR_LOG(warning)
                                    << print()
                                    << L" D3D11 GPU-direct video stood down: the decoder went away.";
                            } else {
                                auto hw_frame = it->second.pop_hw();
                                if (hw_frame) {
                                    video_filter_.frame = std::move(hw_frame);
                                    progress            = true;
                                } else {
                                    // Carry the decoder's end-of-file across to the
                                    // filter's flag, because nothing else will.
                                    //
                                    // On the software path the decoder marks EOF by
                                    // pushing a sentinel into `output`, and the video
                                    // filter reads it and raises video_filter_.eof.
                                    // GPU-direct frames bypass the filter graph
                                    // entirely, so that sentinel is never read and the
                                    // flag stayed false for ever.
                                    //
                                    // buffer_eof_ is computed from it, and the loop
                                    // wrap is gated on buffer_eof_. So a looping clip
                                    // played with gpu-direct-decode ran to a few
                                    // frames short of the end and stopped dead --
                                    // position frozen, "Waiting for video frame..."
                                    // repeating, nothing decoded again. On both
                                    // mixers. An explicit SEEK moved it and it then
                                    // froze at the new position, which is what
                                    // separated this from a stale-queue problem.
                                    // Mirrored from the decoder every time, never
                                    // latched. Setting it once and leaving it made the
                                    // clip wrap and then sit at 0.00 for ever: the
                                    // wrap resets the decoder's eof, but a latched
                                    // flag still said end-of-file, so nothing was
                                    // decoded after the first loop.
                                    // ...and not while a flush is outstanding. A
                                    // loop wrap requests the seek and returns; the
                                    // decode thread performs it later. In that window
                                    // the decoder still reports the end of file it had
                                    // already reached, so acting on it here asked for
                                    // the wrap again -- the clip reached 0.00 and sat
                                    // there, re-seeking to the start for ever instead
                                    // of playing from it.
                                    video_filter_.eof = it->second.at_eof() && !it->second.flush_pending();
                                }
                            }
                        }
                    } else
#endif
                    if (!video_filter_.frame) {
                        futures.push_back(video_executor_->begin_invoke([&]() { return video_filter_(); }));
                    }

                    if (!audio_filter_.frame) {
                        futures.push_back(audio_executor_->begin_invoke([&]() { return audio_filter_(audio_cadence[0]); }));
                    }

                    for (auto& future : futures) {
                        progress |= future.get();
                    }
                }

                if ((!video_filter_.frame && !video_filter_.eof) || (!audio_filter_.frame && !audio_filter_.eof)) {
                    if (!progress) {
                        if (warning_debounce++ % 500 == 100) {
                            if (!video_filter_.frame && !video_filter_.eof) {
                                CASPAR_LOG(warning) << print() << " Waiting for video frame...";
                            } else if (!audio_filter_.frame && !audio_filter_.eof) {
                                CASPAR_LOG(warning) << print() << " Waiting for audio frame...";
                            } else {
                                CASPAR_LOG(warning) << print() << " Waiting for frame...";
                            }
                        }

                        // TODO (perf): Avoid live loop.
                        // 1ms keeps CPU usage acceptable while minimising the
                        // pipeline fill-up latency after a loop seek.
                        std::this_thread::sleep_for(std::chrono::milliseconds(warning_debounce > 25 ? 10 : 1));
                    }
                    continue;
                }

                warning_debounce = 0;

                // TODO (fix)
                // if (start_ != AV_NOPTS_VALUE && frame.pts < start_) {
                //    seek_internal(start_);
                //    continue;
                //}

                const auto start_time = input_->start_time != AV_NOPTS_VALUE ? input_->start_time : 0;

                bool use_video_pts = video_filter_.frame != nullptr;

                if (video_filter_.frame) {
                    frame.video      = std::move(video_filter_.frame);
                    last_valid_video = frame.video;
#ifdef _WIN32
                    if (gpu_direct_video_) {
                        // GPU-direct: frame came from decoder, use stream time_base
                        auto* st = input_->streams[gpu_direct_decoder_idx_];
                        frame.start_time = start_time;
                        frame.pts        = av_rescale_q(frame.video->pts, st->time_base, TIME_BASE_Q) - start_time;
                        auto content_fr  = av_guess_frame_rate(nullptr, st, nullptr);
                        frame.duration   = av_rescale_q(1, av_inv_q(content_fr), TIME_BASE_Q);
                    } else
#endif
                    {
                        const auto tb    = av_buffersink_get_time_base(video_filter_.sink);
                        const auto fr    = av_buffersink_get_frame_rate(video_filter_.sink);
                        frame.start_time = start_time;
                        frame.pts        = av_rescale_q(frame.video->pts, tb, TIME_BASE_Q) - start_time;
                        frame.duration   = av_rescale_q(1, av_inv_q(fr), TIME_BASE_Q);
                    }
                } else if (last_valid_video) {
                    frame.video = last_valid_video; // Keep the last video frame if we have an audio tail!
                }

                if (audio_filter_.frame) {
                    frame.audio      = std::move(audio_filter_.frame);
                    const auto tb    = av_buffersink_get_time_base(audio_filter_.sink);
                    const auto sr    = av_buffersink_get_sample_rate(audio_filter_.sink);
                    frame.start_time = start_time;
                    if (!use_video_pts) {
                        frame.pts        = av_rescale_q(frame.audio->pts, tb, TIME_BASE_Q) - start_time;
                    }
                    if (frame.duration <= 0) {
                        frame.duration   = av_rescale_q(frame.audio->nb_samples, {1, sr}, TIME_BASE_Q);
                    }
                }

                if (current_seek_target_.load() != AV_NOPTS_VALUE && frame.pts != AV_NOPTS_VALUE) {
                    if (frame.pts < current_seek_target_.load() - (frame.duration > 0 ? frame.duration / 2 : 0)) {
                        last_dropped_frame = std::move(frame);
                        frame = Frame{};
                        continue;
                    } else {
                        current_seek_target_ = AV_NOPTS_VALUE;
                        last_dropped_frame = Frame{};
                    }
                }

                frame.frame = core::draw_frame(
#ifdef _WIN32
                    [&]() -> core::const_frame {
                        // Hardware decoding can decline after the fact -- H.264 High
                        // 10 and VP9 fall back to software on this hardware. The
                        // decoder then emits ordinary frames while the producer is
                        // still set up to wait for hardware surfaces, and the
                        // channel stalls on "Waiting for video frame...". Stand the
                        // GPU-direct path down the moment a software frame appears.
#if defined(ENABLE_VULKAN) && LIBAVUTIL_VERSION_MAJOR >= 60
                        // FFmpeg's Vulkan compute decoders (prores, prores_raw, ffv1, dpx)
                        // allocate on the mixer's own device, so their planes reach the
                        // mixer with a device-local copy and no host memory at all.
                        // Handled before the stand-down below, which would otherwise read
                        // "not D3D11" as "decoding in software".
                        if (gpu_direct_video_ && vk_mixer_device_ && frame.video &&
                            frame.video->format == AV_PIX_FMT_VULKAN && !gpu_direct_failed_) {
                            // LOCK THE FRAME around read, submit and write-back. FFmpeg's
                            // own words (hwcontext_vulkan.h): "Users SHOULD only ever lock
                            // just before command submission in order to get accurate frame
                            // properties, and unlock immediately after command submission
                            // without waiting for it to finish."
                            //
                            // Doing it unlocked is not a theoretical race. `sem_value` was
                            // read here on the channel thread while FFmpeg's decode threads
                            // advanced it, so our submission could signal a value FFmpeg had
                            // already used -- and a timeline semaphore signalled twice with
                            // the same value is undefined. Measured 2026-08-21: ONE ProRes
                            // producer ran indefinitely; FOUR lost the device within three
                            // seconds, with a matching `nvlddmkm stopped responding` TDR. It
                            // is ours rather than FFmpeg's -- four concurrent Vulkan ProRes
                            // decoders sharing one device in ffmpeg.exe, ~1500 frames each,
                            // complete cleanly.
                            //
                            // The host wait inside `copy_planes` happens under this lock,
                            // which is safe because `sem_value` is the completion value of
                            // work FFmpeg has ALREADY submitted by the time
                            // avcodec_receive_frame hands the frame over -- FFmpeg does not
                            // need this lock to make that value arrive.
                            auto* vk_fc  = reinterpret_cast<AVHWFramesContext*>(frame.video->hw_frames_ctx->data);
                            auto* vk_fhw = static_cast<AVVulkanFramesContext*>(vk_fc->hwctx);
                            auto* vk_f   = reinterpret_cast<AVVkFrame*>(frame.video->data[0]);

                            struct frame_lock
                            {
                                AVVulkanFramesContext* fhw;
                                AVHWFramesContext*     fc;
                                AVVkFrame*             f;
                                frame_lock(AVVulkanFramesContext* a, AVHWFramesContext* b, AVVkFrame* c)
                                    : fhw(a), fc(b), f(c)
                                {
                                    if (fhw && fhw->lock_frame)
                                        fhw->lock_frame(fc, f);
                                }
                                ~frame_lock()
                                {
                                    if (fhw && fhw->unlock_frame)
                                        fhw->unlock_frame(fc, f);
                                }
                            } vk_lock(vk_fhw, vk_fc, vk_f);

                            common::bit_depth plane_depth = common::bit_depth::bit8;
                            bool              has_alpha   = false;
                            auto              planes = describe_av_vulkan_planes(frame.video.get(), plane_depth, has_alpha);
                            if (!gpu_direct_logged_) {
                                CASPAR_LOG(debug) << L"[av_producer] AVVkFrame described as "
                                                  << static_cast<int>(planes.size()) << L" planes, depth "
                                                  << static_cast<int>(plane_depth) << L", alpha " << has_alpha;
                                // The premise of copying on the mixer's graphics queue is that
                                // these images are CONCURRENT, i.e. queue_family is
                                // VK_QUEUE_FAMILY_IGNORED (0xFFFFFFFF). If it names a family
                                // instead they are EXCLUSIVE and reading them from another
                                // family is undefined -- so this is asserted out loud rather
                                // than assumed.
                                CASPAR_LOG(info) << L"[av_producer] AVVkFrame queue_family[0]=0x"
                                                 << std::hex << vk_f->queue_family[0] << std::dec
                                                 << (vk_f->queue_family[0] == VK_QUEUE_FAMILY_IGNORED
                                                         ? L" (CONCURRENT, as intended)"
                                                         : L" (EXCLUSIVE -- the graphics-queue copy is UNDEFINED)");
                            }

                            // Built on the first Vulkan frame, not at producer start: the
                            // importer needs nothing but the device, but constructing it
                            // eagerly would allocate a command buffer and a fence for every
                            // producer whether or not the decoder ever chose Vulkan.
                            if (!planes.empty() && !vk_importer_) {
                                try {
                                    vk_importer_ =
                                        std::make_unique<accelerator::vulkan::av_vulkan_importer>(vk_mixer_device_);
                                } catch (const std::exception& e) {
                                    CASPAR_LOG(warning) << print() << L" Vulkan GPU-direct importer failed: "
                                                        << u16(e.what());
                                }
                            }

                            std::vector<std::shared_ptr<core::texture>> textures;
                            if (!planes.empty() && vk_importer_ &&
                                vk_importer_->copy_planes(planes, plane_depth, textures)) {
                                // Record the value the copy signalled, under the lock taken
                                // above -- which is what makes this read-modify-write safe
                                // against FFmpeg's own decode threads. FFmpeg waits on it
                                // before reusing a pooled frame, so skipping it would let the
                                // decoder overwrite an image the copy is still reading.
                                for (std::size_t i = 0; i < planes.size(); ++i)
                                    vk_f->sem_value[i] += 1;

                                if (!gpu_direct_logged_) {
                                    gpu_direct_logged_ = true;
                                    CASPAR_LOG(info)
                                        << print() << L" Vulkan GPU-direct video active: "
                                        << static_cast<int>(planes.size())
                                        << L" planes decoded by an FFmpeg compute shader on the mixer's own "
                                           L"device and copied device-local (no CPU frame, no readback).";
                                }

                                auto desc = core::pixel_format_desc(has_alpha ? core::pixel_format::ycbcra
                                                                              : core::pixel_format::ycbcr);
                                for (const auto& pl : planes)
                                    desc.planes.push_back(
                                        core::pixel_format_desc::plane(pl.width, pl.height, 1, plane_depth));
                                desc.color_space    = get_color_space(frame.video, stream_color_space_);
                                desc.color_transfer = note_colour(frame.video);
                                switch (frame.video->chroma_location) {
                                    case AVCHROMA_LOC_CENTER:
                                        desc.chroma_location = core::chroma_location::center;
                                        break;
                                    case AVCHROMA_LOC_TOPLEFT:
                                        desc.chroma_location = core::chroma_location::topleft;
                                        break;
                                    default:
                                        desc.chroma_location = core::chroma_location::left;
                                        break;
                                }

                                array<const std::int32_t> audio_data;
                                if (frame.audio) {
                                    const int                 channel_count = 16;
                                    std::vector<std::int32_t> buf(frame.audio->nb_samples * channel_count, 0);
                                    auto  src_channels = frame.audio->ch_layout.nb_channels;
                                    auto* src          = reinterpret_cast<std::int32_t*>(frame.audio->data[0]);
                                    for (int i = 0; i < frame.audio->nb_samples; ++i) {
                                        for (int j = 0; j < std::min(channel_count, src_channels); ++j) {
                                            buf[i * channel_count + j] = src[i * src_channels + j];
                                        }
                                    }
                                    audio_data = array<const std::int32_t>(buf);
                                }

                                // No host pixels, reported honestly by host_image_state().
                                std::vector<array<const std::uint8_t>> image_data;
                                for (std::size_t i = 0; i < planes.size(); ++i)
                                    image_data.emplace_back(static_cast<std::size_t>(0));

                                return core::const_frame(this, std::move(image_data), std::move(audio_data), desc,
                                                         std::move(textures));
                            }

                            // Stop trying rather than pay for a failed import every frame.
                            gpu_direct_failed_ = true;
                            CASPAR_LOG(warning) << print()
                                                << L" Vulkan GPU-direct plane copy failed; falling back to the "
                                                   L"CPU transfer path for this producer.";
                        }

                        // A Vulkan frame the branch above declined still must not be read as
                        // a software frame by the stand-down below: the decoder is a hardware
                        // one and will keep producing AV_PIX_FMT_VULKAN.
                        if (frame.video && frame.video->format == AV_PIX_FMT_VULKAN) {
                            auto sw_frame = alloc_frame();
                            // AV_PIX_FMT_NONE lets av_hwframe_transfer_data pick the pool's
                            // own software format, which is the decoder's native planar
                            // layout -- there is no single right answer to hardcode here the
                            // way NV12 is for D3D11.
                            sw_frame->format = AV_PIX_FMT_NONE;
                            if (av_hwframe_transfer_data(sw_frame.get(), frame.video.get(), 0) == 0) {
                                av_frame_copy_props(sw_frame.get(), frame.video.get());
                                frame.video = std::move(sw_frame);
                            }

                            // Stand the path down HERE, with the true reason. Leaving it to
                            // the check below would log "this stream decodes in software",
                            // which is exactly what did not happen: the decoder is a Vulkan
                            // one and the copy to the mixer is what declined.
                            if (gpu_direct_video_) {
                                gpu_direct_video_ = false;
                                if (gpu_direct_decoder_idx_ >= 0) {
                                    auto it = decoders_.find(gpu_direct_decoder_idx_);
                                    if (it != decoders_.end())
                                        it->second.gpu_direct_mode_ = false;
                                }
                                CASPAR_LOG(info) << print()
                                                 << L" Vulkan GPU-direct video stood down; decoded frames are "
                                                    L"being read back to host memory.";
                            }
                        }
#endif

                        if (gpu_direct_video_ && frame.video && frame.video->format != AV_PIX_FMT_D3D11 &&
                            frame.video->format != AV_PIX_FMT_VULKAN) {
                            gpu_direct_video_ = false;
                            if (gpu_direct_decoder_idx_ >= 0) {
                                auto it = decoders_.find(gpu_direct_decoder_idx_);
                                if (it != decoders_.end())
                                    it->second.gpu_direct_mode_ = false;
                            }
                            CASPAR_LOG(info) << print()
                                             << L" D3D11 GPU-direct video stood down: this stream decodes in "
                                                L"software.";
                        }

                        if (gpu_direct_video_ && frame.video && frame.video->format == AV_PIX_FMT_D3D11) {
                            auto* d3d11_tex = reinterpret_cast<ID3D11Texture2D*>(frame.video->data[0]);
                            auto  array_idx = static_cast<int>(reinterpret_cast<intptr_t>(frame.video->data[1]));

                            // The bridge is built on the first hardware frame, not
                            // at producer start: only now are the surface format
                            // and dimensions actually known.
                            D3D11_TEXTURE2D_DESC hw_desc = {};
                            if (d3d11_tex)
                                d3d11_tex->GetDesc(&hw_desc);

                            if (!gpu_direct_failed_ && d3d11_bridge_ && d3d11_tex &&
                                d3d11_bridge_->setup_planes(frame.video->width, frame.video->height,
                                                            hw_desc.Format)) {

                                auto planes = d3d11_bridge_->convert_planes(d3d11_tex, array_idx);
                                if (planes.first && planes.second) {
                                    if (!gpu_direct_logged_) {
                                        gpu_direct_logged_ = true;
                                        // Name the format actually decoded, not a
                                        // guess: this line used to say NV12 for
                                        // everything, which reads as "your 10-bit
                                        // clip was handed over as 8-bit".
                                        // setup_planes accepts only these three.
                                        const wchar_t* surface = hw_desc.Format == DXGI_FORMAT_P010   ? L"P010"
                                                                 : hw_desc.Format == DXGI_FORMAT_P016 ? L"P016"
                                                                                                      : L"NV12";
                                        CASPAR_LOG(info)
                                            << print() << L" D3D11 GPU-direct video active: " << surface
                                            << L" planes handed to the mixer, "
                                               L"which performs the colour conversion (no CPU frame, no "
                                               L"VideoProcessor).";
                                    }

                                    // Semi-planar exactly as the decoder produced
                                    // it. The mixer's shader converts, so this
                                    // matches the software path by construction.
                                    // bit16 for P010/P016: their significant bits are
                                    // high-aligned in each word, so the mixer's
                                    // precision factor must be 1.0 -- exactly what
                                    // bit16 selects. See pixel_format::nv12.
                                    const auto plane_depth = d3d11_bridge_->plane_depth();
                                    auto desc = core::pixel_format_desc(core::pixel_format::nv12);
                                    desc.planes.push_back(core::pixel_format_desc::plane(
                                        frame.video->width, frame.video->height, 1, plane_depth));
                                    desc.planes.push_back(core::pixel_format_desc::plane(
                                        frame.video->width / 2, frame.video->height / 2, 2, plane_depth));
                                    desc.color_space    = get_color_space(frame.video, stream_color_space_);
                                    desc.color_transfer = note_colour(frame.video);
                                    if (frame.video->chroma_location != AVCHROMA_LOC_UNSPECIFIED) {
                                        switch (frame.video->chroma_location) {
                                            case AVCHROMA_LOC_LEFT:
                                                desc.chroma_location = core::chroma_location::left;
                                                break;
                                            case AVCHROMA_LOC_CENTER:
                                                desc.chroma_location = core::chroma_location::center;
                                                break;
                                            case AVCHROMA_LOC_TOPLEFT:
                                                desc.chroma_location = core::chroma_location::topleft;
                                                break;
                                            default:
                                                desc.chroma_location = core::chroma_location::left;
                                                break;
                                        }
                                    }

                                    array<const std::int32_t> audio_data;
                                    if (frame.audio) {
                                        const int                 channel_count = 16;
                                        std::vector<std::int32_t> buf(frame.audio->nb_samples * channel_count, 0);
                                        auto  src_channels = frame.audio->ch_layout.nb_channels;
                                        auto* src          = reinterpret_cast<std::int32_t*>(frame.audio->data[0]);
                                        for (int i = 0; i < frame.audio->nb_samples; ++i) {
                                            for (int j = 0; j < std::min(channel_count, src_channels); ++j) {
                                                buf[i * channel_count + j] = src[i * src_channels + j];
                                            }
                                        }
                                        audio_data = array<const std::int32_t>(buf);
                                    }

                                    // No host pixels: consumers that need them ask
                                    // the channel, which reads back the composited
                                    // frame. host_image_state() reports this
                                    // honestly as `unavailable`.
                                    std::vector<array<const std::uint8_t>> image_data;
                                    image_data.emplace_back(static_cast<std::size_t>(0));
                                    image_data.emplace_back(static_cast<std::size_t>(0));

                                    // Already core::texture on both backends, so
                                    // the hand-over below is identical for each.
                                    std::vector<std::shared_ptr<core::texture>> textures;
                                    textures.push_back(std::move(planes.first));
                                    textures.push_back(std::move(planes.second));

                                    return core::const_frame(this, std::move(image_data), std::move(audio_data), desc,
                                                             std::move(textures));
                                }

                                // Extraction failed. Stop trying: it will not
                                // start working, and retrying every frame would
                                // cost a D3D11 pass per frame for nothing.
                                if (!gpu_direct_failed_) {
                                    gpu_direct_failed_ = true;
                                    CASPAR_LOG(warning)
                                        << print()
                                        << L" D3D11 GPU-direct plane extraction failed; falling back to the CPU "
                                           L"transfer path for this producer.";
                                }
                            } else if (!gpu_direct_failed_) {
                                gpu_direct_failed_ = true;
                                CASPAR_LOG(warning) << print()
                                                    << L" D3D11 GPU-direct setup failed; falling back to the CPU "
                                                       L"transfer path for this producer.";
                            }

                            // Bridge failed — fall through to CPU path with transfer
                            auto sw_frame = alloc_frame();
                            sw_frame->format = AV_PIX_FMT_NV12;
                            sw_frame->width  = frame.video->width;
                            sw_frame->height = frame.video->height;
                            if (av_hwframe_transfer_data(sw_frame.get(), frame.video.get(), 0) == 0) {
                                av_frame_copy_props(sw_frame.get(), frame.video.get());
                                frame.video = sw_frame;
                            }
                        }
                        return core::const_frame(
                            make_frame(this, *frame_factory_, frame.video, frame.audio,
                                get_color_space(frame.video, stream_color_space_), scale_mode_,
                                straight_alpha_for(frame.video),
                                note_colour(frame.video)));
                    }()
#else
                    make_frame(this, *frame_factory_, frame.video, frame.audio, get_color_space(frame.video, stream_color_space_), scale_mode_, straight_alpha_for(frame.video), note_colour(frame.video))
#endif
                );

                frame.frame_count = frame_count_++;

                graph_->set_value("decode-time", decode_timer.elapsed() * format_desc_.fps * 0.5);

                {
                    boost::unique_lock<boost::mutex> buffer_lock(buffer_mutex_);
                    buffer_cond_.wait(buffer_lock, [&] { return buffer_.size() < buffer_capacity_; });
                    if (seek_ == AV_NOPTS_VALUE) {
                        buffer_.push_back(frame);
                    }
                }

                if (format_desc_.field_count != 2 || frame_count_ % 2 == 1) {
                    // Update the frame-time every other frame when interlaced
                    graph_->set_value("frame-time", frame_timer.elapsed() * format_desc_.hz * 0.5);
                    frame_timer.restart();
                }

                decode_timer.restart();

                graph_->set_value("buffer", static_cast<double>(buffer_.size()) / static_cast<double>(buffer_capacity_));

                boost::range::rotate(audio_cadence, std::end(audio_cadence) - 1);

                // Tick all animated filter parameters and push updated values into
                // the live filter graph via avfilter_graph_send_command.
                // Called here because both video and audio filter futures have been
                // get()-ed above, so the graph is idle and safe to command.
                apply_filter_param_tweens();
            } catch (boost::thread_interrupted&) {
                throw;
            } catch (std::exception& e) {
                CASPAR_LOG(error) << print() << " Exception in decode loop (will retry): " << e.what();
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            } catch (...) {
                CASPAR_LOG(error) << print() << " Unknown exception in decode loop (will retry)";
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
        }
    }

    void update_state()
    {
        std::wstringstream stats;
        stats.precision(2);
        stats << std::fixed;
        stats << u16(print()) << L" fps: " << current_fps_;
        graph_->set_text(stats.str());

        boost::lock_guard<boost::mutex> lock(state_mutex_);
        state_["file/clip"] = {start().value_or(0) / format_desc_.fps, duration().value_or(0) / format_desc_.fps};
        state_["file/time"] = {time() / format_desc_.fps, file_duration().value_or(0) / format_desc_.fps};
        state_["loop"]      = loop_.load();
        state_["pingpong"]  = pingpong_.load();
        if (hdr10_seen_.load(std::memory_order_relaxed)) {
            // Queryable rather than only logged, so automation can read what a source
            // declares without scraping a log line.
            state_["file/hdr10/max-dml"]  = hdr10_max_dml_;
            state_["file/hdr10/min-dml"]  = hdr10_min_dml_;
            state_["file/hdr10/max-cll"]  = hdr10_max_cll_;
            state_["file/hdr10/max-fall"] = hdr10_max_fall_;
        }
    }

    core::draw_frame prev_frame(const core::video_field field)
    {
        CASPAR_SCOPE_EXIT { update_state(); };

        boost::lock_guard<boost::mutex> lock(buffer_mutex_);

        // Don't start a new frame on the 2nd field
        if (field != core::video_field::b) {
            if (frame_flush_ || !frame_) {
                if (!buffer_.empty()) {
                    frame_          = buffer_[0].frame;
                    frame_time_     = buffer_[0].pts;
                    frame_duration_ = buffer_[0].duration;
                    frame_flush_    = false;
                }
            }
        }

        return core::draw_frame::still(frame_);
    }

    bool is_ready()
    {
        boost::lock_guard<boost::mutex> lock(buffer_mutex_);
        return !buffer_.empty() || frame_;
    }

    core::draw_frame next_frame(const core::video_field field)
    {
        auto now = std::chrono::steady_clock::now();
        // frames_since_update_ is accumulated at actual consumption points below,
        // not here, so the fps counter reflects real file frames consumed per second
        // at any speed (e.g. speed=2 shows ~50fps on a 25fps channel).
        auto duration_sec = std::chrono::duration_cast<std::chrono::duration<double>>(now - last_fps_update_).count();

        if (duration_sec >= 1.0) {
            current_fps_ = (double)frames_since_update_ / duration_sec;
            frames_since_update_ = 0;
            last_fps_update_ = now;
        }

        // Periodic producer TIMING log every 5s
        diag_frames_++;
        {
            auto diag_elapsed = std::chrono::duration<double>(now - diag_start_).count();
            if (diag_elapsed >= 5.0 && diag_frames_ > 0) {
                CASPAR_LOG(trace) << print() << " TIMING: frames=" << diag_frames_
                                  << " underflows=" << diag_underflows_;
                diag_frames_ = 0;
                diag_underflows_ = 0;
                diag_start_ = now;
            }
        }

        CASPAR_SCOPE_EXIT { update_state(); };

        boost::lock_guard<boost::mutex> lock(buffer_mutex_);

        // ---- the resident range, if there is one -------------------------------------------
        if (cache_serving_ && !cache_can_serve()) {
            // A seek arrived, or LOOP/PINGPONG was turned off. The decoder has been parked
            // wherever it last was, so put it back where the cache left off instead of
            // resuming from somewhere unrelated.
            cache_serving_ = false;
            seek_          = frame_time_;
            buffer_.clear();
            buffer_cond_.notify_all();
        } else if (cache_can_serve()) {
            return serve_from_cache();
        }

        // When speed is negative and no frame has been produced yet, the decode
        // thread will have started buffering from the IN point.  Issue an initial
        // seek to the OUT point so reverse playback starts from the end of the clip.
        // Guard with !rev_active_ so we fire exactly once per reverse session.
        // rev_active_ is reset whenever playback direction changes forward.
        if (speed_.load() < 0.0 && !frame_ && rev_frames_.empty() && !rev_active_) {
            const int64_t  s0       = start_.load() != AV_NOPTS_VALUE ? start_.load() : 0LL;
            const int64_t  dur      = duration_.load();
            const int64_t  indur    = input_duration_.load();
            int64_t        end_pts  = AV_NOPTS_VALUE;
            if (dur != AV_NOPTS_VALUE && dur > 0)
                end_pts = s0 + dur;
            else if (indur != AV_NOPTS_VALUE && indur > 0)
                end_pts = indur;
            if (end_pts != AV_NOPTS_VALUE) {
                // Seek buffer_capacity_ frames before the exclusive end so the
                // decoder fills a complete batch rather than producing 0-1 frames
                // immediately before EOF.
                const int64_t one_frame = av_rescale_q(1, format_tb_, TIME_BASE_Q);
                const int64_t seek_pos  = std::max(s0, end_pts - static_cast<int64_t>(buffer_capacity_) * one_frame);
                rev_active_ = true;
                seek_ = seek_pos;
                buffer_.clear();
                buffer_cond_.notify_all();
            }
            return core::draw_frame{};
        }

        // Pre-roll threshold: hold output until this many frames are buffered
        // after a seek/loop before releasing to the first consumer call.
        // With flush-in-place the first decoded frame is correct, so 2 frames
        // of headroom is enough to absorb A/V sync jitter without the ~80ms
        // extra stall that the old value of 4 caused at 25fps.
        // In reverse mode each backward seek only produces 1 usable frame before
        // hitting EOF again, so the 2-frame pre-roll requirement is bypassed.
        const bool in_reverse     = speed_.load() < 0.0;
        const bool have_rev_cache = in_reverse && !rev_frames_.empty();

        // Eagerly consume the first post-seek frame before the pre-roll check.
        // This is necessary for paused playback (speed=0): frames_to_advance is
        // always 0 so the normal consumption path is never reached, which means
        // frame_flush_ is never cleared and frame_ is never updated.
        // By consuming here we also handle the EOF-with-1-frame case (buffer_eof_=true
        // but only 1 frame decoded so the size<2 pre-roll guard would otherwise stall).
        //
        // GATED on the two cases the comment above actually describes. Ungated, this block runs
        // on every seek and every loop wrap DURING NORMAL PLAYBACK too -- and because it pops
        // without returning, execution falls through to the normal consumption path below, which
        // pops again. The frame taken here is then overwritten before it is ever shown, so one
        // frame is silently eaten per discontinuity.
        //
        // MEASURED 2026-08-19 from the picture: `SEEK 40 LENGTH 8 LOOP` showed markers
        // {41..47} and never 40; from IN 20 it showed {21..27} and never 20 -- always exactly
        // the IN frame, independent of IN and LENGTH. A trace of the decode loop proved the
        // frame was not the producer's fault: all eight frames were pushed to buffer_ on every
        // iteration, pts 1600000..1880000, so the loss was entirely on the consumer side.
        //
        // speed == 0: frames_to_advance is always 0, the normal path never runs, and without
        //             this frame_flush_ is never cleared and frame_ never updated -- the
        //             original reason for the block.
        // buffer_eof_: the EOF-with-one-frame case, where the 2-frame pre-roll guard below would
        //             otherwise stall forever. Also from the original comment.
        if (frame_flush_ && !buffer_.empty() && !in_reverse &&
            (speed_.load() == 0.0 || buffer_eof_)) {
            frame_          = buffer_[0].frame;
            frame_time_     = buffer_[0].pts;
            frame_duration_ = buffer_[0].duration;
            frame_flush_    = false;
            buffer_.pop_front();
            buffer_cond_.notify_all();
        }

        // For forward play: 2-frame pre-roll to absorb pipeline jitter.
        // For reverse play: strict full batch pre-roll (buffer_capacity_) to maximize caching and avoid micro-seeks,
        //                   but bypass if EOF is reached and the batch ends prematurely.
        const bool reverse_req_met = in_reverse && (buffer_.size() >= static_cast<size_t>(buffer_capacity_) || buffer_eof_);
        const bool drop_to_wait    = !have_rev_cache && (buffer_.empty() || (!in_reverse && frame_flush_ && buffer_.size() < 2) || (in_reverse && !reverse_req_met));

        if (drop_to_wait) {
            auto start    = start_.load();
            auto duration = duration_.load();

            start    = start != AV_NOPTS_VALUE ? start : 0;
            auto end = duration != AV_NOPTS_VALUE ? start + duration : INT64_MAX;

            if ((buffer_eof_ || growing_) && !frame_flush_ && !in_reverse) {
                if (pingpong_ && speed_.load() > 0.0) {
                    // Forward playback hit OUT point — flip to reverse.
                    // Seek buffer_capacity_ frames before the current position so
                    // the first reverse batch is full rather than 0-1 frames.
                    const double  spd_abs   = std::abs(speed_.load());
                    const int64_t one_frame = av_rescale_q(1, format_tb_, TIME_BASE_Q);
                    const int64_t s0        = start_.load() != AV_NOPTS_VALUE ? start_.load() : 0LL;
                    speed_       = -spd_abs;
                    speed_accum_ = 0.0;
                    rev_frames_.clear();
                    // The OUT frame has just been shown forward, so the reverse batch starts
                    // BELOW it and each endpoint appears exactly once per sweep. Without this
                    // the trace reads `... 26 27 r27 r26 ...` -- OUT twice, which the CUDA
                    // producers do not do (measured: `... 25 26 25 ...`).
                    rev_batch_top_ = frame_time_;
                    rev_active_  = true;   // seek is issued here, no need for initial-seek guard
                    buffer_eof_  = false;

                    // Seed the reverse sweep from frames already shown, so the turn costs no seek
                    // and no decode. MEASURED before this existed: the OUT frame was DISPLAYED
                    // twice at 17 of 19 turnarounds -- the last forward tick delivered OUT and
                    // this tick returned it again as a still while the reverse batch decoded.
                    // Delivery order was already correct; what was wrong was the dwell.
                    int64_t ring_low = frame_time_;
                    for (const auto& sf : shown_) {
                        if (sf.pts < frame_time_) {   // never re-serve OUT: endpoint once per sweep
                            rev_frames_.push_back(sf);
                            ring_low = std::min(ring_low, sf.pts);
                        }
                    }
                    if (!rev_frames_.empty())
                        rev_batch_top_ = ring_low;   // the batch below must not replay the ring

                    const int64_t seek_pos = std::max(s0,
                        rev_batch_top_ - static_cast<int64_t>(buffer_capacity_ - 1) * one_frame);
                    seek_ = seek_pos;
                    buffer_.clear();
                    buffer_cond_.notify_all();

                    // Deliberately a small inline serve rather than pop_rev_frame(), which is
                    // declared further down in this function and carries the IN-crossing logic.
                    // So the fast path is taken only when the frame served cannot be the one that
                    // crosses IN; anything nearer than that falls through to the still and is
                    // handled next tick by the normal path, which is the measured one.
                    if (!rev_frames_.empty()) {
                        const auto& nf = rev_frames_.back();
                        if (nf.duration > 0 && nf.pts - nf.duration >= s0) {
                            frame_          = nf.frame;
                            frame_time_     = nf.pts;
                            frame_duration_ = nf.duration;
                            rev_frames_.pop_back();
                            graph_->set_value("buffer",
                                              static_cast<double>(rev_frames_.size()) /
                                                  static_cast<double>(buffer_capacity_));
                            return frame_;
                        }
                    }
                    return core::draw_frame::still(frame_);
                }
                if (frame_time_ < end && frame_duration_ != AV_NOPTS_VALUE) {
                    frame_time_ += frame_duration_;
                } else if (frame_time_ < end) {
                    frame_time_ = input_duration_;
                }
                return core::draw_frame::still(frame_);
            }
            // Hold the last decoded frame as a still rather than going black during:
            //  - reverse-mode buffer stalls after each backward seek
            //  - forward seek pre-roll (frame_flush_=true, buffer not yet full)
            //  - seeking past the last video frame (buffer_eof_ with frame_flush_ still set)
            // Only signal a true underflow (black) if no frame has ever been decoded.
            if (frame_) {
                return core::draw_frame::still(frame_);
            }
            graph_->set_tag(diagnostics::tag_severity::WARNING, "underflow");
            latency_ += 1;
            diag_underflows_++;
            return core::draw_frame{};
        }

        if (!in_reverse && format_desc_.field_count == 2) {
            // Check if the next frame is the correct 'field'
            auto is_field_1 = (buffer_[0].frame_count % 2) == 0;
            if ((field == core::video_field::a && !is_field_1) || (field == core::video_field::b && is_field_1)) {
                graph_->set_tag(diagnostics::tag_severity::WARNING, "underflow");
                latency_ += 1;
                diag_underflows_++;
                return core::draw_frame{};
            }
        }

        if (latency_ != -1) {
            CASPAR_LOG(warning) << print() << " Latency: " << latency_;
            latency_ = -1;
        }

        // Speed control: accumulate fractional rate each output tick.
        //   speed > 1.0  => skip intermediate decoded frames (fast forward)
        //   0 < speed < 1 => hold frames (slow motion)
        //   speed == 0  => freeze
        //   speed < 0   => reverse playback (batch mode):
        //                  Collects a buffer-full of decoded frames then serves them
        //                  in reverse order, issuing the next batch seek in the
        //                  background.  This amortises the keyframe-seek overhead
        //                  across buffer_capacity_ frames, giving smooth reverse on
        //                  both intra-only and long-GOP (H.264/HEVC) sources.
        {
            const double spd = speed_.load();

            if (spd < 0.0) {
                // --- Reverse (batch) ---
                speed_accum_ += -spd;
                const int frames_to_step = static_cast<int>(speed_accum_);
                speed_accum_ -= static_cast<double>(frames_to_step);

                if (frames_to_step == 0) {
                    // Slow-motion hold: no file frame consumed this tick.
                    return core::draw_frame::still(frame_);
                }
                // file frames consumed this tick (1 returned + frames_to_step-1 skipped)
                frames_since_update_ += frames_to_step;

                const int64_t start_l = start_.load() != AV_NOPTS_VALUE ? start_.load() : 0LL;

                // Helper: consume the next frame from rev_frames_ and handle IN-point
                // boundary (pingpong / loop / freeze).
                auto pop_rev_frame = [&]() {
                    // Fast-reverse: skip intermediate frames when |speed| > 1
                    for (int i = 1; i < frames_to_step && rev_frames_.size() > 1; ++i)
                        rev_frames_.pop_back();

                    frame_          = rev_frames_.back().frame;
                    frame_time_     = rev_frames_.back().pts;
                    frame_duration_ = rev_frames_.back().duration;
                    retain_shown(rev_frames_.back());
                    rev_frames_.pop_back();

                    graph_->set_value("buffer", static_cast<double>(rev_frames_.size()) /
                                                    static_cast<double>(buffer_capacity_));

                    // Check if the next step would cross the IN point
                    if (frame_duration_ > 0 && frame_time_ - frame_duration_ < start_l) {
                        if (pingpong_) {
                            speed_       = std::abs(speed_.load());
                            speed_accum_ = 0.0;
                            rev_frames_.clear();
                            rev_batch_top_ = INT64_MAX;
                            // The IN frame has just been shown in reverse, so forward resumes
                            // at the NEXT one -- the same endpoint-once rule as the OUT end.
                            // Falls back to start_l for a single-frame range, where there is no
                            // next one and seeking past OUT would leave the range entirely.
                            const auto    dur_pp = duration_.load();
                            const int64_t end_pp = dur_pp != AV_NOPTS_VALUE ? start_l + dur_pp : INT64_MAX;
                            int64_t       resume = frame_duration_ > 0 ? start_l + frame_duration_ : start_l;
                            if (resume >= end_pp)
                                resume = start_l;
                            seek_        = resume;
                            buffer_.clear();
                            buffer_cond_.notify_all();
                        } else if (loop_) {
                            const auto    dur     = duration_.load();
                            // Seek buffer_capacity_ frames before the clip end for
                            // a full first reverse batch.
                            const int64_t one_f   = av_rescale_q(1, format_tb_, TIME_BASE_Q);

                            const int64_t end_abs = dur != AV_NOPTS_VALUE
                                                        ? start_l + dur
                                                        : (input_duration_.load() != AV_NOPTS_VALUE
                                                               ? input_duration_.load() : 0LL);
                            const int64_t seek_pos = std::max(start_l,
                                end_abs - static_cast<int64_t>(buffer_capacity_) * one_f);
                            rev_frames_.clear();
                            rev_batch_top_ = INT64_MAX;
                            seek_ = seek_pos;
                            buffer_.clear();
                            buffer_cond_.notify_all();
                        }
                        // else: freeze — rev_frames_ drains, pre-roll holds still
                    }
                };

                // --- Serve from existing reverse batch if available ---
                if (!rev_frames_.empty()) {
                    pop_rev_frame();
                    return frame_;
                }

                // --- rev_frames_ empty: capture the buffer as a new batch ---
                // (buffer is guaranteed non-empty because the pre-roll guard passed)
                // Capture batch in forward order; pop_back() serves highest PTS first.
                for (const auto& f : buffer_)
                    if (f.pts < rev_batch_top_)
                        rev_frames_.push_back(f);
                frame_flush_ = false;
                buffer_.clear();
                buffer_cond_.notify_all();

                if (rev_frames_.empty()) {
                    // Everything decoded had already been shown. Ask for the bottom of the
                    // range and hold the last frame. The frame at start_l is always below
                    // rev_batch_top_ whenever that bound is above start_l, so this cannot spin.
                    if (rev_batch_top_ != INT64_MAX && rev_batch_top_ > start_l) {
                        seek_ = start_l;
                        buffer_cond_.notify_all();
                    }
                    return core::draw_frame::still(frame_);
                }

                const int64_t batch_start_pts = rev_frames_.front().pts;
                const int64_t batch_start_dur = rev_frames_.front().duration;
                const int64_t batch_count     = static_cast<int64_t>(rev_frames_.size());

                // Pre-issue the next batch seek so the decode thread works in
                // parallel while we serve the current batch.
                // Seek BACK strictly by the actual batch_count frames from batch_start
                // so the next batch is fully BEFORE this one with no gap or overlap.
                if (batch_start_dur > 0 && batch_count > 0) {
                    const int64_t step_back_frames = (buffer_eof_ && batch_count < static_cast<int64_t>(buffer_capacity_))
                                                         ? static_cast<int64_t>(buffer_capacity_)
                                                         : batch_count;
                    // CLAMPED to the IN point, not skipped. Skipping is what stalled reverse
                    // pingpong dead: MEASURED 2026-08-19, `SEEK 20 LENGTH 8 PINGPONG` delivered
                    // 20..27 forward, then r27..r22, then NOTHING for the rest of the run. The
                    // batch below 22 starts at pts(16), under the IN point at pts(20), so no
                    // seek was issued at all -- rev_frames_ empty, buffer_ empty, nothing
                    // pending, permanent halt. Any pingpong or reverse loop whose remaining
                    // distance to IN is shorter than one batch ended there.
                    //
                    // Clamping makes that last short batch overlap the one just served, which
                    // is why rev_batch_top_ exists: it trims the overlap at capture. In the
                    // unclamped case the two are contiguous by construction and the trim is a
                    // no-op, so ordinary mid-clip reverse is untouched.
                    int64_t next_target = batch_start_pts - step_back_frames * batch_start_dur;
                    if (next_target < start_l)
                        next_target = start_l;
                    rev_batch_top_ = batch_start_pts;
                    if (batch_start_pts > start_l) {
                        seek_ = next_target;
                        buffer_cond_.notify_all();
                    }
                }

                pop_rev_frame();
                return frame_;
            }
// --- Forward ---
            speed_accum_ += spd;
            const int frames_to_advance = static_cast<int>(speed_accum_);
            speed_accum_ -= static_cast<double>(frames_to_advance);

            if (frames_to_advance == 0) {
                // Slow-motion hold: no file frame consumed this tick.
                return core::draw_frame::still(frame_);
            }

            // Fast-forward: discard (frames_to_advance - 1) intermediate frames,
            // tracking how many were actually available so fps is accurate when
            // the buffer underflows.
            int actually_consumed = 1;
            for (int i = 1; i < frames_to_advance && buffer_.size() > 1; ++i) {
                buffer_.pop_front();
                buffer_cond_.notify_all();
                ++actually_consumed;
            }
            frames_since_update_ += actually_consumed;
        }

        frame_          = buffer_[0].frame;
        frame_time_     = buffer_[0].pts;
        frame_duration_ = buffer_[0].duration;
        frame_flush_    = false;

        retain_shown(buffer_[0]);
        buffer_.pop_front();
        buffer_cond_.notify_all();

        graph_->set_value("buffer", static_cast<double>(buffer_.size()) / static_cast<double>(buffer_capacity_));

        // Normalised fps bar: 1.0 = decoder keeping up with requested speed.
        // Target is channel_fps * |speed|; bar fills to 1.0 when on target.
        {
            const double target_fps = format_desc_.fps * std::max(0.01, std::abs(speed_.load()));
            if (target_fps > 0.0 && current_fps_ > 0.0)
                graph_->set_value("fps", std::min(1.0, current_fps_ / target_fps));
        }

        return frame_;
    }

    void seek(int64_t time)
    {
        CASPAR_SCOPE_EXIT { update_state(); };

        int64_t target_pts = av_rescale_q(time, format_tb_, TIME_BASE_Q);

        // Clamp target_pts to valid range to prevent seeking beyond EOF (black screen)
        {
            const int64_t start_l = start_.load() != AV_NOPTS_VALUE ? start_.load() : 0LL;
            int64_t end_l = AV_NOPTS_VALUE;
            
            if (duration_.load() != AV_NOPTS_VALUE) {
                end_l = start_l + duration_.load();
            } else if (input_duration_.load() != AV_NOPTS_VALUE) {
                end_l = input_duration_.load();
            }

            if (end_l != AV_NOPTS_VALUE) {
                // If seeking exactly to or past the end, clamp to the last frame.
                // We estimate the last frame start by subtracting one frame duration.
                // If frame_duration_ is not set yet, use channel frame duration as best guess.
                const int64_t one_frame = frame_duration_ > 0 ? frame_duration_ : av_rescale_q(1, format_tb_, TIME_BASE_Q);
                if (target_pts >= end_l) {
                   target_pts = std::max(start_l, end_l - one_frame);
                }
            }
        }

        {
            boost::lock_guard<boost::mutex> lock(buffer_mutex_);
            buffer_.clear();
            rev_frames_.clear();
            // An explicit seek is the one case that really does invalidate the resident range:
            // the user is asking for somewhere else, and the frames held are somewhere they no
            // longer are. The loop wrap, which shares seek_internal() with this, must not.
            drop_cache();

            // Pre-announce the seek position so that time() / frame_number() reflect
            // the new position immediately — even on a paused layer where next_frame()
            // is never called to update frame_time_ from the decoded buffer.
            // target_pts (TIME_BASE_Q, relative to container start) is the same
            // coordinate space that next_frame() uses for frame_time_, so this is
            // exactly what next_frame() would set it to once it eventually runs.
            frame_time_ = target_pts;

            if (speed_.load() < 0.0) {
                // If the user seeks backward to a frame, we must actually seek the decode 
                // thread to the START of the upcoming reverse batch so the requested frame
                // is the first one served (the END of the forward decoded batch).
                const int64_t start_l = start_.load() != AV_NOPTS_VALUE ? start_.load() : 0LL;
                const int64_t one_frame = av_rescale_q(1, format_tb_, TIME_BASE_Q);
                seek_ = std::max(start_l, target_pts - static_cast<int64_t>(buffer_capacity_ - 1) * one_frame);
                rev_active_ = true;
            } else {
                seek_ = target_pts;
            }
            
            // Set frame_flush_ here (under the lock) so next_frame()'s eager-consume
            // path is primed before the decode thread runs seek_internal().
            // Without this, next_frame() sees frame_flush_=false + empty buffer and
            // silently returns still(old_frame); if seeks arrive faster than the
            // decode thread processes them, frame_flush_ never transitions to true
            // and the sought frame is never consumed even after it lands in the buffer.
            frame_flush_ = true;
            // Mark as active so the initial-reverse-seek guard doesn't fire and
            // override a user-issued seek position.
            if (speed_.load() < 0.0) {
                rev_active_ = true;
                // Important: Ensure we don't leave the consumer in a state where it thinks it's
                // finished or at EOF after a seek.
                buffer_eof_ = false;
            }
            buffer_cond_.notify_all();
            graph_->set_value("buffer", static_cast<double>(buffer_.size()) / static_cast<double>(buffer_capacity_));
        }
    }

    int64_t time() const
    {
        if (frame_time_ == AV_NOPTS_VALUE) {
            // TODO (fix) How to handle NOPTS case?
            return 0;
        }

        return av_rescale_q(frame_time_, TIME_BASE_Q, format_tb_);
    }

    void loop(bool loop)
    {
        CASPAR_SCOPE_EXIT { update_state(); };

        loop_ = loop;
    }

    bool loop() const { return loop_; }

    void start(int64_t start)
    {
        CASPAR_SCOPE_EXIT { update_state(); };
        start_ = av_rescale_q(start, format_tb_, TIME_BASE_Q);
    }

    std::optional<int64_t> start() const
    {
        auto start = start_.load();
        return start != AV_NOPTS_VALUE ? av_rescale_q(start, TIME_BASE_Q, format_tb_) : std::optional<int64_t>();
    }

    void duration(int64_t duration)
    {
        CASPAR_SCOPE_EXIT { update_state(); };

        duration_ = av_rescale_q(duration, format_tb_, TIME_BASE_Q);
    }

    std::optional<int64_t> duration() const
    {
        const auto duration = duration_.load();
        if (duration == AV_NOPTS_VALUE) {
            return {};
        }
        return av_rescale_q(duration, TIME_BASE_Q, format_tb_);
    }

    std::optional<int64_t> file_duration() const
    {
        const auto input_duration = input_duration_.load();
        if (input_duration == AV_NOPTS_VALUE) {
            return {};
        }
        return av_rescale_q(input_duration, TIME_BASE_Q, format_tb_);
    }

    /// Budget for `shown_`, in bytes rather than frames, so the depth scales itself with the
    /// raster instead of needing a threshold per resolution: about 6 frames at 1080p, 2 at 4K, and
    /// ZERO at 12288x6144, where one frame is ~300 MB. That is the intended answer for the 12K
    /// virtual-production assets -- they are played through cuda_prores, which has no turnaround
    /// hold to fix, and buying a smoother turn there for gigabytes of RAM would be a bad trade.
    /// Default budget, overridable with `<loop-cache-mb>`. 0 disables the cache entirely.

    static int64_t frame_bytes(const Frame& f)
    {
        if (!f.video)
            return 0;
        int64_t n = av_image_get_buffer_size(static_cast<AVPixelFormat>(f.video->format),
                                             f.video->width, f.video->height, 1);
        if (n <= 0) {
            // A hardware frame carries no host buffer size, so this fails for D3D11. What
            // retention actually holds there is the pair of extracted mixer textures, so
            // estimate NV12 at 16 bits per sample. Deliberately pessimistic: erring high only
            // shortens the ring, erring low would let it run past its budget.
            n = static_cast<int64_t>(f.video->width) * f.video->height * 3;
        }
        return n;
    }

    /// Remember a frame that has just been delivered forward. On the frame path, so it is a push
    /// and a bounded pop and nothing else.
    void retain_shown(const Frame& f)
    {
        // Only a bounded, repeating range can be replayed, and a growing input is not one.
        if (cache_budget_ <= 0 || growing_ || !(loop_.load() || pingpong_.load())) {
            if (!shown_.empty())
                drop_cache();
            return;
        }

        // The cache is only usable as a CONTIGUOUS run: an index step has to mean one frame.
        // A gap means frames were dropped or the position jumped, so the run restarts rather
        // than being quietly wrong by however many frames went missing.
        //
        // It can grow from EITHER end. Frames arrive ascending when playing forward and
        // DESCENDING when playing in reverse, and a reverse loop never delivers a forward frame
        // at all -- so a cache that only appended would never fill for `SPEED -1 LOOP`, which is
        // exactly one of the cases this exists to smooth.
        enum class grow
        {
            back,
            front,
            have_it,
            restart
        } how = grow::restart;

        if (shown_.empty()) {
            how = grow::back;
        } else if (f.pts >= shown_.front().pts && f.pts <= shown_.back().pts) {
            how = grow::have_it;   // replaying a frame already resident: not a discontinuity
        } else if (shown_.back().duration > 0 && f.pts == shown_.back().pts + shown_.back().duration) {
            how = grow::back;
        } else if (f.duration > 0 && f.pts + f.duration == shown_.front().pts) {
            how = grow::front;
        }

        if (how == grow::have_it)
            return;
        if (how == grow::restart) {
            drop_cache();
            how = grow::back;
        }

        if (cache_refused_)
            return;

        const int64_t sz = frame_bytes(f);

        // Decide ONCE, for the WHOLE range, before a single frame is held.
        //
        // Filling up to a cap and hoping the range fits is the wrong shape, and was measurably
        // wrong: a range larger than the budget filled to the cap, never reached both ends, never
        // completed, and therefore pinned memory that could never be served from. Reserving the
        // whole range up front means the cache either works or costs nothing -- and it is what
        // lets the shared allowance below be an actual bound rather than a per-layer suggestion.
        if (cache_reserved_ == 0) {
            const int64_t d_c  = duration_.load();
            const int64_t fdur = f.duration;
            if (sz <= 0 || fdur <= 0 || d_c == AV_NOPTS_VALUE || d_c <= 0) {
                cache_refused_ = true;   // no declared range, or nothing to measure it in
                return;
            }
            const int64_t frames = (d_c + fdur - 1) / fdur;
            const int64_t need   = frames * sz;

            if (need > cache_budget_) {
                cache_refused_ = true;
                CASPAR_LOG(debug) << print() << L" loop cache declined: the range needs "
                                  << (need >> 20) << L" MB, the per-producer budget is "
                                  << (cache_budget_ >> 20) << L" MB";
                return;
            }
            if (!loop_cache_reserve(need)) {
                cache_refused_ = true;
                CASPAR_LOG(info) << print() << L" loop cache declined: " << (need >> 20)
                                 << L" MB would exceed the server-wide loop-cache-total-mb of "
                                 << (loop_cache_total_limit() >> 20) << L" MB";
                return;
            }
            cache_reserved_ = need;
        }

        // Safety bound only: the reservation above already sized this, so exceeding it would mean
        // the range is not what it was measured to be.
        const int cap = static_cast<int>(cache_reserved_ / sz);

        // Retain the MIXER frame and drop the decoded AVFrame. This matters on both paths, for
        // different reasons:
        //
        //   * software -- the draw_frame already holds the mixer's own copy of the picture, so
        //     keeping the AVFrame beside it would double the cost of the ring for nothing;
        //   * GPU-direct -- the AVFrame is a D3D11 decoder surface from a BOUNDED DXVA pool.
        //     Holding a handful of those hostage would starve the hardware decoder, which is the
        //     real hazard on that path. It is NOT that a retained frame lacks its own picture:
        //     the shared y/uv pair is staging that every frame is copied OUT of, into textures
        //     from device::create_texture, whose pool only recycles them once the last reference
        //     drops. So the picture is genuinely owned -- it is the decode surface that must not
        //     be.
        //
        // The reverse path reads only .frame, .pts and .duration from these, so nothing
        // downstream misses the rest.
        Frame keep;
        keep.frame       = f.frame;
        keep.start_time  = f.start_time;
        keep.pts         = f.pts;
        keep.duration    = f.duration;
        keep.frame_count = f.frame_count;

        if (how == grow::front)
            shown_.push_front(std::move(keep));
        else
            shown_.push_back(std::move(keep));

        // Over budget: drop from the end OPPOSITE the one just extended, so the window keeps
        // moving with playback instead of eating the frames it is about to need.
        while (static_cast<int>(shown_.size()) > cap) {
            if (how == grow::front)
                shown_.pop_back();
            else
                shown_.pop_front();
            cache_complete_ = false;   // losing an end means the range is no longer whole
        }

        // Complete when the run spans the requested range. Contiguity is already guaranteed
        // above, so the two ends are enough to know every frame between them is present.
        const int64_t s_c = start_.load() != AV_NOPTS_VALUE ? start_.load() : 0LL;
        const int64_t d_c = duration_.load();
        if (d_c != AV_NOPTS_VALUE && d_c > 0 && !shown_.empty()) {
            cache_complete_ = shown_.front().pts <= s_c &&
                              shown_.back().pts + shown_.back().duration >= s_c + d_c;
        } else {
            // No declared duration means no known range to be complete over. A whole-file loop
            // is left to the decoder rather than guessed at.
            cache_complete_ = false;
        }
    }

    void drop_cache()
    {
        shown_.clear();
        cache_complete_ = false;
        cache_serving_  = false;
        cache_index_    = 0;
        loop_cache_release(cache_reserved_);
        cache_reserved_ = 0;
        // A new range may well fit where the last one did not, so the refusal is not permanent.
        // This runs on an explicit seek or when looping is turned off, never per frame.
        cache_refused_ = false;
    }

    bool cache_can_serve() const
    {
        return cache_complete_ && cache_budget_ > 0 && !growing_ &&
               seek_.load() == AV_NOPTS_VALUE && (loop_.load() || pingpong_.load());
    }

    /// Play the resident range directly. No decoder, no seek, and on the GPU-direct path no
    /// upload either -- the frames are already mixer textures.
    core::draw_frame serve_from_cache()
    {
        const int n = static_cast<int>(shown_.size());
        if (n <= 0)
            return core::draw_frame::still(frame_);

        if (!cache_serving_) {
            // Line the cursor up with the frame just delivered, so entering the cache is not
            // itself a jump.
            cache_index_ = 0;
            for (int i = 0; i < n; ++i) {
                if (shown_[i].pts == frame_time_) {
                    cache_index_ = i;
                    break;
                }
            }
            cache_serving_ = true;
            CASPAR_LOG(info) << print() << L" loop cache serving " << n
                              << L" frames; the decoder is idle for this range";
        }

        const double spd = speed_.load();
        speed_accum_ += spd;
        const int step = static_cast<int>(speed_accum_);
        speed_accum_ -= static_cast<double>(step);
        if (step == 0)
            return core::draw_frame::still(frame_);   // slow motion: no new frame this tick

        int idx = cache_index_;
        for (int i = 0; i < std::abs(step); ++i) {
            idx += step > 0 ? 1 : -1;
            if (idx >= n) {
                if (pingpong_.load() && n >= 2) {
                    speed_ = -std::abs(spd);
                    idx    = n - 2;      // endpoint once per sweep, as everywhere else
                } else {
                    idx = 0;             // loop wrap
                }
            } else if (idx < 0) {
                if (pingpong_.load() && n >= 2) {
                    speed_ = std::abs(spd);
                    idx    = 1;
                } else {
                    idx = n - 1;         // reverse loop wrap
                }
            }
        }
        idx = std::max(0, std::min(n - 1, idx));

        cache_index_    = idx;
        frame_          = shown_[idx].frame;
        frame_time_     = shown_[idx].pts;
        frame_duration_ = shown_[idx].duration;
        frames_since_update_ += std::abs(step);

        graph_->set_value("buffer", 1.0);
        return frame_;
    }

    void speed(double spd)
    {
        const double old = speed_.exchange(spd);
        // When switching direction, discard any stale reverse batch so the next
        // call to next_frame() starts fresh in the new direction.
        if ((old < 0.0) != (spd < 0.0)) {
            boost::lock_guard<boost::mutex> lock(buffer_mutex_);
            rev_frames_.clear();
            rev_batch_top_ = INT64_MAX;
            // The cache is NOT dropped on a direction change: it holds a RANGE, and
            // reversing through that range needs exactly the frames it already has.
            rev_active_ = false;  // reset so initial-seek fires for new reverse session
            speed_accum_ = 0.0;
            
            if (spd < 0.0) {
                // Dynamically inflate the decode caching limit specifically for reverse mode.
                // Buffering a massive amount of sequential frames drastically offsets
                // keyframe-seek overhead and provides fully seamless backward sweeps.
                const int fps = static_cast<int>(format_desc_.fps);
                buffer_capacity_ = (fps > 0) ? fps : 30;
                if (buffer_capacity_ > 60) buffer_capacity_ = 60;
                if (buffer_capacity_ < 15) buffer_capacity_ = 15;
            } else {
                // Restore to the configured live forward latency minimum
                buffer_capacity_ = std::max(1, static_cast<int>(format_desc_.fps) / 4);
            }
        }
    }
    double speed() const { return speed_.load(); }

    void pingpong(bool pp)
    {
        CASPAR_SCOPE_EXIT { update_state(); };
        pingpong_ = pp;
    }
    bool pingpong() const { return pingpong_.load(); }

  private:
    bool want_packet()
    {
        return std::any_of(decoders_.begin(), decoders_.end(), [](auto& p) { return p.second.want_packet(); });
    }

    bool schedule()
    {
        auto result = false;

        std::shared_ptr<AVPacket> packet;
        while (want_packet() && input_.try_pop(packet)) {
            result = true;

            if (!packet) {
                for (auto& p : decoders_) {
                    p.second.push(nullptr);
                }
            } else {
                // sources_ is the filter-graph routing table, and it was also
                // the only thing deciding which decoder a packet reaches. The
                // GPU-direct video decoder is deliberately absent from it --
                // reset() skips registering it because its frames go straight
                // to the mixer -- so after the first seek it was fed nothing
                // and every video packet was dropped on the floor.
                //
                // It worked until the first seek only by accident of ordering:
                // the initial reset() runs before gpu_direct_video_ is set, so
                // the video stream was registered that one time. Every later
                // reset() -- and a loop wrap is one -- dropped it.
                //
                // What that looked like: the clip played to the end, wrapped to
                // 0.00 and froze there for ever, on both mixers, with nothing
                // in the log. Nothing was logged because the run loop still saw
                // progress (schedule() kept popping and discarding packets)
                // right up until the input hit real end-of-file, at which point
                // frame_count_ was still 0 and the `frame_count_ > 2` guard
                // turned the loop wrap off. An explicit mid-clip SEEK froze the
                // same way, which is what ruled out anything loop-specific.
                bool routed = sources_.find(packet->stream_index) != sources_.end();
#ifdef _WIN32
                // Route by decoder, not by filter source, for the one decoder
                // that has no filter source by design.
                routed = routed || (gpu_direct_video_ && packet->stream_index == gpu_direct_decoder_idx_);
#endif
                if (routed) {
                    auto it = decoders_.find(packet->stream_index);
                    if (it != decoders_.end()) {
                        // TODO (fix): limit it->second.input.size()?
                        it->second.push(std::move(packet));
                    }
                }
            }
        }

        std::vector<int> eof;

        for (auto& p : sources_) {
            auto it = decoders_.find(p.first);
            if (it == decoders_.end()) {
                continue;
            }

            auto nb_requests = 0U;
            for (auto source : p.second) {
                nb_requests = std::max(nb_requests, av_buffersrc_get_nb_failed_requests(source));
            }

            if (nb_requests == 0) {
                continue;
            }

            auto frame = it->second.pop();
            if (!frame) {
                continue;
            }

            // Update stream-level color metadata from the first decoded frame
            // that carries valid properties.  Codecs like ProRes store color
            // information in the bitstream frame header rather than the container,
            // so codecpar fields are UNSPECIFIED while decoded frames have correct
            // values.  Updating the fallback here ensures get_color_space() returns
            // the right matrix even if the filter graph strips per-frame metadata.
            if (frame->colorspace != AVCOL_SPC_UNSPECIFIED && stream_color_space_ == AVCOL_SPC_UNSPECIFIED) {
                stream_color_space_ = static_cast<AVColorSpace>(frame->colorspace);
            }
            if (frame->color_trc != AVCOL_TRC_UNSPECIFIED && stream_color_trc_ == AVCOL_TRC_UNSPECIFIED) {
                stream_color_trc_ = static_cast<AVColorTransferCharacteristic>(frame->color_trc);
            }
            if (frame->chroma_location != AVCHROMA_LOC_UNSPECIFIED && stream_chroma_loc_ == AVCHROMA_LOC_UNSPECIFIED) {
                stream_chroma_loc_ = static_cast<AVChromaLocation>(frame->chroma_location);
            }

            for (auto& source : p.second) {
                if (!frame->data[0]) {
                    FF(av_buffersrc_close(source, frame->pts, 0));
                } else {
                    // TODO (fix) Guard against overflow?
                    FF(av_buffersrc_write_frame(source, frame.get()));
                }
                result = true;
            }

            // End Of File
            if (!frame->data[0]) {
                eof.push_back(p.first);
            }
        }

        for (auto index : eof) {
            sources_.erase(index);
        }

        return result;
    }

    void seek_internal(int64_t time)
    {
        time = time != AV_NOPTS_VALUE ? time : 0;

#ifdef _WIN32
        // A seek lands on the keyframe at or before the target, so something has
        // to discard the frames between the two. On the software path the filter
        // graph does it: reset() rebuilds the spec with `fps=...:start_time=`,
        // which drops everything before the target. GPU-direct frames never
        // enter that graph, so their only equivalent is the run loop's
        // drop-to-target, and that runs off current_seek_target_.
        //
        // Only the explicit-seek call site set it. The loop wrap did not, so a
        // clip looping on an IN point that is not a keyframe restarted from the
        // keyframe on every wrap -- `PLAY ... SEEK 40 LENGTH 1 LOOP` on a clip
        // keyframed every 25 sat on frames 27-28 instead of 40, varying with
        // timing. Setting it here covers every caller rather than one, which is
        // the same reason the trim lives in reset() on the software side.
        //
        // Gated: on the software path the fps filter already trims, and the
        // explicit-seek call site sets this unconditionally as it always has,
        // so software behaviour is untouched.
        //
        // `|| gpu_direct_decode_requested()`, because gpu_direct_video_ is not known yet the one
        // time it matters most. The INITIAL seek runs here before the decoder has emitted a
        // hardware surface, so the flag is still false, the target is never set, and nothing
        // drops the keyframe pre-roll: MEASURED 2026-08-20, `PLAY ... SEEK 20 LENGTH 8` on h264
        // with GPU-direct delivered frames 0..19 to the screen once at startup -- about 0.8 s of
        // the wrong content -- while every one of the 28 loop wraps after it was a clean 20..27,
        // because by then the flag was true.
        //
        // The config flag is known from the first call, so it stands in for "GPU-direct may
        // engage". If it was asked for and then declines, this sets the target on a software
        // path -- which is exactly what the explicit-seek call site has always done
        // unconditionally, so that combination is already exercised rather than new.
        if (gpu_direct_video_ || gpu_direct_decode_requested())
            current_seek_target_ = time;
#endif

        time = time + (input_->start_time != AV_NOPTS_VALUE ? input_->start_time : 0);

        // TODO (fix) Dont seek if time is close future.
        if (seekable_) {
            input_.seek(time);
        }
        frame_flush_ = true;
        frame_count_ = 0;
        buffer_eof_  = false;

        // Flush decoders in-place: keeps threads and codec contexts alive.
        // avcodec_flush_buffers on intra-only codecs (ProRes, NotchLC) is near-instant;
        // the very next packet produces a decoded frame with no pipeline warmup.
        // H.264/HEVC GPU decoders also stay warm through the flush.
        for (auto& [idx, dec] : decoders_) {
            dec.flush();
        }

        // Clear stale buffered frames so playback jumps to the loop start immediately
        // instead of continuing to drain pre-loop frames from the buffer.
        {
            boost::lock_guard<boost::mutex> buffer_lock(buffer_mutex_);
            buffer_.clear();
            // The frame-advance accumulator has to go with them. It counts progress towards
            // the next frame of a timeline position that no longer applies, and the buffer it
            // was counting against has just been emptied -- so carrying it over makes the first
            // tick after the discontinuity advance further than the speed asked for.
            //
            // Here rather than at the call sites, for the same reason current_seek_target_ is
            // set here (see the note above): this function is the common path for the explicit
            // seek, the loop wrap and the initial start, and only the pingpong flips and
            // speed() were resetting it. The loop wrap was the one that got forgotten, which is
            // the same omission that comment records.
            speed_accum_ = 0.0;
            // Same argument for the retained run: it is a contiguous stretch ending where playback
            // WAS, and after a seek or a loop wrap it no longer adjoins where playback resumes, so
            // seeding a turnaround from it would serve frames from the wrong part of the clip.
            // The CACHE is deliberately NOT dropped here, and that is the whole difference
            // between this working and not. seek_internal() is the common path for the explicit
            // seek, the LOOP WRAP and the initial start -- and the wrap runs on the DECODE
            // thread, which is ahead of the consumer. Dropping the cache here wiped it every
            // time round the loop, while the consumer was still mid-range, so the resident run
            // could never reach both ends and the cache never completed. MEASURED: it sat at 2-7
            // frames of an 8-frame range forever.
            //
            // This is the mirror image of speed_accum_ directly above, which belongs here
            // precisely BECAUSE every caller invalidates it. A user seek does invalidate the
            // cache, so that drop lives at the explicit-seek call site in seek() instead.
            buffer_cond_.notify_all();
        }

        reset(time);
    }

    void reset(int64_t start_time)
    {
        // Discard animated parameters — their AVFilterContext* pointers are
        // about to be invalidated by the graph rebuild.
        {
            boost::lock_guard<boost::mutex> lock(param_tween_mutex_);
            video_param_tweens_.clear();
            audio_param_tweens_.clear();
        }

        video_filter_ =
            Filter(vfilter_, input_, decoders_, start_time, AVMEDIA_TYPE_VIDEO, format_desc_, decode_adapter_,
                   vk_mixer_device_);
        audio_filter_ = Filter(afilter_, input_, decoders_, start_time, AVMEDIA_TYPE_AUDIO, format_desc_);

        sources_.clear();
#ifdef _WIN32
        if (!gpu_direct_video_) {
#endif
            for (auto& p : video_filter_.sources) {
                sources_[p.first].push_back(p.second);
            }
#ifdef _WIN32
        }
#endif
        for (auto& p : audio_filter_.sources) {
            sources_[p.first].push_back(p.second);
        }

        std::vector<int> keys;
        // Flush unused inputs.
        for (auto& p : decoders_) {
            if (sources_.find(p.first) == sources_.end()) {
#ifdef _WIN32
                // The GPU-direct video decoder has no filter-graph source on
                // purpose -- see the guard above, which skips registering it
                // because the frames go straight from the decoder to the mixer.
                // That makes it look unused here, so it was being erased out
                // from under the reader, and the decode loop then threw
                // "invalid map<K, T> key" from decoders_.at() on every
                // iteration for the life of the producer. It is in use; skip it.
                if (gpu_direct_video_ && p.first == gpu_direct_decoder_idx_)
                    continue;
#endif
                keys.push_back(p.first);
            }
        }

        for (auto& key : keys) {
            decoders_.erase(key);
        }
    }

    // -------------------------------------------------------------------------
    // VFPARAM / AFPARAM: per-frame tween tick + avfilter_graph_send_command
    // Called once per produced frame from the decode thread (run loop), after
    // both video and audio filter futures have been awaited, so no concurrent
    // filter graph access is in flight at that point.
    // -------------------------------------------------------------------------
    void apply_filter_param_tweens()
    {
        boost::lock_guard<boost::mutex> lock(param_tween_mutex_);
        char res_buf[512];

        auto apply = [&](std::map<std::string, std::map<std::string, FilterParamTween>>& tweens,
                         AVFilterGraph*                                                   fgraph) {
            if (!fgraph)
                return;
            for (auto& filter_entry : tweens) {
                const auto& fname = filter_entry.first;
                for (auto& param_entry : filter_entry.second) {
                    const auto& pname = param_entry.first;
                    auto&       tween = param_entry.second;

                    tween.tick();
                    const auto val     = tween.fetch();
                    const auto val_str = std::to_string(val);

                    const auto ret = avfilter_graph_send_command(
                        fgraph, fname.c_str(), pname.c_str(), val_str.c_str(), res_buf, sizeof(res_buf), 0);

                    if (ret < 0 && ret != AVERROR(ENOSYS)) {
                        constexpr size_t errbuf_size = 128;
                        char errbuf[errbuf_size];
                        av_strerror(ret, errbuf, errbuf_size);
                        CASPAR_LOG(warning)
                            << "[ffmpeg] VFPARAM send_command(" << fname << ", " << pname << "=" << val_str
                            << ") failed: " << errbuf;
                    }
                }
            }
        };

        apply(video_param_tweens_, video_filter_.graph.get());
        apply(audio_param_tweens_, audio_filter_.graph.get());
    }

  public:
    void set_filter_param(bool                is_video,
                          const std::string&  filter_name,
                          const std::string&  param_name,
                          double              value,
                          int                 duration_frames,
                          const std::wstring& tween_name)
    {
        boost::lock_guard<boost::mutex> lock(param_tween_mutex_);
        auto& tweens = is_video ? video_param_tweens_ : audio_param_tweens_;
        tweens[filter_name][param_name].set_target(value, duration_frames, tween_name);
    }
    
    // -------------------------------------------------------------------------
    
    void set_vfilter(const std::string& filter)
    {
        vfilter_ = filter;
        seek(time());
    }

    void set_afilter(const std::string& filter)
    {
        afilter_ = filter;
        seek(time());
    }

    // -------------------------------------------------------------------------

    std::string print() const
    {
        const int          position = std::max(static_cast<int>(time() - start().value_or(0)), 0);
        std::ostringstream str;
        str << std::fixed << std::setprecision(4) << "ffmpeg[" << name_ << "|"
            << av_q2d({position * format_tb_.num, format_tb_.den}) << "/"
            << av_q2d({static_cast<int>(duration().value_or(0LL)) * format_tb_.num, format_tb_.den}) << "]";
        return str.str();
    }
};

AVProducer::AVProducer(std::shared_ptr<core::frame_factory> frame_factory,
                       core::video_format_desc              format_desc,
                       std::string                          name,
                       std::string                          path,
                       std::optional<std::string>           vfilter,
                       std::optional<std::string>           afilter,
                       std::optional<int64_t>               start,
                       std::optional<int64_t>               seek,
                       std::optional<int64_t>               duration,
                       std::optional<bool>                  loop,
                       int                                  seekable,
                       core::frame_geometry::scale_mode     scale_mode,
                       bool                                 growing)
    : impl_(new Impl(std::move(frame_factory),
                     std::move(format_desc),
                     std::move(name),
                     std::move(path),
                     std::move(vfilter.value_or("")),
                     std::move(afilter.value_or("")),
                     std::move(start),
                     std::move(seek),
                     std::move(duration),
                     loop.value_or(false),
                     seekable,
                     scale_mode,
                     growing))
{
}

AVProducer& AVProducer::alpha_declaration(int declaration)
{
    impl_->alpha_declaration_ = declaration;
    // Keep the effective default consistent with an explicit declaration, so a frame that
    // says nothing and a stream that never speaks both land on the operator's answer.
    const auto d = static_cast<core::alpha_declaration>(declaration);
    if (d == core::alpha_declaration::premultiplied)
        impl_->straight_alpha_ = false;
    else if (d == core::alpha_declaration::straight)
        impl_->straight_alpha_ = true;
    return *this;
}

AVProducer& AVProducer::straight_alpha(bool straight)
{
    impl_->straight_alpha_ = straight;
    return *this;
}

core::draw_frame AVProducer::next_frame(const core::video_field field) { return impl_->next_frame(field); }

core::draw_frame AVProducer::prev_frame(const core::video_field field) { return impl_->prev_frame(field); }

bool AVProducer::is_ready() { return impl_->is_ready(); }

AVProducer& AVProducer::seek(int64_t time)
{
    impl_->seek(time);
    return *this;
}

AVProducer& AVProducer::loop(bool loop)
{
    impl_->loop(loop);
    return *this;
}

bool AVProducer::loop() const { return impl_->loop(); }

AVProducer& AVProducer::start(int64_t start)
{
    impl_->start(start);
    return *this;
}

int64_t AVProducer::time() const { return impl_->time(); }

int64_t AVProducer::start() const { return impl_->start().value_or(0); }

AVProducer& AVProducer::duration(int64_t duration)
{
    impl_->duration(duration);
    return *this;
}

int64_t AVProducer::duration() const { return impl_->duration().value_or(std::numeric_limits<int64_t>::max()); }

AVProducer& AVProducer::set_vfilter(const std::string& filter)
{
    impl_->set_vfilter(filter);
    return *this;
}

AVProducer& AVProducer::set_afilter(const std::string& filter)
{
    impl_->set_afilter(filter);
    return *this;
}

AVProducer& AVProducer::set_filter_param(bool                is_video,
                                         const std::string&  filter_name,
                                         const std::string&  param_name,
                                         double              value,
                                         int                 duration_frames,
                                         const std::wstring& tween)
{
    impl_->set_filter_param(is_video, filter_name, param_name, value, duration_frames, tween);
    return *this;
}

AVProducer& AVProducer::speed(double spd)
{
    impl_->speed(spd);
    return *this;
}

double AVProducer::speed() const { return impl_->speed(); }

AVProducer& AVProducer::pingpong(bool pp)
{
    impl_->pingpong(pp);
    return *this;
}

bool AVProducer::pingpong() const { return impl_->pingpong(); }

core::monitor::state AVProducer::state() const
{
    boost::lock_guard<boost::mutex> lock(impl_->state_mutex_);
    return impl_->state_;
}

}} // namespace caspar::ffmpeg

