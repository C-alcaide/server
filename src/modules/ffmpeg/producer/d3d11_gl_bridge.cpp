#include "../StdAfx.h"
#include "d3d11_gl_bridge.h"

#ifdef _WIN32

#include <d3d11.h>
#include <dxgi1_2.h>
#include <GL/glew.h>
#include <GL/wglew.h>

#include <accelerator/ogl/util/device.h>
#include <accelerator/ogl/util/texture.h>

extern "C" {
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_d3d11va.h>
}

namespace caspar { namespace ffmpeg {

struct d3d11_gl_bridge::impl
{
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
    HANDLE interop_device_ = nullptr;
    HANDLE interop_object_ = nullptr;
    GLuint interop_gl_tex_ = 0;

    // GL context for the bridge thread
    HWND   dummy_hwnd_  = nullptr;
    HDC    dc_          = nullptr;
    HGLRC  gl_ctx_      = nullptr;

    int width_  = 0;
    int height_ = 0;
    bool active_ = false;

    bool init(AVBufferRef* hw_device_ctx, void* ogl_device_ptr)
    {
        if (!hw_device_ctx || !ogl_device_ptr)
            return false;

        auto* ogl_dev = static_cast<accelerator::ogl::device*>(ogl_device_ptr);

        // Get the D3D11 device from FFmpeg's hw_device_ctx
        auto* hwctx = reinterpret_cast<AVHWDeviceContext*>(hw_device_ctx->data);
        if (!hwctx || hwctx->type != AV_HWDEVICE_TYPE_D3D11VA)
            return false;
        auto* d3d11_hwctx = static_cast<AVD3D11VADeviceContext*>(hwctx->hwctx);
        d3d11_device_ = d3d11_hwctx->device;
        d3d11_ctx_    = d3d11_hwctx->device_context;
        if (!d3d11_device_ || !d3d11_ctx_)
            return false;

        // Load WGL_NV_DX_interop2 functions via GLEW (must have GL context current)
        // We'll use the global wglDXOpenDeviceNV etc. from wglew.h

        // Create shared GL context for the bridge
        void* mixer_hglrc = ogl_dev->native_gl_context();
        if (!mixer_hglrc)
            return false;

        WNDCLASSA wc   = {};
        wc.lpfnWndProc = DefWindowProcA;
        wc.hInstance   = GetModuleHandle(nullptr);
        wc.lpszClassName = "CasparCG_FFmpeg_D3D11GL";
        RegisterClassA(&wc);

        dummy_hwnd_ = CreateWindowA("CasparCG_FFmpeg_D3D11GL", "", 0, 0, 0, 1, 1,
                                    nullptr, nullptr, wc.hInstance, nullptr);
        dc_ = GetDC(dummy_hwnd_);

        PIXELFORMATDESCRIPTOR pfd = {};
        pfd.nSize    = sizeof(pfd);
        pfd.nVersion = 1;
        pfd.dwFlags  = PFD_SUPPORT_OPENGL;
        pfd.iPixelType = PFD_TYPE_RGBA;
        pfd.cColorBits = 32;
        int fmt = ChoosePixelFormat(dc_, &pfd);
        SetPixelFormat(dc_, fmt, &pfd);

        gl_ctx_ = wglCreateContext(dc_);
        if (!gl_ctx_ || !wglShareLists(static_cast<HGLRC>(mixer_hglrc), gl_ctx_)) {
            CASPAR_LOG(warning) << L"[av_producer] Failed to create shared GL context for D3D11-GL bridge";
            cleanup();
            return false;
        }
        wglMakeCurrent(dc_, gl_ctx_);

        // Open DX interop device
        interop_device_ = wglDXOpenDeviceNV(d3d11_device_);
        if (!interop_device_) {
            CASPAR_LOG(warning) << L"[av_producer] wglDXOpenDeviceNV failed — using CPU path";
            cleanup();
            return false;
        }

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
        desc.MiscFlags        = 0;

        HRESULT hr = d3d11_device_->CreateTexture2D(&desc, nullptr, &bgra_texture_);
        if (FAILED(hr))
            return false;

        // Create video processor for NV12->BGRA
        HRESULT vhr;
        vhr = d3d11_device_->QueryInterface(__uuidof(ID3D11VideoDevice), reinterpret_cast<void**>(&video_device_));
        if (FAILED(vhr))
            return false;

        d3d11_ctx_->QueryInterface(__uuidof(ID3D11VideoContext), reinterpret_cast<void**>(&video_ctx_));

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
        interop_object_ = wglDXRegisterObjectNV(
            interop_device_, bgra_texture_, interop_gl_tex_, GL_TEXTURE_2D, WGL_ACCESS_READ_ONLY_NV);
        if (!interop_object_) {
            CASPAR_LOG(warning) << L"[av_producer] wglDXRegisterObjectNV failed";
            return false;
        }

        return true;
    }

    std::shared_ptr<core::texture> convert(AVFrame* d3d11_frame, void* ogl_device_ptr)
    {
        if (!active_ || !d3d11_frame)
            return nullptr;

        auto* nv12_tex    = reinterpret_cast<ID3D11Texture2D*>(d3d11_frame->data[0]);
        int   array_index = static_cast<int>(reinterpret_cast<intptr_t>(d3d11_frame->data[1]));

        if (!nv12_tex)
            return nullptr;

        if (!setup_for_size(d3d11_frame->width, d3d11_frame->height))
            return nullptr;

        auto* ogl_dev = static_cast<accelerator::ogl::device*>(ogl_device_ptr);

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

        // Run the video processor: NV12->BGRA
        D3D11_VIDEO_PROCESSOR_STREAM stream = {};
        stream.Enable        = TRUE;
        stream.pInputSurface = input_view;

        video_ctx_->VideoProcessorBlt(video_processor_, vp_output_view_, 0, 1, &stream);
        input_view->Release();

        // Lock the D3D11 texture for GL access
        if (!wglDXLockObjectsNV(interop_device_, 1, &interop_object_))
            return nullptr;

        // Create OGL texture and copy from the interop texture
        std::shared_ptr<accelerator::ogl::texture> ogl_tex;
        try {
            ogl_tex = ogl_dev->dispatch_sync([&]() {
                auto tex = ogl_dev->create_texture(width_, height_, 4, common::bit_depth::bit8);
                tex->copy_from(static_cast<int>(interop_gl_tex_));
                return tex;
            });
        } catch (...) {
            wglDXUnlockObjectsNV(interop_device_, 1, &interop_object_);
            return nullptr;
        }

        wglDXUnlockObjectsNV(interop_device_, 1, &interop_object_);
        return ogl_tex;
    }

    void cleanup()
    {
        teardown_interop();
        teardown_video_processor();

        if (interop_device_) {
            wglDXCloseDeviceNV(interop_device_);
            interop_device_ = nullptr;
        }
        if (gl_ctx_) {
            wglMakeCurrent(nullptr, nullptr);
            wglDeleteContext(gl_ctx_);
            gl_ctx_ = nullptr;
        }
        if (dc_ && dummy_hwnd_) {
            ReleaseDC(dummy_hwnd_, dc_);
            dc_ = nullptr;
        }
        if (dummy_hwnd_) {
            DestroyWindow(dummy_hwnd_);
            dummy_hwnd_ = nullptr;
        }
        UnregisterClassA("CasparCG_FFmpeg_D3D11GL", GetModuleHandle(nullptr));
        active_ = false;
    }

    ~impl() { cleanup(); }

  private:
    void teardown_interop()
    {
        if (interop_object_) {
            wglDXUnregisterObjectNV(interop_device_, interop_object_);
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

// Public interface delegation

d3d11_gl_bridge::d3d11_gl_bridge()
    : impl_(std::make_unique<impl>())
{
}

d3d11_gl_bridge::~d3d11_gl_bridge() = default;

bool d3d11_gl_bridge::init(AVBufferRef* hw_device_ctx, void* ogl_device_ptr)
{
    return impl_->init(hw_device_ctx, ogl_device_ptr);
}

std::shared_ptr<void> d3d11_gl_bridge::convert(AVFrame* d3d11_frame, void* ogl_device_ptr)
{
    return impl_->convert(d3d11_frame, ogl_device_ptr);
}

bool d3d11_gl_bridge::is_active() const
{
    return impl_->active_;
}

void d3d11_gl_bridge::cleanup()
{
    impl_->cleanup();
}

}} // namespace caspar::ffmpeg

#endif // _WIN32
