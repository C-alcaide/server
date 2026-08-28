/*
 * Copyright (c) 2025 CasparCG Contributors
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
 *
 * This module uses the Spout SDK (https://github.com/leadedge/Spout2),
 * licensed under BSD 2-Clause, which is compatible with GPL-3.
 */

#include "spout_consumer.h"

#include <Spout.h>
#include <common/executor.h>
#include <common/diagnostics/graph.h>
#include <common/timer.h>
#include <common/log.h>
#include <common/utf.h>
#include <core/frame/frame.h>
#include <core/frame/pixel_format.h>
#include <core/consumer/frame_consumer.h>
#include <core/consumer/channel_info.h>
#include <core/video_format.h>
#include <core/monitor/monitor.h>
#include <accelerator/ogl/util/texture.h>

// The FBO and blit entry points come from the Spout SDK's own extension loader,
// NOT from GLEW. GLEW macro-renames these names and declares them dllimport while
// SpoutGLextensions.h declares them as plain extern, so the two cannot share a
// translation unit -- see the note at the top of producer/spout_gl_bridge.h, where
// that conflict forced the producer's GL half into a separate file. Here the Spout
// loader already has everything needed, so no second translation unit is required.
#include <SpoutGLextensions.h>

#ifdef ENABLE_VULKAN
// All four of these are deliberately GLEW-free headers, which is what lets the
// Vulkan interop live in this Spout-including translation unit at all.
#include <accelerator/vulkan/util/device.h>
#include <accelerator/vulkan/util/gl_export_bridge.h>
#include <accelerator/vulkan/util/texture.h>
#include <accelerator/vulkan/util/texture_wrapper.h>
#include <common/bit_depth.h>
#endif

#include <memory>
#include <atomic>
#include <chrono>
#include <mutex>
#include <future>
#include <sstream>
#include <vector>
#include <iostream>
#include <algorithm>
#include <iterator>
#include <cstring>
#include <thread>

// For Context Creation
#include <windows.h>
#include <gl/GL.h>

// WGL extension constants for shared-context creation
#ifndef WGL_CONTEXT_MAJOR_VERSION_ARB
#define WGL_CONTEXT_MAJOR_VERSION_ARB 0x2091
#endif
#ifndef WGL_CONTEXT_MINOR_VERSION_ARB
#define WGL_CONTEXT_MINOR_VERSION_ARB 0x2092
#endif
#ifndef WGL_CONTEXT_PROFILE_MASK_ARB
#define WGL_CONTEXT_PROFILE_MASK_ARB  0x9126
#endif
#ifndef WGL_CONTEXT_CORE_PROFILE_BIT_ARB
#define WGL_CONTEXT_CORE_PROFILE_BIT_ARB 0x00000001
#endif
typedef HGLRC(WINAPI* PFNWGLCREATECONTEXTATTRIBSARBPROC)(HDC, HGLRC, const int*);

// FFmpeg for pixel-format conversion and downscaling
#pragma warning(push)
#pragma warning(disable: 4244)  // possible loss of data (FFmpeg internal macros)
extern "C" {
#include <libswscale/swscale.h>
#include <libavutil/pixfmt.h>
}
#pragma warning(pop)

namespace caspar { namespace spout {

namespace {

class gl_context
{
    HWND  hwnd_ = nullptr;
    HDC   hdc_  = nullptr;
    HGLRC hglrc_ = nullptr;
    bool  shared_ = false;

  public:
    /// `expect_no_share` suppresses the "nothing to share" warning for a backend where
    /// that is the normal state rather than a fault. On the Vulkan mixer
    /// `native_gl_context()` is never overridden, so there is deliberately no GL context
    /// to share and the consumer imports exportable Vulkan memory instead -- warning
    /// about it on every Vulkan channel made the log triage report an unexplained
    /// warning on a run where the fast path was working.
    gl_context(void* share_context = nullptr, bool expect_no_share = false)
    {
        WNDCLASS wc      = {0};
        wc.lpfnWndProc   = DefWindowProc;
        wc.hInstance     = GetModuleHandle(NULL);
        wc.lpszClassName = L"CasparCG_Spout_Consumer_Context";
        RegisterClass(&wc);

        hwnd_ = CreateWindow(wc.lpszClassName, L"Spout Consumer Context", 0, 0, 0, 0, 0, 0, 0, wc.hInstance, 0);

        if (hwnd_) {
            hdc_ = GetDC(hwnd_);
            PIXELFORMATDESCRIPTOR pfd;
            ZeroMemory(&pfd, sizeof(pfd));
            pfd.nSize      = sizeof(pfd);
            pfd.nVersion   = 1;
            pfd.dwFlags    = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
            pfd.iPixelType = PFD_TYPE_RGBA;
            pfd.cColorBits = 32;
            pfd.cDepthBits = 16;
            pfd.iLayerType = PFD_MAIN_PLANE;

            int format = ChoosePixelFormat(hdc_, &pfd);
            SetPixelFormat(hdc_, format, &pfd);

            // Create a bootstrap context to load WGL extensions
            HGLRC temp_ctx = wglCreateContext(hdc_);
            if (!temp_ctx) return;
            wglMakeCurrent(hdc_, temp_ctx);

            auto wglCreateContextAttribsARB_fn = reinterpret_cast<PFNWGLCREATECONTEXTATTRIBSARBPROC>(
                wglGetProcAddress("wglCreateContextAttribsARB"));

            if (wglCreateContextAttribsARB_fn) {
                int attribs[] = {
                    WGL_CONTEXT_MAJOR_VERSION_ARB, 4,
                    WGL_CONTEXT_MINOR_VERSION_ARB, 5,
                    WGL_CONTEXT_PROFILE_MASK_ARB,  WGL_CONTEXT_CORE_PROFILE_BIT_ARB,
                    0
                };

                // Try shared context with the OGL mixer (enables zero-copy texture path)
                if (share_context) {
                    hglrc_ = wglCreateContextAttribsARB_fn(hdc_, reinterpret_cast<HGLRC>(share_context), attribs);
                    if (hglrc_) {
                        shared_ = true;
                    } else {
                        CASPAR_LOG(warning)
                            << L"[spout_consumer] could not share a GL context with the mixer (error "
                            << static_cast<unsigned int>(GetLastError())
                            << L"); falling back to reading the frame back and sending pixels.";
                        // Re-establish bootstrap context on failure
                        wglMakeCurrent(hdc_, temp_ctx);
                    }
                } else if (expect_no_share) {
                    CASPAR_LOG(debug)
                        << L"[spout_consumer] no GL context to share, as expected on this backend; "
                           L"using the imported-Vulkan-memory path instead.";
                } else {
                    CASPAR_LOG(warning)
                        << L"[spout_consumer] the channel exposed no GL context to share; falling back to "
                           L"reading the frame back and sending pixels.";
                }

                if (!hglrc_)
                    hglrc_ = wglCreateContextAttribsARB_fn(hdc_, nullptr, attribs);

                wglMakeCurrent(nullptr, nullptr);
                wglDeleteContext(temp_ctx);
            } else {
                // Fallback: use basic context
                hglrc_ = temp_ctx;
            }
        }
    }

    ~gl_context()
    {
        if (hglrc_) {
            wglMakeCurrent(nullptr, nullptr);
            wglDeleteContext(hglrc_);
        }
        if (hdc_) {
            ReleaseDC(hwnd_, hdc_);
        }
        if (hwnd_) {
            DestroyWindow(hwnd_);
        }
        UnregisterClass(L"CasparCG_Spout_Consumer_Context", GetModuleHandle(NULL));
    }

    bool make_current()
    {
        if (hdc_ && hglrc_) {
            return wglMakeCurrent(hdc_, hglrc_) == TRUE;
        }
        return false;
    }

    bool is_shared() const { return shared_; }
};

} // namespace

struct spout_consumer_impl : public core::frame_consumer
{
    std::string                 sender_name_;
    std::unique_ptr<Spout>      sender_;
    std::unique_ptr<gl_context> context_;
    core::video_format_desc     format_desc_;

    // GL context sharing for zero-copy GPU texture path
    void* gl_share_context_ = nullptr;
    std::atomic<bool> gpu_path_active_{false};

    // True when the GPU path also did the downscale, i.e. the frame never reached
    // host memory *and* the send was small. Reported separately from
    // gpu_path_active_ because "zero-copy" and "cheap" used to be mutually
    // exclusive here and the difference is the whole point of the change.
    std::atomic<bool> gpu_downscale_active_{false};

    // Optional downscale cap (0 = no cap = native resolution).
    // Set via AMCP: ADD x SPOUT "Name" MAX_WIDTH 1920 MAX_HEIGHT 1080
    int max_w_ = 0;
    int max_h_ = 0;

    // Output dimensions (computed in initialize)
    int  out_w_ = 0;
    int  out_h_ = 0;

    // ── Preview frame-rate divisor: send every Nth frame (1 = every frame) ──
    // A preview does not need 60 Hz, and skipping a send is a branch. Counted on
    // the calling (video) thread so a skipped frame costs nothing at all -- not
    // even the executor hand-off.
    int      every_nth_   = 1;
    uint64_t frame_index_ = 0;

    // Alternative spelling of the same thing for an operator who knows the rate they
    // want rather than the divisor. Resolved to every_nth_ in initialize(), because
    // that is the first point at which the channel's own rate is known.
    int target_fps_ = 0;

    // ── GPU downscale (both backends): small destination texture + FBO pair ──
    // Lives in the consumer's own GL context, on the executor thread. Recreated
    // only when the output dimensions change.
    unsigned int blit_tex_      = 0;
    unsigned int blit_fbo_read_ = 0;
    unsigned int blit_fbo_draw_ = 0;
    int          blit_w_        = 0;
    int          blit_h_        = 0;
    bool         blit_failed_   = false; // sticky: stop retrying a broken FBO every frame

#ifdef ENABLE_VULKAN
    // ── Vulkan backend: our own exportable image, imported once into GL ──
    //
    // The Vulkan mixer exposes NO GL context to share (image_mixer does not override
    // native_gl_context(), so channel_info::gl_share_context is null on that backend)
    // and its composite attachment is not exportable memory -- create_attachment()
    // rather than create_exportable_texture(). So neither half of the OGL path
    // applies: there is nothing to share a context with, and nothing GL could import
    // even if there were. Instead the mixer's attachment is blitted into an image
    // this consumer owns and which IS exportable, and that image is imported into
    // the consumer's own (unshared) context. GL_EXT_memory_object does not require
    // context sharing, only that the memory be exportable.
    void*                                                   vk_device_ = nullptr;
    std::shared_ptr<accelerator::vulkan::texture>           vk_shared_tex_;
    std::unique_ptr<accelerator::vulkan::gl_shared_texture> vk_gl_import_;
    bool                                                    vk_interop_failed_ = false;
    bool                                                    vk_send_failed_logged_ = false;
#endif

    std::vector<uint8_t> out_buf_;  // top-down BGRA8 output buffer

    // swscale: lazily rebuilt on executor when format or dimensions change.
    // For format-conversion-only frames (no scaling, src dims == dst dims) we
    // split the image into kSwsBands horizontal bands and process them in
    // parallel — each band gets its own SwsContext (sws_scale is NOT
    // thread-safe for shared contexts).
    // When the dimensions differ (downscale via MAX_WIDTH/MAX_HEIGHT) the
    // output is already small so single-threaded processing of ctxs_[0] is
    // fast enough.
    static constexpr int kSwsBands = 4;
    AVPixelFormat   src_av_fmt_        = AV_PIX_FMT_NONE;
    SwsContext*     sws_ctxs_[kSwsBands] = {};  // [0] is also the single-thread fallback

    void free_sws_ctxs() noexcept {
        for (auto*& c : sws_ctxs_) { if (c) { sws_freeContext(c); c = nullptr; } }
    }

    // ── GPU downscale helpers. Executor thread only, context current. ──────────
    //
    // Freeing has to happen on the same thread that made the names, which is why
    // the destructor runs its teardown as an executor task rather than inline.
    void release_gl_blit() noexcept
    {
        if (blit_fbo_read_) { glDeleteFramebuffersEXT(1, &blit_fbo_read_); blit_fbo_read_ = 0; }
        if (blit_fbo_draw_) { glDeleteFramebuffersEXT(1, &blit_fbo_draw_); blit_fbo_draw_ = 0; }
        if (blit_tex_)      { glDeleteTextures(1, &blit_tex_);             blit_tex_      = 0; }
        blit_w_ = blit_h_ = 0;
    }

    /// Lazily (re)create the small destination texture and the FBO pair for it.
    /// Returns false once and then stays false for a given size, so a driver that
    /// refuses the FBO does not cost a rebuild attempt on every frame.
    bool ensure_blit_target(int w, int h)
    {
        if (blit_tex_ && blit_w_ == w && blit_h_ == h)
            return true;
        if (blit_failed_)
            return false;

        release_gl_blit();

        glGenTextures(1, &blit_tex_);
        if (!blit_tex_) { blit_failed_ = true; return false; }
        glBindTexture(GL_TEXTURE_2D, blit_tex_);
        // RGBA8 regardless of the channel's depth: this is a preview, and Spout's
        // shared-texture format is 8-bit anyway.
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glBindTexture(GL_TEXTURE_2D, 0);

        glGenFramebuffersEXT(1, &blit_fbo_read_);
        glGenFramebuffersEXT(1, &blit_fbo_draw_);
        if (!blit_fbo_read_ || !blit_fbo_draw_) {
            release_gl_blit();
            blit_failed_ = true;
            return false;
        }

        // The draw FBO's attachment never changes, so bind it once here. The read
        // FBO's does change (a different mixer texture each frame), so it is
        // attached per blit.
        glBindFramebufferEXT(GL_DRAW_FRAMEBUFFER, blit_fbo_draw_);
        glFramebufferTexture2DEXT(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, blit_tex_, 0);
        const auto ok = glCheckFramebufferStatusEXT(GL_DRAW_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE_EXT;
        glBindFramebufferEXT(GL_DRAW_FRAMEBUFFER, 0);

        if (!ok) {
            CASPAR_LOG(warning) << L"[spout_consumer] incomplete draw framebuffer for the "
                                << w << L"x" << h << L" preview; using the readback path.";
            release_gl_blit();
            blit_failed_ = true;
            return false;
        }

        blit_w_ = w;
        blit_h_ = h;
        return true;
    }

    /// Downscale `src_tex` (`src_w` x `src_h`) into the cached small texture.
    /// Returns its GL name, or 0 to mean "fall back".
    ///
    /// Orientation is preserved: both rectangles are given in the same winding, so a
    /// top-down source stays top-down and the existing bInvert=false on SendTexture
    /// keeps its meaning.
    unsigned int downscale_on_gpu(unsigned int src_tex, int src_w, int src_h)
    {
        if (!ensure_blit_target(out_w_, out_h_))
            return 0;

        glBindFramebufferEXT(GL_READ_FRAMEBUFFER, blit_fbo_read_);
        glFramebufferTexture2DEXT(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, src_tex, 0);
        if (glCheckFramebufferStatusEXT(GL_READ_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE_EXT) {
            glFramebufferTexture2DEXT(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, 0, 0);
            glBindFramebufferEXT(GL_READ_FRAMEBUFFER, 0);
            return 0;
        }

        glBindFramebufferEXT(GL_DRAW_FRAMEBUFFER, blit_fbo_draw_);
        glBlitFramebufferEXT(0, 0, src_w, src_h,
                             0, 0, out_w_, out_h_,
                             GL_COLOR_BUFFER_BIT, GL_LINEAR);

        // Detach the source so this FBO holds no reference to a mixer texture that
        // may be recycled before the next frame.
        glFramebufferTexture2DEXT(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, 0, 0);
        glBindFramebufferEXT(GL_READ_FRAMEBUFFER, 0);
        glBindFramebufferEXT(GL_DRAW_FRAMEBUFFER, 0);

        return blit_tex_;
    }

    bool init_sws_ctxs(int sw, int sh, AVPixelFormat sfmt, int dw, int dh) {
        free_sws_ctxs();
        for (auto*& c : sws_ctxs_) {
            c = sws_getContext(sw, sh, sfmt, dw, dh, AV_PIX_FMT_BGRA,
                               SWS_FAST_BILINEAR, nullptr, nullptr, nullptr);
            if (!c) { free_sws_ctxs(); return false; }
        }
        return true;
    }

    // Dedicated thread for GL context + Spout + swscale work.
    // send() ALWAYS returns make_ready_future(true) immediately so CasparCG's
    // output pipeline is never stalled by the Spout consumer — even for large
    // HDR channels where sws_scale takes many ms per frame.
    //
    // busy_ is an atomic flag: if the executor is still processing the previous
    // frame when the next one arrives, we drop the incoming frame rather than
    // queueing it. This caps latency and prevents backlog accumulation.
    caspar::executor                    executor_;
    std::atomic<bool>                   busy_{ false };

    spl::shared_ptr<diagnostics::graph> graph_;
    caspar::timer                       frame_timer_;

    // FPS counter — sampled on the calling thread once per second
    std::chrono::steady_clock::time_point last_fps_update_{ std::chrono::steady_clock::now() };
    int    frames_since_update_ = 0;
    double current_fps_         = 0.0;

    // ── Map any CasparCG pixel_format_desc to the matching AVPixelFormat ──
    static AVPixelFormat caspar_to_av_fmt(const core::pixel_format_desc& pfd)
    {
        using pf = core::pixel_format;
        using bd = common::bit_depth;

        if (pfd.planes.empty()) return AV_PIX_FMT_NONE;
        const bool b16 = pfd.planes[0].depth != bd::bit8;

        switch (pfd.format) {
            case pf::bgra:  return b16 ? AV_PIX_FMT_BGRA64LE  : AV_PIX_FMT_BGRA;
            case pf::rgba:  return b16 ? AV_PIX_FMT_RGBA64LE  : AV_PIX_FMT_RGBA;
            case pf::argb:  return b16 ? AV_PIX_FMT_ARGB      : AV_PIX_FMT_ARGB;   // no 16-bit ARGB in FFmpeg
            case pf::abgr:  return b16 ? AV_PIX_FMT_ABGR      : AV_PIX_FMT_ABGR;
            case pf::bgr:   return b16 ? AV_PIX_FMT_BGR48LE   : AV_PIX_FMT_BGR24;
            case pf::rgb:   return b16 ? AV_PIX_FMT_RGB48LE   : AV_PIX_FMT_RGB24;
            case pf::gray:
            case pf::luma:  return b16 ? AV_PIX_FMT_GRAY16LE  : AV_PIX_FMT_GRAY8;
            case pf::gbrap: return b16 ? AV_PIX_FMT_GBRAP16LE : AV_PIX_FMT_GBRAP;
            case pf::gbrp: {
                if (!b16)                                      return AV_PIX_FMT_GBRP;
                if (pfd.planes[0].depth == bd::bit10)          return AV_PIX_FMT_GBRP10LE;
                if (pfd.planes[0].depth == bd::bit12)          return AV_PIX_FMT_GBRP12LE;
                return AV_PIX_FMT_GBRP16LE;
            }
            case pf::ycbcr: {
                if (pfd.planes.size() < 2) return AV_PIX_FMT_NONE;
                const int yw = pfd.planes[0].width,  yh = pfd.planes[0].height;
                const int cw = pfd.planes[1].width,  ch = pfd.planes[1].height;
                if (ch == yh && cw == yw)           return b16 ? AV_PIX_FMT_YUV444P10LE : AV_PIX_FMT_YUV444P;
                if (ch == yh && cw * 2 == yw)       return b16 ? AV_PIX_FMT_YUV422P10LE : AV_PIX_FMT_YUV422P;
                if (ch * 2 == yh && cw * 2 == yw)   return b16 ? AV_PIX_FMT_YUV420P10LE : AV_PIX_FMT_YUV420P;
                return AV_PIX_FMT_NONE;
            }
            case pf::ycbcra: return b16 ? AV_PIX_FMT_YUVA444P10LE : AV_PIX_FMT_YUVA444P;
            case pf::uyvy:   return AV_PIX_FMT_UYVY422;
            default:         return AV_PIX_FMT_NONE;
        }
    }

    spout_consumer_impl(std::wstring name, int max_w, int max_h, int every_nth, int target_fps)
        : max_w_(max_w)
        , max_h_(max_h)
        , every_nth_((std::max)(1, every_nth))
        , target_fps_((std::max)(0, target_fps))
        , executor_(L"Spout Consumer")
    {
        sender_name_.reserve(name.length());
        for (wchar_t c : name)
            sender_name_.push_back(static_cast<char>(c));

        if (sender_name_.empty())
            sender_name_ = "CasparCG Spout";

        graph_ = spl::make_shared<diagnostics::graph>();
        graph_->set_text(print());
        graph_->set_color("frame-time",    diagnostics::color(0.5f, 1.0f, 0.2f));
        graph_->set_color("dropped-frame", diagnostics::color(0.3f, 0.6f, 0.3f));
        diagnostics::register_graph(graph_);
    }

    ~spout_consumer_impl()
    {
        // context_ (and its HWND/HGLRC, created lazily on the executor thread
        // in send()) must be destroyed on that same thread — Win32 windows and
        // WGL contexts are single-thread-owned. Destroying them here, on
        // whatever thread drops this consumer, would call DestroyWindow/
        // wglDeleteContext cross-thread. Run the teardown as an executor task
        // instead of just draining the queue.
        executor_.invoke([this] {
            // GL names and the imported Vulkan texture belong to this thread's
            // context, so they have to go before the context does.
            if (context_)
                context_->make_current();
            release_gl_blit();
#ifdef ENABLE_VULKAN
            vk_gl_import_.reset();
            vk_shared_tex_.reset();
#endif
            if (sender_)
                sender_->ReleaseSender();
            sender_.reset();
            context_.reset();
        });
        free_sws_ctxs();
    }

    void initialize(const core::video_format_desc& format_desc, const core::channel_info& channel_info, int port_index) override
    {
        format_desc_ = format_desc;

        // Capture the mixer's GL context handle for shared-context creation.
        gl_share_context_ = channel_info.gl_share_context;

#ifdef ENABLE_VULKAN
        // Null unless the channel is on the Vulkan mixer. This is what makes the
        // Vulkan zero-copy path reachable at all -- gl_share_context is always null
        // there, so is_shared() can never be the gate for that backend.
        vk_device_ = channel_info.vk_device;
#endif

        // Compute output dimensions (native by default; capped if MAX_WIDTH/MAX_HEIGHT set).
        if (max_w_ > 0 || max_h_ > 0) {
            const int sw = format_desc.square_width;
            const int sh = format_desc.square_height;
            double scale = 1.0;
            if (max_w_ > 0 && sw > max_w_) scale = static_cast<double>(max_w_) / sw;
            if (max_h_ > 0 && sh * scale > max_h_) scale = static_cast<double>(max_h_) / sh;
            const int raw_w = static_cast<int>(sw * scale);
            const int raw_h = static_cast<int>(sh * scale);
            out_w_ = (std::max)(2, raw_w - (raw_w % 2));
            out_h_ = (std::max)(2, raw_h - (raw_h % 2));
        } else {
            out_w_ = format_desc.width;
            out_h_ = format_desc.height;
        }

        // FPS is a friendlier spelling of EVERY_NTH, resolvable only now that the
        // channel's rate is known. Rounds down to the nearest achievable rate --
        // asking for 30 on a 50p channel sends every other frame (25), not 30, since
        // only whole frames can be skipped. EVERY_NTH wins if both were given.
        if (target_fps_ > 0 && every_nth_ == 1) {
            const double channel_fps = format_desc.fps;
            if (channel_fps > 0.0 && target_fps_ < channel_fps) {
                every_nth_ = (std::max)(1, static_cast<int>(channel_fps / target_fps_));
                CASPAR_LOG(info) << L"[spout_consumer] FPS " << target_fps_ << L" on a "
                                 << channel_fps << L" Hz channel: sending every " << every_nth_
                                 << L" frames (" << (channel_fps / every_nth_) << L" Hz).";
            }
        }

        out_buf_.assign(static_cast<size_t>(out_w_) * out_h_ * 4, 0);

        // Force swscale rebuild on next frame (dimensions may have changed).
        src_av_fmt_ = AV_PIX_FMT_NONE;
        free_sws_ctxs();

        // The GPU blit target and the Vulkan shared image are both sized from
        // out_w_/out_h_, so a re-initialise at a new format invalidates them. They
        // are owned by the executor thread's context, so drop them there.
        executor_.begin_invoke([this] {
            if (context_ && context_->make_current()) {
                release_gl_blit();
#ifdef ENABLE_VULKAN
                vk_gl_import_.reset();
                vk_shared_tex_.reset();
#endif
            }
            blit_failed_ = false;
        });
    }

#ifdef ENABLE_VULKAN
    /// Vulkan backend zero-copy send. Executor thread, context current.
    ///
    /// Returns false to mean "take the readback path", having logged the reason at
    /// most once -- a silent fallback here is indistinguishable from the defect this
    /// exists to fix.
    bool try_send_vulkan(const std::shared_ptr<core::texture>& core_tex)
    {
        if (!vk_device_ || vk_interop_failed_)
            return false;

        auto* wrapper = dynamic_cast<accelerator::vulkan::texture_wrapper*>(core_tex.get());
        if (!wrapper)
            return false;

        if (!accelerator::vulkan::gl_import_supported()) {
            vk_interop_failed_ = true;
            CASPAR_LOG(info) << L"[spout_consumer] this GL context cannot import Vulkan memory "
                                L"(GL_EXT_memory_object missing); the Vulkan zero-copy path is "
                                L"unavailable and every frame will be read back and converted.";
            return false;
        }

        // The mixer's renderpass must have finished before its attachment can be a
        // blit source.
        wrapper->ensure_render_complete();
        auto src = wrapper->vk_texture();
        if (!src)
            return false;

        auto* dev = static_cast<accelerator::vulkan::device*>(vk_device_);

        // One exportable image and one GL import per output size, not per frame:
        // each import creates a GL memory object and a GL texture.
        if (!vk_shared_tex_ || !vk_gl_import_ || vk_shared_tex_->width() != out_w_ ||
            vk_shared_tex_->height() != out_h_) {
            vk_gl_import_.reset();
            vk_shared_tex_.reset();

            try {
                // swap_rb: the mixer's 8-bit attachment holds BGRA bytes in an
                // RGBA-declared image, and a blit preserves byte order, so without this
                // the published texture arrives red/blue exchanged against the OGL
                // mixer's. Measured through a real receiver -- see device.h.
                vk_shared_tex_ =
                    dev->create_exportable_texture(out_w_, out_h_, 4, common::bit_depth::bit8, true);
                if (!vk_shared_tex_)
                    throw std::runtime_error("create_exportable_texture returned null");
                vk_gl_import_ = std::make_unique<accelerator::vulkan::gl_shared_texture>(vk_shared_tex_);
            } catch (const std::exception& e) {
                vk_gl_import_.reset();
                vk_shared_tex_.reset();
                vk_interop_failed_ = true;
                CASPAR_LOG(info) << L"[spout_consumer] cannot share a Vulkan image with OpenGL ("
                                 << u16(e.what())
                                 << L"); the Vulkan zero-copy path is unavailable and every frame "
                                    L"will be read back and converted.";
                return false;
            }
        }

        // Scaling blit straight into the shared image. This is both halves at once:
        // the downscale and the escape from swscale.
        if (!dev->blit_to_shared(src, vk_shared_tex_))
            return false;

        // CHECK THE RETURN VALUE. Setting gpu_path_active_ before this and ignoring the
        // result made `state()` report a zero-copy send that never happened: measured
        // 2026-08-28, `spout/gpu-path` true on the Vulkan mixer while `GetSenderList()`
        // in a receiving process was EMPTY -- no sender had been created at all. A flag
        // that exists to distinguish the fast path from the fallback must not be set by
        // intent.
        if (!sender_->SendTexture(static_cast<GLuint>(vk_gl_import_->gl_id()),
                                  GL_TEXTURE_2D,
                                  static_cast<unsigned int>(out_w_),
                                  static_cast<unsigned int>(out_h_),
                                  false)) {
            if (!vk_send_failed_logged_) {
                vk_send_failed_logged_ = true;
                CASPAR_LOG(warning)
                    << L"[spout_consumer] SendTexture refused the imported Vulkan texture ("
                    << out_w_ << L"x" << out_h_
                    << L"); falling back to reading the frame back and sending pixels.";
            }
            return false;
        }

        gpu_path_active_      = true;
        gpu_downscale_active_ = (src->width() != out_w_ || src->height() != out_h_);
        return true;
    }
#endif

    std::future<bool> send(const core::video_field field, core::const_frame frame) override
    {
        // Quick pre-checks on the calling (video) thread — no heavy work here.
        if (!frame.width() || !frame.height() || out_w_ == 0 || out_h_ == 0)
            return caspar::make_ready_future(true);

        const AVPixelFormat src_fmt = caspar_to_av_fmt(frame.pixel_format_desc());
        if (src_fmt == AV_PIX_FMT_NONE)
            return caspar::make_ready_future(true);

        // Frame-rate divisor. Checked here, on the calling thread, before anything
        // else: a skipped preview frame then costs one increment and a compare, not
        // an executor hand-off. Counted rather than timed, so it stays deterministic
        // and does not drift against the channel's tick.
        if (every_nth_ > 1 && (frame_index_++ % static_cast<uint64_t>(every_nth_)) != 0)
            return caspar::make_ready_future(true);

        // FPS counter — updated every second in the calling (video) thread.
        {
            auto now = std::chrono::steady_clock::now();
            ++frames_since_update_;
            const auto dur = std::chrono::duration_cast<std::chrono::duration<double>>(now - last_fps_update_).count();
            if (dur >= 1.0) {
                current_fps_         = frames_since_update_ / dur;
                frames_since_update_ = 0;
                last_fps_update_     = now;
                std::wstringstream ss;
                ss << std::fixed << std::setprecision(2) << print() << L" Fps: " << current_fps_;
                graph_->set_text(ss.str());
            }
        }

        // Frame-drop: if the executor is still processing the previous frame,
        // skip this one rather than queueing it. This guarantees send() always
        // returns immediately and CasparCG's output pipeline is never stalled
        // by the cost of sws_scale (which can take 50+ ms for 6000×1700
        // GBRAP16LE). const_frame is ref-counted so capturing it by value in
        // the lambda extends its lifetime until the executor finishes with it.
        if (busy_.exchange(true)) {
            graph_->set_tag(diagnostics::tag_severity::WARNING, "dropped-frame");
            return caspar::make_ready_future(true);  // drop — still processing previous
        }

        frame_timer_.restart();
        executor_.begin_invoke([this, frame, src_fmt]() mutable {
            const int src_w = format_desc_.width;
            const int src_h = format_desc_.height;

            if (!context_) {
#ifdef ENABLE_VULKAN
                const bool expect_no_share = (vk_device_ != nullptr);
#else
                const bool expect_no_share = false;
#endif
                context_ = std::make_unique<gl_context>(gl_share_context_, expect_no_share);
                if (context_->is_shared()) {
                    CASPAR_LOG(info) << L"[spout_consumer] GL context shared with mixer - GPU texture path available.";
                }
            }
            if (!context_->make_current()) {
                busy_ = false;
                return;
            }

            if (!sender_) {
                sender_ = std::make_unique<Spout>();
                sender_->SetFrameCount(true);
                sender_->SetSenderName(sender_name_.c_str());
            }

            // ── GPU texture path: zero-copy via SendTexture() ──
            //
            // The dimension equality that used to guard this is gone. It made
            // small-and-CPU or full-res-and-GPU the only two choices, and the only
            // combination that scales to several previews is small-and-GPU: a
            // 1920x1080 readback costs 3.12 ms (19% of a frame) against 0.088 ms at
            // 256x144, and the downscale itself is a GPU blit. Requesting MAX_WIDTH
            // therefore used to disable the very path that made it affordable.
            if (auto core_tex = frame.texture()) {
                // OGL mixer: the texture lives in the mixer's context, so this needs
                // the shared context to name it at all.
                if (context_->is_shared()) {
                    if (auto ogl_tex = std::dynamic_pointer_cast<accelerator::ogl::texture>(core_tex)) {
                        auto send_id = static_cast<unsigned int>(ogl_tex->id());
                        auto send_w  = ogl_tex->width();
                        auto send_h  = ogl_tex->height();

                        const bool want_scale = (send_w != out_w_ || send_h != out_h_);
                        if (want_scale) {
                            // Not named `small`: <windows.h> drags in rpcndr.h, which
                            // #defines small as char.
                            if (auto scaled_id = downscale_on_gpu(send_id, send_w, send_h)) {
                                send_id = scaled_id;
                                send_w  = out_w_;
                                send_h  = out_h_;
                            } else {
                                send_id = 0; // FBO unavailable -- fall through to readback
                            }
                        }

                        // bInvert=false: OGL mixer output is already top-down in texture.
                        // The return value is checked for the same reason as on the Vulkan
                        // path below -- a refused send must not report gpu-path true.
                        if (send_id && sender_->SendTexture(static_cast<GLuint>(send_id),
                                                            GL_TEXTURE_2D,
                                                            static_cast<unsigned int>(send_w),
                                                            static_cast<unsigned int>(send_h),
                                                            false)) {
                            gpu_path_active_      = true;
                            gpu_downscale_active_ = want_scale;
                            graph_->set_value("frame-time", frame_timer_.elapsed() * 1000.0);
                            busy_ = false;
                            return;
                        }
                    }
                }

#ifdef ENABLE_VULKAN
                // Vulkan mixer: a different mechanism entirely, because there is no
                // shared context and the composite is not exportable. See
                // try_send_vulkan().
                if (try_send_vulkan(core_tex)) {
                    graph_->set_value("frame-time", frame_timer_.elapsed() * 1000.0);
                    busy_ = false;
                    return;
                }
#endif
            }

            // ── CPU fallback path: swscale + SendImage() ──
            gpu_path_active_      = false;
            gpu_downscale_active_ = false;

            // Rebuild swscale contexts when format or output dimensions change.
            if (src_fmt != src_av_fmt_ || !sws_ctxs_[0]) {
                src_av_fmt_ = src_fmt;
                // Use FAST_BILINEAR: much faster than BILINEAR for large frames;
                // quality difference is imperceptible at monitor preview sizes.
                if (!init_sws_ctxs(src_w, src_h, src_fmt, out_w_, out_h_)) {
                    busy_ = false; return;
                }
            }

            // Build per-plane source pointer/stride arrays.
            const auto& pfd2 = frame.pixel_format_desc();
            const uint8_t* sp[4] = {};
            int            ss[4] = {};
            const int nplanes = static_cast<int>(pfd2.planes.size());
            for (int n = 0; n < nplanes && n < 4; ++n) {
                sp[n] = reinterpret_cast<const uint8_t*>(frame.image_data(n).data());
                ss[n] = pfd2.planes[n].linesize;
            }

            // Output: packed BGRA8 into out_buf_.
            // If no scaling is needed (src dims == dst dims, i.e. pure format
            // conversion), split into kSwsBands horizontal bands and convert
            // in parallel — this turns a 50+ ms single-threaded job (e.g.
            // 6000×1700 GBRAP16LE) into ~12 ms with 4 threads.
            // For scaled output (MAX_WIDTH/MAX_HEIGHT set) the single-threaded
            // path on ctxs_[0] is fast enough since the output is small.
            if (src_w == out_w_ && src_h == out_h_) {
                const int bands  = kSwsBands;
                const int band_h = src_h / bands;
                std::vector<std::thread> thr;
                thr.reserve(bands);
                for (int t = 0; t < bands; ++t) {
                    thr.emplace_back([&, t]() noexcept {
                        const int y0 = t * band_h;
                        const int h  = (t == bands - 1) ? src_h - y0 : band_h;
                        // Per-plane source offset
                        const uint8_t* spb[4] = {};
                        int            ssb[4] = {};
                        for (int n = 0; n < nplanes && n < 4; ++n) {
                            spb[n] = sp[n] + static_cast<size_t>(y0) * ss[n];
                            ssb[n] = ss[n];
                        }
                        // Packed dest offset
                        uint8_t* dpb[4] = {
                            out_buf_.data() + static_cast<size_t>(y0) * out_w_ * 4,
                            nullptr, nullptr, nullptr };
                        int dsb[4] = { out_w_ * 4, 0, 0, 0 };
                        sws_scale(sws_ctxs_[t], spb, ssb, 0, h, dpb, dsb);
                    });
                }
                for (auto& th : thr) th.join();
            } else {
                // Scaled output: single thread is fine (output is small).
                uint8_t* dp[4] = { out_buf_.data(), nullptr, nullptr, nullptr };
                int      ds[4] = { out_w_ * 4,      0,       0,       0       };
                sws_scale(sws_ctxs_[0], sp, ss, 0, src_h, dp, ds);
            }

            // Send top-down BGRA8 pixels.
            sender_->SendImage(out_buf_.data(),
                               static_cast<unsigned int>(out_w_),
                               static_cast<unsigned int>(out_h_),
                               GL_BGRA_EXT,
                               false);

            graph_->set_value("frame-time", frame_timer_.elapsed() * 1000.0);
            busy_ = false;
        });

        // Always return immediately — never stall CasparCG's output pipeline.
        return caspar::make_ready_future(true);
    }

    std::wstring print() const override
    {
        std::wstring wname(sender_name_.begin(), sender_name_.end());
        return L"SPOUT Consumer: " + wname;
    }

    std::wstring name() const override { return L"SPOUT"; }

    int index() const override
    {
        // Derive a stable, deterministic index from the sender name.
        // This ensures that the temporary probe consumer created by CasparCG
        // when processing a REMOVE command returns the same index as the one
        // originally registered via ADD — both use the same sender name.
        // Range 10000–19999 avoids DeckLink(300+), Screen(600+), FFmpeg(100000+).
        return 10000 + static_cast<int>(std::hash<std::string>{}(sender_name_) % 10000);
    }

    bool needs_cpu_frame_data() const override { return !gpu_path_active_; }

    /// Reported because nothing else can tell zero-copy from a correct-looking CPU
    /// fallback. Both produce the right picture at the right size; they differ by
    /// 3.12 ms per frame per preview, and before this the only way to know which one
    /// ran was a debugger. A check that cannot see which path was taken cannot fail
    /// for a change that breaks the fast one.
    caspar::core::monitor::state state() const override
    {
        caspar::core::monitor::state state;
        state["spout/sender-name"]   = sender_name_;
        state["spout/gpu-path"]      = gpu_path_active_.load();
        state["spout/gpu-downscale"] = gpu_downscale_active_.load();
        state["spout/out-width"]     = out_w_;
        state["spout/out-height"]    = out_h_;
        state["spout/every-nth"]     = every_nth_;
        state["spout/fps"]           = current_fps_;
        return state;
    }
};

// ── Helper: read optional integer param from AMCP token list ──────────────────
static int get_int_param(const std::vector<std::wstring>& params,
                         const std::wstring& key, int default_val = 0)
{
    for (size_t i = 0; i + 1 < params.size(); ++i) {
        std::wstring upper = params[i];
        std::transform(upper.begin(), upper.end(), upper.begin(), ::toupper);
        if (upper == key) {
            try { return std::stoi(params[i + 1]); }
            catch (...) {}
        }
    }
    return default_val;
}

spl::shared_ptr<core::frame_consumer> create_spout_consumer(
    const std::vector<std::wstring>&                         params,
    const core::video_format_repository&                     format_repository,
    const std::vector<spl::shared_ptr<core::video_channel>>& channels,
    const core::channel_info&                                channel_info)
{
    if (params.empty() || params[0] != L"SPOUT")
        return core::frame_consumer::empty();

    std::wstring name = (params.size() > 1) ? params[1] : L"";

    // Optional: ADD x SPOUT "Name" MAX_WIDTH 1920 MAX_HEIGHT 1080 [EVERY_NTH 2 | FPS 30]
    const int max_w     = get_int_param(params, L"MAX_WIDTH");
    const int max_h     = get_int_param(params, L"MAX_HEIGHT");
    const int every_nth = get_int_param(params, L"EVERY_NTH");
    const int fps       = get_int_param(params, L"FPS");

    return spl::make_shared<spout_consumer_impl>(name, max_w, max_h, every_nth, fps);
}

}} // namespace caspar::spout
