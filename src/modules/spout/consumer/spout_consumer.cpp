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
#include <core/frame/frame.h>
#include <core/frame/pixel_format.h>
#include <core/consumer/frame_consumer.h>
#include <core/video_format.h>
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

// For Context Creation
#include <windows.h>
#include <gl/GL.h>

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

  public:
    gl_context()
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

            hglrc_ = wglCreateContext(hdc_);
        }
    }

    ~gl_context()
    {
        if (hglrc_) {
            wglDeleteContext(hglrc_);
        }
        if (hdc_) {
            ReleaseDC(hwnd_, hdc_);
        }
        if (hwnd_) {
            DestroyWindow(hwnd_);
        }
    }

    bool make_current()
    {
        if (hdc_ && hglrc_) {
            return wglMakeCurrent(hdc_, hglrc_) == TRUE;
        }
        return false;
    }
};

} // namespace

struct spout_consumer_impl : public core::frame_consumer
{
    std::string                 sender_name_;
    std::unique_ptr<Spout>      sender_;
    std::unique_ptr<gl_context> context_;
    core::video_format_desc     format_desc_;

    // Optional downscale cap (0 = no cap = native resolution).
    // Set via AMCP: ADD x SPOUT "Name" MAX_WIDTH 1920 MAX_HEIGHT 1080
    int max_w_ = 0;
    int max_h_ = 0;

    // Output dimensions (computed in initialize)
    int  out_w_ = 0;
    int  out_h_ = 0;

    std::vector<uint8_t> out_buf_;  // top-down BGRA8 output buffer

    // swscale: lazily rebuilt on executor when format or dimensions change
    AVPixelFormat src_av_fmt_ = AV_PIX_FMT_NONE;
    SwsContext*   sws_ctx_    = nullptr;

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

    spout_consumer_impl(std::wstring name, int max_w, int max_h)
        : max_w_(max_w)
        , max_h_(max_h)
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
        // Drain the executor before releasing resources it uses on its thread.
        executor_.invoke([] {});
        if (sender_)
            sender_->ReleaseSender();
        if (sws_ctx_) {
            sws_freeContext(sws_ctx_);
            sws_ctx_ = nullptr;
        }
    }

    void initialize(const core::video_format_desc& format_desc, const core::channel_info& channel_info, int port_index) override
    {
        format_desc_ = format_desc;

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

        out_buf_.assign(static_cast<size_t>(out_w_) * out_h_ * 4, 0);

        // Force swscale rebuild on next frame (dimensions may have changed).
        src_av_fmt_ = AV_PIX_FMT_NONE;
        if (sws_ctx_) { sws_freeContext(sws_ctx_); sws_ctx_ = nullptr; }
    }

    std::future<bool> send(const core::video_field field, core::const_frame frame) override
    {
        // Quick pre-checks on the calling (video) thread — no heavy work here.
        if (!frame.width() || !frame.height() || out_w_ == 0 || out_h_ == 0)
            return caspar::make_ready_future(true);

        const AVPixelFormat src_fmt = caspar_to_av_fmt(frame.pixel_format_desc());
        if (src_fmt == AV_PIX_FMT_NONE)
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

            if (!context_)
                context_ = std::make_unique<gl_context>();
            if (!context_->make_current()) {
                busy_ = false;
                return;
            }

            if (!sender_) {
                sender_ = std::make_unique<Spout>();
                sender_->SetSenderName(sender_name_.c_str());
            }

            // Rebuild swscale when format or output dimensions change.
            if (src_fmt != src_av_fmt_ || !sws_ctx_) {
                if (sws_ctx_) { sws_freeContext(sws_ctx_); sws_ctx_ = nullptr; }
                src_av_fmt_ = src_fmt;
                // Use FAST_BILINEAR: much faster than BILINEAR for large frames;
                // quality difference is imperceptible at monitor preview sizes.
                sws_ctx_ = sws_getContext(
                    src_w, src_h, src_fmt,
                    out_w_, out_h_, AV_PIX_FMT_BGRA,
                    SWS_FAST_BILINEAR, nullptr, nullptr, nullptr);
                if (!sws_ctx_) { busy_ = false; return; }
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

            // Output: packed BGRA8 into out_buf_
            uint8_t* dp[4] = { out_buf_.data(), nullptr, nullptr, nullptr };
            int      ds[4] = { out_w_ * 4,      0,       0,       0       };
            sws_scale(sws_ctx_, sp, ss, 0, src_h, dp, ds);

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

    caspar::core::monitor::state state() const override { return {}; }
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

    // Optional: ADD x SPOUT "Name" MAX_WIDTH 1920 MAX_HEIGHT 1080
    const int max_w = get_int_param(params, L"MAX_WIDTH");
    const int max_h = get_int_param(params, L"MAX_HEIGHT");

    return spl::make_shared<spout_consumer_impl>(name, max_w, max_h);
}

}} // namespace caspar::spout
