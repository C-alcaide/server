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
#include <mutex>
#include <future>
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
    static std::atomic<int>     instances_;
    std::string                 sender_name_;
    std::unique_ptr<Spout>      sender_;
    std::unique_ptr<gl_context> context_;
    core::video_format_desc     format_desc_;

    // Optional downscale cap (0 = no cap = native resolution).
    // Set via AMCP: ADD x SPOUT "Name" MAX_WIDTH 1920 MAX_HEIGHT 1080
    int max_w_ = 0;
    int max_h_ = 0;

    // Output dimensions (computed in initialize)
    int             out_w_ = 0;
    int             out_h_ = 0;
    bool            need_scale_ = false;
    std::vector<uint8_t> out_buf_;   // top-down BGRA output buffer (scaled or native)
    SwsContext*     sws_ctx_ = nullptr;

    spl::shared_ptr<diagnostics::graph> graph_;
    caspar::timer                       frame_timer_;
    const int                           instance_id_;

    spout_consumer_impl(std::wstring name, int max_w, int max_h)
        : instance_id_(instances_++)
        , max_w_(max_w)
        , max_h_(max_h)
    {
        sender_name_.reserve(name.length());
        for (wchar_t c : name)
            sender_name_.push_back(static_cast<char>(c));

        if (sender_name_.empty())
            sender_name_ = "CasparCG Spout";

        graph_ = spl::make_shared<diagnostics::graph>();
        graph_->set_text(print());
        graph_->set_color("frame-time", diagnostics::color(0.5f, 1.0f, 0.2f));
        diagnostics::register_graph(graph_);
    }

    ~spout_consumer_impl()
    {
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

        // Default: native resolution.  With MAX_WIDTH/MAX_HEIGHT, downscale while
        // preserving the square-pixel aspect ratio (square_width × square_height).
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
            need_scale_ = (out_w_ != format_desc.width || out_h_ != format_desc.height);
        } else {
            out_w_ = format_desc.width;
            out_h_ = format_desc.height;
            need_scale_ = false;
        }

        out_buf_.assign(static_cast<size_t>(out_w_) * out_h_ * 4, 0);

        if (need_scale_) {
            if (sws_ctx_) { sws_freeContext(sws_ctx_); sws_ctx_ = nullptr; }
            sws_ctx_ = sws_getContext(
                format_desc.width, format_desc.height, AV_PIX_FMT_BGRA,
                out_w_,            out_h_,             AV_PIX_FMT_BGRA,
                SWS_BILINEAR, nullptr, nullptr, nullptr);
        }
    }

    std::future<bool> send(const core::video_field field, core::const_frame frame) override
    {
        frame_timer_.restart();

        if (!context_)
            context_ = std::make_unique<gl_context>();
        if (!context_->make_current())
            return caspar::make_ready_future(false);

        if (!sender_) {
            sender_ = std::make_unique<Spout>();
            sender_->SetSenderName(sender_name_.c_str());
        }

        if (!frame.width() || !frame.height() || out_w_ == 0 || out_h_ == 0)
            return caspar::make_ready_future(true);

        // Source: packed BGRA, width × height, row stride = width * 4.
        const int      src_w        = format_desc_.width;
        const int      src_h        = format_desc_.height;
        const int      src_stride   = src_w * 4;
        const uint8_t* src          = reinterpret_cast<const uint8_t*>(frame.image_data(0).data());
        const int      out_stride   = out_w_ * 4;

        if (need_scale_ && sws_ctx_) {
            // Scale into out_buf_ (top-down).
            const uint8_t* sp[4] = { src,             nullptr, nullptr, nullptr };
            const int      ss[4] = { src_stride,       0,       0,       0      };
            uint8_t*       dp[4] = { out_buf_.data(), nullptr, nullptr, nullptr };
            const int      ds[4] = { out_stride,       0,       0,       0      };
            sws_scale(sws_ctx_, sp, ss, 0, src_h, dp, ds);
        } else {
            // Native resolution: copy rows directly into out_buf_ (top-down).
            std::memcpy(out_buf_.data(), src, static_cast<size_t>(src_stride) * src_h);
        }

        // Send top-down BGRA data as-is. The Spout SDK (both GL-DX and CPU/DX
        // paths) expects top-down pixel data when bInvert=false:
        //   - CPU path (WriteDX11pixels): row 0 → DX texture row 0 (top) = correct
        //   - GL path (WriteGLDXpixels): glTexSubImage2D row 0 → GL y=0 (bottom);
        //     the GL-DX interop implicitly maps GL-bottom ↔ DX-top, so row 0 of
        //     top-down data ends up at DX top = correct
        // A manual Y-flip would reverse the correct orientation in both paths.
        sender_->SendImage(out_buf_.data(),
                           static_cast<unsigned int>(out_w_),
                           static_cast<unsigned int>(out_h_),
                           GL_BGRA_EXT,
                           false);  // bInvert=false: data is top-down, no extra flip needed

        graph_->set_value("frame-time", frame_timer_.elapsed() * 1000.0);
        return caspar::make_ready_future(true);
    }

    std::wstring print() const override
    {
        std::wstring wname(sender_name_.begin(), sender_name_.end());
        return L"SPOUT Consumer: " + wname;
    }

    std::wstring name() const override { return L"SPOUT"; }

    int index() const override { return instance_id_; }

    caspar::core::monitor::state state() const override { return {}; }
};

std::atomic<int> spout_consumer_impl::instances_{0};

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
