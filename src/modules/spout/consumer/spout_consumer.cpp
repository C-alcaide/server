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

// Maximum Spout output dimensions — textures larger than this are downscaled.
// Spout shared memory grows as W*H*4 bytes; very large textures (e.g. 6000×1700)
// can cause GL driver failures or silent fallbacks that corrupt the image.
static constexpr int SPOUT_MAX_W = 1920;
static constexpr int SPOUT_MAX_H = 1080;

struct spout_consumer_impl : public core::frame_consumer
{
    static std::atomic<int>     instances_;
    std::string                 sender_name_;
    std::unique_ptr<Spout>      sender_;
    std::unique_ptr<gl_context> context_;
    bool                        initialized_ = false;
    core::video_format_desc     format_desc_;

    // Scaling state — reused across frames
    int             spout_w_ = 0;
    int             spout_h_ = 0;
    std::vector<uint8_t> out_buf_;   // final BGRA bottom-up buffer sent to Spout
    SwsContext*     sws_ctx_ = nullptr;

    spl::shared_ptr<diagnostics::graph> graph_;
    caspar::timer                       frame_timer_;
    const int                           instance_id_;

    ~spout_consumer_impl()
    {
        if (sender_) {
            sender_->ReleaseSender();
        }
        if (sws_ctx_) {
            sws_freeContext(sws_ctx_);
            sws_ctx_ = nullptr;
        }
    }

    spout_consumer_impl(std::wstring name)
        : instance_id_(instances_++)
    {
        // Simple manual conversion from wstring to string (lossy for non-ascii but sufficient for Spout names usually)
        sender_name_.reserve(name.length());
        for(wchar_t c : name) {
            sender_name_.push_back(static_cast<char>(c));
        }

        if (sender_name_.empty()) {
            sender_name_ = "CasparCG Spout";
        }
        
        graph_ = spl::make_shared<diagnostics::graph>();
        graph_->set_text(print());
        graph_->set_color("frame-time", diagnostics::color(0.5f, 1.0f, 0.2f));
        diagnostics::register_graph(graph_);
    }

    void initialize(const core::video_format_desc& format_desc, const core::channel_info& channel_info, int port_index) override
    {
        format_desc_ = format_desc;

        // Compute output dimensions: preserve square pixel aspect ratio,
        // capped to SPOUT_MAX_W × SPOUT_MAX_H.
        // Use square_width/square_height so anamorphic modes display correctly.
        const int sw = format_desc.square_width;
        const int sh = format_desc.square_height;

        double scale = 1.0;
        if (sw > SPOUT_MAX_W) scale = static_cast<double>(SPOUT_MAX_W) / sw;
        if (sh * scale > SPOUT_MAX_H) scale = static_cast<double>(SPOUT_MAX_H) / sh;

        // Round down to nearest even (GPU textures prefer even dimensions).
        // Use (std::max) form to prevent Windows.h min/max macro expansion.
        const int raw_w = static_cast<int>(sw * scale);
        const int raw_h = static_cast<int>(sh * scale);
        spout_w_ = (std::max)(2, raw_w - (raw_w % 2));
        spout_h_ = (std::max)(2, raw_h - (raw_h % 2));

        out_buf_.assign(static_cast<size_t>(spout_w_) * spout_h_ * 4, 0);

        // Recreate swscale context
        if (sws_ctx_) { sws_freeContext(sws_ctx_); sws_ctx_ = nullptr; }
        sws_ctx_ = sws_getContext(
            format_desc.width,  format_desc.height,  AV_PIX_FMT_BGRA,
            spout_w_,           spout_h_,            AV_PIX_FMT_BGRA,
            SWS_BILINEAR, nullptr, nullptr, nullptr);
    }

    std::future<bool> send(const core::video_field field, core::const_frame frame) override
    {
        frame_timer_.restart();

        // Spout requires an active OpenGL context on the calling thread.
        if (!context_) {
            context_ = std::make_unique<gl_context>();
        }

        if (context_) { 
             if(!context_->make_current()) {
                 return caspar::make_ready_future(false);
             }
        }
        
        if (!sender_) {
            sender_ = std::make_unique<Spout>();
            sender_->SetSenderName(sender_name_.c_str());
            initialized_ = true;
        }

        // Check if frame is valid
        if (!frame.width() || !frame.height() || !sws_ctx_) return caspar::make_ready_future(true);

        // Source buffer: format_desc_.width × format_desc_.height, packed BGRA.
        // (Mixer always produces packed BGRA; linesize == width*4.)
        const int src_w = format_desc_.width;
        const int src_linesize = src_w * 4;
        const uint8_t* src = reinterpret_cast<const uint8_t*>(frame.image_data(0).data());

        // Scale down to spout_w_ × spout_h_ (square-pixel, fits in SPOUT_MAX).
        // sws_scale writes top-down BGRA into out_buf_.
        {
            const uint8_t* src_planes[4] = { src, nullptr, nullptr, nullptr };
            const int      src_strides[4] = { src_linesize, 0, 0, 0 };
            uint8_t*       dst_planes[4] = { out_buf_.data(), nullptr, nullptr, nullptr };
            const int      dst_strides[4] = { spout_w_ * 4, 0, 0, 0 };
            sws_scale(sws_ctx_, src_planes, src_strides, 0, format_desc_.height,
                      dst_planes, dst_strides);
        }

        // Flip rows in-place: CasparCG frames are top-down; Spout/GL expects bottom-up.
        {
            const int row_bytes = spout_w_ * 4;
            for (int y = 0; y < spout_h_ / 2; ++y) {
                uint8_t* a = out_buf_.data() + static_cast<size_t>(y)               * row_bytes;
                uint8_t* b = out_buf_.data() + static_cast<size_t>(spout_h_ - 1 - y) * row_bytes;
                // swap rows via XOR — no extra buffer needed
                for (int x = 0; x < row_bytes; ++x) {
                    a[x] ^= b[x]; b[x] ^= a[x]; a[x] ^= b[x];
                }
            }
        }

        sender_->SendImage(
            out_buf_.data(),
            static_cast<unsigned int>(spout_w_),
            static_cast<unsigned int>(spout_h_),
            GL_BGRA_EXT,
            false  // already bottom-up after manual flip
        );

        // Update diagnostics
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
    
    caspar::core::monitor::state state() const override 
    { 
        return caspar::core::monitor::state(); 
    }
};

std::atomic<int> spout_consumer_impl::instances_{0};

spl::shared_ptr<core::frame_consumer> create_spout_consumer(
    const std::vector<std::wstring>&                         params,
    const core::video_format_repository&                     format_repository,
    const std::vector<spl::shared_ptr<core::video_channel>>& channels,
    const core::channel_info&                                channel_info)
{
    if (params.empty() || params[0] != L"SPOUT") {
        return core::frame_consumer::empty();
    }

    std::wstring name = L"";
    if (params.size() > 1) {
        name = params[1];
    }

    return spl::make_shared<spout_consumer_impl>(name);
}

}} // namespace caspar::spout
