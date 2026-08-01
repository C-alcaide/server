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

#include "spout_producer.h"

#include <core/producer/frame_producer.h>
#include <core/frame/frame_factory.h>
#include <core/frame/draw_frame.h>
#include <core/frame/frame.h>
#include <core/frame/pixel_format.h>

// The zero-copy receive needs a texture owned by the mixer's OpenGL device.
// Reaching that device means GLEW, and GLEW cannot share a translation unit
// with the Spout SDK -- both declare the GL framebuffer entry points, with
// different linkage. The GLEW side is behind this header.
#include "spout_gl_bridge.h"
#include <common/log.h>
#include <common/timer.h>
#include <common/diagnostics/graph.h>

#include <Spout.h>

#include <atomic>
#include <chrono>
#include <mutex>
#include <queue>
#include <sstream>
#include <thread>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <condition_variable>

#include <windows.h>
#include <gl/GL.h>

#pragma warning(push)
#pragma warning(disable: 4244)
extern "C" {
#include <libavutil/frame.h>
#include <libavutil/imgutils.h>
}
#include "../../ffmpeg/util/av_util.h"
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
    /// `share_context` is the mixer's HGLRC. Sharing lists with it is what makes
    /// a texture created on the mixer's device usable from this thread, and so
    /// what allows Spout to deliver straight into it instead of into host
    /// memory. Passing nullptr gives the previous behaviour: a standalone
    /// context, usable only for the readback path.
    explicit gl_context(void* share_context = nullptr)
    {
        WNDCLASS wc      = {0};
        wc.lpfnWndProc   = DefWindowProc;
        wc.hInstance     = GetModuleHandle(NULL);
        wc.lpszClassName = L"CasparCG_Spout_Producer_Context";
        RegisterClass(&wc);

        hwnd_ = CreateWindow(wc.lpszClassName, L"Spout Producer Context", 0, 0, 0, 0, 0, 0, 0, wc.hInstance, 0);

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

            if (hglrc_ && share_context) {
                if (wglShareLists(reinterpret_cast<HGLRC>(share_context), hglrc_)) {
                    shared_ = true;
                } else {
                    CASPAR_LOG(warning)
                        << L"[spout_producer] wglShareLists failed; falling back to the readback path.";
                }
            }
        }
    }

    bool is_shared() const { return shared_; }

    ~gl_context()
    {
        if (hglrc_) wglDeleteContext(hglrc_);
        if (hdc_) ReleaseDC(hwnd_, hdc_);
        if (hwnd_) DestroyWindow(hwnd_);
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

struct spout_producer : public core::frame_producer
{
    const std::wstring                   name_;
    std::string                          sender_name_ascii_;

    spl::shared_ptr<core::frame_factory> frame_factory_;

    // Non-null only on the OpenGL mixer. Spout shares a texture between
    // processes, so when the mixer is OpenGL the frame can be received straight
    // into a texture the mixer owns and never touch host memory. Any other
    // mixer keeps the readback path.
    ogl_device_handle                         ogl_device_;
    // Non-null only on the Vulkan mixer. There is no list sharing to exploit
    // there, so instead Vulkan exports an image's memory and GL imports it, and
    // Spout receives into a texture that aliases what the mixer samples.
    vk_device_handle                          vk_device_;
    void*                                     mixer_hglrc_ = nullptr;
    // Enough textures that one being read by the mixer is never the one being
    // written by the next receive. The mixer holds at most a couple of frames.
    static constexpr int                                          TEX_SLOTS = 4;
    std::vector<std::shared_ptr<core::texture>>                    tex_ring_;
    int                                                            tex_next_ = 0;
    std::atomic<bool>                                             gpu_path_active_{false};

    std::thread                          worker_thread_;
    std::atomic<bool>                    running_;

    std::queue<core::draw_frame>         frames_;
    mutable std::mutex                   frames_mutex_;

    spl::shared_ptr<diagnostics::graph>  graph_;
    caspar::timer                        frame_timer_;

    // FPS counter — updated in worker_loop when a frame is received
    std::chrono::steady_clock::time_point last_fps_update_{ std::chrono::steady_clock::now() };
    int    frames_since_update_ = 0;
    double current_fps_         = 0.0;

    spout_producer(const core::frame_producer_dependencies& dependencies,
                   const std::wstring& name)
        : frame_factory_(dependencies.frame_factory)
        , name_(name)
        , running_(true)
    {
        // sender_name_ascii_ = std::string(name.begin(), name.end());
        sender_name_ascii_.reserve(name.length());
        for(wchar_t c : name) {
            sender_name_ascii_.push_back(static_cast<char>(c));
        }
        if (sender_name_ascii_.empty()) {
            sender_name_ascii_ = "Spout Sender";
        }
        
        // The mixer's own GL context has to be fetched from the GL thread, and
        // only exists at all when the mixer is OpenGL.
        ogl_device_ = get_mixer_ogl_device(*frame_factory_);
        if (!ogl_device_)
            vk_device_ = get_mixer_vk_device(*frame_factory_);

        graph_ = spl::make_shared<diagnostics::graph>();
        graph_->set_text(print());
        graph_->set_color("frame-time", diagnostics::color(0.5f, 1.0f, 0.2f));
        graph_->set_color("buffer-size", diagnostics::color(0.2f, 0.5f, 1.0f));
        diagnostics::register_graph(graph_);
    }

    ~spout_producer()
    {
        running_ = false;
        if (worker_thread_.joinable()) {
            worker_thread_.join();
        }
    }

    void initialize()
    {
        worker_thread_ = std::thread([this]() {
            worker_loop();
        });
    }

    /// Shared by both receive paths so the reported rate means the same thing
    /// whichever one is running.
    void tick_fps()
    {
        const auto now = std::chrono::steady_clock::now();
        ++frames_since_update_;
        const auto dur =
            std::chrono::duration_cast<std::chrono::duration<double>>(now - last_fps_update_).count();
        if (dur >= 1.0) {
            current_fps_         = frames_since_update_ / dur;
            frames_since_update_ = 0;
            last_fps_update_     = now;
            std::wstringstream ss;
            ss << std::fixed << std::setprecision(2) << print() << L" Fps: " << current_fps_;
            graph_->set_text(ss.str());
        }
    }

    void worker_loop()
    {
        // Preferred: a context on the mixer's own DC, sharing its lists, so
        // Spout can deliver into a texture the mixer owns. Anything else (a
        // Vulkan mixer, or sharing refused) falls back to the standalone
        // context and the readback path below.
        void* shared_hglrc = nullptr;
        void* shared_hdc   = nullptr;
        // CASPAR_SPOUT_FORCE_READBACK exists to compare the two receive paths on
        // one mixer. Without it the zero-copy path is taken whenever it can be,
        // so the readback path is only ever seen on a non-OpenGL mixer and any
        // difference between them is confounded with the mixer.
        const bool force_readback = std::getenv("CASPAR_SPOUT_FORCE_READBACK") != nullptr;
        const bool zero_copy =
            !force_readback && ogl_device_ &&
            create_shared_context(ogl_device_, shared_hglrc, shared_hdc) &&
            wglMakeCurrent(reinterpret_cast<HDC>(shared_hdc), reinterpret_cast<HGLRC>(shared_hglrc));

        std::unique_ptr<gl_context> context;
        // On the Vulkan mixer there are no lists to share, so the receive goes
        // into a standalone context -- but into a texture whose memory is a
        // Vulkan image the mixer samples, so it still never reaches the host.
        bool vk_shared = false;
        if (!zero_copy) {
            destroy_shared_context(shared_hglrc);
            shared_hglrc = nullptr;
            context      = std::make_unique<gl_context>();
            if (!context->make_current()) {
                CASPAR_LOG(error) << "Spout Producer: Failed to create GL context";
                return;
            }
            vk_shared = !force_readback && vk_device_ != nullptr;
        }
        CASPAR_LOG(info) << L"[spout_producer] "
                         << (zero_copy ? L"GL context shared with the mixer -- receiving straight into a mixer "
                                         L"texture, no readback."
                             : vk_shared
                                 ? L"receiving into a GL texture backed by Vulkan mixer memory, no readback."
                                 : L"no shared GL context -- receiving into host memory (readback path).");

        // Declared after `context` so these are released while it is still
        // current: the GL names belong to it, and this thread is the only one
        // allowed to free them.
        std::vector<vk_shared_slot> vk_slots;
        int                         vk_next = 0;

        auto receiver = std::make_unique<Spout>();

        if (!sender_name_ascii_.empty())
            receiver->SetReceiverName(sender_name_ascii_.c_str());

        // Persistent receive buffer — reallocated only when the sender changes resolution.
        unsigned int     cur_w = 0, cur_h = 0;
        std::vector<uint8_t> pixel_buf;

        while (running_)
        {
            frame_timer_.restart();
            bool frame_received = false;

            // ---- (Re)connect phase -------------------------------------------------
            // ReceiveTexture() is called ONLY here, when we don't yet have a valid
            // connection or the sender changed resolution.  In steady state the loop
            // skips directly to ReceiveImage() so we never do two receive calls per frame.
            if (cur_w == 0 || cur_h == 0) {
                if (receiver->ReceiveTexture()) {
                    cur_w = receiver->GetSenderWidth();
                    cur_h = receiver->GetSenderHeight();
                    if (cur_w > 0 && cur_h > 0) {
                        if (zero_copy) {
                            // Allocated on the GL thread, which owns the device;
                            // shared lists make them visible from this one.
                            tex_ring_.clear();
                            for (int i = 0; i < TEX_SLOTS; ++i)
                                tex_ring_.push_back(create_mixer_texture(
                                    ogl_device_, static_cast<int>(cur_w), static_cast<int>(cur_h)));
                            tex_next_ = 0;
                        } else if (vk_shared) {
                            vk_slots = create_vk_shared_slots(
                                vk_device_, static_cast<int>(cur_w), static_cast<int>(cur_h), TEX_SLOTS);
                            vk_next = 0;
                            if (vk_slots.empty()) {
                                // The driver refused the import. Keep going on the
                                // path that always works rather than stopping.
                                vk_shared = false;
                                pixel_buf.resize(static_cast<size_t>(cur_w) * cur_h * 4);
                            }
                        } else {
                            pixel_buf.resize(static_cast<size_t>(cur_w) * cur_h * 4);
                        }
                    } else {
                        cur_w = cur_h = 0;
                    }
                }
                if (cur_w == 0) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(5));
                    continue;
                }
            }

            // ---- Steady-state receive phase ----------------------------------------
            // ---- Zero-copy receive ---------------------------------------------
            // Spout shares a GPU texture between processes. With lists shared
            // with the mixer, it can be copied GPU-to-GPU into a texture the
            // mixer owns, and the frame is handed over as that texture. The
            // readback path below pulls the same pixels through host memory --
            // 8.3 MB per 1080p frame down and the same back up -- to deliver an
            // identical picture.
            if (zero_copy && !tex_ring_.empty()) {
                unsigned int new_w = receiver->GetSenderWidth();
                unsigned int new_h = receiver->GetSenderHeight();
                if (new_w != cur_w || new_h != cur_h) {
                    cur_w = cur_h = 0;
                    tex_ring_.clear();
                } else {
                    auto& tex = tex_ring_[tex_next_];
                    tex_next_ = (tex_next_ + 1) % static_cast<int>(tex_ring_.size());

                    // Dimensions come from the connect phase above; this overload
                    // takes the destination texture and the invert flag only.
                    if (receiver->ReceiveTexture(texture_gl_id(tex), GL_TEXTURE_2D, false, 0)) {
                        // The mixer reads this texture on its own context, so the
                        // copy has to have landed before the frame is published.
                        glFinish();

                        core::pixel_format_desc pfd(core::pixel_format::bgra);
                        pfd.planes.push_back(core::pixel_format_desc::plane(
                            static_cast<int>(cur_w), static_cast<int>(cur_h), 4, common::bit_depth::bit8));

                        auto empty = std::make_shared<std::vector<uint8_t>>(0);
                        std::vector<array<const std::uint8_t>> img;
                        img.emplace_back(empty->data(), 0, std::move(empty));
                        auto astore = std::make_shared<std::vector<std::int32_t>>(0);
                        array<const std::int32_t> audio(astore->data(), 0, std::move(astore));

                        {
                            std::lock_guard<std::mutex> lock(frames_mutex_);
                            frames_.push(core::draw_frame(core::const_frame(
                                this, std::move(img), std::move(audio), pfd, tex)));
                            if (frames_.size() > 5) frames_.pop();
                        }
                        frame_received = true;
                        gpu_path_active_ = true;
                        tick_fps();
                    }
                }

                graph_->set_value("frame-time", frame_timer_.elapsed() * 1000.0);
                {
                    std::lock_guard<std::mutex> lock(frames_mutex_);
                    graph_->set_value("buffer-size", static_cast<double>(frames_.size()));
                }
                if (!frame_received)
                    std::this_thread::sleep_for(std::chrono::milliseconds(5));
                continue;
            }

            // ---- Vulkan mixer: receive into shared Vulkan memory ---------------
            // Same shape as the OpenGL branch above, and the same saving: Spout's
            // GPU-to-GPU copy lands in memory the mixer already owns. The
            // difference is only how the destination came to be reachable from
            // here -- imported Vulkan memory rather than a shared list.
            if (vk_shared && !vk_slots.empty()) {
                unsigned int new_w = receiver->GetSenderWidth();
                unsigned int new_h = receiver->GetSenderHeight();
                if (new_w != cur_w || new_h != cur_h) {
                    cur_w = cur_h = 0;
                    vk_slots.clear();
                } else {
                    auto& slot = vk_slots[vk_next];
                    vk_next    = (vk_next + 1) % static_cast<int>(vk_slots.size());

                    if (receiver->ReceiveTexture(slot.gl_id, GL_TEXTURE_2D, false, 0)) {
                        // Nothing else orders this write against the mixer's read.
                        glFinish();

                        // rgba, not bgra: see vk_shared_slot::frame_texture. The
                        // mixer's bgra case swizzles, and Spout has already left
                        // these bytes in RGBA order.
                        core::pixel_format_desc pfd(core::pixel_format::rgba);
                        pfd.planes.push_back(core::pixel_format_desc::plane(
                            static_cast<int>(cur_w), static_cast<int>(cur_h), 4, common::bit_depth::bit8));

                        auto                                   empty = std::make_shared<std::vector<uint8_t>>(0);
                        std::vector<array<const std::uint8_t>> img;
                        img.emplace_back(empty->data(), 0, std::move(empty));
                        auto                      astore = std::make_shared<std::vector<std::int32_t>>(0);
                        array<const std::int32_t> audio(astore->data(), 0, std::move(astore));

                        {
                            std::lock_guard<std::mutex> lock(frames_mutex_);
                            frames_.push(core::draw_frame(core::const_frame(
                                this, std::move(img), std::move(audio), pfd, slot.frame_texture)));
                            if (frames_.size() > 5)
                                frames_.pop();
                        }
                        frame_received   = true;
                        gpu_path_active_ = true;
                        tick_fps();
                    }
                }

                graph_->set_value("frame-time", frame_timer_.elapsed() * 1000.0);
                {
                    std::lock_guard<std::mutex> lock(frames_mutex_);
                    graph_->set_value("buffer-size", static_cast<double>(frames_.size()));
                }
                if (!frame_received)
                    std::this_thread::sleep_for(std::chrono::milliseconds(5));
                continue;
            }

            // ---- Readback path (any mixer other than OpenGL) -------------------
            // bInvert=false: what Spout shares is already top-down.
            //
            // This used to pass true, on the reasoning that a Spout texture is
            // an OpenGL texture and therefore bottom-up. The shared texture is
            // not what a sender handed over, though -- the SDK normalises on the
            // way in, which is why SendTexture defaults bInvert to true (a GL
            // texture is bottom-up, so invert it) while SendImage defaults it to
            // false (a pixel buffer already is not). Both defaults land on the
            // same orientation, and receiving with true then turned the picture
            // upside down. Demonstrated with this server's own Spout consumer
            // feeding this producer: identical content, inverted.
            if (receiver->ReceiveImage(pixel_buf.data(), GL_BGRA_EXT, false, 0)) {
                // Detect sender resolution change — will reconnect next iteration.
                unsigned int new_w = receiver->GetSenderWidth();
                unsigned int new_h = receiver->GetSenderHeight();
                if (new_w != cur_w || new_h != cur_h) {
                    cur_w = cur_h = 0;
                    pixel_buf.clear();
                } else {
                    // Copy pixels into an AVFrame owned by the draw_frame.
                    AVFrame* av_frame = av_frame_alloc();
                    av_frame->width  = static_cast<int>(cur_w);
                    av_frame->height = static_cast<int>(cur_h);
                    av_frame->format = AV_PIX_FMT_BGRA;
                    if (av_frame_get_buffer(av_frame, 32) >= 0) {
                        std::memcpy(av_frame->data[0], pixel_buf.data(), pixel_buf.size());

                        AVFrame* audio_frame = av_frame_alloc();
                        std::shared_ptr<AVFrame> sv(av_frame,    [](AVFrame* f){ av_frame_free(&f); });
                        std::shared_ptr<AVFrame> sa(audio_frame, [](AVFrame* f){ av_frame_free(&f); });

                        auto mframe = ffmpeg::make_frame(this, *frame_factory_, std::move(sv), std::move(sa));

                        std::lock_guard<std::mutex> lock(frames_mutex_);
                        frames_.push(core::draw_frame(std::move(mframe)));
                        if (frames_.size() > 5) frames_.pop();
                        frame_received = true;

                        // FPS counter — updated here, inside worker thread.
                        auto fps_now = std::chrono::steady_clock::now();
                        ++frames_since_update_;
                        const auto fps_dur = std::chrono::duration_cast<std::chrono::duration<double>>(fps_now - last_fps_update_).count();
                        if (fps_dur >= 1.0) {
                            current_fps_         = frames_since_update_ / fps_dur;
                            frames_since_update_ = 0;
                            last_fps_update_     = fps_now;
                            std::wstringstream ss;
                            ss << std::fixed << std::setprecision(2) << print() << L" Fps: " << current_fps_;
                            graph_->set_text(ss.str());
                        }
                    } else {
                        av_frame_free(&av_frame);
                    }
                }
            } else if (!receiver->IsConnected()) {
                // Sender disappeared — reset so we re-enter the connect phase.
                cur_w = cur_h = 0;
                pixel_buf.clear();
            }

            graph_->set_value("frame-time", frame_timer_.elapsed() * 1000.0);
            {
                std::lock_guard<std::mutex> lock(frames_mutex_);
                graph_->set_value("buffer-size", static_cast<double>(frames_.size()));
            }

            if (!frame_received)
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }

        receiver->ReleaseReceiver();

        if (shared_hglrc) {
            wglMakeCurrent(nullptr, nullptr);
            destroy_shared_context(shared_hglrc);
        }
    }

    core::draw_frame receive_impl(core::video_field /*field*/, int /*nb_samples*/) override
    {
        std::lock_guard<std::mutex> lock(frames_mutex_);
        if (!frames_.empty()) {
            auto f = frames_.front();
            frames_.pop();
            return f;
        }
        return {};
    }

    caspar::core::monitor::state state() const override 
    { 
        return caspar::core::monitor::state(); 
    }
    
    std::wstring print() const override
    {
        return L"SPOUT Producer: " + name_;
    }
    
    std::wstring name() const override 
    { 
        return L"SPOUT"; 
    }

    bool is_ready() override
    {
        std::lock_guard<std::mutex> lock(frames_mutex_);
        return !frames_.empty();
    }
};

spl::shared_ptr<core::frame_producer> create_spout_producer(
    const core::frame_producer_dependencies& dependencies,
    const std::vector<std::wstring>&         params)
{
    if (params.empty()) return core::frame_producer::empty();

    std::wstring name_arg = L"";
    bool match = false;
    
    if (params[0].find(L"[SPOUT]") == 0) {
        match = true;
        if(params.size() > 1) name_arg = params[1];
    } else if (params[0].find(L"spout://") == 0) {
        match = true;
        name_arg = params[0].substr(8);
    } else if (params[0] == L"SPOUT") {
         match = true;
         if(params.size() > 1) name_arg = params[1];
    }

    if (!match) return core::frame_producer::empty();

    auto producer = spl::make_shared<spout_producer>(dependencies, name_arg);
    producer->initialize();
    return producer;
}

}} // namespace