#include "spout_consumer.h"

#include <Spout.h>
#include <common/executor.h>
#include <common/diagnostics/graph.h>
#include <common/timer.h>
#include <core/frame/frame.h>
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

// For Context Creation
#include <windows.h>
#include <gl/GL.h>

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
    bool                        initialized_ = false;

    spl::shared_ptr<diagnostics::graph> graph_;
    caspar::timer                       frame_timer_;
    const int                           instance_id_;

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

    ~spout_consumer_impl()
    {
        if (sender_) {
            sender_->ReleaseSender();
        }
    }

    void initialize(const core::video_format_desc& format_desc, const core::channel_info& channel_info, int port_index) override
    {
       // Initialization logic can go here
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
            // Use instance name
            sender_->SetSenderName(sender_name_.c_str());
            initialized_ = true;
        }

        // Check if frame is valid
        if(!frame.width() || !frame.height()) return caspar::make_ready_future(true);

        auto width  = frame.width();
        auto height = frame.height();
        
        const void* data = frame.image_data(0).data(); 

        sender_->SendImage(
            reinterpret_cast<const unsigned char*>(data), 
            static_cast<unsigned int>(width), 
            static_cast<unsigned int>(height), 
            GL_BGRA_EXT, 
            false
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
