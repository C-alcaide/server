#include "spout_producer.h"

#include <core/producer/frame_producer.h>
#include <core/frame/frame_factory.h>
#include <core/frame/draw_frame.h>
#include <common/log.h>
#include <common/timer.h>
#include <common/diagnostics/graph.h>

#include <Spout.h>

#include <atomic>
#include <mutex>
#include <queue>
#include <thread>
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

  public:
    gl_context()
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
        }
    }

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
    static std::atomic<int>              instances_;
    const int                            instance_no_;
    const std::wstring                   name_;
    std::string                          sender_name_ascii_;

    spl::shared_ptr<core::frame_factory> frame_factory_;
    
    std::thread                          worker_thread_;
    std::atomic<bool>                    running_;

    std::queue<core::draw_frame>         frames_;
    mutable std::mutex                   frames_mutex_;

    spl::shared_ptr<diagnostics::graph>  graph_;
    caspar::timer                        frame_timer_;

    spout_producer(const core::frame_producer_dependencies& dependencies,
                   const std::wstring& name)
        : frame_factory_(dependencies.frame_factory)
        , instance_no_(instances_++)
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

    void worker_loop()
    {
        auto context = std::make_unique<gl_context>();
        if (!context->make_current()) {
            CASPAR_LOG(error) << "Spout Producer: Failed to create GL context";
            return;
        }

        auto receiver = std::make_unique<Spout>();
        
        if (!sender_name_ascii_.empty()) {
            receiver->SetReceiverName(sender_name_ascii_.c_str());
        }

        while (running_) 
        {
            frame_timer_.restart();
            bool frame_received = false;

            if (receiver->ReceiveTexture()) {
                unsigned int width = receiver->GetSenderWidth();
                unsigned int height = receiver->GetSenderHeight();
                
                if (width > 0 && height > 0) {
                    AVFrame* av_frame = av_frame_alloc();
                    av_frame->width = width;
                    av_frame->height = height;
                    av_frame->format = AV_PIX_FMT_BGRA;
                    
                    if (av_frame_get_buffer(av_frame, 32) >= 0) {
                        if (receiver->ReceiveImage(av_frame->data[0], GL_BGRA_EXT, false, 0)) {
                             AVFrame* audio_frame = av_frame_alloc();
                             
                             std::shared_ptr<AVFrame> shared_video(av_frame, [](AVFrame* f){ av_frame_free(&f); });
                             std::shared_ptr<AVFrame> shared_audio(audio_frame, [](AVFrame* f){ av_frame_free(&f); });

                             auto mframe = ffmpeg::make_frame(this, *frame_factory_, std::move(shared_video), std::move(shared_audio));
                             core::draw_frame dframe(std::move(mframe));

                             {
                                 std::lock_guard<std::mutex> lock(frames_mutex_);
                                 frames_.push(dframe);
                                 if (frames_.size() > 5) frames_.pop();
                             }
                             frame_received = true;
                        } else {
                            av_frame_free(&av_frame);
                        }
                    } else {
                         av_frame_free(&av_frame);
                    }
                }
            }
            
            graph_->set_value("frame-time", frame_timer_.elapsed() * 1000.0);
            {
               std::lock_guard<std::mutex> lock(frames_mutex_);
               graph_->set_value("buffer-size", static_cast<double>(frames_.size()));
            }

            if (!frame_received) {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
        }
        
        receiver->ReleaseReceiver();
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

std::atomic<int> spout_producer::instances_{0};

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