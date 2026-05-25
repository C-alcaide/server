/*
 * Copyright (c) 2011 Sveriges Television AB <info@casparcg.com>
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
 * Author: Robert Nagy, ronag89@gmail.com
 */

#include "screen_consumer.h"

#include <GL/glew.h>
#include <SFML/Window.hpp>

#include <common/array.h>
#include <common/diagnostics/graph.h>
#include <common/future.h>
#include <common/gl/gl_check.h>
#include <common/log.h>
#include <common/memory.h>
#include <common/param.h>
#include <common/timer.h>
#include <common/utf.h>

#include <core/consumer/channel_info.h>
#include <core/consumer/frame_consumer.h>
#include <core/frame/frame.h>
#include <core/frame/geometry.h>
#include <core/video_format.h>
#include <core/diagnostics/osd_graph.h>

#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/property_tree/ptree.hpp>

#include <tbb/concurrent_queue.h>

#include <iomanip>
#include <mutex>
#include <thread>
#include <tuple> // std::ignore
#include <utility>
#include <vector>

#if defined(_MSC_VER)
#include <windows.h>

#pragma warning(push)
#pragma warning(disable : 4244)
#else
#include "../util/x11_util.h"
#endif

#include <chrono>
#include <sstream>
#include <iomanip>

#include "consumer_screen_fragment.h"
#include "consumer_screen_vertex.h"
#include <accelerator/ogl/util/shader.h>
#include <accelerator/ogl/util/texture.h>

#if defined(_MSC_VER)
#include <GL/wglew.h>
#endif

#ifdef _MSC_VER
// ---------------------------------------------------------------------------
// win32_gl_window — lightweight Win32 + WGL window that replaces sf::Window.
//
// SFML2's shared-context mechanism (wglShareLists on a static singleton)
// permanently corrupts WGL state after a window is closed, making subsequent
// sf::Window::create() calls crash.  This class creates standalone WGL
// contexts with NO shared context, eliminating the race entirely.
// ---------------------------------------------------------------------------
namespace {

struct win32_gl_window
{
    HWND   hwnd_   = nullptr;
    HDC    hdc_    = nullptr;
    HGLRC  hglrc_  = nullptr;
    int    width_  = 0;
    int    height_ = 0;
    bool   resized_ = false;
    bool   closed_  = false;

    bool   shared_ = false;

    // WGL extension function pointers (loaded during bootstrap)
    PFNWGLCREATECONTEXTATTRIBSARBPROC wglCreateContextAttribsARB_ = nullptr;
    PFNWGLSWAPINTERVALEXTPROC         wglSwapIntervalEXT_         = nullptr;

    static LRESULT CALLBACK WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
    {
        auto* self = reinterpret_cast<win32_gl_window*>(GetWindowLongPtr(hWnd, GWLP_USERDATA));
        switch (msg) {
            case WM_SIZE:
                if (self) {
                    self->width_   = LOWORD(lParam);
                    self->height_  = HIWORD(lParam);
                    self->resized_ = true;
                }
                return 0;
            case WM_CLOSE:
                if (self) {
                    self->closed_ = true;
                }
                return 0;  // Don't let DefWindowProc destroy the window
            case WM_ERASEBKGND:
                return 1;  // Prevent flicker
            case WM_PRINTCLIENT:
            {
                // Handle PrintWindow capture by reading the GL front buffer
                // and painting it to the provided DC.  This runs on the render
                // thread (pollEvents dispatches here) so the GL context is current.
                if (self && self->width_ > 0 && self->height_ > 0) {
                    HDC target_dc = reinterpret_cast<HDC>(wParam);
                    int w = self->width_;
                    int h = self->height_;

                    // Read the front buffer (what's currently displayed)
                    std::vector<uint8_t> pixels(w * h * 4);
                    glReadBuffer(GL_FRONT);
                    glReadPixels(0, 0, w, h, GL_BGRA_EXT, GL_UNSIGNED_BYTE, pixels.data());

                    // glReadPixels returns bottom-up; SetDIBitsToDevice with
                    // negative height handles the flip automatically.
                    BITMAPINFO bmi       = {};
                    bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
                    bmi.bmiHeader.biWidth    = w;
                    bmi.bmiHeader.biHeight   = h;  // positive = bottom-up (matches GL)
                    bmi.bmiHeader.biPlanes   = 1;
                    bmi.bmiHeader.biBitCount = 32;
                    bmi.bmiHeader.biCompression = BI_RGB;

                    SetDIBitsToDevice(target_dc,
                                      0, 0, w, h,
                                      0, 0, 0, h,
                                      pixels.data(), &bmi, DIB_RGB_COLORS);
                }
                return 0;
            }
            default:
                return DefWindowProc(hWnd, msg, wParam, lParam);
        }
    }

    static const wchar_t* window_class_name()
    {
        static std::once_flag flag;
        static const wchar_t* name = L"CasparCG_ScreenConsumer";
        std::call_once(flag, [&] {
            WNDCLASSW wc  = {};
            wc.style      = CS_OWNDC;
            wc.lpfnWndProc  = WndProc;
            wc.hInstance    = GetModuleHandle(nullptr);
            wc.lpszClassName = name;
            wc.hCursor      = LoadCursor(nullptr, IDC_ARROW);
            RegisterClassW(&wc);
        });
        return name;
    }

    void create(int w, int h, const std::string& title,
                bool borderless, bool fullscreen, bool resizable, bool closeable,
                void* share_context = nullptr)
    {
        // --- Step 1: Create the real OS window ---
        DWORD style   = WS_CLIPSIBLINGS | WS_CLIPCHILDREN;
        DWORD exStyle = 0;

        if (fullscreen || borderless) {
            style |= WS_POPUP;
        } else {
            style |= WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU;
            if (resizable)
                style |= WS_THICKFRAME | WS_MINIMIZEBOX | WS_MAXIMIZEBOX;
        }

        // Adjust rect so the CLIENT area is exactly w x h
        RECT rect = {0, 0, w, h};
        AdjustWindowRectEx(&rect, style, FALSE, exStyle);

        std::wstring wtitle(title.begin(), title.end());
        hwnd_ = CreateWindowExW(
            exStyle, window_class_name(), wtitle.c_str(), style,
            CW_USEDEFAULT, CW_USEDEFAULT,
            rect.right - rect.left, rect.bottom - rect.top,
            nullptr, nullptr, GetModuleHandle(nullptr), nullptr);

        if (!hwnd_)
            throw std::runtime_error("CreateWindowEx failed");

        // Store 'this' for WndProc dispatch
        SetWindowLongPtr(hwnd_, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(this));

        hdc_ = GetDC(hwnd_);
        if (!hdc_)
            throw std::runtime_error("GetDC failed");

        // --- Step 2: Set pixel format ---
        PIXELFORMATDESCRIPTOR pfd = {};
        pfd.nSize        = sizeof(pfd);
        pfd.nVersion     = 1;
        pfd.dwFlags      = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
        pfd.iPixelType   = PFD_TYPE_RGBA;
        pfd.cColorBits   = 32;
        pfd.cDepthBits   = 0;
        pfd.cStencilBits = 0;
        pfd.iLayerType   = PFD_MAIN_PLANE;

        int pf = ChoosePixelFormat(hdc_, &pfd);
        if (!pf || !SetPixelFormat(hdc_, pf, &pfd))
            throw std::runtime_error("SetPixelFormat failed");

        // --- Step 3: Bootstrap GL context ---
        // Create a basic GL context to load WGL extensions
        HGLRC temp_ctx = wglCreateContext(hdc_);
        if (!temp_ctx)
            throw std::runtime_error("wglCreateContext (bootstrap) failed");
        wglMakeCurrent(hdc_, temp_ctx);

        // Load WGL extension function pointers
        wglCreateContextAttribsARB_ = reinterpret_cast<PFNWGLCREATECONTEXTATTRIBSARBPROC>(
            wglGetProcAddress("wglCreateContextAttribsARB"));
        wglSwapIntervalEXT_ = reinterpret_cast<PFNWGLSWAPINTERVALEXTPROC>(
            wglGetProcAddress("wglSwapIntervalEXT"));

        if (wglCreateContextAttribsARB_) {
            // Create GL 4.5 Core profile context
            int attribs[] = {
                WGL_CONTEXT_MAJOR_VERSION_ARB, 4,
                WGL_CONTEXT_MINOR_VERSION_ARB, 5,
                WGL_CONTEXT_PROFILE_MASK_ARB,  WGL_CONTEXT_CORE_PROFILE_BIT_ARB,
                0
            };

            // Try shared context first (enables zero-copy OGL mixer texture binding)
            if (share_context) {
                hglrc_ = wglCreateContextAttribsARB_(hdc_, reinterpret_cast<HGLRC>(share_context), attribs);
                if (hglrc_) {
                    shared_ = true;
                    CASPAR_LOG(info) << "[screen_consumer] GL context shared with mixer — zero-copy OGL path enabled.";
                } else {
                    CASPAR_LOG(warning) << "[screen_consumer] Shared GL context creation failed (error="
                                        << GetLastError() << "), falling back to standalone context.";
                    // Re-establish bootstrap context — a failed wglCreateContextAttribsARB
                    // can leave the WGL state dirty on some NVIDIA drivers.
                    wglMakeCurrent(hdc_, temp_ctx);
                }
            }

            if (!hglrc_)
                hglrc_ = wglCreateContextAttribsARB_(hdc_, nullptr, attribs);

            wglMakeCurrent(nullptr, nullptr);
            wglDeleteContext(temp_ctx);

            if (!hglrc_)
                throw std::runtime_error("wglCreateContextAttribsARB failed");

            wglMakeCurrent(hdc_, hglrc_);
        } else {
            // Fallback: use the basic context as-is
            hglrc_ = temp_ctx;
        }

        width_  = w;
        height_ = h;

        ShowWindow(hwnd_, SW_SHOW);
        UpdateWindow(hwnd_);
    }

    void close()
    {
        if (hglrc_) {
            wglMakeCurrent(nullptr, nullptr);
            wglDeleteContext(hglrc_);
            hglrc_ = nullptr;
        }
        if (hdc_ && hwnd_) {
            ReleaseDC(hwnd_, hdc_);
            hdc_ = nullptr;
        }
        if (hwnd_) {
            DestroyWindow(hwnd_);
            hwnd_ = nullptr;
        }
    }

    ~win32_gl_window() { close(); }

    void setPosition(int x, int y)
    {
        if (hwnd_)
            SetWindowPos(hwnd_, nullptr, x, y, 0, 0, SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE);
    }

    void setMouseCursorVisible(bool visible)
    {
        // ShowCursor is per-thread reference-counted; we just set the class cursor
        if (hwnd_) {
            SetClassLongPtr(hwnd_, GCLP_HCURSOR,
                reinterpret_cast<LONG_PTR>(visible ? LoadCursor(nullptr, IDC_ARROW) : nullptr));
        }
    }

    bool setActive(bool active)
    {
        if (active)
            return wglMakeCurrent(hdc_, hglrc_) == TRUE;
        else
            return wglMakeCurrent(nullptr, nullptr) == TRUE;
    }

    HWND getSystemHandle() const { return hwnd_; }

    void display()
    {
        if (hdc_)
            SwapBuffers(hdc_);
    }

    void setVerticalSyncEnabled(bool enabled)
    {
        if (wglSwapIntervalEXT_)
            wglSwapIntervalEXT_(enabled ? 1 : 0);
    }

    struct Size { unsigned int x, y; };
    Size getSize() const
    {
        RECT r;
        if (hwnd_ && GetClientRect(hwnd_, &r))
            return {static_cast<unsigned int>(r.right), static_cast<unsigned int>(r.bottom)};
        return {static_cast<unsigned int>(width_), static_cast<unsigned int>(height_)};
    }

    // Process pending window messages; returns true if any were processed.
    // Sets resized_/closed_ flags for the caller to check.
    bool pollEvents()
    {
        resized_ = false;
        closed_  = false;
        MSG msg;
        bool had = false;
        while (PeekMessage(&msg, hwnd_, 0, 0, PM_REMOVE)) {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
            had = true;
        }
        return had;
    }
};

} // anonymous namespace
#endif // _MSC_VER

namespace caspar { namespace screen {

std::unique_ptr<accelerator::ogl::shader> get_shader()
{
    return std::make_unique<accelerator::ogl::shader>(std::string(reinterpret_cast<const char*>(vertex_shader)), std::string(reinterpret_cast<const char*>(fragment_shader)));
}

enum class stretch
{
    none,
    uniform,
    fill,
    uniform_to_fill
};

struct configuration
{
    enum class aspect_ratio
    {
        aspect_4_3 = 0,
        aspect_16_9,
        aspect_invalid,
    };

    enum class colour_spaces
    {
        RGB               = 0,
        datavideo_full    = 1,
        datavideo_limited = 2
    };

    std::wstring    name          = L"Screen consumer";
    int             screen_index  = 0;
    int             screen_x      = 0;
    int             screen_y      = 0;
    int             screen_width  = 0;
    int             screen_height = 0;
    screen::stretch stretch       = screen::stretch::fill;
    bool            windowed      = true;
    bool            key_only      = false;
    bool            sbs_key       = false;
    aspect_ratio    aspect        = aspect_ratio::aspect_invalid;
    bool            vsync         = false;
    bool            interactive   = true;
    bool            borderless    = false;
    bool            always_on_top = false;
    colour_spaces   colour_space  = colour_spaces::RGB;
    bool            high_bitdepth = false;
    bool            gpu_texture   = false;
    bool            no_taskbar    = false;
    bool            closeable     = true;
    bool            no_activate   = false;
    int             tone_map_op   = 0;    // 0=none(passthrough), 7=hlg_ootf
    float           display_peak_luminance = 1000.0f;
    int             channel_transfer = 0;  // set at init: 2=sdr/rec709, 3=pq, 4=hlg
};

struct frame
{
    GLuint                         pbo = 0;
    GLuint                         tex = 0;
    char*                          ptr = nullptr;
    std::shared_ptr<core::texture> texture;
    GLsync                         fence = nullptr;
};

struct screen_consumer;

struct display_strategy
{
    virtual ~display_strategy() {}
    virtual frame init_frame(const configuration& config, const core::video_format_desc& format_desc) = 0;
    virtual void  cleanup_frame(frame& frame)                                                         = 0;
    virtual void  do_tick(screen_consumer* self)                                                      = 0;
};
struct gpu_strategy;
struct host_strategy;

struct screen_consumer
{
    const configuration     config_;
    core::video_format_desc format_desc_;
    int                     channel_index_;

    std::vector<frame> frames_;

    int screen_width_  = format_desc_.width;
    int screen_height_ = format_desc_.height;
    int square_width_  = format_desc_.square_width;
    int square_height_ = format_desc_.square_height;
    int screen_x_      = 0;
    int screen_y_      = 0;

    void* gl_share_context_ = nullptr;

    std::vector<core::frame_geometry::coord> draw_coords_;

#ifdef _MSC_VER
    win32_gl_window window_;
#else
    sf::Window window_;
#endif

    spl::shared_ptr<diagnostics::graph> graph_;
    caspar::timer                       tick_timer_;

    // FPS counter
    std::chrono::steady_clock::time_point last_fps_update_;
    int                     frames_since_update_ = 0;
    double                  current_fps_ = 0.0;

    // Periodic drop diagnostics
    std::chrono::steady_clock::time_point diag_start_ = std::chrono::steady_clock::now();
    int diag_sends_   = 0;
    int diag_drops_   = 0;

    tbb::concurrent_bounded_queue<core::const_frame> frame_buffer_;

    std::unique_ptr<accelerator::ogl::shader> shader_;
    GLuint                                    vao_;
    GLuint                                    vbo_;

    std::atomic<bool> is_running_{true};
    std::thread       thread_;

    spl::shared_ptr<display_strategy> strategy_;

    screen_consumer(const screen_consumer&)            = delete;
    screen_consumer& operator=(const screen_consumer&) = delete;

  public:
    screen_consumer(const configuration& config, const core::video_format_desc& format_desc, int channel_index,
                    void* gl_share_context = nullptr, bool use_vulkan = false)
        : config_(config)
        , format_desc_(format_desc)
        , channel_index_(channel_index)
        , gl_share_context_(gl_share_context)
        , strategy_((config.gpu_texture || use_vulkan) ? spl::make_shared<display_strategy, gpu_strategy>()
                                                       : spl::make_shared<display_strategy, host_strategy>())
    {
        if (config_.gpu_texture) {
            CASPAR_LOG(info) << print() << " Using GPU texture for rendering.";
        } else if (use_vulkan) {
            CASPAR_LOG(info) << print() << " Auto-promoted to GPU mode (Vulkan mixer active).";
        } else {
            CASPAR_LOG(info) << print() << " Using frame copied to host for rendering.";
        }

        if (format_desc_.format == core::video_format::ntsc &&
            config_.aspect == configuration::aspect_ratio::aspect_4_3) {
            // Use default values which are 4:3.
        } else {
            if (config_.aspect == configuration::aspect_ratio::aspect_16_9) {
                square_width_ = format_desc.height * 16 / 9;
            } else if (config_.aspect == configuration::aspect_ratio::aspect_4_3) {
                square_width_ = format_desc.height * 4 / 3;
            }
        }

        frame_buffer_.set_capacity(1);

        graph_->set_color("tick-time", diagnostics::color(0.0f, 0.6f, 0.9f));
        graph_->set_color("frame-time", diagnostics::color(0.1f, 1.0f, 0.1f));
        graph_->set_color("dropped-frame", diagnostics::color(0.3f, 0.6f, 0.3f));
        graph_->set_text(print());
        diagnostics::register_graph(graph_);

#if defined(_MSC_VER)
        DISPLAY_DEVICE              d_device = {sizeof(d_device), 0};
        std::vector<DISPLAY_DEVICE> displayDevices;
        for (int n = 0; EnumDisplayDevices(nullptr, n, &d_device, NULL); ++n) {
            displayDevices.push_back(d_device);
        }

        if (config_.screen_index >= displayDevices.size()) {
            CASPAR_LOG(warning) << print() << L" Invalid screen-index: " << config_.screen_index;
        }

        DEVMODE devmode = {};
        if (!EnumDisplaySettings(displayDevices[config_.screen_index].DeviceName, ENUM_CURRENT_SETTINGS, &devmode)) {
            CASPAR_LOG(warning) << print() << L" Could not find display settings for screen-index: "
                                << config_.screen_index;
        }

        screen_x_      = devmode.dmPosition.x;
        screen_y_      = devmode.dmPosition.y;
        screen_width_  = devmode.dmPelsWidth;
        screen_height_ = devmode.dmPelsHeight;
#else
        if (config_.screen_index > 1) {
            CASPAR_LOG(warning) << print() << L" Screen-index is not supported on linux";
        }
#endif

        if (config.windowed) {
            screen_x_ += config.screen_x;
            screen_y_ += config.screen_y;

            if (config.screen_width > 0 && config.screen_height > 0) {
                screen_width_  = config.screen_width;
                screen_height_ = config.screen_height;
            } else if (config.screen_width > 0) {
                screen_width_  = config.screen_width;
                screen_height_ = square_height_ * config.screen_width / square_width_;
            } else if (config.screen_height > 0) {
                screen_height_ = config.screen_height;
                screen_width_  = square_width_ * config.screen_height / square_height_;
            } else {
                // Default to channel resolution, but clamp to the target display
                // so the window doesn't span across multiple monitors (which can
                // interfere with Vulkan output windows on adjacent displays).
                screen_width_  = (std::min)(square_width_,  screen_width_);
                screen_height_ = (std::min)(square_height_, screen_height_);
            }
        }

        thread_ = std::thread([this] {
            try {
#ifdef _MSC_VER
                // Win32 + WGL: create window and GL 4.5 core context directly,
                // bypassing SFML's shared-context mechanism which corrupts WGL
                // state after repeated create/close cycles.
                {
                    int w = config_.sbs_key ? screen_width_ * 2 : screen_width_;
                    int h = screen_height_;
                    bool fullscreen = !config_.windowed && !config_.borderless;
                    bool resizable  = config_.windowed && !config_.borderless;
                    window_.create(w, h, u8(print()),
                                   config_.borderless, fullscreen, resizable, config_.closeable,
                                   gl_share_context_);
                }

                window_.setPosition(screen_x_, screen_y_);
                window_.setMouseCursorVisible(config_.interactive);
                std::ignore = window_.setActive(true);
#else
                // Non-Windows: use SFML as before
#if SFML_VERSION_MAJOR >= 3
                    sf::VideoMode mode{
                        sf::Vector2u(config_.sbs_key ? screen_width_ * 2 : screen_width_, screen_height_),
                        sf::VideoMode::getDesktopMode().bitsPerPixel
                    };
                    sf::ContextSettings settings{
                        .depthBits         = 0,
                        .stencilBits       = 0,
                        .antiAliasingLevel = 0,
                        .majorVersion      = 4,
                        .minorVersion      = 5,
                        .attributeFlags    = sf::ContextSettings::Attribute::Core
                    };
                    auto state = config_.windowed || config_.borderless ? sf::State::Windowed : sf::State::Fullscreen;
                    auto style = config_.borderless ? sf::Style::None : sf::Style::Default;
                    window_.create(mode, u8(print()), style, state, settings);
#else
                    const auto    window_style = config_.borderless ? sf::Style::None
                                                 : config_.windowed ? sf::Style::Resize | sf::Style::Close
                                                                    : sf::Style::Fullscreen;
                    sf::VideoMode desktop      = sf::VideoMode::getDesktopMode();
                    sf::VideoMode mode(
                        config_.sbs_key ? screen_width_ * 2 : screen_width_, screen_height_, desktop.bitsPerPixel);
                    window_.create(mode,
                                   u8(print()),
                                   window_style,
                                   sf::ContextSettings(0, 0, 0, 4, 5, sf::ContextSettings::Attribute::Core));
#endif
                window_.setPosition(sf::Vector2i(screen_x_, screen_y_));
                window_.setMouseCursorVisible(config_.interactive);
                std::ignore = window_.setActive(true);
#endif // _MSC_VER

                if (config_.always_on_top) {
#ifdef _MSC_VER
                    HWND hwnd = window_.getSystemHandle();
                    SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE);
#else
                    window_always_on_top(window_);
#endif
                }

#ifdef _MSC_VER
                {
                    HWND     hwnd = window_.getSystemHandle();
                    LONG_PTR ex   = GetWindowLongPtr(hwnd, GWL_EXSTYLE);

                    if (config_.no_taskbar) {
                        ex = (ex | WS_EX_TOOLWINDOW) & ~(LONG_PTR)WS_EX_APPWINDOW;
                        SetWindowLongPtr(hwnd, GWL_EXSTYLE, ex);
                        // Style change requires a hide/show cycle to take effect
                        ShowWindow(hwnd, SW_HIDE);
                        ShowWindow(hwnd, SW_SHOW);
                        ex = GetWindowLongPtr(hwnd, GWL_EXSTYLE); // refresh after show
                    }

                    if (config_.no_activate) {
                        SetWindowLongPtr(hwnd, GWL_EXSTYLE, ex | WS_EX_NOACTIVATE);
                    }
                }
#endif

                if (glewInit() != GLEW_OK) {
                    CASPAR_THROW_EXCEPTION(gl::ogl_exception() << msg_info("Failed to initialize GLEW."));
                }

                if (!GLEW_VERSION_4_5 && (glewIsSupported("GL_ARB_sync GL_ARB_shader_objects GL_ARB_multitexture "
                                                          "GL_ARB_direct_state_access GL_ARB_texture_barrier") == 0u)) {
                    CASPAR_THROW_EXCEPTION(not_supported() << msg_info(
                                               "Your graphics card does not meet the minimum hardware requirements "
                                               "since it does not support OpenGL 4.5 or higher."));
                }

                GL(glGenVertexArrays(1, &vao_));
                GL(glGenBuffers(1, &vbo_));
                GL(glBindVertexArray(vao_));
                GL(glBindBuffer(GL_ARRAY_BUFFER, vbo_));

                shader_ = get_shader();
                shader_->use();
                shader_->set("background", 0);
                shader_->set("window_width", screen_width_);

                for (int n = 0; n < 2; ++n) {
                    frames_.push_back(strategy_->init_frame(config_, format_desc_));
                }

                GL(glDisable(GL_DEPTH_TEST));
                GL(glClearColor(0.0, 0.0, 0.0, 0.0));
                GL(glViewport(
                    0, 0, config_.sbs_key ? format_desc_.width * 2 : format_desc_.width, format_desc_.height));

                calculate_aspect();

                window_.setVerticalSyncEnabled(config_.vsync);
                if (config_.vsync) {
                    CASPAR_LOG(info) << print() << " Enabled vsync.";
                }

                shader_->set("colour_space", config_.colour_space);
                shader_->set("tone_map_op", config_.tone_map_op);
                shader_->set("display_peak_luminance", config_.display_peak_luminance);
                shader_->set("channel_transfer", config_.channel_transfer);
                if (config_.colour_space == configuration::colour_spaces::datavideo_full ||
                    config_.colour_space == configuration::colour_spaces::datavideo_limited) {
                    CASPAR_LOG(info) << print() << " Enabled colours conversion for DataVideo TC-100/TC-200 "
                                     << (config_.colour_space == configuration::colour_spaces::datavideo_full
                                             ? "(Full Range)."
                                             : "(Limited Range).");
                }

                glClear(GL_COLOR_BUFFER_BIT);
                window_.display();

                while (is_running_) {
                    tick();
                }
            } catch (tbb::user_abort&) {
            } catch (...) {
                CASPAR_LOG_CURRENT_EXCEPTION();
                is_running_ = false;
            }

            // Cleanup must not throw — the GL context may be invalid.
            try {
                // Drain any stale GL errors so cleanup calls start clean.
                while (glGetError() != GL_NO_ERROR) {}

                for (auto frame : frames_) {
                    strategy_->cleanup_frame(frame);
                }

                shader_.reset();
                glDeleteVertexArrays(1, &vao_);
                glDeleteBuffers(1, &vbo_);
            } catch (...) {
                CASPAR_LOG(warning) << print() << L" Exception during GL cleanup — ignoring.";
            }

#ifdef _MSC_VER
            // Win32 path: direct close, no SFML shared context involvement.
            window_.close();
#else
            // Non-Windows: close under the SFML mutex to prevent context races.
            {
                std::lock_guard<std::recursive_mutex> lock(
                    core::diagnostics::osd::sfml_context_mutex());
                std::ignore = window_.setActive(false);
                window_.close();
            }
#endif
        });
    }

    ~screen_consumer()
    {
        is_running_ = false;
        frame_buffer_.abort();
        thread_.join();
    }

    bool poll()
    {
        int       count = 0;
#ifdef _MSC_VER
        if (window_.pollEvents()) {
            count = 1;
            if (window_.resized_) calculate_aspect();
            if (window_.closed_ && config_.closeable) {
                is_running_ = false;
            }
        }
#elif SFML_VERSION_MAJOR >= 3
        while (const auto e = window_.pollEvent()) {
            count++;
            if (e->is<sf::Event::Resized>()) {
                calculate_aspect();
            } else if (e->is<sf::Event::Closed>()) {
                if (config_.closeable) {
                    is_running_ = false;
                }
            }
        }
#else
        sf::Event e;
        while (window_.pollEvent(e)) {
            count++;
            if (e.type == sf::Event::Resized) {
                calculate_aspect();
            } else if (e.type == sf::Event::Closed) {
                if (config_.closeable) {
                    is_running_ = false;
                }
            }
        }
#endif
        return count > 0;
    }

    void tick()
    {
        // Present the PREVIOUS frame first.  SwapBuffers may block waiting
        // for DWM composition — doing it here lets that wait overlap with
        // the mixer producing the next frame, instead of stalling the
        // output pipeline after rendering.
        window_.display();

        strategy_->do_tick(this);

        glFlush(); // Submit GL commands without blocking — SwapBuffers on next tick handles sync

        std::rotate(frames_.begin(), frames_.begin() + 1, frames_.end());

        graph_->set_value("tick-time", tick_timer_.elapsed() * format_desc_.fps * 0.5);
        tick_timer_.restart();
    }

    void draw()
    {
        GL(glBufferData(GL_ARRAY_BUFFER,
                        static_cast<GLsizeiptr>(sizeof(core::frame_geometry::coord)) * draw_coords_.size(),
                        draw_coords_.data(),
                        GL_STATIC_DRAW));

        auto stride = static_cast<GLsizei>(sizeof(core::frame_geometry::coord));

        auto vtx_loc = shader_->get_attrib_location("Position");
        auto tex_loc = shader_->get_attrib_location("TexCoordIn");

        GL(glEnableVertexAttribArray(vtx_loc));
        GL(glEnableVertexAttribArray(tex_loc));

        GL(glVertexAttribPointer(vtx_loc, 2, GL_DOUBLE, GL_FALSE, stride, nullptr));
        GL(glVertexAttribPointer(tex_loc, 4, GL_DOUBLE, GL_FALSE, stride, (GLvoid*)(2 * sizeof(GLdouble))));

        shader_->set("window_width", screen_width_);

        if (config_.sbs_key) {
            auto coords_size = static_cast<GLsizei>(draw_coords_.size());

            // First half fill
            shader_->set("key_only", false);
            GL(glDrawArrays(GL_TRIANGLES, 0, coords_size / 2));

            // Second half key
            shader_->set("key_only", true);
            GL(glDrawArrays(GL_TRIANGLES, coords_size / 2, coords_size / 2));
        } else {
            shader_->set("key_only", config_.key_only);
            GL(glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(draw_coords_.size())));
        }

        GL(glDisableVertexAttribArray(vtx_loc));
        GL(glDisableVertexAttribArray(tex_loc));
        GL(glBindTexture(GL_TEXTURE_2D, 0));
    }

    std::future<bool> send(core::video_field field, const core::const_frame& frame)
    {
        // Screen is a progressive display — skip field B on interlaced channels.
        if (field == core::video_field::b)
            return caspar::make_ready_future(is_running_.load());

        // FPS Calc
        auto now = std::chrono::steady_clock::now();
        frames_since_update_++;
        auto duration_sec = std::chrono::duration_cast<std::chrono::duration<double>>(now - last_fps_update_).count();
        
        if (duration_sec >= 1.0) {
            current_fps_ = (double)frames_since_update_ / duration_sec;
            frames_since_update_ = 0;
            last_fps_update_ = now;
            
            std::wstringstream stats;
            stats.precision(2);
            stats << std::fixed;
            stats << print() << L" Fps: " << current_fps_;
            graph_->set_text(stats.str());
        }

        diag_sends_++;
        if (!frame_buffer_.try_push(frame)) {
            // Buffer full — drop the oldest frame and push the latest.
            // Never block the output thread; screen is a preview consumer.
            core::const_frame discard;
            frame_buffer_.try_pop(discard);
            if (!frame_buffer_.try_push(frame)) {
                graph_->set_tag(diagnostics::tag_severity::WARNING, "dropped-frame");
                diag_drops_++;
            }
        }

        // Periodic TIMING log every 5 seconds
        {
            auto diag_now = std::chrono::steady_clock::now();
            auto diag_elapsed = std::chrono::duration<double>(diag_now - diag_start_).count();
            if (diag_elapsed >= 5.0 && diag_sends_ > 0) {
                CASPAR_LOG(trace) << print() << L" TIMING: sends=" << diag_sends_
                                  << L" drops=" << diag_drops_;
                diag_start_ = diag_now;
                diag_sends_ = 0;
                diag_drops_ = 0;
            }
        }

        return caspar::make_ready_future(is_running_.load());
    }

    std::wstring channel_and_format() const
    {
        return L"[" + std::to_wstring(channel_index_) + L"|" + format_desc_.name + L"]";
    }

    std::wstring print() const { return config_.name + L" " + channel_and_format(); }

    void calculate_aspect()
    {
        if (config_.windowed) {
            screen_height_ = window_.getSize().y;
            screen_width_  = window_.getSize().x;
        }

        GL(glViewport(0, 0, screen_width_, screen_height_));

        std::pair<float, float> target_ratio = none();
        if (config_.stretch == screen::stretch::fill) {
            target_ratio = Fill();
        } else if (config_.stretch == screen::stretch::uniform) {
            target_ratio = uniform();
        } else if (config_.stretch == screen::stretch::uniform_to_fill) {
            target_ratio = uniform_to_fill();
        }

        if (config_.sbs_key) {
            draw_coords_ = {
                // First half fill
                {-target_ratio.first, target_ratio.second, 0.0, 0.0}, // upper left
                {0, target_ratio.second, 1.0, 0.0},                   // upper right
                {0, -target_ratio.second, 1.0, 1.0},                  // lower right

                {-target_ratio.first, target_ratio.second, 0.0, 0.0},  // upper left
                {0, -target_ratio.second, 1.0, 1.0},                   // lower right
                {-target_ratio.first, -target_ratio.second, 0.0, 1.0}, // lower left

                // Second half key
                {0, target_ratio.second, 0.0, 0.0},                   // upper left
                {target_ratio.first, target_ratio.second, 1.0, 0.0},  // upper right
                {target_ratio.first, -target_ratio.second, 1.0, 1.0}, // lower right

                {0, target_ratio.second, 0.0, 0.0},                   // upper left
                {target_ratio.first, -target_ratio.second, 1.0, 1.0}, // lower right
                {0, -target_ratio.second, 0.0, 1.0}                   // lower left
            };
        } else {
            draw_coords_ = {
                //    vertex    texture
                {-target_ratio.first, target_ratio.second, 0.0, 0.0}, // upper left
                {target_ratio.first, target_ratio.second, 1.0, 0.0},  // upper right
                {target_ratio.first, -target_ratio.second, 1.0, 1.0}, // lower right

                {-target_ratio.first, target_ratio.second, 0.0, 0.0}, // upper left
                {target_ratio.first, -target_ratio.second, 1.0, 1.0}, // lower right
                {-target_ratio.first, -target_ratio.second, 0.0, 1.0} // lower left
            };
        }
    }

    std::pair<float, float> none() const
    {
        float width =
            static_cast<float>(config_.sbs_key ? square_width_ * 2 : square_width_) / static_cast<float>(screen_width_);
        float height = static_cast<float>(square_height_) / static_cast<float>(screen_height_);

        return std::make_pair(width, height);
    }

    std::pair<float, float> uniform() const
    {
        float aspect = static_cast<float>(config_.sbs_key ? square_width_ * 2 : square_width_) /
                       static_cast<float>(square_height_);
        float width  = std::min(1.0f, static_cast<float>(screen_height_) * aspect / static_cast<float>(screen_width_));
        float height = static_cast<float>(screen_width_ * width) / static_cast<float>(screen_height_ * aspect);

        return std::make_pair(width, height);
    }

    static std::pair<float, float> Fill() { return std::make_pair(1.0f, 1.0f); }

    std::pair<float, float> uniform_to_fill() const
    {
        float wr =
            static_cast<float>(config_.sbs_key ? square_width_ * 2 : square_width_) / static_cast<float>(screen_width_);
        float hr    = static_cast<float>(square_height_) / static_cast<float>(screen_height_);
        float r_inv = 1.0f / std::min(wr, hr);

        float width  = wr * r_inv;
        float height = hr * r_inv;

        return std::make_pair(width, height);
    }
};

struct host_strategy : public display_strategy
{
    virtual ~host_strategy() {}

    virtual frame init_frame(const configuration& config, const core::video_format_desc& format_desc) override
    {
        screen::frame frame;
        auto          flags = GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT | GL_MAP_WRITE_BIT;
        GL(glCreateBuffers(1, &frame.pbo));
        auto size_multiplier = config.high_bitdepth ? 2 : 1;
        GL(glNamedBufferStorage(frame.pbo, format_desc.size * size_multiplier, nullptr, flags));
        frame.ptr = reinterpret_cast<char*>(
            GL2(glMapNamedBufferRange(frame.pbo, 0, format_desc.size * size_multiplier, flags)));

        GL(glCreateTextures(GL_TEXTURE_2D, 1, &frame.tex));
        GL(glTextureParameteri(frame.tex,
                               GL_TEXTURE_MIN_FILTER,
                               (config.colour_space == configuration::colour_spaces::datavideo_full ||
                                config.colour_space == configuration::colour_spaces::datavideo_limited)
                                   ? GL_NEAREST
                                   : GL_LINEAR));
        GL(glTextureParameteri(frame.tex,
                               GL_TEXTURE_MAG_FILTER,
                               (config.colour_space == configuration::colour_spaces::datavideo_full ||
                                config.colour_space == configuration::colour_spaces::datavideo_limited)
                                   ? GL_NEAREST
                                   : GL_LINEAR));
        GL(glTextureParameteri(frame.tex, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
        GL(glTextureParameteri(frame.tex, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
        GL(glTextureStorage2D(
            frame.tex, 1, config.high_bitdepth ? GL_RGBA16 : GL_RGBA8, format_desc.width, format_desc.height));
        GL(glClearTexImage(
            frame.tex, 0, GL_BGRA, config.high_bitdepth ? GL_UNSIGNED_SHORT : GL_UNSIGNED_BYTE, nullptr));

        return frame;
    }

    virtual void cleanup_frame(frame& frame) override
    {
        glUnmapNamedBuffer(frame.pbo);
        glGetError(); // drain any error from unmap
        glDeleteBuffers(1, &frame.pbo);
        glDeleteTextures(1, &frame.tex);
    }

    virtual void do_tick(screen_consumer* self) override
    {
        core::const_frame in_frame;

        while (!self->frame_buffer_.try_pop(in_frame) && self->is_running_) {
            // TODO (fix)
            if (!self->poll()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
        }

        if (!in_frame) {
            return;
        }

        // Upload
        {
            auto& frame = self->frames_.front();

            while (frame.fence != nullptr) {
                auto wait = glClientWaitSync(frame.fence, 0, 0);
                if (wait == GL_ALREADY_SIGNALED || wait == GL_CONDITION_SATISFIED) {
                    glDeleteSync(frame.fence);
                    frame.fence = nullptr;
                }
                if (!self->poll()) {
                    // TODO (fix)
                    std::this_thread::sleep_for(std::chrono::milliseconds(2));
                }
            }

            auto size_multiplier = self->config_.high_bitdepth ? 2 : 1;
            auto& img = in_frame.image_data(0);
            if (img.data() && static_cast<int>(img.size()) >= self->format_desc_.size * size_multiplier) {
                std::memcpy(frame.ptr, img.begin(), self->format_desc_.size * size_multiplier);
            } else {
                // Frame has no CPU pixel data (e.g. VK mixer with readback skipped
                // due to race during consumer add).  Zero-fill to avoid garbled output.
                std::memset(frame.ptr, 0, self->format_desc_.size * size_multiplier);
            }

            GL(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, frame.pbo));
            GL(glTextureSubImage2D(frame.tex,
                                   0,
                                   0,
                                   0,
                                   self->format_desc_.width,
                                   self->format_desc_.height,
                                   GL_BGRA,
                                   self->config_.high_bitdepth ? GL_UNSIGNED_SHORT : GL_UNSIGNED_BYTE,
                                   nullptr));
            GL(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0));

            frame.fence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
        }

        // Display
        {
            auto& frame = self->frames_.back();

            GL(glClear(GL_COLOR_BUFFER_BIT));

            GL(glActiveTexture(GL_TEXTURE0));
            GL(glBindTexture(GL_TEXTURE_2D, frame.tex));

            self->draw();
        }
    }
};

struct gpu_strategy : public display_strategy
{
    virtual ~gpu_strategy()
    {
#ifdef ENABLE_VULKAN
        cleanup_vk_interop();
#endif
    }

    // Lazily-created fallback resources for non-GL textures (e.g. Vulkan mixer).
    GLuint   fallback_pbo_ = 0;
    GLuint   fallback_tex_ = 0;
    char*    fallback_ptr_ = nullptr;
    int      fallback_w_   = 0;
    int      fallback_h_   = 0;
    bool     fallback_hbd_ = false;

#ifdef ENABLE_VULKAN
    // VK→GL interop: import VK texture memory directly into GL (zero-copy).
    GLuint   vk_gl_mem_obj_     = 0;
    GLuint   vk_gl_tex_         = 0;
    HANDLE   vk_cached_handle_  = nullptr;
    int      vk_cached_w_       = 0;
    int      vk_cached_h_       = 0;
    bool     vk_cached_hbd_     = false;
    bool     vk_interop_ok_     = true;   // optimistic; set false on failure
    bool     vk_ext_loaded_     = false;

    // GL semaphore for explicit VK→GL synchronization (avoids relying on
    // implicit driver sync which breaks when another VK consumer accesses
    // the same shared texture — e.g. vk_readback_strategy for DeckLink).
    GLuint   vk_gl_semaphore_         = 0;
    HANDLE   vk_cached_sem_handle_    = nullptr;
    bool     vk_sem_ok_               = false;

    // Dynamic preview downscale: blit full-res VK texture → window-sized local GL texture.
    // This decouples the rendered frame from VK shared memory, reducing GPU contention
    // with production outputs (decklink, vulkan_output).
    GLuint   preview_tex_       = 0;
    GLuint   preview_read_fbo_  = 0;
    GLuint   preview_draw_fbo_  = 0;
    int      preview_w_         = 0;
    int      preview_h_         = 0;
    bool     preview_hbd_       = false;

    PFNGLCREATEMEMORYOBJECTSEXTPROC     glCreateMemoryObjectsEXT_     = nullptr;
    PFNGLDELETEMEMORYOBJECTSEXTPROC     glDeleteMemoryObjectsEXT_     = nullptr;
    PFNGLIMPORTMEMORYWIN32HANDLEEXTPROC glImportMemoryWin32HandleEXT_ = nullptr;
    PFNGLTEXTURESTORAGEMEM2DEXTPROC     glTextureStorageMem2DEXT_     = nullptr;

    // GL_EXT_semaphore_win32 — explicit GPU-side VK→GL sync
    PFNGLGENSEMAPHORESEXTPROC              glGenSemaphoresEXT_              = nullptr;
    PFNGLDELETESEMAPHORESEXTPROC           glDeleteSemaphoresEXT_           = nullptr;
    PFNGLIMPORTSEMAPHOREWIN32HANDLEEXTPROC glImportSemaphoreWin32HandleEXT_ = nullptr;
    PFNGLWAITSEMAPHOREEXTPROC              glWaitSemaphoreEXT_              = nullptr;
    PFNGLSEMAPHOREPARAMETERUI64VEXTPROC    glSemaphoreParameterui64vEXT_    = nullptr;

    void load_vk_gl_extensions()
    {
        if (vk_ext_loaded_)
            return;
        vk_ext_loaded_ = true;

        glCreateMemoryObjectsEXT_     = (PFNGLCREATEMEMORYOBJECTSEXTPROC)wglGetProcAddress("glCreateMemoryObjectsEXT");
        glDeleteMemoryObjectsEXT_     = (PFNGLDELETEMEMORYOBJECTSEXTPROC)wglGetProcAddress("glDeleteMemoryObjectsEXT");
        glImportMemoryWin32HandleEXT_ = (PFNGLIMPORTMEMORYWIN32HANDLEEXTPROC)wglGetProcAddress("glImportMemoryWin32HandleEXT");
        glTextureStorageMem2DEXT_     = (PFNGLTEXTURESTORAGEMEM2DEXTPROC)wglGetProcAddress("glTextureStorageMem2DEXT");

        glGenSemaphoresEXT_              = (PFNGLGENSEMAPHORESEXTPROC)wglGetProcAddress("glGenSemaphoresEXT");
        glDeleteSemaphoresEXT_           = (PFNGLDELETESEMAPHORESEXTPROC)wglGetProcAddress("glDeleteSemaphoresEXT");
        glImportSemaphoreWin32HandleEXT_ = (PFNGLIMPORTSEMAPHOREWIN32HANDLEEXTPROC)wglGetProcAddress("glImportSemaphoreWin32HandleEXT");
        glWaitSemaphoreEXT_              = (PFNGLWAITSEMAPHOREEXTPROC)wglGetProcAddress("glWaitSemaphoreEXT");
        glSemaphoreParameterui64vEXT_    = (PFNGLSEMAPHOREPARAMETERUI64VEXTPROC)wglGetProcAddress("glSemaphoreParameterui64vEXT");

        if (!glCreateMemoryObjectsEXT_ || !glImportMemoryWin32HandleEXT_ || !glTextureStorageMem2DEXT_) {
            vk_interop_ok_ = false;
            CASPAR_LOG(info) << L"[screen] GL_EXT_memory_object_win32 not available — VK interop disabled, using PBO fallback";
        }

        if (glGenSemaphoresEXT_ && glImportSemaphoreWin32HandleEXT_ && glWaitSemaphoreEXT_ &&
            glDeleteSemaphoresEXT_ && glSemaphoreParameterui64vEXT_) {
            vk_sem_ok_ = true;
        } else {
            CASPAR_LOG(info) << L"[screen] GL_EXT_semaphore_win32 not available — using implicit sync (may fail with VK readback modes)";
        }
    }

    void cleanup_vk_interop()
    {
        cleanup_vk_semaphore();
        cleanup_vk_import();
        cleanup_preview();
    }

    void cleanup_vk_semaphore()
    {
        if (vk_gl_semaphore_ && glDeleteSemaphoresEXT_) {
            glDeleteSemaphoresEXT_(1, &vk_gl_semaphore_);
            vk_gl_semaphore_ = 0;
        }
        vk_cached_sem_handle_ = nullptr;
    }

    void cleanup_vk_import()
    {
        if (vk_gl_tex_) {
            glDeleteTextures(1, &vk_gl_tex_);
            vk_gl_tex_ = 0;
        }
        if (vk_gl_mem_obj_ && glDeleteMemoryObjectsEXT_) {
            glDeleteMemoryObjectsEXT_(1, &vk_gl_mem_obj_);
            vk_gl_mem_obj_ = 0;
        }
        vk_cached_handle_ = nullptr;
    }

    void cleanup_preview()
    {
        if (preview_draw_fbo_) {
            glDeleteFramebuffers(1, &preview_draw_fbo_);
            preview_draw_fbo_ = 0;
        }
        if (preview_read_fbo_) {
            glDeleteFramebuffers(1, &preview_read_fbo_);
            preview_read_fbo_ = 0;
        }
        if (preview_tex_) {
            glDeleteTextures(1, &preview_tex_);
            preview_tex_ = 0;
        }
        preview_w_ = 0;
        preview_h_ = 0;
    }

    bool try_vk_interop(const core::const_frame& in_frame, screen_consumer* self)
    {
        load_vk_gl_extensions();
        if (!vk_interop_ok_)
            return false;

        auto tex = in_frame.texture();
        if (!tex)
            return false;

        // Ensure the VK renderpass has finished writing before GL reads.
        // Without this the GPU may still be rendering → garbled output.
        tex->ensure_render_complete();

        HANDLE handle = static_cast<HANDLE>(tex->export_win32_handle());
        if (!handle)
            return false;

        auto w     = tex->tex_width();
        auto h     = tex->tex_height();
        auto hbd   = tex->tex_is_hbd();
        auto alloc = static_cast<GLuint64>(tex->export_alloc_size());

        if (w <= 0 || h <= 0 || alloc == 0)
            return false;

        // Re-import only when the VK handle or dimensions change
        if (handle != vk_cached_handle_ || w != vk_cached_w_ || h != vk_cached_h_ || hbd != vk_cached_hbd_) {
            cleanup_vk_import();

            // Import VK device memory into GL
            glCreateMemoryObjectsEXT_(1, &vk_gl_mem_obj_);

            // Duplicate the handle so GL owns an independent copy.
            // The VK texture owns the original handle and may close it.
            HANDLE dup_handle = nullptr;
            if (!DuplicateHandle(GetCurrentProcess(), handle,
                                 GetCurrentProcess(), &dup_handle,
                                 0, FALSE, DUPLICATE_SAME_ACCESS)) {
                CASPAR_LOG(warning) << L"[screen] DuplicateHandle failed for VK→GL interop — disabling";
                vk_interop_ok_ = false;
                return false;
            }
            glImportMemoryWin32HandleEXT_(vk_gl_mem_obj_, alloc,
                                          GL_HANDLE_TYPE_OPAQUE_WIN32_EXT, dup_handle);

            // Create GL texture backed by the imported memory
            glCreateTextures(GL_TEXTURE_2D, 1, &vk_gl_tex_);
            glTextureParameteri(vk_gl_tex_, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTextureParameteri(vk_gl_tex_, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTextureParameteri(vk_gl_tex_, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTextureParameteri(vk_gl_tex_, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            // The VK mixer stores pixels in BGRA channel order (the fragment
            // shader reads subpassLoad().bgra).  The raw memory is imported as
            // GL_RGBA so R and B are swapped.  Fix with texture swizzle.
            glTextureParameteri(vk_gl_tex_, GL_TEXTURE_SWIZZLE_R, GL_BLUE);
            glTextureParameteri(vk_gl_tex_, GL_TEXTURE_SWIZZLE_B, GL_RED);
            glTextureStorageMem2DEXT_(vk_gl_tex_, 1, hbd ? GL_RGBA16 : GL_RGBA8,
                                      w, h, vk_gl_mem_obj_, 0);

            auto err = glGetError();
            if (err != GL_NO_ERROR) {
                CASPAR_LOG(warning) << L"[screen] GL error during VK interop import (0x"
                                    << std::hex << err << L") — disabling interop";
                cleanup_vk_interop();
                vk_interop_ok_ = false;
                return false;
            }

            vk_cached_handle_ = handle;
            vk_cached_w_      = w;
            vk_cached_h_      = h;
            vk_cached_hbd_    = hbd;

            static bool logged = false;
            if (!logged) {
                CASPAR_LOG(info) << L"[screen] VK→GL zero-copy interop active ("
                                 << w << L"x" << h << (hbd ? L" 16-bit" : L" 8-bit") << L")";
                logged = true;
            }
        }

        // Bind the imported GL texture and draw.
        // Use explicit GL semaphore wait when available — this properly handles
        // the case where another VK consumer (e.g. vk_readback_strategy for
        // DeckLink) is concurrently accessing the same shared texture memory.
        // Without this, NVIDIA's implicit VK→GL sync can miss the concurrent
        // VK access, resulting in black/grey output.
        if (vk_sem_ok_) {
            void*    sem_handle = tex->render_semaphore_handle();
            uint64_t sem_value  = tex->render_semaphore_value();

            if (sem_handle && sem_value > 0) {
                // Re-import the semaphore if the handle changed
                if (sem_handle != vk_cached_sem_handle_) {
                    cleanup_vk_semaphore();

                    glGenSemaphoresEXT_(1, &vk_gl_semaphore_);

                    HANDLE dup_sem = nullptr;
                    DuplicateHandle(GetCurrentProcess(), static_cast<HANDLE>(sem_handle),
                                    GetCurrentProcess(), &dup_sem,
                                    0, FALSE, DUPLICATE_SAME_ACCESS);
                    if (dup_sem) {
                        glImportSemaphoreWin32HandleEXT_(vk_gl_semaphore_,
                                                         GL_HANDLE_TYPE_OPAQUE_WIN32_EXT, dup_sem);
                        vk_cached_sem_handle_ = sem_handle;

                        static bool sem_logged = false;
                        if (!sem_logged) {
                            CASPAR_LOG(info) << L"[screen] Explicit VK→GL semaphore sync active";
                            sem_logged = true;
                        }
                    } else {
                        glDeleteSemaphoresEXT_(1, &vk_gl_semaphore_);
                        vk_gl_semaphore_ = 0;
                        vk_sem_ok_ = false;
                        CASPAR_LOG(warning) << L"[screen] DuplicateHandle for VK semaphore failed — falling back to implicit sync";
                    }
                }

                if (vk_gl_semaphore_) {
                    // GPU-side wait: tells GL not to read the texture until VK
                    // signals the timeline semaphore at sem_value.  Zero CPU cost.
                    GLenum src_layout = GL_LAYOUT_SHADER_READ_ONLY_EXT;
                    glSemaphoreParameterui64vEXT_(vk_gl_semaphore_, GL_D3D12_FENCE_VALUE_EXT, &sem_value);
                    glWaitSemaphoreEXT_(vk_gl_semaphore_, 0, nullptr,
                                        1, &vk_gl_tex_, &src_layout);

                    // Drain any GL error from the semaphore calls — some drivers
                    // report GL_INVALID_VALUE when the timeline value hasn't been
                    // signalled yet or on first use.  The GPU-side sync still works
                    // (the wait is enqueued in the command stream regardless of the
                    // client-side error flag).  Without this drain the stale error
                    // propagates to the next GL() macro and kills the render thread.
                    while (glGetError() != GL_NO_ERROR) {}
                }
            }
        }

        // Dynamic preview downscale: if the VK texture is larger than the window,
        // blit to a window-sized local GL texture.  This:
        //   1. Minimizes the time the VK-shared texture is referenced by GL
        //   2. Decouples SwapBuffers from VK synchronization
        //   3. Reduces fragment shader texture read bandwidth
        auto win_w = static_cast<int>(self->window_.getSize().x);
        auto win_h = static_cast<int>(self->window_.getSize().y);

        // Only downscale if the texture is significantly larger than the window
        // (at least 1.5× in either dimension).  Otherwise render directly.
        bool needs_downscale = (w > win_w * 3 / 2) || (h > win_h * 3 / 2);

        if (needs_downscale && win_w > 0 && win_h > 0) {
            // Ensure preview resources match current window size
            if (preview_w_ != win_w || preview_h_ != win_h || preview_hbd_ != hbd) {
                if (preview_draw_fbo_) glDeleteFramebuffers(1, &preview_draw_fbo_);
                if (preview_read_fbo_) glDeleteFramebuffers(1, &preview_read_fbo_);
                if (preview_tex_) glDeleteTextures(1, &preview_tex_);

                // Create local GL texture at window size (not VK-backed)
                glCreateTextures(GL_TEXTURE_2D, 1, &preview_tex_);
                glTextureParameteri(preview_tex_, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                glTextureParameteri(preview_tex_, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                glTextureParameteri(preview_tex_, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
                glTextureParameteri(preview_tex_, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
                // Swizzle R↔B: the VK texture stores BGRA in RGBA layout.
                // glBlitFramebuffer copies raw pixels (no swizzle applied), so
                // the preview texture inherits the same channel order.
                glTextureParameteri(preview_tex_, GL_TEXTURE_SWIZZLE_R, GL_BLUE);
                glTextureParameteri(preview_tex_, GL_TEXTURE_SWIZZLE_B, GL_RED);
                glTextureStorage2D(preview_tex_, 1, hbd ? GL_RGBA16 : GL_RGBA8, win_w, win_h);

                // Create FBOs for the blit
                glCreateFramebuffers(1, &preview_read_fbo_);
                glNamedFramebufferTexture(preview_read_fbo_, GL_COLOR_ATTACHMENT0, vk_gl_tex_, 0);

                glCreateFramebuffers(1, &preview_draw_fbo_);
                glNamedFramebufferTexture(preview_draw_fbo_, GL_COLOR_ATTACHMENT0, preview_tex_, 0);

                preview_w_   = win_w;
                preview_h_   = win_h;
                preview_hbd_ = hbd;

                CASPAR_LOG(info) << L"[screen] Preview downscale active: " << w << L"x" << h
                                 << L" → " << win_w << L"x" << win_h
                                 << (hbd ? L" 16-bit" : L" 8-bit");
            }

            // Update the read FBO attachment if the VK texture was re-imported
            // (handle changed → vk_gl_tex_ was recreated)
            glNamedFramebufferTexture(preview_read_fbo_, GL_COLOR_ATTACHMENT0, vk_gl_tex_, 0);

            // Blit: hardware-accelerated downscale from full-res VK texture → window-sized preview
            glBlitNamedFramebuffer(preview_read_fbo_, preview_draw_fbo_,
                                   0, 0, w, h,
                                   0, 0, win_w, win_h,
                                   GL_COLOR_BUFFER_BIT, GL_LINEAR);

            // Bind the small local preview texture for rendering
            GL(glActiveTexture(GL_TEXTURE0));
            GL(glBindTexture(GL_TEXTURE_2D, preview_tex_));
        } else {
            // Texture is close to window size — render directly from VK-shared texture
            GL(glActiveTexture(GL_TEXTURE0));
            GL(glBindTexture(GL_TEXTURE_2D, vk_gl_tex_));
        }
        return true;
    }
#endif // ENABLE_VULKAN

    virtual frame init_frame(const configuration& config, const core::video_format_desc& format_desc) override
    {
        return frame();
    }
    virtual void cleanup_frame(frame& frame) override
    {
        if (frame.fence) {
            glDeleteSync(frame.fence);
            frame.fence = nullptr;
        }
        frame.texture.reset();
        if (fallback_pbo_) {
            glUnmapNamedBuffer(fallback_pbo_);
            glDeleteBuffers(1, &fallback_pbo_);
            fallback_pbo_ = 0;
            fallback_ptr_ = nullptr;
        }
        if (fallback_tex_) {
            glDeleteTextures(1, &fallback_tex_);
            fallback_tex_ = 0;
        }
#ifdef ENABLE_VULKAN
        cleanup_vk_interop();
#endif
    }

    virtual void do_tick(screen_consumer* self) override
    {
        core::const_frame in_frame;

        self->poll();

        while (!self->frame_buffer_.try_pop(in_frame) && self->is_running_) {
            // TODO (fix)
            if (!self->poll()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
        }

        // Display
        {
            auto& frame = self->frames_.front();

            while (frame.fence != nullptr && self->is_running_) {
                auto wait = glClientWaitSync(frame.fence, 0, 0);
                if (wait == GL_ALREADY_SIGNALED || wait == GL_CONDITION_SATISFIED) {
                    glDeleteSync(frame.fence);
                    frame.fence = nullptr;
                    frame.texture.reset();
                }

                if (!self->poll()) {
                    // TODO (fix)
                    std::this_thread::sleep_for(std::chrono::milliseconds(2));
                }
            }

            if (!in_frame || !self->is_running_) {
                self->graph_->set_value("tick-time", self->tick_timer_.elapsed() * self->format_desc_.fps * 0.5);
                self->tick_timer_.restart();
                return;
            }

            GL(glClear(GL_COLOR_BUFFER_BIT));

            if (in_frame.texture()) {
                auto ogl_tex = std::dynamic_pointer_cast<accelerator::ogl::texture>(in_frame.texture());
                if (ogl_tex && self->window_.shared_) {
                    // OGL mixer with shared GL contexts: bind directly (zero-copy GPU path)
                    ogl_tex->bind(0);
#ifdef ENABLE_VULKAN
                } else if (try_vk_interop(in_frame, self)) {
                    // VK mixer: zero-copy via GL_EXT_memory_object_win32
                    // (texture already bound by try_vk_interop)
#endif
                } else {
                    // Non-OGL texture (Vulkan mixer) — upload CPU pixels via PBO
                    auto w   = self->format_desc_.width;
                    auto h   = self->format_desc_.height;
                    auto hbd = self->config_.high_bitdepth;
                    auto sz  = static_cast<GLsizeiptr>(self->format_desc_.size) * (hbd ? 2 : 1);

                    if (!fallback_pbo_ || fallback_w_ != w || fallback_h_ != h || fallback_hbd_ != hbd) {
                        if (fallback_pbo_) {
                            glUnmapNamedBuffer(fallback_pbo_);
                            glDeleteBuffers(1, &fallback_pbo_);
                        }
                        if (fallback_tex_)
                            glDeleteTextures(1, &fallback_tex_);

                        auto flags = GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT | GL_MAP_WRITE_BIT;
                        GL(glCreateBuffers(1, &fallback_pbo_));
                        GL(glNamedBufferStorage(fallback_pbo_, sz, nullptr, flags));
                        fallback_ptr_ = reinterpret_cast<char*>(
                            GL2(glMapNamedBufferRange(fallback_pbo_, 0, sz, flags)));

                        GL(glCreateTextures(GL_TEXTURE_2D, 1, &fallback_tex_));
                        GL(glTextureParameteri(fallback_tex_, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
                        GL(glTextureParameteri(fallback_tex_, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
                        GL(glTextureParameteri(fallback_tex_, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
                        GL(glTextureParameteri(fallback_tex_, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
                        GL(glTextureStorage2D(fallback_tex_, 1, hbd ? GL_RGBA16 : GL_RGBA8, w, h));

                        fallback_w_   = w;
                        fallback_h_   = h;
                        fallback_hbd_ = hbd;
                    }

                    auto& img = in_frame.image_data(0);
                    if (img.data() && static_cast<GLsizeiptr>(img.size()) >= sz)
                        std::memcpy(fallback_ptr_, img.data(), sz);

                    GL(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, fallback_pbo_));
                    GL(glTextureSubImage2D(fallback_tex_, 0, 0, 0, w, h,
                                           GL_BGRA, hbd ? GL_UNSIGNED_SHORT : GL_UNSIGNED_BYTE, nullptr));
                    GL(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0));

                    GL(glActiveTexture(GL_TEXTURE0));
                    GL(glBindTexture(GL_TEXTURE_2D, fallback_tex_));
                }

                self->draw();

                frame.fence   = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
                frame.texture = in_frame.texture();
            }
        }
    }
};

struct screen_consumer_proxy : public core::frame_consumer
{
    configuration                    config_;
    std::unique_ptr<screen_consumer> consumer_;
    bool                             use_vulkan_ = false;

  public:
    explicit screen_consumer_proxy(configuration config)
        : config_(std::move(config))
    {
    }

    // frame_consumer

    void initialize(const core::video_format_desc& format_desc,
                    const core::channel_info&      channel_info,
                    int                            port_index) override
    {
        consumer_.reset();
        use_vulkan_ = channel_info.use_vulkan;
        // Set channel transfer so the screen shader knows what EOTF to apply
        switch (channel_info.default_color_transfer) {
            case core::color_transfer::pq:  config_.channel_transfer = 3; break;
            case core::color_transfer::hlg: config_.channel_transfer = 4; break;
            default:                        config_.channel_transfer = 2; break; // sdr/rec709
        }
        consumer_ = std::make_unique<screen_consumer>(config_, format_desc, channel_info.index,
                                                      channel_info.gl_share_context, use_vulkan_);
    }

    std::future<bool> send(core::video_field field, core::const_frame frame) override
    {
        return consumer_->send(field, frame);
    }

    std::wstring print() const override { return consumer_ ? consumer_->print() : L"[screen_consumer]"; }

    std::wstring name() const override { return L"screen"; }

    bool has_synchronization_clock() const override { return false; }

    // When Vulkan mixer is active, gpu_strategy is always used (auto-promoted)
    // and VK→GL zero-copy interop bypasses CPU entirely.
    bool needs_cpu_frame_data() const override { return !use_vulkan_; }

    int index() const override { return 600 + (config_.key_only ? 10 : 0) + config_.screen_index; }

    core::monitor::state state() const override
    {
        core::monitor::state state;
        state["screen/name"]          = config_.name;
        state["screen/index"]         = config_.screen_index;
        state["screen/key_only"]      = config_.key_only;
        state["screen/always_on_top"] = config_.always_on_top;
        return state;
    }
};

spl::shared_ptr<core::frame_consumer> create_consumer(const std::vector<std::wstring>&     params,
                                                      const core::video_format_repository& format_repository,
                                                      const std::vector<spl::shared_ptr<core::video_channel>>& channels,
                                                      const core::channel_info& channel_info)
{
    if (params.empty() || !boost::iequals(params.at(0), L"SCREEN")) {
        return core::frame_consumer::empty();
    }

    configuration config;

    config.high_bitdepth = (channel_info.depth != common::bit_depth::bit8);

    if (params.size() > 1) {
        try {
            config.screen_index = std::stoi(params.at(1));
        } catch (...) {
        }
    }

    config.windowed      = !contains_param(L"FULLSCREEN", params);
    config.gpu_texture   = contains_param(L"GPU", params);
    config.key_only      = contains_param(L"KEY_ONLY", params);
    config.sbs_key       = contains_param(L"SBS_KEY", params);
    config.interactive   = !contains_param(L"NON_INTERACTIVE", params);
    config.borderless    = contains_param(L"BORDERLESS", params);
    config.always_on_top = contains_param(L"ALWAYS_ON_TOP", params);
    config.vsync         = contains_param(L"VSYNC", params);
    config.no_taskbar    = contains_param(L"NO_TASKBAR", params);
    config.closeable     = !contains_param(L"NO_CLOSE", params);
    config.no_activate   = !config.interactive || contains_param(L"NO_ACTIVATE", params);

    // MONITORING: convenience preset — borderless, always-on-top, taskbar-hidden, close-proof, no focus steal, no cursor
    if (contains_param(L"MONITORING", params)) {
        config.interactive   = false;
        config.borderless    = true;
        config.always_on_top = true;
        config.no_taskbar    = true;
        config.closeable     = false;
        config.no_activate   = true;
    }

    if (contains_param(L"NAME", params)) {
        config.name = get_param(L"NAME", params);
    }

    if (contains_param(L"X", params)) {
        config.screen_x = get_param(L"X", params, 0);
    }
    if (contains_param(L"Y", params)) {
        config.screen_y = get_param(L"Y", params, 0);
    }
    if (contains_param(L"WIDTH", params)) {
        config.screen_width = get_param(L"WIDTH", params, 0);
    }
    if (contains_param(L"HEIGHT", params)) {
        config.screen_height = get_param(L"HEIGHT", params, 0);
    }

    if (contains_param(L"TONE_MAP", params)) {
        auto tm = get_param(L"TONE_MAP", params);
        if (boost::iequals(tm, L"reinhard"))
            config.tone_map_op = 1;
        else if (boost::iequals(tm, L"aces_filmic"))
            config.tone_map_op = 2;
        else if (boost::iequals(tm, L"aces_rrt"))
            config.tone_map_op = 3;
        else if (boost::iequals(tm, L"hlg_ootf"))
            config.tone_map_op = 7;
    }
    if (contains_param(L"PEAK_LUMINANCE", params)) {
        config.display_peak_luminance = static_cast<float>(get_param(L"PEAK_LUMINANCE", params, 1000));
    }

    if (config.sbs_key && config.key_only) {
        CASPAR_LOG(warning) << L" Key-only not supported with configuration of side-by-side fill and key. Ignored.";
        config.key_only = false;
    }

    return spl::make_shared<screen_consumer_proxy>(config);
}

spl::shared_ptr<core::frame_consumer>
create_preconfigured_consumer(const boost::property_tree::wptree&                      ptree,
                              const core::video_format_repository&                     format_repository,
                              const std::vector<spl::shared_ptr<core::video_channel>>& channels,
                              const core::channel_info&                                channel_info)
{
    configuration config;

    config.high_bitdepth = (channel_info.depth != common::bit_depth::bit8);

    config.name          = ptree.get(L"name", config.name);
    config.screen_index  = ptree.get(L"device", config.screen_index + 1) - 1;
    config.screen_x      = ptree.get(L"x", config.screen_x);
    config.screen_y      = ptree.get(L"y", config.screen_y);
    config.screen_width  = ptree.get(L"width", config.screen_width);
    config.screen_height = ptree.get(L"height", config.screen_height);
    config.windowed      = ptree.get(L"windowed", config.windowed);
    config.key_only      = ptree.get(L"key-only", config.key_only);
    config.sbs_key       = ptree.get(L"sbs-key", config.sbs_key);
    config.vsync         = ptree.get(L"vsync", config.vsync);
    config.interactive   = ptree.get(L"interactive", config.interactive);
    config.borderless    = ptree.get(L"borderless", config.borderless);
    config.always_on_top = ptree.get(L"always-on-top", config.always_on_top);
    config.gpu_texture   = ptree.get(L"gpu-texture", config.gpu_texture);
    config.no_taskbar    = ptree.get(L"no-taskbar", config.no_taskbar);
    config.closeable     = ptree.get(L"closeable", config.closeable);
    config.no_activate   = ptree.get(L"no-activate", config.no_activate);

    auto colour_space_value = ptree.get(L"colour-space", L"RGB");
    config.colour_space     = configuration::colour_spaces::RGB;
    if (colour_space_value == L"datavideo-full")
        config.colour_space = configuration::colour_spaces::datavideo_full;
    else if (colour_space_value == L"datavideo-limited")
        config.colour_space = configuration::colour_spaces::datavideo_limited;

    if (config.sbs_key && config.key_only) {
        CASPAR_LOG(warning) << L" Key-only not supported with configuration of side-by-side fill and key. Ignored.";
        config.key_only = false;
    }

    if ((config.colour_space == configuration::colour_spaces::datavideo_full ||
         config.colour_space == configuration::colour_spaces::datavideo_limited) &&
        config.sbs_key) {
        CASPAR_LOG(warning) << L" Side-by-side fill and key not supported for DataVideo TC100/TC200. Ignored.";
        config.sbs_key = false;
    }

    if ((config.colour_space == configuration::colour_spaces::datavideo_full ||
         config.colour_space == configuration::colour_spaces::datavideo_limited) &&
        config.key_only) {
        CASPAR_LOG(warning) << L" Key only not supported for DataVideo TC100/TC200. Ignored.";
        config.key_only = false;
    }

    auto tone_map_str = ptree.get(L"auto-tone-map", L"");
    if (!tone_map_str.empty()) {
        if (tone_map_str == L"none")
            config.tone_map_op = 0;
        else if (tone_map_str == L"reinhard")
            config.tone_map_op = 1;
        else if (tone_map_str == L"aces_filmic")
            config.tone_map_op = 2;
        else if (tone_map_str == L"aces_rrt")
            config.tone_map_op = 3;
        else if (tone_map_str == L"hlg_ootf")
            config.tone_map_op = 7;
    }
    config.display_peak_luminance = ptree.get(L"display-peak-luminance", config.display_peak_luminance);

    auto stretch_str = ptree.get(L"stretch", L"fill");
    if (stretch_str == L"none") {
        config.stretch = screen::stretch::none;
    } else if (stretch_str == L"uniform") {
        config.stretch = screen::stretch::uniform;
    } else if (stretch_str == L"uniform_to_fill") {
        config.stretch = screen::stretch::uniform_to_fill;
    }

    auto aspect_str = ptree.get(L"aspect-ratio", L"default");
    if (aspect_str == L"16:9") {
        config.aspect = configuration::aspect_ratio::aspect_16_9;
    } else if (aspect_str == L"4:3") {
        config.aspect = configuration::aspect_ratio::aspect_4_3;
    }

    return spl::make_shared<screen_consumer_proxy>(config);
}

}} // namespace caspar::screen
