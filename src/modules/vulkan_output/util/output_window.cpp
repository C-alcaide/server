/*
 * Copyright (c) 2026 CasparCG Contributors
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
 */

#include "output_window.h"

#include <common/except.h>
#include <common/log.h>

#ifdef _WIN32
#include <vulkan/vulkan_win32.h>
#include <windows.h>
#include <dwmapi.h>
#pragma comment(lib, "dwmapi.lib")
#endif

#include <atomic>
#include <thread>

namespace caspar { namespace vulkan_output {

#ifdef _WIN32

static const wchar_t* const kWindowClassName = L"CasparCG_VulkanOutput";

static LRESULT CALLBACK wnd_proc(HWND hwnd, UINT msg, WPARAM wparam, LPARAM lparam)
{
    switch (msg) {
        case WM_CLOSE:
            return 0; // Block close

        case WM_SYSCOMMAND:
            // Block minimize, restore, close via system menu
            switch (wparam & 0xFFF0) {
                case SC_CLOSE:
                case SC_MINIMIZE:
                case SC_RESTORE:
                    return 0;
            }
            break;

        case WM_MOUSEACTIVATE:
            return MA_NOACTIVATEANDEAT; // Don't steal focus on click

        case WM_SETCURSOR:
            SetCursor(nullptr); // Hide cursor (per-message, reliable)
            return TRUE;

        case WM_WINDOWPOSCHANGING: {
            // Prevent move, resize, z-order change, hide
            auto* pos = reinterpret_cast<WINDOWPOS*>(lparam);
            pos->flags |= SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER;
            pos->flags &= ~SWP_HIDEWINDOW;
            pos->hwndInsertAfter = HWND_TOPMOST;
            return 0;
        }

        case WM_ACTIVATE:
            // Re-assert TOPMOST when focus returns
            SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0,
                         SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE);
            break;

        case WM_SIZE:
            if (wparam == SIZE_MINIMIZED)
                return 0; // Block minimize
            break;

        case WM_ERASEBKGND: {
            HDC hdc = reinterpret_cast<HDC>(wparam);
            RECT rc;
            GetClientRect(hwnd, &rc);
            FillRect(hdc, &rc, static_cast<HBRUSH>(GetStockObject(BLACK_BRUSH)));
            return 1;
        }

        case WM_PAINT: {
            PAINTSTRUCT ps;
            HDC hdc = BeginPaint(hwnd, &ps);
            FillRect(hdc, &ps.rcPaint, static_cast<HBRUSH>(GetStockObject(BLACK_BRUSH)));
            EndPaint(hwnd, &ps);
            return 0;
        }
    }
    return DefWindowProcW(hwnd, msg, wparam, lparam);
}

static void register_window_class()
{
    static std::once_flag flag;
    std::call_once(flag, [] {
        WNDCLASSEXW wc{};
        wc.cbSize        = sizeof(wc);
        wc.style         = CS_HREDRAW | CS_VREDRAW;
        wc.lpfnWndProc   = wnd_proc;
        wc.hInstance     = GetModuleHandleW(nullptr);
        wc.hCursor       = nullptr; // No cursor
        wc.hbrBackground = static_cast<HBRUSH>(GetStockObject(BLACK_BRUSH));
        wc.lpszClassName = kWindowClassName;
        RegisterClassExW(&wc);
    });
}

struct output_window::impl
{
    HWND              hwnd_   = nullptr;
    int               width_  = 0;
    int               height_ = 0;
    std::atomic<bool> closed_{false};
    std::thread       msg_thread_;

    impl(const display_info& display)
        : width_(display.width)
        , height_(display.height)
    {
        register_window_class();

        // Create window on a dedicated message pump thread for responsiveness.
        std::atomic<bool> ready{false};
        msg_thread_ = std::thread([this, &display, &ready] {
            // Per-monitor DPI awareness (thread-level, loaded dynamically)
            using SetDpiCtxFn = DPI_AWARENESS_CONTEXT(WINAPI*)(DPI_AWARENESS_CONTEXT);
            auto set_dpi = reinterpret_cast<SetDpiCtxFn>(
                GetProcAddress(GetModuleHandleW(L"user32.dll"), "SetThreadDpiAwarenessContext"));
            DPI_AWARENESS_CONTEXT prev_ctx = nullptr;
            if (set_dpi)
                prev_ctx = set_dpi(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);

            // WS_EX_TOOLWINDOW = hidden from taskbar and Alt+Tab
            hwnd_ = CreateWindowExW(WS_EX_TOOLWINDOW | WS_EX_TOPMOST,
                                    kWindowClassName,
                                    L"CasparCG Vulkan Output",
                                    WS_POPUP | WS_VISIBLE,
                                    display.pos_x,
                                    display.pos_y,
                                    width_,
                                    height_,
                                    nullptr,
                                    nullptr,
                                    GetModuleHandleW(nullptr),
                                    nullptr);

            if (!hwnd_) {
                ready = true;
                return;
            }

            // DWM protection: prevent Aero Peek / Show Desktop from hiding output
            BOOL exclude = TRUE;
            DwmSetWindowAttribute(hwnd_, DWMWA_EXCLUDED_FROM_PEEK, &exclude, sizeof(exclude));
            DwmSetWindowAttribute(hwnd_, DWMWA_DISALLOW_PEEK, &exclude, sizeof(exclude));

            // Bring to foreground (attach to foreground thread for permission)
            DWORD fg_thread = GetWindowThreadProcessId(GetForegroundWindow(), nullptr);
            DWORD our_thread = GetCurrentThreadId();
            if (fg_thread != our_thread)
                AttachThreadInput(our_thread, fg_thread, TRUE);
            SetForegroundWindow(hwnd_);
            BringWindowToTop(hwnd_);
            if (fg_thread != our_thread)
                AttachThreadInput(our_thread, fg_thread, FALSE);

            ready = true;

            // Message pump — keeps window responsive
            MSG msg;
            while (!closed_) {
                DWORD result = MsgWaitForMultipleObjects(0, nullptr, FALSE, 100, QS_ALLINPUT);
                if (result == WAIT_OBJECT_0) {
                    while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
                        if (msg.message == WM_QUIT) {
                            closed_ = true;
                            break;
                        }
                        TranslateMessage(&msg);
                        DispatchMessageW(&msg);
                    }
                }
            }

            // Destroy window on the thread that created it (Win32 requirement)
            if (hwnd_) {
                DestroyWindow(hwnd_);
                hwnd_ = nullptr;
            }

            if (set_dpi && prev_ctx)
                set_dpi(prev_ctx);
        });

        // Wait for window creation (timeout after 5 seconds)
        int wait_ms = 0;
        while (!ready && wait_ms < 5000) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            ++wait_ms;
        }

        if (!hwnd_)
            CASPAR_THROW_EXCEPTION(caspar_exception() << msg_info("Failed to create output window"));

        CASPAR_LOG(info) << L"[vulkan_output] Created FSE window on " << display.display_name << L" at ("
                         << display.pos_x << L"," << display.pos_y << L") " << width_ << L"x" << height_;
    }

    ~impl()
    {
        closed_ = true;
        if (hwnd_) {
            PostMessage(hwnd_, WM_QUIT, 0, 0);
        }
        if (msg_thread_.joinable())
            msg_thread_.join();
    }

    vk::SurfaceKHR create_surface(vk::Instance instance)
    {
        VkWin32SurfaceCreateInfoKHR create_info{};
        create_info.sType     = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
        create_info.hinstance = GetModuleHandleW(nullptr);
        create_info.hwnd      = hwnd_;

        VkSurfaceKHR surface = VK_NULL_HANDLE;
        VkResult     result  = vkCreateWin32SurfaceKHR(instance, &create_info, nullptr, &surface);
        if (result != VK_SUCCESS)
            CASPAR_THROW_EXCEPTION(caspar_exception() << msg_info("Failed to create Vulkan Win32 surface"));

        return vk::SurfaceKHR(surface);
    }
};

#else // Linux: VK_KHR_display — no platform window needed

struct output_window::impl
{
    int              width_  = 0;
    int              height_ = 0;
    VkDisplayKHR     display_ = VK_NULL_HANDLE;

    impl(const display_info& display)
        : width_(display.width)
        , height_(display.height)
        , display_(display.vk_display)
    {
        if (display_ == VK_NULL_HANDLE) {
            CASPAR_THROW_EXCEPTION(caspar_exception()
                << msg_info("No VK_KHR_display handle for output. "
                            "Ensure display is not managed by X11/Wayland compositor."));
        }

        CASPAR_LOG(info) << L"[vulkan_output] Linux VK_KHR_display output: "
                         << display.display_name << L" " << width_ << L"x" << height_;
    }

    ~impl() = default;

    vk::SurfaceKHR create_surface(vk::Instance instance)
    {
        // Surface creation for VK_KHR_display is handled by output_device::create_display_surface().
        // This method should not be called on Linux — the consumer uses the display surface path directly.
        CASPAR_THROW_EXCEPTION(caspar_exception()
            << msg_info("output_window::create_surface() should not be called on Linux. "
                        "Use output_device::create_display_surface() directly."));
    }
};

#endif

output_window::output_window(const display_info& display)
    : impl_(std::make_unique<impl>(display))
{
}

output_window::~output_window() = default;

int output_window::width() const { return impl_->width_; }
int output_window::height() const { return impl_->height_; }

vk::SurfaceKHR output_window::create_surface(vk::Instance instance)
{
    return impl_->create_surface(instance);
}

bool output_window::should_close() const
{
#ifdef _WIN32
    return impl_->closed_.load();
#else
    return false;
#endif
}

#ifdef _WIN32
void* output_window::native_handle() const { return impl_->hwnd_; }
#endif

}} // namespace caspar::vulkan_output
