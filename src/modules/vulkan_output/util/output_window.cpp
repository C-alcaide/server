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
#endif

#include <atomic>

namespace caspar { namespace vulkan_output {

#ifdef _WIN32

static const wchar_t* const kWindowClassName = L"CasparCG_VulkanOutput";

static LRESULT CALLBACK wnd_proc(HWND hwnd, UINT msg, WPARAM wparam, LPARAM lparam)
{
    if (msg == WM_CLOSE)
        return 0; // Prevent user closing the output window
    if (msg == WM_ERASEBKGND)
        return 1; // Prevent flicker
    return DefWindowProcW(hwnd, msg, wparam, lparam);
}

static void register_window_class()
{
    static std::once_flag flag;
    std::call_once(flag, [] {
        WNDCLASSEXW wc{};
        wc.cbSize        = sizeof(wc);
        wc.style         = CS_OWNDC;
        wc.lpfnWndProc   = wnd_proc;
        wc.hInstance     = GetModuleHandleW(nullptr);
        wc.hCursor       = LoadCursorW(nullptr, IDC_ARROW);
        wc.hbrBackground = static_cast<HBRUSH>(GetStockObject(BLACK_BRUSH));
        wc.lpszClassName = kWindowClassName;
        RegisterClassExW(&wc);
    });
}

struct output_window::impl
{
    HWND             hwnd_   = nullptr;
    int              width_  = 0;
    int              height_ = 0;
    std::atomic<bool> closed_{false};

    impl(const display_info& display)
    {
        register_window_class();

        width_  = display.width;
        height_ = display.height;

        // WS_POPUP = no title bar, no border. Position exactly on target display.
        hwnd_ = CreateWindowExW(WS_EX_APPWINDOW | WS_EX_TOPMOST,
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

        if (!hwnd_)
            CASPAR_THROW_EXCEPTION(caspar_exception() << msg_info("Failed to create output window"));

        // Hide cursor on the output window
        ShowCursor(FALSE);

        CASPAR_LOG(info) << L"[vulkan_output] Created window on " << display.display_name << L" at ("
                         << display.pos_x << L"," << display.pos_y << L") " << width_ << L"x" << height_;
    }

    ~impl()
    {
        if (hwnd_) {
            DestroyWindow(hwnd_);
            hwnd_ = nullptr;
        }
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

#else

// Linux/macOS stub — to be implemented with XCB/Wayland or MoltenVK
struct output_window::impl
{
    int width_  = 0;
    int height_ = 0;

    impl(const display_info& display)
        : width_(display.width)
        , height_(display.height)
    {
        CASPAR_LOG(warning) << L"[vulkan_output] Platform window not implemented; output will not display.";
    }

    ~impl() = default;

    vk::SurfaceKHR create_surface(vk::Instance /*instance*/)
    {
        CASPAR_THROW_EXCEPTION(caspar_exception() << msg_info("Vulkan output window not implemented on this platform"));
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

}} // namespace caspar::vulkan_output
