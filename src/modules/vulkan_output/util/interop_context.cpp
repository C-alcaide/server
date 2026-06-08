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

#include "interop_context.h"

#include <common/log.h>

#ifdef _WIN32

#include <GL/wglew.h>

namespace caspar { namespace vulkan_output {

interop_context::interop_context()
{
    // We must be on the OGL device thread (its context is current).
    HGLRC parent_rc = wglGetCurrentContext();
    HDC   parent_dc = wglGetCurrentDC();

    if (!parent_rc || !parent_dc) {
        CASPAR_LOG(error) << L"[interop_context] No current GL context — must be called from OGL device thread.";
        return;
    }

    // Create a hidden window + DC for the shared context
    WNDCLASSW wc{};
    wc.lpfnWndProc   = DefWindowProcW;
    wc.hInstance     = GetModuleHandleW(nullptr);
    wc.lpszClassName = L"CasparCG_VK_Interop_Context";
    RegisterClassW(&wc);

    hwnd_ = CreateWindowW(wc.lpszClassName, L"", 0, 0, 0, 1, 1, nullptr, nullptr, wc.hInstance, nullptr);
    if (!hwnd_) {
        CASPAR_LOG(error) << L"[interop_context] Failed to create window.";
        UnregisterClassW(wc.lpszClassName, wc.hInstance);
        return;
    }

    hdc_ = GetDC(hwnd_);

    // Match pixel format with parent — use the same pixel format INDEX, not
    // ChoosePixelFormat (which may return a different index for the new DC,
    // causing shared context creation to fail or produce undefined behavior).
    int pf = GetPixelFormat(parent_dc);
    PIXELFORMATDESCRIPTOR pfd{};
    pfd.nSize = sizeof(pfd);
    DescribePixelFormat(parent_dc, pf, sizeof(pfd), &pfd);
    if (!SetPixelFormat(hdc_, pf, &pfd)) {
        // Fallback: the exact format index may not be valid on this DC.
        // Use ChoosePixelFormat to find the closest match.
        int fallback_pf = ChoosePixelFormat(hdc_, &pfd);
        if (!SetPixelFormat(hdc_, fallback_pf, &pfd)) {
            CASPAR_LOG(error) << L"[interop_context] Failed to set pixel format.";
            ReleaseDC(hwnd_, hdc_);
            DestroyWindow(hwnd_);
            UnregisterClassW(wc.lpszClassName, wc.hInstance);
            hwnd_ = nullptr;
            hdc_  = nullptr;
            return;
        }
    }

    // Create a shared GL context via wglCreateContextAttribsARB (GL 4.5 core)
    auto wglCreateContextAttribsARB = reinterpret_cast<HGLRC(WINAPI*)(HDC, HGLRC, const int*)>(
        wglGetProcAddress("wglCreateContextAttribsARB"));

    if (wglCreateContextAttribsARB) {
        int attribs[] = {
            0x2091, 4, // WGL_CONTEXT_MAJOR_VERSION_ARB
            0x2092, 5, // WGL_CONTEXT_MINOR_VERSION_ARB
            0x9126, 0x00000001, // WGL_CONTEXT_PROFILE_MASK_ARB = CORE
            0
        };
        hglrc_ = wglCreateContextAttribsARB(hdc_, parent_rc, attribs);
    }

    if (!hglrc_) {
        // Fallback: legacy shared context
        hglrc_ = wglCreateContext(hdc_);
        if (hglrc_) {
            wglShareLists(parent_rc, hglrc_);
        }
    }

    if (!hglrc_) {
        CASPAR_LOG(error) << L"[interop_context] Failed to create shared GL context.";
        ReleaseDC(hwnd_, hdc_);
        DestroyWindow(hwnd_);
        hwnd_ = nullptr;
        hdc_  = nullptr;
        return;
    }

    valid_ = true;

    // Start the worker thread
    thread_ = std::thread([this] { thread_func(); });
}

interop_context::~interop_context()
{
    if (thread_.joinable()) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        cv_.notify_one();
        thread_.join();
    }

    if (hglrc_) {
        wglDeleteContext(hglrc_);
    }
    if (hdc_ && hwnd_) {
        ReleaseDC(hwnd_, hdc_);
    }
    if (hwnd_) {
        DestroyWindow(hwnd_);
        UnregisterClassW(L"CasparCG_VK_Interop_Context", GetModuleHandleW(nullptr));
    }
}

void interop_context::dispatch_async(std::function<void()> task)
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        tasks_.push(std::move(task));
    }
    cv_.notify_one();
}

void interop_context::dispatch_sync(std::function<void()> task)
{
    bool done = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        tasks_.push([&] {
            task();
            {
                std::lock_guard<std::mutex> lk(mutex_);
                done = true;
            }
            done_cv_.notify_one();
        });
    }
    cv_.notify_one();

    std::unique_lock<std::mutex> lock(mutex_);
    done_cv_.wait(lock, [&] { return done; });
}

void interop_context::thread_func()
{
    SetThreadDescription(GetCurrentThread(), L"VK Interop GL");

    if (!wglMakeCurrent(hdc_, hglrc_)) {
        CASPAR_LOG(error) << L"[interop_context] Failed to make shared GL context current.";
        valid_ = false;
        return;
    }

    // Init GLEW on this context
    glewExperimental = GL_TRUE;
    glewInit();

    while (true) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this] { return !tasks_.empty() || stop_; });
            if (stop_ && tasks_.empty())
                break;
            task = std::move(tasks_.front());
            tasks_.pop();
        }
        task();
    }

    wglMakeCurrent(nullptr, nullptr);
}

}} // namespace caspar::vulkan_output

#else // Linux

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <pthread.h>

namespace caspar { namespace vulkan_output {

interop_context::interop_context()
{
    // On Linux, the OGL device uses EGL. We need to get the current EGL display
    // and context to create a shared child context.
    auto parent_display = eglGetCurrentDisplay();
    auto parent_context = eglGetCurrentContext();

    if (parent_display == EGL_NO_DISPLAY || parent_context == EGL_NO_CONTEXT) {
        CASPAR_LOG(error) << L"[interop_context] No current EGL context — must be called from OGL device thread.";
        return;
    }

    egl_display_ = parent_display;

    // Get the config of the parent context
    EGLint config_id = 0;
    eglQueryContext(parent_display, parent_context, EGL_CONFIG_ID, &config_id);

    EGLConfig config = nullptr;
    EGLint num_configs = 0;
    EGLint config_attribs[] = {EGL_CONFIG_ID, config_id, EGL_NONE};
    eglChooseConfig(parent_display, config_attribs, &config, 1, &num_configs);

    if (num_configs == 0) {
        CASPAR_LOG(error) << L"[interop_context] Failed to retrieve parent EGL config.";
        return;
    }

    // Create shared EGL context (GL 4.5 core)
    EGLint ctx_attribs[] = {
        EGL_CONTEXT_MAJOR_VERSION, 4,
        EGL_CONTEXT_MINOR_VERSION, 5,
        EGL_CONTEXT_OPENGL_PROFILE_MASK, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,
        EGL_NONE
    };

    egl_context_ = eglCreateContext(parent_display, config, parent_context, ctx_attribs);
    if (!egl_context_ || egl_context_ == EGL_NO_CONTEXT) {
        CASPAR_LOG(error) << L"[interop_context] Failed to create shared EGL context.";
        egl_context_ = nullptr;
        return;
    }

    egl_surface_ = nullptr; // surfaceless rendering

    valid_ = true;

    // Start the worker thread
    thread_ = std::thread([this] { thread_func(); });
}

interop_context::~interop_context()
{
    if (thread_.joinable()) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        cv_.notify_one();
        thread_.join();
    }

    if (egl_context_) {
        eglDestroyContext(static_cast<EGLDisplay>(egl_display_),
                          static_cast<EGLContext>(egl_context_));
    }
}

void interop_context::dispatch_async(std::function<void()> task)
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        tasks_.push(std::move(task));
    }
    cv_.notify_one();
}

void interop_context::dispatch_sync(std::function<void()> task)
{
    bool done = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        tasks_.push([&] {
            task();
            {
                std::lock_guard<std::mutex> lk(mutex_);
                done = true;
            }
            done_cv_.notify_one();
        });
    }
    cv_.notify_one();

    std::unique_lock<std::mutex> lock(mutex_);
    done_cv_.wait(lock, [&] { return done; });
}

void interop_context::thread_func()
{
    pthread_setname_np(pthread_self(), "VK Interop GL");

    // EGL API binding is per-thread state — must bind OpenGL API on this thread
    eglBindAPI(EGL_OPENGL_API);

    // Make shared context current (surfaceless — EGL_NO_SURFACE)
    if (!eglMakeCurrent(static_cast<EGLDisplay>(egl_display_),
                        EGL_NO_SURFACE, EGL_NO_SURFACE,
                        static_cast<EGLContext>(egl_context_))) {
        CASPAR_LOG(error) << L"[interop_context] Failed to make shared EGL context current.";
        valid_ = false;
        return;
    }

    // Init GLEW on this context
    glewExperimental = GL_TRUE;
    glewInit();

    while (true) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this] { return !tasks_.empty() || stop_; });
            if (stop_ && tasks_.empty())
                break;
            task = std::move(tasks_.front());
            tasks_.pop();
        }
        task();
    }

    eglMakeCurrent(static_cast<EGLDisplay>(egl_display_),
                   EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
}

}} // namespace caspar::vulkan_output

#endif // _WIN32
