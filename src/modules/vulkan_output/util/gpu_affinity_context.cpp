/*
 * Copyright (c) 2024 CasparCG contributors
 *
 * CasparCG is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#include "gpu_affinity_context.h"

#include <common/log.h>
#include <common/except.h>

#include <GL/glew.h>
#include <GL/wglew.h>

#include <cstring>

namespace caspar { namespace vulkan_output {

// ─── WGL_NV_gpu_affinity function pointers ──────────────────────────────────

typedef BOOL(WINAPI* PFNWGLENUMGPUSNVPROC)(UINT iGpuIndex, HGPUNV* phGpu);
typedef BOOL(WINAPI* PFNWGLENUMGPUDEVICESNVPROC)(HGPUNV hGpu, UINT iDeviceIndex, PGPU_DEVICE lpGpuDevice);
typedef HDC(WINAPI* PFNWGLCREATEAFFINITYDCNVPROC)(const HGPUNV* phGpuList);
typedef BOOL(WINAPI* PFNWGLDELETEDCNVPROC)(HDC hdc);

static PFNWGLENUMGPUSNVPROC          wglEnumGpusNV_          = nullptr;
static PFNWGLENUMGPUDEVICESNVPROC    wglEnumGpuDevicesNV_    = nullptr;
static PFNWGLCREATEAFFINITYDCNVPROC  wglCreateAffinityDCNV_  = nullptr;
static PFNWGLDELETEDCNVPROC          wglDeleteDCNV_          = nullptr;

static bool load_affinity_extensions()
{
    static bool loaded = false;
    static bool available = false;
    if (loaded) return available;
    loaded = true;

    wglEnumGpusNV_         = reinterpret_cast<PFNWGLENUMGPUSNVPROC>(wglGetProcAddress("wglEnumGpusNV"));
    wglEnumGpuDevicesNV_   = reinterpret_cast<PFNWGLENUMGPUDEVICESNVPROC>(wglGetProcAddress("wglEnumGpuDevicesNV"));
    wglCreateAffinityDCNV_ = reinterpret_cast<PFNWGLCREATEAFFINITYDCNVPROC>(wglGetProcAddress("wglCreateAffinityDCNV"));
    wglDeleteDCNV_         = reinterpret_cast<PFNWGLDELETEDCNVPROC>(wglGetProcAddress("wglDeleteDCNV"));

    available = (wglEnumGpusNV_ && wglEnumGpuDevicesNV_ && wglCreateAffinityDCNV_ && wglDeleteDCNV_);
    if (!available) {
        CASPAR_LOG(warning) << L"[gpu_affinity] WGL_NV_gpu_affinity not available — multi-GPU interop disabled";
    }
    return available;
}

// ─── Construction / destruction ─────────────────────────────────────────────

gpu_affinity_context::gpu_affinity_context(int gpu_index, int width, int height)
    : gpu_index_(gpu_index)
    , width_(width)
    , height_(height)
{
    // Start the thread — it will create the affinity context and signal when ready
    std::promise<bool> init_promise;
    auto init_future = init_promise.get_future();

    running_ = true;
    thread_ = std::thread([this, &init_promise] {
        try {
            thread_func();
            // If thread_func returns normally during init, it means success was already signaled
        } catch (...) {
            // If we haven't signaled yet, signal failure
            try { init_promise.set_exception(std::current_exception()); } catch (...) {}
        }
    });

    // Wait for initialization on the affinity thread
    // We'll signal from inside thread_func after context creation
    // Actually, let's restructure: create the context synchronously via dispatch
    // Wait for the thread to start and create its context
    dispatch([this, &init_promise] {
        // This runs on the affinity thread
        if (!create_affinity_context(gpu_index_)) {
            init_promise.set_exception(std::make_exception_ptr(
                caspar_exception() << msg_info("Failed to create GPU affinity context for GPU " + std::to_string(gpu_index_))));
            return;
        }

        // Initialize GLEW on this context
        glewExperimental = GL_TRUE;
        auto glew_result = glewInit();
        if (glew_result != GLEW_OK) {
            init_promise.set_exception(std::make_exception_ptr(
                caspar_exception() << msg_info("GLEW init failed on affinity context")));
            return;
        }

        // Query LUID
        auto glGetUnsignedBytevEXT = reinterpret_cast<void(APIENTRY*)(GLenum, GLubyte*)>(
            wglGetProcAddress("glGetUnsignedBytevEXT"));
        if (glGetUnsignedBytevEXT) {
            glGetUnsignedBytevEXT(0x9462 /*GL_DEVICE_LUID_EXT*/, device_luid_);
            device_luid_valid_ = true;
        }

        // Create upload texture
        glGenTextures(1, &upload_texture_);
        glBindTexture(GL_TEXTURE_2D, upload_texture_);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width_, height_, 0, GL_BGRA, GL_UNSIGNED_BYTE, nullptr);
        glBindTexture(GL_TEXTURE_2D, 0);

        // Create double-buffered PBOs for async upload
        glGenBuffers(2, pbo_);
        for (int i = 0; i < 2; ++i) {
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_[i]);
            glBufferData(GL_PIXEL_UNPACK_BUFFER, static_cast<GLsizeiptr>(width_) * height_ * 4,
                         nullptr, GL_STREAM_DRAW);
        }
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        CASPAR_LOG(info) << L"[gpu_affinity] Created OGL context on GPU " << gpu_index_
                         << L" (" << width_ << L"x" << height_ << L")";
        init_promise.set_value(true);
    });

    init_future.get(); // Throws if init failed
}

gpu_affinity_context::~gpu_affinity_context()
{
    if (running_) {
        // Clean up GL resources on the context thread
        dispatch([this] {
            if (upload_texture_) {
                glDeleteTextures(1, &upload_texture_);
                upload_texture_ = 0;
            }
            if (pbo_[0]) {
                glDeleteBuffers(2, pbo_);
                pbo_[0] = pbo_[1] = 0;
            }

            // Unbind and destroy context
            wglMakeCurrent(nullptr, nullptr);
            if (affinity_rc_) {
                wglDeleteContext(affinity_rc_);
                affinity_rc_ = nullptr;
            }
            if (affinity_dc_ && wglDeleteDCNV_) {
                wglDeleteDCNV_(affinity_dc_);
                affinity_dc_ = nullptr;
            }
        });

        running_ = false;
        queue_cv_.notify_all();
        if (thread_.joinable())
            thread_.join();
    }
}

// ─── Thread management ──────────────────────────────────────────────────────

void gpu_affinity_context::dispatch(std::function<void()> func)
{
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        work_queue_.push(std::move(func));
    }
    queue_cv_.notify_one();
}

void gpu_affinity_context::thread_func()
{
    SetThreadDescription(GetCurrentThread(), L"GPU Affinity OGL");

    while (running_) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait(lock, [this] { return !work_queue_.empty() || !running_; });
            if (!running_ && work_queue_.empty())
                break;
            task = std::move(work_queue_.front());
            work_queue_.pop();
        }
        if (task)
            task();
    }
}

// ─── Affinity context creation ──────────────────────────────────────────────

bool gpu_affinity_context::create_affinity_context(int gpu_index)
{
    // We need a temporary context to load WGL extensions first.
    // Use the calling thread's current context if available, or create one.
    // Since this runs on our own thread, we need a bootstrap.

    // Step 1: Create a temporary hidden window + basic OGL context to load WGL extensions
    WNDCLASSEXW wc{};
    wc.cbSize        = sizeof(WNDCLASSEXW);
    wc.lpfnWndProc   = DefWindowProcW;
    wc.hInstance     = GetModuleHandle(nullptr);
    wc.lpszClassName = L"CasparGPUAffinityBootstrap";
    RegisterClassExW(&wc);

    HWND temp_hwnd = CreateWindowExW(0, wc.lpszClassName, L"", 0, 0, 0, 1, 1, nullptr, nullptr, wc.hInstance, nullptr);
    HDC  temp_dc   = GetDC(temp_hwnd);

    PIXELFORMATDESCRIPTOR pfd{};
    pfd.nSize      = sizeof(PIXELFORMATDESCRIPTOR);
    pfd.nVersion   = 1;
    pfd.dwFlags    = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
    pfd.iPixelType = PFD_TYPE_RGBA;
    pfd.cColorBits = 32;

    int pf = ChoosePixelFormat(temp_dc, &pfd);
    SetPixelFormat(temp_dc, pf, &pfd);

    HGLRC temp_rc = wglCreateContext(temp_dc);
    wglMakeCurrent(temp_dc, temp_rc);

    // Load WGL extensions
    if (!load_affinity_extensions()) {
        wglMakeCurrent(nullptr, nullptr);
        wglDeleteContext(temp_rc);
        ReleaseDC(temp_hwnd, temp_dc);
        DestroyWindow(temp_hwnd);
        return false;
    }

    // Step 2: Enumerate GPUs
    HGPUNV gpu_handle = nullptr;
    if (!wglEnumGpusNV_(static_cast<UINT>(gpu_index), &gpu_handle) || !gpu_handle) {
        CASPAR_LOG(error) << L"[gpu_affinity] GPU index " << gpu_index << L" not found via WGL_NV_gpu_affinity";
        wglMakeCurrent(nullptr, nullptr);
        wglDeleteContext(temp_rc);
        ReleaseDC(temp_hwnd, temp_dc);
        DestroyWindow(temp_hwnd);
        return false;
    }

    // Log GPU device info
    GPU_DEVICE gpu_device{};
    gpu_device.cb = sizeof(GPU_DEVICE);
    if (wglEnumGpuDevicesNV_(gpu_handle, 0, &gpu_device)) {
        CASPAR_LOG(info) << L"[gpu_affinity] GPU " << gpu_index << L": "
                         << gpu_device.DeviceString;
    }

    // Step 3: Create affinity DC for the target GPU
    HGPUNV gpu_list[2] = {gpu_handle, nullptr};
    affinity_dc_ = wglCreateAffinityDCNV_(gpu_list);
    if (!affinity_dc_) {
        CASPAR_LOG(error) << L"[gpu_affinity] Failed to create affinity DC for GPU " << gpu_index;
        wglMakeCurrent(nullptr, nullptr);
        wglDeleteContext(temp_rc);
        ReleaseDC(temp_hwnd, temp_dc);
        DestroyWindow(temp_hwnd);
        return false;
    }

    // Set pixel format on the affinity DC
    pf = ChoosePixelFormat(affinity_dc_, &pfd);
    SetPixelFormat(affinity_dc_, pf, &pfd);

    // Step 4: Create the real OGL context on the affinity DC
    // Try to get an OpenGL 4.5 core context via WGL_ARB_create_context
    auto wglCreateContextAttribsARB = reinterpret_cast<HGLRC(WINAPI*)(HDC, HGLRC, const int*)>(
        wglGetProcAddress("wglCreateContextAttribsARB"));

    if (wglCreateContextAttribsARB) {
        int attribs[] = {
            0x2091, 4, // WGL_CONTEXT_MAJOR_VERSION_ARB
            0x2092, 5, // WGL_CONTEXT_MINOR_VERSION_ARB
            0x9126, 0x00000001, // WGL_CONTEXT_PROFILE_MASK_ARB = CORE
            0 // terminator
        };
        affinity_rc_ = wglCreateContextAttribsARB(affinity_dc_, nullptr, attribs);
    }

    if (!affinity_rc_) {
        // Fallback to legacy context
        affinity_rc_ = wglCreateContext(affinity_dc_);
    }

    // Destroy the temporary context
    wglMakeCurrent(nullptr, nullptr);
    wglDeleteContext(temp_rc);
    ReleaseDC(temp_hwnd, temp_dc);
    DestroyWindow(temp_hwnd);
    UnregisterClassW(wc.lpszClassName, wc.hInstance);

    if (!affinity_rc_) {
        CASPAR_LOG(error) << L"[gpu_affinity] Failed to create OpenGL context on affinity DC";
        if (wglDeleteDCNV_) wglDeleteDCNV_(affinity_dc_);
        affinity_dc_ = nullptr;
        return false;
    }

    // Make the affinity context current on this thread
    if (!wglMakeCurrent(affinity_dc_, affinity_rc_)) {
        CASPAR_LOG(error) << L"[gpu_affinity] Failed to make affinity context current";
        wglDeleteContext(affinity_rc_);
        affinity_rc_ = nullptr;
        if (wglDeleteDCNV_) wglDeleteDCNV_(affinity_dc_);
        affinity_dc_ = nullptr;
        return false;
    }

    return true;
}

// ─── Frame upload ───────────────────────────────────────────────────────────

GLuint gpu_affinity_context::upload_frame(const uint8_t* pixels, int width, int height, int stride)
{
    if (!pixels || width <= 0 || height <= 0)
        return upload_texture_;

    // Resize texture if needed
    if (width != width_ || height != height_) {
        width_  = width;
        height_ = height;
        glBindTexture(GL_TEXTURE_2D, upload_texture_);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width_, height_, 0, GL_BGRA, GL_UNSIGNED_BYTE, nullptr);
        glBindTexture(GL_TEXTURE_2D, 0);

        for (int i = 0; i < 2; ++i) {
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_[i]);
            glBufferData(GL_PIXEL_UNPACK_BUFFER, static_cast<GLsizeiptr>(width_) * height_ * 4,
                         nullptr, GL_STREAM_DRAW);
        }
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    }

    // Double-buffered PBO upload:
    // Frame N: copy pixels into PBO[current], DMA PBO[previous] → texture
    int current_pbo  = pbo_index_;
    int previous_pbo = 1 - pbo_index_;
    pbo_index_ = previous_pbo; // Swap for next frame

    // Upload previous frame's PBO → texture (async DMA, should be done by now)
    glBindTexture(GL_TEXTURE_2D, upload_texture_);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_[previous_pbo]);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width_, height_, GL_BGRA, GL_UNSIGNED_BYTE, nullptr);

    // Map current PBO and copy new pixel data into it
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_[current_pbo]);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, static_cast<GLsizeiptr>(width_) * height_ * 4,
                 nullptr, GL_STREAM_DRAW); // Orphan for async
    void* mapped = glMapBuffer(GL_PIXEL_UNPACK_BUFFER, GL_WRITE_ONLY);
    if (mapped) {
        // Handle stride != width*4
        if (stride == width * 4) {
            memcpy(mapped, pixels, static_cast<size_t>(width) * height * 4);
        } else {
            for (int row = 0; row < height; ++row) {
                memcpy(static_cast<uint8_t*>(mapped) + row * width * 4,
                       pixels + row * stride,
                       static_cast<size_t>(width) * 4);
            }
        }
        glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER);
    }

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    glBindTexture(GL_TEXTURE_2D, 0);

    return upload_texture_;
}

}} // namespace caspar::vulkan_output
