/*
 * Copyright (c) 2024 CasparCG contributors
 *
 * CasparCG is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#pragma once

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>
#include <GL/glew.h>

#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>

namespace caspar { namespace vulkan_output {

/**
 * Creates an OpenGL context bound to a specific GPU via WGL_NV_gpu_affinity.
 *
 * This enables zero-copy OGL→VK interop on multi-GPU systems where the main
 * CasparCG OGL mixer runs on GPU 0 but outputs are connected to GPU 1+.
 *
 * The context runs on its own dedicated thread (like the main ogl::device)
 * and provides dispatch_sync/dispatch_async for serialized GL work.
 *
 * Usage:
 *   1. Create with the target gpu_index
 *   2. Upload frame pixels via upload_frame()
 *   3. shared_texture_pool uses blit_from_texture with the uploaded texture
 *   4. Signal and interop to VK as normal
 */
class gpu_affinity_context
{
  public:
    /// Creates an affinity OGL context on the specified GPU.
    /// Throws if WGL_NV_gpu_affinity is unavailable or gpu_index is invalid.
    explicit gpu_affinity_context(int gpu_index, int width, int height);
    ~gpu_affinity_context();

    gpu_affinity_context(const gpu_affinity_context&)            = delete;
    gpu_affinity_context& operator=(const gpu_affinity_context&) = delete;

    /// Run a function on the affinity context's GL thread (blocking).
    template <typename Func>
    auto dispatch_sync(Func&& func) -> decltype(func())
    {
        using result_t = decltype(func());
        std::promise<result_t> promise;
        auto future = promise.get_future();

        dispatch([f = std::forward<Func>(func), &promise]() mutable {
            try {
                if constexpr (std::is_void_v<result_t>) {
                    f();
                    promise.set_value();
                } else {
                    promise.set_value(f());
                }
            } catch (...) {
                promise.set_exception(std::current_exception());
            }
        });

        return future.get();
    }

    /// Run a function on the affinity context's GL thread asynchronously.
    /// Returns a future that can be polled with wait_for() — essential for
    /// abortable shutdown paths (caller can stop waiting if its `running_`
    /// flag goes false while the affinity executor is wedged).
    template <typename Func>
    auto dispatch_async(Func&& func) -> std::future<decltype(func())>
    {
        using result_t = decltype(func());
        auto promise = std::make_shared<std::promise<result_t>>();
        auto future  = promise->get_future();

        dispatch([f = std::forward<Func>(func), promise]() mutable {
            try {
                if constexpr (std::is_void_v<result_t>) {
                    f();
                    promise->set_value();
                } else {
                    promise->set_value(f());
                }
            } catch (...) {
                promise->set_exception(std::current_exception());
            }
        });

        return future;
    }

    /// Upload CPU frame data to the internal texture. Returns the GL texture ID.
    /// Must be called via dispatch_sync (on the affinity thread).
    GLuint upload_frame(const uint8_t* pixels, int width, int height, int stride);

    /// Get the current upload texture ID (valid on the affinity GL thread).
    GLuint texture_id() const { return upload_texture_; }

    /// Width/height of the internal texture
    int width() const { return width_; }
    int height() const { return height_; }

    /// GPU index this context is bound to
    int gpu_index() const { return gpu_index_; }

    /// Get the device LUID of this context's GPU (for verification)
    const uint8_t* device_luid() const { return device_luid_; }
    bool           device_luid_valid() const { return device_luid_valid_; }

  private:
    void dispatch(std::function<void()> func);
    void thread_func();
    bool create_affinity_context(int gpu_index);

    int gpu_index_;
    int width_;
    int height_;

    // GL resources (owned by the affinity thread)
    GLuint upload_texture_ = 0;
    GLuint pbo_[2]         = {0, 0};
    int    pbo_index_      = 0; // Double-buffered PBO for async upload
    bool   first_frame_    = true;

    // Device identification
    uint8_t device_luid_[8]    = {};
    bool    device_luid_valid_ = false;

    // Win32 handles
    HDC   affinity_dc_  = nullptr;
    HGLRC affinity_rc_  = nullptr;

    // Thread management
    std::thread              thread_;
    std::atomic<bool>        running_{false};
    std::mutex               queue_mutex_;
    std::condition_variable  queue_cv_;
    std::queue<std::function<void()>> work_queue_;
};

}} // namespace caspar::vulkan_output
