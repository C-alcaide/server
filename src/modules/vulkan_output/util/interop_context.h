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

#pragma once

#include <GL/glew.h>

#ifdef _WIN32
#include <windows.h>
#endif

#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>

namespace caspar { namespace vulkan_output {

// Lightweight GL context that shares texture objects with the main OGL device.
// Runs its own thread for executing GL commands (blit, signal) without blocking
// the mixer's OGL device thread. This is the key scalability enabler: the interop
// blit runs concurrently with the mixer's next frame render.
//
// Must be created while the OGL device context is current (i.e., inside a
// dispatch_sync on the OGL device). The shared context inherits all texture names.
class interop_context
{
  public:
    // Creates the shared context. Must be called from the OGL device thread.
    interop_context();
    ~interop_context();

    interop_context(const interop_context&)            = delete;
    interop_context& operator=(const interop_context&) = delete;

    // Queue a GL task and return immediately (non-blocking).
    void dispatch_async(std::function<void()> task);

    // Queue a GL task and wait for completion.
    void dispatch_sync(std::function<void()> task);

    // Returns true if the context was successfully created.
    bool valid() const { return valid_; }

  private:
    void thread_func();

    HWND  hwnd_  = nullptr;
    HDC   hdc_   = nullptr;
    HGLRC hglrc_ = nullptr;
    bool  valid_ = false;

    std::thread             thread_;
    std::mutex              mutex_;
    std::condition_variable cv_;
    std::condition_variable done_cv_;
    std::queue<std::function<void()>> tasks_;
    bool                    stop_ = false;
};

}} // namespace caspar::vulkan_output
