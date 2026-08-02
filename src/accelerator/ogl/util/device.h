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

#pragma once

#include <accelerator/accelerator.h>
#include <common/array.h>
#include <common/bit_depth.h>

#include <functional>
#include <future>

#ifdef WIN32
#include <GL/glew.h>
#endif

namespace caspar { namespace accelerator { namespace ogl {

class device final
    : public std::enable_shared_from_this<device>
    , public accelerator_device
{
  public:
    device();
    ~device();

    device(const device&) = delete;

    device& operator=(const device&) = delete;

    std::shared_ptr<class texture> create_texture(int width, int height, int stride, common::bit_depth depth, bool clear = true);
    array<uint8_t>                 create_array(int size);

    std::future<std::shared_ptr<class texture>>
    copy_async(const array<const uint8_t>& source, int width, int height, int stride, common::bit_depth depth);
    std::future<array<const uint8_t>> copy_async(const std::shared_ptr<class texture>& source);

    /// Box-filtered reduction of `source` followed by a readback: `levels`
    /// successive exact 2x2 averagings on the GPU, then one small copy back.
    ///
    /// For a consumer that wants a summary of the frame rather than the frame --
    /// see core::texture::read_pixels_reduced(). Reading level 3 of a 1080p frame
    /// moves 129 KB instead of 8.29 MB.
    ///
    /// The returned array is always packed 8-bit BGRA; the tuple carries the
    /// reduced width and height, which floor at each halving and so are not
    /// necessarily width>>levels.
    ///
    /// Takes the source as a bare GL name and dimensions rather than a
    /// shared_ptr<texture>, because the caller is texture::read_pixels_reduced(),
    /// a const member with no shared handle on itself. Safe because that caller
    /// blocks on the returned future, so the source outlives the chain -- only the
    /// first blit reads it, and every intermediate is drawn from the device's own
    /// pool.
    std::future<std::tuple<array<const uint8_t>, int, int>>
    reduce_and_copy_async(unsigned int source_id, int source_width, int source_height, int levels);
    template <typename Func>
    auto dispatch_async(Func&& func)
    {
        using result_type = decltype(func());
        using task_type   = std::packaged_task<result_type()>;

        auto task   = std::make_shared<task_type>(std::forward<Func>(func));
        auto future = task->get_future();
        dispatch([=] { (*task)(); });
        return future;
    }

    template <typename Func>
    auto dispatch_sync(Func&& func)
    {
        return dispatch_async(std::forward<Func>(func)).get();
    }

    std::wstring version() const;

    /// Return the native GL context handle (HGLRC on Windows, EGLContext on Linux) for context sharing.
    void* native_gl_context() const;

    /// Return the EGL display handle (Linux only, nullptr on Windows).
    void* native_egl_display() const;

    boost::property_tree::wptree info() const;
    std::future<void>            gc();

  private:
    void dispatch(std::function<void()> func);
    struct impl;
    std::shared_ptr<impl> impl_;
};

}}} // namespace caspar::accelerator::ogl
