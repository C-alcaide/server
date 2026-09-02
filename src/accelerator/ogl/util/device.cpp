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
#include "device.h"

#include "buffer.h"
#include "context.h"
#include "shader.h"
#include "texture.h"

#include <common/array.h>
#include <common/assert.h>
#include <common/env.h>
#include <common/except.h>
#include <common/gl/gl_check.h>
#include <common/os/thread.h>

#include <GL/glew.h>

#ifdef WIN32
#include "../../d3d/d3d_device.h"
#include <GL/wglew.h>
#endif

#include <boost/asio/deadline_timer.hpp>
#include <boost/asio/dispatch.hpp>
#include <boost/asio/spawn.hpp>
#include <boost/property_tree/ptree.hpp>

#include <tbb/concurrent_queue.h>
#include <tbb/concurrent_unordered_map.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <future>
#include <thread>

namespace caspar { namespace accelerator { namespace ogl {

using namespace boost::asio;

struct device::impl : public std::enable_shared_from_this<impl>
{
    using texture_queue_t = tbb::concurrent_bounded_queue<std::shared_ptr<texture>>;
    using buffer_queue_t  = tbb::concurrent_bounded_queue<std::shared_ptr<buffer>>;

    std::unique_ptr<device_context> context_;

    // Outer dimension: 0 = unorm8, 1 = unorm16, 2 = fp16. See create_texture.
    std::array<std::array<tbb::concurrent_unordered_map<size_t, texture_queue_t>, 4>, 3> device_pools_;
    std::array<tbb::concurrent_unordered_map<size_t, buffer_queue_t>, 2>                 host_pools_;

    GLuint fbo_;

    /// Dedicated read/draw pair for the downscale chain in
    /// reduce_and_copy_async(). Separate from fbo_, which the kernels keep bound
    /// as GL_FRAMEBUFFER; these are only ever touched through the DSA
    /// glBlitNamedFramebuffer path, so a reduction cannot disturb whatever the
    /// mixer has attached. Created lazily -- most servers never ask for one.
    GLuint reduce_read_fbo_ = 0;
    GLuint reduce_draw_fbo_ = 0;

    std::wstring version_;

#ifdef WIN32
    std::shared_ptr<d3d::d3d_device> d3d_device_;
    std::shared_ptr<void>            interop_handle_;
#endif

    io_context                             io_context_;
    decltype(make_work_guard(io_context_)) work_;
    std::thread                            thread_;
    std::thread::id                        thread_id_;

    // ── GPU→CPU readback wait prediction ──
    // Rolling estimate of how long a readback takes, so copy_async can sleep for
    // about that long instead of polling on a fixed cadence. Published through
    // info() as gl.summary.readback_predicted_us.
    static constexpr int64_t fine_poll_us          = 250;
    static constexpr int64_t max_predicted_wait_us = 100'000;
    std::atomic<int64_t>     readback_us_{0};

    impl()
        : context_(new device_context())
        , work_(make_work_guard(io_context_))
    {
        CASPAR_LOG(info) << L"Initializing OpenGL Device.";

        context_->bind();

        auto err = glewInit();
        if (err != GLEW_OK && err != 4) { // GLEW_ERROR_NO_GLX_DISPLAY
            std::stringstream str;
            str << "Failed to initialize GLEW (" << (int)err << "): " << glewGetErrorString(err) << std::endl;
            CASPAR_THROW_EXCEPTION(gl::ogl_exception() << msg_info(str.str()));
        }

#ifdef WIN32
        if (wglewInit() != GLEW_OK) {
            CASPAR_THROW_EXCEPTION(gl::ogl_exception() << msg_info("Failed to initialize GLEW."));
        }
#endif

        version_ = u16(reinterpret_cast<const char*>(GL2(glGetString(GL_VERSION)))) + L" " +
                   u16(reinterpret_cast<const char*>(GL2(glGetString(GL_VENDOR))));

        CASPAR_LOG(info) << L"Initialized OpenGL " << version();

        // GL_RENDERER, and named the same way the Vulkan mixer names its card, because a
        // measurement harness has to know WHICH GPU to sample. GL_VERSION and GL_VENDOR give
        // "4.5.0 NVIDIA 582.53 NVIDIA Corporation" -- a vendor, not a model -- and NVML
        // enumerates by PCI bus id, so on a two-card box there is nothing to match against and
        // the sampler either guesses index 0 or reports nothing.
        //
        // Measured 2026-08-22: every GPU and VRAM figure for an OpenGL-mixer arm of
        // `encode-matrix` came back empty for exactly this reason, while the Vulkan arms were
        // attributed correctly -- the Vulkan side happens to log its adapter from the D3D11
        // import bridge (`d3d11_import_bridge.cpp`), which the OpenGL side has no equivalent of.
        CASPAR_LOG(info) << L"[ogl::device] the mixer's GPU is OpenGL renderer (\""
                         << u16(reinterpret_cast<const char*>(GL2(glGetString(GL_RENDERER))))
                         << L"\")";

        if (!GLEW_VERSION_4_5 && !glewIsSupported("GL_ARB_sync GL_ARB_shader_objects GL_ARB_multitexture "
                                                  "GL_ARB_direct_state_access GL_ARB_texture_barrier")) {
            CASPAR_THROW_EXCEPTION(not_supported()
                                   << msg_info("Your graphics card does not meet the minimum hardware requirements "
                                               "since it does not support OpenGL 4.5 or higher."));
        }

        GL(glCreateFramebuffers(1, &fbo_));
        GL(glBindFramebuffer(GL_FRAMEBUFFER, fbo_));

        context_->unbind();

#ifdef WIN32
        if (env::properties().get(L"configuration.html.enable-gpu", false)) {
            d3d_device_ = d3d::d3d_device::get_device();
        }
        if (d3d_device_) {
            interop_handle_ = std::shared_ptr<void>(wglDXOpenDeviceNV(d3d_device_->device()), [](void* p) {
                if (p)
                    wglDXCloseDeviceNV(p);
            });

            if (!interop_handle_)
                CASPAR_THROW_EXCEPTION(gl::ogl_exception() << msg_info("Failed to initialize d3d interop."));
        }
#endif

        thread_ = std::thread([&] {
            thread_id_ = std::this_thread::get_id();
            context_->bind();
            set_thread_name(L"OpenGL Device");
            io_context_.run();
            context_->unbind();
        });
    }

    ~impl()
    {
        work_.reset();
        thread_.join();

        context_->bind();

        for (auto& pool : host_pools_)
            pool.clear();

        for (auto& pools : device_pools_)
            for (auto& pool : pools)
                pool.clear();

        // Use raw call instead of GL() macro — destructors must not throw.
        glDeleteFramebuffers(1, &fbo_);
        if (reduce_read_fbo_)
            glDeleteFramebuffers(1, &reduce_read_fbo_);
        if (reduce_draw_fbo_)
            glDeleteFramebuffers(1, &reduce_draw_fbo_);
        while (glGetError() != GL_NO_ERROR) {}
    }

    template <typename Func>
    auto spawn_async(Func&& func)
    {
        using result_type = decltype(func(std::declval<yield_context>()));
        using task_type   = std::packaged_task<result_type(yield_context)>;

        auto task   = task_type(std::forward<Func>(func));
        auto future = task.get_future();
        boost::asio::spawn(io_context_, std::move(task), [](std::exception_ptr e) {
            if (e)
                std::rethrow_exception(e);
        });
        return future;
    }

    template <typename Func>
    auto dispatch_async(Func&& func)
    {
        using result_type = decltype(func());
        using task_type   = std::packaged_task<result_type()>;

        auto task   = task_type(std::forward<Func>(func));
        auto future = task.get_future();
        boost::asio::dispatch(io_context_, std::move(task));
        return future;
    }

    template <typename Func>
    auto dispatch_sync(Func&& func) -> decltype(func())
    {
        // If already on the device thread, execute directly to avoid deadlock.
        if (std::this_thread::get_id() == thread_id_)
            return func();
        return dispatch_async(std::forward<Func>(func)).get();
    }

    std::wstring version() { return version_; }

    std::shared_ptr<texture>
    create_texture(int width, int height, int stride, common::bit_depth depth, bool clear, common::render_format format)
    {
        CASPAR_VERIFY(stride > 0 && stride < 5);
        CASPAR_VERIFY(width > 0 && height > 0);

        // The pool index must include the render format, not just the depth. GL fixes a
        // texture's internal format at glTextureStorage2D and it cannot be changed
        // afterwards, so pooling fp16 and unorm16 textures together would hand an fp16
        // texture back as a unorm16 one -- set_depth() would agree and the storage would
        // silently disagree. Rows: 0 = unorm8, 1 = unorm16, 2 = fp16.
        const auto depth_pool_index = format == common::render_format::fp16
                                          ? 2
                                          : (depth == common::bit_depth::bit8 ? 0 : 1);

        // TODO (perf) Shared pool.
        auto pool = &device_pools_[depth_pool_index][stride - 1][(width << 16 & 0xFFFF0000) | (height & 0x0000FFFF)];

        std::shared_ptr<texture> tex;
        if (!pool->try_pop(tex)) {
            tex = std::make_shared<texture>(width, height, stride, depth, format);
        }
        tex->set_depth(depth);

        if (clear) {
            tex->clear();
        }

        auto ptr = tex.get();
        return std::shared_ptr<texture>(
            ptr, [tex = std::move(tex), pool, self = shared_from_this()](texture*) mutable { pool->push(tex); });
    }

    std::shared_ptr<buffer> create_buffer(int size, bool write)
    {
        CASPAR_VERIFY(size > 0);

        // TODO (perf) Shared pool.
        auto pool = &host_pools_[static_cast<int>(write ? 1 : 0)][size];

        std::shared_ptr<buffer> buf;
        if (!pool->try_pop(buf)) {
            // TODO (perf) Avoid blocking in create_array.
            dispatch_sync([&] { buf = std::make_shared<buffer>(size, write); });
        }

        buf->set_owner_device(this);

        auto ptr = buf.get();
        return std::shared_ptr<buffer>(ptr, [buf = std::move(buf), self = shared_from_this()](buffer*) mutable {
            auto pool = &self->host_pools_[static_cast<int>(buf->write() ? 1 : 0)][buf->size()];
            pool->push(std::move(buf));
        });
    }

    array<uint8_t> create_array(int size)
    {
        auto buf = create_buffer(size, true);
        auto ptr = reinterpret_cast<uint8_t*>(buf->data());
        return array<uint8_t>(ptr, buf->size(), std::move(buf));
    }

    std::future<std::shared_ptr<texture>>
    copy_async(const array<const uint8_t>& source, int width, int height, int stride, common::bit_depth depth)
    {
        return dispatch_async([=, this] {
            std::shared_ptr<buffer> buf;

            // The array may already carry the PBO it was written into, which lets
            // us skip a memcpy -- but only if that PBO was created by *this*
            // device. A GL buffer name is meaningless in another context, and a
            // routed frame can reach a mixer holding a different ogl::device.
            auto tmp = source.storage<std::shared_ptr<buffer>>();
            if (tmp && *tmp && (*tmp)->owner_device() == static_cast<const void*>(this)) {
                buf = *tmp;
            } else {
                buf = create_buffer(static_cast<int>(source.size()), true);
                // TODO (perf) Copy inside a TBB worker.
                std::memcpy(buf->data(), source.data(), source.size());
            }

            // Frame upload: always unorm, the source is an integer AVFrame.
            auto tex = create_texture(width, height, stride, depth, false, common::render_format::unorm);
            tex->copy_from(*buf);
            // TODO (perf) save tex on source
            return tex;
        });
    }

    std::future<array<const uint8_t>> copy_async(const std::shared_ptr<texture>& source)
    {
        return spawn_async([=, this](yield_context yield) { return read_back(source, yield); });
    }

    /// Downscale by successive exact halvings, then read the small result back.
    ///
    /// A GL_LINEAR blit into an exactly-halved target samples the four contributing
    /// texels at equal weight, so each pass is a true 2x2 box average and `levels`
    /// of them compose into a 2^levels box filter -- the same thing a caller
    /// averaging a region would compute, just done on the GPU over 1/4^levels the
    /// bytes. glGenerateMipmap is not an option: pooled textures are allocated with
    /// glTextureStorage2D(..., levels=1, ...) and have no mip chain.
    std::future<std::tuple<array<const uint8_t>, int, int>>
    reduce_and_copy_async(GLuint source_id, int source_width, int source_height, int levels)
    {
        return spawn_async([=, this](yield_context yield) -> std::tuple<array<const uint8_t>, int, int> {
            const int n = std::clamp(levels, 0, 8);

            std::vector<std::pair<int, int>> chain;
            int                              w = source_width;
            int                              h = source_height;
            for (int i = 0; i < n; ++i) {
                const int nw = std::max(1, w / 2);
                const int nh = std::max(1, h / 2);
                if (nw == w && nh == h)
                    break; // already 1x1
                chain.emplace_back(nw, nh);
                w = nw;
                h = nh;
            }
            // Always at least one pass, so the result is RGBA8 even when the source
            // is a 16-bit channel texture -- the documented contract is packed
            // 8-bit regardless of the texture's own depth.
            if (chain.empty())
                chain.emplace_back(w, h);

            if (!reduce_read_fbo_) {
                GL(glCreateFramebuffers(1, &reduce_read_fbo_));
                GL(glCreateFramebuffers(1, &reduce_draw_fbo_));
            }

            // The first blit reads the caller's texture by name; everything after
            // that reads the previous pooled intermediate.
            std::shared_ptr<texture> cur;
            GLuint                   cur_id = source_id;
            int                      cur_w  = source_width;
            int                      cur_h  = source_height;

            for (auto [nw, nh] : chain) {
                auto dst = create_texture(nw, nh, 4, common::bit_depth::bit8, false, common::render_format::unorm);
                GL(glNamedFramebufferTexture(reduce_read_fbo_, GL_COLOR_ATTACHMENT0, cur_id, 0));
                GL(glNamedFramebufferTexture(reduce_draw_fbo_, GL_COLOR_ATTACHMENT0, dst->id(), 0));
                GL(glNamedFramebufferReadBuffer(reduce_read_fbo_, GL_COLOR_ATTACHMENT0));
                GL(glNamedFramebufferDrawBuffer(reduce_draw_fbo_, GL_COLOR_ATTACHMENT0));
                GL(glBlitNamedFramebuffer(reduce_read_fbo_, reduce_draw_fbo_, 0, 0, cur_w, cur_h, 0, 0, nw, nh,
                                          GL_COLOR_BUFFER_BIT, GL_LINEAR));
                cur    = dst;
                cur_id = static_cast<GLuint>(dst->id());
                cur_w  = nw;
                cur_h  = nh;
            }

            // Detach so the chain does not keep the last texture alive inside the
            // FBO after it goes back to the pool.
            GL(glNamedFramebufferTexture(reduce_read_fbo_, GL_COLOR_ATTACHMENT0, 0, 0));
            GL(glNamedFramebufferTexture(reduce_draw_fbo_, GL_COLOR_ATTACHMENT0, 0, 0));

            return {read_back(cur, yield), cur_w, cur_h};
        });
    }

  private:
    /// The readback itself: copy to a PBO, then wait for the fence without ever
    /// blocking the GL thread. Shared by copy_async() and reduce_and_copy_async()
    /// so the wait prediction below has one implementation and one set of
    /// statistics.
    array<const uint8_t> read_back(const std::shared_ptr<texture>& source, yield_context yield)
    {
        auto buf = create_buffer(source->size(), false);
        source->copy_to(*buf);

        auto fence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);

        GL(glFlush());

        // Wait for the readback by yielding to the io_context between checks,
        // never by blocking: this runs on the single GL thread, so blocking it
        // would stall every other channel's uploads and composition too.
        //
        // The wait used to be a flat 2 ms poll, which cost up to 2 ms of
        // latency per frame per channel and woke the thread on a fixed cadence
        // regardless of how long the copy actually takes. Readback duration is
        // very stable for a given resolution, so predict it: sleep for roughly
        // the last measured duration, then poll finely. Warm, this lands within
        // a few hundred microseconds of completion after a single wakeup.
        const auto start = std::chrono::steady_clock::now();

        auto predicted_us = readback_us_.load(std::memory_order_relaxed);
        // Undershoot the prediction so the fine poll converges from below
        // rather than overshooting into added latency.
        auto first_wait_us = predicted_us > fine_poll_us ? predicted_us - fine_poll_us : 0;

        deadline_timer timer(io_context_);
        for (auto n = 0;; ++n) {
            const auto wait_us = n == 0 ? first_wait_us : fine_poll_us;
            if (wait_us > 0) {
                timer.expires_from_now(boost::posix_time::microseconds(wait_us));
                timer.async_wait(yield);
            }

            auto wait = glClientWaitSync(fence, 0, 1);
            if (wait == GL_ALREADY_SIGNALED || wait == GL_CONDITION_SATISFIED) {
                break;
            }
            // GL_WAIT_FAILED is an ERROR, not "not yet": the sync object is unusable and
            // will never signal, so treating it as a timeout spun this loop forever with
            // no diagnostic. GL_TIMEOUT_EXPIRED is the only result worth another pass.
            if (wait == GL_WAIT_FAILED) {
                CASPAR_THROW_EXCEPTION(caspar_exception() << msg_info(
                    "glClientWaitSync failed on a readback fence; it will never signal"));
            }
            // And bound the polling. This loop had no exit but success, which is the safe
            // direction for a SLOW gpu -- unlike the Vulkan backend, it never recycled
            // anything still in flight, so waiting cost latency rather than correctness --
            // but it also meant a wedged driver hung the GL thread silently and forever,
            // stalling every channel that shares it. The Vulkan side now throws after the
            // same budget; matching it here is what keeps the two backends' failure
            // behaviour comparable. See accelerator/vulkan/util/gpu_wait.h.
            if (std::chrono::steady_clock::now() - start > std::chrono::seconds(10)) {
                CASPAR_THROW_EXCEPTION(caspar_exception() << msg_info(
                    "a readback fence did not signal within the wait budget"));
            }
        }

        glDeleteSync(fence);

        const auto elapsed_us = static_cast<int64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start)
                .count());

        // Asymmetric moving average. Sleeping for a prediction that is too
        // LONG turns directly into output latency, so track downwards fast
        // (weight 1/2) and upwards slowly (weight 1/8): an abrupt drop in
        // GPU load converges in two or three frames, while a one-off
        // scheduling spike barely moves the estimate.
        int64_t next;
        if (predicted_us == 0) {
            next = elapsed_us;
        } else if (elapsed_us < predicted_us) {
            next = (predicted_us + elapsed_us) / 2;
        } else {
            next = (predicted_us * 7 + elapsed_us) / 8;
        }
        readback_us_.store(std::clamp<int64_t>(next, 0, max_predicted_wait_us), std::memory_order_relaxed);

        auto ptr  = reinterpret_cast<uint8_t*>(buf->data());
        auto size = buf->size();
        return array<const uint8_t>(ptr, size, std::move(buf));
    }

  public:
    boost::property_tree::wptree info() const
    {
        boost::property_tree::wptree info;

        boost::property_tree::wptree pooled_device_buffers;
        size_t                       total_pooled_device_buffer_size  = 0;
        size_t                       total_pooled_device_buffer_count = 0;

        for (size_t i = 0; i < device_pools_.size(); ++i) {
            auto& depth_pools = device_pools_.at(i);
            for (size_t j = 0; j < depth_pools.size(); ++j) {
                auto& pools      = depth_pools.at(j);
                bool  mipmapping = j > 3;
                auto  stride     = mipmapping ? j - 3 : j + 1;

                for (auto& pool : pools) {
                    auto width  = pool.first >> 16;
                    auto height = pool.first & 0x0000FFFF;
                    auto size   = width * height * stride;
                    auto count  = pool.second.size();

                    if (count == 0)
                        continue;

                    boost::property_tree::wptree pool_info;

                    pool_info.add(L"stride", stride);
                    pool_info.add(L"mipmapping", mipmapping);
                    pool_info.add(L"width", width);
                    pool_info.add(L"height", height);
                    pool_info.add(L"size", size);
                    pool_info.add(L"count", count);

                    total_pooled_device_buffer_size += size * count;
                    total_pooled_device_buffer_count += count;

                    pooled_device_buffers.add_child(L"device_buffer_pool", pool_info);
                }
            }
        }

        info.add_child(L"gl.details.pooled_device_buffers", pooled_device_buffers);

        boost::property_tree::wptree pooled_host_buffers;
        size_t                       total_read_size   = 0;
        size_t                       total_write_size  = 0;
        size_t                       total_read_count  = 0;
        size_t                       total_write_count = 0;

        for (size_t i = 0; i < host_pools_.size(); ++i) {
            auto& pools    = host_pools_.at(i);
            auto  is_write = i == 1;

            for (auto& pool : pools) {
                auto size  = pool.first;
                auto count = pool.second.size();

                if (count == 0)
                    continue;

                boost::property_tree::wptree pool_info;

                pool_info.add(L"usage", is_write ? L"write_only" : L"read_only");
                pool_info.add(L"size", size);
                pool_info.add(L"count", count);

                pooled_host_buffers.add_child(L"host_buffer_pool", pool_info);

                (is_write ? total_write_count : total_read_count) += count;
                (is_write ? total_write_size : total_read_size) += size * count;
            }
        }

        info.add_child(L"gl.details.pooled_host_buffers", pooled_host_buffers);
        info.add(L"gl.summary.pooled_device_buffers.total_count", total_pooled_device_buffer_count);
        info.add(L"gl.summary.pooled_device_buffers.total_size", total_pooled_device_buffer_size);
        // info.add_child(L"gl.summary.all_device_buffers", texture::info());
        info.add(L"gl.summary.pooled_host_buffers.total_read_count", total_read_count);
        info.add(L"gl.summary.pooled_host_buffers.total_write_count", total_write_count);
        info.add(L"gl.summary.pooled_host_buffers.total_read_size", total_read_size);
        info.add(L"gl.summary.pooled_host_buffers.total_write_size", total_write_size);
        info.add_child(L"gl.summary.all_host_buffers", buffer::info());
        info.add(L"gl.summary.readback_predicted_us", readback_us_.load(std::memory_order_relaxed));

        return info;
    }

    std::future<void> gc()
    {
        return spawn_async([this](yield_context yield) {
            CASPAR_LOG(info) << " ogl: Running GC.";

            try {
                for (auto& depth_pools : device_pools_) {
                    for (auto& pools : depth_pools) {
                        for (auto& pool : pools)
                            pool.second.clear();
                    }
                }
                for (auto& pools : host_pools_) {
                    for (auto& pool : pools)
                        pool.second.clear();
                }
            } catch (...) {
                CASPAR_LOG_CURRENT_EXCEPTION();
            }
        });
    }
};

device::device()
    : impl_(new impl())
{
}
device::~device() {}
std::shared_ptr<texture> device::create_texture(int                   width,
                                                int                   height,
                                                int                   stride,
                                                common::bit_depth     depth,
                                                bool                  clear,
                                                common::render_format format)
{
    auto tex = impl_->create_texture(width, height, stride, depth, clear, format);
    tex->set_device(shared_from_this());
    return tex;
}
array<uint8_t> device::create_array(int size) { return impl_->create_array(size); }
std::future<std::shared_ptr<texture>>
device::copy_async(const array<const uint8_t>& source, int width, int height, int stride, common::bit_depth depth)
{
    return impl_->copy_async(source, width, height, stride, depth);
}
#ifdef WIN32
std::shared_ptr<void> device::d3d_interop() const { return impl_->interop_handle_; }
#endif

std::future<array<const uint8_t>> device::copy_async(const std::shared_ptr<texture>& source)
{
    return impl_->copy_async(source);
}
std::future<std::tuple<array<const uint8_t>, int, int>>
device::reduce_and_copy_async(unsigned int source_id, int source_width, int source_height, int levels)
{
    return impl_->reduce_and_copy_async(static_cast<GLuint>(source_id), source_width, source_height, levels);
}
void device::dispatch(std::function<void()> func) { boost::asio::dispatch(impl_->io_context_, std::move(func)); }
std::wstring                 device::version() const { return impl_->version(); }
void*                        device::native_gl_context() const { return impl_->context_->native_handle(); }
void*                        device::native_egl_display() const { return impl_->context_->native_egl_display(); }
boost::property_tree::wptree device::info() const { return impl_->info(); }
std::future<void>            device::gc() { return impl_->gc(); }
}}} // namespace caspar::accelerator::ogl
