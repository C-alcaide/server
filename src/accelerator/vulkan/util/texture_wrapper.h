/*
 * Copyright 2025
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
 * Author: CasparVP
 */

#pragma once

#include "texture.h"

#include <core/frame/frame.h>

#include <atomic>
#include <memory>

namespace caspar { namespace accelerator { namespace vulkan {

/**
 * Wraps a vulkan::texture as a core::texture so that it can be returned
 * from image_mixer::render() and inspected by consumers downstream.
 *
 * bind()/unbind() are no-ops — Vulkan textures cannot be bound to GL
 * texture units. Consumers that need GPU-native access should dynamic_cast
 * to texture_wrapper and call vk_texture() instead.
 */
class texture_wrapper final : public core::texture
{
  public:
    explicit texture_wrapper(std::shared_ptr<vulkan::texture> tex)
        : tex_(std::move(tex))
    {
    }

    texture_wrapper(std::shared_ptr<vulkan::texture> tex, std::function<void()> wait_fn,
                    void* sem_handle = nullptr, uint64_t sem_value = 0)
        : tex_(std::move(tex))
        , wait_fn_(std::move(wait_fn))
        , sem_handle_(sem_handle)
        , sem_value_(sem_value)
    {
    }

    void bind(int /*index*/) override {} // No-op for Vulkan
    void unbind() override {}            // No-op for Vulkan

    void*              export_win32_handle() const override { return tex_->export_win32_handle(); }
    unsigned long long export_alloc_size() const override   { return static_cast<unsigned long long>(tex_->alloc_size()); }
    int                tex_width() const override           { return tex_->width(); }
    int                tex_height() const override          { return tex_->height(); }
    bool               tex_is_hbd() const override          { return tex_->depth() != common::bit_depth::bit8; }

    std::shared_ptr<vulkan::texture> vk_texture() const { return tex_; }

    int width() const { return tex_->width(); }
    int height() const { return tex_->height(); }

    /// Returns a Win32 HANDLE to an exportable timeline VkSemaphore that is
    /// signaled when the mixer's GPU rendering completes.  Returns nullptr
    /// if not available.  Caller must NOT close the handle.
    void*    render_semaphore_handle() const override { return sem_handle_; }
    /// Returns the timeline value that the semaphore will reach on completion.
    uint64_t render_semaphore_value() const override  { return sem_value_; }

    /// Call before importing the VK texture on a different device.
    /// Waits for the mixer's GPU rendering to complete (fence wait).
    /// No-op if already complete or no fence was attached.
    /// Thread-safe: multiple consumers can call concurrently; only the first
    /// caller actually waits, subsequent calls are no-ops.
    void ensure_render_complete() const override
    {
        if (wait_completed_.test_and_set(std::memory_order_acquire))
            return; // Another thread already waited or is waiting
        if (wait_fn_) {
            wait_fn_();
        }
    }

  private:
    std::shared_ptr<vulkan::texture>      tex_;
    std::function<void()>                 wait_fn_;
    void*                                 sem_handle_ = nullptr;
    uint64_t                              sem_value_  = 0;
    mutable std::atomic_flag              wait_completed_ = ATOMIC_FLAG_INIT;
};

}}} // namespace caspar::accelerator::vulkan
