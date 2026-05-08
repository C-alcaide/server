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

#define VK_USE_PLATFORM_WIN32_KHR
#include <vulkan/vulkan.h>

#include <GL/glew.h>

#include <atomic>
#include <cstdint>
#include <memory>

namespace caspar { namespace accelerator { namespace ogl {
class device;
}}} // namespace caspar::accelerator::ogl

namespace caspar { namespace vulkan_output {

class vulkan_device;

// Double-buffered shared texture pair for zero-copy OGL → VK transfer.
// Manages GL_EXT_memory_object textures exported to Vulkan via Win32 handles,
// plus GL_EXT_semaphore / VK_KHR_external_semaphore for synchronization.
//
// Usage:
//   1. OGL thread: blit_from_texture(gl_texture_id) → copies into current write buffer
//   2. OGL thread: signal_gl() → signals the GL semaphore
//   3. VK thread:  wait_vk() → waits on the VK semaphore
//   4. VK thread:  vk_image() → returns the VkImage to present from
//   5. VK thread:  swap() → advance to next buffer
class shared_texture_pool
{
  public:
    /// Standard constructor: manages GL calls via the provided ogl::device thread
    shared_texture_pool(std::shared_ptr<accelerator::ogl::device> ogl_device,
                        vulkan_device&                            vk_device,
                        uint32_t                                  width,
                        uint32_t                                  height,
                        bool                                      use_16bit = false);

    /// Affinity constructor: caller is responsible for running on a valid GL context thread.
    /// Use this when creating from gpu_affinity_context::dispatch_sync().
    shared_texture_pool(vulkan_device& vk_device,
                        uint32_t       width,
                        uint32_t       height,
                        bool           use_16bit = false);

    ~shared_texture_pool();

    shared_texture_pool(const shared_texture_pool&)            = delete;
    shared_texture_pool& operator=(const shared_texture_pool&) = delete;

    // Called on OGL thread: copies from source OGL texture into current write slot
    void blit_from_texture(GLuint source_texture_id, int width, int height);

    // Called on OGL thread: signals that the write is complete
    void signal_gl();

    // Called on VK thread: waits for the GL signal
    VkSemaphore wait_semaphore_vk() const;

    // Called on VK thread: returns the VkImage for the current read slot
    VkImage     current_vk_image() const;
    VkImageView current_vk_image_view() const;

    // Called on VK thread after present: advance read/write indices
    void swap();

    uint32_t width() const { return width_; }
    uint32_t height() const { return height_; }

  private:
    struct slot
    {
        // GL side
        GLuint  gl_memory_object = 0;
        GLuint  gl_texture       = 0;
        GLuint  gl_semaphore     = 0;

        // VK side
        VkImage        vk_image      = VK_NULL_HANDLE;
        VkImageView    vk_image_view = VK_NULL_HANDLE;
        VkDeviceMemory vk_memory     = VK_NULL_HANDLE;
        VkSemaphore    vk_semaphore  = VK_NULL_HANDLE;

        // Shared handles
        HANDLE memory_handle    = nullptr;
        HANDLE semaphore_handle = nullptr;
    };

    void create_slot(slot& s);
    void destroy_slot(slot& s);
    void destroy_slot_vk_only(slot& s); // VK + handle cleanup only (no GL context needed)

    std::shared_ptr<accelerator::ogl::device> ogl_device_;
    vulkan_device&                            vk_device_;
    uint32_t                                  width_      = 0;
    uint32_t                                  height_     = 0;
    bool                                      use_16bit_  = false;

    static constexpr int BUFFER_COUNT = 3;
    slot                 slots_[BUFFER_COUNT];
    int                  write_index_ = 0;
    std::atomic<int>     read_index_{0};

    // FBO pair for format-converting blit (used when use_16bit_ is true)
    GLuint               read_fbo_  = 0;
    GLuint               draw_fbo_  = 0;
};

}} // namespace caspar::vulkan_output
