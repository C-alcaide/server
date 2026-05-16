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

#include <common/memory.h>

#include <GL/glew.h>

#define VK_USE_PLATFORM_WIN32_KHR
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_win32.h>

#include <map>
#include <memory>
#include <mutex>

namespace caspar { namespace accelerator {

namespace ogl {
class device;
class channel_texture_store;
} // namespace ogl

namespace vulkan {

class device;

/// Zero-copy VK→GL texture bridge for previz channel texture sampling.
///
/// After the Vulkan mixer composites a channel, the output texture is posted
/// to this bridge.  The bridge creates a shared VkImage with exportable
/// memory, blits the composited result into it, and imports the memory into
/// the previz OpenGL context via GL_EXT_memory_object_win32.  Synchronization
/// is fence-based: post_channel() waits for the VK blit to complete before
/// returning, so GL can safely sample the shared texture immediately.
///
/// If GL_EXT_memory_object is unavailable, a CPU-readback fallback is used.
///
/// Modeled after vulkan_output/util/shared_texture_pool but in the
/// VK-renders → GL-reads direction.
class previz_texture_bridge
{
  public:
    previz_texture_bridge(const spl::shared_ptr<device>&            vk_device,
                          const std::shared_ptr<ogl::device>&       ogl_device);
    ~previz_texture_bridge();

    previz_texture_bridge(const previz_texture_bridge&)            = delete;
    previz_texture_bridge& operator=(const previz_texture_bridge&) = delete;

    /// Called on the VK dispatch thread after compositing a channel.
    /// Blits the VK source texture into a shared exportable image.
    /// The GL side can then sample it via get_gl_texture().
    ///
    /// @param channel_id    CasparCG channel index (1-based).
    /// @param source        The composited VK texture (device-local).
    /// @param source_layout Current layout of the source image.
    /// @param width         Texture width in pixels.
    /// @param height        Texture height in pixels.
    /// @param use_16bit     true for RGBA16, false for RGBA8.
    void post_channel(int                              channel_id,
                      VkImage                          source,
                      VkImageLayout                    source_layout,
                      int                              width,
                      int                              height,
                      bool                             use_16bit);

    /// Called on the OGL previz thread.  Returns the GL texture ID for the
    /// given channel, or 0 if no texture has been posted yet.
    struct gl_entry
    {
        GLuint tex_id = 0;
        int    width  = 0;
        int    height = 0;
    };
    gl_entry get_gl_texture(int channel_id);

    /// Posts all available channel GL textures into the given texture store.
    /// Called on the OGL previz thread before previz rendering.
    void sync_to_store(ogl::channel_texture_store& store);

    /// Whether zero-copy interop is active (vs CPU fallback).
    bool interop_available() const { return interop_available_; }

  private:
    struct channel_slot;

    void          create_slot(channel_slot& s, int width, int height, bool use_16bit);
    void          destroy_slot(channel_slot& s);

    spl::shared_ptr<device>            vk_device_;
    std::shared_ptr<ogl::device>       ogl_device_;
    bool                               interop_available_ = false;
    bool                               use_linear_tiling_ = false;

    mutable std::mutex                 mutex_;
    std::map<int, channel_slot>        slots_; // channel_id → slot

    // VK function pointers (loaded once)
    PFN_vkGetMemoryWin32HandleKHR      vkGetMemoryWin32HandleKHR_    = nullptr;
};

}}} // namespace caspar::accelerator::vulkan
