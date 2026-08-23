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

#include <common/bit_depth.h>
#include <common/render_format.h>
#include <core/frame/frame.h>
#include <cstdint>
#include <memory>
#include <vector>

namespace caspar { namespace accelerator { namespace ogl {

class device;

class texture final : public core::texture
{
  public:
    texture(int               width,
            int               height,
            int               stride,
            common::bit_depth depth  = common::bit_depth::bit8,
            /// Render targets on a channel with a linear working space pass fp16 here.
            /// Input textures, which are fed from integer AVFrames, must stay unorm.
            common::render_format format = common::render_format::unorm);
    texture(const texture&) = delete;
    texture(texture&& other);
    ~texture();

    texture& operator=(const texture&) = delete;
    texture& operator=(texture&& other);

#ifdef WIN32
    void copy_from(int source);
#endif
    void copy_from(class buffer& source);
    void copy_to(class buffer& dest);

    void attach();
    void clear();

    virtual void bind(int index) override;
    virtual void unbind() override;

    /// Publish this texture's contents to the share group, and record a fence a reader in
    /// another context can wait on.
    ///
    /// Called by the mixer when it finishes compositing into this texture AND is not about to
    /// read it back. The readback path already flushes -- `device::read_back` creates a fence
    /// and calls `glFlush` -- so before consumers could decline the readback, every composited
    /// frame was published as a side effect of being copied to host memory. Declining it removed
    /// the flush along with the copy.
    void publish_render(class device& dev);

    /// Wait, on the calling context, for the writes `publish_render` published.
    ///
    /// A server-side `glWaitSync`, so it costs no CPU and does not block the caller's thread; it
    /// orders the reader's subsequent commands behind the mixer's. A no-op when nothing has been
    /// published, which is the case for input textures and for any frame whose mixer did read
    /// back.
    void ensure_render_complete() const override;

    int               width() const;
    int               height() const;
    int               stride() const;
    common::bit_depth depth() const;
    void              set_depth(common::bit_depth depth);

    /// The numeric format this texture's storage was created with. Immutable: GL fixes the
    /// internal format at glTextureStorage2D, which is why the device's texture pool keys
    /// on it rather than calling a setter (see device::create_texture).
    common::render_format format() const;

    int size() const;
    int id() const;

    /// Enable on-demand GPU readback for PRINT RAW.
    /// Called automatically by ogl::device::create_texture().
    void set_device(std::weak_ptr<device> dev);

    /// The device that owns this texture's GL context, or nullptr. Lets a
    /// consumer dispatch GL work (e.g. a compute pack) onto the mixer's GL
    /// thread where this texture and its context are valid.
    std::shared_ptr<device> get_device() const;

    /// Identity of the owning ogl::device, so a mixer can tell whether this
    /// texture belongs to its own GL context before binding it.
    const void* owner_device() const override;

    bool tex_is_hbd() const override { return depth() != common::bit_depth::bit8; }

    /// Read pixel data from the GPU texture (dispatches to GL thread).
    /// Zero-cost during normal playback — only called by write_frame_png.
    std::vector<std::uint8_t> read_pixels() const override;

    /// Box-filtered reduction plus readback, for a consumer that needs a summary
    /// of the picture rather than the picture. See core::texture.
    std::vector<std::uint8_t> read_pixels_reduced(int levels, int& out_width, int& out_height) const override;

  private:
    struct impl;
    std::unique_ptr<impl> impl_;
};

}}} // namespace caspar::accelerator::ogl
