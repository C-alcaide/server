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
#include "texture.h"

#include "buffer.h"
#include "device.h"

#include <common/bit_depth.h>
#include <common/gl/gl_check.h>
#include <common/log.h>

#include <GL/glew.h>

namespace caspar { namespace accelerator { namespace ogl {

static GLenum FORMAT[] = {0, GL_RED, GL_RG, GL_BGR, GL_BGRA};

// Row 0 is unorm8, row 1 unorm16, row 2 fp16. The fp16 row is selected by
// render_format rather than by bit_depth -- see common/render_format.h for why the two
// are separate concepts. External format stays GL_BGRA for stride 4, exactly as the
// unorm16 row does, so channel order is unchanged on the float path.
static GLenum INTERNAL_FORMAT[][5] = {{0, GL_R8, GL_RG8, GL_RGB8, GL_RGBA8},
                                      {0, GL_R16, GL_RG16, GL_RGB16, GL_RGBA16},
                                      {0, GL_R16F, GL_RG16F, GL_RGB16F, GL_RGBA16F}};
static GLenum TYPE[][5]            = {{0, GL_UNSIGNED_BYTE, GL_UNSIGNED_BYTE, GL_UNSIGNED_BYTE, GL_UNSIGNED_INT_8_8_8_8_REV},
                           {0, GL_UNSIGNED_SHORT, GL_UNSIGNED_SHORT, GL_UNSIGNED_SHORT, GL_UNSIGNED_SHORT},
                           {0, GL_HALF_FLOAT, GL_HALF_FLOAT, GL_HALF_FLOAT, GL_HALF_FLOAT}};

/// Which row of INTERNAL_FORMAT/TYPE a (depth, format) pair selects.
static int format_row(common::bit_depth depth, common::render_format fmt)
{
    if (fmt == common::render_format::fp16)
        return 2;
    return depth == common::bit_depth::bit8 ? 0 : 1;
}

struct texture::impl
{
    GLuint                id_     = 0;
    GLsizei               width_  = 0;
    GLsizei               height_ = 0;
    GLsizei               stride_ = 0;
    GLsizei               size_   = 0;
    common::bit_depth     depth_;
    common::render_format format_;
    std::weak_ptr<device> device_;

    impl(const impl&)            = delete;
    impl& operator=(const impl&) = delete;

  public:
    impl(int width, int height, int stride, common::bit_depth depth, common::render_format format)
        : width_(width)
        , height_(height)
        , stride_(stride)
        , depth_(depth)
        , format_(format)
        // fp16 is two bytes per component, the same as unorm16, so the readback size is
        // unchanged -- what differs is the interpretation, not the extent.
        , size_(width * height * stride *
                ((depth == common::bit_depth::bit8 && format == common::render_format::unorm) ? 1 : 2))
    {
        GL(glCreateTextures(GL_TEXTURE_2D, 1, &id_));
        GL(glTextureParameteri(id_, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
        GL(glTextureParameteri(id_, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
        GL(glTextureParameteri(id_, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
        GL(glTextureParameteri(id_, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
        GL(glTextureStorage2D(id_, 1, INTERNAL_FORMAT[format_row(depth_, format_)][stride_], width_, height_));
    }

    ~impl() { glDeleteTextures(1, &id_); }

    void bind() { GL(glBindTexture(GL_TEXTURE_2D, id_)); }

    void bind(int index)
    {
        GL(glActiveTexture(GL_TEXTURE0 + index));
        bind();
    }

    void unbind() { GL(glBindTexture(GL_TEXTURE_2D, 0)); }

    void attach() { GL(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + 0, GL_TEXTURE_2D, id_, 0)); }

    void clear()
    {
        GL(glClearTexImage(id_, 0, FORMAT[stride_], TYPE[format_row(depth_, format_)][stride_], nullptr));
    }

#ifdef WIN32
    void copy_from(int texture_id)
    {
        GL(glCopyImageSubData(
            texture_id, GL_TEXTURE_2D, 0, 0, 0, 0, id_, GL_TEXTURE_2D, 0, 0, 0, 0, width_, height_, 1));
    }
#endif

    void copy_from(buffer& src)
    {
        src.bind();

        if (width_ % 16 > 0) {
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        } else {
            glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
        }

        GL(glTextureSubImage2D(id_,
                               0,
                               0,
                               0,
                               width_,
                               height_,
                               FORMAT[stride_],
                               TYPE[format_row(depth_, format_)][stride_],
                               nullptr));

        src.unbind();
    }

    void copy_to(buffer& dst)
    {
        dst.bind();
        GL(glGetTextureImage(
            id_, 0, FORMAT[stride_], TYPE[format_row(depth_, format_)][stride_], size_, nullptr));
        dst.unbind();
    }
};

texture::texture(int width, int height, int stride, common::bit_depth depth, common::render_format format)
    : impl_(new impl(width, height, stride, depth, format))
{
}
texture::texture(texture&& other)
    : impl_(std::move(other.impl_))
{
}
texture::~texture() {}
texture& texture::operator=(texture&& other)
{
    impl_ = std::move(other.impl_);
    return *this;
}
void texture::bind(int index) { impl_->bind(index); }
void texture::unbind() { impl_->unbind(); }
void texture::attach() { impl_->attach(); }
void texture::clear() { impl_->clear(); }
#ifdef WIN32
void texture::copy_from(int source) { impl_->copy_from(source); }
#endif
void              texture::copy_from(buffer& source) { impl_->copy_from(source); }
void              texture::copy_to(buffer& dest) { impl_->copy_to(dest); }
int               texture::width() const { return impl_->width_; }
int               texture::height() const { return impl_->height_; }
int               texture::stride() const { return impl_->stride_; }
common::bit_depth texture::depth() const { return impl_->depth_; }
void              texture::set_depth(common::bit_depth depth) { impl_->depth_ = depth; }
common::render_format texture::format() const { return impl_->format_; }
int               texture::size() const { return impl_->size_; }
int               texture::id() const { return impl_->id_; }

void texture::set_device(std::weak_ptr<device> dev) { impl_->device_ = std::move(dev); }

std::shared_ptr<device> texture::get_device() const { return impl_->device_.lock(); }

const void* texture::owner_device() const { return static_cast<const void*>(impl_->device_.lock().get()); }

std::vector<std::uint8_t> texture::read_pixels() const
{
    auto dev = impl_->device_.lock();
    if (!dev)
        return {};

    const GLuint tex_id    = impl_->id_;
    const int    s         = impl_->stride_;
    const int    depth_idx = format_row(impl_->depth_, impl_->format_);
    const int    total     = impl_->size_;

    std::vector<std::uint8_t> result(total);
    dev->dispatch_sync([&] {
        GL(glGetTextureImage(tex_id, 0, FORMAT[s], TYPE[depth_idx][s], total, result.data()));
    });
    return result;
}

std::vector<std::uint8_t> texture::read_pixels_reduced(int levels, int& out_width, int& out_height) const
{
    out_width = out_height = 0;

    auto dev = impl_->device_.lock();
    if (!dev)
        return {};

    // The device owns the downscale chain and the readback: it has the FBOs, the
    // pool the intermediates are drawn from, and the non-blocking fence wait that
    // must not be duplicated here.
    //
    // Blocking by design. This is issued from a consumer's own thread -- the DMX
    // senders run at 10-30 Hz, well off the channel tick -- never from the channel
    // thread or the GL thread. Blocking is also what makes passing a bare GL name
    // safe: this texture cannot be destroyed while the call is outstanding.
    try {
        auto [data, w, h] = dev->reduce_and_copy_async(impl_->id_, impl_->width_, impl_->height_, levels).get();
        if (!data.data() || data.size() == 0)
            return {};

        out_width  = w;
        out_height = h;
        return std::vector<std::uint8_t>(data.begin(), data.end());
    } catch (...) {
        CASPAR_LOG(warning) << L"[ogl::texture] reduced readback failed; the caller will fall back to a full one.";
        return {};
    }
}

}}} // namespace caspar::accelerator::ogl
