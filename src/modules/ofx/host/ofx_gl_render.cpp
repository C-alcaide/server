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
 */

#include "ofx_gl_render.h"

#include <GL/glew.h>

#include <SFML/Window/Context.hpp>

#include <mutex>
#include <stdexcept>

namespace caspar { namespace ofx {

namespace {
std::once_flag g_glew_init_flag;
bool           g_glew_ok = false;
} // namespace

struct gl_backend::impl
{
    sf::Context  context;      ///< offscreen GL context, active on the creating thread
    unsigned int source_tex = 0;
    unsigned int output_tex = 0;
    int          out_w      = 0;
    int          out_h      = 0;

    impl()
    {
        context.setActive(true);
        std::call_once(g_glew_init_flag, [] {
            glewExperimental = GL_TRUE;
            g_glew_ok        = (glewInit() == GLEW_OK);
        });
        if (!g_glew_ok)
            throw std::runtime_error("[ofx] GLEW init failed for OpenGL render backend");
    }

    ~impl()
    {
        context.setActive(true);
        if (source_tex)
            glDeleteTextures(1, &source_tex);
        if (output_tex)
            glDeleteTextures(1, &output_tex);
        context.setActive(false);
    }
};

gl_backend::gl_backend()
    : impl_(std::make_unique<impl>())
{
}

gl_backend::~gl_backend() = default;

bool gl_backend::make_current() { return impl_->context.setActive(true); }

unsigned int gl_backend::upload_source(const std::uint8_t* rgba, int width, int height)
{
    if (impl_->source_tex == 0)
        glGenTextures(1, &impl_->source_tex);

    glBindTexture(GL_TEXTURE_2D, impl_->source_tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba);
    glBindTexture(GL_TEXTURE_2D, 0);
    return impl_->source_tex;
}

unsigned int gl_backend::ensure_output(int width, int height)
{
    if (impl_->output_tex == 0 || impl_->out_w != width || impl_->out_h != height) {
        if (impl_->output_tex == 0)
            glGenTextures(1, &impl_->output_tex);
        glBindTexture(GL_TEXTURE_2D, impl_->output_tex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glBindTexture(GL_TEXTURE_2D, 0);
        impl_->out_w = width;
        impl_->out_h = height;
    }
    return impl_->output_tex;
}

void gl_backend::readback_output(std::uint8_t* rgba, int width, int height)
{
    GLuint fbo = 0;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, impl_->output_tex, 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE) {
        GLint prev_align = 4;
        glGetIntegerv(GL_PACK_ALIGNMENT, &prev_align);
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, rgba);
        glPixelStorei(GL_PACK_ALIGNMENT, prev_align);
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &fbo);
}

}} // namespace caspar::ofx
