/*
 * Copyright (c) 2025 CasparCG Contributors
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

// hap_gl_decode.cpp
// GL compressed texture upload + FBO render pass implementation.
// ---------------------------------------------------------------------------
#include "hap_gl_decode.h"

#include <common/log.h>

#include <stdexcept>
#include <string>

namespace caspar { namespace hap {

// ---------------------------------------------------------------------------
// GLSL shaders
// ---------------------------------------------------------------------------

static const char* VS_FULLSCREEN = R"GLSL(
#version 410 core
out vec2 v_uv;
void main()
{
    // Fullscreen triangle: 3 vertices, no VBO.
    //  id=0 -> (-1,-1)  uv=(0,0)
    //  id=1 -> ( 3,-1)  uv=(2,0)
    //  id=2 -> (-1, 3)  uv=(0,2)
    float x = float((gl_VertexID & 1) << 2) - 1.0;
    float y = float((gl_VertexID & 2) << 1) - 1.0;
    v_uv = vec2((x + 1.0) * 0.5, (y + 1.0) * 0.5);
    gl_Position = vec4(x, y, 0.0, 1.0);
}
)GLSL";

// Passthrough: HAP (DXT1) and HAP Alpha (DXT5).
// The texture unit decompresses DXT blocks automatically during sampling.
// CasparCG's mixer uses BGRA convention internally, so swizzle R<->B.
static const char* FS_PASSTHROUGH = R"GLSL(
#version 410 core
in vec2 v_uv;
out vec4 out_color;
uniform sampler2D u_texture;
void main()
{
    out_color = texture(u_texture, v_uv).bgra;
}
)GLSL";

// HAP Q: Scaled YCoCg-DXT5 -> RGB.
// In DXT5 for YCoCg: R = Co, G = Cg, B = scale_code, A = Y (luminance).
// Y is stored in the DXT5 alpha channel for highest precision.
static const char* FS_YCOCG = R"GLSL(
#version 410 core
in vec2 v_uv;
out vec4 out_color;
uniform sampler2D u_texture;
void main()
{
    vec4 s = texture(u_texture, v_uv);
    // DXT5 channel mapping: R=Co, G=Cg, B=scale_code, A=Y
    float scale = (s.b * 255.0 / 8.0) + 1.0;
    float Co = (s.r - (128.0 / 255.0)) / scale;
    float Cg = (s.g - (128.0 / 255.0)) / scale;
    float Y  = s.a;
    // CasparCG mixer uses BGRA convention: swap R and B.
    out_color = vec4(Y - Co - Cg, Y + Cg, Y + Co - Cg, 1.0);
}
)GLSL";

// HAP Q Alpha: Scaled YCoCg-DXT5 for color + separate DXT5 for alpha.
// Color texture: R=Co, G=Cg, B=scale_code, A=Y (same as HAP Q).
static const char* FS_YCOCG_ALPHA = R"GLSL(
#version 410 core
in vec2 v_uv;
out vec4 out_color;
uniform sampler2D u_color;
uniform sampler2D u_alpha;
void main()
{
    vec4 c = texture(u_color, v_uv);
    // DXT5 channel mapping: R=Co, G=Cg, B=scale_code, A=Y
    float scale = (c.b * 255.0 / 8.0) + 1.0;
    float Co = (c.r - (128.0 / 255.0)) / scale;
    float Cg = (c.g - (128.0 / 255.0)) / scale;
    float Y  = c.a;
    // CasparCG mixer uses BGRA convention: swap R and B.
    vec3 rgb = vec3(Y + Co - Cg, Y + Cg, Y - Co - Cg);
    float alpha = texture(u_alpha, v_uv).r;
    out_color = vec4(rgb.bgr, alpha);
}
)GLSL";

// ---------------------------------------------------------------------------
// Shader compilation helpers
// ---------------------------------------------------------------------------

static GLuint compile_shader(GLenum type, const char* source)
{
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &source, nullptr);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[1024];
        glGetShaderInfoLog(s, sizeof(log), nullptr, log);
        glDeleteShader(s);
        throw std::runtime_error(std::string("[hap_gl] Shader compile: ") + log);
    }
    return s;
}

static GLuint link_program(GLuint vs, GLuint fs)
{
    GLuint p = glCreateProgram();
    glAttachShader(p, vs);
    glAttachShader(p, fs);
    glLinkProgram(p);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[1024];
        glGetProgramInfoLog(p, sizeof(log), nullptr, log);
        glDeleteProgram(p);
        throw std::runtime_error(std::string("[hap_gl] Program link: ") + log);
    }
    return p;
}

// ---------------------------------------------------------------------------
// HapGLDecoder
// ---------------------------------------------------------------------------

HapGLDecoder::HapGLDecoder() {}

HapGLDecoder::~HapGLDecoder()
{
    if (program_passthrough_) glDeleteProgram(program_passthrough_);
    if (program_ycocg_)       glDeleteProgram(program_ycocg_);
    if (program_ycocg_alpha_) glDeleteProgram(program_ycocg_alpha_);
    if (fbo_)                 glDeleteFramebuffers(1, &fbo_);
    if (vao_)                 glDeleteVertexArrays(1, &vao_);
}

void HapGLDecoder::compile_shaders()
{
    if (initialized_) return;

    GLuint vs = compile_shader(GL_VERTEX_SHADER, VS_FULLSCREEN);

    {
        GLuint fs = compile_shader(GL_FRAGMENT_SHADER, FS_PASSTHROUGH);
        program_passthrough_ = link_program(vs, fs);
        glDeleteShader(fs);
        glUseProgram(program_passthrough_);
        glUniform1i(glGetUniformLocation(program_passthrough_, "u_texture"), 0);
    }
    {
        GLuint fs = compile_shader(GL_FRAGMENT_SHADER, FS_YCOCG);
        program_ycocg_ = link_program(vs, fs);
        glDeleteShader(fs);
        glUseProgram(program_ycocg_);
        glUniform1i(glGetUniformLocation(program_ycocg_, "u_texture"), 0);
    }
    {
        GLuint fs = compile_shader(GL_FRAGMENT_SHADER, FS_YCOCG_ALPHA);
        program_ycocg_alpha_ = link_program(vs, fs);
        glDeleteShader(fs);
        glUseProgram(program_ycocg_alpha_);
        glUniform1i(glGetUniformLocation(program_ycocg_alpha_, "u_color"), 0);
        glUniform1i(glGetUniformLocation(program_ycocg_alpha_, "u_alpha"), 1);
    }

    glDeleteShader(vs);
    glUseProgram(0);

    initialized_ = true;
}

void HapGLDecoder::create_fbo()
{
    if (fbo_) return;
    glGenFramebuffers(1, &fbo_);
    glGenVertexArrays(1, &vao_);
}

void HapGLDecoder::init_slot(HapDecodeSlot& slot, int width, int height,
                              HapTextureFormat primary_fmt, HapTextureFormat alpha_fmt)
{
    compile_shaders();
    create_fbo();

    slot.width  = width;
    slot.height = height;
    slot.compressed_format = hap_texture_to_gl(primary_fmt);

    // Create primary compressed texture
    glGenTextures(1, &slot.compressed_tex);
    glBindTexture(GL_TEXTURE_2D, slot.compressed_tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    // Allocate immutable storage for compressed data
    glTexStorage2D(GL_TEXTURE_2D, 1, slot.compressed_format, width, height);

    // Create alpha compressed texture if needed (HapM)
    if (alpha_fmt != primary_fmt || primary_fmt == HapTextureFormat::A_RGTC1) {
        slot.alpha_format = hap_texture_to_gl(alpha_fmt);
        glGenTextures(1, &slot.compressed_alpha);
        glBindTexture(GL_TEXTURE_2D, slot.compressed_alpha);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexStorage2D(GL_TEXTURE_2D, 1, slot.alpha_format, width, height);
    }

    glBindTexture(GL_TEXTURE_2D, 0);

    // Output texture will be set externally (from ogl_device_->create_texture)
    slot.initialized = true;
}

void HapGLDecoder::decode(HapDecodeSlot& slot,
                           HapVariant variant,
                           const uint8_t* texture_data, size_t texture_size,
                           const uint8_t* alpha_data, size_t alpha_size)
{
    if (!slot.initialized || !slot.output_tex) return;

    // Upload compressed data to the compressed texture(s).
    glBindTexture(GL_TEXTURE_2D, slot.compressed_tex);
    glCompressedTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                              slot.width, slot.height,
                              slot.compressed_format,
                              static_cast<GLsizei>(texture_size),
                              texture_data);

    if (variant == HapVariant::HapQAlpha && slot.compressed_alpha && alpha_data) {
        glBindTexture(GL_TEXTURE_2D, slot.compressed_alpha);
        glCompressedTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                                  slot.width, slot.height,
                                  slot.alpha_format,
                                  static_cast<GLsizei>(alpha_size),
                                  alpha_data);
    }
    glBindTexture(GL_TEXTURE_2D, 0);

    // Save current GL state
    GLint prev_fbo = 0, prev_viewport[4] = {};
    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &prev_fbo);
    glGetIntegerv(GL_VIEWPORT, prev_viewport);

    // Bind FBO with output texture as color attachment
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo_);
    glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D, slot.output_tex->id(), 0);
    glViewport(0, 0, slot.width, slot.height);

    // Select shader program
    GLuint prog = 0;
    switch (variant) {
    case HapVariant::Hap:
    case HapVariant::HapAlpha:
    case HapVariant::HapR:
        prog = program_passthrough_;
        break;
    case HapVariant::HapQ:
        prog = program_ycocg_;
        break;
    case HapVariant::HapQAlpha:
        prog = program_ycocg_alpha_;
        break;
    default:
        prog = program_passthrough_;
        break;
    }
    glUseProgram(prog);

    // Bind textures
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, slot.compressed_tex);

    if (variant == HapVariant::HapQAlpha && slot.compressed_alpha) {
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, slot.compressed_alpha);
    }

    // Draw fullscreen triangle (attributeless rendering)
    glBindVertexArray(vao_);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glBindVertexArray(0);

    // Restore GL state
    glUseProgram(0);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D, 0, 0);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, prev_fbo);
    glViewport(prev_viewport[0], prev_viewport[1], prev_viewport[2], prev_viewport[3]);

    // Ensure the render pass is complete before the mixer samples this texture.
    glFinish();
}

}} // namespace caspar::hap
