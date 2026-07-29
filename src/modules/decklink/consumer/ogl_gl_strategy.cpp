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

#include "../StdAfx.h"

#include "ogl_gl_strategy.h"
#include "gpu_output_buffer_pool.h"

#include <accelerator/ogl/util/device.h>
#include <accelerator/ogl/util/texture.h>

#include <common/log.h>

#include <GL/glew.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace caspar { namespace decklink {

namespace {

// GL 4.3 compute shader: pack a subregion of an RGBA texture into V210
// (10-bit 4:2:2). Ported from vk_readback_v210.comp (descriptor sets/push
// constants -> binding=N + uniforms). One invocation packs one 6-pixel group.
const char* const k_v210_cs = R"GLSL(
#version 430
layout(local_size_x = 64, local_size_y = 1) in;
layout(binding = 0) uniform sampler2D src_tex;
layout(std430, binding = 1) buffer V210Output { uint data[]; } v210_out;
uniform int u_src_x;
uniform int u_src_y;
uniform int u_dst_w;
uniform int u_dst_h;
uniform int u_groups_per_row;
uniform int u_use_bt2020;
uniform int u_is_16bit;

void rgb_to_ycbcr_bt709(int R, int G, int B, out int Y, out int Cb, out int Cr) {
    Y  = 64  + ((222951 * R + 750098 * G + 75663 * B) >> 20);
    Cb = 512 + ((-100459 * R - 337802 * G + 438223 * B) >> 20);
    Cr = 512 + ((438223 * R - 398337 * G - 39908 * B) >> 20);
    Y = clamp(Y, 64, 940); Cb = clamp(Cb, 64, 960); Cr = clamp(Cr, 64, 960);
}
void rgb_to_ycbcr_bt2020(int R, int G, int B, out int Y, out int Cb, out int Cr) {
    Y  = 64  + ((275375 * R + 710743 * G + 62594 * B) >> 20);
    Cb = 512 + ((-146420 * R - 377856 * G + 524288 * B) >> 20);
    Cr = 512 + ((524288 * R - 482393 * G - 41857 * B) >> 20);
    Y = clamp(Y, 64, 940); Cb = clamp(Cb, 64, 960); Cr = clamp(Cr, 64, 960);
}
void main() {
    int group_x = int(gl_GlobalInvocationID.x);
    int row     = int(gl_GlobalInvocationID.y);
    if (row >= u_dst_h || group_x >= u_groups_per_row) return;
    int px_base = group_x * 6;
    int R[6], G[6], B[6];
    for (int i = 0; i < 6; ++i) {
        vec4 pixel = vec4(0.0);
        if ((px_base + i) < u_dst_w)
            pixel = texelFetch(src_tex, ivec2(u_src_x + px_base + i, u_src_y + row), 0);
        if (u_is_16bit != 0) {
            R[i] = int(pixel.r * 1023.0 + 0.5); G[i] = int(pixel.g * 1023.0 + 0.5); B[i] = int(pixel.b * 1023.0 + 0.5);
        } else {
            R[i] = int(pixel.b * 1023.0 + 0.5); G[i] = int(pixel.g * 1023.0 + 0.5); B[i] = int(pixel.r * 1023.0 + 0.5);
        }
    }
    int Y[6], Cb[6], Cr[6];
    for (int i = 0; i < 6; ++i) {
        if (u_use_bt2020 != 0) rgb_to_ycbcr_bt2020(R[i], G[i], B[i], Y[i], Cb[i], Cr[i]);
        else                   rgb_to_ycbcr_bt709(R[i], G[i], B[i], Y[i], Cb[i], Cr[i]);
    }
    int Cb0 = (Cb[0] + Cb[1] + 1) >> 1; int Cr0 = (Cr[0] + Cr[1] + 1) >> 1;
    int Cb1 = (Cb[2] + Cb[3] + 1) >> 1; int Cr1 = (Cr[2] + Cr[3] + 1) >> 1;
    int Cb2 = (Cb[4] + Cb[5] + 1) >> 1; int Cr2 = (Cr[4] + Cr[5] + 1) >> 1;
    uint w0 = uint(Cb0)  | (uint(Y[0]) << 10) | (uint(Cr0) << 20);
    uint w1 = uint(Y[1]) | (uint(Cb1)  << 10) | (uint(Y[2]) << 20);
    uint w2 = uint(Cr1)  | (uint(Y[3]) << 10) | (uint(Cb2) << 20);
    uint w3 = uint(Y[4]) | (uint(Cr2)  << 10) | (uint(Y[5]) << 20);
    uint base = uint(row) * uint(u_groups_per_row) * 4u + uint(group_x) * 4u;
    v210_out.data[base + 0u] = w0; v210_out.data[base + 1u] = w1;
    v210_out.data[base + 2u] = w2; v210_out.data[base + 3u] = w3;
}
)GLSL";

// GL 4.3 compute shader: extract a subregion of the RGBA texture as BGRA8.
const char* const k_bgra_cs = R"GLSL(
#version 430
layout(local_size_x = 16, local_size_y = 16) in;
layout(binding = 0) uniform sampler2D src_tex;
layout(std430, binding = 1) buffer BGRAOutput { uint data[]; } bgra_out;
uniform int u_src_x;
uniform int u_src_y;
uniform int u_dst_w;
uniform int u_dst_h;
void main() {
    int x = int(gl_GlobalInvocationID.x);
    int y = int(gl_GlobalInvocationID.y);
    if (x >= u_dst_w || y >= u_dst_h) return;
    vec4 pixel = texelFetch(src_tex, ivec2(u_src_x + x, u_src_y + y), 0);
    uint B = uint(pixel.b * 255.0 + 0.5);
    uint G = uint(pixel.g * 255.0 + 0.5);
    uint R = uint(pixel.r * 255.0 + 0.5);
    uint A = uint(pixel.a * 255.0 + 0.5);
    bgra_out.data[uint(y) * uint(u_dst_w) + uint(x)] = B | (G << 8u) | (R << 16u) | (A << 24u);
}
)GLSL";

GLuint compile_compute(const char* src)
{
    GLuint sh = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(sh, 1, &src, nullptr);
    glCompileShader(sh);
    GLint ok = 0;
    glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[2048] = {0};
        glGetShaderInfoLog(sh, sizeof(log) - 1, nullptr, log);
        CASPAR_LOG(error) << L"[ogl_gl_strategy] compute compile failed: " << log;
        glDeleteShader(sh);
        return 0;
    }
    GLuint prog = glCreateProgram();
    glAttachShader(prog, sh);
    glLinkProgram(prog);
    glDeleteShader(sh);
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[2048] = {0};
        glGetProgramInfoLog(prog, sizeof(log) - 1, nullptr, log);
        CASPAR_LOG(error) << L"[ogl_gl_strategy] compute link failed: " << log;
        glDeleteProgram(prog);
        return 0;
    }
    return prog;
}

// Independent C++ reference of the v210 shader math (BT.709/2020, 8-bit BGRA
// source), used by the one-shot parity self-test. Packs one row of the source
// (RGBA8 texels, mixer stores BGRA) into v210 exactly as k_v210_cs does.
void cpu_ref_v210_row(const std::uint8_t* rgba, int dst_w, int groups_per_row, bool bt2020, std::uint32_t* out)
{
    auto to10 = [](std::uint8_t v) { return int(double(v) / 255.0 * 1023.0 + 0.5); };
    for (int g = 0; g < groups_per_row; ++g) {
        int Y[6], Cb[6], Cr[6];
        for (int i = 0; i < 6; ++i) {
            int px = g * 6 + i;
            int R = 0, G = 0, B = 0;
            if (px < dst_w) {
                const std::uint8_t* p = rgba + px * 4; // p[0]=R,p[1]=G,p[2]=B texel channels
                R = to10(p[2]);                        // shader: R = pixel.b
                G = to10(p[1]);
                B = to10(p[0]);                        // shader: B = pixel.r
            }
            if (bt2020) {
                Y[i]  = 64 + ((275375 * R + 710743 * G + 62594 * B) >> 20);
                Cb[i] = 512 + ((-146420 * R - 377856 * G + 524288 * B) >> 20);
                Cr[i] = 512 + ((524288 * R - 482393 * G - 41857 * B) >> 20);
            } else {
                Y[i]  = 64 + ((222951 * R + 750098 * G + 75663 * B) >> 20);
                Cb[i] = 512 + ((-100459 * R - 337802 * G + 438223 * B) >> 20);
                Cr[i] = 512 + ((438223 * R - 398337 * G - 39908 * B) >> 20);
            }
            Y[i]  = std::min(940, std::max(64, Y[i]));
            Cb[i] = std::min(960, std::max(64, Cb[i]));
            Cr[i] = std::min(960, std::max(64, Cr[i]));
        }
        int  Cb0 = (Cb[0] + Cb[1] + 1) >> 1, Cr0 = (Cr[0] + Cr[1] + 1) >> 1;
        int  Cb1 = (Cb[2] + Cb[3] + 1) >> 1, Cr1 = (Cr[2] + Cr[3] + 1) >> 1;
        int  Cb2 = (Cb[4] + Cb[5] + 1) >> 1, Cr2 = (Cr[4] + Cr[5] + 1) >> 1;
        std::uint32_t* w = out + g * 4;
        w[0] = std::uint32_t(Cb0) | (std::uint32_t(Y[0]) << 10) | (std::uint32_t(Cr0) << 20);
        w[1] = std::uint32_t(Y[1]) | (std::uint32_t(Cb1) << 10) | (std::uint32_t(Y[2]) << 20);
        w[2] = std::uint32_t(Cr1) | (std::uint32_t(Y[3]) << 10) | (std::uint32_t(Cb2) << 20);
        w[3] = std::uint32_t(Y[4]) | (std::uint32_t(Cr2) << 10) | (std::uint32_t(Y[5]) << 20);
    }
}

} // namespace

struct ogl_gl_strategy::impl
{
    const bool                       is_hdr_;
    const bool                       use_bt2020_;
    const bool                       needs_v210_;
    spl::shared_ptr<format_strategy> fallback_;

    std::shared_ptr<gpu_output_buffer_pool> pool_;

    // GL objects live on the mixer's GL thread; tracked so we can free them there.
    std::weak_ptr<accelerator::ogl::device> gl_device_;
    GLuint                                  prog_    = 0;
    GLuint                                  ssbo_    = 0;
    std::size_t                             ssbo_sz_ = 0;
    bool                                    broken_  = false; // shader failed to build
    bool                                    parity_done_ = false; // one-shot self-test

    impl(bool is_hdr, bool use_bt2020, spl::shared_ptr<format_strategy> fallback, bool needs_v210)
        : is_hdr_(is_hdr)
        , use_bt2020_(use_bt2020)
        , needs_v210_(needs_v210)
        , fallback_(std::move(fallback))
    {
    }

    ~impl()
    {
        auto dev = gl_device_.lock();
        if (dev && (prog_ || ssbo_)) {
            GLuint prog = prog_, ssbo = ssbo_;
            dev->dispatch_sync([&] {
                if (prog)
                    glDeleteProgram(prog);
                if (ssbo)
                    glDeleteBuffers(1, &ssbo);
            });
        }
    }

    int row_bytes(int width) const
    {
        return needs_v210_ ? ((width + 47) / 48) * 128 : width * 4;
    }

    // Runs on the GL thread: (re)build program + SSBO, dispatch the pack, read back.
    bool pack_on_gl(int tex_id, int src_x, int src_y, int dst_w, int dst_h, bool is_16bit, void* out, std::size_t out_sz)
    {
        if (!prog_) {
            prog_ = compile_compute(needs_v210_ ? k_v210_cs : k_bgra_cs);
            if (!prog_) {
                broken_ = true;
                return false;
            }
        }
        if (ssbo_sz_ < out_sz) {
            if (ssbo_)
                glDeleteBuffers(1, &ssbo_);
            glGenBuffers(1, &ssbo_);
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_);
            glBufferData(GL_SHADER_STORAGE_BUFFER, out_sz, nullptr, GL_STREAM_READ);
            ssbo_sz_ = out_sz;
        }

        glUseProgram(prog_);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(tex_id));
        glUniform1i(glGetUniformLocation(prog_, "src_tex"), 0);
        glUniform1i(glGetUniformLocation(prog_, "u_src_x"), src_x);
        glUniform1i(glGetUniformLocation(prog_, "u_src_y"), src_y);
        glUniform1i(glGetUniformLocation(prog_, "u_dst_w"), dst_w);
        glUniform1i(glGetUniformLocation(prog_, "u_dst_h"), dst_h);

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssbo_);

        if (needs_v210_) {
            const int groups_per_row = row_bytes(dst_w) / 16;
            glUniform1i(glGetUniformLocation(prog_, "u_groups_per_row"), groups_per_row);
            glUniform1i(glGetUniformLocation(prog_, "u_use_bt2020"), use_bt2020_ ? 1 : 0);
            glUniform1i(glGetUniformLocation(prog_, "u_is_16bit"), is_16bit ? 1 : 0);
            glDispatchCompute((groups_per_row + 63) / 64, dst_h, 1);
        } else {
            glDispatchCompute((dst_w + 15) / 16, (dst_h + 15) / 16, 1);
        }

        glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, out_sz, out);

        // One-shot correctness gate: compare the GPU pack of row 0 against an
        // independent C++ reference of the same math on the actual texels. Fires
        // on the first non-black frame so it validates real colour, not black.
        if (needs_v210_ && !is_16bit && !parity_done_) {
            const auto*         g   = static_cast<const std::uint32_t*>(out);
            const std::uint32_t bw0 = 512u | (64u << 10) | (512u << 20); // black v210 word 0
            const std::uint32_t bw1 = 64u | (512u << 10) | (64u << 20);  // black v210 word 1
            const bool          non_black = out_sz >= 8 && (g[0] != bw0 || g[1] != bw1);
            if (non_black) {
                parity_done_     = true;
                const int groups = row_bytes(dst_w) / 16;
                glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(tex_id));
                GLint tw = 0, th = 0;
                glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &tw);
                glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &th);
                if (tw > 0 && th > 0 && src_y < th) {
                    std::vector<std::uint8_t> full(static_cast<std::size_t>(tw) * th * 4);
                    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, full.data());
                    const std::uint8_t*        src_row = full.data() + (static_cast<std::size_t>(src_y) * tw + src_x) * 4;
                    std::vector<std::uint32_t> ref(static_cast<std::size_t>(groups) * 4);
                    cpu_ref_v210_row(src_row, dst_w, groups, use_bt2020_, ref.data());
                    int mism = 0, first = -1;
                    for (std::size_t i = 0; i < ref.size(); ++i)
                        if (ref[i] != g[i]) {
                            ++mism;
                            if (first < 0)
                                first = static_cast<int>(i);
                        }
                    if (mism == 0)
                        CASPAR_LOG(info) << L"[ogl_gl_strategy] v210 parity self-test PASS (" << groups
                                         << L" groups @ row " << src_y << L").";
                    else
                        CASPAR_LOG(warning)
                            << L"[ogl_gl_strategy] v210 parity self-test: " << mism << L"/" << ref.size()
                            << L" words differ (first@" << first << L" gpu=" << g[first] << L" ref=" << ref[first]
                            << L").";
                }
            }
        }
        return true;
    }

    std::shared_ptr<void> convert(const core::video_format_desc& decklink_format_desc,
                                  const port_configuration&      config,
                                  const core::const_frame&       frame1)
    {
        const int         dst_w  = decklink_format_desc.width;
        const int         dst_h  = decklink_format_desc.height;
        const std::size_t out_sz = static_cast<std::size_t>(row_bytes(dst_w)) * dst_h;

        auto out = acquire_pinned_output(pool_, out_sz, 128);

        auto tex = frame1.texture();
        auto* gl_tex = dynamic_cast<accelerator::ogl::texture*>(tex.get());
        if (broken_ || !gl_tex) {
            std::memset(out.get(), 0, out_sz); // no OGL texture / broken shader -> black
            return out;
        }
        auto dev = gl_tex->get_device();
        if (!dev) {
            std::memset(out.get(), 0, out_sz);
            return out;
        }
        gl_device_ = dev;

        const bool is_16bit = !frame1.pixel_format_desc().planes.empty() &&
                              frame1.pixel_format_desc().planes[0].depth != common::bit_depth::bit8;
        const int  tex_id   = gl_tex->id();
        const int  src_x    = config.src_x;
        const int  src_y    = config.src_y;

        void*       dst = out.get();
        const bool  ok  = dev->dispatch_sync([&] { return pack_on_gl(tex_id, src_x, src_y, dst_w, dst_h, is_16bit, dst, out_sz); });
        if (!ok)
            std::memset(out.get(), 0, out_sz);
        return out;
    }
};

ogl_gl_strategy::ogl_gl_strategy(bool                             is_hdr,
                                 bool                             use_bt2020,
                                 spl::shared_ptr<format_strategy> fallback,
                                 bool                             needs_v210)
    : impl_(std::make_unique<impl>(is_hdr, use_bt2020, std::move(fallback), needs_v210))
{
    CASPAR_LOG(info) << L"[ogl_gl_strategy] GPU-direct OpenGL output: " << (needs_v210 ? L"v210" : L"bgra")
                     << (is_hdr ? L" hdr" : L"") << (use_bt2020 ? L" bt2020" : L"");
}

ogl_gl_strategy::~ogl_gl_strategy() = default;

BMDPixelFormat ogl_gl_strategy::get_pixel_format()
{
    return impl_->needs_v210_ ? bmdFormat10BitYUV : bmdFormat8BitBGRA;
}

int ogl_gl_strategy::get_row_bytes(int width) { return impl_->row_bytes(width); }

std::shared_ptr<void> ogl_gl_strategy::allocate_frame_data(const core::video_format_desc& format_desc)
{
    const std::size_t sz = static_cast<std::size_t>(impl_->row_bytes(format_desc.width)) * format_desc.height;
    auto              buf = acquire_pinned_output(impl_->pool_, sz, 128);
    std::memset(buf.get(), 0, sz);
    return buf;
}

std::shared_ptr<void> ogl_gl_strategy::convert_frame_for_port(const core::video_format_desc& channel_format_desc,
                                                              const core::video_format_desc& decklink_format_desc,
                                                              const port_configuration&      config,
                                                              const core::const_frame&       frame1,
                                                              const core::const_frame&       frame2,
                                                              BMDFieldDominance              field_dominance)
{
    return impl_->convert(decklink_format_desc, config, frame1);
}

spl::shared_ptr<format_strategy> try_create_ogl_gl_strategy(bool                             is_hdr,
                                                            bool                             use_bt2020,
                                                            spl::shared_ptr<format_strategy> fallback,
                                                            bool                             needs_v210)
{
    return spl::make_shared<ogl_gl_strategy>(is_hdr, use_bt2020, std::move(fallback), needs_v210);
}

}} // namespace caspar::decklink
