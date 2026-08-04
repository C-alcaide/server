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
#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// NVIDIA GPUDirect for Video (DVP) — optional Tier-2 GPU->sysmem readback tail.
// Only available on CUDA/DVP builds; the DVP headers pull in GL/gl.h, which GLEW
// has already superseded above (its include guards make gl.h a no-op).
#ifdef DECKLINK_CUDA_DVP_ENABLED
#include <dvpapi_gl.h>
#include <malloc.h>
#include <unordered_map>
#endif

namespace caspar { namespace decklink {

namespace {

// GL 4.3 compute shader: pack a subregion of an RGBA texture into V210
// (10-bit 4:2:2). Mirrors the CPU v210 packer (v210_strategies.cpp) exactly:
// the legal-range RGB->YCbCr matrix is computed on the host and uploaded in
// u_mat[9], the 8-bit input is scaled <<2 (as R8<<2), chroma is co-sited nearest
// (even pixel, no averaging), and the luma/chroma offsets match. One invocation
// packs one 6-pixel group.
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
uniform int u_is_16bit;
uniform int u_first_line;   // first output row this pass writes (0, or 1 for field 2)
uniform int u_line_step;    // 1 = progressive, 2 = interlaced field
uniform int u_key_only;     // 1 = output the alpha (as a grey key)
uniform int u_mat[9];       // legal-range fixed-point RGB->YCbCr (host-computed, matches CPU)

int clamp10(int v) { return clamp(v, 0, 1023); }

void main() {
    int group_x = int(gl_GlobalInvocationID.x);
    int row     = u_first_line + int(gl_GlobalInvocationID.y) * u_line_step;
    if (row >= u_dst_h || group_x >= u_groups_per_row) return;
    int px_base = group_x * 6;
    int R[6], G[6], B[6];
    for (int i = 0; i < 6; ++i) {
        vec4 p = vec4(0.0);
        if ((px_base + i) < u_dst_w)
            p = texelFetch(src_tex, ivec2(u_src_x + px_base + i, u_src_y + row), 0);
        if (u_key_only != 0) {
            int a = (u_is_16bit != 0) ? int(p.a * 1023.0 + 0.5) : (int(p.a * 255.0 + 0.5) << 2);
            R[i] = a; G[i] = a; B[i] = a;
        } else if (u_is_16bit != 0) {
            R[i] = int(p.r * 1023.0 + 0.5); G[i] = int(p.g * 1023.0 + 0.5); B[i] = int(p.b * 1023.0 + 0.5);
        } else {
            // NO UN-SWIZZLE. texelFetch reads the TEXTURE's components, and the mixer's
            // render target holds true RGB: the image shader ends with
            // `fragColor = col.bgra`, which converts its internally BGR-ordered working
            // colour back to RGB order on the way out. So p.r is red here.
            //
            // This used to read p.b into R and p.r into B, "un-swizzling to real RGB" as
            // though the texture were BGRA. That is the HOST byte order — `texture.cpp`
            // transfers with GL_BGRA, so host buffers really are B,G,R,A, which is why
            // the BGRA memcpy consumer path is correct — but it is not what a texel
            // fetch sees. The un-swizzle therefore CREATED a red/blue exchange.
            //
            // Measured on an SDI loopback with a saturated reference: a pure red patch
            // came back pure blue, 6.44 dB against the reference, and 46.94 dB once the
            // capture was R/B-swapped back — i.e. an exact channel exchange with the
            // picture otherwise pristine. The 16-bit branch above never un-swizzled and
            // was correct all along.
            R[i] = int(p.r * 255.0 + 0.5) << 2; G[i] = int(p.g * 255.0 + 0.5) << 2; B[i] = int(p.b * 255.0 + 0.5) << 2;
        }
    }
    int Y[6];
    for (int i = 0; i < 6; ++i)
        Y[i] = clamp10(((64 << 20) + u_mat[0]*R[i] + u_mat[1]*G[i] + u_mat[2]*B[i]) >> 20);
    // Co-sited nearest chroma from the even pixel of each pair (matches CPU).
    int Cb0 = clamp10(((1025 << 19) + u_mat[3]*R[0] + u_mat[4]*G[0] + u_mat[5]*B[0]) >> 20);
    int Cr0 = clamp10(((1025 << 19) + u_mat[6]*R[0] + u_mat[7]*G[0] + u_mat[8]*B[0]) >> 20);
    int Cb1 = clamp10(((1025 << 19) + u_mat[3]*R[2] + u_mat[4]*G[2] + u_mat[5]*B[2]) >> 20);
    int Cr1 = clamp10(((1025 << 19) + u_mat[6]*R[2] + u_mat[7]*G[2] + u_mat[8]*B[2]) >> 20);
    int Cb2 = clamp10(((1025 << 19) + u_mat[3]*R[4] + u_mat[4]*G[4] + u_mat[5]*B[4]) >> 20);
    int Cr2 = clamp10(((1025 << 19) + u_mat[6]*R[4] + u_mat[7]*G[4] + u_mat[8]*B[4]) >> 20);
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
uniform int u_first_line;
uniform int u_line_step;
uniform int u_key_only;
void main() {
    int x   = int(gl_GlobalInvocationID.x);
    int row = u_first_line + int(gl_GlobalInvocationID.y) * u_line_step;
    if (x >= u_dst_w || row >= u_dst_h) return;
    vec4 pixel = texelFetch(src_tex, ivec2(u_src_x + x, u_src_y + row), 0);
    uint B, G, R, A;
    if (u_key_only != 0) {
        uint a = uint(pixel.a * 255.0 + 0.5); B = a; G = a; R = a; A = a;
    } else {
        B = uint(pixel.b * 255.0 + 0.5); G = uint(pixel.g * 255.0 + 0.5);
        R = uint(pixel.r * 255.0 + 0.5); A = uint(pixel.a * 255.0 + 0.5);
    }
    bgra_out.data[uint(row) * uint(u_dst_w) + uint(x)] = B | (G << 8u) | (R << 16u) | (A << 24u);
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

// Legal-range RGB->YCbCr fixed-point matrix, computed exactly like the CPU
// packer's create_int_matrix (v210_strategies.cpp): luma x 876*1024/1023, chroma
// x 896*1024/1023, then x1024. Uploaded to the shader as u_mat so the GPU and CPU
// use identical coefficients (single source of truth for colour parity).
std::array<std::int32_t, 9> legal_range_v210_matrix(bool bt2020)
{
    static const double bt709[9] = {0.212639005871510,
                                    0.715168678767756,
                                    0.072192315360734,
                                    -0.114592177555732,
                                    -0.385407822444268,
                                    0.5,
                                    0.5,
                                    -0.454155517037873,
                                    -0.045844482962127};
    static const double bt2020c[9] = {0.262700212011267,
                                      0.677998071518871,
                                      0.059301716469862,
                                      -0.139630430187157,
                                      -0.360369569812843,
                                      0.5,
                                      0.5,
                                      -0.459784529009814,
                                      -0.040215470990186};
    const double* c                = bt2020 ? bt2020c : bt709;
    const double  luma_range       = 876.0 * 1024.0 / 1023.0;
    const double  chroma_range     = 896.0 * 1024.0 / 1023.0;
    std::array<std::int32_t, 9> m{};
    for (int i = 0; i < 9; ++i)
        m[i] = static_cast<std::int32_t>(std::lround(c[i] * (i < 3 ? luma_range : chroma_range) * 1024.0));
    return m;
}

// Independent C++ reference of the v210 shader math, used by the one-shot parity
// self-test: legal-range matrix `m`, 8-bit input scaled <<2, co-sited nearest chroma
// (even pixel).
//
// The row is downloaded with `glGetTexImage(..., GL_RGBA, ...)`, so p[0] is the
// TEXTURE's red component — and the mixer's render target holds true RGB (its image
// shader ends `fragColor = col.bgra`). So p[0]=R, p[1]=G, p[2]=B.
//
// This previously claimed "mixer memory BGRA: p[0]=B, p[1]=G, p[2]=R" and read p[2]
// into R, mirroring the same mistake in the shader. Because reference and shader shared
// the assumption they agreed, the parity self-test PASSED while both were wrong — a
// reference that copies the implementation's assumption cannot audit it. Fixing only
// one of the two would have turned this into a loud parity failure instead, which is
// the behaviour to want.
void cpu_ref_v210_row(const std::uint8_t*                rgba,
                      int                                dst_w,
                      int                                groups_per_row,
                      const std::array<std::int32_t, 9>& m,
                      std::uint32_t*                     out)
{
    auto clamp10 = [](int v) { return v < 0 ? 0 : (v > 1023 ? 1023 : v); };
    for (int g = 0; g < groups_per_row; ++g) {
        int R[6], G[6], B[6];
        for (int i = 0; i < 6; ++i) {
            int px = g * 6 + i;
            if (px < dst_w) {
                const std::uint8_t* p = rgba + px * 4; // GL_RGBA download of an RGB target
                R[i] = int(p[0]) << 2;
                G[i] = int(p[1]) << 2;
                B[i] = int(p[2]) << 2;
            } else {
                R[i] = G[i] = B[i] = 0;
            }
        }
        int Y[6];
        for (int i = 0; i < 6; ++i)
            Y[i] = clamp10(((64 << 20) + m[0] * R[i] + m[1] * G[i] + m[2] * B[i]) >> 20);
        auto cb = [&](int i) { return clamp10(((1025 << 19) + m[3] * R[i] + m[4] * G[i] + m[5] * B[i]) >> 20); };
        auto cr = [&](int i) { return clamp10(((1025 << 19) + m[6] * R[i] + m[7] * G[i] + m[8] * B[i]) >> 20); };
        int Cb0 = cb(0), Cr0 = cr(0), Cb1 = cb(2), Cr1 = cr(2), Cb2 = cb(4), Cr2 = cr(4);
        std::uint32_t* w = out + g * 4;
        w[0] = std::uint32_t(Cb0) | (std::uint32_t(Y[0]) << 10) | (std::uint32_t(Cr0) << 20);
        w[1] = std::uint32_t(Y[1]) | (std::uint32_t(Cb1) << 10) | (std::uint32_t(Y[2]) << 20);
        w[2] = std::uint32_t(Cr1) | (std::uint32_t(Y[3]) << 10) | (std::uint32_t(Cb2) << 20);
        w[3] = std::uint32_t(Y[4]) | (std::uint32_t(Cr2) << 10) | (std::uint32_t(Y[5]) << 20);
    }
}

// Independent C++ reference of the bgra shader (RGBA texel -> BGRA8 uint32).
void cpu_ref_bgra_row(const std::uint8_t* rgba, int dst_w, std::uint32_t* out)
{
    for (int x = 0; x < dst_w; ++x) {
        const std::uint8_t* p = rgba + x * 4; // p[0]=R,p[1]=G,p[2]=B,p[3]=A
        out[x] = std::uint32_t(p[2]) | (std::uint32_t(p[1]) << 8) | (std::uint32_t(p[0]) << 16) |
                 (std::uint32_t(p[3]) << 24);
    }
}

} // namespace

#ifdef DECKLINK_CUDA_DVP_ENABLED
namespace {

// Process-level refcount for the DVP GL context. dvpInitGLContext /
// dvpCloseGLContext act on the current (mixer) GL context, so with several DVP
// ports sharing one GL thread they must be initialised once and closed once —
// otherwise the first port's teardown would close the context out from under
// the others (R7). All acquire/release calls run on the mixer GL thread, so the
// mutex only guards the refcount and cached alignment constants.
struct dvp_gl_ctx
{
    std::mutex    mtx;
    int           refs         = 0;
    bool          ok           = false;
    std::uint32_t buf_align    = 0;
    std::uint32_t stride_align = 0;
    std::uint32_t sem_align    = 0;
    std::uint32_t sem_size     = 0;
};

dvp_gl_ctx& dvp_ctx()
{
    static dvp_gl_ctx c;
    return c;
}

// GL thread only (context current). Returns the DVP alignment constants.
bool dvp_ctx_acquire(std::uint32_t& buf_align,
                     std::uint32_t& stride_align,
                     std::uint32_t& sem_align,
                     std::uint32_t& sem_size)
{
    auto&                       c = dvp_ctx();
    std::lock_guard<std::mutex> lk(c.mtx);
    if (c.refs == 0) {
        c.ok = false;
        if (dvpInitGLContext(DVP_DEVICE_FLAGS_SHARE_APP_CONTEXT) == DVP_STATUS_OK) {
            std::uint32_t po = 0, ps = 0;
            if (dvpGetRequiredConstantsGLCtx(&c.buf_align, &c.stride_align, &c.sem_align, &c.sem_size, &po, &ps) ==
                DVP_STATUS_OK)
                c.ok = true;
            else
                dvpCloseGLContext();
        }
    }
    if (!c.ok)
        return false;
    ++c.refs;
    buf_align    = c.buf_align;
    stride_align = c.stride_align;
    sem_align    = c.sem_align;
    sem_size     = c.sem_size;
    return true;
}

// GL thread only (context current). Closes the context on the last release.
void dvp_ctx_release()
{
    auto&                       c = dvp_ctx();
    std::lock_guard<std::mutex> lk(c.mtx);
    if (c.refs > 0 && --c.refs == 0 && c.ok) {
        dvpCloseGLContext();
        c.ok = false;
    }
}

} // namespace
#endif // DECKLINK_CUDA_DVP_ENABLED

struct ogl_gl_strategy::impl
{
    const bool                       is_hdr_;
    const bool                       use_bt2020_;
    const bool                       needs_v210_;
    const bool                       use_dvp_;
    spl::shared_ptr<format_strategy> fallback_;

    std::shared_ptr<gpu_output_buffer_pool> pool_;

    // GL objects live on the mixer's GL thread; tracked so we can free them there.
    std::weak_ptr<accelerator::ogl::device> gl_device_;
    GLuint                                  prog_    = 0;
    GLuint                                  ssbo_    = 0;
    std::size_t                             ssbo_sz_ = 0;
    bool                                    broken_  = false; // shader failed to build
    bool                                    parity_done_ = false; // one-shot self-test
    std::array<std::int32_t, 9>             v210_mat_;    // legal-range matrix (matches CPU)

    impl(bool is_hdr, bool use_bt2020, spl::shared_ptr<format_strategy> fallback, bool needs_v210, bool use_dvp)
        : is_hdr_(is_hdr)
        , use_bt2020_(use_bt2020)
        , needs_v210_(needs_v210)
        , use_dvp_(use_dvp)
        , fallback_(std::move(fallback))
        , v210_mat_(legal_range_v210_matrix(use_bt2020))
    {
    }

    ~impl()
    {
        auto dev = gl_device_.lock();
        bool has_dvp = false;
#ifdef DECKLINK_CUDA_DVP_ENABLED
        has_dvp = dvp_inited_;
#endif
        if (dev && (prog_ || ssbo_ || has_dvp)) {
            GLuint prog = prog_, ssbo = ssbo_;
            dev->dispatch_sync([&] {
#ifdef DECKLINK_CUDA_DVP_ENABLED
                dvp_teardown();
#endif
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

#ifdef DECKLINK_CUDA_DVP_ENABLED
    // NVIDIA GPUDirect-for-Video (DVP) readback state. Replaces glGetBufferSubData
    // with a hardware-synchronised GPU-buffer -> page-locked-sysmem DMA. All calls
    // run on the mixer GL thread (context current) as SHARE_APP_CONTEXT requires.
    struct dvp_sync
    {
        volatile std::uint32_t* sem     = nullptr;
        std::uint32_t           acquire = 0;
        std::uint32_t           release = 0;
        DVPSyncObjectHandle     handle  = 0;
    };

    bool            dvp_inited_    = false; // dvpInitGLContext done
    bool            dvp_ctx_held_  = false; // holds a dvp_ctx() refcount
    bool            dvp_failed_    = false; // gave up -> use glGetBufferSubData
    bool            dvp_logged_ok_ = false;
    std::uint32_t   dvp_buf_align_ = 0, dvp_stride_align_ = 0, dvp_sem_align_ = 0, dvp_sem_size_ = 0;
    GLuint          dvp_gpu_ssbo_  = 0; // ssbo id currently registered
    DVPBufferHandle dvp_gpu_buf_   = 0; // registered ssbo handle
    dvp_sync        dvp_ext_sync_;
    dvp_sync        dvp_gpu_sync_;
    std::unordered_map<void*, DVPBufferHandle> dvp_sysmem_; // pinned out ptr -> handle

    bool dvp_init_sync(dvp_sync& s)
    {
        s.sem = static_cast<volatile std::uint32_t*>(_aligned_malloc(dvp_sem_size_, dvp_sem_align_));
        if (!s.sem)
            return false;
        s.sem[0]  = 0;
        s.acquire = 0;
        s.release = 0;
        DVPSyncObjectDesc d{};
        d.externalClientWaitFunc = nullptr;
        d.sem                    = const_cast<std::uint32_t*>(s.sem);
        d.flags                  = 0;
        if (dvpImportSyncObject(&d, &s.handle) != DVP_STATUS_OK) {
            // The semaphore was ours until the import took it; on failure it is
            // still ours to free.
            _aligned_free(const_cast<std::uint32_t*>(s.sem));
            s.sem    = nullptr;
            s.handle = 0;
            return false;
        }
        return true;
    }

    void dvp_free_sync(dvp_sync& s)
    {
        if (s.handle) {
            dvpFreeSyncObject(s.handle);
            s.handle = 0;
        }
        if (s.sem) {
            _aligned_free(const_cast<std::uint32_t*>(s.sem));
            s.sem = nullptr;
        }
    }

    bool dvp_init_ctx()
    {
        if (!dvp_ctx_acquire(dvp_buf_align_, dvp_stride_align_, dvp_sem_align_, dvp_sem_size_))
            return false;
        dvp_ctx_held_ = true;
        if (dvp_init_sync(dvp_ext_sync_) && dvp_init_sync(dvp_gpu_sync_))
            return true;

        // Partial construction: dvp_inited_ stays false, so dvp_teardown() will
        // early-return and never clean any of this up. Unwind here instead --
        // otherwise the first sync object, its semaphore, and (worse) the
        // process-wide dvp_gl_ctx refcount leak, so dvpCloseGLContext() is never
        // called for the lifetime of the process.
        dvp_free_sync(dvp_ext_sync_);
        dvp_free_sync(dvp_gpu_sync_);
        dvp_ctx_release();
        dvp_ctx_held_ = false;
        return false;
    }

    DVPBufferHandle dvp_register_sysmem(void* p, std::size_t bytes)
    {
        auto it = dvp_sysmem_.find(p);
        if (it != dvp_sysmem_.end())
            return it->second;
        DVPSysmemBufferDesc desc{};
        desc.width   = static_cast<std::uint32_t>(bytes);
        desc.height  = 1;
        desc.stride  = static_cast<std::uint32_t>(bytes);
        desc.size    = static_cast<std::uint32_t>(bytes);
        desc.format  = DVP_BUFFER;
        desc.type    = DVP_UNSIGNED_BYTE;
        desc.bufAddr = p;
        DVPBufferHandle h = 0;
        if (dvpCreateBuffer(&desc, &h) != DVP_STATUS_OK)
            return 0;
        if (dvpBindToGLCtx(h) != DVP_STATUS_OK) {
            dvpDestroyBuffer(h);
            return 0;
        }
        dvp_sysmem_[p] = h;
        return h;
    }

    // Called before the compute dispatch: ensure DVP is initialised, the SSBO and
    // the sysmem output are registered, and GL waits for any prior DVP read of the
    // SSBO to finish. Returns true when DVP is ready to service this frame.
    bool dvp_prepare(GLuint ssbo, void* out, std::size_t bytes)
    {
        if (dvp_failed_)
            return false;
        if (!dvp_inited_) {
            if (!dvp_init_ctx()) {
                dvp_failed_ = true;
                CASPAR_LOG(warning) << L"[ogl_gl_strategy] DVP-GL init failed; using glGetBufferSubData.";
                return false;
            }
            dvp_inited_ = true;
        }
        if (dvp_gpu_ssbo_ != ssbo) {
            if (dvp_gpu_buf_) {
                dvpFreeBuffer(dvp_gpu_buf_);
                dvp_gpu_buf_ = 0;
            }
            if (dvpCreateGPUBufferGL(ssbo, &dvp_gpu_buf_) != DVP_STATUS_OK) {
                dvp_failed_ = true;
                return false;
            }
            dvp_gpu_ssbo_ = ssbo;
        }
        if (!dvp_register_sysmem(out, bytes)) {
            dvp_failed_ = true;
            return false;
        }
        // GL must not overwrite the SSBO until the previous DVP read completed.
        dvpMapBufferWaitAPI(dvp_gpu_buf_);
        return true;
    }

    // Called after the compute dispatch + glMemoryBarrier: DMA the packed SSBO to
    // the page-locked sysmem output with HW sync. Returns true on success.
    bool dvp_finish(void* out, std::size_t bytes)
    {
        auto it = dvp_sysmem_.find(out);
        if (it == dvp_sysmem_.end()) {
            dvp_failed_ = true;
            return false;
        }
        // Signal that GL has finished writing the SSBO, then have DVP wait on it.
        if (dvpMapBufferEndAPI(dvp_gpu_buf_) != DVP_STATUS_OK) {
            dvp_failed_ = true;
            return false;
        }
        dvp_gpu_sync_.release++;
        dvpBegin();
        dvpMapBufferWaitDVP(dvp_gpu_buf_);
        DVPStatus st = dvpMemcpy(dvp_gpu_buf_,
                                 dvp_ext_sync_.handle,
                                 dvp_ext_sync_.acquire,
                                 DVP_TIMEOUT_IGNORED,
                                 it->second,
                                 dvp_gpu_sync_.handle,
                                 dvp_gpu_sync_.release,
                                 0,
                                 0,
                                 static_cast<std::uint32_t>(bytes));
        dvpMapBufferEndDVP(dvp_gpu_buf_);
        dvpEnd();
        if (st != DVP_STATUS_OK) {
            dvp_failed_ = true;
            return false;
        }
        // Block until the copy has landed in sysmem before DeckLink DMAs the frame.
        dvpBegin();
        dvpSyncObjClientWaitComplete(dvp_gpu_sync_.handle, DVP_TIMEOUT_IGNORED);
        dvpEnd();
        if (!dvp_logged_ok_) {
            CASPAR_LOG(info) << L"[ogl_gl_strategy] DVP-GL readback active (GPUDirect for Video).";
            dvp_logged_ok_ = true;
        }
        return true;
    }

    void dvp_teardown()
    {
        if (!dvp_inited_)
            return;
        for (auto& kv : dvp_sysmem_) {
            dvpUnbindFromGLCtx(kv.second);
            dvpDestroyBuffer(kv.second);
        }
        dvp_sysmem_.clear();
        if (dvp_gpu_buf_) {
            dvpFreeBuffer(dvp_gpu_buf_);
            dvp_gpu_buf_  = 0;
            dvp_gpu_ssbo_ = 0;
        }
        dvp_free_sync(dvp_ext_sync_);
        dvp_free_sync(dvp_gpu_sync_);
        if (dvp_ctx_held_) {
            dvp_ctx_release();
            dvp_ctx_held_ = false;
        }
        dvp_inited_ = false;
    }
#endif // DECKLINK_CUDA_DVP_ENABLED

    struct field_pass
    {
        int tex_id;
        int first_line; // first output row this field writes (0 or 1)
        int line_step;  // 1 progressive, 2 interlaced
    };

    // Runs on the GL thread: (re)build program + SSBO, dispatch each field pass, read back.
    bool pack_on_gl(const std::vector<field_pass>& passes,
                    int                            src_x,
                    int                            src_y,
                    int                            dst_w,
                    int                            dst_h,
                    bool                           is_16bit,
                    bool                           key_only,
                    void*                          out,
                    std::size_t                    out_sz)
    {
        if (!prog_) {
            prog_ = compile_compute(needs_v210_ ? k_v210_cs : k_bgra_cs);
            if (!prog_) {
                broken_ = true;
                return false;
            }
        }
        if (ssbo_sz_ < out_sz) {
            if (ssbo_) {
#ifdef DECKLINK_CUDA_DVP_ENABLED
                // DVP still holds a registration for this buffer object. It has to
                // be released *before* the GL object goes away: dvp_prepare() only
                // notices the id changed on the next frame, by which point
                // dvpFreeBuffer() would be operating on a deleted buffer.
                if (dvp_gpu_buf_ && dvp_gpu_ssbo_ == ssbo_) {
                    dvpFreeBuffer(dvp_gpu_buf_);
                    dvp_gpu_buf_  = 0;
                    dvp_gpu_ssbo_ = 0;
                }
#endif
                glDeleteBuffers(1, &ssbo_);
            }
            glGenBuffers(1, &ssbo_);
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_);
            glBufferData(GL_SHADER_STORAGE_BUFFER, out_sz, nullptr, GL_STREAM_READ);
            ssbo_sz_ = out_sz;
        }

        glUseProgram(prog_);
        glActiveTexture(GL_TEXTURE0);
        glUniform1i(glGetUniformLocation(prog_, "src_tex"), 0);
        glUniform1i(glGetUniformLocation(prog_, "u_src_x"), src_x);
        glUniform1i(glGetUniformLocation(prog_, "u_src_y"), src_y);
        glUniform1i(glGetUniformLocation(prog_, "u_dst_w"), dst_w);
        glUniform1i(glGetUniformLocation(prog_, "u_dst_h"), dst_h);
        glUniform1i(glGetUniformLocation(prog_, "u_key_only"), key_only ? 1 : 0);
        const int groups_per_row = needs_v210_ ? row_bytes(dst_w) / 16 : 0;
        if (needs_v210_) {
            glUniform1i(glGetUniformLocation(prog_, "u_groups_per_row"), groups_per_row);
            glUniform1i(glGetUniformLocation(prog_, "u_is_16bit"), is_16bit ? 1 : 0);
            glUniform1iv(glGetUniformLocation(prog_, "u_mat"), 9, v210_mat_.data());
        }
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssbo_);

        bool dvp_ready = false;
#ifdef DECKLINK_CUDA_DVP_ENABLED
        if (use_dvp_)
            dvp_ready = dvp_prepare(ssbo_, out, out_sz);
#endif

        for (const auto& p : passes) {
            glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(p.tex_id));
            glUniform1i(glGetUniformLocation(prog_, "u_first_line"), p.first_line);
            glUniform1i(glGetUniformLocation(prog_, "u_line_step"), p.line_step);
            const int rows = (dst_h - p.first_line + p.line_step - 1) / p.line_step;
            if (rows <= 0)
                continue;
            if (needs_v210_)
                glDispatchCompute((groups_per_row + 63) / 64, rows, 1);
            else
                glDispatchCompute((dst_w + 15) / 16, (rows + 15) / 16, 1);
        }

        glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);

        bool read_done = false;
#ifdef DECKLINK_CUDA_DVP_ENABLED
        if (dvp_ready)
            read_done = dvp_finish(out, out_sz);
#endif
        if (!read_done) {
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_);
            glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, out_sz, out);
        }

        // One-shot correctness gate: compare the GPU pack of row 0 against an
        // independent C++ reference of the same math on the actual texels. Fires
        // on the first non-black progressive frame (single pass, not key-only).
        if (needs_v210_ && !is_16bit && !key_only && passes.size() == 1 && !parity_done_) {
            const auto*         g   = static_cast<const std::uint32_t*>(out);
            const std::uint32_t bw0 = 512u | (64u << 10) | (512u << 20); // black v210 word 0
            const std::uint32_t bw1 = 64u | (512u << 10) | (64u << 20);  // black v210 word 1
            const bool          non_black = out_sz >= 8 && (g[0] != bw0 || g[1] != bw1);
            if (non_black) {
                parity_done_     = true;
                const int groups = groups_per_row;
                glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(passes[0].tex_id));
                GLint tw = 0, th = 0;
                glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &tw);
                glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &th);
                if (tw > 0 && th > 0 && src_y < th) {
                    std::vector<std::uint8_t> full(static_cast<std::size_t>(tw) * th * 4);
                    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, full.data());
                    const std::uint8_t*        src_row = full.data() + (static_cast<std::size_t>(src_y) * tw + src_x) * 4;
                    std::vector<std::uint32_t> ref(static_cast<std::size_t>(groups) * 4);
                    cpu_ref_v210_row(src_row, dst_w, groups, v210_mat_, ref.data());
                    // Compare only fully in-bounds groups: past (tw - src_x) the GPU
                    // reads out-of-texture texels (0) while the CPU ref runs off the
                    // row, so cropped ports would otherwise false-mismatch at the edge.
                    const int limit_px     = std::max(0, std::min(dst_w, tw - src_x));
                    const int valid_groups = std::min(groups, limit_px / 6);
                    const std::size_t cmp  = static_cast<std::size_t>(valid_groups) * 4;
                    int mism = 0, first = -1;
                    for (std::size_t i = 0; i < cmp; ++i)
                        if (ref[i] != g[i]) {
                            ++mism;
                            if (first < 0)
                                first = static_cast<int>(i);
                        }
                    if (mism == 0)
                        CASPAR_LOG(info) << L"[ogl_gl_strategy] v210 parity self-test PASS (" << valid_groups
                                         << L" groups @ row " << src_y << L").";
                    else
                        CASPAR_LOG(warning)
                            << L"[ogl_gl_strategy] v210 parity self-test: " << mism << L"/" << cmp
                            << L" words differ (first@" << first << L" gpu=" << g[first] << L" ref=" << ref[first]
                            << L").";
                }
            }
        }

        // BGRA-path parity self-test (first non-black progressive frame).
        if (!needs_v210_ && !key_only && passes.size() == 1 && !parity_done_) {
            const auto* g         = static_cast<const std::uint32_t*>(out);
            const bool  non_black = out_sz >= 8 && g[0] != 0u && g[0] != 0xFF000000u;
            if (non_black) {
                parity_done_ = true;
                glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(passes[0].tex_id));
                GLint tw = 0, th = 0;
                glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &tw);
                glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &th);
                if (tw > 0 && th > 0 && src_y < th) {
                    std::vector<std::uint8_t> full(static_cast<std::size_t>(tw) * th * 4);
                    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, full.data());
                    const std::uint8_t*        src_row = full.data() + (static_cast<std::size_t>(src_y) * tw + src_x) * 4;
                    std::vector<std::uint32_t> ref(static_cast<std::size_t>(dst_w));
                    cpu_ref_bgra_row(src_row, dst_w, ref.data());
                    // Compare only in-bounds pixels (see v210 note above).
                    const int limit_px = std::max(0, std::min(dst_w, tw - src_x));
                    int mism = 0, first = -1;
                    for (int i = 0; i < limit_px; ++i)
                        if (ref[i] != g[i]) {
                            ++mism;
                            if (first < 0)
                                first = i;
                        }
                    if (mism == 0)
                        CASPAR_LOG(info) << L"[ogl_gl_strategy] bgra parity self-test PASS (" << limit_px
                                         << L" px @ row " << src_y << L").";
                    else
                        CASPAR_LOG(warning) << L"[ogl_gl_strategy] bgra parity self-test: " << mism << L"/" << limit_px
                                            << L" px differ (first@" << first << L").";
                }
            }
        }
        return true;
    }

    std::shared_ptr<void> convert(const core::video_format_desc& decklink_format_desc,
                                  const port_configuration&      config,
                                  const core::const_frame&       frame1,
                                  const core::const_frame&       frame2,
                                  BMDFieldDominance              field_dominance)
    {
        const int         dst_w  = decklink_format_desc.width;
        const int         dst_h  = decklink_format_desc.height;
        const std::size_t out_sz = static_cast<std::size_t>(row_bytes(dst_w)) * dst_h;

        auto out = acquire_pinned_output(pool_, out_sz, 128);

        auto  tex1    = frame1.texture();
        auto* gl_tex1 = dynamic_cast<accelerator::ogl::texture*>(tex1.get());
        if (broken_ || !gl_tex1) {
            std::memset(out.get(), 0, out_sz); // no OGL texture / broken shader -> black
            return out;
        }
        auto dev = gl_tex1->get_device();
        if (!dev) {
            std::memset(out.get(), 0, out_sz);
            return out;
        }
        gl_device_ = dev;

        const bool is_16bit = !frame1.pixel_format_desc().planes.empty() &&
                              frame1.pixel_format_desc().planes[0].depth != common::bit_depth::bit8;
        const bool key_only = config.key_only;
        const int  src_x    = config.src_x;
        const int  src_y    = config.src_y;

        // Build the field passes. Progressive: one full pass. Interlaced: weave the
        // two fields (frame1/frame2) into alternate output rows, matching the CPU
        // strategy's field-dominance line assignment.
        std::vector<field_pass> passes;
        accelerator::ogl::texture* gl_tex2 = nullptr;
        if (field_dominance != bmdProgressiveFrame && frame2)
            gl_tex2 = dynamic_cast<accelerator::ogl::texture*>(frame2.texture().get());
        if (gl_tex2) {
            const int fl1 = (field_dominance == bmdUpperFieldFirst) ? 0 : 1;
            const int fl2 = (field_dominance != bmdUpperFieldFirst) ? 0 : 1;
            passes.push_back({gl_tex1->id(), fl1, 2});
            passes.push_back({gl_tex2->id(), fl2, 2});
        } else {
            passes.push_back({gl_tex1->id(), 0, 1});
        }

        void*      dst = out.get();
        const bool ok  = dev->dispatch_sync(
            [&] { return pack_on_gl(passes, src_x, src_y, dst_w, dst_h, is_16bit, key_only, dst, out_sz); });
        if (!ok)
            std::memset(out.get(), 0, out_sz);
        return out;
    }
};

ogl_gl_strategy::ogl_gl_strategy(bool                             is_hdr,
                                 bool                             use_bt2020,
                                 spl::shared_ptr<format_strategy> fallback,
                                 bool                             needs_v210,
                                 bool                             use_dvp)
    : impl_(std::make_unique<impl>(is_hdr, use_bt2020, std::move(fallback), needs_v210, use_dvp))
{
    CASPAR_LOG(info) << L"[ogl_gl_strategy] GPU-direct OpenGL output: " << (needs_v210 ? L"v210" : L"bgra")
                     << (is_hdr ? L" hdr" : L"") << (use_bt2020 ? L" bt2020" : L"")
                     << (use_dvp ? L" dvp" : L"");
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
    return impl_->convert(decklink_format_desc, config, frame1, frame2, field_dominance);
}

spl::shared_ptr<format_strategy> try_create_ogl_gl_strategy(bool                             is_hdr,
                                                            bool                             use_bt2020,
                                                            spl::shared_ptr<format_strategy> fallback,
                                                            bool                             needs_v210,
                                                            bool                             use_dvp)
{
    return spl::make_shared<ogl_gl_strategy>(is_hdr, use_bt2020, std::move(fallback), needs_v210, use_dvp);
}

}} // namespace caspar::decklink
