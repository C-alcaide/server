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

#include "ofx_cuda_render.h"

#include <common/log.h>
#include <stdexcept>
#include <vector>
#include <string>

#ifdef CASPAR_OFX_CUDA
#include <cuda_runtime.h>
#include <nppi_data_exchange_and_initialization.h>
#include <nppi_geometry_transforms.h>
#include <nppi_arithmetic_and_logical_operations.h>
#include <npp.h>
#endif

namespace caspar { namespace ofx {

#ifdef CASPAR_OFX_CUDA

namespace {
void cuda_verify(cudaError_t e, const char* what)
{
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("[ofx] ") + what + ": " + cudaGetErrorString(e));
}
void npp_verify(NppStatus s, const char* what)
{
    if (s != NPP_SUCCESS)
        throw std::runtime_error(std::string("[ofx] ") + what + " (NPP status " + std::to_string(static_cast<int>(s)) + ")");
}
} // namespace

struct cuda_backend::impl
{
    void*        src     = nullptr; // legacy upload_source buffer (bottom-up RGBA)
    void*        out     = nullptr; // plug-in output buffer (device)
    void*        raw     = nullptr; // convert_source: uploaded raw source (top-down)
    void*        swap    = nullptr; // convert_source: channel-swapped intermediate
    void*        conv    = nullptr; // convert_source: final bottom-up RGBA source
    void*        flip    = nullptr; // mirror_output: top-down RGBA ready for the VK array
    std::size_t  src_sz  = 0;
    std::size_t  out_sz  = 0;
    std::size_t  raw_sz  = 0;
    std::size_t  swap_sz = 0;
    std::size_t  conv_sz = 0;
    void*        premul  = nullptr; // premultiply scratch; see convert_source step 4
    std::size_t  premul_sz = 0;
    std::size_t  flip_sz = 0;
    cudaStream_t stream_ = nullptr;
    NppStreamContext npp_ctx_{};

    impl()
    {
        int n = 0;
        if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0)
            throw std::runtime_error("[ofx] no CUDA device available");
        cudaSetDevice(0);
        cuda_verify(cudaStreamCreate(&stream_), "cudaStreamCreate");
        // Fill the NPP stream context for the current device, then point it at our stream so all
        // NPP calls below run on it (instead of the legacy global default stream).
        npp_verify(nppGetStreamContext(&npp_ctx_), "nppGetStreamContext");
        npp_ctx_.hStream = stream_;
    }

    ~impl()
    {
        for (void* p : {src, out, raw, swap, conv, flip})
            if (p)
                cudaFree(p);
        if (stream_)
            cudaStreamDestroy(stream_);
    }

    void ensure(void*& buf, std::size_t& cur, std::size_t need)
    {
        if (cur < need) {
            if (buf)
                cudaFree(buf);
            if (cudaMalloc(&buf, need) != cudaSuccess)
                throw std::runtime_error("[ofx] cudaMalloc failed");
            cur = need;
        }
    }
};

cuda_backend::cuda_backend()
    : impl_(std::make_unique<impl>())
{
}

cuda_backend::~cuda_backend() = default;

void* cuda_backend::upload_source(const std::uint8_t* rgba, int width, int height)
{
    const std::size_t sz = static_cast<std::size_t>(width) * height * 4;
    impl_->ensure(impl_->src, impl_->src_sz, sz);
    if (cudaMemcpy(impl_->src, rgba, sz, cudaMemcpyHostToDevice) != cudaSuccess)
        throw std::runtime_error("[ofx] cudaMemcpy H2D failed");
    return impl_->src;
}

void* cuda_backend::ensure_output(int width, int height)
{
    const std::size_t sz = static_cast<std::size_t>(width) * height * 4;
    impl_->ensure(impl_->out, impl_->out_sz, sz);
    return impl_->out;
}

void cuda_backend::readback_output(std::uint8_t* rgba, int width, int height)
{
    const std::size_t sz = static_cast<std::size_t>(width) * height * 4;
    if (cudaMemcpy(rgba, impl_->out, sz, cudaMemcpyDeviceToHost) != cudaSuccess)
        throw std::runtime_error("[ofx] cudaMemcpy D2H failed");
}

void* cuda_backend::convert_source(const std::uint8_t* raw,
                                   int                 src_stride,
                                   bool                is_bgra,
                                   bool                straight_alpha,
                                   int                 width,
                                   int                 height)
{
    const int         step = width * 4;                            // tightly-packed device row bytes
    const std::size_t sz   = static_cast<std::size_t>(step) * height;
    const NppiSize    roi{width, height};

    impl_->ensure(impl_->raw, impl_->raw_sz, sz);
    impl_->ensure(impl_->conv, impl_->conv_sz, sz);

    // 1) Upload the raw top-down source, tightening any input row padding to `step`.
    cuda_verify(cudaMemcpy2DAsync(impl_->raw,
                                  step,
                                  raw,
                                  src_stride,
                                  step,
                                  height,
                                  cudaMemcpyHostToDevice,
                                  impl_->stream_),
                "cudaMemcpy2DAsync source H2D");

    // 2) Channel swap BGRA->RGBA when needed (dst[i] = src[order[i]]).
    const std::uint8_t* pre = static_cast<const std::uint8_t*>(impl_->raw);
    if (is_bgra) {
        impl_->ensure(impl_->swap, impl_->swap_sz, sz);
        const int order[4] = {2, 1, 0, 3};
        npp_verify(nppiSwapChannels_8u_C4R_Ctx(static_cast<const Npp8u*>(impl_->raw),
                                               step,
                                               static_cast<Npp8u*>(impl_->swap),
                                               step,
                                               roi,
                                               order,
                                               impl_->npp_ctx_),
                   "nppiSwapChannels BGRA->RGBA");
        pre = static_cast<const std::uint8_t*>(impl_->swap);
    }

    // 3) Vertical mirror (top-down -> OFX bottom-up) into the final source buffer.
    npp_verify(nppiMirror_8u_C4R_Ctx(reinterpret_cast<const Npp8u*>(pre),
                                     step,
                                     static_cast<Npp8u*>(impl_->conv),
                                     step,
                                     roi,
                                     NPP_HORIZONTAL_AXIS,
                                     impl_->npp_ctx_),
               "nppiMirror source vertical");


    // 4) Premultiply straight alpha.
    //
    // Not the in-place variant. nppiAlphaPremul_8u_AC4IR is documented to leave the
    // alpha channel alone, and measurably does not: an opaque source goes in as
    // (0,0,255,255) and comes out (0,0,255,0), RGB correct and alpha destroyed. That
    // is what made every OFX CUDA plug-in that reads its source emit a fully
    // transparent frame -- invisible on a fill+key output, and correct-looking in any
    // test that only checks RGB.
    if (straight_alpha) {
        impl_->ensure(impl_->premul, impl_->premul_sz, sz);
        npp_verify(nppiAlphaPremul_8u_AC4R_Ctx(static_cast<const Npp8u*>(impl_->conv),
                                               step,
                                               static_cast<Npp8u*>(impl_->premul),
                                               step,
                                               roi,
                                               impl_->npp_ctx_),
                   "nppiAlphaPremul source");
        // Put the alpha channel back. Measured, not inferred: NPP writes zero into the
        // destination's alpha whichever variant is used -- in-place or not, and whatever
        // the destination held beforehand. Both were tried. Copying channel 3 across
        // afterwards is the only form that survives the test.
        npp_verify(nppiCopy_8u_C4CR_Ctx(static_cast<const Npp8u*>(impl_->conv) + 3,
                                        step,
                                        static_cast<Npp8u*>(impl_->premul) + 3,
                                        step,
                                        roi,
                                        impl_->npp_ctx_),
                   "restore alpha after premultiply");
        std::swap(impl_->conv, impl_->premul);
        std::swap(impl_->conv_sz, impl_->premul_sz);
    }


    return impl_->conv;
}

void* cuda_backend::mirror_output(void* out_dev, int width, int height)
{
    const int         step = width * 4;
    const std::size_t sz   = static_cast<std::size_t>(step) * height;
    const NppiSize    roi{width, height};

    impl_->ensure(impl_->flip, impl_->flip_sz, sz);
    npp_verify(nppiMirror_8u_C4R_Ctx(static_cast<const Npp8u*>(out_dev),
                                     step,
                                     static_cast<Npp8u*>(impl_->flip),
                                     step,
                                     roi,
                                     NPP_HORIZONTAL_AXIS,
                                     impl_->npp_ctx_),
               "nppiMirror output vertical");
    return impl_->flip;
}

void cuda_backend::sync() { cuda_verify(cudaStreamSynchronize(impl_->stream_), "cudaStreamSynchronize"); }

void* cuda_backend::stream() const { return impl_->stream_; }

#else // CASPAR_OFX_CUDA not defined — CUDA was not built; backend is unavailable.

struct cuda_backend::impl
{};

cuda_backend::cuda_backend() { throw std::runtime_error("[ofx] CUDA render backend not built"); }
cuda_backend::~cuda_backend()                                            = default;
void* cuda_backend::upload_source(const std::uint8_t*, int, int) { return nullptr; }
void* cuda_backend::ensure_output(int, int) { return nullptr; }
void  cuda_backend::readback_output(std::uint8_t*, int, int) {}
void* cuda_backend::convert_source(const std::uint8_t*, int, bool, bool, int, int) { return nullptr; }
void* cuda_backend::mirror_output(void*, int, int) { return nullptr; }
void  cuda_backend::sync() {}
void* cuda_backend::stream() const { return nullptr; }

#endif

}} // namespace caspar::ofx

