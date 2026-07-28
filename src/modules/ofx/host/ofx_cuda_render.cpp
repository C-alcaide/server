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

#include <stdexcept>

#ifdef CASPAR_OFX_CUDA
#include <cuda_runtime.h>
#endif

namespace caspar { namespace ofx {

#ifdef CASPAR_OFX_CUDA

struct cuda_backend::impl
{
    void*        src     = nullptr;
    void*        out     = nullptr;
    std::size_t  src_sz  = 0;
    std::size_t  out_sz  = 0;
    cudaStream_t stream_ = nullptr;

    impl()
    {
        int n = 0;
        if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0)
            throw std::runtime_error("[ofx] no CUDA device available");
        cudaSetDevice(0);
    }

    ~impl()
    {
        if (src)
            cudaFree(src);
        if (out)
            cudaFree(out);
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

void* cuda_backend::stream() const { return impl_->stream_; }

#else // CASPAR_OFX_CUDA not defined — CUDA was not built; backend is unavailable.

struct cuda_backend::impl
{};

cuda_backend::cuda_backend() { throw std::runtime_error("[ofx] CUDA render backend not built"); }
cuda_backend::~cuda_backend()                                            = default;
void* cuda_backend::upload_source(const std::uint8_t*, int, int) { return nullptr; }
void* cuda_backend::ensure_output(int, int) { return nullptr; }
void  cuda_backend::readback_output(std::uint8_t*, int, int) {}
void* cuda_backend::stream() const { return nullptr; }

#endif

}} // namespace caspar::ofx
