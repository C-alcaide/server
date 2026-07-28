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

#pragma once

#include <cstdint>
#include <memory>

namespace caspar { namespace ofx {

/// CUDA render backend for the OFX GPU-render extension (Windows). When a plug-in advertises
/// CUDA render support and the host enables it, the images the plug-in fetches carry CUDA
/// device pointers (kOfxImagePropData). This backend allocates device buffers, uploads the
/// source frame, and reads the output back to CPU. Uses only the CUDA runtime host API (no
/// nvcc / .cu). Construction throws if no CUDA device is available or CUDA was not built.
class cuda_backend
{
  public:
    cuda_backend();
    ~cuda_backend();

    cuda_backend(const cuda_backend&)            = delete;
    cuda_backend& operator=(const cuda_backend&) = delete;

    /// Upload an 8-bit RGBA image into the reusable source device buffer; returns its device ptr.
    void* upload_source(const std::uint8_t* rgba, int width, int height);

    /// Ensure the reusable output device buffer exists (width*height*4); returns its device ptr.
    void* ensure_output(int width, int height);

    /// Copy the output device buffer back into an 8-bit RGBA host buffer.
    void readback_output(std::uint8_t* rgba, int width, int height);

    /// On-device source convert (no CPU passes): upload the RAW top-down source (BGRA or RGBA) once,
    /// then swap channels (BGRA->RGBA), mirror vertically (top-down -> OFX bottom-up), and
    /// premultiply straight alpha — all on the backend stream via NPP. Returns the bottom-up RGBA
    /// device buffer the plug-in reads. Throws on failure.
    void* convert_source(const std::uint8_t* raw,
                         int                 src_stride,
                         bool                is_bgra,
                         bool                straight_alpha,
                         int                 width,
                         int                 height);

    /// On-device output convert: mirror the plug-in's bottom-up RGBA output vertically back to
    /// top-down, on the backend stream via NPP. Returns a top-down RGBA device buffer ready for a
    /// single contiguous cudaMemcpy2DToArray. Throws on failure.
    void* mirror_output(void* out_dev, int width, int height);

    /// Block until all work enqueued on the backend stream has completed.
    void sync();

    /// Opaque CUDA stream handle for kOfxImageEffectPropCudaStream (may be null = default stream).
    void* stream() const;

  private:
    struct impl;
    std::unique_ptr<impl> impl_;
};

}} // namespace caspar::ofx
