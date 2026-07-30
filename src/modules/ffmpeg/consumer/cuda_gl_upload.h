/*
 * Copyright (c) 2026 CasparCG Contributors
 *
 * This file is part of CasparCG (www.casparcg.com)
 * and is licensed under the GNU General Public License v3.
 */

#pragma once

#include <memory>

namespace caspar { namespace accelerator { namespace ogl {
class texture;
class device;
}}} // namespace caspar::accelerator::ogl

namespace caspar { namespace ffmpeg {

/// Copies the mixer's composited texture straight into CUDA device memory, so a
/// recording never travels to host memory and back.
///
/// Without this the frame makes a full round trip: the channel reads it back
/// (measured 2.7 GB/s, so 2.95 ms at 1080p and 11.50 ms at 4K) and NVENC then
/// uploads the very same pixels again. Both legs disappear when the encoder is
/// handed a frame that is already on the device.
///
/// No colour conversion happens here and none is needed: NVENC accepts RGB
/// input directly, so the copy is byte-for-byte out of the GL texture.
class cuda_gl_uploader
{
  public:
    cuda_gl_uploader();
    ~cuda_gl_uploader();

    cuda_gl_uploader(const cuda_gl_uploader&)            = delete;
    cuda_gl_uploader& operator=(const cuda_gl_uploader&) = delete;

    /// True when this build has CUDA and a device is usable.
    static bool available();

    /// Run all CUDA work in this context (a CUcontext, as void*).
    ///
    /// Required, not optional: FFmpeg's CUDA device context creates its own
    /// context, and device pointers are not valid across contexts. Asking FFmpeg
    /// to use the primary context instead does not work here either -- the
    /// DeckLink DVP and CUDA ProRes modules activate it first, and FFmpeg then
    /// fails with "Primary context already active with incompatible flags".
    void set_context(void* cu_context);

    /// Copies `tex` into the linear device allocation at `dst`.
    ///
    /// MUST be called on a thread that has the mixer's GL context current --
    /// CUDA's GL interop registers against the calling thread's context. Call it
    /// through accelerator::ogl::device::dispatch_sync rather than by making a
    /// second context current: a private wglShareLists context is what the ProRes
    /// consumer does, and it is the pattern that made OGL GPU affinity
    /// impossible to add (see GPU_AFFINITY_PLAN.md).
    bool copy_to_device(accelerator::ogl::texture& tex, void* dst, size_t dst_pitch);

    /// Unregisters everything. Must run on the mixer's GL thread and while the
    /// CUDA context is still alive, so the uploader remembers the device the
    /// first time it is used and does this itself at destruction. Getting it
    /// wrong crashes at teardown rather than in use, which is a poor trade.
    void release();

    /// Human-readable reason the last call failed, for logging.
    const char* last_error() const;

  private:
    struct impl;
    std::unique_ptr<impl> impl_;
};

}} // namespace caspar::ffmpeg
