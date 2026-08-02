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

#pragma once

#include <cstddef>
#include <memory>
#include <string>

namespace caspar { namespace core {
class texture;
}} // namespace caspar::core

namespace caspar { namespace ffmpeg {

/**
 * Copies the Vulkan mixer's composited image into a CUDA device buffer, for NVENC.
 *
 * The Vulkan counterpart of cuda_gl_uploader. Same job, unrelated mechanism: the GL
 * one registers a GL texture with CUDA, this imports the image's exported memory with
 * cudaImportExternalMemory -- the same route decklink's cuda_vk_strategy already
 * takes to reach the same attachment.
 *
 * ── Why this did not exist ───────────────────────────────────────────────────
 * GPU-direct recording used to decline on the Vulkan mixer with "its composition
 * target is not allocated exportable, so CUDA cannot import it". That was not true:
 * device::create_attachment allocates with vk::ExportMemoryAllocateInfo precisely so
 * consumers can import it, and decklink has been doing so all along.
 *
 * ── Why the import is cached ─────────────────────────────────────────────────
 * cudaImportExternalMemory costs on the order of tens of milliseconds, which is more
 * than a frame. The mixer's attachment pool deliberately recycles a small set of
 * allocations to keep exported handles stable across frames (see image_kernel's
 * attachment pool), so keying the cache on the handle means a handful of entries cover
 * a channel indefinitely.
 */
class cuda_vk_uploader
{
  public:
    cuda_vk_uploader();
    ~cuda_vk_uploader();

    cuda_vk_uploader(const cuda_vk_uploader&)            = delete;
    cuda_vk_uploader& operator=(const cuda_vk_uploader&) = delete;

    /// Whether a usable CUDA device exists at all. Cheap; safe before construction.
    static bool available();

    /// The CUcontext the destination buffers belong to. Must be FFmpeg's own: device
    /// pointers are not valid across contexts, and writing an encoder frame through a
    /// pointer from another context takes the process down with an access violation
    /// rather than an error (see the note in cuda_gl_upload.h).
    void set_context(void* cu_context);

    /**
     * Imports `tex` -- which must be a Vulkan texture_wrapper carrying an exportable
     * image -- and copies it into `dst` with `dst_pitch` row bytes.
     *
     * Waits for the mixer's render to complete first. Unlike the GL path there is no
     * device thread to dispatch onto and no context to be current, so this can be
     * called from the consumer's own thread.
     *
     * Returns false and sets last_error() on any failure; the caller must fall back to
     * the host path.
     */
    bool copy_to_device(const std::shared_ptr<core::texture>& tex, void* dst, std::size_t dst_pitch);

    /// Releases every cached import. For a resolution change, where the attachment
    /// pool is rebuilt and the old handles become stale.
    void release();

    const char* last_error() const;

  private:
    struct impl;
    std::unique_ptr<impl> impl_;
};

}} // namespace caspar::ffmpeg
