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

#include <memory>
#include <string>

namespace caspar { namespace accelerator { namespace vulkan {

class texture;

/**
 * Vulkan → OpenGL memory sharing: a GL texture whose storage *is* a Vulkan
 * image's memory, so a GL producer can render straight into what the Vulkan
 * mixer will sample, with no copy in either direction.
 *
 * This is the second half of interop item 2a (see docs/plans/GPU_INTEROP_PLAN.md).
 * The mechanism is `GL_EXT_memory_object` plus `VK_KHR_external_memory_win32`
 * (or `_fd` on Linux): Vulkan allocates and exports, GL imports and aliases.
 *
 * ── What is proved, and what is not ──────────────────────────────────────────
 * On the reference GPU (RTX A4000, driver 582.53) a round trip through this
 * mechanism is byte-exact: GL renders a per-pixel pattern into an imported
 * 1920x1080 RGBA8 image and Vulkan reads back 0 of 2073600 pixels wrong, with
 * GL row 0 landing on Vulkan row 0. Two constraints came out of that probe and
 * are load-bearing here:
 *
 *  - The Vulkan image must use `eOptimal` tiling. With `eLinear`, GL rejects
 *    the storage as "memory object too small" -- the two APIs disagree about
 *    row pitch. `create_exportable_texture` uses eOptimal, so this holds, but
 *    it rules this path out for the LINEAR workaround previz applies on Pascal.
 *  - The image needs `eColorAttachment` in its usage flags to be rendered
 *    into rather than only sampled. `create_exportable_texture` has it.
 *
 * ── Synchronisation ──────────────────────────────────────────────────────────
 * Nothing here orders GL's writes against Vulkan's reads. The caller must do
 * that, and `glFinish()` after rendering is enough to start with: it measured
 * 0.053 ms per frame against the ≈9.2 ms per layer per frame the host round
 * trip it replaces costs. `GL_EXT_semaphore` is the better answer eventually,
 * but the measurement says it is not where the time goes.
 *
 * ── Threading ────────────────────────────────────────────────────────────────
 * Construct, use and destroy on one thread, with the same GL context current
 * throughout. The Vulkan texture is kept alive by this object.
 */
class gl_shared_texture
{
  public:
    /// `vk_tex` must have come from `device::create_exportable_texture`, which is
    /// what gives it exportable memory and eColorAttachment. Requires a current
    /// GL context. Throws if the extension is missing or the import is rejected;
    /// callers are expected to fall back to the host path in that case.
    explicit gl_shared_texture(std::shared_ptr<texture> vk_tex);
    ~gl_shared_texture();

    gl_shared_texture(const gl_shared_texture&)            = delete;
    gl_shared_texture& operator=(const gl_shared_texture&) = delete;

    /// The GL texture name aliasing the Vulkan image. Never 0 on a live object.
    unsigned int gl_id() const { return gl_texture_; }

    const std::shared_ptr<texture>& vk_texture() const { return vk_tex_; }

    int width() const;
    int height() const;

  private:
    std::shared_ptr<texture> vk_tex_;
    unsigned int             gl_memory_object_ = 0;
    unsigned int             gl_texture_       = 0;
};

/// True when the current GL context exposes `GL_EXT_memory_object` and the
/// platform's import entry point. Cheap after the first call; safe to ask on
/// each context, but the answer is cached per process.
bool gl_import_supported();

/// Imports `handle` (a Vulkan external-memory handle for `size` bytes) as GL
/// memory and creates a `width` x `height` texture on it with `internal_format`
/// (a GL sized format such as GL_RGBA8). `out_memory_object` and `out_texture`
/// receive the new names.
///
/// Returns an empty string on success, or a description of what the driver
/// rejected. Requires a current GL context. Exposed separately so
/// `previz_texture_bridge`, which allocates its own VkImage rather than going
/// through `create_exportable_texture`, shares this code instead of repeating it.
std::string gl_import_memory_as_texture(void*         handle,
                                        unsigned long long size,
                                        unsigned int  internal_format,
                                        int           width,
                                        int           height,
                                        unsigned int& out_memory_object,
                                        unsigned int& out_texture);

/// Deletes a texture and memory object made by `gl_import_memory_as_texture`,
/// zeroing both names. Requires the same current GL context. Ignores zeros, so
/// it is safe on a half-built pair.
void gl_release_imported_texture(unsigned int& memory_object, unsigned int& tex);

}}} // namespace caspar::accelerator::vulkan
