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

// spout_gl_bridge.h
//
// The few things the Spout producer needs from the OpenGL mixer's device,
// declared without dragging in GLEW.
//
// GLEW and the Spout SDK cannot share a translation unit. GLEW macro-renames
// the GL entry points (glBindFramebufferEXT becomes __glewBindFramebufferEXT)
// and declares them dllimport; SpoutGLextensions.h declares the same names as
// plain extern, so including both gives C4273 "inconsistent DLL linkage" on
// every framebuffer entry point. That is a real conflict over which loader owns
// those pointers, not a warning worth suppressing.
//
// So the GLEW side lives in spout_gl_bridge.cpp, which never sees a Spout
// header, and the Spout side includes only this.
// ---------------------------------------------------------------------------
#pragma once

#include <memory>
#include <vector>

namespace caspar {

namespace core {
class frame_factory;
class texture;
} // namespace core

namespace spout {

/// Opaque handle to the mixer's OpenGL device. Null when the mixer is not
/// OpenGL, which is the signal to take the readback path.
using ogl_device_handle = std::shared_ptr<void>;

/// Returns the mixer's OpenGL device, or null if this is not an OpenGL mixer.
ogl_device_handle get_mixer_ogl_device(core::frame_factory& factory);

/// Create a GL context that shares lists with the mixer's, for use on the Spout
/// receive thread. Returns false if sharing could not be established.
///
/// The context is created on the mixer's own device context, from the mixer's
/// own thread -- the same thing the HAP producer does. Creating one on a fresh
/// hidden window instead makes wglShareLists fail, because the two contexts then
/// have unrelated pixel formats.
///
/// `out_hglrc` and `out_hdc` are HGLRC and HDC. Caller makes the context current
/// on its own thread and deletes it with destroy_shared_context().
bool create_shared_context(const ogl_device_handle& device, void*& out_hglrc, void*& out_hdc);

/// Delete a context from create_shared_context(). Safe with nulls.
void destroy_shared_context(void* hglrc);

/// Allocate a texture on the mixer's device, on the device's own thread.
/// Returned as core::texture so this header stays free of accelerator types.
std::shared_ptr<core::texture>
create_mixer_texture(const ogl_device_handle& device, int width, int height);

/// The GL name of a texture returned by create_mixer_texture, for handing to
/// Spout's ReceiveTexture.
unsigned int texture_gl_id(const std::shared_ptr<core::texture>& tex);

// ─── Vulkan mixer ───────────────────────────────────────────────────────────
//
// The Vulkan mixer cannot share GL lists, so the trick the OpenGL path uses does
// not apply. Instead Vulkan allocates the image and exports its memory, GL
// imports it, and Spout receives into the GL texture that aliases it -- so the
// pixels land in the mixer's own memory without a host round trip either way.
// See accelerator/vulkan/util/gl_export_bridge.h for what that mechanism can and
// cannot do.

/// Opaque handle to the mixer's Vulkan device. Null when the mixer is not
/// Vulkan, which is the signal that this path does not apply.
using vk_device_handle = std::shared_ptr<void>;

/// Returns the mixer's Vulkan device, or null if this is not a Vulkan mixer.
vk_device_handle get_mixer_vk_device(core::frame_factory& factory);

/// One receive slot: a Vulkan image the mixer samples and the GL texture name
/// aliasing the same memory for Spout to receive into.
struct vk_shared_slot
{
    /// Hand this to the mixer as the frame's texture.
    ///
    /// Label the frame `pixel_format::rgba`, not `bgra`: Spout's ReceiveTexture
    /// leaves RGBA-ordered bytes in the destination, and the Vulkan mixer's
    /// bgra case applies a .bgra swizzle that would exchange red and blue. The
    /// OpenGL path labels the same pixels bgra because its own upload path
    /// reorders on the way in and the two have to agree; this one has no upload.
    std::shared_ptr<core::texture> frame_texture;

    /// Give this to Spout's ReceiveTexture.
    unsigned int gl_id = 0;

    /// Owns the GL import. Must be released with the same GL context current
    /// that created it, so keep slots on the receive thread and let them die
    /// there -- before the context does.
    std::shared_ptr<void> import;
};

/// Allocate `count` slots of `width` x `height` on the Vulkan device and import
/// each into the *current* GL context. Returns an empty vector if the driver
/// refuses, which is the caller's signal to use the readback path.
std::vector<vk_shared_slot>
create_vk_shared_slots(const vk_device_handle& device, int width, int height, int count);

}} // namespace caspar::spout
