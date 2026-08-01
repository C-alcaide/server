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

// spout_gl_bridge.cpp
//
// The GLEW half of the Spout producer's zero-copy path. No Spout header may be
// included here -- see the note in spout_gl_bridge.h for why the two cannot
// meet in one translation unit.
// ---------------------------------------------------------------------------

#include "spout_gl_bridge.h"

#include <common/bit_depth.h>
#include <core/frame/frame.h>
#include <core/frame/frame_factory.h>

#include <accelerator/ogl/image/image_mixer.h>
#include <accelerator/ogl/util/device.h>
#include <accelerator/ogl/util/texture.h>

#ifdef ENABLE_VULKAN
#include <accelerator/vulkan/image/image_mixer.h>
#include <accelerator/vulkan/util/device.h>
#include <accelerator/vulkan/util/gl_export_bridge.h>
#include <accelerator/vulkan/util/texture.h>
#include <accelerator/vulkan/util/texture_wrapper.h>
#endif

#include <common/log.h>
#include <common/utf.h>

#include <windows.h>

namespace caspar { namespace spout {

ogl_device_handle get_mixer_ogl_device(core::frame_factory& factory)
{
    auto* ogl_mixer = dynamic_cast<accelerator::ogl::image_mixer*>(&factory);
    if (ogl_mixer == nullptr)
        return nullptr;

    auto device = ogl_mixer->get_ogl_device();
    if (!device)
        return nullptr;

    // Keeps the device alive for as long as the producer holds the handle, and
    // hands it back through a void alias so the header stays GLEW-free.
    return std::static_pointer_cast<void>(device);
}

bool create_shared_context(const ogl_device_handle& device, void*& out_hglrc, void*& out_hdc)
{
    out_hglrc = nullptr;
    out_hdc   = nullptr;
    if (!device)
        return false;

    auto dev = std::static_pointer_cast<accelerator::ogl::device>(device);

    dev->dispatch_sync([&]() {
        HGLRC main_hglrc = wglGetCurrentContext();
        HDC   hdc        = wglGetCurrentDC();
        if (!main_hglrc || !hdc)
            return;

        // wglShareLists requires the new context to be untouched, and both to
        // come from the same pixel format -- hence creating it here, on the
        // mixer's own DC, rather than on a window of our own.
        wglMakeCurrent(nullptr, nullptr);

        HGLRC shared = wglCreateContext(hdc);
        if (shared) {
            if (wglShareLists(main_hglrc, shared)) {
                out_hglrc = shared;
                out_hdc   = hdc;
            } else {
                CASPAR_LOG(warning) << L"[spout_producer] wglShareLists failed";
                wglDeleteContext(shared);
            }
        }

        // Whatever happened, give the mixer its context back before returning.
        wglMakeCurrent(hdc, main_hglrc);
    });

    return out_hglrc != nullptr;
}

void destroy_shared_context(void* hglrc)
{
    if (hglrc)
        wglDeleteContext(reinterpret_cast<HGLRC>(hglrc));
}

std::shared_ptr<core::texture>
create_mixer_texture(const ogl_device_handle& device, int width, int height)
{
    if (!device || width <= 0 || height <= 0)
        return nullptr;

    auto dev = std::static_pointer_cast<accelerator::ogl::device>(device);

    // Allocation has to happen on the device's own thread; shared lists are what
    // make the result reachable from the Spout receive thread afterwards.
    std::shared_ptr<accelerator::ogl::texture> tex;
    dev->dispatch_sync(
        [&]() { tex = dev->create_texture(width, height, 4, common::bit_depth::bit8); });

    return tex;
}

unsigned int texture_gl_id(const std::shared_ptr<core::texture>& tex)
{
    auto ogl_tex = std::dynamic_pointer_cast<accelerator::ogl::texture>(tex);
    return ogl_tex ? static_cast<unsigned int>(ogl_tex->id()) : 0u;
}

// ─── Vulkan mixer ───────────────────────────────────────────────────────────

vk_device_handle get_mixer_vk_device(core::frame_factory& factory)
{
#ifdef ENABLE_VULKAN
    auto* vk_mixer = dynamic_cast<accelerator::vulkan::image_mixer*>(&factory);
    if (vk_mixer == nullptr)
        return nullptr;

    auto device = vk_mixer->get_vk_device();
    if (!device)
        return nullptr;

    return std::static_pointer_cast<void>(device);
#else
    (void)factory;
    return nullptr;
#endif
}

std::vector<vk_shared_slot>
create_vk_shared_slots(const vk_device_handle& device, int width, int height, int count)
{
    std::vector<vk_shared_slot> slots;
#ifdef ENABLE_VULKAN
    if (!device || width <= 0 || height <= 0 || count <= 0)
        return slots;

    auto dev = std::static_pointer_cast<accelerator::vulkan::device>(device);

    try {
        for (int i = 0; i < count; ++i) {
            // Allocated on the Vulkan device (create_exportable_texture
            // dispatches to its thread itself), then imported here, on the
            // caller's GL context -- which is the receive thread's, and the only
            // one allowed to use or free the GL name.
            auto vk_tex = dev->create_exportable_texture(width, height, 4, common::bit_depth::bit8);
            auto import = std::make_shared<accelerator::vulkan::gl_shared_texture>(vk_tex);

            vk_shared_slot slot;
            slot.gl_id         = import->gl_id();
            slot.frame_texture = std::static_pointer_cast<core::texture>(
                std::make_shared<accelerator::vulkan::VkReadableTextureWrapper>(vk_tex, dev));
            slot.import = std::static_pointer_cast<void>(import);
            slots.push_back(std::move(slot));
        }
    } catch (const std::exception& e) {
        // Partial slots are useless and their GL names belong to this context,
        // so drop them here rather than handing back half a ring.
        slots.clear();
        CASPAR_LOG(warning) << L"[spout_producer] cannot share a Vulkan texture with OpenGL ("
                            << u16(e.what()) << L"); using the readback path.";
    }
#else
    (void)device;
    (void)width;
    (void)height;
    (void)count;
#endif
    return slots;
}

}} // namespace caspar::spout
