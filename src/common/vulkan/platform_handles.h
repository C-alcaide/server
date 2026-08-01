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

// Canonical platform-agnostic handle type definitions for Vulkan external
// memory/semaphore interop.  Both the mixer (accelerator::vulkan) and the
// output consumer (vulkan_output) include this via their own namespace-
// forwarding headers so that constants never diverge.
//
// Windows: opaque Win32 HANDLEs via VK_KHR_external_memory_win32
// Linux:   POSIX file descriptors via VK_KHR_external_memory_fd

#ifdef _WIN32
#include <windows.h>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_win32.h>
#else
#include <vulkan/vulkan.h>
#include <unistd.h>
#endif

namespace caspar { namespace vulkan_common { namespace platform {

// ─── Vulkan external memory handle type ─────────────────────────────────────
#ifdef _WIN32
inline constexpr VkExternalMemoryHandleTypeFlagBits kExternalMemoryHandleType =
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
inline constexpr VkExternalSemaphoreHandleTypeFlagBits kExternalSemaphoreHandleType =
    VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
inline constexpr const char* kExtMemExtName = VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME;
inline constexpr const char* kExtSemExtName = VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME;
#else
inline constexpr VkExternalMemoryHandleTypeFlagBits kExternalMemoryHandleType =
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
inline constexpr VkExternalSemaphoreHandleTypeFlagBits kExternalSemaphoreHandleType =
    VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
inline constexpr const char* kExtMemExtName = VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME;
inline constexpr const char* kExtSemExtName = VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME;
#endif

// ─── GL import handle type ──────────────────────────────────────────────────
//
// The values GL_EXT_memory_object{,_win32,_fd} actually defines. They used to
// read 0x9462 and 0x9464, which are GL_DEVICE_LUID_EXT and GL_DEVICE_NODE_MASK_EXT
// -- not handle types at all. The driver rejected every import with
// GL_INVALID_VALUE and "Unknown handle type", and because no call site checked
// glGetError, every VK -> GL import silently produced an unusable texture.
#ifdef _WIN32
inline constexpr unsigned int kGlHandleType = 0x9587; // GL_HANDLE_TYPE_OPAQUE_WIN32_EXT
#else
inline constexpr unsigned int kGlHandleType = 0x9586; // GL_HANDLE_TYPE_OPAQUE_FD_EXT
#endif

// ─── Platform native handle type ────────────────────────────────────────────
#ifdef _WIN32
using native_handle_t = HANDLE;
inline constexpr native_handle_t kInvalidHandle = nullptr;
#else
using native_handle_t = int;
inline constexpr native_handle_t kInvalidHandle = -1;
#endif

/// Close a platform handle if valid. Sets it to kInvalidHandle after closing.
inline void close_handle(native_handle_t& h)
{
    if (h == kInvalidHandle)
        return;
#ifdef _WIN32
    CloseHandle(h);
#else
    ::close(h);
#endif
    h = kInvalidHandle;
}

}}} // namespace caspar::vulkan_common::platform
