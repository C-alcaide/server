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

// Platform-agnostic handle type definitions for Vulkan/GL external memory interop.
// Windows: opaque Win32 HANDLEs
// Linux:   POSIX file descriptors (fd)

#ifdef _WIN32
#define VK_USE_PLATFORM_WIN32_KHR
#include <vulkan/vulkan.h>
#include <windows.h>
#else
#include <vulkan/vulkan.h>
#endif

#include <GL/glew.h>

namespace caspar { namespace vulkan_output { namespace platform {

// ─── Vulkan external memory handle type ─────────────────────────────────────
#ifdef _WIN32
inline constexpr VkExternalMemoryHandleTypeFlagBits kExternalMemoryHandleType =
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
inline constexpr VkExternalSemaphoreHandleTypeFlagBits kExternalSemaphoreHandleType =
    VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
inline constexpr const char* kVkExternalMemoryExtName  = VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME;
inline constexpr const char* kVkExternalSemExtName     = VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME;
#else
inline constexpr VkExternalMemoryHandleTypeFlagBits kExternalMemoryHandleType =
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
inline constexpr VkExternalSemaphoreHandleTypeFlagBits kExternalSemaphoreHandleType =
    VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
inline constexpr const char* kVkExternalMemoryExtName  = VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME;
inline constexpr const char* kVkExternalSemExtName     = VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME;
#endif

// ─── GL import handle type ──────────────────────────────────────────────────
#ifdef _WIN32
inline constexpr GLenum kGlHandleType = GL_HANDLE_TYPE_OPAQUE_WIN32_EXT;
inline constexpr const char* kGlMemoryObjectExtName = "GL_EXT_memory_object_win32";
inline constexpr const char* kGlSemaphoreExtName    = "GL_EXT_semaphore_win32";
#else
inline constexpr GLenum kGlHandleType = GL_HANDLE_TYPE_OPAQUE_FD_EXT;
inline constexpr const char* kGlMemoryObjectExtName = "GL_EXT_memory_object_fd";
inline constexpr const char* kGlSemaphoreExtName    = "GL_EXT_semaphore_fd";
#endif

// ─── Platform handle type ───────────────────────────────────────────────────
#ifdef _WIN32
using native_handle_t = HANDLE;
inline constexpr native_handle_t kInvalidHandle = nullptr;
#else
using native_handle_t = int;
inline constexpr native_handle_t kInvalidHandle = -1;
#endif

}}} // namespace caspar::vulkan_output::platform
