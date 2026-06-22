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

#include <vulkan/vulkan.h>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <vulkan/vulkan_win32.h>
#else
#include <unistd.h>
#endif

namespace caspar { namespace vulkan_output { namespace platform {

// ─── External memory handle type constants ──────────────────────────────────

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

// ─── Platform native handle type ────────────────────────────────────────────

#ifdef _WIN32
using native_handle_t = HANDLE;
inline constexpr native_handle_t kInvalidHandle = nullptr;
#else
using native_handle_t = int;
inline constexpr native_handle_t kInvalidHandle = -1;
#endif

// ─── Handle lifecycle ───────────────────────────────────────────────────────

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

// Mark a handle as consumed (e.g., after vkAllocateMemory with VkImportMemoryFdInfoKHR
// which takes ownership of the fd). Does NOT close — just invalidates tracking.
inline void consume_handle(native_handle_t& h) { h = kInvalidHandle; }

}}} // namespace caspar::vulkan_output::platform
