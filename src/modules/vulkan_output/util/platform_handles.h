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

// Thin forwarding header — all Vulkan platform constants live in the shared
// common/vulkan/platform_handles.h.  This file imports them into the
// vulkan_output::platform namespace for backward compatibility, and adds
// GL-specific extension name constants that only the output module needs.

#include <common/vulkan/platform_handles.h>

#include <GL/glew.h>

namespace caspar { namespace vulkan_output { namespace platform {

// ─── Forwarded from vulkan_common::platform ─────────────────────────────────
using vulkan_common::platform::kExternalMemoryHandleType;
using vulkan_common::platform::kExternalSemaphoreHandleType;
using vulkan_common::platform::kExtMemExtName;
using vulkan_common::platform::kExtSemExtName;
using vulkan_common::platform::kGlHandleType;
using vulkan_common::platform::native_handle_t;
using vulkan_common::platform::kInvalidHandle;
using vulkan_common::platform::close_handle;

// Aliases for output-specific naming (used in vulkan_device.cpp extension lists)
inline constexpr const char* kVkExternalMemoryExtName = kExtMemExtName;
inline constexpr const char* kVkExternalSemExtName    = kExtSemExtName;

// ─── GL extension name constants (output module only) ───────────────────────
#ifdef _WIN32
inline constexpr const char* kGlMemoryObjectExtName = "GL_EXT_memory_object_win32";
inline constexpr const char* kGlSemaphoreExtName    = "GL_EXT_semaphore_win32";
#else
inline constexpr const char* kGlMemoryObjectExtName = "GL_EXT_memory_object_fd";
inline constexpr const char* kGlSemaphoreExtName    = "GL_EXT_semaphore_fd";
#endif

}}} // namespace caspar::vulkan_output::platform
