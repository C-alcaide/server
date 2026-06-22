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
// accelerator::vulkan::platform namespace for backward compatibility.

#include <common/vulkan/platform_handles.h>

namespace caspar { namespace accelerator { namespace vulkan { namespace platform {

// ─── Forwarded from vulkan_common::platform ─────────────────────────────────
using vulkan_common::platform::kExternalMemoryHandleType;
using vulkan_common::platform::kExternalSemaphoreHandleType;
using vulkan_common::platform::kExtMemExtName;
using vulkan_common::platform::kExtSemExtName;
using vulkan_common::platform::kGlHandleType;
using vulkan_common::platform::native_handle_t;
using vulkan_common::platform::kInvalidHandle;
using vulkan_common::platform::close_handle;

}}}} // namespace caspar::accelerator::vulkan::platform
