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

#include <string>
#include <vector>

namespace caspar { namespace common {

/// Vulkan device extensions that a subsystem outside the accelerator wants enabled.
///
/// WHY A REGISTRY RATHER THAN A DIRECT CALL. The list that matters here is FFmpeg's, and
/// FFmpeg publishes it as `av_vk_get_optional_device_extensions()` -- an avutil function.
/// The accelerator does not link FFmpeg and should not start; the ffmpeg module does. So the
/// module that owns the knowledge deposits it, and the module that creates the device reads
/// it, with neither depending on the other.
///
/// ORDERING is what makes this work and is worth stating: the Vulkan device is created during
/// channel setup, inside `main()`, while these registrations happen in static initialisers at
/// image load. Measured on this tree, the mixer's device is up before the ffmpeg module's own
/// `Initialized ffmpeg module` line, so a module-init hook would have been too late -- static
/// init is not a shortcut here, it is the only point early enough.
///
/// A name that this GPU does not support is not an error. The accelerator enables what is
/// present and ignores the rest, so an over-broad list costs nothing.
void register_vulkan_device_extension_request(std::vector<std::string> names);

/// Everything registered so far, in registration order, deduplicated.
const std::vector<std::string>& requested_vulkan_device_extensions();

}} // namespace caspar::common
