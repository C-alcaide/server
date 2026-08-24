/*
 * Copyright (c) 2011 Sveriges Television AB <info@casparcg.com>
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

namespace caspar { namespace gstreamer { namespace runtime {

/// The GStreamer installation this module binds to, resolved once from
/// <configuration><gstreamer><path>, then GSTREAMER_1_0_ROOT_MSVC_X86_64.
/// Empty when neither names a directory that exists.
const std::wstring& root_path();

/// Loads the GStreamer libraries by explicit path and calls gst_init.
/// Throws if the installation is missing or gst_init fails. Idempotent, and
/// the failure is sticky: a second call reports the same error without
/// retrying the load.
///
/// Every gst_* call in this module must be reached through a code path that
/// called this first — the libraries are delay-loaded, so calling one before
/// the explicit load turns a missing installation into a loader exception
/// rather than a message.
void ensure_initialized();

/// True when ensure_initialized() has succeeded. Never throws, never loads.
bool is_initialized();

/// gst_version_string(), or an empty string before initialization.
const std::wstring& version();

}}} // namespace caspar::gstreamer::runtime
