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
#include <vector>

namespace caspar { namespace accelerator { namespace ocio {

/// Process-wide access to the loaded OpenColorIO config.
///
/// Deliberately a pure-std interface with no OCIO types in it. Two reasons: OCIO's headers
/// stay confined to the one translation unit that implements this, and the rest of the tree
/// -- protocol in particular, which needs the discovery commands -- does not acquire a
/// compile-time dependency on the library. It also means a build with ENABLE_OCIO=OFF
/// compiles unchanged, with available() answering false.
///
/// The config is process-wide rather than per-channel because it is what makes a look
/// reproducible: pinning one built-in config URI for the whole server means a colour space
/// name resolves to the same transform on every channel and in every log line.

/// Was the server built with OCIO support?
bool available();

/// OCIO library version, e.g. "2.5.2". Empty when unavailable.
std::string version();

/// The URI of the currently loaded config. Loads the default on first call.
std::string config_uri();

/// Replace the loaded config.
///
/// Returns false on failure -- an unknown URI, a malformed file -- having logged why, and
/// leaves the previously loaded config in place. A failed load must not leave the server
/// with no colour config, because commands already accepted refer to it.
bool load_config(const std::string& uri);

/// Colour space names in the loaded config, in config order.
std::vector<std::string> colorspaces();

/// Display names in the loaded config.
std::vector<std::string> displays();

/// View names available for `display`. Empty if the display is unknown.
std::vector<std::string> views(const std::string& display);

/// Whether `name` names a colour space in the loaded config. For validating an AMCP
/// argument at command time, so a bad name fails the command rather than the frame.
bool has_colorspace(const std::string& name);

/// Whether `display` exists and offers `view`.
bool has_display_view(const std::string& display, const std::string& view);

}}} // namespace caspar::accelerator::ocio
