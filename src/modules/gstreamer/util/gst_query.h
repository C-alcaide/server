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

#include <protocol/amcp/amcp_command_context.h>

#include <string>

namespace caspar { namespace gstreamer {

/// `GST INFO` — what was loaded, from where, and how much of it registered.
///
/// The questions this answers are the ones that otherwise cost a server restart with
/// GST_DEBUG turned up: is GStreamer even initialised, which install did it take, and did
/// the plugin scan blacklist anything. A blacklisted plugin is the failure mode that looks
/// like a missing element three commands later.
std::wstring info_command(protocol::amcp::command_context& ctx);

/// `GST LIST [substring]` — element factories whose name contains the substring.
///
/// Without an argument this would be 1600-odd lines, so it requires one. The point is
/// answering "is there an srt sink on this box" without leaving the control protocol.
std::wstring list_command(protocol::amcp::command_context& ctx);

}} // namespace caspar::gstreamer
