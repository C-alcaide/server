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
 *
 * Author: Robert Nagy, ronag89@gmail.com
 */

#pragma once

#include <mutex>

namespace caspar { namespace core { namespace diagnostics { namespace osd {

void register_sink();
void show_graphs(bool value);
void shutdown();

// Global mutex to prevent SFML2 WGL context races between the DIAG window's
// render thread and other threads creating sf::Window instances (e.g. screen
// consumer).  Both sides must hold this mutex when interacting with WGL.
std::recursive_mutex& sfml_context_mutex();

}}}} // namespace caspar::core::diagnostics::osd
