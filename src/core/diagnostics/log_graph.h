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

namespace caspar { namespace core { namespace diagnostics { namespace log {

/// Send every diagnostics metric to the log as well as to the OSD window.
///
/// The OSD graph window is the only consumer of `diagnostics::graph` in stock CasparCG, which
/// makes frame time, queue depth, decode cost and the dropped-frame tags visible to a human
/// watching a screen and to nothing else. That is fine for operating and useless for
/// measuring: a battery cannot read a graph, and "did the buffer deplete at the loop wrap"
/// is not answerable after the fact.
///
/// This registers a second sink through the same `spi::register_sink_factory` SPI the OSD
/// window uses, so it receives every `set_value` / `set_text` / `set_tag` from every
/// component that already publishes them -- producers, mixer, consumers, audio mixer, route
/// producer -- with no per-site instrumentation and nothing to keep in step as components
/// come and go.
///
/// Off unless `<log-diagnostics>` is set, because it is per-graph-per-interval output on the
/// frame path.
void register_sink();

}}}} // namespace caspar::core::diagnostics::log
