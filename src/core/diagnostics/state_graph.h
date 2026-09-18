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

#include <core/monitor/monitor.h>

#include <functional>
#include <memory>

namespace caspar { namespace core { namespace diagnostics { namespace publish {

/// Publish every diagnostics graph into `monitor::state`, so a client can redraw the DIAG
/// window and surface its events without a screen or a log file.
///
/// The third `graph_sink`. `osd_graph` draws the graphs for a person at the machine and
/// `log_graph` writes them as lines for a battery or a post-mortem; neither reaches a client
/// on the network. This one folds what every graph writes into one `monitor::state` tree and
/// hands it to the shell at a fixed cadence, and the shell puts it where the other two
/// facades already are -- the control API's hub (`/v1/tree/diag`, `/v1/value/diag/...`,
/// `/v1/events` with a `/diag` prefix) and the OSC client (`/diag/...`).
///
/// THE TREE. A graph has no owner the SPI can see -- `graph.h` gives it a text, values, tags
/// and colours, and nothing else -- so the tree is keyed by a per-graph id rather than by
/// channel, exactly as the DIAG window lists graphs flat. `text` is the identity a reader
/// uses, the same string the window shows:
///
///     /diag/interval                 seconds between publications (the time axis)
///     /diag/{id}/text                the graph's caption, e.g. "ffmpeg[clip.mp4|...]"
///     /diag/{id}/{value}             [last, max] over the last interval
///     /diag/{id}/tag/{name}          [count since enabled, "warning" | "info"]
///     /diag/{id}/color/{name}        the line colour the window would draw, packed RGBA
///
/// The published snapshot's keys are relative to the root (`{id}/text`), as a channel's are
/// relative to `channel/{n}`; the shell supplies the root name.
///
/// Values are FOLDED, not streamed. A graph writes `frame-time` once per frame; publishing
/// per frame would be the per-tick tree cost this tree's CLAUDE.md measures (a leaf per graph
/// per frame, inside every subscriber's diff). `[last, max]` keeps the level and the spike,
/// which is what a person reads off the window. Tags are COUNTED, as the log sink counts
/// them: a dropped frame is an event, and a sampled event is a missed one.
///
/// OPT-IN, and cheap when off. `<publish-diagnostics>` in the config or `DIAG PUBLISH ON`
/// over AMCP turns it on; the default is off. The sink is registered unconditionally so the
/// toggle works at runtime -- a graph collects its sinks when it is created -- and every
/// write it receives while disabled is one relaxed atomic load. Turning it off publishes one
/// empty tree, so the sub-tree disappears and an events subscriber sees every leaf reverted.
///
/// THE THREAD. Publication runs on its own thread, not on a channel tick: the fold walks
/// every live graph under a mutex and builds a map, which is work the frame budget should
/// not carry, and the OSD sink already owns a thread for the same reason.

using publish_fn_t = std::function<void(std::shared_ptr<const monitor::state>)>;

/// Register the sink factory and remember where a published tree goes. Call once, at
/// start-up, before any graph is created; `publish` is invoked on the publication thread.
void register_sink(double interval_seconds, publish_fn_t publish);

/// Turn publication on or off at runtime. Idempotent.
void set_enabled(bool enabled);
bool enabled();

/// The configured cadence, for the reply to a query.
double interval();

}}}} // namespace caspar::core::diagnostics::publish
