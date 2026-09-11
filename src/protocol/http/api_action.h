/*
 * Copyright (c) 2026 CasparCG Contributors
 *
 * This file is part of CasparCG (www.casparcg.com).
 *
 * CasparCG is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#pragma once

#include "api_context.h"
#include "api_status.h"

#include <core/timeline/transport.h>

#include "api_value.h"

#include <string>
#include <vector>

namespace caspar { namespace protocol { namespace http {

/// Where a verb is aimed. Public because a deferred batch carries them.
struct action_target
{
    int         channel = 0;
    int         layer   = -1; //< -1 when the verb addresses the whole channel
    std::string verb;
};

/// One validated op of a batch.
/// One transport verb for one document, inside a batch.
struct batch_timeline
{
    std::string                       name;
    core::timeline::transport_command cmd;
};

/// `{"op": "graph", "name": "look", "verb": "attach", "layer": 1}`.
struct batch_graph
{
    std::string name;
    std::string verb;
    int         layer = 0;
};

/// A node-parameter write inside a batch: the address split, the value, and the gesture label.
struct batch_node
{
    int                     layer = 0;
    std::string             path;   ///< `node/<id>/<param>`
    core::monitor::vector_t value;
    std::string             label;
};

struct batch_op
{
    std::size_t index  = 0;
    bool        is_set = false;
    /// A TIMELINE verb rather than a field write or a layer action.
    ///
    /// A third flag rather than an enum, because `is_set` was already a bool and turning it into
    /// an enum would touch every reader for no information they do not already have.
    bool           is_timeline = false;

    /// A GRAPH verb -- `attach`, `detach`, `undo`, `redo` -- addressed by document name.
    bool is_graph = false;

    /// A NODE-PARAMETER write. Distinct from `is_set` because it does not go through
    /// `apply_transform`: a node parameter's constant lives in the attached DOCUMENT, not on
    /// the layer's transform, so the write is `set_node_param` and nothing else reaches it.
    ///
    /// THIS IS WHY IT IS A FOURTH FLAG rather than a special case inside `is_set`: routing it
    /// through the transform path is what it did before, and a transform write for an address
    /// no transform has silently did nothing while the batch reported success.
    bool is_node = false;

    prepared_set   set;
    action_target  action;
    batch_timeline timeline;
    batch_graph    graph;
    batch_node     node;
    int            channel = 0;
};

/// A batch that has passed validation and is ready to apply -- now or on a named frame.
struct batch_plan
{
    std::vector<batch_op> ops;
    std::string           label;
    std::string           queue;
    std::string           peer;

    /// The frame this batch is waiting for, on `reference_channel`. 0 means "as soon as
    /// possible", which is the ordinary case.
    std::int64_t at_frame          = 0;
    int          reference_channel = 0;
};

/// `POST /v1/action/channel/{n}[/stage/layer/{m}]/{verb}`.
///
/// The verbs that need nothing but a layer index -- `play`, `stop`, `pause`, `resume`,
/// `preview`, `clear` -- go straight to `stage_base`, which is the same object AMCP's
/// handlers call. The two that need a producer built from a string, `load` and `play` with
/// a clip, are delegated to AMCP: parsing a clip name into a producer is the producer
/// registry's job and is not duplicated here.
///
/// The body is optional. `{"clip":"AMB","loop":true,"seek":100,"length":250}` is the load
/// form; an empty body is the bare verb.
api_reply run_action(const api_context& ctx, const std::string& path, const std::string& body, const std::string& peer);

/// `POST /v1/batch`.
///
/// `{"label":"...", "ops":[{"op":"set","path":"...","value":...}, {"op":"action","path":"...", ...}]}`
///
/// **Validate every op first, apply nothing on any failure.** The reply names the failing
/// index and carries that op's own status, so a client fixes the op rather than the batch.
///
/// Applied through `core::stage_delayed`, one per channel: every op is queued against a
/// blocked executor, every touched channel is locked, and only then are the executors
/// released -- so the ops land inside the same frame on every channel they touch. That is
/// the same mechanism AMCP's `BEGIN`/`COMMIT` uses, written against the core type rather
/// than shared with it (see the commit message for why sharing would cost more).
api_reply run_batch(const api_context& ctx, const std::string& body, const std::string& peer, batch_plan& deferred);

/// Validate a batch body without touching a stage. Fills `out`; on failure the reply names
/// the failing index and nothing has been applied because nothing has been attempted.
api_reply validate_batch(const api_context& ctx, const std::string& body, batch_plan& out);

/// Apply an already-validated batch, atomically across every channel it touches.
api_reply apply_batch(const api_context& ctx, const batch_plan& plan);

}}} // namespace caspar::protocol::http
