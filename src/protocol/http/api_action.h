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
struct batch_op
{
    std::size_t   index  = 0;
    bool          is_set = false;
    prepared_set  set;
    action_target action;
    int           channel = 0;
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
