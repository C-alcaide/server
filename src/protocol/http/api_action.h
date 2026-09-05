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

#include <string>

namespace caspar { namespace protocol { namespace http {

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
api_reply run_batch(const api_context& ctx, const std::string& body, const std::string& peer);

}}} // namespace caspar::protocol::http
