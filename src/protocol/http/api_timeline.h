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

// THE TIMELINE DOCUMENT ON THE WIRE, and the routes that carry it.
//
// The codec lives HERE rather than in `core/timeline`, where the plan first put it, for one
// build reason that is not negotiable in this tree: Boost.JSON is compiled from source into
// exactly one translation unit and `protocol_http` deliberately has NO precompiled header
// because of it -- `boost_prelude.h` records the four seconds per translation unit that Beast
// and Boost.JSON cost, and CLAUDE.md records that a header inside a PCH needs the full
// touch-everything-and-delete-the-pch sweep on every edit. `core` has a PCH. So the document
// TYPE is in core, which is what the stage and AMCP both need, and the JSON is here.
//
// WIRE UNITS: SECONDS, always, as a JSON number. Frames, timecode and bars are accepted as
// LITERALS -- `{"frames": 300}`, `{"tc": "00:00:12:00"}`, `{"bars": 4}` -- and converted at PUT
// against the document's own `rate` and `tempo`, so nothing downstream has to know which form
// the author used. A string is an EXPRESSION and goes to the grammar.

#include "api_context.h"
#include "api_status.h"

#include <core/timeline/model.h>

#include <string>

namespace caspar { namespace protocol { namespace http {

/// `PUT /v1/timeline/{name}` -- store a document and report its faults.
///
/// The document is stored whether or not it resolves; `timeline_invalid` carries the faults and
/// the client can GET back what it sent. See `docs/faults.yaml`.
api_reply put_timeline(const api_context& ctx, const std::string& name, const std::string& body);

/// `GET /v1/timeline/{name}` -- the document as stored, with its faults and its resolution
/// summary. `GET /v1/timeline` with an empty name lists what is loaded.
api_reply get_timeline(const api_context& ctx, const std::string& name);

/// `DELETE /v1/timeline/{name}`.
api_reply delete_timeline(const api_context& ctx, const std::string& name);

/// `GET /v1/timeline/{name}/resolved?at=<seconds>` -- instances, and the per-layer owner at
/// `at`. This is what the client DRAWS (`L28`): the collision rules are applied here, once,
/// rather than reimplemented in every client and disagreeing.
api_reply get_timeline_resolved(const api_context& ctx, const std::string& name,
                                const std::string& query);

/// Parse a document out of JSON. Exposed for AMCP's `TIMELINE LOAD`, which has the same codec
/// and a different transport.
///
/// Returns an `api_reply` with `code == ok` on success; the document lands in `out`. A parse
/// failure is `bad_request` (the JSON or a value was wrong) rather than `timeline_invalid`
/// (the document was understood and does not resolve) -- a client acts on those differently.
api_reply parse_timeline_document(const std::string& body, const std::string& name,
                                  core::timeline::timeline_document& out);

}}} // namespace caspar::protocol::http
