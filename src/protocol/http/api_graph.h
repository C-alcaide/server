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

// THE NODE GRAPH ON THE WIRE.
//
// Here rather than in `core/graph` for the same build reason `api_timeline.cpp` is here: Boost.JSON
// is compiled from source into one translation unit and `protocol_http` deliberately has no
// precompiled header because of it. The document TYPE is in core, which is what the stage and AMCP
// both need; the JSON is here.
//
// THE EDGE SPELLING IS `"n1.out"`, one string, not a pair of fields. A client draws an edge from a
// port to a port, and the compact form is what an author reads back without decoding an object per
// edge. It splits on the LAST '.', because a node id may not contain one -- the validator refuses
// an id with '/' or '.' precisely so this stays unambiguous, and so that the ADDRESS
// `node/<id>/<param>` stays unambiguous too.
//
// WHAT A PUT ANSWERS WITH, and each part earns its place:
//
//   `revision`  the store's counter. Server-assigned, so a third party can order two versions.
//   `order`     node ids in evaluation order. This is the thing a client cannot compute for
//               itself without reimplementing the topological sort AND the coercion rules.
//   `faults`    one entry per fault, each naming a node, an edge or a port. `severity` splits
//               "this cannot run" from "this runs and loses something".
//   `edges`     echoed WITH their server-assigned ids, because a client that sent none has no
//               other way to learn them -- and it needs them to address a fault or to mute one.
//
// A DOCUMENT WITH `error` FAULTS IS STILL STORED and answers `graph_invalid`. That is the
// timeline's precedent and it matters more here: a graph is edited while it is on air. What is
// REFUSED instead is a DECODE fault -- malformed JSON, an edge spelling that does not split, a
// `ui` blob that is not JSON -- for the reason commit 0 refused a timeline path: those name the
// grammar rather than another object, and the grammar does not change while the author types.

#include "api_context.h"
#include "api_status.h"

#include <string>

namespace caspar { namespace protocol { namespace http {

/// `GET /v1/graph` -- every loaded document, with its revision, counts, faults and attachment.
api_reply list_graphs(const api_context& ctx);

/// `PUT /v1/graph/{name}` -- store a document.
///
/// `graph_invalid` when it has `error` faults, and it is stored anyway. `bad_request` for a decode
/// fault or a name disagreement between the path and the body.
api_reply put_graph(const api_context& ctx, const std::string& name, const std::string& body);

/// `GET /v1/graph/{name}` -- the document as stored, with `faults`, `order` and `attached`.
api_reply get_graph(const api_context& ctx, const std::string& name);

/// `DELETE /v1/graph/{name}` -- remove it, detaching first if it is attached.
api_reply delete_graph(const api_context& ctx, const std::string& name);

/// `GET /v1/graph/{name}/history` -- the undo stack, newest last, plus `can_undo`/`can_redo`.
api_reply get_graph_history(const api_context& ctx, const std::string& name);

/// `POST /v1/graph/{name}/{attach|detach}` -- put the document on a layer, or take it off.
///
/// `{"channel": n, "layer": m}` for `attach`; `detach` needs neither, because a document is
/// attached to at most one layer and the store knows which.
///
/// A ROUTE RATHER THAN AMCP-ONLY, which the plan's endpoint table did not list: without it a
/// client that speaks only HTTP could store a graph, read it, undo it and delete it, and never
/// put it on a layer. The AMCP `GRAPH <ch>-<layer> ATTACH` façade exists as well, for an operator
/// at a console.
api_reply graph_attach_verb(const api_context& ctx, const std::string& name,
                            const std::string& verb, const std::string& body);

/// `POST /v1/graph/{name}/{undo|redo}` -- one step, answering the resulting document.
api_reply graph_history_verb(const api_context& ctx, const std::string& name,
                             const std::string& verb);

}}} // namespace caspar::protocol::http
