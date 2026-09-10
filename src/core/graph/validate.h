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

// IS THIS DOCUMENT BUILDABLE, and if not, WHICH PART is wrong.
//
// Pure: no stage, no store, no allocation the caller cannot see. Called at PUT, and called again
// on a COPY by `connections/preview` so that "may I draw this edge" and "will this PUT succeed"
// cannot give different answers -- two validators would let a client be offered a gesture the
// server then refuses, which is worse than not offering it.
//
// EVERY FAULT NAMES SOMETHING THE AUTHOR CAN SEE. A node id, an edge id, a port name. "the graph
// is invalid" is not something a client can act on, which is the lesson `timeline_invalid`'s
// `details` array already carries: a client highlights the thing, it does not report a sentence.
//
// AND A DOCUMENT WITH FAULTS IS STILL STORED. That is `timeline_invalid`'s precedent and it
// matters more here, because a graph is edited WHILE IT IS ON AIR: an operator who mistypes a
// port name must not lose the grade that is currently rendering. So an attached layer whose
// document stops compiling keeps its LAST GOOD plan and the stage publishes `graph_stale`. The
// exception is a decode fault -- a malformed key or an unknown class -- which is refused the way
// commit 0 made a timeline path refuse, because those name a registry rather than another object
// and the registry does not change while the author types.
//
// WHAT IS CHECKED, and every one of these is a real failure mode rather than a completeness list:
//
//   the registry           an unknown class or port name. The 202-and-no-picture class: without
//                          this, a typo stores, renders nothing, and is found on air.
//   ids                    duplicate node ids (every reference becomes ambiguous, and an ADDRESS
//                          `node/<id>/<p>` would resolve to two nodes), duplicate edge ids,
//                          empty ids, and an id containing '/' or '.' -- the two characters the
//                          address grammar splits on.
//   direction              `from` must be an output and `to` an input. Checked rather than
//                          inferred, so a client that swapped them is told which is which.
//   domain                 through `coerce`, the ONE table. An error refuses; a legal-but-lossy
//                          join is REPORTED as a `coercion` fault and the document compiles.
//   fan-in                 at most one edge into an input. Fan-OUT is unlimited (that is what a
//                          graph is for); two things writing one input has no defined order.
//   required inputs        an unconnected required input is an error, not a default.
//   parameters             type, arity and range, against the SAME QUANTITIES the write path
//                          checks -- the port descriptor's `type`, `arity`, `min`, `max` and
//                          `bounding`.
//
//                          NOT the same CODE, and that is worth stating rather than glossing:
//                          `check_and_bound` lives in `api_value.cpp`, in `protocol_http`, and
//                          returns an `api_reply`. `core` cannot depend on either. So these are
//                          two implementations of one contract, and what keeps them from
//                          drifting is that both read the same `port_desc` and that `api-graph`
//                          asserts a PUT and a `/v1/value` write refuse the SAME value with the
//                          same reason. If that check ever goes green while one of them accepts,
//                          the contract has drifted and the battery is the only thing that would
//                          say so.
//   the roots              exactly one `input` and one `output`, and the output reachable.
//   cycles                 Kahn's, and the fault NAMES THE EDGE that closes the loop. "there is
//                          a cycle" makes a client search; naming the edge lets them delete it.
//   the pass cap           at most 16 image-producing steps. REFUSED rather than clamped: a
//                          clamped graph renders something the author did not author.
//   the stage              `display` is legal; a cross-stage edge is not, and there is nothing
//                          to cross in v1 because every port is `working` -- so this check is
//                          structural, waiting for the first class that declares otherwise.
// `ui` WELL-FORMEDNESS IS NOT CHECKED HERE. It is raw JSON text and `core` has no JSON parser --
// `protocol_http` compiles Boost.JSON into exactly one translation unit and has no precompiled
// header because of it, which is the same reason `api_timeline.cpp` is not in core. So the codec
// checks it at decode, which is also where it belongs: it is the only place that ever parses the
// blob, and a malformed one would break the next GET rather than the PUT that stored it.

#include "model.h"

#include <string>
#include <vector>

namespace caspar { namespace core { namespace graph {

/// Validate `doc`, and on success fill `topo_order` with node ids in evaluation order.
///
/// `topo_order` is filled ONLY when there is no `error` fault; a partial order for a broken
/// graph would be a thing a client could accidentally draw. An empty return means the document
/// is buildable exactly as sent.
std::vector<graph_fault> validate(const graph_document& doc, std::vector<std::string>& topo_order);

/// Any `error` in the list? The one predicate everything else asks.
bool has_error(const std::vector<graph_fault>&);

/// Aborts the boot on a disagreement between the validator and its own rules.
void graph_validate_self_test();

}}} // namespace caspar::core::graph
