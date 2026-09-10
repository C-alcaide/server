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

#include "boost_prelude.h"

#include <string>

namespace caspar { namespace protocol { namespace http {

/// The status vocabulary.
///
/// A generated control surface is the reason this is a closed set of named codes rather
/// than HTTP statuses: a client that receives `field_out_of_range` can show the range that
/// was violated and put the control back where it was, and one that receives
/// `400 Bad Request` can only show a shrug.
enum class api_code
{
    ok = 0,
    unknown_path,             ///< no such path in the tree
    not_writable,             ///< a read-only path was written to
    field_wrong_type,         ///< the JSON type does not match the descriptor's
    field_out_of_range,       ///< outside min/max, and the bounding rule did not absorb it
    field_missing,            ///< a required value was absent
    field_conflict,           ///< a compare-and-set whose expected value did not match
    field_bound,              ///< the target is owned by a binding; UNBIND it first
                              ///<
                              ///< Distinct from `not_writable`, and the difference is the
                              ///< whole point: `not_writable` means the field is derived and
                              ///< never settable, while `field_bound` means it is settable
                              ///< and something else currently owns it -- so a control
                              ///< surface can offer `UNBIND` rather than greying the slider
                              ///< out forever.
    channel_not_found,        ///< the address resolves to no channel
    layer_not_found,          ///< ...or to no layer
    producer_not_ready,       ///< a valid write that cannot be honoured yet
    batch_op_failed,          ///< with the failing index and that op's own status
    not_supported_on_backend, ///< the fork's own: one mixer implements a field, the other does not
    timeline_not_found,       ///< no document by that name
    timeline_invalid,         ///< the document is bad; whether it was STORED depends on which half
                              ///<
                              ///< TWO CASES, and the reply says which by whether a GET finds the
                              ///< document afterwards:
                              ///<
                              ///< A RESOLUTION fault -- a reference to an object that does not
                              ///< exist, a cycle, an end before its start -- is STORED. That is
                              ///< the difference from `bad_request`: the client can GET back what
                              ///< it sent and see the errors against it, because a half-authored
                              ///< show is the normal state of a document being edited, and an
                              ///< expression legitimately names an object not written yet.
                              ///<
                              ///< A DECODE fault -- a malformed key, an unknown easing, a PATH
                              ///< THAT NAMES NO PARAMETER -- leaves nothing behind. A path names a
                              ///< registry, not another object, and the registry does not change
                              ///< while the author types, so there is no edit in progress for
                              ///< storing it to help.
                              ///<
                              ///< `details` carries one entry per fault with `object`,
                              ///< `expression` and `reason`, so a client can highlight the
                              ///< expression the author typed rather than reporting that the
                              ///< document is invalid.
    graph_not_found,          ///< no node graph by that name
    graph_invalid,            ///< the graph has an `error` fault. STORED, and that is the point:
                              ///<
                              ///< a graph is edited WHILE IT IS ON AIR, so an operator who
                              ///< mistypes a port name must not lose the grade that is
                              ///< rendering. `details` carries one entry per fault with
                              ///< `severity`, `node`, `edge`, `port` and `reason`; an attached
                              ///< layer keeps its LAST GOOD plan and the stage publishes
                              ///< `graph_stale`.
                              ///<
                              ///< A `coercion` entry is NOT an error and does not produce this
                              ///< code: a legal-but-lossy join (a mask read as an image, a real
                              ///< rounded to a whole number) compiles and renders, and the entry
                              ///< exists so an editor can draw a warning on the edge instead of
                              ///< the server either refusing it or doing it silently.
    graph_attached,           ///< that document is already attached to a layer
                              ///<
                              ///< One document, at most one layer. The attached document's
                              ///< parameter values ARE the operator's constant, so a second
                              ///< attachment would be two answers to "what is this parameter's
                              ///< value". Reusing a look on another layer is a PUT under another
                              ///< name, which is also what makes the two independently gradeable.
                              ///<
                              ///< Re-attaching the SAME document to the SAME layer is not this
                              ///< error -- it is idempotent, which a client retrying after a
                              ///< timeout depends on.
    unauthorized,
    bad_request,
    internal,
};

const char* to_string(api_code c);

/// One reply, before it is wrapped.
struct api_reply
{
    api_code    code = api_code::ok;
    std::string message;
    json::array details; ///< structured context: the failing index, the range, the component
    json::value result = nullptr;

    static api_reply ok_with(json::value v) { return api_reply{api_code::ok, "", {}, std::move(v)}; }
    static api_reply fail(api_code c, std::string msg, json::array det = {})
    {
        return api_reply{c, std::move(msg), std::move(det), nullptr};
    }
};

/// The uniform envelope carried by EVERY reply, success or failure.
///
/// Application errors arrive as HTTP 200 with a non-zero `status.code`; the HTTP status is
/// reserved for transport. Two reasons that separation is worth the oddity: an intermediary
/// that retries or logs by HTTP status should not treat "you asked for a path that does not
/// exist" as a server fault, and a client that has to branch on both layers gets one place
/// to look instead of two that can disagree.
///
/// `server` names which server answered, so a client attached to more than one -- or
/// logging their replies together -- can tell them apart. It costs nothing now and cannot
/// be added later without breaking every client that parses the envelope.
json::object envelope(const api_reply& r, const std::string& server_name);

}}} // namespace caspar::protocol::http
