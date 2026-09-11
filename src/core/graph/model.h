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

// WHAT A NODE GRAPH IS, as data. No behaviour, no JSON, no stage -- the same division
// `timeline/model.h` draws, for the same reason: AMCP and HTTP both need the type and neither
// should own it.
//
// A DOCUMENT, not a transform field. The prototype it replaces was a 16-slot array of one fixed
// record inside `image_transform`, index-addressed, with no edges, no ports and no types. That
// shape cannot carry a node's identity across an edit -- delete node 3 and every reference to
// node 4 means something else -- which is `L48`'s failure exactly. So:
//
//   NODE IDS ARE THE CLIENT'S AND ARE REQUIRED. A client that draws a graph already has ids for
//   the things it drew; inventing server-side ones would make every client keep a mapping, and
//   the first thing they would do is send their own anyway. Ids are what an ADDRESS is built
//   from (`node/<id>/<param>`), so they are the one thing that must survive an edit.
//
//   EDGE IDS ARE SERVER-ASSIGNED WHEN ABSENT, echoed, and stable thereafter. Drawing an edge is
//   a gesture with no natural name, so requiring one would make a client invent `e17`; but an
//   edge needs an id for a fault to point at, and `mute` is addressable per edge in v2.
//
//   `ui` IS STORED UNINTERPRETED. Node positions, the client's camera, a collapsed group: raw
//   JSON text, echoed back verbatim, never parsed beyond a well-formedness check. The server has
//   no opinion about where a box is, and the moment it has a SCHEMA for that it owns a client's
//   layout decisions forever. Well-formedness is checked because a malformed blob would break
//   the next GET rather than the PUT that stored it.
//
// PERSISTENCE IS THE CLIENT'S, as for timelines. The store holds what is loaded; a show file is
// the client's artefact. Consistent with the study's answer for the timeline, and the reason is
// the same: the server has no idea which of five clients' documents is the canonical one.

#include <core/monitor/monitor.h>

#include <cstdint>
#include <string_view>
#include <map>
#include <string>
#include <vector>

namespace caspar { namespace core { namespace graph {

/// WHERE in the layer's draw the graph's passes happen. The decision the prototype got wrong.
///
/// `working` is the default and the design: scene-linear, in the working gamut, before tone-map
/// and the OETF -- so a node CDL is the same operation as `MIXER CDL` and a node exposure is a
/// stop of light. Reached by splitting the layer draw at the boundary the kernel already knows:
/// a head pass stops at the working-space point, the node passes run there, and a tail pass does
/// the output half against the real target.
///
/// `display` keeps the PROTOTYPE'S placement byte-for-byte, and it is kept rather than removed
/// because it is a legitimate thing to want: a correction expressed on the picture as encoded --
/// a broadcast-legal clamp, a display-referred trim -- is not the same operation in linear light.
/// Measured 2026-09-11: the two placements put the same CDL 42.00 LSB apart, identically on both
/// mixers, so this is a real choice and not a compatibility shim.
///
/// A single graph is entirely in one stage. Cross-stage edges are refused rather than converted,
/// because inserting an EOTF into the middle of a chain the author did not ask for is exactly the
/// assumption the port tags exist to prevent.
enum class graph_stage : std::uint8_t
{
    working,
    display,
};

const char* stage_name(graph_stage);
bool        stage_from_name(std::string_view, graph_stage& out);

/// One node instance.
struct graph_node
{
    std::string id;    ///< the client's, required, and what `node/<id>/<param>` addresses
    std::string cls;   ///< a `node_class::id`
    std::string label; ///< the operator's name for THIS instance, if they gave it one

    /// Parameter values, keyed by port name. Values only -- the descriptor lives in the registry,
    /// so a document does not carry types and cannot disagree with the server about them.
    ///
    /// SPARSE: a parameter at its default is absent. That is what keeps the published leaf count
    /// bounded (a reader has the descriptor and can fill the default in) and it makes "the client
    /// did not set this" distinguishable from "the client set it to the default", which matters
    /// for a preset diff.
    std::map<std::string, monitor::vector_t> params;

    /// Client presentation, stored as raw JSON text and never interpreted.
    std::string ui;
};

/// One edge. `from` is always an output and `to` always an input; the validator enforces it
/// rather than the parser guessing, so a client that swapped them gets told which port is which.
struct graph_edge
{
    std::string id; ///< server-assigned (`e<N>`) when the client sends none
    std::string from_node, from_port;
    std::string to_node, to_port;

    /// A MUTED EDGE IS DISCONNECTED WHILE STAYING IN THE DOCUMENT, which is the point: an
    /// operator taking a branch out during a show wants it back, and deleting the edge loses
    /// where it went. Mute is a VALUE, not topology -- so the plan does not change, and a
    /// timeline can step it.
    bool muted = false;
};

/// The document.
struct graph_document
{
    std::string  name;
    /// Server-assigned, from the store's counter. Never client-supplied: two clients with their
    /// own numbering could not be ordered by a third, which is the whole use of the number.
    std::int64_t revision = 0;

    /// `working` IS THE DEFAULT, stated here rather than left to the initialiser because the
    /// consequence is not local: a document that does not mention `stage` renders through the
    /// head/tail split, so a split that is broken or half-landed is the path EVERY graph takes.
    ///
    /// That is not hypothetical. The first implementation of the split was measured 68 LSB out
    /// and reverted rather than half-landed for exactly this reason, and neither `api-graph` nor
    /// `graph-stack` could see it -- neither looks at a pixel. `grade-graph` is the only thing
    /// that adjudicates this field, and it captures BOTH stages in one arm so an agreement
    /// cannot be confused with a graph that never ran.
    graph_stage stage = graph_stage::working;

    std::vector<graph_node> nodes;
    std::vector<graph_edge> edges;

    /// Document-level client presentation -- the editor's camera, say. Uninterpreted.
    std::string ui;
};

/// One thing wrong with a document, or one thing worth telling the client about it.
///
/// TWO SEVERITIES, and the difference is whether the graph can run. An `error` means it cannot
/// compile; a `coercion` means it compiles and the client should draw a warning on the edge,
/// because a legal-but-lossy join (a mask read as an image, a real rounded to an int) is
/// something an author should see rather than something the server should refuse. Reporting a
/// coercion rather than silently doing it is the whole reason the field exists.
struct graph_fault
{
    enum class severity : std::uint8_t
    {
        error,
        coercion,
    };

    severity    sev = severity::error;
    std::string node; ///< empty when the fault is on an edge or the document
    std::string edge;
    std::string port;
    std::string reason;

    bool is_error() const { return sev == severity::error; }
};

const char* severity_name(graph_fault::severity);

}}} // namespace caspar::core::graph
