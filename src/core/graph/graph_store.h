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

// WHERE THE GRAPHS LIVE. `timeline_store`'s shape, and the concurrency argument is the same one:
// an immutable `shared_ptr<const stored_graph>` per name under a mutex, so a PUT builds and
// validates a whole new entry on the API executor and swaps a pointer, while the tick takes a
// copy of the pointer and never blocks on a parse, a validation or a topological sort.
//
// WHAT IS DIFFERENT FROM THE TIMELINE STORE, and each difference is a decision rather than drift:
//
//   TWO REVISION COUNTERS, and keeping them apart is load-bearing. `revision()` moves on a
//   STRUCTURE change -- put, erase, undo, redo, attach, detach -- and is mixed into the stage's
//   `structure_revision` fingerprint, so a client re-walks the tree when the shape of what it is
//   drawing changes. `values_revision()` moves on `patch_params` and is deliberately NOT in the
//   fingerprint: an operator dragging a slider, or a timeline ramping a node parameter, changes
//   values fifty times a second, and mixing that in would make every client re-walk every
//   channel's whole tree once per frame. That is `L376`'s rule and it is the reason the
//   plan/values split exists at all.
//
//   AN ATTACHMENT, at most one per document. The attached document's parameter values ARE the
//   operator's constant -- there is no second table seeded from the document -- so there is
//   exactly one answer to "what is this parameter's value". A second `ATTACH` is refused with
//   `graph_attached` rather than silently moving the graph, because moving it would take a look
//   off air on a layer whose operator did not ask for anything.
//
//   HISTORY, which the timeline has none of. A graph is EDITED -- dozens of small gestures, each
//   of which an operator expects to be able to take back -- where a timeline document is authored
//   and then PUT whole. One entry per GESTURE rather than per write is the whole difficulty: a
//   slider drag is fifty `patch_params` calls and one undo, so consecutive patches carrying the
//   same `label` coalesce into one entry.
//
// PERSISTENCE IS THE CLIENT'S, as for timelines. This holds what is loaded.

#include "model.h"
#include "plan.h"
#include "validate.h"

#include <chrono>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace caspar { namespace core { namespace graph {

/// One stored document plus everything derived from it. Immutable once published.
struct stored_graph
{
    graph_document           document;
    std::vector<graph_fault> faults;
    /// Node ids in evaluation order. Empty exactly when `ok()` is false.
    std::vector<std::string> order;

    /// THE COMPILED PLAN, and null exactly when `ok()` is false.
    ///
    /// Built HERE, at PUT, and never in the tick: the pointer IS the still-frame fingerprint, so
    /// allocating one per frame would make a paused unchanging graph look different every frame
    /// and defeat the cache it feeds. See `plan.h` for the attribute/signal split this rests on.
    std::shared_ptr<const node_plan> plan;

    /// The document's parameter values in the plan's flat layout. Rebuilt on every `put` and
    /// every `patch_params` -- which is the cheap half, and the reason the two are separate.
    node_values values;

    /// Can this graph run? No `error` faults. A `coercion` fault does not stop it -- that is the
    /// entire point of the severity split.
    bool ok() const { return !has_error(faults); }
};

/// Where a document is attached, if it is.
struct attachment
{
    int channel = 0;
    int layer   = -1;

    bool valid() const { return channel > 0 && layer >= 0; }
};

/// One entry in a document's undo history.
struct history_entry
{
    std::int64_t                          revision = 0;
    std::string                           label;   ///< what the client called the gesture
    std::chrono::system_clock::time_point at;
    /// `put`, `patch`, `attach`, `detach` -- what KIND of change this was, so a client can
    /// present "undo the last edit" differently from "undo taking it off air".
    std::string kind;
};

class graph_store
{
  public:
    /// Replace (or create) a document. Validated HERE, on the caller's thread.
    ///
    /// A document with faults is still STORED, which matters more here than for a timeline: a
    /// graph is edited while it is ON AIR, so an operator who mistypes a port name must not lose
    /// the grade that is currently rendering. The attached layer keeps its last good plan and the
    /// stage publishes `graph_stale`.
    ///
    /// `label` names the gesture for `history`. Empty means "an edit".
    std::shared_ptr<const stored_graph> put(graph_document doc, const std::string& label = "");

    std::shared_ptr<const stored_graph> get(const std::string& name) const;

    /// Remove it. Detaches first if it is attached -- a client deleting a look means "take it off
    /// air", and refusing would leave them with a document they cannot get rid of without
    /// remembering where it is attached. Flagged as a policy call in the plan (G6).
    bool erase(const std::string& name);

    std::vector<std::string> names() const;

    /// Write parameter VALUES without touching the structure.
    ///
    /// This is what an operator's slider and a timeline's `commit` both go through, and it is why
    /// `values_revision()` exists: it bumps that and NOT `revision()`, so a value change does not
    /// make every client re-walk the tree.
    ///
    /// Keys are `node/<id>/<param>` -- the ADDRESS form, not a pair -- because that is what the
    /// overlays, the publication and the write path all key on, and converting between two
    /// spellings at the boundary is where a mismatch would hide.
    ///
    /// Returns null if there is no such document. A key naming no node or port is IGNORED rather
    /// than refused: this is called from the stage executor on a document that may have been
    /// re-PUT since the driver was resolved, and throwing on the frame path over a stale key
    /// would take a channel down.
    std::shared_ptr<const stored_graph> patch_params(
        const std::string& name, const std::map<std::string, monitor::vector_t>& values,
        const std::string& label = "");

    /// Claim `name` for a layer. False if it is already attached somewhere -- the caller answers
    /// `graph_attached`.
    bool                      claim(const std::string& name, attachment where);
    void                      release(const std::string& name);
    std::optional<attachment> attached(const std::string& name) const;
    /// Which document a layer has, if any. The stage's per-tick lookup.
    std::optional<std::string> attached_to(int channel, int layer) const;

    /// Undo/redo one entry. Null when there is nothing to undo.
    ///
    /// REDO IS CLEARED BY A PUT, which is the ordinary editor rule: once you branch, the old
    /// future is not reachable and offering it would apply an edit to a document it was never
    /// made against.
    std::shared_ptr<const stored_graph> undo(const std::string& name);
    std::shared_ptr<const stored_graph> redo(const std::string& name);
    std::vector<history_entry>          history(const std::string& name) const;
    bool                                can_undo(const std::string& name) const;
    bool                                can_redo(const std::string& name) const;

    /// STRUCTURE. Mixed into the stage fingerprint.
    std::int64_t revision() const;
    /// VALUES. Deliberately not in the fingerprint -- see the header comment.
    std::int64_t values_revision() const;

  private:
    struct slot
    {
        std::shared_ptr<const stored_graph>              current;
        std::deque<std::shared_ptr<const stored_graph>>  past;
        std::deque<std::shared_ptr<const stored_graph>>  future;
        std::deque<history_entry>                        entries;
        std::optional<attachment>                        where;
        /// The label the last coalescing change carried, so a run of slider writes under one
        /// label folds into one entry instead of fifty.
        std::string last_label;
    };

    /// DEPTH 64, chosen and not measured (G5). A whole document per entry, and a graph is small
    /// -- tens of nodes -- so 64 of them is kilobytes rather than anything worth bounding by
    /// memory. The number is about how far back an operator would reasonably reach.
    static constexpr std::size_t history_depth = 64;

    mutable std::mutex                     lock_;
    std::unordered_map<std::string, slot>  slots_;
    std::int64_t                           revision_        = 0;
    std::int64_t                           values_revision_ = 0;

    /// Build the entry for a document, validating it. Never called under the lock.
    /// Non-const on purpose: `patch_params` has to set the entry's revision back to the
    /// document's existing one before publishing it, because a value change must not move
    /// the structure revision. Every caller converts to `const` on publication.
    static std::shared_ptr<stored_graph> build(graph_document doc);
    /// Push `s.current` onto `past`, trim, clear `future`, and record the entry. Under the lock.
    void remember(slot& s, const std::string& label, const char* kind, bool coalesce);
};

/// Aborts the boot on a disagreement between the store and its own rules.
void graph_store_self_test();

}}} // namespace caspar::core::graph
