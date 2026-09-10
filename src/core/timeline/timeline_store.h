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

// WHERE THE DOCUMENTS LIVE, and the one thing that hands a resolved copy to the tick.
//
// One store for the server, not one per channel, because a document may drive several channels
// (commit 17) and the transport that owns it lives on exactly one. A per-channel store would make
// "which channel owns this show" a property of where the client happened to PUT it.
//
// THE CONCURRENCY SHAPE, which is the only interesting thing here. Two writers and one reader,
// with wildly different budgets:
//
//   the API executor PUTs a document. Rare, and may take as long as it likes.
//   the stage executor READS the resolved form, once per tick per channel, and must never block
//   on a parse or a resolve.
//
// So the store keeps an immutable `shared_ptr<const entry>` per name under a mutex, and the tick
// takes a copy of the pointer. A PUT builds a whole new entry, resolves it, and swaps the pointer
// in; the tick that is mid-frame keeps the old one until its next tick. Nothing is resolved on the
// stage executor and nothing the tick holds can be mutated under it.
//
// The alternative -- a mutex the tick takes -- would put a client's PUT on the critical path of
// every frame on every channel, and a resolve of a large document inside it.

#include "resolver.h"
#include "transport.h"

#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace caspar { namespace core { namespace timeline {

/// One stored document plus the resolution of it. Immutable once published.
struct stored_timeline
{
    timeline_document document;
    resolved_timeline resolved;
    trigger_log       triggers; //< what the resolution above was computed against
};

/// WHERE THE HOME CHANNEL'S PLAYHEAD IS, so another channel can follow it.
///
/// A document declares ONE channel and that channel owns its transport: it takes the commands,
/// advances the position off its own frame counter, and chases house timecode if asked. Every
/// other channel the document addresses is a GUEST -- it evaluates the same resolution at the
/// same position and drives only its own layers.
///
/// WHY A SNAPSHOT RATHER THAN A SHARED TRANSPORT. The transport is mutated per tick (the chase
/// correction, the anchor on a re-rate) and each channel ticks on its own thread, so sharing the
/// object would put a mutex inside the frame path of every channel and make one channel's chase
/// visible to another's arithmetic. A snapshot published once per tick by the owner and read once
/// per tick by each guest is a copy of four scalars and cannot be raced into an inconsistent
/// state, because it is replaced whole.
///
/// `frame` is the HOME channel's frame counter, which is the one thing a guest cannot compute for
/// itself: a guest reading a snapshot has no way to know whether it is this frame's or the
/// previous one's, and a stale snapshot is a document that appears to have stopped. It is
/// published rather than gated on, because two channels on the same format are within a frame of
/// each other by construction and gating would make a guest stutter on the normal case.
struct playhead
{
    flicks          position = 0;
    transport_state state    = transport_state::stopped;
    std::uint64_t   frame    = 0;         //< the home channel's counter when this was taken
    std::int64_t    revision = 0;         //< the document revision it was taken against
    bool            chasing  = false;
};

class timeline_store
{
  public:
    /// Replace (or create) a document. Resolves it HERE, on the caller's thread.
    ///
    /// A document whose resolution fails is still STORED, and that is deliberate: the client
    /// needs to be able to GET back the thing it sent and see the errors against it, and a
    /// half-authored show is the normal state of a document being edited. Nothing evaluates a
    /// document whose `resolved.ok()` is false, so an invalid one is inert rather than dangerous.
    ///
    /// Returns the entry, errors included.
    std::shared_ptr<const stored_timeline> put(timeline_document doc);

    /// The current entry for a name, or null.
    std::shared_ptr<const stored_timeline> get(const std::string& name) const;

    bool erase(const std::string& name);

    std::vector<std::string> names() const;

    /// Re-resolve `name` against a new trigger log -- what a GO does. Null if there is no such
    /// document.
    std::shared_ptr<const stored_timeline> retrigger(const std::string& name, trigger_log triggers);

    /// A monotonic counter over the whole store: every put, erase and retrigger bumps it.
    ///
    /// The stage's `structure_revision` fingerprint mixes this in, so a client that walks the
    /// tree when the revision moves also picks up a document appearing or changing. It is a
    /// COUNTER rather than a hash of the contents because the question it answers is "has
    /// anything changed since I looked", and a counter answers that without hashing documents
    /// that may be large.
    std::int64_t revision() const;

    /// The home channel says where its playhead is. Once per tick, from the stage executor.
    ///
    /// Does NOT bump `revision()`: the fingerprint exists so a client re-walks the tree when its
    /// STRUCTURE changes, and a playhead moves every frame. Mixing it in would make every client
    /// re-walk every channel's whole tree once per frame on every playing document.
    void publish_playhead(const std::string& name, const playhead& p);

    /// Where the home channel's playhead was when it last published. Null if nothing has -- which
    /// is the honest answer for a document nobody has ever played, and the reason a guest drives
    /// nothing rather than driving position zero.
    std::optional<playhead> playhead_of(const std::string& name) const;

  private:
    mutable std::mutex                                                        lock_;
    std::unordered_map<std::string, std::shared_ptr<const stored_timeline>>   docs_;
    /// SEPARATE from `docs_`, so a PUT that replaces a document mid-show does not rewind it. An
    /// operator editing a cue while the show runs is the normal case, not an exception.
    std::unordered_map<std::string, playhead>                                 playheads_;
    std::int64_t                                                              revision_ = 0;
};

}}} // namespace caspar::core::timeline
