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

  private:
    mutable std::mutex                                                        lock_;
    std::unordered_map<std::string, std::shared_ptr<const stored_timeline>>   docs_;
    std::int64_t                                                              revision_ = 0;
};

}}} // namespace caspar::core::timeline
