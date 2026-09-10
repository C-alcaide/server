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

#include "timeline_store.h"

#include <algorithm>

namespace caspar { namespace core { namespace timeline {

std::shared_ptr<const stored_timeline> timeline_store::put(timeline_document doc)
{
    auto entry = std::make_shared<stored_timeline>();

    // The document's own revision is the STORE's counter at the moment it lands, not something
    // the client sends. A client-supplied revision would let two clients disagree about which
    // version is newer, and the whole point of the number is that a third party can order them.
    {
        std::lock_guard<std::mutex> l(lock_);
        doc.revision = ++revision_;
    }

    entry->document = std::move(doc);
    entry->resolved = resolve(entry->document, entry->triggers);

    const auto name = entry->document.name;
    {
        std::lock_guard<std::mutex> l(lock_);
        docs_[name] = entry;
    }
    return entry;
}

std::shared_ptr<const stored_timeline> timeline_store::get(const std::string& name) const
{
    std::lock_guard<std::mutex> l(lock_);
    const auto                  it = docs_.find(name);
    return it == docs_.end() ? nullptr : it->second;
}

bool timeline_store::erase(const std::string& name)
{
    std::lock_guard<std::mutex> l(lock_);
    if (docs_.erase(name) == 0)
        return false;
    // THE PLAYHEAD GOES WITH IT. Leaving it would mean a document PUT again under the same name
    // resumes a show that no longer exists -- from a position no client asked for and none can
    // see, because nothing publishes a playhead for a document that is not loaded.
    playheads_.erase(name);
    ++revision_;
    return true;
}

std::vector<std::string> timeline_store::names() const
{
    std::vector<std::string>    out;
    std::lock_guard<std::mutex> l(lock_);
    out.reserve(docs_.size());
    for (const auto& kv : docs_)
        out.push_back(kv.first);
    std::sort(out.begin(), out.end());
    return out;
}

std::shared_ptr<const stored_timeline> timeline_store::retrigger(const std::string& name,
                                                                 trigger_log        triggers)
{
    std::shared_ptr<const stored_timeline> old;
    {
        std::lock_guard<std::mutex> l(lock_);
        const auto                  it = docs_.find(name);
        if (it == docs_.end())
            return nullptr;
        old = it->second;
    }

    // Resolved OUTSIDE the lock. A large document's resolution is the one expensive thing in
    // this file, and holding the lock across it would block every tick that wants to read any
    // document -- including ones on other channels that have nothing to do with this trigger.
    auto entry      = std::make_shared<stored_timeline>();
    entry->document = old->document;
    entry->triggers = std::move(triggers);
    entry->resolved = resolve(entry->document, entry->triggers);

    {
        std::lock_guard<std::mutex> l(lock_);
        const auto                  it = docs_.find(name);
        // Someone may have PUT or erased it while this was resolving. Their write is newer, so
        // it wins: re-resolving against a stale document and storing the result would silently
        // undo a PUT that had already been acknowledged.
        if (it == docs_.end() || it->second != old)
            return it == docs_.end() ? nullptr : it->second;
        entry->document.revision = ++revision_;
        it->second               = entry;
    }
    return entry;
}

void timeline_store::publish_playhead(const std::string& name, const playhead& p)
{
    std::lock_guard<std::mutex> guard(lock_);
    playheads_[name] = p;
}

std::optional<playhead> timeline_store::playhead_of(const std::string& name) const
{
    std::lock_guard<std::mutex> guard(lock_);
    const auto                  it = playheads_.find(name);
    if (it == playheads_.end())
        return std::nullopt;
    return it->second;
}

std::int64_t timeline_store::revision() const
{
    std::lock_guard<std::mutex> l(lock_);
    return revision_;
}

}}} // namespace caspar::core::timeline
