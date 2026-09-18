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

#include <core/monitor/monitor.h>

#include <atomic>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace caspar { namespace protocol { namespace http {

/// Where the tick hands the API its snapshots.
///
/// The channel publishes once per frame from the tick thread; readers are HTTP requests on
/// whatever thread the executor runs them on. The hub holds one immutable snapshot per
/// channel behind an atomic, so publishing is a pointer store and reading is a pointer
/// load -- no lock on the tick thread, and no reader can ever see a half-built map.
class state_hub
{
  public:
    using snapshot = std::shared_ptr<const core::monitor::state>;

    /// Called from the channel tick. Must not block: it stores a pointer and notifies.
    void publish(int channel_index, snapshot s);

    /// The last snapshot for a channel, or nullptr if that channel has never ticked.
    snapshot get(int channel_index) const;

    /// Every channel index that has published at least once, ascending.
    std::vector<int> channels() const;

    /// A root that is not a channel -- `diag`, the diagnostics publication. Same contract as
    /// `publish`: a pointer store from whatever thread produces it, and the observer fires
    /// (with index 0, which no channel has) so the events fan-out runs. An EMPTY snapshot
    /// removes the root: the tree stops listing it and a subscriber sees its leaves reverted.
    void     publish_root(const std::string& name, snapshot s);
    snapshot get_root(const std::string& name) const;

    /// Called on the TICK THREAD immediately after a snapshot is stored, with the channel
    /// index. Whatever it does must be a hand-off -- a post onto another executor -- and
    /// never work: this runs inside the frame.
    using observer_t = std::function<void(int)>;
    void set_observer(observer_t observer);

  private:
    // A map rather than a vector because channel indices are 1-based and need not be
    // contiguous -- a config can define channels 1 and 3.
    mutable std::mutex      mutex_;
    std::map<int, snapshot> latest_;
    std::map<std::string, snapshot> roots_;
    observer_t              observer_;
};

}}} // namespace caspar::protocol::http
