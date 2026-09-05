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

#include "state_hub.h"

namespace caspar { namespace protocol { namespace http {

void state_hub::publish(int channel_index, snapshot s)
{
    // Called from the channel tick, once per frame per channel. The lock is held for a
    // pointer assignment against a map that is only ever grown at startup, so it is
    // uncontended in steady state -- and it is a lock rather than a per-channel atomic
    // only because the map itself has to be safe to iterate while a late channel is added.
    observer_t observer;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        latest_[channel_index] = std::move(s);
        observer               = observer_;
    }

    // Outside the lock. The observer posts onto another executor, and calling it under the
    // hub's own mutex would let that executor's work deadlock against a reader here.
    if (observer)
        observer(channel_index);
}

void state_hub::set_observer(observer_t observer)
{
    std::lock_guard<std::mutex> lock(mutex_);
    observer_ = std::move(observer);
}

state_hub::snapshot state_hub::get(int channel_index) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto                        it = latest_.find(channel_index);
    return it == latest_.end() ? nullptr : it->second;
}

std::vector<int> state_hub::channels() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<int>            result;
    result.reserve(latest_.size());
    for (const auto& p : latest_)
        result.push_back(p.first);
    return result;
}

}}} // namespace caspar::protocol::http
