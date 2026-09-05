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

#include "api_status.h"
#include "state_hub.h"

#include <chrono>
#include <map>
#include <string>
#include <vector>

namespace caspar { namespace protocol { namespace http {

/// Does `path` fall under `prefix`, on SEGMENT boundaries?
///
/// A plain `starts_with` makes `/channel/1` match `/channel/10`, so a client subscribed to
/// one channel silently receives another's. That is not a hypothetical: with ten channels
/// configured, `/channel/1/` is a prefix of nothing surprising but `/channel/1` matches
/// every path on channels 1 and 10 through 19.
bool prefix_matches(const std::string& path, const std::string& prefix);

/// One client's subscription. Owned by the WebSocket session; touched only on the API
/// executor, never on a tick thread and never on two threads at once.
struct subscription
{
    std::string              id;
    std::vector<std::string> prefixes;
    int                      throttle_ms       = 0;
    bool                     repetition_filter = true;

    std::chrono::steady_clock::time_point last_sent{};

    /// What this subscriber was last told, per path. Per-connection rather than shared:
    /// two clients with different throttles are at different points in time, so a single
    /// server-side "last published" set would send one of them a diff against the other's
    /// view.
    std::map<std::string, core::monitor::vector_t> last_values;
};

/// Collect this subscription's pending events.
///
/// Returns an empty object when there is nothing to send -- either nothing changed, or the
/// throttle interval has not elapsed. `frame` carries each touched channel's frame number,
/// so a client can say WHICH frame a value belongs to and correlate it with a picture.
json::object collect_events(subscription& sub, const state_hub& hub, const std::string& server_name);

/// Parse a client's `subscribe` message into a subscription. Rejects a prefix that does not
/// start `/channel/`, and more than `max_prefixes` of them.
api_reply parse_subscribe(const json::value& msg, int max_prefixes, subscription& out);

}}} // namespace caspar::protocol::http
