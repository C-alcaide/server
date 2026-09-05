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
#include "http_config.h"

#include <chrono>
#include <map>
#include <mutex>
#include <string>

namespace caspar { namespace protocol { namespace http {

/// Challenge/response over a plain socket.
///
/// `GET /v1/auth` returns a per-connection-attempt challenge and the server's salt. The
/// client answers on every subsequent request with
///
///     Authorization: Caspar <challenge>:<hex(sha256(hex(sha256(password+salt)) + challenge))>
///
/// The client echoes the challenge it is answering so a wrong answer can be attributed to
/// one, and therefore consume it.
///
/// **What this is for, and what it is not.** It keeps the password off the wire on a plain
/// HTTP connection, which is worth having on a studio LAN. It is NOT a password store --
/// there is no key stretching, and `<password>` sits in the config in plain text, so
/// anybody who can read the config has the password whatever happens here. And it is not
/// confidentiality: every request and every event after the handshake is still cleartext.
/// This buys "not shouting the password"; it does not buy a safe path off-segment, and the
/// feature doc says so where an operator will read it.
class auth_state
{
  public:
    explicit auth_state(const http_config& cfg);

    bool enabled() const { return enabled_; }

    /// A fresh challenge, and the salt to use with it. Both hex.
    api_reply issue_challenge();

    /// Check one `Authorization` header value. Empty is "no header at all".
    ///
    /// A challenge is single-use and is consumed whether the answer was right or wrong, so
    /// a wrong answer cannot be retried against the same challenge and a right one cannot
    /// be replayed. That makes a client fetch `/v1/auth` per request, which for an API
    /// whose read side is a WebSocket subscription is a cost worth paying.
    bool check(const std::string& header);

  private:
    bool        enabled_ = false;
    std::string salt_;
    std::string expected_; //< hex(sha256(password + salt)), the value a client also computes

    mutable std::mutex                                            mutex_;
    std::map<std::string, std::chrono::steady_clock::time_point>   live_;
    std::uint64_t                                                  counter_ = 0;
};

}}} // namespace caspar::protocol::http
