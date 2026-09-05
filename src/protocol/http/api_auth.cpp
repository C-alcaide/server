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

#include "api_auth.h"

#include <common/sha256.h>
#include <common/utf.h>

#include <algorithm>
#include <random>

namespace caspar { namespace protocol { namespace http {

namespace {

/// A challenge lives this long. Long enough for a slow client on a busy LAN, short enough
/// that the set of outstanding challenges cannot grow without bound from a client that asks
/// and never answers.
constexpr auto challenge_lifetime = std::chrono::seconds(30);

std::string random_hex(std::size_t bytes)
{
    // `random_device` rather than a seeded PRNG: a predictable challenge is the same as no
    // challenge, and this runs once per request rather than once per frame.
    std::random_device                           rd;
    std::uniform_int_distribution<unsigned>      dist(0, 255);
    static const char*                           digits = "0123456789abcdef";
    std::string                                  out;
    out.reserve(bytes * 2);
    for (std::size_t i = 0; i < bytes; ++i) {
        const auto b = static_cast<unsigned char>(dist(rd));
        out.push_back(digits[b >> 4]);
        out.push_back(digits[b & 0x0f]);
    }
    return out;
}

/// Compare without an early exit on the first differing byte.
///
/// The timing signal from `==` on a 64-character hex string is not a realistic attack over
/// a studio LAN, and writing the comparison this way costs one line -- so the reason not to
/// is only that it did not occur to anyone.
bool constant_time_equal(const std::string& a, const std::string& b)
{
    if (a.size() != b.size())
        return false;
    unsigned char diff = 0;
    for (std::size_t i = 0; i < a.size(); ++i)
        diff |= static_cast<unsigned char>(a[i] ^ b[i]);
    return diff == 0;
}

} // namespace

auth_state::auth_state(const http_config& cfg)
{
    enabled_ = cfg.auth == L"password";
    if (!enabled_)
        return;

    // The salt is per SERVER RUN, not per install. It is not secret -- the client is told
    // it -- and regenerating it means a captured exchange is worthless against the next
    // start. The cost is that a client cannot cache its derived value across a restart,
    // which for a handshake this cheap is not a cost.
    salt_     = random_hex(16);
    expected_ = sha256::hex_of(u8(cfg.password) + salt_);
}

api_reply auth_state::issue_challenge()
{
    if (!enabled_) {
        json::object r;
        r["auth"] = "off";
        return api_reply::ok_with(std::move(r));
    }

    const auto now       = std::chrono::steady_clock::now();
    auto       challenge = random_hex(16);

    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto it = live_.begin(); it != live_.end();)
            it = (now - it->second > challenge_lifetime) ? live_.erase(it) : std::next(it);
        live_.emplace(challenge, now);
        ++counter_;
    }

    json::object r;
    r["auth"]      = "password";
    r["salt"]      = salt_;
    r["challenge"] = challenge;
    r["algorithm"] = "Authorization: Caspar <challenge>:<sha256(sha256(password+salt)+challenge)>, hex, lower case";
    r["expires_s"] = static_cast<std::int64_t>(challenge_lifetime.count());
    return api_reply::ok_with(std::move(r));
}

bool auth_state::check(const std::string& header)
{
    if (!enabled_)
        return true;

    // `Authorization: Caspar <challenge>:<answer>`.
    //
    // The client names the challenge it is answering, and that is not decoration. The first
    // version left it out and tried every live challenge in turn -- which works, and means
    // a WRONG answer cannot be attributed to a challenge, so it cannot consume one. The
    // challenge then stayed live for its whole 30 seconds and could be guessed at
    // repeatedly. Measured: a deliberately corrupted answer was rejected, and the correct
    // answer to the SAME challenge was still accepted afterwards.
    static const std::string prefix = "Caspar ";
    if (header.size() <= prefix.size() || header.compare(0, prefix.size(), prefix) != 0)
        return false;

    auto body = header.substr(prefix.size());
    body.erase(0, body.find_first_not_of(" 	"));
    const auto colon = body.find(':');
    if (colon == std::string::npos)
        return false;

    auto challenge = body.substr(0, colon);
    auto answer    = body.substr(colon + 1);
    const auto lower = [](std::string& x) {
        std::transform(x.begin(), x.end(), x.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    };
    lower(challenge);
    lower(answer);

    // Consume it FIRST, whatever the answer turns out to be. One attempt per challenge, so
    // a wrong answer costs the attacker a round trip to `/v1/auth` rather than nothing.
    const auto now   = std::chrono::steady_clock::now();
    bool       live  = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto it = live_.begin(); it != live_.end();)
            it = (now - it->second > challenge_lifetime) ? live_.erase(it) : std::next(it);

        const auto it = live_.find(challenge);
        if (it != live_.end()) {
            live = true;
            live_.erase(it);
        }
    }
    if (!live)
        return false;

    return constant_time_equal(answer, sha256::hex_of(expected_ + challenge));
}

}}} // namespace caspar::protocol::http
