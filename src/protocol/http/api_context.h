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

#include <core/producer/stage.h>

#include <functional>
#include <memory>

namespace caspar { namespace protocol { namespace http {

/// What the API is allowed to reach.
///
/// A lookup function rather than the shell's `std::vector<amcp::channel_context>`, for one
/// reason that is worth the indirection: `channel_context` lives in `protocol/amcp`, and
/// `protocol_http` links `common` and `core` only. Taking it would put the whole AMCP
/// command layer on this library's include path to obtain one `stage_base` pointer, and
/// would make the API depend on AMCP in exactly the direction the design is trying to
/// avoid -- the two façades are meant to be siblings over one state, not a stack.
struct api_context
{
    /// The stage for a 1-based channel index, or nullptr if there is no such channel.
    std::function<std::shared_ptr<core::stage_base>(int)> stage;
};

}}} // namespace caspar::protocol::http
