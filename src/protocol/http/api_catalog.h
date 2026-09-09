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

#include "api_context.h"
#include "api_status.h"

#include <string>

namespace caspar { namespace protocol { namespace http {

/// `GET /v1/catalog` -- what this server can be asked to PLAY.
///
/// The other half of parameter discovery. `/v1/tree/.../foreground/params` describes a producer
/// that is ALREADY RUNNING; this says what exists to run. A client that has both can offer the
/// choice and then draw the panel, with no per-effect knowledge compiled into it.
///
/// `kind` narrows it to one format -- `"ofx"` or `"isf"` -- and anything else is `unknown_path`
/// rather than an empty list, because an empty list is what a server with no plug-ins installed
/// correctly returns and the two must not look the same.
api_reply catalog(const api_context& ctx, const std::string& kind);

}}} // namespace caspar::protocol::http
