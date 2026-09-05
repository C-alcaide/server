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

    /// The same channel's CONCRETE stage. A batch needs it, because `core::stage_delayed`
    /// wraps a `stage` rather than a `stage_base` -- and that is what gives a batch its
    /// cross-channel atomicity, so the narrower type is worth carrying.
    std::function<std::shared_ptr<core::stage>(int)> concrete_stage;

    /// How many channels exist. A batch has to know before it starts, because it takes a
    /// delayed stage for every one of them.
    std::function<int()> channel_count;

    /// Run one AMCP command line and report its reply code and text.
    ///
    /// Only for the actions that need a producer built from a string -- `PLAY` and `LOAD`
    /// with a clip. Everything the API can do itself, it does itself, through `stage_base`.
    ///
    /// It is a function rather than a repository pointer for the reason the whole file
    /// exists: `protocol_http` links core and common only, and `amcp_command_repository`
    /// would drag the AMCP command layer onto its include path. The shell owns both and
    /// can bridge them in ten lines.
    struct amcp_reply
    {
        int          code = 0; //< the AMCP reply code, e.g. 202; 0 if it could not be parsed
        std::wstring text;
    };
    std::function<amcp_reply(const std::wstring& command)> amcp;
};

}}} // namespace caspar::protocol::http
