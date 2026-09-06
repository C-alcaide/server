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

#include <core/frame/transform_fields.h>
#include <core/monitor/monitor.h>
#include <core/producer/stage.h>

#include <functional>
#include <memory>
#include <string>

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

    /// Write one field of one stage object -- a screen or a camera -- on one channel.
    ///
    /// A bridge for the same reason `amcp` above is one, and it is the sharper case: the
    /// previz renderer lives in `accelerator`, which `protocol_http` does not link and must
    /// not. The shell owns both and does the `dynamic_cast` in a dozen lines.
    ///
    /// `object` is `"camera"`, `"view_camera"`, or `"screen/<name>"`; `field` is a row's `path`
    /// out of `core::fields::screen_fields()` / `camera_fields()`. A NAME rather than a
    /// `field_meta*`, for two reasons: `field_meta` deliberately carries no accessors -- that is
    /// what the descriptor split bought, and the typed row is `typed_field<screen_meta>` or
    /// `typed_field<previz_camera>` depending on the object -- and a plain string leaves no
    /// pointer whose lifetime has to cross this boundary. The caller has already validated type,
    /// arity and range against the descriptor.
    ///
    /// GOES THROUGH THE RENDERER'S MUTATORS rather than writing the struct, and that is the
    /// point rather than an implementation detail: each mutator also re-applies the mesh
    /// transform and calls `update_projections()`. A write that skipped them would set a value
    /// that changes nothing on screen -- 202 and no picture, the exact shape of the
    /// `MIXER EXPOSURE` allowlist defect.
    struct stage_write
    {
        bool applied = false;
        /// The renderer was reached and DECLINED: the mutator ran and the field still does not
        /// hold what was asked for. A separate flag from `applied` because a client acts on it
        /// differently -- this one means "your value was rejected", not "your path was wrong".
        ///
        /// It is decided in the bridge rather than by the caller, and it has to be: the caller
        /// would have to compare a read-back against the raw JSON operand, and those differ
        /// harmlessly wherever the descriptor canonicalises. An integer field sent as JSON `3`
        /// arrives as a double and reads back as an int32; an enumeration set by ordinal reads
        /// back as its name; every float member widens. The bridge compares INTENDED against
        /// ACTUAL, both read through the same accessor, so only a real refusal shows up.
        bool declined = false;
        /// Filled when `applied` is false. Distinguishes the cases a client must tell apart:
        /// no such channel, no previz on this channel, no such screen, or a field that has no
        /// mutator in the renderer.
        std::string             reason;
        core::monitor::vector_t previous;
        core::monitor::vector_t intended;
        core::monitor::vector_t written;
    };
    std::function<stage_write(int                            channel,
                              const std::string&             object,
                              const std::string&             field,
                              const core::monitor::vector_t& value)>
        set_stage_field;
};

}}} // namespace caspar::protocol::http
