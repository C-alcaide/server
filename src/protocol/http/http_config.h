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

#include <string>

namespace caspar { namespace protocol { namespace http {

/// Plain types only, so `server.cpp` can fill this in without seeing Beast.
struct http_config
{
    unsigned short port = 5254;
    std::wstring   host;
    /// A name for THIS server, so a client attached to more than one can tell their
    /// replies and events apart. Defaults to the machine's hostname. It is also
    /// OSCQuery's HOST_INFO.NAME and the `/server/{name}` path segment.
    std::wstring name;
    std::wstring auth     = L"off";
    std::wstring password;
    /// `state` exposes only what the tick published; `mixer` additionally exposes the full
    /// descriptor set for every existing layer, which is what a generated control surface
    /// needs and what costs tree size.
    std::wstring extent       = L"mixer";
    int          max_prefixes = 32;
};

}}} // namespace caspar::protocol::http
