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

// The ONLY place Beast and Boost.JSON are included.
//
// Two reasons it is one file rather than an include in each translation unit:
//
//   * The tree builds with `/W4 /WX` on MSVC and `-Werror` on gcc, and Boost's include
//     path is marked SYSTEM but MSVC still reports template-instantiation warnings at the
//     point of USE rather than the point of include. Wrapping the includes here catches
//     what can be caught at include time; anything left is handled per-target in
//     CMakeLists.txt, which is where a reader will look for it.
//   * `protocol`'s precompiled header is force-included into every one of its translation
//     units, and adding Beast to it would drag ~4 s of template instantiation into every
//     file in the library -- and, in this tree, would need the full PCH sweep from
//     CLAUDE.md (touch every source, delete both the .pch AND its .obj) on every edit.
//     `protocol_http` therefore has no PCH and Beast never reaches one.

#if defined(_MSC_VER)
#pragma warning(push)
// 4100 unreferenced formal parameter, 4127 conditional expression is constant,
// 4244/4245/4267 conversion and signed/unsigned mismatch in Asio's integer plumbing,
// 4324 structure padded due to alignas, 4456/4457/4459 declaration hides another,
// 4702 unreachable code, 4834 discarding a [[nodiscard]] return, 4996 deprecation.
#pragma warning(disable : 4100 4127 4189 4244 4245 4267 4324 4456 4457 4459 4702 4834 4996)
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

#include <boost/asio/dispatch.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/post.hpp>
#include <boost/asio/strand.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/version.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/json.hpp>

#if defined(_MSC_VER)
#pragma warning(pop)
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

namespace caspar { namespace protocol { namespace http {

namespace beast     = boost::beast;
namespace bhttp     = boost::beast::http;
namespace websocket = boost::beast::websocket;
namespace asio      = boost::asio;
namespace json      = boost::json;
using tcp           = boost::asio::ip::tcp;

}}} // namespace caspar::protocol::http
