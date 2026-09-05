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

#include "http_config.h"
#include "state_hub.h"

#include <memory>

// Beast is deliberately absent from this header: `server.cpp` includes it, and dragging
// Asio's templates into the shell's translation units would cost minutes of compile time
// for a class it only constructs once. `io_context` is forward-declared for the same
// reason -- the shell already has one, and the API shares it.
namespace boost { namespace asio {
class io_context;
}} // namespace boost::asio

namespace caspar { namespace protocol { namespace http {

class ws_session;

/// The control API listener.
///
/// Accept, read and write run on the shell's shared `io_context` -- exactly the shape
/// `AsyncEventServer` already uses for AMCP. Request HANDLING does not: every request body
/// runs on a dedicated executor thread and the reply is posted back. That is not
/// decoration. A tree serialisation is hundreds of kilobytes of JSON, and a write blocks on
/// a stage future; doing either on an `io_context` thread would stall AMCP and OSC, which
/// share it.
class http_server
{
  public:
    http_server(std::shared_ptr<boost::asio::io_context> io_context,
                std::shared_ptr<state_hub>               hub,
                http_config                              config);
    ~http_server();

    http_server(const http_server&)            = delete;
    http_server& operator=(const http_server&) = delete;

    unsigned short port() const;

  private:
    // The WebSocket session reaches into the server's executor, config and hub. It is part
    // of the same object in every way except allocation -- it just cannot be a member,
    // because there is one per connection.
    friend class ws_session;

    struct impl;
    std::shared_ptr<impl> impl_;
};

}}} // namespace caspar::protocol::http
