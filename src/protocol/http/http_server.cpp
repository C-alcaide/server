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

#include "http_server.h"

#include "api_status.h"
#include "api_tree.h"
#include "boost_prelude.h"

#include <common/except.h>
#include <common/executor.h>
#include <common/log.h>
#include <common/utf.h>

#include <boost/asio/io_context.hpp>

#include <atomic>
#include <string>

namespace caspar { namespace protocol { namespace http {

namespace {

/// Percent-decoding, because a path segment can legitimately contain one. Anything
/// malformed is left as written rather than rejected: a stray `%` in a path is going to
/// fail as `unknown_path` a moment later with a far more useful message than "bad escape".
std::string url_decode(std::string_view in)
{
    std::string out;
    out.reserve(in.size());
    for (size_t i = 0; i < in.size(); ++i) {
        if (in[i] == '%' && i + 2 < in.size()) {
            const auto hex = std::string(in.substr(i + 1, 2));
            char*      end = nullptr;
            const auto v   = std::strtol(hex.c_str(), &end, 16);
            if (end && *end == '\0') {
                out.push_back(static_cast<char>(v));
                i += 2;
                continue;
            }
        }
        out.push_back(in[i] == '+' ? ' ' : in[i]);
    }
    return out;
}

struct request_target
{
    std::string path;
    std::string query;
};

request_target split_target(std::string_view target)
{
    const auto q = target.find('?');
    if (q == std::string_view::npos)
        return {url_decode(target), std::string()};
    return {url_decode(target.substr(0, q)), std::string(target.substr(q + 1))};
}

bool starts_with(const std::string& s, const char* p)
{
    const std::string pp(p);
    return s.size() >= pp.size() && s.compare(0, pp.size(), pp) == 0;
}

} // namespace

struct http_server::impl : public std::enable_shared_from_this<http_server::impl>
{
    std::shared_ptr<boost::asio::io_context> io_context_;
    std::shared_ptr<state_hub>               hub_;
    http_config                              config_;
    std::string                              server_name_;

    tcp::acceptor acceptor_;

    // Every request body runs here, never on an `io_context` thread. See the header.
    executor api_executor_{L"http-api"};

    std::atomic<int> subscriptions_{0};

    impl(std::shared_ptr<boost::asio::io_context> io_context, std::shared_ptr<state_hub> hub, http_config config)
        : io_context_(std::move(io_context))
        , hub_(std::move(hub))
        , config_(std::move(config))
        , server_name_(u8(config_.name))
        , acceptor_(*io_context_)
    {
        const auto address = config_.host.empty() ? asio::ip::make_address("0.0.0.0")
                                                  : asio::ip::make_address(u8(config_.host));
        const tcp::endpoint endpoint(address, config_.port);

        acceptor_.open(endpoint.protocol());
        acceptor_.set_option(asio::socket_base::reuse_address(true));
        acceptor_.bind(endpoint);
        acceptor_.listen(asio::socket_base::max_listen_connections);
    }

    void start() { accept(); }

    void stop()
    {
        boost::system::error_code ec;
        acceptor_.close(ec);
    }

    void accept()
    {
        auto self = shared_from_this();
        acceptor_.async_accept(asio::make_strand(*io_context_), [self](boost::system::error_code ec, tcp::socket sock) {
            if (ec) {
                // A closed acceptor during shutdown is the normal path out of this loop,
                // and logging it as an error every time the server stops would train the
                // reader to ignore the line that matters.
                if (ec != asio::error::operation_aborted)
                    CASPAR_LOG(warning) << L"[http-api] accept failed: " << u16(ec.message());
                return;
            }
            self->run_session(std::move(sock));
            self->accept();
        });
    }

    // ---------------------------------------------------------------------------------
    // Routing
    // ---------------------------------------------------------------------------------

    /// Answer one request. Runs on `api_executor_`, so it may take as long as it takes.
    api_reply route(bhttp::verb method, const std::string& path, const std::string& query)
    {
        // `?HOST_INFO` is OSCQuery's capability probe and is answered on ANY path, which is
        // what the spec says and what makes it usable as a liveness check without knowing
        // the address space first.
        if (query == "HOST_INFO")
            return api_reply::ok_with(host_info(config_, subscriptions_.load()));

        if (method != bhttp::verb::get && method != bhttp::verb::head)
            return api_reply::fail(api_code::bad_request,
                                   "only GET is implemented in this build; writes arrive in a later commit");

        if (path == "/" || path == "/v1")
            return api_reply::ok_with(host_info(config_, subscriptions_.load()));

        if (path == "/v1/tree")
            return api_reply::ok_with(build_tree(*hub_, config_));

        if (starts_with(path, "/v1/tree/"))
            return tree_at(*hub_, config_, path.substr(std::string("/v1/tree").size()));

        if (starts_with(path, "/v1/value/"))
            return read_value(*hub_, path.substr(std::string("/v1/value").size()));

        return api_reply::fail(api_code::unknown_path, "no such endpoint: " + path);
    }

    /// The HTTP status for an application code.
    ///
    /// Application failures are 200 with a non-zero `status.code` -- see `api_status.h` for
    /// why. The exceptions are the two that really are transport: an endpoint that does not
    /// exist AT ALL (as opposed to a state path that does not), and authentication.
    static bhttp::status http_status_for(const api_reply& r, const std::string& path)
    {
        if (r.code == api_code::unauthorized)
            return bhttp::status::unauthorized;
        if (r.code == api_code::unknown_path && !starts_with(path, "/v1/tree") && !starts_with(path, "/v1/value"))
            return bhttp::status::not_found;
        return bhttp::status::ok;
    }

    // ---------------------------------------------------------------------------------
    // Session
    // ---------------------------------------------------------------------------------

    void run_session(tcp::socket sock)
    {
        auto self   = shared_from_this();
        auto stream = std::make_shared<beast::tcp_stream>(std::move(sock));
        auto buffer = std::make_shared<beast::flat_buffer>();
        read(std::move(stream), std::move(buffer));
    }

    void read(std::shared_ptr<beast::tcp_stream> stream, std::shared_ptr<beast::flat_buffer> buffer)
    {
        auto self = shared_from_this();
        auto req  = std::make_shared<bhttp::request<bhttp::string_body>>();

        // A generous timeout rather than none: a half-open connection from a client that
        // lost power would otherwise hold a socket for the life of the server.
        stream->expires_after(std::chrono::seconds(30));

        bhttp::async_read(*stream,
                          *buffer,
                          *req,
                          [self, stream, buffer, req](boost::system::error_code ec, std::size_t) {
                              if (ec) {
                                  if (ec != bhttp::error::end_of_stream && ec != asio::error::operation_aborted)
                                      CASPAR_LOG(debug) << L"[http-api] read: " << u16(ec.message());
                                  self->close(stream);
                                  return;
                              }
                              self->handle(stream, buffer, req);
                          });
    }

    void handle(const std::shared_ptr<beast::tcp_stream>&                  stream,
                const std::shared_ptr<beast::flat_buffer>&                 buffer,
                const std::shared_ptr<bhttp::request<bhttp::string_body>>& req)
    {
        auto       self   = shared_from_this();
        const auto target = split_target(std::string(req->target()));
        const auto method = req->method();
        const auto keep   = req->keep_alive();
        const auto ver    = req->version();

        // Off the io_context thread from here. Nothing below touches the socket until the
        // reply is posted back onto its strand.
        api_executor_.begin_invoke([self, stream, buffer, target, method, keep, ver]() {
            api_reply reply;
            try {
                reply = self->route(method, target.path, target.query);
            } catch (...) {
                CASPAR_LOG_CURRENT_EXCEPTION();
                reply = api_reply::fail(api_code::internal, "unhandled exception building the reply");
            }

            const auto status = http_status_for(reply, target.path);
            auto       body   = json::serialize(json::value(envelope(reply, self->server_name_)));

            asio::post(stream->get_executor(),
                       [self, stream, buffer, status, body = std::move(body), method, keep, ver]() mutable {
                           self->write(stream, buffer, status, std::move(body), method, keep, ver);
                       });
        });
    }

    void write(const std::shared_ptr<beast::tcp_stream>&  stream,
               const std::shared_ptr<beast::flat_buffer>& buffer,
               bhttp::status                              status,
               std::string                                body,
               bhttp::verb                                method,
               bool                                       keep_alive,
               unsigned                                   version)
    {
        auto self = shared_from_this();
        auto res  = std::make_shared<bhttp::response<bhttp::string_body>>(status, version);
        res->set(bhttp::field::server, "CasparCG");
        res->set(bhttp::field::content_type, "application/json");
        // The tree is fetched by a browser-based client as often as by a native one, and
        // without this every such fetch fails in a way that looks like the server is down.
        res->set(bhttp::field::access_control_allow_origin, "*");
        res->keep_alive(keep_alive);
        if (method != bhttp::verb::head)
            res->body() = std::move(body);
        res->prepare_payload();

        bhttp::async_write(*stream,
                           *res,
                           [self, stream, buffer, res, keep_alive](boost::system::error_code ec, std::size_t) {
                               if (ec) {
                                   CASPAR_LOG(debug) << L"[http-api] write: " << u16(ec.message());
                                   self->close(stream);
                                   return;
                               }
                               if (!keep_alive) {
                                   self->close(stream);
                                   return;
                               }
                               self->read(stream, buffer);
                           });
    }

    static void close(const std::shared_ptr<beast::tcp_stream>& stream)
    {
        boost::system::error_code ec;
        stream->socket().shutdown(tcp::socket::shutdown_send, ec);
    }
};

http_server::http_server(std::shared_ptr<boost::asio::io_context> io_context,
                         std::shared_ptr<state_hub>               hub,
                         http_config                              config)
    : impl_(std::make_shared<impl>(std::move(io_context), std::move(hub), std::move(config)))
{
    impl_->start();
}

http_server::~http_server() { impl_->stop(); }

unsigned short http_server::port() const { return impl_->config_.port; }

}}} // namespace caspar::protocol::http
