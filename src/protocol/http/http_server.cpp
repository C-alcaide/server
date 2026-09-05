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

#include "api_action.h"
#include "api_auth.h"
#include "api_events.h"
#include "api_status.h"
#include "api_tree.h"
#include "api_value.h"
#include "json_state.h"
#include "boost_prelude.h"

#include <common/except.h>
#include <common/executor.h>
#include <common/log.h>
#include <common/utf.h>

#include <boost/asio/io_context.hpp>

#include <atomic>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

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

/// One live `/v1/events` connection.
///
/// The subscription and its per-connection diff live here rather than in the hub, because
/// two clients with different throttles are at DIFFERENT POINTS IN TIME: a shared
/// server-side "last published" set would hand one of them a diff computed against the
/// other's view, and the value it skipped would never be sent again.
///
/// Lifetime is by `shared_ptr` from the session's own async operations; the server holds
/// only a `weak_ptr`, so a client that disappears is collected on the next fan-out with no
/// explicit deregistration to get wrong.
class ws_session : public std::enable_shared_from_this<ws_session>
{
  public:
    ws_session(websocket::stream<beast::tcp_stream> ws, std::shared_ptr<http_server::impl> owner)
        : ws_(std::move(ws))
        , owner_(std::move(owner))
    {
    }

    void run(bhttp::request<bhttp::string_body> req);
    void notify();
    void close();

    bool subscribed() const { return subscribed_.load(std::memory_order_acquire); }

  private:
    void read();
    void handle_text(std::string text);
    void send(std::string payload);
    void write_next();

    websocket::stream<beast::tcp_stream> ws_;
    std::shared_ptr<http_server::impl>   owner_;
    beast::flat_buffer                   buffer_;

    // Touched only on the API executor. `notify()` is called from anywhere and only posts.
    subscription      sub_;
    std::atomic<bool> subscribed_{false};
    std::atomic<bool> collecting_{false};

    // Serialises writes: Beast permits exactly one outstanding write per stream, and a
    // 25 Hz fan-out will absolutely produce a second one while the first is in flight.
    std::deque<std::string> outbox_;
    bool                    writing_ = false;
    std::atomic<bool>       closed_{false};
};

struct http_server::impl : public std::enable_shared_from_this<http_server::impl>
{
    std::shared_ptr<boost::asio::io_context> io_context_;
    std::shared_ptr<state_hub>               hub_;
    http_config                              config_;
    api_context                              context_;
    std::string                              server_name_;

    tcp::acceptor acceptor_;
    auth_state    auth_;

    // Every request body runs here, never on an `io_context` thread. See the header.
    executor api_executor_{L"http-api"};

    std::mutex                             ws_mutex_;
    std::vector<std::weak_ptr<ws_session>>  ws_sessions_;

    /// Batches waiting for a frame. Touched only on the API executor -- posted there by
    /// the request that created one, and drained there by the per-tick fan-out.
    std::vector<batch_plan> pending_batches_;

    /// Set while a fan-out is queued. Without it a 25 Hz tick on four channels queues 100
    /// tasks a second onto the API executor whether or not the previous one has run, and a
    /// momentarily slow subscriber turns into an unbounded queue.
    std::atomic<bool> fanout_queued_{false};

    impl(std::shared_ptr<boost::asio::io_context> io_context,
         std::shared_ptr<state_hub>               hub,
         http_config                              config,
         api_context                              context)
        : io_context_(std::move(io_context))
        , hub_(std::move(hub))
        , config_(std::move(config))
        , context_(std::move(context))
        , server_name_(u8(config_.name))
        , acceptor_(*io_context_)
        , auth_(config_)
    {
        const auto address = config_.host.empty() ? asio::ip::make_address("0.0.0.0")
                                                  : asio::ip::make_address(u8(config_.host));
        const tcp::endpoint endpoint(address, config_.port);

        acceptor_.open(endpoint.protocol());
        acceptor_.set_option(asio::socket_base::reuse_address(true));
        acceptor_.bind(endpoint);
        acceptor_.listen(asio::socket_base::max_listen_connections);
    }

    void start()
    {
        // The hub calls this on the TICK THREAD. It sets a flag and posts; everything real
        // happens on the API executor.
        std::weak_ptr<impl> weak = shared_from_this();
        hub_->set_observer([weak](int) {
            auto self = weak.lock();
            if (!self)
                return;
            if (self->fanout_queued_.exchange(true, std::memory_order_acq_rel))
                return;
            self->api_executor_.begin_invoke([self] {
                self->fanout_queued_.store(false, std::memory_order_release);
                self->fan_out();
            });
        });

        accept();
    }

    /// Resolve a deferred batch's target frame and queue it. On the API executor, which
    /// is also where `drain_batches` runs, so `pending_batches_` needs no lock.
    ///
    /// `in_frames` arrives negated from `validate_batch` -- the validator has no idea what
    /// frame the server is on, and giving it one would mean handing the whole hub to a
    /// function whose entire job is to touch nothing.
    api_reply park_batch(batch_plan plan)
    {
        std::int64_t now = 0;
        if (const auto snap = hub_->get(plan.reference_channel)) {
            for (const auto& kv : *snap) {
                if (kv.first == "frame" && !kv.second.empty()) {
                    const auto v = data_to_json(kv.second.front());
                    if (v.is_int64() || v.is_uint64() || v.is_double())
                        now = static_cast<std::int64_t>(v.to_number<double>());
                    break;
                }
            }
        }

        if (plan.at_frame < 0)
            plan.at_frame = now - plan.at_frame; //< in_frames, negated by the validator
        else if (plan.at_frame <= now)
            // Already gone. Refused rather than applied immediately: a cue that missed its
            // frame is a timing failure the operator has to know about, and quietly firing
            // it late is how a show ends up out of sync with nothing in the log.
            return api_reply::fail(api_code::bad_request,
                                   "at_frame " + std::to_string(plan.at_frame) + " is not in the future; channel " +
                                       std::to_string(plan.reference_channel) + " is on frame " + std::to_string(now));

        json::object r;
        r["scheduled"] = true;
        r["at_frame"]  = plan.at_frame;
        r["channel"]   = plan.reference_channel;
        r["now"]       = now;
        r["ops"]       = static_cast<std::int64_t>(plan.ops.size());

        pending_batches_.push_back(std::move(plan));
        return api_reply::ok_with(std::move(r));
    }

    /// Fire every parked batch whose frame has arrived. On the API executor, once per tick.
    void drain_batches()
    {
        if (pending_batches_.empty())
            return;

        std::vector<batch_plan> keep;
        keep.reserve(pending_batches_.size());

        for (auto& plan : pending_batches_) {
            std::int64_t now = -1;
            if (const auto snap = hub_->get(plan.reference_channel)) {
                for (const auto& kv : *snap) {
                    if (kv.first == "frame" && !kv.second.empty()) {
                        const auto v = data_to_json(kv.second.front());
                        if (v.is_int64() || v.is_uint64() || v.is_double())
                            now = static_cast<std::int64_t>(v.to_number<double>());
                        break;
                    }
                }
            }
            if (now < 0) {
                // The reference channel stopped publishing -- it was cleared, or the server
                // is shutting down. Drop the batch and say so rather than holding it for a
                // frame that will never arrive.
                CASPAR_LOG(warning) << L"[api] dropping a batch scheduled for frame " << plan.at_frame
                                    << L": channel " << plan.reference_channel << L" is not publishing";
                continue;
            }
            if (now < plan.at_frame) {
                keep.push_back(std::move(plan));
                continue;
            }

            if (now > plan.at_frame)
                CASPAR_LOG(warning) << L"[api] batch late by " << (now - plan.at_frame) << L" frame(s): scheduled for "
                                    << plan.at_frame << L", applying on " << now;

            const auto reply = apply_batch(context_, plan);
            if (reply.code != api_code::ok)
                CASPAR_LOG(error) << L"[api] a scheduled batch failed on frame " << now << L": "
                                  << u16(reply.message);
        }

        pending_batches_ = std::move(keep);
    }

    /// Give every live subscriber whatever its prefixes say it is owed. On the API
    /// executor.
    void fan_out()
    {
        drain_batches();

        std::vector<std::shared_ptr<ws_session>> live;
        {
            std::lock_guard<std::mutex> lock(ws_mutex_);
            live.reserve(ws_sessions_.size());
            for (auto it = ws_sessions_.begin(); it != ws_sessions_.end();) {
                if (auto s = it->lock()) {
                    live.push_back(std::move(s));
                    ++it;
                } else {
                    it = ws_sessions_.erase(it);
                }
            }
        }
        for (auto& s : live)
            s->notify();
    }

    void add_ws_session(const std::shared_ptr<ws_session>& s)
    {
        std::lock_guard<std::mutex> lock(ws_mutex_);
        ws_sessions_.push_back(s);
    }

    int count_subscriptions()
    {
        std::lock_guard<std::mutex> lock(ws_mutex_);
        int n = 0;
        for (auto it = ws_sessions_.begin(); it != ws_sessions_.end();) {
            if (auto s = it->lock()) {
                if (s->subscribed())
                    ++n;
                ++it;
            } else {
                it = ws_sessions_.erase(it);
            }
        }
        return n;
    }

    void stop()
    {
        boost::system::error_code ec;
        acceptor_.close(ec);
        hub_->set_observer(nullptr);

        std::vector<std::shared_ptr<ws_session>> live;
        {
            std::lock_guard<std::mutex> lock(ws_mutex_);
            for (auto& w : ws_sessions_)
                if (auto s = w.lock())
                    live.push_back(std::move(s));
            ws_sessions_.clear();
        }
        for (auto& s : live)
            s->close();
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
    api_reply route(bhttp::verb        method,
                    const std::string& path,
                    const std::string& query,
                    const std::string& body,
                    const std::string& peer,
                    const std::string& authorization)
    {
        // Before anything else, and on every request. `/v1/auth` is the one endpoint that
        // has to be reachable without an answer, because it is where the question comes
        // from -- and it reveals only a random challenge and a salt the client is meant to
        // have.
        if (path == "/v1/auth")
            return auth_.issue_challenge();

        if (!auth_.check(authorization))
            return api_reply::fail(api_code::unauthorized,
                                   "GET /v1/auth for a challenge, then answer on every request");

        // `?HOST_INFO` is OSCQuery's capability probe and is answered on ANY path, which is
        // what the spec says and what makes it usable as a liveness check without knowing
        // the address space first. It sits BELOW the authentication check on purpose: with
        // `<auth>password</auth>` configured, an unauthenticated client learns nothing at
        // all, not even the server's name.
        if (query == "HOST_INFO")
            return api_reply::ok_with(host_info(config_, count_subscriptions()));

        if (method == bhttp::verb::post) {
            if (starts_with(path, "/v1/action/"))
                return run_action(context_, path.substr(std::string("/v1/action").size()), body, peer);
            if (path == "/v1/batch") {
                batch_plan deferred;
                auto       reply = run_batch(context_, body, peer, deferred);
                if (reply.code == api_code::ok && !deferred.ops.empty())
                    return park_batch(std::move(deferred));
                return reply;
            }
            return api_reply::fail(api_code::unknown_path, "nothing accepts POST at " + path);
        }

        if (method == bhttp::verb::put) {
            if (starts_with(path, "/v1/value/"))
                return write_value(context_, *hub_, path.substr(std::string("/v1/value").size()), body, peer);
            return api_reply::fail(api_code::unknown_path, "nothing is writable at " + path);
        }

        if (method != bhttp::verb::get && method != bhttp::verb::head)
            return api_reply::fail(api_code::bad_request, "this build accepts GET, PUT and POST");

        if (path == "/" || path == "/v1")
            return api_reply::ok_with(host_info(config_, count_subscriptions()));

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

        if (websocket::is_upgrade(*req)) {
            // The event socket is on the same port and carries the same state, so it takes
            // the same answer -- in the upgrade request's own `Authorization` header, which
            // is the only chance a WebSocket handshake gives.
            if (!auth_.check(std::string(req->operator[](bhttp::field::authorization)))) {
                auto reply   = api_reply::fail(api_code::unauthorized,
                                               "GET /v1/auth for a challenge, then answer on the upgrade request");
                auto body    = json::serialize(json::value(envelope(reply, server_name_)));
                const auto v = req->version();
                asio::post(stream->get_executor(), [self, stream, buffer, body = std::move(body), v]() mutable {
                    self->write(
                        stream, buffer, bhttp::status::unauthorized, std::move(body), bhttp::verb::get, false, v);
                });
                return;
            }

            if (target.path != "/v1/events") {
                // An upgrade on any other path is answered as an ordinary request, so a
                // client that mistyped the path gets `unknown_path` and its message rather
                // than a socket that opens and then says nothing.
                auto reply  = api_reply::fail(api_code::unknown_path,
                                              "the event socket is at /v1/events, not " + target.path);
                auto body   = json::serialize(json::value(envelope(reply, server_name_)));
                const auto v = req->version();
                asio::post(stream->get_executor(), [self, stream, buffer, body = std::move(body), v]() mutable {
                    self->write(stream, buffer, bhttp::status::not_found, std::move(body), bhttp::verb::get, false, v);
                });
                return;
            }

            // The stream moves into the session and this HTTP loop ends here.
            auto session = std::make_shared<ws_session>(
                websocket::stream<beast::tcp_stream>(std::move(*stream)), self);
            add_ws_session(session);
            session->run(std::move(*req));
            return;
        }

        const auto method  = req->method();
        const auto keep    = req->keep_alive();
        const auto ver     = req->version();
        const auto reqbody = req->body();
        const auto authorization = std::string(req->operator[](bhttp::field::authorization));

        std::string peer;
        try {
            peer = beast::get_lowest_layer(*stream).socket().remote_endpoint().address().to_string();
        } catch (...) {
            peer = "?";
        }

        // Off the io_context thread from here. Nothing below touches the socket until the
        // reply is posted back onto its strand.
        api_executor_.begin_invoke(
            [self, stream, buffer, target, method, keep, ver, reqbody, peer, authorization]() {
            api_reply reply;
            try {
                reply = self->route(method, target.path, target.query, reqbody, peer, authorization);
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

// -----------------------------------------------------------------------------------------
// ws_session
// -----------------------------------------------------------------------------------------

void ws_session::run(bhttp::request<bhttp::string_body> req)
{
    ws_.set_option(websocket::stream_base::timeout::suggested(beast::role_type::server));
    ws_.set_option(websocket::stream_base::decorator(
        [](websocket::response_type& res) { res.set(bhttp::field::server, "CasparCG"); }));

    auto self = shared_from_this();
    ws_.async_accept(req, [self](boost::system::error_code ec) {
        if (ec) {
            CASPAR_LOG(debug) << L"[http-api] ws accept: " << u16(ec.message());
            return;
        }
        self->read();
    });
}

void ws_session::read()
{
    auto self = shared_from_this();
    ws_.async_read(buffer_, [self](boost::system::error_code ec, std::size_t) {
        if (ec) {
            if (ec != websocket::error::closed && ec != asio::error::operation_aborted)
                CASPAR_LOG(debug) << L"[http-api] ws read: " << u16(ec.message());
            self->closed_.store(true, std::memory_order_release);
            return;
        }
        auto text = beast::buffers_to_string(self->buffer_.data());
        self->buffer_.consume(self->buffer_.size());
        self->handle_text(std::move(text));
        self->read();
    });
}

void ws_session::handle_text(std::string text)
{
    auto self = shared_from_this();

    // Onto the API executor, for the same reason a request body goes there: parsing and
    // the first collect walk the state, and this handler is running on an `io_context`
    // thread that AMCP and OSC share.
    owner_->api_executor_.begin_invoke([self, text = std::move(text)] {
        api_reply   reply;
        json::value msg;
        try {
            msg = json::parse(text);
        } catch (...) {
            reply = api_reply::fail(api_code::bad_request, "message is not valid JSON");
            self->send(json::serialize(json::value(envelope(reply, self->owner_->server_name_))));
            return;
        }

        const std::string op =
            msg.is_object() && msg.as_object().if_contains("op") && msg.as_object().at("op").is_string()
                ? std::string(msg.as_object().at("op").as_string().c_str())
                : std::string();

        if (op == "subscribe") {
            subscription sub;
            reply = parse_subscribe(msg, self->owner_->config_.max_prefixes, sub);
            if (reply.code == api_code::ok) {
                self->sub_ = std::move(sub);
                self->subscribed_.store(true, std::memory_order_release);

                json::object r;
                r["subscribed"] = true;
                r["id"]         = self->sub_.id;
                r["prefixes"]   = static_cast<std::int64_t>(self->sub_.prefixes.size());
                reply.result    = std::move(r);
            }
        } else if (op == "unsubscribe") {
            self->subscribed_.store(false, std::memory_order_release);
            self->sub_ = subscription{};
            json::object r;
            r["subscribed"] = false;
            reply.result    = std::move(r);
        } else {
            reply = api_reply::fail(api_code::bad_request, "unknown op; this build accepts subscribe and unsubscribe");
        }

        self->send(json::serialize(json::value(envelope(reply, self->owner_->server_name_))));

        // The first message after subscribing is the whole subscribed set, not a diff --
        // `last_values` is empty, so every matching key reads as changed. That is what
        // gives a client its initial state without a second REST round trip.
        if (self->subscribed())
            self->notify();
    });
}

void ws_session::notify()
{
    if (!subscribed() || closed_.load(std::memory_order_acquire))
        return;

    // One collect at a time per session. Ticks arrive faster than a slow client drains,
    // and queueing a collect per tick would grow without bound; skipping is correct
    // because the next collect diffs against the same `last_values` and therefore carries
    // whatever this one would have.
    if (collecting_.exchange(true, std::memory_order_acq_rel))
        return;

    auto self = shared_from_this();
    owner_->api_executor_.begin_invoke([self] {
        json::object msg;
        try {
            msg = collect_events(self->sub_, *self->owner_->hub_, self->owner_->server_name_);
        } catch (...) {
            CASPAR_LOG_CURRENT_EXCEPTION();
        }
        self->collecting_.store(false, std::memory_order_release);
        if (!msg.empty())
            self->send(json::serialize(json::value(std::move(msg))));
    });
}

void ws_session::send(std::string payload)
{
    auto self = shared_from_this();
    asio::post(ws_.get_executor(), [self, payload = std::move(payload)]() mutable {
        if (self->closed_.load(std::memory_order_acquire))
            return;
        self->outbox_.push_back(std::move(payload));
        if (!self->writing_)
            self->write_next();
    });
}

void ws_session::write_next()
{
    if (outbox_.empty()) {
        writing_ = false;
        return;
    }
    writing_ = true;

    auto self = shared_from_this();
    ws_.text(true);
    ws_.async_write(asio::buffer(outbox_.front()), [self](boost::system::error_code ec, std::size_t) {
        self->outbox_.pop_front();
        if (ec) {
            if (ec != websocket::error::closed && ec != asio::error::operation_aborted)
                CASPAR_LOG(debug) << L"[http-api] ws write: " << u16(ec.message());
            self->closed_.store(true, std::memory_order_release);
            self->writing_ = false;
            return;
        }
        self->write_next();
    });
}

void ws_session::close()
{
    if (closed_.exchange(true, std::memory_order_acq_rel))
        return;
    auto self = shared_from_this();
    asio::post(ws_.get_executor(), [self] {
        boost::system::error_code ec;
        self->ws_.close(websocket::close_code::going_away, ec);
    });
}

http_server::http_server(std::shared_ptr<boost::asio::io_context> io_context,
                         std::shared_ptr<state_hub>               hub,
                         http_config                              config,
                         api_context                              context)
    : impl_(std::make_shared<impl>(std::move(io_context), std::move(hub), std::move(config), std::move(context)))
{
    impl_->start();
}

http_server::~http_server() { impl_->stop(); }

unsigned short http_server::port() const { return impl_->config_.port; }

}}} // namespace caspar::protocol::http
