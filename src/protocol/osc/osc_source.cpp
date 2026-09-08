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

#include "../StdAfx.h"

#include "osc_source.h"

#include "oscpack/OscReceivedElements.h"

#include <common/log.h>

#include <boost/asio.hpp>

#include <atomic>
#include <map>
#include <mutex>
#include <thread>
#include <vector>

namespace caspar { namespace protocol { namespace osc {

namespace {

/// The address every packet this receiver accepts must start with.
///
/// A fixed prefix rather than an arbitrary address, so a packet meant for something else on the
/// same port cannot populate a channel by accident -- and so a misaddressed packet is counted
/// as dropped rather than half-applied.
constexpr const char* kPrefix = "/casparcg/source/";

/// A packet larger than this is not read. OSC over UDP is small by design; the largest thing
/// this receiver can legitimately be sent is a handful of floats.
constexpr std::size_t kMaxPacket = 8192;

/// One argument as a double, whatever OSC type tag it arrived with.
///
/// int32, int64, float, double and bool are all accepted, because a control surface sends
/// whichever its author felt like and refusing an int for a value that is obviously a number
/// would be a needless incompatibility. A string or a blob is refused -- it is not a number and
/// guessing would be worse than dropping.
bool arg_to_double(const ::osc::ReceivedMessageArgument& a, double& out)
{
    try {
        if (a.IsFloat()) {
            out = a.AsFloatUnchecked();
            return true;
        }
        if (a.IsDouble()) {
            out = a.AsDoubleUnchecked();
            return true;
        }
        if (a.IsInt32()) {
            out = a.AsInt32Unchecked();
            return true;
        }
        if (a.IsInt64()) {
            out = static_cast<double>(a.AsInt64Unchecked());
            return true;
        }
        if (a.IsBool()) {
            out = a.AsBoolUnchecked() ? 1.0 : 0.0;
            return true;
        }
    } catch (...) {
    }
    return false;
}

} // namespace

struct osc_source::impl
{
    const std::string    name;
    const unsigned short port;

    mutable std::mutex            lock;
    std::map<std::string, double> values;
    std::uint64_t                 packets = 0;
    std::uint64_t                 dropped = 0;

    boost::asio::io_context                              io;
    std::unique_ptr<boost::asio::ip::udp::socket>        socket;
    std::atomic<bool>                                    bound{false};
    std::atomic<bool>                                    running{true};
    std::thread                                          thread;

    impl(std::string n, unsigned short p)
        : name(std::move(n))
        , port(p)
    {
        boost::system::error_code ec;
        socket = std::make_unique<boost::asio::ip::udp::socket>(io);
        socket->open(boost::asio::ip::udp::v4(), ec);
        if (!ec) {
            // REUSE, so a source replaced on the same port binds rather than failing -- the
            // previous socket's close may not have been reaped by the OS yet. Without it,
            // `SOURCE ADD knobs OSC 7400` twice in a row leaves the second one deaf.
            socket->set_option(boost::asio::socket_base::reuse_address(true), ec);
            socket->bind(boost::asio::ip::udp::endpoint(boost::asio::ip::udp::v4(), port), ec);
        }

        if (ec) {
            CASPAR_LOG(warning) << L"[osc-source] could not listen on port " << port << L": "
                                << u16(ec.message())
                                << L". The source exists and every channel reads as absent, so a "
                                   L"binding to it will report BROKEN rather than silently doing "
                                   L"nothing.";
            return;
        }

        bound = true;
        CASPAR_LOG(info) << L"[osc-source] " << u16(name) << L" listening on udp/" << port
                         << L" for " << u16(kPrefix) << u16(name) << L"/...";

        thread = std::thread([this] { run(); });
    }

    ~impl()
    {
        running = false;
        if (socket) {
            boost::system::error_code ec;
            // Closed before the join, which is what unblocks the blocking receive. Cancelling
            // alone is not enough on Windows: `receive_from` returns only when the handle goes.
            socket->close(ec);
        }
        if (thread.joinable())
            thread.join();
    }

    void run()
    {
        std::vector<char>                 buf(kMaxPacket);
        boost::asio::ip::udp::endpoint    from;

        while (running) {
            boost::system::error_code ec;
            const auto                n = socket->receive_from(boost::asio::buffer(buf), from, 0, ec);

            if (!running)
                break;
            if (ec || n == 0)
                continue;

            try {
                handle(::osc::ReceivedPacket(buf.data(), static_cast<::osc::int32>(n)));
            } catch (...) {
                // A MALFORMED packet must change nothing and must be visible. oscpack throws on
                // a bad size or a bad type tag string, and swallowing that silently would make a
                // sender with a bug indistinguishable from one that is not sending.
                std::lock_guard<std::mutex> g(lock);
                ++dropped;
            }
        }
    }

    void handle(const ::osc::ReceivedPacket& packet)
    {
        if (packet.IsBundle()) {
            // Recursed, because a control surface batching four knobs into one bundle is normal
            // and dropping the bundle would lose all four.
            ::osc::ReceivedBundle bundle(packet);
            for (auto it = bundle.ElementsBegin(); it != bundle.ElementsEnd(); ++it)
                handle(::osc::ReceivedPacket(it->Contents(), it->Size()));
            return;
        }

        const ::osc::ReceivedMessage msg(packet);
        const std::string            address = msg.AddressPattern() ? msg.AddressPattern() : "";

        const std::string mine = std::string(kPrefix) + name;
        if (address.rfind(mine, 0) != 0) {
            std::lock_guard<std::mutex> g(lock);
            ++dropped;
            return;
        }

        // `/casparcg/source/knobs`        -> the channel is "value"
        // `/casparcg/source/knobs/fader`  -> the channel is "fader"
        //
        // "value" as the bare-address default so a sender that addresses the source itself works
        // -- and it matches the LFO source's own channel name, so a binding written against one
        // reads naturally against the other.
        std::string channel = "value";
        if (address.size() > mine.size()) {
            if (address[mine.size()] != '/') {
                std::lock_guard<std::mutex> g(lock);
                ++dropped; // `/casparcg/source/knobsX` is not this source
                return;
            }
            channel = address.substr(mine.size() + 1);
        }

        std::vector<double> args;
        for (auto it = msg.ArgumentsBegin(); it != msg.ArgumentsEnd(); ++it) {
            double v = 0.0;
            if (arg_to_double(*it, v))
                args.push_back(v);
        }

        if (args.empty()) {
            std::lock_guard<std::mutex> g(lock);
            ++dropped;
            return;
        }

        std::lock_guard<std::mutex> g(lock);
        ++packets;
        if (args.size() == 1) {
            values[channel] = args[0];
        } else {
            // Several arguments become several channels, `<channel>/0`, `/1`, ... A vec3 sent as
            // one message is how a phone's accelerometer or a lighting desk's RGB arrives, and
            // three bindings then address three numbers.
            for (std::size_t i = 0; i < args.size(); ++i)
                values[channel + "/" + std::to_string(i)] = args[i];
        }
    }
};

osc_source::osc_source(std::string name, unsigned short port)
    : impl_(std::make_shared<impl>(std::move(name), port))
{
}

osc_source::~osc_source() = default;

bool osc_source::value(const std::string& channel, double& out) const
{
    std::lock_guard<std::mutex> g(impl_->lock);

    // The receiver's own telemetry, as channels. Bindable, which is not the point -- the point
    // is that they are READABLE: "nothing is arriving" and "something is arriving at the wrong
    // address" are different faults and an operator has to be able to tell them apart.
    if (channel == "packets") {
        out = static_cast<double>(impl_->packets);
        return true;
    }
    if (channel == "dropped") {
        out = static_cast<double>(impl_->dropped);
        return true;
    }

    auto it = impl_->values.find(channel);
    if (it == impl_->values.end())
        return false;
    out = it->second;
    return true;
}

std::vector<std::string> osc_source::channels() const
{
    std::lock_guard<std::mutex> g(impl_->lock);

    // Whatever has ARRIVED, plus the two telemetry channels. There is no declared channel set:
    // a receiver cannot know what a sender will address until it does, which is exactly why a
    // binding to an unknown channel reports `broken` rather than being refused at BIND time --
    // an operator legitimately binds before the surface is switched on.
    std::vector<std::string> out{"packets", "dropped"};
    for (const auto& kv : impl_->values)
        out.push_back(kv.first);
    return out;
}

std::string osc_source::describe() const
{
    std::lock_guard<std::mutex> g(impl_->lock);
    return "osc udp/" + std::to_string(impl_->port) + " " + std::string(kPrefix) + impl_->name +
           "/... (" + (impl_->bound ? "listening" : "NOT BOUND") + ", " +
           std::to_string(impl_->packets) + " accepted, " + std::to_string(impl_->dropped) + " dropped)";
}

bool osc_source::listening() const { return impl_->bound; }

}}} // namespace caspar::protocol::osc
