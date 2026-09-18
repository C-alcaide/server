/*
 * Copyright (c) 2011 Sveriges Television AB <info@casparcg.com>
 *
 * This file is part of CasparCG (www.casparcg.com).
 *
 * CasparCG is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * CasparCG is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with CasparCG. If not, see <http://www.gnu.org/licenses/>.
 */

#include "osc_sink.h"

#include "oscpack/OscOutboundPacketStream.h"

#include <common/log.h>

#include <boost/asio.hpp>

#include <array>
#include <atomic>
#include <mutex>

namespace caspar { namespace protocol { namespace osc {

struct osc_sink::impl
{
    std::string                    name;
    boost::asio::io_context        io;
    boost::asio::ip::udp::socket   socket{io};
    boost::asio::ip::udp::endpoint endpoint;
    bool                           open = false;
    std::mutex                     mutex;
    std::array<char, 512>          buffer{};
    std::atomic<std::uint64_t>     sent{0};
    std::atomic<std::uint64_t>     dropped{0};

    impl(std::string n, const std::string& host, unsigned short port)
        : name(std::move(n))
    {
        boost::system::error_code ec;
        const auto                addr = boost::asio::ip::make_address(host, ec);
        if (ec) {
            CASPAR_LOG(warning) << L"[osc_sink] " << u16(name) << L": '" << u16(host)
                                << L"' is not a numeric address; the sink will drop everything";
            return;
        }
        endpoint = boost::asio::ip::udp::endpoint(addr, port);
        socket.open(endpoint.protocol(), ec);
        if (ec) {
            CASPAR_LOG(warning) << L"[osc_sink] " << u16(name) << L": socket open failed: " << u16(ec.message());
            return;
        }
        // Never block the stage tick: a full send buffer drops and counts.
        socket.non_blocking(true, ec);
        open = true;
        CASPAR_LOG(info) << L"[osc_sink] " << u16(name) << L" -> udp/" << u16(host) << L":" << port;
    }

    bool send(const std::string& address, double value)
    {
        if (!open) {
            dropped.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        std::lock_guard<std::mutex> lock(mutex);
        try {
            ::osc::OutboundPacketStream o(buffer.data(), static_cast<unsigned long>(buffer.size()));
            o << ::osc::BeginMessage(address.c_str()) << static_cast<float>(value) << ::osc::EndMessage;
            boost::system::error_code ec;
            const auto                n = socket.send_to(boost::asio::buffer(o.Data(), o.Size()), endpoint, 0, ec);
            if (ec || n != o.Size()) {
                dropped.fetch_add(1, std::memory_order_relaxed);
                return false;
            }
        } catch (const std::exception&) {
            // oscpack's OutOfBufferMemoryException: an address longer than the buffer allows.
            dropped.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        sent.fetch_add(1, std::memory_order_relaxed);
        return true;
    }
};

osc_sink::osc_sink(std::string name, const std::string& host, unsigned short port)
    : impl_(std::make_shared<impl>(std::move(name), host, port))
{
}

osc_sink::~osc_sink() = default;

bool osc_sink::send(const std::string& address, double value) { return impl_->send(address, value); }

bool osc_sink::ready() const { return impl_->open; }

std::string osc_sink::describe() const
{
    if (!impl_->open)
        return "osc NOT OPEN (sent 0, dropped " + std::to_string(impl_->dropped.load()) + ")";
    return "osc -> udp/" + impl_->endpoint.address().to_string() + ":" + std::to_string(impl_->endpoint.port()) +
           " (sent " + std::to_string(impl_->sent.load()) + ", dropped " + std::to_string(impl_->dropped.load()) +
           ")";
}

}}} // namespace caspar::protocol::osc
