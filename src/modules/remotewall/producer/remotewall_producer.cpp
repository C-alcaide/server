/*
 * Copyright (c) 2026 CasparCG Contributors
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

#include "remotewall_producer.h"

#include <core/frame/draw_frame.h>
#include <core/frame/frame.h>
#include <core/frame/frame_factory.h>
#include <core/frame/pixel_format.h>
#include <core/monitor/monitor.h>
#include <core/producer/frame_producer.h>
#include <core/video_format.h>

#include <common/log.h>

#include <boost/algorithm/string.hpp>

#include <cstddef>
#include <cstdint>

namespace caspar { namespace remotewall {

module_config& config()
{
    static module_config c;
    return c;
}

namespace {

// Phase 0 placeholder producer: registers, negotiates a channel-sized BGRA frame, and renders a
// solid marker colour so we can confirm the module is wired end to end (AMCP -> factory -> frame).
// The live cloudXR receiver replaces receive_impl() in Phase 1.
class remotewall_producer : public core::frame_producer
{
    spl::shared_ptr<core::frame_factory> frame_factory_;
    core::video_format_desc              format_desc_;
    const int                            width_;
    const int                            height_;
    const int                            port_;
    core::monitor::state                 state_;

  public:
    remotewall_producer(const core::frame_producer_dependencies& deps, int port)
        : frame_factory_(deps.frame_factory)
        , format_desc_(deps.format_desc)
        , width_(deps.format_desc.width)
        , height_(deps.format_desc.height)
        , port_(port)
    {
        CASPAR_LOG(info) << L"[remotewall] producer created (port " << port_ << L", " << width_ << L"x"
                         << height_ << L") -- Phase 0 placeholder (no receiver yet).";
    }

    core::draw_frame receive_impl(const core::video_field /*field*/, int /*nb_samples*/) override
    {
        core::pixel_format_desc pfd(core::pixel_format::bgra);
        pfd.planes.push_back(core::pixel_format_desc::plane(width_, height_, 4));

        auto              frame = frame_factory_->create_frame(this, pfd);
        auto* const       data  = frame.image_data(0).data();
        const std::size_t n     = static_cast<std::size_t>(width_) * height_ * 4;
        for (std::size_t i = 0; i + 4 <= n; i += 4) {
            data[i + 0] = 64;  // B
            data[i + 1] = 48;  // G
            data[i + 2] = 16;  // R
            data[i + 3] = 255; // A (opaque)
        }
        return core::draw_frame(std::move(frame));
    }

    core::monitor::state state() const override { return state_; }
    std::wstring         print() const override { return L"remotewall[" + std::to_wstring(port_) + L"]"; }
    std::wstring         name() const override { return L"remotewall"; }
    bool                 is_ready() override { return true; }
};

} // namespace

spl::shared_ptr<core::frame_producer> create_producer(const core::frame_producer_dependencies& dependencies,
                                                      const std::vector<std::wstring>&          params)
{
    if (params.empty() || !boost::iequals(params.at(0), L"REMOTEWALL"))
        return core::frame_producer::empty();

    int port = config().listen_port;
    for (std::size_t i = 1; i < params.size(); ++i) {
        if (boost::iequals(params[i], L"PORT") && i + 1 < params.size()) {
            try {
                port = std::stoi(params[++i]);
            } catch (...) {
            }
        }
    }

    return spl::make_shared<remotewall_producer>(dependencies, port);
}

void register_remotewall_producer(const core::module_dependencies& dependencies)
{
    dependencies.producer_registry->register_producer_factory(L"REMOTEWALL Producer", &create_producer);
}

}} // namespace caspar::remotewall
