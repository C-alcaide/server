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
#include <cstring>
#include <vector>

#include "RecvWall.h" // vendored receiver C API

namespace caspar { namespace remotewall {

module_config& config()
{
    static module_config c;
    return c;
}

namespace {

// Phase 1 producer: binds the cloudXR receiver (UDP/RTP+FEC -> per-tile NVDEC -> composite) and
// serves the latest wall frame. CPU-readback path (RecvWallGetLatest copies the composite to host);
// the CUDA<->VK/GL zero-copy hand-off replaces this in Phase 2. Renders a marker frame until signal.
class remotewall_producer : public core::frame_producer
{
    spl::shared_ptr<core::frame_factory> frame_factory_;
    core::video_format_desc              format_desc_;
    const int                            fallback_w_;
    const int                            fallback_h_;
    const int                            port_;
    const int                            tiles_;
    const int                            codec_;

    RecvWallHandle*           rw_ = nullptr;
    std::vector<std::uint8_t> staging_;
    RecvWallFrameInfo         info_{};
    unsigned long long        ver_        = 0;
    int                       wall_w_     = 0;
    int                       wall_h_     = 0;
    bool                      have_frame_ = false;
    core::monitor::state      state_;

  public:
    remotewall_producer(const core::frame_producer_dependencies& deps, int port, int tiles, int codec)
        : frame_factory_(deps.frame_factory)
        , format_desc_(deps.format_desc)
        , fallback_w_(deps.format_desc.width)
        , fallback_h_(deps.format_desc.height)
        , port_(port)
        , tiles_(tiles)
        , codec_(codec)
    {
        RecvWallConfig cfg{};
        cfg.listenPort    = static_cast<unsigned short>(port_);
        cfg.expectedTiles = tiles_; // 0 = auto from the first SyncMeta
        cfg.codec         = codec_; // 0 = HEVC, 1 = H.264
        cfg.fullRange     = 1;      // full-swing RGB
        cfg.pixelOrder    = 0;      // 0 = BGRA (matches the bgra frame tag)
        cfg.useGpuConvert = 1;      // CUDA NV12->BGRA composite (lib falls back to CPU on failure)
        rw_               = RecvWallInit(&cfg);
        CASPAR_LOG(info) << L"[remotewall] receiver " << (rw_ ? L"started" : L"FAILED") << L" on UDP port "
                         << port_ << L" (tiles=" << tiles_ << L", codec=" << codec_ << L").";
    }

    ~remotewall_producer()
    {
        if (rw_)
            RecvWallShutdown(rw_);
    }

    core::draw_frame make_marker_frame()
    {
        core::pixel_format_desc pfd(core::pixel_format::bgra);
        pfd.planes.push_back(core::pixel_format_desc::plane(fallback_w_, fallback_h_, 4));
        auto              frame = frame_factory_->create_frame(this, pfd);
        auto*             data  = frame.image_data(0).data();
        const std::size_t n     = static_cast<std::size_t>(fallback_w_) * fallback_h_ * 4;
        for (std::size_t i = 0; i + 4 <= n; i += 4) {
            data[i + 0] = 64;
            data[i + 1] = 48;
            data[i + 2] = 16;
            data[i + 3] = 255;
        }
        return core::draw_frame(std::move(frame));
    }

    core::draw_frame receive_impl(const core::video_field /*field*/, int /*nb_samples*/) override
    {
        if (rw_) {
            unsigned gw = 0, gh = 0;
            if (RecvWallGetGeometry(rw_, &gw, &gh, nullptr, nullptr, nullptr, nullptr) && gw > 0 && gh > 0) {
                const std::size_t need = static_cast<std::size_t>(gw) * gh * 4;
                if (staging_.size() != need)
                    staging_.assign(need, 0);
                RecvWallWaitNewFrame(rw_, ver_, 8);
                if (RecvWallGetLatest(rw_,
                                      staging_.data(),
                                      static_cast<int>(gw) * 4,
                                      static_cast<int>(staging_.size()),
                                      &ver_,
                                      &info_) == 1) {
                    have_frame_ = true;
                }
                wall_w_ = static_cast<int>(gw);
                wall_h_ = static_cast<int>(gh);
            }
        }

        if (!have_frame_ || wall_w_ <= 0 || wall_h_ <= 0)
            return make_marker_frame();

        // Copy the top-down BGRA composite into a wall-sized frame (Phase 1 CPU readback).
        core::pixel_format_desc pfd(core::pixel_format::bgra);
        pfd.planes.push_back(core::pixel_format_desc::plane(wall_w_, wall_h_, 4));

        auto      frame      = frame_factory_->create_frame(this, pfd);
        const int dst_stride = frame.pixel_format_desc().planes[0].linesize;
        const int src_stride = wall_w_ * 4;
        auto*     dst        = frame.image_data(0).data();
        for (int y = 0; y < wall_h_; ++y)
            std::memcpy(dst + static_cast<std::size_t>(y) * dst_stride,
                        staging_.data() + static_cast<std::size_t>(y) * src_stride,
                        static_cast<std::size_t>(src_stride));
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

    int port  = config().listen_port;
    int tiles = 0; // auto-detect from the first SyncMeta
    int codec = 0; // 0 = HEVC, 1 = H.264
    for (std::size_t i = 1; i < params.size(); ++i) {
        if (boost::iequals(params[i], L"PORT") && i + 1 < params.size()) {
            try {
                port = std::stoi(params[++i]);
            } catch (...) {
            }
        } else if (boost::iequals(params[i], L"TILES") && i + 1 < params.size()) {
            try {
                tiles = std::stoi(params[++i]);
            } catch (...) {
            }
        } else if (boost::iequals(params[i], L"CODEC") && i + 1 < params.size()) {
            codec = boost::iequals(params[++i], L"h264") ? 1 : 0;
        }
    }

    return spl::make_shared<remotewall_producer>(dependencies, port, tiles, codec);
}

void register_remotewall_producer(const core::module_dependencies& dependencies)
{
    dependencies.producer_registry->register_producer_factory(L"REMOTEWALL Producer", &create_producer);
}

}} // namespace caspar::remotewall
