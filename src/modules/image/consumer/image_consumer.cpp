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
 *
 * Author: Robert Nagy, ronag89@gmail.com
 * Author: Julian Waller, julian@superfly.tv
 */

#include "image_consumer.h"

#include <common/array.h>
#include <common/bit_depth.h>
#include <common/env.h>
#include <common/except.h>
#include <common/future.h>
#include <common/log.h>

#include <core/consumer/channel_info.h>
#include <core/frame/frame.h>

#include <boost/algorithm/string.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

#include <algorithm>
#include <fstream>
#include <utility>
#include <vector>

#include <ffmpeg/util/av_assert.h>
#include <ffmpeg/util/av_util.h>

#include "../util/image_algorithms.h"
#include "../util/image_converter.h"
#include "../util/image_view.h"

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4244)
#endif
extern "C" {
#define __STDC_CONSTANT_MACROS
#define __STDC_LIMIT_MACROS
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libavutil/pixfmt.h>
}
#ifdef _MSC_VER
#pragma warning(pop)
#endif

namespace caspar::image {

struct image_consumer : public core::frame_consumer
{
    const std::wstring filename_;
    int                frames_waited_{0};

    explicit image_consumer(std::wstring filename)
        : filename_(std::move(filename))
    {
    }

    void initialize(const core::video_format_desc& /*format_desc*/,
                    const core::channel_info& channel_info,
                    int                       port_index) override
    {
    }

    std::future<bool> send(core::video_field field, core::const_frame frame) override
    {
        // The vulkan mixer's 1-frame pipeline delay means the first frame
        // after this consumer is added may have empty CPU readback data
        // (rendered before cpu_readback_needed was set).  Skip it and wait
        // for the next tick which will have valid pixel data.
        const auto& data = frame.image_data(0);
        if (data.data() == nullptr || data.size() == 0) {
            if (++frames_waited_ < 4) {
                return make_ready_future(true);  // stay alive, wait for valid frame
            }
            // Give up after 4 empty frames to avoid hanging forever
            CASPAR_LOG(warning) << L"[image_consumer] No valid CPU frame data after "
                                << frames_waited_ << L" ticks — capturing empty frame.";
        }

        auto filename = filename_;

        std::thread async([frame, filename] {
            try {
                std::string filename2;

                if (filename.empty())
                    filename2 =
                        u8(env::media_folder() +
                           boost::posix_time::to_iso_wstring(boost::posix_time::second_clock::local_time()) + L".png");
                else
                    filename2 = u8(env::media_folder() + filename + L".png");

                std::fstream file_stream(filename2, std::fstream::out | std::fstream::trunc | std::fstream::binary);
                if (!file_stream)
                    FF_RET(AVERROR(EINVAL), "fstream_open");

                const AVCodec* codec = avcodec_find_encoder(AV_CODEC_ID_PNG);
                if (!codec)
                    FF_RET(AVERROR(EINVAL), "avcodec_find_encoder");

                auto ctx = std::shared_ptr<AVCodecContext>(avcodec_alloc_context3(codec),
                                                           [](AVCodecContext* ptr) { avcodec_free_context(&ptr); });

                // Determine if this is a high bit-depth frame
                const auto& pix_desc  = frame.pixel_format_desc();
                bool        is_hi_dep = pix_desc.planes.size() > 0 &&
                                 pix_desc.planes[0].depth != common::bit_depth::bit8;

                // For 16-bit frames, output PNG16 (RGBA64BE); for 8-bit, standard RGBA
                AVPixelFormat target_fmt = is_hi_dep ? AV_PIX_FMT_RGBA64BE : AV_PIX_FMT_RGBA;

                ctx->width     = static_cast<int>(frame.width());
                ctx->height    = static_cast<int>(frame.height());
                ctx->pix_fmt   = target_fmt;
                ctx->time_base = {1, 1};
                ctx->framerate = {0, 1};

                FF(avcodec_open2(ctx.get(), codec, nullptr));

                // Build source AVFrame from the frame's plane data
                auto av_frame   = ffmpeg::alloc_frame();
                av_frame->width  = static_cast<int>(frame.width());
                av_frame->height = static_cast<int>(frame.height());
                av_frame->pts    = 0;

                if (pix_desc.format == core::pixel_format::bgra) {
                    // Mixer always outputs packed BGRA (1 plane, 4 components)
                    if (is_hi_dep) {
                        av_frame->format      = AV_PIX_FMT_BGRA64LE;
                        av_frame->linesize[0] = static_cast<int>(frame.width()) * 8;
                    } else {
                        av_frame->format      = AV_PIX_FMT_BGRA;
                        av_frame->linesize[0] = static_cast<int>(frame.width()) * 4;
                    }
                    av_frame->data[0] = const_cast<uint8_t*>(frame.image_data(0).data());
                } else if (pix_desc.format == core::pixel_format::gbrp ||
                           pix_desc.format == core::pixel_format::gbrap) {
                    // Planar GBR(A) — unlikely from mixer but possible from direct frame path
                    bool has_alpha = (pix_desc.planes.size() >= 4);
                    int  bpc       = is_hi_dep ? 2 : 1; // bytes per component
                    if (is_hi_dep) {
                        av_frame->format = has_alpha ? AV_PIX_FMT_GBRAP16LE : AV_PIX_FMT_GBRP16LE;
                    } else {
                        av_frame->format = has_alpha ? AV_PIX_FMT_GBRAP : AV_PIX_FMT_GBRP;
                    }
                    for (size_t i = 0; i < pix_desc.planes.size() && i < 4; ++i) {
                        av_frame->data[i]     = const_cast<uint8_t*>(frame.image_data(static_cast<int>(i)).data());
                        av_frame->linesize[i] = pix_desc.planes[i].width * pix_desc.planes[i].stride * bpc;
                    }
                } else {
                    // Fallback: assume packed BGRA-like format
                    av_frame->format      = AV_PIX_FMT_BGRA;
                    av_frame->linesize[0] = static_cast<int>(frame.width()) * 4;
                    av_frame->data[0]     = const_cast<uint8_t*>(frame.image_data(0).data());
                }

                // Straighten alpha — PNG stores straight alpha, mixer produces premultiplied.
                // Must be done BEFORE converting to RGBA64BE because the un-premultiply
                // operates via native uint16_t* which requires little-endian data on x64.
                if (is_hi_dep && pix_desc.format == core::pixel_format::bgra) {
                    // Work on the native-endian BGRA64LE source data directly.
                    // We need a writable copy since frame data is const.
                    auto  src_size = av_frame->linesize[0] * av_frame->height;
                    auto  buf      = std::vector<uint8_t>(frame.image_data(0).begin(),
                                                         frame.image_data(0).begin() + src_size);
                    auto* data     = reinterpret_cast<uint16_t*>(buf.data());
                    const int stride16 = av_frame->linesize[0] / 2; // in uint16_t units
                    const int w        = av_frame->width;
                    const int h        = av_frame->height;
                    for (int y = 0; y < h; ++y) {
                        uint16_t* row = data + y * stride16;
                        for (int x = 0; x < w; ++x) {
                            // BGRA component order: B=0, G=1, R=2, A=3
                            uint16_t& b = row[x * 4 + 0];
                            uint16_t& g = row[x * 4 + 1];
                            uint16_t& r = row[x * 4 + 2];
                            uint16_t  a = row[x * 4 + 3];
                            if (a != 0 && a != 65535) {
                                r = static_cast<uint16_t>(std::min(65535, static_cast<int>(r) * 65535 / a));
                                g = static_cast<uint16_t>(std::min(65535, static_cast<int>(g) * 65535 / a));
                                b = static_cast<uint16_t>(std::min(65535, static_cast<int>(b) * 65535 / a));
                            }
                        }
                    }
                    av_frame->data[0] = buf.data();

                    // Convert the un-premultiplied BGRA64LE to RGBA64BE for PNG encoding
                    auto av_frame2 = convert_image_frame(av_frame, target_fmt);

                    FF(avcodec_send_frame(ctx.get(), av_frame2.get()));
                    FF(avcodec_send_frame(ctx.get(), nullptr));

                    auto pkt =
                        std::shared_ptr<AVPacket>(av_packet_alloc(), [](AVPacket* ptr) { av_packet_free(&ptr); });
                    int ret = 0;
                    while (ret >= 0) {
                        ret = avcodec_receive_packet(ctx.get(), pkt.get());
                        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
                            break;
                        FF_RET(ret, "avcodec_receive_packet");

                        file_stream.write(reinterpret_cast<const char*>(pkt->data), pkt->size);
                        av_packet_unref(pkt.get());
                    }

                    CASPAR_LOG(info) << L"[image_consumer] Written " << u16(filename2);
                    return;
                }

                // Convert to target format for PNG encoding
                auto av_frame2 = convert_image_frame(av_frame, target_fmt);

                // Straighten alpha for 8-bit path
                if (is_hi_dep) {
                    // Non-BGRA 16-bit path (e.g. GBR planar): un-premultiply not needed
                    // as those formats don't come from the premultiplied mixer output.
                } else {
                    image_view<bgra_pixel> view(av_frame2->data[0], av_frame2->width, av_frame2->height);
                    unmultiply(view);
                }

                FF(avcodec_send_frame(ctx.get(), av_frame2.get()));
                FF(avcodec_send_frame(ctx.get(), nullptr));

                auto pkt = std::shared_ptr<AVPacket>(av_packet_alloc(), [](AVPacket* ptr) { av_packet_free(&ptr); });
                int  ret = 0;
                while (ret >= 0) {
                    ret = avcodec_receive_packet(ctx.get(), pkt.get());
                    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
                        break;
                    FF_RET(ret, "avcodec_receive_packet");

                    file_stream.write(reinterpret_cast<const char*>(pkt->data), pkt->size);
                    av_packet_unref(pkt.get());
                }

            } catch (...) {
                CASPAR_LOG_CURRENT_EXCEPTION()
            }
        });
        async.detach();

        return make_ready_future(false);
    }

    std::wstring print() const override { return L"image[]"; }

    std::wstring name() const override { return L"image"; }

    int index() const override { return 100; }

    core::monitor::state state() const override
    {
        core::monitor::state state;
        state["image/filename"] = u8(filename_);
        return state;
    }
};

spl::shared_ptr<core::frame_consumer> create_consumer(const std::vector<std::wstring>&     params,
                                                      const core::video_format_repository& format_repository,
                                                      const std::vector<spl::shared_ptr<core::video_channel>>& channels,
                                                      const core::channel_info& channel_info)
{
    if (params.empty() || !boost::iequals(params.at(0), L"IMAGE"))
        return core::frame_consumer::empty();

    std::wstring filename;

    if (params.size() > 1)
        filename = params.at(1);

    return spl::make_shared<image_consumer>(filename);
}

} // namespace caspar::image
