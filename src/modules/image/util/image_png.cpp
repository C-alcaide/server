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

#include "image_png.h"

#include <common/log.h>

#include <memory>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/frame.h>
#include <libavutil/imgutils.h>
}

namespace caspar { namespace image {

std::vector<std::uint8_t> encode_png_bgra8(const std::uint8_t* data, int width, int height)
{
    std::vector<std::uint8_t> out;
    if (!data || width <= 0 || height <= 0)
        return out;

    const AVCodec* codec = avcodec_find_encoder(AV_CODEC_ID_PNG);
    if (!codec) {
        CASPAR_LOG(error) << L"[image] no PNG encoder in this FFmpeg build";
        return out;
    }

    auto ctx = std::shared_ptr<AVCodecContext>(avcodec_alloc_context3(codec),
                                               [](AVCodecContext* p) { avcodec_free_context(&p); });
    if (!ctx)
        return out;

    // RGBA RATHER THAN BGRA, and the swap is done by hand below.
    //
    // FFmpeg's PNG encoder accepts `AV_PIX_FMT_RGBA` and not `AV_PIX_FMT_BGRA`, so the choice is
    // a swscale context or a loop. For one still image at channel resolution the loop is cheaper
    // than building and tearing down a scaler, and it is four lines with nothing to get wrong --
    // where a scaler configured with the wrong source format produces a plausible picture with
    // red and blue exchanged, which is this repository's most-repeated defect.
    ctx->pix_fmt   = AV_PIX_FMT_RGBA;
    ctx->width     = width;
    ctx->height    = height;
    ctx->time_base = AVRational{1, 1};

    if (avcodec_open2(ctx.get(), codec, nullptr) < 0) {
        CASPAR_LOG(error) << L"[image] could not open the PNG encoder";
        return out;
    }

    auto frame = std::shared_ptr<AVFrame>(av_frame_alloc(), [](AVFrame* p) { av_frame_free(&p); });
    if (!frame)
        return out;
    frame->format = ctx->pix_fmt;
    frame->width  = width;
    frame->height = height;
    if (av_frame_get_buffer(frame.get(), 0) < 0) {
        CASPAR_LOG(error) << L"[image] could not allocate the PNG encode buffer";
        return out;
    }

    for (int y = 0; y < height; ++y) {
        const auto* src = data + static_cast<std::size_t>(y) * width * 4;
        auto*       dst = frame->data[0] + static_cast<std::size_t>(y) * frame->linesize[0];
        for (int x = 0; x < width; ++x) {
            dst[x * 4 + 0] = src[x * 4 + 2]; // R <- B
            dst[x * 4 + 1] = src[x * 4 + 1]; // G
            dst[x * 4 + 2] = src[x * 4 + 0]; // B <- R
            dst[x * 4 + 3] = src[x * 4 + 3]; // A
        }
    }

    if (avcodec_send_frame(ctx.get(), frame.get()) < 0)
        return out;
    // FLUSHED IMMEDIATELY: PNG is one packet per frame, and without the null send the encoder
    // holds it waiting for more input that never arrives.
    avcodec_send_frame(ctx.get(), nullptr);

    auto pkt = std::shared_ptr<AVPacket>(av_packet_alloc(), [](AVPacket* p) { av_packet_free(&p); });
    for (;;) {
        const int ret = avcodec_receive_packet(ctx.get(), pkt.get());
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
            break;
        if (ret < 0) {
            CASPAR_LOG(error) << L"[image] the PNG encoder refused the frame";
            out.clear();
            break;
        }
        out.insert(out.end(), pkt->data, pkt->data + pkt->size);
        av_packet_unref(pkt.get());
    }
    return out;
}

}} // namespace caspar::image
