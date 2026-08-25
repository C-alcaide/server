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

#include "../StdAfx.h"

#include "gst_frame.h"

#include <common/log.h>
#include <common/scope_exit.h>
#include <common/utf.h>

#include <ffmpeg/util/av_util.h>

extern "C" {
#include <libavutil/frame.h>
#include <libavutil/pixfmt.h>
}

#include <gst/gst.h>
#include <gst/video/video.h>

#include <vector>

namespace caspar { namespace gstreamer {

namespace {

/// GStreamer and FFmpeg agree on memory layout for each of these, so a mapped GstVideoFrame
/// can be described to ffmpeg::make_frame by pointing an AVFrame at its planes. Every entry
/// is a format the mixer already handles; adding one that it does not would silently produce
/// the wrong picture rather than fail.
AVPixelFormat to_av_pixel_format(GstVideoFormat format)
{
    switch (format) {
        case GST_VIDEO_FORMAT_BGRA:
            return AV_PIX_FMT_BGRA;
        case GST_VIDEO_FORMAT_RGBA:
            return AV_PIX_FMT_RGBA;
        case GST_VIDEO_FORMAT_ARGB:
            return AV_PIX_FMT_ARGB;
        case GST_VIDEO_FORMAT_ABGR:
            return AV_PIX_FMT_ABGR;
        case GST_VIDEO_FORMAT_I420:
            return AV_PIX_FMT_YUV420P;
        case GST_VIDEO_FORMAT_Y42B:
            return AV_PIX_FMT_YUV422P;
        case GST_VIDEO_FORMAT_Y444:
            return AV_PIX_FMT_YUV444P;
        case GST_VIDEO_FORMAT_UYVY:
            return AV_PIX_FMT_UYVY422;
        case GST_VIDEO_FORMAT_NV12:
            return AV_PIX_FMT_NV12;
        case GST_VIDEO_FORMAT_P010_10LE:
            // P010 keeps its 10 bits in the HIGH bits of each 16-bit word, so raw/65535 is
            // already the normalised value -- which is why av_util describes it as nv12 at
            // bit16 rather than bit10. See av_util.cpp's P010 case for why that is not a typo.
            return AV_PIX_FMT_P010LE;
        case GST_VIDEO_FORMAT_I420_10LE:
            return AV_PIX_FMT_YUV420P10LE;
        case GST_VIDEO_FORMAT_I422_10LE:
            return AV_PIX_FMT_YUV422P10LE;
        case GST_VIDEO_FORMAT_Y444_10LE:
            return AV_PIX_FMT_YUV444P10LE;
        default:
            return AV_PIX_FMT_NONE;
    }
}

core::color_space to_color_space(const GstVideoInfo& info)
{
    switch (GST_VIDEO_INFO_COLORIMETRY(&info).matrix) {
        case GST_VIDEO_COLOR_MATRIX_BT601:
            return core::color_space::bt601;
        case GST_VIDEO_COLOR_MATRIX_BT2020:
            return core::color_space::bt2020;
        default:
            // RGB and UNKNOWN included: the mixer's default, and the matrix is unused for RGB.
            return core::color_space::bt709;
    }
}

} // namespace

const char* supported_caps_formats()
{
    // This list is not "what GStreamer can produce" — it is exactly what ffmpeg::make_frame
    // can describe to the mixer, and it earns that narrowness. The x-padded variants (BGRx,
    // RGBx) are out because they map onto a format with alpha, and the padding byte is not
    // required to be opaque.
    //
    // NV12 and YUY2 were on this list in the upstream module, were the cause of a
    // hardware-decoded file playing as 164 perfectly black frames, and were removed:
    // `pixel_format_desc` returned `invalid` rather than failing, so the frame arrived with
    // no planes and the producer counted every one of them as *received*.
    //
    // **NV12 is back here, and P010 with it, because this tree is not that tree.** CasparVP
    // has `core::pixel_format::nv12` and both mixers carry a `case 15:` for it, so a
    // semi-planar buffer is described as two planes — Y as 1 component, CbCr as 2 at half
    // resolution — and the shader does the colour conversion. That matters beyond avoiding a
    // conversion: it is the *mixer's* conversion, with the channel's colour management, rather
    // than `videoconvert`'s guess at a matrix and a range.
    //
    // YUY2 is still out. Nothing in this tree describes it either.
    return "BGRA, RGBA, ARGB, ABGR, NV12, P010_10LE, I420, Y42B, Y444, UYVY, "
           "I420_10LE, I422_10LE, Y444_10LE";
}

std::shared_ptr<const core::frame_metadata> captions_of(GstBuffer* buffer)
{
    if (buffer == nullptr)
        return nullptr;

    // A buffer may carry more than one -- a source serving both 608 and 708 attaches one of
    // each -- so this iterates rather than taking `gst_buffer_get_video_caption_meta`, which
    // returns only the first and would silently drop the other.
    auto metadata = std::make_shared<core::frame_metadata>();
    gpointer state = nullptr;
    while (auto* meta = gst_buffer_iterate_meta_filtered(buffer, &state, GST_VIDEO_CAPTION_META_API_TYPE)) {
        auto* cc = reinterpret_cast<GstVideoCaptionMeta*>(meta);
        if (cc->data == nullptr || cc->size == 0)
            continue;
        core::frame_metadata::caption out;
        out.format = static_cast<int>(cc->caption_type);
        out.data.assign(cc->data, cc->data + cc->size);
        metadata->captions.push_back(std::move(out));
    }

    // Null rather than an empty object: `const_frame::metadata()` returns a shared empty for
    // the overwhelmingly common no-captions case, and allocating one per frame to say "nothing
    // here" would put a cost on the whole server to serve the few frames that carry any.
    return metadata->captions.empty() ? nullptr : metadata;
}

core::draw_frame make_frame(void*                    tag,
                            core::frame_factory&     frame_factory,
                            GstSample*               sample,
                            std::shared_ptr<AVFrame> audio)
{
    auto* caps   = gst_sample_get_caps(sample);
    auto* buffer = gst_sample_get_buffer(sample);

    if (caps == nullptr || buffer == nullptr)
        return core::draw_frame{};

    GstVideoInfo info;
    if (!gst_video_info_from_caps(&info, caps)) {
        CASPAR_LOG(warning) << L"[gstreamer] Sample carries caps that are not raw video.";
        return core::draw_frame{};
    }

    const auto pix_fmt = to_av_pixel_format(GST_VIDEO_INFO_FORMAT(&info));
    if (pix_fmt == AV_PIX_FMT_NONE) {
        CASPAR_LOG(warning) << L"[gstreamer] Unsupported video format "
                            << u16(std::string(gst_video_format_to_string(GST_VIDEO_INFO_FORMAT(&info))));
        return core::draw_frame{};
    }

    // The caps above should make this unreachable, but a format the mixer cannot describe
    // produces a frame with no planes rather than an error, and that renders as black with
    // nothing logged. Ask first.
    std::vector<int> data_map;
    if (ffmpeg::pixel_format_desc(pix_fmt, GST_VIDEO_INFO_WIDTH(&info), GST_VIDEO_INFO_HEIGHT(&info), data_map)
            .format == core::pixel_format::invalid) {
        CASPAR_LOG(warning) << L"[gstreamer] The mixer has no pixel format for "
                            << u16(std::string(gst_video_format_to_string(GST_VIDEO_INFO_FORMAT(&info))))
                            << L"; dropping the frame rather than rendering black.";
        return core::draw_frame{};
    }

    // gst_video_frame_map, rather than gst_buffer_map: it honours the GstVideoMeta a hardware
    // element may attach, so padded strides and per-plane offsets are read from the buffer
    // instead of being assumed from the caps.
    GstVideoFrame video_frame;
    if (!gst_video_frame_map(&video_frame, &info, buffer, GST_MAP_READ)) {
        CASPAR_LOG(warning) << L"[gstreamer] Failed to map a video buffer for reading.";
        return core::draw_frame{};
    }

    CASPAR_SCOPE_EXIT { gst_video_frame_unmap(&video_frame); };

    auto av_frame = ffmpeg::alloc_frame();
    av_frame->format = pix_fmt;
    av_frame->width  = GST_VIDEO_FRAME_WIDTH(&video_frame);
    av_frame->height = GST_VIDEO_FRAME_HEIGHT(&video_frame);

    const auto planes = GST_VIDEO_FRAME_N_PLANES(&video_frame);
    for (guint n = 0; n < planes && n < AV_NUM_DATA_POINTERS; ++n) {
        av_frame->data[n]     = static_cast<uint8_t*>(GST_VIDEO_FRAME_PLANE_DATA(&video_frame, n));
        av_frame->linesize[n] = GST_VIDEO_FRAME_PLANE_STRIDE(&video_frame, n);
    }

    // make_frame copies every plane into a frame the mixer owns, so the mapping above only has
    // to outlive this call — which is why unmapping on scope exit is safe.
    // `ffmpeg::make_frame` hands back a mutable_frame; metadata belongs on the const_frame,
    // after the hand-over, because that is the form the mixer and the consumers see.
    core::const_frame frame(
        ffmpeg::make_frame(tag, frame_factory, std::move(av_frame), std::move(audio), to_color_space(info)));

    if (auto captions = captions_of(buffer))
        frame = frame.with_metadata(std::move(captions));

    return core::draw_frame(std::move(frame));
}

}} // namespace caspar::gstreamer
