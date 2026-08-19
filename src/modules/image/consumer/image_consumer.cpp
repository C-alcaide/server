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
#include <common/filesystem.h>
#include <common/future.h>
#include <common/log.h>
#include <common/timer.h>

#include <core/consumer/channel_info.h>
#include <core/frame/frame.h>

#include <boost/algorithm/string.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>

#include <algorithm>
#include <fstream>
#include <optional>
#include <utility>
#include <vector>

#include <ffmpeg/util/av_assert.h>
#include <ffmpeg/util/av_util.h>

#include "../util/image_algorithms.h"
#include "../util/image_converter.h"
#include "../util/image_view.h"

extern "C" {
#define __STDC_CONSTANT_MACROS
#define __STDC_LIMIT_MACROS
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libavutil/pixfmt.h>
}

namespace caspar::image {

// Resolve a client-supplied IMAGE filename to its output path, or nothing when it
// escapes the media folder.
//
// The filename arrives from "ADD <channel> IMAGE <filename>" and the path used to
// be built by concatenation with no containment check:
//
//     filename2 = u8(env::media_folder() + filename + L".png");
//
// so "ADD 1 IMAGE ../../evil" wrote outside the media folder. This is the same
// defect bd14cab7a fixed for PRINT RAW -- which is the fork-added copy of this
// consumer -- and unlike the recording consumers (ffmpeg, replay) this one always
// prefixes the media folder and never honours an absolute path, so
// media-folder-relative is unambiguously its contract.
//
// Subdirectories stay legal ("ADD 1 IMAGE sub/name"): only escaping is rejected.
std::optional<boost::filesystem::path> resolve_image_output(const std::wstring& filename)
{
    boost::system::error_code ec;
    const auto                base = boost::filesystem::canonical(env::media_folder(), ec);
    if (ec)
        return {};

    // An empty filename means "timestamp it", which no client controls.
    const std::wstring leaf =
        filename.empty() ? boost::posix_time::to_iso_wstring(boost::posix_time::second_clock::local_time()) : filename;

    // weakly_canonical because the target does not exist yet, but the directories
    // leading to it do -- and only a real canonicalization resolves a symlink
    // planted inside the media folder. An absolute `leaf` replaces the base here,
    // and is then rejected by the containment check below.
    const auto resolved = boost::filesystem::weakly_canonical(base / (leaf + L".png"), ec);
    if (ec || !is_within_base(resolved, base))
        return {};

    return resolved;
}

struct image_consumer : public core::frame_consumer
{
    const std::wstring filename_;
    const std::string  ocio_display_;
    const std::string  ocio_view_;
    int                frames_waited_{0};

    explicit image_consumer(std::wstring filename, std::string display = {}, std::string view = {})
        : filename_(std::move(filename))
        , ocio_display_(std::move(display))
        , ocio_view_(std::move(view))
    {
    }

    /// This consumer's own view, if it was given one.
    ///
    /// The IMAGE consumer is the first to carry one because it is the one a measurement can
    /// read back: capturing through it while the channel holds a DIFFERENT view is what
    /// distinguishes "the mixer fanned out and output routed correctly" from "everyone got
    /// the channel's view", and it does so at the same 1 LSB gate as every other flat-patch
    /// battery.
    std::pair<std::string, std::string> ocio_view() const override
    {
        return {ocio_display_, ocio_view_};
    }

    void initialize(const core::video_format_desc& /*format_desc*/,
                    const core::channel_info& channel_info,
                    int                       port_index) override
    {
    }

    // How many ticks to wait for the mixer's CPU readback to land before giving up.
    //
    // Adding this consumer is itself what turns readback on -- output.cpp only reads
    // back when some consumer asks for it -- so the frames immediately after ADD
    // legitimately have no host pixels: the enable has to propagate and a composited
    // frame has to make the GPU->host round trip. The budget used to be 4 ticks, which
    // on a 1080i50 channel is 4 fields, i.e. 40 ms / two frames, and that was not
    // enough on a Vulkan channel that had readback switched off: roughly half the
    // captures in a deck+screen configuration timed out. Ticks are cheap (this
    // consumer only exists for the duration of one capture) so the budget is generous.
    static constexpr int max_wait_ticks = 50;

    /// Ticks to let a requested view actually get rendered before capturing.
    ///
    /// `video_channel` collects the consumers' views at the START of a tick and hands them
    /// to the mixer, so a consumer that attached DURING a tick is not in that set yet and
    /// the frame it receives is the channel's own view. Every long-lived consumer sorts
    /// itself out on the next tick; this one is one-shot and would capture the fallback
    /// every time.
    ///
    /// Measured 2026-08-13: without this, a capture asking for `Un-tone-mapped` on a channel
    /// showing `ACES 2.0` came back byte-identical to the channel's view -- the fallback,
    /// indistinguishable from the feature not working at all.
    static constexpr int view_settle_ticks = 3;

    std::future<bool> send(core::video_field field, core::const_frame frame) override
    {
        if (!ocio_display_.empty() && frames_waited_ < view_settle_ticks) {
            ++frames_waited_;
            return make_ready_future(true);
        }

        const auto& data = frame.image_data(0);
        if (data.data() == nullptr || data.size() == 0) {
            if (++frames_waited_ < max_wait_ticks) {
                return make_ready_future(true); // stay alive, wait for valid frame
            }
            // Give up -- but do NOT fall through to the encoder. It builds a
            // std::vector from image_data(0).begin() and .begin() + size, which on an
            // empty array is nullptr plus an offset, and it threw from there: the
            // capture produced no file, and the only trace was a stack dump reading
            // "Exception: No diagnostic information available." The previous message
            // here claimed it was "capturing empty frame", which it never did.
            //
            // Deliberately not writing a black placeholder either: a black PNG would
            // flow into an image comparison and read as a render regression, which is
            // a worse failure than an obviously absent file plus this message.
            CASPAR_LOG(error) << L"[image_consumer] No CPU frame data after " << frames_waited_
                              << L" ticks - giving up, no file written for '" << filename_ << L"'.";
            return make_ready_future(false);
        }

        auto filename = filename_;

        std::thread async([frame, filename] {
            try {
                const auto resolved = resolve_image_output(filename);
                if (!resolved) {
                    CASPAR_LOG(error) << L"[image_consumer] refusing to write outside the media folder: " << filename;
                    return;
                }
                const std::string filename2 = u8(resolved->wstring());

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

                // What the MIXER handed us, before this consumer touches a byte.
                //
                // Kept because it is what identified the swscale BGRA64->RGBA64 defect
                // described in docs/UPSTREAM_SYNC_2026-08-18.md section 3.1.1: everything
                // downstream of here is shared by both mixer backends, so logging the frame
                // as it arrives is what splits "the mixer produced this" from "this consumer
                // mangled it". No battery can currently distinguish those, and without this
                // line the defect reads as a 16-bit precision fault in the OGL backend.
                //
                // Trace level, so it costs nothing at the default log level.
                if (frame.image_data(0).size() > 0) {
                    const auto& d0  = frame.image_data(0);
                    const auto* p16 = reinterpret_cast<const std::uint16_t*>(d0.data());
                    const auto* p8  = d0.data();
                    CASPAR_LOG(trace) << L"[image_consumer] MIXER_OUT"
                                      << L" fmt=" << static_cast<int>(pix_desc.format)
                                      << L" planes=" << pix_desc.planes.size()
                                      << L" depth0=" << static_cast<int>(pix_desc.planes[0].depth)
                                      << L" hi_dep=" << (is_hi_dep ? 1 : 0)
                                      << L" bytes=" << d0.size()
                                      << L" first8="
                                      << (is_hi_dep && d0.size() >= 16
                                              ? (std::to_wstring(p16[0]) + L"," + std::to_wstring(p16[1]) + L"," +
                                                 std::to_wstring(p16[2]) + L"," + std::to_wstring(p16[3]) + L"," +
                                                 std::to_wstring(p16[4]) + L"," + std::to_wstring(p16[5]) + L"," +
                                                 std::to_wstring(p16[6]) + L"," + std::to_wstring(p16[7]))
                                          : d0.size() >= 8
                                              ? (std::to_wstring(p8[0]) + L"," + std::to_wstring(p8[1]) + L"," +
                                                 std::to_wstring(p8[2]) + L"," + std::to_wstring(p8[3]) + L"," +
                                                 std::to_wstring(p8[4]) + L"," + std::to_wstring(p8[5]) + L"," +
                                                 std::to_wstring(p8[6]) + L"," + std::to_wstring(p8[7]))
                                              : std::wstring(L"(short)"));
                }

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
                    // 8-bit mixer outputs packed BGRA (1 plane, 4 components)
                    if (is_hi_dep) {
                        av_frame->format      = AV_PIX_FMT_BGRA64LE;
                        av_frame->linesize[0] = static_cast<int>(frame.width()) * 8;
                    } else {
                        av_frame->format      = AV_PIX_FMT_BGRA;
                        av_frame->linesize[0] = static_cast<int>(frame.width()) * 4;
                    }
                    av_frame->data[0] = const_cast<uint8_t*>(frame.image_data(0).data());
                } else if (pix_desc.format == core::pixel_format::rgba) {
                    // 16-bit mixer outputs packed RGBA directly (no .bgra swizzle)
                    if (is_hi_dep) {
                        av_frame->format      = AV_PIX_FMT_RGBA64LE;
                        av_frame->linesize[0] = static_cast<int>(frame.width()) * 8;
                    } else {
                        av_frame->format      = AV_PIX_FMT_RGBA;
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
                if (is_hi_dep && (pix_desc.format == core::pixel_format::bgra ||
                                  pix_desc.format == core::pixel_format::rgba)) {
                    // Work on the native-endian source data directly.
                    // We need a writable copy since frame data is const.
                    auto  src_size = av_frame->linesize[0] * av_frame->height;
                    auto  buf      = std::vector<uint8_t>(frame.image_data(0).begin(),
                                                         frame.image_data(0).begin() + src_size);
                    auto* data     = reinterpret_cast<uint16_t*>(buf.data());
                    const int stride16 = av_frame->linesize[0] / 2; // in uint16_t units
                    const int w        = av_frame->width;
                    const int h        = av_frame->height;

                    // Permute B and R here, in the pass that is already running, rather than
                    // leaving it to swscale.
                    //
                    // swscale's packed-16 -> packed-16 COMPONENT PERMUTATION is lossy. Measured
                    // over a 256-pixel probe: bgra64le -> rgba64be deviates by up to 32 LSB16 on
                    // 734 of 1024 components, and no flag avoids it -- accurate_rnd, bitexact and
                    // every sws_dither setting give byte-identical error. Its ENDIAN SWAP is not
                    // lossy: rgba64le -> rgba64be is exact, which is precisely why only the
                    // OpenGL mixer's 16-bit captures were ever wrong and Vulkan's never were.
                    // (bgra64le -> gbrap16le -> rgba64be is exact too, and was the other
                    // candidate; it costs a second pass, and this costs nothing.)
                    //
                    // Doing it here is free. The loop already holds a writable copy and already
                    // reads and writes all three components, so the exchange adds no allocation,
                    // no pass over memory and no swscale work. It only relabels what the buffer
                    // is, leaving swscale the byte swap it performs exactly.
                    //
                    // Three earlier attempts at this were reverted for tripling the wall clock
                    // of a 16-bit capture run. That regression was not real: measured back to
                    // back on a settled box, two runs of the SAME build gave 258 s and 133 s,
                    // and the 133 s beat the unfixed control's 141 s. The 3x was a
                    // first-run-after-build warm-up artefact each time. Measure this path twice
                    // before believing a timing result from it -- see
                    // docs/PLAN_BGRA64_CAPTURE_FIX.md section 8.2.2.
                    const bool swap_rb = (pix_desc.format == core::pixel_format::bgra);

                    for (int y = 0; y < h; ++y) {
                        uint16_t* row = data + y * stride16;
                        for (int x = 0; x < w; ++x) {
                            // RGBA: R=0, G=1, B=2, A=3; BGRA: B=0, G=1, R=2, A=3
                            uint16_t& c0 = row[x * 4 + 0];
                            uint16_t& c1 = row[x * 4 + 1];
                            uint16_t& c2 = row[x * 4 + 2];
                            uint16_t  a  = row[x * 4 + 3];
                            if (a != 0 && a != 65535) {
                                c0 = static_cast<uint16_t>(std::min(65535, static_cast<int>(c0) * 65535 / a));
                                c1 = static_cast<uint16_t>(std::min(65535, static_cast<int>(c1) * 65535 / a));
                                c2 = static_cast<uint16_t>(std::min(65535, static_cast<int>(c2) * 65535 / a));
                            }
                            // Unconditional -- the exchange is about layout, not about alpha, so
                            // it must also happen for the fully opaque and fully clear pixels the
                            // un-premultiply above skips.
                            if (swap_rb)
                                std::swap(c0, c2);
                        }
                    }
                    av_frame->data[0] = buf.data();
                    if (swap_rb) {
                        av_frame->format = AV_PIX_FMT_RGBA64LE;
                    }

                    caspar::timer conv_timer;
                    auto          av_frame2 = convert_image_frame(av_frame, target_fmt);
                    const auto conv_ms = conv_timer.elapsed() * 1000.0;

                    caspar::timer enc_timer;
                    FF(avcodec_send_frame(ctx.get(), av_frame2.get()));
                    FF(avcodec_send_frame(ctx.get(), nullptr));
                    CASPAR_LOG(trace) << L"[image_consumer] TIMING convert=" << conv_ms
                                      << L"ms send=" << (enc_timer.elapsed() * 1000.0) << L"ms";

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

    // ADD <ch> IMAGE <name> ["<display>" "<view>"] -- this consumer's own OCIO view.
    //
    // Both or neither: a display without a view is not a transform, and accepting it
    // silently would render the channel's view while looking configured.
    std::string display, view;
    if (params.size() > 3) {
        display = u8(params.at(2));
        view    = u8(params.at(3));
    } else if (params.size() == 3) {
        CASPAR_THROW_EXCEPTION(user_error() << msg_info(
            L"IMAGE takes a display AND a view, both quoted, or neither."));
    }

    // Reject at ADD time as well as at write time, so the operator gets a failure
    // on the command rather than a silent no-op several frames later.
    if (!filename.empty() && !resolve_image_output(filename))
        CASPAR_THROW_EXCEPTION(user_error()
                               << msg_info(L"IMAGE filename must resolve inside the media folder: " + filename));

    return spl::make_shared<image_consumer>(filename, std::move(display), std::move(view));
}

} // namespace caspar::image
