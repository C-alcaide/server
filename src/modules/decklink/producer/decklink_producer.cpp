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
 */

#include "../StdAfx.h"

#include "decklink_producer.h"
#include "sdi_signalling.h"

#include "../util/util.h"

#include <chrono>
#include <sstream>
#include <iomanip>

#include <common/env.h>
#include <boost/property_tree/ptree.hpp>
#include <common/diagnostics/graph.h>
#include <common/except.h>
#include <common/executor.h>
#include <common/log.h>
#include <common/param.h>
#include <common/scope_exit.h>
#include <common/timer.h>

#include <ffmpeg/util/av_assert.h>
#include <ffmpeg/util/av_util.h>

#include <core/diagnostics/call_context.h>
#include <core/frame/draw_frame.h>
#include <core/frame/frame_factory.h>
#include <core/monitor/monitor.h>
#include <core/producer/frame_producer.h>

#include <boost/algorithm/string.hpp>
#include <boost/range/adaptor/transformed.hpp>

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4244)
#endif
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavfilter/avfilter.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
#include <libavformat/avformat.h>
#include <libavutil/buffer.h>
#include <libavutil/channel_layout.h>
#include <libavutil/opt.h>
#include <libavutil/pixfmt.h>
#include <libavutil/samplefmt.h>
#include <libavutil/timecode.h>
}
#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include <boost/format.hpp>

#include <mutex>
#include <thread>
#include <atomic>
#include <map>
#include <condition_variable>

#include "../decklink_api.h"

using namespace caspar::ffmpeg;

namespace caspar { namespace decklink {

class decklink_producer; // Forward declaration

class DeckLinkSyncManager
{
    struct Group
    {
        std::vector<decklink_producer*> producers;
        size_t                          expected_peers = 1;
        bool                            started        = false;
        std::mutex                      mutex;
    };

    std::map<int, std::shared_ptr<Group>> groups_;
    std::mutex                            manager_mutex_;
    std::condition_variable               monitor_cv_;
    std::thread                           monitor_thread_;
    std::atomic<bool>                     monitoring_{false};

    DeckLinkSyncManager() = default;

    void start_monitor_loop();

  public:
    static DeckLinkSyncManager& instance()
    {
        static DeckLinkSyncManager instance;
        return instance;
    }

    ~DeckLinkSyncManager()
    {
        monitoring_ = false;
        monitor_cv_.notify_all();
        if (monitor_thread_.joinable())
            monitor_thread_.join();
    }

    void register_producer(int group_id, int peers, decklink_producer* producer);
    void unregister_producer(int group_id, decklink_producer* producer);
};

struct Filter
{
    std::shared_ptr<AVFilterGraph> graph        = nullptr;
    AVFilterContext*               sink         = nullptr;
    AVFilterContext*               video_source = nullptr;
    AVFilterContext*               audio_source = nullptr;

    Filter() {}

    Filter(std::string                          filter_spec,
           AVMediaType                          type,
           const core::video_format_desc&       format_desc,
           const com_ptr<IDeckLinkDisplayMode>& dm,
           bool                                 hdr)
    {
        BMDTimeScale timeScale;
        BMDTimeValue frameDuration;
        dm->GetFrameRate(&frameDuration, &timeScale);

        if (type == AVMEDIA_TYPE_VIDEO) {
            if (filter_spec.empty()) {
                filter_spec = "null";
            }

            boost::rational<int> bmdFramerate(
                timeScale / 1000 * (dm->GetFieldDominance() == bmdProgressiveFrame ? 1 : 2), frameDuration / 1000);
            bool doFps = bmdFramerate != (format_desc.framerate * format_desc.field_count);
            bool i2p   = (dm->GetFieldDominance() != bmdProgressiveFrame) && (1 == format_desc.field_count);

            std::string deintStr = (doFps || i2p) ? ",bwdif=mode=send_field" : ",yadif=mode=send_field_nospatial";
            switch (dm->GetFieldDominance()) {
                case bmdUpperFieldFirst:
                    filter_spec += deintStr + ":parity=tff:deint=all";
                    break;
                case bmdLowerFieldFirst:
                    filter_spec += deintStr + ":parity=bff:deint=all";
                    break;
                case bmdUnknownFieldDominance:
                    filter_spec += deintStr + ":parity=auto:deint=interlaced";
                    break;
            }

            if (doFps) {
                filter_spec += (boost::format(",fps=%d/%d") % (format_desc.time_scale * format_desc.field_count) %
                                format_desc.duration)
                                   .str();
            }
        } else {
            if (filter_spec.empty()) {
                filter_spec = "anull";
            }
            filter_spec +=
                (boost::format(",aresample=sample_rate=%d:async=2000") % format_desc.audio_sample_rate).str();
        }

        AVFilterInOut* outputs = nullptr;
        AVFilterInOut* inputs  = nullptr;

        CASPAR_SCOPE_EXIT
        {
            avfilter_inout_free(&inputs);
            avfilter_inout_free(&outputs);
        };

        int video_input_count = 0;
        int audio_input_count = 0;
        {
            auto graph2 = avfilter_graph_alloc();
            if (!graph2) {
                FF_RET(AVERROR(ENOMEM), "avfilter_graph_alloc");
            }

            CASPAR_SCOPE_EXIT
            {
                avfilter_graph_free(&graph2);
                avfilter_inout_free(&inputs);
                avfilter_inout_free(&outputs);
            };

            FF(avfilter_graph_parse2(graph2, filter_spec.c_str(), &inputs, &outputs));

            for (auto cur = inputs; cur; cur = cur->next) {
                const auto filter_type = avfilter_pad_get_type(cur->filter_ctx->input_pads, cur->pad_idx);
                if (filter_type == AVMEDIA_TYPE_VIDEO) {
                    video_input_count += 1;
                } else if (filter_type == AVMEDIA_TYPE_AUDIO) {
                    audio_input_count += 1;
                }
            }
        }

        graph = std::shared_ptr<AVFilterGraph>(avfilter_graph_alloc(),
                                               [](AVFilterGraph* ptr) { avfilter_graph_free(&ptr); });

        if (!graph) {
            FF_RET(AVERROR(ENOMEM), "avfilter_graph_alloc");
        }

        FF(avfilter_graph_parse2(graph.get(), filter_spec.c_str(), &inputs, &outputs));

        auto pix_fmt = (hdr ? AV_PIX_FMT_YUV422P10 : AV_PIX_FMT_UYVY422);
        for (auto cur = inputs; cur; cur = cur->next) {
            const auto filter_type = avfilter_pad_get_type(cur->filter_ctx->input_pads, cur->pad_idx);

            if (filter_type == AVMEDIA_TYPE_VIDEO) {
                if (video_source) {
                    CASPAR_THROW_EXCEPTION(ffmpeg_error_t() << boost::errinfo_errno(EINVAL)
                                                            << msg_info_t("only single video input supported"));
                }
                const auto sar = boost::rational<int>(format_desc.square_width, format_desc.square_height) /
                                 boost::rational<int>(format_desc.width, format_desc.height);

                auto args =
                    (boost::format("video_size=%dx%d:pix_fmt=%d:time_base=%d/%d:sar=%d/%d:frame_rate=%d/%d") %
                     dm->GetWidth() % dm->GetHeight() % pix_fmt % 1 % AV_TIME_BASE % sar.numerator() %
                     sar.denominator() % (timeScale / 1000 * (dm->GetFieldDominance() == bmdProgressiveFrame ? 1 : 2)) %
                     (frameDuration / 1000))
                        .str();
                auto name = (boost::format("in_%d") % 0).str();

                FF(avfilter_graph_create_filter(
                    &video_source, avfilter_get_by_name("buffer"), name.c_str(), args.c_str(), nullptr, graph.get()));
                FF(avfilter_link(video_source, 0, cur->filter_ctx, cur->pad_idx));
            } else if (filter_type == AVMEDIA_TYPE_AUDIO) {
                if (audio_source) {
                    CASPAR_THROW_EXCEPTION(ffmpeg_error_t() << boost::errinfo_errno(EINVAL)
                                                            << msg_info_t("only single audio input supported"));
                }

                auto args = (boost::format("time_base=%d/%d:sample_rate=%d:sample_fmt=%s:channel_layout=%#x") % 1 %
                             format_desc.audio_sample_rate % format_desc.audio_sample_rate % AV_SAMPLE_FMT_S32 %
                             ffmpeg::get_channel_layout_mask_for_channels(format_desc.audio_channels))
                                .str();
                auto name = (boost::format("in_%d") % 0).str();

                FF(avfilter_graph_create_filter(
                    &audio_source, avfilter_get_by_name("abuffer"), name.c_str(), args.c_str(), nullptr, graph.get()));
                FF(avfilter_link(audio_source, 0, cur->filter_ctx, cur->pad_idx));
            } else {
                CASPAR_THROW_EXCEPTION(ffmpeg_error_t() << boost::errinfo_errno(EINVAL)
                                                        << msg_info_t("only video and audio filters supported"));
            }
        }

        if (type == AVMEDIA_TYPE_VIDEO) {
            FF(avfilter_graph_create_filter(
                &sink, avfilter_get_by_name("buffersink"), "out", nullptr, nullptr, graph.get()));

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4245)
#endif
            AVPixelFormat pix_fmts[] = {pix_fmt, AV_PIX_FMT_NONE};
            FF(av_opt_set_int_list(sink, "pix_fmts", pix_fmts, -1, AV_OPT_SEARCH_CHILDREN));
#ifdef _MSC_VER
#pragma warning(pop)
#endif
        } else if (type == AVMEDIA_TYPE_AUDIO) {
            FF(avfilter_graph_create_filter(
                &sink, avfilter_get_by_name("abuffersink"), "out", nullptr, nullptr, graph.get()));
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4245)
#endif

            AVSampleFormat sample_fmts[]  = {AV_SAMPLE_FMT_S32, AV_SAMPLE_FMT_NONE};
            int            sample_rates[] = {format_desc.audio_sample_rate, 0};
            FF(av_opt_set_int_list(sink, "sample_fmts", sample_fmts, -1, AV_OPT_SEARCH_CHILDREN));
            FF(av_opt_set_int_list(sink, "sample_rates", sample_rates, 0, AV_OPT_SEARCH_CHILDREN));

            // TODO - we might want to force the filter to produce 16 channels
            // But this segfaults (changing the property name causes it to fail with an error)
            // As 16 channel packets are fed into the filter, with the filter set to the same, that is what we get out
            /*
            AVChannelLayout channel_layout = AV_CHANNEL_LAYOUT_STEREO;
            av_channel_layout_default(&channel_layout, format_desc.audio_channels);

            FF(av_opt_set_chlayout(sink, "ch_layouts", &channel_layout, AV_OPT_SEARCH_CHILDREN));
            av_channel_layout_uninit(&channel_layout);
             */

#ifdef _MSC_VER
#pragma warning(pop)
#endif
        } else {
            CASPAR_THROW_EXCEPTION(ffmpeg_error_t()
                                   << boost::errinfo_errno(EINVAL) << msg_info_t("invalid output media type"));
        }

        {
            const auto cur = outputs;

            if (!cur || cur->next) {
                CASPAR_THROW_EXCEPTION(ffmpeg_error_t() << boost::errinfo_errno(EINVAL)
                                                        << msg_info_t("invalid filter graph output count"));
            }

            if (avfilter_pad_get_type(cur->filter_ctx->output_pads, cur->pad_idx) != type) {
                CASPAR_THROW_EXCEPTION(ffmpeg_error_t() << boost::errinfo_errno(EINVAL)
                                                        << msg_info_t("invalid filter output media type"));
            }

            FF(avfilter_link(cur->filter_ctx, cur->pad_idx, sink, 0));
        }

        FF(avfilter_graph_config(graph.get(), nullptr));

        CASPAR_LOG(debug) << avfilter_graph_dump(graph.get(), nullptr);
    }
};

struct Decoder
{
    Decoder(const Decoder&) = delete;

    bool hdr_ = false;

  public:
    std::shared_ptr<AVCodecContext> ctx;

    Decoder() = default;

    explicit Decoder(bool hdr, const com_ptr<IDeckLinkDisplayMode>& mode)
        : hdr_(hdr)
    {
        const auto codec = avcodec_find_decoder(AV_CODEC_ID_V210);
        if (!codec) {
            FF_RET(AVERROR_DECODER_NOT_FOUND, "avcodec_find_decoder");
        }

        ctx = std::shared_ptr<AVCodecContext>(avcodec_alloc_context3(codec),
                                              [](AVCodecContext* ptr) { avcodec_free_context(&ptr); });
        if (!ctx) {
            FF_RET(AVERROR(ENOMEM), "avcodec_alloc_context3");
        }

        auto params = std::shared_ptr<AVCodecParameters>(avcodec_parameters_alloc(),
                                                         [](AVCodecParameters* ptr) { avcodec_parameters_free(&ptr); });
        if (!params) {
            FF_RET(AVERROR(ENOMEM), "avcodec_parameters_alloc");
        }
        params->width      = mode->GetWidth();
        params->height     = mode->GetHeight();
        params->codec_type = AVMEDIA_TYPE_VIDEO;
        params->codec_id   = AV_CODEC_ID_V210;
        params->format     = AV_PIX_FMT_YUV422P10;

        FF(avcodec_parameters_to_context(ctx.get(), params.get()));

        // int thread_count = env::properties().get(L"configuration.ffmpeg.producer.threads", 0);
        FF(av_opt_set_image_size(ctx.get(), "video_size", mode->GetWidth(), mode->GetHeight(), 0));
        FF(avcodec_open2(ctx.get(), codec, nullptr));
    }

    std::shared_ptr<AVFrame> decode(IDeckLinkVideoInputFrame* video, const com_ptr<IDeckLinkDisplayMode>& mode)
    {
        void* video_bytes = nullptr;
        if (SUCCEEDED(video->GetBytes(&video_bytes)) && video_bytes) {
            video->AddRef();

            auto frame = std::shared_ptr<AVFrame>(av_frame_alloc(), [video](AVFrame* ptr) {
                video->Release();
                av_frame_free(&ptr);
            });
            if (!frame)
                FF_RET(AVERROR(ENOMEM), "av_frame_alloc");

            if (hdr_) {
                const auto size = video->GetRowBytes() * video->GetHeight();
                AVPacket   packet;
                av_init_packet(&packet);
                packet.data = reinterpret_cast<uint8_t*>(video_bytes);
                packet.size = size;
                FF(avcodec_send_packet(ctx.get(), &packet));
                FF(avcodec_receive_frame(ctx.get(), frame.get()));
            } else {
                frame->format      = AV_PIX_FMT_UYVY422;
                frame->width       = video->GetWidth();
                frame->height      = video->GetHeight();
                frame->data[0]     = reinterpret_cast<uint8_t*>(video_bytes);
                frame->linesize[0] = video->GetRowBytes();

                // Refcount the card's DMA buffer so the filter graph references it instead of
                // copying it. av_buffersrc_write_frame is add_frame_flags(..., KEEP_REF), which
                // av_frame_ref's the input -- and a frame with buf[0] == nullptr cannot be
                // referenced, so it gets allocated and memcpy'd wholesale. That was a full
                // 4.15 MB copy per 1080p frame per input, on the driver's callback thread,
                // happening purely because this AVFrame was unmanaged.
                //
                // READONLY because the memory belongs to the driver: a filter that needs to
                // write must then copy-on-write rather than scribble on the capture buffer.
                {
                    const int plane_size = video->GetRowBytes() * video->GetHeight();
                    video->AddRef(); // released by the AVBufferRef's free callback below
                    frame->buf[0] = av_buffer_create(
                        frame->data[0],
                        plane_size,
                        [](void* opaque, uint8_t*) { static_cast<IDeckLinkVideoInputFrame*>(opaque)->Release(); },
                        video,
                        AV_BUFFER_FLAG_READONLY);
                    if (!frame->buf[0]) {
                        // Ownership never transferred; drop the ref we just took. The frame is
                        // still usable, it just falls back to being copied by buffersrc.
                        video->Release();
                    }
                }
#if LIBAVCODEC_VERSION_MAJOR < 61
                frame->key_frame = 1;
#else
                frame->flags |= AV_FRAME_FLAG_KEY;
#endif
            }

#if LIBAVCODEC_VERSION_MAJOR < 61
            frame->interlaced_frame = mode->GetFieldDominance() != bmdProgressiveFrame;
            frame->top_field_first  = mode->GetFieldDominance() == bmdUpperFieldFirst ? 1 : 0;
#else
            frame->flags |= mode->GetFieldDominance() != bmdProgressiveFrame ? AV_FRAME_FLAG_INTERLACED : 0;
            frame->flags |= mode->GetFieldDominance() == bmdUpperFieldFirst ? AV_FRAME_FLAG_TOP_FIELD_FIRST : 0;
#endif

            return frame;
        }
        return nullptr;
    }
};

/// `specified`, when given, reports whether the CARD supplied colorspace metadata, as
/// opposed to the caller's fallback being used. The mixers need that: an SDI input
/// carries no colour space in the payload, so a sub-720 signal with no metadata is
/// conventionally BT.601, while metadata that says BT.709 must be honoured whatever the
/// raster size.
const wchar_t* metadata_name(core::color_space cs)
{
    switch (cs) {
        case core::color_space::bt601:  return L"bt601";
        case core::color_space::bt709:  return L"bt709";
        case core::color_space::bt2020: return L"bt2020";
        case core::color_space::p3_d65: return L"p3-d65";
        case core::color_space::p3_dci: return L"p3-dci";
        default:                        return L"unknown";
    }
}

const wchar_t* metadata_name(core::color_transfer ct)
{
    switch (ct) {
        case core::color_transfer::pq:  return L"pq";
        case core::color_transfer::hlg: return L"hlg";
        default:                        return L"sdr";
    }
}

/// SMPTE ST 291-1 identifiers for the two packets that carry colour signalling on SDI.
constexpr uint8_t ANC_DID_SMPTE_291      = 0x41;
constexpr uint8_t ANC_SDID_ST352_VPID    = 0x01;
constexpr uint8_t ANC_SDID_ST2108_HDRWCG = 0x0C;

/// Pull ST 352 and ST 2108-1 off an incoming frame.
///
/// `census`, when given, is filled with every DID/SDID present rather than only the two we
/// decode. It is what turns "the signalling did not arrive" into an answerable question:
/// an empty census means the card handed us no ancillary data at all (on legacy hardware
/// `IDeckLinkVideoFrameAncillaryPackets` needs a v210 input -- see
/// `BMDDeckLinkVANCRequires10BitYUVVideoFrames`), whereas a census listing packets that do
/// not include 41h/01h means the SOURCE is not signalling. Those two have identical symptoms
/// and completely different fixes, and guessing between them is how this area went wrong the
/// first time.
sdi_signalling read_sdi_signalling(IDeckLinkVideoInputFrame* video, std::wstring* census = nullptr)
{
    sdi_signalling out;

    // A BLIND SPOT THIS FUNCTION HAD UNTIL 2026-08-16, and it mattered: every failure below
    // used to return the same empty census, so "the input frame does not offer the ancillary
    // interface at all" was indistinguishable from "it offers it and the frame carried
    // nothing". Those have different causes -- the first is the capture not being in a
    // VANC-capable pixel format, or a device that does not do ancillary capture; the second
    // is a genuinely silent sender -- and reporting them identically is how an entire
    // investigation ends up aimed at the wrong end of the link.
    if (census != nullptr) {
        wchar_t buf[64];
        swprintf(buf, 64, L" [input pixfmt %08X]", static_cast<unsigned>(video->GetPixelFormat()));
        *census += buf;
    }

    // THE INTEROP HEADER IN THIS TREE IS SDK 12.3.1 AND THE DRIVER IS 15.3, and the ancillary
    // packet API was revised in between. `IDeckLinkAncillaryPacket` gained `GetDataSpace()`,
    // which changed its IID, which cascaded to the iterator and to this container -- the
    // previous three are what SDK 15.3 now calls `..._v15_2`, and `IID_...` in our header is
    // the v15_2 value. Querying a 15.3 input frame with it returns E_NOINTERFACE, which is
    // exactly the "interface not offered" this function reported.
    //
    // The new method was APPENDED, so the vtable prefix is unchanged and the five methods
    // used here are identical in both revisions. Casting the new interface to the old
    // declaration is therefore safe for everything below; what would NOT be safe is calling
    // GetDataSpace through it, and nothing here does.
    //
    // Preferring the new IID and falling back keeps this working against both driver
    // generations, which matters because the fix belongs in a regenerated interop header and
    // this is not that.
    static const GUID IID_AncillaryPackets_15_3 = {
        0x8A72D630, 0x8070, 0x4D05, {0x8A, 0x93, 0xE6, 0x0C, 0x40, 0xEE, 0x08, 0x8A}};

    // TWO CONTROLS, because "QueryInterface said no" is only meaningful if QueryInterface
    // ever says yes on this object. Both of these IIDs are byte-identical between the 12.3.1
    // header this module is built from and the 15.3 driver installed here -- checked, not
    // assumed -- so a failure from either is about the frame rather than about a stale IID.
    //
    //   metadata extensions : the interface the colour readers use. If this succeeds, the
    //                         input frame does answer optional QueryInterface calls, and the
    //                         ancillary refusal below is a real statement about ancillary.
    //   legacy ancillary    : IDeckLinkVideoFrameAncillary, the pre-packet line-buffer
    //                         interface. If THIS succeeds while the packet interfaces do not,
    //                         the card is capturing VANC and only the modern accessor is
    //                         missing -- which would be a way in that costs no migration.
    if (census != nullptr) {
        IUnknown* probe = nullptr;
        if (SUCCEEDED(video->QueryInterface(IID_IDeckLinkVideoFrameMetadataExtensions, (void**)&probe)) &&
            probe != nullptr) {
            probe->Release();
            *census += L" [metadata-extensions: yes]";
        } else {
            *census += L" [metadata-extensions: NO]";
        }
        // NOT a QueryInterface. `IDeckLinkVideoFrameAncillary` is reached through
        // `IDeckLinkVideoFrame::GetAncillaryData`, a plain method on the frame -- asking for
        // it by IID fails on every frame ever made, which is a fact about the question and
        // not about the card. Getting that wrong once already produced a "legacy-ancillary:
        // NO" that meant nothing.
        IDeckLinkVideoFrameAncillary* legacy = nullptr;
        if (SUCCEEDED(video->GetAncillaryData(&legacy)) && legacy != nullptr) {
            wchar_t buf[64];
            swprintf(buf, 64, L" [legacy-ancillary: yes, pixfmt %08X]", static_cast<unsigned>(legacy->GetPixelFormat()));
            *census += buf;
            legacy->Release();
        } else {
            *census += L" [legacy-ancillary: NO]";
        }
    }

    IDeckLinkVideoFrameAncillaryPackets* raw = nullptr;
    if (FAILED(video->QueryInterface(IID_AncillaryPackets_15_3, (void**)&raw)) || raw == nullptr) {
        if (FAILED(video->QueryInterface(IID_IDeckLinkVideoFrameAncillaryPackets, (void**)&raw)) ||
            raw == nullptr) {
            if (census != nullptr) {
                *census += L" ANCILLARY INTERFACE NOT OFFERED BY THE INPUT FRAME (neither 15.3 nor v15_2 IID)";
            }
            return out;
        }
        if (census != nullptr) {
            *census += L" [v15_2 ancillary IID]";
        }
    }
    auto packets = wrap_raw<com_ptr>(raw, true);
    out.anc_interface_available = true;

    IDeckLinkAncillaryPacketIterator* raw_iter = nullptr;
    if (FAILED(packets->GetPacketIterator(&raw_iter)) || raw_iter == nullptr) {
        if (census != nullptr) {
            *census += L" INTERFACE PRESENT BUT NO ITERATOR";
        }
        return out;
    }
    auto iter = wrap_raw<com_ptr>(raw_iter, true);

    IDeckLinkAncillaryPacket* raw_packet = nullptr;
    while (SUCCEEDED(iter->Next(&raw_packet)) && raw_packet != nullptr) {
        auto packet = wrap_raw<com_ptr>(raw_packet, true);
        raw_packet  = nullptr;

        const uint8_t did  = packet->GetDID();
        const uint8_t sdid = packet->GetSDID();

        if (census != nullptr) {
            wchar_t buf[32];
            swprintf(buf, 32, L" %02Xh/%02Xh", did, sdid);
            *census += buf;
        }

        if (did != ANC_DID_SMPTE_291) {
            continue;
        }

        const void*  data = nullptr;
        unsigned int size = 0;
        // UInt8 asks for the payload as b7-b0 of each word, which is the form both standards
        // are written in -- ST 352's own note warns that the payload uses the two LSBs of the
        // 10-bit packet, so taking the 10-bit form here would mean shifting it back.
        if (FAILED(packet->GetBytes(bmdAncillaryPacketFormatUInt8, &data, &size)) || data == nullptr) {
            continue;
        }
        const auto* bytes = static_cast<const uint8_t*>(data);

        if (sdid == ANC_SDID_ST352_VPID && size >= 4) {
            parse_vpid(bytes, out);
        } else if (sdid == ANC_SDID_ST2108_HDRWCG && size >= 2) {
            parse_st2108(bytes, size, out);
        }
    }

    return out;
}

core::color_space get_color_space(IDeckLinkVideoInputFrame* video,
                                 core::color_space         fallback = core::color_space::unknown)
{
    IDeckLinkVideoFrameMetadataExtensions* md = nullptr;

    if (SUCCEEDED(video->QueryInterface(IID_IDeckLinkVideoFrameMetadataExtensions, (void**)&md))) {
        auto     metadata = wrap_raw<com_ptr>(md, true);
        LONGLONG color_space;
        if (SUCCEEDED(md->GetInt(bmdDeckLinkFrameMetadataColorspace, &color_space))) {
            if (color_space == bmdColorspaceRec2020) {
                return core::color_space::bt2020;
            } else if (color_space == bmdColorspaceRec601) {
                return core::color_space::bt601;
            }
            return core::color_space::bt709;
        }
    }

    // Metadata not available (typical for SDI inputs) -- the caller's fallback, which is
    // `unknown` for an SDR input, so the mixer's SD convention still applies to untagged
    // sub-720 signals. That is what an SD-SDI feed wants.
    return fallback;
}

core::color_transfer get_color_transfer(IDeckLinkVideoInputFrame* video, core::color_transfer fallback = core::color_transfer::sdr)
{
    IDeckLinkVideoFrameMetadataExtensions* md = nullptr;

    if (SUCCEEDED(video->QueryInterface(IID_IDeckLinkVideoFrameMetadataExtensions, (void**)&md))) {
        auto     metadata = wrap_raw<com_ptr>(md, true);
        LONGLONG eotf;
        if (SUCCEEDED(md->GetInt(bmdDeckLinkFrameMetadataHDRElectroOpticalTransferFunc, &eotf))) {
            if (eotf == 2) {        // CEA 861.3: PQ (ST 2084)
                return core::color_transfer::pq;
            } else if (eotf == 3) { // CEA 861.3: HLG (ARIB STD-B67)
                return core::color_transfer::hlg;
            }
            return core::color_transfer::sdr;
        }
    }

    // Metadata not available (typical for SDI inputs) — use caller's fallback.
    return fallback;
}

com_ptr<IDeckLinkDisplayMode> get_display_mode(const com_iface_ptr<IDeckLinkInput>& device,
                                               BMDDisplayMode                       format,
                                               BMDPixelFormat                       pix_fmt,
                                               BMDSupportedVideoModeFlags           flag)
{
    IDeckLinkDisplayMode*         m = nullptr;
    IDeckLinkDisplayModeIterator* iter;
    if (SUCCEEDED(device->GetDisplayModeIterator(&iter))) {
        auto iterator = wrap_raw<com_ptr>(iter, true);
        while (SUCCEEDED(iterator->Next(&m)) && m != nullptr && m->GetDisplayMode() != format) {
            m->Release();
        }
    }

    if (!m)
        CASPAR_THROW_EXCEPTION(user_error()
                               << msg_info("Device could not find requested video-format: " + std::to_string(format)));

    com_ptr<IDeckLinkDisplayMode> mode = wrap_raw<com_ptr>(m, true);

    BMDDisplayMode actualMode = bmdModeUnknown;
    BOOL           supported  = false;

    if (FAILED(device->DoesSupportVideoMode(bmdVideoConnectionUnspecified,
                                            mode->GetDisplayMode(),
                                            pix_fmt,
                                            bmdNoVideoInputConversion,
                                            flag,
                                            &actualMode,
                                            &supported)))
        CASPAR_THROW_EXCEPTION(caspar_exception()
                               << msg_info(L"Could not determine whether device supports requested video format: " +
                                           get_mode_name(mode)));
    else if (!supported)
        CASPAR_LOG(info) << L"Device may not support video-format: " << get_mode_name(mode);
    else if (actualMode != bmdModeUnknown)
        CASPAR_LOG(warning) << L"Device supports video-format with conversion: " << get_mode_name(mode);

    return mode;
}
static com_ptr<IDeckLinkDisplayMode> get_display_mode(const com_iface_ptr<IDeckLinkInput>& device,
                                                      core::video_format                   fmt,
                                                      BMDPixelFormat                       pix_fmt,
                                                      BMDSupportedVideoModeFlags           flag)
{
    return get_display_mode(device, get_decklink_video_format(fmt), pix_fmt, flag);
}

BMDPixelFormat get_pixel_format2(bool hdr) { return hdr ? bmdFormat10BitYUV : bmdFormat8BitYUV; }

class SharedDeckLinkInput : public IDeckLinkInputCallback
{
    int                           device_index_;
    com_ptr<IDeckLink>            decklink_;
    com_iface_ptr<IDeckLinkInput> input_;

    mutable std::recursive_mutex          mutex_;
    std::vector<IDeckLinkInputCallback*> listeners_;

    bool           enabled_          = false;
    int            enable_ref_count_ = 0;
    bool           started_          = false;
    int            start_ref_count_  = 0;
    BMDDisplayMode current_display_mode_;
    BMDPixelFormat current_pixel_format_;
    bool           audio_enabled_ = false;

    std::atomic<long> ref_count_{1};

  public:
    SharedDeckLinkInput(int device_index)
        : device_index_(device_index)
    {
        decklink_ = get_device(device_index_);
        input_    = iface_cast<IDeckLinkInput>(decklink_);
    }

    ~SharedDeckLinkInput()
    {
        // Ensure stopped
        if (started_) {
            try {
                input_->StopStreams();
            } catch (...) {}
        }
        if (enabled_) {
            try {
                input_->DisableVideoInput();
                input_->SetCallback(nullptr);
            } catch (...) {}
        }
    }

    void enable_video_input(BMDDisplayMode displayMode, BMDPixelFormat pixelFormat, BMDVideoInputFlags flags)
    {
        std::lock_guard<std::recursive_mutex> lock(mutex_);
        if (enabled_) {
            // Already running — a second producer is joining the shared input.
            // Do NOT call EnableVideoInput while streams are running; format detection
            // already has the correct mode. The new listener will receive frames at
            // the current format and can call get_current_display_mode() to initialize.
        } else {
            if (FAILED(input_->EnableVideoInput(displayMode, pixelFormat, flags))) {
                CASPAR_THROW_EXCEPTION(ffmpeg_error_t() << boost::errinfo_api_function("EnableVideoInput"));
            }
            current_display_mode_ = displayMode;
            current_pixel_format_ = pixelFormat;
            enabled_              = true;
            input_->SetCallback(this);
        }
        enable_ref_count_++;
    }

    BMDDisplayMode get_current_display_mode() const
    {
        std::lock_guard<std::recursive_mutex> lock(mutex_);
        return current_display_mode_;
    }

    void enable_audio_input(BMDAudioSampleRate sampleRate, BMDAudioSampleType sampleType, uint32_t channelCount)
    {
        std::lock_guard<std::recursive_mutex> lock(mutex_);
        if (enabled_ && audio_enabled_) {
             // Check compatibility... (omitted for brevity, assume compatible)
        }
        
        if (!audio_enabled_) {
             if (FAILED(input_->EnableAudioInput(sampleRate, sampleType, channelCount))) {
                  CASPAR_THROW_EXCEPTION(ffmpeg_error_t() << boost::errinfo_api_function("EnableAudioInput"));
             }
             audio_enabled_ = true;
        }
    }

    void disable_video_input()
    {
        std::lock_guard<std::recursive_mutex> lock(mutex_);
        if (enable_ref_count_ > 0) {
            enable_ref_count_--;
            if (enable_ref_count_ == 0 && enabled_) {
                input_->DisableVideoInput();
                input_->SetCallback(nullptr);
                enabled_ = false;
            }
        }
    }

    void start_streams()
    {
        std::lock_guard<std::recursive_mutex> lock(mutex_);
        if (!started_) {
            if (FAILED(input_->StartStreams())) {
                CASPAR_THROW_EXCEPTION(ffmpeg_error_t() << boost::errinfo_api_function("StartStreams"));
            }
            started_ = true;
        }
        start_ref_count_++;
    }

    void stop_streams()
    {
        std::lock_guard<std::recursive_mutex> lock(mutex_);
        if (start_ref_count_ > 0) {
            start_ref_count_--;
            if (start_ref_count_ == 0 && started_) {
                input_->StopStreams();
                started_ = false;
            }
        }
    }

    void add_listener(IDeckLinkInputCallback* cb)
    {
        std::lock_guard<std::recursive_mutex> lock(mutex_);
        listeners_.push_back(cb);
    }

    void remove_listener(IDeckLinkInputCallback* cb)
    {
        std::lock_guard<std::recursive_mutex> lock(mutex_);
        listeners_.erase(std::remove(listeners_.begin(), listeners_.end(), cb), listeners_.end());
    }

    // IDeckLinkInputCallback
    HRESULT STDMETHODCALLTYPE VideoInputFrameArrived(IDeckLinkVideoInputFrame* video,
                                                     IDeckLinkAudioInputPacket* audio) override
    {
        std::lock_guard<std::recursive_mutex> lock(mutex_);
        for (auto cb : listeners_) {
            cb->VideoInputFrameArrived(video, audio);
        }
        return S_OK;
    }

    HRESULT STDMETHODCALLTYPE VideoInputFormatChanged(BMDVideoInputFormatChangedEvents events,
                                                      IDeckLinkDisplayMode*            mode,
                                                      BMDDetectedVideoInputFormatFlags flags) override
    {
        std::lock_guard<std::recursive_mutex> lock(mutex_);

        if (!enabled_)
            return S_OK;

        // Only react to display mode changes. Colorspace-only events (pixel format mismatch
        // between requested and signal format) fire repeatedly and must be ignored — they
        // would cause PauseStreams+StartStreams on every frame, resetting the stream clock to 0.
        if (!(events & bmdVideoInputDisplayModeChanged))
            return S_OK;

        auto newMode = mode->GetDisplayMode();

        // Guard against repeated format-detection callbacks for the same mode.
        if (newMode == current_display_mode_)
            return S_OK;

        // BMD-recommended sequence: PauseStreams -> EnableVideoInput -> FlushStreams -> StartStreams
        input_->PauseStreams();
        if (SUCCEEDED(input_->EnableVideoInput(newMode, current_pixel_format_, bmdVideoInputEnableFormatDetection))) {
            current_display_mode_ = newMode;
        }
        input_->FlushStreams();

        // Notify listeners BEFORE StartStreams so they can update mode_ and rebuild filters.
        // If StartStreams is called first, frames arrive with the new dimensions while mode_
        // still holds the old value, causing the dimension-guard to discard every frame.
        for (auto cb : listeners_) {
            cb->VideoInputFormatChanged(events, mode, flags);
        }

        if (started_) {
            input_->StartStreams();
        }
        return S_OK;
    }

    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID iid, LPVOID* ppv) override
    {
        if (!ppv)
            return E_POINTER;
        *ppv = nullptr;
        if (iid == IID_IUnknown || iid == IID_IDeckLinkInputCallback) {
            *ppv = static_cast<IDeckLinkInputCallback*>(this);
            AddRef();
            return S_OK;
        }
        return E_NOINTERFACE;
    }
    ULONG STDMETHODCALLTYPE AddRef() override { return 1; }
    ULONG STDMETHODCALLTYPE Release() override { return 1; }
};

class DeckLinkInputManager
{
    std::map<int, std::weak_ptr<SharedDeckLinkInput>> inputs_;
    std::mutex                                        mutex_;

    DeckLinkInputManager() = default;

  public:
    static DeckLinkInputManager& instance()
    {
        static DeckLinkInputManager inst;
        return inst;
    }

    std::shared_ptr<SharedDeckLinkInput> get(int device_index)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        std::shared_ptr<SharedDeckLinkInput> ptr = inputs_[device_index].lock();
        if (!ptr) {
            ptr                   = std::make_shared<SharedDeckLinkInput>(device_index);
            inputs_[device_index] = ptr;
        }
        return ptr;
    }
};

class decklink_producer : public IDeckLinkInputCallback
{
    const int                           device_index_;
    core::monitor::state                state_;
    mutable std::mutex                  state_mutex_;
    spl::shared_ptr<diagnostics::graph> graph_;
    caspar::timer                       tick_timer_;

    std::shared_ptr<SharedDeckLinkInput>      shared_input_ = DeckLinkInputManager::instance().get(device_index_);
    com_ptr<IDeckLink>                        decklink_     = get_device(device_index_);
    com_iface_ptr<IDeckLinkInput>             input_        = iface_cast<IDeckLinkInput>(decklink_);
    com_iface_ptr<IDeckLinkProfileAttributes> attributes_   = iface_cast<IDeckLinkProfileAttributes>(decklink_);

    const std::wstring model_name_ = get_model_name(decklink_);

    core::video_format_desc              format_desc_;
    std::vector<int>                     audio_cadence_ = format_desc_.audio_cadence;
    spl::shared_ptr<core::frame_factory> frame_factory_;
    const core::video_format_repository  format_repository_;

    int64_t frame_count_ = 0;

    double in_sync_  = 0.0;
    double out_sync_ = 0.0;

    bool freeze_on_lost_;
    bool has_signal_ = false;
    bool hdr_;
    //: One line per producer naming the colourspace/EOTF that actually arrived. Without it
    //: "the tag the consumer signalled is the tag the producer received" is not assertable:
    //: both readers consume the metadata and neither records what it saw.
    bool logged_input_metadata_ = false;
    //: How many frames to watch before reporting an absence of ancillary data. 150 is about
    //: three seconds at 50p -- long enough that "nothing in 150 frames" is a statement about
    //: the source rather than about which frame happened to be first.
    static constexpr int ANC_SEARCH_FRAMES = 150;
    int                  frames_seen_      = 0;

    int sync_group_ = 0;
    int sync_peers_ = 1;
    friend class DeckLinkSyncManager;

    core::draw_frame last_frame_;

    // FPS counter
    std::chrono::steady_clock::time_point last_fps_update_;
    int                     frames_since_update_ = 0;
    double                  current_fps_ = 0.0;

    int                                                        buffer_capacity_ = 4;
    std::deque<std::pair<core::draw_frame, core::video_field>> buffer_;
    mutable std::mutex                                         buffer_mutex_;

    std::exception_ptr exception_;
    std::mutex         exception_mutex_;

    com_ptr<IDeckLinkDisplayMode> mode_;

    core::video_format_desc input_format;

    std::string vfilter_;
    std::string afilter_;

    Filter video_filter_;
    Filter audio_filter_;

    Decoder video_decoder_;

  public:
    decklink_producer(core::video_format_desc                     format_desc,
                      int                                         device_index,
                      const spl::shared_ptr<core::frame_factory>& frame_factory,
                      const core::video_format_repository&        format_repository,
                      std::string                                 vfilter,
                      std::string                                 afilter,
                      const std::wstring&                         format,
                      bool                                        freeze_on_lost,
                      bool                                        hdr,
                      int                                         sync_group,
                      int                                         sync_peers)
        : device_index_(device_index)
        , format_desc_(std::move(format_desc))
        , frame_factory_(frame_factory)
        , format_repository_(format_repository)
        , freeze_on_lost_(freeze_on_lost)
        , hdr_(hdr)
        , input_format(format_desc_)
        , vfilter_(std::move(vfilter))
        , afilter_(std::move(afilter))
        , sync_group_(sync_group)
        , sync_peers_(sync_peers)
    {
        // use user-provided format if available, or choose the channel's output format
        if (!format.empty()) {
            input_format = format_repository.find(format);
        }

        mode_ = get_display_mode(input_, input_format.format, get_pixel_format2(hdr_), bmdSupportedVideoModeDefault);
        video_filter_  = Filter(vfilter_, AVMEDIA_TYPE_VIDEO, format_desc_, mode_, hdr_);
        audio_filter_  = Filter(afilter_, AVMEDIA_TYPE_AUDIO, format_desc_, mode_, hdr_);
        video_decoder_ = Decoder(hdr_, mode_);

        boost::range::rotate(audio_cadence_, std::end(audio_cadence_) - 1);

        graph_->set_color("tick-time", diagnostics::color(0.0f, 0.6f, 0.9f));
        graph_->set_color("late-frame", diagnostics::color(0.6f, 0.3f, 0.3f));
        graph_->set_color("frame-time", diagnostics::color(1.0f, 0.0f, 0.0f));
        graph_->set_color("dropped-frame", diagnostics::color(0.3f, 0.6f, 0.3f));
        graph_->set_color("output-buffer", diagnostics::color(0.0f, 1.0f, 0.0f));
        graph_->set_color("in-sync", diagnostics::color(1.0f, 0.2f, 0.0f));
        graph_->set_color("out-sync", diagnostics::color(0.0f, 0.2f, 1.0f));
        graph_->set_text(print());
        diagnostics::register_graph(graph_);

        BOOL status = FALSE;
        int  flags  = bmdVideoInputEnableFormatDetection;

        if (!format.empty()) {
            flags = 0;
        } else if (FAILED(attributes_->GetFlag(BMDDeckLinkSupportsInputFormatDetection, &status)) || !status) {
            CASPAR_LOG(warning) << L"Decklink producer does not support auto detect input, you can explicitly choose a "
                                   L"format by appending FORMAT";
            flags = 0;
        }

        if (sync_group_ > 0) {
            flags |= bmdVideoInputSynchronizeToCaptureGroup;
            CASPAR_LOG(info) << print() << " Joining Sync Group " << sync_group_ << " (Expect " << sync_peers_
                             << " peers)";

            {
                com_ptr<IDeckLinkConfiguration> config;
                if (SUCCEEDED(decklink_->QueryInterface(IID_IDeckLinkConfiguration, (void**)&config))) {
                    config->SetInt(bmdDeckLinkConfigCaptureGroup, (int64_t)sync_group_);
                }
            }
        }

        shared_input_->enable_video_input(mode_->GetDisplayMode(), get_pixel_format2(hdr_), flags);

        // If a prior producer already changed the format via auto-detection, the shared input
        // is already at a different mode. Sync this producer's mode and filters accordingly
        // so the dimension-guard doesn't discard all incoming frames.
        {
            auto actual_mode_bmd = shared_input_->get_current_display_mode();
            if (actual_mode_bmd != mode_->GetDisplayMode()) {
                mode_          = get_display_mode(input_, actual_mode_bmd, get_pixel_format2(hdr_), bmdSupportedVideoModeDefault);
                video_filter_  = Filter(vfilter_, AVMEDIA_TYPE_VIDEO, format_desc_, mode_, hdr_);
                audio_filter_  = Filter(afilter_, AVMEDIA_TYPE_AUDIO, format_desc_, mode_, hdr_);
                video_decoder_ = Decoder(hdr_, mode_);
                auto actual_fmt = get_caspar_video_format(actual_mode_bmd);
                if (actual_fmt != core::video_format::invalid)
                    input_format = format_repository_.find_format(actual_fmt);
                in_sync_  = 0.0;
                out_sync_ = 0.0;
            }
        }

        shared_input_->enable_audio_input(bmdAudioSampleRate48kHz,
                                          bmdAudioSampleType32bitInteger,
                                          static_cast<int>(format_desc_.audio_channels));

        shared_input_->add_listener(this);

        if (sync_group_ > 0) {
            // Register with Sync Manager and wait for it to start us
            DeckLinkSyncManager::instance().register_producer(sync_group_, sync_peers_, this);
        } else {
            shared_input_->start_streams();
        }

        CASPAR_LOG(info) << print() << L" Initialized";
    }

    // Called by SyncManager
    bool check_signal_locked() const
    {
        int64_t                  locked = 0;
        com_ptr<IDeckLinkStatus> status;
        if (SUCCEEDED(decklink_->QueryInterface(IID_IDeckLinkStatus, (void**)&status))) {
            if (SUCCEEDED(status->GetInt(bmdDeckLinkStatusVideoInputSignalLocked, &locked))) {
                return locked != 0;
            }
        }
        return false;
    }

    void start_streams()
    {
        try {
            shared_input_->start_streams();
            CASPAR_LOG(info) << print() << " Started synchronized stream";
        } catch (...) {
            CASPAR_LOG(error) << print() << " Failed to start synchronized stream";
        }
    }

    ~decklink_producer()
    {
        shared_input_->remove_listener(this);

        if (sync_group_ > 0) {
            DeckLinkSyncManager::instance().unregister_producer(sync_group_, this);
        }

        shared_input_->stop_streams();
        shared_input_->disable_video_input();
    }

    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID, LPVOID*) override { return E_NOINTERFACE; }
    ULONG STDMETHODCALLTYPE   AddRef() override { return 1; }
    ULONG STDMETHODCALLTYPE   Release() override { return 1; }

    HRESULT STDMETHODCALLTYPE VideoInputFormatChanged(BMDVideoInputFormatChangedEvents notificationEvents,
                                                      IDeckLinkDisplayMode*            newDisplayMode,
                                                      BMDDetectedVideoInputFormatFlags /*detectedSignalFlags*/) override
    {
        try {
            auto newMode = newDisplayMode->GetDisplayMode();
            auto fmt     = get_caspar_video_format(newMode);

            if (fmt == input_format.format) {
                // This gets called often if the enabled pixel format doesn't match the signal
                // https://forum.blackmagicdesign.com/viewtopic.php?f=12&t=144234 So if the video format hasn't actually
                // changed, then we can ignore this event. In the future we may wish to respect this in order to unpack
                // the pixels ourselves
                return S_OK;
            }

            auto new_fmt = format_repository_.find_format(fmt);

            CASPAR_LOG(info) << print() << L" Input format changed from " << input_format.name << L" to "
                             << new_fmt.name;

            // Only update filter/decoder state here. Stream restart (EnableVideoInput,
            // FlushStreams, StartStreams) is handled by SharedDeckLinkInput::VideoInputFormatChanged,
            // which is the actual registered DeckLink callback and the only valid call site.

            // reinitializing filters because not all filters can handle on-the-fly format changes
            input_format = new_fmt;
            mode_        = get_display_mode(input_, newMode, get_pixel_format2(hdr_), bmdSupportedVideoModeDefault);

            graph_->set_text(print());

            video_filter_  = Filter(vfilter_, AVMEDIA_TYPE_VIDEO, format_desc_, mode_, hdr_);
            audio_filter_  = Filter(afilter_, AVMEDIA_TYPE_AUDIO, format_desc_, mode_, hdr_);
            video_decoder_ = Decoder(hdr_, mode_);

            // Reset sync state so the sync-drift check doesn't immediately fire with stale timestamps,
            // and clear the buffer of any frames from the old format.
            in_sync_  = 0.0;
            out_sync_ = 0.0;
            {
                std::lock_guard<std::mutex> lock(buffer_mutex_);
                buffer_.clear();
            }
            return S_OK;
        } catch (...) {
            std::lock_guard<std::mutex> lock(exception_mutex_);
            exception_ = std::current_exception();
            return E_FAIL;
        }
    }

    HRESULT STDMETHODCALLTYPE VideoInputFrameArrived(IDeckLinkVideoInputFrame*  video,
                                                     IDeckLinkAudioInputPacket* audio) override
    {
        caspar::timer frame_timer;

        CASPAR_SCOPE_EXIT
        {
            size_t buffer_size = 0;
            {
                std::lock_guard<std::mutex> lock(buffer_mutex_);
                buffer_size = buffer_.size();
            }

            {
                std::lock_guard<std::mutex> lock(state_mutex_);
                state_["file/name"]              = model_name_;
                state_["file/path"]              = device_index_;
                state_["file/format"]            = format_desc_.name;
                state_["file/audio/sample-rate"] = format_desc_.audio_sample_rate;
                state_["file/audio/channels"]    = format_desc_.audio_channels;
                state_["file/fps"]               = format_desc_.fps;
                state_["profiler/time"]          = {frame_timer.elapsed(), format_desc_.fps};
                state_["buffer"]                 = {static_cast<int>(buffer_size), buffer_capacity_};
                state_["has_signal"]             = has_signal_;

                if (video) {
                    state_["file/video/width"]  = static_cast<int>(video->GetWidth());
                    state_["file/video/height"] = static_cast<int>(video->GetHeight());
                }
            }

            graph_->set_value("frame-time", frame_timer.elapsed() * format_desc_.fps * 0.5);
            graph_->set_value("output-buffer", static_cast<float>(buffer_size) / static_cast<float>(buffer_capacity_));
        };

        try {
            graph_->set_value("tick-time", tick_timer_.elapsed() * format_desc_.hz * 0.5);
            tick_timer_.restart();

            BMDTimeValue         in_video_pts   = 0LL;
            BMDTimeValue         in_audio_pts   = 0LL;
            // `10BIT` asserts BT.2020/HLG. It is a LAST RESORT and no longer the usual case.
            //
            // The rationale that used to sit here said SDI inputs do not carry colour metadata
            // because that interface "only works for HDMI". That is false on this hardware and
            // was measured false on 2026-08-16: the card writes an ST 352 payload identifier
            // from the consumer's frame metadata, the receiving driver parses it, and the
            // producer reads bt709/sdr, bt2020/hlg and bt2020/pq back exactly as configured.
            // The reason it looked otherwise for so long is that the diagnostic sampled one
            // frame -- see the census in `read_sdi_signalling`.
            //
            // So the ordering in the frame handler puts the card's own answer first and this
            // assertion third, behind the wire. What remains true is that `10BIT` is a
            // BIT-DEPTH request being read as a colour declaration, which is a guess about
            // content the operator may not have seen. It is kept because removing it would
            // change behaviour for anyone relying on it, and because a genuinely untagged
            // 10-bit feed has to be called something -- but when it is what decides the
            // colour, that is now logged rather than silent.
            core::color_space    color_space    = hdr_ ? core::color_space::bt2020 : core::color_space::unknown;
            core::color_transfer color_transfer = hdr_ ? core::color_transfer::hlg  : core::color_transfer::sdr;

            if (video) {
                const auto flags = video->GetFlags();
                has_signal_      = !(flags & bmdFrameHasNoInputSource);
                if (freeze_on_lost_ && !has_signal_) {
                    frame_count_ = 0;
                    return S_OK;
                }

                // Discard transition frames from the previous format.
                if (static_cast<long>(video->GetWidth())  != mode_->GetWidth() ||
                    static_cast<long>(video->GetHeight()) != mode_->GetHeight()) {
                    CASPAR_LOG(warning) << print() << L" Discarding frame " << video->GetWidth() << L"x" << video->GetHeight()
                                       << L" expected " << mode_->GetWidth() << L"x" << mode_->GetHeight();
                    return S_OK;
                }

                // The frame-metadata interface first, since a card that populates it has
                // already done this work; the ancillary data underneath it when it does not,
                // which on SDI is always. Order matters only in that the two must not
                // contradict each other silently -- see the log below, which reports both.
                std::wstring anc_census;
                const auto   anc = read_sdi_signalling(video, logged_input_metadata_ ? nullptr : &anc_census);

                // SAMPLE ACROSS FRAMES, NOT ONE. Ancillary is not present on every frame: a
                // minimal SDK-based capturer watching a source that emits one CDP per frame
                // saw the ancillary interface on 52 frames out of 99. Reporting the FIRST
                // frame therefore says "no ancillary on this input" about half the time on a
                // source that plainly has some -- which is exactly the false negative this
                // log spent a whole session producing. Keep looking until something turns up,
                // and only give up after enough frames that absence means something.

                // PRECEDENCE, most authoritative first:
                //
                //   1. the card's own frame metadata -- a parsed HDMI InfoFrame, and the only
                //      source here that has already been decoded for us
                //   2. ST 352 on the wire -- what the sender actually declared
                //   3. the 10BIT configuration assertion -- an operator's claim about a feed,
                //      made without seeing it
                //   4. the mixer's raster convention, for a signal that declares nothing
                //
                // Asking with an `unknown`/`sdr` fallback is what keeps 1 and 3 apart: called
                // with the configured value as the fallback, a card that said nothing and a
                // card that agreed with the configuration return the same answer, and the
                // wire could then never outrank a flag typed at the command line.
                const auto card_space    = get_color_space(video, core::color_space::unknown);
                const auto card_transfer = get_color_transfer(video, core::color_transfer::sdr);

                if (card_space != core::color_space::unknown) {
                    color_space = card_space;
                } else if (anc.colorimetry_specified) {
                    color_space = anc.color_space;
                }

                // No `unknown` exists in `color_transfer`, so "the card declared SDR" and "the
                // card declared nothing" are the same value and cannot be told apart here.
                // The wire therefore wins outright when it specifies: a VPID that says PQ is a
                // statement, and an absent InfoFrame is not.
                if (anc.transfer_specified) {
                    color_transfer = anc.color_transfer;
                } else if (card_transfer != core::color_transfer::sdr) {
                    color_transfer = card_transfer;
                }

                // SAY WHAT ARRIVED, ONCE. The consumer signals colourspace and EOTF, and
                // these readers consume them, but nothing logged or exposed the result -- so
                // "the tag the consumer sent is the tag the producer received" was not an
                // assertable statement, on any rig, and a loopback could only infer it from
                // pixels. That inference is confounded: the channel's <color-space> drives
                // the consumer's ENCODE matrix as well as the tag it signals, so a picture
                // difference cannot separate "the tag was honoured" from "the encode
                // changed".
                //
                // The census is here for the case that actually occurred: everything reading
                // `unknown`. Without it, "no ancillary data reached us" and "ancillary data
                // reached us and carried no VPID" look identical and have different fixes.
                ++frames_seen_;
                const bool worth_reporting = anc.anc_interface_available || frames_seen_ >= ANC_SEARCH_FRAMES;
                if (!logged_input_metadata_ && worth_reporting) {
                    logged_input_metadata_ = true;

                    // WHAT THE CARD SAYS ABOUT ITSELF, once. The frame refuses every
                    // ancillary accessor while answering other optional interfaces, so the
                    // question moved from "are we asking correctly" to "does this device do
                    // ancillary capture at all, in this profile". These four are what the SDK
                    // points at for that: the 10-bit requirement (satisfied here -- the frame
                    // is v210), HANC input support as the nearest published proxy for whether
                    // the model does ancillary capture, and the profile and duplex mode,
                    // because an 8K Pro split into four sub-devices is a different device from
                    // the same card in one-device mode as far as capabilities go.
                    BOOL     needs_10bit = FALSE, hanc_in = FALSE;
                    LONGLONG profile_id = 0, duplex = 0;
                    attributes_->GetFlag(BMDDeckLinkVANCRequires10BitYUVVideoFrames, &needs_10bit);
                    // 'dshi' -- BMDDeckLinkSupportsHANCInput, which postdates the 12.3.1
                    // header this module is generated from, hence the literal.
                    attributes_->GetFlag(static_cast<BMDDeckLinkAttributeID>(0x64736869), &hanc_in);
                    attributes_->GetInt(BMDDeckLinkProfileID, &profile_id);
                    attributes_->GetInt(BMDDeckLinkDuplex, &duplex);
                    CASPAR_LOG(info) << print() << L" input device: vanc-requires-10bit="
                                     << (needs_10bit ? L"true" : L"false") << L" supports-hanc-input="
                                     << (hanc_in ? L"true" : L"false") << L" profile=0x" << std::hex << profile_id
                                     << L" duplex=0x" << duplex << std::dec;
                    // THE CARD'S RAW ANSWER, separate from the resolved one. The resolved
                    // value cannot answer "did the wire tell us this": `sdr` doubles as "no
                    // information", and under `10BIT` the fallback is bt2020/hlg, so a
                    // resolved `hlg` may be the card speaking or may be the flag. Printing
                    // what GetInt actually returned makes VPID's arrival observable even
                    // though the driver consumes the packet rather than listing it.
                    // SAY WHEN THE FLAG DECIDED IT. With the card supplying real values this
                    // should almost never fire; if it does, the feed is untagged and the
                    // colour is a configuration guess rather than a reading, which is exactly
                    // the case an operator needs to know about and the one that used to be
                    // indistinguishable from a correct reading.
                    if (hdr_ && card_space == core::color_space::unknown && !anc.colorimetry_specified) {
                        CASPAR_LOG(warning)
                            << print()
                            << L" the 10BIT flag is deciding the colour: this feed declares no colour space, so it is"
                               L" being treated as BT.2020/HLG by configuration rather than by anything on the wire";
                    }

                    CASPAR_LOG(info) << print() << L" card metadata: colorspace="
                                     << metadata_name(card_space) << L" transfer="
                                     << metadata_name(card_transfer)
                                     << L" (card_space_known=" << (card_space != core::color_space::unknown)
                                     << L")";
                    CASPAR_LOG(info) << print() << L" input metadata: colorspace="
                                     << metadata_name(color_space)
                                     << L" transfer=" << metadata_name(color_transfer)
                                     << L" | anc packets:" << (anc_census.empty() ? L" (none)" : anc_census);
                    if (anc.vpid_present) {
                        wchar_t vpid[64];
                        swprintf(vpid, 64, L"%02X %02X %02X %02X", anc.vpid[0], anc.vpid[1], anc.vpid[2], anc.vpid[3]);
                        CASPAR_LOG(info) << print() << L" ST 352 VPID: " << vpid;
                    }
                    if (anc.mastering_display_present) {
                        CASPAR_LOG(info) << print() << L" ST 2108-1 mastering display: max="
                                         << anc.max_display_mastering_luminance << L" cd/m2 min="
                                         << anc.min_display_mastering_luminance << L" cd/m2";
                    }
                    if (anc.content_light_level_present) {
                        CASPAR_LOG(info) << print() << L" ST 2108-1 content light level: MaxCLL="
                                         << anc.max_content_light_level << L" MaxFALL="
                                         << anc.max_frame_average_light_level;
                    }
                }
                auto src    = video_decoder_.decode(video, mode_);

                BMDTimeValue duration;
                if (SUCCEEDED(video->GetStreamTime(&in_video_pts, &duration, AV_TIME_BASE))) {
                    src->pts = in_video_pts;
                }

                if (src) {
                    if (video_filter_.video_source) {
                        FF(av_buffersrc_write_frame(video_filter_.video_source, src.get()));
                    }
                    if (audio_filter_.video_source) {
                        FF(av_buffersrc_write_frame(audio_filter_.video_source, src.get()));
                    }
                }
            }

            if (audio) {
                auto src    = std::shared_ptr<AVFrame>(av_frame_alloc(), [](AVFrame* ptr) { av_frame_free(&ptr); });
                src->format = AV_SAMPLE_FMT_S32;
                av_channel_layout_default(&src->ch_layout, format_desc_.audio_channels);
                src->sample_rate = format_desc_.audio_sample_rate;

                void* audio_bytes = nullptr;
                if (SUCCEEDED(audio->GetBytes(&audio_bytes)) && audio_bytes) {
                    audio->AddRef();
                    src = std::shared_ptr<AVFrame>(src.get(), [src, audio](AVFrame* ptr) { audio->Release(); });
                    src->nb_samples  = audio->GetSampleFrameCount();
                    src->data[0]     = reinterpret_cast<uint8_t*>(audio_bytes);
                    src->linesize[0] = src->nb_samples * format_desc_.audio_channels *
                                       av_get_bytes_per_sample(static_cast<AVSampleFormat>(src->format));

                    if (SUCCEEDED(audio->GetPacketTime(&in_audio_pts, format_desc_.audio_sample_rate))) {
                        src->pts = in_audio_pts;
                    }

                    if (video_filter_.audio_source) {
                        FF(av_buffersrc_write_frame(video_filter_.audio_source, src.get()));
                    }
                    if (audio_filter_.audio_source) {
                        FF(av_buffersrc_write_frame(audio_filter_.audio_source, src.get()));
                    }
                }
            }

            av_buffersink_set_frame_size(audio_filter_.sink, audio_cadence_[0]);
            while (true) {
                {
                    auto av_video = alloc_frame();
                    auto av_audio = alloc_frame();

                    // TODO (fix) this may get stuck if the decklink sends a frame of video or audio

                    int vret = av_buffersink_get_frame_flags(video_filter_.sink, av_video.get(), AV_BUFFERSINK_FLAG_PEEK);
                    if (vret < 0) {
                        if (vret != AVERROR(EAGAIN) && vret != AVERROR_EOF)
                            CASPAR_LOG(warning) << print() << L" video filter sink error: " << vret;
                        return S_OK;
                    }

                    int aret = av_buffersink_get_frame_flags(audio_filter_.sink, av_audio.get(), AV_BUFFERSINK_FLAG_PEEK);
                    if (aret < 0) {
                        if (aret != AVERROR(EAGAIN) && aret != AVERROR_EOF)
                            CASPAR_LOG(warning) << print() << L" audio filter sink error: " << aret;
                        return S_OK;
                    }
                }
                auto av_video = alloc_frame();
                auto av_audio = alloc_frame();

                // TODO (fix) auto V/A sync even if decklink is wrong.

                av_buffersink_get_frame(video_filter_.sink, av_video.get());
                av_buffersink_get_samples(audio_filter_.sink, av_audio.get(), audio_cadence_[0]);

                auto video_tb = av_buffersink_get_time_base(video_filter_.sink);
                auto audio_tb = av_buffersink_get_time_base(audio_filter_.sink);

                // CASPAR_LOG(trace) << "decklink a/v pts:" << av_video->pts << " " << av_audio->pts;

                auto in_sync = static_cast<double>(in_video_pts) / AV_TIME_BASE -
                               static_cast<double>(in_audio_pts) / format_desc_.audio_sample_rate;
                auto out_sync = static_cast<double>(av_video->pts * video_tb.num) / video_tb.den -
                                static_cast<double>(av_audio->pts * audio_tb.num) / audio_tb.den;

                if (std::abs(in_sync - in_sync_) > 0.01) {
                    CASPAR_LOG(warning) << print() << " in-sync changed: " << in_sync;
                }
                in_sync_ = in_sync;

                if (std::abs(out_sync - out_sync_) > 0.01) {
                    CASPAR_LOG(warning) << print() << " out-sync changed: " << out_sync;
                }
                out_sync_ = out_sync;

                // If filter output sync has drifted too far from input sync, recreate the filters to resync.
                // This usually happens after signal loss/regain from receiving incomplete frames.
                const double frame_duration_threshold = 1.5 / input_format.hz;
                const double sync_drift               = std::abs(out_sync - in_sync);
                if (sync_drift > frame_duration_threshold) {
                    CASPAR_LOG(warning) << print() << " Excessive A/V sync drift detected ("
                                        << static_cast<int>(sync_drift * 1000) << "ms), recreating filters to resync";

                    // Recreate filters to clear all buffered data
                    video_filter_ = Filter(vfilter_, AVMEDIA_TYPE_VIDEO, format_desc_, mode_, hdr_);
                    audio_filter_ = Filter(afilter_, AVMEDIA_TYPE_AUDIO, format_desc_, mode_, hdr_);

                    in_sync_  = 0.0;
                    out_sync_ = 0.0;

                    // Skip this iteration and start fresh
                    return S_OK;
                }

                graph_->set_value("in-sync", in_sync * 2.0 + 0.5);
                graph_->set_value("out-sync", out_sync * 2.0 + 0.5);

                auto frame = core::draw_frame(make_frame(this,
                                                        *frame_factory_,
                                                        av_video,
                                                        av_audio,
                                                        color_space,
                                                        core::frame_geometry::scale_mode::stretch,
                                                        false,
                                                        color_transfer));
                auto field = core::video_field::progressive;
                if (format_desc_.field_count == 2) {
                    field = frame_count_ % 2 == 0 ? core::video_field::a : core::video_field::b;
                }

                {
                    std::lock_guard<std::mutex> lock(buffer_mutex_);

                    buffer_.emplace_back(std::make_pair(frame, field));
                    frame_count_++;

                    if (buffer_.size() > buffer_capacity_) {
                        buffer_.pop_front();
                        // If interlaced, pop a second frame, to drop a whole source frame.
                        if (format_desc_.field_count == 2)
                            buffer_.pop_front();
                        graph_->set_tag(diagnostics::tag_severity::WARNING, "dropped-frame");
                    }
                }

                boost::range::rotate(audio_cadence_, std::end(audio_cadence_) - 1);
            }
        } catch (...) {
            std::lock_guard<std::mutex> lock(exception_mutex_);
            exception_ = std::current_exception();
            return E_FAIL;
        }

        return S_OK;
    }

    core::draw_frame get_frame(const core::video_field field, bool use_last_frame)
    {
        // FPS Calc
        auto now = std::chrono::steady_clock::now();
        frames_since_update_++;
        auto duration_sec = std::chrono::duration_cast<std::chrono::duration<double>>(now - last_fps_update_).count();
        
        if (duration_sec >= 1.0) {
            current_fps_ = (double)frames_since_update_ / duration_sec;
            frames_since_update_ = 0;
            last_fps_update_ = now;
            
            std::wstringstream stats;
            stats.precision(2);
            stats << std::fixed;
            stats << print() << L" Fps: " << current_fps_;
            graph_->set_text(stats.str());
        }

        {
            std::lock_guard<std::mutex> lock(exception_mutex_);
            if (exception_ != nullptr) {
                std::rethrow_exception(exception_);
            }
        }

        core::draw_frame frame;
        bool             wrong_field = false;
        {
            std::lock_guard<std::mutex> lock(buffer_mutex_);
            if (!buffer_.empty()) {
                auto& candidate = buffer_.front();
                if (candidate.second == field || candidate.second == core::video_field::progressive) {
                    frame = std::move(candidate.first);
                    buffer_.pop_front();
                } else {
                    wrong_field = true;
                }
            } else {
                graph_->set_tag(diagnostics::tag_severity::WARNING, "late-frame");
            }

            graph_->set_value("output-buffer",
                              static_cast<float>(buffer_.size()) / static_cast<float>(buffer_capacity_));
        }

        if (wrong_field) {
            return last_frame_;
        } else if (!frame && (freeze_on_lost_ || use_last_frame)) {
            return last_frame_;
        } else {
            if (frame) {
                last_frame_ = frame;
            }
            return frame;
        }
    }

    bool is_ready()
    {
        std::lock_guard<std::mutex> lock(buffer_mutex_);
        return !buffer_.empty() || last_frame_;
    }

    std::wstring print() const
    {
        return model_name_ + L" [" + std::to_wstring(device_index_) + L"|" + input_format.name + L"]";
    }

    core::monitor::state state() const
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        return state_;
    }
};

class decklink_producer_proxy : public core::frame_producer
{
    std::unique_ptr<decklink_producer> producer_;
    const uint32_t                     length_;
    executor                           executor_;

  public:
    explicit decklink_producer_proxy(const core::video_format_desc&              format_desc,
                                     const spl::shared_ptr<core::frame_factory>& frame_factory,
                                     const core::video_format_repository&        format_repository,
                                     int                                         device_index,
                                     const std::string&                          vfilter,
                                     const std::string&                          afilter,
                                     uint32_t                                    length,
                                     const std::wstring&                         format,
                                     bool                                        freeze_on_lost,
                                     bool                                        hdr,
                                     int                                         sync_group = 0,
                                     int                                         sync_peers = 1)
        : length_(length)
        , executor_(L"decklink_producer[" + std::to_wstring(device_index) + L"]")
    {
        auto ctx = core::diagnostics::call_context::for_thread();
        executor_.invoke([=] {
            core::diagnostics::call_context::for_thread() = ctx;
            com_initialize();
            producer_.reset(new decklink_producer(format_desc,
                                                  device_index,
                                                  frame_factory,
                                                  format_repository,
                                                  vfilter,
                                                  afilter,
                                                  format,
                                                  freeze_on_lost,
                                                  hdr,
                                                  sync_group,
                                                  sync_peers));
        });
    }

    ~decklink_producer_proxy() override
    {
        executor_.invoke([=] {
            producer_.reset();
            com_uninitialize();
        });
    }

    core::monitor::state state() const override { return producer_->state(); }

    // frame_producer

    core::draw_frame receive_impl(const core::video_field field, int nb_samples) override
    {
        return producer_->get_frame(field, false);
    }

    core::draw_frame first_frame(const core::video_field field) override { return receive_impl(field, 0); }

    core::draw_frame last_frame(const core::video_field field) override
    {
        return core::draw_frame::still(producer_->get_frame(field, true));
    }

    bool is_ready() override { return producer_->is_ready(); }

    uint32_t nb_frames() const override { return length_; }

    std::wstring print() const override { return producer_->print(); }

    std::wstring name() const override { return L"decklink"; }
};

spl::shared_ptr<core::frame_producer> create_producer(const core::frame_producer_dependencies& dependencies,
                                                      const std::vector<std::wstring>&         params)
{
    if (params.empty() || !boost::iequals(params.at(0), "decklink"))
        return core::frame_producer::empty();

    auto device_index = get_param(L"DEVICE", params, -1);
    if (device_index == -1)
        device_index = std::stoi(params.at(1));

    auto freeze_on_lost = contains_param(L"FREEZE_ON_LOST", params);

    auto hdr = contains_param(L"10BIT", params);

    auto format_str = get_param(L"FORMAT", params);

    auto filter_str = get_param(L"FILTER", params);
    auto length     = get_param(L"LENGTH", params, std::numeric_limits<uint32_t>::max());

    boost::ireplace_all(filter_str, L"DEINTERLACE_BOB", L"YADIF=1:-1");
    boost::ireplace_all(filter_str, L"DEINTERLACE_LQ", L"SEPARATEFIELDS");
    boost::ireplace_all(filter_str, L"DEINTERLACE", L"YADIF=0:-1");

    auto vfilter = boost::to_lower_copy(get_param(L"VF", params, filter_str));
    auto afilter = boost::to_lower_copy(get_param(L"AF", params, get_param(L"FILTER", params, L"")));

    auto sync_group = get_param(L"SYNC_GROUP", params, 0);
    auto sync_peers = get_param(L"SYNC_PEERS", params, 1);

    if (sync_group == 0 && device_index > 0) {
        try {
            auto& pt = env::properties();
            // CasparCG decklink sync config structure is configuration.decklink-sync.device-1...
            std::wstring base_key = L"configuration.decklink-sync.device-" + std::to_wstring(device_index);
            
            // Check if key exists using get_optional, as get throws if key not found and default value overload might not work for path
            // The get with default value works for paths
            sync_group = pt.get<int>(base_key + L".group", 0);
            
            if (sync_group != 0) {
                sync_peers = pt.get<int>(base_key + L".peers", 1);
            }
        } catch (...) {
            // Ignore config error
        }
    }

    return spl::make_shared<decklink_producer_proxy>(dependencies.format_desc,
                                                     dependencies.frame_factory,
                                                     dependencies.format_repository,
                                                     device_index,
                                                     u8(vfilter),
                                                     u8(afilter),
                                                     length,
                                                     format_str,
                                                     freeze_on_lost,
                                                     hdr,
                                                     sync_group,
                                                     sync_peers);
}

void DeckLinkSyncManager::register_producer(int group_id, int peers, decklink_producer* producer)
{
    std::lock_guard<std::mutex> lock(manager_mutex_);
    if (groups_.find(group_id) == groups_.end()) {
        groups_[group_id]                 = std::make_shared<Group>();
        groups_[group_id]->expected_peers = peers;
    }
    auto group = groups_[group_id];

    std::lock_guard<std::mutex> g_lock(group->mutex);
    group->producers.push_back(producer);

    // If we have enough producers registered, make sure monitor thread is running
    if (group->producers.size() >= group->expected_peers && !monitoring_) {
        monitoring_     = true;
        monitor_thread_ = std::thread(&DeckLinkSyncManager::start_monitor_loop, this);
    }
    // Wake monitor to check the new registration
    monitor_cv_.notify_all();
}

void DeckLinkSyncManager::unregister_producer(int group_id, decklink_producer* producer)
{
    std::lock_guard<std::mutex> lock(manager_mutex_);
    auto                        it = groups_.find(group_id);
    if (it != groups_.end()) {
        bool should_erase = false;
        {
            std::lock_guard<std::mutex> g_lock(it->second->mutex);
            auto&                       vec = it->second->producers;
            vec.erase(std::remove(vec.begin(), vec.end(), producer), vec.end());
            should_erase = vec.empty();
        }
        // Erase after releasing group lock
        if (should_erase) {
            groups_.erase(it);
        }
    }
}

void DeckLinkSyncManager::start_monitor_loop()
{
    while (monitoring_) {
        {
            std::unique_lock<std::mutex> lock(manager_mutex_);
            monitor_cv_.wait_for(lock, std::chrono::milliseconds(50), [this] { return !monitoring_.load(); });
        }

        if (!monitoring_)
            break;

        std::lock_guard<std::mutex> lock(manager_mutex_);
        if (groups_.empty())
            continue;

        for (auto& [id, group] : groups_) {
            std::lock_guard<std::mutex> g_lock(group->mutex);
            if (group->started)
                continue;

            // Check if we have enough peers registered
            if (group->producers.size() < group->expected_peers)
                continue;

            bool all_locked = true;
            for (auto* p : group->producers) {
                if (!p->check_signal_locked()) {
                    all_locked = false;
                    break;
                }
            }

            if (all_locked && !group->producers.empty()) {
                CASPAR_LOG(info) << "Sync Group " << id << ": All signals locked. Starting synchronized capture.";
                // Start ALL producers in the sync group so their SharedDeckLinkInput
                // state (started_, start_ref_count_) is correct for format changes and cleanup.
                for (size_t i = 0; i < group->producers.size(); ++i) {
                    group->producers[i]->start_streams();
                    CASPAR_LOG(info) << "Sync Group " << id << ": Peer " << i << " started.";
                }
                group->started = true;
            }
        }
    }
}

}} // namespace caspar::decklink
