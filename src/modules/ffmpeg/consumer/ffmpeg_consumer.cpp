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

#include "ffmpeg_consumer.h"

#include "../util/av_assert.h"
#include "../util/av_util.h"
#include "cuda_gl_upload.h"
#include "cuda_vk_upload.h"

#include <accelerator/vulkan/util/av_vulkan_export.h>
#include "../util/vulkan_hwdevice.h"

#include <accelerator/ogl/util/device.h>
#include <accelerator/ogl/util/texture.h>

#include <chrono>
#include <sstream>
#include <iomanip>

extern "C" {
#include <libavutil/opt.h>
}

// Include LTC Module
#include "../../ltc/ltc_input.h"

// AV_CODEC_ID_TIMECODE was removed in FFmpeg 7; define a local sentinel for tmcd-stream detection.
// The actual stream codec_id is set to AV_CODEC_ID_NONE — FFmpeg 7 identifies tmcd tracks by codec_tag.
#ifndef AV_CODEC_ID_TIMECODE
constexpr AVCodecID AV_CODEC_ID_TIMECODE = static_cast<AVCodecID>(0x17800);
#endif

#include <common/bit_depth.h>
#include <common/diagnostics/graph.h>
#include <common/env.h>
#include <common/executor.h>
#include <common/future.h>
#include <common/log.h>
#include <common/memory.h>
#include <common/scope_exit.h>
#include <common/timer.h>

#include <core/consumer/channel_info.h>
#include <core/frame/frame.h>
#include <core/video_format.h>

#if defined(__GNUC__) && __GNUC__ == 14
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstringop-overflow"
#endif
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/regex.hpp>
#if defined(__GNUC__) && __GNUC__ == 14
#pragma GCC diagnostic pop
#endif

#include <boost/crc.hpp>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavfilter/avfilter.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
#include <libavformat/avformat.h>
#include <libavutil/channel_layout.h>
#include <libavutil/csp.h>
#include <libavutil/mastering_display_metadata.h>
#include <libavutil/hwcontext.h>
// This header #includes <cuda.h>, so it needs the CUDA toolkit's include path. The ffmpeg
// module only adds that (and only defines CASPAR_FFMPEG_HAS_CUDA) when CUDAToolkit is
// found, which today happens only inside its `if (MSVC)` block -- so on Linux the include
// failed as "cuda.h: No such file or directory" from inside libavutil.
#ifdef CASPAR_FFMPEG_HAS_CUDA
#include <libavutil/hwcontext_cuda.h>
#endif

#if LIBAVUTIL_VERSION_MAJOR >= 60

#include <libavutil/hwcontext_vulkan.h>

#endif
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>
#include <libavutil/pixfmt.h>
#include <libavutil/samplefmt.h>
}

#include <tbb/concurrent_queue.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_invoke.h>
#include <common/os/thread.h>

#include <memory>
#include <optional>
#include <thread>
#include <vector>

namespace caspar { namespace ffmpeg {

// TODO multiple output streams
// TODO multiple output files
// TODO run video filter, video encoder, audio filter, audio encoder in separate threads.
// TODO realtime with smaller buffer?

struct Stream
{
    std::shared_ptr<AVFilterGraph> graph  = nullptr;
    AVFilterContext*               sink   = nullptr;
    AVFilterContext*               source = nullptr;

    std::shared_ptr<AVCodecContext> enc = nullptr;
    AVStream*                       st  = nullptr;
    bool                            is_ltc = false;
    uint32_t                        last_frame_number = 0;

    // Non-owning: the consumer owns these and outlives the Stream.
    AVBufferRef*             gpu_frames_ctx = nullptr;
    /// Which hardware pixel format `gpu_frames_ctx` holds: CUDA for NVENC, Vulkan for
    /// FFmpeg 8's Vulkan encoders. Meaningless when `gpu_frames_ctx` is null.
    AVPixelFormat            gpu_hw_pix_fmt = AV_PIX_FMT_CUDA;
    /// Copies the mixer's composite into a Vulkan frame. Set only on the Vulkan encode path.
    class accelerator::vulkan::av_vulkan_exporter* vk_exporter = nullptr;
    class cuda_gl_uploader*  gpu_uploader   = nullptr;
    class cuda_vk_uploader*  gpu_uploader_vk = nullptr;
    std::atomic<bool>*       gpu_direct     = nullptr;
    int                      gpu_failures   = 0;
    /// How many consecutive GPU-direct failures it takes to abandon the path.
    ///
    /// More than one, because the commonest failure is transient: the hardware frame pool
    /// momentarily empty while the encoder holds every frame in it. At one, a single busy
    /// moment demoted the whole recording to host readback -- slower, and a different picture.
    /// 32 frames is a little over a second at 25p, long enough that a genuinely broken path
    /// still gives up promptly and short enough that a real stall is not ridden out for
    /// minutes. NOT reset on success: a path that misses one frame in ten is broken too, just
    /// more slowly, and this counts every miss over the recording's life.
    static constexpr int     kGpuFailuresBeforeGivingUp = 32;
    /// The reason last reported, so a CHANGE of reason is logged rather than only the first
    /// one. Without this the log says why the first frame failed and then goes silent, which
    /// cannot distinguish "the same guard every frame" from "one guard at startup and a
    /// different one later" -- and those want opposite fixes.
    std::string              gpu_last_logged_reason;

    //: HDR10 static metadata, attached to every frame handed to the encoder when the channel
    //: is PQ or HLG and <hdr-metadata> was configured. Off unless both are true -- see the
    //: note where it is read.
    bool   hdr_static_metadata = false;
    double hdr_min_dml         = 0.0;
    double hdr_max_dml         = 0.0;
    int    hdr_max_cll         = 0;
    int    hdr_max_fall        = 0;

    Stream(AVFormatContext*                    oc,
           std::string                         suffix,
           AVCodecID                           codec_id,
           const core::video_format_desc&      format_desc,
           bool                                realtime,
           common::bit_depth                   depth,
           std::map<std::string, std::string>& options,
           core::color_space                   channel_cs       = core::color_space::bt709,
           core::color_transfer                channel_transfer = core::color_transfer::sdr,
           AVBufferRef*                        gpu_frames       = nullptr,
           class cuda_gl_uploader*             uploader         = nullptr,
           std::atomic<bool>*                  gpu_direct_flag  = nullptr,
           class cuda_vk_uploader*             uploader_vk      = nullptr,
           AVPixelFormat                       hw_pix_fmt       = AV_PIX_FMT_CUDA,
           class accelerator::vulkan::av_vulkan_exporter* exporter = nullptr,
           const char*                         vk_convert_filter = nullptr,
           int                                 field_mode_request = -1)
        : gpu_frames_ctx(gpu_frames)
        , gpu_uploader(uploader)
        , gpu_direct(gpu_direct_flag)
        , gpu_uploader_vk(uploader_vk)
        , gpu_hw_pix_fmt(hw_pix_fmt)
        , vk_exporter(exporter)
    {
        if (codec_id == AV_CODEC_ID_TIMECODE) {
            is_ltc = true;
            st = avformat_new_stream(oc, nullptr);
            if (!st) {
                 FF_RET(AVERROR(ENOMEM), "avformat_new_stream");
            }
            
            st->codecpar->codec_type = AVMEDIA_TYPE_DATA;
            st->codecpar->codec_tag  = MKTAG('t', 'm', 'c', 'd');
            st->codecpar->codec_id   = AV_CODEC_ID_NONE; // tmcd identified by tag in FFmpeg 7+
            st->time_base            = av_inv_q(AVRational{format_desc.framerate.numerator(),
                                                           format_desc.framerate.denominator()});
            // st->avg_frame_rate is usually not set for data streams but maybe good for ref
            return;
        }

        std::map<std::string, std::string> stream_options;

        {
            auto tmp = std::move(options);
            for (auto& p : tmp) {
                if (boost::algorithm::ends_with(p.first, suffix)) {
                    const auto key = p.first.substr(0, p.first.size() - suffix.size());
                    stream_options.emplace(key, std::move(p.second));
                } else {
                    options.insert(std::move(p));
                }
            }
        }

        std::string filter_spec = "";
        {
            const auto it = stream_options.find("filter");
            if (it != stream_options.end()) {
                filter_spec = std::move(it->second);
                stream_options.erase(it);
            }
        }

        // The Vulkan encode path's own conversion filter, passed as an argument rather than
        // pushed through the options map. Going through the map looked tidier and did not work:
        // the caller's `options["filter:v"]` never arrived as `stream_options["filter"]`, so the
        // chain stayed empty, lavfi inserted a software `auto_scale` after the Vulkan buffersrc,
        // and graph configuration failed with "Impossible to convert between the formats
        // supported by the filter 'in_0' and the filter 'auto_scale_0'". An explicit parameter
        // cannot be lost in translation.
        //
        // A user-supplied filter still wins -- the decision upstream declines the Vulkan path
        // entirely when one was given, because this chain has to own the format negotiation.
        if (filter_spec.empty() && vk_convert_filter != nullptr)
            filter_spec = vk_convert_filter;

        auto codec = avcodec_find_encoder(codec_id);
        {
            const auto it = stream_options.find("codec");
            if (it != stream_options.end()) {
                codec = avcodec_find_encoder_by_name(it->second.c_str());
                stream_options.erase(it);
            } else if (suffix == ":v") {
                const auto v_it = options.find("vcodec");
                if (v_it != options.end()) {
                    codec = avcodec_find_encoder_by_name(v_it->second.c_str());
                    options.erase(v_it);
                }
            } else if (suffix == ":a") {
                const auto a_it = options.find("acodec");
                if (a_it != options.end()) {
                    codec = avcodec_find_encoder_by_name(a_it->second.c_str());
                    options.erase(a_it);
                }
            }
        }

        if (!codec) {
            FF_RET(AVERROR(EINVAL), "avcodec_find_encoder");
        }

        // ── NVENC input format ────────────────────────────────────────────
        // NVENC takes BGRA/RGBA surfaces directly and converts to YCbCr inside
        // the encoder, on the GPU. This used to insert "format=nv12,hwupload_cuda"
        // with a comment claiming the conversion happened on the GPU via
        // scale_cuda. It did not: format=nv12 is host libswscale, and scale_cuda
        // cannot accept RGB input at all -- it rejects it at graph-config time.
        //
        // Handing NVENC the RGB frame is better on every axis measured (1080p,
        // 300 frames, h264_nvenc p4): 4.30 s of CPU instead of 5.55 s -- about
        // 4.2 ms per frame, a fifth of a core at 50 fps -- 50.6 dB instead of
        // 48.9 dB against the lossless source, and a marginally smaller file.
        // Forcing nv12 threw away chroma on the host that the encoder would
        // otherwise have subsampled itself, which is where the quality went.
        //
        // Nothing needs to be inserted to get there: the buffersink already
        // negotiates against codec->pix_fmts below, and picks bgra for an 8-bit
        // channel and rgba for a 16-bit one.
        //
        // A CUDA device is still attached when the caller's own filter chain
        // asks for one, so an explicit "hwupload_cuda,..." in the filter option
        // keeps working.
        AVBufferRef* hw_device_ctx_ = nullptr;
        bool         use_nvenc_hw_  = false;
        if (codec->type == AVMEDIA_TYPE_VIDEO &&
            (filter_spec.find("cuda") != std::string::npos || filter_spec.find("hwupload") != std::string::npos)) {
            if (av_hwdevice_ctx_create(&hw_device_ctx_, AV_HWDEVICE_TYPE_CUDA, nullptr, nullptr, 0) == 0) {
                use_nvenc_hw_ = true;
                CASPAR_LOG(info) << L"[ffmpeg] CUDA filter in the chain: hw device attached";
            } else {
                CASPAR_LOG(warning) << L"[ffmpeg] CUDA hw device creation failed - the cuda filters in \""
                                    << u16(filter_spec) << L"\" will not configure";
            }
        }
        // The filter graph and the encoder each take their own reference; this one
        // was never released, so every recording leaked a CUDA device context.
        CASPAR_SCOPE_EXIT
        {
            if (hw_device_ctx_)
                av_buffer_unref(&hw_device_ctx_);
        };

        AVFilterInOut* outputs = nullptr;
        AVFilterInOut* inputs  = nullptr;

        CASPAR_SCOPE_EXIT
        {
            avfilter_inout_free(&inputs);
            avfilter_inout_free(&outputs);
        };

        graph = std::shared_ptr<AVFilterGraph>(avfilter_graph_alloc(),
                                               [](AVFilterGraph* ptr) { avfilter_graph_free(&ptr); });

        if (!graph) {
            FF_RET(AVERROR(ENOMEM), "avfilter_graph_alloc");
        }

        if (codec->type == AVMEDIA_TYPE_VIDEO) {
            if (filter_spec.empty()) {
                filter_spec = "null";
            }
        } else {
            if (filter_spec.empty()) {
                filter_spec = "anull";
            }
        }

        // PARSE, CREATE, GIVE THE DEVICE, *THEN* INIT -- the segment API rather than
        // `avfilter_graph_parse2`, and the order is the whole point.
        //
        // `parse2` creates AND INITIALISES in one call (`graphparser.c` calls
        // `avfilter_init_dict` inside it), and a hardware filter decides at INIT time whether
        // it can accept hardware frames: `vf_libplacebo.c` reads `avctx->hw_device_ctx` in its
        // init, sets `have_hwdevice` from it, and only then offers `AV_PIX_FMT_VULKAN` as an
        // input format at all (`libplacebo_query_format`, guarded on that flag). Assigning the
        // device afterwards is too late by construction, however early it looks in the code.
        //
        // The symptom was lavfi inserting a SOFTWARE `auto_scale` straight after the Vulkan
        // buffersrc and then failing to configure: "Impossible to convert between the formats
        // supported by the filter 'in_0' and the filter 'auto_scale_0'". A graph log made it
        // clear the device HAD been assigned -- `1 of 3 filter(s) took the hw device` -- which
        // is what ruled out the obvious explanations and left the ordering.
        //
        // This is the order `fftools/ffmpeg_filter.c` uses for exactly this reason: segment
        // parse, `segment_create_filters`, assign `hw_device_ctx` to every filter carrying
        // AVFILTER_FLAG_HWDEVICE, then apply. Worth copying rather than re-deriving.
        {
            AVFilterGraphSegment* seg = nullptr;
            FF(avfilter_graph_segment_parse(graph.get(), filter_spec.c_str(), 0, &seg));
            CASPAR_SCOPE_EXIT { avfilter_graph_segment_free(&seg); };
            FF(avfilter_graph_segment_create_filters(seg, 0));

            // The device the frames pool was allocated from, so filter, pool and encoder agree
            // by construction. Falls back to the CUDA device a user's own cuda filters need.
            AVBufferRef* filter_device = nullptr;
            if (gpu_frames_ctx) {
                if (auto* frames = reinterpret_cast<AVHWFramesContext*>(gpu_frames_ctx->data))
                    filter_device = frames->device_ref;
            }
            if (!filter_device && use_nvenc_hw_)
                filter_device = hw_device_ctx_;

            if (filter_device) {
                for (unsigned i = 0; i < graph->nb_filters; i++) {
                    if (graph->filters[i]->filter->flags & AVFILTER_FLAG_HWDEVICE)
                        graph->filters[i]->hw_device_ctx = av_buffer_ref(filter_device);
                }
            }

            FF(avfilter_graph_segment_apply(seg, 0, &inputs, &outputs));
        }

        {
            auto cur = inputs;

            if (!cur || cur->next) {
                CASPAR_THROW_EXCEPTION(ffmpeg_error_t() << boost::errinfo_errno(EINVAL)
                                                        << msg_info_t("invalid filter graph input count"));
            }

            if (codec->type == AVMEDIA_TYPE_VIDEO) {
                const auto sar = boost::rational<int>(format_desc.square_width, format_desc.square_height) /
                                 boost::rational<int>(format_desc.width, format_desc.height);

                // PAIR FIELDS on an interlaced channel, and only on the host path.
                //
                // The GPU-direct routes hand the encoder a device frame per tick; pairing two of
                // them is a GPU line-interleave that does not exist yet, so those routes decline
                // interlaced channels outright (see the decline chains below) and this flag can
                // simply require the host path rather than duplicate the reason here.
                //
                // Field dominance has to be derived from the video mode because the core has
                // carried no field dominance since the 2018 refactor -- `video_format_desc` has
                // `field_count` and nothing else. Same rule as
                // cuda_prores/consumer/prores_consumer.cu's `format_is_tff`: SD PAL/NTSC are
                // bottom-field-first, everything else interlaced is top. Two consumers deriving
                // this separately is a hazard, and it is recorded as one in the docs.
                // `-interlaced` decides this; the channel only says what is possible.
                //
                //   auto (default)  pair when the channel is interlaced
                //   0               never pair -- write every tick as its own progressive
                //                   frame, which is what this consumer did before pairing
                //                   existed. Legitimate when the deliverable is 50p.
                //   1               pair, and complain if the channel has nothing to pair
                const bool channel_interlaced = format_desc.field_count == 2;
                const bool want_pair          = field_mode_request != 0;

                if (field_mode_request == 1 && !channel_interlaced) {
                    CASPAR_LOG(warning) << L"[ffmpeg] -interlaced 1 on a progressive channel: "
                                           L"there is no second field to pair, recording "
                                           L"progressive";
                }

                pair_fields_ = channel_interlaced && want_pair &&
                               (!gpu_frames_ctx || vk_exporter != nullptr);

                if (channel_interlaced && !want_pair) {
                    CASPAR_LOG(info) << L"[ffmpeg] -interlaced 0: recording the interlaced "
                                        L"channel at field rate as progressive frames. Note that "
                                        L"consecutive ticks can be the same picture, in which "
                                        L"case this writes each one twice.";
                }
                tff_         = !(format_desc.format == core::video_format::pal ||
                         format_desc.format == core::video_format::ntsc);
                if (pair_fields_) {
                    CASPAR_LOG(info) << L"[ffmpeg] interlaced recording: pairing two channel "
                                        L"ticks into one "
                                     << (tff_ ? L"top" : L"bottom")
                                     << L"-field-first frame, so the file is "
                                     << format_desc.framerate.numerator() /
                                            format_desc.framerate.denominator()
                                     << L" fps field-coded rather than "
                                     << (format_desc.framerate.numerator() * 2) /
                                            format_desc.framerate.denominator()
                                     << L" fps progressive";
                }

                // 8-bit mixer outputs BGRA (.bgra shader swizzle); 16-bit outputs RGBA directly.
                // With the GPU-direct path the graph carries CUDA frames, so the
                // buffersrc has to be told that and given the frames context --
                // otherwise it is configured for host BGRA and rejects them.
                // `gpu_hw_pix_fmt` rather than a hardcoded CUDA: the same GPU-direct plumbing
                // now carries either CUDA frames (NVENC) or Vulkan frames (FFmpeg 8's Vulkan
                // encoders), and the buffersrc has to be told which. Everything downstream --
                // the buffersrc parameters, the encoder's hw_frames_ctx and hw_device_ctx --
                // was already generic over the frames context and needed no change.
                const auto pix_fmt = gpu_frames_ctx ? gpu_hw_pix_fmt
                                     : (depth == common::bit_depth::bit8) ? AV_PIX_FMT_BGRA
                                                                          : AV_PIX_FMT_RGBA64LE;

                // colorspace/range are not decoration. Left unspecified, the graph has
                // no input colour information, so the scale filter libswscale inserts
                // to reach the encoder's YCbCr format converts with its own default
                // matrix -- BT.601 -- while the encoder tags the file BT.709. The
                // recording then decodes with the wrong matrix: neutrals survive and
                // saturated colour shifts, which is why it went unnoticed.
                //
                // The mixer's frames are full-range RGB, and make_av_video_frame tags
                // them AVCOL_SPC_RGB / AVCOL_RANGE_JPEG. Declaring the same here is
                // what lets the conversion be negotiated instead of guessed.
                auto args = (boost::format("video_size=%dx%d:pix_fmt=%d:time_base=%d/%d:sar=%d/%d:frame_rate=%d/%d"
                                           ":colorspace=%d:range=%d") %
                             format_desc.width % format_desc.height % pix_fmt % format_desc.duration %
                             (format_desc.time_scale * (pair_fields_ ? 1 : format_desc.field_count)) %
                             sar.numerator() % sar.denominator() %
                             (format_desc.framerate.numerator() *
                              (pair_fields_ ? 1 : format_desc.field_count)) %
                             format_desc.framerate.denominator() %
                             static_cast<int>(gpu_frames_ctx ? AVCOL_SPC_RGB : AVCOL_SPC_RGB) %
                             static_cast<int>(AVCOL_RANGE_JPEG))
                                .str();
                auto name = (boost::format("in_%d") % 0).str();

                // ALLOCATE, PARAMETERISE, THEN INIT -- in that order, and only that order works
                // for a hardware format.
                //
                // `avfilter_graph_create_filter` initialises immediately, and buffersrc
                // validates on init: "Setting BufferSourceContext.pix_fmt to a HW format
                // requires hw_frames_ctx to be non-NULL!". The frames context can only be
                // supplied through `av_buffersrc_parameters_set`, which has to happen BEFORE
                // init -- so creating and then parameterising, which is what this did, fails
                // for every hardware format. Measured 2026-08-22: all four Vulkan encoders took
                // the path, logged it, and produced a 0-byte file, because the graph never
                // configured.
                //
                // The software path is unchanged in effect: same args, same filter, just built
                // in two steps instead of one.
                source = avfilter_graph_alloc_filter(graph.get(), avfilter_get_by_name("buffer"), name.c_str());
                if (!source)
                    FF_RET(AVERROR(ENOMEM), "avfilter_graph_alloc_filter");

                if (gpu_frames_ctx) {
                    AVBufferSrcParameters* par = av_buffersrc_parameters_alloc();
                    if (!par)
                        FF_RET(AVERROR(ENOMEM), "av_buffersrc_parameters_alloc");
                    par->format        = gpu_hw_pix_fmt;
                    par->hw_frames_ctx = gpu_frames_ctx;
                    // Not named "ret": the FF macro declares its own.
                    const auto set_result = av_buffersrc_parameters_set(source, par);
                    av_free(par);
                    FF(set_result);
                }

                // Init last, now that a hardware format has its frames context.
                FF(avfilter_init_str(source, args.c_str()));
                FF(avfilter_link(source, 0, cur->filter_ctx, cur->pad_idx));
            } else if (codec->type == AVMEDIA_TYPE_AUDIO) {
                auto args = (boost::format("time_base=%d/%d:sample_rate=%d:sample_fmt=%s:channel_layout=%#x") % 1 %
                             format_desc.audio_sample_rate % format_desc.audio_sample_rate % AV_SAMPLE_FMT_S32 %
                             get_channel_layout_mask_for_channels(format_desc.audio_channels))
                                .str();
                auto name = (boost::format("in_%d") % 0).str();

                FF(avfilter_graph_create_filter(
                    &source, avfilter_get_by_name("abuffer"), name.c_str(), args.c_str(), nullptr, graph.get()));
                FF(avfilter_link(source, 0, cur->filter_ctx, cur->pad_idx));
            } else {
                CASPAR_THROW_EXCEPTION(ffmpeg_error_t() << boost::errinfo_errno(EINVAL)
                                                        << msg_info_t("invalid filter input media type"));
            }
        }

        if (codec->type == AVMEDIA_TYPE_VIDEO) {
            sink = FFMEM(avfilter_graph_alloc_filter(graph.get(), avfilter_get_by_name("buffersink"), "out"));

            // TODO codec->profiles
            // TODO FF(av_opt_set_int_list(sink, "framerates", codec->supported_framerates, { 0, 0 },
            // AV_OPT_SEARCH_CHILDREN));

            // ── Which pixel format to negotiate ───────────────────────────
            // Handing the sink the codec's whole list lets it choose by
            // conversion cost from the channel's BGRA, and for H.264/HEVC that
            // lands on yuv444p -- High 4:4:4 Predictive, a profile most players
            // and effectively every hardware decoder refuse. A recording that
            // will not play back is a poor default, so prefer 8-bit 4:2:0 where
            // the codec offers it.
            //
            // Deliberately narrow in scope:
            //   - Codecs with no 4:2:0 mode (ProRes, DNxHD) keep the old
            //     negotiation; there is nothing to prefer.
            //   - Hardware encoders keep it too. They accept RGB and subsample
            //     inside the encoder, which measured better than doing it here
            //     (see the NVENC note above).
            //   - An explicit pix_fmt from the caller wins outright. That also
            //     closes a latent inconsistency: pix_fmt in the options dict is
            //     applied to the codec context at avcodec_open2 and would
            //     silently disagree with the format the sink actually emits.
            std::vector<AVPixelFormat> pix_fmts;
            {
                // Consumed here rather than left in the dict: the sink is what
                // actually decides the format, and enc->pix_fmt is read back from
                // it below, so letting the option also reach avcodec_open2 would
                // be redundant at best and contradictory at worst. Erasing it also
                // stops it being reported as an unused option when it was used.
                const auto requested = [&]() -> std::string {
                    for (const auto* key : {"pix_fmt", "pixel_format"}) {
                        for (auto* map : {&stream_options, &options}) {
                            const auto it = map->find(key);
                            if (it != map->end()) {
                                auto value = std::move(it->second);
                                map->erase(it);
                                return value;
                            }
                        }
                    }
                    return {};
                }();

                const auto supports = [&](AVPixelFormat f) {
                    for (auto p = codec->pix_fmts; p && *p != AV_PIX_FMT_NONE; ++p)
                        if (*p == f)
                            return true;
                    return false;
                };

                if (!requested.empty()) {
                    const auto f = av_get_pix_fmt(requested.c_str());
                    if (f != AV_PIX_FMT_NONE && supports(f)) {
                        pix_fmts = {f};
                    } else {
                        CASPAR_LOG(warning) << L"[ffmpeg] " << u16(codec->name) << L" does not support pix_fmt \""
                                            << u16(requested) << L"\"; negotiating instead";
                    }
                } else if (!(codec->capabilities & AV_CODEC_CAP_HARDWARE) && supports(AV_PIX_FMT_YUV420P)) {
                    pix_fmts = {AV_PIX_FMT_YUV420P};
                }
            }

            // buffersink's `pix_fmts` / `color_spaces` / `color_ranges` options are
            // AV_OPT_FLAG_DEPRECATED in FFmpeg 8 and disappear with libavfilter 12
            // (FF_API_BUFFERSINK_OPTS), replaced by array-typed `pixel_formats` /
            // `colorspaces` / `colorranges`. Both forms resolve on 8.1, so this guard is
            // the FFmpeg 9 bill paid early rather than something 8.1 requires.
            //
            // The array form takes a COUNT where the int-list form took a TERMINATOR, so
            // each count is read before any terminator is appended -- appending first
            // would offer the sink AV_PIX_FMT_NONE as though it were a real format.
            if (!pix_fmts.empty()) {
#if LIBAVUTIL_VERSION_MAJOR >= 60 // FFmpeg 8
                FF(av_opt_set_array(sink,
                                    "pixel_formats",
                                    AV_OPT_SEARCH_CHILDREN | AV_OPT_ARRAY_REPLACE,
                                    0,
                                    static_cast<unsigned int>(pix_fmts.size()),
                                    AV_OPT_TYPE_PIXEL_FMT,
                                    pix_fmts.data()));
#else
                pix_fmts.push_back(AV_PIX_FMT_NONE);
                FF(av_opt_set_int_list(sink, "pix_fmts", pix_fmts.data(), AV_PIX_FMT_NONE, AV_OPT_SEARCH_CHILDREN));
#endif
            } else {
#if LIBAVUTIL_VERSION_MAJOR >= 60 // FFmpeg 8
                // Counted from codec->pix_fmts rather than read via
                // avcodec_get_supported_config, so that this and the `supports` lambda
                // above read the SAME list. Two sources of truth here could offer the
                // sink a format `supports` had already rejected.
                unsigned int nb_codec_fmts = 0;
                for (auto q = codec->pix_fmts; q != nullptr && *q != AV_PIX_FMT_NONE; ++q)
                    ++nb_codec_fmts;
                FF(av_opt_set_array(sink,
                                    "pixel_formats",
                                    AV_OPT_SEARCH_CHILDREN | AV_OPT_ARRAY_REPLACE,
                                    0,
                                    nb_codec_fmts,
                                    AV_OPT_TYPE_PIXEL_FMT,
                                    codec->pix_fmts));
#else
                FF(av_opt_set_int_list(sink, "pix_fmts", codec->pix_fmts, AV_PIX_FMT_NONE, AV_OPT_SEARCH_CHILDREN));
#endif
            }

            // ── Which matrix libswscale converts RGB->YCbCr with ──────────
            // The encoder below is told the channel's colour space, so the file
            // is *tagged* BT.709. Nothing told the graph, so libswscale used its
            // own default -- BT.601 -- and the recording came out encoded with
            // one matrix and labelled with the other.
            //
            // The error is invisible on neutrals and obvious on saturated
            // colour, which is why it survived: a grey ramp round-trips fine
            // while a saturated green went (15,223,5) in and (0,189,0) out, a
            // shift that reproduces exactly as "encode 601, decode 709". The
            // GPU-direct NVENC path was never affected -- it hands RGB to the
            // encoder and never runs libswscale -- so the two recording paths
            // disagreed with each other, which is how it surfaced.
            //
            // Constraining the sink rather than inserting a scale filter keeps
            // format negotiation intact: an RGB encoder still gets RGB and no
            // conversion is inserted at all.
            {
                // Is every format the sink may settle on an RGB one? An RGB encoder
                // takes the channel's frames as they are, and asking for a YCbCr
                // matrix there is meaningless -- the link genuinely is RGB.
                const auto is_rgb_fmt = [](AVPixelFormat f) {
                    const auto* d = av_pix_fmt_desc_get(f);
                    return d && (d->flags & AV_PIX_FMT_FLAG_RGB);
                };
                bool rgb_output = true;
                if (!pix_fmts.empty()) {
                    for (auto f : pix_fmts)
                        if (f != AV_PIX_FMT_NONE && !is_rgb_fmt(f))
                            rgb_output = false;
                } else {
                    for (auto q = codec->pix_fmts; q && *q != AV_PIX_FMT_NONE; ++q)
                        if (!is_rgb_fmt(*q))
                            rgb_output = false;
                }

                if (!rgb_output) {
                    // Exactly one matrix, and not AVCOL_SPC_RGB alongside it. Offering
                    // RGB as an alternative is not harmless: converting from the
                    // channel's RGB costs nothing, so negotiation picks it and the
                    // constraint silently does nothing -- the sink then reports
                    // "matrix gbr" on a yuv420p link and libswscale converts with its
                    // own default anyway. That was the first attempt at this fix.
                    const AVColorSpace cs = channel_cs == core::color_space::bt2020 ? AVCOL_SPC_BT2020_NCL
                                            : channel_cs == core::color_space::bt601
                                                ? AVCOL_SPC_SMPTE170M
                                                : AVCOL_SPC_BT709;
                    const AVColorSpace spaces[] = {cs, AVCOL_SPC_UNSPECIFIED};
#if LIBAVUTIL_VERSION_MAJOR >= 60 // FFmpeg 8
                    // ONE element, not two: AVCOL_SPC_UNSPECIFIED is the int-list
                    // terminator, and passing it as a member of the array would offer
                    // "unspecified" as an acceptable matrix -- which is the
                    // no-constraint case this whole block exists to prevent.
                    FF(av_opt_set_array(sink,
                                        "colorspaces",
                                        AV_OPT_SEARCH_CHILDREN | AV_OPT_ARRAY_REPLACE,
                                        0,
                                        1,
                                        AV_OPT_TYPE_INT,
                                        spaces));
#else
                    FF(av_opt_set_int_list(sink, "color_spaces", spaces, AVCOL_SPC_UNSPECIFIED,
                                           AV_OPT_SEARCH_CHILDREN));
#endif

                    // Matches enc->color_range below. A full/limited mismatch is the
                    // same class of bug with a different signature: crushed blacks and
                    // clipped whites rather than shifted saturated colour.
                    const AVColorRange ranges[] = {AVCOL_RANGE_MPEG, AVCOL_RANGE_UNSPECIFIED};
#if LIBAVUTIL_VERSION_MAJOR >= 60 // FFmpeg 8
                    FF(av_opt_set_array(sink,
                                        "colorranges",
                                        AV_OPT_SEARCH_CHILDREN | AV_OPT_ARRAY_REPLACE,
                                        0,
                                        1,
                                        AV_OPT_TYPE_INT,
                                        ranges));
#else
                    FF(av_opt_set_int_list(sink, "color_ranges", ranges, AVCOL_RANGE_UNSPECIFIED,
                                           AV_OPT_SEARCH_CHILDREN));
#endif
                }
            }

        } else if (codec->type == AVMEDIA_TYPE_AUDIO) {
            sink = FFMEM(avfilter_graph_alloc_filter(graph.get(), avfilter_get_by_name("abuffersink"), "out"));
            // TODO codec->profiles

#if LIBAVUTIL_VERSION_MAJOR >= 60 // FFmpeg 8
            const void* sample_fmts;
            int         nb_sample_fmts = 0;
            FF(avcodec_get_supported_config(
                nullptr, codec, AV_CODEC_CONFIG_SAMPLE_FORMAT, 0, &sample_fmts, &nb_sample_fmts));

            FF(av_opt_set_array(sink,
                                "sample_formats",
                                AV_OPT_SEARCH_CHILDREN | AV_OPT_ARRAY_REPLACE,
                                0,
                                nb_sample_fmts,
                                AV_OPT_TYPE_SAMPLE_FMT,
                                sample_fmts));

            const void* sample_rates;
            int         nb_sample_rates = 0;
            FF(avcodec_get_supported_config(
                nullptr, codec, AV_CODEC_CONFIG_SAMPLE_RATE, 0, &sample_rates, &nb_sample_rates));

            FF(av_opt_set_array(sink,
                                "samplerates",
                                AV_OPT_SEARCH_CHILDREN | AV_OPT_ARRAY_REPLACE,
                                0,
                                nb_sample_rates,
                                AV_OPT_TYPE_INT,
                                sample_rates));
#else
            FF(av_opt_set_int_list(sink, "sample_fmts", codec->sample_fmts, -1, AV_OPT_SEARCH_CHILDREN));
            FF(av_opt_set_int_list(sink, "sample_rates", codec->supported_samplerates, 0, AV_OPT_SEARCH_CHILDREN));
#endif

            // -- Which channel layout to negotiate ------------------------
            //
            // A CasparCG channel always carries 16 audio channels, and nothing here
            // constrained the sink's layout, so those 16 reached the encoder unchanged
            // and the graph never downmixed. FFmpeg 7's native AAC encoder took them
            // anyway and wrote `channel_layout=unknown`; FFmpeg 8 refuses with
            // `Unsupported channel layout "9.1.6"` and avcodec_open2 returns EINVAL --
            // so every file and stream consumer carrying audio failed to open. A latent
            // bug that FFmpeg 8 stopped covering for, not an FFmpeg 8 regression.
            //
            // Measured 2026-08-19: h264, h265, prores, dnxhd, dnxhr and xdcam consumers
            // all failed, so this was never AAC-specific despite presenting that way.
            //
            // Two sources of truth, in this order:
            //
            //   1. An explicit `-ac N` from the caller wins, and it has to be honoured
            //      HERE. `ac` is a CLI-level option, not an AVCodecContext AVOption, so
            //      passing it through to avcodec_open2 did nothing whatsoever -- the
            //      encoder still saw 16 channels. This is also the ONLY route that fixes
            //      the native AAC encoder, which publishes no layout list to constrain
            //      against.
            //   2. Otherwise the encoder's own published list, when it has one. That
            //      fixes ac3 and mp2, which have the same latent bug.
            //
            // Deliberately NOT a blanket downmix default: PCM in WAV records all 16
            // channels correctly today and must keep doing so, and an encoder that
            // accepts any layout publishes no list precisely because it needs none.
            const auto requested_channels = [&]() -> int {
                // Consumed rather than left in the dict, like pix_fmt above: the sink is
                // what decides the layout, enc->ch_layout is read back from it below, and
                // leaving the key in place would have it reported as an unused option
                // when it was in fact used.
                for (const auto* key : {"ac", "channels"}) {
                    for (auto* map : {&stream_options, &options}) {
                        const auto it = map->find(key);
                        if (it != map->end()) {
                            const auto value = std::move(it->second);
                            map->erase(it);
                            try {
                                const auto n = std::stoi(value);
                                if (n <= 0 || n > 64) {
                                    CASPAR_THROW_EXCEPTION(
                                        ffmpeg_error_t()
                                        << boost::errinfo_errno(EINVAL)
                                        << msg_info_t("audio channel count out of range: " + value));
                                }
                                return n;
                            } catch (const std::invalid_argument&) {
                                CASPAR_THROW_EXCEPTION(ffmpeg_error_t()
                                                       << boost::errinfo_errno(EINVAL)
                                                       << msg_info_t("invalid audio channel count: " + value));
                            } catch (const std::out_of_range&) {
                                CASPAR_THROW_EXCEPTION(ffmpeg_error_t()
                                                       << boost::errinfo_errno(EINVAL)
                                                       << msg_info_t("audio channel count out of range: " + value));
                            }
                        }
                    }
                }
                return 0;
            }();

#if LIBAVUTIL_VERSION_MAJOR >= 60 // FFmpeg 8
            if (requested_channels > 0) {
                AVChannelLayout want{};
                av_channel_layout_default(&want, requested_channels);
                const auto set_result = av_opt_set_array(sink,
                                                         "channel_layouts",
                                                         AV_OPT_SEARCH_CHILDREN | AV_OPT_ARRAY_REPLACE,
                                                         0,
                                                         1,
                                                         AV_OPT_TYPE_CHLAYOUT,
                                                         &want);
                av_channel_layout_uninit(&want);
                FF(set_result);
            } else {
                const void* ch_layouts    = nullptr;
                int         nb_ch_layouts = 0;
                FF(avcodec_get_supported_config(
                    nullptr, codec, AV_CODEC_CONFIG_CHANNEL_LAYOUT, 0, &ch_layouts, &nb_ch_layouts));

                // A NULL list means "publishes no constraint", which is NOT the same as
                // "supports nothing". pcm_s16le accepts any layout and publishes none;
                // the native AAC encoder also publishes none and then rejects 16 channels
                // inside its own init, against a table it does not expose. So the list can
                // only be applied when one actually exists, and its absence is why `-ac`
                // above is the only lever for AAC.
                if (ch_layouts != nullptr && nb_ch_layouts > 0) {
                    FF(av_opt_set_array(sink,
                                        "channel_layouts",
                                        AV_OPT_SEARCH_CHILDREN | AV_OPT_ARRAY_REPLACE,
                                        0,
                                        nb_ch_layouts,
                                        AV_OPT_TYPE_CHLAYOUT,
                                        ch_layouts));
                }
            }
#else
            if (requested_channels > 0) {
                const int64_t layouts[] = {
                    static_cast<int64_t>(get_channel_layout_mask_for_channels(requested_channels)), 0};
                FF(av_opt_set_int_list(sink, "channel_layouts", layouts, 0, AV_OPT_SEARCH_CHILDREN));
            }
#endif

        } else {
            CASPAR_THROW_EXCEPTION(ffmpeg_error_t()
                                   << boost::errinfo_errno(EINVAL) << msg_info_t("invalid output media type"));
        }

        FF(avfilter_init_str(sink, nullptr));

        {
            const auto cur = outputs;

            if (!cur || cur->next) {
                CASPAR_THROW_EXCEPTION(ffmpeg_error_t() << boost::errinfo_errno(EINVAL)
                                                        << msg_info_t("invalid filter graph output count"));
            }

            if (avfilter_pad_get_type(cur->filter_ctx->output_pads, cur->pad_idx) != codec->type) {
                CASPAR_THROW_EXCEPTION(ffmpeg_error_t() << boost::errinfo_errno(EINVAL)
                                                        << msg_info_t("invalid filter output media type"));
            }

            FF(avfilter_link(cur->filter_ctx, cur->pad_idx, sink, 0));
        }

        // EVERY hardware-capable filter needs the device, whichever API it is.
        //
        // This used to fire only for `use_nvenc_hw_`, so on the Vulkan encode path the
        // conversion filter got no device, could not accept hardware frames, and lavfi
        // inserted a SOFTWARE `auto_scale` between the Vulkan buffersrc and it -- which then
        // failed to configure at all: "Impossible to convert between the formats supported by
        // the filter 'in_0' and the filter 'auto_scale_0'". The recording took the path, logged
        // it, and produced a 0-byte file. Measured 2026-08-22.
        //
        // The device comes from the frames context on the GPU-direct paths -- that is the same
        // device the pool was allocated from, so the filter, the pool and the encoder all agree
        // by construction rather than by coincidence.
        AVBufferRef* filter_device = nullptr;
        if (gpu_frames_ctx) {
            if (auto* frames = reinterpret_cast<AVHWFramesContext*>(gpu_frames_ctx->data))
                filter_device = frames->device_ref;
        }
        if (!filter_device && use_nvenc_hw_)
            filter_device = hw_device_ctx_;

        if (filter_device) {
            int given = 0;
            for (unsigned i = 0; i < graph->nb_filters; i++) {
                if (graph->filters[i]->filter->flags & AVFILTER_FLAG_HWDEVICE) {
                    graph->filters[i]->hw_device_ctx = av_buffer_ref(filter_device);
                    ++given;
                }
            }
            // Which filters actually took a device, and what the chain is. Without this the
            // symptom -- lavfi inserting a software `auto_scale` after the hardware source --
            // says only that some filter refused hardware frames, not which or why.
            std::string chain;
            for (unsigned i = 0; i < graph->nb_filters; i++) {
                chain += (i ? " -> " : "");
                chain += graph->filters[i]->filter->name;
            }
            CASPAR_LOG(info) << L"[ffmpeg] filter graph: " << u16(chain) << L"; " << given
                             << L" of " << graph->nb_filters << L" filter(s) took the hw device";
        } else if (gpu_frames_ctx) {
            CASPAR_LOG(warning) << L"[ffmpeg] a GPU frames pool exists but no hw device could be "
                                   L"derived from it; hardware filters will refuse the frames";
        }

        FF(avfilter_graph_config(graph.get(), nullptr));

        st = avformat_new_stream(oc, nullptr);
        if (!st) {
            FF_RET(AVERROR(ENOMEM), "avformat_new_stream");
        }

        enc = std::shared_ptr<AVCodecContext>(avcodec_alloc_context3(codec),
                                              [](AVCodecContext* ptr) { avcodec_free_context(&ptr); });

        if (!enc) {
            FF_RET(AVERROR(ENOMEM), "avcodec_alloc_context3")
        }

        if (codec->type == AVMEDIA_TYPE_VIDEO) {
            st->time_base = av_inv_q(av_buffersink_get_frame_rate(sink));

            // Ensure the frame_rate is set in a way that rtmp will find it
            st->avg_frame_rate = av_buffersink_get_frame_rate(sink);

            enc->width               = av_buffersink_get_w(sink);
            enc->height              = av_buffersink_get_h(sink);
            enc->framerate           = av_buffersink_get_frame_rate(sink);
            enc->sample_aspect_ratio = av_buffersink_get_sample_aspect_ratio(sink);
            enc->time_base           = st->time_base;
            enc->pix_fmt             = static_cast<AVPixelFormat>(av_buffersink_get_format(sink));

            // Which format the graph settled on decides whether a host colour
            // conversion runs at all: an RGB format here means the frame goes to
            // the encoder untouched, anything else means libswscale is in the
            // path. Worth one line at open time -- it is otherwise invisible.
            if (const char* fmt_name = av_get_pix_fmt_name(enc->pix_fmt)) {
                const auto* d      = av_pix_fmt_desc_get(enc->pix_fmt);
                const bool  is_hw  = d && (d->flags & AV_PIX_FMT_FLAG_HWACCEL);
                const bool  is_rgb = d && (d->flags & AV_PIX_FMT_FLAG_RGB);
                // lavfi 9.16.100 per FFmpeg's own doc/APIchanges. The CI image (Ubuntu
                // 24.04) ships FFmpeg 6.1 at lavfi 9.12, so this is guarded rather than
                // required -- the line is a diagnostic, and losing the matrix from it is
                // not worth refusing to build.
#if LIBAVFILTER_VERSION_INT >= AV_VERSION_INT(9, 16, 100)
                const auto neg_cs = av_buffersink_get_colorspace(sink);
#else
                const auto neg_cs = AVCOL_SPC_UNSPECIFIED;
#endif
                CASPAR_LOG(info) << L"[ffmpeg] " << u16(codec->name) << L" input format " << u16(fmt_name)
                                 << (is_hw    ? L" (device frames, no readback)"
                                     : is_rgb ? L" (no host conversion)"
                                              : L" (host conversion via libswscale)")
                                 << L", matrix " << u16(av_color_space_name(neg_cs) ? av_color_space_name(neg_cs)
                                                                                    : "unknown");
            }

            enc->color_range         = AVCOL_RANGE_MPEG;
            switch (channel_cs) {
                case core::color_space::bt2020:
                    enc->color_primaries = AVCOL_PRI_BT2020;
                    enc->colorspace      = AVCOL_SPC_BT2020_NCL;
                    enc->color_trc       = (channel_transfer == core::color_transfer::pq)
                                               ? AVCOL_TRC_SMPTE2084
                                           : (channel_transfer == core::color_transfer::hlg)
                                               ? AVCOL_TRC_ARIB_STD_B67
                                               : AVCOL_TRC_BT709;
                    break;
                case core::color_space::bt601:
                    enc->color_primaries = AVCOL_PRI_BT470BG;
                    enc->colorspace      = AVCOL_SPC_SMPTE170M;
                    enc->color_trc       = AVCOL_TRC_SMPTE170M;
                    break;
                default: // bt709
                    enc->color_primaries = AVCOL_PRI_BT709;
                    enc->colorspace      = AVCOL_SPC_BT709;
                    enc->color_trc       = AVCOL_TRC_BT709;
                    break;
            }

            // HDR10 STATIC METADATA, and only when it was actually configured.
            //
            // The colour fields above describe how to DECODE the picture and can be derived
            // from the channel. These four describe the DISPLAY the content was graded on,
            // which nothing in the server can know -- so they are attached only when
            // <hdr-metadata> says so. Inventing defaults would put a claim about someone's
            // grading suite into every HDR stream we emit, which is worse than sending
            // nothing: a downstream tone-mapper would act on it.
            //
            // Gated on the transfer as well: ST 2086 alongside an SDR stream is a
            // contradiction, and CTA-861.3 has no meaning without an HDR EOTF.
            {
                const auto take = [&](const char* key) -> std::string {
                    const auto it = options.find(key);
                    if (it == options.end()) {
                        return {};
                    }
                    auto value = it->second;
                    // Erased so it never reaches the encoder's AVDictionary, where an
                    // unrecognised option is an error rather than a no-op.
                    options.erase(it);
                    return value;
                };
                const auto max_dml  = take("hdr_max_dml");
                const auto min_dml  = take("hdr_min_dml");
                const auto max_cll  = take("hdr_max_cll");
                const auto max_fall = take("hdr_max_fall");

                const bool is_hdr = channel_transfer == core::color_transfer::pq ||
                                    channel_transfer == core::color_transfer::hlg;
                if (!max_dml.empty() && is_hdr) {
                    hdr_static_metadata = true;
                    hdr_max_dml         = std::stod(max_dml);
                    hdr_min_dml         = min_dml.empty() ? 0.005 : std::stod(min_dml);
                    hdr_max_cll         = max_cll.empty() ? 0 : std::stoi(max_cll);
                    hdr_max_fall        = max_fall.empty() ? 0 : std::stoi(max_fall);

                    // ON THE CODEC CONTEXT, BEFORE avcodec_open2 -- not on the frames.
                    //
                    // Attaching it per frame is the obvious reading of the API and produces
                    // nothing: measured with both libx264 and libx265, the stream carried only
                    // x264's own version SEI. FFmpeg's encoder wrappers build their HDR10 SEI
                    // during INIT, from `avctx->decoded_side_data`, so anything arriving with
                    // the frames is too late and is simply ignored -- silently, which is why
                    // this needed a stream capture to notice rather than a return code.
                    // Two APIs, two versions, both from FFmpeg's doc/APIchanges:
                    // av_frame_side_data_new and AV_FRAME_SIDE_DATA_FLAG_UNIQUE arrived in
                    // lavu 59.3.100, and AVCodecContext.[nb_]decoded_side_data in lavc
                    // 61.2.100 -- both FFmpeg 7.0. Ubuntu 24.04, which is the Linux CI
                    // image, ships 6.1 (lavu 58.29 / lavc 60), so neither exists there.
                    //
                    // The fallback returns nullptr, which every caller below already
                    // handles -- the HDR10 static metadata is simply not attached. That is
                    // a real loss of function on an old FFmpeg, and the honest one: there
                    // is no pre-7.0 way to reach the encoder's init-time side data, which
                    // is the whole finding this block records.
                    const auto add_side_data = [&](AVFrameSideDataType type, size_t size) -> void* {
#if LIBAVUTIL_VERSION_INT >= AV_VERSION_INT(59, 3, 100) &&                                             LIBAVCODEC_VERSION_INT >= AV_VERSION_INT(61, 2, 100)
                        auto* sd = av_frame_side_data_new(&enc->decoded_side_data,
                                                          &enc->nb_decoded_side_data, type, size,
                                                          AV_FRAME_SIDE_DATA_FLAG_UNIQUE);
                        if (sd != nullptr) {
                            std::memset(sd->data, 0, size);
                        }
                        return sd != nullptr ? sd->data : nullptr;
#else
                        (void)type;
                        (void)size;
                        return nullptr;
#endif
                    };

                    if (auto* raw = add_side_data(AV_FRAME_DATA_MASTERING_DISPLAY_METADATA,
                                                  sizeof(AVMasteringDisplayMetadata))) {
                        auto* mdm = reinterpret_cast<AVMasteringDisplayMetadata*>(raw);
                        // Primaries from FFmpeg's own table for whatever the encoder is tagged
                        // with, so they cannot drift from `enc->color_primaries` the way a
                        // second hardcoded copy would.
                        if (const auto* desc = av_csp_primaries_desc_from_id(enc->color_primaries)) {
                            // NOTE the order: this struct is documented r, g, b, which is NOT
                            // the g, b, r the H.265 SEI uses -- the wrapper reorders it.
                            // Filling it in SEI order here yields a packet that parses cleanly
                            // and describes a display that does not exist.
                            mdm->display_primaries[0][0] = desc->prim.r.x;
                            mdm->display_primaries[0][1] = desc->prim.r.y;
                            mdm->display_primaries[1][0] = desc->prim.g.x;
                            mdm->display_primaries[1][1] = desc->prim.g.y;
                            mdm->display_primaries[2][0] = desc->prim.b.x;
                            mdm->display_primaries[2][1] = desc->prim.b.y;
                            mdm->white_point[0]          = desc->wp.x;
                            mdm->white_point[1]          = desc->wp.y;
                            mdm->has_primaries           = 1;
                        }
                        mdm->max_luminance = av_d2q(hdr_max_dml, INT_MAX);
                        mdm->min_luminance = av_d2q(hdr_min_dml, INT_MAX);
                        mdm->has_luminance = 1;
                    }

                    if (hdr_max_cll > 0 || hdr_max_fall > 0) {
                        if (auto* raw = add_side_data(AV_FRAME_DATA_CONTENT_LIGHT_LEVEL,
                                                      sizeof(AVContentLightMetadata))) {
                            auto* cll    = reinterpret_cast<AVContentLightMetadata*>(raw);
                            cll->MaxCLL  = static_cast<unsigned>(hdr_max_cll);
                            cll->MaxFALL = static_cast<unsigned>(hdr_max_fall);
                        }
                    }

                    CASPAR_LOG(info) << L"[ffmpeg] HDR10 static metadata: mastering display "
                                     << hdr_max_dml << L"/" << hdr_min_dml << L" cd/m2, MaxCLL "
                                     << hdr_max_cll << L", MaxFALL " << hdr_max_fall;
                } else if (!max_dml.empty()) {
                    CASPAR_LOG(warning)
                        << L"[ffmpeg] <hdr-metadata> ignored: the channel's transfer is not PQ or "
                           L"HLG, and mastering display metadata alongside an SDR stream would be "
                           L"a contradiction rather than extra information.";
                }
            }
        } else if (codec->type == AVMEDIA_TYPE_AUDIO) {
            st->time_base = {1, av_buffersink_get_sample_rate(sink)};

            enc->sample_fmt  = static_cast<AVSampleFormat>(av_buffersink_get_format(sink));
            enc->sample_rate = av_buffersink_get_sample_rate(sink);
            enc->time_base   = st->time_base;

            FF(av_buffersink_get_ch_layout(sink, &enc->ch_layout));

        } else {
            // TODO
        }

        if (realtime && codec->capabilities & AV_CODEC_CAP_SLICE_THREADS) {
            enc->thread_type = FF_THREAD_SLICE;
        }

        if (oc->oformat->flags & AVFMT_GLOBALHEADER) {
            enc->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
        }

        // Tell the encoder the frames are field-coded. Set here rather than left to the caller
        // because `-flags +ildct` on the AMCP line does not reach the encoder at all -- it is
        // reported as an unused option -- so an operator has no way to ask for this, and a
        // paired frame written without the flag is a combed picture labelled progressive.
        //
        // ILDCT is the one that matters for the intra codecs recorded here (it selects the
        // interlaced scan and halves the DCT's vertical extent). ILME is set alongside it for
        // the inter codecs, where it is what makes motion estimation field-aware; the intra
        // encoders ignore it.
        if (pair_fields_) {
            enc->flags |= AV_CODEC_FLAG_INTERLACED_DCT | AV_CODEC_FLAG_INTERLACED_ME;
        }

        // GPU-direct: the buffersink emits the CUDA frames the buffersrc was given,
        // and NVENC refuses device memory without being told which pool it came
        // from ("hw_frames_ctx must be set when using GPU frames as input").
        if (gpu_frames_ctx) {
            if (auto* sink_frames = av_buffersink_get_hw_frames_ctx(sink))
                enc->hw_frames_ctx = av_buffer_ref(sink_frames);
            else
                enc->hw_frames_ctx = av_buffer_ref(gpu_frames_ctx);
            if (auto* frames = reinterpret_cast<AVHWFramesContext*>(enc->hw_frames_ctx->data))
                enc->hw_device_ctx = av_buffer_ref(frames->device_ref);
        }

        // Only reached when the caller's own filter chain contains cuda filters.
        if (use_nvenc_hw_ && hw_device_ctx_) {
            enc->hw_device_ctx = av_buffer_ref(hw_device_ctx_);
            // A chain ending in hwupload_cuda leaves the buffersink emitting
            // hardware frames, and the encoder then needs the frames context.
            AVBufferRef* frames_ctx = av_buffersink_get_hw_frames_ctx(sink);
            if (frames_ctx) {
                enc->hw_frames_ctx = av_buffer_ref(frames_ctx);
            }
        }

        // `-q:v` IS NOT AN AVOPTION, so passing it through the dict below did nothing at all.
        //
        // libavcodec has `global_quality` and, separately, `qscale` as a NAME FOR A FLAG BIT
        // (`options_table.h`: an `AV_OPT_TYPE_CONST` in the `flags` unit). There is no `q`
        // option and no numeric `qscale` option -- the translation from `-q:v N` to
        // `global_quality = N * FF_QP2LAMBDA` plus `AV_CODEC_FLAG_QSCALE` lives in the ffmpeg
        // CLI, which this consumer is not. So `-q:v 4` in a consumer's args was parsed,
        // forwarded, refused by `avcodec_open2` as unknown, and copied back into the leftovers
        // map where nothing looked at it: a quality setting that returned 202 and changed
        // nothing.
        //
        // Measured 2026-08-23, before this: `prores422hq_vulkan_q` (`-q:v 4`) produced a
        // 375 MB file and `prores422hq_vulkan` (no quantiser) 370 MB, from the same ten seconds
        // of the same source. A forced quantiser of 4 cannot come out the same size as rate
        // control, which is what said the option was inert.
        //
        // What it is worth on the Vulkan ProRes encoder is not marginal. `prores_ks_vulkan`
        // runs Kostya's trellis quantiser search as a compute shader, and a non-zero
        // `force_quant` -- which is `global_quality / FF_QP2LAMBDA`, straight out of
        // `proresenc_kostya_common.c` -- sets the specialisation constant that bypasses it
        // (`proresenc_kostya_vulkan.c`, the `trellis_node` pipeline). Measured 2026-08-22
        // outside CasparCG, 500 frames of 1080p: 11 s at the default, 4 s with `-q:v 4`, and
        // 4 s is what the RGBA64->yuv422p10 conversion costs on its own.
        //
        // NOTE THE UNITS. `global_quality` is in lambda, so an operator writing
        // `-global_quality 4` directly is asking for 4/118 of a quantiser step, which rounds to
        // the same zero that means "use rate control". That one is passed through untouched
        // because it is a real option and the value is the operator's to choose; only `q` and
        // `qscale` are translated, and they are erased from the dict so `avcodec_open2` does not
        // see a name it will refuse.
        if (codec->type == AVMEDIA_TYPE_VIDEO) {
            for (const auto* key : {"q", "qscale"}) {
                const auto it = stream_options.find(key);
                if (it == stream_options.end())
                    continue;
                try {
                    const auto qv = std::stod(it->second);
                    enc->global_quality = static_cast<int>(qv * FF_QP2LAMBDA);
                    enc->flags |= AV_CODEC_FLAG_QSCALE;
                    CASPAR_LOG(info)
                        << L"[ffmpeg] fixed-quantiser encoding: -" << u16(key) << L" " << qv
                        << L" becomes global_quality=" << enc->global_quality
                        << L" with AV_CODEC_FLAG_QSCALE. The ffmpeg CLI does this conversion "
                           L"for you; nothing here did, so the option used to be discarded.";
                } catch (...) {
                    CASPAR_LOG(warning) << L"[ffmpeg] -" << u16(key) << L" " << u16(it->second)
                                        << L" is not a number, so it is ignored.";
                }
                stream_options.erase(it);
                break;
            }
        }

        auto dict = to_dict(std::move(stream_options));
        CASPAR_SCOPE_EXIT { av_dict_free(&dict); };
        FF(avcodec_open2(enc.get(), codec, &dict));
        for (auto& p : to_map(&dict)) {
            options[p.first] = p.second + suffix;
        }

        FF(avcodec_parameters_from_context(st->codecpar, enc.get()));

        if (codec->type == AVMEDIA_TYPE_AUDIO && !(codec->capabilities & AV_CODEC_CAP_VARIABLE_FRAME_SIZE)) {
            av_buffersink_set_frame_size(sink, enc->frame_size);
        }

        // Release the local hw_device_ctx ref (graph and enc hold their own refs)
        if (hw_device_ctx_) {
            av_buffer_unref(&hw_device_ctx_);
        }
    }

    /// Read the one plane of an `AVVkFrame` into the exporter's pure-std description.
    ///
    /// `AVFrame::data[0]` on a Vulkan frame is an `AVVkFrame*`, not pixels. Its `img[0]`,
    /// `sem[0]`, `sem_value[0]` and `layout[0]` are what the copy has to be told: the image to
    /// write, the timeline semaphore to signal, the value to signal past, and the layout to
    /// leave it in so FFmpeg's own bookkeeping stays true.
    ///
    /// Returns false when the frame is not a Vulkan frame or carries no image, which is a
    /// refusal rather than a guess -- writing through a null image would fault inside the
    /// driver with nothing in the log to say why.
    static bool fill_vk_plane_dest(const AVFrame* frame, accelerator::vulkan::av_plane_dest& dest)
    {
#if LIBAVUTIL_VERSION_MAJOR >= 60
        auto* vk = reinterpret_cast<AVVkFrame*>(frame->data[0]);
        if (!vk || !vk->img[0] || !vk->sem[0])
            return false;
        dest.image     = reinterpret_cast<void*>(vk->img[0]);
        dest.semaphore = reinterpret_cast<void*>(vk->sem[0]);
        dest.sem_value = vk->sem_value[0];
        dest.layout    = static_cast<int>(vk->layout[0]);
        return true;
#else
        return false;
#endif
    }

    /// Writes back what the exporter did to the AVVkFrame, so FFmpeg's own submits stay legal.
    static bool store_vk_plane_result(AVFrame* frame, const accelerator::vulkan::av_plane_dest& dest)
    {
#if LIBAVUTIL_VERSION_MAJOR >= 60
        auto* vk = reinterpret_cast<AVVkFrame*>(frame->data[0]);
        if (!vk)
            return false;
        vk->sem_value[0] = dest.signalled_value;
        vk->layout[0]    = static_cast<VkImageLayout>(dest.final_layout);
        return true;
#else
        (void)frame;
        (void)dest;
        return false;
#endif
    }

    /// Why the last GPU-direct frame could not be built. Empty when the last one succeeded.
    ///
    /// THE CALLER USED TO REPORT `gpu_uploader->last_error()` FOR EVERY CAUSE, and on the
    /// Vulkan encode path that object is the CUDA-GL uploader, which has not failed and returns
    /// an empty string. So a real failure logged as "GPU-direct recording failed ()" -- naming
    /// the wrong component and giving no reason at all. Measured 2026-08-22 chasing why
    /// `prores_ks_vulkan` stops at one channel: the message appeared twice, the exporter itself
    /// logged nothing, and there was no way from outside to tell which of five guards tripped.
    std::string gpu_direct_reason_;

    // ── interlaced pairing ──────────────────────────────────────────────────────────────
    // A 1080i5000 channel ticks at FIELD rate: 50 full-height frames a second, and a consumer
    // that wants a field-coded file has to pair them itself. The core used to do this for every
    // consumer -- `draw_frame::interlace()` tagged two frames upper/lower and the OpenGL mixer
    // composited them with GL_POLYGON_STIPPLE -- until e1fffcfa5 (upstream, Feb 2018) deleted
    // the field model and moved the job "into modules". DeckLink got explicit pairing in that
    // same commit; this consumer got nothing, and still carried a `// TODO - field alignment`.
    //
    // What it did instead was declare the input rate as framerate * field_count and write every
    // tick as its own progressive frame. That is not merely unsignalled: with a 25p source on a
    // 50-tick channel it writes each picture TWICE. Measured 2026-08-23 before this change, the
    // first eight video frames of a 1080i5000 recording hashed as four identical pairs -- double
    // the file for the same pictures, tagged progressive.
    bool              pair_fields_ = false;
    bool              tff_         = true;
    core::const_frame pending_field_;
    std::int64_t      pair_pts_ = 0;

    /// Wraps the mixer's composited texture in a frame the encoder can read -- a CUDA frame for
    /// NVENC, a Vulkan frame for the FFmpeg Vulkan encoders. Returns null if the frame carries
    /// no usable texture or the copy fails, leaving the cause in `gpu_direct_reason_`.
    /// Build a GPU frame. `second_field` non-empty pairs the two composites into one
    /// field-coded frame, which only the Vulkan route can do -- see the interlaced branch.
    std::shared_ptr<AVFrame> make_cuda_video_frame(const core::const_frame& in_frame,
                                                  const core::const_frame& second_field = core::const_frame{},
                                                  bool                     a_is_top     = true)
    {
        gpu_direct_reason_.clear();

        // NO TEXTURE MEANS NOTHING WAS COMPOSITED, which is not a failure -- it is what a
        // channel with no producer looks like, and what a `route://` channel looks like for the
        // few seconds before its source plays. The Vulkan mixer attaches a texture to every
        // frame it does composite (`image_mixer.cpp`, `make_result`), so the two cases do not
        // overlap.
        //
        // This used to return null, which the caller counted as a GPU-direct failure and acted
        // on: a recording consumer declared in `casparcg.config` starts BEFORE anything is
        // played, so the path was abandoned on its first frame -- every time, for every such
        // consumer.
        //
        // What that cost was not the recording. `gpu_frames_ctx` stays non-null and this
        // function remains the only way a video frame is built, so once the channel had
        // something to composite the frames were device frames again and the file was written
        // normally. What it cost was the CPU readback that clearing the flag switches back on,
        // for every tick of the rest of the recording, consumed by nobody: 8 MB a frame at
        // 8-bit and 16 at 16-bit, on the channel thread, which is exactly the cost this whole
        // path exists to avoid. Measured 2026-08-23 on a `prores422hq_vulkan` step, the mixer
        // now logs "CPU readback SKIPPED" for the same configuration that used to log "CPU
        // readback required by consumer ffmpeg" four seconds in and never recover.
        //
        // Encoding black is what the host path records for the same frame, so this is parity
        // rather than a substitution -- and the frames before PLAY are now in the file instead
        // of missing from it.
        auto tex = in_frame.texture();

        auto frame = alloc_frame();
        const int got = av_hwframe_get_buffer(gpu_frames_ctx, frame.get(), 0);
        if (got < 0) {
            // POOL EXHAUSTION IS THE INTERESTING CASE, and it is transient by nature: the pool
            // is fixed at its initial size, the encoder holds frames while it works, and an
            // encoder momentarily behind empties it. Reported distinctly from a genuine
            // allocation failure because the two want opposite responses -- wait, or give up.
            char err[AV_ERROR_MAX_STRING_SIZE]{};
            av_strerror(got, err, sizeof(err));
            gpu_direct_reason_ = std::string("no frame available from the hardware pool (")
                                 + err + ") -- the encoder is holding all of them";
            return nullptr;
        }

        // The CUDA route has no interleave: cuda_gl_uploader/cuda_vk_uploader copy a whole
        // texture into a CUDA frame and neither takes a line stride. Refusing is not a
        // formality -- falling through would copy field A over the whole frame and then field B
        // over the whole frame, so the recording would be field B alone at half rate, which
        // looks like a working interlaced file.
        if (second_field && !vk_exporter) {
            gpu_direct_reason_ = "this route cannot interleave two fields (CUDA has no strided "
                                 "image copy here); the host path pairs them instead";
            return nullptr;
        }

        if (!tex) {
            if (!vk_exporter) {
                // The CUDA/NVENC path has no equivalent clear, so an idle channel still costs
                // it the frame. Named rather than left as an empty reason, and NOT counted as
                // a reason to abandon the path -- see the caller.
                gpu_direct_reason_ = "the channel composited nothing (idle) and this path "
                                     "cannot encode black without the mixer";
                return nullptr;
            }
            accelerator::vulkan::av_plane_dest dest;
            dest.image     = reinterpret_cast<void*>(frame->data[0]);
            dest.semaphore = nullptr;
            dest.width     = frame->width;
            dest.height    = frame->height;
            if (!fill_vk_plane_dest(frame.get(), dest)) {
                gpu_direct_reason_ = "the AVVkFrame was incompletely described";
                return nullptr;
            }
            if (!vk_exporter->clear_to_black(dest)) {
                gpu_direct_reason_ = "the black fill was refused (see [vk::av_export])";
                return nullptr;
            }
            if (!store_vk_plane_result(frame.get(), dest)) {
                gpu_direct_reason_ = "the AVVkFrame result could not be written back";
                return nullptr;
            }
            return frame;
        }

        if (auto* gl_tex = dynamic_cast<accelerator::ogl::texture*>(tex.get())) {
            if (!gpu_uploader) {
                gpu_direct_reason_ = "no CUDA-GL uploader on this channel";
                return nullptr;
            }
            auto dev = gl_tex->get_device();
            if (!dev) {
                gpu_direct_reason_ = "the OpenGL texture has no device";
                return nullptr;
            }

            // On the mixer's own GL thread: CUDA registers GL objects against the
            // calling thread's context, and dispatching here avoids standing up a
            // second shared context (see cuda_gl_upload.h).
            const bool ok = dev->dispatch_sync([&] {
                return gpu_uploader->copy_to_device(*gl_tex, frame->data[0], static_cast<size_t>(frame->linesize[0]));
            });
            if (!ok)
                return nullptr;
        } else if (vk_exporter) {
            // The Vulkan encode path: no cross-API import at all. FFmpeg's frames live on the
            // mixer's own VkDevice (`make_vulkan_hwdevice_from_mixer` hands out one context per
            // device), so this is an image copy the mixer's own queue performs.
            //
            // The frame is BGRA here and the encoder wants planar YUV; the conversion is a
            // `libplacebo` filter in the chain, not this copy's business. ProRes accepts only
            // yuv422p10/yuv444p10/yuva444p10, and `scale_vulkan` cannot produce any of them.
            accelerator::vulkan::av_plane_dest dest;
            dest.image     = reinterpret_cast<void*>(frame->data[0]);
            dest.semaphore = nullptr;
            dest.width     = tex->tex_width();
            dest.height    = tex->tex_height();
            if (!fill_vk_plane_dest(frame.get(), dest)) {
                gpu_direct_reason_ = "the AVVkFrame was incompletely described";
                return nullptr;
            }
            const bool copied =
                second_field ? vk_exporter->copy_from_textures(tex, second_field.texture(),
                                                              a_is_top, dest)
                             : vk_exporter->copy_from_texture(tex, dest);
            if (!copied) {
                gpu_direct_reason_ = "the Vulkan image copy was refused (see [vk::av_export])";
                return nullptr;
            }
            // The exporter signalled the frame's timeline semaphore and moved its image, so
            // FFmpeg's own bookkeeping has to learn both -- see `av_plane_dest`, whose comment
            // carries the validation-layer finding this fixes.
            if (!store_vk_plane_result(frame.get(), dest)) {
                gpu_direct_reason_ = "the AVVkFrame result could not be written back";
                return nullptr;
            }
        } else if (gpu_uploader_vk) {
            // Vulkan: no device thread to hop onto and no context to be current --
            // the image's exported memory is imported straight into CUDA.
            if (!gpu_uploader_vk->copy_to_device(tex, frame->data[0], static_cast<size_t>(frame->linesize[0])))
                return nullptr;
        } else {
            return nullptr;
        }

        frame->width  = tex->tex_width();
        frame->height = tex->tex_height();

        // The interlaced flags have to be set HERE for the GPU path, because this frame came
        // from av_hwframe_get_buffer rather than from make_av_video_frame, which is where the
        // host path sets them. Without them the encoder still coded interlaced -- the codec
        // flag says so -- but signalled no dominance, and the file came out labelled BOTTOM
        // field first while the line interleave had put field A on the TOP lines. Measured
        // 2026-08-23: field_order=bb from a top-first pairing, which a player would show with
        // the two fields in the wrong temporal order.
        if (second_field) {
            frame->flags |= AV_FRAME_FLAG_INTERLACED;
            if (a_is_top)
                frame->flags |= AV_FRAME_FLAG_TOP_FIELD_FIRST;
        }
        return frame;
    }

    void send(std::tuple<core::const_frame, std::int64_t, std::int64_t, core::video_field>& data,
              const core::video_format_desc&                             format_desc,
              std::function<void(std::shared_ptr<AVPacket>)>             cb)
    {
        std::shared_ptr<AVFrame>  frame;
        std::shared_ptr<AVPacket> pkt;

        const auto [in_frame, video_pts, audio_pts, field] = data;

        if (is_ltc) {
             if (!in_frame) return;

             pkt = alloc_packet();
             // 4 bytes payload, big endian frame count typically for TMCD
             FF(av_new_packet(pkt.get(), 4));
             pkt->stream_index = st->index;
             pkt->pts = video_pts;
             pkt->dts = video_pts;
             pkt->duration = 1;
             
             // Get current LTC frame number or extrapolate
             uint32_t current_fn = caspar::ltc::LTCInput::instance().get_current_frame_number(25); // assuming 25?
             
             // QuickTime tmcd uses Big Endian u32 frame count
             pkt->data[0] = (current_fn >> 24) & 0xFF;
             pkt->data[1] = (current_fn >> 16) & 0xFF;
             pkt->data[2] = (current_fn >> 8) & 0xFF;
             pkt->data[3] = (current_fn) & 0xFF;

             cb(std::move(pkt));
             return;
        }

        // The pts a paired frame carries is its PAIR index, not the tick index: pairing halves
        // the output rate, and the buffersrc is parameterised to match (see the constructor).
        std::int64_t out_video_pts = video_pts;

        if (in_frame) {
            if (enc->codec_type == AVMEDIA_TYPE_VIDEO) {
                // Hold field A, emit on field B. The AUDIO stream is deliberately untouched and
                // keeps consuming every tick -- pairing lives here, per stream, precisely so
                // that both fields' audio is still written, which it already was.
                if (pair_fields_) {
                    if (field != core::video_field::b) {
                        pending_field_ = in_frame;
                        return;
                    }
                    if (!pending_field_) {
                        // B with no preceding A. Guessing -- pairing it with itself, or emitting
                        // it alone -- would put a wrong parity or a half-rate hole in the file.
                        CASPAR_LOG(warning)
                            << L"[ffmpeg] interlaced recording saw field B with no field A; "
                               L"dropping it";
                        return;
                    }
                    frame = gpu_frames_ctx
                                ? make_cuda_video_frame(pending_field_, in_frame, tff_)
                                : make_av_video_frame(pending_field_, format_desc, in_frame, tff_);
                    pending_field_ = core::const_frame{};
                    out_video_pts  = pair_pts_++;
                    if (!frame) {
                        // Same accounting as the progressive GPU miss below: the frame is lost,
                        // and the reason is already in gpu_direct_reason_.
                        gpu_failures++;
                        return;
                    }
                } else if (gpu_frames_ctx) {
                    frame = make_cuda_video_frame(in_frame);
                    if (!frame) {
                        // ONE MISS IS NOT A BROKEN PATH, and this used to abandon the path
                        // on the FIRST one. Two of the causes are transient by nature: the
                        // hardware frame pool momentarily empty because the encoder holds
                        // every frame in it, and -- until the black fill above -- a channel
                        // that had not started playing yet.
                        //
                        // THE FRAME IS STILL LOST, and clearing the flag does not save it.
                        // `gpu_frames_ctx` is non-null for the whole recording and this
                        // function is the only way a video frame is built, so there is no host
                        // path to fall back to -- the old message promising one was false. All
                        // clearing the flag does is make the channel resume a CPU readback
                        // that nothing here consumes, which is a cost rather than a fallback.
                        //
                        // It is kept only as a last resort, after enough consecutive misses
                        // that the path is evidently broken rather than busy, and the message
                        // now says what it really means.
                        ++gpu_failures;
                        const auto reason = gpu_direct_reason_.empty()
                                                ? std::string("no reason recorded")
                                                : gpu_direct_reason_;
                        if (reason != gpu_last_logged_reason) {
                            gpu_last_logged_reason = reason;
                            CASPAR_LOG(warning)
                                << L"[ffmpeg] GPU-direct recording dropped a frame (" << gpu_failures
                                << L" so far): " << u16(reason) << L".";
                        }
                        if (gpu_failures == kGpuFailuresBeforeGivingUp && gpu_direct) {
                            gpu_direct->store(false, std::memory_order_relaxed);
                            CASPAR_LOG(error)
                                << L"[ffmpeg] GPU-direct recording has failed "
                                << kGpuFailuresBeforeGivingUp << L" times (" << u16(reason)
                                << L"). This encoder takes only device frames, so the frames "
                                   L"are being LOST rather than recorded another way -- the "
                                   L"recording will be short by every one of them.";
                        }
                        return;
                    }
                } else {
                    frame = make_av_video_frame(in_frame, format_desc);
                }
                frame->pts = out_video_pts;
            } else if (enc->codec_type == AVMEDIA_TYPE_AUDIO) {
                frame      = make_av_audio_frame(in_frame, format_desc);
                frame->pts = audio_pts;
            } else {
                // TODO
            }
            FF(av_buffersrc_write_frame(source, frame.get()));
        } else {
            FF(av_buffersrc_close(source, AV_NOPTS_VALUE, 0));
        }

        while (true) {
            pkt     = alloc_packet();
            int ret = avcodec_receive_packet(enc.get(), pkt.get());

            if (ret == AVERROR(EAGAIN)) {
                frame = alloc_frame();
                ret   = av_buffersink_get_frame(sink, frame.get());
                if (ret == AVERROR(EAGAIN)) {
                    return;
                }
                if (ret == AVERROR_EOF) {
                    FF(avcodec_send_frame(enc.get(), nullptr));
                } else {
                    FF_RET(ret, "av_buffersink_get_frame");
                    FF(avcodec_send_frame(enc.get(), frame.get()));
                }
            } else if (ret == AVERROR_EOF) {
                return;
            } else {
                FF_RET(ret, "avcodec_receive_packet");
                pkt->stream_index = st->index;
                av_packet_rescale_ts(pkt.get(), enc->time_base, st->time_base);
                cb(std::move(pkt));
            }
        }
    }
};

struct ffmpeg_consumer : public core::frame_consumer
{
    core::monitor::state    state_;
    mutable std::mutex      state_mutex_;
    int                     channel_index_ = -1;
    core::video_format_desc format_desc_;
    bool                    realtime_ = false;
    std::int64_t            video_pts = 0;
    std::int64_t            audio_pts = 0;

    spl::shared_ptr<diagnostics::graph> graph_;

    std::string path_;
    std::string args_;

    std::exception_ptr exception_;
    std::mutex         exception_mutex_;

    // Satisfied once every stream is open and the header is written, or with the
    // exception that stopped that happening. initialize() waits on it so that a
    // consumer which can never encode a frame is reported at ADD time rather
    // than answering 202 and failing silently. See initialize().
    std::promise<void> open_result_;
    bool               open_result_set_ = false;

    tbb::concurrent_bounded_queue<std::tuple<core::const_frame, std::int64_t, std::int64_t, core::video_field>> frame_buffer_;
    std::thread                                                                              frame_thread_;

    common::bit_depth depth_;
    bool              use_vulkan_ = false;

    // ── GPU-direct recording ──────────────────────────────────────────────
    // When the encoder is NVENC and nothing in the chain needs host pixels, the
    // composited texture is copied straight into a CUDA frame and handed to the
    // encoder. Otherwise the channel reads the frame back and NVENC uploads the
    // same pixels again: measured 2.95 ms down at 1080p and 11.50 ms at 4K, plus
    // the upload, for a round trip that produces nothing.
    //
    // Decided once, at construction, because the encoder and the filter graph are
    // configured for one or the other and cannot change per frame. If the interop
    // later fails, gpu_direct_ goes false; needs_cpu_frame_data() is polled every
    // tick, so the channel resumes readbacks from the next one.
    std::atomic<bool>            gpu_direct_{false};
    cuda_gl_uploader             gpu_uploader;
    cuda_vk_uploader             gpu_uploader_vk;

    using av_buffer_ptr = std::unique_ptr<AVBufferRef, void (*)(AVBufferRef*)>;
    static av_buffer_ptr null_av_buffer() { return {nullptr, [](AVBufferRef* p) { av_buffer_unref(&p); }}; }
    av_buffer_ptr gpu_device_ctx = null_av_buffer();
    av_buffer_ptr gpu_frames_ctx = null_av_buffer();
    /// Which hardware format `gpu_frames_ctx` holds. CUDA unless the Vulkan encode path claimed
    /// this recording.
    AVPixelFormat gpu_hw_pix_fmt_ = AV_PIX_FMT_CUDA;
    /// Built lazily on the first frame, because the mixer's `device*` is only reachable through
    /// a composited texture -- the consumer is never handed one directly.
#ifdef ENABLE_VULKAN
    std::unique_ptr<accelerator::vulkan::av_vulkan_exporter> vk_exporter_;
#endif
    /// The mixer's `accelerator::vulkan::device*`, from `channel_info`. Null on any other
    /// backend, and the Vulkan encode path declines rather than assuming one.
    void* vk_mixer_device_ = nullptr;
    /// The GPU conversion filter the Vulkan encode path chose, or null. Held rather than pushed
    /// into the options map, which silently dropped it.
    const char* vk_convert_filter_ = nullptr;

    /// A CUDA frames pool the encoder can read directly.
    ///
    /// The copy out of the mixer's target is byte-for-byte, so sw_format has to name
    /// the byte order the mixer actually wrote -- and the two mixers do not agree.
    /// The OpenGL target is R,G,B,A, so RGB0. The Vulkan target is B,G,R,A despite its
    /// VkFormat being eR8G8B8A8Unorm: the format names the storage, the shader decides
    /// what goes in it, and decklink's cuda_vk_strategy has always read that same
    /// attachment as BGRA. Declaring RGB0 for it swaps red and blue -- which is exactly
    /// what the first Vulkan recording did, and the kind of thing that looks like a
    /// colour-management problem rather than a byte-order one.
    ///
    /// NVENC lists both among its inputs and converts to YCbCr internally, which
    /// measured *better* than converting on the host beforehand.
    /// A Vulkan frames pool on the MIXER's device, for an FFmpeg Vulkan encoder.
    ///
    /// `make_vulkan_hwdevice_from_mixer` is the same function the producers use and it caches
    /// one context per `VkDevice` -- deliberately, because a context per user means a queue
    /// mutex per user guarding one queue, which is what lost the device at four producers. The
    /// consumer must therefore take that one rather than create its own.
    ///
    /// `sw_format` is BGRA because that is what the 8-bit mixer composites and what the exporter
    /// copies; the encoder's own format is reached by the conversion filter downstream.
    av_buffer_ptr make_vulkan_frames_ctx(int width, int height)
    {
        auto fail = null_av_buffer();
#if defined(ENABLE_VULKAN) && LIBAVUTIL_VERSION_MAJOR >= 60
        if (!vk_mixer_device_)
            return fail;
        AVBufferRef* dev = make_vulkan_hwdevice_from_mixer(vk_mixer_device_);
        if (!dev)
            return fail;
        gpu_device_ctx = av_buffer_ptr(dev, [](AVBufferRef* p) { av_buffer_unref(&p); });

        AVBufferRef* frames = av_hwframe_ctx_alloc(gpu_device_ctx.get());
        if (!frames)
            return fail;
        auto owned = av_buffer_ptr(frames, [](AVBufferRef* p) { av_buffer_unref(&p); });

        auto* ctx      = reinterpret_cast<AVHWFramesContext*>(owned->data);
        ctx->format = AV_PIX_FMT_VULKAN;
        // RGBA64, AND THE CHANNEL MUST THEREFORE BE 16-BIT. Not a precision preference -- a
        // channel-order one, measured 2026-08-22 outside CasparCG entirely:
        //
        //   format=rgba64,hwupload,libplacebo=format=yuv422p10   red stays red
        //   format=rgba,  hwupload,libplacebo=format=yuv422p10   red stays red
        //   format=bgra,  hwupload,libplacebo=format=yuv422p10   RED AND BLUE EXCHANGED
        //
        // `hwupload`/`hwdownload` round-trip BGRA correctly, and `scale_vulkan` reads it
        // correctly, so the exchange is `vf_libplacebo` applying the pixel descriptor's
        // component order on top of the one VK_FORMAT_B8G8R8A8_UNORM already performs. An
        // upstream defect, and one this path must not depend on the sign of.
        //
        // The mixer's 8-bit attachment holds BGRA bytes -- the shader writes `col.bgra` into an
        // eR8G8B8A8Unorm image -- and its 16-bit attachment holds RGBA, because
        // `image_kernel.cpp` sets `output_bgra` only at bit8 (there is no eB16G16R16A16Unorm to
        // swizzle into). So the 16-bit composite is the one that already matches a format
        // libplacebo handles, and taking it needs no swizzle, no scratch image, and no reliance
        // on two exchanges cancelling.
        ctx->sw_format = AV_PIX_FMT_RGBA64;
        ctx->width     = width;
        ctx->height    = height;
        // Same reasoning as the CUDA pool: deep enough that an encoder holding a few frames
        // never starves the copy, shallow enough not to sit on VRAM.
        ctx->initial_pool_size = 8;

        if (av_hwframe_ctx_init(owned.get()) < 0)
            return fail;
        return owned;
#else
        (void)width;
        (void)height;
        return fail;
#endif
    }

    av_buffer_ptr make_cuda_frames_ctx(int width, int height)
    {
        auto fail = null_av_buffer();

        AVBufferRef* dev = nullptr;
        if (av_hwdevice_ctx_create(&dev, AV_HWDEVICE_TYPE_CUDA, nullptr, nullptr, 0) < 0)
            return fail;
        gpu_device_ctx = av_buffer_ptr(dev, [](AVBufferRef* p) { av_buffer_unref(&p); });

        AVBufferRef* frames = av_hwframe_ctx_alloc(gpu_device_ctx.get());
        if (!frames)
            return fail;
        auto owned = av_buffer_ptr(frames, [](AVBufferRef* p) { av_buffer_unref(&p); });

        auto* ctx     = reinterpret_cast<AVHWFramesContext*>(owned->data);
        ctx->format    = AV_PIX_FMT_CUDA;
        ctx->sw_format = use_vulkan_ ? AV_PIX_FMT_BGR0 : AV_PIX_FMT_RGB0;
        ctx->width     = width;
        ctx->height    = height;
        // Deep enough that the encoder holding a couple of frames never starves
        // the copy, shallow enough not to sit on VRAM: 4K RGB0 is 33 MB a frame.
        ctx->initial_pool_size = 8;

        if (av_hwframe_ctx_init(owned.get()) < 0)
            return fail;

        // The copies must run in the same CUDA context the encoder's frames were
        // allocated from -- device pointers are not valid across contexts, and
        // using one from the wrong context kills the process outright with no
        // exception to catch. Asking FFmpeg for the primary context instead is
        // not an option: the DeckLink DVP and CUDA ProRes modules have already
        // activated it by now, and FFmpeg refuses with "Primary context already
        // active with incompatible flags".
        // AVCUDADeviceContext comes from the header guarded above. Without CUDA the
        // uploaders are stubs and av_hwdevice_ctx_create(AV_HWDEVICE_TYPE_CUDA) at the top
        // of this function has already failed, so there is nothing to hand a context to.
#ifdef CASPAR_FFMPEG_HAS_CUDA
        auto* hw_dev  = reinterpret_cast<AVHWDeviceContext*>(gpu_device_ctx->data);
        auto* cuda_hw = static_cast<AVCUDADeviceContext*>(hw_dev->hwctx);
        gpu_uploader.set_context(cuda_hw->cuda_ctx);
        gpu_uploader_vk.set_context(cuda_hw->cuda_ctx);
#endif

        return owned;
    }

    /// `-interlaced`: -1 auto (pair when the channel is interlaced), 0 never, 1 always.
    /// Read from the AMCP options in the frame thread, before the streams are constructed.
    int field_mode_request = -1;

    // FPS counter
    std::chrono::steady_clock::time_point last_fps_update_;
    int                     frames_since_update_ = 0;
    double                  current_fps_ = 0.0;

  public:
    ffmpeg_consumer(std::string path, std::string args, bool realtime, common::bit_depth depth, bool use_vulkan)
        : channel_index_([&] {
            boost::crc_16_type result;
            result.process_bytes(path.data(), path.length());
            return result.checksum();
        }())
        , realtime_(realtime)
        , path_(std::move(path))
        , args_(std::move(args))
        , depth_(depth)
        , use_vulkan_(use_vulkan)
    {
        state_["file/path"] = u8(path_);

        frame_buffer_.set_capacity(realtime_ ? 1 : 64);

        diagnostics::register_graph(graph_);
        graph_->set_color("frame-time", diagnostics::color(0.1f, 1.0f, 0.1f));
        graph_->set_color("dropped-frame", diagnostics::color(0.3f, 0.6f, 0.3f));
        graph_->set_color("input", diagnostics::color(0.7f, 0.4f, 0.4f));
    }

    ~ffmpeg_consumer()
    {
        if (frame_thread_.joinable()) {
            frame_buffer_.push({core::const_frame{}, -1, -1, core::video_field::progressive});
            frame_thread_.join();
        }

        // Before the CUDA contexts below are released. The uploaders' resources --
        // the OpenGL one's registrations, the Vulkan one's external-memory imports --
        // have to be undone while the context they were made in is still alive, and
        // the GL ones additionally on the mixer's GL thread. Leaving it to member
        // destruction order does neither, and the process dies at teardown.
        gpu_uploader.release();
        gpu_uploader_vk.release();
    }

    // frame consumer

    void initialize(const core::video_format_desc& format_desc,
                    const core::channel_info&      channel_info,
                    int                            port_index) override
    {
        if (frame_thread_.joinable()) {
            CASPAR_THROW_EXCEPTION(invalid_operation() << msg_info("Cannot reinitialize ffmpeg-consumer."));
        }

        format_desc_   = format_desc;
        channel_index_ = channel_info.index;
        vk_mixer_device_ = channel_info.vk_device;

        graph_->set_text(print());

        frame_thread_ = std::thread([=, this] {
            caspar::set_thread_name(L"ffmpeg_consumer_frame_thread");
            caspar::set_thread_realtime_priority();
            try {
                std::map<std::string, std::string> options;
                {
                    static boost::regex opt_exp("-(?<NAME>[^\\s]+)(\\s+(?<VALUE>[^\\s]+))?");
                    for (auto it = boost::sregex_iterator(args_.begin(), args_.end(), opt_exp);
                         it != boost::sregex_iterator();
                         ++it) {
                        options[(*it)["NAME"].str().c_str()] =
                            (*it)["VALUE"].matched ? (*it)["VALUE"].str().c_str() : "";
                    }
                }

                boost::filesystem::path full_path = path_;

                static boost::regex prot_exp("^.+:.*");
                if (!boost::regex_match(path_, prot_exp)) {
                    if (!full_path.is_absolute()) {
                        full_path = u8(env::media_folder()) + path_;
                    }

                    // TODO -y?
                    if (boost::filesystem::exists(full_path)) {
                        // Retry removal — on Windows the OS may briefly hold
                        // the file handle after the previous consumer closed it.
                        for (int attempt = 0; attempt < 10; ++attempt) {
                            boost::system::error_code ec;
                            boost::filesystem::remove(full_path, ec);
                            if (!ec)
                                break;
                            std::this_thread::sleep_for(std::chrono::milliseconds(100));
                        }
                    }

                    boost::filesystem::create_directories(full_path.parent_path());
                }

                AVFormatContext* oc = nullptr;

                {
                    std::string format;
                    {
                        const auto format_it = options.find("format");
                        if (format_it != options.end()) {
                            format = std::move(format_it->second);
                            options.erase(format_it);
                        }
                    }

                    FF(avformat_alloc_output_context2(
                        &oc, nullptr, !format.empty() ? format.c_str() : nullptr, path_.c_str()));
                }
                
                // LTC Metadata Injection
                if (options.count("ltc") > 0 || options.count("TIMECODE_SOURCE") > 0) {
                     std::string tc = caspar::ltc::LTCInput::instance().get_current_timecode_string();
                     av_dict_set(&oc->metadata, "timecode", tc.c_str(), 0);
                     CASPAR_LOG(info) << "Injected Start Timecode: " << tc;
                }

                CASPAR_SCOPE_EXIT { avformat_free_context(oc); };

                std::optional<Stream> video_stream;
                if (oc->oformat->video_codec != AV_CODEC_ID_NONE) {
                    if (oc->oformat->video_codec == AV_CODEC_ID_H264 && options.find("preset:v") == options.end()) {
                        // "veryfast" is an x264/x265 preset name and means nothing to
                        // any other encoder. NVENC rejects it outright ("Undefined
                        // constant ... in 'veryfast'") and the whole recording fails;
                        // ProRes merely warns. Both used to happen, for two reasons:
                        //
                        //  - The condition tests the *container's* default codec, and
                        //    mov/mp4 both default to H.264 -- so the preset was injected
                        //    even for "-vcodec prores_ks".
                        //  - The NVENC exemption looked only at "codec:v" and "c:v", but
                        //    the encoder is selected from "vcodec" as well (see the Stream
                        //    constructor). "ADD 1 FILE out.mp4 -vcodec h264_nvenc" -- the
                        //    documented spelling -- therefore got veryfast applied and
                        //    never recorded anything.
                        //
                        // So key it off what it is really about: inject the preset only
                        // when the encoder in use is x264/x265, meaning either no explicit
                        // override (the container default resolves to libx264) or an
                        // override naming one of them.
                        const auto selected = [&]() -> std::string {
                            for (const auto* key : {"vcodec", "codec:v", "c:v"}) {
                                const auto it = options.find(key);
                                if (it != options.end())
                                    return it->second;
                            }
                            return {};
                        }();
                        if (selected.empty() || selected.find("x264") != std::string::npos ||
                            selected.find("x265") != std::string::npos) {
                            options["preset:v"] = "veryfast";
                        }
                    }
                    // ── Decide the GPU-direct path ────────────────────────
                    // Everything here has to hold, and each for a concrete reason:
                    //   NVENC          -- it is the only encoder that takes device
                    //                     memory here, and it accepts RGB so no
                    //                     colour conversion kernel is needed.
                    //   no user filter -- lavfi filters operate on host frames.
                    //   8-bit channel  -- a 16-bit channel's texture is RGBA16 and
                    //                     NVENC's RGB inputs are 8-bit.
                    //   either mixer   -- OpenGL registers the mixer texture with
                    //                     CUDA, Vulkan imports the attachment's
                    //                     exported memory. This used to decline on
                    //                     Vulkan claiming the composition target was
                    //                     not exportable; create_attachment allocates
                    //                     with ExportMemoryAllocateInfo and decklink
                    //                     had been importing it all along.
                    //   CUDA present   -- obviously.
                    // Anything false leaves the existing host path untouched.
                    {
                        const auto selected_codec = [&]() -> std::string {
                            for (const auto* key : {"vcodec", "codec:v", "c:v"}) {
                                const auto it = options.find(key);
                                if (it != options.end())
                                    return it->second;
                            }
                            return {};
                        }();
                        const bool has_filter = options.count("filter:v") > 0 || options.count("vf") > 0;

                        // `-interlaced auto|0|1`. Consumed here rather than passed through:
                        // it is a CasparCG-side decision about how two channel ticks become one
                        // frame, and anything left in `options` is handed to FFmpeg and then
                        // reported as an unused option.
                        for (const auto* key : {"interlaced", "interlaced:v"}) {
                            const auto it = options.find(key);
                            if (it == options.end())
                                continue;
                            const auto v = boost::to_lower_copy(it->second);
                            if (v == "auto" || v.empty())
                                field_mode_request = -1;
                            else if (v == "0" || v == "false" || v == "off" || v == "progressive")
                                field_mode_request = 0;
                            else if (v == "1" || v == "true" || v == "on" || v == "interlaced")
                                field_mode_request = 1;
                            else
                                CASPAR_LOG(warning)
                                    << L"[ffmpeg] -interlaced '" << u16(it->second)
                                    << L"' is not auto, 0 or 1 -- using auto";
                            options.erase(it);
                            break;
                        }
                        const bool will_pair = format_desc.field_count == 2 && field_mode_request != 0;
                        // An explicit pixel format is a request the GPU path cannot
                        // honour: its frames are CUDA/RGB0 and lavfi cannot convert
                        // device frames to an arbitrary host format. Without this,
                        // "-vcodec hevc_nvenc -pix_fmt p010le" configured the sink for
                        // p010le, graph configuration failed with "Impossible to convert
                        // between the formats supported by the filter 'Parsed_null_0'"
                        // and the recording produced no file at all.
                        const bool has_pix_fmt = options.count("pix_fmt") > 0 || options.count("pix_fmt:v") > 0 ||
                                                 options.count("pixel_format") > 0 ||
                                                 options.count("pixel_format:v") > 0;

                        // ── The Vulkan encode path ──────────────────────────────────────
                        //
                        // Tried first, because `-vcodec prores_ks_vulkan` is an explicit request
                        // and must not be quietly answered by the CUDA path instead.
                        //
                        // THE ALLOWLIST IS BY MECHANISM, and it is a refusal rather than a claim
                        // of support -- the same shape as the decode-side one, for the same
                        // reason. `prores_ks_vulkan` and `ffv1_vulkan` are COMPUTE encoders and
                        // need only the compute queue; `h264_vulkan` and `hevc_vulkan` are
                        // `VK_KHR_video_encode` codecs and need the encode queue declared as
                        // `qf[2]`. Omitting an encoder costs a slower recording; accepting one
                        // the device cannot back cost a faulting decode thread last time.
                        //
                        // THE CONVERTER IS `libplacebo` FOR ALL FOUR, and `scale_vulkan` is not
                        // a fallback. Measured 2026-08-22 on a half-red/half-blue still:
                        // `scale_vulkan` gets the channel order right but the LEVELS wrong --
                        // 253 comes back as 172, a limited-range conversion applied to already
                        // limited-range data -- where `libplacebo` returns 229 through the same
                        // probe. It also converts RGB to only NV12/YUV420P/YUV444P, all 8-bit,
                        // so it could never feed ProRes, which takes only
                        // yuv422p10/yuv444p10/yuva444p10.
                        //
                        // Worth naming how `scale_vulkan` came to be in this table: the Phase 0
                        // spike recorded h264/hevc through it as "works" on the strength of a
                        // file being produced and probing as h264. Nothing looked at the picture.
                        //
                        // AV1 is deliberately absent from the table rather than refused by name:
                        // FFmpeg queries the device itself and says "Device does not support
                        // encoding av1!", which is a better message than one written here.
                        const char* vk_convert_filter = nullptr;
                        if (selected_codec == "prores_ks_vulkan" || selected_codec == "ffv1_vulkan")
                            vk_convert_filter = "libplacebo=format=yuv422p10";
                        else if (selected_codec == "h264_vulkan" || selected_codec == "hevc_vulkan")
                            vk_convert_filter = "libplacebo=format=nv12";

                        if (vk_convert_filter != nullptr) {
                            const char* vk_decline = nullptr;
                            if (!use_vulkan_)
                                vk_decline = "the channel does not run the Vulkan mixer";
                            else if (has_filter)
                                vk_decline = "a video filter was supplied, and this path owns the chain";
                            else if (depth_ == common::bit_depth::bit8)
                                vk_decline = "the channel is 8-bit, whose composite is BGRA -- and "
                                             "libplacebo exchanges red and blue on a BGRA Vulkan "
                                             "frame; use <color-depth>16</color-depth>";

                            if (vk_decline == nullptr) {
                                gpu_frames_ctx = make_vulkan_frames_ctx(format_desc.width, format_desc.height);
                                if (!gpu_frames_ctx)
                                    vk_decline = "could not create the Vulkan frames context";
                            }

                            if (vk_decline == nullptr) {
                                gpu_hw_pix_fmt_ = AV_PIX_FMT_VULKAN;
#ifdef ENABLE_VULKAN
                                vk_exporter_ = std::make_unique<accelerator::vulkan::av_vulkan_exporter>(
                                    vk_mixer_device_);
#endif
                                vk_convert_filter_ = vk_convert_filter;
                                gpu_direct_.store(true, std::memory_order_relaxed);
                                CASPAR_LOG(info) << L"[ffmpeg] Vulkan encode: " << u16(selected_codec)
                                                 << L" on the mixer's own device, converting with "
                                                 << u16(vk_convert_filter)
                                                 << L" -- the composite never reaches host memory";
                            } else {
                                CASPAR_LOG(info) << L"[ffmpeg] Vulkan encode not used for "
                                                 << u16(selected_codec) << L": " << u16(vk_decline)
                                                 << L"; the host path will run instead";
                            }
                        }

                        const char* decline = nullptr;
                        if (gpu_frames_ctx)
                            decline = "the Vulkan encode path already claimed this recording";
                        else if (will_pair)
                            decline = "the channel is interlaced and this route cannot interleave "
                                      "two fields -- CUDA has no strided image copy here, so the "
                                      "host path pairs them instead. Pass -interlaced 0 to record "
                                      "field rate progressive on this route instead";
                        else if (selected_codec.find("nvenc") == std::string::npos)
                            decline = "encoder is not NVENC";
                        else if (has_filter)
                            decline = "a video filter was supplied";
                        else if (has_pix_fmt)
                            decline = "an explicit pixel format was requested";
                        else if (depth_ != common::bit_depth::bit8)
                            decline = "channel is not 8-bit";
                        else if (use_vulkan_ ? !cuda_vk_uploader::available() : !cuda_gl_uploader::available())
                            decline = "no usable CUDA device";

                        if (decline == nullptr) {
                            gpu_frames_ctx = make_cuda_frames_ctx(format_desc.width, format_desc.height);
                            if (!gpu_frames_ctx)
                                decline = "could not create the CUDA frames context";
                        }

                        if (decline == nullptr) {
                            gpu_direct_.store(true, std::memory_order_relaxed);
                            CASPAR_LOG(info) << L"[ffmpeg] GPU-direct recording active: the composited texture goes "
                                                L"straight to NVENC, with no readback.";
                        } else if (selected_codec.find("nvenc") != std::string::npos) {
                            CASPAR_LOG(info) << L"[ffmpeg] GPU-direct recording not used: " << u16(decline) << L".";
                        }
                    }

                    // The parameter is a pointer to a forward-declared type, so a null one
                    // compiles with no Vulkan backend present -- which is what `vk_decline`
                    // above would have produced anyway. Without this the unique_ptr member
                    // and its destructor were instantiated and the link failed on
                    // `av_vulkan_exporter::~av_vulkan_exporter`.
#ifdef ENABLE_VULKAN
                    auto* vk_exporter_arg = vk_exporter_.get();
#else
                    class accelerator::vulkan::av_vulkan_exporter* vk_exporter_arg = nullptr;
#endif
                    video_stream.emplace(oc, ":v", oc->oformat->video_codec, format_desc, realtime_, depth_, options, channel_info.default_color_space, channel_info.default_color_transfer, gpu_frames_ctx.get(), &gpu_uploader, &gpu_direct_, &gpu_uploader_vk, gpu_hw_pix_fmt_, vk_exporter_arg, vk_convert_filter_, field_mode_request);

                    {
                        std::lock_guard<std::mutex> lock(state_mutex_);
                        state_["file/fps"] = av_q2d(av_buffersink_get_frame_rate(video_stream->sink));
                    }
                }

                std::optional<Stream> audio_stream;
                if (oc->oformat->audio_codec != AV_CODEC_ID_NONE) {
                    audio_stream.emplace(oc, ":a", oc->oformat->audio_codec, format_desc, realtime_, depth_, options, channel_info.default_color_space, channel_info.default_color_transfer);
                }

                if (!(oc->oformat->flags & AVFMT_NOFILE)) {
                    // TODO (fix) interrupt_cb
                    auto dict = to_dict(std::move(options));
                    CASPAR_SCOPE_EXIT { av_dict_free(&dict); };
                    FF(avio_open2(&oc->pb, full_path.string().c_str(), AVIO_FLAG_WRITE, nullptr, &dict));
                    options = to_map(&dict);
                }

                {
                    auto dict = to_dict(std::move(options));
                    CASPAR_SCOPE_EXIT { av_dict_free(&dict); };
                    FF(avformat_write_header(oc, &dict));
                    options = to_map(&dict);
                }

                // Everything that can fail up front has succeeded: the encoders
                // are open and the container header is written. Release
                // initialize(), which has been waiting to find out.
                if (!open_result_set_) {
                    open_result_set_ = true;
                    open_result_.set_value();
                }

                {
                    for (auto& p : options) {
                        CASPAR_LOG(warning) << print() << " Unused option " << p.first << "=" << p.second;
                    }
                }

                tbb::concurrent_bounded_queue<std::shared_ptr<AVPacket>> packet_buffer;
                packet_buffer.set_capacity(realtime_ ? 1 : 128);
                auto packet_thread = std::thread([&] {
                    try {
                        CASPAR_SCOPE_EXIT
                        {
                            if (!(oc->oformat->flags & AVFMT_NOFILE)) {
                                FF(avio_closep(&oc->pb));
                            }
                        };

                        std::map<int, int64_t> count;

                        std::shared_ptr<AVPacket> pkt;
                        while (true) {
                            packet_buffer.pop(pkt);
                            if (!pkt) {
                                break;
                            }
                            count[pkt->stream_index] += 1;
                            FF(av_interleaved_write_frame(oc, pkt.get()));
                        }

                        auto video_st = video_stream ? video_stream->st : nullptr;
                        auto audio_st = audio_stream ? audio_stream->st : nullptr;

                        // The gate is upstream's, from a segfault fix: av_write_trailer on a
                        // stream that never received a packet crashed. Skipping it silently,
                        // though, leaves a file with no `moov` atom -- unplayable, no error, and
                        // a plausible size on disk. This branch has never been observed to fire
                        // here; it is instrumented because the failure it guards is invisible.
                        //
                        // Not to be confused with reading a recording too early: a backlogged
                        // encoder keeps writing for SECONDS after the channel is torn down
                        // (measured 4 s past KILL on a 431 MB file), and a file read before the
                        // consumer's `Uninitialized` line also shows no moov -- because it has
                        // not been appended yet. That cost a defect report before it was
                        // measured properly.
                        const auto v_count = video_st ? count[video_st->index] : -1;
                        const auto a_count = audio_st ? count[audio_st->index] : -1;
                        if ((!video_st || v_count) && (!audio_st || a_count)) {
                            FF(av_write_trailer(oc));
                        } else {
                            CASPAR_LOG(error)
                                << print()
                                << L" wrote no container trailer, so the file is UNPLAYABLE: "
                                << L"video packets=" << v_count << L", audio packets=" << a_count
                                << L" (-1 means the stream does not exist). A stream that "
                                   L"received no packet at all is the cause.";
                        }

                    } catch (...) {
                        CASPAR_LOG_CURRENT_EXCEPTION();
                        // TODO
                        packet_buffer.abort();
                    }
                });
                CASPAR_SCOPE_EXIT
                {
                    if (packet_thread.joinable()) {
                        // TODO Is nullptr needed?
                        packet_buffer.push(nullptr);
                        packet_buffer.abort();
                        packet_thread.join();
                    }
                };

                auto packet_cb = [&](std::shared_ptr<AVPacket>&& pkt) { packet_buffer.push(std::move(pkt)); };

                std::int64_t frame_number = 0;
                while (true) {
                    {
                        std::lock_guard<std::mutex> lock(state_mutex_);
                        state_["file/frame"] = frame_number++;
                    }

                    std::tuple<core::const_frame, std::int64_t, std::int64_t, core::video_field> data;
                    frame_buffer_.pop(data);
                    graph_->set_value("input",
                                      static_cast<double>(frame_buffer_.size() + 0.001) / frame_buffer_.capacity());

                    caspar::timer frame_timer;
                    tbb::parallel_invoke(
                        [&] {
                            if (video_stream) {
                                video_stream->send(data, format_desc, packet_cb);
                            }
                        },
                        [&] {
                            if (audio_stream) {
                                audio_stream->send(data, format_desc, packet_cb);
                            }
                        });
                    graph_->set_value("frame-time", frame_timer.elapsed() * format_desc.fps * 0.5);

                    if (!std::get<0>(data)) {
                        packet_buffer.push(nullptr);
                        break;
                    }
                }

                packet_thread.join();
            } catch (...) {
                {
                    std::lock_guard<std::mutex> lock(exception_mutex_);
                    exception_ = std::current_exception();
                }
                // If we never got as far as writing the header, initialize() is
                // still waiting; hand it the reason rather than letting it time
                // out. After that point the promise is already satisfied and the
                // exception surfaces through send() as before.
                if (!open_result_set_) {
                    open_result_set_ = true;
                    open_result_.set_exception(std::current_exception());
                }
            }
        });

        // Wait for the encoder to actually open. Everything above runs on the
        // frame thread, so without this ADD answered 202 OK for a consumer that
        // could never encode anything -- "ADD 1 FILE out.mp4 -vcodec av1_nvenc"
        // on hardware with no AV1 encoder reported success, wrote no file, and
        // only surfaced as a 404 on REMOVE.
        //
        // Bounded, because a slow-but-valid encoder must not be reported as a
        // failure: on timeout we say nothing and let the old behaviour stand,
        // with send() rethrowing if it does turn out to have failed.
        auto opened = open_result_.get_future();
        if (opened.wait_for(std::chrono::seconds(5)) == std::future_status::ready) {
            opened.get(); // rethrows the open failure to the caller of ADD
        } else {
            CASPAR_LOG(info) << print() << L" still opening after 5s; reporting success and continuing to wait.";
        }
    }

    std::future<bool> send(core::video_field field, core::const_frame frame) override
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
            stats << u16(print()) << L" Fps: " << current_fps_;
            graph_->set_text(stats.str());
        }

        // TODO - field alignment

        {
            std::lock_guard<std::mutex> lock(exception_mutex_);
            if (exception_ != nullptr) {
                std::rethrow_exception(exception_);
            }
        }

        if (!frame_buffer_.try_push({frame, video_pts, audio_pts, field})) {
            graph_->set_tag(diagnostics::tag_severity::WARNING, "dropped-frame");
            CASPAR_LOG(warning) << "Dropped frame in ffmpeg consumer [" << path_ << "]";
        }

        video_pts += 1;
        audio_pts += frame.audio_data().size() / format_desc_.audio_channels;

        graph_->set_value("input", static_cast<double>(frame_buffer_.size() + 0.001) / frame_buffer_.capacity());

        return make_ready_future(true);
    }

    std::wstring print() const override { return L"ffmpeg[" + u16(path_) + L"]"; }

    std::wstring name() const override { return L"ffmpeg"; }

    bool has_synchronization_clock() const override { return false; }

    /// False only while the GPU-direct path is actually carrying frames. The
    /// channel polls this every tick, so a mid-run failure restores the readback
    /// on the next one.
    bool needs_cpu_frame_data() const override { return !gpu_direct_.load(std::memory_order_relaxed); }

    int index() const override { return 100000 + channel_index_; }

    core::monitor::state state() const override
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        return state_;
    }
};

spl::shared_ptr<core::frame_consumer> create_consumer(const std::vector<std::wstring>&     params,
                                                      const core::video_format_repository& format_repository,
                                                      const std::vector<spl::shared_ptr<core::video_channel>>& channels,
                                                      const core::channel_info& channel_info)
{
    if (params.size() < 2 || (!boost::iequals(params.at(0), L"STREAM") && !boost::iequals(params.at(0), L"FILE")))
        return core::frame_consumer::empty();

    auto                     path = u8(params.at(1));
    std::vector<std::string> args;
    for (auto n = 2; n < params.size(); ++n) {
        args.emplace_back(u8(params[n]));
    }
    return spl::make_shared<ffmpeg_consumer>(
        path, boost::join(args, " "), boost::iequals(params.at(0), L"STREAM"), channel_info.depth,
        channel_info.use_vulkan);
}

spl::shared_ptr<core::frame_consumer>
create_preconfigured_consumer(const boost::property_tree::wptree&                      ptree,
                              const core::video_format_repository&                     format_repository,
                              const std::vector<spl::shared_ptr<core::video_channel>>& channels,
                              const core::channel_info&                                channel_info)
{
    // <hdr-metadata> is spelled the same way as on the DeckLink consumer deliberately: the
    // same four numbers describe the same mastering display, and an operator should not have
    // to learn two spellings to say one thing about their grade. They are folded into the
    // args string because that is the only channel into the encoder setup, and consumed there
    // before FFmpeg sees them.
    auto args = u8(ptree.get<std::wstring>(L"args", L""));
    if (const auto hdr = ptree.get_child_optional(L"hdr-metadata")) {
        const auto append = [&](const char* key, const wchar_t* element) {
            if (const auto value = hdr->get_optional<double>(element)) {
                args += " -" + std::string(key) + " " + std::to_string(*value);
            }
        };
        append("hdr_max_dml", L"max-dml");
        append("hdr_min_dml", L"min-dml");
        append("hdr_max_cll", L"max-cll");
        append("hdr_max_fall", L"max-fall");
    }

    return spl::make_shared<ffmpeg_consumer>(u8(ptree.get<std::wstring>(L"path", L"")),
                                             args,
                                             ptree.get(L"realtime", false),
                                             channel_info.depth,
                                             channel_info.use_vulkan);
}
}} // namespace caspar::ffmpeg
