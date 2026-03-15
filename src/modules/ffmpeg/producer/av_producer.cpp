#include "av_producer.h"

#include "av_input.h"
#include "filter_param_tween.h"

#include "../util/av_assert.h"
#include "../util/av_util.h"

#include <boost/exception/exception.hpp>
#include <boost/format.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/range/algorithm/rotate.hpp>
#include <boost/rational.hpp>
#include <boost/thread.hpp>
#include <boost/thread/condition_variable.hpp>
#include <boost/thread/mutex.hpp>

#include <common/diagnostics/graph.h>
#include <common/env.h>
#include <common/except.h>
#include <common/executor.h>
#include <common/os/thread.h>
#include <common/scope_exit.h>
#include <common/timer.h>

#include <core/frame/draw_frame.h>
#include <core/frame/frame_factory.h>
#include <core/monitor/monitor.h>

#ifdef _WIN32
#include <d3d11.h>
#endif

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
#include <libavutil/avutil.h>
#include <libavutil/channel_layout.h>
#include <libavutil/error.h>
#include <libavutil/opt.h>
#include <libavutil/pixfmt.h>
#include <libavutil/samplefmt.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_d3d11va.h>
}
#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include <algorithm>
#include <atomic>
#include <deque>
#include <iomanip>
#include <memory>
#include <queue>
#include <sstream>
#include <string>
#include <thread>

namespace caspar { namespace ffmpeg {

const AVRational TIME_BASE_Q = {1, AV_TIME_BASE};

struct Frame
{
    std::shared_ptr<AVFrame> video;
    std::shared_ptr<AVFrame> audio;
    core::draw_frame         frame;
    int64_t                  start_time  = AV_NOPTS_VALUE;
    int64_t                  pts         = AV_NOPTS_VALUE;
    int64_t                  duration    = 0;
    int64_t                  frame_count = 0;
};

AVPixelFormat get_pix_fmt_with_alpha(AVPixelFormat fmt)
{
    switch (fmt) {
        case AV_PIX_FMT_YUV420P:
            return AV_PIX_FMT_YUVA420P;
        case AV_PIX_FMT_YUV422P:
            return AV_PIX_FMT_YUVA422P;
        case AV_PIX_FMT_YUV444P:
            return AV_PIX_FMT_YUVA444P;
        default:
            break;
    }
    return fmt;
}

const AVCodec* get_decoder(AVCodecID codec_id)
{
    // enforce use of libvpx for vp8 and vp9 codecs to be able
    // to decode webm files with alpha channel
    const AVCodec* result = nullptr;
    if (codec_id == AV_CODEC_ID_VP9)
        result = avcodec_find_decoder_by_name("libvpx-vp9");
    else if (codec_id == AV_CODEC_ID_VP8)
        result = avcodec_find_decoder_by_name("libvpx");
    return result != nullptr ? result : avcodec_find_decoder(codec_id);
}

// TODO (fix) Handle ts discontinuities.
// TODO (feat) Forward options.

core::color_space get_color_space(const std::shared_ptr<AVFrame>& video)
{
    auto result = core::color_space::bt709;
    if (video) {
        switch (video->colorspace) {
            case AVColorSpace::AVCOL_SPC_BT2020_NCL:
                result = core::color_space::bt2020;
                break;
            case AVColorSpace::AVCOL_SPC_BT470BG:
            case AVColorSpace::AVCOL_SPC_SMPTE170M:
            case AVColorSpace::AVCOL_SPC_SMPTE240M:
                result = core::color_space::bt601;
                break;
            default:
                break;
        }
    }

    return result;
}

class Decoder
{
    static enum AVPixelFormat get_hw_format(AVCodecContext* ctx, const enum AVPixelFormat* pix_fmts)
    {
        const enum AVPixelFormat* p;

        for (p = pix_fmts; *p != -1; p++) {
            if (*p == AV_PIX_FMT_D3D11)
                return *p;
        }

        av_log(ctx, AV_LOG_ERROR, "Failed to get HW surface format.\n");
        return AV_PIX_FMT_NONE;
    }

    Decoder(const Decoder&)            = delete;
    Decoder& operator=(const Decoder&) = delete;

    AVStream*         st       = nullptr;
    int64_t           next_pts = AV_NOPTS_VALUE;
    std::atomic<bool> eof      = {false};

    std::queue<std::shared_ptr<AVPacket>> input;
    mutable boost::mutex                  input_mutex;
    boost::condition_variable             input_cond;
    // 4 pre-staged packets: reduces decoder input starvation between schedule()
    // iterations, especially at high bitrates (e.g. 12K ProRes).
    int                                   input_capacity = 4;

    std::queue<std::shared_ptr<AVFrame>> output;
    mutable boost::mutex                 output_mutex;
    boost::condition_variable            output_cond;
    // Will be raised to ctx->thread_count after avcodec_open2 for video streams
    // so the decode thread is never blocked by output backpressure when the full
    // frame-threading pool has frames ready (e.g. 16 threads, old cap was 8).
    int                                  output_capacity = 8;

    boost::thread             thread;
    std::atomic<bool>         flush_requested_{false};
    boost::mutex              flush_mutex_;
    boost::condition_variable flush_done_cond_;

  public:
    std::shared_ptr<AVCodecContext> ctx;
    // When HW decoding is active, pix_fmt on ctx becomes AV_PIX_FMT_D3D11 (a HW surface
    // format). The filter buffersrc needs a real CPU pixel format instead. This stores
    // the SW pixel format to use for both the buffersrc args and frame transfer.
    AVPixelFormat sw_pix_fmt = AV_PIX_FMT_NONE;

    Decoder() = default;

    explicit Decoder(AVStream* stream)
        : st(stream)
    {
        const auto codec = get_decoder(stream->codecpar->codec_id);

        if (!codec) {
            FF_RET(AVERROR_DECODER_NOT_FOUND, "avcodec_find_decoder");
        }

        ctx = std::shared_ptr<AVCodecContext>(avcodec_alloc_context3(codec),
                                              [](AVCodecContext* ptr) { avcodec_free_context(&ptr); });

        if (!ctx) {
            FF_RET(AVERROR(ENOMEM), "avcodec_alloc_context3");
        }

        FF(avcodec_parameters_to_context(ctx.get(), stream->codecpar));

        if (stream->metadata != NULL) {
            auto entry = av_dict_get(stream->metadata, "alpha_mode", NULL, AV_DICT_MATCH_CASE);
            if (entry != NULL && entry->value != NULL && *entry->value == '1')
                ctx->pix_fmt = get_pix_fmt_with_alpha(ctx->pix_fmt);
        }

        int thread_count = env::properties().get(L"configuration.ffmpeg.producer.threads", 0);
        FF(av_opt_set_int(ctx.get(), "threads", thread_count, 0));

        ctx->pkt_timebase = stream->time_base;

        bool slice_threads = env::properties().get(L"configuration.ffmpeg.producer.slice-threads", true);

        if (ctx->codec_type == AVMEDIA_TYPE_VIDEO) {
            // slice-threads=true  → FF_THREAD_SLICE only: decodes one large frame across all
            //   cores in parallel slices. Lower per-frame latency but all threads fire at once,
            //   which can starve the mixer on high thread counts.
            // slice-threads=false → FFmpeg default (FF_THREAD_FRAME | FF_THREAD_SLICE): lets
            //   FFmpeg choose; frame threading is used when the codec supports it, which spreads
            //   CPU load more evenly but delays first output by thread_count frames.
            if (slice_threads)
                ctx->thread_type = FF_THREAD_SLICE;

            ctx->framerate           = av_guess_frame_rate(nullptr, stream, nullptr);
            ctx->sample_aspect_ratio = av_guess_sample_aspect_ratio(nullptr, stream, nullptr);

            if (codec->id == AV_CODEC_ID_H264 || codec->id == AV_CODEC_ID_HEVC || codec->id == AV_CODEC_ID_VP9) {
                AVBufferRef* hw_device_ctx = nullptr;
                if (av_hwdevice_ctx_create(&hw_device_ctx, AV_HWDEVICE_TYPE_D3D11VA, nullptr, nullptr, 0) == 0) {
                    ctx->hw_device_ctx = hw_device_ctx;
                    ctx->get_format    = get_hw_format;
                    // D3D11VA always transfers to NV12 (or P010 for 10-bit). We advertise NV12
                    // so the filter's buffersrc is configured for a real CPU pixel format.
                    sw_pix_fmt = AV_PIX_FMT_NV12;
                }
            }
        } else if (ctx->codec_type == AVMEDIA_TYPE_AUDIO) {
        }

        FF(avcodec_open2(ctx.get(), codec, nullptr));

        // For video with frame threading the codec resolves threads=0 to
        // hardware_concurrency (e.g. 16). Raise the output queue to match so
        // the decode thread is never blocked waiting for the filter to drain
        // when the entire thread pool finishes frames simultaneously.
        if (ctx->codec_type == AVMEDIA_TYPE_VIDEO && ctx->thread_count > output_capacity) {
            output_capacity = ctx->thread_count;
        }

        thread = boost::thread([=]() {
            while (!thread.interruption_requested()) {
                try {
                    auto av_frame = alloc_frame();
                    auto ret      = avcodec_receive_frame(ctx.get(), av_frame.get());

                    if (ret == AVERROR(EAGAIN)) {
                        std::shared_ptr<AVPacket> packet;
                        {
                            boost::unique_lock<boost::mutex> lock(input_mutex);
                            // Also wake on flush_requested_ so the flush is performed
                            // from this thread (avcodec_flush_buffers is not thread-safe
                            // with concurrent send/receive calls).
                            input_cond.wait(lock, [&]() { return !input.empty() || flush_requested_.load(); });
                            if (flush_requested_.load()) {
                                // Perform the in-place flush from within the decode thread.
                                {
                                    boost::lock_guard<boost::mutex> out_lock(output_mutex);
                                    while (!output.empty())
                                        output.pop();
                                }
                                output_cond.notify_all();
                                avcodec_flush_buffers(ctx.get());
                                next_pts         = AV_NOPTS_VALUE;
                                eof              = false;
                                flush_requested_ = false;
                                flush_done_cond_.notify_all();
                                continue;
                            }
                            packet = std::move(input.front());
                            input.pop();
                        }
                        FF(avcodec_send_packet(ctx.get(), packet.get()));
                    } else if (ret == AVERROR_EOF) {
                        avcodec_flush_buffers(ctx.get());
                        av_frame->pts = next_pts;
                        next_pts      = AV_NOPTS_VALUE;
                        eof           = true;

                        {
                            boost::unique_lock<boost::mutex> lock(output_mutex);
                            output_cond.wait(lock, [&]() { return output.size() < output_capacity || flush_requested_.load(); });
                            if (!flush_requested_.load())
                                output.push(std::move(av_frame));
                        }
                    } else {
                        // Handle HW frame transfer
                        if (av_frame->format == AV_PIX_FMT_D3D11) {
                            auto sw_frame = alloc_frame();
                            // Request the specific SW pixel format that was advertised to the filter's
                            // buffersrc (sw_pix_fmt, e.g. NV12). Width/height must also be set before
                            // the call so FFmpeg can allocate the destination CPU buffer correctly.
                            sw_frame->format = sw_pix_fmt;
                            sw_frame->width  = av_frame->width;
                            sw_frame->height = av_frame->height;
                            int transfer_ret = av_hwframe_transfer_data(sw_frame.get(), av_frame.get(), 0);
                            if (transfer_ret < 0) {
                                char errbuf[AV_ERROR_MAX_STRING_SIZE];
                                av_strerror(transfer_ret, errbuf, AV_ERROR_MAX_STRING_SIZE);
                                CASPAR_LOG(error) << "Error transferring the data to system memory: " << errbuf;
                                continue;
                            }
                            // Copy all frame properties: pts, colorspace, color_range, side data, etc.
                            av_frame_copy_props(sw_frame.get(), av_frame.get());
                            av_frame = sw_frame;
                        }

                        FF_RET(ret, "avcodec_receive_frame");

                        // TODO: Maybe Fixed in:
                        // https://github.com/FFmpeg/FFmpeg/commit/33203a08e0a26598cb103508327a1dc184b27bc6
                        // NOTE This is a workaround for DVCPRO HD.
#if LIBAVCODEC_VERSION_MAJOR < 61
                        if (av_frame->width > 1024 && av_frame->interlaced_frame) {
                            av_frame->top_field_first = 1;
                        }
#else
                        if (av_frame->width > 1024 && (av_frame->flags & AV_FRAME_FLAG_INTERLACED)) {
                            av_frame->flags |= AV_FRAME_FLAG_TOP_FIELD_FIRST;
                        }
#endif

                        // TODO (fix) is this always best?
                        av_frame->pts = av_frame->best_effort_timestamp;

                        auto duration_pts = av_frame->duration;
                        if (duration_pts <= 0) {
                            if (ctx->codec_type == AVMEDIA_TYPE_VIDEO) {
#if LIBAVCODEC_VERSION_MAJOR < 62
                                const int ticks_per_frame = ctx->ticks_per_frame;
#else
                                // https://github.com/FFmpeg/FFmpeg/commit/e930b834a928546f9cbc937f6633709053448232#diff-115616f8a2b59cab3aac4e7f4c8c31e69e94e7fcfa339b9f65b0bf34308aa80fR682
                                const int ticks_per_frame =
                                    (ctx->codec_descriptor && (ctx->codec_descriptor->props & AV_CODEC_PROP_FIELDS))
                                        ? 2
                                        : 1;
#endif
                                const auto ticks = av_stream_get_parser(st) ? av_stream_get_parser(st)->repeat_pict + 1
                                                                            : ticks_per_frame;
                                duration_pts     = static_cast<int64_t>(AV_TIME_BASE) * ctx->framerate.den * ticks /
                                               ctx->framerate.num / ticks_per_frame;
                                duration_pts = av_rescale_q(duration_pts, {1, AV_TIME_BASE}, st->time_base);
                            } else if (ctx->codec_type == AVMEDIA_TYPE_AUDIO) {
                                duration_pts = av_rescale_q(av_frame->nb_samples, {1, ctx->sample_rate}, st->time_base);
                            }
                        }

                        if (duration_pts > 0) {
                            next_pts = av_frame->pts + duration_pts;
                        } else {
                            next_pts = AV_NOPTS_VALUE;
                        }

                        {
                            boost::unique_lock<boost::mutex> lock(output_mutex);
                            // Also wake on flush_requested_ so we don't deadlock if a
                            // flush arrives while the output queue is full.
                            output_cond.wait(lock, [&]() { return output.size() < output_capacity || flush_requested_.load(); });
                            if (!flush_requested_.load())
                                output.push(std::move(av_frame));
                        }
                    }
                } catch (boost::thread_interrupted&) {
                    break;
                } catch (const std::exception& e) {
                    CASPAR_LOG(warning) << "Decoder thread exception (packet dropped): " << e.what();
                } catch (...) {
                    CASPAR_LOG_CURRENT_EXCEPTION();
                }
            }

            // Cleanup any dangling flush requests if thread exits
            if (flush_requested_.load()) {
                boost::lock_guard<boost::mutex> lock(flush_mutex_);
                flush_requested_ = false;
                flush_done_cond_.notify_all();
            }
        });
    }

    ~Decoder()
    {
        try {
            if (thread.joinable()) {
                thread.interrupt();
                thread.join();
            }
        } catch (boost::thread_interrupted&) {
            // Do nothing...
        }
    }

    // Flush the decoder in-place: clears queues, calls avcodec_flush_buffers from
    // within the decode thread (thread-safety requirement), and resets eof/next_pts.
    // The decode thread stays alive — for intra-only codecs (ProRes, NotchLC) the
    // very next packet produces a frame immediately, giving zero-stutter loop seeks.
    void flush()
    {
        // 1. Drop all pending input packets.
        {
            boost::lock_guard<boost::mutex> lock(input_mutex);
            while (!input.empty())
                input.pop();
        }
        // 2. Ask the decode thread to flush; wake it in case it is blocked waiting.
        flush_requested_ = true;
        eof              = false;
        input_cond.notify_all();
        output_cond.notify_all();
        // 3. Wait until the decode thread confirms the flush is done.
        boost::unique_lock<boost::mutex> lock(flush_mutex_);
        if (!flush_done_cond_.wait_for(lock, boost::chrono::milliseconds(500), [&]() { return !flush_requested_.load(); })) {
            CASPAR_LOG(warning) << "Decoder flush timed out - continuing anyway";
            flush_requested_ = false;
        }
    }

    bool want_packet() const
    {
        if (eof) {
            return false;
        }

        {
            boost::lock_guard<boost::mutex> lock(input_mutex);
            return input.size() < input_capacity;
        }
    }

    void push(std::shared_ptr<AVPacket> packet)
    {
        if (eof) {
            return;
        }

        {
            boost::lock_guard<boost::mutex> lock(input_mutex);
            input.push(std::move(packet));
        }

        input_cond.notify_all();
    }

    std::shared_ptr<AVFrame> pop()
    {
        std::shared_ptr<AVFrame> frame;

        {
            boost::lock_guard<boost::mutex> lock(output_mutex);

            if (!output.empty()) {
                frame = std::move(output.front());
                output.pop();
            }
        }

        if (frame) {
            output_cond.notify_all();
        } else if (eof) {
            frame = alloc_frame();
        }

        return frame;
    }
};

struct Filter
{
    std::shared_ptr<AVFilterGraph>  graph;
    AVFilterContext*                sink = nullptr;
    std::map<int, AVFilterContext*> sources;
    std::shared_ptr<AVFrame>        frame;
    bool                            eof = false;

    Filter() = default;

    Filter(std::string                    filter_spec,
           const Input&                   input,
           std::map<int, Decoder>&        streams,
           int64_t                        start_time,
           AVMediaType                    media_type,
           const core::video_format_desc& format_desc)
    {
        if (media_type == AVMEDIA_TYPE_VIDEO) {
            if (filter_spec.empty()) {
                filter_spec = "null";
            }

            auto deint = u8(
                env::properties().get<std::wstring>(L"configuration.ffmpeg.producer.auto-deinterlace", L"interlaced"));

            if (deint != "none") {
                filter_spec += (boost::format(",bwdif=mode=send_field:parity=auto:deint=%s") % deint).str();
            }

            filter_spec += (boost::format(",fps=fps=%d/%d:start_time=%f") %
                            (format_desc.framerate.numerator() * format_desc.field_count) %
                            format_desc.framerate.denominator() % (static_cast<double>(start_time) / AV_TIME_BASE))
                               .str();
        } else if (media_type == AVMEDIA_TYPE_AUDIO) {
            if (filter_spec.empty()) {
                filter_spec = "anull";
            }

            // Find first audio stream to get a time_base for the first_pts calculation
            AVRational tb = {1, format_desc.audio_sample_rate};
            for (auto n = 0U; n < input->nb_streams; ++n) {
                const auto st             = input->streams[n];
                const auto codec_channels = st->codecpar->ch_layout.nb_channels;
                if (st->codecpar->codec_type == AVMEDIA_TYPE_AUDIO && codec_channels > 0) {
                    tb = {1, st->codecpar->sample_rate};
                    break;
                }
            }
            filter_spec += (boost::format(",aresample=async=1000:first_pts=%d:min_comp=0.01:osr=%d,"
                                          "asetnsamples=n=1024:p=0") %
                            av_rescale_q(start_time, TIME_BASE_Q, tb) % format_desc.audio_sample_rate)
                               .str();
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
                const auto type = avfilter_pad_get_type(cur->filter_ctx->input_pads, cur->pad_idx);
                if (type == AVMEDIA_TYPE_VIDEO) {
                    video_input_count += 1;
                } else if (type == AVMEDIA_TYPE_AUDIO) {
                    audio_input_count += 1;
                }
            }
        }

        std::vector<AVStream*> av_streams;
        for (auto n = 0U; n < input->nb_streams; ++n) {
            const auto st = input->streams[n];

            const auto codec_channels = st->codecpar->ch_layout.nb_channels;
            if (st->codecpar->codec_type == AVMEDIA_TYPE_AUDIO && codec_channels == 0) {
                continue;
            }

            auto disposition = st->disposition;
            if (!disposition || disposition == AV_DISPOSITION_DEFAULT) {
                av_streams.push_back(st);
            }
        }

        if (audio_input_count == 1) {
            auto count = std::count_if(av_streams.begin(), av_streams.end(), [](auto s) {
                return s->codecpar->codec_type == AVMEDIA_TYPE_AUDIO;
            });

            // TODO (fix) Use some form of stream meta data to do this.
            // https://github.com/CasparCG/server/issues/833
            if (count > 1) {
                filter_spec = (boost::format("amerge=inputs=%d,") % count).str() + filter_spec;
            }
        }

        if (video_input_count == 1) {
            std::stable_sort(av_streams.begin(), av_streams.end(), [](auto lhs, auto rhs) {
                return lhs->codecpar->codec_type == AVMEDIA_TYPE_VIDEO && lhs->codecpar->height > rhs->codecpar->height;
            });

            std::vector<AVStream*> video_av_streams;
            std::copy_if(av_streams.begin(), av_streams.end(), std::back_inserter(video_av_streams), [](auto s) {
                return s->codecpar->codec_type == AVMEDIA_TYPE_VIDEO;
            });

            // TODO (fix) Use some form of stream meta data to do this.
            // https://github.com/CasparCG/server/issues/832
            if (video_av_streams.size() >= 2 &&
                video_av_streams[0]->codecpar->height == video_av_streams[1]->codecpar->height) {
                filter_spec = "alphamerge," + filter_spec;
            }
        }

        graph = std::shared_ptr<AVFilterGraph>(avfilter_graph_alloc(),
                                               [](AVFilterGraph* ptr) { avfilter_graph_free(&ptr); });

        if (!graph) {
            FF_RET(AVERROR(ENOMEM), "avfilter_graph_alloc");
        }

        FF(avfilter_graph_parse2(graph.get(), filter_spec.c_str(), &inputs, &outputs));

        // inputs
        {
            for (auto cur = inputs; cur; cur = cur->next) {
                const auto type = avfilter_pad_get_type(cur->filter_ctx->input_pads, cur->pad_idx);
                if (type != AVMEDIA_TYPE_VIDEO && type != AVMEDIA_TYPE_AUDIO) {
                    CASPAR_THROW_EXCEPTION(ffmpeg_error_t() << boost::errinfo_errno(EINVAL)
                                                            << msg_info_t("only video and audio filters supported"));
                }

                unsigned index = 0;

                // TODO find stream based on link name
                while (true) {
                    if (index == av_streams.size()) {
                        graph = nullptr;
                        return;
                    }
                    if (av_streams.at(index)->codecpar->codec_type == type &&
                        sources.find(static_cast<int>(index)) == sources.end()) {
                        break;
                    }
                    index++;
                }

                index = av_streams.at(index)->index;

                auto it = streams.find(index);
                if (it == streams.end()) {
                    it = streams.emplace(index, input->streams[index]).first;
                }

                auto st = it->second.ctx;

                if (st->codec_type == AVMEDIA_TYPE_VIDEO) {
                    // If the decoder uses HW acceleration, ctx->pix_fmt is a HW surface format
                    // (e.g. AV_PIX_FMT_D3D11). We must configure the buffersrc with the real
                    // CPU pixel format that the decoder will produce after the HW->SW transfer.
                    // sw_pix_fmt is set to AV_PIX_FMT_NV12 when HW decoding is active.
                    const auto src_fmt = (it->second.sw_pix_fmt != AV_PIX_FMT_NONE)
                                             ? it->second.sw_pix_fmt
                                             : st->pix_fmt;
                    auto args = (boost::format("video_size=%dx%d:pix_fmt=%d:time_base=%d/%d") % st->width % st->height %
                                 src_fmt % st->pkt_timebase.num % st->pkt_timebase.den)
                                    .str();
                    auto name = (boost::format("in_%d") % index).str();

                    if (st->sample_aspect_ratio.num > 0 && st->sample_aspect_ratio.den > 0) {
                        args +=
                            (boost::format(":sar=%d/%d") % st->sample_aspect_ratio.num % st->sample_aspect_ratio.den)
                                .str();
                    }

                    if (st->framerate.num > 0 && st->framerate.den > 0) {
                        args += (boost::format(":frame_rate=%d/%d") % st->framerate.num % st->framerate.den).str();
                    }

                    AVFilterContext* source = nullptr;
                    FF(avfilter_graph_create_filter(
                        &source, avfilter_get_by_name("buffer"), name.c_str(), args.c_str(), nullptr, graph.get()));
                    FF(avfilter_link(source, 0, cur->filter_ctx, cur->pad_idx));
                    sources.emplace(index, source);
                } else if (st->codec_type == AVMEDIA_TYPE_AUDIO) {
                    char channel_layout[128];
                    FF(av_channel_layout_describe(&st->ch_layout, channel_layout, sizeof(channel_layout)));

                    auto args = (boost::format("time_base=%d/%d:sample_rate=%d:sample_fmt=%s:channel_layout=%#x") %
                                 st->pkt_timebase.num % st->pkt_timebase.den % st->sample_rate %
                                 av_get_sample_fmt_name(st->sample_fmt) % channel_layout)
                                    .str();
                    auto name = (boost::format("in_%d") % index).str();

                    AVFilterContext* source = nullptr;
                    FF(avfilter_graph_create_filter(
                        &source, avfilter_get_by_name("abuffer"), name.c_str(), args.c_str(), nullptr, graph.get()));
                    FF(avfilter_link(source, 0, cur->filter_ctx, cur->pad_idx));
                    sources.emplace(index, source);
                } else {
                    CASPAR_THROW_EXCEPTION(ffmpeg_error_t() << boost::errinfo_errno(EINVAL)
                                                            << msg_info_t("invalid filter input media type"));
                }
            }
        }

        if (media_type == AVMEDIA_TYPE_VIDEO) {
            FF(avfilter_graph_create_filter(
                &sink, avfilter_get_by_name("buffersink"), "out", nullptr, nullptr, graph.get()));

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4245)
#endif
            const AVPixelFormat pix_fmts[] = {AV_PIX_FMT_RGB24,
                                              AV_PIX_FMT_BGR24,
                                              AV_PIX_FMT_BGRA,
                                              AV_PIX_FMT_ARGB,
                                              AV_PIX_FMT_RGBA,
                                              AV_PIX_FMT_ABGR,
                                              AV_PIX_FMT_YUV444P,
                                              AV_PIX_FMT_YUV444P10,
                                              AV_PIX_FMT_YUV444P12,
                                              AV_PIX_FMT_YUV422P,
                                              AV_PIX_FMT_YUV422P10,
                                              AV_PIX_FMT_YUV422P12,
                                              AV_PIX_FMT_YUV420P,
                                              AV_PIX_FMT_YUV420P10,
                                              AV_PIX_FMT_YUV420P12,
                                              AV_PIX_FMT_YUV410P,
                                              AV_PIX_FMT_YUVA444P,
                                              AV_PIX_FMT_YUVA422P,
                                              AV_PIX_FMT_YUVA420P,
                                              AV_PIX_FMT_UYVY422,
                                              // bwdif needs planar rgb
                                              AV_PIX_FMT_GBRP,
                                              AV_PIX_FMT_GBRP10,
                                              AV_PIX_FMT_GBRP12,
                                              AV_PIX_FMT_GBRP16,
                                              AV_PIX_FMT_GBRAP,
                                              AV_PIX_FMT_GBRAP16,
                                              AV_PIX_FMT_NONE};
            FF(av_opt_set_int_list(sink, "pix_fmts", pix_fmts, -1, AV_OPT_SEARCH_CHILDREN));
#ifdef _MSC_VER
#pragma warning(pop)
#endif
        } else if (media_type == AVMEDIA_TYPE_AUDIO) {
            FF(avfilter_graph_create_filter(
                &sink, avfilter_get_by_name("abuffersink"), "out", nullptr, nullptr, graph.get()));
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4245)
#endif
            const AVSampleFormat sample_fmts[] = {AV_SAMPLE_FMT_S32, AV_SAMPLE_FMT_NONE};
            FF(av_opt_set_int_list(sink, "sample_fmts", sample_fmts, -1, AV_OPT_SEARCH_CHILDREN));

            FF(av_opt_set_int(sink, "all_channel_counts", 1, AV_OPT_SEARCH_CHILDREN));

            const int sample_rates[] = {format_desc.audio_sample_rate, -1};
            FF(av_opt_set_int_list(sink, "sample_rates", sample_rates, -1, AV_OPT_SEARCH_CHILDREN));
#ifdef _MSC_VER
#pragma warning(pop)
#endif
        } else {
            CASPAR_THROW_EXCEPTION(ffmpeg_error_t()
                                   << boost::errinfo_errno(EINVAL) << msg_info_t("invalid output media type"));
        }

        // output
        {
            const auto cur = outputs;

            if (!cur || cur->next) {
                CASPAR_THROW_EXCEPTION(ffmpeg_error_t() << boost::errinfo_errno(EINVAL)
                                                        << msg_info_t("invalid filter graph output count"));
            }

            if (avfilter_pad_get_type(cur->filter_ctx->output_pads, cur->pad_idx) != media_type) {
                CASPAR_THROW_EXCEPTION(ffmpeg_error_t() << boost::errinfo_errno(EINVAL)
                                                        << msg_info_t("invalid filter output media type"));
            }

            FF(avfilter_link(cur->filter_ctx, cur->pad_idx, sink, 0));
        }

        FF(avfilter_graph_config(graph.get(), nullptr));

        CASPAR_LOG(debug) << avfilter_graph_dump(graph.get(), nullptr);
    }

    bool operator()(int nb_samples = -1)
    {
        if (frame || eof) {
            return false;
        }

        if (!sink || sources.empty()) {
            eof   = true;
            frame = nullptr;
            return true;
        }

        auto av_frame = alloc_frame();
        auto ret      = nb_samples >= 0 ? av_buffersink_get_samples(sink, av_frame.get(), nb_samples)
                                        : av_buffersink_get_frame(sink, av_frame.get());

        if (ret == AVERROR(EAGAIN)) {
            return false;
        }
        if (ret == AVERROR_EOF) {
            eof   = true;
            frame = nullptr;
            return true;
        }
        FF_RET(ret, "av_buffersink_get_frame");
        frame = std::move(av_frame);
        return true;
    }
};

struct AVProducer::Impl
{
    caspar::core::monitor::state state_;
    mutable boost::mutex         state_mutex_;

    spl::shared_ptr<diagnostics::graph> graph_;

    const std::shared_ptr<core::frame_factory> frame_factory_;
    const core::video_format_desc              format_desc_;
    const AVRational                           format_tb_;
    const std::string                          name_;
    const std::string                          path_;

    Input                  input_;
    std::map<int, Decoder> decoders_;
    Filter                 video_filter_;
    Filter                 audio_filter_;

    std::map<int, std::vector<AVFilterContext*>> sources_;

    std::atomic<int64_t> start_{AV_NOPTS_VALUE};
    std::atomic<int64_t> duration_{AV_NOPTS_VALUE};
    std::atomic<int64_t> input_duration_{AV_NOPTS_VALUE};
    std::atomic<int64_t> seek_{AV_NOPTS_VALUE};
    std::atomic<bool>    loop_{false};
    std::atomic<bool>    pingpong_{false};  // ping-pong: auto-reverse at each end
    bool                 growing_{false};

    std::string afilter_;
    std::string vfilter_;

    // Per-parameter animation state for VFPARAM / AFPARAM CALL commands.
    // Keyed by [filter_type_name][param_name].  Accessed from the AMCP thread
    // (set_filter_param) and the decode thread (apply_filter_param_tweens),
    // so protected by param_tween_mutex_.
    std::map<std::string, std::map<std::string, FilterParamTween>> video_param_tweens_;
    std::map<std::string, std::map<std::string, FilterParamTween>> audio_param_tweens_;
    mutable boost::mutex                                           param_tween_mutex_;

    int                              seekable_ = 2;
    core::frame_geometry::scale_mode scale_mode_;
    int64_t                          frame_count_    = 0;
    bool                             frame_flush_    = true;
    int64_t                          frame_time_     = AV_NOPTS_VALUE;
    int64_t                          frame_duration_ = AV_NOPTS_VALUE;
    std::atomic<double>              speed_{1.0};    // playback rate: 0.5 = half-speed, 2.0 = double-speed
    double                           speed_accum_    = 0.0; // fractional accumulator, protected by buffer_mutex_
    std::deque<Frame>                rev_frames_;            // batch of decoded frames for reverse playback (served back→front)
    bool                             rev_active_     = false; // true once the initial reverse seek has been issued
    core::draw_frame                 frame_;

    std::deque<Frame>         buffer_;
    mutable boost::mutex      buffer_mutex_;
    boost::condition_variable buffer_cond_;
    std::atomic<bool>         buffer_eof_{false};
    int                       buffer_capacity_ = 0; // set in constructor from config
    std::atomic<int64_t>      current_seek_target_{AV_NOPTS_VALUE};

    std::optional<caspar::executor> video_executor_;
    std::optional<caspar::executor> audio_executor_;

    int latency_ = 0;

    std::chrono::steady_clock::time_point last_fps_update_;
    int                     frames_since_update_ = 0;
    double                  current_fps_ = 0.0;

    boost::thread thread_;

    Impl(std::shared_ptr<core::frame_factory> frame_factory,
         core::video_format_desc              format_desc,
         std::string                          name,
         std::string                          path,
         std::string                          vfilter,
         std::string                          afilter,
         std::optional<int64_t>               start,
         std::optional<int64_t>               seek,
         std::optional<int64_t>               duration,
         bool                                 loop,
         int                                  seekable,
         core::frame_geometry::scale_mode     scale_mode,
         bool                                 growing)
        : growing_(growing)
        , frame_factory_(frame_factory)
        , format_desc_(format_desc)
        , format_tb_({format_desc.duration, format_desc.time_scale * format_desc.field_count})
        , name_(name)
        , path_(path)
        , input_(path, graph_, seekable >= 0 && seekable < 2 ? std::optional<bool>(false) : std::optional<bool>())
        , start_(start ? av_rescale_q(*start, format_tb_, TIME_BASE_Q) : AV_NOPTS_VALUE)
        , duration_(duration ? av_rescale_q(*duration, format_tb_, TIME_BASE_Q) : AV_NOPTS_VALUE)
        , loop_(loop)
        , afilter_(afilter)
        , vfilter_(vfilter)
        , seekable_(seekable)
        , scale_mode_(scale_mode)
        , video_executor_(L"video-executor")
        , audio_executor_(L"audio-executor")
    {
        diagnostics::register_graph(graph_);
        graph_->set_color("underflow", diagnostics::color(0.6f, 0.3f, 0.9f));
        graph_->set_color("frame-time", diagnostics::color(0.0f, 1.0f, 0.0f));
        graph_->set_color("decode-time", diagnostics::color(0.0f, 1.0f, 1.0f));
        graph_->set_color("buffer", diagnostics::color(1.0f, 1.0f, 0.0f));

        const int default_buffer_depth = std::max(1, static_cast<int>(format_desc_.fps) / 4);
        buffer_capacity_ = env::properties().get(L"configuration.ffmpeg.producer.buffer-depth", default_buffer_depth);
        CASPAR_LOG(debug) << print() << " buffer-depth: " << buffer_capacity_;

        state_["file/name"] = u8(name_);
        state_["file/path"] = u8(path_);
        state_["loop"]      = loop;
        update_state();

        CASPAR_LOG(debug) << print() << " seekable: " << seekable_;

        thread_ = boost::thread([=] {
            try {
                run(seek);
            } catch (boost::thread_interrupted&) {
                // Do nothing...
            } catch (ffmpeg::ffmpeg_error_t& ex) {
                if (auto errn = boost::get_error_info<ffmpeg_errn_info>(ex)) {
                    if (*errn == AVERROR_EXIT) {
                        return;
                    }
                }
                CASPAR_LOG_CURRENT_EXCEPTION();
            } catch (...) {
                CASPAR_LOG_CURRENT_EXCEPTION();
            }
        });
    }

    ~Impl()
    {
        input_.abort();

        try {
            if (thread_.joinable()) {
                thread_.interrupt();
                thread_.join();
            }
        } catch (boost::thread_interrupted&) {
            // Do nothing...
        }

        video_executor_.reset();
        audio_executor_.reset();

        CASPAR_LOG(debug) << print() << " Joined";
    }

    void run(std::optional<int64_t> firstSeek)
    {
        std::vector<int> audio_cadence = format_desc_.audio_cadence;

        input_.reset();
        {
            core::monitor::state streams;
            for (auto n = 0UL; n < input_->nb_streams; ++n) {
                auto st                             = input_->streams[n];
                auto framerate                      = av_guess_frame_rate(nullptr, st, nullptr);
                streams[std::to_string(n) + "/fps"] = {framerate.num, framerate.den};
            }

            boost::lock_guard<boost::mutex> lock(state_mutex_);
            state_["file/streams"] = streams;
        }

        if (input_duration_ == AV_NOPTS_VALUE) {
            int64_t v_dur = AV_NOPTS_VALUE;
            for (auto n = 0UL; n < input_->nb_streams; ++n) {
                auto st = input_->streams[n];
                if (st->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
                    if (st->duration != AV_NOPTS_VALUE) {
                        v_dur = av_rescale_q(st->duration, st->time_base, {1, AV_TIME_BASE});
                    } else if (input_->duration != AV_NOPTS_VALUE) {
                        // Some formats (like MXF) don't have stream duration, use global.
                        v_dur = input_->duration;
                    }
                    break;
                }
            }
            if (v_dur != AV_NOPTS_VALUE && v_dur > 0) {
                input_duration_ = v_dur;
            } else {
                input_duration_ = input_->duration;
            }
        }

        {
            const auto start = start_.load();
            if (duration_ == AV_NOPTS_VALUE && input_duration_.load() > 0) {
                if (start != AV_NOPTS_VALUE) {
                    duration_ = input_duration_.load() - start;
                } else {
                    duration_ = input_duration_.load();
                }
            }

            const auto firstStart = firstSeek ? av_rescale_q(*firstSeek, format_tb_, TIME_BASE_Q) : start;
            if (firstStart != AV_NOPTS_VALUE) {
                seek_internal(firstStart);
            } else {
                reset(input_->start_time != AV_NOPTS_VALUE ? input_->start_time : 0);
            }
        }

        set_thread_name(L"[ffmpeg::av_producer]");

        boost::range::rotate(audio_cadence, std::end(audio_cadence) - 1);

        Frame frame;
        Frame last_dropped_frame;
        std::shared_ptr<AVFrame> last_valid_video;
        timer frame_timer;
        timer decode_timer;

        int warning_debounce = 0;

        while (!thread_.interruption_requested()) {
            try {
                {
                    const auto seek = seek_.exchange(AV_NOPTS_VALUE);

                    if (seek != AV_NOPTS_VALUE) {
                        try {
                            seek_internal(seek);
                            current_seek_target_ = seek;
                            last_dropped_frame   = Frame{};
                            last_valid_video.reset();
                            frame                = Frame{};
                        } catch (const std::exception& e) {
                            CASPAR_LOG(warning) << print() << " Seek (graph rebuild) failed: " << e.what() << " - retrying";
                            // Restore the seek command so the next loop iteration retries it,
                            // unless another newer seek has already been enqueued!
                            int64_t expected = AV_NOPTS_VALUE;
                            seek_.compare_exchange_strong(expected, seek);
                            std::this_thread::sleep_for(std::chrono::milliseconds(20));
                        }
                        continue;
                    }
                }

                {
                    // TODO (perf) seek as soon as input is past duration or eof.

                    auto start    = start_.load();
                    auto duration = duration_.load();

                    start       = start != AV_NOPTS_VALUE ? start : 0;
                    auto end    = duration != AV_NOPTS_VALUE ? start + duration : INT64_MAX;
                    auto time   = frame.pts != AV_NOPTS_VALUE ? frame.pts + frame.duration : 0;
                    
                    if (frame.frame_count == 0 && frame_count_ == 0 && current_seek_target_.load() != AV_NOPTS_VALUE) {
                        // Special case: we are just starting (or just seeked) and haven't decoded any frames yet.
                        // We must NOT calculate EOF based on `frame` because `frame` is empty/zero!
                        // This prevents an immediate EOF triggering before the decode pipeline produces the first result.
                        buffer_eof_ = false;
                    } else {
                        buffer_eof_ = !growing_ && ((video_filter_.eof && audio_filter_.eof) ||
                                      av_rescale_q(time, TIME_BASE_Q, format_tb_) >= av_rescale_q(end, TIME_BASE_Q, format_tb_));
                    }

                    if (buffer_eof_) {
                        if (current_seek_target_.load() != AV_NOPTS_VALUE && (last_dropped_frame.video || last_dropped_frame.audio)) {
                            // We hit EOF while fast-forwarding to a seek target (the target was beyond the video).
                            // Render and push the very last dropped frame so we don't output a black screen.
                            last_dropped_frame.frame = core::draw_frame(
                                make_frame(this, *frame_factory_, last_dropped_frame.video, last_dropped_frame.audio, get_color_space(last_dropped_frame.video), scale_mode_));
                            last_dropped_frame.frame_count = frame_count_++;

                            boost::unique_lock<boost::mutex> buffer_lock(buffer_mutex_);
                            buffer_cond_.wait(buffer_lock, [&] { return buffer_.size() < buffer_capacity_; });
                            if (seek_.load() == AV_NOPTS_VALUE) {
                                buffer_.push_back(std::move(last_dropped_frame));
                                buffer_cond_.notify_all();
                            }
                            current_seek_target_ = AV_NOPTS_VALUE;
                        }

                        if (loop_ && !pingpong_ && frame_count_ > 2 && seek_.load() == AV_NOPTS_VALUE && speed_.load() >= 0.0) {
                            // Normal loop — seek back to IN point.
                            // Only auto-loop if no user seek is pending; if seek_ is set the
                            // next iteration will consume it and we must not override it with start.
                            // ALSO disabled if playing in reverse, let frontend logic handle reverse boundary.
                            frame = Frame{};
                            seek_internal(start);
                        } else {
                            // ping-pong, non-looping, or a user seek is pending:
                            // stall here so next_frame() / the top-of-loop seek_ check can act.
                            std::this_thread::sleep_for(std::chrono::milliseconds(10));
                        }
                        // TODO (fix) Limit live polling due to bugs.
                        continue;
                    }
                }

                bool progress = false;
                {
                    progress |= schedule();

                    std::vector<std::future<bool>> futures;

                    if (!video_filter_.frame) {
                        futures.push_back(video_executor_->begin_invoke([&]() { return video_filter_(); }));
                    }

                    if (!audio_filter_.frame) {
                        futures.push_back(audio_executor_->begin_invoke([&]() { return audio_filter_(audio_cadence[0]); }));
                    }

                    for (auto& future : futures) {
                        progress |= future.get();
                    }
                }

                if ((!video_filter_.frame && !video_filter_.eof) || (!audio_filter_.frame && !audio_filter_.eof)) {
                    if (!progress) {
                        if (warning_debounce++ % 500 == 100) {
                            if (!video_filter_.frame && !video_filter_.eof) {
                                CASPAR_LOG(warning) << print() << " Waiting for video frame...";
                            } else if (!audio_filter_.frame && !audio_filter_.eof) {
                                CASPAR_LOG(warning) << print() << " Waiting for audio frame...";
                            } else {
                                CASPAR_LOG(warning) << print() << " Waiting for frame...";
                            }
                        }

                        // TODO (perf): Avoid live loop.
                        // 1ms keeps CPU usage acceptable while minimising the
                        // pipeline fill-up latency after a loop seek.
                        std::this_thread::sleep_for(std::chrono::milliseconds(warning_debounce > 25 ? 10 : 1));
                    }
                    continue;
                }

                warning_debounce = 0;

                // TODO (fix)
                // if (start_ != AV_NOPTS_VALUE && frame.pts < start_) {
                //    seek_internal(start_);
                //    continue;
                //}

                const auto start_time = input_->start_time != AV_NOPTS_VALUE ? input_->start_time : 0;

                bool use_video_pts = video_filter_.frame != nullptr;

                if (video_filter_.frame) {
                    frame.video      = std::move(video_filter_.frame);
                    last_valid_video = frame.video;
                    const auto tb    = av_buffersink_get_time_base(video_filter_.sink);
                    const auto fr    = av_buffersink_get_frame_rate(video_filter_.sink);
                    frame.start_time = start_time;
                    frame.pts        = av_rescale_q(frame.video->pts, tb, TIME_BASE_Q) - start_time;
                    frame.duration   = av_rescale_q(1, av_inv_q(fr), TIME_BASE_Q);
                } else if (last_valid_video) {
                    frame.video = last_valid_video; // Keep the last video frame if we have an audio tail!
                }

                if (audio_filter_.frame) {
                    frame.audio      = std::move(audio_filter_.frame);
                    const auto tb    = av_buffersink_get_time_base(audio_filter_.sink);
                    const auto sr    = av_buffersink_get_sample_rate(audio_filter_.sink);
                    frame.start_time = start_time;
                    if (!use_video_pts) {
                        frame.pts        = av_rescale_q(frame.audio->pts, tb, TIME_BASE_Q) - start_time;
                    }
                    if (frame.duration <= 0) {
                        frame.duration   = av_rescale_q(frame.audio->nb_samples, {1, sr}, TIME_BASE_Q);
                    }
                }

                if (current_seek_target_.load() != AV_NOPTS_VALUE && frame.pts != AV_NOPTS_VALUE) {
                    if (frame.pts < current_seek_target_.load() - (frame.duration > 0 ? frame.duration / 2 : 0)) {
                        last_dropped_frame = std::move(frame);
                        frame = Frame{};
                        continue;
                    } else {
                        current_seek_target_ = AV_NOPTS_VALUE;
                        last_dropped_frame = Frame{};
                    }
                }

                frame.frame = core::draw_frame(
                    make_frame(this, *frame_factory_, frame.video, frame.audio, get_color_space(frame.video), scale_mode_));
                frame.frame_count = frame_count_++;

                graph_->set_value("decode-time", decode_timer.elapsed() * format_desc_.fps * 0.5);

                {
                    boost::unique_lock<boost::mutex> buffer_lock(buffer_mutex_);
                    buffer_cond_.wait(buffer_lock, [&] { return buffer_.size() < buffer_capacity_; });
                    if (seek_ == AV_NOPTS_VALUE) {
                        buffer_.push_back(frame);
                    }
                }

                if (format_desc_.field_count != 2 || frame_count_ % 2 == 1) {
                    // Update the frame-time every other frame when interlaced
                    graph_->set_value("frame-time", frame_timer.elapsed() * format_desc_.hz * 0.5);
                    frame_timer.restart();
                }

                decode_timer.restart();

                graph_->set_value("buffer", static_cast<double>(buffer_.size()) / static_cast<double>(buffer_capacity_));

                boost::range::rotate(audio_cadence, std::end(audio_cadence) - 1);

                // Tick all animated filter parameters and push updated values into
                // the live filter graph via avfilter_graph_send_command.
                // Called here because both video and audio filter futures have been
                // get()-ed above, so the graph is idle and safe to command.
                apply_filter_param_tweens();
            } catch (boost::thread_interrupted&) {
                throw;
            } catch (std::exception& e) {
                CASPAR_LOG(error) << print() << " Exception in decode loop (will retry): " << e.what();
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            } catch (...) {
                CASPAR_LOG(error) << print() << " Unknown exception in decode loop (will retry)";
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
        }
    }

    void update_state()
    {
        std::wstringstream stats;
        stats.precision(2);
        stats << std::fixed;
        stats << u16(print()) << L" fps: " << current_fps_;
        graph_->set_text(stats.str());

        boost::lock_guard<boost::mutex> lock(state_mutex_);
        state_["file/clip"] = {start().value_or(0) / format_desc_.fps, duration().value_or(0) / format_desc_.fps};
        state_["file/time"] = {time() / format_desc_.fps, file_duration().value_or(0) / format_desc_.fps};
        state_["loop"]      = loop_.load();
        state_["pingpong"]  = pingpong_.load();
    }

    core::draw_frame prev_frame(const core::video_field field)
    {
        CASPAR_SCOPE_EXIT { update_state(); };

        // Don't start a new frame on the 2nd field
        if (field != core::video_field::b) {
            if (frame_flush_ || !frame_) {
                boost::lock_guard<boost::mutex> lock(buffer_mutex_);

                if (!buffer_.empty()) {
                    frame_          = buffer_[0].frame;
                    frame_time_     = buffer_[0].pts;
                    frame_duration_ = buffer_[0].duration;
                    frame_flush_    = false;
                }
            }
        }

        return core::draw_frame::still(frame_);
    }

    bool is_ready()
    {
        boost::lock_guard<boost::mutex> lock(buffer_mutex_);
        return !buffer_.empty() || frame_;
    }

    core::draw_frame next_frame(const core::video_field field)
    {
        auto now = std::chrono::steady_clock::now();
        frames_since_update_++;
        auto duration_sec = std::chrono::duration_cast<std::chrono::duration<double>>(now - last_fps_update_).count();

        if (duration_sec >= 1.0) {
            current_fps_ = (double)frames_since_update_ / duration_sec;
            frames_since_update_ = 0;
            last_fps_update_ = now;
        }

        CASPAR_SCOPE_EXIT { update_state(); };

        boost::lock_guard<boost::mutex> lock(buffer_mutex_);

        // When speed is negative and no frame has been produced yet, the decode
        // thread will have started buffering from the IN point.  Issue an initial
        // seek to the OUT point so reverse playback starts from the end of the clip.
        // Guard with !rev_active_ so we fire exactly once per reverse session.
        // rev_active_ is reset whenever playback direction changes forward.
        if (speed_.load() < 0.0 && !frame_ && rev_frames_.empty() && !rev_active_) {
            const int64_t  s0       = start_.load() != AV_NOPTS_VALUE ? start_.load() : 0LL;
            const int64_t  dur      = duration_.load();
            const int64_t  indur    = input_duration_.load();
            int64_t        end_pts  = AV_NOPTS_VALUE;
            if (dur != AV_NOPTS_VALUE && dur > 0)
                end_pts = s0 + dur;
            else if (indur != AV_NOPTS_VALUE && indur > 0)
                end_pts = indur;
            if (end_pts != AV_NOPTS_VALUE) {
                // Seek buffer_capacity_ frames before the exclusive end so the
                // decoder fills a complete batch rather than producing 0-1 frames
                // immediately before EOF.
                const int64_t one_frame = av_rescale_q(1, format_tb_, TIME_BASE_Q);
                const int64_t seek_pos  = std::max(s0, end_pts - static_cast<int64_t>(buffer_capacity_) * one_frame);
                rev_active_ = true;
                seek_ = seek_pos;
                buffer_.clear();
                buffer_cond_.notify_all();
            }
            return core::draw_frame{};
        }

        // Pre-roll threshold: hold output until this many frames are buffered
        // after a seek/loop before releasing to the first consumer call.
        // With flush-in-place the first decoded frame is correct, so 2 frames
        // of headroom is enough to absorb A/V sync jitter without the ~80ms
        // extra stall that the old value of 4 caused at 25fps.
        // In reverse mode each backward seek only produces 1 usable frame before
        // hitting EOF again, so the 2-frame pre-roll requirement is bypassed.
        const bool in_reverse     = speed_.load() < 0.0;
        const bool have_rev_cache = in_reverse && !rev_frames_.empty();

        // Eagerly consume the first post-seek frame before the pre-roll check.
        // This is necessary for paused playback (speed=0): frames_to_advance is
        // always 0 so the normal consumption path is never reached, which means
        // frame_flush_ is never cleared and frame_ is never updated.
        // By consuming here we also handle the EOF-with-1-frame case (buffer_eof_=true
        // but only 1 frame decoded so the size<2 pre-roll guard would otherwise stall).
        if (frame_flush_ && !buffer_.empty() && !in_reverse) {
            frame_          = buffer_[0].frame;
            frame_time_     = buffer_[0].pts;
            frame_duration_ = buffer_[0].duration;
            frame_flush_    = false;
            buffer_.pop_front();
            buffer_cond_.notify_all();
        }

        // For forward play: 2-frame pre-roll to absorb pipeline jitter.
        // For reverse play: strict full batch pre-roll (buffer_capacity_) to maximize caching and avoid micro-seeks,
        //                   but bypass if EOF is reached and the batch ends prematurely.
        const bool reverse_req_met = in_reverse && (buffer_.size() >= static_cast<size_t>(buffer_capacity_) || buffer_eof_);
        const bool drop_to_wait    = !have_rev_cache && (buffer_.empty() || (!in_reverse && frame_flush_ && buffer_.size() < 2) || (in_reverse && !reverse_req_met));

        if (drop_to_wait) {
            auto start    = start_.load();
            auto duration = duration_.load();

            start    = start != AV_NOPTS_VALUE ? start : 0;
            auto end = duration != AV_NOPTS_VALUE ? start + duration : INT64_MAX;

            if ((buffer_eof_ || growing_) && !frame_flush_ && !in_reverse) {
                if (pingpong_ && speed_.load() > 0.0) {
                    // Forward playback hit OUT point — flip to reverse.
                    // Seek buffer_capacity_ frames before the current position so
                    // the first reverse batch is full rather than 0-1 frames.
                    const double  spd_abs   = std::abs(speed_.load());
                    const int64_t one_frame = av_rescale_q(1, format_tb_, TIME_BASE_Q);
                    const int64_t s0        = start_.load() != AV_NOPTS_VALUE ? start_.load() : 0LL;
                    speed_       = -spd_abs;
                    speed_accum_ = 0.0;
                    rev_frames_.clear();
                    rev_active_  = true;   // seek is issued here, no need for initial-seek guard
                    buffer_eof_  = false;
                    const int64_t seek_pos = std::max(s0,
                        frame_time_ - static_cast<int64_t>(buffer_capacity_ - 1) * one_frame);
                    seek_ = seek_pos;
                    buffer_.clear();
                    buffer_cond_.notify_all();
                    return core::draw_frame::still(frame_);
                }
                if (frame_time_ < end && frame_duration_ != AV_NOPTS_VALUE) {
                    frame_time_ += frame_duration_;
                } else if (frame_time_ < end) {
                    frame_time_ = input_duration_;
                }
                return core::draw_frame::still(frame_);
            }
            // Hold the last decoded frame as a still rather than going black during:
            //  - reverse-mode buffer stalls after each backward seek
            //  - forward seek pre-roll (frame_flush_=true, buffer not yet full)
            //  - seeking past the last video frame (buffer_eof_ with frame_flush_ still set)
            // Only signal a true underflow (black) if no frame has ever been decoded.
            if (frame_) {
                return core::draw_frame::still(frame_);
            }
            graph_->set_tag(diagnostics::tag_severity::WARNING, "underflow");
            latency_ += 1;
            return core::draw_frame{};
        }

        if (!in_reverse && format_desc_.field_count == 2) {
            // Check if the next frame is the correct 'field'
            auto is_field_1 = (buffer_[0].frame_count % 2) == 0;
            if ((field == core::video_field::a && !is_field_1) || (field == core::video_field::b && is_field_1)) {
                graph_->set_tag(diagnostics::tag_severity::WARNING, "underflow");
                latency_ += 1;
                return core::draw_frame{};
            }
        }

        if (latency_ != -1) {
            CASPAR_LOG(warning) << print() << " Latency: " << latency_;
            latency_ = -1;
        }

        // Speed control: accumulate fractional rate each output tick.
        //   speed > 1.0  => skip intermediate decoded frames (fast forward)
        //   0 < speed < 1 => hold frames (slow motion)
        //   speed == 0  => freeze
        //   speed < 0   => reverse playback (batch mode):
        //                  Collects a buffer-full of decoded frames then serves them
        //                  in reverse order, issuing the next batch seek in the
        //                  background.  This amortises the keyframe-seek overhead
        //                  across buffer_capacity_ frames, giving smooth reverse on
        //                  both intra-only and long-GOP (H.264/HEVC) sources.
        {
            const double spd = speed_.load();

            if (spd < 0.0) {
                // --- Reverse (batch) ---
                speed_accum_ += -spd;
                const int frames_to_step = static_cast<int>(speed_accum_);
                speed_accum_ -= static_cast<double>(frames_to_step);

                if (frames_to_step == 0) {
                    return core::draw_frame::still(frame_);
                }

                const int64_t start_l = start_.load() != AV_NOPTS_VALUE ? start_.load() : 0LL;

                // Helper: consume the next frame from rev_frames_ and handle IN-point
                // boundary (pingpong / loop / freeze).
                auto pop_rev_frame = [&]() {
                    // Fast-reverse: skip intermediate frames when |speed| > 1
                    for (int i = 1; i < frames_to_step && rev_frames_.size() > 1; ++i)
                        rev_frames_.pop_back();

                    frame_          = rev_frames_.back().frame;
                    frame_time_     = rev_frames_.back().pts;
                    frame_duration_ = rev_frames_.back().duration;
                    rev_frames_.pop_back();

                    graph_->set_value("buffer", static_cast<double>(rev_frames_.size()) /
                                                    static_cast<double>(buffer_capacity_));

                    // Check if the next step would cross the IN point
                    if (frame_duration_ > 0 && frame_time_ - frame_duration_ < start_l) {
                        if (pingpong_) {
                            speed_       = std::abs(speed_.load());
                            speed_accum_ = 0.0;
                            rev_frames_.clear();
                            seek_        = start_l;
                            buffer_.clear();
                            buffer_cond_.notify_all();
                        } else if (loop_) {
                            const auto    dur     = duration_.load();
                            // Seek buffer_capacity_ frames before the clip end for
                            // a full first reverse batch.
                            const int64_t one_f   = av_rescale_q(1, format_tb_, TIME_BASE_Q);

                            const int64_t end_abs = dur != AV_NOPTS_VALUE
                                                        ? start_l + dur
                                                        : (input_duration_.load() != AV_NOPTS_VALUE
                                                               ? input_duration_.load() : 0LL);
                            const int64_t seek_pos = std::max(start_l,
                                end_abs - static_cast<int64_t>(buffer_capacity_) * one_f);
                            rev_frames_.clear();
                            seek_ = seek_pos;
                            buffer_.clear();
                            buffer_cond_.notify_all();
                        }
                        // else: freeze — rev_frames_ drains, pre-roll holds still
                    }
                };

                // --- Serve from existing reverse batch if available ---
                if (!rev_frames_.empty()) {
                    pop_rev_frame();
                    return frame_;
                }

                // --- rev_frames_ empty: capture the buffer as a new batch ---
                // (buffer is guaranteed non-empty because the pre-roll guard passed)
                // Capture batch in forward order; pop_back() serves highest PTS first.
                const int64_t batch_start_pts = buffer_.front().pts;
                const int64_t batch_start_dur = buffer_.front().duration;
                const int64_t batch_count     = static_cast<int64_t>(buffer_.size());
                for (const auto& f : buffer_)
                    rev_frames_.push_back(f);
                frame_flush_ = false;
                buffer_.clear();
                buffer_cond_.notify_all();

                // Pre-issue the next batch seek so the decode thread works in
                // parallel while we serve the current batch.
                // Seek BACK strictly by the actual batch_count frames from batch_start
                // so the next batch is fully BEFORE this one with no gap or overlap.
                if (batch_start_dur > 0 && batch_count > 0) {
                    const int64_t step_back_frames = (buffer_eof_ && batch_count < static_cast<int64_t>(buffer_capacity_))
                                                         ? static_cast<int64_t>(buffer_capacity_)
                                                         : batch_count;
                    const int64_t next_target =
                        batch_start_pts - step_back_frames * batch_start_dur;
                    if (next_target >= start_l) {
                        seek_ = next_target;
                        buffer_cond_.notify_all();
                    }
                }

                pop_rev_frame();
                return frame_;
            }
// --- Forward ---
            speed_accum_ += spd;
            const int frames_to_advance = static_cast<int>(speed_accum_);
            speed_accum_ -= static_cast<double>(frames_to_advance);

            if (frames_to_advance == 0) {
                // Slow motion: return current frame without consuming buffer
                return core::draw_frame::still(frame_);
            }

            // Fast-forward: discard (frames_to_advance - 1) intermediate frames
            for (int i = 1; i < frames_to_advance && buffer_.size() > 1; ++i) {
                buffer_.pop_front();
                buffer_cond_.notify_all();
            }
        }

        frame_          = buffer_[0].frame;
        frame_time_     = buffer_[0].pts;
        frame_duration_ = buffer_[0].duration;
        frame_flush_    = false;

        buffer_.pop_front();
        buffer_cond_.notify_all();

        graph_->set_value("buffer", static_cast<double>(buffer_.size()) / static_cast<double>(buffer_capacity_));

        return frame_;
    }

    void seek(int64_t time)
    {
        CASPAR_SCOPE_EXIT { update_state(); };

        int64_t target_pts = av_rescale_q(time, format_tb_, TIME_BASE_Q);

        // Clamp target_pts to valid range to prevent seeking beyond EOF (black screen)
        {
            const int64_t start_l = start_.load() != AV_NOPTS_VALUE ? start_.load() : 0LL;
            int64_t end_l = AV_NOPTS_VALUE;
            
            if (duration_.load() != AV_NOPTS_VALUE) {
                end_l = start_l + duration_.load();
            } else if (input_duration_.load() != AV_NOPTS_VALUE) {
                end_l = input_duration_.load();
            }

            if (end_l != AV_NOPTS_VALUE) {
                // If seeking exactly to or past the end, clamp to the last frame.
                // We estimate the last frame start by subtracting one frame duration.
                // If frame_duration_ is not set yet, use channel frame duration as best guess.
                const int64_t one_frame = frame_duration_ > 0 ? frame_duration_ : av_rescale_q(1, format_tb_, TIME_BASE_Q);
                if (target_pts >= end_l) {
                   target_pts = std::max(start_l, end_l - one_frame);
                }
            }
        }

        {
            boost::lock_guard<boost::mutex> lock(buffer_mutex_);
            buffer_.clear();
            rev_frames_.clear();
            
            if (speed_.load() < 0.0) {
                // If the user seeks backward to a frame, we must actually seek the decode 
                // thread to the START of the upcoming reverse batch so the requested frame
                // is the first one served (the END of the forward decoded batch).
                const int64_t start_l = start_.load() != AV_NOPTS_VALUE ? start_.load() : 0LL;
                const int64_t one_frame = av_rescale_q(1, format_tb_, TIME_BASE_Q);
                seek_ = std::max(start_l, target_pts - static_cast<int64_t>(buffer_capacity_ - 1) * one_frame);
                rev_active_ = true;
            } else {
                seek_ = target_pts;
            }
            
            // Set frame_flush_ here (under the lock) so next_frame()'s eager-consume
            // path is primed before the decode thread runs seek_internal().
            // Without this, next_frame() sees frame_flush_=false + empty buffer and
            // silently returns still(old_frame); if seeks arrive faster than the
            // decode thread processes them, frame_flush_ never transitions to true
            // and the sought frame is never consumed even after it lands in the buffer.
            frame_flush_ = true;
            // Mark as active so the initial-reverse-seek guard doesn't fire and
            // override a user-issued seek position.
            if (speed_.load() < 0.0) {
                rev_active_ = true;
                // Important: Ensure we don't leave the consumer in a state where it thinks it's
                // finished or at EOF after a seek.
                buffer_eof_ = false;
            }
            buffer_cond_.notify_all();
            graph_->set_value("buffer", static_cast<double>(buffer_.size()) / static_cast<double>(buffer_capacity_));
        }
    }

    int64_t time() const
    {
        if (frame_time_ == AV_NOPTS_VALUE) {
            // TODO (fix) How to handle NOPTS case?
            return 0;
        }

        return av_rescale_q(frame_time_, TIME_BASE_Q, format_tb_);
    }

    void loop(bool loop)
    {
        CASPAR_SCOPE_EXIT { update_state(); };

        loop_ = loop;
    }

    bool loop() const { return loop_; }

    void start(int64_t start)
    {
        CASPAR_SCOPE_EXIT { update_state(); };
        start_ = av_rescale_q(start, format_tb_, TIME_BASE_Q);
    }

    std::optional<int64_t> start() const
    {
        auto start = start_.load();
        return start != AV_NOPTS_VALUE ? av_rescale_q(start, TIME_BASE_Q, format_tb_) : std::optional<int64_t>();
    }

    void duration(int64_t duration)
    {
        CASPAR_SCOPE_EXIT { update_state(); };

        duration_ = av_rescale_q(duration, format_tb_, TIME_BASE_Q);
    }

    std::optional<int64_t> duration() const
    {
        const auto duration = duration_.load();
        if (duration == AV_NOPTS_VALUE) {
            return {};
        }
        return av_rescale_q(duration, TIME_BASE_Q, format_tb_);
    }

    std::optional<int64_t> file_duration() const
    {
        const auto input_duration = input_duration_.load();
        if (input_duration == AV_NOPTS_VALUE) {
            return {};
        }
        return av_rescale_q(input_duration, TIME_BASE_Q, format_tb_);
    }

    void speed(double spd)
    {
        const double old = speed_.exchange(spd);
        // When switching direction, discard any stale reverse batch so the next
        // call to next_frame() starts fresh in the new direction.
        if ((old < 0.0) != (spd < 0.0)) {
            boost::lock_guard<boost::mutex> lock(buffer_mutex_);
            rev_frames_.clear();
            rev_active_ = false;  // reset so initial-seek fires for new reverse session
            speed_accum_ = 0.0;
            
            if (spd < 0.0) {
                // Dynamically inflate the decode caching limit specifically for reverse mode.
                // Buffering a massive amount of sequential frames drastically offsets
                // keyframe-seek overhead and provides fully seamless backward sweeps.
                const int fps = static_cast<int>(format_desc_.fps);
                buffer_capacity_ = (fps > 0) ? fps : 30;
                if (buffer_capacity_ > 60) buffer_capacity_ = 60;
                if (buffer_capacity_ < 15) buffer_capacity_ = 15;
            } else {
                // Restore to the configured live forward latency minimum
                buffer_capacity_ = env::properties().get(L"configuration.ffmpeg.producer.buffer-depth",
                                                         std::max(1, static_cast<int>(format_desc_.fps) / 4));
            }
        }
    }
    double speed() const { return speed_.load(); }

    void pingpong(bool pp)
    {
        CASPAR_SCOPE_EXIT { update_state(); };
        pingpong_ = pp;
    }
    bool pingpong() const { return pingpong_.load(); }

  private:
    bool want_packet()
    {
        return std::any_of(decoders_.begin(), decoders_.end(), [](auto& p) { return p.second.want_packet(); });
    }

    bool schedule()
    {
        auto result = false;

        std::shared_ptr<AVPacket> packet;
        while (want_packet() && input_.try_pop(packet)) {
            result = true;

            if (!packet) {
                for (auto& p : decoders_) {
                    p.second.push(nullptr);
                }
            } else if (sources_.find(packet->stream_index) != sources_.end()) {
                auto it = decoders_.find(packet->stream_index);
                if (it != decoders_.end()) {
                    // TODO (fix): limit it->second.input.size()?
                    it->second.push(std::move(packet));
                }
            }
        }

        std::vector<int> eof;

        for (auto& p : sources_) {
            auto it = decoders_.find(p.first);
            if (it == decoders_.end()) {
                continue;
            }

            auto nb_requests = 0U;
            for (auto source : p.second) {
                nb_requests = std::max(nb_requests, av_buffersrc_get_nb_failed_requests(source));
            }

            if (nb_requests == 0) {
                continue;
            }

            auto frame = it->second.pop();
            if (!frame) {
                continue;
            }

            for (auto& source : p.second) {
                if (!frame->data[0]) {
                    FF(av_buffersrc_close(source, frame->pts, 0));
                } else {
                    // TODO (fix) Guard against overflow?
                    FF(av_buffersrc_write_frame(source, frame.get()));
                }
                result = true;
            }

            // End Of File
            if (!frame->data[0]) {
                eof.push_back(p.first);
            }
        }

        for (auto index : eof) {
            sources_.erase(index);
        }

        return result;
    }

    void seek_internal(int64_t time)
    {
        time = time != AV_NOPTS_VALUE ? time : 0;
        time = time + (input_->start_time != AV_NOPTS_VALUE ? input_->start_time : 0);

        // TODO (fix) Dont seek if time is close future.
        if (seekable_) {
            input_.seek(time);
        }
        frame_flush_ = true;
        frame_count_ = 0;
        buffer_eof_  = false;

        // Flush decoders in-place: keeps threads and codec contexts alive.
        // avcodec_flush_buffers on intra-only codecs (ProRes, NotchLC) is near-instant;
        // the very next packet produces a decoded frame with no pipeline warmup.
        // H.264/HEVC GPU decoders also stay warm through the flush.
        for (auto& [idx, dec] : decoders_) {
            dec.flush();
        }

        // Clear stale buffered frames so playback jumps to the loop start immediately
        // instead of continuing to drain pre-loop frames from the buffer.
        {
            boost::lock_guard<boost::mutex> buffer_lock(buffer_mutex_);
            buffer_.clear();
            buffer_cond_.notify_all();
        }

        reset(time);
    }

    void reset(int64_t start_time)
    {
        // Discard animated parameters — their AVFilterContext* pointers are
        // about to be invalidated by the graph rebuild.
        {
            boost::lock_guard<boost::mutex> lock(param_tween_mutex_);
            video_param_tweens_.clear();
            audio_param_tweens_.clear();
        }

        video_filter_ = Filter(vfilter_, input_, decoders_, start_time, AVMEDIA_TYPE_VIDEO, format_desc_);
        audio_filter_ = Filter(afilter_, input_, decoders_, start_time, AVMEDIA_TYPE_AUDIO, format_desc_);

        sources_.clear();
        for (auto& p : video_filter_.sources) {
            sources_[p.first].push_back(p.second);
        }
        for (auto& p : audio_filter_.sources) {
            sources_[p.first].push_back(p.second);
        }

        std::vector<int> keys;
        // Flush unused inputs.
        for (auto& p : decoders_) {
            if (sources_.find(p.first) == sources_.end()) {
                keys.push_back(p.first);
            }
        }

        for (auto& key : keys) {
            decoders_.erase(key);
        }
    }

    // -------------------------------------------------------------------------
    // VFPARAM / AFPARAM: per-frame tween tick + avfilter_graph_send_command
    // Called once per produced frame from the decode thread (run loop), after
    // both video and audio filter futures have been awaited, so no concurrent
    // filter graph access is in flight at that point.
    // -------------------------------------------------------------------------
    void apply_filter_param_tweens()
    {
        boost::lock_guard<boost::mutex> lock(param_tween_mutex_);
        char res_buf[512];

        auto apply = [&](std::map<std::string, std::map<std::string, FilterParamTween>>& tweens,
                         AVFilterGraph*                                                   fgraph) {
            if (!fgraph)
                return;
            for (auto& filter_entry : tweens) {
                const auto& fname = filter_entry.first;
                for (auto& param_entry : filter_entry.second) {
                    const auto& pname = param_entry.first;
                    auto&       tween = param_entry.second;

                    tween.tick();
                    const auto val     = tween.fetch();
                    const auto val_str = std::to_string(val);

                    const auto ret = avfilter_graph_send_command(
                        fgraph, fname.c_str(), pname.c_str(), val_str.c_str(), res_buf, sizeof(res_buf), 0);

                    if (ret < 0 && ret != AVERROR(ENOSYS)) {
                        constexpr size_t errbuf_size = 128;
                        char errbuf[errbuf_size];
                        av_strerror(ret, errbuf, errbuf_size);
                        CASPAR_LOG(warning)
                            << "[ffmpeg] VFPARAM send_command(" << fname << ", " << pname << "=" << val_str
                            << ") failed: " << errbuf;
                    }
                }
            }
        };

        apply(video_param_tweens_, video_filter_.graph.get());
        apply(audio_param_tweens_, audio_filter_.graph.get());
    }

  public:
    void set_filter_param(bool                is_video,
                          const std::string&  filter_name,
                          const std::string&  param_name,
                          double              value,
                          int                 duration_frames,
                          const std::wstring& tween_name)
    {
        boost::lock_guard<boost::mutex> lock(param_tween_mutex_);
        auto& tweens = is_video ? video_param_tweens_ : audio_param_tweens_;
        tweens[filter_name][param_name].set_target(value, duration_frames, tween_name);
    }
    
    // -------------------------------------------------------------------------
    
    void set_vfilter(const std::string& filter)
    {
        vfilter_ = filter;
        seek(time());
    }

    void set_afilter(const std::string& filter)
    {
        afilter_ = filter;
        seek(time());
    }

    // -------------------------------------------------------------------------

    std::string print() const
    {
        const int          position = std::max(static_cast<int>(time() - start().value_or(0)), 0);
        std::ostringstream str;
        str << std::fixed << std::setprecision(4) << "ffmpeg[" << name_ << "|"
            << av_q2d({position * format_tb_.num, format_tb_.den}) << "/"
            << av_q2d({static_cast<int>(duration().value_or(0LL)) * format_tb_.num, format_tb_.den}) << "]";
        return str.str();
    }
};

AVProducer::AVProducer(std::shared_ptr<core::frame_factory> frame_factory,
                       core::video_format_desc              format_desc,
                       std::string                          name,
                       std::string                          path,
                       std::optional<std::string>           vfilter,
                       std::optional<std::string>           afilter,
                       std::optional<int64_t>               start,
                       std::optional<int64_t>               seek,
                       std::optional<int64_t>               duration,
                       std::optional<bool>                  loop,
                       int                                  seekable,
                       core::frame_geometry::scale_mode     scale_mode,
                       bool                                 growing)
    : impl_(new Impl(std::move(frame_factory),
                     std::move(format_desc),
                     std::move(name),
                     std::move(path),
                     std::move(vfilter.value_or("")),
                     std::move(afilter.value_or("")),
                     std::move(start),
                     std::move(seek),
                     std::move(duration),
                     loop.value_or(false),
                     seekable,
                     scale_mode,
                     growing))
{
}

core::draw_frame AVProducer::next_frame(const core::video_field field) { return impl_->next_frame(field); }

core::draw_frame AVProducer::prev_frame(const core::video_field field) { return impl_->prev_frame(field); }

bool AVProducer::is_ready() { return impl_->is_ready(); }

AVProducer& AVProducer::seek(int64_t time)
{
    impl_->seek(time);
    return *this;
}

AVProducer& AVProducer::loop(bool loop)
{
    impl_->loop(loop);
    return *this;
}

bool AVProducer::loop() const { return impl_->loop(); }

AVProducer& AVProducer::start(int64_t start)
{
    impl_->start(start);
    return *this;
}

int64_t AVProducer::time() const { return impl_->time(); }

int64_t AVProducer::start() const { return impl_->start().value_or(0); }

AVProducer& AVProducer::duration(int64_t duration)
{
    impl_->duration(duration);
    return *this;
}

int64_t AVProducer::duration() const { return impl_->duration().value_or(std::numeric_limits<int64_t>::max()); }

AVProducer& AVProducer::set_vfilter(const std::string& filter)
{
    impl_->set_vfilter(filter);
    return *this;
}

AVProducer& AVProducer::set_afilter(const std::string& filter)
{
    impl_->set_afilter(filter);
    return *this;
}

AVProducer& AVProducer::set_filter_param(bool                is_video,
                                         const std::string&  filter_name,
                                         const std::string&  param_name,
                                         double              value,
                                         int                 duration_frames,
                                         const std::wstring& tween)
{
    impl_->set_filter_param(is_video, filter_name, param_name, value, duration_frames, tween);
    return *this;
}

AVProducer& AVProducer::speed(double spd)
{
    impl_->speed(spd);
    return *this;
}

double AVProducer::speed() const { return impl_->speed(); }

AVProducer& AVProducer::pingpong(bool pp)
{
    impl_->pingpong(pp);
    return *this;
}

bool AVProducer::pingpong() const { return impl_->pingpong(); }

core::monitor::state AVProducer::state() const
{
    boost::lock_guard<boost::mutex> lock(impl_->state_mutex_);
    return impl_->state_;
}

}} // namespace caspar::ffmpeg

