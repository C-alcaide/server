/*
 * Copyright (c) 2025 CasparCG Contributors
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

// hap_demuxer.cpp
// libavformat demux implementation for the HAP decoder.
// ---------------------------------------------------------------------------
#include "hap_demuxer.h"

#include <common/log.h>
#include <common/utf.h>

#ifdef _MSC_VER
#  pragma warning(push)
#  pragma warning(disable: 4244)
#endif
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/samplefmt.h>
}
#ifdef _MSC_VER
#  pragma warning(pop)
#endif

#include <stdexcept>

namespace caspar { namespace hap {

struct HapDemuxer::impl
{
    AVFormatContext* fmt_ctx   = nullptr;
    int              video_idx = -1;
    int              num_den[2] = {25, 1};
    int64_t          total_frames_cached = -1;

    // Audio
    int              audio_idx          = -1;
    int              audio_sample_rate_ = 48000;
    int              audio_channels_    = 0;
    AVCodecContext*  audio_codec_ctx_   = nullptr;
    AVFrame*         audio_frame_       = nullptr;
    std::vector<int32_t> audio_buf_;

    // Custom AVIO for large-buffer reads.
    struct IoCtx {
        FILE*  fp = nullptr;
        static int read_cb(void* opaque, uint8_t* buf, int size) {
            IoCtx* c = static_cast<IoCtx*>(opaque);
            size_t n = fread(buf, 1, (size_t)size, c->fp);
            if (n == 0) return AVERROR_EOF;
            return (int)n;
        }
        static int64_t seek_cb(void* opaque, int64_t offset, int whence) {
            IoCtx* c = static_cast<IoCtx*>(opaque);
            if (whence == AVSEEK_SIZE) return -1;
#ifdef _MSC_VER
            int r = _fseeki64(c->fp, offset, whence);
#else
            int r = fseeko(c->fp, (off_t)offset, whence);
#endif
            if (r != 0) return -1;
#ifdef _MSC_VER
            return _ftelli64(c->fp);
#else
            return (int64_t)ftello(c->fp);
#endif
        }
    };
    IoCtx* ioctx_ = nullptr;

    impl() = default;

    ~impl()
    {
        if (audio_frame_)     { av_frame_free(&audio_frame_);            audio_frame_     = nullptr; }
        if (audio_codec_ctx_) { avcodec_free_context(&audio_codec_ctx_); audio_codec_ctx_ = nullptr; }
        if (fmt_ctx) {
            AVIOContext* pb = fmt_ctx->pb;
            if (pb) fmt_ctx->pb = nullptr;
            avformat_close_input(&fmt_ctx);
            if (pb) avio_context_free(&pb);
        }
        if (ioctx_) {
            if (ioctx_->fp) fclose(ioctx_->fp);
            delete ioctx_;
            ioctx_ = nullptr;
        }
    }

    void open(const std::string& path)
    {
        constexpr int IO_BUF = 4 * 1024 * 1024; // 4 MB

        fmt_ctx = avformat_alloc_context();
        if (!fmt_ctx)
            throw std::runtime_error("[hap_demuxer] avformat_alloc_context failed");

        uint8_t* iobuf = static_cast<uint8_t*>(av_malloc(IO_BUF));
        if (!iobuf) {
            avformat_free_context(fmt_ctx); fmt_ctx = nullptr;
            throw std::runtime_error("[hap_demuxer] av_malloc IO buffer failed");
        }

        ioctx_ = new IoCtx;
#ifdef _MSC_VER
        ioctx_->fp = _fsopen(path.c_str(), "rb", _SH_DENYNO);
#else
        ioctx_->fp = fopen(path.c_str(), "rb");
#endif
        if (!ioctx_->fp) {
            delete ioctx_; ioctx_ = nullptr;
            av_free(iobuf);
            avformat_free_context(fmt_ctx); fmt_ctx = nullptr;
            throw std::runtime_error("[hap_demuxer] Cannot open file: " + path);
        }

        AVIOContext* avio = avio_alloc_context(
            iobuf, IO_BUF, 0, ioctx_,
            IoCtx::read_cb, nullptr, IoCtx::seek_cb);
        if (!avio) {
            fclose(ioctx_->fp); delete ioctx_; ioctx_ = nullptr;
            av_free(iobuf);
            avformat_free_context(fmt_ctx); fmt_ctx = nullptr;
            throw std::runtime_error("[hap_demuxer] avio_alloc_context failed");
        }

        fmt_ctx->pb = avio;

        if (avformat_open_input(&fmt_ctx, path.c_str(), nullptr, nullptr) < 0)
            throw std::runtime_error("[hap_demuxer] avformat_open_input failed: " + path);
        if (avformat_find_stream_info(fmt_ctx, nullptr) < 0)
            throw std::runtime_error("[hap_demuxer] avformat_find_stream_info failed");

        for (unsigned i = 0; i < fmt_ctx->nb_streams; ++i) {
            AVStream* st = fmt_ctx->streams[i];
            if (st->codecpar->codec_type != AVMEDIA_TYPE_VIDEO) continue;
            if (st->codecpar->codec_id   != AV_CODEC_ID_HAP) continue;
            video_idx = (int)i;
            AVRational fr = st->avg_frame_rate;
            if (fr.den > 0 && fr.num > 0) { num_den[0] = fr.num; num_den[1] = fr.den; }
            if (st->nb_frames > 0) total_frames_cached = st->nb_frames;
            break;
        }

        if (video_idx < 0)
            throw std::runtime_error("[hap_demuxer] No HAP video stream in: " + path);

        // Audio
        for (unsigned ii = 0; ii < fmt_ctx->nb_streams; ++ii) {
            AVStream* ast = fmt_ctx->streams[ii];
            if (ast->codecpar->codec_type != AVMEDIA_TYPE_AUDIO) continue;
            const AVCodec* acodec = avcodec_find_decoder(ast->codecpar->codec_id);
            if (!acodec) continue;
            AVCodecContext* acc = avcodec_alloc_context3(acodec);
            if (!acc) continue;
            if (avcodec_parameters_to_context(acc, ast->codecpar) < 0 ||
                avcodec_open2(acc, acodec, nullptr) < 0) {
                avcodec_free_context(&acc);
                continue;
            }
            audio_idx          = (int)ii;
            audio_sample_rate_ = ast->codecpar->sample_rate;
            audio_channels_    = ast->codecpar->ch_layout.nb_channels;
            audio_codec_ctx_   = acc;
            audio_frame_       = av_frame_alloc();
            break;
        }

        CASPAR_LOG(info) << L"[hap_demuxer] Opened: " << path.c_str()
                         << L"  fps=" << num_den[0] << L"/" << num_den[1]
                         << (audio_idx >= 0
                             ? (L"  audio ch=" + std::to_wstring(audio_channels_)
                                + L" rate="    + std::to_wstring(audio_sample_rate_))
                             : L"  no audio");
    }

    HapPacket read_next()
    {
        AVPacket* pkt = av_packet_alloc();
        if (!pkt) throw std::runtime_error("av_packet_alloc");

        for (;;) {
            int ret = av_read_frame(fmt_ctx, pkt);
            if (ret == AVERROR_EOF || ret < 0) {
                av_packet_free(&pkt);
                HapPacket out; out.is_eof = true;
                return out;
            }
            if (pkt->stream_index == audio_idx && audio_codec_ctx_) {
                decode_audio_pkt(pkt);
                av_packet_unref(pkt);
                continue;
            }
            if (pkt->stream_index != video_idx) {
                av_packet_unref(pkt);
                continue;
            }

            HapPacket out;
            out.pts = pkt->pts;
            out.audio_samples = std::move(audio_buf_);

            // HAP packets from FFmpeg's demuxer contain the raw HAP section data
            // (section header + optional Snappy-compressed DXT payload).
            // Transfer ownership zero-copy.
            AVPacket* owned = av_packet_alloc();
            if (owned) {
                av_packet_move_ref(owned, pkt);
                out.pkt_handle = std::shared_ptr<void>(owned,
                    [](void* p) { AVPacket* pp = static_cast<AVPacket*>(p); av_packet_free(&pp); });
                out.raw_payload      = owned->data;
                out.raw_payload_size = static_cast<size_t>(owned->size);
            } else {
                out.data.assign(pkt->data, pkt->data + pkt->size);
            }

            av_packet_free(&pkt);
            return out;
        }
    }

    bool seek_to_frame(int64_t frame_number)
    {
        AVStream* st       = fmt_ctx->streams[video_idx];
        AVRational frame_dur = av_inv_q(st->avg_frame_rate);
        int64_t target_pts   = av_rescale_q(frame_number, frame_dur, st->time_base);

        audio_buf_.clear();
        if (audio_codec_ctx_)
            avcodec_flush_buffers(audio_codec_ctx_);

        int ret = av_seek_frame(fmt_ctx, video_idx, target_pts, AVSEEK_FLAG_BACKWARD);
        if (ret < 0)
            CASPAR_LOG(warning) << L"[hap_demuxer] seek_to_frame " << frame_number << L" failed";
        return ret >= 0;
    }

    void decode_audio_pkt(AVPacket* pkt)
    {
        if (avcodec_send_packet(audio_codec_ctx_, pkt) < 0) return;
        while (avcodec_receive_frame(audio_codec_ctx_, audio_frame_) >= 0) {
            append_audio_frame(audio_frame_);
            av_frame_unref(audio_frame_);
        }
    }

    void append_audio_frame(AVFrame* frame)
    {
        const int n  = frame->nb_samples;
        const int ch = audio_channels_;
        switch (frame->format) {
        case AV_SAMPLE_FMT_S16: {
            const auto* d = (const int16_t*)frame->data[0];
            for (int i = 0; i < n * ch; i++) audio_buf_.push_back((int32_t)d[i] << 16);
            break; }
        case AV_SAMPLE_FMT_S16P:
            for (int s = 0; s < n; s++)
                for (int c = 0; c < ch; c++)
                    audio_buf_.push_back((int32_t)((const int16_t*)frame->data[c])[s] << 16);
            break;
        case AV_SAMPLE_FMT_S32: {
            const auto* d = (const int32_t*)frame->data[0];
            audio_buf_.insert(audio_buf_.end(), d, d + n * ch);
            break; }
        case AV_SAMPLE_FMT_S32P:
            for (int s = 0; s < n; s++)
                for (int c = 0; c < ch; c++)
                    audio_buf_.push_back(((const int32_t*)frame->data[c])[s]);
            break;
        case AV_SAMPLE_FMT_FLT: {
            const auto* d = (const float*)frame->data[0];
            for (int i = 0; i < n * ch; i++)
                audio_buf_.push_back((int32_t)((double)d[i] * 2147483647.0));
            break; }
        case AV_SAMPLE_FMT_FLTP:
            for (int s = 0; s < n; s++)
                for (int c = 0; c < ch; c++)
                    audio_buf_.push_back(
                        (int32_t)((double)((const float*)frame->data[c])[s] * 2147483647.0));
            break;
        default: break;
        }
    }
};

// ---------------------------------------------------------------------------
// HapDemuxer
// ---------------------------------------------------------------------------

HapDemuxer::HapDemuxer(const std::wstring& path)
    : impl_(std::make_unique<impl>()) { impl_->open(u8(path)); }

HapDemuxer::~HapDemuxer() = default;

bool          HapDemuxer::valid()       const { return impl_ && impl_->video_idx >= 0; }
HapPacket     HapDemuxer::read_packet()       { return impl_->read_next(); }
bool          HapDemuxer::seek_to_frame(int64_t f) { return impl_->seek_to_frame(f); }
bool          HapDemuxer::has_audio()   const { return impl_->audio_idx >= 0; }
int           HapDemuxer::audio_sample_rate() const { return impl_->audio_sample_rate_; }
int           HapDemuxer::audio_channels()    const { return impl_->audio_channels_; }

HapFrameInfo HapDemuxer::frame_info() const
{
    HapFrameInfo fi;
    if (impl_ && impl_->fmt_ctx && impl_->video_idx >= 0) {
        const AVCodecParameters* cp = impl_->fmt_ctx->streams[impl_->video_idx]->codecpar;
        fi.width  = cp->width;
        fi.height = cp->height;
    }
    return fi;
}

void HapDemuxer::frame_rate(int& num, int& den) const
    { num = impl_->num_den[0]; den = impl_->num_den[1]; }

int64_t HapDemuxer::total_frames() const { return impl_->total_frames_cached; }

int64_t HapDemuxer::duration_us() const
{
    if (!impl_->fmt_ctx) return -1;
    if (impl_->fmt_ctx->duration == AV_NOPTS_VALUE) return -1;
    return impl_->fmt_ctx->duration;
}

}} // namespace caspar::hap
