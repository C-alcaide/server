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
 *
 * This module requires the NVIDIA CUDA Toolkit (https://developer.nvidia.com/cuda-toolkit).
 * ProRes format reference: Apple Inc. "ProRes RAW White Paper" (public documentation).
 */

// prores_demuxer.cpp
// libavformat demux implementation for ProRes CUDA decoder.
// ---------------------------------------------------------------------------
#include "prores_demuxer.h"

#include <common/log.h>
#include <common/utf.h>

#ifdef _MSC_VER
#  pragma warning(push)
#  pragma warning(disable: 4244)  // conversion from int to smaller int — in FFmpeg headers
#endif
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/samplefmt.h>
#include <libavutil/timestamp.h>
}
#ifdef _MSC_VER
#  pragma warning(pop)
#endif

#include <stdexcept>

namespace caspar { namespace cuda_prores {

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

static std::string wstr_to_utf8(const std::wstring& ws)
{
    return u8(ws);
}

// ProRes codec tag to profile index.
static int tag_to_profile(uint32_t tag)
{
    switch (tag) {
        case MKTAG('a','p','c','o'): return 0;  // Proxy
        case MKTAG('a','p','c','l'): return 1;  // LT
        case MKTAG('a','p','c','n'): return 2;  // Standard
        case MKTAG('a','p','c','h'): return 3;  // HQ
        case MKTAG('a','p','4','h'): return 4;  // 4444
        case MKTAG('a','p','4','x'): return 4;  // 4444 XQ (treat same)
        default:                     return 3;  // fallback: HQ
    }
}

// ---------------------------------------------------------------------------
// struct impl
// ---------------------------------------------------------------------------

struct ProResDemuxer::impl
{
    AVFormatContext* fmt_ctx   = nullptr;
    int              video_idx = -1;      // video stream index
    int              profile   = 3;       // ProRes profile (from codec tag)
    int              num_den[2] = {25, 1};
    int64_t          total_frames_cached = -1;
    bool             looped = false;      // we've looped at least once (for logging)

    // ── Audio ──────────────────────────────────────────────────────────────
    int              audio_idx          = -1;
    int              audio_sample_rate_ = 48000;
    int              audio_channels_    = 0;
    AVCodecContext*  audio_codec_ctx_   = nullptr;
    AVFrame*         audio_frame_       = nullptr;
    std::vector<int32_t> audio_buf_;   // decoded int32_t interleaved, since last video frame

    impl() = default;

    ~impl()
    {
        if (audio_frame_)     { av_frame_free        (&audio_frame_);       audio_frame_     = nullptr; }
        if (audio_codec_ctx_) { avcodec_free_context (&audio_codec_ctx_);   audio_codec_ctx_ = nullptr; }
        if (fmt_ctx)          avformat_close_input(&fmt_ctx);
    }

    void open(const std::string& path)
    {
        if (avformat_open_input(&fmt_ctx, path.c_str(), nullptr, nullptr) < 0)
            throw std::runtime_error("avformat_open_input failed: " + path);

        if (avformat_find_stream_info(fmt_ctx, nullptr) < 0)
            throw std::runtime_error("avformat_find_stream_info failed");

        // Find the first video stream with a ProRes codec.
        for (unsigned i = 0; i < fmt_ctx->nb_streams; ++i) {
            AVStream* st = fmt_ctx->streams[i];
            if (st->codecpar->codec_type != AVMEDIA_TYPE_VIDEO)
                continue;
            if (st->codecpar->codec_id != AV_CODEC_ID_PRORES)
                continue;
            video_idx = (int)i;
            profile   = tag_to_profile((uint32_t)st->codecpar->codec_tag);

            AVRational fr = st->avg_frame_rate;
            if (fr.den > 0 && fr.num > 0) {
                num_den[0] = fr.num;
                num_den[1] = fr.den;
            }
            if (st->nb_frames > 0)
                total_frames_cached = st->nb_frames;
            break;
        }

        if (video_idx < 0)
            throw std::runtime_error("No ProRes video stream in: " + path);

        // ── Find & open audio stream ──────────────────────────────────────
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

        CASPAR_LOG(info) << L"[prores_demuxer] Opened: "
                         << path.c_str()
                         << L"  profile=" << profile
                         << L"  fps=" << num_den[0] << L"/" << num_den[1]
                         << (audio_idx >= 0
                             ? (L"  audio ch=" + std::to_wstring(audio_channels_)
                                + L" rate=" + std::to_wstring(audio_sample_rate_))
                             : L"  no audio");
    }

    ProResPacket read_next()
    {
        AVPacket* pkt = av_packet_alloc();
        if (!pkt)
            throw std::runtime_error("av_packet_alloc");

        int consecutive_errors = 0;
        for (;;) {
            int ret = av_read_frame(fmt_ctx, pkt);
            if (ret == AVERROR_EOF) {
                av_packet_free(&pkt);
                ProResPacket out;
                out.is_eof = true;
                return out;
            }
            if (ret < 0) {
                // Transient errors (e.g. EAGAIN after a seek) — skip this packet
                // and try the next one rather than signalling a false EOF.
                // Bail out after too many consecutive errors to avoid an infinite loop
                // on a truly unreadable file.
                char errbuf[64];
                av_strerror(ret, errbuf, sizeof(errbuf));
                CASPAR_LOG(debug) << L"[prores_demuxer] av_read_frame transient error: " << errbuf << L" — retrying";
                av_packet_unref(pkt);
                if (++consecutive_errors > 32) {
                    CASPAR_LOG(warning) << L"[prores_demuxer] Too many consecutive read errors — signalling EOF";
                    av_packet_free(&pkt);
                    ProResPacket out;
                    out.is_eof = true;
                    return out;
                }
                continue;
            }
            consecutive_errors = 0;
            // Decode audio packets and buffer them for the next video frame.
            if (pkt->stream_index == audio_idx && audio_codec_ctx_) {
                decode_audio_pkt(pkt);
                av_packet_unref(pkt);
                continue;
            }
            if (pkt->stream_index != video_idx) {
                av_packet_unref(pkt);
                continue;
            }

            ProResPacket out;
            out.pts          = pkt->pts;
            out.audio_samples = std::move(audio_buf_);
            out.data.assign(pkt->data, pkt->data + pkt->size);
            av_packet_free(&pkt);
            return out;
        }
    }

    ProResPacket seek_and_loop()
    {
        if (!looped)
            CASPAR_LOG(debug) << L"[prores_demuxer] EOF reached — looping";
        looped = true;

        audio_buf_.clear();  // discard audio accumulated before the loop point
        av_seek_frame(fmt_ctx, video_idx, 0, AVSEEK_FLAG_BACKWARD);
        if (audio_codec_ctx_)
            avcodec_flush_buffers(audio_codec_ctx_);

        return read_next();
    }
    bool seek_to_frame(int64_t frame_number)
    {
        AVStream* st   = fmt_ctx->streams[video_idx];
        // Convert frame index to stream PTS:
        //   target_pts = frame_number * (1/fps) / time_base
        //              = frame_number * fps.den * time_base.den
        //                            / (fps.num * time_base.num)
        AVRational frame_dur = av_inv_q(st->avg_frame_rate);  // {fps.den, fps.num}
        int64_t target_pts   = av_rescale_q(frame_number, frame_dur, st->time_base);

        audio_buf_.clear();
        if (audio_codec_ctx_)
            avcodec_flush_buffers(audio_codec_ctx_);

        int ret = av_seek_frame(fmt_ctx, video_idx, target_pts, AVSEEK_FLAG_BACKWARD);
        if (ret < 0)
            CASPAR_LOG(warning) << L"[prores_demuxer] seek_to_frame " << frame_number << L" failed: " << ret;
        return ret >= 0;
    }
    // ── Audio helpers ─────────────────────────────────────────────────────
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
// ProResDemuxer
// ---------------------------------------------------------------------------

ProResDemuxer::ProResDemuxer(const std::wstring& path)
    : impl_(new impl())
{
    impl_->open(wstr_to_utf8(path));
}

ProResDemuxer::~ProResDemuxer() = default;

ProResPacket ProResDemuxer::read_packet()
{
    return impl_->read_next();
}

bool ProResDemuxer::valid() const
{
    return impl_ && impl_->video_idx >= 0;
}

void ProResDemuxer::frame_rate(int& num, int& den) const
{
    num = impl_->num_den[0];
    den = impl_->num_den[1];
}

int64_t ProResDemuxer::total_frames() const
{
    return impl_->total_frames_cached;
}

int64_t ProResDemuxer::duration_us() const
{
    if (!impl_->fmt_ctx) return -1;
    if (impl_->fmt_ctx->duration == AV_NOPTS_VALUE) return -1;
    return impl_->fmt_ctx->duration;  // already in AV_TIME_BASE (microseconds)
}

bool ProResDemuxer::seek_to_frame(int64_t frame_number)
{
    return impl_->seek_to_frame(frame_number);
}

// ---------------------------------------------------------------------------
// Static: parse icpf frame header
// ---------------------------------------------------------------------------

bool ProResDemuxer::parse_frame_info(const uint8_t* data, int size,
                                     ProResFrameInfo& out)
{
    // Minimum frame size: 8-byte box + 2-byte hdr_size + 18 bytes minimum hdr
    if (size < 28)
        return false;

    // Skip 4-byte frame_size + 4-byte 'icpf'
    const uint8_t* fhdr = data + 8;  // frame header starts here
    const int fhdr_max  = size - 8;

    // [0..1] frame_header_size (BE16)
    int hdr_size = (fhdr[0] << 8) | fhdr[1];
    if (hdr_size < 18 || hdr_size > fhdr_max)
        return false;

    // Width / height
    out.width  = (fhdr[8] << 8) | fhdr[9];
    out.height = (fhdr[10] << 8) | fhdr[11];

    // frame_type at bits [3:2] of byte 12 (see FFmpeg proresdec.c)
    out.frame_type = (fhdr[12] >> 2) & 3;

    // Color metadata: bytes 14, 15, 16 (same offsets as FFmpeg)
    out.color_primaries = fhdr[14];
    out.transfer_func   = fhdr[15];
    out.color_matrix    = fhdr[16];

    // Picture section starts after frame header.
    const uint8_t* phdr = fhdr + hdr_size;
    const int      phdr_avail = size - 8 - hdr_size;
    if (phdr_avail < 8)
        return false;

    // Picture header: byte[0] contains (hdr_size_in_bytes << 3)?
    // From FFmpeg decode_picture_header: hdr_size = buf[0] >> 3
    int pic_hdr_size = phdr[0] >> 3;
    if (pic_hdr_size < 8 || pic_hdr_size > phdr_avail)
        return false;

    // log2_slice_mb_width from phdr[7]
    int log2_smb_w = phdr[7] >> 4;
    int log2_smb_h = phdr[7] & 0xF;
    if (log2_smb_h != 0)
        return false;  // only horizontal slicing supported

    int mbs_per_slice = 1 << log2_smb_w;  // power of 2
    int mb_width      = (out.width  + 15) / 16;
    int mb_height     = (out.height + 15) / 16;
    if (out.frame_type != 0)
        mb_height = (out.height + 31) / 32;  // interlaced: field height

    int slices_per_row = (mb_width + mbs_per_slice - 1) / mbs_per_slice;
    int num_slices     = slices_per_row * mb_height;

    out.mbs_per_slice  = mbs_per_slice;
    out.slices_per_row = slices_per_row;
    out.num_slices     = num_slices;

    return (out.width > 0 && out.height > 0 && num_slices > 0);
}

// ---------------------------------------------------------------------------
// Audio accessors
// ---------------------------------------------------------------------------
bool ProResDemuxer::has_audio()         const { return impl_->audio_idx >= 0; }
int  ProResDemuxer::audio_sample_rate() const { return impl_->audio_sample_rate_; }
int  ProResDemuxer::audio_channels()    const { return impl_->audio_channels_; }
int  ProResDemuxer::profile()           const { return impl_->profile; }

}} // namespace caspar::cuda_prores
