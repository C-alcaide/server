// mxf_muxer.cpp
// MXF container muxer for ProRes + PCM32 audio via libavformat + AsyncFileWriter.
#include "mxf_muxer.h"

#include <cerrno>
#include <cstring>
#include <cstdio>
#include <stdexcept>

// ─── AVIO callbacks ──────────────────────────────────────────────────────────

// Called by libavformat for every output buffer it wants written.
int MxfMuxer::avio_write_packet(void *opaque, const uint8_t *buf, int buf_size)
{
    auto *self = static_cast<MxfMuxer*>(opaque);
    if (!self->writer_.write(buf, (size_t)buf_size))
        return AVERROR(EIO);
    self->logical_offset_ += buf_size;
    return buf_size;
}

// Called for seeks (needed for MXF header / index patch-back).
// We handle SEEK_HOLE and SEEK_DATA as unsupported; AVIOContext.seekable handles the rest.
int64_t MxfMuxer::avio_seek(void *opaque, int64_t offset, int whence)
{
    // MXF OP1a requires seekable output for the footer partition.
    // Our AsyncFileWriter supports sequential writes only; for true seekability
    // during recording we would need to keep offsets and patch at close().
    // For now: return the current offset for SEEK_CUR, reject all else so that
    // libavformat falls back to a non-seekable mux path (OP1a still works but
    // without random-access footer).
    auto *self = static_cast<MxfMuxer*>(opaque);
    if (whence == SEEK_CUR) return self->logical_offset_;
    if (whence == AVSEEK_SIZE) return -1; // unknown total size
    // For absolute seeks (after recording) we would need to reopen the file.
    // Signal non-seekable for now.
    (void)self; (void)offset;
    return AVERROR(ESPIPE);
}

// ─── open() ──────────────────────────────────────────────────────────────────

bool MxfMuxer::open(const wchar_t       *path,
                    const MxfVideoTrackInfo &video,
                    const MxfAudioTrackInfo &audio,
                    const char           *format_name,
                    size_t                sector_size)
{
    video_ = video;
    audio_ = audio;

    // Open the async file writer
    if (!writer_.open(path, AsyncFileWriter::kDefaultSlotBytes, sector_size))
        return false;

    // --- libavformat setup ------------------------------------------------
    const AVOutputFormat *ofmt = av_guess_format(format_name, nullptr, nullptr);
    if (!ofmt) {
        fprintf(stderr, "[MxfMuxer] av_guess_format(\"%s\") failed\n", format_name);
        return false;
    }

    if (avformat_alloc_output_context2(&fmt_ctx_, ofmt, nullptr, nullptr) < 0)
        return false;

    // Custom AVIO context backed by AsyncFileWriter
    constexpr int kAvioBufSize = 4 * 1024 * 1024; // 4 MB avio buffer
    avio_buf_ = static_cast<uint8_t*>(av_malloc(kAvioBufSize));
    if (!avio_buf_) return false;

    avio_ctx_ = avio_alloc_context(
        avio_buf_, kAvioBufSize,
        1,          // write_flag
        this,       // opaque
        nullptr,    // read_packet
        avio_write_packet,
        avio_seek);
    if (!avio_ctx_) return false;

    // Mark as non-seekable to avoid libavformat trying random seeks during recording.
    avio_ctx_->seekable = 0;
    fmt_ctx_->pb = avio_ctx_;
    fmt_ctx_->flags |= AVFMT_FLAG_CUSTOM_IO;

    // --- Video stream -----------------------------------------------------
    video_stream_ = avformat_new_stream(fmt_ctx_, nullptr);
    if (!video_stream_) return false;

    video_stream_->codecpar->codec_type  = AVMEDIA_TYPE_VIDEO;
    video_stream_->codecpar->codec_id    = AV_CODEC_ID_PRORES;
    video_stream_->codecpar->width       = video_.width;
    video_stream_->codecpar->height      = video_.height;
    video_stream_->codecpar->format      = AV_PIX_FMT_YUV422P10;
    video_stream_->codecpar->color_primaries       = (AVColorPrimaries)video_.color.color_primaries;
    video_stream_->codecpar->color_trc             = (AVColorTransferCharacteristic)video_.color.transfer_characteristic;
    video_stream_->codecpar->color_space           = (AVColorSpace)video_.color.color_space;

    // ProRes codec tag from fourcc
    video_stream_->codecpar->codec_tag = video_.prores_fourcc;

    video_stream_->time_base = { video_.frame_rate.den, video_.frame_rate.num };

    // --- Audio stream (PCM32 LE) ------------------------------------------
    audio_stream_ = avformat_new_stream(fmt_ctx_, nullptr);
    if (!audio_stream_) return false;

    audio_stream_->codecpar->codec_type    = AVMEDIA_TYPE_AUDIO;
    audio_stream_->codecpar->codec_id      = AV_CODEC_ID_PCM_S32LE;
    audio_stream_->codecpar->sample_rate   = audio_.sample_rate;
    audio_stream_->codecpar->ch_layout.nb_channels = audio_.channels;
    audio_stream_->codecpar->format        = AV_SAMPLE_FMT_S32;
    audio_stream_->codecpar->bits_per_coded_sample = 32;
    audio_stream_->time_base = { 1, audio_.sample_rate };

    // --- Write file header ------------------------------------------------
    // NOTE: avformat_write_header() is deferred to the first write_video() call
    // so that set_start_timecode() can be called after open() but before
    // the first frame.  The header_written_ flag guards this.
    return true;
}

// ─── set_start_timecode() ────────────────────────────────────────────────────

void MxfMuxer::set_start_timecode(const SmpteTimecode &tc)
{
    start_tc_ = tc;
    tc_set_   = true;
}

// ─── write_header_if_needed() ────────────────────────────────────────────────
// Internal helper: write the MXF file header on the first write_video() call.

static bool write_mxf_header(AVFormatContext *fmt_ctx, AVStream *video_stream,
                              bool tc_set, const SmpteTimecode &start_tc)
{
    if (tc_set && start_tc.valid) {
        char tc_str[16];
        start_tc.to_string(tc_str, sizeof(tc_str));
        av_dict_set(&video_stream->metadata, "timecode", tc_str, 0);
    }
    AVDictionary *opts = nullptr;
    if (avformat_write_header(fmt_ctx, &opts) < 0) {
        av_dict_free(&opts);
        fprintf(stderr, "[MxfMuxer] avformat_write_header failed\n");
        return false;
    }
    av_dict_free(&opts);
    return true;
}

// ─── write_video() ───────────────────────────────────────────────────────────

bool MxfMuxer::write_video(const uint8_t *data, size_t size, int64_t pts)
{
    if (!fmt_ctx_ || !video_stream_) return false;

    // Write the MXF header on the first call (deferred to allow set_start_timecode)
    if (!header_written_) {
        if (!write_mxf_header(fmt_ctx_, video_stream_, tc_set_, start_tc_))
            return false;
        header_written_ = true;
    }


    AVPacket *pkt = av_packet_alloc();
    if (!pkt) return false;

    // Wrap data in a packet without copying (data must stay valid until av_write_frame returns)
    pkt->data = const_cast<uint8_t*>(data);
    pkt->size = static_cast<int>(size);
    pkt->pts  = pts;
    pkt->dts  = pts;
    pkt->flags = AV_PKT_FLAG_KEY; // ProRes frames are all I-frames
    pkt->stream_index = video_stream_->index;
    av_packet_rescale_ts(pkt, { video_.frame_rate.den, video_.frame_rate.num },
                         video_stream_->time_base);

    int ret = av_write_frame(fmt_ctx_, pkt);
    pkt->data = nullptr; pkt->size = 0; // we own the data, don't let ff free it
    av_packet_free(&pkt);

    if (ret < 0) {
        char err[128]; av_strerror(ret, err, sizeof(err));
        fprintf(stderr, "[MxfMuxer] av_write_frame (video) failed: %s\n", err);
        return false;
    }
    ++video_pts_;
    return true;
}

// ─── write_audio() ───────────────────────────────────────────────────────────

bool MxfMuxer::write_audio(const int32_t *samples, int num_samples)
{
    if (!fmt_ctx_ || !audio_stream_) return false;

    // Accumulate samples to handle any codec frame-size requirement.
    // For raw PCM (AV_CODEC_ID_PCM_S32LE) frame_size = 0, so flush immediately.
    const size_t byte_size = (size_t)num_samples * audio_.channels * sizeof(int32_t);

    AVPacket *pkt = av_packet_alloc();
    if (!pkt) return false;

    pkt->data = reinterpret_cast<uint8_t*>(const_cast<int32_t*>(samples));
    pkt->size = static_cast<int>(byte_size);
    pkt->pts  = audio_pts_;
    pkt->dts  = audio_pts_;
    pkt->flags = AV_PKT_FLAG_KEY;
    pkt->stream_index = audio_stream_->index;

    int ret = av_write_frame(fmt_ctx_, pkt);
    pkt->data = nullptr; pkt->size = 0;
    av_packet_free(&pkt);

    if (ret < 0) {
        char err[128]; av_strerror(ret, err, sizeof(err));
        fprintf(stderr, "[MxfMuxer] av_write_frame (audio) failed: %s\n", err);
        return false;
    }
    audio_pts_ += num_samples;
    return true;
}

// ─── close() ─────────────────────────────────────────────────────────────────

bool MxfMuxer::close()
{
    if (!fmt_ctx_) return false;

    // If no frames were written, still write the header before the trailer
    if (!header_written_) {
        write_mxf_header(fmt_ctx_, video_stream_, tc_set_, start_tc_);
        header_written_ = true;
    }

    // Flush libavformat (writes footer partition if applicable)
    av_write_trailer(fmt_ctx_);

    // The trailer goes through avio_ctx_ → avio_write_packet → writer_.write()
    // Flush the AVIO context buffers
    if (avio_ctx_) avio_flush(avio_ctx_);

    // Drain all in-flight AsyncFileWriter writes, then close the file
    // (no tail needed — MXF header/footer are already emitted via avformat)
    writer_.close(nullptr, 0);

    // Free libavformat objects
    if (avio_ctx_) {
        // avio_buf_ is managed by avio_alloc_context — freed here
        avio_context_free(&avio_ctx_);
        avio_buf_ = nullptr;
    }
    avformat_free_context(fmt_ctx_);
    fmt_ctx_       = nullptr;
    video_stream_  = nullptr;
    audio_stream_  = nullptr;
    return true;
}

// ─── Destructor ──────────────────────────────────────────────────────────────

MxfMuxer::~MxfMuxer()
{
    if (fmt_ctx_) close();
}
