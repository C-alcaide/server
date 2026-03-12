// mxf_muxer.h
// MXF container muxer for ProRes + PCM audio, backed by AsyncFileWriter.
//
// Uses libavformat to build the MXF container (OP-Atom or OP1a) and routes
// all writes through a custom AVIOContext that calls AsyncFileWriter so the
// data path remains unbuffered / IOCP-driven on NVMe.
//
// Usage
// ─────────────────────────────────────────────────────────────────────────────
//   MxfMuxer mxf;
//   MxfVideoTrackInfo v{ 3840, 2160, {1,25}, fourcc_HQ, color_SDR };
//   MxfAudioTrackInfo a{ 16, 48000 };
//   mxf.open(L"out.mxf", v, a);
//   while (capturing) {
//       mxf.write_video(prores_frame_data, prores_frame_size, frame_number);
//       mxf.write_audio(pcm32_samples, sample_count_this_frame);
//   }
//   mxf.close();
#pragma once

#ifdef __cplusplus
extern "C" {
#endif
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/opt.h>
#ifdef __cplusplus
}
#endif

#include "../storage/async_file_writer.h"

#include <string>
#include <vector>
#include <cstdint>

struct MxfColorInfo {
    int color_primaries;      // AVCOL_PRI_*
    int transfer_characteristic; // AVCOL_TRC_*
    int color_space;          // AVCOL_SPC_*
};

static inline MxfColorInfo MXF_COLOR_SDR_709  = { 1, 1, 1 };
static inline MxfColorInfo MXF_COLOR_HDR_HLG  = { 9, 18, 9 };
static inline MxfColorInfo MXF_COLOR_HDR_PQ   = { 9, 16, 9 };

struct MxfVideoTrackInfo {
    int      width, height;
    struct { int num, den; } frame_rate; // e.g. {25, 1}
    uint32_t prores_fourcc;              // 'apco' / 'apcl' / 'apcn' / 'apch' / 'ap4h'
    MxfColorInfo color;
};

struct MxfAudioTrackInfo {
    int channels;    // typically 16
    int sample_rate; // 48000
};

class MxfMuxer {
public:
    MxfMuxer() = default;
    ~MxfMuxer();

    // Open output MXF file.
    // format_name: "mxf_opatom" for OP-Atom (one track per file) or
    //              "mxf" for OP1a (interleaved video + audio in one file).
    bool open(const wchar_t *path,
              const MxfVideoTrackInfo &video,
              const MxfAudioTrackInfo &audio,
              const char *format_name  = "mxf",
              size_t      sector_size  = 4096);

    bool write_video(const uint8_t *data, size_t size, int64_t pts);
    bool write_audio(const int32_t *samples, int num_samples);
    bool close();

private:
    static int  avio_write_packet(void *opaque, const uint8_t *buf, int buf_size);
    static int64_t avio_seek(void *opaque, int64_t offset, int whence);

    AVFormatContext *fmt_ctx_     = nullptr;
    AVIOContext     *avio_ctx_    = nullptr;
    uint8_t         *avio_buf_   = nullptr;
    AVStream        *video_stream_ = nullptr;
    AVStream        *audio_stream_ = nullptr;

    AsyncFileWriter  writer_;
    int64_t          logical_offset_ = 0; // tracks seek position for avio

    MxfVideoTrackInfo video_;
    MxfAudioTrackInfo audio_;
    int64_t  video_pts_ = 0;
    int64_t  audio_pts_ = 0;

    // Audio sample accumulator (MXF may require specific frame sizes)
    std::vector<int32_t> audio_buf_;
    int  audio_frame_size_ = 0; // samples per audio frame (codec_ctx->frame_size, or 1 for PCM)
};
