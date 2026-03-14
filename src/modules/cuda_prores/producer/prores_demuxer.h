// prores_demuxer.h
// libavformat-based ProRes demuxer for CasparCG's CUDA ProRes decoder.
//
// Design:
//  - Uses libavformat ONLY for container I/O (av_read_frame).
//  - Does NOT use libavcodec for decoding; passes raw icpf packets to the
//    CUDA decode pipeline.
//  - Supports MOV, MXF, and MKV containers.
//  - Looped playback: when end-of-file is reached, seeks back to the first
//    video keyframe.
//  - Thread-safe: all demux calls happen on the producer's read thread.
// ---------------------------------------------------------------------------
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace caspar { namespace cuda_prores {

// Parsed frame metadata from the icpf picture header.
struct ProResFrameInfo
{
    int     width          = 0;
    int     height         = 0;
    int     frame_type     = 0;  // 0=progressive, 1=top-field-first, 2=bottom-field-first
    uint8_t color_primaries  = 1; // 1=Rec.709, 9=BT.2020
    uint8_t transfer_func    = 1; // 1=Rec.709, 14=HLG, 16=PQ
    uint8_t color_matrix     = 1; // 1=Rec.709, 9=BT.2020-NCL
    int     profile          = 3; // ProResProfile index (3=HQ)
    int     mbs_per_slice    = 4; // read from picture header log2_slice_mb_width
    int     slices_per_row   = 0; // derived
    int     num_slices       = 0; // total slices per picture
};

// Raw icpf packet from the container.
struct ProResPacket
{
    std::vector<uint8_t> data; // complete icpf frame bytes (includes 8-byte box header)
    int64_t              pts;  // presentation timestamp (in stream time-base)
    bool                 is_eof = false;
    std::vector<int32_t> audio_samples; // decoded interleaved int32_t; empty if no audio track
};

// ---------------------------------------------------------------------------

class ProResDemuxer
{
  public:
    // Open the file.  Throws std::runtime_error on failure.
    explicit ProResDemuxer(const std::wstring& path);
    ~ProResDemuxer();

    ProResDemuxer(const ProResDemuxer&)            = delete;
    ProResDemuxer& operator=(const ProResDemuxer&) = delete;

    // Read the next video packet.  Returns is_eof=true at end.
    // Can be called repeatedly after EOF to loop (demuxer handles seek).
    ProResPacket read_packet();

    // Parse icpf frame header from packet data.
    // Returns false if the frame is malformed.
    static bool parse_frame_info(const uint8_t* data, int size,
                                 ProResFrameInfo& out);

    // Container frame rate (numerator, denominator).
    void frame_rate(int& num, int& den) const;

    // Total frame count (-1 if unknown).
    int64_t total_frames() const;

    // Stream duration in microseconds (-1 if unknown).
    int64_t duration_us() const;

    // Seek to a specific frame position (0-based frame index).
    // Uses av_seek_frame — works mid-play or after EOF.
    // Returns true on success. Clears any buffered audio state.
    bool seek_to_frame(int64_t frame_number);

    // Audio track info (zero/false if the file has no audio).
    bool has_audio()         const;
    int  audio_sample_rate() const;
    int  audio_channels()    const;

    // True if the stream has been opened successfully.
    bool valid() const;

  private:
    struct impl;
    std::unique_ptr<impl> impl_;
};

}} // namespace caspar::cuda_prores
