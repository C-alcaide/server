// mov_muxer.h
// Minimal QuickTime .mov muxer for ProRes + embedded PCM audio + SMPTE timecode.
//
// Write sequence:
//   1. open()         — create file, write ftyp atom, open mdat
//   2. write_video()  — write one encoded ProRes frame, accumulate stts/stsz/stco
//   3. write_audio()  — write one PCM audio chunk, accumulate audio stts/stsz/stco
//   4. write_timecode() — record SMPTE TC for this frame (called once per frame,
//                         in sync with write_video); omit to disable tmcd track
//   5. close()        — back-patch mdat size, write moov tree (incl. tmcd if set)
//
// Atom writer is self-contained with no external dependencies beyond Win32 and CRT.
#pragma once

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <stdint.h>
#include <string>
#include <vector>
#include <memory>

#include "../timecode.h"

// ColorDesc mirrors ProResColorDesc for muxer-level metadata.
struct MovColorInfo {
    uint8_t  color_primaries;     // 1=Rec.709, 9=BT.2020
    uint8_t  transfer_function;   // 1=Rec.709, 14=HLG, 16=PQ
    uint8_t  color_matrix;        // 1=Rec.709, 9=BT.2020-NCL
    bool     has_hdr;             // true → write mdcv + clli atoms
    uint16_t mdcv_primaries_x[3];
    uint16_t mdcv_primaries_y[3];
    uint16_t mdcv_white_x, mdcv_white_y;
    uint32_t mdcv_max_lum, mdcv_min_lum;
    uint16_t clli_max_cll, clli_max_fall;
};

struct MovVideoTrackInfo {
    int      width, height;
    uint32_t timebase_num, timebase_den; // frame rate as fraction (e.g. 1/25)
    uint32_t prores_fourcc;              // 'apco' / 'apcl' / 'apcn' / 'apch' / 'ap4h'
    MovColorInfo color;
};

struct MovAudioTrackInfo {
    int      channels;
    int      sample_rate;
    // Format: signed 32-bit little-endian PCM ('sowt')
    // samples_per_chunk: how many PCM samples per write_audio call
};

class MovMuxer {
public:
    // Open output file. Uses FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED.
    // sector_size must match the physical sector size of the target drive (typ. 4096).
    bool open(const std::wstring &path,
              const MovVideoTrackInfo &video,
              const MovAudioTrackInfo &audio,
              uint32_t sector_size = 4096);

    // Append one ProRes frame (raw encoded bytes, any size).
    // pts is in video track timescale units (= frame number for constant fps).
    bool write_video(const uint8_t *data, size_t size, uint64_t pts);

    // Append one PCM audio chunk (interleaved PCM32 samples).
    bool write_audio(const int32_t *samples, int num_samples);

    // Record the SMPTE timecode for the current frame.  Must be called once per
    // frame, after write_video().  If never called, no 'tmcd' track is written.
    // The first call establishes the start timecode; subsequent calls append per-
    // frame values stored as big-endian uint32_t frame-count samples in mdat.
    bool write_timecode(const SmpteTimecode &tc);

    // Flush, write moov, close file. Must be called exactly once on success.
    bool close();

    ~MovMuxer();

private:
    HANDLE       file_       = INVALID_HANDLE_VALUE;
    HANDLE       iocp_       = nullptr;
    uint32_t     sector_size_ = 4096;
    std::wstring path_;         // stored in open() for use in close()

    MovVideoTrackInfo video_;
    MovAudioTrackInfo audio_;

    // Running file position (in bytes, aligned to sector_size_)
    uint64_t write_pos_    = 0;
    uint64_t mdat_start_   = 0; // file offset of mdat content start (after 8-byte header)

    // Accumulated index tables
    struct SampleEntry { uint32_t size; uint64_t file_offset; uint32_t duration; };
    std::vector<SampleEntry> video_samples_;
    std::vector<SampleEntry> audio_samples_;

    // OVERLAPPED write tracking (single outstanding write per call)
    OVERLAPPED ov_ = {};
    std::vector<uint8_t> write_buf_; // sector-aligned staging buffer for write

    // Timecode track (optional QuickTime 'tmcd' track).
    // Populated by write_timecode(); absent if never called.
    struct TcSample { uint32_t frame_count; uint64_t file_offset; };
    std::vector<TcSample> tc_samples_;
    bool tc_drop_frame_ = false; // matches the first recorded SmpteTimecode

    bool write_aligned(const uint8_t *data, size_t size);
    void build_moov(std::vector<uint8_t> &moov_buf);
    void write_video_trak(std::vector<uint8_t> &buf);
    void write_audio_trak(std::vector<uint8_t> &buf);
    void write_tmcd_trak (std::vector<uint8_t> &buf);

    // Atom builder helpers
    static void begin_atom(std::vector<uint8_t> &buf, const char *type);
    static void end_atom  (std::vector<uint8_t> &buf, size_t start);
    static void put_u8    (std::vector<uint8_t> &buf, uint8_t v);
    static void put_u16   (std::vector<uint8_t> &buf, uint16_t v);
    static void put_u32   (std::vector<uint8_t> &buf, uint32_t v);
    static void put_u64   (std::vector<uint8_t> &buf, uint64_t v);
    static void put_bytes (std::vector<uint8_t> &buf, const uint8_t *data, size_t len);
};
