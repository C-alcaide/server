// mov_muxer.cpp
// QuickTime .mov muxer implementation for ProRes + PCM audio.
// File writes use FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED (unbuffered async I/O)
// with sector-aligned staging buffers for maximum NVMe throughput.
#include "mov_muxer.h"

#include <cassert>
#include <cstring>
#include <cstdio>
#include <algorithm>

// ─── Atom helpers ────────────────────────────────────────────────────────────

void MovMuxer::put_u8(std::vector<uint8_t> &b, uint8_t v)
{ b.push_back(v); }

void MovMuxer::put_u16(std::vector<uint8_t> &b, uint16_t v)
{ b.push_back(v >> 8); b.push_back(v & 0xFF); }

void MovMuxer::put_u32(std::vector<uint8_t> &b, uint32_t v) {
    b.push_back((v >> 24) & 0xFF); b.push_back((v >> 16) & 0xFF);
    b.push_back((v >>  8) & 0xFF); b.push_back( v        & 0xFF);
}

void MovMuxer::put_u64(std::vector<uint8_t> &b, uint64_t v) {
    put_u32(b, (uint32_t)(v >> 32));
    put_u32(b, (uint32_t)(v & 0xFFFFFFFF));
}

void MovMuxer::put_bytes(std::vector<uint8_t> &b, const uint8_t *data, size_t len)
{ b.insert(b.end(), data, data + len); }

// Push a placeholder atom header; returns the index for end_atom().
void MovMuxer::begin_atom(std::vector<uint8_t> &b, const char *type) {
    // Push 4-byte size (filled in by end_atom) + 4-byte type
    b.push_back(0); b.push_back(0); b.push_back(0); b.push_back(0);
    b.push_back((uint8_t)type[0]); b.push_back((uint8_t)type[1]);
    b.push_back((uint8_t)type[2]); b.push_back((uint8_t)type[3]);
}

// Patch atom size field at `start` position.
void MovMuxer::end_atom(std::vector<uint8_t> &b, size_t start) {
    uint32_t sz = (uint32_t)(b.size() - start);
    b[start + 0] = (sz >> 24) & 0xFF;
    b[start + 1] = (sz >> 16) & 0xFF;
    b[start + 2] = (sz >>  8) & 0xFF;
    b[start + 3] =  sz        & 0xFF;
}

// ─── File I/O ────────────────────────────────────────────────────────────────

// Write `size` bytes from `data` at current write_pos_, sector-aligned.
// Blocks until write completes (IOCP wait), then advances write_pos_.
bool MovMuxer::write_aligned(const uint8_t *data, size_t size) {
    if (!size) return true;

    // Round up to sector boundary
    size_t aligned = (size + sector_size_ - 1) & ~(size_t)(sector_size_ - 1);
    if (write_buf_.size() < aligned)
        write_buf_.assign(aligned, 0);
    memset(write_buf_.data(), 0, aligned);
    memcpy(write_buf_.data(), data, size);

    OVERLAPPED ov = {};
    ov.Offset     = (DWORD)( write_pos_        & 0xFFFFFFFF);
    ov.OffsetHigh = (DWORD)((write_pos_ >> 32) & 0xFFFFFFFF);

    DWORD written = 0;
    BOOL  ok = WriteFile(file_, write_buf_.data(), (DWORD)aligned, nullptr, &ov);
    if (!ok && GetLastError() != ERROR_IO_PENDING) return false;

    DWORD bytes_transferred = 0;
    ULONG_PTR key = 0;
    OVERLAPPED *pov = nullptr;
    if (!GetQueuedCompletionStatus(iocp_, &bytes_transferred, &key, &pov, INFINITE))
        return false;

    write_pos_ += aligned;
    return true;
}

// ─── open / close / write ────────────────────────────────────────────────────

bool MovMuxer::open(const std::wstring &path,
                    const MovVideoTrackInfo &video,
                    const MovAudioTrackInfo &audio,
                    uint32_t sector_size)
{
    video_       = video;
    audio_       = audio;
    sector_size_ = sector_size;

    path_ = path; // store for close()

    file_ = CreateFileW(
        path.c_str(),
        GENERIC_WRITE,
        0, nullptr,
        CREATE_ALWAYS,
        FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED | FILE_FLAG_WRITE_THROUGH,
        nullptr);
    if (file_ == INVALID_HANDLE_VALUE) return false;

    iocp_ = CreateIoCompletionPort(file_, nullptr, 0, 1);
    if (!iocp_) { CloseHandle(file_); return false; }

    // Write ftyp atom
    std::vector<uint8_t> ftyp;
    size_t s = 0;
    begin_atom(ftyp, "ftyp");
    // major brand 'qt  ', version 0, compatible brands
    ftyp.push_back('q'); ftyp.push_back('t'); ftyp.push_back(' '); ftyp.push_back(' ');
    put_u32(ftyp, 0); // minor version
    ftyp.push_back('q'); ftyp.push_back('t'); ftyp.push_back(' '); ftyp.push_back(' ');
    end_atom(ftyp, 0);
    if (!write_aligned(ftyp.data(), ftyp.size())) return false;

    // Begin mdat: write 8-byte header with size=0 (extended), back-patched at close
    // We use 64-bit "extended" mdat: size=1, 'mdat', then 8-byte actual size
    std::vector<uint8_t> mdat_hdr = {
        0,0,0,1, 'm','d','a','t',      // size=1 means 64-bit follows
        0,0,0,0,0,0,0,0                // will be back-patched at close
    };
    if (!write_aligned(mdat_hdr.data(), mdat_hdr.size())) return false;
    mdat_start_ = write_pos_; // content starts here
    return true;
}

bool MovMuxer::write_video(const uint8_t *data, size_t size, uint64_t pts) {
    uint64_t file_offset = write_pos_;
    if (!write_aligned(data, size)) return false;
    video_samples_.push_back({(uint32_t)size, file_offset, 1}); // duration=1 frame
    return true;
}

bool MovMuxer::write_audio(const int32_t *samples, int num_samples) {
    size_t byte_size = (size_t)num_samples * audio_.channels * sizeof(int32_t);
    uint64_t file_offset = write_pos_;
    if (!write_aligned(reinterpret_cast<const uint8_t*>(samples), byte_size)) return false;
    audio_samples_.push_back({(uint32_t)byte_size, file_offset, (uint32_t)num_samples});
    return true;
}

// ─── moov construction ───────────────────────────────────────────────────────

void MovMuxer::write_video_trak(std::vector<uint8_t> &buf) {
    size_t trak_start = buf.size(); begin_atom(buf, "trak");

    // tkhd
    {   size_t s = buf.size(); begin_atom(buf, "tkhd");
        put_u32(buf, 0x0000000F); // version=0, flags=track_enabled|in_movie|in_preview
        put_u32(buf, 0); put_u32(buf, 0); // creation / modification time
        put_u32(buf, 1); // track ID = 1
        put_u32(buf, 0); // reserved
        uint64_t total_duration = video_samples_.size(); // in movie timescale (= fps den)
        put_u32(buf, (uint32_t)total_duration);
        put_u32(buf, 0); put_u32(buf, 0); // reserved
        put_u16(buf, 0); put_u16(buf, 0); // layer, alt-group
        put_u16(buf, 0); put_u16(buf, 0); // volume (video=0), reserved
        // 3×3 unity matrix (fixed point)
        uint32_t mat[] = {0x00010000,0,0, 0,0x00010000,0, 0,0,0x40000000};
        for (auto v : mat) put_u32(buf, v);
        put_u32(buf, (uint32_t)video_.width  << 16); // width  (fixed 16.16)
        put_u32(buf, (uint32_t)video_.height << 16); // height
        end_atom(buf, s);
    }

    // mdia
    {   size_t mdia_s = buf.size(); begin_atom(buf, "mdia");

        // mdhd
        {   size_t s = buf.size(); begin_atom(buf, "mdhd");
            put_u32(buf, 0); // version=0
            put_u32(buf, 0); put_u32(buf, 0); // creation, modification
            put_u32(buf, video_.timebase_den); // timescale = fps numerator
            put_u32(buf, (uint32_t)video_samples_.size()); // duration = frame count
            put_u16(buf, 0x55C4); // language = 'und'
            put_u16(buf, 0);      // pre-defined
            end_atom(buf, s);
        }

        // hdlr
        {   size_t s = buf.size(); begin_atom(buf, "hdlr");
            put_u32(buf, 0); put_u32(buf, 0); // version, pre_defined
            buf.push_back('v'); buf.push_back('i'); buf.push_back('d'); buf.push_back('e');
            put_u32(buf, 0); put_u32(buf, 0); put_u32(buf, 0); // reserved
            buf.push_back('V'); buf.push_back('i'); buf.push_back('d'); buf.push_back('e');
            buf.push_back('o'); buf.push_back(0);
            end_atom(buf, s);
        }

        // minf → vmhd + dinf + stbl
        {   size_t minf_s = buf.size(); begin_atom(buf, "minf");

            // vmhd
            {   size_t s = buf.size(); begin_atom(buf, "vmhd");
                put_u32(buf, 1); // version=0, flags=1
                put_u16(buf, 0); put_u16(buf, 0); put_u16(buf, 0); put_u16(buf, 0);
                end_atom(buf, s);
            }

            // dinf → dref
            {   size_t s = buf.size(); begin_atom(buf, "dinf");
                size_t d = buf.size(); begin_atom(buf, "dref");
                put_u32(buf, 0); // version + flags
                put_u32(buf, 1); // entry count
                size_t e = buf.size(); begin_atom(buf, "url ");
                put_u32(buf, 1); // self-contained flag
                end_atom(buf, e);
                end_atom(buf, d);
                end_atom(buf, s);
            }

            // stbl
            {   size_t stbl_s = buf.size(); begin_atom(buf, "stbl");

                // stsd
                {   size_t s = buf.size(); begin_atom(buf, "stsd");
                    put_u32(buf, 0); // version + flags
                    put_u32(buf, 1); // entry count
                    // ProRes sample entry
                    size_t e = buf.size();
                    // fourcc is the ProRes codec tag
                    char cc[5] = {
                        (char)(video_.prores_fourcc >> 24),
                        (char)(video_.prores_fourcc >> 16),
                        (char)(video_.prores_fourcc >>  8),
                        (char)(video_.prores_fourcc      ), 0};
                    begin_atom(buf, cc);
                    put_bytes(buf, (const uint8_t*)"\0\0\0\0\0\0", 6); // reserved
                    put_u16(buf, 1); // data ref index
                    put_u16(buf, 0); // version
                    put_u16(buf, 0); // revision
                    put_u32(buf, 0); // vendor
                    put_u16(buf, (uint16_t)video_.width);
                    put_u16(buf, (uint16_t)video_.height);
                    put_u32(buf, 0x00480000); // h DPI 72.0 fixed
                    put_u32(buf, 0x00480000); // v DPI 72.0 fixed
                    put_u32(buf, 0); // data size
                    put_u16(buf, 1); // frame count per sample
                    // compressor name (32 bytes, Pascal string)
                    const char *enc_name = "\x0BCUDA-ProRes";
                    for (int i = 0; i < 32; i++)
                        buf.push_back(i < (int)strlen(enc_name) ? (uint8_t)enc_name[i] : 0);
                    put_u16(buf, 24); // depth
                    put_u16(buf, 0xFFFF); // color table id = -1

                    // colr atom
                    {   size_t cs = buf.size(); begin_atom(buf, "colr");
                        buf.push_back('n'); buf.push_back('c'); buf.push_back('l');buf.push_back('c');
                        put_u16(buf, video_.color.color_primaries);
                        put_u16(buf, video_.color.transfer_function);
                        put_u16(buf, video_.color.color_matrix);
                        end_atom(buf, cs);
                    }

                    // HDR metadata atoms
                    if (video_.color.has_hdr) {
                        // mdcv
                        {   size_t cs = buf.size(); begin_atom(buf, "mdcv");
                            for (int i = 0; i < 3; i++) {
                                put_u16(buf, video_.color.mdcv_primaries_x[i]);
                                put_u16(buf, video_.color.mdcv_primaries_y[i]);
                            }
                            put_u16(buf, video_.color.mdcv_white_x);
                            put_u16(buf, video_.color.mdcv_white_y);
                            put_u32(buf, video_.color.mdcv_max_lum);
                            put_u32(buf, video_.color.mdcv_min_lum);
                            end_atom(buf, cs);
                        }
                        // clli
                        {   size_t cs = buf.size(); begin_atom(buf, "clli");
                            put_u16(buf, video_.color.clli_max_cll);
                            put_u16(buf, video_.color.clli_max_fall);
                            end_atom(buf, cs);
                        }
                    }
                    end_atom(buf, e); // ProRes sample entry
                    end_atom(buf, s); // stsd
                }

                // stts: time-to-sample (all samples are 1 frame duration = constant)
                {   size_t s = buf.size(); begin_atom(buf, "stts");
                    put_u32(buf, 0); // version + flags
                    put_u32(buf, 1); // entry count
                    put_u32(buf, (uint32_t)video_samples_.size());
                    put_u32(buf, video_.timebase_num); // sample duration in timescale units
                    end_atom(buf, s);
                }

                // stsc: sample-to-chunk (1 sample per chunk)
                {   size_t s = buf.size(); begin_atom(buf, "stsc");
                    put_u32(buf, 0);
                    put_u32(buf, 1); // 1 entry
                    put_u32(buf, 1); // first chunk
                    put_u32(buf, 1); // samples per chunk
                    put_u32(buf, 1); // sample description index
                    end_atom(buf, s);
                }

                // stsz: sample sizes
                {   size_t s = buf.size(); begin_atom(buf, "stsz");
                    put_u32(buf, 0); // version + flags
                    put_u32(buf, 0); // default sample size = 0 (variable)
                    put_u32(buf, (uint32_t)video_samples_.size());
                    for (auto &e : video_samples_) put_u32(buf, e.size);
                    end_atom(buf, s);
                }

                // stco / co64: chunk offsets
                // Use co64 since NVMe files can easily exceed 4 GB
                {   size_t s = buf.size(); begin_atom(buf, "co64");
                    put_u32(buf, 0);
                    put_u32(buf, (uint32_t)video_samples_.size());
                    for (auto &e : video_samples_) put_u64(buf, e.file_offset);
                    end_atom(buf, s);
                }

                end_atom(buf, stbl_s);
            }
            end_atom(buf, minf_s);
        }
        end_atom(buf, mdia_s);
    }
    end_atom(buf, trak_start);
}

void MovMuxer::write_audio_trak(std::vector<uint8_t> &buf) {
    if (audio_samples_.empty()) return;

    size_t trak_start = buf.size(); begin_atom(buf, "trak");

    // tkhd
    {   size_t s = buf.size(); begin_atom(buf, "tkhd");
        put_u32(buf, 0x0000000F);
        put_u32(buf, 0); put_u32(buf, 0);
        put_u32(buf, 2); // track ID = 2
        put_u32(buf, 0);
        uint64_t dur = 0;
        for (auto &e : audio_samples_) dur += e.duration;
        put_u32(buf, (uint32_t)dur);
        put_u32(buf, 0); put_u32(buf, 0);
        put_u16(buf, 0); put_u16(buf, 0);
        put_u16(buf, 0x0100); put_u16(buf, 0); // volume = 1.0
        uint32_t mat[] = {0x00010000,0,0, 0,0x00010000,0, 0,0,0x40000000};
        for (auto v : mat) put_u32(buf, v);
        put_u32(buf, 0); put_u32(buf, 0); // width/height = 0 for audio
        end_atom(buf, s);
    }

    // mdia
    {   size_t mdia_s = buf.size(); begin_atom(buf, "mdia");

        {   size_t s = buf.size(); begin_atom(buf, "mdhd");
            put_u32(buf, 0);
            put_u32(buf, 0); put_u32(buf, 0);
            put_u32(buf, (uint32_t)audio_.sample_rate);
            uint64_t dur = 0;
            for (auto &e : audio_samples_) dur += e.duration;
            put_u32(buf, (uint32_t)dur);
            put_u16(buf, 0x55C4); put_u16(buf, 0);
            end_atom(buf, s);
        }

        {   size_t s = buf.size(); begin_atom(buf, "hdlr");
            put_u32(buf, 0); put_u32(buf, 0);
            buf.push_back('s'); buf.push_back('o'); buf.push_back('u'); buf.push_back('n');
            put_u32(buf, 0); put_u32(buf, 0); put_u32(buf, 0);
            buf.push_back('A'); buf.push_back('u'); buf.push_back('d'); buf.push_back('i');
            buf.push_back('o'); buf.push_back(0);
            end_atom(buf, s);
        }

        {   size_t minf_s = buf.size(); begin_atom(buf, "minf");

            {   size_t s = buf.size(); begin_atom(buf, "smhd");
                put_u32(buf, 0);
                put_u16(buf, 0); put_u16(buf, 0); // balance, reserved
                end_atom(buf, s);
            }

            {   size_t s = buf.size(); begin_atom(buf, "dinf");
                size_t d = buf.size(); begin_atom(buf, "dref");
                put_u32(buf, 0); put_u32(buf, 1);
                size_t e = buf.size(); begin_atom(buf, "url ");
                put_u32(buf, 1);
                end_atom(buf, e); end_atom(buf, d); end_atom(buf, s);
            }

            {   size_t stbl_s = buf.size(); begin_atom(buf, "stbl");

                // stsd: 'sowt' (signed 32-bit little-endian PCM)
                {   size_t s = buf.size(); begin_atom(buf, "stsd");
                    put_u32(buf, 0); put_u32(buf, 1);
                    size_t e = buf.size(); begin_atom(buf, "sowt");
                    put_bytes(buf, (const uint8_t*)"\0\0\0\0\0\0", 6);
                    put_u16(buf, 1); // data ref
                    put_u16(buf, 0); put_u16(buf, 0); // version, revision
                    put_u32(buf, 0); // vendor
                    put_u16(buf, (uint16_t)audio_.channels);
                    put_u16(buf, 32); // bits per sample
                    put_u16(buf, 0);  // compression id
                    put_u16(buf, 0);  // packet size
                    put_u32(buf, (uint32_t)audio_.sample_rate << 16); // fixed 16.16
                    end_atom(buf, e); end_atom(buf, s);
                }

                // stts
                {   size_t s = buf.size(); begin_atom(buf, "stts");
                    // Compress runs of equal durations
                    std::vector<std::pair<uint32_t,uint32_t>> runs;
                    for (auto &e : audio_samples_) {
                        if (runs.empty() || runs.back().second != e.duration)
                            runs.push_back({1, e.duration});
                        else
                            runs.back().first++;
                    }
                    put_u32(buf, 0);
                    put_u32(buf, (uint32_t)runs.size());
                    for (auto &r : runs) { put_u32(buf, r.first); put_u32(buf, r.second); }
                    end_atom(buf, s);
                }

                // stsc
                {   size_t s = buf.size(); begin_atom(buf, "stsc");
                    put_u32(buf, 0); put_u32(buf, 1);
                    put_u32(buf, 1); put_u32(buf, 1); put_u32(buf, 1);
                    end_atom(buf, s);
                }

                // stsz
                {   size_t s = buf.size(); begin_atom(buf, "stsz");
                    put_u32(buf, 0); put_u32(buf, 0);
                    put_u32(buf, (uint32_t)audio_samples_.size());
                    for (auto &e : audio_samples_) put_u32(buf, e.size);
                    end_atom(buf, s);
                }

                // co64
                {   size_t s = buf.size(); begin_atom(buf, "co64");
                    put_u32(buf, 0);
                    put_u32(buf, (uint32_t)audio_samples_.size());
                    for (auto &e : audio_samples_) put_u64(buf, e.file_offset);
                    end_atom(buf, s);
                }

                end_atom(buf, stbl_s);
            }
            end_atom(buf, minf_s);
        }
        end_atom(buf, mdia_s);
    }
    end_atom(buf, trak_start);
}

bool MovMuxer::close() {
    if (file_ == INVALID_HANDLE_VALUE) return false;

    // Back-patch mdat extended size field.
    // mdat header is at offset: 0 (ftyp, sector-aligned so ftyp occupies >= 1 sector)
    // We wrote ftyp then 16-byte mdat header, then data.
    // The 8-byte actual-size field is at file offset (ftyp_size_rounded + 8).
    // Simpler: we know mdat_start_ is after the 16-byte header, so header is at
    // (mdat_start_ - round_up(16)).  Use SetFilePointerEx + WriteFile (buffered ok here).
    //
    // For simplicity at this stage, re-open with regular I/O for the patch.
    // Production implementation should track exact ftyp end offset.
    uint64_t mdat_content_end = write_pos_;
    uint64_t mdat_hdr_offset  = mdat_start_ - sector_size_; // ftyp was exactly one sector
    uint64_t mdat_total_size  = mdat_content_end - mdat_hdr_offset;
    // extended size = mdat_total_size (includes the 16-byte header)

    // Build moov
    std::vector<uint8_t> moov;
    size_t moov_s = 0; (void)moov_s;
    begin_atom(moov, "moov");

    // mvhd
    {   size_t s = moov.size(); begin_atom(moov, "mvhd");
        put_u32(moov, 0); put_u32(moov, 0); put_u32(moov, 0);
        put_u32(moov, video_.timebase_den); // timescale = fps den * num ... use fps_num
        put_u32(moov, (uint32_t)video_samples_.size()); // duration in timescale units
        put_u32(moov, 0x00010000); // preferred rate
        put_u16(moov, 0x0100);     // preferred volume
        for (int i = 0; i < 10; i++) moov.push_back(0); // reserved
        uint32_t mat[] = {0x00010000,0,0, 0,0x00010000,0, 0,0,0x40000000};
        for (auto v : mat) put_u32(moov, v);
        put_u32(moov, 0); put_u32(moov, 0); put_u32(moov, 0); put_u32(moov, 0); // pre-defined
        put_u32(moov, 0); put_u32(moov, 0);
        put_u32(moov, (uint32_t)(audio_samples_.empty() ? 2 : 3)); // next track ID
        end_atom(moov, s);
    }

    write_video_trak(moov);
    write_audio_trak(moov);

    end_atom(moov, moov_s);

    // moov is small (few MB at most) — write with regular buffered I/O
    // by closing the unbuffered handle and re-opening with buffered.
    // Close unbuffered handle; re-open with buffered I/O for moov + back-patch
    CloseHandle(iocp_); iocp_ = nullptr;
    CloseHandle(file_); file_ = INVALID_HANDLE_VALUE;

    // Re-open with buffered GENERIC_READ|WRITE to patch mdat size field
    HANDLE bf = CreateFileW(
        path_.c_str(),
        GENERIC_WRITE | GENERIC_READ, 0, nullptr, OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL, nullptr);
    if (bf == INVALID_HANDLE_VALUE) return false;

    // Back-patch little-endian 64-bit mdat size at offset (mdat_start_ - 8).
    // We stored write_pos_ which is sector-aligned; real content end is exactly
    // write_pos_ (no partial-sector waste is visible at this layer — the last
    // sector is zero-padded, but file size reflects the actual position).
    uint64_t mdat_hdr_at = mdat_start_ - 8; // offset of the extended-size field (8 bytes into the 16-byte mdat header)
    // mdat total size = from start of mdat header (mdat_start_ - 16) to end of content
    uint64_t mdat_header_start = mdat_start_ - 16; // size(4) + 'mdat'(4) + extended_size(8)
    uint64_t mdat_total        = write_pos_ - mdat_header_start; // includes 16-byte header

    // Seek to extended-size field
    LARGE_INTEGER li;
    li.QuadPart = (LONGLONG)mdat_hdr_at;
    SetFilePointerEx(bf, li, nullptr, FILE_BEGIN);
    // Write big-endian 64-bit size (QuickTime is big-endian)
    uint8_t sz_buf[8];
    for (int i = 7; i >= 0; i--) { sz_buf[i] = mdat_total & 0xFF; mdat_total >>= 8; }
    DWORD written = 0;
    WriteFile(bf, sz_buf, 8, &written, nullptr);

    // Append moov at end of file
    li.QuadPart = 0;
    SetFilePointerEx(bf, li, nullptr, FILE_END);
    WriteFile(bf, moov.data(), (DWORD)moov.size(), &written, nullptr);

    CloseHandle(bf);
    path_.clear();
    return true;
}

MovMuxer::~MovMuxer() {
    if (iocp_) { CloseHandle(iocp_); iocp_ = nullptr; }
    if (file_ != INVALID_HANDLE_VALUE) { CloseHandle(file_); file_ = INVALID_HANDLE_VALUE; }
}
