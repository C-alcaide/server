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

// hap_demuxer.h
// libavformat-based HAP demuxer for the CasparCG HAP decoder.
//
// Uses avformat for container I/O only — the raw compressed HAP packets
// are passed directly to the HAP frame parser + Snappy decompression pipeline.
// Supports MOV, AVI, MP4.
// ---------------------------------------------------------------------------
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace caspar { namespace hap {

struct HapFrameInfo
{
    int width  = 0;
    int height = 0;
};

// One compressed video packet from the container.
struct HapPacket
{
    std::vector<uint8_t> data;          // compressed payload (used only if pkt_handle is null)
    std::shared_ptr<void> pkt_handle;   // owns the AVPacket* (zero-copy path)
    const uint8_t*        raw_payload      = nullptr;
    size_t                raw_payload_size = 0;
    int64_t              pts     = 0;
    bool                 is_eof  = false;
    std::vector<int32_t> audio_samples;

    const uint8_t* payload_data() const { return raw_payload ? raw_payload : data.data(); }
    size_t         payload_size() const { return raw_payload ? raw_payload_size : data.size(); }
};

class HapDemuxer
{
  public:
    explicit HapDemuxer(const std::wstring& path);
    ~HapDemuxer();

    HapDemuxer(const HapDemuxer&)            = delete;
    HapDemuxer& operator=(const HapDemuxer&) = delete;

    bool valid() const;

    HapFrameInfo frame_info() const;

    HapPacket read_packet();

    void    frame_rate(int& num, int& den) const;
    int64_t total_frames() const;
    int64_t duration_us() const;

    bool seek_to_frame(int64_t frame_number);

    bool has_audio()         const;
    int  audio_sample_rate() const;
    int  audio_channels()    const;

  private:
    struct impl;
    std::unique_ptr<impl> impl_;
};

}} // namespace caspar::hap
