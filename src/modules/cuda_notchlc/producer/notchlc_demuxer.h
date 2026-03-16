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
 * NotchLC is a codec specification by Derivative Inc., available under the
 * Creative Commons Attribution 4.0 International License.
 */

// notchlc_demuxer.h
// libavformat-based NotchLC demuxer for the CasparCG CUDA NotchLC decoder.
//
// Uses avformat for container I/O only — the raw compressed NotchLC packets
// are passed directly to the CUDA decode pipeline (nvcomp LZ4 + GPU kernels).
// Supports MOV, MKV, AVI, MP4.
// ---------------------------------------------------------------------------
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace caspar { namespace cuda_notchlc {

// NotchLC compression format byte (first 12 bytes of every container packet).
enum class NotchLCFormat : uint32_t {
    LZF          = 0,   // LZF — not supported by GPU path; falls back to FFmpeg
    LZ4          = 1,   // LZ4 — handled by nvcomp
    Uncompressed = 2,   // Raw — DMA directly to device
};

// Metadata parsed from the first decompressed NotchLC frame header.
struct NotchLCFrameInfo
{
    int width  = 0;
    int height = 0;
};

// One compressed video packet from the container.
struct NotchLCPacket
{
    std::vector<uint8_t> data;          // compressed payload (used only if pkt_handle is null)
    std::shared_ptr<void> pkt_handle;   // owns the AVPacket* (zero-copy path; frees via av_packet_free)
    const uint8_t*        raw_payload      = nullptr;  // points into pkt_handle's buffer
    size_t                raw_payload_size = 0;
    int64_t              pts  = 0;
    bool                 is_eof = false;
    NotchLCFormat        format = NotchLCFormat::LZ4;
    uint32_t             uncompressed_size = 0; // from 12-byte NotchLC packet header
    std::vector<int32_t> audio_samples;         // decoded interleaved int32_t

    // Convenience: returns the payload pointer regardless of which path is active.
    const uint8_t* payload_data() const { return raw_payload ? raw_payload : data.data(); }
    size_t         payload_size() const { return raw_payload ? raw_payload_size : data.size(); }
};

// ---------------------------------------------------------------------------

class NotchLCDemuxer
{
  public:
    explicit NotchLCDemuxer(const std::wstring& path);
    ~NotchLCDemuxer();

    NotchLCDemuxer(const NotchLCDemuxer&)            = delete;
    NotchLCDemuxer& operator=(const NotchLCDemuxer&) = delete;

    bool valid() const;

    NotchLCFrameInfo frame_info() const;  // width/height from container stream parameters

    NotchLCPacket read_packet();

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

}} // namespace caspar::cuda_notchlc
