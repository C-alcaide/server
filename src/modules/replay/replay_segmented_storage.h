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
 * This module uses libvmx (https://github.com/openmediatransport/libvmx),
 * licensed under MIT, which is compatible with GPL-3.
 *
 * The .mav/.idx file format and segmented storage strategy are derived from
 * the CasparCG replay module
 * (https://github.com/krzyc/CasparCG-Server/tree/master/src/modules/replay).
 * Copyright (c) 2011 Sveriges Television AB <info@casparcg.com>
 * Copyright (c) 2013 Technical University of Lodz Multimedia Centre <office@cm.p.lodz.pl>
 * Authors: Jan Starzak <jan@ministryofgoodsteps.com>,
 *          Krzysztof Pyrkosz <pyrkosz@o2.pl>
 */

#pragma once

#include "replay_extended_index.h"
#include <string>
#include <vector>
#include <deque>
#include <mutex>
#include <cstdint>
#include <boost/filesystem.hpp>

namespace caspar { namespace replay {

class ReplaySegmentedWriter {
public:
    ReplaySegmentedWriter();
    ~ReplaySegmentedWriter();

    /**
     * Start a new recording session.
     * @param base_path The base directory and filename prefix (e.g., "C:/data/rec").
     * @param max_duration_sec Maximum duration of the buffer in seconds. Old segments are deleted.
     * @param segment_duration_sec Duration of each segment in seconds.
     */
    bool Open(const boost::filesystem::path& base_path, int max_duration_sec, int segment_duration_sec, int width, int height, double fps);

    /**
     * Writes a single frame of VMX compressed data.
     * @param data Pointer to the compressed frame data.
     * @param size Size of the data in bytes.
     * @param timestamp Timestamp of the frame (UTC microseconds).
     */
    bool WriteFrame(const void* data, size_t size, uint64_t timestamp);
    
    void Close();

private:
    void RotateSegment(uint64_t start_timestamp);
    void CleanupOldSegments();

    boost::filesystem::path base_path_;
    int max_duration_sec_;
    bool is_open_;

    int current_segment_index_;
    uint64_t current_segment_start_timestamp_;
    
    // Format
    int width_;
    int height_;
    double fps_;
    
    FILE* file_mav_;
    FILE* file_idx_;
    
    struct SegmentInfo {
        int index;
        uint64_t start_timestamp;
        uint64_t end_timestamp;
        boost::filesystem::path mav_path;
        boost::filesystem::path idx_path;
    };
    std::deque<SegmentInfo> segments_;
    
    uint64_t segment_duration_us_;
};

class ReplaySegmentedReader {
public:
    ReplaySegmentedReader();
    ~ReplaySegmentedReader();

    /**
     * Opens a segmented recording by scanning the folder for matching segments.
     * @param base_path The base path used in Writer.
     */
    bool Open(const boost::filesystem::path& base_path);
    void Close();

    /**
     * Retrieves a frame by global index (logical index across all segments).
     */
    bool GetFrame(size_t index, std::vector<uint8_t>& data, uint64_t& timestamp);

    /**
     * Retrieves only the timestamp for a global index. 
     */
    bool GetTimestamp(size_t index, uint64_t& timestamp);
    
    /**
     * Finds the global frame index closest to the given timestamp.
     */
    size_t SeekTimestamp(uint64_t timestamp);
    
    /**
     * Rescan for new segments and update indices.
     */
    void Refresh();

    size_t GetTotalFrames() const;
    uint64_t GetDuration() const;
    
    // Video Format Info
    int GetWidth() const;
    int GetHeight() const;
    double GetFps() const;
    
    boost::filesystem::path GetPath() const { return base_path_; }

private:
    struct SegmentReadInfo {
        int index;
        uint64_t start_timestamp;
        uint64_t end_timestamp;
        boost::filesystem::path mav_path;
        boost::filesystem::path idx_path;
        
        std::vector<replay_index_entry_v2> indices;
        size_t global_start_frame; // The global frame index where this segment starts
    };
    std::vector<SegmentReadInfo> segments_;
    
    size_t total_frames_;
    uint64_t total_duration_; // end - start of range
    
    // Format
    int width_;
    int height_;
    double fps_;
    
    FILE* cached_file_mav_;
    int cached_segment_index_;
    boost::filesystem::path base_path_;
};

class VmxTranscoder {
public:
    struct ExportJob {
        boost::filesystem::path input_path;
        uint64_t in_point;
        uint64_t out_point;
    };

    /**
     * Exports a series of clips (concatenated) to a single file using FFmpeg.
     */
    static bool ExportJobs(const std::vector<ExportJob>& jobs, const boost::filesystem::path& output_path);

    /**
     * Exports a clip from the reader to a file using FFmpeg.
     * note: This is a blocking operation.
     */
    static bool ExportClip(ReplaySegmentedReader& reader, uint64_t start_time, uint64_t end_time, const boost::filesystem::path& output_path);
};

}}
