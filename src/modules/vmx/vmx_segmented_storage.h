#pragma once

#include "vmx_extended_index.h"
#include <string>
#include <vector>
#include <deque>
#include <mutex>
#include <cstdint>
#include <boost/filesystem.hpp>

namespace caspar { namespace vmx {

class VmxSegmentedWriter {
public:
    VmxSegmentedWriter();
    ~VmxSegmentedWriter();

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

class VmxSegmentedReader {
public:
    VmxSegmentedReader();
    ~VmxSegmentedReader();

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
        
        std::vector<vmx_index_entry_v2> indices;
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
    static bool ExportClip(VmxSegmentedReader& reader, uint64_t start_time, uint64_t end_time, const boost::filesystem::path& output_path);
};

}}
