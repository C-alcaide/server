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

#include "replay_segmented_storage.h"
#include <common/log.h>
// Include vmxcodec.h for VMX_DecodeBGRA
#include "vmxcodec.h"

#include <iostream>
#include <algorithm>
#include <cstdio>
#include <boost/lexical_cast.hpp>
#include <string>
#include <vector>

#ifdef _WIN32
#include <share.h> // For _fsopen
#endif

#ifdef _WIN32
#define POPEN _popen
#define PCLOSE _pclose
#else
#define POPEN popen
#define PCLOSE pclose
#endif

namespace caspar { namespace replay {

// ----------------------------------------------------------------------------
// ReplaySegmentedWriter
// ----------------------------------------------------------------------------

ReplaySegmentedWriter::ReplaySegmentedWriter()
    : max_duration_sec_(0)
    , is_open_(false)
    , current_segment_index_(-1)
    , current_segment_start_timestamp_(0)
    , file_mav_(nullptr)
    , file_idx_(nullptr)
    , segment_duration_us_(60000000)
{
}

ReplaySegmentedWriter::~ReplaySegmentedWriter()
{
    Close();
}

bool ReplaySegmentedWriter::Open(const boost::filesystem::path& base_path, int max_duration_sec, int segment_duration_sec, int width, int height, double fps)
{
    Close();

    // Normalize path (remove extension if any)
    boost::filesystem::path p = base_path;
    if (p.has_extension()) {
        p.replace_extension("");
    }

    // Create directory for the recording (e.g. media/rec1/)
    if (!boost::filesystem::exists(p)) {
        boost::filesystem::create_directories(p);
    }

    // Set base path for files inside (e.g. media/rec1/rec1)
    // This ensures we get media/rec1/rec1.mav.000 instead of media/rec1.mav.000
    base_path_ = p / p.filename();

    max_duration_sec_ = max_duration_sec;
    // Use member variable initialized in Open
    segment_duration_us_ = (uint64_t)segment_duration_sec * 1000000;
    if (segment_duration_us_ == 0) segment_duration_us_ = 60 * 1000000;

    width_ = width;
    height_ = height;
    fps_ = fps;

    is_open_ = true;
    current_segment_index_ = -1; 
    return true;
}

void ReplaySegmentedWriter::Close()
{
    if (file_mav_) {
        fclose(file_mav_);
        file_mav_ = nullptr;
    }
    if (file_idx_) {
        fclose(file_idx_);
        file_idx_ = nullptr;
    }
    is_open_ = false;
    segments_.clear();
}

void ReplaySegmentedWriter::RotateSegment(uint64_t start_timestamp)
{
    if (file_mav_) fclose(file_mav_);
    if (file_idx_) fclose(file_idx_);

    current_segment_index_++;
    current_segment_start_timestamp_ = start_timestamp;

    std::string idx_str = std::to_string(current_segment_index_);
    // Zero pad index to 3 digits
    while (idx_str.length() < 3) idx_str = "0" + idx_str;

    boost::filesystem::path mav_path = base_path_.string() + ".mav." + idx_str;
    boost::filesystem::path idx_path = base_path_.string() + ".idx." + idx_str;

#ifdef _WIN32
    file_mav_ = _fsopen(mav_path.string().c_str(), "wb", _SH_DENYNO);
    file_idx_ = _fsopen(idx_path.string().c_str(), "wb", _SH_DENYNO);
#else
    file_mav_ = fopen(mav_path.string().c_str(), "wb");
    file_idx_ = fopen(idx_path.string().c_str(), "wb");
#endif

    if (!file_mav_ || !file_idx_) {
        std::cerr << "[VMX] Failed to rotate segment: " << mav_path << std::endl;
        is_open_ = false;
        return;
    }

    // Write segment header
    replay_segment_header header;
    header.segment_index = current_segment_index_;
    header.start_timestamp = start_timestamp;
    header.end_timestamp = 0; // Updated later potentially, or not critical
    header.width = width_;
    header.height = height_;
    header.fps = fps_;
    header.segment_duration_sec = (uint32_t)(segment_duration_us_ / 1000000);
    fwrite(&header, sizeof(header), 1, file_idx_);

    SegmentInfo info;
    info.index = current_segment_index_;
    info.start_timestamp = start_timestamp;
    info.end_timestamp = start_timestamp; 
    info.mav_path = mav_path;
    info.idx_path = idx_path;

    segments_.push_back(info);
}

bool ReplaySegmentedWriter::WriteFrame(const void* data, size_t size, uint64_t timestamp)
{
    if (!is_open_) return false;

    // First frame or segment full?
    if (current_segment_index_ == -1 || (timestamp - current_segment_start_timestamp_) > segment_duration_us_) {
        RotateSegment(timestamp);
        if (!is_open_) return false;
    }

    // Get current offset in MAV file
    // ftell returns long, _ftelli64 is safer for large files but segments are small (<2GB)
    int64_t offset = _ftelli64(file_mav_);
    
    // Write Data
    if (fwrite(data, 1, size, file_mav_) != size) {
        return false;
    }
    fflush(file_mav_);

    // Write Index
    replay_index_entry_v2 entry;
    entry.file_offset = offset;
    entry.timestamp = timestamp;
    fwrite(&entry, sizeof(entry), 1, file_idx_);
    fflush(file_idx_);

    // Update current segment end time
    if (!segments_.empty()) {
        segments_.back().end_timestamp = timestamp;
    }

    CleanupOldSegments();
    return true;
}

void ReplaySegmentedWriter::CleanupOldSegments()
{
    if (max_duration_sec_ <= 0) return;
    
    // Safety: we should not rely solely on segments_.size() > 1
    // Instead, check if total duration truly exceeds constraints
    
    uint64_t max_us = (uint64_t)max_duration_sec_ * 1000000;
    
    // We iterate from oldest to newest (front of deque)
    while (segments_.size() > 1) {
        SegmentInfo& old = segments_.front();
        SegmentInfo& current = segments_.back();
        
        // Duration from start of oldest to start of current
        // (Use start of current because current is still growing)
        uint64_t current_span = current.start_timestamp - old.start_timestamp;
        
        if (current_span > max_us) {
            // Safe to delete 'old'
            
            // Try to delete files
            // On Windows, if a file is open with sharing permissions, unique deletion behaviors occur.
            // _fsopen in Readers uses _SH_DENYNO whcih permits read/write but NOT Delete access usually.
            // If we fail to delete, it means a reader has it open.
            // In that case, we should NOT remove it from our tracking list yet, or we lose track of it forever.
            
            boost::system::error_code ec;
            boost::filesystem::remove(old.mav_path, ec);
            
            if (ec) {
                // Failed to delete (likely locked by reader)
                // Stop cleanup for now. Wait for reader to finish.
                // Log warning optionally
                // std::cerr << "[VMX] Could not cleanup segment: " << old.mav_path << " (in use)" << std::endl;
                break; 
            }
            
            // Delete Index (usually succeeds)
            boost::filesystem::remove(old.idx_path, ec);
            
            segments_.pop_front();
        } else {
            // Not old enough
            break;
        }
    }
}

// ----------------------------------------------------------------------------
// ReplaySegmentedReader
// ----------------------------------------------------------------------------

ReplaySegmentedReader::ReplaySegmentedReader()
    : total_frames_(0)
    , total_duration_(0)
    , cached_file_mav_(nullptr)
    , cached_segment_index_(-1)
{
}

ReplaySegmentedReader::~ReplaySegmentedReader()
{
    Close();
}

bool ReplaySegmentedReader::Open(const boost::filesystem::path& base_path)
{
    Close();
    base_path_ = base_path;
    namespace fs = boost::filesystem;
    
    fs::path parent;
    std::string base_name;
    std::string file_prefix;
    
    // Check new folder structure: path is a directory (or path base without ext is directory)
    fs::path p = base_path;
    if (p.has_extension()) p.replace_extension("");
    
    if (fs::exists(p) && fs::is_directory(p)) {
        parent = p;
        base_name = p.filename().string();
        // Files inside will be named like folder
        file_prefix = (p / base_name).string();
    } else {
        // Old structure: files in parent folder
        parent = base_path.parent_path();
        base_name = base_path.filename().string();
        file_prefix = base_path.string();
        if (base_path.has_extension()) {
             // If base_path was passed with extension (e.g. file.mav), strip it for prefix
             fs::path temp = base_path;
             temp.replace_extension("");
             file_prefix = temp.string();
             base_name = temp.filename().string();
        }
    }
    
    if (!fs::exists(parent)) return false;

    std::vector<SegmentReadInfo> discovered;

    try {
        for (fs::directory_iterator it(parent); it != fs::directory_iterator(); ++it) {
            std::string filename = it->path().filename().string();
            // Match base_name.idx.00000
            if (filename.find(base_name + ".idx.") == 0) {
                // Parse index
                std::string idx_part = filename.substr((base_name + ".idx.").length());
                int index = std::stoi(idx_part);
                
                SegmentReadInfo info;
                info.index = index;
                info.idx_path = it->path();
                info.mav_path = file_prefix + ".mav." + idx_part;
                
                if (fs::exists(info.mav_path)) {
                    discovered.push_back(info);
                }
            }
        }
    } catch (...) {
        return false;
    }

    // Sort by index
    std::sort(discovered.begin(), discovered.end(), [](const SegmentReadInfo& a, const SegmentReadInfo& b) {
        return a.index < b.index;
    });

    // Load indices
    total_frames_ = 0;
    for (auto& seg : discovered) {
#ifdef _WIN32
        FILE* f = _fsopen(seg.idx_path.string().c_str(), "rb", _SH_DENYNO);
#else
        FILE* f = fopen(seg.idx_path.string().c_str(), "rb");
#endif
        if (!f) continue;
        
        // Read Header
        replay_segment_header header;
        if (fread(&header, sizeof(header), 1, f) != 1) {
            fclose(f);
            continue;
        }
    
        width_ = header.width;
        height_ = header.height;
        fps_ = header.fps;

        _fseeki64(f, 0, SEEK_END);
        long size = ftell(f);
        fseek(f, sizeof(header), SEEK_SET);
        
        long entry_count = (size - sizeof(header)) / sizeof(replay_index_entry_v2);
        if (entry_count > 0) {
            seg.indices.resize(entry_count);
            fread(seg.indices.data(), sizeof(replay_index_entry_v2), entry_count, f);

            // Correct timestamps from entries
            seg.start_timestamp = seg.indices.front().timestamp;
            seg.end_timestamp = seg.indices.back().timestamp;
            
            seg.global_start_frame = total_frames_;
            total_frames_ += entry_count;
            segments_.push_back(seg);
        }
        fclose(f);
    }
    
    if (!segments_.empty()) {
        total_duration_ = segments_.back().end_timestamp - segments_.front().start_timestamp;
    }

    return !segments_.empty();
}

void ReplaySegmentedReader::Close()
{
    if (cached_file_mav_) {
        fclose(cached_file_mav_);
        cached_file_mav_ = nullptr;
    }
    cached_segment_index_ = -1;
    segments_.clear();
}


bool ReplaySegmentedReader::GetTimestamp(size_t index, uint64_t& timestamp)
{
    if (index >= total_frames_) return false;

    // Find Segment by global index
    auto it = std::upper_bound(segments_.begin(), segments_.end(), index, 
        [](size_t idx, const SegmentReadInfo& seg) {
            return idx < seg.global_start_frame;
        });
    
    if (it == segments_.begin()) return false;
    auto& seg = *(--it);

    size_t local_index = index - seg.global_start_frame;
    if (local_index >= seg.indices.size()) return false;

    timestamp = seg.indices[local_index].timestamp;
    return true;
}

bool ReplaySegmentedReader::GetFrame(size_t index, std::vector<uint8_t>& data, uint64_t& timestamp)
{
    if (index >= total_frames_) return false;

    // Find Segment
    auto it = std::upper_bound(segments_.begin(), segments_.end(), index, 
        [](size_t idx, const SegmentReadInfo& seg) {
            return idx < seg.global_start_frame;
        });
    
    if (it == segments_.begin()) return false;
    auto& seg = *(--it);

    size_t local_index = index - seg.global_start_frame;
    if (local_index >= seg.indices.size()) return false;

    const auto& entry = seg.indices[local_index];
    timestamp = entry.timestamp;

    // Open MAV file
    if (cached_segment_index_ != seg.index) {
        if (cached_file_mav_) fclose(cached_file_mav_);
#ifdef _WIN32
        cached_file_mav_ = _fsopen(seg.mav_path.string().c_str(), "rb", _SH_DENYNO);
#else
        cached_file_mav_ = fopen(seg.mav_path.string().c_str(), "rb");
#endif
        cached_segment_index_ = seg.index;
    }
    if (!cached_file_mav_) return false;

    // Determine size (next offset - this offset)
    int64_t next_offset;
    if (local_index + 1 < seg.indices.size()) {
        next_offset = seg.indices[local_index + 1].file_offset;
    } else {
        // End of file
        _fseeki64(cached_file_mav_, 0, SEEK_END);
        next_offset = _ftelli64(cached_file_mav_);
    }
    
    if (next_offset <= entry.file_offset) return false;
    
    // Safety cap for frame size (e.g. 100MB) to prevent length_error on corruption
    int64_t diff = next_offset - entry.file_offset;
    if (diff > 100 * 1024 * 1024) return false;

    size_t size = (size_t)diff;
    // Ensure vector is exactly the size of the frame data
    if (data.size() != size) data.resize(size);
    
    _fseeki64(cached_file_mav_, entry.file_offset, SEEK_SET);
    if (fread(data.data(), 1, size, cached_file_mav_) != size) return false;

    return true;
}

size_t ReplaySegmentedReader::SeekTimestamp(uint64_t timestamp)
{
    // Find segment
    for(const auto& seg : segments_) {
        if (timestamp >= seg.start_timestamp && timestamp <= seg.end_timestamp) {
            // Binary search inside segment indices
            auto it = std::lower_bound(seg.indices.begin(), seg.indices.end(), timestamp,
                [](const replay_index_entry_v2& entry, uint64_t ts) {
                    return entry.timestamp < ts;
                });
            
            size_t local = std::distance(seg.indices.begin(), it);
            if (local >= seg.indices.size()) local = seg.indices.size() - 1;
            return seg.global_start_frame + local;
        }
    }
    // Handle out of bounds
    if (segments_.empty()) return 0;
    if (timestamp < segments_.front().start_timestamp) return 0;
    return total_frames_ - 1;
}

size_t ReplaySegmentedReader::GetTotalFrames() const { return total_frames_; }

void ReplaySegmentedReader::Refresh() 
{
    if (segments_.empty()) {
        Open(base_path_);
        return;
    }

    // Incremental refresh:
    // 1. Update last segment
    // 2. Scan for next segments
    
    // --- Update Last Segment ---
    auto& last_seg = segments_.back();
#ifdef _WIN32
    FILE* f = _fsopen(last_seg.idx_path.string().c_str(), "rb", _SH_DENYNO);
#else
    FILE* f = fopen(last_seg.idx_path.string().c_str(), "rb");
#endif
    if (f) {
        long long current_count = (long long)last_seg.indices.size();
        long long offset = sizeof(replay_segment_header) + (current_count * sizeof(replay_index_entry_v2));
        
        _fseeki64(f, 0, SEEK_END);
        long long file_size = _ftelli64(f);
       
        if (file_size > offset) {
            _fseeki64(f, offset, SEEK_SET);
            long long new_entries_bytes = file_size - offset;
            long long new_count = new_entries_bytes / sizeof(replay_index_entry_v2);
            
            if (new_count > 0) {
                std::vector<replay_index_entry_v2> buffer(new_count);
                if (fread(buffer.data(), sizeof(replay_index_entry_v2), new_count, f) == (size_t)new_count) {
                    last_seg.indices.insert(last_seg.indices.end(), buffer.begin(), buffer.end());
                    if (!last_seg.indices.empty()) {
                        last_seg.end_timestamp = last_seg.indices.back().timestamp;
                    }
                    total_frames_ += (size_t)new_count;
                }
            }
        }
        fclose(f);
    }

    // --- Check for Next Segments ---
    // Infer prefix from last segment path
    std::string path_str = last_seg.idx_path.string();
    size_t last_dot_idx = path_str.rfind(".idx.");
    if (last_dot_idx == std::string::npos) return; 
    
    std::string prefix = path_str.substr(0, last_dot_idx);
    int next_idx = last_seg.index + 1;
    
    while (true) {
        // Try likely padding formats (001, 1)
        std::vector<std::string> trials;
        std::string p3 = std::to_string(next_idx);
        while (p3.length() < 3) p3 = "0" + p3;
        trials.push_back(p3);
        
        std::string p_raw = std::to_string(next_idx);
        if (p_raw != p3) trials.push_back(p_raw);
        
        boost::filesystem::path next_idx_path;
        bool found = false;
        
        for (const auto& s : trials) {
            boost::filesystem::path p = prefix + ".idx." + s;
            if (boost::filesystem::exists(p)) {
                next_idx_path = p;
                found = true;
                break;
            }
        }
        
        if (!found) break; // No next segment found

        // Found new segment -> Verify MAV exists
        std::string mav_s = next_idx_path.string();
        size_t pos = mav_s.rfind(".idx.");
        if (pos != std::string::npos) mav_s.replace(pos, 5, ".mav.");
        
        if (!boost::filesystem::exists(mav_s)) break; // Index without Mav
        
        // Open and Read new segment
#ifdef _WIN32
        FILE* f2 = _fsopen(next_idx_path.string().c_str(), "rb", _SH_DENYNO);
#else
        FILE* f2 = fopen(next_idx_path.string().c_str(), "rb");
#endif      
        if (!f2) break;
        
        replay_segment_header h;
        if (fread(&h, sizeof(h), 1, f2) != 1) { fclose(f2); break; }
        
        _fseeki64(f2, 0, SEEK_END);
        long long sz = _ftelli64(f2);
        _fseeki64(f2, sizeof(h), SEEK_SET);
        
        long long count = (sz - sizeof(h)) / sizeof(replay_index_entry_v2);
        if (count > 0) {
            SegmentReadInfo new_seg;
            new_seg.index = next_idx;
            new_seg.idx_path = next_idx_path;
            new_seg.mav_path = mav_s;
            
            new_seg.indices.resize(count);
            fread(new_seg.indices.data(), sizeof(replay_index_entry_v2), count, f2);
            new_seg.start_timestamp = new_seg.indices.front().timestamp;
            new_seg.end_timestamp = new_seg.indices.back().timestamp;
            new_seg.global_start_frame = total_frames_;
            
            total_frames_ += (size_t)count;
            segments_.push_back(new_seg);
        }
        fclose(f2);
        
        next_idx++;
        // Continue loop to find next segment...
    }
    
    if (!segments_.empty()) {
        total_duration_ = segments_.back().end_timestamp - segments_.front().start_timestamp;
    }
}

uint64_t ReplaySegmentedReader::GetDuration() const { return total_duration_; }

int ReplaySegmentedReader::GetWidth() const { return width_; }
int ReplaySegmentedReader::GetHeight() const { return height_; }
double ReplaySegmentedReader::GetFps() const { return fps_; }

// ----------------------------------------------------------------------------
// VmxTranscoder
// ----------------------------------------------------------------------------

bool VmxTranscoder::ExportJobs(const std::vector<ExportJob>& jobs, const boost::filesystem::path& output_path)
{
    if (jobs.empty()) return false;
    
    // First job determines format
    ReplaySegmentedReader reader;
    if (!reader.Open(jobs[0].input_path)) {
         std::cerr << "[VMX] Failed to open first input: " << jobs[0].input_path << std::endl;
         return false;
    }
    
    int width = reader.GetWidth();
    int height = reader.GetHeight();
    double fps = reader.GetFps();
    
    reader.Close(); 
    
    CASPAR_LOG(info) << L"[VMX] Exporting " << jobs.size() << " clips to " << output_path.wstring();
    
    // Setup generic ffmpeg command based on first clip format
    // Redirect output to log file for debugging
    std::string log_path = output_path.string() + ".log";
    std::string cmd = "ffmpeg -y -f rawvideo -vcodec rawvideo -pix_fmt bgra -s " 
        + std::to_string(width) + "x" + std::to_string(height) 
        + " -r " + std::to_string((int)fps) 
        + " -i - -c:v libx264 -preset fast -pix_fmt yuv420p \"" + output_path.string() + "\" > \"" + log_path + "\" 2>&1";
    
    CASPAR_LOG(info) << L"[VMX] Running: " << u16(cmd);

    FILE* pipe = POPEN(cmd.c_str(), "wb");
    if (!pipe) {
        CASPAR_LOG(error) << L"[VMX] Failed to open pipe: " << u16(cmd);
        return false;
    }
    
    // Write fake header or rely on ffmpeg auto-detect rawvideo
    // For rawvideo bgra, ffmpeg just needs data.
    
    // Reuse buffers
    std::vector<uint8_t> compressed_data;
    std::vector<uint8_t> raw_bgra;
    raw_bgra.resize(width * height * 4); // BGRA

    // Create Decoder 
    // IMPORTANT: For Export, we need BEST quality, or same as recording?
    // Use DEFAULT
    VMX_SIZE dim = { width, height };
    VMX_INSTANCE* vmx = VMX_Create(dim, VMX_PROFILE_DEFAULT, VMX_COLORSPACE_BT709); 
    if (!vmx) {
         CASPAR_LOG(error) << L"[VMX] Export Failed to create VMX instance.";
         PCLOSE(pipe);
         return false;
    }

    int total_frames_exported = 0;

    for (const auto& job : jobs) {
        ReplaySegmentedReader current_reader;
        // Construct full path if needed, usually input_path is absolute or relative to cwd
        if (!current_reader.Open(job.input_path)) {
             CASPAR_LOG(error) << L"[VMX] Failed to open input: " << job.input_path.wstring();
             continue; 
        }
        
        // Validate format
        if (current_reader.GetWidth() != width || current_reader.GetHeight() != height) {
             CASPAR_LOG(error) << L"[VMX] Format mismatch in: " << job.input_path.wstring();
             CASPAR_LOG(error) << L"Expected: " << width << L"x" << height << L" Got: " << current_reader.GetWidth() << L"x" << current_reader.GetHeight();
             continue;
        }

        // Seek
        size_t start_index = current_reader.SeekTimestamp(job.in_point);
        size_t end_index = current_reader.GetTotalFrames(); 
        
        CASPAR_LOG(info) << L"[VMX] Processing Job: " << job.input_path.wstring() << L" StartIndex: " << start_index << L" EndIndex: " << end_index 
                         << L" StartTS: " << job.in_point << L" EndTS: " << job.out_point;

        // Loop
        size_t current_index = start_index;
        uint64_t current_ts = 0;
        
        while (current_index < end_index) {
            if (!current_reader.GetFrame(current_index, compressed_data, current_ts)) {
                CASPAR_LOG(warning) << L"[VMX] Export GetFrame failed at index: " << current_index;
                break;
            }
            
            if (job.out_point > 0 && current_ts > job.out_point) {
                // CASPAR_LOG(info) << L"[VMX] Export reached out point at index: " << current_index << L" TS: " << current_ts;
                break;
            }

            // Parse Container Format: [Audio Size (4)] [Audio Data] [Video Data]
            if (compressed_data.size() < 4) {
                 CASPAR_LOG(error) << L"[VMX] Frame data too small at index: " << current_index;
                 current_index++; continue;
            }
            
            uint32_t audio_size = 0;
            memcpy(&audio_size, compressed_data.data(), 4);
            
            if (compressed_data.size() < (4 + audio_size)) {
                 CASPAR_LOG(error) << L"[VMX] Frame data incomplete audio at index: " << current_index;
                 current_index++; continue;
            }
            
            uint8_t* video_ptr = compressed_data.data() + 4 + audio_size;
            int video_len = (int)(compressed_data.size() - 4 - audio_size);

            if (VMX_LoadFrom(vmx, video_ptr, video_len) != VMX_ERR_OK) {
                 CASPAR_LOG(error) << L"[VMX] Export VMX_LoadFrom failed at index: " << current_index;
                 current_index++;
                 continue; 
            }

            // Stride
            int stride = width * 4;
            if (VMX_DecodeBGRA(vmx, raw_bgra.data(), stride) != VMX_ERR_OK) {
                  CASPAR_LOG(error) << L"[VMX] Export VMX_DecodeBGRA failed at index: " << current_index;
                  current_index++;
                  continue;
            }

            if (fwrite(raw_bgra.data(), 1, raw_bgra.size(), pipe) != raw_bgra.size()) {
                CASPAR_LOG(error) << L"[VMX] Pipe write error at index: " << current_index;
                break; // Pipe broken
            }
            // Flush periodically to ensure data makes it to pipe
            if (total_frames_exported % 25 == 0) fflush(pipe);
            
            total_frames_exported++;
            current_index++;
        }
    }
    
    fflush(pipe); 
    PCLOSE(pipe);
    VMX_Destroy(vmx);
    
    CASPAR_LOG(info) << L"[VMX] Export complete. Total Frames: " << total_frames_exported;
    return true;

}

bool VmxTranscoder::ExportClip(ReplaySegmentedReader& reader, uint64_t start_time, uint64_t end_time, const boost::filesystem::path& output_path)
{
    ExportJob job;
    job.input_path = reader.GetPath();
    job.in_point = start_time;
    job.out_point = end_time;
    std::vector<ExportJob> jobs = { job };
    return ExportJobs(jobs, output_path);
}

}}
