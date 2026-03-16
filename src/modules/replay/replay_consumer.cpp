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
 * Derived from the CasparCG replay module
 * (https://github.com/krzyc/CasparCG-Server/tree/master/src/modules/replay).
 * Copyright (c) 2011 Sveriges Television AB <info@casparcg.com>
 * Copyright (c) 2013 Technical University of Lodz Multimedia Centre <office@cm.p.lodz.pl>
 * Authors: Robert Nagy <ronag89@gmail.com>,
 *          Jan Starzak <jan@ministryofgoodsteps.com>,
 *          Krzysztof Pyrkosz <pyrkosz@o2.pl>
 */

#include "replay_consumer.h"
#include <iostream>
#include <vector>
#include <boost/lexical_cast.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <common/env.h>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <common/log.h>

namespace caspar { namespace replay {

replay_consumer::replay_consumer(std::string path, VMX_PROFILE quality)
    : quality_(quality)
    , writer_(std::make_unique<ReplaySegmentedWriter>())
    , max_duration_sec_(86400) // Default 24 hours
    , segment_duration_sec_(60) // Default 60 seconds
{
    // Check for query params in path
    size_t q_pos = path.find('?');
    if (q_pos != std::string::npos) {
        std::string query = path.substr(q_pos + 1);
        path = path.substr(0, q_pos);
        
        // Parse "max_duration=HH:MM" (preferred) or seconds
        if (query.find("max_duration=") != std::string::npos) {
            try {
                size_t val_start = query.find("max_duration=") + 13;
                size_t val_end = query.find('&', val_start);
                std::string val = query.substr(val_start, val_end - val_start);
                
                if (val.find(':') != std::string::npos) {
                    std::vector<std::string> parts;
                    boost::split(parts, val, boost::is_any_of(":"));
                    if (parts.size() >= 2) {
                        int h = std::stoi(parts[0]);
                        int m = std::stoi(parts[1]);
                        max_duration_sec_ = (h * 60 + m) * 60;
                    }
                } else {
                    int s = std::stoi(val);
                    max_duration_sec_ = s;
                }
            } catch (...) {}
        } else if (query.find("duration=") != std::string::npos) {
            try {
                size_t val_start = query.find("duration=") + 9;
                size_t val_end = query.find('&', val_start);
                std::string val = query.substr(val_start, val_end - val_start);
                
                // Check if contains ':'
                if (val.find(':') != std::string::npos) {
                    // HH:MM format
                    std::vector<std::string> parts;
                    boost::split(parts, val, boost::is_any_of(":"));
                    if (parts.size() >= 2) {
                        int h = std::stoi(parts[0]);
                        int m = std::stoi(parts[1]);
                        max_duration_sec_ = (h * 60 + m) * 60;
                    }
                } else {
                    // Plain minutes
                    int m = std::stoi(val);
                    max_duration_sec_ = m * 60;
                }
            } catch (...) {}
        }
        
        if (query.find("segment_duration=") != std::string::npos) {
            try {
                size_t val_start = query.find("segment_duration=") + 17;
                size_t val_end = query.find('&', val_start);
                std::string val = query.substr(val_start, val_end - val_start);
                segment_duration_sec_ = std::stoi(val);
            } catch (...) {}
        } else if (query.find("segment=") != std::string::npos) {
            try {
                size_t val_start = query.find("segment=") + 8;
                size_t val_end = query.find('&', val_start);
                std::string val = query.substr(val_start, val_end - val_start);
                segment_duration_sec_ = std::stoi(val);
            } catch (...) {}
        }
    }

    // Trim trailing separators from path
    boost::trim_right_if(path, boost::is_any_of("/\\"));

    // Resolve path relative to media folder if needed
    boost::filesystem::path p(u16(path)); // Bug fix: path string used directly
    if (!p.is_absolute()) {
         p = boost::filesystem::path(env::media_folder()) / p;
    }
    
    // Issue 4: Overwrite protection
    // Check if files exist (check .mav.idx as indicator)
    boost::filesystem::path check_p = p;
    if (check_p.extension() == ".mav") check_p.replace_extension("");
    
    // We check if the "base" files exist. The writer creates [base].mav.idx
    if (boost::filesystem::exists(check_p.string() + ".mav.idx") || boost::filesystem::exists(check_p.string() + ".mav.000")) {
         // Exists, append timestamp
         // Use strict ISO 8601 basic format or similar safe chars
         auto now = std::chrono::system_clock::now();
         auto time_t_now = std::chrono::system_clock::to_time_t(now);
         std::tm tm_now;
         
         #ifdef _WIN32
             localtime_s(&tm_now, &time_t_now);
         #else
             localtime_r(&time_t_now, &tm_now);
         #endif
         
         std::stringstream ss;
         ss << "_" << std::put_time(&tm_now, "%Y%m%d_%H%M%S");
         
         std::string new_filename = check_p.stem().string() + ss.str() + check_p.extension().string();
         p = check_p.parent_path() / new_filename;
         
         CASPAR_LOG(warning) << L"VMX Consumer: Destination exists. Renaming to: " << p.wstring();
    }

    // Ensure directory exists
    if (!boost::filesystem::exists(p.parent_path())) {
         boost::filesystem::create_directories(p.parent_path());
    }

    path_ = u8(p.wstring());

    state_["file/path"] = u16(path_);
    state_["vmx/quality"] = std::to_wstring((int)quality_);
    
    // Diagnostics
    graph_ = spl::make_shared<diagnostics::graph>();
    graph_->set_color("frame-time", diagnostics::color(0.1f, 1.0f, 0.1f));
    graph_->set_value("buffered-video", 0.0);
    graph_->set_text(print());
    diagnostics::register_graph(graph_);
    
    // Open Segmented Writer
    // path_ is now clean without query string
    // Pass full path, writer will append suffixes
    
    // Remove extension .mav if present to get clean base name
    boost::filesystem::path base_p(u16(path_));
    if (base_p.extension() == ".mav") {
        base_p.replace_extension("");
    }
    // Re-add .mav to base so segments are name.mav.000
    // Actually the writer implementation appends .mav.000
    // So if we pass "name", file is "name.mav.000"
    // Wait, implementation: base_path_.string() + ".mav." + idx_str;
    // So if base is "clean", files are "clean.mav.000". Correct.
    
    // writer_->Open(base_p, max_duration_sec_);
    // However, original code used .mav extension explicitly.
    // Let's ensure base_p is without extension.
    if (base_p.has_extension() && base_p.extension() == ".mav")
        base_p = base_p.replace_extension("");
    
    // Defer Open to initialize when we know format
}

replay_consumer::~replay_consumer()
{
    if (vmx_) {
        VMX_Destroy(vmx_);
        vmx_ = nullptr;
    }
}

void replay_consumer::initialize(const core::video_format_desc& format_desc,
                    const core::channel_info&      channel_info,
                    int                            port_index)
{
    channel_index_ = channel_info.index;
    width_ = format_desc.width;
    height_ = format_desc.height;
    
    // Initialize VMX
    VMX_SIZE dim = { (int)width_, (int)height_ };
    // Assuming SQ profile ok
    vmx_ = VMX_Create(dim, quality_, VMX_COLORSPACE_BT709);
    
    // Write Headers
    // Using 2 channels as default for now if not available in channel_info easily, 
    // though channel_info.audio_channel_layout should support it.
    // Assuming stereo for VMX standard usage often.
    // int audio_channels = 2; // unused

    // Open Segmented Writer
    boost::filesystem::path base_p(u16(path_));
    if (base_p.has_extension() && base_p.extension() == ".mav")
        base_p = base_p.replace_extension("");

    fps_ = format_desc.fps > 0 ? format_desc.fps : 25.0;
    
    // Default 1 hour if not set? No, strictly use config.
    if (writer_) writer_->Open(base_p, max_duration_sec_, segment_duration_sec_, width_, height_, fps_);
}

std::future<bool> replay_consumer::send(core::video_field field, const core::const_frame frame)
{
    if (!vmx_ || !writer_) return make_ready_future(false);
    
    // Check pixel format - expect BGRA
    auto now = std::chrono::steady_clock::now();
    frames_since_update_++;
    auto duration_sec = std::chrono::duration_cast<std::chrono::duration<double>>(now - last_fps_update_).count();
    
    if (duration_sec >= 1.0) {
        current_fps_ = frames_since_update_ / duration_sec;
        frames_since_update_ = 0;
        last_fps_update_ = now;
        
        if (graph_.get()) {
             std::wstringstream stats;
             stats.precision(2);
             stats << std::fixed;
             stats << print() << L" - Fps: " << current_fps_ << L" frames:" << frames_written_ << L" time:" << (double)frames_written_ / fps_;
             graph_->set_text(stats.str());
        }
    }

    auto& data_array = frame.image_data(0);
    const uint8_t* data = data_array.data();
    int stride = (int)width_ * 4; // BGRA 32bit assumed
    // Could check data_array.size() / height_ for stride if needed
    if (data_array.size() >= (size_t)(stride * height_)) {
         stride = (int)(data_array.size() / height_);
    }
    
    // Assume progressive for now (0)
    int interlaced = 0;
    
    int res = VMX_EncodeBGRA(vmx_, (unsigned char*)data, stride, interlaced);
    if (res != VMX_ERR_OK) {
        return make_ready_future(false);
    }
    
    // Max buffer size needed as per docs is width*height*4?
    // VMX typically compresses, so 4*w*h is definitely safe upper bound.
    size_t max_size = (size_t)width_ * height_ * 4;
    std::vector<uint8_t> buffer(max_size);
    
    int size = VMX_SaveTo(vmx_, buffer.data(), (int)max_size);
    /* 
    Updated to match replay module format roughly
    Format per frame in .mav:
       uint32 audio_size
       byte[] audio_data
       ... VMX stream ...
    */
    
    if (size > 0) {
        // Audio
        // Frame audio is vector<int32_t>
        auto& audio_vec = frame.audio_data();
        uint32_t audio_bytes = (uint32_t)(audio_vec.size() * sizeof(int32_t));
        
        // Aggregate payload
        size_t total_size = sizeof(uint32_t) + audio_bytes + (size_t)size;
        std::vector<uint8_t> payload(total_size);
        
        uint8_t* ptr = payload.data();
        memcpy(ptr, &audio_bytes, sizeof(uint32_t)); 
        ptr += sizeof(uint32_t);
        
        if (audio_bytes > 0) {
            memcpy(ptr, audio_vec.data(), audio_bytes); 
            ptr += audio_bytes;
        }

        memcpy(ptr, buffer.data(), (size_t)size);
        
        // Timestamp (Microseconds since epoch)
        uint64_t timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();

        writer_->WriteFrame(payload.data(), payload.size(), timestamp);
        
        frames_written_++;
        
        graph_->set_value("buffered-video", (double)size / (double)max_size);
    }
    
    return make_ready_future(true);
}

core::monitor::state replay_consumer::state() const
{
    std::lock_guard<std::mutex> lock(state_mutex_);
    return state_;
}

}}
