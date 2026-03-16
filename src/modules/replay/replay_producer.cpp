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

#include "replay_producer.h"
#include <iostream>
#include <vector>
#include <cstring>
#include <boost/lexical_cast.hpp>
#include <common/array.h>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <boost/filesystem.hpp>
#include <common/executor.h>
#include <common/param.h>
#include <common/utf.h>
#include <common/env.h>
#include <common/future.h> 
#include <core/frame/frame.h>
#include <core/frame/pixel_format.h>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <ctime>
#include <cstdint>
#include <algorithm>

namespace caspar { namespace replay {

// Helper for new format parsing
static uint64_t parse_replay_timestamp_helper_v2(const std::wstring& s, double fps) {
    std::string s_utf8 = u8(s);
    using namespace boost::posix_time;
    // Format: yyyy-mm-dd-hh-mm-ss-ff
    // Check if we have enough '-'
    int dashes = 0;
    for (char c : s_utf8) if (c == '-') dashes++;
    
    // Attempt parse yyyy-mm-dd-hh-mm-ss-ff 
    if (dashes >= 6) {
        try {
            std::vector<int> parts;
            std::string temp;
            std::stringstream ss(s_utf8);
            while(std::getline(ss, temp, '-')) {
                parts.push_back(std::stoi(temp));
            }
            if (parts.size() >= 6) {
                int y = parts[0];
                int m = parts[1];
                int d = parts[2];
                int h = parts[3];
                int M = parts[4];
                int S = parts[5];
                int f = (parts.size() > 6) ? parts[6] : 0;
                
                ptime pt(boost::gregorian::date((unsigned short)y, (unsigned short)m, (unsigned short)d), time_duration(h, M, S));
                ptime epoch(boost::gregorian::date(1970,1,1));
                time_duration diff = pt - epoch;
                uint64_t us = diff.total_microseconds();
                if (fps > 0) {
                     us += (uint64_t)((double)f / fps * 1000000.0);
                }
                return us;
            }
        } catch(...) {}
    }
    
    // Fallback to ISO / existing
    try {
        // handle 'T' separator
        size_t t_pos = s_utf8.find('T');
        if (t_pos != std::string::npos) s_utf8[t_pos] = ' ';
        
        ptime pt = time_from_string(s_utf8);
        ptime epoch(boost::gregorian::date(1970,1,1));
        time_duration diff = pt - epoch;
        return diff.total_microseconds();
    } catch(...) {
        return 0;
    }
}

// Updated helper
static std::pair<uint64_t, bool> parse_replay_arg_extended(const std::wstring& s, double fps) {
    if (s.empty() || s == L"0") return {0, false}; 
    
    // Check for separators indicating Time
    if (s.find(L':') != std::wstring::npos || s.find(L'-') != std::wstring::npos) {
        uint64_t ts = parse_replay_timestamp_helper_v2(s, fps);
        return {ts, true}; // true = is_timestamp
    }
    
    // Parse as number
    try {
        uint64_t val = boost::lexical_cast<uint64_t>(s);
        return {val, false}; // false = is_frame_index
    } catch (...) {}
    
    return {0, false};
}

// Helper to get formatted string
static std::wstring format_replay_timestamp(uint64_t timestamp_us, double fps) {
    if (timestamp_us == 0) return L"0000-00-00-00-00-00-00";
    
    using namespace boost::posix_time;
    ptime epoch(boost::gregorian::date(1970,1,1));
    ptime pt = epoch + microseconds(timestamp_us);
    
    // Extract base components
    tm t = to_tm(pt);
    
    // Calculate frame part
    // Get fractional seconds from timestamp_us
    uint64_t seconds_us = (timestamp_us / 1000000) * 1000000;
    uint64_t frac_us = timestamp_us - seconds_us;
    
    int frame = 0;
    if (fps > 0) {
        frame = (int)((double)frac_us / 1000000.0 * fps + 0.5);
    }
    
    std::wstringstream wss;
    wss << std::setfill(L'0') << std::setw(4) << (t.tm_year + 1900) << L"-"
        << std::setw(2) << (t.tm_mon + 1) << L"-"
        << std::setw(2) << t.tm_mday << L"-"
        << std::setw(2) << t.tm_hour << L"-"
        << std::setw(2) << t.tm_min << L"-"
        << std::setw(2) << t.tm_sec << L"-"
        << std::setw(2) << frame;
            
    return wss.str();
}

replay_producer::replay_producer(std::string path, spl::shared_ptr<core::frame_factory> frame_factory)
    : path_(std::move(path))
    , frame_factory_(frame_factory)
    , reader_(std::make_unique<ReplaySegmentedReader>())
    , vmx_(nullptr)
{
    // Resolve path relative to media folder if needed
    boost::filesystem::path p(u16(path_));
    if (!p.is_absolute())
    {
        p = boost::filesystem::path(env::media_folder()) / p;
    }
    
    // Check if path is a file or directory
    boost::filesystem::path base_p(p);
    if (base_p.extension() == ".mav" || base_p.extension() == ".idx") {
        base_p.replace_extension("");
    }
    
    // Normalize path string
    path_ = u8(p.wstring());

    state_["file/path"] = u16(path_);
    
    // Diagnostics
    graph_ = spl::make_shared<diagnostics::graph>();
    graph_->set_text(print());
    diagnostics::register_graph(graph_);

    graph_->set_color("read-time", diagnostics::color(1.0f, 0.5f, 0.0f));   // Orange
    graph_->set_color("decode-time", diagnostics::color(0.0f, 1.0f, 1.0f)); // Cyan
    graph_->set_color("buffer", diagnostics::color(0.0f, 1.0f, 0.0f));      // Green
    
    CASPAR_LOG(info) << L"VMX Producer opening: " << base_p.wstring();

    if (reader_->Open(base_p)) {
        width_ = reader_->GetWidth();
        height_ = reader_->GetHeight();
        fps_ = reader_->GetFps();
        duration_ = (int64_t)reader_->GetTotalFrames();
        
        CASPAR_LOG(info) << L"VMX Producer opened. Width: " << width_ << L" Height: " << height_ << L" FPS: " << fps_ << L" Frames: " << duration_;

        // Initialize Points
        in_point_ = 0;
        out_point_ = 0; // 0 means end of file (dynamic)
        
        if (width_ > 0 && height_ > 0) {
            VMX_SIZE dim = { width_, height_ };
            // Use DEFAULT profile for decoding
            vmx_ = VMX_Create(dim, VMX_PROFILE_DEFAULT, VMX_COLORSPACE_BT709);
        } else {
            CASPAR_LOG(error) << L"VMX Producer: Invalid dimensions " << width_ << L"x" << height_;
        }
    } else {
        CASPAR_LOG(error) << L"VMX Producer failed to open: " << base_p.wstring();
    }
}

replay_producer::~replay_producer()
{
    if (vmx_) VMX_Destroy(vmx_);
    // reader_ cleans up automatically
}


core::draw_frame replay_producer::receive_impl(core::video_field field, int nb_samples)
{
    if (!vmx_) return core::draw_frame();

    // Periodically refresh reader to find new segments
    // Check distance to end
    int64_t dist = duration_ - frame_num_;
    bool close_to_live = (speed_ > 0 && dist < (fps_ * 2)); // Within 2 seconds
    
    // Always refresh if duration is 0 (first load) or close to live, but throttle when playing live
    if (frames_since_update_ == 0 || duration_ == 0 || (close_to_live && ++refresh_skip_counter_ > 25)) {
        reader_->Refresh();
        refresh_skip_counter_ = 0;
        
        int64_t current_len = (int64_t)reader_->GetTotalFrames();
        // If length changed, update duration
        if (current_len > duration_) {
            duration_ = current_len;
        }
    }
    
    int64_t next_f = frame_num_;
    
    // FPS Calc
    auto now = std::chrono::steady_clock::now();
    frames_since_update_++;
    auto duration_sec = std::chrono::duration_cast<std::chrono::duration<double>>(now - last_fps_update_).count();
    
    if (duration_sec >= 1.0) {
        current_fps_ = frames_since_update_ / duration_sec;
        frames_since_update_ = 0;
        last_fps_update_ = now;
    }
    
    // Playback Speed Logic
    double spd = speed_;

    // Issue #4: Mute audio if speed is 0
    bool is_paused = (std::abs(spd) < 0.001);
    // Issue #5: Mute audio if at end of file
    bool is_eof = false;

    if (std::abs(spd - 1.0) < 0.001) {
        next_f++;
        fractional_frame_ = 0.0;
    } else {
        fractional_frame_ += spd;
        int64_t frame_delta = (int64_t)fractional_frame_;
        fractional_frame_ -= (double)frame_delta;
        next_f += frame_delta;
    }

    if (next_f < 0) {
        if (loop_) next_f = out_point_ > 0 ? out_point_ - 1 : (duration_ > 0 ? duration_ - 1 : 0);
        else next_f = 0;
    }
    
    // Check if we reached the end
    int64_t end_check = (out_point_ > 0 && out_point_ < duration_) ? out_point_ : duration_;

    // If out_point_ is 0 (dynamic end), use current duration_
    if (out_point_ == 0 && end_check == 0 && duration_ > 0) end_check = duration_;
    // Ensure end_check cannot be 0 if duration is available (fixes playback stop on load)
    if (end_check == 0 && duration_ > 0) end_check = duration_;

    // Check for looping
    if (end_check > 0 && next_f >= end_check) {
        if (loop_) {
             next_f = in_point_;
        } else {
             // Clamping to last frame
             next_f = end_check > 0 ? end_check - 1 : 0;
             is_eof = true;
             
             // If we were fast forwarding and hit the end, reset speed to normal (1.0)
             // This mimics behavior of catching up to live
             if (speed_ > 1.0) {
                 speed_ = 1.0;
                 fractional_frame_ = 0.0;
             }
        }
    }
    
    // Update frame number
    frame_num_ = next_f;
    
    // Update state for OSC
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        state_["file/frame"] = std::to_wstring(frame_num_);
        state_["file/fps"] = std::to_wstring(fps_);
        if (fps_ > 0.0)
            state_["file/time"] = std::to_wstring((double)frame_num_ / fps_);
        state_["file/length"] = std::to_wstring(duration_);
    }
    
    uint64_t timestamp = 0;
    bool has_frame = false;

    if (reader_) {
        auto t1 = std::chrono::high_resolution_clock::now();
        has_frame = reader_->GetFrame((size_t)frame_num_, read_buffer_, timestamp);
        auto t2 = std::chrono::high_resolution_clock::now();
        
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            state_["vmx/read_ms"] = std::to_wstring(std::chrono::duration<double, std::milli>(t2 - t1).count());
            state_["buffer/size"] = std::to_wstring(read_buffer_.size());
        }

        // Update graph
        {
             double d_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
             graph_->set_value("read-time", d_ms);
             // Size in KB
             graph_->set_value("buffer", (double)read_buffer_.size() / 1024.0);
             
             // Update text
             std::wstringstream stats;
             stats << u16(path_) << L" " << frame_num_ << L"/" << duration_ << L" (" << timestamp << L")";
             graph_->set_text(stats.str());
        }
    }
    
    if (has_frame) {
         state_["vmx/timestamp"] = std::to_wstring(timestamp);
         // Formatted timestamp
         if (timestamp > 0 && fps_ > 0) {
             state_["vmx/timestamp_formatted"] = format_replay_timestamp(timestamp, fps_);
         }
         
         if (read_buffer_.empty()) return current_frame_;

         // Parse Audio Header (uint32_t size + data)
         uint8_t* ptr = read_buffer_.data();
         size_t remaining = read_buffer_.size();
         
         if (remaining < 4) return current_frame_;
         
         uint32_t audio_size = *(uint32_t*)ptr;
         // Handle potential endianness if needed, but consumer writes raw uint32
         
         ptr += 4; remaining -= 4;
         
         const uint8_t* audio_ptr = nullptr;
         uint32_t audio_bytes = 0;
         
         // Extract Audio Data Pointer
         if (remaining >= audio_size) {
             if (audio_size > 0) {
                 audio_ptr = ptr;
                 audio_bytes = audio_size;
             }
             
             // Skip audio
             ptr += audio_size; remaining -= audio_size;
         } else {
              return current_frame_;
         }

         if (width_ <= 0 || height_ <= 0 || width_ > 16384 || height_ > 16384) {
             CASPAR_LOG(error) << L"VMX Producer: Invalid dimensions " << width_ << L"x" << height_ << L". Skipping frame.";
             return current_frame_;
         }

         if (VMX_LoadFrom(vmx_, ptr, (int)remaining) == VMX_ERR_OK) {
             // Create Frame DESC
             core::pixel_format_desc desc(core::pixel_format::bgra);

             desc.planes.emplace_back(width_, height_, 4);


             // Create Frame
             auto mutable_frame = frame_factory_->create_frame(this, desc);
             
             // Copy Audio
             if (audio_bytes > 0 && audio_ptr) {
                 size_t num_samples = audio_bytes / sizeof(int32_t);
                 if (num_samples > 0) {
                      std::vector<int32_t> temp_audio(num_samples);
                      if (is_paused || is_eof) {
                           // Issue #4: Play silence when paused to avoid sample loop/glitch
                           std::fill(temp_audio.begin(), temp_audio.end(), 0);
                      } else {
                           memcpy(temp_audio.data(), audio_ptr, audio_bytes);
                      }
                      mutable_frame.audio_data() = caspar::array<int32_t>(std::move(temp_audio));
                 }
             }
             
             // Decode directly into frame buffer
             // Accessing image_data(0)
             auto& data_array = mutable_frame.image_data(0);
             int stride = width_ * 4; 
             
             auto t_d1 = std::chrono::high_resolution_clock::now();
             int res = VMX_DecodeBGRA(vmx_, (unsigned char*)data_array.data(), stride);
             auto t_d2 = std::chrono::high_resolution_clock::now();

             double d_ms = std::chrono::duration<double, std::milli>(t_d2 - t_d1).count();
             {
                 std::lock_guard<std::mutex> lock(state_mutex_);
                 state_["vmx/decode_ms"] = std::to_wstring(d_ms);
                 graph_->set_value("decode-time", d_ms);
             }

             if (res == VMX_ERR_OK) {
                 // Success - move mutable_frame into draw_frame
                 current_frame_ = core::draw_frame(std::move(mutable_frame));
             }
         }
    }
    
    return current_frame_;
}

std::future<std::wstring> replay_producer::call(const std::vector<std::wstring>& params)
{
    using namespace caspar::common;

    if (params.empty())
        CASPAR_THROW_EXCEPTION(caspar::invalid_argument() << caspar::msg_info(L"No command specified"));

    const std::wstring& command = params.front();

    if (boost::iequals(command, L"SEEK")) {
        // Stop verbose logging for seek which can be spammed
        // CASPAR_LOG(info) << L"VMX Producer SEEK";
        if (params.size() < 2)
            CASPAR_THROW_EXCEPTION(caspar::invalid_argument() << caspar::msg_info(L"Missing argument"));
        
        try {
            if (boost::iequals(params[1], L"LIVE")) {
                // Force update duration from file
                if (reader_) {
                    reader_->Refresh();
                    int64_t current_len = (int64_t)reader_->GetTotalFrames();
                    if (current_len > 0) duration_ = current_len;
                }

                        int64_t target = duration_.load() > 0 ? duration_.load() - 1 : 0;
                // Back off 25 frames (increased from 10) to avoid stutter
                if (target > 25) target -= 25;
                else target = 0;
                frame_num_ = target;
                
            } else {
                std::pair<uint64_t, bool> arg = parse_replay_arg_extended(params[1], fps_);
                if (arg.second) {
                     // Arg is timestamp -> find frame
                     if (reader_) {
                         size_t f = reader_->SeekTimestamp(arg.first);
                         frame_num_ = f;
                     }
                } else {
                    int64_t seek_frame = (int64_t)arg.first;
                    frame_num_ = seek_frame;
                }
            }
            fractional_frame_ = 0.0;
        } catch(...) {}
        return caspar::make_ready_future(std::wstring(L"OK"));
    }
    else if (boost::iequals(command, L"IN")) {
         if (params.size() < 2)
            CASPAR_THROW_EXCEPTION(caspar::invalid_argument() << caspar::msg_info(L"Missing argument"));
         
         auto arg = parse_replay_arg_extended(params[1], fps_);
         if (arg.second) {
             if (reader_) in_point_ = (int64_t)reader_->SeekTimestamp(arg.first);
         } else {
             in_point_ = (int64_t)arg.first;
         }
         return caspar::make_ready_future(std::wstring(L"OK"));
    }
    else if (boost::iequals(command, L"OUT")) {
         if (params.size() < 2)
            CASPAR_THROW_EXCEPTION(caspar::invalid_argument() << caspar::msg_info(L"Missing argument"));
         
         auto arg = parse_replay_arg_extended(params[1], fps_);
         if (arg.second) {
             if (reader_) out_point_ = (int64_t)reader_->SeekTimestamp(arg.first);
         } else {
             out_point_ = (int64_t)arg.first;
         }
         return caspar::make_ready_future(std::wstring(L"OK"));
    }
    else if (boost::iequals(command, L"LOOP")) {
         if (params.size() > 1) {
            try {
                loop_ = boost::lexical_cast<bool>(params[1]);
            } catch(...) { loop_ = !loop_; }
         } else {
            loop_ = !loop_;
         }
         return caspar::make_ready_future(std::wstring(L"OK"));
    }
     else if (boost::iequals(command, L"SPEED")) {
         if (params.size() > 1) {
            try {
                speed_ = boost::lexical_cast<double>(params[1]);
            } catch(...) {}
         }
         return caspar::make_ready_future(std::wstring(L"OK"));
    }
    else if (boost::iequals(command, L"EXPORT")) {
        // EXPORT [output_filename] ([input_filename] [start] [end])...
        if (params.size() < 2)
            CASPAR_THROW_EXCEPTION(caspar::invalid_argument() << caspar::msg_info(L"Missing filename"));
        
        std::wstring output_filename_w = params[1];
        std::string current_path_str = u8(path_);
        double fps = fps_;

        // Capture params by value (copy)
        std::vector<std::wstring> args(params.begin() + 2, params.end());
        
        std::thread([current_path_str, output_filename_w, args, fps]() {
            try {
                CASPAR_LOG(info) << L"Export Thread Started for: " << output_filename_w;
                // Parse args into jobs
                std::vector<VmxTranscoder::ExportJob> jobs;
                
                size_t i = 0;
                // Default input is current file
                boost::filesystem::path default_input_p(u16(current_path_str));
                if (default_input_p.extension() == ".mav" || default_input_p.extension() == ".idx")
                     default_input_p.replace_extension("");
                     
                // If no parsed args, just export current
                if (args.empty()) {
                     VmxTranscoder::ExportJob job;
                     job.input_path = default_input_p;
                     job.in_point = 0;
                     job.out_point = UINT64_MAX; // Max
                     jobs.push_back(job);
                     CASPAR_LOG(info) << L"Export Job Adding full clip: " << default_input_p.wstring();
                } else {
                     // Loop
                     while (i < args.size()) {
                         // Check if args[i] is a timestamp/frame or a filename
                         auto check = parse_replay_arg_extended(args[i], fps);
                         // Heuristic: If parsing returns valid val OR string is "0", it's a Value
                         bool is_val = (check.first != 0 || check.second || args[i] == L"0");
                         
                         boost::filesystem::path input_p = default_input_p;
                         
                         // If it's NOT a value, it's a filename
                         if (!is_val) {
                             // It is a filename
                             boost::filesystem::path p(u16(args[i]));
                             if (!p.is_absolute()) p = boost::filesystem::path(env::media_folder()) / p;
                             if (p.extension() == ".mav" || p.extension() == ".idx") p.replace_extension("");
                             input_p = p;
                             i++;
                             
                             // If explicit input filename is provided, we use it, 
                             // effectively behaving as a "global" export for this job
                         } 
                         // Else: Use default_input_p (implicit) from the producer context
                         
                         // Now parse In/Out
                         std::pair<uint64_t, bool> in_arg = {0, false};
                         std::pair<uint64_t, bool> out_arg = {0, false};
                         bool in_prov = false;
                         bool out_prov = false;
                         
                         if (i < args.size()) {
                             auto check_in = parse_replay_arg_extended(args[i], fps);
                             bool is_val_in = (check_in.first != 0 || check_in.second || args[i] == L"0");
                             
                             if (is_val_in) {
                                  in_arg = check_in;
                                  in_prov = true;
                                  i++;
                                  
                                  // Try parse OUT
                                  if (i < args.size()) {
                                      auto check_out = parse_replay_arg_extended(args[i], fps);
                                      bool is_val_out = (check_out.first != 0 || check_out.second || args[i] == L"0");
                                      if (is_val_out) {
                                          out_arg = check_out;
                                          out_prov = true;
                                          i++;
                                      }
                                  }
                             }
                         }
                         
                         // Resolve Job
                        ReplaySegmentedReader r;
                        if (r.Open(input_p)) {
                              uint64_t final_in = 0;
                              uint64_t final_out = UINT64_MAX;
                              
                              // In
                              if (in_arg.second) { // Is timestamp
                                   final_in = in_arg.first;
                              } else {
                                   // Frame
                                   size_t idx = (size_t)in_arg.first;
                                   if (idx < r.GetTotalFrames()) r.GetTimestamp(idx, final_in);
                              }
                              
                              // Out
                              if (out_prov) {
                                  if (out_arg.second) {
                                       final_out = out_arg.first;
                                  } else {
                                       size_t idx = (size_t)out_arg.first;
                                       if (idx == 0 && !out_prov) final_out = UINT64_MAX; 
                                       else if (idx < r.GetTotalFrames()) r.GetTimestamp(idx, final_out);
                                  }
                              }
                              
                              VmxTranscoder::ExportJob job;
                              job.input_path = input_p;
                              job.in_point = final_in;
                              job.out_point = final_out;
                              jobs.push_back(job);
                              CASPAR_LOG(info) << L"Export Job Added: " << input_p.wstring() << L" In: " << final_in << L" Out: " << final_out;
                        } else {
                             CASPAR_LOG(error) << L"Export Job Failed to open input: " << input_p.wstring();
                        }
                     }
                }

                if (!jobs.empty()) {
                    boost::filesystem::path out_path(u16(output_filename_w));
                    if (!out_path.is_absolute()) {
                         out_path = boost::filesystem::path(env::media_folder()) / out_path;
                    }
                    CASPAR_LOG(info) << L"Export Process Starting -> " << out_path.wstring();
                    if (VmxTranscoder::ExportJobs(jobs, out_path)) {
                         CASPAR_LOG(info) << L"Export Process Complete.";
                    } else {
                         CASPAR_LOG(error) << L"Export Process Failed.";
                    }
                } else {
                     CASPAR_LOG(warning) << L"Export Process Aborted. No valid jobs.";
                }

            } catch(const std::exception& e) {
                CASPAR_LOG(error) << L"Export Thread Exception: " << e.what();
            } catch(...) {
                CASPAR_LOG(error) << L"Export Thread Unknown Exception";
            }
        }).detach();

        return caspar::make_ready_future(std::wstring(L"Export Started"));
    }

    return caspar::make_ready_future(std::wstring(L"OK"));
}

void replay_producer::configure(const std::vector<std::wstring>& params)
{
    // Skip index 0 (filename)
    for (size_t i = 1; i < params.size(); ++i) {
        std::wstring p = params[i];
        boost::to_upper(p);
        
        if (p == L"LOOP") {
            // LOOP without value means true
            loop_ = true;
        } else if (p == L"IN") {
            if (i + 1 < params.size()) {
                auto arg = parse_replay_arg_extended(params[++i], fps_);
                if (arg.second) {
                    if (reader_) in_point_ = (int64_t)reader_->SeekTimestamp(arg.first);
                } else {
                    in_point_ = (int64_t)arg.first;
                }
            }
        } else if (p == L"OUT") {
            if (i + 1 < params.size()) {
                auto arg = parse_replay_arg_extended(params[++i], fps_);
                if (arg.second) {
                    if (reader_) out_point_ = (int64_t)reader_->SeekTimestamp(arg.first);
                } else {
                    out_point_ = (int64_t)arg.first;
                }
            }
        } else if (p == L"SEEK") {
            if (i + 1 < params.size()) {
                std::wstring val = params[++i];
                if (boost::iequals(val, L"LIVE")) {
                    if (reader_) {
                        reader_->Refresh();
                        int64_t len = (int64_t)reader_->GetTotalFrames();
                        int64_t target = len > 0 ? len - 1 : 0;
                        // Back off 25 frames
                        if (target > 25) target -= 25;
                        else target = 0;
                        frame_num_ = target;
                    }
                } else {
                    auto arg = parse_replay_arg_extended(val, fps_);
                    if (arg.second) {
                        if (reader_) frame_num_ = (int64_t)reader_->SeekTimestamp(arg.first);
                    } else {
                        frame_num_ = (int64_t)arg.first;
                    }
                }
            }
        } else if (p == L"LENGTH") {
             if (i + 1 < params.size()) {
                 auto arg = parse_replay_arg_extended(params[++i], fps_);
                 int64_t len = 0;
                 if (arg.second && fps_ > 0) {
                     len = (int64_t)((double)arg.first / 1000000.0 * fps_);
                 } else {
                     len = (int64_t)arg.first;
                 }
                 out_point_ = in_point_ + len;
             }
        }
    }
}

core::monitor::state replay_producer::state() const { return state_; }

bool replay_producer::is_ready() 
{ 
    if (!vmx_) return false;
    if (!reader_) return false;
    return true; 
}

}}
