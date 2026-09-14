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
         
         // Add milliseconds to avoid sub-second collisions
         auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
             now.time_since_epoch()) % 1000;
         ss << "_" << std::setfill('0') << std::setw(3) << ms.count();
         
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
    // -- DRAIN THE QUEUE BEFORE CLOSING ANYTHING ---------------------------------------
    //
    // The worker holds raw pointers into `vmx_` and `writer_`, so the ORDER here is the whole
    // of its safety: ask it to stop, let it finish the frames it already has, join it, and
    // only then close the writer and destroy the codec. `worker_loop` exits on an EMPTY
    // queue rather than on the flag, so "stop" means "finish what you accepted" -- the
    // frames in it are frames the operator believes are recorded.
    worker_stop_.store(true, std::memory_order_release);
    queue_cv_.notify_all();
    if (worker_.joinable())
        worker_.join();

    // Close writer first to flush pending data and finalize index
    if (writer_) {
        try {
            writer_->Close();
        } catch (...) {
            CASPAR_LOG_CURRENT_EXCEPTION();
        }
    }
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

    // STARTED HERE, NOT IN THE CONSTRUCTOR, because the worker touches `vmx_`, `width_`,
    // `height_` and the open writer, and none of those exist until this call: a consumer is
    // constructed from the AMCP string and only learns the channel's format when it is added.
    if (vmx_ && writer_ && !worker_.joinable())
        worker_ = std::thread([this] { worker_loop(); });
}

std::future<bool> replay_consumer::send(core::video_field field, const core::const_frame frame)
{
    if (!vmx_ || !writer_)
        return make_ready_future(false);

    // -- THE CHANNEL THREAD DOES A COPY AND A PUSH, AND NOTHING ELSE --------------------
    //
    // Everything that was here -- the VMX encode, the payload assembly, `fwrite` + `fflush`
    // twice -- is on `worker_loop()` now. `output.cpp` waits on this future and measures the
    // wait as the channel's remaining headroom, so what happens between here and the return
    // is paid by every layer on the channel, not only by the recording.
    //
    // Measured 2026-09-14 at 1080p50 before this split: `consume_load` 0.0003 idle against
    // 0.127 recording -- an eighth of the frame budget, on the thread that paces the channel.
    //
    // The copy is not avoidable: the const_frame's host buffer is recycled as soon as the
    // channel moves on, and holding the frame alive instead would pin a mixer readback
    // buffer for as long as the queue is deep.
    auto& data_array = frame.image_data(0);
    if (data_array.size() == 0 || data_array.data() == nullptr)
        return make_ready_future(true);

    int stride = (int)width_ * 4; // BGRA 32bit assumed
    if (data_array.size() >= (size_t)(stride * height_))
        stride = (int)(data_array.size() / height_);

    pending_frame pf;
    pf.stride = stride;
    pf.bgra.assign(data_array.data(), data_array.data() + data_array.size());
    auto& audio_vec = frame.audio_data();
    pf.audio.assign(audio_vec.begin(), audio_vec.end());

    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        if (queue_.size() >= queue_capacity_) {
            // -- DROP, DO NOT BLOCK, AND NEVER RETURN `false` --------------------------
            //
            // Blocking here would put the disk's worst case straight back onto the channel,
            // which is the whole defect this queue exists to remove. Returning `false` would
            // be worse still: `output::do_send` reads a false future as "this consumer has
            // failed" and erases it from `consumers_` with no log line at all, so a single
            // slow moment would end a buffer armed for hours.
            //
            // A dropped frame is one frame missing from the recording, and the index carries
            // real timestamps, so playback stays correct across the gap.
            frames_dropped_.fetch_add(1, std::memory_order_relaxed);
            if (!drop_warned_) {
                drop_warned_ = true;
                CASPAR_LOG(warning) << print()
                                    << L" write queue full; dropping frames. The volume is not keeping up "
                                       L"with the channel. Logged once; the count is published as "
                                       L"`dropped_frames`.";
            }
            graph_->set_tag(diagnostics::tag_severity::WARNING, "dropped-frame");
            return make_ready_future(true);
        }
        queue_.push_back(std::move(pf));
        graph_->set_value("queue", (double)queue_.size() / (double)queue_capacity_);
    }
    queue_cv_.notify_one();

    return make_ready_future(true);
}

void replay_consumer::worker_loop()
{
    // The encode buffer, the FPS window and `frames_written_` belong to this thread now:
    // nothing else touches them, so none of them needs a lock.
    while (true) {
        pending_frame f;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait(lock,
                           [this] { return worker_stop_.load(std::memory_order_acquire) || !queue_.empty(); });
            // DRAIN BEFORE STOPPING. The frames already accepted from the channel are frames
            // the operator believes are recorded, and the segment index has to be finalised
            // over all of them -- so the exit is "the queue is empty", not "stop was asked".
            if (queue_.empty())
                return;
            f = std::move(queue_.front());
            queue_.pop_front();
        }

        try {
            write_one(f);
        } catch (const std::exception&) {
            // `std::exception` rather than `...`: this tree builds with /EHa, under which a
            // catch-all also catches access violations and would turn memory corruption into
            // a recording that quietly stops.
            CASPAR_LOG_CURRENT_EXCEPTION();
        }
    }
}

void replay_consumer::write_one(pending_frame& f)
{
    auto now = std::chrono::steady_clock::now();
    frames_since_update_++;
    auto duration_sec = std::chrono::duration_cast<std::chrono::duration<double>>(now - last_fps_update_).count();

    if (duration_sec >= 1.0) {
        current_fps_         = frames_since_update_ / duration_sec;
        frames_since_update_ = 0;
        last_fps_update_     = now;

        if (graph_.get()) {
            std::wstringstream stats;
            stats.precision(2);
            stats << std::fixed;
            stats << print() << L" - Fps: " << current_fps_ << L" frames:" << frames_written_
                  << L" time:" << (double)frames_written_ / fps_;
            graph_->set_text(stats.str());
        }
    }

    // Assume progressive for now (0)
    int interlaced = 0;

    int res = VMX_EncodeBGRA(vmx_, (unsigned char*)f.bgra.data(), f.stride, interlaced);
    if (res != VMX_ERR_OK) {
        // A BAD FRAME SKIPS A FRAME; IT DOES NOT END THE RECORDING. This used to return
        // `false` from `send()`, which `output::do_send` reads as "this consumer has failed"
        // and acts on by erasing it from `consumers_` with no log line of any kind.
        if (!encode_failure_warned_) {
            encode_failure_warned_ = true;
            CASPAR_LOG(warning) << print() << L" VMX encode rejected a frame (error " << res
                                << L"); skipping it and continuing to record. Logged once.";
        }
        return;
    }

    // Reuse pre-allocated encode buffer instead of allocating per-frame
    size_t max_size = (size_t)width_ * height_ * 4;
    if (encode_buffer_.size() < max_size) {
        encode_buffer_.resize(max_size);
    }

    int size = VMX_SaveTo(vmx_, encode_buffer_.data(), (int)max_size);
    /*
    Format per frame in .mav:
       uint32 audio_size
       byte[] audio_data
       ... VMX stream ...
    */

    if (size > 0) {
        uint32_t audio_bytes = (uint32_t)(f.audio.size() * sizeof(int32_t));

        // Aggregate payload
        size_t               total_size = sizeof(uint32_t) + audio_bytes + (size_t)size;
        std::vector<uint8_t> payload(total_size);

        uint8_t* ptr = payload.data();
        memcpy(ptr, &audio_bytes, sizeof(uint32_t));
        ptr += sizeof(uint32_t);

        if (audio_bytes > 0) {
            memcpy(ptr, f.audio.data(), audio_bytes);
            ptr += audio_bytes;
        }

        memcpy(ptr, encode_buffer_.data(), (size_t)size);

        // Timestamp (Microseconds since epoch)
        uint64_t timestamp =
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch())
                .count();

        writer_->WriteFrame(payload.data(), payload.size(), timestamp);

        frames_written_++;

        graph_->set_value("buffered-video", (double)size / (double)max_size);
    }

    // Published from the worker because `dropped_frames` is the only number that says the
    // volume is not keeping up -- the future this consumer returns is always ready, so it
    // cannot carry that and never could.
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        state_["dropped_frames"] = frames_dropped_.load(std::memory_order_relaxed);
        state_["frames_written"] = frames_written_;
    }
}

core::monitor::state replay_consumer::state() const
{
    std::lock_guard<std::mutex> lock(state_mutex_);
    return state_;
}

}}
