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
 * This module links against libltc (https://github.com/x42/libltc),
 * licensed under LGPL-2.1+, which is compatible with GPL-3.
 */

#pragma once
#include <string>
#include <vector>
#include <mutex>
#include <chrono>
#include <cstdint>

namespace caspar { namespace ltc {
    class LTCInput {
    public: 
        static LTCInput& instance();
        void start();
        std::string get_current_timecode_string();
        uint32_t get_current_frame_number(int fps);
        bool is_valid();

        // Genlock anchor: maps the most-recently decoded house timecode to the
        // steady_clock instant it was captured, so consumers can align their own
        // steady_clock-stamped data to the house frame grid.
        // Returns false if there is no currently-valid LTC signal.
        //   out_frame = absolute frame number of that timecode at `fps`
        //   out_time  = steady_clock instant the timecode was decoded
        bool get_timecode_anchor(int fps,
                                 uint32_t&                              out_frame,
                                 std::chrono::steady_clock::time_point& out_time);
        
        // Device management
        std::vector<std::string> get_capture_devices();

        /// Select the capture device. **Returns true if a stream was opened, which is NOT the
        /// same as the requested device having been found** -- an unknown name falls back to
        /// the default input and still succeeds. That fallback is deliberate: a missing audio
        /// interface must not take a playout server down.
        ///
        /// The comment here used to read "Returns true if found and set", which was wrong from
        /// the day the fallback was written, and `INFO LTC` reported the REQUESTED name as
        /// though it were the open one -- so an operator who mistyped a device was told they
        /// had it. Ask `get_active_device_name()` for what is actually open and
        /// `is_device_fallback()` for whether the request was honoured.
        bool set_capture_device(const std::string& device_name);

        /// The name the operator ASKED for, verbatim. Empty if none was ever set.
        std::string get_current_device_name();

        /// The device actually OPEN, from PortAudio's own table. Differs from the above
        /// whenever the requested name did not resolve; empty if no stream is open.
        std::string get_active_device_name();

        /// Was a device requested by name and not found, so the default is open instead?
        bool is_device_fallback();

        bool is_using_system_clock();
        void shutdown();
    };
}}