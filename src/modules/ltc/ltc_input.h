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

namespace caspar { namespace ltc {
    class LTCInput {
    public: 
        static LTCInput& instance();
        void start();
        std::string get_current_timecode_string();
        uint32_t get_current_frame_number(int fps);
        bool is_valid();
        
        // Device management
        std::vector<std::string> get_capture_devices();
        bool set_capture_device(const std::string& device_name); // Returns true if found and set
        std::string get_current_device_name();
        bool is_using_system_clock();
    };
}}