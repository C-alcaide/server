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