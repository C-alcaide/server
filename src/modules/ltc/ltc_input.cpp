#include <ltc.h>
#include <decoder.h>
#include "ltc_input.h"
#include <mutex>
#include <atomic>
#include <string>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <chrono>
#include <ctime>

#include <common/env.h>
#include <common/utf.h>
#include <boost/property_tree/ptree.hpp>

#define MINIAUDIO_IMPLEMENTATION
#pragma warning(push)
#pragma warning(disable : 4244)
#include "miniaudio.h"
#pragma warning(pop)

namespace caspar { namespace ltc {
    
static const wchar_t* LTC_CONFIG_ROOT = L"configuration.ltc";

class LTCInputImpl {
    ma_context context;
    ma_device device;
    ma_device_config deviceConfig;
    LTCDecoder* decoder = nullptr;
    std::mutex timecode_mutex;
    SMPTETimecode last_timecode = {0};
    
    // We hold the latest valid frame for retrieval
    std::atomic<bool> valid_signal{false};
    std::atomic<bool> running{false};

    // System Time Fallback
    std::chrono::time_point<std::chrono::steady_clock> last_signal_time;

    // Device Management
    ma_device_info* pCaptureInfos = nullptr;
    ma_uint32 captureCount = 0;
    ma_device_info* pPlaybackInfos = nullptr;
    ma_uint32 playbackCount = 0;
    std::string current_device_name;
    std::mutex device_mutex;

    static void data_callback(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount) {
        // This runs on audio thread
        LTCInputImpl* self = (LTCInputImpl*)pDevice->pUserData;
        if (self && self->decoder) {
            // Write to decoder
            // miniaudio f32 -> ltc expects float (if supported) or u8
            // libltc decoder.c supports float
            ltc_decoder_write_float(self->decoder, const_cast<float*>(static_cast<const float*>(pInput)), frameCount, 0); 
            
            // We use LTCFrameExt for the decoder read, then convert to SMPTETimecode
            LTCFrameExt ltc_frame;
            bool got_frame = false;
            SMPTETimecode temp_tc = {0};
            while(ltc_decoder_read(self->decoder, &ltc_frame)) {
               ltc_frame_to_time(&temp_tc, &ltc_frame.ltc, 0);
               got_frame = true;
            }

            if (got_frame) {
                std::lock_guard<std::mutex> lock(self->timecode_mutex);
                self->last_timecode = temp_tc;
                self->valid_signal = true;
                self->last_signal_time = std::chrono::steady_clock::now();
            }
        }
    }

public:
    static LTCInputImpl& instance() {
        static LTCInputImpl instance;
        return instance;
    }
    
    // Member initialization
    LTCInputImpl() {
        if (ma_context_init(NULL, 0, NULL, &context) != MA_SUCCESS) {
             std::cerr << "Failed to init miniaudio context\n";
        }
    };
    
    ~LTCInputImpl() {
        if (running) {
             ma_device_uninit(&device);
        }
        ma_context_uninit(&context);
    }

    void start() {
        std::lock_guard<std::mutex> lock(device_mutex);
        if (running) return;
        
        // Read configuration
        try {
            auto& pt = caspar::env::properties();
            boost::optional<std::wstring> dev = pt.get_optional<std::wstring>(L"configuration.ltc.device");
            if (dev && current_device_name.empty()) { // Use config if current is empty (first start)
                 current_device_name = caspar::u8(*dev);
            }
        } catch (...) {
            // Ignore if config not found or invalid
        }

        // Create decoder (48kHz sample rate, 25fps default - it auto-detects anyway usually)
        if (!decoder) decoder = ltc_decoder_create(48000, 25);  
        
        deviceConfig = ma_device_config_init(ma_device_type_capture);
        deviceConfig.capture.format = ma_format_f32;
        deviceConfig.capture.channels = 1;
        deviceConfig.sampleRate = 48000;
        deviceConfig.dataCallback = data_callback;
        deviceConfig.pUserData = this;

        // Find device ID if name is set
        ma_device_id* pDeviceID = NULL;
        if (!current_device_name.empty()) {
             // Ensure devices are enumerated
             if (!pCaptureInfos) {
                 ma_context_get_devices(&context, &pPlaybackInfos, &playbackCount, &pCaptureInfos, &captureCount);
             }
             
             for (ma_uint32 i = 0; i < captureCount; ++i) {
                 if (current_device_name == pCaptureInfos[i].name) {
                     pDeviceID = &pCaptureInfos[i].id;
                     break;
                 }
             }
        }

        // Initialize device
        if (ma_device_init(&context, &deviceConfig, &device) != MA_SUCCESS) {
            // Try with default if specific fails? Or just fail?
            // Fallback to default if specific failed
            if (pDeviceID != NULL) {
                 std::cerr << "Failed to init specific device, trying default.\n";
                 if (ma_device_init(&context, &deviceConfig, &device) != MA_SUCCESS) {
                     std::cerr << "Failed to init miniaudio device for LTC\n";
                     return;
                 }
            } else {
                std::cerr << "Failed to init miniaudio device for LTC\n";
                return;
            }
        }
        
        // If we used default (pDeviceID was NULL or failed), we should update current_device_name
        if (pDeviceID == NULL) {
             // Get device info to populate name? 
             // miniaudio API for getting name of initialized device is ma_device_get_name?
             // Actually we can just say "Default" if we don't know.
             // Or better, re-enumerate and check isDefault.
        }

        if (ma_device_start(&device) == MA_SUCCESS) {
            running = true;
        }
    }

    std::vector<std::string> get_capture_devices() {
        std::lock_guard<std::mutex> lock(device_mutex);
        if (!pCaptureInfos) {
            ma_context_get_devices(&context, &pPlaybackInfos, &playbackCount, &pCaptureInfos, &captureCount);
        }
        std::vector<std::string> devices;
        for (ma_uint32 i = 0; i < captureCount; ++i) {
            devices.push_back(pCaptureInfos[i].name);
        }
        return devices;
    }

    bool set_capture_device(const std::string& name) {
        std::lock_guard<std::mutex> lock(device_mutex);
        
        if (running) {
             ma_device_stop(&device);
             ma_device_uninit(&device);
             running = false;
        }
        
        current_device_name = name;
        
        // Restart (unlocking/locking happens in start, but we hold lock here so we need careful logic)
        // Actually start() locks device_mutex. That would deadlock.
        // I should refactor start() to not lock, or have internal method.
        // Let's modify start() to NOT lock, and caller locks.
        // Wait, start() is public via instance().start().
        
        // Let's release lock before calling member functions that lock, but that's unsafe.
        // Better: Make internal methods or recursive mutex.
        // For simplicity, let's just do the logic inline here.
        
        if (!decoder) decoder = ltc_decoder_create(48000, 25);
        
        deviceConfig = ma_device_config_init(ma_device_type_capture);
        deviceConfig.capture.format = ma_format_f32;
        deviceConfig.capture.channels = 1;
        deviceConfig.sampleRate = 48000;
        deviceConfig.dataCallback = data_callback;
        deviceConfig.pUserData = this;
        
        ma_device_id* pDeviceID = NULL;
        
        // Ensure devices are enumerated
        if (!pCaptureInfos) {
             ma_context_get_devices(&context, &pPlaybackInfos, &playbackCount, &pCaptureInfos, &captureCount);
        }
         
        bool found = false;
        for (ma_uint32 i = 0; i < captureCount; ++i) {
             if (name == pCaptureInfos[i].name) {
                 pDeviceID = &pCaptureInfos[i].id;
                 found = true;
                 break;
             }
        }
        
        if (!found && !name.empty() && name != "Default") {
            // Device name not found
            // Fallback to default/empty?
            // Keep running false?
            return false; 
        }

        if (pDeviceID != NULL) {
            deviceConfig.capture.pDeviceID = pDeviceID;
        }

        if (ma_device_init(&context, &deviceConfig, &device) != MA_SUCCESS) { // 3-arg form for miniaudio
             std::cerr << "Failed to init device\n";
             return false;
        }

        if (ma_device_start(&device) == MA_SUCCESS) {
            running = true;
        }
        return true;
    }

    std::string get_current_device_name() {
         return current_device_name.empty() ? "Default" : current_device_name;
    }
    
    // ... [Rest of existing methods: get_current_timecode_string, etc.]
    
    std::string get_current_timecode_string() {
        std::lock_guard<std::mutex> lock(timecode_mutex);
        
        bool use_fallback = !valid_signal;
        
        if (valid_signal) {
             auto now = std::chrono::steady_clock::now();
             // 1 second timeout for signal loss
             if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_signal_time).count() > 1000) {
                 use_fallback = true;
             }
        }

        if (use_fallback) {
             time_t now = time(0);
             struct tm tstruct;
             localtime_s(&tstruct, &now); 
             char buffer[16];
             sprintf(buffer, "%02d:%02d:%02d:00", tstruct.tm_hour, tstruct.tm_min, tstruct.tm_sec);
             return std::string(buffer);
        }

        char buffer[16];
        sprintf(buffer, "%02d:%02d:%02d:%02d", 
            last_timecode.hours, last_timecode.mins, 
            last_timecode.secs, last_timecode.frame);
        return std::string(buffer);
    }
    
    uint32_t get_current_frame_number(int fps) {
         // ... [Identical to before]
         std::lock_guard<std::mutex> lock(timecode_mutex);
         
         bool use_fallback = !valid_signal;
        
         if (valid_signal) {
             auto now = std::chrono::steady_clock::now();
             if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_signal_time).count() > 1000) {
                 use_fallback = true;
             }
         }

         if (use_fallback) {
             time_t now = time(0);
             struct tm tstruct;
             localtime_s(&tstruct, &now); 
             return (tstruct.tm_hour * 3600 + tstruct.tm_min * 60 + tstruct.tm_sec) * fps;
         }

         return (last_timecode.hours * 3600 + last_timecode.mins * 60 + last_timecode.secs) * fps + last_timecode.frame;
    }
    
    bool is_valid() {
        if (!valid_signal) return false;
        
        // Signal lost check
        std::lock_guard<std::mutex> lock(timecode_mutex);
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_signal_time).count() > 1000) {
             return false;
        }
        return true;
    }
    
    bool is_using_system_clock() {
        return !is_valid();
    }
};

LTCInput& LTCInput::instance() {
    static LTCInput wrapper;
    return wrapper;
}
void LTCInput::start() { LTCInputImpl::instance().start(); }
std::string LTCInput::get_current_timecode_string() { return LTCInputImpl::instance().get_current_timecode_string(); }
uint32_t LTCInput::get_current_frame_number(int fps) { return LTCInputImpl::instance().get_current_frame_number(fps); }
bool LTCInput::is_valid() { return LTCInputImpl::instance().is_valid(); }

// New methods bindings
std::vector<std::string> LTCInput::get_capture_devices() { return LTCInputImpl::instance().get_capture_devices(); }
bool LTCInput::set_capture_device(const std::string& name) { return LTCInputImpl::instance().set_capture_device(name); }
std::string LTCInput::get_current_device_name() { return LTCInputImpl::instance().get_current_device_name(); }
bool LTCInput::is_using_system_clock() { return LTCInputImpl::instance().is_using_system_clock(); }

void init() {
    LTCInput::instance().start();
}

}} 
