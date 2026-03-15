#include <core/frame/frame_factory.h>
#include <core/producer/frame_producer.h>
#include <core/video_format.h>
#include <core/frame/pixel_format.h>
#include <core/frame/draw_frame.h>
#include <core/frame/frame.h>
#include <common/executor.h>
#include <common/memory.h>
#include <common/array.h>
#include <common/log.h>
#include <common/utf.h>

#include "system_audio_producer.h"

#define MINIAUDIO_IMPLEMENTATION
#pragma warning(push)
#pragma warning(disable : 4244)
#include "../miniaudio.h"
#pragma warning(pop)

#include <iostream>
#include <mutex>
#include <atomic>
#include <vector>
#include <queue>
#include <cstring>
#include <algorithm> // For std::copy, std::min
#include <string>

namespace caspar { namespace system_audio {

static const std::wstring PRODUCER_NAME = L"system_audio";

spl::shared_ptr<core::frame_producer> create_producer(const core::frame_producer_dependencies& dependencies,
                                                      const std::vector<std::wstring>&         params)
{
    if (params.empty() || params[0] != L"system_audio") {
        return core::frame_producer::empty();
    }

    std::wstring device_name = L"";
    if (params.size() > 1 && params[1] != L"EMPTY") {
        device_name = params[1];
    }
    
    // Check parameters for DEVICE=... 
    for(size_t i=1; i<params.size(); ++i) {
        if(params[i].rfind(L"DEVICE=", 0) == 0) {
            device_name = params[i].substr(7);
        }
    }
    
    return spl::make_shared<system_audio_producer>(dependencies, device_name, params);
}

class system_audio_producer::impl
{
    ma_context context;
    ma_device device;
    ma_device_config deviceConfig;
    
    std::string device_name_u8;
    
    std::mutex audio_mutex;
    std::vector<int32_t> audio_buffer; // Interleaved S32 samples
    
    core::video_format_desc format_desc;
    spl::shared_ptr<core::frame_factory> frame_factory_;
    
    std::atomic<bool> running{false};

    static void data_callback(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount) {
        impl* self = (impl*)pDevice->pUserData;
        if (!self || !pInput) return;

        std::lock_guard<std::mutex> lock(self->audio_mutex);
        
        // Input is S32, frameCount samples per channel
        const int32_t* input_samples = (const int32_t*)pInput;
        size_t sample_count = frameCount * self->format_desc.audio_channels;
        
        // Append to buffer
        self->audio_buffer.insert(self->audio_buffer.end(), input_samples, input_samples + sample_count);
    }
    
public:
    impl(const core::frame_producer_dependencies& dependencies, const std::wstring& device_name)
        : frame_factory_(dependencies.frame_factory)
        , format_desc(dependencies.format_desc)
    {
        if (!device_name.empty()) {
            device_name_u8 = caspar::u8(device_name);
        }

        if (ma_context_init(NULL, 0, NULL, &context) != MA_SUCCESS) {
            CASPAR_LOG(error) << "system_audio: Failed to init miniaudio context";
            return;
        }

        deviceConfig = ma_device_config_init(ma_device_type_capture);
        deviceConfig.capture.format   = ma_format_s32;
        deviceConfig.capture.channels = format_desc.audio_channels;
        deviceConfig.sampleRate       = format_desc.audio_sample_rate;
        deviceConfig.dataCallback     = data_callback;
        deviceConfig.pUserData        = this;
        
        // Select device if specified
        if (!device_name_u8.empty()) {
             ma_device_info* pPlaybackInfos;
             ma_uint32 playbackCount;
             ma_device_info* pCaptureInfos;
             ma_uint32 captureCount;
             
             if (ma_context_get_devices(&context, &pPlaybackInfos, &playbackCount, &pCaptureInfos, &captureCount) == MA_SUCCESS) {
                 bool found = false;
                 for (ma_uint32 i = 0; i < captureCount; ++i) {
                     std::string name(pCaptureInfos[i].name);
                     if (name == device_name_u8 || name.find(device_name_u8) != std::string::npos) {
                         deviceConfig.capture.pDeviceID = &pCaptureInfos[i].id;
                         CASPAR_LOG(info) << "system_audio: Selected device: " << name;
                         found = true;
                         break;
                     }
                 }
                 if (!found) {
                     CASPAR_LOG(warning) << "system_audio: Device not found: " << device_name_u8 << ". Using default.";
                 }
             }
        } else {
             CASPAR_LOG(info) << "system_audio: Using default capture device";
        }

        if (ma_device_init(&context, &deviceConfig, &device) != MA_SUCCESS) {
            CASPAR_LOG(error) << "system_audio: Failed to init device";
            ma_context_uninit(&context);
            return;
        }

        if (ma_device_start(&device) != MA_SUCCESS) {
             CASPAR_LOG(error) << "system_audio: Failed to start device";
             ma_device_uninit(&device);
             ma_context_uninit(&context);
             return;
        }
        
        running = true;
    }
    
    ~impl() {
        if (running) {
            ma_device_uninit(&device);
        }
        ma_context_uninit(&context);
    }
    
    core::draw_frame get_frame(int nb_samples) {
        // Calculate expected samples per frame
        size_t samples_per_channel;
        if (nb_samples > 0) {
            samples_per_channel = nb_samples;
        } else {
            samples_per_channel = static_cast<size_t>(format_desc.audio_sample_rate * format_desc.duration / format_desc.time_scale); 
        }
        size_t total_samples_needed = samples_per_channel * format_desc.audio_channels;
        
        std::vector<int32_t> output_samples;
        output_samples.resize(total_samples_needed, 0); // Silence by default
        
        {
            std::lock_guard<std::mutex> lock(audio_mutex);
            if (audio_buffer.size() >= total_samples_needed) {
                // We have exactly or more than needed
                std::copy(audio_buffer.begin(), audio_buffer.begin() + total_samples_needed, output_samples.begin());
                audio_buffer.erase(audio_buffer.begin(), audio_buffer.begin() + total_samples_needed);
            } else {
                // Underrun handling
                if (!audio_buffer.empty()) {
                    size_t available = audio_buffer.size();
                    size_t to_copy = std::min(available, total_samples_needed);
                    std::copy(audio_buffer.begin(), audio_buffer.begin() + to_copy, output_samples.begin());
                    audio_buffer.clear();
                    // Could log underrun occasionally
                }
            }
        }
        
        core::pixel_format_desc pix_desc(core::pixel_format::bgra);
        pix_desc.planes.push_back(core::pixel_format_desc::plane(format_desc.width, format_desc.height, 4));
        auto frame = frame_factory_->create_frame(this, pix_desc);
        
        // Populate audio data
        // Using vector move constructor for caspar::array to handle memory safely
        if (!output_samples.empty()) {
            frame.audio_data() = caspar::array<int32_t>(std::move(output_samples));
        }
        
        return core::draw_frame(std::move(frame));
    }
};

system_audio_producer::system_audio_producer(const core::frame_producer_dependencies& dependencies,
                                             const std::wstring&                      device_name,
                                             const std::vector<std::wstring>&         params)
    : impl_(new impl(dependencies, device_name))
{
}

system_audio_producer::~system_audio_producer() = default;

core::draw_frame system_audio_producer::receive_impl(const core::video_field /*field*/, int nb_samples)
{
    return impl_->get_frame(nb_samples);
}

std::wstring system_audio_producer::print() const { return L"system_audio[" + name() + L"]"; }
std::wstring system_audio_producer::name() const { return PRODUCER_NAME; }
core::monitor::state system_audio_producer::state() const { return core::monitor::state(); }

}} // namespace caspar::system_audio