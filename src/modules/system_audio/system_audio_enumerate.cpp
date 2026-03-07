/**
 * system_audio_enumerate.cpp
 *
 * Standalone translation unit for enumerate_capture_devices().
 * This is kept separate so that MINIAUDIO_IMPLEMENTATION is defined exactly
 * once and does not interfere with other modules.  The broken-override
 * methods in system_audio_producer.h do not affect this TU.
 */

#define MA_NO_DECODING
#define MA_NO_ENCODING
#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

#include "system_audio.h"

#include <string>
#include <vector>

namespace caspar { namespace system_audio {

std::vector<std::wstring> enumerate_capture_devices()
{
    std::vector<std::wstring> result;

    ma_context ctx;
    if (ma_context_init(nullptr, 0, nullptr, &ctx) != MA_SUCCESS)
        return result;

    ma_device_info* playback_infos = nullptr;
    ma_uint32       playback_count = 0;
    ma_device_info* capture_infos  = nullptr;
    ma_uint32       capture_count  = 0;

    if (ma_context_get_devices(&ctx,
                               &playback_infos, &playback_count,
                               &capture_infos,  &capture_count) == MA_SUCCESS) {
        for (ma_uint32 i = 0; i < capture_count; ++i) {
            std::string n(capture_infos[i].name);
            result.push_back(std::wstring(n.begin(), n.end()));
        }
    }

    ma_context_uninit(&ctx);
    return result;
}

}} // namespace caspar::system_audio
