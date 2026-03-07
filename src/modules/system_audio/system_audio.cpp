#include "system_audio.h"

#include <core/module_dependencies.h>
#include <core/producer/frame_producer_registry.h>

// enumerate_capture_devices() is implemented in producer/system_audio_producer.cpp
// where MINIAUDIO_IMPLEMENTATION is defined.

namespace caspar { namespace system_audio {

void init(const core::module_dependencies& dependencies)
{
    dependencies.producer_registry->register_producer_factory(L"system_audio", create_producer);
}

void uninit()
{
}

}} // namespace caspar::system_audio