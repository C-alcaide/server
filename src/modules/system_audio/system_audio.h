#pragma once

#include <core/module_dependencies.h>
#include <core/producer/frame_producer.h>

#include <string>
#include <vector>

namespace caspar { namespace system_audio {

void init(const core::module_dependencies& dependencies);

void uninit();

spl::shared_ptr<core::frame_producer> create_producer(const core::frame_producer_dependencies& dependencies,
                                                      const std::vector<std::wstring>&         params);

}} // namespace caspar::system_audio