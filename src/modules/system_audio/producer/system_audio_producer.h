#pragma once

#include <core/frame/frame_factory.h>
#include <core/producer/frame_producer.h>
#include <common/executor.h>

#include <string>
#include <vector>

namespace caspar { namespace system_audio {

class system_audio_producer : public core::frame_producer
{
public:
    explicit system_audio_producer(const core::frame_producer_dependencies& dependencies,
                                   const std::wstring&                      device_name,
                                   const std::vector<std::wstring>&         params);
    ~system_audio_producer();

    core::draw_frame receive_impl(const core::video_field, int) override;
    
    std::wstring         print() const override;
    std::wstring         name() const override;
    core::monitor::state state() const override;
    bool                 is_ready() override { return true; }

private:
    struct impl;
    std::unique_ptr<impl> impl_;
};

}} // namespace caspar::system_audio