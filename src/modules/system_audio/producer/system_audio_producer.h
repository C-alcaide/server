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

    // Non-standard producer extras (no override — not in frame_producer base)
    bool    is_looping() const { return false; }
    void    set_looping(bool /*loop*/) {}
    bool    is_seekable() const { return false; }
    void    seek(int64_t /*to*/) {}
    int64_t duration() const { return -1; }
    int64_t time() const { return -1; }
    int     index() const { return 1000; }
    bool    has_synchronization_clock() const { return false; }

private:
    class impl;
    std::unique_ptr<impl> impl_;
};

}} // namespace caspar::system_audio