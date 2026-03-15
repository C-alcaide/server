#pragma once

#include <core/producer/frame_producer.h>
#include <core/frame/frame_factory.h>
#include <core/video_format.h>
#include <common/diagnostics/graph.h>
#include <string>
#include <mutex>
#include <atomic>
#include <thread>
#include <vector>
#include "vmx_segmented_storage.h"
#include "vmxcodec.h"

// Link libvmx
#pragma comment(lib, "libvmx.lib")

namespace caspar { namespace vmx {

struct vmx_producer : public core::frame_producer
{
    core::monitor::state    state_;
    mutable std::mutex      state_mutex_;

    std::string             path_;
    int                     channel_index_ = -1;
    
    std::unique_ptr<VmxSegmentedReader> reader_;

    VMX_INSTANCE*           vmx_ = nullptr;
    int                     width_ = 0;
    int                     height_ = 0;
    double                  fps_ = 25.0;
    
    // FPS counter
    std::chrono::steady_clock::time_point last_fps_update_;
    int                     frames_since_update_ = 0;
    int                     refresh_skip_counter_ = 0;
    double                  current_fps_ = 0.0;
    
    // Diagnostics
    spl::shared_ptr<diagnostics::graph> graph_;
    
    // Playback state
    std::atomic<int64_t>    frame_num_ = 0;
    std::atomic<double>     speed_ = 1.0;
    double                  fractional_frame_ = 0.0;
    std::atomic<bool>       loop_ = false;
    std::atomic<int64_t>    duration_ = 0;
    std::atomic<int64_t>    in_point_ = 0;
    std::atomic<int64_t>    out_point_ = -1; // -1 means end of file

    // Decoding
    std::vector<uint8_t>    read_buffer_;
    core::draw_frame        current_frame_;

public:
    vmx_producer(std::string path, spl::shared_ptr<core::frame_factory> frame_factory);
    ~vmx_producer();

    core::draw_frame receive_impl(core::video_field field, int nb_samples) override;
    std::future<std::wstring> call(const std::vector<std::wstring>& params) override;
    bool is_ready() override;
    
    // Configures producer from PLAY command arguments
    void configure(const std::vector<std::wstring>& params);

    std::wstring         print() const override { return L"vmx[" + u16(path_) + L"]"; }
    std::wstring         name() const override { return L"vmx"; }
    core::monitor::state state() const override;

private:
    spl::shared_ptr<core::frame_factory> frame_factory_;
};

}}
