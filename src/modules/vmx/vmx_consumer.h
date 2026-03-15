#pragma once

#include <core/consumer/frame_consumer.h>
#include <core/consumer/channel_info.h>
#include <core/frame/frame.h>
#include <core/video_format.h>
#include <common/utf.h>
#include <string>
#include <mutex>
#include <atomic>
#include <thread>
#include <vector>
#include "vmx_segmented_storage.h"
#include <common/diagnostics/graph.h>

// VMX Header
#include "vmxcodec.h" 

// Link libvmx
#pragma comment(lib, "libvmx.lib")

namespace caspar { namespace vmx {

struct vmx_consumer : public core::frame_consumer
{
    core::monitor::state    state_;
    mutable std::mutex      state_mutex_;
    std::string             path_;
    int                     channel_index_ = -1;
    
    std::unique_ptr<VmxSegmentedWriter> writer_;
    int                     max_duration_sec_;
    int                     segment_duration_sec_;

    VMX_INSTANCE*           vmx_ = nullptr;
    int                     width_ = 0;
    int                     height_ = 0;
    
    // FPS counter
    std::chrono::steady_clock::time_point last_fps_update_;
    int                     frames_since_update_ = 0;
    double                  current_fps_ = 0.0;
    
    // Stats
    int64_t                 frames_written_ = 0;
    double                  fps_ = 25.0;
    
    VMX_PROFILE             quality_ = VMX_PROFILE_SQ;
    
    // Audio buffer
    std::vector<int32_t>    audio_buffer_;
    
    // Diagnostics
    spl::shared_ptr<diagnostics::graph> graph_;

public:
    vmx_consumer(std::string path, VMX_PROFILE quality);
    ~vmx_consumer();

    void initialize(const core::video_format_desc& format_desc,
                    const core::channel_info&      channel_info,
                    int                            port_index) override;

    std::future<bool> send(core::video_field field, core::const_frame frame) override;

    std::wstring print() const override { return L"vmx[" + u16(path_) + L"]"; }
    std::wstring name() const override { return L"vmx"; }
    bool has_synchronization_clock() const override { return false; }
    int index() const override { return 200000 + channel_index_; }
    core::monitor::state state() const override;
};
}}
