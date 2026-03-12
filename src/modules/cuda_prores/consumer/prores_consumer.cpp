// prores_consumer.cpp
// CUDA ProRes consumer: receives composited CasparCG frames, encodes to
// ProRes 422 / 4444 via the GPU pipeline, writes to .mov or .mxf.
//
// Implementation notes
// ─────────────────────────────────────────────────────────────────────────────
// • Frames arrive from CasparCG's mixer as BGRA or YUV422P10 depending on the
//   channel format.  We convert to V210 on-GPU before feeding the ProRes encoder.
// • An encoder thread owns the cudaStream_t and file writer to avoid blocking
//   the mixer thread.  Frame data is transferred via a bounded SPSC queue.
// • GPU conversion BGRA→YUV422P10 (with correct Rec.709 matrix) is written
//   as a CUDA kernel in the next phase; for the current skeleton we capture
//   the frame pixels into a staging buffer without GPU conversion.
//
// Phase 5 TODO items are marked with "// TODO(Phase5)".
#include "prores_consumer.h"

#include <common/except.h>
#include <common/log.h>
#include <common/param.h>
#include <common/memory.h>

#include <core/consumer/frame_consumer.h>
#include <core/frame/frame.h>
#include <core/video_format.h>
#include <core/monitor/monitor.h>
#include <core/channel_info.h>

#include <boost/property_tree/ptree.hpp>
#include <boost/algorithm/string.hpp>

#include <cuda_runtime.h>

#include "../cuda/cuda_prores_frame.h"
#include "../cuda/cuda_prores_tables.cuh"
#include "../muxer/mov_muxer.h"
#include "../muxer/mxf_muxer.h"

#include <atomic>
#include <chrono>
#include <future>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace caspar { namespace cuda_prores {

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------
struct prores_config {
    std::wstring output_path;     // directory
    std::wstring filename_pattern; // e.g. L"prores_%04d.mov" (frame-based) or fixed
    int          profile        = 3;     // 3=HQ
    bool         use_mxf        = false; // false=MOV
    int          device_index   = 0;     // CUDA device
};

// ---------------------------------------------------------------------------
// Consumer
// ---------------------------------------------------------------------------
class prores_consumer_impl : public core::frame_consumer {
public:
    prores_consumer_impl(prores_config cfg, int consumer_index)
        : cfg_(std::move(cfg))
        , index_(consumer_index)
    {
        CASPAR_LOG(info) << L"[cuda_prores] Initialised consumer #" << index_
                         << L" profile=" << cfg_.profile
                         << L" " << (cfg_.use_mxf ? L"MXF" : L"MOV")
                         << L" → " << cfg_.output_path;
    }

    ~prores_consumer_impl()
    {
        stop();
    }

    // ── frame_consumer interface ──────────────────────────────────────────

    void initialize(const core::video_format_desc& format_desc,
                    const core::channel_info& /*channel_info*/,
                    int /*port_index*/) override
    {
        format_desc_ = format_desc;

        // Upload ProRes tables to CUDA constant memory
        prores_tables_upload();

        // TODO(Phase5): create ProResFrameCtx, open file writer, start encode thread
        CASPAR_LOG(info) << L"[cuda_prores] initialize "
                         << format_desc_.width << L"×" << format_desc_.height
                         << L" " << format_desc_.fps << L"fps";
    }

    std::future<bool> send(const core::video_field /*field*/, core::const_frame frame) override
    {
        if (!frame) return caspar::make_ready_future(false);

        // TODO(Phase5): push frame into encode queue, return future that resolves
        //               when the GPU has consumed the frame data.
        // For now: accept frame and return true (no-op skeleton).
        (void)frame;
        ++frame_number_;
        return caspar::make_ready_future(true);
    }

    std::future<bool> call(const std::vector<std::wstring>& params) override
    {
        // AMCP CALL support: e.g. CALL 1-10 START / STOP
        if (!params.empty()) {
            auto cmd = boost::to_upper_copy(params[0]);
            if (cmd == L"STOP") {
                stop();
                return caspar::make_ready_future(true);
            }
        }
        return caspar::make_ready_future(false);
    }

    core::monitor::state state() const override
    {
        core::monitor::state s;
        // Populate state map for diagnostics
        std::lock_guard<std::mutex> lock(state_mutex_);
        return state_;
    }

    std::wstring name() const override  { return L"cuda_prores"; }
    int          index() const override { return index_; }
    std::wstring print() const override
    {
        return L"cuda_prores[" + std::to_wstring(index_) + L"|"
               + std::to_wstring(cfg_.profile) + L"]";
    }

private:
    void stop()
    {
        running_ = false;
        // TODO(Phase5): signal encode thread, flush + close file writer
    }

    prores_config            cfg_;
    int                      index_;
    core::video_format_desc  format_desc_;
    std::atomic<bool>        running_{true};
    int64_t                  frame_number_ = 0;

    mutable std::mutex       state_mutex_;
    core::monitor::state     state_;
};

// ---------------------------------------------------------------------------
// Factory helpers
// ---------------------------------------------------------------------------
static prores_config parse_params(const std::vector<std::wstring>& params)
{
    prores_config cfg;
    cfg.output_path = caspar::get_param(L"PATH", params, L".");
    cfg.profile     = caspar::get_param(L"PROFILE", params, 3);
    auto codec      = caspar::get_param(L"CODEC", params, std::wstring(L"MOV"));
    cfg.use_mxf     = boost::iequals(codec, L"MXF");
    return cfg;
}

static prores_config parse_xml(const boost::property_tree::wptree& elem)
{
    prores_config cfg;
    cfg.output_path      = elem.get(L"path",    L".");
    cfg.filename_pattern = elem.get(L"filename", L"prores.mov");
    cfg.profile          = elem.get(L"profile",  3);
    auto codec = elem.get(L"codec", std::wstring(L"mov"));
    cfg.use_mxf = boost::iequals(codec, L"mxf");
    return cfg;
}

// ---------------------------------------------------------------------------
// Exported factory functions
// ---------------------------------------------------------------------------
spl::shared_ptr<core::frame_consumer>
create_consumer(const std::vector<std::wstring>& params,
                const core::video_format_repository& /*format_repository*/,
                const std::vector<spl::shared_ptr<core::video_channel>>& /*channels*/,
                const core::channel_info& /*channel_info*/)
{
    if (boost::to_upper_copy(params.at(0)) != L"CUDA_PRORES")
        return core::frame_consumer::empty();

    return spl::make_shared<prores_consumer_impl>(parse_params(params), 1);
}

spl::shared_ptr<core::frame_consumer>
create_preconfigured_consumer(const boost::property_tree::wptree& element,
                              const core::video_format_repository& /*format_repository*/,
                              const std::vector<spl::shared_ptr<core::video_channel>>& /*channels*/,
                              const core::channel_info& /*channel_info*/)
{
    return spl::make_shared<prores_consumer_impl>(parse_xml(element), 1);
}

}} // namespace caspar::cuda_prores
