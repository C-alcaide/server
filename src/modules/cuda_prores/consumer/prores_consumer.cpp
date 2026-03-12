// prores_consumer.cpp
// CasparCG frame_consumer implementation for the CUDA ProRes recording pipeline.
//
// Frame flow
// ─────────────────────────────────────────────────────────────────────────────
// Mixer thread → send() → FrameJob enqueued → Encode thread:
//   1. Wait on frame_ready condition
//   2. Copy BGRA pixels to pinned staging buffer (host side)
//   3. cudaMemcpyAsync(d_bgra, h_bgra, HostToDevice, stream)
//   4. k_bgra_to_v210  (CUDA, in-stream)
//   5. prores_encode_frame (CUDA, in-stream)
//   6. cudaStreamSynchronize
//   7. muxer.write_video() + write_timecode()
//   8. muxer.write_audio() if audio present
//   9. promise.set_value(true)
//
// On the first frame, set_start_timecode() is called on the muxer before the
// first write so that MXF timecode metadata is embedded in the file header.
// ─────────────────────────────────────────────────────────────────────────────

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
#include "../cuda/cuda_bgra_to_v210.cuh"
#include "../muxer/mov_muxer.h"
#include "../muxer/mxf_muxer.h"
#include "../timecode.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

namespace caspar { namespace cuda_prores {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static void cuda_check_consumer(cudaError_t e, const char *ctx)
{
    if (e != cudaSuccess) {
        CASPAR_LOG(error) << L"[cuda_prores] CUDA error in " << ctx << ": "
                          << cudaGetErrorString(e);
        throw std::runtime_error(std::string("[cuda_prores] CUDA error: ") + ctx);
    }
}

// Returns the V210 row stride for a given width (DeckLink convention).
static size_t v210_row_bytes(int w) {
    return static_cast<size_t>((w + 5) / 6) * 16;
}
static size_t v210_frame_bytes(int w, int h) {
    return v210_row_bytes(w) * h;
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------
struct prores_config {
    std::wstring output_path;
    std::wstring filename_pattern; // e.g. L"prores_%04d.mov"
    int          profile        = 3;     // 3=HQ
    bool         use_mxf        = false; // false=MOV
    int          device_index   = 0;     // CUDA device
    int          slices_per_row = 4;     // 4 is good for 4K30
};

// ---------------------------------------------------------------------------
// FrameJob — one entry in the encode queue
// ---------------------------------------------------------------------------
struct FrameJob {
    core::const_frame      frame;
    int64_t                frame_number;
    SmpteTimecode          tc;           // synthetic TC from frame counter
    std::promise<bool>     done;
};

// ---------------------------------------------------------------------------
// ProRes frame context helpers (mirrors test/test_prores_encode.cpp)
// ---------------------------------------------------------------------------
static void alloc_frame_ctx(ProResFrameCtx &ctx,
                             int width, int height, int profile,
                             int slices_per_row)
{
    ctx.width          = width;
    ctx.height         = height;
    ctx.profile        = profile;
    ctx.slices_per_row = slices_per_row;
    ctx.q_scale        = 8;

    const int mb_w = width  / 8;
    const int mb_h = height / 8;
    ctx.num_slices       = (mb_w / slices_per_row) * mb_h;
    ctx.blocks_per_slice = slices_per_row * 6;

    const int y_px = width * height;
    const int c_px = (width / 2) * height;
    cuda_check_consumer(cudaMalloc(&ctx.d_y,  y_px * sizeof(int16_t)), "d_y");
    cuda_check_consumer(cudaMalloc(&ctx.d_cb, c_px * sizeof(int16_t)), "d_cb");
    cuda_check_consumer(cudaMalloc(&ctx.d_cr, c_px * sizeof(int16_t)), "d_cr");
    cuda_check_consumer(cudaMalloc(&ctx.d_coeffs_y,  y_px * sizeof(int16_t)), "d_coeffs_y");
    cuda_check_consumer(cudaMalloc(&ctx.d_coeffs_cb, c_px * sizeof(int16_t)), "d_coeffs_cb");
    cuda_check_consumer(cudaMalloc(&ctx.d_coeffs_cr, c_px * sizeof(int16_t)), "d_coeffs_cr");

    const size_t slice_elems = (size_t)ctx.num_slices * ctx.blocks_per_slice * 64;
    cuda_check_consumer(cudaMalloc(&ctx.d_coeffs_slice, slice_elems * sizeof(int16_t)), "d_coeffs_slice");
    const size_t bs_size = slice_elems * sizeof(int16_t) * 2 + ctx.num_slices * 32;
    cuda_check_consumer(cudaMalloc(&ctx.d_bitstream,     bs_size),                              "d_bitstream");
    cuda_check_consumer(cudaMalloc(&ctx.d_slice_offsets, (ctx.num_slices + 1) * sizeof(uint32_t)), "d_slice_offsets");
    cuda_check_consumer(cudaMalloc(&ctx.d_bit_counts,     ctx.num_slices * sizeof(uint32_t)),   "d_bit_counts");

    ctx.cub_temp_bytes = 8 * 1024 * 1024; // 8 MB: sufficient up to ~32K slices
    cuda_check_consumer(cudaMalloc(&ctx.d_cub_temp, ctx.cub_temp_bytes), "d_cub_temp");

    const size_t frame_buf_size = (size_t)width * height * 4;
    ctx.h_frame_buf_size = frame_buf_size;
    cuda_check_consumer(cudaMallocHost(&ctx.h_frame_buf, frame_buf_size), "h_frame_buf");
}

static void free_frame_ctx(ProResFrameCtx &ctx)
{
    cudaFree(ctx.d_y);   cudaFree(ctx.d_cb);  cudaFree(ctx.d_cr);
    cudaFree(ctx.d_coeffs_y); cudaFree(ctx.d_coeffs_cb); cudaFree(ctx.d_coeffs_cr);
    cudaFree(ctx.d_coeffs_slice);
    cudaFree(ctx.d_bitstream); cudaFree(ctx.d_slice_offsets); cudaFree(ctx.d_bit_counts);
    cudaFree(ctx.d_cub_temp);
    cudaFreeHost(ctx.h_frame_buf);
    std::memset(&ctx, 0, sizeof(ctx));
}

// ---------------------------------------------------------------------------
// Consumer implementation
// ---------------------------------------------------------------------------
class prores_consumer_impl : public core::frame_consumer {
public:
    prores_consumer_impl(prores_config cfg, int consumer_index)
        : cfg_(std::move(cfg))
        , index_(consumer_index)
    {
        CASPAR_LOG(info) << L"[cuda_prores] Created consumer #" << index_
                         << L" profile=" << cfg_.profile
                         << L" " << (cfg_.use_mxf ? L"MXF" : L"MOV")
                         << L" → " << cfg_.output_path;
    }

    ~prores_consumer_impl() override
    {
        stop();
    }

    // ── frame_consumer interface ──────────────────────────────────────────

    void initialize(const core::video_format_desc& format_desc,
                    const core::channel_info& /*channel_info*/,
                    int /*port_index*/) override
    {
        format_desc_ = format_desc;
        prores_tables_upload();

        // Select CUDA device
        cuda_check_consumer(cudaSetDevice(cfg_.device_index), "cudaSetDevice");
        cuda_check_consumer(cudaStreamCreateWithFlags(&encode_stream_, cudaStreamNonBlocking),
                            "cudaStreamCreateWithFlags");

        // Allocate GPU resources
        alloc_frame_ctx(frame_ctx_, format_desc_.width, format_desc_.height,
                        cfg_.profile, cfg_.slices_per_row);

        // Pinned staging buffers: BGRA (input) and V210 (intermediate)
        const size_t bgra_bytes = (size_t)format_desc_.width * format_desc_.height * 4;
        const size_t v210_bytes = v210_frame_bytes(format_desc_.width, format_desc_.height);
        cuda_check_consumer(cudaMallocHost(&h_bgra_, bgra_bytes),  "h_bgra_");
        cuda_check_consumer(cudaMalloc(&d_bgra_, bgra_bytes),      "d_bgra_");
        cuda_check_consumer(cudaMalloc(&d_v210_, v210_bytes),      "d_v210_");

        // Derive output file path
        // Build filename from pattern or default scheme
        const std::wstring filename =
            cfg_.filename_pattern.empty()
                ? build_filename(frame_number_)
                : cfg_.filename_pattern;
        const std::wstring full_path = cfg_.output_path + L"\\" + filename;

        // Determine ProRes fourcc from profile
        static const uint32_t PRORES_FOURCC[] = {
            0x6170636Fu, // 'apco' Proxy
            0x6170636Cu, // 'apcl' LT
            0x6170636Eu, // 'apcn' Standard
            0x61706368u, // 'apch' HQ
            0x61703468u, // 'ap4h' 4444
        };
        const uint32_t fourcc = PRORES_FOURCC[std::min(cfg_.profile, 4)];

        // Open the muxer
        if (cfg_.use_mxf) {
            MxfVideoTrackInfo vi{};
            vi.width  = format_desc_.width;
            vi.height = format_desc_.height;
            vi.frame_rate = { format_desc_.fps_den, format_desc_.fps_num };
            vi.prores_fourcc = fourcc;
            vi.color = MXF_COLOR_SDR_709;

            MxfAudioTrackInfo ai{};
            ai.channels    = 16;
            ai.sample_rate = 48000;

            mxf_muxer_ = std::make_unique<MxfMuxer>();
            if (!mxf_muxer_->open(full_path.c_str(), vi, ai)) {
                CASPAR_LOG(error) << L"[cuda_prores] Failed to open MXF: " << full_path;
                mxf_muxer_.reset();
            }
        } else {
            MovVideoTrackInfo vi{};
            vi.width  = format_desc_.width;
            vi.height = format_desc_.height;
            vi.timebase_num = (uint32_t)format_desc_.fps_den; // duration per frame
            vi.timebase_den = (uint32_t)format_desc_.fps_num; // timescale
            vi.prores_fourcc = fourcc;
            vi.color.color_primaries    = 1; // Rec.709
            vi.color.transfer_function  = 1;
            vi.color.color_matrix       = 1;
            vi.color.has_hdr            = false;

            MovAudioTrackInfo ai{};
            ai.channels    = 16;
            ai.sample_rate = 48000;

            mov_muxer_ = std::make_unique<MovMuxer>();
            if (!mov_muxer_->open(full_path, vi, ai)) {
                CASPAR_LOG(error) << L"[cuda_prores] Failed to open MOV: " << full_path;
                mov_muxer_.reset();
            }
        }

        // Start encode thread
        encode_thread_ = std::thread(&prores_consumer_impl::encode_loop, this);

        CASPAR_LOG(info) << L"[cuda_prores] Recording → " << full_path;
    }

    std::future<bool> send(const core::video_field /*field*/, core::const_frame frame) override
    {
        if (!frame) return caspar::make_ready_future(false);
        if (!running_) return caspar::make_ready_future(false);

        // Build a synthetic SMPTE timecode from the frame counter.
        // If DeckLink capture is integrated at a later stage, the real TC
        // from the CaptureToken should be passed through instead.
        const uint32_t fps = (uint32_t)(format_desc_.fps + 0.5);
        const int64_t  fn  = frame_number_++;
        SmpteTimecode tc;
        tc.valid      = true;
        tc.drop_frame = false;
        tc.frames     = (uint8_t)(fn % fps);
        const int64_t total_sec = fn / fps;
        tc.seconds = (uint8_t)(total_sec % 60);
        tc.minutes = (uint8_t)((total_sec / 60) % 60);
        tc.hours   = (uint8_t)((total_sec / 3600) % 24);

        FrameJob job;
        job.frame        = std::move(frame);
        job.frame_number = fn;
        job.tc           = tc;

        auto future = job.done.get_future();
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            if (frame_queue_.size() >= kMaxQueueDepth) {
                // Drop oldest frame to avoid unbounded growth under heavy load
                frame_queue_.front().done.set_value(false);
                frame_queue_.pop();
                CASPAR_LOG(warning) << L"[cuda_prores] Queue overflow — dropped oldest frame";
            }
            frame_queue_.push(std::move(job));
        }
        queue_cv_.notify_one();
        return future;
    }

    std::future<bool> call(const std::vector<std::wstring>& params) override
    {
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
        std::lock_guard<std::mutex> lock(state_mutex_);
        return state_;
    }

    std::wstring name()  const override { return L"cuda_prores"; }
    int          index() const override { return index_; }
    std::wstring print() const override
    {
        return L"cuda_prores[" + std::to_wstring(index_) + L"|"
               + std::to_wstring(cfg_.profile) + L"]";
    }

private:
    // ── Encode loop (runs on encode_thread_) ─────────────────────────────
    void encode_loop()
    {
        bool first_frame = true;

        while (running_ || !queue_empty()) {
            FrameJob job;
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                queue_cv_.wait_for(lock, std::chrono::milliseconds(50),
                    [this]{ return !frame_queue_.empty() || !running_; });
                if (frame_queue_.empty()) continue;
                job = std::move(frame_queue_.front());
                frame_queue_.pop();
            }

            bool ok = encode_one(job, first_frame);
            first_frame = false;
            job.done.set_value(ok);
        }

        // Flush and close muxers
        if (mov_muxer_) { mov_muxer_->close(); mov_muxer_.reset(); }
        if (mxf_muxer_) { mxf_muxer_->close(); mxf_muxer_.reset(); }

        CASPAR_LOG(info) << L"[cuda_prores] Encode thread exited cleanly.";
    }

    bool encode_one(FrameJob &job, bool first_frame)
    {
        const auto &fmt = format_desc_;

        // 1. Copy BGRA from CasparCG frame to pinned staging buffer
        const uint8_t *frame_pixels = job.frame.image_data(0).begin();
        const size_t   bgra_bytes   = (size_t)fmt.width * fmt.height * 4;
        std::memcpy(h_bgra_, frame_pixels, std::min(bgra_bytes, job.frame.image_data(0).size()));

        // 2. H→D async copy of BGRA
        cudaError_t err = cudaMemcpyAsync(
            d_bgra_, h_bgra_, bgra_bytes, cudaMemcpyHostToDevice, encode_stream_);
        if (err != cudaSuccess) {
            CASPAR_LOG(error) << L"[cuda_prores] cudaMemcpyAsync(bgra) failed: "
                              << cudaGetErrorString(err);
            return false;
        }

        // 3. BGRA → V210 conversion (on GPU)
        err = launch_bgra_to_v210(d_bgra_, d_v210_, fmt.width, fmt.height, encode_stream_);
        if (err != cudaSuccess) {
            CASPAR_LOG(error) << L"[cuda_prores] launch_bgra_to_v210 failed: "
                              << cudaGetErrorString(err);
            return false;
        }

        // If this is the first frame, set the start timecode on the MXF muxer
        // (must happen before avformat_write_header, which is deferred to first write)
        if (first_frame && mxf_muxer_ && job.tc.valid)
            mxf_muxer_->set_start_timecode(job.tc);

        // 4. ProRes encode (V210 → ProRes bitstream in h_frame_buf)
        static const ProResColorDesc k_sdr_709 = {1, 1, 1, {}, {}, 0, 0, 0, 0, 0, 0};
        size_t encoded_size = 0;
        err = prores_encode_frame(
            &frame_ctx_,
            (const uint32_t *)d_v210_,
            frame_ctx_.h_frame_buf,
            &encoded_size,
            encode_stream_,
            &k_sdr_709);
        if (err != cudaSuccess) {
            CASPAR_LOG(error) << L"[cuda_prores] prores_encode_frame failed: "
                              << cudaGetErrorString(err);
            return false;
        }

        // 5. Mux video frame
        if (mov_muxer_) {
            if (!mov_muxer_->write_video(frame_ctx_.h_frame_buf, encoded_size, job.frame_number))
                return false;
            if (job.tc.valid)
                mov_muxer_->write_timecode(job.tc);
        }
        if (mxf_muxer_) {
            if (!mxf_muxer_->write_video(frame_ctx_.h_frame_buf, encoded_size, job.frame_number))
                return false;
        }

        // 6. Update diagnostics state
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            state_[L"frame"]   = job.frame_number;
            state_[L"profile"] = cfg_.profile;
        }

        return true;
    }

    bool queue_empty() const {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        return frame_queue_.empty();
    }

    void stop()
    {
        if (!running_.exchange(false)) return; // already stopped
        queue_cv_.notify_all();
        if (encode_thread_.joinable())
            encode_thread_.join();

        // GPU resource cleanup
        if (encode_stream_) {
            cudaStreamSynchronize(encode_stream_);
            cudaStreamDestroy(encode_stream_);
            encode_stream_ = nullptr;
        }
        if (frame_ctx_.d_y) free_frame_ctx(frame_ctx_);
        if (h_bgra_) { cudaFreeHost(h_bgra_); h_bgra_ = nullptr; }
        if (d_bgra_) { cudaFree(d_bgra_);     d_bgra_ = nullptr; }
        if (d_v210_) { cudaFree(d_v210_);     d_v210_ = nullptr; }
    }

    std::wstring build_filename(int64_t /*fn*/) const
    {
        // Default: single file named by timestamp or fixed name
        return cfg_.use_mxf ? L"prores.mxf" : L"prores.mov";
    }

    // ── Configuration ─────────────────────────────────────────────────────
    prores_config            cfg_;
    int                      index_;
    core::video_format_desc  format_desc_;

    // ── CUDA resources ────────────────────────────────────────────────────
    cudaStream_t             encode_stream_ = nullptr;
    ProResFrameCtx           frame_ctx_     = {};
    uint8_t                 *h_bgra_        = nullptr; // pinned staging
    uint8_t                 *d_bgra_        = nullptr; // device BGRA
    uint32_t                *d_v210_        = nullptr; // device V210

    // ── Muxers (only one is active at a time) ─────────────────────────────
    std::unique_ptr<MovMuxer>  mov_muxer_;
    std::unique_ptr<MxfMuxer>  mxf_muxer_;

    // ── Encode queue ──────────────────────────────────────────────────────
    static constexpr size_t kMaxQueueDepth = 8;
    mutable std::mutex       queue_mutex_;
    std::condition_variable  queue_cv_;
    std::queue<FrameJob>     frame_queue_;
    std::thread              encode_thread_;

    // ── State ─────────────────────────────────────────────────────────────
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
    cfg.filename_pattern = elem.get(L"filename", L"");
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


}} // namespace caspar::cuda_prores
