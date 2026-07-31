/*
 * Copyright (c) 2026 CasparCG Contributors
 *
 * This file is part of CasparCG (www.casparcg.com).
 *
 * CasparCG is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * CasparCG is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with CasparCG. If not, see <http://www.gnu.org/licenses/>.
 */

#include "remotewall_producer.h"

#include <core/frame/draw_frame.h>
#include <core/frame/frame.h>
#include <core/frame/frame_factory.h>
#include <core/frame/pixel_format.h>
#include <core/monitor/monitor.h>
#include <core/producer/frame_producer.h>
#include <core/video_format.h>

#include <common/array.h>
#include <common/bit_depth.h>
#include <common/log.h>
#include <common/utf.h>

#include <boost/algorithm/string.hpp>

#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <future>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "RecvWall.h" // vendored receiver C API

#ifdef ENABLE_VULKAN
#include <accelerator/vulkan/image/image_mixer.h>
#include <accelerator/vulkan/util/device.h>

#include <cuda_runtime.h>

#include "cuda_vk_texture.h" // src/modules/cuda_vk_texture.h (via the .. include dir)
#endif

namespace caspar { namespace remotewall {

module_config& config()
{
    static module_config c;
    return c;
}

namespace {

// Map the stream's free-text colour tag (e.g. "BT709", "BT2020/PQ", "BT2020/HLG") to CasparCG's
// colour-space / transfer enums so the mixer applies the right colour handling / tone-map.
std::pair<core::color_space, core::color_transfer> parse_colorspace(const char* cs)
{
    // cs is always info_.colorSpace, a fixed char[16] wire field with no
    // guaranteed NUL terminator if all 16 bytes are used — bound the length
    // instead of treating it as a plain C-string.
    std::string s = cs ? std::string(cs, ::strnlen(cs, 16)) : std::string();
    for (auto& ch : s)
        ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));

    core::color_space    space = core::color_space::bt709;
    core::color_transfer trc   = core::color_transfer::sdr;
    if (s.find("2020") != std::string::npos)
        space = core::color_space::bt2020;
    else if (s.find("601") != std::string::npos)
        space = core::color_space::bt601;
    if (s.find("PQ") != std::string::npos || s.find("2084") != std::string::npos)
        trc = core::color_transfer::pq;
    else if (s.find("HLG") != std::string::npos)
        trc = core::color_transfer::hlg;
    return {space, trc};
}

// Native cloudXR tile-wall producer. On the Vulkan mixer it publishes the receiver's device
// composite straight into an exportable VK texture (zero-copy, no readback); elsewhere it falls
// back to a CPU readback. Renders a marker frame until the wall is receiving.
class remotewall_producer : public core::frame_producer
{
    spl::shared_ptr<core::frame_factory> frame_factory_;
    core::video_format_desc              format_desc_;
    const int                            fallback_w_;
    const int                            fallback_h_;
    const int                            port_;
    const int                            tiles_;
    const int                            codec_;

    RecvWallHandle*           rw_ = nullptr;
    RecvWallConfig            cfg_{};
    int                       device_ = 0;
    std::mutex                rw_mtx_;    // guards rw_ + pool during a live reconfigure (SET)
    bool                      zero_copy_ok_ = false;
    std::vector<std::uint8_t> staging_;
    RecvWallFrameInfo         info_{};
    unsigned long long        ver_        = 0;
    int                       wall_w_     = 0;
    int                       wall_h_     = 0;
    bool                      have_frame_ = false;
    core::monitor::state      state_;
    mutable std::mutex        state_mtx_; // guards state_ (written on mixer thread, read on monitor thread)

#ifdef ENABLE_VULKAN
    std::shared_ptr<accelerator::vulkan::device>   vk_device_;
    bool                                           use_vulkan_ = false;
    std::vector<std::shared_ptr<CudaVkTexture>>    pool_;
    std::shared_ptr<CudaVkTexture>                 bound_;    // current GPU write target
    int                                            pool_w_ = 0;
    int                                            pool_h_ = 0;
    int                                            pool_bytes_ = 4; // 4 = RGBA8, 8 = RGBA16 (HDR)
    core::draw_frame                               last_df_ = core::draw_frame::empty();
#endif

  public:
    remotewall_producer(const core::frame_producer_dependencies& deps,
                        const RecvWallConfig&                    cfg,
                        int                                      device,
                        bool                                     device_auto)
        : frame_factory_(deps.frame_factory)
        , format_desc_(deps.format_desc)
        , fallback_w_(deps.format_desc.width)
        , fallback_h_(deps.format_desc.height)
        , port_(cfg.listenPort)
        , tiles_(cfg.expectedTiles)
        , codec_(cfg.codec)
    {
        cfg_    = cfg;
        device_ = device;
#ifdef ENABLE_VULKAN
        if (auto* vk_mixer = dynamic_cast<accelerator::vulkan::image_mixer*>(frame_factory_.get())) {
            vk_device_  = vk_mixer->get_vk_device();
            use_vulkan_ = vk_device_ != nullptr;
        }
        // Multi-GPU: unless the user pinned DEVICE, decode on the same physical GPU the Vulkan
        // mixer runs on (match by device UUID) so the CUDA composite and the VK texture live on
        // one GPU (no cross-device copy for the zero-copy path).
        if (use_vulkan_ && device_auto) {
            const int matched = cuda_device_for_vk();
            if (matched >= 0)
                device_ = matched;
        }
#endif
        init_receiver();
    }

    ~remotewall_producer() { teardown_receiver(); }

#ifdef ENABLE_VULKAN
    // Find the CUDA device index whose UUID matches the Vulkan mixer's physical GPU (-1 if none).
    int cuda_device_for_vk()
    {
        if (!vk_device_)
            return -1;
        vk::PhysicalDevice             phys = vk_device_->getVkPhysicalDevice();
        vk::PhysicalDeviceIDProperties idp{};
        vk::PhysicalDeviceProperties2  p2{};
        p2.pNext = &idp;
        phys.getProperties2(&p2);

        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess)
            return -1;
        for (int i = 0; i < count; ++i) {
            cudaDeviceProp prop{};
            if (cudaGetDeviceProperties(&prop, i) != cudaSuccess)
                continue;
            if (std::memcmp(prop.uuid.bytes, idp.deviceUUID.data(), 16) == 0)
                return i;
        }
        return -1;
    }
#endif

    // (Re)bind the receiver from cfg_. Zero-copy engages only on the Vulkan mixer without a sync
    // group (sync groups use the receiver's host-side reorder buffer, so the composite is not
    // published GPU-direct). Safe to call again after teardown_receiver() for a live reconfigure.
    void init_receiver()
    {
#ifdef ENABLE_VULKAN
        if (use_vulkan_)
            cudaSetDevice(device_);
        zero_copy_ok_ = use_vulkan_ && cfg_.syncGroup[0] == 0;
#endif
        cfg_.cudaDevice = device_; // decode on the resolved (mixer-matched) GPU
        rw_         = RecvWallInit(&cfg_);
        ver_        = 0;
        have_frame_ = false;
        wall_w_ = wall_h_ = 0;
        staging_.clear();
        CASPAR_LOG(info) << L"[remotewall] receiver " << (rw_ ? L"started" : L"FAILED") << L" on UDP port "
                         << static_cast<int>(cfg_.listenPort) << L" (tiles=" << cfg_.expectedTiles
                         << L", codec=" << cfg_.codec << L", device=" << device_ << L", path="
#ifdef ENABLE_VULKAN
                         << (zero_copy_ok_ ? L"vulkan-zerocopy" : L"cpu-readback")
#else
                         << L"cpu-readback"
#endif
                         << L", syncgroup='" << u16(cfg_.syncGroup) << L"').";
    }

    void teardown_receiver()
    {
        if (!rw_)
            return;
#ifdef ENABLE_VULKAN
        if (use_vulkan_)
            RecvWallUnbindCudaArray(rw_);
        bound_.reset();
        pool_.clear();
        pool_w_ = pool_h_ = 0;
        last_df_ = core::draw_frame::empty();
#endif
        RecvWallShutdown(rw_);
        rw_ = nullptr;
    }

    core::draw_frame make_marker_frame()
    {
        core::pixel_format_desc pfd(core::pixel_format::bgra);
        pfd.planes.push_back(core::pixel_format_desc::plane(fallback_w_, fallback_h_, 4));
        auto              frame = frame_factory_->create_frame(this, pfd);
        auto*             data  = frame.image_data(0).data();
        const std::size_t n     = static_cast<std::size_t>(fallback_w_) * fallback_h_ * 4;
        for (std::size_t i = 0; i + 4 <= n; i += 4) {
            data[i + 0] = 64;
            data[i + 1] = 48;
            data[i + 2] = 16;
            data[i + 3] = 255;
        }
        return core::draw_frame(std::move(frame));
    }

    core::draw_frame receive_impl(const core::video_field /*field*/, int /*nb_samples*/) override
    {
        std::lock_guard<std::mutex> lk(rw_mtx_);
#ifdef ENABLE_VULKAN
        if (zero_copy_ok_ && rw_)
            return receive_vulkan();
#endif
        return receive_cpu();
    }

    // CPU-readback path: copy the top-down BGRA composite into a wall-sized frame.
    core::draw_frame receive_cpu()
    {
        int bytes = 4;
        if (rw_) {
            unsigned gw = 0, gh = 0;
            if (RecvWallGetGeometry(rw_, &gw, &gh, nullptr, nullptr, nullptr, nullptr) && gw > 0 && gh > 0) {
                bytes                  = RecvWallGetPixelBytes(rw_);
                const std::size_t need = static_cast<std::size_t>(gw) * gh * bytes;
                if (staging_.size() != need)
                    staging_.assign(need, 0);
                RecvWallWaitNewFrame(rw_, ver_, 8);
                if (RecvWallGetLatest(rw_,
                                      staging_.data(),
                                      static_cast<int>(gw) * bytes,
                                      static_cast<int>(staging_.size()),
                                      &ver_,
                                      &info_) == 1) {
                    have_frame_ = true;
                }
                wall_w_ = static_cast<int>(gw);
                wall_h_ = static_cast<int>(gh);
            }
        }

        if (!have_frame_ || wall_w_ <= 0 || wall_h_ <= 0)
            return make_marker_frame();

        update_state(wall_w_, wall_h_);
        const auto              cst   = parse_colorspace(info_.colorSpace);
        const common::bit_depth depth = bytes == 8 ? common::bit_depth::bit16 : common::bit_depth::bit8;
        core::pixel_format_desc pfd(core::pixel_format::bgra, cst.first, cst.second);
        pfd.planes.push_back(core::pixel_format_desc::plane(wall_w_, wall_h_, 4, depth));

        auto      frame      = frame_factory_->create_frame(this, pfd);
        const int dst_stride = frame.pixel_format_desc().planes[0].linesize;
        const int src_stride = wall_w_ * bytes;
        auto*     dst        = frame.image_data(0).data();
        for (int y = 0; y < wall_h_; ++y)
            std::memcpy(dst + static_cast<std::size_t>(y) * dst_stride,
                        staging_.data() + static_cast<std::size_t>(y) * src_stride,
                        static_cast<std::size_t>(src_stride));
        return core::draw_frame(std::move(frame));
    }

#ifdef ENABLE_VULKAN
    // Zero-copy path: the receiver writes the composite straight into a bound exportable VK
    // texture (device->array). We double-buffer across a small pool so the mixer reads a stable
    // texture while the next wall is written into a fresh one.
    void ensure_pool(int w, int h)
    {
        const int bytes = rw_ ? RecvWallGetPixelBytes(rw_) : 4;
        if (pool_w_ == w && pool_h_ == h && pool_bytes_ == bytes && !pool_.empty())
            return;
        RecvWallUnbindCudaArray(rw_);
        bound_.reset();
        pool_.clear();
        last_df_ = core::draw_frame::empty();
        cudaSetDevice(device_);
        const common::bit_depth depth  = bytes == 8 ? common::bit_depth::bit16 : common::bit_depth::bit8;
        constexpr int           kSlots = 4;
        for (int i = 0; i < kSlots; ++i) {
            auto vk_tex = vk_device_->create_exportable_texture(w, h, 4, depth);
            pool_.push_back(
                std::make_shared<CudaVkTexture>(vk_tex, static_cast<VkDevice>(vk_device_->getVkDevice())));
        }
        pool_w_     = w;
        pool_h_     = h;
        pool_bytes_ = bytes;
    }

    std::shared_ptr<CudaVkTexture> pick_free(const std::shared_ptr<CudaVkTexture>& except)
    {
        for (auto& t : pool_)
            if (t != except && t->is_free())
                return t;
        return nullptr;
    }

    core::draw_frame make_texture_frame(const std::shared_ptr<CudaVkTexture>& cvt, int w, int h)
    {
        const auto              cst   = parse_colorspace(info_.colorSpace);
        const common::bit_depth depth = pool_bytes_ == 8 ? common::bit_depth::bit16 : common::bit_depth::bit8;
        core::pixel_format_desc pfd(core::pixel_format::bgra, cst.first, cst.second);
        pfd.is_straight_alpha = false; // composite alpha is forced opaque
        pfd.planes.push_back(core::pixel_format_desc::plane(w, h, 4, depth));

        auto                                   store = std::make_shared<std::vector<std::uint8_t>>(0);
        array<const std::uint8_t>              dummy(store->data(), 0, std::move(store));
        std::vector<array<const std::uint8_t>> img_vec;
        img_vec.push_back(std::move(dummy));

        auto                      astore = std::make_shared<std::vector<std::int32_t>>(0);
        array<const std::int32_t> audio(astore->data(), 0, std::move(astore));

        return core::draw_frame(core::const_frame(this, std::move(img_vec), std::move(audio), pfd, cvt->core_texture()));
    }

    core::draw_frame receive_vulkan()
    {
        unsigned gw = 0, gh = 0;
        if (!RecvWallGetGeometry(rw_, &gw, &gh, nullptr, nullptr, nullptr, nullptr) || gw == 0 || gh == 0)
            return last_df_ ? last_df_ : make_marker_frame();

        wall_w_ = static_cast<int>(gw);
        wall_h_ = static_cast<int>(gh);
        ensure_pool(static_cast<int>(gw), static_cast<int>(gh));

        if (!bound_) {
            bound_ = pick_free(nullptr);
            if (bound_)
                RecvWallBindCudaArray(rw_, bound_->array());
        }
        if (!bound_)
            return last_df_ ? last_df_ : make_marker_frame();

        RecvWallWaitNewFrame(rw_, ver_, 8);
        if (RecvWallPeekInfo(rw_, &ver_, &info_) == 1) {
            // A new wall was written into `bound_`. Present it, then rebind a fresh free slot so
            // the next wall does not overwrite the frame the mixer is about to read.
            auto present = bound_;
            have_frame_  = true;
            update_state(static_cast<int>(gw), static_cast<int>(gh));

            auto next = pick_free(present);
            if (next) {
                bound_ = next;
                RecvWallBindCudaArray(rw_, bound_->array());
            } // else: pool exhausted this tick; keep bound_ = present (drop-oldest live edge)

            last_df_ = make_texture_frame(present, static_cast<int>(gw), static_cast<int>(gh));
        }

        return last_df_ ? last_df_ : make_marker_frame();
    }
#endif

    // Publish per-frame metadata (geometry, timecode, colour, FEC/drops, and the full source camera)
    // as monitor state -> emitted over OSC, and available to CALL ... REMOTEWALL INFO/CAMERA.
    void update_state(int w, int h)
    {
        RecvWallStats st{};
        if (rw_)
            RecvWallGetStats(rw_, &st);

        core::monitor::state s;
        s["remotewall/port"]           = static_cast<int>(cfg_.listenPort);
        s["remotewall/wall"]           = std::vector<int>{w, h};
        s["remotewall/grid"]           = std::vector<int>{static_cast<int>(info_.gridCols),
                                                          static_cast<int>(info_.gridRows)};
        s["remotewall/fps"]            = info_.fpsDen ? static_cast<double>(info_.fpsNum) / info_.fpsDen : 0.0;
        s["remotewall/colorspace"]     = std::string(info_.colorSpace, ::strnlen(info_.colorSpace, sizeof(info_.colorSpace)));
        s["remotewall/frame"]          = static_cast<double>(info_.globalFrameIndex);
        if (info_.tcValid) {
            char tc[24];
            std::snprintf(tc, sizeof tc, "%02u:%02u:%02u%c%02u", info_.tcHours, info_.tcMinutes,
                          info_.tcSeconds, info_.tcDrop ? ';' : ':', info_.tcFrames);
            s["remotewall/timecode"] = std::string(tc);
        }
        s["remotewall/fec-recovered"]  = static_cast<double>(st.fecRecovered);
        s["remotewall/drops"]          = static_cast<double>(st.auQueueDrops);
        s["remotewall/frames-decoded"] = static_cast<double>(st.framesDecoded);
        s["remotewall/bit-depth"]      = rw_ ? RecvWallGetPixelBytes(rw_) * 2 : 8; // bits/channel (8 or 16)
        s["remotewall/device"]         = device_;

        // Full source camera (CameraMeta: camToWorld/worldToCam/proj 4x4, intrinsics, fov, frustum).
        const float* cam = info_.cam;
        s["remotewall/camera/camtoworld"] = std::vector<double>(cam, cam + 16);
        s["remotewall/camera/worldtocam"] = std::vector<double>(cam + 16, cam + 32);
        s["remotewall/camera/proj"]       = std::vector<double>(cam + 32, cam + 48);
        s["remotewall/camera/intrinsics"] = std::vector<double>{cam[48], cam[49], cam[50], cam[51]};
        s["remotewall/camera/fov"]        = std::vector<double>{cam[52], cam[53]};
        s["remotewall/camera/frustum"] =
            std::vector<double>{cam[56], cam[57], cam[58], cam[59], cam[60], cam[61]};
        std::lock_guard<std::mutex> lk(state_mtx_);
        state_ = s;
    }

    // AMCP: CALL <ch-layer> REMOTEWALL INFO | CAMERA
    std::future<std::wstring> call(const std::vector<std::wstring>& params) override
    {
        std::wstring result;
        // Was only taken by the SET branch; INFO/CAMERA read rw_/info_ without
        // it, racing a concurrent SET's teardown_receiver()/init_receiver().
        std::lock_guard<std::mutex> lk(rw_mtx_);
        if (!params.empty() && boost::iequals(params.at(0), L"REMOTEWALL")) {
            const std::wstring sub = params.size() > 1 ? params.at(1) : L"";
            if (boost::iequals(sub, L"INFO")) {
                RecvWallStats st{};
                if (rw_)
                    RecvWallGetStats(rw_, &st);
                char tc[24] = "--:--:--:--";
                if (info_.tcValid)
                    std::snprintf(tc, sizeof tc, "%02u:%02u:%02u%c%02u", info_.tcHours, info_.tcMinutes,
                                  info_.tcSeconds, info_.tcDrop ? ';' : ':', info_.tcFrames);
                std::ostringstream os;
                os << "port " << static_cast<int>(cfg_.listenPort) << " wall " << wall_w_ << "x" << wall_h_ << " grid "
                   << info_.gridCols << "x" << info_.gridRows << " fps "
                   << (info_.fpsDen ? static_cast<double>(info_.fpsNum) / info_.fpsDen : 0.0) << " tc " << tc
                   << " colorspace " << std::string(info_.colorSpace, ::strnlen(info_.colorSpace, sizeof(info_.colorSpace)))
                   << " depth " << (rw_ ? RecvWallGetPixelBytes(rw_) * 2 : 8)
                   << " device " << device_ << " frame " << info_.globalFrameIndex << " fec " << st.fecRecovered
                   << " drops " << st.auQueueDrops << "\r\n";
                result = u16(os.str());
            } else if (boost::iequals(sub, L"CAMERA")) {
                const float*       cam = info_.cam;
                std::ostringstream os;
                os << "intrinsics fx " << cam[48] << " fy " << cam[49] << " cx " << cam[50] << " cy " << cam[51]
                   << " fov " << cam[52] << " " << cam[53] << " frustum " << cam[56] << " " << cam[57] << " "
                   << cam[58] << " " << cam[59] << " " << cam[60] << " " << cam[61] << "\r\n";
                os << "proj";
                for (int i = 32; i < 48; ++i)
                    os << " " << cam[i];
                os << "\r\n";
                result = u16(os.str());
            } else if (boost::iequals(sub, L"SET") && params.size() >= 4) {
                // Live reconfigure: rebind the receiver with the new value. Guarded so it does not
                // race receive_impl. Keys: port | syncgroup | bindip.
                const std::wstring key = params.at(2);
                const std::wstring val = params.at(3);
                bool ok = true;
                if (boost::iequals(key, L"port")) {
                    try {
                        cfg_.listenPort = static_cast<unsigned short>(std::stoi(val));
                    } catch (...) {
                        ok = false;
                    }
                } else if (boost::iequals(key, L"syncgroup")) {
                    std::snprintf(cfg_.syncGroup, sizeof cfg_.syncGroup, "%s", u8(val).c_str());
                } else if (boost::iequals(key, L"bindip")) {
                    std::snprintf(cfg_.bindIp, sizeof cfg_.bindIp, "%s", u8(val).c_str());
                } else {
                    ok = false;
                }
                if (ok) {
                    teardown_receiver();
                    init_receiver();
                }
                result = ok ? L"" : L"402 CALL ERROR (unknown REMOTEWALL SET key)\r\n";
            }
        }
        std::promise<std::wstring> pr;
        pr.set_value(result);
        return pr.get_future();
    }

    core::monitor::state state() const override
    {
        std::lock_guard<std::mutex> lk(state_mtx_);
        return state_;
    }
    std::wstring         print() const override { return L"remotewall[" + std::to_wstring(cfg_.listenPort) + L"]"; }
    std::wstring         name() const override { return L"remotewall"; }
    bool                 is_ready() override { return true; }
};

} // namespace

spl::shared_ptr<core::frame_producer> create_producer(const core::frame_producer_dependencies& dependencies,
                                                      const std::vector<std::wstring>&          params)
{
    if (params.empty() || !boost::iequals(params.at(0), L"REMOTEWALL"))
        return core::frame_producer::empty();

    int          port        = config().listen_port;
    int          tiles       = 0; // auto-detect from the first SyncMeta
    int          codec       = 0; // 0 = HEVC, 1 = H.264, 2 = AV1
    int          device      = config().cuda_device;
    bool         device_auto = true; // match the Vulkan mixer's GPU unless DEVICE is given
    std::wstring bind_ip;
    std::wstring sync_group;
    for (std::size_t i = 1; i < params.size(); ++i) {
        if (boost::iequals(params[i], L"PORT") && i + 1 < params.size()) {
            try {
                port = std::stoi(params[++i]);
            } catch (...) {
            }
        } else if (boost::iequals(params[i], L"TILES") && i + 1 < params.size()) {
            try {
                tiles = std::stoi(params[++i]);
            } catch (...) {
            }
        } else if (boost::iequals(params[i], L"CODEC") && i + 1 < params.size()) {
            const std::wstring c = params[++i];
            codec                = boost::iequals(c, L"h264") ? 1 : boost::iequals(c, L"av1") ? 2 : 0;
        } else if (boost::iequals(params[i], L"DEVICE") && i + 1 < params.size()) {
            try {
                device      = std::stoi(params[++i]);
                device_auto = false;
            } catch (...) {
            }
        } else if (boost::iequals(params[i], L"BINDIP") && i + 1 < params.size()) {
            bind_ip = params[++i];
        } else if (boost::iequals(params[i], L"SYNCGROUP") && i + 1 < params.size()) {
            sync_group = params[++i];
        }
    }

    RecvWallConfig cfg{};
    cfg.listenPort    = static_cast<unsigned short>(port);
    cfg.expectedTiles = tiles;
    cfg.codec         = codec;
    cfg.fullRange     = 1; // full-swing RGB
    cfg.pixelOrder    = 0; // 0 = BGRA (matches the bgra frame tag)
    cfg.useGpuConvert = 1; // CUDA NV12->BGRA composite (lib falls back to CPU on failure)
    std::snprintf(cfg.bindIp, sizeof cfg.bindIp, "%s", u8(bind_ip).c_str());
    std::snprintf(cfg.syncGroup, sizeof cfg.syncGroup, "%s", u8(sync_group).c_str());

    return spl::make_shared<remotewall_producer>(dependencies, cfg, device, device_auto);
}

void register_remotewall_producer(const core::module_dependencies& dependencies)
{
    dependencies.producer_registry->register_producer_factory(L"REMOTEWALL Producer", &create_producer);
}

}} // namespace caspar::remotewall
