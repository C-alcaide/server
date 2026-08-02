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

#include "html_gpu_bridge.h"

#include <common/array.h>
#include <common/executor.h>
#include <common/log.h>

#include <core/frame/draw_frame.h>
#include <core/frame/frame.h>
#include <core/frame/frame_factory.h>
#include <core/frame/pixel_format.h>

#include <atomic>
#include <chrono>
#include <sstream>
#include <vector>

#ifdef _WIN32

#include <accelerator/vulkan/util/d3d11_import_bridge.h>

#include <d3d11_1.h>
#include <d3d11_4.h>
#include <dxgi1_2.h>
#include <wrl/client.h>

namespace caspar { namespace html {

namespace {

using Microsoft::WRL::ComPtr;

/// Staging slots. Four matches the producer's own frame queue depth and the slot
/// count the ISF producer settled on; deeper mostly buys latency.
constexpr int kSlots = 4;

/// How long submit() will wait for its own blit before treating the device as
/// wedged. Generous compared with a ~1 ms 8 MB device-local copy, but bounded:
/// this runs on the process-global CEF UI thread and must never park there.
constexpr DWORD kFenceWaitMs = 100;

/// Consecutive failures before the bridge stops trying. ~0.6 s at 50 fps. Past
/// that, OpenSharedResource1 is a kernel-handle syscall being burned for nothing.
constexpr int kMaxConsecutiveFailures = 30;

std::wstring hr_hex(HRESULT hr)
{
    std::wstringstream ss;
    ss << L"0x" << std::hex << static_cast<unsigned long>(hr);
    return ss.str();
}

} // namespace

struct html_gpu_bridge::impl
{
    /// One staging texture, plus the shared handle the Vulkan side imports it by.
    ///
    /// The handle is created once and stays valid for the bridge's life, which is
    /// what lets d3d11_import_bridge cache the import. Importing CEF's own handle
    /// directly is not an option: CEF documents it as pool-owned and valid only
    /// inside the callback, so the same value may later name a different resource.
    struct slot
    {
        ComPtr<ID3D11Texture2D> tex;
        HANDLE                  shared = nullptr;
        std::atomic<bool>       busy{false};
        int                     width  = 0;
        int                     height = 0;
    };

    core::frame_factory* factory_    = nullptr;
    const void*          stream_tag_ = nullptr;

    ComPtr<ID3D11Device>        device_;
    ComPtr<ID3D11Device1>       device1_;
    ComPtr<ID3D11DeviceContext> context_;
    ComPtr<ID3D11Fence>         fence_;
    ComPtr<ID3D11DeviceContext4> context4_;
    HANDLE                      fence_event_ = nullptr;
    UINT64                      fence_value_ = 0;
    /// Fence path unavailable (pre-11.4 runtime): fall back to polling a query.
    ComPtr<ID3D11Query> flush_query_;

    std::unique_ptr<accelerator::vulkan::d3d11_import_bridge> import_;

    slot                                  slots_[kSlots];
    DXGI_FORMAT                           slot_format_ = DXGI_FORMAT_UNKNOWN;
    std::function<void(core::draw_frame)> on_frame_;
    std::unique_ptr<executor>             worker_;

    std::atomic<bool>         healthy_{true};
    std::atomic<bool>         probing_{true};
    std::atomic<int>          consecutive_failures_{0};
    std::atomic<int>          in_flight_{0};
    std::atomic<std::int64_t> dropped_{0};
    std::atomic<std::int64_t> stage_wait_us_{0};

    ~impl()
    {
        // Ordering matters and is easy to get wrong: stop accepting work, drain
        // the worker (phase two may be mid-copy), release the Vulkan imports
        // (which waits on the GPU), and only then let the D3D11 textures the
        // imports alias go.
        healthy_ = false;
        if (worker_)
            worker_.reset();
        if (import_)
            import_->release_imports();
        import_.reset();

        for (auto& s : slots_) {
            if (s.shared)
                CloseHandle(s.shared);
            s.shared = nullptr;
            s.tex.Reset();
        }
        if (fence_event_)
            CloseHandle(fence_event_);
    }

    bool create_device(int adapter_index, std::wstring& reason)
    {
        ComPtr<IDXGIFactory1> factory;
        HRESULT               hr = CreateDXGIFactory1(IID_PPV_ARGS(&factory));
        if (FAILED(hr)) {
            reason = L"CreateDXGIFactory1 failed " + hr_hex(hr);
            return false;
        }

        ComPtr<IDXGIAdapter1> adapter;
        hr = factory->EnumAdapters1(static_cast<UINT>(adapter_index), &adapter);
        if (FAILED(hr)) {
            reason = L"DXGI adapter " + std::to_wstring(adapter_index) + L" not found";
            return false;
        }

        DXGI_ADAPTER_DESC1 desc{};
        adapter->GetDesc1(&desc);

        // D3D_DRIVER_TYPE_UNKNOWN is required when an adapter is supplied. Passing
        // HARDWARE with a non-null adapter fails; passing nullptr for the adapter
        // silently picks adapter 0, which is the bug this whole lookup exists to
        // avoid on a multi-GPU machine.
        hr = D3D11CreateDevice(adapter.Get(),
                               D3D_DRIVER_TYPE_UNKNOWN,
                               nullptr,
                               0,
                               nullptr,
                               0,
                               D3D11_SDK_VERSION,
                               &device_,
                               nullptr,
                               &context_);
        if (FAILED(hr)) {
            reason = L"D3D11CreateDevice failed " + hr_hex(hr);
            return false;
        }
        if (FAILED(device_.As(&device1_))) {
            reason = L"ID3D11Device1 unavailable (OpenSharedResource1 needs it)";
            return false;
        }

        // Preferred completion signal. Falls back to a query below; both are
        // exact, the fence just costs less than a spin.
        ComPtr<ID3D11Device5> device5;
        if (SUCCEEDED(device_.As(&device5)) && SUCCEEDED(context_.As(&context4_))) {
            if (FAILED(device5->CreateFence(0, D3D11_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence_)))) {
                fence_.Reset();
                context4_.Reset();
            } else {
                fence_event_ = CreateEventW(nullptr, FALSE, FALSE, nullptr);
                if (!fence_event_) {
                    fence_.Reset();
                    context4_.Reset();
                }
            }
        }
        if (!fence_) {
            D3D11_QUERY_DESC qd{};
            qd.Query = D3D11_QUERY_EVENT;
            if (FAILED(device_->CreateQuery(&qd, &flush_query_))) {
                reason = L"neither ID3D11Fence nor a D3D11_QUERY_EVENT is available";
                return false;
            }
        }

        CASPAR_LOG(info) << L"[html/gpu] D3D11 device on adapter " << adapter_index << L" '" << desc.Description
                         << L"', completion via " << (fence_ ? L"ID3D11Fence" : L"ID3D11Query");
        return true;
    }

    bool build_slots(int width, int height, DXGI_FORMAT format, std::wstring& reason)
    {
        D3D11_TEXTURE2D_DESC td{};
        td.Width          = static_cast<UINT>(width);
        td.Height         = static_cast<UINT>(height);
        td.MipLevels      = 1;
        td.ArraySize      = 1;
        td.Format         = format;
        td.SampleDesc     = {1, 0};
        td.Usage          = D3D11_USAGE_DEFAULT;
        td.BindFlags      = D3D11_BIND_SHADER_RESOURCE;
        // NT handle so Vulkan can import it; no keyed mutex, because the only two
        // users are this device and the import, and the fence below already orders
        // them.
        td.MiscFlags      = D3D11_RESOURCE_MISC_SHARED_NTHANDLE | D3D11_RESOURCE_MISC_SHARED;

        for (auto& s : slots_) {
            s.tex.Reset();
            if (s.shared) {
                CloseHandle(s.shared);
                s.shared = nullptr;
            }
            HRESULT hr = device_->CreateTexture2D(&td, nullptr, &s.tex);
            if (FAILED(hr)) {
                reason = L"CreateTexture2D failed " + hr_hex(hr);
                return false;
            }
            ComPtr<IDXGIResource1> res;
            if (FAILED(s.tex.As(&res))) {
                reason = L"IDXGIResource1 unavailable on the staging texture";
                return false;
            }
            hr = res->CreateSharedHandle(nullptr, DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE, nullptr,
                                         &s.shared);
            if (FAILED(hr)) {
                reason = L"CreateSharedHandle failed " + hr_hex(hr);
                return false;
            }
            s.width  = width;
            s.height = height;
            s.busy   = false;
        }
        slot_format_ = format;
        return true;
    }

    /// Blocks until everything submitted so far has retired on the GPU.
    ///
    /// Not optional. CEF releases the source back to its pool the moment the
    /// callback returns, and the texture has no keyed mutex, so a queued-but-
    /// unexecuted copy would read a recycled surface. That produces a torn frame,
    /// not a crash -- it will not reproduce reliably, so it has to be prevented
    /// rather than detected.
    bool wait_for_gpu()
    {
        if (fence_ && context4_) {
            const UINT64 v = ++fence_value_;
            if (FAILED(context4_->Signal(fence_.Get(), v)))
                return false;
            if (fence_->GetCompletedValue() >= v)
                return true;
            if (FAILED(fence_->SetEventOnCompletion(v, fence_event_)))
                return false;
            return WaitForSingleObject(fence_event_, kFenceWaitMs) == WAIT_OBJECT_0;
        }

        context_->End(flush_query_.Get());
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(kFenceWaitMs);
        BOOL       done     = FALSE;
        while (std::chrono::steady_clock::now() < deadline) {
            const HRESULT hr = context_->GetData(flush_query_.Get(), &done, sizeof(done), 0);
            if (hr == S_OK && done)
                return true;
            if (FAILED(hr))
                return false;
            SwitchToThread();
        }
        return false;
    }

    int acquire_slot()
    {
        for (int i = 0; i < kSlots; ++i) {
            bool expected = false;
            if (slots_[i].busy.compare_exchange_strong(expected, true))
                return i;
        }
        return -1;
    }

    /// Phase two. Runs on the worker: imports the staging slot into the mixer's
    /// device and builds the frame. Frees the slot when the copy has retired.
    void finish(int index, core::pixel_format fmt, int width, int height)
    {
        auto& s = slots_[index];

        std::shared_ptr<core::texture> tex;
        const bool ok = import_ && import_->copy_texture(s.shared, width, height, tex);
        // The Vulkan copy reads the staging texture, so the slot cannot be reused
        // until it has retired -- which copy_texture guarantees on its next call,
        // but the next call may be for a different slot. Wait here instead.
        if (import_)
            import_->wait_for_previous_copy();

        s.busy = false;
        in_flight_.fetch_sub(1);

        if (!ok || !tex) {
            note_failure(L"Vulkan import/copy failed");
            return;
        }

        consecutive_failures_ = 0;
        probing_              = false;

        core::pixel_format_desc desc(fmt);
        desc.planes.push_back(core::pixel_format_desc::plane(width, height, 4, common::bit_depth::bit8));

        // One empty host plane: the frame is GPU-only, and host_image_state()
        // reports `unavailable` for it, which is how consumers and wrapping
        // producers are supposed to find out.
        auto empty = std::make_shared<std::vector<std::uint8_t>>(0);
        std::vector<array<const std::uint8_t>> planes;
        planes.emplace_back(empty->data(), 0, std::move(empty));

        core::const_frame frame(stream_tag_, std::move(planes), array<const std::int32_t>{}, desc, std::move(tex));
        if (on_frame_)
            on_frame_(core::draw_frame(std::move(frame)));
    }

    void note_failure(const std::wstring& what)
    {
        const int n = consecutive_failures_.fetch_add(1) + 1;
        if (n == 1)
            CASPAR_LOG(warning) << L"[html/gpu] " << what << L".";
        if (n >= kMaxConsecutiveFailures && healthy_.exchange(false)) {
            CASPAR_LOG(error) << L"[html/gpu] giving up after " << n
                              << L" consecutive failures; the channel will hold the last good frame. "
                                 L"Reload the producer to retry.";
        }
    }

    /// Writes a known pattern into slot 0 and runs it all the way through the
    /// Vulkan import, so that everything which can fail without a real frame does
    /// so in create() rather than on air.
    bool self_test(std::wstring& reason)
    {
        std::vector<std::uint32_t> pattern(static_cast<size_t>(slots_[0].width) * slots_[0].height, 0xFF20A0C0u);
        context_->UpdateSubresource(slots_[0].tex.Get(), 0, nullptr, pattern.data(),
                                    static_cast<UINT>(slots_[0].width * 4), 0);
        if (!wait_for_gpu()) {
            reason = L"self-test: the GPU did not signal completion";
            return false;
        }

        std::shared_ptr<core::texture> tex;
        if (!import_->copy_texture(slots_[0].shared, slots_[0].width, slots_[0].height, tex) || !tex) {
            reason = L"self-test: the Vulkan import could not copy the staging texture";
            return false;
        }
        import_->wait_for_previous_copy();
        return true;
    }
};

html_gpu_bridge::html_gpu_bridge()
    : impl_(new impl())
{
}

html_gpu_bridge::~html_gpu_bridge() = default;

std::unique_ptr<html_gpu_bridge> html_gpu_bridge::create(core::frame_factory&                  factory,
                                                         const void*                           stream_tag,
                                                         std::function<void(core::draw_frame)> on_frame,
                                                         std::wstring&                         reason)
{
    if (factory.gpu_device_backend() != core::gpu_backend::vulkan) {
        // The OpenGL mixer needs the WGL_NV_DX_interop2 route instead. It is a
        // different mechanism, not a parameter, and is not built yet -- say so
        // rather than failing obscurely.
        reason = L"only the Vulkan mixer is supported (the OpenGL WGL_NV_DX_interop path is not built)";
        return nullptr;
    }

    auto* vk_device = factory.gpu_device_handle();
    if (!vk_device) {
        reason = L"the mixer reported no GPU device";
        return nullptr;
    }

    const int adapter = accelerator::vulkan::dxgi_adapter_for_vk_device(vk_device);
    if (adapter < 0) {
        reason = L"could not match the mixer's Vulkan device to a DXGI adapter";
        return nullptr;
    }

    std::unique_ptr<html_gpu_bridge> self(new html_gpu_bridge());
    auto&                            m = *self->impl_;
    m.factory_    = &factory;
    m.stream_tag_ = stream_tag;
    m.on_frame_   = std::move(on_frame);

    if (!m.create_device(adapter, reason))
        return nullptr;

    try {
        m.import_ = std::make_unique<accelerator::vulkan::d3d11_import_bridge>(vk_device);
    } catch (const std::exception& e) {
        reason = L"d3d11_import_bridge: " + u16(e.what());
        return nullptr;
    }

    // Sized from the channel; a differing visible rect is handled by rebuilding.
    if (!m.build_slots(64, 64, DXGI_FORMAT_B8G8R8A8_UNORM, reason))
        return nullptr;
    if (!m.self_test(reason))
        return nullptr;

    m.worker_ = std::make_unique<executor>(L"html-gpu");
    m.worker_->set_capacity(kSlots);

    CASPAR_LOG(info) << L"[html/gpu] bridge ready (adapter " << adapter << L", " << kSlots << L" slots).";
    return self;
}

bool html_gpu_bridge::submit(void* shared_handle, shared_surface_order order, int x, int y, int w, int h)
{
    auto& m = *impl_;
    if (!m.healthy_ || !shared_handle || w <= 0 || h <= 0)
        return false;

    const DXGI_FORMAT want =
        order == shared_surface_order::bgra8 ? DXGI_FORMAT_B8G8R8A8_UNORM : DXGI_FORMAT_R8G8B8A8_UNORM;

    // CopySubresourceRegion requires matching formats, and the slots are sized to
    // the visible rect. Either changing means rebuilding the ring -- which cannot
    // happen while a copy is in flight, so drain first.
    if (m.slots_[0].width != w || m.slots_[0].height != h || m.slot_format_ != want) {
        if (m.worker_)
            m.worker_->invoke([] {});
        std::wstring reason;
        if (!m.build_slots(w, h, want, reason)) {
            m.note_failure(L"could not rebuild the staging ring: " + reason);
            return false;
        }
        if (m.import_)
            m.import_->release_imports();
        CASPAR_LOG(info) << L"[html/gpu] staging ring rebuilt for " << w << L"x" << h << L".";
    }

    const int index = m.acquire_slot();
    if (index < 0) {
        // Every slot busy: the mixer is behind. Same back-pressure the host path
        // already applies by dropping the oldest queued frame.
        m.dropped_.fetch_add(1);
        return false;
    }

    const auto start = std::chrono::steady_clock::now();

    ComPtr<ID3D11Texture2D> opened;
    HRESULT                 hr = m.device1_->OpenSharedResource1(shared_handle, IID_PPV_ARGS(&opened));
    if (FAILED(hr) || !opened) {
        m.slots_[index].busy = false;
        if (m.probing_) {
            // The one thing create() cannot test. On a multi-GPU machine the
            // compositor may have chosen an adapter this device cannot import
            // from, and it only shows up now.
            CASPAR_LOG(warning) << L"[html/gpu] the compositor's shared texture could not be opened on the mixer's "
                                   L"adapter (" << hr_hex(hr)
                                << L"). It is compositing on a different GPU; set "
                                   L"configuration.html.gpu-direct-adapter-luid, or turn gpu-direct off.";
        }
        m.note_failure(L"OpenSharedResource1 failed " + hr_hex(hr));
        return false;
    }

    // Crop to the visible rect. Copying coded_size instead is the classic way to
    // get a frame that is subtly shifted or rescaled rather than obviously broken.
    D3D11_BOX box{static_cast<UINT>(x),     static_cast<UINT>(y),     0,
                  static_cast<UINT>(x + w), static_cast<UINT>(y + h), 1};
    m.context_->CopySubresourceRegion(m.slots_[index].tex.Get(), 0, 0, 0, 0, opened.Get(), 0, &box);
    m.context_->Flush();

    const bool completed = m.wait_for_gpu();

    m.stage_wait_us_.fetch_add(
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count());

    if (!completed) {
        m.slots_[index].busy = false;
        m.note_failure(L"the staging copy did not complete within the deadline");
        return false;
    }

    const auto fmt = order == shared_surface_order::bgra8 ? core::pixel_format::bgra : core::pixel_format::rgba;
    m.in_flight_.fetch_add(1);
    m.worker_->begin_invoke([&m, index, fmt, w, h] { m.finish(index, fmt, w, h); });
    return true;
}

bool         html_gpu_bridge::healthy() const { return impl_->healthy_; }
bool         html_gpu_bridge::probing() const { return impl_->probing_; }
int          html_gpu_bridge::in_flight() const { return impl_->in_flight_; }
std::int64_t html_gpu_bridge::dropped() const { return impl_->dropped_; }
std::int64_t html_gpu_bridge::take_stage_wait_us() { return impl_->stage_wait_us_.exchange(0); }

}} // namespace caspar::html

#endif // _WIN32
