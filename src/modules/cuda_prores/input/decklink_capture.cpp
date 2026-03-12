// decklink_capture.cpp
//
// Standalone DeckLink capture using CUDA pinned allocator for GPUDirect.
// Compatible with Windows DeckLink SDK 12.x.
//
// Build prerequisites on Windows:
//   - DeckLink SDK installed, interop/ headers available
//   - CUDA Toolkit 12.x
//   - COM initialized by the caller (CoInitialize / CoInitializeEx)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <comdef.h>

#include "decklink_capture.h"

#include <cstdio>
#include <stdexcept>
#include <string>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static void check_cuda(cudaError_t err, const char* context)
{
    if (err != cudaSuccess) {
        char buf[256];
        snprintf(buf, sizeof(buf), "[DecklinkCapture] CUDA error in %s: %s", context, cudaGetErrorString(err));
        throw std::runtime_error(buf);
    }
}

// Obtain the N-th IDeckLink device (1-based index to match SDK convention).
static IDeckLink* enumerate_device(int device_index)
{
    IDeckLinkIterator* iter = nullptr;
#ifdef _WIN32
    HRESULT hr = CoCreateInstance(CLSID_CDeckLinkIterator, nullptr, CLSCTX_ALL,
                                  IID_IDeckLinkIterator, reinterpret_cast<void**>(&iter));
    if (FAILED(hr) || !iter)
        throw std::runtime_error("[DecklinkCapture] DeckLink drivers not found (CoCreateInstance failed)");
#else
    iter = CreateDeckLinkIteratorInstance();
    if (!iter)
        throw std::runtime_error("[DecklinkCapture] DeckLink drivers not found");
#endif

    IDeckLink* device = nullptr;
    int        idx    = 0;
    while (iter->Next(&device) == S_OK) {
        ++idx;
        if (idx == device_index) {
            iter->Release();
            return device; // caller owns ref
        }
        device->Release();
        device = nullptr;
    }
    iter->Release();

    char buf[128];
    snprintf(buf, sizeof(buf), "[DecklinkCapture] Device %d not found", device_index);
    throw std::runtime_error(buf);
}

// ---------------------------------------------------------------------------
// Construction / destruction
// ---------------------------------------------------------------------------
DecklinkCapture::DecklinkCapture(int            device_index,
                                 BMDDisplayMode display_mode,
                                 int            group_id,
                                 FrameCallback  on_frame)
    : device_index_(device_index)
    , display_mode_(display_mode)
    , group_id_(group_id)
    , on_frame_(std::move(on_frame))
{
    // --- CUDA stream -------------------------------------------------------
    check_cuda(cudaStreamCreateWithFlags(&copy_stream_, cudaStreamNonBlocking),
               "cudaStreamCreateWithFlags");

    // --- DeckLink device ---------------------------------------------------
    device_ = enumerate_device(device_index_);

    // IDeckLinkInput
    if (FAILED(device_->QueryInterface(IID_IDeckLinkInput,
                reinterpret_cast<void**>(&input_)))) {
        device_->Release();
        throw std::runtime_error("[DecklinkCapture] Device has no IDeckLinkInput");
    }

    // IDeckLinkConfiguration (optional — only needed for sync capture group)
    device_->QueryInterface(IID_IDeckLinkConfiguration,
                            reinterpret_cast<void**>(&config_));

    // --- Pinned allocator --------------------------------------------------
    //  4K V210 frame size: row bytes = ceil(width/6)*16*4/3, worst case ~22 MB
    constexpr size_t kMaxFrameBytes = 22 * 1024 * 1024;
    allocator_ = std::make_unique<CudaPinnedAllocator>(kMaxFrameBytes, kVramRingSize + 2);

    // Register before EnableVideoInput so the first Allocate arrives in time
    if (FAILED(input_->SetVideoInputFrameMemoryAllocator(allocator_.get()))) {
        // Not fatal — fall back to driver-managed host buffers; GPUDirect DMA disabled
        fprintf(stderr, "[DecklinkCapture] SetVideoInputFrameMemoryAllocator failed — "
                        "GPUDirect disabled, using pageable host buffers\n");
    }
}

DecklinkCapture::~DecklinkCapture()
{
    stop();

    if (input_)  input_->Release();
    if (config_) config_->Release();
    if (device_) device_->Release();

    for (int i = 0; i < kVramRingSize; i++) {
        if (d_vram_[i]) cudaFree(d_vram_[i]);
    }

    if (copy_stream_) cudaStreamDestroy(copy_stream_);
}

// ---------------------------------------------------------------------------
// start() / stop()
// ---------------------------------------------------------------------------
void DecklinkCapture::init_vram_ring(size_t frame_bytes)
{
    if (vram_frame_bytes_ == frame_bytes) return; // already sized
    // Free any existing VRAM allocations
    for (int i = 0; i < kVramRingSize; i++) {
        if (d_vram_[i]) { cudaFree(d_vram_[i]); d_vram_[i] = nullptr; }
    }
    for (int i = 0; i < kVramRingSize; i++) {
        check_cuda(cudaMalloc(&d_vram_[i], frame_bytes), "cudaMalloc VRAM ring");
    }
    vram_frame_bytes_ = frame_bytes;
    vram_ring_head_   = 0;
    fprintf(stdout, "[DecklinkCapture] VRAM ring initialised: %d × %.1f MB\n",
            kVramRingSize, frame_bytes / (1024.0 * 1024.0));
}

bool DecklinkCapture::start()
{
    if (running_) return true;

    // Sync capture group (optional)
    BMDVideoInputFlags flags = bmdVideoInputFlagDefault;
    if (group_id_ > 0 && config_) {
        HRESULT hr = config_->SetInt(bmdDeckLinkConfigCapturePassThroughMode,
                                     bmdDeckLinkCapturePassThroughModeDirect);
        (void)hr; // best-effort
        // Enable synchronised capture group
        hr = config_->SetInt(bmdDeckLinkConfigSDIOutputLinkConfiguration, group_id_);
        (void)hr;
        flags = static_cast<BMDVideoInputFlags>(flags | bmdVideoInputSynchronizeToCaptureGroup);
        fprintf(stdout, "[DecklinkCapture] Sync capture group %d enabled\n", group_id_);
    }

    // Enable V210 10-bit video input
    if (FAILED(input_->EnableVideoInput(display_mode_, bmdFormat10BitYUV, flags))) {
        fprintf(stderr, "[DecklinkCapture] EnableVideoInput failed\n");
        return false;
    }

    // Enable PCM32 audio: 48 kHz, 16 channels
    if (FAILED(input_->EnableAudioInput(bmdAudioSampleRate48kHz,
                                        bmdAudioSampleType32bitInteger, 16))) {
        fprintf(stderr, "[DecklinkCapture] EnableAudioInput failed (non-fatal)\n");
    }

    if (FAILED(input_->SetCallback(this))) {
        fprintf(stderr, "[DecklinkCapture] SetCallback failed\n");
        input_->DisableVideoInput();
        return false;
    }

    if (FAILED(input_->StartStreams())) {
        fprintf(stderr, "[DecklinkCapture] StartStreams failed\n");
        input_->DisableVideoInput();
        return false;
    }

    running_ = true;
    return true;
}

void DecklinkCapture::stop()
{
    if (!running_) return;
    running_ = false;
    input_->StopStreams();
    input_->DisableVideoInput();
    // Flush any async copies before VRAM ring is freed
    cudaStreamSynchronize(copy_stream_);
}

// ---------------------------------------------------------------------------
// IDeckLinkInputCallback — VideoInputFormatChanged
// ---------------------------------------------------------------------------
HRESULT DecklinkCapture::VideoInputFormatChanged(
    BMDVideoInputFormatChangedEvents /*events*/,
    IDeckLinkDisplayMode*            newMode,
    BMDDetectedVideoInputFormatFlags /*flags*/)
{
    if (!newMode) return S_OK;
    display_mode_ = newMode->GetDisplayMode();
    fprintf(stdout, "[DecklinkCapture] Input format changed → %d x %d\n",
            static_cast<int>(newMode->GetWidth()),
            static_cast<int>(newMode->GetHeight()));
    return S_OK;
}

// ---------------------------------------------------------------------------
// IDeckLinkInputCallback — VideoInputFrameArrived
// ---------------------------------------------------------------------------
HRESULT DecklinkCapture::VideoInputFrameArrived(
    IDeckLinkVideoInputFrame*  video,
    IDeckLinkAudioInputPacket* audio)
{
    if (!running_) return S_OK;

    // ---- Video ----
    CaptureToken token = {};
    if (video) {
        const uint32_t w    = static_cast<uint32_t>(video->GetWidth());
        const uint32_t h    = static_cast<uint32_t>(video->GetHeight());
        const size_t   size = static_cast<size_t>(video->GetRowBytes()) * h;

        // Lazy VRAM ring initialisation (size known only after first frame hits)
        if (size > 0) {
            init_vram_ring(size);
        }

        void* host_ptr = nullptr;
        if (SUCCEEDED(video->GetBytes(&host_ptr)) && host_ptr && size > 0) {
            // Select next VRAM ring slot
            const int slot = vram_ring_head_ % kVramRingSize;
            ++vram_ring_head_;

            // Async DMA: pinned host → device VRAM
            // The consumer MUST call cudaStreamSynchronize(token.copy_stream) before reading d_vram.
            cudaError_t ce = cudaMemcpyAsync(d_vram_[slot], host_ptr, size,
                                             cudaMemcpyHostToDevice, copy_stream_);
            if (ce != cudaSuccess) {
                fprintf(stderr, "[DecklinkCapture] cudaMemcpyAsync failed: %s\n",
                        cudaGetErrorString(ce));
                return S_OK; // drop frame
            }

            // Keep the DeckLink frame alive until the async copy is consumed.
            // The DeckLink allocator's pinned buffer is live as long as the frame
            // is AddRef'd.  The consumer calls release_fn() after GPU work.
            video->AddRef();

            token.d_vram        = d_vram_[slot];
            token.byte_size     = size;
            token.width         = w;
            token.height        = h;
            token.frame_counter = frame_counter_++;
            token.copy_stream   = copy_stream_;
            token.release_fn    = [video]() { video->Release(); };

            // Extract SMPTE RP188 timecode from embedded SDI ancillary data.
            // Try RP188-Any first (covers both VITC and LTC); fall back to VITC.
            {
                IDeckLinkTimecode *tc_iface = nullptr;
                HRESULT hr = video->GetTimecode(bmdTimecodeRP188Any, &tc_iface);
                if ((FAILED(hr) || !tc_iface) && tc_iface) {
                    tc_iface->Release();
                    tc_iface = nullptr;
                }
                if (SUCCEEDED(hr) && tc_iface) {
                    uint8_t h_tc = 0, m_tc = 0, s_tc = 0, f_tc = 0;
                    if (SUCCEEDED(tc_iface->GetComponents(&h_tc, &m_tc, &s_tc, &f_tc))) {
                        BMDTimecodeFlags flags = tc_iface->GetFlags();
                        token.tc.hours      = h_tc;
                        token.tc.minutes    = m_tc;
                        token.tc.seconds    = s_tc;
                        token.tc.frames     = f_tc;
                        token.tc.drop_frame = (flags & bmdTimecodeIsDropFrame) != 0;
                        token.tc.valid      = true;
                    }
                    tc_iface->Release();
                }
            }
        }
    }

    // ---- Audio ----
    AudioCapturePacket apkt = {};
    if (audio) {
        void* audio_bytes = nullptr;
        if (SUCCEEDED(audio->GetBytes(&audio_bytes)) && audio_bytes) {
            apkt.samples      = static_cast<const int32_t*>(audio_bytes);
            apkt.sample_count = static_cast<uint32_t>(audio->GetSampleFrameCount());
            apkt.channel_count = 16;
            apkt.sample_rate   = 48000;
        }
    }

    // Invoke consumer callback (runs on DeckLink driver thread — must be quick)
    if (token.d_vram) {
        try {
            on_frame_(token, apkt);
        } catch (const std::exception& e) {
            fprintf(stderr, "[DecklinkCapture] on_frame_ threw: %s\n", e.what());
        }
    }

    return S_OK;
}
