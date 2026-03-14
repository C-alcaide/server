// decklink_capture.h
// GPU-accelerated DeckLink SDI capture front-end for the cuda_prores pipeline.
//
// Architecture
// ─────────────────────────────────────────────────────────────────────────────
//  Each DecklinkCapture instance wraps one IDeckLinkInput device.
//
//  Capture callback (VideoInputFrameArrived) runs on the DeckLink driver thread:
//    1. frame->GetBytes(&ptr)   → already in pinned host memory (CudaPinnedAllocator)
//    2. cudaMemcpyAsync(d_vram, ptr, HostToDevice, capture_stream_)
//    3. Push CaptureToken onto a lock-free SPSC queue → encoder thread consumes
//    4. ReleaseBuffer called when encoder signals token_done
//
//  Sync capture group: if group_id > 0, the SDK genlock is used so multiple
//  DeckLink 8K Pro cards capture in phase with each other.
//
#pragma once

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <comutil.h>    // pulls in COM/RPC headers that define 'interface' keyword

#if defined(_MSC_VER)
// Resolved via the ../decklink include path added by CMakeLists.txt
#include "interop/DeckLinkAPI.h"
#else
#include "linux_interop/DeckLinkAPI.h"
#endif

#include "cuda_pinned_allocator.h"
#include "../timecode.h"

#include <cuda_runtime.h>
#include <functional>
#include <memory>
#include <cstdint>

// -------------------------------------------------------------------
// CaptureToken — the unit of work pushed from capture → encoder thread
// -------------------------------------------------------------------
struct CaptureToken {
    void          *d_vram;          // device pointer (VRAM), one full V210 frame
    size_t         byte_size;       // actual frame bytes (may vary with format)
    uint32_t       width;
    uint32_t       height;
    bool           is_interlaced  = false; // detected from VideoInputFormatChanged
    bool           is_tff         = true;  // true = upper field first
    int            timebase_num   = 1000;  // frame duration  (e.g. 1000 for 25 fps)
    int            timebase_den   = 25000; // frame timescale (e.g. 25000 for 25 fps)
    int64_t        frame_counter;   // absolute frame counter (0-based)
    SmpteTimecode  tc;              // SMPTE RP188 timecode from SDI embeds
    cudaStream_t   copy_stream;     // stream used for the HostToDevice memcpy
    // For GPUDirect: the pinned host buffer backing this frame.
    // The encoder must call release_fn() when the token is fully consumed.
    std::function<void()> release_fn;
};

// -------------------------------------------------------------------
// AudioCapturePacket — PCM audio aligned to the video frame
// -------------------------------------------------------------------
struct AudioCapturePacket {
    const int32_t *samples;         // pointer into DeckLink-owned buffer
    uint32_t       sample_count;    // per-channel sample count this frame
    uint32_t       channel_count;
    uint32_t       sample_rate;
};

// Callback types for the consumer side
using FrameCallback = std::function<void(CaptureToken, AudioCapturePacket)>;

// -------------------------------------------------------------------
// DecklinkCapture
// -------------------------------------------------------------------
class DecklinkCapture final : public IDeckLinkInputCallback {
public:
    // device_index: 0-based index in IDeckLinkIterator enumeration order
    // display_mode: e.g. bmdMode4K2160p25 (see DeckLinkAPIDispatch.cpp)
    // group_id:     sync capture group (0 = no sync, >0 = genlock group)
    // on_frame:     called from the DeckLink driver thread — must be fast
    explicit DecklinkCapture(int           device_index,
                             BMDDisplayMode display_mode,
                             int            group_id,
                             FrameCallback  on_frame);
    ~DecklinkCapture();

    // Start/stop streaming
    bool start();
    void stop();

    // Device VRAM ring (device pointers; allocated in constructor)
    static constexpr int kVramRingSize   = 4;   // GPU VRAM ring slots
    static constexpr int kPinnedPoolSize  = 32;  // pinned host buffer pool (must cover DeckLink's pre-allocation ring)

    // IUnknown (minimal — lifetime managed by owner)
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID, void **) override { return E_NOINTERFACE; }
    ULONG   STDMETHODCALLTYPE AddRef()  override { return 1; }
    ULONG   STDMETHODCALLTYPE Release() override { return 1; }

    // IDeckLinkInputCallback
    HRESULT STDMETHODCALLTYPE VideoInputFormatChanged(
        BMDVideoInputFormatChangedEvents,
        IDeckLinkDisplayMode *,
        BMDDetectedVideoInputFormatFlags) override;

    HRESULT STDMETHODCALLTYPE VideoInputFrameArrived(
        IDeckLinkVideoInputFrame *,
        IDeckLinkAudioInputPacket *) override;

private:
    void init_vram_ring(size_t frame_bytes);

    int            device_index_;
    BMDDisplayMode display_mode_;
    int            group_id_;
    FrameCallback  on_frame_;

    // DeckLink COM objects
    IDeckLink             *device_    = nullptr;
    IDeckLinkInput        *input_     = nullptr;
    IDeckLinkConfiguration *config_  = nullptr;

    // CUDA resources
    cudaStream_t   copy_stream_  = nullptr;
    void          *d_vram_[kVramRingSize] = {};  // pre-allocated device frames
    size_t         vram_frame_bytes_ = 0;
    int            vram_ring_head_   = 0;

    // Pinned host allocator
    std::unique_ptr<CudaPinnedAllocator> allocator_;

    // Detected input format (set by VideoInputFormatChanged before first frame)
    bool    detected_interlaced_   = false;
    bool    detected_tff_          = true;
    int     detected_timebase_num_ = 1000;
    int     detected_timebase_den_ = 25000;

    int64_t frame_counter_ = 0;
    bool    running_       = false;
};
