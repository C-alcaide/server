# nvJPEG + NVIDIA Optical Flow — Implementation Plan

## 1. Executive Summary

Add **optional GPU-accelerated JPEG encoding/decoding** (nvJPEG) and
**hardware optical flow frame interpolation** (NVOF SDK 5.0.7 + FRUC) to the
CasparCG replay module as alternatives to the current VMX CPU codec and simple
frame-repeat slow-motion.

nvJPEG offloads codec work to the GPU, freeing CPU cores for other CasparCG
tasks.  NVIDIA Optical Flow uses a **dedicated hardware unit** (OFA — Optical
Flow Accelerator, Turing and above) to produce flow vectors at near-zero CUDA
cost, and the bundled **FRUC library** (Frame Rate Up-Conversion) uses those
vectors to warp and blend intermediate frames for smooth slow-motion.

Both features are **optional** — VMX remains the default codec, and frame-repeat
remains the default slow-motion.  The GPU path activates only when an NVIDIA
GPU is detected and the user opts in.

---

## 2. VMX vs nvJPEG — Codec Comparison

### 2.1 Architecture

| Aspect | VMX (libvmx) | nvJPEG (CUDA) |
|---|---|---|
| Transform | 8×8 DCT | 8×8 DCT |
| Entropy coding | Golomb / Exp-Golomb LUTs | Huffman |
| Chroma | 4:2:2:4 (native alpha) | 4:2:0, 4:2:2, 4:4:4 (no alpha) |
| Bit depth | **8-bit and 10-bit** (P216, PA16) | **8-bit only** |
| Parallelism | Slice-based (16 px strips, CPU threads) | GPU-wide (CUDA cores) |
| SIMD | Fused DCT+Quant+Zigzag SSE4/AVX2 | N/A (CUDA kernels) |
| Execution | CPU only | GPU (CUDA), CPU for Huffman (hybrid) |
| Interlace | First-class (field-aware slicing) | Not native |
| Preview decode | DC-only 1/8 res fast path | Scale decode (HW: 1/2, 1/4, 1/8) |
| HW decode accel | N/A | Ampere+ (A100, A30, Hopper, Ada) |
| HW encode accel | N/A | Jetson Thor only |
| License | MIT (github.com/openmediatransport) | CUDA Toolkit (NVIDIA EULA) |

### 2.2 Encode Performance (1080p, quality ≈ 90)

| Codec | Device | Streams @ 50 fps | Encode time/frame | Bitrate |
|---|---|---|---|---|
| **VMX SQ** | i7 AVX2 | **~6–8×** 1080p50 | ~2–3 ms | ~130 Mbps |
| **VMX HQ** | i7 AVX2 | ~4–6× 1080p50 | ~3–4 ms | ~260 Mbps |
| nvJPEG (CUDA) | RTX 3070 | ~4–6× 1080p50 (est.) | ~3–5 ms | variable |
| nvJPEG (CUDA) | RTX 4090 | ~10–16× 1080p50 (est.) | ~1–3 ms | variable |

### 2.3 Decode Performance (1080p)

| Codec | Device | Decode time/frame | Notes |
|---|---|---|---|
| **VMX** | i7 AVX2 | ~2 ms | Fused SIMD inverse path |
| nvJPEG (hybrid) | RTX 3070 | ~2–4 ms | CPU Huffman + GPU IDCT |
| nvJPEG (HW) | A100/Ada | ~0.5–1 ms | Dedicated JPEG HW decoder |

### 2.4 Multi-Stream & 4K Scaling

| Scenario | VMX (CPU) | nvJPEG (GPU) |
|---|---|---|
| 4× 1080p50 encode | i7-13700K, ~50% CPU | RTX 3070, ~40% CUDA |
| 8× 1080p50 encode | i9-13900K, ~80% CPU | RTX 4090, ~30% CUDA |
| 1× 4K50 encode | i7-13700K, ~40% CPU (4 threads) | RTX 3070, ~30% CUDA |
| 4× 4K50 encode | Threadripper 32C, ~80% CPU | 2× RTX 4090 or A6000 |

### 2.5 Verdict

**VMX wins for most broadcast replay scenarios** — faster per-frame on CPU,
native 10-bit and alpha, no GPU dependency, MIT licensed.

**nvJPEG wins when:**
- You have a powerful GPU and a weak CPU
- You want to offload codec to GPU to free CPU for other CasparCG channels
- You need MJPEG compatibility with third-party tools
- You're on Ampere+ and can use HW decode acceleration

**Key nvJPEG limitations:**
- **No 10-bit** — JPEG standard is 8-bit
- **No alpha channel** — 3-channel max for encode
- **GPU memory overhead** — each nvJPEG state needs ~100–400 MB VRAM
- **PCIe bottleneck** — BGRA frames must be uploaded to GPU (1080p = ~8 MB/frame)
- **Competes with Optical Flow for GPU resources**

---

## 3. NVIDIA Optical Flow (NVOF) for Slow-Motion

### 3.1 Overview

The NVOF SDK 5.0.7 provides access to a **dedicated hardware optical flow
engine** (OFA unit) on Turing and above GPUs.  Unlike RIFE (which uses CUDA/
Tensor cores for neural-network inference), NVOF uses fixed-function hardware
that runs at near-zero CUDA utilization.

The SDK includes two components:
1. **NvOF API** — Low-level optical flow vector estimation
2. **NvOFFRUC** — Frame Rate Up-Conversion library (pre-built DLL) that
   internally uses NvOF to interpolate frames

### 3.2 NVOF API Architecture

```
NvOFHandle (nvOpticalFlowCommon.h)
    │
    ├── NvOFCuda    — CUDA backend (NvOFCuda.h)
    ├── NvOFD3D11   — Direct3D 11 backend
    ├── NvOFD3D12   — Direct3D 12 backend
    └── NvOFVulkan  — Vulkan backend

Input:  Two frames (NV12, ABGR8, or Grayscale8)
Output: Flow vectors (NV_OF_FLOW_VECTOR: int16 flowx, int16 flowy in S10.5 format)
        per grid cell (1×1, 2×2, or 4×4 pixels)
```

**Key capabilities:**
- Output grid sizes: 1×1, 2×2, 4×4
- Performance levels: SLOW (best quality), MEDIUM, FAST
- Bidirectional flow: `NV_OF_PRED_DIRECTION_BOTH`
- ROI support for selective region processing
- Global flow estimation
- Cost/confidence output per vector
- Max resolution: 8192×8192 (per `NV_OF_CAPS_WIDTH_MAX` / `HEIGHT_MAX`)

### 3.3 NvOFFRUC — Frame Rate Up-Conversion

The FRUC library is a **pre-built DLL** (`NvOFFRUC.dll`) that provides a
turnkey frame interpolation pipeline:

```
NvOFFRUCCreate()           → Initialize with width, height, surface format
NvOFFRUCRegisterResource() → Register GPU buffers (D3D11 textures or CUDA ptrs)
NvOFFRUCProcess()          → Feed frame + timestamp → get interpolated frame
NvOFFRUCDestroy()          → Cleanup
```

**Supported surface formats:** NV12, ARGB (via `NvOFFRUCSurfaceFormat`)
**Resource types:** CUDA (CuDevicePtr or CuArray), DirectX 11

**Key properties:**
- Min 3 registered resources (input ping-pong + output)
- Max 10 registered resources
- Timestamp-based interpolation (automatic midpoint)
- Frame repetition detection (outputs flag when quality is too low)
- D3D11 Fence or KeyedMutex synchronization for DX interop

### 3.4 Performance

| Resolution | OFA inference | FRUC total (flow + warp) | GPU load |
|---|---|---|---|
| 1080p | ~1–2 ms | ~3–5 ms | ~0% CUDA (OFA HW) + ~5% warp |
| 4K | ~3–5 ms | ~8–12 ms | ~0% CUDA (OFA HW) + ~10% warp |

Since the OFA unit is dedicated hardware, it runs **in parallel** with CUDA
compute.  The only CUDA cost is the warping kernel inside FRUC.

### 3.5 NVOF vs RIFE for Slow-Motion

| Criterion | NVOF + FRUC | RIFE (ncnn/TensorRT) |
|---|---|---|
| Quality | Good (hardware-fixed logic) | **Excellent** (learned motion model) |
| Occlusion handling | Limited (simple blending) | **Strong** (learned occlusion masks) |
| License | **NVIDIA EULA** (proprietary DLL) | **MIT** (open source) |
| Hardware | Turing+ NVIDIA only | Any GPU (Vulkan) or NVIDIA (TRT) |
| GPU load | **~0% CUDA** (dedicated HW) | ~30–60% CUDA/Tensor cores |
| Speed @ 1080p | **~3–5 ms** (FRUC total) | ~5–8 ms (TensorRT FP16) |
| Speed @ 4K | **~8–12 ms** | ~15–25 ms |
| Frame repetition detect | Built-in | Manual (PSNR/SSIM) |
| Integration complexity | DLL load + register/process | Model loading + inference pipeline |

**Recommendation:** Support **both** — NVOF as the fast/lightweight default on
NVIDIA GPUs, RIFE as the high-quality option.  Let the user choose via AMCP
parameter.

---

## 4. Integration Architecture

### 4.1 Codec Abstraction Layer

Currently the replay module directly calls VMX functions.  To support nvJPEG as
an alternative, introduce a codec interface:

```cpp
// replay_codec.h
class replay_codec {
public:
    virtual ~replay_codec() = default;

    // Encode BGRA frame to compressed buffer. Returns compressed size, 0 on error.
    virtual int encode(const uint8_t* bgra, int stride, int interlaced,
                       uint8_t* out_buf, int out_buf_size) = 0;

    // Load compressed data for decode
    virtual bool load(const uint8_t* data, int size) = 0;

    // Decode to BGRA buffer
    virtual bool decode_bgra(uint8_t* bgra, int stride) = 0;

    // Codec name for diagnostics
    virtual const char* name() const = 0;

    // Codec FourCC for file header (e.g. "VMX ", "JPEG")
    virtual const char* fourcc() const = 0;
};

// Factory
std::unique_ptr<replay_codec> create_replay_codec(
    const std::string& type,   // "vmx" | "nvjpeg"
    int width, int height,
    const std::string& quality, // "LQ" | "SQ" | "HQ"
    int gpu_id = -1);           // -1 = CPU only
```

### 4.2 nvJPEG Codec Implementation

```cpp
// replay_nvjpeg_codec.h / .cpp
class nvjpeg_codec : public replay_codec {
    nvjpegHandle_t       handle_;
    nvjpegJpegState_t    state_;
    nvjpegEncoderState_t enc_state_;
    nvjpegEncoderParams_t enc_params_;
    cudaStream_t         stream_;
    int                  width_, height_;
    int                  quality_;        // 1-100

    // GPU buffers (persistent, avoid per-frame alloc)
    CUdeviceptr          d_input_;        // BGRA frame on GPU
    CUdeviceptr          d_output_[3];    // Planar BGR decode output
    size_t               d_input_pitch_;

public:
    int encode(const uint8_t* bgra, int stride, int interlaced,
               uint8_t* out_buf, int out_buf_size) override;
    bool load(const uint8_t* data, int size) override;
    bool decode_bgra(uint8_t* bgra, int stride) override;
    const char* name() const override { return "nvjpeg"; }
    const char* fourcc() const override { return "JPEG"; }
};
```

**Encode pipeline:**
```
CPU BGRA → cudaMemcpyAsync(H2D) → nvjpegEncodeImage(BGRI) →
nvjpegEncodeRetrieveBitstream() → CPU compressed buffer → write to .mav
```

**Decode pipeline:**
```
CPU compressed JPEG → nvjpegDecode(BGRI) → GPU interleaved BGR →
cudaMemcpyAsync(D2H) → CPU BGRA buffer
```

**Note:** nvJPEG input format `NVJPEG_INPUT_BGRI` maps to interleaved BGR
(3 bytes/pixel).  Our pipeline uses BGRA (4 bytes/pixel), so we need a
lightweight CUDA kernel for BGR↔BGRA conversion on GPU, or strip alpha on
CPU before upload.

### 4.3 Optical Flow Interpolation

#### Option A: NvOFFRUC (recommended — turnkey)

```cpp
// replay_nvof_interpolator.h / .cpp
class nvof_interpolator {
    NvOFFRUCHandle       fruc_;
    CUdeviceptr          d_input_[2];     // Ping-pong input buffers
    CUdeviceptr          d_output_;       // Interpolated output
    int                  width_, height_;
    bool                 initialized_ = false;

public:
    bool init(int width, int height, int gpu_id = 0);
    void destroy();

    // Feed a new frame (called every real decoded frame)
    // Returns true if an interpolated frame was produced
    bool process(const uint8_t* bgra_frame, double timestamp_ms,
                 uint8_t* out_interpolated, bool* was_repeated);
};
```

**Integration into replay_producer:**
```
receive_impl()
    │
    ├─ speed == 1.0 → normal decode (unchanged)
    │
    └─ speed < 1.0 (slow-mo) → interpolation path
        │
        ├─ Decode real frame via VMX/nvJPEG
        │
        ├─ Upload to GPU (cudaMemcpy H2D)
        │
        ├─ NvOFFRUCProcess(frame, timestamp)
        │   ├─ OFA HW: compute flow vectors (~1-2 ms, 0% CUDA)
        │   ├─ Warp kernel: blend using flow (~1-2 ms)
        │   └─ Output: interpolated ARGB on GPU
        │
        ├─ Download result (cudaMemcpy D2H)
        │
        └─ Build CasparCG draw_frame
```

#### Option B: Raw NvOF API (advanced — custom warping)

For more control over the interpolation quality, use the low-level NvOF API
to get flow vectors and implement custom warping:

```cpp
// Using NvOFCuda
auto nvof = NvOFCuda::Create(cuContext, width, height,
    NV_OF_BUFFER_FORMAT_ABGR8,
    NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR,
    NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR,
    NV_OF_MODE_OPTICALFLOW,
    NV_OF_PERF_LEVEL_MEDIUM);

nvof->Init(NV_OF_OUTPUT_VECTOR_GRID_SIZE_4);

// Create buffers
auto inputBufs  = nvof->CreateBuffers(NV_OF_BUFFER_USAGE_INPUT, 2);
auto outputBufs = nvof->CreateBuffers(NV_OF_BUFFER_USAGE_OUTPUT, 1);

// Per frame pair:
inputBufs[0]->UploadData(frame_A_data);
inputBufs[1]->UploadData(frame_B_data);
nvof->Execute(inputBufs[0].get(), inputBufs[1].get(), outputBufs[0].get());
outputBufs[0]->DownloadData(flow_vectors);  // NV_OF_FLOW_VECTOR array

// Custom warp: use flow_vectors to sample frame_A and frame_B
// at fractional position t, with custom occlusion handling
```

This approach gives better quality control but requires writing the warp kernel.
**Start with Option A (FRUC), upgrade to Option B later if needed.**

---

## 5. File Format Considerations

### 5.1 .mav File Header Extension

The current `.mav` file header identifies the codec via `video_fourcc`:

```cpp
struct replay_file_header_ex {
    char video_fourcc[4]; // "VMX " or "JPEG"
    char audio_fourcc[4]; // "in32"
    int  audio_channels;
};
```

When nvJPEG is used for recording, set `video_fourcc = "JPEG"`.  The producer
reads this header and selects the appropriate codec for decoding.

### 5.2 Compatibility Matrix

| File recorded with | Can decode with VMX | Can decode with nvJPEG |
|---|---|---|
| VMX | **Yes** | No |
| nvJPEG | No | **Yes** |

Files are **not cross-compatible** — the compressed bitstreams are fundamentally
different formats.  This is documented and expected.

---

## 6. AMCP Interface

### 6.1 Recording (Consumer)

```
ADD 1 REPLAY recording CODEC VMX QUALITY SQ        (default, unchanged)
ADD 1 REPLAY recording CODEC NVJPEG QUALITY 85     (nvJPEG, quality 1-100)
ADD 1 REPLAY recording CODEC NVJPEG QUALITY 85 GPU 0
```

### 6.2 Playback with Interpolation

```
PLAY 1-1 recording SPEED 0.25 INTERPOLATION NVOF      (hardware optical flow)
PLAY 1-1 recording SPEED 0.25 INTERPOLATION RIFE      (AI neural network)
PLAY 1-1 recording SPEED 0.5  INTERPOLATION BLEND     (existing frame average)
PLAY 1-1 recording SPEED 0.5  INTERPOLATION NONE      (frame repeat)
```

### 6.3 Runtime Control

```
CALL 1-1 INTERPOLATION NVOF|RIFE|BLEND|NONE
CALL 1-1 CODEC VMX|NVJPEG                   (for live recording switch)
```

### 6.4 Producer State (OSC/diagnostics)

```
file/codec          = "vmx" | "nvjpeg"
file/interpolation  = "nvof" | "rife" | "blend" | "none"
nvof/flow_ms        = 2.1        (OFA hardware flow time)
nvof/warp_ms        = 1.8        (FRUC warp time)
nvof/repeated       = false      (FRUC frame repetition flag)
nvjpeg/encode_ms    = 3.2
nvjpeg/decode_ms    = 2.5
gpu/vram_used_mb    = 412
```

---

## 7. CMake Integration

```cmake
# In src/modules/replay/CMakeLists.txt

# --- Optional: nvJPEG codec (requires CUDA Toolkit) ---
option(ENABLE_NVJPEG "GPU-accelerated JPEG encoding/decoding via nvJPEG" OFF)

if(ENABLE_NVJPEG)
    find_package(CUDAToolkit REQUIRED)
    target_compile_definitions(replay PRIVATE ENABLE_NVJPEG)
    target_link_libraries(replay PRIVATE CUDA::nvjpeg CUDA::cudart)
    target_sources(replay PRIVATE
        codec/replay_nvjpeg_codec.cpp
    )
endif()

# --- Optional: NVIDIA Optical Flow (requires NVOF SDK) ---
option(ENABLE_NVOF "Hardware optical flow interpolation via NVOF SDK" OFF)

if(ENABLE_NVOF)
    find_package(CUDAToolkit REQUIRED)

    # NVOF SDK path (user-provided or auto-detected)
    set(NVOF_SDK_DIR "" CACHE PATH "Path to Optical_Flow_SDK_5.0.7")

    if(NOT NVOF_SDK_DIR)
        message(FATAL_ERROR "NVOF_SDK_DIR must be set when ENABLE_NVOF is ON")
    endif()

    # NvOF base library (MIT licensed, compiled from source)
    add_library(nvof_base STATIC
        ${NVOF_SDK_DIR}/Common/NvOFBase/NvOF.cpp
        ${NVOF_SDK_DIR}/Common/NvOFBase/NvOFCuda.cpp
        ${NVOF_SDK_DIR}/Common/Utils/NvOFUtils.cpp
        ${NVOF_SDK_DIR}/Common/Utils/NvOFUtilsCuda.cpp
    )
    target_include_directories(nvof_base PUBLIC
        ${NVOF_SDK_DIR}/NvOFInterface
        ${NVOF_SDK_DIR}/Common/NvOFBase
        ${NVOF_SDK_DIR}/Common/Utils
    )
    target_link_libraries(nvof_base PUBLIC CUDA::cuda_driver)

    # NvOFFRUC (pre-built DLL, NVIDIA EULA)
    # The DLL is loaded at runtime via LoadLibrary/dlopen
    target_include_directories(replay PRIVATE
        ${NVOF_SDK_DIR}/NvOFFRUC/Interface
    )

    target_compile_definitions(replay PRIVATE ENABLE_NVOF)
    target_link_libraries(replay PRIVATE nvof_base CUDA::cudart)
    target_sources(replay PRIVATE
        interpolation/nvof_interpolator.cpp
    )
endif()
```

### 7.1 Runtime DLL Loading

NvOFFRUC.dll is loaded dynamically (not linked), matching the SDK sample pattern:

```cpp
// Load at runtime — no link-time dependency
HMODULE hDLL = LoadLibraryExW(L"NvOFFRUC.dll", NULL, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
auto NvOFFRUCCreate = (PtrToFuncNvOFFRUCCreate)GetProcAddress(hDLL, "NvOFFRUCCreate");
auto NvOFFRUCProcess = (PtrToFuncNvOFFRUCProcess)GetProcAddress(hDLL, "NvOFFRUCProcess");
// ... etc
```

This means the NVOF feature degrades gracefully — if `NvOFFRUC.dll` is not
present, the feature is simply unavailable at runtime.

---

## 8. Implementation Phases

### Phase 1 — Codec Abstraction + nvJPEG Encode (Medium)

1. Create `replay_codec` abstract interface
2. Wrap existing VMX calls in `vmx_codec` implementation
3. Implement `nvjpeg_codec` with CUDA encode pipeline
4. Add `CODEC` AMCP parameter to consumer
5. Update `.mav` header to store codec FourCC
6. Test: record with nvJPEG, verify file integrity
7. Benchmark: encode latency, GPU utilization, bitrate vs quality

### Phase 2 — nvJPEG Decode + Round-Trip (Medium)

1. Implement nvJPEG decode in `nvjpeg_codec`
2. Auto-detect codec from `.mav` file header FourCC
3. Write BGR↔BGRA conversion kernel (or CPU path)
4. Test: record with nvJPEG → playback with nvJPEG
5. Test: cross-codec error handling (VMX file + nvJPEG decoder = clean error)

### Phase 3 — NvOFFRUC Integration (Medium)

1. Implement `nvof_interpolator` using FRUC DLL
2. Runtime DLL loading with graceful fallback
3. CUDA context management (share with nvJPEG or create dedicated)
4. Integrate into `replay_producer::receive_impl()` slow-motion path
5. Add `INTERPOLATION NVOF` AMCP parameter
6. Test at 1080p50 with 0.25× and 0.5× speeds
7. Verify frame repetition detection works correctly

### Phase 4 — Production Hardening (Medium)

1. GPU memory pool: pre-allocate VRAM for codec + NVOF to avoid per-frame allocs
2. Async pipeline: overlap CPU decode with GPU upload/interpolation
3. Graceful fallback: if GPU OOM or NVOF fails, auto-switch to CPU path
4. Multi-GPU: allow codec on GPU 0, NVOF on GPU 1
5. Config: `<replay-gpu>`, `<replay-codec>`, `<replay-interpolation>` in casparcg.config

### Phase 5 — Raw NvOF API (Optional, Advanced)

1. Replace FRUC DLL with direct NvOF API calls
2. Implement custom warp kernel with:
   - Bidirectional flow (forward + backward)
   - Per-pixel confidence weighting from cost buffer
   - Occlusion detection via forward-backward consistency check
3. ROI-based interpolation (only interpolate motion areas)
4. Benchmark quality vs FRUC vs RIFE

---

## 9. Timing Budget Analysis

### 9.1 nvJPEG Encode + Record (1080p50, 20 ms budget)

| Step | Time | Notes |
|---|---|---|
| Receive frame from mixer | ~0.5 ms | BGRA in CPU memory |
| cudaMemcpyAsync H2D (8.3 MB) | ~0.7 ms | PCIe 3.0 x16 |
| nvjpegEncodeImage | ~3–5 ms | CUDA cores, quality 85 |
| nvjpegEncodeRetrieveBitstream | ~0.5 ms | D2H compressed stream |
| Write to .mav file | ~0.5 ms | Async I/O |
| **Total** | **~5–7 ms** | **Fits within 20 ms** |

### 9.2 NVOF Slow-Motion Playback (1080p50 @ 0.25×, 20 ms budget)

| Step | Time | Notes |
|---|---|---|
| VMX decode frame | ~2 ms | CPU, cached frame A |
| cudaMemcpyAsync H2D (8.3 MB) | ~0.7 ms | Upload frame to GPU |
| NvOFFRUCProcess (OFA + warp) | ~3–5 ms | Hardware flow + CUDA warp |
| cudaMemcpyAsync D2H (8.3 MB) | ~0.7 ms | Download interpolated frame |
| Build draw_frame | ~0.5 ms | |
| **Total** | **~7–9 ms** | **Fits within 20 ms** |

### 9.3 Combined: nvJPEG Decode + NVOF Interpolation (1080p50, 20 ms budget)

| Step | Time | Notes |
|---|---|---|
| nvJPEG decode (GPU) | ~2–4 ms | Frame already on GPU! |
| NvOFFRUCProcess | ~3–5 ms | No extra upload needed |
| cudaMemcpyAsync D2H | ~0.7 ms | Final result only |
| Build draw_frame | ~0.5 ms | |
| **Total** | **~6–10 ms** | **Fits within 20 ms** |

**Key insight:** When using nvJPEG for both recording and playback, the decoded
frame is **already on GPU memory** — eliminating the H2D upload that NVOF
would otherwise need.  This makes nvJPEG + NVOF a natural pairing.

---

## 10. GPU Memory Budget

| Component | VRAM per instance | Notes |
|---|---|---|
| nvJPEG handle + state | ~100–200 MB | Encoder + decoder state |
| nvJPEG encode buffer | ~32 MB | Pre-allocated for 1080p |
| nvJPEG decode buffer | ~32 MB | Pre-allocated for 1080p |
| NVOF (OFA session) | ~50–100 MB | Flow vector buffers |
| NvOFFRUC state | ~200–400 MB | Internal warp buffers |
| Frame buffers (3×) | ~25 MB | Ping-pong input + output |
| **Total per stream** | **~450–800 MB** | |

**RTX 3070 (8 GB):** 1–2 streams with NVOF, or 2–3 without NVOF
**RTX 4090 (24 GB):** 4–6 streams with NVOF
**A6000 (48 GB):** 8+ streams

---

## 11. Dependencies

| Dependency | Version | License | Required by | Size |
|---|---|---|---|---|
| CUDA Toolkit | ≥ 12.0 | NVIDIA EULA | nvJPEG + NVOF | ~2 GB (dev) |
| nvJPEG | In CUDA Toolkit | NVIDIA EULA | Codec | ~20 MB (DLLs) |
| NVOF SDK 5.0.7 | 5.0.7 | MIT (base) + EULA (FRUC) | Interpolation | ~5 MB |
| NvOFFRUC.dll | In NVOF SDK | NVIDIA EULA | FRUC interpolation | ~8 MB |
| libvmx | master | MIT | Default codec | ~200 KB |

---

## 12. Risk Assessment

| Risk | Impact | Probability | Mitigation |
|---|---|---|---|
| nvJPEG 8-bit only loses precision | Quality loss in 10-bit workflows | Medium | Default to VMX for 10-bit; nvJPEG only for 8-bit |
| No alpha in nvJPEG | Broken compositing | High if alpha needed | VMX remains default; nvJPEG only when alpha not required |
| NvOFFRUC.dll proprietary | Can't ship in open source | High | Runtime DLL loading; user provides DLL from NVIDIA |
| FRUC interpolation artifacts | Visible in broadcast | Medium | Frame repetition flag → fallback to frame repeat |
| GPU OOM with multi-stream | Crash | Medium | VRAM budget tracking; refuse to init if insufficient |
| PCIe bandwidth bottleneck at 4K | Missed frames | Medium | nvJPEG decode-on-GPU avoids H2D transfer |
| NVOF unavailable (pre-Turing) | No interpolation | Low (most modern GPUs) | Graceful fallback to frame repeat/RIFE |
| CUDA context conflicts with CasparCG OpenGL | Corruption | Medium | Dedicated CUDA context; no GL-CUDA interop |

---

## 13. File Layout After Implementation

```
src/modules/replay/
    CMakeLists.txt                          # Updated with ENABLE_NVJPEG, ENABLE_NVOF
    replay_producer.h / .cpp                # Modified: codec selection, interpolation
    replay_consumer.h / .cpp                # Modified: codec selection
    replay.h / .cpp                         # Updated: CODEC, INTERPOLATION params
    replay_segmented_storage.h / .cpp       # Unchanged
    replay_file_operations.h / .cpp         # Minor: FourCC handling
    replay_extended_index.h                 # Unchanged

    codec/
        replay_codec.h                      # Abstract codec interface
        replay_vmx_codec.cpp                # VMX wrapper (refactor of existing)
        replay_nvjpeg_codec.h               # nvJPEG codec header
        replay_nvjpeg_codec.cpp             # nvJPEG CUDA implementation

    interpolation/
        replay_interpolator.h               # Abstract interpolation interface
        nvof_interpolator.h                 # NVOF/FRUC header
        nvof_interpolator.cpp               # NVOF/FRUC implementation

    rife/                                   # (from existing RIFE plan)
        rife_interpolator.h
        rife_interpolator.cpp
        rife_ncnn_impl.cpp
        models/
```

---

## 14. Decision Matrix — When to Use What

| Scenario | Codec | Interpolation | Why |
|---|---|---|---|
| Standard 1080p replay | VMX SQ | NVOF | Best CPU/GPU split |
| 10-bit HDR replay | VMX HQ | NVOF | nvJPEG can't do 10-bit |
| Alpha channel (keyed) | VMX HQ | BLEND | nvJPEG has no alpha |
| Weak CPU + strong GPU | nvJPEG | NVOF | Offload everything to GPU |
| Maximum slow-mo quality | VMX SQ | RIFE | Best interpolation quality |
| Pre-Turing GPU | VMX SQ | RIFE (ncnn/Vulkan) | No OFA hardware |
| No NVIDIA GPU | VMX SQ | BLEND/NONE | CPU only |
| Multi-cam 8× 1080p | VMX SQ | NONE (record only) | Minimize GPU load |
| 4K + slow-mo | VMX SQ | NVOF | NVOF handles 4K in budget |
| Archive/export | VMX HQ | N/A | Best quality, smallest file |
