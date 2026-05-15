# RIFE Slow-Motion Frame Interpolation — Implementation Plan

## 1. Executive Summary

Add AI-powered frame interpolation to the CasparCG replay module using
**RIFE (Real-Time Intermediate Flow Estimation)** to produce broadcast-quality
slow-motion output.  Current slow-motion plays back at fractional speed by
repeating/skipping decoded VMX frames.  RIFE synthesises intermediate frames
from adjacent real frames, producing motion that looks natural instead of
stuttery.

**Target:** 0.25× slow-mo at 1080p50 in real-time on an RTX 3070.

---

## 2. VMX Codec vs nvJPEG (MAV Edition) — Performance Comparison

### Architecture

| Aspect | VMX (libvmx) | MJPEG (libjpeg-turbo) | MJPEG (nvJPEG / CUDA) |
|---|---|---|---|
| Transform | 8×8 DCT | 8×8 DCT | 8×8 DCT |
| Entropy coding | Golomb / Exp-Golomb LUTs | Huffman | Huffman |
| Chroma | 4:2:2:4 (native alpha) | 4:2:0 or 4:2:2 (no alpha) | 4:2:0 or 4:2:2 (no alpha) |
| Parallelism | Slice-based (16 px strips, multi-thread) | Restart intervals (optional) | GPU-wide parallelism |
| SIMD | Fused DCT+Quant+Zigzag in SSE4/AVX2 | Separate stages | N/A (CUDA kernels) |
| Execution | CPU only | CPU only | GPU (CUDA cores) |
| Interlace | First-class (field-aware slicing) | Not native | Not native |
| 10-bit | Yes (P216, PA16) | Rare | No |
| Preview decode | DC-only 1/8 res fast path | No | No |

### Encode Performance (1080p, quality ≈ 90)

| Codec | Device | Streams @ 50 fps | Encode time / frame | Bitrate |
|---|---|---|---|---|
| **VMX SQ** | i7 AVX2 | **~6–8×** 1080p50 | ~2–3 ms | ~130 Mbps |
| **VMX HQ** | i7 AVX2 | ~4–6× 1080p50 | ~3–4 ms | ~260 Mbps |
| libjpeg-turbo | i7 AVX2 | ~3× 1080i50 | ~6–8 ms | ~variable |
| nvJPEG | RTX 3070 | ~4–6× 1080p50 (claimed) | ~3–5 ms | ~variable |

### Decode Performance (1080p)

| Codec | Device | Decode time / frame | Notes |
|---|---|---|---|
| **VMX** | i7 AVX2 | ~2 ms | Fused SIMD inverse path |
| libjpeg-turbo | i7 AVX2 | ~4–6 ms | No fused path |
| nvJPEG (decode) | RTX 3070 | Not implemented in MAV fork | Marked "low priority" |

### Verdict

**VMX is already faster than both libjpeg-turbo and nvJPEG for our use case.**
It was purpose-built for broadcast replay (vMix Instant Replay origin), with
fused DCT+Quant+Zigzag SIMD passes and slice-based parallelism.  The nvJPEG
approach only makes sense if you're stuck with MJPEG files and need GPU
offloading — but we aren't.  VMX encodes ~2–3× faster than libjpeg-turbo on
the same CPU and its decode path is similarly faster.

The GPU should be reserved for RIFE inference, not wasted on JPEG encoding.

---

## 3. Why RIFE

| Criterion | NVOF (SDK 5.0) | RIFE v4.25+ |
|---|---|---|
| Quality | Good (hardware-fixed logic) | Excellent (learned motion model) |
| Occlusion handling | Limited | Strong (learned occlusion masks) |
| License | Proprietary SDK | **MIT** (compatible with GPL-3) |
| Hardware requirement | NVOFA unit (Turing+) | Any CUDA GPU (or Vulkan via ncnn) |
| GPU load | Dedicated HW, ~0% CUDA | CUDA/Tensor cores, ~30-60% GPU |
| Inference @ 1080p | ~2–3 ms (HW unit) | ~5–8 ms (TensorRT FP16) |
| Active development | Last release Feb 2023 | Active (v4.26, Sept 2024) |
| Industry adoption | Niche | DaVinci Resolve, Topaz, FlowFrames |

RIFE wins on quality, license, portability, and active maintenance.  NVOF wins
only on raw speed and zero-GPU-load, but at 5–8 ms per frame RIFE is fast
enough for real-time 1080p50.

---

## 4. Integration Options

### Option A: TensorRT (recommended for production)

- Export RIFE v4.25 model to ONNX → build TensorRT engine at first run
- C++ integration via TensorRT C API
- FP16 inference on Tensor cores → ~5 ms / frame @ 1080p
- Engine is GPU-specific (cached after first build, ~30s build time)
- Dependency: TensorRT runtime DLLs (~500 MB)

### Option B: ncnn + Vulkan (portable fallback)

- Uses Tencent ncnn framework with Vulkan compute shaders
- Works on NVIDIA, AMD, Intel GPUs
- C++ library, CMake-friendly, MIT licensed
- Slower than TensorRT (~8–15 ms / frame) but no CUDA dependency
- Existing project: `rife-ncnn-vulkan` (MIT license, battle-tested)

### Option C: ONNX Runtime + CUDA (middle ground)

- Load RIFE ONNX model directly
- Less optimal than TensorRT but simpler build
- ~8–12 ms / frame @ 1080p FP16

**Recommendation:** Start with **Option B (ncnn/Vulkan)** for rapid prototyping
and broad GPU compatibility, then add **Option A (TensorRT)** as an optional
high-performance path once correctness is validated.

---

## 5. Architecture

```
replay_producer::receive_impl()
    │
    ├─ speed == 1.0 → normal decode path (unchanged)
    │
    └─ speed < 1.0 (slow-mo) → RIFE interpolation path
        │
        ├─ Compute fractional position: t = frac(frame_accumulator)
        │   e.g. at 0.25× speed: t cycles through 0.25, 0.50, 0.75, 0.00
        │
        ├─ When t == 0.0 → output real decoded frame (no interpolation)
        │
        └─ When t != 0.0 → interpolate
            │
            ├─ frame_A = VMX decode of floor frame
            ├─ frame_B = VMX decode of ceil frame
            │
            ├─ Upload frame_A, frame_B to GPU (if not cached)
            │
            ├─ rife_interpolator->interpolate(frame_A, frame_B, t)
            │   → returns interpolated GPU buffer
            │
            ├─ Download result to CPU (BGRA)
            │
            └─ Build CasparCG draw_frame from result
```

### Key Components

```
src/modules/replay/
    rife/
        rife_interpolator.h        // Abstract interface
        rife_interpolator.cpp      // Factory + model loading
        rife_ncnn_impl.cpp         // ncnn/Vulkan backend
        rife_tensorrt_impl.cpp     // TensorRT backend (optional)
        models/                    // Bundled RIFE v4.25 model weights
            rife-v4.25.param       // ncnn model definition
            rife-v4.25.bin         // ncnn model weights
```

### Interface

```cpp
class rife_interpolator {
public:
    virtual ~rife_interpolator() = default;

    // Interpolate between two BGRA frames at position t ∈ (0, 1)
    // Returns false on error. Result written to dst_bgra.
    virtual bool interpolate(
        const uint8_t* src_a_bgra,   // frame N
        const uint8_t* src_b_bgra,   // frame N+1
        uint8_t*       dst_bgra,     // output interpolated frame
        int            width,
        int            height,
        float          t) = 0;       // 0.0 = frame A, 1.0 = frame B

    // Estimated time per interpolation in milliseconds
    virtual double estimated_ms() const = 0;
};

// Factory — returns nullptr if no GPU available
std::unique_ptr<rife_interpolator> create_rife_interpolator(
    int width, int height, int gpu_id = 0);
```

---

## 6. Frame Scheduling for Fractional Speeds

For speed = 0.25× at 50 fps output:

```
Output frame 0:  real frame 0           (t=0.00, no interp)
Output frame 1:  interp(0, 1, 0.25)     (t=0.25)
Output frame 2:  interp(0, 1, 0.50)     (t=0.50)
Output frame 3:  interp(0, 1, 0.75)     (t=0.75)
Output frame 4:  real frame 1           (t=0.00, no interp)
Output frame 5:  interp(1, 2, 0.25)     (t=0.25)
...
```

For speed = 0.5× at 50 fps output:

```
Output frame 0:  real frame 0           (t=0.00)
Output frame 1:  interp(0, 1, 0.50)     (t=0.50)
Output frame 2:  real frame 1           (t=0.00)
Output frame 3:  interp(1, 2, 0.50)     (t=0.50)
...
```

### Timing Budget (1080p50, 20 ms per output frame)

| Step | Time | Notes |
|---|---|---|
| VMX decode frame A | ~2 ms | Cached from previous iteration |
| VMX decode frame B | ~2 ms | Read + decode |
| GPU upload (2 frames) | ~1 ms | PCIe 3.0 x16, 2× 8 MB |
| RIFE inference | ~5–8 ms | ncnn Vulkan or TensorRT FP16 |
| GPU download (1 frame) | ~0.5 ms | 8 MB result |
| CasparCG frame build | ~0.5 ms | Pixel format conversion |
| **Total** | **~11–14 ms** | **Fits within 20 ms budget** |

---

## 7. AMCP Interface

### New parameter: INTERPOLATION

```
PLAY 1-1 recording SPEED 0.25 INTERPOLATION RIFE
PLAY 1-1 recording SPEED 0.5 INTERPOLATION BLEND    (existing behavior)
PLAY 1-1 recording SPEED 0.5 INTERPOLATION NONE     (frame repeat)
```

Default behavior when `INTERPOLATION` is omitted:
- `speed >= 0.5`: NONE (frame repeat, current behavior)
- `speed < 0.5`: RIFE if available, else NONE

### Runtime control

```
CALL 1-1 INTERPOLATION RIFE|BLEND|NONE
```

### Producer state

```
file/interpolation = "rife" | "blend" | "none"
file/rife_ms       = 6.2        (last interpolation time in ms)
```

---

## 8. Implementation Phases

### Phase 1 — ncnn/Vulkan Prototype (est. scope: medium)

1. Add ncnn as FetchContent dependency (header-only Vulkan backend)
2. Convert RIFE v4.25 PyTorch → ONNX → ncnn format
3. Implement `rife_ncnn_impl.cpp`
4. Integrate into `replay_producer::receive_impl()` slow-motion path
5. Add `INTERPOLATION` AMCP parameter
6. Test at 1080p50 with 0.25× and 0.5× speeds
7. Benchmark GPU memory and latency

### Phase 2 — Production Hardening

1. Frame caching: keep last 2 decoded VMX frames in a ring buffer
   to avoid redundant decodes
2. Async GPU pipeline: upload frame B while interpolating A→B
3. Graceful fallback: if RIFE inference exceeds frame budget,
   automatically switch to frame-repeat for that frame
4. Model bundling: ship `.param` + `.bin` in release package
5. Config option: `<rife-model-path>` in casparcg.config

### Phase 3 — TensorRT Backend (optional, high-performance)

1. Add TensorRT as optional CMake dependency
2. Implement `rife_tensorrt_impl.cpp`
3. Auto-build TensorRT engine on first use (cached)
4. Runtime backend selection based on available libraries

### Phase 4 — Advanced Features

1. Multi-GPU support (interpolation on GPU 1, rendering on GPU 0)
2. Reverse slow-motion with RIFE (interpolate backward)
3. Variable speed ramping with smooth interpolation transitions
4. RIFE-based 1080i→1080p deinterlacing (better than line-doubling)

---

## 9. Dependencies

| Dependency | Version | License | Size | Required |
|---|---|---|---|---|
| ncnn | ≥ 1.0.20240410 | BSD-3 | ~2 MB (lib) | Yes (Phase 1) |
| Vulkan SDK | ≥ 1.3 | Apache 2.0 | Headers only (runtime loader) | Yes (Phase 1) |
| RIFE v4.25 model | v4.25 | MIT | ~15 MB (.param + .bin) | Yes |
| TensorRT | ≥ 10.x | NVIDIA EULA | ~500 MB (DLLs) | No (Phase 3) |

### CMake Integration

```cmake
option(ENABLE_RIFE "AI frame interpolation for slow-motion" OFF)

if(ENABLE_RIFE)
    # ncnn via FetchContent
    FetchContent_Declare(ncnn
        GIT_REPOSITORY https://github.com/Tencent/ncnn.git
        GIT_TAG        20240410
    )
    set(NCNN_VULKAN ON CACHE BOOL "" FORCE)
    set(NCNN_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(NCNN_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
    set(NCNN_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(ncnn)

    target_compile_definitions(replay PRIVATE ENABLE_RIFE)
    target_link_libraries(replay PRIVATE ncnn)
    target_sources(replay PRIVATE
        rife/rife_interpolator.cpp
        rife/rife_ncnn_impl.cpp
    )
endif()
```

---

## 10. Risk Assessment

| Risk | Impact | Mitigation |
|---|---|---|
| RIFE exceeds 20 ms budget on RTX 3070 | Dropped frames | Auto-fallback to frame-repeat; use `--scale=0.5` (half-res flow) |
| GPU memory pressure (RIFE + OpenGL rendering) | OOM crash | Monitor VRAM; RIFE needs ~300 MB; RTX 3070 has 8 GB |
| Vulkan driver incompatibility | Crash on init | Graceful fallback to CPU-only (no interpolation) |
| ncnn model accuracy differs from PyTorch | Visual artifacts | Validate with PSNR/SSIM against PyTorch reference |
| Interlaced content | Field-level artifacts | Deinterlace before interpolation (already done by VMX progressive decode) |
| Reverse playback + RIFE | Wrong motion direction | Swap frame A/B and use `t' = 1 - t` |

---

## 11. File Layout After Implementation

```
src/modules/replay/
    CMakeLists.txt                      # Updated with ENABLE_RIFE option
    replay_producer.h
    replay_producer.cpp                 # Modified: RIFE path in receive_impl
    replay_consumer.h
    replay_consumer.cpp                 # Unchanged
    replay_segmented_storage.h/cpp      # Unchanged
    replay_file_operations.h/cpp        # Unchanged
    replay_extended_index.h             # Unchanged
    replay.h / replay.cpp              # Updated: INTERPOLATION param parsing
    rife/
        rife_interpolator.h             # Abstract interface
        rife_interpolator.cpp           # Factory, model path resolution
        rife_ncnn_impl.h               # ncnn backend header
        rife_ncnn_impl.cpp             # ncnn Vulkan implementation
        models/
            rife-v4.25-lite.param      # ncnn model definition
            rife-v4.25-lite.bin        # ncnn model weights (~7 MB)
```
