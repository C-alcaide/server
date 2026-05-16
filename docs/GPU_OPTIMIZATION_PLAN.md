# GPU Resource Optimization — Eliminating Wasteful Roundtrips

> Analysis and implementation plan for applying PR #1651's "keep data on GPU"
> strategy across all custom modules. Covers pitfalls, production risks, and
> quality concerns that may not surface immediately during testing.

**Status**: Phases 1–5 implemented (items 1–5 & 7 from audit). All 6 audit fixes applied. Builds clean.

## Background

[CasparCG/server PR #1651](https://github.com/CasparCG/server/pull/1651) introduced
GPU texture sharing on `const_frame` — allowing the screen consumer to render directly
from the mixer's GPU texture instead of downloading to CPU and re-uploading. This made
the screen consumer "essentially free" when combined with any other consumer.

This document extends that strategy to all custom modules that currently waste resources
on unnecessary GPU→CPU→GPU roundtrips.

---

## Current State Audit

### Consumers that ARE optimized (GPU-native)
| Consumer | `needs_cpu_frame_data()` | GPU path |
|----------|:---:|---|
| Screen (GPU mode) | `false` | Direct OGL texture bind / VK→GL import |
| Vulkan Output | `false` | VK-native, zero-copy |
| DeckLink (CUDA-VK) | `false` | VK texture → CUDA surface → v210 pack on GPU |
| **Spout** *(Phase 2)* | `!gpu_path_active_` | Shared GL context + `glCopyImageSubData` → `SendTexture` |
| **ProRes encoder** *(Phase 3)* | `!gpu_direct_active_` | Shared GL context + CUDA GL register → GPU-direct encode |
| **FFmpeg consumer** *(Phase 4)* | Conditional | GPU texture accepted from mixer |

### Consumers that still use CPU readback
| Consumer | Issue | Impact |
|----------|-------|--------|
| NDI | `image_data()` → raw pointer to NDI SDK | MEDIUM — no GPU path in standard SDK |
| Replay (VMX) | `image_data()` → VMX CPU encode | LOW — VMX designed for CPU |
| sACN / ArtNet | Few pixels at low refresh rate | LOW — minimal bandwidth |

### Producer optimizations
| Producer | Status |
|----------|--------|
| **FFmpeg (D3D11VA)** *(Phase 5)* | GPU-direct decode → OGL texture via WGL_NV_DX_interop2 |
| CUDA ProRes decoder | Already zero-copy (CUDA→GL/VK) |
| CUDA NotchLC decoder | Already zero-copy (CUDA→GL/VK) |

### Mixer-level optimizations
| Feature | Status |
|---------|--------|
| **OGL readback skip** *(Phase 1.1)* | `cpu_readback_needed_` atomic flag gates `copy_async()` |
| **OGL still-frame cache** *(Phase 1.2)* | Fingerprint-based cache skips GPU composition when inputs unchanged |

---

## Phase 1: OGL Mixer Foundation

### Step 1.1: CPU Readback Skip

**What:** Override `set_cpu_readback_needed()` in OGL `image_mixer`; gate `copy_async()`.

**Files:**
- `src/accelerator/ogl/image/image_mixer.h`
- `src/accelerator/ogl/image/image_mixer.cpp`

**Implementation:**
- Add `std::atomic<bool> cpu_readback_needed_{true}` to `image_renderer`
- Inside `dispatch_async` lambda, after `draw()`:
  - If `!cpu_readback_needed_`: return `{empty_array.share(), target_texture}`
  - Otherwise: return `{ogl_->copy_async(target_texture).share(), target_texture}`
- Override `set_cpu_readback_needed()` on `image_mixer` to forward to renderer

**Thread safety:** Written on channel thread, read on GL thread. The `io_context` dispatch
provides happens-before; relaxed atomic ordering is sufficient.

**Production risks:**
- If a consumer incorrectly declares `needs_cpu_frame_data()=false` but still calls
  `image_data()`, it will get a null pointer → crash. Contract is the same as VK mixer.
- Adding/removing consumers at runtime changes the flag. The channel tick re-evaluates
  `any_consumer_needs_cpu_data()` each frame, so this is handled automatically.

### Step 1.2: Still-Frame Cache

**What:** Skip GPU composition when inputs (textures + transforms) are unchanged.

**Files:**
- `src/accelerator/ogl/image/image_mixer.cpp`

**Implementation:**
- Fingerprint: `vector<pair<shared_ptr<texture>, image_transform>>` per item across layers
- Compare with previous tick's fingerprint before dispatching to GL thread
- On match: return `{cached_cpu_, cached_texture_}` immediately

**Fingerprinting approach:**
- Store `shared_ptr<texture>` (not raw pointers) to prevent pool-recycled addresses from
  causing false cache hits (ABA problem — **Audit Fix #1**)
- `image_transform` has `operator==` — directly comparable
- Clear cache when `layers` is empty (same as VK mixer)

**Production risks:**
- **False positive (stale frame displayed):** Prevented by using `shared_ptr<texture>` which
  keeps old textures alive and prevents address reuse from the texture pool.
- **Memory leak from cached textures:** Cache holds `shared_ptr<texture>`. Cleared when
  `layers` is empty (STOP/CLEAR/REMOVE scenarios).

---

## Phase 2: Spout Consumer GPU Path

### Step 2.1: Shared GL Context

**Files:** `src/modules/spout/consumer/spout_consumer.cpp`

**Implementation:**
- Get `channel_info.gl_share_context` (HGLRC) in `initialize()`
- Create child context via `wglShareLists(mixer_hglrc, consumer_hglrc)`
- On failure: fall back to existing CPU path (log warning)

### Step 2.2: SendTexture() Direct Path

- Override `needs_cpu_frame_data() { return !gpu_path_active_; }`
- `gpu_path_active_` is `std::atomic<bool>` (**Audit Fix #4**)
- OGL mixer + shared context: `dynamic_pointer_cast<ogl::texture>(frame.texture())` →
  `sender_->SendTexture(ogl_tex->id(), GL_TEXTURE_2D, w, h, bInvert)`
- Fallback: existing `SendImage()` CPU path

**Window class cleanup:** `UnregisterClass` in destructor (**Audit Fix #6**)

**Production risks:**
- **Texture orientation flip:** OGL textures are bottom-up. If `bInvert` is wrong, image
  appears upside-down. Must verify against screen_consumer's usage of the same texture.
- **Resolution downscaling:** If `MAX_WIDTH`/`MAX_HEIGHT` is configured, GPU texture is
  full-res. GPU path disabled when scaling is active.
- **Context destruction order:** If channel stops while Spout is mid-send, the shared
  context becomes invalid. The executor thread + drain pattern handles this.

---

## Phase 3: ProRes Consumer GPU-Direct Encode

### Step 3.1: CUDA Import from GL Texture

**Files:** `src/modules/cuda_prores/consumer/prores_consumer.cu`

**Implementation:**
- Create shared GL context on encode thread via `wglShareLists`
- Register mixer's GL texture with CUDA via `cuGraphicsGLRegisterImage`
- Map registered resource → `cudaArray` → encode directly

### Step 3.2: `needs_cpu_frame_data()` Override

```cpp
bool needs_cpu_frame_data() const override { return !gpu_direct_active_; }
```

- `gpu_direct_active_` is `std::atomic<bool>` (**Audit Fix #4**)
- Set during `initialize()` based on shared context + CUDA interop success

---

## Phase 4: FFmpeg Consumer GPU Texture Support

**Files:** `src/modules/ffmpeg/consumer/ffmpeg_consumer.cpp`

The FFmpeg file-output consumer accepts GPU textures from the mixer, reading
them back to CPU only when needed for software encoding. This avoids the
mixer-level readback when the FFmpeg consumer is the only one attached.

**Production risks:**
- **Software encoder breakage:** Must NOT activate GPU-only mode for `libx264`/`libx265`.
- **User filter incompatibility:** If user passes custom filters, the filter graph may need
  CPU frames. Fallback logic handles this gracefully.

---

## Phase 5: FFmpeg Producer D3D11VA → GL Direct

### Overview

**Before:** D3D11VA decode → `av_hwframe_transfer_data` → CPU NV12 → memcpy into
mutable_frame → mixer re-uploads to GPU

**After:** D3D11VA decode → D3D11 Video Processor (NV12→BGRA on GPU) → WGL_NV_DX_interop2
→ GL texture → `glCopyImageSubData` into pooled OGL texture → `frame.texture()` → mixer
uses directly (zero-copy)

### Implementation

**Files:**
- `src/modules/ffmpeg/producer/av_producer.cpp`
- `src/modules/ffmpeg/producer/d3d11_gl_bridge.h` *(new)*
- `src/modules/ffmpeg/producer/d3d11_gl_bridge.cpp` *(new)*

**Bridge architecture (separate TU for MSVC namespace bug):**
- `d3d11_gl_bridge::impl` owns D3D11 Video Processor, WGL interop device, shared GL context
- `convert(AVFrame*)` → VideoProcessorBlt NV12→BGRA → WGL lock → `glCopyImageSubData` →
  WGL unlock → returns `shared_ptr<ogl::texture>` (as `shared_ptr<void>`)
- Bridge textures use the OGL texture pool via `ogl_dev->create_texture()` (**Audit Fix #5**)
- Window class unregistered in `cleanup()` (**Audit Fix #6**)

**Decoder integration:**
- `gpu_direct_mode_` atomic flag on decoder
- Decoded D3D11 frames pushed to `hw_output` queue (separate from `output`)
- Both queues flushed on seek (**Audit Fix #2**)
- GPU-direct only enabled for: progressive video, matching fps, no custom vfilters

**Fallback path (Audit Fix #3):**
When bridge `convert()` fails:
1. `av_hwframe_transfer_data()` → CPU NV12
2. `sws_scale()` NV12→BGRA (inline conversion)
3. BGRA frame passed to `make_frame()` normally

Previously the NV12 frame went directly to `make_frame()` which didn't recognize
the format → `pixel_format::invalid` → black frame.

### NV12 Handling
- Decoded texture is NV12 (Y + interleaved UV)
- Mixer expects BGRA
- D3D11 Video Processor handles NV12→BGRA on GPU (zero CPU involvement)
- Fallback uses sws_scale for CPU conversion

### Filter Graph Constraint
- **Critical:** `bwdif` (deinterlace) and `fps` filters require CPU frames
- GPU-direct path only viable for progressive, native-framerate sources
- Gate: `if (progressive && !needs_filter) enable_gpu_direct()`

**Production risks:**
- **Frame pool exhaustion:** FFmpeg's `hw_frames_ctx` has limited pool (default ~20 frames).
  If we hold textures too long for import, decoder blocks. Must import+release quickly.
- **D3D11 device mismatch:** FFmpeg creates its own D3D11 device. Shared handles only
  work on the same adapter. Must verify same GPU.
- **Deinterlaced content regression:** If progressive detection is wrong (e.g., TFF content
  flagged as progressive), GPU path sends interlaced frames unprocessed → visual combing.

---

## Core Framework Changes

### `const_frame` Texture Support

**Files**: `src/core/frame/frame.h`, `src/core/frame/frame.cpp`

`const_frame` gained an optional `std::shared_ptr<core::texture>` field.
Producers can attach a GPU texture at construction time. The mixer's `visit()`
method checks `frame.texture()` and uses it directly when present, bypassing
the normal CPU upload path.

### `frame_factory` GPU Device Handle

**File**: `src/core/frame/frame_factory.h`

Added `virtual void* gpu_device_handle() const { return nullptr; }` so producers
can discover the mixer's OGL device at initialization time (needed for shared GL
context creation in Phase 5).

The OGL `image_mixer::impl` overrides this to return the `ogl::device*`.

### Consumer CPU Readback Query

**File**: `src/core/consumer/output.cpp`

`any_consumer_needs_cpu_data()` iterates all attached consumers. If any returns
`needs_cpu_frame_data() == true`, the mixer performs CPU readback. Otherwise
readback is skipped entirely.

---

## Quality Risk Analysis

These changes modify the **data path** of pixels, not their values. Quality *should* be
identical. However, subtle bugs can introduce visual differences:

| Risk | Cause | Symptom |
|------|-------|---------|
| Texture flip | Wrong `bInvert` / UV coordinate origin | Image upside-down |
| Color matrix bypass | CPU path applied BT.601→709 in sws_scale; GPU path skips it | Slight color shift on SD content |
| Premultiplied alpha | CPU readback may un-premultiply; GPU path doesn't | Semi-transparent edges look different |
| Bit depth truncation | 16-bit VK texture → 8-bit consumer without dithering | Banding in gradients |
| Still-frame cache stale hit | Fingerprint false positive | 1-frame-old content displayed |
| Semaphore/fence race | Reading texture before render completes | Tearing, corruption (only under load) |
| NV12 color range | Full-range vs limited-range decode mismatch | Crushed blacks / blown highlights |

**Testing protocol:**
1. Record reference output via existing CPU path (ProRes 4444 lossless)
2. Enable GPU optimization
3. Record identical content again
4. Frame-by-frame PSNR comparison — any frame below infinity dB indicates a difference
5. Focus areas: alpha edges, SD content, gradients, static-to-motion transitions

---

## Unchanged-Frame Detection — Why Consumers Should NOT Skip Sending

| Consumer | Safe to skip? | Why |
|----------|:---:|---|
| Screen | Partially | Could skip present, but V-sync timing expects regular swaps |
| Spout | **No** | Receivers use `SetFrameCount` semaphore — skipping breaks sync |
| NDI | **No** | NDI protocol expects steady frame delivery; gaps → receiver disconnect |
| ProRes/FFmpeg record | **No** | Missing frames → wrong duration, broken timecode, A/V desync |
| Replay record | **No** | Frame-accurate playback requires every slot filled |
| DeckLink SDI | **No** | SDI output requires a frame every clock tick — blanking if skipped |

**The correct pattern:** The still-frame cache (Phase 1.2) already provides the benefit
of unchanged-frame detection — it skips **GPU composition work** but still produces a
valid frame (cached result) every tick. Every consumer receives a frame every tick; they
just get the same cached GPU texture. This is the safe approach.

---

## Audit Fixes Applied

Six issues were identified and fixed across two audit passes:

| # | Severity | Issue | Fix |
|---|----------|-------|-----|
| 1 | Critical | Still-frame cache fingerprint used raw `texture*` → ABA from pool recycling | Changed to `shared_ptr<texture>` |
| 2 | Critical | `hw_output` queue not flushed on seek → stale D3D11 frames after seek | Added `hw_output` clear in flush handler |
| 3 | Critical | GPU-direct fallback passes NV12 to `make_frame()` → `pixel_format::invalid` → black | Added `sws_scale` NV12→BGRA conversion |
| 4 | Medium | `gpu_path_active_` / `gpu_direct_active_` were plain `bool` → data race | Changed to `std::atomic<bool>` |
| 5 | Low | Bridge textures bypassed OGL pool → VRAM churn | Changed to `ogl_dev->create_texture()` |
| 6 | Low | Window classes never unregistered → OS resource leak | Added `UnregisterClass` in destructors |

---

## Dependency Graph

```
Phase 1.1 (OGL readback skip)─────────┐
Phase 1.2 (OGL still-frame cache)─────┤ can be done in parallel
                                       │
Phase 2 (Spout GPU)────────────────────┤ parallel; full benefit after 1.1
                                       │
Phase 3 (ProRes GPU-direct)────────────┤ parallel; OGL mixer only
                                       │
Phase 4 (FFmpeg consumer GPU)──────────┤ parallel; independent
                                       │
Phase 5 (FFmpeg producer D3D11VA)──────┘ LAST (most complex/risky)
```

All phases have CPU fallback paths, so partial implementation is safe. Each phase
independently provides value.

---

## Files Modified (Complete List)

| Phase | File | Change |
|-------|------|--------|
| 1.1 | `src/accelerator/ogl/image/image_mixer.h` | Add `set_cpu_readback_needed()` override |
| 1.1 | `src/accelerator/ogl/image/image_mixer.cpp` | Add atomic + conditional readback |
| 1.2 | `src/accelerator/ogl/image/image_mixer.cpp` | Add fingerprint cache (shared_ptr) |
| 2 | `src/modules/spout/consumer/spout_consumer.cpp` | Shared GL context + SendTexture + atomic flag |
| 3 | `src/modules/cuda_prores/consumer/prores_consumer.cu` | CUDA GL register + atomic flags |
| 4 | `src/modules/ffmpeg/consumer/ffmpeg_consumer.cpp` | GPU texture acceptance |
| 5 | `src/modules/ffmpeg/producer/av_producer.cpp` | D3D11VA GPU-direct + hw_output queue + NV12→BGRA fallback |
| 5 | `src/modules/ffmpeg/producer/d3d11_gl_bridge.h` | Bridge header (new file) |
| 5 | `src/modules/ffmpeg/producer/d3d11_gl_bridge.cpp` | Bridge impl (new file) + pool textures + UnregisterClass |
| 5 | `src/modules/ffmpeg/CMakeLists.txt` | Added bridge source files |
| — | `src/core/frame/frame.h` / `frame.cpp` | Texture field on const_frame |
| — | `src/core/frame/frame_factory.h` | `gpu_device_handle()` virtual |
| — | `src/core/consumer/output.cpp` | `any_consumer_needs_cpu_data()` |

---

## NDI Advanced SDK Assessment

### Availability
- **Paid license** from Vizrt (contact NDI sales team)
- Requires NDA + commercial agreement
- NOT included in the free NDI SDK v6 (which we currently use)
- Not publicly downloadable — must apply for access

### What it would provide
- `NDIlib_video_frame_v3_t` with GPU memory metadata
- Direct CUDA/D3D11 buffer passing (no CPU staging)
- Higher performance for GPU-rendered content

### Recommendation
Not worth pursuing unless the Vizrt license is already available. The standard SDK's CPU
path remains the practical ceiling. Phase 1 optimizations (readback skip + still-frame
cache) already reduce the cost of producing CPU frames for NDI.
