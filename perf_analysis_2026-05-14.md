# CasparVP Performance Analysis — 2026-05-14

## Test Setup

- **Channel 1**: 7680×2160 bt2020/pq 16-bit (tested at p25000 and p60000)
- **Channel 2**: 1272×600 p5000 (control — always perfect)
- **Content**: ProRes 422 12288×6144 @ 25fps via CUDA decoder, looped (~30s loop point)
- **GPU 0**: NVIDIA RTX A4000 (16GB) — primary
- **GPU 1**: NVIDIA Quadro P4000 (8GB) — secondary cross-GPU output
- **Measurement**: Per-component TIMING diagnostics every 5 seconds, 40-second windows (8 samples per scenario)
- **Matrix**: 2 mixers (VK, OGL) × 3 output combos (DL+VK, VK-only, DL-only) × 3 screen modes (none, 2×CPU, 2×GPU) = 18 scenarios × 2 frame rates = 36 total

Frame size at 7680×2160 RGBA16: ~127 MB/frame.

> **NOTE (2026-05-15)**: The test matrix above predates the `<gpu-readback-mode>` config parameter.
> The next test battery should expand the DeckLink scenarios to include all readback modes:
> `auto`, `cuda`, `vulkan`, `vulkan-dma`, `cpu`. This adds up to 5 readback variants for each
> VK-mixer + DeckLink scenario (OGL mixer ignores the parameter). At minimum, re-test scenarios
> 1, 3, 7, 9 at 60fps with each readback mode to draw proper conclusions about:
> - `vulkan-dma` vs `vulkan` under CUDA ProRes load (SM contention hypothesis)
> - `vulkan-dma` PCIe bandwidth impact on cross-GPU drops (gpu:1)
> - `cpu` baseline (pure AVX2, no GPU readback at all — useful as control)
> - Whether `cuda` with P1 semaphore interop still drops frames or if the VK readback modes
>   are strictly better under heavy decode

---

## Results — 25fps (target: 40.0ms)

| # | Mixer | Outputs | Screens | Ch1 Avg (ms) | Ch1 Late/5s | DL Late | DL Drops | VK gpu:0 Drops | VK gpu:1 Drops | Screen Drops |
|---|-------|---------|---------|-------------|-------------|---------|----------|----------------|----------------|-------------|
| 1 | VK | DL+VK | None | 40.0 | 0-9 | 0-6 | 0-2 | 0 | 14-49 | — |
| 2 | VK | DL+VK | 2×CPU | 40.0-43.5 | 0-12 | 0-10 | 0-1 | 0 | 26-62 | 0 |
| 3 | VK | DL+VK | 2×GPU | 40.0-41.4 | 0-7 | 0-4 | 0-2 | 0 | 6-20 | 0 |
| 4 | VK | VK | None | 40.0 | 0 | — | — | 0 | 0 | — |
| 5 | VK | VK | 2×CPU | 40.0 | 0 | — | — | 0 | 0 | 0 |
| 6 | VK | VK | 2×GPU | 40.0 | 0 | — | — | 0 | 0 | 0 |
| 7 | VK | DL | None | 40.0-41.3 | 0-7 | 0-4 | 0-2 | — | — | — |
| 8 | VK | DL | 2×CPU | 39.7-41.5 | 0-5 | 0-4 | 0 | — | — | 0 |
| 9 | VK | DL | 2×GPU | 39.8-41.2 | 0-5 | 0-4 | 0 | — | — | 0 |
| 10 | OGL | DL+VK | None | 40.0 | 0 | 0 | 0 | 0 | 0 | — |
| 11 | OGL | DL+VK | 2×CPU | 40.0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 12 | OGL | DL+VK | 2×GPU | 40.0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 13 | OGL | VK | None | 40.0 | 0 | — | — | 0 | 0 | — |
| 14 | OGL | VK | 2×CPU | 40.0 | 0 | — | — | 0 | 0 | 0 |
| 15 | OGL | VK | 2×GPU | 40.0 | 0 | — | — | 0 | 0 | 0 |
| 16 | OGL | DL | None | 40.0-48.1* | 0-8* | 0-21* | 0-32* | — | — | — |
| 17 | OGL | DL | 2×CPU | 40.0 | 0 | 0 | 0 | — | — | 0 |
| 18 | OGL | DL | 2×GPU | 40.0 | 0 | 0 | 0 | — | — | 0 |

*Scenario 16 had a single startup spike (48.1ms / 8 late, DL 21 late / 32 drops) then stabilized perfectly.

**25fps verdict**: Everything works. The only persistent issue is `vulkan_output[gpu:1]` cross-GPU drops (scenarios 1-3), which is a PCIe bandwidth limitation, not a software bug.

---

## Results — 60fps (target: 16.7ms)

### VK Mixer @ 60fps

| # | Outputs | Screens | Ch1 Avg (ms) | Ch1 Late/5s | DL Late/5s | DL Drops/5s | VK gpu:0 Drops/5s | VK gpu:1 Drops/5s | Screen Drops | Throughput |
|---|---------|---------|-------------|-------------|------------|-------------|-------------------|-------------------|-------------|------------|
| 1 | DL+VK | None | 23-24 | 85-91 | 82-96 | 41-53 | 0 | 98-110 | — | ~42fps |
| 2 | DL+VK | 2×CPU | 23-24.5 | 85-87 | 85-96 | 43-55 | 0 | 98-106 | 0 | ~42fps |
| 3 | DL+VK | 2×GPU | 24-24.7 | 86-88 | 91-97 | 40-49 | 0 | 96-107 | 0 | ~41fps |
| **4** | **VK** | **None** | **16.7** | **0** | — | — | **0** | **0** | — | **60fps** |
| **5** | **VK** | **2×CPU** | **16.7** | **0-1** | — | — | **0** | **0** | **0** | **60fps** |
| **6** | **VK** | **2×GPU** | **16.7** | **0** | — | — | **0** | **0** | **0** | **60fps** |
| 7 | DL | None | 20.9-22.4 | 76-102 | 63-76 | 47-67 | — | — | — | ~45fps |
| 8 | DL | 2×CPU | 22-23.7 | 89-95 | 75-88 | 42-62 | — | — | 0 | ~43fps |
| 9 | DL | 2×GPU | 22.1-23.3 | 90-97 | 72-87 | 46-61 | — | — | 0 | ~43fps |

### OGL Mixer @ 60fps

| # | Outputs | Screens | Ch1 Avg (ms) | Ch1 Late/5s | DL Late/5s | DL Drops/5s | VK gpu:0 Drops/5s | VK gpu:1 Drops/5s | Screen Drops | Throughput |
|---|---------|---------|-------------|-------------|------------|-------------|-------------------|-------------------|-------------|------------|
| 10 | DL+VK | None | 16.7 | 48-87 | 0 | 0 | 0 | 148-149 | — | 60fps (ch), ~50% VK1 loss |
| 11 | DL+VK | 2×CPU | 18.0-18.2 | 93-100 | 22-26 | 6-17 | 0 | 114-118 | 0 | ~55fps |
| 12 | DL+VK | 2×GPU | 19.1-19.7 | 102-110 | 38-46 | 5-12 | 0 | 88-102 | 0 | ~51fps |
| **13** | **VK** | **None** | **72-163!!** | 1-3 | — | — | **20-144** | **26-179** | — | **6-14fps** |
| **14** | **VK** | **2×CPU** | **63-143!!** | 1-2 | — | — | **27-96** | **33-80** | 0 | **7-16fps** |
| **15** | **VK** | **2×GPU** | **106-124!!** | 1-7 | — | — | **25-64** | **26-106** | 0 | **8-10fps** |
| 16 | DL | None | 16.3-27.1* | 30-53 | 0-113* | 0-28* | — | — | — | 60fps steady |
| 17 | DL | 2×CPU | 16.7 | 41-60 | 0 | 0 | — | — | 0 | 60fps |
| 18 | DL | 2×GPU | 16.6-16.7 | 39-96 | 0-1 | 0-13 | — | — | 0 | 60fps |

*Scenario 16 had a startup burst, stabilized to 16.7ms / 0 DL drops after first window.

---

## Findings

### F1. VK mixer + VK-only outputs = the only perfect path at 60fps

Scenarios 4-6 are the **only** configurations that achieve true 60fps (16.7ms average, 0 late, 0 drops everywhere) including both GPUs. Adding 2× screen consumers (CPU or GPU mode) causes **zero** degradation. This is the Vulkan pipeline working entirely in GPU memory — no readbacks, no interop, no CPU copies.

### F2. OGL + VK-only outputs is catastrophic at 60fps

Scenarios 13-15 collapse to **6-16fps** (72-163ms per frame) with VK outputs dropping 80-98% of frames. The GL→VK interop path (GL blit → VK semaphore → coordinator submit) cannot sustain 16.7ms intervals. At 25fps (40ms budget) it works fine — this is a pure throughput ceiling.

### F3. VK mixer + DeckLink = capped at ~42fps regardless of screens

All VK+DL scenarios (1-3, 7-9) converge to ~22-24ms average (42-45fps). The bottleneck is the Vulkan→host CPU readback needed by DeckLink. At 127MB/frame, the GPU→CPU transfer + format conversion + DMA to DeckLink cannot be done in 16.7ms. Adding screens does **not** make this worse — screens add 0 drops in all cases.

### F4. OGL + DeckLink-only works perfectly at 60fps

Scenarios 16-18 achieve 16.7ms average with 0 DeckLink drops in steady state. The OGL→CPU→DeckLink path is highly optimized (PBO readback has been tuned for years). The channel-level "late" metric (39-96/5s) uses a 15% threshold (>19.2ms flagged as late) and is overly sensitive — it doesn't reflect actual DeckLink delivery problems.

### F5. Cross-GPU vulkan_output[gpu:1] drops are a PCIe bottleneck

The Quadro P4000 on GPU 1 drops frames consistently:
- **25fps VK+DL+VK**: 14-62 drops/5s (even with no screens)
- **60fps**: 88-149 drops/5s (OGL+DL+VK), 96-110 drops/5s (VK+DL+VK)
- **60fps VK+VK-only**: 0 drops (the only exception!)

The zero-drop case (VK+VK) suggests the cross-GPU transfer CAN work at 60fps when it's the only consumer of the Vulkan frame — the issue is contention when DeckLink readback competes for GPU memory bandwidth.

### F6. Screen consumers never drop frames

**Zero drops across all 36 scenarios.** The non-blocking send with drop-oldest, auto-promotion to GPU strategy when VK mixer is active, and GL context sharing all work correctly. Screen consumers are completely transparent to the pipeline.

### F7. Producer never underflows

The CUDA ProRes decoder (12288×6144 → 7680×2160) keeps up at both 25fps and 60fps. No underflow was observed in any scenario.

---

## Root Cause Analysis

### The DeckLink CUDA-VK pipeline stall (F3)

The DeckLink consumer does **not** use naive CPU readback. When VK mixer is active, it uses `cuda_vk_strategy` — a GPU-direct pipeline:
1. VK render completes → `ensure_render_complete()` (CPU-side fence wait)
2. CUDA imports the VK texture via `cudaImportExternalMemory` (Win32 handle, cached)
3. CUDA kernel reads VK surface directly, packs v210 on GPU (one kernel launch)
4. Async D2H copies only packed v210 (~22MB for 3840×2160, ~42MB for 7680×2160) to pinned host memory
5. Double-buffered ping-pong: returns previous frame's completed buffer

`needs_cpu_frame_data()` returns `false`, so the VK mixer skips `copy_async()` entirely. PCIe bandwidth is ~6× less than full RGBA16 readback.

The bottleneck is **not** PCIe bandwidth but **serialization and GPU contention**:
- `ensure_render_complete()` does a CPU-side VkFence wait, stalling the DeckLink thread until the VK render finishes. In contrast, VK-only outputs use GPU→GPU semaphore chains with zero CPU involvement.
- `cudaStreamSynchronize()` at the top of each frame creates an additional sync point (waiting for previous frame's async D2H).
- CUDA and Vulkan share GPU 0 — launching a CUDA kernel while the next VK render is in flight causes GPU context switching overhead.
- These serial dependencies (fence wait → CUDA kernel → D2H → sync) chain into ~22ms total, exceeding the 16.7ms budget.

### The OGL→VK interop collapse (F2)

When OGL mixer is active but VK outputs are configured, the frame must cross the GL/VK boundary:
1. OGL renders to GL texture
2. GL→VK interop (external memory import or blit through shared handle)
3. VK compositor/output receives frame
4. VK present

At 25fps this path has 40ms budget and works. At 60fps the interop synchronization (GL fence → VK semaphore) serializes everything, and each handoff adds latency that cascades. Without DeckLink in the mix (which provides its own timing), VK outputs starve.

### Cross-GPU bandwidth contention (F5)

PCIe 3.0 x16 theoretical: ~15.75 GB/s. The cross-GPU transfer for 7680×2160 RGBA16 is ~127MB/frame × 60fps = ~7.62 GB/s. When DeckLink's CUDA-VK pipeline is also active on GPU 0, there's GPU scheduler contention: both the CUDA v210 packing kernel and the cross-GPU DMA copy compete for the same GPU compute/copy engines. When VK-only is configured (no DeckLink), the cross-GPU transfer has uncontested access and succeeds.

---

## Proposed Solutions

### P1. ~~Eliminate CPU fence wait in CUDA-VK pipeline~~ ✅ IMPLEMENTED (addresses F3)

**Status**: Implemented and verified. The CPU fence wait has been replaced with GPU-side VK→CUDA timeline semaphore interop.

**What was done**:
- VK render now signals an exportable `VkSemaphore` (timeline) at queue submit time (`image_kernel.cpp`)
- The semaphore handle (`HANDLE`) and value are propagated through the frame pipeline: `frame_data` → `texture_wrapper` → `core::texture` base class
- `cuda_vk_strategy` imports the semaphore via `cudaImportExternalSemaphore` (Win32 handle, multi-slot cache for rotating frame_data slots)
- `try_gpu_wait()` enqueues `cudaWaitExternalSemaphoresAsync` — the CUDA stream waits on the GPU for VK render completion, zero CPU involvement
- Double-buffered device output (`d_v210_[2]`, `d_bgra_[2]`) with per-buffer `cudaEvent_t` for async D2H overlap
- `VK_KHR_external_semaphore_win32` extension added to the Vulkan device

**Measured results** (VK mixer + DeckLink, 7680×2160 bt2020/pq 16-bit @ 60fps, ProRes 422 12288×6144):
- **Before**: fence wait = 22ms, sync = 8ms, total = 30ms → ~42fps effective, DL drops 40-67/5s
- **After**: fence wait = 0.06ms, sync = 28ms, total = 28ms → GPU throughput-limited (VK render + CUDA kernel share GPU 0)
- **Without CUDA decode load** (PNG/solid): fence = 0ms, sync = 2-4ms, total = 2-4ms → perfect 60fps, DL drops = 0

**Key insight**: The semaphore interop successfully eliminated the CPU stall (22ms → 0.06ms). However, the remaining ~28ms sync time is **GPU contention** — VK render and CUDA v210 packing compete for GPU 0's compute resources when the CUDA ProRes decoder is also active. This is a hardware throughput ceiling, not a synchronization issue. With lighter GPU load (no CUDA decode), the pipeline achieves perfect 60fps.

**Remaining opportunity**: Triple-buffering or moving CUDA decode to GPU 1 could further reduce contention.

### P2. Dedicated transfer queue for cross-GPU copy (addresses F5)

**Problem**: Cross-GPU copy to P4000 competes with DeckLink readback on the same command queue/PCIe bandwidth.

**Approach**:
- Use a dedicated Vulkan transfer queue (most GPUs expose `VK_QUEUE_TRANSFER_BIT` queues separate from graphics)
- Submit the cross-GPU blit on this queue with a timeline semaphore dependency on render completion
- This allows the GPU scheduler to pipeline cross-GPU DMA vs host readback

**Expected gain**: Reduce contention between DeckLink readback and cross-GPU transfer. The VK+VK-only zero-drop result proves the hardware CAN do it.

**Complexity**: Medium. Requires queue family enumeration and separate command pool/buffer management for transfers.

### P3. Skip OGL→VK interop at high frame rates (addresses F2)

**Problem**: The GL→VK path is fundamentally too slow at 60fps for this resolution.

**Approach**: When OGL mixer is active and only VK outputs are configured:
- Either: force-switch to VK mixer automatically (it works perfectly in this config)
- Or: route OGL output through a CPU staging path (OGL PBO readback → VK host-visible upload) — this is essentially the OGL→DeckLink path redirected to VK, and we know that path works at 60fps

**Option A** (auto-switch to VK mixer) is preferred since the VK mixer + VK output path is perfect. The server could detect this at config time and warn/override.

**Expected gain**: Eliminates the catastrophic 6-16fps scenario entirely.

**Complexity**: Low (auto-switch) to Medium (CPU staging fallback).

### P4. Reduce the channel-level "late" threshold or make it adaptive (addresses F4 noise)

**Problem**: The 15% threshold (`frame_ms > expected_ms × 1.15`) generates high "late" counts even when DeckLink reports 0 drops. At 60fps, 19.2ms threshold is triggered by normal frame time variance (OS scheduling, GPU clock jitter) that doesn't cause visible artifacts.

**Approach**: Either:
- Increase threshold to 25-30% for high frame rates (>30fps)
- Or: only count consecutive late frames as a problem
- Or: use a running average deviation rather than per-frame threshold

**Expected gain**: Cleaner diagnostic output. No performance impact — this is metrics-only.

**Complexity**: Low.

### P5. Investigate frame-size-aware cross-GPU strategy (longer term)

**Problem**: At 127MB/frame, even PCIe 3.0 x16 is strained for two simultaneous high-bandwidth consumers.

**Approach**:
- Compress frames before cross-GPU transfer (e.g., GPU-side lossless compression via `VK_EXT_host_image_copy` or NvComp)
- Or: render a lower-resolution proxy on GPU 1 if it's a preview output
- Or: use RDMA/GPUDirect if both GPUs support peer-to-peer (check `vkGetPhysicalDeviceGroupProperties`)

**Expected gain**: Could halve cross-GPU bandwidth requirements. Most impactful for multi-GPU setups.

**Complexity**: High.

### P6. Pure Vulkan readback strategy — `vk_readback_strategy` (addresses F3)

**Status**: ✅ Implemented and verified. Config option: `<gpu-readback-mode>vulkan</gpu-readback-mode>`

**What was done**:
- New `vk_readback_strategy` class (`vk_readback_strategy.cpp`, ~700 lines) implements a pure-Vulkan GPU readback path for DeckLink, completely bypassing CUDA in the readback pipeline
- Creates its own `VkDevice` matched to the mixer GPU by LUID, prefers compute-only queue families for minimal contention with the mixer's graphics queue
- Imports mixer frame textures via `VK_KHR_external_memory_win32` (Win32 shared handles)
- Imports timeline semaphores via `VK_KHR_external_semaphore_win32` for GPU-side render completion synchronization (no CPU fence wait)
- GLSL compute shaders for on-GPU format conversion: `vk_readback_v210.comp` (YUV v210 packing) and `vk_readback_bgra.comp` (BGRA copy)
- Triple-buffered with `VkFence` for async host readback overlap
- Fallback: if semaphore import fails, falls back to CPU-side fence wait with warning log

**Measured results** (VK mixer + DeckLink, 7680×2160 bt2020/pq 16-bit @ 60fps, ProRes 422 12288×6144):
- **VK readback timing** (from `[vk_readback] DIAG` info logs):
  - import = 0ms (cached external memory handles)
  - sync = 5-8ms (GPU-side timeline semaphore wait for VK mixer render completion)
  - submit = 0.1-0.3ms (compute shader dispatch + host transfer)
  - **total = 5-8ms** per frame
- **CUDA readback timing** (from P1, same conditions): sync = 28ms, total = 28ms
- **DeckLink end-to-end**: ~32ms with both strategies under full ProRes decode load
- **Without CUDA decode load** (PNG/solid): both strategies achieve perfect 60fps, DL drops = 0

**Key insight**: The VK readback path is **3-5× faster** than CUDA readback (5-8ms vs 28ms) because:
1. Compute-only queue avoids graphics queue contention
2. Pure Vulkan path doesn't compete with CUDA for GPU SM scheduling
3. Timeline semaphore import is more efficient within the same API

However, the DeckLink end-to-end time remains ~32ms under ProRes load because both strategies wait for the same VK mixer render output, which is itself slowed to ~32ms by CUDA ProRes decode competing for GPU 0's SMs. The readback is no longer the bottleneck — the **mixer render under CUDA decode contention** is.

**Remaining opportunity**: Moving CUDA ProRes decode to GPU 1 would eliminate SM contention, allowing the VK mixer to render at full speed and the VK readback's 5-8ms timing to translate directly into ~22-25ms end-to-end (within 60fps budget).

---

## Priority Ranking

| Priority | Solution | Status | Impact | Effort |
|----------|----------|--------|--------|--------|
| ~~1~~ | ~~**P1** — VK→CUDA semaphore interop~~ | ✅ **Done** | Fence 22ms → 0.06ms. Perfect 60fps without CUDA decode load. GPU contention remains under heavy decode. | — |
| ~~1~~ | ~~**P6** — Pure Vulkan readback strategy~~ | ✅ **Done** | VK compute readback 28ms → 5-8ms (no CUDA load). Under CUDA ProRes: 17-37ms (SM contention). | — |
| 1 | **P7** — Vulkan DMA readback (`vulkan-dma`) | **Prototype** | Uses `vkCmdCopyImageToBuffer` on transfer-only queue (DMA engine) + CPU v210 pack. Bypasses compute SMs entirely — should eliminate CUDA contention. Trade: ~3× PCIe bandwidth, ~2-4ms CPU. | Low |
| 1 | **P3** — Auto-switch OGL→VK mixer | Open | Eliminates catastrophic 6-16fps failure mode entirely | Low |
| 2 | **P2** — Dedicated transfer queue | Open | Reduces cross-GPU drops when DL is also active | Medium |
| 3 | **P4** — Adaptive late threshold | Open | Noise reduction in diagnostics | Low |
| 4 | **P5** — Frame-size-aware cross-GPU | Open | Future-proofing for multi-GPU | High |

---

## Summary

The Vulkan pipeline is exceptionally capable when operating purely in GPU memory (VK mixer + VK outputs = perfect at 60fps with screens). Three GPU readback strategies are now available for DeckLink output (configured via `<gpu-readback-mode>`, only active when the channel uses `<accelerator>vulkan</accelerator>`; ignored with OGL mixing):

1. **CUDA** (`cuda`): Uses `cuda_vk_strategy` with GPU-side VK→CUDA timeline semaphore interop (P1). Eliminates CPU fence stall (22ms → 0.06ms) but CUDA v210 packing competes with CUDA ProRes decode for GPU SMs, resulting in 21-24ms avg under load (~42fps, with frame drops).

2. **Vulkan compute** (`vulkan`): Uses `vk_readback_strategy` (P6) — a pure Vulkan compute path with its own VkDevice, compute-only queue, and GLSL compute shaders for format conversion. Without CUDA load: 5-8ms readback. Under CUDA ProRes decode: 17-37ms fence wait due to SM contention between VK compute and CUDA (~27fps, no drops).

3. **Vulkan DMA** (`vulkan-dma`): Uses the same VK import/fence infrastructure but replaces the compute shader with `vkCmdCopyImageToBuffer` on a **transfer-only queue** (DMA/Copy engine), then delegates v210 packing to the existing CPU AVX2 path. The DMA engine runs in parallel with compute SMs, so CUDA ProRes decode should not contend. Trade-offs: ~3× PCIe bandwidth (raw RGBA vs packed v210), ~2-4ms CPU for format conversion. **Prototype — needs testing.**

All strategies apply to both HDR/10-bit (v210) and SDR/8-bit (BGRA) configurations. Without heavy CUDA decode load, all GPU strategies achieve perfect 60fps.

The remaining bottlenecks are: GL→VK interop stalls (OGL mixer, catastrophic at 60fps — P3), GPU scheduler contention (cross-GPU + DeckLink — P2), and diagnostic noise (late threshold — P4). Addressing P3 (auto-switch to VK mixer) would eliminate the most severe remaining failure mode with low effort.

### GPU topology detection

The server now logs GPU interconnect topology at startup via `vulkan_device::log_gpu_topology()`:
- Vulkan device group enumeration (detects NVLink/SLI multi-GPU groups)
- CUDA P2P attribute queries (native atomic support → NVLink, peer access → PCIe P2P, neither → staged via system RAM)
- Current system: RTX A4000 ↔ Quadro P4000 uses staged transfers (no P2P, different architectures)
