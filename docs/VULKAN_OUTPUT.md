# Vulkan Output Module

Low-latency, direct-to-display GPU output consumer for CasparCG. Bypasses the desktop compositor entirely using the Vulkan graphics API, targeting professional broadcast scenarios where frame-accurate timing and deterministic latency are critical.

**Supported platforms**: Windows (via `VK_EXT_full_screen_exclusive` + Win32 surfaces; or `VK_KHR_display` on Windows 11 with Specialized Monitors) and Linux (via `VK_KHR_display` direct scanout).

## Table of Contents

- [System Requirements](#system-requirements)
- [Architecture Overview](#architecture-overview)
- [GPU Tier System](#gpu-tier-system)
- [Frame Transfer Pipeline](#frame-transfer-pipeline)
- [Multi-GPU Transfer](#multi-gpu-transfer)
- [GPU Topology Detection](#gpu-topology-detection)
- [DeckLink CUDA-Vulkan Interop](#decklink-cuda-vulkan-interop)
- [Color Space Conversion](#color-space-conversion)
- [HDR Output](#hdr-output)
- [Synchronization](#synchronization)
- [Subregion Output](#subregion-output)
- [Display Hot-Plug](#display-hot-plug)
- [AMCP Commands](#amcp-commands)
- [Configuration Reference](#configuration-reference)
- [Configuration Examples](#configuration-examples)
- [Build Requirements](#build-requirements)
- [Linux Testing](#linux-testing)
- [Diagnostics](#diagnostics)
- [Troubleshooting](#troubleshooting)

---

## System Requirements

### Windows (Fullscreen Exclusive — default path)

| Requirement | Details |
|---|---|
| **OS** | Windows 10 or later (any edition) |
| **GPU** | NVIDIA Quadro/RTX A-series/RTX Pro (Pro tier) or GeForce RTX (Consumer tier) |
| **Driver** | NVIDIA R535+ (Vulkan 1.3+). R550+ recommended for multi-GPU stability |
| **Display** | Must be attached to the Windows desktop (extended mode) |
| **Admin** | Not required for FSE path |

### Windows (VK_KHR_display direct scanout — optional)

| Requirement | Details |
|---|---|
| **OS** | **Windows 11** (build 22000+) — Enterprise, Pro for Workstations, or IoT Enterprise |
| **GPU** | NVIDIA Quadro/RTX A-series/RTX Pro (Pro tier only) |
| **Driver** | NVIDIA R535+ with `configureDriver.exe` v2.8+ |
| **One-time setup** | Run `configureDriver.exe --set 6` as administrator, then restart |
| **Display** | Must be removed from Windows desktop (Settings → Display → Advanced → "Remove display from desktop") |
| **Admin** | Required for initial configureDriver setup (UAC elevation) |

> **Note**: Windows 10 does NOT support VK_KHR_display regardless of GPU or driver version. The "Specialized Monitors" / "Remove display from desktop" feature is Windows 11 only.

### Linux (VK_KHR_display direct scanout — default path)

| Requirement | Details |
|---|---|
| **OS** | Any Linux distribution with Vulkan support |
| **GPU** | NVIDIA with proprietary driver ≥ 535 (or Mesa for AMD — untested) |
| **Display** | Must NOT be managed by X11/Wayland compositor (`xrandr --output DP-X --off`) |
| **Permissions** | Access to `/dev/dri/card*` (typically via `video` group) |
| **Admin** | Not required (no configureDriver equivalent needed) |

### Automation Features (Windows)

The module includes automatic recovery for common display configuration issues:

| Feature | Behavior |
|---|---|
| **configureDriver auto-invoke** | On Win11 with Pro GPU, automatically runs `configureDriver.exe --set 6` with UAC if VK_KHR_display reports no displays. Skipped on Win10. |
| **Display detach** | After configureDriver succeeds, automatically detaches the target display from the Windows desktop for VK_KHR_display acquisition |
| **Display reattach for FSE** | If VK_KHR_display fails and FSE fallback is needed, detects if the target display was left detached (from crash/previous run) and reattaches via `displayswitch /extend` |
| **GPU-aware display targeting** | FSE window placement matches the Vulkan GPU name against Windows display adapter names to target the correct monitor |

---

## Architecture Overview

The module is organized into ten layers:

```
┌──────────────────────────────────────────────────────────────┐
│ vulkan_output_consumer          (core::frame_consumer impl)  │
│   ├── Frame buffering + present loop (dedicated thread)      │
│   ├── Grid-aligned frame pacing (multi-output sync)          │
│   ├── Swapchain management + hot-plug recovery               │
│   └── Identify overlay                                       │
├──────────────────────────────────────────────────────────────┤
│ vk_device_manager               (Shared VkDevice registry)   │
│   ├── Singleton: one VkDevice per physical GPU               │
│   ├── Thread-safe weak_ptr registry with auto-cleanup        │
│   └── Multiple consumers on the same GPU share one device    │
├──────────────────────────────────────────────────────────────┤
│ gpu_frame_cache                 (Per-GPU transfer cache)     │
│   ├── One transfer per GPU per frame (deduplication)         │
│   ├── First-caller-wins coordination (mutex + condvar)       │
│   ├── Binary semaphore tracking (one wait per frame)         │
│   └── Owns interop_context, shared_texture_pool, affinity   │
├──────────────────────────────────────────────────────────────┤
│ color_convert_pipeline          (Vulkan compute shader)      │
│   ├── DISABLED: mixer now performs all color conversion      │
│   ├── Retained for future per-consumer display transforms    │
│   └── Bypassed entirely (no GPU cost)                        │
├──────────────────────────────────────────────────────────────┤
│ vulkan_device                   (VkInstance + VkDevice RAII)  │
│   ├── GPU enumeration + tier detection + LUID query          │
│   ├── VK_KHR_display (Linux) / FSE+Win32 surface (Windows)   │
│   ├── VBlank fence via VK_EXT_display_control                │
│   └── queue_mutex for thread-safe queue submission           │
├──────────────────────────────────────────────────────────────┤
│ interop_context                 (Dedicated GL blit thread)    │
│   ├── Shared GL context (WGL on Windows, EGL on Linux)       │
│   ├── Own worker thread ("VK Interop GL")                    │
│   └── dispatch_async() for non-blocking blit + signal        │
├──────────────────────────────────────────────────────────────┤
│ shared_texture_pool             (OGL ↔ VK zero-copy)         │
│   ├── Triple-buffered GL_EXT_memory_object textures          │
│   ├── GL_EXT_semaphore / VK_KHR_external_semaphore sync      │
│   └── 8-bit (SDR) or 16-bit (HDR) pixel format               │
├──────────────────────────────────────────────────────────────┤
│ gpu_affinity_context            (Cross-GPU OGL bridge)        │
│   ├── WGL_NV_gpu_affinity (Win) / EGL device enum (Linux)    │
│   ├── Dedicated thread with dispatch_sync() work queue       │
│   └── PBO double-buffered upload (fallback path)             │
├──────────────────────────────────────────────────────────────┤
│ cuda_peer_transfer              (GPU→GPU DMA)                │
│   ├── cudaMemcpyPeerAsync: async PCIe/NVLink DMA (no CPU copy)│
│   ├── GPU-side event chain (zero CPU sync points)            │
│   ├── CUDA↔GL interop on both GPUs                          │
│   └── Graceful fallback to PBO path if CUDA unavailable      │
├──────────────────────────────────────────────────────────────┤
│ nvapi_helpers                   (NVIDIA NvAPI integration)    │
│   ├── EDID readback (HDR capability, luminance, bit depth)   │
│   ├── EDID injection / persistence (headless outputs)        │
│   ├── Dedicated display acquire / release (crash-safe blank) │
│   ├── Hardware HDR (NvAPI UHDA: display engine PQ + BT.2020) │
│   ├── Quadro Sync II framelock configuration                 │
│   └── GSync status monitoring                                │
└──────────────────────────────────────────────────────────────┘
```

### Data Flow (OGL Mixer)

When the **OGL mixer** is active, frames are transferred from OpenGL to Vulkan via interop:

```
CasparCG Mixer (OGL, GPU A)
   │
   ├─ Same-GPU zero-copy path (preferred):
   │   send() ──dispatch_sync──▸ gpu_frame_cache::submit_frame()
   │                                │  (first caller wins — deduplicates per GPU)
   │                                │
   │              interop_context (dedicated GL thread)
   │                                │
   │              OGL texture ──blit──▸ shared_texture_pool (GL_EXT_memory_object)
   │                                     │  signal_gl() + swap()
   │                              Platform handle export (Win32 HANDLE / fd)
   │                                     │
   │                            VkImage (VK_KHR_external_memory)
   │                                     │
   │              [present thread]  vkCmdBlitImage ──▸ swapchain image
   │
   ├─ Cross-GPU CUDA peer DMA path (fastest cross-GPU):
   │   gpu_frame_cache::submit_frame() (first caller wins)
   │   OGL texture ──cudaGraphicsGLRegisterImage──▸ CUDA array (GPU A)
   │                                                 │
   │                          cudaMemcpy2DFromArray ──▸ staging buffer (GPU A)
   │                                  cudaEventRecord(src_ready)
   │                                                       │
   │                              cudaStreamWaitEvent(src_ready)  [GPU-side]
   │                                   cudaMemcpyPeerAsync (PCIe/NVLink DMA)
   │                                  cudaEventRecord(peer_event)
   │                                                       │
   │                              cudaStreamWaitEvent(peer_event) [GPU-side]
   │                                              staging buffer (GPU B)
   │                                                       │
   │                          cudaMemcpy ──▸ PBO (GPU B)
   │                                                       │
   │                                     glTexSubImage2D ──▸ shared_texture_pool
   │                                                              │
   │                                                     VkImage ──▸ swapchain image
   │
   ├─ Cross-GPU PBO fallback path (no CUDA required):
   │   gpu_frame_cache::submit_frame() (first caller wins)
   │   frame.image_data() ──PBO upload──▸ texture (GPU B, affinity context)
   │                                        │
   │                          glCopyImageSubData ──▸ shared_texture_pool
   │                                                       │
   │                                              VkImage ──▸ swapchain image
   │
   ├─ CPU fallback path:
   │   frame.image_data() ──memcpy──▸ staging buffer
   │                                     │
   │                              vkCmdCopyBufferToImage ──▸ swapchain image
   │
   └──▸ vkQueuePresentKHR (under queue_mutex) ──▸ Display
```

### Data Flow (Vulkan Mixer)

When the **Vulkan mixer** is active, the transfer is simpler — no GL interop is needed. The consumer detects the Vulkan mixer via `dynamic_pointer_cast<texture_wrapper>` on the frame's texture:

```
CasparCG Mixer (Vulkan, GPU A)
   │
   ├─ Same-GPU path (preferred):
   │   render() produces VkImage + exported platform handle (Win32 HANDLE / fd)
   │         │
   │   [present thread]  ensure_render_complete() (fence wait)
   │                     import_vk_texture(HANDLE) → VkImage on output device
   │                     vkCmdBlitImage (imported image → swapchain image)
   │
   ├─ Cross-GPU CUDA path (import fails → LUID mismatch detected):
   │   begin_cuda_d2h(HANDLE, src_luid):
   │     CUDA matches src_luid to a CUDA device index
   │     cudaImportExternalMemory(HANDLE) → CUDA buffer (GPU A)
   │     cudaMemcpyAsync (device → host staging buffer)
   │   complete_cuda_d2h():
   │     host staging → vkCmdCopyBufferToImage → swapchain image
   │
   ├─ CPU fallback path:
   │   frame.image_data() ──memcpy──▸ staging buffer
   │                     vkCmdCopyBufferToImage ──▸ swapchain image
   │
   └──▸ vkQueuePresentKHR (under queue_mutex) ──▸ Display
```

The `interop_context`, `shared_texture_pool`, and `gpu_affinity_context` layers are bypassed entirely when the Vulkan mixer is active — they exist only for the OGL→VK bridge.

#### Shared VkDevice and Frame Cache

When multiple consumers target the same GPU (e.g., a 4-output video wall on one GPU), the `vk_device_manager` singleton ensures they all share a single `VkDevice`. This is required for `VK_NV_present_barrier` (which frame-locks swapchains within a device) and avoids the overhead of redundant Vulkan device creation.

The `gpu_frame_cache` deduplicates the mixer→output frame transfer: the first consumer to call `submit_frame()` for a given frame generation performs the actual blit/transfer, while all other consumers on the same GPU block on a condition variable and receive the result. This means a 4-output setup does one transfer per frame, not four.

The GL→VK binary semaphore (OGL mixer path) is also tracked per-frame: `try_consume_semaphore()` uses an atomic exchange so only the first consumer's `vkQueueSubmit` waits on the interop semaphore. Subsequent consumers skip the wait, avoiding a binary semaphore double-wait violation. The `gpu_frame_cache` then bridges this to a timeline semaphore that all consumers wait on in their `vkQueueSubmit`. The Vulkan mixer path bypasses this entirely — synchronization is handled by `ensure_render_complete()` (fence wait) before the imported image is used.

All `vkQueueSubmit` and `vkQueuePresentKHR` calls are serialized through `vulkan_device::queue_mutex()`, satisfying Vulkan's external synchronization requirement on queue objects.

**Same-GPU zero-copy (OGL mixer)** eliminates all CPU-side pixel copies. The OGL mixer output is blitted into a shared texture backed by exportable GPU memory (`GL_EXT_memory_object`), which is imported into Vulkan as a `VkImage` via a platform handle (Win32 HANDLE on Windows, POSIX fd on Linux). The blit runs on a dedicated `interop_context` GL thread via `dispatch_sync()`, so the transfer completes before `submit_frame()` returns. Synchronization between the OpenGL and Vulkan timelines uses paired `GL_EXT_semaphore` / `VkSemaphore` objects.

**Same-GPU zero-copy (Vulkan mixer)** is even simpler: the mixer produces a `VkImage` backed by exportable memory (`VK_KHR_external_memory`). The output consumer calls `ensure_render_complete()` (a GPU fence wait ensuring the render pass is finished), then imports the platform handle as a `VkImage` on the output's device. Since both sides are on the same GPU, the import shares the same physical memory — no copy occurs. The consumer then performs a direct `vkCmdBlitImage` from the imported image into the swapchain image. Imported images are cached (up to 8 slots) to avoid repeated `vkCreateImage` + `vkBindImageMemory` calls as the mixer rotates through its render target pool.

**Cross-GPU CUDA peer DMA (OGL mixer)** uses the GPU's DMA copy engines to transfer pixels directly from GPU A to GPU B over PCIe or NVLink — no CPU involvement. The entire pipeline is fully async with GPU-side event synchronization: `src_ready_event_` signals when the source staging buffer is written, `peer_event_` signals when the peer DMA completes. No CPU sync points exist in the read→peer→write chain (except the final `cudaStreamSynchronize` in `write_dest()` before the GL texture upload). This is the fastest cross-GPU path (~15 GB/s on PCIe 3.0 x16, ~31 GB/s on PCIe 4.0, up to 600 GB/s on NVLink). When matching GPU architectures support P2P, the transfer is a single PCIe hop; otherwise the driver stages through system RAM transparently.

**Cross-GPU CUDA D2H (Vulkan mixer)** is activated when `import_vk_texture()` fails (indicating the source VkImage is on a different GPU). The consumer imports the mixer's platform handle into CUDA on the source GPU (matched by LUID via `cudaGetDeviceProperties`), performs an async device-to-host copy to a staging buffer, then uploads the staging data to a Vulkan image on the output GPU via `vkCmdCopyBufferToImage`. This path involves a host-memory hop but still uses the GPU's DMA copy engine for the D2H transfer.

**Cross-GPU PBO fallback (OGL mixer)** is used when CUDA is unavailable. The mixer's CPU-side pixel buffer (which CasparCG already maintains for non-GPU consumers) is uploaded to GPU B via a double-buffered Pixel Buffer Object on the affinity context's GL thread.

For HDR workflows, all shared textures are allocated as 16-bit (`RGBA16`) instead of 8-bit (`BGRA8`), preserving the full dynamic range from the mixer through to the display.

The CPU fallback path (direct staging buffer upload to Vulkan) is used when no GL interop is possible, e.g. in multi-vendor GPU setups where neither CUDA nor GPU affinity is available.

### Platform Summary

| Aspect | Windows | Linux |
|--------|---------|-------|
| **Display surface** | `VK_EXT_full_screen_exclusive` + Win32 window | `VK_KHR_display` (direct scanout) |
| **GL context** | WGL (shared context via `wglShareLists`) | EGL (surfaceless shared context) |
| **External memory** | `GL_EXT_memory_object_win32` / Win32 HANDLE | `GL_EXT_memory_object_fd` / POSIX fd |
| **External semaphore** | `GL_EXT_semaphore_win32` / Win32 HANDLE | `GL_EXT_semaphore_fd` / POSIX fd |
| **GPU affinity** | `WGL_NV_gpu_affinity` | `EGL_EXT_device_enumeration` |
| **CUDA peer transfer** | Same API (cudaMemcpyPeerAsync) | Same API (cudaMemcpyPeerAsync) |
| **NvAPI (Sync/EDID/HDR)** | Full support | Not available (stubs compiled) |
| **HDR output** | NvAPI UHDA hardware + `VK_EXT_hdr_metadata` | `VK_EXT_hdr_metadata` only |
| **Display blanker** | `display_blanker.exe` companion | Not needed (exclusive display) |
| **Thread naming** | `SetThreadDescription` | `pthread_setname_np` |
| **Process kill (TDR)** | `TerminateProcess` | `kill(SIGKILL)` |

---

## GPU Tier System

The module detects the GPU capability tier at runtime and adapts its output strategy:

| Tier | Detection | Output Method | Features |
|------|-----------|---------------|----------|
| **Pro** | `VK_KHR_display`, or `VK_NV_present_barrier`, or recognized professional GPU name | Direct display mode (if `VK_KHR_display` available) or fullscreen window | VBlank fences, display mode selection, present barriers |
| **Consumer** | None of the above | Borderless fullscreen window with `VkSurfaceKHR` via Win32 (Windows only) | Basic frame delivery, no VBlank timing |

**Pro tier** is detected when any of the following is true:
1. `VK_KHR_display` extension is available and enumerates displays (see caveat below)
2. `VK_NV_present_barrier` extension is available — indicates Quadro Sync hardware
3. The GPU name matches a known professional model (Quadro, RTX A-series, Ada, Tesla)

When `VK_KHR_display` is available **and** the target display is enumerated through it, the module uses direct display mode (bypasses the OS compositor entirely). When the GPU is detected as Pro but no display is enumerated through `VK_KHR_display`, the module falls back to a fullscreen window path while still reporting Pro tier.

#### VK_KHR_display on Windows

On **Windows 11** (Enterprise, Pro for Workstations, or IoT Enterprise editions), `VK_KHR_display` direct scanout is supported through the **Specialized Monitors** / **"Remove display from desktop"** feature. This requires:

1. **Windows 11** (build 22000+) — Windows 10 does NOT support this feature regardless of edition
2. **NVIDIA professional GPU** (Quadro, RTX A-series, RTX Pro) with driver ≥ 535
3. **`configureDriver.exe --set 6`** run as administrator — enables the `ExposeVK_KHR_display` driver registry key
4. **System restart** after running configureDriver — the driver only loads the setting at boot
5. **Display removed from Windows desktop** — either manually via Settings → Display → Advanced → "Remove display from desktop", or programmatically via `ChangeDisplaySettingsExW`

The module automates steps 3–5:
- On first run with a Pro GPU that reports no VK_KHR_display outputs, it launches `configureDriver.exe --set 6` with UAC elevation
- If configureDriver succeeds and VK_KHR_display still reports no displays, it attempts to detach the target display from the Windows desktop
- If detach succeeds but enumeration still fails, it warns that a restart is needed and reattaches the display
- If configureDriver fails or is unavailable, or if the OS is Windows 10, the module falls back to FSE immediately

**On Windows 10**: VK_KHR_display is **not available**. The "Specialized Monitors" / "Remove display from desktop" feature does not exist in Windows 10. All output goes through `VK_EXT_full_screen_exclusive` + Win32 surfaces. The `configureDriver --set 6` command may succeed silently but has no effect.

**Bottom line**: On Windows 10, all output uses the FSE fullscreen window path. On Windows 11 with the correct edition and setup, VK_KHR_display direct scanout is possible but requires one-time admin setup + restart.

#### FSE Fullscreen Window (Windows — primary path)

The fullscreen exclusive window path is the default and most reliable output method on Windows. It:

- Automatically identifies the target display by matching the Vulkan GPU name against Windows display adapter names (`EnumDisplayDevicesW`)
- If the target display was left detached from a previous run (crash recovery), automatically reattaches it using `displayswitch /extend`
- Creates a borderless `WS_EX_TOPMOST` window positioned on the target display
- Uses `VK_EXT_full_screen_exclusive` with `MAILBOX` present mode for low-latency output

The display-targeting logic uses this priority:
1. `<display-name>` config option — match by monitor device string or device ID (substring, case-insensitive)
2. GPU name matching — find displays attached to the configured GPU adapter, pick the Nth output
3. `EnumDisplayMonitors` by index — last resort fallback

#### Linux

On **Linux**, `VK_KHR_display` works as intended: when no compositor (X11/Wayland) manages a display, it appears through `vkGetPhysicalDeviceDisplayPropertiesKHR` and can be acquired for direct scanout via DRM/KMS. This is the **primary output path on Linux** — simpler than Windows because there is no compositor to fight.

**Linux platform stack**:
- **Surface creation**: `VK_KHR_display` direct mode (no windows, no compositor)
- **GL interop**: EGL shared contexts via `EGL_KHR_surfaceless_context`
- **External memory**: `GL_EXT_memory_object_fd` / `VK_KHR_external_memory_fd` (POSIX file descriptors)
- **External semaphores**: `GL_EXT_semaphore_fd` / `VK_KHR_external_semaphore_fd`
- **GPU affinity**: `EGL_EXT_device_enumeration` + `EGL_EXT_platform_device` (select GPU by DRM render node)
- **CUDA peer transfer**: Works identically (same CUDA Runtime API)
- **NvAPI**: Not available — Quadro Sync and EDID injection disabled. HDR via `VK_EXT_hdr_metadata` directly.
- **Display blanker**: Not needed — `VK_KHR_display` owns the display exclusively

**Requirements for Linux VK_KHR_display**:
- NVIDIA proprietary driver ≥ 535 (or Mesa for AMD, though untested)
- The target display must NOT be managed by X11/Wayland (either run headless, or use `xrandr --output DP-X --off` to release it)
- User must have access to `/dev/dri/card*` (typically via `video` group)

**Advantages over Windows**:
- True direct scanout — zero compositor overhead
- No FSE workarounds or DWM fighting
- No display blanker needed
- Simpler code path (no Win32 window creation, no FSE state machine)

#### Actual output path used (Windows)

On **Windows 10** or Windows 11 without Specialized Monitors support, all deployments use the **"Pro (fullscreen)"** or **"Consumer (fullscreen)"** path.

On **Windows 11** with Specialized Monitors enabled and `configureDriver --set 6` applied + restarted, Pro GPUs can use the **"Pro (direct display)"** path — identical to Linux.

The fullscreen window path uses `VK_EXT_full_screen_exclusive` with `MAILBOX` present mode for low-latency presentation. On NVIDIA professional drivers (Quadro, RTX A-series), this achieves near-direct-scanout performance. The module:

1. Detects if the target display was left detached (from a previous crash or VK_KHR_display attempt)
2. Reattaches it automatically via `displayswitch /extend` if needed
3. Creates a GPU-targeted borderless fullscreen window on the correct display
4. If `VK_EXT_full_screen_exclusive` is supported, requests exclusive access for lowest latency

**Consumer tier** creates a `WS_EX_TOPMOST | WS_EX_TOOLWINDOW | WS_EX_APPWINDOW` borderless window covering the target monitor and presents through the standard Vulkan WSI (Window System Integration) path. This still benefits from Vulkan's explicit swapchain control and VSync timing.

#### Actual output path used (Linux)

All Linux deployments use the **"Pro (direct display)"** path:

The module enumerates displays via `vkGetPhysicalDeviceDisplayPropertiesKHR`, selects the target by output index, and creates a `VkSurfaceKHR` directly from the `VkDisplayKHR` handle. No windows or window system is involved. The display is exclusively owned by the Vulkan instance for the duration of output.

This provides true hardware-level scanout with zero compositor overhead, deterministic timing, and full control over display modes (resolution, refresh rate, HDR metadata).

The tier label logged at startup indicates the path used:

```
[vulkan_output] ... initialized. Tier: Pro (direct display)   -- VK_KHR_display path (Linux, or Windows with display released)
[vulkan_output] ... initialized. Tier: Pro (fullscreen)       -- Pro GPU, fullscreen window (normal Windows path)
[vulkan_output] ... initialized. Tier: Consumer (fullscreen)  -- Consumer GPU, fullscreen window (Windows only)
```

---

## Frame Transfer Pipeline

### Buffering

The consumer maintains a thread-safe frame queue between the CasparCG mixer thread and the dedicated present thread:

1. **Mixer thread** calls `send()` — pushes `const_frame` into the queue
2. If the queue exceeds `buffer-depth`, the oldest frame is dropped (logged as `late-frame`)
3. **Present thread** waits for `delay_frames + 1` frames to accumulate before starting presentation
4. Present thread pops the oldest frame, records a Vulkan command buffer, and submits it

### Presentation Delay

The `<delay>` parameter introduces a fixed N-frame latency between when a frame enters the buffer and when it is presented. This compensates for downstream pipeline latency — video scalers, audio de-embedders, LED processors, or SDI-to-IP converters that introduce their own processing delay.

For example, if an LED processor adds 2 frames of latency, setting `<delay>2</delay>` means the Vulkan output will hold frames 2 frames longer, allowing the combined system latency to align with other outputs (like DeckLink) that may have their own delay compensation.

The `<delay-ms>` parameter adds a sub-frame delay in milliseconds on top of `<delay>`. This is useful for fine-tuning A/V sync when video and audio are routed through different consumers (e.g. Vulkan output + PortAudio). The value is clamped to `[0, one frame period]` — use whole-frame `<delay>` for coarser adjustments. See [PORTAUDIO_MODULE.md](PORTAUDIO_MODULE.md) Scenario 9 for a worked example.

### Swapchain

The swapchain is created with:

- `VK_PRESENT_MODE_MAILBOX_KHR` (preferred) — the GPU processes frames immediately without vsync queue blocking. The display still refreshes at vsync, picking the latest submitted frame. Since the present loop self-paces at exactly the target frame rate (via grid-aligned pacing), every frame is displayed. Falls back to `FIFO_RELAXED` or `FIFO` if MAILBOX is unavailable. When present barriers are enabled (`<sync-group>`), FIFO is forced for sync group correctness.
- Image count = `max(minImageCount + 1, buffer_depth)` — ensures enough images for smooth pipelining
- Transfer destination usage — frames are blitted/copied into swapchain images

Surface format selection priority:

| Transfer Mode | Preferred Format | Color Space |
|--------------|-----------------|-------------|
| SDR | `VK_FORMAT_B8G8R8A8_UNORM` | `VK_COLOR_SPACE_SRGB_NONLINEAR_KHR` |
| PQ (HDR10) | `VK_FORMAT_A2B10G10R10_UNORM_PACK32` | `VK_COLOR_SPACE_HDR10_ST2084_EXT` |
| HLG | `VK_FORMAT_R16G16B16A16_SFLOAT` | (first available) |

### Interlaced Content

Field B is skipped — only field A is presented. This matches the behavior of the screen consumer.

---

## Multi-GPU Transfer

When the CasparCG mixer runs on a different GPU than the Vulkan output display, the module automatically detects the GPU mismatch (via LUID comparison) and activates the cross-GPU transfer pipeline.

### Detection

At startup, the module compares the LUID (Locally Unique Identifier) of:
1. The **mixer device** GPU (where the OGL or Vulkan mixer runs)
2. The **Vulkan output device** GPU (where the display is connected)

If the LUIDs differ, the cross-GPU path is activated. The LUID is queried via `VkPhysicalDeviceIDProperties` on the Vulkan side and via GPU affinity context LUID (WGL on Windows, EGL device on Linux) or `VkPhysicalDeviceIDProperties` on the mixer side (depending on whether the OGL or Vulkan mixer is active).

### Transfer Hierarchy

The module tries paths in order of performance:

| Priority | Path | Bandwidth | CPU Load | Requirements |
|----------|------|-----------|----------|--------------|
| 1 | CUDA peer DMA | ~15 GB/s (PCIe 3.0 x16) | Zero | CUDA Toolkit, both NVIDIA GPUs |
| 2 | PBO upload | ~6 GB/s | Moderate | GPU affinity (WGL or EGL) |
| 3 | CPU staging | ~3 GB/s | High | None (always available) |

### CUDA Peer DMA (Preferred)

When the CUDA Toolkit is available at build time (`CASPAR_CUDA_PEER_ENABLED`), the module:

1. Queries `cudaGLGetDevices()` on each GPU's GL context to discover CUDA device indices
2. Enables `cudaDeviceEnablePeerAccess()` for direct PCIe DMA (or NVLink if available)
3. Queries `cudaDeviceGetP2PAttribute(cudaDevP2PAttrNativeAtomicSupported)` to detect NVLink vs PCIe P2P
4. Allocates staging buffers on both GPUs
5. Creates a dedicated `peer_stream_` with `src_ready_event_` and `peer_event_` for fully async GPU-side synchronization

Per-frame transfer (all async, zero CPU sync until final GL upload):
```
Phase 1 (OGL thread, GPU A): cudaStreamWaitEvent(peer_event)  [wait for previous DMA to finish reading src]
                              cudaGraphicsMapResources → cudaMemcpy2DFromArray → staging_A
                              cudaEventRecord(src_ready_event)  [signal staging_A is ready]
Phase 2 (any thread):         cudaStreamWaitEvent(src_ready_event)  [GPU-side wait for Phase 1]
                              cudaMemcpyPeerAsync(staging_B, dev_B, staging_A, dev_A, peer_stream)
                              cudaEventRecord(peer_event)  [signal DMA complete]
Phase 3 (affinity thread, GPU B): cudaStreamWaitEvent(peer_event)  [GPU-side wait for Phase 2]
                                  cudaMemcpy → PBO → glTexSubImage2D → shared_pool
```

Even without NVLink or direct P2P support, `cudaMemcpyPeerAsync` still works — the driver routes through system RAM transparently, but uses the GPU's DMA engines rather than CPU memcpy, reducing CPU overhead.

### PBO Upload Fallback

When CUDA is not available (no toolkit at build time, or CUDA initialization fails at runtime):

1. The mixer's CPU-side pixel buffer (`frame.image_data(0)`) is used
2. A double-buffered PBO (Pixel Buffer Object) on the affinity context streams pixels to GPU B
3. The uploaded texture is blitted into the shared interop pool

This path still avoids blocking the mixer thread. The double-buffered PBO overlaps the upload of frame N with the presentation of frame N-1.

### GPU Affinity Context

The affinity context creates an OpenGL context pinned to a specific GPU for cross-GPU scenarios:

**Windows** — uses NVIDIA `WGL_NV_gpu_affinity` extension:

1. `wglEnumGpusNV()` enumerates physical GPUs
2. `wglCreateAffinityDCNV()` creates a device context bound to the target GPU
3. A full OpenGL 4.5 context is created on this DC with its own dedicated thread
4. The context's LUID is verified to match the Vulkan device

**Linux** — uses `EGL_EXT_device_enumeration` + `EGL_EXT_platform_device`:

1. `eglQueryDevicesEXT()` enumerates EGL devices (maps to DRM render nodes)
2. `eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT, device)` creates a display for the target GPU
3. A surfaceless OpenGL 4.5 core context is created via `eglCreateContext()`
4. The DRM device path (e.g., `/dev/dri/renderD129`) identifies which GPU is used

This context provides the GL environment needed for both the CUDA interop (GPU B side) and the shared_texture_pool GL operations.

### Log Output

```
[vulkan_output] GPU LUID mismatch — mixer on GPU 0, output on GPU 1. Activating cross-GPU transfer.
[vulkan_output] Cross-GPU interop enabled via affinity bridge (GPU 1) (16-bit for HDR).
[vulkan_output] CUDA peer DMA enabled (device 0 → device 1)
[vulkan_output] NVLink detected between device 0 and device 1 (perf_rank=1)
[vulkan_output] NVLink provides ~600 GB/s bidirectional bandwidth
```

Without NVLink but with P2P:
```
[vulkan_output] PCIe P2P direct access between device 0 and device 1 (perf_rank=0, no NVLink)
```

Without any P2P (different architectures):
```
[vulkan_output] No direct P2P — peer copy will stage through system RAM (still faster than CPU memcpy)
```

If CUDA peer is unavailable:
```
[vulkan_output] CUDA peer transfer unavailable: cudaGLGetDevices failed — using PBO upload fallback.
```

### GPU Topology Detection

At module initialization, `vulkan_device::log_gpu_topology()` performs a one-time scan of the system's GPU interconnect topology:

1. **Vulkan device group enumeration** — creates a temporary `VkInstance` and calls `vkEnumeratePhysicalDeviceGroups()` to detect NVLink/SLI bridges (multi-GPU device groups with `subsetAllocation = true`)
2. **CUDA P2P attribute probing** — for every GPU pair, queries `cudaDeviceGetP2PAttribute()` to determine:
   - Whether direct P2P access is possible (`cudaDevP2PAttrAccessSupported`)
   - Performance rank (`cudaDevP2PAttrPerformanceRank`) — higher means faster (NVLink > PCIe)
   - Native atomic support (`cudaDevP2PAttrNativeAtomicSupported`) — indicates NVLink

Example startup output:
```
[vulkan_output] GPU topology: 4 device group(s)
[vulkan_output]   Group 0: NVIDIA RTX A4000 (single GPU)
[vulkan_output]   Group 1: NVIDIA RTX A4000 (single GPU)
[vulkan_output]   Group 2: Quadro P4000 (single GPU)
[vulkan_output]   Group 3: Quadro P4000 (single GPU)
[vulkan_output]   No NVLink bridge detected. Cross-GPU transfers use CUDA peer DMA (PCIe).
[vulkan_output] CUDA P2P topology (2 devices):
[vulkan_output]   NVIDIA RTX A4000 <-> Quadro P4000: staged (system RAM) (perf_rank=0)
```

With NVLink:
```
[vulkan_output] GPU topology: 1 device group(s)
[vulkan_output]   Group 0: NVIDIA RTX A6000 x2 (NVLink/SLI bridge, subsetAllocation)
[vulkan_output] CUDA P2P topology (2 devices):
[vulkan_output]   NVIDIA RTX A6000 <-> NVIDIA RTX A6000: direct P2P (perf_rank=1, NVLink)
```

---

## HDR Output

### Transfer Functions

| Setting | Standard | Typical Use |
|---------|----------|-------------|
| `sdr` | BT.709 gamma | Standard broadcast, LED walls |
| `pq` | SMPTE ST 2084 (Perceptual Quantizer) | HDR10, Dolby Vision base layer, HDR monitors |
| `hlg` | ARIB STD-B67 (Hybrid Log-Gamma) | Live broadcast HDR, backward-compatible with SDR displays |

### HDR Metadata

When a non-SDR transfer is configured, the module sets SMPTE ST 2086 mastering display metadata via `VK_EXT_hdr_metadata`:

- **Display primaries**: BT.2020 (R: 0.708/0.292, G: 0.170/0.797, B: 0.131/0.046)
- **White point**: D65 (0.3127/0.3290)
- **MaxCLL**: Maximum Content Light Level (nits) — configurable via `<max-cll>`
- **MaxFALL**: Maximum Frame-Average Light Level (nits) — configurable via `<max-fall>`
- **MinLuminance**: Fixed at 0.001 cd/m²

### EDID Auto-Detection

When `<edid-auto-hdr>true</edid-auto-hdr>` is set, the module reads the connected display's EDID via NvAPI at startup. If the display reports HDR capability (HDR Static Metadata Data Block in CTA-861 extension), the module:

1. Automatically switches from SDR to PQ transfer
2. Reads the display's reported maximum luminance and uses it as `MaxCLL`
3. Re-applies HDR metadata with the display-specific values
4. Logs the detected capabilities:

```
[vulkan_output] EDID auto-detected HDR (PQ) display: DELL UP2718Q MaxCLL=1000 cd/m²
```

This is useful in multi-display setups where some outputs are HDR-capable and others are not — the module adapts without manual configuration.

### Hardware HDR Acceleration (NvAPI Display Engine)

On NVIDIA GPUs, when HDR output is configured (PQ or HLG), the module attempts to enable hardware-accelerated HDR signaling via `NvAPI_Disp_HdrColorControl`. This uses the GPU's **display engine** — dedicated scanout-stage hardware — to perform PQ EOTF encoding and BT.2020 gamut mapping at scanout, allowing the mixer to output in a linear or scRGB working space.

**How it works:**

1. The module resolves the output index to an NvAPI display ID
2. Queries `NvAPI_Disp_GetHdrCapabilities` to verify ST2084 PQ support
3. Calls `NvAPI_Disp_HdrColorControl` with `NV_HDR_MODE_UHDA`
4. The Vulkan swapchain receives linear scRGB FP16 values (RGB(1,1,1) = 80 nits)
5. The display engine hardware converts to HDR10 (PQ + BT.2020 + 10-bit) at scanout

**Benefits:**
- Zero GPU shader cost — the conversion runs in dedicated display engine hardware
- No additional frame latency — the transform happens at the scanout stage

**Activation:**
- Automatic when `<transfer>pq</transfer>` or `<transfer>hlg</transfer>` is set and the display supports ST2084
- Also activates via `<edid-auto-hdr>` when the display reports HDR capability
- Falls back gracefully if the display or driver doesn't support hardware HDR
- Does not affect Quadro Sync or present barrier synchronization (operates per-output, independent of framelock)

**Log output when active:**
```
[vulkan_output] Display HDR caps: ST2084=1 EDR=1 maxLum=1000 minLum=50
[vulkan_output] Hardware HDR enabled (NvAPI UHDA mode). Display engine performs PQ + BT.2020 conversion. MaxCLL=1000 MaxFALL=400
[vulkan_output] Hardware HDR active — display engine handles PQ + BT.2020.
```

The mastering display metadata (ST2086) sent to the display uses BT.2020 primaries, D65 white point, and the configured MaxCLL/MaxFALL values. On shutdown, the module restores SDR mode automatically.

---

## DeckLink CUDA-Vulkan Interop

When the Vulkan mixer is active (`<accelerator>vulkan</accelerator>`), the DeckLink consumer uses `cuda_vk_strategy` to receive rendered frames directly from the Vulkan pipeline via GPU-side synchronization — eliminating the CPU fence wait that previously dominated the DeckLink output path.

### Architecture

```
Vulkan Mixer (image_kernel)
    │
    ├── render() produces VkImage + timeline semaphore
    │     └── vkQueueSubmit signals VkSemaphore (timeline, value N)
    │              └── exports platform handle via VK_KHR_external_semaphore_{win32,fd}
    │
    ├── core::texture interface propagates handle + value
    │     └── frame.texture()->render_semaphore_handle()
    │     └── frame.texture()->render_semaphore_value()
    │
    └── cuda_vk_strategy (DeckLink consumer)
          ├── cudaImportExternalSemaphore (platform handle → CUDA external semaphore)
          ├── cudaWaitExternalSemaphoresAsync (GPU-side wait for VK render completion)
          ├── CUDA BGRA→v210 conversion kernel
          ├── cudaMemcpyAsync (device → host, double-buffered)
          └── DeckLink ScheduleVideoFrame
```

### Key Design Points

- **GPU-side semaphore wait**: `cudaWaitExternalSemaphoresAsync` enqueues a GPU-side wait on the CUDA stream — no CPU blocking. The CUDA kernel for v210 conversion starts immediately after the Vulkan render completes on the GPU timeline.
- **Double-buffered output**: Two device buffers (`d_v210_[0]`, `d_v210_[1]`) and per-buffer `cudaEvent_t` allow pipelining: frame N's D2H transfer overlaps with frame N+1's GPU conversion.
- **Semaphore handle caching**: A multi-slot cache (`MAX_CACHED_SEMS = 8`) maps platform handles to imported CUDA external semaphores, avoiding repeated `cudaImportExternalSemaphore` calls. The Vulkan mixer rotates through 3-4 `frame_data` slots, each with a unique semaphore handle.
- **Graceful fallback**: If the frame's texture doesn't provide a semaphore handle (e.g., OGL mixer), the strategy falls back to CPU pixel readback.

### Performance

| Metric | Before (CPU fence) | After (GPU semaphore) |
|--------|--------------------|-----------------------|
| Fence/sync wait | 22ms (`glClientWaitSync`) | 0.06ms (`cudaWaitExternalSemaphoresAsync`) |
| CPU thread blocked | Yes (22ms) | No |
| Total frame time | ~30ms | ~28ms (GPU throughput limited) |

The remaining 28ms is GPU execution time (VK render + CUDA v210 conversion), not CPU blocking. The CPU is free to service other consumers during this time.

### Files

- `src/modules/decklink/consumer/cuda_vk_strategy.cpp` — DeckLink CUDA-VK output strategy
- `src/accelerator/vulkan/image/image_kernel.cpp` — timeline semaphore creation and signal
- `src/core/frame/frame.h` — `core::texture` base class with virtual semaphore methods

---

## Color Space Conversion

Color space conversion is handled by the **mixer** (fragment shader), **not** by the Vulkan output consumer. The mixer performs EOTF linearization, gamut matrix multiplication, and OETF encoding during compositing — so frames arrive at the Vulkan output already in the channel's target color space and transfer function.

> **Important**: The `<gamut>` and `<eotf>` settings in the `<vulkan-output>` config block are currently **non-functional** (parsed but ignored). All color conversion is controlled by the **channel-level** `<color-space>` and `<color-transfer>` settings, which define the mixer's output color space. The `<transfer>` setting (sdr/pq/hlg) remains functional — it controls hardware HDR signaling, VK_EXT_hdr_metadata, NvAPI UHDA, and 16-bit frame cache selection.

The `color_convert_pipeline` compute shader is retained in the codebase but is currently **disabled** (bypassed with zero GPU cost). It may be re-enabled in the future for per-consumer display adaptation when the output display differs from the channel's working color space.

### Architecture (current)

```
Source (BT.709 sRGB, BT.2020 PQ, etc.)
    │
    ▼ Mixer fragment shader:
    │   1. Apply EOTF (linearize source)
    │   2. 3×3 gamut matrix (source → channel target)
    │   3. Apply OETF (encode to channel transfer)
    │
    ▼ Composited frame in channel target gamut/transfer
    │
    ▼ blit to swapchain (Vulkan output)
    │
    ▼ vkQueuePresentKHR
```

When hardware HDR acceleration (NvAPI UHDA) is active, the display engine may perform additional PQ encoding at the scanout stage.

### Output Gamuts (reserved — currently no-op)

These values are parsed but have **no effect** on pixel output. They are reserved for future per-consumer display adaptation. Color gamut is currently controlled by the channel's `<color-space>` setting.

| Config Value | Standard | Typical Use |
|-------------|----------|-------------|
| `bt709` | ITU-R BT.709 / sRGB | Default — no conversion, direct pass-through |
| `bt2020` | ITU-R BT.2020 | HDR10, broadcast HDR, wide-gamut displays |
| `p3-d65` | Display P3 (D65 white) | Apple displays, wide-gamut monitors, HDR grading |
| `p3-dci` | DCI-P3 (DCI white) | Digital cinema projection (gamma 2.6) |
| `adobe-rgb` | Adobe RGB (1998) | Photography, print proofing |

Aliases: `p3` and `display-p3` → `p3-d65`; `dci-p3` → `p3-dci`; `2020` → `bt2020`; `adobergb` → `adobe-rgb`.

### Transfer Functions / EOTF (reserved — currently no-op)

These values are parsed but have **no effect** on pixel output. They are reserved for future per-consumer display adaptation. Transfer function is currently controlled by the channel's `<color-transfer>` setting.

| Config Value | Standard | Typical Use |
|-------------|----------|-------------|
| `srgb` | IEC 61966-2-1 (~gamma 2.2) | Default — matches mixer working space |
| `linear` | 1:1 (no curve) | Compositing previews, light-linear workflows |
| `pq` / `st2084` | SMPTE ST 2084 | HDR10, Dolby Vision base, broadcast HDR mastering |
| `hlg` | ARIB STD-B67 | Live broadcast HDR, backward-compatible |
| `gamma24` / `2.4` | Pure gamma 2.4 | EBU broadcast reference monitors |
| `gamma26` / `2.6` | Pure gamma 2.6 | DCI cinema projection |

### Automatic Inference (reserved — currently no-op)

When `<gamut>` or `<eotf>` are not explicitly set, the module infers defaults from the `<transfer>` setting. These inferred values are stored but currently have no effect on pixel output:

| `<transfer>` | Inferred gamut | Inferred EOTF |
|-------------|---------------|----------------|
| `sdr` | bt709 | srgb |
| `pq` | bt2020 | pq |
| `hlg` | bt2020 | hlg |

Note: `<transfer>` itself IS functional — it controls hardware HDR signaling and metadata (see [HDR Output](#hdr-output)). Only the gamut/eotf values derived from it are currently unused.

### Configuration

To get HDR10 (BT.2020 + PQ) output, configure at the **channel level**:

```xml
<channel>
    <video-mode>2160p5000</video-mode>
    <color-space>bt2020</color-space>
    <color-transfer>pq</color-transfer>
    <color-depth>16</color-depth>
    <consumers>
        <vulkan-output>
            <gpu>0</gpu>
            <device>1</device>
            <transfer>pq</transfer>         <!-- Enables HW HDR signaling -->
            <hdr-metadata>
                <max-cll>1000</max-cll>
                <max-fall>400</max-fall>
            </hdr-metadata>
        </vulkan-output>
    </consumers>
</channel>
```

The channel's `<color-space>` and `<color-transfer>` determine the mixer's output pixel values. The consumer's `<transfer>` enables hardware HDR metadata signaling to the display.

### Performance (reserved pipeline — currently disabled)

When the per-consumer color conversion compute shader is re-enabled in the future, expected performance:
- **1080p60**: < 0.1 ms per frame (negligible)
- **2160p60**: ~0.3 ms per frame
- **4320p60 (8K)**: ~1.2 ms per frame

The intermediate image is RGBA16F (16-bit float), providing sufficient precision for all gamut conversions and transfer functions without banding artifacts.

---

## Synchronization

### Grid-Aligned Frame Pacing

Each vulkan output's present thread independently computes the same absolute deadline from `steady_clock`'s epoch, so all outputs at the same frame rate wake and present at the same wall-clock instant — without any inter-output communication or coordinator.

```
interval  = 1s / fps                             (e.g. 40ms at 25fps, 16.67ms at 60fps)
now_ns    = steady_clock::now()
deadline  = ((now_ns / interval) + 1) * interval  (next grid tick)
sleep_until(deadline)
```

This snaps every output to the same absolute time grid. For example, at 25fps the grid ticks are at 40ms, 80ms, 120ms, 160ms... from the `steady_clock` epoch. Every output running at the same fps independently computes the same next tick and sleeps to it.

**Why they stay in sync:**
- **Same fps → same grid** — all outputs at the same frame rate hit the same grid ticks
- **No drift** — the grid is anchored to the `steady_clock` epoch, not relative to the previous frame
- **Independent threads** — each output has its own present thread, so one slow output can't block another
- **Buffer queue absorbs jitter** — the depth-3 queue tolerates up to 3 frames of `send()` timing variation
- **GPU semaphore sync** — `vkQueueSubmit` waits on the GL interop semaphore, ensuring the GPU won't start the Vulkan blit until the GL blit has finished (driver-level, no CPU involvement)
- **Shared VkDevice** — multiple consumers on the same GPU share one `VkDevice` via `vk_device_manager`, which is a prerequisite for present barriers and eliminates redundant device creation

With MAILBOX present mode, `vkQueuePresentKHR` returns immediately (no vsync blocking), so the grid-aligned sleep is the sole pacing mechanism. With FIFO mode (used when present barriers are enabled), `vkWaitForFences` provides hardware-locked pacing and the grid sleep is effectively a no-op.

### Queue Thread Safety

When multiple consumers share a `VkDevice` (via `vk_device_manager`), all Vulkan queue operations must be externally synchronized per the Vulkan specification. The `vulkan_device` exposes a `queue_mutex()` that consumers lock around `vkQueueSubmit` and `vkQueuePresentKHR` calls.

Additionally, `vkDeviceWaitIdle` (used during swapchain recreation and shutdown) is also wrapped in the queue mutex to prevent concurrent queue submissions.

### Vulkan Present Barriers (`VK_NV_present_barrier`)

For multi-output setups where multiple Vulkan consumers must present at exactly the same time (e.g., a video wall driven by one GPU), the `<sync-group>` parameter enables Vulkan present barriers.

All swapchains with the same non-zero `sync-group` value are frame-locked at the driver level. The NVIDIA driver holds all `vkQueuePresentKHR` calls until every swapchain in the group has submitted, then releases them simultaneously on the same VBlank.

```xml
<!-- GPU 0 drives two outputs, frame-locked together -->
<channel>
    <consumers>
        <vulkan-output>
            <gpu>0</gpu>
            <device>1</device>
            <sync-group>1</sync-group>
        </vulkan-output>
    </consumers>
</channel>
<channel>
    <consumers>
        <vulkan-output>
            <gpu>0</gpu>
            <device>2</device>
            <sync-group>1</sync-group>
        </vulkan-output>
    </consumers>
</channel>
```

Requires:
- `VK_NV_present_barrier` device extension (NVIDIA driver 525+ with Quadro/RTX Pro GPU)
- All outputs in the same sync group must be on the same GPU
- All outputs in the same sync group must share the same `VkDevice` (handled automatically by `vk_device_manager`)

### Quadro Sync II (NvAPI GSync)

For multi-GPU framelock — where separate physical GPUs must have their VSync signals locked together — the module integrates with NVIDIA Quadro Sync II boards via NvAPI.

```xml
<vulkan-output>
    <gsync>
        <enabled>true</enabled>
        <master>true</master>
        <reference>external</reference>
    </gsync>
</vulkan-output>
```

| Parameter | Description |
|-----------|-------------|
| `enabled` | Enable GSync framelock for this output |
| `master` | Designate this output as the timing master. All other outputs in the sync group will slave to this output's VSync. Exactly one output should be master. |
| `reference` | `internal` = use the GPU's own VSync as the timing reference. `external` = use the BNC house sync input on the Quadro Sync II card (genlock). |

The module monitors sync status and exposes it via OSC:

```
vulkan-output/gsync/available  = true
vulkan-output/gsync/synced     = true
vulkan-output/gsync/house-sync = true
vulkan-output/gsync/role       = master
```

### Synchronization Hierarchy

For maximum timing accuracy, combine both mechanisms:

```
                 House Sync (BNC)
                       │
                 Quadro Sync II
                       │
              ┌────────┴────────┐
              │                 │
           GPU 0             GPU 1
           (master)          (slave)
              │                 │
        ┌─────┴─────┐    ┌─────┴─────┐
     Output 1    Output 2  Output 3  Output 4
     sync-group=1          sync-group=2
```

- **Quadro Sync II** locks GPU 0 and GPU 1 VSync signals together (NvAPI)
- **Present barriers** lock the per-GPU outputs together (`sync-group`)
- **House sync** provides external genlock to the Quadro Sync II card

---

## Subregion Output

The subregion feature allows a portion of the channel's frame to be mapped to a portion of the physical display. This enables multi-display tiling from a single wide channel.

```xml
<subregion>
    <src-x>0</src-x>        <!-- X offset into the source channel frame -->
    <src-y>0</src-y>        <!-- Y offset into the source channel frame -->
    <dest-x>0</dest-x>      <!-- X offset on the physical output -->
    <dest-y>0</dest-y>      <!-- Y offset on the physical output -->
    <width>1920</width>     <!-- Region width (0 = full source width) -->
    <height>1080</height>   <!-- Region height (0 = full source height) -->
</subregion>
```

### Example: 3840×1080 Channel Across Two 1920×1080 Outputs

```xml
<!-- Left half -->
<vulkan-output>
    <gpu>0</gpu>
    <device>1</device>
    <subregion>
        <src-x>0</src-x>
        <src-y>0</src-y>
        <width>1920</width>
        <height>1080</height>
    </subregion>
    <sync-group>1</sync-group>
</vulkan-output>

<!-- Right half -->
<vulkan-output>
    <gpu>0</gpu>
    <device>2</device>
    <subregion>
        <src-x>1920</src-x>
        <src-y>0</src-y>
        <width>1920</width>
        <height>1080</height>
    </subregion>
    <sync-group>1</sync-group>
</vulkan-output>
```

The blit operation uses `VK_FILTER_LINEAR` for scaling when the source region size differs from the output display resolution.

---

## Display Hot-Plug

The consumer handles display disconnection and reconnection gracefully:

| Event | Response |
|-------|----------|
| `VK_ERROR_SURFACE_LOST_KHR` | Marks display as lost, logs warning |
| `VK_ERROR_OUT_OF_DATE_KHR` | Immediate swapchain recreation |
| `VK_SUBOPTIMAL_KHR` | Swapchain recreation (resolution may have changed) |
| Display lost + `on-disconnect=retry` | Retries surface validation every 50 frames (~1 second at 50fps) |
| Display lost + `on-disconnect=hold` | Holds the last presented frame, stops attempting presents |
| Display lost + `on-disconnect=black` | Clears output to black |
| Display reconnected | Recreates swapchain, re-applies HDR metadata, resumes presentation |

The `display-lost` state is exposed via OSC at `vulkan-output/display-lost`.

---

## Display Blanker

> **Windows only.** On Linux, `VK_KHR_display` acquires the display exclusively — no desktop or compositor content is ever visible on the output. A blanker is not needed.

### Problem

When CasparCG is not running — during restart, crash, or maintenance — the Windows desktop becomes visible on GPU output connectors. In broadcast environments, this is unacceptable: desktop icons, taskbars, and notification popups should never reach program output.

### Solution: `display_blanker.exe`

A standalone companion tool that keeps selected displays black independently of CasparCG. It ships alongside `casparcg.exe` and provides two complementary blanking mechanisms:

| Mechanism | Tier | How it works | Survives crash of... |
|-----------|------|-------------|---------------------|
| **Black TOPMOST windows** | All GPUs | Creates `WS_EX_TOPMOST` black popup windows covering the output monitors. Sits between the desktop and any fullscreen application. | CasparCG (blanker stays running) |
| **NvAPI dedicated display** | Pro (Quadro/RTX A) | Acquires displays via `NvAPI_DISP_AcquireDedicatedDisplay`. The NVIDIA driver detaches the display from DWM at the driver level — no desktop is composited to that output. | CasparCG AND the blanker (driver holds the state) |

### Usage

```
display_blanker.exe                  Interactive — opens config window
display_blanker.exe --autostart      Load saved config, minimize to tray
display_blanker.exe --match MTT      Blank monitors matching "MTT" (VDD)
display_blanker.exe --all            Blank all non-primary monitors
```

### Configuration GUI

Double-click the tray icon or right-click → "Configure..." to open the settings window:

- **Monitor list** with checkboxes — each entry shows: display name, adapter, resolution, and position
- **★ Primary indicator** — the Windows main display is marked with a star and `[Windows main display]`
- **Dedicated displays** section (Pro GPUs only) — lists displays configured as "dedicated" in NVIDIA Control Panel
- **Enable blanking** toggle
- **Start with Windows** checkbox (HKCU registry)
- **15-second confirmation countdown** — after applying changes, a timer dialog appears on the primary monitor. If not confirmed (e.g., because you blanked the control monitor), changes automatically revert

### Confirmation Safety

Like Windows display settings, after applying any change the blanker shows:

```
┌──────────────────────────────────────────┐
│      Confirm Display Settings            │
├──────────────────────────────────────────┤
│                                          │
│  Do you want to keep these settings?     │
│                                          │
│  Reverting in 12 seconds...              │
│                                          │
│  [Keep Changes]        [Revert]          │
└──────────────────────────────────────────┘
```

If you accidentally blank the monitor showing this dialog, you can't click "Keep Changes", so the countdown expires and everything reverts. This prevents lockouts.

### Integration with CasparCG

Add `<display-blanker>true</display-blanker>` to a vulkan-output consumer to auto-launch the blanker when CasparCG starts:

```xml
<vulkan-output>
    <gpu>0</gpu>
    <device>1</device>
    <display-blanker>true</display-blanker>
</vulkan-output>
```

The blanker is launched with `--match` using the `<display-name>` filter (if set), or `--all` otherwise. The blanker's single-instance mutex prevents duplicate launches — if it's already running, the second launch exits silently.

### Dedicated Displays (NvAPI)

For Pro-tier NVIDIA GPUs, the blanker can acquire displays via the NVIDIA dedicated display API. This provides the strongest guarantee:

1. In **NVIDIA Control Panel**, configure target outputs as "Use NVIDIA GPU exclusively" (dedicated mode)
2. In the blanker config window, check the dedicated displays you want to acquire
3. The driver detaches those displays from DWM entirely — no Windows compositing occurs on those outputs

**Key properties:**
- The display stays black even if the blanker process exits (the driver holds the "dedicated" state)
- Only an explicit release or reboot returns the display to the desktop
- CasparCG's `VK_KHR_display` renders directly on the dedicated display — the blanker's acquisition doesn't interfere with Vulkan direct display mode
- On clean exit, the blanker releases the acquisition; on crash, the driver keeps it acquired (output stays black)

### INI File

Settings are persisted to `display_blanker.ini` next to the executable:

```ini
[General]
Enabled=1

[Monitors]
Count=2
Output0=\\.\DISPLAY5
Output1=\\.\DISPLAY6

[DedicatedDisplays]
Count=1
DisplayId0=2147880067
```

---

## VK_KHR_display on Windows (Direct Display)

> **Windows 11 only.** On Linux, `VK_KHR_display` is always available with the proprietary NVIDIA driver — no special configuration is needed.

### Background

On Windows, the NVIDIA driver does **not** export `VK_KHR_display` by default. The Desktop Window Manager (DWM) owns all display outputs, so `vkGetPhysicalDeviceDisplayPropertiesKHR` returns zero displays unless the driver is explicitly configured to expose them.

Two steps are required for true Direct Display output:

1. **Export the extension** — Run `configureDriver.exe --option 6` (admin, one-time)
2. **Remove displays from the desktop** — Detach the target monitors from the Windows desktop

Both steps are **handled automatically** by CasparCG VP when a `<vulkan-output>` consumer is configured on a Pro GPU. On shutdown, detached displays are automatically returned to the Windows desktop.

Once both steps are done, the vulkan_output consumer uses `VK_KHR_display` for exclusive scanout — no DWM compositing, no FSE window, minimal latency.

### Supported GPUs

`configureDriver.exe` supports NVIDIA professional GPUs:

- **Ada generation**: RTX 6000, RTX 5880, RTX 5000, RTX 4500, RTX 4000, RTX 4000 SFF
- **Ampere**: RTX A6000, RTX A5500, RTX A5000, RTX A4500, RTX A4000H, RTX A4000
- **Turing**: Quadro RTX 8000, RTX 6000, RTX 5000, RTX 4000
- **Volta/Pascal**: Quadro GV100, GP100, P6000, P5000, **P4000**
- **Maxwell**: Quadro M6000 24GB, M6000, M5000, **M4000**

### Auto-Configuration (Option B)

CasparCG VP automatically handles both prerequisites when a `<vulkan-output>` consumer initializes on a Pro GPU:

**Step 1: Extension Export (module init)**

If `configureDriver.exe` is found next to `casparcg.exe` and Pro GPUs report no `VK_KHR_display` displays, it is invoked automatically:

```
[info] [vulkan_output] Pro GPU detected but VK_KHR_display shows no displays.
       Running configureDriver.exe --option 6 to export VK_KHR_display...
[info] [vulkan_output] configureDriver.exe completed successfully.
       NOTE: A system restart may be required for the change to take effect.
```

**Step 2: Display Detach (per-consumer)**

When a vulkan_output consumer finds its target display still attached to the Windows desktop, it uses the Win32 CCD API (`SetDisplayConfig`) to programmatically remove it:

```
[info] [vulkan_output] Pro GPU but VK_KHR_display reports no displays.
       Attempting to detach output 1 from Windows desktop...
[info] [vulkan_output] Successfully detached \\.\DISPLAY3 from Windows desktop.
       Display is now available for VK_KHR_display.
```

On shutdown, any display detached by CasparCG is **automatically reattached** to the Windows desktop:

```
[info] [vulkan_output] Reattached \\.\DISPLAY3 to Windows desktop.
```

**Requirements:**
- Place `configureDriver.exe` in the same directory as `casparcg.exe`
- CasparCG must run with **administrator** privileges (the driver setting is a protected registry key)
- A **system restart** may be needed after the first run of `configureDriver.exe` for the driver to pick up the new setting
- Windows 11 Enterprise, Pro for Workstations, or IoT Enterprise (required for the "Remove display from desktop" API)

If CasparCG is not running as admin, a log hint is emitted:

```
[warning] [vulkan_output] configureDriver.exe requires administrator privileges.
          Run CasparCG as admin, or execute manually: configureDriver.exe --option 6
```

### Manual Setup

If auto-configuration is not desired or CasparCG cannot run as admin:

1. Download `configureDriver.exe` from https://www.nvidia.com/en-us/drivers/driver-utility/
2. Open an elevated Command Prompt
3. Run: `configureDriver.exe --option 6`
4. Restart the system
5. (Optional) In Windows Settings, manually remove target output displays from the desktop — CasparCG will do this automatically if it has permissions, but manual removal persists across reboots

### Fallback Behavior

If `VK_KHR_display` is not available (extension not exported, display not removed from desktop, or non-Pro GPU), the vulkan_output consumer falls back to:

1. **Fullscreen Exclusive (FSE) window** — Borderless TOPMOST window with `VK_EXT_full_screen_exclusive` for DWM bypass
2. **Display blanker** — Optional companion process for desktop hiding

The FSE path works on all GPUs without any driver configuration.

### Reference

- NVIDIA sample: https://github.com/nvpro-samples/gl_render_vk_ddisplay
- Microsoft docs: https://learn.microsoft.com/en-us/windows-hardware/drivers/display/specialized-monitors

---

## AMCP Commands

### INFO VULKAN_OUTPUT

Enumerates all detected Vulkan display outputs across all GPUs.

```
>> INFO VULKAN_OUTPUT
<< 201 INFO VULKAN_OUTPUT OK
<< <vulkan-outputs>
<<   <output>
<<     <gpu-index>0</gpu-index>
<<     <output-index>1</output-index>
<<     <gpu-name>NVIDIA RTX A4000</gpu-name>
<<     <display-name>DP-1</display-name>
<<     <width>3840</width>
<<     <height>2160</height>
<<     <tier>pro</tier>
<<   </output>
<< </vulkan-outputs>
```

When NvAPI is available, the output also includes EDID information (manufacturer, model, HDR support, 10-bit support, maximum luminance).

### ADD (AMCP)

```
ADD 1 VULKAN_OUTPUT [output_index] [gpu_index] [MODE video_mode] [DELAY frames] [DELAY_MS ms]
```

Examples:
```
ADD 1 VULKAN_OUTPUT 1          -- Output 1 on GPU 0
ADD 1 VULKAN_OUTPUT 2 1        -- Output 2 on GPU 1
ADD 1 VULKAN_OUTPUT 1 0 DELAY 2  -- Output 1, 2-frame presentation delay
ADD 1 VULKAN_OUTPUT 1 0 DELAY 3 DELAY_MS 5  -- 3 frames + 5ms sub-frame delay
```

### CALL (Runtime)

```
CALL 1-500 IDENTIFY    -- Flash a color overlay for 3 seconds to identify the output
```

Each output displays a distinct color (Blue, Green, Red, Cyan, Magenta, Yellow) based on its output index for easy identification in multi-display setups.

---

## Configuration Reference

All options for the `<vulkan-output>` consumer block in `casparcg.config`:

| Element | Type | Default | Description |
|---------|------|---------|-------------|
| `<gpu>` | int | `0` | Physical GPU index (0-based) |
| `<device>` | int | `1` | Display output index (1-based) |
| `<buffer-depth>` | int | `3` | Pre-scheduled swapchain frame count |
| `<delay>` | int | `0` | Presentation delay in frames |
| `<delay-ms>` | double | `0.0` | Sub-frame delay in milliseconds (added on top of `<delay>`) |
| `<video-mode>` | string | *(channel)* | Explicit output video mode |
| `<identify-on-start>` | bool | `false` | Flash identification overlay on startup |
| `<on-disconnect>` | enum | `retry` | `hold` \| `black` \| `retry` |
| `<transfer>` | enum | `sdr` | `sdr` \| `pq` \| `hlg` — controls HW HDR signaling, metadata, and 16-bit frame cache |
| `<gamut>` | enum | *(auto)* | **Reserved (currently no-op)**. Parsed but unused — color conversion is handled by the mixer at channel level |
| `<eotf>` | enum | *(auto)* | **Reserved (currently no-op)**. Parsed but unused — see `<gamut>` above |
| `<edid-auto-hdr>` | bool | `false` | Auto-detect HDR from display EDID |
| `<edid-emulation>` | bool | `false` | Inject synthetic EDID on unconnected outputs (admin, Pro GPU) |
| `<persist-edid>` | bool | `false` | Lock current EDID so display survives cable disconnect |
| `<display-name>` | string | *(empty)* | Select monitor by substring match on device name |
| `<display-blanker>` | bool | `false` | Auto-launch display_blanker.exe on startup |
| `<hdr-metadata>` | | | |
| &nbsp;&nbsp;`<max-cll>` | int | `1000` | Maximum Content Light Level (nits) |
| &nbsp;&nbsp;`<max-fall>` | int | `400` | Maximum Frame-Average Light Level (nits) |
| &nbsp;&nbsp;`<transfer>` | enum | *(parent)* | Override transfer in metadata |
| `<subregion>` | | | |
| &nbsp;&nbsp;`<src-x>` | int | `0` | Source X offset |
| &nbsp;&nbsp;`<src-y>` | int | `0` | Source Y offset |
| &nbsp;&nbsp;`<dest-x>` | int | `0` | Destination X offset |
| &nbsp;&nbsp;`<dest-y>` | int | `0` | Destination Y offset |
| &nbsp;&nbsp;`<width>` | int | `0` | Region width (0 = full) |
| &nbsp;&nbsp;`<height>` | int | `0` | Region height (0 = full) |
| `<gsync>` | | | |
| &nbsp;&nbsp;`<enabled>` | bool | `false` | Enable Quadro Sync framelock |
| &nbsp;&nbsp;`<master>` | bool | `false` | This output is the sync master |
| &nbsp;&nbsp;`<reference>` | enum | `internal` | `internal` \| `external` (house sync) |
| `<sync-group>` | int | `0` | Present barrier group (0 = disabled) |

---

## Configuration Examples

### Basic SDR Output

Single output on the first GPU, first display connector:

```xml
<channel>
    <video-mode>1080p5000</video-mode>
    <consumers>
        <vulkan-output>
            <gpu>0</gpu>
            <device>1</device>
        </vulkan-output>
    </consumers>
</channel>
```

### HDR10 Output with PQ

```xml
<channel>
    <video-mode>2160p5000</video-mode>
    <color-depth>16</color-depth>
    <color-space>bt2020</color-space>
    <color-transfer>pq</color-transfer>
    <consumers>
        <vulkan-output>
            <gpu>0</gpu>
            <device>1</device>
            <transfer>pq</transfer>
            <hdr-metadata>
                <max-cll>1000</max-cll>
                <max-fall>400</max-fall>
            </hdr-metadata>
        </vulkan-output>
    </consumers>
</channel>
```

### Auto-Detect HDR from Display

Let the module read the display's EDID and decide:

```xml
<vulkan-output>
    <gpu>0</gpu>
    <device>1</device>
    <edid-auto-hdr>true</edid-auto-hdr>
</vulkan-output>
```

### Wide-Gamut Display P3 Output

To output Display P3 content, configure the **channel** color space. The consumer's `<gamut>` setting has no effect:

```xml
<channel>
    <video-mode>2160p5000</video-mode>
    <color-space>p3-d65</color-space>
    <color-transfer>srgb</color-transfer>
    <color-depth>16</color-depth>
    <consumers>
        <vulkan-output>
            <gpu>0</gpu>
            <device>1</device>
        </vulkan-output>
    </consumers>
</channel>
```

### DCI Cinema Projection

Output for DCI-P3 projectors using gamma 2.6:

```xml
<channel>
    <video-mode>2048x1080p2400</video-mode>
    <color-space>p3-dci</color-space>
    <color-transfer>gamma26</color-transfer>
    <color-depth>16</color-depth>
    <consumers>
        <vulkan-output>
            <gpu>0</gpu>
            <device>1</device>
        </vulkan-output>
    </consumers>
</channel>
```

### HDR10 with Explicit Gamut and Transfer

Full HDR10 pipeline with BT.2020 gamut and PQ transfer — configured at channel level, with hardware HDR signaling on the consumer:

```xml
<channel>
    <video-mode>2160p5000</video-mode>
    <color-space>bt2020</color-space>
    <color-transfer>pq</color-transfer>
    <color-depth>16</color-depth>
    <consumers>
        <vulkan-output>
            <gpu>0</gpu>
            <device>1</device>
            <transfer>pq</transfer>
            <hdr-metadata>
                <max-cll>1000</max-cll>
                <max-fall>400</max-fall>
            </hdr-metadata>
        </vulkan-output>
    </consumers>
</channel>
```

### EBU Broadcast Reference Monitor

BT.709 primaries with gamma 2.4 for EBU tech 3320 compliance:

```xml
<channel>
    <video-mode>1080p5000</video-mode>
    <color-space>bt709</color-space>
    <color-transfer>gamma24</color-transfer>
    <consumers>
        <vulkan-output>
            <gpu>0</gpu>
            <device>1</device>
        </vulkan-output>
    </consumers>
</channel>
```

### HLG Live Broadcast

BT.2020 wide gamut with Hybrid Log-Gamma for backward-compatible HDR:

```xml
<channel>
    <video-mode>2160p5000</video-mode>
    <color-space>bt2020</color-space>
    <color-transfer>hlg</color-transfer>
    <color-depth>16</color-depth>
    <consumers>
        <vulkan-output>
            <gpu>0</gpu>
            <device>1</device>
            <transfer>hlg</transfer>
            <hdr-metadata>
                <max-cll>1000</max-cll>
                <max-fall>400</max-fall>
            </hdr-metadata>
        </vulkan-output>
    </consumers>
</channel>
```

### 4-Output Video Wall (Single GPU)

One wide channel split across four 1080p displays in a 2×2 grid:

```xml
<channel>
    <video-mode>3840x2160p5000</video-mode>
    <consumers>
        <!-- Top-left -->
        <vulkan-output>
            <gpu>0</gpu>
            <device>1</device>
            <subregion>
                <src-x>0</src-x>
                <src-y>0</src-y>
                <width>1920</width>
                <height>1080</height>
            </subregion>
            <sync-group>1</sync-group>
        </vulkan-output>
        <!-- Top-right -->
        <vulkan-output>
            <gpu>0</gpu>
            <device>2</device>
            <subregion>
                <src-x>1920</src-x>
                <src-y>0</src-y>
                <width>1920</width>
                <height>1080</height>
            </subregion>
            <sync-group>1</sync-group>
        </vulkan-output>
        <!-- Bottom-left -->
        <vulkan-output>
            <gpu>0</gpu>
            <device>3</device>
            <subregion>
                <src-x>0</src-x>
                <src-y>1080</src-y>
                <width>1920</width>
                <height>1080</height>
            </subregion>
            <sync-group>1</sync-group>
        </vulkan-output>
        <!-- Bottom-right -->
        <vulkan-output>
            <gpu>0</gpu>
            <device>4</device>
            <subregion>
                <src-x>1920</src-x>
                <src-y>1080</src-y>
                <width>1920</width>
                <height>1080</height>
            </subregion>
            <sync-group>1</sync-group>
        </vulkan-output>
    </consumers>
</channel>
```

### Multi-GPU with Quadro Sync II and House Sync

Two GPUs, each driving two outputs, all genlocked to external reference:

```xml
<!-- GPU 0: Outputs 1-2 (master) -->
<channel>
    <video-mode>1080p5000</video-mode>
    <consumers>
        <vulkan-output>
            <gpu>0</gpu>
            <device>1</device>
            <sync-group>1</sync-group>
            <gsync>
                <enabled>true</enabled>
                <master>true</master>
                <reference>external</reference>
            </gsync>
        </vulkan-output>
    </consumers>
</channel>
<channel>
    <video-mode>1080p5000</video-mode>
    <consumers>
        <vulkan-output>
            <gpu>0</gpu>
            <device>2</device>
            <sync-group>1</sync-group>
            <gsync>
                <enabled>true</enabled>
                <master>false</master>
            </gsync>
        </vulkan-output>
    </consumers>
</channel>

<!-- GPU 1: Outputs 1-2 (slave) -->
<channel>
    <video-mode>1080p5000</video-mode>
    <consumers>
        <vulkan-output>
            <gpu>1</gpu>
            <device>1</device>
            <sync-group>2</sync-group>
            <gsync>
                <enabled>true</enabled>
                <master>false</master>
            </gsync>
        </vulkan-output>
    </consumers>
</channel>
<channel>
    <video-mode>1080p5000</video-mode>
    <consumers>
        <vulkan-output>
            <gpu>1</gpu>
            <device>2</device>
            <sync-group>2</sync-group>
            <gsync>
                <enabled>true</enabled>
                <master>false</master>
            </gsync>
        </vulkan-output>
    </consumers>
</channel>
```

### Multi-GPU without Quadro Sync (Automatic Cross-GPU)

When outputs span multiple GPUs without Quadro Sync hardware, the module automatically detects the GPU mismatch and activates the cross-GPU transfer pipeline:

```xml
<!-- Mixer runs on GPU 0 (default), but outputs are on GPU 1 -->
<channel>
    <video-mode>1080p5000</video-mode>
    <consumers>
        <!-- This output is on a display connected to GPU 1 -->
        <vulkan-output>
            <gpu>1</gpu>
            <device>1</device>
        </vulkan-output>
        <!-- Another output on GPU 1 -->
        <vulkan-output>
            <gpu>1</gpu>
            <device>2</device>
            <sync-group>1</sync-group>
        </vulkan-output>
    </consumers>
</channel>
```

No special configuration is needed — the module:
1. Detects the LUID mismatch between mixer GPU (0) and output GPU (1)
2. Creates a `WGL_NV_gpu_affinity` context on GPU 1
3. Attempts CUDA peer DMA (if CUDA Toolkit was available at build time)
4. Falls back to PBO upload if CUDA is unavailable
5. Uses `shared_texture_pool` on GPU 1 for Vulkan interop

### Multi-GPU HDR Video Wall

```xml
<!-- 4K HDR content split across 4 outputs on GPU 1 (different from mixer GPU 0) -->
<channel>
    <video-mode>2160p5000</video-mode>
    <color-depth>16</color-depth>
    <color-space>bt2020</color-space>
    <consumers>
        <vulkan-output>
            <gpu>1</gpu>
            <device>1</device>
            <transfer>pq</transfer>
            <subregion>
                <src-x>0</src-x>
                <src-y>0</src-y>
                <width>1920</width>
                <height>1080</height>
            </subregion>
            <sync-group>1</sync-group>
        </vulkan-output>
        <vulkan-output>
            <gpu>1</gpu>
            <device>2</device>
            <transfer>pq</transfer>
            <subregion>
                <src-x>1920</src-x>
                <src-y>0</src-y>
                <width>1920</width>
                <height>1080</height>
            </subregion>
            <sync-group>1</sync-group>
        </vulkan-output>
    </consumers>
</channel>
```

> **Performance note**: At 4K60 BGRA16 (16-bit HDR), the frame size is ~66 MB/frame → ~4 GB/s sustained. This comfortably fits within PCIe 3.0 x16 bandwidth (15.7 GB/s). For 8K or higher refresh rates, NVLink is recommended.

### LED Wall with Presentation Delay

Compensate for a 3-frame delay in the LED processor:

```xml
<vulkan-output>
    <gpu>0</gpu>
    <device>1</device>
    <delay>3</delay>
    <buffer-depth>5</buffer-depth>
</vulkan-output>
```

> **Note**: Set `buffer-depth` to at least `delay + 2` to avoid frame drops during the initial fill.

### Output Identification on Startup

Useful for commissioning multi-display setups:

```xml
<vulkan-output>
    <gpu>0</gpu>
    <device>1</device>
    <identify-on-start>true</identify-on-start>
</vulkan-output>
```

Each output flashes a distinct color (blue, green, red, cyan, magenta, yellow) for 3 seconds based on its output index.

### Display Name Matching

Select the output monitor by name instead of index — portable between machines:

```xml
<vulkan-output>
    <gpu>0</gpu>
    <device>1</device>
    <display-name>BNQ</display-name>
</vulkan-output>
```

The `<display-name>` value is matched as a case-insensitive substring against the Windows device name, adapter name, and monitor model. Useful for configs that should work across different machines (e.g., "MTT" matches Virtual Display Driver monitors, "BNQ" matches BenQ monitors).

### Display Blanker (Crash-Safe Output)

Keep outputs black when CasparCG is not rendering:

```xml
<vulkan-output>
    <gpu>0</gpu>
    <device>1</device>
    <display-blanker>true</display-blanker>
    <display-name>MTT</display-name>
</vulkan-output>
```

This launches `display_blanker.exe` alongside CasparCG. The blanker creates black windows on matching monitors and (on Pro GPUs) can acquire dedicated display ownership for driver-level crash safety. See the [Display Blanker](#display-blanker) section for details.

### EDID Emulation (Headless Outputs)

Force a GPU output to report as connected even without a physical display:

```xml
<vulkan-output>
    <gpu>0</gpu>
    <device>2</device>
    <edid-emulation>true</edid-emulation>
</vulkan-output>
```

Injects a synthetic EDID matching the channel's video mode. Requires administrator privileges and a Pro-tier GPU. Useful for outputs that feed downstream video processing equipment (scalers, LED processors) that don't respond to EDID queries.

---

## Build Requirements

| Dependency | Required | Platform | Purpose |
|------------|----------|----------|---------|
| **Vulkan SDK** | Yes | Both | Core Vulkan API, validation layers |
| **NvAPI SDK** | Optional | Windows only | EDID readback, hardware HDR (display engine), Quadro Sync II, dedicated display |
| **CUDA Toolkit** | Optional | Both | GPU-to-GPU peer DMA for cross-GPU transfer |
| **GLEW** | Yes (via accelerator) | Both | GL extension loading for zero-copy interop |
| **EGL** (libEGL-dev) | Yes | Linux only | GL context creation, device enumeration |
| **DRM** (libdrm-dev) | Optional | Linux only | Display identification in logs |

### Built Artifacts

| Artifact | Platform | Description |
|----------|----------|-------------|
| `casparcg` / `casparcg.exe` | Both | Main server — includes vulkan_output module |
| `display_blanker.exe` | Windows only | Companion tool — display blanking (see [Display Blanker](#display-blanker)) |

Both are placed in the `build/shell/` output directory.

### Vulkan SDK

**Windows**: Set the `VULKAN_SDK` environment variable or install to the default `C:\VulkanSDK\` path. The build system auto-detects the latest version.

**Linux**: Install the LunarG Vulkan SDK packages (`vulkan-sdk`, `libvulkan-dev`). Standard distro packages also work (e.g., `apt install libvulkan-dev vulkan-validationlayers-dev`).

### NvAPI SDK (Windows only)

Set `NVAPI_SDK_PATH` CMake variable or place the SDK at `D:\Github\nvapi-main`. If not found, the module compiles without `CASPAR_NVAPI_ENABLED` — EDID auto-detection and Quadro Sync features are disabled, but all Vulkan output functionality remains available.

On Linux, NvAPI is not available. The module compiles with stub implementations — all NvAPI-dependent features gracefully report "not available."

### CUDA Toolkit

When CUDA Toolkit is installed (detected via CMake's `find_package(CUDAToolkit)`), the module compiles with `CASPAR_CUDA_PEER_ENABLED` and links `cudart_static`. This enables the CUDA peer DMA path for cross-GPU transfer.

If the CUDA Toolkit is not found at build time, the module compiles without CUDA support — multi-GPU still works via the PBO upload fallback path. No runtime error occurs.

Tested with CUDA 12.x. The module uses only CUDA Runtime API (no kernels, no .cu files) so it compiles as standard C++. Works identically on Windows and Linux.

### Linux Docker Build

A Dockerfile for Linux CI validation is provided:

```bash
# Compile-only validation (no GPU required):
docker build --target build-casparcg -f tools/linux/Dockerfile.vulkan-output .

# Full image with runtime dependencies:
docker build -f tools/linux/Dockerfile.vulkan-output -t casparcg-vulkan .
```

### Required Vulkan Extensions

The module requires or uses the following Vulkan extensions:

| Extension | Required | Platform | Purpose |
|-----------|----------|----------|---------|
| `VK_KHR_swapchain` | Yes | Both | Swapchain presentation |
| `VK_KHR_display` | No | Linux (primary) | Direct display mode (bypasses compositor) |
| `VK_KHR_surface` | Yes | Both | Surface abstraction |
| `VK_KHR_win32_surface` | No | Windows | Win32 window surface |
| `VK_KHR_external_memory_win32` | No | Windows | Zero-copy OGL↔VK shared memory |
| `VK_KHR_external_memory_fd` | No | Linux | Zero-copy OGL↔VK shared memory |
| `VK_KHR_external_semaphore_win32` | No | Windows | Zero-copy synchronization |
| `VK_KHR_external_semaphore_fd` | No | Linux | Zero-copy synchronization |
| `VK_NV_present_barrier` | No | Both | Multi-output frame-lock |
| `VK_EXT_hdr_metadata` | No | Both | HDR static metadata |
| `VK_EXT_display_control` | No | Both | VBlank fence timing |
| `VK_EXT_full_screen_exclusive` | No | Windows | FSE hint for DWM bypass |

---

## Linux Testing

For detailed Linux testing procedures, see [`src/modules/vulkan_output/LINUX_TESTING.md`](../src/modules/vulkan_output/LINUX_TESTING.md).

**Quick start on Linux:**

1. Ensure NVIDIA driver ≥ 535 is installed
2. Release the target display from the compositor: `xrandr --output DP-X --off`
3. Verify displays are visible to Vulkan: `vulkaninfo | grep displayProperties`
4. Start CasparCG: `./casparcg --log-level debug`
5. Add output: `ADD 1 vulkan_output 1`

**Key differences from Windows deployment:**
- No FSE window or blanker needed — `VK_KHR_display` provides exclusive access
- First-time setup requires releasing the display from X11/Wayland
- HDR uses `VK_EXT_hdr_metadata` directly (no NvAPI hardware path)
- Quadro Sync II framelock is not available

---

## Diagnostics

The consumer publishes real-time diagnostics via the CasparCG graph system:

| Graph Channel | Color | Description |
|---------------|-------|-------------|
| `frame-time` | Green | Time to record+submit+present one frame (normalized to frame period) |
| `tick-time` | Blue | Time between successive present calls (should be ~1.0 at correct fps) |
| `dropped-frame` | Red | Frame was dropped due to acquire/present failure |
| `late-frame` | Purple | Frame buffer overflowed — mixer is producing faster than output is consuming |
| `buffered-video` | Cyan | Current buffer fill level (0.0–1.0 relative to `buffer-depth`) |
| `vblank-drift` | Orange | Time between present submission and actual VBlank signal (Pro tier only) |

### OSC State

The following state paths are published via OSC:

```
vulkan-output/gpu             = 0
vulkan-output/output          = 1
vulkan-output/tier            = "pro"
vulkan-output/frames          = 123456
vulkan-output/display-lost    = false
vulkan-output/sync-group      = 1
vulkan-output/present-barrier = true
vulkan-output/delay           = 0
vulkan-output/delay-ms        = 0.0
vulkan-output/gsync/available = true
vulkan-output/gsync/synced    = true
vulkan-output/gsync/house-sync = true
vulkan-output/gsync/role      = "master"
```

---

## Troubleshooting

### "Vulkan output not found: gpu=0 output=1"

The specified GPU/output combination doesn't exist or isn't available for direct display.

- Run `INFO VULKAN_OUTPUT` to see available outputs
- For Pro tier: ensure the display is set to "dedicated" in NVIDIA Control Panel
- Check that the Vulkan SDK is installed and the `vulkan-1.dll` runtime is on PATH

### "Zero-copy unavailable, falling back to CPU"

The OpenGL ↔ Vulkan interop extensions are not available.

- Verify `GL_EXT_memory_object` and `GL_EXT_semaphore` are supported by the OGL driver
- Multi-vendor GPU configurations (e.g., Intel iGPU + NVIDIA dGPU) may not support cross-API memory sharing
- CPU fallback adds ~1–2ms per frame at 1080p, more at 4K

### "GPU LUID mismatch — activating cross-GPU transfer"

This is an informational message, not an error. The module detected that the target display is on a different GPU than the mixer and has activated the appropriate transfer pipeline.

- If followed by "CUDA peer DMA enabled" → optimal path, no action needed
- If followed by "CUDA peer transfer unavailable" → PBO fallback is used, which is still functional

### "CUDA peer transfer unavailable"

CUDA peer DMA could not be initialized. The module falls back to PBO upload.

Common causes:
- CUDA Toolkit was not found at build time (`CASPAR_CUDA_PEER_ENABLED` not defined)
- `cudaGLGetDevices()` failed — the GL context wasn't created with a CUDA-capable driver
- Mixed GPU vendor setup (e.g., NVIDIA + AMD) — CUDA only works with NVIDIA GPUs
- The CUDA runtime failed to initialize (driver mismatch or out-of-memory)

The PBO fallback is automatic and still provides smooth output.

### "Affinity context LUID doesn't match Vulkan device"

The `WGL_NV_gpu_affinity` context was created but its GPU doesn't match the Vulkan physical device.

- This may indicate a driver bug or mismatch between GPU enumeration order
- Verify GPU ordering: `INFO VULKAN_OUTPUT` shows the Vulkan enumeration, while NVIDIA Control Panel shows the WGL enumeration
- As a workaround, swap the `<gpu>` index value

### "GPU affinity context failed"

`WGL_NV_gpu_affinity` is not available or the target GPU index is invalid.

- Requires NVIDIA driver with `WGL_NV_gpu_affinity` support (Quadro/RTX/GeForce 400+)
- Verify the GPU index exists: `INFO VULKAN_OUTPUT` lists available GPUs
- The module falls back to CPU upload (SDR only) when affinity fails

### "VK_NV_present_barrier not available"

`sync-group` was configured but the extension isn't supported.

- Requires NVIDIA driver 525 or newer
- Only available on Quadro / RTX Pro GPUs
- The outputs will still work, but without driver-level frame-lock

### "Display disconnected (surface lost)"

The display was physically disconnected or the desktop configuration changed.

- With `<on-disconnect>retry</on-disconnect>` (default), the module will automatically recover when the display is reconnected
- Monitor `vulkan-output/display-lost` via OSC for alerting
- If the display was removed from the NVIDIA mosaic/surround configuration, a CasparCG restart may be required

### High `frame-time` Values

If `frame-time` consistently exceeds 0.5:

- Increase `<buffer-depth>` to 4–5 to absorb timing jitter
- Verify zero-copy path is active (check for "Zero-copy OGL→VK interop enabled" in the log)
- Check for GPU thermal throttling (`nvidia-smi -q -d PERFORMANCE`)

### Best Practices

1. **Always use Pro tier for broadcast** — configure displays as "dedicated" in NVIDIA Control Panel to enable `VK_KHR_display` and bypass the compositor.

2. **Set `buffer-depth` appropriately** — 3 is a good default. Increase to 4–5 for 4K or multi-output setups. For delay-compensated outputs, use `delay + 2` minimum.

3. **Use `sync-group` for multi-output on a single GPU** — this is the most reliable way to frame-lock multiple outputs without additional hardware.

4. **Use Quadro Sync II for multi-GPU framelock** — present barriers only work within a single GPU. Cross-GPU VSync synchronization requires Quadro Sync II hardware. However, cross-GPU *frame transfer* works automatically without any special hardware.

5. **Enable `identify-on-start` during commissioning** — each output flashes a distinct color, making it easy to verify physical wiring.

6. **Use `edid-auto-hdr` for mixed SDR/HDR setups** — the module will auto-detect which displays support HDR and configure accordingly.

7. **Monitor diagnostics** — the `vblank-drift` graph (Pro tier) shows the actual timing accuracy. Values consistently above 0.1 indicate timing issues.

8. **Install CUDA Toolkit for best cross-GPU performance** — even without NVLink, CUDA peer DMA is significantly faster than PBO upload because it uses the GPU's dedicated copy engines rather than CPU memcpy. The toolkit is only needed at build time.

9. **Use NVLink for 4K+ cross-GPU HDR** — at 4K60 16-bit, frame sizes reach ~66 MB, requiring ~4 GB/s sustained. PCIe 3.0 x16 handles this (15.7 GB/s), but NVLink (~600 GB/s) provides massive headroom and lower latency.

10. **Use `<gamut>` and `<eotf>` for precise color control** — the legacy `<transfer>` setting still works (infers bt2020 + pq/hlg automatically), but explicit `<gamut>` and `<eotf>` give full control. For standard BT.709 SDR output, leave both unset — the compute pass is completely bypassed with zero overhead.

11. **Use `<display-blanker>` in production** — prevents desktop exposure during crashes or restarts. On Pro GPUs with dedicated display, the output stays black even if both CasparCG and the blanker crash.

12. **Use `<display-name>` for portable configs** — instead of hard-coding `<device>` indices that change between machines, match by monitor manufacturer substring (e.g., `BNQ` for BenQ, `MTT` for VDD, `DEL` for Dell).

13. **Run `display_blanker.exe --autostart` as a Windows service or startup item** — for fully unattended systems, this ensures blanking is active before CasparCG launches, eliminating the brief desktop flash during system boot.
