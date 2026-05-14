# Vulkan Mixer Assessment

**Date:** May 11, 2026
**Commit:** [7964b57](https://github.com/CasparCG/server/commit/7964b57a52fb640c78e1236e040af1b93b20ac1c)
**PR:** [#1677](https://github.com/CasparCG/server/pull/1677) — "Initial support for Vulkan"
**Author:** Niklas Pandersson (niklaspandersson)
**Scope:** 37 files changed, ~4500 lines added

## Overview

Drop-in replacement for the OpenGL image mixer, merged into CasparCG `master`. Disabled by default behind the `ENABLE_VULKAN` CMake flag. Configurable via `<accelerator>vulkan</accelerator>` in `casparcg.config` (default is `auto` which evaluates to `opengl`).

## What's Implemented (at parity with upstream OGL)

| Feature | Status |
|---------|--------|
| Push/visit/pop layer compositing | ✅ |
| All ~30 Photoshop blend modes | ✅ |
| All pixel formats (BGRA, RGBA, YCbCr, YCbCrA, UYVY, etc.) | ✅ |
| Chroma key with spill suppression | ✅ |
| Levels (min/max input/output + gamma) | ✅ |
| Contrast / Saturation / Brightness | ✅ |
| Local key + layer key | ✅ |
| Color space (BT.601 / BT.709) | ✅ |
| Opacity, straight alpha | ✅ |
| Crop / clip / perspective transforms | ✅ |
| Rotation / scale / translation / anchor | ✅ |
| Texture Q correction (projective) | ✅ |
| 8/10/12/16-bit depth | ✅ |
| Texture + buffer pooling with GC | ✅ |
| Diagnostics (`info()`) | ✅ |
| Async GPU transfers (timeline semaphores) | ✅ |
| Bindless textures | ✅ |
| Config selection (`auto`/`opengl`/`vulkan`) | ✅ |

## Architecture

- **Dependencies:** vk-bootstrap 1.4.328, VulkanMemoryAllocator 3.3.0, Vulkan SDK 1.4.328
- **Min Vulkan version:** 1.3 + `VK_KHR_dynamic_rendering_local_read` + `VK_EXT_robustness2`
- **Threading:** Single graphics queue via boost::asio io_context (mirrors OGL pattern)
- **Shader compilation:** SPIR-V via `glslc` at build time, embedded via `bin2c`
- **Frame buffer:** Triple-buffered command buffers with fence sync
- **Descriptor model:** Bindless texture array (8 slots), 64 descriptor sets

## Production Blockers

### 1. D3D Texture Import — Stubbed Out

```cpp
throw std::runtime_error("d3d texture import not supported on vulkan accelerator");
```

On Windows, this blocks any producer handing off D3D textures (Decklink capture, NDI, etc.).

**Fix:** Use `VK_KHR_external_memory_win32` to import D3D shared handles as `VkImage`. The existing `vulkan_output` module already does this pattern in reverse. Straightforward plumbing, no architectural changes needed.

### 2. No CUDA Zero-Copy Interop

> **Status update (May 2026):** The VK→CUDA direction is now implemented for DeckLink output. `cuda_vk_strategy` uses `cudaWaitExternalSemaphoresAsync` to GPU-wait on the VK render timeline semaphore. The CUDA→VK direction (GPU decoders writing to VK textures) remains unimplemented.

The OGL mixer uses `cudaGraphicsGLRegisterImage` for GPU-decoded frames (ProRes, NotchLC) to go directly to GL textures without CPU readback. No Vulkan equivalent is wired up.

**Fix:** Use `cudaImportExternalMemory()` with `VK_KHR_external_memory`. Allocate Vulkan texture with export flag → CUDA imports the handle → decoder writes directly. Actually cleaner than the GL path. NVIDIA documents this workflow extensively.

### 3. Screen Consumer Remains OGL-Only

`screen_consumer.cpp` directly calls GL APIs. The host-path (CPU pixel data → PBO upload) still works, but the zero-copy `gpu_strategy` path (directly binding `frame.texture()`) breaks since there's no GL texture with Vulkan backend.

**Impact:** Low — screen consumer is a preview tool. The host-path penalty is negligible for a preview window. Real output consumers (Decklink, NDI, streaming) receive frames via CPU anyway. DeckLink with VK mixer now uses GPU-side VK→CUDA semaphore interop (`cuda_vk_strategy`) instead of CPU pixels.

**Possible fixes:**
- VK→GL interop bridge via `VK_KHR_external_memory` (restores zero-copy)
- SFML 3 Vulkan surface support (already in build system)
- Leave as-is (CPU fallback works fine)

### 4. Small Descriptor Pool

`DescriptorPoolSize = 64` — complex multi-layer compositions could exhaust this. Would need profiling under real broadcast scenarios.

### 5. No Mipmapping

OGL device pool supports mipmap textures (pool indices > 3). Vulkan textures created with `mipLevels = 1` only. Affects quality of downscaled content.

### 6. Single Graphics Queue

All rendering AND transfer operations share one `vkQueueSubmit`. Vulkan's architecture benefits from separate transfer queues — performance optimization left on the table.

### 7. Driver Requirements

Requires Vulkan 1.3 + extensions (`VK_KHR_dynamic_rendering_local_read`, `VK_EXT_robustness2`). Limits compatibility to recent drivers:
- NVIDIA: 525+
- AMD: RDNA2+ 
- Intel: Arc

## CasparVP-Specific Gap

The Vulkan fragment shader (~578 lines) ports only **upstream** OGL shader features. The CasparVP branch has a ~1700-line shader with major additions NOT present in the Vulkan shader:

| Feature Category | Lines (approx) |
|-----------------|----------------|
| 360° projection (equirectangular, yaw/pitch/roll/FOV, lens distortion) | ~200 |
| Curved screen / soft-edge blending (multi-projector) | ~150 |
| ACES color pipeline (EOTF, gamut mapping, tone mapping) | ~120 |
| White balance, lift/midtone/gain, ASC CDL | ~80 |
| 3D LUT, hue curves, tone curves | ~100 |
| Sharpening, film grain, blur (6 types) | ~150 |
| 2D shapes (SDF rect/circle/ellipse with gradients) | ~100 |
| Secondary qualifier (HSL key) | ~50 |

Porting these to Vulkan GLSL would require translating ~1100 lines of shader code plus wiring the additional uniform inputs through the `uniform_block` push constants.

## Verdict

**For upstream CasparCG master:** Functionally complete for image mixing. Main blockers are D3D import (Windows) and CUDA interop (GPU decode). On Linux headless without D3D/CUDA, usable today.

**For CasparVP:** Requires porting all extended shader features, plus implementing the additional uniform/texture inputs those features need. Significant effort but no architectural barriers. VK→CUDA semaphore interop for DeckLink output is now implemented (`cuda_vk_strategy`), eliminating the CPU fence bottleneck (22ms→0.06ms). Cross-GPU async peer copy (`cudaMemcpyPeerAsync` with GPU-side event chain) and GPU topology detection at startup are also in place.

**Both production blockers (D3D + CUDA) are fixable** with existing Vulkan APIs and well-documented patterns. CUDA blocker #2 is partially resolved (VK→CUDA output path working; CUDA→VK input path remains). They are additive work isolated to specific functions, not redesign.
