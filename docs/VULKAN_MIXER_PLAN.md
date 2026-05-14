# Plan: Vulkan Mixer Integration into CasparVP

**Date:** May 13, 2026
**Branch:** CasparVPV (forked from CasparVP)
**Upstream Reference:** PR [#1677](https://github.com/CasparCG/server/pull/1677) — "Initial support for Vulkan"

## TL;DR

Port the upstream Vulkan image mixer (`src/accelerator/vulkan/`) into CasparVP as a selectable alternative to the OGL mixer. Both paths coexist via `<accelerator>opengl|vulkan</accelerator>` config. The work spans 5 phases: merge upstream code, fix production blockers (D3D import, CUDA interop), port CasparVP shader features, optimize consumer plumbing for VK-native frames, and add cross-GPU VK→VK transfer.

## Scope

**In scope:** Mixer replacement (the image compositing pipeline).
**Out of scope (unchanged):**
- `vulkan_output` consumer's display/presentation code (swapchain, FSE, HDR metadata, display enumeration)
- These are already Vulkan-native and work correctly

**Future work (not in this plan):**
- `VK_KHR_video_encode` for hardware-accelerated FFmpeg/streaming encoding directly from VK textures
- Vulkan-based screen consumer (replace SFML+OGL preview)
- ~~VK→D3D interop for Decklink consumer (zero-copy SDI output)~~ → **Superseded**: VK→CUDA timeline semaphore interop implemented in `cuda_vk_strategy.cpp`. GPU-side `cudaWaitExternalSemaphoresAsync` eliminates CPU fence wait (22ms→0.06ms). See [DeckLink CUDA-Vulkan Interop](VULKAN_OUTPUT.md#decklink-cuda-vulkan-interop).
- Spout consumer (inherently OGL, no VK equivalent in protocol)
- NDI GPU-direct (NDI SDK doesn't support VK yet)

---

## Phase 1: Merge Upstream Vulkan Accelerator

**Depends on:** Nothing (starting point)
**Estimated effort:** ~2 days

### Tasks

1. Cherry-pick/merge PR #1677 files into `src/accelerator/vulkan/` (37 files, ~4500 lines)
   - `image/image_mixer.h`, `image/image_mixer.cpp` — core mixer
   - `image/image_kernel.h`, `image/image_kernel.cpp` — shader dispatch
   - `util/device.h`, `util/device.cpp` — VkDevice/queue/command management
   - `util/texture.h`, `util/texture.cpp` — VkImage wrapper
   - `util/buffer.h`, `util/buffer.cpp` — VkBuffer wrapper (staging)
   - `util/renderpass.h`, `util/renderpass.cpp` — dynamic rendering + draw dispatch
   - `shaders/*.frag`, `shaders/*.vert` — SPIR-V shaders
   - `CMakeLists.txt` — build integration

2. Add `ENABLE_VULKAN` CMake option and wire into `src/accelerator/CMakeLists.txt`

3. Update `src/accelerator/accelerator.h` — add `accelerator_backend::vulkan` enum (already in upstream)

4. Update `src/accelerator/accelerator.cpp` — `create_image_mixer()` factory switch (already in upstream):
   ```
   if (backend_ == vulkan) → vulkan::image_mixer
   else → ogl::image_mixer
   ```

5. Update config parsing in `src/shell/server.cpp` to read `<accelerator>vulkan</accelerator>` and call `accelerator.set_backend()`

### Verification

- Build with `ENABLE_VULKAN=ON`, run with `<accelerator>opengl</accelerator>` — no behavior change
- Run with `<accelerator>vulkan</accelerator>` — basic playback works (CPU pixel path)

### Key files

- `src/accelerator/accelerator.h` — `accelerator_backend` enum, `set_backend()`, `create_image_mixer()`
- `src/accelerator/accelerator.cpp` — factory switch
- `src/shell/server.cpp` — config parsing for `<accelerator>`
- `src/core/mixer/image/image_mixer.h` — abstract interface (push/visit/pop/render)

---

## Phase 2: Fix Production Blockers

**Depends on:** Phase 1
**Estimated effort:** ~3 days

### 2A. D3D Texture Import

Currently throws `"d3d texture import not supported on vulkan accelerator"`.

- Implement `vulkan::image_mixer::import_d3d_texture()` using `VK_KHR_external_memory_win32`
- D3D shared handle → `vkAllocateMemory` with `VkImportMemoryWin32HandleInfoKHR` → `VkImage`
- Reference: `vulkan_output/util/shared_texture_pool.cpp` already does this pattern (VK→GL direction)
- Needed for: Decklink capture, NDI input, any D3D-based producer

### 2B. CUDA Zero-Copy Interop

> **Status: Partially implemented** — The VK→CUDA semaphore pipeline is working for DeckLink output (`cuda_vk_strategy`). The VK mixer exports timeline semaphores via `VK_KHR_external_semaphore_win32`, and `core::texture` propagates semaphore handle/value through the frame pipeline. CUDA decoders writing directly to VK memory (the reverse direction) is not yet implemented.

Currently the VK mixer only receives CPU pixel data (no GPU texture import from CUDA decoders).

- Replace `cudaGraphicsGLRegisterImage` with `cudaImportExternalMemory()` + `VK_KHR_external_memory`
- Allocate VK texture with `VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT` export flag
- CUDA imports the exported handle → decoder writes directly to VK memory
- Modules to update:
  - `src/modules/cuda_prores/util/cuda_gl_texture.h` → new `cuda_vk_texture.h`
  - `src/modules/cuda_notchlc/util/cuda_gl_texture.h` → same pattern
  - `src/modules/cuda_gl_interop_lock.h` → rename/extend to `cuda_interop_lock.h`
- Both modules need `#ifdef ENABLE_VULKAN` dual paths (GL or VK depending on active mixer)

### 2C. render() Must Return GPU Texture

> **Status: Partially implemented** — `vulkan::texture_wrapper` wraps `VkImage` and implements `core::texture` with `render_complete_semaphore_handle()` and `render_complete_semaphore_value()` virtual methods. The DeckLink consumer (`cuda_vk_strategy`) uses these to GPU-wait on VK render completion. The `frame_data::submit()` path in `image_kernel.cpp` creates exportable timeline semaphores and signals them in `vkQueueSubmit`.

The upstream VK mixer's `render()` returns `{cpu_pixels, nullptr}` — no GPU texture.

- Modify VK `render()` to return `{cpu_pixels, vk_texture_wrapper}` where `vk_texture_wrapper` implements `core::texture`
- Create `vulkan::texture_wrapper` that wraps `VkImage` and implements `core::texture` interface
- This enables zero-copy for vulkan_output consumer (VK→VK, no CPU readback)
- The `frame.texture()` will hold a VK texture instead of OGL texture

### Key files

- `src/core/frame/frame.h` — `const_frame::texture()` returns `shared_ptr<core::texture>`
- `src/core/frame/texture.h` — abstract `core::texture` base class
- `src/accelerator/ogl/util/texture.h` — OGL implementation (reference)

---

## Phase 3: Port CasparVP Shader Features

**Depends on:** Phase 1 (can run parallel with Phase 2)
**Estimated effort:** ~5-7 days

### 3A. Shader Translation

Port ~1100 lines of CasparVP-specific GLSL from `shader.frag` to Vulkan GLSL.

| Feature Group | Lines | Priority |
|---|---|---|
| 360° equirectangular projection | ~200 | High |
| Curved screen / edge blending | ~150 | High |
| ACES color pipeline (EOTF, gamut, tone map) | ~120 | High |
| Blur (6 types) | ~150 | Medium |
| White balance, lift/midtone/gain, ASC CDL | ~120 | Medium |
| 3D LUT, hue curves, tone curves | ~100 | Medium |
| Sharpening, film grain | ~90 | Medium |
| 2D shapes (SDF) | ~100 | Low |
| Secondary qualifier (HSL key) | ~60 | Low |
| Split toning, gamut compression | ~70 | Low |

### 3B. Uniform Block Restructuring (parallel with 3A)

The OGL path uses 180+ individual `shader->set()` calls. For Vulkan:
- Group uniforms into a UBO or push constant block (max 128 bytes push constants, rest in UBO)
- Map CasparVP's `image_kernel.cpp` uniform-setting code to VK descriptor writes
- Reference: `src/accelerator/ogl/image/image_kernel.cpp` (lines 300-750)

### 3C. Texture Samplers (parallel with 3A)

CasparVP shader uses additional texture inputs beyond upstream:
- `lut3d_tex` (sampler3D) — 3D LUT
- `hue_curve_tex` (sampler2D, 256×1 RGBA32F) — hue curves
- `curve_lut_tex` (sampler2D, 256×1 RGBA32F) — tone curves
- Add these to the VK descriptor set (bindless texture array already has 8 slots)

### Key files

- `src/accelerator/ogl/image/shader.frag` — 1700-line CasparVP OGL shader (source of truth)
- `src/accelerator/ogl/image/image_kernel.h/.cpp` — uniform dispatch (180+ uniforms)
- Upstream VK shader (`src/accelerator/vulkan/shaders/`) — target for porting

---

## Phase 4: Consumer Plumbing Optimization

**Depends on:** Phase 2C
**Estimated effort:** ~3 days

### 4A. vulkan_output Consumer — VK-Native Path

When mixer is Vulkan, `frame.texture()` holds a VK texture. Skip the entire GL↔VK interop chain:
- No `shared_texture_pool`, no `interop_context`, no `gpu_affinity_context`
- Direct `vkCmdBlitImage` from mixer's VK texture → swapchain image
- Same-GPU: zero-copy (just a layout transition + blit)
- Eliminates: binary semaphore, timeline semaphore, GL signal/wait

Implementation:
- In `send()`: detect `frame.texture()` type. If VK texture, store VkImage directly
- In `present_frame()`: skip GL interop branch, blit directly from VK texture
- Keep OGL path for backward compatibility when `<accelerator>opengl</accelerator>`

### 4B. screen_consumer — Fallback (parallel with 4A)

Screen consumer uses OGL directly. With Vulkan mixer:
- CPU path (`frame.image_data()`) still works — no change needed
- GPU zero-copy path breaks — use CPU fallback (preview only, negligible cost)

### 4C. Other Consumers (no changes needed)

- **decklink_consumer**: ~~Uses CPU pixels → unchanged~~ → **Updated**: When VK mixer active, `cuda_vk_strategy` uses VK→CUDA timeline semaphore interop for GPU-side sync. Double-buffered v210 output with per-buffer CUDA events. Falls back to CPU pixels with OGL mixer.
- **ffmpeg_consumer**: Uses CPU pixels → unchanged
- **cuda_prores consumer**: Uses CPU pixels → unchanged
- **NDI**: Uses CPU pixels → unchanged

### Key files

- `src/modules/vulkan_output/consumer/vulkan_output_consumer.cpp` — main consumer
- `src/modules/vulkan_output/util/gpu_frame_cache.h` — frame cache (bypass for VK mixer)
- `src/modules/screen/consumer/screen_consumer.cpp` — screen consumer

---

## Phase 5: Cross-GPU VK→VK Transfer

**Depends on:** Phase 4A
**Estimated effort:** ~2 days

With VK mixer, cross-GPU transfer becomes VK→VK instead of OGL→VK:
- Export VK texture memory via `VK_KHR_external_memory`
- CUDA imports both source (GPU A) and destination (GPU B) via `cudaImportExternalMemory`
- `cudaMemcpy2D` between the two CUDA-imported buffers
- No GL context needed on either GPU
- Eliminates: `gpu_affinity_context`, `interop_context`, PBO fallback
- Falls back to CPU staging (`upload_frame_cpu`) if CUDA unavailable

### Key files

- `src/modules/vulkan_output/util/cuda_peer_transfer.h/.cpp` — current CUDA transfer (GL-based)

---

## Verification Matrix

| Phase | Test | Expected Result |
|---|---|---|
| 1 | `PLAY 1-10 AMB LOOP` with `opengl` accelerator | Identical to current behavior |
| 1 | `PLAY 1-10 AMB LOOP` with `vulkan` accelerator | Basic playback works (CPU path) |
| 2B | ProRes/NotchLC decode with VK mixer | GPU decode without CPU readback |
| 3 | All CasparVP effects (360, grading, blur, shapes) | Identical output VK vs OGL |
| 4A | vulkan_output with VK mixer | Zero-copy present, no GL interop logs |
| 4A | vulkan_output with OGL mixer | Unchanged behavior (regression) |
| 5 | Cross-GPU with VK mixer | Latency reduced by ~1 frame |

---

## Decisions

- **Dual paths:** Both OGL and VK mixers coexist. `<accelerator>auto</accelerator>` defaults to `opengl`, switched to `vulkan` after parity testing.
- **Shader approach:** Line-by-line GLSL port. VK GLSL is nearly identical — only descriptor binding syntax differs.
- **No VK_NV_present_barrier cross-GPU:** Confirmed TDR on different VkDevices. Software barrier or Quadro Sync hardware only.
- **Scope:** Mixer replacement only. vulkan_output display code unchanged.

## Further Considerations

1. **Descriptor pool:** Upstream uses 64 sets. CasparVP's extra textures may need 128+. Profile under complex compositions.
2. **Mipmapping:** Upstream VK textures are `mipLevels=1`. Add mipmap generation post-Phase 3 for downscale quality.
3. **Multi-queue:** Upstream uses 1 graphics queue. Dedicated transfer queue improves async uploads — defer to post-integration.
4. **VK_KHR_video_encode:** Future follow-up for FFmpeg consumer (hardware encode from VK textures).
