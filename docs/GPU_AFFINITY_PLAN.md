# Per-Channel GPU Affinity for the Mixer

## Goal

Route each channel's image mixer to the same GPU as its vulkan-output consumer,
eliminating cross-GPU PCIe copies (~1–2 ms per 4K frame) in multi-GPU setups.

---

## Current Architecture

```
accelerator::get_device()  ──►  ONE vulkan::device  (picks first discrete GPU)
     │
     ├── channel 1 mixer (image_mixer on GPU 0)
     ├── channel 2 mixer (image_mixer on GPU 0)
     └── channel N mixer (image_mixer on GPU 0)

vulkan_output consumer (GPU 1)
     └── frame_cache detects LUID mismatch → cross-GPU staging copy
```

**Problem**: If channel 2's output is on GPU 1 but its mixer is on GPU 0, every
frame traverses PCIe (CUDA peer DMA or CPU staging). This is wasted bandwidth
when the mixer could have run on GPU 1 from the start.

---

## Proposed Architecture

```
accelerator::get_device(gpu_index)  ──►  map<int, vulkan::device>
     │
     ├── channel 1:  gpu_affinity=0  → mixer on GPU 0 (same as its output)
     ├── channel 2:  gpu_affinity=1  → mixer on GPU 1 (same as its output)
     └── channel 3:  (no affinity)   → mixer on GPU 0 (default)
```

**Result**: Same-GPU zero-copy path is used (VkImage handle import, no PCIe).

---

## Implementation Phases

### Phase 1: Config — Add `<gpu>` to Channel Level

**File**: `src/shell/casparcg.config` (schema) + parsing in `src/shell/server.cpp`

```xml
<channel>
  <video-mode>1080p5000</video-mode>
  <gpu>1</gpu>                         <!-- NEW: mixer GPU affinity -->
  <consumers>
    <vulkan-output>
      <gpu>1</gpu>                     <!-- output on same GPU -->
      <device>1</device>
    </vulkan-output>
  </consumers>
</channel>
```

**Behavior**:
- If `<gpu>` is present at channel level → use that GPU for the mixer
- If absent → auto-infer from the first `<vulkan-output>` consumer's `<gpu>` value
- If no vulkan-output consumer → default to GPU 0 (existing behavior)

**Changes**:
1. In `setup_channels()`, parse `<gpu>` from each `<channel>` ptree node
2. If not specified, do a lookahead scan of `<consumers><vulkan-output><gpu>` children
3. Pass the resolved `gpu_index` to `accelerator_.create_image_mixer()`

---

### Phase 2: Accelerator — Multi-Device Management

**Files**: `src/accelerator/accelerator.h`, `src/accelerator/accelerator.cpp`

Replace single-device model with per-GPU-index map:

```cpp
struct accelerator::impl {
    // OLD: std::shared_ptr<accelerator_device> device_;
    // NEW:
    std::map<int, std::shared_ptr<accelerator_device>> devices_;
    std::mutex devices_mutex_;

    std::shared_ptr<accelerator_device> get_device(int gpu_index = 0);
    std::unique_ptr<core::image_mixer> create_image_mixer(int channel_id, common::bit_depth depth, int gpu_index = 0);
};
```

`get_device(gpu_index)`:
- If `devices_[gpu_index]` exists → return it
- Otherwise → create new `vulkan::device(gpu_index)` (or `ogl::device(gpu_index)`), store and return

**Key constraint**: Each `vulkan::device` is a separate VkInstance + VkDevice + thread.
Multiple mixers on the same GPU share the same device (as today). Only cross-GPU
splits require new devices.

---

### Phase 3: Vulkan Mixer Device — Accept GPU Index

**Files**: `src/accelerator/vulkan/util/device.h`, `src/accelerator/vulkan/util/device.cpp`

Currently:
```cpp
device::impl::impl() {
    // ...
    gpu_selector.prefer_gpu_device_type(vkb::PreferredDeviceType::discrete)
                .select();  // picks "best" discrete GPU
}
```

Change to accept `gpu_index` parameter:
```cpp
device::device(int gpu_index = 0);

// In impl constructor:
impl(int gpu_index) {
    // enumerate all discrete GPUs
    auto all_gpus = gpu_selector.select_devices();
    // pick the one at position gpu_index (with LUID dedup)
    _vkb_physical_device = all_gpus[gpu_index];
}
```

This mirrors how `vulkan_output::vulkan_device` already accepts `gpu_index` in its
constructor and calls `select_physical_device(gpu_index)`.

---

### Phase 4: OGL Mixer Device — GPU Affinity (Windows)

**Files**: `src/accelerator/ogl/util/context.cpp`, `src/accelerator/ogl/util/device.h`

On Windows, WGL provides `wglEnumGpusNV()` / `WGL_NV_gpu_affinity` to create
GL contexts bound to a specific GPU. SFML's `sf::Context` doesn't support this,
so we need:

```cpp
// New constructor:
device_context(int gpu_index);

// Windows implementation using WGL_NV_gpu_affinity:
HGPUNV gpus[8];
wglEnumGpusNV(gpu_index, &gpus[gpu_index]);
HDC affinity_dc = wglCreateAffinityDCNV(&gpus[gpu_index], ...);
HGLRC ctx = wglCreateContextAttribsARB(affinity_dc, ...);
```

On Linux, EGL provides `EGL_EXT_device_enumeration` for the same purpose.

**Note**: If WGL_NV_gpu_affinity is not available (non-NVIDIA), fall back to
default context (existing behavior). Log a warning that GPU affinity is not
supported.

---

### Phase 5: Shared Resources Across Mixer Devices

**Concern**: The `channel_texture_store` (used by previz 3D) and `previz_bridge`
currently assume a single VkDevice. With multiple devices, cross-channel previz
texture sharing needs an explicit cross-device copy.

**Solution**:
- `channel_texture_store` already works at the GL level (via OGL previz device)
- The previz OGL device can remain singular (GPU 0) since previz is a monitoring tool
- Each channel posts its composited texture to the store via `copy_async` — this
  already goes through the OGL device which handles interop

**No change needed** if we keep the previz device on GPU 0 and accept that previz
sampling a GPU-1 channel's texture involves one small copy. Previz is a dev/preview
tool, not a latency-critical output path.

---

### Phase 6: Frame Cache Simplification

**File**: `src/modules/vulkan_output/consumer/vulkan_output_consumer.cpp`

With mixer-on-same-GPU, the `is_cross_gpu()` case becomes rare (only if user
misconfigures `<channel><gpu>0` with `<vulkan-output><gpu>1`). The frame_cache
cross-GPU staging code remains as a safety fallback but is no longer exercised
in the common case.

The consumer's `send()` path simplifies:
- VK-native path (same GPU) → always taken when config is correct
- Cross-GPU fallback → only when user intentionally splits mixer/output GPUs

**Optional enhancement**: Log a performance warning at startup if channel mixer
GPU ≠ output GPU:
```
[warning] Channel 2 mixer is on GPU 0 but vulkan-output targets GPU 1.
          Cross-GPU transfer will add ~1-2ms latency. Consider setting <gpu>1</gpu> at channel level.
```

---

## Config Inference Logic (Phase 1 Detail)

```
for each <channel> in config:
    gpu_affinity = channel_ptree.get("gpu", -1)   // explicit

    if gpu_affinity == -1:
        // Auto-infer from consumers
        for each child in <consumers>:
            if child is "vulkan-output":
                gpu_affinity = child.get("gpu", 0)
                break

    if gpu_affinity == -1:
        gpu_affinity = 0   // default

    mixer = accelerator.create_image_mixer(channel_id, depth, gpu_affinity)
```

---

## Risk Assessment

| Risk | Mitigation |
|------|-----------|
| Multiple VkDevices increase VRAM usage (duplicate pipelines, texture pools) | Pools are lazy-allocated; pipeline shaders are small (~50KB each) |
| Thread contention with multiple device threads | Each device has its own IO thread — no contention |
| OGL GPU affinity not available on all drivers | Fallback to default GPU with warning; cross-GPU path still works |
| Config validation: gpu index out of range | Log error + fall back to GPU 0 at startup |
| Existing single-GPU deployments regressed | No change in behavior when all channels use GPU 0 (same device returned) |
| previz cross-device texture sharing | previz stays on GPU 0; small copy acceptable for monitoring |

---

## Implementation Order

| # | Phase | Effort | Dependency |
|---|-------|--------|-----------|
| 1 | Config parsing + gpu_index pass-through | Small | None |
| 2 | Accelerator multi-device map | Medium | Phase 1 |
| 3 | Vulkan mixer device gpu_index selection | Medium | Phase 2 |
| 4 | OGL mixer device GPU affinity (Windows) | Medium | Phase 2 |
| 5 | Previz cross-device handling | Small | Phase 3 |
| 6 | Frame cache simplification + warnings | Small | Phase 3 |

Phases 3 and 4 are independent (Vulkan backend vs OGL backend) and can be done
in parallel. Phase 4 is only needed if the OGL backend is still used; if the
project is Vulkan-only going forward, it can be deferred.

---

## Validation Plan

1. **Single GPU (regression)**: Existing config with no `<gpu>` at channel level →
   verify mixer still lands on GPU 0, no behavior change
2. **Multi-GPU same affinity**: `<channel><gpu>1` + `<vulkan-output><gpu>1` →
   verify `frame_cache_->is_cross_gpu() == false`, zero-copy path taken
3. **Multi-GPU mismatch (intentional)**: `<channel><gpu>0` + `<vulkan-output><gpu>1` →
   verify cross-GPU fallback still works, warning logged
4. **Auto-inference**: No `<gpu>` at channel level, `<vulkan-output><gpu>1` →
   verify mixer auto-selects GPU 1
5. **Performance**: Measure frame latency (send→present) with and without affinity
   on a dual-GPU system — expect ~1-2ms improvement
