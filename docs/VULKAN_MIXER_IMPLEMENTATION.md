# Vulkan Mixer Implementation — Technical Reference

> **Scope**: This document describes the Vulkan GPU image mixer and its
> associated readback strategies as implemented on the `CasparVPV` branch.
> All path references are relative to the CasparVP repository root.

---

## 1. Architecture Overview

The Vulkan mixer replaces the OpenGL image mixer as an alternative GPU
compositing backend. It is selected at channel startup via the
`<accelerator>vulkan</accelerator>` configuration element. The system is
split into three logical stages:

```
┌───────────┐     ┌───────────────┐     ┌──────────────────┐
│  Ingest    │────▸│  Composition  │────▸│  Readback /      │
│  (decode)  │     │  (VK mixer)   │     │  Consumer Output │
└───────────┘     └───────────────┘     └──────────────────┘
     GPU 0              GPU 0              GPU 0 or 1
```

1. **Ingest** — Producers (FFmpeg, CUDA ProRes, CUDA NotchLC) decode media
   into CPU arrays or, when available, directly into Vulkan textures via
   `cuda_vk_texture.h` zero-copy interop.

2. **Composition** — The `vulkan::image_mixer` composites all layers into a
   single render target (`VK_FORMAT_R16G16B16A16_UNORM` at 16-bit depth,
   `VK_FORMAT_R8G8B8A8_UNORM` at 8-bit). The output is an exportable
   `VkImage` backed by `VK_KHR_external_memory_win32`.

3. **Readback / Output** — Downstream consumers convert the composited
   texture to their wire format. DeckLink consumers use one of five
   GPU-readback strategies; the `vulkan_output` consumer uses GPU-native
   VK→VK zero-copy.

---

## 2. Vulkan Device Initialization

**File**: `src/accelerator/vulkan/util/device.h`, `device.cpp`

### 2.1 Instance & Physical Device Selection

The device is created via **VkBootstrap** (`vkb::InstanceBuilder`,
`vkb::PhysicalDeviceSelector`). Key choices:

| Setting | Value | Rationale |
|---|---|---|
| API version | Vulkan 1.3 | Required for `synchronization2`, `dynamicRendering` |
| Headless | `true` | No swapchain — the mixer renders to offscreen attachments |
| Validation layers | Debug builds only | Enables `VK_EXT_debug_utils` |
| GPU preference | Discrete | `vkb::PreferredDeviceType::discrete` |

### 2.2 Required Features

| Feature | Extension / Version |
|---|---|
| `descriptorIndexing`, `descriptorBindingPartiallyBound`, `runtimeDescriptorArray`, `shaderSampledImageArrayNonUniformIndexing` | VK 1.2 |
| `timelineSemaphore` | VK 1.2 |
| `scalarBlockLayout` | VK 1.2 |
| `dynamicRendering` | VK 1.3 |
| `synchronization2` | VK 1.3 |
| `dynamicRenderingLocalRead` | `VK_KHR_dynamic_rendering_local_read` |
| External memory export | `VK_KHR_external_memory_win32` |
| External semaphore export | `VK_KHR_external_semaphore_win32` |

### 2.3 Memory Management

- **VMA** (`VmaAllocator`) is used for staging buffers.
- **Manual allocation** with `VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT`
  is used for attachment textures (required for cross-device export).
- **Pool recycling**: Attachment, device-texture, and host-buffer pools use
  `tbb::concurrent_bounded_queue` keyed by `(width << 16 | height)` to
  recycle allocations without per-frame alloc/free.

### 2.4 Device LUID

The physical device LUID (`VkPhysicalDeviceIDProperties`) is queried at
init and stored on each exported texture. Downstream consumers (CUDA
interop, pure-VK readback) use the LUID to match the correct GPU when
creating their own device.

### 2.5 Dispatch Thread

All GPU work is serialized onto a single `boost::asio::io_context` thread
(`set_thread_name(L"Vulkan Device")`). Callers use `dispatch_async()` /
`dispatch_sync()` to enqueue work. This avoids external synchronization
around the VkDevice.

---

## 3. Graphics Pipeline

**File**: `src/accelerator/vulkan/util/pipeline.h`, `pipeline.cpp`

Two pipeline instances are created at device init — one for 8-bit RGBA and
one for 16-bit RGBA:

```cpp
_pipelines[0] = pipeline(_device, VK_FORMAT_R8G8B8A8_UNORM, ...);
_pipelines[1] = pipeline(_device, VK_FORMAT_R16G16B16A16_UNORM, ...);
```

### 3.1 Descriptor Layout

| Set | Binding | Type | Description |
|---|---|---|---|
| 0 | 0 | `VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT` | Background (target) texture for blending |
| 0 | 1 | `VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER` × 8 | Source texture array (Y, CbCr, BGRA, etc.) with partial binding |
| 0 | 2 | `VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC` | Per-draw `uniform_block` (752 bytes) |
| 0 | 3 | `VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT` | Local key attachment |
| 0 | 4 | `VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT` | Layer key attachment |

### 3.2 UBO Ring Buffer

64 descriptor sets are pre-allocated. Each maps to a fixed 256-byte-aligned
offset within a single large UBO. The pipeline rotates through these slots
across draws within a renderpass, avoiding per-draw descriptor writes.

### 3.3 Samplers

Two immutable samplers are created:

- **Linear** (`VK_FILTER_LINEAR`, `VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE`)
  — used for normal textures.
- **Nearest** (`VK_FILTER_NEAREST`) — used when exact texel sampling is
  needed (e.g., local/layer key).

---

## 4. Renderpass & Compositing

**File**: `src/accelerator/vulkan/util/renderpass.h`, `renderpass.cpp`

### 4.1 Dynamic Rendering with Local Read

The mixer uses `VK_KHR_dynamic_rendering_local_read` instead of traditional
VkRenderPass objects. This allows reading the current color attachment as an
input attachment within the same rendering scope — critical for blend mode
compositing where each draw reads the background it composites onto.

Each draw call produces a `layer_info` containing:
- The target attachment and optional key attachments
- The 752-byte `uniform_block` filled by `image_kernel`
- Clipped and transformed vertex coordinates

On `commit()`, all layers are batched into a single command buffer recording:
1. Vertex data is uploaded to a host-visible vertex buffer.
2. UBO data is written as a contiguous block.
3. For each layer: descriptor set is bound with dynamic offset →
   `vkCmdDraw()` with the layer's vertex count.
4. A timeline semaphore is signaled on submit (see §5).

### 4.2 Attachment Management

Attachments (`create_attachment()`) are VkImages with all of:
- `TRANSFER_SRC` — for GPU→CPU readback
- `INPUT_ATTACHMENT` — for dynamic rendering local read
- `COLOR_ATTACHMENT` — for rendering into
- `TRANSFER_DST` — for clears
- `SAMPLED` — for use as a texture source

Attachments use **export memory** (`VK_KHR_external_memory_win32`) so they
can be imported by consumer-side VkDevices or CUDA.

A per-frame-slot attachment pool (`frame_data::attachment_pool_`, max 4)
recycles allocations. This keeps the underlying `VkDeviceMemory` and its
Win32 HANDLE stable across frames, which is critical because
`cudaImportExternalMemory()` costs 10–150 ms and must be avoided per-frame.

### 4.3 Triple Buffering

The kernel maintains 3 `frame_data` slots (`frame_buffer_size = 3`), each
with its own command buffer and fence. `create_renderpass()` advances to the
next slot, waits for its fence (previous frame N-3), resets the command
buffer, and returns a new `renderpass` bound to that slot.

This means up to 2 frames can be in-flight on the GPU while the mixer
prepares a third.

---

## 5. Timeline Semaphores & Cross-Device Synchronization

### 5.1 Render Semaphore

Each `frame_data` slot creates an **exportable timeline VkSemaphore**:

```cpp
VkExternalSemaphoreHandleTypeFlagBits::eOpaqueWin32
VkSemaphoreType::eTimeline
```

On each `submit()`, the timeline value increments and is signaled when the
GPU finishes the command buffer. The Win32 HANDLE is cached after the first
`vkGetSemaphoreWin32HandleKHR` call and exposed via:

- `renderpass::render_semaphore_handle()` → HANDLE
- `renderpass::render_semaphore_value()` → uint64

### 5.2 texture_wrapper

**File**: `src/accelerator/vulkan/util/texture_wrapper.h`

The composited attachment is wrapped in `texture_wrapper` (implements
`core::texture`) and returned to the channel output pipeline. It exposes:

| Method | Purpose |
|---|---|
| `vk_texture()` | Native VK texture for GPU-native consumers |
| `export_win32_handle()` | Win32 HANDLE to the VkDeviceMemory |
| `export_alloc_size()` | Allocation size for CUDA import |
| `render_semaphore_handle()` | Timeline semaphore HANDLE |
| `render_semaphore_value()` | Timeline value to wait for |
| `ensure_render_complete()` | Fence wait (thread-safe, one-shot) |

Consumers choose their access path:
- **vulkan_output**: calls `vk_texture()` — zero-copy, no readback
- **CUDA readback**: imports the Win32 HANDLE + semaphore via `cudaImportExternalMemory` / `cudaImportExternalSemaphore`, waits GPU-side
- **VK readback**: imports via `VK_KHR_external_memory_win32` on a consumer-side VkDevice
- **CPU readback**: calls `ensure_render_complete()` then `image_data()`

### 5.3 Fence Wait Ordering

The `wait_fn` lambda captures the `renderpass` shared_ptr. Consumers that
need GPU-side sync (CUDA/VK) use the timeline semaphore directly. Consumers
that need CPU-side sync call `ensure_render_complete()`, which is
`atomic_flag`-guarded so concurrent consumers only wait once.

---

## 6. Compositing Shader

**File**: `src/accelerator/vulkan/image/fragment_shader.frag`

A single monolithic GLSL 450 fragment shader handles all compositing,
effects, and color management. The shader is ~2000 lines and uses the
752-byte `uniform_block` (§6.1) to select code paths at runtime via
bitfield flags.

### 6.1 uniform_block (752 bytes)

**File**: `src/accelerator/vulkan/util/uniform_block.h`

The UBO is laid out for **std140** rules (vec3 padded to vec4, mat3 stored
as 3×vec4). All fields have documented byte offsets. Key sections:

| Offset | Field | Description |
|---|---|---|
| 0 | `color_space_index` | BT.601 / BT.709 / BT.2020 |
| 4 | `precision_factor[4]` | Per-plane normalization (bit depth) |
| 20 | `blend_mode` | 0..28+ blend modes |
| 28 | `pixel_format` | Source pixel format enum |
| 96 | `flags` | 32-bit bitfield (see below) |
| 176–291 | Color grading | EOTF, gamut matrices, tone mapping |
| 300–344 | Lift/Midtone/Gain | 3-way color corrector |
| 412–440 | Split toning | Shadow/highlight color |
| 604–639 | Blur | Gaussian, directional, radial, tilt-shift |
| 640–735 | Shape overlay | SDF shapes with gradient/stroke |
| 736 | `flags2` | Extended feature flags |

### 6.2 shader_flags Bitfield

32 flags controlling which shader paths are active:

| Bit | Flag | Effect |
|---|---|---|
| 0 | `is_straight_alpha` | Pre-multiply alpha conversion |
| 1–2 | `has_local_key` / `has_layer_key` | Key compositing |
| 3 | `invert` | Luminance inversion |
| 4 | `levels` | Input/output levels with gamma |
| 5 | `csb` | Brightness/saturation/contrast |
| 6–7 | `chroma` / `chroma_show_mask` | Chroma keying |
| 8–9 | `is_360` / `is_curved` | Equirectangular projection + curved screen |
| 10 | `color_grading` | Full grading pipeline (EOTF→gamut→tone map) |
| 11–12 | `flip_h` / `flip_v` | Horizontal/vertical flip |
| 13 | `white_balance` | Temperature + tint |
| 14 | `lmg_enable` | Lift/midtone/gain |
| 15 | `hue_shift_enable` | Hue rotation |
| 16 | `tonebalance_enable` | Shadow/highlight recovery |
| 17 | `linear_sat_enable` | Linear saturation |
| 18 | `cdl_enable` | ASC CDL |
| 19 | `split_tone_enable` | Split toning |
| 20 | `gamut_compress` | ACES gamut compression |
| 21 | `lut3d_enable` | 3D LUT application |
| 22 | `hue_curve_enable` | Hue-vs-hue / hue-vs-sat curves |
| 23 | `sharpen_enable` | Unsharp mask |
| 24 | `grain_enable` | Film grain synthesis |
| 25 | `qualifier_enable` | Secondary color qualifier |
| 26 | `rgb_levels_enable` | Per-channel RGB levels |
| 27 | `curves_enable` | Curve adjustments |
| 28 | `blur_enable` | Blur (Gaussian/directional/radial/tilt-shift) |
| 29–30 | `shape_enable` / `shape_stroke` | Shape overlay with SDF rendering |
| 31 | `edge_blend` | Multi-projector edge blending |

### 6.3 Blend Modes

28+ blend modes implemented in the fragment shader as per-pixel operations
on the background input attachment: Normal, Multiply, Screen, Overlay,
Darken, Lighten, Color Dodge, Color Burn, Hard Light, Soft Light,
Difference, Exclusion, Linear Dodge, Linear Burn, Vivid Light, Linear Light,
Pin Light, Hard Mix, Reflect, Glow, Phoenix, and more.

### 6.4 360° and Curved Screen

When `is_360` is set, the shader performs equirectangular-to-rectilinear
projection using yaw/pitch/roll Euler angles, configurable FOV, and optional
frustum offsets. Lens distortion (k1/k2/k3) is applied in UV space.

`is_curved` adds cylindrical/spherical screen curvature (`screen_arc`
parameter) on top of the projection.

### 6.5 Color Grading Pipeline

When `color_grading` is set, the shader applies the full pipeline:

1. **EOTF decode** (`input_transfer`) — sRGB, PQ (ST 2084), HLG, Linear
2. **Input gamut → working gamut** (`input_to_working` mat3)
3. Per-pixel grading (LMG, CDL, white balance, etc.)
4. **Tone mapping** (`tone_mapping_op`) — Reinhard, ACES, Hable
5. **Working gamut → output gamut** (`working_to_output` mat3)
6. **OETF encode** (`output_transfer`)

Seven gamut matrix presets are compiled in:
BT.601, BT.709, BT.2020, DCI-P3, ACES AP0, ACES AP1, Display P3.

---

## 7. Still-Frame Cache

**File**: `src/accelerator/vulkan/image/image_mixer.cpp`

When inputs are unchanged between ticks (a "still" frame — same texture
pointers and identical `image_transform` per item), the mixer short-circuits
GPU composition entirely and returns the cached `texture_wrapper` +
CPU-pixel future from the previous tick.

This is implemented via a lightweight fingerprint:
```cpp
vector<pair<const void* /*texture_ptr*/, image_transform>> fingerprint;
```

If `fingerprint == prev_fingerprint_`, the cached result is returned
without touching the GPU. This reduces mixer GPU load from 60 fps to
~0 fps during still scenes, freeing GPU cycles for CUDA decode.

---

## 8. CPU Readback Skip

When **all** attached consumers use GPU-native paths (e.g., only
`vulkan_output` is connected), the mixer skips the GPU→CPU readback
entirely by checking `cpu_readback_needed_`:

```cpp
if (!cpu_readback_needed_.load(std::memory_order_relaxed)) {
    // return empty pixel array + texture_wrapper
}
```

This saves:
- One staging buffer allocation
- One layout transition barrier
- ~127 MB/frame of wasted PCIe bandwidth at 7680×2160 × 8 bytes/pixel

The flag is set to `false` by consumers that call
`image_mixer::set_cpu_readback_needed(false)`.

---

## 9. Texture Upload Paths

### 9.1 Zero-Copy (CUDA → VK)

When a CUDA producer (ProRes, NotchLC) decodes directly into a
`vulkan::texture_wrapper`, the mixer receives it via `frame.texture()` and
uses the VK texture directly — no CPU→GPU upload.

```cpp
auto vk_wrapper = dynamic_pointer_cast<texture_wrapper>(frame.texture());
item.textures.push_back(make_ready_future(vk_wrapper->vk_texture()));
```

### 9.2 Opaque Upload (pre-staged)

When `mutable_frame` was created by the VK mixer's own
`create_frame()`, the freeze callback stages pixel data into a VK buffer
and uploads asynchronously. The resulting `future_texture` vector is
attached as `frame.opaque()`.

### 9.3 CPU Upload (fallback)

For frames without a VK texture or opaque data, pixel arrays are uploaded
per-plane via `device::copy_async()`:
- Create/recycle a host-visible staging `buffer`
- `memcpy` pixels into the buffer
- `vkCmdCopyBufferToImage` with layout transitions
- Return the texture as a `future<shared_ptr<texture>>`

---

## 10. DeckLink Readback Strategies

The DeckLink consumer must convert the composited VK texture into
DeckLink's wire format (v210 YCbCr or 8-bit BGRA). Five readback modes
are available via `<gpu-readback-mode>`:

### 10.1 Configuration

**Files**: `src/modules/decklink/consumer/config.h`, `config.cpp`

```xml
<decklink>
  <gpu-readback-mode>auto</gpu-readback-mode>
</decklink>
```

| Value | Enum | Description |
|---|---|---|
| `auto` | `auto_select` | Try CUDA → Vulkan → CPU (default) |
| `cuda` | `cuda` | CUDA-VK interop |
| `vulkan` | `vulkan` | VK compute shader packing |
| `vulkan-dma` | `vulkan_dma` | VK DMA copy + CPU v210 pack |
| `cpu` | `cpu` | CPU-only (AVX2/memcpy) |

> **Note**: GPU readback modes only apply when the channel uses the Vulkan
> mixer (`<accelerator>vulkan</accelerator>`). With OpenGL, the DeckLink
> consumer always uses the CPU-based `v210_strategy` / `bgra_strategy`.

### 10.2 Strategy Selection

**File**: `src/modules/decklink/consumer/decklink_consumer.cpp`
— `create_format_strategy()`

```
if (!use_vulkan) return cpu_strategy;    // OGL: CPU only

switch (gpu_readback_mode) {
    auto_select → try cuda_vk → try vk_readback → cpu
    cuda        → try cuda_vk → cpu
    vulkan      → try vk_readback(dma=false) → cpu
    vulkan_dma  → try vk_readback(dma=true) → cpu
    cpu         → cpu
}
```

Each GPU strategy wraps a CPU fallback strategy for partial operations
(e.g., vulkan-dma uses VK DMA for readback but CPU AVX2 for v210 packing).

### 10.3 CUDA-VK Interop (`cuda`)

**Files**: `cuda_vk_strategy.h/cpp`, `cuda_vk_kernels.cu`, `cuda_vk_v210.cuh`

1. Imports the mixer's VK texture via `cudaImportExternalMemory()` using the
   Win32 HANDLE → maps to `cudaMipmappedArray` → creates `cudaSurfaceObject`.
2. Imports the timeline semaphore via `cudaImportExternalSemaphore()`.
3. Waits GPU-side (`cudaWaitExternalSemaphoresAsync`) for render completion.
4. CUDA kernels (`v210_pack_kernel`, `bgra_copy_kernel`) run on compute SMs:
   - Extract subregion (src_x, src_y, region_w, region_h)
   - Convert colorspace (BT.709/BT.2020 → v210 YCbCr) or copy BGRA
   - Write directly to pinned host memory (`cudaMallocHost`)
5. Triple-buffered: 3 host buffers + stream events for async D2H.
6. **Import caching**: Up to 8 texture/semaphore imports cached; only
   re-imported when the Win32 HANDLE changes.

**Advantage**: Entire pipeline runs on GPU — no CPU involvement.
**Disadvantage**: Compute kernels run on SMs — contends with CUDA ProRes/NotchLC decode.

### 10.4 Pure Vulkan Compute (`vulkan`)

**Files**: `vk_readback_strategy.h/cpp`, `vk_readback_v210.comp`, `vk_readback_bgra.comp`

1. Creates a **consumer-side VkDevice** on the same physical GPU (matched
   by LUID), with a **compute-only queue family** (avoids graphics queue
   contention).
2. Imports the mixer's VK texture via `VK_KHR_external_memory_win32` →
   creates a `VkImageView` on the consumer device.
3. Compute shader (`vk_readback_v210.comp`) packs RGBA → v210 entirely on
   GPU:
   - Reads 6 RGBA pixels per workgroup invocation
   - Converts to YCbCr (BT.709 or BT.2020 matrix)
   - Packs into 4×uint32 v210 words
   - Writes to a host-visible `VkBuffer` (SSBO)
4. The host-visible buffer is mapped and returned as the frame pointer.
5. Triple-buffered with fences.

**Advantage**: No CUDA dependency; v210 packing on GPU saves CPU.
**Disadvantage**: Compute shader runs on SMs — same contention as CUDA mode
under heavy SM workloads (CUDA ProRes decode).

### 10.5 Vulkan DMA (`vulkan-dma`)

**Files**: `vk_readback_strategy.h/cpp` (DMA path within same file)

1. Creates a consumer-side VkDevice on the same GPU, but selects a
   **transfer-only queue family** (`VK_QUEUE_TRANSFER_BIT` without
   `COMPUTE` or `GRAPHICS`).
2. Imports the mixer's VK texture (same as compute mode).
3. Issues `vkCmdCopyImageToBuffer` to copy the subregion from the imported
   image into a host-visible staging buffer. This uses the GPU's **DMA/Copy
   engine**, which is a separate hardware unit from the compute SMs.
4. Fences + triple buffering (same as compute path).
5. The raw RGBA pixel data is wrapped in a `const_frame` and passed to
   the CPU fallback strategy for v210 packing (AVX2 SIMD).

**Advantage**: Zero SM usage — the DMA engine runs in parallel with CUDA
decode. Designed specifically for CUDA ProRes + VK mixer scenarios.
**Disadvantage**: CPU must perform v210 packing (AVX2), so higher CPU load
than pure-GPU modes.

### 10.6 CPU Fallback (`cpu`)

The mixer's GPU→CPU readback (`device::copy_async()`) produces raw RGBA
pixels. The CPU strategy uses AVX2 SIMD intrinsics to pack v210 or convert
BGRA. This is always available and is the fallback for all GPU strategies.

---

## 11. Pixel Formats & Bit Depths

### 11.1 Mixer Output Format

| Bit Depth | VkFormat | Bytes/pixel | Notes |
|---|---|---|---|
| 8-bit | `R8G8B8A8_UNORM` | 4 | SDR content |
| 16-bit | `R16G16B16A16_UNORM` | 8 | HDR / 10-bit / 12-bit content |

The format is **unsigned normalized integer** (`UNORM`), not half-float
(`SFLOAT`). This is significant because:
- v210 packing can consume `uint16_t` values directly
- No float→int conversion overhead in readback
- CPU AVX2 `v210_strategy<uint16_t>` works on raw pixel data

### 11.2 Source Texture Formats

Input textures support all CasparCG pixel formats (via `pixel_format` enum
in the UBO):

| Format | Planes | Components |
|---|---|---|
| `bgra` | 1 | B8G8R8A8 or B16G16R16A16 |
| `rgba` | 1 | R8G8B8A8 or R16G16B16A16 |
| `ycbcr` | 3 | Y, Cb, Cr (4:2:0 / 4:2:2) |
| `ycbcra` | 4 | Y, Cb, Cr, A |
| `ycbcr_a` | 2 | Packed YCbCr + separate A |

The fragment shader handles colorspace conversion from any source format to
the target attachment format.

---

## 12. Geometry & Transforms

**Files**: `src/accelerator/vulkan/util/transforms.h/cpp`, `matrix.h/cpp`

### 12.1 Transform Stack

The mixer maintains a `transform_stack_` (vector of `draw_transforms`).
Each `push()` call combines the new `frame_transform` with the current top
via `combine_transform()`. This handles:

- Fill translation/scale (`fill_translation`, `fill_scale`)
- Clip rectangle (`clip_translation`, `clip_scale`)
- Anchor point
- Rotation (`angle`)
- Perspective (`perspective_scale`)

### 12.2 Vertex Computation

`matrix.cpp` computes a 4×4 vertex matrix from the combined transform:
anchor → scale → rotate → translate. This matrix is applied to geometry
coordinates to produce clip-space positions.

### 12.3 Polygon Clipping

`transforms.cpp` implements Sutherland-Hodgman polygon clipping against the
viewport edges. This clips geometry to `[0, 1]` in both X and Y,
interpolating texture coordinates along clipped edges.

Perspective-correct texture coordinate interpolation uses the `Q` factor:
```
Q = 1 / w_clip
tex_coord_corrected = tex_coord * Q
```

---

## 13. Layer Compositing Model

### 13.1 Layer Hierarchy

```
Channel
  └─ Layer (blend_mode)
       ├─ Sublayers (recursive)
       └─ Items (individual draw calls)
            ├─ Normal item → draw directly to target
            ├─ Key item → draw to key attachment
            └─ Mix item → draw to mix attachment, then composite
```

### 13.2 Blend Mode Fast Path

When `blend_mode == normal`, items within a layer are drawn directly to the
target attachment without an intermediate layer texture. Non-normal blend
modes require:
1. Draw all items to a temporary layer attachment
2. Draw the layer attachment onto the target with the specified blend mode

### 13.3 Key Compositing

- **Local key**: Set via `is_key` on a frame transform. The key item renders
  to `local_key_texture`, which is used as an alpha mask for subsequent
  non-key items.
- **Layer key**: Persists across items within a layer. The previous layer's
  local key becomes the next layer's layer key.

---

## 14. Performance Optimizations

### 14.1 Still-Frame Cache (§7)

Skips GPU composition entirely when inputs haven't changed.

### 14.2 CPU Readback Skip (§8)

Skips GPU→CPU transfer when only GPU-native consumers are attached.

### 14.3 Attachment Pool Recycling (§4.2)

Stable Win32 HANDLEs avoid expensive CUDA re-imports per frame.

### 14.4 Import Caching (CUDA strategy)

Up to 8 texture + 8 semaphore CUDA imports cached, keyed by Win32 HANDLE.

### 14.5 Subregion Extraction on GPU

DeckLink ports typically display a subregion (e.g., `src-x=3840` for a
right-half crop of a 7680-wide canvas). GPU strategies extract only the
relevant subregion, reducing PCIe bandwidth by up to 6× vs full-frame
readback.

### 14.6 Triple Buffering

All paths use 3-deep buffering (mixer command buffers, readback staging
buffers, DeckLink schedule buffers) to hide GPU latency and keep the
pipeline fully occupied.

### 14.7 Transfer-Only Queue (DMA mode)

The `vulkan-dma` mode selects a `VK_QUEUE_TRANSFER_BIT`-only queue family.
On NVIDIA GPUs, this maps to the dedicated Copy/DMA engine, which operates
independently from the compute SMs. This is specifically designed to avoid
GPU compute contention when CUDA decode workloads saturate the SMs.

---

## 15. Configuration Reference

### 15.1 Channel-Level

```xml
<channels>
  <channel>
    <video-mode>7680x2160p6000</video-mode>
    <accelerator>vulkan</accelerator>    <!-- or "ogl" -->
    <consumers>
      <decklink>
        <gpu-readback-mode>auto</gpu-readback-mode>
        <!-- auto | cuda | vulkan | vulkan-dma | cpu -->
      </decklink>
    </consumers>
  </channel>
</channels>
```

### 15.2 DeckLink Consumer

| Parameter | Default | Description |
|---|---|---|
| `<gpu-readback-mode>` | `auto` | Readback strategy selection |
| `<hdr>` | `false` | Enable HDR metadata output |
| `<pixel-format>` | `rgba` | Wire format: `rgba` (BGRA 8-bit) or `yuv` (v210) |
| `<primary>/<device>` | `1` | DeckLink device index |
| `<primary>/<src-x>` | `0` | Subregion X offset in mixer canvas |
| `<primary>/<region-w>` | `0` | Subregion width (0 = full width) |

### 15.3 Backwards Compatibility

The XML parser also accepts `<gpu-strategy>` as a legacy alias for
`<gpu-readback-mode>`. The mapping is identical.

---

## 16. File Map

### Mixer Core
| File | Purpose |
|---|---|
| `src/accelerator/vulkan/image/image_mixer.h/cpp` | Public mixer interface, layer management, still-frame cache |
| `src/accelerator/vulkan/image/image_kernel.h/cpp` | Renderpass creation, triple-buffered command buffers, UBO filling, timeline semaphore export |
| `src/accelerator/vulkan/image/fragment_shader.frag` | ~2000-line GLSL 450 fragment shader |
| `src/accelerator/vulkan/image/vertex_shader.vert` | Pass-through vertex shader |

### Vulkan Utilities
| File | Purpose |
|---|---|
| `src/accelerator/vulkan/util/device.h/cpp` | VkDevice, VMA, memory pools, async dispatch |
| `src/accelerator/vulkan/util/pipeline.h/cpp` | Graphics pipeline, descriptor layout, UBO ring |
| `src/accelerator/vulkan/util/renderpass.h/cpp` | Dynamic rendering, layer batching, command recording |
| `src/accelerator/vulkan/util/texture.h/cpp` | VkImage wrapper, Win32 handle export, LUID |
| `src/accelerator/vulkan/util/texture_wrapper.h` | Core::texture adapter for cross-device export |
| `src/accelerator/vulkan/util/buffer.h/cpp` | VMA staging buffer |
| `src/accelerator/vulkan/util/matrix.h/cpp` | 4×4 vertex matrix computation |
| `src/accelerator/vulkan/util/transforms.h/cpp` | Transform composition, polygon clipping |
| `src/accelerator/vulkan/util/draw_params.h` | Draw call parameter struct |
| `src/accelerator/vulkan/util/uniform_block.h` | 752-byte UBO struct + shader_flags enum |

### Readback Strategies
| File | Purpose |
|---|---|
| `src/modules/decklink/consumer/config.h/cpp` | `gpu_readback_mode_t` enum + XML parsing |
| `src/modules/decklink/consumer/decklink_consumer.cpp` | `create_format_strategy()` strategy factory |
| `src/modules/decklink/consumer/cuda_vk_strategy.h/cpp` | CUDA-VK interop readback |
| `src/modules/decklink/consumer/cuda_vk_kernels.cu` | CUDA v210/BGRA pack kernels |
| `src/modules/decklink/consumer/cuda_vk_v210.cuh` | v210 packing device functions |
| `src/modules/decklink/consumer/vk_readback_strategy.h/cpp` | Pure-VK compute + DMA readback |
| `src/modules/decklink/consumer/vk_readback_v210.comp` | GLSL compute shader for v210 packing |
| `src/modules/decklink/consumer/vk_readback_bgra.comp` | GLSL compute shader for BGRA copy |

### Integration
| File | Purpose |
|---|---|
| `src/modules/cuda_vk_texture.h` | CUDA → VK texture zero-copy wrapper |
| `src/core/consumer/channel_info.h` | `use_vulkan` flag per channel |
| `src/core/frame/frame.h/cpp` | `const_frame` lazy readback, `texture()` accessor |
| `src/accelerator/accelerator.h/cpp` | Backend selection (OGL vs Vulkan) |
