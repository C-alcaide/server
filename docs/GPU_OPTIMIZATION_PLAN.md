# GPU Resource Optimization — Eliminating Wasteful Roundtrips

> Analysis and implementation plan for applying PR #1651's "keep data on GPU"
> strategy across all custom modules. Covers pitfalls, production risks, and
> quality concerns that may not surface immediately during testing.

> **Looking for how to *use* any of this?** See
> [PIPELINE_EFFICIENCY_GUIDE.md](PIPELINE_EFFICIENCY_GUIDE.md) — which paths
> exist, how to stay on them, what limits a channel, and how to diagnose one that
> is late. This document is the reasoning, the measurements and the ideas that
> were rejected; that one is the operating instructions.

**Status**: Phases 1–7 implemented. Phase 5 (GPU-direct decode) was found never to have
executed and has been rewritten — it is now byte-exact against the CPU path. Phases 6–7
(native semi-planar upload; GPU planes carried separately on `const_frame`) are new.

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
| **FFmpeg consumer** *(Phase 4)* | `!gpu_direct_` | NVENC: mixer texture → CUDA frame → encoder, no readback (OpenGL only). Host path everywhere else |

### Consumers that still use CPU readback
| Consumer | Issue | Impact |
|----------|-------|--------|
| NDI | `image_data()` → raw pointer to NDI SDK | MEDIUM — no GPU path in standard SDK |
| Replay (VMX) | `image_data()` → VMX CPU encode | LOW — VMX designed for CPU |
| Image (PRINT) | `image_data()` → PNG encode | LOW — one frame per capture, not per tick |
| sACN / ArtNet | Reads ~2.4% of the pixels, forces 100% of the readback | **HIGH — see below** |

> **"Few pixels at low refresh rate" is not the same as low bandwidth.** The DMX
> consumers rasterise a handful of fixture quads on their own 10–30 Hz thread, but
> `needs_cpu_frame_data()` is polled per channel tick, so declaring it forces the
> mixer to read the *whole* composited frame back at the *channel's* rate — 8.29 MB
> at 50 fps, 415 MB/s, to compute a few RGB triples. On an LED-wall channel that is
> the single most expensive consumer attached, and it is the one whose output needs
> the fewest bytes. Fix planned: `texture::read_pixels_reduced()`, a box-filtered
> 1/8 readback pulled by the consumer on its own clock (129 KB at 10 Hz).

### Consumers that never touch pixels

These declared nothing and so defaulted to `needs_cpu_frame_data() == true`, re-arming
the readback for the entire channel. Since `any_consumer_needs_cpu_data()` short-circuits
on the first `true`, one of these was enough to defeat every GPU-native consumer sharing
the channel — and the shipped config puts `<system-audio />` on channel 1, so a stock
install paid for it.

| Consumer | `needs_cpu_frame_data()` | Why it is safe |
|----------|:---:|---|
| OAL (`system-audio`) | `false` | `send()` reads `frame.audio_data()` only |
| PortAudio | `false` | same |
| ProRes bypass | `false` | `send()` discards the frame; recording is driven off DeckLink callbacks |

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

**Fingerprinting approach** (rewritten — see "Fingerprint completeness" below):
- Store `shared_ptr<texture>` (not raw pointers) to prevent pool-recycled addresses from
  causing false cache hits (ABA problem — **Audit Fix #1**)
- Compare **all** plane textures, the combined `image_transform`, `frame_geometry`,
  `pixel_format_desc`, the owning layer's blend mode and the item's position in the
  layer/sublayer tree, plus channel-wide state (target dimensions, colour space/transfer,
  `auto_color_convert`, tone-map parameters, calibration LUT identity/strength/bypass)
- An unresolved upload future marks the fingerprint **incomplete**; incomplete never matches
- Clear cache when `layers` is empty (same as VK mixer)

**Production risks:**
- **False positive (stale frame displayed):** the real risk of this optimisation. Addressed
  by comparing every input (see below) and by never matching an incomplete fingerprint.
- **Memory leak from cached textures:** Cache holds `shared_ptr<texture>`. Cleared when
  `layers` is empty (STOP/CLEAR/REMOVE scenarios).

### Fingerprint completeness (correction to the original design)

The first implementation compared only `textures[0]` and `image_transform`, per top-level
layer. That left four ways to freeze a frame on air, all since fixed:

| Gap | Effect |
|---|---|
| OGL did not walk `sublayers` at all | any change inside a sublayer was invisible to the cache |
| `geometry`, `pix_desc` and the layer's `blend_mode` were not compared | a mesh/scale-mode/blend change did not invalidate |
| only plane 0 was compared | a YUV item whose chroma planes changed but whose Y texture was reused did not invalidate |
| channel-wide colour state was not compared | `set_target_color` / tone-map changes did not invalidate (the calibration LUT was already handled by explicit invalidation) |
| `image_transform::operator==` itself skipped `projection` fields | notably the entire ICVFX block, so a tracked camera moving the inner frustum over a static plate froze the wall |

An unresolved texture future used to read as `nullptr`, so two different frames could also
compare equal while still uploading.

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

> ### ⚠ Status (2026-07-29): the description below was aspirational, not a record
>
> The consumer does **not** accept GPU textures. It never overrode
> `needs_cpu_frame_data()`, so it has always taken the full mixer readback — the
> log says so plainly at every `ADD`: `output[1] CPU readback required by
> consumer ffmpeg`. Read the paragraph below as the goal it was, not as
> implemented behaviour.
>
> What *was* implemented was a filter insertion, `format=nv12,hwupload_cuda`,
> commented as doing the conversion "on GPU via scale_cuda". Both halves of that
> claim were wrong:
>
> - `format=nv12` is **host libswscale**. No part of it ran on the GPU.
> - `scale_cuda` was never in the graph, and could not have done this job
>   anyway: it rejects RGB input outright at graph-config time
>   (`Unsupported input format: yuva420p`), so the filter named in the
>   justification is incapable of the conversion it was credited with.
>
> The conversion was also unnecessary. `h264_nvenc` lists `bgra` among its
> accepted input formats and converts to YCbCr inside the encoder, on the GPU.
> Measured at 1080p over 300 frames (`h264_nvenc -preset p4`):
>
> | path | CPU (utime) | PSNR vs source | file |
> |---|---|---|---|
> | `format=nv12,hwupload_cuda` (was) | 5.55 s | 48.88 dB | 138 KiB |
> | `format=nv12` alone | — | 48.88 dB (identical) | — |
> | RGB straight to NVENC (now) | **4.30 s** | **50.59 dB** | 131 KiB |
>
> So the forced conversion cost ~4.2 ms of CPU per frame — a fifth of a core at
> 50 fps — *and* lost 1.7 dB, because it subsampled chroma on the host that the
> encoder would otherwise have subsampled itself. `hwupload_cuda` contributed
> nothing measurable: it is PSNR-identical to `format=nv12` alone and slightly
> slower.
>
> Fixed by inserting nothing. The buffersink already negotiates against
> `codec->pix_fmts`, and with no filter forced it settles on `bgra` for an 8-bit
> channel and `rgba` for a 16-bit one. Verified on both mixer backends — the
> recordings are byte-identical in size across OGL and VK, and the encoder now
> logs its negotiated input format at open time:
> `[ffmpeg] h264_nvenc input format bgra (no host conversion)`.
>
> Two defects found alongside it, both fixed in the same commit:
>
> - **NVENC recording was broken via the documented option spelling.** The
>   x264-preset guard tested `codec:v` and `c:v` only, but the encoder is also
>   selected from `vcodec`. `ADD 1 FILE out.mp4 -vcodec h264_nvenc` therefore got
>   `preset:v=veryfast` applied, which NVENC rejects (`Undefined constant ... in
>   'veryfast'`), failing the recording outright.
> - **A CUDA device context leaked per recording.** `hw_device_ctx_` was a raw
>   local; the graph and encoder each took a reference and the original was never
>   released.
>
> **The readback is now gone too — done 2026-07-30, OpenGL only.** See
> "GPU-direct recording" below.

The FFmpeg file-output consumer accepts GPU textures from the mixer, reading
them back to CPU only when needed for software encoding. This avoids the
mixer-level readback when the FFmpeg consumer is the only one attached.

**Production risks:**
- **Software encoder breakage:** Must NOT activate GPU-only mode for `libx264`/`libx265`.
  (Verified for the format change above: `libx264` still negotiates `yuv444p` through
  libswscale, unchanged.)
- **User filter incompatibility:** If user passes custom filters, the filter graph may need
  CPU frames. Fallback logic handles this gracefully.

---

## Phase 5: FFmpeg Producer D3D11VA → GL Direct

> ### ⚠ Status (2026-07-29): rewritten and now byte-exact — see "Phase 5 completed" below
>
> ### The original design never actually ran
>
> Three separate reasons, found by instrumenting the decision rather than reading
> the code:
>
> 1. **A missing forwarder.** `ogl::image_mixer::impl` overrides
>    `gpu_device_handle()`, but the outer `image_mixer` — the `frame_factory`
>    producers hold — did not forward it, so it returned the base class's
>    `nullptr`. Every interop path that asks the mixer for its GL device declined
>    silently. Fixed.
> 2. **The eligibility check runs too early.** It requires
>    `dec.sw_pix_fmt != AV_PIX_FMT_NONE`, but a D3D11VA decoder only resolves
>    `ctx->sw_pix_fmt` after its first `get_format` callback — i.e. during
>    decoding, not at `avcodec_open2`. At producer start it is always `NONE`, so
>    the path declines every time. **Still open**: the decision has to be deferred
>    until the first hardware frame arrives.
> 3. **It failed silently**, so 1 and 2 were invisible. Each branch now logs its
>    reason once, and a one-shot diagnostic reports the layout decoded frames
>    arrive in and the mixer format they map to.
>
> The path is now **opt-in** via
> `configuration.ffmpeg.producer.gpu-direct-decode` (default `false`). Fixing the
> forwarder alone would have silently switched every progressive H.264/HEVC clip
> on the OpenGL backend onto the VideoProcessor's driver-defined colour
> conversion.
>
> **The right completion is no longer this design.** Now that
> `core::pixel_format::nv12` exists (phase 6), the decoded NV12 surface should be
> imported as its two planes and converted by the mixer's shader — bit-exact with
> the software path by construction, and 10-bit capable — instead of being
> converted by the VideoProcessor and imported as BGRA.

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

---

---

## Phase 5 completed: GPU-direct decode by NV12 plane import

The VideoProcessor is gone. The decoded NV12 surface is handed to the mixer as
its **two planes**, and the mixer's shader performs the colour conversion — the
same shader, with the same matrix, range and chroma siting, that a
software-decoded frame goes through.

**Verified byte-exact.** With `gpu-direct-decode` on and off, the same clip and
frame produce PNG captures with identical sha256. That is a property the
VideoProcessor could not have: its matrix and range are driver-defined.

`const_frame` gained the ability to carry GPU planes separately (see phase 7
below), which is what makes this expressible at all — previously a producer with
GPU-side frames had to convert to a single interleaved texture first, i.e. in the
one place that cannot see the channel's colour management.

### Four defects, none of them visible before the path was instrumented

| Defect | Effect |
|---|---|
| `wglGetProcAddress` called with no current GL context | every WGL_NV_DX_interop2 pointer was null; the bridge reported "not available" |
| The bridge created its own GL context and `wglShareLists`'d against the mixer's | fails, because the mixer's context is current on the GL device thread. It now dispatches onto that thread and borrows the mixer's context, as the rest of the codebase does |
| Eligibility required `ctx->sw_pix_fmt` at producer start | a D3D11VA decoder only resolves it in its first `get_format` callback, so it was always `NONE`. The decision is now made on the first hardware frame |
| The extraction shader sampled with normalised coordinates | decoded surfaces are padded to the codec's macroblock grid (H.264 stores 1080 as **1088**), so the sample spanned the padding and rescaled the picture — a ~4 row vertical shift at 1080p. It now indexes texels with `Load()` |

That last one is the useful lesson for any future interop work: **a decoded
surface is not the size of the picture.** It was caught only because the parity
test compares against the CPU path; it would have been invisible to a "does it
look right?" check.

Other requirements worth remembering: the hardware frame pool must be allocated
with `D3D11_BIND_SHADER_RESOURCE` (FFmpeg's default binds decode-only, so no
shader resource view can be created over it), plane views need the D3D11.3
`CreateShaderResourceView1` with `PlaneSlice`, and FFmpeg's D3D11 device lock
must be held around the extraction because the decoder shares that immediate
context.

**Still opt-in** via `configuration.ffmpeg.producer.gpu-direct-decode`. It is now
colour-exact rather than driver-defined, but it has been exercised on one GPU and
one codec pair only.

## Phase 6: Native semi-planar upload (NV12 / P010)

**Files**: `src/core/frame/pixel_format.h`, `src/modules/ffmpeg/util/av_util.cpp`,
`src/accelerator/{ogl/image/shader.frag,vulkan/image/fragment_shader.frag}`,
`src/core/frame/write_frame.cpp`

**Before:** a hardware-decoded frame arrives semi-planar (Y plane + interleaved
Cb/Cr plane) and was *described* as three planar planes, so `make_frame`
deinterleaved the chroma — and for P010 shifted every sample `>>6` — two bytes at
a time on TBB workers, into persistently-mapped `GL_MAP_COHERENT_BIT` memory.
That is uncached write-combining memory on discrete GPUs, where narrow scattered
writes from several threads defeat the write-combine buffers. One full-frame CPU
pass per decoded frame, purely to reshape data the GPU can sample directly.

**After:** `core::pixel_format::nv12` describes the decoder's own layout — plane 0
Y, plane 1 Cb/Cr at half resolution with stride 2 — which both backends already
support as RG8/RG16 textures. Both shaders read `.r` and `.rg`. `make_frame` is a
straight per-plane row copy for every format.

> **Enum ordering is load-bearing:** the mixer shaders `switch` on
> `pixel_format`'s *numeric* value. New formats must be **appended** before
> `count`; inserting one silently reinterprets every format after it.

**P010 needs no arithmetic.** Its 10 bits are high-aligned in each 16-bit word, so
`raw/65535` is already the correct normalised value. Declaring the planes `bit16`
(not `bit10`) selects `precision_factor = 1.0` and the normalisation is exact.
`bit10` would apply the ×64 scale meant for low-aligned planar 10-bit.

### Parity method (reusable for any pixel-path change)

The A/B that validated this is worth repeating for future work, because it is the
only way to distinguish "different" from "wrong":

1. **Use a *static* source.** The first attempt used `testsrc2`, which is
   animated; the captures were of different frames and every comparison was
   meaningless noise (~24 dB against everything). `smptehdbars` is static.
2. **A/B the same backend against the previous implementation**, by stashing just
   the changed file and rebuilding that one module. This is the measurement that
   actually answers "did I change the picture?".
3. **Attribute any cross-backend difference with a matched control** — the same
   content and subsampling through a code path the change did not touch.

Results for this phase:

| Comparison | Result |
|---|---|
| NV12 (H.264 8-bit), old vs new implementation | **byte-exact identical** |
| P010 (HEVC 10-bit), old vs new | 0.49% of samples differ by exactly 1 LSB at 8-bit output, PSNR 71.2 dB — the new path is *more* accurate, since the old `>>6` discarded the low 6 bits. A scale error is excluded: the delta is 1 LSB regardless of sample value |
| 16-bit channel, P010 | full 0..65535 range, correct scale, both backends |
| OGL vs VK on the new path | 49.58 dB, max 17.51% |
| **Control**: OGL vs VK on the *unchanged* planar 4:2:0 path | 49.56 dB, max 16.94% — i.e. the cross-backend difference is pre-existing chroma sampling at hard edges, not introduced here |

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

### The frame contract, made explicit

Three additions turned this from a set of conventions into something checkable.

**1. `host_image_state()`** — `image_data()` returned an empty array both when
readback had been deliberately skipped and when it merely had not landed yet.
Callers reading "empty" as "black" emitted black frames; callers reading it as
"has pixels" dereferenced null (the risk noted in Phase 1.1 above).
`const_frame::host_image_state()` now returns `unavailable` / `deferred` /
`available`, and `has_host_image()` is the shorthand. Resolving a completed
readback is free, so it is cheap to call, and it never blocks on one in flight.

**2. `core::texture::owner_device()`** — an opaque identity of the device owning
the memory (the `VkDevice`, or the `ogl::device`). A mixer may bind a texture
natively **only** when this matches its own device. `dynamic_pointer_cast` alone
distinguishes backends, not devices, and with per-channel GPU affinity a route
between channels on different GPUs hands over memory the receiving device must
not touch.

**3. One resolver per mixer** — `resolve_item_textures()` replaced three
divergent branches per backend with a single documented order:

| Step | Source | Notes |
|---|---|---|
| 1 | GPU texture owned by this device | zero copy |
| 2 | GPU texture owned by another device | external-memory import **not implemented**; falls through |
| 3 | pre-staged upload futures on `opaque()` | validated against the owning device |
| 4 | host planes | rejected when `host_image_state() == unavailable` |
| 5 | nothing usable | item dropped, warned once |

#### Cross-GPU routes were silently broken

Verified on a two-GPU host (RTX A4000 + Quadro P4000), channel 1 on GPU 0 routed
into channel 2 on GPU 1. Three defects stacked up, none of which produced a
diagnostic:

| Defect | Symptom |
|---|---|
| `opaque()` carried the pre-staged upload futures with no owner, and the receiving mixer trusted them | GPU 0's `VkImage`s bound on GPU 1 — channel 2 rendered nothing, silently |
| `device::copy_async()` reused the staging buffer embedded in the source array regardless of which device owned it | a GPU 0 `VkBuffer` referenced in a submit to GPU 1 → `vk::Queue::submit: ErrorDeviceLost`, killing the channel |
| `const_frame::with_tag()` dropped the pending readback future (it copied `image_data_` — a vector of *empty* arrays when unresolved — and built a non-lazy frame) | every routed frame permanently lost the ability to produce CPU pixels; `route_producer` retags every frame |

All three are fixed: the staged payload and the staging buffers carry their
owning device (VMA allocator for Vulkan, an owner tag for OpenGL) and are
rejected when foreign, and `with_tag()` preserves the future. The cross-GPU route
now falls back to a host upload with a one-shot warning instead of rendering
nothing or taking the GPU down. Implementing step 2 (external-memory import)
would remove that readback.

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

## Is sequential layer receive a bottleneck? — measured 2026-07-29

`stage::receive` pulls layers one at a time on the stage executor. The standing
proposal was a `prefetching_producer` decorator with a bounded queue plus
parallel fan-out across layers, on the premise that one producer blocking in
`receive()` costs the channel its frame for every consumer.

Worth separating two cases first, because only one of them is a hazard:

- A producer returning an **empty** frame is already harmless.
  [layer.cpp](../src/core/producer/layer.cpp) substitutes `last_frame()`.
- A producer that **waits** — stalled read, CEF paint, a lock held by a decoder —
  delays every later layer in the loop.

The stage now measures the second case and publishes it under `receive` in
channel state (`INFO <ch>`): `tick_avg_us`, `tick_peak_us`, `budget_percent`,
`peak_budget_percent`, `layers`, and `slowest_layer` / `slowest_producer` /
`slowest_avg_us` / `slowest_peak_us` so a blocking producer can be named rather
than suspected. Recomputed once a second, re-emitted every tick (see the note in
the code about why both).

Measured at 1080p50 — 20 ms budget — with an NDI consumer attached so
composition and readback actually run, `bars.mov` looping on every layer:

| layers | receive avg | of budget | receive peak | of budget | **actual tick** | late frames |
|---|---|---|---|---|---|---|
| 0 (idle) | 0 | 0 % | 0 | 0 % | 20.00 ms | 1 / 687 |
| 1 | 90 µs | 0.45 % | 149 µs | 0.74 % | 20.00 ms | 4 / 1347 |
| 6 (mixed file+colour) | 225 µs | 1.1 % | 446 µs | 2.2 % | 20.00 ms | 89 / 2848 |
| 8 | 846 µs | 4.2 % | 6.9 ms | 34.6 % | 20.00 ms | 67 / 2113 |
| 16 | 932 µs | 4.7 % | 6.6 ms | 33.2 % | **30.61 ms** | 457 / 4205 |
| 24 | 1394 µs | 7.0 % | 6.8 ms | 34.2 % | **45.72 ms** | 845 / 5030 |

**Conclusion: the premise does not hold in this tree, and the decorator is not
justified.** The decisive rows are the last two. At 24 concurrent file producers
the channel has collapsed to 45.7 ms per tick against a 20 ms nominal — less
than half rate, 845 late frames — and `stage::receive` accounts for **1.4 ms of
those 45.7, about 3 %**. The other 97 % is composition, readback, the consumer,
and decode threads competing for CPU. Parallelising layer receive would address
3 % of the problem, at the price of a frame of added latency per decorated
producer and of the deliberate sources-before-routes ordering
(`orderSourceLayers`) and the single-threaded keyframe invariant
(`KEYFRAMES.md`). Sequential receive costs between 0.5 % and 7 % of the frame
across every load tested.

**What would reverse this.** `receive/budget_percent` large while
`timing/consume_load` (see `PORTAUDIO_MODULE.md`) is small — time genuinely going
into pulling layers rather than into the consumer. Not observed at any load
here. The figures are published precisely so this stays falsifiable in a real
deployment rather than being re-argued from the code.

Two cautions for anyone repeating the measurement:

- **Attach a consumer.** With no consumer the channel skips composition *and*
  readback, and `output.cpp` publishes no timing block at all — so the load is
  unrepresentative and "did we miss frames?" is unanswerable. Measured without
  one, 16 layers showed a receive peak of *108 %* of the frame budget; with a
  consumer attached the same scenario shows 33 %. The uncomfortable-looking
  number came from the less realistic configuration.
- **Peaks are worst at the edge of capacity, not past it.** 8 layers peaks
  higher than 24 relative to how much trouble the channel is in, because a
  producer that has fallen behind returns empty immediately (cheap), while one
  just barely keeping up is the one that waits.

Harness: `CasparCG-TestRunner/vkdispatch/` sibling, `recv_probe.py`.

---

## What actually limits layer count — measured 2026-07-29

Having established that `stage::receive` is 3 % of the tick on a channel running
at half rate, the obvious next question is what the other 97 % is. The channel
tick already timed itself — `produce-time`, `mix-time`, `consume-time`,
`osc-time`, `frame-time` — but only into the diagnostics graph, which is a
picture. Those same measurements are now published under `tick` in channel
state, with `unaccounted` for whatever the four phases do not explain.

Decomposed at 1080p50 (20 ms budget), NDI consumer attached, N layers each
playing `bars.mov`:

| layers | produce | **mix** | consume | osc | total | actual tick |
|---|---|---|---|---|---|---|
| 0 | 0.02 | 0.02 | 19.88 | 0.02 | 19.92 | 20.00 ms |
| 1 | 0.16 | 0.08 | 19.62 | 0.04 | 19.86 | 20.00 ms |
| 4 | 0.46 | 0.50 | 18.84 | 0.00 | 19.80 | 20.00 ms |
| 8 | 0.88 | 1.76 | 16.74 | 0.08 | 19.38 | 19.89 ms |
| 16 | 1.54 | **27.90** | 0.28 | 0.12 | 29.72 | 31.60 ms |
| 24 | 2.44 | **39.42** | 0.10 | 0.26 | 41.96 | 38.54 ms |

Two things are visible here. First, **the back-pressure clock working**: as `mix`
grows from 0.02 to 1.76 ms, `consume` shrinks from 19.88 to 16.74 to absorb it,
and the tick stays at 20.00 ms. That is the design in `PORTAUDIO_MODULE.md`
behaving as advertised, and it is now directly observable rather than argued.
Second, at 16 layers the slack is gone (`consume` → 0.28 ms) and `mix` alone
exceeds the frame period, so the channel goes late.

### It is upload bandwidth, and it depends on the source pixel format

`mix` includes waiting for the frame's staged uploads, so a bandwidth limit shows
up there rather than as a separate phase. Re-running the identical test with only
the *source clip* changed:

| layers | `mix`, `bars.mov` (gbrap16le, 15 MB/frame) | `mix`, `bars8.mp4` (yuv420p, 3 MB/frame) | tick | late |
|---|---|---|---|---|
| 8 | 4.92 ms | 0.52 ms | 20.00 ms | 2 |
| 16 | 27.90 ms | **0.30 ms** | 20.00 ms | **4** |
| 24 | 39.42 ms | **0.48 ms** | 20.01 ms | **7** |

**Twenty-four layers of 8-bit 1080p50 run at exactly nominal rate with 0.48 ms
of mix time.** The composition path was never the constraint. The collapse was
host→GPU upload bandwidth, and `bars.mov` is a lossless RGBA source that decodes
to `gbrap16le` — four 16-bit planes, 15 MB per frame per layer. At 24 layers that
is 360 MB per frame.

Measured plateau: **9.1 GB/s** (24 × 15 MB / 38.54 ms), against 15.8 GB/s
theoretical for the A4000's PCIe 3.0 x16 link and roughly 12 GB/s achievable in
practice. The channel is running the bus at about three quarters of its usable
throughput, which is the ceiling being hit.

### Capacity rule

At ~9.1 GB/s of usable upload, a frame period of `T` ms affords roughly
`9.1 × T / 1000` GB of uploads. Dividing by bytes per frame per layer:

| source | bytes/frame at 1080p | layers at 1080p50 (20 ms) | at 1080p25 (40 ms) |
|---|---|---|---|
| 8-bit 4:2:0 (NV12 / yuv420p) | 3.0 MB | ~60 | ~120 |
| 8-bit RGBA | 8.3 MB | ~22 | ~44 |
| 16-bit RGBA + alpha (`gbrap16le`) | 15.0 MB | ~12 | ~24 |

The 16-bit row predicts a ceiling of ~12 layers, and the finer sweep put the
crossover at 10–11. Close enough to use for planning, and it explains the shape
of the earlier numbers: beyond the crossover `mix` costs ~1.7–1.9 ms per added
1080p50 layer, which is 15 MB at ~9 GB/s.

**Consequences.**

- **Phase 6 (native NV12/P010 upload) attacked exactly this constraint**, which
  was not the reason it was originally proposed. Uploading two semi-planar planes
  instead of a converted RGBA frame cuts upload bytes for hardware-decoded YUV
  content, and upload bytes are what sets the layer ceiling.
- **The readback is a much smaller term.** A 1080p RGBA readback for a CPU
  consumer is 8.3 MB per frame — 415 MB/s at 50 Hz, under 5 % of the budget. So
  the remaining Phase 4 work (removing the FFmpeg consumer's readback) is worth
  doing for the CPU cost, not for bandwidth.
- **Do not benchmark with lossless RGBA content** unless that is what the
  deployment plays. `bars.mov` costs five times what ordinary 8-bit material
  does, and the difference is the whole result.

### Corrections this measurement forced

Recorded because each was a conclusion that looked well-supported and was wrong:

- **"The channel collapses at 16+ layers."** It collapses at 16+ layers *of
  16-bit RGBA*. With 8-bit 4:2:0 it was still at nominal rate at 24 layers, which
  was as far as the test went — the real ceiling is higher and was never reached.
- **"`mix` scaling 1.76 → 27.90 ms for a 2× layer increase is a cliff."** An
  artefact of sampling 8 then 16 with nothing between, plus run-to-run variance
  (the same 8-layer configuration measured 1.76 ms and 4.92 ms in two runs). The
  finer sweep shows a roughly linear ~1.8 ms per layer.
- **"~1.8 ms to composite one 1080p layer means the composition path is 10–20×
  slower than the hardware."** It is not compositing time at all; it is 15 MB of
  upload at bus speed. The arithmetic was right and the attribution was wrong.
- **The first version of this instrumentation reported 19.9 ms of `produce` on an
  idle channel.** `caspar::timer::elapsed()` measures from construction, and the
  accumulation had been moved to the end of the tick, so every phase read as
  nearly the whole tick. Each duration is now captured where its phase ends.

Harness: `CasparCG-TestRunner/vkdispatch/tick_probe.py`.

---

## The producer was silently degrading 10-bit and 4:4:4 — fixed 2026-07-30

Following the upload-bandwidth finding above, the next question was which source
types reach the mixer in a more expensive representation than they need. Measured
by playing one clip per source type and reading the arrival diagnostic
(`decoded frames arrive as X -> mixer pixel_format N`), the answer turned out to
be a correctness problem rather than a bandwidth one.

| source | arrived as (before) | arrives as (now) |
|---|---|---|
| H.264 4:2:0 8-bit | `yuv420p` | `yuv420p` |
| **H.264 High 10** (`yuv420p10le`) | **`yuv420p`** — truncated to 8-bit | **`yuv420p10le`** |
| **H.264 High 4:4:4** (`yuv444p`) | **`yuv420p`** — chroma subsampled | **`yuv444p`** |
| ProRes 422 HQ | `yuv422p10le` | `yuv422p10le` |
| ProRes 4444 + alpha | `gbrap16le` | `gbrap16le` |
| DNxHR HQ | `yuv422p` | `yuv422p` |
| FFV1 planar RGB | `gbrp` | `gbrp` |
| QuickTime RLE RGBA | `gbrap` | `gbrap` |
| v210 | `yuv422p10le` | `yuv422p10le` |

**Every 10-bit clip was being truncated to 8 bits, and every 4:4:4 clip
subsampled to 4:2:0, before the mixer ever saw them.** Measured on the source
alone, that conversion costs 43.0 dB for 10-bit content and 40.3 dB for 4:4:4 —
applied to every frame, silently.

### Cause

`bwdif` is appended to *every* video filter graph unless
`configuration.ffmpeg.producer.auto-deinterlace` is `none`. With the default
`deint=interlaced` it only acts on frames flagged interlaced, so on progressive
content it is a pass-through — but it is still *in* the graph, and its format
constraints therefore govern negotiation for all content. It lists `yuv420p`
first among its supported formats, and the buffersink was handed a fixed list of
36 formats and left to choose, so the choice was driven by the filter's
preference rather than by the source.

Removing bwdif confirms it: with `auto-deinterlace=none` the same clips arrive as
`yuv420p10le`, `yuv444p`, `nv12` and `yuva444p12le` respectively.

This is also the explanation for something previously recorded as an oddity of
the GPU path. `CasparCG-TestRunner/gpudirect/README.md` notes that 10-bit HEVC
measured ~46 dB *different* between GPU-direct and software decode, and concluded
"the GPU-direct output is the more accurate of the two … it is the filter graph's
format negotiation to fix, not the GPU-direct path". That was right, and this is
that fix.

### Fix

The buffersink is no longer offered formats that could lose picture. The offer is
filtered against what the stream declares, keeping only formats with at least the
source's bit depth, at least its chroma resolution, and alpha if the source has
alpha. Negotiation is then free to choose among them and cannot choose a lossy
one. **bwdif stays in the graph for all content — deinterlacing is unchanged.**

The restriction is derived from `codecpar->format`, deliberately *not* from
`sw_pix_fmt`. `sw_pix_fmt` is probed from the hardware frames context at open and
stays at `NV12` even when the decoder subsequently declines hardware decoding and
falls back to software — which is exactly what happens for High 10 and 4:4:4.
Restricting against `NV12` permits an 8-bit 4:2:0 result, i.e. precisely the loss
being prevented. The first version of this fix did that and changed nothing;
the log line reporting `restricted … for source nv12` on a `yuv420p10le` clip is
what gave it away. `codecpar` describes the content and is unaffected by how it
is decoded.

If no lossless candidate exists the full list is restored and the reason logged,
so the change can never do worse than the previous behaviour.

### Verification

- **Deinterlacing still works, measured rather than assumed.** Interlaced clips
  built from a *moving* source (a static source weaves into an identical frame
  and cannot show combing — the same trap as the gpudirect harness). Row-adjacency
  PSNR of the raw combed frame vs the channel output: 29.33 → **39.02 dB**
  (8-bit) and 28.94 → **39.12 dB** (10-bit). Combing is removed, and interlaced
  10-bit now keeps its precision as well.
- **The mixer renders the newly-arriving formats correctly.** Channel output vs a
  reference decode of the same frame: `yuv444p` **70.8 dB** (essentially exact —
  4:4:4 needs no chroma reconstruction), `yuv420p10le` 39.6 dB and `yuv420p`
  38.7 dB (both limited by chroma upsampling differing between GPU sampling and
  swscale, as expected).
- **No filter errors** on any of the eleven clips tested, on both backends, and
  the arrival table is identical for OpenGL and Vulkan.

### Costs

**Upload bytes rise for the affected content**, because it is no longer being
thrown away: 10-bit 4:2:0 goes 2.97 → 5.93 MB/frame and 4:4:4 2.97 → 5.93 MB.
Against the capacity table above that lowers the layer ceiling for those sources.
This is correctness bought with bandwidth, and it is the right trade — but it is
a real change to a deployment's headroom and should be planned for.

### Two follow-ups, both fixed 2026-07-30

**ProRes 4444 no longer converts YUV→RGB.** It was arriving as `gbrap16le`, which
loses no picture but costs a full-frame swscale. The cause was that the offered
list stopped at `yuva444p12`: bwdif has no 12-bit YUVA support and promotes it to
16-bit, so the only formats both bwdif and the sink accepted were RGB. Added
`AV_PIX_FMT_YUVA{420,422,444}P16` to the offered list, and the corresponding
`ycbcra`/`bit16` entries to `av_util.cpp`'s mapping — without those the format
would have mapped to `pixel_format::invalid`. ProRes 4444 now arrives as
`yuva444p16le`, 4 planes, same 15.82 MB/frame, with the conversion gone.

Verified against a reference decode at **46.4 dB**, per-channel r 45.1 / g 53.4 /
b 44.8. Green being much better than red and blue is chroma rounding, not a
matrix error — a wrong matrix would show a large systematic bias rather than an
RMS error of ~1.4 levels out of 255. The mixer now does this YUV→RGB conversion
instead of swscale, so this was the thing worth checking.

**`sw_pix_fmt` is corrected once a software frame reveals the truth.** It is
probed from the hardware frames context when the decoder is opened, before anyone
knows whether hardware decoding will be used; when it declines, the decoder emits
its native format while `sw_pix_fmt` still says `NV12`. The buffersrc was
therefore configured with `NV12` for a 10-bit stream — FFmpeg only warns
("Changing video frame properties on the fly…") and copes, but it is a false
declaration, and it repeated on every filter rebuild. An ordinary frame is
authoritative about its own layout, so it is now taken from there. Mismatch
warnings per clip drop from 3 to 1; the remaining one is the first graph build,
before any frame exists, and is unavoidable without deferring graph construction.

**The buffersrc's colour metadata had the same shape of problem, also fixed.**
For codecs that carry colour in the bitstream rather than the container — ProRes
among them — `codecpar` reports unspecified, so the buffersrc was configured
`csp: unknown range: unknown` and every real frame then disagreed with it. The
`Decoder` now records the colour space and range reported by decoded frames, and
the buffersrc uses those wherever the container said nothing. (Learned on the
`Decoder` rather than reusing the existing `stream_color_space_`, which the run
thread only updates when it *pops* frames — by then the graph has already been
built. The decode thread sees them first.)

### Result

`Changing video frame properties on the fly` warnings, per clip, across the
nine-source matrix on both backends:

| | before | after |
|---|---|---|
| ProRes 4444 | 3 | 1 |
| ProRes 422 | 3 | 1 |
| H.264 High 10 | 3 | 1 |
| H.264 High 4:4:4 | 3 | 1 |
| everything else | 0 | 0 |

The remaining one is the very first graph build, before any frame has been
decoded and while nothing better is knowable. Removing it would mean deferring
graph construction until the first frame arrives — a restructuring of producer
startup that is not worth it for a warning that is accurate at the moment it is
emitted.

Pixel output is unchanged by the colour fix (ProRes 4444 46.42 dB, H.264 4:4:4
70.81 dB against reference decodes, both identical to before it). That is the
expected result: the declaration was wrong, but with the format restriction in
place no conversion in these graphs depended on it. It matters for graphs where
one *does* occur — user-supplied filters, scaling — which is exactly where a
silently-wrong colour declaration would have been hardest to track down.

---

## CUDA ProRes producer discarded 4444 alpha — fixed 2026-07-30

The `CUDA_PRORES` producer decodes on the GPU and hands the mixer a texture it
allocated itself, so the uncompressed frame never crosses the bus — measured at
47 % less server CPU on six layers of ProRes 422 HQ, 55 % on twelve, and 68 % on
ProRes 4444. Since upload bandwidth is what limits layer count, that removes the
clip's contribution to the binding constraint rather than merely saving CPU.

But every ffmpeg-encoded ProRes 4444 clip composited as **opaque**: black where
the background should have shown, and `a=255` in a recorded transparent channel,
while the same file through the ffmpeg producer was correct.

### Cause

`decode_alpha_to_host` parsed the slice header for Y, Cb and Cr exactly as
FFmpeg does, but took the alpha payload size from an explicit field:

```c
if (hdr_bytes > 9)
    alpha_size = ((int)sl[8] << 8) | sl[9];   // otherwise stays 0
```

FFmpeg never reads such a field. In `decode_slice_thread` the alpha size is
always the remainder, `slice_data_size - y - u - v - slice_hdr_size`. And
`prores_ks` writes an **8-byte** slice header, so `hdr_bytes > 9` was false,
`alpha_size` stayed 0 for every slice, and each one took the "fill fully opaque"
path a few lines below.

Confirmed against a real file before changing anything — first slice:
`total=28 hdr_bytes=8 y=7 cb=5 cr=5`, leaving **3 bytes of alpha** that the
parser was reading as none. The frame header's byte 17 was `0x02`, correctly
identifying 16-bit alpha, so the failure was specifically the payload size and
not alpha detection.

Now computed as the remainder when it is not explicitly sized. Only reachable for
4444 — `decode_alpha_to_host` is called under `is_444` — so a 4:2:2 slice never
evaluates it.

### Verification

| | before | after |
|---|---|---|
| 4444 with transparency, over red: corner | (0,0,0) black | **(255,0,0) red** |
| transparent channel recorded: corner alpha | `a=255` | **`a=0`** |
| ProRes 422, 6 layers | 2.55 → 1.35 cores | 2.58 → 1.38 |
| ProRes 4444, 6 layers | 4.86 → 1.44 cores | 4.85 → 1.54 |
| picture vs ffmpeg producer | 39.87 / 46.68 dB | 39.87 / 46.68 dB |

The CPU savings and picture are unchanged; only the alpha channel differs.

Worth noting how it was found: the failure was uniform opacity rather than
noise, which pointed at a fill path rather than a decode error, and the "fill
fully opaque in case alpha plane is absent" comment named the branch. Parsing the
file's own slice header then turned a hypothesis into a measurement before any
code changed.

---

## QuickTime RLE / Animation with alpha — verified end to end 2026-07-30

The format work above changed what these clips hand the mixer (`argb` → `rgba`),
and it was verified on `smptehdbars`, which is opaque. Animation is used *for*
its alpha, so that is the case that needed checking. A clip with genuine
transparency — an opaque green disc on a fully transparent field, `qtrle`,
`argb`, corner RGBA `00000000` — was played and recorded on both mixers:

| | OpenGL | Vulkan |
|---|---|---|
| arrives at mixer as | `rgba` | `rgba` |
| composited over red: corner / centre | 255,0,0 / 0,255,0 | 255,0,0 / 0,255,0 |
| recording to `qtrle` | `argb`, 136 frames | `argb`, 137 frames |
| transparent channel recorded, corner RGBA | `0,0,0,0` | `0,0,0,0` |

So: alpha survives decode, compositing and encode, on both backends. The corner
showing the background layer rather than black is the thing to look for — black
there means the alpha was ignored and transparent pixels were drawn.

Two notes for anyone repeating this. CasparCG colour strings are `#AARRGGBB`, so
`#FF0000FF` is opaque *blue*; a "wrong" result that is exactly the wrong primary
is usually this. And `qtrle` negotiates `argb` on its own — the pixel-format
preference added above only narrows codecs that offer 4:2:0, which `qtrle` does
not, so alpha-capable codecs are left to negotiate as before.

GPU-direct recording does not engage here and should not: it is NVENC-only, and
NVENC carries no alpha. `qtrle` recordings take the host path.

Harness: `CasparCG-TestRunner/vkdispatch/qtrle_check.py`.

---

## Three follow-ups closed by measurement — 2026-07-30

Recorded because each looked like a defect and two turned out not to be.

### v210 rendering 10 dB below ProRes 422 — not a defect

Both arrive as `yuv422p10le` and take the same mixer path, yet v210 measured
29.79 dB against a reference decode where ProRes measured 39.79. The cause was
the harness. **The v210 decoder reports no colour metadata at all**
(`color_space=unknown`) where ProRes reports `bt470bg`; the channel then assumes
BT.709, which is correct for HD, while swscale defaults to BT.601. Decoding the
reference with `in_color_matrix=bt709` puts v210 at **39.56 dB**, in line with
everything else.

`quality_matrix.py` now detects a source with no declared matrix and pins the
reference to BT.709 to match the channel. Worth knowing generally: any
comparison against an ffmpeg decode of metadata-less content will show a
spurious ~10 dB gap.

### Packed ARGB/ABGR shader swizzles — real, unreachable, left alone

The bug is real: an `argb` source renders at 6.4 dB. But after excluding
`AV_PIX_FMT_{ARGB,ABGR}` from what the producer negotiates, **nothing in the
tree can produce `pixel_format::argb` or `::abgr`** — `av_util.cpp` is the only
source of them, and `write_frame.cpp` only consumes them. The shader cases are
dead code.

Two further facts made fixing them a poor use of effort:

- The two backends **disagree** on the swizzle for the same format (OpenGL uses
  `.brga`/`.grab`, Vulkan `.gbar`/`.abgr`). Two implementations differing is
  conclusive proof neither was ever exercised.
- The reachable packed case verifies at `inf` dB with an identity-looking
  swizzle, which does *not* follow from reading `FORMAT[stride]=GL_BGRA`. So the
  upload byte order is not what the code suggests, and deriving the correct
  swizzle on paper produces the wrong answer — it would have to be established
  empirically, by re-enabling a format that nothing uses.

Both shaders now carry a comment saying the cases are wrong, unreachable, and
must not be re-enabled without deriving the swizzles by experiment.

### Vulkan cannot sample `rgb24` — declined as a feature, made diagnosable

Supporting packed 24-bit RGB on Vulkan would mean expanding to four components
during upload, i.e. **33 % more upload bytes** than the `gbrp` fallback that
negotiation already picks and that renders at `inf`. Upload bandwidth is the
binding constraint on layer count (above), so this would be strictly worse than
doing nothing.

What was worth fixing is the failure mode. The VK mixer threw from inside the
channel tick with a driver-level error, taking the channel down, and gave no
hint why. It now checks `optimalTilingFeatures` for the 3-component format and
throws a message naming the cause and the two ways out. It should be
unreachable — producers no longer negotiate these formats — but an unreachable
path that kills a channel is worth one format query.

---

## GPU-direct recording: NVENC without the readback — 2026-07-30

With NVENC taking host BGRA, a recording made a full round trip: the channel read
the composited frame back, and NVENC uploaded the identical pixels again. The
consumer now copies the mixer's texture straight into a CUDA frame and hands that
to the encoder, and `needs_cpu_frame_data()` returns false while it does — so a
record-only channel skips its readback entirely.

**No colour-conversion kernel is involved, and none is needed.** NVENC accepts RGB
input (this is what the earlier BGRA-direct fix established), so a CUDA frames
context with `sw_format = AV_PIX_FMT_RGB0` takes the mixer's `GL_RGBA8` texels
byte-for-byte. That means no `.cu` file and no nvcc — just `cudart`, the driver
library and the CUDA runtime API.

### Why it is worth having at 4K and marginal at HD

The readback's cost scales with pixel count while the frame budget does not.
Measured from the OpenGL device's own estimate:

| format | MB/frame | readback | effective rate |
|---|---|---|---|
| 1080p | 7.91 | 2.95 ms | 2.62 GB/s |
| 2160p | 31.64 | **11.50 ms** | 2.69 GB/s |

Readback runs at ~2.7 GB/s, roughly 3.4× slower than the ~9.1 GB/s upload rate
measured elsewhere in this document — GPU→host is the less optimised direction.
At 4K the round trip is about 11.4 ms down plus 3.4 ms back up, per frame.

Server CPU, same encoder and settings, GPU path versus host path:

| | GPU-direct | host | saving |
|---|---|---|---|
| 1080p, 4 layers | 1.14 cores | 1.21 | 5.8 % |
| 2160p, 2 layers | 0.89 / 1.76 / 1.69 | 1.09 / 2.06 / 1.99 | **18.3 / 14.6 / 15.1 %** |

Three runs at 4K; absolute CPU varies between runs on this machine, the ~0.3 core
delta does not.

### Correctness

**Pixel-identical.** Recording the same fixed frame twice with `h264_nvenc`, once
GPU-direct and once with `-filter:v null` to decline it, gives **inf dB** between
the two files. Same encoder, same settings, only the frame path differs — which
is the control that matters, because comparing against `libx264` measures
encoder differences (28.8 dB) rather than the thing under test.

The log confirms the readback stops: `No consumer needs CPU readback (1
consumers); mixer readback skipped`.

### When it engages

All of these, each for a concrete reason:

| condition | why |
|---|---|
| encoder is NVENC | the only encoder here that takes device memory, and it accepts RGB |
| no user video filter | lavfi filters operate on host frames |
| 8-bit channel | a 16-bit channel's texture is RGBA16; NVENC's RGB inputs are 8-bit |
| OpenGL mixer | the Vulkan path needs `cuda_vk_texture` and is not built |
| CUDA present | — |

Anything else leaves the host path exactly as it was, and the reason is logged
whenever NVENC was selected. If the interop fails mid-recording the frame is
dropped, `gpu_direct` is cleared, and since `needs_cpu_frame_data()` is polled
every tick the channel resumes readbacks from the next one.

### Two things that crash rather than fail

Both cost a debugging cycle and are worth knowing before touching this code.

- **The CUDA context is not interchangeable.** FFmpeg's CUDA device context
  creates its own `CUcontext`, and device pointers are not valid across contexts —
  writing the encoder's frames through a pointer from another context killed the
  process with an access violation and no exception to catch. Asking FFmpeg for
  the primary context instead (`primary_ctx`) does not work either: the DeckLink
  DVP and CUDA ProRes modules activate it during startup, and FFmpeg then refuses
  with *"Primary context already active with incompatible flags"*. The copies
  therefore push FFmpeg's own context, via the driver API.
- **The GL registrations must be released deliberately.** They have to be undone
  on the mixer's GL thread and while the CUDA context is still alive. Leaving that
  to member destruction order does neither, and the process died at `REMOVE`
  rather than during recording. The uploader remembers the device on first use and
  `~ffmpeg_consumer` calls `release()` before the contexts go.

### Bit depth: an NVENC limit and a limit of this path, which are not the same

Worth stating precisely, because "GPU-direct is 8-bit" invites the wrong
conclusion. Measured on the reference rig by feeding each encoder `p010le`:

| encoder | 10-bit output |
|---|---|
| `h264_nvenc` | **no** — 8-bit only in hardware; reports *"No capable devices found"* |
| `hevc_nvenc` | **yes** — Main 10, produces `yuv420p10le` |
| `av1_nvenc` | unavailable on Pascal/Ampere — fails at 8-bit too, so this is the GPU generation rather than the depth |

So NVENC is not the constraint. The constraint is this path's byte-for-byte copy
of a `GL_RGBA8` texture into `AV_PIX_FMT_RGB0`, which is what makes it
kernel-free. A 16-bit channel's texture is RGBA16 and NVENC accepts no packed
16-bit RGB format matching it (`x2rgb10le` is 10 bits in 32; `gbrp16le` and
`yuv444p16le` are planar), so supporting it needs the conversion kernel the
design was built to avoid.

Ten-bit recording therefore works today, through the host path, from an 8-bit or
a 16-bit channel: `-vcodec hevc_nvenc -pix_fmt p010le`.

**A defect this exposed, fixed here.** An explicit `-pix_fmt` combined with
GPU-direct produced *no file at all*. The buffersink was narrowed to the
requested format while the graph carried CUDA frames, and configuration failed
with *"Impossible to convert between the formats supported by the filter
'Parsed_null_0'"*. An explicit pixel format is now a decline condition, alongside
a user filter, so the request is honoured through the host path instead of
breaking the recording. Verified: `hevc_nvenc -pix_fmt p010le` records
`yuv420p10le` from both an 8-bit and a 16-bit channel, and the ordinary
GPU-direct path is still pixel-identical to the host path at `inf` dB.

### Not done: Vulkan, and the reason is not the import path

The obvious assumption — that this only needs `cuda_vk_texture.h` in place of
`cudaGraphicsGLRegisterImage` — is wrong, and the attempt is worth recording so
nobody repeats it.

The import machinery is all present and correct: `CudaVkTexture` imports a
VkImage's memory as a `cudaArray_t`, `texture_wrapper` exposes the render
timeline semaphore for a GPU-side wait, and unlike OpenGL no context has to be
made current, so the copy can run on the consumer's own thread. That was built
and it compiled. It then failed at run time with:

```
[cuda_vk_texture] cudaImportExternalMemory: OS call failed or operation not
supported on this OS
```

**The mixer's composition target is not allocated with export capability.**
`create_exportable_texture()` exists but is used only by *producers* that create
their own textures to hand to the mixer (`cuda_prores`, `cuda_notchlc`, `ofx`,
`remotewall`). The mixer's own output comes from `pass->default_attachment()`,
out of the renderpass attachment pool, which allocates ordinary device memory.

Making that exportable is not a consumer-side change. It means allocating
attachments with `VkExportMemoryAllocateInfo` throughout the Vulkan mixer, and
exportable allocations generally cannot be sub-allocated by VMA — they need
dedicated allocations. So it changes the memory behaviour of every attachment on
every Vulkan channel, and needs its own validation of VRAM use and allocation
count before it could be trusted. That is the prerequisite, and it is the whole
of the remaining work; the consumer side is understood.

Until then the consumer declines on a Vulkan mixer with that reason logged, and
records exactly as before.

Harness: `CasparCG-TestRunner/vkdispatch/gpurec_check.py` and
`readback_scale.py`.

---

## Skipping the deinterlacer on declared-progressive streams — 2026-07-30

`bwdif` is appended to every video filter graph. With the default
`deint=interlaced` it only acts on frames flagged interlaced, so on progressive
content it does nothing — but it is still *in* the graph, so its format
constraints apply. It has no semi-planar support, so a hardware-decoded NV12
frame gets de-interleaved to `yuv420p` by libswscale on every frame, for a filter
that then passes it straight through.

It is now omitted when the container **explicitly** declares the stream
progressive (`field_order == AV_FIELD_PROGRESSIVE`) — `unknown` keeps it — and
only under `deint=interlaced`, since `deint=all` means the caller wants
everything deinterlaced. Containers do lie, so the `Decoder` records whether any
frame was actually flagged interlaced, and every filter rebuild consults it: a
mis-declared file gets its deinterlacer back and logs why.

**Measured: ~18 % CPU.** Twelve layers of hardware-decoded 1080p25 H.264, three
consecutive runs, bwdif present versus absent: 3.85 → 3.14, 3.84 → 3.17 and
3.89 → 3.17 cores.

> Measure this as a within-run A/B only. Absolute CPU for the identical
> configuration varied between 2.4 and 3.9 cores across sessions on this machine,
> so cross-run comparisons say nothing. And do not baseline against
> `auto-deinterlace=all`: that makes bwdif genuinely deinterlace every frame
> rather than pass through, which flatters the change to ~47 %. The honest
> comparison is against the *default* `interlaced` mode.

What now reaches the mixer, and what it exposed:

| source | before | after |
|---|---|---|
| H.264 4:2:0 8-bit (hw) | `yuv420p`, 3 planes | **`nv12`, 2 planes** |
| ProRes 4444 | `yuva444p16le` (promoted) | **`yuva444p12le`** |
| FFV1 planar RGB | `gbrp` | `gbrp` (see below) |
| QuickTime RLE RGBA | `gbrap` | **`rgba`** |

NV12 arriving as two planes is what the native semi-planar upload path was built
for; until now the filter graph had been undoing it for every software-transferred
hardware-decoded frame.

### Two latent mixer bugs this exposed

Both were invisible because the filter graph had always converted packed formats
to planar before the mixer saw them. Both are avoided by not offering the format,
and both are worth fixing properly at some point.

- **Packed ARGB/ABGR render wrong.** A QuickTime RLE clip, whose decoder emits
  `argb`, rendered at **6.4 dB** against a reference where every other format
  scores 38 dB or better. The shader swizzles (`pixel_format` 3 and 4,
  `.brga`/`.grab`) do not match either plausible upload convention. `rgba` is
  correct — the same clip scores `inf` once negotiation picks it.
- **The Vulkan mixer cannot take `rgb24` at all.** It throws the moment such a
  frame reaches it, which an FFV1 RGB clip did as soon as a CPU consumer was
  attached. This is a legitimate hardware limitation rather than a defect:
  Vulkan implementations are not required to support 3-component 8-bit formats
  as sampled images, and this one does not. OpenGL renders them correctly, so
  supporting them on Vulkan means expanding to 4 components during upload.

`AV_PIX_FMT_{RGB24,BGR24,ARGB,ABGR}` are therefore no longer offered to the
filter graph.

### Verification

Nine source types, both backends, channel output against a reference decode of
the same frame (static source, `SEEK` to a fixed frame):

| clip | arrives as | OpenGL | Vulkan |
|---|---|---|---|
| H.264 4:2:0 8-bit | `nv12` | 38.73 | 38.38 |
| H.264 4:2:0 10-bit | `yuv420p10le` | 39.62 | 39.19 |
| H.264 4:4:4 8-bit | `yuv444p` | 70.81 | 70.81 |
| ProRes 422 HQ | `yuv422p10le` | 39.79 | 39.72 |
| ProRes 4444 | `yuva444p12le` | 46.42 | 46.42 |
| DNxHR HQ | `yuv422p` | 40.01 | 39.92 |
| FFV1 RGB | `gbrp` | inf | inf |
| QuickTime RLE | `rgba` | inf | inf |
| v210 | `yuv422p10le` | 29.79 | 29.78 |

Deinterlacing unaffected: interlaced clips still score 39.02 dB (8-bit) and
39.12 dB (10-bit) against combed baselines of 29.33 and 28.94.

**v210's 29.79 dB is pre-existing, not a regression** — it measures identically
with the deinterlacer forced back in, and it arrives in the same format
(`yuv422p10le`) as ProRes 422, which scores 39.79. Same mixer path, same arrival
format, 10 dB apart, so the difference is in the source or the reference decode
rather than in anything changed here. Worth a look on its own.

Harness: `CasparCG-TestRunner/vkdispatch/quality_matrix.py` and `cpu_probe.py`.

Harness: `CasparCG-TestRunner/vkdispatch/upload_matrix.py` and `deint_check.py`.

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

---

## Measured: what skipping the readback actually buys — 2026-08-02

Baseline is the tree immediately before this work (`1ff94e989`), built with the same
toolchain and run against the same binary directory, so only the source differs.

**Scenario:** four 1080p50 channels, each with `<system-audio />` as its only consumer,
each playing a *moving* clip on loop (static content lets the still-frame cache skip
composition entirely and masks the whole test). Baseline reads back 4 × 8.29 MB × 50 =
1.66 GB/s; the current tree reads back nothing. 25 s warm-up discarded, 60 s measured,
GPU sampled twice a second on the mixer's adapter.

| | CPU (s/60 s) | GPU util | mem util | private MB |
|---|---|---|---|---|
| before, run 1 | 164.3 | 32.8% | 1.9% | 2833 |
| before, run 2 | 161.0 | 32.6% | 2.1% | 2856 |
| after, run 1 | 162.4 | 25.1% | 5.2% | 2747 |
| after, run 2 | 167.3 | 26.0% | 6.8% | 2757 |

**~22% less GPU utilisation** on the mixer's adapter, and about 90 MB less private
memory — consistent with not allocating readback staging buffers for four channels.

**Process CPU time does not move at all**, and that corrects how this was originally
justified. A readback is `glGetTextureImage` into a PBO followed by a fence wait that
*yields* rather than blocks; the transfer is done by the GPU's copy engine. So "415
MB/s" is a true statement about bytes and a false one about host cycles. The saving is
GPU time, PCIe bandwidth and VRAM, which is what buys headroom on a busy mixer — not
CPU.

It is also below the noise floor on a **single** 1080p50 channel: the first pass at
that scale showed 95.6 s vs 97.2 s CPU and no measurable difference. The effect scales
with pixels × channels × rate, so it is worth having on a multi-channel or 4K server
and is invisible on a small one.

`utilization.memory` rising rather than falling is reported as observed. It is a
coarse "fraction of time the memory controller was busy" counter, not a bandwidth
figure, and with overall GPU busy time down by a quarter the distribution it samples is
not comparable between the two runs. Not over-interpreted here.
