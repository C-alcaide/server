# OpenFX (OFX) Integration Plan — CasparVP

> Status: **Proposal / prototype plan**
> Target branch: `CasparVPV`
> Author: engineering
> Last updated: 2026-07-27
> Upstream standard: <https://github.com/AcademySoftwareFoundation/openfx> (OFX 1.5.x, BSD-3)

---

## 1. Executive summary

Add an **OpenFX host** to CasparVP so third-party `.ofx` plug-ins (blur, glow, keyers,
color, distortion, transitions, generators) can process CasparCG layers. OpenFX is the
de-facto VFX plug-in standard (Nuke, Resolve, Scratch, Baselight, Vegas), so adopting it
instantly exposes a large ecosystem of effects instead of us re-implementing each one.

**Verdict: feasible, medium-large effort.** The fundamentals align:

- OpenFX ships a reusable **`HostSupport`** C++ library (BSD-3, GPL-compatible) that does
  most host plumbing (bundle discovery, suites, param/clip/instance lifecycle).
- CasparCG `mutable_frame`/`const_frame` already expose **CPU-accessible pixel buffers**
  (`image_data()` arrays in [pixel_format.h](../src/core/frame/pixel_format.h) /
  [frame.h](../src/core/frame/frame.h)), which is exactly what the standard OFX render
  model needs.
- CasparVP's existing **keyframe timeline** maps naturally onto OFX animated parameters.

**Supported render modes (all three, first-class):**

| Mode | API | Platform | CasparVP path |
|---|---|---|---|
| **CPU** | Standard OFX render (`ubyte`/`ushort`/`float`) | Win + Linux | Producer path, CPU buffers |
| **OpenGL** | `OfxImageEffectOpenGLRenderSuiteV1` (standard) | Win + Linux | OGL mixer GL context, textures on GPU |
| **CUDA** | OFX GPU-render extension (`ofxGPURender.h`: CUDA-enabled render) | **Windows only** | CUDA device buffers + stream; reuses fork's `BUILD_CUDA_MODULES` toolchain (cuda_prores/cuda_notchlc); new CUDA–GL interop |

**Main friction point:** CasparVP compositing is GPU-first (Vulkan **and** OpenGL mixers),
while OFX has **no Vulkan** render path. We bridge this by supporting all three OFX-defined
modes: CPU is the universal baseline (producer path, frames on CPU before mixer upload,
no extra readback), **OpenGL** keeps GL-capable plug-ins on the GPU via the OGL mixer
context, and **CUDA (Windows only)** targets the high-end commercial plug-ins (Sapphire,
RE:Vision, etc.) by reusing the fork's **existing CUDA build system** (`BUILD_CUDA_MODULES`
auto-detected via `check_language(CUDA)`; `cuda_prores`/`cuda_notchlc` modules with CUDA
arch config, separable compilation, pinned-host allocator, DeckLink capture). The genuinely
new work is the OFX CUDA render suite + CUDA–GL interop, not the toolchain. Metal (macOS)
and Vulkan remain out of scope.

---

## 2. Background: how OpenFX works

| Concept | Meaning | CasparVP mapping |
|---|---|---|
| **Host** | The app that loads plug-ins and drives them | New `ofx` module = host |
| **Plug-in** | `.ofx` DLL bundle implementing effect actions | Loaded at runtime from a plug-in dir |
| **Suite** | Function table the host provides (Property, ImageEffect, Param, Clip, Memory, MultiThread, Message, Progress, TimeLine) | Implemented by host module |
| **Actions** | `describe`, `describeInContext`, `createInstance`, `render`, `beginSequenceRender`, `isIdentity`, `getRegionOfDefinition`, `getClipPreferences`, `destroyInstance` | Called from producer lifecycle |
| **Clip** | Named image input/output (`Source`, `Output`, optional `Mask`) | Bridged to CasparCG frames |
| **Contexts** | Filter, Generator, Transition, Paint, General, Retimer | Phase-gated (Filter first) |
| **Param** | double/int/bool/choice/RGBA/string/curve/pushbutton | AMCP + keyframe timeline |

**Pixel formats** OFX supports: `ubyte` (8-bit), `ushort` (16-bit), `float` (32-bit), in
RGBA / RGB / Alpha, premultiplied or unpremultiplied. CasparCG mixer frames are typically
`bgra` 8-bit (and 16-bit paths exist). A **channel-swizzle + premultiply-convention +
bit-depth bridge** is required.

**GPU render** in OFX (all supported here except where noted):
- **OpenGL** — the *standard* `OfxImageEffectOpenGLRenderSuiteV1`. Host provides a GL
  context; plug-in renders into a host-supplied GL texture. Cross-platform.
- **CUDA** — the GPU-render extension declared in `ofxGPURender.h` (the Nuke/Resolve
  convention now shipped with the OpenFX headers). Plug-ins advertise
  `kOfxImageEffectPropCudaRenderSupported`/`...CudaEnabled`; the host sets up the CUDA
  context/stream (`kOfxImageEffectPropCudaStream`) and passes **CUDA device pointers** as
  image buffers. **Scoped to Windows.** Reuses the fork's existing CUDA build system
  (`BUILD_CUDA_MODULES`, CUDA language + arch config); the new work is the OFX CUDA suite
  and CUDA–GL interop.
- **Metal** (macOS) and **Vulkan** — **not supported** (Metal out of scope; Vulkan is not
  defined by OFX at all).

---

## 3. Recommended architecture

Run OFX as a **`frame_producer` wrapper** (a "filter producer") that sits in front of a
source producer, mirroring the way ffmpeg filters work. The wrapper receives the source's
`draw_frame`, obtains an image buffer in the **negotiated render mode**, runs the OFX
`render` action, and emits a new frame. For CPU mode, producer frames are already on the
CPU **before** mixer upload, so no extra readback is introduced. For OpenGL/CUDA modes the
buffer stays on the GPU and is handed to the mixer as a texture where possible.

```
[ source producer ] --draw_frame--> [ ofx_producer (host) ] --draw_frame--> [ mixer ]
                                          |
                                          +-- render-mode negotiation (CUDA > OpenGL > CPU)
                                          +-- OFX plug-in .render() on CPU / GL tex / CUDA ptr
                                          +-- params <- AMCP / keyframe timeline
```

### Render-mode negotiation

The host advertises which modes it supports and inspects each plug-in's capability flags,
then picks the best available for that plug-in (preference order, configurable):
**CUDA (Windows) → OpenGL → CPU**. A per-effect/per-config override can force a mode (e.g.
pin to CPU for a misbehaving plug-in). Buffer handling per mode:

- **CPU** — `mutable_frame` `image_data()` buffers; swizzle/premult/bit-depth via bridge.
- **OpenGL** — allocate/borrow a GL texture in the OGL mixer's context; plug-in renders
  into it; the texture is attached to the output frame with no CPU roundtrip.
- **CUDA (Windows)** — allocate a CUDA device buffer and stream; register interop with the
  frame's GL texture (CUDA–GL interop); the plug-in renders on-device. Reuses the fork's
  existing CUDA toolchain/build option; CUDA–GL interop is the new piece.

### Proposed module layout

```
src/modules/ofx/
  ofx.cpp                    // module init(): registers producer factory (+ AMCP)
  ofx.h
  host/
    ofx_host.{h,cpp}         // OFX host built on openfx HostSupport
    ofx_plugin_cache.{h,cpp} // scan OFX_PLUGIN_PATH, cache descriptors
    ofx_suites.{h,cpp}       // any host-specific suite overrides
    ofx_message.{h,cpp}      // Message/Progress suite -> CasparCG log
  bridge/
    ofx_image_bridge.{h,cpp} // frame <-> OFX image: swizzle, premult, bit depth
    ofx_param_bridge.{h,cpp} // OFX params <-> image_transform/keyframes/AMCP
    ofx_clip.{h,cpp}         // clip wrappers for Source/Output/Mask
  render/
    ofx_render_mode.{h,cpp}  // capability detection + mode negotiation
    ofx_render_cpu.{h,cpp}   // CPU buffer backend
    ofx_render_gl.{h,cpp}    // OpenGL render suite backend (OGL mixer context)
    ofx_render_cuda.{h,cpp}  // CUDA backend (Windows only; #ifdef _WIN32 + CUDA)
  producer/
    ofx_producer.{h,cpp}     // frame_producer wrapper (Filter context first)
  CMakeLists.txt
```

### Wiring points (verified against current source)

1. **Module registration** — `src/modules/CMakeLists.txt` add `add_subdirectory(ofx)`;
   the module's `CMakeLists.txt` uses `casparcg_add_module_project(ofx SOURCES ...
   INIT_FUNCTION "ofx::init")` (same pattern as
   [artnet/CMakeLists.txt](../src/modules/artnet/CMakeLists.txt)). The generated
   `included_modules.h` (via `included_modules.tmpl`) then calls `ofx::init` automatically.
2. **`ofx::init(const core::module_dependencies&)`** registers a producer factory on
   `dependencies.producer_registry` (same as
   [image::init](../src/modules/image/image.cpp)) and, later, AMCP commands via the
   command repo.
3. **Frame creation** — use `frame_factory::create_frame(tag, pixel_format_desc[, depth])`
   ([frame_factory.h](../src/core/frame/frame_factory.h)) to allocate output buffers.
4. **Producer contract** — implement `receive_impl(video_field, nb_samples)` returning a
   `draw_frame`, plus `state()/print()/name()/is_ready()`
   ([frame_producer.h](../src/core/producer/frame_producer.h)). Hold the source producer
   internally and call its `receive`.

---

## 4. Implementation tiers

Tiers describe **capability depth**. Each phase (Section 5) advances one or more tiers.

### Tier 0 — Host foundation
- Load OFX bundles, enumerate plug-ins, read descriptors, manage instance lifecycle.
- Implement mandatory suites: Property, ImageEffect, Clip, Param, Memory, MultiThread,
  Message. CPU `ubyte` RGBA only. Filter context only.

### Tier 1 — Usable filter effects (MVP)
- End-to-end: source producer → OFX filter → mixer, on real HD content.
- Correct color: BGRA↔RGBA swizzle, premultiplied-alpha handling, `getClipPreferences`.
- Static parameters set at creation via AMCP.

### Tier 2 — Control & animation
- Full param bridge (double/int/bool/choice/RGBA/pushbutton).
- **Keyframe-timeline binding** so OFX params animate with playback position (reuse
  CasparVP keyframe engine).
- AMCP commands to add/remove/reconfigure effects live.
- TimeLine suite so temporal plug-ins know the current frame.

### Tier 3 — Depth & performance
- 16-bit (`ushort`) and 32-bit (`float`) image support; tie into fork's higher-bit-depth
  and color-managed paths.
- Host frame-threading / MultiThread suite for CPU parallelism.
- `isIdentity` / region-of-interest / region-of-definition optimizations; instance reuse.

### Tier 4 — Additional contexts
- **Generator** context → OFX generators as CasparCG producers.
- **Transition** context → hook into transition producer.
- Optional **Mask** clip from a second layer / key input.

### Tier 5 — OpenGL GPU render
- Implement standard `OfxImageEffectOpenGLRenderSuiteV1` bound to the **OGL mixer's GL
  context** so GL-capable plug-ins render on GPU with no CPU roundtrip.
- Texture allocation/attachment through the mixer; flush/fence synchronization.
- Mode negotiation prefers OpenGL over CPU when a plug-in supports it.

### Tier 6 — CUDA GPU render (Windows only)
- Implement the OFX CUDA render extension (`ofxGPURender.h`): advertise
  `kOfxImageEffectPropCudaRenderSupported`, provide `kOfxImageEffectPropCudaStream`, pass
  **CUDA device buffers** to `render`.
- **Reuse** the fork's CUDA build system (`BUILD_CUDA_MODULES`, `project(... LANGUAGES CUDA)`,
  arch config, separable compilation, pinned-host allocator) — the toolchain already exists.
- CUDA–GL interop so buffers stay on-device end-to-end (the new piece).
- Gated behind `#ifdef _WIN32` + CUDA toolkit availability; mode negotiation prefers CUDA
  first on Windows when supported.

### Tier 7 — Hardening & UX
- Out-of-process / crash-guarded plug-in hosting so a faulty `.ofx` cannot crash the
  server; timeouts and blocklist.
- Plug-in browser + parameter UI in `casparcg-360-client`.
- Config surface in `casparcg.config` (plug-in search paths, enable/disable, threading,
  **preferred/forced render mode per plug-in**).

---

## 5. Recommended phases

Each phase has a concrete goal, deliverables, and exit criteria.

### Phase 1 — Spike: host loads & renders one frame  *(Tier 0)*
**Goal:** prove the ABI + pixel bridge with a real plug-in.
- Vendor `openfx` via Conan/submodule; build `HostSupport`.
- Minimal host that scans a dir, loads a sample plug-in from the OpenFX repo `Examples/`
  (e.g. Basic/Invert), and renders one static `ubyte` RGBA frame.
- Dump result to disk for visual verification.
**Exit:** a known sample plug-in transforms one frame correctly (round-trip pixels match).

### Phase 2 — Filter producer MVP  *(Tier 1)*
**Goal:** OFX effect processes a live layer end-to-end.
- `ofx_producer` wraps a source producer; per-frame CPU render into a fresh frame.
- Correct swizzle/premult/`getClipPreferences`; handle non-multiple resolutions.
- Register producer factory; play via a syntax like
  `PLAY 1-10 [ofx] "com.vendor.effect" SOURCE [ffmpeg] "clip"`.
**Exit:** a real filter plug-in (e.g. blur) is visibly applied to playing HD video with
correct colors and alpha.

### Phase 3 — Parameters + keyframe animation  *(Tier 2)*
**Goal:** effects are controllable and animatable.
- Full param bridge; AMCP `MIXER OFX ...` (or dedicated `OFX PARAM ...`) commands.
- Bind params to the keyframe timeline; TimeLine suite reports current position.
- Live add/remove/reconfigure.
**Exit:** a param can be set via AMCP and animated over time via a keyframe track.

### Phase 4 — Depth, threading, optimization  *(Tier 3)*
**Goal:** production-grade quality and speed.
- 16-bit / float support wired into fork's color pipeline.
- MultiThread + host frame-threading; `isIdentity`/ROI/RoD; instance caching.
**Exit:** stable 50–60 fps HD with a representative CPU plug-in on target hardware, and
correct output at higher bit depths.

### Phase 5 — More contexts  *(Tier 4)*
**Goal:** generators & transitions.
- Generator context → producer; Transition context → transition hook; optional Mask clip.
**Exit:** an OFX generator plays as a producer and an OFX transition runs between layers.

### Phase 6 — OpenGL GPU render  *(Tier 5)*
**Goal:** keep GL-capable plug-ins on the GPU.
- OpenGL render suite integrated with the OGL mixer's context; mode negotiation adds GL.
**Exit:** a GL-render OFX plug-in runs without GPU→CPU→GPU roundtrips on the OGL path.

### Phase 7 — CUDA GPU render (Windows)  *(Tier 6)*
**Goal:** accelerate high-end commercial plug-ins on Windows.
- CUDA render extension via `ofxGPURender.h`; host CUDA context/stream; device buffers.
- CUDA–GL interop; reuse fork's `BUILD_CUDA_MODULES` toolchain; `#ifdef _WIN32` gating.
- Negotiation prefers CUDA → OpenGL → CPU (configurable/overridable).
**Exit:** a CUDA-capable OFX plug-in renders on-device on Windows; graceful fallback to
OpenGL/CPU when CUDA is unavailable or the plug-in lacks CUDA support.

### Phase 8 — Hardening & client UX  *(Tier 7)*
**Goal:** safe, discoverable, configurable.
- Crash isolation/timeouts/blocklist; config paths + per-plug-in render-mode selection;
  client plug-in browser + param UI.
**Exit:** a crashing plug-in is contained; users can browse/configure effects and choose a
render mode from the client.

---

## 6. Key technical challenges & mitigations

| Challenge | Impact | Mitigation |
|---|---|---|
| **Three render modes** (CPU/OpenGL/CUDA) to maintain | Complexity, testing surface | Shared bridge + thin per-mode backends behind a negotiation layer; CPU is always-available fallback |
| **CPU mode cost** | Extra CPU work; potential readback | Run OFX in producer path (CPU frames pre-upload); prefer GPU modes when available |
| **Pixel convention mismatch** (BGRA vs RGBA, premult, bit depth) | Wrong colors/alpha | Dedicated `ofx_image_bridge`; honor `getClipPreferences`; unit tests vs reference images |
| **CUDA–GL interop & context sharing** | Corruption/stalls if mismanaged | Reuse fork's CUDA build/toolchain (cuda_prores/notchlc); careful context/stream ownership + fences |
| **Performance at HD/UHD 50–60 fps** | Dropped frames | Prefer CUDA/OpenGL; MultiThread suite for CPU; `isIdentity`/ROI skips; per-layer opt-in |
| **Third-party native DLL stability** | A bad plug-in crashes server | Crash guard/timeouts (Phase 8); optional out-of-process host |
| **Vulkan unsupported by OFX** | No accel on Vulkan mixer | OFX effects use CUDA/OpenGL/CPU regardless of active mixer; document limitation |
| **Metal-only / OpenCL-only plug-ins** | Some plug-ins unavailable | Out of scope; negotiation falls back to CPU if the plug-in also supports it |
| **Threading vs CasparCG frame pipeline** | Contention/ordering | Confine plug-in calls to the producer's worker; respect OFX threading properties |

---

## 7. Licensing

- **OpenFX headers + HostSupport are BSD-3** → compatible with CasparCG's **GPLv3**.
- Commercial `.ofx` plug-ins are **separate runtime binaries** loaded by the host; this is
  the normal host/plug-in relationship and does **not** impose GPL on the plug-ins nor pull
  their licenses into our tree (no static linking of proprietary code).
- Bundle only the BSD OpenFX support code; ship no third-party plug-ins.

---

## 8. Dependencies & build

- Add `openfx` (headers + `HostSupport`) via Conan or a git submodule under `src/modules/ofx/`.
  Includes `ofxGPURender.h` for the CUDA/OpenGL render extension declarations.
- New CMake module target `ofx` with `INIT_FUNCTION "ofx::init"`; add to
  `src/modules/CMakeLists.txt`. Link the OGL accelerator for the OpenGL backend; on Windows,
  reuse the fork's `BUILD_CUDA_MODULES` toolchain (CUDA language + arch config) for the
  CUDA backend.
- Config additions (`casparcg.config`): plug-in search paths (`OFX_PLUGIN_PATH`),
  enable/disable, threading limits, per-plug-in blocklist, and **preferred/forced render
  mode** (`cuda` | `opengl` | `cpu` | `auto`).

---

## 9. Out of scope (initial)

- Full parity with every Resolve/Nuke GPU plug-in quirk (best-effort compatibility only).
- **Metal** (macOS) and **OpenCL** render backends.
- **CUDA on Linux** (Windows-only in this plan, matching the fork's CUDA build).
- Vulkan-accelerated OFX (not defined by the standard).
- Paint/Retimer contexts.
- Node-graph/timeline UI beyond simple per-layer effect stacks.

---

## 10. Quick reference — files to touch

- `src/modules/CMakeLists.txt` — add `add_subdirectory(ofx)`.
- `src/modules/ofx/**` — new module (host, bridge, render backends CPU/GL/CUDA, producer).
- `src/modules/ofx/CMakeLists.txt` — `casparcg_add_module_project(ofx ... INIT_FUNCTION "ofx::init")`;
  link OGL accelerator; reuse `BUILD_CUDA_MODULES` toolchain for the CUDA backend.
- AMCP: `src/protocol/amcp/AMCPCommandsImpl.cpp` — register `OFX`/`MIXER OFX` commands (Phase 3).
- Config parsing (shell/env) — plug-in paths, options, and render-mode selection (Phase 8).
- Client (`casparcg-360-client`) — plug-in browser + param UI + render-mode picker (Phase 8).
