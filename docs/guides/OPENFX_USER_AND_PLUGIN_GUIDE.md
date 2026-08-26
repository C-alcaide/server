# CasparCG OpenFX (OFX) — User Manual & Plugin Author Guide

> Module: `src/modules/ofx` · Companion: [OPENFX_IMPLEMENTATION.md](../architecture/OPENFX_IMPLEMENTATION.md)

This guide has two parts:

- **Part 1 — User Manual**: installing plug-ins, configuration, and driving effects over AMCP.
- **Part 2 — Plugin Author Guide**: how to write OFX plug-ins that run **correctly and efficiently**
  (including GPU zero-copy) inside CasparCG.

---
---

# Part 1 — User Manual

## 1.1 Installing plug-ins

CasparCG loads standard `.ofx` bundles. Point the host at your plug-in folders in either of two ways:

- **Config** (recommended) — one or more `<plugin-path>` entries in the `<ofx>` block (see 1.2).
- **Environment** — `OFX_PLUGIN_PATH` (semicolon/colon-separated), honoured in addition to config.

A bundle has the standard OFX layout:
```
MyPlugin.ofx.bundle/Contents/Win64/MyPlugin.ofx        (Windows)
MyPlugin.ofx.bundle/Contents/Linux-x86-64/MyPlugin.ofx (Linux)
```

On start-up the log lists what was discovered:
```
[ofx] OpenFX host initialised (image-effect API v1); discovered N plug-in(s).
[ofx]   com.vendor.effect (My Effect) v1.0
```

## 1.2 Configuration (`casparcg.config`)

```xml
<configuration>
  <ofx>
    <plugin-path>C:\OFX\Plugins</plugin-path>
    <plugin-path>D:\More\OFX</plugin-path>   <!-- may repeat -->
    <enable-opengl>true</enable-opengl>       <!-- OpenGL render backend -->
    <enable-cuda>false</enable-cuda>          <!-- CUDA render backend -->
    <blocklist>
      <plugin>com.vendor.crashy</plugin>      <!-- never instantiate these ids -->
    </blocklist>
  </ofx>
</configuration>
```

Environment overrides (useful for quick tests):

| Variable | Effect |
|---|---|
| `OFX_PLUGIN_PATH` | Extra plug-in search path(s). |
| `CASPARCG_OFX_ENABLE_GL` = `1` | Enable the OpenGL backend. |
| `CASPARCG_OFX_ENABLE_CUDA` = `1` | Enable the CUDA backend. |

> The GPU backends are opt-in. With both disabled, plug-ins run on the CPU.

## 1.3 Applying an effect (AMCP `PLAY` / `LOADBG`)

The producer token is **`[OFX]`** followed by the plug-in id.

![OFX producer modes: filter, generator, and transition](images/ofx_modes.png)

**Filter** (wrap a source):
```
PLAY 1-10 [OFX] "com.vendor.effect" AMB          # effect over the AMB clip
PLAY 1-10 [OFX] "com.vendor.effect" #FF808080    # effect over a solid colour
```

**Generator** (no source; plug-in must support the Generator context):
```
PLAY 1-10 [OFX] "com.vendor.generator"
```

**Transition** (blend two sources over N frames):
```
PLAY 1-10 [OFX] "com.vendor.dissolve" TRANSITION clipA clipB 50
```
`clipA` = `SourceFrom`, `clipB` = `SourceTo`, `50` = frames over which the `Transition` parameter ramps
`0 → 1` (default 25 if omitted).

## 1.4 Controlling parameters (AMCP `CALL`)

```
CALL 1-10 OFX LIST
CALL 1-10 OFX SET <name> <v...>          # numeric / bool(0|1) / choice(index)
CALL 1-10 OFX SETSTR <name> <text...>    # string parameters
CALL 1-10 OFX KEY <name> <frame> <v...> [tween]
CALL 1-10 OFX CLEARKEYS <name>
```

`LIST` returns machine-readable metadata (ideal for building UIs):
```
scale OfxParamTypeDouble "scale" dim=1 min=0 max=100 def=1
mode  OfxParamTypeChoice "Blend Mode" dim=1 def=2 choices="Normal","Add","Screen"
title OfxParamTypeString "Title" dim=1
```
- `dim` = number of components (RGBA=4, 2D=2, …).
- `min`/`max` = display range; `*` means unbounded.
- `def` = default (for choice, the default index).

Examples:
```
CALL 1-10 OFX SET scale 0.5
CALL 1-10 OFX SET color 1 0 0.5 1          # RGBA, 4 components
CALL 1-10 OFX SETSTR title "Breaking News"
CALL 1-10 OFX KEY scale 0 0                 # keyframe: value 0 at frame 0
CALL 1-10 OFX KEY scale 50 1 easeinoutsine # value 1 at frame 50, eased
```

## 1.5 Choosing the render backend

The host picks **CUDA → OpenGL → CPU** based on what the plug-in supports *and* what you enabled:

| Channel mixer | GL plug-in | CUDA plug-in |
|---|---|---|
| OpenGL (`<accelerator>opengl</accelerator>`) | **zero-copy** (core-profile) or compat fallback | CPU |
| Vulkan (`<accelerator>vulkan</accelerator>`) | compatibility GL + readback | **zero-copy** |

For maximum performance, match the channel accelerator to your plug-ins' GPU support (Vulkan mixer for CUDA
plug-ins, OpenGL mixer for GL plug-ins).

![How the host selects a backend: host-enable AND plug-in-advertises, then CUDA → OpenGL → CPU](images/ofx_backends.png)

## 1.6 Client integration (casparcg-360-client)

`ofx_params.py` parses `OFX LIST` and builds AMCP commands:
```python
from ofx_params import parse_ofx_list, set_command, OfxParamPanel
params = parse_ofx_list(list_response)          # -> [OfxParam(...)]
cmd = set_command("1-10", "scale", [0.5])        # "CALL 1-10 OFX SET scale 0.5"
```
An optional `OfxParamPanel` (PyQt6) builds sliders/checkboxes/combos/line-edits from the parsed metadata and
emits `command_ready` signals to send to the server.

## 1.7 Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Effect has no visible result, source shows through | Plug-in failed to instantiate or render (see log); it may be blocklisted after 3 strikes. |
| `... uses a non-core GL profile; using the compatibility render path` | Legacy fixed-function GL plug-in; runs correctly via the fallback (not zero-copy). |
| `... blocklisted; not instantiating` | The plug-in crashed/timed out ≥3×. Remove it or fix it; restart to clear. |
| Frames dropped, warnings about "render over time budget" | The plug-in exceeds the 2 s render budget. |
| GPU effect runs on CPU | Enable the backend (`enable-opengl`/`enable-cuda`) and match the channel mixer. |
| No plug-ins discovered | Check `<plugin-path>` / `OFX_PLUGIN_PATH` and the bundle layout. |

Operator best practices: keep plug-in ids explicit and quoted; test a plug-in on a spare channel before
air; prefer plug-ins that declare `isIdentity` (they cost nothing when neutral).

---
---

# Part 2 — Plugin Author Guide

This part explains what the CasparCG host supports and how to make your OFX plug-in **fast** here — most
importantly how to qualify for the **zero-copy GPU** paths.

## 2.1 What the host supports

- **API**: OpenFX Image Effect API v1 (OpenFX 1.5.1 HostSupport).
- **Contexts**: Filter, Generator, General, Transition.
- **Components**: `kOfxImageComponentRGBA` (and Alpha).
- **Pixel depths**: `kOfxBitDepthByte` (8), `kOfxBitDepthShort` (16), `kOfxBitDepthFloat` (32f).
- **Suites**: Property, Parameter, ImageEffect, Memory, Message, Interact, **MultiThread** (real, parallel),
  and the **OpenGL** and **CUDA** GPU-render suites/props.
- **Parameter types**: integer, double, boolean, choice, RGB, RGBA, double2D, integer2D, pushbutton,
  string, group, page.

## 2.2 The golden rules for efficiency

1. **Support a GPU render mode** and keep it clean so you qualify for zero-copy.
2. **Declare `isIdentity`** so neutral frames cost nothing.
3. **Stay within the 2-second render budget** — never block, never sleep, never do disk I/O on the render
   thread. Three overruns/crashes blocklist your plug-in.
4. **Never crash.** The host catches faults, but repeated crashes blocklist you.
5. **Honour premultiplication** — the host works in premultiplied alpha.

![The three OFX render paths and where a plug-in's GPU support lets it skip CPU read-back](images/ofx_dataflow.png)

## 2.3 Zero-copy OpenGL (OGL mixer)

To get zero-copy on the OpenGL mixer, your GL render **must use core-profile GL** (shaders + VAOs/VBOs/FBOs).
The mixer's GL device is a **core 4.5 profile** — fixed-function calls (`glBegin`, `glVertex2f`, `glOrtho`,
the matrix stack) are invalid there and will make the host fall back to a slower compatibility context +
readback.

Requirements:
- `describe`: `kOfxImageEffectPropOpenGLRenderSupported = "true"`.
- In `render`: check `kOfxImageEffectPropOpenGLEnabled`; fetch source/output via the OpenGL render suite
  (`clipLoadTexture`) — their `kOfxImageEffectPropOpenGLTextureIndex` are GL texture ids.
- The **host binds the output texture to an FBO and sets the viewport** before calling you; render into the
  currently-bound framebuffer.
- **Do not leak GL state or errors** — a lingering `glGetError()` is treated as "not core-clean" and forces
  the fallback. Reset any state you change.

Orientation/channels are handled by the host (it Y-flips and maps channels for the mixer). See
`test/coregl_orientation_test.cpp` for a minimal, correct example (it uses only `glScissor`/`glClear`).

## 2.4 Zero-copy CUDA (Vulkan mixer)

To get zero-copy on the Vulkan mixer:
- `describe`: `kOfxImageEffectPropCudaRenderSupported = "true"` (optionally
  `kOfxImageEffectPropCPURenderSupported = "false"` to force CUDA).
- In `render`: check `kOfxImageEffectPropCudaEnabled`; the images you fetch carry **CUDA device pointers** in
  `kOfxImagePropData` (with `kOfxImagePropRowBytes`). Run your kernel/`cudaMemset`/`cudaMemcpy` on those
  device pointers; call `cudaDeviceSynchronize()` (or use the provided stream) before returning.
- Link the CUDA runtime; a `.cu`/nvcc kernel is optional (runtime API alone works — see
  `test/cuda_fill_test.cpp`).

The host uploads the source to a device buffer, hands you the device pointers, and copies your device output
straight into the mixer's Vulkan texture (no CPU readback).

> **Current caveats to be aware of** (see the implementation doc): the CUDA path uploads the source
> bottom-up with **no Y-flip**, labels the output `bgra`, and does not pin the CUDA device to the Vulkan
> GPU. Spatial or channel-dependent CUDA plug-ins should be validated on a single-GPU host; a Y-flip/channel
> fix will follow once a kernel-based test plug-in exists.

## 2.5 CPU rendering & multithreading

- Always support CPU render (`kOfxImageEffectPropCPURenderSupported`) unless you are GPU-only by design; it
  is the universal fallback and the path used for 16-bit/float and transitions.
- The images you fetch carry host CPU buffers (`kOfxImagePropData` + `kOfxImagePropRowBytes`). Pixels are
  8/16-bit or float RGBA.
- Use the **MultiThread suite** for heavy CPU work: `multiThreadNumCPUs` reports up to 8, and `multiThread`
  runs your function across real worker threads (each guarded by the host). Split your render window by
  `threadIndex`/`threadMax`. Use the suite's mutexes for shared state.

## 2.6 Premultiplication, depth & components

- The host feeds **premultiplied** RGBA and expects premultiplied output by default. If your effect assumes
  straight alpha, request it via `getClipPreferences` (`kOfxImageClipPropPreMultiplication`); the host
  converts precisely.
- Support 8-bit to qualify for the zero-copy GPU paths; support 16-bit/float for quality (these use the CPU
  path today).
- Declare supported components/depths in `describe`; the host negotiates via `getClipPreferences`.

## 2.7 Transitions

- Declare the **Transition** context and define the mandatory `Transition` double parameter (0..1); the host
  sets it each frame.
- Fetch `SourceFrom` and `SourceTo`; output `From*(1-t) + To*t` (or your wipe/dissolve). See
  `test/transition_mix_test.cpp`.

## 2.8 Parameters that present well in CasparCG

- Set `kOfxParamPropDefault`, and **display** min/max (`kOfxParamPropDisplayMin`/`DisplayMax`) — these drive
  slider ranges in `OFX LIST`. Leave hard min/max unbounded only when appropriate.
- Give parameters clear `kOfxPropLabel`s and, for choices, `kOfxParamPropChoiceOption` labels — they are
  surfaced verbatim to clients.
- Implement `isIdentity` (return the identity input clip when your settings are neutral).

## 2.9 Efficiency checklist

- [ ] Declares a GPU render mode (**core-profile** GL and/or CUDA) **and** CPU fallback.
- [ ] GL: no fixed-function calls; no leaked GL state/errors.
- [ ] CUDA: operates on the provided device pointers; synchronises before return.
- [ ] Supports 8-bit (zero-copy) plus 16-bit/float for quality.
- [ ] Implements `isIdentity`.
- [ ] Uses the MultiThread suite for heavy CPU work; no blocking/I/O on the render thread.
- [ ] Renders within ~16–33 ms (well under the 2 s budget); never crashes.
- [ ] Honours premultiplied alpha (or declares straight via clip preferences).
- [ ] Provides labels, defaults, and display ranges for all parameters.

## 2.10 Minimal reference plug-ins

| Reference | Shows |
|---|---|
| `src/modules/ofx/test/coregl_orientation_test.cpp` | core-profile GL render + MultiThread suite probe |
| `src/modules/ofx/test/cuda_fill_test.cpp` | CUDA render using device pointers (runtime API only) |
| `src/modules/ofx/test/transition_mix_test.cpp` | Transition context: SourceFrom/SourceTo + Transition param |
| OpenFX SDK `Examples/Basic`, `Examples/Invert` | CPU filters with parameters |

Build them by enabling `BUILD_OFX_SAMPLE_PLUGINS` in the CMake configure; they are emitted into
`build/ofx-plugins/`.
