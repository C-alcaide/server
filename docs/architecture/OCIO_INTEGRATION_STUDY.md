# OpenColorIO in CasparVP — Design Study

**Status: study, and implemented.** Written 2026-08-11; status corrected 2026-08-16.

> **The integration is done on `feature/ocio-mixer`, both mixers, at parity.** Input
> transform, channel display transform, per-consumer views, custom configs and the
> interactions with the built-in grading chain are all in and measured.
>
> * **Using it:** [`OCIO_USER_GUIDE.md`](../guides/OCIO_USER_GUIDE.md) — commands, config elements,
>   refusal codes.
> * **What changed in rendered output, with the numbers:** `CHANGELOG.md`.
> * **How it is verified:** §5.4 below.
> * **This document** is the design rationale: why the first attempt could not have worked,
>   and why the shipped one is shaped as it is. Read it for *why*, not for *what is done* —
>   its banner said "in progress, Vulkan 4 batches of 6 through" for four days after the work
>   finished, which is the failure mode a status line in a design document invites.
>
> `OCIO_HANDOFF_2026-08-11.md` was the resume point while this was being built. It is gone,
> pruned 2026-08-16: every item in its "What is NOT verified" and "Open decisions" had been
> struck through, its defect accounts live in `CHANGELOG.md`, and its four durable pieces
> moved here — the gates to §5.4, the one remaining question to §7.9, and its sources to §9.
> `git log -- docs/OCIO_HANDOFF_2026-08-11.md` has the rest.

This document exists because OCIO was attempted once, on branch `origin/feature/ocio-support`,
and abandoned. It first establishes *why* that attempt could not have worked — the reasons
are structural and none of them are OCIO's fault — and then describes what a correct
integration would look like. External facts are current as of August 2026; sources are
listed in §9.

---

## 1. The previous attempt, and why it was doomed

Commit **`b304665b8`** "feat: add OCIO support via FFmpeg 8.x with local build"
(24 Feb 2026, on `origin/feature/ocio-support`, **not** in `CasparVPV` history). Four
files: `Bootstrap_Windows.cmake`, `av_producer.cpp`, `bluefish_producer.cpp`,
`ffmpeg_producer.cpp`.

The approach was to enable FFmpeg 8.x's OCIO **video filter** and expose it as producer
parameters — `OCIO_CONFIG`, `OCIO_INPUT`, `OCIO_OUTPUT`, `OCIO_DISPLAY`, `OCIO_VIEW` —
which were assembled into an `ocio=` filter string prepended to the producer's video
filter chain:

```
PLAY 1-1 clip.mp4 OCIO_CONFIG ocio://default OCIO_INPUT ACEScg \
    OCIO_DISPLAY "sRGB - Display" OCIO_VIEW "ACES 2.0 - SDR 100 nits (Rec.709)"
```

Four independent reasons that cannot work in a mixer:

**1.1 Wrong stage in the pipeline.** The filter sits per-producer, upstream of the mixer.
A display/view transform must be applied **once per channel, after compositing** — a
per-producer filter structurally cannot express it. Worse, the fork's own ACES chain then
runs downstream of it, so every graded layer passes through *two* colour pipelines in
series, each unaware of the other.

**1.2 It is a CPU filter.** The OCIO filter landed in FFmpeg **8.1**, and ASWF's own
encoding guidelines say plainly that "OCIO using the CPU is not super fast… with future
exploration of using vulkan to accelerate this." Running it per frame, per layer, at
channel cadence, on the same cores that are decoding, is the exact inverse of the
property `COLOR_GRADING.md` claims for the current chain: *"All processing runs on the GPU
in a single GLSL fragment shader pass, with zero CPU overhead per frame."*

**1.3 The ABI mix — this is the "memory space" problem.** `Bootstrap_Windows.cmake` was
changed to deploy MSYS2/MinGW64 runtime DLLs into an MSVC-linked process:
`libOpenColorIO_2_5`, `libImath-3_2`, `libminizip-ng-1`, `libyaml-cpp`, **`libstdc++-6`,
`libgcc_s_seh-1`, `libwinpthread-1`**. FFmpeg's own C ABI tolerates that. A C++ library
does not: two C++ runtimes with two heaps in one address space, objects allocated by one
and released by the other, incompatible exception machinery. That is undefined behaviour
by construction, and it is what "incompatible with FFmpeg's memory space" was describing.
**vcpkg ships an MSVC-native OCIO — this was entirely avoidable and must never be repeated.**

**1.4 Blast radius.** The change dragged a whole FFmpeg 7 → 8 major upgrade with it: the
buffersink API break in `av_producer.cpp` (`avfilter_graph_create_filter` →
`avfilter_graph_alloc_filter` + `avfilter_init_str`, because `pix_fmts` is no longer
settable post-init) and removed `AVFrame` fields in `bluefish_producer.cpp`. A colour
management feature that has to modify the Bluefish producer has been mis-scoped.

**The conclusion to carry forward:** OCIO was never the problem. The integration point
was. OCIO's value is on the GPU, at configure time, inside the mixer — the furthest
possible place from an `AVFrame`.

---

## 2. What OCIO 2.5 actually is (current facts)

| | |
| :--- | :--- |
| **2.5.0** | Vulkan support, built-in **ACES 2.0** configs, `GradingHueCurveTransform`, new colour-space attributes. In the **VFX Reference Platform CY2026**. Requires **C++17** (11/14 dropped) |
| **2.5.1** | Vulkan **texture binding indices reworked** (issue #2225) — **breaks ABI**. Pin ≥2.5.1 or the Vulkan path is wrong |
| **2.5.2** | **13 May 2026.** Fixes **CVE-2026-42450**: stack buffer overflow via unsafe `sscanf` in the `.spi3d`, `.spi1d`, `.cube` and `.lut` parsers. ABI-compatible with 2.5.1 |

**The CVE is not incidental to us.** CasparCG already ingests operator-supplied LUT files
(`MIXER LUT`). If OCIO ever parses one, **≥2.5.2 is mandatory, not advisory.**

### 2.1 The GPU model — the fact that changes everything

OCIO's GPU path is **a shader generator, not a pixel processor**:

* **Once per transform change** (CPU, off the frame path): `CreateShaderDesc(language, …)`
  → `processor->extractGpuShaderInfo(shaderDesc)` → out comes GLSL/HLSL/MSL **source
  text**, a list of 1D/2D and 3D **LUT textures** to upload, and a list of **uniforms** to
  bind. The application compiles the text into its own fragment shader and uploads the
  textures.
* **Per frame**: update `DynamicProperty` values and push the uniforms. **No CPU work per
  pixel, ever.**

That is exactly the shape of the existing mixer, and it is the opposite of what §1
attempted. Available languages include `GPU_LANGUAGE_GLSL_4_0` and — new in 2.5 —
**`GPU_LANGUAGE_GLSL_VK_4_6`**, with accessors for descriptor-set and texture-binding
indices and uniform-buffer offsets. The Vulkan mixer is therefore *supported*, not a
research project.

### 2.2 Built-in configs — nothing to ship

`ocio://default` resolves to the latest ACES 2 CG config; named URIs
`ocio://cg-config-v4.0.0_aces-v2.0_ocio-v2.5` and
`ocio://studio-config-v4.0.0_aces-v2.0_ocio-v2.5` pin exact versions. No YAML files, no
LUT directories, no install-path configuration. For a playout server this materially
reduces the operational surface — and pinning the URI makes the colour behaviour of a
build reproducible.

**The pin is now a default rather than an absolute — resolved 2026-08-14.**

This section previously recorded a finding: `accelerator::ocio::load_config(uri)` implemented
custom-config loading correctly and **had no caller**, so a studio with its own config had no
way to use it, and the 3D-LUT branch of both OCIO uploaders was unreachable code that could
never be measured — no built-in space emits a 3D LUT.

Both are resolved. `<ocio-config>` in the `<configuration>` block gives `load_config` its
caller (`shell/server.cpp:236`), taking a filesystem path or an `ocio://` built-in URI, and
refusing at startup rather than warning: loading keeps the previous config on failure, so a
warning would leave the server on the built-in config while the operator believed otherwise.
Omitted, the pinned URI is still what loads, so the reproducibility argument above survives
for every build that does not opt out.

That also made the 3D-LUT branch reachable, and **measuring it found it broken on OpenGL** —
`cli.py ocio-lut3d` drives a generated config whose one colour space emits exactly one 3D
texture; both mixers now agree at 6/6 within 1.0 LSB, worst 0.71.

There is still no `$OCIO` environment route, and that remains deliberate: the config comes
from the config file or the pin, not from the machine's environment.

Operator-facing detail — the element, its failure modes and the worked example — is in
[`OCIO_USER_GUIDE.md`](../guides/OCIO_USER_GUIDE.md) §6.1.

### 2.3 The substantive quality argument

The fork's chain is built on **ACES 1.x approximations**: `COLOR_GRADING.md` describes
`ACES_RRT` as "Stephen Hill's approximation of the ACES Reference Rendering Transform"
and the gamut compression limits as matching "the ACES 1.3 Gamut Compression reference
values". OCIO 2.5 ships **ACES 2.0, exactly**, as the reference implementation. ACES 2.0
changed the rendering transform substantially. That gap — approximation of 1.x versus
exact 2.0 — is the real reason to do this work, and it is not something more shader
tuning can close.

---

## 3. What OCIO would have to fit into

### 3.1 The current colour path

`MIXER COLORSPACE` → integer enums (`input_transfer`, `output_transfer`, `input_gamut`,
`output_gamut`) → an index into a hardcoded `k_direct[5][5][9]` table of 3x3 matrices,
set as a `mat3` uniform, plus branch-selected transfer functions inside one shader
([`image_kernel.cpp:505-671`](../src/accelerator/ogl/image/image_kernel.cpp#L505-L671)).

Everything colour is expressed as *uniforms into one fixed program*. OCIO does not
produce a matrix and an enum; it produces **source code**. That mismatch is the whole
integration problem, and §4.4 is the answer to it.

### 3.2 Blocker A — one process-wide shader singleton

[`image_shader.cpp:34-63`](../src/accelerator/ogl/image/image_shader.cpp#L34-L63): a
`std::weak_ptr<shader> g_shader` behind a mutex, compiled once from the build-time
embedded source and shared by every channel and layer in the process. The Vulkan side is
the same idea one step earlier — `glslc` runs at **build time**
([`accelerator/CMakeLists.txt:83-98`](../src/accelerator/CMakeLists.txt#L83-L98)) and the
SPIR-V is baked in.

OCIO requires *different shader text per transform*. This must become a keyed cache
(§4.4), and on Vulkan it additionally requires runtime GLSL→SPIR-V compilation, i.e.
linking **shaderc** — new to the build, though the Vulkan SDK is already a hard
requirement.

### 3.3 Blocker B — render targets are normalized integer on both mixers

* OGL: `GL_RGBA8` / `GL_RGBA16` only —
  [`texture.cpp:35`](../src/accelerator/ogl/util/texture.cpp#L35)
* Vulkan: `eR8G8B8A8Unorm` / `eR16G16B16A16Unorm`, with one pipeline pre-created per depth
  — [`device.cpp:542-543`](../src/accelerator/vulkan/util/device.cpp#L542-L543)

`COLOR_GRADING.md:558-561` already documents the consequence: *"Inputs strictly clip at
1.0… Super-white and negative values in the source are lost… Do not use Linear EXR or
scRGB sources."* Its stated workaround is to use log or PQ sources.

**Scene-referred ACES is precisely what that destroys.** AP0 and AP1 conversions produce
negative channel values as a matter of routine, and linear scene-referred data exceeds 1.0
by design. An ACES working space in a `UNORM` render target is not a compromise; it is a
contradiction. **This is the larger of the two architectural changes and a hard
precondition** (§4.3).

### 3.4 What already exists and helps — more than expected

* **3D LUT texture plumbing, both mixers.** OGL creates a `GL_TEXTURE_3D` with
  `GL_RGB32F` and re-uploads when the data pointer changes
  ([`image_kernel.cpp:900-904`](../src/accelerator/ogl/image/image_kernel.cpp#L900-L904));
  Vulkan has `eR32G32B32A32Sfloat` LUT images
  ([`image_kernel.cpp:743-799`](../src/accelerator/vulkan/image/image_kernel.cpp#L743-L799)).
  OCIO's `get3DTexture()` / `getTexture()` map onto this almost directly. The
  upload-on-change discipline is already the right one.
* **Per-format pipeline construction on Vulkan** (`device.cpp:542-543`) makes adding a
  float variant a localized change rather than a refactor.
* **Texture unit budget**: `plane0..3`, `local_key`, `layer_key`, `background`,
  `lut3d_tex`, `hue_curve_tex`, `blend_mask_tex` ≈ 10 in use. OCIO typically adds 1–4.
  Comfortable on desktop GL/Vulkan, but it should be a runtime assertion rather than an
  assumption.

---

## 4. The proper architecture

### 4.1 Principle

**OCIO generates shader code and LUT textures at configure time and never touches a
pixel.** Every per-frame cost is a uniform update. If a design puts OCIO anywhere near an
`AVFrame`, a `swscale` call, or a producer, it has reproduced §1.

### 4.2 The insertion points are already carved out — the working space is ACEScg

The single most important structural fact, and it was not obvious until the shader was read
directly: [`shader.frag:106`](../src/accelerator/ogl/image/shader.frag#L106) is

```glsl
uniform mat3  input_to_working;  // input gamut -> ACEScg (AP1)
```

**The working space is already ACEScg (AP1) linear**, and the chain
(`COLOR_GRADING.md:462-490`) is already three parts:

| steps | stage | space |
| :--- | :--- | :--- |
| 4–7 | EOTF decode + gamut → ACEScg | **front wrapper** |
| 8–21 | ~14 creative grading tools | scene-linear ACEScg |
| 27–29 | tone map + output gamut + OETF | **back wrapper** |
| 30 | film grain | display-referred |

That is exactly the shape OCIO needs. **OCIO replaces the front wrapper, the back wrapper,
or both, and the creative tools in the middle neither move nor change behaviour** — they
already operate on scene-linear ACEScg either way.

This retires the caveat that would otherwise have dominated the design: turning OCIO on does
*not* change what CDL slope, curves or saturation do, and stored look values remain
meaningful. Had the grading chain operated on display-encoded values, every one of them
would have needed reinterpretation.

The back wrapper's placement is still a real question: the display/view transform must be
applied once to the *composited* image, and there is no post-composite colour stage in
either mixer today (see §7 Q2 for whether it belongs to the channel or the consumer).

### 4.3 A float working space is a precondition — but "float" is a range change, not a precision upgrade

#### 4.3.1 Scope: five allocation sites, and `depth_` stops being a depth

Every intermediate in the OGL mixer is created at the channel's `depth_`
(`bit8`/`bit10`/`bit12`/`bit16`, [`bit_depth.h`](../src/common/bit_depth.h)):

| texture | when allocated | [`image_mixer.cpp`](../src/accelerator/ogl/image/image_mixer.cpp) |
| :--- | :--- | :--- |
| `target_texture` | always — the channel composite | `:259` |
| `layer_texture` | any layer with a non-normal blend mode | `:370` |
| `local_mix_texture` | MIX transitions / precomp | `:445` |
| `local_key_texture` | key masks (single channel) | `:434` |
| `cal_texture` | LED calibration LUT, final full-screen pass | `:264` |

…plus the Vulkan equivalents and its per-format pipelines
([`device.cpp:542-543`](../src/accelerator/vulkan/util/device.cpp#L542-L543)).

So the change is not "add one internal format". It is that **`depth_` stops being a bit
depth and becomes a pixel format** (`unorm8` / `unorm16` / `fp16` / `fp32`), threaded
through both mixers, both accelerators' texture allocators, and the consumer readback
paths that assume 1 or 2 bytes per component
([`image_mixer.cpp:836`](../src/accelerator/ogl/image/image_mixer.cpp#L836)).

#### 4.3.2 What this does **not** fix — do not justify it on grading accuracy

A natural but wrong argument for this change is "stacked grading tools will survive
better". They already do. The entire grading chain for one layer — CDL, curves, 3D LUT,
saturation, white balance, chroma, tone map, ~30 operations — runs in **one fragment
shader invocation**, in 32-bit float registers, and writes **once**. Verified against
`image_kernel::draw`, which sets every grading uniform on the single shader and issues one
draw per item; `COLOR_GRADING.md:544` states the same thing and is accurate.

**There is no inter-tool quantization to remove.** Stacking ten tools costs exactly what
stacking one costs.

The round-trips that *do* exist are the five textures in §4.3.1 — so grading is requantized
when a layer passes through a **blend mode**, a **MIX transition**, or the **calibration LUT
pass**. At `bit16` those are already negligible: one quantization is 1/65535, roughly six
bits below a 10-bit output step, so five of them are invisible. On a `bit8` channel they are
real — but the fix there is to configure 16 bits, not to go float.

#### 4.3.3 What it *does* fix — range

Two things `unorm` cannot represent at all:

* **values above 1.0** — a gain or lift that overshoots survives in-register and is then
  clipped at the write; linear EXR and scRGB sources are clipped on arrival
* **negative values** — produced routinely by wide-to-narrow gamut conversion and by
  anything touching AP0/AP1

This is why `COLOR_GRADING.md:558-561` instructs operators to avoid linear EXR and prefer
log or PQ sources. Float removes that restriction. **It is an enabling change for a
scene-referred working space, not an accuracy improvement to the existing tools.**

#### 4.3.4 fp16 is not a free upgrade — it redistributes precision

| signal level | `unorm16` step | `fp16` ulp | |
| :--- | ---: | ---: | :--- |
| near 1.0 | 1.53e-5 | 4.88e-4 | fp16 **32x coarser** |
| 1/64 (~1.6 %) | 1.53e-5 | 1.53e-5 | crossover |
| 0.1 % | 1.53e-5 | 9.5e-7 | fp16 16x finer |

Against output quantization: a 10-bit step is 9.8e-4, so fp16 near white is 2x finer —
acceptable. **A 12-bit step is 2.4e-4, so fp16 near white is 2x *coarser* than the output.**
`bit_depth` includes `bit12`, so on a 12-bit path a blanket switch to fp16 would be a
measurable highlight regression.

The resolution is that **the right format follows what the buffer holds**:

| buffer contents | correct format | why |
| :--- | :--- | :--- |
| display-encoded, [0,1] (today) | `unorm16` | the OETF has already redistributed precision perceptually; fp16 is strictly worse at the top |
| **linear scene-referred** (the ACES working space) | `fp16` | linear needs *relative* precision; fp16 holds a constant ~0.1 % everywhere, where `unorm16` gives 1.5 % error at 1 % signal — visible shadow banding |

`fp32` is the format that is safe regardless, at 2x the bandwidth of the current path: a
1080p intermediate goes 16.6 MB → 33 MB and 4K 66 MB → 133 MB, **per intermediate**, and
there can be five.

#### 4.3.5 Therefore: format follows working space

Make the format a consequence of the channel's colour configuration, exactly as §4.5 keeps
`k_direct` alongside OCIO. Display-referred channels stay `unorm16` and are **bit-identical
to today**; only a channel using a linear/ACES working space allocates float.

That constraint is what makes phase 0 provable: the existing batteries must come back
**byte-identical** on the unorm path — not "within 1 LSB", identical — while the float path
is exercised by new fixtures carrying negatives and values above 1.0, which the current
path cannot express at all. A blanket format switch could not be verified that cleanly,
because it would change every existing configuration's output at once.

### 4.4 Replace the shader singleton with a variant cache

Key the cache on OCIO's own processor/shader cache identifier rather than on a
hand-rolled hash of the parameters — OCIO exposes one precisely so applications can do
this, and it accounts for config contents, not just the space names.

Two rules that decide whether this is usable in production:

1. **Never compile on the frame path.** Build the variant on the command/config-change
   path, keep serving the previous variant until the new one is ready, then swap. Compiling
   inside `receive`/`draw` means every `MIXER OCIO` command drops frames.
2. **Bound the cache and evict.** A long-running channel that cycles transforms must not
   accumulate programs.

### 4.5 Coexistence — a per-stage contract, not free mixing

**Keep `k_direct` and make OCIO opt-in.** `k_direct` is not legacy debt to be retired: it is
the 1-LSB-gated path, it has zero dependencies, it is cheap on the GPU, and it is correct for
the common broadcast cases (709→709, 2020→709). ACES 2.0 exactness is needed for film-style
pipelines, not for every channel on the server. It is the fast path.

Given §4.2's three-part chain, coexistence is a **mutual-exclusion contract per stage**:

| stage | options | rule |
| :--- | :--- | :--- |
| front (input → ACEScg) | `k_direct` / OCIO input transform / auto-color-convert | **exactly one** |
| middle (grading) | the fork's tools | always, unchanged |
| back (ACEScg → display) | fork tone map + output gamut + OETF / OCIO display+view | **exactly one** |

There is already a precedent for exactly this shape: `COLOR_GRADING.md:494` documents
`MIXER COLORSPACE` and auto-color-convert as *"mutually exclusive — if `MIXER COLORSPACE` is
active, it takes priority."* Extend that rule to three-way and enforce it at command time.

**The thing to forbid explicitly:** never let OCIO handle the gamut while the fork handles
the transfer, or let an OCIO view transform (which contains the RRT/ODT) run with the fork's
tone mapping still enabled. Both produce a silent double transform, and the latter is the
failure the abandoned branch of §1 would have shipped.

Free mixing inside a stage is not a feature to be added later. It has no correct semantics.

### 4.6 AMCP surface

```
MIXER <ch>-<layer> OCIO <config-uri> <input-colorspace>            # input transform
MIXER <ch>          OCIO DISPLAY <display> <view>                  # output transform
MIXER <ch>-<layer> OCIO OFF                                        # back to k_direct
```

Colour space, display and view names stay **strings** — they are config-defined and
enumerating them into an enum would defeat the point of using configs. Validate at
command time against the loaded config and **fail the command**, never the frame. Default
the config URI to a pinned built-in (§2.2) so a bare `MIXER 1-1 OCIO` is reproducible
across installs.

**A discovery command is required, not optional.** The operator-facing surface goes from
~20 documented enums to hundreds of config-defined strings, and the mitigation is that the
client populates its own controls (§8.4). That only works if the client can ask the server
what the server actually has:

```
INFO OCIO                 -> config URI, OCIO library version
INFO OCIO COLORSPACES     -> enumerated colour space names
INFO OCIO DISPLAYS        -> displays, and the views available per display
```

OCIO exposes all of this from the loaded config (`getColorSpaceNames`, `getDisplays`,
`getViews`). Without such a query the client hardcodes lists, they drift from the server's
config, and a show file silently references a space that no longer exists. Returning the
library version in the same response also gives the client what it needs to warn about the
determinism concern in §8.2.

---

## 5. Verification

The 1-LSB harness in `d:\Github\CasparCG-TestRunner` cannot model OCIO (§4.5). What it
can do, and must:

1. **A model backed by OCIO's own CPU processor, inside the harness.** `PyOpenColorIO` is on
   PyPI as **`opencolorio`**, officially maintained by ASWF, currently **2.5.2 (13 May 2026)**,
   with **Windows x86-64 wheels** for Python 3.9–3.14. So the harness gains this with
   `pip install opencolorio` — **no C++ build, no `ocioconvert` subprocess.**

   This fits the existing structure exactly. `core/color_math.py` is a closed-form model of
   the shader; a new `core/ocio_reference.py` is an OCIO-backed model of the same shape —
   same interface, same `core/verdict.py`, same 1-LSB machinery — computing expectations via
   `config.getProcessor(...).getDefaultCPUProcessor()`. The batteries do not need to learn a
   new pattern; only the model source changes.

   **What this oracle does and does not prove.** It compares OCIO-CPU against
   OCIO-on-our-GPU, so any discrepancy isolates to **our integration** — the splice point,
   the LUT uploads, the texture bindings and descriptor sets, the channel-order trap, the
   float render target. That is precisely the class of bug worth hunting, and it is a strong
   oracle for it. It does not independently verify ACES 2.0 semantics; that is a trust
   decision, and it is the same one the tree already makes about FFmpeg's decoders and the
   GPU drivers (§8.1).

   **A tolerance must still be established empirically.** OCIO maintains GPU unit tests in
   `tests/gpu/` that check GPU output against CPU, so parity is a maintained property
   upstream — but it is not bit-exact, and GPU/CPU differences (including banding) have been
   reported against OCIO historically. The tolerance is therefore a **finding to be measured
   and recorded on a trivial transform first**, not a number to be guessed. Establish it
   before phase 2 depends on it.
2. **OGL ↔ Vulkan parity on identical OCIO transforms.** Required, per `CLAUDE.md`. Note
   the standing channel-order trap: OGL grades in BGR, Vulkan in RGB, and greys are
   invariant under a red/blue exchange — OCIO fixtures must use asymmetric per-channel
   values.
3. **Float-render-target regression.** After §4.3, the existing `conformance` and `grading`
   batteries must come back **byte-identical** on display-referred channels — not "within
   1 LSB", identical, because §4.3.5 keeps those channels on `unorm16` and any difference
   at all would mean the format selection leaked. This is the measurement that makes the
   render-target change defensible and it should be run and recorded *before* any OCIO code
   exists.
   On the float path the same batteries establish a new 1-LSB baseline; they cannot be
   compared to the unorm results, since the two formats quantize differently by design
   (§4.3.4).
4. **New fixtures the current path cannot represent**: patches carrying negative values
   and values above 1.0. Their existence is itself the test of §4.3.

### 5.4 The gates, in the order to run them

Migrated from `OCIO_HANDOFF_2026-08-11.md` when that file was pruned; it is the one part of
it that was still worth running rather than reading.

```bash
cd d:\Github\CasparCG-TestRunner
python cli.py conformance   --server d:\Github\CasparVP\build\shell\casparcg.exe --mixer ogl
python cli.py grading       --server d:\Github\CasparVP\build\shell\casparcg.exe --mixer ogl
python cli.py ocio          --server d:\Github\CasparVP\build\shell\casparcg.exe --mixer ogl
python cli.py vk-validation --server d:\Github\CasparVP\build\shell\casparcg.exe
# then --mixer vulkan for the first three. Parity is required, not optional.
python -m pytest tests/ -q     # includes the OCIO oracle and the vk-validation guards
```

Since that list was written the OCIO surface has grown its own batteries — `ocio-display`,
`ocio-gamut-compress`, `ocio-exposure`, `ocio-lut3d` and `consumer-view` — and they gate the
parts `cli.py ocio` does not reach. The harness's own `CLAUDE.md` is the current index; this
list is the minimum.

**Redirect to a file and grep. Do not pipe a battery through `tail`** — it races the summary
line. And check the run registry before starting: a battery competing with another session's
run dies mid-way with no error in its own log.

---

## 6. Phasing

| phase | deliverable | gate |
| :--- | :--- | :--- |
| 0 | `depth_` becomes a pixel format; opt-in float render target on both mixers, OCIO not involved | existing batteries **byte-identical** on unorm channels; new float baseline at 1 LSB; negative/super-white fixtures representable |
| 1 | shader variant cache replacing the singleton; no behaviour change | existing batteries unchanged; no frame drops on transform switch |
| 2 | OCIO input transform, per layer, OGL | A/B vs `ocioconvert` |
| 3 | Vulkan parity (`GPU_LANGUAGE_GLSL_VK_4_6` + shaderc) | OGL ↔ Vulkan parity |
| 4 | display/view transform, post-composite, per channel | A/B vs `ocioconvert` end-to-end |
| 5 | dynamic properties for interactive grading | no shader recompile on parameter change |

Phase 0 is worth doing on its own merits even if OCIO is never adopted: it makes the clamp
documented at `COLOR_GRADING.md:558-561` **opt-out per channel** rather than absolute, which
unlocks linear EXR and scRGB sources — and that is also what
[`IMAGE_SEQUENCE_AND_TIMELINE_PLAN.md`](../plans/IMAGE_SEQUENCE_AND_TIMELINE_PLAN.md) §4.2 needs for
`gbrpf32le` sequence input. **The two projects share this dependency**, which is a reason to
do phase 0 first regardless of which feature is ultimately wanted.

Note what phase 0 does *not* buy, so it is not oversold internally: no improvement to
stacked grading accuracy (§4.3.2). The gains are range, linear-source support, and shadow
precision in a linear working space.

### Dependency handling — the one non-negotiable

**vcpkg `opencolorio` ≥ 2.5.2, `x64-windows`, MSVC-native.** Never MinGW, never MSYS2
(§1.3). Transitive deps: `expat`, `imath`, `minizip-ng`, `yaml-cpp`, `lcms`, `openexr`,
`glew`. All must build under the MSVC **14.50-from-BuildTools** pin that nvcc 12.9 imposes
(see `BUILDING_WORKFLOW.md`) — verify this before writing any integration code, because it
is the step most likely to fail and the cheapest to test.

Side benefit: OCIO pulls in **OpenEXR**, which lowers the marginal cost of the OIIO option
in `IMAGE_SEQUENCE_AND_TIMELINE_PLAN.md` §3.5 tier 3.

---

## 7. Open questions

1. **Is ACES 2.0 exactness the actual requirement?** If the deliverable is "a defensible
   ACES pipeline" then yes and this study is the route. If it is "looks right on the wall",
   the existing 1.x approximations may already be sufficient and phase 0 alone (float
   working space) is the higher-value change.
2. **Does the display transform have to be per channel, or per consumer?** A channel
   feeding an LED processor and an SDI monitor simultaneously needs two different view
   transforms from one composite. That argues for the transform living in the consumer,
   not the channel — which is a larger change than §4.2 describes and should be settled
   before phase 4.
3. **What is the tolerance between OCIO GPU and `ocioconvert` CPU?** Unmeasured. It bounds
   every claim phases 2–4 can make, so it should be measured early on a trivial transform.
4. **Does the 14.50 pin accept the vcpkg OCIO dependency chain?** Untested, and it is a
   go/no-go for the whole study.
5. **Does the existing `MIXER LUT` path stay, or move to OCIO's `FileTransform`?** OCIO
   would bring better format coverage and, at ≥2.5.2, a hardened parser — but it also
   makes an operator-facing feature depend on the new stack.
6. ~~**`fp16` or `fp32` for the float path?**~~ **MEASURED 2026-08-12** —
   `CasparCG-TestRunner/cli.py banding`, and
   `CasparCG-TestRunner/docs/render_format_quantum_2026-08-12.md`.

   §4.3.4's arithmetic holds exactly. fp16's render-target quantum measures **32.0 LSB16
   near white** (LSB16 = 1/65535) — **2.0x a 12-bit output step**, 0.5x a 10-bit one — and
   halves per octave down the range precisely as `2^(e-10)` predicts: 32, 32, 16, 4, 1 at
   levels 0.9, 0.6, 0.3, 0.1, 0.02. unorm reads 1.0 at every level. Both mixers
   byte-identical.

   So fp16 is a real highlight regression against a **12-bit or better** output and
   comfortably fine for 10-bit and below. The regression is confined to the top two
   octaves.

   **Two corrections to the premise of this entry**, both from trying to run it as written:

   * **There is no `bit12` channel.** `<color-depth>` accepts only 8 or 16
     (`server.cpp:302-304`) and the channel depth is `color_depth == 16 ? bit16 : bit8`.
     `bit12` is only ever a *source* pixel format. §4.3.4's "on a 12-bit path" is an
     argument about a path no configuration produces — 12-bit is a **consumer** property,
     so the answer depends on the deepest consumer a float channel feeds.
   * **The measurement does not cover the five intermediates of §4.3.1.** It measures the
     composite that reaches the consumer. Open question 7 is untouched.

   **Still undecided, and this battery cannot decide it:** whether fp16's *shadow*
   precision is worth having — the half of §4.3.4 that argues FOR fp16, and the reason to
   want a float working space at all. At 0.002 fp16's ulp is an eighth of a 16-bit capture
   LSB, so a 16-bit unorm capture floors out and both formats read the same. Neither
   confirmed nor refuted; it needs a different instrument.
7. **Do the five intermediates all need the same format?** The calibration LUT pass
   (`cal_texture`) is the last thing before output and is display-encoded by then, so it
   may be able to stay `unorm16` even on a float channel — saving bandwidth at the point
   where it is most expensive. Unverified.
8. **Fix `parse_cube_file` now, independently of OCIO?** §8.5 found an unbounded allocation
   and silently ignored `DOMAIN_MIN`/`DOMAIN_MAX` in the existing `.cube` parser. Both are
   worth fixing whether or not OCIO is adopted — the domain bug means operator LUTs with a
   non-unit domain are being applied incorrectly today, with no diagnostic. Bounding the size
   is a two-line change.
9. **Does the GPU-versus-CPU agreement hold at 16 bit?** Every OCIO measurement to date is at
   8-bit capture depth — `(0,202,255)` against a model's `(0,202,255)`, delta `[0 0 0]`. A
   16-bit capture has 256× the resolution to disagree in, and nothing has looked. Carried
   over from `OCIO_HANDOFF_2026-08-11.md`, where it was the only open item left when that
   file was pruned.

---

## 8. Risk register

An earlier draft of this section listed eight downsides. Three were challenged on review and
**three do not survive investigation** — they are recorded here as retired, with the evidence,
because "we already considered that and it isn't a problem" is worth more than silence.

### 8.1 RETIRED — "verification gets weaker"

*Claimed:* the fork's 1-LSB credibility rests on a closed-form transcription of the shader;
ACES 2.0 cannot be transcribed, so OCIO paths are verified more weakly, forever.

*Why it does not stand:* it conflates two questions. Verifying **ACES 2.0 semantics** is not
our job — OCIO is the ASWF reference implementation and *is* the operative definition. Taking
it as correct is the same trust the tree already extends to FFmpeg's decoders, the Vulkan
driver and the C++ standard library, none of which have closed-form models either. What *is*
our job is verifying **our integration**, and for that OCIO-CPU-versus-our-GPU is a strong
oracle precisely *because* it is the same library: any difference is our bug by construction.
`PyOpenColorIO` makes this cheap (§5.1), and OCIO maintains CPU/GPU parity tests upstream.

*What remains:* the GPU↔CPU tolerance is not bit-exact and must be measured (§5.1). That is a
task, not a risk.

### 8.2 RETIRED — "output becomes a function of a third-party version"

*Claimed:* a security patch could change pixels on someone else's schedule.

*Why it does not stand:* the same is true of our own shaders — every commit that touches
`shader.frag` changes rendered output, which is exactly why `CHANGELOG.md` leads with
behaviour changes. OCIO adds one more version to a set that is already versioned, and the
mitigation is the one the tree already applies to FFmpeg: **pin it.** `ffmpeg-lib` is a
vendored external project pinned at 7.0.2 (`CASPARCG_EXTERNAL_PROJECTS`); OCIO would be pinned
the same way, with a built-in config URI (§2.2) pinning the transform data alongside it.

*What remains, as a requirement rather than a risk:* record the OCIO library version and
config URI in channel state / OSC and return them from `INFO OCIO` (§4.6), so a look is
reproducible and a mismatch is visible rather than inferred.

### 8.3 RETIRED — "cluster nodes desync"

*Claimed:* two nodes on different OCIO versions produce different pixels, seaming an LED wall.

*Why it does not stand:* it is 9.2 restated, and the premise already fails for the server
itself — a cluster running mismatched CasparCG builds is misconfigured regardless of OCIO.
Version pinning is an existing deployment invariant, not a new one.

### 8.4 REDUCED to a design requirement — operator surface

*Claimed:* ~20 validated enums become hundreds of config strings; a valid-but-wrong name
produces silently wrong colour.

*Correct handling:* the client populates its controls by enumeration, so the operator never
types a colour space name. This is standard division of labour and it works — but it is only
sound if the client can discover what the server has, which is why the `INFO OCIO` family in
§4.6 is a requirement of this design rather than a nicety. Server-side validation still fails
the command; the client's job is to make invalid input unreachable, not to be the only check.

### 8.5 REVERSED — LUT parsing attack surface

*Claimed:* routing operator LUT files through OCIO widens the attack surface, given
CVE-2026-42450.

*Investigation says the opposite.* The tree parses `.cube` itself, at
[`AMCPCommandsImpl.cpp:2624-2669`](../src/protocol/amcp/AMCPCommandsImpl.cpp#L2624-L2669),
and that parser is weaker than OCIO's post-2.5.2 one:

* **Unbounded allocation.** `lut->size` comes straight from the file and
  `data.reserve(size³·3)` follows with no bound check. `LUT_3D_SIZE 700` asks for ~16 GB;
  `2000` asks for ~96 GB. The AMCP dispatcher does catch (`AMCPProtocolStrategy.cpp:228`), so
  this is not a crash — but a live server can be driven into memory exhaustion before the
  throw arrives.
* **`DOMAIN_MIN` / `DOMAIN_MAX` are silently skipped** (`:2642`). A LUT authored with a
  non-unit domain — routine for log LUTs — is applied wrongly with no diagnostic. This is a
  correctness defect, and the more likely one to bite in practice.
* Throwing paths with no local handling: `std::stoi(line.substr(12))` throws on a bare
  `LUT_3D_SIZE` line or non-numeric size; `std::stof` on the strength argument (`:2709`) throws
  on garbage. Contained by the dispatcher, but they surface as opaque errors.
* It does *not* have the CVE-2026-42450 pattern — `sscanf` here is `%f` into floats, not `%s`
  into a fixed buffer.

**So OCIO's `FileTransform` would most likely be a security and correctness improvement.**
Two of those three defects are worth fixing on their own merits, independently of whether
OCIO is ever adopted — see §7 Q8.

---

### Risks that do stand

**9.6 GPU cost and register pressure — now the top technical risk.** OCIO's generated ACES 2
view transform is long and carries multiple LUT lookups, and it would be spliced into a shader
already executing ~30 grading operations. The plausible failure is reduced occupancy and an
fps drop on multi-channel 4K, which is exactly the configuration this fork exists to serve.
**Unmeasured, and measurable early**: generate the shader for a representative ACES 2 view
transform, splice it into the current shader, and profile before building anything else.

**9.7 Compile latency and warm-up.** Shader generation plus GLSL/SPIR-V compilation plus LUT
upload happens on a `MIXER OCIO` command, and ACES 2.0's DRT is heavy. §4.4's off-frame-path
cache handles steady state, but the *first* use of a transform still compiles, so channels
need a warm-up at init and the set of transforms to pre-warm is a configuration question.
Mitigating detail: `MIXER COLORSPACE` is **not** tweened — the trailing `2.0` in the
`COLOR_GRADING.md` examples is exposure, not a duration — so a colour change never has to
cross-fade two shader variants. Interactive grading parameters stay in the fork's tools and on
uniforms, so they never trigger a recompile either.

**9.8 Build-chain risk under the 14.50 pin.** Unchanged from §6, and still the go/no-go:
seven transitive vcpkg dependencies must build with the BuildTools 14.50 toolset that nvcc
12.9 forces. Cheapest thing to test, most likely thing to fail.

**9.9 Governance — resist migrating the grading tools.** OCIO 2.x offers
`GradingPrimaryTransform`, `GradingRGBCurveTransform`, `GradingToneTransform` and 2.5's
`GradingHueCurveTransform`. Migrating the fork's tools onto them would move tweened per-frame
parameters into OCIO's dynamic-property system and abandon the closed-form 1-LSB grading gate
— which, unlike §8.1, would be a real loss, because the grading tools *are* ours and *are*
transcribable. The §4.5 contract exists partly to make this boundary explicit.

**9.10 Two systems to document.** `docs/` is already large and kept current. Every colour
question acquires two answers, and `COLOR_GRADING.md` needs a clear statement of which path a
given channel is on.

### The risk that was never real

It costs nothing per frame. OCIO's GPU path is uniform updates at cadence (§2.1); steady-state
playout is as cheap as today plus the spliced shader's own arithmetic (§8.6). Every real cost
sits at configure time, in verification effort, or in operational discipline — not in
throughput.

---

## 9. Sources

* [OCIO 2.5 release notes](https://opencolorio.readthedocs.io/en/v2.5.0/releases/ocio_2_5.html)
  — Vulkan, ACES 2 built-in configs, API changes, C++17
* [OCIO GPU / shaders API](https://opencolorio.readthedocs.io/en/latest/api/shaders.html)
  — `GpuShaderDesc`, textures, uniforms, dynamic properties
* [OCIO releases](https://github.com/AcademySoftwareFoundation/OpenColorIO/releases)
  — 2.5.1 Vulkan binding rework / ABI break, 2.5.2 CVE-2026-42450
* [ASWF Encoding Guidelines — FFmpeg OCIO filter](https://academysoftwarefoundation.github.io/EncodingGuidelines/FfmpegOcio.html)
  — the filter is FFmpeg 8.1, CPU, "not super fast"
* [vcpkg `opencolorio` port](https://vcpkg.io/en/package/opencolorio.html)
  — MSVC-native builds, versions and dependency list
* [`opencolorio` on PyPI](https://pypi.org/project/opencolorio/) — official ASWF Python
  bindings, 2.5.2, Windows x86-64 wheels; the basis of the harness model in §5.1
* [OCIO `tests/gpu/GPUUnitTest.h`](https://github.com/AcademySoftwareFoundation/OpenColorIO/blob/master/tests/gpu/GPUUnitTest.h)
  — upstream GPU-versus-CPU parity test framework
* [Developing with OpenColorIO](https://opencolorio.readthedocs.io/en/latest/guides/developing/developing.html)
  — roles, display/view, processor cost
* [Colorspaces — `isData`](https://opencolorio.readthedocs.io/en/latest/guides/authoring/colorspaces.html)
* [OCIO 2.0 release notes — data space handling](https://github.com/AcademySoftwareFoundation/OpenColorIO/blob/main/docs/releases/ocio_2_0.rst)
* [OpenImageIO `colorconvert` / `unpremult`](https://github.com/OpenImageIO/oiio/blob/master/src/include/OpenImageIO/imagebufalgo.h)
* In-tree: `b304665b8`, `origin/feature/ocio-support`, `docs/guides/COLOR_GRADING.md`,
  `BUILDING_WORKFLOW.md`, `OCIO_USER_GUIDE.md`
