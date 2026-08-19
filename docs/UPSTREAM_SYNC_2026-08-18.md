# Upstream sync, 2026-08-18 — CasparVP onto `CasparCG/server` master, and FFmpeg 8.1.2

**Branch: `upstream-sync-ffmpeg8`, off `grading-nodes`.** Four merge commits, `6311fd623`
→ `32b40d4fc` → `a83d127be` → `bc94f4713`. Afterwards
`git rev-list --left-right --count HEAD...upstream/master` is **649 / 0**.

This document is the record of what was merged, what was deliberately NOT merged, what the
before-image was, and — the part most likely to matter later — the four things that broke
the build and how each presented, since two of them were caused by the merge strategy
itself and would recur on the next sync.

---

## 1. Where we had diverged

| | |
| :--- | :--- |
| Merge base | `35549a908`, 2026-03-09, upstream's *"chore: add ubuntu:26.04 arm to build matrix"* |
| Ahead / behind before the sync | **645 / 71** |
| Diff against `upstream/master` | 497 files, **+110,468 / −4,160** |
| Authors of our 645 | C-alcaide, all of them |

The 26:1 insertion ratio is the reason a merge was tractable at all: the fork is
overwhelmingly *additive*, so most of upstream's work lands in files we extended rather
than replaced.

**"71 behind" overstated the gap by more than half.** Of those 71:

* **19 were already ours** — 17 C-alcaide commits that upstream had merged (the
  `MIXER WHITEBALANCE / LIFT-MIDTONE-GAIN / HUESHIFT / TONEBALANCE / SPLITTONE / CDL`
  series, the BT.601 green-coefficient fix, the `glUniform3f` fix, the DeckLink keying
  check, the executor batch-command race) plus 2 merge commits for those PRs.
* **~10 were upstream's Vulkan accelerator series** (#1677), whose *content* we already
  had via `fedf6ce09`.
* **5 dependabot, ~8 CI/distro.**

Genuine gap: roughly **29 commits**, of which the FFmpeg 8 set was 5.

---

## 2. Why it was staged, and the staging itself

A single `git merge upstream/master` conflicts in **55 files** and conflates C++20, the
Vulkan merge and the FFmpeg pin. If a battery number then moved, nothing would say which
of the three moved it. Staged by upstream's first-parent history instead:

| Stage | Merged to | Conflicts | Carries |
| :--- | :--- | ---: | :--- |
| **M1** `6311fd623` | `89bcc9c02` | 6 files / 11 regions | FFmpeg 8 **source support**, transient ENOMEM, `ic_mutex` fix, AMCP emoji crash, NDI discovery. Pin still 7.0.2 |
| **M2** `32b40d4fc` | `8c7383783` | 19 / 33 | **C++20**, executor race, DeckLink keyer check, pragma cleanup |
| **M3** `a83d127be` | `d603ee91f` | 28 / 180 | upstream's **Vulkan accelerator** (#1677), `av_input` graph_ teardown |
| **M4** `bc94f4713` | `upstream/master` | 12 / 29 | **the FFmpeg 8.1.2 pin**, DeckLink fixes, OAL/screen, throttling |

Each stage is separately buildable and measurable, and the only stage that can change
rendered bytes — the pin — arrives alone.

---

## 3. The before-image, on one verified binary

`build/shell/casparcg.exe`, sha256
`60a02320b1dffad6ea9519c699360f149c9e22f2954610cf6c4c1c2debbb7fb9`, FFmpeg 7.x DLLs
(`avcodec-61`, `swscale-8`, `postproc-58` present). The hash was taken before and after
every arm and did not change, so all of it measures one build.

| arm | 8-bit | 16-bit |
| :--- | :--- | :--- |
| `conformance --mixer ogl` | **100/100** within 1.0 LSB | 0/100, worst **33.00 LSB** at `#FF00FF` |
| `conformance --mixer vulkan` | **100/100**, worst 0.55 at `#BFBFBF` | 99/100, worst 1.12 at `#4080BF` |
| `grading --mixer ogl` | **48/48** | 0/48 |
| `grading --mixer vulkan` | **48/48** | 46/48 |

**What this does not cover.** Only the IMAGE-consumer capture path, only flat colour
patches from a colour producer, and no decode: `conformance` and `grading` drive BGRA
patches with no producer, so neither reaches the YCbCr decode, the DeckLink path, the
Spout consumer or the still-image loader. `flat-decoded`, `sdi-output`, `sdi-input`,
`signalling` and `mixer-parity` were **not** run for this baseline and are owed.

**Two arms failed on the first pass** with server-startup errors — port 5254 refused, and
`template/ is not writable` — and were re-run clean with no orphan servers and no
permission problem remaining. Both were transient. Worth recording because on the first
read the surviving `vulkan_8` summary was mistaken for `ogl_8`'s: **check startup errors
per arm before believing a number, and never read a summary out of a concatenated log.**

### 3.1 A pre-existing finding this exposed, unrelated to the sync

OGL at **16-bit** fails everything (`conformance` 0/100, `grading` 0/48) where Vulkan at
the same depth is near-clean (99/100, 46/48). Both arms started cleanly, so it is real.

33 LSB16 is ≈ 0.13 LSB8, so the *pixels* are healthy and a flat 1.0 LSB gate at 16-bit is
simply ~256× tighter than the same gate at 8-bit — that part is a harness-usage artefact,
not a defect. What is **not** explained by gate tightness is the **backend asymmetry**: if
the gate were merely too tight, both mixers would fail it. It needs its own investigation.
The default 8-bit runs are structurally blind to it, which is why it had not been seen.

**Characterised further 2026-08-18, and it is now well bounded.** A second, independent
battery reproduces it, and four candidate causes are eliminated.

`image-convert` (new, §9 item 6) loads a flat still through the IMAGE producer and compares
against the value the file holds — no colour model, no conversion, no arithmetic:

| mixer | bit-depth | control (no swscale) | worst | verdict |
| :--- | :--- | ---: | ---: | :--- |
| ogl | 8 | 0.00 | 0.00 | PASS |
| ogl | **16** | **9.00 LSB16** | 10.00 | FAIL |
| vulkan | **16** | **0.00** | 0.00 | PASS |

So the same asymmetry appears on a still image as on `conformance`'s colour producer, which
makes it a property of the OGL path rather than of either fixture.

**Eliminated:**

* **Not swscale.** The `rgba8` control never reaches `convert_image_frame` and carries the
  same 9 LSB16. Excess over the control is +1.00 LSB, i.e. swscale contributes essentially
  nothing.
* **Not the render target's quantisation.** `banding` on the OGL mixer reports *"unorm
  control: 1.0 LSB worst across 5 levels, against unorm16's own 1.0"* — the unorm16 target
  resolves a true 16-bit step. The 32-LSB16 fp16 quantum in the harness's
  `render_format_quantum_2026-08-12.md` is a property of `<render-format>fp16`, and
  `<render-format>` defaults to `unorm` (`server.cpp:416`).
* **Not a stale or contended capture.** The exe hash did not change mid-run, and the clean
  serial re-runs reproduce it.
* **Not the decode.** `flat-decoded` passes on both mixers (§8.2), and this reproduces with
  no decoder in the path at all.

**What it looks like** is a small *offset or scale* error rather than a precision loss: 9
LSB16 on a nominal 52428 (0.8) is a relative error of 1.7e-4, too small to be any output bit
depth's step and too large to be 16-bit rounding. It is invisible at 8 bits (0.035 LSB8),
which is why every default-depth battery passes and why it went unseen.

**Hunted 2026-08-18, narrowed considerably, CAUSE NOT FOUND.** Recorded in full because two
plausible explanations were tested and disproved, and re-deriving them would be waste.

The exact 16-bit values, read with ffmpeg rather than PIL (PIL truncates 16-bit RGBA silently,
which cost one wrong conclusion here):

| variant | upload format | want R,G,B | OGL deviation | Vulkan |
| :--- | :--- | :--- | :--- | :--- |
| `rgba8` (no swscale) | `rgba` 8-bit | 52428, 39321, 26214 | **-8, +6, -9** | 0, 0, 0 |
| `rgb8` | -> BGRA 8-bit | 45746, 29555, 16448 | **-7, +5, -8** | 0, 0, 0 |
| `rgb16` | -> GBRAP16LE | 39321, 22873, 49087 | **+10, -3, -10** | 0, 0, 0 |
| `rgba16` | -> GBRAP16LE | 32896, 19532, 42662 | **+9, -2, -9** | 0, 0, 0 |

**Established:**

* **Spatially uniform.** Exactly ONE distinct value per channel across all 2,073,600 pixels.
  So not dithering, not the 64x64 -> 1920x1080 upscale, not sampling — a deterministic
  per-value transform.
* **Not a function of the value alone.** 0.6 appears twice: as `rgba8`'s G at **+6** and
  `rgb16`'s R at **+10**. That rules out a transfer-curve round trip and any scalar gain,
  both of which would map equal inputs to equal outputs.
* **It pairs by UPLOAD PIXEL FORMAT.** The two GBRAP16LE variants share a signature; the two
  8-bit packed variants share another. That points at the OGL upload/sample of a given
  `pixel_format`, not at anything downstream.
* **No colour conversion runs.** Confirmed at trace level, not inferred:
  `[ogl_kernel] NO_CONVERT frame #1 auto=false src_cs=1 src_ct=0 tgt_cs=1 tgt_ct=0 fmt=12`.
  So `do_input_convert` / `do_output_convert` are false and neither gamut matrix is applied.
* **Not swscale.** `rgba8` reaches the mixer without `convert_image_frame` and still deviates.
* **A colour producer through the same OGL path at 16-bit is accurate to 1.0 LSB16**
  (`banding`'s unorm control), where the image producer is 9 LSB16 out. So it is specific to
  the still-image upload rather than general to OGL at 16 bits.

**Disproved along the way, both of which looked convincing:**

1. *"The 16-bit arms secretly capture 8-bit."* PIL reports `mode="RGBA"` for 16-bit RGBA PNGs
   while truncating them. The IHDR of the same files says `bit_depth=16`. The captures are
   genuinely 16-bit.
2. *"It is a near-identity gamut round trip."* A single 3x3 fits the deviations with residuals
   of 0.39 LSB16 against deviations up to 10 — but that is 12 equations against 9 free
   parameters, so the fit proves far less than it appears to, and the trace above shows no
   matrix is applied at all.

**The next step, for whoever picks this up:** instrument the OGL upload and sample for
`pixel_format::rgba` and `pixel_format::gbrap16` specifically — read back the texel the shader
sampled and compare it against the byte that was uploaded. That isolates upload from render
from readback, which no battery here can currently do. Do NOT change the mixer before that
measurement exists; two hypotheses have already died in this paragraph.

### 3.1.1 CAUSE FOUND: swscale's BGRA64 -> RGBA64 permutation is lossy. FIX NOT LANDED.

**Cause, proven outside CasparCG entirely.** `image_consumer` hands `convert_image_frame` an
`AV_PIX_FMT_BGRA64LE` frame and asks for `AV_PIX_FMT_RGBA64BE` — a channel permutation plus a
byte swap. swscale gets the permutation wrong. One pixel, through `ffmpeg` alone:

```
bgra64le -> rgba64be   52428,39321,26214  ->  52420,39327,26205     LOSSY (-8,+6,-9)
bgra64le -> rgba64le   52428,39321,26214  ->  52420,39327,26205     LOSSY (identical)
bgra64le -> bgra64le   52428,39321,26214  ->  52428,39321,26214     exact (no-op)
rgba64le -> rgba64be   52428,39321,26214  ->  52428,39321,26214     exact (byte swap only)
```

The loss is in the **permutation**, not the endianness, and it is cross-channel: greys drifted
1-4 LSB16 while a saturated colour drifted 8-9, G gaining where R and B lost — the shape of a
round trip through a YUV intermediate.

**Why it looked like an OGL defect.** OGL reports `pixel_format::bgra`, Vulkan reports `rgba`.
So Vulkan's capture was a byte swap and exact; OGL's went through the permutation. It is
neither backend's fault — it is the capture path — and it is invisible at 8 bits (0.035 LSB8),
which is why every default-depth battery passes. Identical on 7.0.2 and 8.1.2, so it long
predates the FFmpeg 8 migration.

**Confirmed by instrumentation**, not inference: a `MIXER_OUT` trace in this consumer (kept,
at trace level) logs the frame as it arrives. For five colour-producer patches the mixer's
values were **exact** while every resulting PNG was deviated — which is what moved the fault
from the mixer to the capture path.

### Two fixes attempted, both REVERTED

| attempt | pixels | capture timing |
| :--- | :--- | :--- |
| hand permutation, swscale does the byte swap | exact | **28/36 conversions, heavy timeouts, deltas to 65535** |
| hand permutation AND byte swap, no swscale at all | exact — `image-convert` 4/4 on ogl16, vk16, ogl8; `conformance --quick` 36/36 worst 0.96, 0 timeouts | **full 100-conversion matrix: 18 timeouts, 85/89, worst 49780** |

Both were reverted. The second is the instructive one: it is *exactly right on pixels* and
still unshippable, because at full parallel load 18 of 2300 captures never completed inside
the harness's 10 s budget — and a capture that does not complete reports as a wild value, not
as a missing one. The pre-fix code has **zero** timeouts on that same matrix. Trading a
0.035 LSB8 error nobody can see for 18 lost captures is a bad trade.

`--quick` (36 conversions) showed 0 timeouts and passed cleanly, which is exactly why it was
not sufficient evidence: the defect it needed to expose only appears at the full matrix's
parallelism.

**The plan for landing it is [`PLAN_BGRA64_CAPTURE_FIX.md`](PLAN_BGRA64_CAPTURE_FIX.md)**,
which carries the design, the ruled-out cheap fix, the verification gates and the upstream
report. In short: Not a raw `std::vector` handed to the encoder. The by-hand
conversion should fill a properly allocated destination (`av_frame_get_buffer`, 64-byte
aligned, refcounted) rather than reusing the un-premultiply scratch buffer, since alignment and
refcounting are the plausible reasons the encoder slowed down. Then re-measure the **full**
matrix, not `--quick`, and require 0 timeouts as a gate alongside the LSB figure.

**Until then the pre-existing behaviour stands**: OGL 16-bit captures carry ~9 LSB16 of error
from this permutation, invisible at 8 bits, and `conformance --bit-depth 16 --mixer ogl` fails
by construction. It is not a regression from this sync.

**Out of scope for this sync — it is pre-existing and reproduces on the 7.0.2 binary too.**

---

## 4. What was deliberately NOT merged

### 4.1 Upstream's CEF shared-texture FRAME-IMPORT path (`2427604fa`)

A parallel implementation of this fork's own GPU-direct HTML path
(`html_gpu_bridge.{h,cpp}`, 768 lines, plus `ogl/util/dx_interop.cpp`, which already gives
the OGL device `interop_handle_` and `wglDXOpenDeviceNV`). Adopting upstream's half would
make `frame_factory::import_d3d_texture` a **pure virtual** that both mixers need stubs for,
to serve a path nothing in this fork calls; and it would change what the HTML producer
renders, which **no battery in `CLAUDE.md`'s table covers**.

The boundary actually drawn, after the first attempt at it failed to build (§6.1):

* **Adopted**: `accelerator/d3d/{d3d_device,d3d_device_context,d3d_texture2d}` — added to
  `accelerator/CMakeLists.txt` and compiled. Not optional: `html.cpp`'s
  `is_gpu_shared_texture_enabled()` probes the D3D device to decide whether CEF can hand
  over a shared texture at all, and without these it is an unresolved symbol at link time.
* **Deferred**: `frame_factory::import_d3d_texture` and every override of it, and
  upstream's `OnAcceleratedPaint` in `html_producer.cpp`.

**Owed:** reconcile the two GPU-direct HTML paths, or delete one. Until then these files
re-conflict at every sync.

**Investigated 2026-08-18 and decided — see
[`CEF_GPU_DIRECT_RECONCILIATION.md`](CEF_GPU_DIRECT_RECONCILIATION.md).** Keep this fork's
`html_gpu_bridge`: upstream's `import_d3d_texture` is a **throw stub on the Vulkan
accelerator** (`vulkan/image/image_mixer.cpp:389`) and nothing in their path checks the mixer
backend before requesting shared textures, so adopting it would mean a per-frame exception on
this fork's default `<accelerator>vulkan</accelerator>`. That document also carries the
comparison, the shape an upstream PR would take, and the harness the efficiency half of the
argument would need first — none of which is acted on.

### 4.2 Upstream's OAL consumer rewrite (part of `91a102022`)

`src/modules/oal/consumer/oal_consumer.cpp` was restored to its pre-M4 state. Upstream
replaced our `duration_`-based drift compensation with a `delay_`/`num_silence` latency
model; ours is 427 lines against upstream's 405 with 498 differing lines, and the two
cannot be interleaved a region at a time (§6.2 is what happens when you try).

The **screen** consumer half of `91a102022` IS taken, including its new `<delay>` option,
documented in `casparcg.config` per this tree's config rule.

**Owed:** a real merge of the OAL consumer, or a decision to keep ours permanently.

### 4.3 `/arch:AVX2` (upstream `2b97ac61d` drops it)

Kept at AVX2, noted in `Bootstrap_Windows.cmake`. `decklink/consumer/v210_strategies.cpp`
uses AVX2 intrinsics with **no runtime CPU dispatch in either tree**, so lowering the
baseline buys this fork no portability it can use — and `/arch:` changes the
auto-vectorisation baseline for every translation unit, which can move floating-point
results in the mixer. Revisit only together with a runtime dispatch, and measure the mixer
when you do.

### 4.4 Upstream's AMCP bind-address implementation (`168647e5b`)

Ours resolves through `tcp::resolver`, so it accepts a **hostname**; upstream's uses
`make_address`, which takes a numeric address only. Both reject IPv6. Taking upstream's
would have been a capability regression dressed as a sync, so our `(port, host)` signature
and `std::wstring` stay.

### 4.5 `AV_PIX_FMT_RGB24` / `BGR24` at the producer's filter sink

Upstream's `av_producer` sink list still offers both. Ours deliberately omits them: the
Vulkan mixer cannot sample a 3-component 8-bit image and `device::create_texture` throws
once per frame when offered one.

---

## 5. The FFmpeg 8 migration, as it actually turned out

**The pin block auto-merged.** No conflict resolution was needed for the migration itself:
`ffmpeg-8.1.2-full_build-shared.7z` with a SHA256, sonames `avcodec-62`, `avdevice-62`,
`avfilter-11`, `avformat-62`, `avutil-60`, `swresample-6`, `swscale-9`, and **no `postproc`
line** — libpostproc does not exist in 8.x.

**And none of the API work was strictly required by 8.1**, established from the headers
rather than the changelog:

* `av_opt_set_int_list` is gated `FF_API_OPT_INT_LIST (LIBAVUTIL_VERSION_MAJOR < 61)`, and
  8.1's libavutil is **60** — the macro still resolves.
* buffersink's `pix_fmts` / `color_spaces` / `color_ranges` are gated
  `FF_API_BUFFERSINK_OPTS (LIBAVFILTER_VERSION_MAJOR < 12)`, and 8.1's libavfilter is
  **11** — the old option names still resolve, marked `AV_OPT_FLAG_DEPRECATED`.

Upstream's guarded array-option code is therefore the **FFmpeg 9 bill paid early**. It was
taken anyway, because diverging on those regions means re-conflicting on them at every
future sync — not because 8.1 needs it.

Two details of the array form that are easy to get wrong, both now carried in comments:

* it takes a **COUNT** where the int-list form took a **TERMINATOR**, so the count must be
  read before any terminator is appended;
* `colorspaces` gets **1** element, not 2 — `AVCOL_SPC_UNSPECIFIED` was the terminator, and
  passing it as a member re-enables the unconstrained case the block exists to prevent.

---

## 6. What broke the build, and how each presented

Both stages M2 and M4 resolved most conflict regions to "ours". That rule is right when
ours is a superset, and it was verified per region — but it has a failure mode that no
per-region check can see: **a feature spans conflicting and non-conflicting hunks, so
resolving the conflicts to ours while the auto-merge brings upstream's other half leaves a
file that is half of each.** The build is the only thing that catches it.

### 6.1 `import_d3d_texture` — an `override` that overrode nothing

`error C3668: 'import_d3d_texture': the method with override specifier did not override any
base class method`, from four files. M2 resolved `frame_factory.h` to ours (no pure
virtual) while `core/mixer/image/image_mixer.h`, `accelerator/ogl/image/image_mixer.{h,cpp}`
and `html_producer.cpp` did **not** conflict and took upstream's half.

**This is worth reading carefully, because `CLAUDE.md` documents the identical message with
a different cause** — a stale precompiled header, producing "an `override` that did not
override a base method identical to it". Here the base method genuinely was not there.
Checking which of the two it was is the difference between deleting `*.pch` and fixing the
merge, and the wrong guess wastes a full rebuild.

### 6.2 `oal_consumer.cpp` — `duration_` and `num_silence` both undeclared

`error C2065` on two identifiers. The auto-merge replaced our member block (`duration_`)
with upstream's (`delay_`), while the conflict resolution kept our function body that uses
`duration_`; and upstream's silence-queueing loop arrived by auto-merge while its
`num_silence` declaration sat in a region resolved to ours. Neither side's code was wrong.
The *combination* was.

**Lesson for the next sync:** after resolving, grep the resolved file for identifiers the
kept side introduces and confirm each is still declared. A conflict region is not the unit
of a feature.

---

### 6.3 C++20 has a CUDA cost, and it is this fork's alone

Not a merge error — a genuine consequence of upstream's `f9fa5c342`, and the one thing in
this sync upstream cannot have hit, because upstream has no CUDA modules.

**Measured 2026-08-18, and the first description of this was imprecise.** The change was
originally made after seeing one error and pinning the standard, without establishing that
C++20 could not work. It has since been established, by compiling one real `.cu` twice with
the exact flags ninja used and only `-std` differing:

| `-std` | result |
| :--- | :--- |
| `c++20` | exit 2, **exactly one error** |
| `c++17` | **exit 0**, clean |

The error:

```
C:\...\MSVC.50.35717\include\chrono(5125):
    error C2760: syntax error: '}' unexpected here; expected 'expression'
```

It fires on `cuda_prores/consumer/prores_consumer.cu`,
`cuda_prores/consumer/prores_bypass_consumer.cu` and
`cuda_notchlc/cuda/notchlc_decode.cu`, which reach `<chrono>` transitively through
`common/log.h` -> Boost.Log.

**The construct, and whose fault it is.** `chrono:5125` is a C++20 **designated
initializer** — `return {.tm_sec = _Seconds, .tm_min = _Minutes, ...}` — inside a region
guarded by `#if _HAS_CXX20` (nearest guard at `chrono:2890`), which is why `-std=c++17`
never sees it.

It is NOT that `cl.exe` cannot compile that header at C++20: it does, in every ordinary
`.cpp` in this tree, since the whole C++ side is C++20 and `common/log.h` is everywhere.
The error arrives from `cl.exe` compiling nvcc's **generated intermediate**
(`tmpxft_*_notchlc_decode.compute_89.cudafe1.cpp`), so what fails is the nvcc
preprocess-and-rewrite step feeding cl.exe something it then rejects. A two-pass pipeline
limitation, not an MSVC one.

**One error, one construct, one header — not a class of failure.** An earlier draft of this
section implied the latter; the probe says otherwise. Other C++20 constructs could still
appear in other headers a `.cu` reaches, but that is an unquantified risk rather than an
observed one. `decklink` and `remotewall` also carry `.cu` sources and would follow.

**Fix applied:** `CUDA_STANDARD 17` / `CUDA_STANDARD_REQUIRED ON` in
`casparcg_add_module_project`, so every module's CUDA sources stay at C++17 while its C++
sources move to C++20. Set centrally rather than on the two targets that failed first,
because the other two have the same exposure.

**Why this is defensible:** it leaves the `.cu` translation units on exactly the standard
the whole tree used before this sync — the configuration they were last known to build
under — rather than inventing a third one.

**Why not simply move to a newer CUDA.** `CMAKE_CUDA_ARCHITECTURES` is `52;61;80;86;89`,
and `52`/`61` are Maxwell and Pascal. `nvcc --list-gpu-arch` on 12.9 still lists
`compute_50/52/53` and `compute_60/61/62`; **CUDA 13.0 removes those families**, so 12.9 is
the last toolkit that can target them and upgrading to escape this is a hardware-support
decision, not a build one. Note the flags already carry
`-Wno-deprecated-gpu-targets` — the toolkit is warning about those arches today, so this is
a hold rather than a settled position.

**The escape route, if the mixed boundary ever proves troublesome.** Because the failure is
one construct in `<chrono>`, reached only transitively via `common/log.h` -> Boost.Log, the
alternative to pinning the standard is to stop pulling `<chrono>` into `.cu` translation
units at all — a thin logging shim declared in a header with no Boost in it and defined in a
`.cpp`. That would let CUDA compile at C++20 and remove the mixed-standard boundary entirely.
It costs `CASPAR_LOG` inside `.cu` files and is a bounded refactor rather than a rewrite.
Worth doing only if the boundary below is measured to be a real problem; `CUDA_STANDARD 17`
is the cheaper answer while it is not.

**The risk this creates, and how much of it is now measured.** The link boundary is
mixed-standard and std types DO cross it: every CUDA module registers a consumer or producer
factory, so `std::wstring`, `spl::shared_ptr` and `boost::property_tree` pass between a
C++17 object and a C++20 one.

**Layout: measured, no difference.** A probe compiled twice with identical flags and only
`/std:` differing reports the same size and alignment for every crossing type, on the same
STL (`_MSVC_STL_VERSION=145`, `_ITERATOR_DEBUG_LEVEL=0`), with `_HAS_CXX20` flipping 0 -> 1
as expected:

| type | C++17 | C++20 |
| :--- | :--- | :--- |
| `std::string` / `std::wstring` | 32 / 8 | 32 / 8 |
| `std::shared_ptr<T>` / `weak_ptr<T>` | 16 / 8 | 16 / 8 |
| `std::unique_ptr<T>` | 8 / 8 | 8 / 8 |
| `std::vector<std::wstring>` | 24 / 8 | 24 / 8 |
| `std::map<std::wstring, int>` | 16 / 8 | 16 / 8 |
| `boost::property_tree::ptree` / `wptree` | 40 / 8 | 40 / 8 |

**What that does NOT settle**, and it is the half that cannot be settled statically: **ODR on
inline function bodies that differ under `_HAS_CXX20`.** Where a header defines a member
differently for the two standards, both object files carry a definition and the linker keeps
one arbitrarily. Layout parity makes that unlikely to matter for these types; it does not
prove it.

**Runtime: exercised, and it holds.** `cli.py run --decoder cuda_prores --mixer vulkan
--codec prores_422` over 24 cases on the 8.1.2 build: the module loads, registers its producer
factory, and decodes. So `std::wstring`, `spl::shared_ptr` and `boost::property_tree` do cross
the C++17/C++20 boundary in real use — factory registration and AMCP-driven producer creation
both traverse it — without corruption or crash.

4 of those 24 cases fail at PSNR 24.5, tagged `[decoder]`. **That is pre-existing, not the
sync:** the same matrix on the PRE-SYNC 7.0.2 binary (C++17 throughout, no mixed boundary)
fails 5 of 24 at the identical PSNR and the identical fault stage. The failing sets only
partly overlap, so it is intermittent rather than deterministic — a separate cuda_prores
defect worth its own investigation, and explicitly not evidence about the standard mismatch.

So the residual risk is now narrow: layout is measured identical, the boundary is exercised
behaviourally, and what remains unproven is only ODR on `_HAS_CXX20`-conditional inline
bodies, which no runtime pass can rule out. Still owed: `--decoder cuda_notchlc`, and
`sdi-output` for the DeckLink `.cu` readback paths. Revisit if CUDA ships a C++20 host path,
or take the escape route above and remove the boundary entirely.

---

### 6.4 Upstream's `d3d_texture2d` needed one accessor we already had

`accelerator/d3d/d3d_texture2d.cpp` calls `ogl->d3d_interop()`, one of the two accessors
stage M2 dropped when it resolved `ogl/util/device.{h,cpp}` to ours. The underlying handle
was never missing — `device::impl` has held `interop_handle_` since `ogl/util/dx_interop`
landed — only the public accessor was.

Added as a two-line `#ifdef WIN32` accessor returning `impl_->interop_handle_`. That is what
makes §4.1's boundary possible at all: upstream's D3D files compile **unmodified**, which
is what keeps future syncs clean, while their frame-import path stays unadopted.

### 6.5 The whole grading command block landed TWICE, and half of it would have shipped

The worst of the four, because only half of it was a compile error.

`AMCPCommandsImpl.cpp` **did not conflict** in any stage. Our grading commands sit scattered
through the file (added over time, interleaved with other commands) while upstream's arrived
as one contiguous block, so git saw two independent insertions rather than a conflict and
kept **both**.

What that produced:

| Duplicated | Consequence |
| :--- | :--- |
| `grade_param`, `grade_require` | `error C2084: function already has a body` — **loud** |
| `mixer_whitebalance/lift/midtone/gain/hueshift/tonebalance/splittone/cdl_command` (8) | `C2084` again — **loud** |
| `register_channel_command(L"MIXER WHITEBALANCE", …)` and the other 7 | **SILENT.** Compiles, links, runs, and registers every one of the eight commands twice |

The registration half is the part worth remembering. It is not a compile error, not a
warning, and not something any battery in `CLAUDE.md`'s table asks about — `grading` drives
the commands and checks the pixels, and a command registered twice still produces the right
picture. It was found only by grepping registrations for duplicates *after* the definition
errors pointed at the file, and the natural stopping point was one step earlier: fix the
C2084s, watch the build go green, ship a repository with sixteen entries where there should
be eight.

**Resolution:** removed upstream's copy of all three (303 lines of definitions plus the 8
registrations), keeping ours — consistent with §7, since upstream's versions are our own
PRs squashed. Upstream's `mixer_rgb_triple_command` template, which factors lift/midtone/gain
into one implementation, is a genuine simplification and is **not** adopted here; taking it
belongs in its own change, measured against `grading`.

**Generalised check for the next sync**, because two of the four break-causes were this same
shape: after resolving, for every file that carries work upstream also merged, grep for
duplicate *definitions* AND duplicate *registrations / table entries*. A file that did not
conflict is not a file that merged cleanly. Verified this way for the mixer files and both
shaders: `image_kernel.cpp` (both backends), `transforms.cpp` (both), `frame_transform.cpp`
and `fragment_shader.frag` have no duplicates. `ogl/image/shader.frag` does have 5
duplicated uniform names, but identically before and after the sync — pre-existing, not
merge-induced, and out of scope here.

---

---

## 7. Resolutions that would have caused a silent regression

These built fine either way. They are here because "it compiles" would have been the only
signal.

* **`shader.cpp`** — upstream's side is the **pre-fix** `glUniform3f`. Ours carries
  `static_cast<float>(value2)  // fixed: was value1`, which is `46124f779`, our own PR,
  plus a 4-component `set()` overload upstream lacks. Taking upstream's would have
  reintroduced a bug this fork reported and upstream merged.

* **`image_kernel.cpp` split tone.** Upstream uploads `split_shadow_color` as
  `sc[0], sc[1], sc[2]`; ours as `sc[2], sc[1], sc[0]`. That is `34aae3cb2`, which moved
  every swizzle out of the kernel into the shader. **Our shader has two
  `apply_split_tone` call sites that disagree on purpose** —
  [`shader.frag:1979`](../src/accelerator/ogl/image/shader.frag#L1979) passes the uniform
  straight, `:2153` passes `.bgr`, because the first runs where the pixel is BGR and the
  second where it is RGB. A kernel-reversed uniform is correct at **both**. Upstream's
  straight-through upload would have left `:1979` unswapped on a BGR pixel and `:2153`
  double-swapped — and **no grey patch can see either**, which is `CLAUDE.md`'s
  channel-order trap exactly.

* **`accelerator.cpp`** — resolving its six regions to ours left a brace delta of **+1**:
  git had put a boundary inside `set_backend`'s body and then treated eight lines of
  upstream's `get_device()` as *common context* in the middle of it, so "keep ours" spliced
  half of one function into another. Caught by checking brace balance before accepting the
  resolution. Rebuilt as our whole file, after confirming every upstream-only line in it is
  the single-`device_` variant our per-GPU `std::map<int, ...> devices_` replaces.

---

## 8. Verification status

| What changed | Covering battery | Run? |
| :--- | :--- | :--- |
| FFmpeg pin — capture-side swscale | `conformance` + `grading`, both mixers | **before-image only** |
| FFmpeg pin — decode path | `flat-decoded`, `sdi-input`, `source-colorspace` | **no** |
| FFmpeg consumer pixels / metadata | `signalling --stream`, `sdi-output` | **no** |
| C++20 codegen across the tree | `conformance` + `grading` + `mixer-parity` | **before-image only** |
| accelerator code (M3) | `vk-validation` | **no** |
| swscale in `image_producer` still-load | **nothing** — see `FFMPEG_8_MIGRATION.md` §9 | n/a |
| swscale scaling in `spout_consumer` | **nothing** | n/a |

**Nothing after the merge has been measured yet.** The build is the current gate; the
after-image is owed and is the next step.

`FFMPEG_8_MIGRATION.md` §5.1 was corrected as part of this work: it claimed swscale was
confined to two call sites with "neither in the 1 LSB path", and that `conformance` and
`grading` "cannot detect a regression here". `convert_image_frame` has **four** callers and
one is the **IMAGE consumer** ([`image_consumer.cpp:310`](../src/modules/image/consumer/image_consumer.cpp#L310),
[`:333`](../src/modules/image/consumer/image_consumer.cpp#L333)), which runs it on every
captured frame and takes the `SWS_ACCURATE_RND | SWS_FULL_CHR_H_INT` branch at 16-bit. The
swscale rewrite is therefore *inside* the tightest gates in the harness, not outside them —
and because it is common to every battery and to both mixers, a regression there moves
every number at once and survives every parity check. `sdi-output` and `signalling` bypass
`convert_image_frame` and are the probes that separate a capture fault from a mixer fault.

---

## 8.1 The after-image: FFmpeg 8.1.2 moved nothing that is gated

`build-sync/shell/casparcg.exe`, sha256
`511ffafce7bde79063b933e88d15d3903c218c9bd6e2263f2fc70aff80a64fc5`, unchanged before and
after the run. Same eight arms as §3, same commands, so the numbers compare arm for arm.

| arm | 7.0.2 | 8.1.2 | |
| :--- | :--- | :--- | :--- |
| `conformance --mixer ogl --bit-depth 8` | 100/100, worst 0.55 at `#BFBFBF` | 100/100, worst 0.55 at `#BFBFBF` | same |
| `conformance --mixer vulkan --bit-depth 8` | 100/100, worst 0.55 at `#BFBFBF` | 100/100, worst 0.55 at `#BFBFBF` | same |
| `conformance --mixer ogl --bit-depth 16` | 0/100, worst 33.00 at `#FF00FF` | 0/100, worst 33.00 at `#FF00FF` | same |
| `conformance --mixer vulkan --bit-depth 16` | 99/100, worst 1.12 at `#4080BF` | 99/100, worst 1.12 at `#4080BF` | same |
| `grading --mixer ogl --bit-depth 8` | 48/48 | 48/48 | same |
| `grading --mixer vulkan --bit-depth 8` | 48/48 | 48/48 | same |
| `grading --mixer ogl --bit-depth 16` | 0/48 | 0/48 | same |
| `grading --mixer vulkan --bit-depth 16` | 46/48 | 46/48 | same |

Pass counts matching is the weak version of this claim. The strong version: comparing the
**per-conversion** result lines — each carrying its verdict, its worst LSB figure and the
patch that figure occurred at — across the four `conformance` arms gives **800 results and
zero differences**. Not "within tolerance": identical, including the pre-existing OGL 16-bit
failures reproducing at exactly 33.00 LSB at exactly `#FF00FF`.

So the swscale re-architecture did not move a gated number through
`convert_image_frame`, on either flag path — `flags=0` at 8-bit and
`SWS_ACCURATE_RND | SWS_FULL_CHR_H_INT` at 16-bit — on either mixer. Which is the answer one
would predict for two conversions that are channel permutations and byte-order swaps with no
resampling and no arithmetic, and it is now measured rather than predicted.

**What this does NOT establish**, and the list is longer than the result:

* **No decode ran.** `conformance` and `grading` drive a BGRA colour producer, so the YCbCr
  path, `flat-decoded`, `sdi-input` and `source-colorspace` are all untouched. FFmpeg 8's
  decoders are the largest part of the change and none of it was exercised.
* **No consumer but IMAGE.** `sdi-output`, `signalling`, `signalling --stream`,
  `consumer-view` — none run. The FFmpeg *consumer* is where the array-option migration of
  §5 actually lives, and nothing here touched it.
* **The two uncovered swscale sites are still uncovered** (§9): the still-load conversions in
  `image_producer`, and the Spout `SWS_FAST_BILINEAR` downscale, which is the only site that
  resamples and therefore the one most exposed to a rewrite.
* **The mixed-standard CUDA link boundary is untested** (§6.3). Nothing ran
  `cuda_prores` or `cuda_notchlc`.
* **`mixer-parity` and `vk-validation` did not run**, so the two backends were compared only
  against the colour model, never against each other on a picture with spatial detail.

A green result on eight arms is evidence about eight arms.

---

## 8.2 The decode path: bit-identical, and the PNG prediction confirmed

The gap §8.1 declared largest. `flat-decoded` is the only 1 LSB decode gate here, and its
load-bearing check needs no colour model at all: a clip holding `u = v = neutral` must render
grey.

| binary | mixer | result | worst delta | neutral spread |
| :--- | :--- | :--- | ---: | ---: |
| 7.0.2 | ogl | 29/29 PASS | 0.01 | 0.00-0.01 |
| **8.1.2** | **ogl** | **29/29 PASS** | **0.01** | **0.00-0.01** |
| **8.1.2** | **vulkan** | **29/29 PASS** | **0.00** | **0.00** |

Six formats each — `420p10 pq`, `420p10 hlg`, `422p10 pq`, `444p10 pq`, `420p12 pq`,
`420p8 sdr`. One patch is skipped in `420p12 pq` because the *encoder* could not hold the
frame uniform (2 distinct values), which the battery reports as skipped rather than passed.

**The strong version of the claim comes from the captures, not the summaries.** Each run
saves the rendered frame per patch, so the two OGL runs can be compared directly:

    files differing in BYTES : 29 / 29
    files differing in PIXELS:  0 / 29      worst channel delta: 0

So FFmpeg 8.1.2 decodes every one of these formats to **bit-identical** pixels, and both
mixers agree. Combined with §8.1's 800 identical conformance results, the two largest
surfaces of this migration are measured and unmoved.

**And the byte/pixel split is `FFMPEG_8_MIGRATION.md` §5.2 happening in front of us.** That
section predicted FFmpeg 8.0's `pngenc` switch to PAETH prediction would change PNG file bytes
while leaving decoded pixels identical. Every capture differs in bytes; none differs in a
pixel. That also retroactively earns the check made earlier in this work — that no harness
module hashes PNG bytes rather than comparing pixels. If one did, it would be failing right
now, on every capture, for a reason that has nothing to do with correctness.

### 8.2.1 The first attempt at this measurement was invalid, and why

The first `flat-decoded` run reported `delta inf` on all 29 patches — no measurement at all,
not a wrong value — while the server's own log carried one benign warning. **Self-inflicted
contention**: the full `pytest` suite and two compiler probes were running against the same
GPU at the same time, and the driving shell then survived a `TaskStop` and kept launching
further arms, each spawning its own server.

Two things worth carrying forward. `inf` is not a failure to investigate as a colour defect —
it means the capture never settled, and the first question is what else was running.
And stopping a battery mid-flight does **not** stop the battery: the shell dies, its `python`
children do not, and their servers outlive both. The clean re-run — one arm at a time, nothing
else on the box — produced the numbers above in about 3.5 minutes per arm.

---

## 8.3 KNOWN REGRESSION, already reported upstream: the FFmpeg consumer's audio

**This is a shipping blocker for the FFmpeg consumer on 8.x, and an upstream PR for it already
exists** (opened by this team before this sync was measured; fill in the number here). No fix
is carried in this fork -- take upstream's when it lands. Recorded only because the blast
radius measured here is worth having and because nothing else in this document would tell a
reader why `signalling --stream` fails.

`ADD 1 STREAM ... -acodec aac` returns **501 ADD FAILED**, with the server logging

```
[aac] Unsupported channel layout "9.1.6"
ffmpeg_consumer.cpp: avcodec_open2(enc.get(), codec, &dict) -> EINVAL (-22)
[warning] Unused option ac=2
```

`9.1.6` is sixteen channels, CasparCG's internal count. The sink is never constrained to the
layouts the encoder accepts -- `ffmpeg_consumer.cpp:591` still carries the original
`// TODO: need to translate codec->ch_layouts` -- so the graph hands the encoder the source
layout and any encoder with a restricted list refuses it. `-ac` does not help: it is reported
as unused.

**Blast radius, one FRESH SERVER per variant** (a first pass reused one server and produced a
non-monotonic result set, which is what consumer accumulation looks like rather than data):

| variant | 7.0.2 | 8.1.2 |
| :--- | :--- | :--- |
| video only (audio defaults to mp2) | **FAIL** | FAIL |
| x265 + aac, no `-ac` | OK | **FAIL** |
| x265 + aac `-ac 2` | OK | **FAIL** |
| x265 + pcm_s16le `-ac 2` | OK | OK |
| x264 + aac `-ac 2` | OK | **FAIL** |
| x264 + aac, no `-ac` | OK | **FAIL** |

So on 8.1.2 every real audio encoder fails and only PCM survives, which covers essentially
all streaming and recording. Two details worth keeping:

* **`video only` fails on 7.0.2 as well.** That arm takes the container's default audio codec
  (mp2, which advertises only mono and stereo), so it is the same defect reached by a
  different route and it is NOT new in 8.x -- it was simply never measured, because no battery
  drives a video-only stream.
* Setting the sink's layouts from `avcodec_get_supported_config(nullptr, codec,
  AV_CODEC_CONFIG_CHANNEL_LAYOUT, ...)` plus buffersink's new array-typed `channel_layouts`
  option fixes the **mp2** case and NOT the aac one -- measured here before the change was
  reverted. Presumably the native AAC encoder advertises no static list without a configured
  context, so the query returns nothing and the constraint is skipped. Noting it so the
  upstream fix is not assumed to be that one-liner.

---

## 9. Owed

1. ~~**The after-image**~~ — **done, §8.1: 800 per-conversion results, zero differences.**
2. ~~`flat-decoded`~~ — **done, §8.2: bit-identical decoded pixels, both mixers.** Still
   owed: `sdi-input`, `signalling --stream`, `sdi-output`, `mixer-parity`, `vk-validation`.
   The FFmpeg *consumer* is now the largest unmeasured surface, since that is where §5's
   array-option migration actually lives and `signalling --stream` is the battery for it.
3. ~~`cli.py run --decoder cuda_prores` / `--decoder cuda_notchlc`~~ — **the link boundary is
   verified; NotchLC decode is not, and cannot be.** Measured 2026-08-19. Both CUDA modules
   load and initialise at runtime — `[cuda_prores] Module initialised`,
   `Initialized cuda_prores module.`, `[cuda_notchlc] Module initialised` — and `cuda_prores`
   decoded 11 of 26 cases successfully, best PSNR 52.2. A C++17/C++20 ABI or unresolved-symbol
   fault would fail every case rather than eleven, so §6.3's mixed-standard boundary works in
   practice and not only at link time.
   * **12 of the 26 failed at a uniform PSNR 24.5, and that is unexplained.** The pass/fail
     split is *not* accounted for by the transfer combination — the same combination appears on
     both sides — and there is no pre-sync baseline for these cases, so it cannot be called
     pre-existing either. Owed as its own investigation.
   * **`cuda_notchlc` decode is unreachable: the harness has no NotchLC source media.** 664
     cases reference a `notchlc` source and no such file exists under `media/sources`, so the
     decoder ran zero cases. The module loading is all that is currently checkable. Generating
     or obtaining a NotchLC fixture is owed before that half means anything.
4. Reconcile the two GPU-direct HTML paths (§4.1). **Parked by decision, 2026-08-18** — the
   findings, the PR shape and the harness it would need are written up in
   `docs/CEF_GPU_DIRECT_RECONCILIATION.md`. Not a gap in this sync; a deferred choice.
5. Merge or retire the OAL consumer divergence (§4.2). **Still open, and it is a decision
   rather than a measurement** — both implementations work; what is missing is a ruling on
   which one this fork keeps.
6. ~~Investigate the OGL 16-bit asymmetry (§3.1)~~ — **done, 2026-08-19, and it was not the
   mixer.** It was `swscale`'s packed-16 to packed-16 component permutation in the IMAGE
   consumer, which dithers a conversion that reduces no bit depth. Pre-existing and fork-only:
   upstream's `convert_image_frame` call sites all target 8 bit, where the same permutation is
   exact. OpenGL 16-bit conformance goes **0/100 -> 99/100** within 1.0 LSB, which is exactly
   the Vulkan figure. Full account in `docs/PLAN_BGRA64_CAPTURE_FIX.md` §8.2, the FFmpeg
   reproducer in `docs/swsprobe/`.
7. ~~A battery for the still-load swscale conversions~~ — **written, `image-convert`**, 4 PNG
   flavours with an `rgba8` control that bypasses swscale, asymmetric per-channel values, and a
   depth-mismatch guard. 26 tests pass. **Not yet committed** in the harness repo: `cli.py` and
   `CLAUDE.md` there carry another session's uncommitted work, and staging them would sweep it
   into this commit.

### 9.1 Measured 2026-08-19 — seven of seven, after fixing the blocker

| arm | verdict |
| :--- | :--- |
| `vk-validation` | **pass** — core validation, best-practices positive control, 12/12 commands accepted |
| `sdi-output --mixer ogl` | **pass** — v210 AVX2/scalar pack parity PASS (16-bit), 16-bit v210 +4.26 dB over 8-bit |
| `sdi-output --mixer vulkan` | **pass** — figures byte-identical to OpenGL, which is the parity requirement |
| `sdi-input` | **pass** — CasparCG 43.20 dB, FFmpeg on the same wire 42.61 dB (−0.60 dB) |
| `signalling` (DeckLink loopback 1:4) | **pass** — colour survives bt709/sdr, bt2020/hlg, bt2020/pq |
| `signalling --stream` | **pass, after the fix below** — 3/3 arms, 455 packets each, HDR10 round-trips |
| `mixer-parity` | **pass** — 6/6 rasters identical between backends, max diff **0**, gate 1 LSB (PAL, NTSC, 1080i5000, 1080p2500, 2160p2500, 2600x1500p25) |

So the DeckLink surface is verified on both backends, and the Vulkan API-usage check is clean.

**`signalling --stream` is blocked by the known 16-channel audio-layout defect, not by this
sync.** Isolated across five consumer option sets — bare mpegts, +aac 2ch, +one HDR option,
+all four HDR options, and h264+aac — **all five fail identically** with
`avcodec_open2 → EINVAL`, from:

```
[aac @ ...] Unsupported channel layout "9.1.6"
[mp2 @ ...] Specified channel layout ... is not supported by the mp2 encoder
[mp2 @ ...] Supported channel layouts:  mono  stereo
```

CasparCG channels always carry 16 channels, and `ffmpeg_consumer.cpp` constrains the audio
buffersink's sample formats and rates to what the encoder supports but not its channel layouts
— the commented-out `ch_layouts` TODO is still there. FFmpeg 7 encoded 16-channel AAC anyway
and wrote `channel_layout=unknown`; FFmpeg 8 refuses. A latent bug that FFmpeg 8 stopped
covering for, diagnosed and written up on 2026-08-18 in
`d:\Github\CasparCG-server\PR_ffmpeg8_aac_16ch.md`, still unfiled.

**Two things this measurement adds to that write-up.**

* The HDR consumer options are **not** implicated, and are in fact working correctly: the log
  carries `<hdr-metadata> ignored: the channel's transfer is not PQ or HLG`, which is the right
  refusal for an SDR channel. The first hypothesis here — that §5's option migration had broken
  the fork's HDR options — is wrong, and the isolation is what killed it.
* **`-ac 2` is ignored.** The battery passes it and the encoder still reports `9.1.6`, so the
  requested channel count never reaches the audio graph. The existing write-up records that
  `-c:a`, `-acodec` and container changes do not help; it never tested `-ac`. This matters
  because the known `ch_layouts` patch fixes `ac3` and `mp2` but provably **not** `aac` (the
  native AAC encoder publishes no layout list), whereas honouring `-ac` would.

**And the blast radius is wider than the existing write-up states.** That document is titled
"recording audio to AAC is broken" and frames the failure around AAC in mp4. Running `cli.py run`
over 381 cases before it was stopped, **every single error was an FFmpeg consumer**, across
codecs that do not share an audio codec:

```
56  ffmpeg_h264 / h265      14  ffmpeg_prores       13  ffmpeg_dnxhd
11  ffmpeg_dnxhr             6  ffmpeg_dnxh          3  ffmpeg_xdcam
```

Not one non-FFmpeg consumer errored. So **recording to file is broadly broken in this fork on
FFmpeg 8**, not merely AAC-in-mp4, and that is a larger problem than the stalled verification
arm it was found through. Whether all of them fail for the identical channel-layout reason is
not yet confirmed — ProRes in `.mov` and XDCAM in `.mxf` default to different audio codecs — so
the per-consumer cause is owed before the scope claim is rewritten upstream.

**FIXED 2026-08-19, commit `24a182b85`.** The audio buffersink is now constrained by the
encoder's published channel layouts, and `-ac N` is honoured in the graph instead of being
dropped. `signalling --stream` goes from `AMCP 501` with zero packets and an aborted battery to
**3/3 arms passing, 455 packets each**, with HDR10 static metadata round-tripping through both
ffprobe and our own producer. `ADD 1 FILE out.wav` still writes 16-channel PCM, so the path that
worked is untouched.

**Still not fixed, as a stated limit:** `ADD 1 FILE out.mp4` with no `-ac` fails, because AAC
publishes no layout list and there is no non-arbitrary channel count to pick. Defaulting it to
stereo is a behaviour decision rather than a bug fix. `ac3` and `mp2` now open where they
refused, but a finalised-file measurement for them is owed — the first attempt used a `bytes > 0`
gate that an empty 48-byte mp4 container passes, which is the exact class of check this repo's
own notes warn against.
It is out of scope for the sync itself: nothing in §5's migration caused it, and `signalling
--stream` measures colour and HDR10 signalling, which cannot be reached while the consumer
refuses to open.

### Two things this sync learned that outlive it

* **A timing verdict from `conformance` needs two interleaved runs per arm.** Three attempts at
  the §3.1 fix were reverted for a 3x wall-clock regression that did not exist: every attempt
  was measured on its first run after a build and compared against a control that was not. Two
  runs of the *same* binary, back to back on a settled box, differ by 1.9x — larger than the
  effect being chased. Owed as a harness change, not a note here.
* **Capture timeouts are not currently a gating signal.** A run can report a pass count while
  silently failing to capture; the count is then over the patches that survived. `conformance`
  should fail, or at minimum flag, a run with any capture timeout.
