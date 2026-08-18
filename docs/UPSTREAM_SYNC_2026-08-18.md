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

Ruled out already: the exe did not change mid-run, and `<render-format>` defaults to
`unorm` (`server.cpp:416`), so the 32-LSB16 fp16 quantum recorded in the harness's
`render_format_quantum_2026-08-12.md` is not the explanation either.

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

**CUDA 12.9's nvcc cannot parse MSVC 14.50's C++20 `<chrono>`:**

```
C:\...\MSVC.50.35717\include\chrono(5125):
    error C2760: syntax error: '}' unexpected here; expected 'expression'
```

It fires on `cuda_prores/consumer/prores_consumer.cu`,
`cuda_prores/consumer/prores_bypass_consumer.cu` and
`cuda_notchlc/cuda/notchlc_decode.cu`, which reach `<chrono>` transitively through
`common/log.h`. `decklink` and `remotewall` also carry `.cu` sources and would follow.

**Fix applied:** `CUDA_STANDARD 17` / `CUDA_STANDARD_REQUIRED ON` in
`casparcg_add_module_project`, so every module's CUDA sources stay at C++17 while its C++
sources move to C++20. Set centrally rather than on the two targets that failed first,
because the other two have the same exposure.

**Why this is defensible:** it leaves the `.cu` translation units on exactly the standard
the whole tree used before this sync — the configuration they were last known to build
under — rather than inventing a third one.

**KNOWN RISK, declared rather than resolved.** This makes the link boundary
mixed-standard, and std types DO cross it: every CUDA module registers a consumer or
producer factory, so `std::wstring`, `spl::shared_ptr` and `boost::property_tree` pass
between a C++17 object and a C++20 one. MSVC's standard library is ABI-compatible across
`/std:c++17` and `/std:c++20` in practice but does not contractually promise to be, and
nothing here has measured it. What would catch a problem is not a compile but a run:
`cli.py run --decoder cuda_prores` and `--decoder cuda_notchlc`, plus `sdi-output` for the
DeckLink `.cu` readback paths. **None has been run.** Revisit when CUDA ships a C++20 host
parser.

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

## 9. Owed

1. **The after-image**: re-run the §3 matrix on the new binary and diff against it.
2. `flat-decoded`, `sdi-input`, `signalling --stream`, `sdi-output`, `mixer-parity`,
   `vk-validation` — none has been run on either side of this sync.
3. Reconcile the two GPU-direct HTML paths (§4.1).
4. Merge or retire the OAL consumer divergence (§4.2).
5. Investigate the OGL 16-bit asymmetry (§3.1) — pre-existing, not caused here.
6. A battery for the still-load swscale conversions, which nothing covers; the oracle needs
   no colour model, because an 8-bit RGB PNG and an 8-bit RGBA PNG holding the same values
   differ only in whether `is_frame_compatible_with_mixer` sends them through swscale.
   Asymmetric per-channel values are mandatory — these are channel permutations.
