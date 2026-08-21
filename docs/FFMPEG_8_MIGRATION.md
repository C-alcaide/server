# Migrating CasparVP from FFmpeg 7.0.2 to 8.x — Investigation

**Status: investigation written 2026-08-17, corrected 2026-08-18.** The migration it plans
for was already carried out upstream, and the tree has since been built and run against
8.1.2 — see the correction below.

> **Headline: the API migration is far smaller than expected — close to zero.** Of the 24
> deprecated API groups FFmpeg removed in the 8.x cycle, **CasparVP references none of
> them**, and the two that would have bitten (`FF_API_INTERLACED_FRAME`,
> `FF_API_TICKS_PER_FRAME`) already have version-guarded dual paths in the tree, one of them
> gated on `< 62` — i.e. someone already wrote the FFmpeg 8 branch.
>
> The real cost is **not** the API. It is a swscale engine rewrite that can change rendered
> bytes, through `convert_image_frame` and the Spout downscale.
>
> **Amended 2026-08-18:** this used to read "confined to two call sites… neither in the 1 LSB
> path". Both halves were wrong. `convert_image_frame` has four callers, one of them the
> **IMAGE consumer**, which runs it on every captured frame — so the swscale rewrite sits
> inside `conformance` and `grading` rather than outside them. See §5.1.
>
> This corrects [`GSTREAMER_INTEGRATION_PLAN.md`](GSTREAMER_INTEGRATION_PLAN.md) §4.1, which
> called the migration "the one genuinely large unknown in this plan". On the evidence below
> it is not.

---

## 0. Correction, 2026-08-18 — upstream already migrated

Everything below was written against **CasparVP's** tree *as it then was*, pinned to 7.0.2.
**The fork is on 8.1.2 now** — the four-stage upstream sync landed at `bc94f4713`, so every
present-tense "we are on 7.0.2" in the sections below is describing the starting point of this
investigation, not the tree. Section 6.1.1 onward is written against 8.1.2 and is current. It
also did not check where **upstream** stood. Upstream `CasparCG/server` master has carried FFmpeg 8
since before this document was written:

| Commit | What it did |
| :--- | :--- |
| `480e9b00e` `feat: support FFmpeg 8 (#1715)` | the API work in `ffmpeg_consumer.cpp` and `av_producer.cpp` |
| `88ccb0303` `feat: build with ffmpeg 8.1 on windows` | the pin, the sonames, and `bluefish_producer.cpp` |
| `d9dd82632` `fix: decklink producer ffmpeg 8` | the DeckLink producer |
| `0525dca72` / `970245ecd` | the libpostproc removal (§5.3), and its Windows walk-back on the 2.5.x branch |

`upstream/master`'s `Bootstrap_Windows.cmake` pins `ffmpeg-8.1.2-full_build-shared.7z` with
`avcodec-62`, `avdevice-62`, `avfilter-11`, `avformat-62`, `avutil-60`, `swresample-6`,
`swscale-9` and **no `postproc` line** — which is §7's steps 3 and 5, already done.

**So for this fork the job is a rebase, not a migration.** `grading-nodes` is **71 commits
behind** `upstream/master`, and those 71 also carry C++20 and the merged Vulkan accelerator,
which is the actual cost — not the FFmpeg API.

**Built and run, 2026-08-18.** Branch `proto/gstreamer-ffmpeg8` in `d:\Github\CasparCG-server`,
off `upstream/master`, Ninja + RelWithDebInfo + MSVC 14.50, 213 targets, 22 runtime DLLs
shipped; the server answers AMCP with `201 VERSION OK / 2.6.0 30fcb2f4e Dev` and
`200 INFO OK / 1 1080p5000 PLAYING`. §1's caveat — *only a build proves the build* — is
discharged for the upstream tree.

**It is not discharged for this fork.** Nothing fork-only has been compiled against 8.x:
`cuda_prores`, `cuda_notchlc`, `spout`, `isf`, `ofx`, the OCIO stages and the Vulkan mixer.
Those are where §1's unknowns (struct layout, header reorganisation, unchanged signatures
with changed semantics) can still land.

One fix was needed on top of upstream to get a running binary, and it is not FFmpeg's:
the imported SFML targets have Debug and Release configurations only, so RelWithDebInfo
falls back to Debug and the exe imports `sfml-*-d-2.dll` while the copy step ships
`sfml-*-2.dll`. It then exits `STATUS_DLL_NOT_FOUND` before its logger starts. CasparVP
already carried the `MAP_IMPORTED_CONFIG_RELWITHDEBINFO Release` mapping; upstream did not.

---

## 1. Method, and what it does not prove

Changelogs describe intent; headers describe fact. So rather than read release notes, this
compared **what we use** against **what exists**, mechanically:

1. Every file including a libav header — **27 files across 10 modules**, not just the
   `ffmpeg` module (`bluefish`, `cuda_notchlc`, `cuda_prores`, `decklink`, `hap`, `image`,
   `isf`, `newtek`, `oal`, `spout` all touch libav).
2. Extracted 156 distinct FFmpeg function-style symbols, 132 constants and 224 struct member
   names used by those files.
3. Tested each for existence in the FFmpeg 8.1 public headers — `d:\Github\FFmpeg` at
   `n8.1.1-7-g3728de467d`, the project that defines them.
4. Independently, took FFmpeg's own git history for the 24 `FF_API_*` removal commits in the
   8.x cycle, extracted the identifiers each removal actually deleted, and intersected that
   with our usage.

**What this does not prove: that it compiles.** It cannot see struct *layout* changes, header
reorganisation, changed semantics behind an unchanged signature, or member accesses written
as `.field` rather than `->field`. Only a build against 8.x proves the build. Treat §3 as
"nothing known-removed is referenced", not as "it will compile clean".

---

## 2. Sonames — why any of this is on the table

| Library | 7.x (current pin) | 8.x |
| :--- | :--- | :--- |
| libavcodec | 61 | **62** |
| libavutil | 59 | **60** |
| libavformat | 61 | **62** |
| libavfilter | 10 | **11** |
| libswscale | 8 | **9** |
| libswresample | 5 | **6** |

Read from `libav*/version_major.h` and `libavutil/version.h`. Every DLL base name changes,
which is what makes GStreamer's own FFmpeg (61/59/61/10/8/5) stop colliding — but that is a
consequence, not the subject of this document.

---

## 3. What breaks at compile time

### 3.1 Removed API we reference: none

All 24 removal groups: `FF_API_ALLOW_FLUSH`, `FF_API_AVCODEC_CLOSE`, `FF_API_AVFFT`,
`FF_API_AVSTREAM_SIDE_DATA`, `FF_API_BKTR_DEVICE`, `FF_API_BUFFER_MIN_SIZE`,
`FF_API_DROPCHANGED`, `FF_API_FF_PROFILE_LEVEL`, `FF_API_FRAME_KEY`, `FF_API_FRAME_PKT`,
`FF_API_GET_DUR_ESTIMATE_METHOD`, `FF_API_H274_FILM_GRAIN_VCS`,
`FF_API_HDR_VIVID_THREE_SPLINE`, `FF_API_INTERLACED_FRAME`, `FF_API_LAVF_SHORTEST`,
`FF_API_LINK_PUBLIC`, `FF_API_OPENGL_DEVICE`, `FF_API_PALETTE_HAS_CHANGED`,
`FF_API_QUALITY_FACTOR`, `FF_API_SDL2_DEVICE`, `FF_API_SUBFRAMES`,
`FF_API_TICKS_PER_FRAME`, `FF_API_VDPAU_ALLOC_GET_SET`, `FF_API_VULKAN_CONTIGUOUS_MEMORY`.

**Intersection with our source: empty.**

`FF_API_LINK_PUBLIC` deserves a note because it looks like it should have hurt — it makes
`AVFilterLink` members private, and both [`decklink_producer.cpp`](../src/modules/decklink/producer/decklink_producer.cpp)
and [`ffmpeg_consumer.cpp`](../src/modules/ffmpeg/consumer/ffmpeg_consumer.cpp) drive filter
graphs. They are safe because they already read link properties through the accessors —
`av_buffersink_get_w`, `av_buffersink_get_h`, `av_buffersink_get_frame_rate`,
`av_buffersink_get_time_base`, `av_buffersink_get_colorspace`, `av_buffersink_get_ch_layout` —
rather than touching fields.

### 3.2 Already migrated, ahead of time

| Removal | Site | State |
| :--- | :--- | :--- |
| `FF_API_INTERLACED_FRAME` (`AVFrame.interlaced_frame`, `.top_field_first`) | [`av_producer.cpp:1661`](../src/modules/ffmpeg/producer/av_producer.cpp#L1661), [`:1759`](../src/modules/ffmpeg/producer/av_producer.cpp#L1759), [`decklink_producer.cpp:437`](../src/modules/decklink/producer/decklink_producer.cpp#L437), [`:444`](../src/modules/decklink/producer/decklink_producer.cpp#L444) | `#if LIBAVCODEC_VERSION_MAJOR < 61` guards; the ≥61 branch already uses `AV_FRAME_FLAG_INTERLACED` / `AV_FRAME_FLAG_TOP_FIELD_FIRST` |
| `FF_API_TICKS_PER_FRAME` (`AVCodecContext.ticks_per_frame`) | [`av_producer.cpp:1775`](../src/modules/ffmpeg/producer/av_producer.cpp#L1775) | `#if LIBAVCODEC_VERSION_MAJOR < 62` — the FFmpeg 8 branch derives ticks from `AV_CODEC_PROP_FIELDS`, and cites the upstream commit |
| `AV_CODEC_ID_TIMECODE` (removed in 7) | [`ffmpeg_consumer.cpp:45`](../src/modules/ffmpeg/consumer/ffmpeg_consumer.cpp#L45) | guarded local sentinel |
| `AV_FRAME_FLAG_KEY` | `bluefish_producer.cpp:474`, `decklink_producer.cpp:440` | already the new flag |

`bluefish_producer.cpp` sets only the new flags with no guard, so it is 8.x-ready and
7.0-incompatible-by-luck rather than by design — worth noticing but not a defect, since 61 is
the floor.

---

## 4. What compiles but incurs debt

Present in 8.1, marked `attribute_deprecated`. These do **not** block the migration; they are
the FFmpeg 9 bill.

| API | Sites | Replacement |
| :--- | :--- | :--- |
| `AVCodec.pix_fmts` | [`ffmpeg_consumer.cpp:421`](../src/modules/ffmpeg/consumer/ffmpeg_consumer.cpp#L421), `:444`, `:478` | `avcodec_get_supported_config(…, AV_CODEC_CONFIG_PIX_FORMAT, …)` |
| `AVCodec.sample_fmts` | `ffmpeg_consumer.cpp:526` | `AV_CODEC_CONFIG_SAMPLE_FORMAT` |
| `AVCodec.supported_samplerates` | `ffmpeg_consumer.cpp:530` | `AV_CODEC_CONFIG_SAMPLE_RATE` |
| `AVCodec.ch_layouts` | `ffmpeg_consumer.cpp:533` — already a `TODO`, commented out | `AV_CODEC_CONFIG_CHANNEL_LAYOUT` |
| `av_init_packet` | [`decklink_producer.cpp:401`](../src/modules/decklink/producer/decklink_producer.cpp#L401) — one call | `av_packet_alloc` / zero-init |

`av_stream_get_parser` (`av_producer.cpp:1784`) is **not** deprecated in 8.1 — it is a plain
public function. No action.

The `pix_fmts` cluster is worth doing during the migration rather than after: it is five
mechanical edits, `avcodec_get_supported_config` already exists in 7.1 so the change is
back-compatible with the current pin, and it removes the only real deprecation cluster in the
tree.

---

## 5. What we lose, and how to mitigate

### 5.1 swscale output may change — the actual risk

FFmpeg 8 carries an extensive swscale re-architecture: a new `SwsGraph` / `SwsOps` / `uops`
pipeline with per-component dependency solving, new SIMD backends and 3DLUT support. This is
an engine rewrite, not a tidy-up, and `sws_getContext` now carries the note *"this function is
to be removed after a saner alternative is written"*.

**Corrected 2026-08-18: swscale is in the 1 LSB path, on the capture side.** The claim this
paragraph used to make — that `conformance`, `grading` and `flat-decoded` cannot touch
swscale and therefore cannot detect a regression in it — is wrong, and it was wrong in the
direction that matters: it declared a gap where coverage exists, and it would have sent the
next reader looking for a new battery instead of re-running the two that already gate at
1 LSB.

`convert_image_frame` is not the still-image loader's private helper. It has **four**
callers, and one of them is the **IMAGE consumer**:

| Caller | Conversion | Flags |
| :--- | :--- | :--- |
| [`image_consumer.cpp:310`](../src/modules/image/consumer/image_consumer.cpp#L310), [`:333`](../src/modules/image/consumer/image_consumer.cpp#L333) | mixer readback → `AV_PIX_FMT_RGBA` (8-bit) or `AV_PIX_FMT_RGBA64BE` (16-bit), per [`:221`](../src/modules/image/consumer/image_consumer.cpp#L221) | `0` / `SWS_ACCURATE_RND \| SWS_FULL_CHR_H_INT` |
| [`image_producer.cpp:56`](../src/modules/image/producer/image_producer.cpp#L56), [`:82`](../src/modules/image/producer/image_producer.cpp#L82) | still load: `rgb24`/`bgr24` → `BGRA`; anything >8-bit or undescribable → `GBRAP16LE` | as above, by source depth |
| [`image_scroll_producer.cpp:143`](../src/modules/image/producer/image_scroll_producer.cpp#L143) | → `BGRA` | `0` |
| [`isf_image_load.cpp:29`](../src/modules/isf/isf_image_load.cpp#L29) | → `RGBA` | `0` |

The consumer's call is **unconditional on every captured frame**. The mixer hands it BGRA or
BGRA64, the PNG target is RGBA or RGBA64BE, so `src->format == pixFmt` is never true and
`convert_image_frame`'s early return never fires. Sixteen-bit captures take the
`SWS_ACCURATE_RND | SWS_FULL_CHR_H_INT` branch, because the flag choice is on source depth.

So **every battery that captures through the IMAGE consumer runs swscale once per frame** —
`conformance`, `grading`, `blend-mask`, `grade-window`, `alpha-domain`, `blend-domain`,
`grade-extremes`, `calibration`, `flat-decoded`, the `ocio*` family. Both conversions are
channel permutations and byte-order swaps with no resampling and no arithmetic, so the
correct answer is **exactly lossless** and the existing 1 LSB gates are the right instrument
at the right tolerance.

Two consequences, and the second is the reason this is worth a paragraph rather than a
footnote:

* **The before-image is a `conformance` + `grading` run**, both mixers, not a bespoke
  fixture set. That is cheaper than what step 1 of §8 asks for, and it is already automated.
* **It is a common-mode risk.** A swscale change moves every captured number in every
  battery at once, which presents as *"the mixer changed"* rather than *"the capture
  changed"* — and the two mixers would move together, so parity passes. If a broad, uniform
  shift appears after the pin moves, the IMAGE consumer is the first suspect and the shader
  is the last. The distinguishing probe is a non-IMAGE consumer: `sdi-output` and
  `signalling` do not go through `convert_image_frame`.

**What genuinely has no coverage** is narrower than the old claim, and still real:

| Site | Flags | Covered by |
| :--- | :--- | :--- |
| `image_producer` still load — `rgb24`→`BGRA`, `rgb48`→`GBRAP16LE` | `0` / `SWS_ACCURATE_RND \| SWS_FULL_CHR_H_INT` | **nothing** — no battery loads a still through the IMAGE producer |
| [`spout_consumer.cpp:230`](../src/modules/spout/consumer/spout_consumer.cpp#L230) | `SWS_FAST_BILINEAR` | **nothing** — reachable only via `cli.py run`, and it is the one site that actually *scales* |
| `image_scroll_producer`, `isf_image_load` | `0` | nothing, but both are permutations to 8-bit packed and the least exposed of the set |

`SWS_FAST_BILINEAR` remains the most exposed flag in the tree: it is explicitly a
speed-over-accuracy path, which is what an engine rewrite re-tunes, and it is the only one
doing real resampling.

**Mitigation, in order:**

1. Capture reference output from both sites under 7.0.2 **before** changing the pin — a
   handful of fixtures through `image_converter` and one Spout capture. Without a
   before-image there is nothing to compare against afterwards.
2. `SWS_FAST_BILINEAR` is the more exposed of the two: it is explicitly a
   speed-over-accuracy path, which is exactly the kind of thing a rewrite re-tunes. If it
   moves, `SWS_BILINEAR` is the conservative substitute, at a measured cost.
3. Longer term, both sites are candidates for removal rather than migration — the tree
   already scales on GPU everywhere else, and `spout_consumer` notes `sws_scale` costing
   "50+ ms for 6000×1700".

### 5.2 PNG byte-level output changes

FFmpeg 8.0 sets `pngenc`'s default prediction method to PAETH. PNG is lossless, so **decoded
pixels are identical** and any pixel-comparing test is unaffected — including `conformance`,
which captures through the IMAGE consumer and compares against a colour model. What changes
is file bytes and size. Mitigation: nothing, unless something hashes PNG files rather than
comparing pixels; worth confirming no harness module does.

### 5.3 libpostproc is gone entirely

FFmpeg removed the whole library in commit `8c920c4c39`, 2025-05-05 — there is no
`libpostproc/` in the 8.1 tree at all, so there is no `postproc-*.dll` to ship.

**Impact: one line.** We reference it only as a file to copy —
[`Bootstrap_Windows.cmake:97`](../src/CMakeModules/Bootstrap_Windows.cmake#L97),
`casparcg_add_runtime_dependency(".../postproc-58.dll")`. No source file in the tree
includes a postproc header or calls into it, so nothing is lost in function; the line just
has to go, or the copy step fails on a missing file.

### 5.4 Things to check but probably not losses

* **Old HLS protocol handler removed (8.1).** Grep configs and media paths for `hls://`
  before the switch. Almost certainly unused here.
* **TLS peer certificate verification on by default** — announced in 8.0 as landing "on the
  next major version bump", so FFmpeg 9, not 8. When it lands, HTTPS/TLS sources with
  self-signed certificates start failing closed. Track it; do not pre-empt it.
* **OpenSSL < 1.1.1 support dropped, yasm support dropped.** Build-side only; we consume
  prebuilt shared binaries.
* **OpenMAX encoders deprecated.** Unused.

### 5.5 Build-level: nothing appears to be lost

Our 7.0.2 is a gyan full build with 90 configure flags — `--enable-gpl --enable-version3
--enable-libsrt --enable-libx264 --enable-libx265 --enable-nvenc --enable-nvdec
--enable-cuda-llvm`. gyan's 8.1.2 full build advertises all of those plus `libplacebo`,
`libvpl`, `libjxl`, `libsvtjpegxs`, `librist`, `libsvtav1`, `vulkan` and `amf`.

**Verified 2026-08-18, from the artifact rather than the download page.** CMake's
`ExternalProject_Add` extracts the `.7z` itself, so the check needed no 7z extractor — only
a configure. `ffmpeg.exe -version` on the extracted `ffmpeg-8.1.2-full_build-shared.7z`
reports `8.1.2-full_build-www.gyan.dev`, libs `60.26.102 / 62.28.102 / 62.12.102 /
11.14.102 / 9.5.102 / 6.3.102`, and a configure line carrying every flag §5.5 claimed plus
the ones §6 depends on:

```
--enable-gpl --enable-version3 --enable-shared
--enable-vulkan --enable-libshaderc --enable-libplacebo --enable-opencl
--enable-nvenc --enable-nvdec --enable-cuvid --enable-ffnvcodec --enable-cuda-llvm
--enable-d3d11va --enable-d3d12va --enable-dxva2 --enable-libvpl --enable-amf --enable-vaapi
--enable-libsvtjpegxs --enable-liboapv --enable-libjxl --enable-libsvtav1 --enable-libvvenc
--enable-libx264 --enable-libx265 --enable-libaom --enable-libdav1d --enable-librav1e
--enable-libsrt --enable-librist --enable-libzmq --enable-libssh --enable-gnutls
--enable-libzimg --enable-lcms2 --enable-mediafoundation
```

So JPEG-XS (`libsvtjpegxs`), APV (`liboapv`), the Vulkan stack and the D3D12 hwaccels in §6
are present in the package we would actually ship, not merely in the upstream tree. Nothing
in our 7.0.2 flag set is absent; `postproc` is gone, which is §5.3 and expected.

---

## 6. What we gain

Beyond making GStreamer conflict-free. We are on 7.0.2, so 7.1 lands too.

### 6.1 Vulkan hardware paths — the largest gain for this fork

This tree has a Vulkan mixer and an explicit GPU-direct thesis
([`GPU_INTEROP_ARCHITECTURE.md`](GPU_INTEROP_ARCHITECTURE.md)). FFmpeg 8 adds:

| Feature | Version |
| :--- | :--- |
| **ProRes Vulkan hwaccel** and **ProRes Vulkan encoder** | 8.1 |
| **ProRes RAW decoder** and ProRes RAW Vulkan hwaccel | 8.0 |
| **swscale Vulkan support** | 8.1 |
| DPX Vulkan hwaccel | 8.1 |
| VP9 Vulkan hwaccel | 8.0 |
| AV1 Vulkan encoder | 8.0 |
| Vulkan compute codec optimisations | 8.1 |

ProRes on Vulkan is the striking one: [`cuda_prores`](../src/modules/cuda_prores/) exists
because ProRes needed a GPU path and CUDA was the only one available. A Vulkan hwaccel
decodes into an image the Vulkan mixer can consume without an interop hop — and unlike the
CUDA path it is not NVIDIA-only. ProRes **RAW** decoding is a capability the tree does not
have at all.

#### 6.1.1 Measured, 2026-08-21: ProRes decodes on Vulkan, and both blockers were ours

The ProRes Vulkan hwaccel is now wired end to end behind
`<vulkan-decode>` (off by default, and requiring `<gpu-direct-decode>`). Everything up to
and including the decoder's own initialisation works; the decode call itself faults.

**What was built.** FFmpeg is handed the *mixer's* `VkDevice` through
`AVVulkanDeviceContext` on a reserved compute queue family
(`src/modules/ffmpeg/util/vulkan_hwdevice.cpp`), and the mixer imports the decoded
`AVVkFrame` by copying its per-plane images into pooled mixer textures on the graphics
queue (`src/accelerator/vulkan/util/av_vulkan_import.cpp`), honouring FFmpeg's
timeline-semaphore contract. Three details are load-bearing and each is documented at its
site: `AV_VK_FRAME_FLAG_DISABLE_MULTIPLANE`, so a 3-plane 10-bit frame arrives as three
`VkImage`s rather than one multi-planar image; a **second declared queue family**, so the
images are allocated `VK_SHARING_MODE_CONCURRENT` and the graphics-queue copy needs no
queue-family ownership transfer; and enabling, at mixer device creation, the ~24 extensions
and ~39 features FFmpeg's decoders assume — FFmpeg reads feature support from the *physical*
device, so it emits SPIR-V for capabilities the logical device may never have enabled, with
no error path on either side.

**What blocked it, and it was not Vulkan.** `avcodec_receive_frame` took an access violation
on the first frame — 13,675 in five seconds, every one at that call. The cause was our own
`av_log` callback: it formatted its prefix with `avc->item_name(ptr)` unconditionally, and
`AVClass::item_name` is allowed to be NULL. `libavutil/log.c` says so itself —
`return (cls->item_name ? cls->item_name : av_default_item_name)(obj);` — and
`FFVulkanContext`'s class (`libavutil/vulkan.c`) omits the field. So the first log line a
Vulkan compute decoder emitted jumped to address 0. Seven AVClass initialisers in 8.1 omit
it, `rawdec` among them, so this was never Vulkan-specific; Vulkan is simply the first one
this tree reached.

**Six A/B runs said "not this" and every one of them was right**, which is the lesson worth
keeping: they all varied the decoder and none varied the logger.

| candidate | result |
| :--- | :--- |
| the mixer's shared device | FFmpeg creating its own Vulkan device faulted identically |
| the device extension and feature set | 24 extensions + 39 features enabled; no change |
| the queue family / image sharing mode | no change; CONCURRENT is still required for the copy |
| our pre-created single-plane frame pool | letting FFmpeg allocate its own faulted identically |
| frame threading | `threads=1` only turned a fatal crash into a caught per-frame fault |
| the `vulkan-1.dll` shipped in `build/shell` | removing it, so the process used the 1.4.309 system loader, changed nothing |
| the stale FFmpeg 7 DLLs in `build/shell` | a running server loads only the 8.x set — checked by enumerating its modules |

What finally named it: a vectored exception handler (`CASPARVP_VEH_TRACE`), because `/EHa`
turns the access violation into a C++ exception that the decode thread's `catch (...)`
swallows before the unhandled-exception filter can see it, and
`CASPAR_LOG_CURRENT_CALL_STACK()` is a `// TODO (fix)` stub. For a jump to address 0 the
return address is on top of the stack; that gave `casparcg.exe+0x8898c5`, and DbgHelp against
the PDB turned it into `caspar::ffmpeg::log_callback+0xf5`.

**What it does now.** `PLAY prores_422_bt709_sdr` on the Vulkan mixer with
`<vulkan-decode>true` logs *"Vulkan GPU-direct video active: 3 planes decoded by an FFmpeg
compute shader on the mixer's own device and copied device-local (no CPU frame, no
readback)"*, and **eight concurrent ProRes layers** run with all eight on that path, no device
loss, no fallback and no TDR. Host CPU, interleaved arms, two rounds each, with the
producer's own decision line read back so an arm that silently took the other path could not
pass as a saving:

| layers | software | vulkan-decode | |
| :--- | :--- | :--- | :--- |
| 1 | 1.21 cores | 1.02 | −15.5% |
| 4 | 1.79 cores (1.75–1.84) | 1.12 (1.11–1.13) | **−37.5%** |

The saving grows with decoding layers, which is the expected shape: the mixer's own cost is
fixed and only the decode moves.

**An earlier draft of this section claimed −90%, from 0.25 → 0.02 cores at one layer. That
was wrong** and is worth recording as such: 0.25 cores for a live 1080p25 server is not
plausible, and those figures came from a machine that had taken four TDRs in the preceding
fifteen minutes. The same measurement on a settled machine gives the table above. A number
that looks too good is a reason to re-measure, not to publish.

**THE CONCURRENCY BUG — one AVHWDeviceContext per producer.** This is what limited the path to
a single producer, and it was ours.

`make_vulkan_hwdevice_from_mixer` built a fresh `AVHWDeviceContext` for every producer.
FFmpeg serialises its own queue submissions with a mutex that lives *on that context*
(`vulkan_device_init` installs `lock_queue` when the application leaves it null), so two
producers meant **two independent mutexes guarding one VkQueue** — the queue from the family we
hand over. FFmpeg's decode threads then submitted to it concurrently. One shared, cached
context per `VkDevice` fixes it: 2 layers went from failing 6 of 6 interleaved runs to 3 of 3
clean, and 4 and 8 layers are clean.

It also dissolves the result that had "ruled FFmpeg out": four concurrent Vulkan ProRes
decoders inside one `ffmpeg.exe` complete cleanly because that is **one** context and therefore
one mutex. The difference was never the decoder — it was how many contexts wrapped the device.

**What found it: the Vulkan validation layer**, added at runtime behind
`CASPARVP_VK_VALIDATION=1` because validation was `#ifdef _DEBUG` and every measured build is
RelWithDebInfo — so the configuration under test could never be validated. It reported
`UNASSIGNED-Threading-MultipleThreads-Write, vkQueueSubmit2(): THREADING ERROR : object of
type VkQueue is simultaneously used`, repeatedly, until it hit the layer's duplicate limit.

**A second real bug it found on the way:** enabling `VK_EXT_shader_object` — which is on
FFmpeg's own optional list, so this code enabled it — without enabling
`VkPhysicalDeviceShaderObjectFeaturesEXT::shaderObject`. FFmpeg took its shader-object path
and called `vkCreateShadersEXT`/`vkCmdBindShadersEXT` against a device where the feature was
false; validation named it as `VUID-vkCmdDispatch-None-08606` and three siblings. Extension
features are now queried and enabled per extension, so the two can never disagree again. This
did **not** fix the device loss on its own.

**Four things were accused and acquitted**, and the write-ups they produced were wrong before
this: the semaphore signal, waiting for our own copy, the frames pool, and the thread count.
The A/B that convicted the semaphore signal — 356 device-lost with it, 0 without — was
invalid, because the arm "without" had only one of four producers ever active and so was not
doing concurrent work at all. Repeats then showed 6 of 6 failures in *both* arms. **A single
30-second soak is not a verdict on a race**, and an arm that changes how much work happens is
not a control.

**One validation finding remains, and it is upstream.**
`THREADING ERROR : object of type VkFence is simultaneously used` still appears three times in
a 30-second four-layer run — and `ffmpeg.exe` alone, four concurrent decoders on one device
with its own `debug=1` validation, reports the same thing plus a
`VUID-vkCmdDispatch-storageBuffers-06936` we do not hit. So it is FFmpeg's behaviour in
FFmpeg's own tested configuration, not something this integration introduces.

**It is not byte-identical to software decode, and that is FFmpeg's.** On the same pinned
frame (LOAD + PAUSE + SEEK 7), channel output differs on **285 of 2,073,600 pixels (0.0137%)
by at most 2 codes of 255**; PRINT RAW agrees at 279, so the difference is in the decode
rather than a readback route. `ffmpeg.exe` decoding the same clip its own two ways, with no
CasparCG in the picture, differs from itself on **2,137 of 4,147,200 10-bit samples (0.0515%),
max 5 codes of 1023**. A compute-shader IDCT is not obliged to match a SIMD one bit for bit,
the same way JPEG's integer and float IDCTs do not. So the option stays opt-in, and no doc
should describe this path as byte-identical.

**Still owed: a battery.** `decode-cost` has no `vulkan-decode` arm — it sweeps
`gpu-direct-decode`, which does nothing for ProRes because there is no D3D11VA ProRes decoder
— so every figure here was taken by hand with scratchpad rigs. They belong in the harness
before they are quoted again, and the case that matters most is the **concurrent-producer
soak**: it is the one that would have caught the device loss, and the one that would have
caught the bogus −90% too, because its control would have shown only one producer active.

**Two things that made this hard to see, worth carrying forward:**

* **A default `get_format` picks the software format even with `-hwaccel vulkan`.** The
  standalone probe that "verified prores_vulkan" reported `Format yuv422p10le chosen by
  get_format()` — the Vulkan decoder was never exercised. It needs
  `-hwaccel_output_format vulkan` to be forced. Any measurement of this decoder must show
  the `Format vulkan chosen by get_format()` line or it measured the software path.
* **`build/shell/ffmpeg.exe` was FFmpeg 7.0.2** while the DLLs beside it were 8.1.2, so a
  probe run with it could not use any 8.x feature. **Fixed 2026-08-21** by running
  `--target casparcg_copy_dependencies`; the cause and the standing check are in
  `BUILDING_WORKFLOW.md` pitfall #7. It was not only a probe hazard — `scanner.exe` shells
  out to the `ffprobe` beside it, so the media scanner was probing files with 7.0.2 while
  the server decoded them with 8.1.2. The whole 7.x DLL set is quarantined in
  `build/shell/_stale_ffmpeg7/`; a running server was verified to load only the 8.x set. The
  8.1.2 CLI also lives at `build/ffmpeg-lib-prefix/src/ffmpeg-lib/bin/ffmpeg.exe`.

#### 6.1.2 All three ProRes routes, measured (Phase 3.1) — and why both stay opt-in

The plan asked whether the GPU path should become the default for ProRes. It should not, yet,
and the reason is that host CPU is the only axis on which the answer is obvious.

Three routes, 4x 1080p layers, interleaved arms, 2-3 rounds each, on the Vulkan mixer. Two
controls: each arm must report its own decode route from the producer's decision line, and the
playhead must advance across the sampling window — the second one exists because two runs of
the identical rig gave 1.83 and 0.83 cores, and 0.83 was an idle server.

| | | software | `CUDA_PRORES` | `prores_vulkan` |
| :--- | :--- | :--- | :--- | :--- |
| **422 HQ** | CPU cores | 1.82 (1.79–1.84) | 1.51 (−16.8%) | **1.11 (−38.5%)** |
| | peak host RSS | 2231 MB | **473 MB** | 683 MB |
| **4444 + alpha** | CPU cores | 2.78 | 1.17 (−58%) | **1.12 (−60%)** |
| | peak host RSS | 3580 MB | **553 MB** | 693 MB |
| | cue latency (`CALL SEEK`) | 99 ms | **84 ms** | 104 ms |
| | `fps` mean | 0.990 | **0.999** | 0.996 |
| | `frame-time` mean / max | 0.497 / 0.637 | 0.500 / **0.525** | 0.511 / **3.487** |

**The verdict is split, which is the finding.** `prores_vulkan` is the cheapest decoder —
essentially free, sitting at ~1.12 cores whether the content is 422 or 4444, so what remains is
the mixer's fixed cost. `CUDA_PRORES` is the better playout citizen: a third of the host memory,
the fastest cue, and the only route with no frame-time excursions at all.

**Vulkan hitched at the loop wrap — FIXED, see 6.1.3.** As first measured: 28 `fps` samples
below nominal in a 24-second window against CUDA's 7 and software's 0, the deeper ones at
**0.96** (a dropped frame), landing every ~3 seconds on a 3-second looping clip, plus two
`frame-time` spikes of 3.30 and 3.49 against a 0.51 mean.

The cause was frame-threading delay after the wrap's flush, and
`AV_CODEC_RECEIVE_FRAME_FLAG_SYNCHRONOUS` removes it. Re-measured on the same fixture, 4
layers, 2 rounds, `fps` samples below nominal:

| arm | with the flag | without |
| :--- | :--- | :--- |
| **vulkan** | **8** | **34** |
| software | 9 | 12 |
| cuda | 14 | 12 |

No `frame-time` excursion above 1.0 on any arm with it on. The CUDA row is a **control**: that
producer bypasses avcodec entirely, so the flag cannot reach it, and its 14-vs-12 is the noise
floor this comparison sits on. The Vulkan path is uniquely sensitive because its per-frame
chain is the longest — a host wait on the decoder's timeline semaphore plus a copy submitted
through the device thread — so the frame-threading delay was what pushed the post-wrap refill
past the tick budget.

So on stability Vulkan is now **level with or better than** the alternatives, and the earlier
table's stability column should be read as history.

`<vulkan-decode>` and the `CUDA_PRORES` keyword still both stay **opt-in**, but for one
remaining reason rather than two: the **2/255 IDCT difference** recorded in 6.1.1. That is a
rendered-output change for every existing ProRes config, which is the tree's highest bar, and
it is FFmpeg's own difference rather than something a fix here can remove.

**Two instrument defects found while producing this table**, both of the "cannot fail" kind:

* The first cue-latency probe reported **551 ms for every arm and every round** — it was
  measuring its own 500 ms poll interval. It would have been published as "cue latency is
  identical on all three routes". It polls at 20 ms now and reports its floor, so a value
  pinned to the floor is visibly not a measurement.
* The path detector looked for `cuda_prores` in the log while the producer logs under
  `[prores_producer]`/`[prores_demuxer]`, so the CUDA arm reported itself as `software` while
  plainly running. The path control caught it; without the control the −18% would have been
  attributed to the wrong decoder.

**Caveat on the `[diag]` columns**: they are the server's own graph values, and different
producers publish different graph *names* — the software and Vulkan routes report `buffer`
where the CUDA route reports `queue-fill`. Same-named columns are comparable across arms;
different-named ones are not the same quantity. Figures came from `<log-diagnostics>`, enabled
on **every** arm, since it puts a line per graph per interval on the frame path.

**Owed:** these are scratchpad rigs. `decode-cost` is the harness home and gaining `cuda` and
`vulkan` arms there is the follow-up — together with the loop-wrap hitch, which no battery can
currently see.

#### 6.1.3 `AV_CODEC_RECEIVE_FRAME_FLAG_SYNCHRONOUS` — adopted, and it fixes the loop-wrap drops

Phase 4's first item. `avcodec_receive_frame_flags()` (lavc 62.22.101; our pin is 62.28.102)
takes `AV_CODEC_RECEIVE_FRAME_FLAG_SYNCHRONOUS`, documented as *"the decoder will bypass frame
threading and return the next frame as soon as possible ... may deliver frames earlier than the
advertised `AVCodecContext.delay`"*. With `threads=0` resolving to `hardware_concurrency`, a
frame-threaded decoder holds `thread_count` frames before emitting the first, and after a SEEK
every one of those is latency the operator waits through.

**Armed by a flush, disarmed by the first frame.** Not unconditionally: the flag defeats frame
threading, which is what pays for steady-state throughput — the same reason `output_capacity`
is raised to `thread_count`. So it applies to exactly the frames a cue is waiting on.

**Measured safe.** On a looping clip, the worst case since every wrap re-arms it, server CPU
moved **+4.1% in one round and −4.6% in the next** — a sign flip, so noise, not a cost.

**What it measurably fixes: the loop-wrap drops on the Vulkan decode path** — the very defect
6.1.2 named as the reason that option could not be a default. A wrap flushes the decoder and
the frame the channel needs is the next one, so the frame-threading delay starved it every
time round. Same fixture, 4 layers, 2 rounds, `fps` samples below nominal:

| arm | with | without |
| :--- | :--- | :--- |
| **vulkan** | **8** | **34** |
| software | 9 | 12 |
| cuda | 14 | 12 |

and no `frame-time` excursion above 1.0 with it on, against two of 3.3-3.5 without. **CUDA is
the control**: that producer never touches avcodec, so the flag cannot reach it, and its
14-vs-12 is the noise floor.

**It does NOT measurably improve cue latency, which is what it was adopted for.** 78.7 vs
78.2 ms median over 39 interleaved seeks, on both the hardware and software decode paths, with
every single value in both arms between 78 and 80 ms — **pinned at two channel ticks**, because
the probe watches the playhead in `INFO` and that only advances on a tick. So quote the
loop-wrap result and not a cue-latency one; settling the latter needs a producer-side
timestamp from seek to first published frame, which nothing in the harness has.

The lesson is worth keeping separately from the result: the first write-up of this change said
"measured safe, not measured to help", and it was wrong — not because the measurement was
wrong, but because it was of the wrong quantity. Failing to find a benefit on the axis you
expected is not the same as there being none.

Worth noting for the next attempt: on a **hardware** decode path there may be nothing to skip
at all, since hwaccel decoding is not frame-threaded the way software decoding is — which is
its own argument for measuring the software path separately rather than assuming one number
covers both.

#### 6.1.4 The rest of Phase 4, closed with reasons rather than commits

Phase 4 was explicitly "ranked by measured value, only if Phase 1 says so". Two of its items
are done (`alpha_mode` in 6.1.x's sibling work, `RECEIVE_FRAME_FLAG_SYNCHRONOUS` in 6.1.3).
The remainder are closed as **not worth doing**, and the reasoning is recorded so nobody
re-derives it:

**D3D11 `BindFlags`/`MiscFlags` on `AVD3D11VADeviceContext` (lavu 60.24.100) — declined.**
The plan credited this with replacing the fork's manual frames-pool creation inside
`get_format` "with its documented P010 hazard". It does not remove that hazard. The hazard is
that `D3D11_BIND_SHADER_RESOURCE` lands on a pool which `av_hwframe_transfer_data` also uses,
and 10-bit HEVC then fails outright — and the new device-level field is documented as applying
*"globally to all AVD3D11VAFramesContext allocated from this device context"*, which is the
same scope. So it moves where the flag is set and changes nothing about the risk.

It also only replaces half of what that block does. The other half is `initial_pool_size += 16`,
because the GPU-direct path holds surfaces while they queue for extraction and without the
headroom the decoder hits "Static surface pool size exceeded" and the channel goes black. The
FFmpeg-8 equivalent is `AVCodecContext.extra_hw_frames`, which this tree does not use anywhere
— so adopting it would be an unmeasured change to pool sizing on a path that is now **on by
default**. A refactor with no measured benefit, real regression risk on every config, and a
hazard it does not actually fix is the wrong trade; the existing block stays, with its comment
explaining its shape.

**`ffv1_vulkan` / `dpx_vulkan` — not applicable here.** The plan gated these on "only if those
formats are actually used". They are not: no producer in `src/` handles FFV1 or DPX, there are
no fixtures for either in the harness, and the only mentions of those names in the tree are in
comments describing which decoders `<vulkan-decode>` covers. Nothing to measure and nothing to
adopt until a real asset appears.

**`prores_ks_vulkan` encoder — out of scope**, as the plan itself said: it takes Vulkan frames
with no readback and is interesting for the recording consumers, but it is an encode project
and mixing it into a decode one is how both get half-done.

### 6.2 Codecs that matter in this domain

* **JPEG-XS** — decoder, encoder, parser and raw muxer/demuxer via `libsvtjpegxs` (8.1). The
  mezzanine codec for SMPTE 2110-22 contribution. Pairs directly with GStreamer's
  `rtpvrawpay`/`rtpvrawdepay` story in the integration plan.
* **APV** (Advanced Professional Video) — decoder, encoder via `libopenapv`, parser, and
  MP4/ISOBMFF muxing (8.0). A new professional intermediate codec.
* **VVC** — decoder complete including all Screen Content Coding (IBC, palette mode, ACT),
  plus VVC in Matroska and VAAPI decode (8.0).
* **LCEVC** — parser, metadata bitstream filter, enhancement-layer export in MPEG-TS (8.1).
* **MPEG-H 3D Audio** decoding via `mpeghdec` (8.1).

### 6.3 D3D12 and Windows capture

`vf_scale_d3d12`, `vf_deinterlace_d3d12`, `vf_mestimate_d3d12`, D3D12 H.264 and AV1 encoders
(8.1), `vf_scale_d3d11` (8.0), and `gfxcapture` — Windows.Graphics.Capture based window and
monitor capture (8.1).

The D3D12 additions are more interesting than they look in isolation: GStreamer's
`d3d12fisheyedewarp` and `d3d12remap` — the elements flagged as most relevant to the
projection work — are also D3D12. A migration that brings D3D12 into the FFmpeg path makes
that a shared problem with one bridge rather than two.

### 6.4 Broadcast metadata

* **RCWT closed-caption demuxer** (7.1) — the FFmpeg half of the caption story that
  GStreamer's `ccextractor`/`cccombiner` covers from the other side.
* **HDR10+ metadata passthrough** when decoding/encoding with libaom-av1 (8.0) — relevant to
  [`HDR_GUIDE.md`](HDR_GUIDE.md).
* **EXIF metadata parsing** (8.1); `colordetect` filter (8.0).

### 6.5 API quality

`avcodec_get_supported_config()` replaces four separate deprecated arrays with one queryable
call that works on a configured context rather than a static codec — which is the correct
answer for the negotiation logic in `ffmpeg_consumer.cpp` and materially simpler than what is
there now.

---

## 7. FFmpeg 9 is out, and this still stops at 8.1.2

Checked 2026-08-18 against the FFmpeg tree, not a release note. **FFmpeg 9.0 is real** —
`RELEASE` reads `9.0`, tag `n9.0` and branch `release/9.0` are dated **2026-08-03**.

| Library | 7.x | 8.x | **9.0** |
| :--- | :--- | :--- | :--- |
| libavcodec | 61 | 62 | **63** |
| libavutil | 59 | 60 | **61** |
| libavformat | 61 | 62 | **63** |
| libavfilter | 10 | 11 | **12** |
| libswscale | 8 | 9 | **10** |
| libswresample | 5 | 6 | **7** |

**It buys nothing for the reason the migration is on the table.** The GStreamer motive is
soname collision with GStreamer 1.28.6's 61/59/61/10/8/5. 8.x already has zero overlap;
9.0's overlap is equally zero. Any major bump settles it, and 8.1.2 is the one both upstream
and the `CasparCG/dependencies` mirror already ship.

**And unlike 7→8, it costs.** §4 called the deprecation cluster "the FFmpeg 9 bill"; 9.0 is
where it comes due, and it breaks code **upstream ships today**:

| Removed at 9.0 | Evidence | Site |
| :--- | :--- | :--- |
| `AVCodec.pix_fmts`, `.sample_fmts`, `.supported_samplerates` | `libavcodec/codec.h` matches `pix_fmts` once at `n8.1.2`, **zero times** at `n9.0` | `ffmpeg_consumer.cpp:223`, `:257`, `:258` |
| `av_init_packet` | still declared at `n9.0` but inside `#if FF_API_INIT_PACKET`, which is `(LIBAVCODEC_VERSION_MAJOR < 63)` — false at 63 | `decklink_producer.cpp:354` |
| default-off TLS verification | `FF_API_NO_DEFAULT_TLS_VERIFY` is `(LIBAVFORMAT_VERSION_MAJOR < 63)` | not a compile error — HTTPS sources with self-signed certificates begin failing closed, which §5.4 predicted for this release |

Two that were expected to hurt and do not: `av_stream_get_parser` is still a plain public
function at `n9.0`, and `AVFilterContext`'s `inputs` / `nb_inputs` / `outputs` are still
public members — `FF_API_CONTEXT_PUBLIC` does not privatise them.

Three more reasons the answer is "not yet":

* **No package.** `CasparCG/dependencies`, `ffmpeg` tag, holds `5.1.2`, `7.0.2`, `8.0.1`
  and `8.1.2` only. Pinning 9 means hosting our own artifact and diverging from upstream's
  `Bootstrap_Windows.cmake` — the exact divergence the rebase exists to remove.
* **swscale has not settled.** 9 continues the rewrite behind §5.1's risk: `doc/APIchanges`
  records `SwsBackend` and `SwsContext.backends` added 2026-06-03. Moving to 9 re-opens the
  same unmeasured question rather than closing it.
* **It is two weeks old.** No point release yet.

**The cheap hedge is worth taking now, on 8.1.2.** Replacing the three `AVCodec` arrays with
`avcodec_get_supported_config()` and the one `av_init_packet` with `av_packet_alloc` is
about five mechanical sites, compiles against 8.1 and 7.1 alike, and is the **whole** of the
FFmpeg 9 API bill. Done on the current pin it is verifiable against the existing batteries,
and it is a clean upstream contribution on its own.

---

## 8. Proposed order of work

**Revised 2026-08-18.** Steps 3 and 5 of the original list — bump the pin, delete the
`postproc` line, build — were done upstream (§0), so the remaining work is a port rather
than a migration.

1. **Capture swscale references under 7.0.2.** Unchanged, and still first: there is no going
   back for a before-image. It now applies only to the **fork's own** call sites, since
   upstream has neither — `image_converter.cpp` and `spout_consumer.cpp` (§5.1).
2. **Migrate the deprecation cluster on the current pin** — `avcodec_get_supported_config`
   and the one `av_init_packet`. Back-compatible with 7.0.2 and 8.1, verifiable against the
   existing batteries, and it is the entire FFmpeg 9 bill (§7).
3. **Rebase the fork onto `upstream/master`.** 71 commits, of which FFmpeg 8 is the small
   part; C++20 and the merged Vulkan accelerator are the large part.
4. **Build the fork-only modules against 8.x** — `cuda_prores`, `cuda_notchlc`, `spout`,
   `isf`, `ofx`, the OCIO stages, the Vulkan mixer. This is where §1's unknowns can still
   land; upstream building proves nothing about them. Remember that a header change does not
   trigger a rebuild in this tree — touch every source, and delete `build/**/cmake_pch.*.pch`,
   per `BUILDING_WORKFLOW.md`.
5. **Run the full battery set**, both mixers, then re-measure the step 1 swscale fixtures.
6. **GStreamer is already unblocked** — coexistence with FFmpeg 8 was measured on
   2026-08-18 and is recorded in [`GSTREAMER_INTEGRATION_PLAN.md`](GSTREAMER_INTEGRATION_PLAN.md)
   §3.5. It does not wait on steps 3–5, because it needs only a host built against 8.x, and
   one now exists.

Steps 1–2 are useful on their own and carry no FFmpeg 8 risk, which makes them the right
place to start regardless of when the rebase happens.

---

## 9. Verification

| What changed | What covers it |
| :--- | :--- |
| FFmpeg pin, decode path | `flat-decoded` (the 1 LSB decode gate), `sdi-input`, `source-colorspace` |
| FFmpeg consumer pixels/metadata | `signalling --stream`, `sdi-output` |
| Anything reaching the mixer | `conformance` + `grading`, **both mixers** |
| swscale on the **capture** side (`image_consumer` → RGBA / RGBA64BE) | `conformance` + `grading`, **both mixers** — see §5.1 |
| swscale on the **still-load** side (`image_producer`) | **nothing** |
| swscale **scaling** (`spout_consumer`, `SWS_FAST_BILINEAR`) | **nothing** — `cli.py run --consumer spout` captures a picture, at no gate |

**Corrected 2026-08-18.** This table used to say `image_converter` had no coverage at all.
It has coverage on its busiest path: the IMAGE consumer calls it on every captured frame, so
every 1 LSB battery in the harness exercises it once per frame (§5.1). Re-running
`conformance` and `grading` on both mixers before and after the pin moves **is** the swscale
before/after for that path, and no new battery is needed for it.

**Two real gaps remain**, and they are narrower than the retracted claim:

* **The still-load conversions.** Nothing in the harness loads a still image through the
  IMAGE producer, so `rgb24`→`BGRA` and `rgb48`→`GBRAP16LE` are unmeasured. This is worth a
  battery and it is cheap to build honestly, because the oracle needs no colour model: an
  8-bit RGB PNG and an 8-bit RGBA PNG holding *the same values* differ only in whether
  `is_frame_compatible_with_mixer` sends them through swscale, so the RGBA capture is the
  reference for the RGB one. Same construction at 16 bits with `rgb48` against `rgba64`.
  Asymmetric per-channel values are mandatory — these are channel permutations, and a grey
  patch is invariant under every way of getting one wrong.
* **The Spout downscale.** The only site that resamples, on the only flag
  (`SWS_FAST_BILINEAR`) a rewrite is likely to re-tune, reachable only through `cli.py run`.

**And one trap that the coverage creates.** Because the capture-side conversion is common to
every battery and to both mixers, a swscale regression moves every number at once and
survives every parity check. It reads as a mixer fault and is not one. `sdi-output` and
`signalling` bypass `convert_image_frame` entirely and are therefore the probes that
separate the two.

---

## 10. Open items

* ~~Verify gyan 8.1.2's actual configure line (§5.5)~~ — **closed 2026-08-18**. CMake
  extracts the `.7z` during configure, so no 7z extractor was needed; the flags are in §5.5.
* Whether anything in configs or media paths uses `hls://` (§5.4).
* ~~Whether any harness module hashes PNG bytes rather than comparing pixels (§5.2)~~ —
  **closed 2026-08-18: none does, and the two that hash are immune anyway.** No module
  references a stored PNG digest. `core/mix_cases.py`'s `sha256` compares one channel against
  another *within a single run* (the `route://` tripwire), and `core/check_plan.py`'s `md5` is
  frame-to-frame *within one recording*. Both compare outputs of the same encoder build to
  each other, so a change to `pngenc`'s default prediction method cancels. `source_digest`
  (`media/generate_references.py:162`) digests the **fixture file**, not a render, and
  fixtures come from PATH FFmpeg rather than the server — already 8.1.1 — so the pin bump
  causes no digest churn either.
* ~~The build itself~~ — **closed for the upstream tree 2026-08-18** (§0): it configures,
  builds and runs. **Still open for the fork's own modules**, which have not been compiled
  against 8.x at all.
* The §5.1 swscale before/after. **Revised 2026-08-18**: the capture-side path is covered by
  `conformance` + `grading` and needs a re-run, not a battery (§9). What still has no battery
  is the **still-load** path through the IMAGE producer and the **Spout downscale**, and §9
  now says how the first of those would be built.

## 11. Sources

* **FFmpeg 8.1** — `d:\Github\FFmpeg` at `n8.1.1-7-g3728de467d`. Sonames from
  `libav*/version_major.h` and `libavutil/version.h`; removals from the project's own git
  history (`git log --grep="remove deprecated FF_API"`, 24 commits); deprecation markers and
  the `sws_getContext` note from the headers; feature lists from `Changelog`.
* **CasparVP** — the 27 files including a libav header, at `feature/ocio-mixer`.
* **gyan.dev** — the builds page, for 8.1.2's advertised external libraries. Advertised, not
  verified against the artifact.
* **`CasparCG/dependencies`** — GitHub releases, `ffmpeg` tag, for package availability;
  re-read 2026-08-18 for §7 (`5.1.2`, `7.0.2`, `8.0.1`, `8.1.2`; no 9.x).
* **FFmpeg 9.0** (§7) — the same local tree at tags `n8.1.2` and `n9.0`: `RELEASE`,
  `libav*/version_major.h`, `libavcodec/codec.h`, `libavcodec/packet.h`,
  `libavformat/version_major.h`, `libavfilter/avfilter.h`, `doc/APIchanges`.
* **The 8.1.2 artifact itself** (§5.5) — `ffmpeg.exe -version` from the package CMake
  extracted into `build-ffmpeg8/ffmpeg-lib-prefix`, rather than the download page.
