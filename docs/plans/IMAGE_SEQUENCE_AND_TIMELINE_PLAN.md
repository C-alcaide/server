# Image-Sequence Playout and Timeline (EDL) Playout — Plan

**Status: proposal. Nothing here is implemented.** Written 2026-08-11 from an
investigation of what the tree can do today. Two related features are described; they
share a mechanism but are separable, and only the first is cheap.

---

## 1. Where the tree stands today

### There is no image-sequence producer

Two producers touch stills, both registered in
[`image.cpp:34-35`](../../src/modules/image/image.cpp#L34-L35):

* **Image Producer** — [`image_producer.cpp`](../../src/modules/image/producer/image_producer.cpp)
  loads exactly one file in its constructor, builds one `draw_frame`, and `receive_impl`
  returns that same frame forever ([`:105`](../../src/modules/image/producer/image_producer.cpp#L105)).
  Params: `LENGTH`, `SCALE_MODE`. It is a hold, not a sequence.
* **Image Scroll Producer** — scrolls one oversize still. Also not a sequence.

### Why a `frame%04d.dpx` pattern cannot reach FFmpeg's `image2` demuxer

Both the image producer ([`:136`](../../src/modules/image/producer/image_producer.cpp#L136))
and the ffmpeg producer ([`ffmpeg_producer.cpp:406`](../../src/modules/ffmpeg/producer/ffmpeg_producer.cpp#L406))
resolve their filename through `find_file_within_dir_or_absolute` → `probe_path`
([`filesystem.cpp:33-60`](../../src/common/filesystem.cpp#L33-L60)), which directory-iterates
and compares case-insensitively against **real directory entries**. A pattern matches
nothing, so `create_producer` returns `frame_producer::empty()` and AMCP answers 404.

The only route that skips file resolution is the `://` branch at
[`ffmpeg_producer.cpp:405-414`](../../src/modules/ffmpeg/producer/ffmpeg_producer.cpp#L405-L414).
It was tested to see whether a URL could smuggle a pattern through. It cannot:

| URL handed to `avformat_open_input` | outcome |
| :--- | :--- |
| `seq/frame%04d.png` (no scheme) | `format_name=image2` — but caspar rejects before this |
| `file://C:/…/frame%04d.png` | image2 **is** selected by probe; open fails, *"no file or sequence"* |
| `file:///C:/…/frame%04d.png` | same failure |
| `file:C:/…/frame%04d.png` | `format_name=image2` — but contains no `://`, so caspar routes it back to the rejecting path |

The two forms that satisfy caspar's `://` test are exactly the two whose Windows path
will not open. There is no usable workaround from the client side.

Separately: `.tga`, `.tiff`, `.tif`, `.jp2` sit on the ffmpeg producer's explicit
*invalid* extension list ([`:329-330`](../../src/modules/ffmpeg/producer/ffmpeg_producer.cpp#L329-L330)),
so they are declined by that producer on both branches — deliberately, so the image
producer wins for stills. A sequence unlock has to account for this.

### What the fork already added (stills only)

`f2bc3cac0` "image: Add HDR/16-bit/BT.2020 support to image producer and consumer"
added `.exr`, `.dpx`, `.hdr` to the accepted extensions
([`image_loader.cpp:159-161`](../../src/modules/image/util/image_loader.cpp#L159-L161)).
Effect: a **single** DPX or EXR frame plays and passes through the ACES chain. Sequences
were not part of it. The rest of the fork's work in this module is output-side (16-bit
PNG in the IMAGE consumer, path containment, Vulkan still-frame fixes).

### There is no EDL / timeline / playlist support of any kind

A grep for `EDL`, `OpenTimelineIO`, `otio`, `cmx3600`, `fcpxml`, `aaf`, `playlist`,
`timeline` across `src/` and `docs/` returns nothing relevant. Playlists have always
lived in AMCP clients.

---

## 2. Measurements

All figures below were produced with **the FFmpeg the server actually links** —
`build/ffmpeg-lib-prefix/src/ffmpeg-lib/bin/`, **version 7.0.2-full_build (gyan.dev)**,
`avcodec-61`. An earlier pass used the system "Ffmpeg 8.1 Custom" on `PATH`; every
result below was re-run against the linked build and reproduced. Do not quote these
numbers against a different FFmpeg without re-running.

**Decoders present in the linked build:** `dpx`, `exr`, `hdr`, `jpeg2000`, `pfm`, `png`,
`sgi`, `targa`, `tiff`, `webp`. **Demuxers:** `image2`, `image2pipe`. There is no
`cineon` decoder.

### 2.1 Throughput at 1080p — decoder is not the bottleneck

**Fixture:** 30 frames of `testsrc2` at 1920x1080, written as
`sh010_v001.%07d.dpx` (`-pix_fmt gbrp10le`) and `sh010_v001.%07d.exr` (`-pix_fmt gbrpf32le`,
uncompressed).

| source | pix_fmt out | per frame | 30 frames, rtime | rate | disk needed @1080p50 |
| :--- | :--- | ---: | ---: | ---: | ---: |
| DPX 10-bit | `gbrp10le` | 7.9 MB | 0.197 s | 152 fps | 396 MB/s |
| EXR f32 uncompressed | `gbrpf32le` | 23.8 MB | 0.300 s | 100 fps | 1.19 GB/s |

### 2.2 Throughput at 4K — decoder **is** the bottleneck

**Fixture:** 20 frames of `testsrc2` at 3840x2160 — DPX `-pix_fmt gbrp10le`, and EXR
half-float (`-pix_fmt gbrp16le`) with `-compression zip1`.

| source | per frame | 20 frames, rtime | rate | ms/frame | disk @4K50 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| DPX 10-bit | 31.7 MB | 0.547 s | **36.6 fps** | 27 ms | 1.58 GB/s |
| EXR half, ZIP1 | 5.4 MB on disk (47.8 MB decoded) | 1.074 s | **18.6 fps** | 54 ms | 268 MB/s |

**A single decode context misses 4K50 for DPX and misses even 4K25 for EXR half/ZIP.**
This is the finding that promotes §4.1 from a design decision to a hard requirement
above 1080p.

Two details matter more than the rates:

* **EXR's parallelism is already exhausted.** utime 3.672 s against rtime 1.074 s is
  ~3.4x — slice threading is working and it is still too slow. The only remaining lever
  is **frame-level** parallelism: several decode contexts on different frames at once.
* **DPX cannot use slice threading at all.** `ffmpeg -h decoder=dpx` reports
  *"Threading capabilities: none"*, against exr's *"frame and slice"*. Its observed
  0.844 s CPU / 0.547 s wall (~1.5x) is I/O overlap, not decode threading. FFmpeg cannot
  frame-thread around a decoder that does not declare the capability, so this cannot be
  fixed with `-threads`.

Against a 20 ms budget at 50 fps, 4K DPX needs ~1.4 concurrent decoders and 4K EXR
half/ZIP ~2.7. **A read-ahead ring holding 4 decode contexts covers both with headroom** —
the fix is cheap, it is simply not optional.

Note also that compressed EXR is the easier *storage* case and the harder *CPU* case,
and uncompressed DPX the reverse. Whichever format the pipeline actually uses decides
which resource runs out first.

### 2.3 Colour metadata FFmpeg throws away

`ffprobe` on the fixtures, showing what reaches the producer:

| file | reported |
| :--- | :--- |
| EXR f32 | `color_transfer=linear`, **`color_primaries=unknown`** |
| DPX 10-bit | **`color_transfer=unknown`**, `color_primaries=unknown`, `color_space=gbr` |

The EXR `chromaticities` attribute is present in the file and FFmpeg discards it, so
**an ACES AP0 EXR and a linear Rec.709 EXR are indistinguishable to CasparVP.** The DPX
header likewise carries transfer-characteristic and colorimetric bytes that FFmpeg's
`dpx` decoder does not map, so a Cineon-log plate looks identical to a display-referred
one.

For a fork whose identity is an ACES chain gated at 1 LSB this is the most consequential
gap in the whole feature — it is why §4.3 needs an explicit param rather than a guess,
and it is the strongest argument for §3.5.

Note in passing: FFmpeg's exr decoder *does* expose `-layer` and `-part` options, so
multi-part and named-layer EXR are reachable one at a time. That is cruder than OIIO but
it is not a hard wall.

### 2.4 Conditions and caveats

* **Warm page cache**, default threading, this workstation. Cold NVMe is the real ceiling
  for the bandwidth columns and will be lower.
* `testsrc2` is **synthetic and compresses unusually well** — 5.4 MB for a 4K half-float
  frame is ~1/9 of uncompressed. Real plates will be several times larger on disk. ZIP
  decode cost tracks output pixels more than input bytes, so the fps figure is probably
  not badly optimistic, but this is **unmeasured on real material** and should be
  re-measured on an actual plate before any hardware is specified.
* The 1080p EXR fixture is uncompressed (worst case I/O, best case CPU); the 4K one is
  ZIP1 (the reverse). DWAA/DWAB has not been measured at either raster.

### 2.5 Seek

**Seek is frame-exact.** `-ss 0.4` on the 25 fps DPX pattern produced framemd5
`6d9197dd6b7b142609b64cf8c0c96c5d`, byte-identical to decoding `sh010_v001.0000011.dpx`
on its own. This is the important result: `SEEK`, `IN`/`OUT`, `LOOP`, `LENGTH` and
`SPEED` would all come from the existing `av_producer` unchanged, and the harness's
`seek` battery already covers that surface.

`duration` is reported correctly (1.200000 s = 30 frames at 25 fps), so `nb_frames()`,
`INFO` and OSC have something truthful to report.

---

## 3. Third-party options surveyed, and why none is adopted

### 3.1 DJV — no

[DJV 3](https://github.com/grizzlypeak3d/DJV) is an **application**, built on
[tlRender](https://github.com/darbyjohnston/tlRender) plus feather-tk (the author's own
UI toolkit). "Add DJV" really means "add tlRender". BSD-3, so GPL-3 compatibility is not
the objection. The objections are structural:

* **Dependency chain.** tlRender requires feather-tk, Imath, minizip-ng and
  OpenTimelineIO, and its build story is a CMake super-build that compiles all
  dependencies from source. This tree currently has *no* image-library dependency at all
  — `find_package` appears once in [`src/CMakeLists.txt:31`](../../src/CMakeLists.txt#L31),
  for Git. Every new dependency must build under the MSVC **14.50-from-BuildTools** pin
  that nvcc 12.9 imposes on the whole project (see `BUILDING_WORKFLOW.md`). That cost
  recurs on every toolchain bump, not once.
* **Two clocks.** tlRender's value is its player: timeline-clock driven, with
  read-ahead/read-behind. CasparCG pulls at channel cadence from `receive_impl`. You
  cannot run both clocks, so you would drive the reader yourself and discard the player
  — which is the part being paid for.
* **Colour overlap.** tlRender does its own OCIO colour management. CasparVP has its own
  ACES chain behind a 1-LSB conformance gate. You would want tlRender's IO layer with
  colour bypassed, removing another large fraction of the library.

### 3.2 xSTUDIO — wrong shape

[xSTUDIO](https://www.dneg.com/creative-technology/xstudio-developer-information) (DNEG,
ASWF-hosted, Apache 2.0) has a good reader-plugin API and a deliberately embeddable
viewport, but it is an application on a CAF actor framework. It is built to embed *other
things into it*, not to be embedded into a playout server.

### 3.3 mrv2, Open RV — applications. Same objection as DJV.

### 3.4 tlRender, revisited — genuinely good at the *other* feature

Playing a timeline as a source is what tlRender is actually built for, and CasparCG has
nothing like it. If §5 becomes a real project and the flat-JSON route in §5.5 proves
insufficient, tlRender is the library to reconsider — on its merits as a timeline
engine, not as an image reader.

### 3.5 OpenImageIO — a metadata argument, not a playback one

[OIIO](https://github.com/AcademySoftwareFoundation/OpenImageIO) (Apache-2.0) is the only
option here that is genuinely library-shaped: no UI toolkit, no actor framework, embeds
cleanly. It deserves a fuller treatment than the others because it is the one that might
eventually be justified. The case for it is narrower than its reputation suggests, and it
is almost entirely about metadata.

**What it would actually buy:**

1. **Colour metadata (the real benefit).** §2.3 measured what FFmpeg discards: EXR
   `chromaticities` and the DPX transfer/colorimetric bytes. OIIO surfaces both as
   attributes, along with timecode and `framesPerSecond` — the last being how a sequence
   could get a truthful frame rate instead of `image2`'s hardcoded 25. For an ACES
   pipeline this is the difference between reading the plate's gamut and asking an
   operator to remember it.
2. **Writing DPX/EXR with correct metadata.** The IMAGE consumer writes PNG only (8/16-bit,
   `6676f7b27`). If plate output for a VFX pull, or ACES EXR masters off the VP stage,
   becomes a requirement, OIIO writes those headers properly. FFmpeg can *encode* both,
   so this too is a metadata-fidelity argument rather than a capability one.
3. **Format reach at the margins.** Cineon — FFmpeg has no decoder at all (§2). Deep EXR,
   RAW via libraw, PSD, HEIF, IFF, RLA. Cineon is essentially dead and the rest are
   irrelevant to playout, so this tier is thin.

**What it does not buy, which is what decides the question:**

* **No read-ahead or streaming scheduler.** `ImageCache` is built for renderers doing
  random tile access, not linear playout at channel cadence. §4.1 still has to be written.
* **No seek / loop / in-out / duration model.** `av_producer` provides all of it, and §2.5
  measured seek as frame-exact. Adopting OIIO as *the* reader means writing a producer
  from scratch and re-earning that behaviour.
* **No frame-rate model**, and its buffers still need converting into `AVFrame` /
  `draw_frame` exactly as FFmpeg's do.
* **It does not fix the 4K problem either.** OIIO's DPX reader is also single-image-serial.
  The answer at 4K is concurrent decode contexts (§2.2, §4.1) regardless of which library
  supplies the reader.

**The expectation it does not meet:** multi-part and named-layer EXR was the assumed
trump card and it is not one — FFmpeg's exr decoder has `-layer` and `-part` (§2.3).
Cruder, but not a wall.

**Cost in this tree.** Imath + OpenEXR, libtiff, libjpeg-turbo, libpng, zlib, fmt, and
optionally OCIO — every one of which must build under the MSVC 14.50-from-BuildTools pin.
The tree has zero image-library dependencies today. And OIIO's `ImageBufAlgo`/OCIO colour
conversion overlaps the fork's own ACES chain, so its use would have to be disciplined to
pixels and metadata only.

**Conclusion — three tiers, in order:**

| tier | what | dependency cost |
| :--- | :--- | :--- |
| 1. now | FFmpeg for pixels (§4.0) plus explicit `COLORSPACE`/`TRANSFER` params (§4.3) | none |
| 2. cheap upgrade | a small header sniffer for EXR `chromaticities` and the DPX transfer/colorimetric bytes, used to **default** those params correctly — ~200 lines | none |
| 3. OIIO | when a concrete requirement appears that tiers 1–2 cannot meet | large |

Tier 2 removes the biggest footgun in the whole feature for no dependency at all, which
is why OIIO is not recommended now. The likeliest trigger for tier 3 is DPX/EXR
**writing** with correct metadata; the least likely is Cineon or deep-EXR ingest.

---

## 4. Phase A — image-sequence playout

### 4.0 The unlock (small)

In [`ffmpeg_producer.cpp`](../../src/modules/ffmpeg/producer/ffmpeg_producer.cpp):

1. Detect a numbered pattern (`av_filename_number_test`-style), or a directory
   containing one, **before** `find_file_within_dir_or_absolute`.
2. On a match, skip `probe_path` and pass the pattern through to `Input` with
   `av_find_input_format("image2")` forced. The mechanism already exists —
   `PROTOCOLS_TREATED_AS_FORMATS` at
   [`av_input.cpp:150-165`](../../src/modules/ffmpeg/producer/av_input.cpp#L150-L165)
   does exactly this shape of thing for `dshow`/`v4l2`/`iec61883`.
3. Teach the extension gate at [`:329-333`](../../src/modules/ffmpeg/producer/ffmpeg_producer.cpp#L329-L333)
   that `.dpx`/`.exr`/`.tif` on a *pattern* are valid even though they are declined for
   single files.
4. A `FRAMERATE` param, because a sequence carries no frame rate of its own — `image2`
   defaults to 25 and that will be wrong roughly always.
5. `CLS` / media-scanner listing: a sequence should appear as one entry, not 4,000.

This part is genuinely small. Everything below is not.

### 4.1 Concurrent read-ahead — mandatory above 1080p

**Not a tuning exercise. §2.2 measured that a single decode context misses 4K50 for DPX
and misses 4K25 for EXR half/ZIP.** Two separate reasons, both unfixable with `-threads`:
the `dpx` decoder declares no threading capability at all, and the `exr` decoder's slice
threading is already saturated at ~3.4x and still too slow. The only remaining lever is
frame-level parallelism — several decode contexts working on different frames at once.

Sizing from §2.2, against a 20 ms budget at 50 fps: 4K DPX needs ~1.4 concurrent
decoders, 4K EXR half/ZIP ~2.7. **A ring of 4 decode contexts covers both with headroom.**

The memory arithmetic is the other half. `av_input` is already threaded, but its buffering
was tuned for compressed streams where a packet is kilobytes. At 8–48 MB/frame the shape
changes: a 2-second 1080p DPX ring is ~400 MB, and 4K EXR f32 is ~4 GB. Ring depth,
back-pressure, and **what a starved producer does** — repeat, drop, or refuse to start —
are all explicit design decisions, and the last one must be a deliberate choice rather
than whatever falls out.

### 4.2 `gbrp10le` / `gbrpf32le` into the mixer — where the bugs will be

The fork's 16-bit/BT.2020 still-image path (`f2bc3cac0`) is the reuse point, but
**planar f32 is new**, and it arrives exactly where the ACES chain wants its input.
This is the interesting part of the whole feature and the part most likely to be subtly
wrong. Note the standing channel-order trap (`CLAUDE.md`): the OGL mixer grades in BGR
and the Vulkan mixer in RGB, greys are invariant under a red/blue exchange, so any
fixture for this must use asymmetric per-channel values.

### 4.3 Colour interpretation

§2.3 measured the problem: FFmpeg reports `color_primaries=unknown` for EXR and
`color_transfer=unknown` for DPX, discarding the EXR `chromaticities` attribute and the
DPX header's transfer/colorimetric bytes. So the producer cannot distinguish an ACES AP0
plate from a linear Rec.709 one, or a Cineon-log DPX from a display-referred one.

Two steps, in order:

1. **Explicit params** (`TRANSFER`, `COLORSPACE`), with the default documented as a
   deliberate choice rather than an accident. Log-encoded plates arriving as if they were
   display-referred is a silent, plausible-looking failure — the worst kind.
2. **A header sniffer to set those defaults correctly** — EXR `chromaticities`, DPX
   transfer/colorimetric. ~200 lines, no new dependency (§3.5 tier 2). This is the
   highest value-per-line item in the entire plan.

### 4.4 Measurement — a precondition, not a follow-up

Per `CLAUDE.md`, this belongs in `d:\Github\CasparCG-TestRunner`, not a scratchpad
script: a new sequence fixture family (DPX 10-bit, EXR half + f32, 16-bit PNG, each with
the binary cell marker so `read_frame_marker` can verify frame identity), a `cli.py`
subcommand, and parity across `--mixer ogl` and `--mixer vulkan`. Until that exists,
nothing about §4.2 or §4.3 is measured.

The throughput figures in §2 were taken with `ffmpeg -benchmark` outside the server, so
they bound the decoder, not the producer. They need a harness equivalent that measures
**dropped frames at channel cadence**, because that is the number that decides whether
4K playout works — and it is the one §4.1's ring depth has to be tuned against.

---

## 5. Phase B — timeline / EDL playout

### 5.1 What an NLE export actually contains

| export | carries | problem |
| :--- | :--- | :--- |
| **CMX3600 EDL** (`.edl`) | reel names, source + record timecodes, cuts, simple dissolves | media mapping is **not in the file** — reel name → path is a separate conform step. No effects, no reliable retimes |
| **Premiere XML / FCPXML / AAF** | transitions, speed, per-clip effects, audio levels | per-NLE, messy, semantics differ between them |
| **OTIO** (`.otio`) | normalized timeline model | Premiere does not export it directly; you arrive via one of the above through an adapter |

### 5.2 The constraint that decides the architecture

OpenTimelineIO's **C++ core reads only `.otio` / `.otioz` / `.otiod`**. Every legacy
adapter — CMX3600, AAF, FCPXML — is a **Python** plugin, and they have been moved out
into separate repositories ([otio-cmx3600-adapter](https://github.com/OpenTimelineIO/otio-cmx3600-adapter),
[otio-aaf-adapter](https://github.com/OpenTimelineIO/otio-aaf-adapter)).

A C++ playout server therefore cannot read a Premiere EDL without embedding Python or
requiring an offline conversion. Read that as a hint about where the seam belongs rather
than an obstacle: **all format-specific mess stays outside the server.**

### 5.3 The primitives that already exist

* `PLAY 1-1 clip SEEK n LENGTH m` — one EDL event with source in/out
* `LOADBG … AUTO` ([`AMCPCommandsImpl.cpp:297`](../../src/protocol/amcp/AMCPCommandsImpl.cpp#L297))
  — preloaded, frame-accurate auto-follow to the next event
* [`transition_producer`](../../src/core/producer/transition/transition_producer.cpp) /
  [`sting_producer`](../../src/core/producer/transition/sting_producer.cpp) — dissolves, stings
* layers + `MIXER` — the V2/V3 tracks and their opacity, geometry and grade

So conforming an EDL is: parse → resolve reels to absolute paths → emit a chain of
`LOADBG`/`AUTO`. **That is a client-side job**, and it is how this ecosystem already
works (`casparcg-360-client` is an AMCP consumer on that pattern).

### 5.4 What the client-side chain cannot do

1. More than ~2 simultaneous video tracks with per-clip grading — layer and mixer
   semantics run out quickly.
2. **Global timecode seek across cut points.** "Go to 01:00:14:07" on the conformed
   sequence. No chain of `LOADBG` can provide this, because nothing owns the timeline.
   **This is the only item that genuinely requires a server-side producer.**
3. Retimes and speed ramps.
4. Prerolling more than one event deep.

If (2) is not required, Phase B should not be built — the client-side chain is the
correct answer and it is achievable in days.

### 5.5 The shape, if (2) is required

**Outside the server** — a conform tool in Python: OTIO plus the CMX3600/AAF/FCPXML
adapters, emitting a flat JSON event list with resolved absolute paths, source in/out,
record in/out and track assignment. Every format-specific horror lives here, where it
can be iterated without a 383-target rebuild.

**Inside the server** — a timeline producer taking that list, owning a global frame
counter, keeping N child producers preloaded, and cutting frame-exactly. This needs
**no new third-party dependency**:
[`isf_producer.cpp:671-690`](../../src/modules/isf/isf_producer.cpp#L671-L690) already
demonstrates a producer constructing child producers through `frame_producer_registry`,
which is the mechanism to build on.

The two phases compose: if a timeline event can point at a sequence pattern, Phase A
becomes a timeline source and DPX/EXR conform playout falls out of the same work.

### 5.6 Out of scope, and worth being blunt about

None of this renders Premiere's **effects**. What round-trips is cuts, dissolves,
opacity, geometry, speed, and this fork's own ACES/grading chain. If the expectation is
"it looks like the Premiere timeline", nothing satisfies that — not DJV, not tlRender,
not xSTUDIO. They conform geometry and time, not effects.

---

## 6. Open questions

1. Is global timecode seek across a conformed sequence actually required, or is the
   requirement "play a cut list back-to-back frame-accurately"? This decides whether
   Phase B exists at all.
2. What sequence formats are really in the pipeline — DPX 10-bit, EXR half/DWAA, 16-bit
   PNG? §2.2 showed compression flips the constraint: uncompressed DPX is I/O-bound,
   compressed EXR is CPU-bound. DWAA/DWAB is unmeasured at any raster, and all figures
   come from synthetic `testsrc2` rather than a real plate.
3. Target raster and rate. This is now the question that decides scope: 1080p is
   comfortable on a single decode context, 4K is not (§2.2). 1080p50 DPX is 396 MB/s;
   4K50 is ~1.58 GB/s and needs a storage conversation before a code conversation.
4. Is DPX/EXR **output** wanted (plate pull, ACES masters)? That is the one requirement
   that would justify OIIO (§3.5 tier 3), and it is worth answering early because it
   changes the dependency decision.
5. Should sequence playout aim at the GPU-direct paths (`cuda_gl_texture` /
   `cuda_vk_texture`) rather than landing in system memory first? Not investigated —
   and §2.2's 4K CPU-decode ceiling makes it more interesting than it first looked.
