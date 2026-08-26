# CasparVP features — what this fork adds, and what state each part is in

Everything here is **fork-specific work**: either a module upstream does not have, or an upstream
module this fork has substantially rewritten. Produced mechanically rather than from memory —
module directories and registered AMCP command names diffed against `d:\Github\server-upstream`.

> **The second half of that sentence was added on 2026-08-27, and the omission was the point.** The
> original rule was "absent from upstream", which is a test for a fork-only *module* and not for
> fork-specific *work* — so it silently excluded the four heaviest divergences in the tree:
> `ffmpeg` (**10,500** changed lines), `screen` (2,963), `decklink` (2,382) and `image` (789). The
> folder claimed "nothing fork-specific is missed" while missing more changed code than everything
> it covered. A mechanical criterion is only as good as the question it encodes.

**Numbers, so the scale is honest:** 19 fork-only modules, **91 fork-specific AMCP commands**,
60 documents in `docs/`, 73 harness batteries.

> **That command count was 58 until 2026-08-26, and the correction is instructive.** The first
> count scanned only `AMCPCommandsImpl.cpp` and missed every command a MODULE registers for
> itself -- tracking (18), keyframes (8), cluster (4), GStreamer (2), Vulkan output (1). A 36 %
> undercount in the one document whose job is to be the inventory. Counted now across all of
> `src/**/*.cpp` and diffed against upstream's own registrations, which is the only way that
> holds as modules come and go.

## Every module, and why it is or is not here

Regenerate by diffing `src/modules/*` against `d:\Github\server-upstream`. The point of the table
is that **omissions are stated rather than inferred from absence** — the previous version of this
folder had no such list, so five undocumented modules looked exactly like five modules that did not
need documenting.

| module | vs upstream | document |
| :--- | ---: | :--- |
| `ffmpeg` | 10,500 lines | `ffmpeg-producer-and-consumer.md` |
| `screen` | 2,963 | `screen-consumer.md` |
| `decklink` | 2,382 | `decklink-output.md` |
| `image` | 789 | `image-consumer-and-producer.md` |
| `html` | 282 | `html-gpu-direct.md` |
| `newtek` | 56 | **none — checked, not overlooked.** NDI wiring only |
| `oal` | 47 | **none — checked.** audio device plumbing |
| `bluefish` | 27 | **none — checked.** covered as a gap in `decklink-output.md` §3 |
| `flash` | **0** | **none — untouched by this fork.** Upstream's document applies verbatim |
| fork-only modules (19) | n/a | one document each, listed below |

`ofx` and `dmx_common` are fork-only but share documents with their neighbours —
`isf-and-openfx.md` and `dmx-sacn-artnet.md` — because they are one feature between them.

---

## How to read these documents, and why they exist

`docs/` already holds 6 MB across 60 files, and every fork feature is *mentioned* somewhere. The
problem was never missing writing — it is that the material sits in four different genres with no
canonical entry point per feature:

| genre | example | what it is good for |
| :--- | :--- | :--- |
| **plan** | `GSTREAMER_INTEGRATION_PLAN.md` (59 KB) | what was intended, before it was built |
| **study** | `OCIO_INTEGRATION_STUDY.md` (44 KB) | why an approach was chosen |
| **guide** | `GSTREAMER_GUIDE.md` (40 KB) | how an operator drives it |
| **report** | `UPSTREAM_SYNC_2026-08-18.md` (65 KB) | what happened once |

None of those answers *"what is implemented today, and can I trust it"*. That is what a document
in this folder answers, and it is the only question it answers.

**These documents are derived from the CODE and the COMMITS, not from the documents above.** That
is a deliberate rule, not a stylistic preference. `CLAUDE.md` says docs in this tree are claims to
verify against the source rather than ground truth, and on 2026-08-26 alone an audit found four
doc claims that had outlived their code — a battery recorded as owed that already existed, a
render field described as needed that both mixers had, a VRAM figure understating every path by
2.5×, and a picture measurement four days stale. Aggregating existing prose would launder those
into something that reads more authoritative than its sources.

So every claim here carries its evidence: a file and line, a commit, or a battery and its
numbers. Where something is unverified, it says so.

---

## State vocabulary

Used consistently below, and meant literally:

| state | meaning |
| :--- | :--- |
| **shipped** | implemented, driven by a harness battery, numbers recorded |
| **shipped, unmeasured** | implemented and in use, but no battery can currently fail for it |
| **partial** | some of the surface works; the gaps are named in the feature's own document |
| **experimental** | present and reachable, not relied on, may change |
| **planned** | a document exists; the code does not |

**"shipped, unmeasured" is not a lesser grade of shipped — it is a warning.** The ICVFX gain was
in that state and carried a red/blue exchange on the OpenGL mixer for an unknown period, because
nothing drove it. Undocumented and untested proved to be the same gap seen from two sides.

---

## The features

### Colour and grading

| feature | commands | coverage | state | document |
| :--- | ---: | :--- | :--- | :--- |
| **Grading chain** (CDL, lift/midtone/gain, curves, hue curves, qualifier, split tone, white balance, exposure, gain, RGB levels, tone balance, linear saturation, sharpen, blur, grain, shape) | ~26 | `grading`, `grade-extremes`, `grade-window`, `banding`, `bokeh-luma` | shipped | **[colour-grading-and-ocio.md](colour-grading-and-ocio.md)** |
| **ACES / colour management, gamut compression** | 2 | `conformance`, `gamut-compress`, `gamut-sweep`, `flat-gamut-compress`, `ws-tonemap` | shipped | **[colour-grading-and-ocio.md](colour-grading-and-ocio.md)** |
| **OCIO integration** | 4 | `ocio`, `ocio-display`, `ocio-look`, `ocio-lut3d`, `ocio-exposure`, `ocio-gamut-compress` | shipped | **[colour-grading-and-ocio.md](colour-grading-and-ocio.md)** |
| 3D LUTs | 1 | `ocio-lut3d`, `cdl-file` | shipped | [colour-grading-and-ocio.md](colour-grading-and-ocio.md) |

### Projection and virtual production

| feature | commands | coverage | state | document |
| :--- | ---: | :--- | :--- | :--- |
| **Projection warp, blend, curve, distortion, frustum, lens, offset** | 10 | `geometry`, `blend-mask`, `calibration`, `venue-test` | **partial** | **[projection-and-icvfx.md](projection-and-icvfx.md)** |
| **ICVFX inner/outer frustum** | 2 | `icvfx-parity` (gain only) | **partial** | **[projection-and-icvfx.md](projection-and-icvfx.md)** |
| **PREVIZ 3D module** | 13 | **none** | **shipped, unmeasured** | **[previz.md](previz.md)** |
| **Camera tracking** | **18** | **none** | shipped, unmeasured | **[camera-tracking.md](camera-tracking.md)** |
| Projection calibration | 1 | `calibration` | shipped | `PROJECTION_CALIBRATION.md` |

### GPU pipeline

| feature | coverage | state | document |
| :--- | :--- | :--- | :--- |
| **Vulkan mixer** | `conformance`, `grading`, `mixer-parity`, `vk-validation` (**cannot currently fail**) | shipped | **[vulkan-mixer.md](vulkan-mixer.md)** |
| **CUDA ProRes producer + consumer** | `prores-parity`, `producer-swap`, `playback-scaling`, `encode-matrix`, `coexistence` | shipped | **[cuda-prores.md](cuda-prores.md)** |
| **CUDA NotchLC producer** | `producer-swap`, `coexistence` | shipped | **[cuda-notchlc.md](cuda-notchlc.md)** |
| **Vulkan output consumer** | `consumer-view` only — **metadata uncovered** | partial | **[vulkan-output.md](vulkan-output.md)** |
| GPU interop (CUDA↔VK, D3D11↔VK, VK↔GL) | `coexistence`, `gpu-direct-parity` | shipped | `GPU_INTEROP_ARCHITECTURE.md` |
| **GStreamer producer/consumer** | `gstreamer`, `gst-consumer-cost`, `gst-dll-probe`, `coexistence` | shipped | **[gstreamer.md](gstreamer.md)** |
| **HAP producer** | `loop-boundary` | shipped | **[hap.md](hap.md)** |
| **ISF / OpenFX plugin hosts** | **none** | shipped, unmeasured | **[isf-and-openfx.md](isf-and-openfx.md)** |
| **Spout** | **none** | shipped, unmeasured | **[spout.md](spout.md)** |
| **remotewall** | **none** | shipped, unmeasured | **[remotewall-and-portaudio.md](remotewall-and-portaudio.md)** |

### Signal, sync and control

| feature | commands | coverage | state | document |
| :--- | ---: | :--- | :--- | :--- |
| **DMX / Art-Net / sACN** | — | `dmx` | shipped | **[dmx-sacn-artnet.md](dmx-sacn-artnet.md)** |
| **LTC timecode** | 2 | **none** | shipped, unmeasured | **[ltc-timecode.md](ltc-timecode.md)** |
| **Cluster sync** | 4 | **none** | shipped, unmeasured | **[cluster-sync.md](cluster-sync.md)** |
| **Keyframes** | 8 | **none** | shipped, unmeasured | **[keyframes.md](keyframes.md)** |
| **PortAudio** | 1 | **none** | shipped, unmeasured | **[remotewall-and-portaudio.md](remotewall-and-portaudio.md)** |
| **Replay** | — | **none** | shipped, unmeasured | **[replay.md](replay.md)** |
| AMF / PRINT RAW | 2 | `amf` | shipped | — |

---

## Priority order for the documents still to write

Not alphabetical, and not by size. **By where defects have actually hidden**, which the
2026-08-26 audit measured rather than guessed:

1. **Projection and ICVFX** — 12 commands, all undocumented before this folder existed, one
   carrying a live colour defect. Written; see the document above.
2. **PREVIZ** — written; see the document above. 13 commands, still **no coverage**, and the
   document says what a first battery should check and in what order.
3. **CUDA ProRes** — written. Chosen third because it is the most heavily measured part of the
   fork, so its document is mostly a place to put numbers that were scattered across commits.
4. **LTC, Spout, HAP, NotchLC, DMX/sACN/Art-Net** — written. Each had no document of its own;
   each turned out to contain something not written down anywhere, which is the argument for
   doing the rest. Three examples: the FFmpeg consumer asks LTC for a frame number with a
   hardcoded `25` and a question mark in the comment; the Spout producer accepts three different
   syntaxes for the same thing; and the fork's three GPU codec producers use three *different*
   mixer-handoff strategies, each for a defensible reason and none of them the house style.
5. **Camera tracking, keyframes, cluster sync** — written. All three register their own AMCP
   commands, which is how the inventory came to be 36 % short before this batch. Camera tracking
   turned out to be the **second-largest command family in the fork** (18) with no coverage at all,
   and three of its commands (`POSITION_SCALE`, `WORLDALIGN`, `ZOOM_LUT`) appear in no document
   including its own guide.
6. **Replay, remotewall, PortAudio, ISF/OpenFX** — written. Every fork module now has a feature
   document or is covered by one.
7. **Vulkan mixer, Vulkan output, GStreamer, grading + OCIO** — written, as state-and-coverage
   summaries pointing at the guides rather than new prose. Even here each surfaced something the
   guides do not say in one place: the Vulkan output consumer's **signalling has no coverage at
   all** and a metadata defect leaves the picture correct, which makes it the highest-value missing
   battery in the fork; and the GStreamer GPU route refuses `d3d11convert` for the same reason CUDA
   ProRes went planar — two modules reaching one conclusion independently.

**Every fork feature now has a document.** What remains is diagrams and the HTML build, below.

## What this folder does not yet have

**Diagrams — three drawn, three still owed, two deliberately deferred.**

Drawn, by `docs/diagrams/generate_feature_diagrams.py`, following the palette and the
`layout_check.Layout` conventions of the existing generators:

| figure | in | why it earns one |
| :--- | :--- | :--- |
| `feature_codec_handoff.png` | ProRes, NotchLC, HAP | three producers hand the mixer three different things — two paths reaching one place |
| `feature_notchlc_pipeline.png` | NotchLC | ten steps with a host round-trip at step 3; the order is the point |
| `feature_projection_order.png` | projection and ICVFX | the chain's order is the point |

**The projection figure was drawn wrong first, and that is worth recording.** An earlier draft
guessed `LENS → CURVE → PROJECTION → DISTORTION` from the command names. The shader does neither:
the curve is applied to the destination uv *first*, and the distortion is applied *before* the lens
model, which is one step with the rotation rather than a separate one. Reading `shader.frag` fixed
it. **A diagram is a claim, and a guessed order is the easiest kind to publish confidently.**

`layout_check` earned its place too: it rejected overlapping panels and arrows crossing label text
that a renderer would have drawn without complaint.

Still owed: **PREVIZ** (two routes to one piece of ICVFX state) and **camera tracking** (five
alignment commands composing in an order that is itself undocumented — so drawing it would
probably answer that gap).

Deliberately deferred: **cluster sync** and **replay**. Both would illustrate timing that nothing
measures, and per the rule above a diagram is a claim like any other. Worth drawing once their
first batteries exist.

**An HTML build.** The markdown is the source of truth so it diffs and reviews; a sectioned HTML
render for reading is a separate step.
7. **Vulkan output consumer, GStreamer, Vulkan mixer, grading, OCIO** — large, well documented
   elsewhere, and last precisely because they are the best covered.


---

## Template

`_TEMPLATE.md` in this folder. Every section is there because its absence has cost something in
this tree; the template says which, so a section is not dropped for looking like boilerplate.
