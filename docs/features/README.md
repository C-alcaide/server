# CasparVP features — what this fork adds, and what state each part is in

Everything here is **absent from upstream CasparCG**. The list was produced mechanically rather
than from memory: module directories and registered AMCP command names diffed against
`d:\Github\server-upstream`, so nothing stock is included and nothing fork-specific is missed.

**Numbers, so the scale is honest:** 19 fork-only modules, **91 fork-specific AMCP commands**,
60 documents in `docs/`, 73 harness batteries.

> **That command count was 58 until 2026-08-26, and the correction is instructive.** The first
> count scanned only `AMCPCommandsImpl.cpp` and missed every command a MODULE registers for
> itself -- tracking (18), keyframes (8), cluster (4), GStreamer (2), Vulkan output (1). A 36 %
> undercount in the one document whose job is to be the inventory. Counted now across all of
> `src/**/*.cpp` and diffed against upstream's own registrations, which is the only way that
> holds as modules come and go.

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
| Grading chain (CDL, lift/midtone/gain, curves, hue curves, qualifier, split tone, white balance, exposure, gain, RGB levels, tone balance, linear saturation, sharpen, blur, grain, shape) | 20 | `grading`, `grade-extremes`, `grade-window`, `banding`, `bokeh-luma` | shipped | `COLOR_GRADING.md` |
| ACES / colour management, gamut compression | 2 | `conformance`, `gamut-compress`, `gamut-sweep`, `flat-gamut-compress`, `ws-tonemap` | shipped | `COLOR_GRADING.md`, `HDR_GUIDE.md` |
| OCIO integration | 4 | `ocio`, `ocio-display`, `ocio-look`, `ocio-lut3d`, `ocio-exposure`, `ocio-gamut-compress` | shipped | `OCIO_USER_GUIDE.md` |
| 3D LUTs | 1 | `ocio-lut3d`, `cdl-file` | shipped | `COLOR_GRADING.md` |

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
| Vulkan mixer | `conformance`, `grading`, `mixer-parity`, `vk-validation` (**cannot currently fail** — see `CLAUDE.md`) | shipped | `VULKAN_MIXER_IMPLEMENTATION.md` |
| **CUDA ProRes producer + consumer** | `prores-parity`, `producer-swap`, `playback-scaling`, `encode-matrix`, `coexistence` | shipped | **[cuda-prores.md](cuda-prores.md)** |
| **CUDA NotchLC producer** | `producer-swap`, `coexistence` | shipped | **[cuda-notchlc.md](cuda-notchlc.md)** |
| Vulkan output consumer | `signalling`, `consumer-view` | partial | `VULKAN_OUTPUT.md` |
| GPU interop (CUDA↔VK, D3D11↔VK, VK↔GL) | `coexistence`, `gpu-direct-parity` | shipped | `GPU_INTEROP_ARCHITECTURE.md` |
| GStreamer producer/consumer | `gstreamer`, `gst-consumer-cost`, `gst-dll-probe` | shipped | `GSTREAMER_GUIDE.md` |
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
7. **Vulkan output consumer, GStreamer, Vulkan mixer, grading chain, OCIO** — deliberately last.
   These are the best-documented and best-covered parts of the fork, so a feature document adds
   least here; what they need is a state-and-coverage summary pointing at the guides and batteries
   that already exist, not new prose.

## What this folder does not yet have

**Diagrams.** Five documents record one as owed and name which criterion it meets — projection
(transform order), PREVIZ (two routes to one state), CUDA ProRes (order, and two handoff paths),
CUDA NotchLC (a ten-step chain with a host round-trip in the middle), camera tracking (five
alignment commands composing in an undocumented order). Two more deliberately *defer* rather than
owe: cluster sync and replay would both illustrate timing that nothing measures, and a diagram is
a claim like any other.

**An HTML build.** The markdown is the source of truth so it diffs and reviews; a sectioned HTML
render for reading is a separate step.
7. **Vulkan output consumer, GStreamer, Vulkan mixer, grading, OCIO** — large, well documented
   elsewhere, and last precisely because they are the best covered.


---

## Template

`_TEMPLATE.md` in this folder. Every section is there because its absence has cost something in
this tree; the template says which, so a section is not dropped for looking like boilerplate.
