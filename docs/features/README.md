# CasparVP features — what this fork adds, and what state each part is in

Everything here is **absent from upstream CasparCG**. The list was produced mechanically rather
than from memory: module directories and registered AMCP command names diffed against
`d:\Github\server-upstream`, so nothing stock is included and nothing fork-specific is missed.

**Numbers, so the scale is honest:** 19 fork-only modules, **58 fork-specific AMCP commands**,
60 documents in `docs/`, 73 harness batteries.

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
| PREVIZ 3D module | 13 | none | **shipped, unmeasured** | `PREVIZ_3D_MODULE.md` |
| Camera tracking | — | none | shipped, unmeasured | `CAMERA_TRACKING.md` |
| Projection calibration | 1 | `calibration` | shipped | `PROJECTION_CALIBRATION.md` |

### GPU pipeline

| feature | coverage | state | document |
| :--- | :--- | :--- | :--- |
| Vulkan mixer | `conformance`, `grading`, `mixer-parity`, `vk-validation` (**cannot currently fail** — see `CLAUDE.md`) | shipped | `VULKAN_MIXER_IMPLEMENTATION.md` |
| CUDA ProRes producer + consumer | `prores-parity`, `producer-swap`, `playback-scaling`, `encode-matrix` | shipped | `CUDA_PRORES` |
| CUDA NotchLC producer | `producer-swap`, `coexistence` | shipped | — |
| Vulkan output consumer | `signalling`, `consumer-view` | partial | `VULKAN_OUTPUT.md` |
| GPU interop (CUDA↔VK, D3D11↔VK, VK↔GL) | `coexistence`, `gpu-direct-parity` | shipped | `GPU_INTEROP_ARCHITECTURE.md` |
| GStreamer producer/consumer | `gstreamer`, `gst-consumer-cost`, `gst-dll-probe` | shipped | `GSTREAMER_GUIDE.md` |
| HAP producer | `loop-boundary` | shipped | — |
| ISF / OpenFX plugin hosts | none | shipped, unmeasured | `OPENFX_INTEGRATION_PLAN.md` |
| Spout / remotewall | none | shipped, unmeasured | — |

### Signal, sync and control

| feature | commands | coverage | state | document |
| :--- | ---: | :--- | :--- | :--- |
| DMX / Art-Net / sACN | — | `dmx` | shipped | `DMX_LIGHTING.md` |
| LTC timecode | 2 | none | shipped, unmeasured | — |
| Cluster sync | — | none | shipped, unmeasured | `CLUSTER_SYNC.md` |
| Keyframes | — | none | shipped, unmeasured | `KEYFRAMES.md` |
| PortAudio | 1 | none | shipped, unmeasured | `PORTAUDIO_MODULE.md` |
| Replay | — | none | shipped, unmeasured | — |
| AMF / PRINT RAW | 2 | `amf` | shipped | — |

---

## Priority order for the documents still to write

Not alphabetical, and not by size. **By where defects have actually hidden**, which the
2026-08-26 audit measured rather than guessed:

1. **Projection and ICVFX** — 12 commands, all undocumented before this folder existed, one
   carrying a live colour defect. Written; see the document above.
2. **PREVIZ** — 13 commands, no documentation, no coverage. The largest undocumented surface in
   the fork and the same shape as ICVFX before it broke.
3. **LTC, keyframes, cluster sync, replay, spout, remotewall** — shipped, unmeasured, undocumented.
4. Everything with an existing guide — those need a state summary and a pointer, not a rewrite.

---

## Template

`_TEMPLATE.md` in this folder. Every section is there because its absence has cost something in
this tree; the template says which, so a section is not dropped for looking like boilerplate.
