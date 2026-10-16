# Reading the numbers — what the test runner measures, and what the units mean

**Who this is for:** anyone who has to read a test result and decide whether to worry —
without being a colour scientist. It explains the method in plain language and defines every
unit the reports use. No prior knowledge of video engineering is assumed.

For *how to run* the tool see `OPERATION_GUIDE.md` in the Test Runner repo. For the
catalogue of what each battery does see `CLAUDE.md` in the Test Runner repo. This document is the
one that explains *why the numbers are the numbers*.

---

## 1. What the tool actually does

In one sentence: **it shows the video server a picture it already knows the right answer
for, captures what comes out the other end, and compares.**

```
  something we control  ->  CasparCG server  ->  captured frame  ->  compared against
  (a flat colour, or                                                  something that is
   a video file)                                                      NOT CasparCG
```

That last point is the whole game. Comparing CasparCG's output to CasparCG's own output
proves only that it is consistent, not that it is *correct*. So every real check compares
against an independent answer, and the tests differ mainly in **where that independent answer
comes from**.

There are two kinds, and almost everything else follows from the difference:

| | what it shows the server | where the right answer comes from | how close it demands |
| :--- | :--- | :--- | :--- |
| **a conformance battery** | a flat colour patch — no video file, no decoding | **arithmetic** — the same formula the shader is supposed to implement, worked out independently in Python | **1 LSB** (see below) |
| **a picture test** | a real video file | an **FFmpeg render** of the same file, or the server's own decoded frame | **a PSNR figure in dB** |

The first is exact and narrow. The second is approximate and realistic. Both exist because
neither is sufficient alone — §5 explains why.

---

## 2. The units, in plain language

### LSB — "one step"

A pixel's brightness is stored as a whole number. In 8-bit video each colour channel runs
**0 to 255**; in 16-bit, **0 to 65535**. **One LSB is one step on that scale** — the smallest
difference that can be stored at all. (It stands for *least significant bit*, the last digit
of the number.)

So "**within 1 LSB**" does not mean "roughly right" or "within 1%". It means:

> the server's answer and the correct answer are either identical or adjacent — there is no
> value in between them that could have been produced.

That is the tightest claim this tool can make. On an 8-bit scale one step is about 0.4% of
full brightness, and far below what an eye can see; the point of the gate is not visibility
but **arithmetic exactness**.

Two conventions you will see:

* **LSB** on its own means *8-bit steps* (out of 255). Most reports use this even when the
  channel is 16-bit, because it is the unit people have intuition for.
* **LSB16** means *16-bit steps* (out of 65535). Used where the fine detail matters — for
  example `render_format_quantum_2026-08-12.md` reports that the `fp16` render target's own
  rounding step is 32 LSB16 near white, which is 2× a 12-bit step.

### dB and PSNR — "how close are these two whole pictures?"

**PSNR** (*peak signal-to-noise ratio*) squashes the difference between two entire images
into a single number, measured in **decibels (dB)**. It is what you use when you cannot
predict the exact answer and can only ask "how far apart are these?".

Two things to know:

* **Higher is better.** Identical images score **infinity** (`inf`).
* **It is a logarithmic scale**, so the steps are not linear. **Every +6 dB means the error
  halved.** A jump from 30 dB to 36 dB is not a 20% improvement — it is twice as accurate.

A rough reading guide, for this pipeline:

| PSNR | what it usually means here |
| :--- | :--- |
| `inf` | byte-for-byte identical |
| 50 dB and up | effectively exact; differences are rounding only |
| 40–50 dB | very close — a lossless or near-lossless codec round trip |
| 30–40 dB | normal for a lossy codec (H.264, H.265, ProRes 422) |
| 25–30 dB | expected only for heavily compressed or texture-compressed sources (HAP, NotchLC) |
| below 25 dB | something is wrong, or the two images are not the same picture at all |

This is why the pass mark is **not one number**. The tool picks the threshold from the codec,
because a lossy codec cannot physically reach the score a PNG can
(`parallel.py::_get_decoder_psnr_threshold`):

| source | gate |
| :--- | ---: |
| PNG, TGA, TIFF | 50 dB |
| QuickTime RLE, EXR | 45 dB |
| ProRes 4444 | 40 dB |
| ProRes 422 | 35 dB |
| H.264, H.265, VP9, AV1, DNxHD, XDCAM | 30 dB |
| HAP, NotchLC | 25 dB |

Holding HAP to a PNG's standard would fail it forever for doing exactly what it is designed
to do.

### SSIM — "does it look like the same picture?"

**SSIM** (*structural similarity*) scores two images from **0 to 1**, where 1 is identical.
Unlike PSNR it is not about how far individual numbers moved but whether the *structure* —
edges, texture, local contrast — survived. A picture can have a mediocre PSNR and a high
SSIM if it was uniformly shifted slightly, and a decent PSNR with a poor SSIM if something
structural broke.

Two things about SSIM **as used here** are worth knowing, because both are unusual:

1. **It is the worst of the red, green and blue channels — not the average.** Averaging the
   three into a single greyscale figure would make the metric completely blind to a red/blue
   channel swap, which scores a perfect 1.0 when averaged. A channel swap is the single most
   common GPU bug in this codebase's history, so the metric is deliberately built to notice
   it (`comparator.py::compute_ssim`).
2. **SSIM never decides pass or fail.** It is computed, printed in the result line and stored
   in the database, but no gate reads it — only PSNR and LSB do. Treat it as a diagnostic
   that helps explain *why* something failed, not as a verdict. This surprises people,
   because it sits right next to PSNR in the output and looks like a second gate.

---

## 3. Which measurement is used when, and why

| situation | measurement | why that one |
| :--- | :--- | :--- |
| flat colour patch, no video file in the path | **LSB, gate 1** | with no decoding and no resampling, the correct answer is *computable*. Anything looser would be throwing away certainty you already have |
| a real video file, compared against an FFmpeg render | **PSNR in dB** | the decode itself is approximate and FFmpeg rounds differently, so exactness is not available. The question becomes "how far apart", not "are they equal" |
| the same test on both mixers (OpenGL and Vulkan) | **compared to each other** | they must agree; a difference is a defect in one of them regardless of which is "right" |
| two frames from one server, differing only in one command | **LSB, and often exactly 0.00** | everything upstream cancels, so no colour model is involved at all — the strongest comparison available |
| structure, channel order, "is this even the right picture" | **SSIM** (reported, not gated) | catches the class of fault where values are plausible but arranged wrongly |

---

## 4. Words the reports use

| term | plain meaning |
| :--- | :--- |
| **battery** | one focused test suite answering one question (e.g. `conformance`, `ocio-look`). Run with `python cli.py <name>` |
| **patch** | a flat rectangle of a single colour, played to the server as the test signal |
| **gate** | the pass mark. "Gated at 1 LSB" = it fails if it is more than one step out |
| **oracle** / **model** | the independent source of the right answer — the Python arithmetic, or an FFmpeg render. If the oracle is wrong the test is worthless, which is why the tree cares so much about where they come from |
| **closed-form** | the answer was calculated directly from a formula rather than measured. It is what makes a 1 LSB gate possible |
| **control** | a case whose result is known in advance, included to prove the test itself is working. See §5 |
| **separation** | how far apart two deliberately-different results are. A test that expects "no change" is only meaningful if a comparable test can show a change |
| **parity** | OpenGL and Vulkan producing the same output. Required, not optional |
| **fp16 band** | an allowance for the fact that a scene-linear composite is stored in 16-bit *floating point*, whose rounding step varies with brightness. Where it applies, the gate is `max(1 LSB, fp16 band)` rather than a flat 1 LSB |
| **LSB / LSB16** | see §2 |
| **verdict** | the pass/fail decision, computed as a pure function of the evidence, so the same evidence always yields the same answer |

---

## 5. What makes a result trustworthy

This is the part that matters most, and it is not about units at all.

**A number matching a model does not prove the feature works.** A command the server accepted
and then ignored will often still produce a picture that matches *some* model — just not the
one you meant. So the batteries are built to fail in that case:

* **A control that must move.** If a test asserts "this setting changes nothing", there is
  always a companion case where the same setting *must* change something. If the second case
  also shows no change, the test reports failure — because the most likely explanation is
  that the command never reached the shader at all, not that the feature works perfectly.
  (Worked example: `cli.py ws-tonemap` asserts 0.00 on two channel types, and 62–71 LSB on a
  third. Without the third, both zeros would be meaningless.)
* **Asymmetric colours.** Test patches deliberately use different values per channel. Greys
  are unchanged by a red/blue swap, so a grey-only test cannot see one of the most common
  bugs here.
* **Both mixers.** A fault present in both OpenGL and Vulkan passes any test that only
  compares them to each other, so absolute checks exist too.

**And a loose gate can hide a real defect.** A worked example from this tree: a bug put a
constant colour cast on every high-bit-depth video source. The 1 LSB batteries could not see
it, because they use flat colours and never invoke the video decoder at all. The picture
tests *did* exercise the faulty code — but 1.5 steps of error on a decoded frame disappears
inside a PSNR taken over a million pixels. It was found only once a test was built that was
both *decoded* and *flat*.

That is the reason both families exist, and the reason a green run is evidence rather than
proof: **a check can only fail in the ways it was built to fail.**

---

## 6. The maths behind each comparison

§1 says every check compares against an answer that does not come from CasparCG. This section
says what that answer is made of. There are four groups, from the strictest to the loosest,
and you can skip the formulas: the point of each item is *where the right answer comes from*.

The rule that ties them together: **every expected value comes from a published standard or an
independent library** (ITU-R, SMPTE, ACES, OpenColorIO, FFmpeg), never from the server's own
code. A model copied from a shader agrees with that shader by construction, mistakes included.

### 6.1 Closed-form models — the 1 LSB batteries

Python works out, independently, what each pixel should be, and the server's answer must land
within one step of it.

| what | how the right answer is obtained | where |
| :--- | :--- | :--- |
| **Gamut conversion** (e.g. BT.2020 → BT.709) | **Derived, not copied.** Each colour space's published primaries and white point give an RGB → XYZ matrix; the two white points are matched with a **Bradford chromatic adaptation**; the product is the RGB → RGB matrix, `M = XYZ→RGB_dst · Bradford(wp_src→wp_dst) · RGB→XYZ_src`. Cross-checked against OpenColorIO. This is how four wrong ACEScg matrices in the server were found | `gamut_reference.py::derived_matrix` |
| **Transfer functions** | Written from the standards: sRGB, BT.709, **PQ (SMPTE ST 2084)**, **HLG (ARIB STD-B67 / BT.2100)** with its system-gamma (OOTF) step, ARRI LogC3, Sony S-Log3, plain power gammas. HLG ↔ PQ applies the BT.2100 OOTF and rescales luminance | `color_math.py` (`_eotf_*`, `_oetf_*`, `hlg_pq_ootf`) |
| **Tone mapping and gamut compression** | Reinhard, the ACES filmic curve, the ACES 1.x RRT, and the gamut compressor (a distance-based compression with per-channel limits) | `color_math.py::apply_tone_mapping`, `apply_gamut_compress` |
| **Grading operators** | One model per operator: ASC CDL (slope / offset / power, then saturation weighted by Rec.709 luma); lift / gamma / gain; white balance; split-tone; linear saturation; HSV hue shift and hue curves; a qualifier with smoothstep edges; curves via a spline lookup table; levels; invert | `grading_math.py` (one `GradeOp` class each) |
| **3D LUTs** | A trilinear interpolation that copies the GPU sampler exactly, including the half-texel coordinate `(v·(N−1) + 0.5) / N` | `grading_math.py::_trilinear` |
| **YCbCr decode** | BT.601 / BT.709 / BT.2020 matrices with the coefficients derived from Kr and Kb, plus the legal-range scaling for each bit depth | `decode_math.py::ycbcr_to_rgb` |
| **Blending and compositing** | Closed-form blend modes (overlay, soft light and others), premultiplied-alpha compositing, opacity. For geometry, polygon clipping and point-in-polygon tests predict which pixels a crop or clip region keeps | `mix_cases.py` |
| **Precision limits** | The size of one storage step for each render format: one 16-bit step for unorm, and for fp16 the gap between adjacent half-precision values, `2^(e−10)` (10 mantissa bits). This separates "banding the format cannot avoid" from "banding a bug added" | `banding.py::expected_quantum`, `fp16_ulp` |
| **OCIO** | The official OpenColorIO library, pinned to the version the server links. The gate allows one fp16 step, so the GPU and CPU implementations are compared within a stated limit rather than an arbitrary one | `ocio_reference.py` |
| **Projection and geometry** | For the screen view: a camera projection re-derived from the documented conventions predicts where a screen lands, in pixels. For lens distortion: closed-form identities instead of a full model — radial distortion at radius `r0` is exactly a field-of-view change, `fov' = 2·atan(radial(r0)·tan(fov/2))`, and the tangential terms are told apart by the axis each one moves. The four lens models (rectilinear, cylinder, sphere, fisheye) are separated by their latitude laws down the centre column, and the declared model must **win** against the other three rather than merely fit | `previz_place.py`, `projection.py` |

### 6.2 Picture metrics — real media

Used where exactness is not available (§2 explains the units).

* **PSNR** — `10 · log10(MAX² / MSE)`, overall and per channel, at 8 or 16 bits. PNGs are read
  at their full 16 bits, because a standard 8-bit read silently throws precision away
  (`comparator.py::compute_psnr`, `compute_psnr_per_channel`).
* **SSIM** — scikit-image's implementation, scored per channel, keeping the *worst* channel
  (`comparator.py::compute_ssim`).
* **The reference** is FFmpeg rendering the same file, never the server.

### 6.3 Diagnosing a failure

When a comparison fails, these say *what kind* of wrong it is.

| check | how it works | what it tells you | where |
| :--- | :--- | :--- | :--- |
| **Red/blue swap** | Swap R and B in the capture and recompute PSNR. More than 10 dB better means the channel order is wrong | a byte-order bug, not a colour-maths one | `comparator.py::detect_channel_swap` |
| **Garbled frame** | Mean horizontal gradient energy: above 0.15 is noise or tearing. Also flags frames that are mostly black with a noisy remainder | a corrupt or partially rendered frame | `comparator.py::detect_garbled_frame` |
| **Shift probe** | Search a few pixels of offset for the best-aligned PSNR | a non-zero winner means the colours are right and the sampling is displaced — a geometry bug | `diagnostics.py::shift_probe` |
| **Boundary split** | PSNR computed separately for the interiors of flat patches and for the ring of pixels where the colour changes | "the colour maths is off" versus "the sampling or chroma siting is off". In one case 36.4 dB became 44.9 dB once the 1.5% of edge pixels were excluded, which cleared the HLG maths entirely | `diagnostics.py::boundary_split` |
| **Combing score** | Mean difference between adjacent rows, compared only against a progressive copy of the same content | an interlacing fault. Fine horizontal detail scores the same as combing, so an absolute value means nothing | `diagnostics.py::comb_score` |
| **Three-way comparison** | Decoded frame vs mixer output vs reference | which stage introduced the error: decoder or mixer | `comparator.py::compare_images` |

### 6.4 Timing and statistics

* **Burnt-in frame marker.** Each test clip carries a 16-cell binary code: a white and a black
  sentinel (so the on/off threshold is measured from the picture itself and survives a colour
  conversion), 12 payload bits, even parity, and a closing white sentinel. It turns seek and
  frame-accuracy checks into exact comparisons — "the frame that came back IS frame 42" —
  rather than "the picture changed" (`frame_number.py::marker_bits`).
* **Flaky tests.** Failure rates carry a **Wilson score interval**, so "15 of 15 passed" reads
  honestly as "the failure rate is below about 18%" rather than "never fails". Two builds run
  interleaved are compared with **Fisher's exact test**, to decide whether a change in flake
  rate is real (`flake_rate.py::wilson_interval`, `fisher_exact_two_sided`).
* **Testing the tests.** A known bug is planted and the check must go red; a check that stays
  green on its own mutation cannot fail (`mutation_battery.md`).

---

## 7. Reading a result line

```
PASS | PSNR=52.3dB SSIM=0.9987 max_diff=0.004
```

* **PASS/FAIL** — the verdict, from the gate that applies to this case.
* **PSNR** — how close the capture is to the reference, in dB. Compare it to the gate for
  this codec (§2), not to some universal standard.
* **SSIM** — structural similarity, worst of R/G/B. Diagnostic only; it did not decide the
  verdict.
* **max_diff** — the largest single-pixel difference, normalised to 0–1. Useful when PSNR
  looks acceptable but one region is badly wrong: PSNR is an average and can absorb a small
  catastrophe.

A battery report instead reads like:

```
#F7F5F2  separation    71.00 LSB
#E01F1A  separation     0.00 LSB
```

which is two captures from one server compared with each other, in steps of 255 — no model,
no reference file, and therefore nothing upstream that could explain a difference away.

---

## 8. If you remember four things

1. **1 LSB = one step out of 255.** It means "exact", not "close enough".
2. **dB is logarithmic and higher is better; +6 dB is twice as accurate.** The pass mark
   depends on the codec, because lossy codecs cannot reach a lossless score.
3. **SSIM is reported but never gates**, and it is the worst channel rather than the average
   so that a red/blue swap cannot hide in it.
4. **A test that cannot fail proves nothing** — which is why almost every check here comes
   with a control that must move.
