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

A rough reading guide, for two pictures that are meant to be the same:

| PSNR | what it usually means |
| :--- | :--- |
| `inf` | byte-for-byte identical |
| 50 dB and up | effectively exact; differences are rounding only |
| 40–50 dB | very close — rounding plus small differences in how chroma is upsampled or how YCbCr is converted to RGB |
| 30–40 dB | a visible-on-inspection difference: worth explaining |
| below 30 dB | the two were processed differently in some way that matters, or are not the same picture at all |

#### What the per-codec decode gate measures — and what it does not

The decode check compares **CasparCG's decode of a test file** against **FFmpeg's decode of
the same file** (`parallel.py::_ensure_source_reference`). Both start from the same compressed
bytes, so **the codec's compression loss is in both pictures and cancels out**. The score is
how far apart two *decoders* are — never how lossy the *codec* is.

**This section used to say the opposite**: that the gate varies by codec "because a lossy codec
cannot physically reach the score a PNG can". That describes a different measurement — a
decode compared against the original master before encoding — which the harness makes only where
a master can be regenerated (`output.decode_vs_master`, below). Corrected 2026-10-16.

The gates as they stand (`parallel.py::_get_decoder_psnr_threshold`), next to what was
measured on this box (`data/results.db`; runs up to 2026-08-21 unless stated):

| source | gate | measured mean (runs) |
| :--- | ---: | :--- |
| PNG, TGA, TIFF | 50 dB | — |
| QuickTime RLE, EXR | 45 dB | — |
| ProRes 4444 | 40 dB | 49.9 dB (140) |
| ProRes 422 | 35 dB | 47.5 dB (209) |
| H.264, H.265, VP9, AV1, DNxHD, XDCAM | 30 dB | 37.5–51.0 dB depending on codec |
| HAP | 25 dB | **nothing recorded** — a placeholder with nothing behind it |
| NotchLC | **40 dB** (was 25) | **53.5 dB** CUDA decoder / **58.7 dB** FFmpeg route, every case on both mixers (2026-10-16). Before the fix: 27.5 / 28.2 dB |

Read the gates as **"how well two decoders are expected to agree"**. The spread between codecs
comes from things like bit depth, chroma subsampling and which YCbCr → RGB conversion each side
applies.

**NotchLC's 25 dB gate was hiding a server defect — found 2026-10-16, fixed the same day.**
The ~28 dB looked like the codec's loss and was not. FFmpeg tags NotchLC `colorspace=RGB`, which
names no YUV matrix: FFmpeg's own converter reads that as BT.601, and **both CasparCG routes read it
as BT.709**. Each decode was internally consistent — they simply used different matrices, and the
25 dB gate had been set to fit the result. The tell, in hindsight: CasparCG's FFmpeg route uses
*the same decoder* as the reference and still landed 28 dB away, so the difference had to be in the
conversion to RGB. (This section's first correction guessed the 8-bit reference was to blame; it was
not — the reference was right.)

Which matrix is right was settled against the **master**: `notchlc_none_sdr.mov` is FFmpeg's
`smptehdbars` colour bars under the frame marker, which regenerates exactly. The decoded planes land
**48.0 dB** from the master through BT.601 full range and **29.0 dB** through BT.709 — about 5 LSB of
error on every saturated colour and none on greys, which is why a grey ramp would never have shown it.
Both routes now use BT.601; the gate is 40 dB.

**HAP's 25 dB is still an open question**: no HAP decode has ever been recorded against it. HAP *is*
lossy by design (GPU texture compression), but that loss cancels in this check like any other codec's,
so 25 dB is not evidence of anything until someone measures it.

#### `output.decode_vs_master` — how lossy the codec actually is

Where a fixture's master can be regenerated, every case also compares CasparCG's decoded frame
against **the picture before encoding** (`codec_master.py`). This one *does* include the codec's own
loss, so it answers "NotchLC vs ProRes"-style questions. Gate 45 dB. Measured 2026-10-16, both
mixers, 8/8 each:

| route | decode vs master | mean error |
| :--- | ---: | :--- |
| CasparCG FFmpeg route | 50.4 dB | under 1 LSB |
| CasparCG CUDA decoder | 48.0 dB | 0.77 LSB |
| the same planes through BT.709 (the old behaviour) | 29.0 dB | ~5 LSB on saturated colour |

So **NotchLC decoded correctly is "very close" territory**, not "heavily compressed". Two limits:
only NotchLC has a regenerable master today (no other fixture's source is kept), and the check
**confirms the master first** — it must match FFmpeg's decode of the fixture at 40 dB or more, or the
check skips, so a replaced fixture cannot fail every decoder for a reason that has nothing to do with
them. The master is opaque while the fixture carries an alpha ramp; the check compares colour only.

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
  honestly as "the failure rate is below about 20%" rather than "never fails". Two builds run
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
2. **dB is logarithmic and higher is better; +6 dB is twice as accurate.** The decode gate
   compares two decoders of the same file, so it measures their agreement, not how lossy the
   codec is.
3. **SSIM is reported but never gates**, and it is the worst channel rather than the average
   so that a red/blue swap cannot hide in it.
4. **A test that cannot fail proves nothing** — which is why almost every check here comes
   with a control that must move.

---

## Appendix — more depth, with worked examples

For readers who want to see the arithmetic. Nothing above depends on it. The code is quoted
from the harness itself, trimmed of comments; the numbers were produced by running it.

### A.1 What the harness is built on

| library | what it is used for |
| :--- | :--- |
| **NumPy** | almost everything: 3×3 gamut matrices and their inverses, transfer curves, image differences and PSNR, LUT interpolation, the diagnostics |
| Python's `math`, `statistics`, `fractions` | scalar arithmetic: projection identities, fp16 step sizes, exact frame-rate maths |
| **scikit-image** | SSIM only (`skimage.metrics.structural_similarity`) |
| **PyOpenColorIO** | the official OpenColorIO library, as the independent reference for the OCIO checks — pinned to the version the server links, so a library difference cannot read as an integration bug |
| **Pillow**, **imageio**, **pypng** | reading and writing PNGs; imageio and pypng because they keep 16-bit precision |
| **FFmpeg** (external program) | renders the reference frames for picture tests, and generates the test media |

There is no SciPy: the two statistical tests in A.4 are a few lines of plain Python each.
**None of the harness imports or reuses the server's code.** That is the point — a model that
shares code with what it checks agrees with it by construction.

### A.2 Worked example: deriving a gamut matrix

Converting a colour from BT.709 (HD) to BT.2020 (UHD) is a 3×3 matrix. The server has its own
copy of that matrix; the harness *derives* it from first principles instead, using only the
numbers the standards publish:

* each colour space's **primaries** — the chromaticity (x, y) of its pure red, green and blue;
* its **white point** — both of these are D65, (0.3127, 0.3290).

The recipe, in three steps:

1. **RGB → XYZ for each space.** XYZ is the CIE's device-independent colour space. Put each
   primary's chromaticity in a column, then scale the columns so that RGB (1, 1, 1) lands
   exactly on the white point.
2. **Adapt the white point** with the **Bradford** transform, if the two whites differ. It
   converts to a cone-response space, scales each channel by the ratio of the two whites,
   and converts back. For BT.709 → BT.2020 the whites are the same and this step is the
   identity; for ACES (D60) → BT.709 (D65) it is not.
3. **Chain them:** `M = (RGB→XYZ of the target)⁻¹ · Bradford · (RGB→XYZ of the source)`.

```python
# core/gamut_reference.py
_BRADFORD = np.array([[ 0.8951, 0.2664, -0.1614],
                      [-0.7502, 1.7135,  0.0367],
                      [ 0.0389, -0.0685, 1.0296]])

def _rgb_to_xyz(prim, wp):
    (xr, yr), (xg, yg), (xb, yb) = prim
    M = np.array([[xr / yr, xg / yg, xb / yb],
                  [1.0, 1.0, 1.0],
                  [(1 - xr - yr) / yr, (1 - xg - yg) / yg, (1 - xb - yb) / yb]])
    xw, yw = wp
    # scale each column so that RGB (1,1,1) maps onto the white point
    return M * np.linalg.solve(M, np.array([xw / yw, 1.0, (1 - xw - yw) / yw]))

def _adapt(src_wp, dst_wp):
    f = lambda wp: np.array([wp[0] / wp[1], 1.0, (1 - wp[0] - wp[1]) / wp[1]])
    a, b = _BRADFORD @ f(src_wp), _BRADFORD @ f(dst_wp)
    return np.linalg.inv(_BRADFORD) @ np.diag(b / a) @ _BRADFORD

def derived_matrix(src, dst):
    pa, wa = PRIMARIES[src]
    pb, wb = PRIMARIES[dst]
    return np.linalg.inv(_rgb_to_xyz(pb, wb)) @ _adapt(wa, wb) @ _rgb_to_xyz(pa, wa)
```

Running it for BT.709 → BT.2020 gives:

```
[[0.627404  0.329283  0.043313]
 [0.069097  0.919540  0.011362]
 [0.016391  0.088013  0.895595]]
```

That is the matrix ITU-R publishes in **BT.2087** (0.6274 0.3293 0.0433 / 0.0691 0.9195 0.0114 /
0.0164 0.0880 0.8956), reached without copying it. Two sanity checks anyone can do by eye:
each row sums to 1.000 (white stays white), and the diagonal dominates (red stays mostly red).

This is how four wrong matrices in the server were found: the derivation and OpenColorIO
agreed with each other, and the server's constants agreed with neither. Note the limit,
though — the server's wrong matrices **round-tripped perfectly** (converting there and back
returned the original colour), because they were consistent inverses of each other. A round
trip proves self-consistency; only an external reference proves correctness.

### A.3 Worked example: one 1 LSB check

A conformance battery plays a flat patch and asks the model what must come out. Here, an
8-bit patch of RGB (200, 60, 30) — deliberately unequal channels, see §5 — on a BT.709
source, converted to a BT.2020 output, both using the BT.709 transfer function:

```python
from core import color_math as cm

signal   = (200/255, 60/255, 30/255)
expected = cm.compute_expected_rgb(signal,
                                   source_transfer="bt709", target_transfer="bt709",
                                   source_gamut="bt709",    target_gamut="bt2020")
[round(v * 255, 3) for v in expected]      # -> [166.728, 82.77, 46.858]
```

Inside `compute_expected_rgb` the steps are: decode the signal to linear light with the
BT.709 transfer function, apply the A.2 matrix, re-encode with the target's transfer
function, and scale to code values.

The expected answer is a fraction, but a pixel can only hold a whole number, so the gate is
`|captured − expected| ≤ 1` for each channel. For red that admits **166 or 167 and nothing
else**. To illustrate the verdict (these captures are made up, not measured):

| captured | red error | verdict |
| :--- | ---: | :--- |
| (167, 83, 47) | 0.27 | PASS — the nearest whole numbers to the model |
| (166, 82, 46) | 0.73 | PASS — still adjacent to the exact answer |
| (168, 83, 47) | 1.27 | **FAIL** — a value exists between this and the right answer |
| (47, 83, 167) | 119.7 | **FAIL** — red and blue exchanged; a grey patch would have passed this |

The real batteries run dozens of patches like this across every supported combination of
colour space and transfer, on both mixers.

### A.4 Why a Wilson interval, and not "k out of n"

Some failures are intermittent, so the harness reruns a check and reports a failure *rate*.
The obvious estimate — failures ÷ runs, plus or minus the textbook normal approximation —
collapses at the edges: 15 passes out of 15 gives a failure rate of 0% **with zero
uncertainty**, which claims far more than 15 runs can show.

The **Wilson score interval** keeps an honest width at the edges:

```python
# core/flake_rate.py
def wilson_interval(k, n, z=1.96):          # z = 1.96 -> a 95% interval
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = (z / d) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))
```

`wilson_interval(15, 15)` returns **(0.796, 1.000)**: the pass rate is at least about 80%, so
the honest reading of "15 out of 15 passed" is **"the failure rate is below about 20%"** —
not "it never fails". To get that bound under 5% takes 73 clean runs in a row.

When two builds are compared ("did this fix make the flake rarer?"), the runs are
interleaved — A, B, A, B — so that both see the same machine conditions, and the two counts
go through **Fisher's exact test** (`fisher_exact_two_sided`). It is exact rather than
approximate, which matters at the small counts these runs produce.

### A.5 How PSNR and LSB relate

PSNR and LSB are introduced separately in §2, but they are the same quantity seen two ways.
PSNR is computed from the **mean squared error** (MSE):

```python
# core/comparator.py, compute_psnr
mse = np.mean((a - b) ** 2)                 # a, b as float64
if mse == 0:
    return float('inf')
max_val = 255.0 if img_a.dtype == np.uint8 else 65535.0
return 10.0 * np.log10((max_val ** 2) / mse)
```

If **every** pixel is off by the same `e` LSB, then MSE = `e²` and PSNR = `20 · log10(255 / e)`:

| uniform error | PSNR (8-bit) |
| ---: | ---: |
| 0.5 LSB | 54.2 dB |
| **1 LSB** | **48.1 dB** |
| 2 LSB | 42.1 dB |
| 4 LSB | 36.1 dB |

The same arithmetic gives the "+6 dB is twice as accurate" rule from §2: halving `e` adds
`20 · log10(2) ≈ 6.02 dB`. On a 16-bit scale one step is 96.3 dB, because 65535 is 257 times
255.

**And this is why PSNR alone can mislead.** An MSE of 1 is also what you get when **one
pixel in a thousand is off by 32 LSB** and the rest are exact (`0.001 × 31.6² = 1`). Both
pictures score 48.1 dB, and only one of them has a visible fault. That is why result lines
also print `max_diff` (§7), and why the diagnostics in §6.3 split a frame into regions
instead of trusting one average.
