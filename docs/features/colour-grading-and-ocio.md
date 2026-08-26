# Colour grading and OCIO

> **State:** shipped
> **Modules:** `src/accelerator/{ogl,vulkan}/image` (the shaders and kernels), `src/accelerator/ocio`
> **Commands:** ~26 fork-specific AMCP commands — the largest family in the fork
> **Coverage:** 14 batteries, all gating at **1 LSB**

A full grading chain in the mixer — CDL, lift/midtone/gain, curves, hue curves, a qualifier,
split tone, white balance, exposure, levels, tone balance, saturation, sharpen, blur, grain — plus
ACES colour management, gamut compression, 3D LUTs and OCIO. Per layer, on both mixer backends.

This is the best-documented and best-covered part of the fork. Detail:
[`../guides/COLOR_GRADING.md`](../guides/COLOR_GRADING.md),
[`../guides/HDR_GUIDE.md`](../guides/HDR_GUIDE.md),
[`../guides/OCIO_USER_GUIDE.md`](../guides/OCIO_USER_GUIDE.md), and

> **Where the 24 grading operators are documented, stated so an audit does not re-flag it.**
> `MIXER BLUR`, `CDL`, `CDL_FILE`, `CURVES`, `EXPOSURE`, `GAIN`, `GAMUTCOMPRESS`, `GRADE`, `GRAIN`,
> `HUECURVE`, `HUESHIFT`, `LIFT`, `LINEARSATURATION`, `LUT3D`, `MESH`, `MIDTONE`, `QUALIFIER`,
> `RGBLEVELS`, `SHAPE`, `SHARPEN`, `SPLITTONE`, `TONEBALANCE`, `WHITEBALANCE`, `OCIO_DISPLAY` and
> `OCIO_LOOK` are documented **in the guides, not here** — `COLOR_GRADING.md` for the chain,
> `IMAGE_EFFECTS.md` for blur and sharpen, `MIXER_SHAPE.md` for the mask, `OCIO_USER_GUIDE.md` for
> the OCIO pair. This document covers state, decisions and measurement; the per-operator syntax
> lives once, in the guide that owns it. A second copy is a claim that can go stale independently.
[`../architecture/OCIO_INTEGRATION_STUDY.md`](../architecture/OCIO_INTEGRATION_STUDY.md) — which
the code cites by section number, so it is a live reference rather than history.

This document exists for the two things those do not say in one place: which battery covers what,
and where the numbers come from.

---

## 1. Where the numbers come from

**Almost nothing here is invented, and that is the rule rather than an aspiration.** Colour
management, transfer functions, gamut matrices, tone curves, YCbCr matrices and LUT formats are
published standards; a constant in this tree is either the standard's value or a bug. The order of
authority, from `CLAUDE.md`:

1. the body that defines it — ACES/AMPAS, ASWF (OpenColorIO), SMPTE, ITU-R, EBU, DCI
2. the vendor, where the vendor defines it — ARRI, Sony, Blackmagic for camera log curves and
   native gamuts
3. other open-source implementations, as **corroboration only** — two projects agreeing may mean
   one copied the other

**Why that is stated first rather than assumed.** Four of the seven ACEScg gamut matrices in this
tree were wrong — `bt2020`, `p3-d65`, `arri_wg3`, `sgamut3cine`, by up to 0.41 per element. They
were plausible numbers nobody had derived from primaries. What settled it was OCIO plus a
colorimetric derivation, and the fix reached upstream only because the standard was consulted
rather than the existing code trusted.

**An internal check cannot validate against a standard.** `arri_wg3` and `sgamut3cine` round-tripped
`to_output @ to_working` to *exactly* the identity while both were wrong — consistent inverses of
each other. Round trips prove self-consistency and nothing else.

**An approximation must be named as one.** The gamut compressor carries ACES's limits with a
different curve, measured at mean 0.030 / max 0.350 from the reference. It was called "ACES 1.3
Reference Gamut Compress" in three places, which claims conformance the code does not have.

---

## 2. Verification — the coverage map

| what | battery | gate |
| :--- | :--- | :--- |
| Colour-space conversions | `conformance` | 1 LSB |
| Grading operators, single and neutral | `grading` | 1 LSB |
| Operators at extremes | `grade-extremes` | 1 LSB |
| Windowed / masked grading | `grade-window` | 1 LSB |
| Gamut compression | `gamut-compress`, `flat-gamut-compress`, `gamut-sweep` | 1 LSB |
| Working-space tone mapping | `ws-tonemap` | 1 LSB |
| Banding / gradient integrity | `banding` | — |
| Bokeh / blur luma behaviour | `bokeh-luma` | — |
| OCIO transforms | `ocio`, `ocio-display`, `ocio-look`, `ocio-lut3d`, `ocio-exposure`, `ocio-gamut-compress` | 1 LSB |
| CDL from file | `cdl-file` | 1 LSB |

Latest, both backends, 2026-08-26: `conformance` 23/23 patches per conversion, worst **0.55 LSB**;
`grading` 8/8 patches, worst **0.55** against a 1.0 gate, neutrals **0.00**.

**What every one of those batteries cannot see.** They drive a **colour producer**, which is BGRA —
so none of them calls the YCbCr decode. That is how `ycbcra_to_rgba` counted in 8-bit codes for
high-bit-depth sources undetected: exact for an 8-bit texture, and leaving a **−0.4981** chroma
offset for 10-bit video in a 16-bit texture — a green cast on greys, on every high-bit-depth YCbCr
source. Both mixers were wrong identically, so every parity check passed too. It took a
**flat decoded** fixture, neutral by construction, to see it.

The lesson generalises and is the reason this section exists: **a 1 LSB gate on flat colour is a
strong claim about a narrow path.**

---

## 3. Known gaps

1. **`MIXER BRIGHTNESS` / `SATURATION` / `CONTRAST` are not covered by `grading`** — their luma
   weights come from the source's YCbCr matrix rather than from the command, which puts them
   outside that battery's model.
2. **Tweened forms of every grading command are untested**, fork-wide.
3. **Sharpening and blur are covered only at the flat-field identity**; film grain is checked
   distributionally by `grain`, not per pixel.
4. **`p3_d65`, `p3_dci` and `adobe_rgb` are not reachable through `MIXER COLORSPACE`** and so are
   not covered by `conformance` — they need the auto path.
5. **The gamut compressor is an approximation** at mean 0.030 / max 0.350 from the ACES reference;
   named as one now, in the three places that previously claimed conformance.

---

## 4. Related commits

Extensively traced in the guides. The two worth naming here because they are about *method* rather
than a feature:

| commit | why it matters |
| :--- | :--- |
| the gamut-matrix fix | four of seven matrices wrong, found by deriving from primaries rather than trusting a round trip |
| the YCbCr code-scale fix | a defect invisible to every 1 LSB battery, found by inventing a fixture the harness had recorded as impossible |

---

## 5. Diagrams

`../guides/COLOR_GRADING.md` already carries the chain-order flowchart — which is what makes "CDL
runs before the LUT" answerable at a glance instead of by reading fourteen table rows — and
`HDR_GUIDE.md` carries the transfer-function figures. Nothing owed here; a second copy would be a
second thing to keep true.
