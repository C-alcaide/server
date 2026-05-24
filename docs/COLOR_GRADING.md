# Color Grading & Color Management

GPU-accelerated ACES color management and professional color grading tools for CasparCG Server. All processing runs on the GPU in a single GLSL fragment shader pass, with zero CPU overhead per frame.

For virtual production features (360° projection, curved screen compensation, playback speed, flip), see [VIRTUAL_PRODUCTION_FEATURES.md](VIRTUAL_PRODUCTION_FEATURES.md). For HDR channel configuration, DeckLink/Vulkan HDR output, and file recording metadata, see [HDR_GUIDE.md](HDR_GUIDE.md). For blur, sharpening, and film grain details, see [IMAGE_EFFECTS.md](IMAGE_EFFECTS.md).

## Table of Contents

1. [ACES Color Management](#aces-color-management) — Color space conversion, HDR tone mapping, camera log handling
2. [ASC CDL](#asc-cdl) — Industry-standard Slope/Offset/Power color correction
3. [3D LUT](#3d-lut) — Load `.cube` look-up tables for creative looks
4. [Linear Saturation](#linear-saturation) — Scene-linear saturation control
5. [Split Toning](#split-toning) — Independent shadow/highlight color tinting
6. [Gamut Compression](#gamut-compression) — ACES-style out-of-gamut recovery
7. [Hue Curves](#hue-curves) — Hue-vs-Hue, Hue-vs-Saturation, Hue-vs-Luminance, Sat-vs-Sat curves
8. [Secondary Qualifier](#secondary-qualifier) — HSL keyer with per-key corrections
9. [Sharpening](#sharpening) — Laplacian unsharp mask
10. [Film Grain](#film-grain) — Procedural photographic grain emulation
11. [Internal Pipeline](#internal-pipeline) — Full processing order, two color management paths
12. [Supported Standards](#supported-standards)
13. [Limitations & Best Practices](#limitations--best-practices)
14. [Common Workflows](#common-workflows)

---

## ACES Color Management

The color management pipeline converts between color spaces, applies HDR tone mapping, and handles camera log curves — all per-layer.

### AMCP Command

```bash
MIXER [channel]-[layer] COLORSPACE [input_transfer] [input_gamut] [tonemapping] [output_gamut] [output_transfer] [exposure]
MIXER [channel]-[layer] COLORSPACE NONE         # Disable
MIXER [channel]-[layer] COLORSPACE              # Query
```

### Parameters

| Parameter | Description | Options |
| :--- | :--- | :--- |
| **input_transfer** | EOTF of the source media | `LINEAR`, `SRGB`, `REC709`, `PQ`, `HLG`, `LOGC3`, `SLOG3` |
| **input_gamut** | Color primaries of the source | `BT709`, `BT2020`, `DCIP3`, `ACES_AP0`, `ACES_AP1`, `ACESCG`, `ARRI_WG3`, `SGAMUT3_CINE` |
| **tonemapping** | HDR compression algorithm | `NONE`, `REINHARD`, `ACES_FILMIC`, `ACES_RRT`, `ACES_RRT_709`, `ACES_RRT_P3`, `ACES_RRT_2020_PQ` |
| **output_gamut** | Target display primaries | Same as `input_gamut` |
| **output_transfer** | OETF for the display | `LINEAR`, `SRGB`, `REC709`, `PQ`, `HLG` |
| **exposure** | Linear exposure multiplier (default `1.0`) | Float (e.g. `2.0` = +1 stop) |

### Tone Mapping Operators

| Operator | Description |
| :--- | :--- |
| `NONE` | Hard clip at 1.0 — no compression |
| `REINHARD` | Simple global operator $x/(x+1)$. Preserves hue, desaturates highlights |
| `ACES_FILMIC` | Narkowicz approximation. High contrast "filmic" look, slight black crush |
| `ACES_RRT` | Stephen Hill's approximation of the ACES Reference Rendering Transform. Standard cinema look with desaturated highlights and smooth rolloff |
| `ACES_RRT_709` | Full ACES RRT + ODT for Rec.709/sRGB (100 nit) display. Uses segmented spline tonecurves from the official ACES specification. Overrides output gamut/transfer to BT.709/sRGB |
| `ACES_RRT_P3` | Full ACES RRT + ODT for DCI-P3 (D65, 48 nit) display. Overrides output gamut/transfer to P3 |
| `ACES_RRT_2020_PQ` | Full ACES RRT + ODT for Rec.2020 PQ (1000 nit) HDR display. Overrides output gamut/transfer to BT.2020/PQ |

> **Note:** The `ACES_RRT_709`, `ACES_RRT_P3`, and `ACES_RRT_2020_PQ` operators incorporate both RRT and ODT in a single pass. When using these, the `output_gamut` and `output_transfer` parameters are effectively overridden by the ODT's target space.

### Usage Examples

```bash
# ARRI LogC camera to HDTV
MIXER 1-10 COLORSPACE LOGC3 ARRI_WG3 ACES_RRT BT709 REC709

# HDR PQ to SDR
MIXER 1-10 COLORSPACE PQ BT2020 REINHARD BT709 SRGB

# Full ACES pipeline to Rec.709 display
MIXER 1-10 COLORSPACE LOGC3 ARRI_WG3 ACES_RRT_709 BT709 SRGB

# Full ACES pipeline to HDR display
MIXER 1-10 COLORSPACE LOGC3 ARRI_WG3 ACES_RRT_2020_PQ BT2020 PQ

# Exposure boost (+1 stop) before tone mapping
MIXER 1-10 COLORSPACE LOGC3 ARRI_WG3 ACES_RRT BT709 REC709 2.0

# Disable
MIXER 1-10 COLORSPACE NONE
```

---

## ASC CDL

Industry-standard ASC Color Decision List (Slope/Offset/Power) with per-channel control and global saturation. Operates in scene-linear space per the ASC CDL specification.

### AMCP Command

```bash
MIXER [channel]-[layer] CDL [sR] [sG] [sB] [oR] [oG] [oB] [pR] [pG] [pB] [saturation] [duration] [tween]
MIXER [channel]-[layer] CDL RESET               # Reset to identity
MIXER [channel]-[layer] CDL                      # Query
```

### Parameters

| Parameter | Description | Default |
| :--- | :--- | :--- |
| **sR sG sB** | Slope (gain) per channel | `1.0 1.0 1.0` |
| **oR oG oB** | Offset (lift) per channel | `0.0 0.0 0.0` |
| **pR pG pB** | Power (gamma) per channel | `1.0 1.0 1.0` |
| **saturation** | Global saturation (optional) | `1.0` |
| **duration** | Tween duration in frames (optional) | `0` |
| **tween** | Tween curve type (optional) | `linear` |

The formula applied per channel is: $\text{out} = \text{clamp}(\text{in} \times \text{slope} + \text{offset})^{\text{power}}$

### Usage Examples

```bash
# Warm the image: boost red gain, reduce blue
MIXER 1-10 CDL 1.2 1.0 0.85 0 0 0 1 1 1

# Print-film emulation with lifted blacks
MIXER 1-10 CDL 0.9 0.9 0.9 0.02 0.02 0.02 1.1 1.0 0.95 0.9

# Animated grade over 50 frames
MIXER 1-10 CDL 1.1 1.0 0.9 0 0 0 1 1 1 1.0 50 EASEINOUTQUAD

# Reset to neutral
MIXER 1-10 CDL RESET
```

---

## 3D LUT

Load industry-standard `.cube` 3D look-up tables for creative color transforms. Supports any cube size (commonly 17×17×17, 33×33×33, or 65×65×65). LUT data is uploaded as a `GL_TEXTURE_3D` with trilinear interpolation and cached until the LUT changes.

### AMCP Command

```bash
MIXER [channel]-[layer] LUT3D [path.cube] [strength]
MIXER [channel]-[layer] LUT3D NONE              # Remove LUT
MIXER [channel]-[layer] LUT3D                   # Query
```

### Parameters

| Parameter | Description | Default |
| :--- | :--- | :--- |
| **path** | Path to `.cube` file (absolute, or relative to media folder) | — |
| **strength** | Blend factor `0.0`–`1.0` (0 = bypass, 1 = full LUT) | `1.0` |

### Usage Examples

```bash
# Load a film emulation LUT
MIXER 1-10 LUT3D "luts/FilmLook.cube"

# Half-strength LUT for a subtler look
MIXER 1-10 LUT3D "luts/FilmLook.cube" 0.5

# Remove LUT
MIXER 1-10 LUT3D NONE
```

### File Format

Standard `.cube` format with `LUT_3D_SIZE` header and `R G B` triplets. The parser ignores `TITLE`, `DOMAIN_MIN`, `DOMAIN_MAX`, comments (`#`), and 1D LUT sections.

---

## Linear Saturation

Scene-linear saturation control using Rec.709 luminance weighting. Operates in the scene-linear working space before tone mapping, providing perceptually smooth results that avoid the clipping artifacts of display-referred saturation.

### AMCP Command

```bash
MIXER [channel]-[layer] LINEARSATURATION [value] [duration] [tween]
MIXER [channel]-[layer] LINEARSATURATION         # Query
```

### Parameters

| Parameter | Description | Default |
| :--- | :--- | :--- |
| **value** | Saturation multiplier (`0.0` = mono, `1.0` = unchanged, `>1.0` = boost) | `1.0` |
| **duration** | Tween duration in frames | `0` |
| **tween** | Tween curve type | `linear` |

### Usage Examples

```bash
# Desaturate to 50%
MIXER 1-10 LINEARSATURATION 0.5

# Boost saturation 20%
MIXER 1-10 LINEARSATURATION 1.2

# Animated desaturation over 2 seconds
MIXER 1-10 LINEARSATURATION 0.0 50 EASEINOUTQUAD
```

---

## Split Toning

Applies independent color tints to shadows and highlights. The balance parameter controls where the shadow/highlight crossover point sits in the luminance range.

### AMCP Command

```bash
MIXER [channel]-[layer] SPLITTONE [shR] [shG] [shB] [hiR] [hiG] [hiB] [balance] [duration] [tween]
MIXER [channel]-[layer] SPLITTONE RESET          # Reset to neutral
MIXER [channel]-[layer] SPLITTONE                # Query
```

### Parameters

| Parameter | Description | Default |
| :--- | :--- | :--- |
| **shR shG shB** | Shadow tint color (RGB, `0.0`–`1.0`) | `0 0 0` |
| **hiR hiG hiB** | Highlight tint color (RGB, `0.0`–`1.0`) | `0 0 0` |
| **balance** | Shadow/highlight crossover point (`0.0`–`1.0`) | `0.5` |
| **duration** | Tween duration in frames | `0` |
| **tween** | Tween curve type | `linear` |

### Usage Examples

```bash
# Cool shadows + warm highlights (classic "teal and orange")
MIXER 1-10 SPLITTONE 0.0 0.1 0.2 0.2 0.1 0.0

# Blue shadows only, balance shifted toward highlights
MIXER 1-10 SPLITTONE 0.0 0.0 0.15 0.0 0.0 0.0 0.3

# Reset
MIXER 1-10 SPLITTONE RESET
```

---

## Gamut Compression

ACES-style gamut compression that maps out-of-gamut colors (negative channel values that arise from wide-to-narrow gamut conversions) back toward the achromatic axis. Prevents neon fringing on saturated colors when converting from wide gamuts like ARRI Wide Gamut or S-Gamut3 to BT.709.

### AMCP Command

```bash
MIXER [channel]-[layer] GAMUTCOMPRESS [enable] [cyan_limit] [magenta_limit] [yellow_limit]
MIXER [channel]-[layer] GAMUTCOMPRESS            # Query
```

### Parameters

| Parameter | Description | Default |
| :--- | :--- | :--- |
| **enable** | `1` = enable, `0` = disable | — |
| **cyan_limit** | Compression limit for cyan axis | `1.147` |
| **magenta_limit** | Compression limit for magenta axis | `1.264` |
| **yellow_limit** | Compression limit for yellow axis | `1.312` |

The default limits match the ACES 1.3 Gamut Compression reference values.

### Usage Examples

```bash
# Enable with default limits
MIXER 1-10 GAMUTCOMPRESS 1

# Custom limits for aggressive compression
MIXER 1-10 GAMUTCOMPRESS 1 1.1 1.2 1.3

# Disable
MIXER 1-10 GAMUTCOMPRESS 0
```

---

## Hue Curves

Four independent curve types for targeted hue, saturation, and luminance adjustments based on input hue or saturation. Each curve is a 256-entry LUT built from control points with linear interpolation. Multiple curve types can be active simultaneously — they are merged into a single texture.

### AMCP Command

```bash
MIXER [channel]-[layer] HUECURVE [type] [h1] [v1] [h2] [v2] ...
MIXER [channel]-[layer] HUECURVE RESET           # Clear all curves
MIXER [channel]-[layer] HUECURVE                 # Query
```

### Curve Types

| Type | Input Axis | Output Axis | Neutral Value |
| :--- | :--- | :--- | :--- |
| `HUE_HUE` | Hue position (0–1) | Hue offset (degrees, wrapped) | `0.0` |
| `HUE_SAT` | Hue position (0–1) | Saturation multiplier | `1.0` |
| `HUE_LUM` | Hue position (0–1) | Luminance offset | `0.0` |
| `SAT_SAT` | Saturation (0–1) | Saturation multiplier | `1.0` |

### Parameters

Control points are provided as `[hue_position] [value]` pairs. At least 2 control points are required. Hue positions are normalized `0.0`–`1.0` (where 0 = 0°, 0.5 = 180°, 1.0 = 360°).

### Usage Examples

```bash
# Desaturate greens (hue ≈ 0.33) while boosting blues (hue ≈ 0.66)
MIXER 1-10 HUECURVE HUE_SAT 0.0 1.0 0.33 0.3 0.5 1.0 0.66 1.5 1.0 1.0

# Shift red hues toward orange
MIXER 1-10 HUECURVE HUE_HUE 0.0 10.0 0.1 5.0 0.2 0.0 1.0 0.0

# Reduce saturation of already-desaturated pixels
MIXER 1-10 HUECURVE SAT_SAT 0.0 0.5 0.3 1.0 1.0 1.0

# Clear all curves
MIXER 1-10 HUECURVE RESET
```

> **Note:** Setting a new curve of a given type merges it with existing curves of other types. To replace a specific curve type, simply send it again — only that channel is overwritten.

---

## Secondary Qualifier

HSL-based secondary color qualifier that isolates a specific color range and applies targeted corrections (exposure, saturation, hue shift) only to the qualified pixels. Unqualified pixels are left untouched. The key mask uses soft edges for smooth transitions.

### AMCP Command

```bash
MIXER [channel]-[layer] QUALIFIER [target_hue] [hue_width] [min_sat] [max_sat] [min_lum] [max_lum] [softness] [exposure] [saturation] [hue_offset] [duration] [tween]
MIXER [channel]-[layer] QUALIFIER 0              # Disable
MIXER [channel]-[layer] QUALIFIER                # Query
```

### Parameters

| Parameter | Description | Range |
| :--- | :--- | :--- |
| **target_hue** | Centre hue to isolate (degrees) | `0.0`–`360.0` |
| **hue_width** | Width of the hue selection window (degrees) | `0.0`–`180.0` |
| **min_sat** | Minimum saturation threshold | `0.0`–`1.0` |
| **max_sat** | Maximum saturation threshold | `0.0`–`1.0` |
| **min_lum** | Minimum luminance threshold | `0.0`–`1.0` |
| **max_lum** | Maximum luminance threshold | `0.0`–`1.0` |
| **softness** | Soft edge width for key transitions | `0.0`–`1.0` |
| **exposure** | Exposure offset for qualified pixels | Float (e.g. `0.5` = +½ stop) |
| **saturation** | Saturation offset for qualified pixels | Float (e.g. `-0.3` = desaturate) |
| **hue_offset** | Hue rotation for qualified pixels (degrees) | Float |
| **duration** | Tween duration in frames (optional) | Integer |
| **tween** | Tween curve type (optional) | `linear` |

### Usage Examples

```bash
# Isolate blue sky (hue ≈ 210°) and boost saturation
MIXER 1-10 QUALIFIER 210 30 0.2 1.0 0.3 1.0 0.1 0.0 0.3 0.0

# Isolate skin tones (hue ≈ 30°) and reduce saturation
MIXER 1-10 QUALIFIER 30 20 0.15 0.8 0.2 0.9 0.15 0.0 -0.2 0.0

# Shift green foliage toward teal
MIXER 1-10 QUALIFIER 120 40 0.1 1.0 0.1 0.8 0.1 0.0 0.0 -30.0

# Disable qualifier
MIXER 1-10 QUALIFIER 0
```

---

## Sharpening

3×3 Laplacian-based unsharp mask applied directly after texture sampling, before any color grading. Works on all layer types including 360° and curved screen projections.

### AMCP Command

```bash
MIXER [channel]-[layer] SHARPEN [amount] [radius] [duration] [tween]
MIXER [channel]-[layer] SHARPEN                  # Query
```

### Parameters

| Parameter | Description | Default |
| :--- | :--- | :--- |
| **amount** | Sharpening strength (`0.0` = off, `1.0` = standard, `>1.0` = aggressive) | `0.0` |
| **radius** | Kernel radius multiplier (controls the sampling spread in pixels) | `1.0` |
| **duration** | Tween duration in frames | `0` |
| **tween** | Tween curve type | `linear` |

### Usage Examples

```bash
# Standard sharpening
MIXER 1-10 SHARPEN 0.5

# Aggressive sharpening with wider radius
MIXER 1-10 SHARPEN 1.5 2.0

# Disable sharpening
MIXER 1-10 SHARPEN 0

# Animated sharpen reveal
MIXER 1-10 SHARPEN 1.0 1.0 25 EASEINOUTQUAD
```

---

## Film Grain

Procedural photographic grain emulation applied in display-referred space (after the OETF encoding). Uses a hash-based noise function with photographic response — grain is more visible in midtones and less visible in deep shadows and bright highlights, matching the behavior of real film stock.

### AMCP Command

```bash
MIXER [channel]-[layer] GRAIN [intensity] [size] [duration] [tween]
MIXER [channel]-[layer] GRAIN                    # Query
```

### Parameters

| Parameter | Description | Default |
| :--- | :--- | :--- |
| **intensity** | Grain visibility (`0.0` = off, `0.05` = subtle, `0.15` = heavy) | `0.0` |
| **size** | Grain particle size multiplier (`1.0` = pixel-level, `2.0` = coarser) | `1.0` |
| **duration** | Tween duration in frames | `0` |
| **tween** | Tween curve type | `linear` |

### Usage Examples

```bash
# Subtle film grain
MIXER 1-10 GRAIN 0.04

# Heavy grain with larger particles (16mm look)
MIXER 1-10 GRAIN 0.12 2.0

# Disable grain
MIXER 1-10 GRAIN 0

# Fade grain in over 50 frames
MIXER 1-10 GRAIN 0.08 1.0 50 LINEAR
```

---

## Internal Pipeline

All color grading runs on the GPU in a single fragment shader pass. The processing order from texture fetch to fragment output is:

| Step | Operation | Controlled By |
| :--- | :--- | :--- |
| 1 | **Texture Fetch** | UV coordinates (projection, curve warp, flip) |
| 2 | **Sharpening** | `MIXER SHARPEN` |
| 3 | **Premultiply Alpha** | Automatic (if straight alpha source) |
| 4 | **EOTF** (decode to linear) | `MIXER COLORSPACE` or auto-color-convert |
| 5 | **Input Gamut → Working Space** | `MIXER COLORSPACE` or auto-color-convert (see below) |
| 6 | **Gamut Compression** | `MIXER GAMUTCOMPRESS` |
| 7 | **Exposure** | `MIXER COLORSPACE` exposure / auto luminance scaling |
| 8 | **ASC CDL** | `MIXER CDL` |
| 9 | **3D LUT** | `MIXER LUT3D` |
| 10 | **Linear Saturation** | `MIXER LINEARSATURATION` |
| 11 | **White Balance** | `MIXER WHITEBALANCE` |
| 12 | **Lift / Midtone / Gain** | `MIXER LIFT`, `MIXER MIDTONE`, `MIXER GAIN` |
| 13 | **Split Toning** | `MIXER SPLITTONE` |
| 14 | **Secondary Qualifier** | `MIXER QUALIFIER` |
| 15 | **Hue Shift** | `MIXER HUESHIFT` |
| 16 | **Hue Curves** | `MIXER HUECURVE` |
| 17 | **Tonal Balance** | `MIXER TONEBALANCE` |
| 18 | **RGB Levels** | `MIXER RGBLEVELS` |
| 19 | **Tone Curves** | `MIXER CURVES` |
| 20 | **Legacy Levels / CSB** | `MIXER LEVELS`, `MIXER BRIGHTNESS`, `MIXER SATURATION`, `MIXER CONTRAST` |
| 21 | **Invert** | `MIXER INVERT` |
| 22 | **Shape Overlay** | `MIXER SHAPE` |
| 23 | **Opacity** | `MIXER OPACITY` |
| 24 | **Keying** | `MIXER KEYER` |
| 25 | **Blend Mode** | `MIXER BLEND` |
| 26 | **Chroma Key** | `MIXER CHROMA` |
| 27 | **Tone Mapping** | `MIXER COLORSPACE` tonemapping / auto (ACES RRT for HDR→SDR) |
| 28 | **Working Space → Output Gamut** | `MIXER COLORSPACE` or auto-color-convert (see below) |
| 29 | **OETF** (encode for display) | `MIXER COLORSPACE` or auto-color-convert |
| 30 | **Film Grain** | `MIXER GRAIN` |

> **Design note:** Grading operations (steps 8–21) run in scene-linear space, after the EOTF decode and gamut conversion but before tone mapping and output encoding. This ensures perceptually correct, display-independent results. Sharpening runs on raw texture samples (step 2) to avoid sharpening color grading artifacts. Film grain is applied last (step 30) in display-referred space so it has the correct photographic response.

### Two Color Management Paths

The EOTF/gamut/OETF wrapper (steps 4–7 and 27–29) can be activated in two ways. They are **mutually exclusive** — if `MIXER COLORSPACE` is active, it takes priority:

| | **MIXER COLORSPACE** (manual) | **auto-color-convert** (automatic) |
| :--- | :--- | :--- |
| **Activated by** | `MIXER COLORSPACE` command (sets `color_grade.enable`) | `<auto-color-convert>true</auto-color-convert>` in channel config (default) |
| **Working space** | ACEScg (ACES AP1, D60 white point) | Target channel gamut (BT.709 or BT.2020, D65) |
| **Gamut matrices** | Input → ACEScg → Output (via chromatic adaptation) | Direct standard matrices (ITU-R BT.2087, no intermediate) |
| **Gamut accuracy** | Optimized for perceptual grading quality | Norm-correct to ITU standard (< 1 LSB deviation) |
| **Tone mapping** | User-selected operator | Automatic ACES RRT for HDR→SDR; none otherwise |
| **When to use** | Camera log workflows, creative grading, explicit control | Mixed SDR/HDR playout without manual setup |

**Key points:**

- **Grading tools work with both paths.** All color grading tools (CDL, LMG, white balance, hue shift, curves, levels, saturation, qualifier, etc.) have independent flags and operate between the EOTF and OETF regardless of which path activated the color conversion. When `auto-color-convert` provides the linearization, grading tools operate in the target channel's linear space.
- **MIXER COLORSPACE overrides auto.** When you send a `MIXER COLORSPACE` command on a layer, that layer switches from the auto path to the manual ACEScg path. The auto path is skipped entirely for that layer.
- **Auto handles luminance scaling.** The auto path automatically adjusts exposure for cross-transfer conversions: SDR→HLG (×0.1, mapping 100-nit reference white to HLG scene level), HLG→PQ (×10.0), and PQ→HLG (÷10.0 in shader).

---

## Supported Standards

### Transfer Functions (EOTF / OETF)

| Name | Standard | Notes |
| :--- | :--- | :--- |
| `LINEAR` | — | No curve applied |
| `SRGB` | IEC 61966-2-1 | Standard web/monitor gamma |
| `REC709` | ITU-R BT.709 | HDTV standard gamma |
| `PQ` | SMPTE ST.2084 | HDR10 / Dolby Vision |
| `HLG` | ARIB STD-B67 | Broadcast HDR |
| `LOGC3` | ARRI Alexa LogC3 | Input only |
| `SLOG3` | Sony S-Log3 | Input only |

### Color Gamuts

| Name | Standard | Notes |
| :--- | :--- | :--- |
| `BT709` | ITU-R BT.709 | Standard HDTV / sRGB primaries |
| `BT2020` | ITU-R BT.2020 | UHDTV / HDR |
| `DCIP3` | DCI-P3 | Digital Cinema (D65 white point) |
| `ACES_AP0` | SMPTE ST.2065-1 | ACES archival (encompasses all visible colors) |
| `ACES_AP1` / `ACESCG` | Academy S-2014-004 | ACEScg working space |
| `ARRI_WG3` | ARRI Wide Gamut 3 | ARRI camera native |
| `SGAMUT3_CINE` | Sony S-Gamut3.Cine | Sony camera native |

---

## Limitations & Best Practices

### 16-bit Integer Precision

CasparCG uses normalized integer textures (0–65535 mapped to 0.0–1.0) internally. It does **not** use floating-point textures (EXR/half-float).

- **Impact**: Inputs strictly clip at 1.0 (paper white). Super-white and negative values in the source are lost.
- **Workaround**: Do **not** use Linear EXR or scRGB sources where data exceeds 1.0.
- **Recommended**: Use **Log-encoded** (LogC3, S-Log3) or **PQ/HLG** sources. These formats compress highlight data to fit within the 0.0–1.0 signal range, allowing the tone mapper to work effectively.

### Alpha Channel

All color grading operations affect RGB channels only. The alpha channel is passed through untouched.

### Grading Order

The pipeline order is fixed. To achieve a specific look, consider which operations interact:

- **CDL** is early in the chain — it affects everything downstream including LUT and split tone.
- **3D LUT** is applied after CDL but before saturation and white balance — use it for creative looks, not technical transforms (use `COLORSPACE` for those).
- **Qualifier** corrections are applied in-place without disrupting the rest of the grading chain.
- **Film Grain** is the very last operation — it is never affected by color grading.

---

## Common Workflows

### 1. ARRI LogC Camera to HDTV

Standard "Alexa to Rec.709" workflow:
```bash
MIXER 1-10 COLORSPACE LOGC3 ARRI_WG3 ACES_RRT BT709 REC709
```

### 2. Full ACES Pipeline to Rec.709 Display

Using the reference ACES RRT+ODT instead of the Hill approximation:
```bash
MIXER 1-10 COLORSPACE LOGC3 ARRI_WG3 ACES_RRT_709 BT709 SRGB
```

### 3. Full ACES Pipeline to HDR Display

For LED walls or HDR monitors running PQ:
```bash
MIXER 1-10 COLORSPACE LOGC3 ARRI_WG3 ACES_RRT_2020_PQ BT2020 PQ
```

### 4. HDR (PQ) to SDR Down-mapping

Convert HDR content for standard monitors:
```bash
MIXER 1-10 COLORSPACE PQ BT2020 REINHARD BT709 SRGB
```

### 5. Log Footage Exposure Fix

Boost underexposed Log footage by 1 stop ($2.0\times$) before tone mapping:
```bash
MIXER 1-10 COLORSPACE LOGC3 ARRI_WG3 ACES_RRT BT709 REC709 2.0
```

### 6. Color Pass-through

Non-destructive test to ensure the pipeline is active but neutral:
```bash
MIXER 1-10 COLORSPACE SRGB BT709 NONE BT709 SRGB
```

### 7. Creative Grade Stack

Combine multiple grading tools for a complete look:
```bash
# Set up color pipeline
MIXER 1-10 COLORSPACE LOGC3 ARRI_WG3 ACES_RRT BT709 REC709

# Apply a film LUT at 60% strength
MIXER 1-10 LUT3D "luts/Kodak2383.cube" 0.6

# Warm CDL grade
MIXER 1-10 CDL 1.1 1.0 0.9 0.01 0.0 -0.01 1.0 1.0 1.05

# Teal shadows / warm highlights
MIXER 1-10 SPLITTONE 0.0 0.08 0.12 0.12 0.06 0.0

# Add subtle grain
MIXER 1-10 GRAIN 0.04
```

### 8. Sky Enhancement with Qualifier

Isolate and boost the blue sky without affecting other colors:
```bash
MIXER 1-10 QUALIFIER 210 30 0.2 1.0 0.3 1.0 0.1 0.2 0.3 0.0
```
