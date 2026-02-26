# New Visual Features: ACES Color Management & 360° Projection

This feature set introduces a GPU-accelerated ACES (Academy Color Encoding System) color management pipeline and 360° Equirectangular Projection support to CasparCG Server.

## 360° Equirectangular Projection

This feature maps standard 2D layers onto a virtual 360° sphere, allowing for immersive panning and tilting within an equirectangular video stream. It is ideal for virtual production, dome projection, or 360° video workflows.

### AMCP Command

```bash
MIXER [channel]-[layer] PROJECTION [yaw] [pitch] [roll] [fov] [duration] [tween]
MIXER [channel]-[layer] PROJECTION 0 0 0 0         // Disable (FOV 0 = Off)
```

### Parameters

| Parameter | Description | Range / Unit |
| :--- | :--- | :--- |
| **yaw** | Horizontal rotation (Panning) | -180.0 to 180.0 degrees |
| **pitch** | Vertical rotation (Tilting) | -90.0 to 90.0 degrees |
| **roll** | Z-axis rotation (Dutch angle) | -180.0 to 180.0 degrees |
| **fov** | Field of View (Zoom) | 1.0 to 179.0 degrees (0 = Disabled) |
| **duration**| Tween duration in frames | Integer |
| **tween** | Tween curve type | `linear`, `ease`, `ease-in`, `ease-out`, etc. |

### Usage Examples

**1. Standard 360° View**
Set a 90-degree FOV facing forward.
```bash
MIXER 1-10 PROJECTION 0 0 0 90
```

**2. Look Up and Right**
Pan 45° right, look 30° up.
```bash
MIXER 1-10 PROJECTION 45 30 0 90
```

**3. Animated Camera Move**
Smoothly pan 180 degrees over 2 seconds (50 frames at 25fps).
```bash
MIXER 1-10 PROJECTION 180 0 0 90 50 EASEINOUTQUAD
```

---

## ACES Color Management

The color management pipeline allows for high-quality color space conversions, HDR tone mapping, and camera log handling directly within the mixer.

## AMCP Command

The new command controls the color pipeline for a specific layer.

```bash
MIXER [channel]-[layer] COLORSPACE [input_transfer] [input_gamut] [tonemapping] [output_gamut] [output_transfer] [exposure]
```

### Parameters

| Parameter | Description | Options |
| :--- | :--- | :--- |
| **input_transfer** | The Transfer Characteristic (EOTF) of the source media. | `LINEAR`, `SRGB`, `REC709`, `PQ`, `HLG`, `LOGC3`, `SLOG3` |
| **input_gamut** | The Color Primaries of the source media. | `BT709`, `BT2020`, `DCIP3`, `ACES_AP0`, `ACES_AP1`, `ACESCG`, `ARRI_WG3`, `SGAMUT3_CINE` |
| **tonemapping** | The algorithm used to compress high-dynamic-range content into the output display range. | `NONE`, `REINHARD`, `ACES_FILMIC`, `ACES_RRT` |
| **output_gamut** | The target Color Primaries for the display. | Same list as `input_gamut`. |
| **output_transfer** | The target Transfer Characteristic (OETF) for the display. | `LINEAR`, `SRGB`, `REC709`, `PQ`, `HLG` |
| **exposure** | (Optional) Linear exposure multiplier applied before tone mapping. Default is `1.0`. | Float value (e.g., `1.5` boosts exposure by 50%). |

### Control Commands

| Action | Command |
| :--- | :--- |
| **Query Status** | `MIXER 1-10 COLORSPACE` |
| **Disable** | `MIXER 1-10 COLORSPACE NONE` |

---

## Supported Standards

### Transfer Functions (EOTF / OETF)
*   **SRGB**: Standard web/monitor gamma (IEC 61966-2-1).
*   **REC709**: HDTV standard gamma (ITU-R BT.709).
*   **PQ**: Perceptual Quantizer (ST.2084), used for HDR10/Dolby Vision.
*   **HLG**: Hybrid Log-Gamma (ARIB STD-B67), used for broadcast HDR.
*   **LOGC3**: ARRI Alexa LogC3 curve (Input only).
*   **SLOG3**: Sony S-Log3 curve (Input only).
*   **LINEAR**: No curve applied.

### Color Gamuts
*   **BT709**: Standard HDTV / sRGB primaries.
*   **BT2020**: UHDTV / HDR standards.
*   **DCIP3**: Digital Cinema P3 (D65 white point).
*   **ACES_AP0**: ACES 2065-1 (Archival).
*   **ACES_AP1 / ACESCG**: ACEScg (VFX/Compositing working space).
*   **ARRI_WG3**: ARRI Alexa Wide Gamut 3.
*   **SGAMUT3_CINE**: Sony S-Gamut3.Cine.

### Tone Mapping Operators
*   **NONE**: Hard clipping at 1.0 (creates harsh white burns).
*   **REINHARD**: Simple global operator ($x / (x+1)$). Preserves hue but desaturates highlights.
*   **ACES_FILMIC**: Narkowicz approximation. High contrast "filmic" look, crushes blacks slightly.
*   **ACES_RRT**: Stephen Hill's approximation of the official ACES RRT (Reference Rendering Transform). Standard "Cinema" look with desaturated highlights and smooth rolloff.

---

## Internal Pipeline

The processing happens entirely on the GPU (GLSL Fragment Shader) in the following order:

1.  **Input Transform (IDT)**: Source pixels are decoded from encoded `input_transfer` to Linear Light.
2.  **Gamut Conversion**: Linear RGB is converted from `input_gamut` to the **ACEScg** working space (ACES AP1 primaries).
3.  **Exposure**: The linear ACEScg values are multiplied by the `exposure` parameter.
4.  **Tone Mapping**: High dynamic range values are compressed into the 0.0-1.0 display range using the selected operator.
5.  **Output Transform (ODT)**: Values are converted from ACEScg to `output_gamut`.
6.  **Encoding**: Linear values are encoded using the `output_transfer` function (OETF) for display.

---

## Limitations & Best Practices

### 16-bit Integer Precision
CasparCG uses normalized integer textures (0-65535 mapped to 0.0-1.0) internally. It does **not** use floating-point textures (EXR/half-float).

*   **Impact**: Inputs strictly clip at 1.0 (paper white).
*   **Workaround**: Do **not** use Linear EXR or scRGB sources where data exceeds 1.0.
*   **Recommended**: Use **Log-encoded** (LogC3, S-Log3) or **PQ/HLG** sources. These formats "compress" regular highlight data to fit within the 0.0-1.0 signal range, allowing the tone mapper to work effectively.

### Alpha Channel
Changes affect RGB channels only. The Alpha channel is passed through untouched.

---

## Common Workflows

### 1. ARRI LogC Camera to HDTV
Standard "Alexas to Rec.709" workflow.
```bash
MIXER 1-10 COLORSPACE LOGC3 ARRI_WG3 ACES_RRT BT709 REC709
```

### 2. HDR (PQ) to SDR
Down-mapping HDR content for standard monitors.
```bash
MIXER 1-10 COLORSPACE PQ BT2020 REINHARD BT709 SRGB
```

### 3. Log Footage Exposure Fix
Boosting underexposed Log footage by 1 stop ($2.0\times$) before tone mapping.
```bash
MIXER 1-10 COLORSPACE LOGC3 ARRI_WG3 ACES_RRT BT709 REC709 2.0
```

### 4. Color Pass-through
Non-destructive test to ensure pipeline is active but neutral.
```bash
MIXER 1-10 COLORSPACE SRGB BT709 NONE BT709 SRGB
```
