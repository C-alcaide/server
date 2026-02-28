# Virtual Production Features

This feature set introduces GPU-accelerated ACES (Academy Color Encoding System) color management, 360° Equirectangular Projection, and real-time playback speed control to CasparCG Server, designed for virtual production, live events, and advanced broadcast workflows.

## Table of Contents

1. [360° Equirectangular Projection](#-equirectangular-projection)
2. [360° Projection Offset](#-projection-offset)
3. [Playback Speed Control](#playback-speed-control)
4. [Ping-Pong Loop](#ping-pong-loop)
5. [Flip (Mirror)](#flip-mirror)
6. [ACES Color Management](#aces-color-management)

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

## 360° Projection Offset

Slides the viewport across the sphere along the X and Y axes **without changing the viewing orientation** (yaw/pitch/roll are unaffected). This is the correct way to reposition a 360° layer on screen independent of the projection command — the sphere wraps continuously so there is never any missing content at the edges.

### AMCP Command

```bash
MIXER [channel]-[layer] PROJECTION_OFFSET [x] [y] [duration] [tween]
MIXER [channel]-[layer] PROJECTION_OFFSET         // Query current offset
```

### Parameters

| Parameter | Description | Range / Unit |
| :--- | :--- | :--- |
| **x** | Horizontal lens-shift | NDC units — `+1.0` = one half-screen width to the right |
| **y** | Vertical lens-shift | NDC units — `+1.0` = one half-screen height upward |
| **duration** | Tween duration in frames | Integer |
| **tween** | Tween curve type | `linear`, `ease`, `ease-in`, `ease-out`, etc. |

> **Note:** NDC (Normalized Device Coordinate) units are relative to the half-screen size. An offset of `1.0` moves the view centre to the edge of the frame. Values beyond `±1.0` are valid and wrap around the sphere.

### Usage Examples

**1. Pan the view right by a quarter screen**
```bash
MIXER 1-10 PROJECTION_OFFSET 0.5 0.0
```

**2. Reset offset to centre**
```bash
MIXER 1-10 PROJECTION_OFFSET 0 0
```

**3. Animated horizontal pan across sphere**
Smoothly drift the viewport right over 100 frames.
```bash
MIXER 1-10 PROJECTION_OFFSET 2.0 0.0 100 LINEAR
```

**4. Combined with PROJECTION**
Set a 90° FOV facing forward, then offset the viewport up slightly.
```bash
MIXER 1-10 PROJECTION 0 0 0 90
MIXER 1-10 PROJECTION_OFFSET 0.0 0.3
```

---

## Playback Speed Control

Real-time playback speed control on any FFmpeg-loaded clip. Speed can be changed at any time without reloading the source, and the OSC position/duration state remains accurate throughout.

### AMCP Command

Speed can be set at load/play time as a parameter, or changed at runtime via `CALL`:

```bash
PLAY [channel]-[layer] [clip] SPEED [value]    # load and play at given speed
LOAD [channel]-[layer] [clip] SPEED [value]    # pre-load at given speed
CALL [channel]-[layer] SPEED [value]           # change speed on the running clip
CALL [channel]-[layer] SPEED                   # query current speed
```

`SPEED` can be combined with other `PLAY`/`LOAD` flags:

```bash
PLAY 1-10 MyClip LOOP SPEED 0.5
PLAY 1-10 MyClip IN 25 OUT 200 SPEED 2.0
PLAY 1-10 MyClip PINGPONG SPEED 0.5
```

### Parameters

| Parameter | Description | Range / Unit |
| :--- | :--- | :--- |
| **value** | Playback speed multiplier | Float — `1.0` = normal, `0.5` = half speed, `2.0` = double speed, `0.0` = freeze, negative = reverse |

### Speed Behaviour

| Speed Value | Behaviour |
| :--- | :--- |
| `> 1.0` | Fast forward — intermediate frames are dropped |
| `1.0` | Normal playback |
| `0.0 < speed < 1.0` | Slow motion — frames are repeated |
| `0.0` | Freeze on current frame |
| `< 0.0` | Reverse playback (see codec notes below) |

### Codec Considerations for Reverse Playback

Reverse playback works by seeking backward frame-by-frame each tick. The quality of reverse depends heavily on the source codec:

| Codec Type | Examples | Reverse Quality |
| :--- | :--- | :--- |
| **Intra-only** (recommended) | ProRes, DNxHD, Motion JPEG, DV, JPEG-XS | Smooth — every frame is a keyframe, seek is instant |
| **Long-GOP** | H.264, H.265/HEVC, MPEG-2 | Stuttery — seeks must wait for the previous keyframe to decode |

For production use, transcode long-GOP sources to an intra-only codec before using reverse playback.

### Audio at Non-Unity Speed

Audio frames are not resampled — they are played or dropped at the current speed:

- `SPEED 0.5` → audio frames repeat (sounds lower pitched / slow)
- `SPEED 2.0` → audio frames are skipped (sounds faster)
- `SPEED < 0.0` → audio plays **forward** at normal pitch (not reversed)

For correct pitch-compensated audio, combine with an audio filter:
```bash
CALL 1-10 AF "atempo=0.5"   # Valid range: 0.5–2.0 per filter
# Chain multiple for extreme values:
CALL 1-10 AF "atempo=0.5,atempo=0.5"   # 0.25× speed
```

### Looping with Reverse Playback

When loop is enabled (`CALL 1-10 LOOP 1`), reverse playback wraps from the IN point back to the OUT point seamlessly. Without loop enabled, playback freezes at the first frame.

### Usage Examples

**1. Half speed slow motion**
```bash
CALL 1-10 SPEED 0.5
```

**2. Double speed fast forward**
```bash
CALL 1-10 SPEED 2.0
```

**3. Freeze on current frame**
```bash
CALL 1-10 SPEED 0.0
```

**4. Full reverse playback**
```bash
CALL 1-10 SPEED -1.0
```

**5. Slow reverse**
```bash
CALL 1-10 SPEED -0.5
```

**6. Resume normal playback**
```bash
CALL 1-10 SPEED 1.0
```

**7. Query current speed**
```bash
CALL 1-10 SPEED
```

---

## Ping-Pong Loop

Automatically reverses playback direction each time a clip boundary is reached, creating a seamless back-and-forth loop without any manual command timing. This is fully frame-accurate — the direction flip happens deterministically inside `next_frame()` the moment the boundary is crossed.

### AMCP Command

Ping-pong can be set at load/play time or toggled at runtime via `CALL`:

```bash
PLAY  [channel]-[layer] [clip] PINGPONG            # load and play with ping-pong
LOAD  [channel]-[layer] [clip] PINGPONG            # pre-load with ping-pong
CALL  [channel]-[layer] PINGPONG [1|0]             # enable / disable at runtime
CALL  [channel]-[layer] PINGPONG                   # query current state
```

`PINGPONG` and `SPEED` can be combined with other `PLAY`/`LOAD` flags in the usual way:

```bash
PLAY 1-10 MyClip PINGPONG IN 25 OUT 200
PLAY 1-10 MyClip PINGPONG SPEED 0.5
```

### Behaviour

- Playback oscillates continuously between the IN point and the OUT point (or the full clip if neither is set).
- The absolute speed magnitude is preserved on every flip. `CALL 1-10 SPEED 0.5` with ping-pong enabled produces slow-motion bouncing.
- Ping-pong implies continuous looping — `LOOP` is not needed and has no additional effect when `PINGPONG` is set.
- **Codec note**: The reverse leg of each bounce has the same codec limitations as `SPEED -1.0`. Intra-only codecs (ProRes, DNxHD, MJPEG) give the smoothest results.

### Usage Examples

**1. Enable ping-pong at normal speed**
```bash
CALL 1-10 PINGPONG 1
```

**2. Ping-pong at half speed**
```bash
CALL 1-10 SPEED 0.5
CALL 1-10 PINGPONG 1
```

**3. Disable ping-pong (resumes forward looping if LOOP is set)**
```bash
CALL 1-10 PINGPONG 0
```

**4. Query current state**
```bash
CALL 1-10 PINGPONG
```

### OSC State

Both speed and ping-pong state are published to the OSC tree:
```
/channel/1/stage/layer/10/speed
/channel/1/stage/layer/10/pingpong
```

---

## Flip (Mirror)

GPU-accelerated horizontal and/or vertical mirror applied per-layer at no performance cost. Works on all layer types — standard clips, live inputs, and 360° layers equally. The flip is applied to the final UV sample coordinates, so it correctly mirrors the output image regardless of any other transforms on the layer.

### AMCP Command

```bash
MIXER [channel]-[layer] FLIP [mode]    # Set flip mode
MIXER [channel]-[layer] FLIP           # Query current mode
```

### Parameters

| Parameter | Description |
| :--- | :--- |
| **H** | Horizontal mirror — left becomes right |
| **V** | Vertical mirror — top becomes bottom |
| **HV** | Both axes — equivalent to a 180° rotation |
| **NONE** | Reset — no mirror (default) |

### Usage Examples

**1. Mirror a layer horizontally**
```bash
MIXER 1-10 FLIP H
```

**2. Mirror vertically**
```bash
MIXER 1-10 FLIP V
```

**3. Both axes**
```bash
MIXER 1-10 FLIP HV
```

**4. Reset**
```bash
MIXER 1-10 FLIP NONE
```

**5. Query current state**
```bash
MIXER 1-10 FLIP
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
