# Keyframes Module

The keyframes module provides NLE-style per-property animation on a unified timeline.
Keyframe timelines are bound to a channel/layer and driven by the media producer's
playback position, giving frame-accurate animation of any MIXER property — geometry,
color grading, blur, shapes, projection, and more.

## Architecture

```
  Client (AMCP)
       │
       ▼
  keyframe_commands ── parse JSON, validate, dispatch
       │
       ▼
  stage executor ── single-threaded, owns all KF state
       │
       ├─ kf_timelines_      map<layer, keyframe_timeline>
       ├─ kf_armed_          set<layer>
       ├─ kf_last_values_    map<layer, kf_values>  (change detection)
       ├─ kf_media_time_override_   (from SEEK / CALL SEEK)
       └─ kf_last_frame_number_     (for override clearing)
       │
       ▼
  Render loop (per frame)
       │
       ├─ tick tweens
       ├─ for each armed layer:
       │    ├─ compute media_time from producer frame_number / fps
       │    ├─ interpolate timeline → sparse kf_values
       │    ├─ change-detect against last frame
       │    └─ apply_kf_to_transform → overwrite tweened_transform
       └─ render layers
```

### Key Design Decisions

- **Sparse values** — Each keyframe only stores the fields it animates.  A keyframe
  with `{"opacity": 0.5}` does not affect position, color, or any other property.
- **Per-field interpolation** — Each field is interpolated independently between the
  two nearest keyframes that contain that field.  Fields absent from a keyframe hold
  their last value.
- **Single-threaded** — All keyframe state lives on the stage executor thread.
  No mutexes, no races.
- **Easing resolved at parse time** — Easing names are converted to function pointers
  during JSON parsing, so the per-frame interpolation path has zero string operations.
- **Field descriptors** — A single table in `keyframe_fields.cpp` maps JSON key names
  to getter/setter/default/interpolation-kind.  Adding a new animatable property
  requires adding one line to this table.

---

## AMCP Commands

All commands target a specific channel and layer: `KEYFRAMES <verb> <channel>-<layer> [args]`

### KEYFRAMES SET

Upload a keyframe timeline.  The layer is automatically **disarmed** after SET.

```
KEYFRAMES SET 1-10 ({"keyframes":[
  {"time_secs":0.0, "easing":"LINEAR", "opacity":1.0, "fill_x":0.0},
  {"time_secs":2.0, "easing":"EASEINOUTCUBIC", "opacity":0.0, "fill_x":0.5},
  {"time_secs":5.0, "easing":"LINEAR", "opacity":1.0, "fill_x":0.0}
]})
```

- Wrap the JSON in parentheses `(...)` to protect it from AMCP tokenization.
- Only include the fields you want to animate (sparse).
- `time_secs` is relative to the media producer's playback position (frame 0 = 0.0s).
- `easing` controls the interpolation curve **out of** this keyframe into the next.
- Maximum 10,000 keyframes per timeline; maximum 1 MB JSON payload.

**Response:** `202 KEYFRAMES OK`

### KEYFRAMES ARM

Activate keyframe evaluation for the layer.  Returns `404` if no timeline has been SET.

```
KEYFRAMES ARM 1-10
```

**Response:** `202 KEYFRAMES OK` or `404 KEYFRAMES ERROR ...`

### KEYFRAMES DISARM

Stop keyframe evaluation.  The timeline is preserved — ARM again to resume.

```
KEYFRAMES DISARM 1-10
```

**Response:** `202 KEYFRAMES OK`

### KEYFRAMES CLEAR

Remove the timeline and all associated state for the layer.

```
KEYFRAMES CLEAR 1-10
```

**Response:** `202 KEYFRAMES OK`

### KEYFRAMES GET

Retrieve the current timeline as JSON.

```
KEYFRAMES GET 1-10
```

**Response:** `201 KEYFRAMES OK` followed by JSON, or `404` if no timeline is set.

### KEYFRAMES PATCH

Update specific fields of the keyframe nearest to the given time (within 1 ms).

```
KEYFRAMES PATCH 1-10 2.0 ({"opacity":0.75, "fill_x":0.25})
```

- Only the fields in the patch are modified; existing fields are preserved.
- Unknown field names are silently ignored.

**Response:** `202 KEYFRAMES OK` or `404 KEYFRAMES ERROR KF not found at given time`

### KEYFRAMES SEEK

Override the media time used for keyframe evaluation.  Useful for scrubbing the
timeline while the producer is paused.

```
KEYFRAMES SEEK 1-10 3.5
```

The override is automatically cleared when the producer's `frame_number` advances
(i.e., when playback resumes).

**Response:** `202 KEYFRAMES OK`

### KEYFRAMES STATUS

Query armed state and keyframe count (single atomic snapshot).

```
KEYFRAMES STATUS 1-10
```

**Response:** `201 KEYFRAMES OK` followed by `{"armed":true,"keyframe_count":3}`

---

## Typical Workflow

```
1.  PLAY 1-10 myVideo                          ← load and play media
2.  KEYFRAMES SET 1-10 ({"keyframes":[...]})    ← upload timeline
3.  KEYFRAMES ARM 1-10                          ← activate
4.  (playback runs — keyframes auto-evaluate)
5.  KEYFRAMES PATCH 1-10 2.0 ({"opacity":0.8}) ← live adjustment
6.  KEYFRAMES DISARM 1-10                       ← pause animation
7.  KEYFRAMES ARM 1-10                          ← resume
8.  KEYFRAMES CLEAR 1-10                        ← clean up
```

### Scrubbing While Paused

```
PAUSE 1-10
KEYFRAMES SEEK 1-10 0.0     ← jump to start of timeline
KEYFRAMES SEEK 1-10 1.5     ← preview at 1.5 seconds
KEYFRAMES SEEK 1-10 3.0     ← preview at 3.0 seconds
RESUME 1-10                  ← override auto-clears on next frame
```

---

## Interpolation Modes

Each field has a **kind** that controls how it is interpolated:

| Kind | Behavior | Example fields |
|------|----------|----------------|
| **continuous** | Standard lerp with easing | `opacity`, `fill_x`, `brightness`, `blur_radius` |
| **angular** | Shortest-path 360° wrapping | `angle`, `hue_shift`, `proj_yaw`, `blur_angle`, `shape_gradient_angle` |
| **discrete** | Holds source keyframe value, no interpolation | `invert`, `flip_h`, `blur_type`, `blend_mode`, `shape_type` |

### Easing Functions

The `easing` field on each keyframe controls the curve **from** that keyframe **to**
the next.  Names are case-insensitive.

| Category | In | Out | InOut |
|----------|----|-----|-------|
| Linear | `LINEAR` | — | — |
| Cubic (default) | `EASEIN` / `EASEINCUBIC` | `EASEOUT` / `EASEOUTCUBIC` | `EASE` / `EASEINOUTCUBIC` |
| Quad | `EASEINQUAD` | `EASEOUTQUAD` | `EASEINOUTQUAD` |
| Quart | `EASEINQUART` | `EASEOUTQUART` | `EASEINOUTQUART` |
| Quint | `EASEINQUINT` | `EASEOUTQUINT` | `EASEINOUTQUINT` |
| Sine | `EASEINSINE` | `EASEOUTSINE` | `EASEINOUTSINE` |
| Expo | `EASEINEXPO` | `EASEOUTEXPO` | `EASEINOUTEXPO` |
| Circ | `EASEINCIRC` | `EASEOUTCIRC` | `EASEINOUTCIRC` |
| Back | `EASEINBACK` | `EASEOUTBACK` | `EASEINOUTBACK` |
| Bounce | `EASEINBOUNCE` | `EASEOUTBOUNCE` | `EASEINOUTBOUNCE` |
| Elastic | `EASEINELASTIC` | `EASEOUTELASTIC` | `EASEINOUTELASTIC` |

Unknown easing names fall back to `LINEAR` with a one-time warning in the log.

---

## Animatable Fields

All values are `double`.  Angular fields are in **degrees** in JSON (converted to
radians internally).  Boolean/enum fields use `0.0`/`1.0` (threshold at 0.5).

### Basic

| Field | Default | Description |
|-------|---------|-------------|
| `opacity` | 1.0 | Layer opacity |
| `contrast` | 1.0 | Contrast multiplier |
| `brightness` | 1.0 | Brightness multiplier |
| `saturation` | 1.0 | Saturation multiplier |

### Geometry

| Field | Default | Description |
|-------|---------|-------------|
| `anchor_x`, `anchor_y` | 0.0 | Anchor point (normalised) |
| `fill_x`, `fill_y` | 0.0 | Fill translation |
| `fill_sx`, `fill_sy` | 1.0 | Fill scale |
| `clip_x`, `clip_y` | 0.0 | Clip translation |
| `clip_sx`, `clip_sy` | 1.0 | Clip scale |
| `angle` | 0.0 | Rotation (degrees, angular) |

### Crop

| Field | Default | Description |
|-------|---------|-------------|
| `crop_ul_x`, `crop_ul_y` | 0.0 | Upper-left crop corner |
| `crop_lr_x`, `crop_lr_y` | 1.0 | Lower-right crop corner |

### Perspective

| Field | Default | Description |
|-------|---------|-------------|
| `persp_ul_x/y`, `persp_ur_x/y` | corners | Corner-pin positions |
| `persp_lr_x/y`, `persp_ll_x/y` | corners | Corner-pin positions |

### Projection

| Field | Default | Description |
|-------|---------|-------------|
| `proj_enable` | 0 | Enable projection (discrete) |
| `proj_yaw`, `proj_pitch`, `proj_roll` | 0.0 | Rotation (degrees, angular) |
| `proj_fov` | 90.0 | Field of view (degrees, angular) |
| `proj_offset_x`, `proj_offset_y` | 0.0 | Lens shift (NDC) |
| `proj_screen_arc` | 0.0 | Curved screen arc (degrees, angular) |
| `proj_curve_enable` | 0 | Enable curve compensation (discrete) |
| `proj_curve_type` | 0 | 0=flat, 1=cylinder, 2=sphere (discrete) |

### White/Tone Balance

| Field | Default | Description |
|-------|---------|-------------|
| `temperature` | 0.0 | -1 (cool) to +1 (warm) |
| `tint` | 0.0 | -1 (magenta) to +1 (green) |
| `shadows` | 0.0 | Shadow tone (-1..+1) |
| `highlights` | 0.0 | Highlight tone (-1..+1) |

### 3-Way Color

| Field | Default | Description |
|-------|---------|-------------|
| `lift_r/g/b` | 0.0 | Shadow offset per channel |
| `mid_r/g/b` | 1.0 | Midtone power per channel |
| `gain_r/g/b` | 1.0 | Highlight multiplier per channel |

### Hue / Flip

| Field | Default | Description |
|-------|---------|-------------|
| `hue_shift` | 0.0 | Hue rotation (degrees, angular) |
| `linear_saturation` | 1.0 | Linear saturation multiplier |
| `invert` | 0 | Invert image (discrete) |
| `flip_h`, `flip_v` | 0 | Mirror horizontal/vertical (discrete) |

### Levels

| Field | Default | Description |
|-------|---------|-------------|
| `levels_min_in`, `levels_max_in` | 0.0, 1.0 | Master input range |
| `levels_gamma` | 1.0 | Master gamma |
| `levels_min_out`, `levels_max_out` | 0.0, 1.0 | Master output range |

### Per-Channel RGB Levels

| Field pattern | Default | Description |
|---------------|---------|-------------|
| `rgb_r_min_in`, `rgb_r_max_in`, `rgb_r_gamma`, `rgb_r_min_out`, `rgb_r_max_out` | 0/1/1/0/1 | Red channel |
| `rgb_g_*` | same | Green channel |
| `rgb_b_*` | same | Blue channel |

### Blur

| Field | Default | Description |
|-------|---------|-------------|
| `blur_radius` | 0.0 | Blur strength |
| `blur_angle` | 0.0 | Direction (degrees, angular) |
| `blur_center_x/y` | 0.5 | Zoom/radial center |
| `blur_tilt_y`, `blur_tilt_h` | 0.5, 0.2 | Tilt-shift parameters |
| `blur_type` | 0 | 0=gaussian, 1=box, 2=directional, 3=zoom, 4=tilt_shift, 5=lens (discrete) |

### ASC CDL

| Field | Default | Description |
|-------|---------|-------------|
| `cdl_slope_r/g/b` | 1.0 | Slope per channel |
| `cdl_offset_r/g/b` | 0.0 | Offset per channel |
| `cdl_power_r/g/b` | 1.0 | Power per channel |
| `cdl_saturation` | 1.0 | CDL saturation |

### Split Toning

| Field | Default | Description |
|-------|---------|-------------|
| `split_shadow_r/g/b` | 0.0 | Shadow tint RGB |
| `split_highlight_r/g/b` | 0.0 | Highlight tint RGB |
| `split_balance` | 0.5 | Shadow/highlight crossover |

### Gamut Compression

| Field | Default | Description |
|-------|---------|-------------|
| `gamut_compress` | 0 | Enable (discrete) |
| `gc_cyan`, `gc_magenta`, `gc_yellow` | 1.147, 1.264, 1.312 | ACES limits |

### LUT / Sharpening / Grain

| Field | Default | Description |
|-------|---------|-------------|
| `lut3d_strength` | 1.0 | 3D LUT mix (0..1) |
| `sharpen_amount` | 0.0 | Sharpening strength |
| `sharpen_radius` | 1.0 | Sharpening kernel radius |
| `grain_intensity` | 0.0 | Film grain strength |
| `grain_size` | 1.0 | Grain pattern scale |

### Secondary Qualifier

| Field | Default | Description |
|-------|---------|-------------|
| `qualifier_enable` | 0 | Enable HSL qualifier (discrete) |
| `qual_target_hue` | 0.0 | Target hue (0..1) |
| `qual_hue_width` | 0.1 | Hue selection width |
| `qual_min_sat`, `qual_max_sat` | 0.2, 1.0 | Saturation range |
| `qual_min_lum`, `qual_max_lum` | 0.0, 1.0 | Luminance range |
| `qual_softness` | 0.1 | Feather width |
| `qual_exposure` | 0.0 | Exposure offset for qualified region |
| `qual_sat_offset` | 0.0 | Saturation offset |
| `qual_hue_offset` | 0.0 | Hue offset (degrees) |

### Color Grade

| Field | Default | Description |
|-------|---------|-------------|
| `color_grade_enable` | 0 | Enable pipeline (discrete) |
| `color_grade_exposure` | 1.0 | Exposure multiplier |
| `color_grade_input_transfer` | 0 | Transfer function enum (discrete) |
| `color_grade_input_gamut` | 0 | Input gamut enum (discrete) |
| `color_grade_tone_mapping` | 0 | Tone mapping enum (discrete) |
| `color_grade_output_gamut` | 0 | Output gamut enum (discrete) |
| `color_grade_output_transfer` | 0 | Output transfer enum (discrete) |

### Shape

| Field | Default | Description |
|-------|---------|-------------|
| `shape_enable` | 0 | Enable shape overlay (discrete) |
| `shape_type` | 0 | 0=rect, 1=rounded_rect, 2=circle, 3=ellipse (discrete) |
| `shape_fill_type` | 0 | 0=solid, 1=linear, 2=radial, 3=conic (discrete) |
| `shape_center_x/y` | 0.5 | Center position |
| `shape_size_x/y` | 0.5 | Size |
| `shape_corner_radius` | 0.0 | Corner radius |
| `shape_edge_softness` | 0.005 | Anti-aliasing feather |
| `shape_gradient_angle` | 0.0 | Gradient angle (degrees, angular) |
| `shape_gradient_cx/cy` | 0.5 | Gradient center |
| `shape_stroke_enable` | 0 | Enable stroke (discrete) |
| `shape_stroke_width` | 0.0 | Stroke width |
| `shape_color1_r/g/b/a` | 1,1,1,1 | Primary fill RGBA |
| `shape_color2_r/g/b/a` | 0,0,0,0 | Secondary fill RGBA |
| `shape_stroke_r/g/b/a` | 1,1,1,1 | Stroke RGBA |

### Chroma Key

| Field | Default | Description |
|-------|---------|-------------|
| `chroma_enable` | 0 | Enable chroma key (discrete) |
| `chroma_target_hue` | 0.0 | Target hue |
| `chroma_hue_width` | 0.0 | Hue selection width |
| `chroma_min_sat` | 0.0 | Minimum saturation |
| `chroma_min_bright` | 0.0 | Minimum brightness |
| `chroma_softness` | 0.0 | Key softness |
| `chroma_spill` | 0.0 | Spill suppression |
| `chroma_spill_sat` | 1.0 | Spill suppression saturation |
| `chroma_show_mask` | 0 | Show matte (discrete) |

### Subsystem Enables & Enum Selectors

| Field | Default | Description |
|-------|---------|-------------|
| `enable_geometry` | 0 | Enable geometry modifiers (discrete) |
| `blur_enable` | 0 | Enable blur subsystem (discrete) |
| `rgb_enable` | 0 | Enable per-channel RGB levels (discrete) |
| `curves_enable` | 0 | Enable tone curves (discrete) |
| `blend_mode` | 0 | Blend mode enum (discrete) |

### Auto-Enable Behavior

When keyframe values include fields from certain subsystems but the corresponding
enable flag is **not** explicitly included in the keyframe:

- **Geometry fields** (`fill_x`, `angle`, `crop_*`, etc.) → `enable_geometry` is
  automatically set to `true`.
- **Blur fields** (`blur_radius`, `blur_center_x`, etc.) → `blur_enable` is set to
  `true` if `blur_radius > 0`.
- **RGB level fields** (`rgb_r_gamma`, etc.) → `rgb_enable` is set to `true`.

To explicitly control the enable state, include the enable field in your keyframe.

---

## Examples

### Fade In/Out with Position Slide

```
KEYFRAMES SET 1-10 ({"keyframes":[
  {"time_secs":0.0,  "easing":"EASEINOUTSINE", "opacity":0.0, "fill_x":-0.2},
  {"time_secs":1.0,  "easing":"LINEAR",        "opacity":1.0, "fill_x":0.0},
  {"time_secs":9.0,  "easing":"EASEINOUTSINE", "opacity":1.0, "fill_x":0.0},
  {"time_secs":10.0, "easing":"LINEAR",        "opacity":0.0, "fill_x":0.2}
]})
KEYFRAMES ARM 1-10
```

### Animated Color Grade Over Time

```
KEYFRAMES SET 1-20 ({"keyframes":[
  {"time_secs":0.0, "easing":"EASEINOUTCUBIC",
   "temperature":0.0, "saturation":1.0, "contrast":1.0},
  {"time_secs":5.0, "easing":"EASEINOUTCUBIC",
   "temperature":0.3, "saturation":0.7, "contrast":1.2},
  {"time_secs":10.0, "easing":"LINEAR",
   "temperature":0.0, "saturation":1.0, "contrast":1.0}
]})
KEYFRAMES ARM 1-20
```

### Rotating Projection (Virtual Camera Pan)

```
KEYFRAMES SET 1-10 ({"keyframes":[
  {"time_secs":0.0,  "easing":"EASEINOUTQUAD", "proj_enable":1, "proj_yaw":0.0},
  {"time_secs":3.0,  "easing":"EASEINOUTQUAD", "proj_yaw":45.0},
  {"time_secs":6.0,  "easing":"EASEINOUTQUAD", "proj_yaw":-30.0},
  {"time_secs":10.0, "easing":"LINEAR",        "proj_yaw":0.0}
]})
KEYFRAMES ARM 1-10
```

### Shape Wipe Transition

```
KEYFRAMES SET 1-10 ({"keyframes":[
  {"time_secs":0.0, "easing":"EASEINOUTCUBIC",
   "shape_enable":1, "shape_size_x":0.0, "shape_size_y":0.0,
   "shape_edge_softness":0.05},
  {"time_secs":2.0, "easing":"LINEAR",
   "shape_size_x":1.0, "shape_size_y":1.0}
]})
KEYFRAMES ARM 1-10
```

### Live Adjustment with PATCH

```
KEYFRAMES SET 1-10 ({"keyframes":[
  {"time_secs":0.0, "easing":"LINEAR", "opacity":1.0, "blur_radius":0.0},
  {"time_secs":5.0, "easing":"LINEAR", "opacity":0.5, "blur_radius":10.0}
]})
KEYFRAMES ARM 1-10

REM  Adjust the second keyframe without re-uploading the entire timeline:
KEYFRAMES PATCH 1-10 5.0 ({"opacity":0.8, "blur_radius":5.0})
```

---

## Best Practices

1. **Only include fields you animate.**  Sparse keyframes are cheaper to evaluate and
   easier to maintain.  Don't set `opacity:1.0` on every keyframe if it never changes.

2. **ARM after SET.**  SET always disarms the layer.  This prevents partially-uploaded
   timelines from being evaluated.

3. **Use PATCH for live tweaks.**  PATCH modifies individual fields of a single keyframe
   without re-uploading the entire timeline.  It's atomic and avoids the arm/disarm cycle.

4. **Use SEEK for offline preview.**  When the producer is paused, SEEK lets you scrub
   the timeline without affecting media playback.  The override clears automatically
   when playback resumes.

5. **Keep keyframe counts reasonable.**  While the system supports up to 10,000
   keyframes, most use cases need 2–20.  Each armed layer evaluates O(fields × keyframes)
   per frame.

6. **Use appropriate easing.**  `EASEINOUTCUBIC` gives natural-looking motion.  `LINEAR`
   is best for mechanical or constant-rate changes.  `EASEINBACK` / `EASEOUTBACK` add
   subtle overshoot.  `EASEOUTBOUNCE` gives a bounce effect.

7. **Check STATUS before ARM in automation.**  `KEYFRAMES STATUS` returns the armed
   state and keyframe count in a single atomic operation — useful for client-side state
   management.

8. **CLEAR when done.**  Keyframe timelines persist until explicitly cleared.  If you
   reload a different clip on the same layer, CLEAR the old timeline first.

9. **Angles are in degrees.**  All angular fields (`angle`, `proj_yaw`, `hue_shift`,
   `blur_angle`, `shape_gradient_angle`, `proj_screen_arc`, etc.) use degrees in JSON.
   Angular interpolation automatically takes the shortest path around 360°.

10. **Discrete fields snap, they don't interpolate.**  Boolean and enum fields
    (`invert`, `flip_h`, `blur_type`, `blend_mode`, `shape_type`, etc.) hold the
    source keyframe's value for the entire segment.  They switch at the exact keyframe
    boundary.

---

## Limitations

- **No audio keyframing.**  The `volume` field on `audio_transform` is not currently
  exposed.  Use MIXER VOLUME for audio fades.
- **No curve point animation.**  Tone curve control points and hue curve LUTs cannot
  be keyframed (they are array/pointer types, not scalar).
- **No 3D LUT switching.**  The `lut3d` shared pointer is not keyframeable.  Use
  `lut3d_strength` to crossfade a LUT in/out.
- **KEYFRAMES SEEK is for paused scrubbing.**  The override is cleared as soon as the
  producer's frame number advances.  It is not a persistent time offset.
- **Keyframes override MIXER tweens.**  While armed, keyframe values overwrite any
  active MIXER FILL/OPACITY/etc. tweens on the same fields every frame.  Disarm the
  timeline to let MIXER tweens take effect again.
