# MIXER SHAPE — 2D Shape & Gradient Overlay

The `MIXER SHAPE` command renders GPU-accelerated 2D vector shapes directly
onto any CasparCG layer using Signed Distance Field (SDF) shaders. Shapes
can be filled with solid colours or linear, radial, and conic gradients, each
with independently controlled per-stop opacity. Outlines (strokes) are also
supported. All numeric parameters animate using the standard CasparCG
tweening system.

The shape is composited *over* the layer's existing content, so it works on
top of video, images, solid colours, or transparent layers equally.

---

## Table of Contents

1. [Command Syntax](#command-syntax)
2. [Core Parameters](#core-parameters)
3. [Shape Types](#shape-types)
4. [Fill Types & Gradients](#fill-types--gradients)
5. [Colour & Per-Stop Opacity](#colour--per-stop-opacity)
6. [Stroke (Outline)](#stroke-outline)
7. [Animation & Tweening](#animation--tweening)
8. [Querying Current State](#querying-current-state)
9. [Disabling a Shape](#disabling-a-shape)
10. [Worked Examples](#worked-examples)
11. [Developer Reference](#developer-reference)

---

## Command Syntax

```amcp
MIXER <channel>-<layer> SHAPE <type> <cx> <cy> <w> <h>
    [CORNER_RADIUS <r>]
    [SOFTNESS <s>]
    [FILL <fill_type>]
    [COLOR1 <#rrggbbaa>]
    [COLOR2 <#rrggbbaa>]
    [ANGLE <degrees>]
    [GRADIENT_CENTER <gx> <gy>]
    [STROKE <width> <#rrggbbaa>]
    [DURATION <frames>]
    [TWEEN <curve>]
```

All keyword arguments are optional and order-independent after the five
positional parameters (`type cx cy w h`).

---

## Core Parameters

| Parameter | Type | Default | Range | Description |
| :--- | :--- | :--- | :--- | :--- |
| `type` | string | — | see [Shape Types](#shape-types) | Shape geometry |
| `cx` | float | — | 0.0 – 1.0 | Horizontal centre (normalised, 0 = left edge, 1 = right edge) |
| `cy` | float | — | 0.0 – 1.0 | Vertical centre (normalised, 0 = top edge, 1 = bottom edge) |
| `w` | float | — | 0.0 – 1.0 | Width of the bounding box (normalised) |
| `h` | float | — | 0.0 – 1.0 | Height of the bounding box (normalised) |
| `CORNER_RADIUS r` | float | `0.0` | 0.0 – 0.5 | Corner rounding radius; only used by `ROUNDED_RECT` |
| `SOFTNESS s` | float | `0.005` | 0.0 – 0.1 | Anti-aliasing feather width at the shape edge |

> **Coordinate system:** All positions and sizes are in *normalised layer space*
> — `0.0` is the left/top edge and `1.0` is the right/bottom edge of the layer.
> The centre of the frame is `0.5 0.5`.

---

## Shape Types

| Value | Description |
| :--- | :--- |
| `RECT` | Axis-aligned rectangle |
| `ROUNDED_RECT` | Rectangle with rounded corners — requires `CORNER_RADIUS` |
| `CIRCLE` | Perfect circle — `w` and `h` should be equal; radius = `w / 2` |
| `ELLIPSE` | Ellipse fitted to the `w × h` bounding box |

---

## Fill Types & Gradients

Specified with the `FILL` keyword. Requires `COLOR1` and optionally `COLOR2`.

| Value | Description |
| :--- | :--- |
| `SOLID` | Flat colour fill using `COLOR1`. `COLOR2` is ignored. |
| `LINEAR` | Linear gradient from `COLOR1` to `COLOR2` along `ANGLE`. |
| `RADIAL` | Radial gradient radiating outward from `GRADIENT_CENTER`. |
| `CONIC` | Conic (angular sweep) gradient rotating around `GRADIENT_CENTER`. |

### `FILL LINEAR` — additional parameters

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `ANGLE <degrees>` | `0.0` | Direction of the gradient in degrees. `0` = left→right, `90` = top→bottom, `180` = right→left, `270` = bottom→top. Any intermediate angle is supported. |

### `FILL RADIAL` and `FILL CONIC` — additional parameters

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `GRADIENT_CENTER <gx> <gy>` | `0.5 0.5` | Normalised origin of the radial or conic gradient (relative to the full layer frame, not the shape bounding box). |

---

## Colour & Per-Stop Opacity

Colours are specified as 8-digit hex strings: `#RRGGBBAA`

| Byte | Channel | Range |
| :--- | :--- | :--- |
| `RR` | Red | `00` – `FF` |
| `GG` | Green | `00` – `FF` |
| `BB` | Blue | `00` – `FF` |
| `AA` | **Alpha / opacity of this stop** | `00` = fully transparent · `FF` = fully opaque |

The alpha of each stop is **independent**. The shader interpolates all four
channels — R, G, B, and A — across the gradient, so you can fade from a
fully opaque colour to a completely transparent one in one step.

A 6-digit `#RRGGBB` shorthand is also accepted; it is treated as `#RRGGBBFF`
(fully opaque).

| Example colour | Result |
| :--- | :--- |
| `#FF0000FF` | Solid red, fully opaque |
| `#FF000080` | Red at ~50% opacity |
| `#FF000080 COLOR2 #0000FFFF` | Red (50% opaque) → Blue (fully opaque) gradient |
| `#FF0000FF COLOR2 #FF000000` | Solid red fading to fully transparent |

> When using `FILL SOLID`, only `COLOR1` is used. The `AA` byte still
> controls the overall opacity of the filled shape.

---

## Stroke (Outline)

Add an outline ring around the shape edge with `STROKE <width> <#rrggbbaa>`.

| Parameter | Type | Default | Range | Description |
| :--- | :--- | :--- | :--- | :--- |
| `width` | float | `0.0` | 0.0 – 0.1 | Stroke width in normalised units |
| `#rrggbbaa` | hex colour | `#FFFFFFFF` | — | Stroke colour and opacity |

The stroke is rendered on top of the fill, centred on the SDF boundary.
Omitting `STROKE` or setting `width` to `0.0` disables the outline.

---

## Animation & Tweening

All numeric parameters animate using `DURATION <frames>` and `TWEEN <curve>`:

- **Animatable:** `cx`, `cy`, `w`, `h`, `corner_radius`, `softness`,
  `color1` (all 4 channels including alpha), `color2` (all 4 channels),
  `gradient_angle`, `gradient_center`, `stroke_width`, `stroke_color`.
- **Snapped (no interpolation):** `type`, `fill_type`, `stroke_enable` —
  these take effect immediately.

```amcp
MIXER 1-1 SHAPE CIRCLE 0.5 0.5 0.4 0.4 DURATION 50 TWEEN EASEINOUTQUAD
```

### Available tween curves

`LINEAR`, `EASE`, `EASEIN`, `EASEOUT`, `EASEINOUT`, `EASEINQUAD`,
`EASEOUTQUAD`, `EASEINOUTQUAD`, `EASEINCUBIC`, `EASEOUTCUBIC`,
`EASEINOUTCUBIC`, `EASEINQUART`, `EASEOUTQUART`, `EASEINOUTQUART`, and others.

---

## Querying Current State

Send the command with no arguments to read back the current configuration:

```amcp
MIXER 1-1 SHAPE
```

**Response when a shape is active:**
```
201 MIXER OK
CIRCLE 0.500000 0.500000 0.400000 0.400000 CORNER_RADIUS 0.000000 SOFTNESS 0.005000 FILL RADIAL COLOR1 #FF000080 COLOR2 #0000FFFF ANGLE 0.000000 GRADIENT_CENTER 0.500000 0.500000 STROKE 0.000000 #FFFFFFFF
```

**Response when no shape is active:**
```
201 MIXER OK
NONE
```

---

## Disabling a Shape

Pass `NONE` as the type to remove the shape from the layer:

```amcp
MIXER 1-1 SHAPE NONE
```

---

## Worked Examples

### 1. Solid white rectangle (lower-third bar)
A white bar covering the bottom quarter of the frame.
```amcp
PLAY 1-1 #00000000
MIXER 1-1 SHAPE RECT 0.5 0.875 1.0 0.25 FILL SOLID COLOR1 #FFFFFFFF
```

### 2. Semi-transparent black overlay
Full-frame darkening at ~60% opacity — useful as a title card background over video.
```amcp
PLAY 1-1 MyClip
MIXER 1-1 SHAPE RECT 0.5 0.5 1.0 1.0 FILL SOLID COLOR1 #00000099
```

### 3. Crisp anti-aliased red circle
```amcp
PLAY 1-1 #00000000
MIXER 1-1 SHAPE CIRCLE 0.5 0.5 0.3 0.3 FILL SOLID COLOR1 #FF0000FF SOFTNESS 0.005
```

### 4. Radial spotlight vignette over video
Red glow in the centre, fading to fully transparent at the edges — composited over a video clip.
```amcp
PLAY 1-1 MyClip
MIXER 1-1 SHAPE CIRCLE 0.5 0.5 0.8 0.8 FILL RADIAL COLOR1 #FF0000FF COLOR2 #FF000000 GRADIENT_CENTER 0.5 0.5
```

### 5. Linear gradient fade — opaque top, transparent bottom
Ideal for a fade-to-transparent title safe area at the top of frame.
```amcp
MIXER 1-1 SHAPE RECT 0.5 0.25 1.0 0.5 FILL LINEAR COLOR1 #000000FF COLOR2 #00000000 ANGLE 90
```

### 6. Conic colour wheel (full-frame)
```amcp
PLAY 1-1 #00000000
MIXER 1-1 SHAPE RECT 0.5 0.5 1.0 1.0 FILL CONIC COLOR1 #FF0000FF COLOR2 #0000FFFF GRADIENT_CENTER 0.5 0.5
```

### 7. Rounded rectangle panel with white outline
Dark semi-transparent panel card with a visible border — typical lower-third background.
```amcp
MIXER 1-1 SHAPE ROUNDED_RECT 0.5 0.5 0.6 0.3 CORNER_RADIUS 0.05 FILL SOLID COLOR1 #1A1A1AE0 STROKE 0.004 #FFFFFFFF
```

### 8. Animated circle expand
Circle grows from nothing to half the frame width over 30 frames with an ease-out cubic.
```amcp
MIXER 1-1 SHAPE CIRCLE 0.5 0.5 0.0 0.0 FILL SOLID COLOR1 #FF6600FF
MIXER 1-1 SHAPE CIRCLE 0.5 0.5 0.5 0.5 FILL SOLID COLOR1 #FF6600FF DURATION 30 TWEEN EASEOUTCUBIC
```

### 9. Gradient with animated per-stop opacity
The bottom fade starts invisible and animates to fully opaque over 25 frames —
useful for revealing a gradient overlay without a hard cut.
```amcp
MIXER 1-1 SHAPE RECT 0.5 0.5 1.0 1.0 FILL LINEAR COLOR1 #000000FF COLOR2 #00000000 ANGLE 90
MIXER 1-1 SHAPE RECT 0.5 0.5 1.0 1.0 FILL LINEAR COLOR1 #000000FF COLOR2 #000000FF ANGLE 90 DURATION 25 TWEEN EASEIN
```

### 10. Stroke-only focus ring (transparent fill)
A yellow ring around the centre of the frame — useful as a focus indicator or reticle.
```amcp
PLAY 1-1 MyClip
MIXER 1-1 SHAPE CIRCLE 0.5 0.5 0.5 0.5 FILL SOLID COLOR1 #00000000 STROKE 0.006 #FFFF00FF SOFTNESS 0.003
```

---

## Developer Reference

This section covers the internal implementation for developers extending or
modifying the feature.

### Pipeline

```
AMCPCommandsImpl.cpp  →  frame_transform.h (shape_config struct)
  →  frame_transform.cpp (tween / operator==)
  →  transforms.cpp (compositor routing)
  →  image_kernel.cpp (uniform dispatch)
  →  shader.frag (SDF + gradient GLSL)
```

### Files Modified

| File | Change |
| :--- | :--- |
| `src/core/frame/frame_transform.h` | Added `shape_type`, `shape_fill_type` enums and `shape_config` struct; added `shape` field to `image_transform` |
| `src/core/frame/frame_transform.cpp` | Extended `image_transform::tween()` for all shape fields; extended `operator==` |
| `src/protocol/amcp/AMCPCommandsImpl.cpp` | Added `parse_shape_type`, `parse_fill_type`, `parse_hex_color`, `rgba_to_hex` helpers and `mixer_shape_command`; registered `MIXER SHAPE` command |
| `src/accelerator/ogl/util/transforms.cpp` | Added `if (other.shape.enable)` routing in `apply_transform_colour_values` |
| `src/accelerator/ogl/util/shader.h` | Added `set(name, double, double, double, double)` overload declaration |
| `src/accelerator/ogl/util/shader.cpp` | Implemented the new `set` vec4 overload (calls `glUniform4f`) |
| `src/accelerator/ogl/image/image_kernel.cpp` | Added shape uniform dispatch block |
| `src/accelerator/ogl/image/shader.frag` | Added shape uniform declarations, SDF + gradient helper functions, compositing block in `main()` |

### Data Structure — `shape_config` (`frame_transform.h`)

```cpp
enum class shape_type    : int { rect = 0, rounded_rect = 1, circle = 2, ellipse = 3 };
enum class shape_fill_type : int { solid = 0, linear = 1, radial = 2, conic = 3 };

struct shape_config final {
    bool                  enable          = false;
    shape_type            type            = shape_type::rect;
    std::array<double, 2> center          = {0.5, 0.5};
    std::array<double, 2> size            = {0.5, 0.5};
    double                corner_radius   = 0.0;
    double                edge_softness   = 0.005;
    shape_fill_type       fill_type       = shape_fill_type::solid;
    std::array<double, 4> color1          = {1.0, 1.0, 1.0, 1.0}; // RGBA straight
    std::array<double, 4> color2          = {0.0, 0.0, 0.0, 0.0};
    double                gradient_angle  = 0.0;
    std::array<double, 2> gradient_center = {0.5, 0.5};
    bool                  stroke_enable   = false;
    double                stroke_width    = 0.0;
    std::array<double, 4> stroke_color    = {1.0, 1.0, 1.0, 1.0};
};
```

### UV Coordinate System in the Shader

`TexCoord2.st` (output by the vertex shader) carries 0–1 screen-space
coordinates: `(0,0)` = top-left, `(1,1)` = bottom-right. This is the
coordinate space used for all SDF position/size comparisons. Unlike
`TexCoord`, this coordinate does **not** carry a perspective `q` component,
so shape geometry is rectilinear and independent of `MIXER PERSPECTIVE` or
`MIXER ROTATION` transforms applied to the layer.

### SDF Functions

| Type | GLSL function | Note |
| :--- | :--- | :--- |
| `RECT` | `sdf_box(p, half_size)` | |
| `ROUNDED_RECT` | `sdf_rounded_box(p, half_size, r)` | `r` in same normalised space |
| `CIRCLE` | `sdf_circle(p, w/2)` | `h` ignored — use `w == h` |
| `ELLIPSE` | `sdf_ellipse(p, half_size)` | Håvlgaard iterative approximation |

`d < 0` = inside the shape · `d > 0` = outside. Alpha is derived via
`1.0 - smoothstep(-softness, 0.0, d)`.

### Compositing Order

The shape composite is applied in `main()` **after** the full colour-grading
pipeline (ACES, white balance, LMG, hue, tone curves, levels, CSB, invert)
but **before** opacity, key, blend-mode, and chroma-key processing. This means:

- The layer's underlying content is fully colour-graded before the shape is
  applied on top.
- The shape's specified RGBA colours are **not** affected by the layer's
  colour grading.
- `MIXER OPACITY` applies to the composited result (shape + content together).
- Global blend modes and chroma keying operate on the final composited pixel.

### Tweening Notes (`frame_transform.cpp`)

- **`do_tween()`** is called for: `center[2]`, `size[2]`, `corner_radius`,
  `edge_softness`, `color1[4]`, `color2[4]`, `gradient_angle`,
  `gradient_center[2]`, `stroke_width`, `stroke_color[4]`.
- **Snapped to `dest`** (no interpolation): `enable`, `type`, `fill_type`,
  `stroke_enable`.

### Adding More Gradient Stops (Future Work)

The current implementation supports two stops (`color1`, `color2`). To extend
to N stops, add a `std::array<double, 4> colors[N]` and a
`std::array<double, N> stops` (positions 0–1) to `shape_config`, upload them
as `glUniform4fv` arrays via `set_float_array`, and replace the `mix()` call
in `shape_compute_fill()` with a loop over the stop array.
