# CasparCG Blur Commands Guide

The CasparCG mixer includes hardware-accelerated OpenGL blurs to achieve distinct creative treatments. The blur modifier manipulates the raw texture samples based on requested types, sizes (radii), and directional vectors natively within the GPU.

## Command Syntax

```amcp
MIXER <channel>-<layer> BLUR <radius> [type] [angle] [center_x] [center_y] [tilt_y] [tilt_h] [duration] [tween]
```
> **Note:** To disable a blur effect entirely, set `radius` to `0`. 

### Parameters:
*   `radius` (Required): The sheer size/intensity of the effect (in pixels).
*   `type` (Optional): String name specifying the algorithm applied (Default is `gaussian`). 
*   `angle` (Optional): Vector orientation driving the blur (used for `directional`, `tilt-shift`). Represented in degrees (Default is `0`). 
*   `center_x`, `center_y` (Optional): The normalized `[0.0, 1.0]` x/y focal point (used for origin tracking in `zoom`, `tilt-shift`, or `lens`). Default is `0.5`, center.
*   `tilt_y` (Optional): Base origin parameter for gradient blurring (used in `tilt-shift`).
*   `tilt_h` (Optional): Base threshold boundary size for gradient boundaries (used in `tilt-shift`).
*   `duration`, `tween` (Optional): Standard animation duration/tween type string. 

## Supported Blur Types

### `gaussian` (Default)
Applies a standard, organic soft Gaussian falloff blur matching physical camera diffusion across a general radius vector. This is perfect for background obscuring and standard out-of-focus simulation.
**Example (Blur layer 10 by 15 pixels softly):**
`MIXER 1-10 BLUR 15 gaussian`

### `box`
A uniform averaging filter across a block of pixels. Computationally cheaper but visually produces slight "squared" or "stepped" artifacts at high thresholds. Useful for blocky/stylized aesthetics.
**Example (High intensity box blur):**
`MIXER 1-10 BLUR 20 box`

### `directional`
Smeared motion blur that drags pixels violently along a defined `angle`, simulating fast movement or whipping pans across the screen. 
**Example (Motion blur dragged across a 45-degree angle):**
`MIXER 1-10 BLUR 30 directional 45`

### `zoom`
Radiates blurring dynamically outwards originating from the given `center_x` / `center_y` position. Creates extreme warp-speed / hyperspace rush effects.
**Example (Warp zoom originating slightly off-center right):**
`MIXER 1-10 BLUR 25 zoom 0 0.8 0.5`

### `tilt-shift`
Re-creates the miniature model effect achievable on physically tilted optics. Leaves a sharp central line while sharply blurring top and bottom planes based on rotation (`angle`), positioning (`tilt_y`), and band width (`tilt_h`).
**Example (Tilt-shift with a thin sharp band angled slightly across the screen):**
`MIXER 1-10 BLUR 15 tilt-shift 15 0.5 0.5 0.5 0.1`

### `lens` (Bokeh)
A high-quality rendering simulation of optical bokeh using pentagon/hexagon sampling rotations. Bright highlights generally pop and expand into rounded overlapping rings simulating heavy depth-of-field separation.
**Example (Simulate a heavy cinematic background depth-of-field):**
`MIXER 1-10 BLUR 18 lens`
