# LED-Wall Color Calibration

Channel-master color calibration for LED volumes, applied as a single
display-to-display 3D LUT over the **final composited channel output**. Every
consumer attached to the channel (DeckLink SDI, `vulkan_output` HDMI/DP, NDI,
file) receives the corrected pixels.

The calibration LUT is intended for **whole-screen** correction: one channel
drives one LED screen and carries one calibration LUT. Individual panels or
tiles are **not** addressed — that remains the job of the LED processor.

The LUT itself is typically solved by [OpenVPCal](https://github.com/Netflix-Skunkworks/OpenVPCal)
(camera-based, Apache-2.0) and exported as a `.cube` 3D LUT. CasparVP only
*applies* the LUT; it does not perform the colorimetric solve.

## How it differs from `MIXER ... LUT3D`

| | `MIXER <ch>-<layer> LUT3D` | `CALIBRATION <ch> LUT` |
| :--- | :--- | :--- |
| Scope | Single layer | Whole channel (post-composite) |
| Applied | Per-layer, before compositing | Once, on the final output frame |
| Use case | Creative look on one source | LED-wall device calibration |
| Reaches all consumers | No (only that layer) | Yes |

Both use the same `.cube` parser and the same GPU 3D-LUT sampling path, so the
OpenGL and Vulkan back-ends produce identical results.

## AMCP Commands

```bash
CALIBRATION <channel> LUT <file.cube> [strength]   # load / replace the LUT
CALIBRATION <channel> CLEAR                          # remove the LUT
CALIBRATION <channel> BYPASS <0|1>                   # toggle without unloading
CALIBRATION <channel> INFO                           # query current state
CALIBRATION <channel>                                # query (same as INFO)
```

### Parameters

| Parameter | Description | Range / Unit |
| :--- | :--- | :--- |
| **file.cube** | Path to a `.cube` 3D LUT. Absolute, or relative to the media folder. | Path |
| **strength** | Blend between the input and the LUT result. | 0.0–1.0 (default 1.0) |
| **0\|1** | `BYPASS 1` disables the LUT without unloading; `BYPASS 0` re-enables. | Flag |

### Responses

```text
202 CALIBRATION OK                 # LUT / CLEAR / BYPASS accepted
404 CALIBRATION LOAD FAILED        # .cube not found or malformed
403 CALIBRATION ERROR              # bad subcommand / missing argument
201 CALIBRATION OK                 # INFO query (state on the next line)
NONE                               #   no LUT loaded
ENABLED SIZE 33 STRENGTH 1 BYPASS 0 PATH C:/luts/wall1.cube
```

## Typical Workflow

1. **Generate patches** with OpenVPCal for the target wall (gamut, EOTF, peak
   luminance). Export as a DPX/TIFF/PNG sequence (display-encoded).
2. **Bypass** any existing calibration so patches are measured clean:
   ```bash
   CALIBRATION 1 BYPASS 1
   ```
3. **Play the patch sequence** on the calibration channel with an identity grade
   and the correct wall EOTF/range.
4. **Capture** the camera return (recorded plate file, or live DeckLink capture).
5. **Solve** with OpenVPCal (headless): it ingests the plate and exports a
   display-to-display `.cube`.
6. **Apply** the solved LUT and re-enable correction:
   ```bash
   CALIBRATION 1 LUT wall1.cube
   CALIBRATION 1 BYPASS 0
   ```

> **Important:** Always `BYPASS 1` before shooting calibration patches. Measuring
> patches with the calibration LUT still active would fold the previous
> correction into the new solve.

## Notes

- The LUT is applied **after** the channel's output color encoding
  (display-to-display), so the input `.cube` should be display-encoded in the
  channel's output color space — exactly what OpenVPCal's default
  *display-to-display* export produces.
- `strength` lets you dial back the correction (e.g. for A/B comparison) without
  re-exporting the LUT.
- Calibration state is held on the channel mixer and is identical across the
  OpenGL and Vulkan back-ends.

## Client UI (casparcg-360-client)

The 360 client provides a **Calibration** tab that orchestrates the whole
workflow against the installed OpenVPCal binary (headless) and the
`CALIBRATION` AMCP command:

- **Setup** — channel selector, OpenVPCal binary path, working folder
  (persisted to `calibration_config.json`).
- **Wall Parameters** — target gamut/EOTF, peak luminance (nits), camera gamut,
  resolution, and EOTF-correction / gamut-compression / avoid-clipping toggles.
- **Workflow** — *Generate Patterns* → *Bypass while shooting* → *Solve* (pick
  the recorded plate folder) → the exported `.cube` is auto-filled.
- **Channel Calibration LUT** — *Apply* / *Clear* / *Query (INFO)* with a
  strength control; these map directly to the `CALIBRATION` subcommands.
- A log console shows OpenVPCal output and the AMCP responses.

The client builds an OpenVPCal *project settings* JSON and invokes the binary
with `--ui=false`; it does not bundle OpenVPCal's Python stack.

