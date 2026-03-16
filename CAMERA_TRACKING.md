# Camera Tracking Module

The `tracking` module enables real-time camera tracking data to drive CasparCG layer transforms — either 2D fill/scale/rotation or full 360° equirectangular projection — all frame-accurately, without round-tripping through AMCP.

---

## Table of Contents

1. [Overview](#overview)
2. [How It Works](#how-it-works)
3. [Supported Protocols](#supported-protocols)
4. [Starting Position and Zeroing](#starting-position-and-zeroing)
5. [AMCP Command Reference](#amcp-command-reference)
6. [Transform Modes](#transform-modes)
7. [Zoom Calibration](#zoom-calibration)
8. [Axis Offsets and Scaling](#axis-offsets-and-scaling)
9. [Configuration File](#configuration-file)
10. [Building with VRPN](#building-with-vrpn)
11. [Worked Examples](#worked-examples)
12. [Architecture Reference](#architecture-reference)

---

## Overview

This module receives tracking data from physical camera rigs (head trackers, optical trackers, inertial systems) and maps each packet to a specific CasparCG channel/layer. Depending on the configured mode the data drives either:

- **360° mode** — the `MIXER PROJECTION` system: yaw/pitch/roll and FOV are updated each time a packet arrives, letting viewers pan inside an equirectangular video as the physical camera moves.
- **2D mode** — the `MIXER FILL` / `MIXER ROTATION` system: pan/tilt shift the layer's position, roll rotates it, and zoom adjusts the fill scale. Useful for tracking-responsive lower thirds, tracked planar inserts, or parallax-shifted graphics.

Multiple cameras, channels, and layers can all be active simultaneously. Each binding is independent.

---

## How It Works

```
Physical camera
     │ (UDP / VRPN)
     ▼
Protocol Receiver          (freed_receiver / osc_receiver / …)
     │ decoded camera_data struct (radians, mm, raw zoom)
     ▼
tracker_registry::on_data()
     │ look up each binding that matches camera_id
     ▼
stage::apply_transform()   (duration=0, tween=linear)
     │ same path as MIXER PROJECTION / MIXER FILL AMCP commands
     ▼
Mixer → Output
```

Tracking data is injected directly into the stage transform pipeline on the receiver's IO thread, bypassing the AMCP round-trip. At 50 Hz FreeD data hitting a 50 Hz channel the transform is always one packet fresh — typically ≤ 1 frame of latency.

---

## Supported Protocols

| Protocol | Default Port | Description |
| :--- | :--- | :--- |
| `FREED` | 6301 | FreeD D1 UDP (29-byte, big-endian, XOR checksum). The de-facto industry standard. Most tracking vendors support it natively: Mo-Sys, Stype, Ncam, Trackmen, OptiTrack, Vicon (via bridge). Multiple cameras share one port using the camera ID nibble in byte 26. |
| `FREED_PLUS` | 6301 | Stype FreeD+ extended format: identical to FreeD D1 for the first 26 bytes, then adds a 12-byte extension with 32-bit high-precision angles (1/8 388 608 degree per unit). Falls back to standard D1 automatically when a 29-byte packet arrives. |
| `OSC` | 9000 | OSC 1.0 UDP. No external library required. Uses the address schema `/camera/{id}/pan`, `/tilt`, `/roll`, `/zoom`, `/focus`, `/x`, `/y`, `/z`. Angles in degrees; position in mm; zoom/focus in raw 0–65535 or normalised 0.0–1.0. |
| `VRPN` | — | VRPN tracker client (`vrpn_Tracker_Remote`). Quaternion pose converted to yaw/pitch/roll. Analogue channel 0 used as zoom. Optional — requires `-DBUILD_TRACKING_VRPN=ON` at CMake configure time. |

---

## Starting Position and Zeroing

### Transforms are absolute, not relative

Every incoming tracking packet **fully replaces** the previous transform. There is no accumulation. The injected value is:

```
injected = data.angle × axis_scale + axis_offset
```

This means:
- At `scale=1.0` and `offset=0`, a tracker that returns `pan=0` will always snap the layer back to neutral.
- Whatever the physical tracker sends, that is what the layer shows — immediately, every packet.

### Before the first packet arrives

The layer retains whatever transform it had before `TRACKING BIND` was called. If you
set `MIXER 1-1 PROJECTION 0 0 0 90` and then bind the tracker, the layer will hold
that position until the first tracking packet lands. Use this as your "waiting" pose.

### Setting where neutral (zero) is

Use `TRACKING OFFSET` to shift the tracker’s zero point without touching anything
else. You do **not** need to unbind/rebind. The change takes effect on the very next
packet:

```amcp
— The camera is resting at pan=+5° due to a mount offset.
— We want that physical position to show as 0° on screen.
TRACKING 1-1 OFFSET -5 0 0
```

### One-click zero with `TRACKING ZERO`

Instead of calculating the offset manually, send `TRACKING ZERO` while the camera is
pointing at the desired neutral position. The module reads the latest received data
and automatically sets the offsets so that the current physical pose maps exactly to
`(yaw=0, pitch=0, roll=0)` in 360 mode, or `(fill_translation=0, 0)` in 2D mode.

```amcp
— Point the camera at the intended centre-of-frame, then:
TRACKING 1-1 ZERO
```

The new offsets are visible immediately in `TRACKING INFO`. No rebind required.

### Summary: what requires a rebind?

| Change | Needs rebind? |
| :--- | :--- |
| Adjust neutral position | No — use `TRACKING OFFSET` or `TRACKING ZERO` |
| Flip or scale an axis | No — use `TRACKING SCALE` |
| Change zoom calibration range | No — use `TRACKING SCALE` (third parameter) |
| Change default FOV | No — use `TRACKING DEFAULT_FOV` |
| Switch between 360 and 2D mode | **Yes** — requires rebind |
| Switch to a different camera ID | **Yes** — requires rebind |
| Move to a different UDP port | **Yes** — requires rebind |

---

## AMCP Command Reference

All commands use the standard CasparCG channel-layer syntax: `TRACKING <ch>-<layer> <subcommand> [params]`.

---

### `TRACKING BIND`

Creates or replaces a tracking binding on the specified layer.

```
TRACKING <ch>-<layer> BIND <protocol>
    [PORT <port>]
    [HOST <host>]
    [CAMERA <id>]
    [MODE <2D|360>]
```

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `protocol` | — | `FREED`, `FREED_PLUS`, `OSC`, or `VRPN` (required) |
| `PORT` | 6301 | UDP port to listen on (FreeD/FreeD+/OSC) |
| `HOST` | — | VRPN server URL, e.g. `Tracker0@192.168.1.50` |
| `CAMERA` | 0 | Camera ID to accept. Use `-1` to accept all cameras on this port. |
| `MODE` | `360` | `360` injects into the equirectangular projection; `2D` injects into fill/scale/rotation |

**Examples:**
```amcp
TRACKING 1-1 BIND FREED PORT 6301 CAMERA 1 MODE 360
TRACKING 1-2 BIND FREED_PLUS PORT 6302 CAMERA 2 MODE 360
TRACKING 1-5 BIND OSC PORT 9100 CAMERA 0 MODE 2D
TRACKING 2-1 BIND VRPN HOST Tracker0@192.168.1.50 CAMERA 0 MODE 360
```

If a binding already exists on this channel/layer it is replaced and the old receiver's reference count is decremented.

---

### `TRACKING UNBIND`

Removes the binding from a channel/layer.

```
TRACKING <ch>-<layer> UNBIND
```

---

### `TRACKING OFFSET`

Sets a fixed angular offset (in degrees) added to the decoded tracking angles after scaling. Use this to trim the rest position of the camera without physically re-zeroing the tracker.

```
TRACKING <ch>-<layer> OFFSET <pan_deg> <tilt_deg> <roll_deg>
```

**Example** — compensate a 1.5° downward tilt at the mount:
```amcp
TRACKING 1-1 OFFSET 0 -1.5 0
```

---

### `TRACKING SCALE`

Sets per-axis scale factors and the zoom calibration range.

```
TRACKING <ch>-<layer> SCALE <pan_scale> <tilt_scale> <zoom_full_range>
```

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `pan_scale` | `1.0` | Multiplier applied to decoded pan before offset. Use `-1.0` to flip the axis (counter-tracking: layer moves opposite to camera). |
| `tilt_scale` | `1.0` | Multiplier applied to decoded tilt. Use `-1.0` to flip. |
| `zoom_full_range` | `65535` | The raw zoom value your lens/encoder sends when at its widest angle. Adjust this to calibrate zoom-to-FOV mapping. |

**Example** — flip pan for counter-tracking, calibrate zoom encoder range:
```amcp
TRACKING 1-1 SCALE -1.0 1.0 32000
```

---

### `TRACKING ZERO`

Captures the tracker’s current physical position as the new neutral / home position. Sets the pan, tilt, and roll offsets so that wherever the camera is pointing right now produces `(0, 0, 0)` injected into the layer.

```
TRACKING <ch>-<layer> ZERO
```

Returns `404` if no binding exists, or if no data has been received yet from the bound camera.

**Example workflow:**
```amcp
// 1. Bind the tracker
TRACKING 1-1 BIND FREED PORT 6301 CAMERA 1 MODE 360

// 2. Wait for data, then aim the camera at the desired start view
// 3. Capture that as home — no calculation needed
TRACKING 1-1 ZERO
```

---

### `TRACKING DEFAULT_FOV`

Sets the wide-end FOV (in degrees) used by the zoom lens formula. This is the FOV the lens produces when `zoom_raw == zoom_full_range`. Live — no rebind needed.

```
TRACKING <ch>-<layer> DEFAULT_FOV <degrees>
```

**Example** — a 12mm lens on Super35 has an ~85° horizontal FOV at its widest:
```amcp
TRACKING 1-1 DEFAULT_FOV 85
```

---

### `TRACKING INFO`

Returns the current binding configuration and the most recently received data for the bound camera.

```
TRACKING <ch>-<layer> INFO
```

**Response:**
```
201 TRACKING OK
PROTOCOL FREED
PORT 6301
CAMERA 1
MODE 360
PAN_SCALE 1.0
TILT_SCALE 1.0
PAN_OFFSET 0.0
TILT_OFFSET -1.5
ROLL_OFFSET 0.0
ZOOM_FULL_RANGE 65535.0
ZOOM_DEFAULT_FOV 90.0
LAST_PAN -0.023
LAST_TILT 12.451
LAST_ROLL 0.001
LAST_X 0.0
LAST_Y 1532.5
LAST_Z 4800.0
LAST_ZOOM 29450
LAST_FOCUS 0
```

---

### `TRACKING LIST`

Lists all active bindings across all channels and layers.

```
TRACKING LIST
```

**Response:**
```
200 TRACKING OK
1-1 PROTOCOL FREED PORT 6301 CAMERA 1 MODE 360
1-5 PROTOCOL OSC PORT 9100 CAMERA 0 MODE 2D

```

---

## Transform Modes

### 360° Mode (`MODE 360`)

Injects decoded angles and computed FOV directly into the `MIXER PROJECTION` system, which drives the GPU equirectangular shader. The binding is exactly equivalent to — and real-time-updates — the same parameters set by:

```amcp
MIXER 1-1 PROJECTION <yaw_deg> <pitch_deg> <roll_deg> <fov_deg>
```

| Tracking value | Struct field set |
| :--- | :--- |
| Pan (yaw) | `projection.yaw` (radians) |
| Tilt (pitch) | `projection.pitch` (radians) |
| Roll | `projection.roll` (radians) |
| Zoom → FOV | `projection.fov` (radians, via lens formula) |

The 360 layer must have an equirectangular video loaded and projection enabled. Projection is enabled automatically when `fov > 0`. Set `fov = 0` (or zoom to the maximum telephoto raw value) to disable it.

> **Curved-screen compensation** (`MIXER PROJECTION_CURVE`) is independent of the tracking binding. Set it once with the AMCP command; the tracking system will not touch it.

---

### 2D Mode (`MODE 2D`)

Injects tracking data into the standard layer fill/scale/rotation transforms, equivalent to simultaneously running:

```amcp
MIXER 1-1 FILL <pan> <tilt_inv> 1 1
MIXER 1-1 ROTATION <roll_deg>
MIXER 1-1 FILL 0 0 <scale> <scale>
```

| Tracking value | Struct field set |
| :--- | :--- |
| Pan | `fill_translation[0]` |
| Tilt | `fill_translation[1]` (inverted so camera-up maps to layer-up) |
| Roll | `angle` (radians) |
| Zoom → scale | `fill_scale[0]` and `fill_scale[1]` (uniform, via lens formula) |

Use `TRACKING SCALE` to tune how much pan/tilt movement shifts the layer. At the default `pan_scale = 1.0`, one radian of pan moves the layer by one full frame width — large. For subtle parallax effects a `pan_scale` of `0.05` to `0.2` is more typical.

---

## Zoom Calibration

FreeD zoom values are raw encoder counts with no standard unit. The module converts them to FOV using a **realistic lens formula**:

$$\text{fov} = 2 \arctan\!\left( \tan\!\left(\frac{\text{fov\_wide}}{2}\right) \cdot \frac{\text{zoom\_full\_range}}{\max(z, 1)} \right)$$

Where:
- **fov\_wide** — FOV at the widest angle (default 90°, i.e. the FOV when `zoom_raw == zoom_full_range`)
- **zoom\_full\_range** — raw encoder value representing the widest angle (set via `TRACKING SCALE`)
- **z** — current raw zoom value from the tracker

This formula mimics the angular behaviour of a real zoom lens: at `z = zoom_full_range` → FOV = fov\_wide; as z decreases → FOV widens; as z increases beyond full_range → FOV narrows.

### Optional Lookup Table (config file only)

For lenses with non-linear zoom encoders, a per-binding lookup table can be specified in `casparcg.config`. Values between entries are linearly interpolated:

```xml
<binding>
  <protocol>FREED</protocol>
  <port>6301</port>
  <zoom_lookup>
    <entry raw="0"     fov="90.0"/>
    <entry raw="8000"  fov="65.0"/>
    <entry raw="20000" fov="30.0"/>
    <entry raw="40000" fov="12.0"/>
    <entry raw="65535" fov="4.5"/>
  </zoom_lookup>
</binding>
```

---

## Axis Offsets and Scaling

The full transform chain applied to each decoded angle before injection:

```
decoded_angle_rad
    × axis_scale          (from TRACKING SCALE)
    + axis_offset_rad     (from TRACKING OFFSET, converted from degrees internally)
    = injected_angle_rad
```

### Tips

| Goal | Command |
| :--- | :--- |
| Flip pan (counter-tracking) | `TRACKING 1-1 SCALE -1.0 1.0 65535` |
| Trim tilt 2° upward | `TRACKING 1-1 OFFSET 0 2.0 0` |
| Halve pan sensitivity in 2D | `TRACKING 1-1 SCALE 0.5 0.5 65535` |
| Invert tilt axis | `TRACKING 1-1 SCALE 1.0 -1.0 65535` |

---

## Configuration File

Receivers can be pre-started when the server boots by adding a `<tracking>` block to `casparcg.config`. This is useful in unattended deployments where AMCP clients connect after the server is already running.

```xml
<configuration>
  ...
  <tracking>
    <!-- Pre-start a FreeD receiver on the standard port -->
    <receiver>
      <protocol>FREED</protocol>
      <port>6301</port>
    </receiver>

    <!-- Pre-start an OSC receiver on a custom port -->
    <receiver>
      <protocol>OSC</protocol>
      <port>9100</port>
    </receiver>

    <!-- Pre-start a VRPN connection -->
    <receiver>
      <protocol>VRPN</protocol>
      <host>Tracker0@192.168.1.50</host>
    </receiver>
  </tracking>
  ...
</configuration>
```

> **Note:** Only receivers are started from config. Channel/layer bindings are always established at runtime via `TRACKING BIND`. This mirrors the standard CasparCG design where consumers and producers are bound by AMCP clients after connect.

---

## Building with VRPN

VRPN support is off by default to keep the default build light. To enable it, pass the flag at CMake configure time:

```bat
cmake ..\src -G Ninja -DBUILD_TRACKING_VRPN=ON
```

CMake will automatically fetch VRPN from GitHub via `FetchContent` and compile the client libraries. No manual dependency installation is required.

Without this flag the `vrpn_receiver` class compiles but `start()` is a no-op that logs a message. All other protocols work regardless of this flag.

---

## Worked Examples

### Example 1 — Live 360° camera feed

A PTZ camera on a robotic head streams FreeD over UDP. We want the 360 layer to follow the camera exactly.

```amcp
// Load a 360 equirectangular clip on layer 1
PLAY 1-1 360_studio_loop

// Enable projection (any non-zero FOV activates it)
MIXER 1-1 PROJECTION 0 0 0 90

// Bind FreeD, camera ID 1, 360 mode (default)
TRACKING 1-1 BIND FREED PORT 6301 CAMERA 1

// Tilt is mounted 3° down — trim it
TRACKING 1-1 OFFSET 0 3.0 0
```

The layer now shows exactly what the locked-off camera sees, interactively, every time a FreeD packet arrives.

---

### Example 2 — Tracked lower third (2D counter-tracking)

A graphic lower third should appear fixed in screen space even as the camera pans. We bind it in 2D mode with pan and tilt axes flipped (counter-tracking).

```amcp
// Load a transparent HTML lower third on layer 10
PLAY 1-10 [HTML] http://localhost:8080/lower_third

// Bind with flipped axes so the graphic counter-moves with the camera
TRACKING 1-10 BIND FREED PORT 6301 CAMERA 1 MODE 2D
TRACKING 1-10 SCALE -0.15 -0.10 65535
```

A pan_scale of `-0.15` means the layer shifts 0.15 frame-widths per radian of pan — enough to cancel parallax from a moderate zoom.

---

### Example 3 — OSC from a custom motion-capture system

A bespoke body-tracking system sends OSC bundles to port 9200. Camera ID is always 0.

```amcp
TRACKING 1-3 BIND OSC PORT 9200 CAMERA 0 MODE 360
```

The sender must use the `/camera/0/pan`, `/camera/0/tilt`, etc. schema. The z message triggers the frame push, so it must always be sent last in each update bundle.

---

### Example 4 — Multiple cameras on one server

Two cameras simultaneously driving different layers:

```amcp
TRACKING 1-1 BIND FREED PORT 6301 CAMERA 1 MODE 360
TRACKING 1-3 BIND FREED PORT 6301 CAMERA 2 MODE 360
```

Both cameras share the same UDP port (standard FreeD multiplexing). The registry routes each packet to the correct layer based on the camera ID.

---

## Architecture Reference

### Source Files

```
src/modules/tracking/
├── camera_data.h               Normalised packet struct
├── tracker_binding.h           Per-layer config (mode, scales, camera_id, zoom params)
├── tracker_registry.h/.cpp     Singleton dispatcher + transform injection
├── receiver_manager.h/.cpp     Ref-counted receiver lifecycle
├── tracking_commands.h/.cpp    AMCP command implementations
├── tracking.h/.cpp             Module init/uninit + config-file loading
├── CMakeLists.txt
└── protocol/
    ├── receiver_base.h         Abstract receiver interface
    ├── freed_receiver.h/.cpp   FreeD D1 UDP
    ├── freed_plus_receiver.h/.cpp  Stype FreeD+ extended
    ├── osc_receiver.h/.cpp     OSC 1.0 UDP (no external lib)
    └── vrpn_receiver.h/.cpp    VRPN (optional)
```

### FreeD Packet Layout (D1, 29 bytes)

```
Byte  0      : Message type (0xD1)
Bytes  1– 3  : Pan   (24-bit signed, 1/32768 degree/unit)
Bytes  4– 6  : Tilt  (24-bit signed, 1/32768 degree/unit)
Bytes  7– 9  : Roll  (24-bit signed, 1/32768 degree/unit)
Bytes 10–13  : X position (32-bit signed, 1/64 mm/unit)
Bytes 14–17  : Y position (32-bit signed, 1/64 mm/unit)
Bytes 18–21  : Z position (32-bit signed, 1/64 mm/unit)
Bytes 22–23  : Zoom  (16-bit unsigned, vendor-raw)
Bytes 24–25  : Focus (16-bit unsigned, vendor-raw)
Byte  26     : User bits — high nibble = camera ID (0–15)
Bytes 27–28  : Checksum (sum of all 29 bytes & 0xFF == 0x40)
```

### OSC Address Schema

```
/camera/{id}/pan    f   degrees
/camera/{id}/tilt   f   degrees
/camera/{id}/roll   f   degrees
/camera/{id}/x      f   millimetres
/camera/{id}/y      f   millimetres
/camera/{id}/z      f   millimetres  ← frame is pushed when this arrives
/camera/{id}/zoom   f   0–65535 (or 0.0–1.0 normalised)
/camera/{id}/focus  f   0–65535 (or 0.0–1.0 normalised)
```

### VRPN Coordinate Conventions

VRPN uses a right-hand Y-up coordinate system. Quaternion components in the callback are `(qx, qy, qz, qw)` — note `qw` is at index `[3]`. The receiver extracts ZYX Euler angles matching CasparVP's yaw/pitch/roll convention:

- **Yaw** (Z): horizontal rotation, positive = right
- **Pitch** (Y): vertical rotation, positive = up
- **Roll** (X): tilt rotation, positive = clockwise

Position values from VRPN are in metres and are multiplied by 1000 to convert to millimetres before being stored in `camera_data`.
