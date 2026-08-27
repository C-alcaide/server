# Camera tracking — the order the transform is composed

> **State and measurements:** [`../features/camera-tracking.md`](../features/camera-tracking.md)
> **Operator guide:** [`../guides/CAMERA_TRACKING.md`](../guides/CAMERA_TRACKING.md)
> **This document is why-it-is-shaped-this-way.** Operating instructions live in `guides/`, current state and figures in `features/`.

Eighteen `TRACKING` commands adjust one camera pose, and **the order they apply in was recorded
nowhere**. Operator documentation is in
[`../guides/CAMERA_TRACKING.md`](../guides/CAMERA_TRACKING.md); state is in
[`../features/camera-tracking.md`](../features/camera-tracking.md). This file is the sequence, read
from `tracker_registry.cpp::inject_transform`.

**Why it needs writing down:** several of these commands are alternative ways to express the same
correction, and two of them are *mutually exclusive paths* rather than composable adjustments. An
operator who sets both gets one silently ignored.

---

## 1. Two alignment paths, and they do not compose

This is the thing to understand before anything else.

| path | commands | how position is mapped |
| :--- | :--- | :--- |
| **legacy per-axis** | `POSITION_SCALE`, `OFFSET`, `SCALE`, `ZERO` | each axis scaled and offset independently: `pos · position_scale` |
| **rigid survey** | `WORLDALIGN` | one rigid transform: `world_m = align_scale · R · tracker_mm + align_t` |

`align_enable` selects between them (`tracker_binding.h:135-137`):

> *"Only applied in mode_previz. `align_enable = false` leaves the legacy per-axis offset /
> `position_scale` path untouched."*

So **`WORLDALIGN` replaces `POSITION_SCALE`; it does not refine it.** Setting both is not additive —
whichever path `align_enable` selects is the one that runs, and the other's values sit unused. The
rigid form also folds the millimetre→metre unit change into `align_scale`, which is why a
`position_scale` of `0.001` has no counterpart in the survey path.

`WORLDALIGN`'s rotation is **row-major, tracker→world**, solved client-side (Umeyama) from a survey
of known points. Orientation is applied by composing `R` with the tracker's own rotation rather than
by rotating the position twice.

---

## 2. The sequence, per frame

From `inject_transform`, in order:

```
1.  rotation      pan  = data.pan  · pan_scale  + pan_offset
                  tilt = data.tilt · tilt_scale + tilt_offset
                  roll = data.roll             + roll_offset

2.  lens profile  sample by (zoom, focus) -> fov_rad, distortion, nodal_forward_m
                  a profile's fov overrides compute_fov(zoom) when present

3.  nodal offset  shift the tracked POSITION by the lens-local nodal vector,
                  expressed in world space via the camera-frame basis
                  (at rest: forward +Z, right +X, up +Y)
                  lens.nodal_forward_m AUGMENTS a manual NODAL forward -- they add

4.  position      legacy:  offset_x = pos_x · position_scale        (mm -> m)
                  survey:  world_m  = align_scale · R · pos + align_t

5.  depth of field  map the decoded focus value to a bokeh blur radius
```

**Two orderings inside that are worth knowing:**

**Scale before offset, per axis.** `data.pan · pan_scale + pan_offset` — so an offset is in *output*
units, after scaling. Setting a scale after tuning an offset changes what that offset means.

**The nodal offset is applied to position, before the position mapping.** It is expressed in
millimetres at that point (`nf = nodal_fwd_m · 1000.0`), because the tracker's positions are
millimetres and the mapping to metres has not happened yet. A nodal correction and a
`position_scale` change therefore interact: the nodal vector is *not* rescaled by
`position_scale`, it is converted independently.

**The lens profile's nodal forward adds to the manual one.** `NODAL` does not override a profile —
if a lens profile carries a forward offset and you also set `NODAL`, the camera moves by the sum.
That is deliberate (a profile describes the lens, `NODAL` trims the rig) and is the most likely
source of a doubled parallax error.

---

## 3. What the commands do not do

**`ZERO` is not part of this sequence.** It captures a reference rather than adding a term, and lives
on the legacy path only.

**`GENLOCK` and `DELAY` are timing, not geometry.** They decide *when* a sample applies, not where
the camera is. A tracking system that looks spatially correct but lags the picture is a `DELAY`
problem and nothing in §2 will fix it.

**`ZOOM_LUT` feeds step 2, not step 1.** It maps a raw zoom reading to a real focal length so the
lens profile can be sampled meaningfully. A wrong `ZOOM_LUT` therefore presents as a wrong *FOV*,
which looks like a wrong `DEFAULT_FOV` and is not.

---

## 4. Modes change which of this runs

`mode_previz` is the only mode where `WORLDALIGN` applies at all. In 360 mode the position is routed
differently again — `tracker_binding.h:110` records that X and Y become
`projection.offset_x`/`offset_y`, i.e. **horizontal and vertical lens shift** rather than a camera
translation.

So the same `TRACKING OFFSET` means different things in different modes, and that is a property of
the mode rather than a bug. Establish the mode before tuning anything.

---

## 5. Unverified

**No battery drives any of the eighteen commands.** Nothing here is measured: not the sequence in
§2, not the mutual exclusion in §1, not the mode differences in §4. This document is read from the
source, and the source is the authority — where this file and `tracker_registry.cpp` disagree, the
code wins and the disagreement is a finding.

Operator-facing detail is in [`../guides/CAMERA_TRACKING.md`](../guides/CAMERA_TRACKING.md), which
deliberately does not invent a recommended sequence — §2 above is the order, read from the source.
