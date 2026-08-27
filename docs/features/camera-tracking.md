# Camera tracking

> **State:** shipped, unmeasured
> **Modules:** `src/modules/tracking`
> **Commands:** **18** fork-specific AMCP commands, all registered by the module itself
> **Architecture:** [`../architecture/CAMERA_TRACKING_TRANSFORM.md`](../architecture/CAMERA_TRACKING_TRANSFORM.md)
> **Guide:** [`../guides/CAMERA_TRACKING.md`](../guides/CAMERA_TRACKING.md)
> **Coverage:** **none**

Receives live camera position, rotation and lens data from a tracking system and drives a
channel's projection from it, so rendered content stays correct as the camera moves. Five wire
protocols, a lens-profile model, and timecode alignment via LTC.

Operator detail is in [`../guides/CAMERA_TRACKING.md`](../guides/CAMERA_TRACKING.md). This
document is the state, the command inventory, and what is measured.

---

## 1. What is implemented today

**Five receiver protocols** (`src/modules/tracking/protocol/`):

| protocol | file | note |
| :--- | :--- | :--- |
| FreeD | `freed_receiver.cpp` | the industry baseline |
| FreeD+ | `freed_plus_receiver.cpp` | extended variant |
| OpenTrackIO | `opentrackio_receiver.cpp` | the newer open standard |
| PSN (PosiStageNet) | `psn_receiver.cpp` | with a vendored library in `vendor/psn` |
| OSC | `osc_receiver.cpp` | generic, for anything that can send OSC |
| VRPN | `vrpn_receiver.cpp` | six, if VRPN is counted separately |

**Eighteen commands**, which is the second-largest fork-specific command family after the grading
chain and larger than PREVIZ:

```
TRACKING BIND            TRACKING UNBIND         TRACKING LIST         TRACKING INFO
TRACKING TARGET_CAMERA   TRACKING TARGET_MAP     TRACKING OFFSET       TRACKING ZERO
TRACKING SCALE           TRACKING POSITION_SCALE TRACKING DELAY        TRACKING GENLOCK
TRACKING LENS            TRACKING ZOOM_LUT       TRACKING DEFAULT_FOV  TRACKING NODAL
TRACKING DOF             TRACKING WORLDALIGN
```

Grouped by what they are for: **binding** a receiver to a channel (`BIND`/`UNBIND`/`LIST`/`INFO`),
**aiming** it (`TARGET_CAMERA`/`TARGET_MAP`), **aligning** the coordinate system
(`OFFSET`/`ZERO`/`SCALE`/`POSITION_SCALE`/`WORLDALIGN`), **timing** it
(`DELAY`/`GENLOCK`), and **lens modelling** (`LENS`/`ZOOM_LUT`/`DEFAULT_FOV`/`NODAL`/`DOF`).

**It consumes LTC** for timecode alignment — `tracker_registry.cpp:624` calls
`get_timecode_anchor`, and `TRACKING INFO` reports `LTC_VALID` and `LTC_TC`
(`tracking_commands.cpp:360-361`). That coupling is documented in
[`ltc-timecode.md`](ltc-timecode.md) from the other side.

---

## 2. How to drive it

```
TRACKING LIST
TRACKING BIND freed 1
TRACKING TARGET_CAMERA 1 main
TRACKING ZERO 1
TRACKING DELAY 1 2
TRACKING INFO 1
```

Config root is `configuration.tracking`.

Full parameter lists are in the operator guide. **They are not repeated here on purpose** — a
second copy of eighteen parameter lists is a second thing to keep true, and this document's job is
the state.

---

## 3. Design decisions, and what they cost

**Alignment is decomposed into five separate commands** (`OFFSET`, `ZERO`, `SCALE`,
`POSITION_SCALE`, `WORLDALIGN`) rather than one transform. That makes each step recallable and
scriptable during a venue setup, which is how tracking is actually commissioned; the cost is that
the order they compose in is not obvious from the command names and is not documented anywhere —
see §5.

**A lens profile is a first-class object** (`lens_profile.cpp`), with `ZOOM_LUT` for measured
zoom→FOV curves. Real lenses are not linear in zoom, and a LUT is the honest way to model that
rather than a polynomial fit.

**Timecode alignment is optional and silent when absent** — the same fallback shape as LTC itself.
A tracking session with no LTC works and produces plausible-looking data.

---

## 4. Verification — what is measured, and what is not

**Nothing.** No battery in the harness drives tracking, and there is no fixture for any of the five
protocols.

This is the largest untested command surface in the fork — 18 commands — and it is the one where
absence of coverage is most defensible: the inputs are live network packets from hardware, and a
meaningful end-to-end check needs a tracking source. But *some* of it is testable without hardware,
and that is the gap worth naming:

1. **The receivers are packet parsers.** FreeD is a fixed-size binary datagram with a documented
   layout; feeding a synthesised packet and asserting the decoded position is a unit test needing
   no hardware at all.
2. **The alignment maths is pure.** `OFFSET`/`ZERO`/`SCALE`/`WORLDALIGN` compose into a transform
   that can be checked against hand-computed values.
3. **`ZOOM_LUT` interpolation** is a table lookup; endpoints and midpoints are checkable.

None of those needs a camera. All three are absent.

---

## 5. Known gaps

1. **No coverage of any kind**; §4 lists three things testable without hardware.
2. **The composition order of the five alignment commands is undocumented.** With `image_transform`
   composition the fork has already been bitten twice by field-by-field merge rules
   (`per_channel_levels`, `blend_mask`); an 18-command transform stack with no written order is the
   same hazard with more surface.
3. **`POSITION_SCALE`, `WORLDALIGN` and `ZOOM_LUT` appear in no document at all**, including the
   operator guide — they were found by reading registrations for this document.
4. **Five protocols, one implementation each, none with a fixture.** A regression in any receiver
   would be invisible until a shoot.

---

## 6. Related commits

Not traced; the module predates this document.

---

## 7. Diagrams

**Owed.** This is the clearest case in the fork for an *order* diagram: five alignment commands, a
lens model and a delay all compose into one camera transform, and §5.2 records that the order is
undocumented. Operator-facing — a rendered PNG, and drawing it would likely surface the same
ordering question the gap names.
