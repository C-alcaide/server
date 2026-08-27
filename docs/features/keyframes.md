# Keyframes — timeline animation of mixer state

> **State:** shipped, unmeasured
> **Modules:** `src/modules/keyframes`
> **Commands:** 8 fork-specific AMCP commands, registered by the module
> **Architecture:** none, deliberately — a tween table over the existing transform system; no structural decision to record
> **Guide:** [`../guides/KEYFRAMES.md`](../guides/KEYFRAMES.md)
> **Coverage:** **none**

Animates mixer state over time from a keyframe list, rather than one tween per command. Arm a
timeline, seek it, and the mixer follows — which is how a show cue with twenty simultaneous
parameter moves is expressed as one object instead of twenty `MIXER` commands with matching
durations.

Operator detail is in [`../guides/KEYFRAMES.md`](../guides/KEYFRAMES.md). This document is the
state and the coverage.

---

## 1. What is implemented today

Eight commands (`keyframe_commands.cpp`):

| command | purpose |
| :--- | :--- |
| `KEYFRAMES SET` | define the keyframe list |
| `KEYFRAMES PATCH` | modify it in place |
| `KEYFRAMES GET` | read it back |
| `KEYFRAMES CLEAR` | discard it |
| `KEYFRAMES ARM` / `DISARM` | enable/disable playback |
| `KEYFRAMES SEEK` | move the timeline position |
| `KEYFRAMES STATUS` | query armed state and position |

**The field vocabulary is the substance: 200 entries in `keyframe_fields.cpp`.** It covers the
geometry (`anchor_x`, `fill_x`, `fill_sx`, `clip_x`…), the basic mixer state (`opacity`,
`contrast`, `brightness`, `saturation`), and — critically — the fork's own grading and projection
fields. **This is the only place in the fork where the whole `image_transform` surface is
enumerated by name in one table**, which makes it the de facto index of what is animatable.

**It auto-enables the flags a field needs.** `keyframe_fields.cpp:430` sets
`enable_geometry_modifiers = true` when any geometry field is animated
(`keyframe_fields.h:64` documents the rule). Without that, animating a geometry field would set a
value the mixer's geometry gate never reads — the same class of silent no-op as the transform
allowlist trap.

---

## 2. How to drive it

```
KEYFRAMES SET 1-1 ...
KEYFRAMES ARM 1-1
KEYFRAMES SEEK 1-1 0
KEYFRAMES STATUS 1-1
```

The keyframe list syntax is in the operator guide and is not duplicated here.

---

## 3. Design decisions, and what they cost

**A parallel field vocabulary, not a reflection over `image_transform`.** 200 hand-written entries
mean a new `image_transform` field is *not* animatable until someone adds it here — a third place
to remember alongside the two mixer allowlists. `CLAUDE.md` already documents what forgetting one
of those costs: a command that returns `202` and changes nothing. This is the same trap with a
third door.

The upside is that the table can carry per-field metadata (defaults, and which enable flag to set)
that reflection could not supply, and the auto-enable in §1 is exactly that metadata earning its
keep.

**Arm/disarm separate from set.** A list can be built and inspected before it drives anything,
which matters when the alternative is discovering a bad cue live.

---

## 4. Verification — what is measured, and what is not

**Nothing.** No battery arms a timeline or checks that a field animates.

Two things make this a worse gap than the raw command count suggests:

1. **The 200-entry field table has no consistency check** against `image_transform`. A field
   removed or renamed in the struct leaves a dead entry here; a field added leaves a missing one,
   and the missing case is silent.

   **That check was run by hand on 2026-08-26** and the table is in good shape: of 71
   `image_transform` fields, 8 are absent and **7 of those are legitimately not animatable** —
   `blend_mask` (a texture), `grade_nodes` (a node graph), `geometry_override`, `is_key`, `is_mix`
   and `layer_depth` (modes and ordering, not continuous values), and `ocio` (a config selection).

   **The eighth is an asymmetry worth a look: `hue_curves` is absent while `curves` is present.**
   Both are curve data of the same shape, so either tone curves should not be animatable or hue
   curves should be. Not a defect — nothing breaks — but it is an inconsistency nobody chose, and
   it is the kind that becomes a support question.
2. **The auto-enable logic is the interesting part and is untested.** Animating a geometry field
   without `enable_geometry_modifiers` produces no movement and no error.

---

## 5. Known gaps

1. **No coverage.** §4.1 describes a cheap mechanical check worth having first.
2. **A third allowlist.** Any new `image_transform` field must be added here as well as to both
   mixers' `apply_transform_colour_values`; nothing enforces or reports that.
3. **No tween-shape verification.** The tween functions are shared with the `MIXER` commands'
   `[tween]` argument, which is itself untested fork-wide.

---

## 6. Related commits

Not traced; the module predates this document.

---

## 7. Diagrams

Not warranted. The interesting content is a 200-row table and a state machine with two states
(armed/disarmed) — neither benefits from a picture.
