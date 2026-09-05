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

**The field vocabulary is the substance: 193 animatable names.** It covers the geometry
(`anchor_x`, `fill_x`, `fill_sx`, `clip_x`…), the basic mixer state (`opacity`, `contrast`,
`brightness`, `saturation`), and — critically — the fork's own grading and projection fields.

**The table is no longer written here.** It is DERIVED from `core::fields::all()`, the transform
field registry, and the names are checked at startup against `FROZEN_KF_NAMES` — 193 of them — so a
rename or a dropped entry in the registry fails immediately rather than silently changing what a
saved timeline animates. That check earned its place on its first run: a `std::vector` holding the
generated name strings reallocated as it grew, invalidating every `c_str()` already handed to the
table, and the names came out wrong with no crash to say so.

**It auto-enables the flags a field needs.** `apply_kf_to_transform` in `keyframe_fields.cpp` sets
`enable_geometry_modifiers = true` when any geometry field is animated
(`keyframe_fields.h` documents the rule). Without that, animating a geometry field would set a
value the mixer's geometry gate never reads — the same class of silent no-op as the transform
allowlist trap.

---

## 2. How to drive it

```
KEYFRAMES 1-1 SET ...
KEYFRAMES 1-1 ARM
KEYFRAMES 1-1 SEEK 0
KEYFRAMES 1-1 STATUS
```

The keyframe list syntax is in the operator guide and is not duplicated here.

---

## 3. Design decisions, and what they cost

**A declared vocabulary, not a reflection over `image_transform` — but no longer a SECOND one.**
C++20 has no reflection, so the animatable names have to be written down somewhere; the names are
not derivable from the members in any case (`fill_x` is `fill_translation[0]`, `mid_r` is
`midtone[0]`). What changed is *where*: this used to be ~200 hand-written entries, a third list to
remember alongside both mixers' `apply_transform_colour_values`, and a new `image_transform` field
was not animatable until someone added it here. It is now a projection of
`core::fields::all()` — the same declaration the control API describes fields from and the mixers'
composition is checked against.

The cost is a layer of indirection between a keyframe name and the member it moves, and one real
constraint: the registry stores angles in radians, so the projection into keyframe names converts
to degrees on the way out and back on the way in, because that is what saved timelines contain.

**Arm/disarm separate from set.** A list can be built and inspected before it drives anything,
which matters when the alternative is discovering a bad cue live.

---

## 4. Verification — what is measured, and what is not

**Nothing.** No battery arms a timeline or checks that a field animates.

Two things make this a worse gap than the raw command count suggests:

1. **The field table now has one consistency check, and it is a name check rather than a
   coverage check.** `FROZEN_KF_NAMES` fails startup if the registry stops generating a name that
   saved timelines use. What it still cannot see is a field ADDED to `image_transform` and left out
   of the registry — the missing case, which stays silent.

   **A coverage check was run by hand on 2026-08-26** and the table is in good shape: of 71
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
2. **Still an allowlist, now shared.** A new `image_transform` field must be added to
   `core::fields` or it is animatable nowhere and describable nowhere — which is an improvement on
   three separate lists, but is not the same as being enforced. Both mixers still carry their own
   hand-written composition tables; the registry asserts agreement with them rather than replacing
   them.
3. **No tween-shape verification.** The tween functions are shared with the `MIXER` commands'
   `[tween]` argument, which is itself untested fork-wide.

---

## 6. Related commits

Not traced; the module predates this document.

---

## 7. Diagrams

Not warranted. The interesting content is a 193-name table and a state machine with two states
(armed/disarmed) — neither benefits from a picture.
