# Screen consumer — the fork's windowed output

> **State:** shipped
> **Module:** `src/modules/screen` — **2,963 lines** different from upstream in two files
> (`consumer/screen_consumer.cpp`, `consumer/screen.frag`)
> **Commands:** consumer name and its parameters, including **three the fork adds** — `MONITORING`, `TONE_MAP`, `PEAK_LUMINANCE` (§1b)
> **Architecture:** none, deliberately — the structural point is §1 below: this is the instrument most batteries measure through
> **Guide:** none — upstream owns the consumer's operation and it is discussed in context across six guides. **Not because the changes are internal**: §1b documents three fork-only `ADD` parameters that live nowhere else.
> **Coverage:** used as the capture surface by `playback-scaling`, `mixer-parity`,
> `consumer-view` and most picture batteries

Two files, three thousand changed lines — one of the heaviest per-file divergences in the tree, and
it had no entry in this folder because the module is not fork-only.

---

## 1. Why it matters more than a preview window

**The screen consumer is the instrument most batteries measure through.** `mixer-parity`,
`playback-scaling` and the channel-count ceilings all capture from it, so a defect here does not
merely look wrong on a monitor — it contaminates numbers attributed to the mixer, the producer or
the decoder. That is the reason this document exists rather than a line in an operator guide.

`screen.frag` carries its own colour handling, so it is a **second place** where the fork's colour
decisions are implemented — the first being the two mixer shaders. Three implementations of related
arithmetic is exactly the shape that produced the ICVFX and HAP Q defects.

---

## 1b. Three `ADD` parameters upstream does not have

Checked mechanically against `server-upstream` and documented nowhere until 2026-08-27. This file
previously said the fork's changes here were *internal*; these three are not.

```
ADD 1 SCREEN 1 MONITORING
ADD 1 SCREEN 1 TONE_MAP aces_rrt PEAK_LUMINANCE 600
```

| parameter | meaning |
| :--- | :--- |
| `MONITORING` | a **convenience preset**, not a mode: borderless, always-on-top, hidden from the taskbar, close-proof, no focus steal and no cursor. Equivalent to setting six flags by hand, which is what it exists to avoid |
| `TONE_MAP <op>` | `reinhard`, `aces_filmic`, `aces_rrt` or `hlg_ootf`, case-insensitive. **An unrecognised value leaves tone mapping unchanged** rather than erroring |
| `PEAK_LUMINANCE <nits>` | default `1000`. Feeds the **`hlg_ootf` operator only** — it reaches one shader branch, so it does nothing with the other three tone-map operators or with none |

`MONITORING` is the one worth knowing about: a confidence monitor that cannot be closed, cannot
steal focus and shows no cursor is otherwise six parameters, and nothing pointed at it.

---

## 2. Verification

There is no battery that measures the screen consumer *as the thing under test*. It is exercised
constantly and asserted about never.

**That is the gap, and it is structural:** a battery capturing through the screen consumer cannot
use that same capture to prove the screen consumer correct. Breaking the circularity needs a second,
independent capture route — `consumer-view` or a DeckLink loop — compared against it.

---

## 3. Known gaps

1. **No battery treats it as the subject.** See §2; this is the highest-value missing check for the
   module, because everything else's numbers depend on it.
2. **`screen.frag`'s colour path is not compared against the mixer shaders'**, though they implement
   overlapping arithmetic.
3. **Vulkan output is a separate module** (`vulkan_output`) with its own document; the relationship
   between the two is not written down anywhere.
