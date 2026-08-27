# Screen consumer — the fork's windowed output

> **State:** shipped
> **Module:** `src/modules/screen` — **2,963 lines** different from upstream in two files
> (`consumer/screen_consumer.cpp`, `consumer/screen.frag`)
> **Commands:** consumer name and its parameters
> **Architecture:** none, deliberately — the fork's changes are internal; the structural point (it is the instrument most batteries measure through) is in features/screen-consumer.md
> **Guide:** none, deliberately — Upstream owns the screen consumer's operation; this fork's changes are internal and the consumer is discussed in context across six guides. No dedicated guide.
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
