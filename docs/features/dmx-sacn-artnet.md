# DMX lighting — sACN and Art-Net

> **State:** shipped
> **Modules:** `src/modules/sacn`, `src/modules/artnet`, `src/modules/dmx_common`
> **Commands:** none — consumers configured on `ADD` or in the config
> **Coverage:** `dmx`

Samples regions of a channel's picture and sends the average colour of each region as DMX, so
lights follow the video. Two transports — sACN (E1.31) and Art-Net — over one shared geometry and
colour-averaging core.

Operator detail is in [`../guides/DMX_LIGHTING.md`](../guides/DMX_LIGHTING.md). This document is
the state, the shared-core structure, and what is measured.

---

## 1. What is implemented today

**`dmx_common` is the shared half**, and it is the part worth knowing about because it is why the
two transports behave identically (`fixture_geometry.h`):

| piece | purpose |
| :--- | :--- |
| `FixtureType` | fixture layout kinds |
| `box`, `rect`, `point` | fixture placement in normalised picture space |
| `compute_rect(box, index, count)` | splits one box into `count` fixture rectangles |
| `average_color(const uint8_t*, ...)` and `average_color(const core::const_frame&, rect)` | the sampling itself, from raw bytes or from a frame |
| `computed_fixture` | a resolved fixture with its rectangle |

Both `src/modules/artnet/util/fixture_calculation.h` and
`src/modules/sacn/util/fixture_calculation.h` include it — so a change to the averaging or the
geometry affects both transports at once. That is the design working, and also the risk: there is
no battery that runs the two transports against each other to confirm they still agree.

**sACN validates its parameters properly**, which is rarer in this fork than it should be
(`sacn_consumer.cpp`):

| parameter | range | on violation |
| :--- | :--- | :--- |
| `UNIVERSE` | 1 – 63999 | `user_error`, *"sACN universe must be between 1 and 63999"* |
| `PRIORITY` | 1 – 200 | `user_error`, *"sACN priority must be between 1 and 200"* |

Both are the E1.31 spec's own limits, and both produce a real error response rather than clamping
silently.

---

## 2. How to drive it

```
ADD 1 SACN UNIVERSE 1 PRIORITY 100 FIXTURES ...
```

`FIXTURES` is parsed by `get_fixtures_params`; the layout syntax is in
[`../guides/DMX_LIGHTING.md`](../guides/DMX_LIGHTING.md) and is not repeated here — a second copy
would be a second thing to keep true.

---

## 3. Design decisions, and what they cost

**One geometry and averaging core, two transports.** `dmx_common` exists so a fixture layout means
the same thing on sACN and Art-Net. The cost is a shared blast radius with nothing testing that the
two agree.

**Averaging over a rectangle, not sampling a point.** A point sample on video is noisy frame to
frame; an average over a fixture's rectangle is stable. The cost is that a small bright feature
inside a large fixture rectangle barely moves the output — which is a lighting-design property to
be aware of, not a defect.

**Two `average_color` overloads**, one taking raw bytes and one taking a `const_frame`. Convenient,
but it means there are two entry points to keep consistent if the pixel format ever changes.

---

## 4. Verification — what is measured, and what is not

| what | battery |
| :--- | :--- |
| DMX output values against expected fixture averages | `dmx` |

**What is not covered:**

- **sACN and Art-Net are never compared to each other.** They share `dmx_common`, so a defect in
  the shared core moves both identically and any check comparing them would pass. But a divergence
  in either transport's *packetisation* is exactly what such a check would catch, and it does not
  exist.
- **The parameter validation is unexercised.** `UNIVERSE 0` and `PRIORITY 255` should produce
  `user_error`; nothing asserts they do. That is a two-line battery addition and would be the
  cheapest coverage in this document.
- **Frame-rate behaviour** — how often DMX is emitted relative to the channel's tick — is not
  measured.

---

## 5. Known gaps

1. **No cross-transport check** between sACN and Art-Net over the shared core.
2. **Range validation untested**, despite being the one place in this feature that fails loudly.
3. **`dmx_common` has no document of its own** and is not a module an operator ever names — it is
   here rather than in its own file for that reason.

---

## 6. Related commits

Not traced; the modules predate this document.

---

## 7. Diagrams

The operator guide already carries the fixture-layout figures, which is the right place for them —
that is where someone placing fixtures is reading. Nothing further owed here.
