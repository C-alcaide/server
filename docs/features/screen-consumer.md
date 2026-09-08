# Screen consumer — the fork's windowed output

> **State:** shipped
> **Module:** `src/modules/screen` — **2,963 lines** different from upstream in two files
> (`consumer/screen_consumer.cpp`, `consumer/screen.frag`)
> **Commands:** consumer name and its parameters, including **three the fork adds** — `MONITORING`, `TONE_MAP`, `PEAK_LUMINANCE` (§1b)
> **Architecture:** none, deliberately — the structural point is §1 below: this is the instrument most batteries measure through
> **Guide:** none — upstream owns the consumer's operation and it is discussed in context across six guides. **Not because the changes are internal**: §1b documents three fork-only `ADD` parameters that live nowhere else.
> **Coverage:** used as the capture surface by `playback-scaling`, `mixer-parity`,
> `consumer-view` and most picture batteries. `previz-interact` is the first battery that measures
> **the consumer itself** — its window's mouse and keyboard handling (§1c)

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

## 1c. The window is an input surface — since 2026-09-08

`<interactive>` (default `true`, or `NON_INTERACTIVE` on the `ADD` form) used to do one thing: hide
the cursor. It now also decides whether the window's mouse and keyboard reach the channel.

`win32_gl_window::WndProc` turns `WM_MOUSEMOVE`, the nine button messages, `WM_MOUSEWHEEL`,
`WM_MOUSELEAVE` (via `TrackMouseEvent`), `WM_KEYDOWN/UP` and `WM_CHAR` into a `core::input_event`
on a small queue, drained in the same `poll()` that runs the window pump — the render thread, with
the GL context current. `create_consumer` holds the channel as a **`std::weak_ptr`**, because
channel → output → consumer → channel would otherwise be a cycle.

Three details that are the fork's own and are easy to get wrong:

* **coordinates come from the message's `lParam`, never `GetCursorPos`.** That is what lets a
  battery drive the window with `PostMessage` and what stops a click landing on whatever window is
  on top. It is also load-bearing for the coverage: a sink reading the real cursor treats a posted
  message as a silent no-op, so `previz-interact` compiles that exact fault in to prove it can fail.
* **the picture rect is the LIVE one.** Normalisation uses `width_`/`height_` as maintained by
  `WM_SIZE`, not the size the window had at construction — the 2015 interaction API captured it
  once and was wrong after any resize.
* **the letterbox is subtracted, and points in the bars are rejected.** `calculate_aspect()`
  already computes the draw rect for `none`/`uniform`/`fill`/`uniform_to_fill`; the old API mapped
  the bars into `[0,1]` instead.

Consecutive moves coalesce at the source, with the queue capped at 256, so a fast mouse cannot
outrun a tick or grow the queue without bound.

**Both window implementations, since 2026-09-08.** `win32_gl_window` fills `raw_input` from
`lParam`; the SFML path fills the same struct from `sf::Event`. Everything after that point is
shared -- which is why the letterbox rejection and the live-rect normalisation exist once rather
than twice.

The SFML side needs three things Win32 gets for free, and each is one of the defects the
2013-2018 API shipped with:

* **the button order is translated, not cast.** SFML orders Left/Right/Middle and `input_event`
  orders Left/Middle/Right (CEF's), so `sfml_button` is a `switch`. The old code cast between the
  two enums and exchanged right with middle for five years.
* **the modifier mask is QUERIED.** An `sf::Event::MouseMoved` carries a position and nothing
  about what is held, so `sfml_modifiers` reads `sf::Keyboard::isKeyPressed` and
  `sf::Mouse::isButtonPressed`. The old code never set modifiers at all, which is why no in-page
  drag ever worked while every click did.
* **key codes are mapped to WINDOWS virtual keys.** `INPUT <ch> KEY DOWN <vk>` documents a numeric
  virtual key, CEF's `windows_key_code` is windows-style on every platform including Linux, and
  `previz_renderer::input` compares against `VK_LEFT`. Passing SFML's own enum through would make
  `KEY DOWN 37` mean the left arrow on Windows and `sf::Keyboard::B` on Linux. Unmapped keys
  report 0 and the event is dropped -- a partial table is honest, and a fallthrough passing the
  raw code would put a wrong keystroke into a page.

The three live in `sfml_input_helpers.inl`, included by the consumer AND by
`sfml_input_self_test.cpp` -- a standalone program that drives a real SFML window with XTest
injection on an X display. 21 checks, run by hand under WSL; its own header carries the build
command. XTest rather than `XSendEvent` because the modifier query reads the X server's real
pointer state, which a synthetic event does not update.

Where the events go is `docs/features/previz.md` §2.2 — today, previz. The dispatch point
(`video_channel::input`) tries the image mixer first and the stage second, so the same events reach
a layer's producer when previz is not active.

---

## 2. Verification

**As of 2026-09-08 there is one**, and it covers exactly one part: `previz-interact` measures the
window's input handling (§1c) on both mixers, including a `<interactive>false</interactive>` control
arm that must ignore the identical messages. Everything else about the consumer is still exercised
constantly and asserted about never.

**That remains the gap, and it is structural:** a battery capturing through the screen consumer
cannot use that same capture to prove the screen consumer correct. Breaking the circularity needs a
second, independent capture route — `consumer-view` or a DeckLink loop — compared against it. Note
that `previz-interact` escapes the circularity only because it reads the **control API**, not a
capture.

---

## 3. Known gaps

1. **No battery treats it as the subject.** See §2; this is the highest-value missing check for the
   module, because everything else's numbers depend on it.
2. **`screen.frag`'s colour path is not compared against the mixer shaders'**, though they implement
   overlapping arithmetic.
3. **Vulkan output is a separate module** (`vulkan_output`) with its own document; the relationship
   between the two is not written down anywhere.
