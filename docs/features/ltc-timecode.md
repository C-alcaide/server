# LTC timecode

> **State:** shipped, unmeasured
> **Modules:** `src/modules/ltc`
> **Commands:** 2 (`LTC LOAD`, `INFO LTC`)
> **Architecture:** none, deliberately — one input device and a clock fallback; the guide covers it completely
> **Guide:** [`../guides/LTC_TIMECODE.md`](../guides/LTC_TIMECODE.md)
> **Coverage:** **none**

Reads linear timecode from an audio input and makes it available process-wide as a clock, so
recordings can be stamped with house time and tracking data can be aligned to it. Falls back to
the system clock when no valid LTC is present, and says which it is using.

---

## 1. What is implemented today

| piece | evidence |
| :--- | :--- |
| A process-wide singleton, `caspar::ltc::LTCInput::instance()` | `ltc_input.h` |
| Capture-device selection at runtime | `set_capture_device(const std::string&)` returning false when not found |
| Validity flag, so a consumer can tell real LTC from the fallback | `is_valid()`, `is_using_system_clock()` |
| Frame-accurate anchor for a given fps | `get_timecode_anchor(int fps, ...)` |
| Timecode as a string, and as a frame number | `get_current_timecode_string()`, `get_current_frame_number(fps)` |

**Three consumers depend on it, which is what makes it more than a readout:**

| consumer | use | reference |
| :--- | :--- | :--- |
| FFmpeg consumer | stamps recordings with the current timecode | `ffmpeg_consumer.cpp:1911` |
| Camera tracking | aligns tracking samples to a timecode anchor | `tracker_registry.cpp:624` |
| Tracking query | reports `LTC_VALID` and `LTC_TC` | `tracking_commands.cpp:360-361` |

**A hardcoded frame rate worth knowing about.** `ffmpeg_consumer.cpp:1483` calls
`get_current_frame_number(25)` with the comment *"assuming 25?"* — so on a channel that is not
25p the frame number that reaches that call site is computed against the wrong rate. The timecode
*string* path does not have this problem. Recorded here because the question mark is in the source
and has not been answered.

---

## 2. How to drive it

Config — the device may be named, and `INFO LTC` reports what was resolved:

```xml
<ltc>
    <device>Line In (Realtek Audio)</device>
</ltc>
```

Runtime:

```
LTC LOAD "Line In (Realtek Audio)"
INFO LTC
```

`LTC LOAD` returns `202 LTC LOAD OK` when **a** stream opened and `404 LTC LOAD ERROR` only when
nothing could open at all.

> **This paragraph said `404` meant "the device name does not match", and called that "a real
> failure response rather than a silent no-op". It is the exact opposite.** An unmatched name is
> stored as typed, resolves to index `-1`, and `start_unlocked()` then falls back to
> `Pa_GetDefaultInputDevice()` — which usually opens, so the reply is `202`. `INFO LTC` then echoes
> **the name you asked for**, not the device that opened. So a typo returns success, shows your
> misspelling back to you, and reports `valid = false` — which reads as a cable or level fault.
> This is a silent no-op of the worst kind, and the doc was recommending it as a virtue.
> Corrected 2026-08-27; [`../guides/LTC_TIMECODE.md`](../guides/LTC_TIMECODE.md) §2 has the full
> three-way failure table.

`INFO LTC` reports `ltc.timecode` and `ltc.source`, where source is `LTC` or `System Clock`. **That
second field is the one to check first**: everything downstream still works when LTC is absent,
because the fallback is silent by design.

---

## 3. Design decisions, and what they cost

**A singleton, deliberately.** Timecode is a property of the house, not of a channel, and three
unrelated subsystems consume it. The cost is that there is exactly one capture device for the
whole process and no way to have two channels on different timecode sources.

**The system-clock fallback is silent.** A recording made with no LTC connected is stamped with
system time and looks identical to one stamped with house time. `is_using_system_clock()` exists
precisely so a caller can tell, and `INFO LTC` surfaces it — but nothing forces a check.

---

## 4. Verification — what is measured, and what is not

**Nothing.** No battery reads timecode, checks the fallback, or verifies a stamped recording
carries the timecode it should.

What a first battery could do without hardware, **corrected 2026-08-27 because the version written
here would have failed**: it proposed driving `LTC LOAD` with an impossible name and asserting
`404`. The server answers `202`, so that assertion is false and the natural response to a red test
would have been to weaken it.

The check worth writing instead asserts the **real** behaviour, which is also the more useful thing
to pin:

1. `LTC LOAD` an impossible name → expect **`202`**, and then `INFO LTC` reporting **that same
   impossible name** in `device`. That is the silent-fallback trap, and a test is the only thing
   that will keep it documented.
2. `INFO LTC` reports `System Clock` in `source` and a plausible timecode — the fallback.
3. `LTC LOAD` a name from `INFO LTC`'s own `devices` list → `202`, and `device` echoes it. Proves
   the reply cannot distinguish (1) from (3), which is the finding.

None of that needs an audio interface. Real LTC decode needs a signal generator and is a separate
problem.

---

## 5. Known gaps

1. **No coverage at all**; see §4 for a hardware-free starting point.
2. **`get_current_frame_number(25)` is hardcoded at `ffmpeg_consumer.cpp:1483`**, with a question
   mark in the comment. Either it should take the channel's rate or the call site should explain
   why 25 is right.
3. **One device per process**, by design, undocumented until now.
4. **Drop-frame and 24/30/60 rates are unverified** — the anchor API takes an fps, but nothing
   exercises anything but the common case.

---

## 6. Related commits

Not traced. The module predates this document; a commit list is only worth adding if each line
says why the commit matters, and inventing that from subject lines is the kind of claim this
folder exists to avoid.

---

## 7. Diagrams

Not warranted. One input, one singleton, three readers — no ordering subtlety, no two paths to the
same state. `CLAUDE.md`'s bar is deliberately high and this does not meet it.
