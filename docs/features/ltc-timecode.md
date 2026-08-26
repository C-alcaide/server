# LTC timecode

> **State:** shipped, unmeasured
> **Modules:** `src/modules/ltc`
> **Commands:** 2 (`LTC LOAD`, `INFO LTC`)
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

`LTC LOAD` returns `202 LTC LOAD OK` on success and **`404 LTC LOAD ERROR`** when the device name
does not match a capture device — a real failure response rather than a silent no-op, which is
better than most of this fork's runtime commands manage.

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

What a first battery could do without hardware: drive `LTC LOAD` with a name that cannot exist and
assert `404`; then assert `INFO LTC` reports `System Clock` and a plausible timecode. That covers
the failure path and the fallback — the two things most likely to be silently wrong — and needs no
audio interface. Real LTC decode needs a signal generator and is a separate problem.

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
