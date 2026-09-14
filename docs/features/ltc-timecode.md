# LTC timecode

> **State:** shipped; measured since 2026-09-14
> **Modules:** `src/modules/ltc`
> **Commands:** 2 (`LTC LOAD`, `INFO LTC`)
> **Architecture:** none, deliberately — one input device and a clock fallback; the guide covers it completely
> **Guide:** [`../guides/LTC_TIMECODE.md`](../guides/LTC_TIMECODE.md)
> **Coverage:** `ltc` — 4/4 both mixers, including a real decode

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

**`cli.py ltc`, 4/4 on both mixers, 2026-09-14** — the module's first coverage, and it found a
real defect on its first run (§4.1).

| check | what it holds |
| :--- | :--- |
| `INFO LTC` answers and names the clock it is on | with no signal the honest answer is `System Clock` and `valid=false`; the server keeps producing a timecode so a stamped recording carries one, and says where it came from |
| an unknown device falls back **and the reply says so** | `LTC LOAD` on a name that does not exist answers **202**, which is correct, and `device`/`requested`/`fallback` now make the fallback visible |
| and a device that DOES exist is honoured | the companion — without it the check above passes for a server that can never honour a request |
| **a real LTC signal DECODES** | sent `10:11:12:00`, read back `10:11:12:08`, `valid=true`, `source=LTC` |

**The decode arm needs no timecode hardware.** LTC is generated by **libltc's own encoder**
(`core/ltc_fixture/ltcgen.c` drives the reference implementation rather than reimplementing
SMPTE 12M), played out through the PortAudio consumer into VB-Audio's `CABLE Input`, and read
back from `CABLE Output`. A hand-written encoder in the harness could be wrong in the same
direction as the decoder and agree with it — a fixture that cannot fail.

**It is also the only check here that reads a timecode off a wire.** The other three drive
device bookkeeping and would all pass on a server whose decoder did nothing, which is why the
arm reports its own absence rather than skipping when no virtual cable is installed.

**Still not measured:** drop-frame, and 24/30/60 fps — the fixture is 25 fps and the decoder is
created with a PAL-typical bit-period estimate (§5.4). Real hardware LTC, and jam-sync
behaviour on signal loss.

### 4.1 What the first run found

**`LTC LOAD` on any string at all answered 202 and then reported that string as the open
device.** `find_device_by_name` returns −1 for an unknown name, `start_unlocked()` then opens
`Pa_GetDefaultInputDevice()` and returns true, and `INFO LTC` published the *requested* name.
Measured on this box: a load of `no_such_device_xyz` was really listening to
*"Webcam 4 (NDI Webcam Audio)"*.

**The fallback is not the defect.** A missing audio interface must not take a playout server
down. Reporting the requested name as though it were the open one is, because it removes the
operator's only way to notice — they see a plausible timecode, off the system clock, from a
webcam microphone. `INFO LTC` now reports `device` (what is actually open), `requested`, and
`fallback`, and a warning is logged once at the fallback. The header comment on
`set_capture_device` said *"Returns true if found and set"*, which had been wrong since the
fallback was written.

**And the doc's own proposed check would have been wrong**, which is the more useful half of
this. §4 used to propose asserting **404** for an impossible device. The server answers 202 —
so the check would have been red against correct behaviour, and the natural response is to
weaken it. *Assert the behaviour the server should have, not the one a sentence in a doc
predicts.*

---

## 5. Known gaps

1. ~~**No coverage at all.**~~ **CLOSED 2026-09-14** — `cli.py ltc`, 4/4 both mixers, and the
   decode arm turned out to need no hardware after all: VB-Audio's virtual cable is installed
   on this box, so the signal loops out of the PortAudio consumer and back into the LTC input.
   The "hardware-free starting point" this line pointed at was three device-bookkeeping checks;
   there are four, and the fourth reads a timecode off a wire.
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
