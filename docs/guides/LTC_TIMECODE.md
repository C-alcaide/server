# LTC Timecode

> **State and measurements:** [`../features/ltc-timecode.md`](../features/ltc-timecode.md)
> **This document is how-to.** Per [`../README.md`](../README.md), measured figures live once in `features/`; a tolerance an operator acts on may appear here, the measurements behind it should not.

Reads **linear timecode** (LTC) from an audio input and makes it the server's clock, so recordings,
replay and cluster playback can be aligned to a house timecode generator instead of to wall time.

When no valid LTC is present the server falls back to the **system clock** and says so — that
fallback is the single most important thing to check when timecode looks wrong, because a server on
the system clock behaves normally and produces plausible timestamps.

---

## 1. Configure the input device

LTC arrives as audio, so the device is an audio *input*. Set it in `casparcg.config`:

```xml
<configuration>
    <ltc>
        <device>Line In (Focusrite USB)</device>
    </ltc>
</configuration>
```

The name is matched against the audio input devices the machine reports. Get the list with
`INFO PORTAUDIO`, or from `INFO LTC` (below), which reports the same set.

Leave `<device>` out and no LTC is read at all — the server runs on the system clock.

---

## 2. Switch device at runtime

```
LTC LOAD "Line In (Focusrite USB)"
```

| reply | meaning |
| :--- | :--- |
| `202 LTC LOAD OK` | the device was found and capture started |
| `404 LTC LOAD ERROR` | **no such device** — the name did not match anything |
| `400 ERROR` | no device name given |

**A `404` here means the name, not the signal.** The command only opens the device; it does not
wait for timecode to appear or validate it. A device that opens but carries no LTC returns `202`
and then reports `valid = false` in `INFO LTC`. Those are two different failures and only the
second one is about your cable.

---

## 3. Check what it is actually doing

```
INFO LTC
```

```xml
<ltc>
    <timecode>10:32:14:07</timecode>
    <valid>true</valid>
    <source>LTC</source>
    <device>Line In (Focusrite USB)</device>
    <devices>
        <device>Line In (Focusrite USB)</device>
        <device>Microphone (Realtek)</device>
    </devices>
</ltc>
```

| field | read it for |
| :--- | :--- |
| `timecode` | the current position as `HH:MM:SS:FF` |
| `valid` | **is the timecode being decoded?** `false` with a device set means the device opened but the signal is absent, silent, at the wrong level, or not LTC |
| `source` | **`LTC` or `System Clock`.** This is the field that tells you whether timecode is in use at all |
| `device` | which input is open |
| `devices` | every input the machine offers, for `LTC LOAD` |

**`source` is the field to read first.** A server that fell back to `System Clock` keeps running and
keeps producing timestamps, so nothing downstream looks broken — it is simply no longer aligned to
the house. There is no error, because falling back is the designed behaviour.

---

## 4. Typical bring-up

1. `INFO LTC` — read `devices` to get the exact device name the machine reports.
2. `LTC LOAD "<that name>"` — expect `202`.
3. `INFO LTC` again — check **`source` is `LTC`** and `valid` is `true`.
4. If `valid` is `false`: the device is open, so it is the signal. Check the generator is running,
   the cable is on the right input, and the level is sane — LTC decodes poorly when very quiet or
   clipped.
5. Put the working device name into `casparcg.config` so it survives a restart.

---

## 5. What is not covered

**No battery drives LTC.** Nothing in the harness reads `INFO LTC`, sets a device, or checks the
system-clock fallback, so this feature is documented and unmeasured. Recorded in
[`../features/ltc-timecode.md`](../features/ltc-timecode.md), which owns the state and any figures.

**Frame-rate handling is not documented here** because it is not settled in one place — LTC carries
its own frame rate and the channel has one, and what happens when they disagree is worth
establishing before it is written down as advice.
