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

The name is matched against every PortAudio device with at least one input channel. Get the list
from `INFO LTC` (§3) or `INFO PORTAUDIO`, which enumerate the same devices.

**The match is EXACT — not a partial match**, unlike the PortAudio *consumer*'s `<device>`, which
`casparcg.config` documents as "partial device name match". The same-looking element behaves
differently in the two places. Copy the name from `INFO LTC` verbatim, punctuation and case included.

**A name that matches nothing does not disable LTC and does not raise an error — it opens the
system DEFAULT input device instead.** Read §2 before relying on either the reply or `INFO LTC`'s
`device` field to tell you which input is open.

Leave `<device>` out entirely and the default input device is opened as well; what stops LTC being
used is the absence of a decodable signal, which shows as `source = System Clock`.

---

## 2. Switch device at runtime

```
LTC LOAD "Line In (Focusrite USB)"
```

| reply | what it actually means |
| :--- | :--- |
| `202 LTC LOAD OK` | **a** capture stream opened — **not necessarily the one you named** |
| `404 LTC LOAD ERROR` | no usable input device at all, or the stream failed to open |
| `400 ERROR` | no device name given |

**`202` does not mean your device was found.** An unmatched name is stored as-is, resolves to index
`-1`, and the input then falls back to `Pa_GetDefaultInputDevice()` — which usually opens fine, so
the reply is `202`. `404` appears only when *nothing* opens: no input device on the machine, the
chosen device reporting no input channels, or a stream-open failure.

**And `INFO LTC`'s `device` field echoes the name you asked for, not the device that was opened.**
A typo therefore comes back to you looking accepted: `202`, your misspelling reflected in `device`,
and `valid = false` — which reads exactly like a cable or level problem. It is not.

So there are **three** distinct failures, and the reply distinguishes none of them:

| symptom | actual cause |
| :--- | :--- |
| `202`, `device` shows your name, `valid = false` | either the signal is bad **or the name never matched** and you are listening to the default input. Compare your string character-for-character against `INFO LTC`'s `devices` list |
| `202`, `valid = false`, name definitely correct | now it is the signal: generator off, wrong input, or level too quiet or clipped |
| `404` | no input device is usable at all — a driver or hardware problem, not a name |

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
| `device` | **the name that was requested**, not the device actually opened — see §2. `Default` means none was ever set |
| `devices` | every input the machine offers, for `LTC LOAD` |

**`source` is the field to read first.** A server that fell back to `System Clock` keeps running and
keeps producing timestamps, so nothing downstream looks broken — it is simply no longer aligned to
the house. There is no error, because falling back is the designed behaviour.

---

## 4. Typical bring-up

1. `INFO LTC` — read `devices` to get the exact device name the machine reports.
2. `LTC LOAD "<that name>"` — expect `202`.
3. `INFO LTC` again — check **`source` is `LTC`** and `valid` is `true`.
4. If `valid` is `false`, **check the name before the cable.** Because an unmatched name silently
   opens the default input (§2), the first thing to rule out is a mismatch: compare what you sent
   against the `devices` list character-for-character. Only once it matches exactly is `valid =
   false` evidence about the signal — generator running, cable on the right input, level neither
   very quiet nor clipped.
5. Put the working device name into `casparcg.config` so it survives a restart — exactly as it
   appears in `devices`, since that match is exact too.

**A runtime `LTC LOAD` wins over the config for the rest of the session.** The configured device is
applied only when no device name has been set yet, so after an `LTC LOAD` the config value is not
re-read until the server restarts.

---

## 5. What is not covered

**No battery drives LTC.** Nothing in the harness reads `INFO LTC`, sets a device, or checks the
system-clock fallback, so this feature is documented and unmeasured. Recorded in
[`../features/ltc-timecode.md`](../features/ltc-timecode.md), which owns the state and any figures.

**Frame-rate handling is not documented here** because it is not settled in one place — LTC carries
its own frame rate and the channel has one, and what happens when they disagree is worth
establishing before it is written down as advice.
