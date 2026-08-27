# PortAudio — host audio API access

> **State:** shipped, unmeasured
> **Modules:** `src/modules/portaudio`
> **Commands:** 1 (`INFO PORTAUDIO`, fork-only)
> **Architecture:** none, deliberately — an enumeration and a device open
> **Guide:** [`../guides/PORTAUDIO_MODULE.md`](../guides/PORTAUDIO_MODULE.md)
> **Coverage:** **none**

Audio I/O through PortAudio, giving access to host audio APIs that the stock consumer set does not
reach — which on Windows means WASAPI and ASIO rather than only MME.

> **Split out of `remotewall-and-portaudio.md` on 2026-08-27**, where this module was paired with
> `remotewall` because both were small. That premise still holds for this one; it stopped holding
> for the other.

---

## 1. What is implemented today

`INFO PORTAUDIO` enumerates the available APIs and devices, replying `201 INFO PORTAUDIO OK`
(`AMCPCommandsImpl.cpp:4595`). The enumeration is the useful part: it is how an operator discovers
what device names to configure, and the reply lists APIs by index (`API 0: MME`, `API 1: Win…`).

Operator detail: [`../guides/PORTAUDIO_MODULE.md`](../guides/PORTAUDIO_MODULE.md).

---

## 2. Verification

**Nothing** — and this is **the cheapest untested command in the fork to cover**. `INFO PORTAUDIO`
takes no parameters, needs no hardware beyond whatever the machine has, and returns a `201` with a
device list. Asserting it replies `201` and names at least one API is a few lines, and would prove
the module initialises at all.

Worth stating what that would *not* cover: whether any enumerated device can actually be opened, and
whether audio played through one is correct. Neither is reachable without knowing the rig's devices.

---

## 3. Known gaps

1. **No coverage**, despite being trivially coverable — §2.
2. **No stated relationship to the stock audio consumers.** When to prefer PortAudio over the
   default path is not written down anywhere, which is the question an operator actually arrives
   with.
3. **`INFO PORTAUDIO` was one of the undocumented fork commands** listed in the root `CLAUDE.md` as
   of 2026-08-26; this document and its guide are what close it.
