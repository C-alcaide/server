# Remotewall and PortAudio

Two unrelated modules, documented together because each is small and neither justifies a file of
its own. Both are shipped with no coverage.

---

# Remotewall

> **State:** shipped, unmeasured
> **Modules:** `src/modules/remotewall` (producer, with `vendored/cloudxr` and `vendored/recvwall`)
> **Commands:** none — a producer named `remotewall`
> **Architecture:** none, deliberately — conventional network and audio-device plumbing
> **Guide:** [`../guides/REMOTEWALL_MODULE.md`](../guides/REMOTEWALL_MODULE.md), [`../guides/PORTAUDIO_MODULE.md`](../guides/PORTAUDIO_MODULE.md)
> **Coverage:** **none**

Receives video from a remote source as a layer, over CloudXR. Intended for a wall driven from
elsewhere on the network rather than from local media.

## What is implemented today

| piece | evidence |
| :--- | :--- |
| Producer named `remotewall` | `remotewall_producer.cpp:514` |
| CloudXR vendored | `vendored/cloudxr` |
| `recvwall` receiver vendored | `vendored/recvwall` |
| Config root | `configuration.remotewall` |
| Exportable Vulkan texture path | `remotewall_producer.cpp:324` |

**Two vendored dependencies** is the fact that matters most here: CloudXR is an NVIDIA SDK with its
own version and platform constraints, and `recvwall` is a second vendored component. A module with
vendored third-party code inside a fork is the hardest kind to keep current, and nothing records
which versions these are or when they were taken.

Operator detail: [`../guides/REMOTEWALL_MODULE.md`](../guides/REMOTEWALL_MODULE.md).
A native rewrite was planned and not done — [`../plans/REMOTEWALL_NATIVE_MODULE_PLAN.md`](../plans/REMOTEWALL_NATIVE_MODULE_PLAN.md),
which is in `plans/` because it describes intent rather than state.

## Verification

**Nothing.** No battery starts a remote source. This one genuinely needs a second endpoint, so the
realistic first check is narrower: that the producer refuses legibly when no source is reachable,
rather than hanging or presenting black indefinitely.

## Known gaps

1. **No coverage**, and no record of the vendored CloudXR/recvwall versions.
2. **The `eUndefined` layout fix** (`0f1c5fb38`) applies here too and was verified only on ProRes.
3. **A planned native rewrite** exists as a plan and has not happened; the plan should not be read
   as describing the module.

---

# PortAudio

> **State:** shipped, unmeasured
> **Modules:** `src/modules/portaudio`
> **Commands:** 1 (`INFO PORTAUDIO`)
> **Coverage:** **none**

Audio I/O through PortAudio, giving access to host audio APIs that the stock consumer set does not
reach — which on Windows means WASAPI and ASIO rather than only MME.

## What is implemented today

`INFO PORTAUDIO` enumerates the available APIs and devices, replying `201 INFO PORTAUDIO OK`
(`AMCPCommandsImpl.cpp:4595`). The enumeration is the useful part: it is how an operator discovers
what device names to configure, and the reply lists APIs by index (`API 0: MME`, `API 1: Win…`).

Detail: [`../guides/PORTAUDIO_MODULE.md`](../guides/PORTAUDIO_MODULE.md).

## Verification

**Nothing** — but this is the cheapest untested command in the fork to cover: `INFO PORTAUDIO`
takes no parameters, needs no hardware beyond whatever the machine has, and returns a `201` with a
device list. Asserting it replies `201` and names at least one API is a few lines and would at least
prove the module initialises.

## Known gaps

1. **No coverage**, despite being trivially coverable — see above.
2. **`INFO PORTAUDIO` is one of the nine remaining undocumented fork commands** as of 2026-08-26;
   this section is what closes it.
3. **No stated relationship to the stock audio consumers** — when to prefer PortAudio over the
   default path is not written down anywhere.

---

## Diagrams

Neither warrants one. Remotewall is a single network path; PortAudio is an enumeration and a device
open.
