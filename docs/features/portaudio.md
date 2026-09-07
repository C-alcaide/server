# PortAudio — host audio API access

> **State:** shipped, unmeasured
> **Modules:** `src/modules/portaudio`
> **Commands:** 1 (`INFO PORTAUDIO`, fork-only)
> **Architecture:** none, deliberately — an enumeration and a device open
> **Guide:** [`../guides/PORTAUDIO_MODULE.md`](../guides/PORTAUDIO_MODULE.md)
> **Coverage:** **none**

Audio I/O through PortAudio, giving access to host audio APIs that the stock consumer set does not
reach — which on Windows means WASAPI and ASIO rather than only MME.

---

## 0. ASIO — built in since 2026-09-07, and it was not before

**This sentence used to be false in the only way that matters.** The module has always carried
ASIO code — `API=ASIO`, `host_api_preference::asio`, and a whole `shared_portaudio_capture` for
ASIO's one-stream-per-device rule — but the build never compiled PortAudio's ASIO host API, so
`Pa_HostApiTypeIdToHostApiIndex(paASIO)` returned −1 and every one of those paths was unreachable.
`API=ASIO` resolved to device −1. Measured on this box: `INFO PORTAUDIO` returned MME,
DirectSound and WASAPI, and **zero** ASIO.

The reason was licensing, and it expired. Steinberg dual-licensed the ASIO SDK under **GPLv3** on
**2025-10-15** (SDK 2.3.4); before that it was proprietary-only and no GPLv3 project could ship a
build linking it — the same reason Audacity has never shipped ASIO. The SDK's own `LICENSE.txt`
carries the dual grant, so a GPLv3 project may now both link and distribute it.

`src/modules/portaudio/CMakeLists.txt` therefore fetches the SDK by pinned URL and SHA-256 and
enables `PA_USE_ASIO` by default on Windows. `-DENABLE_ASIO=OFF` builds without it;
`-DASIOSDK_ROOT_DIR=<path>` uses a local copy instead of downloading.

**What it buys, measured against the same hardware — and it is not latency:**

| | channels | reported latency |
| :--- | ---: | ---: |
| ASIO `Blackmagic Audio` | **8**, one device | 0.0200 |
| WASAPI `DeckLink 8K Pro (1–4)` | 2 each, four devices | 0.0030 |

The gain is **one 8-channel device instead of four stereo pairs**. The two latency figures are not
comparable and must not be quoted as a regression: ASIO's is the driver's currently configured
buffer, set in Blackmagic's own ASIO control panel, while WASAPI's is its shared-mode default.

Two build traps, both already solved elsewhere in this tree for `openfx_host`: the SDK needs the
source root on its include path, because the global `/FI` is a relative path; and it needs
`/UUNICODE /U_UNICODE`, because `asiolist.cpp` calls the TCHAR registry APIs with `char*`.

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
