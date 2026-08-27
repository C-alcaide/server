# Vulkan output consumer — direct display output

> **State:** partial
> **Modules:** `src/modules/vulkan_output`
> **Commands:** 1 (`INFO VULKAN_OUTPUT`)
> **Coverage:** `consumer-view`, `vulkan-output-signalling`

Presents a channel straight to a display through Vulkan, bypassing the window compositor, with
control over the HDR signalling, the colour volume and the sync behaviour. Intended for LED
processors and projectors that need a specific signal rather than a desktop window.

Detail is in [`../architecture/VULKAN_OUTPUT.md`](../architecture/VULKAN_OUTPUT.md) (98 KB, the
largest document in the tree). This document is the state and the coverage gap.

---

## 1. What is implemented today

Configuration surface, read from `config_` in `vulkan_output_consumer.cpp`:

| group | options |
| :--- | :--- |
| Display selection | `display_name`, `gpu_index`, `dest_x`, `dest_y` |
| Signalling | `eotf`, `gamut`, `display_peak_luminance` |
| EDID | `edid_emulation`, `edid_auto_hdr` |
| Sync | `gsync_enabled`, `delay_frames`, `delay_ms`, `buffer_depth` |
| Misc | `display_blanker` |

**NVAPI is used for the parts Vulkan cannot reach** (`nvapi_helpers.h`): G-Sync status
(`get_sync_status`, line 468) and EDID emulation (line 586). Both are guarded on `nvapi_ &&
is_available()`, so the consumer runs without NVAPI and quietly does less.

`INFO VULKAN_OUTPUT` reports what was resolved — the first thing to check when the output looks
wrong, because most of the options above degrade silently when unavailable.

---

## 2. Design decisions, and what they cost

**EDID emulation instead of trusting the sink.** A projector or LED processor often reports an EDID
that does not describe what it actually wants; emulating one makes the driver present the signal the
device needs. The cost is a hard NVIDIA dependency for that feature and behaviour that varies with
the driver.

**Signalling is configured, not inferred.** `eotf`, `gamut` and `display_peak_luminance` are stated
rather than derived from the channel's colour space, because the two are genuinely independent — a
BT.2020 PQ channel may be feeding a device that wants something else. The cost is that a
misconfiguration is invisible in the picture: the pixels are right and the metadata is wrong.

---

## 3. Verification

| what | battery |
| :--- | :--- |
| Picture through the consumer | `consumer-view` |
| **What it signals** — transfer, gamut, HDR metadata, the actual surface | `vulkan-output-signalling` |

**This was the fork's highest-value missing battery until 2026-08-27, and the diagnosis was wrong.**
It had been recorded since 2026-08-17 as "no test drives the metadata". The real obstacle was that
**nothing could observe the metadata**:

* `INFO VULKAN_OUTPUT` enumerates displays and says nothing about colour;
* `state()` reported presentation, sync and frame counts and not one signalling field;
* the harness's OSC listener discarded every address but frame, fps and producer.

Three walls, none of them a test. `state()` now publishes transfer, gamut, `gamut-inert`, `hw-hdr`,
MaxCLL/MaxFALL, the mastering-luminance pair, and **the surface format and colour space the
swapchain actually got** — the last pair being the honest answer to "what is this output
signalling", because `pick_surface_format` walks a preference list and falls back.

### What it measured on its first run

| | requested | delivered |
| :--- | :--- | :--- |
| transfer | `pq` | `transfer=pq` reported |
| gamut | `bt2020` | **inert** — see below |
| surface | an HDR10 surface | **`bgra8` / `srgb_nonlinear`** |
| `hw-hdr` | — | `false` |

**HDR10 requested, 8-bit sRGB delivered.** Windows 10 build 19045 has no `VK_KHR_display`, so the
consumer takes the fullscreen-exclusive path where no HDR surface is offered. This also explains the
two `vulkan_out` matrix cases sitting at **18.7 dB against a 40 dB gate** at PQ and HLG while SDR
passes: the output is 8-bit sRGB where the check's model expects PQ-encoded.

**`<gamut>` is read, accepted and ignored.** The per-consumer colour conversion pass is disabled
behind a literal `if (false && …)`, so the framebuffer carries the *channel's* colour space. The
consumer had warned about this at startup for some time; the warning was being handed to a
verbosity-gated log callback and dropped, so nobody had read it. `gamut-inert` now reports it from
the same predicate the warning uses, so the two cannot drift.

**The battery gates what is universal and names what is local.** Gated: the config round trip, and
internal contradictions such as `hw-hdr true` on an SDR surface or an HDR colour space on an 8-bit
format — wrong on any platform. Named rather than failed: the HDR degradation and the inert gamut,
because both are the current state of this platform and this code, and a check that can never pass
here stops being read.

**Still uncovered:** whether the signalled metadata *reaches a display*. That needs NvAPI read-back
or a capture device, and nothing here has either — so `nvapi`, `UHDA` and `edid` remain untested.
What is asserted is what the server says it is sending, which is the half that was previously
unobservable.

---

## 4. Known gaps

1. ~~No metadata coverage at all~~ — **closed**, see §3. What replaced it is narrower and worth
   stating precisely: the battery asserts what the **server says** it is sending. Whether that
   reaches a display is still unmeasured and needs NvAPI read-back or a capture device.
2. **`nvapi`, `UHDA` and `edid` are still untested by anything.** `nvapi_helpers` has genuine
   read-back — `read_edid`, `get_sync_status`, and `set_uhda_hdr` reads back what the driver holds
   — so the capability exists and nothing calls it from a test. That is the next battery here.
3. **HDR cannot be reached on this platform at all.** No `VK_KHR_display` on Windows 10 build
   19045, so every HDR request degrades to an 8-bit sRGB surface. So the HDR *code paths* —
   `hw-hdr`, the ST.2086 block, the scRGB surface — have never executed here, and the battery
   names that rather than pretending to cover it.
4. **A latent double gamut mapping on the hardware-HDR path.** The consumer presents an scRGB
   (BT.709-primaries, linear) surface and lets the display engine map it to BT.2020; with a
   BT.2020 channel those primaries are mapped twice. It warns at startup and is unreachable here
   because `hw-hdr` is false — it will bite the first time it is true on a non-BT.709 channel.
5. **`consumer-view` covers the picture only**, on one display configuration.
6. **`delay_frames` / `delay_ms` / `buffer_depth` interact** and their combined behaviour is not
   documented in one place.

---

## 5. Related commits

| commit | why it matters |
| :--- | :--- |
| `0f1c5fb38` | exportable-texture layout; this consumer imports CUDA external memory on its LRU cache path |
| `b1da34a34` | added the interop lock to this consumer's evict-then-import path, which recurs for the life of the consumer |

---

## 6. Diagrams

Deferred. The signalling chain is worth a picture, but drawing one now would illustrate metadata
behaviour that **nothing verifies** — and per this folder's rule a diagram is a claim like any
other. Worth drawing once §3's battery exists.
