# Vulkan output consumer — direct display output

> **State:** partial
> **Modules:** `src/modules/vulkan_output`
> **Commands:** 1 (`INFO VULKAN_OUTPUT`)
> **Architecture:** [`../architecture/VULKAN_OUTPUT.md`](../architecture/VULKAN_OUTPUT.md)
> **Guide:** [`../guides/VULKAN_OUTPUT.md`](../guides/VULKAN_OUTPUT.md)
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
| Signalling | `transfer` (the one that acts), plus `eotf`, `gamut`, `display_peak_luminance` — **all three inert**, §below |
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

**Signalling is configured, not inferred.** These are stated rather than derived from the channel's
colour space, because the two are genuinely independent — a BT.2020 PQ channel may be feeding a
device that wants something else. The cost is that a misconfiguration is invisible in the picture:
the pixels are right and the metadata is wrong.

**But only `transfer` actually acts.** `eotf`, `gamut` and `display_peak_luminance` are all parsed
and then reach nothing on this consumer:

| setting | what happens to it |
| :--- | :--- |
| `transfer` | drives the surface format and colour space the swapchain requests — the live one |
| `gamut` | read and ignored, deliberately (§below), and it says so via a startup warning and `gamut-inert` |
| `eotf` | assigned in `config.cpp` and **read nowhere else at all** |
| `display_peak_luminance` | reaches only the disabled conversion pipeline. Its two use sites are inside `if (false && …)` and behind `if (color_pipeline_ && …)`, and `color_pipeline_` is only ever constructed in the first of those — so it is always null |

`<gamut>`'s inertness was found and documented; the other two were presented here as working until
2026-08-27. Only `gamut` announces itself, which is why it was the one that got noticed.

---

## 3. Verification

| what | battery |
| :--- | :--- |
| Picture through the consumer | `consumer-view` |
| **What it signals** — transfer, gamut, HDR metadata, the actual surface | `vulkan-output-signalling` |

**The obstacle here was never a missing test — it was that nothing could OBSERVE the metadata.**
Worth knowing before recording any gap as "no battery":

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

**Re-measured 2026-08-27 on a freshly rebuilt binary and unchanged:** `vulkan-output-signalling`
reports **3/3 cases consistent**, with PQ and HLG both degrading to `bgra8` / `srgb_nonlinear` and
`hw-hdr false`, MaxCLL 1000 / MaxFALL 400 round-tripping on all three. The two degradations are
named rather than failed, which is the behaviour the battery was built for.
| `hw-hdr` | — | `false` |

**HDR10 requested, 8-bit sRGB delivered.** Windows 10 build 19045 has no `VK_KHR_display`, so the
consumer takes the fullscreen-exclusive path where no HDR surface is offered. This also explains the
two `vulkan_out` matrix cases sitting at **18.7 dB against a 40 dB gate** at PQ and HLG while SDR
passes: the output is 8-bit sRGB where the check's model expects PQ-encoded.

**`<gamut>` is read and ignored, and that is deliberate — not a defect.** The per-consumer colour
conversion pass is disabled behind a literal `if (false && …)`, and the comment at
`vulkan_output_consumer.cpp:800` explains why: **the channel's colour space and transfer determine
every pixel value in the framebuffer**, so running this pass on top would **double-convert**.
Colour belongs to the mixer, and the consumer presents what the mixer produced.

The same comment carries a **FUTURE RE-ENABLEMENT PLAN**, whose key point is that a correct version
must take its input gamut from `channel_info` rather than from config — because config can disagree
with what the mixer actually emitted, which is exactly how a double conversion would arise.

So the only real problem was silence: an operator could set `<gamut>p3_d65</gamut>` for a P3 wall
and get no conversion and no warning, which looks identical to a conversion that worked. That is
fixed — the startup warning is surfaced and `gamut-inert` is reported every run, both from one
predicate so they cannot drift. **Enabling the pass as it stands would be a regression**, not a fix.

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
4. **A latent double gamut mapping on the hardware-HDR path, and it cannot be verified on this
   rig.** The consumer presents an scRGB (BT.709-primaries, linear) surface and lets the display
   engine map it to BT.2020; with a BT.2020 channel those primaries are mapped twice. It warns at
   startup. **Two independent blockers, not one:** Windows 10 build 19045 has no `VK_KHR_display`,
   *and* no HDR display is attached — so hardware HDR cannot be reached even after an OS upgrade.
   Deliberately **not** fixed speculatively: a fix that cannot be measured is a claim, and this
   session has already had two plausible fixes die on contact with a measurement.
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
