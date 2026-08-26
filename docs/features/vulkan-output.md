# Vulkan output consumer — direct display output

> **State:** partial
> **Modules:** `src/modules/vulkan_output`
> **Commands:** 1 (`INFO VULKAN_OUTPUT`)
> **Coverage:** `consumer-view`; **metadata is uncovered — see §3**

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

## 3. Verification — and the gap that matters

| what | battery |
| :--- | :--- |
| Picture through the consumer | `consumer-view` |

**Nothing measures what this consumer SIGNALS.** That is a known, tracked gap, recorded in
`CLAUDE.md` since 2026-08-17 and still open: no harness module references `nvapi`, `UHDA` or `edid`.
`signalling` drives DeckLink and the FFmpeg stream; this consumer is reachable only through
`cli.py run`, which checks pictures.

So a change to `eotf`, `gamut`, `display_peak_luminance`, `edid_emulation` or `edid_auto_hdr`
**currently has no test that can fail** — and §2 explains why that is worse here than elsewhere: a
metadata defect leaves the picture correct.

This is exactly the shape that produced the ICVFX defect (undriven surface, plausible-looking
output) and it is the highest-value missing battery in the fork, not merely an omission.

---

## 4. Known gaps

1. **No metadata coverage at all** — the tracked gap above. It needs a capture path that reads back
   what was signalled, not what was rendered.
2. **`consumer-view` covers the picture only**, on one display configuration.
3. **NVAPI-dependent features degrade silently.** `INFO VULKAN_OUTPUT` reports the resolved state,
   but nothing asserts that a requested EDID emulation actually happened.
4. **`delay_frames` / `delay_ms` / `buffer_depth` interact** and their combined behaviour is not
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
