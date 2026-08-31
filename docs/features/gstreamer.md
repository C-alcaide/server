# GStreamer producer and consumer

> **State:** shipped
> **Modules:** `src/modules/gstreamer`
> **Commands:** 2 (`GST INFO`, `GST LIST`)
> **Architecture:** none, deliberately — its GPU route reuses vulkan::d3d11_import_bridge, covered by GPU_INTEROP_ARCHITECTURE.md
> **Guide:** [`../guides/GSTREAMER_GUIDE.md`](../guides/GSTREAMER_GUIDE.md)
> **Coverage:** `gstreamer`, `gst-consumer-cost`, `gst-dll-probe`, `coexistence`, `source-motion`, `raster-capacity` — the last two drive the producer as a generative load source; see below for the two fixture defects that made a ladder pass while a third of its frames were repeats

Plays and records through GStreamer pipelines, so anything GStreamer can source — SRT, RTSP,
NDI via a plugin, hardware capture — becomes a producer, and any GStreamer sink becomes a consumer.
The GPU route keeps hardware-decoded frames on the GPU into the mixer.

Operator detail is in [`../guides/GSTREAMER_GUIDE.md`](../guides/GSTREAMER_GUIDE.md), plus the
module's own `src/modules/gstreamer/README.md`. This document is the state and the decisions.

---

## The producer as a load source, and the two things that made it lie

Added 2026-08-31, for `raster-capacity --producer gst`: `videotestsrc` through the
GStreamer producer is real frames changing with no file, no codec and no disk, which is
what a capacity ladder needs and a flat colour cannot give. Two defects in the fixture had
to be fixed first, and both produced a **passing** ladder.

**The pixel format must be pinned, or the measurement is of `videoconvert`.** Without
`format=BGRA` in the caps, `videotestsrc` emits its native I420 and `videoconvert`
converts to what the appsink negotiated — RGBA, confirmed from the server log as
`pixel_format 2 (1 plane, 4 stride)`, `linesize 20000`. At four channels of 5000x3000p50
that conversion starved the producer on **~1800 of 5010 ticks per channel**: the channel
repeated the previous picture a third of the time while reporting **0 late frames and a
period of exactly 20.00 ms**. Pinning BGRA makes `videotestsrc` generate the format
directly and `videoconvert` a passthrough; 1080p50 x 4 then starves on **1 tick in
~20040**.

**Starvation is only visible in `gstreamer/starved`.** A log grep found nothing on the
run above — it is a guess at what the server prints. The producer already publishes the
direct symptom, and `gst_producer.cpp` says why `received` cannot substitute:

> Whether the pipeline is keeping ahead of the channel, which "frames received" cannot
> say: a source delivering 25 fps into a 50 fps channel still counts every frame it sent.
> `starved` counts ticks that found the queue empty and had to repeat the last picture —
> the direct symptom — and `queue_peak` says how much slack there was when it did not.

**What the fixed fixture then established.** Four channels, BGRA, `pattern=ball`, both
mixers: **1080p5000 holds** (1 repeated tick in ~20040); **3840x2160p50 starves at
21 %/channel** and **5000x3000p50 at 32 %/channel**. So on this rig a CPU-generated
source cannot saturate the bus above 1080p — producing the frames costs more than
uploading them — and any ladder above 1080p driven this way is measuring the fixture. The
evidence that it is the CPU side is the controlled change: removing one colour conversion
took 1080p from starving to clean.

**Not established:** whether four channels at 5000x3000p50 hold the tick with real
frames. Both generative sources fail for their own reasons — this one starves, and the ISF
generator induces a raster-independent ~1.4–5.4 % late-frame rate — so the question is
open and a GPU-side source (a GPU-direct decoded file, or `route://`) is what would answer
it.

## 1. What is implemented today

| piece | note |
| :--- | :--- |
| Producer, software route | any GStreamer pipeline as a layer |
| Producer, **GPU route** | D3D11 hardware decode imported into the mixer, no host copy |
| Consumer | any GStreamer sink |
| `GST INFO` / `GST LIST` | plugin and element discovery |

Config:

```xml
<gstreamer>
    <path>…</path>
    <plugin-path>…</plugin-path>
    <auto-load>…</auto-load>
</gstreamer>
```

---

## 2. Design decisions, and what they cost

**The GPU route deliberately does NOT append `d3d11convert`**, unlike the upstream module — and the
reasoning, recorded in `gst_gpu_bridge.h:41-62`, is the most instructive decision in this module:

- Upstream appends `d3d11convert` and hands the mixer **one BGRA texture**.
- The cost that matters is not bandwidth: **`d3d11convert` picks the YCbCr matrix and the range
  itself**, so the conversion happens outside the channel's colour management, with a matrix nobody
  chose.
- **10-bit survives without it.** `d3d11h265dec` produces P010, carried here at `bit16`, where a
  `d3d11convert` to BGRA8 would throw two bits away.

So this module hands the mixer the decoder's own planes and lets the shader convert — the *same*
argument that later drove the CUDA ProRes planar handoff, reached independently in a different
module. Two modules, one conclusion, and it is worth knowing they agree.

**Shared NT handles into Vulkan** via `vulkan::d3d11_import_bridge::copy_planes`
(`gst_gpu_bridge.h:62`) — the same bridge the FFmpeg D3D11VA path uses, so a change to it affects
both. That was flagged during the Vulkan Video work as needing re-verification for exactly this
reason.

**FFmpeg DLL collision is a real operational hazard.** GStreamer ships its own FFmpeg libraries; if
they reach the process ahead of the server's, behaviour changes in ways that look like a decoder
bug. `gst-dll-probe` exists to detect that, and the guide names the one plugin to exclude.

---

## 3. Verification

| what | battery | result |
| :--- | :--- | :--- |
| Producer and consumer, both mixers | `gstreamer` | 14/14 |
| Consumer cost | `gst-consumer-cost` | — |
| FFmpeg DLL collision detection | `gst-dll-probe` | — |
| Coexistence with other GPU paths | `coexistence` | engaged in company |

**What is not covered:** the range of pipelines. The batteries drive a small set; GStreamer's whole
point is that the pipeline is arbitrary, and an arbitrary pipeline is not testable. Treat the 14/14
as covering the *bridge*, not the space of things you can put through it.

There is also **no CUDA route** — the module's GPU path is D3D11 only. That was investigated and
recorded as a finding rather than a to-do; see the guide.

---

## 4. Known gaps

1. **Pipeline space is untestable**, so coverage attests the bridge only.
2. **`d3d11_import_bridge::copy_planes` is shared with the FFmpeg D3D11VA path** and a change to it
   needs both verified — easy to forget, since the two modules look unrelated.
3. **No CUDA route**, by investigation rather than oversight.

---

## 5. Related commits

Traced in the guide and in `../deprecated/GSTREAMER_INTEGRATION_PLAN.md` (in `deprecated/` because
the module shipped — read it for intent only).

---

## 6. Diagrams

The guide already carries them. Nothing owed here.
