# GStreamer producer and consumer

> **State:** shipped
> **Modules:** `src/modules/gstreamer`
> **Commands:** 2 (`GST INFO`, `GST LIST`)
> **Guide:** [`../guides/GSTREAMER_GUIDE.md`](../guides/GSTREAMER_GUIDE.md)
> **Coverage:** `gstreamer`, `gst-consumer-cost`, `gst-dll-probe`, `coexistence`

Plays and records through GStreamer pipelines, so anything GStreamer can source — SRT, RTSP,
NDI via a plugin, hardware capture — becomes a producer, and any GStreamer sink becomes a consumer.
The GPU route keeps hardware-decoded frames on the GPU into the mixer.

Operator detail is in [`../guides/GSTREAMER_GUIDE.md`](../guides/GSTREAMER_GUIDE.md), plus the
module's own `src/modules/gstreamer/README.md`. This document is the state and the decisions.

---

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
