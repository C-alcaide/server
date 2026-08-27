# FFmpeg — the fork's decode and encode paths

> **State:** shipped, and the largest single divergence in the tree
> **Module:** `src/modules/ffmpeg` — **10,500 lines** different from upstream across 20 files
> **Commands:** no dedicated `MIXER` command; reached through `PLAY`, `ADD` and configuration
> **Architecture:** [`../architecture/FFMPEG_8_MIGRATION.md`](../architecture/FFMPEG_8_MIGRATION.md)
> **Guide:** [`../guides/PLAYBACK_AND_RECORDING_GUIDE.md`](../guides/PLAYBACK_AND_RECORDING_GUIDE.md)
> **Coverage:** `decode-cost`, `gpu-direct-parity`, `encode-matrix`, `encode-parity`, `flat-decoded`,
> `source-colorspace`, `loop-boundary`, `seek`, `vk-decode-soak`

**This document exists because the folder's original scope rule excluded it.** `features/` was built
from *modules absent from upstream*, and `ffmpeg` is not absent — it is merely rewritten. Ten
thousand changed lines is the heaviest fork-specific work in the tree and it had no entry here.

---

## 1. What is implemented today

| piece | fork-only file | what it does |
| :--- | :--- | :--- |
| GPU-direct **encode**, Vulkan mixer | `consumer/cuda_vk_upload.{h,cpp}` | composited texture straight into CUDA device memory; a recording never reaches host memory |
| GPU-direct **encode**, OpenGL mixer | `consumer/cuda_gl_upload.{h,cpp}` | the same for the GL backend, via `cudaGraphicsGLRegisterImage` |
| **Vulkan compute decode** | `util/vulkan_hwdevice.{h,cpp}` | wraps *the mixer's own* `VkDevice` in an `AVVulkanDeviceContext`, so FFmpeg 8.1's ProRes / ProRes RAW / FFV1 / DPX compute decoders allocate on the device the mixer already has — no copy |
| **Vulkan Video decode** (H.264/HEVC) | `av_producer.cpp:1809-1829`, `device.cpp:939` | a real video-decode queue family, an allowlist, and `<vulkan-video-decode>` to gate it |
| **Declared colour space** | `util/av_color.h` | `unknown` is returned as a real answer instead of being flattened to BT.709 — see §3 |
| **Tweened filter parameters** | `producer/filter_param_tween.h` | animatable values inside an FFmpeg filter string |
| Loop cache, GPU-direct D3D11VA, seek/loop/ping-pong rework | `producer/av_producer.cpp` | the bulk of the 10,500 lines |

---

## 2. Configuration

Every element below is fork-specific. All live under `<ffmpeg><producer>`:

```
gpu-direct-decode          D3D11VA zero-copy for h264/hevc/vp9/av1
hardware-decode-adapter    which GPU the hardware decoder uses
vulkan-decode              FFmpeg 8 COMPUTE decoders (ProRes, ProRes RAW, FFV1, DPX)
vulkan-video-decode        VK_KHR_video_decode for H.264/HEVC — separate switch, separate risk
loop-cache-mb              per-producer resident cache for short loops
loop-cache-total-mb        process-wide ceiling for the above
threads / slice-threads    decoder threading
buffer-depth               decode-ahead
auto-deinterlace
```

**`vulkan-decode` and `vulkan-video-decode` are deliberately separate switches.** The compute
decoders need only a compute queue; the video decoders need a decode queue *and* a codec profile
the driver may refuse. A fault in one must not disable the other.

---

## 3. Design decisions worth knowing

**A queue must EXIST, not merely be requested** (`av_producer.cpp:1814`). The video-decode family is
discovered on the physical device and the allowlist is consulted only when the queue is really
there — because a codec that needs a queue it cannot have *faults* rather than declining, which is
what the VP9 finding recorded.

**`unknown` colour space is an answer, not a gap.** `av_color.h` returns what the frame *declares*.
Flattening an undeclared source to BT.709 is what made SD material decode with HD coefficients; the
SD/HD convention is now applied deliberately by `core::decode_color_space` rather than by accident
in the parser.

**The loop cache is dropped at the explicit-seek call site, not inside `seek_internal`.** A loop
wrap stays inside the range, so it does not invalidate a cache *of* that range. Dropping it in the
common path wiped the cache every time round the loop while the consumer was still mid-range, so it
never completed — a gate that never opens looks identical to a feature switched off. The reasoning
lives in this repo's `CLAUDE.md` because it cost three separate fixes to get right.

---

## 4. Verification

Measured by `decode-cost` (`force_cpu` / `auto` = D3D11VA / `cuda` / `vulkan` arms, with
`ENGAGE_REQUIRED` so a silently-declined arm cannot be reported as a saving), `gpu-direct-parity`
against a software reference, `encode-matrix` and `encode-parity` for the consumer, and
`flat-decoded` — the only 1 LSB decode gate.

**Not covered:** HDR and BT.2020 on most arms; the A/B between D3D11VA and Vulkan Video has not been
run; `vulkan-video-decode` has no dedicated battery.

---

## 5. HDR signalling survives a stream — measured

The colour description lives in the **video bitstream** as VUI in H.264/HEVC, not in the container.
MPEG-TS has no colour signalling of its own, so the transport cannot affect it — a structural
argument, and therefore measured rather than asserted. Read back with **ffprobe**, which is not our
code, so this is evidence about the stream rather than our encoder agreeing with our decoder.

| channel | over UDP | over SRT |
| :--- | :--- | :--- |
| `bt709` / `sdr` | `bt709` / `bt709` / `bt709` | — |
| `bt2020` / `hlg` | `bt2020` / `arib-std-b67` / `bt2020nc` | — |
| `bt2020` / `pq` | `bt2020` / `smpte2084` / `bt2020nc` | `bt2020` / `smpte2084` / `bt2020nc` |

**Mastering display volume and MaxCLL/MaxFALL travel too**, given an `<hdr-metadata>` block —
ST 2086 plus CTA-861.3, which is what HDR10 delivery requires. Measured over UDP via ffprobe:
`red_x 35400/50000` (0.708, BT.2020 red), `white_point 15635/50000` (0.3127, D65),
`max_luminance 10000000/10000` (1000 cd/m²), `min_luminance 50/10000` (0.005),
`max_content 1000`, `max_average 400`. Both **libx265 and libx264** carry it.

Configuration and the two deliberate constraints are in
[`../guides/HDR_GUIDE.md`](../guides/HDR_GUIDE.md). Measured 2026-08-17 by `signalling --stream`.

---

## 6. Known gaps

1. **No A/B for Vulkan Video against D3D11VA.** Both are implemented; nothing compares their cost or
   picture. This is the deliverable the Vulkan Video plan was written for.
2. ~~`av_producer.cpp.bak` in `producer/`~~ — **deleted 2026-08-27**, after checking rather than
   assuming. It was untracked *and* gitignored, so there was no git copy and deletion was permanent:
   2191 lines against the live file's 5874, matching no commit in the last 60 that touched
   `av_producer.cpp`. Exactly one symbol existed in it and not in the tree — `get_color_space` — and
   that had been extracted to `util/av_color.h`, where five call sites still use it. So it held
   nothing the tree had lost, which is the only basis on which an untracked file should be removed.
3. **16-channel AAC is broken on FFmpeg 8** (upstream defect, WAV is the workaround).
