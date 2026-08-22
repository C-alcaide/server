# Playback and recording — every route, what it costs, and which to pick

For the person running the machine, not the person building it. Every number here was measured on
the reference box (RTX A4000 + Quadro P4000, driver 582.53) with the harness batteries named
beside it, so you can re-measure rather than trust.

If you read one thing: **the fast paths are not interchangeable, and each one needs the channel
configured a particular way.** A recording that silently runs three times slower than it should
is almost always a fast path that declined because the channel was 8-bit when it needed 16, or
Vulkan when it needed OpenGL. The server always says so in its log. §7 is how to check.

---

## Contents

1. [Three different "how many" questions](#1-three-different-how-many-questions)
2. [Playback: the four decode routes](#2-playback-the-four-decode-routes)
3. [Recording: the four encode routes](#3-recording-the-four-encode-routes)
4. [Commands you can paste](#4-commands-you-can-paste)
5. [Why the pictures differ](#5-why-the-pictures-differ)
6. [How many at once](#6-how-many-at-once)
7. [Checking which path you actually got](#7-checking-which-path-you-actually-got)
8. [Which path for which job](#8-which-path-for-which-job)
9. [What none of this covers](#9-what-none-of-this-covers)

---

## 1. Three different "how many" questions

These get confused constantly, and the numbers are not interchangeable.

| question | shape | what limits it |
| :--- | :--- | :--- |
| "how many **playout channels**?" | N channels, one producer each | decode cost and upload bandwidth, N independent ticks |
| "how many **ISO recordings**?" | N channels, one recording each | encode cost, N independent ticks |
| "how many **outputs on one channel**?" | 1 channel, several consumers | **one** frame budget shared between them |

The third is the one that surprises people. A channel walks its consumers on its own tick and
**cannot advance until every one of them has taken the frame**, so the slowest consumer sets the
pace for all the others. That is how a single slow recording makes an on-air SDI output late.
Eight recordings on eight channels get eight frame budgets; eight recordings on one channel share
one.

And a per-unit cost cannot be multiplied to get a ceiling. The mixer's per-tick work is not
linear in load, and upload bandwidth saturates before CPU does — so a route that is cheapest at
one channel is not necessarily the one that reaches furthest.

---

## 2. Playback: the four decode routes

### At a glance

| route | how you select it | requires | host memory touched? |
| :--- | :--- | :--- | :--- |
| **Software** | `<gpu-direct-decode>false</gpu-direct-decode>` | nothing | yes — decode and upload |
| **D3D11VA GPU-direct** | `<gpu-direct-decode>true</gpu-direct-decode>` *(default)* | a codec the GPU decodes (H.264, HEVC, VP9, AV1) | **no** |
| **CUDA ProRes** | `PLAY 1-1 CUDA_PRORES "clip"` | ProRes source, NVIDIA GPU | no |
| **FFmpeg Vulkan** | `<vulkan-decode>true</vulkan-decode>` | Vulkan mixer; ProRes, ProRes RAW, FFV1 or DPX | no |

### The exact path each one takes

**Software.** The safe path, and the only one that handles every codec.

```
file → libavcodec software decoder → AVFrame (yuv420p / yuv422p10le / …)
     → filter graph (may convert format)
     → host buffer
     → PCIe upload, one texture per plane
     → mixer: shader converts YCbCr → RGB
```

The planes are uploaded as-is and the **shader** does the YCbCr→RGB conversion, so no CPU colour
conversion happens. What costs is the decode itself and the upload.

**D3D11VA GPU-direct** — the default, and the picture never enters host memory.

```
file → D3D11VA hardware decoder → NV12 texture on the GPU (DXGI_FORMAT_NV12)
     → shared handle → the mixer's device
        (OpenGL: WGL_NV_DX_interop · Vulkan: the D3D11 import bridge)
     → two plane views: R8_UNORM for Y, R8G8_UNORM for UV
     → mixer: shader converts YCbCr → RGB
```

Only for codecs the GPU's decode block handles. ProRes is **not** one of them — no NVIDIA GPU has
a ProRes decoder — so a ProRes clip on this route logs *"the decoder has no hardware device"* and
falls back to software. That is expected, not a fault.

**CUDA ProRes** — the fork's own decoder, for ProRes specifically.

```
file → CUDA kernels: bitstream parse, dequantise, IDCT
     → BGRA16 buffer in GPU memory
     → CUDA→OpenGL or CUDA→Vulkan copy
     → mixer texture
```

**FFmpeg Vulkan compute** — new in FFmpeg 8, compute shaders rather than a fixed-function block.

```
file → FFmpeg Vulkan compute decoder (prores · prores_raw · ffv1 · dpx)
     → VkImage planes on the mixer's own device
     → imported directly, no copy across devices
     → mixer: shader converts YCbCr → RGB
```

Because it is a compute shader it needs no ProRes hardware, which is why ProRes decodes on the
GPU here and cannot on the D3D11VA route.

### What they cost

Measured with `decode-cost`, three interleaved rounds per arm, all four producers confirmed on
their route's fast path:

| source | software | CUDA ProRes | FFmpeg Vulkan |
| :--- | ---: | ---: | ---: |
| ProRes 422 HQ, 10-bit | 1.90 cores | 1.26 (−33.7%) | **1.16 (−38.9%)** |
| ProRes 4444, 12-bit + alpha | 2.85 cores | 1.25 (−56.1%) | **1.16 (−59.3%)** |

**The Vulkan decoder costs the same 1.16 cores whether the content is 422 or 4444.** The decode
is effectively free and what remains is the mixer's own fixed cost. The software decoder's cost,
by contrast, rises with the format's difficulty — which is exactly why the heavier the source,
the more a GPU route is worth.

---

## 3. Recording: the four encode routes

### At a glance

| route | how you select it | requires | host memory touched? |
| :--- | :--- | :--- | :--- |
| **Host (CPU encoders)** | any `-vcodec` with no fast path | nothing | yes — readback + convert |
| **NVENC GPU-direct** | `-vcodec h264_nvenc` / `hevc_nvenc` | **8-bit** channel, CUDA | **no** |
| **FFmpeg Vulkan** | `-vcodec prores_ks_vulkan` / `ffv1_vulkan` / `h264_vulkan` / `hevc_vulkan` | **16-bit** channel, Vulkan mixer | no |
| **CUDA ProRes** | `ADD 1 CUDA_PRORES …` | **OpenGL** mixer | no |

**No single channel satisfies all three fast paths.** NVENC needs 8-bit, the Vulkan encoders need
16-bit, and CUDA ProRes needs the OpenGL mixer. Choose the channel for the recording you intend.

### The exact path each one takes

**Host.** Everything that has no fast path.

```
mixer texture → readback to host memory
                  8-bit channel  → BGRA
                  16-bit channel → RGBA64LE
              → libswscale converts to the encoder's pixel format
                  (e.g. yuv422p10le for ProRes, yuv420p for H.264)
              → CPU encoder
```

Two costs: the readback (about 8 MB per 1080p frame at 8-bit, 16 MB at 16-bit) and the swscale
conversion, both on top of the encoder itself.

**NVENC GPU-direct.**

```
mixer texture (8-bit) → CUDA copy, byte-for-byte, no conversion
                      → CUDA frame:  RGB0  on the OpenGL mixer
                                     BGR0  on the Vulkan mixer
                      → NVENC does RGB→YCbCr in hardware
                      → bitstream
```

The frame format differs by mixer because the two mixers hold their 8-bit composite in opposite
byte order. This is handled for you; it is listed because it explains why the path is byte-exact
and needs no conversion kernel — and therefore why it is restricted to 8-bit.

**FFmpeg Vulkan encoders.**

```
mixer texture (16-bit RGBA) → VkImage copy on the mixer's own device
                            → libplacebo converts on the GPU
                                 → yuv422p10  for prores_ks_vulkan / ffv1_vulkan
                                 → nv12       for h264_vulkan / hevc_vulkan
                            → encoder
```

The 16-bit requirement is a **colour-order** constraint, not a quality preference: the 8-bit
composite is BGRA and libplacebo exchanges red and blue on a BGRA Vulkan frame, so an 8-bit
channel is refused outright rather than recorded wrongly.

`h264_vulkan` and `hevc_vulkan` run on the **NVENC hardware block** reached through Vulkan —
measured at 15–39% block utilisation, where the compute encoders read 0%. `prores_ks_vulkan` and
`ffv1_vulkan` are compute shaders and use no fixed-function unit.

**CUDA ProRes consumer.**

```
mixer OpenGL texture → CUDA-GL map on the mixer's own thread
                     → BGRA8 buffer in GPU memory
                     → red/blue exchange (the texture and a readback differ in byte order)
                     → CUDA kernels: BGRA → v210 (422) or YUVA444P10 (4444)
                     → GPU DCT, quantise, entropy code
                     → .mov or .mxf
```

### What they cost

`encode-matrix`, 1080p2500, two interleaved rounds, every fast path confirmed engaged:

| codec | route | cores | NVENC block | frames kept | MB / 10 s |
| :--- | :--- | ---: | ---: | ---: | ---: |
| ProRes 422 HQ | `prores_ks_vulkan` | **1.46** | 0% | 258/258 | 50.9 |
| | `CUDA_PRORES` | 1.64 | 0% | 260/260 | 55.4 |
| | `prores_aw` (CPU) | 2.24 | 0% | 260/260 | 22.4 |
| | `prores_ks` (CPU) | 2.32 | 0% | **138/260** | 14.4 |
| H.264 | `h264_nvenc` † | **1.37** | 9% | 255/255 | 0.1 |
| | `h264_vulkan` | 1.40 | 15% | 259/259 | 0.3 |
| | `libx264` | 2.18 | 0% | 262/262 | 0.1 |
| HEVC | `hevc_nvenc` † | **1.39** | 11% | 258/258 | 0.1 |
| | `hevc_vulkan` | 1.43 | 36% | 256/256 | 6.7 |
| | `libx265` | 2.83 | 0% | 252/252 | 0.1 |
| FFV1 | `ffv1_vulkan` | **1.50** | 0% | 258/258 | **210.0** |
| | `ffv1` (CPU) | 2.26 | 0% | 260/260 | 11.8 |

† NVENC requires `tools/use_local_ffmpeg.sh apply` on this machine — see §8.

**Two rows to read carefully.**

`prores_ks` **kept 138 of 260 frames.** It is not a cheaper recording, it is half a recording: the
encoder cannot sustain 1080p25 and the channel drops what it cannot take. **`prores_aw` can**, for
slightly less CPU. If you are recording ProRes on the CPU, ask for `prores_aw`.

`ffv1_vulkan` wrote **210 MB against 11.8 MB** for the same ten seconds. FFV1 is lossless, so that
is not a quality difference — it is a weaker entropy coder. Eighteen times the disk is often
enough to decide the trade on its own.

---

## 4. Commands you can paste

**Playback**

```
PLAY 1-1 "clip"                          # default route: D3D11VA GPU-direct if the codec allows
PLAY 1-1 CUDA_PRORES "clip"              # the CUDA ProRes decoder
```

Software decode, or the Vulkan decoders, are configuration rather than commands:

```xml
<gpu-direct-decode>false</gpu-direct-decode>   <!-- force software -->
<vulkan-decode>true</vulkan-decode>            <!-- FFmpeg Vulkan compute decoders -->
```

**Recording — a 16-bit Vulkan-mixer channel**

```
ADD 1 FILE "out.mov" -vcodec prores_ks_vulkan -ac 2
ADD 1 FILE "out.mkv" -vcodec ffv1_vulkan -ac 2
ADD 1 FILE "out.mov" -vcodec h264_vulkan -b:v 50M -ac 2
REMOVE 1 FILE "out.mov" -vcodec prores_ks_vulkan -ac 2
```

**Recording — an 8-bit channel**

```
ADD 1 FILE "out.mp4" -vcodec h264_nvenc -b:v 60M
ADD 1 FILE "out.mov" -vcodec prores_aw -pix_fmt yuv422p10le
ADD 1 CUDA_PRORES PATH "D:/recordings" FILENAME "out.mov" PROFILE 3
REMOVE 1 CUDA_PRORES
```

Three things that catch people out:

* **`REMOVE` needs the same arguments as the `ADD`.** `REMOVE 1 FILE` on its own returns
  `404 REMOVE FAILED` and finalises nothing — the container is left without its trailer and will
  not open. The server builds a consumer from the parameters to identify which one to remove.
* **`-ac 2` on FFmpeg 8.** The channel's default 16-channel audio layout is refused by the AAC
  encoder (*"Unsupported channel layout 9.1.6"*), which fails the whole recording for a reason
  that has nothing to do with video.
* **Set a bitrate.** The GPU encoders' defaults are untuned and vary by more than a factor of ten
  between routes producing the same picture.

---

## 5. Why the pictures differ

Every GPU route was compared against the CPU encoder on a still, one frame each. The numbers are
small; what matters is *where* the difference sits.

| route | mean difference | where it sits |
| :--- | ---: | :--- |
| `prores_aw` vs `prores_ks` | 0.17 LSB | at colour boundaries |
| `CUDA_PRORES` | 0.89 LSB | 63% at colour boundaries |
| `prores_ks_vulkan` | 2.55 LSB | 86% at colour boundaries |
| `h264_vulkan` | 2.70 LSB | 55% at boundaries, rest is block noise |
| `hevc_nvenc` | 1.64 LSB | 53% at boundaries |

**Why "at colour boundaries" is the reassuring answer.** All these codecs store colour at half
horizontal resolution (4:2:2) or quarter (4:2:0). At a hard vertical edge — the join between two
colour bars — there is no single correct answer for the shared colour sample, and two
implementations legitimately choose differently. The disagreement is then **one or two pixel
columns wide at each transition and nothing in between**, which is what these percentages
describe.

A difference spread evenly across **flat areas** would be the worrying one: that is a colour
error, not a resampling choice. That is exactly how two real defects were caught this month — a
red/blue exchange reads as ~166 LSB with only 5% at boundaries, because flat areas are wrong
everywhere.

**Why a grey test pattern is a bad check.** Grey has equal red and blue, so it is unchanged by a
red/blue exchange and by most colour-matrix errors. Both defects above would have passed a grey
ramp cleanly. Use colour bars or anything with saturated, *asymmetric* colour.

**One number that is not the encoder.** For FFV1 — a lossless codec — any difference at all comes
from the RGB→YCbCr conversion, not the encoding: the CPU route converts with libswscale and the
Vulkan route with libplacebo. The 2.49 LSB there says nothing about either encoder.

---

## 6. How many at once

*(Measured with `iso-scaling`, `playback-scaling` and `consumer-scaling`. Numbers are filled in
below once the ladders complete — this section is deliberately empty rather than estimated.)*

---

## 7. Checking which path you actually got

**Never assume.** Every fast path announces itself, and every decline says why. Recording:

```
[ffmpeg] Vulkan encode: prores_ks_vulkan on the mixer's own device, converting with
         libplacebo=format=yuv422p10 -- the composite never reaches host memory
[ffmpeg] GPU-direct recording active: the composited texture goes straight to NVENC
[cuda_prores] GPU-direct path active (CUDA-GL interop on the mixer's own thread)
```

and the declines, which name the cause:

```
[ffmpeg] Vulkan encode not used for prores_ks_vulkan: the channel is 8-bit, whose composite is
         BGRA -- and libplacebo exchanges red and blue on a BGRA Vulkan frame; use
         <color-depth>16</color-depth>
[ffmpeg] GPU-direct recording not used: channel is not 8-bit.
```

Confirm the channel stopped reading back — this is the line that proves it end to end:

```
output[1] No consumer needs CPU readback (1 consumers); mixer readback skipped.
```

If instead you see `CPU readback required by consumer <name>`, that consumer is on the host path
and **every other consumer on that channel loses the benefit too**, because the readback happens
once per tick and is shared.

Playback:

```
[ffmpeg] Vulkan GPU-direct video active
[ffmpeg] D3D11 GPU-direct video not used: the decoder has no hardware device
         (codec not hardware-accelerated here)
```

---

## 8. Which path for which job

### If the channel runs the Vulkan mixer

Use the **Vulkan encoders** and set the channel to 16-bit. They are the only GPU recording route
that works on a Vulkan channel without a rebuild, and `prores_ks_vulkan` is the cheapest ProRes
route measured. `<vulkan-decode>` on the same channel gives you the Vulkan *decoders* too, and
decode and encode then share one device with no cross-API import anywhere in the chain.

```xml
<accelerator>vulkan</accelerator>
<color-depth>16</color-depth>
<vulkan-decode>true</vulkan-decode>
```

The one thing you give up is NVENC, which needs 8-bit. On this machine that is currently moot —
see below.

### If you need H.264 or HEVC specifically

**NVENC is the better path where it is available**: cheaper than the Vulkan encoder, closer to the
CPU encoder's picture, and far better rate-controlled by default (0.1 MB against 6.7 MB on HEVC).
It also uses *less* of the NVENC block for the same work, so the NVENC API drives that unit more
efficiently than Vulkan video-encode does.

**But it does not work with the pinned FFmpeg on this machine.** The pinned build requires NVIDIA
driver 610+; the box runs 582.53 and **cannot be raised**, because driver R580 is the last branch
supporting the Pascal Quadro P4000 in the second slot. `tools/use_local_ffmpeg.sh apply` swaps in
a locally built FFmpeg that restores NVENC at the same FFmpeg version — read that script first,
its codec set is narrower and the swap reverts on the next build.

Without the swap, use `h264_vulkan` / `hevc_vulkan`. They reach the same silicon.

### If you are recording ProRes

- **Vulkan mixer, 16-bit** → `prores_ks_vulkan`.
- **OpenGL mixer, 8-bit** → `CUDA_PRORES`.
- **CPU** → `prores_aw`, never `prores_ks`. The latter cannot sustain 1080p25.

### If a DeckLink output is on air

This is where the shared frame budget matters most. The SDI output and every recording on that
channel take the same frame, and the channel waits for all of them — so **a slow recording on the
same channel can make the on-air output late.**

Two rules follow:

1. **Keep the on-air channel's consumer list short.** If you need several recordings, put them on
   their own channels rather than on the channel that is on air. Separate channels get separate
   frame budgets.
2. **Do not mix a host-path consumer onto a GPU-direct channel.** The composite is read back once
   per tick if *any* consumer needs host pixels, and that readback is then paid by the channel
   regardless of how GPU-native the other consumers are. One CPU recording removes the benefit for
   all of them.

DeckLink output itself has a GPU-direct path (NVIDIA DVP, "Tier-2 GPU-direct output"), so an
SDI-out channel can stay GPU-resident end to end — but only if nothing else on it asks for host
pixels.

### If the source is ProRes and you also want to record

Decode and encode compete for the same GPU. `<vulkan-decode>` plus `prores_ks_vulkan` keeps both
on one device with no copies, which is the least-contended combination measured — but the ceilings
in §6 are what should decide the count, not this reasoning.

---

## 9. What none of this covers

* **1080p25 only.** No 4K, no 50p. Both change the balance: upload bandwidth and readback scale
  with pixels, so GPU routes gain as the raster grows.
* **One clip, one still.** The cost figures use a moving ProRes clip and the picture figures a
  still. Real content with grain or motion costs more to encode.
* **No alpha.** The 4444 and key/fill cases are measured only by `decode-cost`, not here.
* **Nothing is tuned.** Every encoder ran at its default rate control, which is why recorded sizes
  vary by over ten times between routes producing the same picture.
* **AV1 encoding is unavailable**, on any route. This GPU is Ampere and has no AV1 encoder;
  FFmpeg declines with *"No capable devices found"*. `av1_vulkan` needs an Ada-generation card.
* **The ceilings in §6 are for the shapes stated there** and cannot be converted into each other.

---

## Re-measuring

```bash
cd d:\Github\CasparCG-TestRunner
python cli.py encode-matrix     --server <casparcg.exe> --codec prores    # routes, cost, picture
python cli.py decode-cost       --server <casparcg.exe> --arms force_cpu,cuda,vulkan
python cli.py iso-scaling       --server <casparcg.exe> --media media/sources   # ISO recordings
python cli.py playback-scaling  --server <casparcg.exe>                    # playout channels
python cli.py consumer-scaling  --server <casparcg.exe> --base decklink    # outputs per channel
python cli.py encode-visual     # contact sheet of what each route produced
```

Every one of those refuses to report a number when its fast path declined, which is the property
that makes the tables above worth anything.
