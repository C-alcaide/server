# HAP — GPU-decoded intermediate codec

> **State:** shipped
> **Modules:** `src/modules/hap` (producer; `cpu`, `gl`, `snappy`, `util` subsystems)
> **Commands:** none of its own — a plain `PLAY` of a HAP file
> **Guide:** [`../guides/HAP_PLAYBACK.md`](../guides/HAP_PLAYBACK.md)
> **Architecture:** [`../architecture/HAP_DECODE_ROUTES.md`](../architecture/HAP_DECODE_ROUTES.md), [`../architecture/GPU_CODEC_HANDOFF.md`](../architecture/GPU_CODEC_HANDOFF.md)
> **Coverage:** `loop-boundary`, `mixer-parity --codec hap_q --decoder hap_native`

Plays HAP, HAP Alpha, HAP Q and HAP Q Alpha by handing the compressed DXT blocks straight to the
GPU as compressed textures. Snappy decompression runs on worker threads; the GPU never sees an
uncompressed frame in host memory, and for HAP Q the YCoCg→RGB resolve happens in the mixer shader.

---

## 1. What is implemented today

Texture formats, from `hap_frame_parser.h:52-54`:

| variant | fourCC | texture format | notes |
| :--- | :--- | :--- | :--- |
| Hap | `Hap1` | `RGB_DXT1` (0x0B) | no alpha |
| Hap Alpha | `Hap5` | `RGBA_DXT5` (0x0E) | straight alpha |
| Hap Q | `HapY` | `YCoCg_DXT5` (0x0F) | scaled YCoCg, resolved in the shader |
| Hap Q Alpha | `HapM` | 0x0D multi-texture container | YCoCg + an alpha-only DXT5 second section (0x0C) |

**The shader does the YCoCg resolve, not the producer** — `pixel_format::ycocg_dxt5` and
`ycocg_dxt5a` exist for exactly this, as shader cases 13 and 14. That keeps the frame compressed
all the way to the sampler and means the channel's colour management applies to the result rather
than to a producer-side conversion. Same argument as the CUDA ProRes planar handoff, reached
independently.

**Snappy, not LZ4** — a deliberate difference from the NotchLC path, noted in the module's own
header at `hap_producer.cpp:23`: *"GL compressed textures instead of CUDA kernels, Snappy instead
of LZ4"*. The two GPU codec producers in this fork took opposite implementation routes, and that
comment is the only place it was written down before this document.

---

## 2. How to drive it

**The `HAP` keyword is required.** `create_hap_producer` returns `empty()` unless `params[0]` is `HAP`
(`hap_producer.cpp:1588-1589`), so a plain `PLAY` of a HAP file **does not reach this module at
all** — FFmpeg decodes it on the CPU instead, and for HAP Q that means the YCoCg resolve described
in §1 never runs:

```
PLAY 1-1 HAP "clip_hapq.mov"
PLAY 1-1 HAP "clip_hapq.mov" LOOP
PLAY 1-1 HAP "clip_hapq.mov" PINGPONG SPEED 0.5
```

Also accepted after the file: `LOOP`, `PINGPONG`, `SPEED`, `SEEK`, `LENGTH`, `IN`/`OUT`.

No config elements.

**Why the error is worth recording rather than just correcting.** Both routes play the file and
both look right at a glance, so "no keyword needed" is a claim that survives casual use — and it
made the module's own coverage look better than it was. The harness's `hap` codec id maps to the
keyword-less route (`decoder="ffmpeg"`), which is why the first parity run on HAP Q measured the
CPU decode and could not have failed. See §4.

---

## 3. Design decisions, and what they cost

**Compressed textures all the way to the sampler.** The frame is uploaded as DXT and decoded by the
texture unit, so there is no full-resolution intermediate at any point. The cost is that HAP is
tied to formats the GPU can sample natively — which is why the variant table in §1 is the whole
feature and not a subset of a larger capability.

**Worker-thread Snappy with a bounded queue** (`hap_producer.cpp:92`): workers drain the raw queue
as fast as they can, and the bound is what stops a fast disk from growing an unbounded backlog.

**No flush at the loop wrap**, and this is worth knowing because it differs from the CUDA ProRes
producer. HAP's behaviour was the *reference* that exposed a defect elsewhere: on a four-frame loop
the CUDA ProRes producer showed only markers {40, 41} — two frames lost per iteration — while HAP
showed all four. The comparison is recorded in `av_producer`'s loop-wrap reasoning; HAP is the one
that was right.

---

## 4. Verification — what is measured, and what is not

| what | battery |
| :--- | :--- |
| Loop boundary behaviour | `loop-boundary` |
| HAP Q, both mixers, six rasters | `mixer-parity --codec hap_q --decoder hap_native` |

The parity battery was added on 2026-08-26 and **found two defects on its first honest run**. Both
were in code no existing battery could reach, because the only HAP fixture was `Hap1` — plain RGB
DXT1, decoded entirely by the texture unit, whose colour never touches our arithmetic.

### The chroma offset was 0.5 and should have been 128/255

The DXT5 blocks store 8-bit codes and neutral chroma is code 128, so the normalised offset is
`128/255 = 0.501961`. `0.5` is code 127.5, which is not representable. Three independent sources
agree: the derivation, **this module's own OGL shader** (`hap_gl_decode.cpp` FS_YCOCG, which had it
right from the start), and FFmpeg's reference decoder (`libavcodec/texturedsp.c:359-361`,
`co = (r - 128) / s`).

The error has a signature, and the measurement matched it *before* anything was changed — which is
what makes the diagnosis more than a plausible story:

| | R | G | B |
| :--- | ---: | ---: | ---: |
| predicted error | **0** (Co and Cg shift together in `Y+Co-Cg`) | `+e/scale` | `-2e/scale` |
| measured at 1920x1080, max LSB | **0** | **0** | **1** |
| measured, signed mean | +0.000 | +0.000 | **-0.800** |

Fixed in both mixer shaders. After the fix, **both rasters that require no resampling are
byte-identical between the backends** — 1 LSB → 0 at `1080i5000` and `1080p2500`.

### The Vulkan path resolves YCoCg after filtering, and that is accepted

`pipeline.cpp:345` sets `eLinear`, so `texture()` interpolates `(Co, Cg, scale, Y)` and case 13 then
divides by an **interpolated** `scale`. Interpolate-then-divide is not divide-then-interpolate when
the term is a denominator. The OGL route resolves at native resolution in the module's own FBO pass
and filters the result, so it is the more correct of the two.

| raster | source | max diff | pixels over 1 LSB |
| :--- | :--- | ---: | ---: |
| 1080p2500, 1080i5000 | 1:1, no resampling | **0** | 0.000% |
| PAL 720x576 | downscaled | 5 | 0.002% |
| 2600x1500p25 | rescaled | 14 | 0.171% |
| 2160p2500 3840x2160 | upscaled | 16 | 0.133% |

Zero at 1:1 and non-zero only when resampling is exactly what the model predicts. The `png_8`
control is **byte-identical** between the backends at the same rasters, so the two mixers scale
identically and this is specific to the HAP Q route.

**This is accepted rather than fixed**, and the reasoning is recorded so it can be revisited:
sampling the DXT blocks and resolving in the fragment shader is the reference approach that every
HAP implementation uses, Vidvox's own sample shaders included — our OGL FBO resolve is the unusual
one. The magnitude sits below DXT5 YCoCg's own compression error, so correcting it would buy
accuracy the source does not carry. And the fix costs precisely what the design exists to save: a
full-resolution intermediate texture plus a resolve pass, per frame per layer, which is the
zero-copy compressed-texture handoff itself. **If colour fringing on scaled HAP Q is ever reported,
the fix is to resolve into an intermediate texture the way the OGL route already does.**

### What is still not covered

**No comparison against a reference decoder.** Parity says the two backends agree, not that either
is right. The offset fix is argued from three external sources rather than from a captured
comparison — the 14 ffmpeg-rendered `hap_q` / `hap_alpha` ground-truth references now exist in the
harness, so `output.ground_truth` on those codecs is the next check and has not been run.

**`CALL SEEK` was lost on ~21% of Vulkan captures — fixed, and it uncovered a second defect
underneath.** Measured before: the Vulkan half captured frame **0** where the OpenGL half captured
the pinned frame 7, in **5 of 24 captures**; OpenGL never did it, 0 of 24.

Not a harness settle problem, which is what it first looked like. The pin is deterministic — `LOAD`
pauses on frame 0 and the script issues `CALL 1-1 SEEK 7` — so frame 0 is exactly the frame `LOAD`
leaves behind, and the symptom was a **seek that never took effect**. `gl_loop` validated an item's
epoch under `done_mutex_`, dropped the lock, decoded (a BC upload, an FBO pass or a CPU
decompression), and pushed into a queue the seek had emptied in the meantime. `last_frame()` then
consumed `seek_done_` on that first successful pop, cached the pre-seek picture and cleared the
flag — so nothing ever revisited it and a paused channel held frame 0 for the rest of the run.

Fixed by tagging each queued frame with its seek epoch and discarding stale ones **at the pop**,
in `receive_impl` (before its empty test, so a queue of only pre-seek frames reads as an underrun)
and in `last_frame` (before it consumes `seek_done_`).

**The discard had to move to the consumer, and the reason is the second defect.** Dropping the
stale frame on the decode thread — the obvious implementation — produced **1450
`vk::DeviceLostError` in eight minutes**. Rewriting it to release from the mixer thread instead
still produced 1749. The thread was never the problem: `submitSingleTimeCommands` submits without
waiting and `texture::impl::~impl()` frees immediately, so **any** early release of an uploaded
texture frees the `VkImage` while the GPU is still writing into it. That is a latent defect in the
Vulkan accelerator, not in HAP — a published frame outlives its copy by a wide margin, so nothing
had ever released one early enough to expose it. `copy_compressed_async` now waits on its timeline
value the way `copy_async` beside it already did.

| binary | frame mismatches / 24 | `vk::DeviceLostError` |
| :--- | ---: | ---: |
| before | 5 | 0 |
| discard on the decode thread | 0 | **1450** |
| reverted baseline | 4 | 0 |
| discard on the consumer thread | 0 | **1749** |
| + upload wait | **0** | **0** |

The last row is four runs of six rasters with **every raster measured in every run** and the four
tables byte-for-byte identical. At the observed rate a clean sweep is ~1% likely by chance.

**HAP Q Alpha has no fixture and no reachable code.** ffmpeg's hap encoder writes
hap/hap_alpha/hap_q only, so `HapM` cannot be produced here. It is also unreachable: nothing in the
tree publishes `ycocg_dxt5a`: the `use_vk_upload_` branch tests
`item.result.variant != HapVariant::HapQAlpha` and falls through to the CPU decoder for that one
variant, because the zero-copy path cannot carry two textures.
Shader case 14 is dead on **both** mixers.

---

## 5. Known gaps

1. **No comparison against a reference decoder** on any variant. Parity between the backends is
   now covered; correctness against an external oracle is not. The ground-truth references exist —
   see §4.
2. **Plain HAP and HAP Alpha have no picture coverage at all.** Only `hap_q` is driven by
   `mixer-parity`; `hap_alpha` has a fixture and references but no battery pointed at it.
3. **HAP Q Alpha (`HapM`) is unreachable and unfixturable** — §4. Shader case 14 is dead on both
   mixers, and the channel-order fix applied to it there is a reading, not a measurement.
4. ~~`CALL SEEK` is lost on ~21% of Vulkan captures~~ — **fixed**, see §4, along with the Vulkan
   texture-lifetime defect it exposed.
5. **No cost measurement** — Snappy worker count versus channel count has never been measured, so
   there is no channel-count ceiling for HAP as there is for ProRes.

---

## 6. Related commits

Not traced; the module predates this document. Its loop-wrap behaviour is referenced from
`av_producer.cpp`'s reasoning as the correct reference case.

---

## 7. Diagrams

![What each GPU codec producer hands the mixer](../images/feature_codec_handoff.png)

HAP is the middle column: it keeps the DXT compressed all the way to the sampler and resolves
YCoCg in the shader, where CUDA ProRes hands over planes and CUDA NotchLC hands over a packed
texture. Generated by `docs/diagrams/generate_feature_diagrams.py`.
