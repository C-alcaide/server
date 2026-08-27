# HAP Playback

HAP is an intermediate codec that keeps its frames as **GPU-compressed textures** all the way to the
sampler — no full-resolution intermediate anywhere, and no CPU decode of the picture. It is the
codec of choice for heavy multi-layer playback where ProRes or h264 would exhaust decode headroom.

CasparVP plays HAP, HAP Alpha, HAP Q and HAP Q Alpha.

---

## 1. The `HAP` keyword is required

**This is the single thing to get right.** The native producer refuses to open a file unless the
first parameter is `HAP`:

```
PLAY 1-1 HAP "clip.mov"
```

A plain `PLAY 1-1 "clip.mov"` **still plays** — through FFmpeg, on the CPU. You get a picture, so
nothing looks wrong. What you lose is the entire point of the codec: the compressed-texture path,
and for HAP Q the shader-side YCoCg resolve.

**How to tell which one you got:** the server logs `[hap_producer]` lines on startup — the variant
it detected, the file's resolution and frame rate. No `[hap_producer]` line means you are on the
FFmpeg path and the keyword was missing.

---

## 2. Full syntax

```
PLAY 1-1 HAP "clip.mov"
PLAY 1-1 HAP "clip.mov" LOOP
PLAY 1-1 HAP "clip.mov" PINGPONG
PLAY 1-1 HAP "clip.mov" SPEED 0.5
PLAY 1-1 HAP "clip.mov" SEEK 120 LENGTH 240
PLAY 1-1 HAP FILE "clip.mov" IN 120 OUT 360
```

| parameter | meaning |
| :--- | :--- |
| `LOOP` | loop at the out point |
| `PINGPONG` | play forward, then backward, repeatedly |
| `SPEED <n>` | playback rate; negative plays in reverse |
| `SEEK <n>` / `IN <n>` / `START <n>` | first frame — three spellings of the same thing |
| `LENGTH <n>` | how many frames to play |
| `OUT <n>` | last frame |
| `FILE <path>` | the path, when you would rather be explicit |

No configuration elements — HAP needs nothing in `casparcg.config`.

---

## 3. The four variants, and which to author

| variant | fourCC | alpha | notes |
| :--- | :--- | :--- | :--- |
| HAP | `Hap1` | no | smallest, DXT1, lowest quality |
| HAP Alpha | `Hap5` | **yes**, straight | DXT5 |
| HAP Q | `HapY` | no | **best quality**; scaled YCoCg in DXT5, resolved on the GPU |
| HAP Q Alpha | `HapM` | yes | HAP Q plus a separate alpha texture |

**HAP Q is the one to author for quality**, and the one whose colour is computed by this fork's own
shader code rather than by the GPU's texture unit.

**HAP Q Alpha is the least-exercised path.** It falls back to a CPU decode even on the Vulkan mixer,
because the zero-copy route cannot carry two textures, so it does not get the codec's main benefit.
If you need alpha and performance, prefer HAP Alpha.

---

## 4. Two behaviours worth knowing before a show

**Scaling a HAP Q clip costs a little colour accuracy on the Vulkan mixer.** The mixer samples the
compressed texture and resolves YCoCg afterwards, so filtering happens before the resolve. At native
raster the two mixers are byte-identical; where the raster forces resampling they differ by 5–16 LSB
at a fraction of a percent of pixels, concentrated at edges. It is below the codec's own compression
error and is accepted rather than a defect —
[`../features/hap.md`](../features/hap.md) §4 has the measurements and the reasoning.

Play HAP Q at the channel's native raster where you can, and this does not arise at all.

**Loop behaviour is the reference case in this codebase.** On a short loop HAP shows every frame,
where the CUDA ProRes producer once lost two per iteration. HAP performs no flush at the wrap, which
is the behaviour the other producer was corrected toward — so if you are comparing loop precision
between codecs, HAP is the one that was right.

---

## 5. Interlaced output

HAP is **excluded from interlaced video modes** in this fork's test matrix, and that exclusion is
deliberate rather than an oversight. Treat HAP into an interlaced channel as untested.

---

## 6. What is measured, and what is not

Covered by `mixer-parity --codec hap_q --decoder hap_native` (both backends, six rasters) and
`loop-boundary`. Figures live in [`../features/hap.md`](../features/hap.md); implementation notes in
[`../architecture/HAP_DECODE_ROUTES.md`](../architecture/HAP_DECODE_ROUTES.md).

**Not measured:**

* **No comparison against a reference decoder** on any variant — parity says the two mixers agree,
  not that either is right.
* **Plain HAP and HAP Alpha have no picture coverage at all**; only HAP Q is driven.
* **HAP Q Alpha has no fixture** — ffmpeg's HAP encoder cannot produce `HapM`, so one has to be
  authored elsewhere.
* **No cost measurement.** The Snappy worker count against channel count has never been profiled, so
  there is no channel-count ceiling for HAP as there is for ProRes.
