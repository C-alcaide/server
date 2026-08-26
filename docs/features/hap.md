# HAP — GPU-decoded intermediate codec

> **State:** shipped
> **Modules:** `src/modules/hap` (producer; `cpu`, `gl`, `snappy`, `util` subsystems)
> **Commands:** none of its own — a plain `PLAY` of a HAP file
> **Coverage:** `loop-boundary`

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

No keyword — HAP is detected from the file:

```
PLAY 1-1 "clip_hapq.mov"
PLAY 1-1 "clip_hapq.mov" LOOP
```

No config elements.

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

**What is not covered:** the picture. No battery compares a HAP-decoded frame against a reference
decoder, on any of the four variants. The DXT decode is done by the texture unit so it is unlikely
to be wrong, but the **YCoCg resolve in the shader is our code** — and the scaled-YCoCg maths
(`scale = c.b * (255/8) + 1`, then `Co`/`Cg` divided by it) is exactly the kind of per-channel
arithmetic where a red/blue exchange hides. Both mixers implement it separately, cases 13 and 14.

Given the ICVFX precedent, a HAP Q parity check between the two mixers on an **asymmetric** source
is the obvious missing battery and would be cheap.

---

## 5. Known gaps

1. **No picture coverage on any variant.** A mixer-parity check on HAP Q with an asymmetric source
   is the highest-value first battery, for the reason in §4.
2. **HAP Q Alpha's second section (0x0C alpha-only DXT5) is the least-exercised path** and has no
   fixture in the harness.
3. **No cost measurement** — Snappy worker count versus channel count has never been measured, so
   there is no channel-count ceiling for HAP as there is for ProRes.

---

## 6. Related commits

Not traced; the module predates this document. Its loop-wrap behaviour is referenced from
`av_producer.cpp`'s reasoning as the correct reference case.

---

## 7. Diagrams

Not warranted alone — one linear chain, no branch. If a decode-route diagram is drawn for
`../guides/PLAYBACK_AND_RECORDING_GUIDE.md`, HAP is one row on it.
