# CUDA NotchLC — GPU decode

> **State:** shipped
> **Modules:** `src/modules/cuda_notchlc`
> **Commands:** none of its own — the `CUDA_NOTCHLC` keyword on `PLAY`
> **Coverage:** `producer-swap`, `coexistence`

Decodes NotchLC entirely on the GPU: nvcomp LZ4 decompression, then CUDA kernels for the Y, UV and
alpha sections, then a YCoCg→BGRA16 convert, then a zero-copy handoff to the mixer. Built for the
12K 360° material that this fork's projection work targets.

---

## 1. What is implemented today

The per-frame pipeline, from the module's own header (`notchlc_decode.cu:27-38`):

| step | what |
| :--- | :--- |
| 1 | upload compressed payload, host→device async through pinned staging |
| 2 | **nvcomp LZ4** decompress — or a direct copy for the Uncompressed format |
| 3 | sync, read the first 256 bytes back as the block header |
| 4 | parse 10 little-endian `u32` header fields to derive every section offset |
| 5 | `k_notch_y_decode` — one thread per 4×4 luma block |
| 6 | `k_notch_uv_decode` — one thread per 16×16 chroma block |
| 7 | `k_notch_a_decode`, or `fill_opaque` when there is no alpha |
| 8 | `k_notch_ycocg_to_bgra16` — one thread per pixel |
| 9 | `cudaMemcpy2DToArrayAsync` into the mixer's texture, or to host for the headless variant |

**Step 3 is a synchronisation point in the middle of the frame**, and it is unavoidable: the
section offsets live in the compressed payload, so nothing downstream can be launched until the
header has come back to the host. That is the structural difference from ProRes, whose slice table
is parsed on the CPU before any upload.

**Chroma is decoded at 16×16 granularity against 4×4 luma** — a 4:2:0-like ratio in blocks rather
than samples, which is where the format's compression comes from and why the UV kernel has a
different thread mapping from the Y kernel.

---

## 2. How to drive it

The keyword is required and the decoder is unreachable without it
(`notchlc_producer.cpp:1378-1380`):

```
PLAY 1-1 CUDA_NOTCHLC "clip.mov"
PLAY 1-1 CUDA_NOTCHLC FILE "clip.mov" DEVICE 0 LOOP
```

The long form takes `FILE`, an optional `DEVICE <index>` to pin the CUDA device, and the usual
`LOOP`.

---

## 3. Design decisions, and what they cost

**Still hands the mixer a packed BGRA16 texture**, unlike CUDA ProRes 4:2:2 which now hands over
planes. That is not an oversight — NotchLC's kernels produce YCoCg rather than YCbCr, and the
mixer's planar YCbCr path would need a YCoCg plane format that does not exist. The HAP producer
solved the same problem differently by keeping YCoCg compressed to the sampler and resolving in the
shader (cases 13/14). **Three GPU codec producers in this fork, three different handoff
strategies** — worth knowing before assuming one is the house style.

**LZ4 via nvcomp, where HAP uses Snappy on CPU workers.** Both are deliberate and were chosen
independently; `hap_producer.cpp:23` records the contrast explicitly.

**Slot depth is 5** and the buffers are sized from the file's own maxima
(`max_compressed`/`max_uncompressed`), logged at startup. At 12K that is the dominant VRAM cost of
the module.

---

## 4. Verification — what is measured, and what is not

| what | battery | result |
| :--- | :--- | :--- |
| Survives alternating loads against CUDA ProRes on one layer, 12K, both mixers | `producer-swap` | 12/12 swaps, no fatal line |
| Runs concurrently with three other GPU interop paths | `coexistence` | engaged 1/1 in company, 0 late frames |

**What is not covered: the picture.** Nothing compares a NotchLC-decoded frame against a reference
decoder — there is no equivalent of `prores-parity` for this format, because there is no
second decoder to compare against. The YCoCg→BGRA16 convert is our code and per-channel, which by
the ICVFX precedent is where a channel exchange would hide, and **greys are invariant under it**.

A usable substitute exists without a reference decoder: decode the same file on both mixers and
compare. That catches a mixer-side divergence but not a shared error in the kernel — worth stating,
because it is the difference between "the two agree" and "the two are right".

---

## 5. Known gaps

1. **No picture verification of any kind**, and no reference decoder to build one against. §4
   describes the partial substitute and what it cannot catch.
2. **The mid-frame host sync (step 3) is unmeasured** — it is the obvious candidate for why this
   route would scale differently from ProRes, and no battery has isolated it.
3. **No channel-count ceiling.** `playback-scaling` has no `notchlc` route, so the figure that
   exists for ProRes (8 channels at 1080p) has no counterpart here.
4. **`fill_opaque` for alpha-less material** writes 1023 per sample on the host and uploads it. At
   12K that is a full-frame transfer per frame for a constant; never profiled.

---

## 6. Related commits

| commit | why it matters |
| :--- | :--- |
| `d8d384934` | `notchlc_decode_ctx_create` leaked every prior allocation on partial failure — same defect as the ProRes decoder, found in the same audit pass |
| `3e1d02b4d` | the CUDA↔Vulkan import/release pair had no lock; this producer takes that path on the Vulkan mixer |

---

## 7. Diagrams

**Owed, and this one earns it on the strongest of the three criteria: the order is the point.** Ten
steps with a host round-trip at step 3 is precisely what prose describes badly, and the contrast
with ProRes (CPU-side slice table, no mid-frame sync) is the thing a reader most needs to see.
Operator-facing enough for a rendered PNG.
