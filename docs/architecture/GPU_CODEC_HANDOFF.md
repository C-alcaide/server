# Three GPU codec producers, three different handoffs

> **State and measurements:** [`../features/cuda-notchlc.md`](../features/cuda-notchlc.md), [`../features/cuda-prores.md`](../features/cuda-prores.md), [`../features/hap.md`](../features/hap.md)
> **Operator guide:** [`../guides/HAP_PLAYBACK.md`](../guides/HAP_PLAYBACK.md), [`../guides/CUDA_PRORES_OPERATION_GUIDE.md`](../guides/CUDA_PRORES_OPERATION_GUIDE.md), [`../guides/PIPELINE_EFFICIENCY_GUIDE.md`](../guides/PIPELINE_EFFICIENCY_GUIDE.md)
> **This document is why-it-is-shaped-this-way.** Operating instructions live in `guides/`, current state and figures in `features/`.

This fork has three producers that decode on the GPU, and **each hands the mixer something
different**. That is not drift — each choice follows from its codec — but it means there is no house
style to copy when adding a fourth, and the differences are invisible from the outside.

HAP's own routes are covered separately in
[`HAP_DECODE_ROUTES.md`](HAP_DECODE_ROUTES.md); this file is the comparison.

---

## 1. What each one hands over

| producer | codec | hands the mixer | who converts colour |
| :--- | :--- | :--- | :--- |
| **cuda_prores** | ProRes 4:2:2 on Vulkan | **three 10-bit planes** (`ycbcr`) | the mixer shader |
| **cuda_prores** | ProRes 4444, or anything on OpenGL | one packed **BGRA16** texture | the producer |
| **cuda_notchlc** | NotchLC | one packed **BGRA16** texture | the producer (CUDA kernel) |
| **hap** | HAP Q on Vulkan | the **compressed BC3** texture | the mixer shader |
| **hap** | HAP on OpenGL | resolved **RGBA** texture | the producer (own FBO pass) |

**Converting in the mixer is better where it is possible**, and for one reason: the channel's colour
management, working space and output transform all live in the mixer, so a producer-side conversion
happens *before* any of them can apply. That is the argument behind both the ProRes planar handoff
and HAP Q's compressed handoff, reached independently.

---

## 2. Why each one is what it is

**ProRes 4:2:2 → planes.** YCbCr planes are what the decoder already produces, and the mixer has a
planar `ycbcr` path. Nothing is packed or converted on the way. **4444 does not do this** because
four full-size 16-bit planes cost 8 bytes a pixel, exactly what BGRA16 costs — there is nothing to
win, and it is not free: FFmpeg decodes ProRes 4444 to `yuva444p12` while this decoder produces
10-bit, so alpha normalises differently and the straight-alpha premultiply lands a fraction low.
**The halving is a property of 4:2:2, not of planar.**

**Nothing planar on OpenGL**, either. Its interop target is a single opaque `cudaArray_t`, so three
planes would mean three `cudaGraphicsGLRegisterImage` registrations per slot — the exact call that
crashed the NVIDIA driver until a process-wide mutex serialised it.

**NotchLC → packed BGRA16.** Its kernels produce **YCoCg**, not YCbCr, and the mixer has no YCoCg
*plane* format. HAP solved the same problem the other way, by keeping YCoCg compressed and resolving
in the shader — but that only works because BC3 is a format the sampler understands. NotchLC's
YCoCg is not in a sampler-native layout, so there is nothing to hand over uncompressed.

**HAP Q → compressed BC3.** The GPU's texture unit decodes DXT natively, so the frame never exists
uncompressed anywhere. The cost is that filtering happens *before* the shader resolves YCoCg, and
`scale` is a per-texel divisor — so the picture differs from a resolve-then-filter path wherever the
raster resamples. Measured 5–16 LSB; accepted, and covered in `../features/hap.md` §4.

---

## 3. The structural difference: where the frame stops

**NotchLC returns to the host mid-frame, and the other two do not.**

```
notchlc_decode.cu, steps 1-9:
  1  upload compressed payload, host -> device
  2  nvcomp LZ4 decompress on the device
  3  SYNC, read the first 256 bytes back TO THE HOST      <- the stop
  4  parse 10 little-endian u32 header fields
  5-8 Y / UV / alpha kernels, then YCoCg -> BGRA16
  9  cudaMemcpy2DToArrayAsync into the mixer's texture
```

Step 3 is unavoidable: the section offsets live *inside* the compressed payload, so nothing
downstream can be launched until the block header has come back. ProRes parses its slice table on
the CPU **before** any upload and has no equivalent stop; HAP's demuxer likewise.

That single difference is the most likely reason NotchLC scales differently from the other two, and
**no battery has isolated it** — which is exactly why it is written here rather than assumed.

---

## 4. Decompression: three answers

| producer | decompressor | where |
| :--- | :--- | :--- |
| cuda_prores | its own entropy decode | GPU |
| cuda_notchlc | **nvcomp LZ4** | GPU |
| hap | **Snappy** | CPU worker threads, bounded queue |

HAP's choice is recorded in its own header as deliberate contrast (`hap_producer.cpp:23`). Neither
is wrong: LZ4-on-GPU keeps the CPU free, Snappy-on-CPU keeps the GPU free for the mixer. Which
matters depends on which resource the show is short of — and this fork runs both, which is why
`coexistence` exists as a battery.

---

## 5. What this means for a fourth producer

There is **no default to copy**. Ask in this order:

1. **Is the decoded layout sampler-native?** If yes, hand it over as-is and let the mixer convert —
   the colour-management argument in §1 decides it.
2. **Is it planar and subsampled?** Then planes win, but only while subsampling makes them cheaper
   than packed. Check the arithmetic; 4444 is the counter-example.
3. **Does the mixer have a pixel format for it?** NotchLC's YCoCg did not, and inventing one for a
   single producer is a larger change than converting in the kernel.
4. **On OpenGL, count your interop registrations.** One `cudaArray_t` per slot is the constraint that
   forced ProRes's asymmetry.

---

## 6. Unverified

* **No reference-decoder comparison for NotchLC or HAP.** ProRes has `prores-parity` against
  FFmpeg's CPU decoder; the other two have no second decoder to compare against.
* **NotchLC's mid-frame sync (§3) is unmeasured.**
* **`fill_opaque` for alpha-less NotchLC material** writes 1023 per sample on the host and uploads
  it — a full-frame transfer per frame for a constant, never profiled.
* **The three routes have never been run at 2160p with a screen consumer**; only 1080p.
