# DeckLink — GPU-direct SDI output

> **State:** shipped
> **Module:** `src/modules/decklink` — **2,382 lines** different from upstream across 37 files
> **Commands:** no dedicated `MIXER` command; a consumer name plus configuration
> **Architecture:** [`../architecture/DECKLINK_GPU_DIRECT_OUTPUT.md`](../architecture/DECKLINK_GPU_DIRECT_OUTPUT.md)
> **Guide:** [`../guides/DECKLINK_OUTPUT.md`](../guides/DECKLINK_OUTPUT.md)
> **Coverage:** `sdi-output`, `sdi-input`, `signalling`, `anc-check`, `decklink-input-cost`,
> `consumer-scaling`

Gets the composited frame from the mixer to the SDI card without a host round trip, on either
backend, and signals colour and HDR metadata correctly on the way out.

---

## 1. What is implemented today

**Four output strategies**, selected by what the mixer and the card support:

| strategy | fork-only file | path |
| :--- | :--- | :--- |
| CUDA↔Vulkan | `cuda_vk_strategy.{h,cpp}`, `cuda_vk_kernels.cu`, `cuda_vk_v210.cuh` | mixer texture → CUDA → V210 packed on GPU → card |
| Vulkan compute readback | `vk_readback_strategy.{h,cpp}`, `vk_readback_bgra.comp`, `vk_readback_v210.comp` | packed by a compute shader, read back directly |
| OpenGL | `ogl_gl_strategy.{h,cpp}` | the GL backend's equivalent |
| NVIDIA DVP | `dvp_support.{h,cpp}` | direct-to-video-card transfer where the hardware allows |

Supporting fork-only pieces: `gpu_output_buffer_pool.{h,cpp}` (reusing pinned output buffers),
`color_primaries.h`, and `vanc_hdr_strategy.cpp` for the HDR static-metadata VANC block.

**The V210 packing happens on the GPU** in every GPU path — `cuda_vk_v210.cuh` and
`vk_readback_v210.comp` are two implementations of the same packing, one per backend, which makes
them a parity question rather than an implementation detail.

---

## 2. Verification

| what | battery |
| :--- | :--- |
| Output pixels, including the HDR block | `sdi-output --hdr-metadata` |
| Colour signalling and HDR static metadata | `signalling` |
| Ancillary data | `anc-check` |
| Input path and its cost | `sdi-input`, `decklink-input-cost` |

**A note the numbers need:** the readback path had a red/blue exchange on the Vulkan side that a
16-bit reading initially hid, and a lingering server made that reading lie. Fixed; recorded in
`../../CHANGELOG.md`.

---

## 3. Subregion placement on the GPU

The four destination fields — `dest-x`, `dest-y`, `width`, `height` — used to coerce the consumer to
the CPU readback. The Vulkan compute readback implements them as of 2026-08-27, measured identical
to the CPU path over the SDI loopback:

| `dest-x` | CPU | Vulkan |
| :--- | ---: | ---: |
| 114 (even, 6-aligned) | 62.92 dB | **62.92 dB** |
| 115 (odd, straddles a V210 group) | 42.79 dB | **42.79 dB** |

Against a historical **7.91 dB** for this geometry on `vulkan`.

**Why it was cheap:** both shaders already iterated over *output* V210 groups, so deciding per
output pixel whether it lies inside the destination rectangle costs four push constants — and
removes the 6-pixel alignment problem entirely, because every group is computed from scratch rather
than read-modified-written. Writing every group also blacks the surround, so there is no clear pass.

**The ~20 dB at an odd `dest-x` is inherent to 4:2:2**, not a defect: both paths agree exactly, and
the signature is a precision-independent error (all three wire formats within 0.6 dB, 16-bit gaining
+0.06 dB over 8-bit instead of +3.94). Nobody had measured that about the CPU path either.

**Both GPU packers do it**, measured `subregion 100,200,640,360,114,70` over the SDI loopback:

| mode | | coercion |
| :--- | ---: | :--- |
| `auto` (resolves to `cuda`) | 62.92 dB | none |
| `cuda` | 62.92 dB | none |
| `vulkan` | 62.92 dB | none |
| `cpu` | 62.92 dB | n/a |
| `vulkan-dma` | 62.92 dB | coerced to `cpu` |

**CUDA mattered more than Vulkan here.** `create_format_strategy` resolves `auto` to CUDA first
(P1 against the Vulkan compute path's P6, from the matrix in `fedf6ce09`), so until the CUDA packer
placed it too, a **default** install with a destination rectangle was still coerced to the CPU. The
Vulkan-only version needed `auto` specially resolved to `vulkan`; with both packers placing, that
special case was deleted and the coercion is one comparison against `vulkan_dma`.

**Still coerced:** `vulkan-dma` alone — a `VkBufferImageCopy` carries one image offset and cannot
express a rectangle inside a larger frame, so there is no shader in that path to place anything.

**And the battery could not name `cuda` until 2026-08-27** — `--gpu-readback-mode` omitted it, so
the mode a default install uses was the one mode never swept. The guard meant to prevent exactly
that had hardcoded the same incomplete list; it now derives the set from `config.cpp`'s parser.

---

## 4. Known gaps

1. **Not every configuration axis is swept.** Which DeckLink options select genuinely different
   code and which the harness cannot yet reach is tracked separately — several are unreachable
   from the harness at all.
2. **The four strategies are not compared against each other on one clip.** Each is measured;
   nothing asserts they agree.
3. **`bluefish`** (27 changed lines) is the other SDI card and has no coverage or document.
