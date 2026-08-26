# DeckLink — GPU-direct SDI output

> **State:** shipped
> **Module:** `src/modules/decklink` — **2,382 lines** different from upstream across 37 files
> **Commands:** no dedicated `MIXER` command; a consumer name plus configuration
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

## 3. Known gaps

1. **Not every configuration axis is swept.** Which DeckLink options select genuinely different
   code and which the harness cannot yet reach is tracked separately — several are unreachable
   from the harness at all.
2. **The four strategies are not compared against each other on one clip.** Each is measured;
   nothing asserts they agree.
3. **`bluefish`** (27 changed lines) is the other SDI card and has no coverage or document.
