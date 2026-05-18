# CUDA ProRes — Implementation Guide

## Overview

This module provides three components sharing the same GPU ProRes encode/decode kernel pipeline:

| Component | Role |
|---|---|
| **Bypass Consumer** (`CUDA_PRORES_BYPASS`) | Real-time GPU encoder fed by DeckLink SDI V210 capture |
| **Mixer Consumer** (`CUDA_PRORES`) | Real-time GPU encoder fed by the CasparCG compositor |
| **ProRes Producer** (`PLAY … CUDA_PRORES`) | GPU-accelerated ProRes decoder for playout |

---

## Part 1 — GPU Encoder (Bypass Consumer)

### Overview

The bypass encoder accepts raw **V210** (10-bit packed 4:2:2 YCbCr) frames directly from a Blackmagic DeckLink SDI input, runs the entire encode pipeline on the GPU, and writes conformant `.mov` or `.mxf` files to disk — bypassing the CasparCG GPU mixer entirely.

**Key properties:**
- Registered as AMCP consumer `CUDA_PRORES_BYPASS`
- Supports all four 422 profiles (Proxy / LT / Standard / HQ) and ProRes 4444/4444 XQ
- Handles progressive (720p50, 1080p25/50) and interlaced signals (1080i50)
- The CasparCG **channel** format is irrelevant: auto-detects the actual SDI input format via `bmdVideoInputEnableFormatDetection` — you can run a 720p50 channel and record a 1080i50 input with no configuration change
- No dropped frames observed at 1080i50 HQ (~136 Mbps average) on RTX A5000

---

## Hardware & Software Stack

| Layer | Component |
|---|---|
| Capture card | Blackmagic DeckLink 8K Pro (BMD SDK 12.x COM interfaces) |
| GPU | NVIDIA RTX A5000 (sm_86, 24 GB VRAM) |
| Pixel format in | V210 — 10-bit 4:2:2 YCbCr, packed as 4 words per 6 pixels |
| Codec out | Apple ProRes 422 HQ — `icpf` bitstream inside QuickTime `.mov` or MXF |
| Build | NVCC (CUDA 12.x) + MSVC via CMake/Ninja, Windows only |
| Libraries | CUDA Runtime, CUB (NVIDIA's device-side primitives, part of CUDA) |
| Container | Custom minimal QuickTime `.mov` muxer + optional MXF muxer |

---

## Pipeline Overview

```
DeckLink SDI input
        │  VideoInputFrameArrived() — on DeckLink driver thread
        │  bmdFrameHasNoInputSource frames are silently dropped (pre-lock garbage)
        │  Synchronous cudaMemcpyAsync H→D + cudaStreamSynchronize
        │  (host pinned buffer is freed back to DeckLink ring immediately)
        ▼
  Device VRAM Ring (4 slots × ~22 MB)
        │  CaptureToken pushed to encode queue
        │  (token carries: d_vram ptr, width, height, is_interlaced,
        │   is_tff, timebase_num/den, SMPTE RP188 timecode)
        ▼
  Encode Thread — encode_one()
        │  [First frame only] init_encoder_and_muxer() called with
        │  actual capture dimensions from CaptureToken — NOT channel format
        │
        │  Progressive path:
        ▼  ① prores_encode_frame(ctx, d_v210, ...)
        │
        │  Interlaced path:
        ▼  ① launch_v210_unpack_field(d_v210, field_a_planes, parity_a)
           ② launch_v210_unpack_field(d_v210, ctx.d_y/cb/cr, parity_b)
           ③ prores_encode_from_yuv_fields_422(field_a, field_b, ...)

        Each encode sub-call:
        ▼  k_v210_unpack / k_v210_unpack_field    (CUDA kernel)
           Planar YUV422P10 (d_y, d_cb, d_cr)    int16_t, values [0,1023]
        ▼  k_dct_quantise  ×3 (Y, Cb, Cr)
           Quantised DCT coefficients             int16_t
        ▼  k_interleave_luma / k_interleave_chroma
           Per-slice interleaved coefficient array
        ▼  cuda_prores_enc_frame_raw (two GPU passes)
             Pass 1: k_count_bits → k_bits_to_bytes → k_compute_slice_sizes
             Scan:   CUB ExclusiveSum (device-side prefix sum)
             Pass 2: k_encode_slices
           Encoded bitstream (d_bitstream) + seek table (d_slice_offsets)
        │  cudaStreamSynchronize + D→H memcpy (pinned output buffer)
        ▼
  Host: assemble icpf box (frame header + picture header(s) + seek table + slices)
        ▼
  MovMuxer::write_video() / MxfMuxer::write_video()
  Async unbuffered file write → .mov / .mxf on NVMe
```

---

## Format Detection (Key Design)

The bypass consumer does **not** configure its encoder from the CasparCG channel format.  Instead:

1. `initialize()` stores pending encoder parameters (output path, fourcc, color info) and starts the DeckLink capture with `bmdVideoInputEnableFormatDetection` enabled.
2. DeckLink fires `VideoInputFormatChanged()` with the actual detected signal.  The callback stores `detected_interlaced_`, `detected_tff_`, `detected_timebase_num_/den_` and restarts streams in the new mode via `StopStreams → EnableVideoInput → StartStreams`.
3. Each `CaptureToken` carries `width`, `height`, `is_interlaced`, `is_tff`, `timebase_num`, `timebase_den` from the detected format.
4. On the **first frame**, `encode_one()` calls `init_encoder_and_muxer()` with the token's actual dimensions.  CUDA buffers, encoder context and the muxer file are created here with the real capture format.

This means the channel can be 720p50 while recording a 1080i50 input — the MOV will be 1920×1080 interlaced at 25 fps with correct duration and seekbar.

---

## Source Files — Encoder / Consumer

| File | Role |
|---|---|
| `consumer/prores_bypass_consumer.cu` | Main compiled bypass consumer (registered as `CUDA_PRORES_BYPASS`) |
| `consumer/prores_consumer.cu` | GPU-mixer-path consumer (compiled; separate registration) |
| `consumer/prores_consumer.cpp` | **Not compiled** (not in CMakeLists) — reference only |
| `input/decklink_capture.h/.cpp` | DeckLink capture front-end: SDI → VRAM ring, format detection, pinned pool |
| `input/cuda_pinned_allocator.h/.cpp` | Auto-growing CUDA pinned host memory pool (IDeckLinkMemoryAllocator) |
| `cuda/cuda_prores_frame.h/.cu` | `ProResFrameCtx` struct; encoder API; progressive + interlaced encode paths |
| `cuda/cuda_prores_v210_unpack.cuh` | `k_v210_unpack` (progressive); `k_v210_unpack_field` (interlaced field extraction) |
| `cuda/cuda_prores_dct_quant.cuh` | IJG Loeffler 8×8 forward DCT + ProRes quantisation kernel |
| `cuda/cuda_prores_entropy.cu` | Two-pass slice entropy encoder: bit-count → CUB prefix-sum → write |
| `cuda/cuda_prores_rice.cuh` | `BitPacker`; hybrid Rice/exp-Golomb VLC encoder (device functions) |
| `cuda/cuda_prores_tables.cuh` | Quantisation matrices, scan order tables, profile bitrate targets |
| `muxer/mov_muxer.h/.cpp` | Minimal QuickTime `.mov` muxer (unbuffered async I/O, no FFmpeg) |
| `muxer/mxf_muxer.h/.cpp` | MXF muxer for broadcast delivery |

---

## Step 1 — V210 Unpack (`cuda_prores_v210_unpack.cuh`)

### What is V210?

V210 is Blackmagic's native 10-bit 4:2:2 capture format (SMPTE 422M).  Each group of **6 pixels** is packed into **4 × 32-bit words**:

```
Word 0: [ Cr0:10 | Y0:10 | Cb0:10 ]   (bits 29:20 | 19:10 | 9:0)
Word 1: [  Y2:10 | Cb1:10 |  Y1:10 ]
Word 2: [ Cb2:10 |  Y3:10 | Cr1:10 ]
Word 3: [  Y5:10 | Cr2:10 |  Y4:10 ]
```

One CUDA thread processes one 6-pixel group.  Grid = `(width × height / 6 + 127) / 128` blocks of 128 threads.

**Output**: three separate `int16_t` planes (`d_y`, `d_cb`, `d_cr`), values in `[0, 1023]`.

**Non-multiples of 6**: widths like 1280 (720p) are not multiples of 6 (`1280 % 6 = 2`).  The kernel uses floor division `groups_per_row = width/6 = 213`, writing only complete groups.  The 2 edge pixels remain zero from the `cudaMemsetAsync` pre-zeroing in `prores_encode_frame`.  The old guard `if (width % 6 != 0) return cudaErrorInvalidValue` has been removed.

### Interlaced variant (`k_v210_unpack_field`)

For 1080i50, DeckLink delivers both fields interleaved line-by-line (full 1920×1080, lines 0/2/4/… = top field, lines 1/3/5/… = bottom field).  `k_v210_unpack_field` takes a `parity` parameter (0 = even, 1 = odd) and maps:

```
full_row = parity + field_row × 2
```

Output planes hold only `field_height = 540` rows.

---

## Step 2 — Forward DCT + Quantisation (`cuda_prores_dct_quant.cuh`)

One **CUDA thread block = one 8×8 pixel block** (64 threads per block).  Called three times per field/frame pass: luma → Cb → Cr.

Implements the **IJG Loeffler integer 8-point forward DCT**, matching FFmpeg's `ff_jpeg_fdct_islow_10` exactly.  Shared memory (`__shared__ int32_t s_block[64]`) exchanges data between the row and column passes.

DC bias: a flat 512-value 10-bit block produces DC = `512 × 32 = 0x4000` after the DCT.  The kernel subtracts this before quantising (`if (tid == 0) raw -= 0x4000`).

Quantisation and scan reorder happen in a single step per thread.

---

## Step 3 — Coefficient Interleave (`cuda_prores_frame.cu`)

ProRes entropy coding operates on **slices** (horizontal strips of macroblocks).  `k_interleave_luma` and `k_interleave_chroma` rearrange `d_coeffs_y/cb/cr` from raster order into per-slice layout `d_coeffs_slice[s][blocks_per_slice][64]`.

### mbs_per_slice fix

The AMCP `SLICES` parameter is **mbs_per_slice** (values 1/2/4/8 — powers of 2), not slices_per_row.  `slices_per_row` is derived as `(width/16) / mbs_per_slice` with ceiling division for partial last slices.  Earlier code divided again, producing non-power-of-2 values like 30 for 1920-wide streams, corrupting the `log2(mbs_per_slice) << 4` field in the picture header.

---

## Step 4 — Entropy Encoding (`cuda_prores_entropy.cu`)

Two-pass GPU schedule:

```
k_count_bits      ← 1 thread/slice, null BitPacker (count only)
k_bits_to_bytes   ← ceil division: bits → bytes per component
k_compute_sizes   ← d_sizes[s] = 6 + Y_bytes + Cb_bytes + Cr_bytes
CUB ExclusiveSum  ← byte offsets per slice (device-side prefix sum)
k_set_total       ← d_offsets[num_slices] = total bytes
k_encode_slices   ← 1 thread/slice, writes slice header + Y + Cb + Cr
```

ProRes uses **hybrid Rice / exp-Golomb VLC** with adaptive codebooks per component.

---

## Step 5 — Frame Header Assembly

After the GPU work completes and the bitstream is copied to pinned host memory, the CPU assembles the `icpf` box:

```
[4]   frame_size        big-endian uint32
[4]   'icpf'            magic
[148] frame_header
[8]   picture_header    (×2 for interlaced)
[num_slices × 2]  seek_table   big-endian uint16 per-slice byte counts
[variable]  slice_data
```

**Critical bit positions** (common source of bugs):
- `frame_type` (0=progressive, 1=TFF, 2=BFF) lives at bits **[3:2]** of frame_flags byte 12 — `value << 2`, not `value`.
- Picture header byte 7: `log2(mbs_per_slice) << 4` — requires mbs_per_slice to be a power of 2.

---

## Interlaced Encoding (1080i50)

`prores_bypass_consumer.cu` runs the pipeline **twice** per logical output frame — once per field — then assembles a single `icpf` box with two picture headers:

```
Field 0 (temporal first — top for TFF, bottom for BFF):
  launch_v210_unpack_field(d_v210, d_field_a_y/cb/cr, parity_a)
  prores_encode_from_yuv_fields call stores field A bitstream

Field 1 (temporal second):
  launch_v210_unpack_field(d_v210, ctx.d_y/cb/cr, parity_b)
  prores_encode_from_yuv_fields_422(...) assembles icpf with both fields
```

### Geometry for 1080i50

| Parameter | Value | Derivation |
|---|---|---|
| `field_height` | 540 | 1080 / 2 |
| MB rows per field | 34 | ceil(540 / 16) |
| DCT block rows (Y) | 68 | ceil(540 / 8) — ceiling needed because 540 % 8 = 4 |
| Slices per field | 510 | 34 × 15 (15 = 1920/16/8 slices/row) |

---

## Host-Side Capture Pipeline (`input/decklink_capture.cpp`)

### VRAM ring

- 4 VRAM slots pre-allocated at first frame (`init_vram_ring`), sized to the actual captured frame.
- Each `VideoInputFrameArrived` selects the next ring slot round-robin, does **synchronous** `cudaMemcpyAsync + cudaStreamSynchronize`, then returns.  The host pinned buffer is released back to DeckLink immediately — no `AddRef` needed.

### Pinned allocator (`cuda_pinned_allocator.cpp`)

- **Auto-growing pool**: `AllocateBuffer` pops from a free list; if empty it calls `cudaMallocHost` once and adds the new slot to `all_` for destruction cleanup.  This handles DeckLink's pre-allocation ring (which calls `AllocateBuffer` many more times than the initial `pool_size`) without ever falling back to slow per-frame allocation.
- All slots freed via `cudaFreeHost` in the destructor.

### Pre-lock frame filtering

Frames with `bmdFrameHasNoInputSource` are silently dropped in `VideoInputFrameArrived`.  These are the black/garbage frames DeckLink delivers before it locks onto the SDI signal, and must not reach the encoder.

---

## QuickTime MOV Muxer (`muxer/mov_muxer.cpp`)

- Uses `FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED` for direct unbuffered async I/O to NVMe
- `co64` 64-bit chunk offsets (multi-minute 4K HQ files exceed 4 GB)
- `colr` atom: QuickTime `nclc` box, carrying `color_primaries`, `transfer_function`, `color_matrix`
- `fiel` atom: `fields=2, detail=9` (TFF) when `is_interlaced=true`
- Optional `mdcv` + `clli` atoms for HDR mastering display metadata (SMPTE ST 2086)
- `tmcd` track: SMPTE RP188 timecode extracted from SDI ancillary data via `GetTimecode(bmdTimecodeRP188Any, ...)`

### Duration fix

`timebase_num` (e.g. 1000 for 1080i50/25fps in CasparCG) must be multiplied into all duration fields.  Five locations: `mvhd.duration`, video `tkhd.duration`, video `mdhd.duration`, audio `tkhd.duration` (converted to movie timescale), TC `tkhd.duration`.  Without this, VLC shows the entire clip as already at 100% from the start.

---

## Diagnostics Graph (`common/diagnostics/graph.h`)

Registered in the `prores_bypass_consumer_impl` constructor.  Four channels:

| Channel | Colour | Meaning |
|---|---|---|
| `encode-time` | Green | Encode duration / frame interval (1.0 = real-time) |
| `queue-depth` | Orange | Frame queue fill level (0–1) |
| `dropped-frame` | Red tag | Frame dropped due to queue overflow |
| `encode-error` | Pink tag | CUDA encode failure |

The graph title updates every frame:
```
cuda_prores_bypass[2|dev1] | 250 fr (10.0s)
```

**Important**: elapsed time uses `capture_hz_` (set from `token.timebase_den / token.timebase_num` on first frame), not `format_desc_.hz`.  For interlaced, the CasparCG channel hz may differ from the actual capture frame rate (e.g. 720p50 channel → `hz=50`, but 1080i50 input → `capture_hz_=25`), which would show half the real elapsed time.

---

## Profiles and Bitrates

| Profile | FourCC | Approx. Mbps @ 1080i50 | `q_scale` default |
|---|---|---|---|
| Proxy | `apco` | ~18 | 8 |
| LT | `apcs` | ~45 | 8 |
| Standard | `apcn` | ~90 | 8 |
| HQ | `apch` | ~136 | 8 |
| 4444 | `ap4h` | ~220 | 8 |

---

## AMCP Usage

```
ADD 1 CUDA_PRORES_BYPASS [PATH "D:/recordings"] [DEVICE 1] [FILENAME "clip.mov"]
                         [PROFILE 3] [QSCALE 8] [SLICES 4] [MXF]
REMOVE 1 CUDA_PRORES_BYPASS
```

| Parameter | Default | Notes |
|---|---|---|
| `PATH` | current dir | Output directory (forward slashes) |
| `FILENAME` | auto-timestamped | e.g. `prores_20260313_153000.mov` |
| `DEVICE` | 1 | DeckLink device (1-based) |
| `PROFILE` | 3 | 0=Proxy 1=LT 2=Standard 3=HQ 4=4444 5=4444XQ |
| `QSCALE` | 8 | Quantisation scale 1–31 (1=best quality) |
| `SLICES` | 4 | mbs_per_slice: 1/2/4/8 (must be power of 2) |
| `MXF` | (absent) | Write MXF instead of MOV |

The channel format only provides the initial DeckLink mode hint; the actual SDI input format is always auto-detected.

---

## Building

```bat
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
cmake --build D:\Github\CasparCG-cuda\out\build\x64-RelWithDebInfo -j4
```

Output: `D:\Github\CasparCG-cuda\out\build\x64-RelWithDebInfo\shell\casparcg.exe`

---

## Key Bugs Fixed (History)

| Bug | Symptom | Root cause | Fix |
|---|---|---|---|
| `mbs_per_slice` corruption | `unsupported slice resolution: 16x1` | `SLICES` param is mbs_per_slice; old code divided again → non-power-of-2 | Direct assignment + ceiling slices_per_row derivation |
| Pool exhaustion spam | `[CudaPinnedAllocator] Pool exhausted (x500…)` | DeckLink pre-allocates its ring up-front, exceeding fixed pool size | Auto-growing pool: grow by 1 slot on demand, reused forever |
| Sync H→D copy | Pool exhaustion continued even at pool=32 | `AddRef` held frames alive; DeckLink needed more ring slots | Synchronous `cudaStreamSynchronize` in callback; no `AddRef` needed |
| Pre-lock black frames | First 5–6 frames are black | DeckLink delivers pre-lock garbage with `bmdFrameHasNoInputSource` | Drop frames with that flag before encoding |
| MOV duration wrong | VLC seekbar at 100% from frame 1 | `timebase_num` (1000) not multiplied into duration fields | Multiplied in 5 muxer locations |
| Wrong format when channel ≠ input | Black video / green artifacts | Encoder sized to CasparCG channel format, not actual SDI input | Lazy first-frame init from CaptureToken dimensions; `bmdVideoInputEnableFormatDetection` |
| Elapsed time half actual | Diag showed 35s for 1:10 clip | `format_desc_.hz` = 50 (channel) instead of 25 (1080i25 capture) | Use `capture_hz_` = `tb_den/tb_num` from first token |
| 720p `cudaErrorInvalidValue` | Every frame fails with `invalid argument` | `launch_v210_unpack` guards `width % 6 != 0`; 1280 % 6 = 2 | Removed guards; kernel floor-divides safely; edge pixels pre-zeroed |

**Key properties:**
- Supports all four 422 profiles (Proxy / LT / Standard / HQ) and ProRes 4444/4444 XQ
- Handles progressive (1080p25/50) and interlaced signals (1080i50)
- Multi-device: each DeckLink card gets its own CUDA stream, ring buffer, and encoder context
- No dropped frames observed at 1080i50 HQ (~136 Mbps average) on RTX A5000

---

## Hardware & Software Stack

| Layer | Component |
|---|---|
| Capture card | Blackmagic DeckLink 8K Pro (BMD SDK COM interfaces) |
| GPU | NVIDIA RTX A5000 (sm_86, 24 GB VRAM) |
| Pixel format in | V210 — 10-bit 4:2:2 YCbCr, packed as 4 words per 6 pixels |
| Codec out | Apple ProRes 422 HQ — `icpf` bitstream inside QuickTime `.mov` |
| Build | NVCC (CUDA 12.x) + MSVC via CMake/Ninja, Windows only |
| Libraries | CUDA Runtime, CUB (NVIDIA's device-side primitives, part of CUDA) |
| Container | Custom minimal QuickTime `.mov` muxer (no FFmpeg dependency at runtime) |

---

## Pipeline Overview

```
DeckLink card (hardware)
        │  V210 frame via COM callback (memcpy to pinned ring slot)
        ▼
  Pinned Host Ring Buffer  (8 slots × 5.5 MB each)
        │  cudaMemcpyAsync H→D
        ▼
  Device V210 Buffer (d_v210)
        │
        ▼  ① k_v210_unpack / k_v210_unpack_field
  Planar YUV422P10 (d_y, d_cb, d_cr)   int16_t, 10-bit values [0,1023]
        │
        ▼  ② k_dct_quantise  (×3: Y, Cb, Cr)
  Quantised DCT Coefficients (d_coeffs_y/cb/cr)   int16_t
        │
        ▼  ③ k_interleave_luma / k_interleave_chroma
  Per-Slice Interleaved Coefficient Array (d_coeffs_slice)
        │
        ▼  ④ cuda_prores_enc_frame_raw  (two passes)
             Pass 1: k_count_bits → k_bits_to_bytes → k_compute_slice_sizes
             Scan:   CUB ExclusiveSum → per-slice byte offsets
             Pass 2: k_encode_slices
  Encoded Slice Bitstream (d_bitstream)  +  Seek Table (d_slice_offsets)
        │  cudaStreamSynchronize + cudaMemcpyAsync D→H (pinned)
        ▼
  Host: assemble icpf box  (frame header + picture header + seek table + slice data)
        │
        ▼
  MovMuxer::write_video()
  Async file write → .mov on NVMe
```

For **interlaced** signals (1080i50), the pipeline runs **twice** per logical output frame — once for the top field (even lines) and once for the bottom field (odd lines) — then both encoded fields are assembled into a single `icpf` box containing two picture headers.

---

## Source Files

| File | Role |
|---|---|
| `cuda/cuda_prores_frame.h` | `ProResFrameCtx` struct; encoder API declarations |
| `cuda/cuda_prores_frame.cu` | Top-level orchestration; `build_frame_header`; `build_picture_header`; `k_interleave_luma/chroma` kernels; interlaced and progressive encode paths |
| `cuda/cuda_prores_v210_unpack.cuh` | `k_v210_unpack` (progressive); `k_v210_unpack_field` (interlaced field extraction) |
| `cuda/cuda_prores_dct_quant.cuh` | IJG Loeffler 8×8 forward DCT + ProRes quantisation kernel |
| `cuda/cuda_prores_entropy.cu` | Two-pass slice entropy encoder: bit-count pass → CUB prefix-sum → write pass |
| `cuda/cuda_prores_rice.cuh` | `BitPacker`; hybrid Rice/exp-Golomb VLC encoder (device functions) |
| `cuda/cuda_prores_tables.cuh` | Quantisation matrices, scan order tables, profile bitrate targets |
| `cuda/cuda_bgra_to_yuva444p10.cuh` | Optional BGRA→YUVA444P10 conversion for ProRes 4444 path |
| `muxer/mov_muxer.h/.cpp` | Minimal QuickTime `.mov` muxer: `ftyp`, `mdat`, `moov/trak/stbl` |
| `test/decklink_prores_capture.cpp` | CLI capture app: DeckLink COM integration, ring buffer, encoder threads, stats |

---

## Step 1 — V210 Unpack (`cuda_prores_v210_unpack.cuh`)

### What is V210?

V210 is Blackmagic's native 10-bit 4:2:2 capture format (SMPTE 422M).  Each group of **6 pixels** is packed into **4 × 32-bit words**:

```
Word 0: [ Cr0:10 | Y0:10 | Cb0:10 ]   (bits 29:20 | 19:10 | 9:0)
Word 1: [  Y2:10 | Cb1:10 |  Y1:10 ]
Word 2: [ Cb2:10 |  Y3:10 | Cr1:10 ]
Word 3: [  Y5:10 | Cr2:10 |  Y4:10 ]
```

One CUDA thread processes one 6-pixel group.  Grid = `(width × height / 6 + 127) / 128` blocks of 128 threads.

**Output**: three separate `int16_t` planes (`d_y`, `d_cb`, `d_cr`), values in `[0, 1023]`. No DC level shift is applied here — that is deferred to the DCT quantise step.

### Interlaced variant (`k_v210_unpack_field`)

For 1080i50, the DeckLink SDK delivers both fields interleaved line-by-line (full 1920×1080 frame, lines 0/2/4/… = top field, lines 1/3/5/… = bottom field).  `k_v210_unpack_field` takes a `field` parameter (0 or 1) and maps:

```
full_row = field + field_row × 2
```

Output planes hold only `field_height = 540` rows.  The same thread count formula applies using `field_height` instead of full height.

---

## Step 2 — Forward DCT + Quantisation (`cuda_prores_dct_quant.cuh`)

### Kernel mapping

One **CUDA thread block = one 8×8 pixel block** (64 threads per block).  Grid = `(plane_width/8, ceil(plane_height/8))` blocks.  Called three times per field/frame pass: luma → Cb → Cr.

### DCT algorithm

Implements the **IJG Loeffler integer 8-point forward DCT**, matching FFmpeg's `ff_jpeg_fdct_islow_10` exactly.  The computation is two-pass:

1. **Row pass** (`dct_row_pass`): 8 threads, each handling one row. Uses `PASS1_BITS=1`, `CONST_BITS=13`. Even/odd butterfly decomposition with fixed-point multiplications. Output scaled ×2 for DC and DESCALEd for AC.
2. **Column pass** (`dct_col_pass`): 8 threads, each handling one column. `DESCALE` shift = `CONST_BITS + OUT_SHIFT = 15`. Output fits in `int16_t`.

Shared memory (`__shared__ int32_t s_block[64]`) is used to exchange data between the two passes.

### DC bias

A flat 512-value (mid-grey) 10-bit block produces DC = `512 × 32 = 0x4000` after the DCT.  FFmpeg's `encode_dcs` subtracts this bias before quantising.  The kernel handles this:

```cuda
if (tid == 0) raw -= 0x4000;  // tid==0 = DC coefficient position
```

### Quantisation and scan reorder

**In a single step**, each thread:
1. Looks up the ProRes scan order for its coefficient position (`c_scan_order` or `c_scan_order_interlaced` for interlaced signals)
2. Reads the coefficient from `s_block[scan_order[tid]]`
3. Divides by `q_table[profile][tid] × q_scale` (rounding half-away-from-zero)
4. Clamps to `int16_t`

The quantisation matrices for each profile live in `__constant__` memory (`c_quant_luma`, `c_quant_chroma`), providing very fast access.

`q_scale` is a global adaptive scalar in range `[1, 31]`; lower = better quality / more bits. The capture app uses a fixed `q_scale=8`.

---

## Step 3 — Coefficient Interleave (`cuda_prores_frame.cu`)

ProRes entropy coding operates on **slices**, not full planes.  A slice is a horizontal strip of macroblocks.  For HQ at 1080p, with `mbs_per_slice=8`:

- Each 16×16 ProRes macroblock → 4Y + 2Cb + 2Cr = 8 blocks of 64 coefficients
- One slice = 8 MBs → 64Y + 16Cb + 16Cr blocks = 96 × 64 `int16_t` values

`d_coeffs_y/cb/cr` hold blocks in row-major raster order (how the DCT produced them). `k_interleave_luma` and `k_interleave_chroma` rearrange them into the per-slice layout `d_coeffs_slice[s][blocks_per_slice][64]` needed by the entropy coder.

One thread = one 8×8 block.  The block's slice index and position within the slice are derived from its `(blk_row, blk_col)` coordinates.

---

## Step 4 — Entropy Encoding (`cuda_prores_entropy.cu`)

ProRes uses a two-pass approach because each slice's data must be preceded by a two-byte size field, requiring the encoder to know the encoded size before writing.  The entire two-pass operation runs on the GPU.

### Coding algorithm

ProRes uses a **hybrid Rice / exp-Golomb VLC**, with adaptive codebook selection:

- **DC coefficients**: DPCM-predicted delta between consecutive blocks' DCs within the slice, then mapped to unsigned with sign embedding: `code = 2×val ^ sign`. The first block uses a fixed codebook (`FIRST_DC_CB`); subsequent use adaptive codebooks from `c_dc_codebook[7]`.
- **AC coefficients**: Run-Level coding across scan positions. For each non-zero `(run, |level|−1)` pair, two separate adaptive codebooks from `c_run_to_cb[16]` and `c_level_to_cb[10]` select the VLC table. A separate sign bit follows. Trailing zeros are not coded.

The `BitPacker` struct provides MSB-first bit packing via a `uint64_t` accumulator, flushing bytes as they fill.

### Two-pass GPU schedule

```
k_count_bits      ← 1 thread/slice, null BitPacker (count only)
k_bits_to_bytes   ← ceil division: bits → bytes per component
k_compute_sizes   ← d_sizes[s] = 6 + Y_bytes + Cb_bytes + Cr_bytes
CUB ExclusiveSum  ← byte offsets per slice (device-side prefix sum)
k_set_total       ← d_offsets[num_slices] = total bytes
k_encode_slices   ← 1 thread/slice, writes slice header + Y + Cb + Cr
```

**`CUB ExclusiveSum`** (from NVIDIA's CUB library) computes the prefix sum entirely on the GPU in a single asynchronous call, avoiding a CPU round-trip.

### Slice format (ProRes 422)

```
Byte 0:    0x30         ← header size in bits (6 bytes × 8 = 48 = 0x30)
Byte 1:    q_scale
Bytes 2-3: Y_size       ← big-endian uint16
Bytes 4-5: Cb_size      ← big-endian uint16
[Y data]   Y_size bytes
[Cb data]  Cb_size bytes
[Cr data]  (total - 6 - Y_size - Cb_size) bytes  ← implicit size
```

---

## Step 5 — Frame Header Assembly (host-side, `cuda_prores_frame.cu`)

After GPU work completes and the bitstream is memcpy'd to the pinned host buffer, the CPU assembles the full ProRes frame container:

### `icpf` box layout (progressive)

```
[4]  frame_size        big-endian uint32, total frame bytes including this field
[4]  'icpf'            magic
[148] frame_header     (see below)
[8]  picture_header    for the single picture
[num_slices × 2]  seek_table   big-endian uint16 per-slice raw byte sizes
[variable]  slice_data         concatenated encoded slices
```

### Frame header (148 bytes)

```
[0-1]   header_size    = 148
[2-3]   version        = 0
[4-7]   encoder_id     = 'CUDA'
[8-9]   width
[10-11] height
[12]    frame_flags:
          bits [7:6] = chroma format (0b10 = 4:2:2, 0b00 = 4:4:4)
          bits [3:2] = frame_type   (0 = progressive, 1 = TFF interlaced, 2 = BFF)
[13]    reserved
[14]    color_primaries    (1 = Rec.709, 9 = BT.2020)
[15]    transfer_function  (1 = Rec.709, 14 = HLG, 16 = PQ)
[16]    color_matrix       (1 = Rec.709, 9 = BT.2020-NCL)
[17]    alpha_channel_type (0 = none, 2 = 16-bit for 4444)
[18]    reserved
[19]    matrix_flags = 0x03  (both luma + chroma quant matrices present)
[20-83] luma quantisation matrix   (64 bytes)
[84-147] chroma quantisation matrix (64 bytes)
```

> **Critical**: `frame_type` lives at bits **[3:2]** of byte 12 (i.e. `value << 2`), NOT bits [1:0].  FFmpeg's `proresdec2.c` reads it as `(buf[12] >> 2) & 3`.

### Picture header (8 bytes)

```
[0]    hdr_size_bits = 0x40  (= 8 bytes × 8 bits)
[1-4]  picture_data_size     big-endian uint32, total bytes from [0] to end of slice data
[5-6]  num_slices            big-endian uint16
[7]    log2(mbs_per_slice) << 4
```

### Seek table

One `uint16_t` per slice storing the **raw byte count** of that slice's encoded data (including its 6-byte slice header). FFmpeg's decoder uses these directly as byte offsets — no shift needed.

---

## Interlaced Encoding (1080i50)

When `ctx->is_interlaced = true`, `prores_encode_frame` runs the pipeline **twice in one call**:

```
Field 0 (top / even lines):
  k_v210_unpack_field(..., field=0)  →  540-line planar buffer
  DCT + quant (3 planes)
  k_interleave_luma / k_interleave_chroma
  cuda_prores_enc_frame_raw  (510 slices)
  cudaStreamSynchronize
  cudaMallocHost + cudaMemcpy  →  save field 0 bitstream to pinned host RAM

Field 1 (bottom / odd lines):
  k_v210_unpack_field(..., field=1)  →  540-line planar buffer
  DCT + quant (3 planes)
  k_interleave_luma / k_interleave_chroma
  cuda_prores_enc_frame_raw  (510 slices)
  cudaStreamSynchronize

Assemble icpf:
  [frame_size][icpf][frame_hdr(TFF)][pic_hdr_0][seek_0][data_0][pic_hdr_1][seek_1][data_1]
```

### Geometry for 1080i50

| Parameter | Value | Derivation |
|---|---|---|
| `field_height` | 540 | 1080 / 2 |
| MB rows per field | 34 | ceil(540 / 16) |
| Slices per field | 510 | 34 × 15 (15 = 1920/16/8 slices/row) |
| DCT block rows (Y) | 68 | ceil(540 / 8) — ceiling needed because 540 % 8 = 4 |
| luma DCT blocks | 16 320 | 68 × 240 |
| chroma DCT blocks | 8 160 | 68 × 120 |

The DCT launcher uses ceiling division `(plane_height + 7) / 8` so the partial last block row (covering lines 536–539, zero-padded by the kernel's bounds check) is included.

---

## Ring Buffer and Threading Model

Each DeckLink device runs two threads plus the DeckLink capture callback:

```
DeckLink driver thread
  └─ CaptureCallback::VideoInputFrameArrived()
       memcpy V210 → ring_buf[write_count % kRingN]
       write_count++; ring_cv.notify_one()

Encoder thread (per device)
  └─ while(not stopped):
       wait on ring_cv
       for each available slot:
         cudaMemcpyAsync ring_buf[slot] → d_v210  (H→D)
         prores_encode_frame(...)
         MovMuxer::write_video(...)
         read_count++

Stats thread (shared)
  └─ every 10 s: print fps, Mbps, dropped for each device
```

**Ring size** = 8 slots × ~5.5 MB = ~44 MB pinned memory per device.  If the GPU encoder falls 8 frames behind, the next arrival is dropped and noted.

The encoder thread calls `prores_encode_frame` **synchronously** — the function internally does `cudaStreamSynchronize` after each field to retrieve the seek table offsets before assembling the bitstream.  Frame N+1's H→D copy (`cudaMemcpyAsync`) starts on the stream while the previous frame's seek table copy runs.

---

## QuickTime MOV Muxer (`muxer/mov_muxer.cpp`)

A self-contained muxer written from scratch (no libavformat dependency):

- Uses `FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED` for direct, un-buffered async I/O to NVMe
- Writes `ftyp` → `mdat` (streaming, no seek) → `moov` (flushed at `close()`)
- `co64` chunk offsets (64-bit) since multi-minute 4K HQ files exceed 4 GB
- `colr` atom: `nclc` (QuickTime color info), carrying `color_primaries`, `transfer_function`, `color_matrix`
- `fiel` atom: added when `is_interlaced=true` — `fields=2, detail=9` (TFF dominant). This makes FFprobe and QuickTime correctly report `field_order=tb`
- Optional `mdcv` + `clli` atoms for HDR mastering display metadata (SMPTE ST 2086)
- Optional `tmcd` track for SMPTE timecode

### moov atom structure

```
moov
├── mvhd
└── trak (video)
    ├── tkhd
    └── mdia
        ├── mdhd
        ├── hdlr  (vide)
        └── minf
            ├── vmhd
            ├── dinf → dref
            └── stbl
                ├── stsd
                │   └── apch (ProRes HQ sample entry)
                │       ├── colr (nclc)
                │       ├── fiel  ← interlaced signal
                │       └── mdcv / clli  ← HDR only
                ├── stts  (1 entry: all samples = 1 frame duration)
                ├── stsc  (1 sample/chunk)
                ├── stsz  (per-frame byte sizes)
                └── co64  (per-frame file offsets)
```

---

## Profiles and Bitrates

| Profile | FourCC | Approx. Mbps @ 1080i50 | `q_scale` typical |
|---|---|---|---|
| Proxy | `apco` | ~18 | 24 |
| LT | `apcl` | ~45 | 12 |
| Standard | `apcn` | ~90 | 8 |
| HQ | `apch` | ~136 | 8 |
| 4444 | `ap4h` | ~220 | 6 |

The `q_scale` is currently fixed.  The codebase has scaffolding for adaptive q_scale (binary search between passes) but the capture app uses a hardcoded value.

---

## Key Implementation Notes and Bugs Fixed

### 1. Seek table convention
FFmpeg's `proresdec` reads each seek table entry as a **raw byte count**.  An earlier version wrote `sz >> 1` (half-bytes), causing every seek offset to be wrong.

### 2. `frame_type` bit position
`frame_type` (0=progressive, 1=TFF, 2=BFF) lives at bits **[3:2]** of frame_flags byte 12.  Writing it to bits [1:0] causes the decoder to see `frame_type=0` (progressive) and fail with "wrong slice data size" on every interlaced frame.

### 3. DCT ceiling division
For `field_height=540`, `540/8 = 67` block rows — missing the last 4 lines.  The DCT launcher uses `(plane_height+7)/8` so the partial 68th row is processed (zero-padded by the kernel's bounds check).

### 4. `fiel` atom for container signalling
Without a `fiel` QuickTime atom, players report `field_order=unknown`.  The muxer now writes `fiel{fields=2, detail=9}` (TFF) when `is_interlaced=true`.

---

## Building

```bat
cd tests_standalone
cmake -G "Ninja" -B build -DCMAKE_BUILD_TYPE=Release
cd build
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
ninja decklink_prores_capture
```

Requires:
- CUDA Toolkit 12.x
- Visual Studio 2026 (or 2022) with MSVC
- Blackmagic DeckLink Desktop Video SDK (headers + `DeckLinkAPI_i.c`)
- No other external dependencies (CUB ships with CUDA Toolkit)

---

## Running

```bat
decklink_prores_capture.exe [options]

  --count    N          capture from first N DeckLink devices
  --devices  "A,B"      capture from named devices (partial name match)
  --profile  hq         proxy | lt | std | hq  (default: hq)
  --duration 120        capture duration in seconds (default: 240)
  --outdir   D:\clips   output directory (default: current directory)
```

Output filename: `d{N}_{DeviceName}_{profile}.mov`  
Example: `d1_DeckLink_8K_Pro_1_hq.mov`

---

## Part 2 — CUDA ProRes Producer (Decoder)

### Overview

The ProRes producer (`prores_producer.cpp`) is a `core::frame_producer` that decodes ProRes
files on the GPU and feeds frames into the CasparCG compositor.  It uses the same CUDA decode
kernels as the encoder but in reverse, and delivers frames via the zero-copy CUDA-GL interop
path (Windows) or a pinned host-copy fallback.

**Key properties:**
- Registered as AMCP producer keyword `CUDA_PRORES`
- Activated by: `PLAY 1-1 CUDA_PRORES <filename>`
- Supports MOV, MXF, MKV, MP4 containers via libavformat demuxing
- Zero-copy CUDA→GL interop path via shared WGL context (avoids PCIe D→H→D round-trip)
- Non-blocking seek via `seek_request_` atomic flag consumed in `read_loop()`
- IN/OUT/LENGTH frame range limiting; loop via `seek_to_frame()` (no demuxer reopen)
- Color matrix per-frame metadata + optional AMCP override (BT.709 / BT.601 / BT.2020)

---

### Source Files — Producer

| File | Role |
|---|---|
| `producer/prores_producer.h` | Public API: `register_prores_producer()` declaration |
| `producer/prores_producer.cpp` | `prores_producer_impl` struct; `create_prores_producer()` factory; `register_prores_producer()` |
| `producer/prores_demuxer.h` | `ProResDemuxer` class; `ProResFrameInfo`; `ProResPacket`; `seek_to_frame()` declaration |
| `producer/prores_demuxer.cpp` | libavformat-based container reader; `seek_to_frame()` using `av_rescale_q`; audio decode |
| `cuda/cuda_prores_decode.h/.cu` | `ProResDecodeCtx`; `prores_decode_frame()` (zero-copy); `prores_decode_frame_to_host()` |
| `util/cuda_gl_texture.h` | `CudaGLTexture`: wraps `cudaGraphicsGLRegisterImage`, `map()`, `unmap()` |

---

### Pipeline

```
libavformat (ProResDemuxer::read_packet())
        │  ProResPacket {data=icpf bytes, audio_samples}
        ▼
read_loop() [on read_thread_]
        │
        │  [seek_request_ >= 0] → demuxer_->seek_to_frame(target)
        │                          flush ready_queue_, reset counters
        ▼
  prores_decode_frame / prores_decode_frame_to_host
        │  Windows zero-copy path:
        │    CudaGLTexture::map() → cudaArray
        │    CUDA kernels write BGRA16 directly into GL texture
        │    CudaGLTexture::unmap()
        │    pixel_format::rgba  (no extra swizzle — correct BGRA convention)
        │
        │  Host-copy fallback:
        │    CUDA kernels write BGRA16 to pinned h_bgra16_[slot]
        │    frame_factory_->create_frame() + memcpy
        │    pixel_format::bgra  (.bgra swizzle matches PBO upload format)
        ▼
  ready_queue_  (MAX_QUEUED=2 slots, 5 decode slots total)
        ▼
receive_impl() [on CasparCG output thread]
        │  video_field::b → return cached_frame_  (25p on 50i field dedup)
        │  video_field::a → pop from ready_queue_, cache, return
        │  Updates DIAG frame-time, graph title, fps counter
        ▼
  core::draw_frame → CasparCG compositor
```

---

### ProResDemuxer — `seek_to_frame()`

```cpp
bool ProResDemuxer::impl::seek_to_frame(int64_t frame_number) {
    AVStream* st = video_stream;
    int64_t pts = av_rescale_q(frame_number,
                               av_inv_q(st->avg_frame_rate),
                               st->time_base);
    audio_buf_.clear();
    avcodec_flush_buffers(audio_ctx_);  // prevent stale samples after seek
    return av_seek_frame(fmt_ctx_, video_stream_idx, pts,
                         AVSEEK_FLAG_BACKWARD) >= 0;
}
```

`av_rescale_q` converts a 0-based frame index to a PTS value in the video stream's time base.
`AVSEEK_FLAG_BACKWARD` ensures the seek lands on or before the target frame; `read_packet()`
skips any stale non-video packets that follow.

---

### `seek_request_` Atomic — Non-Blocking Seek

The read loop uses a priority check at the top of each iteration:

```cpp
while (true) {
    std::unique_lock lk(queue_mutex_);
    queue_cv_.wait(lk, [this] {
        return stop_flag_ || ready_queue_.size() < MAX_QUEUED || seek_request_ >= 0;
    });
    const int64_t seek_target = seek_request_.exchange(-1LL);
    if (seek_target >= 0) {
        while (!ready_queue_.empty()) ready_queue_.pop();  // flush stale frames
        stop_flag_ = false;     // cancel any EOF stop
        lk.unlock(); queue_cv_.notify_all();
        demuxer_->seek_to_frame(seek_target);
        frame_count_ = video_frame_count = seek_target;
        // reset audio/fps counters
        continue;
    }
    if (stop_flag_) break;
}
```

`call()` resolves the target frame and wakes the loop:
```cpp
// call() target resolution (seek):
if      (val == L"rel" || val == L"current")   target = frame_count_.load();
else if (val == L"start" || val == L"in")      target = 0;
else if (val == L"end")                        target = total_frames_ - 1;
else                                           target = lexical_cast<int64_t>(val);

// Optional relative offset (params[2]): CALL 1-10 seek rel +25 / -25
if (params.size() > 2) target += lexical_cast<int64_t>(params[2]);

target = clamp(target, 0, total_frames_ - 1);
seek_request_ = target;
queue_cv_.notify_one();
```

The `rel` / `current` target is just `frame_count_` (the last frame handed to `receive_impl`); adding a positive or negative offset then gives you ±N-frame jog without needing to know the absolute position.

This ensures a seek takes effect within one frame interval regardless of queue state or EOF.

---

### IN / OUT / LENGTH — Frame Range

Set at open time from PLAY parameters.  Applied in two places:

**1. Startup seek (before read thread starts):**
```cpp
if (in_frame_ > 0) {
    demuxer_->seek_to_frame(in_frame_);
    frame_count_ = in_frame_;
}
read_thread_ = std::thread([this] { read_loop(); });
```

**2. Per-frame OUT check (in read_loop after each decoded frame is pushed):**
```cpp
++video_frame_count;
if (out_frame_ >= 0 && video_frame_count >= out_frame_) {
    if (loop_) {
        demuxer_->seek_to_frame(in_frame_);
        video_frame_count = frame_count_ = in_frame_;
        // reset audio/fps counters
    } else {
        stop_flag_ = true;
    }
    queue_cv_.notify_all();
}
```

**3. EOF loop (also seeks rather than reopening demuxer):**
```cpp
if (pkt.is_eof) {
    if (loop_) {
        demuxer_->seek_to_frame(in_frame_);
        video_frame_count = frame_count_ = in_frame_;
    } else {
        stop_flag_ = true;
    }
}
```

`LENGTH` is converted at parse time: `out_frame_ = in_frame_ + length_param`.

---

### Color Matrix Handling

ProRes encodes the color matrix index in the frame header (byte 16):
`1`=BT.709, `5`/`6`=BT.601, `9`=BT.2020-NCL, `0`=unspecified.

The effective matrix passed to the CUDA decode kernels:
```cpp
const int cm = (color_matrix_override_ >= 0) ? color_matrix_override_
                                              : (int)fi.color_matrix;
prores_decode_frame(&ctx, pkt.data.data(), pkt.size, cm, ...);
```

The `COLOR_MATRIX` AMCP parameter maps:
`709`/`BT709` → `1`, `601`/`BT601` → `6`, `2020`/`BT2020` → `9`, `AUTO` → `-1` (per-frame
metadata).  Matrix `0` (unspecified) and other unknown values are treated as BT.709 by the
decode kernel.

---

### R/B Channel Swap — Zero-Copy Path

The CUDA decode kernels write raw BGRA16 bytes (B at offset 0, G at 1, R at 2, A at 3).
When transferred via `cudaMemcpy2DToArrayAsync` into a `GL_RGBA16` cudaArray, OpenGL stores
the values verbatim: R-slot = B_val, G-slot = G_val, B-slot = R_val.

When the image mixer samples this texture it reads `(r, g, b, a)` = `(B_val, G_val, R_val,
A_val)` — the BGRA convention that CasparCG expects.  Using `pixel_format::rgba` (identity
swizzle) passes this through unchanged.  Using `pixel_format::bgra` would apply an *extra*
R↔B swap, producing wrong colours.

The **host-copy path** works differently: `glTextureSubImage2D` is called with
`FORMAT=GL_BGRA`, so OpenGL swaps B↔R when storing to `GL_RGBA16`.  The texture then holds
correct `(R_val, G_val, B_val, A_val)` in its slots, and `pixel_format::bgra` (`.bgra`
swizzle) correctly re-swaps back to BGRA convention for the pipeline.

---

### DIAG Graph

| Track | Formula | Notes |
|---|---|---|
| `frame-time` | `frame_timer_.elapsed() * format_desc_.hz * 0.5` | Updated on every `receive_impl` call (including B-field). Normalised: 1.0 = one frame period. |
| `decode-time` | `decode_timer.elapsed() * format_desc_.fps * 0.5` | Measured per decoded frame in `read_loop`. |
| `queue-fill` | `(ready_queue_.size() + 1) / (MAX_QUEUED + 1)` | Updated after each push. |
| `dropped` | tag: WARNING | Emitted when CUDA decode returns a non-success error code. |

Title format (updated on every A-field in `receive_impl`):
```
clip.mov  125 / 700  |  5.0s / 28.0s  |  25.0fps
```
- `125 / 700` = current frame / total frames (total omitted when unknown)
- `5.0s / 28.0s` = elapsed / total duration (total omitted when unknown)
- `25.0fps` = live output fps over a rolling ~1-second window

---

### Threading Model

```
Constructor thread (caller):
  cudaSetDevice(cuda_device_)
  ProResDemuxer::open()                — libavformat init
  first read_packet()                  — probe frame geometry (ProResFrameInfo)
  prores_decode_ctx_create() × NUM_SLOTS (5)
  ogl_device_->dispatch_sync():
    create NUM_SLOTS GL textures (GL_RGBA16)
    [WIN32] wglShareLists(main_hglrc, shared_hglrc_)
  [fallback] cudaMallocHost() × NUM_SLOTS  (host-copy path)
  seek_to_frame(in_frame_)  if in_frame_ > 0
  Launch read_thread_

read_thread_  (name "prores-read"):
  cudaSetDevice(cuda_device_)
  [WIN32] wglMakeCurrent(hdc_, shared_hglrc_)
  CudaGLTexture::register() × NUM_SLOTS
  read_loop(): demux → decode → push to ready_queue_

CasparCG output thread:
  receive_impl()  — pops from ready_queue_, caches for B-field, returns draw_frame
  call()          — sets seek_request_, notifies queue_cv_
```

---

### Building (ProRes producer)

```bat
rem Build only the cuda_prores module and the CasparCG executable:
cmake --build D:\Github\CasparCG-cuda\out\build\x64-RelWithDebInfo --target cuda_prores -j4
cmake --build D:\Github\CasparCG-cuda\out\build\x64-RelWithDebInfo --target casparcg -j4
```

---

### Producer Bugs Fixed

| Bug | Symptom | Root cause | Fix |
|---|---|---|---|
| R/B channel swap (zero-copy) | Red and blue channels inverted on screen | `pixel_format::bgra` applied extra R↔B swizzle after CUDA wrote BGRA order | Changed to `pixel_format::rgba` for zero-copy path |
| BT.601 files decoded as 709 | Washed-out / shifted colours on SD content | Only matrix values 1 and 9 handled; 5, 6, 0 fell through to 709 | Added BT.601 (5/6) and unspecified (0→709) handling |
| Loop reopened demuxer | 1–2 frame glitch at loop point; IN position lost | `demuxer_ = make_unique<ProResDemuxer>(path_)` on every EOF | Changed to `seek_to_frame(in_frame_)` for both EOF and OUT |
| Seek did not wake read thread | CALL seek had ~1 s latency on paused/stopped producer | `seek_request_` set but `queue_cv_` not notified | Added `queue_cv_.notify_one()` after setting `seek_request_` |
| DIAG title showed 0.0 s | Title showed `0.0s / 0.0s` for files without duration metadata | `total_seconds_` only set if `duration_us() > 0`; display was unconditional | Conditional display: show `/ Xs` only when `total_seconds_ > 0` |
