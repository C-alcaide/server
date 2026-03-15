# CasparCG CUDA ProRes — Operation Guide

## Overview

The module provides two recording consumers **and** a GPU-accelerated ProRes playback producer:

| Role | AMCP keyword | Use case |
|---|---|---|
| **Producer** | `PLAY 1-1 CUDA_PRORES <file>` | GPU-decode a ProRes `.mov`/`.mxf`/`.mkv` file for playout |
| **Consumer** | `ADD 1 CUDA_PRORES …` | GPU-encode the channel compositor output to ProRes |
| **Bypass consumer** | `ADD 1 CUDA_PRORES_BYPASS …` | GPU-encode a raw DeckLink SDI input directly to ProRes |

All three share the same GPU encode/decode kernel pipeline (NVIDIA CUDA).  Consumers write `.mov` or `.mxf` files; the producer reads them back.

---

## CUDA_PRORES Producer (Playback)

The ProRes producer uses CUDA to GPU-decode ProRes frames and feeds them into the CasparCG compositor via the zero-copy CUDA-GL interop path (Windows) or a host-copy fallback (Linux / when `wglShareLists` fails).  Audio is decoded by libavcodec from the same container and delivered at the channel's native cadence.

### PLAY Command Syntax

```
PLAY <channel>-<layer> CUDA_PRORES <filename>
    [LOOP]
    [SEEK  <frame>]
    [IN    <frame>]
    [START <frame>]
    [OUT   <frame>]
    [LENGTH <frames>]
    [COLOR_MATRIX  709|2020|601|AUTO]
    [DEVICE <cuda-index>]
    [FILE <filename>]
```

`<filename>` is a path relative to the CasparCG **media** folder (or an absolute path).  Extension may be omitted; `.mov`, `.mxf`, `.mkv`, `.mp4` are probed in that order.

| Parameter | Default | Description |
|---|---|---|
| `LOOP` | off | Loop playback. On EOF (or OUT) the producer seeks back to `IN`/`SEEK` instead of reopening the file. |
| `SEEK` (or `IN` or `START`) | `0` | First frame to play (0-based frame index). Seek is applied before the read thread starts — the first delivered frame is exactly this frame. |
| `OUT` | end of file | Exclusive stop frame. Playback stops (or loops) just *before* this frame. |
| `LENGTH` | rest of file | Frame count from the `SEEK`/`IN` position. Converted to `OUT = IN + LENGTH` internally. Ignored if `OUT` is also given. |
| `COLOR_MATRIX` | `AUTO` | Override the colour matrix embedded in the ProRes file. `709` = BT.709, `601` = BT.601, `2020` = BT.2020, `AUTO` = use per-frame metadata (default). Useful when the file has wrong or absent matrix metadata. |
| `DEVICE` | `0` | CUDA GPU index (0-based). Selects which GPU decodes. |
| `FILE` | — | Explicit keyword form; equivalent to placing the filename positionally. |

### CALL — Seek and Loop Control

```
CALL <channel>-<layer> seek <target> [<offset>]
CALL <channel>-<layer> loop [0|1]
```

**`seek` target values:**

| Target | Meaning |
|---|---|
| `<integer>` | Absolute frame number (0-based) |
| `rel` or `current` | Current frame position (useful with an offset) |
| `start` or `in` | Frame 0 (rewind to beginning) |
| `end` | Last frame of the file |

**Optional offset** (third parameter): a signed integer added to the resolved target.

Examples:
```
CALL 1-10 seek 250              ; jump to frame 250
CALL 1-10 seek rel +25          ; advance 25 frames
CALL 1-10 seek rel -25          ; rewind 25 frames
CALL 1-10 seek start            ; rewind to beginning
CALL 1-10 seek end              ; jump to last frame
CALL 1-10 loop                  ; query loop state → returns "0" or "1"
CALL 1-10 loop 1                ; enable loop
CALL 1-10 loop 0                ; disable loop
```

Seek is non-blocking: it posts a `seek_request_` to the read thread, which flushes the output queue and seeks at the next opportunity (within one frame interval).

### OSC State Keys

The ProRes producer publishes OSC state at the standard CasparCG monitor path `/channel/<ch>/stage/<layer>/`:

| OSC key | Value | Description |
|---|---|---|
| `file/name` | string | Filename (without directory path) |
| `file/path` | string | Full absolute file path |
| `file/time` | `[current_s, total_s]` | Current playback position and total duration in seconds |
| `file/loop` | bool | Current loop state |
| `width` | int | Frame width in pixels |
| `height` | int | Frame height in pixels |

These match the keys published by CasparCG's built-in FFmpeg `av_producer`, so existing monitoring software works without changes.

### Diagnostics Graph

The ProRes producer registers a diagnostics graph with the following tracks:

| Track | Colour | Meaning |
|---|---|---|
| `frame-time` | Bright green | Inter-receive call interval as a fraction of frame period. Should stay near 0.5 (normalised by `hz × 0.5`). |
| `decode-time` | Dark green | CUDA decode time as a fraction of frame period. Should stay well below 1.0. |
| `queue-fill` | Blue | Output queue fill level as a fraction of `MAX_QUEUED`. |
| `dropped` | Red flash | A frame was dropped — decode took too long or the queue was full. |

The graph title updates every frame and shows:
```
clip.mov  125 / 700  |  5.0s / 28.0s  |  25.0fps
```
(filename · current frame / total frames · current time / total time · current output fps)

### Producer Examples

```amcp
; Basic playback
PLAY 1-10 CUDA_PRORES colorbars_hq

; Loop from frame 50 to frame 200
PLAY 1-10 CUDA_PRORES colorbars_hq LOOP IN 50 OUT 200

; Play exactly 500 frames starting at frame 100
PLAY 1-10 CUDA_PRORES colorbars_hq SEEK 100 LENGTH 500

; Force BT.601 colour matrix (e.g. SD content tagged as 709)
PLAY 1-10 CUDA_PRORES sd_clip COLOR_MATRIX 601

; Stop playback
STOP 1-10

; Live seek while playing
CALL 1-10 seek 0
CALL 1-10 seek rel +25
CALL 1-10 loop 1
```

---

## Recording Consumers

Both consumers encode on the GPU (NVIDIA CUDA) to an Apple-compliant ProRes bitstream and mux into `.mov` or `.mxf`.

---

## ProRes Profiles

| `PROFILE` | FourCC | Apple name | Pixel format | Typical bitrate @ 1080i50 |
|---|---|---|---|---|
| `0` | `apco` | Proxy | 4:2:2 10-bit | ~45 Mbps |
| `1` | `apcs` | LT | 4:2:2 10-bit | ~102 Mbps |
| `2` | `apcn` | Standard 422 | 4:2:2 10-bit | ~147 Mbps |
| **`3`** | `apch` | **422 HQ** (default) | **4:2:2 10-bit** | **~220 Mbps** |
| `4` | `ap4h` | 4444 | 4:4:4 12-bit + optional alpha | ~330 Mbps |
| `5` | `ap4x` | 4444 XQ | 4:4:4 12-bit + optional alpha | ~500 Mbps |

---

## AMCP Parameters — CUDA_PRORES (Consumer)

```
ADD <channel> CUDA_PRORES
    PATH      <output-directory>
    FILENAME  <filename.mov>
    [PROFILE  0-5]
    [CODEC    MOV|MXF]
    [QSCALE   1-31]
    [SLICES   1|2|4|8]
    [ALPHA    0|1]
    [HDR      SDR|HLG|PQ]
    [MAXCLL   <nits>]
    [MAXFALL  <nits>]
```

| Parameter | Default | Description |
|---|---|---|
| `PATH` | `.` | Output directory. **Use forward slashes** — backslashes are mangled by AMCP's C-escape processing. |
| `FILENAME` | `prores_YYYYMMDD_HHMMSS.mov` | Output filename. If omitted, a timestamped name is generated automatically. |
| `PROFILE` | `3` | ProRes variant. See table above. |
| `CODEC` | `MOV` | Container format: `MOV` or `MXF`. |
| `QSCALE` | `8` | Quantization scale 1–31. Lower = better quality / larger file. `8` matches Apple Reference Encoder HQ. `1` = maximum quality. |
| `SLICES` | `4` | Parallel horizontal slices per macroblock row. Valid: `1`, `2`, `4`, `8`. Higher = more GPU parallelism, diminishing returns above 4. |
| `ALPHA` | `1` | Include alpha channel. Only relevant for profiles 4 and 5 (4444). Set `ALPHA 0` to reduce file size when alpha is not needed. |
| `HDR` | `SDR` | Colour space: `SDR` (BT.709), `HLG` (BT.2020 HLG), `PQ` (HDR10 PQ). |
| `MAXCLL` | `1000` | Maximum Content Light Level in nits. Used only with `HDR PQ`. |
| `MAXFALL` | `400` | Maximum Frame-Average Light Level in nits. Used only with `HDR PQ`. |

**Note:** `CUDA_PRORES` (consumer) always occupies consumer slot **1** on the channel. Only one CUDA_PRORES consumer can be active per channel at a time. The `CUDA_PRORES` producer (`PLAY`) and consumer (`ADD`) are independent — both can be active simultaneously on the same or different channels.

---

## AMCP Parameters — CUDA_PRORES_BYPASS

All parameters above apply, plus:

| Parameter | Default | Description |
|---|---|---|
| `DEVICE` | `1` | DeckLink device index (1-based). **Required** — specifies which physical SDI input to capture. |
| `CUDA_DEVICE` | `0` | CUDA GPU index (0-based). Useful in multi-GPU systems to pin encoding to a specific card. |

`CUDA_PRORES_BYPASS` always occupies consumer slot **2**. It does not require a `PLAY` command — it captures directly from the SDI input.

---

## Consumer Command Examples

### Standard recording (compositor output → HQ ProRes)

```amcp
PLAY 1 DECKLINK DEVICE 1
ADD 1 CUDA_PRORES PATH "D:/recordings" FILENAME "camera1.mov" PROFILE 3
```

### Stop recording

```amcp
REMOVE 1 CUDA_PRORES
STOP 1
```

### Direct SDI capture on two inputs simultaneously

```amcp
ADD 1 CUDA_PRORES_BYPASS PATH "D:/recordings" FILENAME "sdi1.mov" DEVICE 1 PROFILE 3
ADD 2 CUDA_PRORES_BYPASS PATH "D:/recordings" FILENAME "sdi2.mov" DEVICE 2 PROFILE 3
```

### Stop bypass recording

```amcp
REMOVE 1 CUDA_PRORES_BYPASS
REMOVE 2 CUDA_PRORES_BYPASS
```

### 4444 with alpha (for graphics/CG output)

```amcp
ADD 1 CUDA_PRORES PATH "D:/graphics" FILENAME "out_4444.mov" PROFILE 4 ALPHA 1
```

### 4444 without alpha (saves ~25% file size)

```amcp
ADD 1 CUDA_PRORES PATH "D:/graphics" FILENAME "out_4444.mov" PROFILE 4 ALPHA 0
```

### MXF container

```amcp
ADD 1 CUDA_PRORES_BYPASS PATH "D:/mxf" FILENAME "camera1.mxf" DEVICE 1 PROFILE 3 CODEC MXF
```

### Maximum quality (QSCALE 1)

```amcp
ADD 1 CUDA_PRORES PATH "D:/masters" FILENAME "master.mov" PROFILE 3 QSCALE 1
```

### HDR HLG recording

```amcp
ADD 1 CUDA_PRORES_BYPASS PATH "D:/hdr" FILENAME "hlg.mov" DEVICE 1 PROFILE 3 HDR HLG
```

### HDR PQ / HDR10

```amcp
ADD 1 CUDA_PRORES_BYPASS PATH "D:/hdr" FILENAME "hdr10.mov" DEVICE 1 PROFILE 3 HDR PQ MAXCLL 4000 MAXFALL 400
```

---

## casparcg.config (pre-configured consumers)

Consumers can be defined in the config to start automatically on launch:

```xml
<channel>
    <video-mode>1080i5000</video-mode>
    <consumers>
        <cuda_prores>
            <path>D:/recordings</path>
            <profile>3</profile>
            <codec>mov</codec>
            <qscale>8</qscale>
            <slices>4</slices>
        </cuda_prores>
    </consumers>
</channel>
```

```xml
<channel>
    <video-mode>1080i5000</video-mode>
    <consumers>
        <cuda_prores_bypass>
            <path>D:/recordings</path>
            <device>1</device>
            <profile>3</profile>
            <codec>mov</codec>
            <qscale>8</qscale>
        </cuda_prores_bypass>
    </consumers>
</channel>
```

---

## Diagnostics

Both consumers register a diagnostics graph visible in the CasparCG client:

| Track | Colour | Meaning |
|---|---|---|
| `encode-time` | Green | Encode time as a fraction of frame time. Should stay below 1.0. |
| `queue-depth` | Orange | Encode queue fill level (0–8 frames). A rising queue warns of sustained overload. |
| `dropped-frame` | Red flash | A frame was discarded because the queue was full. Indicates the GPU cannot keep up. |
| `encode-error` | Pink flash | An encode or mux error occurred. Check the log for details. |

The title bar shows: `cuda_prores[1|3] | 750 fr (30.0s)` — consumer slot, profile, frame count, and elapsed time.

---

## Best Use Guidelines

**Use CUDA_PRORES when:**
- You need to record the CasparCG compositor output (graphics, mixed signals, playout)
- You want frame-accurate recording synced to the channel timeline
- Profile 4/5 (4444) is needed for graphics with alpha

**Use CUDA_PRORES_BYPASS when:**
- You need to ISO-record a raw SDI feed without any CasparCG processing overhead
- CPU usage is critical — bypass uses ~5% sys CPU vs ~24% for the FILE consumer at equivalent quality
- You need multi-camera ingest on a single machine

**Path formatting:**
- Always use forward slashes: `D:/recordings/show1` — backslashes are interpreted as AMCP escape sequences and will corrupt the path

**Bitrate guide (1080i50, 2 simultaneous channels):**

| Profile | Per-channel | 2-channel total | Storage per hour |
|---|---|---|---|
| Proxy (0) | ~45 Mbps | ~90 Mbps | ~40 GB |
| LT (1) | ~102 Mbps | ~204 Mbps | ~92 GB |
| Standard (2) | ~147 Mbps | ~294 Mbps | ~132 GB |
| **HQ (3)** | **~220 Mbps** | **~440 Mbps** | **~198 GB** |
| 4444 (4) | ~330 Mbps | ~660 Mbps | ~297 GB |
| 4444 XQ (5) | ~500 Mbps | ~1 Gbps | ~450 GB |

**SLICES:**
- `4` (default) is the right choice for 1080 on most modern GPUs
- Use `8` only on high-core-count server GPUs (e.g. A100, H100) for 4K
- `1` or `2` may be needed for compatibility with certain NLEs that don't support multi-slice ProRes
