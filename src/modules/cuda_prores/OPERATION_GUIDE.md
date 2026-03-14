# CasparCG CUDA ProRes Consumer — Operation Guide

## Overview

The module provides two independent consumers:

| Consumer | AMCP keyword | Consumer slot | Use case |
|---|---|---|---|
| **CUDA_PRORES** | `CUDA_PRORES` | 1 | Records the channel compositor output — requires a `PLAY` source on the channel |
| **CUDA_PRORES_BYPASS** | `CUDA_PRORES_BYPASS` | 2 | Records direct SDI input from a DeckLink device — no `PLAY` needed |

Both encode on the GPU (NVIDIA CUDA) to an Apple-compliant ProRes bitstream and mux into `.mov` or `.mxf`.

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

## AMCP Parameters — CUDA_PRORES

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

**Note:** `CUDA_PRORES` always occupies consumer slot **1** on the channel. Only one CUDA_PRORES consumer can be active per channel at a time.

---

## AMCP Parameters — CUDA_PRORES_BYPASS

All parameters above apply, plus:

| Parameter | Default | Description |
|---|---|---|
| `DEVICE` | `1` | DeckLink device index (1-based). **Required** — specifies which physical SDI input to capture. |
| `CUDA_DEVICE` | `0` | CUDA GPU index (0-based). Useful in multi-GPU systems to pin encoding to a specific card. |

`CUDA_PRORES_BYPASS` always occupies consumer slot **2**. It does not require a `PLAY` command — it captures directly from the SDI input.

---

## Command Examples

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
