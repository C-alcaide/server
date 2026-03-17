# HDR & Wide-Gamut Guide for CasparVP

This guide covers the HDR-related features added to CasparVP: channel color configuration, BT.2020 / PQ / HLG propagation through the pipeline, DeckLink HDR input/output, FFmpeg consumer color metadata, and High Frame Rate (HFR) support.

---

## Table of Contents

1. [Overview of the Pipeline](#overview-of-the-pipeline)
2. [Channel Configuration](#channel-configuration)
3. [DeckLink Input (Capture)](#decklink-input-capture)
4. [DeckLink Output (Playout)](#decklink-output-playout)
5. [FFmpeg Consumer (File Recording)](#ffmpeg-consumer-file-recording)
6. [CUDA ProRes Consumer (Recording)](#cuda-prores-consumer-recording)
7. [CUDA ProRes Producer (Playback)](#cuda-prores-producer-playback)
8. [High Frame Rate Formats](#high-frame-rate-formats)
9. [Complete Config Examples](#complete-config-examples)
10. [Quick Reference Table](#quick-reference-table)
11. [Notes & Known Limitations](#notes--known-limitations)

---

## Overview of the Pipeline

```
┌────────────────────────────────────────────────────────────────┐
│  Source                                                        │
│  DeckLink capture  → bmdDeckLinkFrameMetadata* per frame       │
│  FFmpeg producer   → AVFrame color_trc / colorspace per frame  │
│  CUDA ProRes prod. → ProRes bitstream color_matrix/transfer_func│
│  All map to: color_space (bt601/709/2020) + color_transfer (sdr/pq/hlg)│
└───────────────────────┬────────────────────────────────────────┘
                        │  pixel_format_desc { color_space, color_transfer }
                        ▼
┌────────────────────────────────────────────────────────────────┐
│  OGL Mixer                                                     │
│  - YCbCr↔BGRA conversion uses correct colour matrix           │
│  - Output BGRA frame carries channel color_space +             │
│    color_transfer in pixel_format_desc                         │
└───────────────────────┬────────────────────────────────────────┘
                        │
          ┌─────────────┼──────────────┐
          ▼             ▼              ▼
┌──────────────────┐ ┌───────────────┐ ┌────────────────────┐
│  DeckLink output │ │ FFmpeg consumer│ │ CUDA ProRes consumer│
│  - HDR v210      │ │ enc->color_trc│ │ - MOV/MXF color    │
│  - EOTF signaled │ │ enc->primaries│ │   metadata         │
│  - primaries set │ │ enc->colorsp. │ │ - HDR mode auto-   │
└──────────────────┘ └───────────────┘ │   derived from ch. │
                                       └────────────────────┘
```

Color space and transfer function are declared **once on the channel** and flow automatically to every consumer. Individual consumers can optionally override them.

---

## Channel Configuration

### `casparcg.config` — `<channel>` element

```xml
<channel>
  <!-- Standard definition of video mode, depth etc. -->
  <video-mode>2160p5000</video-mode>
  <color-depth>16</color-depth>

  <!-- Wide-gamut colour primaries: bt709 (default) | bt2020 -->
  <color-space>bt2020</color-space>

  <!-- Transfer function:  sdr (default) | pq | hlg -->
  <color-transfer>pq</color-transfer>

  <consumers>
    ...
  </consumers>
</channel>
```

### `<color-space>` values

| Value | Meaning | Typical use |
|-------|---------|-------------|
| `bt709` | ITU-R BT.709 primaries (default) | HD SDR; standard broadcast |
| `bt2020` | ITU-R BT.2020 primaries | UHD HDR; wide-gamut |

> `bt601` is supported in the DeckLink input pipeline (auto-detected) but is not a valid channel config value—channels always output BT.709 or BT.2020.

### `<color-transfer>` values

| Value | Standard | Typical use |
|-------|----------|-------------|
| `sdr` | BT.709 gamma (default) | All SDR workflows |
| `pq` | SMPTE ST 2084 (Perceptual Quantizer) | HDR10, cinema mastering |
| `hlg` | ARIB STD-B67 (Hybrid Log-Gamma) | HDR broadcast (BBC, NHK) |

> Setting `color-transfer` has no visual effect unless `color-space` is also `bt2020`. An SDR channel with `color-transfer pq` is technically inconsistent but will not error — the metadata will simply propagate.

### `<color-depth>` requirement for HDR

For HDR output you must use 16-bit depth. 8-bit channels can carry BT.2020 metadata but will lose precision:

```xml
<color-depth>16</color-depth>   <!-- required for meaningful HDR -->
```

---

## DeckLink Input (Capture)

The DeckLink producer **auto-detects** color space and transfer function from SDK metadata on every input frame — no configuration is needed.

### What is detected automatically

| SDK metadata key | Maps to |
|-----------------|---------|
| `bmdDeckLinkFrameMetadataColorspace` → `bmdColorspaceRec2020` | `core::color_space::bt2020` |
| `bmdDeckLinkFrameMetadataColorspace` → `bmdColorspaceRec601` | `core::color_space::bt601` |
| `bmdDeckLinkFrameMetadataHDRElectroOpticalTransferFunc` → `2` (PQ) | `core::color_transfer::pq` |
| `bmdDeckLinkFrameMetadataHDRElectroOpticalTransferFunc` → `1` (HLG) | `core::color_transfer::hlg` |
| (default / unset) | `core::color_space::bt709`, `core::color_transfer::sdr` |

### 10-bit capture

To capture HDR in 10-bit (required for PQ/HLG), configure the input in 10BIT pixel format in your DeckLink producer settings. The decoded frame will be v210, decoded to YUV444P10 AVFrames with all color fields set.

### Frame metadata flow

```
DeckLink SDK frame
  → bmdDeckLinkFrameMetadataColorspace     → color_space  per frame
  → bmdDeckLinkFrameMetadataHDRElectroOpticalTransferFunc → color_transfer per frame
  → make_frame() → pixel_format_desc { color_space, color_transfer }
  → AVFrame { colorspace, color_primaries, color_trc, color_range }
```

Frames from a mixed input (e.g. a source that switches between SDR and HDR) will carry the correct metadata on a per-frame basis.

---

## DeckLink Output (Playout)

### Automatic HDR enabling

When a channel has `color-space bt2020` and `color-transfer pq` or `hlg`, the DeckLink consumer **automatically**:
- Switches to HDR v210 10-bit output format
- Sets `bmdFrameContainsHDRMetadata` on every output frame
- Signals the correct EOTF via `bmdDeckLinkFrameMetadataHDRElectroOpticalTransferFunc`
- Signals BT.2020 colour primaries and white point via the HDR static metadata extension

No explicit `<hdr>true</hdr>` flag is needed — it is derived from the channel's color settings.

### DeckLink consumer config in `casparcg.config`

```xml
<consumers>
  <decklink>
    <device>1</device>
    <embedded-audio>true</embedded-audio>
    <!-- Inherits color-space and color-transfer from the channel.    -->
    <!-- Override per-consumer only if this output differs:           -->
    <!-- <color-space>bt2020</color-space>                           -->
    <!-- <color-transfer>pq</color-transfer>                         -->

    <!-- Optional static HDR mastering display metadata:             -->
    <hdr-metadata>
      <min-dml>0.005</min-dml>    <!-- min display mastering luminance (nits) -->
      <max-dml>1000</max-dml>     <!-- max display mastering luminance (nits) -->
      <max-fall>100</max-fall>    <!-- max frame-average light level (nits) -->
      <max-cll>1000</max-cll>     <!-- max content light level (nits)        -->
    </hdr-metadata>
  </decklink>
</consumers>
```

### EOTF values sent on the wire (SDK constants)

| `color-transfer` | EOTF value sent | Standard |
|-----------------|----------------|---------|
| `pq` | `2` (`EOTF::PQ`) | SMPTE ST 2084 |
| `hlg` | `3` (`EOTF::HLG`) | ARIB STD-B67 |

### Colour primaries sent for BT.2020

| Field | Value |
|-------|-------|
| Red | (0.708, 0.292) |
| Green | (0.170, 0.797) |
| Blue | (0.131, 0.046) |
| White point | (0.3127, 0.3290) — D65 |

---

## FFmpeg Consumer (File Recording)

The FFmpeg consumer sets color metadata on the **encoder context** before opening, so it is written into the container/stream header — not just per-frame.

### What gets written into the file

| Channel setting | Field set on encoder | Value |
|----------------|---------------------|-------|
| `bt2020` + `pq` | `color_primaries` | `AVCOL_PRI_BT2020` |
| | `colorspace` | `AVCOL_SPC_BT2020_NCL` |
| | `color_trc` | `AVCOL_TRC_SMPTE2084` |
| `bt2020` + `hlg` | `color_primaries` | `AVCOL_PRI_BT2020` |
| | `colorspace` | `AVCOL_SPC_BT2020_NCL` |
| | `color_trc` | `AVCOL_TRC_ARIB_STD_B67` |
| `bt709` (default) | `color_primaries` | `AVCOL_PRI_BT709` |
| | `colorspace` | `AVCOL_SPC_BT709` |
| | `color_trc` | `AVCOL_TRC_BT709` |

Additionally, every YCbCr `AVFrame` fed to the encoder carries matching `color_primaries`, `colorspace`, `color_trc`, and `color_range = MPEG` — ensuring frame-level and container-level metadata are consistent.

### Verifying output with ffprobe

```
ffprobe -v error -show_streams -select_streams v my_recording.mov
```

Look for:
```
color_space=bt2020nc
color_primaries=bt2020
color_transfer=smpte2084    ← PQ
# or
color_transfer=arib-std-b67 ← HLG
```

### Recommended FFmpeg consumer config for HDR ProRes 4444 XQ

```xml
<consumers>
  <ffmpeg>
    <path>D:\Recordings\output.mov</path>
    <args>-vcodec prores_ks -profile:v 4 -pix_fmt yuv422p10le</args>
  </ffmpeg>
</consumers>
```

> Channels with `color-depth 16` output `BGRA64LE` frames. The FFmpeg filter graph will convert to the codec's required pixel format automatically.

---

## CUDA ProRes Consumer (Recording)

The `CUDA_PRORES` (and `CUDA_PRORES_BYPASS`) consumer now **automatically inherits the channel's `color-transfer`** when no explicit `HDR` override is specified. This means a BT.2020 PQ channel will produce PQ-tagged ProRes MOV/MXF output without any extra AMCP parameters.

### Automatic inheritance behaviour

| Channel `<color-transfer>` | Default CUDA ProRes HDR mode | MOV/MXF output |  
|---------------------------|------------------------------|----------------|  
| `sdr` (default) | SDR BT.709 | Standard Rec.709 colour tags |  
| `hlg` | HLG BT.2020 | HLG primaries + ARIB matrix |  
| `pq` | PQ HDR10 | PQ primaries + ST 2086 mastering + CLL/FALL |  

### Override via AMCP

An explicit `HDR` parameter on the AMCP command still overrides the channel default:

```
ADD 1-10 CUDA_PRORES PATH D:/recordings HDR PQ MAXCLL 4000 MAXFALL 400
ADD 1-10 CUDA_PRORES PATH D:/recordings HDR HLG
ADD 1-10 CUDA_PRORES PATH D:/recordings HDR SDR    # force SDR even on HDR channel
```

When `HDR` is omitted, the mode is derived from `channel_info.default_color_transfer`.

### XML config — inherits automatically

In `casparcg.config`, leaving `<hdr>` absent (or not specifying it) inherits from the channel:

```xml
<consumers>
  <cuda_prores>
    <path>D:\Recordings</path>
    <profile>3</profile>       <!-- 0=Proxy 1=LT 2=Standard 3=HQ 4=4444 5=4444XQ -->
    <codec>MOV</codec>         <!-- MOV or MXF -->
    <!-- <hdr>PQ</hdr>  ← omit to inherit from channel color-transfer -->
    <max_cll>1000</max_cll>
    <max_fall>400</max_fall>
  </cuda_prores>
</consumers>
```

To explicitly force a mode regardless of channel config:

```xml
<hdr>PQ</hdr>   <!-- PQ | HLG | SDR -->
```

Same behaviour applies to `CUDA_PRORES_BYPASS`.

---

## CUDA ProRes Producer (Playback)

The CUDA ProRes producer (used when playing `.mov` / `.mxf` ProRes files via the CUDA decode path) now **reads the EOTF and colour primaries from the ProRes bitstream header** and propagates them into `pixel_format_desc` on every decoded frame.

### What is read from the bitstream

| ProRes frame header field | Value | Maps to |
|--------------------------|-------|---------|
| `color_matrix` | `1` (Rec.709) | `core::color_space::bt709` |
| `color_matrix` | `5` or `6` (Rec.601) | `core::color_space::bt601` |
| `color_matrix` | `9` (BT.2020) | `core::color_space::bt2020` |
| `transfer_func` | `1` (Rec.709 gamma) | `core::color_transfer::sdr` |
| `transfer_func` | `14` (HLG) | `core::color_transfer::hlg` |
| `transfer_func` | `16` (PQ / ST 2084) | `core::color_transfer::pq` |

This propagation happens per-frame, so a file that switches colour tagging mid-stream will be handled correctly.

### What this enables

- **HDR ProRes files** (produced by Final Cut Pro, Resolve, or a previous CUDA ProRes recording) are now correctly tagged as PQ or HLG when fed through the CasparVP mixer — the downstream DeckLink output will signal the correct EOTF and the FFmpeg consumer will write the correct encoder colour metadata.
- **SDR content on an HDR channel** retains its SDR tag from the file; no channel override is applied (the file metadata wins at the frame level).
- **Per-consumer colour-matrix override** (`MATRIX BT.2020` etc.) continues to work as before.

### Colour-matrix AMCP override

An integer `MATRIX` override still applies to the colour *space* (primaries/matrix) only:

```
PLAY 1-10 my_hdr_prores.mov MATRIX 9   # force BT.2020 colour space
```

The transfer function (`transfer_func`) is always read from the bitstream when no container-level override has been explicitly programmed.

---

## High Frame Rate Formats

Four new video modes have been added for DeckLink 8K Pro and compatible hardware:

| `video-mode` | Resolution | Frame rate | Audio cadence |
|-------------|-----------|-----------|--------------|
| `1080p10000` | 1920×1080 | 100 fps | 480 samples/frame |
| `1080p12000` | 1920×1080 | 120 fps | 400 samples/frame |
| `2160p10000` | 3840×2160 | 100 fps | 480 samples/frame |
| `2160p12000` | 3840×2160 | 120 fps | 400 samples/frame |

These map to the SDK modes `bmdModeHD1080p100`, `bmdModeHD1080p120`, `bmdMode4K2160p100`, and `bmdMode4K2160p120`.

### HFR + HDR combined example

```xml
<channel>
  <video-mode>2160p10000</video-mode>
  <color-depth>16</color-depth>
  <color-space>bt2020</color-space>
  <color-transfer>pq</color-transfer>
  <consumers>
    <decklink>
      <device>1</device>
      <embedded-audio>true</embedded-audio>
    </decklink>
  </consumers>
</channel>
```

---

## Complete Config Examples

### Standard UHD HDR10 playout (BT.2020 PQ, DeckLink 8K Pro)

```xml
<channel>
  <video-mode>2160p5000</video-mode>
  <color-depth>16</color-depth>
  <color-space>bt2020</color-space>
  <color-transfer>pq</color-transfer>
  <consumers>
    <decklink>
      <device>1</device>
      <embedded-audio>true</embedded-audio>
      <hdr-metadata>
        <min-dml>0.005</min-dml>
        <max-dml>1000</max-dml>
        <max-fall>400</max-fall>
        <max-cll>1000</max-cll>
      </hdr-metadata>
    </decklink>
  </consumers>
</channel>
```

### HLG broadcast output (BT.2020 HLG, DeckLink)

```xml
<channel>
  <video-mode>1080i5000</video-mode>
  <color-depth>16</color-depth>
  <color-space>bt2020</color-space>
  <color-transfer>hlg</color-transfer>
  <consumers>
    <decklink>
      <device>1</device>
      <embedded-audio>true</embedded-audio>
    </decklink>
  </consumers>
</channel>
```

### HDR file recording — FFmpeg consumer (BT.2020 PQ)

```xml
<channel>
  <video-mode>2160p5000</video-mode>
  <color-depth>16</color-depth>
  <color-space>bt2020</color-space>
  <color-transfer>pq</color-transfer>
  <consumers>
    <ffmpeg>
      <path>D:\Recordings\output.mov</path>
      <args>-vcodec prores_ks -profile:v 4 -pix_fmt yuv422p10le</args>
    </ffmpeg>
  </consumers>
</channel>
```

### HDR file recording — CUDA ProRes consumer (auto-inherits PQ from channel)

```xml
<channel>
  <video-mode>2160p5000</video-mode>
  <color-depth>16</color-depth>
  <color-space>bt2020</color-space>
  <color-transfer>pq</color-transfer>
  <consumers>
    <cuda_prores>
      <path>D:\Recordings</path>
      <profile>3</profile>          <!-- HQ -->
      <codec>MOV</codec>
      <!-- hdr omitted: automatically uses PQ from channel color-transfer -->
      <max_cll>1000</max_cll>
      <max_fall>400</max_fall>
    </cuda_prores>
  </consumers>
</channel>
```

Or via AMCP (omitting `HDR` also inherits from the channel):
```
ADD 1-10 CUDA_PRORES PATH D:/recordings MAXCLL 1000 MAXFALL 400
```

### HDR capture + simultaneous playout and record

```xml
<channel>
  <video-mode>2160p5000</video-mode>
  <color-depth>16</color-depth>
  <color-space>bt2020</color-space>
  <color-transfer>pq</color-transfer>
  <consumers>
    <decklink>
      <device>2</device>             <!-- output on device 2 -->
      <embedded-audio>true</embedded-audio>
    </decklink>
    <ffmpeg>
      <path>D:\Recordings\output.mov</path>
      <args>-vcodec prores_ks -profile:v 4 -pix_fmt yuv422p10le</args>
    </ffmpeg>
  </consumers>
</channel>
```

Use `PLAY 1-10 DECKLINK DEVICE 1` to route DeckLink device 1 as input to this channel.

### Override color-transfer on a single consumer

A BT.2020 PQ channel can output HLG on one specific DeckLink port while keeping PQ elsewhere:

```xml
<channel>
  <video-mode>2160p5000</video-mode>
  <color-depth>16</color-depth>
  <color-space>bt2020</color-space>
  <color-transfer>pq</color-transfer>
  <consumers>
    <decklink>
      <device>1</device>
      <!-- no override: inherits PQ from channel -->
    </decklink>
    <decklink>
      <device>2</device>
      <color-transfer>hlg</color-transfer>   <!-- override to HLG on this port -->
    </decklink>
  </consumers>
</channel>
```

---

## Quick Reference Table

| Scenario | `color-space` | `color-transfer` | `color-depth` |
|---------|--------------|----------------|---------------|
| Standard HD/UHD SDR | `bt709` (default) | `sdr` (default) | `8` |
| UHD HDR10 (PQ) | `bt2020` | `pq` | `16` |
| UHD HLG broadcast | `bt2020` | `hlg` | `16` |
| Wide-gamut SDR (future) | `bt2020` | `sdr` | `16` |

### CUDA ProRes — HDR mode resolution

| `HDR` AMCP param present | Channel `<color-transfer>` | Effective mode |
|--------------------------|---------------------------|----------------|
| Yes (`HDR PQ`) | any | PQ (override) |
| Yes (`HDR HLG`) | any | HLG (override) |
| Yes (`HDR SDR`) | any | SDR (override) |
| No (omitted) | `pq` | PQ (inherited) |
| No (omitted) | `hlg` | HLG (inherited) |
| No (omitted) | `sdr` / default | SDR (inherited) |

### CUDA ProRes producer — bitstream transfer function mapping

| `transfer_func` in bitstream | Reported downstream |
|-----------------------------|---------------------|
| `1` (Rec.709 gamma) | `color_transfer::sdr` |
| `14` (HLG / ARIB STD-B67) | `color_transfer::hlg` |
| `16` (PQ / SMPTE ST 2084) | `color_transfer::pq` |
| other / unset | `color_transfer::sdr` |

---

## Notes & Known Limitations

- **OGL mixer YCbCr conversion** uses the channel's `color-space` for the correct coefficients (BT.601/709/2020). The mixer output is always BGRA — consumer conversion to YCbCr for encoding uses the same color metadata.

- **DeckLink keyer** is now properly validated via `DoesSupportVideoMode(bmdSupportedVideoModeKeying)` rather than blanket-disabling on BT.2020 channels. Devices such as the 8K Pro that support keying in wide-gamut modes work correctly.

- **Dynamic per-frame transfer function switching** is supported on the *input* side (DeckLink capture and file playback detect it per frame). On the *output* side the transfer function is set at channel-start time and is fixed for the session — runtime changes require a channel restart.

- **No tone-mapping is performed** by the mixer. If you play SDR content on a BT.2020/PQ channel it will be treated as narrow-range WCG PQ. Use the FFmpeg producer filter options (e.g. `zscale`) for tone-mapping if needed.

- **HDR static metadata** (MaxCLL, MaxFALL, mastering display luminance) is set per-consumer via `<hdr-metadata>` in the DeckLink consumer config. The FFmpeg consumer currently writes only the EOTF/primaries/matrix into the encoder context; mastering display SEI is not set automatically — use `-x265-params` or Dolby Vision tooling for full HDR10 SEI.

- **HFR formats** (100/120 fps at 1080p and 2160p) require a DeckLink 8K Pro or equivalent hardware capable of those modes.

- **CUDA ProRes consumer** (`CUDA_PRORES` / `CUDA_PRORES_BYPASS`) HDR mode is now derived from the channel's `<color-transfer>` when no explicit `HDR` AMCP parameter is given. Existing workflows that pass `HDR PQ` or `HDR HLG` explicitly are unaffected.

- **CUDA ProRes producer** reads `color_matrix` and `transfer_func` from the ProRes bitstream header on every decoded frame. HDR ProRes files produced by Final Cut Pro, DaVinci Resolve, or a previous CUDA ProRes recording will therefore correctly signal PQ or HLG to the downstream mixer and consumers. Prior to this fix, all CUDA-decoded ProRes frames were reported as SDR regardless of container metadata.
