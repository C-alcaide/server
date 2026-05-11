# CasparCG Replay Module

## Overview

The Replay module provides high-performance recording and instant playback of video using the VMX (Video Media eXchange) codec and the `.mav` segmented file format. It is designed for live production workflows such as sports replay, time-shift buffering, and highlight export.

**Key capabilities:**
*   Record any CasparCG channel to `.mav` files at configurable quality profiles.
*   Play recordings back instantly — including files that are still being recorded ("growing file" / Play-while-Recording).
*   Circular buffer recording with automatic cleanup of old segments.
*   Full transport control: seek by frame number or wall-clock timestamp, variable speed, loop, in/out points.
*   Background export of clips or concatenated highlight reels to MP4 via FFmpeg.
*   Audio is embedded alongside video in each frame of the `.mav` container (32-bit integer PCM).

---

## 1. Technical Architecture

### Codec — libvmx

The module uses [libvmx](https://github.com/openmediatransport/libvmx) (MIT license), a GPU-friendly intra-frame video codec. Every frame is independently compressed (no inter-frame dependencies), which enables:
*   **Instant seek** to any frame without decoding a GOP.
*   **Low, fixed latency** — encode and decode are single-frame operations.
*   BGRA 8-bit pixel format throughout (BT.709 color space).

### File Format — `.mav` / `.idx`

Recordings are stored as **segmented files** inside a folder named after the recording.

```
media/
  my_recording/
    my_recording.mav.000    # Video+audio data (segment 0)
    my_recording.idx.000    # Frame index       (segment 0)
    my_recording.mav.001    # Segment 1
    my_recording.idx.001
    ...
```

**Index file structure (`.idx`):**

Each `.idx` file begins with a `replay_segment_header` (segment index, start/end timestamps, width, height, fps, segment duration), followed by an array of `replay_index_entry_v2` records — one per frame — containing:
*   `file_offset` (int64) — byte offset of the frame in the corresponding `.mav` file.
*   `timestamp` (uint64) — UTC microseconds since epoch.

**Data file structure (`.mav`):**

Each frame is stored as a flat binary record:
```
[uint32  audio_size_bytes]
[byte[]  audio_data]          // 32-bit integer PCM samples
[byte[]  vmx_compressed_data] // Intra-frame compressed video
```

The header magic is `OMAV` (Open Media Audio/Video).

### Segmented & Circular Storage

*   **Segmented recording:** The writer rotates to a new `.mav`/`.idx` pair whenever the configured `segment` duration elapses (default 60 s). Small segment files keep individual file sizes manageable and allow fine-grained cleanup.
*   **Circular buffer:** When `max_duration` is set, the writer automatically deletes the oldest segments once total recording duration exceeds the limit. Segments locked by an active reader are skipped until released.

### Growing-File Playback

The producer periodically calls `Refresh()` on the segmented reader to discover new index entries appended by the writer and to detect new segment files. When playing close to the live edge (within ~2 seconds), refresh frequency increases. This enables play-while-record with minimal delay.

### Overwrite Protection

If the consumer detects that the target filename already exists, it automatically appends a timestamp suffix (e.g., `my_recording_20260511_143022.mav`) and logs a warning. Existing recordings are never silently overwritten.

### Export Pipeline

`EXPORT` spawns a background thread that:
1.  Opens one or more source `.mav` recordings via `ReplaySegmentedReader`.
2.  Decodes each frame through libvmx to raw BGRA.
3.  Pipes raw video into FFmpeg (`rawvideo` → `libx264`, `yuv420p`, `fast` preset).
4.  Outputs the result to the specified file (typically `.mp4`).

Multiple source clips with individual in/out points can be concatenated into a single output file.

---

## 2. Recording (Consumer)

Start recording a channel by adding a `REPLAY` consumer.

### Syntax
```
ADD <channel> REPLAY <filename>[?<options>] [<quality>]
```
*The filename is relative to the `media/` folder defined in your configuration.*

### Recording Options
Append query parameters to the filename to configure segmentation and buffering:

| Parameter | Format | Default | Description |
| :--- | :--- | :--- | :--- |
| `segment` | seconds | `60` | Duration of each segment file. |
| `max_duration` | `HH:MM` or seconds | `86400` (24 h) | Maximum total buffer duration. Older segments are deleted. |

### Quality Profiles

Specify the quality as the third argument to `ADD`. If omitted, `SQ` is used.

| Profile | Type | Description |
| :--- | :--- | :--- |
| `LQ` | Legacy | Low quality / low bitrate. |
| `SQ` | Legacy | **Default.** Good balance for general replay. |
| `HQ` | Legacy | Mastering quality, high bitrate. |
| `OMT_LQ` | OMT | Adaptive low quality with full DC precision. |
| `OMT_SQ` | OMT | Higher quality than `SQ`, full DC precision. |
| `OMT_HQ` | OMT | Adaptive high quality with full DC precision. |

The `OMT` (Open Media Transport) profiles use updated tuning that prioritizes full DC precision at optimized bitrates.

**1080p Bitrates (at 60 fps):**

| Profile | Target Bitrate | DC Precision | Notes |
| :--- | :--- | :--- | :--- |
| **LQ** | 86 Mbps | Low | Legacy Low Quality |
| **OMT_LQ** | 86 Mbps | Low | Same bitrate, adaptable quality |
| **SQ** | **130 Mbps** | **Low** | Default. Good for general replay. |
| **OMT_SQ** | **200 Mbps** | **High** | Higher quality than SQ. Full DC precision. |
| **HQ** | 260 Mbps | High | Mastering quality. |
| **OMT_HQ** | 260 Mbps | High | Same bitrate, adaptable quality. |

**2160p (4K) Bitrates (at 60 fps):**

| Profile | Target Bitrate | DC Precision | Notes |
| :--- | :--- | :--- | :--- |
| **SQ** | 400 Mbps | Low | Good balance |
| **OMT_SQ** | 300 Mbps | Low | Tuned for bandwidth efficiency |
| **HQ** | 800 Mbps | High | High bandwidth requirement |
| **OMT_HQ** | 600 Mbps | High | Optimized high quality |

*Bitrates scale linearly with frame rate (e.g., 1080p30 will be roughly half the 60 fps values).*

### Examples

**Standard Recording (SQ quality):**
```
ADD 1 REPLAY "my_recording"
```

**High Quality Recording:**
```
ADD 1 REPLAY "my_high_quality_recording" HQ
```

**Segmented Recording (no circular cleanup):**
```
ADD 1 REPLAY "my_channel_buffer?segment=60"
```
Segments are created every 60 seconds but never deleted.

**Circular Buffer Recording (time-shift):**
Record the last hour in 1-minute segments:
```
ADD 1 REPLAY "my_channel_buffer?max_duration=01:00&segment=60"
```
This creates `media/my_channel_buffer/` containing:
*   `my_channel_buffer.mav.000`, `.001`, `.002`, …
*   `my_channel_buffer.idx.000`, `.001`, `.002`, …

Segments older than 1 hour are automatically deleted.

**Stop Recording:**
```
REMOVE 1 REPLAY "my_recording"
```

---

## 3. Playback (Producer)

Play a `.mav` recording using the standard `PLAY` command. The producer auto-detects segmented and growing files.

### Syntax
```
PLAY <channel>-<layer> <filename> [LOOP] [SEEK <val>] [IN <val>] [OUT <val>] [LENGTH <val>]
```

The `.mav` extension is optional — the producer will resolve it automatically.

### Examples
```
# Play a completed recording
PLAY 1-10 "my_recording"

# Play a live recording buffer (growing file)
PLAY 1-10 "my_channel_buffer"

# Play looping between frames 100–500
PLAY 1-1 "my_recording" LOOP IN 100 OUT 500

# Play a specific time range with a fixed duration (250 frames = 10 s at 25 fps)
PLAY 1-1 "my_recording" IN 2026-03-03-12-00-00-00 LENGTH 250

# Start playback at the live edge of a growing recording
PLAY 1-1 "growing_file" SEEK LIVE
```

### Growing-File Playback
You can `PLAY` a file that is still being recorded. The producer will continuously discover new frames as they are written. When playing near the live edge, refresh rate increases automatically to minimize latency.

### Audio Behavior
*   Audio is muted automatically when the producer is **paused** (`SPEED 0`) or has reached **end-of-file** without looping, preventing sample-loop glitches.
*   At non-unity speeds, audio is passed through as-is (not pitch-shifted).

---

## 4. Timestamp Format

All time-based parameters (`SEEK`, `IN`, `OUT`, `EXPORT` in/out points) accept either **frame integers** or **wall-clock timestamps**.

**Timestamp format:** `yyyy-mm-dd-hh-mm-ss-ff`

| Field | Description |
| :--- | :--- |
| `yyyy` | Year |
| `mm` | Month |
| `dd` | Day |
| `hh` | Hour (24 h) |
| `mm` | Minute |
| `ss` | Second |
| `ff` | Frame number within that second |

**Alternative ISO format:** `2026-03-03T12:00:00`

**Frame integers:** Plain numbers (e.g., `0`, `100`, `5000`) are interpreted as global frame indices.

The module distinguishes between the two formats automatically: if the value contains `-` or `:` separators, it is parsed as a timestamp; otherwise it is treated as a frame number.

---

## 5. Transport Commands

Control the Replay producer at runtime via `CALL`.

### SEEK
Jump to a specific position.

```
CALL 1-10 SEEK 1000                        # Frame 1000
CALL 1-10 SEEK 2026-03-03-12-00-00-00      # Wall-clock timestamp
CALL 1-10 SEEK LIVE                         # Latest recorded frame (backs off ~25 frames to avoid stutter)
```

### IN / OUT
Set playback boundaries. Playback starts at `IN` and stops (or loops) at `OUT`.

```
CALL 1-10 IN 2026-03-03-10-00-00-00
CALL 1-10 OUT 5000
```

Setting `OUT` to `0` resets it to the dynamic end of file.

### SPEED
Adjust playback speed. Fractional frame accumulation ensures smooth motion at non-integer speeds.

```
CALL 1-10 SPEED 2.0    # 200% speed (fast forward)
CALL 1-10 SPEED 0.5    # 50% speed (slow motion)
CALL 1-10 SPEED 0      # Pause
CALL 1-10 SPEED -1.0   # Reverse playback
```

When fast-forwarding past the end of a growing file, speed automatically resets to `1.0`.

### LOOP
Toggle looping between `IN` and `OUT` points.

```
CALL 1-10 LOOP 1     # Enable
CALL 1-10 LOOP 0     # Disable
```

Called without an argument, `LOOP` toggles the current state.

### EXPORT
Export clips to an external file in the background. The export uses FFmpeg and does not interrupt playback.

**Syntax:**
```
CALL <channel>-<layer> EXPORT <output_filename> [<input_filename>] <in> <out> [...]
```

**Examples:**

*   **Simple export** — frames 100–200 of the current file:
    ```
    CALL 1-10 EXPORT "out.mp4" 100 200
    ```

*   **Time-based export:**
    ```
    CALL 1-10 EXPORT "highlight.mp4" 2026-03-03-12-00-00-00 2026-03-03-12-05-00-00
    ```

*   **Explicit input file** (not the one currently playing):
    ```
    CALL 1-10 EXPORT "out.mp4" "other_recording.mav" 0 500
    ```

*   **Concatenation** — combine multiple clips into one file:
    ```
    CALL 1-10 EXPORT "combined.mp4" "part1.mav" 0 100 "part2.mav" 50 150
    ```

*   **Mixed concatenation** — current file + another file:
    ```
    CALL 1-10 EXPORT "promo.mp4" 0 50 "intro.mav" 0 100
    ```

Export output defaults to `libx264` / `yuv420p` / `fast` preset. Output path is relative to the media folder.

> **Note:** Only one export can run at a time per producer. If an export is already in progress, the command returns an error. Wait for the current export to finish before starting another.

---

## 6. OSC Output

The producer publishes real-time state via OSC:

| Address | Value |
| :--- | :--- |
| `/channel/{ch}/stage/layer/{l}/file/time` | Current time (seconds) |
| `/channel/{ch}/stage/layer/{l}/file/frame` | Current frame number |
| `/channel/{ch}/stage/layer/{l}/file/fps` | Source file FPS |
| `/channel/{ch}/stage/layer/{l}/file/length` | Total frame count |
| `/channel/{ch}/stage/layer/{l}/vmx/timestamp` | Current timestamp (UTC microseconds) |
| `/channel/{ch}/stage/layer/{l}/vmx/timestamp_formatted` | Formatted timestamp (`yyyy-mm-dd-hh-mm-ss-ff`) |
| `/channel/{ch}/stage/layer/{l}/vmx/read_ms` | Last frame read time (ms) |
| `/channel/{ch}/stage/layer/{l}/vmx/decode_ms` | Last frame decode time (ms) |

---

## 7. Diagnostics

The module registers graphs in the CasparCG Diagnostics window:

*   **Producer graph:**
    *   `read-time` (orange) — disk I/O time per frame.
    *   `decode-time` (cyan) — VMX decode time per frame.
    *   `buffer` (green) — compressed frame size.
    *   Title bar shows filename, current frame / total frames, and timestamp.

*   **Consumer graph:**
    *   `frame-time` (green) — encode time per frame.
    *   `buffered-video` — ratio of compressed frame size to max buffer.
    *   Title bar shows filename, FPS, frame count, and elapsed time.

---

## 8. Best Practices

### Storage
*   Use fast local SSDs. At HQ 1080p60, sustained write throughput is ~33 MB/s per channel. At 4K HQ, this rises to ~100 MB/s.
*   For circular buffers, prefer short segments (30–60 s) to allow timely cleanup of locked segments.
*   Ensure the media folder has sufficient free space for the configured `max_duration` plus overhead.

### Quality Selection
*   Use `SQ` for day-to-day replay and time-shift. It provides a good balance of quality and bandwidth.
*   Use `OMT_SQ` when you need higher quality without jumping to full `HQ` bitrates — it delivers full DC precision at a moderate bitrate increase.
*   Reserve `HQ` / `OMT_HQ` for mastering or archive workflows where quality is paramount.
*   `LQ` is suitable for proxy recording or bandwidth-constrained environments.

### Live Replay Workflow
1.  Start a circular buffer recording on the source channel:
    ```
    ADD 1 REPLAY "live_buffer?max_duration=01:00&segment=60"
    ```
2.  On a separate output channel, play the buffer and seek to the moment of interest:
    ```
    PLAY 2-10 "live_buffer"
    CALL 2-10 SEEK 2026-05-11-14-30-00-00
    CALL 2-10 SPEED 0.5
    ```
3.  When done with slow-motion review, jump back to live:
    ```
    CALL 2-10 SPEED 1.0
    CALL 2-10 SEEK LIVE
    ```
4.  Export a highlight clip in the background without interrupting playback:
    ```
    CALL 2-10 EXPORT "highlight_goal.mp4" 2026-05-11-14-30-00-00 2026-05-11-14-30-15-00
    ```

### Performance Tips
*   The `SEEK LIVE` command backs off ~25 frames from the true live edge to prevent read/write contention stutter.
*   When fast-forwarding a growing file and catching up to the live edge, the producer automatically resets speed to `1.0`.
*   Export runs on a separate thread and does not affect channel playback frame rate.
